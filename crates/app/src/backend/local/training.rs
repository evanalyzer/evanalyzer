//! Bridges `evanalyzer_core`'s AI classifier training jobs (backend-agnostic,
//! no notion of a "project") to `ProjectSettings`: turns a project's labeled
//! objects/images into training data and starts the job.

use crate::api::{
    CancelHandle, PixelTrainingParams, RunningTraining, StartTrainingError, TrainedClassifier,
    TrainingItems,
};
use crate::backend::local::job::join_job;
use crate::workspace::extensions::image_entry_ext::ImageEntryExt;
use evanalyzer_cfg::core_types::{
    InternalErrors, ObjectClass, SegmentationClass, TrainingProgressEvent,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, PixelClassLabel,
};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_core::{ObjectTrainingJob, PixelTrainingJob, SavedClassifier, TrainingImage};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

/// A built training job for either classifier mode, so the rest of this
/// module doesn't need to match on `AiLearningClassifierSettings` itself.
enum TrainingJob {
    Pixel(PixelTrainingJob),
    Object(ObjectTrainingJob),
}

impl TrainingJob {
    fn items(&self) -> TrainingItems {
        match self {
            TrainingJob::Pixel(job) => TrainingItems::Images(job.images.len()),
            TrainingJob::Object(job) => TrainingItems::Objects(job.objects.len()),
        }
    }

    fn run_async(
        self,
    ) -> (
        JoinHandle<Result<SavedClassifier, InternalErrors>>,
        Receiver<TrainingProgressEvent>,
        Arc<AtomicBool>,
    ) {
        match self {
            TrainingJob::Pixel(job) => job.run_async(),
            TrainingJob::Object(job) => job.run_async(),
        }
    }
}

/// Gathers every already-labeled image/object in `project` and builds the
/// training job matching `settings.classifier`'s mode.
///
/// "Labeled" means the object carries at least one `object_class` - this
/// mirrors the AI training dialog's only labeling mechanism
/// (`assign_object_class`, which writes `object_class` regardless of
/// classifier mode), and `analyze`'s own convention of training/running over
/// every image already in the project rather than requiring an explicit
/// per-run selection.
///
/// For pixel classifiers, `class_labels` are `SegmentationClass`-typed (a
/// pixel classifier's output becomes a new `segmentation_class`, replacing
/// thresholding - see `PixelClassLabel`'s doc comment) while the project only
/// ever assigns `ObjectClass` labels (`Class::id`). Both are `u32`-backed IDs
/// over the same project class list (`ObjectClass::from_segmentation_class`
/// already does this numeric passthrough the other way), so labeled objects
/// are bridged into `SegmentationClass(id)` here rather than needing a
/// second, parallel pixel-labeling UI.
fn build_training_job(
    project: &ProjectSettings,
    settings: AiLearningSettings,
    pixel_params: PixelTrainingParams,
) -> Result<TrainingJob, InternalErrors> {
    match &settings.classifier {
        AiLearningClassifierSettings::Pixel { class_labels, .. } => {
            let images = gather_pixel_training_images(project, class_labels)?;
            Ok(TrainingJob::Pixel(PixelTrainingJob {
                settings,
                images,
                channel: pixel_params.channel,
                t_stack: pixel_params.t_stack,
                z_stack_handling: pixel_params.z_stack_handling,
            }))
        }
        AiLearningClassifierSettings::Object { .. } => {
            let objects = gather_labeled_objects(project);
            Ok(TrainingJob::Object(ObjectTrainingJob { settings, objects }))
        }
    }
}

/// Gathers the project's labeled training data for `settings`' classifier
/// mode (see [`build_training_job`]) and starts training on it. Fails up
/// front with [`StartTrainingError::NoTrainingData`] instead of starting a
/// run that can only fail later with a less helpful "cannot train on zero
/// samples".
pub(crate) fn start_training(
    project: &ProjectSettings,
    settings: AiLearningSettings,
    pixel_params: PixelTrainingParams,
) -> Result<RunningTraining, StartTrainingError> {
    let job = build_training_job(project, settings, pixel_params);
    let ret = match job {
        Ok(job) => {
            let items = job.items();
            if items.count() == 0 {
                return Err(StartTrainingError::NoTrainingData);
            }

            let (handle, events, cancel) = job.run_async();
            Ok(RunningTraining::from_parts(
                events,
                CancelHandle::new(cancel),
                items,
                Box::new(move || join_job(handle, "Training worker").map(TrainedClassifier)),
            ))
        }
        Err(error) => Err(StartTrainingError::Failed(error)),
    };
    ret
}

/// `project.images.list` is keyed by path *relative* to `project.images.root`
/// (see `ProjectExt::add_image_to_list`) - not directly openable. Every
/// labeled image found here is joined against `images.root` before being
/// handed to the job, which is the one thing that actually reads the file;
/// forgetting this join silently fails every `ImageReader::new` call (each
/// image gets marked `ImageFailed` and skipped), which is indistinguishable
/// from "no training data" downstream (`fit_*`'s "cannot train on zero
/// samples") - hence erroring out here instead, with a message that actually
/// names the problem.
fn gather_pixel_training_images(
    project: &ProjectSettings,
    class_labels: &[PixelClassLabel],
) -> Result<Vec<TrainingImage>, InternalErrors> {
    let mut images = Vec::new();
    for (rel_path, entry) in &project.images.list {
        let active = entry.active_series(&project.images.settings);
        let Some(series) = entry.series.get(&active) else {
            continue;
        };
        let labeled_objects: Vec<ObjectMetricSettings> = series
            .objects
            .iter()
            .filter(|object| !object.exclude_from_training)
            .filter_map(|object| {
                let label = resolve_pixel_label(class_labels, &object.object_class)?;
                let mut labeled = object.clone();
                labeled.segmentation_class = label;
                Some(labeled)
            })
            .collect();
        if labeled_objects.is_empty() {
            continue;
        }
        let Some(root) = project.images.root.as_ref() else {
            return Err(InternalErrors::InvalidArgument(
                "Project has labeled objects but no image folder is set - set the project's image folder before training a pixel classifier".into(),
            ));
        };
        images.push(TrainingImage {
            path: root.join(rel_path),
            series: active,
            labeled_objects,
        });
    }
    Ok(images)
}

fn gather_labeled_objects(project: &ProjectSettings) -> Vec<ObjectMetricSettings> {
    project
        .images
        .list
        .values()
        .filter_map(|entry| {
            entry
                .series
                .get(&entry.active_series(&project.images.settings))
        })
        .flat_map(|series| series.objects.iter())
        .filter(|object| !object.object_class.is_empty() && !object.exclude_from_training)
        .cloned()
        .collect()
}

/// Resolves an object's `object_class` set to the one `class_labels` entry it
/// unambiguously matches - `None` if it matches zero or more than one (see
/// `training::object::resolve_label` for the same rule applied on the object
/// classifier side, where the ambiguity is reported per-object instead of
/// silently dropped; there's no training-progress channel to report through
/// yet at this project-gathering stage).
fn resolve_pixel_label(
    class_labels: &[PixelClassLabel],
    object_class: &std::collections::HashSet<ObjectClass>,
) -> Option<SegmentationClass> {
    let mut matches = class_labels.iter().filter(|label| {
        object_class
            .iter()
            .any(|oc| oc.to_u32() == Some(label.class.as_u32()))
    });
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(first.class)
}

#[cfg(test)]
mod tests {
    use crate::backend::LocalFileSystem;
    use crate::workspace::ai_learning::{
        load_classifier_settings, pixel_class_labels_from_project, save_trained_model,
    };
    use evanalyzer_cfg::settings::ai_learning_settings::ObjectClassLabel;
    use std::path::PathBuf;

    use super::*;
    use evanalyzer_cfg::core_types::{ObjectId, SegmentationClass};
    use evanalyzer_cfg::settings::ai_learning_object_settings::{
        AiLearningObjectFeatureSettings, ObjectMetric,
    };
    use evanalyzer_cfg::settings::ai_learning_settings::AiLearningBackendSettings;
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use std::collections::HashSet;

    fn class_label(id: u32, name: &str) -> PixelClassLabel {
        PixelClassLabel {
            class: SegmentationClass(id),
            name: name.into(),
        }
    }

    fn labeled_object(id: u32, class: ObjectClass) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: ObjectId(id.into()),
            object_class: HashSet::from([class]),
            ..Default::default()
        }
    }

    fn two_class_object_settings() -> AiLearningSettings {
        AiLearningSettings {
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: Default::default(),
            backend: AiLearningBackendSettings::RandomForest(Default::default()),
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings {
                    metrics: vec![ObjectMetric::Area],
                },
                class_labels: vec![
                    ObjectClassLabel {
                        class: ObjectClass::Valid(1),
                        name: "A".into(),
                    },
                    ObjectClassLabel {
                        class: ObjectClass::Valid(2),
                        name: "B".into(),
                    },
                ],
            },
        }
    }

    /// Object training reads only already-computed metrics, so no image
    /// files are needed.
    fn project_with_labeled_objects(classes: &[u32]) -> ProjectSettings {
        let mut series = SeriesSettings::default();
        for (i, class) in classes.iter().enumerate() {
            series.objects.push(ObjectMetricSettings {
                area: 10 + 1000 * i,
                object_class: [ObjectClass::Valid(*class)].into(),
                ..Default::default()
            });
        }
        let mut entry = ImageEntry::default();
        entry.series.insert(0, series);
        let mut project = ProjectSettings::default();
        project.images.list.insert(PathBuf::from("img.tif"), entry);
        project
    }

    #[test]
    fn resolve_pixel_label_matches_the_one_configured_class() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Valid(1)]);
        assert_eq!(
            resolve_pixel_label(&labels, &object_class),
            Some(SegmentationClass(1))
        );
    }

    #[test]
    fn resolve_pixel_label_is_none_for_an_unconfigured_class() {
        let labels = vec![class_label(1, "Cell")];
        let object_class = HashSet::from([ObjectClass::Valid(99)]);
        assert_eq!(resolve_pixel_label(&labels, &object_class), None);
    }

    #[test]
    fn resolve_pixel_label_is_none_when_ambiguous() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Valid(1), ObjectClass::Valid(2)]);
        assert_eq!(resolve_pixel_label(&labels, &object_class), None);
    }

    #[test]
    fn gather_pixel_training_images_skips_images_with_no_labeled_objects() {
        let mut project = ProjectSettings::default();
        project.images.root = Some(PathBuf::from("/data/images"));
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        let path = PathBuf::from("a.tif");
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path.clone(),
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(
                    0,
                    SeriesSettings {
                        objects: vec![labeled_object(1, ObjectClass::Valid(1))],
                        ..Default::default()
                    },
                )]),
            },
        );
        let unlabeled_path = PathBuf::from("b.tif");
        project.images.list.insert(
            unlabeled_path.clone(),
            ImageEntry {
                rel_path: unlabeled_path,
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(0, SeriesSettings::default())]),
            },
        );

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let class_labels = pixel_class_labels_from_project(&project, &selected);
        let images = gather_pixel_training_images(&project, &class_labels).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].path, PathBuf::from("/data/images/a.tif"));
        assert_eq!(
            images[0].labeled_objects[0].segmentation_class,
            SegmentationClass(1)
        );
    }

    #[test]
    fn gather_pixel_training_images_errors_when_no_image_root_is_set_but_data_is_labeled() {
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        let path = PathBuf::from("a.tif");
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path,
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(
                    0,
                    SeriesSettings {
                        objects: vec![labeled_object(1, ObjectClass::Valid(1))],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let class_labels = pixel_class_labels_from_project(&project, &selected);
        let Err(err) = gather_pixel_training_images(&project, &class_labels) else {
            panic!("labeled data with no image root must error, not train on 0 samples");
        };
        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("no image folder is set"));
    }

    #[test]
    fn gather_labeled_objects_flattens_across_images_and_skips_unlabeled() {
        let mut project = ProjectSettings::default();
        let path = PathBuf::from("a.tif");
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path,
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(
                    0,
                    SeriesSettings {
                        objects: vec![
                            labeled_object(1, ObjectClass::Valid(1)),
                            ObjectMetricSettings {
                                id: ObjectId(2),
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                )]),
            },
        );

        let objects = gather_labeled_objects(&project);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id, ObjectId(1));
    }

    #[test]
    fn gather_labeled_objects_skips_a_labeled_object_marked_excluded() {
        let mut project = ProjectSettings::default();
        let path = PathBuf::from("a.tif");
        let mut excluded = labeled_object(1, ObjectClass::Valid(1));
        excluded.exclude_from_training = true;
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path,
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(
                    0,
                    SeriesSettings {
                        objects: vec![excluded, labeled_object(2, ObjectClass::Valid(1))],
                        ..Default::default()
                    },
                )]),
            },
        );

        let objects = gather_labeled_objects(&project);

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id, ObjectId(2));
    }

    #[test]
    fn gather_pixel_training_images_skips_an_excluded_object_even_if_its_class_is_selected() {
        let mut project = ProjectSettings::default();
        project.images.root = Some(PathBuf::from("/data/images"));
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        let path = PathBuf::from("a.tif");
        let mut excluded = labeled_object(1, ObjectClass::Valid(1));
        excluded.exclude_from_training = true;
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path,
                file_size: 0,
                selected_series: 0,
                series: std::collections::BTreeMap::from([(
                    0,
                    SeriesSettings {
                        objects: vec![excluded],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let class_labels = pixel_class_labels_from_project(&project, &selected);
        let images = gather_pixel_training_images(&project, &class_labels).unwrap();

        assert!(
            images.is_empty(),
            "the only labeled object is excluded, so the image has nothing to train from"
        );
    }

    #[test]
    fn start_training_without_labeled_objects_fails_up_front() {
        let result = start_training(
            &ProjectSettings::default(),
            two_class_object_settings(),
            PixelTrainingParams::default(),
        );
        assert!(matches!(result, Err(StartTrainingError::NoTrainingData)));
    }

    #[test]
    fn start_training_reports_its_items_and_trains_a_model() {
        let training = start_training(
            &project_with_labeled_objects(&[1, 2]),
            two_class_object_settings(),
            PixelTrainingParams::default(),
        )
        .unwrap();
        assert_eq!(training.items(), TrainingItems::Objects(2));

        let events: Vec<_> = training.events().iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::Finished { .. }))
        );
        training.wait().expect("two labeled objects should train");
    }

    #[test]
    fn training_cancelled_right_away_never_surfaces_a_different_error() {
        let training = start_training(
            &project_with_labeled_objects(&[1, 2]),
            two_class_object_settings(),
            PixelTrainingParams::default(),
        )
        .unwrap();
        training.cancel_handle().cancel();
        for _ in training.events() {}
        match training.wait() {
            Ok(_) | Err(InternalErrors::Cancelled) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn a_saved_model_reads_back_the_settings_it_was_trained_with() {
        let settings = two_class_object_settings();
        let training = start_training(
            &project_with_labeled_objects(&[1, 2]),
            settings.clone(),
            PixelTrainingParams::default(),
        )
        .unwrap();
        for _ in training.events() {}
        let classifier = training.wait().expect("two labeled objects should train");

        let dir = tempfile::tempdir().unwrap();
        let path = save_trained_model(
            &LocalFileSystem::default(),
            &classifier,
            dir.path(),
            "model",
        )
        .unwrap();

        // `AiLearningSettings` has no `PartialEq`; its serialized form is
        // what the model file stores anyway.
        assert_eq!(
            serde_json::to_value(
                load_classifier_settings(&LocalFileSystem::default(), &path).unwrap()
            )
            .unwrap(),
            serde_json::to_value(&settings).unwrap()
        );
    }
}
