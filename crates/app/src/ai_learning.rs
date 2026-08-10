//! Bridges `evanalyzer_core`'s AI classifier training jobs (backend-agnostic,
//! no notion of a "project") to `ProjectSettings` (this crate's domain) -
//! shared by the GUI's `ai_learning_controller` and the CLI's
//! `train-classifier` command so neither duplicates "how do I turn a
//! project's labeled objects into training data," mirroring how
//! `evanalyzer_core::generate_analyze_job_from_project_settings` is the
//! single bridge pipeline execution goes through.

use evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel, PixelClassLabel,
};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_core::{
    ObjectTrainingJob, PixelTrainingJob, SavedClassifier, TrainingImage, TrainingProgressEvent,
    save_classifier_to_file,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Extra parameters `PixelTrainingJob` needs that aren't part of the portable
/// `AiLearningSettings` model descriptor - object training needs none of
/// these, since it reads no images (see `ObjectTrainingJob`'s doc comment).
#[derive(Debug, Clone)]
pub struct PixelTrainingParams {
    pub channel: i32,
    pub t_stack: i32,
    pub z_stack_handling: ZStackHandling,
}

impl Default for PixelTrainingParams {
    fn default() -> Self {
        Self {
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandling::SingleStack,
        }
    }
}

/// A runnable training job for either classifier mode, so callers don't need
/// to match on `AiLearningClassifierSettings` themselves to call `run`/`run_async`.
pub enum TrainingJob {
    Pixel(PixelTrainingJob),
    Object(ObjectTrainingJob),
}

impl TrainingJob {
    pub fn run(
        &self,
        progress: Sender<TrainingProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<SavedClassifier, InternalErrors> {
        match self {
            TrainingJob::Pixel(job) => job.run(progress, cancel),
            TrainingJob::Object(job) => job.run(progress, cancel),
        }
    }

    pub fn run_async(
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
pub fn build_training_job(
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
        let Some(series) = entry.series.get(&entry.selected_series) else {
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
            series: entry.selected_series,
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
        .filter_map(|entry| entry.series.get(&entry.selected_series))
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

/// Classes that appear in at least one non-excluded project object's
/// `object_class` set, across every image (not just labeled/relevant ones -
/// a broader "does any data exist for this class at all" scan than
/// `gather_pixel_training_images` or `gather_labeled_objects` do, since it
/// just needs the set of ids, not the objects themselves). Objects with
/// `exclude_from_training` set don't count, so a class whose only labeled
/// objects have all been excluded doesn't get pre-checked as if it had
/// usable data. Also used by the GUI to pre-check classes with existing data
/// in the training dialog's class checklist.
pub fn used_object_classes(project: &ProjectSettings) -> std::collections::HashSet<ObjectClass> {
    project
        .images
        .list
        .values()
        .filter_map(|entry| entry.series.get(&entry.selected_series))
        .flat_map(|series| series.objects.iter())
        .filter(|object| !object.exclude_from_training)
        .flat_map(|object| object.object_class.iter().copied())
        .collect()
}

/// Builds `AiLearningClassifierSettings::Pixel::class_labels` from a
/// project's classification classes, restricted to `selected` - the training
/// dialog's own class checklist, shared with the object classifier side (see
/// `object_class_labels_from_project`). Not restricted to classes that
/// already have data: a class can be selected ahead of painting/labeling it,
/// same as on the object side - it's the dialog's default pre-check
/// (`used_object_classes`), not this function, that steers users toward
/// classes with existing data.
///
/// `ObjectClass::BACKGROUND` is always included regardless of `selected`,
/// so `class_labels[0]` (the dense training-label index every `fit_*`
/// backend resolves against, see `training/pixel.rs`/`training/object.rs`)
/// is always Background - `classes()` already pins Background first in
/// project order (`ClassificationSettings::new`/`move_up`/`move_down`
/// enforce this), so forcing its inclusion here can't reorder anything
/// else, only guarantee it's never silently dropped by an unchecked
/// checklist entry.
pub fn pixel_class_labels_from_project(
    project: &ProjectSettings,
    selected: &std::collections::HashSet<ObjectClass>,
) -> Vec<PixelClassLabel> {
    project
        .classification
        .classes()
        .iter()
        .filter(|class| class.id == ObjectClass::BACKGROUND || selected.contains(&class.id))
        .filter_map(|class| {
            class.id.to_u32().map(|id| PixelClassLabel {
                class: SegmentationClass(id),
                name: class.name.clone(),
            })
        })
        .collect()
}

/// Builds `AiLearningClassifierSettings::Object::class_labels` restricted to
/// `selected` - the object training dialog's own class checklist. Not
/// inferred from which classes happen to have data: object classifiers are
/// commonly trained to distinguish a deliberately curated subset of classes,
/// so the choice is left to the user rather than auto-including everything
/// with at least one example.
///
/// `ObjectClass::BACKGROUND` is always included regardless of `selected` -
/// see `pixel_class_labels_from_project`'s doc comment for why this keeps
/// `class_labels[0]` pinned to Background.
pub fn object_class_labels_from_project(
    project: &ProjectSettings,
    selected: &std::collections::HashSet<ObjectClass>,
) -> Vec<ObjectClassLabel> {
    project
        .classification
        .classes()
        .iter()
        .filter(|class| class.id == ObjectClass::BACKGROUND || selected.contains(&class.id))
        .map(|class| ObjectClassLabel {
            class: class.id,
            name: class.name.clone(),
        })
        .collect()
}

/// Where a trained model named `model_name` is saved for the project rooted
/// at `project_dir` - mirrors `generate_analyze_job_from_project_settings`'s
/// `<project_dir>/results/...` convention.
pub fn model_output_path(project_dir: &Path, model_name: &str) -> PathBuf {
    project_dir
        .join("models")
        .join(format!("{model_name}.{EVANALYZER_TRAINED_AI_MODELS}"))
}

/// Persists a trained classifier under `<project_dir>/models/<model_name>`,
/// creating the `models` directory if needed. Returns the path written to.
pub fn save_trained_model(
    classifier: &SavedClassifier,
    project_dir: &Path,
    model_name: &str,
) -> Result<PathBuf, InternalErrors> {
    let path = model_output_path(project_dir, model_name);
    save_classifier_to_file(classifier, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{ObjectId, SegmentationClass};
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
    fn used_object_classes_ignores_a_class_whose_only_object_is_excluded() {
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
                        objects: vec![excluded],
                        ..Default::default()
                    },
                )]),
            },
        );

        assert!(used_object_classes(&project).is_empty());
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
    fn pixel_class_labels_from_project_only_includes_the_selected_classes() {
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(2),
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Both classes have data, but only Valid(1) is selected - same
        // selection-is-a-user-choice behavior as the object classifier side.
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
                            labeled_object(2, ObjectClass::Valid(2)),
                        ],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let labels = pixel_class_labels_from_project(&project, &selected);

        // Background is always included ahead of the selection (see
        // `pixel_class_labels_from_project_always_includes_background`), so
        // it lands at index 0 regardless of `selected`.
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].name, "Cell");
    }

    #[test]
    fn pixel_class_labels_from_project_allows_selecting_a_class_with_no_labeled_pixels_yet() {
        // Unlike the old (usage-restricted) behavior, a class can be
        // selected ahead of painting any pixels for it - mirrors
        // `object_class_labels_from_project`, which never restricted this.
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Not Yet Painted".into(),
            ..Default::default()
        });

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let labels = pixel_class_labels_from_project(&project, &selected);

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].name, "Not Yet Painted");
    }

    #[test]
    fn pixel_class_labels_from_project_always_includes_background_at_index_zero() {
        // Background defaults to *unchecked* in the training dialog's class
        // checklist (`used_object_classes` only pre-checks classes with
        // existing labeled data, and users rarely explicitly paint
        // "Background"), so `selected` alone must never be able to drop it -
        // every `fit_*` backend resolves training labels as a dense index
        // into this array (see `training/pixel.rs`), so index 0 needs to be
        // a stable, always-present anchor rather than whatever the first
        // *selected* class happens to be.
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(4),
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(8),
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Background deliberately absent from `selected`.
        let selected = HashSet::from([ObjectClass::Valid(4), ObjectClass::Valid(8)]);
        let labels = pixel_class_labels_from_project(&project, &selected);

        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].class, SegmentationClass(0));
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].class, SegmentationClass(4));
        assert_eq!(labels[2].class, SegmentationClass(8));
    }

    #[test]
    fn object_class_labels_from_project_only_includes_the_selected_classes() {
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(2),
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Both classes have data, but only Valid(1) is selected - the
        // selection is a user choice, not inferred from usage.
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
                            labeled_object(2, ObjectClass::Valid(2)),
                        ],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Valid(1)]);
        let labels = object_class_labels_from_project(&project, &selected);

        // Background is always included ahead of the selection (see
        // `object_class_labels_from_project_always_includes_background`), so
        // it lands at index 0 regardless of `selected`.
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].name, "Cell");
    }

    #[test]
    fn object_class_labels_from_project_always_includes_background_at_index_zero() {
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(4),
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: ObjectClass::Valid(8),
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Background deliberately absent from `selected`.
        let selected = HashSet::from([ObjectClass::Valid(4), ObjectClass::Valid(8)]);
        let labels = object_class_labels_from_project(&project, &selected);

        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].class, ObjectClass::BACKGROUND);
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].class, ObjectClass::Valid(4));
        assert_eq!(labels[2].class, ObjectClass::Valid(8));
    }

    #[test]
    fn model_output_path_uses_the_models_subfolder_and_evamodel_extension() {
        let path = model_output_path(Path::new("/proj"), "my-model");
        assert_eq!(path, PathBuf::from("/proj/models/my-model.evamodel"));
    }
}
