//! # ai_learning
//!
//! Training-related project logic the UI needs: which classes have data,
//! building class-label lists from the project, and saving/reading model
//! files through the backend.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::TrainedClassifier;
use crate::workspace::extensions::image_entry_ext::ImageEntryExt;
use evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningSettings, ObjectClassLabel, PixelClassLabel,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use serde::Deserialize;
use std::path::{Path, PathBuf};

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
        .filter_map(|entry| {
            entry
                .series
                .get(&entry.active_series(&project.images.settings))
        })
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
        .filter(|class| {
            class.global() == ObjectClass::BACKGROUND || selected.contains(&class.global())
        })
        .map(|class| PixelClassLabel {
            class: SegmentationClass(class.id),
            name: class.name.clone(),
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
        .filter(|class| {
            class.global() == ObjectClass::BACKGROUND || selected.contains(&class.global())
        })
        .map(|class| ObjectClassLabel {
            class: class.global(),
            name: class.name.clone(),
        })
        .collect()
}

/// Where a trained model named `model_name` is saved for the project rooted
/// at `project_dir` - mirrors `generate_analyze_job_from_project_settings`'s
/// `<project_dir>/results/...` convention.
/// Whether a step of the project (pipelines and modules) applies the model
/// file `model` - an AI pixel or object classifier step whose model has the
/// same file name (steps may hold it relative or absolute).
pub fn model_is_used(project: &ProjectSettings, model: &Path) -> bool {
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    let Some(name) = model.file_name() else {
        return false;
    };
    crate::workspace::class_ids::all_commands(project).any(|command| {
        let path = match command {
            PipelineCommand::PixelClassifier(s) => &s.model_path,
            PipelineCommand::AiObjectClassifier(s) => &s.model_path,
            _ => return false,
        };
        path.file_name() == Some(name)
    })
}

/// A new pipeline applying the trained model at `model` - for the training
/// panel's "Add to pipeline". Pixel classifier: the model, then connected
/// components and extract objects, so the classes come out as objects (each
/// trained class mapped to the project class it was trained from - the
/// object then has that class's color). Object classifier: the model alone,
/// on the objects the pipelines before it found (no image of its own; it
/// goes last) - its input classes the classes `project`'s pipelines give
/// objects ([`classes_the_pipelines_give`]), so objects an earlier step
/// removed stay removed - adding the predicted class. Background is not mapped (stays
/// background / leaves the object as it is). `id`: a free pipeline id.
pub fn pipeline_for_model(
    project: &ProjectSettings,
    model: &Path,
    classifier: &evanalyzer_cfg::settings::ai_learning_settings::AiLearningClassifierSettings,
    id: evanalyzer_cfg::core_types::PipelineId,
) -> evanalyzer_cfg::settings::pipeline_settings::PipelineSettings {
    use evanalyzer_cfg::core_types::ImageAddress;
    use evanalyzer_cfg::settings::ai_learning_settings::AiLearningClassifierSettings;
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        AiObjectClassifierSettings, ClassificationMappingSettings, ExtractObjectsSettings,
        PixelClassifierSettings, SegmentationMappingSettings,
    };
    use evanalyzer_cfg::settings::pipeline_settings::{
        PipelineSettings, PipelineStepSettings, StepContent,
    };
    let step = |command| PipelineStepSettings {
        id: Default::default(),
        enabled: true,
        content: StepContent::Command(command),
    };
    let name = format!(
        "AI · {}",
        model
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default()
    );
    let (image_source, steps) = match classifier {
        AiLearningClassifierSettings::Pixel {
            feature_spec,
            class_labels,
            ..
        } => (
            ImageAddress::Channel(feature_spec.image_channels.first().copied().unwrap_or(0)),
            vec![
                step(PipelineCommand::PixelClassifier(PixelClassifierSettings {
                    model_path: model.to_path_buf(),
                    segmentation_mapping: class_labels
                        .iter()
                        .filter(|l| l.class != SegmentationClass::BACKGROUND)
                        .map(|l| SegmentationMappingSettings {
                            model_class: l.class,
                            segmentation_class: l.class,
                        })
                        .collect(),
                    ..Default::default()
                })),
                step(PipelineCommand::ConnectedComponents(Default::default())),
                step(PipelineCommand::ExtractObjects(
                    ExtractObjectsSettings::default(),
                )),
            ],
        ),
        AiLearningClassifierSettings::Object { class_labels, .. } => (
            ImageAddress::None,
            vec![step(PipelineCommand::AiObjectClassifier(
                AiObjectClassifierSettings {
                    model_path: model.to_path_buf(),
                    segmentation_mapping: class_labels
                        .iter()
                        .filter(|l| l.class != ObjectClass::BACKGROUND)
                        .map(|l| ClassificationMappingSettings {
                            object_class: l.class,
                            output_class: l.class,
                        })
                        .collect(),
                    input_classes: classes_the_pipelines_give(project),
                    ..Default::default()
                },
            ))],
        ),
    };
    PipelineSettings {
        id,
        name,
        description: Some("Added by the AI training panel.".into()),
        image_source,
        enabled: true,
        steps,
    }
}

/// The project classes `project`'s enabled pipelines give objects, in id
/// order and without Background - what an object classifier added after
/// them works on. A step that may give any class (annotations keeping their
/// classes, Extract Objects without a mapping) adds every class of the
/// project.
pub fn classes_the_pipelines_give(project: &ProjectSettings) -> Vec<ObjectClass> {
    use crate::workspace::module_insights::{assigned_classes, resolved_commands};
    let mut ids = std::collections::BTreeSet::new();
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        for step in &pipeline.steps {
            for command in resolved_commands(step, &project.modules) {
                let (classes, any) = assigned_classes(&command.command);
                if any {
                    ids.extend(project.classification.classes().iter().map(|c| c.id));
                }
                ids.extend(classes.into_iter().filter_map(|c| match c {
                    ObjectClass::Global(n) => Some(n),
                    _ => None,
                }));
            }
        }
    }
    ids.into_iter()
        .map(ObjectClass::Global)
        .filter(|c| *c != ObjectClass::BACKGROUND)
        .collect()
}

pub fn model_output_path(project_dir: &Path, model_name: &str) -> PathBuf {
    project_dir
        .join("models")
        .join(format!("{model_name}.{EVANALYZER_TRAINED_AI_MODELS}"))
}

/// Persists a trained classifier under `<project_dir>/models/<model_name>`,
/// creating the `models` directory if needed. Returns the path written to.
///
/// Written through `files` (the backend's), next to the project.
pub fn save_trained_model(
    files: &dyn crate::api::FileSystem,
    classifier: &TrainedClassifier,
    project_dir: &Path,
    model_name: &str,
) -> Result<PathBuf, InternalErrors> {
    let path = model_output_path(project_dir, model_name);
    files.write_file(&path, &classifier.to_bytes()?)?;
    Ok(path)
}

/// Reads only the settings (metadata, backend, classes, features) a saved
/// model file was trained with - all front ends need for info dialogs,
/// class-mapping rows and "retrain from existing model", without exposing
/// the fitted model itself.
///
/// Read through `files` (the backend's), so in remote mode the model file on
/// the server is read. Only the `settings` part is parsed - the fitted model
/// can be large and isn't needed here.
pub fn load_classifier_settings(
    files: &dyn crate::api::FileSystem,
    path: &Path,
) -> Result<AiLearningSettings, InternalErrors> {
    #[derive(Deserialize)]
    struct SettingsOnly {
        settings: AiLearningSettings,
    }
    let bytes = files.read_file(path)?;
    serde_json::from_slice::<SettingsOnly>(&bytes)
        .map(|model| model.settings)
        .map_err(|e| {
            InternalErrors::Internal(format!(
                "'{}' is not a valid classifier model: {e}",
                path.display()
            ))
        })
}

#[cfg(test)]
mod tests {

    /// "Add to pipeline": a pixel model becomes classifier → connected
    /// components → extract objects on its first channel, its classes
    /// (not background) mapped to themselves; an object model a classifier
    /// step without an image of its own.
    #[test]
    fn a_model_becomes_a_pipeline_applying_it() {
        use evanalyzer_cfg::core_types::{ImageAddress, PipelineId};
        use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
        use evanalyzer_cfg::settings::ai_learning_settings::{
            AiLearningClassifierSettings, ObjectClassLabel, PixelClassLabel,
        };
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        let model = Path::new("/data/study/models/nuclei.evamodel");
        let pixel = AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings {
                channels: vec![vec![]],
                image_channels: vec![2, 3],
                image_channel_names: vec![],
            },
            input_color: Default::default(),
            class_labels: vec![
                PixelClassLabel {
                    class: SegmentationClass::BACKGROUND,
                    name: "Background".into(),
                },
                PixelClassLabel {
                    class: SegmentationClass(4),
                    name: "Nucleus".into(),
                },
            ],
        };
        let p = pipeline_for_model(&ProjectSettings::default(), model, &pixel, PipelineId(7));
        assert_eq!(p.id, PipelineId(7));
        assert_eq!(p.name, "AI · nuclei");
        assert!(matches!(p.image_source, ImageAddress::Channel(2)));
        let commands: Vec<&PipelineCommand> = p.steps.iter().filter_map(|s| s.command()).collect();
        assert_eq!(commands.len(), 3);
        let PipelineCommand::PixelClassifier(c) = commands[0] else {
            panic!("classifier first")
        };
        assert_eq!(c.model_path, model);
        assert_eq!(c.segmentation_mapping.len(), 1, "background not mapped");
        assert_eq!(c.segmentation_mapping[0].model_class, SegmentationClass(4));
        assert_eq!(
            c.segmentation_mapping[0].segmentation_class,
            SegmentationClass(4)
        );
        assert!(matches!(
            commands[1],
            PipelineCommand::ConnectedComponents(_)
        ));
        assert!(matches!(commands[2], PipelineCommand::ExtractObjects(_)));

        let object = AiLearningClassifierSettings::Object {
            feature_spec:
                evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings {
                    metrics: vec![],
                    ..Default::default()
                },
            class_labels: vec![
                ObjectClassLabel {
                    class: ObjectClass::BACKGROUND,
                    name: "Background".into(),
                },
                ObjectClassLabel {
                    class: ObjectClass::Global(5),
                    name: "Apoptotic".into(),
                },
            ],
        };
        let p = pipeline_for_model(&ProjectSettings::default(), model, &object, PipelineId(8));
        assert!(matches!(p.image_source, ImageAddress::None));
        let [step] = p.steps.as_slice() else {
            panic!("one step")
        };
        let Some(PipelineCommand::AiObjectClassifier(c)) = step.command() else {
            panic!("object classifier")
        };
        assert_eq!(c.segmentation_mapping.len(), 1);
        assert_eq!(
            c.segmentation_mapping[0].output_class,
            ObjectClass::Global(5)
        );
    }

    /// The object classifier "Add to pipeline" adds works on the classes the
    /// enabled pipelines give objects: a mapped Extract Objects' class and a
    /// classify step's output class - not a disabled pipeline's. A step that
    /// may give any class (Extract Objects without a mapping) brings in
    /// every class of the project.
    #[test]
    fn the_added_object_classifier_works_on_the_classes_the_pipelines_give() {
        use evanalyzer_cfg::core_types::PipelineId;
        use evanalyzer_cfg::settings::ai_learning_settings::{
            AiLearningClassifierSettings, ObjectClassLabel,
        };
        use evanalyzer_cfg::settings::classification_settings::Class;
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ClassifyObjectsSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
        };
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };
        let step = |command| PipelineStepSettings {
            id: Default::default(),
            enabled: true,
            content: StepContent::Command(command),
        };
        let pipeline = |id, enabled, steps| PipelineSettings {
            id: PipelineId(id),
            name: format!("p{id}"),
            description: None,
            image_source: Default::default(),
            enabled,
            steps,
        };
        let extract = |class_mapping| {
            PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                class_mapping,
                ..Default::default()
            })
        };
        let mapped = |n| {
            vec![ExtractClassMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class: ObjectClass::Global(n),
            }]
        };

        let mut project = ProjectSettings::default();
        for id in [3, 7, 8, 9] {
            project.classification.classes_mut().push(Class {
                id,
                ..Default::default()
            });
        }
        project.pipelines.push(pipeline(
            1,
            true,
            vec![
                step(extract(mapped(7))),
                step(PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                    input_classes: vec![ObjectClass::Global(7)],
                    output_class: ObjectClass::Global(3),
                    ..Default::default()
                })),
            ],
        ));
        project
            .pipelines
            .push(pipeline(2, false, vec![step(extract(mapped(8)))]));
        assert_eq!(
            classes_the_pipelines_give(&project),
            [ObjectClass::Global(3), ObjectClass::Global(7)]
        );

        let object = AiLearningClassifierSettings::Object {
            feature_spec: Default::default(),
            class_labels: vec![ObjectClassLabel {
                class: ObjectClass::Global(5),
                name: "Apoptotic".into(),
            }],
        };
        let p = pipeline_for_model(
            &project,
            Path::new("/m/apoptotic.evamodel"),
            &object,
            PipelineId(9),
        );
        let Some(PipelineCommand::AiObjectClassifier(c)) = p.steps[0].command() else {
            panic!("object classifier")
        };
        assert_eq!(
            c.input_classes,
            [ObjectClass::Global(3), ObjectClass::Global(7)]
        );

        project
            .pipelines
            .push(pipeline(3, true, vec![step(extract(vec![]))]));
        assert_eq!(
            classes_the_pipelines_give(&project),
            [3, 7, 8, 9].map(ObjectClass::Global),
            "without a mapping: any class of the project"
        );
    }

    /// A step applies a model when its model file has the same name, also
    /// held relative or under another folder; inside a module too.
    #[test]
    fn a_model_is_used_by_a_classifier_step_with_its_file() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };
        let model = Path::new("/data/study/models/nuclei.evamodel");
        let mut project = ProjectSettings::default();
        assert!(!model_is_used(&project, model));
        project.pipelines.push(PipelineSettings {
            steps: vec![PipelineStepSettings {
                id: Default::default(),
                enabled: true,
                content: StepContent::Command(PipelineCommand::PixelClassifier(
                    PixelClassifierSettings {
                        model_path: PathBuf::from("models/nuclei.evamodel"),
                        ..Default::default()
                    },
                )),
            }],
            id: evanalyzer_cfg::core_types::PipelineId(1),
            name: "Nuclei".into(),
            description: None,
            image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
            enabled: true,
        });
        assert!(model_is_used(&project, model));
        assert!(!model_is_used(
            &project,
            Path::new("/data/study/models/spots.evamodel")
        ));
    }

    use crate::backend::LocalFileSystem;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

    use super::*;
    use evanalyzer_cfg::core_types::{ObjectId, SegmentationClass};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use std::collections::HashSet;

    fn labeled_object(id: u32, class: ObjectClass) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: ObjectId(id.into()),
            object_class: HashSet::from([class]),
            ..Default::default()
        }
    }

    #[test]
    fn used_object_classes_ignores_a_class_whose_only_object_is_excluded() {
        let mut project = ProjectSettings::default();
        let path = PathBuf::from("a.tif");
        let mut excluded = labeled_object(1, ObjectClass::Global(1));
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
    fn pixel_class_labels_from_project_only_includes_the_selected_classes() {
        let mut project = ProjectSettings::default();
        project.classification.classes_mut().push(Class {
            id: 1,
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: 2,
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
                            labeled_object(1, ObjectClass::Global(1)),
                            labeled_object(2, ObjectClass::Global(2)),
                        ],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Global(1)]);
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
            id: 1,
            name: "Not Yet Painted".into(),
            ..Default::default()
        });

        let selected = HashSet::from([ObjectClass::Global(1)]);
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
            id: 4,
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: 8,
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Background deliberately absent from `selected`.
        let selected = HashSet::from([ObjectClass::Global(4), ObjectClass::Global(8)]);
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
            id: 1,
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: 2,
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
                            labeled_object(1, ObjectClass::Global(1)),
                            labeled_object(2, ObjectClass::Global(2)),
                        ],
                        ..Default::default()
                    },
                )]),
            },
        );

        let selected = HashSet::from([ObjectClass::Global(1)]);
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
            id: 4,
            name: "Cell".into(),
            ..Default::default()
        });
        project.classification.classes_mut().push(Class {
            id: 8,
            name: "Nucleus".into(),
            ..Default::default()
        });

        // Background deliberately absent from `selected`.
        let selected = HashSet::from([ObjectClass::Global(4), ObjectClass::Global(8)]);
        let labels = object_class_labels_from_project(&project, &selected);

        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].class, ObjectClass::BACKGROUND);
        assert_eq!(labels[0].name, "Background");
        assert_eq!(labels[1].class, ObjectClass::Global(4));
        assert_eq!(labels[2].class, ObjectClass::Global(8));
    }

    #[test]
    fn model_output_path_uses_the_models_subfolder_and_evamodel_extension() {
        let path = model_output_path(Path::new("/proj"), "my-model");
        assert_eq!(path, PathBuf::from("/proj/models/my-model.evamodel"));
    }

    // -- start_training ---------------------------------------------------

    #[test]
    fn load_classifier_settings_reports_a_missing_file_as_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            load_classifier_settings(
                &LocalFileSystem::default(),
                &dir.path().join("missing.model")
            )
            .is_err()
        );
    }
}
