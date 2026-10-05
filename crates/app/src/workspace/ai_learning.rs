//! Training-related project logic the UI needs: which classes have data,
//! building class-label lists from the project, and saving/reading model
//! files through the backend.

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
