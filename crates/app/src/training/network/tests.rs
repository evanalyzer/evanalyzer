//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::files::LocalFileSystem;
use crate::training::TrainedArtifact;
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::TrainingProgressEvent;
use evanalyzer_cfg::settings::deep_learning_settings::{
    DeepLearningSettings, NetworkTrainingSettings, ValidationSettings,
};
use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings, ZStackHandling};
use std::path::{Path, PathBuf};

const SPOT: ObjectClass = ObjectClass::Global(1);
const AREA: ObjectClass = ObjectClass::Global(2);
const OTHER: ObjectClass = ObjectClass::Global(9);

fn annotation(class: ObjectClass, [x0, y0, x1, y1]: [u32; 4]) -> ObjectMetricSettings {
    let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize;
    ObjectMetricSettings {
        object_class: [class].into(),
        bbox: [x0, y0, x1, y1],
        mask_data: bitvec![u64, Lsb0; 1; n],
        area: n,
        ..Default::default()
    }
}

fn spot(x: u32, y: u32) -> ObjectMetricSettings {
    annotation(SPOT, [x, y, x, y])
}

/// A 16-bit grey 48 x 40 TIFF with bright blobs at `spots`.
fn write_spot_tiff(path: &Path, spots: &[(u32, u32)]) {
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(48, 40, |x, y| {
        let mut v = 200.0f32;
        for &(sx, sy) in spots {
            let (dx, dy) = (x as f32 - sx as f32, y as f32 - sy as f32);
            v += 3000.0 * (-(dx * dx + dy * dy) / 8.0).exp();
        }
        image::Luma([v as u16])
    });
    img.save(path).unwrap();
}

/// A project in `dir` whose images (name → annotations) are TIFFs there.
fn project(dir: &Path, images: Vec<(&str, Vec<ObjectMetricSettings>)>) -> ProjectSettings {
    let mut project = ProjectSettings::default();
    project.images.root = Some(dir.to_path_buf());
    for (name, objects) in images {
        write_spot_tiff(&dir.join(name), &[(10, 10), (30, 20)]);
        let mut entry = ImageEntry::default();
        entry.rel_path = PathBuf::from(name);
        entry.series.insert(
            0,
            SeriesSettings {
                objects,
                ..Default::default()
            },
        );
        project.images.list.insert(PathBuf::from(name), entry);
    }
    project
}

fn settings(epochs: usize) -> DeepLearningSettings {
    let mut settings = DeepLearningSettings::default();
    settings.classes.objects = SPOT;
    settings.classes.area = AREA;
    settings.training = NetworkTrainingSettings {
        epochs,
        patches_per_epoch: 1,
        batch_size: 1,
        patch_size: 32,
        validation: ValidationSettings::Off,
        patience: None,
        ..NetworkTrainingSettings::default()
    };
    settings
}

fn request(project: ProjectSettings, epochs: usize) -> NetworkTrainingRequest {
    NetworkTrainingRequest {
        project,
        settings: settings(epochs),
        z_stack_handling: ZStackHandling::SingleStack,
        save_to: None,
    }
}

#[test]
fn only_images_with_annotations_of_the_trainings_classes_are_gathered() {
    let dir = tempfile::tempdir().unwrap();
    let mut excluded = spot(5, 5);
    excluded.exclude_from_training = true;
    let project = project(
        dir.path(),
        vec![
            (
                "a.tif",
                vec![
                    annotation(AREA, [0, 0, 20, 20]),
                    spot(10, 10),
                    annotation(OTHER, [1, 1, 2, 2]),
                ],
            ),
            ("b.tif", vec![annotation(OTHER, [1, 1, 2, 2]), excluded]),
        ],
    );
    let images = gather_images(&project, &settings(1).classes).unwrap();
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].path, dir.path().join("a.tif"));
    assert_eq!(
        images[0].labeled_objects.len(),
        2,
        "the other class left out"
    );
}

#[test]
fn annotations_without_an_image_folder_are_an_error() {
    let mut project = ProjectSettings::default();
    let mut entry = ImageEntry::default();
    entry.series.insert(
        0,
        SeriesSettings {
            objects: vec![spot(1, 1)],
            ..Default::default()
        },
    );
    project.images.list.insert(PathBuf::from("a.tif"), entry);
    let error = gather_images(&project, &settings(1).classes)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no image folder"), "{error}");
}

#[test]
fn a_project_without_annotations_of_the_classes_has_no_training_data() {
    let dir = tempfile::tempdir().unwrap();
    let project = project(
        dir.path(),
        vec![("a.tif", vec![annotation(OTHER, [1, 1, 2, 2])])],
    );
    assert!(matches!(
        start_network_training(request(project, 1)),
        Err(StartTrainingError::NoTrainingData)
    ));
}

#[test]
fn a_network_is_trained_on_the_project_and_saved_in_its_models_folder() {
    let dir = tempfile::tempdir().unwrap();
    let project = project(
        dir.path(),
        vec![(
            "a.tif",
            vec![annotation(AREA, [0, 0, 47, 39]), spot(10, 10), spot(30, 20)],
        )],
    );
    let training = start_network_training(request(project, 2)).unwrap();
    assert_eq!(training.items(), TrainingItems::Images(1));
    let events: Vec<_> = training.events().iter().collect();
    assert!(matches!(
        events.last(),
        Some(TrainingProgressEvent::Finished { .. })
    ));
    let model = training.wait().unwrap();

    let path = model
        .save(&LocalFileSystem::default(), dir.path(), "spots")
        .unwrap();
    assert_eq!(path, dir.path().join("models").join("spots.evanet"));
    // The AI step's model card knows it as a SpotNet file.
    use crate::ai::model_file::{ModelFileNote, ModelKind, model_file_note};
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        matches!(
            model_file_note(ModelKind::SpotNet, &path, Ok(&bytes)),
            ModelFileNote::Fits(_)
        ),
        "not a SpotNet file"
    );
    // The bytes as a server would send them come back as the same file.
    let again = TrainedNetworkModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(again.0, model.0);
}

#[test]
fn wrong_settings_are_refused_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    let project = project(
        dir.path(),
        vec![(
            "a.tif",
            vec![spot(10, 10), annotation(AREA, [0, 0, 20, 20])],
        )],
    );
    let mut req = request(project, 1);
    req.settings.classes.area = SPOT;
    let Err(StartTrainingError::Failed(error)) = start_network_training(req) else {
        panic!("not refused")
    };
    assert!(error.to_string().contains("different classes"), "{error}");
}
