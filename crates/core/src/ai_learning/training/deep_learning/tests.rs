//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use bitvec::prelude::*;
use evanalyzer_nn::augment::AugmentParams;
use evanalyzer_nn::model::spotnet::{PercentileNormalization, SpotNetConfig};
use evanalyzer_nn::train::{NetworkTrainParams, Validation};

const SPOT: ObjectClass = ObjectClass::Global(1);
const AREA: ObjectClass = ObjectClass::Global(2);
const VALIDATION: ObjectClass = ObjectClass::Global(3);

/// A 16-bit grey TIFF of `width x height`: background 200, a bright blob
/// (sigma 2 px) at every spot.
fn write_spot_tiff(path: &Path, width: u32, height: u32, spots: &[(u32, u32)]) {
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(width, height, |x, y| {
        let mut v = 200.0f32;
        for &(sx, sy) in spots {
            let (dx, dy) = (x as f32 - sx as f32, y as f32 - sy as f32);
            v += 3000.0 * (-(dx * dx + dy * dy) / 8.0).exp();
        }
        image::Luma([v as u16])
    });
    img.save(path).unwrap();
}

/// An annotation of `class`: the rectangle `[x0, y0, x1, y1]` (inclusive)
/// filled - one pixel when the corners are equal.
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

fn params(epochs: usize) -> SpotNetTrainParams {
    SpotNetTrainParams {
        common: NetworkTrainParams {
            epochs,
            patches_per_epoch: 2,
            batch_size: 2,
            patch_size: 32,
            learning_rate: 1e-2,
            augment: AugmentParams::none(),
            validation: Validation::Off,
            patience: None,
        },
        network: SpotNetConfig {
            in_channels: 1,
            channels: 4,
            ..SpotNetConfig::default()
        },
        sigma: 1.5,
        normalization: PercentileNormalization::default(),
    }
}

fn job(images: Vec<TrainingImage>, epochs: usize) -> DeepLearningTrainingJob {
    DeepLearningTrainingJob {
        model: DeepLearningModel::SpotNet(params(epochs)),
        start: None,
        images,
        classes: DeepLearningClasses {
            objects: SPOT,
            area: AREA,
            validation: Some(VALIDATION),
        },
        channels: vec![0],
        z_stack_handling: ZStackHandling::SingleStack,
        seed: 1,
    }
}

/// One 64 x 48 image with three spots and the given annotations.
fn one_image(dir: &tempfile::TempDir, annotations: Vec<ObjectMetricSettings>) -> TrainingImage {
    let path = dir.path().join("spots.tif");
    write_spot_tiff(&path, 64, 48, &[(10, 10), (20, 12), (50, 30)]);
    TrainingImage {
        path,
        series: 0,
        labeled_objects: annotations,
    }
}

#[test]
fn a_region_is_its_area_plus_the_halo_within_the_image() {
    let dir = tempfile::tempdir().unwrap();
    let image = one_image(&dir, vec![annotation(AREA, [5, 5, 24, 16]), spot(10, 10)]);
    let regions = job(vec![image], 1).regions(17).unwrap();
    assert_eq!(regions.len(), 1);
    let tile = regions[0].tile;
    // 5 - 17 clamped to 0; 24 + 17 = 41; 16 + 17 = 33.
    assert_eq!(
        (tile.offset_x, tile.offset_y, tile.width, tile.height),
        (0, 0, 42, 34)
    );
    assert_eq!(regions[0].annotations.len(), 2);
}

#[test]
fn overlapping_areas_are_one_region_far_ones_two() {
    let dir = tempfile::tempdir().unwrap();
    let near = vec![
        annotation(AREA, [5, 5, 10, 10]),
        annotation(AREA, [12, 5, 18, 10]),
    ];
    let regions = job(vec![one_image(&dir, near)], 1).regions(2).unwrap();
    assert_eq!(regions.len(), 1);

    let far = vec![
        annotation(AREA, [2, 2, 6, 6]),
        annotation(AREA, [50, 30, 60, 40]),
    ];
    let regions = job(vec![one_image(&dir, far)], 1).regions(2).unwrap();
    assert_eq!(regions.len(), 2);
}

#[test]
fn an_image_without_areas_gives_no_region_and_spots_elsewhere_stay_out() {
    let dir = tempfile::tempdir().unwrap();
    let regions = job(vec![one_image(&dir, vec![spot(10, 10)])], 1)
        .regions(17)
        .unwrap();
    assert!(regions.is_empty());

    let image = one_image(&dir, vec![annotation(AREA, [2, 2, 6, 6]), spot(50, 30)]);
    let regions = job(vec![image], 1).regions(2).unwrap();
    assert_eq!(
        regions[0].annotations.len(),
        1,
        "the far spot is not in the region"
    );
}

#[test]
fn a_regions_labels_are_its_spots_areas_and_validation_areas() {
    let dir = tempfile::tempdir().unwrap();
    let image = one_image(
        &dir,
        vec![
            annotation(AREA, [4, 4, 11, 11]),
            annotation(VALIDATION, [12, 4, 15, 11]),
            spot(6, 7),
            // Not one pixel: no spot.
            annotation(SPOT, [8, 8, 9, 9]),
        ],
    );
    let job = job(vec![image], 1);
    let region = &job.regions(2).unwrap()[0];
    let labels = spot_labels(region, &job.classes);
    let (x0, y0) = (region.tile.offset_x, region.tile.offset_y);
    assert_eq!((x0, y0), (2, 2));
    assert_eq!(
        labels.spots,
        vec![Spot {
            x: (6 - x0) as f32 + 0.5,
            y: (7 - y0) as f32 + 0.5
        }]
    );
    let at = |mask: &[bool], x: usize, y: usize| mask[(y - y0) * labels.width + (x - x0)];
    assert!(at(&labels.area, 4, 4) && at(&labels.area, 11, 11) && !at(&labels.area, 12, 4));
    assert!(at(&labels.validation, 12, 4) && !at(&labels.validation, 11, 4));
}

#[test]
fn a_region_is_read_in_the_jobs_channels() {
    let dir = tempfile::tempdir().unwrap();
    let image = one_image(&dir, vec![annotation(AREA, [8, 8, 12, 12])]);
    let job = job(vec![image], 1);
    let region = &job.regions(0).unwrap()[0];
    let read = read_region(&job, region).unwrap();
    assert_eq!((read.width, read.height, read.channels), (5, 5, 1));
    // The blob at (10, 10) is the region's brightest pixel, in its middle.
    let brightest = (0..25)
        .max_by(|&a, &b| read.data[a].total_cmp(&read.data[b]))
        .unwrap();
    assert_eq!(brightest, 2 * 5 + 2);

    let mut other = job.clone();
    other.channels = vec![3];
    let error = read_region(&other, region).unwrap_err().to_string();
    assert!(error.contains("channel 3"), "{error}");
}

#[test]
fn a_job_trains_a_spotnet_and_saves_it() {
    let dir = tempfile::tempdir().unwrap();
    let image = one_image(
        &dir,
        vec![
            annotation(AREA, [0, 0, 63, 47]),
            spot(10, 10),
            spot(20, 12),
            spot(50, 30),
        ],
    );
    let (tx, rx) = std::sync::mpsc::channel();
    let trained = job(vec![image], 3)
        .run(tx, Arc::new(AtomicBool::new(false)))
        .unwrap();
    let TrainingStats::Network {
        epochs_run, check, ..
    } = trained.stats
    else {
        panic!("{:?}", trained.stats)
    };
    assert_eq!((epochs_run, check), (3, NetworkCheck::Off));
    let events: Vec<TrainingProgressEvent> = rx.try_iter().collect();
    assert_eq!(events[0], TrainingProgressEvent::Started { total: 1 });
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, TrainingProgressEvent::Epoch { .. }))
            .count(),
        3
    );
    assert!(matches!(
        events.last(),
        Some(TrainingProgressEvent::Finished { .. })
    ));

    let path = dir.path().join("models").join("spots.evanet");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    trained.model.save(&path).unwrap();
    let loaded = SpotNetModel::read(&path, &evanalyzer_nn::Device::flex()).unwrap();
    assert_eq!(loaded.info["seed"], "1");
}

#[test]
fn a_job_without_a_training_area_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let (tx, _rx) = std::sync::mpsc::channel();
    let error = job(vec![one_image(&dir, vec![spot(10, 10)])], 1)
        .run(tx, Arc::new(AtomicBool::new(false)))
        .unwrap_err()
        .to_string();
    assert!(error.contains("No training area"), "{error}");
}

#[test]
fn a_cancelled_job_stops() {
    let dir = tempfile::tempdir().unwrap();
    let image = one_image(&dir, vec![annotation(AREA, [0, 0, 63, 47]), spot(10, 10)]);
    let (tx, _rx) = std::sync::mpsc::channel();
    let result = job(vec![image], 5).run(tx, Arc::new(AtomicBool::new(true)));
    assert!(matches!(result, Err(InternalErrors::Cancelled)));
}
