//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! SpotNet through [`Model`]: pre → infer → post, as the engine runs it.

use super::{
    PercentileNormalization, Spot, SpotLabels, SpotNetConfig, SpotNetModel, SpotNetParams,
    SpotNetSettings, SpotNetTrainParams,
};
use crate::NnError;
use crate::augment::AugmentParams;
use crate::model::{Dataset, Model, Normalization, OwnedImage, Trainable, Written};
use crate::pieces::Pieces;
use crate::train::{NetworkTrainParams, TrainControl, TrainDevice, Validation};
use burn::tensor::Device;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

/// An untrained model (random weights from `seed`) of `in_channels`.
fn random_model(in_channels: usize, seed: u64) -> SpotNetModel {
    let network = SpotNetConfig {
        in_channels,
        channels: 4,
        ..SpotNetConfig::default()
    };
    SpotNetModel {
        network: super::network::SpotNet::new(&network, seed, &Device::flex()),
        settings: SpotNetSettings {
            network,
            normalization: PercentileNormalization::default(),
            sigma: 1.5,
        },
        info: BTreeMap::new(),
        device: Device::flex(),
    }
}

fn params(pieces: Pieces) -> SpotNetParams {
    SpotNetParams {
        normalization: Normalization::ModelDefault,
        pieces,
        threshold: 0.5,
    }
}

/// `model` on `image` through pre → infer → post: the labels and what was
/// written.
fn run(
    model: &SpotNetModel,
    image: &OwnedImage,
    params: &SpotNetParams,
) -> Result<(Vec<u32>, Written), NnError> {
    let (input, plan) = SpotNetModel::pre(&model.to_header(), image.view(), params)?;
    let raw = model.infer(&input)?;
    let mut out = vec![0; image.width * image.height];
    let written = SpotNetModel::post(raw, plan, params, &mut out)?;
    Ok((out, written))
}

/// The probability map only (pre → infer).
fn probabilities(model: &SpotNetModel, image: &OwnedImage, params: &SpotNetParams) -> Vec<f32> {
    let (input, _) = SpotNetModel::pre(&model.to_header(), image.view(), params).unwrap();
    model.infer(&input).unwrap().probability
}

/// A test image like a microscope's: background, a blob (sigma 2 px) per
/// spot, scaled by `gain` (8-bit vs 16-bit), `channels` copies.
fn spot_image(
    width: usize,
    height: usize,
    spots: &[Spot],
    gain: f32,
    channels: usize,
) -> OwnedImage {
    let mut plane = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut v = 0.1;
            for s in spots {
                let (dx, dy) = (x as f32 + 0.5 - s.x, y as f32 + 0.5 - s.y);
                v += 0.8 * (-(dx * dx + dy * dy) / 8.0).exp();
            }
            plane[y * width + x] = v * gain;
        }
    }
    OwnedImage {
        width,
        height,
        channels,
        data: plane.repeat(channels),
    }
}

#[test]
fn a_tile_in_pieces_gives_what_one_pass_gives() {
    // 600 x 300, two channels: pieces of 256 (each with the 17 px halo) vs
    // the tile in one pass - equal up to float rounding.
    let model = random_model(2, 3);
    let (width, height) = (600, 300);
    let data = (0..2 * width * height)
        .map(|i| ((i * 7919) % 101) as f32)
        .collect();
    let image = OwnedImage {
        width,
        height,
        channels: 2,
        data,
    };
    let whole = probabilities(&model, &image, &params(Pieces::fixed(4096)));
    let pieces = probabilities(&model, &image, &params(Pieces::fixed(256)));
    let worst = whole
        .iter()
        .zip(&pieces)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "pieces differ by {worst}");
}

/// Training images held in memory.
struct Images(Vec<(OwnedImage, SpotLabels)>);

impl Dataset<SpotLabels> for Images {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Result<(OwnedImage, SpotLabels), NnError> {
        Ok(self.0[index].clone())
    }
}

const SPOTS: [Spot; 4] = [
    Spot { x: 8.5, y: 8.5 },
    Spot { x: 22.5, y: 12.5 },
    Spot { x: 14.5, y: 24.5 },
    Spot { x: 1.5, y: 18.5 },
];

/// A SpotNet trained on one 32 x 32 image with [`SPOTS`] (overfit - the
/// point here is the way through `Model`, not generalising).
fn trained() -> (SpotNetModel, OwnedImage) {
    let image = spot_image(32, 32, &SPOTS, 1.0, 1);
    let labels = SpotLabels {
        width: 32,
        height: 32,
        spots: SPOTS.to_vec(),
        area: vec![true; 32 * 32],
        validation: vec![false; 32 * 32],
    };
    let data = Images(vec![(image.clone(), labels)]);
    let training = training_params(150);
    let progress = |_| {};
    let cancel = AtomicBool::new(false);
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed: 7,
        device: TrainDevice::Cpu,
        cores: 1,
    };
    // Through the trait the training job uses.
    let trained = <SpotNetModel as Trainable>::train(None, &data, &training, &control).unwrap();
    (trained.model, image)
}

/// The settings of [`trained`]: 1 channel, 8 feature maps, one patch (the
/// whole image) per epoch, nothing held back.
fn training_params(epochs: usize) -> SpotNetTrainParams {
    SpotNetTrainParams {
        common: NetworkTrainParams {
            epochs,
            patches_per_epoch: 1,
            batch_size: 1,
            patch_size: 128,
            learning_rate: 1e-2,
            augment: AugmentParams::none(),
            validation: Validation::Off,
            patience: None,
        },
        network: SpotNetConfig {
            in_channels: 1,
            channels: 8,
            ..SpotNetConfig::default()
        },
        sigma: 1.5,
        normalization: PercentileNormalization::default(),
    }
}

/// One 32 x 32 image with [`SPOTS`], all of it a training area.
fn one_image() -> Images {
    let image = spot_image(32, 32, &SPOTS, 1.0, 1);
    let labels = SpotLabels {
        width: 32,
        height: 32,
        spots: SPOTS.to_vec(),
        area: vec![true; 32 * 32],
        validation: vec![false; 32 * 32],
    };
    Images(vec![(image, labels)])
}

fn quiet_training(
    start: Option<&std::path::Path>,
    params: &SpotNetTrainParams,
) -> Result<crate::train::Trained<SpotNetModel>, NnError> {
    let progress = |_| {};
    let cancel = AtomicBool::new(false);
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed: 3,
        device: TrainDevice::Cpu,
        cores: 1,
    };
    <SpotNetModel as Trainable>::train(start, &one_image(), params, &control)
}

#[test]
fn a_model_file_is_fine_tuned_and_keeps_its_settings() {
    // A model of sigma 1.5 saved, then fine-tuned with settings asking for
    // another sigma and network: the file's stay, the weights change.
    let first = quiet_training(None, &training_params(2)).unwrap().model;
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("spots.evanet");
    first.save(&path).unwrap();
    let mut other = training_params(2);
    other.sigma = 3.0;
    other.network.channels = 16;
    let tuned = quiet_training(Some(&path), &other).unwrap().model;
    assert_eq!(tuned.settings, first.settings);
    let image = spot_image(32, 32, &SPOTS, 1.0, 1);
    let p = params(Pieces::fixed(4096));
    assert_ne!(
        probabilities(&first, &image, &p),
        probabilities(&tuned, &image, &p)
    );
}

#[test]
fn a_missing_model_file_to_fine_tune_is_an_error() {
    let dir = tempfile::TempDir::new().unwrap();
    assert!(quiet_training(Some(&dir.path().join("none.evanet")), &training_params(1)).is_err());
}

#[test]
fn a_larger_patch_takes_more_ram_per_sample() {
    let small = training_params(1);
    let mut large = training_params(1);
    large.common.patch_size *= 2;
    let (a, b) = (
        SpotNetModel::ram_per_sample(&small),
        SpotNetModel::ram_per_sample(&large),
    );
    // Twice the side, four times the pixels.
    assert!(a > 0 && (b as f64 / a as f64 - 4.0).abs() < 0.01, "{a} {b}");
}

#[test]
fn a_trained_model_marks_every_spot_as_its_own_one_pixel_object() {
    let (model, image) = trained();
    let (labels, written) = run(&model, &image, &params(Pieces::fixed(4096))).unwrap();
    assert_eq!(written, Written::Instances(SPOTS.len() as u32));
    // Every annotated spot found once, on its pixel or the next one: the
    // spot at the left edge (x 1.5) comes out at x 0 - beyond the edge the
    // network sees zeros (no mirrored border yet, TODO).
    let found: Vec<(usize, usize, u32)> = labels
        .iter()
        .enumerate()
        .filter(|&(_, &id)| id != 0)
        .map(|(i, &id)| (i % 32, i / 32, id))
        .collect();
    for spot in SPOTS {
        let (sx, sy) = (spot.x as usize, spot.y as usize);
        let near: Vec<_> = found
            .iter()
            .filter(|&&(x, y, _)| x.abs_diff(sx) <= 1 && y.abs_diff(sy) <= 1)
            .collect();
        assert_eq!(near.len(), 1, "spot ({sx}, {sy}): {found:?}");
    }
    let mut ids: Vec<u32> = found.iter().map(|&(_, _, id)| id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3, 4]);
}

#[test]
fn the_same_picture_in_8_and_16_bit_gives_the_same_spots() {
    // The model file's normalisation (per channel, per tile) takes the
    // gain out.
    let model = random_model(1, 4);
    let low = spot_image(64, 48, &SPOTS, 200.0, 1);
    let high = spot_image(64, 48, &SPOTS, 50_000.0, 1);
    let p = params(Pieces::fixed(4096));
    let (a, b) = (
        probabilities(&model, &low, &p),
        probabilities(&model, &high, &p),
    );
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn another_normalisation_chosen_by_the_step_changes_the_input() {
    let model = random_model(1, 4);
    let image = spot_image(64, 48, &SPOTS, 200.0, 1);
    let default = probabilities(&model, &image, &params(Pieces::fixed(4096)));
    let none = SpotNetParams {
        normalization: Normalization::None,
        ..params(Pieces::fixed(4096))
    };
    assert_ne!(default, probabilities(&model, &image, &none));
}

#[test]
fn the_header_is_read_from_the_file_without_its_weights() {
    let model = random_model(2, 1);
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("spots.evanet");
    model.save(&path).unwrap();
    assert_eq!(SpotNetModel::header(&path).unwrap(), model.to_header());
    let loaded = SpotNetModel::load(&path, &Device::flex(), false).unwrap();
    assert_eq!(loaded.settings, model.settings);
}

#[test]
fn an_image_with_other_channels_than_the_model_is_refused() {
    let model = random_model(2, 1);
    let image = spot_image(16, 16, &[], 1.0, 1);
    let result = run(&model, &image, &params(Pieces::fixed(4096)));
    assert!(matches!(result, Err(NnError::InvalidInput(m)) if m.contains("2 channel")));
}

#[test]
fn a_threshold_outside_0_to_1_is_refused() {
    let model = random_model(1, 1);
    let image = spot_image(16, 16, &[], 1.0, 1);
    let bad = SpotNetParams {
        threshold: 1.5,
        ..params(Pieces::fixed(4096))
    };
    assert!(
        matches!(run(&model, &image, &bad), Err(NnError::InvalidInput(m)) if m.contains("threshold"))
    );
}

#[test]
fn an_impossible_normalisation_is_refused() {
    let model = random_model(1, 1);
    let image = spot_image(16, 16, &[], 1.0, 1);
    let bad = SpotNetParams {
        normalization: Normalization::Percentile {
            low: 90.0,
            high: 10.0,
        },
        ..params(Pieces::fixed(4096))
    };
    let result = SpotNetModel::pre(&model.to_header(), image.view(), &bad);
    assert!(result.is_err());
}

#[test]
fn an_output_of_the_wrong_size_is_refused() {
    let model = random_model(1, 1);
    let image = spot_image(16, 16, &[], 1.0, 1);
    let p = params(Pieces::fixed(4096));
    let (input, plan) = SpotNetModel::pre(&model.to_header(), image.view(), &p).unwrap();
    let raw = model.infer(&input).unwrap();
    let mut out = vec![0; 10];
    assert!(SpotNetModel::post(raw, plan, &p, &mut out).is_err());
}

#[test]
fn an_empty_image_is_refused() {
    let model = random_model(1, 1);
    let image = OwnedImage {
        width: 0,
        height: 0,
        channels: 1,
        data: vec![],
    };
    assert!(
        SpotNetModel::pre(
            &model.to_header(),
            image.view(),
            &params(Pieces::fixed(4096))
        )
        .is_err()
    );
}
