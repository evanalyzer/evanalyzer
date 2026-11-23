//! # Subtract Reference Channel: tests
//!
//! On synthetic scenes as the Python reference builds them: a filament
//! network (thresholded smoothed noise, then a Gaussian of σ = 1.2 px),
//! EV spots (σ = 1.3 px) only in the target - half of them on the network -
//! photon noise and an offset of 100; target = 0.37 × network + spots +
//! background, the reference the network shifted by (0.8, −1.3) px. The
//! network is periodic, so the reference's shift is exact (in the
//! frequency domain).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::TranslateImage;
use crate::algos::libs::phase_correlation::test_support::{Lcg, fourier_blur, fourier_shift};
use crate::algos::{
    Alignment, Background, EstimationScope, Factor, ImageAlgorithm, Interpolation, OutputImage,
    Padding, Sharpness, SubtractReferenceChannel,
};
use crate::image::{ImageContainer, ManagedImage};
use crate::pipeline::pipeline_cache::{CacheAddress, GlobalPipelineCache};
use crate::pipeline::pipeline_context::PipelineContext;
use crate::pipeline::pipeline_measure::MeasurementValues;
use evanalyzer_cfg::core_types::{ImageChannelIdx, InternalErrors};
use evanalyzer_cfg::settings::pipeline_command_settings::SubtractReferenceChannelSettings;
use kornia_image::{Image, ImageSize};
use std::collections::BTreeMap;
use std::sync::Arc;

const SIDE: usize = 256;
/// The true factor: target structure / reference structure.
const A: f32 = 0.37;
/// The network's brightness in the reference (photons).
const NETWORK: f32 = 400.0;
/// The spots' peak (photons).
const SPOT: f32 = 300.0;
const SPOT_SIGMA: f32 = 1.3;
/// The peak of the reference-only dots (photons): brighter than most of
/// the network, as dye aggregates in a serum channel.
const DOT: f32 = 800.0;
/// The camera offset of both channels.
const OFFSET: f32 = 100.0;
/// The reference channel's index.
const REFERENCE: u32 = 4;
/// The reference's shift against the target (dy, dx).
const SHIFT: (f64, f64) = (0.8, -1.3);

/// A synthetic image pair.
struct Scene {
    /// The network as it lies in the target, 0 - 1.
    network: Vec<f32>,
    /// The spots' centres (x, y).
    spots: Vec<(f32, f32)>,
    target: Vec<f32>,
    reference: Vec<f32>,
}

/// What a scene is built with.
#[derive(Clone, Copy)]
struct Recipe {
    /// The reference's shift against the target.
    shift: (f64, f64),
    /// The reference shows the network (else only background and noise).
    structure: bool,
    /// The target shows the network.
    target_network: bool,
    /// The target's network blurred by this much more (σ, px): out of
    /// focus against the reference, so the subtraction leaves leftovers.
    target_blur: f64,
    noise: bool,
    /// Bright dots in the reference only (dark in the target, e.g. dye
    /// aggregates): their number.
    reference_dots: usize,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            shift: SHIFT,
            structure: true,
            target_network: true,
            target_blur: 0.0,
            noise: true,
            reference_dots: 0,
        }
    }
}

fn scene(recipe: Recipe) -> Scene {
    let (w, h) = (SIDE, SIDE);
    let mut rng = Lcg(42);
    // The network: smoothed noise, its brightest quarter, smoothed again.
    let noise: Vec<f32> = (0..w * h).map(|_| rng.next() - 0.5).collect();
    let smooth = fourier_blur(&noise, w, h, 3.0);
    let mut sorted = smooth.clone();
    sorted.sort_by(f32::total_cmp);
    let level = sorted[sorted.len() * 3 / 4];
    let binary: Vec<f32> = smooth.iter().map(|v| (*v > level) as u8 as f32).collect();
    let network = fourier_blur(&binary, w, h, 1.2);
    let shifted = fourier_shift(&network, w, h, recipe.shift.0, recipe.shift.1);
    let target_network = if recipe.target_blur > 0.0 {
        fourier_blur(&network, w, h, recipe.target_blur)
    } else {
        network.clone()
    };

    // 50 spots, half of them on the network.
    let mut spots = Vec::new();
    while spots.len() < 50 {
        let (x, y) = (
            4.0 + rng.next() * (w - 8) as f32,
            4.0 + rng.next() * (h - 8) as f32,
        );
        let on_network = network[y as usize * w + x as usize] > 0.6;
        if (spots.len() % 2 == 0) == on_network {
            spots.push((x, y));
        }
    }
    let mut spot_image = vec![0f32; w * h];
    for &(cx, cy) in &spots {
        for y in (cy as usize).saturating_sub(6)..(cy as usize + 7).min(h) {
            for x in (cx as usize).saturating_sub(6)..(cx as usize + 7).min(w) {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                spot_image[y * w + x] += SPOT * (-d2 / (2.0 * SPOT_SIGMA * SPOT_SIGMA)).exp();
            }
        }
    }

    // Drawn from their own generator: the other scenes stay as they were.
    let mut dot_rng = Lcg(7);
    let mut dot_image = vec![0f32; w * h];
    for _ in 0..recipe.reference_dots {
        let (cx, cy) = (
            4.0 + dot_rng.next() * (w - 8) as f32,
            4.0 + dot_rng.next() * (h - 8) as f32,
        );
        for y in (cy as usize).saturating_sub(6)..(cy as usize + 7).min(h) {
            for x in (cx as usize).saturating_sub(6)..(cx as usize + 7).min(w) {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                dot_image[y * w + x] += DOT * (-d2 / (2.0 * SPOT_SIGMA * SPOT_SIGMA)).exp();
            }
        }
    }

    let mut photons = |mean: f32| -> f32 {
        if recipe.noise {
            mean + mean.max(0.0).sqrt() * rng.normal()
        } else {
            mean
        }
    };
    let ramp = |x: usize, y: usize, a: f32, b: f32| {
        30.0 + a * x as f32 / w as f32 + b * y as f32 / h as f32
    };
    let mut target = vec![0f32; w * h];
    let mut reference = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let structure = if recipe.target_network {
                A * NETWORK * target_network[i]
            } else {
                0.0
            };
            target[i] = OFFSET + photons(structure + spot_image[i] + ramp(x, y, 40.0, 20.0));
            let reference_structure = if recipe.structure {
                NETWORK * shifted[i]
            } else {
                0.0
            };
            reference[i] =
                OFFSET + photons(reference_structure + dot_image[i] + ramp(x, y, -20.0, 30.0));
        }
    }
    Scene {
        network,
        spots,
        target,
        reference,
    }
}

/// The command with its defaults (but a 10 px median), reading the
/// reference from channel 4.
fn command() -> SubtractReferenceChannel {
    let mut command = SubtractReferenceChannel::from(SubtractReferenceChannelSettings::default());
    command.id = "reference".into();
    command.reference_channel = ImageChannelIdx(REFERENCE);
    // A smaller median than the default 30 px: the same path, a debug
    // build's tests stay fast (the ramps are slower still).
    command.background = Background::Median { radius: 10.0 };
    command
}

/// The result image and the measurements (by their key's end).
struct Run {
    image: Vec<f32>,
    values: BTreeMap<String, f64>,
}

impl Run {
    fn value(&self, key: &str) -> f64 {
        *self
            .values
            .get(key)
            .unwrap_or_else(|| panic!("no measurement {key}: {:?}", self.values))
    }
}

fn gray(width: usize, height: usize, values: Vec<f32>) -> Arc<ImageContainer> {
    Arc::new(ImageContainer::F32Gray(ManagedImage {
        data: Image::new(ImageSize { width, height }, values).unwrap(),
        tile_offset: Default::default(),
        plane: None,
        dimensionality: Default::default(),
    }))
}

fn try_run(
    command: &SubtractReferenceChannel,
    target: Arc<ImageContainer>,
    reference: Arc<ImageContainer>,
) -> Result<Run, InternalErrors> {
    let size = target.size();
    let mut ctx = PipelineContext::new_from_image_test(
        Image::new(size, vec![0.0; size.width * size.height]).unwrap(),
    )
    .unwrap();
    ctx.image = target;
    ctx.image_meta.nr_of_bits = 16;
    let mut cache = GlobalPipelineCache::default();
    cache.image_cache.insert(
        CacheAddress::Channel((REFERENCE as i32, ctx.image_meta.image_tile_info)),
        reference,
    );
    command.run(&mut ctx, &mut cache)?;
    let image = ctx.image.as_f32_slice().unwrap().to_vec();
    let values = cache
        .measurements
        .measurements
        .iter()
        .filter_map(|m| match &m.values {
            MeasurementValues::Image(values) => values
                .first()
                .map(|(_, v)| (m.info.key.rsplit('/').next().unwrap().to_string(), *v)),
            MeasurementValues::Objects(_) => None,
        })
        .collect();
    Ok(Run { image, values })
}

fn run(command: &SubtractReferenceChannel, scene: &Scene) -> Run {
    try_run(
        command,
        gray(SIDE, SIDE, scene.target.clone()),
        gray(SIDE, SIDE, scene.reference.clone()),
    )
    .unwrap()
}

/// The pixels on the network (> 0.5) at least 6 px from every spot and 8
/// px from the border.
fn network_pixels(scene: &Scene) -> Vec<usize> {
    (0..SIDE * SIDE)
        .filter(|&i| {
            let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
            scene.network[i] > 0.5
                && (8.0..SIDE as f32 - 8.0).contains(&x)
                && (8.0..SIDE as f32 - 8.0).contains(&y)
                && scene
                    .spots
                    .iter()
                    .all(|(sx, sy)| (sx - x).powi(2) + (sy - y).powi(2) > 36.0)
        })
        .collect()
}

fn mean_at(image: &[f32], pixels: &[usize]) -> f64 {
    pixels.iter().map(|&i| image[i] as f64).sum::<f64>() / pixels.len() as f64
}

fn rms_at(image: &[f32], pixels: &[usize]) -> f64 {
    (pixels
        .iter()
        .map(|&i| (image[i] as f64).powi(2))
        .sum::<f64>()
        / pixels.len() as f64)
        .sqrt()
}

/// The largest value within 1 px of each spot's centre.
fn spot_peaks(image: &[f32], scene: &Scene) -> Vec<f32> {
    scene
        .spots
        .iter()
        .filter(|(x, y)| {
            (8.0..SIDE as f32 - 8.0).contains(x) && (8.0..SIDE as f32 - 8.0).contains(y)
        })
        .map(|&(x, y)| {
            let (cx, cy) = (x.round() as usize, y.round() as usize);
            (cy - 1..=cy + 1)
                .flat_map(|yy| (cx - 1..=cx + 1).map(move |xx| image[yy * SIDE + xx]))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

#[test]
fn the_factor_is_fitted_on_aligned_channels() {
    let aligned = scene(Recipe {
        shift: (0.0, 0.0),
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    // The default median: 10 px eats into the wider filaments (-2 %).
    cmd.background = Background::Median { radius: 30.0 };
    let result = run(&cmd, &aligned);
    let a = result.value("factor");
    assert!((a - A as f64).abs() < 0.02 * A as f64, "a = {a}");
    assert!(result.value("r2") > 0.95, "{:?}", result.values);
}

#[test]
fn the_shift_is_found() {
    let result = run(&command(), &scene(Recipe::default()));
    // The reference lies (0.8, -1.3) off: it is shifted back.
    let (dy, dx) = (result.value("shift_y"), result.value("shift_x"));
    assert!(
        (dy + SHIFT.0).abs() < 0.05 && (dx + SHIFT.1).abs() < 0.05,
        "({dy}, {dx})"
    );
    assert!(
        result.value("alignment_quality") > 0.3,
        "{:?}",
        result.values
    );
    assert_eq!(result.value("fallback_used"), 0.0);
}

#[test]
fn the_structure_goes_and_the_spots_stay() {
    let scene = scene(Recipe::default());
    let mut cmd = command();
    cmd.clip_negative = false;
    let corrected = run(&cmd, &scene);
    // Factor 0: the target as processed (background, smoothing), nothing
    // subtracted.
    cmd.factor = Factor::Fixed { factor: 0.0 };
    let uncorrected = run(&cmd, &scene);

    let network = network_pixels(&scene);
    let before = mean_at(&uncorrected.image, &network);
    let after = mean_at(&corrected.image, &network);
    assert!(
        after.abs() < 0.1 * before,
        "structure {before:.1} -> {after:.1}"
    );

    // The spots as the same processing shows them without any network.
    let spots_only = self::scene(Recipe {
        target_network: false,
        noise: false,
        ..Default::default()
    });
    let reference_peaks = spot_peaks(&run(&cmd, &spots_only).image, &spots_only);
    let peaks = spot_peaks(&corrected.image, &scene);
    let kept = peaks.iter().sum::<f32>() / reference_peaks.iter().sum::<f32>();
    assert!(kept > 0.85, "spots keep {kept:.2}");
}

#[test]
fn aligning_before_fitting_matters() {
    let scene = scene(Recipe::default());
    let network = network_pixels(&scene);
    let mut cmd = command();
    cmd.clip_negative = false;
    let aligned = rms_at(&run(&cmd, &scene).image, &network);
    cmd.alignment = Alignment::None;
    let unaligned = rms_at(&run(&cmd, &scene).image, &network);
    assert!(
        unaligned > 1.5 * aligned,
        "leftover {aligned:.1} aligned, {unaligned:.1} not"
    );
}

#[test]
fn smoothing_keeps_the_factor_from_being_biased_low() {
    let aligned = scene(Recipe {
        shift: (0.0, 0.0),
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    let smoothed = run(&cmd, &aligned).value("factor");
    cmd.smoothing_sigma = 0.0;
    let raw = run(&cmd, &aligned).value("factor");
    assert!(
        raw < smoothed,
        "a {raw:.4} unsmoothed, {smoothed:.4} smoothed"
    );
    assert!(
        (A as f64 - raw).abs() > (A as f64 - smoothed).abs(),
        "a {raw:.4} unsmoothed, {smoothed:.4} smoothed"
    );
}

#[test]
fn dots_only_in_the_reference_dont_pull_the_factor_down() {
    // Bright in the reference, dark in the target: far below the line. Left
    // in the fit, they tilt it down, the network's brightest pixels then lie
    // above it and are left out as if they were EVs - until a is about 0.
    let dotted = scene(Recipe {
        shift: (0.0, 0.0),
        reference_dots: 40,
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    let result = run(&cmd, &dotted);
    let a = result.value("factor");
    assert!((a - A as f64).abs() < 0.05 * A as f64, "a = {a}");
    assert_eq!(result.value("fallback_used"), 0.0);
}

#[test]
fn a_target_without_the_reference_signal_subtracts_nothing() {
    // The reference shows the network, the target only its spots: the
    // factor is about 0 and precise - a result, not a failed fit (R² is
    // about 0 here and would have rejected it). A slightly negative
    // estimate is set to 0.
    let empty = scene(Recipe {
        shift: (0.0, 0.0),
        target_network: false,
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    let result = run(&cmd, &empty);
    let a = result.value("factor");
    assert!((0.0..0.01 * A as f64).contains(&a), "a = {a}");
    assert_eq!(result.value("fallback_used"), 0.0);
}

#[test]
fn a_reference_without_structure_falls_back() {
    let flat = scene(Recipe {
        structure: false,
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    let result = run(&cmd, &flat);
    assert_eq!(result.value("factor"), 0.0);
    assert_eq!(result.value("fallback_used"), 1.0);
}

#[test]
fn a_shift_beyond_the_largest_falls_back() {
    let far = scene(Recipe {
        shift: (8.0, 0.0),
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::Auto {
        max_shift: 5.0,
        min_quality: 0.3,
        fallback_dx: 0.5,
        fallback_dy: -0.5,
    };
    let result = run(&cmd, &far);
    assert_eq!(
        (result.value("shift_y"), result.value("shift_x")),
        (-0.5, 0.5)
    );
    assert_eq!(result.value("fallback_used"), 1.0);
}

#[test]
fn fixed_factor_and_shift_subtract_exactly() {
    let scene = scene(Recipe::default());
    let mut cmd = command();
    cmd.background = Background::None;
    cmd.smoothing_sigma = 0.0;
    cmd.clip_negative = false;
    cmd.alignment = Alignment::Fixed { dx: 1.3, dy: -0.8 };
    cmd.factor = Factor::Fixed { factor: A };
    let result = run(&cmd, &scene);

    // Translate Image on the reference, then target - a · shifted.
    let mut ctx = PipelineContext::new_from_image_test(
        Image::new(
            ImageSize {
                width: SIDE,
                height: SIDE,
            },
            scene.reference.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    TranslateImage {
        dy: -0.8f32 as f64,
        dx: 1.3f32 as f64,
        interpolation: Interpolation::Cubic,
        padding: Padding::Replicate,
        pad_value: 0.0,
    }
    .run(&mut ctx, &mut GlobalPipelineCache::default())
    .unwrap();
    let shifted = ctx.image.as_f32_slice().unwrap();
    let expected: Vec<f32> = scene
        .target
        .iter()
        .zip(shifted)
        .map(|(s, t)| s - (A * t + 0.0))
        .collect();
    assert_eq!(result.image, expected);
    assert_eq!(result.value("factor"), A as f64);
}

#[test]
fn the_exposure_ratio_sets_the_factor() {
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    cmd.factor = Factor::ExposureRatio {
        calibration: 2.0,
        exposure_target_ms: 50.0,
        exposure_reference_ms: 200.0,
    };
    let result = run(&cmd, &scene(Recipe::default()));
    assert_eq!(result.value("factor"), 0.5);
}

#[test]
fn saturated_pixels_dont_move_the_factor() {
    let mut aligned = scene(Recipe {
        shift: (0.0, 0.0),
        ..Default::default()
    });
    let mut cmd = command();
    cmd.alignment = Alignment::None;
    let clean = run(&cmd, &aligned);
    // 300 saturated pixels on the network.
    for &i in network_pixels(&aligned).iter().step_by(20).take(300) {
        aligned.target[i] = 65535.0;
    }
    let saturated = run(&cmd, &aligned);
    let (a, b) = (clean.value("factor"), saturated.value("factor"));
    assert!((a - b).abs() < 0.01 * a, "a {a:.4} -> {b:.4}");
    assert!(saturated.value("saturated_fraction") > 0.0);
    assert_eq!(clean.value("saturated_fraction"), 0.0);
}

#[test]
fn edge_tolerance_dims_the_fringe_of_a_leftover_shift() {
    let scene = scene(Recipe::default());
    // One pixel off in y (0.2 instead of -0.8): a bright fringe on one side
    // of every filament.
    let near_network: Vec<usize> = (0..SIDE * SIDE)
        .filter(|&i| scene.network[i] > 0.1)
        .collect();
    let fringe = |tolerance: f32| {
        let mut cmd = command();
        cmd.alignment = Alignment::Fixed { dx: 1.3, dy: 0.2 };
        cmd.factor = Factor::Fixed { factor: A };
        cmd.edge_tolerance = tolerance;
        mean_at(&run(&cmd, &scene).image, &near_network)
    };
    let (without, with) = (fringe(0.0), fringe(1.0));
    assert!(with < without, "fringe {without:.1} -> {with:.1}");
}

#[test]
fn every_measurement_is_written() {
    let result = run(&command(), &scene(Recipe::default()));
    for key in [
        "factor",
        "offset",
        "shift_x",
        "shift_y",
        "alignment_quality",
        "r2",
        "fit_pixel_fraction",
        "saturated_fraction",
        "fallback_used",
    ] {
        assert!(
            result.values.contains_key(key),
            "{key}: {:?}",
            result.values
        );
    }
    let fraction = result.value("fit_pixel_fraction");
    assert!(fraction > 0.01 && fraction < 0.5, "{fraction}");
    assert!(result.value("offset").abs() < 20.0, "{:?}", result.values);
    assert_eq!(result.value("saturated_fraction"), 0.0);
}

#[test]
fn both_scopes_agree_on_one_tile() {
    let scene = scene(Recipe::default());
    let mut cmd = command();
    let image = run(&cmd, &scene).value("factor");
    cmd.estimation_scope = EstimationScope::Tile;
    let tile = run(&cmd, &scene).value("factor");
    assert_eq!(image, tile);
}

#[test]
fn a_volume_is_corrected_plane_by_plane() {
    let scene = scene(Recipe::default());
    let single = run(&command(), &scene);
    let volume = |values: &[f32]| {
        Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: SIDE,
                    height: 2 * SIDE,
                },
                [values, values].concat(),
            )
            .unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: crate::ImageDimensionality::Volume { depth: 2 },
        }))
    };
    let stacked = try_run(&command(), volume(&scene.target), volume(&scene.reference)).unwrap();
    assert_eq!(stacked.image[..SIDE * SIDE], single.image[..]);
    assert_eq!(stacked.image[SIDE * SIDE..], single.image[..]);
    assert_eq!(stacked.value("factor"), single.value("factor"));
}

#[test]
fn what_doesnt_fit_is_refused() {
    let target = gray(32, 32, vec![1.0; 32 * 32]);
    // No channel 7.
    let mut cmd = command();
    cmd.reference_channel = ImageChannelIdx(7);
    let err = try_run(&cmd, target.clone(), gray(32, 32, vec![1.0; 32 * 32]))
        .err()
        .unwrap();
    assert!(err.to_string().contains('7'), "{err}");

    let err = try_run(&command(), target.clone(), gray(32, 16, vec![1.0; 32 * 16]))
        .err()
        .unwrap();
    assert!(err.to_string().contains("differ in size"), "{err}");

    let labels = Arc::new(ImageContainer::U32(ManagedImage {
        data: Image::new(
            ImageSize {
                width: 32,
                height: 32,
            },
            vec![1u32; 32 * 32],
        )
        .unwrap(),
        tile_offset: Default::default(),
        plane: None,
        dimensionality: Default::default(),
    }));
    let err = try_run(&command(), labels, gray(32, 32, vec![1.0; 32 * 32]))
        .err()
        .unwrap();
    assert!(err.to_string().contains("label images"), "{err}");
}

/// The doc comment's short step and the task's full example load, both
/// with the defaults.
#[test]
fn the_documented_module_json_loads() {
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    let short: PipelineCommand = serde_json::from_str(
        r#"{ "type": "SUBTRACT_REFERENCE_CHANNEL", "id": "reference", "referenceChannel": 4 }"#,
    )
    .unwrap();
    let full: PipelineCommand = serde_json::from_str(
        r#"{
          "type": "SUBTRACT_REFERENCE_CHANNEL",
          "target": { "type": "CURRENT_IMAGE" },
          "referenceChannel": 4,
          "cameraOffset": { "type": "NONE" },
          "saturation": { "type": "AUTO" },
          "background": { "type": "MEDIAN", "radius": 30.0 },
          "alignment": { "type": "AUTO", "maxShift": 5.0, "minQuality": 0.3, "fallbackDx": 0.0, "fallbackDy": 0.0 },
          "interpolation": "CUBIC",
          "targetExtraSigma": 0.0,
          "referenceExtraSigma": 0.0,
          "smoothingSigma": 1.0,
          "factor": { "type": "ROBUST_FIT", "noiseFactor": 4.0, "minPercentile": 50.0, "outlierK": 3.0,
                      "fitOffset": true, "maxUncertainty": 0.25, "fallback": 0.0, "minValue": 0.0, "maxValue": 100.0 },
          "factorScale": 1.0,
          "estimationScope": "IMAGE",
          "edgeTolerance": 0.0,
          "clipNegative": true,
          "output": "RESIDUAL"
        }"#,
    )
    .unwrap();
    let (
        PipelineCommand::SubtractReferenceChannel(short),
        PipelineCommand::SubtractReferenceChannel(mut full),
    ) = (short, full)
    else {
        panic!("Subtract Reference Channel steps")
    };
    full.id = "reference".into();
    let defaults = SubtractReferenceChannelSettings {
        id: "reference".into(),
        reference_channel: ImageChannelIdx(4),
        ..Default::default()
    };
    for settings in [short, full] {
        assert_eq!(
            serde_json::to_value(&settings).unwrap(),
            serde_json::to_value(&defaults).unwrap()
        );
    }
}

/// Its measurements are offered to later steps (Exclude Image) although it
/// isn't a measurement step; a plain filter offers none.
#[test]
fn its_measurements_are_offered_to_later_steps() {
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    let settings = SubtractReferenceChannelSettings {
        id: "reference".into(),
        ..Default::default()
    };
    let keys: Vec<String> =
        crate::measurement_outputs(&PipelineCommand::SubtractReferenceChannel(settings))
            .into_iter()
            .map(|info| info.key)
            .collect();
    assert!(keys.contains(&"reference/factor".to_string()), "{keys:?}");
    assert!(
        keys.contains(&"reference/fallback_used".to_string()),
        "{keys:?}"
    );
    let blur = PipelineCommand::default_for_key("blur").unwrap();
    assert!(crate::measurement_outputs(&blur).is_empty());
}

/// A dark crack in the reference (below its background) subtracts nothing
/// by default; with the option off it brightens the residual.
#[test]
fn a_dark_crack_in_the_reference_subtracts_nothing() {
    // Already background-free values: the reference 10 with a crack of -20
    // in column 2, the target 5 with its own shallower crack of -2.
    let (w, h) = (8, 8);
    let reference: Vec<f32> = (0..w * h)
        .map(|i| if i % w == 2 { -20.0 } else { 10.0 })
        .collect();
    let target: Vec<f32> = (0..w * h)
        .map(|i| if i % w == 2 { -2.0 } else { 5.0 })
        .collect();
    let mut cmd = command();
    cmd.background = Background::None;
    cmd.alignment = Alignment::None;
    cmd.smoothing_sigma = 0.0;
    cmd.factor = Factor::Fixed { factor: 0.5 };
    cmd.clip_negative = false;
    let residual = |cmd: &SubtractReferenceChannel| {
        try_run(
            cmd,
            gray(w, h, target.clone()),
            gray(w, h, reference.clone()),
        )
        .unwrap()
        .image
    };

    let on = residual(&cmd);
    // The structure (5 - 0.5 · 10) is gone, the target's crack stays as it is.
    assert_eq!((on[0], on[2]), (0.0, -2.0));
    cmd.only_brighter_reference = false;
    let off = residual(&cmd);
    // -2 - 0.5 · (-20): the crack turns bright.
    assert_eq!((off[0], off[2]), (0.0, 8.0));
}

/// With the target out of focus the subtraction leaves leftovers on the
/// network. In Significance units (the residual over its local noise) they
/// stay below the cut-off almost everywhere - unlike the residual over its
/// typical noise - while the EVs stay above, and beside the structure the
/// values spread by about 1.
#[test]
fn significance_keeps_the_evs_and_drops_the_leftovers() {
    let scene = scene(Recipe {
        target_blur: 1.5,
        ..Default::default()
    });
    let mut cmd = command();
    cmd.clip_negative = false;
    let residual = run(&cmd, &scene);
    cmd.output = OutputImage::Significance;
    let significance = run(&cmd, &scene);
    assert!(significance.value("residual_noise") > 0.0);
    // Beside the structure, away from the EVs.
    let beside = |image: &[f32]| -> Vec<f32> {
        (0..SIDE * SIDE)
            .filter(|&i| {
                let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
                scene.network[i] < 0.02
                    && scene
                        .spots
                        .iter()
                        .all(|(sx, sy)| (sx - x).powi(2) + (sy - y).powi(2) > 64.0)
            })
            .map(|i| image[i])
            .collect()
    };
    let robust_spread = |mut values: Vec<f32>| -> f32 {
        values.sort_by(f32::total_cmp);
        let median = values[values.len() / 2];
        let mut deviations: Vec<f32> = values.iter().map(|v| (v - median).abs()).collect();
        deviations.sort_by(f32::total_cmp);
        deviations[deviations.len() / 2] * 1.4826
    };
    // The residual's noise there: what a single global cut-off would use.
    let noise = robust_spread(beside(&residual.image));

    // Near the network (its flanks too, where a width mismatch leaves its
    // leftovers), away from the EVs.
    let near: Vec<usize> = (0..SIDE * SIDE)
        .filter(|&i| {
            let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
            scene.network[i] > 0.02
                && (8.0..SIDE as f32 - 8.0).contains(&x)
                && (8.0..SIDE as f32 - 8.0).contains(&y)
                && scene
                    .spots
                    .iter()
                    .all(|(sx, sy)| (sx - x).powi(2) + (sy - y).powi(2) > 64.0)
        })
        .collect();
    let above = |image: &[f32], unit: f32, cut: f32| {
        near.iter().filter(|&&i| image[i] / unit > cut).count() as f64 / near.len() as f64
    };
    for (cut, most) in [(3.0, 0.02), (5.0, 0.005)] {
        let (residual, significance) = (
            above(&residual.image, noise, cut),
            above(&significance.image, 1.0, cut),
        );
        assert!(
            significance < most && residual > 2.0 * significance.max(1e-4),
            "pixels near the network above {cut}: residual {residual:.4}, significance \
             {significance:.4}"
        );
    }

    // Every EV above the cut-off of 5 - also those on the network, where
    // the leftovers raise the local noise.
    let mut peaks = spot_peaks(&significance.image, &scene);
    peaks.sort_by(f32::total_cmp);
    let (weakest, median) = (peaks[0], peaks[peaks.len() / 2]);
    assert!(
        weakest > 5.0 && median > 10.0,
        "EVs: weakest {weakest:.1}, median {median:.1}"
    );

    // Beside the structure: noise of about 1.
    let spread = robust_spread(beside(&significance.image));
    assert!(
        (0.7..1.4).contains(&spread),
        "spread beside the structure {spread:.2}"
    );
}

/// With the target out of focus (σ 1.5 px more than the reference),
/// Sharpness Auto finds the reference's extra blur. Unmatched, the sharper
/// reference's peaks don't fit the target's and pull the factor low (0.27
/// for the true 0.37); matched, it is about as close as with the blur set
/// by hand (a little less blur: the estimate counts the damping of the
/// reference's cubic shift too), and the fit tighter.
#[test]
fn auto_sharpness_matches_an_out_of_focus_target() {
    let scene = scene(Recipe {
        target_blur: 1.5,
        ..Default::default()
    });
    let mut cmd = command();
    let unmatched = run(&cmd, &scene);
    cmd.reference_extra_sigma = 1.5;
    let by_hand = run(&cmd, &scene);
    cmd.reference_extra_sigma = 0.0;
    cmd.sharpness = Sharpness::Auto { max_sigma: 3.0 };
    let auto = run(&cmd, &scene);
    let reference_sigma = auto.value("reference_extra_sigma");
    assert!(
        (reference_sigma - 1.5).abs() < 0.3 && auto.value("target_extra_sigma") == 0.0,
        "{:?}",
        auto.values
    );
    let error = |run: &Run| (run.value("factor") - A as f64).abs();
    assert!(
        error(&auto) < 0.6 * error(&unmatched) && error(&auto) < error(&by_hand) + 0.03,
        "factor (true {A}): unmatched {:.3}, by hand {:.3}, auto {:.3} (reference blur {reference_sigma:.2})",
        unmatched.value("factor"),
        by_hand.value("factor"),
        auto.value("factor")
    );
    assert!(
        auto.value("r2") > unmatched.value("r2"),
        "{:?}",
        auto.values
    );
}

/// An estimated blur is only used where the factor fits at least as well
/// as with the blurs set by hand - also with dots in the reference only,
/// fine bright structure the target lacks (as beads of another colour
/// ratio), which can read as a blur difference.
#[test]
fn auto_sharpness_never_fits_worse_than_by_hand() {
    for recipe in [
        Recipe::default(),
        Recipe {
            target_blur: 1.0,
            ..Default::default()
        },
        Recipe {
            reference_dots: 40,
            ..Default::default()
        },
    ] {
        let scene = scene(recipe);
        let mut cmd = command();
        let by_hand = run(&cmd, &scene);
        cmd.sharpness = Sharpness::Auto { max_sigma: 3.0 };
        let auto = run(&cmd, &scene);
        assert!(
            auto.value("r2") >= by_hand.value("r2") - 1e-9,
            "target blur {}, dots {}: R² by hand {:.4}, auto {:.4} ({:?})",
            recipe.target_blur,
            recipe.reference_dots,
            by_hand.value("r2"),
            auto.value("r2"),
            auto.values
        );
    }
}

/// Without the reference's structure in the target there is nothing to
/// compare: the extra blurs set by hand are used.
#[test]
fn auto_sharpness_without_structure_uses_the_blurs_set_by_hand() {
    let scene = scene(Recipe {
        target_network: false,
        ..Default::default()
    });
    let mut cmd = command();
    cmd.sharpness = Sharpness::Auto { max_sigma: 3.0 };
    cmd.reference_extra_sigma = 0.4;
    let result = run(&cmd, &scene);
    assert_eq!(result.value("reference_extra_sigma"), 0.4f32 as f64);
    assert_eq!(result.value("target_extra_sigma"), 0.0);
}

/// No sharpness estimate is no fallback: the blurs set by hand are a fine
/// default, and "Fallback used" stays about the shift and the factor.
#[test]
fn a_missing_sharpness_estimate_is_no_fallback() {
    let scene = scene(Recipe {
        target_blur: 1.0,
        ..Default::default()
    });
    let mut cmd = command();
    // Too small a largest blur: the estimate isn't believed.
    cmd.sharpness = Sharpness::Auto { max_sigma: 0.0 };
    let result = run(&cmd, &scene);
    assert_eq!(result.value("fallback_used"), 0.0, "{:?}", result.values);
}

/// Manual adds no measurements; a project file without the setting is
/// Manual (as before it existed), Auto loads with its largest blur.
#[test]
fn manual_sharpness_is_the_default_and_measures_nothing_new() {
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    let result = run(&command(), &scene(Recipe::default()));
    assert!(!result.values.contains_key("target_extra_sigma"));
    assert!(!result.values.contains_key("reference_extra_sigma"));
    let parse = |json: &str| match serde_json::from_str::<PipelineCommand>(json).unwrap() {
        PipelineCommand::SubtractReferenceChannel(settings) => {
            SubtractReferenceChannel::from(settings).sharpness
        }
        _ => panic!("a Subtract Reference Channel step"),
    };
    assert_eq!(
        parse(r#"{ "type": "SUBTRACT_REFERENCE_CHANNEL", "referenceExtraSigma": 0.9 }"#),
        Sharpness::Manual
    );
    assert_eq!(
        parse(
            r#"{ "type": "SUBTRACT_REFERENCE_CHANNEL",
                 "sharpness": { "type": "AUTO", "maxSigma": 2.5 } }"#
        ),
        Sharpness::Auto { max_sigma: 2.5 }
    );
}
