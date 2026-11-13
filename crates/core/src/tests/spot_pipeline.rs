//! # spot_pipeline
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! The spot pipeline for noisy fluorescence images (e.g. vesicles) end to
//! end on a synthetic image of known spots: Rolling Ball -> Gaussian Blur ->
//! Find Maxima with the Noise cut-off. The same scene is imaged at full and
//! at a quarter of the exposure (Poisson shot noise, a constant camera
//! offset, an uneven background); one noise factor has to find the spots in
//! both.

use evanalyzer_cfg::core_types::{PixelUnits, SegmentationClass, SizeUnits};
use kornia_image::ImageSize;

use crate::{
    F32Gray,
    algos::{BallType, FindMaxima, GaussianBlur, ImageAlgorithm, MaximaCutoff, RollingBall},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

const SIZE: usize = 512;
/// 12-bit camera: counts / 4095 is the relative image.
const FULL_SCALE: f64 = 4095.0;
/// Camera offset in counts - the same at every exposure.
const OFFSET: f64 = 100.0;
/// Background photons at full exposure: 60 on the left to 120 on the right.
const BACKGROUND: (f64, f64) = (60.0, 120.0);
/// Spot width (Gaussian sigma) in pixels: spots of about 4-6 px diameter.
const SPOT_SIGMA: f64 = 1.5;

/// A tiny deterministic generator (xorshift64*), so the test image is the
/// same on every computer.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Poisson with mean `lambda`: Knuth's product method below 30, the
    /// rounded normal approximation above.
    fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda < 30.0 {
            let limit = (-lambda).exp();
            let (mut k, mut p) = (0.0, 1.0);
            loop {
                p *= self.next_f64();
                if p <= limit {
                    return k;
                }
                k += 1.0;
            }
        }
        (lambda + lambda.sqrt() * self.normal()).round().max(0.0)
    }
}

/// The spots: (x, y, peak photons at full exposure) on a jittered 7 x 7
/// grid, peaks from 40 to 280 photons.
fn spots() -> Vec<(f64, f64, f64)> {
    let mut rng = Rng(0x5eed_0001);
    let mut spots = Vec::new();
    for gy in 0..7 {
        for gx in 0..7 {
            let x = 40.0 + gx as f64 * 72.0 + rng.next_f64() * 20.0 - 10.0;
            let y = 40.0 + gy as f64 * 72.0 + rng.next_f64() * 20.0 - 10.0;
            let peak = 40.0 + ((gy * 7 + gx) % 7) as f64 * 40.0;
            spots.push((x, y, peak));
        }
    }
    spots
}

/// The scene imaged at `exposure` (1 = full): photons scale with it, the
/// offset doesn't; every pixel gets Poisson noise.
fn image(exposure: f64, seed: u64) -> PipelineContext {
    let spots = spots();
    let mut rng = Rng(seed);
    let mut pixels = vec![0.0f32; SIZE * SIZE];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let t = x as f64 / (SIZE - 1) as f64;
            let mut photons = BACKGROUND.0 + t * (BACKGROUND.1 - BACKGROUND.0);
            for &(sx, sy, peak) in &spots {
                let d2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                if d2 < 100.0 {
                    photons += peak * (-d2 / (2.0 * SPOT_SIGMA * SPOT_SIGMA)).exp();
                }
            }
            let counts = OFFSET + rng.poisson(photons * exposure);
            pixels[y * SIZE + x] = (counts / FULL_SCALE) as f32;
        }
    }
    let size = ImageSize {
        width: SIZE,
        height: SIZE,
    };
    let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
    ctx.image_meta.nr_of_bits = 16;
    ctx.get_f32_gray_image_mut()
        .unwrap()
        .as_slice_mut()
        .copy_from_slice(&pixels);
    ctx
}

/// The suggested pipeline's image steps for spots of about 5 px.
fn pipeline(cutoff: MaximaCutoff, min_quality: f32) -> Vec<Box<dyn ImageAlgorithm>> {
    vec![
        // Uneven background away; radius well above a spot's.
        Box::new(RollingBall {
            radius: 10.0,
            ball_type: BallType::Paraboloid,
            pre_smooth: true,
        }),
        // Smoothing at the spot size: one peak per spot.
        Box::new(GaussianBlur {
            kernel_size: 9,
            sigma: 1.5,
        }),
        Box::new(FindMaxima {
            radius: 3.0,
            radius_unit: SizeUnits::Pixels,
            cutoff,
            min_quality,
            quality_unit: PixelUnits::Relative,
            inside_segmentation: false,
            exclude_at_edges: false,
            segmentation_class: SegmentationClass(1),
        }),
    ]
}

/// Runs the pipeline and returns the found spots as (x, y).
fn detect(mut ctx: PipelineContext, cutoff: MaximaCutoff, min_quality: f32) -> Vec<(f64, f64)> {
    let mut cache = GlobalPipelineCache::default();
    for step in pipeline(cutoff, min_quality) {
        step.execute(&mut ctx, &mut cache).unwrap();
    }
    let segmentation = ctx.segmentation_map.as_ref().unwrap().as_slice();
    (0..segmentation.len())
        .filter(|&p| segmentation[p] != 0)
        .map(|p| ((p % SIZE) as f64, (p / SIZE) as f64))
        .collect()
}

/// (true spots found, false detections): a found spot lies within 2 px of
/// a true one, each true spot counts once.
fn score(found: &[(f64, f64)]) -> (usize, usize) {
    let truth = spots();
    let mut used = vec![false; found.len()];
    let mut hits = 0;
    for &(tx, ty, _) in &truth {
        let nearest = found
            .iter()
            .enumerate()
            .filter(|&(i, _)| !used[i])
            .map(|(i, &(x, y))| (i, (x - tx).powi(2) + (y - ty).powi(2)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d2)) = nearest
            && d2 <= 4.0
        {
            used[i] = true;
            hits += 1;
        }
    }
    (hits, found.len() - hits)
}

/// The suggested noise factor 5 finds (nearly) every spot from full down
/// to a tenth of the exposure with (nearly) no false detection. 4 is too
/// low here: the noise left by the rolling ball has a longer bright tail
/// than normal noise (9 false detections at full exposure).
#[test]
fn noise_cutoff_finds_the_spots_at_every_exposure() {
    let cutoff = MaximaCutoff::Noise { noise_factor: 5.0 };
    let total = spots().len();
    let full = score(&detect(image(1.0, 1), cutoff, 0.0));
    assert_eq!(full, (total, 0), "full exposure: (found, false)");
    // A quarter and a tenth of the photons: the faintest spots (peak 4-10
    // photons on 6-30 background) sink into the noise.
    let quarter = score(&detect(image(0.25, 2), cutoff, 0.0));
    assert!(
        quarter.0 >= total - 2 && quarter.1 <= 3,
        "quarter {quarter:?}"
    );
    let tenth = score(&detect(image(0.1, 3), cutoff, 0.0));
    assert!(tenth.0 >= total - 8 && tenth.1 <= 3, "tenth {tenth:?}");
}

/// Why the Noise cut-off: a manual cut-off that is right at full exposure
/// loses many spots at a quarter of it (29 of 49 found).
#[test]
fn a_manual_cutoff_does_not_carry_over_to_another_exposure() {
    let manual = 20.0 / FULL_SCALE as f32;
    let total = spots().len();
    let full = score(&detect(image(1.0, 1), MaximaCutoff::Manual, manual));
    assert_eq!(full, (total, 0), "full exposure: (found, false)");
    let quarter = score(&detect(image(0.25, 2), MaximaCutoff::Manual, manual));
    assert!(quarter.0 < total * 2 / 3, "quarter {quarter:?}");
}
