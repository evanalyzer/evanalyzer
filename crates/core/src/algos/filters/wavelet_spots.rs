//! # wavelet_spots
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Spot enhancement by the multiscale product of thresholded à trous wavelet
//! planes (Olivo-Marin 2002): a spot stands out of the noise at several
//! levels of detail at once, noise only at one, so their product keeps the
//! spots. See [`multiscale_product`].

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;
use std::sync::Arc;

use crate::{
    algos::{
        ExecutionScope, ImageAlgorithm,
        libs::{
            robust_stats::{RADIX_BUCKETS, median_and_mad_streamed},
            scale_space::{Tap, convolve_separable_rolling},
        },
    },
    image::ImageContainer,
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Enhances spots by combining several levels of detail: keeps what stands
/// out of the noise at each level from `first_level` to `levels`.
///
/// [Wavelet Spots] -> [Find Maxima] -> [Extract Objects]
///
/// The result is positive at bright spots and exactly 0 everywhere else
/// (background, dark spots, what stands out at one level only). With output
/// Product it is the paper's multiscale product: the image's intensities to
/// the power of the number of combined levels, so a manual cut-off in Find
/// Maxima is the paper's detection level in these units. With output
/// Intensity it is in the image's units. Measure intensities on the
/// original channel, not on this image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Wavelet Spots")]
pub struct WaveletSpots {
    /// How many levels are computed: level i holds details of about 2^i
    /// pixels, so 3 suits spots up to about 8 px.
    #[cmdsmeta(default = 3, min = 2, max = 5, step = 1, summary = true)]
    pub levels: usize,

    /// The finest level that is combined; the levels below are skipped.
    /// Level 1 is mostly noise: 2 gives fewer false spots for spots of 4 px
    /// and more. At most `levels`.
    #[cmdsmeta(default = 1, min = 1, max = 5, step = 1)]
    pub first_level: usize,

    /// A level counts at a pixel only where it is at least this many times
    /// its noise; 3 in the paper.
    #[cmdsmeta(default = 3.0, min = 0.0, max = 20.0, step = 0.5)]
    pub noise_factor: f32,

    /// How the noise of each level is estimated.
    #[cmdsmeta(default = WaveletNoise::PerLevel, visibility = Advanced)]
    pub noise: WaveletNoise,

    /// The unit of the result.
    #[cmdsmeta(default = WaveletOutput::Product)]
    pub output: WaveletOutput,
}

/// How [`WaveletSpots`] estimates the noise of each level.
#[derive(Clone, Copy, PartialEq, Eq, Debug, CommandsMeta)]
pub enum WaveletNoise {
    /// From each level on its own, as in the paper. Where spots or other
    /// structures fill much of the image, the coarser levels count them as
    /// noise and their threshold rises.
    PerLevel,
    /// From the finest level, which holds the least structure, scaled to
    /// the other levels as Gaussian noise passes through the transform.
    /// Steadier with many spots or a structured background.
    FinestLevel,
}

/// The unit of [`WaveletSpots`]' result.
#[derive(Clone, Copy, PartialEq, Eq, Debug, CommandsMeta)]
pub enum WaveletOutput {
    /// The paper's product of the combined levels: intensity to the power
    /// of their number.
    Product,
    /// The geometric mean of the combined levels (the product's n-th root):
    /// the same spots and maxima, in the image's units, so Find Maxima's
    /// cut-off and prominence don't change with the number of levels.
    Intensity,
}

impl ImageAlgorithm for WaveletSpots {
    /// See [`multiscale_product`]. Works on the tile with mirrored borders,
    /// like Gaussian Blur.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                let (width, height) = (input.width(), input.height());
                let n = width * height;
                let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
                multiscale_product(
                    input.data.as_slice(),
                    output.data.as_slice_mut(),
                    &mut a,
                    &mut b,
                    width,
                    height,
                    &self.params(),
                );
                ctx.swap()?;
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Wavelet Spots"
    }

    /// The two approximations; the rows the convolution keeps for its
    /// column pass (up to 2^levels + 1) and the noise estimate's histogram
    /// (256 KB) are small. Measured by `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&OLIVO_MARIN_2002, &STARCK_MURTAGH_1994]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl WaveletSpots {
    fn params(&self) -> Params {
        let levels = self.levels.clamp(1, 16) as u32;
        Params {
            levels,
            first_level: (self.first_level as u32).clamp(1, levels),
            noise_factor: self.noise_factor,
            noise: self.noise,
            output: self.output,
        }
    }
}

/// [`multiscale_product`]'s settings; `1 ≤ first_level ≤ levels`.
#[derive(Clone, Copy, Debug)]
struct Params {
    levels: u32,
    first_level: u32,
    noise_factor: f32,
    noise: WaveletNoise,
    output: WaveletOutput,
}

/// `σ_i = MAD / 0.67`: the divisor Olivo-Marin 2002 (sect. 3) gives for the
/// median absolute deviation of the wavelet coefficients.
const MAD_DIVISOR: f32 = 0.67;

/// The B3-spline kernel `[1, 4, 6, 4, 1] / 16` of the à trous transform at
/// level `level` (1, 2, …): its taps `2^(level-1)` pixels apart - with
/// `2^(level-1) - 1` holes (zeros) in between (Starck & Murtagh 1994, eq.
/// 6 and step 2 of the algorithm; Olivo-Marin 2002, sect. 2).
fn b3_spline(level: u32) -> Vec<Tap> {
    let step = 1usize << (level - 1);
    vec![(0, 6.0 / 16.0), (step, 4.0 / 16.0), (2 * step, 1.0 / 16.0)]
}

/// The standard deviation of the wavelet planes `W_1..=W_levels` of white
/// noise with standard deviation 1, away from the borders: the L2 norm of
/// each plane's impulse response (the noise of a linear filter's output is
/// the input's times that norm). The impulse response of `A_i` is `a_i ⊗
/// a_i`, with `a_i` the 1-D kernels of levels 1..=i convolved, so `W_i` =
/// `a_(i-1) ⊗ a_(i-1) - a_i ⊗ a_i` and `|W_i|² = (a·a)² - 2 (a·b)² + (b·b)²`
/// with `a = a_(i-1)`, `b = a_i`. Starck & Murtagh 1994 scale the noise to
/// the planes the same way; they measure the factors on simulated noise
/// instead of computing them.
fn plane_noise(levels: u32) -> Vec<f32> {
    let radius = 2 * ((1usize << levels) - 1); // support of a_levels
    let len = 2 * radius + 1;
    let mut prev = vec![0.0f64; len];
    prev[radius] = 1.0;
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    let mut norms = Vec::with_capacity(levels as usize);
    for level in 1..=levels {
        let mut next = vec![0.0f64; len];
        for (offset, weight) in b3_spline(level) {
            let weight = weight as f64;
            for i in 0..len {
                if i >= offset {
                    next[i] += weight * prev[i - offset];
                }
                if offset > 0 && i + offset < len {
                    next[i] += weight * prev[i + offset];
                }
            }
        }
        let (aa, ab, bb) = (dot(&prev, &prev), dot(&prev, &next), dot(&next, &next));
        norms.push((aa * aa - 2.0 * ab * ab + bb * bb).max(0.0).sqrt() as f32);
        prev = next;
    }
    norms
}

/// `σ` of a wavelet plane: `MAD / 0.67` of `W = prev - next`, computed on
/// the fly (no copy of the plane); `hist` is [`RADIX_BUCKETS`] counters.
fn mad_noise(prev: &[f32], next: &[f32], hist: &mut [u32]) -> f32 {
    let plane = || prev.iter().zip(next).map(|(&p, &n)| p - n);
    median_and_mad_streamed(prev.len(), plane, hist).map_or(0.0, |(_, mad)| mad / MAD_DIVISOR)
}

/// The multiscale product of Olivo-Marin 2002 (sect. 3) of the `width ×
/// height` image `src`, into `out`:
///
/// 1. **À trous transform** (sect. 2; Starck & Murtagh 1994, sect. 3):
///    `A_0` = the image, `A_i` = `A_(i-1)` convolved with the B3-spline
///    kernel with `2^(i-1) - 1` holes, mirrored at the borders; the wavelet
///    plane `W_i = A_(i-1) - A_i` for `i = 1..=levels`. (Olivo-Marin's sect.
///    2 and appendix write `A_i - A_(i-1)`; his own `W_1 = A_0 - A_1`, his
///    reconstruction `A_0 = A_J + Σ W_i` and Starck & Murtagh's eq. 7 all
///    need `A_(i-1) - A_i`, which also makes bright spots positive.)
/// 2. **Thresholding** (sect. 3): `W_i` is kept where `W_i ≥ k σ_i`, else
///    0. [`WaveletNoise::PerLevel`]: `σ_i = MAD / 0.67`, the median absolute
///    deviation of `W_i` (the paper). [`WaveletNoise::FinestLevel`]: `σ_1`
///    so, `σ_i = σ_1 · n_i / n_1` with `n_i` from [`plane_noise`].
/// 3. **Product** `P = Π_i W̃_i` (sect. 3) over `i = first_level..=levels`
///    into `out` (`first_level` 1 in the paper); with
///    [`WaveletOutput::Intensity`] its `m`-th root, `m` the number of
///    factors. The paper's last step, `|P| ≥ l_d`, is left to Find Maxima's
///    cut-off.
///
/// `a` and `b` (approximations) are scratch, `width * height` values each.
/// The convolutions keep only a few rows for their column pass
/// ([`convolve_separable_rolling`]), the noise estimate a histogram - no
/// third plane.
fn multiscale_product(
    src: &[f32],
    out: &mut [f32],
    a: &mut Vec<f32>,
    b: &mut Vec<f32>,
    width: usize,
    height: usize,
    p: &Params,
) {
    let mut ring = Vec::new();
    let mut hist = vec![0u32; RADIX_BUCKETS];
    let noise_ratio = match p.noise {
        WaveletNoise::PerLevel => Vec::new(),
        WaveletNoise::FinestLevel => {
            let n = plane_noise(p.levels);
            n.iter().map(|&ni| ni / n[0]).collect()
        }
    };
    let mut sigma_1 = 0.0;
    out.fill(1.0);
    a.copy_from_slice(src);
    for level in 1..=p.levels {
        // b = A_i from a = A_(i-1).
        convolve_separable_rolling(a, b, &mut ring, width, height, &b3_spline(level));

        let combined = level >= p.first_level;
        let sigma = match p.noise {
            WaveletNoise::PerLevel if combined => mad_noise(a, b, &mut hist),
            WaveletNoise::PerLevel => 0.0, // not used
            WaveletNoise::FinestLevel => {
                if level == 1 {
                    sigma_1 = mad_noise(a, b, &mut hist);
                }
                sigma_1 * noise_ratio[level as usize - 1]
            }
        };
        if combined {
            let threshold = p.noise_factor * sigma;
            for ((o, &prev), &next) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                let w = prev - next;
                *o = if w >= threshold { *o * w } else { 0.0 };
            }
        }
        std::mem::swap(a, b);
    }
    if p.output == WaveletOutput::Intensity {
        let inv_m = 1.0 / (p.levels - p.first_level + 1) as f32;
        for o in out.iter_mut().filter(|o| **o > 0.0) {
            *o = o.powf(inv_m);
        }
    }
}

static OLIVO_MARIN_2002: CitationMetadata = CitationMetadata {
    cite_key: "olivomarin2002extraction",
    title: "Extraction of Spots in Biological Images Using Multiscale Products",
    authors: &["Jean-Christophe Olivo-Marin"],
    year: 2002,
    container: Some("Pattern Recognition"),
    doi: Some("10.1016/S0031-3203(01)00127-3"),
    url: Some("https://doi.org/10.1016/S0031-3203(01)00127-3"),
    pages: Some("1989-1996"),
};

static STARCK_MURTAGH_1994: CitationMetadata = CitationMetadata {
    cite_key: "starck1994image",
    title: "Image Restoration with Noise Suppression Using the Wavelet Transform",
    authors: &["Jean-Luc Starck", "Fionn Murtagh"],
    year: 1994,
    container: Some("Astronomy and Astrophysics"),
    doi: None,
    url: Some("https://ui.adsabs.harvard.edu/abs/1994A%26A...288..342S"),
    pages: Some("342-348"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
    use crate::algos::libs::scale_space::convolve_separable;
    use kornia_image::ImageSize;

    fn params(levels: u32, k: f32) -> Params {
        Params {
            levels,
            first_level: 1,
            noise_factor: k,
            noise: WaveletNoise::PerLevel,
            output: WaveletOutput::Product,
        }
    }

    fn product(src: &[f32], width: usize, height: usize, levels: u32, k: f32) -> Vec<f32> {
        run(src, width, height, &params(levels, k))
    }

    fn run(src: &[f32], width: usize, height: usize, p: &Params) -> Vec<f32> {
        let n = width * height;
        let mut out = vec![0.0; n];
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        multiscale_product(src, &mut out, &mut a, &mut b, width, height, p);
        out
    }

    /// Deterministic Gaussian noise with standard deviation `sigma` (a
    /// 64-bit LCG, Box-Muller).
    fn noise(n: usize, sigma: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f64 + 0.5) / (1u64 << 24) as f64
        };
        (0..n)
            .map(|_| {
                let (u, v) = (uniform(), uniform());
                ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32 * sigma
            })
            .collect()
    }

    /// A Gaussian spot (σ 1.5 px) of height `height` on `background`, with a
    /// little deterministic ripple so the noise estimate isn't 0.
    fn spotted(w: usize, h: usize, background: f32, height: f32) -> Vec<f32> {
        let (cx, cy) = ((w / 2) as f32, (h / 2) as f32);
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
                let ripple = 0.002 * (((i * 7919) % 13) as f32 - 6.0);
                background + ripple + height * (-(x * x + y * y) / (2.0 * 1.5 * 1.5)).exp()
            })
            .collect()
    }

    #[test]
    fn b3_spline_taps_spread_with_the_level() {
        assert_eq!(b3_spline(1), vec![(0, 0.375), (1, 0.25), (2, 0.0625)]);
        let offsets: Vec<usize> = b3_spline(3).iter().map(|t| t.0).collect();
        assert_eq!(offsets, [0, 4, 8]);
    }

    /// The planes and the last approximation add up to the image again
    /// (Olivo-Marin 2002, sect. 2; Starck & Murtagh 1994, eq. 10).
    #[test]
    fn the_planes_add_up_to_the_image() {
        let (w, h) = (23, 17);
        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37) % 101) as f32 / 100.0)
            .collect();
        let mut a = src.clone();
        let mut b = vec![0.0; w * h];
        let mut tmp = vec![0.0; w * h];
        let mut sum = vec![0.0f32; w * h];
        for level in 1..=4 {
            convolve_separable(&a, &mut b, &mut tmp, w, h, &b3_spline(level));
            for i in 0..w * h {
                sum[i] += a[i] - b[i];
            }
            std::mem::swap(&mut a, &mut b);
        }
        for i in 0..w * h {
            assert!((sum[i] + a[i] - src[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn a_flat_image_has_no_spots() {
        let out = product(&vec![0.3; 20 * 20], 20, 20, 3, 3.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_spot_survives_and_its_surroundings_do_not() {
        let (w, h) = (41, 41);
        let out = product(&spotted(w, h, 0.1, 0.5), w, h, 3, 3.0);
        let centre = out[20 * w + 20];
        assert!(centre > 0.0);
        assert!(out.iter().all(|&v| v <= centre));
        assert_eq!(out[2 * w + 2], 0.0);
    }

    #[test]
    fn a_dark_spot_is_not_enhanced() {
        // Only W ≥ kσ is kept: a dark spot gives negative planes.
        let (w, h) = (31, 31);
        assert_eq!(
            product(&spotted(w, h, 0.6, -0.5), w, h, 3, 3.0)[15 * w + 15],
            0.0
        );
    }

    #[test]
    fn the_command_writes_the_product() {
        let (w, h) = (41, 41);
        let mut ctx = PipelineContext::new_test::<F32Gray>(ImageSize {
            width: w,
            height: h,
        })
        .unwrap();
        let src = spotted(w, h, 0.1, 0.5);
        ctx.get_f32_gray_image_mut()
            .unwrap()
            .as_slice_mut()
            .copy_from_slice(&src);
        WaveletSpots {
            levels: 3,
            first_level: 1,
            noise_factor: 3.0,
            noise: WaveletNoise::PerLevel,
            output: WaveletOutput::Product,
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        assert_eq!(
            ctx.get_f32_gray_image().unwrap().as_slice(),
            &product(&src, w, h, 3, 3.0)[..]
        );
    }

    /// The computed noise of each plane is the plane's response to a delta
    /// in the real transform, and matches the factors Starck & Murtagh
    /// publish for the B3 à trous transform (0.889, 0.200, 0.086, 0.041,
    /// 0.020, measured on simulated noise).
    #[test]
    fn plane_noise_is_the_transforms_response_to_white_noise() {
        let levels = 5;
        let norms = plane_noise(levels);
        for (n, expected) in norms.iter().zip([0.889, 0.200, 0.086, 0.041, 0.020]) {
            assert!((n - expected).abs() < 0.003, "{n} vs {expected}");
        }
        let side = 2 * 2 * ((1 << levels) - 1) + 1; // no border reached
        let mut a = vec![0.0f32; side * side];
        a[side * side / 2] = 1.0;
        let (mut b, mut tmp) = (vec![0.0; side * side], vec![0.0; side * side]);
        for level in 1..=levels {
            convolve_separable(&a, &mut b, &mut tmp, side, side, &b3_spline(level));
            let energy: f32 = a.iter().zip(&b).map(|(p, q)| (p - q) * (p - q)).sum();
            assert!((energy.sqrt() - norms[level as usize - 1]).abs() < 1e-4);
            std::mem::swap(&mut a, &mut b);
        }
    }

    #[test]
    fn intensity_is_the_root_of_the_product() {
        let (w, h) = (41, 41);
        let src = spotted(w, h, 0.1, 0.5);
        let product = run(&src, w, h, &params(3, 3.0));
        let intensity = run(
            &src,
            w,
            h,
            &Params {
                output: WaveletOutput::Intensity,
                ..params(3, 3.0)
            },
        );
        for (p, i) in product.iter().zip(&intensity) {
            assert!((p.cbrt() - i).abs() <= 1e-6 * i.max(1.0));
        }
        assert!(intensity[20 * w + 20] > 0.0);
    }

    /// From `first_level` on: the product of levels 2..=3 is the product of
    /// 1..=3 without level 1's factor - so where both keep the pixel, their
    /// ratio is level 1's plane.
    #[test]
    fn first_level_skips_the_finer_levels() {
        let (w, h) = (41, 41);
        let src = spotted(w, h, 0.1, 0.5);
        let all = product(&src, w, h, 3, 3.0);
        let from_2 = run(
            &src,
            w,
            h,
            &Params {
                first_level: 2,
                ..params(3, 3.0)
            },
        );
        let only_1 = product(&src, w, h, 1, 3.0);
        for i in 0..w * h {
            if all[i] > 0.0 {
                assert!((from_2[i] * only_1[i] - all[i]).abs() <= 1e-5 * all[i]);
            }
        }
        // Fewer factors to pass: at least as many pixels survive.
        let count = |v: &[f32]| v.iter().filter(|&&x| x > 0.0).count();
        assert!(count(&from_2) >= count(&all));
        assert!(from_2[20 * w + 20] > 0.0);
    }

    /// On pure noise both estimates agree: the per-level MAD measures what
    /// the finest level predicts.
    #[test]
    fn on_noise_the_finest_level_predicts_the_others() {
        let (w, h) = (256, 256);
        let src = noise(w * h, 0.1, 7);
        let mut a = src.clone();
        let (mut b, mut tmp) = (vec![0.0; w * h], vec![0.0; w * h]);
        let norms = plane_noise(3);
        let mut sigma_1 = 0.0;
        for level in 1..=3 {
            convolve_separable(&a, &mut b, &mut tmp, w, h, &b3_spline(level));
            let measured = mad_noise(&a, &b, &mut vec![0; RADIX_BUCKETS]);
            if level == 1 {
                sigma_1 = measured;
            }
            let predicted = sigma_1 * norms[level as usize - 1] / norms[0];
            assert!((measured / predicted - 1.0).abs() < 0.1, "level {level}");
            std::mem::swap(&mut a, &mut b);
        }
    }

    /// Many bright spots (left) raise the coarse levels' MAD; the finest
    /// level's estimate keeps a dim spot away from them (right).
    #[test]
    fn finest_level_noise_keeps_a_dim_spot_among_many() {
        let (w, h) = (96, 96);
        let mut src = noise(w * h, 0.01, 3);
        let mut add_spot = |cx: usize, cy: usize, height: f32| {
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 - cx as f32, y as f32 - cy as f32);
                    src[y * w + x] += height * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp();
                }
            }
        };
        for cy in (6..h).step_by(8) {
            for cx in (6..60).step_by(8) {
                add_spot(cx, cy, 1.0);
            }
        }
        add_spot(82, 48, 0.4);
        let dim = 48 * w + 82;
        let per_level = product(&src, w, h, 3, 3.0);
        let finest = run(
            &src,
            w,
            h,
            &Params {
                noise: WaveletNoise::FinestLevel,
                ..params(3, 3.0)
            },
        );
        assert_eq!(per_level[dim], 0.0);
        assert!(finest[dim] > 0.0);
        assert!(finest[6 * w + 6] > 0.0);
    }

    #[test]
    fn small_and_single_row_images_work() {
        product(&[0.5], 1, 1, 3, 3.0);
        product(&[0.1, 0.2, 0.9, 0.2, 0.1, 0.0, 0.3], 7, 1, 3, 3.0);
        let p = Params {
            first_level: 3,
            noise: WaveletNoise::FinestLevel,
            output: WaveletOutput::Intensity,
            ..params(3, 3.0)
        };
        run(&[0.5], 1, 1, &p);
        run(&[0.1, 0.2, 0.9, 0.2, 0.1, 0.0, 0.3], 7, 1, &p);
    }
}
