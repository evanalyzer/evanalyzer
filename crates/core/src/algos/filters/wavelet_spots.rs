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
            robust_stats::median_and_mad_in_place,
            scale_space::{Tap, convolve_separable},
        },
    },
    image::ImageContainer,
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Enhances spots by combining several levels of detail: keeps what stands
/// out of the noise at each level from the finest to `levels`.
///
/// [Wavelet Spots] -> [Find Maxima] -> [Extract Objects]
///
/// The result is the paper's multiscale product: positive at bright spots,
/// exactly 0 everywhere else (background, dark spots, what stands out at
/// one level only). Its values are the image's intensities to the power of
/// `levels`, so a manual cut-off in Find Maxima is the paper's detection
/// level in these units. Measure intensities on the original channel, not
/// on this image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Wavelet Spots")]
pub struct WaveletSpots {
    /// How many levels are combined: level i holds details of about 2^i
    /// pixels, so 3 suits spots up to about 8 px.
    #[cmdsmeta(default = 3, min = 2, max = 5, step = 1, summary = true)]
    pub levels: usize,

    /// A level counts at a pixel only where it is at least this many times
    /// its noise; 3 in the paper.
    #[cmdsmeta(default = 3.0, min = 0.0, max = 20.0, step = 0.5)]
    pub noise_factor: f32,
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
                let (mut a, mut b, mut tmp) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                multiscale_product(
                    input.data.as_slice(),
                    output.data.as_slice_mut(),
                    &mut a,
                    &mut b,
                    &mut tmp,
                    width,
                    height,
                    self.levels.clamp(1, 16) as u32,
                    self.noise_factor,
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

    /// Two approximations and the convolution's row pass, which also holds
    /// the noise estimate's work copy. Measured by `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        12.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&OLIVO_MARIN_2002, &STARCK_MURTAGH_1994]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
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
///    0, with `σ_i = MAD / 0.67`, the median absolute deviation of `W_i`.
/// 3. **Product** `P = Π_i W̃_i` (sect. 3) into `out`. The paper's last step,
///    `|P| ≥ l_d`, is left to Find Maxima's cut-off.
///
/// `a` and `b` (approximations) and `tmp` (convolution rows and the noise
/// estimate's work buffer) are scratch, `width * height` values each.
#[allow(clippy::too_many_arguments)]
fn multiscale_product(
    src: &[f32],
    out: &mut [f32],
    a: &mut Vec<f32>,
    b: &mut Vec<f32>,
    tmp: &mut [f32],
    width: usize,
    height: usize,
    levels: u32,
    noise_factor: f32,
) {
    out.fill(1.0);
    a.copy_from_slice(src);
    for level in 1..=levels {
        // b = A_i from a = A_(i-1).
        convolve_separable(a, b, tmp, width, height, &b3_spline(level));

        // σ_i from W_i = A_(i-1) - A_i (in tmp, reordered by the median).
        for ((t, &prev), &next) in tmp.iter_mut().zip(a.iter()).zip(b.iter()) {
            *t = prev - next;
        }
        let sigma = median_and_mad_in_place(tmp).map_or(0.0, |(_, mad)| mad / MAD_DIVISOR);
        let threshold = noise_factor * sigma;

        for ((o, &prev), &next) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            let w = prev - next;
            *o = if w >= threshold { *o * w } else { 0.0 };
        }
        std::mem::swap(a, b);
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
    use kornia_image::ImageSize;

    fn product(src: &[f32], width: usize, height: usize, levels: u32, k: f32) -> Vec<f32> {
        let n = width * height;
        let mut out = vec![0.0; n];
        let (mut a, mut b, mut tmp) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        multiscale_product(
            src, &mut out, &mut a, &mut b, &mut tmp, width, height, levels, k,
        );
        out
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
            noise_factor: 3.0,
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        assert_eq!(
            ctx.get_f32_gray_image().unwrap().as_slice(),
            &product(&src, w, h, 3, 3.0)[..]
        );
    }

    #[test]
    fn small_and_single_row_images_work() {
        product(&[0.5], 1, 1, 3, 3.0);
        product(&[0.1, 0.2, 0.9, 0.2, 0.1, 0.0, 0.3], 7, 1, 3, 3.0);
    }
}
