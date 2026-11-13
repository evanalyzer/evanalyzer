//! # difference_of_gaussians
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! The difference of two Gaussians 1.6 apart (Marr & Hildreth 1980,
//! appendix B): a Gaussian band-pass for spots that approximates the
//! scale-normalized Laplacian of Gaussian (Lowe 2004, sect. 3), in the same
//! units as [`LaplacianOfGaussian`](crate::algos::LaplacianOfGaussian).

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SizeUnits};
use macros::CommandsMeta;
use std::sync::Arc;

use crate::{
    algos::{
        ExecutionScope, ImageAlgorithm,
        libs::scale_space::{
            LINDEBERG_1998, MARR_HILDRETH_1980, convolve_separable, discrete_gaussian,
            scale_of_sigma,
        },
    },
    image::ImageContainer,
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Enhances spots of one size with a band-pass: a slightly blurred minus a
/// more blurred copy of the image keeps the details of a spot's size and
/// removes finer noise and coarser background and structures.
///
/// [Difference of Gaussians] -> [Find Maxima] -> [Extract Objects]
///
/// Bright spots become positive peaks, dark spots negative ones, flat
/// background 0. Measure intensities on the original channel, not on this
/// image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Difference of Gaussians")]
pub struct DifferenceOfGaussians {
    /// The scale σ of the spots to enhance (the narrower Gaussian; the
    /// wider one is 1.6 σ); as a guide, σ ≈
    /// spot radius / √2 (1.4 for spots of 4 px). Smaller details count as
    /// noise, larger ones as background.
    #[cmdsmeta(default = 1.5, min = 0.1, max = 1000.0, step = 0.1, summary = true)]
    pub sigma: f32,

    /// Unit `sigma` is expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,
}

/// The ratio of the two Gaussians: σ_i / σ_e ≈ 1.6, the narrowest band-pass
/// that still keeps the filter sensitive (Marr & Hildreth 1980, appendix B).
const RATIO: f64 = 1.6;

impl ImageAlgorithm for DifferenceOfGaussians {
    /// `(L(t) - L(1.6² t)) / 0.6`, `L` = the image convolved with the
    /// discrete Gaussian (Lindeberg 1998, appendix A.4.1) at the spot's
    /// `σ` and at 1.6 σ. Lowe 2004 (sect. 3)
    /// shows `G(kσ) - G(σ) ≈ (k - 1) σ² ∇²G`, so dividing by `k - 1` gives
    /// the units of the scale-normalized Laplacian of Gaussian, negated so
    /// bright spots are positive.
    ///
    /// Works on the tile with mirrored borders, like Gaussian Blur.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let pixel_size_nm = {
            let sizes = ctx.pixel_sizes();
            (sizes.px_size_x + sizes.px_size_y) / 2.0
        };
        let t = scale_of_sigma(self.sigma, self.unit, pixel_size_nm);
        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                let (width, height) = (input.width(), input.height());
                difference_of_gaussians(
                    input.data.as_slice(),
                    output.data.as_slice_mut(),
                    width,
                    height,
                    t,
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
        "Difference of Gaussians"
    }

    /// The wider blur and the convolution's row pass (the narrower one goes
    /// straight into the output). Measured by `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&MARR_HILDRETH_1980, &LOWE_2004, &LINDEBERG_1998]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// `(L(t) - L(1.6² t)) / 0.6` of `src` into `out` (see
/// [`DifferenceOfGaussians::execute`]).
fn difference_of_gaussians(src: &[f32], out: &mut [f32], width: usize, height: usize, t: f64) {
    let n = width * height;
    let (mut wide, mut tmp) = (vec![0.0; n], vec![0.0; n]);
    convolve_separable(src, out, &mut tmp, width, height, &discrete_gaussian(t));
    let wide_t = RATIO * RATIO * t;
    convolve_separable(
        src,
        &mut wide,
        &mut tmp,
        width,
        height,
        &discrete_gaussian(wide_t),
    );
    let scale = 1.0 / (RATIO - 1.0) as f32;
    for (o, &w) in out.iter_mut().zip(&wide) {
        *o = (*o - w) * scale;
    }
}

static LOWE_2004: CitationMetadata = CitationMetadata {
    cite_key: "lowe2004distinctive",
    title: "Distinctive Image Features from Scale-Invariant Keypoints",
    authors: &["David G. Lowe"],
    year: 2004,
    container: Some("International Journal of Computer Vision"),
    doi: Some("10.1023/B:VISI.0000029664.99615.94"),
    url: Some("https://doi.org/10.1023/B:VISI.0000029664.99615.94"),
    pages: Some("91-110"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
    use crate::algos::LaplacianOfGaussian;
    use kornia_image::ImageSize;

    fn context(width: usize, height: usize, f: impl Fn(f32, f32) -> f32) -> PipelineContext {
        let mut ctx = PipelineContext::new_test::<F32Gray>(ImageSize { width, height }).unwrap();
        let px = ctx.get_f32_gray_image_mut().unwrap().as_slice_mut();
        for y in 0..height {
            for x in 0..width {
                px[y * width + x] = f(x as f32, y as f32);
            }
        }
        ctx
    }

    fn run(sigma: f32, ctx: &mut PipelineContext) -> Vec<f32> {
        DifferenceOfGaussians {
            sigma,
            unit: SizeUnits::Pixels,
        }
        .execute(ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        ctx.get_f32_gray_image().unwrap().as_slice().to_vec()
    }

    /// A unit-height Gaussian spot of standard deviation `s` at (cx, cy).
    fn spot(cx: f32, cy: f32, s: f32) -> impl Fn(f32, f32) -> f32 {
        move |x, y| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp()
    }

    #[test]
    fn a_flat_image_gives_zero() {
        let out = run(1.5, &mut context(30, 20, |_, _| 0.4));
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn a_bright_spot_is_a_positive_peak_a_dark_one_negative() {
        let out = run(1.5, &mut context(41, 41, spot(20.0, 20.0, 1.5)));
        let centre = out[20 * 41 + 20];
        assert!(centre > 0.0);
        assert!(out.iter().all(|&v| v <= centre));
        let dark = run(
            1.5,
            &mut context(41, 41, |x, y| 1.0 - spot(20.0, 20.0, 1.5)(x, y)),
        );
        assert!((dark[20 * 41 + 20] + centre).abs() < 1e-5);
    }

    /// Lowe 2004: `(G(kσ) - G(σ)) / (k - 1) ≈ σ² ∇²G` - the same units as
    /// the scale-normalized Laplacian of Gaussian at the same σ.
    #[test]
    fn it_approximates_the_laplacian_of_gaussian_in_the_same_units() {
        let image = || context(61, 61, spot(30.0, 30.0, 2.0));
        let d = run(2.0, &mut image())[30 * 61 + 30];
        let mut ctx = image();
        LaplacianOfGaussian {
            sigma: 2.0,
            unit: SizeUnits::Pixels,
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap();
        let l = ctx.get_f32_gray_image().unwrap().as_slice()[30 * 61 + 30];
        // A forward difference over σ…1.6σ: the same order, within ~30 %.
        assert!(d > 0.0 && (d / l - 1.0).abs() < 0.3, "log {l} dog {d}");
    }

    #[test]
    fn small_and_single_row_images_work() {
        run(1.5, &mut context(1, 1, |_, _| 0.5));
        run(1.5, &mut context(7, 1, |x, _| x / 7.0));
    }
}
