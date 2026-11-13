//! # laplacian_of_gaussian
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! The scale-normalized Laplacian of Gaussian of Lindeberg 1998 at the scale
//! of a spot: a band-pass that keeps what has the size of a spot and removes
//! finer noise and coarser background and structures - the standard spot
//! filter before [`FindMaxima`](crate::algos::FindMaxima).

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SizeUnits};
use macros::CommandsMeta;
use std::sync::Arc;

use crate::{
    algos::{
        ExecutionScope, ImageAlgorithm,
        libs::scale_space::{
            LINDEBERG_1998, convolve_separable, discrete_gaussian, scale_of_sigma,
        },
    },
    image::ImageContainer,
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Enhances spots of one size: keeps the details of a spot's size and
/// removes finer noise and coarser background and structures.
///
/// [Laplacian of Gaussian] -> [Find Maxima] -> [Extract Objects]
///
/// Bright spots become positive peaks, dark spots negative ones, flat and
/// evenly sloped background 0. Measure intensities on the original channel,
/// not on this image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Laplacian of Gaussian")]
pub struct LaplacianOfGaussian {
    /// The scale σ of the spots to enhance (Lindeberg's `t = σ²`). A
    /// Gaussian spot of standard deviation σ responds most; as a guide, σ ≈
    /// spot radius / √2 (1.4 for spots of 4 px). Smaller details count as
    /// noise, larger ones as background.
    #[cmdsmeta(default = 1.5, min = 0.1, max = 1000.0, step = 0.1, summary = true)]
    pub sigma: f32,

    /// Unit `sigma` is expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,
}

impl ImageAlgorithm for LaplacianOfGaussian {
    /// `L` = the image convolved with the discrete Gaussian of variance `t`
    /// (Lindeberg 1998, appendix A.4.1), `Lxx + Lyy` by the differences
    /// `L(x+1) - 2L(x) + L(x-1)` (same appendix), the output `-t (Lxx +
    /// Lyy)`: the scale-normalized Laplacian (γ = 1, sect. 5) at the spot's
    /// scale `t = σ²`, negated so bright spots are positive.
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
                laplacian_of_gaussian(
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
        "Laplacian of Gaussian"
    }

    /// The blurred image and the convolution's row pass. Measured by
    /// `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&LINDEBERG_1998]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// `-t (Lxx + Lyy)` of `src` at scale `t` into `out` (see
/// [`LaplacianOfGaussian::execute`]).
fn laplacian_of_gaussian(src: &[f32], out: &mut [f32], width: usize, height: usize, t: f64) {
    let n = width * height;
    let mut blurred = vec![0.0; n];
    {
        let mut tmp = vec![0.0; n];
        convolve_separable(
            src,
            &mut blurred,
            &mut tmp,
            width,
            height,
            &discrete_gaussian(t),
        );
    }
    let t = t as f32;
    // Mirrored neighbours (the border pixel not repeated, as in the
    // convolution): -1 → 1, len → len - 2; a side of 1 has no neighbour.
    let neighbours = |i: usize, len: usize| -> (usize, usize) {
        if len == 1 {
            return (0, 0);
        }
        let before = if i == 0 { 1 } else { i - 1 };
        let after = if i + 1 == len { len - 2 } else { i + 1 };
        (before, after)
    };
    for y in 0..height {
        let (up, down) = neighbours(y, height);
        for x in 0..width {
            let (left, right) = neighbours(x, width);
            let c = blurred[y * width + x];
            let lxx = blurred[y * width + left] - 2.0 * c + blurred[y * width + right];
            let lyy = blurred[up * width + x] - 2.0 * c + blurred[down * width + x];
            out[y * width + x] = -t * (lxx + lyy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
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
        LaplacianOfGaussian {
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
    fn flat_and_sloped_images_give_zero() {
        // The Laplacian of a plane is 0; mirrored borders keep a flat image
        // flat (a slope bends at the border, so only the interior counts).
        let out = run(1.5, &mut context(30, 20, |_, _| 0.4));
        assert!(out.iter().all(|v| v.abs() < 1e-6));
        let out = run(1.5, &mut context(60, 60, |x, y| 0.001 * x + 0.002 * y));
        assert!(out[30 * 60 + 30].abs() < 1e-5);
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

    /// Lindeberg 1998, eq. 36: for a Gaussian spot of variance t₀ the
    /// scale-normalized Laplacian at its centre is largest at t = t₀ - over
    /// σ the response peaks at the spot's own σ.
    #[test]
    fn the_response_is_largest_at_the_spot_sigma() {
        let s = 2.0f32;
        let centre =
            |sigma: f32| run(sigma, &mut context(61, 61, spot(30.0, 30.0, s)))[30 * 61 + 30];
        let best = (10..=40)
            .map(|i| i as f32 * 0.1)
            .max_by(|&a, &b| centre(a).total_cmp(&centre(b)))
            .unwrap();
        assert!((best - s).abs() <= 0.15, "best sigma {best}");
        // The continuous value at t = t₀ (eq. 35 with t₁ = t₂ = t₀, for a
        // unit-height spot): t (t₁ + t₂ + 2t) / (2π (t₀ + t)³) × 2π t₀ = 1/2.
        let at_t0 = centre(s);
        assert!((at_t0 - 0.5).abs() < 0.02, "response {at_t0}");
    }

    #[test]
    fn small_and_single_row_images_work() {
        run(1.5, &mut context(1, 1, |_, _| 0.5));
        run(1.5, &mut context(7, 1, |x, _| x / 7.0));
    }
}
