//! # bandpass
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! A wide Gaussian band-pass `G(σ_small) - G(σ_large)`: removes details
//! finer than σ_small (noise) and structures coarser than σ_large
//! (background), keeps everything in between in the image's intensity
//! units. Unlike the spot filters
//! (Laplacian of Gaussian, Difference of Gaussians) not tuned to one size:
//! objects of different sizes stay.
//!
//! A Gaussian low-pass times a Gaussian high-pass in the frequency domain,
//! `e^(-σ_s² ω²/2) (1 - e^(-σ_h² ω²/2))`, is the difference of two Gaussians
//! in the image, so no Fourier transform is needed: two blurs with the
//! discrete Gaussian (Lindeberg 1998, appendix A.4.1) and a subtraction, the
//! band-pass the difference of Gaussians is in Marr & Hildreth 1980
//! (appendix B) - here with two independent σ instead of their ratio 1.6.

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

/// Removes details finer than `sigma_small` (noise) and structures coarser
/// than `sigma_large` (uneven background), keeps everything in between.
///
/// [Bandpass] -> [Threshold | Find Maxima]
///
/// The result is in the image's intensity units around 0: bright objects
/// positive, the removed background 0. With `sigma_large` 0 it is a plain
/// blur, with `sigma_small` 0 a background subtraction.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Bandpass")]
pub struct Bandpass {
    /// σ of the narrower Gaussian: details finer than about this are
    /// smoothed away (noise). 0: no smoothing.
    #[cmdsmeta(default = 1.0, min = 0.0, max = 10000.0, step = 0.1, summary = true)]
    pub sigma_small: f32,

    /// σ of the wider Gaussian, subtracted: structures coarser than about
    /// this are removed (uneven background). Must be larger than
    /// `sigma_small`; 0: nothing removed.
    #[cmdsmeta(default = 15.0, min = 0.0, max = 10000.0, step = 0.5, summary = true)]
    pub sigma_large: f32,

    /// Unit both σ are expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,
}

impl ImageAlgorithm for Bandpass {
    /// `L(t_s) - L(t_l)`: the image convolved with the discrete Gaussian
    /// (Lindeberg 1998, appendix A.4.1) at `t_s = σ_small²` minus at `t_l =
    /// σ_large²`. `L(t_l)` is left out for `sigma_large` 0; `t_s = 0` is the
    /// image itself.
    ///
    /// Works on the tile with mirrored borders, like Gaussian Blur.
    ///
    /// # Errors
    /// [`InternalErrors::InvalidArgument`] if `sigma_large` isn't 0 and not
    /// larger than `sigma_small` - the band would be empty.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.sigma_large > 0.0 && self.sigma_large <= self.sigma_small {
            return Err(InternalErrors::InvalidArgument(format!(
                "Bandpass: sigma large ({}) must be larger than sigma small ({})",
                self.sigma_large, self.sigma_small
            )));
        }
        let pixel_size_nm = {
            let sizes = ctx.pixel_sizes();
            (sizes.px_size_x + sizes.px_size_y) / 2.0
        };
        let small = scale_of_sigma(self.sigma_small, self.unit, pixel_size_nm);
        let large = (self.sigma_large > 0.0)
            .then(|| scale_of_sigma(self.sigma_large, self.unit, pixel_size_nm));
        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                let (width, height) = (input.width(), input.height());
                bandpass(
                    input.data.as_slice(),
                    output.data.as_slice_mut(),
                    width,
                    height,
                    small,
                    large,
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
        "Bandpass"
    }

    /// The wider blur and the convolution's row pass (the narrower one goes
    /// straight into the output). Measured by `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&LINDEBERG_1998, &MARR_HILDRETH_1980]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// `L(small) - L(large)` of `src` into `out`; without `large` only
/// `L(small)` (see [`Bandpass::execute`]).
fn bandpass(
    src: &[f32],
    out: &mut [f32],
    width: usize,
    height: usize,
    small: f64,
    large: Option<f64>,
) {
    let n = width * height;
    let mut tmp = vec![0.0; n];
    if small > 0.0 {
        convolve_separable(src, out, &mut tmp, width, height, &discrete_gaussian(small));
    } else {
        out.copy_from_slice(src);
    }
    if let Some(large) = large {
        let mut wide = vec![0.0; n];
        convolve_separable(
            src,
            &mut wide,
            &mut tmp,
            width,
            height,
            &discrete_gaussian(large),
        );
        for (o, &w) in out.iter_mut().zip(&wide) {
            *o -= w;
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

    fn command(smaller: f32, larger: f32) -> Bandpass {
        Bandpass {
            sigma_small: smaller,
            sigma_large: larger,
            unit: SizeUnits::Pixels,
        }
    }

    fn run(command: Bandpass, ctx: &mut PipelineContext) -> Vec<f32> {
        command
            .execute(ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        ctx.get_f32_gray_image().unwrap().as_slice().to_vec()
    }

    /// A unit-height Gaussian spot of standard deviation `s` at (cx, cy).
    fn spot(cx: f32, cy: f32, s: f32) -> impl Fn(f32, f32) -> f32 {
        move |x, y| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp()
    }

    #[test]
    fn a_flat_background_becomes_zero() {
        let out = run(command(1.0, 14.0), &mut context(60, 50, |_, _| 0.4));
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn a_slow_gradient_is_removed_and_a_spot_kept() {
        // Background 0.2 → 0.5 across the image plus a spot of height 0.5.
        let image = |x: f32, y: f32| 0.2 + 0.3 * x / 199.0 + 0.5 * spot(100.0, 100.0, 1.5)(x, y);
        let out = run(command(1.0, 14.0), &mut context(200, 200, image));
        // The spot keeps most of its height, the background near it ≈ 0.
        assert!(out[100 * 200 + 100] > 0.3, "spot {}", out[100 * 200 + 100]);
        assert!(
            out[100 * 200 + 160].abs() < 0.01,
            "background {}",
            out[100 * 200 + 160]
        );
    }

    #[test]
    fn objects_of_different_sizes_stay() {
        // A small (σ 1.5) and a large (σ 5) spot: both survive a wide band,
        // unlike with a spot filter tuned to one of them.
        let image = |x: f32, y: f32| spot(40.0, 50.0, 1.5)(x, y) + spot(120.0, 50.0, 5.0)(x, y);
        let out = run(command(0.7, 21.0), &mut context(160, 100, image));
        assert!(out[50 * 160 + 40] > 0.5);
        assert!(out[50 * 160 + 120] > 0.5);
    }

    #[test]
    fn zero_sizes_switch_a_side_off() {
        let image = |x: f32, y: f32| 0.1 + spot(20.0, 20.0, 2.0)(x, y);
        // Nothing removed above: a plain blur keeps the background level.
        let blurred = run(command(1.0, 0.0), &mut context(41, 41, image));
        assert!((blurred[2 * 41 + 2] - 0.1).abs() < 1e-3);
        // Nothing smoothed: the image minus its background, exact at a flat
        // spot-free part.
        let subtracted = run(command(0.0, 14.0), &mut context(41, 41, |_, _| 0.3));
        assert!(subtracted.iter().all(|v| v.abs() < 1e-6));
        // Both 0: the image itself.
        let same = run(command(0.0, 0.0), &mut context(9, 9, image));
        assert_eq!(same[4 * 9 + 4], image(4.0, 4.0));
    }

    #[test]
    fn an_empty_band_is_an_error() {
        let mut ctx = context(10, 10, |_, _| 0.0);
        assert!(
            command(5.0, 2.0)
                .execute(&mut ctx, &mut GlobalPipelineCache::default())
                .is_err()
        );
    }

    #[test]
    fn small_and_single_row_images_work() {
        run(command(1.0, 14.0), &mut context(1, 1, |_, _| 0.5));
        run(command(1.0, 14.0), &mut context(7, 1, |x, _| x / 7.0));
    }
}
