//! # weighted_deviation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::{
    algos::ImageAlgorithm,
    image::ImageContainer,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_imgproc::filter::gaussian_blur;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// A filter that computes the Gaussian-weighted standard deviation of a local neighborhood.
///
/// Unlike a standard deviation filter which treats all pixels in a window equally,
/// the Weighted Deviation uses a Gaussian kernel to give more importance to
/// pixels closer to the center. This is particularly effective for edge-preserving
/// noise analysis and local contrast enhancement.
///
/// This algorithm evaluates local variance by calculating two distinct Gaussian-blurred
/// baselines across the image: the weighted average of the pixel intensities, and the
/// weighted average of the squared intensities. By subtracting the squared mean from
/// the mean of squares, it yields a localized, smooth statistical variance map that
/// highlights micro-textures and subtle surface boundaries without producing blocky artifacts.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::WeightedDeviation;
/// let settings = WeightedDeviation {
///     kernel_size: 7,
///     sigma: 2.0,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct WeightedDeviation {
    /// The size of the local neighborhood window.
    ///
    /// Must be an odd number. Larger windows capture broader texture
    /// variations but increase computational load.
    pub kernel_size: usize,

    /// The standard deviation for the Gaussian weighting function.
    ///
    /// Defines the "softness" of the neighborhood boundaries. A larger
    /// sigma includes more of the surrounding context in the deviation calculation.
    pub sigma: f32,
}

impl ImageAlgorithm for WeightedDeviation {
    /// Computes the local Gaussian-weighted standard deviation for each pixel.
    ///
    /// This algorithm effectively measures local contrast and texture "busyness"
    /// by calculating how much pixel intensities deviate from the local weighted
    /// mean.
    ///
    /// # Mathematical Approach
    /// The weighted variance $\sigma_w^2$ is calculated using the identity:
    /// $\sigma_w^2 = E[X^2]_w - (E[X]_w)^2$
    /// 1. Compute the Gaussian-weighted average of the image ($E[X]_w$).
    /// 2. Compute the Gaussian-weighted average of the squared image ($E[X^2]_w$).
    /// 3. The result is the square root of the difference.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::FormatMismatch`] if the input or scratch pad
    /// are not in the expected `F32Gray` format.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        // Get input image as F32Gray (assuming conversion is handled upstream)
        // If your input is F32Rgb, you'll need a conversion step here first.
        let size = match &ctx.image {
            ImageContainer::F32Gray(img) => img.size(),
            _ => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Gray for both input and scratch pad".into(),
                    found: format!("Input: {:?}, Scratch: {:?}", ctx.image, ctx.scratch_pad),
                });
            }
        };

        // Prepare meanSq (E[X^2])
        // We calculate grayF * grayF into a temporary image
        let mut mean_sq = if let ImageContainer::F32Gray(input) = &ctx.image {
            let mut ms = Image::<f32, 1, CpuAllocator>::new(
                size,
                vec![0.0; size.width * size.height],
                CpuAllocator,
            )
            .map_err(InternalErrors::from_kornia)?;
            ms.as_slice_mut()
                .par_iter_mut()
                .zip(input.as_slice().par_iter())
                .for_each(|(out, &src)| {
                    *out = src * src;
                });
            ms
        } else {
            unreachable!()
        };

        let k_size = (self.kernel_size, self.kernel_size);
        let sigma = (self.sigma, self.sigma);

        // Compute E[X^2] using the scratchpad
        // We blur the squared image
        if let ImageContainer::F32Gray(ref mut scratch) = ctx.scratch_pad {
            gaussian_blur(&mean_sq, scratch, k_size, sigma).map_err(InternalErrors::from_kornia)?;
            std::mem::swap(&mut mean_sq, scratch);
        }

        // Compute E[X] (mean)
        // Now we need to blur the original image.
        // We'll use the scratchpad for this and store the result in 'mean'
        let mut mean = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        if let (ImageContainer::F32Gray(input), ImageContainer::F32Gray(scratch)) =
            (&ctx.image, &mut ctx.scratch_pad)
        {
            gaussian_blur(input, scratch, k_size, sigma).map_err(InternalErrors::from_kornia)?;
            // Move blurred result from scratch to our 'mean' variable
            std::mem::swap(&mut mean, scratch);
        }

        // Final Calculation: sqrt(mean_sq - mean * mean)
        // We write the final result back into the scratch_pad
        if let ImageContainer::F32Gray(ref mut output) = ctx.scratch_pad {
            let s_mean_sq = mean_sq.as_slice();
            let s_mean = mean.as_slice();

            output
                .as_slice_mut()
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    let m = s_mean[i];
                    let ms = s_mean_sq[i];
                    // Variance = E[X^2] - (E[X])^2
                    // We use .max(0.0) to prevent tiny precision errors from causing sqrt(negative)
                    *val = (ms - (m * m)).max(0.0).sqrt();
                });
        }

        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "WeightedDeviation"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_weighted_deviation_edge() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0.0f32; 100];

        // 1. Create a vertical step edge (0.0 on left, 10.0 on right)
        for y in 0..10 {
            for x in 5..10 {
                data[y * 10 + x] = 10.0;
            }
        }

        let input_img = Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)?;

        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        // 2. Setup Algorithm: Small kernel to isolate the edge

        let algo = WeightedDeviation {
            kernel_size: 3,
            sigma: 1.0,
        };

        // 3. Execute
        let mut cache = PipelineCache::default();
        algo.execute(&mut ctx, &mut cache)?;

        // 4. Verification
        if let ImageContainer::F32Gray(result) = ctx.image {
            let res_slice = result.as_slice();

            // At (5,5), which is right on the edge, the deviation should be high
            let edge_dev = res_slice[5 * 10 + 5];

            // At (1,1), which is a flat area (all 0.0), the deviation should be 0.0
            let flat_dev = res_slice[1 * 10 + 1];

            assert!(
                edge_dev > 2.0,
                "Edge deviation should be significant, got {}",
                edge_dev
            );
            assert!(
                flat_dev < 0.01,
                "Flat area deviation should be near zero, got {}",
                flat_dev
            );

            // Ensure no NaNs were produced (common if variance calculation goes slightly negative)
            assert!(!edge_dev.is_nan(), "Result contains NaN");
        } else {
            panic!("Output image was not F32Gray");
        }

        Ok(())
    }
}
