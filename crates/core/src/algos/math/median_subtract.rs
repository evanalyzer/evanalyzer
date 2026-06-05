//! # median_subtract
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{
    ImageAlgorithm, ImageMath, PipelineCache, PipelineContext, RankFilter, RankFilterType,
};
use evanalyzer_cfg::core_types::ImageAddress;
use evanalyzer_cfg::core_types::InternalErrors;
use macros::CommandsMeta;

/// A background subtraction filter that uses a median rank operator.
///
/// This algorithm is highly effective for removing large-scale background
/// variations while preserving small, high-contrast features. It works by
/// estimating the background as the median intensity within a local radius.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::MedianSubtract;
/// let filter = MedianSubtract { radius: 10.0 };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct MedianSubtract {
    /// The radius of the neighborhood used to estimate the background.
    ///
    /// Features smaller than this radius will be preserved, while
    /// larger structures will be treated as background and removed.
    pub radius: f64,
}

impl ImageAlgorithm for MedianSubtract {
    /// Executes the Median Subtraction pipeline.
    ///
    /// The algorithm follows a three-step internal process:
    /// 1. **Snapshot**: Clones the original image into the scratchpad to save the signal.
    /// 2. **Estimation**: Applies a Median filter to the primary image to create a
    ///    background map (removing small objects/noise).
    /// 3. **Subtraction**: Calculates `(Original Signal - Background Map)` using
    ///    the scratchpad and the `ImageMath` algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors`] if the underlying `RankFilter` or `ImageMath` operations fail.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        ctx.scratch_pad = ctx.image.clone();

        let rank = RankFilter {
            radius: self.radius,
            filter_type: RankFilterType::Median,
        };
        rank.execute(ctx, cache)?;

        let mat = ImageMath {
            operand: super::image_math::Operand::Subtract,
            second_image_address: ImageAddress::Scratchpad,
            swap_operands: true,
        };
        mat.execute(ctx, cache)?;

        Ok(())
    }

    fn name(&self) -> &'static str {
        "MedianSubtract"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::{ImageContainer, PixelSizes},
        pipeline::pipeline::PipelineImageMeta,
    };
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_median_subtract_spike_isolation() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Setup a 5x5 image with a single "spike" in the center
        // Original: All 10.0, Center is 50.0
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![10.0f32; 25];
        data[12] = 50.0; // Center pixel

        let input_img = ImageContainer::new_f32_gray_from_image_test(
            Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)?,
        );

        // Prepare context
        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 5,
                    height: 5,
                },
                full_image_width: ImageSize {
                    width: 5,
                    height: 5,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            input_img,
        )?;
        let mut cache = PipelineCache::default();

        // 2. Setup Algorithm (Radius 1 = 3x3 window)
        let algo = MedianSubtract { radius: 1.0 };

        // 3. Execute
        algo.execute(&mut ctx, &mut cache)?;

        // 4. Verify Results
        // Logic:
        // Original Center = 50.0
        // Median(3x3) of Center = 10.0 (because 8 neighbors are 10.0)
        // Result = 50.0 - 10.0 = 40.0
        // Flat areas: 10.0 - 10.0 = 0.0

        if let ImageContainer::F32Gray(result_img) = ctx.image {
            let res = result_img.as_slice();

            let center_pixel = res[12];
            let corner_pixel = res[0];

            // The spike should be preserved (relative to the background)
            assert!(
                (center_pixel - 40.0).abs() < 1e-5,
                "Center should be 40.0, got {}",
                center_pixel
            );

            // The background should be zeroed out
            assert!(
                corner_pixel.abs() < 1e-5,
                "Background should be 0.0, got {}",
                corner_pixel
            );
        } else {
            panic!("Output was not F32Gray");
        }

        Ok(())
    }
    #[test]
    fn test_median_subtract_name() {
        let algo = MedianSubtract { radius: 10.0 };
        assert_eq!(algo.name(), "MedianSubtract");
    }
}
