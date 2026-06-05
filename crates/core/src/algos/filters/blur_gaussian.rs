//! # blur_gaussian
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_imgproc::filter::gaussian_blur;
use macros::CommandsMeta;

/// Smooths an image and reduces background noise using a Gaussian kernel.
///
/// This algorithm applies a localized, bell-curve weighted blur that suppresses
/// high-frequency pixel variations (like camera noise, salt-and-pepper artifacts,
/// or dust) while preserving structural features. It is commonly used as a
/// preprocessing step to optimize thresholding and edge detection tasks.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::GaussianBlur;
///
/// let settings = GaussianBlur {
///     kernel_size: 5,
///     sigma: 2.0
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct GaussianBlur {
    /// The size of the blur matrix.
    ///
    /// Must be an odd number (e.g., 3, 5, 7).
    #[cmdsmeta(default = 3, min = 3, max = 27, summary = true, step = 2)]
    pub kernel_size: usize,

    /// The standard deviation of the Gaussian kernel.
    ///
    /// Higher values create a more significant blur effect.
    /// $$N \approx 6\sigma + 1$$
    #[cmdsmeta(default = 0.34, min = 0.1, max = 5, summary = true, step = 0.1)]
    pub sigma: f32,
}

impl ImageAlgorithm for GaussianBlur {
    /// Applies a gaussian blur to the current image context.
    ///
    /// This implementation performs the blur in-place (via a scratch pad)
    /// and automatically swaps the buffers upon successful completion.
    ///
    /// ### Supported Image Types
    /// * `F32Gray` - Single channel floating point.
    /// * `F32Rgb` - Three channel floating point.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the image and scratch pad types
    /// do not align or are not supported by the box blur kernel.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match (&ctx.image, &mut ctx.scratch_pad) {
            // Handle Grayscale (1 Channel)
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                gaussian_blur(
                    input,
                    output,
                    (self.kernel_size, self.kernel_size),
                    (self.sigma, self.sigma),
                )
                .map_err(InternalErrors::from_kornia)?;
                ctx.swap()?;
                Ok(())
            }
            // Handle RGB (3 Channel)
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                gaussian_blur(
                    input,
                    output,
                    (self.kernel_size, self.kernel_size),
                    (self.sigma, self.sigma),
                )
                .map_err(InternalErrors::from_kornia)?;
                ctx.swap()?;
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Rgb or F32Gray".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Blur"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pipeline_cache::ImageCache;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_blur_execution() {
        // 1. Create a simple 5x5 grayscale image with a white dot in the middle
        let mut image_data = vec![0.0f32; 25];
        image_data[12] = 1.0; // Central pixel is white
        let input_img = Image::<f32, 1, CpuAllocator>::from_size_slice(
            ImageSize {
                width: 5,
                height: 5,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create test image");

        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        let mut cache = PipelineCache::default();

        // 2. Setup the command
        let blur = GaussianBlur {
            kernel_size: 3,
            sigma: 1.0,
        };

        // 3. Execute
        let result = blur.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        // 4. Verify results
        // After a blur, the center pixel (1.0) should be smaller (spread out),
        // and the surrounding pixels should no longer be 0.0.
        if let ImageContainer::F32Gray(output) = &ctx.image {
            let center_pixel = output.get_pixel(2, 2, 0).map(|&v| v).unwrap();
            let edge_pixel = output.get_pixel(0, 0, 0).map(|&v| v).unwrap();

            assert!(center_pixel < 1.0, "Center pixel should have decreased");
            assert!(center_pixel > 0.0, "Center pixel should still have value");
            assert_eq!(
                edge_pixel, 0.0,
                "Edge pixel should remain untouched by a 3x3 blur"
            );

            // Since it's a 3x3 kernel on a 5x5 image, (0,0) might still be 0,
            // but pixels adjacent to the center will definitely change.
            let neighbor_pixel = output.get_pixel(2, 1, 0).map(|&v| v).unwrap();
            assert!(
                neighbor_pixel > 0.0,
                "Neighbor pixel should have gained value"
            );
        } else {
            panic!("Output image was not F32Gray");
        }
    }

    #[test]
    fn test_rgb_blur_execution() {
        // 1. Create a 5x5 RGB image (3 channels per pixel)
        let mut image_data = vec![0.0f32; 25 * 3];
        // Set white dot at center (2,2): index = (2 * 5 + 2) * 3 = 36
        image_data[36] = 1.0; // R
        image_data[37] = 1.0; // G
        image_data[38] = 1.0; // B

        let input_img = Image::<f32, 3, CpuAllocator>::from_size_slice(
            ImageSize {
                width: 5,
                height: 5,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create RGB test image");

        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        let mut cache = PipelineCache::default();

        // 2. Setup the command
        let blur = GaussianBlur {
            kernel_size: 3,
            sigma: 1.0,
        };

        // 3. Execute
        blur.execute(&mut ctx, &mut cache).expect("RGB blur failed");

        // 4. Verify results
        if let ImageContainer::F32Rgb(output) = &ctx.image {
            // Check center pixel reduction
            let r_center = output.get_pixel(2, 2, 0).unwrap();
            assert!(*r_center < 1.0, "Center red channel should have decreased");

            // Check if energy spread to a neighbor (2, 1)
            let r_neighbor = output.get_pixel(2, 1, 0).unwrap();
            let g_neighbor = output.get_pixel(2, 1, 1).unwrap();
            let b_neighbor = output.get_pixel(2, 1, 2).unwrap();

            assert!(*r_neighbor > 0.0, "Neighbor red should have gained value");
            assert!(*g_neighbor > 0.0, "Neighbor green should have gained value");
            assert!(*b_neighbor > 0.0, "Neighbor blue should have gained value");
        } else {
            panic!("Output image was not F32Rgb");
        }
    }

    #[test]
    fn test_name() {
        let blur = GaussianBlur {
            kernel_size: 3,
            sigma: 1.0,
        };
        let name = blur.name();
        assert_eq!(name, "Blur");
    }

    #[test]
    fn test_blur_format_mismatch_error() {
        // Create a 5x5 U32 image (Unsupported type for Gaussian Blur)
        let image_data = vec![0u32; 25];
        let input_img = Image::<u32, 1, CpuAllocator>::from_size_slice(
            ImageSize {
                width: 5,
                height: 5,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create U32 test image");

        // Assuming you have a way to inject a U32 image into the context
        // This relies on your specific PipelineContext API for U32 images
        let mut ctx = PipelineContext::new_from_u32_image_test(input_img).unwrap();
        let mut cache = PipelineCache::default();

        // 2. Setup the command
        let blur = GaussianBlur {
            kernel_size: 3,
            sigma: 1.0,
        };

        // 3. Execute
        let result = blur.execute(&mut ctx, &mut cache);

        // 4. Assert error
        match result {
            Err(InternalErrors::FormatMismatch { expected, found }) => {
                assert!(expected.contains("F32Rgb or F32Gray"));
                assert!(found.contains("U32"));
            }
            _ => panic!("Expected FormatMismatch error, but got {:?}", result),
        }
    }
}
