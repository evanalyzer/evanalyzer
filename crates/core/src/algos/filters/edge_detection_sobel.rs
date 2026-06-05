//! # edge_detection_sobel
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
use kornia_imgproc::filter::sobel;
use macros::CommandsMeta;

/// Extracts directional boundaries by computing spatial image intensity gradients.
///
/// This algorithm applies localized 3x3 kernels to approximate the first derivative
/// of pixel intensities across the horizontal and vertical axes. It highlights
/// areas of sharp luminance changes, producing a continuous gradient map that
/// emphasizes prominent structural edges and surface transitions.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::EdgeDetectionSobel;
/// let filter = EdgeDetectionSobel { kernel_size: 3 };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct EdgeDetectionSobel {
    /// The size of the Sobel operator window.
    ///
    /// Typically 3. Larger values (5, 7) provide a more smoothed
    /// gradient but result in "thicker" edges. Must be an odd number.
    pub kernel_size: usize,
}

impl ImageAlgorithm for EdgeDetectionSobel {
    /// Computes the gradient magnitude using Sobel operators.
    ///
    /// This implementation calculates the horizontal ($G_x$) and vertical ($G_y$)
    /// derivatives and combines them using the Euclidean distance: $\sqrt{G_x^2 + G_y^2}$.
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Gray`.
    /// * **Output:** `F32Gray` (The intensity represents the edge strength).
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the scratch pad is not
    /// compatible with the input dimensions or type.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let (input, output) = match (&ctx.image, &mut ctx.scratch_pad) {
            (ImageContainer::F32Gray(in_img), ImageContainer::F32Gray(out_img)) => {
                (in_img, out_img)
            }
            _ => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Gray for both input and scratch pad".into(),
                    found: format!("Input: {:?}, Scratch: {:?}", ctx.image, ctx.scratch_pad),
                });
            }
        };
        sobel(input, output, self.kernel_size).map_err(InternalErrors::from_kornia)?;
        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "EdgeDetectionSobel"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::pipeline::pipeline_cache::ImageCache;

    use super::*;
    use kornia_image::Image;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_sobel_vertical_edge() {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0.0f32; 100];

        // Create a vertical edge: Left half black (0.0), Right half white (1.0)
        for y in 0..10 {
            for x in 5..10 {
                data[y * 10 + x] = 1.0;
            }
        }

        let input_img = Image::new(size, data, CpuAllocator).unwrap();
        let sobel_algo = EdgeDetectionSobel { kernel_size: 3 };

        // Mock context
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        let mut cache = PipelineCache::default();

        sobel_algo.execute(&mut ctx, &mut cache).unwrap();

        if let ImageContainer::F32Gray(res) = ctx.scratch_pad {
            let pixels = res.as_slice();

            // The edge is at x=5.
            // The Sobel value at (5, 5) should be very high.
            let edge_pixel = pixels[5 * 10 + 5];

            // The value at (0, 0) should be 0.0 (constant black)
            let flat_pixel = pixels[0];

            assert!(
                edge_pixel > 0.5,
                "Sobel should detect the vertical edge. Got: {}",
                edge_pixel
            );
            assert_eq!(flat_pixel, 0.0, "Flat areas should have 0.0 gradient");
        }
    }

    #[test]
    fn test_sobel_format_mismatch_error() {
        // 1. Create a 5x5 RGB image (Unsupported input type for Sobel)
        let image_data = vec![0.0f32; 25 * 3];
        let input_img = Image::<f32, 3, CpuAllocator>::from_size_slice(
            ImageSize {
                width: 5,
                height: 5,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create RGB test image");

        // Initialize Context with F32Rgb
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        let mut cache = PipelineCache::default();

        // 2. Setup the command
        let sobel_algo = EdgeDetectionSobel { kernel_size: 3 };

        // 3. Execute
        let result = sobel_algo.execute(&mut ctx, &mut cache);

        // 4. Assert error
        match result {
            Err(InternalErrors::FormatMismatch { expected, found }) => {
                assert_eq!(expected, "F32Gray for both input and scratch pad");
                assert!(found.contains("Input: F32Rgb"));
            }
            _ => panic!("Expected FormatMismatch error, but got {:?}", result),
        }
    }

    #[test]
    fn test_name() {
        let extractor = EdgeDetectionSobel { kernel_size: 3 };
        let name = extractor.name();
        assert_eq!(name, "EdgeDetectionSobel");
    }
}
