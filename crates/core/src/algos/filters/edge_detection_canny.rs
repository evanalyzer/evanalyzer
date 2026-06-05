//! # edge_detection_canny
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
use kornia_imgproc::filter::spatial_gradient_float;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::f32::consts::PI;

/// Extracts structural boundaries and fine edges using the multi-stage Canny algorithm.
///
/// This algorithm identifies optimal edge locations by calculating spatial intensity
/// gradients, suppressing non-maximum pixel responses to thin lines down to 1-pixel width,
/// and applying a dual-threshold hysteresis loop to preserve weak edges connected
/// to strong ones while completely rejecting isolated noise.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::EdgeDetectionCanny;
/// let edges = EdgeDetectionCanny {
///     kernel_size: 3,
///     threshold_min: 0.1,
///     threshold_max: 0.3,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct EdgeDetectionCanny {
    /// Size of the Gaussian smoothing kernel.
    ///
    /// Must be an odd number (e.g., 3, 5). Larger values reduce
    /// noise but can blur fine edge details.
    pub kernel_size: usize,

    /// Lower bound for hysteresis thresholding [0.0, 1.0].
    ///
    /// Edges with a gradient intensity below this value are discarded.
    pub threshold_min: f32,

    /// Upper bound for hysteresis thresholding [0.0, 1.0].
    ///
    /// Edges with a gradient intensity above this value are considered
    /// "strong" and are automatically preserved.
    pub threshold_max: f32,
}
impl ImageAlgorithm for EdgeDetectionCanny {
    /// Detects edges in an image using the multi-stage Canny algorithm.
    ///
    /// This process involves noise reduction, finding intensity gradients,
    /// non-maximum suppression, and hysteresis thresholding.
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Gray` or `F32Rgb` (Input is usually converted to grayscale internally)
    /// * **Output:** `F32Gray` (A binary-like mask where 1.0 represents an edge)
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the image or scratch pad
    /// cannot be used for gradient calculations.
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

        // Noise Reduction
        let sigma: f32 = calculate_sigma(self.kernel_size);
        gaussian_blur(
            input,
            output,
            (self.kernel_size, self.kernel_size),
            (sigma, sigma),
        )
        .map_err(InternalErrors::from_kornia)?;

        // Gradient Calculation (Sobel)
        let mut grad_x: kornia_image::Image<f32, 1, CpuAllocator> =
            kornia_image::Image::from_size_val(input.size(), 0.0, CpuAllocator)
                .expect("Failed to allocate scratch buffer");
        let mut grad_y: kornia_image::Image<f32, 1, CpuAllocator> =
            kornia_image::Image::from_size_val(input.size(), 0.0, CpuAllocator)
                .expect("Failed to allocate scratch buffer");
        spatial_gradient_float(&output, &mut grad_x, &mut grad_y)
            .map_err(InternalErrors::from_kornia)?;

        let (width, height) = (input.width(), input.height());
        let mut magnitude = vec![0.0f32; width * height];
        let mut direction = vec![0.0f32; width * height];

        // Get slices of the underlying data
        let slice_x = grad_x.as_slice();
        let slice_y = grad_y.as_slice();
        for i in 0..width * height {
            let x = slice_x[i];
            let y = slice_y[i];
            magnitude[i] = (x * x + y * y).sqrt();
            direction[i] = y.atan2(x) * (180.0 / PI); // Convert to degrees
        }

        // Non-Maximum Suppression (NMS)
        let mut suppressed = vec![0.0f32; width * height];
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let angle = direction[idx].rem_euclid(180.0);

                // Determine neighbor offsets based on angle
                let (dx, dy) = if (0.0..22.5).contains(&angle) || (157.5..180.0).contains(&angle) {
                    (1, 0) // Horizontal
                } else if (22.5..67.5).contains(&angle) {
                    (1, 1) // Diagonal 45
                } else if (67.5..112.5).contains(&angle) {
                    (0, 1) // Vertical
                } else {
                    (-1, 1) // Diagonal 135
                };

                let mag = magnitude[idx];
                let p1 = magnitude[(y as isize + dy) as usize * width + (x as isize + dx) as usize];
                let p2 = magnitude[(y as isize - dy) as usize * width + (x as isize - dx) as usize];

                if mag >= p1 && mag >= p2 {
                    suppressed[idx] = mag;
                }
            }
        }

        // Double Thresholding & Hysteresis
        let mut final_edges = vec![0.0f32; width * height];
        for i in 0..width * height {
            if suppressed[i] >= self.threshold_max {
                final_edges[i] = 1.0; // Strong edge
                // Simple check: link neighbors (This can be optimized with a recursive stack)
                check_hysteresis(&suppressed, &mut final_edges, i, width, self.threshold_min);
            }
        }

        // Copy the pixels from final_image into the memory referenced by output
        output.as_slice_mut().copy_from_slice(&final_edges);
        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "EdgeDetectionCanny"
    }
}

/// Performs a depth-first search (DFS) to trace and connect "weak" edge pixels.
///
/// Starting from a confirmed "strong" edge, this function explores the 8-connected
/// neighborhood. Any "weak" pixel (magnitude >= `low`) connected to a strong pixel
/// is promoted to a final edge.
///
/// # Arguments
///
/// * `mag` - The gradient magnitude buffer.
/// * `out` - The output binary mask (modified in-place).
/// * `start_idx` - The 1D index of the "strong" edge to start tracing from.
/// * `width` - The width of the image for coordinate calculations.
/// * `low` - The lower hysteresis threshold.
fn check_hysteresis(mag: &[f32], out: &mut [f32], start_idx: usize, width: usize, low: f32) {
    let mut stack = vec![start_idx];
    let _height = mag.len() / width;

    let neighbors = [
        -(width as isize) - 1,
        -(width as isize),
        -(width as isize) + 1,
        -1,
        1,
        (width as isize) - 1,
        (width as isize),
        (width as isize) + 1,
    ];

    while let Some(curr_idx) = stack.pop() {
        for &offset in &neighbors {
            let n_idx_isize = curr_idx as isize + offset;

            // Bounds check
            if n_idx_isize >= 0 && n_idx_isize < mag.len() as isize {
                let n_idx = n_idx_isize as usize;

                // If it's a weak edge and not already marked as a final edge
                if out[n_idx] == 0.0 && mag[n_idx] >= low {
                    out[n_idx] = 1.0;
                    stack.push(n_idx); // Follow the chain
                }
            }
        }
    }
}

/// Calculates an optimal Gaussian sigma value based on the kernel size.
///
/// This uses the standard OpenCV heuristic to ensure the Gaussian curve
/// fits well within the chosen window size.
///
/// # Formula
/// $\sigma = 0.3 \times ((\text{kernel\_size} - 1) \times 0.5 - 1) + 0.8$
///
/// # Arguments
/// * `kernel_size` - The width/height of the square kernel. Should be an odd number.
fn calculate_sigma(kernel_size: usize) -> f32 {
    // Standard heuristic: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    0.3 * ((kernel_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pipeline_cache::ImageCache;
    // Assuming these exist based on your code
    // use crate::image::{ImageContainer, PipelineContext, CpuAllocator};

    #[test]
    fn test_canny_edge_detection_synthetic_square() {
        // 1. Setup dimensions and thresholds
        let width = 10;
        let height = 10;
        let mut data = vec![0.0f32; width * height];

        // Create a synthetic white square (4x4) in the center
        // This creates clear high-contrast edges
        for y in 3..7 {
            for x in 3..7 {
                data[y * width + x] = 1.0;
            }
        }

        let input_img = kornia_image::Image::<f32, 1, CpuAllocator>::new(
            kornia_image::ImageSize { width, height },
            data,
            CpuAllocator,
        )
        .unwrap();

        // Initialize Context
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        let mut cache = PipelineCache::default();

        // Initialize Algorithm
        let canny = EdgeDetectionCanny {
            kernel_size: 3,
            threshold_min: 0.1,
            threshold_max: 0.3,
        };

        // Execute
        let result = canny.execute(&mut ctx, &mut cache);
        assert!(result.is_ok(), "Canny execution failed: {:?}", result.err());

        // Verify Results
        // After ctx.swap(), the result is in ctx.image
        if let ImageContainer::F32Gray(final_image) = &ctx.image {
            let pixels = final_image.as_slice();

            // Check specific known edge points
            // The edge of our 3..7 square should be at index 3 and 6
            assert!(pixels[3 * width + 3] > 0.0, "Top-left corner not detected");
            assert!(
                pixels[6 * width + 6] > 0.0,
                "Bottom-right corner not detected"
            );

            // Check that the center of the square is NOT an edge (suppressed)
            assert_eq!(
                pixels[5 * width + 5],
                0.0,
                "Flat area incorrectly marked as edge"
            );

            // Check that the far corner is NOT an edge
            assert_eq!(pixels[0], 0.0, "Background incorrectly marked as edge");
        } else {
            panic!("Output image was not F32Gray");
        }
    }
}
