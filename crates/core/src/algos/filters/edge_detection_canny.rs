//! # edge_detection_canny
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, ImageAlgorithm, GlobalPipelineCache, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_imgproc::filter::gaussian_blur;
use kornia_imgproc::filter::spatial_gradient_float;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::f32::consts::PI;
use std::sync::Arc;

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
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (input, output) = match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
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

        let (width, height) = (input.width(), input.height());
        if width == 0 || height == 0 {
            return Err(InternalErrors::Generic(format!(
                "EdgeDetectionCanny requires a non-empty image, got {width}x{height}"
            )));
        }

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
        "Edge Detection Canny"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "canny1986computational",
            title: "A Computational Approach to Edge Detection",
            authors: &["John Canny"],
            year: 1986,
            container: Some("IEEE Transactions on Pattern Analysis and Machine Intelligence"),
            doi: Some("10.1109/TPAMI.1986.4767851"),
            url: Some("https://doi.org/10.1109/TPAMI.1986.4767851"),
            pages: Some("679-698"),
        })
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
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
    if width == 0 {
        return;
    }
    let height = mag.len() / width;
    let mut stack = vec![start_idx];

    // 2D (dx, dy) offsets, not flat 1D offsets - a flat-index bounds check alone
    // can't distinguish "off the left/right edge of the image" from "on the
    // previous/next row", so it must be checked in (x, y) space.
    const NEIGHBOR_OFFSETS: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    while let Some(curr_idx) = stack.pop() {
        let cx = (curr_idx % width) as isize;
        let cy = (curr_idx / width) as isize;

        for &(dx, dy) in &NEIGHBOR_OFFSETS {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || nx >= width as isize || ny < 0 || ny >= height as isize {
                continue;
            }
            let n_idx = ny as usize * width + nx as usize;

            // If it's a weak edge and not already marked as a final edge
            if out[n_idx] == 0.0 && mag[n_idx] >= low {
                out[n_idx] = 1.0;
                stack.push(n_idx); // Follow the chain
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
        let mut cache = GlobalPipelineCache::default();

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
        if let ImageContainer::F32Gray(final_image) = ctx.image.as_ref() {
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

    #[test]
    fn zero_width_image_returns_error_instead_of_panicking() {
        let input_img = kornia_image::Image::<f32, 1, CpuAllocator>::new(
            kornia_image::ImageSize {
                width: 0,
                height: 5,
            },
            vec![],
            CpuAllocator,
        )
        .unwrap();

        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let canny = EdgeDetectionCanny {
            kernel_size: 3,
            threshold_min: 0.1,
            threshold_max: 0.3,
        };

        let result = canny.execute(&mut ctx, &mut cache);
        assert!(
            result.is_err(),
            "expected an error for a zero-width image, got Ok"
        );
    }

    #[test]
    fn zero_height_image_returns_error_instead_of_panicking() {
        let input_img = kornia_image::Image::<f32, 1, CpuAllocator>::new(
            kornia_image::ImageSize {
                width: 5,
                height: 0,
            },
            vec![],
            CpuAllocator,
        )
        .unwrap();

        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let canny = EdgeDetectionCanny {
            kernel_size: 3,
            threshold_min: 0.1,
            threshold_max: 0.3,
        };

        let result = canny.execute(&mut ctx, &mut cache);
        assert!(
            result.is_err(),
            "expected an error for a zero-height image, got Ok"
        );
    }

    #[test]
    fn check_hysteresis_does_not_link_across_the_right_edge_wrap() {
        // width=4: a flat index of (row0, x=3) + 1 lands on (row1, x=0) - a
        // real neighbor under the old flat-index-only bounds check, but not
        // spatially adjacent at all.
        let width = 4;
        let height = 3;
        let low = 0.5;
        let mut mag = vec![0.0f32; width * height];
        mag[4] = 0.9; // row 1, x = 0 - NOT adjacent to (row0, x=3)
        mag[2] = 0.9; // row 0, x = 2 - genuinely 8-adjacent to (row0, x=3)

        let mut out = vec![0.0f32; width * height];
        let start_idx = 3; // row 0, x = 3 (rightmost column)
        out[start_idx] = 1.0;

        check_hysteresis(&mag, &mut out, start_idx, width, low);

        assert_eq!(
            out[4], 0.0,
            "pixel at row1,x0 was linked from row0,x3 - hysteresis wrapped across the row border"
        );
        assert_eq!(
            out[2], 1.0,
            "genuinely adjacent weak pixel (row0,x2) should still be linked"
        );
    }

    #[test]
    fn check_hysteresis_does_not_link_across_the_left_edge_wrap() {
        // width=4: a flat index of (row1, x=0) - 1 lands on (row0, x=3) -
        // again a flat-index neighbor but not a real spatial one.
        let width = 4;
        let height = 3;
        let low = 0.5;
        let mut mag = vec![0.0f32; width * height];
        mag[3] = 0.9; // row 0, x = 3 - NOT adjacent to (row1, x=0)

        let mut out = vec![0.0f32; width * height];
        let start_idx = 4; // row 1, x = 0 (leftmost column)
        out[start_idx] = 1.0;

        check_hysteresis(&mag, &mut out, start_idx, width, low);

        assert_eq!(
            out[3], 0.0,
            "pixel at row0,x3 was linked from row1,x0 - hysteresis wrapped across the row border"
        );
    }
}
