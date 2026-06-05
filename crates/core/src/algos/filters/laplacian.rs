//! # laplacian
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::image::ImageContainer;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::{algos::ImageAlgorithm, pipeline::pipeline_cache::PipelineCache};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use ndarray::{ArrayView3, ArrayViewMut3};

/// Configuration for the Laplacian edge detection filter.
///
/// The Laplacian is a second-order derivative operator used to find regions of
/// rapid intensity change. It is particularly effective for detecting edges
/// and fine details, though it is highly sensitive to noise.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::Laplacian;
/// let filter = Laplacian { kernel_size: 3 };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct Laplacian {
    /// The size of the discrete Laplacian aperture.
    ///
    /// Typically 3. Larger sizes (5, 7) approximate the Laplacian of Gaussian (LoG)
    /// more closely but are more computationally expensive. Must be an odd number.
    pub kernel_size: usize,
}
impl ImageAlgorithm for Laplacian {
    /// Executes the Laplacian edge detection filter.
    ///
    /// This implementation performs a discrete convolution to approximate the Laplacian
    /// operator: $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$.
    ///
    /// Unlike first-order filters (like Sobel), the Laplacian is isotropic and
    /// captures intensity changes in all directions equally.
    ///
    /// ### Implementation Details
    /// * **Scratch Pad:** Requires a pre-allocated scratch pad of the same dimensions
    ///   and type as the input to store the convolution results.
    /// * **Normalization:** Output values may contain negative numbers or values
    ///   exceeding [0.0, 1.0] depending on the kernel weights; ensure the `apply`
    ///   logic handles normalization or clamping.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input image and
    /// `scratch_pad` are not of the same type (both `F32Gray` or both `F32Rgb`).
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match (&ctx.image, &mut ctx.scratch_pad) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                apply_laplacian(input, output, self.kernel_size)?;
                Ok(())
            }
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                apply_laplacian(input, output, self.kernel_size)?;
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray or F32Rgb".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Laplacian"
    }
}

fn get_1d_kernel(kernel_size: usize) -> Vec<f32> {
    match kernel_size {
        1 | 3 => vec![1.0, -2.0, 1.0],
        5 => vec![1.0, 2.0, -6.0, 2.0, 1.0],
        7 => vec![1.0, 4.0, 1.0, -12.0, 1.0, 4.0, 1.0],
        _ => vec![1.0, -2.0, 1.0], // Default fallback
    }
}

fn apply_laplacian<const C: usize>(
    input: &Image<f32, C, CpuAllocator>,
    output: &mut Image<f32, C, CpuAllocator>,
    kernel_size: usize,
) -> Result<(), InternalErrors> {
    let (h, w) = (input.size().height, input.size().width);

    // Standard OpenCV-style kernel size mapping
    let k = if kernel_size == 1 { 3 } else { kernel_size };
    let pad = k / 2;

    if h < k || w < k {
        return Ok(());
    }

    // Map views with error handling
    let in_view = ArrayView3::from_shape((h, w, C), input.as_slice())?;
    let mut out_view = ArrayViewMut3::from_shape((h, w, C), output.as_slice_mut())?;

    if k == 3 {
        // If k=3, we use the hardcoded stencil which is extremely fast.
        let center = in_view.slice(ndarray::s![1..h - 1, 1..w - 1, ..]);
        let top = in_view.slice(ndarray::s![0..h - 2, 1..w - 1, ..]);
        let bottom = in_view.slice(ndarray::s![2..h, 1..w - 1, ..]);
        let left = in_view.slice(ndarray::s![1..h - 1, 0..w - 2, ..]);
        let right = in_view.slice(ndarray::s![1..h - 1, 2..w, ..]);

        let mut out_slice = out_view.slice_mut(ndarray::s![1..h - 1, 1..w - 1, ..]);

        // This single line triggers SIMD instructions (vectorization)
        out_slice.assign(&(&top + &bottom + &left + &right - (4.0 * &center)));
    } else {
        //  Parallel ZIP (For larger kernels)
        let mut out_slice = out_view.slice_mut(ndarray::s![pad..h - pad, pad..w - pad, ..]);
        let kernel_1d = get_1d_kernel(kernel_size);
        let center_idx = k / 2;

        // Using ndarray::Zip without .indexed() if possible, as it's faster
        ndarray::Zip::indexed(&mut out_slice).for_each(|(y_inner, x_inner, c), out_pixel| {
            let mut val = 0.0;
            let yc = y_inner + pad;
            let xc = x_inner + pad;

            // Outside the loop, handle the center pixel contribution exactly once
            val += in_view[[yc, xc, c]] * kernel_1d[center_idx];

            for i in 0..k {
                if i == center_idx {
                    continue;
                } // Already added
                let offset = i as isize - center_idx as isize;
                val += in_view[[yc, (xc as isize + offset) as usize, c]] * kernel_1d[i]; // Horizontal
                val += in_view[[(yc as isize + offset) as usize, xc, c]] * kernel_1d[i]; // Vertical
            }
            *out_pixel = val;
        });
    }

    Ok(())
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn test_laplacian_f32_gray() {
        // 1. Create a 5x5 grayscale image initialized to 0.0
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut input =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let mut output =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();

        // 2. Set the center pixel to 1.0 (an "impulse")
        // Kornia layout is HWC, so index is [row, col, channel]
        {
            let mut view = ArrayViewMut3::from_shape((5, 5, 1), input.as_slice_mut()).unwrap();
            view[[2, 2, 0]] = 1.0;
        }

        let cmd = Laplacian { kernel_size: 1 };
        // 3. Execute the filter
        apply_laplacian(&input, &mut output, cmd.kernel_size).unwrap();

        // 4. Wrap output for easy verification
        let out_view = ArrayView3::from_shape((5, 5, 1), output.as_slice()).unwrap();

        // 5. Verify the Laplacian response
        // Center: (0 + 0 + 0 + 0) - 4 * 1.0 = -4.0
        assert_eq!(out_view[[2, 2, 0]], -4.0);

        // Neighbors: (1.0 + 0 + 0 + 0) - 4 * 0 = 1.0
        assert_eq!(out_view[[1, 2, 0]], 1.0); // Top
        assert_eq!(out_view[[3, 2, 0]], 1.0); // Bottom
        assert_eq!(out_view[[2, 1, 0]], 1.0); // Left
        assert_eq!(out_view[[2, 3, 0]], 1.0); // Right

        // Corners: Should remain 0.0 for a 5-point stencil
        assert_eq!(out_view[[1, 1, 0]], 0.0);
    }

    #[test]
    fn test_laplacian_f32_rgb() {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let input = Image::<f32, 3, CpuAllocator>::from_size_val(size, 1.0, CpuAllocator).unwrap();
        let mut output =
            Image::<f32, 3, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();

        // A flat image (all pixels 1.0) should result in 0.0 Laplacian
        let cmd = Laplacian { kernel_size: 1 };
        apply_laplacian(&input, &mut output, cmd.kernel_size).unwrap();

        let out_view = ArrayView3::from_shape((3, 3, 3), output.as_slice()).unwrap();

        // Check the center pixel across all 3 channels (R, G, B)
        for c in 0..3 {
            assert_eq!(out_view[[1, 1, c]], 0.0);
        }
    }

    #[test]
    fn test_laplacian_kernel_5() {
        let size = ImageSize {
            width: 7,
            height: 7,
        };
        let mut input =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let mut output =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();

        // Set center to 10.0
        {
            let mut view = ArrayViewMut3::from_shape((7, 7, 1), input.as_slice_mut()).unwrap();
            view[[3, 3, 0]] = 10.0;
        }

        // Apply Laplacian with kernel size 5
        apply_laplacian(&input, &mut output, 5).unwrap();

        let out_view = ArrayView3::from_shape((7, 7, 1), output.as_slice()).unwrap();

        // The center should reflect the kernel weight for 5: -6.0 * center_value
        // -6.0 * 10.0 = -60.0
        assert_eq!(out_view[[3, 3, 0]], -60.0);
    }

    #[test]
    fn test_laplacian_rgb_response() {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        // Create an RGB impulse at the center
        let mut input =
            Image::<f32, 3, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let mut output =
            Image::<f32, 3, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();

        {
            let mut view = ArrayViewMut3::from_shape((3, 3, 3), input.as_slice_mut()).unwrap();
            view[[1, 1, 0]] = 1.0; // R
            view[[1, 1, 1]] = 1.0; // G
            view[[1, 1, 2]] = 1.0; // B
        }

        apply_laplacian(&input, &mut output, 3).unwrap();
        let out_view = ArrayView3::from_shape((3, 3, 3), output.as_slice()).unwrap();

        // Center pixel should be -4.0 for all channels
        assert_eq!(out_view[[1, 1, 0]], -4.0);
        assert_eq!(out_view[[1, 1, 1]], -4.0);
        assert_eq!(out_view[[1, 1, 2]], -4.0);
    }

    #[test]
    fn test_laplacian_name() {
        let filter = Laplacian { kernel_size: 3 };
        assert_eq!(filter.name(), "Laplacian");
    }
}
