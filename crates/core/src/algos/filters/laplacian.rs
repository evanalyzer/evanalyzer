//! # laplacian
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::image::ImageContainer;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::pipeline_cache::GlobalPipelineCache,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_image::Image;
use macros::CommandsMeta;
use ndarray::{ArrayView3, ArrayViewMut3};
use std::sync::Arc;

/// Configuration for the Laplacian edge detection filter.
///
/// The Laplacian is a second-order derivative operator used to find regions of
/// rapid intensity change. It is particularly effective for detecting edges
/// and fine details, though it is highly sensitive to noise.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct Laplacian {
    /// The size of the discrete Laplacian aperture.
    ///
    /// Typically 3. Larger sizes (5, 7) approximate the Laplacian of Gaussian (LoG)
    /// more closely but are more computationally expensive. Must be an odd number.
    #[cmdsmeta(default = 3, min = 3, max = 7, summary = true, step = 2)]
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
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                apply_laplacian(input, output, self.kernel_size)?;
                ctx.swap()?;
                Ok(())
            }
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                apply_laplacian(input, output, self.kernel_size)?;
                ctx.swap()?;
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

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&GONZALEZ_2018]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// The textbook discrete Laplacian (spatial filtering chapter).
const GONZALEZ_2018: CitationMetadata = CitationMetadata {
    cite_key: "gonzalez2018digital",
    title: "Digital Image Processing",
    authors: &["Rafael C. Gonzalez", "Richard E. Woods"],
    year: 2018,
    container: Some("Pearson, 4th edition, ISBN 978-0-13-335672-4"),
    doi: None,
    url: None,
    pages: None,
};

/// The second-derivative and smoothing kernels of an aperture of size 5 or
/// 7: binomial smoothing, and binomial smoothing of size `k - 2` convolved
/// with `[1, -2, 1]`. Both are normalized, so every aperture estimates the
/// Laplacian of the image in the same units as the 3x3 stencil (a quadratic
/// `x^2` gives 2 for every size).
fn separable_kernels(k: usize) -> ([f32; 7], [f32; 7]) {
    let (derivative, smoothing): (&[f32], &[f32]) = match k {
        5 => (&[1.0, 0.0, -2.0, 0.0, 1.0], &[1.0, 4.0, 6.0, 4.0, 1.0]),
        _ => (
            &[1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0],
            &[1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        ),
    };
    // Sum of the smoothing weights: 2^(k-1). Response of the derivative to
    // x^2, divided by 2: 2^(k-3).
    let smoothing_sum = (1u32 << (k - 1)) as f32;
    let derivative_gain = (1u32 << (k - 3)) as f32;
    let mut d = [0.0; 7];
    let mut g = [0.0; 7];
    for i in 0..k {
        d[i] = derivative[i] / derivative_gain;
        g[i] = smoothing[i] / smoothing_sum;
    }
    (d, g)
}

/// Writes the Laplacian of `input` into `output`. `kernel_size` 1 or 3 is the
/// 5-point stencil; 5 and 7 smooth across the derivative direction
/// (d²/dx² * smoothing(y) + smoothing(x) * d²/dy²). Pixels closer than
/// half the aperture to the border are 0.
fn apply_laplacian<const C: usize>(
    input: &Image<f32, C>,
    output: &mut Image<f32, C>,
    kernel_size: usize,
) -> Result<(), InternalErrors> {
    let k = match kernel_size {
        0 | 1 | 3 => 3,
        5 | 7 => kernel_size,
        _ => {
            return Err(InternalErrors::InvalidArgument(format!(
                "Laplacian: kernel size must be 1, 3, 5 or 7, got {kernel_size}"
            )));
        }
    };
    let (h, w) = (input.size().height, input.size().width);
    let pad = k / 2;

    // Map views with error handling
    let in_view = ArrayView3::from_shape((h, w, C), input.as_slice())?;
    let mut out_view = ArrayViewMut3::from_shape((h, w, C), output.as_slice_mut())?;
    // The scratch pad holds an earlier step's pixels: clear the border.
    out_view.fill(0.0);

    if h < k || w < k {
        return Ok(());
    }

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
        let (d, g) = separable_kernels(k);
        let mut out_slice = out_view.slice_mut(ndarray::s![pad..h - pad, pad..w - pad, ..]);
        ndarray::Zip::indexed(&mut out_slice).for_each(|(y_inner, x_inner, c), out_pixel| {
            let mut val = 0.0;
            for j in 0..k {
                let y = y_inner + j;
                for i in 0..k {
                    let x = x_inner + i;
                    val += in_view[[y, x, c]] * (d[i] * g[j] + g[i] * d[j]);
                }
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
        let mut input = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        let mut output = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();

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
        let input = Image::<f32, 3>::from_size_val(size, 1.0).unwrap();
        let mut output = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();

        // A flat image (all pixels 1.0) should result in 0.0 Laplacian
        let cmd = Laplacian { kernel_size: 1 };
        apply_laplacian(&input, &mut output, cmd.kernel_size).unwrap();

        let out_view = ArrayView3::from_shape((3, 3, 3), output.as_slice()).unwrap();

        // Check the center pixel across all 3 channels (R, G, B)
        for c in 0..3 {
            assert_eq!(out_view[[1, 1, c]], 0.0);
        }
    }

    /// Every aperture estimates the Laplacian in the same units: x² + y²
    /// has a Laplacian of 4 everywhere.
    #[test]
    fn every_kernel_size_gives_the_laplacian_of_a_quadratic() {
        let size = ImageSize {
            width: 9,
            height: 9,
        };
        let mut input = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        {
            let mut view = ArrayViewMut3::from_shape((9, 9, 1), input.as_slice_mut()).unwrap();
            for y in 0..9 {
                for x in 0..9 {
                    view[[y, x, 0]] = (x * x + y * y) as f32;
                }
            }
        }
        for k in [1, 3, 5, 7] {
            let mut output = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
            apply_laplacian(&input, &mut output, k).unwrap();
            let out_view = ArrayView3::from_shape((9, 9, 1), output.as_slice()).unwrap();
            assert!((out_view[[4, 4, 0]] - 4.0).abs() < 1e-4, "kernel size {k}");
        }
    }

    #[test]
    fn larger_kernels_respond_to_a_flat_image_with_zero_and_clear_the_border() {
        let size = ImageSize {
            width: 9,
            height: 9,
        };
        let input = Image::<f32, 1>::from_size_val(size, 0.5).unwrap();
        for k in [5, 7] {
            // A stale scratch pad: every pixel must be overwritten.
            let mut output = Image::<f32, 1>::from_size_val(size, 9.0).unwrap();
            apply_laplacian(&input, &mut output, k).unwrap();
            assert!(
                output.as_slice().iter().all(|v| v.abs() < 1e-6),
                "kernel size {k}"
            );
        }
    }

    #[test]
    fn unsupported_kernel_sizes_are_rejected() {
        let size = ImageSize {
            width: 9,
            height: 9,
        };
        let input = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        let mut output = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        for k in [2, 4, 9] {
            assert!(
                apply_laplacian(&input, &mut output, k).is_err(),
                "kernel size {k}"
            );
        }
    }

    #[test]
    fn execute_puts_the_result_into_the_image() {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut input = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        input.as_slice_mut()[12] = 1.0;
        let mut ctx = PipelineContext::new_from_image_test(input).unwrap();
        let mut cache = GlobalPipelineCache::default();
        Laplacian { kernel_size: 3 }
            .execute(&mut ctx, &mut cache)
            .unwrap();
        let ImageContainer::F32Gray(result) = ctx.image.as_ref() else {
            panic!("expected F32Gray");
        };
        assert_eq!(result.as_slice()[12], -4.0);
        assert_eq!(result.as_slice()[7], 1.0);
    }

    #[test]
    fn test_laplacian_rgb_response() {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        // Create an RGB impulse at the center
        let mut input = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();
        let mut output = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();

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
        assert_eq!(filter.cite()[0].cite_key, "gonzalez2018digital");
    }
}
