//! # structure_tensor
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    image::ImageContainer,
    pipeline::pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_image::Image;
use kornia_imgproc::filter::gaussian_blur;
use kornia_imgproc::filter::spatial_gradient_float;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::sync::Arc;

/// The specific calculation to extract from the Structure Tensor.
pub enum TensorMode {
    /// Extracts the first (primary) eigenvalue.
    ///
    /// Represents the local image intensity variation in the direction
    /// perpendicular to the edge. Useful for edge detection.
    EigenvaluesX,

    /// Extracts the second (secondary) eigenvalue.
    ///
    /// Represents the local image intensity variation along the edge.
    /// High values typically indicate corners or noise.
    EigenvaluesY,

    /// Computes the local anisotropy (coherence) of the image.
    ///
    /// Measures how strongly the local neighborhood is oriented.
    /// Ranges from 0 (isotropic/noise) to 1 (perfectly oriented/straight edge).
    Coherence,
}

/// Analyzes local image texture, directional orientation, and corner features using a second-moment matrix.
///
/// This algorithm summarizes the predominant directions of the image gradient within a local
/// neighborhood, smoothing the structural data with a Gaussian window. By evaluating the
/// eigenvalues of the resulting matrix tensor, it distinguishes between flat areas (both eigenvalues
/// near zero), straight linear boundaries (one dominant eigenvalue indicating structural direction),
/// and complex corners or intersections (two large eigenvalues).
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::{StructureTensor, Mode};
/// let settings = StructureTensor {
///     mode: Mode::Coherence,
///     kernel_size: 3,
///     sigma: 1.5
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct StructureTensor {
    /// The mathematical output to be produced by the algorithm.
    pub mode: TensorMode,

    /// The size of the integration window used to average the local gradients.
    ///
    /// Larger windows provide more stability against noise but reduce
    /// spatial resolution.
    pub kernel_size: usize,

    /// The standard deviation for the Gaussian weighting of the integration window.
    ///
    /// Controls the spatial "reach" of the neighborhood analysis.
    pub sigma: f32,
}
impl ImageAlgorithm for StructureTensor {
    /// Computes the Structure Tensor and extracts the specified feature (Eigenvalues or Coherence).
    ///
    /// The algorithm requires two pre-allocated `F32Gray` buffers:
    /// 1. The **Source** buffer (`ctx.image`) containing the original intensity data.
    /// 2. The **Scratch Pad** (`ctx.scratch_pad`) where the final computed feature is stored.
    ///
    /// # Pipeline Logic
    /// - Computes local gradients ($I_x, I_y$) using Sobel or Scharr operators.
    /// - Forms the second-moment matrix (Structure Tensor) for each pixel.
    /// - Smooths the tensor components ($I_x^2, I_y^2, I_x I_y$) using the specified `sigma`.
    /// - Calculates the result based on the selected [`Mode`].
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::FormatMismatch`] if either the input image or the
    /// scratch pad are not in `F32Gray` format.
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
        // spatial_gradient_float panics inside kornia itself (chunks_mut(0))
        // for a zero-width image - the `?`/`map_err` calls below can't catch
        // that, since the panic happens before any Result is produced. Guard
        // explicitly instead.
        let size = input.size();
        if size.width == 0 || size.height == 0 {
            return Err(InternalErrors::Generic(format!(
                "StructureTensor requires a non-empty image, got {}x{}",
                size.width, size.height
            )));
        }

        // Compute gradients
        let mut gx = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;

        let mut gy = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        spatial_gradient_float(&input, &mut gx, &mut gy).map_err(InternalErrors::from_kornia)?;

        // Structure tensor components
        // Pre-allocate the images directly (no intermediate temp vectors)
        let size = gx.size();
        let mut jxx = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        let mut jyy = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        let mut jxy = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;

        // Use Rayon to compute all three components in parallel across all CPU cores
        // This is cache-friendly because dx/dy are read once and stay in L1/L2 cache
        jxx.as_slice_mut()
            .par_iter_mut()
            .zip(jyy.as_slice_mut().par_iter_mut())
            .zip(jxy.as_slice_mut().par_iter_mut())
            .zip(gx.as_slice().par_iter())
            .zip(gy.as_slice().par_iter())
            .for_each(|((((out_xx, out_yy), out_xy), &val_x), &val_y)| {
                *out_xx = val_x * val_x;
                *out_yy = val_y * val_y;
                *out_xy = val_x * val_y;
            });

        gaussian_blur(
            &jxx,
            output,
            (self.kernel_size, self.kernel_size),
            (self.sigma, self.sigma),
        )
        .map_err(InternalErrors::from_kornia)?;
        std::mem::swap(&mut jxx, output);

        // Blur Jyy using scratch_pad, then swap data back
        gaussian_blur(
            &jyy,
            output,
            (self.kernel_size, self.kernel_size),
            (self.sigma, self.sigma),
        )
        .map_err(InternalErrors::from_kornia)?;
        std::mem::swap(&mut jyy, output);

        //  Blur Jxy using scratch_pad, then swap data back
        gaussian_blur(
            &jxy,
            output,
            (self.kernel_size, self.kernel_size),
            (self.sigma, self.sigma),
        )
        .map_err(InternalErrors::from_kornia)?;
        std::mem::swap(&mut jxy, output);

        // Eigenvalues λ1, λ2
        // Access raw slices for maximum speed
        let s_jxx = jxx.as_slice();
        let s_jyy = jyy.as_slice();
        let s_jxy = jxy.as_slice();

        // Compute λ1 or λ2 (or Coherence) in a single parallel pass
        output
            .as_slice_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, val)| {
                let ixx = s_jxx[i];
                let iyy = s_jyy[i];
                let ixy = s_jxy[i];

                // tmp = sqrt((Jxx - Jyy)^2 + 4 * Jxy^2)
                let diff = ixx - iyy;
                let tmp = (diff * diff + 4.0 * ixy * ixy).sqrt();

                // Calculate eigenvalues based on your desired output mode
                let l1 = 0.5 * (ixx + iyy + tmp);
                let l2 = 0.5 * (ixx + iyy - tmp);

                // Assign to result based on the setting (Example: EigenvaluesX)
                *val = match self.mode {
                    TensorMode::EigenvaluesX => l1,
                    TensorMode::EigenvaluesY => l2,
                    TensorMode::Coherence => (l1 - l2) / (l1 + l2 + 1e-6),
                };
            });

        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Structure Tensor"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "bigun1987orientation",
            title: "Optimal Orientation Detection of Linear Symmetry",
            authors: &["Josef Bigün", "Gösta H. Granlund"],
            year: 1987,
            container: Some(
                "Proceedings of the First International Conference on Computer Vision (ICCV)",
            ),
            doi: None,
            url: None,
            pages: None,
        })
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_structure_tensor_edge_detection() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0.0f32; 100];

        // 1. Create a vertical edge: Left half 0.0, Right half 1.0
        for y in 0..10 {
            for x in 5..10 {
                data[y * 10 + x] = 1.0;
            }
        }

        let input_img = Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)?;
        // 2. Setup Context
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        // 3. Setup Algorithm (Coherence mode)
        let algo = StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::Coherence,
        };

        // 4. Execute
        let mut cache = GlobalPipelineCache::default();

        algo.execute(&mut ctx, &mut cache)?;

        // 5. Verify Results
        if let ImageContainer::F32Gray(result) = ctx.image.as_ref() {
            let res_slice = result.as_slice();

            // At the edge (column 4 and 5), coherence should be high
            // In the middle of the left/right blocks, coherence should be near 0
            let edge_value = res_slice[5 * 10 + 5];
            let flat_value = res_slice[5 * 10 + 1];

            assert!(
                edge_value > 0.8,
                "Edge coherence should be high, got {}",
                edge_value
            );
            assert!(
                flat_value < 0.1,
                "Flat area coherence should be low, got {}",
                flat_value
            );
        } else {
            panic!("Output image was not Grayscale");
        }

        Ok(())
    }

    #[test]
    fn zero_width_image_returns_error_instead_of_panicking() {
        let size = ImageSize {
            width: 0,
            height: 5,
        };
        let input_img =
            Image::<f32, 1, CpuAllocator>::new(size, vec![], CpuAllocator).expect("image");
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let algo = StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::Coherence,
        };

        let result = algo.execute(&mut ctx, &mut cache);
        assert!(
            result.is_err(),
            "expected an error for a zero-width image, got Ok"
        );
    }

    /// Same vertical-edge layout as `test_structure_tensor_edge_detection`
    /// (left half 0.0, right half 1.0) - reused by the `EigenvaluesX`/
    /// `EigenvaluesY` tests below, which need a fresh context per run since
    /// `execute` consumes/swaps it.
    fn vertical_edge_ctx() -> PipelineContext {
        let size = ImageSize {
            width: 10,
            height: 10,
        };
        let mut data = vec![0.0f32; 100];
        for y in 0..10 {
            for x in 5..10 {
                data[y * 10 + x] = 1.0;
            }
        }
        let input_img = Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator).unwrap();
        PipelineContext::new_from_image_test(input_img).unwrap()
    }

    #[test]
    fn eigenvalues_x_reports_the_larger_eigenvalue_which_is_high_at_a_straight_edge() {
        let mut ctx = vertical_edge_ctx();
        let mut cache = GlobalPipelineCache::default();
        let algo = StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::EigenvaluesX,
        };

        algo.execute(&mut ctx, &mut cache).unwrap();

        if let ImageContainer::F32Gray(result) = ctx.image.as_ref() {
            let res_slice = result.as_slice();
            let edge_value = res_slice[5 * 10 + 5];
            let flat_value = res_slice[5 * 10 + 1];
            assert!(
                edge_value > flat_value,
                "the larger eigenvalue must be higher at the edge ({edge_value}) than in a flat region ({flat_value})"
            );
        } else {
            panic!("Output image was not Grayscale");
        }
    }

    #[test]
    fn eigenvalues_y_never_exceeds_eigenvalues_x_at_the_same_pixel() {
        // l1 (EigenvaluesX) = trace/2 + sqrt(discriminant)/2, l2
        // (EigenvaluesY) = trace/2 - sqrt(discriminant)/2 - l1 >= l2 always,
        // by construction, for every pixel, not just at this particular edge.
        let mut ctx_x = vertical_edge_ctx();
        let mut cache = GlobalPipelineCache::default();
        StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::EigenvaluesX,
        }
        .execute(&mut ctx_x, &mut cache)
        .unwrap();
        let ImageContainer::F32Gray(l1_result) = ctx_x.image.as_ref() else {
            panic!("Output image was not Grayscale");
        };
        let l1 = l1_result.as_slice()[5 * 10 + 5];

        let mut ctx_y = vertical_edge_ctx();
        StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::EigenvaluesY,
        }
        .execute(&mut ctx_y, &mut cache)
        .unwrap();
        let ImageContainer::F32Gray(l2_result) = ctx_y.image.as_ref() else {
            panic!("Output image was not Grayscale");
        };
        let l2 = l2_result.as_slice()[5 * 10 + 5];

        assert!(
            l2 <= l1 + 1e-5,
            "the secondary eigenvalue ({l2}) must never exceed the primary one ({l1})"
        );
    }

    #[test]
    fn execute_returns_format_mismatch_for_an_unsupported_image_type() {
        let img = Image::<u32, 1, CpuAllocator>::from_size_val(
            ImageSize {
                width: 5,
                height: 5,
            },
            0,
            CpuAllocator,
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_u32_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let algo = StructureTensor {
            kernel_size: 3,
            sigma: 1.0,
            mode: TensorMode::Coherence,
        };

        let result = algo.execute(&mut ctx, &mut cache);

        match result {
            Err(InternalErrors::FormatMismatch { expected, .. }) => {
                assert_eq!(expected, "F32Gray for both input and scratch pad");
            }
            _ => panic!("Expected FormatMismatch, got {:?}", result),
        }
    }
}
