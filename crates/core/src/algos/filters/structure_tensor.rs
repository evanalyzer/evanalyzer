//! # structure_tensor
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::pipeline::pipeline_cache::PipelineCache;
use crate::{
    algos::ImageAlgorithm, image::ImageContainer, pipeline::pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_imgproc::filter::gaussian_blur;
use kornia_imgproc::filter::spatial_gradient_float;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;

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
        // Compute gradients
        let size = input.size();
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
        "StructureTensor"
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::pipeline::pipeline_cache::ImageCache;

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
        let mut cache = PipelineCache::default();

        algo.execute(&mut ctx, &mut cache)?;

        // 5. Verify Results
        if let ImageContainer::F32Gray(result) = ctx.image {
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
}
