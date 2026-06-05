//! # intensity_transform
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
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use ndarray::ArrayViewMut3;

/// Specifies how intensity adjustments are calculated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntensityTransformMode {
    /// Parameters are calculated based on image statistics (e.g., histogram analysis).
    Automatic,
    /// Parameters are provided explicitly by the user.
    Manual,
}
/// Configuration for adjusting image contrast and brightness.
///
/// This transformation applies a linear mapping to pixel values.
/// In [`Mode::Manual`], the output is typically calculated as:
/// `output = input * contrast + brightness`.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct IntensityTransformation {
    /// Determines whether to use automated enhancement or user-defined values.
    pub mode: IntensityTransformMode,

    /// Contrast multiplier (gain).
    ///
    /// Only active in [`Mode::Manual`].
    /// Values > 1.0 increase contrast, while values < 1.0 decrease it.
    pub contrast: f32,

    /// Brightness offset (bias).
    ///
    /// Only active in [`Mode::Manual`].
    /// Positive values brighten the image, negative values darken it.
    pub brightness: f32,
}
impl ImageAlgorithm for IntensityTransformation {
    /// Applies contrast and brightness adjustments to the image.
    ///
    /// Depending on the configured [`Mode`], this will either:
    /// * **Manual**: Apply a linear transformation using the user-provided
    ///   `contrast` (gain) and `brightness` (bias).
    /// * **Automatic**: Analyze the image histogram to calculate optimal
    ///   gain and bias parameters for maximum dynamic range.
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Gray` or `F32Rgb`.
    /// * **Output:** Matches input format (processed in-place or via scratch pad).
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the image and scratch pad
    /// types are incompatible for point-wise arithmetic.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match &mut ctx.image {
            // These are different types, so they need separate arms
            ImageContainer::F32Gray(img) => self.process_f32_image(img),
            ImageContainer::F32Rgb(img) => self.process_f32_image(img),
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray or F32Rgb".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "IntensityTransformation"
    }
}

impl IntensityTransformation {
    fn process_f32_image<const C: usize>(
        &self,
        image: &mut Image<f32, C, CpuAllocator>,
    ) -> Result<(), InternalErrors> {
        match self.mode {
            IntensityTransformMode::Automatic => self.equalize_hist_f32(image),
            IntensityTransformMode::Manual => self.apply_manual_f32(image),
        }
    }

    /// Manual Contrast/Brightness: result = contrast * pixel + brightness
    fn apply_manual_f32<const C: usize>(
        &self,
        image: &mut Image<f32, C, CpuAllocator>,
    ) -> Result<(), InternalErrors> {
        let (h, w) = (image.size().height, image.size().width);
        let mut view = ArrayViewMut3::from_shape((h, w, C), image.as_slice_mut())?;

        let c = self.contrast;
        let b = self.brightness;

        view.mapv_inplace(|pixel| {
            // Clamp to [0.0, 1.0] to emulate saturate_cast
            (c * pixel + b).clamp(0.0, 1.0)
        });
        Ok(())
    }

    /// Histogram Equalization for Float Images
    fn equalize_hist_f32<const C: usize>(
        &self,
        image: &mut Image<f32, C, CpuAllocator>,
    ) -> Result<(), InternalErrors> {
        let (h, w) = (image.size().height, image.size().width);
        let total_pixels = (h * w) as f32;
        let mut view = ArrayViewMut3::from_shape((h, w, C), image.as_slice_mut())?;

        // Use 65536 bins for high precision even with floats
        const BINS: usize = 65536;

        for c in 0..C {
            // Quantized Histogram
            let mut hist = vec![0usize; BINS];
            for &pixel in view.slice(ndarray::s![.., .., c]) {
                // Map [0.0, 1.0] to [0, 65535]
                let bin = (pixel * (BINS - 1) as f32).clamp(0.0, (BINS - 1) as f32) as usize;
                hist[bin] += 1;
            }

            // CDF
            let mut cdf = vec![0usize; BINS];
            cdf[0] = hist[0];
            for i in 1..BINS {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            // Map back to float
            view.slice_mut(ndarray::s![.., .., c])
                .mapv_inplace(|pixel| {
                    let bin = (pixel * (BINS - 1) as f32).clamp(0.0, (BINS - 1) as f32) as usize;
                    cdf[bin] as f32 / total_pixels
                });
        }
        Ok(())
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    // Helper to create a dummy context
    fn setup_test_image(width: usize, height: usize, val: f32) -> Image<f32, 1, CpuAllocator> {
        let size = ImageSize { width, height };
        Image::<f32, 1, CpuAllocator>::from_size_val(size, val, CpuAllocator).unwrap()
    }

    #[test]
    fn test_manual_intensity_adjustment() {
        let mut img = setup_test_image(2, 2, 0.5); // 2x2 image, all pixels 0.5

        let algo = IntensityTransformation {
            mode: IntensityTransformMode::Manual,
            contrast: 1.2,   // 0.5 * 1.2 = 0.6
            brightness: 0.1, // 0.6 + 0.1 = 0.7
        };
        algo.process_f32_image(&mut img).unwrap();

        let data = img.as_slice();
        for &pixel in data {
            // Check if (0.5 * 1.2) + 0.1 = 0.7
            assert!((pixel - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    fn test_manual_clamping() {
        let mut img = setup_test_image(2, 2, 0.9);

        let algo = IntensityTransformation {
            mode: IntensityTransformMode::Manual,
            contrast: 2.0,   // 1.8
            brightness: 0.5, // 2.3 -> should clamp to 1.0
        };
        algo.process_f32_image(&mut img).unwrap();

        for &pixel in img.as_slice() {
            assert_eq!(pixel, 1.0);
        }
    }

    #[test]
    fn test_automatic_equalization() {
        let size = ImageSize {
            width: 4,
            height: 1,
        };
        let mut img =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();

        // Create a very skewed image: [0.1, 0.1, 0.1, 1.0]
        {
            let slice = img.as_slice_mut();
            slice[0] = 0.1;
            slice[1] = 0.1;
            slice[2] = 0.1;
            slice[3] = 1.0;
        }

        let algo = IntensityTransformation {
            mode: IntensityTransformMode::Automatic,
            contrast: 1.0,
            brightness: 0.0,
        };
        algo.process_f32_image(&mut img).unwrap();

        let result = img.as_slice();

        // In a 4-pixel image:
        // 0.1 is the 3rd pixel in the CDF (3/4 = 0.75)
        // 1.0 is the 4th pixel in the CDF (4/4 = 1.0)
        // Values should be redistributed
        assert!((result[0] - 0.75).abs() < 0.01);
        assert!((result[3] - 1.0).abs() < 0.01);
    }
}
