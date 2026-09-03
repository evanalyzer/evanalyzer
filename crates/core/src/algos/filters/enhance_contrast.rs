//! # enhance_contrast
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::sync::Arc;

/// Configuration for contrast enhancement and histogram manipulation.
///
/// This algorithm can perform linear contrast stretching, normalization,
/// or histogram equalization to improve the dynamic range of an image.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::EnhanceContrast;
/// let settings = EnhanceContrast {
///     saturated_pixels: 0.01,   // Clip 1% of outliers
///     normalize: true,          // Stretch to [0.0, 1.0]
///     equalize_histogram: false,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct EnhanceContrast {
    /// Percentage of pixels to "clip" from the top and bottom of the histogram.
    ///
    /// Range: [0.0, 1.0]. A value of 0.01 (1%) helps ignore hot/dead pixels
    /// that would otherwise prevent effective contrast stretching.
    pub saturated_pixels: f32,

    /// Whether to linearly stretch the remaining pixel intensities to fill
    /// the full [0.0, 1.0] range.
    pub normalize: bool,

    /// Whether to apply Histogram Equalization.
    ///
    /// This redistributes pixel intensities to achieve a uniform distribution,
    /// which is highly effective for images with low contrast but high noise.
    pub equalize_histogram: bool,
}

impl ImageAlgorithm for EnhanceContrast {
    /// Adjusts the dynamic range and brightness distribution of the image.
    ///
    /// The algorithm follows this sequence:
    /// 1. Outlier clipping based on `saturated_pixels`.
    /// 2. (Optional) Histogram Equalization to flatten the pixel distribution.
    /// 3. (Optional) Linear stretching to normalize intensities to [0.0, 1.0].
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Gray` or `F32Rgb`.
    /// * **Output:** Matches input format (In-place if possible).
    ///
    /// # Side Effects
    /// For RGB images, this implementation typically operates on the Luminance (Y)
    /// channel of a YCbCr conversion (or similar) to prevent color shifting.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the image data cannot be
    /// processed into a histogram.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                output
                    .data
                    .as_slice_mut()
                    .copy_from_slice(input.data.as_slice());
                self.process_f32_gray(&mut output.data);
                ctx.swap()?;
                Ok(())
            }
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                output
                    .data
                    .as_slice_mut()
                    .copy_from_slice(input.data.as_slice());
                self.process_f32_rgb(&mut output.data);
                ctx.swap()?;
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Rgb or F32Gray".into(),
                found: format!("Input: {:?}, Scratch: {:?}", ctx.image, ctx.scratch_pad),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Enhance Contrast"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl EnhanceContrast {
    /// Internal helper to apply contrast enhancement to a single-channel grayscale image.
    ///
    /// This performs the histogram analysis and redistribution directly on the
    /// provided buffer. It handles:
    /// 1. Finding the intensity percentiles for clipping.
    /// 2. Mapping the old intensity values to the new stretched/equalized values.
    ///
    /// # Arguments
    /// * `img` - A mutable reference to the grayscale image buffer. Modified in-place.
    fn process_f32_gray(&self, img: &mut Image<f32, 1, CpuAllocator>) {
        let slice = img.as_slice_mut();

        let hist = compute_f32_histogram(slice);

        if self.equalize_histogram {
            let lut = calculate_equalization_lut(&hist);
            slice.iter_mut().for_each(|p| *p = sample_lut(&lut, *p));
        } else {
            let (hmin, hmax) = get_stretch_bounds(&hist, slice.len(), self.saturated_pixels);
            rescale_slice(slice, hmin, hmax);
        }

        if self.normalize {
            let max_val = slice.iter().fold(0.0f32, |a, &b| a.max(b));
            rescale_slice(slice, 0.0, max_val);
        }
    }

    /// Internal helper to apply contrast enhancement to a three-channel color image.
    ///
    /// This performs the histogram analysis and redistribution directly on the
    /// provided buffer. It handles:
    /// 1. Finding the intensity percentiles for clipping.
    /// 2. Mapping the old intensity values to the new stretched/equalized values.
    ///
    /// # Arguments
    /// * `img` - A mutable reference to the rgb image buffer. Modified in-place.
    fn process_f32_rgb(&self, img: &mut Image<f32, 3, CpuAllocator>) {
        let slice = img.as_slice_mut();

        // Extract Luminance (Rec. 709 weights)
        let luminance: Vec<f32> = slice
            .chunks_exact(3)
            .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
            .collect();

        // Compute Histogram of Luminance
        let hist = compute_f32_histogram(&luminance);

        // Apply Transformation
        if self.equalize_histogram {
            let lut = calculate_equalization_lut(&hist);
            for (i, rgb) in slice.chunks_exact_mut(3).enumerate() {
                let old_lum = luminance[i];
                let new_lum = sample_lut(&lut, old_lum);
                apply_luminance_ratio(rgb, old_lum, new_lum);
            }
        } else {
            let (hmin, hmax) = get_stretch_bounds(&hist, luminance.len(), self.saturated_pixels);
            let range = (hmax - hmin).max(1e-6);

            for (i, rgb) in slice.chunks_exact_mut(3).enumerate() {
                let old_lum = luminance[i];
                let new_lum = ((old_lum - hmin) / range).clamp(0.0, 1.0);
                apply_luminance_ratio(rgb, old_lum, new_lum);
            }
        }

        // Color-Safe Normalization (Global stretch to 1.0 based on max brightness)
        if self.normalize {
            let max_lum = slice
                .chunks_exact(3)
                .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
                .fold(0.0f32, |a, b| a.max(b));

            if max_lum > 1e-6 && max_lum < 1.0 {
                let ratio = 1.0 / max_lum;
                for rgb in slice.chunks_exact_mut(3) {
                    rgb[0] = (rgb[0] * ratio).clamp(0.0, 1.0);
                    rgb[1] = (rgb[1] * ratio).clamp(0.0, 1.0);
                    rgb[2] = (rgb[2] * ratio).clamp(0.0, 1.0);
                }
            }
        }
    }
}

/// Helper: Scales RGB channels by the change in luminance to preserve color
fn apply_luminance_ratio(rgb: &mut [f32], old_lum: f32, new_lum: f32) {
    if old_lum > 1e-6 {
        let ratio = new_lum / old_lum;
        rgb[0] = (rgb[0] * ratio).clamp(0.0, 1.0);
        rgb[1] = (rgb[1] * ratio).clamp(0.0, 1.0);
        rgb[2] = (rgb[2] * ratio).clamp(0.0, 1.0);
    } else {
        // If starting from black, we must add neutral gray to brighten
        rgb[0] = new_lum;
        rgb[1] = new_lum;
        rgb[2] = new_lum;
    }
}

/// Computes a 16-bit resolution histogram for f32 data in the range [0.0, 1.0].
///
/// This uses 65,536 bins to maintain high precision for floating-point images.
///
/// # Panics
/// Will not panic on values outside [0.0, 1.0] due to internal clamping.
fn compute_f32_histogram(data: &[f32]) -> Vec<usize> {
    const BINS: usize = 65536;
    const MAX_IDX: f32 = (BINS - 1) as f32;

    let mut hist = vec![0usize; BINS];

    for &p in data {
        // Use fast_bound-checking or ensure NaN doesn't crash us
        if p.is_nan() {
            continue;
        }

        let bin = (p * MAX_IDX).clamp(0.0, MAX_IDX) as usize;

        // Safety: bin is guaranteed to be < 65536 by the clamp
        unsafe {
            *hist.get_unchecked_mut(bin) += 1;
        }
    }
    hist
}

/// Calculates bounds for contrast stretching based on saturation percentage
fn get_stretch_bounds(hist: &[usize], total: usize, saturated_pixels: f32) -> (f32, f32) {
    let threshold = (total as f32 * saturated_pixels / 200.0) as usize;

    let mut low_bin = 0;
    let mut count = 0;
    for (i, &v) in hist.iter().enumerate() {
        count += v;
        if count > threshold {
            low_bin = i;
            break;
        }
    }

    let mut high_bin = 65535;
    count = 0;
    for (i, &v) in hist.iter().enumerate().rev() {
        count += v;
        if count > threshold {
            high_bin = i;
            break;
        }
    }

    (low_bin as f32 / 65535.0, high_bin as f32 / 65535.0)
}

/// Simple linear rescale for grayscale slices
fn rescale_slice(slice: &mut [f32], min: f32, max: f32) {
    let range = (max - min).max(1e-6);
    slice.iter_mut().for_each(|p| {
        *p = ((*p - min) / range).clamp(0.0, 1.0);
    });
}

/// Computes a Weighted Square Root Histogram Equalization LUT
fn calculate_equalization_lut(hist: &[usize]) -> Vec<f32> {
    let mut weights = vec![0.0f64; 65536];
    for i in 0..65536 {
        let h = hist[i] as f64;
        // Square root weighting reduces over-stretching in flat areas
        weights[i] = if h < 2.0 { h } else { h.sqrt() };
    }

    let sum: f64 = weights.iter().sum::<f64>() * 2.0 - weights[0] - weights[65535];

    let scale = 1.0 / sum.max(1e-9);
    let mut lut = vec![0.0f32; 65536];
    let mut running_sum = weights[0];

    for i in 1..65535 {
        let delta = weights[i];
        running_sum += delta;
        lut[i] = (running_sum * scale) as f32;
        running_sum += delta;
    }
    lut[65535] = 1.0;
    lut
}

/// Helper to sample from our 65536-bin LUT using an f32 input
fn sample_lut(lut: &[f32], val: f32) -> f32 {
    let idx = (val * 65535.0).clamp(0.0, 65535.0) as usize;
    lut[idx]
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{image::PixelSizes, pipeline::pipeline::PipelineImageMeta};

    use super::*;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn test_enhance_contrast_stretch() {
        // 1. Setup: Create a low-contrast 10x10 gray image (values 0.2 to 0.5)
        let width = 10;
        let height = 10;
        let mut data = vec![0.0f32; width * height];
        for i in 0..data.len() {
            // Fill with a gradient from 0.2 to 0.5
            data[i] = 0.2 + (i as f32 / (width * height) as f32) * 0.3;
        }

        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();

        // 2. Prepare Context

        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        // 3. Configure Algorithm: Pure Stretch (no saturation, no equalization)

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: false,
            equalize_histogram: false,
        };

        // 4. Execute
        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        // 5. Verify
        if let ImageContainer::F32Gray(output) = ctx.image.as_ref() {
            let pixels = output.as_slice();

            // The minimum value (originally 0.2) should now be ~0.0
            assert!(pixels[0] < 0.01);

            // The maximum value (originally 0.5) should now be ~1.0
            assert!(pixels[pixels.len() - 1] > 0.99);

            // Check that we didn't get any NaNs
            assert!(pixels.iter().all(|&p| !p.is_nan()));
        } else {
            panic!("Expected F32Gray output");
        }
    }

    #[test]
    fn test_enhance_contrast_equalize_rgb() {
        // Create a dark RGB image
        let width = 2;
        let height = 2;
        let data = vec![0.1f32; width * height * 3]; // Flat dark gray

        let img =
            Image::<f32, 3, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();

        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 2,
                    height: 2,
                },
                full_image_width: ImageSize {
                    width: 2,
                    height: 2,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_rgb_from_image_test(img).into(),
        )
        .unwrap();

        let mut cache = GlobalPipelineCache::default();

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: false,
            equalize_histogram: true,
        };

        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        if let ImageContainer::F32Rgb(output) = ctx.image.as_ref() {
            let pixels = output.as_slice();
            // Equalization on a flat image should generally push values
            // toward the boundaries or keep them consistent.
            // Main check: ensure no color shift (R should still equal G and B)
            assert_eq!(pixels[0], pixels[1]);
            assert_eq!(pixels[1], pixels[2]);
        }
    }

    #[test]
    fn test_enhance_contrast_format_mismatch_error() {
        // 1. Create a U32 image (Unsupported type)
        let img = Image::<u32, 1, CpuAllocator>::from_size_val(
            ImageSize {
                width: 5,
                height: 5,
            },
            0,
            CpuAllocator,
        )
        .unwrap();

        // 2. Initialize Context with U32 image
        let mut ctx = PipelineContext::new_from_u32_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        // 3. Setup the command
        let enhancer = EnhanceContrast {
            saturated_pixels: 0.01,
            normalize: true,
            equalize_histogram: false,
        };

        // 4. Execute
        let result = enhancer.execute(&mut ctx, &mut cache);

        // 5. Assert error
        match result {
            Err(InternalErrors::FormatMismatch { expected, found }) => {
                assert_eq!(expected, "F32Rgb or F32Gray");
                assert!(found.contains("Input")); // Checks that the debug output is present
            }
            _ => panic!("Expected FormatMismatch error, but got {:?}", result),
        }
    }

    #[test]
    fn test_enhance_contrast_gray_equalize() {
        // Gray equalize was only exercised via the RGB test before - covers
        // `process_f32_gray`'s `equalize_histogram` branch directly.
        let width = 4;
        let height = 4;
        let data: Vec<f32> = (0..width * height)
            .map(|i| 0.3 + (i as f32) * 0.01)
            .collect();
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: false,
            equalize_histogram: true,
        };
        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        if let ImageContainer::F32Gray(output) = ctx.image.as_ref() {
            assert!(
                output
                    .as_slice()
                    .iter()
                    .all(|&p| (0.0..=1.0).contains(&p) && !p.is_nan())
            );
        } else {
            panic!("Expected F32Gray output");
        }
    }

    #[test]
    fn test_enhance_contrast_gray_normalize() {
        // Covers `process_f32_gray`'s `normalize` branch: after stretching,
        // the whole slice is rescaled again so the max reaches 1.0.
        let width = 4;
        let height = 4;
        // Deliberately dim (max 0.4) so normalize has visible work to do.
        let data: Vec<f32> = (0..width * height).map(|i| (i as f32) * 0.02).collect();
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: true,
            equalize_histogram: false,
        };
        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        if let ImageContainer::F32Gray(output) = ctx.image.as_ref() {
            let pixels = output.as_slice();
            let max = pixels.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                max > 0.99,
                "normalize should stretch the max back up to ~1.0, got {max}"
            );
        } else {
            panic!("Expected F32Gray output");
        }
    }

    #[test]
    fn test_enhance_contrast_rgb_stretch() {
        // The non-equalize (linear stretch) path for RGB wasn't covered -
        // only the equalize branch was, in `test_enhance_contrast_equalize_rgb`.
        let width = 2;
        let height = 2;
        let data = vec![
            0.2, 0.2, 0.2, // dim pixel
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, // bright pixel
        ];
        let img =
            Image::<f32, 3, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: false,
            equalize_histogram: false,
        };
        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        if let ImageContainer::F32Rgb(output) = ctx.image.as_ref() {
            let pixels = output.as_slice();
            // The dim pixel's luminance was the minimum - stretched to ~0.
            assert!(
                pixels[0] < 0.01,
                "dim pixel should stretch toward black, got {}",
                pixels[0]
            );
            // The bright pixel's luminance was the maximum - stretched to ~1.
            let last_px = &pixels[pixels.len() - 3..];
            assert!(
                last_px[0] > 0.99,
                "bright pixel should stretch toward white, got {}",
                last_px[0]
            );
        } else {
            panic!("Expected F32Rgb output");
        }
    }

    #[test]
    fn test_enhance_contrast_rgb_normalize() {
        let width = 2;
        let height = 1;
        // Both pixels dim (luminance well under 1.0) so the normalize branch
        // (`max_lum > 1e-6 && max_lum < 1.0`) actually fires.
        let data = vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2];
        let img =
            Image::<f32, 3, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let enhancer = EnhanceContrast {
            saturated_pixels: 0.0,
            normalize: true,
            equalize_histogram: false,
        };
        let result = enhancer.execute(&mut ctx, &mut cache);
        assert!(result.is_ok());

        if let ImageContainer::F32Rgb(output) = ctx.image.as_ref() {
            let pixels = output.as_slice();
            let max_lum = pixels
                .chunks_exact(3)
                .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
                .fold(0.0f32, f32::max);
            assert!(
                max_lum > 0.99,
                "normalize should push peak luminance to ~1.0, got {max_lum}"
            );
        } else {
            panic!("Expected F32Rgb output");
        }
    }

    #[test]
    fn apply_luminance_ratio_from_black_adds_neutral_gray_instead_of_dividing_by_zero() {
        // old_lum <= 1e-6 is a separate branch specifically to avoid a
        // division by (near) zero.
        let mut rgb = [0.0f32, 0.0, 0.0];
        apply_luminance_ratio(&mut rgb, 0.0, 0.5);
        assert_eq!(rgb, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn apply_luminance_ratio_scales_color_preserving_channel_ratios() {
        let mut rgb = [0.2f32, 0.4, 0.6];
        apply_luminance_ratio(&mut rgb, 0.4, 0.8);
        // ratio = 2.0, clamped to [0, 1]
        assert_eq!(rgb, [0.4, 0.8, 1.0]);
    }

    #[test]
    fn compute_f32_histogram_ignores_nan_and_clamps_out_of_range_values() {
        let hist = compute_f32_histogram(&[f32::NAN, -1.0, 2.0, 0.5]);
        // NaN is skipped entirely; -1.0 and 2.0 clamp into bin 0 / bin 65535.
        let total: usize = hist.iter().sum();
        assert_eq!(total, 3, "NaN must not be counted");
        assert_eq!(hist[0], 1);
        assert_eq!(hist[65535], 1);
    }

    #[test]
    fn get_stretch_bounds_with_zero_saturation_spans_the_full_data_range() {
        let mut hist = vec![0usize; 65536];
        hist[100] = 1;
        hist[60000] = 1;
        let (lo, hi) = get_stretch_bounds(&hist, 2, 0.0);
        assert!((lo - 100.0 / 65535.0).abs() < 1e-6);
        assert!((hi - 60000.0 / 65535.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_slice_maps_min_max_into_zero_one_and_clamps_outliers() {
        let mut data = vec![0.0f32, 0.5, 1.0, 1.5];
        rescale_slice(&mut data, 0.0, 1.0);
        assert_eq!(data, vec![0.0, 0.5, 1.0, 1.0]); // 1.5 clamps to 1.0
    }

    #[test]
    fn sample_lut_maps_endpoints_and_clamps_out_of_range_input() {
        let mut lut = vec![0.0f32; 65536];
        lut[0] = 0.1;
        lut[65535] = 0.9;
        assert_eq!(sample_lut(&lut, 0.0), 0.1);
        assert_eq!(sample_lut(&lut, 1.0), 0.9);
        assert_eq!(
            sample_lut(&lut, 2.0),
            0.9,
            "out-of-range input should clamp to the last bin"
        );
    }

    #[test]
    fn test_name() {
        let enhancer = EnhanceContrast {
            saturated_pixels: 0.01,
            normalize: true,
            equalize_histogram: false,
        };
        let name = enhancer.name();
        assert_eq!(name, "Enhance Contrast");
    }
}
