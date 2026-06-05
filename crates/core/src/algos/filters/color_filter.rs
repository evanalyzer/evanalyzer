//! # color_filter
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::ImagePlane;
use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::{ImageContainer, ManagedImage};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;

/// Defines a range within the HSV (Hue, Saturation, Value) color space.
///
/// This is commonly used for color-based filtering or "chroma keying."
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::HsvRange;
///
/// let green_filter = HsvRange {
///     min_h: 100.0, max_h: 140.0,
///     min_s: 0.2,   max_s: 1.0,
///     min_v: 0.2,   max_v: 1.0,
/// };
/// ```
pub struct HsvRange {
    /// Minimum Hue angle in degrees [0.0, 360.0].
    pub min_h: f32,
    /// Maximum Hue angle in degrees [0.0, 360.0].
    pub max_h: f32,

    /// Minimum Saturation normalized [0.0, 1.0].
    pub min_s: f32,
    /// Maximum Saturation normalized [0.0, 1.0].
    pub max_s: f32,

    /// Minimum Value (Brightness) normalized [0.0, 1.0].
    pub min_v: f32,
    /// Maximum Value (Brightness) normalized [0.0, 1.0].
    pub max_v: f32,
}

/// A command that filters an image based on a specific HSV color range.
///
/// Pixels falling outside the provided [`HsvRange`] are masked
/// out by setting to black.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::{ColorFilterCommand, HsvRange};
/// let range = HsvRange {
///     min_h: 0.0,   max_h: 30.0, // Red tones
///     min_s: 0.5,   max_s: 1.0,
///     min_v: 0.5,   max_v: 1.0,
/// };
///
/// let command = ColorFilterCommand { range };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct ColorFilterCommand {
    /// The HSV color bounds to be preserved by the filter.
    pub range: HsvRange,
}

impl ImageAlgorithm for ColorFilterCommand {
    /// Filters an RGB image based on HSV color ranges, outputting a grayscale mask.
    ///
    /// Pixels within the [`HsvRange`] are preserved as their original luminance (or white),
    /// while pixels outside the range are set to black (0.0).
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Rgb`
    /// * **Output:** `F32Gray` (via `ctx.scratch_pad`)
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input is not RGB or
    /// if the scratch pad is not a single-channel grayscale buffer.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        // Get the input image (must be RGB)
        let input = match &ctx.image {
            ImageContainer::F32Rgb(img) => img,
            _ => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Rgb".into(),
                    found: format!("{:?}", ctx.image),
                });
            }
        };

        // Prepare the scratch pad (must be F32Gray and same size)
        // If the scratch pad is already the right size/type, this is essentially free.
        if !matches!(ctx.scratch_pad, ImageContainer::F32Gray(_))
            || ctx.scratch_pad.size() != input.size()
        {
            ctx.scratch_pad = ImageContainer::F32Gray(ManagedImage {
                data: Image::<f32, 1, CpuAllocator>::from_size_val(input.size(), 0.0, CpuAllocator)
                    .map_err(|_| {
                        InternalErrors::AllocationError("Failed to resize scratch pad".into())
                    })?,
                tile_offset: input.tile_offset.clone(),
                plane: input.plane.clone(),
            });
        }

        // Get a mutable reference to the underlying gray buffer
        let output_gray = match &mut ctx.scratch_pad {
            ImageContainer::F32Gray(img) => img,
            _ => unreachable!(), // We just ensured it's F32Gray above
        };

        // Process Pixels (unchanged logic, but writing to existing buffer)
        for y in 0..input.height() {
            for x in 0..input.width() {
                let r = *input
                    .get_pixel(x, y, 0)
                    .map_err(InternalErrors::from_kornia)?;
                let g = *input
                    .get_pixel(x, y, 1)
                    .map_err(InternalErrors::from_kornia)?;
                let b = *input
                    .get_pixel(x, y, 2)
                    .map_err(InternalErrors::from_kornia)?;

                let (h, s, v) = rgb_to_hsv(r, g, b);

                // If min > max, we are looking for the "edges" of the circle (e.g. 350 to 10)
                let hue_match = if self.range.min_h <= self.range.max_h {
                    h >= self.range.min_h && h <= self.range.max_h
                } else {
                    h >= self.range.min_h || h <= self.range.max_h
                };

                let in_range = hue_match
                    && s >= self.range.min_s
                    && s <= self.range.max_s
                    && v >= self.range.min_v
                    && v <= self.range.max_v;

                output_gray
                    .set_pixel(x, y, 0, if in_range { v } else { 0.0 })
                    .map_err(InternalErrors::from_kornia)?;
            }
        }

        // This makes the F32Gray output the new 'active' image for the next command
        ctx.swap()?;

        Ok(())
    }

    fn name(&self) -> &'static str {
        "HsvColorFilter"
    }
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let min = r.min(g).min(b);
    let max = r.max(g).max(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let mut h = if delta > 0.0 {
        if max == r {
            ((g - b) / delta) % 6.0
        } else if max == g {
            ((b - r) / delta) + 2.0
        } else {
            ((r - g) / delta) + 4.0
        }
    } else {
        0.0
    };

    h *= 60.0;
    if h < 0.0 {
        h += 360.0;
    }

    (h, s, v)
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelSizes;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_cache::ImageCache;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn test_hsv_filter_green_detection() {
        // 1. Setup: Create a 2x1 RGB image
        // Pixel 0: Pure Green [0.0, 1.0, 0.0] -> Should stay
        // Pixel 1: Pure Blue  [0.0, 0.0, 1.0] -> Should turn black
        let mut img = Image::<f32, 3, CpuAllocator>::from_size_val(
            ImageSize {
                width: 2,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )
        .unwrap();

        // Set pixel (0,0) to Green
        img.set_pixel(0, 0, 0, 0.0).unwrap();
        img.set_pixel(0, 0, 1, 1.0).unwrap();
        img.set_pixel(0, 0, 2, 0.0).unwrap();

        // Set pixel (1,0) to Blue
        img.set_pixel(1, 0, 0, 0.0).unwrap();
        img.set_pixel(1, 0, 1, 0.0).unwrap();
        img.set_pixel(1, 0, 2, 1.0).unwrap();

        // Initialize Context
        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 2,
                    height: 1,
                },
                full_image_width: ImageSize {
                    width: 2,
                    height: 1,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_rgb_from_image_test(img),
        )
        .unwrap();

        let mut cache = PipelineCache::default();

        // 2. Define Filter: Target Green (Hue around 120)
        let filter = ColorFilterCommand {
            range: HsvRange {
                min_h: 100.0,
                max_h: 140.0,
                min_s: 0.5,
                max_s: 1.0,
                min_v: 0.5,
                max_v: 1.0,
            },
        };

        // 3. Execute
        filter
            .execute(&mut ctx, &mut cache)
            .expect("Execution failed");

        // 4. Verify results
        match &ctx.image {
            ImageContainer::F32Gray(out_img) => {
                assert_eq!(out_img.width(), 2);
                assert_eq!(out_img.height(), 1);

                // Green pixel should be visible (Value ≈ 1.0)
                let green_result = *out_img.get_pixel(0, 0, 0).unwrap();
                assert!(green_result > 0.9);

                // Blue pixel should be black (0.0)
                let blue_result = *out_img.get_pixel(1, 0, 0).unwrap();
                assert_eq!(blue_result, 0.0);
            }
            _ => panic!("Output should be F32Gray"),
        }
    }

    #[test]
    fn test_hue_wrap_around_red() {
        let mut img = Image::<f32, 3, CpuAllocator>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )
        .unwrap();
        // Set to a "Reddish" color (Hue ~355)
        img.set_pixel(0, 0, 0, 1.0).unwrap();
        img.set_pixel(0, 0, 1, 0.0).unwrap();
        img.set_pixel(0, 0, 2, 0.1).unwrap();

        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 1,
                    height: 1,
                },
                full_image_width: ImageSize {
                    width: 1,
                    height: 1,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_rgb_from_image_test(img),
        )
        .unwrap();

        let mut cache = PipelineCache::default();

        // Range that wraps: 350 to 10 degrees
        let filter = ColorFilterCommand {
            range: HsvRange {
                min_h: 350.0,
                max_h: 10.0,
                min_s: 0.0,
                max_s: 1.0,
                min_v: 0.0,
                max_v: 1.0,
            },
        };

        filter.execute(&mut ctx, &mut cache).unwrap();

        if let ImageContainer::F32Gray(out_img) = &ctx.image {
            assert!(
                *out_img.get_pixel(0, 0, 0).unwrap() > 0.0,
                "Red pixel should have passed wrap-around filter"
            );
        }
    }

    #[test]
    fn test_color_filter_format_mismatch_error() {
        // 1. Create a 1x1 Grayscale image (Unsupported input type for Color Filter)
        let img = Image::<f32, 1, CpuAllocator>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.5,
            CpuAllocator,
        )
        .unwrap();

        // Initialize Context with F32Gray
        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 1,
                    height: 1,
                },
                full_image_width: ImageSize {
                    width: 1,
                    height: 1,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_gray_from_image_test(img),
        )
        .unwrap();

        let mut cache = PipelineCache::default();

        // 2. Setup the command
        let filter = ColorFilterCommand {
            range: HsvRange {
                min_h: 0.0,
                max_h: 360.0,
                min_s: 0.0,
                max_s: 1.0,
                min_v: 0.0,
                max_v: 1.0,
            },
        };

        // 3. Execute
        let result = filter.execute(&mut ctx, &mut cache);

        // 4. Assert error
        match result {
            Err(InternalErrors::FormatMismatch { expected, found }) => {
                assert_eq!(expected, "F32Rgb");
                // The 'found' string depends on your Debug impl for ImageContainer
                assert!(found.contains("F32Gray"));
            }
            _ => panic!("Expected FormatMismatch error, but got {:?}", result),
        }
    }

    #[test]
    fn test_name() {
        let extractor = ColorFilterCommand {
            range: HsvRange {
                min_h: 0.0,
                max_h: 0.0,
                min_s: 0.0,
                max_s: 0.0,
                min_v: 0.0,
                max_v: 0.0,
            },
        };
        let name = extractor.name();
        assert_eq!(name, "HsvColorFilter");
    }
}
