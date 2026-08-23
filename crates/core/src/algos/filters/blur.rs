//! # blur
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::f32::consts::E;
use std::sync::Arc;

use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::box_blur;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;

/// Smooths an image by averaging pixel intensities within a local neighborhood.
///
/// This algorithm applies a uniform box filter where every pixel within the moving
/// window contributes equally to the final value. It is a computationally fast
/// method used for general image smoothing, blending variations, and rapid noise
/// suppression where edge precision is less critical.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct Blur {
    /// The size of the blur matrix.
    ///
    /// Must be an odd number (e.g., 3, 5, 7)
    #[cmdsmeta(
        default = 3,
        min = 3,
        max = 27,
        rename = "kernel_size",
        display_name = "Kernel size",
        summary = true,
        step = 2
    )]
    pub kernel_size: usize,
}

impl ImageAlgorithm for Blur {
    /// Applies a spatial box blur to the current image context.
    ///
    /// This implementation performs the blur in-place (via a scratch pad)
    /// and automatically swaps the buffers upon successful completion.
    ///
    /// ### Supported Image Types
    /// * `F32Gray` - Single channel floating point.
    /// * `F32Rgb` - Three channel floating point.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the image and scratch pad types
    /// do not align or are not supported by the box blur kernel.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.kernel_size % 2 == 0 {
            return Err(InternalErrors::Internal("kernel_size must be odd".into()));
        }

        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            // Handle Grayscale (1 Channel)
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                output.data = Self::box_blur_edge_replicate(input, self.kernel_size)?;
                ctx.swap()?;
                Ok(())
            }
            // Handle RGB (3 Channel)
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                output.data = Self::box_blur_edge_replicate(input, self.kernel_size)?;
                ctx.swap()?;

                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Rgb or F32Gray".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Blur"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }
}

impl Blur {
    /// Box-blurs `src` with edge-replicated borders.
    ///
    /// `kornia_imgproc::filter::box_blur` always zero-pads at the image
    /// border ("NOTE: This function uses a constant border type", confirmed
    /// empirically: a single bright corner pixel blurs to `1/9`, not the
    /// `4/9` edge-replicate would give). The reference (`docs/blur/blur.cpp`,
    /// `filter3x3`'s default `BLUR_MORE` mode, which is what a bare `{"type":
    /// "blur"}` step actually runs - `docs/blur/blur_settings.hpp` defaults
    /// `mode` to `BLUR_MORE`, not `GAUSSIAN`) replicates edge pixels instead,
    /// so this pads the image by `kernel_size / 2` on every side before
    /// blurring, then crops back to the original size - the zero-padding
    /// artifact falls entirely within the discarded padding.
    fn box_blur_edge_replicate<const C: usize>(
        src: &Image<f32, C, CpuAllocator>,
        kernel_size: usize,
    ) -> Result<Image<f32, C, CpuAllocator>, InternalErrors> {
        let pad = kernel_size / 2;
        let padded_in = Self::pad_edge_replicate(src, pad)?;
        let padded_size = padded_in.size();
        let mut padded_out = Image::<f32, C, CpuAllocator>::new(
            padded_size,
            vec![0.0f32; padded_size.width * padded_size.height * C],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        box_blur(&padded_in, &mut padded_out, (kernel_size, kernel_size))
            .map_err(InternalErrors::from_kornia)?;
        Self::crop_center(&padded_out, src.width(), src.height(), pad)
    }

    /// Pads `src` by `pad` pixels on every side, clamping to the nearest
    /// edge pixel (replicate padding).
    fn pad_edge_replicate<const C: usize>(
        src: &Image<f32, C, CpuAllocator>,
        pad: usize,
    ) -> Result<Image<f32, C, CpuAllocator>, InternalErrors> {
        let (w, h) = (src.width(), src.height());
        let (pw, ph) = (w + 2 * pad, h + 2 * pad);
        let src_data = src.as_slice();
        let mut out = vec![0.0f32; pw * ph * C];
        for y in 0..ph {
            let sy = (y as isize - pad as isize).clamp(0, h as isize - 1) as usize;
            for x in 0..pw {
                let sx = (x as isize - pad as isize).clamp(0, w as isize - 1) as usize;
                let src_idx = (sy * w + sx) * C;
                let dst_idx = (y * pw + x) * C;
                out[dst_idx..dst_idx + C].copy_from_slice(&src_data[src_idx..src_idx + C]);
            }
        }
        Image::<f32, C, CpuAllocator>::new(
            ImageSize {
                width: pw,
                height: ph,
            },
            out,
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)
    }

    /// Crops the `width` x `height` region starting `pad` pixels in from
    /// `src`'s top-left corner.
    fn crop_center<const C: usize>(
        src: &Image<f32, C, CpuAllocator>,
        width: usize,
        height: usize,
        pad: usize,
    ) -> Result<Image<f32, C, CpuAllocator>, InternalErrors> {
        let pw = src.width();
        let src_data = src.as_slice();
        let mut out = vec![0.0f32; width * height * C];
        for y in 0..height {
            let sy = y + pad;
            for x in 0..width {
                let sx = x + pad;
                let src_idx = (sy * pw + sx) * C;
                let dst_idx = (y * width + x) * C;
                out[dst_idx..dst_idx + C].copy_from_slice(&src_data[src_idx..src_idx + C]);
            }
        }
        Image::<f32, C, CpuAllocator>::new(ImageSize { width, height }, out, CpuAllocator)
            .map_err(InternalErrors::from_kornia)
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelSizes;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_cache::ImageCache;
    use kornia_image::Image;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_box_blur_grayscale() {
        // 1. Setup a 5x5 image with a single white pixel in the center
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0.0f32; 25];
        data[12] = 1.0; // Center pixel (x=2, y=2)

        let input_img = Image::new(size, data, CpuAllocator).unwrap();

        let blur_cmd = Blur { kernel_size: 3 };

        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        let mut cache = PipelineCache::default();

        // 2. Execute
        blur_cmd
            .execute(&mut ctx, &mut cache)
            .expect("Blur execution failed");

        // 3. Verify results
        // Because of ctx.swap(), the result is now in ctx.image
        if let ImageContainer::F32Gray(result_img) = ctx.image.as_ref() {
            let pixels = result_img.as_slice();

            // In a 3x3 box blur, the energy of that 1.0 pixel
            // is spread over 9 pixels. 1.0 / 9.0 = 0.1111...
            let expected_val = 1.0 / 9.0;
            let center_val = pixels[12];
            let neighbor_val = pixels[11]; // one to the left
            let corner_val = pixels[0]; // top left (should be 0.0)

            // Check center and immediate neighbor
            assert!(
                (center_val - expected_val).abs() < 1e-5,
                "Center pixel value incorrect"
            );
            assert!(
                (neighbor_val - expected_val).abs() < 1e-5,
                "Neighbor pixel value incorrect"
            );

            // Check far corner (outside the 3x3 reach)
            assert_eq!(corner_val, 0.0, "Far corner should remain black");
        } else {
            panic!("Expected F32Gray in ctx.image after swap");
        }
    }

    /// The reference (`docs/blur/blur.cpp`, `filter3x3`) replicates edge
    /// pixels at the image border - a bare `{"type": "blur"}` step actually
    /// runs this path, since `docs/blur/blur_settings.hpp` defaults `mode`
    /// to `BLUR_MORE`, not `GAUSSIAN`. `kornia_imgproc::filter::box_blur`
    /// always zero-pads instead ("NOTE: This function uses a constant
    /// border type"), which would make every border pixel - and, for tiled
    /// processing, every tile seam - systematically darker than the
    /// reference. A single bright corner pixel makes the difference exact
    /// and unambiguous: zero-padding blurs it to `1/9`, edge-replicate to
    /// `4/9` (the corner's own value gets counted for all 5 out-of-bounds
    /// neighbor positions that clamp back onto it).
    #[test]
    fn test_box_blur_edge_replicates_at_border() {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0.0f32; 25];
        data[0] = 1.0; // top-left corner pixel

        let input_img = Image::new(size, data, CpuAllocator).unwrap();
        let blur_cmd = Blur { kernel_size: 3 };
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        let mut cache = PipelineCache::default();
        blur_cmd
            .execute(&mut ctx, &mut cache)
            .expect("Blur execution failed");

        if let ImageContainer::F32Gray(result_img) = ctx.image.as_ref() {
            let pixels = result_img.as_slice();
            let expected_corner = 4.0 / 9.0;
            assert!(
                (pixels[0] - expected_corner).abs() < 1e-5,
                "corner pixel: got {}, expected {} (edge-replicate); a value near {} means \
                 the border is being zero-padded instead",
                pixels[0],
                expected_corner,
                1.0 / 9.0
            );
        } else {
            panic!("Expected F32Gray in ctx.image after swap");
        }
    }

    #[test]
    fn test_box_blur_rgb() {
        // 1. Setup a 5x5 RGB image
        // We'll put 1.0 in the RED channel at the center,
        // and 0.0 everywhere else.
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        // RGB data length: width * height * 3
        let mut data = vec![0.0f32; 5 * 5 * 3];

        // Center pixel is at (x=2, y=2).
        // In a flat RGB array: index = (y * width + x) * 3
        let center_idx = (2 * 5 + 2) * 3;
        data[center_idx] = 1.0; // Red channel of center pixel
        // data[center_idx + 1] = 0.0; // Green (already 0)
        // data[center_idx + 2] = 0.0; // Blue (already 0)

        let input_img = Image::new(size, data, CpuAllocator).unwrap();
        let blur_cmd = Blur { kernel_size: 3 };

        // Create context with F32Rgb
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img)
            .expect("Failed to create RGB context");
        let mut cache = PipelineCache::default();

        // 2. Execute
        blur_cmd
            .execute(&mut ctx, &mut cache)
            .expect("RGB Blur execution failed");

        // 3. Verify
        if let ImageContainer::F32Rgb(result_img) = ctx.image.as_ref() {
            let pixels = result_img.as_slice();
            let expected_val = 1.0 / 9.0;

            // Check the RED channel at the center
            let center_red = pixels[center_idx];
            assert!(
                (center_red - expected_val).abs() < 1e-5,
                "Center RED channel value incorrect. Got: {}",
                center_red
            );

            // Check a neighbor's RED channel (should also have energy)
            let neighbor_red = pixels[center_idx - 3]; // one pixel left
            assert!(
                (neighbor_red - expected_val).abs() < 1e-5,
                "Neighbor RED channel value incorrect"
            );

            // CRITICAL: Check the GREEN channel at the center
            // It should still be 0.0 because the input green channel was empty.
            let center_green = pixels[center_idx + 1];
            assert_eq!(center_green, 0.0, "Green channel should remain 0.0");

            // Check a far corner RED channel (outside 3x3 kernel reach)
            assert_eq!(pixels[0], 0.0, "Far corner RED should remain 0.0");
        } else {
            panic!("Expected F32Rgb in ctx.image after swap");
        }
    }
    #[test]
    fn test_blur_format_mismatch() {
        // 1. Setup a 5x5 Gray image
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let data_gray = vec![0.0f32; 25];
        let gray_img = Image::new(size, data_gray, CpuAllocator).unwrap();

        // 2. Setup a 5x5 RGB image for the scratch pad
        let data_rgb = vec![0.0f32; 75];
        let rgb_img = Image::new(size, data_rgb, CpuAllocator).unwrap();

        // 3. Manually construct a context with mismatched buffers
        // Usually PipelineContext handles this, but we can force it for the test
        let mut ctx = PipelineContext {
            output_path: None,
            image: ImageContainer::new_f32_gray_from_image_test(gray_img).into(),
            scratch_pad: ImageContainer::new_f32_rgb_from_image_test(rgb_img).into(),
            instance_map: None,
            segmentation_map: None,
            image_meta: PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                },
                full_image_width: size,
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
        };

        let blur_cmd = Blur { kernel_size: 3 };
        let mut cache = PipelineCache::default();

        // 4. Execute and assert the specific error
        let result = blur_cmd.execute(&mut ctx, &mut cache);

        match result {
            Err(InternalErrors::FormatMismatch { expected, found }) => {
                assert_eq!(expected, "F32Rgb or F32Gray");
                assert!(
                    found.contains("F32Gray"),
                    "Error message should mention the actual image format"
                );
            }
            Ok(_) => panic!("Execution should have failed due to format mismatch"),
            Err(e) => panic!("Expected FormatMismatch error, but got: {:?}", e),
        }
    }

    #[test]
    fn test_blur_metadata() {
        let blur = Blur { kernel_size: 5 };
        assert_eq!(blur.name(), "Blur");

        // Cover the Debug implementation and the format! string
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let img = Image::new(size, vec![0.0f32], CpuAllocator).unwrap();
        let container = ImageContainer::new_f32_gray_from_image_test(img);

        // This ensures the code generated for Debug is executed
        let debug_str = format!("{:?}", container);
        assert!(debug_str.contains("F32Gray"));
    }

    #[test]
    fn zero_kernel_size_returns_error_instead_of_dividing_by_it() {
        // `kernel_size` is only validated as `min = 3` at the UI/schema layer
        // - a hand-edited or API-produced pipeline file could smuggle 0
        // through to `box_blur_edge_replicate`'s `pad = kernel_size / 2`.
        // Already covered structurally (0 is even, and the odd-kernel-size
        // check above rejects it), but pin it down explicitly rather than
        // relying on that being a coincidence of the current implementation.
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let img = Image::new(size, vec![0.0f32; 25], CpuAllocator).unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let blur_cmd = Blur { kernel_size: 0 };
        let mut cache = PipelineCache::default();

        let result = blur_cmd.execute(&mut ctx, &mut cache);
        assert!(result.is_err(), "kernel_size=0 should be rejected");
    }

    #[test]
    fn test_blur_kernel_error_path() {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let img = Image::new(size, vec![0.0f32; 25], CpuAllocator).unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();

        // Force an error: Use an even kernel size (most libs reject this)
        let blur_cmd = Blur { kernel_size: 2 };
        let mut cache = PipelineCache::default();

        let result = blur_cmd.execute(&mut ctx, &mut cache);

        // This triggers the '?' error branch for box_blur
        assert!(result.is_err());
    }

    #[test]
    fn test_blur_recovers_from_a_mismatched_scratch_pad() {
        // 1. Create a valid 5x5 Gray image
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let img = Image::new(size, vec![0.0f32; 25], CpuAllocator).unwrap();

        // 2. Create a 3x3 "Wrong" scratch pad
        let wrong_size = ImageSize {
            width: 3,
            height: 3,
        };
        let wrong_scratch = Image::new(wrong_size, vec![0.0f32; 9], CpuAllocator).unwrap();

        // 3. Manually build a context whose scratch pad doesn't match the
        // image size.
        let mut ctx = PipelineContext {
            output_path: None,
            image: ImageContainer::new_f32_gray_from_image_test(img).into(),
            scratch_pad: ImageContainer::new_f32_gray_from_image_test(wrong_scratch).into(), // Mismatched size!
            instance_map: None,
            segmentation_map: None,
            image_meta: PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                },
                full_image_width: size,
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
        };

        let blur_cmd = Blur { kernel_size: 3 };
        let mut cache = PipelineCache::default();

        // 4. Execute. `box_blur_edge_replicate` fully reconstructs its output
        // at the input's size (via pad-then-crop) rather than writing into
        // the scratch pad's pre-existing buffer, so a stale/mismatched
        // scratch pad size no longer matters.
        let result = blur_cmd.execute(&mut ctx, &mut cache);

        assert!(
            result.is_ok(),
            "Blur should recover from a mismatched scratch pad by reconstructing its own \
             correctly-sized output: {result:?}"
        );
        if let ImageContainer::F32Gray(result_img) = ctx.image.as_ref() {
            assert_eq!(result_img.size(), size);
        } else {
            panic!("Expected F32Gray in ctx.image after swap");
        }
    }
}
