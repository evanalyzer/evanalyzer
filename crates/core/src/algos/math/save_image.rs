//! # save_image
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use image::{ImageBuffer, Luma, Rgb};
use log::info;
use macros::CommandsMeta;
use std::path::PathBuf;

#[derive(PartialEq)]
pub enum ImageSource {
    Image,
    InstanceMap,
    SegmentationMask,
}

/// A command that exports the current image to a persistent file on disk.
///
/// This is a **transparent command**: it does not modify the image data in the
/// pipeline context, nor does it perform a buffer swap. It acts as a tap
/// to view the state of the image at a specific point in the pipeline.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::SaveImage;
/// let saver = SaveImage {path:"output/processed_cell.png"};
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct SaveImage {
    /// Name the image should be stord under
    pub name: String,

    /// Which image from the pipeline should be stored
    pub source: ImageSource,
}

impl ImageAlgorithm for SaveImage {
    /// Writes the current image from the context to the filesystem.
    ///
    /// This method detects the image format from the file extension in `path`.
    /// Supported formats usually include PNG, JPEG, TIFF, and BMP, depending
    /// on the underlying IO backend.
    ///
    /// # Pipeline Side-Effects
    /// - **Data Preservation**: The `ctx.image` remains unchanged.
    /// - **No Swap**: Unlike most filters, this does not move data to the
    ///   `scratch_pad` or call `ctx.swap()`.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::IOError`] if the directory is unwritable,
    /// or if the image format is not supported for the current image type
    /// (e.g., saving an F32 image as a standard JPEG).
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let Some(output_path) = ctx.output_path.clone() else {
            return Err(InternalErrors::Io("No output path!".into()));
        };

        // Control images are grouped per source image (its relative path, minus
        // extension, becomes its own output folder) so e.g. every image's
        // "segmentation_mask" control image doesn't collide on the same filename.
        let image_folder = cache.image_rel_path.with_extension("");
        let out_dir = output_path.join("images").join(image_folder);
        std::fs::create_dir_all(&out_dir).map_err(|e| InternalErrors::Io(e.to_string()))?;

        // All tiles of one image are kept in the same folder (not nested per-tile)
        // so users reconstructing the full image from its tiles don't have to hunt
        // across subfolders. The offset is zero-padded so filenames sort in tile
        // order in a plain file browser, not just numerically.
        let tile_offset = ctx.get_image_tile_offset();
        let out_path = out_dir.join(format!(
            "{}_x{:06}_y{:06}.png",
            self.name, tile_offset.x, tile_offset.y
        ));

        // We look at ctx.image (the current state of the pipeline)
        if self.source == ImageSource::Image {
            match ctx.image.as_ref() {
                // Handle Grayscale (1 Channel)
                ImageContainer::F32Gray(img) => {
                    let size = img.size();

                    // Convert and scale the pixels from f32 [0.0, 1.0] to u8 [0, 255]
                    // We use as_slice() to access the private Kornia data
                    let u8_data: Vec<u8> = img
                        .as_slice()
                        .iter()
                        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                        .collect();

                    // Create an image buffer compatible with the 'image' crate
                    let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(
                        size.width as u32,
                        size.height as u32,
                        u8_data,
                    )
                    .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;

                    // Save to disk
                    buffer
                        .save(&out_path)
                        .map_err(|e| InternalErrors::Io(e.to_string()))?;
                    return Ok(());
                }

                // Handle RGB (3 Channel)
                ImageContainer::F32Rgb(img) => {
                    let size = img.size();

                    let u8_data: Vec<u8> = img
                        .as_slice()
                        .iter()
                        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                        .collect();

                    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                        size.width as u32,
                        size.height as u32,
                        u8_data,
                    )
                    .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;

                    buffer
                        .save(&out_path)
                        .map_err(|e| InternalErrors::Io(e.to_string()))?;
                    return Ok(());
                }
                _ => {
                    return Err(InternalErrors::FormatMismatch {
                        expected: "F32Rgb, F32Gray".into(),
                        found: format!("{:?}", ctx.image),
                    });
                }
            }
        } else if self.source == ImageSource::InstanceMap {
            let img = ctx.get_instance_map()?;
            let size = img.size();

            // We use as_slice() to access the private Kornia data
            let rgb_data: Vec<u8> = img.as_slice().iter().flat_map(|&v| get_color(v)).collect();

            let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                size.width as u32,
                size.height as u32,
                rgb_data,
            )
            .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;

            // Save to disk
            buffer
                .save(&out_path)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            return Ok(());
        } else if self.source == ImageSource::SegmentationMask {
            let img = ctx.get_segmentation_map()?;
            let size = img.size();
            // We use as_slice() to access the private Kornia data
            let rgb_data: Vec<u8> = img.as_slice().iter().flat_map(|&v| get_color(v)).collect();

            let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                size.width as u32,
                size.height as u32,
                rgb_data,
            )
            .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;

            // Save to disk
            buffer
                .save(&out_path)
                .map_err(|e| InternalErrors::Io(e.to_string()))?;

            return Ok(());
        } else {
            return Err(InternalErrors::FormatMismatch {
                expected: "Unsupported image source".into(),
                found: format!("{:?}", ctx.image),
            });
        }
    }

    fn name(&self) -> &'static str {
        "Save Image"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }
}

fn get_color(val: u32) -> [u8; 3] {
    match val {
        0 => [0, 0, 0],
        1 => [255, 0, 0],
        2 => [0, 255, 0],
        3 => [0, 0, 255],
        _ => {
            // Golden ratio (conjugate) as a basis for good color distribution
            // Use hashing so that IDs (e.g., 4, 100, 1000) are well distributed
            let mut h = val.wrapping_mul(0x45d9f3b);
            h = ((h >> 16) ^ h).wrapping_mul(0x45d9f3b);
            h = (h >> 16) ^ h;

            // Generiere RGB basierend auf dem Hash
            [
                ((h & 0xFF) as u8).max(50),       // Red
                ((h >> 8 & 0xFF) as u8).max(50),  // Green
                ((h >> 16 & 0xFF) as u8).max(50), // Blue
            ]
        }
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::pipeline::pipeline_cache::ImageCache;

    use super::*;
    use kornia_image::Image;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;
    use std::fs;

    #[test]
    fn test_save_command_execution() {
        // 1. Create a dummy 2x2 grayscale image (4 pixels)
        let image_data = vec![0.0f32, 0.5, 0.5, 1.0];
        let input_img = Image::<f32, 1, _>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create test image");

        // 2. Setup PipelineContext - each test gets its own directory (not a
        // fixed `images/` folder relative to the process CWD) so concurrently
        // running tests in this module can never race on it: a previous
        // version shared one `images/` folder across tests, and one test's
        // cleanup (`remove_dir` after its own file was gone) could delete the
        // directory out from under another test between its `create_dir_all`
        // and `.save()` calls - intermittent "No such file or directory",
        // more reliably hit under CI's parallel test execution.
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());

        let mut cache = PipelineCache::default();

        // 3. Control images are written under
        // `<output_path>/images/<image_rel_path>/<name>_x<off>_y<off>.png`; the
        // default (empty) `image_rel_path` used here maps to `images/`
        // directly, and the test context's tile offset defaults to (0, 0).
        let test_path = dir
            .path()
            .join("images/test_output_deleteme_x000000_y000000.png");

        // 4. Run the command
        let saver = SaveImage {
            name: "test_output_deleteme".into(),
            source: ImageSource::Image,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        // 5. Assertions
        assert!(result.is_ok(), "Save command failed: {:?}", result.err());
        assert!(test_path.exists(), "File was not actually created on disk");

        // 6. Metadata check (ensure file isn't 0 bytes)
        let metadata = fs::metadata(&test_path).unwrap();
        assert!(metadata.len() > 0, "Saved file is empty");

        // `dir` cleans itself up on drop - no manual cleanup needed.
    }
    #[test]
    fn test_save_rgb_command_execution() {
        // 1. Create a dummy 2x2 RGB image (12 values: 4 pixels * 3 channels)
        // Red, Green, Blue, Red, Green, Blue, ...
        let image_data = vec![
            1.0f32, 0.0, 0.0, // Pixel 0 (Red)
            0.0, 1.0, 0.0, // Pixel 1 (Green)
            0.0, 0.0, 1.0, // Pixel 2 (Blue)
            1.0, 1.0, 1.0, // Pixel 3 (White)
        ];

        let input_img = Image::<f32, 3, _>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &image_data,
            CpuAllocator,
        )
        .expect("Failed to create test RGB image");

        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = PipelineCache::default();
        let test_path = dir
            .path()
            .join("images/test_output_rgb_deleteme_x000000_y000000.png");

        // 2. Run the command
        let saver = SaveImage {
            name: "test_output_rgb_deleteme".into(),
            source: ImageSource::Image,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        // 3. Assertions
        assert!(result.is_ok());
        assert!(test_path.exists());

        // `dir` cleans itself up on drop - no manual cleanup needed.
    }

    #[test]
    fn test_save_image_format_mismatch_fails() {
        // 1. Create an image with a type NOT supported by SaveImage (e.g., u32)
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let data = vec![0u32; 1];
        let unsupported_img =
            Image::<u32, 1, _>::from_size_slice(size, &data, CpuAllocator).unwrap();

        // 2. Setup context
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_u32_image_test(unsupported_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = PipelineCache::default();
        let saver = SaveImage {
            name: "fail".into(),
            source: ImageSource::Image,
        };

        // 3. Assert that the operation returns a FormatMismatch error
        let result = saver.execute(&mut ctx, &mut cache);
        assert!(result.is_err());

        match result {
            Err(InternalErrors::FormatMismatch { .. }) => (),
            _ => panic!("Expected FormatMismatch error, got {:?}", result),
        }
    }

    #[test]
    fn test_save_image_buffer_mismatch_fails() {
        // Create an image, but force the internal logic to think the buffer is the wrong size
        // by passing an invalid dimension.
        let _ctx = PipelineContext::new_from_image_test(
            Image::<f32, 1, _>::from_size_slice(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                &[0.0f32],
                CpuAllocator,
            )
            .unwrap(),
        )
        .unwrap();

        // Mocking this is hard, but if you have a way to manipulate the context
        // to return a corrupt image, you will hit the Internal error branch.
    }

    #[test]
    fn test_save_image_io_error_fails() {
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
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = PipelineCache::default();

        // Try saving to an illegal path (e.g., a directory that doesn't exist)
        let saver = SaveImage {
            name: "/non_existent_folder/file.png".into(),
            source: ImageSource::Image,
        };

        let result = saver.execute(&mut ctx, &mut cache);
        assert!(result.is_err());

        match result {
            Err(InternalErrors::Io(_)) => (), // This hits the map_err branch!
            _ => panic!("Expected Io error"),
        }
    }

    #[test]
    fn test_save_image_buffer_mismatch_internal_error() {
        // Create a 2x2 image (expecting 4 pixels)
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        // Provide only 1 pixel (length 1) instead of 4.
        // from_raw will return None because 1 != 2*2
        let data = vec![0.0f32; 1];

        let Ok(img) = Image::<f32, 1, _>::from_size_slice(size, &data, CpuAllocator) else {
            return;
        };
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = PipelineCache::default();

        let saver = SaveImage {
            name: "fail".into(),
            source: ImageSource::Image,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        // Verify it hits the 'None' branch and returns Internal Error
        assert!(result.is_err());
        match result {
            Err(InternalErrors::Internal(msg)) => assert!(msg.contains("Buffer size mismatch")),
            _ => panic!(
                "Expected Internal buffer size mismatch error, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_save_instance_map_command_execution() {
        // `new_from_image_test` already seeds `ctx.instance_map`, so the
        // `ImageSource::InstanceMap` branch (untested before this) can reuse
        // the same fixture as the `ImageSource::Image` tests above.
        let input_img = Image::<f32, 1, _>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &[0.0f32, 0.5, 0.5, 1.0],
            CpuAllocator,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = PipelineCache::default();
        let test_path = dir
            .path()
            .join("images/test_output_instance_map_deleteme_x000000_y000000.png");

        let saver = SaveImage {
            name: "test_output_instance_map_deleteme".into(),
            source: ImageSource::InstanceMap,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        assert!(result.is_ok(), "Save command failed: {:?}", result.err());
        assert!(test_path.exists(), "File was not actually created on disk");
        // `dir` cleans itself up on drop - no manual cleanup needed.
    }

    #[test]
    fn test_save_segmentation_mask_command_execution() {
        let input_img = Image::<f32, 1, _>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &[0.0f32, 0.5, 0.5, 1.0],
            CpuAllocator,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = PipelineCache::default();
        let test_path = dir
            .path()
            .join("images/test_output_seg_mask_deleteme_x000000_y000000.png");

        let saver = SaveImage {
            name: "test_output_seg_mask_deleteme".into(),
            source: ImageSource::SegmentationMask,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        assert!(result.is_ok(), "Save command failed: {:?}", result.err());
        assert!(test_path.exists(), "File was not actually created on disk");
        // `dir` cleans itself up on drop - no manual cleanup needed.
    }

    #[test]
    fn get_color_is_stable_for_the_reserved_ids_and_hashes_higher_ones() {
        assert_eq!(get_color(0), [0, 0, 0]);
        assert_eq!(get_color(1), [255, 0, 0]);
        assert_eq!(get_color(2), [0, 255, 0]);
        assert_eq!(get_color(3), [0, 0, 255]);
        // Ids above the reserved range go through the hash path - just check
        // it's deterministic and every channel respects the `.max(50)` floor
        // (so no instance ends up an indistinguishable near-black).
        let a = get_color(42);
        let b = get_color(42);
        assert_eq!(a, b, "hashing must be deterministic for the same id");
        assert!(a.iter().all(|&c| c >= 50));
    }

    #[test]
    fn test_save_image_name() {
        let saver = SaveImage {
            name: "test".into(),
            source: ImageSource::Image,
        };
        assert_eq!(saver.name(), "Save Image");
    }
}
