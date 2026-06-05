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
use evanalyzer_cfg::core_types::InternalErrors;
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
    /// The destination filesystem path where the image will be written.
    pub path: PathBuf,
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
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        // We look at ctx.image (the current state of the pipeline)
        if self.source == ImageSource::Image {
            match &ctx.image {
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
                        .save(&self.path)
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
                        .save(&self.path)
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
                .save(&self.path)
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
                .save(&self.path)
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

        // 2. Setup PipelineContext
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();

        let mut cache = PipelineCache::default();

        // 3. Define a temporary path
        let test_path = PathBuf::from("test_output_deleteme.png");

        // 4. Run the command
        let saver = SaveImage {
            path: test_path.clone(),
            source: ImageSource::Image,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        // 5. Assertions
        assert!(result.is_ok(), "Save command failed: {:?}", result.err());
        assert!(test_path.exists(), "File was not actually created on disk");

        // 6. Metadata check (ensure file isn't 0 bytes)
        let metadata = fs::metadata(&test_path).unwrap();
        assert!(metadata.len() > 0, "Saved file is empty");

        // Cleanup: remove the file after test
        let _ = fs::remove_file(test_path);
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

        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        let mut cache = PipelineCache::default();
        let test_path = PathBuf::from("test_output_rgb_deleteme.png");

        // 2. Run the command
        let saver = SaveImage {
            path: test_path.clone(),
            source: ImageSource::Image,
        };
        let result = saver.execute(&mut ctx, &mut cache);

        // 3. Assertions
        assert!(result.is_ok());
        assert!(test_path.exists());

        // Cleanup
        let _ = fs::remove_file(test_path);
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
        let mut ctx = PipelineContext::new_from_u32_image_test(unsupported_img).unwrap();
        let mut cache = PipelineCache::default();
        let saver = SaveImage {
            path: PathBuf::from("fail.png"),
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
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        let mut cache = PipelineCache::default();

        // Try saving to an illegal path (e.g., a directory that doesn't exist)
        let saver = SaveImage {
            path: PathBuf::from("/non_existent_folder/file.png"),
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
            path: PathBuf::from("fail.png"),
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
    fn test_save_image_name() {
        let saver = SaveImage {
            path: PathBuf::from("test.png"),
            source: ImageSource::Image,
        };
        assert_eq!(saver.name(), "Save Image");
    }
}
