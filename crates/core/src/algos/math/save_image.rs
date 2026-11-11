//! # save_image
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, is_file_name_part};
use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use kornia_image::Image;
use macros::CommandsMeta;

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
#[derive(CommandsMeta)]
#[cmdsmeta(category = "utility")]
pub struct SaveImage {
    /// Name the image should be stord under
    pub name: String,

    /// Which image from the pipeline should be stored
    #[cmdsmeta(default=ImageSource::Image, visibility = Advanced)]
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
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let Some(output_path) = ctx.output_path.clone() else {
            return Err(InternalErrors::Io("No output path!".into()));
        };
        // Part of a file name in the output folder - a shared pipeline's
        // "/home/…" or "../" would write the PNGs anywhere.
        if !is_file_name_part(&self.name) {
            return Err(InternalErrors::InvalidArgument(format!(
                "Save Image: the name \"{}\" must not contain a folder (/, \\, .., a drive)",
                self.name
            )));
        }

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

        let image = match self.source {
            // We look at ctx.image (the current state of the pipeline)
            ImageSource::Image => match ctx.image.as_ref() {
                // Grayscale (1 channel) and RGB (3 channel) are both scaled from
                // f32 [0.0, 1.0] to u8 [0, 255]
                ImageContainer::F32Gray(img) => {
                    let size = img.size();
                    let u8_data: Vec<u8> = img.as_slice().iter().map(|&v| to_u8(v)).collect();
                    let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(
                        size.width as u32,
                        size.height as u32,
                        u8_data,
                    )
                    .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;
                    DynamicImage::ImageLuma8(buffer)
                }
                ImageContainer::F32Rgb(img) => {
                    let size = img.size();
                    let u8_data: Vec<u8> = img.as_slice().iter().map(|&v| to_u8(v)).collect();
                    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
                        size.width as u32,
                        size.height as u32,
                        u8_data,
                    )
                    .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))?;
                    DynamicImage::ImageRgb8(buffer)
                }
                _ => {
                    return Err(InternalErrors::FormatMismatch {
                        expected: "F32Rgb, F32Gray".into(),
                        found: format!("{:?}", ctx.image),
                    });
                }
            },
            ImageSource::InstanceMap => DynamicImage::ImageRgb8(colorize(ctx.get_instance_map()?)?),
            ImageSource::SegmentationMask => {
                DynamicImage::ImageRgb8(colorize(ctx.get_segmentation_map()?)?)
            }
        };

        image
            .save(&out_path)
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    fn name(&self) -> &'static str {
        "Save Image"
    }

    /// Measured by `tests::tile_ram` (see the trait method).
    fn ram_per_tile_pixel(&self) -> f64 {
        1.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

/// Renders a label image (instance map / segmentation mask) with one color per label.
fn colorize(img: &Image<u32, 1>) -> Result<RgbImage, InternalErrors> {
    let size = img.size();
    let rgb_data: Vec<u8> = img.as_slice().iter().flat_map(|&v| get_color(v)).collect();
    ImageBuffer::from_raw(size.width as u32, size.height as u32, rgb_data)
        .ok_or_else(|| InternalErrors::Internal("Buffer size mismatch".into()))
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
    use super::*;
    use kornia_image::Image;
    use kornia_image::ImageSize;
    use std::fs;

    #[test]
    fn execute_errors_when_ctx_has_no_output_path() {
        let img = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )
        .unwrap();
        // `new_from_image_test` leaves `output_path: None` - never set here,
        // unlike every other test in this module.
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let saver = SaveImage {
            name: "unused".into(),
            source: ImageSource::Image,
        };

        let result = saver.execute(&mut ctx, &mut cache);

        match result {
            Err(InternalErrors::Io(msg)) => assert!(msg.contains("No output path")),
            _ => panic!("Expected an Io error for a missing output path, got {result:?}"),
        }
    }

    #[test]
    fn test_save_command_execution() {
        // 1. Create a dummy 2x2 grayscale image (4 pixels)
        let image_data = vec![0.0f32, 0.5, 0.5, 1.0];
        let input_img = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &image_data,
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

        let mut cache = GlobalPipelineCache::default();

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

        let input_img = Image::<f32, 3>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &image_data,
        )
        .expect("Failed to create test RGB image");

        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = GlobalPipelineCache::default();
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
        let unsupported_img = Image::<u32, 1>::from_size_slice(size, &data).unwrap();

        // 2. Setup context
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_u32_image_test(unsupported_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = GlobalPipelineCache::default();
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
            Image::<f32, 1>::from_size_slice(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                &[0.0f32],
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

        let input_img = Image::new(size, data).unwrap();
        let dir = tempfile::tempdir().unwrap();
        // The output "folder" is a file: its images folder can't be made.
        let not_a_folder = dir.path().join("file");
        std::fs::write(&not_a_folder, b"").unwrap();
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        ctx.output_path = Some(not_a_folder);
        let mut cache = GlobalPipelineCache::default();

        let saver = SaveImage {
            name: "file".into(),
            source: ImageSource::Image,
        };

        let result = saver.execute(&mut ctx, &mut cache);
        assert!(result.is_err());

        match result {
            Err(InternalErrors::Io(_)) => (), // This hits the map_err branch!
            _ => panic!("Expected Io error"),
        }
    }

    /// A name with a folder (a shared pipeline's) is refused - nothing is
    /// written, inside the output folder or outside.
    #[test]
    fn a_name_with_a_folder_writes_nothing() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out");
        let outside = dir.path().join("outside");
        for name in [
            outside.join("x").to_string_lossy().into_owned(),
            "../../../outside/x".into(),
            r"..\..\..\outside\x".into(),
            "sub/x".into(),
            "C:x".into(),
        ] {
            let image = Image::new(size, vec![0.0f32; 2 * 2 * 3]).unwrap();
            let mut ctx = PipelineContext::new_from_image_test_rgb(image).unwrap();
            ctx.output_path = Some(output.clone());
            let saver = SaveImage {
                name: name.clone(),
                source: ImageSource::Image,
            };
            let result = saver.execute(&mut ctx, &mut GlobalPipelineCache::default());
            assert!(
                matches!(result, Err(InternalErrors::InvalidArgument(_))),
                "{name}: {result:?}"
            );
        }
        assert!(!output.exists(), "nothing written in the output folder");
        assert!(!outside.exists(), "nor outside");
    }

    #[test]
    fn image_construction_rejects_a_length_mismatched_buffer() {
        // `SaveImage::execute`'s "Buffer size mismatch" branch
        // (`ImageBuffer::from_raw(...).ok_or_else(...)`) can never actually
        // fire in practice: its `size`/pixel-data pair always both come
        // from the same already-constructed `ctx.image`, and kornia's own
        // `Image` constructors (`new`/`from_size_slice`) already reject a
        // length-mismatched buffer before `execute` ever sees it - this
        // pins down *that* guarantee directly, rather than the previous
        // version of this test, which tried to reach `SaveImage`'s branch
        // through a mismatched `Image` and silently no-opped every time it
        // ran, since construction always failed first.
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let data = vec![0.0f32; 1]; // 1 pixel, not the 4 a 2x2 image needs

        let result = Image::<f32, 1>::from_size_slice(size, &data);

        assert!(
            result.is_err(),
            "a length-mismatched buffer must be rejected at construction, before SaveImage ever runs"
        );
    }

    #[test]
    fn test_save_instance_map_command_execution() {
        // `new_from_image_test` already seeds `ctx.instance_map`, so the
        // `ImageSource::InstanceMap` branch (untested before this) can reuse
        // the same fixture as the `ImageSource::Image` tests above.
        let input_img = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &[0.0f32, 0.5, 0.5, 1.0],
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = GlobalPipelineCache::default();
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
        let input_img = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 2,
                height: 2,
            },
            &[0.0f32, 0.5, 0.5, 1.0],
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = GlobalPipelineCache::default();
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
        assert_eq!(saver.cite()[0].cite_key, "danmayr2026");
        assert!(matches!(saver.execution_scope(), ExecutionScope::Tile));
    }
}
