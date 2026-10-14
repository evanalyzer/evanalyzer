//! # save_image
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::{ImageContainer, Point2d};
use crate::object::Object;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectClass};
use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use kornia_image::Image;
use macros::CommandsMeta;
use std::collections::HashMap;

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

    /// Shows the selected classes in the image
    #[cmdsmeta(display_name = "Display Classes")]
    pub display_classes: Vec<ObjectClass>,

    /// Which image from the pipeline should be stored
    #[cmdsmeta(default=ImageSource::Image, visibility = Advanced)]
    pub source: ImageSource,

    /// Opacity in percent of the filled `display_classes` objects (0 = only
    /// the contours, 100 = solid). Contours are always drawn opaque.
    #[cmdsmeta(default = 50, min = 0, max = 100, step = 5, display_name = "Opacity", visibility = Advanced)]
    pub opacity: u32,
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

        let image = if self.display_classes.is_empty() {
            image
        } else {
            let mut rgb = image.into_rgb8();
            self.draw_classes(&mut rgb, tile_offset, cache);
            DynamicImage::ImageRgb8(rgb)
        };

        image
            .save(&out_path)
            .map_err(|e| InternalErrors::Io(e.to_string()))
    }

    fn name(&self) -> &'static str {
        "Save Image"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl SaveImage {
    /// Paints every object of `display_classes` onto `image` (the current tile):
    /// the area filled in the class color with `opacity`, the contour opaque.
    ///
    /// All fills are painted before any contour, so an overlapping neighbour's
    /// fill never hides a contour. An object carrying several of the selected
    /// classes is drawn in the color of the first one in `display_classes`.
    fn draw_classes(
        &self,
        image: &mut RgbImage,
        tile_offset: Point2d,
        cache: &GlobalPipelineCache,
    ) {
        let alpha = self.opacity.min(100) as f32 / 100.0;
        let objects: Vec<(&Object, [u8; 3])> = cache
            .object_cache
            .values()
            .filter_map(|object| {
                let class = self
                    .display_classes
                    .iter()
                    .find(|class| object.has_object_class(class))?;
                Some((object, class_color(class, &cache.class_colors)))
            })
            .collect();

        for (object, color) in &objects {
            for_each_object_pixel(object, image, tile_offset, |pixel, is_contour| {
                if !is_contour {
                    for (c, &fill) in pixel.0.iter_mut().zip(color) {
                        *c = (*c as f32 * (1.0 - alpha) + fill as f32 * alpha).round() as u8;
                    }
                }
            });
        }
        for (object, color) in &objects {
            for_each_object_pixel(object, image, tile_offset, |pixel, is_contour| {
                if is_contour {
                    pixel.0 = *color;
                }
            });
        }
    }
}

/// Calls `f` for every mask pixel of `object` lying inside `image`, with
/// whether it is a contour pixel (a 4-neighbour is outside the mask).
///
/// The object's bbox is in whole-image coordinates; `tile_offset` is where
/// `image` sits in the whole image.
fn for_each_object_pixel(
    object: &Object,
    image: &mut RgbImage,
    tile_offset: Point2d,
    mut f: impl FnMut(&mut Rgb<u8>, bool),
) {
    let [x_min, y_min, x_max, y_max] = object.bbox.map(i64::from);
    let (tile_x, tile_y) = (tile_offset.x as i64, tile_offset.y as i64);
    let (width, height) = (image.width() as i64, image.height() as i64);
    if x_max < tile_x || y_max < tile_y || x_min >= tile_x + width || y_min >= tile_y + height {
        return;
    }

    let bbox_w = x_max - x_min + 1;
    let bbox_h = y_max - y_min + 1;
    let in_mask = |lx: i64, ly: i64| {
        lx >= 0
            && ly >= 0
            && lx < bbox_w
            && ly < bbox_h
            && object
                .mask_data
                .get((ly * bbox_w + lx) as usize)
                .is_some_and(|b| *b)
    };

    for ly in 0..bbox_h {
        let y = y_min + ly - tile_y;
        if y < 0 || y >= height {
            continue;
        }
        for lx in 0..bbox_w {
            let x = x_min + lx - tile_x;
            if x < 0 || x >= width || !in_mask(lx, ly) {
                continue;
            }
            let is_contour = !in_mask(lx - 1, ly)
                || !in_mask(lx + 1, ly)
                || !in_mask(lx, ly - 1)
                || !in_mask(lx, ly + 1);
            f(image.get_pixel_mut(x as u32, y as u32), is_contour);
        }
    }
}

/// The project color of `class`, or a generated one if the project has none for it.
fn class_color(class: &ObjectClass, class_colors: &HashMap<ObjectClass, u32>) -> [u8; 3] {
    match class_colors.get(class) {
        Some(&rgb) => [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8],
        None => get_color(class.to_u32().unwrap_or(0)),
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
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
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
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
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
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
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
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
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
        let mut ctx = PipelineContext::new_from_image_test_rgb(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = GlobalPipelineCache::default();

        // Try saving to an illegal path (e.g., a directory that doesn't exist)
        let saver = SaveImage {
            name: "/non_existent_folder/file.png".into(),
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
        };

        let result = saver.execute(&mut ctx, &mut cache);
        assert!(result.is_err());

        match result {
            Err(InternalErrors::Io(_)) => (), // This hits the map_err branch!
            _ => panic!("Expected Io error"),
        }
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
            display_classes: vec![],
            source: ImageSource::InstanceMap,
            opacity: 50,
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
            display_classes: vec![],
            source: ImageSource::SegmentationMask,
            opacity: 50,
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

    const RED_CLASS: ObjectClass = ObjectClass::Global(1);
    const OTHER_CLASS: ObjectClass = ObjectClass::Global(2);

    /// A filled `size`x`size` square object whose top-left pixel is at (`x`, `y`)
    /// in whole-image coordinates.
    fn square_object(x: u32, y: u32, size: u32, class: ObjectClass) -> Object {
        let mut object = Object::new(crate::object::ObjectInit {
            id: evanalyzer_cfg::core_types::ObjectId::next(),
            bbox: [x, y, x + size - 1, y + size - 1],
            mask_data: bitvec::prelude::BitVec::repeat(true, (size * size) as usize),
            area: (size * size) as usize,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    fn cache_with(objects: Vec<Object>) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache.class_colors = std::sync::Arc::new(HashMap::from([(RED_CLASS, 0xFF0000)]));
        cache
    }

    fn overlay_saver(display_classes: Vec<ObjectClass>, opacity: u32) -> SaveImage {
        SaveImage {
            name: "overlay".into(),
            display_classes,
            source: ImageSource::Image,
            opacity,
        }
    }

    #[test]
    fn draw_classes_fills_with_opacity_and_draws_an_opaque_contour() {
        let mut image = RgbImage::new(5, 5);
        let cache = cache_with(vec![square_object(1, 1, 3, RED_CLASS)]);

        overlay_saver(vec![RED_CLASS], 50).draw_classes(&mut image, Point2d { x: 0, y: 0 }, &cache);

        // Inside: black blended 50% with the class color (127.5 rounds up).
        assert_eq!(image.get_pixel(2, 2).0, [128, 0, 0]);
        // Every mask pixel touching the mask's border is contour, drawn opaque.
        for (x, y) in [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (3, 3)] {
            assert_eq!(
                image.get_pixel(x, y).0,
                [255, 0, 0],
                "contour at ({x}, {y})"
            );
        }
        // Outside the object nothing changes.
        assert_eq!(image.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(image.get_pixel(4, 4).0, [0, 0, 0]);
    }

    #[test]
    fn draw_classes_opacity_bounds_leave_or_replace_the_fill() {
        let cache = cache_with(vec![square_object(1, 1, 3, RED_CLASS)]);

        let mut image = RgbImage::from_pixel(5, 5, Rgb([10, 20, 30]));
        overlay_saver(vec![RED_CLASS], 0).draw_classes(&mut image, Point2d { x: 0, y: 0 }, &cache);
        assert_eq!(
            image.get_pixel(2, 2).0,
            [10, 20, 30],
            "opacity 0 keeps the image"
        );
        assert_eq!(
            image.get_pixel(1, 1).0,
            [255, 0, 0],
            "contour is drawn anyway"
        );

        let mut image = RgbImage::from_pixel(5, 5, Rgb([10, 20, 30]));
        overlay_saver(vec![RED_CLASS], 100).draw_classes(
            &mut image,
            Point2d { x: 0, y: 0 },
            &cache,
        );
        assert_eq!(image.get_pixel(2, 2).0, [255, 0, 0], "opacity 100 is solid");
    }

    #[test]
    fn draw_classes_only_draws_objects_of_the_selected_classes() {
        let mut image = RgbImage::new(5, 5);
        let cache = cache_with(vec![square_object(1, 1, 3, OTHER_CLASS)]);

        overlay_saver(vec![RED_CLASS], 50).draw_classes(&mut image, Point2d { x: 0, y: 0 }, &cache);

        assert!(image.pixels().all(|p| p.0 == [0, 0, 0]));
    }

    #[test]
    fn draw_classes_falls_back_to_a_generated_color_for_classes_without_one() {
        let mut image = RgbImage::new(5, 5);
        let cache = cache_with(vec![square_object(1, 1, 3, OTHER_CLASS)]);

        overlay_saver(vec![OTHER_CLASS], 50).draw_classes(
            &mut image,
            Point2d { x: 0, y: 0 },
            &cache,
        );

        assert_eq!(image.get_pixel(1, 1).0, get_color(2));
    }

    #[test]
    fn draw_classes_maps_whole_image_coordinates_into_the_tile() {
        // The object sits at (11, 11) of the whole image; the tile starts at (10, 10).
        let mut image = RgbImage::new(5, 5);
        let cache = cache_with(vec![square_object(11, 11, 3, RED_CLASS)]);

        overlay_saver(vec![RED_CLASS], 50).draw_classes(
            &mut image,
            Point2d { x: 10, y: 10 },
            &cache,
        );

        assert_eq!(image.get_pixel(1, 1).0, [255, 0, 0]);
        assert_eq!(image.get_pixel(2, 2).0, [128, 0, 0]);
        assert_eq!(image.get_pixel(0, 0).0, [0, 0, 0]);
    }

    #[test]
    fn draw_classes_contours_are_not_hidden_by_an_overlapping_fill() {
        let mut image = RgbImage::new(6, 6);
        // The second square's fill covers part of the first square's contour.
        let mut cache = cache_with(vec![
            square_object(0, 0, 3, RED_CLASS),
            square_object(1, 1, 4, OTHER_CLASS),
        ]);
        std::sync::Arc::make_mut(&mut cache.class_colors).insert(OTHER_CLASS, 0x0000FF);

        overlay_saver(vec![RED_CLASS, OTHER_CLASS], 100).draw_classes(
            &mut image,
            Point2d { x: 0, y: 0 },
            &cache,
        );

        // (2, 2) is a contour pixel of the red square and inside the blue one.
        assert_eq!(image.get_pixel(2, 2).0, [255, 0, 0]);
    }

    #[test]
    fn execute_with_display_classes_saves_an_rgb_overlay() {
        let input_img = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 5,
                height: 5,
            },
            0.0,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = PipelineContext::new_from_image_test(input_img).unwrap();
        ctx.output_path = Some(dir.path().to_path_buf());
        let mut cache = cache_with(vec![square_object(1, 1, 3, RED_CLASS)]);

        overlay_saver(vec![RED_CLASS], 50)
            .execute(&mut ctx, &mut cache)
            .unwrap();

        let saved = image::open(dir.path().join("images/overlay_x000000_y000000.png"))
            .unwrap()
            .into_rgb8();
        assert_eq!(saved.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(saved.get_pixel(1, 1).0, [255, 0, 0]);
        assert_eq!(saved.get_pixel(2, 2).0, [128, 0, 0]);
    }

    #[test]
    fn test_save_image_name() {
        let saver = SaveImage {
            name: "test".into(),
            display_classes: vec![],
            source: ImageSource::Image,
            opacity: 50,
        };
        assert_eq!(saver.name(), "Save Image");
        assert_eq!(saver.cite()[0].cite_key, "danmayr2026");
        assert!(matches!(saver.execution_scope(), ExecutionScope::Tile));
    }
}
