//! # drawn_images
//!
//! "Images with objects drawn" (export): each exported image as a PNG, its
//! channels in their colors (auto-scaled, 0.5 % / 99.5 %) and every object
//! of the analysis as a box in its class's color. Read from the pyramid
//! level that fits `DrawnImages::max_side` - a whole slide stays a
//! viewable picture. The results database keeps each object's box, not its
//! outline, so objects are boxes.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::ResultsGenerator;
use crate::api::{DrawnImages, ExportProgress};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors, ZProjection};
use evanalyzer_core::ImageChannel;
use evanalyzer_core::ImageContainer;
use evanalyzer_core::{ImageReader, ReadMode};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Writes `images/<name>.png` into `output_dir` for each of `images`
/// (relative paths below `settings.image_root`).
pub(super) fn export_drawn_images(
    database: &ResultsGenerator,
    settings: &DrawnImages,
    images: &[String],
    output_dir: &Path,
    (z_stack, t_stack): (u32, u32),
    cancel: &AtomicBool,
    on_progress: ExportProgress,
) -> Result<(), InternalErrors> {
    let dir = output_dir.join("images");
    std::fs::create_dir_all(&dir)
        .map_err(|e| InternalErrors::Io(format!("Could not create {}: {e}", dir.display())))?;
    let class_colors: HashMap<i32, u32> = database
        .get_object_classes()?
        .into_iter()
        .map(|class| (class.id as i32, class.color))
        .collect();
    for (i, rel_path) in images.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Internal("Export cancelled".into()));
        }
        on_progress("Drawing the objects on the images", i, images.len());
        let boxes = object_boxes(database, rel_path, z_stack, t_stack)?;
        let path = settings.image_root.join(rel_path);
        let picture = render(
            &path,
            settings,
            z_stack as i32,
            t_stack as i32,
            &boxes,
            &class_colors,
        )?;
        let name = Path::new(rel_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("image_{i}"));
        let out: PathBuf = dir.join(format!("{name}.png"));
        picture
            .save(&out)
            .map_err(|e| InternalErrors::Io(format!("Could not write {}: {e}", out.display())))?;
    }
    on_progress(
        "Drawing the objects on the images",
        images.len(),
        images.len(),
    );
    Ok(())
}

/// An object's box `[xmin, ymin, xmax, ymax]` (full-image pixels) and its
/// first class.
type ObjectBox = ([u32; 4], i32);

fn object_boxes(
    database: &ResultsGenerator,
    rel_path: &str,
    z_stack: u32,
    t_stack: u32,
) -> Result<Vec<ObjectBox>, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let mut stmt = database
        .connection()
        .prepare(
            "SELECT bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px, \
             coalesce(object_class_id[1], -1) FROM objects \
             WHERE image_rel_path = ? AND z_stack = ? AND t_stack = ?",
        )
        .map_err(err)?;
    stmt.query_map(duckdb::params![rel_path, z_stack, t_stack], |row| {
        Ok((
            [row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?],
            row.get(4)?,
        ))
    })
    .map_err(err)?
    .collect::<Result<Vec<_>, _>>()
    .map_err(err)
}

/// The image at the level that fits, its channels colored, the boxes on it.
fn render(
    path: &PathBuf,
    settings: &DrawnImages,
    z_stack: i32,
    t_stack: i32,
    boxes: &[ObjectBox],
    class_colors: &HashMap<i32, u32>,
) -> Result<image::RgbImage, InternalErrors> {
    let reader = ImageReader::new(path, ReadMode::Default)?;
    let meta = reader.get_image_meta();
    let series = meta
        .series
        .get(&settings.series)
        .or_else(|| meta.series.values().next())
        .ok_or_else(|| InternalErrors::ImageReadError(format!("{}: no series", path.display())))?;
    let full = series.resolutions.get(&0).ok_or_else(|| {
        InternalErrors::ImageReadError(format!("{}: no resolution", path.display()))
    })?;
    // The largest level whose longer side fits (else the smallest there is).
    let (level, info) = series
        .resolutions
        .iter()
        .find(|(_, r)| r.width.max(r.height) <= settings.max_side as u64)
        .or_else(|| series.resolutions.iter().last())
        .map(|(l, r)| (*l, r.clone()))
        .expect("a resolution");
    let (width, height) = (info.width as usize, info.height as usize);
    let channels = reader.read_image_tile_combined(
        settings.series,
        level,
        ZProjection::None,
        &Some(z_stack..=z_stack),
        t_stack,
        None,
        &ImageTile {
            offset_x: 0,
            offset_y: 0,
            width,
            height,
        },
    )?;

    let mut picture = colorize(&channels, &settings.colors, width, height);

    // The boxes, scaled to the level: 2 px lines in the class's color.
    let sx = width as f32 / full.width.max(1) as f32;
    let sy = height as f32 / full.height.max(1) as f32;
    for (bbox, class) in boxes {
        let c = class_colors.get(class).copied().unwrap_or(0xFFFFFF);
        let color = image::Rgb([(c >> 16) as u8, (c >> 8) as u8, c as u8]);
        let x0 = (bbox[0] as f32 * sx) as i64;
        let y0 = (bbox[1] as f32 * sy) as i64;
        let x1 = ((bbox[2] as f32 * sx) as i64).max(x0 + 1);
        let y1 = ((bbox[3] as f32 * sy) as i64).max(y0 + 1);
        draw_box(&mut picture, [x0, y0, x1, y1], color);
    }
    Ok(picture)
}

/// The channels of a tile as one picture: each in its color (white
/// without one), auto-scaled as the viewer does (see [`window`]). RGB
/// images are taken as they are. Also the split view's thumbnails.
pub fn colorize(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    width: usize,
    height: usize,
) -> image::RgbImage {
    let mut rgb = vec![0f32; width * height * 3];
    for channel in channels {
        let color = colors
            .get(&channel.c_stack)
            .copied()
            .unwrap_or([1.0, 1.0, 1.0]);
        match &*channel.image {
            ImageContainer::F32Gray(image) => {
                let data = image.data.as_slice();
                let (lo, hi) = window(data);
                let range = (hi - lo).max(1e-6);
                for (i, v) in data.iter().enumerate().take(width * height) {
                    let v = ((v - lo) / range).clamp(0.0, 1.0);
                    for k in 0..3 {
                        rgb[i * 3 + k] += v * color[k];
                    }
                }
            }
            ImageContainer::F32Rgb(image) => {
                for (out, v) in rgb.iter_mut().zip(image.data.as_slice()) {
                    *out += v;
                }
            }
            ImageContainer::U32(_) => {}
        }
    }
    image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let i = (y as usize * width + x as usize) * 3;
        image::Rgb([
            (rgb[i].min(1.0) * 255.0) as u8,
            (rgb[i + 1].min(1.0) * 255.0) as u8,
            (rgb[i + 2].min(1.0) * 255.0) as u8,
        ])
    })
}

/// [`colorize`]d, then scaled down to fill `(width, height)` (cropped to
/// its shape) - the split view's filmstrip.
pub fn colorize_thumbnail(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    (width, height): (usize, usize),
    (thumb_w, thumb_h): (u32, u32),
) -> image::RgbImage {
    image::DynamicImage::ImageRgb8(colorize(channels, colors, width, height))
        .resize_to_fill(thumb_w, thumb_h, image::imageops::FilterType::Triangle)
        .into_rgb8()
}

/// The display window of a channel: its 0.5 % and 99.5 % values (of every
/// 10th pixel), as the viewer's auto-scale.
fn window(data: &[f32]) -> (f32, f32) {
    let mut sample: Vec<f32> = data.iter().step_by(10).copied().collect();
    if sample.is_empty() {
        return (0.0, 1.0);
    }
    let len = sample.len();
    let low = (len as f32 * 0.005) as usize;
    let high = ((len as f32 * 0.995) as usize).min(len - 1);
    sample.select_nth_unstable_by(low, |a, b| a.total_cmp(b));
    let lo = sample[low];
    sample.select_nth_unstable_by(high, |a, b| a.total_cmp(b));
    (lo, sample[high])
}

/// A box outline, 2 px, clipped to the picture.
fn draw_box(picture: &mut image::RgbImage, [x0, y0, x1, y1]: [i64; 4], color: image::Rgb<u8>) {
    let (w, h) = (picture.width() as i64, picture.height() as i64);
    let mut put = |x: i64, y: i64| {
        if x >= 0 && y >= 0 && x < w && y < h {
            picture.put_pixel(x as u32, y as u32, color);
        }
    };
    for t in 0..2 {
        for x in x0..=x1 {
            put(x, y0 + t);
            put(x, y1 - t);
        }
        for y in y0..=y1 {
            put(x0 + t, y);
            put(x1 - t, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_is_drawn_as_an_outline_clipped_to_the_picture() {
        let mut picture = image::RgbImage::new(10, 10);
        let red = image::Rgb([255, 0, 0]);
        draw_box(&mut picture, [2, 2, 7, 7], red);
        assert_eq!(*picture.get_pixel(2, 5), red, "left edge");
        assert_eq!(*picture.get_pixel(7, 2), red, "corner");
        assert_eq!(
            *picture.get_pixel(5, 5),
            image::Rgb([0, 0, 0]),
            "inside stays"
        );
        // Past the edge: no panic.
        draw_box(&mut picture, [-5, -5, 20, 20], red);
    }

    #[test]
    fn the_window_skips_the_extremes() {
        // 1000 sampled values: one outlier each way is below 0.5 %.
        let mut data = vec![0.5f32; 10_000];
        data[0] = 0.0;
        data[10] = 1.0;
        assert_eq!(window(&data), (0.5, 0.5));
        assert_eq!(window(&[]), (0.0, 1.0));
    }
}
