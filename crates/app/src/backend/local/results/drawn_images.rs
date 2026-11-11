//! # drawn_images
//!
//! "Images with objects drawn" (export): a PNG per exported image and plane
//! (z, t), its channels in their colors as the viewer shows them and the
//! objects of the export's classes drawn on it - their outline, filled or
//! their box, in their class's color. Read from the pyramid level that
//! fits `DrawnImages::max_side` - a whole slide stays a viewable picture.
//! An object whose mask the results don't keep is drawn as its box.
//!
//! The objects are read twice at most (fills, then contours), one row at a
//! time - an image's masks are never all in RAM.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_generator::{ResultsGenerator, class_filter_sql};
use crate::api::{
    ChannelDisplay, DrawBrightness, DrawStyle, DrawnImages, ExportProgress, ImageDisplay,
};
use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors, ObjectClass, ZProjection};
use evanalyzer_core::ImageChannel;
use evanalyzer_core::ImageContainer;
use evanalyzer_core::{ImageReader, ReadMode};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::range::Range;
use std::sync::atomic::{AtomicBool, Ordering};

/// Opacity of a filled object ([`DrawStyle::Filled`]).
const FILL_ALPHA: f32 = 0.5;

/// Which planes of the results are exported (see `ResultExport`) and
/// whether the results have several T / Z - then the file names say which.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DrawnPlanes {
    pub z_stacks: Option<Range<u32>>,
    pub t_stacks: Option<Range<u32>>,
    pub by_z: bool,
    pub by_t: bool,
}

/// Writes `images/<relative path>[_z…][_t…].png` into `output_dir` for each
/// of `images` (relative paths below `settings.image_root`) and each of its
/// analysed planes within `planes`, with the objects of `classes` (`[]`:
/// all) drawn.
#[allow(clippy::too_many_arguments)]
pub(super) fn export_drawn_images(
    database: &ResultsGenerator,
    settings: &DrawnImages,
    classes: &[ObjectClass],
    images: &[String],
    output_dir: &Path,
    planes: DrawnPlanes,
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
    let wanted: Vec<u32> = classes
        .iter()
        .filter_map(|class| match class {
            ObjectClass::Global(id) => Some(*id),
            ObjectClass::Unset | ObjectClass::Local(_) => None,
        })
        .collect();
    // Classes chosen, but none of them stored: nothing to draw.
    let objects = ObjectQuery::new(database, (!classes.is_empty()).then_some(&wanted[..]))?;
    for (i, rel_path) in images.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Internal("Export cancelled".into()));
        }
        on_progress("Drawing the objects on the images", i, images.len());
        // From the results file - anyone's: it stays below the root (an
        // absolute path would replace it, `..` leave it).
        if !evanalyzer_cfg::core_types::is_contained_rel_path(rel_path) {
            return Err(InternalErrors::InvalidArgument(format!(
                "the image path {rel_path:?} in the results leaves the images folder - not drawn"
            )));
        }
        let display = settings
            .image_display
            .get(rel_path)
            .unwrap_or(&settings.display);
        let source = Source::open(&settings.image_root.join(rel_path), settings)?;
        for (z, t) in image_planes(database, rel_path, &planes)? {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Internal("Export cancelled".into()));
            }
            let channels = source.read(display.z_projection, z, t)?;
            let mut picture = colorize_with(
                &channels,
                &settings.colors,
                display,
                settings.brightness,
                source.width,
                source.height,
            );
            let scale = (
                source.width as f64 / source.full_width.max(1) as f64,
                source.height as f64 / source.full_height.max(1) as f64,
            );
            objects.draw(
                database,
                rel_path,
                (z, t),
                settings.style,
                &mut picture,
                scale,
                &class_colors,
            )?;
            let out = picture_path(&dir, rel_path, (z, t), &planes);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    InternalErrors::Io(format!("Could not create {}: {e}", parent.display()))
                })?;
            }
            picture.save(&out).map_err(|e| {
                InternalErrors::Io(format!("Could not write {}: {e}", out.display()))
            })?;
        }
    }
    on_progress(
        "Drawing the objects on the images",
        images.len(),
        images.len(),
    );
    Ok(())
}

/// `images/<relative path without extension>[_z…][_t…].png` - the image's
/// folders kept, so `A1/img.tif` and `B1/img.tif` don't overwrite each
/// other; the plane only when the results have several.
fn picture_path(dir: &Path, rel_path: &str, (z, t): (u32, u32), planes: &DrawnPlanes) -> PathBuf {
    let mut name = Path::new(rel_path)
        .file_name()
        .map(|n| {
            Path::new(n)
                .with_extension("")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| "image".to_string());
    if planes.by_z {
        name.push_str(&format!("_z{z:04}"));
    }
    if planes.by_t {
        name.push_str(&format!("_t{t:04}"));
    }
    name.push_str(".png");
    let folder = Path::new(rel_path).parent().unwrap_or(Path::new(""));
    dir.join(folder).join(name)
}

/// The planes `(z, t)` of image `rel_path` its analysis ran on, within
/// `planes`, by t then z. Results that don't record them: the planes its
/// objects are on. None: the first exported plane (the image without
/// objects).
fn image_planes(
    database: &ResultsGenerator,
    rel_path: &str,
    planes: &DrawnPlanes,
) -> Result<Vec<(u32, u32)>, InternalErrors> {
    let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
    let source = match database.analysed_planes_recorded() {
        true => {
            "SELECT t AS t_stack, UNNEST(z_stacks_analyzed) AS z_stack \
             FROM (SELECT UNNEST(t_stacks_analyzed) AS t, z_stacks_analyzed FROM images \
                   WHERE image_rel_path = ? AND t_stacks_analyzed IS NOT NULL)"
        }
        false => "SELECT t_stack, z_stack FROM objects WHERE image_rel_path = ?",
    };
    let mut stmt = database
        .connection()
        .prepare(&format!(
            "SELECT DISTINCT z_stack, t_stack FROM ({source}) \
             WHERE t_stack >= 0 AND z_stack >= 0 ORDER BY t_stack, z_stack"
        ))
        .map_err(err)?;
    let within = |value: u32, range: &Option<Range<u32>>| {
        range.is_none_or(|range| range.start <= value && value < range.end)
    };
    let mut found: Vec<(u32, u32)> = stmt
        .query_map([rel_path], |row| {
            Ok((row.get::<_, i64>(0)? as u32, row.get::<_, i64>(1)? as u32))
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    found.retain(|(z, t)| within(*z, &planes.z_stacks) && within(*t, &planes.t_stacks));
    if found.is_empty() {
        found.push((
            planes.z_stacks.map_or(0, |range| range.start),
            planes.t_stacks.map_or(0, |range| range.start),
        ));
    }
    Ok(found)
}

/// An image file, read at the level that fits `max_side`.
struct Source {
    reader: ImageReader,
    series: i32,
    level: i32,
    width: usize,
    height: usize,
    full_width: u64,
    full_height: u64,
}

impl Source {
    fn open(path: &Path, settings: &DrawnImages) -> Result<Self, InternalErrors> {
        let reader = ImageReader::new(&path.to_path_buf(), ReadMode::Default)?;
        let meta = reader.get_image_meta();
        let (series, info) = meta
            .series
            .get_key_value(&settings.series)
            .or_else(|| meta.series.iter().next())
            .ok_or_else(|| {
                InternalErrors::ImageReadError(format!("{}: no series", path.display()))
            })?;
        let full = info.resolutions.get(&0).ok_or_else(|| {
            InternalErrors::ImageReadError(format!("{}: no resolution", path.display()))
        })?;
        // The largest level whose longer side fits (else the smallest there is).
        let (level, size) = info
            .resolutions
            .iter()
            .find(|(_, r)| r.width.max(r.height) <= settings.max_side as u64)
            .or_else(|| info.resolutions.iter().last())
            .map(|(l, r)| (*l, (r.width, r.height)))
            .expect("a resolution");
        let (series, full_width, full_height) = (*series, full.width, full.height);
        Ok(Self {
            reader,
            series,
            level,
            width: size.0 as usize,
            height: size.1 as usize,
            full_width,
            full_height,
        })
    }

    /// Plane `(z, t)` - all planes projected with `projection` (its
    /// objects are on plane 0).
    fn read(
        &self,
        projection: ZProjection,
        z: u32,
        t: u32,
    ) -> Result<Vec<ImageChannel>, InternalErrors> {
        let z_range = (projection == ZProjection::None).then_some(z as i32..=z as i32);
        self.reader.read_image_tile_combined(
            self.series,
            self.level,
            projection,
            &z_range,
            t as i32,
            None,
            &ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: self.width,
                height: self.height,
            },
        )
    }
}

/// Which pass over the objects: the fills go under every contour, so an
/// overlapping neighbour's fill never hides a contour.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Fill,
    Contour,
}

/// The SQL reading one plane's objects: box, classes, mask (if kept) and,
/// for a volume's object, which of its planes this is.
struct ObjectQuery {
    sql: String,
    /// The export's classes, in their order - `None`: all.
    wanted: Option<Vec<u32>>,
}

impl ObjectQuery {
    fn new(database: &ResultsGenerator, wanted: Option<&[u32]>) -> Result<Self, InternalErrors> {
        let mask_expr = match database.has_masks()? {
            true => "mask",
            false => "NULL::BLOB",
        };
        let volume_expr = match database.has_volume_columns()? {
            true => {
                "CASE WHEN ndim = 3 THEN bbox_zmin_px END, \
                     CASE WHEN ndim = 3 THEN bbox_zmax_px - bbox_zmin_px + 1 END"
            }
            false => "NULL::UINTEGER, NULL::UINTEGER",
        };
        let class_filter = match wanted {
            Some([]) => " AND false".to_string(),
            Some(ids) => format!(" AND {}", class_filter_sql("object_class_id", ids)),
            None => String::new(),
        };
        Ok(Self {
            // `{on_plane}` is filled per plane (see `draw`).
            sql: format!(
                "SELECT bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px, \
                 CAST(object_class_id AS INTEGER[]), {mask_expr}, {volume_expr} \
                 FROM objects WHERE image_rel_path = ? AND {{on_plane}} AND t_stack = ?{class_filter}"
            ),
            wanted: wanted.map(<[u32]>::to_vec),
        })
    }

    /// The class an object with `classes` is drawn in: the first of the
    /// export's classes it has (all: its first).
    fn class_of(&self, classes: &[i32]) -> Option<i32> {
        match &self.wanted {
            None => classes.first().copied(),
            Some(wanted) => wanted
                .iter()
                .map(|id| *id as i32)
                .find(|id| classes.contains(id)),
        }
    }

    /// Draws the objects of `rel_path` on plane `(z, t)` onto `picture`
    /// (`scale`: its pixels per image pixel, x and y).
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        database: &ResultsGenerator,
        rel_path: &str,
        (z, t): (u32, u32),
        style: DrawStyle,
        picture: &mut image::RgbImage,
        scale: (f64, f64),
        class_colors: &HashMap<i32, u32>,
    ) -> Result<(), InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let sql = self
            .sql
            .replace("{on_plane}", &database.object_on_plane_sql("", z)?);
        let passes: &[Pass] = match style {
            DrawStyle::Filled => &[Pass::Fill, Pass::Contour],
            DrawStyle::Outline | DrawStyle::Box => &[Pass::Contour],
        };
        let mut stmt = database.connection().prepare(&sql).map_err(err)?;
        for &pass in passes {
            let mut rows = stmt
                .query(duckdb::params![rel_path, t as i64])
                .map_err(err)?;
            while let Some(row) = rows.next().map_err(err)? {
                let bbox: [u32; 4] = [
                    row.get(0).map_err(err)?,
                    row.get(1).map_err(err)?,
                    row.get(2).map_err(err)?,
                    row.get(3).map_err(err)?,
                ];
                let classes = int_list(row.get(4).map_err(err)?);
                let Some(class) = self.class_of(&classes) else {
                    continue;
                };
                let c = class_colors.get(&class).copied().unwrap_or(0xFFFFFF);
                let color = image::Rgb([(c >> 16) as u8, (c >> 8) as u8, c as u8]);
                let mask = match style {
                    DrawStyle::Box => None,
                    DrawStyle::Outline | DrawStyle::Filled => {
                        let stored: Option<Vec<u8>> = row.get(5).map_err(err)?;
                        let volume: (Option<u32>, Option<u32>) =
                            (row.get(6).map_err(err)?, row.get(7).map_err(err)?);
                        stored.and_then(|bytes| plane_mask(&bytes, bbox, volume, z))
                    }
                };
                match (mask, pass) {
                    (Some(mask), pass) => draw_mask(picture, bbox, &mask, scale, color, pass),
                    // No mask: its box, once.
                    (None, Pass::Contour) => draw_box(picture, scaled_box(bbox, scale), color),
                    (None, Pass::Fill) => {}
                }
            }
        }
        Ok(())
    }
}

/// A DuckDB `INTEGER[]` as a list - empty for NULL.
fn int_list(value: duckdb::types::Value) -> Vec<i32> {
    match value {
        duckdb::types::Value::List(values) => values
            .into_iter()
            .filter_map(|v| match v {
                duckdb::types::Value::Int(i) => Some(i),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The mask of an object on plane `z`: its stored `bytes` (see
/// `mask_codec`) - for a volume's object (`(zmin, depth)`) the part on `z`.
/// `None` when nothing is stored or it doesn't fit the box.
fn plane_mask(
    bytes: &[u8],
    [x0, y0, x1, y1]: [u32; 4],
    volume: (Option<u32>, Option<u32>),
    z: u32,
) -> Option<BitVec<u64, Lsb0>> {
    if bytes.is_empty() {
        return None;
    }
    let len = (x1.checked_sub(x0)? as usize + 1) * (y1.checked_sub(y0)? as usize + 1);
    match volume {
        (Some(zmin), Some(depth)) => {
            let all = evanalyzer_core::decode_mask(bytes, len * depth as usize)?;
            let start = z.checked_sub(zmin)? as usize * len;
            all.get(start..start + len).map(BitVec::from_bitslice)
        }
        _ => evanalyzer_core::decode_mask(bytes, len),
    }
}

/// `bbox` (image pixels, inclusive) in picture pixels.
fn scaled_box([x0, y0, x1, y1]: [u32; 4], (sx, sy): (f64, f64)) -> [i64; 4] {
    let px0 = (x0 as f64 * sx) as i64;
    let py0 = (y0 as f64 * sy) as i64;
    [
        px0,
        py0,
        ((x1 as f64 * sx) as i64).max(px0 + 1),
        ((y1 as f64 * sy) as i64).max(py0 + 1),
    ]
}

/// An object's mask (a bit per pixel of `bbox`, row by row) on the picture:
/// [`Pass::Fill`] blends its area with `color` ([`FILL_ALPHA`]),
/// [`Pass::Contour`] paints its contour opaque - picture pixels inside it
/// with a 4-neighbour outside. Each picture pixel samples the image pixel
/// at its centre. Smaller than a picture pixel: one pixel at its centre.
fn draw_mask(
    picture: &mut image::RgbImage,
    bbox: [u32; 4],
    mask: &BitVec<u64, Lsb0>,
    (sx, sy): (f64, f64),
    color: image::Rgb<u8>,
    pass: Pass,
) {
    let [x0, y0, x1, y1] = bbox.map(i64::from);
    let bbox_w = x1 - x0 + 1;
    let inside = |px: i64, py: i64| {
        let x = ((px as f64 + 0.5) / sx).floor() as i64;
        let y = ((py as f64 + 0.5) / sy).floor() as i64;
        x >= x0
            && x <= x1
            && y >= y0
            && y <= y1
            && mask
                .get(((y - y0) * bbox_w + (x - x0)) as usize)
                .is_some_and(|bit| *bit)
    };
    let (w, h) = (picture.width() as i64, picture.height() as i64);
    let [px0, py0, px1, py1] = [
        (x0 as f64 * sx).floor() as i64,
        (y0 as f64 * sy).floor() as i64,
        (x1 as f64 * sx).floor() as i64,
        (y1 as f64 * sy).floor() as i64,
    ];
    let mut any = false;
    for py in py0.max(0)..=py1.min(h - 1) {
        for px in px0.max(0)..=px1.min(w - 1) {
            if !inside(px, py) {
                continue;
            }
            any = true;
            let pixel = picture.get_pixel_mut(px as u32, py as u32);
            match pass {
                Pass::Fill => {
                    for (c, fill) in pixel.0.iter_mut().zip(color.0) {
                        *c = (*c as f32 * (1.0 - FILL_ALPHA) + fill as f32 * FILL_ALPHA).round()
                            as u8;
                    }
                }
                Pass::Contour => {
                    if !inside(px - 1, py)
                        || !inside(px + 1, py)
                        || !inside(px, py - 1)
                        || !inside(px, py + 1)
                    {
                        *pixel = color;
                    }
                }
            }
        }
    }
    if !any && pass == Pass::Contour {
        let cx = ((x0 + x1) as f64 / 2.0 * sx) as i64;
        let cy = ((y0 + y1) as f64 / 2.0 * sy) as i64;
        if cx >= 0 && cy >= 0 && cx < w && cy < h {
            picture.put_pixel(cx as u32, cy as u32, color);
        }
    }
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
    colorize_with(
        channels,
        colors,
        &ImageDisplay::default(),
        DrawBrightness::Auto,
        width,
        height,
    )
}

/// [`colorize`] as `display` shows the image: its hidden channels left
/// out, and with [`DrawBrightness::Viewer`] each channel through its
/// display range (auto-scaled without one).
fn colorize_with(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    display: &ImageDisplay,
    brightness: DrawBrightness,
    width: usize,
    height: usize,
) -> image::RgbImage {
    let mut rgb = vec![0f32; width * height * 3];
    for channel in channels {
        let shown = display
            .channels
            .get(&channel.c_stack)
            .copied()
            .unwrap_or_default();
        if !shown.visible {
            continue;
        }
        let color = colors
            .get(&channel.c_stack)
            .copied()
            .unwrap_or([1.0, 1.0, 1.0]);
        match &*channel.image {
            ImageContainer::F32Gray(image) => {
                let data = image.data.as_slice();
                let (lo, hi) = match (brightness, shown) {
                    (
                        DrawBrightness::Viewer,
                        ChannelDisplay {
                            range: Some(range), ..
                        },
                    ) => range,
                    _ => window(data),
                };
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

    const RED: image::Rgb<u8> = image::Rgb([255, 0, 0]);
    const BLACK: image::Rgb<u8> = image::Rgb([0, 0, 0]);

    /// A `w` x `h` mask, set where `f` says.
    fn mask(w: usize, h: usize, f: impl Fn(usize, usize) -> bool) -> BitVec<u64, Lsb0> {
        (0..w * h).map(|i| f(i % w, i / w)).collect()
    }

    #[test]
    fn a_box_is_drawn_as_an_outline_clipped_to_the_picture() {
        let mut picture = image::RgbImage::new(10, 10);
        draw_box(&mut picture, [2, 2, 7, 7], RED);
        assert_eq!(*picture.get_pixel(2, 5), RED, "left edge");
        assert_eq!(*picture.get_pixel(7, 2), RED, "corner");
        assert_eq!(*picture.get_pixel(5, 5), BLACK, "inside stays");
        // Past the edge: no panic.
        draw_box(&mut picture, [-5, -5, 20, 20], RED);
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

    /// A 5 x 5 square at (2, 2): its border is the contour, its centre not.
    #[test]
    fn the_contour_is_the_mask_pixels_next_to_the_outside() {
        let mut picture = image::RgbImage::new(10, 10);
        let square = mask(5, 5, |_, _| true);
        draw_mask(
            &mut picture,
            [2, 2, 6, 6],
            &square,
            (1.0, 1.0),
            RED,
            Pass::Contour,
        );
        assert_eq!(*picture.get_pixel(2, 4), RED, "left edge");
        assert_eq!(*picture.get_pixel(6, 6), RED, "corner");
        assert_eq!(*picture.get_pixel(4, 4), BLACK, "inside");
        assert_eq!(*picture.get_pixel(1, 4), BLACK, "outside");
    }

    /// Not a box: a hole in the mask has its own contour, the box's corner
    /// outside the mask stays.
    #[test]
    fn the_outline_follows_the_mask_not_the_box() {
        let mut picture = image::RgbImage::new(10, 10);
        // A plus sign in a 3 x 3 box.
        let plus = mask(3, 3, |x, y| x == 1 || y == 1);
        draw_mask(
            &mut picture,
            [0, 0, 2, 2],
            &plus,
            (1.0, 1.0),
            RED,
            Pass::Contour,
        );
        assert_eq!(*picture.get_pixel(0, 0), BLACK, "the box's corner");
        assert_eq!(*picture.get_pixel(1, 0), RED);
    }

    #[test]
    fn a_fill_blends_half_and_leaves_the_outside() {
        let mut picture = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 200]));
        let square = mask(2, 2, |_, _| true);
        draw_mask(
            &mut picture,
            [1, 1, 2, 2],
            &square,
            (1.0, 1.0),
            RED,
            Pass::Fill,
        );
        assert_eq!(*picture.get_pixel(1, 1), image::Rgb([128, 0, 100]));
        assert_eq!(*picture.get_pixel(0, 0), image::Rgb([0, 0, 200]));
    }

    /// A picture at a quarter of the image (a pyramid level): the object
    /// is drawn where it is, at that size.
    #[test]
    fn a_mask_is_drawn_at_the_scale_of_the_picture() {
        let mut picture = image::RgbImage::new(10, 10);
        // 20 x 20 image pixels at (8, 8): 5 x 5 picture pixels at (2, 2).
        let square = mask(20, 20, |_, _| true);
        draw_mask(
            &mut picture,
            [8, 8, 27, 27],
            &square,
            (0.25, 0.25),
            RED,
            Pass::Contour,
        );
        assert_eq!(*picture.get_pixel(2, 4), RED);
        assert_eq!(*picture.get_pixel(6, 6), RED);
        assert_eq!(*picture.get_pixel(4, 4), BLACK);
        assert_eq!(*picture.get_pixel(7, 4), BLACK);
    }

    /// Smaller than a picture pixel: still a dot, it doesn't vanish.
    #[test]
    fn a_tiny_object_stays_a_dot() {
        let mut picture = image::RgbImage::new(10, 10);
        let dot = mask(2, 2, |x, y| x == 1 && y == 1);
        draw_mask(
            &mut picture,
            [40, 40, 41, 41],
            &dot,
            (0.1, 0.1),
            RED,
            Pass::Contour,
        );
        assert_eq!(*picture.get_pixel(4, 4), RED);
    }

    #[test]
    fn a_volume_object_is_cut_to_its_plane() {
        // 2 x 1 box over planes 3 and 4: plane 3 left pixel, plane 4 right.
        let all: BitVec<u64, Lsb0> = [true, false, false, true].into_iter().collect();
        let bytes = evanalyzer_core::encode_mask(&all);
        let on = |z| plane_mask(&bytes, [0, 0, 1, 0], (Some(3), Some(2)), z);
        assert_eq!(
            on(3).unwrap().iter().by_vals().collect::<Vec<_>>(),
            [true, false]
        );
        assert_eq!(
            on(4).unwrap().iter().by_vals().collect::<Vec<_>>(),
            [false, true]
        );
        assert!(on(2).is_none(), "not on that plane");
        assert!(plane_mask(&[], [0, 0, 1, 0], (None, None), 0).is_none());
    }

    #[test]
    fn the_file_keeps_the_folders_and_names_the_plane_when_there_are_several() {
        let dir = Path::new("out/images");
        let one = DrawnPlanes::default();
        assert_eq!(
            picture_path(dir, "A1/img.ome.tif", (0, 0), &one),
            Path::new("out/images/A1/img.ome.png")
        );
        let several = DrawnPlanes {
            by_z: true,
            by_t: true,
            ..Default::default()
        };
        assert_eq!(
            picture_path(dir, "B1/img.tif", (2, 13), &several),
            Path::new("out/images/B1/img_z0002_t0013.png")
        );
    }

    #[test]
    fn a_hidden_channel_is_left_out_and_the_viewer_range_used() {
        let channel = |c, v: f32| ImageChannel {
            image: std::sync::Arc::new(ImageContainer::F32Gray(crate::api::ManagedImage {
                data: kornia_image::Image::new(
                    kornia_image::ImageSize {
                        width: 1,
                        height: 1,
                    },
                    vec![v],
                )
                .unwrap(),
                tile_offset: crate::api::Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: evanalyzer_cfg::core_types::ImageDimensionality::Planar,
            })),
            color: [1.0, 1.0, 1.0],
            c_stack: c,
            name: String::new(),
            is_rgb: false,
            is_visible: true,
        };
        let channels = [channel(0, 0.25), channel(1, 1.0)];
        let colors = [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])].into();
        let display = ImageDisplay {
            z_projection: ZProjection::None,
            channels: [
                (
                    0,
                    ChannelDisplay {
                        visible: true,
                        range: Some((0.0, 0.5)),
                    },
                ),
                (
                    1,
                    ChannelDisplay {
                        visible: false,
                        range: None,
                    },
                ),
            ]
            .into(),
        };
        let picture = colorize_with(&channels, &colors, &display, DrawBrightness::Viewer, 1, 1);
        assert_eq!(*picture.get_pixel(0, 0), image::Rgb([127, 0, 0]));
    }
}
