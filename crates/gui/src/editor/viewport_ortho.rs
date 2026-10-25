//! # viewport_ortho
//!
//! Orthogonal views of a z-stack through the crosshair: the XZ view (the
//! image's row at the crosshair across every plane, below the image) and the
//! YZ view (its column across every plane, beside it).
//!
//! Only what is shown is read: the row / column of the visible part of the
//! image, at the pyramid level the image is shown at - every plane in
//! parallel, one pixel thick, through the same [`ImageSource`] as the tiles
//! (also on a server). The last strip of each axis is kept, compact (see
//! [`ViewChannel`]) - a few hundred KB - so a new display range or colour
//! only renders it again. While the views are off nothing runs or is kept.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::view_channel::ViewChannel;
use crate::editor::viewport_controller::ViewportState;
use crate::editor::viewport_worker::{
    ViewportWorker, channel_ctx, channel_display_color, prepare_image_channels_for_slint,
};
use evanalyzer_app::images::{
    ImageChannel, ImageContainer, ImageSource, ManagedImage, TileRequest,
};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::{ImageDimensionality, ImageTile, InternalErrors, ZProjection};
use kornia_image::{Image, ImageSize};
use log::{debug, warn};
use rayon::prelude::*;
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::path::PathBuf;

/// Which orthogonal view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    /// x across, z down: the row `line`.
    Xz,
    /// z across, y down: the column `line`.
    Yz,
}

/// Everything that decides a strip's samples. Pixels of pyramid level
/// `level`: `line` the row (XZ) or column (YZ), `from..to` the visible part
/// along it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StripKey {
    pub path: PathBuf,
    pub series: i32,
    pub level: i32,
    pub t: i32,
    pub axis: Axis,
    pub line: usize,
    pub from: usize,
    pub to: usize,
    pub depth: i32,
}

/// One orthogonal view's samples: per channel `width` × `height` - XZ: the
/// line's pixels across, a row per plane; YZ: a column per plane, the line's
/// pixels down.
pub(crate) struct Strip {
    pub key: StripKey,
    pub channels: Vec<ViewChannel>,
    pub width: usize,
    pub height: usize,
}

/// The last strip of each view - see the module docs.
#[derive(Default)]
pub(crate) struct OrthoStrips {
    xz: Option<Strip>,
    yz: Option<Strip>,
}

/// Where the image is shown and what of it: the view and the image's size.
pub(crate) struct ShownImage<'a> {
    pub path: &'a PathBuf,
    pub series: i32,
    pub t: i32,
    pub depth: i32,
    /// Full-resolution size.
    pub full: (usize, usize),
    /// The pyramid level shown and its size.
    pub level: i32,
    pub level_size: (usize, usize),
}

/// The strips the views need for `view` and the crosshair at (`x`, `y`)
/// (full-resolution pixels): (XZ, YZ) - `None` for a view whose line is
/// outside the image or whose visible part is empty.
pub(crate) fn strip_keys(
    image: &ShownImage,
    view: &ViewportState,
    x: f32,
    y: f32,
) -> (Option<StripKey>, Option<StripKey>) {
    let (full_w, full_h) = (image.full.0 as f32, image.full.1 as f32);
    let (level_w, level_h) = image.level_size;
    if view.zoom <= 0.0 || full_w <= 0.0 || full_h <= 0.0 {
        return (None, None);
    }
    let scale = level_w as f32 / full_w;
    // The visible part along one axis, in level pixels.
    let visible = |offset: f32, extent: f32, full: f32, level: usize| {
        let from = (-offset / view.zoom).clamp(0.0, full);
        let to = ((extent - offset) / view.zoom).clamp(0.0, full);
        let from = (from * scale).floor() as usize;
        let to = ((to * scale).ceil() as usize).min(level);
        (to > from).then_some((from, to))
    };
    let line = |at: f32, full: f32, level: usize| {
        (at >= 0.0 && at < full).then(|| ((at * scale) as usize).min(level.saturating_sub(1)))
    };
    let key = |axis, line, (from, to)| StripKey {
        path: image.path.clone(),
        series: image.series,
        level: image.level,
        t: image.t,
        axis,
        line,
        from,
        to,
        depth: image.depth,
    };
    let xz = line(y, full_h, level_h)
        .zip(visible(view.offset_x, view.viewport_width, full_w, level_w))
        .map(|(row, range)| key(Axis::Xz, row, range));
    let yz = line(x, full_w, level_w)
        .zip(visible(
            view.offset_y,
            view.viewport_height,
            full_h,
            level_h,
        ))
        .map(|(column, range)| key(Axis::Yz, column, range));
    (xz, yz)
}

/// Reads `key`'s strip from `source`: its line on every plane, in
/// parallel, then laid out as the view shows it. Channels that aren't one
/// value per pixel (interleaved RGB, labels) are left out, as the image
/// view leaves them out. `bits`: the image's bit depth (compact samples).
pub(crate) fn read_strip(
    source: &dyn ImageSource,
    key: &StripKey,
    bits: u16,
) -> Result<Strip, InternalErrors> {
    let len = key.to - key.from;
    let depth = key.depth.max(0) as usize;
    let tile = match key.axis {
        Axis::Xz => ImageTile {
            offset_x: key.from,
            offset_y: key.line,
            width: len,
            height: 1,
        },
        Axis::Yz => ImageTile {
            offset_x: key.line,
            offset_y: key.from,
            width: 1,
            height: len,
        },
    };
    let planes: Vec<Vec<ImageChannel>> = (0..key.depth)
        .into_par_iter()
        .map(|z| {
            source.read_tile(&TileRequest {
                series: key.series,
                resolution_idx: key.level,
                z_projection: ZProjection::None,
                z_range: Some(z..=z),
                t_stack: key.t,
                tile,
            })
        })
        .collect::<Result<_, _>>()?;
    let (width, height) = match key.axis {
        Axis::Xz => (len, depth),
        Axis::Yz => (depth, len),
    };
    let Some(first) = planes.first() else {
        return Ok(Strip {
            key: key.clone(),
            channels: Vec::new(),
            width,
            height,
        });
    };
    let mut channels = Vec::with_capacity(first.len());
    for (c, template) in first.iter().enumerate() {
        let mut samples = vec![0f32; width * height];
        let mut gray = true;
        for (z, plane) in planes.iter().enumerate() {
            let Some(ImageContainer::F32Gray(line)) = plane.get(c).map(|ch| &*ch.image) else {
                gray = false;
                break;
            };
            for (i, &value) in line.as_slice().iter().take(len).enumerate() {
                let at = match key.axis {
                    Axis::Xz => z * len + i,
                    Axis::Yz => i * depth + z,
                };
                samples[at] = value;
            }
        }
        if !gray {
            continue;
        }
        let image = Image::<f32, 1>::new(ImageSize { width, height }, samples)
            .map_err(InternalErrors::from_kornia)?;
        let channel = ImageChannel {
            image: std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: image,
                tile_offset: Default::default(),
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            ..template.clone()
        };
        channels.push(ViewChannel::from_image_channel(&channel, bits));
    }
    Ok(Strip {
        key: key.clone(),
        channels,
        width,
        height,
    })
}

/// How the channels are shown: as in the image view.
pub(crate) struct ChannelLook {
    pub visible: Vec<i32>,
    pub histograms: std::collections::BTreeMap<
        i32,
        Option<evanalyzer_cfg::settings::images_settings::HistogramSettings>,
    >,
    pub colors: std::collections::BTreeMap<i32, [f32; 3]>,
    pub grayscale: bool,
}

/// `strip` rendered like the image: its visible channels with their
/// display range and colour.
pub(crate) fn render_strip(strip: &Strip, look: &ChannelLook) -> SharedPixelBuffer<Rgb8Pixel> {
    let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(strip.width as u32, strip.height as u32);
    let contexts: Vec<_> = strip
        .channels
        .iter()
        .filter_map(|channel| {
            let histogram = look.histograms.get(&channel.c_stack)?.clone()?;
            let color = channel_display_color(
                channel,
                look.colors.get(&channel.c_stack).copied(),
                look.grayscale,
            );
            Some(channel_ctx(
                channel.slice()?,
                histogram,
                color,
                channel.c_stack,
            ))
        })
        .collect();
    prepare_image_channels_for_slint(
        &contexts,
        buffer.make_mut_slice(),
        0,
        false,
        &look.visible,
        false,
    );
    buffer
}

/// One render of the views: each a buffer and where it lies along the
/// image (full-resolution pixels: XZ from x, YZ from y, that long).
pub(crate) struct OrthoFrame {
    pub xz: Option<(SharedPixelBuffer<Rgb8Pixel>, f32, f32)>,
    pub yz: Option<(SharedPixelBuffer<Rgb8Pixel>, f32, f32)>,
    pub depth: i32,
    /// z spacing over the xy pixel size: how much taller a plane is drawn
    /// than a pixel is wide. 1 if the z spacing is unknown.
    pub aniso: f32,
    pub z_known: bool,
}

impl ViewportWorker {
    /// Renders the orthogonal views for the current view and crosshair -
    /// reading a strip only where it changed. Off: frees the strips.
    pub(crate) fn render_ortho(&self, strips: &mut OrthoStrips) {
        let (on, x, y) = self.viewport_controller.ortho();
        if !on {
            *strips = OrthoStrips::default();
            return;
        }
        let (path, series, t, look, pixel_sizes) = {
            let project = self.app_state.get_project();
            let Some(path) = project.get_current_image_path_cloned() else {
                return;
            };
            (
                path,
                project.get_selected_series_idx(),
                project.get_t_stack().map_or(0, |t| t.t_stack),
                ChannelLook {
                    visible: project.get_image_channel_visibilities_vec(),
                    histograms: project.get_image_channel_histograms(),
                    colors: project.get_channel_display_colors(),
                    grayscale: project.images.settings.grayscale,
                },
                project.get_pixel_sizes(),
            )
        };
        // The level the image is shown at - as its sharp render read it.
        let Some(level) = self
            .viewport_cache
            .active_high_res_data
            .read()
            .ok()
            .and_then(|active| active.as_ref().map(|(_, ctx)| ctx.res_idx))
        else {
            return;
        };
        let view = self
            .viewport_controller
            .viewport_state
            .read()
            .unwrap()
            .clone();
        let source = match self.app_state.get_image_source(&path) {
            Ok(source) => source,
            Err(e) => {
                warn!("Orthogonal views: {e:?}");
                return;
            }
        };
        let meta = source.meta();
        // One plane: no views (the viewport shows none) - nothing kept.
        let Some(info) = meta.series.get(&series).filter(|info| info.nr_z_stacks > 1) else {
            *strips = OrthoStrips::default();
            return;
        };
        let (Some(full), Some(shown)) = (info.resolutions.get(&0), info.resolutions.get(&level))
        else {
            return;
        };
        let image = ShownImage {
            path: &path,
            series,
            t,
            depth: info.nr_z_stacks,
            full: (full.width as usize, full.height as usize),
            level,
            level_size: (shown.width as usize, shown.height as usize),
        };
        let (xz_key, yz_key) = strip_keys(&image, &view, x, y);
        let bits = full.nr_bits;
        // Read again only what changed - the two views at once.
        let refresh = |kept: &mut Option<Strip>, key: Option<StripKey>| {
            match key {
                Some(key) if kept.as_ref().is_some_and(|s| s.key == key) => {}
                Some(key) => match read_strip(&*source, &key, bits) {
                    Ok(strip) => *kept = Some(strip),
                    Err(e) => {
                        warn!("Orthogonal view {:?}: {e:?}", key.axis);
                        *kept = None;
                    }
                },
                None => *kept = None,
            };
        };
        let started = std::time::Instant::now();
        let OrthoStrips { xz, yz } = strips;
        rayon::join(|| refresh(xz, xz_key), || refresh(yz, yz_key));
        let read = started.elapsed();

        let level_scale = full.width as f32 / shown.width.max(1) as f32;
        let placed = |strip: &Strip| {
            (
                render_strip(strip, &look),
                strip.key.from as f32 * level_scale,
                (strip.key.to - strip.key.from) as f32 * level_scale,
            )
        };
        let (aniso, z_known) = match pixel_sizes.z {
            Some(z) if z > 0.0 && pixel_sizes.x > 0.0 => (z / pixel_sizes.x, true),
            _ => (1.0, false),
        };
        let frame = OrthoFrame {
            xz: strips.xz.as_ref().map(placed),
            yz: strips.yz.as_ref().map(placed),
            depth: info.nr_z_stacks,
            aniso,
            z_known,
        };
        let kept: usize = [&strips.xz, &strips.yz]
            .into_iter()
            .flatten()
            .flat_map(|s| s.channels.iter().map(ViewChannel::size_in_bytes))
            .sum();
        debug!(
            "Orthogonal views: read {read:?}, total {:?}, {} planes, {kept} bytes kept",
            started.elapsed(),
            info.nr_z_stacks
        );
        self.viewport_controller.post_ortho(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::images::{ImageInfo, ImageMeta, PyramidInfo};
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// A 2-channel 8 × 6 × 4 stack of 16-bit samples: channel c's at
    /// (x, y, z) is `c + 10 x + 100 y + 1000 z`; every request is counted.
    struct Stack {
        meta: ImageMeta,
        requests: Mutex<usize>,
    }

    impl Stack {
        fn new() -> Self {
            let mut series = BTreeMap::new();
            let mut info = ImageInfo {
                nr_c_stacks: 2,
                nr_z_stacks: 4,
                nr_t_stacks: 1,
                ..Default::default()
            };
            info.resolutions.insert(
                0,
                PyramidInfo {
                    nr_bits: 16,
                    color_channels: 1,
                    width: 8,
                    height: 6,
                    ..Default::default()
                },
            );
            series.insert(0, info);
            Self {
                meta: ImageMeta {
                    series,
                    ..Default::default()
                },
                requests: Mutex::new(0),
            }
        }

        /// As the reader normalizes 16-bit samples: `q × (1 / 65535)`.
        fn value(c: usize, x: usize, y: usize, z: usize) -> f32 {
            (c + 10 * x + 100 * y + 1000 * z) as f32 * (1.0 / 65535.0)
        }
    }

    impl ImageSource for Stack {
        fn meta(&self) -> &ImageMeta {
            &self.meta
        }

        fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            *self.requests.lock().unwrap() += 1;
            let z = *req.z_range.as_ref().unwrap().start() as usize;
            let t = req.tile;
            Ok((0..2)
                .map(|c| {
                    let values = (0..t.height)
                        .flat_map(|dy| {
                            (0..t.width)
                                .map(move |dx| Self::value(c, t.offset_x + dx, t.offset_y + dy, z))
                        })
                        .collect();
                    ImageChannel {
                        image: std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                            data: Image::new(
                                ImageSize {
                                    width: t.width,
                                    height: t.height,
                                },
                                values,
                            )
                            .unwrap(),
                            tile_offset: Default::default(),
                            plane: None,
                            dimensionality: ImageDimensionality::Planar,
                        })),
                        color: [1.0, 1.0, 1.0],
                        is_visible: true,
                        c_stack: c as i32,
                        name: String::new(),
                        is_rgb: false,
                    }
                })
                .collect())
        }
    }

    fn key(axis: Axis, line: usize, from: usize, to: usize) -> StripKey {
        StripKey {
            path: PathBuf::from("stack.tif"),
            series: 0,
            level: 0,
            t: 0,
            axis,
            line,
            from,
            to,
            depth: 4,
        }
    }

    fn value_at(channel: &ViewChannel, i: usize) -> f32 {
        channel.value_at(i).unwrap()
    }

    /// XZ: the row across, a row per plane; YZ: a column per plane, the
    /// column down - every value from its plane and channel.
    #[test]
    fn a_strip_lays_the_line_out_across_the_planes() {
        let stack = Stack::new();
        let xz = read_strip(&stack, &key(Axis::Xz, 2, 1, 6), 16).unwrap();
        assert_eq!((xz.width, xz.height, xz.channels.len()), (5, 4, 2));
        for z in 0..4 {
            for i in 0..5 {
                let v = value_at(&xz.channels[1], z * 5 + i);
                assert_eq!(v, Stack::value(1, 1 + i, 2, z), "xz x {} z {z}", 1 + i);
            }
        }
        let yz = read_strip(&stack, &key(Axis::Yz, 7, 0, 6), 16).unwrap();
        assert_eq!((yz.width, yz.height), (4, 6));
        for y in 0..6 {
            for z in 0..4 {
                let v = value_at(&yz.channels[0], y * 4 + z);
                assert_eq!(v, Stack::value(0, 7, y, z), "yz y {y} z {z}");
            }
        }
        assert_eq!(
            *stack.requests.lock().unwrap(),
            8,
            "one line per plane and view"
        );
    }

    /// Exact 16-bit values are kept as such: 2 bytes a sample.
    #[test]
    fn a_strip_is_kept_compact() {
        let strip = read_strip(&Stack::new(), &key(Axis::Xz, 0, 0, 8), 16).unwrap();
        assert_eq!(strip.channels[0].size_in_bytes(), 8 * 4 * 2);
    }

    fn view(zoom: f32, offset_x: f32, offset_y: f32, w: f32, h: f32) -> ViewportState {
        ViewportState {
            viewport_width: w,
            viewport_height: h,
            zoom,
            offset_x,
            offset_y,
            mouse_pos_x: 0.0,
            mouse_pos_y: 0.0,
        }
    }

    fn shown(path: &PathBuf, level_size: (usize, usize)) -> ShownImage<'_> {
        ShownImage {
            path,
            series: 0,
            t: 0,
            depth: 4,
            full: (1000, 800),
            level: 1,
            level_size,
        }
    }

    /// The row / column at the crosshair, only the visible part, in the
    /// level's pixels.
    #[test]
    fn the_strips_are_the_crosshairs_line_in_the_visible_part() {
        let path = PathBuf::from("stack.tif");
        // Zoom 2, the image shifted 200 px left / 100 up: x 100..350 and
        // y 50..250 visible; level 1 is half the size.
        let (xz, yz) = strip_keys(
            &shown(&path, (500, 400)),
            &view(2.0, -200.0, -100.0, 500.0, 400.0),
            300.0,
            120.0,
        );
        let xz = xz.unwrap();
        assert_eq!((xz.axis, xz.line, xz.from, xz.to), (Axis::Xz, 60, 50, 175));
        let yz = yz.unwrap();
        assert_eq!((yz.axis, yz.line, yz.from, yz.to), (Axis::Yz, 150, 25, 125));
    }

    #[test]
    fn a_crosshair_outside_the_image_has_no_strip() {
        let path = PathBuf::from("stack.tif");
        let image = shown(&path, (1000, 800));
        let (xz, yz) = strip_keys(&image, &view(1.0, 0.0, 0.0, 500.0, 400.0), -1.0, 900.0);
        assert!(xz.is_none(), "below the image");
        assert!(yz.is_none(), "left of it");
        // The image scrolled away: nothing of it visible.
        let (xz, _) = strip_keys(&image, &view(1.0, -2000.0, 0.0, 500.0, 400.0), 10.0, 10.0);
        assert!(xz.is_none());
    }

    #[test]
    fn a_strip_renders_with_the_channels_look() {
        let strip = read_strip(&Stack::new(), &key(Axis::Xz, 0, 0, 8), 16).unwrap();
        let look = ChannelLook {
            visible: vec![0],
            histograms: [(
                0,
                Some(
                    evanalyzer_cfg::settings::images_settings::HistogramSettings {
                        min: 0.0,
                        max: 4000.0 / 65535.0,
                        ..Default::default()
                    },
                ),
            )]
            .into(),
            colors: [(0, [0.0, 1.0, 0.0])].into(),
            grayscale: false,
        };
        let buffer = render_strip(&strip, &look);
        assert_eq!((buffer.width(), buffer.height()), (8, 4));
        let pixels = buffer.as_slice();
        assert_eq!((pixels[0].r, pixels[0].b), (0, 0), "green only");
        assert!(
            pixels[8 * 3].g > pixels[0].g,
            "deeper planes are brighter here"
        );
    }
}
