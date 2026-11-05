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

    /// A strip of no planes: its size, nothing read.
    #[test]
    fn a_strip_of_no_planes_is_empty() {
        let stack = Stack::new();
        let mut empty = key(Axis::Yz, 3, 0, 6);
        empty.depth = 0;
        let strip = read_strip(&stack, &empty, 16).unwrap();
        assert!(strip.channels.is_empty());
        assert_eq!((strip.width, strip.height), (0, 6));
        assert_eq!(*stack.requests.lock().unwrap(), 0);
    }

    /// A label map (not one value per pixel to scale) is left out, the
    /// other channels kept.
    #[test]
    fn a_strip_leaves_out_channels_that_are_not_gray() {
        struct Labels(Stack);
        impl ImageSource for Labels {
            fn meta(&self) -> &ImageMeta {
                &self.0.meta
            }
            fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
                let mut channels = self.0.read_tile(req)?;
                let t = req.tile;
                channels[0].image = std::sync::Arc::new(ImageContainer::U32(ManagedImage {
                    data: Image::new(
                        ImageSize {
                            width: t.width,
                            height: t.height,
                        },
                        vec![7u32; t.width * t.height],
                    )
                    .unwrap(),
                    tile_offset: Default::default(),
                    plane: None,
                    dimensionality: ImageDimensionality::Planar,
                }));
                Ok(channels)
            }
        }
        let strip = read_strip(&Labels(Stack::new()), &key(Axis::Xz, 1, 0, 8), 16).unwrap();
        assert_eq!(strip.channels.len(), 1);
        assert_eq!(strip.channels[0].c_stack, 1);
        assert_eq!(
            value_at(&strip.channels[0], 8 + 3),
            Stack::value(1, 3, 1, 1)
        );
    }

    /// No zoom or no image: no strips.
    #[test]
    fn a_degenerate_view_has_no_strips() {
        let path = PathBuf::from("stack.tif");
        let image = shown(&path, (500, 400));
        let (xz, yz) = strip_keys(&image, &view(0.0, 0.0, 0.0, 500.0, 400.0), 10.0, 10.0);
        assert!(xz.is_none() && yz.is_none(), "zoom 0");
        let mut empty = shown(&path, (0, 0));
        empty.full = (0, 0);
        let (xz, yz) = strip_keys(&empty, &view(1.0, 0.0, 0.0, 500.0, 400.0), 0.0, 0.0);
        assert!(xz.is_none() && yz.is_none(), "an empty image");
    }

    /// The crosshair on the image's last pixel still has its line - on the
    /// level's last one, never past it.
    #[test]
    fn the_crosshair_on_the_last_pixel_takes_the_levels_last_line() {
        let path = PathBuf::from("stack.tif");
        let (xz, yz) = strip_keys(
            &shown(&path, (500, 400)),
            &view(1.0, 0.0, 0.0, 2000.0, 2000.0),
            999.9,
            799.9,
        );
        let (xz, yz) = (xz.unwrap(), yz.unwrap());
        assert_eq!((xz.line, xz.from, xz.to), (399, 0, 500));
        assert_eq!((yz.line, yz.from, yz.to), (499, 0, 400));
    }

    /// A channel without a display range (or not visible) stays black.
    #[test]
    fn a_strip_channel_without_a_display_range_is_black() {
        let strip = read_strip(&Stack::new(), &key(Axis::Xz, 0, 0, 8), 16).unwrap();
        let look = ChannelLook {
            visible: vec![0, 1],
            histograms: [(0, None)].into(),
            colors: Default::default(),
            grayscale: false,
        };
        let buffer = render_strip(&strip, &look);
        assert!(
            buffer
                .as_slice()
                .iter()
                .all(|p| (p.r, p.g, p.b) == (0, 0, 0))
        );
    }

    // -- rendering the views for the window ------------------------------------

    use crate::UiState;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::test_support::{
        grayscale_image_path, project_with_fixture_image, project_with_image_file,
        project_with_one_image, test_ui_window, ui_state_with_window,
    };
    use crate::editor::viewport_cache::{ReadContext, ViewportCache};
    use crate::editor::viewport_controller::ViewportController;
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{AppWindow, ViewportState as ViewportSlintState};
    use slint::ComponentHandle;
    use std::sync::Arc;

    struct Window {
        ui: AppWindow,
        app_state: Arc<UiState>,
        worker: ViewportWorker,
        strips: OrthoStrips,
    }

    /// A 400 × 300 view of `project`'s image at zoom 1, its sharp render
    /// read from level `level`, the views on.
    fn window_on(project: evanalyzer_app::project::ProjectWithRuntime, level: i32) -> Window {
        let ui = test_ui_window();
        let app_state = ui_state_with_window(&ui, project);
        let cache = Arc::new(ViewportCache::new(app_state.clone()));
        let viewport = Arc::new(ViewportController::new(
            ui.as_weak(),
            app_state.clone(),
            cache.clone(),
        ));
        {
            let mut vp = viewport.viewport_state.write().unwrap();
            vp.viewport_width = 400.0;
            vp.viewport_height = 300.0;
            vp.zoom = 1.0;
        }
        *cache.active_high_res_data.write().unwrap() = Some((
            Arc::new(Vec::new()),
            ReadContext {
                res_idx: level,
                ..Default::default()
            },
        ));
        let histogram = Arc::new(HistogramController::new(
            ui.as_weak(),
            app_state.clone(),
            viewport.clone(),
        ));
        viewport.set_ortho(true);
        let worker = ViewportWorker::new(app_state.clone(), viewport, histogram, cache);
        Window {
            ui,
            app_state,
            worker,
            strips: OrthoStrips::default(),
        }
    }

    impl Window {
        fn render(&mut self) {
            self.worker.render_ortho(&mut self.strips);
            drain_ui_queue();
        }
        fn view(&self) -> ViewportSlintState<'_> {
            self.ui.global::<ViewportSlintState>()
        }
        fn cross(&self, x: f32, y: f32) {
            self.worker.viewport_controller.set_ortho_cross(x, y);
        }
        fn source(&self) -> Arc<dyn ImageSource> {
            let path = self
                .app_state
                .get_project()
                .get_current_image_path_cloned()
                .unwrap();
            self.app_state.get_image_source(&path).unwrap()
        }
    }

    /// A strip kept from before, to see whether a render keeps it.
    fn kept_strip() -> Option<Strip> {
        Some(read_strip(&Stack::new(), &key(Axis::Xz, 0, 0, 8), 16).unwrap())
    }

    /// The fixture's z-stack: the crosshair's row across the planes below,
    /// its column beside - each sample its plane's, placed where the
    /// visible part lies, drawn with the z spacing.
    #[test]
    fn the_views_show_the_crosshairs_row_and_column_on_every_plane() {
        let mut w = window_on(project_with_fixture_image(), 0);
        let source = w.source();
        let info = source.meta().series[&0].clone();
        let (full_w, full_h) = (
            info.resolutions[&0].width as usize,
            info.resolutions[&0].height as usize,
        );
        let depth = info.nr_z_stacks as usize;
        assert!(depth > 1 && full_w > 3 && full_h > 3, "a z-stack");
        w.cross(1.5, 2.5);
        w.render();

        let (xz_len, yz_len) = (full_w.min(400), full_h.min(300));
        let xz = w.strips.xz.as_ref().expect("the XZ strip");
        assert_eq!((xz.key.line, xz.key.from, xz.key.to), (2, 0, xz_len));
        assert_eq!((xz.width, xz.height), (xz_len, depth));
        let yz = w.strips.yz.as_ref().expect("the YZ strip");
        assert_eq!((yz.key.line, yz.key.from, yz.key.to), (1, 0, yz_len));
        assert_eq!((yz.width, yz.height), (depth, yz_len));
        assert!(!xz.channels.is_empty());

        // Every sample as the reader has it at that pixel and plane.
        let plane = |z: usize| {
            source
                .read_tile(&TileRequest {
                    series: 0,
                    resolution_idx: 0,
                    z_projection: ZProjection::None,
                    z_range: Some(z as i32..=z as i32),
                    t_stack: 0,
                    tile: ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: full_w,
                        height: full_h,
                    },
                })
                .unwrap()
        };
        for z in 0..depth {
            let read = plane(z);
            for (c, channel) in xz.channels.iter().enumerate() {
                let ImageContainer::F32Gray(img) = &*read[c].image else {
                    panic!("gray");
                };
                let at = |x: usize, y: usize| img.as_slice()[y * full_w + x];
                for x in [0, 1, xz_len - 1] {
                    assert_eq!(value_at(channel, z * xz_len + x), at(x, 2), "xz c{c} z{z}");
                }
                for y in [0, 2, yz_len - 1] {
                    assert_eq!(
                        value_at(&yz.channels[c], y * depth + z),
                        at(1, y),
                        "yz c{c} z{z}"
                    );
                }
            }
        }

        let view = w.view();
        assert!(view.get_xz_ready() && view.get_yz_ready());
        assert_eq!((view.get_ortho_x(), view.get_ortho_y()), (1.5, 2.5));
        assert_eq!(view.get_ortho_depth(), depth as i32);
        assert_eq!(
            (view.get_xz_img_x(), view.get_xz_img_w()),
            (0.0, xz_len as f32)
        );
        assert_eq!(
            (view.get_yz_img_y(), view.get_yz_img_h()),
            (0.0, yz_len as f32)
        );
        let image = view.get_xz_image().size();
        assert_eq!((image.width, image.height), (xz_len as u32, depth as u32));
        let image = view.get_yz_image().size();
        assert_eq!((image.width, image.height), (depth as u32, yz_len as u32));
        let sizes = w.app_state.get_project().get_pixel_sizes();
        let z = sizes.z.expect("the fixture project's z spacing");
        assert_eq!(view.get_ortho_aniso(), z / sizes.x);
        assert!(view.get_ortho_z_known());
    }

    /// Unknown z spacing: planes drawn as thick as a pixel is wide, and
    /// the view says so.
    #[test]
    fn without_a_z_spacing_planes_are_drawn_a_pixel_thick() {
        let mut w = window_on(project_with_fixture_image(), 0);
        w.app_state
            .get_project_write()
            .set_global_pixel_size_settings(0.5, 0.5, None);
        w.render();
        assert_eq!(w.view().get_ortho_aniso(), 1.0);
        assert!(!w.view().get_ortho_z_known());
    }

    /// The same crosshair again: nothing read; another row: only the XZ
    /// view is read again.
    #[test]
    fn only_a_view_whose_line_changed_is_read_again() {
        let mut w = window_on(project_with_fixture_image(), 0);
        w.cross(1.0, 1.0);
        w.render();
        // Emptied behind the worker's back: a read fills them again.
        let empty = |strips: &mut OrthoStrips| {
            for strip in [&mut strips.xz, &mut strips.yz] {
                strip.as_mut().unwrap().channels.clear();
            }
        };
        empty(&mut w.strips);

        w.render();
        assert!(w.strips.xz.as_ref().unwrap().channels.is_empty(), "kept");
        assert!(w.strips.yz.as_ref().unwrap().channels.is_empty(), "kept");

        w.cross(1.0, 3.0);
        w.render();
        let xz = w.strips.xz.as_ref().unwrap();
        assert_eq!(xz.key.line, 3);
        assert!(!xz.channels.is_empty(), "the new row read");
        assert!(w.strips.yz.as_ref().unwrap().channels.is_empty(), "kept");
    }

    /// Off: the strips are freed, the images too.
    #[test]
    fn switched_off_the_views_free_their_strips() {
        let mut w = window_on(project_with_fixture_image(), 0);
        w.render();
        assert!(w.strips.xz.is_some() && w.strips.yz.is_some());

        w.worker.viewport_controller.set_ortho(false);
        w.render();
        assert!(w.strips.xz.is_none() && w.strips.yz.is_none());
        assert!(!w.view().get_xz_ready() && !w.view().get_yz_ready());
    }

    /// A crosshair moved off the image: neither view has a line.
    #[test]
    fn a_crosshair_off_the_image_shows_no_views() {
        let mut w = window_on(project_with_fixture_image(), 0);
        w.render();
        w.cross(-5.0, -5.0);
        w.render();
        assert!(w.strips.xz.is_none() && w.strips.yz.is_none());
        assert!(!w.view().get_xz_ready() && !w.view().get_yz_ready());
    }

    /// One plane: no views, nothing kept.
    #[test]
    fn a_single_plane_image_keeps_no_strips() {
        let mut w = window_on(project_with_image_file(grayscale_image_path()), 0);
        w.strips.xz = kept_strip();
        w.render();
        assert!(w.strips.xz.is_none());
    }

    /// Not drawable yet (no image, no sharp render, a level the image
    /// lacks, an unreadable file): the views are left as they are.
    #[test]
    fn without_what_to_draw_from_the_views_are_left_as_they_are() {
        let cases = [
            ("no image", window_on(Default::default(), 0)),
            ("no such level", window_on(project_with_fixture_image(), 9)),
            (
                "unreadable",
                window_on(
                    project_with_image_file(PathBuf::from("/nonexistent/stack.tif")),
                    0,
                ),
            ),
            ("one-image project without a file", {
                let w = window_on(project_with_one_image(), 0);
                *w.worker
                    .viewport_cache
                    .active_high_res_data
                    .write()
                    .unwrap() = None;
                w
            }),
        ];
        for (case, mut w) in cases {
            w.strips.xz = kept_strip();
            w.render();
            assert!(w.strips.xz.is_some(), "{case}: kept");
            assert!(!w.view().get_xz_ready(), "{case}: nothing posted");
        }
    }
}
