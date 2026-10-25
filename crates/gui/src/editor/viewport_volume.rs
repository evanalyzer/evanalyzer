//! # viewport_volume
//!
//! The 3D view of a z-stack: a maximum intensity projection (MIP) of the
//! volume, freely rotated with the mouse - each screen pixel the brightest
//! voxel along its ray, per channel, shown with the channels' colours and
//! display ranges as the image is.
//!
//! The volume is read once, reduced to at most [`MAX_XY`] pixels in x and y
//! (the pyramid level closest above, then every n-th pixel) and
//! [`MAX_DEPTH`] planes, kept compact (u8/u16, see [`ViewChannel`]) - e.g.
//! 256 × 256 × 57 planes in 3.7 MB per 8-bit channel. Planes are read in parallel and
//! made compact one by one: the volume is never held as `f32`. Rays are cast
//! in parallel, at a lower resolution while the view is turned. Nothing is
//! read, rendered or kept while the view is off.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::view_channel::{SampleSlice, Samples, ViewChannel};
use crate::editor::viewport_cache::read_subsampled;
use crate::editor::viewport_ortho::ChannelLook;
use crate::editor::viewport_worker::{
    ViewportWorker, channel_ctx, channel_display_color, prepare_image_channels_for_slint,
};
use evanalyzer_app::images::{ImageInfo, ImageSource, TileRequest};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors, ZProjection};
use log::{debug, warn};
use rayon::prelude::*;
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::path::PathBuf;
use std::sync::Arc;

/// At most this many pixels across in x and y.
pub(crate) const MAX_XY: usize = 256;
/// At most this many planes.
pub(crate) const MAX_DEPTH: usize = 256;
/// The render's longer side at most, at rest and while turning.
const RENDER_MAX: usize = 1024;
const RENDER_TURNING_MAX: usize = 384;

/// The camera: turned by `yaw` (around the screen's vertical axis) and
/// `pitch` (around its horizontal one), in radians; `zoom` 1 fits the
/// volume into the view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
        }
    }
}

/// What a volume is read from and how it is reduced.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VolumeKey {
    pub path: PathBuf,
    pub series: i32,
    pub t: i32,
    /// The pyramid level read, and every how many of its pixels one is kept.
    pub level: i32,
    pub xy_step: usize,
    /// Every how many planes one is kept.
    pub z_step: usize,
}

/// A volume as kept: per channel its planes one below the other (z-major),
/// `width` × `height` × `depth`; a plane is `aniso` times as thick as a
/// pixel is wide.
pub(crate) struct Volume {
    pub key: VolumeKey,
    pub channels: Vec<ViewChannel>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub aniso: f32,
}

/// The level and steps that keep a `nr_z_stacks`-plane series within
/// [`MAX_XY`] × [`MAX_XY`] × [`MAX_DEPTH`]: the smallest level still at
/// least [`MAX_XY`] wide or high (else the full image), every `xy_step`-th
/// pixel of it, every `z_step`-th plane.
pub(crate) fn reduction(info: &ImageInfo) -> Option<(i32, usize, usize)> {
    let longest = |(_, level): (&i32, &evanalyzer_app::images::PyramidInfo)| {
        level.width.max(level.height) as usize
    };
    let level = info
        .resolutions
        .iter()
        .filter(|l| longest(*l) >= MAX_XY)
        .min_by_key(|l| longest(*l))
        .or_else(|| info.resolutions.iter().min_by_key(|(i, _)| **i))?;
    let xy_step = longest(level).div_ceil(MAX_XY).max(1);
    let z_step = (info.nr_z_stacks.max(1) as usize)
        .div_ceil(MAX_DEPTH)
        .max(1);
    Some((*level.0, xy_step, z_step))
}

/// Reads `key`'s volume from `source` (series info `info`): its planes in
/// parallel, each made compact right away, then joined per channel. Channels
/// that aren't one value per pixel are left out, as the image view does.
/// `aniso`: the full image's z spacing over its pixel size.
pub(crate) fn read_volume(
    source: &dyn ImageSource,
    info: &ImageInfo,
    key: &VolumeKey,
    aniso: f32,
) -> Result<Volume, InternalErrors> {
    let level = info
        .resolutions
        .get(&key.level)
        .ok_or_else(|| InternalErrors::Internal("Volume: no such level".into()))?;
    let full_w = info.resolutions.get(&0).map_or(level.width, |l| l.width);
    let bits = info
        .resolutions
        .get(&0)
        .map_or(level.nr_bits, |l| l.nr_bits);
    let planes: Vec<i32> = (0..info.nr_z_stacks).step_by(key.z_step).collect();
    let request = |z: i32| TileRequest {
        series: key.series,
        resolution_idx: key.level,
        z_projection: ZProjection::None,
        z_range: Some(z..=z),
        t_stack: key.t,
        tile: ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: level.width as usize,
            height: level.height as usize,
        },
    };
    let read: Vec<Vec<ViewChannel>> = planes
        .par_iter()
        .map(|&z| {
            let channels = read_subsampled(source, &request(z), key.xy_step)?;
            Ok(channels
                .iter()
                .map(|channel| ViewChannel::from_image_channel(channel, bits))
                .collect())
        })
        .collect::<Result<_, InternalErrors>>()?;
    let Some(first) = read.first() else {
        return Err(InternalErrors::Internal("Volume: no planes".into()));
    };
    let (width, height) = first
        .first()
        .map_or((0, 0), |channel| (channel.width, channel.height));
    let depth = read.len();
    let channels = (0..first.len())
        .filter_map(|c| join_planes(read.iter().map(|plane| plane.get(c)), depth))
        .collect();
    // A pixel of the volume covers `xy_step` pixels of its level, which
    // covers `full_w / level.width` of the full image; a plane `z_step`
    // planes.
    let pixel = key.xy_step as f32 * full_w as f32 / level.width.max(1) as f32;
    Ok(Volume {
        key: key.clone(),
        channels,
        width,
        height,
        depth,
        aniso: aniso * key.z_step as f32 / pixel,
    })
}

/// One channel's planes joined into a volume - `None` if a plane misses it
/// or it isn't one value per pixel. Compact if every plane is, of one kind.
fn join_planes<'a>(
    planes: impl Iterator<Item = Option<&'a ViewChannel>> + Clone,
    depth: usize,
) -> Option<ViewChannel> {
    let planes: Vec<&ViewChannel> = planes.collect::<Option<_>>()?;
    let first = planes.first()?;
    let samples = match &first.samples {
        Samples::U8 { inv, .. } => planes
            .iter()
            .map(|p| match &p.samples {
                Samples::U8 { data, .. } => Some(&data[..]),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|parts| Samples::U8 {
                data: parts.concat().into(),
                inv: *inv,
            }),
        Samples::U16 { inv, .. } => planes
            .iter()
            .map(|p| match &p.samples {
                Samples::U16 { data, .. } => Some(&data[..]),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|parts| Samples::U16 {
                data: parts.concat().into(),
                inv: *inv,
            }),
        Samples::Image(_) => None,
    };
    let samples = match samples {
        Some(samples) => samples,
        // Not compact: as read (f32), if one value per pixel.
        None => {
            let mut values = Vec::with_capacity(first.width * first.height * depth);
            for plane in &planes {
                let slice = plane.slice()?;
                values.extend((0..slice.len()).map(|i| slice.get(i)));
            }
            let size = kornia_image::ImageSize {
                width: first.width,
                height: first.height * depth,
            };
            let image = kornia_image::Image::<f32, 1>::new(size, values).ok()?;
            Samples::Image(Arc::new(evanalyzer_app::images::ImageContainer::F32Gray(
                evanalyzer_app::images::ManagedImage {
                    data: image,
                    tile_offset: Default::default(),
                    plane: None,
                    dimensionality: evanalyzer_cfg::core_types::ImageDimensionality::Volume {
                        depth: depth as u32,
                    },
                },
            )))
        }
    };
    Some(ViewChannel {
        samples,
        width: first.width,
        height: first.height * depth,
        ..(*first).clone()
    })
}

/// The camera's rotation (volume -> view): pitch after yaw.
fn rotation(camera: &Camera) -> [[f32; 3]; 3] {
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    // Rx(pitch) * Ry(yaw)
    [
        [cy, 0.0, sy],
        [sp * sy, cp, -sp * cy],
        [-cp * sy, sp, cp * cy],
    ]
}

/// The maximum intensity projection of one channel's `voxels` (a volume of
/// `volume`'s size, value `to_f32` of a sample) seen by `camera`, into
/// `out` (`out_w` × `out_h`): each pixel the brightest voxel its ray
/// crosses (nearest voxel, one step per voxel), 0 where it misses.
fn project<T: Copy + PartialOrd + Send + Sync>(
    voxels: &[T],
    to_f32: impl Fn(T) -> f32 + Sync,
    volume: &Volume,
    camera: &Camera,
    out: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    let (w, h, d) = (volume.width, volume.height, volume.depth);
    let a = volume.aniso.max(1e-3);
    // Screen pixels per (x) voxel: zoom 1 fits the volume's diagonal.
    let diagonal = ((w * w + h * h) as f32 + (d as f32 * a).powi(2))
        .sqrt()
        .max(1.0);
    let scale = out_w.min(out_h) as f32 / diagonal * camera.zoom;
    let r = rotation(camera);
    // View -> volume: the transpose. A view point (u, v, s) is the volume
    // point r^T (u, v, s), in voxels with z divided by `a`, around the centre.
    let to_volume = |u: f32, v: f32, s: f32| {
        [
            r[0][0] * u + r[1][0] * v + r[2][0] * s + w as f32 / 2.0,
            r[0][1] * u + r[1][1] * v + r[2][1] * s + h as f32 / 2.0,
            (r[0][2] * u + r[1][2] * v + r[2][2] * s) / a + d as f32 / 2.0,
        ]
    };
    // The ray's direction (view z) in voxels, a step of at most one voxel.
    let dir = [r[2][0], r[2][1], r[2][2] / a];
    let longest = dir.iter().fold(0f32, |m, c| m.max(c.abs())).max(1e-6);
    let step = [dir[0] / longest, dir[1] / longest, dir[2] / longest];
    let extent = [w as f32, h as f32, d as f32];
    let plane = w * h;

    out.par_chunks_mut(out_w).enumerate().for_each(|(py, row)| {
        let v = (py as f32 + 0.5 - out_h as f32 / 2.0) / scale;
        for (px, value) in row.iter_mut().enumerate() {
            let u = (px as f32 + 0.5 - out_w as f32 / 2.0) / scale;
            let origin = to_volume(u, v, 0.0);
            // Where the ray is inside the box (slabs), in steps.
            let (mut near, mut far) = (f32::MIN, f32::MAX);
            for axis in 0..3 {
                if step[axis].abs() < 1e-9 {
                    if origin[axis] < 0.0 || origin[axis] >= extent[axis] {
                        near = f32::MAX;
                    }
                    continue;
                }
                let t0 = (0.0 - origin[axis]) / step[axis];
                let t1 = (extent[axis] - origin[axis]) / step[axis];
                near = near.max(t0.min(t1));
                far = far.min(t0.max(t1));
            }
            *value = 0.0;
            if near >= far {
                continue;
            }
            let mut brightest: Option<T> = None;
            let mut t = near.ceil();
            while t < far {
                let x = origin[0] + step[0] * t;
                let y = origin[1] + step[1] * t;
                let z = origin[2] + step[2] * t;
                t += 1.0;
                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }
                let (x, y, z) = (x as usize, y as usize, z as usize);
                if x >= w || y >= h || z >= d {
                    continue;
                }
                let sample = voxels[z * plane + y * w + x];
                if brightest.is_none_or(|b| sample > b) {
                    brightest = Some(sample);
                }
            }
            if let Some(b) = brightest {
                *value = to_f32(b);
            }
        }
    });
}

/// The projection of every channel of `volume`, `out_w` × `out_h` each.
pub(crate) fn project_channels(
    volume: &Volume,
    camera: &Camera,
    out_w: usize,
    out_h: usize,
) -> Vec<Vec<f32>> {
    volume
        .channels
        .iter()
        .map(|channel| {
            let mut out = vec![0f32; out_w * out_h];
            match (&channel.samples, channel.slice()) {
                (Samples::U8 { data, inv }, _) => project(
                    data,
                    |q| q as f32 * inv,
                    volume,
                    camera,
                    &mut out,
                    out_w,
                    out_h,
                ),
                (Samples::U16 { data, inv }, _) => project(
                    data,
                    |q| q as f32 * inv,
                    volume,
                    camera,
                    &mut out,
                    out_w,
                    out_h,
                ),
                (_, Some(SampleSlice::F32(data))) => {
                    project(data, |v| v, volume, camera, &mut out, out_w, out_h)
                }
                _ => {}
            }
            out
        })
        .collect()
}

/// The projections rendered like the image: visible channels with their
/// display range and colour.
pub(crate) fn render_projection(
    volume: &Volume,
    projections: &[Vec<f32>],
    look: &ChannelLook,
    out_w: usize,
    out_h: usize,
) -> SharedPixelBuffer<Rgb8Pixel> {
    let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(out_w as u32, out_h as u32);
    let contexts: Vec<_> = volume
        .channels
        .iter()
        .zip(projections)
        .filter_map(|(channel, values)| {
            let histogram = look.histograms.get(&channel.c_stack)?.clone()?;
            let color = channel_display_color(
                channel,
                look.colors.get(&channel.c_stack).copied(),
                look.grayscale,
            );
            Some(channel_ctx(
                SampleSlice::F32(values),
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

/// The render's size for a view of `view_w` × `view_h` screen pixels:
/// smaller while turning.
fn render_size(view_w: f32, view_h: f32, turning: bool) -> (usize, usize) {
    let cap = if turning {
        RENDER_TURNING_MAX
    } else {
        RENDER_MAX
    } as f32;
    let (w, h) = (view_w.max(1.0), view_h.max(1.0));
    let scale = (cap / w.max(h)).min(1.0);
    (((w * scale) as usize).max(1), ((h * scale) as usize).max(1))
}

impl ViewportWorker {
    /// Renders the 3D view for the current camera - reading the volume only
    /// when the image, series or time point changed. Off: frees it.
    pub(crate) fn render_volume(&self, kept: &mut Option<Volume>) {
        let Some(view) = self.viewport_controller.volume_view() else {
            *kept = None;
            return;
        };
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
        let source = match self.app_state.get_image_source(&path) {
            Ok(source) => source,
            Err(e) => {
                warn!("3D view: {e:?}");
                return;
            }
        };
        let Some(info) = source
            .meta()
            .series
            .get(&series)
            .filter(|info| info.nr_z_stacks > 1)
        else {
            *kept = None;
            return;
        };
        let Some((level, xy_step, z_step)) = reduction(info) else {
            return;
        };
        let key = VolumeKey {
            path,
            series,
            t,
            level,
            xy_step,
            z_step,
        };
        let (aniso, z_known) = match pixel_sizes.z {
            Some(z) if z > 0.0 && pixel_sizes.x > 0.0 => (z / pixel_sizes.x, true),
            _ => (1.0, false),
        };
        let started = std::time::Instant::now();
        if kept.as_ref().is_none_or(|v| v.key != key) {
            // The old one goes first: never two volumes at once.
            *kept = None;
            match read_volume(&*source, info, &key, aniso) {
                Ok(volume) => *kept = Some(volume),
                Err(e) => {
                    warn!("3D view: {e:?}");
                    return;
                }
            }
        }
        let Some(volume) = kept.as_mut() else {
            return;
        };
        // The z spacing may have been set since the volume was read.
        let pixel = key.xy_step as f32 * info.resolutions.get(&0).map_or(1.0, |l| l.width as f32)
            / info
                .resolutions
                .get(&level)
                .map_or(1.0, |l| l.width.max(1) as f32);
        volume.aniso = aniso * key.z_step as f32 / pixel;
        let read = started.elapsed();

        let (out_w, out_h) = render_size(view.width, view.height, view.turning);
        let projections = project_channels(volume, &view.camera, out_w, out_h);
        let buffer = render_projection(volume, &projections, &look, out_w, out_h);
        let kept_bytes: usize = volume.channels.iter().map(ViewChannel::size_in_bytes).sum();
        debug!(
            "3D view: {}x{}x{} voxels, read {read:?}, total {:?} for {out_w}x{out_h}{}, {kept_bytes} bytes kept",
            volume.width,
            volume.height,
            volume.depth,
            started.elapsed(),
            if view.turning { " (turning)" } else { "" }
        );
        self.viewport_controller.post_volume(buffer, z_known);
    }
}

/// What the 3D view shows and how big it is.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VolumeView {
    pub camera: Camera,
    /// The view's size, screen pixels.
    pub width: f32,
    pub height: f32,
    /// Being turned: a quicker, smaller render.
    pub turning: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::images::{
        ImageChannel, ImageContainer, ImageMeta, ManagedImage, PyramidInfo,
    };
    use evanalyzer_cfg::core_types::ImageDimensionality;
    use kornia_image::{Image, ImageSize};
    use std::collections::BTreeMap;
    use std::f32::consts::FRAC_PI_2;

    /// A u8 volume `w` × `h` × `d` with `value(x, y, z)`.
    fn volume(w: usize, h: usize, d: usize, value: impl Fn(usize, usize, usize) -> u8) -> Volume {
        let mut data = Vec::with_capacity(w * h * d);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data.push(value(x, y, z));
                }
            }
        }
        Volume {
            key: VolumeKey {
                path: PathBuf::new(),
                series: 0,
                t: 0,
                level: 0,
                xy_step: 1,
                z_step: 1,
            },
            channels: vec![ViewChannel {
                samples: Samples::U8 {
                    data: data.into(),
                    inv: 1.0 / 255.0,
                },
                width: w,
                height: h * d,
                tile_offset: Default::default(),
                color: [1.0, 1.0, 1.0],
                c_stack: 0,
                name: String::new(),
                is_rgb: false,
            }],
            width: w,
            height: h,
            depth: d,
            aniso: 1.0,
        }
    }

    /// The projection at the pixel nearest `(fx, fy)` (fractions of the
    /// render), in 0..255.
    fn at(p: &[f32], out_w: usize, out_h: usize, fx: f32, fy: f32) -> u8 {
        let x = ((fx * out_w as f32) as usize).min(out_w - 1);
        let y = ((fy * out_h as f32) as usize).min(out_h - 1);
        (p[y * out_w + x] * 255.0).round() as u8
    }

    /// Seen from the top (no turn), each pixel is the brightest voxel of
    /// its column - the z-projection's Max.
    #[test]
    fn from_the_top_it_is_the_max_over_the_planes() {
        // One bright voxel per column, at a column's own plane.
        let v = volume(8, 8, 8, |x, y, z| {
            if z == (x + y) % 8 {
                (10 + 8 * x + y) as u8
            } else {
                1
            }
        });
        // The diagonal fits the render: an 8 × 8 face covers 8 / √192 of it.
        let (out_w, out_h) = (139, 139);
        let p = &project_channels(&v, &Camera::default(), out_w, out_h)[0];
        let side = 8.0 / (192f32).sqrt();
        for y in 0..8 {
            for x in 0..8 {
                let f = |i: usize| 0.5 - side / 2.0 + side * (i as f32 + 0.5) / 8.0;
                assert_eq!(
                    at(p, out_w, out_h, f(x), f(y)),
                    (10 + 8 * x + y) as u8,
                    "({x}, {y})"
                );
            }
        }
        assert_eq!(at(p, out_w, out_h, 0.02, 0.02), 0, "outside the volume");
    }

    /// Turned a quarter around the vertical axis, the rays run along x:
    /// each pixel is the brightest of its row (x) at that height and depth.
    #[test]
    fn turned_a_quarter_the_rays_run_along_x() {
        // Bright only on the plane z = 0, in row y = 2: seen along x, a
        // bright line at y = 2 on one edge (z = 0) - the rest dark.
        let v = volume(8, 8, 8, |_, y, z| if z == 0 && y == 2 { 200 } else { 0 });
        let camera = Camera {
            yaw: FRAC_PI_2,
            ..Camera::default()
        };
        let (out_w, out_h) = (139, 139);
        let p = &project_channels(&v, &camera, out_w, out_h)[0];
        let side = 8.0 / (192f32).sqrt();
        let f = |i: usize| 0.5 - side / 2.0 + side * (i as f32 + 0.5) / 8.0;
        let row: Vec<u8> = (0..8).map(|i| at(p, out_w, out_h, f(i), f(2))).collect();
        assert_eq!(row.iter().filter(|&&b| b == 200).count(), 1, "{row:?}");
        assert!(row[0] == 200 || row[7] == 200, "at an edge: {row:?}");
        assert_eq!(at(p, out_w, out_h, f(4), f(5)), 0);
    }

    /// Thick planes (anisotropy 4): seen from the side, the stack is four
    /// times taller than it is deep in planes.
    #[test]
    fn thick_planes_stretch_the_stack() {
        let mut v = volume(8, 8, 2, |_, _, _| 100);
        v.aniso = 4.0;
        let camera = Camera {
            pitch: FRAC_PI_2,
            ..Camera::default()
        };
        let (out_w, out_h) = (200, 200);
        let p = &project_channels(&v, &camera, out_w, out_h)[0];
        // The lit rows: the stack's 2 planes × 4 = 8 voxels, as tall as wide.
        let lit_rows = (0..out_h)
            .filter(|&y| p[y * out_w + out_w / 2] > 0.0)
            .count();
        let lit_cols = (0..out_w)
            .filter(|&x| p[(out_h / 2) * out_w + x] > 0.0)
            .count();
        assert!(
            (lit_rows as i32 - lit_cols as i32).abs() <= 2,
            "rows {lit_rows}, columns {lit_cols}"
        );
    }

    /// A stack in a 16-bit file read as a compact volume: every plane once,
    /// u16, the planes in order.
    #[test]
    fn a_volume_is_read_plane_by_plane_and_kept_compact() {
        struct Stack(ImageMeta);
        impl ImageSource for Stack {
            fn meta(&self) -> &ImageMeta {
                &self.0
            }
            fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
                let z = *req.z_range.as_ref().unwrap().start() as usize;
                let t = req.tile;
                let values = (0..t.width * t.height)
                    .map(|i| (i + 100 * z) as f32 * (1.0 / 65535.0))
                    .collect();
                Ok(vec![ImageChannel {
                    image: Arc::new(ImageContainer::F32Gray(ManagedImage {
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
                    c_stack: 0,
                    name: String::new(),
                    is_rgb: false,
                }])
            }
        }
        let mut info = ImageInfo {
            nr_c_stacks: 1,
            nr_z_stacks: 5,
            nr_t_stacks: 1,
            ..Default::default()
        };
        info.resolutions.insert(
            0,
            PyramidInfo {
                nr_bits: 16,
                color_channels: 1,
                width: 4,
                height: 3,
                ..Default::default()
            },
        );
        let mut series = BTreeMap::new();
        series.insert(0, info.clone());
        let stack = Stack(ImageMeta {
            series,
            ..Default::default()
        });
        let (level, xy_step, z_step) = reduction(&info).unwrap();
        assert_eq!((level, xy_step, z_step), (0, 1, 1), "small: as it is");
        let key = VolumeKey {
            path: PathBuf::new(),
            series: 0,
            t: 0,
            level,
            xy_step,
            z_step,
        };
        let v = read_volume(&stack, &info, &key, 2.0).unwrap();
        assert_eq!((v.width, v.height, v.depth, v.aniso), (4, 3, 5, 2.0));
        let channel = &v.channels[0];
        assert_eq!(channel.size_in_bytes(), 4 * 3 * 5 * 2, "u16");
        for z in 0..5 {
            assert_eq!(
                channel.value_at(z * 12 + 7).unwrap(),
                (7 + 100 * z) as f32 * (1.0 / 65535.0)
            );
        }
    }

    /// A big stack is read reduced: the level closest above 256, every n-th
    /// pixel of it, every n-th plane.
    #[test]
    fn a_big_stack_is_reduced_to_the_budget() {
        let mut info = ImageInfo {
            nr_z_stacks: 600,
            ..Default::default()
        };
        for (i, size) in [
            (0, 8000u64),
            (1, 4000),
            (2, 2000),
            (3, 1000),
            (4, 500),
            (5, 250),
        ] {
            info.resolutions.insert(
                i,
                PyramidInfo {
                    width: size,
                    height: size / 2,
                    ..Default::default()
                },
            );
        }
        assert_eq!(reduction(&info), Some((4, 2, 3)));
    }

    /// For a feel of the speed: a 256³ volume, rendered at 1024² and while
    /// turning (384²). `cargo test --release -p evanalyzer_gui bench_volume
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_volume_projection() {
        let v = volume(256, 256, 256, |x, y, z| {
            ((x * 7 + y * 13 + z * 3) % 251) as u8
        });
        let camera = Camera {
            yaw: 0.6,
            pitch: 0.4,
            zoom: 1.0,
        };
        for (w, h) in [(1024, 1024), (384, 384)] {
            let _ = project_channels(&v, &camera, w, h);
            let started = std::time::Instant::now();
            for _ in 0..5 {
                let _ = project_channels(&v, &camera, w, h);
            }
            println!(
                "{w}x{h}: {:?} per frame, {} threads",
                started.elapsed() / 5,
                rayon::current_num_threads()
            );
        }
    }
}
