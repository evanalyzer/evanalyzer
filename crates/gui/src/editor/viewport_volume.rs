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

    /// A stack read through [`ImageSource`]: `channels(z)` channels on
    /// plane z, channel c's sample at level-pixel (x, y) `value(c, x, y, z)`.
    struct Planes {
        meta: ImageMeta,
        channels: fn(usize) -> usize,
        value: fn(usize, usize, usize, usize) -> f32,
    }

    impl Planes {
        /// Levels `sizes` (level 0 first), `depth` planes of `bits`; the
        /// samples as an 8-bit reader normalizes them (`q × (1 / 255)`).
        fn new(sizes: &[(u64, u64)], depth: i32, bits: u16) -> Self {
            let mut info = ImageInfo {
                nr_c_stacks: 1,
                nr_z_stacks: depth,
                nr_t_stacks: 1,
                ..Default::default()
            };
            for (i, &(width, height)) in sizes.iter().enumerate() {
                info.resolutions.insert(
                    i as i32,
                    PyramidInfo {
                        nr_bits: bits,
                        color_channels: 1,
                        width,
                        height,
                        ..Default::default()
                    },
                );
            }
            Self {
                meta: ImageMeta {
                    series: BTreeMap::from([(0, info)]),
                    ..Default::default()
                },
                channels: |_| 1,
                value: |_, x, y, z| (x + 3 * y + 40 * z) as f32 * (1.0 / 255.0),
            }
        }

        fn info(&self) -> ImageInfo {
            self.meta.series[&0].clone()
        }
    }

    impl ImageSource for Planes {
        fn meta(&self) -> &ImageMeta {
            &self.meta
        }
        fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
            let z = *req.z_range.as_ref().unwrap().start() as usize;
            let t = req.tile;
            Ok((0..(self.channels)(z))
                .map(|c| {
                    let values = (0..t.width * t.height)
                        .map(|i| {
                            (self.value)(c, t.offset_x + i % t.width, t.offset_y + i / t.width, z)
                        })
                        .collect();
                    ImageChannel {
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
                        c_stack: c as i32,
                        name: String::new(),
                        is_rgb: false,
                    }
                })
                .collect())
        }
    }

    fn key(level: i32, xy_step: usize, z_step: usize) -> VolumeKey {
        VolumeKey {
            path: PathBuf::new(),
            series: 0,
            t: 0,
            level,
            xy_step,
            z_step,
        }
    }

    /// 8-bit samples kept as u8; every second plane and pixel of level 1,
    /// whose pixel is two of the full image's: the planes as thick as
    /// `aniso` × z_step over the 4 full-image pixels a voxel spans.
    #[test]
    fn a_reduced_volume_keeps_every_nth_plane_and_pixel() {
        let planes = Planes::new(&[(16, 12), (8, 6)], 6, 8);
        let v = read_volume(&planes, &planes.info(), &key(1, 2, 2), 3.0).unwrap();
        assert_eq!((v.width, v.height, v.depth), (4, 3, 3));
        assert_eq!(v.aniso, 3.0 * 2.0 / 4.0);
        let channel = &v.channels[0];
        assert!(matches!(channel.samples, Samples::U8 { .. }));
        assert_eq!(channel.size_in_bytes(), 4 * 3 * 3);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..4 {
                    assert_eq!(
                        channel.value_at(z * 12 + y * 4 + x).unwrap(),
                        (2 * x + 6 * y + 80 * z) as f32 * (1.0 / 255.0),
                        "({x}, {y}, {z})"
                    );
                }
            }
        }
    }

    /// Planes that aren't all compact: the volume is kept as read (f32),
    /// every value exact.
    #[test]
    fn a_volume_with_one_inexact_plane_is_kept_as_read() {
        let mut planes = Planes::new(&[(3, 2)], 3, 8);
        planes.value = |_, x, y, z| {
            let q = (x + 3 * y) as f32 * (1.0 / 255.0);
            if z == 1 { q + 1e-4 } else { q }
        };
        let v = read_volume(&planes, &planes.info(), &key(0, 1, 1), 1.0).unwrap();
        let channel = &v.channels[0];
        assert!(matches!(channel.samples, Samples::Image(_)), "f32");
        assert_eq!((channel.width, channel.height), (3, 6));
        for z in 0..3 {
            for i in 0..6 {
                assert_eq!(
                    channel.value_at(z * 6 + i).unwrap(),
                    (planes.value)(0, i % 3, i / 3, z)
                );
            }
        }
    }

    /// The same for 16-bit planes.
    #[test]
    fn a_16_bit_volume_with_one_inexact_plane_is_kept_as_read() {
        let mut planes = Planes::new(&[(3, 2)], 2, 16);
        planes.value = |_, x, y, z| {
            let q = (x + 3 * y) as f32 * (1.0 / 65535.0);
            if z == 1 { q + 1e-7 } else { q }
        };
        let v = read_volume(&planes, &planes.info(), &key(0, 1, 1), 1.0).unwrap();
        let channel = &v.channels[0];
        assert!(matches!(channel.samples, Samples::Image(_)), "f32");
        assert_eq!(channel.value_at(6 + 5).unwrap(), (planes.value)(0, 2, 1, 1));
        assert_eq!(channel.value_at(4).unwrap(), (planes.value)(0, 1, 1, 0));
    }

    /// A channel some plane lacks is left out - the others are kept.
    #[test]
    fn a_channel_missing_on_a_plane_is_left_out() {
        let mut planes = Planes::new(&[(2, 2)], 3, 8);
        planes.channels = |z| if z == 2 { 1 } else { 2 };
        let v = read_volume(&planes, &planes.info(), &key(0, 1, 1), 1.0).unwrap();
        assert_eq!(v.channels.len(), 1);
        assert_eq!(v.channels[0].c_stack, 0);
    }

    /// Asked for a level the image lacks, or no planes: an error, nothing
    /// read.
    #[test]
    fn a_volume_without_its_level_or_planes_is_an_error() {
        let planes = Planes::new(&[(4, 4)], 3, 8);
        assert!(read_volume(&planes, &planes.info(), &key(1, 1, 1), 1.0).is_err());
        let none = Planes::new(&[(4, 4)], 0, 8);
        assert!(read_volume(&none, &none.info(), &key(0, 1, 1), 1.0).is_err());
    }

    /// Every level smaller than the budget: the full image, as it is; no
    /// levels: no volume.
    #[test]
    fn a_small_image_is_read_whole_and_none_without_levels() {
        let planes = Planes::new(&[(200, 100), (100, 50)], 3, 8);
        assert_eq!(reduction(&planes.info()), Some((0, 1, 1)));
        // Only the longer side counts.
        let wide = Planes::new(&[(600, 10)], 300, 8);
        assert_eq!(reduction(&wide.info()), Some((0, 3, 2)));
        assert_eq!(reduction(&ImageInfo::default()), None);
    }

    /// A 16-bit and an f32 volume project like the 8-bit one: the
    /// brightest of each column, from the top.
    #[test]
    fn u16_and_f32_volumes_project_their_maximum() {
        let base = volume(4, 4, 3, |_, _, _| 0);
        let column_max = |x: usize, y: usize| (x + 4 * y) as f32;
        let value = |x: usize, y: usize, z: usize| column_max(x, y) * (z as f32 + 1.0) / 3.0;
        let mut u16_data = Vec::new();
        let mut f32_data = Vec::new();
        for z in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    u16_data.push((value(x, y, z) * 100.0).round() as u16);
                    f32_data.push(value(x, y, z) / 100.0);
                }
            }
        }
        let mut v = base;
        let template = v.channels[0].clone();
        v.channels = vec![
            ViewChannel {
                samples: Samples::U16 {
                    data: u16_data.into(),
                    inv: 1.0 / 10000.0,
                },
                ..template.clone()
            },
            ViewChannel {
                samples: Samples::Image(Arc::new(ImageContainer::F32Gray(ManagedImage {
                    data: Image::new(
                        ImageSize {
                            width: 4,
                            height: 12,
                        },
                        f32_data,
                    )
                    .unwrap(),
                    tile_offset: Default::default(),
                    plane: None,
                    dimensionality: ImageDimensionality::Volume { depth: 3 },
                }))),
                c_stack: 1,
                ..template
            },
        ];
        let (out_w, out_h) = (139, 139);
        let p = project_channels(&v, &Camera::default(), out_w, out_h);
        let side = 4.0 / (41f32).sqrt();
        let f = |i: usize| 0.5 - side / 2.0 + side * (i as f32 + 0.5) / 4.0;
        for y in 0..4 {
            for x in 0..4 {
                let px =
                    ((f(x) * out_w as f32) as usize) + ((f(y) * out_h as f32) as usize) * out_w;
                let expected = column_max(x, y) / 100.0;
                assert!((p[0][px] - expected).abs() < 1e-6, "u16 ({x}, {y})");
                assert!((p[1][px] - expected).abs() < 1e-6, "f32 ({x}, {y})");
            }
        }
    }

    /// A channel that isn't one value per pixel (RGB) projects to nothing.
    #[test]
    fn an_rgb_channel_projects_to_black() {
        let mut v = volume(2, 2, 2, |_, _, _| 9);
        v.channels[0].samples = Samples::Image(Arc::new(ImageContainer::F32Rgb(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: 2,
                    height: 4,
                },
                vec![1.0; 2 * 4 * 3],
            )
            .unwrap(),
            tile_offset: Default::default(),
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        })));
        let p = project_channels(&v, &Camera::default(), 16, 16);
        assert!(p[0].iter().all(|&v| v == 0.0));
    }

    /// Zoomed in 1.5 times: the volume covers 1.5 times the pixels across.
    #[test]
    fn zooming_in_enlarges_the_volume() {
        let v = volume(8, 8, 8, |_, _, _| 50);
        let lit = |zoom: f32| {
            let camera = Camera {
                zoom,
                ..Camera::default()
            };
            let p = &project_channels(&v, &camera, 400, 400)[0];
            (0..400).filter(|&x| p[200 * 400 + x] > 0.0).count() as i32
        };
        let (one, zoomed) = (lit(1.0), lit(1.5));
        assert!(
            (2 * zoomed - 3 * one).abs() <= 4,
            "zoom 1: {one}, zoom 1.5: {zoomed}"
        );
    }

    fn look(visible: Vec<i32>, histograms: &[(i32, Option<(f32, f32)>)]) -> ChannelLook {
        ChannelLook {
            visible,
            histograms: histograms
                .iter()
                .map(|&(c, h)| {
                    (
                        c,
                        h.map(|(min, max)| {
                            evanalyzer_cfg::settings::images_settings::HistogramSettings {
                                min,
                                max,
                                min_limit: 0.0,
                                max_limit: 1.0,
                            }
                        }),
                    )
                })
                .collect(),
            colors: [(0, [1.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.0])].into(),
            grayscale: false,
        }
    }

    /// Two channels: the visible one in its colour and range; hidden or
    /// without a range, a channel adds nothing.
    #[test]
    fn a_projection_renders_the_visible_channels_in_their_colours() {
        let mut v = volume(2, 2, 1, |_, _, _| 0);
        let mut second = v.channels[0].clone();
        second.c_stack = 1;
        v.channels.push(second);
        let projections = vec![vec![0.5; 4], vec![1.0; 4]];

        let red = render_projection(
            &v,
            &projections,
            &look(vec![0], &[(0, Some((0.0, 0.5))), (1, Some((0.0, 1.0)))]),
            2,
            2,
        );
        assert_eq!((red.width(), red.height()), (2, 2));
        assert!(
            red.as_slice()
                .iter()
                .all(|p| (p.r, p.g, p.b) == (255, 0, 0)),
            "full red, no blue: {:?}",
            red.as_slice()[0]
        );

        let both = render_projection(
            &v,
            &projections,
            &look(vec![0, 1], &[(0, None), (1, Some((0.0, 1.0)))]),
            2,
            2,
        );
        assert!(
            both.as_slice().iter().all(|p| (p.r, p.b) == (0, 255)),
            "channel 0 without a range: {:?}",
            both.as_slice()[0]
        );
    }

    /// At rest at most 1024 across, turning 384; never smaller than a
    /// pixel.
    #[test]
    fn the_render_is_capped_and_smaller_while_turning() {
        assert_eq!(render_size(400.0, 300.0, false), (400, 300));
        assert_eq!(render_size(2048.0, 1024.0, false), (1024, 512));
        assert_eq!(render_size(400.0, 300.0, true), (384, 288));
        assert_eq!(render_size(0.0, -5.0, false), (1, 1));
    }

    // -- rendering the 3D view for the window -----------------------------------

    use crate::UiState;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::test_support::{
        grayscale_image_path, project_with_fixture_image, project_with_image_file, test_ui_window,
        ui_state_with_window,
    };
    use crate::editor::viewport_cache::ViewportCache;
    use crate::editor::viewport_controller::ViewportController;
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{AppWindow, ViewportState as ViewportSlintState};
    use evanalyzer_cfg::settings::images_settings::TStackSettings;
    use slint::ComponentHandle;

    struct Window {
        ui: AppWindow,
        app_state: Arc<UiState>,
        worker: ViewportWorker,
        kept: Option<Volume>,
    }

    /// A 400 × 300 view of `project`'s image, the 3D view on (`turning`).
    fn window_on(project: evanalyzer_app::project::ProjectWithRuntime, turning: bool) -> Window {
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
        let histogram = Arc::new(HistogramController::new(
            ui.as_weak(),
            app_state.clone(),
            viewport.clone(),
        ));
        viewport.set_volume_view(Some((Camera::default(), turning)));
        ui.global::<ViewportSlintState>().set_volume_view(true);
        let worker = ViewportWorker::new(app_state.clone(), viewport, histogram, cache);
        Window {
            ui,
            app_state,
            worker,
            kept: None,
        }
    }

    impl Window {
        fn render(&mut self) {
            self.worker.render_volume(&mut self.kept);
            drain_ui_queue();
        }
        fn view(&self) -> ViewportSlintState<'_> {
            self.ui.global::<ViewportSlintState>()
        }
        fn image_size(&self) -> (u32, u32) {
            let size = self.view().get_volume_image().size();
            (size.width, size.height)
        }
        fn info(&self) -> ImageInfo {
            let path = self
                .app_state
                .get_project()
                .get_current_image_path_cloned()
                .unwrap();
            let source = self.app_state.get_image_source(&path).unwrap();
            source.meta().series[&0].clone()
        }
    }

    /// The fixture's z-stack: read once at its size, rendered for the view,
    /// with the z spacing known.
    #[test]
    fn the_3d_view_reads_the_stack_and_shows_it() {
        let mut w = window_on(project_with_fixture_image(), false);
        w.render();
        let info = w.info();
        let v = w.kept.as_ref().expect("the volume kept");
        let (level, xy_step, z_step) = reduction(&info).unwrap();
        assert_eq!(
            (v.key.level, v.key.xy_step, v.key.z_step),
            (level, xy_step, z_step)
        );
        assert_eq!(v.depth, (info.nr_z_stacks as usize).div_ceil(z_step));
        assert_eq!(
            v.width,
            (info.resolutions[&level].width as usize).div_ceil(xy_step)
        );
        assert!(!v.channels.is_empty());
        assert!(w.view().get_volume_ready());
        assert!(w.view().get_volume_z_known());
        assert_eq!(w.image_size(), (400, 300));
        let sizes = w.app_state.get_project().get_pixel_sizes();
        let pixel = xy_step as f32 * info.resolutions[&0].width as f32
            / info.resolutions[&level].width as f32;
        assert_eq!(v.aniso, sizes.z.unwrap() / sizes.x * z_step as f32 / pixel);
    }

    /// Shown with a channel's display range: the volume lights up; without
    /// one, black.
    #[test]
    fn the_3d_view_is_drawn_with_the_display_range() {
        let mut w = window_on(project_with_fixture_image(), false);
        let lit = |w: &Window| {
            let buffer = w.view().get_volume_image().to_rgb8().expect("rgb");
            buffer
                .as_slice()
                .iter()
                .filter(|p| p.r | p.g | p.b > 0)
                .count()
        };
        w.render();
        assert_eq!(lit(&w), 0, "no display range yet");
        w.app_state
            .get_project_write()
            .set_image_histogram_settings_for_channel(0, 0.0, 1e-6, 0.0, 1.0);
        w.render();
        assert!(lit(&w) > 0);
    }

    #[test]
    fn while_turning_the_3d_view_renders_smaller() {
        let mut w = window_on(project_with_fixture_image(), true);
        w.render();
        assert_eq!(w.image_size(), (384, 288));
    }

    /// The same image and time point: the volume kept is used, only its z
    /// spacing brought up to date; another time point is read anew.
    #[test]
    fn the_volume_is_read_again_only_for_another_time_point() {
        let mut w = window_on(project_with_fixture_image(), false);
        w.render();
        // Emptied behind the worker's back: a read fills it again.
        w.kept.as_mut().unwrap().channels.clear();
        let aniso = w.kept.as_ref().unwrap().aniso;
        w.app_state
            .get_project_write()
            .set_global_pixel_size_settings(0.5, 0.5, Some(4.0));
        w.render();
        let v = w.kept.as_ref().unwrap();
        assert!(v.channels.is_empty(), "kept, not read");
        assert_ne!(v.aniso, aniso, "the new z spacing");

        w.app_state
            .get_project_write()
            .set_image_t_stack(&TStackSettings {
                t_stack: 1,
                ..Default::default()
            });
        w.render();
        let v = w.kept.as_ref().unwrap();
        assert_eq!(v.key.t, 1);
        assert!(!v.channels.is_empty(), "read anew");
    }

    /// Unknown z spacing: planes as thick as a pixel, and the view says so.
    #[test]
    fn without_a_z_spacing_the_3d_view_says_so() {
        let mut w = window_on(project_with_fixture_image(), false);
        w.app_state
            .get_project_write()
            .set_global_pixel_size_settings(0.5, 0.5, None);
        w.render();
        assert!(!w.view().get_volume_z_known());
        let v = w.kept.as_ref().unwrap();
        assert_eq!(v.aniso, v.key.z_step as f32 / v.key.xy_step as f32);
    }

    /// Off, or an image of one plane: no volume kept.
    #[test]
    fn off_or_for_one_plane_no_volume_is_kept() {
        let mut w = window_on(project_with_fixture_image(), false);
        w.render();
        w.worker.viewport_controller.set_volume_view(None);
        w.render();
        assert!(w.kept.is_none(), "off");
        assert!(!w.view().get_volume_ready());

        let mut w = window_on(project_with_image_file(grayscale_image_path()), false);
        w.kept = Some(volume(2, 2, 2, |_, _, _| 1));
        w.render();
        assert!(w.kept.is_none(), "one plane");
        assert!(!w.view().get_volume_ready());
    }

    /// No image or an unreadable one: nothing drawn, the volume kept.
    #[test]
    fn without_an_image_to_read_the_3d_view_is_left_as_it_is() {
        for (case, project) in [
            ("no image", Default::default()),
            (
                "unreadable",
                project_with_image_file(PathBuf::from("/nonexistent/stack.tif")),
            ),
        ] {
            let mut w = window_on(project, false);
            w.kept = Some(volume(2, 2, 2, |_, _, _| 1));
            w.render();
            assert!(w.kept.is_some(), "{case}: kept");
            assert!(!w.view().get_volume_ready(), "{case}: nothing posted");
        }
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
