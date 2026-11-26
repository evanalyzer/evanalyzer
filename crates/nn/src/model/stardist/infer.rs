//! # infer
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How StarDist runs over a tile - inside the engine slot: in pieces where
//! the tile needs them (`crate::pieces`), each piece padded at the end
//! (mirrored), normalized on the way, through the network and cropped back
//! to the image's grid cells.

use super::pre::Affine;
use super::{StarDist, StarDistInput, StarDistRaw};
use crate::errors::NnError;
use crate::pieces::{self, Planes, pad_plane};

/// What the network takes on the GPU per pixel of its input, per 32 filters
/// of its first level: half of the 400 bytes the RAM plan has in f32 for
/// `2D_versatile_fluo` (32 filters) - f16.
const GPU_BYTES_PER_PIXEL_PER_32_FILTERS: f64 = 200.0;

/// The tile through the network - whole, or in pieces sized to the RAM (or
/// the GPU's memory) with the receptive field's halo.
pub(super) fn in_pieces(net: &StarDist, input: &StarDistInput<'_>) -> Result<Planes, NnError> {
    let (width, height, channels) = (input.width, input.height, input.channels);
    let piece = |p: &[f32], w: usize, h: usize| predict(net, p, w, h, input.normalization);
    // The receptive field's halo - measured once, on the network alone.
    let local_halo = || {
        if let Some(halo) = net.halo.get() {
            return Ok(*halo);
        }
        let probe = |p: &[f32], w: usize, h: usize| predict(net, p, w, h, None);
        let halo = pieces::receptive_halo(channels, probe)?;
        Ok(*net.halo.get_or_init(|| halo))
    };
    let gpu_bytes = GPU_BYTES_PER_PIXEL_PER_32_FILTERS * net.config.unet_filter_base as f64 / 32.0;
    let fitted = pieces::fit_gpu(input.pieces, net.gpu_memory, gpu_bytes, width, height);
    let plan = pieces::plan(width, height, fitted, local_halo, "StarDist")?;
    pieces::run_fitting(
        input.pixels,
        channels,
        width,
        height,
        plan,
        local_halo,
        piece,
        "StarDist",
    )
}

/// The planes as the network's answer.
pub(super) fn raw(planes: Planes) -> StarDistRaw {
    StarDistRaw {
        grid_height: planes.height,
        grid_width: planes.width,
        n_rays: planes.planes - 1,
        prob: planes.first,
        dist: planes.rest,
    }
}

/// The network on one `width x height` image (`in_channels` planes),
/// padded at the end, normalized and cropped back: probability, then the
/// rays.
fn predict(
    net: &StarDist,
    pixels: &[f32],
    width: usize,
    height: usize,
    normalization: Option<Affine>,
) -> Result<Planes, NnError> {
    let [div_y, div_x] = net.config.div_by();
    let (pw, ph) = (
        width.next_multiple_of(div_x),
        height.next_multiple_of(div_y),
    );
    let (right, bottom) = (pw - width, ph - height);
    let mut input = if (right, bottom) == (0, 0) {
        pixels.to_vec()
    } else {
        let mut padded = Vec::with_capacity(net.config.in_channels * pw * ph);
        for plane in pixels.chunks_exact(width * height) {
            pad_plane(plane, width, height, (0, right, 0, bottom), &mut padded);
        }
        padded
    };
    if let Some(affine) = normalization {
        affine.apply(&mut input);
    }
    let (prob, dist) = net.try_forward_raw(&input, ph, pw)?;
    let [gy, gx] = net.config.grid;
    let (ow, oh) = (pw / gx, ph / gy);
    // The grid cells of the image itself (not of the padding).
    let (grid_width, grid_height) = ((width / gx).clamp(1, ow), (height / gy).clamp(1, oh));
    let crop = |plane: &[f32], out: &mut Vec<f32>| {
        for y in 0..grid_height {
            out.extend_from_slice(&plane[y * ow..y * ow + grid_width]);
        }
    };
    let mut first = Vec::with_capacity(grid_width * grid_height);
    crop(&prob, &mut first);
    let mut rest = Vec::with_capacity(net.config.n_rays * grid_width * grid_height);
    for ray in dist.chunks_exact(ow * oh) {
        crop(ray, &mut rest);
    }
    Ok(Planes {
        planes: 1 + net.config.n_rays,
        width: grid_width,
        height: grid_height,
        first,
        rest,
    })
}
