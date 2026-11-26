//! # infer
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How Cellpose-SAM runs over an image - inside the engine slot: padded at
//! the bottom and right to at least one tile, in overlapping tiles (each
//! normalized while it is filled, its channels filled up), every tile's
//! output weighted by Cellpose's taper and blended.

use super::{CellposeSam, CellposeSamInput, CellposeSamRaw};
use crate::errors::NnError;

/// Fraction of a tile that overlaps its neighbour - Cellpose's own default
/// (`tile_overlap=0.1` in `CellposeModel.eval`).
const TILE_OVERLAP: f32 = 0.1;

/// The image through the network in tiles: the flows and the cell
/// probability at the network's size.
pub(super) fn in_tiles(
    net: &CellposeSam,
    input: &CellposeSamInput<'_>,
) -> Result<CellposeSamRaw, NnError> {
    let (w, h, channels) = (input.width, input.height, input.channels);
    let gray = &input.gray;
    // Padded at the bottom and right to at least one tile (zeros), in
    // tiles; each tile's output weighted by the taper, summed, divided by
    // the summed weights.
    let s = net.config.tile_size;
    let (ph, pw) = (h.max(s), w.max(s));
    let (ys, xs) = (tile_starts(ph, s), tile_starts(pw, s));
    let mask = taper_mask(s);
    let n_out = net.config.out_channels;
    let mut sum = vec![0.0f32; n_out * ph * pw];
    let mut weight = vec![0.0f32; ph * pw];
    let mut tile = vec![0.0f32; channels * s * s];
    for &y0 in &ys {
        for &x0 in &xs {
            tile.fill(0.0);
            for ty in 0..s {
                let y = y0 + ty;
                if y >= h {
                    break;
                }
                for tx in 0..s.min(w.saturating_sub(x0)) {
                    let v = gray[y * w + x0 + tx];
                    let v = input.normalization.map_or(v, |a| a.at(v));
                    tile[ty * s + tx] = v;
                    if input.replicate_gray_channel {
                        for c in 1..channels {
                            tile[(c * s + ty) * s + tx] = v;
                        }
                    }
                }
            }
            let out = net.try_forward_tile(&tile, channels)?;
            for o in 0..n_out {
                for ty in 0..s {
                    let row = ((o * ph) + y0 + ty) * pw + x0;
                    let src = (o * s + ty) * s;
                    for tx in 0..s {
                        sum[row + tx] += out[src + tx] * mask[ty * s + tx];
                    }
                }
            }
            for ty in 0..s {
                let row = (y0 + ty) * pw + x0;
                for tx in 0..s {
                    weight[row + tx] += mask[ty * s + tx];
                }
            }
        }
    }
    let plane = |o: usize, f: fn(f32) -> f32| -> Vec<f32> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(f(sum[(o * ph + y) * pw + x] / weight[y * pw + x]));
            }
        }
        out
    };
    Ok(CellposeSamRaw {
        flow_y: plane(0, |v| v),
        flow_x: plane(1, |v| v),
        cell_prob: plane(2, |v| 1.0 / (1.0 + (-v).exp())),
    })
}

/// Start offsets of the `tile`-wide tiles covering `len` pixels with
/// [`TILE_OVERLAP`] - Cellpose's `transforms.make_tiles` (in f32, as the
/// libtorch engine computed them). One tile at 0 when `len <= tile`.
pub(super) fn tile_starts(len: usize, tile: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let overlap = TILE_OVERLAP.clamp(0.05, 0.5);
    let n = (((1.0 + 2.0 * overlap) * len as f32) / tile as f32).ceil() as usize;
    if n <= 1 {
        return vec![0];
    }
    let span = (len - tile) as f32;
    (0..n)
        .map(|i| (span * i as f32 / (n - 1) as f32) as usize)
        .collect()
}

/// Cellpose's blend weight per pixel of a `tile`² tile
/// (`transforms._taper_mask`): ~1 in the middle, a sigmoid falling toward
/// (never to) 0 at the edges.
pub(super) fn taper_mask(tile: usize) -> Vec<f32> {
    const SIG: f32 = 7.5;
    let center = (tile as f32 - 1.0) / 2.0;
    let mask1d: Vec<f32> = (0..tile)
        .map(|i| {
            let xm = (i as f32 - center).abs();
            1.0 / (1.0 + ((xm - (tile as f32 / 2.0 - 20.0)) / SIG).exp())
        })
        .collect();
    let mut mask = Vec::with_capacity(tile * tile);
    for y in 0..tile {
        for x in 0..tile {
            mask.push(mask1d[y] * mask1d[x]);
        }
    }
    mask
}
