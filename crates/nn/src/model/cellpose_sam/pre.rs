//! # pre
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose-SAM's pre-processing steps: the first channel checked,
//! Cellpose's `normalize99` computed over it (applied later, while each tile
//! is filled - [`Affine::at`]), resized to the size the model runs at.

use super::{CellposeSamParams, DEFAULT_PERCENTILES};
use crate::errors::NnError;
use crate::libs::percentile::percentiles;
use crate::model::{Image, Normalization};
use std::borrow::Cow;

/// The image's first channel - or why the model can't take it.
pub(super) fn first_channel<'a>(
    image: &Image<'a>,
    params: &CellposeSamParams,
) -> Result<&'a [f32], NnError> {
    let (width, height) = (image.width, image.height);
    let (w, h) = (params.net_width, params.net_height);
    if width == 0 || height == 0 || w == 0 || h == 0 || image.channels == 0 {
        return Err(NnError::InvalidInput("Cellpose: an empty image".into()));
    }
    if image.data.len() < width * height {
        return Err(NnError::InvalidInput(format!(
            "Cellpose: the image has {} values, {} expected",
            image.data.len(),
            width * height
        )));
    }
    Ok(&image.data[..width * height])
}

/// The normalization of `gray` asked for, `None` for none:
/// [`DEFAULT_PERCENTILES`] unless the step chose others.
pub(super) fn normalization(gray: &[f32], asked: Normalization) -> Option<Affine> {
    let (low, high) = match asked {
        Normalization::ModelDefault => DEFAULT_PERCENTILES,
        Normalization::Percentile { low, high } => (low, high),
        Normalization::None => return None,
    };
    normalize99(gray, low, high)
}

/// Cellpose's `transforms.normalize99`: the `low` and `high` percentiles
/// become 0 and 1; a tile without contrast becomes 0. Cellpose's limit for
/// "no contrast" (`x99 - x01 <= 1e-3`) is in raw grey levels, below one;
/// the pixels here are scaled to 0..1 by their bit depth, so any contrast
/// at all counts. `None` for a tile of NaN only.
fn normalize99(gray: &[f32], low: f64, high: f64) -> Option<Affine> {
    let range = percentiles(gray, &[low, high])?;
    let (lo, hi) = (range[0], range[1]);
    Some(Affine {
        offset: lo,
        scale: if hi > lo { 1.0 / (hi - lo) } else { 0.0 },
    })
}

/// `gray` (`from` = `(width, height)`) at the size `to`: borrowed when the
/// sizes are equal, else resized bilinearly with `align_corners = false`
/// (PyTorch's `upsample_bilinear2d`) exactly as burn's `interpolate` - which
/// the network ran before - computes it: source positions in f64, the four
/// weighted pixels summed in f32.
pub(super) fn resize_bilinear(
    gray: &[f32],
    from: (usize, usize),
    to: (usize, usize),
) -> Cow<'_, [f32]> {
    use rayon::prelude::*;
    if from == to {
        return Cow::Borrowed(gray);
    }
    let ((sw, sh), (dw, dh)) = (from, to);
    let columns: Vec<(usize, usize, f32)> = (0..dw).map(|x| source(x, sw, dw)).collect();
    let mut out = vec![0.0f32; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let (y0, y1, wy) = source(y, sh, dh);
        let (top, bottom) = (&gray[y0 * sw..(y0 + 1) * sw], &gray[y1 * sw..(y1 + 1) * sw]);
        for (value, &(x0, x1, wx)) in row.iter_mut().zip(&columns) {
            *value = top[x0] * (1.0 - wx) * (1.0 - wy)
                + top[x1] * wx * (1.0 - wy)
                + bottom[x0] * (1.0 - wx) * wy
                + bottom[x1] * wx * wy;
        }
    });
    Cow::Owned(out)
}

/// The two source pixels of output pixel `d` along an axis of `len` source
/// and `out_len` output pixels, and the weight of the second: position
/// `(d + 0.5) * len / out_len - 0.5`.
fn source(d: usize, len: usize, out_len: usize) -> (usize, usize, f32) {
    let ratio = len as f64 / out_len as f64;
    let position = (d as f64 + 0.5) * ratio - 0.5;
    let low = position.floor().max(0.0) as usize;
    let high = (low + 1).min(len - 1);
    (low, high, (position - low as f64).max(0.0) as f32)
}

/// `(x - offset) * scale`, in f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Affine {
    pub(super) offset: f64,
    pub(super) scale: f64,
}

impl Affine {
    pub(super) fn at(self, v: f32) -> f32 {
        ((f64::from(v) - self.offset) * self.scale) as f32
    }
}
