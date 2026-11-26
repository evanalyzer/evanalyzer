//! # pre
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The U-Net's pre-processing steps: the image checked, the normalization
//! computed over the whole tile (before its pieces: one tile, one set of
//! statistics). The normalization itself is applied later, while each piece
//! is copied for the network ([`Affine::apply`]) - no normalized copy of
//! the tile.

use super::{Preprocessing, UNetConfig};
use crate::errors::NnError;
use crate::libs::percentile::percentiles;
use crate::model::{Image, Normalization};

/// The pixels of `image` the network gets (`in_channels` planes) - or why
/// it can't take them.
pub(super) fn checked_pixels<'a>(
    config: &UNetConfig,
    image: &Image<'a>,
) -> Result<&'a [f32], NnError> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width == 0 || height == 0 {
        return Err(NnError::InvalidInput("U-Net: an empty image".into()));
    }
    if channels != config.in_channels {
        return Err(NnError::InvalidInput(format!(
            "U-Net: the model takes {} channel(s), the image has {channels}",
            config.in_channels
        )));
    }
    let len = width * height * channels;
    if image.data.len() < len {
        return Err(NnError::InvalidInput(format!(
            "U-Net: the image has {} values, {len} expected",
            image.data.len()
        )));
    }
    Ok(&image.data[..len])
}

/// The normalization of `pixels` (all channels together) asked for, `None`
/// for none: the model file's `preprocessing` unless the step chose another.
pub(super) fn normalization(
    pixels: &[f32],
    asked: Normalization,
    preprocessing: Preprocessing,
) -> Option<Affine> {
    match asked {
        Normalization::ModelDefault => match preprocessing {
            Preprocessing::None => None,
            Preprocessing::ZeroMeanUnitVariance => Some(zero_mean_unit_variance(pixels)),
        },
        Normalization::Percentile { low, high } => scale_range(pixels, low, high),
        Normalization::None => None,
    }
}

/// bioimage.io's `zero_mean_unit_variance`: `(x - mean) / (std + 1e-6)`
/// (idempotent, so an already normalized image stays as it is).
fn zero_mean_unit_variance(pixels: &[f32]) -> Affine {
    let n = pixels.len().max(1) as f64;
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = pixels
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    Affine {
        offset: mean,
        scale: 1.0 / (var.sqrt() + 1e-6),
    }
}

/// bioimage.io's `scale_range`: the `low` and `high` percentiles become 0
/// and 1, `(x - lo) / (hi - lo + 1e-6)` - a tile without contrast stays
/// finite. `None` for a tile of NaN only.
fn scale_range(pixels: &[f32], low: f64, high: f64) -> Option<Affine> {
    let range = percentiles(pixels, &[low, high])?;
    Some(Affine {
        offset: range[0],
        scale: 1.0 / (range[1] - range[0] + 1e-6),
    })
}

/// `(x - offset) * scale`, in f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Affine {
    pub(super) offset: f64,
    pub(super) scale: f64,
}

impl Affine {
    pub(super) fn apply(self, values: &mut [f32]) {
        for v in values {
            *v = ((f64::from(*v) - self.offset) * self.scale) as f32;
        }
    }
}
