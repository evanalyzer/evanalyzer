//! # pre
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist's pre-processing steps: the image checked, csbdeep's percentile
//! normalization computed over the whole tile (before its pieces). It is
//! applied later, while each piece is copied for the network
//! ([`Affine::apply`]) - no normalized copy of the tile.

use super::{DEFAULT_PERCENTILES, StarDistConfig};
use crate::errors::NnError;
use crate::libs::percentile::percentiles;
use crate::model::{Image, Normalization};

/// The pixels of `image` the network gets (`in_channels` planes) - or why
/// it can't take them.
pub(super) fn checked_pixels<'a>(
    config: &StarDistConfig,
    image: &Image<'a>,
) -> Result<&'a [f32], NnError> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width == 0 || height == 0 {
        return Err(NnError::InvalidInput("StarDist: an empty image".into()));
    }
    if channels != config.in_channels {
        return Err(NnError::InvalidInput(format!(
            "StarDist: the model takes {} channel(s), the image has {channels}",
            config.in_channels
        )));
    }
    let len = width * height * channels;
    if image.data.len() < len {
        return Err(NnError::InvalidInput(format!(
            "StarDist: the image has {} values, {len} expected",
            image.data.len()
        )));
    }
    Ok(&image.data[..len])
}

/// The normalization of `pixels` (all channels together) asked for, `None`
/// for none: [`DEFAULT_PERCENTILES`] unless the step chose others.
pub(super) fn normalization(pixels: &[f32], asked: Normalization) -> Option<Affine> {
    let (low, high) = match asked {
        Normalization::ModelDefault => DEFAULT_PERCENTILES,
        Normalization::Percentile { low, high } => (low, high),
        Normalization::None => return None,
    };
    normalize_mi_ma(pixels, low, high)
}

/// csbdeep's `normalize` (`normalize_mi_ma`, `clip=False`): the `low` and
/// `high` percentiles become 0 and 1, `(x - lo) / (hi - lo + 1e-20)`.
/// `None` for a tile of NaN only.
fn normalize_mi_ma(pixels: &[f32], low: f64, high: f64) -> Option<Affine> {
    let range = percentiles(pixels, &[low, high])?;
    Some(Affine {
        offset: range[0],
        scale: 1.0 / (range[1] - range[0] + 1e-20),
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
