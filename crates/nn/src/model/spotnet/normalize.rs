//! # normalize
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! SpotNet's input normalisation: every channel of an image brought to about
//! 0 (background) to 1 (bright spots), so a model trained on 8-bit images
//! works on 16-bit ones and on dim or bright exposures alike.
//!
//! Per channel (a spot and a reference channel differ in brightness), over
//! the whole image - not per tile or patch: a tile's percentiles depend on
//! where it was cut, and the result would change with the tile size. The
//! same normalisation in training, validation and prediction: it is stored
//! in the model file.

use crate::errors::NnError;
use crate::libs::percentile::percentiles;
use crate::model::OwnedImage;
use serde::{Deserialize, Serialize};

/// The `low` and `high` percentiles (0-100) of each channel become 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PercentileNormalization {
    pub low: f64,
    pub high: f64,
}

impl Default for PercentileNormalization {
    /// 1 % / 99.8 %: background just below the low end, the brightest
    /// spots at the high end - single hot pixels don't squash the rest
    /// (CSBDeep's and StarDist's defaults).
    fn default() -> Self {
        Self {
            low: 1.0,
            high: 99.8,
        }
    }
}

impl PercentileNormalization {
    /// Refuses ranges that can't normalise (also from an untrusted file).
    pub fn check(&self) -> Result<(), NnError> {
        if !(0.0 <= self.low && self.low < self.high && self.high <= 100.0) {
            return Err(NnError::InvalidInput(format!(
                "normalisation: 0 <= low < high <= 100, not {} / {}",
                self.low, self.high
            )));
        }
        Ok(())
    }

    /// The scaling of each channel of `pixels` (`plane` values per channel,
    /// one after the other): its `low` and `high` percentiles become 0 and
    /// 1, `(x - lo) / (hi - lo + 1e-6)` - a channel without contrast stays
    /// finite; `None` for a channel of only NaN (left as it is).
    pub(crate) fn channels(&self, pixels: &[f32], plane: usize) -> Vec<Option<Affine>> {
        pixels
            .chunks(plane)
            .map(|channel| {
                let range = percentiles(channel, &[self.low, self.high])?;
                Some(Affine {
                    offset: range[0],
                    scale: 1.0 / (range[1] - range[0] + 1e-6),
                })
            })
            .collect()
    }

    /// `image` normalised in place, channel by channel (see
    /// [`Self::channels`]).
    pub fn apply(&self, image: &mut OwnedImage) {
        let plane = image.width * image.height;
        let affines = self.channels(&image.data, plane);
        apply_per_channel(&affines, &mut image.data, plane);
    }
}

/// `(x - offset) * scale`, in f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Affine {
    pub(crate) offset: f64,
    pub(crate) scale: f64,
}

/// `data` (`plane` values per channel) scaled in place by each channel's
/// `affines` entry (`None`: as it is).
pub(crate) fn apply_per_channel(affines: &[Option<Affine>], data: &mut [f32], plane: usize) {
    for (channel, affine) in data.chunks_mut(plane).zip(affines) {
        if let Some(Affine { offset, scale }) = *affine {
            for v in channel {
                *v = ((f64::from(*v) - offset) * scale) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PercentileNormalization;
    use crate::model::OwnedImage;

    fn image(channels: Vec<Vec<f32>>) -> OwnedImage {
        let plane = channels[0].len();
        OwnedImage {
            width: plane,
            height: 1,
            channels: channels.len(),
            data: channels.concat(),
        }
    }

    #[test]
    fn each_channel_goes_from_its_low_to_its_high_percentile() {
        // Channel 0 in 8-bit values, channel 1 in 16-bit: both end up 0..1.
        let full = PercentileNormalization {
            low: 0.0,
            high: 100.0,
        };
        let mut img = image(vec![vec![10.0, 110.0, 210.0], vec![1000.0, 3000.0, 5000.0]]);
        full.apply(&mut img);
        for (v, want) in img.data.iter().zip([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]) {
            assert!((v - want).abs() < 1e-5, "{:?}", img.data);
        }
    }

    #[test]
    fn the_same_picture_in_8_and_16_bit_gives_the_same_input() {
        let values: Vec<f32> = (0..200).map(|i| ((i * 37) % 97) as f32).collect();
        let mut a = image(vec![values.clone()]);
        let mut b = image(vec![values.iter().map(|v| v * 257.0 + 30.0).collect()]);
        let n = PercentileNormalization::default();
        n.apply(&mut a);
        n.apply(&mut b);
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn a_flat_channel_stays_finite() {
        let mut img = image(vec![vec![5.0; 10]]);
        PercentileNormalization::default().apply(&mut img);
        assert!(img.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn impossible_ranges_are_refused() {
        for (low, high) in [(50.0, 50.0), (60.0, 40.0), (-1.0, 99.0), (1.0, 101.0)] {
            assert!(
                PercentileNormalization { low, high }.check().is_err(),
                "{low} {high}"
            );
        }
        assert!(PercentileNormalization::default().check().is_ok());
    }
}
