//! # augment
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Augmentation for every network's training: random changes of a training
//! patch that don't change its answer, so the network sees more variety than
//! was annotated and learns what matters instead of the patch by heart.
//!
//! Applied to the image and its **raw labels** (instance ids, class masks,
//! spot points, training-area masks) - never to a model's own targets: a
//! Cellpose flow field or StarDist's ray distances are not plain pictures and
//! a rotated one would be wrong. Each model computes its target from the
//! augmented labels, so the target is right by construction.
//!
//! One file per kind of change, each drawn from [`SeededUniform`]
//! (reproducible: the same seed, the same changes):
//!
//! - [`rotate`] - rotations by multiples of 90° (lossless).
//! - [`flip`] - mirroring left-right and/or top-bottom (lossless).
//! - [`brightness`] - every channel multiplied by its own random factor
//!   (the input only).
//! - [`channel_shift`] - every channel but the first moved by a few whole
//!   pixels against it (the input only).
//!
//! [`SeededUniform`]: crate::train::SeededUniform

pub(crate) mod brightness;
pub(crate) mod channel_shift;
pub(crate) mod flip;
pub(crate) mod rotate;

use crate::train::SeededUniform;
use brightness::Brightness;
use channel_shift::ChannelShift;
use flip::Flip;
use rotate::Rotation;

/// Which augmentations a training applies - the same switches for every
/// network (part of each model's training settings).
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentParams {
    /// Random rotations by 0°, 90°, 180° and 270°.
    pub rotate: bool,
    /// Random mirroring left-right and/or top-bottom.
    pub flip: bool,
    /// Every channel multiplied by its own factor in `low..=high`;
    /// `(1.0, 1.0)` = off.
    pub brightness: (f32, f32),
    /// Every channel but the first moved by up to this many px against it;
    /// 0 = off.
    pub max_channel_shift: usize,
}

impl Default for AugmentParams {
    /// All on: orientation and brightness of a spot or cell never matter,
    /// and up to 3 px is the misalignment of real channels.
    fn default() -> Self {
        Self {
            rotate: true,
            flip: true,
            brightness: (0.7, 1.4),
            max_channel_shift: 3,
        }
    }
}

impl AugmentParams {
    /// Nothing changed - every patch as it is.
    pub fn none() -> Self {
        Self {
            rotate: false,
            flip: false,
            brightness: (1.0, 1.0),
            max_channel_shift: 0,
        }
    }
}

/// The random changes of one patch, drawn once and applied alike to the
/// image and its labels: first the geometry (flip, then rotation) to all of
/// them, then brightness and channel shift to the image only.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Augmentation {
    flip: Flip,
    rotation: Rotation,
    brightness: Brightness,
    shift: ChannelShift,
}

impl Augmentation {
    /// Draws the changes for a patch of `channels` channels.
    pub(crate) fn random(
        params: &AugmentParams,
        channels: usize,
        random: &mut SeededUniform,
    ) -> Self {
        let flip = if params.flip {
            Flip::random(random)
        } else {
            Flip::default()
        };
        let rotation = if params.rotate {
            Rotation::random(random)
        } else {
            Rotation::None
        };
        let (low, high) = params.brightness;
        Self {
            flip,
            rotation,
            brightness: Brightness::random(channels, low, high, random),
            shift: ChannelShift::random(channels, params.max_channel_shift, random),
        }
    }

    /// Width and height after the geometry.
    pub(crate) fn size(&self, width: usize, height: usize) -> (usize, usize) {
        self.rotation.size(width, height)
    }

    /// The image patch (`channels` planes of `width x height`) changed:
    /// geometry, brightness, channel shift.
    pub(crate) fn image(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Vec<f32> {
        let flipped = self.flip.planes(data, width, height, channels);
        let mut out = self.rotation.planes(&flipped, width, height, channels);
        let (w, h) = self.size(width, height);
        self.brightness.apply(&mut out, w * h);
        self.shift.apply(&mut out, w, h);
        out
    }

    /// A label plane (mask, ids, classes) with the geometry only.
    pub(crate) fn label<T: Copy>(&self, data: &[T], width: usize, height: usize) -> Vec<T> {
        let flipped = self.flip.plane(data, width, height);
        self.rotation.plane(&flipped, width, height)
    }

    /// A point (in px, a pixel's centre at +0.5) after the geometry - on the
    /// pixel [`Self::label`] moves its pixel to.
    pub(crate) fn point(&self, x: f32, y: f32, width: usize, height: usize) -> (f32, f32) {
        let (x, y) = self.flip.point(x, y, width, height);
        self.rotation.point(x, y, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::{AugmentParams, Augmentation};
    use crate::train::SeededUniform;

    #[test]
    fn without_augmentation_every_patch_stays_as_it_is() {
        let mut random = SeededUniform::new(1);
        let image: Vec<f32> = (0..12).map(|v| v as f32).collect();
        for _ in 0..20 {
            let a = Augmentation::random(&AugmentParams::none(), 2, &mut random);
            assert_eq!(a.image(&image, 3, 2, 2), image);
            assert_eq!(a.label(&[1u8, 2, 3, 4, 5, 6], 3, 2), vec![1, 2, 3, 4, 5, 6]);
            assert_eq!(a.point(0.5, 1.5, 3, 2), (0.5, 1.5));
        }
    }

    #[test]
    fn image_labels_and_points_move_together() {
        // A 5 x 3 image whose single channel is its pixel index (no
        // brightness, no shift): after any geometry the image, a label of
        // the same values and every moved point agree.
        let (width, height) = (5, 3);
        let index: Vec<usize> = (0..width * height).collect();
        let image: Vec<f32> = index.iter().map(|&v| v as f32).collect();
        let params = AugmentParams {
            brightness: (1.0, 1.0),
            max_channel_shift: 0,
            ..AugmentParams::default()
        };
        let mut random = SeededUniform::new(3);
        for _ in 0..50 {
            let a = Augmentation::random(&params, 1, &mut random);
            let (w, _) = a.size(width, height);
            let moved_image = a.image(&image, width, height, 1);
            let moved_label = a.label(&index, width, height);
            for (i, &v) in moved_label.iter().enumerate() {
                assert_eq!(moved_image[i], v as f32);
            }
            for y in 0..height {
                for x in 0..width {
                    let (px, py) = a.point(x as f32 + 0.5, y as f32 + 0.5, width, height);
                    let at = py.floor() as usize * w + px.floor() as usize;
                    assert_eq!(moved_label[at], y * width + x, "{a:?}");
                }
            }
        }
    }
}
