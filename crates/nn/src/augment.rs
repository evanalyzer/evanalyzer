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
