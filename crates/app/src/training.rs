//! # training
//!
//! Training a pixel or object classifier ([`TrainingRequest`]) or a
//! deep-learning network ([`NetworkTrainingRequest`]), a training running
//! ([`RunningTraining`]), and training in this process ([`local`],
//! [`network`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod local;
pub(crate) mod network;
mod types;

pub use types::*;
