//! # training
//!
//! Training a pixel or object classifier: what is asked for
//! ([`TrainingRequest`]), a training running ([`RunningTraining`]), and
//! training in this process ([`local`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod local;
mod types;

pub use types::*;
