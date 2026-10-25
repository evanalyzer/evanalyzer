//! # commands
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod analyze;
pub(crate) mod common;
pub mod export;
pub mod jobs;
pub mod project;
#[cfg(test)]
pub(crate) mod test_support;
pub mod train_classifier;
pub mod view;
