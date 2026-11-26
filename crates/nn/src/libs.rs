//! # libs
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Building blocks several models share - the same algorithm with the same
//! edge cases for every model using it (each model's pre- and
//! post-processing stays its own; these are what several need exactly
//! alike). Not part of nn's interface, never shared with core.

/// The values at percentiles of data, as `np.percentile` (linear).
pub(crate) mod percentile;
