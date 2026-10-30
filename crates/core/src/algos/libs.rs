//! # libs
//!
//! Building blocks the commands share - geometry with no pipeline types of
//! its own (each command stays one module; these are what several use).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

/// For every pixel the nearest marked pixel (SciPy's exact Euclidean
/// feature transform).
pub(crate) mod feature_transform;
pub(crate) mod row_runs;
/// A uniform grid over bounding boxes, to find the objects near one.
pub(crate) mod spatial_grid;
