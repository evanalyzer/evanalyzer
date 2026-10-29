//! # libs
//!
//! Building blocks the dataset steps share (each command stays one module;
//! these are what several use).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

/// The sparse linear assignment problem (Jonker & Volgenant).
pub(crate) mod lap;
/// The two linking steps of the LAP trackers (Jaqaman et al.): frame to
/// frame, then gap closing, merging and splitting.
pub(crate) mod linking;
