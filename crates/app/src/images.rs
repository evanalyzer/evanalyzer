//! # images
//!
//! Images read tile by tile: [`ImageSource`], its metadata and tiles, and
//! the reader in this process ([`local`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod local;
mod source;
mod types;

pub use source::ImageSource;
pub use types::*;

// The public paths these had before (`evanalyzer_app::images::...`); they
// move into `images/pixels.rs` and `project/` with the project features.
pub use crate::workspace::images::{
    Erased, annotation_from_mask, erase_mask, stroke_mask, union_mask,
};
pub use crate::workspace::plate_layout::{PlateLayout, WellImages, plate_layout, texts_in_group};
