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

pub mod image_entry;
pub mod image_list;
pub(crate) mod local;
mod pixels;
mod source;
mod types;

pub use image_entry::ImageEntryExt;
pub use image_list::{GlobalImageSettingsExt, ImageListExt};
pub use source::ImageSource;
pub use types::*;

pub use pixels::{Erased, annotation_from_mask, erase_mask, stroke_mask, union_mask};
