//! # images
//!
//! Image data front ends work with directly.
//!
//! Re-exported from `evanalyzer_core` rather than wrapped: they carry no I/O,
//! and a remote backend moves pixels as raw bytes (see
//! `backend::remote::wire::pixels`), so front ends never see the difference.
//! Still `core` types - a WASM client will need its own versions.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub use evanalyzer_core::{
    ImageChannel, ImageContainer, ImageInfo, ImageMeta, ManagedImage, Point2d, PyramidInfo,
};
