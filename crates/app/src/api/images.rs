//! Image data front ends work with directly.
//!
//! Re-exported from `evanalyzer_core` rather than wrapped: they carry no I/O,
//! and a remote backend moves pixels as raw bytes (see
//! `backend::remote::wire::pixels`), so front ends never see the difference.
//! Still `core` types - a WASM client will need its own versions.

pub use evanalyzer_core::{
    ImageChannel, ImageContainer, ImageMeta, ManagedImage, Point2d, PyramidInfo,
};
