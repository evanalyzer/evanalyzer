//! # source
//!
//! An opened image, as any backend offers it.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::{ImageChannel, ImageMeta, TileRequest};
use evanalyzer_cfg::core_types::InternalErrors;

/// An opened image, readable tile by tile.
pub trait ImageSource: Send + Sync {
    /// The image's metadata (series, pyramid levels, channels).
    fn meta(&self) -> &ImageMeta;

    /// Reads one tile of every channel of `req.series`.
    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors>;
}
