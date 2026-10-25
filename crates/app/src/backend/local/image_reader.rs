//! # image_reader
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::ImageChannel;
use crate::api::ImageMeta;
use crate::api::ImageSource;
use crate::api::TileRequest;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::{ImageReader, ReadMode};
use std::path::Path;

/// Opens a local image as an [`ImageSource`]: one thread-safe reader, parsed
/// once. [`AppHandle`](crate::workspace::AppHandle) caches it and serves both
/// metadata and tile reads from it, the channels of a tile read in parallel.
pub fn open_image_source(path: &Path) -> Result<ImageReader, InternalErrors> {
    ImageReader::new(&path.to_path_buf(), ReadMode::SplitChannels)
}

impl ImageSource for ImageReader {
    /// Parsed once when the reader was opened.
    fn meta(&self) -> &ImageMeta {
        self.get_image_meta()
    }

    /// Reads the channels in parallel.
    fn read_tile(&self, req: &TileRequest) -> Result<Vec<ImageChannel>, InternalErrors> {
        self.read_image_tile_combined_parallel(
            req.series,
            req.resolution_idx,
            req.z_projection.clone(),
            &req.z_range,
            req.t_stack,
            None,
            &req.tile,
        )
    }
}
