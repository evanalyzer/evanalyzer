//! # images
//!
//! Image data front ends work with directly.
//!
//! Re-exported from `evanalyzer_core` rather than wrapped: they carry no I/O,
//! and a remote backend moves pixels as raw bytes (see
//! `remote::wire::pixels`), so front ends never see the difference.
//! Still `core` types - a WASM client will need its own versions.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::{ImageTile, ZProjection};
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

pub use evanalyzer_core::{
    ChannelInfo as FileChannel, DeviceInfo, ElectronMicroscopyInfo, EnvironmentInfo,
    ExperimenterInfo, FilterInfo, ImageChannel, ImageContainer, ImageInfo, ImageMeta,
    LightSourceInfo, ManagedImage, Point2d, PyramidInfo,
};

/// Which tile to read, at which pyramid level and plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TileRequest {
    pub series: i32,
    pub resolution_idx: i32,
    pub z_projection: ZProjection,
    pub z_range: Option<RangeInclusive<i32>>,
    pub t_stack: i32,
    pub tile: ImageTile,
}
