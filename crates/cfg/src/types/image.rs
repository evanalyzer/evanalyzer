// # image
//
// Plain image coordinates shared by the engine and its front ends - which
// plane and which tile of an image a request or result refers to. Only the
// addressing lives here; pixel data and image readers stay in
// `evanalyzer_core`.
//
// **Author:** Joachim Danmayr
// **Date:** 2026-10-10
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Which plane of an image - for a volume (see [`ImageDimensionality`]) its
/// first plane: it covers `z` to `z + depth - 1`.
#[derive(Default, Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImagePlane {
    pub z: i32,
    pub c: i32,
    pub t: i32,
}

/// Whether an image (or an object in it) is one plane or a volume - 2D or 3D,
/// spelled out at every match. A volume's planes are stored one after the
/// other (z-major), starting at its [`ImagePlane::z`].
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ImageDimensionality {
    /// One plane (2D).
    #[default]
    Planar,
    /// A volume (3D) of `depth` planes.
    Volume { depth: u32 },
}

impl ImageDimensionality {
    /// The number of planes: 1 for a planar image.
    pub fn depth(self) -> u32 {
        match self {
            Self::Planar => 1,
            Self::Volume { depth } => depth,
        }
    }

    /// 2 or 3 - as the results' `ndim` column stores it.
    pub fn ndim(self) -> u8 {
        match self {
            Self::Planar => 2,
            Self::Volume { .. } => 3,
        }
    }
}

#[derive(
    Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd,
)]
pub struct ImageTile {
    pub offset_x: usize,
    pub offset_y: usize,
    pub width: usize,
    pub height: usize,
}

/// How the planes of a z-stack are combined into one image - `None`: they
/// aren't, one plane is read.
#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
pub enum ZProjection {
    #[default]
    None,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}
