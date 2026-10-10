//! Plain image coordinates shared by the engine and its front ends - which
//! plane and which tile of an image a request or result refers to. Only the
//! addressing lives here; pixel data and image readers stay in
//! `evanalyzer_core`.

use serde::{Deserialize, Serialize};

#[derive(Default, Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImagePlane {
    pub z: i32,
    pub c: i32,
    pub t: i32,
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

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZProjection {
    #[default]
    None,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}
