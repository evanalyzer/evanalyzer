use super::errors::NnError;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub mod cellpose_sam;
pub mod knn;
pub mod mlp;
pub mod rtree;
pub mod stardist;
pub mod unet;

pub trait Model: Sized + Send + Sync + 'static {
    /// What the model gives back - plain data.
    type Output;

    /// Builds the architecture from the file's metadata and loads the weights.
    fn load(file: &Path, config: &EngineConfig) -> Result<Self, NnError>;

    fn infer(image: Image<'_>) -> Result<Self::Output, NnError>;
}

#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: &'a [f32],
}

/// Which devices AI inference may use.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeviceSelection {
    /// Every GPU there is, else the CPU.
    #[default]
    Auto,
    Cpu,
    /// These GPUs (indexes); ones that don't exist are ignored.
    Gpus(Vec<usize>),
}

/// How an engine sets itself up, once per process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub devices: DeviceSelection,
    /// Inferences run at once per GPU (at least 1).
    pub gpu_slots: usize,
    /// CPU cores the engine may use.
    pub cores: usize,
}

/// The pieces of a U-Net or StarDist request: one side for a model that
/// sees a limited neighbourhood, one for a model that sees the whole image
/// - the engine knows which it is only after measuring it (`torch::pieces`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pieces {
    /// A limited neighbourhood: the result is the same up to float
    /// rounding whatever the side.
    pub local_side: usize,
    /// The whole image (InstanceNorm): the side is part of the result.
    pub whole_image_side: usize,
}
