//! # cellpose_sam
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose-SAM (Cellpose BSD-3 + SAM ViT-L encoder Apache-2.0): the image
//! resized to the size the model runs at, its channels filled up, run in
//! overlapping 256 x 256 tiles and blended back into flows and the cell
//! probability. The masks from the flows are core's.

use crate::errors::NnError;
use crate::model::{Image, Model};
use burn::tensor::Device;
use std::path::Path;

/// The only input side the network takes: its positional embedding is
/// fixed to a 256 x 256 grid.
pub const TILE_SIDE: usize = 256;

/// Architecture, from the model file's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellposeSamConfig {
    /// Channels the network takes (1-3).
    pub in_channels: usize,
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellposeSamSettings {
    /// The size the model runs at (the image is resized to it).
    pub net_width: usize,
    pub net_height: usize,
    /// Copy the gray image into every channel instead of zero-filling.
    pub replicate_gray_channel: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellposeSamOutput {
    /// Each `net_width * net_height`.
    pub flow_y: Vec<f32>,
    pub flow_x: Vec<f32>,
    /// After the sigmoid.
    pub cell_prob: Vec<f32>,
}

pub struct CellposeSam {
    config: CellposeSamConfig,
    // P6: the burn module.
}

impl CellposeSam {
    pub fn config(&self) -> &CellposeSamConfig {
        &self.config
    }
}

impl Model for CellposeSam {
    type Settings = CellposeSamSettings;
    type Output = CellposeSamOutput;

    fn load(file: &Path, device: &Device) -> Result<Self, NnError> {
        let _ = (file, device);
        todo!("P6")
    }

    fn infer(
        &self,
        image: Image<'_>,
        settings: &CellposeSamSettings,
    ) -> Result<CellposeSamOutput, NnError> {
        let _ = (image, settings);
        todo!("P6")
    }
}
