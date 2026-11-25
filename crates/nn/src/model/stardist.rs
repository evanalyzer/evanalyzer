//! # stardist
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist 2D: the object probability and the ray distances per grid cell.
//! The tile is padded (mirrored) to a multiple of 16 and run in pieces; the
//! star-convex polygons are core's.

use crate::errors::NnError;
use crate::model::{Image, Model};
use crate::pieces::Pieces;
use burn::tensor::Device;
use std::path::Path;

/// Architecture, from the model file's metadata (StarDist's `config.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarDistConfig {
    pub in_channels: usize,
    pub n_rays: usize,
    /// Output cell size in pixels, `[y, x]`.
    pub grid: [usize; 2],
    pub unet_depth: usize,
    pub unet_filter_base: usize,
    /// Filters of the extra convolution after the U-Net, `None` without.
    pub conv_after_unet: Option<usize>,
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StarDistSettings {
    /// How large a tile runs in pieces.
    pub pieces: Pieces,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StarDistOutput {
    /// `grid_height * grid_width` probabilities.
    pub prob: Vec<f32>,
    /// `n_rays` planes of ray distances, each `grid_height * grid_width`.
    pub dist: Vec<f32>,
    pub grid_height: usize,
    pub grid_width: usize,
    pub n_rays: usize,
}

pub struct StarDist {
    config: StarDistConfig,
    // P5: the burn module.
}

impl StarDist {
    pub fn config(&self) -> &StarDistConfig {
        &self.config
    }
}

impl Model for StarDist {
    type Settings = StarDistSettings;
    type Output = StarDistOutput;

    fn load(file: &Path, device: &Device) -> Result<Self, NnError> {
        let _ = (file, device);
        todo!("P5")
    }

    fn infer(
        &self,
        image: Image<'_>,
        settings: &StarDistSettings,
    ) -> Result<StarDistOutput, NnError> {
        let _ = (image, settings);
        todo!("P5")
    }
}
