//! # unet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! U-Net (torch-em `UNet2d`, bioimage.io's U-Nets): the tile padded to a
//! multiple of 16, run in pieces, the prediction cropped back and reduced to
//! the foreground (and boundary) probability.

use crate::errors::NnError;
use crate::model::{Image, Model};
use crate::pieces::Pieces;
use burn::tensor::Device;
use std::path::Path;

/// Architecture, from the model file's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub depth: usize,
    pub initial_features: usize,
    pub gain: usize,
    pub norm: Norm,
}

/// The normalization between a U-Net's convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    None,
    Batch,
    /// Normalizes by the whole image: its pieces can't be exact.
    Instance,
    Group {
        groups: usize,
    },
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UNetSettings {
    /// Channel of the foreground probability (when the output has more than
    /// one; out-of-range values are clamped).
    pub foreground_channel: usize,
    /// The channels are class scores: softmax over them first. Else they are
    /// independent probability maps.
    pub softmax: bool,
    /// A boundary-probability channel (boundary-aware models).
    pub boundary_channel: Option<usize>,
    /// How large a tile runs in pieces.
    pub pieces: Pieces,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UNetOutput {
    /// Foreground probability per pixel (`width * height`).
    pub probabilities: Vec<f32>,
    /// Boundary probability per pixel, if a boundary channel was asked for
    /// and exists.
    pub boundary: Option<Vec<f32>>,
}

pub struct UNet {
    config: UNetConfig,
    // P4: the burn module (`UNet2d`).
}

impl UNet {
    pub fn config(&self) -> &UNetConfig {
        &self.config
    }
}

impl Model for UNet {
    type Settings = UNetSettings;
    type Output = UNetOutput;

    fn load(file: &Path, device: &Device) -> Result<Self, NnError> {
        let _ = (file, device);
        todo!("P4")
    }

    fn infer(&self, image: Image<'_>, settings: &UNetSettings) -> Result<UNetOutput, NnError> {
        let _ = (image, settings);
        todo!("P4")
    }
}
