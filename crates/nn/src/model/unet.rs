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

mod network;

/// Architecture, from the model file's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub depth: usize,
    pub initial_features: usize,
    pub gain: usize,
    pub norm: Norm,
    pub final_activation: FinalActivation,
}

/// The activation after the last convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalActivation {
    None,
    Sigmoid,
    /// Over the channels.
    Softmax,
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
    network: network::UNet2d,
}

impl UNet {
    pub fn config(&self) -> &UNetConfig {
        &self.config
    }

    /// Convolutions in f16 (GPU tensor cores) - an experiment of the
    /// migration's GPU spike, not settled.
    pub fn set_half_convs(&mut self, half: bool) {
        self.network.half = half;
    }

    /// A U-Net of `config` with random weights - for benchmarks and tests
    /// until the weights load from files.
    pub fn with_random_weights(config: UNetConfig, device: &Device) -> Self {
        let network = network::UNet2d::new(&config, device);
        Self { config, network }
    }

    /// A U-Net of `config` with the weights of a torch-em `UNet2d` state
    /// dict in safetensors (PyTorch's tensor names, f32) - every tensor's
    /// shape checked against `config`.
    pub fn with_torch_em_weights(
        config: UNetConfig,
        safetensors: &[u8],
        device: &Device,
    ) -> Result<Self, NnError> {
        let file = safetensors::SafeTensors::deserialize(safetensors)
            .map_err(|e| NnError::InvalidModelFile(format!("no safetensors file: {e}")))?;
        let mut network = network::UNet2d::new(&config, device);
        network.load_torch_em(&file, device)?;
        Ok(Self { config, network })
    }

    /// The raw network output `[1, out_channels, height, width]` for a
    /// `[1, in_channels, height, width]` input (row-major planes), sides
    /// multiples of `2^depth` - no padding, no pieces.
    pub fn forward_raw(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        device: &Device,
    ) -> Vec<f32> {
        let shape = [1, self.config.in_channels, height, width];
        let x = burn::tensor::Tensor::<4>::from_data(
            burn::tensor::TensorData::new(input.to_vec(), shape),
            device,
        );
        let y = self.network.forward(x).into_data();
        y.as_slice::<f32>().expect("f32 output").to_vec()
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
