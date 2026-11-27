//! # network
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The SpotNet network in burn.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Device, Tensor};

///
// input  [2 ch: spot, reference]
// conv 3×3 d=1  → 16 ch, ReLU     ← only this layer changes
/// hidden:  conv 3×3 → 16 channels, ReLU
/// hidden:  conv 3×3 → 16 channels, ReLU   (repeat until the receptive field is ~20 px)
/// ...
/// output:  conv 1×1 →  1 channel, sigmoid  → [1, H × W] "spot probability" per pixel
///
///

#[derive(Module, Debug)]
pub(crate) struct SpotNet {
    /// Hidden 3x3 layers, each followed by ReLU.
    hidden: Vec<Conv2d>,
    /// 1x1 conv to one channel: the spot-centre probability.
    out: Conv2d,
}

/// Architecture, later stored in the model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpotNetConfig {
    /// 1 (spot channel) or 2 (spot + reference).
    pub in_channels: usize,
    /// Feature maps per hidden layer, e.g. 16.
    pub channels: usize,
    /// One entry per hidden 3x3 layer, e.g. [1, 1, 2, 4, 8, 1].
    pub dilations: Vec<usize>,
}

impl Default for SpotNetConfig {
    fn default() -> Self {
        Self {
            in_channels: 2,
            channels: 16,
            dilations: vec![1, 1, 2, 4, 8, 1],
        }
    }
}
impl SpotNetConfig {
    /// Theoretical receptive field in px: 1 + 2 * sum(dilations).
    pub fn receptive_field(&self) -> usize {
        let sum: usize = self.dilations.iter().sum();
        1 + 2 * sum
    }
}

impl SpotNet {
    pub(crate) fn new(config: &SpotNetConfig, device: &Device) -> Self {
        let mut hidden: Vec<Conv2d> = Vec::with_capacity(config.dilations.len());
        let mut channels_from = config.in_channels;

        for &dilation in &config.dilations {
            // First hidden layer gets two input channels from the in and has 16 output channels
            let layer = Conv2dConfig::new([channels_from, config.channels], [3, 3])
                .with_dilation([dilation, dilation])
                .with_padding(PaddingConfig2d::Explicit(
                    dilation, dilation, dilation, dilation,
                ))
                .init(device);
            hidden.push(layer);
            // We come from an hidden layer which always has 16
            channels_from = config.channels;
        }

        // Output layer has one output channel
        let out = Conv2dConfig::new([channels_from, 1], [1, 1])
            .with_dilation([1, 1])
            .init(device);

        Self { hidden, out }
    }

    /// [B, in_channels, H, W] → [B, 1, H, W], values 0..1.
    pub(crate) fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        for layer in &self.hidden {
            x = relu(layer.forward(x));
        }
        x = self.out.forward(x);
        sigmoid(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SpotNetConfig {
        SpotNetConfig::default()
    }

    #[test]
    fn the_receptive_field_is_35_px() {
        assert_eq!(config().receptive_field(), 35);
    }

    #[test]
    fn the_output_has_the_size_of_the_input_and_one_channel() {
        let device = Device::flex();
        let net = SpotNet::new(&config(), &device);
        let x = Tensor::<4>::zeros([1, 2, 64, 48], &device);
        assert_eq!(net.forward(x).dims(), [1, 1, 64, 48]);
    }
}
