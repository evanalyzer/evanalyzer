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

use crate::errors::NnError;
use crate::train::SeededUniform;
use crate::weights::{NamedTensor, tensor_from};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Initializer, PaddingConfig2d};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

/// `conv` with start weights and bias from `random`, distributed as burn's
/// own `Conv2d` default (Kaiming uniform, gain 1/sqrt(3)): uniform in
/// ±1/sqrt(fan_in), fan_in = input channels x kernel height x width. The
/// layer is built with zeros first (cheap) and then filled here.
fn seeded(mut conv: Conv2d, random: &mut SeededUniform, device: &Device) -> Conv2d {
    let [out_channels, in_channels, kh, kw] = conv.weight.dims();
    let bound = 1.0 / ((in_channels * kh * kw) as f32).sqrt();
    let weight = random.values(out_channels * in_channels * kh * kw, bound);
    conv.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(weight, [out_channels, in_channels, kh, kw]),
        device,
    ));
    let bias = random.values(out_channels, bound);
    conv.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(bias, [out_channels]),
        device,
    )));
    conv
}

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

/// Architecture, stored in the model file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Refuses sizes no real SpotNet has - a model file is untrusted, and
    /// the network is allocated from these before any tensor is read.
    pub fn check(&self) -> Result<(), NnError> {
        let bad = |what: &str| Err(NnError::InvalidModelFile(format!("SpotNet config: {what}")));
        if !(1..=8).contains(&self.in_channels) {
            return bad("in_channels must be 1-8");
        }
        if !(1..=256).contains(&self.channels) {
            return bad("channels must be 1-256");
        }
        if !(1..=16).contains(&self.dilations.len()) {
            return bad("1-16 hidden layers");
        }
        if self.dilations.iter().any(|d| !(1..=64).contains(d)) {
            return bad("every dilation must be 1-64");
        }
        Ok(())
    }

    /// Theoretical receptive field in px: 1 + 2 * sum(dilations).
    pub fn receptive_field(&self) -> usize {
        let sum: usize = self.dilations.iter().sum();
        1 + 2 * sum
    }
}

impl SpotNet {
    /// A network of `config` with fresh start weights from `seed`: the same
    /// seed gives the same network, whatever else draws random numbers.
    pub(crate) fn new(config: &SpotNetConfig, seed: u64, device: &Device) -> Self {
        let mut random = SeededUniform::new(seed);
        let mut hidden = Vec::with_capacity(config.dilations.len());
        // The first layer reads the image channels, every later one the
        // feature maps of the layer before.
        let mut channels_from = config.in_channels;
        for &dilation in &config.dilations {
            let layer = Conv2dConfig::new([channels_from, config.channels], [3, 3])
                .with_dilation([dilation, dilation])
                .with_padding(PaddingConfig2d::Explicit(
                    dilation, dilation, dilation, dilation,
                ))
                .with_initializer(Initializer::Zeros)
                .init(device);
            hidden.push(seeded(layer, &mut random, device));
            channels_from = config.channels;
        }

        // Output layer: one channel, the spot-centre probability.
        let out = Conv2dConfig::new([channels_from, 1], [1, 1])
            .with_initializer(Initializer::Zeros)
            .init(device);
        let out = seeded(out, &mut random, device);

        Self { hidden, out }
    }

    /// The weights by name for the model file: `hidden.{i}.weight` /
    /// `.bias`, `out.weight` / `.bias` (convolutions as `[out, in, kh, kw]`).
    pub(crate) fn tensors(&self) -> Vec<NamedTensor> {
        let layers = self
            .hidden
            .iter()
            .enumerate()
            .map(|(i, conv)| (format!("hidden.{i}"), conv))
            .chain([("out".to_string(), &self.out)]);
        let mut out = Vec::new();
        for (name, conv) in layers {
            let weight = conv.weight.val();
            out.push(NamedTensor {
                name: format!("{name}.weight"),
                shape: weight.dims().to_vec(),
                values: weight.into_data().try_to_vec::<f32>().expect("f32 weights"),
            });
            if let Some(bias) = &conv.bias {
                let bias = bias.val();
                out.push(NamedTensor {
                    name: format!("{name}.bias"),
                    shape: bias.dims().to_vec(),
                    values: bias.into_data().try_to_vec::<f32>().expect("f32 weights"),
                });
            }
        }
        out
    }

    /// A network of `config` with the weights of `file` (see
    /// [`Self::tensors`]), every shape checked against `config`.
    pub(crate) fn from_tensors(
        config: &SpotNetConfig,
        file: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<Self, NnError> {
        config.check()?;
        let mut net = Self::new(config, 0, device);
        let load = |conv: &mut Conv2d, name: &str| -> Result<(), NnError> {
            let dims: [usize; 4] = conv.weight.dims();
            conv.weight =
                Param::from_tensor(tensor_from(file, &format!("{name}.weight"), dims, device)?);
            conv.bias = Some(Param::from_tensor(tensor_from(
                file,
                &format!("{name}.bias"),
                [dims[0]],
                device,
            )?));
            Ok(())
        };
        for (i, conv) in net.hidden.iter_mut().enumerate() {
            load(conv, &format!("hidden.{i}"))?;
        }
        load(&mut net.out, "out")?;
        Ok(net)
    }

    /// [B, in_channels, H, W] → [B, 1, H, W], values 0..1: the probability
    /// of a spot centre per pixel. For prediction.
    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        sigmoid(self.forward_logits(x))
    }

    /// The same before the sigmoid: any real number, 0 = 50 %, very negative
    /// = surely no spot. For training - the loss takes these, as
    /// `ln(sigmoid(z))` is only stable computed from `z` itself.
    pub(crate) fn forward_logits(&self, mut x: Tensor<4>) -> Tensor<4> {
        for layer in &self.hidden {
            x = relu(layer.forward(x));
        }
        self.out.forward(x)
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
        let net = SpotNet::new(&config(), 1, &device);
        let x = Tensor::<4>::zeros([1, 2, 64, 48], &device);
        assert_eq!(net.forward(x).dims(), [1, 1, 64, 48]);
    }
}
