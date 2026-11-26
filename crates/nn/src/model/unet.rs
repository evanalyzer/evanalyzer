//! # unet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! U-Net (torch-em `UNet2d`, bioimage.io's U-Nets) as the libtorch engine
//! ran it (removed `crates/ai/src/torch/unet.rs`): the tile normalized if the model
//! file asks for it, mirror-padded to a multiple of `2^depth` with at least
//! [`MARGIN`] px of context, run whole or in pieces (`crate::pieces`), the
//! prediction cropped back and reduced to the foreground (and boundary)
//! probability.
//!
//! The model file (`crate::weights`): architecture `unet2d`, layout version
//! 1 = torch-em's tensor names, `config` = [`UNetConfig`] as JSON (the
//! names of torch-em's `UNet2d` arguments).

use crate::errors::NnError;
use crate::model::{Image, Model};
use crate::pieces::{self, Pieces, Planes, pad_plane};
use crate::weights::{ModelFile, ModelMetadata, NamedTensor};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;
use std::sync::OnceLock;

mod config;
mod network;

pub use config::{ARCH, ARCH_VERSION, FinalActivation, Norm, Preprocessing, UNetConfig};

/// Mirrored context added around the tile, at least the halo bioimage.io
/// U-Nets declare (e.g. 16 px for romantic-shark, 8 px for joyful-pig).
pub const MARGIN: usize = 16;

impl Preprocessing {
    fn apply(self, pixels: &[f32]) -> Option<Vec<f32>> {
        match self {
            Self::None => None,
            Self::ZeroMeanUnitVariance => {
                let n = pixels.len().max(1) as f64;
                let mean = pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
                let var = pixels
                    .iter()
                    .map(|&v| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / n;
                let scale = 1.0 / (var.sqrt() + 1e-6);
                Some(
                    pixels
                        .iter()
                        .map(|&v| ((v as f64 - mean) * scale) as f32)
                        .collect(),
                )
            }
        }
    }
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
    /// A boundary-probability channel (boundary-aware models); ignored when
    /// the output has no such channel.
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
    preprocessing: Preprocessing,
    network: network::UNet2d,
    device: Device,
    /// The receptive-field halo (`crate::pieces::receptive_halo`), measured
    /// on the first tile that needs pieces.
    halo: OnceLock<Option<usize>>,
}

impl UNet {
    fn new(
        config: UNetConfig,
        preprocessing: Preprocessing,
        network: network::UNet2d,
        device: &Device,
    ) -> Self {
        Self {
            config,
            preprocessing,
            network,
            device: device.clone(),
            halo: OnceLock::new(),
        }
    }

    pub fn config(&self) -> &UNetConfig {
        &self.config
    }

    pub fn preprocessing(&self) -> Preprocessing {
        self.preprocessing
    }

    /// Convolutions in f16 (GPU tensor cores, `network::conv_forward`) -
    /// off unless asked for (not decided yet, see the migration plan).
    pub fn set_half_convs(&mut self, half: bool) {
        self.network.set_half(half);
    }

    /// A U-Net of `config` with random weights - for benchmarks and tests.
    pub fn with_random_weights(config: UNetConfig, device: &Device) -> Self {
        let network = network::UNet2d::new(&config, None, device);
        Self::new(config, Preprocessing::None, network, device)
    }

    /// A U-Net of `config` with the weights of a torch-em `UNet2d` state
    /// dict in safetensors (PyTorch's tensor names, f32) - every tensor's
    /// shape checked against `config`. No metadata needed: the dev
    /// converters' way in.
    pub fn with_torch_em_weights(
        config: UNetConfig,
        safetensors: &[u8],
        device: &Device,
    ) -> Result<Self, NnError> {
        config.check()?;
        let file = safetensors::SafeTensors::deserialize(safetensors)
            .map_err(|e| NnError::InvalidModelFile(format!("no safetensors file: {e}")))?;
        let mut network = network::UNet2d::new(&config, Some(Initializer::Zeros), device);
        network.load_torch_em(&file, device)?;
        Ok(Self::new(config, Preprocessing::None, network, device))
    }

    /// A U-Net from a model file's bytes (see the module docs).
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, NnError> {
        let file = ModelFile::parse(bytes)?;
        let metadata = file.expect_arch(ARCH, ARCH_VERSION)?;
        let config: UNetConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("U-Net config: {e}")))?;
        config.check()?;
        let preprocessing = Preprocessing::from_metadata(metadata.preprocessing.as_deref())?;
        let mut network = network::UNet2d::new(&config, Some(Initializer::Zeros), device);
        network.load_torch_em(&file.tensors, device)?;
        Ok(Self::new(config, preprocessing, network, device))
    }

    /// The tensors and metadata of this U-Net's model file (`info`: source,
    /// license, citation, ...) - for [`crate::weights::write`].
    pub fn to_file_contents(
        &self,
        info: std::collections::BTreeMap<String, String>,
    ) -> (Vec<NamedTensor>, ModelMetadata) {
        let metadata = ModelMetadata {
            arch: ARCH.into(),
            arch_version: ARCH_VERSION,
            config: serde_json::to_string(&self.config).expect("a config serializes"),
            preprocessing: match self.preprocessing {
                Preprocessing::None => None,
                Preprocessing::ZeroMeanUnitVariance => {
                    Some(Preprocessing::ZERO_MEAN_UNIT_VARIANCE.into())
                }
            },
            info,
        };
        (self.network.torch_em_tensors(), metadata)
    }

    /// Sets the preprocessing (the dev converters: from the source package).
    pub fn set_preprocessing(&mut self, preprocessing: Preprocessing) {
        self.preprocessing = preprocessing;
    }

    /// The raw network output `[1, out_channels, height, width]` for a
    /// `[1, in_channels, height, width]` input (row-major planes), sides
    /// multiples of `2^depth` - no padding, no pieces, no preprocessing.
    pub fn forward_raw(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        device: &Device,
    ) -> Vec<f32> {
        let shape = [1, self.config.in_channels, height, width];
        let x = Tensor::<4>::from_data(TensorData::new(input.to_vec(), shape), device);
        let y = self.network.forward(x).into_data();
        y.as_slice::<f32>().expect("f32 output").to_vec()
    }

    /// The network on one `width x height` image (`in_channels` planes),
    /// padded and cropped back: the foreground (and boundary) plane.
    fn predict(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
        settings: &UNetSettings,
    ) -> Result<Planes, NnError> {
        let multiple = self.config.size_multiple();
        let (top, bottom) = padding(height, multiple);
        let (left, right) = padding(width, multiple);
        let (pw, ph) = (left + width + right, top + height + bottom);
        let mut padded = Vec::with_capacity(self.config.in_channels * pw * ph);
        for plane in pixels.chunks_exact(width * height) {
            pad_plane(
                plane,
                width,
                height,
                (left, right, top, bottom),
                &mut padded,
            );
        }
        let output = self.forward_raw(&padded, ph, pw, &self.device);
        let channels = self.config.out_channels;
        if output.len() != channels * pw * ph {
            return Err(NnError::Internal(format!(
                "U-Net output has {} values, {} expected",
                output.len(),
                channels * pw * ph
            )));
        }
        // Channel `c`'s value at the tile's pixel (x, y).
        let at = |c: usize, x: usize, y: usize| output[(c * ph + y + top) * pw + x + left];
        let fg = settings.foreground_channel.min(channels - 1);
        let mut first = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = if channels > 1 && settings.softmax {
                    let max = (0..channels).map(|c| at(c, x, y)).fold(f32::MIN, f32::max);
                    let sum: f32 = (0..channels).map(|c| (at(c, x, y) - max).exp()).sum();
                    (at(fg, x, y) - max).exp() / sum
                } else {
                    at(fg, x, y)
                };
                first.push(value);
            }
        }
        // Boundary-aware models (e.g. affable-shark) predict an independent
        // boundary map: it opens thin gaps between touching objects.
        let rest = match settings.boundary_channel {
            Some(b) if channels > 1 && b < channels => {
                let mut rest = Vec::with_capacity(width * height);
                for y in 0..height {
                    for x in 0..width {
                        rest.push(at(b, x, y));
                    }
                }
                rest
            }
            _ => Vec::new(),
        };
        Ok(Planes {
            planes: if rest.is_empty() { 1 } else { 2 },
            width,
            height,
            first,
            rest,
        })
    }
}

/// Padding `(before, after)` for one side of `len` pixels: [`MARGIN`]
/// before, and after whatever brings the total up to a multiple of
/// `multiple` with at least `MARGIN` of context.
fn padding(len: usize, multiple: usize) -> (usize, usize) {
    let total = (len + 2 * MARGIN).next_multiple_of(multiple);
    (MARGIN, total - len - MARGIN)
}

impl Model for UNet {
    type Settings = UNetSettings;
    type Output = UNetOutput;

    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        let mut net = Self::from_bytes(&bytes, device).map_err(|e| match e {
            NnError::InvalidModelFile(m) => {
                NnError::InvalidModelFile(format!("{}: {m}", file.display()))
            }
            other => other,
        })?;
        net.set_half_convs(gpu);
        Ok(net)
    }

    fn infer(&self, image: Image<'_>, settings: &UNetSettings) -> Result<UNetOutput, NnError> {
        let (width, height, channels) = (image.width, image.height, image.channels);
        if width == 0 || height == 0 {
            return Err(NnError::InvalidInput("U-Net: an empty image".into()));
        }
        if channels != self.config.in_channels {
            return Err(NnError::InvalidInput(format!(
                "U-Net: the model takes {} channel(s), the image has {channels}",
                self.config.in_channels
            )));
        }
        let len = width * height * channels;
        if image.data.len() < len {
            return Err(NnError::InvalidInput(format!(
                "U-Net: the image has {} values, {len} expected",
                image.data.len()
            )));
        }
        let normalized = self.preprocessing.apply(&image.data[..len]);
        let pixels = normalized.as_deref().unwrap_or(&image.data[..len]);
        let predict = |p: &[f32], w: usize, h: usize| self.predict(p, w, h, settings);
        let plan = pieces::plan(
            width,
            height,
            settings.pieces,
            || {
                if self.config.sees_whole_image() {
                    return Ok(None);
                }
                if let Some(halo) = self.halo.get() {
                    return Ok(*halo);
                }
                let halo = pieces::receptive_halo(channels, predict)?;
                Ok(*self.halo.get_or_init(|| halo))
            },
            "U-Net",
        )?;
        let planes = match plan {
            Some((halo, piece)) => {
                pieces::run(pixels, channels, width, height, halo, piece, predict)?
            }
            None => predict(pixels, width, height)?,
        };
        Ok(UNetOutput {
            probabilities: planes.first,
            boundary: (planes.planes > 1).then_some(planes.rest),
        })
    }
}

#[cfg(test)]
mod tests;
