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
//! ran it (removed `crates/ai/src/torch/unet.rs`), in three steps
//! ([`Model`]):
//!
//! - `pre`: the image checked, the normalization of the model file (or the
//!   step's) computed over the tile ([`pre`]);
//! - `infer`: mirror-padded to a multiple of `2^depth` with at least
//!   [`MARGIN`] px of context, normalized on the way, run whole or in pieces
//!   (`crate::pieces`), cropped back and reduced to the foreground (and
//!   boundary) probability ([`infer`]);
//! - `post`: thresholded into a mask ([`post`]).
//!
//! The model file (`crate::weights`): architecture `unet2d`, layout version
//! 1 = torch-em's tensor names, `config` = [`UNetConfig`] as JSON (the
//! names of torch-em's `UNet2d` arguments).

use crate::errors::NnError;
use crate::model::{Image, Model, Normalization, Written};
use crate::pieces::Pieces;
use crate::weights::{ModelFile, ModelMetadata, NamedTensor};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;
use std::sync::OnceLock;

mod config;
mod infer;
mod network;
mod post;
mod pre;

pub use config::{ARCH, ARCH_VERSION, FinalActivation, Norm, Preprocessing, UNetConfig};

/// Mirrored context added around the tile, at least the halo bioimage.io
/// U-Nets declare (e.g. 16 px for romantic-shark, 8 px for joyful-pig).
pub const MARGIN: usize = 16;

/// What a U-Net's model file says without its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UNetHeader {
    pub config: UNetConfig,
    /// The normalization it was trained with ([`Normalization::ModelDefault`]).
    pub preprocessing: Preprocessing,
}

impl UNetHeader {
    fn from_metadata(metadata: &ModelMetadata) -> Result<Self, NnError> {
        metadata.expect_arch(ARCH, ARCH_VERSION)?;
        let config: UNetConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("U-Net config: {e}")))?;
        config.check()?;
        let preprocessing = Preprocessing::from_metadata(metadata.preprocessing.as_deref())?;
        Ok(Self {
            config,
            preprocessing,
        })
    }
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UNetParams {
    pub normalization: Normalization,
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
    /// Foreground from this probability on.
    pub probability_threshold: f32,
    /// A boundary from this boundary probability on - not foreground.
    pub boundary_threshold: f32,
    /// The value written for foreground pixels (background: 0).
    pub foreground: u32,
}

/// The tile as the network gets it - made by `pre` only.
pub struct UNetInput<'a> {
    pixels: &'a [f32],
    width: usize,
    height: usize,
    channels: usize,
    /// Applied while each piece is copied for the network.
    normalization: Option<pre::Affine>,
    readout: Readout,
    pieces: Pieces,
}

/// Which planes of the network's output are kept, and how.
#[derive(Debug, Clone, Copy)]
struct Readout {
    foreground_channel: usize,
    softmax: bool,
    boundary_channel: Option<usize>,
}

/// What the network gives back.
#[derive(Debug, Clone, PartialEq)]
pub struct UNetRaw {
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
    /// The memory of the GPU it runs on (`set_half_convs`), `None` on the
    /// CPU: the pieces must fit into it (`pieces::fit_gpu`).
    gpu_memory: Option<u64>,
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
            gpu_memory: None,
        }
    }

    pub fn config(&self) -> &UNetConfig {
        &self.config
    }

    pub fn preprocessing(&self) -> Preprocessing {
        self.preprocessing
    }

    /// The GPU mode (what `load` sets on a GPU): the network in f16 (its
    /// tensor cores, `network::set_half`) and the pieces sized to the GPU's
    /// memory (`pieces::fit_gpu`) - set once after loading.
    pub fn set_half_convs(&mut self, half: bool) {
        self.network.set_half(half);
        self.gpu_memory = if half {
            crate::runtime::gpu_memory(&self.device)
        } else {
            None
        };
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
        let UNetHeader {
            config,
            preprocessing,
        } = UNetHeader::from_metadata(&file.metadata)?;
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
        self.try_forward_raw(input, height, width, device)
            .expect("U-Net output")
    }

    /// [`Self::forward_raw`] with a failing device (out of memory) as an
    /// error.
    fn try_forward_raw(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<Vec<f32>, NnError> {
        let shape = [1, self.config.in_channels, height, width];
        let x = Tensor::<4>::from_data(TensorData::new(input.to_vec(), shape), device);
        crate::errors::read_f32(self.network.forward(x))
    }

    /// What this U-Net's model file says without its weights.
    pub fn to_header(&self) -> UNetHeader {
        UNetHeader {
            config: self.config.clone(),
            preprocessing: self.preprocessing,
        }
    }
}

/// `file`'s name in front of an invalid-file message.
fn in_file(file: &Path) -> impl Fn(NnError) -> NnError + '_ {
    move |e| match e {
        NnError::InvalidModelFile(m) => {
            NnError::InvalidModelFile(format!("{}: {m}", file.display()))
        }
        other => other,
    }
}

impl Model for UNet {
    const ARCH: &'static str = ARCH;
    type Header = UNetHeader;
    type Params = UNetParams;
    type Input<'a> = UNetInput<'a>;
    type Raw = UNetRaw;
    type Plan = ();

    /// Measured 1.61 KB for bioimage.io's `romantic-shark` (depth 4, 64
    /// filters, InstanceNorm) on burn's CPU backend (core's
    /// `tests::tile_ram::measure_ai_tile_commands`, 2026-11-25; libtorch:
    /// 1.55 KB for `joyful-pig`, 0.67 / 1.3 KB for U-Nets of depth 4 from
    /// 32 / 64 filters - BENCHMARK.md "AI models: RAM per inference").
    const NET_BYTES_PER_PIXEL: f64 = 1650.0;
    /// The foreground and boundary probabilities (core's `tests::tile_ram`).
    const POST_BYTES_PER_PIXEL: f64 = 8.0;

    fn header(file: &Path) -> Result<UNetHeader, NnError> {
        let metadata = crate::weights::read_header(file)?;
        UNetHeader::from_metadata(&metadata).map_err(in_file(file))
    }

    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        let mut net = Self::from_bytes(&bytes, device).map_err(in_file(file))?;
        net.set_half_convs(gpu);
        Ok(net)
    }

    fn pre<'a>(
        header: &UNetHeader,
        image: Image<'a>,
        params: &UNetParams,
    ) -> Result<(UNetInput<'a>, ()), NnError> {
        let pixels = pre::checked_pixels(&header.config, &image)?;
        let normalization = pre::normalization(pixels, params.normalization, header.preprocessing);
        let input = UNetInput {
            pixels,
            width: image.width,
            height: image.height,
            channels: image.channels,
            normalization,
            readout: Readout {
                foreground_channel: params.foreground_channel,
                softmax: params.softmax,
                boundary_channel: params.boundary_channel,
            },
            pieces: params.pieces,
        };
        Ok((input, ()))
    }

    fn infer(&self, input: &UNetInput<'_>) -> Result<UNetRaw, NnError> {
        let planes = infer::in_pieces(self, input)?;
        Ok(infer::raw(planes))
    }

    fn post(raw: UNetRaw, _: (), params: &UNetParams, out: &mut [u32]) -> Result<Written, NnError> {
        post::threshold(
            &raw.probabilities,
            raw.boundary.as_deref(),
            params.probability_threshold,
            params.boundary_threshold,
            params.foreground,
            out,
        )?;
        Ok(Written::Mask)
    }
}

#[cfg(test)]
mod tests;
