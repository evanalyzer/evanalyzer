//! # cellpose_sam
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose-SAM (Cellpose BSD-3 + SAM ViT-L encoder Apache-2.0) as the
//! libtorch engine ran it (removed `crates/ai/src/torch/cellpose.rs`), in
//! three steps ([`Model`]):
//!
//! - `pre`: the image's first channel normalized like Cellpose's
//!   `normalize99` (its percentiles computed here, applied in `infer`) and
//!   resized (bilinear) to the size the model runs at ([`pre`]);
//! - `infer`: its channels filled up (zeros, or the image repeated), run in
//!   overlapping 256 x 256 tiles blended with Cellpose's taper
//!   (`transforms.average_tiles`) into the flows and the cell probability
//!   (after a sigmoid) ([`infer`]). The network takes exactly one tile
//!   size: its positional embedding is fixed to it;
//! - `post`: Cellpose's dynamics and mask building, its flow-error check,
//!   the masks scaled back to the image and objects below the minimum size
//!   removed ([`post`]).
//!
//! The model file (`crate::weights`): architecture `cellpose_sam`, see
//! `config.rs`.
//!
//! Portions of `post` ported from Cellpose, Copyright © 2020 Howard Hughes
//! Medical Institute, BSD-3-Clause - see the third-party notices (About >
//! Licenses).

use crate::errors::NnError;
use crate::model::{Image, Model, Normalization, Written};
use crate::weights::{ModelFile, ModelMetadata};
use burn::tensor::{Device, Tensor, TensorData};
use std::borrow::Cow;
use std::path::Path;

mod config;
mod infer;
mod network;
mod post;
mod pre;

pub use config::{ARCH, ARCH_VERSION, CellposeSamConfig};

/// The only input side the published network takes.
pub const TILE_SIDE: usize = 256;

/// The percentiles [`Normalization::ModelDefault`] stands for: Cellpose's
/// `normalize99` defaults, which its models were trained with.
pub const DEFAULT_PERCENTILES: (f64, f64) = (1.0, 99.0);

/// What a Cellpose-SAM's model file says without its weights.
#[derive(Debug, Clone, PartialEq)]
pub struct CellposeSamHeader {
    pub config: CellposeSamConfig,
}

impl CellposeSamHeader {
    fn from_metadata(metadata: &ModelMetadata) -> Result<Self, NnError> {
        metadata.expect_arch(ARCH, ARCH_VERSION)?;
        let config: CellposeSamConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("Cellpose-SAM config: {e}")))?;
        config.check()?;
        Ok(Self { config })
    }
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellposeSamParams {
    pub normalization: Normalization,
    /// The size the model runs at (the image is resized to it).
    pub net_width: usize,
    pub net_height: usize,
    /// Channels handed to the network (1 up to the model's `in_channels`).
    pub input_channels: usize,
    /// Copy the gray image into every channel instead of zero-filling.
    pub replicate_gray_channel: bool,
    /// Cell probability from which a pixel takes part in the dynamics.
    pub probability_threshold: f32,
    /// Euler steps along the flows (Cellpose: 200).
    pub flow_iterations: usize,
    /// Objects whose flows differ more from the predicted ones (mean
    /// squared error) are removed; 0 disables the check (Cellpose: 0.4).
    pub flow_threshold: f32,
    /// Objects of fewer pixels (at the image's size) are removed.
    pub min_object_size: usize,
    /// Cellpose's own dynamics and mask building (interpolated flows, seeds
    /// of more than 10 pixels) instead of every sink starting an object.
    pub cellpose_dynamics: bool,
}

/// The image as the network gets it - made by `pre` only.
pub struct CellposeSamInput<'a> {
    /// The first channel at the network's size.
    gray: Cow<'a, [f32]>,
    width: usize,
    height: usize,
    /// Channels handed to the network.
    channels: usize,
    replicate_gray_channel: bool,
    /// Applied while each tile is filled.
    normalization: Option<pre::Affine>,
}

/// What the network gives back, at the size it ran at.
#[derive(Debug, Clone, PartialEq)]
pub struct CellposeSamRaw {
    /// Each `net_width * net_height`.
    pub flow_y: Vec<f32>,
    pub flow_x: Vec<f32>,
    /// After the sigmoid.
    pub cell_prob: Vec<f32>,
}

/// The image's size and the size the model ran at, for `post`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellposeSamPlan {
    /// `(width, height)`.
    pub image: (usize, usize),
    /// `(width, height)`.
    pub net: (usize, usize),
}

pub struct CellposeSam {
    config: CellposeSamConfig,
    network: network::CellposeSamNet,
    device: Device,
}

impl CellposeSam {
    pub fn config(&self) -> &CellposeSamConfig {
        &self.config
    }

    /// Matrix products and convolutions in f16 (the GPU's tensor cores) or
    /// f32 - set once after loading (back to f32 keeps the f16 rounding).
    pub fn set_half(&mut self, half: bool) {
        self.network.set_half(half);
    }

    /// A Cellpose-SAM from a model file's bytes (see `config.rs`).
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, NnError> {
        let file = ModelFile::parse(bytes)?;
        let CellposeSamHeader { config } = CellposeSamHeader::from_metadata(&file.metadata)?;
        let network = network::CellposeSamNet::load(&file.tensors, &config, device)?;
        Ok(Self {
            config,
            network,
            device: device.clone(),
        })
    }

    /// The network on one tile `[C, S, S]` (row-major planes): `[3, S, S]`
    /// (dY, dX, the cell logit) - no resizing, no tiles, no sigmoid.
    pub fn forward_tile(&self, tile: &[f32], channels: usize) -> Vec<f32> {
        self.try_forward_tile(tile, channels)
            .expect("Cellpose-SAM output")
    }

    /// [`Self::forward_tile`] with a failing device (out of memory) as an
    /// error.
    fn try_forward_tile(&self, tile: &[f32], channels: usize) -> Result<Vec<f32>, NnError> {
        let s = self.config.tile_size;
        let x = Tensor::<4>::from_data(
            TensorData::new(tile.to_vec(), [1, channels, s, s]),
            &self.device,
        );
        crate::errors::read_f32(self.network.forward(x))
    }

    /// What this Cellpose-SAM's model file says without its weights.
    pub fn to_header(&self) -> CellposeSamHeader {
        CellposeSamHeader {
            config: self.config.clone(),
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

impl Model for CellposeSam {
    const ARCH: &'static str = ARCH;
    type Header = CellposeSamHeader;
    type Params = CellposeSamParams;
    type Input<'a> = CellposeSamInput<'a>;
    type Raw = CellposeSamRaw;
    type Plan = CellposeSamPlan;

    /// The tiles' outputs blended over the image (sums and weights): 46
    /// bytes per pixel measured with a stand-in (766 MB at 4096²). The ViT
    /// on one 256² tile is a fixed amount on top (core's `shared_ram_bytes`).
    const NET_BYTES_PER_PIXEL: f64 = 48.0;
    /// The flows and the cell probability, and the dynamics' planes - 42
    /// bytes per pixel (core's `tests::tile_ram`; the image is only scaled
    /// down, so per tile pixel at most).
    const POST_BYTES_PER_PIXEL: f64 = 42.0;

    fn header(file: &Path) -> Result<CellposeSamHeader, NnError> {
        let metadata = crate::weights::read_header(file)?;
        CellposeSamHeader::from_metadata(&metadata).map_err(in_file(file))
    }

    /// On the GPU the matrix products and convolutions in f16 (its tensor
    /// cores; 4.9x faster than f32 at 1024², as close to Cellpose's f32
    /// network as the bf16 model of the libtorch engine - BENCHMARK.md),
    /// on the CPU f32.
    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        let mut net = Self::from_bytes(&bytes, device).map_err(in_file(file))?;
        net.set_half(gpu);
        Ok(net)
    }

    /// Uses the image's first channel, as the libtorch engine did.
    fn pre<'a>(
        header: &CellposeSamHeader,
        image: Image<'a>,
        params: &CellposeSamParams,
    ) -> Result<(CellposeSamInput<'a>, CellposeSamPlan), NnError> {
        let first = pre::first_channel(&image, params)?;
        let normalization = pre::normalization(first, params.normalization);
        let net = (params.net_width, params.net_height);
        let gray = pre::resize_bilinear(first, (image.width, image.height), net);
        let input = CellposeSamInput {
            gray,
            width: net.0,
            height: net.1,
            channels: params.input_channels.clamp(1, header.config.in_channels),
            replicate_gray_channel: params.replicate_gray_channel,
            normalization,
        };
        let plan = CellposeSamPlan {
            image: (image.width, image.height),
            net,
        };
        Ok((input, plan))
    }

    fn infer(&self, input: &CellposeSamInput<'_>) -> Result<CellposeSamRaw, NnError> {
        infer::in_tiles(self, input)
    }

    fn post(
        raw: CellposeSamRaw,
        plan: CellposeSamPlan,
        params: &CellposeSamParams,
        out: &mut [u32],
    ) -> Result<Written, NnError> {
        post::check_size(out, plan.image)?;
        let cells = post::cell_pixels(&raw.cell_prob, params.probability_threshold);
        let mut labels = post::masks(&raw, &cells, plan.net, params);
        post::remove_bad_flow_masks(&mut labels, &raw, plan.net, params.flow_threshold);
        drop(raw);
        post::resize_nearest(&labels, plan.net, plan.image, out);
        let objects = post::remove_small_and_renumber(out, params.min_object_size);
        Ok(Written::Instances(objects))
    }
}

#[cfg(test)]
mod tests;
