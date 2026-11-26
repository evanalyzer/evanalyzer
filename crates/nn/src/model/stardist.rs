//! # stardist
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist 2D as the libtorch engine ran it (removed `crates/ai/src/torch/stardist.rs`),
//! in three steps ([`Model`]):
//!
//! - `pre`: the image checked, the percentiles of the tile computed for
//!   csbdeep's normalization ([`pre`]);
//! - `infer`: padded at the end (mirrored, as StarDist) to the network's
//!   multiple, normalized on the way, run whole or in pieces
//!   (`crate::pieces`, the receptive field probed once), cropped to the
//!   image's grid cells: `side / grid` of them (rounded down, as before -
//!   StarDist itself keeps a last partial cell) - the object probability and
//!   the ray distances per grid cell ([`infer`]);
//! - `post`: star-convex polygons per candidate cell, non-maximum
//!   suppression, rasterized into instance labels ([`post`]).
//!
//! The model file (`crate::weights`): architecture `stardist2d`, see
//! `config.rs`.

use crate::errors::NnError;
use crate::model::{Image, Model, Normalization, Written};
use crate::pieces::Pieces;
use crate::weights::{ModelFile, ModelMetadata};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;
use std::sync::OnceLock;

mod config;
mod infer;
mod network;
mod post;
mod pre;

pub use config::{ARCH, ARCH_VERSION, StarDistConfig};

/// The percentiles [`Normalization::ModelDefault`] stands for: StarDist's
/// examples (`normalize(img, 1, 99.8)`), which its pretrained models were
/// trained with.
pub const DEFAULT_PERCENTILES: (f64, f64) = (1.0, 99.8);

/// What a StarDist's model file says without its weights.
#[derive(Debug, Clone, PartialEq)]
pub struct StarDistHeader {
    pub config: StarDistConfig,
}

impl StarDistHeader {
    fn from_metadata(metadata: &ModelMetadata) -> Result<Self, NnError> {
        metadata.expect_arch(ARCH, ARCH_VERSION)?;
        let config: StarDistConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("StarDist config: {e}")))?;
        config.check()?;
        Ok(Self { config })
    }
}

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarDistParams {
    pub normalization: Normalization,
    /// How large a tile runs in pieces.
    pub pieces: Pieces,
    /// Probability from which a grid cell is a candidate object centre.
    pub probability_threshold: f32,
    /// Pixel overlap (intersection / union) above which a lower-scoring
    /// candidate is suppressed by an overlapping higher-scoring one.
    pub nms_threshold: f32,
}

/// The tile as the network gets it - made by `pre` only.
pub struct StarDistInput<'a> {
    pixels: &'a [f32],
    width: usize,
    height: usize,
    channels: usize,
    /// Applied while each piece is copied for the network.
    normalization: Option<pre::Affine>,
    pieces: Pieces,
}

/// What the network gives back.
#[derive(Debug, Clone, PartialEq)]
pub struct StarDistRaw {
    /// `grid_height * grid_width` probabilities.
    pub prob: Vec<f32>,
    /// `n_rays` planes of ray distances, each `grid_height * grid_width`.
    pub dist: Vec<f32>,
    pub grid_height: usize,
    pub grid_width: usize,
    pub n_rays: usize,
}

/// The tile's size, for `post`.
#[derive(Debug, Clone, Copy)]
pub struct StarDistPlan {
    width: usize,
    height: usize,
}

pub struct StarDist {
    config: StarDistConfig,
    network: network::StarDistNet,
    device: Device,
    /// The receptive-field halo, measured on the first tile that needs
    /// pieces.
    halo: OnceLock<Option<usize>>,
    /// The memory of the GPU it runs on (`set_half_convs`), `None` on the
    /// CPU: the pieces must fit into it (`pieces::fit_gpu`).
    gpu_memory: Option<u64>,
}

impl StarDist {
    pub fn config(&self) -> &StarDistConfig {
        &self.config
    }

    /// The GPU mode (what `load` sets on a GPU): the network in f16 (its
    /// tensor cores, the network's `set_half`) and the pieces sized to the
    /// GPU's memory (`pieces::fit_gpu`) - set once after loading.
    pub fn set_half_convs(&mut self, half: bool) {
        self.network.set_half(half);
        self.gpu_memory = if half {
            crate::runtime::gpu_memory(&self.device)
        } else {
            None
        };
    }

    /// A StarDist from a model file's bytes (see `config.rs`).
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, NnError> {
        let file = ModelFile::parse(bytes)?;
        let StarDistHeader { config } = StarDistHeader::from_metadata(&file.metadata)?;
        let network = network::StarDistNet::load(&file.tensors, &config, device)?;
        Ok(Self {
            config,
            network,
            device: device.clone(),
            halo: OnceLock::new(),
            gpu_memory: None,
        })
    }

    /// The raw network outputs for `[1, C, H, W]` (sides multiples of
    /// `config.div_by()`): the probability plane and the ray planes at the
    /// grid's cells.
    pub fn forward_raw(&self, input: &[f32], height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
        self.try_forward_raw(input, height, width)
            .expect("StarDist output")
    }

    /// [`Self::forward_raw`] with a failing device (out of memory) as an
    /// error.
    fn try_forward_raw(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), NnError> {
        let shape = [1, self.config.in_channels, height, width];
        let x = Tensor::<4>::from_data(TensorData::new(input.to_vec(), shape), &self.device);
        let (prob, dist) = self.network.forward(x);
        Ok((
            crate::errors::read_f32(prob)?,
            crate::errors::read_f32(dist)?,
        ))
    }

    /// What this StarDist's model file says without its weights.
    pub fn to_header(&self) -> StarDistHeader {
        StarDistHeader {
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

impl Model for StarDist {
    type Header = StarDistHeader;
    type Params = StarDistParams;
    type Input<'a> = StarDistInput<'a>;
    type Raw = StarDistRaw;
    type Plan = StarDistPlan;

    /// Measured 0.37 KB for the layout of StarDist's `2D_versatile_fluo`
    /// (BENCHMARK.md "AI models: RAM per inference").
    const NET_BYTES_PER_PIXEL: f64 = 400.0;
    /// The answer (probability and 32 rays per grid cell) and the
    /// candidates' polygons and bit masks - ~100 bytes per pixel on cells of
    /// 24 px (core's `tests::tile_ram`, a well-fitting model's outputs); more
    /// candidates (a lower threshold, larger objects) take more.
    const POST_BYTES_PER_PIXEL: f64 = 100.0;

    fn header(file: &Path) -> Result<StarDistHeader, NnError> {
        let metadata = crate::weights::read_header(file)?;
        StarDistHeader::from_metadata(&metadata).map_err(in_file(file))
    }

    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        let mut net = Self::from_bytes(&bytes, device).map_err(in_file(file))?;
        net.set_half_convs(gpu);
        Ok(net)
    }

    fn pre<'a>(
        header: &StarDistHeader,
        image: Image<'a>,
        params: &StarDistParams,
    ) -> Result<(StarDistInput<'a>, StarDistPlan), NnError> {
        let pixels = pre::checked_pixels(&header.config, &image)?;
        let normalization = pre::normalization(pixels, params.normalization);
        let input = StarDistInput {
            pixels,
            width: image.width,
            height: image.height,
            channels: image.channels,
            normalization,
            pieces: params.pieces,
        };
        let plan = StarDistPlan {
            width: image.width,
            height: image.height,
        };
        Ok((input, plan))
    }

    fn infer(&self, input: &StarDistInput<'_>) -> Result<StarDistRaw, NnError> {
        let planes = infer::in_pieces(self, input)?;
        Ok(infer::raw(planes))
    }

    fn post(
        raw: StarDistRaw,
        plan: StarDistPlan,
        params: &StarDistParams,
        out: &mut [u32],
    ) -> Result<Written, NnError> {
        let (width, height) = (plan.width, plan.height);
        post::check_size(out, width, height)?;
        let candidates = post::candidates(&raw, width, height, params.probability_threshold);
        drop(raw);
        let kept = post::non_max_suppression(candidates, params.nms_threshold);
        let objects = post::rasterize(&kept, width, out);
        Ok(Written::Instances(objects))
    }
}

#[cfg(test)]
mod tests;
