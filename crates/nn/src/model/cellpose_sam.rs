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
//! libtorch engine ran it (removed `crates/ai/src/torch/cellpose.rs`): the image's
//! first channel resized (bilinear) to the size the model runs at, its
//! channels filled up (zeros, or the image repeated), run in overlapping
//! 256 x 256 tiles blended with Cellpose's taper (`transforms.average_tiles`)
//! into the flows and the cell probability (after a sigmoid). The network
//! takes exactly one tile size: its positional embedding is fixed to it.
//!
//! The model file (`crate::weights`): architecture `cellpose_sam`, see
//! `config.rs`.

use crate::errors::NnError;
use crate::model::{Image, Model};
use crate::weights::ModelFile;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;

mod config;
mod network;

pub use config::{ARCH, ARCH_VERSION, CellposeSamConfig};

/// The only input side the published network takes.
pub const TILE_SIDE: usize = 256;

/// Fraction of a tile that overlaps its neighbour - Cellpose's own default
/// (`tile_overlap=0.1` in `CellposeModel.eval`).
const TILE_OVERLAP: f32 = 0.1;

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellposeSamSettings {
    /// The size the model runs at (the image is resized to it).
    pub net_width: usize,
    pub net_height: usize,
    /// Channels handed to the network (1 up to the model's `in_channels`).
    pub input_channels: usize,
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
        let metadata = file.expect_arch(ARCH, ARCH_VERSION)?;
        let config: CellposeSamConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("Cellpose-SAM config: {e}")))?;
        config.check()?;
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
        let s = self.config.tile_size;
        let x = Tensor::<4>::from_data(
            TensorData::new(tile.to_vec(), [1, channels, s, s]),
            &self.device,
        );
        let y = self.network.forward(x).into_data();
        y.as_slice::<f32>().expect("f32 output").to_vec()
    }
}

/// Start offsets of the `tile`-wide tiles covering `len` pixels with
/// [`TILE_OVERLAP`] - Cellpose's `transforms.make_tiles` (in f32, as the
/// libtorch engine computed them). One tile at 0 when `len <= tile`.
fn tile_starts(len: usize, tile: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let overlap = TILE_OVERLAP.clamp(0.05, 0.5);
    let n = (((1.0 + 2.0 * overlap) * len as f32) / tile as f32).ceil() as usize;
    if n <= 1 {
        return vec![0];
    }
    let span = (len - tile) as f32;
    (0..n)
        .map(|i| (span * i as f32 / (n - 1) as f32) as usize)
        .collect()
}

/// Cellpose's blend weight per pixel of a `tile`² tile
/// (`transforms._taper_mask`): ~1 in the middle, a sigmoid falling toward
/// (never to) 0 at the edges.
fn taper_mask(tile: usize) -> Vec<f32> {
    const SIG: f32 = 7.5;
    let center = (tile as f32 - 1.0) / 2.0;
    let mask1d: Vec<f32> = (0..tile)
        .map(|i| {
            let xm = (i as f32 - center).abs();
            1.0 / (1.0 + ((xm - (tile as f32 / 2.0 - 20.0)) / SIG).exp())
        })
        .collect();
    let mut mask = Vec::with_capacity(tile * tile);
    for y in 0..tile {
        for x in 0..tile {
            mask.push(mask1d[y] * mask1d[x]);
        }
    }
    mask
}

impl Model for CellposeSam {
    type Settings = CellposeSamSettings;
    type Output = CellposeSamOutput;

    /// On the GPU the matrix products and convolutions in f16 (its tensor
    /// cores; 4.9x faster than f32 at 1024², as close to Cellpose's f32
    /// network as the bf16 model of the libtorch engine - BENCHMARK.md),
    /// on the CPU f32.
    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        let mut net = Self::from_bytes(&bytes, device).map_err(|e| match e {
            NnError::InvalidModelFile(m) => {
                NnError::InvalidModelFile(format!("{}: {m}", file.display()))
            }
            other => other,
        })?;
        net.set_half(gpu);
        Ok(net)
    }

    /// Uses the image's first channel, as the libtorch engine did.
    fn infer(
        &self,
        image: Image<'_>,
        settings: &CellposeSamSettings,
    ) -> Result<CellposeSamOutput, NnError> {
        let (width, height) = (image.width, image.height);
        let (w, h) = (settings.net_width, settings.net_height);
        if width == 0 || height == 0 || w == 0 || h == 0 || image.channels == 0 {
            return Err(NnError::InvalidInput("Cellpose: an empty image".into()));
        }
        if image.data.len() < width * height {
            return Err(NnError::InvalidInput(format!(
                "Cellpose: the image has {} values, {} expected",
                image.data.len(),
                width * height
            )));
        }
        let channels = settings.input_channels.clamp(1, self.config.in_channels);
        // The first channel at the network's size (bilinear,
        // align_corners = false - PyTorch's `upsample_bilinear2d`).
        let gray = &image.data[..width * height];
        let gray: Vec<f32> = if (w, h) == (width, height) {
            gray.to_vec()
        } else {
            let x = Tensor::<4>::from_data(
                TensorData::new(gray.to_vec(), [1, 1, height, width]),
                &self.device,
            );
            let y = interpolate(
                x,
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(false)
                    .with_output_size([h, w]),
            );
            y.into_data()
                .try_to_vec::<f32>()
                .map_err(|e| NnError::Internal(format!("{e:?}")))?
        };

        // Padded at the bottom and right to at least one tile (zeros), in
        // tiles; each tile's output weighted by the taper, summed, divided by
        // the summed weights.
        let s = self.config.tile_size;
        let (ph, pw) = (h.max(s), w.max(s));
        let (ys, xs) = (tile_starts(ph, s), tile_starts(pw, s));
        let mask = taper_mask(s);
        let n_out = self.config.out_channels;
        let mut sum = vec![0.0f32; n_out * ph * pw];
        let mut weight = vec![0.0f32; ph * pw];
        let mut tile = vec![0.0f32; channels * s * s];
        for &y0 in &ys {
            for &x0 in &xs {
                tile.fill(0.0);
                for ty in 0..s {
                    let y = y0 + ty;
                    if y >= h {
                        break;
                    }
                    for tx in 0..s.min(w.saturating_sub(x0)) {
                        let v = gray[y * w + x0 + tx];
                        tile[ty * s + tx] = v;
                        if settings.replicate_gray_channel {
                            for c in 1..channels {
                                tile[(c * s + ty) * s + tx] = v;
                            }
                        }
                    }
                }
                let out = self.forward_tile(&tile, channels);
                for o in 0..n_out {
                    for ty in 0..s {
                        let row = ((o * ph) + y0 + ty) * pw + x0;
                        let src = (o * s + ty) * s;
                        for tx in 0..s {
                            sum[row + tx] += out[src + tx] * mask[ty * s + tx];
                        }
                    }
                }
                for ty in 0..s {
                    let row = (y0 + ty) * pw + x0;
                    for tx in 0..s {
                        weight[row + tx] += mask[ty * s + tx];
                    }
                }
            }
        }
        let plane = |o: usize, f: fn(f32) -> f32| -> Vec<f32> {
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    out.push(f(sum[(o * ph + y) * pw + x] / weight[y * pw + x]));
                }
            }
            out
        };
        Ok(CellposeSamOutput {
            flow_y: plane(0, |v| v),
            flow_x: plane(1, |v| v),
            cell_prob: plane(2, |v| 1.0 / (1.0 + (-v).exp())),
        })
    }
}

#[cfg(test)]
mod tests;
