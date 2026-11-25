//! # stardist
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist 2D as the libtorch engine ran it (`crates/ai/src/torch/stardist.rs`):
//! the object probability and the ray distances per grid cell. The tile is
//! padded at the end (mirrored, as StarDist) to the network's multiple, run
//! whole or in pieces (`crate::pieces`, the receptive field probed once),
//! and the output cropped to the image's grid cells: `side / grid` of them
//! (rounded down, as before - StarDist itself keeps a last partial cell).
//! The star-convex polygons are core's.
//!
//! The model file (`crate::weights`): architecture `stardist2d`, see
//! `config.rs`.

use crate::errors::NnError;
use crate::model::{Image, Model};
use crate::pieces::{self, Pieces, Planes, pad_plane};
use crate::weights::ModelFile;
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;
use std::sync::OnceLock;

mod config;
mod network;

pub use config::{ARCH, ARCH_VERSION, StarDistConfig};

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
    network: network::StarDistNet,
    device: Device,
    /// The receptive-field halo, measured on the first tile that needs
    /// pieces.
    halo: OnceLock<Option<usize>>,
}

impl StarDist {
    pub fn config(&self) -> &StarDistConfig {
        &self.config
    }

    /// A StarDist from a model file's bytes (see `config.rs`).
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, NnError> {
        let file = ModelFile::parse(bytes)?;
        let metadata = file.expect_arch(ARCH, ARCH_VERSION)?;
        let config: StarDistConfig = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("StarDist config: {e}")))?;
        config.check()?;
        let network = network::StarDistNet::load(&file.tensors, &config, device)?;
        Ok(Self {
            config,
            network,
            device: device.clone(),
            halo: OnceLock::new(),
        })
    }

    /// The raw network outputs for `[1, C, H, W]` (sides multiples of
    /// `config.div_by()`): the probability plane and the ray planes at the
    /// grid's cells.
    pub fn forward_raw(&self, input: &[f32], height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
        let shape = [1, self.config.in_channels, height, width];
        let x = Tensor::<4>::from_data(TensorData::new(input.to_vec(), shape), &self.device);
        let (prob, dist) = self.network.forward(x);
        let read = |t: Tensor<4>| {
            t.into_data()
                .as_slice::<f32>()
                .expect("f32 output")
                .to_vec()
        };
        (read(prob), read(dist))
    }

    /// The network on one `width x height` image (`in_channels` planes),
    /// padded at the end and cropped back: probability, then the rays.
    fn predict(&self, pixels: &[f32], width: usize, height: usize) -> Result<Planes, NnError> {
        let [div_y, div_x] = self.config.div_by();
        let (pw, ph) = (
            width.next_multiple_of(div_x),
            height.next_multiple_of(div_y),
        );
        let (right, bottom) = (pw - width, ph - height);
        let input = if (right, bottom) == (0, 0) {
            pixels.to_vec()
        } else {
            let mut padded = Vec::with_capacity(self.config.in_channels * pw * ph);
            for plane in pixels.chunks_exact(width * height) {
                pad_plane(plane, width, height, (0, right, 0, bottom), &mut padded);
            }
            padded
        };
        let (prob, dist) = self.forward_raw(&input, ph, pw);
        let [gy, gx] = self.config.grid;
        let (ow, oh) = (pw / gx, ph / gy);
        // The grid cells of the image itself (not of the padding).
        let (grid_width, grid_height) = ((width / gx).clamp(1, ow), (height / gy).clamp(1, oh));
        let crop = |plane: &[f32], out: &mut Vec<f32>| {
            for y in 0..grid_height {
                out.extend_from_slice(&plane[y * ow..y * ow + grid_width]);
            }
        };
        let mut first = Vec::with_capacity(grid_width * grid_height);
        crop(&prob, &mut first);
        let mut rest = Vec::with_capacity(self.config.n_rays * grid_width * grid_height);
        for ray in dist.chunks_exact(ow * oh) {
            crop(ray, &mut rest);
        }
        Ok(Planes {
            planes: 1 + self.config.n_rays,
            width: grid_width,
            height: grid_height,
            first,
            rest,
        })
    }
}

impl Model for StarDist {
    type Settings = StarDistSettings;
    type Output = StarDistOutput;

    fn load(file: &Path, device: &Device) -> Result<Self, NnError> {
        let bytes = std::fs::read(file)?;
        Self::from_bytes(&bytes, device).map_err(|e| match e {
            NnError::InvalidModelFile(m) => {
                NnError::InvalidModelFile(format!("{}: {m}", file.display()))
            }
            other => other,
        })
    }

    fn infer(
        &self,
        image: Image<'_>,
        settings: &StarDistSettings,
    ) -> Result<StarDistOutput, NnError> {
        let (width, height, channels) = (image.width, image.height, image.channels);
        if width == 0 || height == 0 {
            return Err(NnError::InvalidInput("StarDist: an empty image".into()));
        }
        if channels != self.config.in_channels {
            return Err(NnError::InvalidInput(format!(
                "StarDist: the model takes {} channel(s), the image has {channels}",
                self.config.in_channels
            )));
        }
        let len = width * height * channels;
        if image.data.len() < len {
            return Err(NnError::InvalidInput(format!(
                "StarDist: the image has {} values, {len} expected",
                image.data.len()
            )));
        }
        let pixels = &image.data[..len];
        let predict = |p: &[f32], w: usize, h: usize| self.predict(p, w, h);
        let plan = pieces::plan(
            width,
            height,
            settings.pieces,
            || {
                if let Some(halo) = self.halo.get() {
                    return Ok(*halo);
                }
                let halo = pieces::receptive_halo(channels, predict)?;
                Ok(*self.halo.get_or_init(|| halo))
            },
            "StarDist",
        )?;
        let planes = match plan {
            Some((halo, piece)) => {
                pieces::run(pixels, channels, width, height, halo, piece, predict)?
            }
            None => predict(pixels, width, height)?,
        };
        Ok(StarDistOutput {
            prob: planes.first,
            dist: planes.rest,
            grid_height: planes.height,
            grid_width: planes.width,
            n_rays: planes.planes - 1,
        })
    }
}

#[cfg(test)]
mod tests;
