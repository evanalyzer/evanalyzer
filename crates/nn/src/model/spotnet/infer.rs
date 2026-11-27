//! # infer
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How SpotNet runs over a tile - inside the engine slot: in pieces where
//! the tile needs them (`crate::pieces`), each piece normalized on the way
//! and through the network; the spot probability per pixel.
//!
//! No padding: the network takes any size (no pooling, stride 1). The halo
//! is known exactly - half the receptive field (`1 + 2 * sum(dilations)`)
//! - so it is not probed like StarDist's: a piece's middle sees what one
//! pass over the tile would show it, the result equal up to float rounding.

use super::file::SpotNetModel;
use super::normalize::{Affine, apply_per_channel};
use crate::errors::NnError;
use crate::pieces::{self, Pieces, Planes};
use burn::tensor::{Tensor, TensorData};

/// The spot probability of every pixel of the `width x height` `pixels`
/// (`channels` planes, one after the other) - whole, or in `pieces` with
/// the receptive field's halo. `normalization`: each channel's scaling,
/// applied to each piece's copy (`None`: the pixels as they are).
pub(crate) fn probabilities(
    model: &SpotNetModel,
    pixels: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    normalization: Option<&[Option<Affine>]>,
    pieces: Pieces,
) -> Result<Vec<f32>, NnError> {
    let halo = model.settings.network.receptive_field() / 2;
    let local_halo = || Ok(Some(halo));
    let predict = |p: &[f32], w: usize, h: usize| predict(model, p, w, h, channels, normalization);
    let plan = pieces::plan(width, height, pieces, local_halo, "SpotNet")?;
    let planes = pieces::run_fitting(
        pixels, channels, width, height, plan, local_halo, predict, "SpotNet",
    )?;
    Ok(planes.first)
}

/// The network on one `width x height` piece: normalized, forward, the
/// probability plane.
fn predict(
    model: &SpotNetModel,
    pixels: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    normalization: Option<&[Option<Affine>]>,
) -> Result<Planes, NnError> {
    let mut input = pixels.to_vec();
    if let Some(affines) = normalization {
        apply_per_channel(affines, &mut input, width * height);
    }
    let x = Tensor::<4>::from_data(
        TensorData::new(input, [1, channels, height, width]),
        &model.device,
    );
    let probability = crate::errors::read_f32(model.network.forward(x))?;
    Ok(Planes {
        planes: 1,
        width,
        height,
        first: probability,
        rest: Vec::new(),
    })
}
