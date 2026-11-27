//! # spotnet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! SpotNet: a small fully convolutional spot detector, trained in the app
//! (`train`), run in three steps ([`Model`]):
//!
//! - `pre`: the image checked, each channel's percentiles of the tile
//!   computed for the normalization (the model file's unless the step chose
//!   others) ([`normalize`]);
//! - `infer`: run whole or in pieces (`crate::pieces`, the halo half the
//!   receptive field - known exactly, not probed), normalized on the way:
//!   the spot probability per pixel ([`infer`]);
//! - `post`: the probability map's peaks above the threshold, each a
//!   one-pixel object - instance labels `1..=n` ([`post`]).
//!
//! The model file (`crate::weights`): architecture `spotnet`, see
//! [`file`].

use crate::errors::NnError;
use crate::model::{Image, Model, Normalization, Written};
use crate::pieces::Pieces;
use burn::tensor::Device;
use std::path::Path;

mod file;
mod infer;
mod network;
mod normalize;
mod post;
mod target;
mod train;
mod validate;

pub use file::{ARCH, ARCH_VERSION, SpotNetHeader, SpotNetModel, SpotNetSettings};
pub use network::SpotNetConfig;
pub use normalize::PercentileNormalization;
pub use train::{Spot, SpotLabels, SpotNetTrainParams, SpotNetTraining, ValidationOutcome, train};
pub use validate::ValidationScore;

/// What a step chooses per run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotNetParams {
    pub normalization: Normalization,
    /// How large a tile runs in pieces.
    pub pieces: Pieces,
    /// A peak of the probability map counts as a spot from this on (0-1).
    pub threshold: f32,
}

/// The tile as the network gets it - made by `pre` only.
pub struct SpotNetInput<'a> {
    pixels: &'a [f32],
    width: usize,
    height: usize,
    channels: usize,
    /// Each channel's scaling, applied while each piece is copied for the
    /// network; `None`: the pixels as they are.
    normalization: Option<Vec<Option<normalize::Affine>>>,
    pieces: Pieces,
}

/// What the network gives back: the spot probability per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotNetRaw {
    pub probability: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// The tile's size, for `post`.
#[derive(Debug, Clone, Copy)]
pub struct SpotNetPlan {
    width: usize,
    height: usize,
}

/// The pixels of `image` the network gets (`in_channels` planes) - or why
/// it can't take them.
fn checked_pixels<'a>(header: &SpotNetHeader, image: &Image<'a>) -> Result<&'a [f32], NnError> {
    let (width, height, channels) = (image.width, image.height, image.channels);
    if width == 0 || height == 0 {
        return Err(NnError::InvalidInput("SpotNet: an empty image".into()));
    }
    let wanted = header.settings.network.in_channels;
    if channels != wanted {
        return Err(NnError::InvalidInput(format!(
            "SpotNet: the model takes {wanted} channel(s), the image has {channels}"
        )));
    }
    let len = width * height * channels;
    if image.data.len() < len {
        return Err(NnError::InvalidInput(format!(
            "SpotNet: the image has {} values, {len} expected",
            image.data.len()
        )));
    }
    Ok(&image.data[..len])
}

impl Model for SpotNetModel {
    const ARCH: &'static str = ARCH;
    type Header = SpotNetHeader;
    type Params = SpotNetParams;
    type Input<'a> = SpotNetInput<'a>;
    type Raw = SpotNetRaw;
    type Plan = SpotNetPlan;

    /// Estimated, not yet measured: 16 feature maps of 4 bytes in the
    /// current and the next layer, plus the input and the output - ~150
    /// bytes, doubled for burn's buffers. To measure with core's
    /// `tests::tile_ram` once SpotNet is an AI step (BENCHMARK.md "AI
    /// models: RAM per inference").
    const NET_BYTES_PER_PIXEL: f64 = 300.0;
    /// The probability map (4 bytes per pixel) and the spot list.
    const POST_BYTES_PER_PIXEL: f64 = 8.0;

    fn header(file: &Path) -> Result<SpotNetHeader, NnError> {
        let metadata = crate::weights::read_header(file)?;
        SpotNetHeader::from_metadata(&metadata).map_err(file::in_file(file))
    }

    /// On a GPU too in f32: the network is small, its convolutions few.
    fn load(file: &Path, device: &Device, _gpu: bool) -> Result<Self, NnError> {
        Self::read(file, device)
    }

    fn pre<'a>(
        header: &SpotNetHeader,
        image: Image<'a>,
        params: &SpotNetParams,
    ) -> Result<(SpotNetInput<'a>, SpotNetPlan), NnError> {
        let pixels = checked_pixels(header, &image)?;
        let percentiles = match params.normalization {
            Normalization::ModelDefault => Some(header.settings.normalization),
            Normalization::Percentile { low, high } => {
                let asked = PercentileNormalization { low, high };
                asked.check()?;
                Some(asked)
            }
            Normalization::None => None,
        };
        let plane = image.width * image.height;
        let input = SpotNetInput {
            pixels,
            width: image.width,
            height: image.height,
            channels: image.channels,
            normalization: percentiles.map(|p| p.channels(pixels, plane)),
            pieces: params.pieces,
        };
        let plan = SpotNetPlan {
            width: image.width,
            height: image.height,
        };
        Ok((input, plan))
    }

    fn infer(&self, input: &SpotNetInput<'_>) -> Result<SpotNetRaw, NnError> {
        let probability = infer::probabilities(
            self,
            input.pixels,
            input.width,
            input.height,
            input.channels,
            input.normalization.as_deref(),
            input.pieces,
        )?;
        Ok(SpotNetRaw {
            probability,
            width: input.width,
            height: input.height,
        })
    }

    fn post(
        raw: SpotNetRaw,
        plan: SpotNetPlan,
        params: &SpotNetParams,
        out: &mut [u32],
    ) -> Result<Written, NnError> {
        let (width, height) = (plan.width, plan.height);
        post::check_size(out, width, height)?;
        if !(0.0..=1.0).contains(&params.threshold) {
            return Err(NnError::InvalidInput(format!(
                "SpotNet: the threshold must be 0-1, not {}",
                params.threshold
            )));
        }
        let spots = post::find_spots(&raw.probability, width, height, None, params.threshold);
        drop(raw);
        Ok(Written::Instances(post::write_spots(&spots, width, out)))
    }
}

#[cfg(test)]
mod tests;
