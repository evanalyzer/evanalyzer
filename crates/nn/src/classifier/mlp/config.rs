//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **The MLP's part of the `.evamodel` format**: [`Mlp`] (JSON) =
//! [`MlpArchitecture`] + `weights`, the bytes of a safetensors file
//! (`crate::weights`, D4 of the burn migration plan: never burn's own
//! format) with metadata architecture [`ARCH`] version [`ARCH_VERSION`]
//! (`config` = the architecture as JSON) and per layer `i` the f32 tensors
//! [`weight_name`] `[out, in]` (PyTorch's layout) and [`bias_name`] `[out]`.
//!
//! Any change here (a field, a variant, a name, a serde attribute) changes
//! what saved `.evamodel` files contain: core's
//! `CURRENT_SAVED_CLASSIFIER_VERSION` goes up with it (older files are
//! refused - no backwards compatibility). The test
//! `classifier::tests::the_file_layout_is_pinned` guards it.

use super::Network;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The architecture name in the weights' metadata.
pub(super) const ARCH: &str = "mlp";
pub(super) const ARCH_VERSION: u32 = 1;

/// The weight tensor of layer `i`.
pub(super) fn weight_name(i: usize) -> String {
    format!("layer{i}.weight")
}

/// The bias tensor of layer `i`.
pub(super) fn bias_name(i: usize) -> String {
    format!("layer{i}.bias")
}

/// The activation between hidden layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MlpActivation {
    #[default]
    Relu,
    Sigmoid,
    Tanh,
}

/// The shape of a trained MLP - what the weights need to be loaded into.
///
/// Part of every saved MLP classifier (`.evamodel`), see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlpArchitecture {
    pub n_in: usize,
    pub hidden: Vec<usize>,
    pub n_out: usize,
    /// The activation it was trained with - used at prediction too.
    pub activation: MlpActivation,
}

/// A trained MLP: its architecture and weights as saved, the network built
/// from them on the first prediction.
#[derive(Debug, Serialize, Deserialize)]
pub struct Mlp {
    pub architecture: MlpArchitecture,
    /// Safetensors bytes, see the module docs.
    pub weights: Vec<u8>,
    #[serde(skip)]
    pub(super) network: OnceLock<Network>,
}
