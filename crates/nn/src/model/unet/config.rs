//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **The U-Net's part of the model file format** - what is written into a
//! model file's metadata (`crate::weights`): the architecture name and
//! layout version, [`UNetConfig`] as JSON (with [`Norm`] and
//! [`FinalActivation`]) and the [`Preprocessing`] names. Shipped model files
//! depend on every name here.
//!
//! Any change in this file (a field, an enum value, a serde attribute, a
//! name) is a breaking change of the file format: either old files still
//! read as before, or [`ARCH_VERSION`] goes up and the converted models are
//! written again. The test `the_file_format_is_pinned` guards it.
//! The tensor names (torch-em's) are part of the format as well:
//! `network::UNet2d::load_torch_em` / `torch_em_tensors`.

use crate::errors::NnError;
use serde::{Deserialize, Serialize};

/// The architecture name in the model file.
pub const ARCH: &str = "unet2d";
/// The newest layout version of [`ARCH`] this code reads.
pub const ARCH_VERSION: u32 = 1;

/// Architecture, from the model file's metadata - torch-em `UNet2d`'s
/// argument names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub depth: usize,
    pub initial_features: usize,
    pub gain: usize,
    pub norm: Norm,
    pub final_activation: FinalActivation,
}

impl UNetConfig {
    /// Refuses sizes no real U-Net has - a model file is untrusted, and the
    /// network is allocated from these before any tensor is read.
    pub fn check(&self) -> Result<(), NnError> {
        let bad = |what: &str| Err(NnError::InvalidModelFile(format!("U-Net config: {what}")));
        if !(1..=16).contains(&self.in_channels) {
            return bad("in_channels must be 1-16");
        }
        if !(1..=64).contains(&self.out_channels) {
            return bad("out_channels must be 1-64");
        }
        if !(1..=6).contains(&self.depth) {
            return bad("depth must be 1-6");
        }
        if !(1..=256).contains(&self.initial_features) {
            return bad("initial_features must be 1-256");
        }
        if !(1..=4).contains(&self.gain) {
            return bad("gain must be 1-4");
        }
        let deepest = self.initial_features * self.gain.pow(self.depth as u32);
        if deepest > 4096 {
            return bad("more than 4096 features at the deepest level");
        }
        if let Norm::Group { groups } = self.norm
            && !(1..=1024).contains(&groups)
        {
            return bad("groups must be 1-1024");
        }
        Ok(())
    }

    /// Input sides must be multiples of this (the network halves them
    /// `depth` times).
    pub fn size_multiple(&self) -> usize {
        1 << self.depth
    }

    /// Whether the network normalizes by the whole image (InstanceNorm,
    /// GroupNorm): its pieces can't be exact (see `crate::pieces`).
    pub fn sees_whole_image(&self) -> bool {
        matches!(self.norm, Norm::Instance | Norm::Group { .. })
    }
}

/// The activation after the last convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalActivation {
    None,
    Sigmoid,
    /// Over the channels.
    Softmax,
}

/// The normalization between a U-Net's convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Norm {
    None,
    Batch,
    /// Normalizes by the whole image: its pieces can't be exact.
    Instance,
    /// Normalizes by the whole image as well.
    Group {
        groups: usize,
    },
}

/// The preprocessing a model file asks for, applied to each tile before
/// the network (before its pieces: one tile, one set of statistics) - see
/// `UNet::infer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preprocessing {
    None,
    /// `(x - mean) / (std + 1e-6)` over all pixels and channels
    /// (bioimage.io's `zero_mean_unit_variance`; idempotent, so an already
    /// normalized image stays as it is).
    ZeroMeanUnitVariance,
}

impl Preprocessing {
    /// The metadata value's name.
    pub const ZERO_MEAN_UNIT_VARIANCE: &str = "zero_mean_unit_variance";

    pub(crate) fn from_metadata(value: Option<&str>) -> Result<Self, NnError> {
        match value {
            None | Some("") | Some("none") => Ok(Self::None),
            Some(Self::ZERO_MEAN_UNIT_VARIANCE) => Ok(Self::ZeroMeanUnitVariance),
            Some(other) => Err(NnError::UnsupportedArchitecture(format!(
                "unknown preprocessing `{other}`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON shipped model files carry: changing it breaks them (see
    /// the module docs) - change this test only with `ARCH_VERSION`.
    #[test]
    fn the_file_format_is_pinned() {
        assert_eq!(ARCH, "unet2d");
        assert_eq!(ARCH_VERSION, 1);
        let config = UNetConfig {
            in_channels: 1,
            out_channels: 2,
            depth: 4,
            initial_features: 64,
            gain: 2,
            norm: Norm::Instance,
            final_activation: FinalActivation::Sigmoid,
        };
        let json = r#"{"in_channels":1,"out_channels":2,"depth":4,"initial_features":64,"gain":2,"norm":"instance","final_activation":"sigmoid"}"#;
        assert_eq!(serde_json::to_string(&config).unwrap(), json);
        assert_eq!(serde_json::from_str::<UNetConfig>(json).unwrap(), config);
        let names = |v: &[&str]| {
            v.iter()
                .map(|n| serde_json::from_str(&format!("\"{n}\"")).unwrap())
                .collect::<Vec<Norm>>()
        };
        assert_eq!(
            names(&["none", "batch", "instance"]),
            [Norm::None, Norm::Batch, Norm::Instance]
        );
        assert_eq!(
            serde_json::to_string(&Norm::Group { groups: 8 }).unwrap(),
            r#"{"group":{"groups":8}}"#
        );
        for (activation, name) in [
            (FinalActivation::None, "none"),
            (FinalActivation::Sigmoid, "sigmoid"),
            (FinalActivation::Softmax, "softmax"),
        ] {
            assert_eq!(
                serde_json::to_string(&activation).unwrap(),
                format!("\"{name}\"")
            );
        }
        assert_eq!(
            Preprocessing::ZERO_MEAN_UNIT_VARIANCE,
            "zero_mean_unit_variance"
        );
        assert_eq!(
            Preprocessing::from_metadata(Some("zero_mean_unit_variance")).unwrap(),
            Preprocessing::ZeroMeanUnitVariance
        );
        assert_eq!(
            Preprocessing::from_metadata(None).unwrap(),
            Preprocessing::None
        );
    }
}
