//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **Cellpose-SAM's part of the model file format** - what is written into a
//! model file's metadata (`crate::weights`): the architecture name and
//! layout version and [`CellposeSamConfig`] as JSON. The tensors keep
//! Cellpose's PyTorch names (`encoder.blocks.{i}.attn.qkv.weight`, ...,
//! `network::CellposeSamNet::load`), the relative-position tables SAM's
//! sizes (resized on loading) - the file loads into Cellpose's own `CPSAM`
//! as it is. Written by `docs/convert_cellpose_sam.py`.
//!
//! Any change in this file (a field, a name, a serde attribute) is a
//! breaking change of the file format: either old files still read as
//! before, or [`ARCH_VERSION`] goes up and the converted models are written
//! again. The test `the_file_format_is_pinned` guards it.

use crate::errors::NnError;
use serde::{Deserialize, Serialize};

/// The architecture name in the model file.
pub const ARCH: &str = "cellpose_sam";
/// The newest layout version of [`ARCH`] this code reads.
pub const ARCH_VERSION: u32 = 1;

/// The network: SAM's ViT-L image encoder (every block with global
/// attention, as Cellpose-SAM sets it) on `tile_size`² tiles cut into
/// `patch_size`² patches, its neck, and a 1x1 convolution to `out_channels`
/// per pixel (dY, dX, cell logit) through a pixel shuffle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellposeSamConfig {
    /// Channels the patch embedding has weights for (an image may give fewer).
    pub in_channels: usize,
    pub tile_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    /// Transformer blocks.
    pub depth: usize,
    pub num_heads: usize,
    /// Hidden size of each block's MLP.
    pub mlp_dim: usize,
    /// Channels of the encoder's neck.
    pub neck_dim: usize,
    pub out_channels: usize,
}

impl CellposeSamConfig {
    /// Refuses sizes no real Cellpose-SAM has - a model file is untrusted,
    /// and the network is allocated from these before any tensor is read.
    pub fn check(&self) -> Result<(), NnError> {
        let bad = |what: &str| {
            Err(NnError::InvalidModelFile(format!(
                "Cellpose-SAM config: {what}"
            )))
        };
        if !(1..=3).contains(&self.in_channels) {
            return bad("in_channels must be 1-3");
        }
        if !(16..=1024).contains(&self.tile_size) || !(1..=32).contains(&self.patch_size) {
            return bad("tile_size must be 16-1024, patch_size 1-32");
        }
        if !self.tile_size.is_multiple_of(self.patch_size) {
            return bad("tile_size must be a multiple of patch_size");
        }
        if !(1..=4096).contains(&self.embed_dim)
            || !(1..=64).contains(&self.num_heads)
            || !self.embed_dim.is_multiple_of(self.num_heads)
        {
            return bad("embed_dim must be 1-4096 and a multiple of num_heads (1-64)");
        }
        if !(1..=64).contains(&self.depth)
            || !(1..=16384).contains(&self.mlp_dim)
            || !(1..=4096).contains(&self.neck_dim)
            || !(1..=16).contains(&self.out_channels)
        {
            return bad("depth 1-64, mlp_dim 1-16384, neck_dim 1-4096, out_channels 1-16");
        }
        Ok(())
    }

    /// Tokens per side of a tile.
    pub fn tokens(&self) -> usize {
        self.tile_size / self.patch_size
    }

    /// The published Cellpose-SAM (`cpsam`, `cpsam_v2`): SAM ViT-L, 256²
    /// tiles, 8² patches.
    pub fn vit_l() -> Self {
        Self {
            in_channels: 3,
            tile_size: 256,
            patch_size: 8,
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            mlp_dim: 4096,
            neck_dim: 256,
            out_channels: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON shipped model files carry (`docs/convert_cellpose_sam.py`
    /// writes it): changing it breaks them - change this test only with
    /// `ARCH_VERSION`.
    #[test]
    fn the_file_format_is_pinned() {
        assert_eq!((ARCH, ARCH_VERSION), ("cellpose_sam", 1));
        let json = r#"{"in_channels":3,"tile_size":256,"patch_size":8,"embed_dim":1024,"depth":24,"num_heads":16,"mlp_dim":4096,"neck_dim":256,"out_channels":3}"#;
        assert_eq!(
            serde_json::to_string(&CellposeSamConfig::vit_l()).unwrap(),
            json
        );
        assert_eq!(
            serde_json::from_str::<CellposeSamConfig>(json).unwrap(),
            CellposeSamConfig::vit_l()
        );
        assert!(CellposeSamConfig::vit_l().check().is_ok());
        assert_eq!(CellposeSamConfig::vit_l().tokens(), 32);
        for bad in [
            CellposeSamConfig {
                in_channels: 4,
                ..CellposeSamConfig::vit_l()
            },
            CellposeSamConfig {
                patch_size: 7,
                ..CellposeSamConfig::vit_l()
            },
            CellposeSamConfig {
                num_heads: 15,
                ..CellposeSamConfig::vit_l()
            },
            CellposeSamConfig {
                depth: 1000,
                ..CellposeSamConfig::vit_l()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
    }
}
