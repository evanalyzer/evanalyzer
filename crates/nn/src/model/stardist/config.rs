//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **StarDist 2D's part of the model file format** - what is written into a
//! model file's metadata (`crate::weights`): the architecture name and
//! layout version and [`StarDistConfig`] as JSON (StarDist's `config.json`
//! values). The tensors keep the Keras layer names (`down_level_0_no_0`,
//! `middle_2`, `features`, `prob`, `dist`; the grid's unnamed convolutions
//! `grid.{i}`), `[out, in, kh, kw]` - `network::StarDistNet::load`. Written by
//! `docs/convert_stardist.py`.
//!
//! Any change in this file (a field, a name, a serde attribute) is a
//! breaking change of the file format: either old files still read as
//! before, or [`ARCH_VERSION`] goes up and the converted models are written
//! again. The test `the_file_format_is_pinned` guards it.

use crate::errors::NnError;
use serde::{Deserialize, Serialize};

/// The architecture name in the model file.
pub const ARCH: &str = "stardist2d";
/// The newest layout version of [`ARCH`] this code reads.
pub const ARCH_VERSION: u32 = 1;

/// StarDist 2D (one class, ReLU, no batch norm, pooling 2x2): the grid's
/// convolutions and poolings, CSBDeep's U-Net, an extra convolution, and
/// the probability (sigmoid) and ray distance heads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StarDistConfig {
    pub in_channels: usize,
    pub n_rays: usize,
    /// Output cell size in pixels, `[y, x]` (powers of 2).
    pub grid: [usize; 2],
    pub unet_depth: usize,
    pub unet_filter_base: usize,
    pub unet_conv_per_depth: usize,
    /// Square, odd.
    pub kernel_size: usize,
    /// Filters of the convolution after the U-Net, 0 = none.
    pub conv_after_unet: usize,
}

impl StarDistConfig {
    /// Refuses sizes no real StarDist has - a model file is untrusted.
    pub fn check(&self) -> Result<(), NnError> {
        let bad = |what: &str| {
            Err(NnError::InvalidModelFile(format!(
                "StarDist config: {what}"
            )))
        };
        if !(1..=16).contains(&self.in_channels) || !(3..=256).contains(&self.n_rays) {
            return bad("in_channels must be 1-16, n_rays 3-256");
        }
        if self.grid.iter().any(|&g| !g.is_power_of_two() || g > 16) {
            return bad("grid sides must be powers of 2 up to 16");
        }
        if !(1..=6).contains(&self.unet_depth)
            || !(1..=256).contains(&self.unet_filter_base)
            || !(1..=4).contains(&self.unet_conv_per_depth)
        {
            return bad("unet_depth 1-6, unet_filter_base 1-256, unet_conv_per_depth 1-4");
        }
        if self.kernel_size % 2 == 0 || self.kernel_size > 7 || self.conv_after_unet > 1024 {
            return bad("kernel_size odd up to 7, conv_after_unet up to 1024");
        }
        Ok(())
    }

    /// Input sides must be multiples of this, per axis `[y, x]` (StarDist's
    /// `axes_div_by`: the grid times the U-Net's poolings).
    pub fn div_by(&self) -> [usize; 2] {
        self.grid.map(|g| g << self.unet_depth)
    }

    /// The official `2D_versatile_fluo`.
    pub fn versatile_fluo() -> Self {
        Self {
            in_channels: 1,
            n_rays: 32,
            grid: [2, 2],
            unet_depth: 3,
            unet_filter_base: 32,
            unet_conv_per_depth: 2,
            kernel_size: 3,
            conv_after_unet: 128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON shipped model files carry (`docs/convert_stardist.py`):
    /// changing it breaks them - change this test only with `ARCH_VERSION`.
    #[test]
    fn the_file_format_is_pinned() {
        assert_eq!((ARCH, ARCH_VERSION), ("stardist2d", 1));
        let json = r#"{"in_channels":1,"n_rays":32,"grid":[2,2],"unet_depth":3,"unet_filter_base":32,"unet_conv_per_depth":2,"kernel_size":3,"conv_after_unet":128}"#;
        assert_eq!(
            serde_json::to_string(&StarDistConfig::versatile_fluo()).unwrap(),
            json
        );
        assert_eq!(
            serde_json::from_str::<StarDistConfig>(json).unwrap(),
            StarDistConfig::versatile_fluo()
        );
        assert!(StarDistConfig::versatile_fluo().check().is_ok());
        assert_eq!(StarDistConfig::versatile_fluo().div_by(), [16, 16]);
        for bad in [
            StarDistConfig {
                grid: [3, 2],
                ..StarDistConfig::versatile_fluo()
            },
            StarDistConfig {
                kernel_size: 4,
                ..StarDistConfig::versatile_fluo()
            },
            StarDistConfig {
                unet_depth: 9,
                ..StarDistConfig::versatile_fluo()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
    }
}
