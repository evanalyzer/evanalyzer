//! # input_normalization
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use macros::CommandsMeta;

/// How the image is scaled before an AI model sees it - computed over each
/// tile (an image up to 4096 x 4096 px is one tile; on a whole slide every
/// tile is scaled by its own pixels, so a tile of mostly background has its
/// noise stretched).
#[derive(Debug, Clone, Copy, PartialEq, CommandsMeta)]
pub enum InputNormalization {
    /// What the model was trained with: StarDist the tile's 1st and 99.8th
    /// percentile, Cellpose its 1st and 99th, a U-Net what its model file
    /// says (e.g. zero mean, unit variance).
    ModelDefault,
    /// These percentiles of the tile become 0 and 1.
    Percentile {
        /// The percentile that becomes 0.
        #[cmdsmeta(default = 1.0, min = 0.0, max = 100.0, step = 0.1)]
        low: f32,
        /// The percentile that becomes 1.
        #[cmdsmeta(default = 99.8, min = 0.0, max = 100.0, step = 0.1)]
        high: f32,
    },
    /// The pixels as read (scaled by their bit depth to 0..1).
    Off,
}

impl InputNormalization {
    /// As nn takes it.
    pub(crate) fn to_nn(self) -> evanalyzer_nn::Normalization {
        match self {
            Self::ModelDefault => evanalyzer_nn::Normalization::ModelDefault,
            Self::Percentile { low, high } => evanalyzer_nn::Normalization::Percentile {
                low: f64::from(low),
                high: f64::from(high),
            },
            Self::Off => evanalyzer_nn::Normalization::None,
        }
    }
}
