//! # piece_size
//!
//! The piece size of the AI steps that run a large tile in pieces (U-Net,
//! StarDist) - a setting of their own, shared by them.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use macros::CommandsMeta;

/// How large the pieces are that a large tile runs through a U-Net or
/// StarDist network in (square; a tile up to a piece runs whole). Smaller
/// pieces take less RAM, larger ones are faster - a U-Net on a slide: 1024
/// ~2.6 GB at once, 2048 ~8.5 GB.
#[derive(Debug, Clone, Copy, PartialEq, CommandsMeta)]
pub enum PieceSize {
    /// As large as the computer's RAM allows (2048, 1024 or 512) for a
    /// model seeing a limited neighbourhood - its result is the same
    /// whatever the size, up to float rounding; 1024 on every computer for
    /// a model normalizing by the whole image (InstanceNorm), whose result
    /// the size changes a little.
    Auto,
    /// The same pieces for every model on every computer.
    Fixed {
        /// Pixels per side of a piece.
        #[cmdsmeta(default = 1024, min = 256, max = 4096, step = 256)]
        side: i32,
    },
}

impl PieceSize {
    /// The pieces a `width x height` tile runs in, for a network taking
    /// `bytes_per_pixel` of its input - the same for the request and the RAM
    /// plan. A fixed side within its limits (a project file may hold any
    /// number); Auto: for a model seeing a limited neighbourhood as large as
    /// this computer's RAM allows (its result is the same up to float
    /// rounding), for one seeing the whole image always
    /// [`evanalyzer_ai::WHOLE_IMAGE_PIECE_SIDE`].
    pub(crate) fn pieces(
        self,
        bytes_per_pixel: f64,
        width: usize,
        height: usize,
    ) -> evanalyzer_ai::Pieces {
        match self {
            PieceSize::Fixed { side } => {
                evanalyzer_ai::Pieces::fixed(side.clamp(256, 4096) as usize)
            }
            PieceSize::Auto => evanalyzer_ai::Pieces {
                local_side: evanalyzer_ai::piece_side_for(
                    bytes_per_pixel,
                    crate::memory::SystemRam::now().total,
                    width,
                    height,
                ),
                whole_image_side: evanalyzer_ai::WHOLE_IMAGE_PIECE_SIDE,
            },
        }
    }
}
