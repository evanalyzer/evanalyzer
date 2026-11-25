//! # pieces
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! A fully convolutional network (U-Net, StarDist) over a large tile in
//! pieces, so its RAM is that of one piece instead of the whole tile. Each
//! piece is run with a halo at least as wide as the model's receptive field,
//! so the result equals the whole tile's up to float rounding - except for a
//! model that normalizes by the whole image (InstanceNorm).
//!
//! P3: the receptive-field probe, InstanceNorm detection and
//! `run_in_pieces` from `crates/ai/src/torch/pieces.rs`. The sizes below come
//! from `crates/ai/src/api.rs` as they are.

/// The piece sides "Auto" chooses from for a model that sees a limited
/// neighbourhood - largest first: there a piece's result equals the whole
/// tile's up to float rounding, so the computer's RAM may choose
/// ([`piece_side_for`]). Larger pieces are faster (more cores used well, the
/// halo computed less often), smaller ones take less RAM.
pub const PIECE_SIDES: [usize; 3] = [2048, 1024, 512];

/// The piece side "Auto" gives a model that sees the whole image
/// (InstanceNorm): there the piece size changes the result (0.06-0.18 % of
/// the pixels), so it is the same on every computer.
pub const WHOLE_IMAGE_PIECE_SIDE: usize = 1024;

/// The halo of the pieces as planned: a model that sees the whole image gets
/// this one; the measured receptive fields were 80-112 px (U-Net, StarDist
/// layouts) - a wider one is rare (at most 240 px, the probe's limit).
pub const PLANNED_HALO: usize = 128;

/// The largest image side a U-Net or StarDist is run at on a tile side of
/// `side` with pieces of `piece`: the side itself up to `piece`, else a piece
/// with its halo on both sides (as planned, [`PLANNED_HALO`]). For RAM plans:
/// a network's RAM grows with the pixels it gets at once.
pub fn largest_model_input(side: usize, piece: usize) -> usize {
    if side <= piece {
        side
    } else {
        side.min(piece + 2 * PLANNED_HALO)
    }
}

/// The piece side for a `width x height` tile of a network taking
/// `bytes_per_pixel` of its input, for a model that sees a limited
/// neighbourhood: the largest of [`PIECE_SIDES`] whose largest input
/// ([`largest_model_input`]) takes at most a third of `ram_bytes` - the
/// computer's RAM (or the user's limit). Allowed to follow the computer: such
/// a model's pieces give the whole tile's result up to float rounding. The
/// smallest when none fits.
pub fn piece_side_for(bytes_per_pixel: f64, ram_bytes: u64, width: usize, height: usize) -> usize {
    let budget = ram_bytes as f64 / 3.0;
    PIECE_SIDES
        .into_iter()
        .find(|&piece| {
            let pixels = largest_model_input(width, piece) * largest_model_input(height, piece);
            bytes_per_pixel * pixels as f64 <= budget
        })
        .unwrap_or(PIECE_SIDES[PIECE_SIDES.len() - 1])
}

/// The pieces of a U-Net or StarDist run: one side for a model that sees a
/// limited neighbourhood, one for a model that sees the whole image - which
/// it is is known only after probing the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pieces {
    /// A limited neighbourhood: the result is the same up to float rounding
    /// whatever the side.
    pub local_side: usize,
    /// The whole image (InstanceNorm): the side is part of the result.
    pub whole_image_side: usize,
}

impl Pieces {
    /// The same side for every model (a step's fixed setting).
    pub fn fixed(side: usize) -> Self {
        Self {
            local_side: side,
            whole_image_side: side,
        }
    }

    /// The larger side - what to plan RAM for before the model is known.
    pub fn largest(&self) -> usize {
        self.local_side.max(self.whole_image_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// "Auto" for a model seeing a limited neighbourhood: a U-Net (1.6 KB per
    /// pixel) and StarDist (0.4 KB) on a 4096² slide tile and smaller images
    /// - 2048² pieces where a third of the RAM holds one, smaller ones on a
    /// small computer; an image up to the piece side whole.
    #[test]
    fn pieces_as_large_as_a_third_of_the_ram_holds() {
        let unet = |ram: u64, side: usize| piece_side_for(1600.0, ram, side, side);
        assert_eq!(unet(64 * GB, 4096), 2048);
        assert_eq!(unet(32 * GB, 4096), 2048);
        assert_eq!(unet(16 * GB, 4096), 1024);
        assert_eq!(unet(8 * GB, 4096), 1024);
        assert_eq!(unet(4 * GB, 4096), 512);
        assert_eq!(unet(GB, 4096), 512, "the smallest when none fits");
        assert_eq!(unet(64 * GB, 2048), 2048, "whole");
        assert_eq!(unet(8 * GB, 2048), 1024);
        assert_eq!(unet(8 * GB, 1024), 2048, "whole: it fits");
        assert_eq!(piece_side_for(400.0, 8 * GB, 4096, 4096), 2048);
        let auto = Pieces {
            local_side: 2048,
            whole_image_side: WHOLE_IMAGE_PIECE_SIDE,
        };
        assert_eq!(auto.largest(), 2048);
        assert_eq!(Pieces::fixed(512).largest(), 512);
    }

    #[test]
    fn the_largest_input_is_a_piece_with_its_halo_or_the_tile() {
        assert_eq!(largest_model_input(4096, 2048), 2048 + 256);
        assert_eq!(largest_model_input(4096, 1024), 1024 + 256);
        assert_eq!(largest_model_input(2048, 2048), 2048);
        assert_eq!(largest_model_input(1100, 1024), 1100);
    }
}
