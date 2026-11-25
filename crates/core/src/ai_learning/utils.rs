//! # utils
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::image::ImageTile;
use crate::object::Object;
use bitvec::prelude::{BitVec, Lsb0};

// -- Tiling / z-stack helpers, shared by pixel training (and the tile grid
// by the object measuring in `object::regions`) ----------------------------

pub fn tile_grid(full_width: usize, full_height: usize, tile_size: usize) -> Vec<ImageTile> {
    let x_steps = full_width.div_ceil(tile_size);
    let y_steps = full_height.div_ceil(tile_size);
    let mut tiles = Vec::with_capacity(x_steps * y_steps);
    for y in 0..y_steps {
        for x in 0..x_steps {
            let offset_x = x * tile_size;
            let offset_y = y * tile_size;
            tiles.push(ImageTile {
                offset_x,
                offset_y,
                width: (full_width - offset_x).min(tile_size),
                height: (full_height - offset_y).min(tile_size),
            });
        }
    }
    tiles
}

pub fn bbox_overlaps_tile(bbox: [u32; 4], tile: &ImageTile) -> bool {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;
    x_min <= tile_x_max && x_max >= tile_x_min && y_min <= tile_y_max && y_max >= tile_y_min
}

/// (x, y) coordinates (in the image's full-resolution grid, `usize`) of every
/// pixel in `object`'s mask that falls within `tile`'s bounds.
pub fn masked_pixels_in_tile(object: &Object, tile: &ImageTile) -> Vec<(usize, usize)> {
    mask_pixels_in_tile(&object.mask_data, object.bbox, tile)
}

/// [`masked_pixels_in_tile`] for a `mask` relative to `bbox` (`[x_min,
/// y_min, x_max, y_max]`, inclusive) - e.g. of stored annotations.
pub fn mask_pixels_in_tile(
    mask: &BitVec<u64, Lsb0>,
    bbox: [u32; 4],
    tile: &ImageTile,
) -> Vec<(usize, usize)> {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;

    let ix_min = x_min.max(tile_x_min);
    let iy_min = y_min.max(tile_y_min);
    let ix_max = x_max.min(tile_x_max);
    let iy_max = y_max.min(tile_y_max);

    if ix_min > ix_max || iy_min > iy_max {
        return Vec::new();
    }

    let width = (x_max - x_min + 1) as usize;
    let mut samples = Vec::new();
    for y in iy_min..=iy_max {
        for x in ix_min..=ix_max {
            let bit = (y - y_min) as usize * width + (x - x_min) as usize;
            if mask.get(bit).is_some_and(|b| *b) {
                samples.push((x as usize, y as usize));
            }
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;

    // -- tile_grid -----------------------------------------------------------

    fn tile_fields(t: &ImageTile) -> (usize, usize, usize, usize) {
        (t.offset_x, t.offset_y, t.width, t.height)
    }

    #[test]
    fn tile_grid_splits_evenly_divisible_dimensions_into_equal_tiles() {
        let tiles = tile_grid(20, 10, 10);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tile_fields(&tiles[0]), (0, 0, 10, 10));
        assert_eq!(tile_fields(&tiles[1]), (10, 0, 10, 10));
    }

    #[test]
    fn tile_grid_shrinks_the_last_tile_for_non_divisible_dimensions() {
        let tiles = tile_grid(15, 8, 10);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].width, 10);
        assert_eq!(tiles[1].width, 5);
        assert_eq!(tiles[0].height, 8);
    }

    #[test]
    fn tile_grid_of_a_size_smaller_than_one_tile_returns_a_single_tile() {
        let tiles = tile_grid(5, 5, 100);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].width, 5);
        assert_eq!(tiles[0].height, 5);
    }

    // -- bbox_overlaps_tile ----------------------------------------------------

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    #[test]
    fn bbox_overlaps_tile_true_when_fully_inside() {
        assert!(bbox_overlaps_tile([2, 2, 5, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_true_when_partially_overlapping_at_an_edge() {
        // bbox spans x 8..12, tile covers x 0..9 (inclusive) - overlaps at x=8.
        assert!(bbox_overlaps_tile([8, 0, 12, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_false_when_entirely_to_the_right() {
        assert!(!bbox_overlaps_tile([20, 0, 25, 5], &tile(0, 0, 10, 10)));
    }

    #[test]
    fn bbox_overlaps_tile_false_when_entirely_below() {
        assert!(!bbox_overlaps_tile([0, 20, 5, 25], &tile(0, 0, 10, 10)));
    }

    // -- masked_pixels_in_tile ------------------------------------------------

    fn square_object(x_min: u32, y_min: u32, side: u32) -> Object {
        let x_max = x_min + side - 1;
        let y_max = y_min + side - 1;
        let mut mask = bitvec![u64, Lsb0; 1; (side * side) as usize];
        mask.set(0, true); // no-op, keeps mask fully filled for clarity
        Object::new(ObjectInit {
            bbox: [x_min, y_min, x_max, y_max],
            mask_data: mask,
            area: (side * side) as usize,
            ..Default::default()
        })
    }

    #[test]
    fn masked_pixels_in_tile_returns_every_mask_pixel_when_the_tile_fully_covers_it() {
        let object = square_object(0, 0, 2); // pixels (0,0),(1,0),(0,1),(1,1)
        let pixels = masked_pixels_in_tile(&object, &tile(0, 0, 10, 10));
        let mut pixels = pixels;
        pixels.sort();
        assert_eq!(pixels, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn masked_pixels_in_tile_clips_to_the_tiles_bounds() {
        let object = square_object(0, 0, 4); // 4x4 object at origin
        // Tile only covers the left half (x: 0..=1).
        let pixels = masked_pixels_in_tile(&object, &tile(0, 0, 2, 4));
        assert!(pixels.iter().all(|&(x, _)| x < 2));
        assert_eq!(pixels.len(), 8); // 2 columns x 4 rows
    }

    #[test]
    fn masked_pixels_in_tile_is_empty_when_the_tile_does_not_overlap_the_object() {
        let object = square_object(0, 0, 2);
        let pixels = masked_pixels_in_tile(&object, &tile(100, 100, 10, 10));
        assert!(pixels.is_empty());
    }
}
