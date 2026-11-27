//! # post
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! From SpotNet's probability map to spots: its peaks above a threshold
//! (Find Maxima on the map = non-maximum suppression), each written as a
//! one-pixel object - instance labels `1..=n` in reading order.

use crate::errors::NnError;
use crate::model::spotnet::train::Spot;

/// `out` must hold one value per pixel of the tile.
pub(crate) fn check_size(out: &[u32], width: usize, height: usize) -> Result<(), NnError> {
    if out.len() != width * height {
        return Err(NnError::InvalidInput(format!(
            "SpotNet: {} values for a {width}x{height} tile",
            out.len()
        )));
    }
    Ok(())
}

/// The peaks of `probability` (`width x height`) above `threshold` on `mask`
/// pixels (`None`: everywhere): a pixel at least as high as its 8
/// neighbours - on a plateau of equal values only the first in reading
/// order. As spots at pixel centres, in reading order.
pub(crate) fn find_spots(
    probability: &[f32],
    width: usize,
    height: usize,
    mask: Option<&[bool]>,
    threshold: f32,
) -> Vec<Spot> {
    let mut spots = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let p = probability[y * width + x];
            if mask.is_some_and(|m| !m[y * width + x]) || p <= threshold {
                continue;
            }
            if is_peak(probability, width, height, x, y, p) {
                spots.push(Spot {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                });
            }
        }
    }
    spots
}

/// Whether `p` at (`x`, `y`) is a peak: no neighbour higher, and none
/// before it in reading order as high (a plateau's first pixel wins).
fn is_peak(probability: &[f32], width: usize, height: usize, x: usize, y: usize, p: f32) -> bool {
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if (dx, dy) == (0, 0) {
                continue;
            }
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let q = probability[ny as usize * width + nx as usize];
            let before = dy < 0 || (dy == 0 && dx < 0);
            if q > p || (before && q == p) {
                return false;
            }
        }
    }
    true
}

/// `spots` as one-pixel objects into `out` (cleared first): the first spot
/// `1`, the next `2`, ... - their number.
pub(crate) fn write_spots(spots: &[Spot], width: usize, out: &mut [u32]) -> u32 {
    out.fill(0);
    for (id, spot) in spots.iter().enumerate() {
        out[spot.y as usize * width + spot.x as usize] = id as u32 + 1;
    }
    spots.len() as u32
}

#[cfg(test)]
mod tests {
    use super::{find_spots, write_spots};
    use crate::model::spotnet::train::Spot;

    fn spot(x: f32, y: f32) -> Spot {
        Spot { x, y }
    }

    #[test]
    fn a_peak_above_the_threshold_is_a_spot_at_its_pixel_centre() {
        // 5 x 3 with one hill at (3, 1).
        #[rustfmt::skip]
        let p = [
            0.0, 0.1, 0.3, 0.4, 0.2,
            0.0, 0.2, 0.6, 0.9, 0.5,
            0.0, 0.1, 0.3, 0.4, 0.2,
        ];
        assert_eq!(find_spots(&p, 5, 3, None, 0.5), vec![spot(3.5, 1.5)]);
    }

    #[test]
    fn peaks_below_the_threshold_or_outside_the_mask_are_no_spots() {
        let p = [0.0, 0.4, 0.0, 0.0, 0.9, 0.0];
        assert_eq!(find_spots(&p, 3, 2, None, 0.5).len(), 1);
        let top_row = [true, true, true, false, false, false];
        assert!(find_spots(&p, 3, 2, Some(&top_row), 0.5).is_empty());
        assert!(find_spots(&[0.0, 0.4, 0.0], 3, 1, None, 0.5).is_empty());
    }

    #[test]
    fn a_plateau_is_one_spot() {
        let p = [0.0, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0];
        assert_eq!(find_spots(&p, 4, 2, None, 0.5), vec![spot(1.5, 0.5)]);
    }

    #[test]
    fn two_close_peaks_are_two_spots() {
        let p = [0.0, 0.9, 0.5, 0.9, 0.0];
        assert_eq!(
            find_spots(&p, 5, 1, None, 0.5),
            vec![spot(1.5, 0.5), spot(3.5, 0.5)]
        );
    }

    #[test]
    fn every_spot_is_its_own_one_pixel_object() {
        let mut out = vec![7u32; 4 * 3];
        let n = write_spots(&[spot(1.5, 0.5), spot(3.5, 2.5)], 4, &mut out);
        assert_eq!(n, 2);
        assert_eq!(out, vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
    }
}
