//! # validate
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Checking a SpotNet on held-back areas - never trained on, never
//! augmented, the weights untouched. Two numbers per check:
//!
//! - the **loss** there (the same weighted cross-entropy as in training):
//!   smooth, it picks the best epoch;
//! - **found spots**: the probability map's peaks above [`THRESHOLD`]
//!   matched one to one with the annotated spots within [`TOLERANCE_PX`] -
//!   precision (how many found are real), recall (how many real are found)
//!   and F1, what a user reads.
//!
//! The network runs on the held-back region in tiles with a margin of half
//! its receptive field, so every pixel is judged with its full surroundings
//! and only one tile is in RAM at a time.

use crate::model::OwnedImage;
use crate::model::spotnet::network::SpotNet;
use crate::model::spotnet::target::gaussian_map;
use crate::model::spotnet::train::{SPOT_WEIGHT, Spot};
use burn::Tensor;
use burn::tensor::{Device, TensorData};

/// A peak counts as a found spot from this probability on.
pub(crate) const THRESHOLD: f32 = 0.5;
/// A found spot is a hit when an annotated one is at most this far away.
pub(crate) const TOLERANCE_PX: f32 = 2.0;

/// The result of checking on held-back areas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationScore {
    /// The weighted cross-entropy there (lower is better).
    pub loss: f32,
    /// Annotated spots in the areas.
    pub annotated: usize,
    /// Spots the network found there.
    pub found: usize,
    /// Found spots matched to an annotated one.
    pub matched: usize,
}

impl ValidationScore {
    /// How many of the found spots are real (1 when none was found).
    pub fn precision(&self) -> f32 {
        if self.found == 0 {
            1.0
        } else {
            self.matched as f32 / self.found as f32
        }
    }

    /// How many of the annotated spots were found (1 when there are none).
    pub fn recall(&self) -> f32 {
        if self.annotated == 0 {
            1.0
        } else {
            self.matched as f32 / self.annotated as f32
        }
    }

    /// The harmonic mean of precision and recall.
    pub fn f1(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// The sums a score is made of, added up over the images.
#[derive(Debug, Default)]
pub(crate) struct Tally {
    loss_sum: f64,
    weight_sum: f64,
    annotated: usize,
    found: usize,
    matched: usize,
}

impl Tally {
    /// `None` without any held-back pixel.
    pub(crate) fn score(&self) -> Option<ValidationScore> {
        (self.weight_sum > 0.0).then(|| ValidationScore {
            loss: (self.loss_sum / self.weight_sum) as f32,
            annotated: self.annotated,
            found: self.found,
            matched: self.matched,
        })
    }

    /// Checks `model` on the pixels of `image` where `held_back` is true.
    pub(crate) fn add(
        &mut self,
        model: &SpotNet,
        image: &OwnedImage,
        spots: &[Spot],
        held_back: &[bool],
        sigma: f32,
        tile: usize,
        margin: usize,
    ) {
        let Some(region) = Region::around(held_back, image.width, image.height) else {
            return;
        };
        let probability = probabilities(model, image, region, tile, margin);
        let mask = region.cut(held_back, image.width);
        let local: Vec<Spot> = spots
            .iter()
            .map(|s| Spot {
                x: s.x - region.x as f32,
                y: s.y - region.y as f32,
            })
            .collect();
        let target = gaussian_map(region.width, region.height, &local, sigma);

        for ((&p, &t), &inside) in probability.iter().zip(&target).zip(&mask) {
            if inside {
                let p = p.clamp(1e-7, 1.0 - 1e-7);
                let weight = 1.0 + SPOT_WEIGHT * t;
                let bce = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
                self.loss_sum += f64::from(weight * bce);
                self.weight_sum += f64::from(weight);
            }
        }
        let found = find_spots(&probability, region.width, region.height, &mask, THRESHOLD);
        let annotated: Vec<Spot> = local
            .into_iter()
            .filter(|s| region.contains_local(s, &mask))
            .collect();
        self.matched += count_matches(&found, &annotated, TOLERANCE_PX);
        self.found += found.len();
        self.annotated += annotated.len();
    }
}

/// A rectangle of an image, in px.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Region {
    /// The bounding box of the `true` pixels; `None` without any.
    pub(crate) fn around(pixels: &[bool], width: usize, height: usize) -> Option<Self> {
        let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0, 0);
        for (i, _) in pixels.iter().enumerate().filter(|&(_, &on)| on) {
            let (x, y) = (i % width, i / width);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        (x0 <= x1).then(|| Self {
            x: x0,
            y: y0,
            width: x1 - x0 + 1,
            height: y1 - y0 + 1,
        })
    }

    /// This rectangle of one plane of an image `width` wide.
    fn cut<T: Copy>(&self, plane: &[T], width: usize) -> Vec<T> {
        (self.y..self.y + self.height)
            .flat_map(|y| {
                plane[y * width + self.x..y * width + self.x + self.width]
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// Whether a spot in this region's coordinates lies on a `mask` pixel.
    fn contains_local(&self, spot: &Spot, mask: &[bool]) -> bool {
        let (x, y) = (spot.x.floor(), spot.y.floor());
        x >= 0.0
            && y >= 0.0
            && (x as usize) < self.width
            && (y as usize) < self.height
            && mask[y as usize * self.width + x as usize]
    }
}

/// The spot probability of every pixel of `region`, computed in tiles of up
/// to `tile` px: each tile reaches `margin` px further on every side (as far
/// as the image goes), and only its inner part is kept - so a pixel near a
/// tile's edge sees the same surroundings as in one big pass.
pub(crate) fn probabilities(
    model: &SpotNet,
    image: &OwnedImage,
    region: Region,
    tile: usize,
    margin: usize,
) -> Vec<f32> {
    let device = Device::flex();
    let step = tile.saturating_sub(2 * margin).max(16);
    let mut out = vec![0.0f32; region.width * region.height];
    for ty in (0..region.height).step_by(step) {
        for tx in (0..region.width).step_by(step) {
            // The kept part, in image coordinates.
            let (kx, ky) = (region.x + tx, region.y + ty);
            let (kw, kh) = (step.min(region.width - tx), step.min(region.height - ty));
            // The window run through the network: the kept part plus margin.
            let wx = kx.saturating_sub(margin);
            let wy = ky.saturating_sub(margin);
            let ww = (kx + kw + margin).min(image.width) - wx;
            let wh = (ky + kh + margin).min(image.height) - wy;
            let window = Region {
                x: wx,
                y: wy,
                width: ww,
                height: wh,
            };
            let plane = image.width * image.height;
            let input: Vec<f32> = (0..image.channels)
                .flat_map(|c| window.cut(&image.data[c * plane..(c + 1) * plane], image.width))
                .collect();
            let x = Tensor::<4>::from_data(
                TensorData::new(input, [1, image.channels, wh, ww]),
                &device,
            );
            let p = model
                .forward(x)
                .into_data()
                .try_to_vec::<f32>()
                .expect("f32 output");
            for y in 0..kh {
                let from = (ky + y - wy) * ww + (kx - wx);
                let to = (ty + y) * region.width + tx;
                out[to..to + kw].copy_from_slice(&p[from..from + kw]);
            }
        }
    }
    out
}

/// The peaks of `probability` (`width x height`) above `threshold` on `mask`
/// pixels: a pixel at least as high as its 8 neighbours - on a plateau of
/// equal values only the first in reading order. As spots at pixel centres.
pub(crate) fn find_spots(
    probability: &[f32],
    width: usize,
    height: usize,
    mask: &[bool],
    threshold: f32,
) -> Vec<Spot> {
    let mut spots = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let p = probability[y * width + x];
            if !mask[y * width + x] || p <= threshold {
                continue;
            }
            let mut peak = true;
            'neighbours: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx, dy) == (0, 0) {
                        continue;
                    }
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let q = probability[ny as usize * width + nx as usize];
                    // A neighbour before this pixel in reading order wins a tie.
                    let before = dy < 0 || (dy == 0 && dx < 0);
                    if q > p || (before && q == p) {
                        peak = false;
                        break 'neighbours;
                    }
                }
            }
            if peak {
                spots.push(Spot {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                });
            }
        }
    }
    spots
}

/// How many `found` spots match an `annotated` one at most `tolerance` px
/// away - one to one, the closest pairs first.
pub(crate) fn count_matches(found: &[Spot], annotated: &[Spot], tolerance: f32) -> usize {
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (i, f) in found.iter().enumerate() {
        for (j, a) in annotated.iter().enumerate() {
            let d = (f.x - a.x).hypot(f.y - a.y);
            if d <= tolerance {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut used_found, mut used_annotated) =
        (vec![false; found.len()], vec![false; annotated.len()]);
    let mut matched = 0;
    for (_, i, j) in pairs {
        if !used_found[i] && !used_annotated[j] {
            used_found[i] = true;
            used_annotated[j] = true;
            matched += 1;
        }
    }
    matched
}

#[cfg(test)]
mod tests {
    use super::{Region, ValidationScore, count_matches, find_spots, probabilities};
    use crate::model::OwnedImage;
    use crate::model::spotnet::network::{SpotNet, SpotNetConfig};
    use crate::model::spotnet::train::Spot;
    use burn::tensor::Device;

    #[test]
    fn tiles_with_margin_give_what_one_pass_over_the_image_gives() {
        // A 2-channel 50 x 37 image of varied values and a random network:
        // small tiles (each with a margin of half the receptive field) match
        // one pass over the whole region.
        let config = SpotNetConfig {
            channels: 4,
            ..SpotNetConfig::default()
        };
        let model = SpotNet::new(&config, 3, &Device::flex());
        let (width, height) = (50, 37);
        let data = (0..2 * width * height)
            .map(|i| ((i * 7919) % 101) as f32 / 100.0)
            .collect();
        let image = OwnedImage {
            width,
            height,
            channels: 2,
            data,
        };
        let region = Region {
            x: 3,
            y: 5,
            width: 41,
            height: 30,
        };
        let margin = config.receptive_field() / 2;
        let whole = probabilities(&model, &image, region, 1000, margin);
        let tiled = probabilities(&model, &image, region, 2 * margin + 16, margin);
        let worst = whole
            .iter()
            .zip(&tiled)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "tiles differ by {worst}");
    }

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
        assert_eq!(find_spots(&p, 5, 3, &[true; 15], 0.5), vec![spot(3.5, 1.5)]);
    }

    #[test]
    fn peaks_below_the_threshold_or_outside_the_mask_are_no_spots() {
        let p = [0.0, 0.4, 0.0, 0.0, 0.9, 0.0];
        assert!(find_spots(&p, 3, 2, &[true; 6], 0.5).len() == 1);
        assert!(find_spots(&p, 3, 2, &[true, true, true, false, false, false], 0.5).is_empty());
        assert!(find_spots(&[0.0, 0.4, 0.0], 3, 1, &[true; 3], 0.5).is_empty());
    }

    #[test]
    fn a_plateau_is_one_spot() {
        let p = [0.0, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0];
        assert_eq!(find_spots(&p, 4, 2, &[true; 8], 0.5), vec![spot(1.5, 0.5)]);
    }

    #[test]
    fn two_close_peaks_are_two_spots() {
        let p = [0.0, 0.9, 0.5, 0.9, 0.0];
        assert_eq!(
            find_spots(&p, 5, 1, &[true; 5], 0.5),
            vec![spot(1.5, 0.5), spot(3.5, 0.5)]
        );
    }

    #[test]
    fn matches_are_within_the_tolerance_and_one_to_one() {
        let annotated = [spot(5.0, 5.0), spot(20.0, 5.0)];
        // Two found near the first annotated one (only one may match it),
        // one too far from the second.
        let found = [spot(5.5, 5.5), spot(6.0, 6.0), spot(23.0, 5.0)];
        assert_eq!(count_matches(&found, &annotated, 2.0), 1);
        assert_eq!(count_matches(&[], &annotated, 2.0), 0);
    }

    #[test]
    fn the_closest_pairs_are_matched_first() {
        // Found A is close to both; B only to the second. Greedy by distance
        // gives A-first, B-second: two matches, not one.
        let annotated = [spot(0.0, 0.0), spot(2.0, 0.0)];
        let found = [spot(0.2, 0.0), spot(3.5, 0.0)];
        assert_eq!(count_matches(&found, &annotated, 2.0), 2);
    }

    #[test]
    fn the_score_reads_as_precision_recall_and_f1() {
        let score = ValidationScore {
            loss: 0.1,
            annotated: 10,
            found: 8,
            matched: 6,
        };
        assert_eq!(score.precision(), 0.75);
        assert_eq!(score.recall(), 0.6);
        assert!((score.f1() - 2.0 * 0.75 * 0.6 / 1.35).abs() < 1e-6);
        let nothing = ValidationScore {
            loss: 0.0,
            annotated: 0,
            found: 0,
            matched: 0,
        };
        assert_eq!(nothing.f1(), 1.0);
    }

    #[test]
    fn a_region_is_the_bounding_box_of_its_pixels() {
        let mut pixels = vec![false; 6 * 4];
        pixels[6 + 2] = true;
        pixels[2 * 6 + 4] = true;
        assert_eq!(
            Region::around(&pixels, 6, 4),
            Some(Region {
                x: 2,
                y: 1,
                width: 3,
                height: 2
            })
        );
        assert_eq!(Region::around(&[false; 4], 2, 2), None);
    }
}
