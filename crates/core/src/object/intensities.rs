//! An object's intensities: per channel, sum / min / max / mean over the
//! pixels of its mask that loaded image regions cover - one implementation
//! for pipeline objects ([`Object::measure_intensities`], on the
//! pipeline's cached tiles) and stored annotations ([`ObjectMetricSettings`],
//! measured by the app on regions it reads from their image).
//!
//! Each mask pixel is taken from the first region of a channel that covers
//! it; RGB counts as luminance ([`sample_channel_pixel`]). The mean is over
//! the covered pixels - a part of the mask no region covers is missing
//! data, not dark.

use super::{Intensity, Object};
use crate::ImageContainer;
use crate::pipeline::pipeline_cache::{channel_pixel_slice, sample_channel_pixel};
use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::ImageTile;
use evanalyzer_cfg::settings::object_settings::{IntensitySettings, ObjectMetricSettings};
use indexmap::IndexMap;

/// A loaded region of one channel: where it lies in the full image, its
/// samples (row-major, 3 per pixel for RGB).
#[derive(Debug, Clone, Copy)]
pub struct ChannelTile<'a> {
    pub channel: i32,
    pub tile: ImageTile,
    pub is_rgb: bool,
    pub samples: &'a [f32],
}

impl<'a> ChannelTile<'a> {
    /// Channel `channel`'s region `tile` with the samples of `image`; RGB
    /// as the image data is (as the pipeline decides it). `None` for a
    /// label image.
    pub fn from_container(
        channel: i32,
        tile: ImageTile,
        image: &'a ImageContainer,
    ) -> Option<Self> {
        let is_rgb = match image {
            ImageContainer::F32Gray(_) => false,
            ImageContainer::F32Rgb(_) => true,
            ImageContainer::U32(_) => return None,
        };
        Some(Self {
            channel,
            tile,
            is_rgb,
            samples: channel_pixel_slice(image)?,
        })
    }

    fn covers(&self, x: usize, y: usize) -> bool {
        x >= self.tile.offset_x
            && y >= self.tile.offset_y
            && x < self.tile.offset_x + self.tile.width
            && y < self.tile.offset_y + self.tile.height
    }
}

/// Sum, min, max and covered pixels per channel, added up over one or more
/// [`IntensityAccumulator::add`] calls (e.g. region by region).
#[derive(Debug, Default, Clone)]
pub struct IntensityAccumulator {
    /// In the order channels were first seen.
    channels: IndexMap<i32, (f64, f32, f32, usize)>,
}

impl IntensityAccumulator {
    /// Adds the set pixels of `mask` (relative to `bbox` = `[x1, y1, x2, y2]`,
    /// inclusive) that `tiles` cover. A channel of `tiles` that covers none
    /// of them still appears in the result (empty).
    pub fn add(&mut self, mask: &BitVec<u64, Lsb0>, bbox: [u32; 4], tiles: &[ChannelTile<'_>]) {
        let [x_min, y_min, x_max, y_max] = bbox.map(|v| v as usize);
        let width = x_max - x_min + 1;
        for (k, tile) in tiles.iter().enumerate() {
            let entry = self
                .channels
                .entry(tile.channel)
                .or_insert((0.0, f32::MAX, f32::MIN, 0));
            // The part of the mask's box this region covers.
            let t = tile.tile;
            let (x0, x1) = (x_min.max(t.offset_x), (x_max + 1).min(t.offset_x + t.width));
            let (y0, y1) = (
                y_min.max(t.offset_y),
                (y_max + 1).min(t.offset_y + t.height),
            );
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            // A pixel an earlier region of the channel covers was taken
            // from that one.
            let earlier: Vec<&ChannelTile<'_>> = tiles[..k]
                .iter()
                .filter(|e| e.channel == tile.channel)
                .collect();
            let (mut sum, mut min, mut max, mut n) = (0f64, f32::MAX, f32::MIN, 0usize);
            for y in y0..y1 {
                let mask_row = (y - y_min) * width;
                let tile_row = (y - t.offset_y) * t.width;
                for x in x0..x1 {
                    if !mask[mask_row + x - x_min] {
                        continue;
                    }
                    if !earlier.is_empty() && earlier.iter().any(|e| e.covers(x, y)) {
                        continue;
                    }
                    let value =
                        sample_channel_pixel(tile.is_rgb, tile.samples, tile_row + x - t.offset_x);
                    sum += value as f64;
                    min = min.min(value);
                    max = max.max(value);
                    n += 1;
                }
            }
            entry.0 += sum;
            entry.1 = entry.1.min(min);
            entry.2 = entry.2.max(max);
            entry.3 += n;
        }
    }

    /// The intensities per channel; a channel without covered pixels is 0.
    pub fn finish(self) -> IndexMap<i32, Intensity> {
        self.channels
            .into_iter()
            .map(|(channel, (sum, min, max, n))| {
                let has_data = n > 0;
                (
                    channel,
                    Intensity {
                        sum_intensity: sum,
                        min_intensity: if has_data { min as f64 } else { 0.0 },
                        max_intensity: if has_data { max as f64 } else { 0.0 },
                        avg_intensity: sum / n.max(1) as f64,
                        pixel_values: Vec::new(),
                    },
                )
            })
            .collect()
    }

    /// [`Self::finish`] as stored intensities (an annotation's).
    pub fn finish_settings(self) -> IndexMap<i32, IntensitySettings> {
        self.finish()
            .into_iter()
            .map(|(channel, i)| {
                (
                    channel,
                    IntensitySettings {
                        sum_intensity: i.sum_intensity,
                        min_intensity: i.min_intensity,
                        max_intensity: i.max_intensity,
                        avg_intensity: i.avg_intensity,
                        pixel_values: Vec::new(),
                    },
                )
            })
            .collect()
    }
}

/// Something with a mask whose pixels can be measured: pipeline objects
/// and stored annotations alike.
pub trait MaskIntensities {
    /// Adds this mask's pixels that `tiles` cover to `acc`.
    fn add_intensities(&self, tiles: &[ChannelTile<'_>], acc: &mut IntensityAccumulator);
}

impl MaskIntensities for Object {
    fn add_intensities(&self, tiles: &[ChannelTile<'_>], acc: &mut IntensityAccumulator) {
        acc.add(&self.mask_data, self.bbox, tiles);
    }
}

impl MaskIntensities for ObjectMetricSettings {
    fn add_intensities(&self, tiles: &[ChannelTile<'_>], acc: &mut IntensityAccumulator) {
        acc.add(&self.mask_data, self.bbox, tiles);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::bitvec;

    fn tile(
        channel: i32,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        samples: &[f32],
    ) -> ChannelTile<'_> {
        ChannelTile {
            channel,
            tile: ImageTile {
                offset_x: x,
                offset_y: y,
                width: w,
                height: h,
            },
            is_rgb: false,
            samples,
        }
    }

    #[test]
    fn only_the_masks_pixels_count_each_at_its_place() {
        // A plus in a 3x3 box at (10, 10); one region holding it, values 1..9.
        let mask = bitvec![u64, Lsb0; 0, 1, 0, 1, 1, 1, 0, 1, 0];
        let values: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let annotation = ObjectMetricSettings {
            bbox: [10, 10, 12, 12],
            mask_data: mask,
            ..Default::default()
        };
        let mut acc = IntensityAccumulator::default();
        annotation.add_intensities(&[tile(0, 10, 10, 3, 3, &values)], &mut acc);
        let i = &acc.finish()[&0];
        // 2 + 4 + 5 + 6 + 8.
        assert_eq!(i.sum_intensity, 25.0);
        assert_eq!((i.min_intensity, i.max_intensity), (2.0, 8.0));
        assert_eq!(i.avg_intensity, 5.0);
    }

    #[test]
    fn regions_added_one_by_one_give_the_same_as_at_once() {
        // A 4x1 strip over two regions of 2 px each.
        let annotation = ObjectMetricSettings {
            bbox: [0, 0, 3, 0],
            mask_data: bitvec![u64, Lsb0; 1, 1, 1, 1],
            ..Default::default()
        };
        let (left, right) = ([1.0, 2.0], [3.0, 4.0]);
        let tiles = [tile(0, 0, 0, 2, 1, &left), tile(0, 2, 0, 2, 1, &right)];

        let mut at_once = IntensityAccumulator::default();
        annotation.add_intensities(&tiles, &mut at_once);
        let mut one_by_one = IntensityAccumulator::default();
        for t in &tiles {
            annotation.add_intensities(std::slice::from_ref(t), &mut one_by_one);
        }
        let (a, b) = (&at_once.finish()[&0], &one_by_one.finish()[&0]);
        assert_eq!(
            (a.sum_intensity, a.min_intensity, a.max_intensity),
            (10.0, 1.0, 4.0)
        );
        assert_eq!(
            (b.sum_intensity, b.min_intensity, b.max_intensity),
            (10.0, 1.0, 4.0)
        );
    }

    #[test]
    fn uncovered_pixels_are_missing_not_dark() {
        // 4 px, the region covers 2 of them (values 6, 8): mean 7, not 3.5.
        let annotation = ObjectMetricSettings {
            bbox: [0, 0, 3, 0],
            mask_data: bitvec![u64, Lsb0; 1, 1, 1, 1],
            ..Default::default()
        };
        let mut acc = IntensityAccumulator::default();
        annotation.add_intensities(&[tile(0, 0, 0, 2, 1, &[6.0, 8.0])], &mut acc);
        // A second channel whose region misses the mask entirely.
        annotation.add_intensities(&[tile(1, 50, 50, 2, 1, &[1.0, 1.0])], &mut acc);
        let result = acc.finish_settings();
        assert_eq!(result[&0].avg_intensity, 7.0);
        assert_eq!(result[&1].sum_intensity, 0.0);
        assert_eq!(
            result.keys().copied().collect::<Vec<_>>(),
            vec![0, 1],
            "channel order kept"
        );
    }
}
