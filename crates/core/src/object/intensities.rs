//! # intensities
//!
//! An object's intensities: per channel, sum / min / max / mean and how the
//! values spread (std, median, quartiles, MAD) over the pixels of its mask
//! that loaded image regions cover - one implementation for pipeline
//! objects ([`Object::measure_intensities`], on the pipeline's cached
//! tiles) and stored annotations ([`ObjectMetricSettings`], measured by the
//! app on regions it reads from their image).
//!
//! Each mask pixel is taken from the first region of a channel that covers
//! it; RGB counts as luminance ([`sample_channel_pixel`]). The mean is over
//! the covered pixels - a part of the mask no region covers is missing
//! data, not dark.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

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

impl Intensity {
    /// Sets how `values` - the object's pixel values in this channel (put in
    /// order here) - spread, from its `avg_intensity`; NaN without values.
    pub fn set_spread(&mut self, values: &mut [f32]) {
        self.set_spread_to(Spread::of(values, self.avg_intensity));
    }

    fn set_spread_to(&mut self, spread: Option<Spread>) {
        let [std, median, lower, upper, mad] = spread.map_or([f64::NAN; 5], |s| {
            [s.std, s.median, s.lower_quartile, s.upper_quartile, s.mad]
        });
        self.std_intensity = std;
        self.median_intensity = median;
        self.lower_quartile_intensity = lower;
        self.upper_quartile_intensity = upper;
        self.mad_intensity = mad;
    }

    /// Whether the spread was measured (not a stored annotation's).
    pub fn has_spread(&self) -> bool {
        !self.median_intensity.is_nan()
    }

    /// As stored with an annotation (the spread none where NaN).
    pub fn to_settings(&self) -> IntensitySettings {
        let stored = |v: f64| v.is_finite().then_some(v);
        IntensitySettings {
            sum_intensity: self.sum_intensity,
            min_intensity: self.min_intensity,
            max_intensity: self.max_intensity,
            avg_intensity: self.avg_intensity,
            std_intensity: stored(self.std_intensity),
            median_intensity: stored(self.median_intensity),
            lower_quartile_intensity: stored(self.lower_quartile_intensity),
            upper_quartile_intensity: stored(self.upper_quartile_intensity),
            mad_intensity: stored(self.mad_intensity),
        }
    }

    /// From an annotation's stored intensities (the spread NaN where none).
    pub fn from_settings(s: &IntensitySettings) -> Self {
        let read = |v: Option<f64>| v.unwrap_or(f64::NAN);
        Self {
            sum_intensity: s.sum_intensity,
            min_intensity: s.min_intensity,
            max_intensity: s.max_intensity,
            avg_intensity: s.avg_intensity,
            std_intensity: read(s.std_intensity),
            median_intensity: read(s.median_intensity),
            lower_quartile_intensity: read(s.lower_quartile_intensity),
            upper_quartile_intensity: read(s.upper_quartile_intensity),
            mad_intensity: read(s.mad_intensity),
        }
    }
}

/// How pixel values spread - computed by [`Spread::of`], kept in
/// [`Intensity`]'s fields.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct Spread {
    pub std: f64,
    pub median: f64,
    pub lower_quartile: f64,
    pub upper_quartile: f64,
    pub mad: f64,
}

impl Spread {
    /// The spread of `values` (put in order here), whose mean is `mean` -
    /// `None` without values. Uses the conventions of the established
    /// object-intensity statistics, so results stay comparable to them:
    /// the population standard deviation; quartiles and median at position
    /// n·f of the sorted values, between the two values around it (the
    /// value there alone at the top end - not numpy's (n-1)·f); the MAD the
    /// same way at n/2 of the distances from the median.
    pub fn of(values: &mut [f32], mean: f64) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let variance = |values: &[f32]| {
            values
                .iter()
                .map(|v| (*v as f64 - mean).powi(2))
                .sum::<f64>()
                / n
        };
        if values.len() > SORT_UP_TO
            && let Some(spread) = Self::from_histogram(values, mean)
        {
            return Some(spread);
        }
        let variance = variance(values);
        if values.len() <= SORT_UP_TO {
            values.sort_unstable_by(f32::total_cmp);
            let sorted = &*values;
            let median = quantile(sorted.len(), 0.5, |i| sorted[i] as f64);
            return Some(Self {
                std: variance.sqrt(),
                median,
                lower_quartile: quantile(sorted.len(), 0.25, |i| sorted[i] as f64),
                upper_quartile: quantile(sorted.len(), 0.75, |i| sorted[i] as f64),
                mad: median_distance(sorted, median),
            });
        }
        // Bigger objects: only the few values needed, picked out in linear
        // time (sorting every object's values cost a pipeline ~20 %).
        let median = selected_quantile(values, 0.5, f32::total_cmp, |v| *v as f64);
        let lower_quartile = selected_quantile(values, 0.25, f32::total_cmp, |v| *v as f64);
        let upper_quartile = selected_quantile(values, 0.75, f32::total_cmp, |v| *v as f64);
        let mut distances: Vec<f64> = values.iter().map(|v| (*v as f64 - median).abs()).collect();
        Some(Self {
            std: variance.sqrt(),
            median,
            lower_quartile,
            upper_quartile,
            mad: selected_quantile(&mut distances, 0.5, f64::total_cmp, |v| *v),
        })
    }
}

impl Spread {
    /// The order statistics from a histogram of `values` - one level per
    /// gray value (raw 8/16-bit values), or 65535 levels over 0..1 (exact
    /// for 8- and 16-bit images scaled to 0..1: 65535 = 255 · 257) - one pass
    /// over the pixels instead of several, for big objects. `None` (the
    /// caller selects instead) when the values span more levels than pays
    /// off, or two different values fall into one level (a float image):
    /// each level keeps the exact value it holds, so the numbers are the
    /// same as by sorting.
    fn from_histogram(values: &[f32], mean: f64) -> Option<Self> {
        let (min, max) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        if !(min.is_finite() && max.is_finite()) || min < 0.0 {
            return None;
        }
        let levels: f32 = if max <= 1.0 { 65535.0 } else { 1.0 };
        // Rounded (values aren't negative here); `round` is a slow call.
        let level = |v: f32| (v * levels + 0.5) as usize;
        let first = level(min);
        let bins = level(max) - first + 1;
        // Zeroing and scanning the bins must cost less than selecting.
        if bins > values.len() * 4 {
            return None;
        }
        let mut count = vec![0u32; bins];
        let mut value = vec![0f32; bins];
        for v in values {
            let b = level(*v) - first;
            if count[b] == 0 {
                value[b] = *v;
            } else if value[b] != *v {
                return None;
            }
            count[b] += 1;
        }
        Some(Self::from_counts(
            &count,
            |b| value[b] as f64,
            values.len(),
            mean,
        ))
    }

    /// The spread of `n` values given as how many fall on each level
    /// (`count`, in order) and the value of each level (`value`).
    fn from_counts(count: &[u32], value: impl Fn(usize) -> f64, n: usize, mean: f64) -> Self {
        let bins = count.len();
        // Over the levels (the same sum as over the pixels, in another order).
        let variance = count
            .iter()
            .enumerate()
            .map(|(b, c)| *c as f64 * (value(b) - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        // The i-th value in order (0-based): its level by the counts.
        let nth = |i: usize| -> f64 {
            let mut seen = 0usize;
            for (b, c) in count.iter().enumerate() {
                seen += *c as usize;
                if seen > i {
                    return value(b);
                }
            }
            value(bins - 1)
        };
        let median = quantile(n, 0.5, nth);
        let lower_quartile = quantile(n, 0.25, nth);
        let upper_quartile = quantile(n, 0.75, nth);
        // The distances from the median in order: walking out from it over
        // the levels, like `median_distance` over sorted values.
        let split = (0..bins)
            .find(|&b| count[b] > 0 && value(b) >= median)
            .unwrap_or(bins);
        let mad = quantile(n, 0.5, |k| {
            // Whole levels at a time: all of a level's values are equally far.
            let (mut left, mut right, mut seen) = (split, split, 0usize);
            loop {
                while left > 0 && count[left - 1] == 0 {
                    left -= 1;
                }
                while right < bins && count[right] == 0 {
                    right += 1;
                }
                let from_left = (left > 0).then(|| median - value(left - 1));
                let from_right = (right < bins).then(|| value(right) - median);
                let (distance, taken) = match (from_left, from_right) {
                    (Some(l), Some(r)) if l <= r => {
                        left -= 1;
                        (l, count[left])
                    }
                    (_, Some(r)) => {
                        right += 1;
                        (r, count[right - 1])
                    }
                    (Some(l), None) => {
                        left -= 1;
                        (l, count[left])
                    }
                    (None, None) => unreachable!("walked past every value"),
                };
                seen += taken as usize;
                if seen > k {
                    return distance;
                }
            }
        });
        Self {
            std: variance.sqrt(),
            median,
            lower_quartile,
            upper_quartile,
            mad,
        }
    }
}

/// Up to this many values are sorted; more are selected from.
const SORT_UP_TO: usize = 32;

/// [`quantile`] of `values` (not sorted, more than one; reordered), the
/// value(s) it needs selected in linear time.
fn selected_quantile<T>(
    values: &mut [T],
    fraction: f64,
    cmp: fn(&T, &T) -> std::cmp::Ordering,
    as_f64: fn(&T) -> f64,
) -> f64 {
    let n = values.len();
    let i = (n as f64 * fraction).floor() as usize;
    let i = i.min(n - 1);
    let (_, at_i, after) = values.select_nth_unstable_by(i, cmp);
    let at_i = as_f64(at_i);
    // The next value in order: the smallest of those after it.
    let at_next = after.iter().min_by(|a, b| cmp(a, b)).map(as_f64);
    quantile(n, fraction, |k| {
        if k == i {
            at_i
        } else {
            at_next.unwrap_or(at_i)
        }
    })
}

/// The quantile of `n` (> 0) sorted values (`at(i)`: the i-th):
/// at q = n·`fraction` between the values at ⌊q⌋ and ⌊q⌋ + 1, the value at
/// ⌊q⌋ alone when no value follows it.
fn quantile(n: usize, fraction: f64, mut at: impl FnMut(usize) -> f64) -> f64 {
    let q = n as f64 * fraction;
    let i = q.floor() as usize;
    let f = q - q.floor();
    if i + 1 < n {
        at(i) * (1.0 - f) + at(i + 1) * f
    } else {
        at(i.min(n - 1))
    }
}

/// [`quantile`] 0.5 of the distances of `sorted` from `median`, in f64 (so
/// ties and rounding are stable) - in order without sorting them: walking out
/// from the median, the distances on both sides grow, so merging the two
/// sides gives them smallest first.
fn median_distance(sorted: &[f32], median: f64) -> f64 {
    let split = sorted.partition_point(|v| (*v as f64) < median);
    let (mut left, mut right) = (split, split);
    // The distances up to the one at index `upto`, the last kept.
    let mut nth = |upto: usize| -> (f64, f64) {
        let (mut previous, mut current) = (f64::NAN, f64::NAN);
        for _ in 0..=upto {
            let from_left = (left > 0).then(|| median - sorted[left - 1] as f64);
            let from_right = (right < sorted.len()).then(|| sorted[right] as f64 - median);
            let next = match (from_left, from_right) {
                (Some(l), Some(r)) if l <= r => {
                    left -= 1;
                    l
                }
                (_, Some(r)) => {
                    right += 1;
                    r
                }
                (Some(l), None) => {
                    left -= 1;
                    l
                }
                (None, None) => unreachable!("walked past every value"),
            };
            previous = current;
            current = next;
        }
        (previous, current)
    };
    let n = sorted.len();
    let i = (n as f64 * 0.5).floor() as usize;
    let (at_i, at_next) = if i + 1 < n {
        nth(i + 1)
    } else {
        let (_, at_i) = nth(i.min(n - 1));
        (at_i, f64::NAN)
    };
    quantile(n, 0.5, |k| if k == i { at_i } else { at_next })
}

/// One channel's sum, min, max and covered pixels' values.
#[derive(Debug, Clone)]
struct ChannelSums {
    sum: f64,
    min: f32,
    max: f32,
    values: Vec<f32>,
    /// From [`LIST_UP_TO`] values on, whole gray values 0..=65535 (8/16-bit
    /// images) are counted per value here instead of listed (256 KB however
    /// big the object); `values` then holds none.
    counts: Option<Box<[u32]>>,
}

/// Values listed per channel before whole gray values are counted instead.
const LIST_UP_TO: usize = 1 << 16;

/// Counts per gray value: 0..=65535.
const GRAY_LEVELS: usize = 1 << 16;

impl ChannelSums {
    fn new() -> Self {
        Self {
            sum: 0.0,
            min: f32::MAX,
            max: f32::MIN,
            values: Vec::new(),
            counts: None,
        }
    }

    fn push(&mut self, value: f32) {
        self.sum += value as f64;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        let level = value as usize;
        let whole = level as f32 == value && level < GRAY_LEVELS;
        match &mut self.counts {
            Some(counts) if whole => counts[level] += 1,
            Some(_) => {
                // Not a gray value after all: back to the list (exact - each
                // count stands for its level's value).
                self.values = self.listed();
                self.counts = None;
                self.values.push(value);
            }
            None => {
                self.values.push(value);
                if self.values.len() >= LIST_UP_TO
                    && self.values.iter().all(|v| {
                        let level = *v as usize;
                        level as f32 == *v && level < GRAY_LEVELS
                    })
                {
                    let mut counts = vec![0u32; GRAY_LEVELS].into_boxed_slice();
                    for v in self.values.drain(..) {
                        counts[v as usize] += 1;
                    }
                    self.values = Vec::new();
                    self.counts = Some(counts);
                }
            }
        }
    }

    /// Every value as a list (the counted ones expanded).
    fn listed(&self) -> Vec<f32> {
        match &self.counts {
            Some(counts) => counts
                .iter()
                .enumerate()
                .flat_map(|(level, c)| std::iter::repeat_n(level as f32, *c as usize))
                .collect(),
            None => self.values.clone(),
        }
    }

    fn len(&self) -> usize {
        match &self.counts {
            Some(counts) => counts.iter().map(|c| *c as usize).sum(),
            None => self.values.len(),
        }
    }
}

/// Sum, min, max and the covered pixels' values per channel, added up over
/// one or more [`IntensityAccumulator::add`] calls (e.g. region by region).
#[derive(Debug, Default, Clone)]
pub struct IntensityAccumulator {
    /// In the order channels were first seen.
    channels: IndexMap<i32, ChannelSums>,
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
                .or_insert_with(ChannelSums::new);
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
                    entry.push(value);
                }
            }
        }
    }

    /// The intensities per channel; a channel without covered pixels is 0
    /// (and no spread).
    pub fn finish(self) -> IndexMap<i32, Intensity> {
        self.channels
            .into_iter()
            .map(|(channel, mut c)| {
                let n = c.len();
                let has_data = n > 0;
                let avg = c.sum / n.max(1) as f64;
                let mut intensity = Intensity {
                    sum_intensity: c.sum,
                    min_intensity: if has_data { c.min as f64 } else { 0.0 },
                    max_intensity: if has_data { c.max as f64 } else { 0.0 },
                    avg_intensity: avg,
                    ..Default::default()
                };
                match &c.counts {
                    Some(counts) => intensity.set_spread_to(Some(Spread::from_counts(
                        counts,
                        |level| level as f64,
                        n,
                        avg,
                    ))),
                    None => intensity.set_spread(&mut c.values),
                }
                (channel, intensity)
            })
            .collect()
    }

    /// [`Self::finish`] as stored intensities (an annotation's).
    pub fn finish_settings(self) -> IndexMap<i32, IntensitySettings> {
        self.finish()
            .into_iter()
            .map(|(channel, i)| (channel, i.to_settings()))
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
pub(crate) mod tests {
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

    /// One case of `tests/object_intensity.json`: a label
    /// image, its pixel values, and the reference statistics per label.
    #[derive(serde::Deserialize)]
    pub(crate) struct ReferenceCase {
        pub name: String,
        pub width: usize,
        pub height: usize,
        pub labels: Vec<u32>,
        pub image: Vec<f32>,
        pub integrated: Vec<f64>,
        pub mean: Vec<f64>,
        pub std: Vec<f64>,
        pub min: Vec<f64>,
        pub max: Vec<f64>,
        pub lower_quartile: Vec<f64>,
        pub median: Vec<f64>,
        pub upper_quartile: Vec<f64>,
        pub mad: Vec<f64>,
    }

    pub(crate) fn reference_cases() -> Vec<ReferenceCase> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/object_intensity.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// Asserts `got` has the reference statistics of label `label` (1-based)
    /// of `case`: sums to rounding, the order statistics exactly.
    pub(crate) fn assert_matches_reference(got: &Intensity, case: &ReferenceCase, label: usize) {
        let i = label - 1;
        let what = format!("{} label {label}", case.name);
        let close = |a: f64, b: f64, name: &str| {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "{what} {name}: {a} vs {b}"
            )
        };
        close(got.sum_intensity, case.integrated[i], "integrated");
        close(got.avg_intensity, case.mean[i], "mean");
        assert_eq!(got.min_intensity, case.min[i], "{what} min");
        assert_eq!(got.max_intensity, case.max[i], "{what} max");
        assert!(got.has_spread(), "{what}: measured from pixels");
        close(got.std_intensity, case.std[i], "std");
        assert_eq!(
            got.lower_quartile_intensity, case.lower_quartile[i],
            "{what} lower quartile"
        );
        assert_eq!(got.median_intensity, case.median[i], "{what} median");
        assert_eq!(
            got.upper_quartile_intensity, case.upper_quartile[i],
            "{what} upper quartile"
        );
        assert_eq!(got.mad_intensity, case.mad[i], "{what} MAD");
    }

    /// Std, quartiles, median and MAD of each label match the
    /// reference values (fixture from
    /// `tests/object_intensity.json`).
    #[test]
    fn spread_matches_reference() {
        let mut compared = 0;
        for case in reference_cases() {
            let labels = case.labels.iter().copied().max().unwrap_or(0) as usize;
            for label in 1..=labels {
                // The label's box and its mask in it.
                let pixels: Vec<(usize, usize)> = (0..case.width * case.height)
                    .filter(|&p| case.labels[p] == label as u32)
                    .map(|p| (p % case.width, p / case.width))
                    .collect();
                let x0 = pixels.iter().map(|p| p.0).min().unwrap();
                let y0 = pixels.iter().map(|p| p.1).min().unwrap();
                let x1 = pixels.iter().map(|p| p.0).max().unwrap();
                let y1 = pixels.iter().map(|p| p.1).max().unwrap();
                let w = x1 - x0 + 1;
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1));
                for (x, y) in &pixels {
                    mask.set((y - y0) * w + (x - x0), true);
                }
                let mut acc = IntensityAccumulator::default();
                acc.add(
                    &mask,
                    [x0, y0, x1, y1].map(|v| v as u32),
                    &[tile(0, 0, 0, case.width, case.height, &case.image)],
                );
                assert_matches_reference(&acc.finish()[&0], &case, label);
                compared += 1;
            }
        }
        assert_eq!(compared, 57);
    }

    #[test]
    fn quantiles_take_the_value_at_n_times_the_fraction() {
        let mut values = [4.0f32, 1.0, 3.0, 2.0];
        let d = Spread::of(&mut values, 2.5).unwrap();
        // n = 4: median at 2 (the third value), quartiles at 1 and 3.
        assert_eq!(
            (d.lower_quartile, d.median, d.upper_quartile),
            (2.0, 3.0, 4.0)
        );
        // Distances from 3: 2, 1, 0, 1 -> sorted 0, 1, 1, 2 -> at 2: 1.
        assert_eq!(d.mad, 1.0);
        assert!((d.std - 1.25f64.sqrt()).abs() < 1e-12);
        // n = 5, at 1.25 between the 2nd and 3rd value.
        let mut values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let d = Spread::of(&mut values, 3.0).unwrap();
        assert_eq!(d.lower_quartile, 2.0 * 0.75 + 3.0 * 0.25);
        assert_eq!(Spread::of(&mut [], 0.0), None);
    }

    /// Sorting (small objects) and selecting (bigger ones) give the same.
    #[test]
    fn selecting_gives_the_same_as_sorting() {
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Few distinct values: ties, as in 8-bit images.
            ((seed >> 33) % 40) as f32 / 255.0
        };
        for n in [33usize, 34, 35, 36, 50, 101, 1000] {
            let values: Vec<f32> = (0..n).map(|_| next()).collect();
            let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
            let selected = Spread::of(&mut values.clone(), mean).unwrap();
            let mut sorted = values.clone();
            sorted.sort_unstable_by(f32::total_cmp);
            let median = quantile(n, 0.5, |i| sorted[i] as f64);
            assert_eq!(selected.median, median, "n {n}");
            assert_eq!(
                selected.lower_quartile,
                quantile(n, 0.25, |i| sorted[i] as f64),
                "n {n}"
            );
            assert_eq!(
                selected.upper_quartile,
                quantile(n, 0.75, |i| sorted[i] as f64),
                "n {n}"
            );
            assert_eq!(selected.mad, median_distance(&sorted, median), "n {n}");
        }
    }

    /// The histogram (big objects of 8- or 16-bit images) gives the same as
    /// sorting; float values fall back to selecting, with the same result.
    #[test]
    fn the_histogram_gives_the_same_as_sorting() {
        let mut seed = 11u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let cases: Vec<Vec<f32>> = vec![
            (0..5000).map(|_| (next() % 60) as f32 / 255.0).collect(),
            (0..100_000)
                .map(|_| (next() % 3000 + 1000) as f32 / 65535.0)
                .collect(),
            (0..200_000)
                .map(|_| (next() % 256) as f32 / 255.0)
                .collect(),
            (0..4000)
                .map(|_| (next() % 1_000_000) as f32 / 1_000_000.0)
                .collect(),
        ];
        for values in cases {
            let n = values.len();
            let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
            let got = Spread::of(&mut values.clone(), mean).unwrap();
            let mut sorted = values.clone();
            sorted.sort_unstable_by(f32::total_cmp);
            let median = quantile(n, 0.5, |i| sorted[i] as f64);
            assert_eq!(got.median, median, "n {n}");
            assert_eq!(
                got.lower_quartile,
                quantile(n, 0.25, |i| sorted[i] as f64),
                "n {n}"
            );
            assert_eq!(
                got.upper_quartile,
                quantile(n, 0.75, |i| sorted[i] as f64),
                "n {n}"
            );
            assert_eq!(got.mad, median_distance(&sorted, median), "n {n}");
        }
    }

    /// Time per pixel of the spread of a big object (release, ignored):
    /// `cargo test --release -p evanalyzer_core --lib bench_spread -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_spread() {
        let mut seed = 3u64;
        let n = 2_000_000;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % 20000 + 5000) as f32 / 65535.0
            })
            .collect();
        for _ in 0..3 {
            let mut v = values.clone();
            let t = std::time::Instant::now();
            let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            std::hint::black_box(Spread::of(&mut v, mean));
            let e = t.elapsed();
            eprintln!(
                "BENCH spread of {n} values: {e:?} ({:.2} ns/value)",
                e.as_nanos() as f64 / n as f64
            );
        }
    }

    /// A strip of `values` (one row) measured through the accumulator.
    fn strip_intensity(values: &[f32]) -> Intensity {
        let n = values.len();
        let mut acc = IntensityAccumulator::default();
        acc.add(
            &BitVec::<u64, Lsb0>::repeat(true, n),
            [0, 0, n as u32 - 1, 0],
            &[tile(0, 0, 0, n, 1, values)],
        );
        acc.finish().shift_remove(&0).unwrap()
    }

    /// Past `LIST_UP_TO` gray values are counted, not listed - the same
    /// spread as from the list (the std up to the order of the sum).
    #[test]
    fn counted_gray_values_give_the_same_spread_as_listed_ones() {
        let mut seed = 7u64;
        let values: Vec<f32> = (0..LIST_UP_TO * 3 + 17)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % 5000 + 300) as f32
            })
            .collect();
        let counted = strip_intensity(&values);
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
        let listed = Spread::of(&mut values.clone(), mean).unwrap();
        assert_eq!(counted.median_intensity, listed.median);
        assert_eq!(counted.lower_quartile_intensity, listed.lower_quartile);
        assert_eq!(counted.upper_quartile_intensity, listed.upper_quartile);
        assert_eq!(counted.mad_intensity, listed.mad);
        assert!((counted.std_intensity - listed.std).abs() <= 1e-12 * listed.std);
        assert_eq!(counted.avg_intensity, mean);
    }

    /// A value that isn't a gray value after counting began: back to the
    /// list, nothing lost.
    #[test]
    fn a_fraction_after_counting_began_goes_back_to_the_list() {
        let mut values: Vec<f32> = (0..LIST_UP_TO + 10).map(|i| (i % 300) as f32).collect();
        values.push(12.5);
        values.extend((0..100).map(|i| i as f32));
        let got = strip_intensity(&values);
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
        let listed = Spread::of(&mut values.clone(), mean).unwrap();
        assert_eq!(got.median_intensity, listed.median);
        assert_eq!(got.lower_quartile_intensity, listed.lower_quartile);
        assert_eq!(got.mad_intensity, listed.mad);
    }

    /// Raw 16-bit gray values (not scaled to 0..1) take the histogram path
    /// too - and give the numbers sorting gives.
    #[test]
    fn the_histogram_works_on_raw_gray_values() {
        let mut values: Vec<f32> = (0..20_000)
            .map(|i| (1000 + (i * 7919) % 3000) as f32)
            .collect();
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
        let histogram = Spread::from_histogram(&values, mean).expect("raw gray values: histogram");
        values.sort_unstable_by(f32::total_cmp);
        let n = values.len();
        let median = quantile(n, 0.5, |i| values[i] as f64);
        assert_eq!(histogram.median, median);
        assert_eq!(
            histogram.lower_quartile,
            quantile(n, 0.25, |i| values[i] as f64)
        );
        assert_eq!(histogram.mad, median_distance(&values, median));
    }
}
