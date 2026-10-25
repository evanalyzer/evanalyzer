//! # Colocalization
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How two channels go together - CellProfiler's MeasureColocalization
//! (cellprofiler/modules/measurecolocalization.py, CellProfiler 4.2.8),
//! value for value: Pearson and the slope, Manders M1/M2, the overlap
//! coefficient with K1/K2 and the rank-weighted colocalization (RWC) above a
//! percentage of each channel's maximum, and Manders M1/M2 above Costes'
//! automatic thresholds - over the whole image (Measure Image
//! Colocalization) or per object of a class (Measure Object Colocalization).
//! The rules CellProfiler applies differently there are kept: over the
//! image a pixel counts when it is *above* the threshold, in an object when
//! it is *at or above* it; RWC ranks the values of the whole image or of
//! all the objects; an object's Costes values use the whole image's Costes
//! thresholds (Measure Object Colocalization searches for them itself).

use crate::algos::GlobalPipelineCache;
use crate::object::Object;
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use crate::pipeline::pipeline_measure::Outputs;
use crate::{ImageTile, image::ImageContainer};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, MeasurementUnit,
    max_gray_value,
};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::sync::Arc;

/// How Costes' automatic thresholds are searched for (CellProfiler's
/// "Method for Costes thresholding").
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum CostesMethod {
    /// No Costes values.
    #[cmdsmeta(display_name = "Off")]
    Off,
    /// A shrinking window of candidates - a few passes over the image, also
    /// for 16-bit images (CellProfiler's "Faster").
    #[cmdsmeta(display_name = "Faster")]
    Faster,
    /// Candidates from the top down, skipping while far off (CellProfiler's
    /// "Fast" - slow on 16-bit images).
    #[cmdsmeta(display_name = "Fast")]
    Fast,
    /// Every candidate from the top down (CellProfiler's "Accurate" - very
    /// slow on 16-bit images).
    #[cmdsmeta(display_name = "Accurate")]
    Accurate,
}

/// A value: (id, name, unit, range).
pub(super) type Value = (
    &'static str,
    &'static str,
    MeasurementUnit,
    Option<(f64, f64)>,
);

/// The values, in the order of the outputs (per object without the slope).
static VALUES: [Value; 11] = [
    (
        "pearson",
        "Pearson",
        MeasurementUnit::None,
        Some((-1.0, 1.0)),
    ),
    ("slope", "Slope", MeasurementUnit::None, None),
    (
        "manders_m1",
        "Manders M1",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    (
        "manders_m2",
        "Manders M2",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    (
        "overlap",
        "Overlap",
        MeasurementUnit::None,
        Some((0.0, 1.0)),
    ),
    ("k1", "K1", MeasurementUnit::None, None),
    ("k2", "K2", MeasurementUnit::None, None),
    (
        "rwc_1",
        "RWC 1",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    (
        "rwc_2",
        "RWC 2",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    (
        "costes_m1",
        "Manders M1 (Costes)",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
    (
        "costes_m2",
        "Manders M2 (Costes)",
        MeasurementUnit::Fraction,
        Some((0.0, 1.0)),
    ),
];

/// The two Costes values come last in [`VALUES`].
const COSTES_VALUES: usize = 2;

/// The slope (in [`VALUES`]): only over the whole image, as in CellProfiler.
pub(super) const SLOPE: usize = 1;

/// One value per entry of [`VALUES`] (NaN: none).
pub(super) type Values = [f64; VALUES.len()];

pub(super) const CITATIONS: [&CitationMetadata; 4] = [
    &CitationMetadata::CELLPROFILER,
    &MANDERS_1993,
    &COSTES_2004,
    &SINGAN_2011,
];

/// The settings Measure Image Colocalization and Measure Object
/// Colocalization share.
#[derive(Clone, Copy)]
pub(super) struct Coloc {
    pub(super) channel_a: ImageChannelIdx,
    pub(super) channel_b: ImageChannelIdx,
    pub(super) threshold_percent: f32,
    pub(super) costes: CostesMethod,
}

impl Coloc {
    /// The image's maximum gray value.
    pub(super) fn scale(&self, cache: &GlobalPipelineCache) -> Result<f64, InternalErrors> {
        max_gray_value(cache.image_meta.nr_of_bits).ok_or_else(|| {
            InternalErrors::InvalidArgument(format!(
                "Colocalization: implausible bit depth {}",
                cache.image_meta.nr_of_bits
            ))
        })
    }

    /// The entries of [`VALUES`] measured: without the Costes ones when off.
    pub(super) fn values(&self) -> impl Iterator<Item = (usize, &'static Value)> {
        let count = match self.costes {
            CostesMethod::Off => VALUES.len() - COSTES_VALUES,
            _ => VALUES.len(),
        };
        VALUES[..count].iter().enumerate()
    }

    /// Every output of the step `step_id`: of the image, or of the objects
    /// (without the slope).
    pub(super) fn outputs(&self, step_id: &str, per_object: bool) -> Vec<MeasurementInfo> {
        let about = format!("channel {} × channel {}", self.channel_a, self.channel_b);
        let outputs = Outputs::of(step_id).about(about);
        self.values()
            .filter(|(i, _)| !per_object || *i != SLOPE)
            .map(|(_, (id, name, unit, range))| {
                let info = if per_object {
                    outputs.object(id, name, *unit)
                } else {
                    outputs.image(id, name, *unit)
                };
                match range {
                    Some((min, max)) => info.range(*min, *max),
                    None => info,
                }
            })
            .collect()
    }

    /// The pixels of channels A and B of every tile of the image.
    pub(super) fn image_tiles(
        &self,
        cache: &GlobalPipelineCache,
    ) -> Result<Vec<TilePair>, InternalErrors> {
        let (a, b) = (self.channel_a.0 as i32, self.channel_b.0 as i32);
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(channel, _)| *channel == a)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(InternalErrors::InvalidArgument(format!(
                "Colocalization: the image has no channel {a}"
            )));
        }
        tiles
            .into_iter()
            .map(|tile| {
                let channel = |channel: i32| {
                    cache
                        .image_cache
                        .get(&CacheAddress::Channel((channel, tile.clone())))
                        .ok_or_else(|| {
                            InternalErrors::InvalidArgument(format!(
                                "Colocalization: the image has no channel {channel}"
                            ))
                        })
                };
                Ok(TilePair {
                    a: channel(a)?,
                    b: channel(b)?,
                })
            })
            .collect()
    }

    /// The (A, B) values of `object`'s pixels the loaded tiles cover - each
    /// from the first tile of its channel covering it.
    pub(super) fn object_pixels(
        &self,
        object: &Object,
        cache: &GlobalPipelineCache,
    ) -> Vec<(f32, f32)> {
        let (a, b) = (self.channel_a.0 as i32, self.channel_b.0 as i32);
        let views = cache.resolve_channel_views_for_bbox(object.bbox);
        let tiles = |channel: i32| -> Vec<(ImageTile, bool, &[f32])> {
            views
                .iter()
                .filter(|((c, _), _, _)| *c == channel)
                .filter_map(|((_, tile), rgb, container)| {
                    Some((*tile, *rgb, channel_pixel_slice(container)?))
                })
                .collect()
        };
        let (tiles_a, tiles_b) = (tiles(a), tiles(b));
        let sample = |tiles: &[(ImageTile, bool, &[f32])], x: usize, y: usize| {
            tiles.iter().find_map(|(t, rgb, samples)| {
                let inside = x >= t.offset_x
                    && y >= t.offset_y
                    && x < t.offset_x + t.width
                    && y < t.offset_y + t.height;
                inside.then(|| {
                    sample_channel_pixel(*rgb, samples, (y - t.offset_y) * t.width + x - t.offset_x)
                })
            })
        };
        let [x0, y0, x1, y1] = object.bbox.map(|v| v as usize);
        let width = x1 - x0 + 1;
        let mut pixels = Vec::with_capacity(object.area);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if !object
                    .mask_data
                    .get((y - y0) * width + (x - x0))
                    .is_some_and(|b| *b)
                {
                    continue;
                }
                if let (Some(va), Some(vb)) = (sample(&tiles_a, x, y), sample(&tiles_b, x, y)) {
                    pixels.push((va, vb));
                }
            }
        }
        pixels
    }
}

/// The same tile of channels A and B.
pub(super) struct TilePair {
    a: Arc<ImageContainer>,
    b: Arc<ImageContainer>,
}

impl TilePair {
    /// Calls `f` with every pixel's (A, B) value.
    fn for_each(&self, mut f: impl FnMut(f64, f64)) {
        fn pixels(c: &ImageContainer) -> (bool, &[f32], usize) {
            let rgb = matches!(c, ImageContainer::F32Rgb(_));
            let slice = channel_pixel_slice(c).unwrap_or_default();
            let len = if rgb { slice.len() / 3 } else { slice.len() };
            (rgb, slice, len)
        }
        let ((rgb_a, xs, len_a), (rgb_b, ys, len_b)) = (pixels(&self.a), pixels(&self.b));
        for i in 0..len_a.min(len_b) {
            f(
                sample_channel_pixel(rgb_a, xs, i) as f64,
                sample_channel_pixel(rgb_b, ys, i) as f64,
            );
        }
    }
}

/// A pass over every pixel of `tiles`, tile by tile in parallel: each tile
/// folds into a `T` of its own, then they are merged.
fn pass<T: Send>(
    tiles: &[TilePair],
    init: impl Fn() -> T + Sync,
    fold: impl Fn(&mut T, f64, f64) + Sync,
    merge: impl Fn(T, T) -> T + Sync + Send,
) -> T {
    tiles
        .par_iter()
        .map(|tile| {
            let mut acc = init();
            tile.for_each(|x, y| fold(&mut acc, x, y));
            acc
        })
        .reduce(&init, merge)
}

/// The whole image's values and the Costes thresholds found (NaN: off).
pub(super) struct ImageResult {
    pub(super) values: Values,
    costes_thresholds: (f64, f64),
}

/// Sums over pixel pairs, added up tile by tile.
#[derive(Clone, Copy, Default)]
struct Sums {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Sums {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.yy += y * y;
        self.xy += x * y;
    }

    fn merge(a: Self, b: Self) -> Self {
        Self {
            n: a.n + b.n,
            x: a.x + b.x,
            y: a.y + b.y,
            xx: a.xx + b.xx,
            yy: a.yy + b.yy,
            xy: a.xy + b.xy,
        }
    }
}

/// The whole image's values (CellProfiler's `run_image_pair_images`).
pub(super) fn image_values(
    tiles: &[TilePair],
    threshold_percent: f64,
    costes: CostesMethod,
    scale: f64,
) -> ImageResult {
    let mut values = [f64::NAN; VALUES.len()];
    // Count, means, maxima.
    let (sums, max_x, max_y) = pass(
        tiles,
        || (Sums::default(), f64::MIN, f64::MIN),
        |(s, mx, my), x, y| {
            s.add(x, y);
            *mx = mx.max(x);
            *my = my.max(y);
        },
        |(a, ax, ay), (b, bx, by)| (Sums::merge(a, b), ax.max(bx), ay.max(by)),
    );
    if sums.n == 0.0 {
        return ImageResult {
            values,
            costes_thresholds: (f64::NAN, f64::NAN),
        };
    }
    let (mean_x, mean_y) = (sums.x / sums.n, sums.y / sums.n);
    // Deviations: Pearson (numpy.corrcoef) and the slope (least squares
    // A·x + B = y).
    let (m2x, m2y, co) = pass(
        tiles,
        || (0.0, 0.0, 0.0),
        |(m2x, m2y, co), x, y| {
            let (dx, dy) = (x - mean_x, y - mean_y);
            *m2x += dx * dx;
            *m2y += dy * dy;
            *co += dx * dy;
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    );
    values[0] = (co / (m2x * m2y).sqrt()).clamp(-1.0, 1.0);
    values[1] = if m2x > 0.0 {
        co / m2x
    } else {
        // A constant channel A: the least-squares solution of least norm.
        mean_x * mean_y / (mean_x * mean_x + 1.0)
    };

    // Above the thresholds: a percentage of each channel's maximum.
    let (thr_x, thr_y) = (
        threshold_percent * max_x / 100.0,
        threshold_percent * max_y / 100.0,
    );
    let ranks = (
        DenseRanks::of(tiles, |x, _| x),
        DenseRanks::of(tiles, |_, y| y),
    );
    let r = ranks.0.count().max(ranks.1.count()) as f64;
    let (tot_x, tot_y, comb, rwc_x, rwc_y) = pass(
        tiles,
        || (0.0, 0.0, Sums::default(), 0.0, 0.0),
        |(tx, ty, comb, wx, wy), x, y| {
            let (above_x, above_y) = (x > thr_x, y > thr_y);
            if above_x {
                *tx += x;
            }
            if above_y {
                *ty += y;
            }
            if above_x && above_y {
                comb.add(x, y);
                let d = (ranks.0.rank(x) as f64 - ranks.1.rank(y) as f64).abs();
                let weight = (r - d) / r;
                *wx += x * weight;
                *wy += y * weight;
            }
        },
        |a, b| {
            (
                a.0 + b.0,
                a.1 + b.1,
                Sums::merge(a.2, b.2),
                a.3 + b.3,
                a.4 + b.4,
            )
        },
    );
    values[2] = comb.x / tot_x;
    values[3] = comb.y / tot_y;
    values[4] = comb.xy / (comb.xx * comb.yy).sqrt();
    values[5] = comb.xy / comb.xx;
    values[6] = comb.xy / comb.yy;
    values[7] = rwc_x / tot_x;
    values[8] = rwc_y / tot_y;

    let costes_thresholds = costes_thresholds_with(tiles, costes, scale, (max_x, max_y));
    if costes != CostesMethod::Off {
        let (cx, cy) = costes_thresholds;
        let (tot_x, tot_y, comb) = pass(
            tiles,
            || (0.0, 0.0, Sums::default()),
            |(tx, ty, comb), x, y| {
                if x > cx {
                    *tx += x;
                }
                if y > cy {
                    *ty += y;
                }
                if x > cx && y > cy {
                    comb.add(x, y);
                }
            },
            |a, b| (a.0 + b.0, a.1 + b.1, Sums::merge(a.2, b.2)),
        );
        values[9] = comb.x / tot_x;
        values[10] = comb.y / tot_y;
    }
    ImageResult {
        values,
        costes_thresholds,
    }
}

/// Costes' thresholds of channels A and B over the whole image (NaN: off).
pub(super) fn costes_thresholds(
    tiles: &[TilePair],
    costes: CostesMethod,
    scale: f64,
) -> (f64, f64) {
    let max = match costes {
        CostesMethod::Fast | CostesMethod::Accurate => pass(
            tiles,
            || (f64::MIN, f64::MIN),
            |(mx, my), x, y| {
                *mx = mx.max(x);
                *my = my.max(y);
            },
            |(ax, ay), (bx, by)| (ax.max(bx), ay.max(by)),
        ),
        CostesMethod::Off | CostesMethod::Faster => (f64::NAN, f64::NAN),
    };
    costes_thresholds_with(tiles, costes, scale, max)
}

/// [`costes_thresholds`] with the channels' maxima (`max`) known.
fn costes_thresholds_with(
    tiles: &[TilePair],
    costes: CostesMethod,
    scale: f64,
    max: (f64, f64),
) -> (f64, f64) {
    match costes {
        CostesMethod::Off => (f64::NAN, f64::NAN),
        CostesMethod::Faster => bisection_costes(tiles, scale),
        CostesMethod::Fast | CostesMethod::Accurate => {
            linear_costes(tiles, scale, costes == CostesMethod::Accurate, max)
        }
    }
}

/// Every value's rank among the different values of a channel (0 for the
/// smallest), as CellProfiler's RWC numbers them.
enum DenseRanks {
    /// Through 65535 levels (exact for 8/16-bit values: each level holds
    /// one value): the rank of each level.
    Levels(Vec<u32>, u32),
    /// The different values, in order.
    Sorted(Vec<f32>),
}

/// Values in 0..=1 on 65535 levels (65535 = 255 · 257).
const LEVELS: f32 = 65535.0;

fn level(v: f32) -> Option<usize> {
    (0.0..=1.0)
        .contains(&v)
        .then(|| (v * LEVELS + 0.5) as usize)
}

impl DenseRanks {
    /// Over every pixel of `tiles`, the channel `pick` chooses.
    fn of(tiles: &[TilePair], pick: impl Fn(f64, f64) -> f64 + Sync) -> Self {
        // Per level its value; a second value in a level: not levels.
        let levels = pass(
            tiles,
            || Some(vec![f32::NAN; LEVELS as usize + 1]),
            |values, x, y| {
                let Some(level_values) = values else { return };
                let v = pick(x, y) as f32;
                match level(v) {
                    Some(l) if level_values[l].is_nan() => level_values[l] = v,
                    Some(l) if level_values[l] == v => {}
                    _ => *values = None,
                }
            },
            |a, b| match (a, b) {
                (Some(mut a), Some(b)) => {
                    for (x, y) in a.iter_mut().zip(b) {
                        if x.is_nan() {
                            *x = y;
                        } else if !y.is_nan() && *x != y {
                            return None;
                        }
                    }
                    Some(a)
                }
                _ => None,
            },
        );
        match levels {
            Some(values) => {
                let mut ranks = vec![0u32; values.len()];
                let mut next = 0u32;
                for (l, v) in values.iter().enumerate() {
                    if !v.is_nan() {
                        ranks[l] = next;
                        next += 1;
                    }
                }
                Self::Levels(ranks, next)
            }
            None => {
                let mut all: Vec<f32> = Vec::new();
                for tile in tiles {
                    tile.for_each(|x, y| all.push(pick(x, y) as f32));
                }
                Self::from_values(all)
            }
        }
    }

    fn from_values(mut values: Vec<f32>) -> Self {
        values.sort_unstable_by(f32::total_cmp);
        values.dedup();
        Self::Sorted(values)
    }

    fn rank(&self, v: f64) -> u32 {
        let v = v as f32;
        match self {
            Self::Levels(ranks, _) => level(v).map_or(0, |l| ranks[l]),
            Self::Sorted(values) => values.partition_point(|x| x.total_cmp(&v).is_lt()) as u32,
        }
    }

    /// How many different values there are.
    fn count(&self) -> u32 {
        match self {
            Self::Levels(_, count) => *count,
            Self::Sorted(values) => values.len() as u32,
        }
    }
}

/// Pearson of the pixels where A < `tx` or B < `ty` (Costes' "combt"),
/// as `scipy.stats.pearsonr`: `Err` for fewer than two pixels (its
/// ValueError), NaN for a constant channel; with the count.
fn subset_pearson(tiles: &[TilePair], tx: f64, ty: f64) -> (usize, Result<f64, ()>) {
    let inside = |x: f64, y: f64| x < tx || y < ty;
    let sums = pass(
        tiles,
        Sums::default,
        |s, x, y| {
            if inside(x, y) {
                s.add(x, y);
            }
        },
        Sums::merge,
    );
    let n = sums.n as usize;
    if n < 2 {
        return (n, Err(()));
    }
    (n, Ok(subset_pearson_with(tiles, inside, sums)))
}

/// Pearson of the pixels `inside` picks, whose sums are `sums`.
fn subset_pearson_with(
    tiles: &[TilePair],
    inside: impl Fn(f64, f64) -> bool + Sync,
    sums: Sums,
) -> f64 {
    let (mx, my) = (sums.x / sums.n, sums.y / sums.n);
    let (m2x, m2y, co) = pass(
        tiles,
        || (0.0, 0.0, 0.0),
        |(m2x, m2y, co), x, y| {
            if inside(x, y) {
                let (dx, dy) = (x - mx, y - my);
                *m2x += dx * dx;
                *m2y += dy * dy;
                *co += dx * dy;
            }
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    );
    if m2x == 0.0 || m2y == 0.0 {
        return f64::NAN;
    }
    (co / (m2x.sqrt() * m2y.sqrt())).clamp(-1.0, 1.0)
}

/// The orthogonal regression line y = a·x + b through the pixels where
/// either channel isn't 0 (Costes' candidates lie on it).
fn costes_line(tiles: &[TilePair]) -> (f64, f64) {
    let nonzero = |x: f64, y: f64| x > 0.0 || y > 0.0;
    let sums = pass(
        tiles,
        Sums::default,
        |s, x, y| {
            if nonzero(x, y) {
                s.add(x, y);
            }
        },
        Sums::merge,
    );
    let (mx, my) = (sums.x / sums.n, sums.y / sums.n);
    let mz = (sums.x + sums.y) / sums.n;
    let (vx, vy, vz) = pass(
        tiles,
        || (0.0, 0.0, 0.0),
        |(vx, vy, vz), x, y| {
            if nonzero(x, y) {
                *vx += (x - mx) * (x - mx);
                *vy += (y - my) * (y - my);
                *vz += (x + y - mz) * (x + y - mz);
            }
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    );
    // numpy.var with ddof=1.
    let (xvar, yvar, zvar) = (
        vx / (sums.n - 1.0),
        vy / (sums.n - 1.0),
        vz / (sums.n - 1.0),
    );
    let covar = 0.5 * (zvar - (xvar + yvar));
    let denom = 2.0 * covar;
    let num = (yvar - xvar) + ((yvar - xvar) * (yvar - xvar) + 4.0 * (covar * covar)).sqrt();
    let a = num / denom;
    (a, my - a * mx)
}

/// Python's float `//`, as CellProfiler's Costes code computes with it.
fn py_floordiv(x: f64, y: f64) -> f64 {
    let mut m = x % y;
    let mut div = (x - m) / y;
    if m != 0.0 && ((y < 0.0) != (m < 0.0)) {
        m += y;
        div -= 1.0;
    }
    let _ = m;
    if div != 0.0 {
        let mut floor = div.floor();
        if div - floor > 0.5 {
            floor += 1.0;
        }
        floor
    } else {
        0.0f64.copysign(x / y)
    }
}

/// CellProfiler's `bisection_costes` ("Faster"): candidates in a window
/// that shrinks around the highest threshold whose remaining pixels still
/// correlate positively.
fn bisection_costes(tiles: &[TilePair], scale_max: f64) -> (f64, f64) {
    let (a, b) = costes_line(tiles);
    let mut left = 1.0;
    let mut right = scale_max;
    let mut mid = py_floordiv(right - left, 6.0 / 5.0) + left;
    let mut lastmid = 0.0;
    let mut valid = 1.0;
    while lastmid != mid {
        let thr_x = mid / scale_max;
        let thr_y = a * thr_x + b;
        let (count, r) = subset_pearson(tiles, thr_x, thr_y);
        if count <= 2 {
            left = mid - 1.0;
        } else {
            match r {
                Err(()) => left = mid - 1.0,
                Ok(r) if r < 0.0 => left = mid - 1.0,
                Ok(r) if r >= 0.0 => {
                    right = mid + 1.0;
                    valid = mid;
                }
                Ok(_) => {} // NaN: neither
            }
        }
        lastmid = mid;
        mid = if right - left > 6.0 {
            py_floordiv(right - left, 6.0 / 5.0) + left
        } else {
            py_floordiv(right - left, 2.0) + left
        };
    }
    let thr_x = (valid - 1.0) / scale_max;
    (thr_x, a * thr_x + b)
}

/// CellProfiler's `linear_costes` ("Fast" / "Accurate"): candidates from
/// the top down until the remaining pixels stop correlating positively.
fn linear_costes(
    tiles: &[TilePair],
    scale_max: f64,
    accurate: bool,
    (max_x, max_y): (f64, f64),
) -> (f64, f64) {
    let i_step = 1.0 / scale_max;
    let (a, b) = costes_line(tiles);
    let img_max = max_x.max(max_y);
    let mut i = i_step * (py_floordiv(img_max, i_step) + 1.0);
    let mut num_true: Option<usize> = None;
    // Without a threshold.
    let all = pass(tiles, Sums::default, |s, x, y| s.add(x, y), Sums::merge);
    let mut cost_reg = subset_pearson_with(tiles, |_, _| true, all);
    let (mut thr_x, mut thr_y) = (i, a * i + b);
    while i > max_x && a * i + b > max_y {
        i -= i_step;
    }
    while i > i_step {
        thr_x = i;
        thr_y = a * i + b;
        let inside = |x: f64, y: f64| x < thr_x || y < thr_y;
        let sums = pass(
            tiles,
            Sums::default,
            |s, x, y| {
                if inside(x, y) {
                    s.add(x, y);
                }
            },
            Sums::merge,
        );
        let positives = sums.n as usize;
        if Some(positives) != num_true {
            if positives < 2 {
                break; // pearsonr's ValueError
            }
            cost_reg = subset_pearson_with(tiles, inside, sums);
            num_true = Some(positives);
        }
        if cost_reg <= 0.0 {
            break;
        } else if accurate || i < i_step * 10.0 {
            i -= i_step;
        } else if cost_reg > 0.45 {
            i -= i_step * 10.0;
        } else if cost_reg > 0.35 {
            i -= i_step * 5.0;
        } else if cost_reg > 0.25 {
            i -= i_step * 2.0;
        } else {
            i -= i_step;
        }
    }
    (thr_x, thr_y)
}

/// Each object's values (CellProfiler's `run_image_pair_objects`) from its
/// (A, B) pixels; `costes` the whole image's Costes thresholds (NaN: off).
pub(super) fn object_values(
    objects: &[&[(f32, f32)]],
    threshold_percent: f64,
    costes: (f64, f64),
) -> Vec<Values> {
    let mut out = vec![[f64::NAN; VALUES.len()]; objects.len()];
    // Thresholds per object, which pixels count: decided first - several
    // values are 0 for every object when no pixel of any object counts.
    let thresholds: Vec<(f64, f64)> = objects
        .iter()
        .map(|pixels| {
            let max = |pick: fn(&(f32, f32)) -> f32| {
                pixels.iter().map(pick).fold(f32::MIN, f32::max) as f64
            };
            (
                (threshold_percent / 100.0) * max(|p| p.0),
                (threshold_percent / 100.0) * max(|p| p.1),
            )
        })
        .collect();
    let counts = |p: &(f32, f32), (tx, ty): (f64, f64)| p.0 as f64 >= tx && p.1 as f64 >= ty;
    let any_combined = objects
        .iter()
        .zip(&thresholds)
        .any(|(pixels, thr)| pixels.iter().any(|p| counts(p, *thr)));
    // RWC ranks over the pixels of all the objects.
    let ranks = (
        DenseRanks::from_values(objects.iter().flat_map(|p| p.iter().map(|v| v.0)).collect()),
        DenseRanks::from_values(objects.iter().flat_map(|p| p.iter().map(|v| v.1)).collect()),
    );
    let r = ranks.0.count().max(ranks.1.count()) as f64;
    let (cx, cy) = costes;
    let costes_any_x = objects.iter().any(|p| p.iter().any(|v| v.0 as f64 > cx));
    let costes_any_y = objects.iter().any(|p| p.iter().any(|v| v.1 as f64 > cy));
    let costes_any = objects
        .iter()
        .any(|p| p.iter().any(|v| v.0 as f64 > cx && v.1 as f64 > cy));

    for ((pixels, &(tx, ty)), values) in objects.iter().zip(&thresholds).zip(&mut out) {
        if pixels.is_empty() {
            continue;
        }
        // Pearson: sum of the products of the deviations over the
        // product of the square roots of their sums of squares.
        let n = pixels.len() as f64;
        let mean_x = pixels.iter().map(|p| p.0 as f64).sum::<f64>() / n;
        let mean_y = pixels.iter().map(|p| p.1 as f64).sum::<f64>() / n;
        let std_x = pixels
            .iter()
            .map(|p| (p.0 as f64 - mean_x).powi(2))
            .sum::<f64>()
            .sqrt();
        let std_y = pixels
            .iter()
            .map(|p| (p.1 as f64 - mean_y).powi(2))
            .sum::<f64>()
            .sqrt();
        values[0] = pixels
            .iter()
            .map(|p| (p.0 as f64 - mean_x) * (p.1 as f64 - mean_y) / (std_x * std_y))
            .sum();

        let (mut tot_x, mut tot_y) = (0.0, 0.0);
        let mut comb = Sums::default();
        let (mut wx, mut wy) = (0.0, 0.0);
        for p in pixels.iter() {
            let (x, y) = (p.0 as f64, p.1 as f64);
            if x >= tx {
                tot_x += x;
            }
            if y >= ty {
                tot_y += y;
            }
            if x >= tx && y >= ty {
                comb.add(x, y);
                let d = (ranks.0.rank(x) as f64 - ranks.1.rank(y) as f64).abs();
                let weight = (r - d) / r;
                wx += x * weight;
                wy += y * weight;
            }
        }
        if any_combined {
            values[2] = comb.x / tot_x;
            values[3] = comb.y / tot_y;
            values[4] = comb.xy / (comb.xx * comb.yy).sqrt();
            values[5] = comb.xy / comb.xx;
            values[6] = comb.xy / comb.yy;
            values[7] = wx / tot_x;
            values[8] = wy / tot_y;
        } else {
            values[2..=8].fill(0.0);
        }

        if !cx.is_nan() {
            // Above the whole image's Costes thresholds - the totals count
            // a pixel at the threshold too, as CellProfiler does.
            let (mut tot_x, mut tot_y) = (0.0, 0.0);
            let mut comb = Sums::default();
            for p in pixels.iter() {
                let (x, y) = (p.0 as f64, p.1 as f64);
                if x >= cx {
                    tot_x += x;
                }
                if y >= cy {
                    tot_y += y;
                }
                if x > cx && y > cy {
                    comb.add(x, y);
                }
            }
            if !costes_any_x {
                tot_x = 0.0;
            }
            if !costes_any_y {
                tot_y = 0.0;
            }
            if costes_any {
                values[9] = comb.x / tot_x;
                values[10] = comb.y / tot_y;
            } else {
                values[9] = 0.0;
                values[10] = 0.0;
            }
        }
    }
    out
}

/// Manders' coefficients.
const MANDERS_1993: CitationMetadata = CitationMetadata {
    cite_key: "manders1993measurement",
    title: "Measurement of co-localization of objects in dual-colour confocal images",
    authors: &["E. M. M. Manders", "F. J. Verbeek", "J. A. Aten"],
    year: 1993,
    container: Some("Journal of Microscopy"),
    doi: Some("10.1111/j.1365-2818.1993.tb03313.x"),
    url: Some("https://doi.org/10.1111/j.1365-2818.1993.tb03313.x"),
    pages: Some("375-382"),
};

/// Costes' automatic thresholds.
const COSTES_2004: CitationMetadata = CitationMetadata {
    cite_key: "costes2004automatic",
    title: "Automatic and quantitative measurement of protein-protein colocalization in live cells",
    authors: &[
        "Sylvain V. Costes",
        "Dirk Daelemans",
        "Edward H. Cho",
        "Zachary Dobbin",
        "George Pavlakis",
        "Stephen Lockett",
    ],
    year: 2004,
    container: Some("Biophysical Journal"),
    doi: Some("10.1529/biophysj.103.038422"),
    url: Some("https://doi.org/10.1529/biophysj.103.038422"),
    pages: Some("3993-4003"),
};

/// Rank-weighted colocalization (RWC).
const SINGAN_2011: CitationMetadata = CitationMetadata {
    cite_key: "singan2011dual",
    title: "Dual channel rank-based intensity weighting for quantitative co-localization of microscopy images",
    authors: &[
        "Vasanth R. Singan",
        "Thouis R. Jones",
        "Kathleen M. Curran",
        "Jeremy C. Simpson",
    ],
    year: 2011,
    container: Some("BMC Bioinformatics"),
    doi: Some("10.1186/1471-2105-12-407"),
    url: Some("https://doi.org/10.1186/1471-2105-12-407"),
    pages: Some("407"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::{
        ImageAlgorithm, MeasureImageColocalization, MeasureObjectColocalization, PipelineContext,
    };
    use crate::image::{ManagedImage, Point2d};
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{MeasurementTarget, ObjectClass, ObjectId};
    use kornia_image::{Image, ImageSize};

    const CELL: ObjectClass = ObjectClass::Global(1);

    /// One case of `tests/colocalization_cellprofiler.json`.
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        width: usize,
        height: usize,
        threshold_percent: f32,
        scale: u32,
        costes: String,
        labels: Vec<u32>,
        a: Vec<f32>,
        b: Vec<f32>,
        image: Vec<Option<f64>>,
        objects: Vec<Vec<Option<f64>>>,
    }

    /// The image split into a left and a right tile.
    fn cache_of(case: &Case) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = if case.scale == 255 { 8 } else { 16 };
        let split = case.width / 2;
        for (channel, values) in [(0, &case.a), (1, &case.b)] {
            for (x0, x1) in [(0, split), (split, case.width)] {
                let samples: Vec<f32> = (0..case.height)
                    .flat_map(|y| (x0..x1).map(move |x| values[y * case.width + x]))
                    .collect();
                let size = ImageSize {
                    width: x1 - x0,
                    height: case.height,
                };
                let image = ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::from_size_slice(size, &samples).unwrap(),
                    tile_offset: Point2d { x: x0, y: 0 },
                    plane: None,
                });
                let tile = ImageTile {
                    offset_x: x0,
                    offset_y: 0,
                    width: x1 - x0,
                    height: case.height,
                };
                cache.add_to_channel_cache(Arc::new(image), channel, tile);
            }
        }
        let max = case.labels.iter().copied().max().unwrap_or(0);
        for label in 1..=max {
            let pixels: Vec<(u32, u32)> = (0..case.labels.len())
                .filter(|&p| case.labels[p] == label)
                .map(|p| ((p % case.width) as u32, (p / case.width) as u32))
                .collect();
            let x0 = pixels.iter().map(|p| p.0).min().unwrap();
            let y0 = pixels.iter().map(|p| p.1).min().unwrap();
            let x1 = pixels.iter().map(|p| p.0).max().unwrap();
            let y1 = pixels.iter().map(|p| p.1).max().unwrap();
            let w = (x1 - x0 + 1) as usize;
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
            for (x, y) in &pixels {
                mask.set((y - y0) as usize * w + (x - x0) as usize, true);
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId(label as u128),
                bbox: [x0, y0, x1, y1],
                area: pixels.len(),
                mask_data: mask,
                ..Default::default()
            });
            object.add_object_class(CELL, Scope::Global).unwrap();
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn close(got: Option<f64>, expected: Option<f64>, what: &str) {
        match (got, expected) {
            (Some(g), Some(e)) => {
                assert!(
                    (g - e).abs() <= 1e-9 * e.abs().max(1.0),
                    "{what}: {g} vs {e}"
                )
            }
            (None, None) => {}
            _ => panic!("{what}: {got:?} vs {expected:?}"),
        }
    }

    /// Every value over the image and per object as CellProfiler's
    /// MeasureColocalization gives it (fixture from
    /// `tests/colocalization_cellprofiler.py`), with each Costes method.
    #[test]
    fn values_are_cellprofilers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/colocalization_cellprofiler.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(cases.len(), 6);
        for case in &cases {
            let mut cache = cache_of(case);
            let costes = match case.costes.as_str() {
                "Faster" => CostesMethod::Faster,
                "Fast" => CostesMethod::Fast,
                _ => CostesMethod::Accurate,
            };
            let mut ctx = PipelineContext::without_image(
                std::path::PathBuf::new(),
                PipelineImageMeta::default(),
            )
            .unwrap();
            let image = MeasureImageColocalization {
                id: "coloc_01".into(),
                channel_a: ImageChannelIdx(0),
                channel_b: ImageChannelIdx(1),
                threshold_percent: case.threshold_percent,
                costes,
            };
            image.execute(&mut ctx, &mut cache).unwrap();
            let objects = MeasureObjectColocalization {
                id: "object_coloc_01".into(),
                objects: CELL,
                channel_a: ImageChannelIdx(0),
                channel_b: ImageChannelIdx(1),
                threshold_percent: case.threshold_percent,
                costes,
            };
            objects.execute(&mut ctx, &mut cache).unwrap();
            let measured = &cache.measurements.measurements;
            for (i, (id, ..)) in VALUES.iter().enumerate() {
                let key = format!("coloc_01/{id}");
                let m = measured.iter().find(|m| m.info.key == key).unwrap();
                let MeasurementValues::Image(values) = &m.values else {
                    panic!("{key}: per image")
                };
                close(
                    values.first().map(|v| v.1),
                    case.image[i],
                    &format!("{}: {key}", case.name),
                );
                if i == SLOPE {
                    continue;
                }
                let key = format!("object_coloc_01/{id}");
                let m = measured.iter().find(|m| m.info.key == key).unwrap();
                let MeasurementValues::Objects(values) = &m.values else {
                    panic!("{key}: per object")
                };
                assert_eq!(values.len(), case.objects.len());
                for v in values {
                    let label = v.object.0 as usize;
                    close(
                        v.value.is_finite().then_some(v.value),
                        case.objects[label - 1][i],
                        &format!("{}: {key} object {label}", case.name),
                    );
                }
            }
        }
    }

    /// Without Costes: no Costes values; per object no slope.
    #[test]
    fn outputs_follow_the_settings() {
        let step = MeasureImageColocalization {
            id: "c".into(),
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes: CostesMethod::Off,
        };
        let keys: Vec<String> = step
            .measurement_outputs()
            .into_iter()
            .map(|o| o.key)
            .collect();
        assert_eq!(keys.len(), 9);
        assert!(!keys.iter().any(|k| k.contains("costes")));
        let with_costes = MeasureImageColocalization {
            costes: CostesMethod::Faster,
            ..step
        };
        assert_eq!(with_costes.measurement_outputs().len(), 11);
        let objects = MeasureObjectColocalization {
            id: "o".into(),
            objects: CELL,
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes: CostesMethod::Faster,
        };
        let outputs = objects.measurement_outputs();
        // 10 per object: no slope.
        assert_eq!(outputs.len(), 10);
        assert!(!outputs.iter().any(|o| o.key == "o/slope"));
        assert!(
            outputs
                .iter()
                .all(|o| o.target == MeasurementTarget::Object)
        );
    }

    #[test]
    fn python_floor_division() {
        assert_eq!(py_floordiv(254.0, 1.2), 211.0);
        assert_eq!(py_floordiv(7.0, 2.0), 3.0);
        assert_eq!(py_floordiv(-7.0, 2.0), -4.0);
        assert_eq!(py_floordiv(0.9, 1.0 / 255.0), 229.0);
    }

    /// Speed and peak RAM on a `BENCH_SIZE`² 16-bit image (default 2048)
    /// in 4 tiles, a 9 × 9 object every 40 px, `BENCH_COSTES` (faster |
    /// fast | off) - release, one per process:
    /// `cargo test --release -p evanalyzer_core --lib bench_measure_colocalization -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_measure_colocalization() {
        let size: usize = std::env::var("BENCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let costes = match std::env::var("BENCH_COSTES").as_deref() {
            Ok("fast") => CostesMethod::Fast,
            Ok("off") => CostesMethod::Off,
            _ => CostesMethod::Faster,
        };
        let mut seed = 5u64;
        let mut noise = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 1000) as f32
        };
        let mut a = vec![0f32; size * size];
        let mut b = vec![0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let signal = if x % 40 < 9 && y % 40 < 9 {
                    20000.0
                } else {
                    0.0
                };
                a[y * size + x] = (signal + noise()) / 65535.0;
                b[y * size + x] = (signal * 0.7 + noise()) / 65535.0;
            }
        }
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 16;
        let half = size / 2;
        for (channel, values) in [(0, &a), (1, &b)] {
            for (x0, y0) in [(0, 0), (half, 0), (0, half), (half, half)] {
                let samples: Vec<f32> = (y0..y0 + half)
                    .flat_map(|y| (x0..x0 + half).map(move |x| values[y * size + x]))
                    .collect();
                let image = ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::from_size_slice(
                        ImageSize {
                            width: half,
                            height: half,
                        },
                        &samples,
                    )
                    .unwrap(),
                    tile_offset: Point2d { x: x0, y: y0 },
                    plane: None,
                });
                let tile = ImageTile {
                    offset_x: x0,
                    offset_y: y0,
                    width: half,
                    height: half,
                };
                cache.add_to_channel_cache(Arc::new(image), channel, tile);
            }
        }
        let mut id = 1u128;
        for y in (0..size - 9).step_by(40) {
            for x in (0..size - 9).step_by(40) {
                let mut object = Object::new(ObjectInit {
                    id: ObjectId(id),
                    bbox: [x as u32, y as u32, x as u32 + 8, y as u32 + 8],
                    area: 81,
                    mask_data: BitVec::<u64, Lsb0>::repeat(true, 81),
                    ..Default::default()
                });
                object.add_object_class(CELL, Scope::Global).unwrap();
                cache.object_cache.insert(object.id.clone(), object);
                id += 1;
            }
        }
        let image = MeasureImageColocalization {
            id: "c".into(),
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes,
        };
        let objects = MeasureObjectColocalization {
            id: "o".into(),
            objects: CELL,
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        let start = std::time::Instant::now();
        image.execute(&mut ctx, &mut cache).unwrap();
        let image_elapsed = start.elapsed();
        let start = std::time::Instant::now();
        objects.execute(&mut ctx, &mut cache).unwrap();
        let objects_elapsed = start.elapsed();
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .map(str::to_string)
            })
            .unwrap_or_default();
        eprintln!(
            "BENCH measure colocalization {size}², {} objects, costes {costes:?}: image {image_elapsed:?}, objects {objects_elapsed:?}, {hwm}",
            id - 1
        );
    }
}
