//! # percentile
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The values at percentiles of a tile, as `np.percentile` with its default
//! `method="linear"`: the two order statistics around position
//! `p / 100 * (n - 1)`, interpolated like NumPy's `_lerp`. NaN is skipped.
//!
//! Without a sorted copy of the tile: a histogram of [`BINS`] bins over the
//! values' range finds the bin of each order statistic; a bin holding one
//! distinct value gives it exactly, any other is cut into bins again over
//! its own range. Integer images (scaled by their bit depth, `k / 65535`)
//! land one value per bin: one pass for the range, one for the histogram.
//! A bin's range always shrinks, so floats end after a few passes as well.

/// Bins per pass: 768 KB of counts, minima and maxima.
const BINS: usize = 65_536;

/// The values at percentiles `ps` (0-100, clamped) of `data`, NaN skipped;
/// `None` for empty or all-NaN data.
pub(crate) fn percentiles(data: &[f32], ps: &[f64]) -> Option<Vec<f64>> {
    let (count, min, max) = range(data)?;
    // Each percentile's two order statistics and the weight between them.
    let positions: Vec<(usize, usize, f64)> = ps
        .iter()
        .map(|&p| {
            let virtual_index = p.clamp(0.0, 100.0) / 100.0 * (count - 1) as f64;
            let below = virtual_index.floor();
            let lower = below as usize;
            (lower, (lower + 1).min(count - 1), virtual_index - below)
        })
        .collect();
    let mut ranks: Vec<usize> = positions.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let values = order_statistics(data, &ranks, min, max);
    let value = |rank: usize| f64::from(values[ranks.binary_search(&rank).expect("asked for")]);
    Some(
        positions
            .into_iter()
            .map(|(a, b, t)| lerp(value(a), value(b), t))
            .collect(),
    )
}

/// NumPy's `_lerp`: from the nearer end, so `t = 1` gives `b` exactly.
/// Equal ends and `t = 0` give `a` - the same for finite values, and no NaN
/// from an infinite neighbour (NumPy's `inf * 0`).
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if a == b || t == 0.0 {
        return a;
    }
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// The number of values that aren't NaN, their minimum and maximum.
fn range(data: &[f32]) -> Option<(usize, f32, f32)> {
    let (mut count, mut min, mut max) = (0usize, f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        if !v.is_nan() {
            count += 1;
            min = min.min(v);
            max = max.max(v);
        }
    }
    (count > 0).then_some((count, min, max))
}

/// The values of rank `ranks` (0-based, ascending) among the values that
/// aren't NaN, all within `min..=max`.
fn order_statistics(data: &[f32], ranks: &[usize], min: f32, max: f32) -> Vec<f32> {
    if min == max {
        return vec![min; ranks.len()];
    }
    if !min.is_finite() || !max.is_finite() || data.len() > u32::MAX as usize {
        return ranks
            .iter()
            .map(|&rank| selected(data, rank, min, max))
            .collect();
    }
    let whole = Histogram::new(data, min, max);
    ranks
        .iter()
        .map(|&rank| order_statistic(data, rank, &whole))
        .collect()
}

/// The value of rank `rank`, starting from the histogram of all values.
fn order_statistic(data: &[f32], rank: usize, whole: &Histogram) -> f32 {
    let mut finer: Option<Histogram> = None;
    // Values below the histogram's range.
    let mut below = 0;
    loop {
        let histogram = finer.as_ref().unwrap_or(whole);
        let (bin, before) = histogram.bin_of(rank - below);
        let (lo, hi) = (histogram.min[bin], histogram.max[bin]);
        if lo == hi {
            return lo;
        }
        if (lo, hi) == (histogram.lo, histogram.hi) {
            // No narrower range (can't happen with monotone bins) - select.
            return selected(data, rank - below, lo, hi);
        }
        below += before;
        finer = Some(Histogram::new(data, lo, hi));
    }
}

/// The value of rank `rank` among the values in `lo..=hi`, from a copy of
/// them - only for infinite ranges and as a safety net.
fn selected(data: &[f32], rank: usize, lo: f32, hi: f32) -> f32 {
    let mut values: Vec<f32> = data
        .iter()
        .copied()
        .filter(|v| (lo..=hi).contains(v))
        .collect();
    *values.select_nth_unstable_by(rank, f32::total_cmp).1
}

/// Counts, minimum and maximum per bin of the values in `lo..=hi`.
struct Histogram {
    lo: f32,
    hi: f32,
    counts: Vec<u32>,
    min: Vec<f32>,
    max: Vec<f32>,
}

impl Histogram {
    /// `lo < hi`, both finite. The bin index rises with the value (each
    /// step a correctly rounded, monotone operation): `lo` is in the first
    /// bin, `hi` in the last.
    fn new(data: &[f32], lo: f32, hi: f32) -> Self {
        let mut histogram = Self {
            lo,
            hi,
            counts: vec![0; BINS],
            min: vec![f32::INFINITY; BINS],
            max: vec![f32::NEG_INFINITY; BINS],
        };
        let (base, scale) = (f64::from(lo), BINS as f64 / (f64::from(hi) - f64::from(lo)));
        for &v in data {
            if (lo..=hi).contains(&v) {
                let bin = (((f64::from(v) - base) * scale) as usize).min(BINS - 1);
                histogram.counts[bin] += 1;
                histogram.min[bin] = histogram.min[bin].min(v);
                histogram.max[bin] = histogram.max[bin].max(v);
            }
        }
        histogram
    }

    /// The bin holding the value of rank `rank` within this histogram, and
    /// how many values lie in the bins before it.
    fn bin_of(&self, rank: usize) -> (usize, usize) {
        let mut before = 0;
        for (bin, &count) in self.counts.iter().enumerate() {
            let count = count as usize;
            if rank < before + count {
                return (bin, before);
            }
            before += count;
        }
        unreachable!("rank {rank} of {before} values")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.percentile` the plain way: sorted, then the same positions.
    fn sorted_reference(data: &[f32], ps: &[f64]) -> Option<Vec<f64>> {
        let mut values: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
        if values.is_empty() {
            return None;
        }
        values.sort_by(f32::total_cmp);
        let n = values.len();
        Some(
            ps.iter()
                .map(|&p| {
                    let index = p / 100.0 * (n - 1) as f64;
                    let below = index.floor();
                    let lower = below as usize;
                    let upper = (lower + 1).min(n - 1);
                    lerp(
                        f64::from(values[lower]),
                        f64::from(values[upper]),
                        index - below,
                    )
                })
                .collect(),
        )
    }

    /// Deterministic pseudo-random numbers in 0..1.
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    const PS: [f64; 7] = [0.0, 1.0, 1.5, 50.0, 99.0, 99.8, 100.0];

    /// Values NumPy's documented linear method gives (`(n - 1) * p / 100`).
    #[test]
    fn small_inputs_give_numpys_values() {
        let got = percentiles(&[4.0, 1.0, 3.0, 2.0], &[0.0, 1.0, 50.0, 100.0]).unwrap();
        assert_eq!(got[0], 1.0);
        assert!((got[1] - 1.03).abs() < 1e-12, "{}", got[1]);
        assert_eq!(got[2], 2.5);
        assert_eq!(got[3], 4.0);
        let ramp: Vec<f32> = (0..1000).map(|k| k as f32).collect();
        let got = percentiles(&ramp, &[1.0, 99.8]).unwrap();
        assert!((got[0] - 9.99).abs() < 1e-9, "{}", got[0]);
        assert!((got[1] - 997.002).abs() < 1e-9, "{}", got[1]);
    }

    #[test]
    fn nan_is_skipped_and_nothing_gives_none() {
        let got = percentiles(&[f32::NAN, 3.0, 1.0, f32::NAN, 2.0], &[50.0]).unwrap();
        assert_eq!(got, [2.0]);
        assert_eq!(percentiles(&[], &[50.0]), None);
        assert_eq!(percentiles(&[f32::NAN; 4], &[50.0]), None);
    }

    #[test]
    fn a_constant_tile_gives_its_value_and_percentiles_are_clamped() {
        assert_eq!(percentiles(&[0.25; 100], &PS).unwrap(), [0.25; 7]);
        assert_eq!(
            percentiles(&[1.0, 2.0], &[-5.0, 250.0]).unwrap(),
            [1.0, 2.0]
        );
    }

    /// 8-, 12- and 16-bit images as the reader gives them (`k / max`).
    #[test]
    fn integer_images_match_the_sorted_reference_exactly() {
        for (bits, seed) in [(8u32, 1u64), (12, 2), (16, 3)] {
            let top = ((1u32 << bits) - 1) as f64;
            let data: Vec<f32> = noise(40_000, seed)
                .into_iter()
                .map(|x| ((x * x * top).round() / top) as f32)
                .collect();
            assert_eq!(
                percentiles(&data, &PS),
                sorted_reference(&data, &PS),
                "{bits} bit"
            );
        }
    }

    #[test]
    fn float_images_match_the_sorted_reference_exactly() {
        let data: Vec<f32> = noise(50_000, 7)
            .into_iter()
            .map(|x| ((x - 0.3) * 1e4 * x) as f32)
            .collect();
        assert_eq!(percentiles(&data, &PS), sorted_reference(&data, &PS));
        // Values packed closer than one bin of the whole range: a few
        // passes, still exact.
        let mut close: Vec<f32> = noise(20_000, 9)
            .into_iter()
            .map(|x| (1.0 + x * 1e-4) as f32)
            .collect();
        close.extend([0.0, 1e6]);
        assert_eq!(percentiles(&close, &PS), sorted_reference(&close, &PS));
    }

    /// Mostly one value (background) and a few others in the same bin.
    #[test]
    fn a_crowded_bin_matches_the_sorted_reference() {
        let mut data = vec![0.5f32; 30_000];
        data.extend(noise(300, 11).into_iter().map(|x| (0.5 + x * 1e-6) as f32));
        data.extend([0.0, 1.0]);
        assert_eq!(percentiles(&data, &PS), sorted_reference(&data, &PS));
    }

    #[test]
    fn infinities_are_values_at_the_ends() {
        let data = [f32::NEG_INFINITY, 0.0, 1.0, 2.0, f32::INFINITY];
        assert_eq!(
            percentiles(&data, &[25.0, 50.0, 75.0]).unwrap(),
            [0.0, 1.0, 2.0]
        );
        assert_eq!(percentiles(&data, &[100.0]).unwrap(), [f64::INFINITY]);
    }
}
