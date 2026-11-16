//! # robust_stats
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Background and noise of an image robust against its objects: the median
//! and the median absolute deviation (MAD) scaled to a standard deviation
//! (Hampel 1974; Donoho & Johnstone 1994 estimate wavelet noise the same
//! way, "divided by 0.6745"). Exact - no histogram - so the result doesn't
//! depend on the value range.

/// Turns the median absolute deviation of normally distributed values into
/// their standard deviation: 1 / Φ⁻¹(3/4) = 1 / 0.6745 (Hampel 1974,
/// Donoho & Johnstone 1994 p. 446).
pub(crate) const MAD_TO_SIGMA: f32 = 1.4826;

/// The background (median) and noise (MAD × [`MAD_TO_SIGMA`]) of
/// `values`, NaN skipped; `None` without a value. Copies the values (4
/// bytes per value).
pub(crate) fn background_and_noise(values: &[f32]) -> Option<(f32, f32)> {
    let mut work: Vec<f32> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    background_and_noise_in_place(&mut work)
}

/// [`background_and_noise`] of `work` (no NaN), which it reorders and
/// overwrites - for a caller that has a scratch buffer at hand.
pub(crate) fn background_and_noise_in_place(work: &mut [f32]) -> Option<(f32, f32)> {
    median_and_mad_in_place(work).map(|(background, mad)| (background, mad * MAD_TO_SIGMA))
}

/// The median and the plain median absolute deviation (not scaled) of
/// `work` (no NaN), which it reorders and overwrites; `None` if empty - for
/// a method whose paper scales the MAD itself.
pub(crate) fn median_and_mad_in_place(work: &mut [f32]) -> Option<(f32, f32)> {
    if work.is_empty() {
        return None;
    }
    let median = median_in_place(work);
    for v in work.iter_mut() {
        *v = (*v - median).abs();
    }
    Some((median, median_in_place(work)))
}

/// The median of `values` (not empty; reordered): the middle value, or the
/// mean of the two middle values for an even count.
pub(crate) fn median_in_place(values: &mut [f32]) -> f32 {
    let odd = values.len() % 2 == 1;
    let mid = values.len() / 2;
    let (lower, &mut upper, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
    if odd {
        upper
    } else {
        let below = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (below + upper) / 2.0
    }
}

/// Buckets per pass of the radix select: 16 of the 32 bits of a value.
pub(crate) const RADIX_BUCKETS: usize = 1 << 16;

/// [`median_and_mad_in_place`] of the `n` values `values()` yields, without
/// a copy of them: a radix select over their bits (two passes each), so
/// `values` is called four times and must yield the same values each
/// time. Picks the same values as the selection by [`f32::total_cmp`], so
/// the result is [`median_and_mad_in_place`]'s. `hist` is scratch,
/// [`RADIX_BUCKETS`] counters.
pub(crate) fn median_and_mad_streamed<I: Iterator<Item = f32>>(
    n: usize,
    values: impl Fn() -> I,
    hist: &mut [u32],
) -> Option<(f32, f32)> {
    if n == 0 {
        return None;
    }
    let median = median_streamed(n, &values, hist);
    let mad = median_streamed(n, &|| values().map(move |v| (v - median).abs()), hist);
    Some((median, mad))
}

/// [`median_in_place`] of the `n` values `values()` yields (see
/// [`median_and_mad_streamed`]).
fn median_streamed<I: Iterator<Item = f32>>(
    n: usize,
    values: &impl Fn() -> I,
    hist: &mut [u32],
) -> f32 {
    let (below, upper) = kth_streamed(n / 2, values, hist);
    if n % 2 == 1 {
        upper
    } else {
        (below + upper) / 2.0
    }
}

/// The `k`-th smallest (from 0) of the values `values()` yields, in
/// [`f32::total_cmp`] order, and the one before it (the `k`-th itself for
/// `k` 0): their high 16 bits counted first, then the low 16 bits of those
/// in the bucket that holds the `k`-th - the largest value below that
/// bucket kept on the way, in case the one before is there. Two passes.
fn kth_streamed<I: Iterator<Item = f32>>(
    k: usize,
    values: &impl Fn() -> I,
    hist: &mut [u32],
) -> (f32, f32) {
    // The bucket holding the value of rank `rank`, and its rank in it.
    let bucket = |hist: &[u32], mut rank: usize| {
        for (b, &count) in hist.iter().enumerate() {
            if rank < count as usize {
                return (b as u32, rank);
            }
            rank -= count as usize;
        }
        unreachable!("the rank is below the number of values")
    };
    hist.fill(0);
    for v in values() {
        hist[(sort_key(v) >> 16) as usize] += 1;
    }
    let (high, rank) = bucket(hist, k);
    hist.fill(0);
    let mut largest_below = 0u32;
    let bucket_start = high << 16;
    for v in values() {
        let key = sort_key(v);
        if key >> 16 == high {
            hist[(key & 0xFFFF) as usize] += 1;
        }
        // Without a branch: about half the values lie below, in no order.
        largest_below = largest_below.max(if key < bucket_start { key } else { 0 });
    }
    let kth = from_sort_key(high << 16 | bucket(hist, rank).0);
    let before = match rank {
        0 if k == 0 => kth,
        0 => from_sort_key(largest_below),
        _ => from_sort_key(high << 16 | bucket(hist, rank - 1).0),
    };
    (before, kth)
}

/// The bits of `v` as an unsigned number in [`f32::total_cmp`] order:
/// negative values with all bits flipped, the others with the sign set.
/// Without a branch (the signs of a wavelet plane come in no order).
fn sort_key(v: f32) -> u32 {
    let bits = v.to_bits();
    bits ^ (((bits as i32 >> 31) as u32) | 1 << 31)
}

/// The value of a [`sort_key`].
fn from_sort_key(key: u32) -> f32 {
    f32::from_bits(key ^ ((((!key) as i32 >> 31) as u32) | 1 << 31))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_and_noise_are_median_and_scaled_mad() {
        assert_eq!(background_and_noise(&[]), None);
        assert_eq!(background_and_noise(&[f32::NAN]), None);
        // Median 3, deviations 2 1 0 1 7 → MAD 1.
        let (background, noise) =
            background_and_noise(&[1.0, 2.0, 3.0, f32::NAN, 4.0, 10.0]).unwrap();
        assert_eq!(background, 3.0);
        assert_eq!(noise, MAD_TO_SIGMA);
        // Even count: mean of the two middle values.
        assert_eq!(background_and_noise(&[4.0, 1.0, 3.0, 2.0]).unwrap().0, 2.5);
    }

    /// The streamed selection gives the in-place one's median and MAD bit
    /// for bit: odd and even counts, duplicates, negative values and zeros
    /// of both signs.
    #[test]
    fn streamed_equals_in_place() {
        let mut state = 1u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let mut cases: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![2.0, 1.0],
            vec![3.0; 6],
            vec![-0.0, 0.0, -0.0, 0.0],
            vec![1.0, 2.0, 2.0, 2.0, 5.0, -7.0],
            vec![-1e-30, 1e-30, -3.0, 4e20, 0.0, -0.0, 2.5],
        ];
        for n in [999, 1000, 4096] {
            cases.push((0..n).map(|_| next()).collect());
            // Few distinct values: many duplicates around the middle.
            cases.push((0..n).map(|_| (next() * 6.0).round()).collect());
        }
        let mut hist = vec![0u32; RADIX_BUCKETS];
        for values in &cases {
            let expected = median_and_mad_in_place(&mut values.clone()).unwrap();
            let got = median_and_mad_streamed(values.len(), || values.iter().copied(), &mut hist)
                .unwrap();
            assert_eq!(
                (got.0.to_bits(), got.1.to_bits()),
                (expected.0.to_bits(), expected.1.to_bits()),
                "{values:?}"
            );
        }
        assert_eq!(
            median_and_mad_streamed(0, || [].into_iter(), &mut hist),
            None
        );
    }
}
