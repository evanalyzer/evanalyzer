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
}
