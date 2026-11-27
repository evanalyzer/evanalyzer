//! # brightness
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Brightness per channel: every channel of a patch multiplied by its own
//! random factor, so the network learns that a fainter or brighter image has
//! the same spots - and that the channels' brightness to each other varies
//! (a spot channel stained more strongly, a reference channel exposed less).
//!
//! The input only: the answer stays the same, so labels are never touched.
//! Multiplied, not shifted: a darker image keeps its black where nothing
//! shines; a different background level is the normalisation's job.

use crate::train::SeededUniform;

/// One factor per channel.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Brightness {
    factors: Vec<f32>,
}

impl Brightness {
    /// `channels` factors, each drawn on its own between `low` and `high`
    /// (e.g. 0.7 and 1.4), evenly on a log scale: halving is as likely as
    /// doubling, so the mean brightness doesn't drift up or down.
    pub(crate) fn random(channels: usize, low: f32, high: f32, random: &mut SeededUniform) -> Self {
        debug_assert!(0.0 < low && low <= high, "{low}..{high}");
        let (ln_low, ln_high) = (low.ln(), high.ln());
        let factors = (0..channels)
            .map(|_| (ln_low + random.unit() * (ln_high - ln_low)).exp())
            .collect();
        Self { factors }
    }

    /// The factor of each channel.
    pub(crate) fn factors(&self) -> &[f32] {
        &self.factors
    }

    /// `data` (one plane of `plane_len` values per channel, one after the
    /// other) changed in place - no copy of the patch.
    pub(crate) fn apply(&self, data: &mut [f32], plane_len: usize) {
        debug_assert_eq!(data.len(), self.factors.len() * plane_len);
        for (plane, &factor) in data.chunks_mut(plane_len).zip(&self.factors) {
            for value in plane {
                *value *= factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Brightness;
    use crate::train::SeededUniform;

    #[test]
    fn every_channel_is_multiplied_by_its_own_factor() {
        let brightness = Brightness {
            factors: vec![0.5, 2.0],
        };
        // Two channels of 2 values each.
        let mut data = vec![1.0, 0.2, 1.0, 0.2];
        brightness.apply(&mut data, 2);
        assert_eq!(data, vec![0.5, 0.1, 2.0, 0.4]);
    }

    #[test]
    fn black_stays_black() {
        let mut random = SeededUniform::new(3);
        let brightness = Brightness::random(1, 0.5, 2.0, &mut random);
        let mut data = vec![0.0; 4];
        brightness.apply(&mut data, 4);
        assert_eq!(data, vec![0.0; 4]);
    }

    #[test]
    fn the_factors_lie_in_the_range_and_differ_per_channel() {
        let mut random = SeededUniform::new(5);
        for _ in 0..200 {
            let brightness = Brightness::random(2, 0.7, 1.4, &mut random);
            let [a, b] = brightness.factors() else {
                panic!()
            };
            assert!(
                (0.7..=1.4).contains(a) && (0.7..=1.4).contains(b),
                "{a}, {b}"
            );
            assert_ne!(a, b, "the channels share one factor");
        }
    }

    #[test]
    fn halving_is_as_likely_as_doubling() {
        // On a log scale between 0.5 and 2: about as many factors below 1 as
        // above, and the log-mean near 0 (no drift).
        let mut random = SeededUniform::new(8);
        let factors: Vec<f32> = (0..4000)
            .map(|_| Brightness::random(1, 0.5, 2.0, &mut random).factors()[0])
            .collect();
        let below = factors.iter().filter(|&&f| f < 1.0).count();
        assert!((1800..2200).contains(&below), "{below} of 4000 below 1");
        let log_mean = factors.iter().map(|f| f.ln()).sum::<f32>() / factors.len() as f32;
        assert!(log_mean.abs() < 0.03, "{log_mean}");
    }

    #[test]
    fn a_range_of_one_value_keeps_the_image() {
        let mut random = SeededUniform::new(1);
        let brightness = Brightness::random(2, 1.0, 1.0, &mut random);
        assert_eq!(brightness.factors(), &[1.0, 1.0]);
    }

    #[test]
    fn the_same_seed_draws_the_same_factors() {
        let draw = |seed| Brightness::random(3, 0.7, 1.4, &mut SeededUniform::new(seed));
        assert_eq!(draw(4), draw(4));
        assert_ne!(draw(4), draw(5));
    }
}
