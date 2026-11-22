//! # axis
//!
//! A chart axis: values to positions - linear, or log10 for values that
//! spread over decades - and the ticks a reader expects (1, 2, 2.5, 5 × 10ⁿ
//! apart; on a log axis the powers of ten, with 2 and 5 between while few).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::number_format::format_number;

/// Values `min`..`max` onto `0.0..=1.0` (from the axis' start).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale {
    pub min: f64,
    pub max: f64,
    pub log: bool,
    /// Between linear ticks (a log axis: unused).
    pub step: f64,
}

impl Scale {
    /// A scale over `lo..hi` widened to whole ticks (about `ticks` of them).
    /// `log`: only when `lo` is above 0 - else linear. An empty or single
    /// value range gets some room around it.
    pub fn nice(lo: f64, hi: f64, ticks: usize, log: bool) -> Self {
        let (mut lo, mut hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        if !lo.is_finite() || !hi.is_finite() {
            return Self { min: 0.0, max: 1.0, log: false, step: 0.25 };
        }
        if log && lo > 0.0 {
            let min = 10f64.powf(lo.log10().floor());
            let mut max = 10f64.powf(hi.log10().ceil());
            if max <= min {
                max = min * 10.0;
            }
            return Self { min, max, log: true, step: 0.0 };
        }
        if hi - lo < f64::EPSILON * hi.abs().max(1.0) {
            let pad = if lo == 0.0 { 1.0 } else { lo.abs() * 0.1 };
            lo -= pad;
            hi += pad;
        }
        let step = nice_step((hi - lo) / ticks.max(1) as f64);
        let mut min = (lo / step).floor() * step;
        let max = (hi / step).ceil() * step;
        // Values that are all positive start at 0 when it is close.
        if lo >= 0.0 && min > 0.0 && min <= (max - min) * 0.5 {
            min = 0.0;
        }
        Self { min, max, log: false, step }
    }

    /// Where `value` lies, 0 at `min`, 1 at `max` (outside: beyond).
    pub fn position(&self, value: f64) -> f64 {
        if self.log {
            let v = value.max(f64::MIN_POSITIVE);
            (v.log10() - self.min.log10()) / (self.max.log10() - self.min.log10())
        } else {
            (value - self.min) / (self.max - self.min)
        }
    }

    /// The ticks: value and label, from `min` to `max` - every `step` (a
    /// log axis: the decades, `wanted` unused).
    pub fn ticks(&self, _wanted: usize) -> Vec<(f64, String)> {
        if self.log {
            let first = self.min.log10().round() as i32;
            let last = self.max.log10().round() as i32;
            let decades = (last - first).max(1) as usize;
            let between: &[f64] = if decades <= 2 { &[2.0, 5.0] } else { &[] };
            let mut ticks = Vec::new();
            for exponent in first..=last {
                let base = 10f64.powi(exponent);
                ticks.push(base);
                if exponent < last {
                    ticks.extend(between.iter().map(|m| m * base));
                }
            }
            return ticks.into_iter().map(|v| (v, format_number(v))).collect();
        }
        let step = self.step;
        let count = ((self.max - self.min) / step).round() as i64;
        let decimals = step_decimals(step);
        (0..=count)
            .map(|i| {
                let v = self.min + i as f64 * step;
                // No "-0" and no 0.30000000000000004.
                let v = (v / step).round() * step;
                let v = if v.abs() < step * 1e-9 { 0.0 } else { v };
                (v, tick_label(v, decimals))
            })
            .collect()
    }
}

/// 1, 2, 2.5 or 5 × 10ⁿ - the smallest of them at least `raw`.
fn nice_step(raw: f64) -> f64 {
    if raw.is_nan() || raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let magnitude = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .map(|m| m * magnitude)
        .find(|step| *step >= raw * (1.0 - 1e-9))
        .unwrap_or(10.0 * magnitude)
}

/// How many decimals ticks `step` apart need (0.25: 2, 2.5: 1, 100: 0;
/// `None`: too many - written as 1.2e-6).
fn step_decimals(step: f64) -> Option<usize> {
    (0..=6).find(|d| {
        let scaled = step * 10f64.powi(*d as i32);
        (scaled - scaled.round()).abs() < 1e-6 * scaled.max(1.0)
    })
}

/// A tick's label: all ticks with the same decimals ("0.0, 2.5, 5.0"),
/// whole ones as the results write numbers.
fn tick_label(value: f64, decimals: Option<usize>) -> String {
    match decimals {
        Some(0) => format_number(value.round()),
        Some(d) => format!("{value:.d$}"),
        None => format_number(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_linear_scale_ends_on_nice_ticks_and_starts_at_zero_when_close() {
        let scale = Scale::nice(3.0, 577.0, 5, false);
        assert_eq!((scale.min, scale.max), (0.0, 600.0));
        let ticks: Vec<f64> = scale.ticks(5).into_iter().map(|t| t.0).collect();
        assert_eq!(ticks, [0.0, 200.0, 400.0, 600.0]);
        assert_eq!(scale.position(300.0), 0.5);
    }

    #[test]
    fn a_narrow_range_keeps_its_start_and_small_steps() {
        let scale = Scale::nice(0.52, 0.91, 4, false);
        assert!(scale.min > 0.0, "far from 0: not stretched down to it");
        let labels: Vec<String> = scale.ticks(4).into_iter().map(|t| t.1).collect();
        assert_eq!(labels, ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]);
        assert!(labels.iter().all(|l| !l.contains("000000")), "{labels:?}");
    }

    #[test]
    fn a_log_scale_spans_whole_decades() {
        let scale = Scale::nice(3.0, 577.0, 5, true);
        assert!(scale.log);
        assert_eq!((scale.min, scale.max), (1.0, 1000.0));
        assert!((scale.position(10.0) - 1.0 / 3.0).abs() < 1e-9);
        let ticks: Vec<f64> = scale.ticks(5).into_iter().map(|t| t.0).collect();
        assert_eq!(ticks, [1.0, 10.0, 100.0, 1000.0]);
    }

    #[test]
    fn log_is_refused_for_values_down_to_zero_and_a_single_value_gets_room() {
        assert!(!Scale::nice(0.0, 10.0, 5, true).log);
        let one = Scale::nice(4.0, 4.0, 5, false);
        assert!(one.min < 4.0 && one.max > 4.0);
    }
}
