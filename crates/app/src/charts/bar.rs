//! # bar
//!
//! A bar per group: its mean, an error bar - ± SD, SEM or the 95 %
//! confidence interval (Student's t) - and every point as a dot (design
//! canvas "Results rework", ChartKinds). Names under the bars, turned when
//! they don't fit; how many under each.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::svg::{Anchor, Svg};
use super::{Drawing, Hit, Rgb, Scale, Theme};
use crate::number_format::format_number;

/// What a bar's error bar shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorBar {
    #[default]
    Sd,
    Sem,
    Ci95,
}

impl ErrorBar {
    /// Its name ("SD", "SEM", "95 % CI").
    pub fn name(self) -> &'static str {
        match self {
            ErrorBar::Sd => "SD",
            ErrorBar::Sem => "SEM",
            ErrorBar::Ci95 => "95 % CI",
        }
    }

    /// The error bar's half length for `n` values with standard deviation
    /// `sd` (none for the CI and SEM of fewer than two).
    pub fn half(self, sd: f64, n: usize) -> f64 {
        match self {
            ErrorBar::Sd => sd,
            _ if n < 2 => f64::NAN,
            ErrorBar::Sem => sd / (n as f64).sqrt(),
            ErrorBar::Ci95 => t_975(n - 1) * sd / (n as f64).sqrt(),
        }
    }
}

/// Student's t at 97.5 % for `df` degrees of freedom (the two-sided 95 %
/// interval) - Cornish-Fisher, within 0.3 % from 3 on, exact below.
fn t_975(df: usize) -> f64 {
    match df {
        0 => f64::NAN,
        1 => 12.706,
        2 => 4.303,
        _ => {
            let z: f64 = 1.959_964;
            let d = df as f64;
            z + (z.powi(3) + z) / (4.0 * d) + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * d * d)
        }
    }
}

/// One bar.
#[derive(Debug, Clone, PartialEq)]
pub struct BarSpec {
    pub label: String,
    pub color: Rgb,
    pub mean: f64,
    pub sd: f64,
    pub count: usize,
    /// Each a dot (e.g. every image's value).
    pub points: Vec<f64>,
    /// What `count` counts (" images"; "" values).
    pub count_noun: String,
}

/// How the bars are drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct BarLook {
    pub y_title: String,
    pub error: ErrorBar,
    pub grid: bool,
    pub dots: bool,
    pub show_n: bool,
}

/// At most this many dots per bar.
const MAX_DOTS: usize = 300;

/// The bars of `bars` as a `width` × `height` drawing - a mark per bar.
pub fn draw(bars: &[BarSpec], look: &BarLook, theme: &Theme, width: f64, height: f64) -> Drawing {
    let mut svg = Svg::new(width, height, theme);
    if bars.is_empty() || width < 80.0 || height < 80.0 {
        return Drawing { svg: svg.finish(), hits: Vec::new() };
    }
    let font = theme.font_size;
    let top = 16.0;
    let slot_guess = (width - 80.0) / bars.len() as f64;
    let turned = bars.iter().map(|b| theme.text_width(&b.label)).fold(0.0, f64::max) > slot_guess - 6.0;
    let names: Vec<String> = bars.iter().map(|b| b.label.chars().take(14).collect()).collect();
    let below = if turned {
        let longest = names.iter().map(|n| theme.text_width(n)).fold(0.0, f64::max);
        (longest * 0.77 + font + 12.0).min(height * 0.4)
    } else if look.show_n {
        font * 2.0 + 18.0
    } else {
        font + 14.0
    };
    let bottom = height - below;

    // From 0 (or below it) up to the highest mean + error or dot.
    let halves: Vec<f64> = bars.iter().map(|b| look.error.half(b.sd, b.count)).collect();
    let mut lo: f64 = 0.0;
    let mut hi: f64 = 0.0;
    for (b, half) in bars.iter().zip(&halves) {
        let h = if half.is_finite() { *half } else { 0.0 };
        lo = lo.min(b.mean - h);
        hi = hi.max(b.mean + h);
        if look.dots {
            for v in &b.points {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
    }
    let wanted = ((bottom - top) / 56.0).round().clamp(2.0, 10.0) as usize;
    let scale = Scale::nice(lo, hi, wanted, false);
    let ticks = scale.ticks(wanted);
    let label_w = ticks.iter().map(|(_, l)| theme.text_width(l)).fold(0.0, f64::max);
    let left = 8.0 + font + 10.0 + label_w + 8.0;
    let right = width - 12.0;
    let y_of = |v: f64| bottom - scale.position(v) * (bottom - top);
    for (value, label) in &ticks {
        let y = y_of(*value);
        if look.grid {
            svg.line(left, y, right, y, theme.grid, 1.0);
        }
        svg.line(left - 4.0, y, left, y, theme.axis, 1.0);
        svg.text(left - 7.0, y + font * 0.35, label, theme.text_dim, Anchor::End, None, false);
    }
    let zero = y_of(0.0);
    svg.line(left, top, left, bottom, theme.axis, 1.0);
    svg.line(left, zero, right, zero, theme.axis, 1.0);
    svg.text_vertical(8.0 + font * 0.5, (top + bottom) / 2.0, &look.y_title, theme.text_dim);

    let slot = (right - left) / bars.len() as f64;
    let bar_w = (slot * 0.55).clamp(6.0, 90.0);
    let mut hits = Vec::with_capacity(bars.len());
    for (i, (b, half)) in bars.iter().zip(&halves).enumerate() {
        let cx = left + slot * (i as f64 + 0.5);
        let y = y_of(b.mean);
        svg.rect(cx - bar_w / 2.0, y.min(zero), bar_w, (zero - y).abs(), b.color, 0.22, Some((b.color, 1.5)), 2.0);
        if look.dots && !b.points.is_empty() {
            let step = (b.points.len() as f64 / MAX_DOTS as f64).max(1.0);
            let mut k: f64 = 0.0;
            while (k as usize) < b.points.len() {
                let v = b.points[k as usize];
                let jitter = ((k * 0.618_034).fract() - 0.5) * bar_w * 0.7;
                svg.circle(cx + jitter, y_of(v), 2.4, b.color, 0.55);
                k += step;
            }
        }
        if half.is_finite() && *half > 0.0 {
            let (ya, yb) = (y_of(b.mean + half), y_of(b.mean - half));
            svg.line(cx, ya, cx, yb, theme.text, 1.5);
            svg.line(cx - bar_w * 0.18, ya, cx + bar_w * 0.18, ya, theme.text, 1.5);
            svg.line(cx - bar_w * 0.18, yb, cx + bar_w * 0.18, yb, theme.text, 1.5);
        }
        if turned {
            svg.text_turned(cx + font * 0.3, bottom + 10.0, &names[i], theme.text, -50.0);
        } else {
            let name: String = b.label.clone();
            svg.text(cx, bottom + font + 6.0, &name, theme.text, Anchor::Middle, None, false);
            if look.show_n {
                let n = format!("n = {}{}", format_number(b.count as f64), b.count_noun);
                svg.text(cx, bottom + font * 2.0 + 10.0, &n, theme.text_dim, Anchor::Middle, None, false);
            }
        }
        hits.push(Hit { index: i, x: cx - slot / 2.0, y: top, width: slot, height: bottom - top });
    }
    Drawing { svg: svg.finish(), hits }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(label: &str, mean: f64, points: Vec<f64>) -> BarSpec {
        BarSpec { label: label.into(), color: Rgb(0x2a6bdf), mean, sd: 4.0, count: 16, points, count_noun: " images".into() }
    }

    #[test]
    fn the_error_bar_is_sd_sem_or_the_t_interval() {
        assert_eq!(ErrorBar::Sd.half(4.0, 16), 4.0);
        assert_eq!(ErrorBar::Sem.half(4.0, 16), 1.0);
        // t(15) = 2.131.
        assert!((ErrorBar::Ci95.half(4.0, 16) - 2.131).abs() < 0.01);
        assert!(ErrorBar::Sem.half(4.0, 1).is_nan(), "no SEM of one value");
    }

    #[test]
    fn every_bar_is_a_mark_with_its_dots_and_error_bar() {
        let look = BarLook { y_title: "Objects per image".into(), error: ErrorBar::Sem, grid: true, dots: true, show_n: true };
        let bars = [bar("Control", 41.0, vec![38.0, 41.0, 44.0]), bar("1 µM", 52.0, vec![50.0, 54.0])];
        let drawing = draw(&bars, &look, &Theme::light(), 600.0, 400.0);
        assert_eq!(drawing.hits.len(), 2);
        assert_eq!(drawing.svg.matches("<circle").count(), 5);
        assert!(drawing.svg.contains(">Control<") && drawing.svg.contains("n = 16 images"));
    }
}
