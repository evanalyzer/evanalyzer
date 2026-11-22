//! # scatter
//!
//! A dot per object over two values (design canvas "Results rework",
//! ChartKinds): both axes ticked, grid lines, how many of how many are
//! drawn. Every dot's place comes back as a mark (`Hit` of size 0), so a
//! click can name the nearest object.
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

/// The dots: `(x, y)` each, all in one colour.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterSpec {
    pub points: Vec<(f64, f64)>,
    pub color: Rgb,
    /// How many objects there are (the dots may be a sample of them).
    pub total: usize,
}

/// How the scatter is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterLook {
    pub x_title: String,
    pub y_title: String,
    pub log_x: bool,
    pub log_y: bool,
    pub grid: bool,
}

/// The scatter of `spec` as a `width` × `height` drawing - a mark per dot
/// (its centre, size 0), in `spec.points`' order.
pub fn draw(spec: &ScatterSpec, look: &ScatterLook, theme: &Theme, width: f64, height: f64) -> Drawing {
    let mut svg = Svg::new(width, height, theme);
    if spec.points.is_empty() || width < 80.0 || height < 80.0 {
        return Drawing { svg: svg.finish(), hits: Vec::new() };
    }
    let font = theme.font_size;
    let top = 16.0 + font + 6.0;
    let bottom = height - (font * 2.0 + 20.0);

    let range = |pick: fn(&(f64, f64)) -> f64| {
        spec.points
            .iter()
            .map(pick)
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
    };
    let (x_lo, x_hi) = range(|p| p.0);
    let (y_lo, y_hi) = range(|p| p.1);
    let y_wanted = ((bottom - top) / 56.0).round().clamp(2.0, 10.0) as usize;
    let y_scale = Scale::nice(y_lo, y_hi, y_wanted, look.log_y);
    let y_ticks = y_scale.ticks(y_wanted);
    let label_w = y_ticks
        .iter()
        .map(|(_, label)| theme.text_width(label))
        .fold(0.0, f64::max);
    let left = 8.0 + font + 10.0 + label_w + 8.0;
    let right = width - 16.0;
    let x_wanted = ((right - left) / 90.0).round().clamp(2.0, 12.0) as usize;
    let x_scale = Scale::nice(x_lo, x_hi, x_wanted, look.log_x);
    let x_of = |v: f64| left + x_scale.position(v) * (right - left);
    let y_of = |v: f64| bottom - y_scale.position(v) * (bottom - top);

    for (value, label) in &y_ticks {
        let y = y_of(*value);
        if look.grid {
            svg.line(left, y, right, y, theme.grid, 1.0);
        }
        svg.line(left - 4.0, y, left, y, theme.axis, 1.0);
        svg.text(left - 7.0, y + font * 0.35, label, theme.text_dim, Anchor::End, None, false);
    }
    for (value, label) in x_scale.ticks(x_wanted) {
        let x = x_of(value);
        if look.grid {
            svg.line(x, top, x, bottom, theme.grid, 1.0);
        }
        svg.line(x, bottom, x, bottom + 4.0, theme.axis, 1.0);
        svg.text(x, bottom + font + 6.0, &label, theme.text_dim, Anchor::Middle, None, false);
    }
    svg.line(left, top, left, bottom, theme.axis, 1.0);
    svg.line(left, bottom, right, bottom, theme.axis, 1.0);
    svg.text_vertical(8.0 + font * 0.5, (top + bottom) / 2.0, &look.y_title, theme.text_dim);
    svg.text((left + right) / 2.0, height - 6.0, &look.x_title, theme.text_dim, Anchor::Middle, None, false);
    let shown = if spec.total > spec.points.len() {
        format!(
            "{} of {} objects drawn (an even sample)",
            format_number(spec.points.len() as f64),
            format_number(spec.total as f64)
        )
    } else {
        format!("{} objects", format_number(spec.points.len() as f64))
    };
    svg.text(left, 16.0, &shown, theme.text_dim, Anchor::Start, None, false);

    let mut hits = Vec::with_capacity(spec.points.len());
    for (i, (x, y)) in spec.points.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let (px, py) = (x_of(*x), y_of(*y));
        svg.circle(px, py, 2.4, spec.color, 0.5);
        hits.push(Hit { index: i, x: px, y: py, width: 0.0, height: 0.0 });
    }
    Drawing { svg: svg.finish(), hits }
}

/// The mark nearest to `(x, y)` within `radius` (its index in the points).
pub fn nearest(hits: &[Hit], x: f64, y: f64, radius: f64) -> Option<usize> {
    hits.iter()
        .map(|hit| (hit.index, (hit.x - x).powi(2) + (hit.y - y).powi(2)))
        .filter(|(_, d2)| *d2 <= radius * radius)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look() -> ScatterLook {
        ScatterLook { x_title: "Area µm²".into(), y_title: "Circularity".into(), log_x: false, log_y: false, grid: true }
    }

    #[test]
    fn every_dot_is_a_mark_and_a_click_names_the_nearest() {
        let spec = ScatterSpec { points: vec![(1.0, 1.0), (5.0, 2.0), (9.0, 9.0)], color: Rgb(0x2a6bdf), total: 10 };
        let drawing = draw(&spec, &look(), &Theme::light(), 500.0, 400.0);
        assert_eq!(drawing.hits.len(), 3);
        assert_eq!(drawing.svg.matches("<circle").count(), 3);
        assert!(drawing.svg.contains("3 of 10 objects drawn"));
        let second = drawing.hits[1];
        assert_eq!(nearest(&drawing.hits, second.x + 3.0, second.y - 2.0, 8.0), Some(1));
        assert_eq!(nearest(&drawing.hits, second.x + 30.0, second.y, 8.0), None, "too far");
    }
}
