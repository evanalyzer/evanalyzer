//! # boxplot
//!
//! A box per group (design canvas "Results rework", ChartBox): Q1 to Q3
//! with the median, whiskers to the furthest values within 1.5 × IQR, the
//! values beyond as dots - whiskers and dots never tell different stories.
//! A y axis with ticks and the value's name, an optional grid, a log
//! scale; each box named below, with how many values it holds.
//!
//! When a few far values would squash every box flat, the axis stops at
//! the whiskers and the values beyond show as a mark at the edge with
//! their number ("▲ 12").
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

/// One box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxSpec {
    pub label: String,
    pub color: Rgb,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub mean: f64,
    /// The values' standard deviation, and how many objects they come from
    /// (= `count` per object) - for a front end's numbers.
    pub sd: f64,
    pub objects: usize,
    /// The furthest values within 1.5 × IQR of Q1 and Q3.
    pub whisker_low: f64,
    pub whisker_high: f64,
    /// The values beyond the whiskers.
    pub outliers: Vec<f64>,
    /// How many values the box holds.
    pub count: usize,
    /// Each a dot - e.g. every image's value (empty: only the far values
    /// are dots).
    pub points: Vec<f64>,
    /// What `count` counts, after the number (" images"; "" values).
    pub count_noun: String,
}

/// How the boxplot is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxplotLook {
    /// The value's name, with its unit ("Area µm²").
    pub y_title: String,
    pub log: bool,
    pub grid: bool,
    /// Every point a dot (else only the far values).
    pub dots: bool,
    /// The mean as a ◇.
    pub mean: bool,
    /// How many values under each box.
    pub show_n: bool,
    /// Whiskers from the smallest to the largest value - no far values -
    /// instead of to the furthest within 1.5 × IQR.
    pub min_max: bool,
}

impl BoxplotLook {
    /// As the boxplot starts: grid, dots, n, Tukey whiskers.
    pub fn new(y_title: impl Into<String>) -> Self {
        Self {
            y_title: y_title.into(),
            log: false,
            grid: true,
            dots: true,
            mean: false,
            show_n: true,
            min_max: false,
        }
    }
}

/// Far values beyond this many times the whiskers' span cut the axis.
const MAX_STRETCH: f64 = 3.0;

/// Turned names are shortened to this many characters.
const TURNED_CHARS: usize = 14;

/// At most this many far values of a box are drawn as dots - beyond, the
/// picture would grow (and slow) without showing more.
const MAX_DOTS: usize = 200;

/// The boxplot of `boxes` as a `width` × `height` drawing.
pub fn draw(boxes: &[BoxSpec], look: &BoxplotLook, theme: &Theme, width: f64, height: f64) -> Drawing {
    let mut svg = Svg::new(width, height, theme);
    if boxes.is_empty() || width < 80.0 || height < 80.0 {
        return Drawing { svg: svg.finish(), hits: Vec::new() };
    }
    let font = theme.font_size;
    // Min - max whiskers: they reach every value, none is far.
    let owned: Vec<BoxSpec>;
    let boxes = if look.min_max {
        owned = boxes
            .iter()
            .map(|b| BoxSpec {
                whisker_low: b.min,
                whisker_high: b.max,
                outliers: Vec::new(),
                ..b.clone()
            })
            .collect();
        &owned[..]
    } else {
        boxes
    };

    // The values' range: the whiskers, and the far values while they don't
    // squash the boxes.
    let fold = |f: fn(f64, f64) -> f64, init: f64, pick: &dyn Fn(&BoxSpec) -> f64| {
        boxes.iter().map(pick).fold(init, f)
    };
    let whisker_lo = fold(f64::min, f64::INFINITY, &|b| b.whisker_low);
    let whisker_hi = fold(f64::max, f64::NEG_INFINITY, &|b| b.whisker_high);
    let all_lo = boxes
        .iter()
        .flat_map(|b| b.outliers.iter().copied())
        .fold(whisker_lo, f64::min);
    let all_hi = boxes
        .iter()
        .flat_map(|b| b.outliers.iter().copied())
        .fold(whisker_hi, f64::max);
    let log = look.log && all_lo > 0.0;
    let span = |lo: f64, hi: f64| if log { hi.log10() - lo.log10() } else { hi - lo };
    let whisker_span = span(whisker_lo, whisker_hi).max(f64::EPSILON);
    let (lo, hi) = if span(all_lo, all_hi) <= whisker_span * MAX_STRETCH {
        (all_lo, all_hi)
    } else {
        (whisker_lo, whisker_hi)
    };

    // The frame: tick labels and the title left, names and counts below -
    // names turned when they don't fit under their box (then no counts).
    let top = 16.0;
    let slot_guess = (width - 80.0) / boxes.len() as f64;
    let widest_name = boxes
        .iter()
        .map(|b| theme.text_width(&b.label))
        .fold(0.0, f64::max);
    let turned = widest_name > slot_guess - 6.0;
    let turned_names: Vec<String> = boxes
        .iter()
        .map(|b| fit(&b.label, theme.font_size * 0.58 * TURNED_CHARS as f64, theme))
        .collect();
    let below = if turned {
        let longest = turned_names
            .iter()
            .map(|name| theme.text_width(name))
            .fold(0.0, f64::max);
        // Turned by 50 degrees: about this high.
        (longest * 0.77 + font + 12.0).min(height * 0.4)
    } else if look.show_n {
        font * 2.0 + 18.0
    } else {
        font + 14.0
    };
    let bottom = height - below;
    let tick_count = ((bottom - top) / 56.0).round().clamp(2.0, 10.0) as usize;
    let scale = Scale::nice(lo, hi, tick_count, log);
    let ticks = scale.ticks(tick_count);
    let label_w = ticks
        .iter()
        .map(|(_, label)| theme.text_width(label))
        .fold(0.0, f64::max);
    let left = 8.0 + font + 10.0 + label_w + 8.0;
    let right = width - 12.0;
    let plot_w = right - left;
    let y_of = |v: f64| bottom - scale.position(v) * (bottom - top);

    // Grid, ticks and the y axis.
    for (value, label) in &ticks {
        let y = y_of(*value);
        if look.grid {
            svg.line(left, y, right, y, theme.grid, 1.0);
        }
        svg.line(left - 4.0, y, left, y, theme.axis, 1.0);
        svg.text(left - 7.0, y + font * 0.35, label, theme.text_dim, Anchor::End, None, false);
    }
    svg.line(left, top, left, bottom, theme.axis, 1.0);
    svg.line(left, bottom, right, bottom, theme.axis, 1.0);
    svg.text_vertical(8.0 + font * 0.5, (top + bottom) / 2.0, &look.y_title, theme.text_dim);

    // The boxes.
    let slot = plot_w / boxes.len() as f64;
    let box_w = (slot * 0.55).clamp(6.0, 90.0);
    let mut hits = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let cx = left + slot * (i as f64 + 0.5);
        let clamp_y = |v: f64| y_of(v).clamp(top, bottom);
        let (y_q1, y_q3, y_med) = (clamp_y(b.q1), clamp_y(b.q3), clamp_y(b.median));
        let (y_lo, y_hi) = (clamp_y(b.whisker_low), clamp_y(b.whisker_high));

        // Whiskers with caps.
        svg.line(cx, y_hi, cx, y_q3, b.color, 1.2);
        svg.line(cx, y_q1, cx, y_lo, b.color, 1.2);
        svg.line(cx - box_w * 0.25, y_hi, cx + box_w * 0.25, y_hi, b.color, 1.2);
        svg.line(cx - box_w * 0.25, y_lo, cx + box_w * 0.25, y_lo, b.color, 1.2);
        // Q1 to Q3, the median across.
        svg.rect(
            cx - box_w / 2.0,
            y_q3,
            box_w,
            (y_q1 - y_q3).max(1.0),
            b.color,
            0.22,
            Some((b.color, 1.5)),
            2.0,
        );
        svg.line(cx - box_w / 2.0, y_med, cx + box_w / 2.0, y_med, b.color, 2.5);

        // The dots - every point, or else the far values (at most MAX_DOTS,
        // an even sample with the extremes) - or at an edge their number.
        let (mut above, mut below) = (0usize, 0usize);
        let dots = if b.points.is_empty() || !look.dots { &b.outliers } else { &b.points };
        let inside: Vec<f64> = dots
            .iter()
            .copied()
            .filter(|v| {
                let y = y_of(*v);
                if y < top {
                    above += 1;
                } else if y > bottom {
                    below += 1;
                }
                (top..=bottom).contains(&y)
            })
            .collect();
        let all_points = !b.points.is_empty() && look.dots;
        let spread = if all_points { 0.8 } else { 0.5 };
        for (k, v) in sample(&inside, MAX_DOTS).into_iter().enumerate() {
            // A little spread sideways, the same every time; among all the
            // points the far ones darker.
            let jitter = ((k as f64 * 0.618_034).fract() - 0.5) * box_w * spread;
            let far = all_points && (v < b.whisker_low || v > b.whisker_high);
            let (color, opacity) = if far { (theme.text, 0.85) } else { (b.color, 0.55) };
            svg.circle(cx + jitter, y_of(v), 2.6, color, opacity);
        }
        if above > 0 {
            svg.polygon(&[(cx - 4.0, top + 6.0), (cx + 4.0, top + 6.0), (cx, top)], b.color, 0.9);
            svg.text(cx + 7.0, top + 7.0, &format_number(above as f64), theme.text_dim, Anchor::Start, None, false);
        }
        if below > 0 {
            svg.polygon(&[(cx - 4.0, bottom - 6.0), (cx + 4.0, bottom - 6.0), (cx, bottom)], b.color, 0.9);
            svg.text(cx + 7.0, bottom - 1.0, &format_number(below as f64), theme.text_dim, Anchor::Start, None, false);
        }

        // The mean: a ◇ on the box.
        if look.mean {
            let y = y_of(b.mean);
            if (top..=bottom).contains(&y) {
                let r = 4.5;
                svg.polygon(&[(cx, y - r), (cx + r, y), (cx, y + r), (cx - r, y)], theme.text, 0.85);
            }
        }

        // Its name and how many, shortened to its slot - or turned.
        if turned {
            svg.text_turned(cx + font * 0.3, bottom + 10.0, &turned_names[i], theme.text, -50.0);
        } else {
            let name = fit(&b.label, slot - 6.0, theme);
            svg.text(cx, bottom + font + 6.0, &name, theme.text, Anchor::Middle, None, false);
            if look.show_n {
                let count = count_label(b.count, &b.count_noun, slot - 6.0, theme);
                svg.text(cx, bottom + font * 2.0 + 10.0, &count, theme.text_dim, Anchor::Middle, None, false);
            }
        }

        hits.push(Hit { index: i, x: cx - slot / 2.0, y: top, width: slot, height: bottom - top });
    }

    Drawing { svg: svg.finish(), hits }
}

/// At most `max` of `values`, evenly picked with the smallest and the
/// largest (sorted).
fn sample(values: &[f64], max: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() <= max || max < 2 {
        return sorted;
    }
    let last = sorted.len() - 1;
    (0..max).map(|i| sorted[i * last / (max - 1)]).collect()
}

/// "n = 16 images", or as room allows "n = 16", "16", "16k".
fn count_label(count: usize, noun: &str, width: f64, theme: &Theme) -> String {
    let number = format_number(count as f64);
    let long = format!("n = {number}{noun}");
    if theme.text_width(&long) <= width {
        return long;
    }
    let long = format!("n = {number}");
    if theme.text_width(&long) <= width {
        return long;
    }
    if theme.text_width(&number) <= width {
        return number;
    }
    let compact = match count {
        c if c >= 1_000_000 => format!("{}M", c / 1_000_000),
        c if c >= 1_000 => format!("{}k", c / 1_000),
        c => c.to_string(),
    };
    fit(&compact, width, theme)
}

/// `text` shortened with "…" to `width`.
fn fit(text: &str, width: f64, theme: &Theme) -> String {
    if theme.text_width(text) <= width {
        return text.to_string();
    }
    let room = ((width / (theme.font_size * 0.58)).floor() as usize).saturating_sub(1);
    if room == 0 {
        return String::new();
    }
    let mut short: String = text.chars().take(room).collect();
    short.push('…');
    short
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(label: &str, q1: f64, median: f64, q3: f64, lo: f64, hi: f64, outliers: Vec<f64>) -> BoxSpec {
        BoxSpec {
            label: label.into(),
            color: Rgb(0x2a6bdf),
            min: lo.min(outliers.iter().copied().fold(lo, f64::min)),
            q1,
            median,
            q3,
            max: hi.max(outliers.iter().copied().fold(hi, f64::max)),
            mean: median,
            sd: 0.0,
            objects: 100,
            whisker_low: lo,
            whisker_high: hi,
            outliers,
            count: 100,
            points: Vec::new(),
            count_noun: String::new(),
        }
    }

    fn look() -> BoxplotLook {
        BoxplotLook::new("Area µm²")
    }

    #[test]
    fn every_box_has_a_mark_to_click_and_its_name_and_count_below() {
        let boxes = [
            spec("dapi@nucleus", 30.0, 36.0, 44.0, 20.0, 55.0, vec![]),
            spec("ch2@spot", 300.0, 385.0, 470.0, 210.0, 577.0, vec![]),
        ];
        let drawing = draw(&boxes, &look(), &Theme::light(), 600.0, 400.0);
        assert_eq!(drawing.hits.len(), 2);
        assert!(drawing.hits[0].x < drawing.hits[1].x);
        assert!(drawing.svg.contains("dapi@nucleus") && drawing.svg.contains("n = 100"));
        assert!(drawing.svg.contains("Area µm²"));
        assert!(drawing.svg.starts_with("<svg") && drawing.svg.ends_with("</svg>"));
    }

    #[test]
    fn far_values_cut_the_axis_and_are_counted_at_its_edge() {
        let boxes = [spec("a", 10.0, 12.0, 14.0, 8.0, 18.0, vec![5000.0, 6000.0])];
        let drawing = draw(&boxes, &look(), &Theme::light(), 400.0, 300.0);
        // The axis ends near the whiskers, not at 6000.
        assert!(!drawing.svg.contains(">6000<") && !drawing.svg.contains(">6 000<"));
        assert!(drawing.svg.contains("<polygon"), "a mark for the values above");
        assert!(drawing.svg.contains(">2</text>"), "their number");
    }

    #[test]
    fn near_outliers_are_drawn_as_dots() {
        let boxes = [spec("a", 10.0, 12.0, 14.0, 8.0, 18.0, vec![21.0, 22.0])];
        let drawing = draw(&boxes, &look(), &Theme::light(), 400.0, 300.0);
        assert_eq!(drawing.svg.matches("<circle").count(), 2);
        assert!(!drawing.svg.contains("<polygon"));
    }

    #[test]
    fn many_far_values_are_drawn_as_a_sample_with_the_extremes() {
        let many: Vec<f64> = (0..5000).map(|i| 20.0 + i as f64 * 0.001).collect();
        let picked = sample(&many, MAX_DOTS);
        assert_eq!(picked.len(), MAX_DOTS);
        assert_eq!(picked.first(), many.first());
        assert_eq!(picked.last(), many.last());
        let boxes = [spec("a", 10.0, 12.0, 14.0, 8.0, 18.0, many)];
        let drawing = draw(&boxes, &look(), &Theme::light(), 400.0, 300.0);
        assert!(drawing.svg.matches("<circle").count() <= MAX_DOTS);
    }

    #[test]
    fn every_point_is_a_dot_the_far_ones_darker() {
        let mut b = spec("a", 10.0, 12.0, 14.0, 8.0, 18.0, vec![30.0]);
        b.points = vec![9.0, 11.0, 12.0, 13.0, 17.0, 30.0];
        let drawing = draw(&[b], &look(), &Theme::light(), 400.0, 300.0);
        assert_eq!(drawing.svg.matches("<circle").count(), 6);
        let dark = format!("<circle cx=\"");
        assert!(drawing.svg.contains(&dark));
        assert!(drawing.svg.contains(&format!("fill=\"{}\" fill-opacity=\"0.85\"", Theme::light().text.hex())));
    }

    #[test]
    fn the_look_options_change_what_is_drawn() {
        let mut b = spec("a", 10.0, 12.0, 14.0, 8.0, 18.0, vec![21.0]);
        b.points = vec![9.0, 11.0, 13.0, 21.0];
        let boxes = [b];
        let theme = Theme::light();
        // Mean: a diamond (a polygon); no n.
        let mut with = look();
        with.mean = true;
        with.show_n = false;
        let drawing = draw(&boxes, &with, &theme, 400.0, 300.0);
        assert!(drawing.svg.contains("<polygon"));
        assert!(!drawing.svg.contains("n = "));
        // No dots: only the far value.
        let mut no_dots = look();
        no_dots.dots = false;
        assert_eq!(draw(&boxes, &no_dots, &theme, 400.0, 300.0).svg.matches("<circle").count(), 1);
        // Min - max whiskers: nothing is far, no dots without the points.
        let mut min_max = no_dots.clone();
        min_max.min_max = true;
        assert_eq!(draw(&boxes, &min_max, &theme, 400.0, 300.0).svg.matches("<circle").count(), 0);
    }

    #[test]
    fn names_that_do_not_fit_are_turned() {
        let boxes: Vec<BoxSpec> = (0..60)
            .map(|i| spec(&format!("well A{i}"), 10.0, 12.0, 14.0, 8.0, 18.0, vec![]))
            .collect();
        let drawing = draw(&boxes, &look(), &Theme::light(), 800.0, 400.0);
        assert!(drawing.svg.contains("rotate(-50)"));
        assert!(drawing.svg.contains(">well A59<"), "every name drawn");
        assert!(!drawing.svg.contains("n = "), "no counts under turned names");
    }

    #[test]
    fn counts_shorten_to_their_slot() {
        let theme = Theme::light();
        assert_eq!(count_label(16, " images", 200.0, &theme), "n = 16 images");
        assert_eq!(count_label(78_349, "", 200.0, &theme), "n = 78349");
        assert_eq!(count_label(78_349, "", 40.0, &theme), "78349");
        assert_eq!(count_label(78_349, "", 22.0, &theme), "78k");
    }

    #[test]
    fn long_names_are_shortened_to_their_slot() {
        let theme = Theme::light();
        let short = fit("ch2@ in-cell-spots coloc ch1and3", 60.0, &theme);
        assert!(short.ends_with('…') && theme.text_width(&short) <= 60.0 + theme.font_size);
    }
}
