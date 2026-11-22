//! # histogram
//!
//! How many values fall in each bin (design canvas "Results rework",
//! ChartKinds): bars on a count axis - linear or log - over the value's
//! axis, the bins' range ticked inside. Values beyond the bins (a range of
//! the 1st to 99th percentile) are counted at the edges: "‹ 41 below",
//! "312 above (to 75400) ›".
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

/// The bins: `edges` has one more entry than `counts`.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSpec {
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub color: Rgb,
    /// Values below and above the bins, and the smallest and the largest.
    pub below: u64,
    pub above: u64,
    pub data_min: f64,
    pub data_max: f64,
}

/// How the histogram is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramLook {
    /// The value's name with its unit ("Area µm²").
    pub x_title: String,
    /// What is counted ("Objects").
    pub y_title: String,
    pub log_counts: bool,
    pub grid: bool,
}

/// The histogram of `spec` as a `width` × `height` drawing - a mark per
/// bin.
pub fn draw(
    spec: &HistogramSpec,
    look: &HistogramLook,
    theme: &Theme,
    width: f64,
    height: f64,
) -> Drawing {
    let mut svg = Svg::new(width, height, theme);
    let bins = spec.counts.len();
    if bins == 0 || spec.edges.len() != bins + 1 || width < 80.0 || height < 80.0 {
        return Drawing {
            svg: svg.finish(),
            hits: Vec::new(),
        };
    }
    let font = theme.font_size;
    let top = 16.0 + font + 6.0;
    let bottom = height - (font * 2.0 + 20.0);

    let max_count = spec.counts.iter().copied().max().unwrap_or(0).max(1) as f64;
    let y_ticks_wanted = ((bottom - top) / 56.0).round().clamp(2.0, 10.0) as usize;
    let y_scale = if look.log_counts {
        // The axis starts below 1 (at ½, under its first tick): a bin with
        // a single value still has a bar.
        Scale {
            min: 0.5,
            ..Scale::nice(1.0, max_count, y_ticks_wanted, true)
        }
    } else {
        Scale::nice(0.0, max_count, y_ticks_wanted, false)
    };
    let y_ticks = y_scale.ticks(y_ticks_wanted);
    let label_w = y_ticks
        .iter()
        .map(|(_, label)| theme.text_width(label))
        .fold(0.0, f64::max);
    let left = 8.0 + font + 10.0 + label_w + 8.0;
    let right = width - 16.0;
    let y_of = |v: f64| bottom - y_scale.position(v).clamp(0.0, 1.0) * (bottom - top);

    let (lo, hi) = (spec.edges[0], spec.edges[bins]);
    let x_ticks_wanted = ((right - left) / 90.0).round().clamp(2.0, 12.0) as usize;
    let x_scale = Scale::exact(lo, hi, x_ticks_wanted);
    let x_of = |v: f64| left + x_scale.position(v) * (right - left);

    for (value, label) in &y_ticks {
        let y = y_of(*value);
        if look.grid {
            svg.line(left, y, right, y, theme.grid, 1.0);
        }
        svg.line(left - 4.0, y, left, y, theme.axis, 1.0);
        svg.text(
            left - 7.0,
            y + font * 0.35,
            label,
            theme.text_dim,
            Anchor::End,
            None,
            false,
        );
    }
    for (value, label) in x_scale.ticks(x_ticks_wanted) {
        let x = x_of(value);
        svg.line(x, bottom, x, bottom + 4.0, theme.axis, 1.0);
        svg.text(
            x,
            bottom + font + 6.0,
            &label,
            theme.text_dim,
            Anchor::Middle,
            None,
            false,
        );
    }

    // The bars, a mark each.
    let mut hits = Vec::with_capacity(bins);
    let gap = if (right - left) / bins as f64 > 6.0 {
        1.0
    } else {
        0.0
    };
    for (i, count) in spec.counts.iter().enumerate() {
        let (x0, x1) = (x_of(spec.edges[i]), x_of(spec.edges[i + 1]));
        if *count > 0 {
            let y = y_of(*count as f64);
            svg.rect(
                x0 + gap / 2.0,
                y,
                (x1 - x0 - gap).max(0.5),
                bottom - y,
                spec.color,
                0.85,
                None,
                1.0,
            );
        }
        hits.push(Hit {
            index: i,
            x: x0,
            y: top,
            width: x1 - x0,
            height: bottom - top,
        });
    }

    svg.line(left, top, left, bottom, theme.axis, 1.0);
    svg.line(left, bottom, right, bottom, theme.axis, 1.0);
    svg.text_vertical(
        8.0 + font * 0.5,
        (top + bottom) / 2.0,
        &look.y_title,
        theme.text_dim,
    );
    svg.text(
        (left + right) / 2.0,
        height - 6.0,
        &look.x_title,
        theme.text_dim,
        Anchor::Middle,
        None,
        false,
    );

    // The values beyond the bins.
    if spec.below > 0 {
        let text = format!(
            "‹ {} below (from {})",
            format_number(spec.below as f64),
            format_number(spec.data_min)
        );
        svg.text(
            left + 4.0,
            16.0,
            &text,
            theme.text_dim,
            Anchor::Start,
            None,
            false,
        );
    }
    if spec.above > 0 {
        let text = format!(
            "{} above (to {}) ›",
            format_number(spec.above as f64),
            format_number(spec.data_max)
        );
        svg.text(right, 16.0, &text, theme.text_dim, Anchor::End, None, false);
    }

    Drawing {
        svg: svg.finish(),
        hits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(counts: Vec<u64>, below: u64, above: u64) -> HistogramSpec {
        let edges = (0..=counts.len()).map(|i| 40.0 + i as f64 * 10.0).collect();
        HistogramSpec {
            edges,
            counts,
            color: Rgb(0x2a6bdf),
            below,
            above,
            data_min: 1.0,
            data_max: 75_400.0,
        }
    }

    fn look() -> HistogramLook {
        HistogramLook {
            x_title: "Area µm²".into(),
            y_title: "Objects".into(),
            log_counts: false,
            grid: true,
        }
    }

    #[test]
    fn every_bin_is_a_mark_and_empty_bins_draw_no_bar() {
        let drawing = draw(
            &spec(vec![3, 0, 7, 1], 0, 0),
            &look(),
            &Theme::light(),
            500.0,
            300.0,
        );
        assert_eq!(drawing.hits.len(), 4);
        // Bars: three; the background has no rect.
        assert_eq!(drawing.svg.matches("<rect").count(), 3);
        assert!(drawing.svg.contains("Area µm²") && drawing.svg.contains("Objects"));
    }

    #[test]
    fn values_beyond_the_bins_are_counted_at_the_edges() {
        let drawing = draw(
            &spec(vec![3, 5], 41, 312),
            &look(),
            &Theme::light(),
            500.0,
            300.0,
        );
        assert!(drawing.svg.contains("‹ 41 below"));
        assert!(drawing.svg.contains("312 above (to 75400) ›"));
    }

    #[test]
    fn a_log_count_axis_ticks_in_decades() {
        let mut log = look();
        log.log_counts = true;
        let drawing = draw(
            &spec(vec![2, 900, 30], 0, 0),
            &log,
            &Theme::light(),
            500.0,
            300.0,
        );
        assert!(drawing.svg.contains(">1000<") && drawing.svg.contains(">10<"));
    }

    #[test]
    fn a_bin_with_one_value_has_a_bar_on_a_log_axis() {
        let mut log = look();
        log.log_counts = true;
        let drawing = draw(
            &spec(vec![1, 0, 40], 0, 0),
            &log,
            &Theme::light(),
            500.0,
            300.0,
        );
        let heights: Vec<f64> = drawing
            .svg
            .split("<rect ")
            .skip(1)
            .filter(|rect| rect.contains("fill-opacity=\"0.85\""))
            .map(|rect| {
                let at = rect.find("height=\"").unwrap() + 8;
                rect[at..at + rect[at..].find('"').unwrap()]
                    .parse()
                    .unwrap()
            })
            .collect();
        assert_eq!(heights.len(), 2, "the bins with values: {heights:?}");
        assert!(heights[0] > 5.0 && heights[0] < heights[1], "{heights:?}");
        assert!(drawing.svg.contains(">1<"), "the first tick stays 1");
    }
}
