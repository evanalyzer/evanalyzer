//! # lines
//!
//! Values along a position (design canvas "Results rework", ChartKinds):
//! a line per series - over time, or along a profile's radius - with its
//! spread as a band (± SD or the IQR), faint individual tracks behind, a
//! highlighted curve (the selected object's) with a mark at its highest
//! value, a legend on top and both axes ticked.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::svg::{Anchor, Svg};
use super::{Drawing, Rgb, Scale, Theme};

/// A point of a series: position, centre, low and high of its band (none:
/// no value there).
pub type LinePoint = (f64, Option<f64>, Option<f64>, Option<f64>);

/// One series.
#[derive(Debug, Clone, PartialEq)]
pub struct LineSeries {
    pub label: String,
    pub color: Rgb,
    pub points: Vec<LinePoint>,
}

/// What a lines chart shows.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LinesSpec {
    pub series: Vec<LineSeries>,
    /// Faint lines behind (e.g. every track), `(position, value)`.
    pub faint: Vec<Vec<(f64, Option<f64>)>>,
    /// A curve on top (the selected object's).
    pub highlight: Vec<(f64, f64)>,
    /// A mark across the plot at a position, and what it says.
    pub marker: Option<(f64, String)>,
    pub x_title: String,
    pub y_title: String,
    /// The value axis starts at 0 (a profile's values).
    pub from_zero: bool,
    /// A note in the corner ("8 of 20 lines shown").
    pub note: String,
}

/// The lines of `spec` as a `width` × `height` drawing (`grid`: grid
/// lines; `highlight`: the selected curve's colour).
pub fn draw(
    spec: &LinesSpec,
    grid: bool,
    highlight: Rgb,
    theme: &Theme,
    width: f64,
    height: f64,
) -> Drawing {
    let mut svg = Svg::new(width, height, theme);
    if spec.series.is_empty() || width < 80.0 || height < 80.0 {
        return Drawing {
            svg: svg.finish(),
            hits: Vec::new(),
        };
    }
    let font = theme.font_size;
    // The legend on top: as many names as fit on a line.
    let mut legend: Vec<(f64, &LineSeries)> = Vec::new();
    let mut x_cursor = 0.0;
    let legend_room = width - 40.0 - theme.text_width(&spec.note);
    for series in &spec.series {
        let w = 20.0 + theme.text_width(&series.label) + 14.0;
        if x_cursor + w > legend_room {
            break;
        }
        legend.push((x_cursor, series));
        x_cursor += w;
    }
    let top = 16.0 + font + 10.0;
    let bottom = height - (font * 2.0 + 20.0);

    let xs = spec
        .series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.0))
        .chain(spec.highlight.iter().map(|p| p.0));
    let (x_lo, x_hi) = xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let ys = spec
        .series
        .iter()
        .flat_map(|s| s.points.iter().flat_map(|p| [p.1, p.2, p.3]))
        .chain(spec.faint.iter().flat_map(|f| f.iter().map(|p| p.1)))
        .flatten()
        .chain(spec.highlight.iter().map(|p| p.1));
    let (mut y_lo, y_hi) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if !x_lo.is_finite() || !y_lo.is_finite() {
        return Drawing {
            svg: svg.finish(),
            hits: Vec::new(),
        };
    }
    if spec.from_zero {
        y_lo = y_lo.min(0.0);
    }
    let y_wanted = ((bottom - top) / 56.0).round().clamp(2.0, 10.0) as usize;
    let y_scale = Scale::nice(y_lo, y_hi, y_wanted, false);
    let y_ticks = y_scale.ticks(y_wanted);
    let label_w = y_ticks
        .iter()
        .map(|(_, label)| theme.text_width(label))
        .fold(0.0, f64::max);
    let left = 8.0 + font + 10.0 + label_w + 8.0;
    let right = width - 16.0;
    let x_wanted = ((right - left) / 90.0).round().clamp(2.0, 12.0) as usize;
    let x_scale = Scale::exact(x_lo, x_hi, x_wanted);
    let x_of = |v: f64| left + x_scale.position(v) * (right - left);
    let y_of = |v: f64| bottom - y_scale.position(v) * (bottom - top);

    for (value, label) in &y_ticks {
        let y = y_of(*value);
        if grid {
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
    for (value, label) in x_scale.ticks(x_wanted) {
        let x = x_of(value);
        if grid {
            svg.line(x, top, x, bottom, theme.grid, 1.0);
        }
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

    // Behind: the faint lines; then each series' band and line.
    for faint in &spec.faint {
        for run in runs(faint.iter().map(|(x, y)| y.map(|y| (x_of(*x), y_of(y))))) {
            svg.polyline(&run, theme.text_dim, 1.0, 0.25, false);
        }
    }
    for series in &spec.series {
        let band: Vec<(f64, f64, f64)> = series
            .points
            .iter()
            .filter_map(|(x, _, lo, hi)| Some((x_of(*x), y_of((*lo)?), y_of((*hi)?))))
            .collect();
        if band.len() >= 2 {
            let mut outline: Vec<(f64, f64)> = band.iter().map(|(x, _, hi)| (*x, *hi)).collect();
            outline.extend(band.iter().rev().map(|(x, lo, _)| (*x, *lo)));
            svg.polygon(&outline, series.color, 0.16);
        }
        for run in runs(
            series
                .points
                .iter()
                .map(|(x, c, _, _)| c.map(|c| (x_of(*x), y_of(c)))),
        ) {
            match run.as_slice() {
                // A value between gaps: a dot.
                [(x, y)] => svg.circle(*x, *y, 2.5, series.color, 1.0),
                _ => svg.polyline(&run, series.color, 2.0, 1.0, false),
            }
        }
    }
    if let Some((position, label)) = &spec.marker {
        let x = x_of(*position);
        svg.polyline(&[(x, top), (x, bottom)], highlight, 1.0, 0.7, true);
        svg.text(
            left + 8.0,
            top + font + 2.0,
            label,
            highlight,
            Anchor::Start,
            None,
            false,
        );
    }
    if spec.highlight.len() >= 2 {
        let curve: Vec<(f64, f64)> = spec
            .highlight
            .iter()
            .map(|(x, y)| (x_of(*x), y_of(*y)))
            .collect();
        svg.polyline(&curve, highlight, 2.5, 1.0, false);
    }

    svg.line(left, top, left, bottom, theme.axis, 1.0);
    svg.line(left, bottom, right, bottom, theme.axis, 1.0);
    svg.text_vertical(
        8.0 + font * 0.5,
        (top + bottom) / 2.0,
        &spec.y_title,
        theme.text_dim,
    );
    svg.text(
        (left + right) / 2.0,
        height - 6.0,
        &spec.x_title,
        theme.text_dim,
        Anchor::Middle,
        None,
        false,
    );
    for (offset, series) in legend {
        let x = left + offset;
        svg.rect(x, 16.0 - 4.0, 14.0, 3.0, series.color, 1.0, None, 1.0);
        svg.text(
            x + 20.0,
            16.0,
            &series.label,
            theme.text_dim,
            Anchor::Start,
            None,
            false,
        );
    }
    if !spec.note.is_empty() {
        svg.text(
            right,
            16.0,
            &spec.note,
            theme.text_dim,
            Anchor::End,
            None,
            false,
        );
    }
    Drawing {
        svg: svg.finish(),
        hits: Vec::new(),
    }
}

/// The runs of a line between its gaps (no value).
fn runs(points: impl Iterator<Item = Option<(f64, f64)>>) -> Vec<Vec<(f64, f64)>> {
    let mut all = vec![Vec::new()];
    for point in points {
        match point {
            Some(p) => all.last_mut().expect("one run").push(p),
            None if all.last().is_some_and(|run| !run.is_empty()) => all.push(Vec::new()),
            None => {}
        }
    }
    all.retain(|run| !run.is_empty());
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> LinesSpec {
        LinesSpec {
            series: vec![
                LineSeries {
                    label: "Nuclei".into(),
                    color: Rgb(0x2a6bdf),
                    points: vec![
                        (0.0, Some(10.0), Some(8.0), Some(12.0)),
                        (5.0, None, None, None),
                        (10.0, Some(14.0), Some(12.0), Some(16.0)),
                        (15.0, Some(15.0), Some(13.0), Some(17.0)),
                    ],
                },
                LineSeries {
                    label: "Cells".into(),
                    color: Rgb(0x7b4fc9),
                    points: vec![
                        (0.0, Some(30.0), None, None),
                        (15.0, Some(25.0), None, None),
                    ],
                },
            ],
            x_title: "time (min)".into(),
            y_title: "objects per image".into(),
            note: "2 of 9 lines shown".into(),
            ..LinesSpec::default()
        }
    }

    #[test]
    fn a_line_per_series_broken_at_gaps_with_a_band_and_a_legend() {
        let drawing = draw(&spec(), true, Rgb(0x2a6bdf), &Theme::light(), 600.0, 400.0);
        // Nuclei: a lone first value (a dot), the gap at 5, a run; Cells: one.
        assert_eq!(drawing.svg.matches("<polyline").count(), 2);
        assert_eq!(drawing.svg.matches("<circle").count(), 1);
        assert_eq!(
            drawing.svg.matches("<polygon").count(),
            1,
            "only Nuclei has a band"
        );
        assert!(drawing.svg.contains(">Nuclei<") && drawing.svg.contains(">Cells<"));
        assert!(drawing.svg.contains("2 of 9 lines shown"));
        assert!(drawing.svg.contains("time (min)"));
    }

    #[test]
    fn a_marker_and_the_highlighted_curve_are_drawn_on_top() {
        let mut with = spec();
        with.highlight = vec![(0.0, 11.0), (15.0, 18.0)];
        with.marker = Some((15.0, "highest 18 at 15".into()));
        let drawing = draw(&with, false, Rgb(0x2a6bdf), &Theme::light(), 600.0, 400.0);
        assert!(drawing.svg.contains("stroke-dasharray"));
        assert!(drawing.svg.contains("highest 18 at 15"));
    }
}
