//! Histogram and scatter charts over the results table's object data.
//!
//! Mirrors `results_loader.rs`'s split: plain data structs + pure functions
//! first (unit-testable without any plotting dependency), then a thin
//! `plotters` rendering layer. The two public entry points per chart type —
//! `render_*` (into an in-memory RGB buffer, for the GUI) and `save_*_png`
//! (straight to a file, for a future CLI command) — call the *same* internal
//! `draw_*` function on a different `BitMapBackend`, so there is exactly one
//! drawing implementation per chart type regardless of caller.
use crate::results::results_loader::{
    ColumnSpec, coloc_label, compute_class, is_colocalized, is_numeric_metric, numeric_value,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::ObjectRow;
use plotters::coord::ReverseCoordTranslate;
use plotters::coord::ranged1d::ReversibleRanged;
use plotters::prelude::*;
use std::path::Path;

// ---------------------------------------------------------------------------
// Data computation
// ---------------------------------------------------------------------------

/// Which categorical dimension scatter points are colored by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorBy {
    None,
    Class,
    Colocalized,
}

/// Columns that can be plotted on a numeric axis — the same metrics
/// `aggregate_rows` can aggregate (see `is_numeric_metric`), filtered to
/// those currently visible. Backs the GUI's column/axis pickers.
pub fn plottable_columns(columns: &[ColumnSpec]) -> Vec<&ColumnSpec> {
    columns
        .iter()
        .filter(|c| c.visible && is_numeric_metric(&c.id))
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBucket {
    pub range_start: f64,
    pub range_end: f64,
    pub count: usize,
}

/// One "color by" group's bucket counts — e.g. one series per class, or one
/// per colocalized status. Every series in a given [`HistogramData`] shares
/// the exact same bucket edges (computed once, from every group's values
/// combined), so overlaying them is a direct, aligned comparison.
#[derive(Debug, Clone)]
pub struct HistogramSeries {
    /// Group label (class name / "Yes" / "No"), or `""` when not grouped
    /// (`ColorBy::None`) — `HistogramData::series` holds exactly one series
    /// in that case.
    pub label: String,
    pub buckets: Vec<HistogramBucket>,
}

#[derive(Debug, Clone)]
pub struct HistogramData {
    pub column_label: String,
    /// One series per "color by" group, all sharing one set of bucket edges.
    /// Never empty when `compute_histogram` returns `Some`.
    pub series: Vec<HistogramSeries>,
    /// Whether bucket edges were spaced equally in log space rather than
    /// linear space — drives `draw_histogram`'s choice of axis. Bucket edges
    /// are always reported back in real (linear) units either way.
    pub log_scale: bool,
    /// Values `<= 0` can't be log-binned and are dropped; this is how many
    /// were, so the GUI can surface a "N values excluded" notice. Always 0
    /// when `log_scale` is false.
    pub excluded_non_positive: usize,
}

impl HistogramData {
    /// Convenience for the common ungrouped case (`ColorBy::None`), where
    /// there's exactly one series: its buckets directly. Panics if `series`
    /// is empty, which `compute_histogram` never produces.
    pub fn buckets(&self) -> &[HistogramBucket] {
        &self.series[0].buckets
    }
}

/// The transformed-space `[min, max]` of `values` under `to_space` — the
/// shared reference range multiple value subsets (e.g. one per "color by"
/// group) can all be binned against, so their bars land on identical edges
/// and can be overlaid for direct comparison.
fn transformed_range(values: &[f64], to_space: &impl Fn(f64) -> f64) -> (f64, f64) {
    let transformed = values.iter().map(|&v| to_space(v));
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for t in transformed {
        min = min.min(t);
        max = max.max(t);
    }
    (min, max)
}

/// Buckets `values` into `bucket_count` equal-width buckets over an
/// already-known transformed-space `[min, max]` range (see
/// `transformed_range`), reporting real-world edges back via `from_space`.
fn bucket_into_range(
    values: &[f64],
    min: f64,
    max: f64,
    bucket_count: usize,
    to_space: &impl Fn(f64) -> f64,
    from_space: &impl Fn(f64) -> f64,
) -> Vec<HistogramBucket> {
    if max <= min {
        // Every value (in the *reference* range) is identical — one bucket
        // holds this subset's count (0 if this subset itself has no values).
        let edge = from_space(min);
        return vec![HistogramBucket {
            range_start: edge,
            range_end: edge,
            count: values.len(),
        }];
    }

    let width = (max - min) / bucket_count as f64;
    let mut counts = vec![0usize; bucket_count];
    for &v in values {
        let t = to_space(v);
        // The max value lands exactly on the last bucket's upper edge;
        // clamp it into that bucket instead of overflowing.
        let idx = (((t - min) / width) as usize).min(bucket_count - 1);
        counts[idx] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| HistogramBucket {
            range_start: from_space(min + i as f64 * width),
            range_end: from_space(min + (i + 1) as f64 * width),
            count,
        })
        .collect()
}

/// Buckets every object's `column_id` value into `bucket_count` buckets spanning
/// [min, max]. `None` if the column isn't found, no object has a value for it,
/// or `bucket_count` is 0.
///
/// `log_scale` spaces buckets equally in log space instead of linear space —
/// the fix for right-skewed data (e.g. cell area) where a few large outliers
/// otherwise crowd almost every value into the first one or two buckets.
/// Values `<= 0` can't be log-binned and are excluded (see
/// `HistogramData::excluded_non_positive`).
pub fn compute_histogram(
    objects: &[ObjectRow],
    column_id: &str,
    columns: &[ColumnSpec],
    bucket_count: usize,
    log_scale: bool,
    color_by: ColorBy,
) -> Option<HistogramData> {
    if bucket_count == 0 {
        return None;
    }
    let column_label = columns.iter().find(|c| c.id == column_id)?.label.clone();

    // (value, group label) pairs; group label is "" for every row when
    // `color_by == None`, so they all land in one series below.
    let mut pairs: Vec<(f64, String)> = objects
        .iter()
        .filter_map(|r| {
            let v = numeric_value(r, column_id)?;
            let group = match color_by {
                ColorBy::None => String::new(),
                ColorBy::Class => compute_class(r),
                ColorBy::Colocalized => coloc_label(is_colocalized(&r.coloc_json)),
            };
            Some((v, group))
        })
        .collect();

    let excluded_non_positive = if log_scale {
        let before = pairs.len();
        pairs.retain(|(v, _)| *v > 0.0);
        before - pairs.len()
    } else {
        0
    };

    if pairs.is_empty() {
        return None;
    }

    let to_space: fn(f64) -> f64 = if log_scale { f64::ln } else { |v| v };
    let from_space: fn(f64) -> f64 = if log_scale { f64::exp } else { |v| v };

    // Every series shares these edges (computed from *every* group's values
    // combined) so overlaid bars land at the same x positions.
    let all_values: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    let (edge_min, edge_max) = transformed_range(&all_values, &to_space);

    let mut group_labels: Vec<String> = pairs.iter().map(|(_, g)| g.clone()).collect();
    group_labels.sort();
    group_labels.dedup();

    let series = group_labels
        .into_iter()
        .map(|label| {
            let group_values: Vec<f64> = pairs
                .iter()
                .filter(|(_, g)| *g == label)
                .map(|(v, _)| *v)
                .collect();
            let buckets = bucket_into_range(
                &group_values,
                edge_min,
                edge_max,
                bucket_count,
                &to_space,
                &from_space,
            );
            HistogramSeries { label, buckets }
        })
        .collect();

    Some(HistogramData {
        column_label,
        series,
        log_scale,
        excluded_non_positive,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
    /// The point's class/colocalized label (see `ColorBy`), or `None` when
    /// `color_by` is `ColorBy::None`.
    pub group: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ScatterData {
    pub x_label: String,
    pub y_label: String,
    pub points: Vec<ScatterPoint>,
    /// The original point count, set only when `compute_scatter` had to
    /// subsample down to `max_points` — drives the GUI's "showing N of M
    /// points" notice.
    pub sampled_from: Option<usize>,
}

/// Pairs each object's `x_col`/`y_col` values into a scatter point (skipping
/// ROIs missing either value), optionally tagging each point with its class
/// or colocalized label for coloring. `None` if either column isn't found or
/// no object has both values.
///
/// When there are more than `max_points` points, deterministically samples an
/// evenly-strided subset (not random) so re-rendering the same filter is
/// stable. `max_points == 0` disables the cap.
pub fn compute_scatter(
    objects: &[ObjectRow],
    x_col: &str,
    y_col: &str,
    color_by: ColorBy,
    columns: &[ColumnSpec],
    max_points: usize,
) -> Option<ScatterData> {
    let x_label = columns.iter().find(|c| c.id == x_col)?.label.clone();
    let y_label = columns.iter().find(|c| c.id == y_col)?.label.clone();

    let mut points: Vec<ScatterPoint> = objects
        .iter()
        .filter_map(|object| {
            let x = numeric_value(object, x_col)?;
            let y = numeric_value(object, y_col)?;
            let group = match color_by {
                ColorBy::None => None,
                ColorBy::Class => Some(compute_class(object)),
                ColorBy::Colocalized => Some(coloc_label(is_colocalized(&object.coloc_json))),
            };
            Some(ScatterPoint { x, y, group })
        })
        .collect();

    if points.is_empty() {
        return None;
    }

    let sampled_from = if max_points > 0 && points.len() > max_points {
        let total = points.len();
        let stride = total as f64 / max_points as f64;
        points = (0..max_points)
            .map(|i| points[((i as f64 * stride) as usize).min(total - 1)].clone())
            .collect();
        Some(total)
    } else {
        None
    };

    Some(ScatterData {
        x_label,
        y_label,
        points,
        sampled_from,
    })
}

/// What each heatmap cell's color encodes.
#[derive(Debug, Clone, PartialEq)]
pub enum HeatmapMetric {
    /// Number of ROIs whose centroid falls in the cell.
    Count,
    /// Average value of the named numeric column, over ROIs whose centroid
    /// falls in the cell (cells with no value end up at 0).
    Average(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatmapCell {
    /// ROIs whose centroid falls in this cell — 0 means "no data", drawn as
    /// background rather than the low end of the color scale.
    pub count: usize,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct HeatmapData {
    pub x_label: String,
    pub y_label: String,
    pub value_label: String,
    pub cols: usize,
    pub rows: usize,
    /// Raster cell size, in image pixels (square cells, same for both axes).
    pub cell_size: f64,
    pub x_min: f64,
    pub y_min: f64,
    /// Row-major: `cells[row * cols + col]`.
    pub cells: Vec<HeatmapCell>,
}

/// Bins every object's centroid (`centroid_x_px`/`centroid_y_px`) into a grid of
/// `cell_size_px`-wide square cells spanning the ROIs' bounding box, and
/// colors each cell either by how many ROIs landed in it (`Count`) or by the
/// average of a numeric column over those ROIs (`Average`) — e.g. "how many
/// nuclei per region" vs. "average channel intensity per region".
///
/// `None` if `objects` is empty, `cell_size_px <= 0`, or (for `Average`) the
/// named column isn't found.
pub fn compute_heatmap(
    objects: &[ObjectRow],
    metric: &HeatmapMetric,
    columns: &[ColumnSpec],
    cell_size_px: f64,
) -> Option<HeatmapData> {
    if objects.is_empty() || cell_size_px <= 0.0 {
        return None;
    }
    let value_label = match metric {
        HeatmapMetric::Count => "Count".to_string(),
        HeatmapMetric::Average(col_id) => columns.iter().find(|c| c.id == *col_id)?.label.clone(),
    };

    let x_min = objects
        .iter()
        .map(|r| r.centroid_x_px)
        .fold(f64::INFINITY, f64::min);
    let x_max = objects
        .iter()
        .map(|r| r.centroid_x_px)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = objects
        .iter()
        .map(|r| r.centroid_y_px)
        .fold(f64::INFINITY, f64::min);
    let y_max = objects
        .iter()
        .map(|r| r.centroid_y_px)
        .fold(f64::NEG_INFINITY, f64::max);

    let cols = (((x_max - x_min) / cell_size_px) as usize) + 1;
    let rows = (((y_max - y_min) / cell_size_px) as usize) + 1;

    let mut counts = vec![0usize; cols * rows];
    let mut sums = vec![0f64; cols * rows];
    for object in objects {
        let ix = (((object.centroid_x_px - x_min) / cell_size_px) as usize).min(cols - 1);
        let iy = (((object.centroid_y_px - y_min) / cell_size_px) as usize).min(rows - 1);
        let idx = iy * cols + ix;
        match metric {
            HeatmapMetric::Count => counts[idx] += 1,
            HeatmapMetric::Average(col_id) => {
                if let Some(v) = numeric_value(object, col_id) {
                    counts[idx] += 1;
                    sums[idx] += v;
                }
            }
        }
    }

    let cells = (0..cols * rows)
        .map(|i| {
            let count = counts[i];
            let value = match metric {
                HeatmapMetric::Count => count as f64,
                HeatmapMetric::Average(_) => {
                    if count > 0 {
                        sums[i] / count as f64
                    } else {
                        0.0
                    }
                }
            };
            HeatmapCell { count, value }
        })
        .collect();

    Some(HeatmapData {
        x_label: "X position (px)".to_string(),
        y_label: "Y position (px)".to_string(),
        value_label,
        cols,
        rows,
        cell_size: cell_size_px,
        x_min,
        y_min,
        cells,
    })
}

// ---------------------------------------------------------------------------
// Rendering (plotters)
// ---------------------------------------------------------------------------

/// A linear pixel<->data mapping derived from two known correspondences
/// (the plotting area's top-left and bottom-right corners, reverse-mapped
/// through plotters' own coordinate spec at render time). Capturing it this
/// way — rather than hand-deriving the axis math ourselves — means we don't
/// have to know which way plotters flips the y-axis (it differs between the
/// heatmap, which flips a second time for image-style coordinates, and
/// everything else); whatever the mapping is, two points pin it down.
#[derive(Debug, Clone, Copy)]
struct PixelToData {
    px0: f64,
    py0: f64,
    dx0: f64,
    dy0: f64,
    px1: f64,
    py1: f64,
    dx1: f64,
    dy1: f64,
}

impl PixelToData {
    fn from_chart<DB: DrawingBackend, X, Y>(
        chart: &ChartContext<'_, DB, Cartesian2d<X, Y>>,
    ) -> Option<Self>
    where
        X: Ranged<ValueType = f64> + ReversibleRanged,
        Y: Ranged<ValueType = f64> + ReversibleRanged,
    {
        let (px_range, py_range) = chart.plotting_area().get_pixel_range();
        // `get_pixel_range()` gives exclusive `Range<i32>` ends; `reverse_translate`
        // only accepts pixels actually inside the backend rect, so probe the last
        // valid pixel (`end - 1`), not the one-past-the-end exclusive bound.
        let (px0, py0) = (px_range.start, py_range.start);
        let (px1, py1) = (px_range.end - 1, py_range.end - 1);
        let (dx0, dy0) = chart.as_coord_spec().reverse_translate((px0, py0))?;
        let (dx1, dy1) = chart.as_coord_spec().reverse_translate((px1, py1))?;
        Some(Self {
            px0: px0 as f64,
            py0: py0 as f64,
            dx0,
            dy0,
            px1: px1 as f64,
            py1: py1 as f64,
            dx1,
            dy1,
        })
    }

    fn contains_pixel(&self, px: f64, py: f64) -> bool {
        let (lo_x, hi_x) = (self.px0.min(self.px1), self.px0.max(self.px1));
        let (lo_y, hi_y) = (self.py0.min(self.py1), self.py0.max(self.py1));
        (lo_x..=hi_x).contains(&px) && (lo_y..=hi_y).contains(&py)
    }

    fn data_at(&self, px: f64, py: f64) -> (f64, f64) {
        let fx = if self.px1 != self.px0 {
            (px - self.px0) / (self.px1 - self.px0)
        } else {
            0.0
        };
        let fy = if self.py1 != self.py0 {
            (py - self.py0) / (self.py1 - self.py0)
        } else {
            0.0
        };
        (
            self.dx0 + fx * (self.dx1 - self.dx0),
            self.dy0 + fy * (self.dy1 - self.dy0),
        )
    }

    fn pixel_at(&self, dx: f64, dy: f64) -> (f64, f64) {
        let fx = if self.dx1 != self.dx0 {
            (dx - self.dx0) / (self.dx1 - self.dx0)
        } else {
            0.0
        };
        let fy = if self.dy1 != self.dy0 {
            (dy - self.dy0) / (self.dy1 - self.dy0)
        } else {
            0.0
        };
        (
            self.px0 + fx * (self.px1 - self.px0),
            self.py0 + fy * (self.py1 - self.py0),
        )
    }
}

#[derive(Debug, Clone)]
enum HitTestData {
    Histogram {
        series: Vec<HistogramSeries>,
        log_scale: bool,
        column_label: String,
    },
    Scatter {
        points: Vec<ScatterPoint>,
        x_label: String,
        y_label: String,
    },
    Heatmap {
        cols: usize,
        rows: usize,
        cell_size: f64,
        x_min: f64,
        y_min: f64,
        cells: Vec<HeatmapCell>,
        value_label: String,
    },
}

/// Answers "what's under the mouse" for a rendered chart — built once at
/// render time (see `draw_histogram`/`draw_scatter`/`draw_heatmap`) from the
/// same data already being plotted, so the GUI can show a hover tooltip
/// without re-querying the database or re-deriving any plotting math.
#[derive(Debug, Clone)]
pub struct ChartHitTester {
    mapping: PixelToData,
    data: HitTestData,
}

impl ChartHitTester {
    /// `px`/`py` are pixel coordinates within the rendered bitmap (same
    /// space as `RenderedChart.width`/`height`). Returns a short
    /// human-readable description of whatever's at that point, or `None`
    /// when the point is outside the plot area / not near any data (e.g. a
    /// scatter point, or a heatmap cell with no ROIs).
    pub fn hit_test(&self, px: f64, py: f64) -> Option<String> {
        if !self.mapping.contains_pixel(px, py) {
            return None;
        }
        let (data_x, data_y) = self.mapping.data_at(px, py);
        match &self.data {
            HitTestData::Histogram {
                series,
                log_scale,
                column_label,
            } => {
                let real_x = if *log_scale { data_x.exp() } else { data_x };
                // Every series shares identical bucket edges (see
                // `compute_histogram`), so finding the hit bucket in the
                // first series also locates it — by index — in every other.
                let first = series.first()?;
                let bucket_idx = first
                    .buckets
                    .iter()
                    .position(|b| real_x >= b.range_start && real_x < b.range_end)
                    .or_else(|| {
                        first
                            .buckets
                            .last()
                            .filter(|b| real_x >= b.range_start)
                            .map(|_| first.buckets.len() - 1)
                    })?;
                let bucket = &first.buckets[bucket_idx];
                let mut text = format!(
                    "{}: {} – {}",
                    column_label,
                    format_axis_value(bucket.range_start),
                    format_axis_value(bucket.range_end),
                );
                if series.len() == 1 && series[0].label.is_empty() {
                    text.push_str(&format!("\nCount: {}", bucket.count));
                } else {
                    for s in series {
                        text.push_str(&format!("\n{}: {}", s.label, s.buckets[bucket_idx].count));
                    }
                }
                Some(text)
            }
            HitTestData::Scatter {
                points,
                x_label,
                y_label,
            } => {
                const MAX_DIST_PX: f64 = 10.0;
                let mut best: Option<(f64, &ScatterPoint)> = None;
                for p in points {
                    let (ppx, ppy) = self.mapping.pixel_at(p.x, p.y);
                    let dist = ((ppx - px).powi(2) + (ppy - py).powi(2)).sqrt();
                    if dist > MAX_DIST_PX {
                        continue;
                    }
                    if best.as_ref().is_none_or(|(best_dist, _)| dist < *best_dist) {
                        best = Some((dist, p));
                    }
                }
                let (_, p) = best?;
                let coords = format!(
                    "{}: {}\n{}: {}",
                    x_label,
                    format_axis_value(p.x),
                    y_label,
                    format_axis_value(p.y)
                );
                Some(match &p.group {
                    Some(g) => format!("{coords}\n{g}"),
                    None => coords,
                })
            }
            HitTestData::Heatmap {
                cols,
                rows,
                cell_size,
                x_min,
                y_min,
                cells,
                value_label,
            } => {
                let col = ((data_x - x_min) / cell_size).floor();
                let row = ((data_y - y_min) / cell_size).floor();
                if col < 0.0 || row < 0.0 {
                    return None;
                }
                let (col, row) = (col as usize, row as usize);
                if col >= *cols || row >= *rows {
                    return None;
                }
                let cell = cells.get(row * cols + col)?;
                if cell.count == 0 {
                    return None;
                }
                Some(format!(
                    "{}: {}\nCount: {}",
                    value_label,
                    format_axis_value(cell.value),
                    cell.count
                ))
            }
        }
    }
}

/// A rendered chart as a row-major RGB8 buffer — the layout
/// `slint::SharedPixelBuffer<Rgb8Pixel>` consumes directly in the GUI (see
/// `crates/gui/src/editor/viewport_controller.rs` for the established
/// pattern), and what `image::save_buffer` would need for a CLI PNG export.
#[derive(Clone)]
pub struct RenderedChart {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// `None` only if plotters failed to report a coordinate mapping for the
    /// rendered plot area — shouldn't happen in practice, but hover support
    /// degrades gracefully (no tooltip) rather than panicking if it ever did.
    pub hit_test: Option<ChartHitTester>,
    /// For heatmaps, the `[min, max]` the color scale/legend actually used
    /// (the resolved value of the `HeatmapRange` passed to `render_heatmap`) —
    /// lets the GUI display/persist the live auto-computed range even while
    /// in `Auto` mode. `None` for histogram/scatter charts.
    pub heatmap_range: Option<(f64, f64)>,
}

/// Fixed, deterministic color cycle for scatter groups (class/colocalized
/// labels) — avoids depending on plotters' `full_palette` feature.
const PALETTE: [RGBColor; 8] = [
    RGBColor(0x1f, 0x77, 0xb4),
    RGBColor(0xff, 0x7f, 0x0e),
    RGBColor(0x2c, 0xa0, 0x2c),
    RGBColor(0xd6, 0x27, 0x28),
    RGBColor(0x94, 0x67, 0xbd),
    RGBColor(0x8c, 0x56, 0x4b),
    RGBColor(0xe3, 0x77, 0xc2),
    RGBColor(0x7f, 0x7f, 0x7f),
];

fn to_internal_error<E: std::fmt::Display>(e: E) -> InternalErrors {
    // Matches the loose error-stringification convention already used
    // throughout `results_loader.rs`/`duckdb.rs`.
    InternalErrors::Io(e.to_string())
}

/// Compact axis-label formatting: whole numbers for large values, two
/// decimals for small ones — just enough to keep histogram axis labels
/// readable without pulling in a full number-formatting dependency.
fn format_axis_value(v: f64) -> String {
    if v.abs() >= 1000.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

fn draw_histogram(
    root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &HistogramData,
) -> Result<Option<ChartHitTester>, InternalErrors> {
    root.fill(&WHITE).map_err(to_internal_error)?;

    let max_count = data
        .series
        .iter()
        .flat_map(|s| s.buckets.iter())
        .map(|b| b.count)
        .max()
        .unwrap_or(0);
    let x_min = data.buckets().first().map_or(0.0, |b| b.range_start);
    let x_max_raw = data.buckets().last().map_or(1.0, |b| b.range_end);
    let x_max = if x_max_raw > x_min {
        x_max_raw
    } else {
        x_min + 1.0
    };

    // In log mode, bars are positioned by the log of their (real-unit) edges
    // so equal-log-space buckets render as equal-width bars on screen; axis
    // labels exponentiate back so they still read in real units. Bucket
    // edges themselves (`b.range_start`/`range_end`) are always real-unit —
    // only the on-screen coordinate is transformed.
    let to_axis = |v: f64| if data.log_scale { v.ln() } else { v };
    let axis_min = to_axis(x_min);
    let axis_max = to_axis(x_max);
    let log_scale = data.log_scale;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .caption(
            format!("Histogram: {}", data.column_label),
            ("sans-serif", 24),
        )
        .build_cartesian_2d(axis_min..axis_max, 0f64..(max_count as f64 * 1.1).max(1.0))
        .map_err(to_internal_error)?;
    let mapping = PixelToData::from_chart(&chart);

    chart
        .configure_mesh()
        .x_desc(data.column_label.clone())
        .y_desc("Count")
        .x_label_formatter(&move |v| format_axis_value(if log_scale { v.exp() } else { *v }))
        .label_style(("sans-serif", 14))
        .axis_desc_style(("sans-serif", 16))
        .draw()
        .map_err(to_internal_error)?;

    let grouped = data.series.len() > 1;
    for (i, series) in data.series.iter().enumerate() {
        // Ungrouped (single series): solid bars, same as before. Grouped:
        // each series gets its own palette color, drawn semi-transparent and
        // overlaid so every group's shape stays visible where they overlap.
        let color = if grouped {
            PALETTE[i % PALETTE.len()].mix(0.45)
        } else {
            PALETTE[0].mix(1.0)
        };
        let handle = chart
            .draw_series(series.buckets.iter().map(|b| {
                Rectangle::new(
                    [
                        (to_axis(b.range_start), 0.0),
                        (to_axis(b.range_end), b.count as f64),
                    ],
                    color.filled(),
                )
            }))
            .map_err(to_internal_error)?;
        if grouped {
            let legend_color = PALETTE[i % PALETTE.len()];
            handle.label(series.label.clone()).legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 16, y + 5)], legend_color.filled())
            });
        }
    }
    if grouped {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 14))
            .draw()
            .map_err(to_internal_error)?;
    }

    root.present().map_err(to_internal_error)?;
    Ok(mapping.map(|mapping| ChartHitTester {
        mapping,
        data: HitTestData::Histogram {
            series: data.series.clone(),
            log_scale: data.log_scale,
            column_label: data.column_label.clone(),
        },
    }))
}

/// Min/max of `values`, padded by 5% on each side; falls back to `(0, 1)` for
/// no values and to a fixed `+/-0.5` window when every value is identical.
fn padded_axis_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values {
        min = min.min(v);
        max = max.max(v);
    }
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if max <= min {
        return (min - 0.5, min + 0.5);
    }
    let pad = (max - min) * 0.05;
    (min - pad, max + pad)
}

/// Computes how much to symmetrically pad `data_w`/`data_h` (a data-space
/// rectangle) so that rendering it into a `plot_px_w`x`plot_px_h` pixel area
/// gives *equal* pixels-per-data-unit on both axes — i.e. so a square in data
/// space (like one heatmap cell) renders as an actual square on screen,
/// regardless of the plot area's own aspect ratio. Only the axis with the
/// *lower* pixel density gets padded (its density is raised to match the
/// other axis, by widening its apparent data range around the same center);
/// the other axis's padding is always `0.0`. Returns `(pad_x, pad_y)`.
///
/// This is the "letterbox" technique: the padded axis shows extra blank space
/// on both sides within the plot, exactly like fitting a fixed-aspect image
/// into a differently-shaped viewport.
fn square_aspect_padding(data_w: f64, data_h: f64, plot_px_w: u32, plot_px_h: u32) -> (f64, f64) {
    if data_w <= 0.0 || data_h <= 0.0 || plot_px_w == 0 || plot_px_h == 0 {
        return (0.0, 0.0);
    }
    let px_per_unit_x = plot_px_w as f64 / data_w;
    let px_per_unit_y = plot_px_h as f64 / data_h;
    let px_per_unit = px_per_unit_x.min(px_per_unit_y);

    let target_w = plot_px_w as f64 / px_per_unit;
    let target_h = plot_px_h as f64 / px_per_unit;
    (
        (target_w - data_w).max(0.0) / 2.0,
        (target_h - data_h).max(0.0) / 2.0,
    )
}

fn draw_scatter(
    root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &ScatterData,
) -> Result<Option<ChartHitTester>, InternalErrors> {
    root.fill(&WHITE).map_err(to_internal_error)?;

    let (x_min, x_max) = padded_axis_bounds(data.points.iter().map(|p| p.x));
    let (y_min, y_max) = padded_axis_bounds(data.points.iter().map(|p| p.y));

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .caption(
            format!("{} vs {}", data.y_label, data.x_label),
            ("sans-serif", 24),
        )
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(to_internal_error)?;
    let mapping = PixelToData::from_chart(&chart);

    chart
        .configure_mesh()
        .x_desc(data.x_label.clone())
        .y_desc(data.y_label.clone())
        .label_style(("sans-serif", 14))
        .axis_desc_style(("sans-serif", 16))
        .draw()
        .map_err(to_internal_error)?;

    let mut groups: Vec<String> = data.points.iter().filter_map(|p| p.group.clone()).collect();
    groups.sort();
    groups.dedup();

    if groups.is_empty() {
        chart
            .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
                data.points.iter().map(|p| (p.x, p.y)),
                3,
                PALETTE[0].filled(),
            ))
            .map_err(to_internal_error)?;
    } else {
        for (i, group) in groups.iter().enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            chart
                .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
                    data.points
                        .iter()
                        .filter(|p| p.group.as_deref() == Some(group.as_str()))
                        .map(|p| (p.x, p.y)),
                    3,
                    color.filled(),
                ))
                .map_err(to_internal_error)?
                .label(group.clone())
                .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
        }
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 14))
            .draw()
            .map_err(to_internal_error)?;
    }

    root.present().map_err(to_internal_error)?;
    Ok(mapping.map(|mapping| ChartHitTester {
        mapping,
        data: HitTestData::Scatter {
            points: data.points.clone(),
            x_label: data.x_label.clone(),
            y_label: data.y_label.clone(),
        },
    }))
}

/// Sequential colormaps offered for the heatmap. Each is a small set of
/// anchor stops, linearly interpolated in RGB space — perceptually uniform-ish
/// (except `Grayscale`) and colorblind-friendlier than a raw hue sweep,
/// without depending on plotters' own colormap feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatmapColorScheme {
    Viridis,
    Magma,
    Plasma,
    Grayscale,
}

impl HeatmapColorScheme {
    /// Label shown in the GUI's color-scheme picker.
    pub fn label(self) -> &'static str {
        match self {
            HeatmapColorScheme::Viridis => "Viridis",
            HeatmapColorScheme::Magma => "Magma",
            HeatmapColorScheme::Plasma => "Plasma",
            HeatmapColorScheme::Grayscale => "Grayscale",
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label.to_ascii_lowercase().as_str() {
            "magma" => HeatmapColorScheme::Magma,
            "plasma" => HeatmapColorScheme::Plasma,
            "grayscale" => HeatmapColorScheme::Grayscale,
            _ => HeatmapColorScheme::Viridis,
        }
    }

    /// All schemes, in the order the GUI's picker should list them.
    pub fn all() -> &'static [HeatmapColorScheme] {
        &[
            HeatmapColorScheme::Viridis,
            HeatmapColorScheme::Magma,
            HeatmapColorScheme::Plasma,
            HeatmapColorScheme::Grayscale,
        ]
    }

    fn stops(self) -> &'static [(f64, u8, u8, u8)] {
        match self {
            HeatmapColorScheme::Viridis => &[
                (0.00, 68, 1, 84),
                (0.25, 59, 82, 139),
                (0.50, 33, 145, 140),
                (0.75, 94, 201, 98),
                (1.00, 253, 231, 37),
            ],
            HeatmapColorScheme::Magma => &[
                (0.00, 0, 0, 4),
                (0.25, 81, 18, 124),
                (0.50, 183, 55, 121),
                (0.75, 252, 137, 97),
                (1.00, 252, 253, 191),
            ],
            HeatmapColorScheme::Plasma => &[
                (0.00, 13, 8, 135),
                (0.25, 126, 3, 168),
                (0.50, 204, 71, 120),
                (0.75, 248, 148, 65),
                (1.00, 240, 249, 33),
            ],
            HeatmapColorScheme::Grayscale => &[(0.00, 20, 20, 20), (1.00, 235, 235, 235)],
        }
    }

    /// Maps `t` (clamped to `[0, 1]`) to a color via this scheme's stops.
    pub fn color(self, t: f64) -> RGBColor {
        interpolate_stops(self.stops(), t)
    }

    /// Same mapping as [`color`](Self::color), as a plain `(r, g, b)` tuple —
    /// so callers outside this module that color cells directly instead of
    /// rendering a bitmap (e.g. the GUI's Matrix view) don't need a
    /// `plotters` dependency just to destructure an `RGBColor`.
    pub fn color_rgb(self, t: f64) -> (u8, u8, u8) {
        let RGBColor(r, g, b) = self.color(t);
        (r, g, b)
    }
}

/// The `[min, max]` a heatmap's color scale (and legend) spans.
///
/// `Auto` reproduces the original behavior — always `[0, max(cell values)]` —
/// recomputed fresh on every render. `Manual` pins an explicit range that
/// stays fixed regardless of what the currently-plotted data's own min/max
/// happen to be, so two different heatmaps rendered with the same `Manual`
/// range map the same value to the same color and are directly comparable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeatmapRange {
    Auto,
    Manual { min: f64, max: f64 },
}

impl HeatmapRange {
    /// Resolves to a concrete `(min, max)` pair, guaranteeing `max > min` (a
    /// degenerate/inverted manual range collapses to a hairline span above
    /// `min` rather than dividing by zero downstream).
    fn resolve(self, cells: &[HeatmapCell]) -> (f64, f64) {
        match self {
            HeatmapRange::Auto => {
                let max = cells
                    .iter()
                    .map(|c| c.value)
                    .fold(0.0f64, f64::max)
                    .max(1e-9);
                (0.0, max)
            }
            HeatmapRange::Manual { min, max } => {
                if max > min {
                    (min, max)
                } else {
                    (min, min + 1e-9)
                }
            }
        }
    }
}

fn interpolate_stops(stops: &[(f64, u8, u8, u8)], t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for i in 0..stops.len() - 1 {
        let (t0, r0, g0, b0) = stops[i];
        let (t1, r1, g1, b1) = stops[i + 1];
        if t <= t1 {
            let local_t = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * local_t) as u8;
            return RGBColor(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1));
        }
    }
    let (_, r, g, b) = stops[stops.len() - 1];
    RGBColor(r, g, b)
}

fn draw_heatmap(
    root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &HeatmapData,
    scheme: HeatmapColorScheme,
    range: HeatmapRange,
) -> Result<(Option<ChartHitTester>, (f64, f64)), InternalErrors> {
    root.fill(&WHITE).map_err(to_internal_error)?;

    let (range_min, range_max) = range.resolve(&data.cells);
    let range_span = range_max - range_min;
    let legend_width = 110i32;
    // `split_horizontally(x)` returns (left part of width x, right remainder)
    // — split at `total_width - legend_width` so the *chart* gets the bulk of
    // the area and the legend is the narrow strip on the right.
    let (total_width, _) = root.dim_in_pixel();
    let (chart_area, legend_area) =
        root.split_horizontally((total_width as i32 - legend_width).max(0));

    let x_max_raw = data.x_min + data.cols as f64 * data.cell_size;
    let y_max_raw = data.y_min + data.rows as f64 * data.cell_size;
    let data_w = data.cols as f64 * data.cell_size;
    let data_h = data.rows as f64 * data.cell_size;

    // First pass: build with the raw (possibly non-square-cell) range just to
    // measure the plotting area's actual pixel size once axis labels/margins
    // are accounted for — `chart_area` is cheap to reuse (`DrawingArea` is a
    // `Clone`-able handle), so this costs one throwaway chart build.
    let probe = ChartBuilder::on(&chart_area.clone())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .caption(format!("Heatmap: {}", data.value_label), ("sans-serif", 24))
        .build_cartesian_2d(data.x_min..x_max_raw, y_max_raw..data.y_min)
        .map_err(to_internal_error)?;
    let (plot_px_w, plot_px_h) = probe.plotting_area().dim_in_pixel();
    drop(probe);

    // Pad whichever axis has spare pixel density so both axes end up with the
    // same pixels-per-data-unit — this is what makes each cell an actual
    // on-screen square regardless of the chart area's own aspect ratio (see
    // `square_aspect_padding`).
    let (pad_x, pad_y) = square_aspect_padding(data_w, data_h, plot_px_w, plot_px_h);
    let x_min = data.x_min - pad_x;
    let x_max = x_max_raw + pad_x;
    let y_top = y_max_raw + pad_y;
    let y_bottom = data.y_min - pad_y;

    let mut chart = ChartBuilder::on(&chart_area)
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .caption(format!("Heatmap: {}", data.value_label), ("sans-serif", 24))
        // Image-style coordinates: y grows downward, so the top-left object ends
        // up top-left on screen rather than bottom-left.
        .build_cartesian_2d(x_min..x_max, y_top..y_bottom)
        .map_err(to_internal_error)?;
    let mapping = PixelToData::from_chart(&chart);

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc(data.x_label.clone())
        .y_desc(data.y_label.clone())
        .label_style(("sans-serif", 14))
        .axis_desc_style(("sans-serif", 16))
        .draw()
        .map_err(to_internal_error)?;

    // Fill pass, then a separate border pass on top — `ShapeStyle` carries a
    // single color for both fill and stroke, so a differently-colored thin
    // border needs its own unfilled `Rectangle` drawn after the filled one.
    chart
        .draw_series(
            data.cells
                .iter()
                .enumerate()
                .filter(|(_, c)| c.count > 0)
                .map(|(i, cell)| {
                    let col = i % data.cols;
                    let row = i / data.cols;
                    let x0 = data.x_min + col as f64 * data.cell_size;
                    let y0 = data.y_min + row as f64 * data.cell_size;
                    let color = scheme.color((cell.value - range_min) / range_span);
                    Rectangle::new(
                        [(x0, y0), (x0 + data.cell_size, y0 + data.cell_size)],
                        color.filled(),
                    )
                }),
        )
        .map_err(to_internal_error)?;
    const CELL_BORDER: RGBColor = RGBColor(110, 110, 110);
    chart
        .draw_series(
            data.cells
                .iter()
                .enumerate()
                .filter(|(_, c)| c.count > 0)
                .map(|(i, _cell)| {
                    let col = i % data.cols;
                    let row = i / data.cols;
                    let x0 = data.x_min + col as f64 * data.cell_size;
                    let y0 = data.y_min + row as f64 * data.cell_size;
                    Rectangle::new(
                        [(x0, y0), (x0 + data.cell_size, y0 + data.cell_size)],
                        ShapeStyle {
                            color: CELL_BORDER.to_rgba(),
                            filled: false,
                            stroke_width: 1,
                        },
                    )
                }),
        )
        .map_err(to_internal_error)?;

    // Legend: a vertical gradient strip with the value scale on its axis.
    let mut legend = ChartBuilder::on(&legend_area)
        .margin_top(10)
        .margin_bottom(10)
        .margin_right(10)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..1f64, range_min..range_max)
        .map_err(to_internal_error)?;
    legend
        .configure_mesh()
        .disable_mesh()
        .x_labels(0)
        .y_desc(data.value_label.clone())
        .label_style(("sans-serif", 13))
        .axis_desc_style(("sans-serif", 14))
        .draw()
        .map_err(to_internal_error)?;
    const LEGEND_STEPS: usize = 64;
    legend
        .draw_series((0..LEGEND_STEPS).map(|i| {
            let t0 = i as f64 / LEGEND_STEPS as f64;
            let t1 = (i + 1) as f64 / LEGEND_STEPS as f64;
            Rectangle::new(
                [
                    (0.0, range_min + t0 * range_span),
                    (1.0, range_min + t1 * range_span),
                ],
                scheme.color(t0).filled(),
            )
        }))
        .map_err(to_internal_error)?;

    root.present().map_err(to_internal_error)?;
    let hit_test = mapping.map(|mapping| ChartHitTester {
        mapping,
        data: HitTestData::Heatmap {
            cols: data.cols,
            rows: data.rows,
            cell_size: data.cell_size,
            x_min: data.x_min,
            y_min: data.y_min,
            cells: data.cells.clone(),
            value_label: data.value_label.clone(),
        },
    });
    Ok((hit_test, (range_min, range_max)))
}

/// Renders into an in-memory RGB8 buffer — used by the GUI to build a
/// `slint::Image` without touching disk.
pub fn render_histogram(
    data: &HistogramData,
    width: u32,
    height: u32,
) -> Result<RenderedChart, InternalErrors> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let hit_test = {
        let root = BitMapBackend::with_buffer(&mut rgb, (width, height)).into_drawing_area();
        draw_histogram(root, data)?
    };
    Ok(RenderedChart {
        width,
        height,
        rgb,
        hit_test,
        heatmap_range: None,
    })
}

/// Renders into an in-memory RGB8 buffer — used by the GUI to build a
/// `slint::Image` without touching disk.
pub fn render_scatter(
    data: &ScatterData,
    width: u32,
    height: u32,
) -> Result<RenderedChart, InternalErrors> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let hit_test = {
        let root = BitMapBackend::with_buffer(&mut rgb, (width, height)).into_drawing_area();
        draw_scatter(root, data)?
    };
    Ok(RenderedChart {
        width,
        height,
        rgb,
        hit_test,
        heatmap_range: None,
    })
}

/// Renders straight to a PNG file. Not called anywhere yet — exists so a
/// future CLI "export chart" command is a thin wrapper around the same
/// drawing code the GUI uses, instead of new plotting logic.
pub fn save_histogram_png(
    data: &HistogramData,
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), InternalErrors> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    draw_histogram(root, data)?;
    Ok(())
}

/// Renders straight to a PNG file. Not called anywhere yet — exists so a
/// future CLI "export chart" command is a thin wrapper around the same
/// drawing code the GUI uses, instead of new plotting logic.
pub fn save_scatter_png(
    data: &ScatterData,
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), InternalErrors> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    draw_scatter(root, data)?;
    Ok(())
}

/// Renders into an in-memory RGB8 buffer — used by the GUI to build a
/// `slint::Image` without touching disk.
pub fn render_heatmap(
    data: &HeatmapData,
    scheme: HeatmapColorScheme,
    range: HeatmapRange,
    width: u32,
    height: u32,
) -> Result<RenderedChart, InternalErrors> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let (hit_test, used_range) = {
        let root = BitMapBackend::with_buffer(&mut rgb, (width, height)).into_drawing_area();
        draw_heatmap(root, data, scheme, range)?
    };
    Ok(RenderedChart {
        width,
        height,
        rgb,
        hit_test,
        heatmap_range: Some(used_range),
    })
}

/// Renders straight to a PNG file. Not called anywhere yet — exists so a
/// future CLI "export chart" command is a thin wrapper around the same
/// drawing code the GUI uses, instead of new plotting logic.
pub fn save_heatmap_png(
    data: &HeatmapData,
    scheme: HeatmapColorScheme,
    range: HeatmapRange,
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), InternalErrors> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    draw_heatmap(root, data, scheme, range)?;
    Ok(())
}

/// Writes an already-rendered chart's pixels straight to a PNG file — used by
/// the GUI's "Save chart" button to persist exactly what's currently on
/// screen (no re-querying/re-rendering involved, unlike `save_histogram_png`
/// and friends which redraw from scratch).
pub fn save_rendered_chart_png(chart: &RenderedChart, path: &Path) -> Result<(), InternalErrors> {
    image::save_buffer(
        path,
        &chart.rgb,
        chart.width,
        chart.height,
        image::ColorType::Rgb8,
    )
    .map_err(to_internal_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_aspect_padding_pads_the_higher_density_axis() {
        // 20x10 data rect (wider than tall) into a SQUARE 100x100px area:
        // px-per-unit is 5 for x (100/20) and 10 for y (100/10) — y is
        // "zoomed in" 2x more than x. To equalize, y's data range gets
        // expanded (reducing its density down to x's), not x's.
        let (pad_x, pad_y) = square_aspect_padding(20.0, 10.0, 100, 100);
        assert_eq!(
            pad_x, 0.0,
            "the already-lower-density axis (x) is never padded"
        );
        assert!(
            pad_y > 0.0,
            "the higher-density axis (y) must be padded down to match"
        );
        // After padding, both axes must have equal px-per-unit.
        let padded_h = 10.0 + 2.0 * pad_y;
        assert!((100.0 / padded_h - 100.0 / 20.0).abs() < 1e-9);
    }

    #[test]
    fn square_aspect_padding_matching_aspect_needs_no_padding() {
        // Data aspect already matches the pixel area's aspect exactly.
        let (pad_x, pad_y) = square_aspect_padding(200.0, 100.0, 400, 200);
        assert_eq!(pad_x, 0.0);
        assert_eq!(pad_y, 0.0);
    }

    #[test]
    fn square_aspect_padding_degenerate_inputs_yield_no_padding() {
        assert_eq!(square_aspect_padding(0.0, 10.0, 100, 100), (0.0, 0.0));
        assert_eq!(square_aspect_padding(10.0, 10.0, 0, 100), (0.0, 0.0));
    }

    #[test]
    fn heatmap_color_scheme_endpoints_and_label_roundtrip() {
        for scheme in HeatmapColorScheme::all() {
            // Every scheme's first/last stop should be reproduced at t=0/t=1.
            let c0 = scheme.color(0.0);
            let c1 = scheme.color(1.0);
            assert_ne!(
                c0.rgb(),
                c1.rgb(),
                "{:?} should visibly change across its range",
                scheme
            );
            assert_eq!(HeatmapColorScheme::from_label(scheme.label()), *scheme);
        }
    }

    #[test]
    fn heatmap_color_scheme_from_label_falls_back_to_viridis() {
        assert_eq!(
            HeatmapColorScheme::from_label("nonsense"),
            HeatmapColorScheme::Viridis
        );
    }

    fn make_object(
        area_px: u64,
        intensities_json: &str,
        object_class_name: Vec<String>,
        coloc_json: &str,
    ) -> ObjectRow {
        ObjectRow {
            image_name: "img.tif".into(),
            image_rel_path: String::new(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: "00000000-0000-0000-0000-000000000001".into(),
            seg_class_name: None,
            seg_class_id: None,
            object_class_name,
            object_class_id: vec![],
            parent_id: None,
            children: vec![],
            track_id: 0,
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            centroid_x_nm: 0.0,
            centroid_y_nm: 0.0,
            area_px,
            area_nm2: area_px as f64,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity: 0.5,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: intensities_json.into(),
            coloc_json: coloc_json.into(),
            bbox_px: [0, 0, 0, 0],
        }
    }

    fn base_columns() -> Vec<ColumnSpec> {
        crate::results::results_loader::build_column_specs(&[], &[])
    }

    // ---- plottable_columns ----

    #[test]
    fn plottable_columns_excludes_non_numeric_and_hidden() {
        let mut cols = base_columns();
        for c in cols.iter_mut() {
            if c.id == "area_nm2" {
                c.visible = false;
            }
        }
        let plottable = plottable_columns(&cols);
        let ids: Vec<&str> = plottable.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"area_px"));
        assert!(ids.contains(&"circularity"));
        assert!(!ids.contains(&"area_nm2"), "hidden column excluded");
        assert!(!ids.contains(&"image"), "non-numeric column excluded");
        assert!(!ids.contains(&"object_id"), "non-numeric column excluded");
    }

    // ---- compute_histogram ----

    #[test]
    fn compute_histogram_buckets_values_across_range() {
        let objects = vec![
            make_object(0, "{}", vec![], "{}"),
            make_object(10, "{}", vec![], "{}"),
            make_object(50, "{}", vec![], "{}"),
            make_object(99, "{}", vec![], "{}"),
        ];
        let columns = base_columns();
        let data =
            compute_histogram(&objects, "area_px", &columns, 10, false, ColorBy::None).unwrap();
        assert_eq!(data.column_label, "Area (px²)");
        assert_eq!(data.buckets().len(), 10);
        assert!(!data.log_scale);
        assert_eq!(data.excluded_non_positive, 0);
        let total: usize = data.buckets().iter().map(|b| b.count).sum();
        assert_eq!(total, 4);
        // width = (99-0)/10 = 9.9: 0 -> bucket 0, 10 -> bucket 1, 50 -> bucket 5, 99 -> bucket 9.
        assert_eq!(data.buckets()[0].count, 1);
        assert_eq!(data.buckets()[1].count, 1);
        assert_eq!(data.buckets()[5].count, 1);
        assert_eq!(
            data.buckets()[9].count,
            1,
            "max value (99) clamps into the last bucket"
        );
    }

    #[test]
    fn compute_histogram_identical_values_single_bucket() {
        let objects = vec![
            make_object(5, "{}", vec![], "{}"),
            make_object(5, "{}", vec![], "{}"),
        ];
        let data = compute_histogram(
            &objects,
            "area_px",
            &base_columns(),
            10,
            false,
            ColorBy::None,
        )
        .unwrap();
        assert_eq!(data.buckets().len(), 1);
        assert_eq!(data.buckets()[0].count, 2);
    }

    #[test]
    fn compute_histogram_unknown_column_or_empty_objects_is_none() {
        let columns = base_columns();
        assert!(compute_histogram(&[], "area_px", &columns, 10, false, ColorBy::None).is_none());
        assert!(
            compute_histogram(
                &[make_object(1, "{}", vec![], "{}")],
                "does_not_exist",
                &columns,
                10,
                false,
                ColorBy::None
            )
            .is_none()
        );
        assert!(
            compute_histogram(
                &[make_object(1, "{}", vec![], "{}")],
                "area_px",
                &columns,
                0,
                false,
                ColorBy::None
            )
            .is_none()
        );
    }

    #[test]
    fn compute_histogram_log_scale_spreads_skewed_outliers_across_buckets() {
        // A few huge outliers alongside many small values — equal-width
        // linear binning would crowd everything into bucket 0; log binning
        // should spread them out instead.
        let mut objects: Vec<ObjectRow> = (1..=20)
            .map(|i| make_object(i * 100, "{}", vec![], "{}"))
            .collect();
        objects.push(make_object(1_000_000, "{}", vec![], "{}"));
        let columns = base_columns();
        let data =
            compute_histogram(&objects, "area_px", &columns, 10, true, ColorBy::None).unwrap();
        assert!(data.log_scale);
        let occupied = data.buckets().iter().filter(|b| b.count > 0).count();
        assert!(
            occupied > 1,
            "log binning should spread values across more than one bucket"
        );
    }

    #[test]
    fn compute_histogram_log_scale_excludes_non_positive_values() {
        let objects = vec![
            make_object(0, "{}", vec![], "{}"),
            make_object(10, "{}", vec![], "{}"),
            make_object(100, "{}", vec![], "{}"),
        ];
        let columns = base_columns();
        let data =
            compute_histogram(&objects, "area_px", &columns, 5, true, ColorBy::None).unwrap();
        assert_eq!(
            data.excluded_non_positive, 1,
            "area_px == 0 can't be log-binned"
        );
        let total: usize = data.buckets().iter().map(|b| b.count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn compute_histogram_color_by_class_produces_one_series_per_class_with_shared_edges() {
        let objects = vec![
            make_object(0, "{}", vec!["ClassA".into()], "{}"),
            make_object(50, "{}", vec!["ClassA".into()], "{}"),
            make_object(99, "{}", vec!["ClassB".into()], "{}"),
        ];
        let columns = base_columns();
        let data =
            compute_histogram(&objects, "area_px", &columns, 10, false, ColorBy::Class).unwrap();

        assert_eq!(data.series.len(), 2, "one series per distinct class");
        let labels: Vec<&str> = data.series.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["ClassA", "ClassB"], "sorted by label");

        // Every series must share identical bucket edges (computed from all
        // classes' values combined), so they can be overlaid directly.
        let class_a = &data.series[0];
        let class_b = &data.series[1];
        assert_eq!(class_a.buckets.len(), class_b.buckets.len());
        for (a, b) in class_a.buckets.iter().zip(class_b.buckets.iter()) {
            assert_eq!(a.range_start, b.range_start);
            assert_eq!(a.range_end, b.range_end);
        }

        // ClassA has 2 ROIs, ClassB has 1 — each series' total matches only
        // its own class's object count, not the combined total.
        assert_eq!(class_a.buckets.iter().map(|b| b.count).sum::<usize>(), 2);
        assert_eq!(class_b.buckets.iter().map(|b| b.count).sum::<usize>(), 1);
    }

    #[test]
    fn compute_histogram_color_by_colocalized_groups_yes_no() {
        let mut coloc_object = make_object(10, "{}", vec![], "{}");
        coloc_object.coloc_json = r#"{"Target":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let not_coloc_object = make_object(20, "{}", vec![], "{}");
        let objects = vec![coloc_object, not_coloc_object];
        let columns = base_columns();

        let data = compute_histogram(
            &objects,
            "area_px",
            &columns,
            5,
            false,
            ColorBy::Colocalized,
        )
        .unwrap();
        let labels: Vec<&str> = data.series.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["No", "Yes"]);
    }

    // ---- compute_scatter ----

    #[test]
    fn compute_scatter_pairs_x_and_y_and_colors_by_class() {
        let objects = vec![
            make_object(10, "{}", vec!["ClassA".into()], "{}"),
            make_object(20, "{}", vec!["ClassB".into()], "{}"),
        ];
        let columns = base_columns();
        let data = compute_scatter(
            &objects,
            "area_px",
            "circularity",
            ColorBy::Class,
            &columns,
            0,
        )
        .unwrap();
        assert_eq!(data.x_label, "Area (px²)");
        assert_eq!(data.y_label, "Circularity");
        assert_eq!(data.points.len(), 2);
        assert_eq!(data.points[0].group.as_deref(), Some("ClassA"));
        assert_eq!(data.points[1].group.as_deref(), Some("ClassB"));
        assert!(data.sampled_from.is_none());
    }

    #[test]
    fn compute_scatter_colors_by_colocalized() {
        let mut coloc_object = make_object(10, "{}", vec![], "{}");
        coloc_object.coloc_json =
            r#"{"Target (1)":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let not_coloc_object = make_object(20, "{}", vec![], "{}");
        let columns = base_columns();
        let data = compute_scatter(
            &[coloc_object, not_coloc_object],
            "area_px",
            "circularity",
            ColorBy::Colocalized,
            &columns,
            0,
        )
        .unwrap();
        assert_eq!(data.points[0].group.as_deref(), Some("Yes"));
        assert_eq!(data.points[1].group.as_deref(), Some("No"));
    }

    #[test]
    fn compute_scatter_caps_and_reports_original_count() {
        let objects: Vec<ObjectRow> = (0..100)
            .map(|i| make_object(i, "{}", vec![], "{}"))
            .collect();
        let columns = base_columns();
        let data = compute_scatter(
            &objects,
            "area_px",
            "circularity",
            ColorBy::None,
            &columns,
            10,
        )
        .unwrap();
        assert_eq!(data.points.len(), 10);
        assert_eq!(data.sampled_from, Some(100));

        // Deterministic: re-computing yields the identical sampled subset.
        let again = compute_scatter(
            &objects,
            "area_px",
            "circularity",
            ColorBy::None,
            &columns,
            10,
        )
        .unwrap();
        assert_eq!(data.points, again.points);
    }

    #[test]
    fn compute_scatter_unknown_column_is_none() {
        let objects = vec![make_object(1, "{}", vec![], "{}")];
        let columns = base_columns();
        assert!(
            compute_scatter(
                &objects,
                "does_not_exist",
                "circularity",
                ColorBy::None,
                &columns,
                0
            )
            .is_none()
        );
    }

    // ---- compute_heatmap ----

    fn object_at(x: f64, y: f64) -> ObjectRow {
        ObjectRow {
            centroid_x_px: x,
            centroid_y_px: y,
            ..make_object(10, "{}", vec![], "{}")
        }
    }

    #[test]
    fn compute_heatmap_count_bins_centroids_into_cells() {
        // 10px cells over a 0..20 x 0..20 span: (0,0)/(1,1) share cell (0,0);
        // (19,19) lands in the far corner cell.
        let objects = vec![
            object_at(0.0, 0.0),
            object_at(1.0, 1.0),
            object_at(19.0, 19.0),
        ];
        let columns = base_columns();
        let data = compute_heatmap(&objects, &HeatmapMetric::Count, &columns, 10.0).unwrap();
        assert_eq!(data.value_label, "Count");
        assert_eq!(data.cols, 2);
        assert_eq!(data.rows, 2);
        assert_eq!(data.cells[0].count, 2);
        assert_eq!(data.cells[0].value, 2.0);
        assert_eq!(data.cells[3].count, 1, "(19,19) falls in the last row/col");
        let total: usize = data.cells.iter().map(|c| c.count).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn compute_heatmap_average_uses_metric_mean_per_cell() {
        let mut a = object_at(0.0, 0.0);
        a.area_px = 10;
        a.area_nm2 = 10.0;
        let mut b = object_at(1.0, 1.0);
        b.area_px = 20;
        b.area_nm2 = 20.0;
        let columns = base_columns();
        let data = compute_heatmap(
            &[a, b],
            &HeatmapMetric::Average("area_px".into()),
            &columns,
            10.0,
        )
        .unwrap();
        assert_eq!(data.value_label, "Area (px²)");
        assert_eq!(data.cells[0].count, 2);
        assert_eq!(data.cells[0].value, 15.0, "average of 10 and 20");
    }

    #[test]
    fn compute_heatmap_empty_objects_or_invalid_cell_size_or_unknown_column_is_none() {
        let columns = base_columns();
        assert!(compute_heatmap(&[], &HeatmapMetric::Count, &columns, 10.0).is_none());
        let objects = vec![object_at(0.0, 0.0)];
        assert!(compute_heatmap(&objects, &HeatmapMetric::Count, &columns, 0.0).is_none());
        assert!(compute_heatmap(&objects, &HeatmapMetric::Count, &columns, -5.0).is_none());
        assert!(
            compute_heatmap(
                &objects,
                &HeatmapMetric::Average("does_not_exist".into()),
                &columns,
                10.0
            )
            .is_none()
        );
    }

    #[test]
    fn compute_heatmap_single_point_is_one_by_one_grid() {
        let objects = vec![object_at(5.0, 5.0)];
        let columns = base_columns();
        let data = compute_heatmap(&objects, &HeatmapMetric::Count, &columns, 10.0).unwrap();
        assert_eq!(data.cols, 1);
        assert_eq!(data.rows, 1);
        assert_eq!(data.cells[0].count, 1);
    }

    // ---- rendering smoke tests ----

    #[test]
    fn render_histogram_produces_correctly_sized_buffer() {
        let data = HistogramData {
            column_label: "Area".into(),
            series: vec![HistogramSeries {
                label: String::new(),
                buckets: vec![
                    HistogramBucket {
                        range_start: 0.0,
                        range_end: 1.0,
                        count: 3,
                    },
                    HistogramBucket {
                        range_start: 1.0,
                        range_end: 2.0,
                        count: 5,
                    },
                ],
            }],
            log_scale: false,
            excluded_non_positive: 0,
        };
        let chart = render_histogram(&data, 200, 150).unwrap();
        assert_eq!(chart.width, 200);
        assert_eq!(chart.height, 150);
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }

    #[test]
    fn render_histogram_log_scale_produces_correctly_sized_buffer() {
        let data = HistogramData {
            column_label: "Area".into(),
            series: vec![HistogramSeries {
                label: String::new(),
                buckets: vec![
                    HistogramBucket {
                        range_start: 10.0,
                        range_end: 100.0,
                        count: 3,
                    },
                    HistogramBucket {
                        range_start: 100.0,
                        range_end: 1_000_000.0,
                        count: 5,
                    },
                ],
            }],
            log_scale: true,
            excluded_non_positive: 2,
        };
        let chart = render_histogram(&data, 200, 150).unwrap();
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }

    #[test]
    fn render_scatter_produces_correctly_sized_buffer() {
        let data = ScatterData {
            x_label: "Area".into(),
            y_label: "Circularity".into(),
            points: vec![
                ScatterPoint {
                    x: 1.0,
                    y: 0.5,
                    group: Some("A".into()),
                },
                ScatterPoint {
                    x: 2.0,
                    y: 0.8,
                    group: Some("B".into()),
                },
            ],
            sampled_from: None,
        };
        let chart = render_scatter(&data, 200, 150).unwrap();
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }

    #[test]
    fn render_heatmap_produces_correctly_sized_buffer() {
        let data = HeatmapData {
            x_label: "X position (px)".into(),
            y_label: "Y position (px)".into(),
            value_label: "Count".into(),
            cols: 2,
            rows: 2,
            cell_size: 10.0,
            x_min: 0.0,
            y_min: 0.0,
            cells: vec![
                HeatmapCell {
                    count: 2,
                    value: 2.0,
                },
                HeatmapCell {
                    count: 0,
                    value: 0.0,
                },
                HeatmapCell {
                    count: 0,
                    value: 0.0,
                },
                HeatmapCell {
                    count: 1,
                    value: 1.0,
                },
            ],
        };
        let chart = render_heatmap(
            &data,
            HeatmapColorScheme::Viridis,
            HeatmapRange::Auto,
            200,
            150,
        )
        .unwrap();
        assert_eq!(chart.width, 200);
        assert_eq!(chart.height, 150);
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }

    #[test]
    fn render_heatmap_auto_range_reports_zero_to_data_max() {
        let data = HeatmapData {
            x_label: "X position (px)".into(),
            y_label: "Y position (px)".into(),
            value_label: "Count".into(),
            cols: 1,
            rows: 1,
            cell_size: 10.0,
            x_min: 0.0,
            y_min: 0.0,
            cells: vec![HeatmapCell {
                count: 3,
                value: 7.5,
            }],
        };
        let chart = render_heatmap(
            &data,
            HeatmapColorScheme::Viridis,
            HeatmapRange::Auto,
            200,
            150,
        )
        .unwrap();
        assert_eq!(chart.heatmap_range, Some((0.0, 7.5)));
    }

    #[test]
    fn render_heatmap_manual_range_is_reported_back_verbatim() {
        let data = HeatmapData {
            x_label: "X position (px)".into(),
            y_label: "Y position (px)".into(),
            value_label: "Count".into(),
            cols: 1,
            rows: 1,
            cell_size: 10.0,
            x_min: 0.0,
            y_min: 0.0,
            // Deliberately far outside the manual range below - manual mode
            // must not let the data's own value influence the reported range.
            cells: vec![HeatmapCell {
                count: 3,
                value: 500.0,
            }],
        };
        let manual = HeatmapRange::Manual {
            min: 2.0,
            max: 10.0,
        };
        let chart = render_heatmap(&data, HeatmapColorScheme::Viridis, manual, 200, 150).unwrap();
        assert_eq!(chart.heatmap_range, Some((2.0, 10.0)));
    }

    #[test]
    fn heatmap_range_manual_inverted_or_degenerate_bounds_still_resolve_to_a_valid_span() {
        let cells = [HeatmapCell {
            count: 1,
            value: 5.0,
        }];
        // max <= min must not divide by zero downstream - collapses to a
        // hairline span just above min instead.
        let (min, max) = HeatmapRange::Manual { min: 5.0, max: 5.0 }.resolve(&cells);
        assert_eq!(min, 5.0);
        assert!(max > min);

        let (min, max) = HeatmapRange::Manual { min: 5.0, max: 1.0 }.resolve(&cells);
        assert_eq!(min, 5.0);
        assert!(max > min);
    }

    #[test]
    fn heatmap_range_auto_ignores_manual_style_values_and_always_starts_at_zero() {
        let cells = [
            HeatmapCell {
                count: 1,
                value: -3.0,
            },
            HeatmapCell {
                count: 1,
                value: 12.0,
            },
        ];
        assert_eq!(HeatmapRange::Auto.resolve(&cells), (0.0, 12.0));
    }

    #[test]
    fn save_rendered_chart_png_writes_a_readable_png() {
        let data = HistogramData {
            column_label: "Area".into(),
            series: vec![HistogramSeries {
                label: String::new(),
                buckets: vec![HistogramBucket {
                    range_start: 0.0,
                    range_end: 1.0,
                    count: 3,
                }],
            }],
            log_scale: false,
            excluded_non_positive: 0,
        };
        let chart = render_histogram(&data, 64, 48).unwrap();
        let dir =
            std::env::temp_dir().join(format!("evanalyzer_chart_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chart.png");

        save_rendered_chart_png(&chart, &path).unwrap();
        let decoded = image::open(&path).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 48);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- hover hit-testing ----

    #[test]
    fn histogram_hit_test_identifies_bucket_under_cursor() {
        let data = HistogramData {
            column_label: "Area".into(),
            series: vec![HistogramSeries {
                label: String::new(),
                buckets: vec![
                    HistogramBucket {
                        range_start: 0.0,
                        range_end: 10.0,
                        count: 3,
                    },
                    HistogramBucket {
                        range_start: 10.0,
                        range_end: 20.0,
                        count: 9,
                    },
                ],
            }],
            log_scale: false,
            excluded_non_positive: 0,
        };
        let chart = render_histogram(&data, 800, 500).unwrap();
        let tester = chart
            .hit_test
            .expect("histogram should produce a hit tester");

        // Margins guarantee the image corners are always outside the plot area.
        assert!(tester.hit_test(0.0, 0.0).is_none());
        assert!(tester.hit_test(799.0, 499.0).is_none());

        // Self-calibrate the plot area's horizontal extent at mid-height
        // instead of hardcoding margin pixel widths.
        let y = 250.0;
        let mut left = None;
        let mut right = None;
        for x in 0..800 {
            if tester.hit_test(x as f64, y).is_some() {
                left.get_or_insert(x);
                right = Some(x);
            }
        }
        let (left, right) = (left.unwrap() as f64, right.unwrap() as f64);

        let near_left = tester.hit_test(left + (right - left) * 0.1, y).unwrap();
        let near_right = tester.hit_test(left + (right - left) * 0.9, y).unwrap();
        assert!(
            near_left.contains("Count: 3"),
            "left side should be the first bucket: {near_left}"
        );
        assert!(
            near_right.contains("Count: 9"),
            "right side should be the second bucket: {near_right}"
        );
    }

    #[test]
    fn histogram_hit_test_reports_one_line_per_group_when_color_by_is_set() {
        let data = HistogramData {
            column_label: "Area".into(),
            series: vec![
                HistogramSeries {
                    label: "ClassA".into(),
                    buckets: vec![HistogramBucket {
                        range_start: 0.0,
                        range_end: 10.0,
                        count: 3,
                    }],
                },
                HistogramSeries {
                    label: "ClassB".into(),
                    buckets: vec![HistogramBucket {
                        range_start: 0.0,
                        range_end: 10.0,
                        count: 7,
                    }],
                },
            ],
            log_scale: false,
            excluded_non_positive: 0,
        };
        let chart = render_histogram(&data, 800, 500).unwrap();
        let tester = chart
            .hit_test
            .expect("histogram should produce a hit tester");

        let y = 250.0;
        let hit = (0..800)
            .find_map(|x| tester.hit_test(x as f64, y))
            .expect("single bucket spans the whole plot width");
        assert!(hit.contains("ClassA: 3"), "expected a ClassA line: {hit}");
        assert!(hit.contains("ClassB: 7"), "expected a ClassB line: {hit}");
    }

    #[test]
    fn scatter_hit_test_finds_nearest_point() {
        let data = ScatterData {
            x_label: "X".into(),
            y_label: "Y".into(),
            points: vec![
                ScatterPoint {
                    x: 1.0,
                    y: 1.0,
                    group: None,
                },
                ScatterPoint {
                    x: 9.0,
                    y: 9.0,
                    group: Some("A".into()),
                },
            ],
            sampled_from: None,
        };
        let chart = render_scatter(&data, 800, 500).unwrap();
        let tester = chart.hit_test.expect("scatter should produce a hit tester");

        let mut hits = std::collections::HashSet::new();
        for x in (0..800).step_by(4) {
            for y in (0..500).step_by(4) {
                if let Some(s) = tester.hit_test(x as f64, y as f64) {
                    hits.insert(s);
                }
            }
        }
        assert!(
            hits.iter()
                .any(|s| s.contains("X: 1.00") && s.contains("Y: 1.00")),
            "expected a hit near (1,1): {hits:?}"
        );
        assert!(
            hits.iter()
                .any(|s| s.contains("X: 9.00") && s.contains("Y: 9.00") && s.contains('A')),
            "expected a hit near (9,9) labeled with its group: {hits:?}"
        );

        assert!(tester.hit_test(0.0, 0.0).is_none());
    }

    #[test]
    fn heatmap_hit_test_reports_cell_value_and_skips_empty_cells() {
        let data = HeatmapData {
            x_label: "X position (px)".into(),
            y_label: "Y position (px)".into(),
            value_label: "Area (px²)".into(),
            cols: 2,
            rows: 2,
            cell_size: 10.0,
            x_min: 0.0,
            y_min: 0.0,
            cells: vec![
                HeatmapCell {
                    count: 2,
                    value: 2.0,
                },
                HeatmapCell {
                    count: 0,
                    value: 0.0,
                },
                HeatmapCell {
                    count: 0,
                    value: 0.0,
                },
                HeatmapCell {
                    count: 5,
                    value: 5.0,
                },
            ],
        };
        let chart = render_heatmap(
            &data,
            HeatmapColorScheme::Viridis,
            HeatmapRange::Auto,
            800,
            500,
        )
        .unwrap();
        let tester = chart.hit_test.expect("heatmap should produce a hit tester");

        let mut hits = std::collections::HashSet::new();
        for x in (0..800).step_by(4) {
            for y in (0..500).step_by(4) {
                if let Some(s) = tester.hit_test(x as f64, y as f64) {
                    hits.insert(s);
                }
            }
        }
        assert!(
            hits.iter()
                .any(|s| s.contains("Area (px²): 2.00") && s.contains("Count: 2")),
            "{hits:?}"
        );
        assert!(
            hits.iter()
                .any(|s| s.contains("Area (px²): 5.00") && s.contains("Count: 5")),
            "{hits:?}"
        );
        assert!(
            !hits.iter().any(|s| s.contains("Count: 0")),
            "empty cells shouldn't show a tooltip"
        );
    }
}
