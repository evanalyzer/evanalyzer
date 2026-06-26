//! Histogram and scatter charts over the results table's ROI data.
//!
//! Mirrors `results_loader.rs`'s split: plain data structs + pure functions
//! first (unit-testable without any plotting dependency), then a thin
//! `plotters` rendering layer. The two public entry points per chart type —
//! `render_*` (into an in-memory RGB buffer, for the GUI) and `save_*_png`
//! (straight to a file, for a future CLI command) — call the *same* internal
//! `draw_*` function on a different `BitMapBackend`, so there is exactly one
//! drawing implementation per chart type regardless of caller.
use crate::results::results_loader::{
    coloc_label, compute_class, is_colocalized, is_numeric_metric, numeric_value, ColumnSpec,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::RoiRow;
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

#[derive(Debug, Clone)]
pub struct HistogramData {
    pub column_label: String,
    pub buckets: Vec<HistogramBucket>,
}

/// Buckets every ROI's `column_id` value into `bucket_count` equal-width
/// buckets spanning [min, max]. `None` if the column isn't found, no ROI has
/// a value for it, or `bucket_count` is 0.
pub fn compute_histogram(
    rois: &[RoiRow],
    column_id: &str,
    columns: &[ColumnSpec],
    bucket_count: usize,
) -> Option<HistogramData> {
    if bucket_count == 0 {
        return None;
    }
    let column_label = columns.iter().find(|c| c.id == column_id)?.label.clone();
    let values: Vec<f64> = rois.iter().filter_map(|r| numeric_value(r, column_id)).collect();
    if values.is_empty() {
        return None;
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let buckets = if max <= min {
        // Every ROI has the same value — one bucket holds them all.
        vec![HistogramBucket {
            range_start: min,
            range_end: min,
            count: values.len(),
        }]
    } else {
        let width = (max - min) / bucket_count as f64;
        let mut counts = vec![0usize; bucket_count];
        for v in &values {
            // The max value lands exactly on the last bucket's upper edge;
            // clamp it into that bucket instead of overflowing.
            let idx = (((v - min) / width) as usize).min(bucket_count - 1);
            counts[idx] += 1;
        }
        counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| HistogramBucket {
                range_start: min + i as f64 * width,
                range_end: min + (i + 1) as f64 * width,
                count,
            })
            .collect()
    };

    Some(HistogramData {
        column_label,
        buckets,
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

/// Pairs each ROI's `x_col`/`y_col` values into a scatter point (skipping
/// ROIs missing either value), optionally tagging each point with its class
/// or colocalized label for coloring. `None` if either column isn't found or
/// no ROI has both values.
///
/// When there are more than `max_points` points, deterministically samples an
/// evenly-strided subset (not random) so re-rendering the same filter is
/// stable. `max_points == 0` disables the cap.
pub fn compute_scatter(
    rois: &[RoiRow],
    x_col: &str,
    y_col: &str,
    color_by: ColorBy,
    columns: &[ColumnSpec],
    max_points: usize,
) -> Option<ScatterData> {
    let x_label = columns.iter().find(|c| c.id == x_col)?.label.clone();
    let y_label = columns.iter().find(|c| c.id == y_col)?.label.clone();

    let mut points: Vec<ScatterPoint> = rois
        .iter()
        .filter_map(|roi| {
            let x = numeric_value(roi, x_col)?;
            let y = numeric_value(roi, y_col)?;
            let group = match color_by {
                ColorBy::None => None,
                ColorBy::Class => Some(compute_class(roi)),
                ColorBy::Colocalized => Some(coloc_label(is_colocalized(&roi.coloc_json))),
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

// ---------------------------------------------------------------------------
// Rendering (plotters)
// ---------------------------------------------------------------------------

/// A rendered chart as a row-major RGB8 buffer — the layout
/// `slint::SharedPixelBuffer<Rgb8Pixel>` consumes directly in the GUI (see
/// `crates/gui/src/editor/viewport_controller.rs` for the established
/// pattern), and what `image::save_buffer` would need for a CLI PNG export.
pub struct RenderedChart {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
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

fn draw_histogram(
    root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &HistogramData,
) -> Result<(), InternalErrors> {
    root.fill(&WHITE).map_err(to_internal_error)?;

    let max_count = data.buckets.iter().map(|b| b.count).max().unwrap_or(0);
    let x_min = data.buckets.first().map_or(0.0, |b| b.range_start);
    let x_max_raw = data.buckets.last().map_or(1.0, |b| b.range_end);
    let x_max = if x_max_raw > x_min { x_max_raw } else { x_min + 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .caption(format!("Histogram: {}", data.column_label), ("sans-serif", 20))
        .build_cartesian_2d(x_min..x_max, 0f64..(max_count as f64 * 1.1).max(1.0))
        .map_err(to_internal_error)?;

    chart
        .configure_mesh()
        .x_desc(data.column_label.clone())
        .y_desc("Count")
        .draw()
        .map_err(to_internal_error)?;

    chart
        .draw_series(data.buckets.iter().map(|b| {
            Rectangle::new(
                [(b.range_start, 0.0), (b.range_end, b.count as f64)],
                PALETTE[0].filled(),
            )
        }))
        .map_err(to_internal_error)?;

    root.present().map_err(to_internal_error)?;
    Ok(())
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

fn draw_scatter(
    root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &ScatterData,
) -> Result<(), InternalErrors> {
    root.fill(&WHITE).map_err(to_internal_error)?;

    let (x_min, x_max) = padded_axis_bounds(data.points.iter().map(|p| p.x));
    let (y_min, y_max) = padded_axis_bounds(data.points.iter().map(|p| p.y));

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .caption(format!("{} vs {}", data.y_label, data.x_label), ("sans-serif", 20))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(to_internal_error)?;

    chart
        .configure_mesh()
        .x_desc(data.x_label.clone())
        .y_desc(data.y_label.clone())
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
            .draw()
            .map_err(to_internal_error)?;
    }

    root.present().map_err(to_internal_error)?;
    Ok(())
}

/// Renders into an in-memory RGB8 buffer — used by the GUI to build a
/// `slint::Image` without touching disk.
pub fn render_histogram(data: &HistogramData, width: u32, height: u32) -> Result<RenderedChart, InternalErrors> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut rgb, (width, height)).into_drawing_area();
        draw_histogram(root, data)?;
    }
    Ok(RenderedChart { width, height, rgb })
}

/// Renders into an in-memory RGB8 buffer — used by the GUI to build a
/// `slint::Image` without touching disk.
pub fn render_scatter(data: &ScatterData, width: u32, height: u32) -> Result<RenderedChart, InternalErrors> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut rgb, (width, height)).into_drawing_area();
        draw_scatter(root, data)?;
    }
    Ok(RenderedChart { width, height, rgb })
}

/// Renders straight to a PNG file. Not called anywhere yet — exists so a
/// future CLI "export chart" command is a thin wrapper around the same
/// drawing code the GUI uses, instead of new plotting logic.
pub fn save_histogram_png(data: &HistogramData, width: u32, height: u32, path: &Path) -> Result<(), InternalErrors> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    draw_histogram(root, data)
}

/// Renders straight to a PNG file. Not called anywhere yet — exists so a
/// future CLI "export chart" command is a thin wrapper around the same
/// drawing code the GUI uses, instead of new plotting logic.
pub fn save_scatter_png(data: &ScatterData, width: u32, height: u32, path: &Path) -> Result<(), InternalErrors> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    draw_scatter(root, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_roi(area_px: u64, intensities_json: &str, object_class_name: Vec<String>, coloc_json: &str) -> RoiRow {
        RoiRow {
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
        assert!(!ids.contains(&"roi_id"), "non-numeric column excluded");
    }

    // ---- compute_histogram ----

    #[test]
    fn compute_histogram_buckets_values_across_range() {
        let rois = vec![
            make_roi(0, "{}", vec![], "{}"),
            make_roi(10, "{}", vec![], "{}"),
            make_roi(50, "{}", vec![], "{}"),
            make_roi(99, "{}", vec![], "{}"),
        ];
        let columns = base_columns();
        let data = compute_histogram(&rois, "area_px", &columns, 10).unwrap();
        assert_eq!(data.column_label, "Area (px²)");
        assert_eq!(data.buckets.len(), 10);
        let total: usize = data.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 4);
        // width = (99-0)/10 = 9.9: 0 -> bucket 0, 10 -> bucket 1, 50 -> bucket 5, 99 -> bucket 9.
        assert_eq!(data.buckets[0].count, 1);
        assert_eq!(data.buckets[1].count, 1);
        assert_eq!(data.buckets[5].count, 1);
        assert_eq!(data.buckets[9].count, 1, "max value (99) clamps into the last bucket");
    }

    #[test]
    fn compute_histogram_identical_values_single_bucket() {
        let rois = vec![make_roi(5, "{}", vec![], "{}"), make_roi(5, "{}", vec![], "{}")];
        let data = compute_histogram(&rois, "area_px", &base_columns(), 10).unwrap();
        assert_eq!(data.buckets.len(), 1);
        assert_eq!(data.buckets[0].count, 2);
    }

    #[test]
    fn compute_histogram_unknown_column_or_empty_rois_is_none() {
        let columns = base_columns();
        assert!(compute_histogram(&[], "area_px", &columns, 10).is_none());
        assert!(compute_histogram(&[make_roi(1, "{}", vec![], "{}")], "does_not_exist", &columns, 10).is_none());
        assert!(compute_histogram(&[make_roi(1, "{}", vec![], "{}")], "area_px", &columns, 0).is_none());
    }

    // ---- compute_scatter ----

    #[test]
    fn compute_scatter_pairs_x_and_y_and_colors_by_class() {
        let rois = vec![
            make_roi(10, "{}", vec!["ClassA".into()], "{}"),
            make_roi(20, "{}", vec!["ClassB".into()], "{}"),
        ];
        let columns = base_columns();
        let data = compute_scatter(&rois, "area_px", "circularity", ColorBy::Class, &columns, 0).unwrap();
        assert_eq!(data.x_label, "Area (px²)");
        assert_eq!(data.y_label, "Circularity");
        assert_eq!(data.points.len(), 2);
        assert_eq!(data.points[0].group.as_deref(), Some("ClassA"));
        assert_eq!(data.points[1].group.as_deref(), Some("ClassB"));
        assert!(data.sampled_from.is_none());
    }

    #[test]
    fn compute_scatter_colors_by_colocalized() {
        let mut coloc_roi = make_roi(10, "{}", vec![], "{}");
        coloc_roi.coloc_json = r#"{"Target (1)":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let not_coloc_roi = make_roi(20, "{}", vec![], "{}");
        let columns = base_columns();
        let data = compute_scatter(
            &[coloc_roi, not_coloc_roi],
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
        let rois: Vec<RoiRow> = (0..100).map(|i| make_roi(i, "{}", vec![], "{}")).collect();
        let columns = base_columns();
        let data = compute_scatter(&rois, "area_px", "circularity", ColorBy::None, &columns, 10).unwrap();
        assert_eq!(data.points.len(), 10);
        assert_eq!(data.sampled_from, Some(100));

        // Deterministic: re-computing yields the identical sampled subset.
        let again = compute_scatter(&rois, "area_px", "circularity", ColorBy::None, &columns, 10).unwrap();
        assert_eq!(data.points, again.points);
    }

    #[test]
    fn compute_scatter_unknown_column_is_none() {
        let rois = vec![make_roi(1, "{}", vec![], "{}")];
        let columns = base_columns();
        assert!(compute_scatter(&rois, "does_not_exist", "circularity", ColorBy::None, &columns, 0).is_none());
    }

    // ---- rendering smoke tests ----

    #[test]
    fn render_histogram_produces_correctly_sized_buffer() {
        let data = HistogramData {
            column_label: "Area".into(),
            buckets: vec![
                HistogramBucket { range_start: 0.0, range_end: 1.0, count: 3 },
                HistogramBucket { range_start: 1.0, range_end: 2.0, count: 5 },
            ],
        };
        let chart = render_histogram(&data, 200, 150).unwrap();
        assert_eq!(chart.width, 200);
        assert_eq!(chart.height, 150);
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }

    #[test]
    fn render_scatter_produces_correctly_sized_buffer() {
        let data = ScatterData {
            x_label: "Area".into(),
            y_label: "Circularity".into(),
            points: vec![
                ScatterPoint { x: 1.0, y: 0.5, group: Some("A".into()) },
                ScatterPoint { x: 2.0, y: 0.8, group: Some("B".into()) },
            ],
            sampled_from: None,
        };
        let chart = render_scatter(&data, 200, 150).unwrap();
        assert_eq!(chart.rgb.len(), 200 * 150 * 3);
    }
}
