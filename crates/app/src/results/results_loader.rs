use evanalyzer_cfg::core_types::InternalErrors;
pub use evanalyzer_core::RoiRow;
use evanalyzer_core::{DuckDbReader, RoiFilter};
use std::collections::BTreeMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Column specification — shared between GUI, CLI, and export
// ---------------------------------------------------------------------------

/// Describes one column in the results table.
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub id: String,
    pub label: String,
    pub filterable: bool,
    /// Whether this column's data should be populated (visible in UI / included in export).
    /// When false, `to_display_row` writes an empty string for this column without
    /// doing any parsing work, and `DatabaseFilter::needs_intensities` can be set
    /// to false if no channel column is visible.
    pub visible: bool,
}

/// One display-ready row: `values[i]` corresponds to `ColumnSpec[i]`.
#[derive(Debug, Clone)]
pub struct DisplayRow {
    pub roi_id: i32,
    /// Pre-formatted string values, one per `ColumnSpec` (in the same order).
    /// Hidden columns have an empty string here.
    pub values: Vec<String>,
}

// ---------------------------------------------------------------------------
// Column helpers — channel discovery and spec construction
// ---------------------------------------------------------------------------

/// Discovers which channel indices appear in `intensities_json` across the
/// provided rows. The returned list is sorted numerically.
pub fn discover_channels(rois: &[RoiRow]) -> Vec<i32> {
    let mut channels = std::collections::BTreeSet::new();
    for roi in rois {
        if roi.intensities_json.is_empty() || roi.intensities_json == "{}" {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&roi.intensities_json) {
            if let Some(obj) = val.as_object() {
                for key in obj.keys() {
                    if let Ok(ch) = key.parse::<i32>() {
                        channels.insert(ch);
                    }
                }
            }
        }
    }
    channels.into_iter().collect()
}

/// Builds the full ordered list of [`ColumnSpec`] for the given channel list.
/// All columns start as visible.
pub fn build_column_specs(channels: &[i32]) -> Vec<ColumnSpec> {
    let mut cols = vec![
        ColumnSpec {
            id: "roi_id".into(),
            label: "ROI ID".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "image".into(),
            label: "Image".into(),
            filterable: true,
            visible: true,
        },
        ColumnSpec {
            id: "class".into(),
            label: "Class".into(),
            filterable: true,
            visible: true,
        },
        ColumnSpec {
            id: "area_px".into(),
            label: "Area (px\u{00B2})".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "area_nm2".into(),
            label: "Area (nm\u{00B2})".into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "circularity".into(),
            label: "Circularity".into(),
            filterable: false,
            visible: true,
        },
    ];
    for &ch in channels {
        cols.push(ColumnSpec {
            id: format!("ch{ch}_min_bit"),
            label: format!("Ch{ch} Min (bit)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("ch{ch}_max_bit"),
            label: format!("Ch{ch} Max (bit)"),
            filterable: false,
            visible: true,
        });
        cols.push(ColumnSpec {
            id: format!("ch{ch}_avg_bit"),
            label: format!("Ch{ch} Avg (bit)"),
            filterable: false,
            visible: true,
        });
    }
    cols
}

// ---------------------------------------------------------------------------
// Row conversion
// ---------------------------------------------------------------------------

/// Converts a [`RoiRow`] into a [`DisplayRow`] using the given column specs.
/// Only visible columns have their values computed; hidden columns get `""`.
/// `intensities_json` is parsed at most once per call, and only if at least
/// one channel column is visible.
pub fn to_display_row(row_idx: usize, roi: &RoiRow, columns: &[ColumnSpec]) -> DisplayRow {
    let needs_intensities = columns.iter().any(|c| c.visible && c.id.starts_with("ch"));

    let intensities: BTreeMap<i32, (f64, f64, f64)> = if needs_intensities {
        parse_intensities(&roi.intensities_json)
    } else {
        BTreeMap::new()
    };

    let class = compute_class(roi);

    let values = columns
        .iter()
        .map(|col| {
            if !col.visible {
                return String::new();
            }
            match col.id.as_str() {
                "roi_id" => roi.object_id.to_string(),
                "image" => roi.image_name.clone(),
                "class" => class.clone(),
                "area_px" => roi.area_px.to_string(),
                "area_nm2" => format!("{:.2}", roi.area_nm2),
                "circularity" => format!("{:.3}", roi.circularity),
                id => channel_value(id, &intensities),
            }
        })
        .collect();

    DisplayRow {
        roi_id: (row_idx + 1) as i32,
        values,
    }
}

fn compute_class(roi: &RoiRow) -> String {
    if roi.object_class_name.is_empty() {
        roi.seg_class_name.clone().unwrap_or_default()
    } else {
        roi.object_class_name.join(", ")
    }
}

/// Parses `intensities_json` into a map of channel → (min_scaled, max_scaled, mean_scaled).
fn parse_intensities(json: &str) -> BTreeMap<i32, (f64, f64, f64)> {
    let mut result = BTreeMap::new();
    if json.is_empty() || json == "{}" {
        return result;
    }
    let Ok(val) = serde_json::from_str::<serde_json::Value>(json) else {
        return result;
    };
    let Some(obj) = val.as_object() else {
        return result;
    };
    for (ch_str, stats) in obj {
        let Ok(ch) = ch_str.parse::<i32>() else {
            continue;
        };
        result.insert(
            ch,
            (
                stats["min_scaled"].as_f64().unwrap_or(0.0),
                stats["max_scaled"].as_f64().unwrap_or(0.0),
                stats["mean_scaled"].as_f64().unwrap_or(0.0),
            ),
        );
    }
    result
}

/// Extracts a formatted value for channel-intensity columns (e.g. `ch0_min_bit`).
fn channel_value(col_id: &str, intensities: &BTreeMap<i32, (f64, f64, f64)>) -> String {
    let rest = match col_id.strip_prefix("ch") {
        Some(r) => r,
        None => return String::new(),
    };
    let under = match rest.find('_') {
        Some(i) => i,
        None => return String::new(),
    };
    let ch: i32 = match rest[..under].parse() {
        Ok(n) => n,
        Err(_) => return String::new(),
    };
    let stat = &rest[under + 1..];
    let Some(&(min, max, avg)) = intensities.get(&ch) else {
        return String::new();
    };
    match stat {
        "min_bit" => format!("{:.0}", min),
        "max_bit" => format!("{:.0}", max),
        "avg_bit" => format!("{:.1}", avg),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Database filter and loader
// ---------------------------------------------------------------------------

pub struct DatabaseFilter {
    /// `None` = no image filter; `Some([])` = nothing passes; `Some([..])` = restrict to these.
    pub image_filter: Option<Vec<String>>,
    /// `None` = no class filter; `Some([])` = nothing passes; `Some([..])` = restrict to these.
    pub class_filter: Option<Vec<String>>,
    pub page_size: usize,
    pub page: usize,
    /// When false the database omits `intensities_json` from the SELECT, avoiding
    /// JSON parsing cost when no channel columns are visible.
    pub needs_intensities: bool,
}

impl Default for DatabaseFilter {
    fn default() -> Self {
        Self {
            image_filter: None,
            class_filter: None,
            page_size: 500,
            page: 0,
            needs_intensities: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Grouping & aggregation
// ---------------------------------------------------------------------------

/// How rows are bucketed into groups before aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBy {
    /// No grouping — caller should display the raw per-ROI rows.
    None,
    /// One group per distinct `image_name`.
    Image,
    /// One group per folder (the directory portion of `image_rel_path`).
    Folder,
    /// One group per key extracted from `image_name` by a regex. The first
    /// capture group is used as the key; if the pattern has no capture group
    /// the whole match is used. Rows whose name does not match are dropped.
    Regex,
}

/// Aggregate applied to every numeric column when grouping is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Min,
    Max,
    Avg,
    Median,
    Stdev,
    Sum,
}

impl AggFunc {
    /// Short suffix shown in grouped column headers, e.g. `Area (px²) [median]`.
    /// Also used as the id fragment in grouped column ids (`area_px__median`).
    pub fn label(self) -> &'static str {
        match self {
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Avg => "avg",
            AggFunc::Median => "median",
            AggFunc::Stdev => "stdev",
            AggFunc::Sum => "sum",
        }
    }

    /// Inverse of [`AggFunc::label`].
    fn from_label(s: &str) -> Option<AggFunc> {
        Some(match s {
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            "avg" => AggFunc::Avg,
            "median" => AggFunc::Median,
            "stdev" => AggFunc::Stdev,
            "sum" => AggFunc::Sum,
            _ => return None,
        })
    }
}

/// Configuration for a grouped/aggregated results view.
#[derive(Debug, Clone)]
pub struct GroupConfig {
    pub group_by: GroupBy,
    /// Regex pattern, only used when `group_by == GroupBy::Regex`.
    pub regex: String,
    /// Aggregates to compute. Each visible numeric metric gets one grouped
    /// column per entry here (e.g. selecting Min+Max+Avg yields three columns
    /// per metric).
    pub aggs: Vec<AggFunc>,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            group_by: GroupBy::None,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
        }
    }
}

/// Returns the directory portion of a relative image path (`""` if none).
fn folder_of(rel_path: &str) -> String {
    std::path::Path::new(rel_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(root)".to_string())
}

/// Computes the grouping key for a single ROI, or `None` if the row should be
/// excluded from the grouped view (only happens for a non-matching regex).
fn group_key(roi: &RoiRow, group_by: GroupBy, regex: Option<&regex::Regex>) -> Option<String> {
    match group_by {
        GroupBy::None | GroupBy::Image => Some(roi.image_name.clone()),
        GroupBy::Folder => Some(folder_of(&roi.image_rel_path)),
        GroupBy::Regex => {
            let re = regex?;
            let caps = re.captures(&roi.image_name)?;
            // Prefer the first capture group; fall back to the whole match.
            let key = caps
                .get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().to_string())?;
            Some(key)
        }
    }
}

/// True for per-ROI columns that carry a numeric value we can aggregate.
/// (Excludes `roi_id`, `image`, `class`.)
fn is_numeric_metric(id: &str) -> bool {
    matches!(id, "area_px" | "area_nm2" | "circularity") || id.starts_with("ch")
}

/// Decimal places used when formatting a base metric's aggregate.
fn metric_precision(id: &str) -> usize {
    match id {
        "area_nm2" => 2,
        "circularity" => 3,
        _ => 1, // area_px and channel bit values
    }
}

/// Pulls the numeric value of a base metric column (e.g. `area_px`,
/// `ch0_min_bit`) from a single ROI, or `None` if absent.
fn metric_value(
    id: &str,
    roi: &RoiRow,
    intensities: &BTreeMap<i32, (f64, f64, f64)>,
) -> Option<f64> {
    match id {
        "area_px" => Some(roi.area_px as f64),
        "area_nm2" => Some(roi.area_nm2),
        "circularity" => Some(roi.circularity),
        _ => {
            let rest = id.strip_prefix("ch")?;
            let under = rest.find('_')?;
            let ch: i32 = rest[..under].parse().ok()?;
            let &(min, max, avg) = intensities.get(&ch)?;
            match &rest[under + 1..] {
                "min_bit" => Some(min),
                "max_bit" => Some(max),
                "avg_bit" => Some(avg),
                _ => None,
            }
        }
    }
}

/// Applies an aggregate to a slice of values. Returns `0.0` for an empty slice.
fn apply_agg(values: &[f64], agg: AggFunc) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match agg {
        AggFunc::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggFunc::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggFunc::Sum => values.iter().sum(),
        AggFunc::Avg => values.iter().sum::<f64>() / values.len() as f64,
        AggFunc::Median => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        AggFunc::Stdev => {
            // Sample standard deviation (n-1), matching spreadsheet STDEV.
            if values.len() < 2 {
                return 0.0;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (values.len() - 1) as f64;
            var.sqrt()
        }
    }
}

/// Groups `rois` according to `config` and produces one aggregated
/// [`DisplayRow`] per group, alongside the matching column specs.
///
/// `base_specs` are the per-ROI column specs (as used by [`to_display_row`]);
/// only the **visible** numeric metrics among them become grouped columns, so
/// the column-visibility filter applies in grouped mode too. Each such metric
/// gets one column per entry in `config.aggs` (label/id carry the aggregate,
/// e.g. `area_px__median` / `Area (px²) [median]`). Groups are returned in key
/// order, always preceded by the group-key and ROI-count columns.
pub fn aggregate_rows(
    rois: &[RoiRow],
    config: &GroupConfig,
    base_specs: &[ColumnSpec],
) -> (Vec<ColumnSpec>, Vec<DisplayRow>) {
    let group_label = match config.group_by {
        GroupBy::Folder => "Folder",
        GroupBy::Regex => "Group",
        _ => "Image",
    };

    // Visible numeric metrics, in their original column order.
    let metrics: Vec<&ColumnSpec> = base_specs
        .iter()
        .filter(|c| c.visible && is_numeric_metric(&c.id))
        .collect();

    // Grouped columns: key, count, then each metric × each selected aggregate.
    let mut specs = vec![
        ColumnSpec {
            id: "group".into(),
            label: group_label.into(),
            filterable: false,
            visible: true,
        },
        ColumnSpec {
            id: "count".into(),
            label: "ROIs".into(),
            filterable: false,
            visible: true,
        },
    ];
    for m in &metrics {
        for agg in &config.aggs {
            specs.push(ColumnSpec {
                id: format!("{}__{}", m.id, agg.label()),
                label: format!("{} [{}]", m.label, agg.label()),
                filterable: false,
                visible: true,
            });
        }
    }

    let regex = if config.group_by == GroupBy::Regex {
        regex::Regex::new(&config.regex).ok()
    } else {
        None
    };
    let needs_intensities = metrics.iter().any(|m| m.id.starts_with("ch"));

    // group key -> (ROI count, base metric id -> collected values)
    let mut groups: BTreeMap<String, (usize, BTreeMap<String, Vec<f64>>)> = BTreeMap::new();
    for roi in rois {
        let Some(key) = group_key(roi, config.group_by, regex.as_ref()) else {
            continue;
        };
        let entry = groups.entry(key).or_default();
        entry.0 += 1;

        let intensities = if needs_intensities {
            parse_intensities(&roi.intensities_json)
        } else {
            BTreeMap::new()
        };
        for m in &metrics {
            if let Some(v) = metric_value(&m.id, roi, &intensities) {
                entry.1.entry(m.id.clone()).or_default().push(v);
            }
        }
    }

    let rows = groups
        .into_iter()
        .enumerate()
        .map(|(idx, (key, (count, metric_vals)))| {
            let values = specs
                .iter()
                .map(|col| match col.id.as_str() {
                    "group" => key.clone(),
                    "count" => count.to_string(),
                    id => {
                        // id is "<base_id>__<agg>"
                        let Some((base_id, agg_label)) = id.rsplit_once("__") else {
                            return String::new();
                        };
                        let Some(agg) = AggFunc::from_label(agg_label) else {
                            return String::new();
                        };
                        match metric_vals.get(base_id) {
                            Some(v) if !v.is_empty() => {
                                format!("{:.*}", metric_precision(base_id), apply_agg(v, agg))
                            }
                            _ => String::new(),
                        }
                    }
                })
                .collect();
            DisplayRow {
                roi_id: (idx + 1) as i32,
                values,
            }
        })
        .collect();

    (specs, rows)
}

pub struct ResultsLoader {
    path: PathBuf,
}

impl ResultsLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn get_rois(&self, filter: DatabaseFilter) -> Result<Vec<RoiRow>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_rois(&RoiFilter {
            image_filter: filter.image_filter,
            class_filter: filter.class_filter,
            page_size: filter.page_size,
            page: filter.page,
            fetch_intensities: filter.needs_intensities,
        })
    }

    pub fn get_image_names(&self) -> Result<Vec<String>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_image_names()
    }

    pub fn get_class_names(&self) -> Result<Vec<String>, InternalErrors> {
        DuckDbReader::open(&self.path)?.get_class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn make_roi(
        image_name: &str,
        object_class_name: Vec<String>,
        seg_class_name: Option<&str>,
        area_px: u64,
        area_nm2: f64,
        circularity: f64,
        intensities_json: &str,
    ) -> RoiRow {
        RoiRow {
            image_name: image_name.into(),
            image_rel_path: String::new(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: "00000000-0000-0000-0000-000000000001".into(),
            seg_class_name: seg_class_name.map(str::to_owned),
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
            area_nm2,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: intensities_json.into(),
            coloc_json: "{}".into(),
        }
    }

    // ---- discover_channels ----

    #[test]
    fn discover_channels_empty_input() {
        assert_eq!(discover_channels(&[]), Vec::<i32>::new());
    }

    #[test]
    fn discover_channels_no_intensities() {
        let roi = make_roi("img.tif", vec![], None, 100, 100.0, 0.9, "{}");
        assert_eq!(discover_channels(&[roi]), Vec::<i32>::new());
    }

    #[test]
    fn discover_channels_single_channel() {
        let roi = make_roi(
            "img.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#,
        );
        assert_eq!(discover_channels(&[roi]), vec![0]);
    }

    #[test]
    fn discover_channels_multiple_images_merged_and_sorted() {
        let roi1 = make_roi(
            "a.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"2":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0}}"#,
        );
        let roi2 = make_roi(
            "b.tif",
            vec![],
            None,
            100,
            100.0,
            0.9,
            r#"{"0":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0},"1":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0}}"#,
        );
        assert_eq!(discover_channels(&[roi1, roi2]), vec![0, 1, 2]);
    }

    // ---- build_column_specs ----

    #[test]
    fn build_column_specs_no_channels_has_six_fixed_cols() {
        let specs = build_column_specs(&[]);
        assert_eq!(specs.len(), 6);
        assert_eq!(specs[0].id, "roi_id");
        assert_eq!(specs[5].id, "circularity");
        assert!(specs.iter().all(|c| c.visible));
    }

    #[test]
    fn build_column_specs_with_channels_adds_three_cols_per_channel() {
        let specs = build_column_specs(&[0, 1]);
        assert_eq!(specs.len(), 6 + 3 * 2);
        let ch_ids: Vec<&str> = specs[6..].iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ch_ids,
            [
                "ch0_min_bit",
                "ch0_max_bit",
                "ch0_avg_bit",
                "ch1_min_bit",
                "ch1_max_bit",
                "ch1_avg_bit"
            ]
        );
    }

    // ---- to_display_row ----

    #[test]
    fn to_display_row_basic_fixed_columns() {
        let roi = make_roi(
            "sample.tif",
            vec!["Nucleus".into()],
            None,
            1234,
            567.89,
            0.923,
            "{}",
        );
        let specs = build_column_specs(&[]);
        let row = to_display_row(0, &roi, &specs);

        assert_eq!(row.roi_id, 1); // row_idx=0 → roi_id=1
        assert_eq!(row.values[0], "00000000-0000-0000-0000-000000000001"); // roi_id col
        assert_eq!(row.values[1], "sample.tif"); // image
        assert_eq!(row.values[2], "Nucleus"); // class from object_class_name
        assert_eq!(row.values[3], "1234"); // area_px
        assert_eq!(row.values[4], "567.89"); // area_nm2
        assert_eq!(row.values[5], "0.923"); // circularity
    }

    #[test]
    fn to_display_row_class_falls_back_to_seg_class_when_object_class_empty() {
        let roi = make_roi("img.tif", vec![], Some("Background"), 0, 0.0, 0.0, "{}");
        let specs = build_column_specs(&[]);
        let row = to_display_row(0, &roi, &specs);
        assert_eq!(row.values[2], "Background");
    }

    #[test]
    fn to_display_row_hidden_column_is_empty_string() {
        let roi = make_roi("img.tif", vec![], None, 100, 100.0, 0.5, "{}");
        let mut specs = build_column_specs(&[]);
        specs[3].visible = false; // hide area_px
        let row = to_display_row(0, &roi, &specs);
        assert_eq!(row.values[3], "");
    }

    #[test]
    fn to_display_row_channel_values_parsed_correctly() {
        let json = r#"{"0":{"sum_raw":0.0,"sum_scaled":0.0,"mean_raw":0.0,"mean_scaled":0.0,"median_raw":0.0,"median_scaled":0.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":100.0,"max_raw":0.0,"max_scaled":200.0}}"#;
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, json);
        let specs = build_column_specs(&[0]);
        let row = to_display_row(0, &roi, &specs);
        // index 6 = ch0_min_bit, 7 = ch0_max_bit, 8 = ch0_avg_bit
        assert_eq!(row.values[6], "100"); // min_scaled rounded
        assert_eq!(row.values[7], "200"); // max_scaled rounded
    }

    #[test]
    fn to_display_row_row_idx_increments_roi_id() {
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, "{}");
        let specs = build_column_specs(&[]);
        assert_eq!(to_display_row(0, &roi, &specs).roi_id, 1);
        assert_eq!(to_display_row(4, &roi, &specs).roi_id, 5);
    }

    // ---- aggregation primitives ----

    #[test]
    fn apply_agg_basic_functions() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(apply_agg(&v, AggFunc::Min), 2.0);
        assert_eq!(apply_agg(&v, AggFunc::Max), 9.0);
        assert_eq!(apply_agg(&v, AggFunc::Sum), 40.0);
        assert_eq!(apply_agg(&v, AggFunc::Avg), 5.0);
        assert_eq!(apply_agg(&v, AggFunc::Median), 4.5); // even count -> mean of middles
        // Sample stdev (n-1) of this set is sqrt(32/7) ≈ 2.1381
        assert!((apply_agg(&v, AggFunc::Stdev) - (32.0f64 / 7.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn apply_agg_edge_cases() {
        assert_eq!(apply_agg(&[], AggFunc::Avg), 0.0);
        assert_eq!(apply_agg(&[42.0], AggFunc::Median), 42.0);
        assert_eq!(apply_agg(&[42.0], AggFunc::Stdev), 0.0); // n<2 -> 0
    }

    // ---- group_key ----

    #[test]
    fn group_key_image_and_folder() {
        let roi = RoiRow {
            image_rel_path: "plate1/wellA/A1_01.tif".into(),
            ..make_roi("A1_01.tif", vec![], None, 0, 0.0, 0.0, "{}")
        };
        assert_eq!(group_key(&roi, GroupBy::Image, None).as_deref(), Some("A1_01.tif"));
        assert_eq!(
            group_key(&roi, GroupBy::Folder, None).as_deref(),
            Some("plate1/wellA")
        );
    }

    #[test]
    fn group_key_regex_capture_group_extracts_well() {
        let re = regex::Regex::new(r"^([A-Z]\d+)_").unwrap();
        let roi = make_roi("A1_02", vec![], None, 0, 0.0, 0.0, "{}");
        assert_eq!(
            group_key(&roi, GroupBy::Regex, Some(&re)).as_deref(),
            Some("A1")
        );
        // Non-matching name is dropped from the grouped view.
        let roi2 = make_roi("control", vec![], None, 0, 0.0, 0.0, "{}");
        assert_eq!(group_key(&roi2, GroupBy::Regex, Some(&re)), None);
    }

    // ---- aggregate_rows ----

    #[test]
    fn aggregate_rows_groups_by_regex_well_and_counts() {
        let rois = vec![
            make_roi("A1_01", vec![], None, 100, 0.0, 0.8, "{}"),
            make_roi("A1_02", vec![], None, 200, 0.0, 0.9, "{}"),
            make_roi("B1_01", vec![], None, 300, 0.0, 0.7, "{}"),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Regex,
            regex: r"^([A-Z]\d+)_".into(),
            aggs: vec![AggFunc::Avg],
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[]));

        // group, count, area_px, area_nm2, circularity
        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].id, "group");
        assert_eq!(specs[1].id, "count");
        assert_eq!(specs[2].id, "area_px__avg");

        assert_eq!(rows.len(), 2); // A1, B1 (sorted)
        assert_eq!(rows[0].values[0], "A1");
        assert_eq!(rows[0].values[1], "2"); // count
        assert_eq!(rows[0].values[2], "150.0"); // avg area_px (100,200)
        assert_eq!(rows[1].values[0], "B1");
        assert_eq!(rows[1].values[1], "1");
    }

    #[test]
    fn aggregate_rows_channel_columns_use_selected_func() {
        let json = |min: f64, max: f64| {
            format!(
                r#"{{"0":{{"min_scaled":{min},"max_scaled":{max},"mean_scaled":0.0}}}}"#
            )
        };
        let rois = vec![
            make_roi("img", vec![], None, 0, 0.0, 0.0, &json(10.0, 100.0)),
            make_roi("img", vec![], None, 0, 0.0, 0.0, &json(30.0, 200.0)),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Min],
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[0]));
        // group, count, area_px, area_nm2, circularity, ch0_min, ch0_max, ch0_avg
        assert_eq!(specs.len(), 8);
        assert_eq!(rows.len(), 1);
        // Min of the per-ROI min_scaled values (10, 30) -> 10
        assert_eq!(rows[0].values[5], "10.0");
        // Min of the per-ROI max_scaled values (100, 200) -> 100
        assert_eq!(rows[0].values[6], "100.0");
    }

    #[test]
    fn aggregate_rows_multiple_aggregates_yield_one_column_each() {
        let rois = vec![
            make_roi("img", vec![], None, 100, 0.0, 0.0, "{}"),
            make_roi("img", vec![], None, 300, 0.0, 0.0, "{}"),
        ];
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Min, AggFunc::Max, AggFunc::Avg],
        };
        let (specs, rows) = aggregate_rows(&rois, &config, &build_column_specs(&[]));
        // group, count, then 3 metrics (area_px, area_nm2, circularity) × 3 aggs
        assert_eq!(specs.len(), 2 + 3 * 3);
        assert_eq!(specs[2].id, "area_px__min");
        assert_eq!(specs[2].label, "Area (px²) [min]");
        assert_eq!(specs[3].id, "area_px__max");
        assert_eq!(specs[4].id, "area_px__avg");
        assert_eq!(rows[0].values[2], "100.0"); // min
        assert_eq!(rows[0].values[3], "300.0"); // max
        assert_eq!(rows[0].values[4], "200.0"); // avg
    }

    #[test]
    fn aggregate_rows_respects_column_visibility() {
        let rois = vec![make_roi("img", vec![], None, 100, 50.0, 0.9, "{}")];
        // Only the area_px metric is visible.
        let mut base_specs = build_column_specs(&[]);
        for spec in base_specs.iter_mut() {
            spec.visible = spec.id == "area_px";
        }
        let config = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
        };
        let (specs, _rows) = aggregate_rows(&rois, &config, &base_specs);
        // group, count, area_px__avg — area_nm2/circularity hidden.
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[2].id, "area_px__avg");
    }
}
