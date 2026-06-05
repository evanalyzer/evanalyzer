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
        ColumnSpec { id: "roi_id".into(),     label: "ROI ID".into(),       filterable: false, visible: true },
        ColumnSpec { id: "image".into(),      label: "Image".into(),        filterable: true,  visible: true },
        ColumnSpec { id: "class".into(),      label: "Class".into(),        filterable: true,  visible: true },
        ColumnSpec { id: "area_px".into(),    label: "Area (px\u{00B2})".into(), filterable: false, visible: true },
        ColumnSpec { id: "area_nm2".into(),   label: "Area (nm\u{00B2})".into(), filterable: false, visible: true },
        ColumnSpec { id: "circularity".into(),label: "Circularity".into(),  filterable: false, visible: true },
    ];
    for &ch in channels {
        cols.push(ColumnSpec { id: format!("ch{ch}_min_bit"), label: format!("Ch{ch} Min (bit)"), filterable: false, visible: true });
        cols.push(ColumnSpec { id: format!("ch{ch}_max_bit"), label: format!("Ch{ch} Max (bit)"), filterable: false, visible: true });
        cols.push(ColumnSpec { id: format!("ch{ch}_avg_bit"), label: format!("Ch{ch} Avg (bit)"), filterable: false, visible: true });
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
    let needs_intensities = columns
        .iter()
        .any(|c| c.visible && c.id.starts_with("ch"));

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
                "roi_id"      => roi.object_id.to_string(),
                "image"       => roi.image_name.clone(),
                "class"       => class.clone(),
                "area_px"     => roi.area_px.to_string(),
                "area_nm2"    => format!("{:.2}", roi.area_nm2),
                "circularity" => format!("{:.3}", roi.circularity),
                id            => channel_value(id, &intensities),
            }
        })
        .collect();

    DisplayRow { roi_id: (row_idx + 1) as i32, values }
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
        let Ok(ch) = ch_str.parse::<i32>() else { continue };
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
        let roi = make_roi("img.tif", vec![], None, 100, 100.0, 0.9,
            r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#,
        );
        assert_eq!(discover_channels(&[roi]), vec![0]);
    }

    #[test]
    fn discover_channels_multiple_images_merged_and_sorted() {
        let roi1 = make_roi("a.tif", vec![], None, 100, 100.0, 0.9,
            r#"{"2":{"sum_raw":1.0,"sum_scaled":1.0,"mean_raw":1.0,"mean_scaled":1.0,"median_raw":1.0,"median_scaled":1.0,"std_raw":0.0,"std_scaled":0.0,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":1.0}}"#,
        );
        let roi2 = make_roi("b.tif", vec![], None, 100, 100.0, 0.9,
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
        assert_eq!(ch_ids, ["ch0_min_bit", "ch0_max_bit", "ch0_avg_bit",
                             "ch1_min_bit", "ch1_max_bit", "ch1_avg_bit"]);
    }

    // ---- to_display_row ----

    #[test]
    fn to_display_row_basic_fixed_columns() {
        let roi = make_roi("sample.tif", vec!["Nucleus".into()], None, 1234, 567.89, 0.923, "{}");
        let specs = build_column_specs(&[]);
        let row = to_display_row(0, &roi, &specs);

        assert_eq!(row.roi_id, 1); // row_idx=0 → roi_id=1
        assert_eq!(row.values[0], "00000000-0000-0000-0000-000000000001"); // roi_id col
        assert_eq!(row.values[1], "sample.tif");  // image
        assert_eq!(row.values[2], "Nucleus");     // class from object_class_name
        assert_eq!(row.values[3], "1234");        // area_px
        assert_eq!(row.values[4], "567.89");      // area_nm2
        assert_eq!(row.values[5], "0.923");       // circularity
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
        assert_eq!(row.values[6], "100");  // min_scaled rounded
        assert_eq!(row.values[7], "200");  // max_scaled rounded
    }

    #[test]
    fn to_display_row_row_idx_increments_roi_id() {
        let roi = make_roi("img.tif", vec![], None, 0, 0.0, 0.0, "{}");
        let specs = build_column_specs(&[]);
        assert_eq!(to_display_row(0, &roi, &specs).roi_id, 1);
        assert_eq!(to_display_row(4, &roi, &specs).roi_id, 5);
    }
}
