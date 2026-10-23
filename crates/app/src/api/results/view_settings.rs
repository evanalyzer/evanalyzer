//! A results view's settings (`ProjectSettings::results_views[].settings`):
//! what the GUI's List, Plate and Charts show - classes by id, columns by
//! key (`Column::as_key`). Kept out of the project schema: the project
//! stores them as plain JSON, so a change here never needs a project
//! migration. Read leniently (`ViewSettings::read`): a part that can't be
//! read takes its defaults. The GUI shows views with them, the CLI exports
//! one (`export --view`).

use crate::api::{Aggregation, BoxplotGroup, ColorSchema, Column, ColumnEntry, TimeCourseLines};
use evanalyzer_cfg::settings::classification_settings::Class;
use serde::{Deserialize, Serialize};

/// The List's columns when a view names none.
pub const DEFAULT_LIST_COLUMNS: [Column; 6] = [
    Column::ImageName,
    Column::ObjectId,
    Column::ObjectClass,
    Column::AreaSizePx,
    Column::AreaSizeNm,
    Column::Circularity,
];

/// The columns `keys` name among `available` (the results' columns), and
/// the keys these results lack.
pub fn resolve_columns(
    keys: &[String],
    classes: &[Class],
    available: &[ColumnEntry],
) -> (Vec<Column>, Vec<String>) {
    let mut columns = Vec::new();
    let mut missing = Vec::new();
    for key in keys {
        let found = Column::from_key(key, classes).or_else(|| {
            Column::measurement_key(key)?;
            available
                .iter()
                .find(|entry| entry.key.as_key(classes) == *key)
                .map(|entry| entry.key.clone())
        });
        match found {
            Some(column) => columns.push(column),
            None => missing.push(key.clone()),
        }
    }
    (columns, missing)
}

/// Histogram bins, as the Charts start.
pub const DEFAULT_HISTOGRAM_BINS: usize = 24;

/// A view's settings, as stored in the project (`ResultsView::settings`).
/// Missing fields (an older project, a view never shown) take the defaults.
#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct ViewSettings {
    pub list: ListView,
    pub plate: PlateView,
    pub chart: ChartView,
}

impl ViewSettings {
    /// A view's stored settings, each section on its own: one the GUI can't
    /// read any more (a setting changed its type) takes its defaults, the
    /// others stay - a view never breaks the project, nor a part of a view
    /// the rest of it.
    pub fn read(value: &serde_json::Value) -> Self {
        fn section<T: serde::de::DeserializeOwned + Default>(
            value: &serde_json::Value,
            name: &str,
        ) -> T {
            value
                .get(name)
                .and_then(|section| serde_json::from_value(section.clone()).ok())
                .unwrap_or_default()
        }
        Self {
            list: section(value, "list"),
            plate: section(value, "plate"),
            chart: section(value, "chart"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ListView {
    /// Class ids shown; `None`: every class (also ones new in an analysis).
    pub classes: Option<Vec<u32>>,
    /// Column keys; `None`: the default columns.
    pub columns: Option<Vec<String>>,
    pub aggregations: Vec<Aggregation>,
    pub rows_per_image: bool,
    pub transposed: bool,
    pub coloc_details: bool,
    /// "all", "enabled" or "disabled".
    pub images: String,
}

impl Default for ListView {
    fn default() -> Self {
        Self {
            classes: None,
            columns: None,
            aggregations: vec![Aggregation::Avg],
            rows_per_image: false,
            transposed: false,
            coloc_details: false,
            images: "all".into(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct PlateView {
    pub class: Option<u32>,
    /// `None`: the first column a plate can show.
    pub column: Option<String>,
    pub aggregation: Aggregation,
    pub color_schema: ColorSchema,
    /// A fixed color scale (min, max); `None`: Auto.
    pub color_scale: Option<(f32, f32)>,
    pub square_size: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ChartView {
    /// "histogram", "scatter", "boxplot" or "overTime".
    pub kind: String,
    pub column: Option<String>,
    pub y_column: Option<String>,
    pub class: Option<u32>,
    pub group_by: BoxplotGroup,
    pub bins: usize,
    pub log_counts: bool,
    /// The scatter's points picker (0..3, see `scatter_points_limit`).
    pub points: i32,
    pub time_lines: TimeCourseLines,
    pub time_median: bool,
    pub time_relative: bool,
    pub time_tracks: bool,
}

impl Default for ChartView {
    fn default() -> Self {
        Self {
            kind: "histogram".into(),
            column: None,
            y_column: None,
            class: None,
            group_by: BoxplotGroup::Class,
            bins: DEFAULT_HISTOGRAM_BINS,
            log_counts: false,
            points: 1,
            time_lines: TimeCourseLines::Class,
            time_median: false,
            time_relative: false,
            time_tracks: false,
        }
    }
}
