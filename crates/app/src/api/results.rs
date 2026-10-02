//! Results-database queries as front ends see them: filters, result tables,
//! chart data and export settings. Executed by a `ResultsSource`.

mod colors;
mod column;

pub(crate) use colors::value_to_color;
pub use colors::{COLOR_SCALE_GRADIENT_STOPS, color_scale_gradient};
pub use column::Column;

use evanalyzer_cfg::core_types::ObjectClass;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::range::Range;

#[derive(Clone, Serialize, Deserialize)]
pub enum View {
    List,
    Heatmap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlateDimensions {
    PLate2x3,
    Plate3x4,
    Plate4x6,
    Plate6x8,
    Plate8x12,
    Plate16x24,
    Plate32x48,
}

impl PlateDimensions {
    /// Returns the matrix dimensions as a `(rows, columns)` tuple.
    pub const fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::PLate2x3 => (2, 3),
            Self::Plate3x4 => (3, 4),
            Self::Plate4x6 => (4, 6),
            Self::Plate6x8 => (6, 8),
            Self::Plate8x12 => (8, 12),
            Self::Plate16x24 => (16, 24),
            Self::Plate32x48 => (32, 48),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WellSize {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregation {
    #[default]
    Avg,
    Min,
    Max,
    Stddev,
    Sum,
    Median,
    Skewness,
}

#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSchema {
    #[default]
    Excel,
    Viridis,
    Plasma,
    Inferno,
    Cividis,
    Coolwarm,
    RedBlue,
    YlGnBu,
    Haline,
    Algae,
    Thermal,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub enum ColorScale {
    #[default]
    Auto,
    Manual(f32, f32),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlaneFilter {
    pub z_stack: u32,
    pub t_stack: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: i32,
    /// Keyset cursor: `None` fetches the first page; `Some(id)` fetches the page starting right after that `object_id`.
    pub after: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlateFilter {
    pub plane: PlaneFilter,
    // Grouping regex, requires follwoing regex output (e.g. A1_01.vsi)
    // - Group1: the match of the group (e.g. A1)
    // - Group2: the match of the plate row (e.g. A)
    // - Group3: the match of the plate col (e.g. 1)
    // - Group4: the match of the image index (e.g. 01)
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub matrix_dimension: Option<PlateDimensions>,
}

// No `Pagination` field here (unlike `ListFilter`/`GroupedByImageFilter`):
// a plate-grouped result's row count is bounded by well count (at most
// `PlateDimensions::Plate32x48` = 1536), not by object count, so it stays
// tiny (well under a MB) even multiplied out over every `column` x
// `aggregation` x `object_class` combination requested at once — the RAM
// risk pagination guards against elsewhere (millions of raw objects, see
// `get_object_list`/`get_grouped_by_image`) doesn't apply to an
// already-aggregated-down-to-wells result.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlateFilterMulti {
    pub plane: PlaneFilter,
    // Grouping regex, requires follwoing regex output (e.g. A1_01.vsi)
    // - Group1: the match of the group (e.g. A1)
    // - Group2: the match of the plate row (e.g. A)
    // - Group3: the match of the plate col (e.g. 1)
    // - Group4: the match of the image index (e.g. 01)
    pub grouping_regex: String,
    pub aggregation: Vec<Aggregation>,
    pub object_class: Vec<ObjectClass>,
    pub column: Vec<Column>,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub matrix_dimension: Option<PlateDimensions>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WellFilter {
    pub plane: PlaneFilter,
    // Name of the group to display
    pub group_name: String,
    // Grouping regex, requires follwoing regex output (e.g. A1_01.vsi)
    // - Group1: the match of the group (e.g. A1)
    // - Group2: the match of the plate row (e.g. A)
    // - Group3: the match of the plate col (e.g. 1)
    // - Group4: the match of the image index (e.g. 01)
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    /// Grid dimensions to lay the well's fields out in. `None` assumes a
    /// 4x4 well (the common case for a plate imager's per-well field
    /// count) rather than fitting to whatever fields were actually found,
    /// so a well missing a field still shows a gap at that field's real
    /// position instead of the grid silently shrinking.
    pub well_size: Option<WellSize>,
    /// Maps grid position -> field index for non-trivial (e.g. snake)
    /// acquisition patterns: `well_order[position]` is the field `idx`
    /// (the regex's 4th capture group, e.g. the "01" in "A1_01.vsi") shown
    /// at that row-major grid position. `None` uses `idx` as the position
    /// directly (1-based: idx 1 -> position 0, top-left).
    pub well_order: Option<Vec<u32>>,
}

/// Same shape as `WellFilter` minus `group_name` — `get_wells_for_plate`
/// answers for every well at once, so there's no single well to name.
#[derive(Clone, Serialize, Deserialize)]
pub struct WellsBatchFilter {
    pub plane: PlaneFilter,
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub well_size: Option<WellSize>,
    pub well_order: Option<Vec<u32>>,
}

/// Same shape as `WellFilter` minus `group_name` — `get_wells_for_plate`
/// answers for every well at once, so there's no single well to name.
#[derive(Clone, Serialize, Deserialize)]
pub struct WellsBatchFilterMulti {
    pub plane: PlaneFilter,
    pub grouping_regex: String,
    pub aggregation: Vec<Aggregation>,
    pub object_class: Vec<ObjectClass>,
    pub column: Vec<Column>,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub well_size: Option<WellSize>,
    pub well_order: Option<Vec<u32>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ImageHeatmapFilter {
    pub plane: PlaneFilter,
    /// Image for which the heatmpa should be generated for
    pub image_rel_path: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    /// Size of a heatmap element (Default 256)
    pub square_size: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ListFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub with_coloc_details: bool,
    pub page: Pagination,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupedByImageFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub aggregation: Vec<Aggregation>,
    pub page: Pagination,
}

#[derive(Serialize, Deserialize)]
pub enum CellValue {
    Empty,
    String(String),
    Float(f32),
    Integer(i32),
    /// Object class with color
    Class((String, u32)),
}

#[derive(Serialize, Deserialize)]
pub struct Cell {
    pub value: CellValue,
    /// Cell background color
    pub bg_color: u32,
    /// If true this cell should be displayed in alternating color, the ui desides on itself what is alternating
    pub alternating_color: bool,
    /// Optional search (display name, key) (Group name of plate and image_rel_path in well view)
    pub search_key: Option<(String, String)>,
    /// True if this cell's value came from a disabled image
    pub disabled: bool,
    /// True if at least one image that contributed to is disabled
    pub any_disabled: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ColumnEntry {
    pub display_name: String,
    pub key: Column,
    pub group: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageEntry {
    pub name: String,
    pub rel_path: PathBuf,
    pub disabled: bool,
}

#[derive(Serialize, Deserialize)]
pub struct DatabaseResult {
    pub column_names: Vec<String>,
    pub row_names: Vec<String>,
    /// One row with its colums
    pub rows: Vec<Vec<Cell>>,
    pub min: f32,
    pub max: f32,
    /// How many source rows this page's query actually matched
    pub source_object_count: usize,
    /// For navigation
    pub row_locations: Vec<(String, [u32; 4])>,
}
/// One histogram: `column`'s value distribution across every object
/// matching `plane`/`images`/`object_classes`, bucketed into `bins`
/// equal-width bins — same plane/images/class scoping every other results
/// view uses (see `ListFilter`/`PlateFilter` in results_generator.rs).
#[derive(Clone, Serialize, Deserialize)]
pub struct HistogramFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database.
    pub object_classes: Option<Vec<ObjectClass>>,
    /// The measurement to bucket.
    pub column: Column,
    /// Number of equal-width bins to bucket `column`'s values into.
    pub bins: usize,
}

/// One scatter plot: every matched object's `(x_column, y_column)` pair,
/// one point per object.
#[derive(Clone, Serialize, Deserialize)]
pub struct ScatterFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database.
    pub object_classes: Option<Vec<ObjectClass>>,
    pub x_column: Column,
    pub y_column: Column,
    /// Caps how many points are actually plotted — a scatter of every
    /// object in a large database is neither readable nor cheap to render.
    /// `None` means no cap (every matched object).
    pub max_points: Option<usize>,
}

/// One boxplot: `column`'s distribution (min/Q1/median/Q3/max, plus
/// outliers) — one box per class in `object_classes` (or every registered
/// class if `None`), so distributions across classes can be compared side
/// by side in a single chart.
#[derive(Clone, Serialize, Deserialize)]
pub struct BoxplotFilter {
    pub plane: PlaneFilter,
    /// `None` means every image in the database.
    pub images: Option<Vec<String>>,
    /// `None` means every object class registered in the database — one
    /// box per class.
    pub object_classes: Option<Vec<ObjectClass>>,
    pub column: Column,
}

/// One equal-width-binned histogram, ready for the GUI to draw straight
/// from `bin_edges`/`counts` — `bin_edges` has `counts.len() + 1` entries
/// (edge `i` / edge `i+1` bound bin `i`).
#[derive(Serialize, Deserialize)]
pub struct HistogramResult {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
}

/// `points` may be a random sample of `total_object_count` rather than
/// every one of them — see `ScatterFilter::max_points`.
#[derive(Serialize, Deserialize)]
pub struct ScatterResult {
    pub points: Vec<ScatterPoint>,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub total_object_count: usize,
}

/// One class's box: min/Q1/median/Q3/max plus any Tukey outliers (values
/// beyond 1.5x the interquartile range from Q1/Q3).
#[derive(Serialize, Deserialize)]
pub struct BoxplotBox {
    pub label: String,
    pub color: u32,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
    pub object_count: usize,
}

#[derive(Serialize, Deserialize)]
pub struct BoxplotResult {
    pub boxes: Vec<BoxplotBox>,
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    #[default]
    XLSX,
    CSV,
    Parquet,
}

/// One export step/document completing — `current`/`total` describe
/// progress *within* the step named by `message` (e.g. "Plate/Well: ch2@spot"
/// at `current=3, total=22` for the 3rd of 22 classes), not overall export
/// progress across List/Plate/Well/Heatmap combined, since those phases
/// have no shared unit to make a single running percentage meaningful.
pub type ExportProgress<'a> = &'a mut dyn FnMut(&str, usize, usize);

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ResultExport {
    /// Directory the export writes its file(s) into — created if missing.
    /// Every document below lives directly under it (`list.xlsx`,
    /// `plate.xlsx`, `well.xlsx`, `heatmap_{image}.xlsx`).
    pub output_dir: PathBuf,
    pub format: ExportFormat,
    #[serde(with = "range_serde")]
    pub t_stacks: Range<u32>,
    #[serde(with = "range_serde")]
    pub z_stacks: Range<u32>,
    /// `[]` means every image in the database (see `resolve_images`).
    pub image_rel_paths: Vec<String>,

    // Variants
    pub columns: Vec<Column>,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    pub grouping_regex: String,
    /// `[]` means every object class registered in the database (see
    /// `export_plate_and_well`/`export_heatmap`).
    pub object_classes: Vec<ObjectClass>,

    // Matrix view options
    pub aggregations: Vec<Aggregation>,
    pub plate_dimension: Option<PlateDimensions>,
    pub well_size: Option<WellSize>,
    pub well_order: Option<Vec<u32>>,
    pub square_size: Option<usize>,

    // What to export
    pub with_list_view: bool,
    pub with_list_coloc_details: bool,
    /// `list_{image}.xlsx` per image instead of one shared `list.xlsx` —
    /// needed once a single image's own object count risks Excel's
    /// 1,048,576-row-per-sheet limit (a large plate scan can comfortably
    /// exceed that combined across images, even if no single image does).
    pub with_list_one_file_per_image: bool,
    /// `grouped_by_image.xlsx`: one row per image, one column per
    /// (selected column × selected aggregation) combination — see
    /// `export_grouped_by_image`. Independent of `with_list_view`: it's a
    /// separate document, not a variant of the per-object list.
    pub with_grouped_by_image_list: bool,
    pub with_plate_view_heatmap: bool,
    pub with_well_view_heatmap: bool,
    pub with_plate_view_list: bool,
    pub with_well_view_list: bool,
    pub with_heatmap: bool,
}

/// serde for `std::range::Range` (no upstream support yet): `(start, end)`.
mod range_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::range::Range;

    pub fn serialize<S: Serializer>(range: &Range<u32>, serializer: S) -> Result<S::Ok, S::Error> {
        (range.start, range.end).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Range<u32>, D::Error> {
        let (start, end) = <(u32, u32)>::deserialize(deserializer)?;
        Ok(Range { start, end })
    }
}
