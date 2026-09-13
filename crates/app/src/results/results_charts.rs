use super::results_generator::{Column, PlaneFilter, ResultsGenerator};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

/// One histogram: `column`'s value distribution across every object
/// matching `plane`/`images`/`object_classes`, bucketed into `bins`
/// equal-width bins — same plane/images/class scoping every other results
/// view uses (see `ListFilter`/`PlateFilter` in results_generator.rs).
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
pub struct HistogramResult {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Copy)]
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
}

/// `points` may be a random sample of `total_object_count` rather than
/// every one of them — see `ScatterFilter::max_points`.
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

pub struct BoxplotResult {
    pub boxes: Vec<BoxplotBox>,
}

pub struct ResultCharts {}

impl ResultCharts {
    pub fn paint_boxplot(
        &self,
        database: &ResultsGenerator,
        filter: &BoxplotFilter,
    ) -> Result<BoxplotResult, InternalErrors> {
        database.get_boxplot(filter)
    }

    pub fn paint_histogram(
        &self,
        database: &ResultsGenerator,
        filter: &HistogramFilter,
    ) -> Result<HistogramResult, InternalErrors> {
        database.get_histogram(filter)
    }

    pub fn paint_scatter(
        &self,
        database: &ResultsGenerator,
        filter: &ScatterFilter,
    ) -> Result<ScatterResult, InternalErrors> {
        database.get_scatter(filter)
    }
}
