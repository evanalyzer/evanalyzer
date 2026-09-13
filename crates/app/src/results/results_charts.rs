use super::results_generator::{Column, PlaneFilter};
use evanalyzer_cfg::core_types::ObjectClass;

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

pub struct ResultCharts {}

impl ResultCharts {
    pub fn paint_boxplot(&self, filter: &BoxplotFilter) {
        let _ = filter;
    }
    pub fn paint_histogram(&self, filter: &HistogramFilter) {
        let _ = filter;
    }
    pub fn paint_scatter(&self, filter: &ScatterFilter) {
        let _ = filter;
    }
}
