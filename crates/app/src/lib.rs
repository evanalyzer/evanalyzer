pub mod ai_learning;
pub mod bioimageio;
pub mod crash_log;
pub mod extensions;
pub mod frontend;
mod project_owner;
pub mod publisher;
mod results;
pub mod settings;
pub mod templates;

pub use frontend::Frontend;
pub use project_owner::{
    AppHandle, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime, ReaderPool,
};

pub mod prelude {
    pub use super::Frontend;
    pub use super::extensions::*;
}

pub mod result {
    pub use crate::results::plate_matrix::{
        PlateCell, PlateMatrixResult, RegexSuggestion, WellCell, WellMatrixResult,
        compute_plate_matrix, compute_well_matrix, resolve_range, row_label, suggest_regex,
    };
    pub use crate::results::results_chart::{
        ChartHitTester, ColorBy, HeatmapCell, HeatmapColorScheme, HeatmapData, HeatmapMetric,
        HeatmapRange, HistogramBucket, HistogramData, HistogramSeries, RenderedChart, ScatterData,
        ScatterPoint, compute_heatmap, compute_histogram, compute_scatter, plottable_columns,
        render_heatmap, render_histogram, render_scatter, save_heatmap_png, save_histogram_png,
        save_rendered_chart_png, save_scatter_png,
    };
    pub use crate::results::results_exporter::ResultsExporter;
    pub use crate::results::results_loader::{
        AggFunc, ClassRow, ColumnSpec, DatabaseFilter, DisplayRow, GroupBy, GroupConfig, ImageRow,
        ObjectRow, ResultsLoader, aggregate_objects_sql, aggregate_rows,
        build_coloc_detail_column_specs, build_column_specs, coloc_filter_label_any,
        coloc_filter_label_no, coloc_filter_label_with, coloc_partner_ids, discover_channels,
        discover_coloc_detail_columns, flatten_coloc_rows, sort_display_rows, to_display_row,
    };
    pub use crate::results::results_window::{
        DEFAULT_WINDOW_PAGES, EvictEdge, PageRowCounts, RowWindow,
    };
}
