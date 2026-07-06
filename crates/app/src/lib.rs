pub mod bioimageio;
pub mod extensions;
pub mod frontend;
mod project_owner;
mod results;
pub mod templates;

pub use frontend::Frontend;
pub use project_owner::{AppHandle, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime};

pub mod prelude {
    pub use super::Frontend;
    pub use super::extensions::*;
}

pub mod result {
    pub use crate::results::results_chart::{
        compute_heatmap, compute_histogram, compute_scatter, plottable_columns, render_heatmap,
        render_histogram, render_scatter, save_heatmap_png, save_histogram_png,
        save_rendered_chart_png, save_scatter_png, ChartHitTester, ColorBy, HeatmapCell,
        HeatmapColorScheme, HeatmapData, HeatmapMetric, HistogramBucket, HistogramData,
        HistogramSeries, RenderedChart, ScatterData, ScatterPoint,
    };
    pub use crate::results::results_exporter::ResultsExporter;
    pub use crate::results::results_loader::{
        AggFunc, ColumnSpec, DatabaseFilter, DisplayRow, GroupBy, GroupConfig, ResultsLoader,
        RoiRow, aggregate_rois_sql, aggregate_rows, build_coloc_detail_column_specs,
        build_column_specs, coloc_filter_label_any, coloc_filter_label_no,
        coloc_filter_label_with, coloc_partner_ids, discover_channels,
        discover_coloc_detail_columns, flatten_coloc_rows, sort_display_rows, to_display_row,
    };
    pub use crate::results::results_window::{
        EvictEdge, PageRowCounts, RowWindow, DEFAULT_WINDOW_PAGES,
    };
}
