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
        compute_histogram, compute_scatter, plottable_columns, render_histogram, render_scatter,
        save_histogram_png, save_scatter_png, ColorBy, HistogramBucket, HistogramData,
        RenderedChart, ScatterData, ScatterPoint,
    };
    pub use crate::results::results_exporter::ResultsExporter;
    pub use crate::results::results_loader::{
        AggFunc, ColumnSpec, DatabaseFilter, DisplayRow, GroupBy, GroupConfig, ResultsLoader,
        RoiRow, aggregate_rows, build_column_specs, coloc_filter_label_any, coloc_filter_label_no,
        coloc_filter_label_with, discover_channels, sort_display_rows, to_display_row,
    };
}
