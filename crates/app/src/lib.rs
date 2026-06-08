pub mod extensions;
pub mod frontend;
mod project_owner;
mod results;
mod templates;

pub use frontend::Frontend;
pub use project_owner::{AppHandle, ProjectOwner, ProjectTmpSettings, ProjectWithRuntime};

pub mod prelude {
    pub use super::Frontend;
    pub use super::extensions::*;
}

pub mod result {
    pub use crate::results::results_exporter::ResultsExporter;
    pub use crate::results::results_loader::{
        AggFunc, ColumnSpec, DatabaseFilter, DisplayRow, GroupBy, GroupConfig, ResultsLoader,
        RoiRow, aggregate_rows, build_column_specs, discover_channels, to_display_row,
    };
}
