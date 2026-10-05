//! - [`api`]: the contract between front ends and backends.
//! - [`backend`]: where the work runs - [`backend::local`] (this process,
//!   the only module calling `evanalyzer_core`) or [`backend::remote`].
//! - [`workspace`]: the UI side - the open project, editing, undo.

mod api;
mod backend;
mod workspace;

pub mod prelude {
    pub use super::workspace::extensions::*;
}

pub mod global {
    pub use crate::workspace::AppHandle;
    pub use crate::workspace::Frontend;
    pub use crate::workspace::crash_log;
    pub use crate::workspace::settings::AppSettings;
    pub use crate::workspace::settings::RecentServer;
}

pub mod results {
    pub use crate::api::Aggregation;
    pub use crate::api::BoxplotBox;
    pub use crate::api::BoxplotFilter;
    pub use crate::api::COLOR_SCALE_GRADIENT_STOPS;
    pub use crate::api::Cell;
    pub use crate::api::CellValue;
    pub use crate::api::ColorScale;
    pub use crate::api::ColorSchema;
    pub use crate::api::Column;
    pub use crate::api::ColumnEntry;
    pub use crate::api::DatabaseResult;
    pub use crate::api::ExportFormat;
    pub use crate::api::GroupedByImageFilter;
    pub use crate::api::HistogramFilter;
    pub use crate::api::HistogramResult;
    pub use crate::api::ImageEntry;
    pub use crate::api::ImageHeatmapFilter;
    pub use crate::api::ListFilter;
    pub use crate::api::Pagination;
    pub use crate::api::PlaneFilter;
    pub use crate::api::PlateDimensions;
    pub use crate::api::PlateFilter;
    pub use crate::api::PlateFilterMulti;
    pub use crate::api::ResultExport;
    pub use crate::api::ResultsSource;
    pub use crate::api::RunStatus;
    pub use crate::api::ScatterFilter;
    pub use crate::api::ScatterResult;
    pub use crate::api::View;
    pub use crate::api::WellFilter;
    pub use crate::api::WellSize;
    pub use crate::api::WellsBatchFilter;
    pub use crate::api::WellsBatchFilterMulti;
    pub use crate::api::color_scale_gradient;

    pub use crate::backend::local::ResultsGenerator as LocalResultsGenerator;
}

pub mod fs {
    pub use crate::api::DirEntry;
    pub use crate::api::FileSystem;
    pub use crate::api::PlaceKind;

    pub use crate::backend::LocalFileSystem;
}

pub mod utils {
    pub use crate::workspace::extensions::utils::wavelength_to_rgb_float;
}

pub mod project {
    pub use crate::workspace::ProjectOwner;
    pub use crate::workspace::ProjectWithRuntime;
    pub use crate::workspace::extensions::image_entry_ext::ImageEntryExt;
    pub use crate::workspace::extensions::object_ext::ObjectExt;
    pub use crate::workspace::extensions::project_ext::ProjectExt;
    pub use crate::workspace::extensions::project_ext::SaveProjectActions;
    pub use crate::workspace::extensions::project_ext::SelectNewProjectRootAction;
    pub use crate::workspace::extensions::project_ext::collect_images_at_root;
    pub use crate::workspace::extensions::project_ext::load_project;
}

pub mod exporter {
    pub use crate::workspace::export::cite_project::cite_project;
}

pub mod bioimageio {
    pub use crate::workspace::bioimageio::configure_from;
    pub use crate::workspace::bioimageio::parse_file;
}

pub mod templates {
    pub use crate::api::TemplateFolders;
    pub use crate::workspace::templates::load_pipeline_templates;
    pub use crate::workspace::templates::load_project_template_from_file;
    pub use crate::workspace::templates::load_project_templates;
}

pub mod ai_learning {
    pub use crate::api::CancelHandle;
    pub use crate::api::ModelDestination;
    pub use crate::api::PixelTrainingParams;
    pub use crate::api::RunningTraining;
    pub use crate::api::StartTrainingError;
    pub use crate::api::TrainingItems;
    pub use crate::api::TrainingRequest;
    pub use crate::workspace::ai_learning::load_classifier_settings;
    pub use crate::workspace::ai_learning::object_class_labels_from_project;
    pub use crate::workspace::ai_learning::pixel_class_labels_from_project;
    pub use crate::workspace::ai_learning::save_trained_model;
    pub use crate::workspace::ai_learning::used_object_classes;
}

pub mod backends {
    pub use crate::api::Backend;
    pub use crate::api::ConnectionSecurity;
    pub use crate::api::SystemInfo;
    pub mod local {
        pub use crate::backend::LocalBackend;
    }
    pub mod remote {
        pub use crate::backend::RemoteBackend;
        pub use crate::backend::ServerCertificate;
        pub use crate::backend::TlsTrust;
        pub use crate::backend::Worker;
        pub use crate::backend::generate_token;
        pub use crate::backend::server_certificate;
    }
}

pub mod analysis {
    pub use crate::api::AnalysisRequest;
    pub use crate::api::CancelHandle;
    pub use crate::api::JobInfo;
    pub use crate::api::JobState;
    pub use crate::api::ProgressEvent;
    pub use crate::api::RunningJob;
}

pub mod preview {
    pub use crate::api::MAX_PREVIEW_VISIBLE_TILES;
    pub use crate::api::PreviewRequest;
    pub use crate::api::PreviewViewport;
    pub use crate::api::StartPreviewError;
}

pub mod images {
    pub use crate::api::ImageChannel;
    pub use crate::api::ImageContainer;
    pub use crate::api::ImageMeta;
    pub use crate::api::ImageSource;
    pub use crate::api::ManagedImage;
    pub use crate::api::Point2d;
    pub use crate::api::PyramidInfo;
    pub use crate::api::TileRequest;
    pub use crate::workspace::images::object_from_mask;
}
