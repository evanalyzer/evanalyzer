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

/// Where AI inference runs; set once at startup, see
/// [`configure_ai_devices`](ai::configure_ai_devices).
pub mod ai {
    pub use evanalyzer_core::ai_devices::AiDeviceOptions;
    pub use evanalyzer_core::ai_devices::AiDeviceSelection;
    pub use evanalyzer_core::ai_devices::DEFAULT_GPU_SLOTS;
    pub use evanalyzer_core::ai_devices::configure_ai_devices;
    pub use evanalyzer_core::ai_devices::parse_gpu_slots;
}

/// A step's settings as the editor shows and changes them - the generated
/// ones, and a Script step's `param`s (declared in its script).
pub mod settings_of {
    pub use evanalyzer_core::{
        apply_command_param, command_parameters, declared_script_params, script_syntax_error,
    };
}

/// Components in the editor: component cards, creating and expanding components, the library,
/// checks.
pub mod components {
    pub use crate::workspace::component_checks::{
        ComponentProblem, ProblemFix, Severity, apply_fix, check_component, check_components,
        check_project, problems_at, step_problem,
    };
    pub use crate::workspace::component_editing::{
        ClassRole, ComponentPlan, ComponentPlanClass, StepList, add_binding, apply_defaults,
        bindable_settings, bound_setting_edited, clone_component_for_step, compatible_parameters,
        component_uses, create_component, edit_parameter, expand_component, exposed_settings,
        list_entry_removed, move_parameter, new_parameter_key, parameter_for_setting,
        plan_component, prune_component_values, remove_binding, remove_component,
        remove_component_step, remove_parameter, remove_unused_components, set_parameter_section,
        setting_at, step_setting, step_settings, toggle_exposed, unused_components,
        would_contain_itself,
    };
    pub use crate::workspace::component_insights::{
        ImageChannels, image_channels, parameter_key, parameter_wiring, rekey_parameters,
        setting_wiring,
    };
    pub use crate::workspace::component_library::{
        components_used_by, content_hash, insert_component_template, library_name,
        merge_component_definitions, modified_since_inserted, project_copy,
        save_component_template, save_pipeline_as_component, update_available, update_from_library,
    };
    pub use evanalyzer_core::component_parameters;
    pub use evanalyzer_core::component_steps_with_values;
    pub use evanalyzer_core::set_component_parameter;
}

pub mod global {
    pub use crate::workspace::AppHandle;
    pub use crate::workspace::Frontend;
    pub use crate::workspace::crash_log;
    pub use crate::workspace::settings::AppSettings;
    pub use crate::workspace::settings::ColorScheme;
    pub use crate::workspace::settings::RecentServer;
    pub use crate::workspace::settings::RecentProject;
    pub use crate::workspace::settings::UserInformation;
    pub use crate::workspace::settings::ViewportMemorySettings;
    pub use crate::workspace::settings::ResultsSplitSettings;
    pub use crate::workspace::settings::PanelSettings;
    pub use crate::workspace::settings::NavigatorMode;
}

pub mod results {
    pub use crate::api::Aggregation;
    pub use crate::api::BoxplotBox;
    pub use crate::api::BoxplotFilter;
    pub use crate::api::BoxplotGroup;
    pub use crate::api::DrawnImages;
    pub use crate::backend::local::results::{colorize, colorize_thumbnail};
    pub use crate::api::{ImageObjectsFilter, ResultObject};
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
    pub use crate::api::Grouping;
    pub use crate::api::HistogramFilter;
    pub use crate::api::HistogramResult;
    pub use crate::api::ImageEntry;
    pub use crate::api::ImageHeatmapFilter;
    pub use crate::api::ListFilter;
    pub use crate::api::Pagination;
    pub use crate::api::PlaneFilter;
    pub use crate::api::PlateFilter;
    pub use crate::api::PlateFilterMulti;
    pub use crate::api::PlateSize;
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
    pub use crate::workspace::extensions::utils::channel_display_name;
    pub use crate::workspace::extensions::utils::wavelength_to_rgb_float;
    pub use crate::workspace::extensions::utils::{assign_step_ids, unassigned};
}

pub mod project {
    pub use crate::workspace::ProjectOwner;
    pub use crate::workspace::ProjectWithRuntime;
    pub use crate::workspace::annotation_measure::measure_on_source;
    pub use crate::workspace::annotations_geojson::{
        ExportReport, ImportReport, PendingImport, apply_import, export_geojson, import_geojson,
        measure_pending, read_geojson,
    };
    pub use crate::workspace::class_ids::{
        ClassIdChange, change_class_id, free_class_ids, next_class_id,
    };
    pub use crate::workspace::extensions::annotations_ext::{AnnotationScope, AnnotationsExt};
    pub use crate::workspace::extensions::image_entry_ext::ImageEntryExt;
    pub use crate::workspace::extensions::object_ext::ObjectExt;
    pub use crate::workspace::extensions::project_ext::ConfigValueSource;
    pub use crate::workspace::extensions::project_ext::ProjectExt;
    pub use crate::workspace::extensions::project_ext::SaveProjectActions;
    pub use crate::workspace::extensions::project_ext::SelectNewProjectRootAction;
    pub use crate::workspace::extensions::project_ext::collect_images_at_root;
    pub use crate::workspace::extensions::project_ext::load_project;
}

pub mod exporter {
    pub use crate::workspace::export::cite_project::{cite_project, command_citations};
}

pub mod bioimageio {
    pub use crate::workspace::bioimageio::configure_from;
    pub use crate::workspace::bioimageio::parse_file;
}

pub mod templates {
    pub use crate::api::TemplateFolders;
    pub use crate::workspace::templates::load_component_templates;
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
        pub use crate::backend::local::system::cpu_ram_diagnostics;
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
    pub use crate::workspace::images::{
        Erased, erase_mask, annotation_from_mask, stroke_mask, union_mask,
    };
    pub use crate::workspace::plate_layout::{PlateLayout, WellImages, plate_layout, texts_in_group};
}
