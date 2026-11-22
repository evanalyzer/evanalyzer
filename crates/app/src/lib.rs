//! # evanalyzer_app
//!
//! One folder per feature.
//!
//! - Features that do the work, each with what travels (types), the trait
//!   a backend offers and the code that runs here (`local`): [`results`],
//!   [`images`], `files`, `jobs`, `training`.
//! - Features of the open project, which reach the work only through the
//!   `Backend` trait: [`project`], [`modules`], [`templates`],
//!   `annotations`, [`ai`], `user`.
//! - `backend`: the `Backend` trait and the local backend.
//! - `remote`: everything on the network - the remote backend, the worker
//!   and the wire between them. No feature knows it exists.
//! - [`charts`]: drawing only, no IO.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub mod ai;
mod annotations;
mod backend;
mod files;
pub mod images;
mod jobs;
pub mod modules;
pub mod project;
mod remote;
pub mod results;
pub mod templates;
mod training;
mod user;

pub mod charts;
pub mod number_format;

pub mod prelude {
    pub use crate::images::image_entry as image_entry_ext;
    pub use crate::images::image_list as images_ext;
    pub use crate::project::annotations as annotations_ext;
    pub use crate::project::classes as classification_ext;
    pub use crate::project::objects as object_ext;
    pub use crate::project::utils;
}

/// One stored object's solidity and Feret diameters, as the results
/// columns compute them - for the selected object's card.
pub mod object_shape {
    pub use evanalyzer_core::{StoredObjectShape, stored_object_shape};
}

/// A step's settings as the editor shows and changes them - the generated
/// ones, and a Script step's `param`s (declared in its script).
pub mod settings_of {
    pub use crate::modules::measurement_choices::{
        MeasurementChoice, measurement_choices, measurement_setting_text,
        module_measurement_choices, show_measurement_choices,
    };
    pub use evanalyzer_core::{
        HsvBounds, apply_command_param, command_parameters, declared_script_params,
        hsv_range_around, script_syntax_error,
    };
}

pub mod global {
    pub use crate::project::AppHandle;
    pub use crate::project::Frontend;
    pub use crate::results::SpeedUnit;
    pub use crate::user::crash_log;
    pub use crate::user::folder::USER_FOLDER_NAME;
    pub use crate::user::recent::RecentProject;
    pub use crate::user::recent::RecentServer;
    pub use crate::user::settings::AppSettings;
    pub use crate::user::settings::ColorScheme;
    pub use crate::user::settings::InterfaceSize;
    pub use crate::user::settings::NavigatorMode;
    pub use crate::user::settings::PanelSettings;
    pub use crate::user::settings::PerformanceSettings;
    pub use crate::user::settings::PredictionOverlayPrefs;
    pub use crate::user::settings::UserInformation;
    pub use crate::user::settings::ViewportMemorySettings;
    pub use crate::user::settings::{ResultsSplitSettings, TrailMode};
    pub use evanalyzer_core::memory::accounting::AccountingAllocator;
}

/// RAM: what a command takes (the step picker shows it), formatted.
pub mod memory {
    pub use evanalyzer_core::memory::analysis::MAX_TILE_SIZE;
    pub use evanalyzer_core::memory::command_ram::{CommandRam, command_ram};
    pub use evanalyzer_core::memory::format_binary_bytes;
}

pub mod fs {
    pub use crate::files::DirEntry;
    pub use crate::files::FileSystem;
    pub use crate::files::Place;
    pub use crate::files::PlaceKind;

    pub use crate::files::LocalFileSystem;
}

pub mod utils {
    pub use crate::project::utils::channel_display_name;
    pub use crate::project::utils::wavelength_to_rgb_float;
    pub use crate::project::utils::{assign_measurement_ids, assign_step_ids, unassigned};
}

pub mod exporter {
    pub use crate::project::cite::{cite_project, command_citations};
}

pub mod bioimageio {
    pub use crate::ai::bioimageio::configure_from;
    pub use crate::ai::bioimageio::parse_file;
}

pub mod ai_learning {
    pub use crate::ai::learning::load_classifier_settings;
    pub use crate::ai::learning::model_is_used;
    pub use crate::ai::learning::model_output_path;
    pub use crate::ai::learning::object_class_labels_from_project;
    pub use crate::ai::learning::pipeline_for_model;
    pub use crate::ai::learning::pixel_class_labels_from_project;
    pub use crate::ai::learning::prediction_target;
    pub use crate::ai::learning::save_trained_model;
    pub use crate::ai::learning::used_object_classes;
    pub use crate::jobs::CancelHandle;
    pub use crate::training::ModelDestination;
    pub use crate::training::PixelTrainingParams;
    pub use crate::training::RunningTraining;
    pub use crate::training::StartTrainingError;
    pub use crate::training::TrainingItems;
    pub use crate::training::TrainingRequest;
}

pub mod backends {
    pub use crate::backend::Backend;
    pub use crate::backend::ConnectionSecurity;
    pub use crate::backend::SystemInfo;
    pub mod local {
        pub use crate::backend::LocalBackend;
        pub use crate::backend::local::system::{
            ResourceLimits, cpu_ram_diagnostics, set_resource_limits,
        };
    }
    pub mod remote {
        pub use crate::remote::RemoteBackend;
        pub use crate::remote::ServerCertificate;
        pub use crate::remote::TlsTrust;
        pub use crate::remote::Worker;
        pub use crate::remote::generate_token;
        pub use crate::remote::server_certificate;
    }
}

pub mod analysis {
    pub use crate::jobs::AnalysisRequest;
    pub use crate::jobs::CancelHandle;
    pub use crate::jobs::JobInfo;
    pub use crate::jobs::JobState;
    pub use crate::jobs::ProgressEvent;
    pub use crate::jobs::RunningJob;
    pub use evanalyzer_core::check_pipeline_channels;
    pub use evanalyzer_core::random_job_name;
    pub use evanalyzer_core::{
        ObjectPredictionCapture, PredictedClass, PredictedObjectClass, PredictionCapture,
    };
}

pub mod preview {
    pub use crate::jobs::PreviewRequest;
    pub use crate::jobs::PreviewViewport;
    pub use crate::jobs::StartPreviewError;
}
