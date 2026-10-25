//! # evanalyzer_core
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

#![allow(dead_code)]

pub mod ai_devices;
pub mod ai_learning;
mod algos;
mod dataset;
mod image;
mod job;
mod object;
mod pipeline;
mod storage;
mod utils;

// System resource sizing (parallelism, reader pool size) based on available RAM
pub use crate::job::job_generator::random_job_name;
pub use crate::storage::duckdb::shared_connection as open_results_database;
pub use crate::storage::duckdb::{RESULTS_SCHEMA_VERSION, reject_newer_results};
pub use crate::storage::mask_codec::{decode_mask, encode_mask};
pub use crate::utils::resources::SystemDiagnostics;
pub use crate::utils::resources::cpu_ram_diagnostics;
pub use crate::utils::resources::recommended_parallelism;
pub use crate::utils::resources::system_diagnostics;
pub use evanalyzer_ai::cuda_is_available;

// Image reader
pub use crate::image::ChannelInfo;
pub use crate::image::F32Gray;
pub use crate::image::F32Rgb;
pub use crate::image::ImageChannel;
pub use crate::image::ImageContainer;
pub use crate::image::ImageInfo;
pub use crate::image::ImageMeta;
pub use crate::image::ImageReader;
pub use crate::image::ImageTypeMarker;
pub use crate::image::ManagedImage;
pub use crate::image::Point2d;
pub use crate::image::PyramidInfo;
pub use crate::image::ReadMode;
pub use crate::image::supported_image_formats;
pub use evanalyzer_cfg::core_types::ImageDimensionality;
pub use evanalyzer_cfg::core_types::ImagePlane;
pub use evanalyzer_cfg::core_types::ImageTile;
pub use evanalyzer_cfg::core_types::ZProjection;

// Object
pub use crate::ai_learning::utils::{bbox_overlaps_tile, tile_grid};
pub use crate::object::Intensity;
pub use crate::object::Object;
pub use crate::object::ObjectInit;
pub use crate::object::Track;
pub use crate::object::intensities::{ChannelTile, IntensityAccumulator, MaskIntensities};
pub use crate::object::regions::{measure_on_planes, ring_of};
pub use crate::pipeline::pipeline_cache::GlobalPipelineCache;
pub use crate::pipeline::pipeline_cache::{channel_pixel_slice, sample_channel_pixel};

// Job execution
pub use crate::algos::{
    ParamDecl, ParamKind, ScriptValue, apply_command_param, command_parameters,
    declared_script_params, script_syntax_error,
};
pub use crate::job::algos_from_config::{into_algorithm, into_pipeline_step};

/// What `command` measures (its `ImageAlgorithm::measurement_outputs`) -
/// none for a step that doesn't measure. The settings become the step for
/// it: a plain copy, nothing is loaded.
pub fn measurement_outputs(
    command: &evanalyzer_cfg::settings::pipeline_command::PipelineCommand,
) -> Vec<evanalyzer_cfg::core_types::MeasurementInfo> {
    use evanalyzer_cfg::settings::pipeline_command::CommandCategory;
    if *command.category() != CommandCategory::Measure {
        return Vec::new();
    }
    into_algorithm(command.clone())
        .map(|step| step.measurement_outputs())
        .unwrap_or_default()
}
pub use crate::job::job_executor::JobExecutor;
pub use crate::job::job_executor::PreviewTileSettings;
pub use crate::job::job_executor::ProgressEvent;
pub use crate::job::job_generator::active_series;
pub use crate::job::job_generator::generate_analyze_job_from_project_settings;
pub use crate::job::job_generator::generate_dataset_job_from_project_settings;
pub use crate::job::job_generator::pipeline_channels;
pub use crate::job::job_generator::pipeline_classes;
pub use crate::job::job_generator::{PreviewOutput, generate_preview_job_from_project_settings};
pub use crate::job::modules::{
    ExpandedCommand, ScopeCounter, UseEnd, UseId, expand_step, module_parameters,
    module_steps_with_values, own_numbers, renumbered, set_module_parameter, value_from_text,
};
pub use crate::job::pipeline_step::PipelineStep;
pub use crate::storage::duckdb::DuckDbExporter;
pub use crate::storage::file::CsvExporter;
pub use crate::storage::memory::MemoryExporter;
pub use crate::storage::{ImageSummary, PipelineResultExporter};

// Dataset
pub use crate::dataset::DatasetAlgorithm;
pub use crate::dataset::DatasetContext;
pub use crate::dataset::tracking::simple_tracker::SimpleTracker;

// AI classifier training
pub use crate::ai_learning::model::SavedClassifier;
pub use crate::ai_learning::model::{
    load_from_file as load_classifier_from_file, save_to_file as save_classifier_to_file,
};
pub use crate::ai_learning::training::object::ObjectTrainingJob;
pub use crate::ai_learning::training::pixel::PixelTrainingJob;
pub use crate::ai_learning::training_job::TrainingImage;
