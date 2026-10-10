#![allow(dead_code)]

#[cfg(feature = "ai")]
pub mod ai_learning;
mod algos;
mod converters;
mod extlibs;
mod image;
mod job;
mod object;
mod pipeline;
mod resources;
mod spatial_grid;
mod storage;

// System resource sizing (parallelism, reader pool size) based on available RAM
pub use crate::resources::SystemDiagnostics;
pub use crate::resources::cpu_ram_diagnostics;
pub use crate::resources::cuda_is_available;
pub use crate::resources::recommended_parallelism;
pub use crate::resources::recommended_reader_pool_size;
pub use crate::resources::system_diagnostics;
pub use crate::storage::duckdb::shared_connection as open_results_database;

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
pub use crate::image::SUPPORTED_IMAGE_FORMATS;
pub use evanalyzer_cfg::core_types::ImagePlane;
pub use evanalyzer_cfg::core_types::ImageTile;
pub use evanalyzer_cfg::core_types::ZProjection;

// Object
pub use crate::object::Intensity;
pub use crate::object::Object;
pub use crate::object::ObjectInit;
pub use crate::pipeline::pipeline_cache::GlobalPipelineCache;

// Job execution
pub use crate::job::algos_from_config::into_algorithm;
pub use crate::job::job_executor::JobExecutor;
pub use crate::job::job_executor::PreviewTileSettings;
pub use crate::job::job_executor::ProgressEvent;
pub use crate::job::job_generator::active_series;
pub use crate::job::job_generator::generate_analyze_job_from_project_settings;
pub use crate::job::job_generator::generate_preview_job_from_project_settings;
pub use crate::storage::PipelineResultExporter;
pub use crate::storage::duckdb::DuckDbExporter;
pub use crate::storage::file::CsvExporter;
pub use crate::storage::memory::MemoryExporter;

// AI classifier training
#[cfg(feature = "ai")]
pub use crate::ai_learning::model::SavedClassifier;
#[cfg(feature = "ai")]
pub use crate::ai_learning::model::{
    load_from_file as load_classifier_from_file, save_to_file as save_classifier_to_file,
};
#[cfg(feature = "ai")]
pub use crate::ai_learning::training::object::ObjectTrainingJob;
#[cfg(feature = "ai")]
pub use crate::ai_learning::training::pixel::PixelTrainingJob;
#[cfg(feature = "ai")]
pub use crate::ai_learning::training_job::TrainingImage;
