#![allow(dead_code)]

#[cfg(feature = "ai")]
pub mod ai_learning;
mod algos;
mod converters;
mod extlibs;
mod image;
mod init;
mod job;
mod object;
mod pipeline;
mod resources;
mod storage;

// Init function which must be called once
pub use crate::init::CoreConfig;
pub use crate::init::init;

// System resource sizing (JVM heap, parallelism) based on available RAM
pub use crate::resources::SystemDiagnostics;
pub use crate::resources::cpu_ram_diagnostics;
pub use crate::resources::cuda_is_available;
pub use crate::resources::recommended_jvm_heap_bytes;
pub use crate::resources::recommended_parallelism;
pub use crate::resources::recommended_reader_pool_size;
pub use crate::resources::system_diagnostics;

// Image reader
pub use crate::image::ChannelInfo;
pub use crate::image::F32Gray;
pub use crate::image::F32Rgb;
pub use crate::image::ImageChannel;
pub use crate::image::ImageContainer;
pub use crate::image::ImageInfo;
pub use crate::image::ImageMeta;
pub use crate::image::ImagePlane;
pub use crate::image::ImageReader;
pub use crate::image::ImageTile;
pub use crate::image::ImageTypeMarker;
pub use crate::image::ManagedImage;
pub use crate::image::PyramidInfo;
pub use crate::image::ReadMode;
pub use crate::image::SUPPORTED_IMAGE_FORMATS;
pub use crate::image::ZProjection;
pub use crate::image::init_java_wrapper;

// Object
pub use crate::object::Object;
// `ObjectInit`/`PipelineCache` are re-exported so downstream crates can build
// `Object`/`DuckDbExporter` fixtures for their own tests (e.g. `evanalyzer_app`'s
// exporter/aggregation integration tests), without duplicating the DuckDB
// schema in hand-written SQL.
pub use crate::object::Intensity;
pub use crate::object::ObjectInit;
pub use crate::pipeline::pipeline_cache::PipelineCache;

// Job execution
pub use crate::job::job_executor::BreakpointMode;
pub use crate::job::job_executor::BreakpointSettings;
pub use crate::job::job_executor::JobExecutor;
pub use crate::job::job_executor::PreviewTileSettings;
pub use crate::job::job_executor::ProgressEvent;
pub use crate::job::job_generator::generate_analyze_job_from_project_settings;
pub use crate::job::job_generator::generate_preview_job_from_project_settings;
pub use crate::storage::PipelineResultExporter;
pub use crate::storage::duckdb::DuckDbExporter;
pub use crate::storage::duckdb::DuckDbReader;
pub use crate::storage::duckdb::ClassRow;
pub use crate::storage::duckdb::ImageRow;
pub use crate::storage::duckdb::ObjectFilter;
pub use crate::storage::duckdb::ObjectRow;
pub use crate::storage::duckdb::{AggregateSpec, AggregatedRow, GroupKeyMode};
pub use crate::storage::duckdb::{
    coloc_filter_label_any, coloc_filter_label_no, coloc_filter_label_with,
};
pub use crate::storage::file::CsvExporter;
pub use crate::storage::memory::MemoryExporter;

// AI classifier training
#[cfg(feature = "ai")]
pub use crate::ai_learning::model::SavedClassifier;
#[cfg(feature = "ai")]
pub use crate::ai_learning::model::{load_from_file as load_classifier_from_file, save_to_file as save_classifier_to_file};
#[cfg(feature = "ai")]
pub use crate::ai_learning::training::object::ObjectTrainingJob;
#[cfg(feature = "ai")]
pub use crate::ai_learning::training::pixel::PixelTrainingJob;
#[cfg(feature = "ai")]
pub use crate::ai_learning::training_job::{TrainingImage, TrainingProgressEvent, TrainingStats};
