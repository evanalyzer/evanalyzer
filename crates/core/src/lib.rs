#![allow(dead_code)]

mod algos;
mod converters;
mod extlibs;
mod image;
mod init;
mod job;
mod pipeline;
mod resources;
mod roi;
mod storage;

// Init function which must be called once
pub use crate::init::CoreConfig;
pub use crate::init::init;

// System resource sizing (JVM heap, parallelism) based on available RAM
pub use crate::resources::recommended_jvm_heap_bytes;
pub use crate::resources::recommended_parallelism;

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

// Region of interest
pub use crate::roi::Roi;
// `RoiInit`/`PipelineCache` are re-exported so downstream crates can build
// `Roi`/`DuckDbExporter` fixtures for their own tests (e.g. `evanalyzer_app`'s
// exporter/aggregation integration tests), without duplicating the DuckDB
// schema in hand-written SQL.
pub use crate::pipeline::pipeline_cache::PipelineCache;
pub use crate::roi::Intensity;
pub use crate::roi::RoiInit;

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
pub use crate::storage::duckdb::RoiFilter;
pub use crate::storage::duckdb::RoiRow;
pub use crate::storage::duckdb::{AggregateSpec, AggregatedRow, GroupKeyMode};
pub use crate::storage::duckdb::{
    coloc_filter_label_any, coloc_filter_label_no, coloc_filter_label_with,
};
pub use crate::storage::file::CsvExporter;
pub use crate::storage::memory::MemoryExporter;
