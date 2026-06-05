use crate::pipeline::pipeline_cache::PipelineCache;
use evanalyzer_cfg::core_types::InternalErrors;
pub mod duckdb;
pub mod file;
pub mod memory;

pub trait PipelineResultExporter: Send + Sync {
    fn export(&self, cache: &PipelineCache) -> Result<(), InternalErrors>;
}
