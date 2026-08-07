use crate::pipeline::pipeline_cache::PipelineCache;
use evanalyzer_cfg::core_types::InternalErrors;
pub mod duckdb;
pub mod file;
pub mod memory;

pub trait PipelineResultExporter: Send + Sync {
    fn export(&self, cache: &PipelineCache) -> Result<(), InternalErrors>;

    /// Called once per image, after every tile/plane `export()` call for that
    /// image has completed, so an image that produced zero objects is still
    /// recorded somewhere — otherwise it leaves no trace at all (an exporter
    /// backed by a per-object table has nothing to insert a row *into* when
    /// there are no objects). Default no-op: only exporters that keep a
    /// separate per-image summary (e.g. `DuckDbExporter`'s `images` table)
    /// need to override this.
    fn finalize_image(&self, _image_rel_path: &std::path::Path) -> Result<(), InternalErrors> {
        Ok(())
    }
}
