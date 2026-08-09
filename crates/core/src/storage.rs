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
    ///
    /// `error`: `None` if every tile/plane for this image exported
    /// successfully, `Some(message)` otherwise. Implementations that record
    /// per-image status (again, `DuckDbExporter`) must persist this rather
    /// than recording every image as successful regardless of `error` -
    /// otherwise a partially-failed image is indistinguishable from a
    /// genuinely complete one once it's in storage.
    fn finalize_image(
        &self,
        _image_rel_path: &std::path::Path,
        _error: Option<&str>,
    ) -> Result<(), InternalErrors> {
        Ok(())
    }
}
