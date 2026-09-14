use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use evanalyzer_cfg::core_types::InternalErrors;
pub mod duckdb;
pub mod file;
pub mod memory;

pub trait PipelineResultExporter: Send + Sync {
    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors>;

    /// Called once per image, after every tile/plane `export()` call for that
    /// image has completed, so an image that produced zero objects is still
    /// recorded somewhere — otherwise it leaves no trace at all (an exporter
    /// backed by a per-object table has nothing to insert a row *into* when
    /// there are no objects). Default no-op: only exporters that keep a
    /// separate per-image summary (e.g. `DuckDbExporter`'s `images` table)
    /// need to override this.
    ///
    /// `width`/`height`: the full (untiled) image's pixel dimensions -
    /// implementations that record per-image metadata (`DuckDbExporter`'s
    /// `images` table) persist these alongside the row.
    ///
    /// `nr_c_stacks`/`nr_z_stacks`/`nr_t_stacks`: the real number of
    /// channel/Z/T planes this image has, straight from its metadata -
    /// *not* how many of them this run actually processed (a Z-projection
    /// run still reports the image's true `nr_z_stacks`, even though it only
    /// ever produces one merged plane). `DuckDbExporter` persists these so
    /// the results view can enumerate per-channel intensity columns from a
    /// cheap `MAX(c_stacks)` over `images` instead of inferring the channel
    /// count from what happens to show up in already-measured objects.
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
        _width: u32,
        _height: u32,
        _nr_c_stacks: u32,
        _nr_z_stacks: u32,
        _nr_t_stacks: u32,
        _error: Option<&str>,
    ) -> Result<(), InternalErrors> {
        Ok(())
    }
}
