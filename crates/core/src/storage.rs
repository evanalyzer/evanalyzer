use crate::{
    pipeline::pipeline_cache::GlobalPipelineCache,
    storage::stored_object::{ObjectFilter, StoredObject},
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
pub mod duckdb;
pub mod file;
pub mod memory;
pub mod stored_object;

pub trait PipelineResultExporter: Send + Sync {
    fn export(&self, cache: &GlobalPipelineCache) -> Result<(), InternalErrors>;

    /// The exported objects for dataset steps to read - `None` for storage
    /// that can't be read back (e.g. a CSV file). Default `None`.
    fn dataset_store(&self) -> Option<&dyn DatasetStore> {
        None
    }

    /// Called once per image, after every tile/plane `export()` call for that
    /// image has completed, so an image that produced zero objects is still
    /// recorded somewhere — otherwise it leaves no trace at all (an exporter
    /// backed by a per-object table has nothing to insert a row *into* when
    /// there are no objects). Default no-op: only exporters that keep a
    /// separate per-image summary (e.g. `DuckDbExporter`'s `images` table)
    /// need to override this.
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

    /// Called once when the run is over, with how it ended - so storage can
    /// tell a complete run from a cancelled, failed or interrupted one
    /// (a crash or a killed process never gets here). Default no-op.
    fn finish_run(&self, _outcome: &Result<(), InternalErrors>) -> Result<(), InternalErrors> {
        Ok(())
    }
}

/// The analysed objects, as dataset steps read them - typed, so steps don't
/// depend on the storage's table layout and can be tested in memory.
pub trait DatasetStore: Send + Sync {
    /// The objects matching `filter`, ordered by image, frame and object id.
    fn get_objects(&self, filter: &ObjectFilter) -> Result<Vec<StoredObject>, InternalErrors>;

    /// Sets the track of each listed object (`0`: none). Objects not
    /// listed keep theirs.
    fn set_track_ids(&self, tracks: &[(ObjectId, u64)]) -> Result<(), InternalErrors>;

    /// The highest track id of the objects that don't carry `class` - `0`
    /// if none is tracked - so a step tracking `class` can number its
    /// tracks after theirs without reading every object.
    fn max_track_id_except(&self, class: ObjectClass) -> Result<u64, InternalErrors>;
}
