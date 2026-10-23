use crate::{
    pipeline::pipeline_cache::GlobalPipelineCache,
    storage::stored_object::{ObjectFilter, StoredObject},
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId};
pub mod duckdb;
pub mod file;
pub mod mask_codec;
pub mod memory;
pub mod stored_object;

/// What an analysed image was, as [`PipelineResultExporter::finalize_image`]
/// records it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageSummary<'a> {
    pub width: u32,
    pub height: u32,
    /// The image's own sizes.
    pub nr_c_stacks: u32,
    pub nr_z_stacks: u32,
    pub nr_t_stacks: u32,
    /// The time points and z-planes the run analysed of it (every time
    /// point × every z-plane) - also those where nothing was found, so the
    /// results tell an empty plane from one never analysed.
    pub t_stacks_analyzed: &'a [i32],
    pub z_stacks_analyzed: &'a [i32],
    /// Seconds between two time points: the project's frame interval, else
    /// the image's own (its metadata); `None` unknown.
    pub frame_interval_s: Option<f64>,
    /// Why "Exclude Image" excluded the image (`None`: it isn't) - it is
    /// written disabled, with this reason.
    pub excluded: Option<&'a str>,
}

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
        _summary: &ImageSummary,
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
