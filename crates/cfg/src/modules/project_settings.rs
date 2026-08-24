use crate::{
    modules::meta_data::MetaData,
    settings::{
        classification_settings::ClassificationSettings, images_settings::ImageSettings,
        pipeline_settings::PipelineSettings, plate_settings::PlateSettings,
    },
    types::classes::ObjectClass,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How close two tile-edge object fragments must be, in pixel connectivity
/// terms, to be considered the same real-world object - see [`TileMergeSettings`].
///
/// `ConnectedComponents` (the usual source of the instance labels tile-merge
/// fragments come from) always uses 8-connectivity, so `EightConnected` is
/// the default: it's what reproduces an untiled run's result exactly for the
/// common case. `FourConnected` is available for segmentation sources that
/// use plain 4-connectivity instead.
#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TileMergeConnectivity {
    FourConnected,
    #[default]
    EightConnected,
}

/// Project-level setting to reassemble objects that a tiled analysis run
/// split across an internal tile boundary (e.g. a whole organ/tissue region
/// on a whole-slide image) back into one correct object after every tile
/// finishes - see `docs/tile_merge_plan.md`.
///
/// `enabled: true` (the default) - a user has no reason to know or care
/// about tiles, they just expect their objects to be detected correctly,
/// which for a tiled run means split fragments merged back together. Every
/// class merges by default too (see `classes_to_not_merge`), so this is a
/// working, correct-by-default setup with nothing to configure. Set
/// `enabled: false` to fully restore the old per-tile-only export path (a
/// complete no-op, zero behavior change) - e.g. for a pipeline that only
/// ever runs single-tile, where the extra buffering/matching pass is pure
/// overhead.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct TileMergeSettings {
    pub enabled: bool,

    /// Opt-out deny-list: objects carrying any of these classes are excluded
    /// from tile-edge merging. Empty (the default) means every class is
    /// merge-eligible once `enabled` - a non-expert user turning this on has
    /// no reason to know about tiles at all and just expects objects to be
    /// detected correctly, not to configure a class list first.
    pub classes_to_not_merge: Vec<ObjectClass>,

    /// 4- or 8-connected boundary adjacency used to decide whether two
    /// fragments from different tiles are actually touching.
    pub connectivity: TileMergeConnectivity,

    /// Safety cap on how many tile-edge fragments a single merge group (one
    /// real-world object's worth of fragments, before matching) may contain
    /// before the run fails loudly instead of building a pathological merge
    /// - mirrors `ExtractObjects.max_objects_before_fail`. Since every class
    /// is merge-eligible by default, this is the guard against a class that
    /// turns out to match thousands of small tile-edge-touching objects
    /// (add it to `classes_to_not_merge` instead), not just a misconfigured
    /// allow-list.
    pub max_fragments_per_group: u32,
}

/// Hand-written rather than `#[derive(Default)]`: `max_fragments_per_group`
/// needs a real cap (`0` would reject every non-empty merge group
/// immediately), same reason `PlateSettings` hand-writes its `Default` too.
impl Default for TileMergeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            classes_to_not_merge: Vec::new(),
            connectivity: TileMergeConnectivity::default(),
            max_fragments_per_group: 10_000,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
    /// On-disk format version of this project file. Absent on files written
    /// before versioning was introduced, which `serde(default)` reads as `0`
    /// - see `CURRENT_PROJECT_SCHEMA_VERSION` and `load_project_settings` (the
    /// crate root's `versioning` module, not `include!()`-d here since this
    /// file also feeds `build.rs`'s schema generator).
    #[serde(default)]
    pub schema_version: u32,

    /// Descriptive information about the project (name, version, etc.).
    pub metadata: MetaData,

    // Defined classes, labels, names and measurment
    pub classification: ClassificationSettings,

    // Plate settings
    pub plate: PlateSettings,

    /// The collection of images and their associated processing states.
    pub images: ImageSettings,

    /// Pipelines to execute
    pub pipelines: Vec<PipelineSettings>,

    /// Setting to reassemble objects split across an internal tile boundary
    /// back into one correct object - on by default, matching what a user
    /// expects without needing to configure anything. Absent on files
    /// written before this feature existed, which `serde(default)` also
    /// reads as enabled - see `TileMergeSettings`.
    #[serde(default)]
    pub tile_merge: TileMergeSettings,
}
