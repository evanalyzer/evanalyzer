use crate::settings::modules::ModuleDefinition;
use crate::{
    settings::meta_data::MetaData,
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
    #[serde(alias = "metadata")]
    pub meta: MetaData,

    // Defined classes, labels, names and measurment
    pub classification: ClassificationSettings,

    // Plate settings
    pub plate: PlateSettings,

    /// Setting to reassemble objects split across an internal tile boundary
    #[serde(default)]
    pub tile_merge: TileMergeSettings,

    /// The collection of images and their associated processing states.
    pub images: ImageSettings,

    /// Pipelines to execute
    pub pipelines: Vec<PipelineSettings>,

    /// The module definitions; pipelines use them as instances
    /// (by `key`). Absent in projects saved before modules existed.
    #[serde(default)]
    pub modules: Vec<ModuleDefinition>,

    /// What the editor keeps with the project: the Tuning panel's pinned
    /// settings, which images were previewed or analysed.
    #[serde(default)]
    pub editor: EditorSettings,

    /// How the results of the project's analyses are shown: its views
    /// (List, Plate, Charts, Tracks with their settings), the same for
    /// every analysis.
    #[serde(default)]
    pub results_views: ResultsViews,
}

/// See `ProjectSettings::results_views`.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ResultsViews {
    pub views: Vec<ResultsView>,
    /// The view shown (its `id`).
    pub active: u32,
}

/// A new project's views: one of each kind, so the Results look like the
/// List / Plate / Charts / Tracks they always had until views are added.
impl Default for ResultsViews {
    fn default() -> Self {
        let view = |id: u32, name: &str, kind: ResultsViewKind| ResultsView {
            id,
            name: name.to_string(),
            kind,
            split: false,
            settings: serde_json::Value::Null,
        };
        Self {
            views: vec![
                view(1, "List", ResultsViewKind::List),
                view(2, "Plate", ResultsViewKind::Plate),
                view(3, "Charts", ResultsViewKind::Charts),
                view(4, "Tracks", ResultsViewKind::Tracks),
            ],
            active: 1,
        }
    }
}

/// One view of the results: what it is (`kind`), whether the image is
/// shown beside it, and its settings - classes, columns, aggregations, the
/// plate's value, the chart's axes, ... The settings are the Results
/// view's own (the GUI reads and writes them); `null` until first shown.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ResultsView {
    /// Stable while the view exists (also when renamed or moved).
    pub id: u32,
    pub name: String,
    pub kind: ResultsViewKind,
    /// "Split with image".
    #[serde(default)]
    pub split: bool,
    #[serde(default)]
    pub settings: serde_json::Value,
}

/// What a results view shows.
#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ResultsViewKind {
    List,
    Plate,
    Charts,
    Tracks,
}

/// See `ProjectSettings::editor`.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EditorSettings {
    /// Settings pinned into the Tuning panel.
    #[serde(default)]
    pub pins: Vec<PinnedSetting>,
    /// Images (relative to the image folder) a preview ran on.
    #[serde(default)]
    pub previewed_images: Vec<std::path::PathBuf>,
    /// Images (relative to the image folder) of the analysis last opened in
    /// Results.
    #[serde(default)]
    pub analysed_images: Vec<std::path::PathBuf>,
}

/// A setting pinned into the Tuning panel: in which pipeline, which step
/// (its id, so it stays when steps move), which setting (`name`, or
/// `list.entry.field` for a list entry's).
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct PinnedSetting {
    pub pipeline: u32,
    pub step: String,
    pub setting: String,
}
