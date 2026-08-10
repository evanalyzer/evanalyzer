use crate::{
    modules::meta_data::MetaData,
    settings::{
        classification_settings::ClassificationSettings, images_settings::ImageSettings,
        pipeline_settings::PipelineSettings, plate_settings::PlateSettings,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
}
