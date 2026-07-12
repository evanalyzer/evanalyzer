use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    CURRENT_PROJECT_SCHEMA_VERSION,
    modules::meta_data::MetaData,
    settings::{
        classification_settings::ClassificationSettings, images_settings::ImageSettings,
        pipeline_settings::PipelineSettings, plate_settings::PlateSettings,
    },
};

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
    /// On-disk format version of this project file. Absent on files written
    /// before versioning was introduced, which `serde(default)` reads as `0`
    /// - see [`CURRENT_PROJECT_SCHEMA_VERSION`] and [`ProjectSettings::migrate`].
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

impl ProjectSettings {
    /// Upgrades a freshly-deserialized `ProjectSettings` to the current
    /// on-disk shape and stamps it with [`CURRENT_PROJECT_SCHEMA_VERSION`].
    ///
    /// Every field added to this struct (or anything it contains) so far has
    /// been an additive `#[serde(default)]` field, so there is nothing to
    /// migrate yet - this is the seam future breaking changes hook into.
    /// Keep each step keyed on the version it upgrades *from*, so a file that
    /// is several versions old runs every step in order:
    ///
    /// ```ignore
    /// if self.schema_version < 2 {
    ///     // migrate version 0/1 data into the version-2 shape
    /// }
    /// ```
    pub fn migrate(&mut self) {
        self.schema_version = CURRENT_PROJECT_SCHEMA_VERSION;
    }
}
