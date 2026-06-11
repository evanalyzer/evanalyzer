use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    modules::meta_data::MetaData,
    settings::{
        classification_settings::ClassificationSettings, images_settings::ImageSettings,
        pipeline_settings::PipelineSettings, plate_settings::PlateSettings,
    },
};

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
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
