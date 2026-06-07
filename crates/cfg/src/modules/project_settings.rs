use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::settings::{
    classification_settings::ClassificationSettings, experimant_meta_settings::ExperimentMetadata,
    images_settings::ImageSettings, pipeline_settings::PipelineSettings,
    plate_settings::PlateSettings,
};

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
    /// Descriptive information about the project (name, version, etc.).
    pub metadata: ExperimentMetadata,

    // Defined classes, labels, names and measurment
    pub classification: ClassificationSettings,

    pub plate: PlateSettings,

    /// The collection of images and their associated processing states.
    pub images: ImageSettings,

    /// Pipelines to execute
    pub pipelines: Vec<PipelineSettings>,
}
