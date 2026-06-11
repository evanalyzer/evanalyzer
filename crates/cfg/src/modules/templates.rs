use crate::modules::classification_settings::ClassificationSettings;
use crate::modules::meta_data::MetaData;
use crate::modules::pipeline_settings::PipelineStepSettings;
use crate::modules::plate_settings::PlateSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PipelineTemplate {
    /// Metadata of this pipeline template
    pub meta: MetaData,

    /// The pipeline settings
    pub pipeline_steps: Vec<PipelineStepSettings>,
}

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTemplate {
    /// Metadata of this project template
    pub meta: MetaData,

    /// Defined classes, labels, names and measurements
    pub classification: ClassificationSettings,

    /// Plate settings
    pub plate: PlateSettings,

    /// Pipelines to execute
    pub pipelines: Vec<PipelineTemplate>,
}
