use crate::modules::classification_settings::ClassificationSettings;
use crate::modules::command_groups::GroupDefinition;
use crate::modules::meta_data::MetaData;
use crate::modules::pipeline_settings::PipelineSettings;
use crate::modules::pipeline_settings::PipelineStepSettings;
use crate::modules::plate_settings::PlateSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Default, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PipelineTemplate {
    /// On-disk format version of this pipeline template (`.evapipe`). Absent
    /// on files written before versioning was introduced, which
    /// `serde(default)` reads as `0` - see `CURRENT_PIPELINE_TEMPLATE_SCHEMA_VERSION`.
    /// Unlike `ProjectSettings.schema_version`, nothing currently rejects a
    /// too-new version or migrates an old one forward on load - this field
    /// only reserves the seam for that.
    #[serde(default)]
    pub schema_version: u32,

    /// Metadata of this pipeline template
    pub meta: MetaData,

    /// The pipeline settings
    pub steps: Vec<PipelineStepSettings>,
}

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Default, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTemplate {
    /// On-disk format version of this project template (`.evapt`). Absent on
    /// files written before versioning was introduced, which `serde(default)`
    /// reads as `0` - see `CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION` and
    /// `load_project_template`, which migrates older ones forward.
    #[serde(default)]
    pub schema_version: u32,

    /// Metadata of this project template
    pub meta: MetaData,

    /// Defined classes, labels, names and measurements
    pub classification: ClassificationSettings,

    /// Plate settings
    pub plate: PlateSettings,

    /// Pipelines to execute
    pub pipelines: Vec<PipelineSettings>,

    /// The command group definitions the pipelines use. Absent in templates
    /// saved before groups existed.
    #[serde(default)]
    pub command_groups: Vec<GroupDefinition>,
}
