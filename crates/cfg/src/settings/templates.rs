use crate::settings::classification_settings::ClassificationSettings;
use crate::settings::meta_data::MetaData;
use crate::settings::modules::ModuleDefinition;
use crate::settings::pipeline_settings::PipelineSettings;
use crate::settings::plate_settings::PlateSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A module saved to a library file (`.evamodule`). Inserting it copies
/// the module into a project, where it lives on its own from then on.
#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModuleTemplate {
    /// On-disk format version (`.evamodule`), see
    /// `CURRENT_MODULE_TEMPLATE_SCHEMA_VERSION`. Like the other templates, not
    /// migrated.
    #[serde(default)]
    pub schema_version: u32,

    /// The module, its fields at the top level of the file. Its `meta` is
    /// the template's: name, description, authors, version ("Update
    /// available" compares it).
    #[serde(flatten)]
    pub module: ModuleDefinition,

    /// The modules it uses inside, so the file is complete on its own.
    #[serde(default)]
    pub uses: Vec<ModuleDefinition>,
}

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Default, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTemplate {
    /// On-disk format version of this project template (`.evatemplate`). Absent on
    /// files written before versioning was introduced, which `serde(default)`
    /// reads as `0` - see `CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION`. Unlike
    /// `ProjectSettings.schema_version`, nothing currently rejects a too-new
    /// version or migrates an old one forward on load - this field only
    /// reserves the seam for that.
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

    /// The module definitions the pipelines use. Absent in templates
    /// saved before modules existed.
    #[serde(default)]
    pub modules: Vec<ModuleDefinition>,
}
