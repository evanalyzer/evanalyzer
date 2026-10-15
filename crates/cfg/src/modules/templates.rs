use crate::modules::classification_settings::ClassificationSettings;
use crate::modules::components::ComponentDefinition;
use crate::modules::meta_data::MetaData;
use crate::modules::pipeline_settings::PipelineSettings;
use crate::modules::plate_settings::PlateSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A component saved to a library file (`.evacomp`). Inserting it copies
/// the component into a project, where it lives on its own from then on.
#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComponentTemplate {
    /// On-disk format version (`.evacomp`), see
    /// `CURRENT_COMPONENT_TEMPLATE_SCHEMA_VERSION`. Like the other templates, not
    /// migrated.
    #[serde(default)]
    pub schema_version: u32,

    /// The component. Its `meta` is the template's: name, description, authors,
    /// version ("Update available" compares it).
    pub component: ComponentDefinition,

    /// The components it uses inside, so the file is complete on its own.
    #[serde(default)]
    pub uses: Vec<ComponentDefinition>,
}

#[allow(unused)]
#[derive(Serialize, Deserialize, Debug, Default, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTemplate {
    /// On-disk format version of this project template (`.evapt`). Absent on
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

    /// The component definitions the pipelines use. Absent in templates
    /// saved before components existed.
    #[serde(default)]
    pub components: Vec<ComponentDefinition>,
}
