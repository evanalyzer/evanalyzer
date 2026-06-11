use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MetaData {
    /// Name of the module
    pub name: String,

    /// A short one line description of the module
    pub short_description: String,

    /// A long detailed description of the module
    pub description: String,

    /// Author first name
    pub author_first_name: String,

    /// Author last name
    pub author_last_name: String,

    /// Author organization
    pub author_organization: String,

    /// Creation time
    pub creation_time: String,
}
