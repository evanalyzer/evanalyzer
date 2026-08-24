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

    /// Every author, in display order (the first entry is treated as the
    /// primary author in template/pipeline picker dialogs, the rest as
    /// co-authors). Free-form display strings, not split into first/last.
    #[serde(default)]
    pub authors: Vec<String>,

    /// Author organization
    pub author_organization: String,

    /// Creation time
    pub creation_time: chrono::DateTime<chrono::Utc>,

    /// Category used to group templates in pickers (e.g. "Cell Biology / Uptake")
    #[serde(default)]
    pub category: String,

    /// Free-form keywords used to search/filter templates
    #[serde(default)]
    pub tags: Vec<String>,

    /// Version of EVAnalyzer that last wrote this file (`CARGO_PKG_VERSION`
    /// of the writing build). Empty for files written before this field
    /// existed, or built by hand rather than saved from the app.
    #[serde(default)]
    pub app_version: String,
}
