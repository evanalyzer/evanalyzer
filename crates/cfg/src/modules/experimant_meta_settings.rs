use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimantMetadata {
    // Personal
    pub first_name: String,
    pub last_name: String,
    pub organization: String,

    // Experiment
    pub name: String,
    pub notes: String,
}
