use crate::types::classes::ObjectClass;
use crate::utils::hex_colors::hex_to_u32;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub id: ObjectClass,
    #[serde(with = "hex_to_u32")]
    #[schemars(with = "String")]
    pub color: u32,
    pub name: String,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationSettings {
    pub classes: Vec<Class>,
}
