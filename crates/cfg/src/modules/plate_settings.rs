use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GroupingMode {
    #[default]
    NoGrouping,
    FolderName,
    FileName,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PlateSettings {
    pub grouping_mode: GroupingMode,
    pub grouping_regex: String,
    pub plate_cols: i32,
    pub plate_rows: i32,
    pub well_cols: i32,
    pub well_rows: i32,
    pub well_image_order: Vec<i32>,
}

impl Default for PlateSettings {
    fn default() -> Self {
        Self {
            grouping_mode: GroupingMode::default(),
            grouping_regex: String::default(),
            plate_cols: 1,
            plate_rows: 1,
            well_cols: 4,
            well_rows: 4,
            well_image_order: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        }
    }
}
