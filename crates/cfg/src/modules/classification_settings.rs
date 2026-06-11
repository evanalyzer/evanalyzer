use crate::utils::hex_colors::hex_to_u32;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::types::classes::ObjectClass;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MeasurementChannels {
    ObjectCount,
    Intersecting,
    AreaSize,
    Perimeter,
    Circularity,
    IntensityMin,
    IntensityMax,
    IntensityAvg,
    IntensitySum,
    Position,
    DistanceCenterToCenter,
    DistanceCenterToSurfaceMin,
    DistanceCenterToSurfaceMax,
    DistanceSurfaceToSurfaceMin,
    DistanceSurfaceToSurfaceMax,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MeasurementStatistics {
    Val,
    Avg,
    Min,
    Max,
    Stdev,
    Median,
    Sum,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub id: ObjectClass,
    #[serde(with = "hex_to_u32")]
    #[schemars(with = "String")]
    pub color: u32,
    pub name: String,
    pub notes: String,
    pub measure: IndexMap<MeasurementChannels, Vec<MeasurementStatistics>>,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationSettings {
    pub classes: Vec<Class>,
}
