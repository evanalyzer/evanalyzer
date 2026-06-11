use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::RangeInclusive, path::PathBuf};

use crate::settings::roi_settings::RoiSettings;

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageSettings {
    /// The absolute base path for all image resources.
    pub root: Option<PathBuf>,

    /// Map of images where the key is the path relative to `root`.
    pub list: IndexMap<PathBuf, ImageEntry>,

    /// Project-wide render and display settings.
    pub settings: GlobalImageSettings,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageEntry {
    pub rel_path: PathBuf,
    pub file_size: u64,
    pub selected_series: i32,
    pub series: BTreeMap<i32, SeriesSettings>, // Image settings for each series
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GlobalImageSettings {
    pub selected_channel: Option<i32>,
    pub channels: BTreeMap<i32, ChannelSettings>, // Key is the channel
    pub pixel_sizes: Option<PixelSizeSettings>,
    pub z_stack: Option<ZStackSettings>,
    pub t_stack: Option<TStackSettings>,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SeriesSettings {
    pub selected_channel: Option<i32>,
    pub channels: BTreeMap<i32, ChannelSettings>, // Key is the channel
    pub image_width: u64,
    pub image_height: u64,
    pub pixel_sizes: PixelSizeSettings,
    pub z_stack: Option<ZStackSettings>,
    pub t_stack: Option<TStackSettings>,
    pub rois: Vec<RoiSettings>,
}

#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ZStackHandling {
    #[default]
    SingleStack,
    AllStacks,
    MaxIntensity,
    MinIntensity,
    AvgIntensity,
    SumIntensity,
    TakeTheMiddle,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ZStackSettings {
    pub z_projection: ZStackHandling,
    pub z_range: Option<RangeInclusive<i32>>,
}

#[derive(Serialize, Deserialize, Default, JsonSchema, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TStackHandling {
    #[default]
    SingleStack,
    AllStacks,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TStackSettings {
    pub stack_handling: TStackHandling,
    pub playback_speed: f32, // Playback speed
    pub t_stack: i32,        // Selected t stack
}
#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HistogramSettings {
    pub min: f32,       // Histogram min value
    pub max: f32,       // Histogram max value
    pub min_limit: f32, // Histogram range
    pub max_limit: f32, // Histogram range
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChannelSettings {
    pub name: String,
    pub emission_wave_length: f32,
    pub visible: Option<bool>,
    pub histogram: Option<HistogramSettings>,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PixelSizeSettings {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for PixelSizeSettings {
    fn default() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }
}
