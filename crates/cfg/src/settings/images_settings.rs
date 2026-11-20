// # images_settings
//
// **Author:** Joachim Danmayr
// **Date:** 2026-06-05
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::RangeInclusive,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::settings::object_settings::ObjectMetricSettings;
use crate::types::image::ZProjection;

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageSettings {
    /// The absolute base path for all image resources.
    pub root: Option<PathBuf>,

    /// Map of images where the key is the path relative to `root`.
    pub list: ImageList,

    /// Project-wide render and display settings.
    pub settings: GlobalImageSettings,

    /// Folders (relative to `root`) whose images an analysis leaves out:
    /// the images right in one, not those in its subfolders. They stay in
    /// `list` - viewable, previewable. Saved with the project so every run
    /// of it (app, CLI, server) analyses the same images.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub skipped_folders: BTreeSet<PathBuf>,

    /// Folders (relative to `root`) folded in the app's images list. Only
    /// how the list shows them - no effect on an analysis; folding marks the
    /// project unsaved but takes no undo step.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub collapsed_folders: BTreeSet<PathBuf>,
}

impl ImageSettings {
    /// Whether an analysis leaves out the image at `rel_path` (relative to
    /// `root`): it lies right in a skipped folder.
    pub fn is_skipped(&self, rel_path: &Path) -> bool {
        !self.skipped_folders.is_empty()
            && self
                .skipped_folders
                .contains(rel_path.parent().unwrap_or(Path::new("")))
    }
}

// In memory the map and every entry are shared between copies until one
// copy changes them (`Arc`, copy-on-write): an undo snapshot of a project
// with 100,000 images copies nothing. The methods are in `evanalyzer_app`'s
// `ImageListExt`.
/// The project's images, keyed by the path relative to the image root.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(transparent)]
#[schemars(inline)]
pub struct ImageList(pub Arc<IndexMap<PathBuf, Arc<ImageEntry>>>);

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageEntry {
    pub rel_path: PathBuf,
    pub file_size: u64,
    pub selected_series: i32,
    pub series: BTreeMap<i32, SeriesSettings>, // Image settings for each series
    /// What the user noted about the image: note, marker, left out. Copied
    /// into the results at a run - the results keep their own copy.
    #[serde(default, skip_serializing_if = "ImageReview::is_empty")]
    pub review: ImageReview,
}

/// The user's review of one image. One record for everything the user says
/// about an image, so later options (checked, tags…) get a place here.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ImageReview {
    /// Free text, e.g. "bubble in the lower left".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// A colour to find the image again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marker: Option<ImageMarker>,
    /// Kept out of the results: still analysed, but not in the numbers
    /// (the results show it disabled, with the reason).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leave_out: Option<LeaveOut>,
}

impl ImageReview {
    pub fn is_empty(&self) -> bool {
        self.note.is_empty() && self.marker.is_none() && self.leave_out.is_none()
    }
}

/// The colours an image can be marked with.
#[derive(
    Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ImageMarker {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Purple,
}

impl ImageMarker {
    pub const ALL: [ImageMarker; 6] = [
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Blue,
        Self::Purple,
    ];

    /// The marker's number: 1 (red) to 6 (purple) - its key in the app and
    /// its value in the results; 0 is no marker.
    pub fn number(self) -> u8 {
        self as u8 + 1
    }

    pub fn from_number(number: i64) -> Option<Self> {
        usize::try_from(number - 1)
            .ok()
            .and_then(|i| Self::ALL.get(i).copied())
    }

    /// The colour's English name, also in exports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Orange => "orange",
            Self::Yellow => "yellow",
            Self::Green => "green",
            Self::Blue => "blue",
            Self::Purple => "purple",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// Why an image is left out of the results.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LeaveOut {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<LeaveOutReason>,
}

impl LeaveOut {
    /// The reason as the results store it (`images.disabled_reason`).
    pub fn reason_text(&self) -> Option<&'static str> {
        self.reason.map(LeaveOutReason::text)
    }
}

#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LeaveOutReason {
    OutOfFocus,
    Artefact,
    Empty,
    Staining,
    Other,
}

impl LeaveOutReason {
    pub const ALL: [LeaveOutReason; 5] = [
        Self::OutOfFocus,
        Self::Artefact,
        Self::Empty,
        Self::Staining,
        Self::Other,
    ];

    /// The reason as shown and stored in the results.
    pub fn text(self) -> &'static str {
        match self {
            Self::OutOfFocus => "Out of focus",
            Self::Artefact => "Artefact",
            Self::Empty => "Empty",
            Self::Staining => "Staining",
            Self::Other => "Other",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GlobalImageSettings {
    pub selected_series: Option<i32>,
    #[serde(default)]
    pub grayscale: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rgb_unlinked: bool,
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
    pub objects: Vec<ObjectMetricSettings>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rgb: bool,
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
    /// The planes as one volume (3D). Not supported yet: module checks
    /// reject it until the commands can run on volumes.
    Volume3d,
}

impl ZStackHandling {
    /// The z-projection an analysis with this handling reads - `None` for
    /// the per-plane ones (and volumes): plane by plane.
    pub fn z_projection(&self) -> ZProjection {
        match self {
            Self::SingleStack | Self::AllStacks | Self::Volume3d => ZProjection::None,
            Self::MaxIntensity => ZProjection::MaxIntensity,
            Self::MinIntensity => ZProjection::MinIntensity,
            Self::AvgIntensity => ZProjection::AvgIntensity,
            Self::SumIntensity => ZProjection::SumIntensity,
            Self::TakeTheMiddle => ZProjection::TakeTheMiddle,
        }
    }
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
    // Playback speed
    pub playback_speed: f32,
    // Selected t stack
    pub t_stack: i32,
    /// Time between two frames in seconds, set for the project
    #[serde(default)]
    pub frame_interval: Option<f32>,
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
    pub emission_wave_length: Option<f32>,
    pub visible: Option<bool>,
    pub histogram: Option<HistogramSettings>,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PixelSizeSettings {
    pub x: f32,
    pub y: f32,
    /// The distance between two z-planes - `None` if unknown (the image
    /// doesn't say and nobody entered it). Never guessed: a volume's
    /// anisotropy (z / xy) depends on it.
    #[serde(default)]
    pub z: Option<f32>,
}

impl Default for PixelSizeSettings {
    fn default() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_plane_handlings_read_no_projection_and_projections_their_own() {
        for handling in [ZStackHandling::SingleStack, ZStackHandling::AllStacks] {
            assert_eq!(handling.z_projection(), ZProjection::None, "plane by plane");
        }
        let projections = [
            (ZStackHandling::MaxIntensity, ZProjection::MaxIntensity),
            (ZStackHandling::MinIntensity, ZProjection::MinIntensity),
            (ZStackHandling::AvgIntensity, ZProjection::AvgIntensity),
            (ZStackHandling::SumIntensity, ZProjection::SumIntensity),
            (ZStackHandling::TakeTheMiddle, ZProjection::TakeTheMiddle),
        ];
        for (handling, projection) in projections {
            assert_eq!(handling.z_projection(), projection);
        }
    }
}
