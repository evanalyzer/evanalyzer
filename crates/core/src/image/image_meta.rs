//! # image_meta
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub use evanalyzer_cfg::core_types::{ImagePlane, ImageTile};

/// Pixel position in an image - e.g. a tile's top-left corner within the
/// full image (`ManagedImage::tile_offset`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Point2d {
    pub x: usize,
    pub y: usize,
}
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One channel as the file describes it. Values the file doesn't have are
/// empty, 0 or `None`. (`serde(default)`: an older server's reply, without
/// the acquisition values, still reads.)
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChannelInfo {
    pub id: String,
    pub name: String,
    pub emission_wave_length: f32, // Emission wave length in nm
    pub contrast_method: String,
    /// Fluorophore, e.g. "Cy5".
    pub fluor: String,
    /// Excitation wavelength in nm, 0 unknown.
    pub excitation_wave_length: f32,
    /// Exposure time configured for the channel [s].
    pub exposure_s: Option<f32>,
    pub detector_gain: Option<f32>,
    pub detector_offset: Option<f32>,
    /// Detector binning (x, y).
    pub binning: Option<(u16, u16)>,
    /// Pinhole size [nm] (confocal).
    pub pinhole_nm: Option<f32>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct PyramidInfo {
    pub nr_bits: u16,
    pub color_channels: u8, // Is either 1, 3 or 4
    pub is_rgb: bool,
    pub width: u64,
    pub height: u64,
    pub tile_width: u64,
    pub tile_height: u64,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct PixelSizes {
    pub px_size_x: f32, // Pixel x size in nm
    pub px_size_y: f32, // Pixel y size in nm
    pub px_size_z: f32, // Pixel z size in nm
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageInfo {
    /// The series' own name and description in the file ("" none).
    pub name: String,
    pub description: String,
    /// Acquisition start as the file says it (ISO 8601), "" unknown.
    pub acquisition_date: String,
    /// Bits actually used per sample (12 for 12-bit data in 16-bit).
    pub significant_bits: Option<u8>,
    /// Stage position of the series [nm] (x, y, z), each `None` if unknown.
    pub stage_position_nm: [Option<f64>; 3],
    pub nr_c_stacks: i32,
    pub nr_z_stacks: i32,
    pub nr_t_stacks: i32,
    pub pixel_sizes: PixelSizes,
    pub resolutions: BTreeMap<i32, PyramidInfo>, // Array of resolutions in case of a pyamid image
    pub channels: BTreeMap<i32, ChannelInfo>, // Contains the channel information <channelIdx | channelinfo>
    /// Time between two frames [s] per the file's metadata
    pub frame_interval: Option<f32>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Objective {
    pub manufacturer: String,
    pub model: String,
    pub magnification: f32,
    /// Numerical aperture, 0 unknown.
    pub numerical_aperture: f32,
    /// Immersion medium ("Oil", "Water", "Air"), "" unknown.
    pub immersion: String,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageMeta {
    pub name: String,
    pub objective: Objective,
    pub series: BTreeMap<i32, ImageInfo>, // Image series
}
