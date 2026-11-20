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
    /// Scan zoom (confocal / scanning acquisition).
    pub zoom: Option<f32>,
    pub detector: DeviceInfo,
    pub light_source: LightSourceInfo,
    /// Filters in the excitation / emission path, in light-path order.
    pub excitation_filters: Vec<FilterInfo>,
    pub emission_filters: Vec<FilterInfo>,
    pub dichroic: DeviceInfo,
}

/// A device of the microscope as the file describes it (stand, detector,
/// dichroic); "" = unknown.
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceInfo {
    pub manufacturer: String,
    pub model: String,
    pub serial_number: String,
    /// Its type, e.g. "Inverted" (stand), "PMT" / "CMOS" (detector).
    pub kind: String,
}

impl DeviceInfo {
    /// Nothing known about it.
    pub fn is_empty(&self) -> bool {
        self.manufacturer.is_empty()
            && self.model.is_empty()
            && self.serial_number.is_empty()
            && self.kind.is_empty()
    }
}

/// A channel's light source; "" / `None` = unknown.
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LightSourceInfo {
    /// Manufacturer, model, serial number; `kind` the class ("Laser",
    /// "Arc", "LightEmittingDiode", ...).
    pub device: DeviceInfo,
    /// E.g. "SolidState" for a laser, "Hg" for an arc lamp.
    pub subtype: String,
    /// Laser medium, e.g. "Ar".
    pub laser_medium: String,
    /// The line used for the channel [nm].
    pub wavelength_nm: Option<f32>,
    /// Nominal power [mW].
    pub power_mw: Option<f32>,
    /// Fraction of the light blocked, 0 to 1.
    pub attenuation: Option<f32>,
}

/// An excitation or emission filter; "" / `None` = unknown.
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterInfo {
    pub manufacturer: String,
    pub model: String,
    /// E.g. "BandPass", "LongPass".
    pub kind: String,
    pub filter_wheel: String,
    /// The transmitted band [nm].
    pub cut_in_nm: Option<f32>,
    pub cut_out_nm: Option<f32>,
    /// Fraction transmitted in the band, 0 to 1.
    pub transmittance: Option<f32>,
}

/// Conditions during the acquisition; `None` = unknown.
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnvironmentInfo {
    pub temperature_c: Option<f32>,
    /// CO₂ as a fraction, 0 to 1 (0.05 = 5 %).
    pub co2_fraction: Option<f32>,
    /// Relative humidity as a fraction, 0 to 1.
    pub humidity_fraction: Option<f32>,
    pub air_pressure_mbar: Option<f32>,
}

/// Who acquired or owns the series; "" = unknown.
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimenterInfo {
    pub first_name: String,
    pub last_name: String,
    pub email: String,
    pub institution: String,
    /// Login name, e.g. in OMERO.
    pub user_name: String,
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
    /// The microscope stand.
    pub microscope: DeviceInfo,
    pub environment: EnvironmentInfo,
    pub experimenter: ExperimenterInfo,
    /// Names of the project and dataset the series belongs to ("" none).
    pub project: String,
    pub dataset: String,
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
    /// Optical correction ("PlanApo", "PlanFluor"), "" unknown.
    pub correction: String,
    pub working_distance_nm: Option<f32>,
    /// Refractive index of the immersion medium.
    pub refractive_index: Option<f32>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageMeta {
    pub name: String,
    pub objective: Objective,
    pub series: BTreeMap<i32, ImageInfo>, // Image series
}
