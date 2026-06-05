use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum PixelUnits {
    #[default]
    #[serde(rename = "bit")]
    Bit,
    #[serde(rename = "%")]
    Percent,
    #[serde(rename = "rel")]
    Relative,
}

impl PixelUnits {
    /// Convert a value in this unit to a normalized [0.0, 1.0] relative value.
    /// `nr_of_bits` is used only for `Bit` (8 → max 255, 16 → max 65535).
    #[allow(dead_code)]
    pub fn to_relative(self, value: f32, nr_of_bits: u8) -> f32 {
        match self {
            PixelUnits::Relative => value,
            PixelUnits::Percent => value / 100.0,
            PixelUnits::Bit => value / ((1u32 << nr_of_bits) - 1) as f32,
        }
    }
}

#[allow(dead_code)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum SizeUnits {
    #[default]
    #[serde(rename = "nm")]
    NanoMeter,
    #[serde(rename = "px")]
    Pixels,
}

impl SizeUnits {
    /// Convert a value in this unit to pixels.
    /// `pixel_size_nm` is the size of one pixel in nanometers (nm/px), used only for `NanoMeter`.
    #[allow(dead_code)]
    pub fn to_pixel(self, value: f32, pixel_size_nm: f32) -> usize {
        match self {
            SizeUnits::Pixels => value as usize,
            SizeUnits::NanoMeter => (value / pixel_size_nm) as usize,
        }
    }
}
