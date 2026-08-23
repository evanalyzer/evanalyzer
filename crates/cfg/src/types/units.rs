use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PixelUnits {
    #[default]
    #[serde(alias = "bit")]
    Bit,
    #[serde(alias = "%")]
    Percent,
    #[serde(alias = "rel")]
    Relative,
}

impl PixelUnits {
    /// Convert a value in this unit to a normalized [0.0, 1.0] relative value.
    /// `nr_of_bits` is used only for `Bit` (8 → max 255, 16 → max 65535).
    #[allow(dead_code)]
    pub fn to_relative(self, value: f32, nr_of_bits: u16) -> f32 {
        match self {
            PixelUnits::Relative => value,
            PixelUnits::Percent => value / 100.0,
            PixelUnits::Bit => value / ((1u32 << nr_of_bits) - 1) as f32,
        }
    }
}

#[allow(dead_code)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SizeUnits {
    #[default]
    #[serde(alias = "nm")]
    NanoMeter,
    #[serde(alias = "px")]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `SCREAMING_SNAKE_CASE` is the canonical on-disk shape, matching every
    /// other enum in the codebase - see `PixelUnits`/`SizeUnits` old `rename`
    /// short forms, kept only as read aliases below.
    #[test]
    fn pixel_units_serializes_as_screaming_snake_case() {
        assert_eq!(serde_json::to_value(PixelUnits::Bit).unwrap(), "BIT");
        assert_eq!(
            serde_json::to_value(PixelUnits::Percent).unwrap(),
            "PERCENT"
        );
        assert_eq!(
            serde_json::to_value(PixelUnits::Relative).unwrap(),
            "RELATIVE"
        );
    }

    /// The old short forms (`"bit"`, `"%"`, `"rel"`) are what every shipped
    /// template/project file on disk was written with before this switch -
    /// they must keep deserializing.
    #[test]
    fn pixel_units_deserializes_both_old_short_forms_and_the_new_form() {
        for (json, expected) in [
            (serde_json::json!("BIT"), PixelUnits::Bit),
            (serde_json::json!("bit"), PixelUnits::Bit),
            (serde_json::json!("PERCENT"), PixelUnits::Percent),
            (serde_json::json!("%"), PixelUnits::Percent),
            (serde_json::json!("RELATIVE"), PixelUnits::Relative),
            (serde_json::json!("rel"), PixelUnits::Relative),
        ] {
            let parsed: PixelUnits = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{json} failed to deserialize: {e}"));
            assert_eq!(parsed, expected, "for input {json}");
        }
    }

    #[test]
    fn size_units_serializes_as_screaming_snake_case() {
        assert_eq!(
            serde_json::to_value(SizeUnits::NanoMeter).unwrap(),
            "NANO_METER"
        );
        assert_eq!(serde_json::to_value(SizeUnits::Pixels).unwrap(), "PIXELS");
    }

    #[test]
    fn size_units_deserializes_both_old_short_forms_and_the_new_form() {
        for (json, expected) in [
            (serde_json::json!("NANO_METER"), SizeUnits::NanoMeter),
            (serde_json::json!("nm"), SizeUnits::NanoMeter),
            (serde_json::json!("PIXELS"), SizeUnits::Pixels),
            (serde_json::json!("px"), SizeUnits::Pixels),
        ] {
            let parsed: SizeUnits = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{json} failed to deserialize: {e}"));
            assert_eq!(parsed, expected, "for input {json}");
        }
    }
}
