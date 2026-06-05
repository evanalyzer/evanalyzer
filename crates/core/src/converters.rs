use evanalyzer_cfg::core_types::InternalErrors;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LengthUnit {
    Meter,
    Centimeter,
    Millimeter,
    Micrometer,
    Nanometer,
}

impl TryFrom<&str> for LengthUnit {
    type Error = InternalErrors;

    fn try_from(unit: &str) -> Result<Self, Self::Error> {
        match unit {
            "m" => Ok(LengthUnit::Meter),
            "cm" => Ok(LengthUnit::Centimeter),
            "mm" => Ok(LengthUnit::Millimeter),
            "µm" | "um" => Ok(LengthUnit::Micrometer),
            "nm" => Ok(LengthUnit::Nanometer),
            _ => Err(InternalErrors::ParseError(format!(
                "Unknown unit: {}",
                unit
            ))),
        }
    }
}

impl LengthUnit {
    pub fn to_nanometers_factor(&self) -> f32 {
        match self {
            LengthUnit::Meter => 1_000_000_000.0,
            LengthUnit::Centimeter => 10_000_000.0,
            LengthUnit::Millimeter => 1_000_000.0,
            LengthUnit::Micrometer => 1_000.0,
            LengthUnit::Nanometer => 1.0,
        }
    }
}

/// Converts a wavelength in nm to an RGB [f32; 3] color.
/// Returns [0.0, 0.0, 0.0] if the wavelength is outside the visible spectrum.
pub fn wavelength_to_rgb_float(wavelength: f32) -> [f32; 3] {
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);

    // Images without given emission wave length have default value 0.
    // In this case we show a grayscale value. Because of float we assume < 1
    if wavelength <= 1.0 {
        return [1.0, 1.0, 1.0];
    }

    // Pure red
    if wavelength == 635.0 {
        return [1.0, 0.0, 0.0];
    }

    // Pure green
    if wavelength == 532.0 {
        return [0.0, 1.0, 0.0];
    }

    // Pure blue
    if wavelength == 450.0 {
        return [0.0, 0.0, 1.0];
    }

    // Calculate base RGB components
    if (380.0..440.0).contains(&wavelength) {
        r = -(wavelength - 440.0) / (440.0 - 380.0);
        b = 1.0;
    } else if (440.0..490.0).contains(&wavelength) {
        g = (wavelength - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (490.0..510.0).contains(&wavelength) {
        g = 1.0;
        b = -(wavelength - 510.0) / (510.0 - 490.0);
    } else if (510.0..580.0).contains(&wavelength) {
        r = (wavelength - 510.0) / (580.0 - 510.0);
        g = 1.0;
    } else if (580.0..645.0).contains(&wavelength) {
        r = 1.0;
        g = -(wavelength - 645.0) / (645.0 - 580.0);
    } else if (645.0..781.0).contains(&wavelength) {
        r = 1.0;
    }

    // Factor for intensity fade-out at the edges of the spectrum
    let factor = if (380.0..420.0).contains(&wavelength) {
        0.3 + 0.7 * (wavelength - 380.0) / (420.0 - 380.0)
    } else if (420.0..701.0).contains(&wavelength) {
        1.0
    } else if (701.0..781.0).contains(&wavelength) {
        0.3 + 0.7 * (780.0 - wavelength) / (780.0 - 700.0)
    } else {
        0.0
    };

    // Apply intensity factor
    [r * factor, g * factor, b * factor]
}

pub fn wavelength_to_rgb_u32(wavelength: f32) -> u32 {
    let color = wavelength_to_rgb_float(wavelength);
    let ret_color: u32 = ((color[0] * 255.0) as u32) << 16
        | ((color[1] * 255.0) as u32) << 8
        | (color[0] * 255.0) as u32;
    ret_color
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LengthUnit parsing ----

    #[test]
    fn length_unit_parses_all_known_symbols() {
        assert_eq!(LengthUnit::try_from("m").unwrap(),  LengthUnit::Meter);
        assert_eq!(LengthUnit::try_from("cm").unwrap(), LengthUnit::Centimeter);
        assert_eq!(LengthUnit::try_from("mm").unwrap(), LengthUnit::Millimeter);
        assert_eq!(LengthUnit::try_from("µm").unwrap(), LengthUnit::Micrometer);
        assert_eq!(LengthUnit::try_from("um").unwrap(), LengthUnit::Micrometer);
        assert_eq!(LengthUnit::try_from("nm").unwrap(), LengthUnit::Nanometer);
    }

    #[test]
    fn length_unit_rejects_unknown_symbol() {
        assert!(LengthUnit::try_from("px").is_err());
        assert!(LengthUnit::try_from("").is_err());
        assert!(LengthUnit::try_from("M").is_err()); // case-sensitive
    }

    #[test]
    fn length_unit_nanometer_factors_are_correct() {
        assert_eq!(LengthUnit::Nanometer.to_nanometers_factor(),    1.0);
        assert_eq!(LengthUnit::Micrometer.to_nanometers_factor(),   1_000.0);
        assert_eq!(LengthUnit::Millimeter.to_nanometers_factor(),   1_000_000.0);
        assert_eq!(LengthUnit::Centimeter.to_nanometers_factor(),   10_000_000.0);
        assert_eq!(LengthUnit::Meter.to_nanometers_factor(),        1_000_000_000.0);
    }

    // ---- wavelength_to_rgb_float ----

    #[test]
    fn wavelength_zero_returns_white() {
        assert_eq!(wavelength_to_rgb_float(0.0), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn wavelength_pure_red_635() {
        assert_eq!(wavelength_to_rgb_float(635.0), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn wavelength_pure_green_532() {
        assert_eq!(wavelength_to_rgb_float(532.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn wavelength_pure_blue_450() {
        assert_eq!(wavelength_to_rgb_float(450.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn wavelength_out_of_spectrum_is_black() {
        // Below 380 nm and above 780 nm → factor = 0 → all channels zero
        let below = wavelength_to_rgb_float(300.0);
        assert_eq!(below, [0.0, 0.0, 0.0]);
        let above = wavelength_to_rgb_float(800.0);
        assert_eq!(above, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn wavelength_components_are_normalised() {
        for wl in [380.0f32, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 780.0] {
            let [r, g, b] = wavelength_to_rgb_float(wl);
            assert!(r >= 0.0 && r <= 1.0, "r out of range at {wl}");
            assert!(g >= 0.0 && g <= 1.0, "g out of range at {wl}");
            assert!(b >= 0.0 && b <= 1.0, "b out of range at {wl}");
        }
    }
}
