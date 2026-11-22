//! # number_format
//!
//! How the results show a number - one format in every view (design canvas
//! "Results rework", C): whole numbers as integers (`385`, not `385.000`),
//! "–" for no value, no thousands separator (`85741096` - as science
//! writes numbers; `.` or `,` would mean the decimal point in one country
//! or the other), the rest with 3 significant digits (`12.3`, `0.0123`),
//! very small or large ones as `1.23e-5`. Only for showing: exports keep
//! the full value.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

/// Shown for no value (an empty cell, NaN).
pub const NO_VALUE: &str = "–";

/// `value` as the results show it.
pub fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return NO_VALUE.to_string();
    }
    let magnitude = value.abs();
    // Whole numbers - counts, areas in pixels - as they are.
    if value.fract() == 0.0 && magnitude < 1e15 {
        return format!("{value:.0}");
    }
    if magnitude >= 1e15 || (magnitude < 1e-3 && magnitude > 0.0) {
        return format!("{value:.2e}");
    }
    // 3 significant digits, at least the whole part: 1234.5 -> 1 235.
    let digits_before = if magnitude >= 1.0 {
        magnitude.log10().floor() as i32 + 1
    } else {
        0
    };
    let decimals = if magnitude >= 1.0 {
        (3 - digits_before).max(0)
    } else {
        // 0.0123: the zeros after the point, then 3 digits.
        2 - magnitude.log10().floor() as i32
    } as usize;
    let text = format!("{value:.decimals$}");
    // Rounding up can make a whole number (0.9996 -> 1.000): as such.
    match text.parse::<f64>() {
        Ok(rounded) if rounded.fract() == 0.0 && decimals > 0 && rounded.abs() >= 1.0 => {
            format!("{rounded:.0}")
        }
        _ => text,
    }
}

/// `value` or [`NO_VALUE`].
pub fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| NO_VALUE.to_string(), format_number)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(value: f64) -> String {
        format_number(value)
    }

    #[test]
    fn whole_numbers_are_integers_with_thousands_apart() {
        assert_eq!(shown(385.0), "385");
        assert_eq!(shown(0.0), "0");
        assert_eq!(shown(1234.0), "1234");
        assert_eq!(shown(85_741_096.0), "85741096");
        assert_eq!(shown(-12_345.0), "-12345");
    }

    #[test]
    fn the_rest_has_three_significant_digits() {
        assert_eq!(shown(12.345), "12.3");
        assert_eq!(shown(1.2345), "1.23");
        assert_eq!(shown(123.45), "123");
        assert_eq!(shown(12_345.6), "12346");
        assert_eq!(shown(0.12345), "0.123");
        assert_eq!(shown(0.012345), "0.0123");
        assert_eq!(shown(-0.5), "-0.500");
        assert_eq!(shown(0.9996), "1");
        assert_eq!(shown(9.9996), "10");
    }

    #[test]
    fn tiny_and_huge_numbers_are_scientific() {
        assert_eq!(shown(0.000_012_3), "1.23e-5");
        assert_eq!(shown(2.5e16), "2.50e16");
    }

    #[test]
    fn no_value_is_a_dash() {
        assert_eq!(shown(f64::NAN), "–");
        assert_eq!(shown(f64::INFINITY), "–");
        assert_eq!(format_optional(None), "–");
        assert_eq!(format_optional(Some(2.0)), "2");
    }
}
