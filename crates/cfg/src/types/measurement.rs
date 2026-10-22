//! What a measurement step measures - the description of its values,
//! shared by the steps (core), the results database and the results views.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// The unit of a measurement's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementUnit {
    /// Without a unit (Pearson, ratios).
    None,
    /// 0..1 (Manders, saturation).
    Fraction,
    /// Gray values.
    Intensity,
    Pixels,
    SquarePixels,
    Micrometers,
    SquareMicrometers,
    Count,
}

/// Whose values they are: one per object, or one per image (or the pixels
/// of a class in it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementTarget {
    Object,
    Image,
}

/// One measurement of a step, e.g. "Pearson (ch1 × ch2)" of the step
/// "Coloc nuclei".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementInfo {
    /// Stable and unique, e.g. "coloc_3/pearson/ch1×ch2".
    pub key: String,
    /// Shown in the results, e.g. "Pearson (ch1 × ch2)".
    pub name: String,
    /// The results' pickers group by it, e.g. "Colocalization · Coloc nuclei".
    pub group: String,
    pub unit: MeasurementUnit,
    pub target: MeasurementTarget,
    /// The values' range, if fixed (Pearson -1..1, Manders 0..1): the
    /// results' color scales use it, so plates stay comparable.
    pub range: Option<(f64, f64)>,
    /// Can be averaged per image or well.
    pub aggregatable: bool,
}

/// A measurement step's name is valid: letters, digits and `_` - it starts
/// its measurements' keys and the export's columns.
pub fn is_valid_measurement_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The first free name "`prefix`_01", "`prefix`_02", ... - `taken` holds the
/// names in use, lower-case (names differing only in case are the same).
pub fn next_measurement_name(prefix: &str, taken: &HashSet<String>) -> String {
    (1..)
        .map(|n| format!("{prefix}_{n:02}"))
        .find(|name| !taken.contains(&name.to_lowercase()))
        .expect("a free number")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_letters_digits_and_underscores() {
        assert!(is_valid_measurement_name("nuclei_coloc_2"));
        for bad in ["", "nuclei coloc", "a/b", "a.b", "äpfel", "x-y"] {
            assert!(!is_valid_measurement_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_new_name_takes_the_first_free_number_whatever_the_case() {
        let taken: HashSet<String> = ["pearson_01", "pearson_03"].map(String::from).into();
        assert_eq!(next_measurement_name("pearson", &taken), "pearson_02");
        let taken: HashSet<String> = ["pearson_01".to_string()].into();
        assert_eq!(next_measurement_name("PEARSON", &taken), "PEARSON_02");
        assert_eq!(
            next_measurement_name("texture", &HashSet::new()),
            "texture_01"
        );
    }
}
