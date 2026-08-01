use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::ops::Add;

// Segmentation class -----------------

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
    Default,
)]
pub struct SegmentationClass(pub u32);

impl SegmentationClass {
    pub const BACKGROUND: Self = SegmentationClass(0);
    pub const MANUAL_ANNOTATED: Self = SegmentationClass(0xFFFFFFFF);

    pub fn from_object_class(class: SegmentationClass) -> Self {
        SegmentationClass(class.0)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
}

impl From<SegmentationClass> for u32 {
    fn from(class: SegmentationClass) -> Self {
        class.0
    }
}

impl std::fmt::Display for SegmentationClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0 => write!(f, "Background"),
            0xFFFFFFFF => write!(f, "Manual"),
            n => write!(f, "{:02}", n),
        }
    }
}

// Object class -----------------

#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    JsonSchema,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ObjectClass {
    #[default]
    Unset,
    Valid(u32),
}

#[allow(dead_code)]
impl ObjectClass {
    pub fn from_segmentation_class(class: SegmentationClass) -> Self {
        ObjectClass::Valid(class.0)
    }

    pub fn to_u32(&self) -> Option<u32> {
        match self {
            ObjectClass::Valid(val) => Some(*val),
            ObjectClass::Unset => None,
        }
    }
    pub fn to_i32(&self) -> i32 {
        match self {
            ObjectClass::Unset => -1,
            ObjectClass::Valid(val) => *val as i32,
        }
    }
}

impl Add for ObjectClass {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            // Case 1: Both are valid numbers
            (ObjectClass::Valid(a), ObjectClass::Valid(b)) => {
                // Using wrapping_add to prevent panics on overflow
                ObjectClass::Valid(a.wrapping_add(b))
            }
            // Case 2: If either one is 'Special', the result is 'Special'
            _ => ObjectClass::Unset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_class_constants_have_the_expected_values() {
        assert_eq!(SegmentationClass::BACKGROUND, SegmentationClass(0));
        assert_eq!(
            SegmentationClass::MANUAL_ANNOTATED,
            SegmentationClass(0xFFFFFFFF)
        );
    }

    #[test]
    fn segmentation_class_from_object_class_copies_the_id() {
        assert_eq!(
            SegmentationClass::from_object_class(SegmentationClass(9)),
            SegmentationClass(9)
        );
    }

    #[test]
    fn segmentation_class_as_u32_returns_the_inner_value() {
        assert_eq!(SegmentationClass(42).as_u32(), 42);
    }

    #[test]
    fn segmentation_class_display_special_cases_background_and_manual() {
        assert_eq!(SegmentationClass::BACKGROUND.to_string(), "Background");
        assert_eq!(SegmentationClass::MANUAL_ANNOTATED.to_string(), "Manual");
        assert_eq!(SegmentationClass(7).to_string(), "07");
    }

    #[test]
    fn segmentation_class_into_u32_unwraps_the_inner_value() {
        let value: u32 = SegmentationClass(5).into();
        assert_eq!(value, 5);
    }

    #[test]
    fn segmentation_class_default_is_background() {
        assert_eq!(SegmentationClass::default(), SegmentationClass(0));
    }

    #[test]
    fn object_class_default_is_unset() {
        assert_eq!(ObjectClass::default(), ObjectClass::Unset);
    }

    #[test]
    fn object_class_from_segmentation_class_wraps_the_id_as_valid() {
        assert_eq!(
            ObjectClass::from_segmentation_class(SegmentationClass(3)),
            ObjectClass::Valid(3)
        );
    }

    #[test]
    fn object_class_to_u32_is_some_for_valid_and_none_for_unset() {
        assert_eq!(ObjectClass::Valid(4).to_u32(), Some(4));
        assert_eq!(ObjectClass::Unset.to_u32(), None);
    }

    #[test]
    fn object_class_to_i32_is_the_id_for_valid_and_negative_one_for_unset() {
        assert_eq!(ObjectClass::Valid(4).to_i32(), 4);
        assert_eq!(ObjectClass::Unset.to_i32(), -1);
    }

    #[test]
    fn adding_two_valid_object_classes_wrapping_adds_their_ids() {
        assert_eq!(
            ObjectClass::Valid(2) + ObjectClass::Valid(3),
            ObjectClass::Valid(5)
        );
        assert_eq!(
            ObjectClass::Valid(u32::MAX) + ObjectClass::Valid(1),
            ObjectClass::Valid(0)
        );
    }

    #[test]
    fn adding_an_unset_object_class_to_anything_yields_unset() {
        assert_eq!(
            ObjectClass::Valid(2) + ObjectClass::Unset,
            ObjectClass::Unset
        );
        assert_eq!(
            ObjectClass::Unset + ObjectClass::Valid(2),
            ObjectClass::Unset
        );
        assert_eq!(ObjectClass::Unset + ObjectClass::Unset, ObjectClass::Unset);
    }
}
