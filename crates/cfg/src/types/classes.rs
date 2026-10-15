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

/// The highest project class id the editor's class choices offer: they show
/// the classes 1 to 32 (and multi-class settings carry 33 flags, 0 to 32).
#[allow(dead_code)]
pub const MAX_CLASS_ID: u32 = 32;

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
    #[serde(alias = "VALID")]
    Global(u32),
    /// A class of a component (`ComponentDefinition::classes`): only valid
    /// inside the component, each instance has its own.
    Local(u32),
}

#[allow(dead_code)]
impl ObjectClass {
    pub const BACKGROUND: Self = ObjectClass::Global(0);

    pub fn from_segmentation_class(class: SegmentationClass) -> Self {
        ObjectClass::Global(class.0)
    }

    pub fn to_u32(&self) -> Option<u32> {
        match self {
            ObjectClass::Global(val) => Some(*val),
            ObjectClass::Unset | ObjectClass::Local(_) => None,
        }
    }
    /// The class as a command parameter's text value (`ParameterDef::value`,
    /// `apply_param_change`): `"-1"` unset, `"3"` a project class, `"L3"` a
    /// component's local class.
    pub fn to_text(&self) -> String {
        match self {
            ObjectClass::Unset => "-1".to_string(),
            ObjectClass::Global(n) => n.to_string(),
            ObjectClass::Local(n) => format!("L{n}"),
        }
    }

    /// Reads [`Self::to_text`]'s form back; `None` if `text` isn't one.
    pub fn from_text(text: &str) -> Option<Self> {
        let text = text.trim();
        if text == "-1" {
            return Some(ObjectClass::Unset);
        }
        match text.strip_prefix('L') {
            Some(local) => local.parse().ok().map(ObjectClass::Local),
            None => text.parse().ok().map(ObjectClass::Global),
        }
    }

    pub fn to_i32(&self) -> i32 {
        match self {
            ObjectClass::Unset | ObjectClass::Local(_) => -1,
            ObjectClass::Global(val) => *val as i32,
        }
    }
}

/// An object class ID of a trained model (what the model predicts), not one
/// of the project's classes - model class IDs are a snapshot taken at
/// training time and only line up with the project's by chance. Fields of
/// this type are left out of the generated `PipelineCommand::object_classes()`.
pub type ModelClass = ObjectClass;

/// The [`SegmentationClass`] counterpart of [`ModelClass`]: a segmentation
/// class ID of a trained model, not one of the project's.
pub type ModelSegmentationClass = SegmentationClass;

impl Add for ObjectClass {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            // Case 1: Both are valid numbers
            (ObjectClass::Global(a), ObjectClass::Global(b)) => {
                // Using wrapping_add to prevent panics on overflow
                ObjectClass::Global(a.wrapping_add(b))
            }
            // Case 2: If either one is 'Special', the result is 'Special'
            _ => ObjectClass::Unset,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn project_classes_are_written_as_global_and_old_valid_ones_still_load() {
        let json = serde_json::to_string(&ObjectClass::Global(3)).unwrap();
        assert_eq!(json, r#"{"GLOBAL":3}"#);
        let old: ObjectClass = serde_json::from_str(r#"{"VALID":3}"#).unwrap();
        assert_eq!(old, ObjectClass::Global(3), "files written before");
        // Projects load through `serde_json::Value`.
        let value: serde_json::Value = serde_json::from_str(r#"{"VALID":3}"#).unwrap();
        assert_eq!(
            serde_json::from_value::<ObjectClass>(value).unwrap(),
            ObjectClass::Global(3)
        );
    }

    use super::*;

    #[test]
    fn the_text_form_reads_back_and_tells_local_from_global() {
        for class in [
            ObjectClass::Unset,
            ObjectClass::Global(0),
            ObjectClass::Global(3),
            ObjectClass::Local(3),
        ] {
            assert_eq!(ObjectClass::from_text(&class.to_text()), Some(class));
        }
        assert_eq!(ObjectClass::Local(1).to_text(), "L1");
        assert_eq!(ObjectClass::from_text(" 4 "), Some(ObjectClass::Global(4)));
        for bad in ["", "L", "x1", "L-1", "-2"] {
            assert_eq!(ObjectClass::from_text(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn local_classes_are_written_with_their_kind() {
        let json = serde_json::to_string(&ObjectClass::Local(2)).unwrap();
        assert_eq!(json, r#"{"LOCAL":2}"#);
        assert_eq!(
            serde_json::from_str::<ObjectClass>(&json).unwrap(),
            ObjectClass::Local(2)
        );
        assert_eq!(ObjectClass::Local(2).to_u32(), None, "not a project class");
    }

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
            ObjectClass::Global(3)
        );
    }

    #[test]
    fn object_class_to_u32_is_some_for_valid_and_none_for_unset() {
        assert_eq!(ObjectClass::Global(4).to_u32(), Some(4));
        assert_eq!(ObjectClass::Unset.to_u32(), None);
    }

    #[test]
    fn object_class_to_i32_is_the_id_for_valid_and_negative_one_for_unset() {
        assert_eq!(ObjectClass::Global(4).to_i32(), 4);
        assert_eq!(ObjectClass::Unset.to_i32(), -1);
    }

    #[test]
    fn adding_two_valid_object_classes_wrapping_adds_their_ids() {
        assert_eq!(
            ObjectClass::Global(2) + ObjectClass::Global(3),
            ObjectClass::Global(5)
        );
        assert_eq!(
            ObjectClass::Global(u32::MAX) + ObjectClass::Global(1),
            ObjectClass::Global(0)
        );
    }

    #[test]
    fn adding_an_unset_object_class_to_anything_yields_unset() {
        assert_eq!(
            ObjectClass::Global(2) + ObjectClass::Unset,
            ObjectClass::Unset
        );
        assert_eq!(
            ObjectClass::Unset + ObjectClass::Global(2),
            ObjectClass::Unset
        );
        assert_eq!(ObjectClass::Unset + ObjectClass::Unset, ObjectClass::Unset);
    }
}
