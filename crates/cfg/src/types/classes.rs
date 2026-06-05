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
