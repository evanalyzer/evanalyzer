use crate::types::classes::ObjectClass;
use crate::utils::hex_colors::hex_to_u32;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub id: ObjectClass,
    #[serde(with = "hex_to_u32")]
    #[schemars(with = "String")]
    pub color: u32,
    pub name: String,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationSettings {
    classes: Vec<Class>,
}

/// `#[derive(Default)]` would give `classes: vec![]`, violating the "always
/// contains at least Background" invariant `new`/`new_from_existing` exist to
/// enforce - route through `new()` instead so every `ClassificationSettings`,
/// however it's constructed, starts with Background present.
impl Default for ClassificationSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassificationSettings {
    pub fn new() -> Self {
        Self {
            classes: vec![Class {
                id: ObjectClass::BACKGROUND,
                color: 0x97978F,
                name: "Background".to_string(),
                notes: "".into(),
            }],
        }
    }

    #[allow(dead_code)]
    pub fn new_from_existing(mut classes: Vec<Class>) -> Self {
        let mut all_classes = vec![Class {
            id: ObjectClass::BACKGROUND,
            color: 0x97978F,
            name: "Background".to_string(),
            notes: "".into(),
        }];
        classes.retain(|class| class.id != ObjectClass::BACKGROUND);
        all_classes.extend(classes);
        Self {
            classes: all_classes,
        }
    }

    #[allow(dead_code)]
    pub fn classes_mut(&mut self) -> &mut Vec<Class> {
        &mut self.classes
    }

    #[allow(dead_code)]
    pub fn classes(&self) -> &Vec<Class> {
        &self.classes
    }
}
