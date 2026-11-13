// # classification_settings
//
// **Author:** Joachim Danmayr
// **Date:** 2026-06-05
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::types::classes::ObjectClass;
use crate::utils::hex_colors::hex_to_u32;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub id: u32,
    #[serde(with = "hex_to_u32")]
    #[schemars(with = "String")]
    pub color: u32,
    pub name: String,
    pub notes: String,
    /// What the results keep of the class's objects - missing in projects
    /// from before it: [`ClassResults::Objects`].
    #[serde(default)]
    pub results: ClassResults,
}

/// What the results keep of a class's objects. The pipeline and the viewer
/// see every class whatever it says.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ClassResults {
    /// Every object with its measurements.
    #[default]
    Objects,
    /// Only for later steps of the pipeline (e.g. the seeds of Grow
    /// Regions): nothing of the class is written to the results - an object
    /// in no other class isn't stored, nor its measurements.
    Intermediate,
}

#[allow(dead_code)] // unused in build.rs' copy of this file
impl Class {
    /// This class as a project class - for the classes of
    /// [`ClassificationSettings`].
    pub fn global(&self) -> ObjectClass {
        ObjectClass::Global(self.id)
    }

    /// This class as a module's local class - for the classes of
    /// `ModuleDefinition::classes`.
    pub fn local(&self) -> ObjectClass {
        ObjectClass::Local(self.id)
    }
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
/// [`ObjectClass::BACKGROUND`]'s number.
const BACKGROUND_ID: u32 = 0;

impl Default for ClassificationSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassificationSettings {
    pub fn new() -> Self {
        Self {
            classes: vec![Class {
                id: BACKGROUND_ID,
                color: 0x97978F,
                name: "Background".to_string(),
                notes: "".into(),
                results: ClassResults::Objects,
            }],
        }
    }

    #[allow(dead_code)]
    pub fn new_from_existing(mut classes: Vec<Class>) -> Self {
        let mut all_classes = vec![Class {
            id: BACKGROUND_ID,
            color: 0x97978F,
            name: "Background".to_string(),
            notes: "".into(),
            results: ClassResults::Objects,
        }];
        classes.retain(|class| class.id != BACKGROUND_ID);
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
