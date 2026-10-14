//! Migrates a [`ProjectSettings`](crate::settings::project_settings::ProjectSettings)
//! document from schema version 3 to version 4. (Project templates hold
//! the same `classification` but aren't migrated - they may break.)
//!
//! Version 4 writes a class definition's id as a plain number: a class of
//! the project is always a project class (`ObjectClass::Global`), so its
//! kind no longer needs to be spelled out.
//!
//! - `{"GLOBAL": 3}` and the older `{"VALID": 3}` become `3`.
//! - A class without a number (`"UNSET"`, which only an imported legacy
//!   project could have) gets the next free one.

use serde_json::{Value, json};

/// The class number of an old `ObjectClass` value, if it has one.
fn class_number(id: &Value) -> Option<u64> {
    match id {
        Value::Number(n) => n.as_u64(),
        Value::Object(kind) => kind.get("GLOBAL").or(kind.get("VALID"))?.as_u64(),
        _ => None,
    }
}

pub fn migrate_from_v3_to_v4(raw: &mut Value) {
    let Some(Value::Array(classes)) = raw
        .get_mut("classification")
        .and_then(|classification| classification.get_mut("classes"))
    else {
        return;
    };
    let numbers: Vec<Option<u64>> = classes
        .iter()
        .map(|class| class.get("id").and_then(class_number))
        .collect();
    let mut next_free = numbers
        .iter()
        .flatten()
        .max()
        .map_or(1, |max| max + 1)
        .max(1);
    for (class, number) in classes.iter_mut().zip(numbers) {
        let number = number.unwrap_or_else(|| {
            next_free += 1;
            next_free - 1
        });
        if let Value::Object(class) = class {
            class.insert("id".into(), json!(number));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::classification_settings::ClassificationSettings;

    fn migrated(classes: Value) -> ClassificationSettings {
        let mut raw = json!({ "classification": { "classes": classes } });
        migrate_from_v3_to_v4(&mut raw);
        serde_json::from_value(raw["classification"].clone()).expect("a valid v4 classification")
    }

    fn class(id: Value) -> Value {
        json!({ "id": id, "color": "#97978F", "name": "", "notes": "" })
    }

    #[test]
    fn global_and_valid_ids_become_their_number() {
        let classification = migrated(json!([
            class(json!({ "GLOBAL": 0 })),
            class(json!({ "VALID": 3 })),
            class(json!({ "GLOBAL": 7 })),
        ]));
        let ids: Vec<u32> = classification.classes().iter().map(|c| c.id).collect();
        assert_eq!(ids, [0, 3, 7]);
    }

    #[test]
    fn classes_without_a_number_get_the_next_free_ones() {
        let classification = migrated(json!([
            class(json!({ "GLOBAL": 0 })),
            class(json!("UNSET")),
            class(json!({ "GLOBAL": 4 })),
            class(json!("UNSET")),
        ]));
        let ids: Vec<u32> = classification.classes().iter().map(|c| c.id).collect();
        assert_eq!(ids, [0, 5, 4, 6]);
    }

    #[test]
    fn an_already_migrated_document_is_left_as_it_is() {
        let classification = migrated(json!([class(json!(0)), class(json!(2))]));
        let ids: Vec<u32> = classification.classes().iter().map(|c| c.id).collect();
        assert_eq!(ids, [0, 2]);
    }
}
