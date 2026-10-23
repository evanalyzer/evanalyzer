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
//!
//! It also names a segmentation class setting what it is:
//!
//! - `objectClassId` (Threshold entries, Cellpose, StarDist, U-Net, the pixel
//!   classifier's mapping) becomes `segmentationClass`.
//! - In the pixel classifier's mapping the model's class, so far
//!   `segmentationClass`, becomes `modelClass` - as in YOLO's mapping.
//! - A module's parameter bindings name settings by their path
//!   (`thresholds.0.object_class_id`); they follow the same renames.
//!
//! An annotated object's `parentId` and `children` go: links are `parents`
//! now (parent class -> parent, set by Relate Objects), and annotations
//! never had any.
//!
//! Colocalization marks the objects of one class now: its `classesToColoc`
//! becomes `objects` (the first class - it was checked like every other) and
//! `overlapping` (the rest), with no class to mark them with (`markAs`). A
//! module parameter bound to `classes_to_coloc` sets `overlapping`.

use serde_json::{Map, Value, json};

/// The class number of an old `ObjectClass` value, if it has one.
fn class_number(id: &Value) -> Option<u64> {
    match id {
        Value::Number(n) => n.as_u64(),
        Value::Object(kind) => kind.get("GLOBAL").or(kind.get("VALID"))?.as_u64(),
        _ => None,
    }
}

fn migrate_class_ids(raw: &mut Value) {
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

/// Renames the keys of `object` and of every object inside it.
fn rename_keys(value: &mut Value) {
    match value {
        Value::Object(object) => {
            rename_in(object);
            object.values_mut().for_each(rename_keys);
        }
        Value::Array(items) => items.iter_mut().for_each(rename_keys),
        _ => {}
    }
}

fn rename_in(object: &mut Map<String, Value>) {
    let Some(class) = object.remove("objectClassId") else {
        return;
    };
    // The pixel classifier's mapping: its `segmentationClass` was the model's.
    if let Some(model_class) = object.remove("segmentationClass") {
        object.insert("modelClass".into(), model_class);
    }
    object.insert("segmentationClass".into(), class);
}

/// `param` of a binding to a step running a command of type `command_type`,
/// with the renamed settings.
fn renamed_path(param: &str, command_type: Option<&str>) -> String {
    let segments: Vec<&str> = param.split('.').collect();
    let pixel_mapping = command_type == Some("PIXEL_CLASSIFIER")
        && segments.first() == Some(&"segmentation_mapping");
    segments
        .iter()
        .map(|segment| match *segment {
            "object_class_id" => "segmentation_class",
            "segmentation_class" if pixel_mapping => "model_class",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// The bindings of a module definition, renamed - before its commands are.
fn rename_bindings(module: &mut Value) {
    let types: Vec<(Value, Option<String>)> = module
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|step| {
            (
                step.get("id").cloned().unwrap_or(Value::Null),
                step.get("command")
                    .and_then(|c| c.get("type"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
            )
        })
        .collect();
    let Some(Value::Array(parameters)) = module.get_mut("parameters") else {
        return;
    };
    for parameter in parameters {
        let Some(Value::Array(bindings)) = parameter.get_mut("bindings") else {
            continue;
        };
        for binding in bindings {
            let command_type = types
                .iter()
                .find(|(id, _)| Some(id) == binding.get("step"))
                .and_then(|(_, t)| t.as_deref());
            if let Some(Value::String(param)) = binding.get_mut("param") {
                *param = renamed_path(param, command_type);
            }
        }
    }
}

/// Renames the settings of every command in `steps`.
fn rename_commands(steps: Option<&mut Value>) {
    if let Some(Value::Array(steps)) = steps {
        for step in steps {
            if let Some(command) = step.get_mut("command") {
                rename_keys(command);
            }
        }
    }
}

/// Renames the segmentation class settings in pipelines and modules.
fn migrate_segmentation_class_settings(raw: &mut Value) {
    if let Some(Value::Array(pipelines)) = raw.get_mut("pipelines") {
        for pipeline in pipelines {
            rename_commands(pipeline.get_mut("steps"));
        }
    }
    if let Some(Value::Array(modules)) = raw.get_mut("modules") {
        for module in modules {
            rename_bindings(module);
            rename_commands(module.get_mut("steps"));
        }
    }
}

/// Drops `parentId` / `children` from the annotated objects.
fn drop_object_relation_fields(raw: &mut Value) {
    let Some(Value::Object(images)) = raw.get_mut("images").and_then(|i| i.get_mut("list"))
    else {
        return;
    };
    for image in images.values_mut() {
        let Some(Value::Object(series)) = image.get_mut("series") else {
            continue;
        };
        for one in series.values_mut() {
            let Some(Value::Array(objects)) = one.get_mut("objects") else {
                continue;
            };
            for object in objects.iter_mut().filter_map(Value::as_object_mut) {
                object.remove("parentId");
                object.remove("children");
            }
        }
    }
}

/// Colocalization's `classesToColoc` -> `objects` + `overlapping` + `markAs`.
fn migrate_colocalization(steps: Option<&mut Value>) {
    let Some(Value::Array(steps)) = steps else {
        return;
    };
    for step in steps {
        let Some(Value::Object(command)) = step.get_mut("command") else {
            continue;
        };
        let is_coloc = matches!(
            command.get("type").and_then(Value::as_str),
            Some("COLOCALIZATION" | "colocalization")
        );
        if !is_coloc {
            continue;
        }
        let Some(Value::Array(classes)) = command.remove("classesToColoc") else {
            continue;
        };
        let mut classes = classes.into_iter();
        command.insert("objects".into(), classes.next().unwrap_or(json!("UNSET")));
        command.insert("overlapping".into(), Value::Array(classes.collect()));
        command.entry("markAs").or_insert(json!("UNSET"));
    }
}

/// A module's bindings to `classes_to_coloc` bind `overlapping`.
fn migrate_colocalization_bindings(module: &mut Value) {
    let Some(Value::Array(parameters)) = module.get_mut("parameters") else {
        return;
    };
    for parameter in parameters {
        let Some(Value::Array(bindings)) = parameter.get_mut("bindings") else {
            continue;
        };
        for binding in bindings {
            if let Some(Value::String(param)) = binding.get_mut("param")
                && param == "classes_to_coloc"
            {
                *param = "overlapping".into();
            }
        }
    }
}

pub fn migrate_from_v3_to_v4(raw: &mut Value) {
    migrate_class_ids(raw);
    migrate_segmentation_class_settings(raw);
    drop_object_relation_fields(raw);
    if let Some(Value::Array(pipelines)) = raw.get_mut("pipelines") {
        for pipeline in pipelines {
            migrate_colocalization(pipeline.get_mut("steps"));
        }
    }
    if let Some(Value::Array(modules)) = raw.get_mut("modules") {
        for module in modules {
            migrate_colocalization_bindings(module);
            migrate_colocalization(module.get_mut("steps"));
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
    fn colocalization_marks_the_first_class_and_overlaps_the_rest() {
        let coloc = |classes: Value| {
            json!({ "id": 3, "command": { "type": "COLOCALIZATION", "classesToColoc": classes,
                "minColocArea": 0.0 } })
        };
        let mut raw = json!({
            "pipelines": [{ "steps": [coloc(json!([{ "GLOBAL": 1 }, { "GLOBAL": 2 }, { "GLOBAL": 4 }]))] }],
            "modules": [{
                "steps": [coloc(json!([]))],
                "parameters": [{ "bindings": [{ "step": 3, "param": "classes_to_coloc" }] }]
            }]
        });
        migrate_from_v3_to_v4(&mut raw);
        let command = &raw["pipelines"][0]["steps"][0]["command"];
        assert_eq!(command["objects"], json!({ "GLOBAL": 1 }));
        assert_eq!(command["overlapping"], json!([{ "GLOBAL": 2 }, { "GLOBAL": 4 }]));
        assert_eq!(command["markAs"], json!("UNSET"));
        assert!(command.get("classesToColoc").is_none());
        assert_eq!(command["minColocArea"], json!(0.0), "the rest stays");
        let module = &raw["modules"][0];
        assert_eq!(module["steps"][0]["command"]["objects"], json!("UNSET"));
        assert_eq!(module["steps"][0]["command"]["overlapping"], json!([]));
        assert_eq!(module["parameters"][0]["bindings"][0]["param"], json!("overlapping"));
    }

    #[test]
    fn annotated_objects_lose_their_old_relation_fields() {
        let mut raw = json!({ "images": { "list": { "a.tif": { "series": { "0": { "objects": [
            { "id": 1, "parentId": null, "children": [], "area": 4 }
        ] } } } } } });
        migrate_from_v3_to_v4(&mut raw);
        assert_eq!(
            raw["images"]["list"]["a.tif"]["series"]["0"]["objects"][0],
            json!({ "id": 1, "area": 4 })
        );
    }

    #[test]
    fn an_already_migrated_document_is_left_as_it_is() {
        let classification = migrated(json!([class(json!(0)), class(json!(2))]));
        let ids: Vec<u32> = classification.classes().iter().map(|c| c.id).collect();
        assert_eq!(ids, [0, 2]);
    }

    #[test]
    fn segmentation_class_settings_are_renamed_in_pipelines_and_modules() {
        let mut raw = json!({
            "pipelines": [{ "steps": [
                { "command": { "type": "THRESHOLD", "thresholds": [
                    { "minThreshold": 1.0, "objectClassId": 2 }
                ]}},
                { "command": { "type": "CELLPOSE", "objectClassId": 1 }},
                { "module": { "key": "G", "values": {} }}
            ]}],
            "modules": [{ "key": "G", "steps": [
                { "id": 1, "command": { "type": "STARDIST", "objectClassId": 3 }}
            ], "parameters": []}]
        });
        migrate_from_v3_to_v4(&mut raw);
        let steps = &raw["pipelines"][0]["steps"];
        assert_eq!(
            steps[0]["command"]["thresholds"][0],
            json!({ "minThreshold": 1.0, "segmentationClass": 2 })
        );
        assert_eq!(
            steps[1]["command"],
            json!({ "type": "CELLPOSE", "segmentationClass": 1 })
        );
        assert_eq!(
            raw["modules"][0]["steps"][0]["command"],
            json!({ "type": "STARDIST", "segmentationClass": 3 })
        );
    }

    #[test]
    fn the_pixel_classifiers_model_class_becomes_model_class() {
        let mut raw = json!({ "pipelines": [{ "steps": [
            { "command": { "type": "PIXEL_CLASSIFIER", "segmentationMapping": [
                { "segmentationClass": 5, "objectClassId": 41 }
            ]}}
        ]}]});
        migrate_from_v3_to_v4(&mut raw);
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["segmentationMapping"][0],
            json!({ "modelClass": 5, "segmentationClass": 41 })
        );
    }

    #[test]
    fn extract_objects_mappings_are_left_alone() {
        let mapping = json!({ "segmentationClass": 1, "objectClass": { "GLOBAL": 1 } });
        let mut raw = json!({ "pipelines": [{ "steps": [
            { "command": { "type": "EXTRACT_OBJECTS", "classMapping": [mapping.clone()] }}
        ]}]});
        migrate_from_v3_to_v4(&mut raw);
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["classMapping"][0],
            mapping
        );
    }

    #[test]
    fn module_bindings_follow_the_renamed_settings() {
        let mut raw = json!({ "modules": [{
            "key": "G",
            "steps": [
                { "id": 1, "command": { "type": "THRESHOLD", "thresholds": [] }},
                { "id": 2, "command": { "type": "PIXEL_CLASSIFIER", "segmentationMapping": [] }},
                { "id": 3, "command": { "type": "EXTRACT_OBJECTS", "classMapping": [] }}
            ],
            "parameters": [{ "bindings": [
                { "step": 1, "param": "thresholds.0.object_class_id" },
                { "step": 2, "param": "segmentation_mapping.0.segmentation_class" },
                { "step": 2, "param": "segmentation_mapping.0.object_class_id" },
                { "step": 3, "param": "class_mapping.0.segmentation_class" },
                { "step": 3, "param": "class_mapping.0.object_class" }
            ]}]
        }]});
        migrate_from_v3_to_v4(&mut raw);
        let params: Vec<&str> = raw["modules"][0]["parameters"][0]["bindings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["param"].as_str().unwrap())
            .collect();
        assert_eq!(
            params,
            [
                "thresholds.0.segmentation_class",
                "segmentation_mapping.0.model_class",
                "segmentation_mapping.0.segmentation_class",
                "class_mapping.0.segmentation_class",
                "class_mapping.0.object_class",
            ]
        );
    }
}
