//! Migrates a [`ProjectSettings`](crate::settings::project_settings::ProjectSettings)
//! document from schema version 4 to version 5. (Templates aren't migrated -
//! they may break.)
//!
//! Version 5 names a segmentation class setting what it is:
//!
//! - `objectClassId` (Threshold entries, Cellpose, StarDist, U-Net, the pixel
//!   classifier's mapping) becomes `segmentationClass`.
//! - In the pixel classifier's mapping the model's class, so far
//!   `segmentationClass`, becomes `modelClass` - as in YOLO's mapping.
//! - A command group's parameter bindings name settings by their path
//!   (`thresholds.0.object_class_id`); they follow the same renames.

use serde_json::{Map, Value};

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

/// The bindings of a group definition, renamed - before its commands are.
fn rename_bindings(group: &mut Value) {
    let types: Vec<(Value, Option<String>)> = group
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
    let Some(Value::Array(parameters)) = group.get_mut("parameters") else {
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

pub fn migrate_from_v4_to_v5(raw: &mut Value) {
    if let Some(Value::Array(pipelines)) = raw.get_mut("pipelines") {
        for pipeline in pipelines {
            rename_commands(pipeline.get_mut("steps"));
        }
    }
    if let Some(Value::Array(groups)) = raw.get_mut("commandGroups") {
        for group in groups {
            rename_bindings(group);
            rename_commands(group.get_mut("steps"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn segmentation_class_settings_are_renamed_in_pipelines_and_groups() {
        let mut raw = json!({
            "pipelines": [{ "steps": [
                { "command": { "type": "THRESHOLD", "thresholds": [
                    { "minThreshold": 1.0, "objectClassId": 2 }
                ]}},
                { "command": { "type": "CELLPOSE", "objectClassId": 1 }},
                { "group": { "key": "G", "values": {} }}
            ]}],
            "commandGroups": [{ "key": "G", "steps": [
                { "id": 1, "command": { "type": "STARDIST", "objectClassId": 3 }}
            ], "parameters": []}]
        });
        migrate_from_v4_to_v5(&mut raw);
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
            raw["commandGroups"][0]["steps"][0]["command"],
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
        migrate_from_v4_to_v5(&mut raw);
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
        migrate_from_v4_to_v5(&mut raw);
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["classMapping"][0],
            mapping
        );
    }

    #[test]
    fn group_bindings_follow_the_renamed_settings() {
        let mut raw = json!({ "commandGroups": [{
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
        migrate_from_v4_to_v5(&mut raw);
        let params: Vec<&str> = raw["commandGroups"][0]["parameters"][0]["bindings"]
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
