//! # settings_defaults
//!
//! Tests that the loader (serde) and the JSON schema (schemars) of the
//! generated pipeline command settings agree on which fields a project
//! file must contain: a field the schema does not list as `required` must
//! load when missing (filled with its default), a required one must fail.
//! A field with a default being optional is what lets a command gain a new
//! parameter without a schema-version bump - old files load, saving writes
//! the new parameter.
//!
//! The generated settings file can't carry inline tests, so they live here.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::settings::pipeline_command::{
    PipelineCommand, all_command_meta, default_command,
};
use serde_json::{Map, Value};

/// Every command's default, serialized - keyed by its `type` tag.
fn serialized_defaults() -> Vec<(String, Map<String, Value>)> {
    all_command_meta()
        .into_iter()
        .map(|meta| {
            let cmd = default_command(meta.id).expect("default command exists");
            let Value::Object(obj) = serde_json::to_value(&cmd).expect("serializes") else {
                panic!("command {} does not serialize to an object", meta.name);
            };
            let tag = obj["type"].as_str().expect("type tag").to_string();
            (tag, obj)
        })
        .collect()
}

/// For every command and every top-level settings field: leaving the field
/// out loads exactly when the schema does not mark it required.
#[test]
fn missing_field_loads_iff_schema_does_not_require_it() {
    let schema = serde_json::to_value(schemars::schema_for!(PipelineCommand)).unwrap();
    let defs = &schema["$defs"];
    let branches = schema["oneOf"]
        .as_array()
        .expect("PipelineCommand is a oneOf");
    let defaults = serialized_defaults();

    let mut mismatches = Vec::new();
    let mut checked_fields = 0;
    for branch in branches {
        let tag = branch["properties"]["type"]["const"]
            .as_str()
            .expect("tag const");
        let def_name = branch["$ref"]
            .as_str()
            .and_then(|r| r.strip_prefix("#/$defs/"))
            .unwrap_or_else(|| panic!("{tag}: branch without a $defs reference"));
        let def = &defs[def_name];
        let required: Vec<&str> = def["required"]
            .as_array()
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let (_, full) = defaults
            .iter()
            .find(|(t, _)| t == tag)
            .unwrap_or_else(|| panic!("{tag}: no default command with this tag"));

        let properties = def["properties"].as_object().cloned().unwrap_or_default();
        for field in properties.keys() {
            let mut partial = full.clone();
            assert!(
                partial.remove(field).is_some(),
                "{tag}: schema property {field:?} is not written on save"
            );
            let loads = serde_json::from_value::<PipelineCommand>(Value::Object(partial)).is_ok();
            let is_required = required.contains(&field.as_str());
            if loads == is_required {
                mismatches.push(format!(
                    "{tag}.{field}: schema required = {is_required}, loads when missing = {loads}"
                ));
            }
            checked_fields += 1;
        }
    }
    assert!(checked_fields > 100, "only {checked_fields} fields checked");
    assert!(
        mismatches.is_empty(),
        "loader and schema disagree:\n{}",
        mismatches.join("\n")
    );
}

/// The full default always loads back to itself (nothing written on save is
/// rejected on load).
#[test]
fn every_default_command_round_trips() {
    for (tag, obj) in serialized_defaults() {
        let cmd: PipelineCommand = serde_json::from_value(Value::Object(obj.clone()))
            .unwrap_or_else(|e| panic!("{tag}: default does not load: {e}"));
        assert_eq!(
            serde_json::to_value(&cmd).unwrap(),
            Value::Object(obj),
            "{tag}"
        );
    }
}

/// An old project file without a parameter added later (here Watershed
/// `smoothingSigma`, ClassifyObjects `originSegmentation`) loads with the
/// default, and saving writes the parameter back into the file.
#[test]
fn old_file_without_parameter_loads_with_default_and_saves_it() {
    for (tag, missing) in [
        ("WATERSHED", "smoothingSigma"),
        ("CLASSIFY_OBJECTS", "originSegmentation"),
    ] {
        let (_, full) = serialized_defaults()
            .into_iter()
            .find(|(t, _)| t == tag)
            .unwrap_or_else(|| panic!("no command {tag}"));
        let mut old = full.clone();
        assert!(
            old.remove(missing).is_some(),
            "{tag} has no field {missing}"
        );

        let cmd: PipelineCommand = serde_json::from_value(Value::Object(old))
            .unwrap_or_else(|e| panic!("{tag} without {missing} does not load: {e}"));
        let saved = serde_json::to_value(&cmd).unwrap();
        assert_eq!(
            saved.get(missing),
            full.get(missing),
            "{tag}: {missing} is not saved with its default"
        );
    }
}
