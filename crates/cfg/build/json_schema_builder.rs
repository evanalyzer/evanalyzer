//! # json_schema_builder
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use schemars::schema_for;
use std::path::Path;

fn write_schema<T: schemars::JsonSchema>(out_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut schema = schema_for!(T);
    // Maps in alphabetical order, so the schema files don't change when
    // fields or types are only reordered in code (serde_json's
    // `preserve_order` would otherwise keep declaration order). schemars
    // still puts its main keywords (title, type, properties, ...) first.
    if let Some(map) = schema.as_object_mut() {
        map.sort_keys();
        map.values_mut()
            .for_each(serde_json::Value::sort_all_objects);
    }
    let schema_json = serde_json::to_string_pretty(&schema)?;
    let out_path = Path::new("../../docs").join(out_file);
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    std::fs::write(out_path, schema_json)?;
    Ok(())
}

/// Generates `docs/project.schema.json` for `ProjectSettings`, the full project
/// file format (`.json` projects saved from the GUI/CLI).
pub fn generate_project_struct_schema() -> Result<(), Box<dyn std::error::Error>> {
    write_schema::<crate::settings::project_settings::ProjectSettings>("project.schema.json")
}

/// Generates `docs/project_template.schema.json` for `ProjectTemplate`
/// (`.evatemplate` files) and `docs/module_template.schema.json` for `ModuleTemplate` (`.evamodule` files). These are deliberately separate from
/// `ProjectSettings` - templates only carry a subset of a full project (no
/// `images`, and `meta` instead of `metadata`) - so validating a template
/// against `project.schema.json` will always fail even when the template
/// itself is correct.
pub fn generate_template_schemas() -> Result<(), Box<dyn std::error::Error>> {
    write_schema::<crate::settings::templates::ProjectTemplate>("project_template.schema.json")?;
    write_schema::<crate::settings::templates::ModuleTemplate>("module_template.schema.json")?;
    Ok(())
}

/// Generates `docs/ai_learning.schema.json` for `AiLearningSettings` - the
/// standalone `--settings` file `train-classifier` reads, and the copy
/// embedded in every saved `.evamodel` classifier.
pub fn generate_ai_learning_settings_schema() -> Result<(), Box<dyn std::error::Error>> {
    write_schema::<crate::settings::ai_learning_settings::AiLearningSettings>(
        "ai_learning.schema.json",
    )
}
