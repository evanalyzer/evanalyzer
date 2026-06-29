use schemars::schema_for;
use std::path::Path;

fn write_schema<T: schemars::JsonSchema>(out_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let schema = schema_for!(T);
    let schema_json = serde_json::to_string_pretty(&schema)?;
    let out_path = Path::new("../../docs").join(out_file);
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    std::fs::write(out_path, schema_json)?;
    Ok(())
}

/// Generates `docs/project.schema.json` for `ProjectSettings`, the full project
/// file format (`.json` projects saved from the GUI/CLI).
pub fn generate_project_struct_schema() -> Result<(), Box<dyn std::error::Error>> {
    write_schema::<crate::modules::project_settings::ProjectSettings>("project.schema.json")
}

/// Generates `docs/project_template.schema.json` for `ProjectTemplate`
/// (`.evapt` files) and `docs/pipeline_template.schema.json` for
/// `PipelineTemplate` (`.evapipe` files). These are deliberately separate from
/// `ProjectSettings` - templates only carry a subset of a full project (no
/// `images`, and `meta` instead of `metadata`) - so validating a template
/// against `project.schema.json` will always fail even when the template
/// itself is correct.
pub fn generate_template_schemas() -> Result<(), Box<dyn std::error::Error>> {
    write_schema::<crate::modules::templates::ProjectTemplate>("project_template.schema.json")?;
    write_schema::<crate::modules::templates::PipelineTemplate>("pipeline_template.schema.json")?;
    Ok(())
}
