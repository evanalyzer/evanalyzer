use schemars::schema_for;
use std::path::Path;

pub fn generate_project_struct_schema() -> Result<(), Box<dyn std::error::Error>> {
    let schema = schema_for!(crate::modules::project_settings::ProjectSettings);
    let schema_json = serde_json::to_string_pretty(&schema)?;
    let out_path = Path::new("../../docs/project.schema.json");
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    std::fs::write(out_path, schema_json)?;
    Ok(())
}
