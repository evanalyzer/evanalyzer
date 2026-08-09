use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::templates::PipelineTemplate;

/// Gets a user prompt and the project's schema as input and the predicted pipeline as output.
pub fn generate_pipeline(
    user_prompt: &str,
    project_schema_file: &str,
) -> Result<PipelineTemplate, InternalErrors> {
    let _ = (user_prompt, project_schema_file);

    let schema = schemars::schema_for!(PipelineTemplate);
    let schema_string = serde_json::to_string_pretty(&schema).unwrap();
    println!("Generated JSON Schema for Local LLM:\n{}", schema_string);

    Err("Not implemented yet".into())
}
