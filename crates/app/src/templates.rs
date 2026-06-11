use std::path::PathBuf;

pub fn get_app_templates_folder() {}

/// Returns the directory where user created pipeline and project templates are stored.
///
/// The folder (and its parents) is created if it does not exist yet.
pub fn get_user_templates_folder() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(std::env::temp_dir);
    let folder = base.join("evanalyzer").join("templates");
    let _ = std::fs::create_dir_all(&folder);
    folder
}

pub fn load_pipeline_templates() {}

pub fn load_project_templates() {}
