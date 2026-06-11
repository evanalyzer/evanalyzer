use evanalyzer_cfg::settings::templates::{PipelineTemplate, ProjectTemplate};
use evanalyzer_cfg::{PIPELINE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use std::path::{Path, PathBuf};

/// Returns the directory where templates shipped with the application are stored.
///
/// This is the `templates` subfolder next to the application binary.
pub fn get_app_templates_folder() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(std::env::temp_dir);
    exe_dir.join("templates")
}

/// Returns the directory where user created pipeline and project templates are stored.
///
/// The folder (and its parents) is created if it does not exist yet.
pub fn get_user_templates_folder() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(std::env::temp_dir);
    let folder = base.join("evanalyzer").join("templates");
    let _ = std::fs::create_dir_all(&folder);
    folder
}

/// Loads all `PipelineTemplate`s found in the user and app templates folders.
///
/// Files are matched by the [`PIPELINE_EXTENSIONS`] extension. The returned
/// templates are paired with the path they were loaded from.
pub fn load_pipeline_templates() -> Vec<(PathBuf, PipelineTemplate)> {
    let mut templates = Vec::new();
    for folder in [get_user_templates_folder(), get_app_templates_folder()] {
        load_templates_from_folder(&folder, PIPELINE_EXTENSIONS, &mut templates);
    }
    templates
}

/// Loads all `ProjectTemplate`s found in the user and app templates folders.
///
/// Files are matched by the [`PROJECT_FILE_TEMPLATE_EXTENSIONS`] extension. The
/// returned templates are paired with the path they were loaded from.
pub fn load_project_templates() -> Vec<(PathBuf, ProjectTemplate)> {
    let mut templates = Vec::new();
    for folder in [get_user_templates_folder(), get_app_templates_folder()] {
        load_templates_from_folder(&folder, PROJECT_FILE_TEMPLATE_EXTENSIONS, &mut templates);
    }
    templates
}

fn load_templates_from_folder<T: serde::de::DeserializeOwned>(
    folder: &Path,
    extension: &str,
    out: &mut Vec<(PathBuf, T)>,
) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let Ok(data) = std::fs::read_to_string(&path) else {
            continue;
        };
        match serde_json::from_str::<T>(&data) {
            Ok(template) => out.push((path, template)),
            Err(e) => log::warn!("Failed to load template {}: {}", path.display(), e),
        }
    }
}
