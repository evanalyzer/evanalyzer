//! Where this machine keeps templates.

use crate::workspace::settings::get_user_folder;
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
    user_templates_folder_in(&get_user_folder())
}

/// The templates folder inside `user_folder`, created if missing.
pub fn user_templates_folder_in(user_folder: &Path) -> PathBuf {
    let folder = user_folder.join("templates");
    let _ = std::fs::create_dir_all(&folder);
    folder
}
