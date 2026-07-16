use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Persisted, per-user application preferences (as opposed to project
/// settings, which travel with the project file).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    #[serde(default)]
    pub dark_mode: bool,
}

/// Returns the application's per-user data directory (`<OS user data dir>/evanalyzer`),
/// where user settings, templates, and other per-user state are stored.
///
/// The folder (and its parents) is created if it does not exist yet.
pub fn get_user_folder() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(std::env::temp_dir);
    let folder = base.join("evanalyzer");
    let _ = std::fs::create_dir_all(&folder);
    folder
}

fn settings_file_path() -> std::path::PathBuf {
    get_user_folder().join("settings.json")
}

/// Loads the persisted app settings, falling back to defaults if the file
/// doesn't exist yet or fails to parse.
pub fn load_app_settings() -> AppSettings {
    std::fs::read_to_string(settings_file_path())
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

/// Persists the app settings, overwriting whatever was there before.
pub fn save_app_settings(settings: &AppSettings) {
    match serde_json::to_string_pretty(settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(settings_file_path(), json) {
                log::warn!("Failed to save app settings: {e}");
            }
        }
        Err(e) => log::warn!("Failed to serialize app settings: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_returns_defaults() {
        let settings = AppSettings::default();
        assert!(!settings.dark_mode);
    }

    #[test]
    fn round_trips_through_json() {
        let settings = AppSettings { dark_mode: true };
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: AppSettings = serde_json::from_str(&json).unwrap();
        assert!(parsed.dark_mode);
    }
}
