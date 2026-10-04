use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Persisted, per-user application preferences (as opposed to project
/// settings, which travel with the project file).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    #[serde(default)]
    pub dark_mode: bool,

    /// Pipeline focus mode: selecting a pipeline shows only its image
    /// channel and object classes.
    #[serde(default)]
    pub pipeline_focus_mode: bool,

    /// Show every step's advanced settings without expanding them.
    #[serde(default)]
    pub always_show_advanced_settings: bool,
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

/// The user folder for an explicitly given `home` - a worker's `--home` -
/// at the place this platform's data folder has below a home folder, so a
/// user finds the same folder whether they work on the machine directly or
/// through a worker: `<home>/.local/share/evanalyzer` (Linux),
/// `<home>/Library/Application Support/evanalyzer` (macOS),
/// `<home>/AppData/Roaming/evanalyzer` (Windows).
pub fn user_folder_in(home: &Path) -> PathBuf {
    let data = if cfg!(windows) {
        home.join("AppData").join("Roaming")
    } else if cfg!(target_os = "macos") {
        home.join("Library").join("Application Support")
    } else {
        home.join(".local").join("share")
    };
    data.join("evanalyzer")
}

/// The settings file inside a user folder (`<user folder>/settings.json`).
pub fn settings_file_in(user_folder: &Path) -> PathBuf {
    user_folder.join("settings.json")
}

/// Loads the app settings from `path` - defaults if it doesn't exist yet or
/// fails to parse.
pub fn load_app_settings_from(path: &Path) -> AppSettings {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

/// Writes the app settings to `path`, overwriting whatever was there and
/// creating its folder if needed.
pub fn save_app_settings_to(path: &Path, settings: &AppSettings) -> std::io::Result<()> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_folder_of_an_explicit_home_follows_the_platform_layout() {
        let folder = user_folder_in(Path::new("/home/alice"));

        let expected = if cfg!(windows) {
            Path::new("/home/alice/AppData/Roaming/evanalyzer").to_path_buf()
        } else if cfg!(target_os = "macos") {
            Path::new("/home/alice/Library/Application Support/evanalyzer").to_path_buf()
        } else {
            Path::new("/home/alice/.local/share/evanalyzer").to_path_buf()
        };
        assert_eq!(folder, expected);
    }

    #[test]
    fn missing_file_returns_defaults() {
        let settings = AppSettings::default();
        assert!(!settings.dark_mode);
    }

    #[test]
    fn round_trips_through_json() {
        let settings = AppSettings {
            dark_mode: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: AppSettings = serde_json::from_str(&json).unwrap();
        assert!(parsed.dark_mode);
    }

    #[test]
    fn load_from_missing_file_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.json");

        let settings = load_app_settings_from(&path);

        assert!(!settings.dark_mode);
    }

    #[test]
    fn load_from_corrupt_json_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, "{ not valid json").unwrap();

        let settings = load_app_settings_from(&path);

        assert!(!settings.dark_mode);
    }

    #[test]
    fn save_then_load_round_trip_preserves_dark_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let settings = AppSettings {
            dark_mode: true,
            ..Default::default()
        };

        save_app_settings_to(&path, &settings).unwrap();
        let loaded = load_app_settings_from(&path);

        assert!(loaded.dark_mode);
    }

    #[test]
    fn pipeline_focus_mode_round_trips_and_defaults_to_off_for_older_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        save_app_settings_to(
            &path,
            &AppSettings {
                pipeline_focus_mode: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(load_app_settings_from(&path).pipeline_focus_mode);

        // A settings file written before the field existed.
        std::fs::write(&path, r#"{ "darkMode": true }"#).unwrap();
        let loaded = load_app_settings_from(&path);
        assert!(loaded.dark_mode);
        assert!(!loaded.pipeline_focus_mode);
        assert!(!loaded.always_show_advanced_settings);
    }

    #[test]
    fn get_user_folder_creates_dir_and_ends_with_evanalyzer() {
        let folder = get_user_folder();

        assert!(folder.is_dir());
        assert_eq!(folder.file_name().unwrap(), "evanalyzer");
    }
}
