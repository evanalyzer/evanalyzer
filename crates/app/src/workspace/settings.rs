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

    /// Servers connected to from the GUI, most recent first - see
    /// [`AppSettings::remember_server`]. Kept on this computer only.
    #[serde(default)]
    pub recent_servers: Vec<RecentServer>,
}

/// A server the user connected to: what the connect dialog offers again.
/// Never the password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentServer {
    pub url: String,
    pub user: String,
    /// The certificate fingerprint the user confirmed for this server - a
    /// different one later means the certificate changed. `None` for
    /// `ws://` and for certificates signed by a public authority.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

/// How many servers [`AppSettings::recent_servers`] keeps.
const MAX_RECENT_SERVERS: usize = 10;

impl AppSettings {
    /// Puts `server` first in [`Self::recent_servers`], replacing an older
    /// entry for the same address and user.
    pub fn remember_server(&mut self, server: RecentServer) {
        self.recent_servers
            .retain(|known| !(known.url == server.url && known.user == server.user));
        self.recent_servers.insert(0, server);
        self.recent_servers.truncate(MAX_RECENT_SERVERS);
    }

    /// The fingerprint confirmed for `url`, if any (by any user: it's the
    /// server's certificate, not the user's).
    pub fn known_fingerprint(&self, url: &str) -> Option<&str> {
        self.recent_servers
            .iter()
            .find(|known| known.url == url && known.fingerprint.is_some())
            .and_then(|known| known.fingerprint.as_deref())
    }
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

    fn server(url: &str, user: &str, fingerprint: Option<&str>) -> RecentServer {
        RecentServer {
            url: url.into(),
            user: user.into(),
            fingerprint: fingerprint.map(str::to_string),
        }
    }

    #[test]
    fn the_latest_server_comes_first_once_and_the_list_stays_short() {
        let mut settings = AppSettings::default();
        settings.remember_server(server("wss://a", "alice", Some("AA")));
        settings.remember_server(server("wss://b", "alice", None));
        settings.remember_server(server("wss://a", "alice", Some("AA")));
        assert_eq!(
            settings.recent_servers,
            [
                server("wss://a", "alice", Some("AA")),
                server("wss://b", "alice", None)
            ]
        );
        // Another user on the same server is another entry, but the
        // server's fingerprint is known for both.
        settings.remember_server(server("wss://a", "bob", None));
        assert_eq!(settings.recent_servers.len(), 3);
        assert_eq!(settings.known_fingerprint("wss://a"), Some("AA"));
        assert_eq!(settings.known_fingerprint("wss://b"), None);

        for i in 0..20 {
            settings.remember_server(server(&format!("wss://{i}"), "x", None));
        }
        assert_eq!(settings.recent_servers.len(), MAX_RECENT_SERVERS);
        assert_eq!(settings.recent_servers[0].url, "wss://19");
    }

    #[test]
    fn settings_from_before_recent_servers_still_load() {
        let settings: AppSettings = serde_json::from_str(r#"{"darkMode":true}"#).unwrap();
        assert!(settings.dark_mode);
        assert!(settings.recent_servers.is_empty());
    }

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
