use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How many servers [`AppSettings::recent_servers`] keeps.
const MAX_RECENT_SERVERS: usize = 10;
/// Projects the start page lists at most.
const MAX_RECENT_PROJECTS: usize = 12;

/// Persisted, per-user application preferences (as opposed to project
/// settings, which travel with the project file).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    /// The colors of the GUI.
    #[serde(default)]
    pub color_scheme: ColorScheme,

    /// What settings saved before there were color schemes had instead of
    /// [`Self::color_scheme`]: turned into it on loading
    /// ([`AppSettings::migrate`]), never saved.
    #[serde(default, rename = "darkMode", skip_serializing)]
    pub(crate) legacy_dark_mode: Option<bool>,

    /// Pipeline focus mode: selecting a pipeline shows only its image
    /// channel and object classes.
    #[serde(default)]
    pub pipeline_focus_mode: bool,

    /// Show every step's advanced settings without expanding them.
    #[serde(default)]
    pub always_show_advanced_settings: bool,

    /// Debug mode: the pipeline editor shows how components are wired
    /// inside - what a parameter sets, which parameter sets a setting.
    #[serde(default)]
    pub debug_mode: bool,

    /// The image histogram's counts on a linear scale (default: log).
    #[serde(default)]
    pub histogram_linear: bool,

    /// The results' split view (results left, the image right; see
    /// [`ResultsSplitSettings`]).
    #[serde(default)]
    pub results_split: ResultsSplitSettings,

    /// Servers connected to from the GUI, most recent first - see
    /// [`AppSettings::remember_server`]. Kept on this computer only.
    #[serde(default)]
    pub recent_servers: Vec<RecentServer>,

    /// Projects opened or saved, most recent first - the start page's list
    /// (see [`AppSettings::remember_project`]).
    #[serde(default)]
    pub recent_projects: Vec<RecentProject>,

    /// At startup the last project opens right away instead of the start
    /// page.
    #[serde(default)]
    pub open_last_project: bool,

    #[serde(default)]
    pub author: Option<UserInformation>,

    /// How much memory the image viewport may spend on smoother pan/zoom.
    /// About this computer's RAM, so the GUI keeps it in this computer's
    /// settings even when connected to a server.
    #[serde(default)]
    pub viewport: ViewportMemorySettings,
}

/// The results' split view: on or off, how wide the results are, and
/// whether the image follows the selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ResultsSplitSettings {
    pub on: bool,
    /// Width of the results pane in logical pixels.
    pub width_px: f32,
    /// A selected object is shown on the image (else only selected).
    pub follow: bool,
    /// The image controls (channels, histogram, overlay) open over the
    /// image.
    pub image_panel: bool,
}

impl Default for ResultsSplitSettings {
    fn default() -> Self {
        Self {
            on: false,
            width_px: 640.0,
            image_panel: false,
            follow: true,
        }
    }
}

/// The colors of the GUI: light, dark, or a softer dark (`Dim`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColorScheme {
    #[default]
    Light,
    Dark,
    Dim,
}

impl AppSettings {
    /// Brings settings saved by an older version up to date: `darkMode`
    /// becomes [`ColorScheme::Dark`] (or stays light).
    fn migrate(mut self) -> Self {
        if let Some(dark) = self.legacy_dark_mode.take()
            && dark
        {
            self.color_scheme = ColorScheme::Dark;
        }
        self
    }
}

/// Memory/smoothness trade-offs of the image viewport - see
/// [`AppSettings::viewport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ViewportMemorySettings {
    /// Keep earlier high-resolution renders around (up to
    /// [`Self::tile_history_budget_mb`]) so panning back or zooming shows
    /// them instead of the blurry whole-image overview. Off: only the
    /// current render and the overview are kept.
    pub tile_history_enabled: bool,

    /// RAM the kept renders may use together, in MB.
    pub tile_history_budget_mb: u32,

    /// How far, in screen pixels, the object overlay is drawn past each
    /// viewport edge, so panning doesn't uncover an empty border right
    /// away. 0 keeps the overlay exactly viewport-sized (least RAM).
    pub object_overlay_margin_px: u32,

    /// Size both caches from this computer's RAM (see
    /// [`ViewportMemorySettings::auto_cache_mb`]) instead of
    /// [`Self::tile_cache_mb`]/[`Self::overview_cache_mb`].
    pub cache_sizes_auto: bool,

    /// Most RAM, in MB, the image tiles read for the viewport may keep, so
    /// panning back or switching back doesn't read them from disk again.
    pub tile_cache_mb: u32,

    /// Most RAM, in MB, the whole-image overviews may keep (one per image
    /// and channel combination recently shown).
    pub overview_cache_mb: u32,

    /// Empty both caches when an analysis starts on this computer, so it
    /// gets that RAM; they fill up again as the viewport is used.
    pub free_caches_on_analysis: bool,

    /// While an analysis runs, pipelines can still be edited and the image
    /// viewed and previewed - its progress is a card over the image. Off: a
    /// dialog blocks the window until the analysis is done (for a computer
    /// whose RAM the analysis needs all of).
    pub edit_during_analysis: bool,

    /// The objects open as a table at the bottom, not as a list in the
    /// right panel (its ⋯ still switches).
    pub objects_as_table: bool,

    /// The card of the selected object next to it on the image.
    pub object_card: bool,

    /// An image opened the first time (no scaling in the project yet) is
    /// auto-scaled, every channel; off: the full value range. The scaling
    /// is kept in the project either way.
    pub auto_scale_new_images: bool,

    /// Channels without an emission wavelength get distinct colours when an
    /// image is opened (kept in the project as their wavelength); off:
    /// white.
    pub color_channels_without_wavelength: bool,

    /// No animated moves: the results' split view jumps to an object
    /// instead of flying there.
    pub reduce_motion: bool,

    /// A preview covers at most [`Self::preview_tiles`] tiles - or, with
    /// this, what the computer running it can take
    /// ([`Self::auto_preview_tiles`]).
    pub preview_tiles_auto: bool,
    pub preview_tiles: u32,

    /// How many tiles show their breakpoint image (the visible ones nearest
    /// the viewport centre) - [`Self::breakpoint_tiles`], or with this as
    /// many as this computer's RAM allows ([`Self::auto_breakpoint_tiles`]).
    /// Kept in RAM while a breakpoint is set (~190 MB per 4096 x 4096 tile).
    pub breakpoint_tiles_auto: bool,
    pub breakpoint_tiles: u32,
}

impl ViewportMemorySettings {
    pub const DEFAULT_TILE_HISTORY_BUDGET_MB: u32 = 512;
    pub const DEFAULT_OBJECT_OVERLAY_MARGIN_PX: u32 = 256;
    pub const DEFAULT_TILE_CACHE_MB: u32 = 1024;
    pub const DEFAULT_OVERVIEW_CACHE_MB: u32 = 256;
    /// Below this a cache can't hold even one typical tile/overview.
    pub const MIN_CACHE_MB: u32 = 64;
    /// Share of the RAM the automatic tile cache takes.
    const AUTO_TILE_CACHE_RAM_PERCENT: u64 = 10;
    const AUTO_TILE_CACHE_MIN_MB: u32 = 128;
    const AUTO_OVERVIEW_CACHE_DIVISOR: u32 = 4;

    /// Cache sizes (tile, overview) in MB for a computer with
    /// `total_ram_bytes`: 10 % of the RAM for tiles, between 128 MB and the
    /// default 1 GB, and a quarter of that for overviews, between 64 MB and
    /// the default 256 MB - so from 16 GB RAM up it's the defaults, and a
    /// small machine keeps most of its RAM.
    pub fn auto_cache_mb(total_ram_bytes: u64) -> (u32, u32) {
        let total_mb = total_ram_bytes / (1024 * 1024);
        let tile = (total_mb * Self::AUTO_TILE_CACHE_RAM_PERCENT / 100).clamp(
            Self::AUTO_TILE_CACHE_MIN_MB as u64,
            Self::DEFAULT_TILE_CACHE_MB as u64,
        ) as u32;
        let overview = (tile / Self::AUTO_OVERVIEW_CACHE_DIVISOR)
            .clamp(Self::MIN_CACHE_MB, Self::DEFAULT_OVERVIEW_CACHE_MB);
        (tile, overview)
    }

    /// Preview tiles a computer with `total_ram_bytes` takes: one per 4 GB
    /// (16 GB: 4), between 1 and 32.
    pub fn auto_preview_tiles(total_ram_bytes: u64) -> u32 {
        ((total_ram_bytes / (4 << 30)) as u32).clamp(1, 32)
    }

    /// Breakpoint tiles a computer with `total_ram_bytes` keeps: about 5 %
    /// of its RAM at ~250 MB a tile (the images and their render) - 16 GB:
    /// 3, 8 GB: 1, between 1 and 16.
    pub fn auto_breakpoint_tiles(total_ram_bytes: u64) -> u32 {
        ((total_ram_bytes / 20 / (250 << 20)) as u32).clamp(1, 16)
    }

    /// The preview's tile limit to ask for: `None` lets the computer that
    /// runs it decide (auto).
    pub fn requested_preview_tiles(&self) -> Option<u32> {
        (!self.preview_tiles_auto).then_some(self.preview_tiles.max(1))
    }

    /// The breakpoint tiles in effect on a computer with `total_ram_bytes`.
    pub fn effective_breakpoint_tiles(&self, total_ram_bytes: u64) -> u32 {
        if self.breakpoint_tiles_auto {
            Self::auto_breakpoint_tiles(total_ram_bytes)
        } else {
            self.breakpoint_tiles.max(1)
        }
    }

    /// The cache sizes (tile, overview) in MB in effect on a computer with
    /// `total_ram_bytes`.
    pub fn effective_cache_mb(&self, total_ram_bytes: u64) -> (u32, u32) {
        if self.cache_sizes_auto {
            Self::auto_cache_mb(total_ram_bytes)
        } else {
            (self.tile_cache_mb, self.overview_cache_mb)
        }
    }
}

impl Default for ViewportMemorySettings {
    fn default() -> Self {
        Self {
            tile_history_enabled: false,
            tile_history_budget_mb: Self::DEFAULT_TILE_HISTORY_BUDGET_MB,
            object_overlay_margin_px: Self::DEFAULT_OBJECT_OVERLAY_MARGIN_PX,
            cache_sizes_auto: true,
            tile_cache_mb: Self::DEFAULT_TILE_CACHE_MB,
            overview_cache_mb: Self::DEFAULT_OVERVIEW_CACHE_MB,
            free_caches_on_analysis: true,
            edit_during_analysis: true,
            objects_as_table: false,
            object_card: false,
            auto_scale_new_images: true,
            color_channels_without_wavelength: false,
            reduce_motion: false,
            preview_tiles_auto: true,
            preview_tiles: 4,
            breakpoint_tiles_auto: true,
            breakpoint_tiles: 1,
        }
    }
}

/// Persisted, per-user application preferences (as opposed to project
/// settings, which travel with the project file).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserInformation {
    /// First name of the user
    pub full_name: String,

    /// Organization of the user
    pub organization: String,
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

/// A project opened or saved: what the start page offers again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentProject {
    /// The project file.
    pub path: PathBuf,
    pub name: String,
    #[serde(default)]
    pub images: u32,
    #[serde(default)]
    pub pipelines: u32,
    /// When it was last opened or saved (seconds since 1970, UTC).
    #[serde(default)]
    pub opened_at: i64,
}

impl AppSettings {
    /// Puts `project` first in [`Self::recent_projects`], replacing an
    /// older entry for the same file.
    pub fn remember_project(&mut self, project: RecentProject) {
        self.recent_projects.retain(|known| known.path != project.path);
        self.recent_projects.insert(0, project);
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
    }

    /// Takes `path` off [`Self::recent_projects`].
    pub fn forget_project(&mut self, path: &std::path::Path) {
        self.recent_projects.retain(|known| known.path != path);
    }

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
        .and_then(|data| serde_json::from_str::<AppSettings>(&data).ok())
        .map(AppSettings::migrate)
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

    fn project(path: &str) -> RecentProject {
        RecentProject {
            path: PathBuf::from(path),
            name: path.into(),
            images: 0,
            pipelines: 0,
            opened_at: 0,
        }
    }

    #[test]
    fn preview_and_breakpoint_tiles_follow_the_ram_unless_set() {
        let gb = |n: u64| n << 30;
        assert_eq!(ViewportMemorySettings::auto_preview_tiles(gb(16)), 4, "as before");
        assert_eq!(ViewportMemorySettings::auto_preview_tiles(gb(2)), 1);
        assert_eq!(ViewportMemorySettings::auto_preview_tiles(gb(512)), 32);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(8)), 1);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(16)), 3);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(512)), 16);

        let mut settings = ViewportMemorySettings::default();
        assert_eq!(settings.requested_preview_tiles(), None, "auto: the running computer decides");
        assert_eq!(settings.effective_breakpoint_tiles(gb(16)), 3);
        settings.preview_tiles_auto = false;
        settings.preview_tiles = 100;
        settings.breakpoint_tiles_auto = false;
        settings.breakpoint_tiles = 0;
        assert_eq!(settings.requested_preview_tiles(), Some(100));
        assert_eq!(settings.effective_breakpoint_tiles(gb(16)), 1, "at least one");
    }

    #[test]
    fn a_remembered_project_goes_first_once_and_the_list_stays_short() {
        let mut settings = AppSettings::default();
        settings.remember_project(project("/a.evaproj"));
        settings.remember_project(project("/b.evaproj"));
        settings.remember_project(project("/a.evaproj"));
        let paths: Vec<_> = settings.recent_projects.iter().map(|p| p.path.clone()).collect();
        assert_eq!(paths, [PathBuf::from("/a.evaproj"), PathBuf::from("/b.evaproj")]);
        for i in 0..20 {
            settings.remember_project(project(&format!("/{i}.evaproj")));
        }
        assert_eq!(settings.recent_projects.len(), MAX_RECENT_PROJECTS);
        settings.forget_project(std::path::Path::new("/19.evaproj"));
        assert_eq!(settings.recent_projects[0].path, PathBuf::from("/18.evaproj"));
    }

    #[test]
    fn settings_saved_before_recent_projects_still_load() {
        let settings: AppSettings = serde_json::from_str("{}").expect("old settings");
        assert!(settings.recent_projects.is_empty());
        assert!(!settings.open_last_project);
    }

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
        assert!(settings.recent_servers.is_empty());
        assert_eq!(settings.migrate().color_scheme, ColorScheme::Dark);
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
        assert_eq!(settings.color_scheme, ColorScheme::Light);
    }

    #[test]
    fn round_trips_through_json() {
        let settings = AppSettings {
            color_scheme: ColorScheme::Dim,
            ..Default::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.color_scheme, ColorScheme::Dim);
    }

    #[test]
    fn load_from_missing_file_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.json");

        let settings = load_app_settings_from(&path);

        assert_eq!(settings.color_scheme, ColorScheme::Light);
    }

    #[test]
    fn load_from_corrupt_json_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, "{ not valid json").unwrap();

        let settings = load_app_settings_from(&path);

        assert_eq!(settings.color_scheme, ColorScheme::Light);
    }

    #[test]
    fn save_then_load_round_trip_preserves_the_color_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let settings = AppSettings {
            color_scheme: ColorScheme::Dark,
            ..Default::default()
        };

        save_app_settings_to(&path, &settings).unwrap();
        let loaded = load_app_settings_from(&path);

        assert_eq!(loaded.color_scheme, ColorScheme::Dark);
    }

    #[test]
    fn dark_mode_saved_by_an_older_version_becomes_the_dark_scheme_and_is_not_saved_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{ "darkMode": true, "debugMode": true }"#).unwrap();

        let loaded = load_app_settings_from(&path);
        assert_eq!(loaded.color_scheme, ColorScheme::Dark);
        assert!(loaded.debug_mode);

        save_app_settings_to(&path, &loaded).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(!saved.contains("darkMode"), "{saved}");
        assert!(saved.contains(r#""colorScheme": "dark""#), "{saved}");

        std::fs::write(&path, r#"{ "darkMode": false }"#).unwrap();
        assert_eq!(
            load_app_settings_from(&path).color_scheme,
            ColorScheme::Light
        );
    }

    #[test]
    fn viewport_memory_settings_default_to_one_overview_and_fill_in_missing_fields() {
        let settings: AppSettings = serde_json::from_str(r#"{"darkMode":true}"#).unwrap();
        assert_eq!(settings.viewport, ViewportMemorySettings::default());
        assert!(!settings.viewport.tile_history_enabled);

        let settings: AppSettings =
            serde_json::from_str(r#"{"viewport":{"tileHistoryEnabled":true}}"#).unwrap();
        assert!(settings.viewport.tile_history_enabled);
        assert_eq!(
            settings.viewport.tile_history_budget_mb,
            ViewportMemorySettings::DEFAULT_TILE_HISTORY_BUDGET_MB
        );
        assert_eq!(
            settings.viewport.object_overlay_margin_px,
            ViewportMemorySettings::DEFAULT_OBJECT_OVERLAY_MARGIN_PX
        );
        // The cache limits were fixed before they became settings: older
        // files get exactly those values.
        assert_eq!(settings.viewport.tile_cache_mb, 1024);
        assert_eq!(settings.viewport.overview_cache_mb, 256);
        assert!(settings.viewport.free_caches_on_analysis);
    }

    #[test]
    fn automatic_cache_sizes_follow_the_ram_within_their_limits() {
        const GB: u64 = 1024 * 1024 * 1024;
        let auto = ViewportMemorySettings::auto_cache_mb;
        assert_eq!(auto(2 * GB), (204, 64));
        assert_eq!(auto(4 * GB), (409, 102));
        assert_eq!(auto(8 * GB), (819, 204));
        // From about 10 GB up: the fixed defaults.
        assert_eq!(auto(16 * GB), (1024, 256));
        assert_eq!(auto(128 * GB), (1024, 256));
        // A tiny or unreadable RAM size still gives usable caches.
        assert_eq!(auto(0), (128, 64));
    }

    #[test]
    fn manual_cache_sizes_apply_only_with_automatic_sizing_off() {
        let mut settings = ViewportMemorySettings {
            tile_cache_mb: 300,
            overview_cache_mb: 70,
            ..Default::default()
        };
        assert!(settings.cache_sizes_auto, "automatic by default");
        assert_eq!(settings.effective_cache_mb(4 << 30), (409, 102));
        settings.cache_sizes_auto = false;
        assert_eq!(settings.effective_cache_mb(4 << 30), (300, 70));
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
        assert_eq!(loaded.color_scheme, ColorScheme::Dark);
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
