//! # settings
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::recent::{RecentProject, RecentServer};
use crate::results::SpeedUnit;
use serde::{Deserialize, Serialize};
use std::path::Path;

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

    /// Debug mode: the pipeline editor shows how modules are wired
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

    /// Which panels over the image are shown and folded (see
    /// [`PanelSettings`]).
    #[serde(default)]
    pub panels: PanelSettings,

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

    /// Folders the user added to the file dialog's places (beside home,
    /// drives, …) - a click there opens them. Of the computer whose files
    /// the dialog shows (a server's on a server).
    #[serde(default)]
    pub file_places: Vec<std::path::PathBuf>,

    /// How much memory the image viewport may spend on smoother pan/zoom.
    /// About this computer's RAM, so the GUI keeps it in this computer's
    /// settings even when connected to a server.
    #[serde(default)]
    pub viewport: ViewportMemorySettings,

    /// How the AI training panel's prediction overlay looks (whether it is
    /// on isn't kept: it starts off).
    #[serde(default)]
    pub prediction_overlay: PredictionOverlayPrefs,

    /// How much of this computer EVAnalyzer may use (Settings ›
    /// Performance) - read at startup; `--max-threads` / `--max-ram` win.
    #[serde(default)]
    pub performance: PerformanceSettings,

    /// How big the GUI draws buttons and text (EVA › View › Size) - this
    /// computer's, as it is about its screen, so the GUI keeps it in this
    /// computer's settings even when connected to a server. The screen's
    /// own scale (Windows display scaling, …) is not used.
    #[serde(default)]
    pub interface_size: InterfaceSize,
    /// EVA › View › "Blur behind panels": panels and dialogs blur what is
    /// behind them. Off by default (it costs GPU/CPU while the image
    /// moves); only drawn by the Skia / FemtoVG renderers, not the
    /// software one (the Windows build).
    #[serde(default)]
    pub panel_blur: bool,
}

/// The GUI's size in percent: one of [`InterfaceSize::PRESETS`]. A value
/// edited by hand into the file is taken as the nearest preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InterfaceSize(u16);

impl Default for InterfaceSize {
    fn default() -> Self {
        Self(100)
    }
}

impl InterfaceSize {
    /// The sizes the GUI offers, in percent.
    pub const PRESETS: [u16; 5] = [100, 125, 150, 175, 200];

    /// The preset at `index` in [`Self::PRESETS`] (clamped).
    pub fn from_index(index: usize) -> Self {
        Self(Self::PRESETS[index.min(Self::PRESETS.len() - 1)])
    }

    /// Which of [`Self::PRESETS`] this is - the nearest one for a value
    /// that isn't one.
    pub fn index(self) -> usize {
        Self::PRESETS
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.abs_diff(self.0))
            .map_or(0, |(i, _)| i)
    }

    /// The preset in percent.
    pub fn percent(self) -> u16 {
        Self::PRESETS[self.index()]
    }

    /// The GUI's scale factor: 1.0 for 100 %.
    pub fn factor(self) -> f32 {
        f32::from(self.percent()) / 100.0
    }
}

/// How much of this computer EVAnalyzer may use - for a computer that also
/// does other work during a long analysis. `None`: as much as there is.
/// Read at startup: the thread pools and the AI engine are set up once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PerformanceSettings {
    /// Most CPU cores.
    pub max_threads: Option<u32>,
    /// Most RAM in MB EVAnalyzer plans with (caches, threads, results
    /// database) - not a hard limit on every allocation.
    pub max_ram_mb: Option<u32>,
}

impl PerformanceSettings {
    /// These limits, with the command line's (`--max-threads`, `--max-ram`)
    /// in place of the saved ones where given.
    pub fn with_overrides(
        &self,
        max_threads: Option<usize>,
        max_ram_mb: Option<u64>,
    ) -> evanalyzer_core::memory::ResourceLimits {
        evanalyzer_core::memory::ResourceLimits {
            max_threads: max_threads.or(self.max_threads.map(|n| n as usize)),
            max_ram_mb: max_ram_mb.or(self.max_ram_mb.map(u64::from)),
        }
    }
}

/// The prediction overlay's look, kept between sessions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PredictionOverlayPrefs {
    /// 0..=1, of a pixel the model is sure of.
    pub opacity: f32,
    /// The background class in its colour too, else left clear.
    pub show_background: bool,
    /// The class and its probability under the cursor (the pixel info).
    pub show_values: bool,
}

impl Default for PredictionOverlayPrefs {
    fn default() -> Self {
        Self {
            opacity: 0.7,
            show_background: true,
            show_values: true,
        }
    }
}

/// The panels over the image as the user left them: shown (EVA › View) and
/// folded to their head, the bar at the bottom of the image folded, the
/// pipelines list above the steps open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PanelSettings {
    pub steps_open: bool,
    pub classes_open: bool,
    pub image_open: bool,
    pub objects_open: bool,
    pub measurements_open: bool,
    pub classes_folded: bool,
    pub image_folded: bool,
    pub objects_folded: bool,
    pub measurements_folded: bool,
    pub image_bar_folded: bool,
    pub pipelines_list_open: bool,
    pub navigator: NavigatorMode,
    /// The step rows' values on a second line under the name (EVA › View).
    pub step_values_below: bool,
}

/// When the image's navigator (the mini-map) shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NavigatorMode {
    /// Only while part of the image is outside the view (zoomed in).
    #[default]
    Auto,
    Always,
    Never,
}

impl Default for PanelSettings {
    fn default() -> Self {
        Self {
            steps_open: true,
            classes_open: true,
            image_open: true,
            objects_open: true,
            measurements_open: true,
            classes_folded: false,
            image_folded: false,
            objects_folded: false,
            measurements_folded: true,
            image_bar_folded: false,
            pipelines_list_open: true,
            navigator: NavigatorMode::Auto,
            step_values_below: false,
        }
    }
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
    /// The selected object's facts folded to their first line.
    pub card_folded: bool,
    /// Tracks: how much of each track's path is drawn.
    pub trail: TrailMode,
    /// `TrailMode::Frames`: this many time points back.
    pub trail_frames: u32,
    /// Tracks: only the selected track's path.
    pub only_selected_track: bool,
    /// Tracks: paths in each track's own color (else its class's).
    pub color_by_track: bool,
    pub speed_unit: SpeedUnit,
}

impl Default for ResultsSplitSettings {
    fn default() -> Self {
        Self {
            on: false,
            width_px: 640.0,
            image_panel: false,
            card_folded: false,
            follow: true,
            trail: TrailMode::Frames,
            trail_frames: 10,
            only_selected_track: false,
            color_by_track: true,
            speed_unit: SpeedUnit::PxPerFrame,
        }
    }
}

/// How much of a track's path is drawn on the image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrailMode {
    Off,
    /// The last `trail_frames` time points.
    #[default]
    Frames,
    Whole,
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

    /// Hold both caches at their smallest size while an analysis runs on
    /// this computer, so it gets that RAM; they grow again when it ends.
    pub free_caches_on_analysis: bool,

    /// The objects open as a table at the bottom, not as a list in the
    /// right panel (its ⋯ still switches).
    pub objects_as_table: bool,

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

    /// The window is drawn by Slint's own software renderer instead of the
    /// graphics card: ~60 MB less RAM, a much faster start and no graphics
    /// driver involved (virtual machines, remote desktop, old computers) -
    /// panning large images costs more CPU. On by default since 2026-11-19
    /// (a settings file keeps its value); off: FemtoVG on wgpu, Direct3D 12
    /// on Windows. Read at startup.
    pub software_renderer: bool,

    /// RAM the undo history may use - [`Self::undo_history_mb`], or with
    /// this what this computer can spare next to a full analysis
    /// ([`Self::auto_undo_history_mb`]). The oldest steps go first, but at
    /// least [`Self::MIN_UNDO_STEPS`] are always kept.
    pub undo_history_auto: bool,
    pub undo_history_mb: u32,
}

impl ViewportMemorySettings {
    pub const DEFAULT_TILE_HISTORY_BUDGET_MB: u32 = 512;
    pub const DEFAULT_OBJECT_OVERLAY_MARGIN_PX: u32 = 256;
    pub const DEFAULT_TILE_CACHE_MB: u32 = evanalyzer_core::memory::viewer::TILE_CACHE_MAX_MB;
    pub const DEFAULT_OVERVIEW_CACHE_MB: u32 =
        evanalyzer_core::memory::viewer::OVERVIEW_CACHE_MAX_MB;
    /// Below this a cache can't hold even one typical tile/overview.
    pub const MIN_CACHE_MB: u32 = evanalyzer_core::memory::viewer::CACHE_MIN_MB;
    pub const DEFAULT_UNDO_HISTORY_MB: u32 = 256;
    /// Undo steps kept whatever they cost.
    pub const MIN_UNDO_STEPS: usize = 10;

    /// Cache sizes (tile, overview) in MB for a computer with
    /// `total_ram_bytes` - see [`evanalyzer_core::memory::viewer::cache_mb`].
    pub fn auto_cache_mb(total_ram_bytes: u64) -> (u32, u32) {
        evanalyzer_core::memory::viewer::cache_mb(total_ram_bytes)
    }

    /// Preview tiles a computer with `total_ram_bytes` takes - see
    /// [`evanalyzer_core::memory::viewer::preview_tiles`].
    pub fn auto_preview_tiles(total_ram_bytes: u64) -> u32 {
        evanalyzer_core::memory::viewer::preview_tiles(total_ram_bytes)
    }

    /// Breakpoint tiles a computer with `total_ram_bytes` keeps - see
    /// [`evanalyzer_core::memory::viewer::breakpoint_tiles`].
    pub fn auto_breakpoint_tiles(total_ram_bytes: u64) -> u32 {
        evanalyzer_core::memory::viewer::breakpoint_tiles(total_ram_bytes)
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

    /// The undo history's RAM in MB on a computer with `total_ram_bytes`
    /// and `cores`, with the caches in effect - see
    /// [`evanalyzer_core::memory::viewer::undo_history_mb`].
    pub fn auto_undo_history_mb(&self, total_ram_bytes: u64, cores: usize) -> u32 {
        let (tile_cache, overview_cache) = self.effective_cache_mb(total_ram_bytes);
        evanalyzer_core::memory::viewer::undo_history_mb(
            total_ram_bytes,
            cores,
            tile_cache,
            overview_cache,
        )
    }

    /// The undo history's RAM in MB in effect on a computer with
    /// `total_ram_bytes` and `cores`.
    pub fn effective_undo_history_mb(&self, total_ram_bytes: u64, cores: usize) -> u32 {
        if self.undo_history_auto {
            self.auto_undo_history_mb(total_ram_bytes, cores)
        } else {
            self.undo_history_mb.max(1)
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
            objects_as_table: false,
            auto_scale_new_images: true,
            color_channels_without_wavelength: false,
            reduce_motion: false,
            preview_tiles_auto: true,
            preview_tiles: 4,
            breakpoint_tiles_auto: true,
            breakpoint_tiles: 1,
            software_renderer: true,
            undo_history_auto: true,
            undo_history_mb: Self::DEFAULT_UNDO_HISTORY_MB,
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

    /// The command line wins over the saved limits; what it doesn't give
    /// stays as saved.
    #[test]
    fn the_command_line_wins_over_the_saved_performance_limits() {
        let saved = PerformanceSettings {
            max_threads: Some(4),
            max_ram_mb: Some(8192),
        };
        let limits = saved.with_overrides(None, None);
        assert_eq!(
            (limits.max_threads, limits.max_ram_mb),
            (Some(4), Some(8192))
        );
        let limits = saved.with_overrides(Some(2), None);
        assert_eq!(
            (limits.max_threads, limits.max_ram_mb),
            (Some(2), Some(8192))
        );
        let limits = PerformanceSettings::default().with_overrides(None, Some(1024));
        assert_eq!((limits.max_threads, limits.max_ram_mb), (None, Some(1024)));
    }

    /// Settings saved before there were limits load without them.
    #[test]
    fn settings_without_performance_limits_load() {
        let settings: AppSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings.performance, PerformanceSettings::default());
        let json = serde_json::to_string(&AppSettings {
            performance: PerformanceSettings {
                max_threads: Some(6),
                max_ram_mb: None,
            },
            ..Default::default()
        })
        .unwrap();
        let back: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.performance.max_threads, Some(6));
    }

    /// The automatic undo history leaves a full analysis on every core its
    /// RAM: 10 % of what is left, between 32 MB and 1 GB.
    #[test]
    fn the_automatic_undo_history_leaves_room_for_a_full_analysis() {
        const GB: u64 = 1024 * 1024 * 1024;
        let auto = ViewportMemorySettings::default();
        // 8 GB, 4 cores: 3 workers (1548 MB), caches 1023 MB, 512 MB app.
        assert_eq!(auto.auto_undo_history_mb(8 * GB, 4), 510);
        assert_eq!(auto.auto_undo_history_mb(16 * GB, 22), 375);
        assert_eq!(auto.auto_undo_history_mb(4 * GB, 4), 152);
        // Nothing left over: the least.
        assert_eq!(auto.auto_undo_history_mb(8 * GB, 16), 32);
        assert_eq!(auto.auto_undo_history_mb(512 * GB, 8), 1024);
        // Bigger caches set by hand leave less.
        let big_caches = ViewportMemorySettings {
            cache_sizes_auto: false,
            tile_cache_mb: 4096,
            ..Default::default()
        };
        assert!(
            big_caches.auto_undo_history_mb(16 * GB, 4) < auto.auto_undo_history_mb(16 * GB, 4)
        );
        let own = ViewportMemorySettings {
            undo_history_auto: false,
            undo_history_mb: 64,
            ..Default::default()
        };
        assert_eq!(own.effective_undo_history_mb(8 * GB, 4), 64);
    }

    #[test]
    fn panel_settings_default_to_every_panel_shown_and_open_and_old_files_get_them() {
        let panels = PanelSettings::default();
        assert!(
            panels.steps_open && panels.classes_open && panels.image_open && panels.objects_open
        );
        assert!(!panels.classes_folded && !panels.image_folded && !panels.objects_folded);
        let old: AppSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.panels, PanelSettings::default());
        let some: PanelSettings = serde_json::from_str(r#"{"imageFolded": true}"#).unwrap();
        assert!(some.image_folded && some.classes_open);
        assert_eq!(some.navigator, NavigatorMode::Auto);
        let never: PanelSettings = serde_json::from_str(r#"{"navigator": "never"}"#).unwrap();
        assert_eq!(never.navigator, NavigatorMode::Never);
    }

    #[test]
    fn preview_and_breakpoint_tiles_follow_the_ram_unless_set() {
        let gb = |n: u64| n << 30;
        assert_eq!(
            ViewportMemorySettings::auto_preview_tiles(gb(16)),
            4,
            "as before"
        );
        assert_eq!(ViewportMemorySettings::auto_preview_tiles(gb(2)), 1);
        assert_eq!(ViewportMemorySettings::auto_preview_tiles(gb(512)), 32);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(8)), 1);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(16)), 3);
        assert_eq!(ViewportMemorySettings::auto_breakpoint_tiles(gb(512)), 16);

        let mut settings = ViewportMemorySettings::default();
        assert_eq!(
            settings.requested_preview_tiles(),
            None,
            "auto: the running computer decides"
        );
        assert_eq!(settings.effective_breakpoint_tiles(gb(16)), 3);
        settings.preview_tiles_auto = false;
        settings.preview_tiles = 100;
        settings.breakpoint_tiles_auto = false;
        settings.breakpoint_tiles = 0;
        assert_eq!(settings.requested_preview_tiles(), Some(100));
        assert_eq!(
            settings.effective_breakpoint_tiles(gb(16)),
            1,
            "at least one"
        );
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

    /// 100 % when nothing is saved; a saved size comes back; a value edited
    /// by hand is taken as the nearest preset.
    #[test]
    fn the_interface_size_defaults_to_100_and_round_trips() {
        assert_eq!(AppSettings::default().interface_size.percent(), 100);
        let parsed: AppSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.interface_size.factor(), 1.0);

        let settings = AppSettings {
            interface_size: InterfaceSize::from_index(2),
            ..Default::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("\"interfaceSize\":150"), "{json}");
        let parsed: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.interface_size.percent(), 150);
        assert_eq!(parsed.interface_size.factor(), 1.5);

        let edited: AppSettings = serde_json::from_str(r#"{"interfaceSize":160}"#).unwrap();
        assert_eq!(edited.interface_size.percent(), 150);
        let edited: AppSettings = serde_json::from_str(r#"{"interfaceSize":10}"#).unwrap();
        assert_eq!(edited.interface_size.percent(), 100);
        let edited: AppSettings = serde_json::from_str(r#"{"interfaceSize":900}"#).unwrap();
        assert_eq!(edited.interface_size.percent(), 200);
        assert_eq!(InterfaceSize::from_index(99).percent(), 200);
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
    fn settings_from_before_recent_servers_still_load() {
        let settings: AppSettings = serde_json::from_str(r#"{"darkMode":true}"#).unwrap();
        assert!(settings.recent_servers.is_empty());
        assert_eq!(settings.migrate().color_scheme, ColorScheme::Dark);
    }
}
