// gui/src/lib.rs

pub use evanalyzer_gui_slint::*;

use evanalyzer_app::{AppHandle, Frontend, ProjectOwner, ProjectWithRuntime};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_core::ImageReader;
use slint::ComponentHandle;
use std::path::PathBuf;
use std::sync::{Arc, RwLockReadGuard, RwLockWriteGuard};

mod editor;
mod helper;
mod prelude;

// ----------------------------------------------------------------
// UiState - shared across all GUI controllers
// Wraps AppHandle and the Slint window handle
// ----------------------------------------------------------------

pub struct UiState {
    pub app: AppHandle, // cloneable handle - no Arc needed, AppHandle is already Arc inside
    pub ui_handle: slint::Weak<AppWindow>,
    pub results_ui_handle: slint::Weak<ResultsWindow>,
}

impl UiState {
    pub fn new(
        app: AppHandle,
        handle: slint::Weak<AppWindow>,
        results_handle: slint::Weak<ResultsWindow>,
    ) -> Self {
        Self {
            app,
            ui_handle: handle,
            results_ui_handle: results_handle,
        }
    }

    /// Acquire a read guard for the project.
    /// Drop before calling `get_project_write` on the same thread.
    pub fn get_project(&self) -> RwLockReadGuard<'_, ProjectWithRuntime> {
        self.app.get_project()
    }

    /// Acquire a write guard for the project.
    /// Exclusive - never hold a read guard on the same thread when calling this.
    pub fn get_project_write(&self) -> RwLockWriteGuard<'_, ProjectWithRuntime> {
        self.app.get_project_write()
    }

    /// Returns or creates a cached image reader for the given path.
    pub fn get_or_create_reader(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<ImageReader>, InternalErrors> {
        self.app.get_or_create_reader(new_path)
    }

    /// Loads a project from disk replacing the current project.
    pub fn load_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        self.app.load_project(path)
    }

    /// Imports an old (`.icproj`) project, replacing the current project.
    /// Returns conversion warnings and the legacy project's image folder, if any.
    pub fn import_legacy_project(
        &self,
        path: &PathBuf,
    ) -> Result<(Vec<String>, Option<String>), InternalErrors> {
        self.app.import_legacy_project(path)
    }

    /// Marks the project as having unsaved changes.
    pub fn mark_dirty(&self) {
        let ui = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_has_unsaved_changes(true);
            }
        })
        .ok();
        self.set_window_title(true);
    }

    /// Clears the unsaved-changes indicator.
    pub fn clear_dirty(&self) {
        let ui = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_has_unsaved_changes(false);
            }
        })
        .ok();
        self.set_window_title(false);
    }

    /// Recomputes and pushes the main window's title from the project's
    /// current file path (`tmp_settings.current_project`, set on load/save)
    /// and `dirty` — the standard "<file> — AppName" pattern most desktop
    /// apps use (VS Code, Office, JetBrains IDEs, ...), with a leading "●"
    /// while there are unsaved changes. `dirty` is passed in by the caller
    /// (`mark_dirty`/`clear_dirty` already know it) rather than read back
    /// from `ToolbarState`, so this works the same whether called from the
    /// UI thread or a background one.
    pub fn set_window_title(&self, dirty: bool) {
        let filename = self
            .get_project()
            .tmp_settings
            .current_project
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy().into_owned());
        let name = filename.unwrap_or_else(|| "Untitled".to_string());
        let title = if dirty {
            format!("● {name} — EVAnalyzer")
        } else {
            format!("{name} — EVAnalyzer")
        };

        let ui = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.set_window_title(title.into());
            }
        })
        .ok();
    }
}

// ----------------------------------------------------------------
// GuiFrontend - implements the app::Frontend trait
// Registered with ProjectOwner so app can push snapshots
// ----------------------------------------------------------------

pub struct GuiFrontend;

impl Frontend for GuiFrontend {
    /// Called by main - starts the Slint event loop.
    fn start(self: Box<Self>, owner: ProjectOwner) {
        if let Err(e) = run(owner) {
            log::error!("GUI exited with error: {}", e);
        }
    }
}

/// Public constructor - called by main.rs
pub fn create() -> GuiFrontend {
    GuiFrontend
}

// ----------------------------------------------------------------
// Internal startup
// ----------------------------------------------------------------

fn run(owner: ProjectOwner) -> Result<(), slint::PlatformError> {
    unsafe {
        std::env::set_var("SLINT_BACKEND", "skia");
        std::env::set_var("SLINT_SCALE_FACTOR", "1.0");
    }

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    let results_ui = ResultsWindow::new()?;
    let results_ui_handle = results_ui.as_weak();

    // Build AppHandle from owner - shares the same Arc<RwLock<ProjectSettings>>
    let app_handle = owner.handle();
    let ui_state = Arc::new(UiState::new(
        app_handle,
        ui_handle.clone(),
        results_ui_handle.clone(),
    ));
    // Reflects whatever project state already exists at startup — either the
    // blank default project, or one passed via a CLI `--project` argument
    // (loaded into `owner` before the GUI was created, so it's already
    // sitting in the shared `ProjectWithRuntime` `ui_state` now points at).
    ui_state.set_window_title(false);

    // Attach callbacks synchronously before the event loop starts.
    // Using invoke_from_event_loop here caused the initial `changed width/height`
    // layout events to fire before the Rust callbacks were registered, so the
    // viewport size was never reported to Rust until the user manually resized.
    let editor = Arc::new(editor::Editor::new(
        ui_handle.clone(),
        results_ui_handle.clone(),
        ui_state.clone(),
    ));
    editor.attach_callbacks();

    ui.run()
}
