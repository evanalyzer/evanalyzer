// gui/src/lib.rs

pub use evanalyzer_gui_slint::*;

use evanalyzer_app::backends::{Backend, ConnectionSecurity};
use evanalyzer_app::global::{AppHandle, ColorScheme, Frontend};
use evanalyzer_app::images::ImageMeta;
use evanalyzer_app::images::ImageSource;
use evanalyzer_app::project::{ProjectOwner, ProjectWithRuntime};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
pub use file_browser::{FileBrowser, FileMode, FileRequest};
use slint::ComponentHandle;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use crate::remote::reconnector::Reconnector;

/// How many steps of undo/redo history to keep - `ProjectSettings` is small
/// (metadata/geometry only, no pixel buffers - see the doc comment on
/// `UiState::undo_stack`), so this is a memory non-issue; it just bounds
/// how far back a very long editing session can rewind.
const UNDO_STACK_LIMIT: usize = 100;

/// Edits that land within this long of the last checkpoint are coalesced into
/// the same undo step instead of each getting their own - otherwise dragging
/// an object or typing into a text field would push dozens of near-identical
/// snapshots and "Undo" would barely seem to do anything per click.
const UNDO_COALESCE_WINDOW: Duration = Duration::from_millis(600);

mod editor;
mod file_browser;
mod helper;
mod license_text;
mod remote;
mod third_party_licenses;

// ----------------------------------------------------------------
// UiState - shared across all GUI controllers
// Wraps AppHandle and the Slint window handle
// ----------------------------------------------------------------

pub struct UiState {
    pub app: AppHandle, // cloneable handle - no Arc needed, AppHandle is already Arc inside
    pub ui_handle: slint::Weak<AppWindow>,
    /// The in-app open/save dialog - browses the backend's files, so remote
    /// mode shows the server's folders.
    pub file_browser: Arc<FileBrowser<AppWindow>>,
    /// Mirrors `ToolbarState.has_unsaved_changes`, but readable synchronously
    /// from any thread (the Slint property can only be read/written on the
    /// UI thread via `invoke_from_event_loop`) - lets background threads
    /// (e.g. "open a project" before doing any I/O) check for unsaved
    /// changes without a round trip through the event loop.
    dirty: AtomicBool,
    /// Snapshots of `ProjectWithRuntime::settings` taken right before a
    /// mutation, oldest first. `ProjectSettings` holds only metadata/geometry
    /// (paths, IDs, object polygons, pipeline params - no pixel buffers), so
    /// cloning the whole struct per checkpoint is cheap even for large
    /// projects. Populated by `get_project_write` (see `maybe_checkpoint_undo`)
    /// so every mutation chokepoint gets undo coverage for free, rather than
    /// having to thread checkpoint calls through every one of the ~60 call
    /// sites across the editor controllers.
    undo_stack: Mutex<VecDeque<ProjectSettings>>,
    /// Snapshots popped off `undo_stack` by `undo()`, so `redo()` can restore
    /// them. Cleared whenever a new checkpoint is pushed (a fresh edit after
    /// an undo invalidates the redone-away future).
    redo_stack: Mutex<VecDeque<ProjectSettings>>,
    /// When the most recent undo checkpoint was taken - drives the
    /// `UNDO_COALESCE_WINDOW` grouping in `maybe_checkpoint_undo`.
    last_checkpoint_at: Mutex<Instant>,
    /// Set by `undo`/`redo` to force the *next* `get_project_write` call to
    /// take a checkpoint regardless of `UNDO_COALESCE_WINDOW` timing.
    /// Without this, an edit landing within the coalesce window right after
    /// an undo would be silently merged into the pre-undo checkpoint instead
    /// of starting a new one - which would skip clearing `redo_stack`, so
    /// "Redo" could later resurrect a state that no longer follows from what
    /// the user just did.
    force_next_checkpoint: AtomicBool,
    /// The user's remembered preferences, loaded once from the backend (the
    /// worker's user folder in remote mode) and saved back there on every
    /// change - kept here so reading one doesn't cost a round trip.
    app_settings: Mutex<evanalyzer_app::global::AppSettings>,
    /// This computer - where "Disconnect" returns to, and where the list of
    /// recent servers is kept whatever the window is connected to.
    local_backend: Arc<dyn Backend>,
}

impl UiState {
    pub fn new(app: AppHandle, handle: slint::Weak<AppWindow>) -> Self {
        let backend = app.backend();
        let app_settings = backend.load_app_settings().unwrap_or_else(|e| {
            log::warn!("Could not load the app settings, using defaults: {e}");
            Default::default()
        });
        let local_backend: Arc<dyn Backend> = if backend.is_remote() {
            Arc::new(evanalyzer_app::backends::local::LocalBackend::default())
        } else {
            Arc::clone(&backend)
        };
        Self {
            file_browser: Arc::new(FileBrowser::new(handle.clone(), Arc::clone(&backend))),
            app,
            ui_handle: handle,
            dirty: AtomicBool::new(false),
            undo_stack: Mutex::new(VecDeque::new()),
            redo_stack: Mutex::new(VecDeque::new()),
            last_checkpoint_at: Mutex::new(Instant::now()),
            force_next_checkpoint: AtomicBool::new(false),
            app_settings: Mutex::new(app_settings),
            local_backend,
        }
    }

    /// This computer's backend - see `local_backend`.
    pub fn local_backend(&self) -> Arc<dyn Backend> {
        Arc::clone(&self.local_backend)
    }

    /// The preferences of the current backend's user (they live in its
    /// user folder - the server's when connected), read again after a
    /// switch.
    pub fn reload_app_settings(&self) {
        let settings = self.app.backend().load_app_settings().unwrap_or_else(|e| {
            log::warn!("Could not load the app settings, using defaults: {e}");
            Default::default()
        });
        *self.app_settings.lock().unwrap() = settings;
    }

    /// The servers connected to before, most recent first - always this
    /// computer's list.
    pub fn recent_servers(&self) -> Vec<evanalyzer_app::global::RecentServer> {
        self.local_backend
            .load_app_settings()
            .map(|settings| settings.recent_servers)
            .unwrap_or_default()
    }

    /// Puts `server` first in this computer's list of recent servers.
    pub fn remember_server(&self, server: evanalyzer_app::global::RecentServer) {
        let result = self
            .local_backend
            .load_app_settings()
            .and_then(|mut settings| {
                settings.remember_server(server);
                self.local_backend.save_app_settings(&settings)
            });
        if let Err(e) = result {
            log::warn!("Could not remember the server: {e}");
        }
    }

    /// The viewport's memory settings - always this computer's, since they
    /// are about this computer's RAM.
    pub fn viewport_memory_settings(&self) -> evanalyzer_app::global::ViewportMemorySettings {
        self.local_backend
            .load_app_settings()
            .map(|settings| settings.viewport)
            .unwrap_or_default()
    }

    /// Saves the viewport's memory settings on this computer.
    pub fn save_viewport_memory_settings(
        &self,
        viewport: evanalyzer_app::global::ViewportMemorySettings,
    ) {
        // Locally the cached preferences are saved to this same file by
        // `update_app_settings` - keep them current so that doesn't write
        // an older value back.
        self.app_settings.lock().unwrap().viewport = viewport.clone();
        let result = self
            .local_backend
            .load_app_settings()
            .and_then(|mut settings| {
                settings.viewport = viewport;
                self.local_backend.save_app_settings(&settings)
            });
        if let Err(e) = result {
            log::warn!("Could not save the viewport memory settings: {e}");
        }
    }

    /// The user's remembered preferences (see `app_settings`).
    pub fn load_app_settings(&self) -> evanalyzer_app::global::AppSettings {
        self.app_settings.lock().unwrap().clone()
    }

    /// Changes one preference and saves all of them through the backend.
    pub fn update_app_settings(
        &self,
        change: impl FnOnce(&mut evanalyzer_app::global::AppSettings),
    ) {
        let settings = {
            let mut settings = self.app_settings.lock().unwrap();
            change(&mut settings);
            settings.clone()
        };
        if let Err(e) = self.app.backend().save_app_settings(&settings) {
            log::warn!("Could not save the app settings: {e}");
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
        self.maybe_checkpoint_undo();
        self.app.get_project_write()
    }

    /// Write access for runtime-only state (`tmp_settings`, e.g. the pipeline
    /// focus) that is not part of the undo history: takes no undo checkpoint.
    /// A checkpoint here would be an empty undo step - and right after an
    /// undo it would even clear the redo stack.
    pub fn get_project_runtime_write(&self) -> RwLockWriteGuard<'_, ProjectWithRuntime> {
        self.app.get_project_write()
    }

    /// Takes an undo snapshot of the current `settings` if enough time has
    /// passed since the last one (see `UNDO_COALESCE_WINDOW`), then calls
    /// `get_project_write` for real. Must run *before* the write guard below
    /// is acquired - it briefly takes its own read guard to clone the
    /// pre-mutation state, which would deadlock (std `RwLock` isn't
    /// reentrant) if taken while a write guard from this same call were
    /// already held.
    fn maybe_checkpoint_undo(&self) {
        let mut last = self.last_checkpoint_at.lock().expect("Poisoned");
        let now = Instant::now();
        let should_checkpoint = self.force_next_checkpoint.swap(false, Ordering::Relaxed)
            || now.duration_since(*last) >= UNDO_COALESCE_WINDOW
            || {
                let stack = self.undo_stack.lock().expect("Poisoned");
                stack.is_empty()
            };
        *last = now;
        drop(last);

        if !should_checkpoint {
            return;
        }

        let snapshot = self.app.get_project().settings.clone();
        let mut undo_stack = self.undo_stack.lock().expect("Poisoned");
        undo_stack.push_back(snapshot);
        if undo_stack.len() > UNDO_STACK_LIMIT {
            undo_stack.pop_front();
        }
        drop(undo_stack);
        self.redo_stack.lock().expect("Poisoned").clear();
        self.push_undo_redo_state_to_ui();
    }

    /// Forgets every undo/redo step - when another project replaces the
    /// current one, the steps belong to the old project: undoing one would
    /// put the old project's content into the new one (and saving it, over
    /// the new project's file).
    pub fn clear_undo_history(&self) {
        self.undo_stack.lock().expect("Poisoned").clear();
        self.redo_stack.lock().expect("Poisoned").clear();
        self.push_undo_redo_state_to_ui();
    }

    /// Restores the most recent undo checkpoint, if any. Returns whether
    /// there was one to restore - the caller is responsible for refreshing
    /// every panel from the restored `settings` afterwards (there's no single
    /// "refresh everything" entry point - see `UndoRedoController`).
    pub fn undo(&self) -> bool {
        let Some(prev) = self.undo_stack.lock().expect("Poisoned").pop_back() else {
            return false;
        };

        let current = self.app.get_project().settings.clone();
        let mut redo_stack = self.redo_stack.lock().expect("Poisoned");
        redo_stack.push_back(current);
        if redo_stack.len() > UNDO_STACK_LIMIT {
            redo_stack.pop_front();
        }
        drop(redo_stack);

        {
            let mut project = self.app.get_project_write();
            project.settings = prev;
            // The restored settings may no longer contain the previously
            // selected object (or contain a different one reusing the same
            // list position) - drop the selection rather than risk it
            // pointing at the wrong object.
            project.set_selected_object(None);
        }
        self.force_next_checkpoint.store(true, Ordering::Relaxed);
        self.push_undo_redo_state_to_ui();
        self.mark_dirty();
        true
    }

    /// Re-applies the most recently undone checkpoint, if any. Same
    /// refresh-after-restore contract as `undo()`.
    pub fn redo(&self) -> bool {
        let Some(next) = self.redo_stack.lock().expect("Poisoned").pop_back() else {
            return false;
        };

        let current = self.app.get_project().settings.clone();
        let mut undo_stack = self.undo_stack.lock().expect("Poisoned");
        undo_stack.push_back(current);
        if undo_stack.len() > UNDO_STACK_LIMIT {
            undo_stack.pop_front();
        }
        drop(undo_stack);

        {
            let mut project = self.app.get_project_write();
            project.settings = next;
            project.set_selected_object(None);
        }
        self.force_next_checkpoint.store(true, Ordering::Relaxed);
        self.push_undo_redo_state_to_ui();
        self.mark_dirty();
        true
    }

    /// Pushes `ToolbarState.can_undo`/`can_redo` to the UI thread so the
    /// toolbar buttons enable/disable themselves.
    fn push_undo_redo_state_to_ui(&self) {
        let can_undo = !self.undo_stack.lock().expect("Poisoned").is_empty();
        let can_redo = !self.redo_stack.lock().expect("Poisoned").is_empty();
        let ui = self.ui_handle.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_can_undo(can_undo);
                w.global::<ToolbarState>().set_can_redo(can_redo);
            }
        })
        .ok();
    }

    /// Returns the metadata of the image at the given path, opening (and
    /// caching) its readers if needed.
    pub fn get_image_meta(&self, new_path: &PathBuf) -> Result<ImageMeta, InternalErrors> {
        self.app.get_image_meta(new_path)
    }

    /// Returns the cached opened image for the given path, opening it
    /// through the backend (local readers or a server) if needed.
    pub fn get_image_source(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<dyn ImageSource>, InternalErrors> {
        self.app.get_image_source(new_path)
    }

    /// Where analysis/preview/training runs - local or a server. Can change
    /// while the program runs ("Connect to server"), so fetch it per use.
    pub fn backend(&self) -> Arc<dyn Backend> {
        self.app.backend()
    }

    /// Loads a project from disk replacing the current project.
    pub fn load_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        self.app.load_project(path)?;
        self.clear_undo_history();
        Ok(())
    }

    /// Replaces the current project with a fresh, blank, unsaved one -
    /// "File > New".
    pub fn new_project(&self) {
        self.app.new_project();
        self.clear_undo_history();
    }

    /// Imports an old (`.icproj`) project, replacing the current project.
    /// Returns conversion warnings and the legacy project's image folder, if any.
    pub fn import_legacy_project(
        &self,
        path: &PathBuf,
    ) -> Result<(Vec<String>, Option<String>), InternalErrors> {
        let imported = self.app.import_legacy_project(path)?;
        self.clear_undo_history();
        Ok(imported)
    }

    /// Marks the project as having unsaved changes.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Relaxed);
        let ui = self.ui_handle.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_has_unsaved_changes(true);
            }
        })
        .ok();
        self.set_window_title(true);
    }

    /// Clears the unsaved-changes indicator.
    pub fn clear_dirty(&self) {
        self.dirty.store(false, Ordering::Relaxed);
        let ui = self.ui_handle.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_has_unsaved_changes(false);
            }
        })
        .ok();
        self.set_window_title(false);
    }

    /// Whether the project has unsaved changes. Safe to call from any
    /// thread, unlike reading `ToolbarState.has_unsaved_changes` directly.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Relaxed)
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
        let mut title = if dirty {
            format!("*{name} - EVAnalyzer")
        } else {
            format!("{name} - EVAnalyzer")
        };
        // Visible in the taskbar and window switcher too, so a local and a
        // remote window can't be confused there either.
        if self.backend().is_remote() {
            title.push_str(" - ");
            title.push_str(&connection_label(self.backend().as_ref()));
        }

        let ui = self.ui_handle.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

pub struct GuiFrontend {
    /// Opened once the window is up, as with "File > Open".
    project: Option<std::path::PathBuf>,
}

impl GuiFrontend {
    /// Opens the project file at `path` at startup (`--project`).
    pub fn with_project(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.project = path;
        self
    }
}

impl Frontend for GuiFrontend {
    /// Called by main - starts the Slint event loop.
    fn start(self: Box<Self>, owner: ProjectOwner) {
        if let Err(e) = run(owner, self.project) {
            log::error!("GUI exited with error: {}", e);
        }
    }
}

/// Public constructor - called by main.rs
pub fn create() -> GuiFrontend {
    GuiFrontend { project: None }
}

// ----------------------------------------------------------------
// Internal startup
// ----------------------------------------------------------------

fn run(
    owner: ProjectOwner,
    project: Option<std::path::PathBuf>,
) -> Result<(), slint::PlatformError> {
    unsafe {
        // Skia has documented first-frame rendering glitches on this backend
        // path (see https://github.com/slint-ui/slint/issues/7845) that
        // FemtoVG doesn't share - left unset so Slint's normal fallback order
        // (Skia -> FemtoVG -> software) picks the more stable renderer instead.
        std::env::set_var("SLINT_SCALE_FACTOR", "1.0");
    }

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();

    load_about_dialog_information(&ui);

    // Build AppHandle from owner - shares the same Arc<RwLock<ProjectSettings>>
    let app_handle = owner.handle();
    let ui_state = Arc::new(UiState::new(app_handle, ui_handle.clone()));
    // Reflects whatever project state already exists at startup — either the
    // blank default project, or one passed via a CLI `--project` argument
    // (loaded into `owner` before the GUI was created, so it's already
    // sitting in the shared `ProjectWithRuntime` `ui_state` now points at).
    ui_state.set_window_title(false);
    apply_user_settings(&ui, &ui_state);
    attach_system_info(&ui, Arc::clone(&ui_state));
    ui_state.file_browser.attach(&ui);

    // Attach callbacks synchronously before the event loop starts.
    // Using invoke_from_event_loop here caused the initial `changed width/height`
    // layout events to fire before the Rust callbacks were registered, so the
    // viewport size was never reported to Rust until the user manually resized.
    let editor = Arc::new(editor::Editor::new(ui_handle.clone(), ui_state.clone()));
    editor.attach_callbacks();
    // `--project`: opened like "File > Open" - a file that can't be read
    // shows the usual warning in the window.
    if let Some(path) = project {
        editor.open_project(&path);
    }
    // Kept alive until the window closes.
    let app = ui_state.app.clone();
    let _connection_watch = show_connection(&ui, move || app.backend(), editor.on_reconnected());

    ui.run()
}

/// About dialog content: version comes from the crate version (which the
/// release CI patches to the git tag before building), licenses are fixed.
/// The system tab is filled per opening by [`attach_system_info`].
fn load_about_dialog_information(ui: &AppWindow) {
    let info = ui.global::<AppInfoState>();
    info.set_version(env!("CARGO_PKG_VERSION").into());
    let paragraphs: Vec<slint::SharedString> = license_text::LICENSE_TEXT
        .split("\n\n")
        .map(|p| p.into())
        .collect();
    info.set_license_paragraphs(slint::ModelRc::new(slint::VecModel::from(paragraphs)));

    let (third_party_groups, third_party_package_count) = third_party_licenses::load();
    info.set_third_party_package_count(third_party_package_count as i32);
    let third_party_groups: Vec<ThirdPartyLicense> = third_party_groups
        .into_iter()
        .map(|g| ThirdPartyLicense {
            id: g.id.into(),
            name: g.name.into(),
            crates: g.crates.into(),
            text_paragraphs: slint::ModelRc::new(slint::VecModel::from(
                g.text_paragraphs
                    .into_iter()
                    .map(slint::SharedString::from)
                    .collect::<Vec<_>>(),
            )),
        })
        .collect();
    info.set_third_party_licenses(slint::ModelRc::new(slint::VecModel::from(
        third_party_groups,
    )));
}

/// The About dialog's system tab: whenever it opens, asks `backend` for the
/// machine the work runs on - in remote mode the worker's, not this one.
/// Fetched on a background thread: the CUDA probe is slow on first use
/// (loading the driver, commonly hundreds of ms), plus a round trip in
/// remote mode.
fn attach_system_info(ui: &AppWindow, ui_state: Arc<UiState>) {
    let ui_weak = ui.as_weak();
    ui.global::<AppInfoState>().on_refresh_system_info(move || {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        let backend = ui_state.backend();
        let info = ui.global::<AppInfoState>();
        let source = if backend.is_remote() {
            format!("worker at {}", backend.description())
        } else {
            "this computer".to_string()
        };
        info.set_system_source(source.into());
        info.set_system_error("".into());
        info.set_system_loading(true);

        let backend = Arc::clone(&backend);
        let ui_weak = ui_weak.clone();
        crate::helper::ui_thread::spawn(move || {
            let result = backend.system_info();
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };
                let info = ui.global::<AppInfoState>();
                info.set_system_loading(false);
                match result {
                    Ok(system) => {
                        info.set_system_os(system.os.into());
                        info.set_cpu_cores(system.cpu_cores as i32);
                        info.set_ram_total(format_ram(system.ram_total_bytes).into());
                        info.set_cuda_available(system.cuda_available);
                    }
                    Err(e) => {
                        info.set_system_os("-".into());
                        info.set_cpu_cores(0);
                        info.set_ram_total("-".into());
                        info.set_cuda_available(false);
                        info.set_system_error(format!("Could not read the system: {e}").into());
                    }
                }
            })
            .ok();
        });
    });
}

/// `bytes` of RAM as GB with one decimal.
fn format_ram(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
}

/// "alice @ workstation:7400" for a server, "This computer" otherwise.
pub fn connection_label(backend: &dyn Backend) -> String {
    if !backend.is_remote() {
        return "This computer".into();
    }
    let url = backend.description();
    let host = url
        .trim_start_matches("ws://")
        .trim_start_matches("wss://")
        .trim_end_matches('/');
    match backend.user() {
        Some(user) => format!("{user} @ {host}"),
        None => host.to_string(),
    }
}

/// Shows where the work happens (`ConnectionState`: status bar badge,
/// lost-connection banner) and keeps it current: once a second it checks
/// the backend `current` returns - it changes on "Connect to server" and
/// "Disconnect" - and whether its connection is still up. A dropped
/// connection is reconnected in the background (see [`Reconnector`]);
/// `on_reconnected` runs once it's back. Returns the timer, which must stay
/// alive as long as the window.
fn show_connection(
    ui: &AppWindow,
    current: impl Fn() -> Arc<dyn Backend> + 'static,
    on_reconnected: impl Fn() + Send + Sync + 'static,
) -> slint::Timer {
    let on_reconnected: Arc<dyn Fn() + Send + Sync> = Arc::new(on_reconnected);
    // The reconnector of the backend shown, if it is a server.
    let reconnector: Arc<Mutex<Option<Arc<Reconnector>>>> = Arc::default();
    let now = Arc::clone(&reconnector);
    ui.global::<ConnectionState>().on_reconnect_now(move || {
        if let Some(reconnector) = now.lock().unwrap().as_ref() {
            reconnector.try_now();
        }
    });

    let mut shown: Option<Arc<dyn Backend>> = None;
    let mut update = {
        let ui = ui.as_weak();
        move || {
            let Some(ui) = ui.upgrade() else {
                return;
            };
            let backend = current();
            let state = ui.global::<ConnectionState>();
            if !shown
                .as_ref()
                .is_some_and(|shown| Arc::ptr_eq(shown, &backend))
            {
                fill_connection_state(&state, backend.as_ref());
                let mut slot = reconnector.lock().unwrap();
                if let Some(old) = slot.take() {
                    old.stop();
                }
                if backend.is_remote() {
                    let on_reconnected = Arc::clone(&on_reconnected);
                    *slot = Some(Arc::new(Reconnector {
                        ui: ui.as_weak(),
                        backend: Arc::clone(&backend),
                        on_reconnected: Box::new(move || on_reconnected()),
                        try_now: Mutex::new(None),
                        stopped: AtomicBool::new(false),
                    }));
                }
                shown = Some(Arc::clone(&backend));
            }
            let connected = backend.is_connected();
            if state.get_connected() != connected {
                state.set_connected(connected);
            }
            if !connected && let Some(reconnector) = reconnector.lock().unwrap().as_ref() {
                reconnector.start();
            }
        }
    };
    update();
    let timer = slint::Timer::default();
    timer.start(slint::TimerMode::Repeated, Duration::from_secs(1), update);
    timer
}

/// `ConnectionState` for `backend`, as long as nothing changes.
fn fill_connection_state(state: &ConnectionState<'_>, backend: &dyn Backend) {
    state.set_remote(backend.is_remote());
    state.set_label(connection_label(backend).into());
    state.set_connected(backend.is_connected());
    state.set_lost_message("".into());
    state.set_can_reconnect(true);
    let security = backend.connection_security();
    state.set_encrypted(matches!(
        security,
        ConnectionSecurity::Local
            | ConnectionSecurity::Encrypted
            | ConnectionSecurity::EncryptedUnverified
    ));
    state.set_verified(security != ConnectionSecurity::EncryptedUnverified);
}

/// Applies the remembered appearance and saves the color scheme the user
/// picks - through `ui_state`, so in remote mode it lives on the worker.
fn apply_user_settings(ui: &AppWindow, ui_state: &Arc<UiState>) {
    apply_appearance(ui, ui_state);

    let ui_state = Arc::clone(ui_state);
    ui.global::<Appearance>().on_scheme_changed(move |scheme| {
        ui_state
            .update_app_settings(|settings| settings.color_scheme = color_scheme_from_ui(scheme));
    });
}

/// The remembered color scheme.
pub(crate) fn apply_appearance(ui: &AppWindow, ui_state: &UiState) {
    let scheme = color_scheme_to_ui(ui_state.load_app_settings().color_scheme);
    ui.global::<Appearance>().invoke_apply(scheme);
}

fn color_scheme_to_ui(scheme: ColorScheme) -> InkScheme {
    match scheme {
        ColorScheme::Light => InkScheme::Light,
        ColorScheme::Dark => InkScheme::Dark,
        ColorScheme::Dim => InkScheme::Dim,
    }
}

fn color_scheme_from_ui(scheme: InkScheme) -> ColorScheme {
    match scheme {
        InkScheme::Light => ColorScheme::Light,
        InkScheme::Dark => ColorScheme::Dark,
        InkScheme::Dim => ColorScheme::Dim,
    }
}

#[cfg(test)]
mod ui_state_tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;

    // -- dirty tracking -----------------------------------------------------

    #[test]
    fn is_dirty_defaults_to_false() {
        let ui_state = test_ui_state();
        assert!(!ui_state.is_dirty());
    }

    #[test]
    fn mark_dirty_and_clear_dirty_round_trip() {
        let ui_state = test_ui_state();

        ui_state.mark_dirty();
        assert!(ui_state.is_dirty());

        ui_state.clear_dirty();
        assert!(!ui_state.is_dirty());
    }

    #[test]
    fn local_backend_is_labelled_this_computer_and_shown_as_local() {
        crate::editor::test_support::ensure_slint_test_platform();
        let backend: Arc<dyn Backend> =
            Arc::new(evanalyzer_app::backends::local::LocalBackend::default());
        assert_eq!(connection_label(backend.as_ref()), "This computer");

        let ui = AppWindow::new().unwrap();
        let shown = Arc::clone(&backend);
        let _timer = show_connection(&ui, move || Arc::clone(&shown), || {});
        let state = ui.global::<ConnectionState>();
        assert!(!state.get_remote());
        assert!(state.get_connected());
        assert!(
            state.get_encrypted() && state.get_verified(),
            "nothing to warn about"
        );
        assert_eq!(state.get_label(), "This computer");
    }

    #[test]
    fn the_connection_display_follows_a_switch_to_a_server_and_back() {
        use evanalyzer_app::backends::remote::{RemoteBackend, Worker};
        crate::editor::test_support::ensure_slint_test_platform();
        let worker = Worker::bind("127.0.0.1:0", "t".into()).unwrap();
        let addr = worker.local_addr().unwrap();
        std::thread::spawn(move || {
            worker.run(Arc::new(
                evanalyzer_app::backends::local::LocalBackend::default(),
            ))
        });
        let local: Arc<dyn Backend> =
            Arc::new(evanalyzer_app::backends::local::LocalBackend::default());
        let remote: Arc<dyn Backend> =
            Arc::new(RemoteBackend::connect(&format!("ws://{addr}"), "t").unwrap());
        let current = Arc::new(Mutex::new(Arc::clone(&local)));

        let ui = AppWindow::new().unwrap();
        let shown = Arc::clone(&current);
        let _timer = show_connection(&ui, move || Arc::clone(&shown.lock().unwrap()), || {});
        let state = ui.global::<ConnectionState>();
        assert!(!state.get_remote());

        *current.lock().unwrap() = Arc::clone(&remote);
        i_slint_backend_testing::mock_elapsed_time(Duration::from_millis(1100));
        assert!(state.get_remote());
        assert_eq!(state.get_label(), addr.to_string());
        assert!(!state.get_encrypted(), "ws://");

        *current.lock().unwrap() = local;
        i_slint_backend_testing::mock_elapsed_time(Duration::from_millis(1100));
        assert!(!state.get_remote());
        assert_eq!(state.get_label(), "This computer");
    }

    #[test]
    fn set_window_title_does_not_panic_with_or_without_a_current_project_path() {
        let ui_state = test_ui_state();
        // No `current_project` set (fresh project) - exercises the
        // `unwrap_or_else("Untitled")` branch.
        ui_state.set_window_title(true);
        ui_state.set_window_title(false);

        {
            let mut project = ui_state.get_project_write();
            project.tmp_settings.current_project = Some(PathBuf::from("/a/b/My Project.evaproj"));
        }
        ui_state.set_window_title(true);
        ui_state.set_window_title(false);
    }

    // -- undo / redo ----------------------------------------------------------

    fn set_name(ui_state: &UiState, name: &str) {
        ui_state.get_project_write().meta.name = name.to_string();
    }

    fn name(ui_state: &UiState) -> String {
        ui_state.get_project().meta.name.clone()
    }

    #[test]
    fn undo_on_an_untouched_project_returns_false() {
        let ui_state = test_ui_state();
        assert!(!ui_state.undo());
    }

    #[test]
    fn redo_with_nothing_undone_returns_false() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "first");
        assert!(!ui_state.redo());
    }

    #[test]
    fn get_project_write_takes_a_checkpoint_that_undo_can_restore() {
        let ui_state = test_ui_state();
        assert_eq!(name(&ui_state), ""); // default project state

        set_name(&ui_state, "first edit");
        assert_eq!(name(&ui_state), "first edit");

        assert!(
            ui_state.undo(),
            "a checkpoint of the pre-edit state must exist"
        );
        assert_eq!(
            name(&ui_state),
            "",
            "undo must restore the state from before the edit"
        );
    }

    #[test]
    fn undo_marks_the_project_dirty_and_clears_the_selected_object() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "first edit");
        ui_state.clear_dirty();
        ui_state
            .get_project_write()
            .set_selected_object(Some(evanalyzer_cfg::core_types::ObjectId(1)));

        ui_state.undo();

        assert!(ui_state.is_dirty());
        assert_eq!(ui_state.get_project().get_selected_object_id(), None);
    }

    #[test]
    fn redo_reapplies_the_state_that_was_just_undone() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "first edit");
        ui_state.undo();
        assert_eq!(name(&ui_state), "");

        assert!(ui_state.redo());
        assert_eq!(name(&ui_state), "first edit");
    }

    #[test]
    fn redo_stack_is_cleared_by_a_fresh_edit_after_an_undo() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "first edit");
        ui_state.undo();
        assert!(
            ui_state.redo_stack.lock().unwrap().len() == 1,
            "sanity check: redo has something to reapply"
        );

        // A fresh edit after the undo must invalidate the redone-away future,
        // per `force_next_checkpoint`'s doc comment.
        set_name(&ui_state, "diverging edit");

        assert!(
            ui_state.redo_stack.lock().unwrap().is_empty(),
            "starting a new edit path after undo must clear the redo stack"
        );
        assert!(!ui_state.redo());
    }

    #[test]
    fn rapid_successive_writes_are_coalesced_into_one_checkpoint() {
        let ui_state = test_ui_state();
        // Both writes land well within `UNDO_COALESCE_WINDOW` of each other
        // (no sleep needed - two sequential calls take microseconds).
        set_name(&ui_state, "edit 1");
        set_name(&ui_state, "edit 2");

        assert_eq!(
            ui_state.undo_stack.lock().unwrap().len(),
            1,
            "two edits within the coalesce window must share one checkpoint"
        );
        // The single checkpoint must be the state from *before* either edit.
        ui_state.undo();
        assert_eq!(name(&ui_state), "");
    }

    // -- undo history limit ---------------------------------------------------

    #[test]
    fn a_new_checkpoint_beyond_the_limit_drops_the_oldest_one() {
        let ui_state = test_ui_state();
        {
            let mut undo_stack = ui_state.undo_stack.lock().unwrap();
            for i in 0..UNDO_STACK_LIMIT {
                let mut settings = ProjectSettings::default();
                settings.meta.name = format!("step {i}");
                undo_stack.push_back(settings);
            }
        }
        ui_state
            .force_next_checkpoint
            .store(true, Ordering::Relaxed);

        set_name(&ui_state, "newest");

        let undo_stack = ui_state.undo_stack.lock().unwrap();
        assert_eq!(undo_stack.len(), UNDO_STACK_LIMIT);
        assert_eq!(undo_stack.front().unwrap().meta.name, "step 1");
        assert_eq!(undo_stack.back().unwrap().meta.name, "");
    }

    #[test]
    fn undo_with_a_full_redo_history_drops_the_oldest_redo_step() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "edited");
        {
            let mut redo_stack = ui_state.redo_stack.lock().unwrap();
            for i in 0..UNDO_STACK_LIMIT {
                let mut settings = ProjectSettings::default();
                settings.meta.name = format!("redo {i}");
                redo_stack.push_back(settings);
            }
        }

        assert!(ui_state.undo());

        let redo_stack = ui_state.redo_stack.lock().unwrap();
        assert_eq!(redo_stack.len(), UNDO_STACK_LIMIT);
        assert_eq!(redo_stack.front().unwrap().meta.name, "redo 1");
        assert_eq!(redo_stack.back().unwrap().meta.name, "edited");
    }

    #[test]
    fn redo_with_a_full_undo_history_drops_the_oldest_undo_step() {
        let ui_state = test_ui_state();
        set_name(&ui_state, "edited");
        assert!(ui_state.undo());
        {
            let mut undo_stack = ui_state.undo_stack.lock().unwrap();
            for i in 0..UNDO_STACK_LIMIT {
                let mut settings = ProjectSettings::default();
                settings.meta.name = format!("undo {i}");
                undo_stack.push_back(settings);
            }
        }

        assert!(ui_state.redo());

        assert_eq!(name(&ui_state), "edited");
        let undo_stack = ui_state.undo_stack.lock().unwrap();
        assert_eq!(undo_stack.len(), UNDO_STACK_LIMIT);
        assert_eq!(undo_stack.front().unwrap().meta.name, "undo 1");
        assert_eq!(undo_stack.back().unwrap().meta.name, "");
    }

    // -- preferences ----------------------------------------------------------

    /// A `UiState` whose user folder can't be created: its home is a file.
    fn ui_state_that_cannot_save_settings() -> Arc<UiState> {
        use std::sync::atomic::AtomicUsize;
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let folder = std::env::temp_dir()
            .join("evanalyzer-gui-test-homes")
            .join(format!(
                "{}-blocked-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&folder).unwrap();
        let home = folder.join("home");
        std::fs::write(&home, "not a folder").unwrap();
        let backend = evanalyzer_app::backends::local::LocalBackend::default()
            .with_home(&home)
            .expect("backend");
        let owner = ProjectOwner::with_backend(Arc::new(backend));
        Arc::new(UiState::new(owner.handle(), slint::Weak::default()))
    }

    #[test]
    fn viewport_memory_settings_are_saved_on_this_computer_and_read_back() {
        let ui_state = test_ui_state();
        let mut viewport = ui_state.viewport_memory_settings();
        viewport.tile_history_budget_mb += 123;

        ui_state.save_viewport_memory_settings(viewport.clone());

        assert_eq!(ui_state.viewport_memory_settings(), viewport);
        assert_eq!(ui_state.load_app_settings().viewport, viewport);
    }

    #[test]
    fn preferences_that_cannot_be_saved_are_still_kept_in_memory() {
        let ui_state = ui_state_that_cannot_save_settings();

        ui_state.update_app_settings(|settings| settings.color_scheme = ColorScheme::Dim);
        assert_eq!(ui_state.load_app_settings().color_scheme, ColorScheme::Dim);

        let mut viewport = ui_state.viewport_memory_settings();
        viewport.tile_history_budget_mb += 7;
        ui_state.save_viewport_memory_settings(viewport.clone());
        assert_eq!(ui_state.load_app_settings().viewport, viewport);

        ui_state.remember_server(evanalyzer_app::global::RecentServer {
            url: "ws://localhost:7400".into(),
            user: "alice".into(),
            fingerprint: None,
        });
        assert!(ui_state.recent_servers().is_empty());
    }

    #[test]
    fn reload_app_settings_reads_what_the_backend_saved() {
        let ui_state = test_ui_state();
        ui_state.update_app_settings(|settings| settings.color_scheme = ColorScheme::Dark);
        *ui_state.app_settings.lock().unwrap() = Default::default();

        ui_state.reload_app_settings();

        assert_eq!(ui_state.load_app_settings().color_scheme, ColorScheme::Dark);
    }

    // -- startup / about dialog -----------------------------------------------

    #[test]
    fn the_about_dialog_shows_the_crate_version_and_the_licenses() {
        let ui = crate::editor::test_support::test_ui_window();

        load_about_dialog_information(&ui);

        let info = ui.global::<AppInfoState>();
        assert_eq!(info.get_version(), env!("CARGO_PKG_VERSION"));
        assert!(slint::Model::row_count(&info.get_license_paragraphs()) > 0);
        assert_eq!(
            slint::Model::row_count(&info.get_third_party_licenses()) > 0,
            info.get_third_party_package_count() > 0
        );
    }

    #[test]
    fn opening_the_about_dialog_shows_this_computers_system() {
        use crate::helper::ui_thread::drain_ui_queue;
        let ui = crate::editor::test_support::test_ui_window();
        let ui_state =
            crate::editor::test_support::ui_state_with_window(&ui, ProjectWithRuntime::default());
        attach_system_info(&ui, Arc::clone(&ui_state));
        let info = ui.global::<AppInfoState>();

        info.invoke_refresh_system_info();
        assert_eq!(info.get_system_source(), "this computer");
        assert!(info.get_system_loading());
        drain_ui_queue();

        assert!(!info.get_system_loading());
        assert_eq!(info.get_system_os(), std::env::consts::OS);
        assert!(info.get_cpu_cores() > 0);
        assert!(info.get_ram_total().ends_with(" GB"));
        assert_eq!(info.get_system_error(), "");
    }

    #[test]
    fn refreshing_the_system_info_after_the_window_closed_does_nothing() {
        use crate::helper::ui_thread::drain_ui_queue;
        let ui = crate::editor::test_support::test_ui_window();
        let ui_state = test_ui_state();
        attach_system_info(&ui, ui_state);
        let info = ui.global::<AppInfoState>();
        info.invoke_refresh_system_info();
        drop(info);
        drop(ui);
        drain_ui_queue();
    }

    #[test]
    fn format_ram_shows_gigabytes_with_one_decimal() {
        assert_eq!(format_ram(0), "0.0 GB");
        assert_eq!(format_ram(1_073_741_824), "1.0 GB");
        assert_eq!(format_ram(3 * 536_870_912), "1.5 GB");
    }

    #[test]
    fn a_picked_color_scheme_is_remembered_and_applied_to_the_results_window() {
        let ui = crate::editor::test_support::test_ui_window();
        let ui_state =
            crate::editor::test_support::ui_state_with_window(&ui, ProjectWithRuntime::default());
        apply_user_settings(&ui, &ui_state);

        for (picked, saved) in [
            (InkScheme::Dark, ColorScheme::Dark),
            (InkScheme::Dim, ColorScheme::Dim),
            (InkScheme::Light, ColorScheme::Light),
        ] {
            ui.global::<Appearance>().invoke_pick(picked);
            assert_eq!(ui_state.load_app_settings().color_scheme, saved);
            assert_eq!(ui.global::<Ink>().get_scheme(), picked);
            assert_eq!(ui.global::<Ink>().get_scheme(), picked);
        }
        ui.global::<Appearance>().invoke_pick(InkScheme::Dark);
        assert_eq!(ui_state.load_app_settings().color_scheme, ColorScheme::Dark);
    }

    #[test]
    fn the_remembered_color_scheme_is_applied_at_start() {
        let ui = crate::editor::test_support::test_ui_window();
        let ui_state =
            crate::editor::test_support::ui_state_with_window(&ui, ProjectWithRuntime::default());
        ui_state.update_app_settings(|settings| settings.color_scheme = ColorScheme::Dim);

        apply_user_settings(&ui, &ui_state);

        assert_eq!(ui.global::<Ink>().get_scheme(), InkScheme::Dim);
        assert_eq!(ui.global::<Ink>().get_scheme(), InkScheme::Dim);
    }

    #[test]
    fn reconnect_now_on_this_computer_does_nothing() {
        let ui = crate::editor::test_support::test_ui_window();
        let backend: Arc<dyn Backend> =
            Arc::new(evanalyzer_app::backends::local::LocalBackend::default());
        let _timer = show_connection(&ui, move || Arc::clone(&backend), || {});

        ui.global::<ConnectionState>().invoke_reconnect_now();

        assert!(ui.global::<ConnectionState>().get_connected());
    }

    /// Without an event loop (the headless test platform has none) the GUI
    /// sets everything up, then reports the error instead of panicking.
    #[test]
    fn starting_the_frontend_without_an_event_loop_returns_instead_of_panicking() {
        use crate::helper::ui_thread::drain_ui_queue;
        crate::editor::test_support::ensure_slint_test_platform();
        crate::helper::ui_thread::fresh_test_queue();
        let owner = crate::editor::test_support::test_project_owner();
        let missing = std::env::temp_dir().join("evanalyzer-gui-no-such-project.evaproj");

        let frontend = Box::new(create().with_project(Some(missing)));
        assert!(frontend.project.is_some());
        frontend.start(owner);
        drain_ui_queue();
    }

    // -- settings dialog ----------------------------------------------------

    /// The settings list scrolls far enough to reach the last setting of the
    /// longest section - in a small window too. (Wrapped descriptions used
    /// to be measured too short, so the scroll range ended early.)
    #[test]
    fn settings_dialog_scrolls_to_its_last_setting() {
        use i_slint_backend_testing::ElementHandle;
        use slint::LogicalPosition;
        use slint::platform::{PointerEventButton, WindowEvent};

        for height in [720.0, 500.0] {
            let ui = crate::editor::test_support::test_ui_window();
            let window = ui.window();
            window.set_size(slint::LogicalSize::new(1280.0, height));
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::ProjectSettings);
            let rect = |label: &str| {
                let e = ElementHandle::find_by_accessible_label(&ui, label)
                    .next()
                    .unwrap_or_else(|| panic!("{label:?} not found"));
                (e.absolute_position(), e.size())
            };

            let (nav, size) = rect("Viewport");
            let at = LogicalPosition::new(nav.x + size.width / 2.0, nav.y + size.height / 2.0);
            window.dispatch_event(WindowEvent::PointerPressed {
                position: at,
                button: PointerEventButton::Left,
            });
            window.dispatch_event(WindowEvent::PointerReleased {
                position: at,
                button: PointerEventButton::Left,
            });

            let scroll_view = ElementHandle::find_by_element_type_name(&ui, "ScrollView")
                .filter(|e| e.absolute_position().x > at.x)
                .max_by(|a, b| a.size().height.total_cmp(&b.size().height))
                .expect("the settings list");
            let (top, view) = (scroll_view.absolute_position(), scroll_view.size());
            let inside = LogicalPosition::new(top.x + view.width / 2.0, top.y + view.height / 2.0);
            for _ in 0..20 {
                window.dispatch_event(WindowEvent::PointerScrolled {
                    position: inside,
                    delta_x: 0.0,
                    delta_y: -300.0,
                });
            }

            let (button, button_size) = rect("Reset to defaults");
            assert!(
                button.y + button_size.height <= top.y + view.height,
                "window height {height}: the last setting ends at {}, the list at {}",
                button.y + button_size.height,
                top.y + view.height
            );
        }
    }
}
