// gui/src/lib.rs

pub use evanalyzer_gui_slint::*;

use evanalyzer_app::{AppHandle, Frontend, ProjectOwner, ProjectWithRuntime, ReaderPool};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_core::ImageReader;
use slint::ComponentHandle;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

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
mod helper;
mod license_text;
mod prelude;
mod third_party_licenses;

// ----------------------------------------------------------------
// UiState - shared across all GUI controllers
// Wraps AppHandle and the Slint window handle
// ----------------------------------------------------------------

pub struct UiState {
    pub app: AppHandle, // cloneable handle - no Arc needed, AppHandle is already Arc inside
    pub ui_handle: slint::Weak<AppWindow>,
    pub results_ui_handle: slint::Weak<ResultsWindow>,
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
            dirty: AtomicBool::new(false),
            undo_stack: Mutex::new(VecDeque::new()),
            redo_stack: Mutex::new(VecDeque::new()),
            last_checkpoint_at: Mutex::new(Instant::now()),
            force_next_checkpoint: AtomicBool::new(false),
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
        slint::invoke_from_event_loop(move || {
            if let Some(w) = ui.upgrade() {
                w.global::<ToolbarState>().set_can_undo(can_undo);
                w.global::<ToolbarState>().set_can_redo(can_redo);
            }
        })
        .ok();
    }

    /// Returns or creates a cached image reader for the given path.
    pub fn get_or_create_reader(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<ImageReader>, InternalErrors> {
        self.app.get_or_create_reader(new_path)
    }

    /// Returns or creates a cached pool of readers for the given path, for
    /// reading multiple channels/Z-slices in parallel.
    pub fn get_or_create_reader_pool(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<ReaderPool>, InternalErrors> {
        self.app.get_or_create_reader_pool(new_path)
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
        self.dirty.store(true, Ordering::Relaxed);
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
        self.dirty.store(false, Ordering::Relaxed);
        let ui = self.ui_handle.clone();
        slint::invoke_from_event_loop(move || {
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
        let title = if dirty {
            format!("*{name} - EVAnalyzer")
        } else {
            format!("{name} - EVAnalyzer")
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
        // Skia has documented first-frame rendering glitches on this backend
        // path (see https://github.com/slint-ui/slint/issues/7845) that
        // FemtoVG doesn't share - left unset so Slint's normal fallback order
        // (Skia -> FemtoVG -> software) picks the more stable renderer instead.
        std::env::set_var("SLINT_SCALE_FACTOR", "1.0");
    }

    let ui = AppWindow::new()?;
    let ui_handle = ui.as_weak();
    let results_ui = ResultsWindow::new()?;
    let results_ui_handle = results_ui.as_weak();

    // Load and apply settings
    load_about_dialog_information(&ui);
    load_user_settings(&ui, &results_ui);

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

/// About dialog content: version comes from the crate version (which the
/// release CI patches to the git tag before building), the rest is read
/// from the host machine once at startup - none of it changes at runtime.
///
/// The CUDA check is probed on a background thread rather than here: loading
/// the CUDA driver and creating a context on first use is slow (commonly
/// hundreds of ms), and nobody looks at the About dialog in the first instant
/// after launch, so there's no reason to make the window wait on it.
fn load_about_dialog_information(ui: &AppWindow) {
    let (cpu_cores, total_ram_bytes) = evanalyzer_core::cpu_ram_diagnostics();
    let info = ui.global::<AppInfoState>();
    info.set_version(env!("CARGO_PKG_VERSION").into());
    info.set_cpu_cores(cpu_cores as i32);
    info.set_ram_total(format!("{:.1} GB", total_ram_bytes as f64 / 1_073_741_824.0).into());
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

    let ui_weak = ui.as_weak();
    std::thread::spawn(move || {
        let cuda_available = evanalyzer_core::cuda_is_available();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<AppInfoState>()
                    .set_cuda_available(cuda_available);
            }
        })
        .ok();
    });
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
}

/// Apply the persisted dark/light preference to both windows. Each window
/// owns its own `Appearance`/`Palette` instance, so this has to be done
/// for both explicitly - see the comment on `Appearance` in style.slint.
fn load_user_settings(ui: &AppWindow, results_ui: &ResultsWindow) {
    let settings = evanalyzer_app::settings::load_app_settings();
    let results_ui_handle = results_ui.as_weak();
    ui.global::<Appearance>().invoke_apply(settings.dark_mode);
    results_ui
        .global::<Appearance>()
        .invoke_apply(settings.dark_mode);

    let results_ui_handle = results_ui_handle.clone();
    ui.global::<Appearance>().on_dark_mode_toggled(move |dark| {
        evanalyzer_app::settings::save_app_settings(&evanalyzer_app::settings::AppSettings {
            dark_mode: dark,
        });
        if let Some(results_ui) = results_ui_handle.upgrade() {
            results_ui.global::<Appearance>().invoke_apply(dark);
        }
    });
}
