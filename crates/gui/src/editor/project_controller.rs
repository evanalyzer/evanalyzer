use crate::AppWindow;
use crate::DialogType;
use crate::GlobalAppState;
use crate::ImagesListState;
use crate::PipelinesPanelState;
use crate::ProjectTemplateDef;
use crate::ProjectTemplateState;
use crate::TagFilterChip;
use crate::ToolbarState;
use crate::UiState;
use crate::UnsavedChangesState;
use crate::WarningState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::pipelines_controller::PipelinesController;
use crate::editor::project_settings_controller::ProjectSettingsController;
use crate::editor::results_list_controller::ResultsListController;
use crate::editor::template_controller::TemplateController;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::extensions::project_ext::SaveProjectActions;
use evanalyzer_app::templates::{load_project_template_from_file, load_project_templates};
use evanalyzer_cfg::LEGACY_PROJECT_FILE_EXTENSION;
use evanalyzer_cfg::PROJECT_FILE_EXTENSIONS;
use evanalyzer_cfg::PROJECT_FILE_TEMPLATE_EXTENSIONS;
use evanalyzer_cfg::settings::templates::ProjectTemplate;
use evanalyzer_core::SUPPORTED_IMAGE_FORMATS;
use log::{info, warn};
use slint::ComponentHandle;
use slint::{ModelRc, SharedString, VecModel};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Sentinel shown as the first entry of the category `ComboBox`; selecting it
/// clears the category filter rather than matching a real category name.
const ALL_CATEGORIES: &str = "All categories";

/// Current state of the "New from Project Template" picker's category/tag
/// filters. Empty `category` and `tags` mean "no filter" (show everything).
#[derive(Default, Clone)]
struct TemplateFilter {
    category: String,
    tags: BTreeSet<String>,
}

/// An action that would discard the current project's unsaved changes if run
/// immediately - gated behind [`ProjectController::guard_discard`], which
/// runs it right away if the project is clean, or stashes it and prompts the
/// user (Save / Discard / Cancel) if it isn't.
enum PendingAction {
    OpenProject(PathBuf),
    ImportLegacy(PathBuf),
    OpenProjectTemplate(PathBuf),
    Quit,
}

pub struct ProjectController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) image_list_controller: Arc<ImagesListController>,
    pub(crate) project_settings_controller: Arc<ProjectSettingsController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
    pub(crate) pipelines_controller: Arc<PipelinesController>,
    pub(crate) results_list_controller: Arc<ResultsListController>,
    pub(crate) template_controller: Arc<TemplateController>,
    /// Project templates currently shown in the "New from Project Template"
    /// dialog. Reloaded from disk whenever the dialog is opened.
    project_templates: Mutex<Vec<ProjectTemplate>>,
    /// Category/tag filter currently applied to `project_templates` in the
    /// picker. Reset whenever the dialog is (re)opened.
    template_filter: Mutex<TemplateFilter>,
    /// The open/import/quit action waiting on the unsaved-changes dialog's
    /// answer, if any (see [`PendingAction`]).
    pending_action: Mutex<Option<PendingAction>>,
}

impl ProjectController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        image_list_controller: Arc<ImagesListController>,
        project_settings_controller: Arc<ProjectSettingsController>,
        classification_controller: Arc<ClassificationController>,
        pipelines_controller: Arc<PipelinesController>,
        results_list_controller: Arc<ResultsListController>,
        template_controller: Arc<TemplateController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            image_list_controller,
            project_settings_controller,
            classification_controller,
            pipelines_controller,
            results_list_controller,
            template_controller,
            project_templates: Mutex::new(Vec::new()),
            template_filter: Mutex::new(TemplateFilter::default()),
            pending_action: Mutex::new(None),
        }
    }

    /// Initializes and opens a new project based on the provided image path.
    ///
    /// This is a top-level coordination method that:
    /// 1. Tears down the current project session and clears existing metadata.
    /// 2. Resolves the directory structure for the new project location.
    /// 3. Resets the Slint UI state (lists, histograms, and viewports) to reflect
    ///    the new project's context.
    ///
    /// # Arguments
    /// * `project_path` - A string slice representing the path to the project file
    ///
    /// # Threading
    /// This method typically triggers a chain of synchronous state resets followed
    /// by an asynchronous filesystem scan to populate the new image list.
    pub fn open_new_project(self: Arc<Self>, project_path: &PathBuf) {
        if let Err(e) = self.app_state.load_project(&project_path) {
            warn!("Could not open project {:?}: {}", project_path, e);
            self.show_warning(
                "Cannot open project",
                &format!("Failed to open '{}': {e}", project_path.display()),
            );
            return;
        }

        let image_root_dir = {
            let mut project = self.app_state.get_project_write();
            project.rest_current_image_path();
            project.images.root.clone().unwrap_or_default()
        };

        // Do all the project load tasks here
        self.image_list_controller.sync_image_list_to_slint();
        self.project_settings_controller
            .sync_project_settings_to_slint();
        self.classification_controller
            .sync_classification_to_slint();
        self.pipelines_controller.sync_pipelines_to_slint();
        self.results_list_controller.sync_results_files_to_slint();
        // A freshly opened project has no unsaved changes yet - also updates
        // the window title to the newly opened file (see `clear_dirty`).
        self.app_state.clear_dirty();

        let ui_weak = self.ui.clone();
        let image_root_dir_str = image_root_dir.to_string_lossy().into_owned();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ImagesListState>()
                    .set_act_image_root_dir(image_root_dir_str.into());
            }
        })
        .ok();
        // Special handling. We set the image root dir to check if the images exist, else a root dir selection dialof will be opened
        self.image_list_controller
            .set_new_image_root(&image_root_dir);
        info!("Project opened!")
    }

    /// Imports an old (`.icproj`) project into a brand-new, unsaved project.
    ///
    /// Unlike [`Self::open_new_project`], the converted project has no recorded
    /// image list yet (the old format only stored an image *folder*, discovered
    /// at runtime) - if the old project's image folder resolves to a real
    /// directory next to the `.icproj` file, it is scanned immediately so the
    /// imported project is usable right away. Any commands or fields the old
    /// project used with no equivalent in this format are reported to the user
    /// as a non-blocking warning, listing what was skipped or approximated.
    pub fn import_legacy_project_file(self: Arc<Self>, legacy_path: &PathBuf) {
        let (warnings, legacy_image_folder) =
            match self.app_state.import_legacy_project(legacy_path) {
                Ok(result) => result,
                Err(e) => {
                    warn!("Could not import legacy project {:?}: {}", legacy_path, e);
                    self.show_warning(
                        "Cannot import legacy project",
                        &format!("Failed to import '{}': {e}", legacy_path.display()),
                    );
                    return;
                }
            };

        self.app_state.mark_dirty();

        let image_root_dir = legacy_image_folder
            .filter(|folder| !folder.is_empty())
            .and_then(|folder| {
                let resolved = legacy_path.parent().unwrap_or(legacy_path).join(&folder);
                resolved.is_dir().then_some(resolved)
            });

        if let Some(root) = &image_root_dir {
            // Scan off-lock (see `images_list_controller::scan_image_root_for_images`
            // for why) - only the fast in-memory apply below needs the write guard.
            let found_images =
                evanalyzer_app::extensions::project_ext::collect_images_at_root(root);
            let mut project = self.app_state.get_project_write();
            project.images.root = Some(root.clone());
            project.apply_scanned_images(found_images);
        }

        // Do all the project load tasks here, mirroring `open_new_project`.
        self.image_list_controller.sync_image_list_to_slint();
        self.project_settings_controller
            .sync_project_settings_to_slint();
        self.classification_controller
            .sync_classification_to_slint();
        self.pipelines_controller.sync_pipelines_to_slint();
        self.results_list_controller.sync_results_files_to_slint();

        if let Some(root) = image_root_dir {
            let ui_weak = self.ui.clone();
            let image_root_dir_str = root.to_string_lossy().into_owned();
            slint::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    ui_ready
                        .global::<ImagesListState>()
                        .set_act_image_root_dir(image_root_dir_str.into());
                }
            })
            .ok();
        }

        info!("Legacy project imported ({} warning(s))", warnings.len());
        if !warnings.is_empty() {
            let message = format!(
                "The project imported successfully, but {} item(s) from the old format had no exact equivalent and were skipped or approximated:\n\n- {}\n\nReview the affected pipeline steps before running this project. Use \"Save As\" to keep this project in the new format.",
                warnings.len(),
                warnings.join("\n- ")
            );
            self.show_warning_info("Legacy project imported with caveats", &message);
        }
    }

    /// Loads a `ProjectTemplate` from an arbitrary path (picked via the
    /// generic "Open" dialog, unlike [`Self::apply_project_template`] which
    /// picks from the bundled/user templates folders) and applies it to the
    /// current project, same as confirming it in the "New from Project
    /// Template" picker would.
    pub fn open_project_template_file(self: Arc<Self>, path: &PathBuf) {
        let template = match load_project_template_from_file(path) {
            Ok(template) => template,
            Err(e) => {
                warn!("Could not load project template {:?}: {}", path, e);
                self.show_warning(
                    "Cannot open project template",
                    &format!("Failed to load '{}': {e}", path.display()),
                );
                return;
            }
        };

        let manager = self.clone();
        slint::invoke_from_event_loop(move || {
            manager.apply_project_template_value(&template);
        })
        .ok();
    }

    /// Shows the generic warning dialog (error style) with `title`/`message`.
    fn show_warning(&self, title: &str, message: &str) {
        self.show_warning_with_style(title, message, false);
    }

    /// Shows the generic warning dialog (informational style) with `title`/`message`.
    fn show_warning_info(&self, title: &str, message: &str) {
        self.show_warning_with_style(title, message, true);
    }

    fn show_warning_with_style(&self, title: &str, message: &str, info: bool) {
        let title = title.to_owned();
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(info);
                warning.set_title(title.into());
                warning.set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        }) {
            warn!("Failed to show warning dialog: {e}");
        }
    }

    /// Attach UI callbacks related to image operations.
    ///
    /// This method registers handlers on the global ImagesListState (currently the
    /// `on_image_filter_text_changed` callback) so that UI-driven image filter actions
    /// are propagated to the background manager and the UI is refreshed on the
    /// Slint event loop.
    ///
    /// Behavior:
    /// - Clones required handles (UI and application state) so the closures can be
    ///   stored and invoked later.
    /// - The registered callback captures a worker/project manager and a weak UI
    ///   handle. It schedules work on the Slint event loop using
    ///   `slint::invoke_from_event_loop`.
    /// - Inside the event loop it attempts to upgrade the weak UI handle; if the
    ///   UI still exists it calls `update_image_list_in_sync` on the manager to
    ///   update the image list to reflect the applied filter.
    ///
    /// Notes:
    /// - The function is non-blocking from the caller's perspective; updates are
    ///   dispatched to the event loop.
    /// - If the UI has been dropped the callback is a no-op (the weak upgrade
    ///   fails). Any errors from scheduling are ignored via `.ok()`.
    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // Open file
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>().on_open_file_clicked(move || {
                manager.open_file_handler();
            });

            // Save (saves to existing path, prompts if none)
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>().on_save_file_clicked(move || {
                manager.save_project();
            });

            // Save As (always prompts for a new path)
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>()
                .on_save_as_file_clicked(move || {
                    manager.save_project_as_handler();
                });

            // Save project as template
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>()
                .on_save_project_as_template_clicked(move || {
                    let name = manager.app_state.get_project().metadata.name.clone();
                    manager
                        .template_controller
                        .start_project_template_save(name);
                });

            // New from project template - open the picker dialog
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>()
                .on_new_from_project_template_clicked(move || {
                    manager.open_project_template_dialog();
                });

            // Project template picker: select - update detail pane
            let manager = Arc::clone(self);
            ui.global::<ProjectTemplateState>().on_select(move |id| {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                let templates = manager.project_templates.lock().expect("Poisoned");
                if let Some(template) = templates.get(id as usize) {
                    let detail = project_template_to_def(id, template);
                    let picker = ui.global::<ProjectTemplateState>();
                    picker.set_detail(detail);
                    picker.set_has_detail(true);
                }
            });

            // Project template picker: confirm - apply template to the project
            let manager = Arc::clone(self);
            ui.global::<ProjectTemplateState>().on_confirm(move |id| {
                manager.apply_project_template(id);
            });

            // Project template picker: cancel - close dialog
            let manager = Arc::clone(self);
            ui.global::<ProjectTemplateState>().on_cancel(move || {
                if let Some(ui) = manager.ui.upgrade() {
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::None);
                }
            });

            // Project template picker: category filter changed
            let manager = Arc::clone(self);
            ui.global::<ProjectTemplateState>()
                .on_select_category(move |category| {
                    let mut filter = manager.template_filter.lock().expect("Poisoned");
                    filter.category = if category.as_str() == ALL_CATEGORIES {
                        String::new()
                    } else {
                        category.to_string()
                    };
                    drop(filter);
                    manager.refresh_filtered_templates();
                });

            // Project template picker: tag chip toggled
            let manager = Arc::clone(self);
            ui.global::<ProjectTemplateState>()
                .on_toggle_tag(move |tag| {
                    let tag = tag.to_string();
                    let mut filter = manager.template_filter.lock().expect("Poisoned");
                    if !filter.tags.remove(&tag) {
                        filter.tags.insert(tag);
                    }
                    drop(filter);
                    manager.refresh_filtered_templates();
                });

            // Open website in the system browser
            ui.global::<ToolbarState>().on_open_website(|| {
                std::thread::spawn(|| {
                    #[cfg(target_os = "linux")]
                    let _ = std::process::Command::new("xdg-open")
                        .arg("https://evanalyzer.org")
                        .spawn();
                    #[cfg(target_os = "macos")]
                    let _ = std::process::Command::new("open")
                        .arg("https://evanalyzer.org")
                        .spawn();
                    #[cfg(target_os = "windows")]
                    let _ = std::process::Command::new("cmd")
                        .args(["/c", "start", "", "https://evanalyzer.org"])
                        .spawn();
                });
            });

            // Unsaved-changes dialog: "Save" - save first, then only run the
            // pending action (if any) once the save actually succeeded. If
            // the user cancels the save-as file picker, or the save fails,
            // the pending action is left in place and nothing is discarded.
            let manager = Arc::clone(self);
            ui.global::<UnsavedChangesState>().on_save(move || {
                let manager = Arc::clone(&manager);
                manager.clone().save_project_then(move |saved| {
                    if saved {
                        if let Some(action) =
                            manager.pending_action.lock().expect("Poisoned").take()
                        {
                            manager.run_pending_action(action);
                        }
                    }
                });
            });

            // Unsaved-changes dialog: "Discard changes" - drop the pending
            // action's guard and run it, losing the unsaved edits.
            let manager = Arc::clone(self);
            ui.global::<UnsavedChangesState>().on_discard(move || {
                if let Some(action) = manager.pending_action.lock().expect("Poisoned").take() {
                    manager.run_pending_action(action);
                }
            });

            // Unsaved-changes dialog: "Cancel" - drop the pending action,
            // leave the current project open untouched.
            let manager = Arc::clone(self);
            ui.global::<UnsavedChangesState>().on_cancel(move || {
                *manager.pending_action.lock().expect("Poisoned") = None;
            });

            // Closing the window: if there are unsaved changes, keep the
            // window open and route the close through the same
            // Save/Discard/Cancel guard as opening another project - the
            // default behaviour (silently hide the window) would otherwise
            // discard unsaved work with no confirmation.
            let manager = Arc::clone(self);
            ui.window().on_close_requested(move || {
                if manager.app_state.is_dirty() {
                    manager.guard_discard(PendingAction::Quit);
                    slint::CloseRequestResponse::KeepWindowShown
                } else {
                    slint::CloseRequestResponse::HideWindow
                }
            });
        }
    }

    /// Evaluates the file type of a given path and dispatches the appropriate loading sequence.
    ///
    /// This is the primary entry point for file interactions. It performs a
    /// preliminary check on the file extension or header to determine if the
    /// target is:
    /// 1. **An Image**: Triggers the standard image viewing/processing pipeline.
    /// 2. **A Result**: Loads previously saved analysis or output data.
    /// 3. **A Project**: Restores a full workspace session, including root paths and state.
    /// 4. **A Template**: Applies predefined configurations or filter stacks to the current view.
    ///
    /// # Threading
    /// Initial file-type identification is synchronous. Depending on the identified
    /// type, it may spawn background threads for heavy I/O (e.g., loading large
    /// project manifests or decoding high-resolution images).
    fn open_file_handler(self: &Arc<Self>) {
        let mut allowed_files = SUPPORTED_IMAGE_FORMATS.to_vec();
        allowed_files.push(PROJECT_FILE_EXTENSIONS);
        allowed_files.push(LEGACY_PROJECT_FILE_EXTENSION);
        allowed_files.push(PROJECT_FILE_TEMPLATE_EXTENSIONS);

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Supported Files", &allowed_files)
            .add_filter("Image Files", &SUPPORTED_IMAGE_FORMATS)
            .add_filter("Project Files", &[PROJECT_FILE_EXTENSIONS])
            .add_filter("Legacy Project Files", &[LEGACY_PROJECT_FILE_EXTENSION])
            .add_filter(
                "Project Template Files",
                &[PROJECT_FILE_TEMPLATE_EXTENSIONS],
            )
            .pick_file()
        {
            let manager = Arc::clone(self);

            std::thread::spawn(move || {
                let ext = path.extension().and_then(|ext| ext.to_str());

                if ext == Some(PROJECT_FILE_EXTENSIONS) {
                    manager.guard_discard(PendingAction::OpenProject(path));
                } else if ext == Some(LEGACY_PROJECT_FILE_EXTENSION) {
                    manager.guard_discard(PendingAction::ImportLegacy(path));
                } else if ext == Some(PROJECT_FILE_TEMPLATE_EXTENSIONS) {
                    manager.guard_discard(PendingAction::OpenProjectTemplate(path));
                } else {
                    manager.image_list_controller.open_new_image(&path);
                }
            });
        }
    }

    /// Serializes the current project state and persists it to the filesystem.
    ///
    /// This method captures the "Source of Truth" from the application state,
    /// including image lists, metadata, and user-defined settings, and writes
    /// it to the project's configuration file.
    ///
    /// # Threading
    /// To prevent UI "stutter" during disk I/O, the serialization and file-writing
    /// process is typically executed on a background thread. The UI is notified
    /// once the save operation is successfully committed.
    ///
    /// # Reliability
    /// In a production environment, this should ideally implement an "atomic save"
    /// pattern (writing to a temporary file first) to prevent data corruption
    /// in the event of a power failure or crash during the write process.
    /// Always shows a Save As dialog, regardless of whether a project path exists.
    fn save_project_as_handler(self: &Arc<Self>) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Project files", &[PROJECT_FILE_EXTENSIONS])
            .save_file()
        {
            let in_thread = self.clone();
            std::thread::spawn(move || {
                // Bind to an owned `Result` first so the write guard from
                // `get_project_write()` is dropped at this `let` (not kept
                // alive across the match arms below, which - as a `match`
                // scrutinee temporary - it otherwise would be). `clear_dirty()`
                // calls `set_window_title()`, which takes a *read* lock on the
                // same project `RwLock`; still holding the write guard there
                // deadlocks the thread against itself.
                let result = in_thread
                    .app_state
                    .get_project_write()
                    .save_project_as(&path);
                match result {
                    Ok(_) => {
                        info!("Project saved as: {}", path.display());
                        in_thread.app_state.clear_dirty();
                    }
                    Err(msg) => {
                        warn!("Project not saved: {}", msg);
                    }
                }
            });
        }
    }

    fn save_project(self: &Arc<Self>) {
        self.save_project_then(|_saved| {});
    }

    /// Saves the project (prompting for a path first if none is set yet,
    /// same as [`Self::save_project_as_handler`]) and calls `on_done(true)`
    /// once the save actually succeeded, or `on_done(false)` if it failed or
    /// the user cancelled the file picker. Used both for the plain "Save"
    /// button and to resume a guarded action after the user picks "Save" on
    /// the unsaved-changes dialog.
    ///
    /// # Threading
    /// The path check below is a cheap in-memory read; everything that
    /// touches disk (serialization, the write, and - if needed - the file
    /// picker's blocking dialog) runs on a background thread so the UI
    /// thread never blocks on save I/O, mirroring `save_project_as_handler`.
    fn save_project_then(self: &Arc<Self>, on_done: impl FnOnce(bool) + Send + 'static) {
        let has_path = self
            .app_state
            .get_project()
            .tmp_settings
            .current_project
            .is_some();

        if !has_path {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Project files", &[PROJECT_FILE_EXTENSIONS])
                .save_file()
            {
                let in_thread = self.clone();
                std::thread::spawn(move || {
                    // See the comment in `save_project_as_handler` above -
                    // same fix: drop the write guard at this `let` instead of
                    // holding it across the match arms, where `clear_dirty()`
                    // would deadlock trying to re-acquire it for reading.
                    let result = in_thread
                        .app_state
                        .get_project_write()
                        .save_project_as(&path);
                    let ok = result.is_ok();
                    match result {
                        Ok(_) => {
                            info!("Project saved: {}", path.display());
                            in_thread.app_state.clear_dirty();
                        }
                        Err(msg) => {
                            warn!("Project not saved: {}", msg);
                        }
                    }
                    on_done(ok);
                });
            } else {
                on_done(false);
            }
            return;
        }

        let in_thread = self.clone();
        std::thread::spawn(move || {
            // Same deadlock-avoidance shape as above: `result` is an owned
            // value, so the write guard from `get_project_write()` is
            // dropped at this `let`, before `clear_dirty()` (which takes a
            // read lock) runs in the match arm below.
            let result = in_thread.app_state.get_project_write().save_project();
            let ok = result == SaveProjectActions::Success;
            match result {
                SaveProjectActions::Success => {
                    info!("Project saved");
                    in_thread.app_state.clear_dirty();
                }
                SaveProjectActions::PleaseSelectFile => {
                    warn!("Project has no path to save to");
                }
                SaveProjectActions::Error => {
                    warn!("Could not save project");
                }
            }
            on_done(ok);
        });
    }

    /// Runs `action` immediately if the project has no unsaved changes;
    /// otherwise stashes it and shows the unsaved-changes confirmation
    /// dialog, whose Save/Discard/Cancel buttons resolve it (wired in
    /// [`Self::attach_callbacks`]).
    fn guard_discard(self: &Arc<Self>, action: PendingAction) {
        if !self.app_state.is_dirty() {
            self.run_pending_action(action);
            return;
        }
        *self.pending_action.lock().expect("Poisoned") = Some(action);
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::UnsavedChangesConfirm);
            }
        })
        .ok();
    }

    /// Executes a [`PendingAction`], safe to call from either the UI thread
    /// or a background thread: I/O-bound actions dispatch their own
    /// background thread, `Quit` dispatches onto the UI event loop.
    fn run_pending_action(self: &Arc<Self>, action: PendingAction) {
        match action {
            PendingAction::OpenProject(path) => {
                let manager = self.clone();
                std::thread::spawn(move || manager.open_new_project(&path));
            }
            PendingAction::ImportLegacy(path) => {
                let manager = self.clone();
                std::thread::spawn(move || manager.import_legacy_project_file(&path));
            }
            PendingAction::OpenProjectTemplate(path) => {
                let manager = self.clone();
                std::thread::spawn(move || manager.open_project_template_file(&path));
            }
            PendingAction::Quit => {
                let ui_weak = self.ui.clone();
                slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let _ = ui.window().hide();
                    }
                })
                .ok();
            }
        }
    }

    /// Opens the "New from Project Template" dialog.
    ///
    /// The dialog opens immediately with an empty list; the available
    /// templates are loaded from disk in the background and populated once
    /// ready, so a slow filesystem doesn't delay opening the dialog.
    fn open_project_template_dialog(self: &Arc<Self>) {
        let manager = self.clone();
        *self.template_filter.lock().expect("Poisoned") = TemplateFilter::default();
        if let Some(ui) = self.ui.upgrade() {
            let picker = ui.global::<ProjectTemplateState>();
            picker.set_selected_id(-1);
            picker.set_has_detail(false);
            picker.set_templates(ModelRc::new(VecModel::from(
                Vec::<ProjectTemplateDef>::new(),
            )));
            picker.set_categories(ModelRc::new(VecModel::from(vec![SharedString::from(
                ALL_CATEGORIES,
            )])));
            picker.set_tag_chips(ModelRc::new(VecModel::from(Vec::<TagFilterChip>::new())));
            picker.set_filters_active(false);
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::ProjectTemplate);
        }
        std::thread::spawn(move || {
            let templates: Vec<ProjectTemplate> = load_project_templates()
                .into_iter()
                .map(|(_path, template)| template)
                .collect();
            *manager.project_templates.lock().expect("Poisoned") = templates;

            if let Err(e) = slint::invoke_from_event_loop(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                if ui.global::<GlobalAppState>().get_active_dialog() != DialogType::ProjectTemplate
                {
                    return;
                }
                manager.refresh_filtered_templates();
            }) {
                warn!("Failed to populate project template picker: {}", e);
            }
        });
    }

    /// Recomputes the category/tag facets and the filtered template list from
    /// `project_templates` + `template_filter`, and pushes them to the picker.
    ///
    /// Called whenever the dialog is (re)opened or a filter changes. `id`s in
    /// the resulting `ProjectTemplateDef`s are always indices into the full
    /// (unfiltered) `project_templates`, since that's what `on_select` /
    /// `on_confirm` look up by.
    fn refresh_filtered_templates(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let templates = self.project_templates.lock().expect("Poisoned");
        let filter = self.template_filter.lock().expect("Poisoned").clone();

        let mut categories_seen: BTreeSet<String> = BTreeSet::new();
        let mut tags_seen: BTreeSet<String> = BTreeSet::new();
        for t in templates.iter() {
            if !t.meta.category.is_empty() {
                categories_seen.insert(t.meta.category.clone());
            }
            tags_seen.extend(t.meta.tags.iter().filter(|s| !s.is_empty()).cloned());
        }

        let mut categories: Vec<SharedString> = vec![SharedString::from(ALL_CATEGORIES)];
        categories.extend(categories_seen.into_iter().map(SharedString::from));

        let tag_chips: Vec<TagFilterChip> = tags_seen
            .into_iter()
            .map(|name| {
                let active = filter.tags.contains(&name);
                TagFilterChip {
                    name: name.into(),
                    active,
                }
            })
            .collect();

        let defs: Vec<ProjectTemplateDef> = templates
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                let category_ok = filter.category.is_empty() || t.meta.category == filter.category;
                let tags_ok = filter.tags.is_empty()
                    || t.meta.tags.iter().any(|tag| filter.tags.contains(tag));
                category_ok && tags_ok
            })
            .map(|(idx, t)| project_template_to_def(idx as i32, t))
            .collect();

        let picker = ui.global::<ProjectTemplateState>();
        let still_visible = defs.iter().any(|d| d.id == picker.get_selected_id());
        if !still_visible {
            picker.set_selected_id(-1);
            picker.set_has_detail(false);
        }
        picker.set_categories(ModelRc::new(VecModel::from(categories)));
        picker.set_tag_chips(ModelRc::new(VecModel::from(tag_chips)));
        picker.set_filters_active(!filter.category.is_empty() || !filter.tags.is_empty());
        picker.set_templates(ModelRc::new(VecModel::from(defs)));
    }

    /// Replaces the current project's classification, plate and pipeline
    /// settings with the ones from the selected project template, then
    /// re-syncs the affected panels.
    fn apply_project_template(self: &Arc<Self>, id: i32) {
        let template = {
            let templates = self.project_templates.lock().expect("Poisoned");
            templates.get(id as usize).cloned()
        };
        let Some(template) = template else {
            warn!(
                "project template picker confirm: unknown template id {}",
                id
            );
            return;
        };
        self.apply_project_template_value(&template);
    }

    /// Applies a `ProjectTemplate` (already loaded, from wherever) to the
    /// current project and refreshes every affected UI panel. Must run on
    /// the UI thread - touches Slint globals directly, not just through the
    /// `sync_*_to_slint` methods.
    fn apply_project_template_value(self: &Arc<Self>, template: &ProjectTemplate) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        let first_pipeline_id = {
            let mut project = self.app_state.get_project_write();
            project.apply_project_template(template);
            project.pipelines.first().map(|p| p.id)
        };
        self.app_state.mark_dirty();

        self.project_settings_controller
            .sync_project_settings_to_slint();
        self.classification_controller
            .sync_classification_to_slint();
        self.pipelines_controller.sync_pipelines_to_slint();

        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);

        match first_pipeline_id {
            Some(pid) => {
                self.pipelines_controller
                    .sync_steps_of_selected_pipeline_to_slint(pid, true);
                let ui_weak = self.ui.clone();
                slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.global::<PipelinesPanelState>()
                            .set_active_pipeline_id(pid.0 as i32);
                    }
                })
                .ok();
            }
            None => {
                let ui_weak = self.ui.clone();
                slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let ps = ui.global::<PipelinesPanelState>();
                        ps.set_active_pipeline_id(0);
                        ps.set_active_pipeline_name("".into());
                        ps.set_active_pipeline_image_source("".into());
                        ps.set_active_commands(ModelRc::default());
                    }
                })
                .ok();
            }
        }
    }
}

/// Builds the `ProjectTemplateDef` shown in the "New from Project Template"
/// dialog for a loaded `ProjectTemplate`.
fn project_template_to_def(id: i32, template: &ProjectTemplate) -> ProjectTemplateDef {
    let author = template.meta.authors.first().cloned().unwrap_or_default();
    let co_authors = template.meta.authors.get(1..).unwrap_or(&[]).join(", ");

    let tags: Vec<SharedString> = template
        .meta
        .tags
        .iter()
        .cloned()
        .map(SharedString::from)
        .collect();

    ProjectTemplateDef {
        id,
        name: template.meta.name.clone().into(),
        short_description: template.meta.short_description.clone().into(),
        description: template.meta.description.clone().into(),
        author: author.into(),
        co_authors: co_authors.into(),
        organization: template.meta.author_organization.clone().into(),
        creation_time: template
            .meta
            .creation_time
            .format("%Y-%m-%d")
            .to_string()
            .into(),
        pipeline_count: template.pipelines.len() as i32,
        category: template.meta.category.clone().into(),
        tags: ModelRc::new(VecModel::from(tags)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::classification_settings::ClassificationSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use evanalyzer_cfg::settings::plate_settings::PlateSettings;
    use slint::Model;

    fn template_with_meta(meta: MetaData, pipeline_count: usize) -> ProjectTemplate {
        ProjectTemplate {
            meta,
            classification: ClassificationSettings::default(),
            plate: PlateSettings::default(),
            pipelines: (0..pipeline_count)
                .map(|_| evanalyzer_cfg::settings::templates::PipelineTemplate {
                    meta: MetaData::default(),
                    pipeline_steps: Vec::new(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn project_template_to_def_carries_the_given_id_and_pipeline_count() {
        let template = template_with_meta(MetaData::default(), 3);
        let def = project_template_to_def(7, &template);

        assert_eq!(def.id, 7);
        assert_eq!(def.pipeline_count, 3);
    }

    #[test]
    fn project_template_to_def_uses_the_first_author_as_the_primary_author() {
        let meta = MetaData {
            authors: vec!["Ada Lovelace".into()],
            ..Default::default()
        };
        let def = project_template_to_def(0, &template_with_meta(meta, 0));

        assert_eq!(def.author.as_str(), "Ada Lovelace");
        assert_eq!(def.co_authors.as_str(), "");
    }

    #[test]
    fn project_template_to_def_joins_remaining_authors_as_co_authors() {
        let meta = MetaData {
            authors: vec![
                "Ada Lovelace".into(),
                "Alan Turing".into(),
                "Grace Hopper".into(),
            ],
            ..Default::default()
        };
        let def = project_template_to_def(0, &template_with_meta(meta, 0));

        assert_eq!(def.author.as_str(), "Ada Lovelace");
        assert_eq!(def.co_authors.as_str(), "Alan Turing, Grace Hopper");
    }

    #[test]
    fn project_template_to_def_leaves_author_empty_when_no_authors_are_set() {
        let def = project_template_to_def(0, &template_with_meta(MetaData::default(), 0));

        assert_eq!(def.author.as_str(), "");
        assert_eq!(def.co_authors.as_str(), "");
    }

    #[test]
    fn project_template_to_def_formats_creation_time_as_year_month_day() {
        let meta = MetaData {
            creation_time: chrono::DateTime::parse_from_rfc3339("2026-03-05T12:34:56Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            ..Default::default()
        };
        let def = project_template_to_def(0, &template_with_meta(meta, 0));

        assert_eq!(def.creation_time.as_str(), "2026-03-05");
    }

    #[test]
    fn project_template_to_def_copies_tags_into_the_slint_model() {
        let meta = MetaData {
            tags: vec!["cells".to_string(), "uptake".to_string()],
            ..Default::default()
        };
        let def = project_template_to_def(0, &template_with_meta(meta, 0));

        let tags: Vec<String> = def.tags.iter().map(|t| t.to_string()).collect();
        assert_eq!(tags, vec!["cells".to_string(), "uptake".to_string()]);
    }

    // -- guard_discard / run_pending_action -----------------------------------

    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::object_list_controller::ObjectListController;
    use crate::editor::results_table_controller::ResultsTableController;
    use crate::editor::test_support::test_ui_state;
    use crate::editor::viewport_controller::ViewportController;

    fn make_controller() -> (Arc<UiState>, Arc<ProjectController>) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_list_controller = Arc::new(ImagesListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(HistogramController::new(
                slint::Weak::default(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            Arc::new(ImageMetaController::new(
                slint::Weak::default(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            object_list_controller.clone(),
        ));
        let project_settings_controller = Arc::new(ProjectSettingsController::new(
            slint::Weak::default(),
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let classification_controller = Arc::new(ClassificationController::new(
            slint::Weak::default(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
        ));
        let template_controller = Arc::new(TemplateController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let pipelines_controller = Arc::new(PipelinesController::new(
            slint::Weak::default(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
            template_controller.clone(),
        ));
        let results_table_controller = Arc::new(ResultsTableController::new(
            slint::Weak::default(),
            ui_state.clone(),
            image_list_controller.clone(),
            project_settings_controller.clone(),
        ));
        let results_list_controller = Arc::new(ResultsListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            results_table_controller,
        ));
        let controller = Arc::new(ProjectController::new(
            slint::Weak::default(),
            ui_state.clone(),
            image_list_controller,
            project_settings_controller,
            classification_controller,
            pipelines_controller,
            results_list_controller,
            template_controller,
        ));
        (ui_state, controller)
    }

    #[test]
    fn guard_discard_stashes_the_action_when_the_project_is_dirty() {
        let (ui_state, controller) = make_controller();
        ui_state.mark_dirty();

        controller.guard_discard(PendingAction::Quit);

        assert!(
            controller.pending_action.lock().unwrap().is_some(),
            "a dirty project must stash the action instead of running it"
        );
    }

    #[test]
    fn guard_discard_runs_immediately_without_stashing_when_the_project_is_clean() {
        let (ui_state, controller) = make_controller();
        assert!(!ui_state.is_dirty());

        // `Quit` runs synchronously via `invoke_from_event_loop` (a safe
        // no-op without a live UI) rather than spawning a background thread,
        // so this is deterministic to assert on.
        controller.guard_discard(PendingAction::Quit);

        assert!(
            controller.pending_action.lock().unwrap().is_none(),
            "a clean project must run the action immediately, not stash it"
        );
    }

    // -- open_new_project -------------------------------------------------------

    fn temp_project_file(
        settings: &evanalyzer_cfg::settings::project_settings::ProjectSettings,
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "evanalyzer_project_controller_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.evaproj");
        std::fs::write(&path, serde_json::to_string_pretty(settings).unwrap()).unwrap();
        path
    }

    #[test]
    fn open_new_project_on_a_nonexistent_file_shows_a_warning_and_leaves_the_project_untouched() {
        let (ui_state, controller) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            project.metadata.name = "should not be overwritten".to_string();
        }

        controller
            .clone()
            .open_new_project(&PathBuf::from("/nonexistent/does_not_exist.evaproj"));

        assert_eq!(
            ui_state.get_project().metadata.name,
            "should not be overwritten"
        );
    }

    #[test]
    fn open_new_project_loads_the_file_and_clears_the_dirty_flag() {
        let (ui_state, controller) = make_controller();
        ui_state.mark_dirty();
        let mut settings = evanalyzer_cfg::settings::project_settings::ProjectSettings::default();
        settings.metadata.name = "Loaded Project".to_string();
        let path = temp_project_file(&settings);

        controller.clone().open_new_project(&path);

        let project = ui_state.get_project();
        assert_eq!(project.metadata.name, "Loaded Project");
        assert!(
            !ui_state.is_dirty(),
            "opening a project must clear the dirty flag"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn open_new_project_resets_the_current_image_selection() {
        let (ui_state, controller) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            project.tmp_settings.current_image = Some(PathBuf::from("/some/old/image.tif"));
        }
        let path = temp_project_file(
            &evanalyzer_cfg::settings::project_settings::ProjectSettings::default(),
        );

        controller.clone().open_new_project(&path);

        assert!(ui_state.get_project().tmp_settings.current_image.is_none());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    // -- import_legacy_project_file ----------------------------------------------

    #[test]
    fn import_legacy_project_file_on_a_nonexistent_file_shows_a_warning_and_leaves_the_project_untouched()
     {
        let (ui_state, controller) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            project.metadata.name = "should not be overwritten".to_string();
        }

        controller
            .clone()
            .import_legacy_project_file(&PathBuf::from("/nonexistent/does_not_exist.icproj"));

        assert_eq!(
            ui_state.get_project().metadata.name,
            "should not be overwritten"
        );
        assert!(
            !ui_state.is_dirty(),
            "a failed import must not mark the project dirty"
        );
    }

    // -- save_project_then --------------------------------------------------------

    fn temp_dir_for(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "evanalyzer_project_controller_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_project_then_with_an_existing_path_saves_to_disk_and_reports_success() {
        let (ui_state, controller) = make_controller();
        let dir = temp_dir_for("save_existing_path");
        let path = dir.join("test.evaproj");
        {
            let mut project = ui_state.get_project_write();
            project.tmp_settings.current_project = Some(path.clone());
        }
        ui_state.mark_dirty();

        let (tx, rx) = std::sync::mpsc::channel();
        controller.save_project_then(move |ok| {
            tx.send(ok).unwrap();
        });

        let ok = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("save_project_then must call on_done");
        assert!(ok);
        assert!(
            !ui_state.is_dirty(),
            "a successful save must clear the dirty flag"
        );
        assert!(path.exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
