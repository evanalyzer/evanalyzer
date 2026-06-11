use crate::AppWindow;
use crate::DialogType;
use crate::GlobalAppState;
use crate::ImagesListState;
use crate::PipelinesPanelState;
use crate::ProjectTemplateDef;
use crate::ProjectTemplateState;
use crate::ToolbarState;
use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::pipelines_controller::PipelinesController;
use crate::editor::project_settings_controller::ProjectSettingsController;
use crate::editor::results_list_controller::ResultsListController;
use crate::editor::template_controller::TemplateController;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::extensions::project_ext::SaveProjectActions;
use evanalyzer_app::templates::load_project_templates;
use evanalyzer_cfg::PROJECT_FILE_EXTENSIONS;
use evanalyzer_cfg::settings::templates::ProjectTemplate;
use evanalyzer_core::SUPPORTED_IMAGE_FORMATS;
use log::{info, warn};
use slint::ComponentHandle;
use slint::{ModelRc, VecModel};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
            ui.global::<ToolbarState>().on_save_as_file_clicked(move || {
                manager.save_project_as_handler();
            });

            // Save project as template
            let manager = Arc::clone(self);
            ui.global::<ToolbarState>()
                .on_save_project_as_template_clicked(move || {
                    let name = manager
                        .app_state
                        .get_project()
                        .metadata
                        .name
                        .clone();
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
            ui.global::<ProjectTemplateState>()
                .on_select(move |id| {
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
            ui.global::<ProjectTemplateState>()
                .on_confirm(move |id| {
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

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Supported Files", &allowed_files)
            .add_filter("Image Files", &SUPPORTED_IMAGE_FORMATS)
            .add_filter("Project Files", &[PROJECT_FILE_EXTENSIONS])
            .pick_file()
        {
            let manager = Arc::clone(self);

            std::thread::spawn(move || {
                let is_project =
                    path.extension().and_then(|ext| ext.to_str()) == Some(PROJECT_FILE_EXTENSIONS);

                if is_project {
                    manager.open_new_project(&path);
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
                match in_thread
                    .app_state
                    .get_project_write()
                    .save_project_as(&path)
                {
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
        let result = self.app_state.get_project_write().save_project();

        if result == SaveProjectActions::PleaseSelectFile {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Project files", &[PROJECT_FILE_EXTENSIONS])
                .save_file()
            {
                let in_thread = self.clone();
                std::thread::spawn(move || {
                    match in_thread
                        .app_state
                        .get_project_write()
                        .save_project_as(&path)
                    {
                        Ok(_) => {
                            info!("Project saved: {}", path.display());
                            in_thread.app_state.clear_dirty();
                        }
                        Err(msg) => {
                            warn!("Project not saved: {}", msg);
                        }
                    };
                });
            }
        } else if result == SaveProjectActions::Success {
            info!("Project saved");
            self.app_state.clear_dirty();
        } else {
            warn!("Could not save project");
        }
    }

    /// Opens the "New from Project Template" dialog.
    ///
    /// The dialog opens immediately with an empty list; the available
    /// templates are loaded from disk in the background and populated once
    /// ready, so a slow filesystem doesn't delay opening the dialog.
    fn open_project_template_dialog(self: &Arc<Self>) {
        let manager = self.clone();
        if let Some(ui) = self.ui.upgrade() {
            let picker = ui.global::<ProjectTemplateState>();
            picker.set_selected_id(-1);
            picker.set_has_detail(false);
            picker.set_templates(ModelRc::new(VecModel::from(Vec::<ProjectTemplateDef>::new())));
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::ProjectTemplate);
        }
        std::thread::spawn(move || {
            let templates: Vec<ProjectTemplate> = load_project_templates()
                .into_iter()
                .map(|(_path, template)| template)
                .collect();
            let defs: Vec<ProjectTemplateDef> = templates
                .iter()
                .enumerate()
                .map(|(idx, t)| project_template_to_def(idx as i32, t))
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
                ui.global::<ProjectTemplateState>()
                    .set_templates(ModelRc::new(VecModel::from(defs)));
            }) {
                warn!("Failed to populate project template picker: {}", e);
            }
        });
    }

    /// Replaces the current project's classification, plate and pipeline
    /// settings with the ones from the selected project template, then
    /// re-syncs the affected panels.
    fn apply_project_template(self: &Arc<Self>, id: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let template = {
            let templates = self.project_templates.lock().expect("Poisoned");
            templates.get(id as usize).cloned()
        };
        let Some(template) = template else {
            warn!("project template picker confirm: unknown template id {}", id);
            return;
        };

        let first_pipeline_id = {
            let mut project = self.app_state.get_project_write();
            project.apply_project_template(&template);
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
    let author = format!(
        "{} {}",
        template.meta.author_first_name, template.meta.author_last_name
    )
    .trim()
    .to_string();

    ProjectTemplateDef {
        id,
        name: template.meta.name.clone().into(),
        short_description: template.meta.short_description.clone().into(),
        description: template.meta.description.clone().into(),
        author: author.into(),
        organization: template.meta.author_organization.clone().into(),
        creation_time: template
            .meta
            .creation_time
            .format("%Y-%m-%d")
            .to_string()
            .into(),
        pipeline_count: template.pipelines.len() as i32,
    }
}
