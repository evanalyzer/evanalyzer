use crate::AppWindow;
use crate::DialogType;
use crate::UiState;
use crate::{GlobalAppState, TemplateMetaSlint, TemplateMetaState};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::templates::get_user_templates_folder;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::{PIPELINE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use log::warn;
use slint::ComponentHandle;
use std::sync::{Arc, Mutex};

/// What the in-progress "save as template" flow is currently saving.
#[derive(Clone, Copy)]
enum TemplateTarget {
    Pipeline(PipelineId),
    Project,
}

/// Drives the "Save as Template" flow shared by pipelines and projects.
///
/// 1. The caller (pipelines or project controller) starts the flow, which opens
///    the metadata dialog (`TemplateMetaState`/`TemplateMetaDialog`).
/// 2. On confirm, a native "Save File" dialog is shown, defaulting to the
///    user's templates folder.
/// 3. The selected metadata + path are handed off to `ProjectExt::save_pipeline_as_template`
///    or `ProjectExt::save_project_as_template`.
pub struct TemplateController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    target: Mutex<Option<TemplateTarget>>,
}

impl TemplateController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            target: Mutex::new(None),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        let manager = self.clone();
        ui.global::<TemplateMetaState>()
            .on_confirm(move || manager.on_confirm());

        let manager = self.clone();
        ui.global::<TemplateMetaState>().on_cancel(move || {
            *manager.target.lock().expect("Poisoned") = None;
            if let Some(ui) = manager.ui.upgrade() {
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::None);
            }
        });
    }

    /// Opens the metadata dialog to save `pipeline_id` as a pipeline template.
    pub fn start_pipeline_template_save(self: &Arc<Self>, pipeline_id: PipelineId, name: String) {
        *self.target.lock().expect("Poisoned") = Some(TemplateTarget::Pipeline(pipeline_id));
        self.open_dialog("Save Pipeline as Template", name);
    }

    /// Opens the metadata dialog to save the current project as a project template.
    pub fn start_project_template_save(self: &Arc<Self>, name: String) {
        *self.target.lock().expect("Poisoned") = Some(TemplateTarget::Project);
        self.open_dialog("Save Project as Template", name);
    }

    fn open_dialog(self: &Arc<Self>, title: &str, name: String) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<TemplateMetaState>();
        state.set_dialog_title(title.into());
        state.set_meta(TemplateMetaSlint {
            name: name.into(),
            short_description: "".into(),
            description: "".into(),
            author_name: "".into(),
            author_organization: "".into(),
        });
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::TemplateMeta);
    }

    /// Metadata dialog confirmed: build `MetaData`, ask for a save location, and persist.
    fn on_confirm(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let Some(target) = self.target.lock().expect("Poisoned").take() else {
            return;
        };

        let meta_slint = ui.global::<TemplateMetaState>().get_meta();
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);

        let author_name = meta_slint.author_name.to_string();
        let mut author_parts = author_name.split_whitespace();
        let author_first_name = author_parts.next().unwrap_or("").to_string();
        let author_last_name = author_parts.collect::<Vec<_>>().join(" ");

        let meta = MetaData {
            name: meta_slint.name.to_string(),
            short_description: meta_slint.short_description.to_string(),
            description: meta_slint.description.to_string(),
            author_first_name,
            author_last_name,
            author_organization: meta_slint.author_organization.to_string(),
            creation_time: chrono::Local::now().to_rfc3339(),
        };

        let templates_folder = get_user_templates_folder();
        let default_file_name = if meta.name.is_empty() {
            "template".to_string()
        } else {
            meta.name.clone()
        };

        let dialog = match target {
            TemplateTarget::Pipeline(_) => rfd::FileDialog::new()
                .add_filter("Pipeline template", &[PIPELINE_EXTENSIONS]),
            TemplateTarget::Project => rfd::FileDialog::new()
                .add_filter("Project template", &[PROJECT_FILE_TEMPLATE_EXTENSIONS]),
        };

        let Some(path) = dialog
            .set_directory(&templates_folder)
            .set_file_name(&default_file_name)
            .save_file()
        else {
            return;
        };

        let app_state = self.app_state.clone();
        std::thread::spawn(move || {
            let result = match target {
                TemplateTarget::Pipeline(pipeline_id) => app_state
                    .get_project_write()
                    .save_pipeline_as_template(meta, pipeline_id, &path),
                TemplateTarget::Project => app_state
                    .get_project_write()
                    .save_project_as_template(meta, &path),
            };

            match result {
                Ok(_) => log::info!("Template saved to {}", path.display()),
                Err(e) => warn!("Failed to save template: {e}"),
            }
        });
    }
}
