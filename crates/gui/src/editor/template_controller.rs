use crate::AppWindow;
use crate::DialogType;
use crate::FileRequest;
use crate::UiState;
use crate::{GlobalAppState, TemplateMetaSlint, TemplateMetaState};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::templates::load_pipeline_templates;
use evanalyzer_app::templates::load_project_templates;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::{PIPELINE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use log::warn;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::collections::BTreeSet;
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
            category: "".into(),
            tags: "".into(),
        });
        state.set_known_categories(ModelRc::new(VecModel::from(Vec::<SharedString>::new())));
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::TemplateMeta);

        // Fetch the categories already in use across existing templates in the
        // background, so the dialog can offer them as quick-pick suggestions
        // without blocking on disk IO.
        let ui_weak = self.ui.clone();
        let backend = self.app_state.backend();
        crate::helper::ui_thread::spawn(move || {
            let mut categories: BTreeSet<String> = BTreeSet::new();
            for (_path, template) in load_project_templates(backend.as_ref()) {
                if !template.meta.category.is_empty() {
                    categories.insert(template.meta.category);
                }
            }
            for (_path, template) in load_pipeline_templates(backend.as_ref()) {
                if !template.meta.category.is_empty() {
                    categories.insert(template.meta.category);
                }
            }
            let categories: Vec<SharedString> = categories.into_iter().map(Into::into).collect();

            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<TemplateMetaState>()
                        .set_known_categories(ModelRc::new(VecModel::from(categories)));
                }
            });
        });
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

        let tags = parse_tags(&meta_slint.tags);
        let author_name = meta_slint.author_name.trim().to_string();
        // Only a single "Author Name" field exists in this dialog, so it
        // only ever sets the primary author (authors[0]) - additional
        // authors are addable today only by hand-editing the saved file.
        let authors = if author_name.is_empty() {
            Vec::new()
        } else {
            vec![author_name]
        };

        let meta = MetaData {
            name: meta_slint.name.to_string(),
            short_description: meta_slint.short_description.to_string(),
            description: meta_slint.description.to_string(),
            authors,
            author_organization: meta_slint.author_organization.to_string(),
            creation_time: chrono::Utc::now(),
            category: meta_slint.category.to_string(),
            tags,
            // Stamped for real by `save_project_as_template`/
            // `save_pipeline_as_template` right before writing - left empty
            // here since this value never reaches disk as-is.
            app_version: String::new(),
        };

        // The backend's own templates folder - the server's in remote mode.
        let templates_folder = self
            .app_state
            .backend()
            .template_folders()
            .map(|folders| folders.user)
            .unwrap_or_default();
        let default_file_name = if meta.name.is_empty() {
            "template".to_string()
        } else {
            meta.name.clone()
        };

        let request = match target {
            TemplateTarget::Pipeline(_) => FileRequest::save_file("Save pipeline template")
                .filter("Pipeline template", &[PIPELINE_EXTENSIONS]),
            TemplateTarget::Project => FileRequest::save_file("Save project template")
                .filter("Project template", &[PROJECT_FILE_TEMPLATE_EXTENSIONS]),
        }
        .start_in(&templates_folder)
        .file_name(&default_file_name);

        let app_state = self.app_state.clone();
        self.app_state.file_browser.open(request, move |path| {
            let Some(path) = path else {
                return;
            };
            Self::write_template(app_state, target, meta, path);
        });
    }

    fn write_template(
        app_state: Arc<UiState>,
        target: TemplateTarget,
        meta: MetaData,
        path: std::path::PathBuf,
    ) {
        crate::helper::ui_thread::spawn(move || {
            let result =
                match target {
                    TemplateTarget::Pipeline(pipeline_id) => {
                        app_state.get_project_write().save_pipeline_as_template(
                            app_state.backend().files(),
                            meta,
                            pipeline_id,
                            &path,
                        )
                    }
                    TemplateTarget::Project => app_state
                        .get_project_write()
                        .save_project_as_template(app_state.backend().files(), meta, &path),
                };

            match result {
                Ok(_) => log::info!("Template saved to {}", path.display()),
                Err(e) => warn!("Failed to save template: {e}"),
            }
        });
    }
}

/// Parses a comma-separated tag string into a trimmed, non-empty tag list.
fn parse_tags(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_tags --------------------------------------------------------------

    #[test]
    fn parse_tags_splits_on_commas_and_trims_whitespace() {
        assert_eq!(
            parse_tags("cells, uptake , microscopy"),
            vec![
                "cells".to_string(),
                "uptake".to_string(),
                "microscopy".to_string()
            ]
        );
    }

    #[test]
    fn parse_tags_drops_empty_entries_from_stray_commas() {
        assert_eq!(
            parse_tags("cells,,uptake,"),
            vec!["cells".to_string(), "uptake".to_string()]
        );
    }

    #[test]
    fn parse_tags_of_an_empty_string_is_an_empty_list() {
        assert!(parse_tags("").is_empty());
    }

    #[test]
    fn parse_tags_of_only_whitespace_is_an_empty_list() {
        assert!(parse_tags("   ").is_empty());
    }

    // -- the save flow, with a window ---------------------------------------

    use crate::editor::test_support::{choose_file, test_ui_windows, ui_state_with_windows};
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_cfg::core_types::ImageAddress;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    fn controller() -> (crate::AppWindow, Arc<UiState>, Arc<TemplateController>) {
        let (ui, results_ui) = test_ui_windows();
        let mut project = evanalyzer_app::project::ProjectWithRuntime::default();
        project.add_pipeline(PipelineSettings {
            id: PipelineId(1),
            name: "Nuclei".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![],
        });
        let ui_state = ui_state_with_windows(&ui, &results_ui, project);
        let controller = Arc::new(TemplateController::new(ui.as_weak(), ui_state.clone()));
        controller.attach_callbacks();
        (ui, ui_state, controller)
    }

    fn fill_meta(ui: &crate::AppWindow) {
        let state = ui.global::<TemplateMetaState>();
        let mut meta = state.get_meta();
        meta.short_description = "Finds nuclei".into();
        meta.author_name = "  Ada  ".into();
        meta.category = "Segmentation".into();
        meta.tags = "nuclei, dapi,".into();
        state.set_meta(meta);
    }

    #[test]
    fn saving_a_pipeline_template_writes_the_chosen_file_with_the_metadata() {
        let (ui, _ui_state, controller) = controller();
        let dir = tempfile::tempdir().unwrap();

        controller.start_pipeline_template_save(PipelineId(1), "Nuclei".into());
        drain_ui_queue();
        let state = ui.global::<TemplateMetaState>();
        assert_eq!(state.get_dialog_title(), "Save Pipeline as Template");
        assert_eq!(state.get_meta().name, "Nuclei");
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::TemplateMeta
        );

        fill_meta(&ui);
        state.invoke_confirm();
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::None
        );
        let target = dir.path().join(format!("nuclei.{PIPELINE_EXTENSIONS}"));
        choose_file(&ui, &target);

        let written = std::fs::read_to_string(&target).expect("template written");
        assert!(written.contains("Finds nuclei"));
        assert!(written.contains("Ada"));
        assert!(written.contains("dapi"));
    }

    #[test]
    fn saving_a_project_template_writes_the_chosen_file() {
        let (ui, _ui_state, controller) = controller();
        let dir = tempfile::tempdir().unwrap();
        controller.start_project_template_save("".into());
        assert_eq!(
            ui.global::<TemplateMetaState>().get_dialog_title(),
            "Save Project as Template"
        );
        ui.global::<TemplateMetaState>().invoke_confirm();
        let target = dir
            .path()
            .join(format!("project.{PROJECT_FILE_TEMPLATE_EXTENSIONS}"));
        choose_file(&ui, &target);
        assert!(target.exists());
    }

    #[test]
    fn cancelling_closes_the_dialog_and_forgets_what_was_being_saved() {
        let (ui, _ui_state, controller) = controller();
        controller.start_pipeline_template_save(PipelineId(1), "Nuclei".into());
        ui.global::<TemplateMetaState>().invoke_cancel();
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::None
        );
        assert!(controller.target.lock().unwrap().is_none());

        // Confirm without a pending save does nothing (no file dialog).
        ui.global::<TemplateMetaState>().invoke_confirm();
        assert!(!ui.global::<crate::FileBrowserState>().get_visible());
    }

    #[test]
    fn cancelling_the_file_dialog_writes_nothing() {
        let (ui, _ui_state, controller) = controller();
        controller.start_pipeline_template_save(PipelineId(1), "Nuclei".into());
        ui.global::<TemplateMetaState>().invoke_confirm();
        drain_ui_queue();
        ui.global::<crate::FileBrowserState>().invoke_cancel();
        drain_ui_queue();
        assert!(!ui.global::<crate::FileBrowserState>().get_visible());
    }
}
