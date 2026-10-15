use crate::AppWindow;
use crate::DialogType;
use crate::FileRequest;
use crate::UiState;
use crate::{GlobalAppState, TemplateMetaSlint, TemplateMetaState};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::templates::load_component_templates;
use evanalyzer_app::templates::load_project_templates;
use evanalyzer_cfg::core_types::PipelineId;
use evanalyzer_cfg::settings::meta_data::AuthorInformation;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::{COMPONENT_TEMPLATE_EXTENSION, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use log::warn;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// What the in-progress "save as template" flow is currently saving.
#[derive(Clone)]
enum TemplateTarget {
    Project,
    /// A pipeline's steps, saved to the component library (`.evacomp`).
    Pipeline(PipelineId),
    /// A component, by key - saved to the component library (`.evacomp`).
    Component(String),
}

/// Drives the "Save as Template" flow shared by projects, components and
/// pipelines saved as components.
///
/// 1. The caller (pipelines or project controller) starts the flow, which opens
///    the metadata dialog (`TemplateMetaState`/`TemplateMetaDialog`).
/// 2. On confirm, a native "Save File" dialog is shown, defaulting to the
///    user's templates folder.
/// 3. The selected metadata + path are handed off to
///    `ProjectExt::save_project_as_template`, `components::save_component_template`
///    or `components::save_pipeline_as_component`.
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

    /// Opens the metadata dialog to save pipeline `pipeline_id` to the
    /// component library, named after the pipeline.
    pub fn start_pipeline_component_save(self: &Arc<Self>, pipeline_id: PipelineId, name: String) {
        *self.target.lock().expect("Poisoned") = Some(TemplateTarget::Pipeline(pipeline_id));
        self.open_dialog("Save Pipeline as Component", name);
    }

    /// Opens the metadata dialog to save the current project as a project template.
    pub fn start_project_template_save(self: &Arc<Self>, name: String) {
        *self.target.lock().expect("Poisoned") = Some(TemplateTarget::Project);
        self.open_dialog("Save Project as Template", name);
    }

    /// Opens the metadata dialog to save component `key` to the component
    /// library, filled in from the component (`meta`) - its version one up, if
    /// it has one.
    pub fn start_component_template_save(self: &Arc<Self>, key: String, meta: MetaData) {
        *self.target.lock().expect("Poisoned") = Some(TemplateTarget::Component(key));
        self.open_dialog_with("Save Component to Library", &meta);
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<TemplateMetaState>();
            let mut shown = state.get_meta();
            shown.version = meta
                .version
                .map(|v| (v + 1).to_string())
                .unwrap_or_else(|| "1".into())
                .into();
            state.set_meta(shown);
        }
    }

    fn open_dialog(self: &Arc<Self>, title: &str, name: String) {
        self.open_dialog_with(
            title,
            &MetaData {
                name,
                ..Default::default()
            },
        );
    }

    fn open_dialog_with(self: &Arc<Self>, title: &str, meta: &MetaData) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<TemplateMetaState>();
        state.set_dialog_title(title.into());
        let author = meta.authors.first();
        state.set_meta(TemplateMetaSlint {
            name: meta.name.clone().into(),
            short_description: meta.short_description.clone().into(),
            description: meta.description.clone().into(),
            author_name: author
                .map(|a| a.full_name.clone())
                .unwrap_or_default()
                .into(),
            author_organization: author
                .map(|a| a.organization.clone())
                .unwrap_or_default()
                .into(),
            category: meta.category.clone().into(),
            tags: meta.tags.join(", ").into(),
            version: meta
                .version
                .map(|v| v.to_string())
                .unwrap_or_default()
                .into(),
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
            for (_path, template) in load_component_templates(backend.as_ref()) {
                if !template.component.meta.category.is_empty() {
                    categories.insert(template.component.meta.category);
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
        let organization = meta_slint.author_organization.to_string();
        // Only a single "Author Name" field exists in this dialog, so it
        // only ever sets the primary author (authors[0]) - additional
        // authors are addable today only by hand-editing the saved file.
        let authors = if author_name.is_empty() {
            Vec::new()
        } else {
            vec![AuthorInformation {
                full_name: author_name,
                organization,
            }]
        };

        let meta = MetaData {
            name: meta_slint.name.to_string(),
            short_description: meta_slint.short_description.to_string(),
            description: meta_slint.description.to_string(),
            authors,
            creation_time: chrono::Utc::now(),
            version: parse_version(&meta_slint.version),
            category: meta_slint.category.to_string(),
            tags,
            // Stamped for real by `save_project_as_template`/
            // `save_component_template` right before writing - left empty
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

        let request = match &target {
            TemplateTarget::Project => FileRequest::save_file("Save project template")
                .filter("Project template", &[PROJECT_FILE_TEMPLATE_EXTENSIONS]),
            TemplateTarget::Component(_) | TemplateTarget::Pipeline(_) => {
                FileRequest::save_file("Save component")
                    .filter("Component", &[COMPONENT_TEMPLATE_EXTENSION])
            }
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
                    TemplateTarget::Project => app_state
                        .get_project_write()
                        .save_project_as_template(app_state.backend().files(), meta, &path),
                    TemplateTarget::Pipeline(pipeline_id) => {
                        evanalyzer_app::components::save_pipeline_as_component(
                            &app_state.get_project(),
                            app_state.backend().files(),
                            meta,
                            pipeline_id,
                            &path,
                        )
                    }
                    TemplateTarget::Component(key) => {
                        evanalyzer_app::components::save_component_template(
                            &mut app_state.get_project_write(),
                            app_state.backend().files(),
                            meta,
                            &key,
                            &path,
                        )
                    }
                };

            match result {
                Ok(_) => log::info!("Template saved to {}", path.display()),
                Err(e) => warn!("Failed to save template: {e}"),
            }
        });
    }
}

/// Parses a comma-separated tag string into a trimmed, non-empty tag list.
/// The template version the user typed - `None` for none (the field is
/// optional; the dialog only accepts whole numbers from 1).
fn parse_version(raw: &str) -> Option<u32> {
    raw.trim().parse().ok().filter(|v| *v >= 1)
}

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
    fn the_version_is_optional_and_a_whole_number_from_one() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version(" 3 "), Some(3));
        assert_eq!(parse_version("0"), None);
        assert_eq!(parse_version("1.5"), None);
        assert_eq!(parse_version("abc"), None);
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
        meta.version = "2".into();
        state.set_meta(meta);
    }

    #[test]
    fn saving_a_command_component_to_the_library_writes_an_evacomp_file() {
        use evanalyzer_cfg::settings::components::ComponentDefinition;
        let (ui, ui_state, controller) = controller();
        ui_state
            .get_project_write()
            .components
            .push(ComponentDefinition {
                key: "SMOOTH".into(),
                meta: MetaData {
                    name: "Smooth".into(),
                    short_description: "Blurs twice".into(),
                    version: Some(2),
                    ..Default::default()
                },
                classes: vec![],
                steps: vec![],
                parameters: vec![],
                origin: None,
            });
        let meta = ui_state.get_project().components[0].meta.clone();
        let dir = tempfile::tempdir().unwrap();

        controller.start_component_template_save("SMOOTH".into(), meta);
        drain_ui_queue();
        let state = ui.global::<TemplateMetaState>();
        assert_eq!(state.get_meta().name, "Smooth");
        assert_eq!(state.get_meta().short_description, "Blurs twice");
        assert_eq!(state.get_meta().version, "3", "one up");

        state.invoke_confirm();
        let target = dir.path().join("smooth.evacomp");
        choose_file(&ui, &target);

        let written = std::fs::read_to_string(&target).expect("component written");
        let json: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(json["component"]["key"], "SMOOTH");
        assert_eq!(json["component"]["meta"]["version"], 3);
        let project = ui_state.get_project();
        let origin = project.components[0]
            .origin
            .as_ref()
            .expect("linked to the file");
        assert_eq!(
            (origin.library.as_str(), origin.version),
            ("smooth.evacomp", 3)
        );
    }

    #[test]
    fn saving_a_pipeline_as_component_writes_an_evacomp_file_and_leaves_the_project() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_settings::{PipelineStepSettings, StepContent};
        let (ui, ui_state, controller) = controller();
        ui_state.get_project_write().pipelines[0].steps = vec![PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(1),
            enabled: true,
            content: StepContent::Command(PipelineCommand::Blur(Default::default())),
        }];
        let dir = tempfile::tempdir().unwrap();

        controller.start_pipeline_component_save(PipelineId(1), "Nuclei".into());
        drain_ui_queue();
        let state = ui.global::<TemplateMetaState>();
        assert_eq!(state.get_dialog_title(), "Save Pipeline as Component");
        assert_eq!(state.get_meta().name, "Nuclei");

        state.invoke_confirm();
        let target = dir.path().join("nuclei.evacomp");
        choose_file(&ui, &target);

        let written = std::fs::read_to_string(&target).expect("component written");
        let json: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(json["component"]["meta"]["name"], "Nuclei");
        assert_eq!(json["component"]["steps"].as_array().unwrap().len(), 1);
        let project = ui_state.get_project();
        assert!(
            project.components.is_empty(),
            "the project gets no component"
        );
        assert_eq!(project.pipelines[0].steps.len(), 1, "nor loses its steps");
    }

    #[test]
    fn saving_a_project_template_writes_the_chosen_file_with_the_metadata() {
        let (ui, _ui_state, controller) = controller();
        let dir = tempfile::tempdir().unwrap();

        controller.start_project_template_save("Nuclei".into());
        drain_ui_queue();
        let state = ui.global::<TemplateMetaState>();
        assert_eq!(state.get_dialog_title(), "Save Project as Template");
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
        let target = dir
            .path()
            .join(format!("nuclei.{PROJECT_FILE_TEMPLATE_EXTENSIONS}"));
        choose_file(&ui, &target);

        let written = std::fs::read_to_string(&target).expect("template written");
        assert!(written.contains("Finds nuclei"));
        assert!(written.contains("Ada"));
        assert!(written.contains("dapi"));
        let json: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(json["meta"]["version"], 2);
    }

    #[test]
    fn saving_a_project_template_without_a_version_writes_none() {
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
        let written = std::fs::read_to_string(&target).expect("template written");
        let json: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert!(
            json["meta"].get("version").is_none(),
            "no version entered: none written"
        );
    }

    #[test]
    fn cancelling_closes_the_dialog_and_forgets_what_was_being_saved() {
        let (ui, _ui_state, controller) = controller();
        controller.start_project_template_save("Nuclei".into());
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
        controller.start_project_template_save("Nuclei".into());
        ui.global::<TemplateMetaState>().invoke_confirm();
        drain_ui_queue();
        ui.global::<crate::FileBrowserState>().invoke_cancel();
        drain_ui_queue();
        assert!(!ui.global::<crate::FileBrowserState>().get_visible());
    }
}
