//! Pipeline focus: shows only the selected pipeline's image channel and
//! object classes.
//!
//! - The header's Focus button switches focus mode on/off and is the only
//!   thing that changes the remembered preference.
//! - While focus mode is on, selecting a pipeline moves the focus to it.
//! - Double-clicking a pipeline focuses it right away (and switches focus
//!   mode on for this session); double-clicking the focused pipeline leaves
//!   the focus again.
//! - "Show all" in the banner leaves the focus for this session.
//!
//! The focus itself is a runtime overlay in the project (see
//! `ProjectExt::focus_pipeline`): the user's own visibility settings are never
//! changed by it, so leaving the focus brings them back exactly.

use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::viewport_controller::ViewportController;
use crate::{AppWindow, PipelineFocusState, PipelinesPanelState};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::PipelineId;
use log::warn;
use slint::ComponentHandle;
use std::sync::Arc;

/// Persists the focus-mode switch.
pub type SaveFocusMode = Box<dyn Fn(bool) + Send + Sync>;

pub struct FocusController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    image_meta_controller: Arc<ImageMetaController>,
    classification_controller: Arc<ClassificationController>,
    viewport_controller: Arc<ViewportController>,
    save_mode: SaveFocusMode,
}

impl FocusController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        image_meta_controller: Arc<ImageMetaController>,
        classification_controller: Arc<ClassificationController>,
        viewport_controller: Arc<ViewportController>,
        save_mode: SaveFocusMode,
    ) -> Self {
        Self {
            ui,
            app_state,
            image_meta_controller,
            classification_controller,
            viewport_controller,
            save_mode,
        }
    }

    /// Wires the Slint callbacks; `initial_mode` is the remembered switch.
    pub fn attach_callbacks(self: &Arc<Self>, initial_mode: bool) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<PipelineFocusState>();
        state.set_enabled(initial_mode);

        let this = self.clone();
        state.on_toggle_mode(move || {
            let Some(ui) = this.ui.upgrade() else {
                return;
            };
            let on = !ui.global::<PipelineFocusState>().get_enabled();
            ui.global::<PipelineFocusState>().set_enabled(on);
            (this.save_mode)(on);
            if on {
                let active = ui.global::<PipelinesPanelState>().get_active_pipeline_id();
                this.focus(active);
            } else {
                this.clear();
            }
        });

        let this = self.clone();
        state.on_pipeline_selected(move |pipeline_id| {
            let enabled = this
                .ui
                .upgrade()
                .is_some_and(|ui| ui.global::<PipelineFocusState>().get_enabled());
            if enabled {
                this.focus(pipeline_id);
            }
        });

        let this = self.clone();
        state.on_pipeline_double_clicked(move |pipeline_id| {
            let Some(ui) = this.ui.upgrade() else {
                return;
            };
            let focused = this.app_state.get_project().focused_pipeline();
            if focused == Some(pipeline_id_of(pipeline_id)) {
                ui.global::<PipelineFocusState>().set_enabled(false);
                this.clear();
            } else {
                ui.global::<PipelineFocusState>().set_enabled(true);
                this.focus(pipeline_id);
            }
        });

        let this = self.clone();
        state.on_exit(move || {
            if let Some(ui) = this.ui.upgrade() {
                ui.global::<PipelineFocusState>().set_enabled(false);
            }
            this.clear();
        });
    }

    /// Re-evaluates the focus after pipelines were edited or deleted, or a
    /// project was opened (which starts without a focus). Refreshes the
    /// channel/class lists and the viewer only if the focus changed; the
    /// banner always.
    pub fn refresh(&self) {
        let changed = {
            let mut project = self.app_state.get_project_write();
            let before = project.tmp_settings.pipeline_focus.clone();
            project.refresh_pipeline_focus();
            project.tmp_settings.pipeline_focus != before
        };
        if changed {
            self.apply();
        } else {
            self.update_banner();
        }
    }

    fn focus(&self, pipeline_id: i32) {
        if pipeline_id < 0 {
            return;
        }
        self.app_state
            .get_project_write()
            .focus_pipeline(pipeline_id_of(pipeline_id));
        self.apply();
    }

    fn clear(&self) {
        self.app_state.get_project_write().clear_pipeline_focus();
        self.apply();
    }

    /// Pushes the current focus to the UI: banner, channel list, class list
    /// and viewer.
    fn apply(&self) {
        self.update_banner();
        if let Err(e) = self.image_meta_controller.sync_image_meta_to_slint() {
            warn!("Failed to refresh the channel list after a focus change: {e:?}");
        }
        self.classification_controller
            .sync_classification_to_slint();
        self.viewport_controller
            .trigger_redraw_low_res_and_high_res();
        self.viewport_controller.trigger_image_redraw_objects();
    }

    /// Shows the focused pipeline's name in the banner (empty = hidden).
    fn update_banner(&self) {
        let name = {
            let project = self.app_state.get_project();
            project
                .focused_pipeline()
                .and_then(|id| project.pipelines.iter().find(|p| p.id == id))
                .map(|p| p.name.clone())
                .unwrap_or_default()
        };
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<PipelineFocusState>()
                    .set_focused_pipeline_name(name.into());
            }
        }) {
            warn!("Failed to update the pipeline focus banner: {e}");
        }
    }
}

fn pipeline_id_of(id: i32) -> PipelineId {
    PipelineId(id as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::object_list_controller::ObjectListController;
    use crate::editor::test_support::{test_ui_windows, ui_state_with_windows};
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::project::ProjectWithRuntime;
    use evanalyzer_cfg::core_types::{ImageAddress, ObjectClass};
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::TransformObjectsSettings;
    use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
    use std::sync::Mutex;

    fn pipeline(id: u32, name: &str, class: u32) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(id),
            name: name.into(),
            description: None,
            image_source: ImageAddress::Scratchpad,
            enabled: true,
            steps: vec![PipelineStepSettings {
                enabled: true,
                command: PipelineCommand::TransformObjects(TransformObjectsSettings {
                    input_class: ObjectClass::Valid(class),
                    output_class: ObjectClass::Valid(class),
                    ..Default::default()
                }),
            }],
        }
    }

    struct Fixture {
        ui: AppWindow,
        _results_ui: crate::ResultsWindow,
        ui_state: Arc<UiState>,
        controller: Arc<FocusController>,
        saved: Arc<Mutex<Vec<bool>>>,
    }

    fn fixture(initial_mode: bool) -> Fixture {
        let (ui, results_ui) = test_ui_windows();
        let mut project = ProjectWithRuntime::default();
        project.pipelines.push(pipeline(1, "Nuclei", 3));
        project.pipelines.push(pipeline(2, "Cells", 4));
        let ui_state = ui_state_with_windows(&ui, &results_ui, project);
        let weak = ui.as_weak();
        let viewport = Arc::new(ViewportController::new(weak.clone(), ui_state.clone()));
        let object_list = Arc::new(ObjectListController::new(
            weak.clone(),
            ui_state.clone(),
            viewport.clone(),
        ));
        let classification = Arc::new(ClassificationController::new(
            weak.clone(),
            ui_state.clone(),
            object_list,
            viewport.clone(),
        ));
        let image_meta = Arc::new(ImageMetaController::new(
            weak.clone(),
            ui_state.clone(),
            viewport.clone(),
        ));
        let saved = Arc::new(Mutex::new(Vec::new()));
        let saved_in = saved.clone();
        let controller = Arc::new(FocusController::new(
            weak,
            ui_state.clone(),
            image_meta,
            classification,
            viewport,
            Box::new(move |on| saved_in.lock().unwrap().push(on)),
        ));
        controller.attach_callbacks(initial_mode);
        Fixture {
            ui,
            _results_ui: results_ui,
            ui_state,
            controller,
            saved,
        }
    }

    impl Fixture {
        fn focused(&self) -> Option<PipelineId> {
            self.ui_state.get_project().focused_pipeline()
        }
        fn state(&self) -> PipelineFocusState<'_> {
            self.ui.global::<PipelineFocusState>()
        }
        fn banner(&self) -> String {
            drain_ui_queue();
            self.state().get_focused_pipeline_name().to_string()
        }
    }

    #[test]
    fn the_remembered_mode_is_restored_on_start() {
        assert!(fixture(true).state().get_enabled());
        assert!(!fixture(false).state().get_enabled());
    }

    #[test]
    fn selecting_a_pipeline_focuses_it_only_while_focus_mode_is_on() {
        let f = fixture(false);
        f.state().invoke_pipeline_selected(1);
        assert_eq!(f.focused(), None);

        let f = fixture(true);
        f.state().invoke_pipeline_selected(1);
        assert_eq!(f.focused(), Some(PipelineId(1)));
        assert_eq!(f.banner(), "Nuclei");
        // The focus follows the selection.
        f.state().invoke_pipeline_selected(2);
        assert_eq!(f.focused(), Some(PipelineId(2)));
        assert_eq!(f.banner(), "Cells");
    }

    #[test]
    fn the_header_button_focuses_the_active_pipeline_and_is_remembered() {
        let f = fixture(false);
        f.ui.global::<PipelinesPanelState>()
            .set_active_pipeline_id(2);
        f.state().invoke_toggle_mode();
        assert!(f.state().get_enabled());
        assert_eq!(f.focused(), Some(PipelineId(2)));

        f.state().invoke_toggle_mode();
        assert!(!f.state().get_enabled());
        assert_eq!(f.focused(), None);
        assert_eq!(f.banner(), "");
        assert_eq!(*f.saved.lock().unwrap(), vec![true, false]);
    }

    #[test]
    fn double_click_focuses_quickly_and_again_leaves_without_changing_the_preference() {
        let f = fixture(false);
        f.state().invoke_pipeline_double_clicked(1);
        assert!(f.state().get_enabled());
        assert_eq!(f.focused(), Some(PipelineId(1)));

        // Double-clicking another pipeline moves the focus ...
        f.state().invoke_pipeline_double_clicked(2);
        assert_eq!(f.focused(), Some(PipelineId(2)));
        // ... double-clicking the focused one leaves it.
        f.state().invoke_pipeline_double_clicked(2);
        assert_eq!(f.focused(), None);
        assert!(!f.state().get_enabled());
        assert!(f.saved.lock().unwrap().is_empty());
    }

    #[test]
    fn show_all_leaves_the_focus_for_this_session() {
        let f = fixture(true);
        f.state().invoke_pipeline_selected(1);
        f.state().invoke_exit();
        assert_eq!(f.focused(), None);
        assert!(!f.state().get_enabled());
        assert!(f.saved.lock().unwrap().is_empty());
    }

    #[test]
    fn opening_another_project_clears_the_banner() {
        let f = fixture(true);
        f.state().invoke_pipeline_selected(1);
        assert_eq!(f.banner(), "Nuclei");
        // A freshly loaded project starts without a focus.
        *f.ui_state.get_project_write() = ProjectWithRuntime::default();
        f.controller.refresh();
        assert_eq!(f.banner(), "");
        // Focus mode itself stays on for the next selection.
        assert!(f.state().get_enabled());
    }

    #[test]
    fn refresh_ends_the_focus_of_a_deleted_pipeline() {
        let f = fixture(true);
        f.state().invoke_pipeline_selected(1);
        f.ui_state.get_project_write().pipelines.remove(0);
        f.controller.refresh();
        assert_eq!(f.focused(), None);
        assert_eq!(f.banner(), "");
    }
}
