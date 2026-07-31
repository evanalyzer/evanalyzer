use crate::AppWindow;
use crate::PipelinesPanelState;
use crate::ToolbarState;
use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::pipelines_controller::PipelinesController;
use crate::editor::project_settings_controller::ProjectSettingsController;
use crate::editor::viewport_controller::ViewportController;
use evanalyzer_cfg::core_types::PipelineId;
use slint::ComponentHandle;
use std::sync::Arc;

/// Wires the toolbar's Undo/Redo buttons (and, via `main.slint`'s global
/// `Ctrl+Z`/`Ctrl+Shift+Z` shortcut, the keyboard) to `UiState::undo`/`redo`.
///
/// `UiState` owns the actual undo/redo stacks and takes snapshots
/// automatically inside `get_project_write` - this controller only has to
/// call `undo`/`redo` and then refresh every panel that could be affected by
/// the restored settings, since there's no single "refresh everything from
/// project state" entry point in this codebase.
pub struct UndoRedoController {
    ui: slint::Weak<AppWindow>,
    app_state: Arc<UiState>,
    image_list_controller: Arc<ImagesListController>,
    project_settings_controller: Arc<ProjectSettingsController>,
    classification_controller: Arc<ClassificationController>,
    pipelines_controller: Arc<PipelinesController>,
    object_list_controller: Arc<ObjectListController>,
    viewport_controller: Arc<ViewportController>,
}

impl UndoRedoController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        image_list_controller: Arc<ImagesListController>,
        project_settings_controller: Arc<ProjectSettingsController>,
        classification_controller: Arc<ClassificationController>,
        pipelines_controller: Arc<PipelinesController>,
        object_list_controller: Arc<ObjectListController>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            image_list_controller,
            project_settings_controller,
            classification_controller,
            pipelines_controller,
            object_list_controller,
            viewport_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        let manager = self.clone();
        ui.global::<ToolbarState>().on_undo_clicked(move || {
            if manager.app_state.undo() {
                manager.refresh_all_panels();
            }
        });

        let manager = self.clone();
        ui.global::<ToolbarState>().on_redo_clicked(move || {
            if manager.app_state.redo() {
                manager.refresh_all_panels();
            }
        });
    }

    /// Refreshes every panel an undo/redo restore could have changed. Mirrors
    /// (a subset of) the sync calls `ProjectController::open_new_project`
    /// makes after loading a project from disk - both are "the in-memory
    /// `ProjectSettings` just changed out from under the UI, repaint
    /// everything" situations.
    ///
    /// `sync_pipelines_to_slint`/`sync_objects_to_slint` only refresh their
    /// respective *list* (pipeline names, object rows) - the currently
    /// open/selected item's own detail panel (a pipeline's step list, an
    /// object's classification) is a separate sync call that only fires on
    /// selection changes, so it goes stale after a restore until the user
    /// re-selects the item. Refresh those explicitly too.
    fn refresh_all_panels(self: &Arc<Self>) {
        self.image_list_controller.sync_image_list_to_slint();
        self.project_settings_controller
            .sync_project_settings_to_slint();
        self.classification_controller
            .sync_classification_to_slint();
        self.pipelines_controller.sync_pipelines_to_slint();
        self.object_list_controller.sync_objects_to_slint();
        self.object_list_controller
            .sync_selected_object_to_slint(false);
        self.viewport_controller.trigger_image_redraw_objects();

        if let Some(ui) = self.ui.upgrade() {
            let active_pipeline_id = ui.global::<PipelinesPanelState>().get_active_pipeline_id();
            if active_pipeline_id != 0 {
                self.pipelines_controller
                    .sync_steps_of_selected_pipeline_to_slint(
                        PipelineId(active_pipeline_id as u32),
                        false,
                    );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::pipelines_controller::PipelinesController;
    use crate::editor::template_controller::TemplateController;
    use crate::editor::test_support::{
        project_with_one_image, test_ui_state_with_project, test_ui_windows,
    };

    fn make_controller() -> (Arc<UiState>, Arc<UndoRedoController>) {
        make_controller_with_ui(slint::Weak::default())
    }

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
    ) -> (Arc<UiState>, Arc<UndoRedoController>) {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), ui_state.clone()));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let template_controller = Arc::new(TemplateController::new(ui.clone(), ui_state.clone()));
        let pipelines_controller = Arc::new(PipelinesController::new(
            ui.clone(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
            template_controller,
        ));
        let image_list_controller = Arc::new(ImagesListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(HistogramController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            Arc::new(ImageMetaController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            object_list_controller.clone(),
        ));
        let project_settings_controller = Arc::new(ProjectSettingsController::new(
            ui.clone(),
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let classification_controller = Arc::new(ClassificationController::new(
            ui.clone(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
        ));
        let controller = Arc::new(UndoRedoController::new(
            ui,
            ui_state.clone(),
            image_list_controller,
            project_settings_controller,
            classification_controller,
            pipelines_controller,
            object_list_controller,
            viewport_controller,
        ));
        (ui_state, controller)
    }

    /// `refresh_all_panels` fans out to every other controller's Slint sync
    /// call - with a dead `ui: Weak::default()` (as if the window had been
    /// closed) every one of those must degrade to a no-op, not panic. This
    /// is the only independently testable behavior in this controller: the
    /// actual undo/redo bookkeeping lives in `UiState::undo`/`redo`, and
    /// `attach_callbacks` only wires Slint callbacks.
    #[test]
    fn refresh_all_panels_does_not_panic_without_a_live_ui() {
        let (_, controller) = make_controller();

        controller.refresh_all_panels();
    }

    // -- attach_callbacks (live AppWindow) -----------------------------------------

    #[test]
    fn attach_callbacks_wires_the_toolbars_undo_and_redo_buttons_to_ui_state() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        // Make an edit so there's a checkpoint to undo.
        ui_state.get_project_write().metadata.name = "edited".to_string();
        assert_eq!(ui_state.get_project().metadata.name, "edited");

        ui.global::<ToolbarState>().invoke_undo_clicked();
        assert_eq!(
            ui_state.get_project().metadata.name,
            "",
            "the wired undo button must call UiState::undo() and restore the pre-edit state"
        );

        ui.global::<ToolbarState>().invoke_redo_clicked();
        assert_eq!(
            ui_state.get_project().metadata.name,
            "edited",
            "the wired redo button must call UiState::redo() and reapply the edit"
        );
    }

    #[test]
    fn attach_callbacks_undo_with_nothing_to_undo_does_not_panic_or_change_project_state() {
        let (ui, _results_ui) = test_ui_windows();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<ToolbarState>().invoke_undo_clicked();

        assert_eq!(ui_state.get_project().metadata.name, "");
    }
}
