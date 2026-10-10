use crate::{
    AppWindow, ResultsWindow, UiState,
    editor::{
        ai_learning_controller::AiLearningController,
        classification_controller::ClassificationController, focus_controller::FocusController,
        histogram_controller::HistogramController, image_meta_controller::ImageMetaController,
        images_list_controller::ImagesListController, object_list_controller::ObjectListController,
        pipeline_worker::PipelineWorker, pipelines_controller::PipelinesController,
        project_settings_controller::ProjectSettingsController,
        results_list_controller::ResultsListController,
        results_state_controller::ResultsStateController, template_controller::TemplateController,
        undo_redo_controller::UndoRedoController, viewport_controller::ViewportController,
        viewport_image_controller::ViewportImageController,
        viewport_object_controller::ViewPortObjectController,
    },
};
use std::sync::Arc;

pub mod ai_learning_controller;
pub mod classification_controller;
pub mod connection_controller;
pub mod focus_controller;
pub mod histogram_controller;
pub mod image_meta_controller;
pub mod images_list_controller;
pub mod object_list_controller;
pub mod pipeline_task;
pub mod pipeline_worker;
pub mod pipelines_controller;
pub mod project_controller;
pub mod project_settings_controller;
pub mod results_list_controller;
pub mod results_state_controller;
pub mod script_editor_controller;
pub mod template_controller;
#[cfg(test)]
pub(crate) mod test_support;
pub mod undo_redo_controller;
pub mod viewport_cache;
pub mod viewport_controller;
pub mod viewport_image_controller;
pub mod viewport_object_controller;
pub mod viewport_task;
pub mod viewport_worker;

pub struct Editor {
    image_list_controller: Arc<images_list_controller::ImagesListController>,
    project_controller: Arc<project_controller::ProjectController>,
    histogram_controller: Arc<histogram_controller::HistogramController>,
    viewport_image_controller: Arc<viewport_image_controller::ViewportImageController>,
    viewport_object_controller: Arc<viewport_object_controller::ViewPortObjectController>,
    viewport_worker: Arc<viewport_worker::ViewportWorker>,
    image_meta_controller: Arc<ImageMetaController>,
    project_settings_controller: Arc<ProjectSettingsController>,
    classification_controller: Arc<ClassificationController>,
    ai_learning_controller: Arc<AiLearningController>,
    object_list_controller: Arc<ObjectListController>,
    pipelines_controller: Arc<PipelinesController>,
    focus_controller: Arc<FocusController>,
    pipeline_worker: Arc<PipelineWorker>,
    results_list_controller: Arc<ResultsListController>,
    template_controller: Arc<TemplateController>,
    undo_redo_controller: Arc<UndoRedoController>,
    results_state_controller: Arc<ResultsStateController>,
    connection_controller: Arc<connection_controller::ConnectionController>,
    script_editor_controller: script_editor_controller::ScriptEditorController,
}

impl Editor {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        results_ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
    ) -> Self {
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), app_state.clone()));
        let view_port_cache = Arc::new(viewport_cache::ViewportCache::new(app_state.clone()));

        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
        ));

        let histogram_controller = Arc::new(HistogramController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
        ));

        let project_settings_controller = Arc::new(ProjectSettingsController::new(
            ui.clone(),
            results_ui.clone(),
            app_state.clone(),
        ));

        let classification_controller = Arc::new(ClassificationController::new(
            ui.clone(),
            app_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
        ));

        let ai_learning_controller =
            Arc::new(AiLearningController::new(ui.clone(), app_state.clone()));

        let image_meta_controller = Arc::new(ImageMetaController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
        ));

        let image_list_controller = Arc::new(ImagesListController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
            histogram_controller.clone(),
            image_meta_controller.clone(),
            object_list_controller.clone(),
            classification_controller.clone(),
        ));

        let results_state_controller = Arc::new(ResultsStateController::new(
            results_ui.clone(),
            app_state.clone(),
            image_list_controller.clone(),
        ));
        let results_list_controller = Arc::new(ResultsListController::new(
            ui.clone(),
            app_state.clone(),
            results_state_controller.clone(),
        ));

        let viewport_object_controller = Arc::new(ViewPortObjectController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
            view_port_cache.clone(),
            image_list_controller.clone(),
            object_list_controller.clone(),
            classification_controller.clone(),
        ));

        let template_controller = Arc::new(TemplateController::new(ui.clone(), app_state.clone()));

        let focus_controller = Arc::new(FocusController::new(
            ui.clone(),
            app_state.clone(),
            image_meta_controller.clone(),
            classification_controller.clone(),
            viewport_controller.clone(),
        ));

        let pipelines_controller = Arc::new(pipelines_controller::PipelinesController::new(
            ui.clone(),
            app_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
            template_controller.clone(),
            focus_controller.clone(),
        ));

        let project_controller = Arc::new(project_controller::ProjectController::new(
            ui.clone(),
            app_state.clone(),
            image_list_controller.clone(),
            project_settings_controller.clone(),
            classification_controller.clone(),
            pipelines_controller.clone(),
            results_list_controller.clone(),
            template_controller.clone(),
        ));

        let viewport_image_controller = Arc::new(ViewportImageController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
            view_port_cache.clone(),
            histogram_controller.clone(),
            image_meta_controller.clone(),
        ));

        let viewport_worker = Arc::new(viewport_worker::ViewportWorker::new(
            app_state.clone(),
            viewport_controller.clone(),
            histogram_controller.clone(),
            view_port_cache.clone(),
        ));

        let pipeline_worker = Arc::new(pipeline_worker::PipelineWorker::new(
            app_state.clone(),
            pipelines_controller.clone(),
            viewport_controller.clone(),
            object_list_controller.clone(),
            classification_controller.clone(),
            results_list_controller.clone(),
        ));

        let undo_redo_controller = Arc::new(UndoRedoController::new(
            ui.clone(),
            app_state.clone(),
            image_list_controller.clone(),
            project_settings_controller.clone(),
            classification_controller.clone(),
            pipelines_controller.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
        ));

        link_plate_settings(&project_settings_controller, &results_state_controller);

        let connection_controller = Arc::new(connection_controller::ConnectionController::new(
            ui.clone(),
            app_state.clone(),
            project_controller.clone(),
        ));

        let script_editor_controller =
            script_editor_controller::ScriptEditorController::new(ui.clone());

        Self {
            connection_controller,
            script_editor_controller,
            image_list_controller,
            project_controller,
            histogram_controller,
            viewport_image_controller,
            viewport_object_controller,
            viewport_worker,
            image_meta_controller,
            project_settings_controller,
            classification_controller,
            ai_learning_controller,
            object_list_controller,
            pipelines_controller,
            focus_controller,
            pipeline_worker,
            results_list_controller,
            template_controller,
            undo_redo_controller,
            results_state_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        self.image_list_controller.attach_callbacks();
        self.project_controller.attach_callbacks();
        self.histogram_controller.attach_callbacks();
        self.viewport_image_controller.attach_callbacks();
        self.image_meta_controller.attach_callbacks();
        self.project_settings_controller.attach_callbacks();
        self.classification_controller.attach_callbacks();
        self.ai_learning_controller.attach_callbacks();
        self.viewport_object_controller.attach_callbacks();
        self.object_list_controller.attach_callbacks();
        self.pipelines_controller.attach_callbacks();
        self.focus_controller.attach_callbacks();
        self.results_list_controller.attach_callbacks();
        self.template_controller.attach_callbacks();
        self.undo_redo_controller.attach_callbacks();
        self.results_state_controller.attach_callbacks();
        self.connection_controller.attach_callbacks();
        self.script_editor_controller.attach_callbacks();

        self.viewport_worker.start_worker();
        self.pipeline_worker.start_worker();
        self.pipelines_controller.follow_server_analyses();
    }

    /// Opens the project file at `path`, as "File > Open" does.
    pub fn open_project(&self, path: &std::path::PathBuf) {
        Arc::clone(&self.project_controller).open_new_project(path);
    }

    /// What to do once the server connection is back after it dropped:
    /// follow the analysis running there again. Callable from any thread.
    pub fn on_reconnected(&self) -> impl Fn() + Send + Sync + 'static {
        let pipelines = Arc::clone(&self.pipelines_controller);
        move || pipelines.follow_server_analyses()
    }
}

/// Keeps the project settings dialog and the results window on the same
/// plate settings (grouping, plate size, well layout): a change in either
/// is stored in the project and shown in the other.
pub(crate) fn link_plate_settings(
    project_settings: &Arc<ProjectSettingsController>,
    results: &Arc<ResultsStateController>,
) {
    let results_weak = Arc::downgrade(results);
    project_settings.on_plate_settings_changed(move || {
        let results_weak = results_weak.clone();
        // The results window refreshes its grids on the UI thread.
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(results) = results_weak.upgrade() {
                results.apply_project_plate_settings();
            }
        })
        .ok();
    });
    let project_settings_weak = Arc::downgrade(project_settings);
    results.on_plate_settings_written(move || {
        if let Some(project_settings) = project_settings_weak.upgrade() {
            project_settings.sync_project_settings_to_slint();
        }
    });
}

#[cfg(test)]
mod editor_new_tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;

    /// `Editor::new()` builds every controller in the app (see the struct
    /// literal at the end of `new()`) - none of their constructors touch the
    /// Slint platform (only `attach_callbacks()`/the worker threads do, which
    /// this deliberately doesn't call), so this exercises the entire
    /// dependency-wiring graph with a dead UI and asserts it doesn't panic.
    /// A future constructor that panics on a `None` upgrade, or a wiring
    /// mistake that passes the wrong controller instance to a dependent's
    /// constructor (a type error would catch a wrong *type*, but not a wrong
    /// *instance* of the same type), would surface here.
    #[test]
    fn new_wires_every_controller_without_panicking() {
        let ui_state = test_ui_state();
        let _editor = Editor::new(slint::Weak::default(), slint::Weak::default(), ui_state);
    }
}
