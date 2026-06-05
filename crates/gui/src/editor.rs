use crate::{
    AppWindow, ResultsWindow, UiState,
    editor::{
        classification_controller::ClassificationController,
        histogram_controller::HistogramController, image_meta_controller::ImageMetaController,
        images_list_controller::ImagesListController, pipeline_worker::PipelineWorker,
        pipelines_controller::PipelinesController,
        project_settings_controller::ProjectSettingsController,
        results_list_controller::ResultsListController,
        results_table_controller::ResultsTableController, roi_list_controller::RoiListController,
        viewport_controller::ViewportController,
        viewport_image_controller::ViewportImageController,
        viewport_roi_controller::ViewPortRoiController,
    },
};
use std::sync::Arc;

pub mod classification_controller;
pub mod histogram_controller;
pub mod image_meta_controller;
pub mod images_list_controller;
pub mod pipeline_task;
pub mod pipeline_worker;
pub mod pipelines_controller;
pub mod project_controller;
pub mod project_settings_controller;
pub mod results_list_controller;
pub mod results_table_controller;
pub mod roi_list_controller;
pub mod viewport_cache;
pub mod viewport_controller;
pub mod viewport_image_controller;
pub mod viewport_roi_controller;
pub mod viewport_task;
pub mod viewport_worker;

pub struct Editor {
    image_list_controller: Arc<images_list_controller::ImagesListController>,
    project_controller: Arc<project_controller::ProjectController>,
    histogram_controller: Arc<histogram_controller::HistogramController>,
    viewport_image_controller: Arc<viewport_image_controller::ViewportImageController>,
    viewport_roi_controller: Arc<viewport_roi_controller::ViewPortRoiController>,
    viewport_worker: Arc<viewport_worker::ViewportWorker>,
    image_meta_controller: Arc<ImageMetaController>,
    project_settings_controller: Arc<ProjectSettingsController>,
    classification_controller: Arc<ClassificationController>,
    roi_list_controller: Arc<RoiListController>,
    pipelines_controller: Arc<PipelinesController>,
    pipeline_worker: Arc<PipelineWorker>,
    results_table_controller: Arc<ResultsTableController>,
    results_list_controller: Arc<ResultsListController>,
}

impl Editor {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        results_ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
    ) -> Self {
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), app_state.clone()));
        let view_port_cache = Arc::new(viewport_cache::ViewportCache::new(app_state.clone()));
        let results_table_controller = Arc::new(ResultsTableController::new(
            results_ui.clone(),
            app_state.clone(),
        ));
        let results_list_controller = Arc::new(ResultsListController::new(
            ui.clone(),
            app_state.clone(),
            results_table_controller.clone(),
        ));

        let roi_list_controller = Arc::new(RoiListController::new(
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
            app_state.clone(),
        ));

        let classification_controller = Arc::new(ClassificationController::new(
            ui.clone(),
            app_state.clone(),
            roi_list_controller.clone(),
            viewport_controller.clone(),
        ));

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
            roi_list_controller.clone(),
        ));

        let viewport_roi_controller = Arc::new(ViewPortRoiController::new(
            ui.clone(),
            app_state.clone(),
            viewport_controller.clone(),
            view_port_cache.clone(),
            image_list_controller.clone(),
            roi_list_controller.clone(),
        ));

        let pipelines_controller = Arc::new(pipelines_controller::PipelinesController::new(
            ui.clone(),
            app_state.clone(),
            roi_list_controller.clone(),
            viewport_controller.clone(),
        ));

        let project_controller = Arc::new(project_controller::ProjectController::new(
            ui.clone(),
            app_state.clone(),
            image_list_controller.clone(),
            project_settings_controller.clone(),
            classification_controller.clone(),
            pipelines_controller.clone(),
            results_list_controller.clone(),
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
            roi_list_controller.clone(),
            classification_controller.clone(),
            results_list_controller.clone(),
        ));

        Self {
            image_list_controller,
            project_controller,
            histogram_controller,
            viewport_image_controller,
            viewport_roi_controller,
            viewport_worker,
            image_meta_controller,
            project_settings_controller,
            classification_controller,
            roi_list_controller,
            pipelines_controller,
            pipeline_worker,
            results_table_controller,
            results_list_controller,
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
        self.viewport_roi_controller.attach_callbacks();
        self.roi_list_controller.attach_callbacks();
        self.pipelines_controller.attach_callbacks();
        self.results_table_controller.attach_callbacks();
        self.results_list_controller.attach_callbacks();

        self.viewport_worker.start_worker();
        self.pipeline_worker.start_worker();
    }
}
