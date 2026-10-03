use crate::{
    DialogType, GlobalAppState, PipelineRunningState, PipelinesPanelState, UiState,
    editor::{
        classification_controller::ClassificationController,
        object_list_controller::ObjectListController, pipeline_task::PipelineTask,
        pipelines_controller::PipelinesController, results_list_controller::ResultsListController,
        viewport_controller::ViewportController,
    },
};
use evanalyzer_app::analysis::AnalysisRequest;
use evanalyzer_app::analysis::ProgressEvent;
use evanalyzer_app::preview::MAX_PREVIEW_VISIBLE_TILES;
use evanalyzer_app::preview::PreviewRequest;
use evanalyzer_app::preview::PreviewViewport;
use evanalyzer_app::preview::StartPreviewError;
use evanalyzer_cfg::core_types::{BreakpointSettings, InternalErrors};
use log::{error, info};
use slint::ComponentHandle;
use std::sync::{Arc, Condvar, Mutex};

pub struct PipelineWorker {
    pub(crate) app_state: Arc<UiState>,
    pub(crate) pipeline_controller: Arc<PipelinesController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
    pub(crate) results_list_controller: Arc<ResultsListController>,
}

impl PipelineWorker {
    pub fn new(
        app_state: Arc<UiState>,
        pipeline_controller: Arc<PipelinesController>,
        viewport_controller: Arc<ViewportController>,
        object_list_controller: Arc<ObjectListController>,
        classification_controller: Arc<ClassificationController>,
        results_list_controller: Arc<ResultsListController>,
    ) -> Self {
        Self {
            app_state,
            pipeline_controller,
            viewport_controller,
            object_list_controller,
            classification_controller,
            results_list_controller,
        }
    }

    pub(crate) fn start_worker(self: &Arc<Self>) {
        let self_handle = Arc::clone(self);
        std::thread::Builder::new()
            .name("PipelineWorker".into())
            .spawn(move || {
                crate::helper::worker_supervisor::run_supervised("PipelineWorker", || {
                    self_handle.run_worker_loop()
                })
            })
            .expect("Failed to spawn pipeline worker thread");
    }

    fn run_worker_loop(self: &Arc<Self>) -> ! {
        let task_request = &self.pipeline_controller.task_request;
        loop {
            let task = wait_for_task(task_request.clone());
            self.run_task(task);
        }
    }

    /// Runs one preview or analysis job to the end, reporting its progress
    /// and result to the UI.
    fn run_task(self: &Arc<Self>, task: PipelineTask) {
        let self_handle = Arc::clone(self);
        {
            let is_preview = task.preview;

            let job = if is_preview {
                // Restrict processing to tiles currently visible in the viewport
                // so the user sees results immediately.
                let viewport = {
                    let vp = self
                        .viewport_controller
                        .viewport_state
                        .read()
                        .expect("Failed to acquire read lock on viewport state");
                    PreviewViewport {
                        offset_x: vp.offset_x,
                        offset_y: vp.offset_y,
                        viewport_width: vp.viewport_width,
                        viewport_height: vp.viewport_height,
                        zoom: vp.zoom,
                    }
                };
                let breakpoint = task
                    .breakpoint
                    .map(|(pipeline_id, pipeline_step_id, mode)| BreakpointSettings {
                        pipeline_id,
                        pipeline_step_id,
                        mode,
                    });
                match self.app_state.backend().start_preview(PreviewRequest {
                    settings: task.project_settings,
                    project_path: task.project_path,
                    viewport,
                    breakpoint,
                }) {
                    Ok(job) => job,
                    Err(StartPreviewError::TooManyTiles { tiles }) => {
                        // Tell the user to zoom in instead of silently grinding
                        // through hundreds of tiles.
                        self.pipeline_controller.disable_auto_preview();
                        let ui_handle = self.app_state.ui_handle.clone();
                        let message = format!(
                            "Zoomed out too far to preview live: the visible area covers {tiles} tiles \
                             (max {MAX_PREVIEW_VISIBLE_TILES}). Zoom in, then run the preview again. \
                             Auto preview has been turned off."
                        );
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>()
                                    .set_status_message(message.into());
                                ui.global::<PipelineRunningState>().set_has_error(true);
                                ui.global::<PipelineRunningState>().set_done(true);
                            }
                        });
                        return;
                    }
                    Err(StartPreviewError::Failed(e)) => {
                        error!("Could not execute job: {e:?}");
                        return;
                    }
                }
            } else {
                match self.app_state.backend().start_analysis(AnalysisRequest {
                    settings: task.project_settings,
                    project_path: task.project_path,
                    job_name: task.job_name,
                    threads: None,
                }) {
                    Ok(job) => job,
                    Err(e) => {
                        error!("Could not execute job: {e:?}");
                        return;
                    }
                }
            };

            info!("Pipeline job started ...");

            *self
                .pipeline_controller
                .pipeline_cancel_flag
                .lock()
                .unwrap() = Some(job.cancel_handle());
            let mut last_ui_update = std::time::Instant::now();
            let mut pipeline_start: Option<std::time::Instant> = None;
            for event in job.events() {
                match event {
                    ProgressEvent::TilesScheduled { total_tiles } => {
                        let ui_handle = self.app_state.ui_handle.clone();
                        let total = total_tiles as i32;
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>().set_total(total);
                                ui.global::<PipelineRunningState>().set_processed(0);
                                ui.global::<PipelineRunningState>()
                                    .set_whole_image_phase(false);
                            }
                        });
                    }
                    ProgressEvent::Started { total } => {
                        info!("Pipeline started: {total} images to process");
                        pipeline_start = Some(std::time::Instant::now());
                        // Clear any stale preview ROIs so the incremental tile updates
                        // start from a clean slate.
                        if is_preview {
                            self_handle
                                .app_state
                                .get_project_write()
                                .tmp_settings
                                .preview_objects
                                .clear();
                        }
                        let ui_handle = self.app_state.ui_handle.clone();
                        let total = total as i32;
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>().set_done(false);
                                ui.global::<PipelineRunningState>().set_has_error(false);
                                ui.global::<PipelineRunningState>().set_total(total);
                                ui.global::<PipelineRunningState>().set_processed(0);
                                ui.global::<PipelineRunningState>()
                                    .set_whole_image_phase(false);
                            }
                        });
                    }
                    ProgressEvent::TileCompleted {
                        tile_index,
                        total_tiles,
                        objects,
                    } => {
                        info!("Tile {tile_index}/{total_tiles} completed");
                        if is_preview {
                            // Append the new ROIs and redraw so the user sees partial results.
                            self_handle
                                .app_state
                                .get_project_write()
                                .tmp_settings
                                .preview_objects
                                .extend(objects);
                            self_handle.object_list_controller.sync_objects_to_slint();
                            self_handle
                                .classification_controller
                                .sync_classification_to_slint();
                            self_handle
                                .viewport_controller
                                .trigger_image_redraw_objects();
                        }
                        let ui_handle = self.app_state.ui_handle.clone();
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>()
                                    .set_processed(tile_index as i32);
                                ui.global::<PipelineRunningState>()
                                    .set_total(total_tiles as i32);
                                ui.global::<PipelineRunningState>()
                                    .set_whole_image_phase(false);
                            }
                        });
                    }
                    ProgressEvent::WholeImagePhaseCompleted {
                        completed,
                        total_tiles,
                    } => {
                        // The whole-image-scoped phase (tile-merge, Voronoi,
                        // etc.) can itself run long after every tile has
                        // already reported `TileCompleted` - without this,
                        // the bar above would sit at 100% while the pipeline
                        // was still genuinely working. `total_tiles` already
                        // reserves a unit for this event (see its own doc
                        // comment), so this only needs to report progress,
                        // not recompute the total.
                        info!("Whole-image phase completed ({completed}/{total_tiles})");
                        let ui_handle = self.app_state.ui_handle.clone();
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>()
                                    .set_processed(completed as i32);
                                ui.global::<PipelineRunningState>()
                                    .set_total(total_tiles as i32);
                                ui.global::<PipelineRunningState>()
                                    .set_whole_image_phase(true);
                            }
                        });
                    }
                    ProgressEvent::ImageCompleted { index, total, path } => {
                        info!(
                            "Pipeline progress: {}/{} - {}",
                            index,
                            total,
                            path.display()
                        );
                        let secs_per_image = pipeline_start
                            .map(|t| secs_per_image(t.elapsed(), index))
                            .unwrap_or(0.0);
                        let eta_str = format!("{:.2}", secs_per_image);
                        let is_last = index == total;
                        let elapsed = last_ui_update.elapsed();
                        if is_last || elapsed >= std::time::Duration::from_millis(100) {
                            last_ui_update = std::time::Instant::now();
                            let ui_handle = self.app_state.ui_handle.clone();
                            let index = index as i32;
                            let total = total as i32;
                            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_handle.upgrade() {
                                    ui.global::<PipelineRunningState>().set_processed(index);
                                    ui.global::<PipelineRunningState>().set_total(total);
                                    ui.global::<PipelinesPanelState>()
                                        .set_eta_seconds_per_image(eta_str.into());
                                }
                            });
                        }
                    }
                    ProgressEvent::BreakpointReached {
                        image,
                        segmentation,
                        instances,
                        tile_offset_x,
                        tile_offset_y,
                        tile_width,
                        tile_height,
                        nr_bits,
                        channel_idx,
                    } => {
                        info!(
                            "Breakpoint image received for tile ({},{}) {}x{}",
                            tile_offset_x, tile_offset_y, tile_width, tile_height
                        );
                        // Store the raw ImageContainer so the viewport worker can
                        // re-render it with live histogram/LUT settings.
                        self_handle.viewport_controller.set_breakpoint_channel(
                            image,
                            segmentation,
                            instances,
                            tile_offset_x,
                            tile_offset_y,
                            tile_width,
                            tile_height,
                            nr_bits,
                            channel_idx,
                        );
                    }
                    ProgressEvent::ImageFailed { path } => {
                        error!("Pipeline image failed: {}", path.display());
                    }
                    ProgressEvent::Finished => {
                        info!("Pipeline job finished - waiting for result");
                        *self
                            .pipeline_controller
                            .pipeline_cancel_flag
                            .lock()
                            .unwrap() = None;
                    }
                }
            }
            // A panic inside the job thread comes back as a normal error
            // (see `RunningJob::wait`), so the user sees it and this worker
            // survives to run the next job.
            let job_result = job.wait();
            let (status_message, is_error) = match job_result {
                Err(InternalErrors::Cancelled) => {
                    info!("Pipeline cancelled by user");
                    ("Cancelled by user.".to_string(), false)
                }
                Err(e) => {
                    error!("Pipeline job error: {e:?}");
                    (format!("Error: {e}"), true)
                }
                Ok(output) => {
                    info!("Pipeline completed successfully");
                    if is_preview {
                        // Replace the incrementally-streamed per-tile ROIs with
                        // the final, tile-merged result - otherwise cross-tile
                        // fragments stay displayed as two separate objects even
                        // though the backend merged them.
                        if let Some(final_objects) = output.preview_objects {
                            let mut project = self_handle.app_state.get_project_write();
                            project.tmp_settings.preview_objects.clear();
                            project.tmp_settings.preview_objects.extend(final_objects);
                        }
                        self_handle.object_list_controller.sync_objects_to_slint();
                        self_handle
                            .classification_controller
                            .sync_classification_to_slint();
                        self_handle
                            .viewport_controller
                            .trigger_image_redraw_objects();
                    } else {
                        self_handle
                            .results_list_controller
                            .sync_results_files_to_slint();
                    }

                    ("Analysis completed successfully.".to_string(), false)
                }
            };
            let ui_handle = self.app_state.ui_handle.clone();
            let is_preview = task.preview;
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_handle.upgrade() {
                    ui.global::<PipelineRunningState>()
                        .set_status_message(status_message.into());
                    ui.global::<PipelineRunningState>().set_has_error(is_error);
                    ui.global::<PipelineRunningState>().set_done(true);

                    // For preview: auto-close on success or cancel; keep open on error
                    if is_preview && !is_error {
                        ui.global::<GlobalAppState>()
                            .set_active_dialog(DialogType::None);
                    }
                }
            });
        }
    }
}

/// Seconds elapsed per completed image, for the ETA display. `index` is
/// always >= 1 for a real `ImageCompleted` event (see job_executor.rs's
/// `completed.fetch_add(1, ..) + 1`), but `.max(1)` keeps this from ever
/// displaying "inf" if that invariant is ever violated.
fn secs_per_image(elapsed: std::time::Duration, index: usize) -> f64 {
    elapsed.as_secs_f64() / (index.max(1) as f64)
}

/// Waits for a pipeline task to become available, blocking until one is posted.
fn wait_for_task(task_request: Arc<(Mutex<Option<PipelineTask>>, Condvar)>) -> PipelineTask {
    let (lock, cvar) = &*task_request;
    let mut task_slot = lock.lock().unwrap();
    while task_slot.is_none() {
        task_slot = cvar.wait(task_slot).unwrap();
    }
    task_slot.take().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secs_per_image_treats_index_zero_as_one_instead_of_displaying_inf() {
        let elapsed = std::time::Duration::from_secs(10);
        assert_eq!(secs_per_image(elapsed, 0), 10.0);
        assert_eq!(secs_per_image(elapsed, 10), 1.0);
        assert_eq!(secs_per_image(elapsed, 1), 10.0);
    }

    #[test]
    fn wait_for_task_returns_immediately_if_a_task_is_already_posted() {
        let task_request = Arc::new((
            Mutex::new(Some(PipelineTask {
                preview: true,
                ..Default::default()
            })),
            Condvar::new(),
        ));

        let task = wait_for_task(task_request.clone());

        assert!(task.preview);
        // Taken out of the slot, not just peeked at.
        assert!(task_request.0.lock().unwrap().is_none());
    }

    #[test]
    fn wait_for_task_blocks_until_a_task_is_posted_from_another_thread() {
        let task_request = Arc::new((Mutex::<Option<PipelineTask>>::new(None), Condvar::new()));
        let poster = task_request.clone();

        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let (lock, cvar) = &*poster;
            *lock.lock().unwrap() = Some(PipelineTask {
                job_name: Some("posted-from-another-thread".to_string()),
                ..Default::default()
            });
            cvar.notify_one();
        });

        let task = wait_for_task(task_request);

        assert_eq!(task.job_name.as_deref(), Some("posted-from-another-thread"));
        handle.join().unwrap();
    }

    // -- running real jobs (UI updates applied via the test queue) -----------

    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::images_list_controller::ImagesListController;
    use crate::editor::results_state_controller::ResultsStateController;
    use crate::editor::template_controller::TemplateController;
    use crate::editor::test_support::{fixture_image_path, test_ui_windows, ui_state_with_windows};
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{AppWindow, ResultsWindow};
    use evanalyzer_app::backends::local::LocalBackend;
    use evanalyzer_app::project::{ProjectExt, ProjectWithRuntime};

    struct Fixture {
        ui: AppWindow,
        _results_ui: ResultsWindow,
        worker: Arc<PipelineWorker>,
        dir: tempfile::TempDir,
        settings: evanalyzer_cfg::settings::project_settings::ProjectSettings,
    }

    /// A worker wired to real windows and a project with a copy of the
    /// fixture image in a temp folder.
    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let images = dir.path().join("images");
        std::fs::create_dir_all(&images).unwrap();
        std::fs::copy(fixture_image_path(), images.join("fixture.ome.tif")).unwrap();
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(images);
        project.scan_image_folder_and_add(&LocalBackend::default());
        let settings = project.settings.clone();

        let (ui, results_ui) = test_ui_windows();
        let ui_state = ui_state_with_windows(&ui, &results_ui, project);
        let w = ui.as_weak();
        let viewport = Arc::new(ViewportController::new(w.clone(), ui_state.clone()));
        let objects = Arc::new(ObjectListController::new(
            w.clone(),
            ui_state.clone(),
            viewport.clone(),
        ));
        let templates = Arc::new(TemplateController::new(w.clone(), ui_state.clone()));
        let pipelines = Arc::new(PipelinesController::new(
            w.clone(),
            ui_state.clone(),
            objects.clone(),
            viewport.clone(),
            templates,
        ));
        let classification = Arc::new(ClassificationController::new(
            w.clone(),
            ui_state.clone(),
            objects.clone(),
            viewport.clone(),
        ));
        let images_list = Arc::new(ImagesListController::new(
            w.clone(),
            ui_state.clone(),
            viewport.clone(),
            Arc::new(HistogramController::new(
                w.clone(),
                ui_state.clone(),
                viewport.clone(),
            )),
            Arc::new(ImageMetaController::new(
                w.clone(),
                ui_state.clone(),
                viewport.clone(),
            )),
            objects.clone(),
        ));
        let results_state = Arc::new(ResultsStateController::new(
            results_ui.as_weak(),
            ui_state.clone(),
            images_list,
        ));
        let results_list = Arc::new(ResultsListController::new(
            w,
            ui_state.clone(),
            results_state,
        ));
        {
            let mut vp = viewport.viewport_state.write().unwrap();
            vp.viewport_width = 256.0;
            vp.viewport_height = 256.0;
            vp.zoom = 1.0;
        }
        let worker = Arc::new(PipelineWorker::new(
            ui_state,
            pipelines,
            viewport,
            objects,
            classification,
            results_list,
        ));
        Fixture {
            ui,
            _results_ui: results_ui,
            worker,
            dir,
            settings,
        }
    }

    impl Fixture {
        fn task(&self, preview: bool) -> PipelineTask {
            PipelineTask {
                project_settings: self.settings.clone(),
                project_path: self.dir.path().to_path_buf(),
                preview,
                breakpoint: None,
                job_name: Some("worker_test".into()),
            }
        }
        fn running(&self) -> PipelineRunningState<'_> {
            self.ui.global::<PipelineRunningState>()
        }
    }

    #[test]
    fn an_analysis_run_reports_progress_and_completion() {
        let f = fixture();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::PipelineRunning);
        f.worker.run_task(f.task(false));
        drain_ui_queue();
        let running = f.running();
        assert!(running.get_done());
        assert!(!running.get_has_error());
        assert_eq!(
            running.get_status_message(),
            "Analysis completed successfully."
        );
        assert!(running.get_total() >= 1);
        assert_eq!(running.get_processed(), running.get_total());
        assert!(
            !f.ui
                .global::<PipelinesPanelState>()
                .get_eta_seconds_per_image()
                .is_empty()
        );
        // The full run's dialog stays open to show the result.
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::PipelineRunning
        );
        assert!(
            f.worker
                .pipeline_controller
                .pipeline_cancel_flag
                .lock()
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_preview_run_closes_its_dialog_when_done() {
        let f = fixture();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::PreviewRendering);
        f.worker.run_task(f.task(true));
        drain_ui_queue();
        assert!(f.running().get_done());
        assert!(!f.running().get_has_error());
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::None
        );
    }

    /// Pins down today's behaviour: a job that can't even start (here its
    /// results folder can't be created) is only logged - the running dialog
    /// gets no result and stays as it was. Flagged as a usability gap; if
    /// that changes, this test should assert the shown error instead.
    #[test]
    fn a_job_that_cannot_start_is_only_logged() {
        let f = fixture();
        std::fs::write(f.dir.path().join("missing"), "a file, not a folder").unwrap();
        let mut task = f.task(false);
        task.project_path = f.dir.path().join("missing").join("deeper");
        f.worker.run_task(task);
        drain_ui_queue();
        assert!(!f.running().get_done());
        assert_eq!(f.running().get_status_message(), "");
    }
}
