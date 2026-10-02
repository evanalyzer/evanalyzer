use crate::{
    DialogType, GlobalAppState, PipelineRunningState, PipelinesPanelState, UiState,
    editor::{
        classification_controller::ClassificationController,
        object_list_controller::ObjectListController, pipeline_task::PipelineTask,
        pipelines_controller::PipelinesController, results_list_controller::ResultsListController,
        viewport_controller::ViewportController,
    },
};
use evanalyzer_app::api::AnalysisRequest;
use evanalyzer_app::api::MAX_PREVIEW_VISIBLE_TILES;
use evanalyzer_app::api::PreviewRequest;
use evanalyzer_app::api::PreviewViewport;
use evanalyzer_app::api::ProgressEvent;
use evanalyzer_app::api::StartPreviewError;
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
        let self_handle = Arc::clone(self);
        loop {
            let task = wait_for_task(task_request.clone());
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
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>()
                                    .set_status_message(message.into());
                                ui.global::<PipelineRunningState>().set_has_error(true);
                                ui.global::<PipelineRunningState>().set_done(true);
                            }
                        });
                        continue;
                    }
                    Err(StartPreviewError::Failed(e)) => {
                        error!("Could not execute job: {e:?}");
                        continue;
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
                        continue;
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
                        let _ = slint::invoke_from_event_loop(move || {
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
                        let _ = slint::invoke_from_event_loop(move || {
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
                        let _ = slint::invoke_from_event_loop(move || {
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
                        let _ = slint::invoke_from_event_loop(move || {
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
                            let _ = slint::invoke_from_event_loop(move || {
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
            let _ = slint::invoke_from_event_loop(move || {
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
}
