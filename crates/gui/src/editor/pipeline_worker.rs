//! # pipeline_worker
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::editor::prediction_overlay::PredictionTile;
use crate::editor::viewport_controller::BreakpointChannelData;
use crate::{
    PipelineRunningState, PipelinesPanelState, PreviewRunState, UiState,
    editor::{
        classification_controller::ClassificationController,
        object_list_controller::ObjectListController, pipeline_task::PipelineTask,
        pipelines_controller::PipelinesController, results_list_controller::ResultsListController,
        viewport_controller::ViewportController,
    },
};
use evanalyzer_app::analysis::AnalysisRequest;
use evanalyzer_app::analysis::ProgressEvent;
use evanalyzer_app::preview::PreviewRequest;
use evanalyzer_app::preview::PreviewViewport;
use evanalyzer_app::preview::StartPreviewError;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::{BreakpointSettings, InternalErrors, PipelineId, StepId};
use evanalyzer_cfg::settings::pipeline_settings::StepContent;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use log::{error, info};
use slint::ComponentHandle;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// Runs `$body` with `$state` the progress state of the job: the preview's
/// (`PreviewRunState`) or the analysis's (`PipelineRunningState`) - both
/// have the same properties.
macro_rules! with_progress {
    ($ui:expr, $preview:expr, |$state:ident| $body:block) => {
        if $preview {
            let $state = $ui.global::<PreviewRunState>();
            $body
        } else {
            let $state = $ui.global::<PipelineRunningState>();
            $body
        }
    };
}

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

    /// Previews run here, one after the other. An analysis gets a thread of
    /// its own: previews can run while it does, once its steps are unlocked
    /// (`PipelinesPanelState.run-locked`).
    fn run_worker_loop(self: &Arc<Self>) -> ! {
        let task_request = &self.pipeline_controller.task_request;
        loop {
            let task = wait_for_task(task_request.clone());
            self.start_task(task);
        }
    }

    /// Runs a preview here and now; starts an analysis on its own thread
    /// (marked running before this returns) and returns that thread.
    fn start_task(self: &Arc<Self>, task: PipelineTask) -> Option<std::thread::JoinHandle<()>> {
        if task.preview {
            self.run_task(task);
            return None;
        }
        let running = &self.pipeline_controller.analysis_running;
        running.store(true, std::sync::atomic::Ordering::SeqCst);
        let worker = Arc::clone(self);
        Some(crate::helper::ui_thread::spawn(move || {
            worker.run_task(task)
        }))
    }

    /// The image shown and its series - what a preview runs for.
    fn shown_image(&self) -> (Option<std::path::PathBuf>, i32) {
        let project = self.app_state.get_project();
        (
            project.get_current_image_path_cloned(),
            project.get_selected_series_idx(),
        )
    }

    /// A job that could not start (e.g. a pipeline reads a channel the images
    /// don't have): logged, and shown to the user in the run status - the
    /// same place "zoomed out too far to preview" is reported.
    fn report_start_failure(&self, preview: bool, what: &str, e: &InternalErrors) {
        error!("{what} could not be started: {e}");
        let message = format!("{what} could not be started: {e}");
        if !preview {
            self.pipeline_controller
                .analysis_running
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let ui_handle = self.app_state.ui_handle.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_handle.upgrade() {
                with_progress!(ui, preview, |running| {
                    running.set_status_message(message.into());
                    running.set_has_error(true);
                    running.set_done(true);
                });
            }
        });
    }

    /// Runs one preview or analysis job to the end, reporting its progress
    /// and result to the UI.
    fn run_task(self: &Arc<Self>, task: PipelineTask) {
        let self_handle = Arc::clone(self);
        {
            let is_preview = task.preview;
            let whole_image = task.whole_image;
            // The image (and series) a preview runs for: its objects go on
            // that one only - not on the next image, if the user switched
            // meanwhile.
            let preview_of = self.shown_image();

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
                let breakpoint =
                    task.breakpoint
                        .map(|(pipeline_id, step_id, mode)| BreakpointSettings {
                            pipeline_id,
                            step_id,
                            mode,
                        });
                // The prediction overlay: the pixel classifier step it shows
                // sends each tile's prediction.
                let overlay = self.viewport_controller.prediction.settings();
                let prediction = overlay
                    .on
                    .then(|| {
                        evanalyzer_app::ai_learning::prediction_target(
                            &task.project_settings,
                            overlay.model.as_deref(),
                        )
                    })
                    .flatten();
                // Nothing to show: the panel says why.
                let hint = match (&prediction, overlay.on, &overlay.model) {
                    (None, true, Some(model)) => format!(
                        "No pipeline applies the model '{}' yet - \"Add to pipeline\" adds one.",
                        model
                            .file_stem()
                            .map(|name| name.to_string_lossy())
                            .unwrap_or_default()
                    ),
                    _ => String::new(),
                };
                let ui_handle = self.app_state.ui_handle.clone();
                let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_handle.upgrade() {
                        ui.global::<crate::AiLearningState>()
                            .set_prediction_hint(hint.into());
                    }
                });
                // Tiles: the preview's limit (auto: the running computer's),
                // the breakpoint's this computer's - its images are kept here.
                let memory = self.viewport_controller.memory_settings();
                match self.app_state.backend().start_preview(PreviewRequest {
                    settings: task.project_settings,
                    project_path: task.project_path,
                    viewport,
                    breakpoint,
                    prediction,
                    max_visible_tiles: memory.requested_preview_tiles(),
                    breakpoint_tiles: memory.effective_breakpoint_tiles(
                        crate::editor::viewport_cache::this_computer_ram_bytes(),
                    ),
                    whole_image,
                }) {
                    Ok(job) => job,
                    Err(StartPreviewError::TooManyTiles { tiles, max }) => {
                        // Tell the user to zoom in instead of silently grinding
                        // through hundreds of tiles.
                        self.pipeline_controller.disable_auto_preview();
                        let ui_handle = self.app_state.ui_handle.clone();
                        let message = format!(
                            "Zoomed out too far to preview live: the visible area covers {tiles} tiles \
                             (max {max}, Settings › Viewport). Zoom in, then run the preview again. \
                             Auto preview has been turned off."
                        );
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                let preview = ui.global::<PreviewRunState>();
                                preview.set_status_message(message.into());
                                preview.set_has_error(true);
                                preview.set_done(true);
                            }
                        });
                        return;
                    }
                    Err(StartPreviewError::Failed(e)) => {
                        self.report_start_failure(true, "The preview", &e);
                        return;
                    }
                }
            } else {
                let backend = self.app_state.backend();
                let started = match &task.attach {
                    Some(id) => backend.attach_job(id),
                    None => backend.start_analysis(AnalysisRequest {
                        settings: task.project_settings,
                        project_path: task.project_path,
                        job_name: task.job_name,
                        threads: None,
                        base_results: task.base_results,
                    }),
                };
                match started {
                    Ok(job) => job,
                    Err(e) => {
                        self.report_start_failure(false, "The analysis", &e);
                        return;
                    }
                }
            };
            let job_id = job.id().map(str::to_string);
            let output_dir = job.output_path().clone();

            info!("Pipeline job started ...");

            // Cancel stops this job: the preview's or the analysis's.
            let cancel_slot = if is_preview {
                &self.pipeline_controller.preview_cancel_flag
            } else {
                &self.pipeline_controller.pipeline_cancel_flag
            };
            *cancel_slot.lock().unwrap() = Some(job.cancel_handle());
            let mut last_ui_update = std::time::Instant::now();
            let mut pipeline_start: Option<std::time::Instant> = None;
            // A run's first breakpoint tile replaces the last run's ones.
            let mut first_breakpoint_tile = true;
            // The same for the prediction overlay's tiles.
            let mut first_prediction_tile = true;
            // The images in work, in the order they started; the status bar
            // names the oldest.
            let mut in_work: Vec<PathBuf> = Vec::new();
            let mut shown_in_work = String::new();
            for event in job.events() {
                match &event {
                    ProgressEvent::ImageStarted { path } => in_work.push(path.clone()),
                    ProgressEvent::ImageCompleted { path, .. }
                    | ProgressEvent::ImageFailed { path } => in_work.retain(|p| p != path),
                    _ => {}
                }
                // Not throttled like the counters below: a dropped update
                // would leave a finished image named until the next event.
                let text = in_work_text(&in_work);
                if !is_preview && text != shown_in_work {
                    shown_in_work = text.clone();
                    let ui_handle = self.app_state.ui_handle.clone();
                    let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_handle.upgrade() {
                            ui.global::<PipelineRunningState>()
                                .set_current_image(text.into());
                        }
                    });
                }
                match event {
                    ProgressEvent::ImageStarted { path } => {
                        log::debug!("Image started: {}", path.display());
                    }
                    ProgressEvent::TilesScheduled { total_tiles } => {
                        let ui_handle = self.app_state.ui_handle.clone();
                        let total = total_tiles as i32;
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                with_progress!(ui, is_preview, |running| {
                                    running.set_total(total);
                                    running.set_processed(0);
                                });
                                if !is_preview {
                                    let running = ui.global::<PipelineRunningState>();
                                    running.set_whole_image_phase(false);
                                    running.set_tiles_total(total);
                                    running.set_tiles_done(0);
                                }
                            }
                        });
                    }
                    ProgressEvent::Started { total } => {
                        info!("Pipeline started: {total} images to process");
                        pipeline_start = Some(std::time::Instant::now());
                        // Clear any stale preview ROIs so the incremental tile updates
                        // start from a clean slate.
                        if is_preview {
                            let mut project = self_handle.app_state.get_project_write();
                            project.tmp_settings.preview_objects.clear();
                            project.tmp_settings.preview_measurements.clear();
                        }
                        let ui_handle = self.app_state.ui_handle.clone();
                        let total = total as i32;
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                with_progress!(ui, is_preview, |running| {
                                    running.set_done(false);
                                    running.set_has_error(false);
                                    running.set_total(total);
                                    running.set_processed(0);
                                });
                                if !is_preview {
                                    let running = ui.global::<PipelineRunningState>();
                                    running.set_whole_image_phase(false);
                                    running.set_images_total(total);
                                    running.set_images_done(0);
                                }
                            }
                        });
                    }
                    ProgressEvent::TileCompleted {
                        tile_index,
                        total_tiles,
                        objects,
                    } => {
                        log::debug!("Tile {tile_index}/{total_tiles} completed");
                        if is_preview && self_handle.shown_image() == preview_of {
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
                        // One image tile by tile: the time left by tiles.
                        let eta = pipeline_start
                            .filter(|_| tile_index > 0 && !is_preview)
                            .map(|t| {
                                let per_tile = t.elapsed().as_secs_f64() / tile_index as f64;
                                time_left(per_tile * total_tiles.saturating_sub(tile_index) as f64)
                            })
                            .unwrap_or_default();
                        let ui_handle = self.app_state.ui_handle.clone();
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                with_progress!(ui, is_preview, |running| {
                                    running.set_processed(tile_index as i32);
                                    running.set_total(total_tiles as i32);
                                });
                                if !is_preview {
                                    let running = ui.global::<PipelineRunningState>();
                                    running.set_whole_image_phase(false);
                                    running.set_tiles_done(tile_index as i32);
                                    running.set_tiles_total(total_tiles as i32);
                                    if running.get_images_total() <= 1 {
                                        running.set_eta(eta.into());
                                    }
                                }
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
                        log::debug!("Whole-image phase completed ({completed}/{total_tiles})");
                        let ui_handle = self.app_state.ui_handle.clone();
                        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                with_progress!(ui, is_preview, |running| {
                                    running.set_processed(completed as i32);
                                    running.set_total(total_tiles as i32);
                                });
                                if !is_preview {
                                    ui.global::<PipelineRunningState>()
                                        .set_whole_image_phase(true);
                                }
                            }
                        });
                    }
                    ProgressEvent::ImageCompleted { index, total, path } => {
                        log::debug!(
                            "Pipeline progress: {}/{} - {}",
                            index,
                            total,
                            path.display()
                        );
                        let secs_per_image = pipeline_start
                            .map(|t| secs_per_image(t.elapsed(), index))
                            .unwrap_or(0.0);
                        let eta_str = format!("{:.2}", secs_per_image);
                        let left = time_left(secs_per_image * total.saturating_sub(index) as f64);
                        let is_last = index == total;
                        let elapsed = last_ui_update.elapsed();
                        if is_last || elapsed >= std::time::Duration::from_millis(100) {
                            last_ui_update = std::time::Instant::now();
                            let ui_handle = self.app_state.ui_handle.clone();
                            let index = index as i32;
                            let total = total as i32;
                            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_handle.upgrade() {
                                    with_progress!(ui, is_preview, |running| {
                                        running.set_processed(index);
                                        running.set_total(total);
                                    });
                                    if !is_preview {
                                        let running = ui.global::<PipelineRunningState>();
                                        running.set_images_done(index);
                                        running.set_images_total(total);
                                        running.set_eta(left.into());
                                    }
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
                        log::debug!(
                            "Breakpoint image received for tile ({},{}) {}x{}",
                            tile_offset_x,
                            tile_offset_y,
                            tile_width,
                            tile_height
                        );
                        // Store the raw ImageContainer so the viewport worker can
                        // re-render it with live histogram/LUT settings.
                        self_handle.viewport_controller.add_breakpoint_tile(
                            BreakpointChannelData {
                                image: Arc::new(image),
                                segmentation: segmentation.map(Arc::new),
                                instances: instances.map(Arc::new),
                                tile_offset_x,
                                tile_offset_y,
                                tile_width,
                                tile_height,
                                nr_bits,
                                channel_idx,
                            },
                            first_breakpoint_tile,
                        );
                        first_breakpoint_tile = false;
                    }
                    ProgressEvent::ImageFailed { path } => {
                        error!("Pipeline image failed: {}", path.display());
                    }
                    ProgressEvent::PredictionReached {
                        capture,
                        tile_offset_x,
                        tile_offset_y,
                    } => {
                        self_handle.viewport_controller.add_prediction_tile(
                            PredictionTile {
                                tile_offset_x,
                                tile_offset_y,
                                capture,
                            },
                            first_prediction_tile,
                        );
                        first_prediction_tile = false;
                    }
                    ProgressEvent::Finished => {
                        info!("Pipeline job finished - waiting for result");
                        *cancel_slot.lock().unwrap() = None;
                    }
                }
            }
            // A panic inside the job thread comes back as a normal error
            // (see `RunningJob::wait`), so the user sees it and this worker
            // survives to run the next job.
            let job_result = job.wait();
            // A preview without a prediction (overlay off, no pixel
            // classifier step any more): the last one's would be stale.
            if is_preview && first_prediction_tile && job_result.is_ok() {
                self.viewport_controller.clear_prediction();
            }
            if !is_preview {
                self.pipeline_controller
                    .analysis_running
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }
            let backend = self.app_state.backend();
            let connection_lost = job_result.is_err() && !backend.is_connected();
            // Seen through to its end here: no need to report it again
            // the next time a window opens.
            if let (Some(id), false) = (&job_id, connection_lost) {
                if let Err(e) = backend.forget_job(id) {
                    log::warn!("Could not forget analysis {id} on the server: {e}");
                }
            }
            // The whole image's objects, for its note.
            let mut objects_found = None;
            // The step a run failed in, if the error says.
            let mut failed_at = None;
            let (status_message, is_error) = match job_result {
                Err(_) if connection_lost && job_id.is_some() => (
                    "The connection to the server was lost. The analysis keeps running \
                     there - it is shown again once EVAnalyzer is connected."
                        .to_string(),
                    true,
                ),
                Err(InternalErrors::Cancelled) => {
                    info!("Pipeline cancelled by user");
                    ("Cancelled by user.".to_string(), false)
                }
                Err(e) => {
                    error!("Pipeline job error: {e:?}");
                    failed_at = e.step();
                    // The preview names its step itself (`failed-where`) and
                    // is of the image shown; a run keeps the image's path.
                    let message = if is_preview {
                        e.message()
                    } else {
                        e.to_string()
                    };
                    (message, true)
                }
                Ok(output) => {
                    info!("Pipeline completed successfully");
                    if is_preview && self_handle.shown_image() != preview_of {
                        info!(
                            "The preview's image isn't shown any more - its objects are left out"
                        );
                    } else if is_preview {
                        // Replace the incrementally-streamed per-tile ROIs with
                        // the final, tile-merged result - otherwise cross-tile
                        // fragments stay displayed as two separate objects even
                        // though the backend merged them.
                        if let Some(final_objects) = output.preview_objects {
                            objects_found = Some(final_objects.len());
                            let mut project = self_handle.app_state.get_project_write();
                            project.tmp_settings.preview_objects.clear();
                            project.tmp_settings.preview_objects.extend(final_objects);
                        }
                        // Its measurements - like its objects.
                        if let Some(measurements) = output.preview_measurements {
                            self_handle
                                .app_state
                                .get_project_write()
                                .tmp_settings
                                .preview_measurements = measurements;
                        }
                        let previewed = self_handle
                            .app_state
                            .get_project()
                            .get_current_image_path_cloned();
                        if let Some(path) = previewed {
                            self_handle.app_state.mark_previewed(path);
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
                            .analysis_finished(&output_dir);
                    }

                    ("Analysis completed successfully.".to_string(), false)
                }
            };
            let failed_where = failed_at
                .map(|(pipeline, step)| {
                    failed_step_text(&self.app_state.get_project(), pipeline, step)
                })
                .unwrap_or_default();
            let ui_handle = self.app_state.ui_handle.clone();
            let is_preview = task.preview;
            let pipelines = self.pipeline_controller.clone();
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_handle.upgrade() {
                    with_progress!(ui, is_preview, |running| {
                        running.set_status_message(status_message.into());
                        running.set_has_error(is_error);
                        running.set_done(true);
                    });
                    if is_preview {
                        let preview = ui.global::<PreviewRunState>();
                        let (pipeline, step) = failed_at.map_or((0, 0), |(p, s)| (p.0, s.0));
                        preview.set_failed_where(failed_where.into());
                        preview.set_failed_pipeline(pipeline as i32);
                        preview.set_failed_step(step as i32);
                    }
                    // The steps unlock with the run's end (a lost server
                    // connection isn't one: it goes on there).
                    if !is_preview && !connection_lost {
                        pipelines.analysis_ended(&ui);
                    }

                    // The preview's pill goes on success or cancel, and stays
                    // on an error - and with the whole image's count. Open
                    // dialogs stay open.
                    if is_preview && !is_error {
                        let preview = ui.global::<PreviewRunState>();
                        match objects_found.filter(|_| whole_image) {
                            Some(found) => preview.set_objects(found as i32),
                            None => preview.set_shown(false),
                        }
                    }
                }
            });
        }
    }
}

/// Where a run failed, for people: "cellpose › #1 Blur" - the pipeline, and
/// the step as its card shows it (its id and the command's or module's
/// name). Empty if the project no longer has the step.
fn failed_step_text(project: &ProjectSettings, pipeline: PipelineId, step: StepId) -> String {
    let Some(pipeline) = project.pipelines.iter().find(|p| p.id == pipeline) else {
        return String::new();
    };
    let Some(step) = pipeline.steps.iter().find(|s| s.id == step) else {
        return String::new();
    };
    let name = match &step.content {
        StepContent::Command(command) => command.name().to_string(),
        StepContent::Module(instance) => project
            .modules
            .iter()
            .find(|d| d.key == instance.key)
            .map_or_else(|| instance.key.clone(), |d| d.meta.name.clone()),
    };
    format!("{} › #{} {name}", pipeline.name.trim(), step.id.0)
}

/// Seconds elapsed per completed image, for the ETA display. `index` is
/// always >= 1 for a real `ImageCompleted` event (see job_executor.rs's
/// `completed.fetch_add(1, ..) + 1`), but `.max(1)` keeps this from ever
/// displaying "inf" if that invariant is ever violated.
/// "~40 s", "~4 min", "~2 h 10 min" - "" when nothing is left.
fn time_left(secs: f64) -> String {
    if !secs.is_finite() || secs < 1.0 {
        return String::new();
    }
    let secs = secs.round() as u64;
    match secs {
        0..60 => format!("~{secs} s"),
        60..3600 => format!("~{} min", (secs + 30) / 60),
        _ => format!("~{} h {} min", secs / 3600, (secs % 3600) / 60),
    }
}

/// The status bar's "in work" text: the image started first of those still
/// running, and how many more run beside it.
fn in_work_text(in_work: &[PathBuf]) -> String {
    let Some(first) = in_work.first() else {
        return String::new();
    };
    let name = first
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    match in_work.len() {
        1 => name,
        n => format!("{name} (+{})", n - 1),
    }
}

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
    use crate::{DialogType, GlobalAppState};

    #[test]
    fn the_time_left_is_rounded_to_what_matters() {
        assert_eq!(time_left(0.4), "");
        assert_eq!(time_left(f64::INFINITY), "");
        assert_eq!(time_left(42.0), "~42 s");
        assert_eq!(time_left(239.0), "~4 min");
        assert_eq!(time_left(7800.0), "~2 h 10 min");
    }

    #[test]
    fn the_status_bar_names_the_oldest_image_in_work() {
        assert_eq!(in_work_text(&[]), "");
        assert_eq!(in_work_text(&[PathBuf::from("plate/a.tif")]), "a.tif");
        assert_eq!(
            in_work_text(&[
                PathBuf::from("plate/a.tif"),
                PathBuf::from("b.tif"),
                PathBuf::from("c.tif")
            ]),
            "a.tif (+2)"
        );
    }

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

    use crate::AppWindow;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::images_list_controller::ImagesListController;
    use crate::editor::results_state_controller::ResultsStateController;
    use crate::editor::template_controller::TemplateController;
    use crate::editor::test_support::{fixture_image_path, test_ui_window};
    use crate::helper::ui_thread::drain_ui_queue;
    use evanalyzer_app::backends::local::LocalBackend;
    use evanalyzer_app::project::{ProjectExt, ProjectWithRuntime};

    struct Fixture {
        ui: AppWindow,
        worker: Arc<PipelineWorker>,
        dir: tempfile::TempDir,
        settings: evanalyzer_cfg::settings::project_settings::ProjectSettings,
    }

    /// A worker wired to real windows and a project with a copy of the
    /// fixture image in a temp folder.
    fn fixture() -> Fixture {
        fixture_on(crate::editor::test_support::test_project_owner())
    }

    /// [`fixture`] on `owner`'s backend.
    fn fixture_on(owner: evanalyzer_app::project::ProjectOwner) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let images = dir.path().join("images");
        std::fs::create_dir_all(&images).unwrap();
        std::fs::copy(fixture_image_path(), images.join("fixture.ome.tif")).unwrap();
        let mut project = ProjectWithRuntime::default();
        project.images.root = Some(images);
        project.scan_image_folder_and_add(&LocalBackend::default());
        let settings = project.settings.clone();

        let ui = test_ui_window();
        let ui_state = crate::editor::test_support::ui_state_with_window_on(&ui, project, owner);
        let w = ui.as_weak();
        let viewport = Arc::new(ViewportController::new(
            w.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
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
            crate::editor::test_support::test_focus_controller(
                w.clone(),
                &ui_state,
                &objects,
                &viewport,
            ),
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
            classification.clone(),
        ));
        let results_state = Arc::new(ResultsStateController::new(
            ui.as_weak(),
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
                whole_image: false,
                breakpoint: None,
                job_name: Some("worker_test".into()),
                attach: None,
                base_results: None,
            }
        }
        fn running(&self) -> PipelineRunningState<'_> {
            self.ui.global::<PipelineRunningState>()
        }
    }

    #[test]
    fn an_analysis_run_reports_progress_and_completion() {
        let f = fixture();
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
        // Apart from the mixed counter: the images; none in work at the end.
        assert_eq!(running.get_images_done(), running.get_images_total());
        assert!(running.get_current_image().is_empty());
        assert!(
            f.worker
                .pipeline_controller
                .pipeline_cancel_flag
                .lock()
                .unwrap()
                .is_none()
        );
    }

    /// A remote fixture: an in-process worker serving the local backend,
    /// and the GUI connected to it.
    fn remote_fixture() -> (Fixture, Arc<dyn evanalyzer_app::backends::Backend>) {
        use evanalyzer_app::backends::remote::{RemoteBackend, Worker};
        let worker = Worker::bind("127.0.0.1:0", "gui-test".into()).unwrap();
        let url = format!("ws://{}", worker.local_addr().unwrap());
        std::thread::spawn(move || worker.run(Arc::new(LocalBackend::default())));
        let remote: Arc<dyn evanalyzer_app::backends::Backend> =
            Arc::new(RemoteBackend::connect(&url, "gui-test").unwrap());
        let owner = evanalyzer_app::project::ProjectOwner::with_backend(Arc::clone(&remote));
        (fixture_on(owner), remote)
    }

    #[test]
    fn an_analysis_running_on_the_server_is_followed_to_its_end_and_then_forgotten() {
        let (f, remote) = remote_fixture();
        // Started elsewhere - e.g. by this window before the connection dropped.
        let mut task = f.task(false);
        let job = remote
            .start_analysis(AnalysisRequest {
                settings: task.project_settings.clone(),
                project_path: task.project_path.clone(),
                job_name: Some("elsewhere".into()),
                threads: Some(1),
                base_results: None,
            })
            .unwrap();
        let id = job.id().unwrap().to_string();

        task.attach = Some(id.clone());
        f.worker.run_task(task);
        drain_ui_queue();

        assert!(f.running().get_done());
        assert!(
            !f.running().get_has_error(),
            "{}",
            f.running().get_status_message()
        );
        assert_eq!(
            f.running().get_status_message(),
            "Analysis completed successfully."
        );
        assert!(f.running().get_total() >= 1);
        let analysis_running = &f.worker.pipeline_controller.analysis_running;
        assert!(!analysis_running.load(std::sync::atomic::Ordering::SeqCst));
        assert!(
            remote.list_jobs().unwrap().iter().all(|j| j.id != id),
            "shown here, so not reported again"
        );
        drop(job);
    }

    #[test]
    fn attaching_where_nothing_is_tracked_is_reported() {
        let f = fixture();
        let mut task = f.task(false);
        task.attach = Some("0123456789abcdef".into());
        f.worker.run_task(task);
        drain_ui_queue();
        assert!(f.running().get_has_error());
        assert!(
            f.running()
                .get_status_message()
                .contains("could not be started"),
            "{}",
            f.running().get_status_message()
        );
        let analysis_running = &f.worker.pipeline_controller.analysis_running;
        assert!(!analysis_running.load(std::sync::atomic::Ordering::SeqCst));
    }

    /// A preview failing in a step: the pill's card says where ("cellpose ›
    /// #3 …"), which step "Show step" opens, and the message without "Error:"
    /// or the image's path.
    #[test]
    fn a_failed_preview_says_the_step_it_failed_in() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
        let f = fixture();
        let mut texture = PipelineCommand::default_for_key("measure_image_texture").unwrap();
        let PipelineCommand::MeasureImageTexture(settings) = &mut texture else {
            panic!("texture")
        };
        settings.id = "texture_01".into();
        settings.gray_levels = 1; // refused: 2 to 256
        let pipeline = PipelineSettings {
            id: PipelineId(7),
            name: "cellpose".into(),
            description: None,
            image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
            enabled: true,
            steps: vec![PipelineStepSettings {
                id: StepId(3),
                enabled: true,
                content: StepContent::Command(texture),
            }],
        };
        f.worker
            .app_state
            .get_project_write()
            .pipelines
            .push(pipeline.clone());
        let mut task = f.task(true);
        task.project_settings.pipelines.push(pipeline);
        f.ui.global::<PreviewRunState>().set_shown(true);

        f.worker.run_task(task);
        drain_ui_queue();

        let preview = f.ui.global::<PreviewRunState>();
        assert!(preview.get_has_error());
        assert!(preview.get_shown(), "the error stays");
        assert_eq!(
            preview.get_failed_where(),
            "cellpose › #3 Measure Image Texture"
        );
        assert_eq!(preview.get_failed_pipeline(), 7);
        assert_eq!(preview.get_failed_step(), 3);
        let message = preview.get_status_message();
        assert!(message.contains("gray levels 2 to 256"), "{message}");
        assert!(
            !message.contains("Error") && !message.contains("fixture.ome.tif"),
            "{message}"
        );
    }

    /// A pixel classifier pipeline (step 5) applying a saved model that
    /// tells dark from bright pixels.
    fn pixel_classifier_pipeline(
        f: &Fixture,
    ) -> evanalyzer_cfg::settings::pipeline_settings::PipelineSettings {
        use evanalyzer_cfg::core_types::SegmentationClass;
        use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
        use evanalyzer_cfg::settings::ai_learning_settings::{
            AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings,
            PixelClassLabel, PixelInputColor, RandomForestSettings,
        };
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            PixelClassifierSettings, SegmentationMappingSettings,
        };
        use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};
        use evanalyzer_core::ai_learning::model::{
            CURRENT_SAVED_CLASSIFIER_VERSION, random_forest::fit_random_forest,
        };
        let rows: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32 / 40.0]).collect();
        let labels: Vec<usize> = (0..40).map(|i| usize::from(i >= 4)).collect();
        let forest = RandomForestSettings {
            n_trees: 10,
            seed: 1,
            ..Default::default()
        };
        let saved = evanalyzer_core::SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier: fit_random_forest(&rows, &labels, &forest).unwrap(),
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: Default::default(),
                backend: AiLearningBackendSettings::RandomForest(forest),
                classifier: AiLearningClassifierSettings::Pixel {
                    feature_spec: AiLearningPixelFeatureSettings {
                        channels: vec![vec![]],
                        image_channels: Vec::new(),
                        image_channel_names: Vec::new(),
                    },
                    input_color: PixelInputColor::Gray,
                    class_labels: vec![
                        PixelClassLabel {
                            class: SegmentationClass(5),
                            name: "Background".into(),
                        },
                        PixelClassLabel {
                            class: SegmentationClass(6),
                            name: "Cell".into(),
                        },
                    ],
                },
            },
        };
        let model = f.dir.path().join("cells.evamodel");
        evanalyzer_core::save_classifier_to_file(&saved, &model).unwrap();
        PipelineSettings {
            id: PipelineId(4),
            name: "cells".into(),
            description: None,
            image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
            enabled: true,
            steps: vec![PipelineStepSettings {
                id: StepId(5),
                enabled: true,
                content: StepContent::Command(PipelineCommand::PixelClassifier(
                    PixelClassifierSettings {
                        model_path: model,
                        segmentation_mapping: vec![SegmentationMappingSettings {
                            model_class: SegmentationClass(6),
                            segmentation_class: SegmentationClass(1),
                        }],
                        min_certainty: 0.0,
                    },
                )),
            }],
        }
    }

    /// With the prediction overlay on, a preview brings each tile's
    /// prediction of the model's step; a preview without one (the step
    /// gone) clears it, and so does switching the overlay off.
    #[test]
    fn a_preview_brings_the_prediction_while_the_overlay_is_on() {
        use crate::editor::prediction_overlay::PredictionOverlaySettings;
        let f = fixture();
        let pipeline = pixel_classifier_pipeline(&f);
        let overlay = &f.worker.viewport_controller.prediction;
        let task = || {
            let mut task = f.task(true);
            task.project_settings.pipelines.push(pipeline.clone());
            task
        };

        f.worker.run_task(task());
        drain_ui_queue();
        assert!(!overlay.has_tiles(), "the overlay is off");

        overlay.set_settings(PredictionOverlaySettings {
            on: true,
            model: Some(PathBuf::from("models/cells.evamodel")),
            ..Default::default()
        });
        f.worker.run_task(task());
        drain_ui_queue();
        assert!(overlay.has_tiles());
        let (class, percent) = overlay.value_at(5.0, 5.0).expect("a prediction at (5, 5)");
        assert!(class == "Background" || class == "Cell", "{class}");
        assert!((50..=100).contains(&percent), "{percent}");
        // The legend: the model's classes, Cell in its project class's colour.
        use slint::Model;
        let ai = f.ui.global::<crate::AiLearningState>();
        let legend: Vec<String> = ai
            .get_prediction_classes()
            .iter()
            .map(|class| class.name.to_string())
            .collect();
        assert_eq!(legend, ["Background", "Cell"]);
        assert_eq!(ai.get_prediction_hint(), "");

        let mut without = f.task(true);
        without.project_settings.pipelines.clear();
        f.worker.run_task(without);
        drain_ui_queue();
        assert!(!overlay.has_tiles(), "no pixel classifier any more");
        assert_eq!(
            ai.get_prediction_classes().row_count(),
            0,
            "the legend goes with it"
        );
        assert!(
            ai.get_prediction_hint()
                .contains("No pipeline applies the model 'cells'"),
            "{}",
            ai.get_prediction_hint()
        );

        f.worker.run_task(task());
        drain_ui_queue();
        assert!(overlay.has_tiles());
        overlay.set_settings(PredictionOverlaySettings::default());
        assert!(!overlay.has_tiles(), "switched off");
    }

    #[test]
    fn a_preview_run_hides_its_pill_when_done_and_leaves_an_open_dialog() {
        let f = fixture();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::StepSettings);
        f.ui.global::<PreviewRunState>().set_shown(true);
        f.worker.run_task(f.task(true));
        drain_ui_queue();
        let preview = f.ui.global::<PreviewRunState>();
        assert!(preview.get_done());
        assert!(!preview.get_has_error());
        assert!(!preview.get_shown());
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::StepSettings
        );
    }

    /// A preview while an analysis runs (editing allowed during it): it
    /// reports to its own state and leaves the analysis' progress alone.
    #[test]
    fn a_preview_leaves_the_progress_of_a_running_analysis_alone() {
        let f = fixture();
        let running = f.running();
        running.set_active(true);
        running.set_processed(3);
        running.set_total(9);
        f.worker.run_task(f.task(true));
        drain_ui_queue();

        assert!(f.ui.global::<PreviewRunState>().get_done());
        assert!(running.get_active());
        assert!(!running.get_done());
        assert_eq!((running.get_processed(), running.get_total()), (3, 9));
        assert_eq!(running.get_status_message(), "");
    }

    /// A finished preview closes only its own overlay - not a dialog the
    /// user opened while it rendered.
    #[test]
    fn a_finished_preview_leaves_another_open_dialog_open() {
        let f = fixture();
        f.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::ProjectSettings);
        f.worker.run_task(f.task(true));
        drain_ui_queue();
        assert_eq!(
            f.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::ProjectSettings
        );
    }

    /// An analysis gets a thread of its own - the worker is free for
    /// previews meanwhile - and counts as running from the start on.
    #[test]
    fn an_analysis_runs_on_its_own_thread_and_is_running_until_it_ends() {
        let f = fixture();
        let analysis_running = &f.worker.pipeline_controller.analysis_running;

        let thread = f.worker.start_task(f.task(false)).expect("its own thread");
        assert!(analysis_running.load(std::sync::atomic::Ordering::SeqCst));
        thread.join().unwrap();
        drain_ui_queue();

        assert!(!analysis_running.load(std::sync::atomic::Ordering::SeqCst));
        assert!(
            f.running().get_done(),
            "{}",
            f.running().get_status_message()
        );
        assert!(
            f.worker.start_task(f.task(true)).is_none(),
            "a preview runs right here"
        );
    }

    /// Pins down today's behaviour: a job that can't even start (here its
    /// results folder can't be created) is only logged - the running dialog
    /// gets no result and stays as it was. Flagged as a usability gap; if
    /// that changes, this test should assert the shown error instead.
    #[test]
    fn an_analysis_with_a_pipeline_on_a_missing_channel_is_refused_visibly() {
        let f = fixture();
        let mut task = f.task(false);
        task.project_settings.pipelines.push(
            evanalyzer_cfg::settings::pipeline_settings::PipelineSettings {
                id: evanalyzer_cfg::core_types::PipelineId(1),
                name: "Spots".into(),
                description: None,
                image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(9),
                enabled: true,
                steps: vec![],
            },
        );
        f.worker.run_task(task);
        drain_ui_queue();

        assert!(f.running().get_has_error());
        let message = f.running().get_status_message().to_string();
        assert!(
            message.contains("pipeline 'Spots' reads channel 9"),
            "{message}"
        );
        assert!(
            !f.dir.path().join("results").exists(),
            "nothing was started, so no results folder"
        );
    }

    #[test]
    fn a_job_that_cannot_start_tells_the_user_why() {
        let f = fixture();
        std::fs::write(f.dir.path().join("missing"), "a file, not a folder").unwrap();
        let mut task = f.task(false);
        task.project_path = f.dir.path().join("missing").join("deeper");
        f.worker.run_task(task);
        drain_ui_queue();
        assert!(f.running().get_done());
        assert!(f.running().get_has_error());
        let message = f.running().get_status_message().to_string();
        assert!(
            message.starts_with("The analysis could not be started:"),
            "{message}"
        );
    }
}
