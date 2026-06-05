use crate::{
    DialogType, GlobalAppState, PipelineRunningState, PipelinesPanelState, UiState,
    editor::{
        classification_controller::ClassificationController, pipeline_task::PipelineTask,
        pipelines_controller::PipelinesController, results_list_controller::ResultsListController,
        roi_list_controller::RoiListController, viewport_controller::ViewportController,
    },
};
use chrono::Utc;
use evanalyzer_cfg::{
    RESULTS_FILE_EXTENSION, core_types::InternalErrors, settings::roi_settings::RoiSettings,
};
use evanalyzer_core::{DuckDbExporter, MemoryExporter, PipelineResultExporter};
use log::{error, info};
use slint::ComponentHandle;
use std::sync::{Arc, Condvar, Mutex};

pub struct PipelineWorker {
    pub(crate) app_state: Arc<UiState>,
    pub(crate) pipeline_controller: Arc<PipelinesController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) roi_list_controller: Arc<RoiListController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
    pub(crate) results_list_controller: Arc<ResultsListController>,
}

impl PipelineWorker {
    pub fn new(
        app_state: Arc<UiState>,
        pipeline_controller: Arc<PipelinesController>,
        viewport_controller: Arc<ViewportController>,
        roi_list_controller: Arc<RoiListController>,
        classification_controller: Arc<ClassificationController>,
        results_list_controller: Arc<ResultsListController>,
    ) -> Self {
        Self {
            app_state,
            pipeline_controller,
            viewport_controller,
            roi_list_controller,
            classification_controller,
            results_list_controller,
        }
    }

    pub(crate) fn start_worker(self: &Arc<Self>) {
        let self_handle = Arc::clone(self);
        std::thread::Builder::new()
            .name("PipelineWorker".into())
            .spawn(move || self_handle.run_worker_loop())
            .expect("Failed to spawn pipeline worker thread");
    }

    fn run_worker_loop(self: &Arc<Self>) -> ! {
        let task_request = &self.pipeline_controller.task_request;
        let self_handle = Arc::clone(self);
        loop {
            let task = wait_for_task(task_request.clone());

            let out_rois: Arc<Mutex<Vec<RoiSettings>>> = Arc::new(Mutex::new(vec![]));
            let is_preview = task.preview;

            let class_names: std::collections::HashMap<_, _> = task
                .project_settings
                .classification
                .classes
                .iter()
                .filter_map(|c| {
                    c.id.to_u32().map(|n| {
                        (
                            evanalyzer_cfg::core_types::ObjectClass::Valid(n),
                            c.name.clone(),
                        )
                    })
                })
                .collect();

            let results_out: Arc<Mutex<dyn PipelineResultExporter>> = if is_preview {
                Arc::new(Mutex::new(MemoryExporter {
                    out_rois: out_rois.clone(),
                }))
            } else {
                let output_dir = task.project_path.join("results");
                info!("Creating output directory: {:?}", output_dir);
                if let Err(e) = std::fs::create_dir_all(&output_dir) {
                    error!("Failed to create output directory: {e}");
                }
                let now = Utc::now();
                let file_date = now.format("%Y%m%dT%H%M%SZ").to_string();
                let job_name =
                    petname::petname(2, "_").expect("Problem in random job name generator");
                let out_path =
                    output_dir.join(format!("{file_date}__{job_name}.{RESULTS_FILE_EXTENSION}"));
                match DuckDbExporter::new(&out_path, class_names) {
                    Ok(exp) => Arc::new(Mutex::new(exp)),
                    Err(e) => {
                        error!("Failed to open result database {}: {e}", out_path.display());
                        continue;
                    }
                }
            };

            info!("Started pipeline worker task");

            let job = evanalyzer_core::generate_job_from_project_settings(
                task.project_settings,
                task.project_path,
                results_out,
            );
            let Ok(mut job_exec) = job else {
                error!("Could not execute job!");
                continue;
            };

            // For preview runs, restrict processing to tiles that are currently
            // visible in the viewport so the user sees results immediately.
            if is_preview {
                let vp = self
                    .viewport_controller
                    .viewport_state
                    .read()
                    .expect("Failed to acquire read lock on viewport state");
                job_exec.preview_tile_settings = Some(evanalyzer_core::PreviewTileSettings {
                    offset_x: vp.offset_x,
                    offset_y: vp.offset_y,
                    viewport_width: vp.viewport_width,
                    viewport_height: vp.viewport_height,
                    zoom: vp.zoom,
                    process_all_tiles: false,
                });

                if let Some((pipeline_id, step_id, mode)) = task.breakpoint {
                    job_exec.breakpoint = Some(evanalyzer_core::BreakpointSettings {
                        pipeline_id,
                        pipeline_step_id: step_id,
                        mode,
                    });
                } else {
                    job_exec.breakpoint = None;
                }
            }

            info!("Pipeline job started ...");

            let Ok(parallel) = std::thread::available_parallelism() else {
                error!("Could not get number of cores!");
                continue;
            };
            let (handle, rx, cancel_flag) = job_exec.run_async(parallel.get() - 1);
            *self
                .pipeline_controller
                .pipeline_cancel_flag
                .lock()
                .unwrap() = Some(cancel_flag);
            let mut last_ui_update = std::time::Instant::now();
            let mut pipeline_start: Option<std::time::Instant> = None;
            for event in rx {
                match event {
                    evanalyzer_core::ProgressEvent::TilesScheduled { total_tiles } => {
                        let ui_handle = self.app_state.ui_handle.clone();
                        let total = total_tiles as i32;
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>().set_total(total);
                                ui.global::<PipelineRunningState>().set_processed(0);
                            }
                        });
                    }
                    evanalyzer_core::ProgressEvent::Started { total } => {
                        info!("Pipeline started: {total} images to process");
                        pipeline_start = Some(std::time::Instant::now());
                        // Clear any stale preview ROIs so the incremental tile updates
                        // start from a clean slate.
                        if is_preview {
                            self_handle
                                .app_state
                                .get_project_write()
                                .tmp_settings
                                .preview_rois
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
                            }
                        });
                    }
                    evanalyzer_core::ProgressEvent::TileCompleted {
                        tile_index,
                        total_tiles,
                        rois,
                    } => {
                        info!("Tile {tile_index}/{total_tiles} completed");
                        if is_preview {
                            // Append the new ROIs and redraw so the user sees partial results.
                            self_handle
                                .app_state
                                .get_project_write()
                                .tmp_settings
                                .preview_rois
                                .extend(rois);
                            self_handle.roi_list_controller.sync_rois_to_slint();
                            self_handle
                                .classification_controller
                                .sync_classification_to_slint();
                            self_handle.viewport_controller.trigger_image_redraw_rois();
                        }
                        let ui_handle = self.app_state.ui_handle.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.global::<PipelineRunningState>()
                                    .set_processed(tile_index as i32);
                                ui.global::<PipelineRunningState>()
                                    .set_total(total_tiles as i32);
                            }
                        });
                    }
                    evanalyzer_core::ProgressEvent::ImageCompleted { index, total, path } => {
                        info!(
                            "Pipeline progress: {}/{} - {}",
                            index,
                            total,
                            path.display()
                        );
                        let secs_per_image = pipeline_start
                            .map(|t| t.elapsed().as_secs_f64() / index as f64)
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
                    evanalyzer_core::ProgressEvent::BreakpointReached {
                        image,
                        tile_offset_x,
                        tile_offset_y,
                        tile_width,
                        tile_height,
                        nr_bits,
                    } => {
                        info!(
                            "Breakpoint image received for tile ({},{}) {}x{}",
                            tile_offset_x, tile_offset_y, tile_width, tile_height
                        );
                        // Store the raw ImageContainer so the viewport worker can
                        // re-render it with live histogram/LUT settings.
                        self_handle.viewport_controller.set_breakpoint_channel(
                            image,
                            tile_offset_x,
                            tile_offset_y,
                            tile_width,
                            tile_height,
                            nr_bits,
                        );
                    }
                    evanalyzer_core::ProgressEvent::ImageFailed { path } => {
                        error!("Pipeline image failed: {}", path.display());
                    }
                    evanalyzer_core::ProgressEvent::Finished => {
                        info!("Pipeline job finished - waiting for result");
                        *self
                            .pipeline_controller
                            .pipeline_cancel_flag
                            .lock()
                            .unwrap() = None;
                    }
                }
            }
            let (status_message, is_error) = match handle.join().expect("pipeline thread panicked")
            {
                Err(InternalErrors::Cancelled) => {
                    info!("Pipeline cancelled by user");
                    ("Cancelled by user.".to_string(), false)
                }
                Err(e) => {
                    error!("Pipeline job error: {e:?}");
                    (format!("Error: {e}"), true)
                }
                Ok(()) => {
                    info!("Pipeline completed successfully");
                    if is_preview {
                        // All tiles have already been streamed in via TileCompleted; just
                        // do a final sync to make sure the UI is consistent.
                        self_handle.roi_list_controller.sync_rois_to_slint();
                        self_handle
                            .classification_controller
                            .sync_classification_to_slint();
                        self_handle.viewport_controller.trigger_image_redraw_rois();
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

/// Waits for a pipeline task to become available, blocking until one is posted.
fn wait_for_task(task_request: Arc<(Mutex<Option<PipelineTask>>, Condvar)>) -> PipelineTask {
    let (lock, cvar) = &*task_request;
    let mut task_slot = lock.lock().unwrap();
    while task_slot.is_none() {
        task_slot = cvar.wait(task_slot).unwrap();
    }
    task_slot.take().unwrap()
}
