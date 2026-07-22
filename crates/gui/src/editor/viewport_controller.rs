use crate::ViewportState as ViewportSlintState;
use crate::editor::viewport_task::{DrawingTask, TaskDispatch};
use crate::helper::color_generators::get_colors_from_class;
use crate::{AppWindow, HistogramData, HistogramState, PipelinesPanelState, UiState};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_core::ImageContainer;
use slint::{Color, ComponentHandle, VecModel};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

#[derive(Clone)]
pub struct ViewportState {
    pub(crate) viewport_width: f32,
    pub(crate) viewport_height: f32,
    pub(crate) zoom: f32,
    pub(crate) offset_x: f32,
    pub(crate) offset_y: f32,
    pub(crate) mouse_pos_x: f32,
    pub(crate) mouse_pos_y: f32,
}

#[derive(Clone)]
pub struct ViewportOverlayState {
    pub(crate) roi_transparency: f32,
}

/// Raw breakpoint image data retained for re-rendering when histogram settings change.
pub struct BreakpointChannelData {
    pub image: Arc<ImageContainer>,
    pub tile_offset_x: usize,
    pub tile_offset_y: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    /// Original image bit depth — forwarded to `ReadContext.bit_depth` so the
    /// pixel-value HUD scales values correctly (e.g. ×65535 for 16-bit).
    pub nr_bits: u8,
    /// The channel the breakpointed pipeline actually started from, so
    /// rendering can look up *that* channel's histogram/LUT settings
    /// instead of an unrelated one. `None` when the pipeline didn't start
    /// from a plain channel address; falls back to channel 0 for rendering.
    pub channel_idx: Option<i32>,
}

pub struct DrawingTaskContainer {
    pub(crate) task_count: Arc<AtomicU32>,
    pub(crate) task_request: Arc<(Mutex<Option<DrawingTask>>, Condvar)>,
}

pub struct Tasks {
    pub(crate) low_res_task: Arc<DrawingTaskContainer>,
    pub(crate) high_res_task: Arc<DrawingTaskContainer>,
    pub(crate) roi_task: Arc<DrawingTaskContainer>,
}

pub struct ViewportController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_state: Arc<RwLock<ViewportState>>,
    pub(crate) overlay_state: Arc<RwLock<ViewportOverlayState>>,
    pub(crate) drawing_tasks: Tasks,
    /// Raw breakpoint image stored for re-rendering with live histogram settings.
    pub(crate) breakpoint_channel: Arc<RwLock<Option<BreakpointChannelData>>>,
    /// When `true` the HighRes viewport worker renders `breakpoint_channel`
    /// instead of loading from disk.
    pub(crate) show_breakpoint: Arc<AtomicBool>,
    pub(crate) high_res_posted_count: Arc<AtomicU64>,
    pub(crate) high_res_last_count_at_false: Arc<AtomicU64>,
    pub(crate) high_res_is_ready: AtomicBool,
}

/// Posts `t` into the worker's single-slot task queue, merging it with
/// whatever's already there instead of overwriting it.
///
/// The slot holds at most one pending task (see `wait_for_task`'s
/// `task_slot.take()` in `viewport_worker.rs`), so a second dispatch that
/// lands before the worker consumes the first would otherwise silently drop
/// its flags - e.g. a debounced pan/zoom redraw (a plain default task)
/// landing right after opening a new image would erase that task's
/// `is_new_image`/`auto_adjust_if_not_set` flags, skipping histogram
/// computation and rendering the image black.
fn merge_into_slot(pair: &(Mutex<Option<DrawingTask>>, Condvar), t: DrawingTask) {
    let (lock, cvar) = pair;
    let mut slot = lock.lock().unwrap();
    *slot = Some(match slot.take() {
        Some(pending) => pending.merge(t),
        None => t,
    });
    cvar.notify_one();
}

impl ViewportController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        let drawing_tasks = Tasks {
            low_res_task: Arc::new(DrawingTaskContainer {
                task_count: Arc::new(AtomicU32::new(0)),
                task_request: Arc::new((Mutex::new(None), Condvar::new())),
            }),
            high_res_task: Arc::new(DrawingTaskContainer {
                task_count: Arc::new(AtomicU32::new(0)),
                task_request: Arc::new((Mutex::new(None), Condvar::new())),
            }),
            roi_task: Arc::new(DrawingTaskContainer {
                task_count: Arc::new(AtomicU32::new(0)),
                task_request: Arc::new((Mutex::new(None), Condvar::new())),
            }),
        };

        Self {
            ui,
            app_state,
            viewport_state: Arc::new(RwLock::new(ViewportState {
                viewport_width: 0.0,
                viewport_height: 0.0,
                zoom: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
                mouse_pos_x: 0.0,
                mouse_pos_y: 0.0,
            })),
            overlay_state: Arc::new(RwLock::new(ViewportOverlayState {
                roi_transparency: 0.8,
            })),
            drawing_tasks,
            breakpoint_channel: Arc::new(RwLock::new(None)),
            show_breakpoint: Arc::new(AtomicBool::new(false)),
            high_res_posted_count: Arc::new(AtomicU64::new(0)),
            high_res_last_count_at_false: Arc::new(AtomicU64::new(0)),
            high_res_is_ready: AtomicBool::new(false),
        }
    }

    pub fn trigger_new_image_redraw(&self) {
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = true;
        task.auto_adjust_selected = false;
        task.is_new_image = true;
        task.fit_to_screen = true;
        task.is_new_series = true;
        self.dispatch_worker_task(task.clone(), TaskDispatch::HighResAndLowRes);
        self.dispatch_worker_task(task, TaskDispatch::Rois);
    }

    pub fn trigger_new_series_redraw(&self) {
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = true;
        task.auto_adjust_selected = false;
        task.is_new_image = false;
        task.fit_to_screen = true;
        task.is_new_series = true;
        self.dispatch_worker_task(task.clone(), TaskDispatch::HighResAndLowRes);
        self.dispatch_worker_task(task, TaskDispatch::Rois);
    }

    pub fn trigger_image_redraw(&self) {
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = false;
        task.auto_adjust_selected = false;
        task.is_new_image = false;
        task.fit_to_screen = false;
        task.is_new_series = false;
        self.dispatch_worker_task(task, TaskDispatch::HighRes);
    }

    pub fn trigger_image_redraw_with_auto_adjust(&self) {
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = false;
        task.auto_adjust_selected = true;
        task.is_new_image = false;
        task.fit_to_screen = false;
        task.is_new_series = false;
        self.dispatch_worker_task(task, TaskDispatch::HighRes);
    }

    pub fn trigger_image_redraw_rois(&self) {
        let task: DrawingTask = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::Rois);
    }

    pub fn trigger_redraw_low_res(&self) {
        //  RESET THE BARRIER HERE: Allow 'false' states to pass through again
        self.high_res_is_ready.store(false, Ordering::SeqCst);

        // Reset your counts if necessary
        self.high_res_posted_count.store(0, Ordering::SeqCst);
        self.high_res_last_count_at_false.store(0, Ordering::SeqCst);

        // Hide the ROI overlay immediately: it is rendered in screen-space so it
        // detaches visually from image objects the moment the viewport pans or zooms.
        // The debounce-triggered trigger_image_redraw_rois() will re-enable it once
        // the overlay has been re-composited at the new viewport position.
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ViewportSlintState>().set_roi_ready(false);
            }
        })
        .ok();
        let task = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::LowRes);
    }

    pub fn trigger_redraw_low_res_and_high_res(&self) {
        //  RESET THE BARRIER HERE: Allow 'false' states to pass through again
        self.high_res_is_ready.store(false, Ordering::SeqCst);

        // Reset your counts if necessary
        self.high_res_posted_count.store(0, Ordering::SeqCst);
        self.high_res_last_count_at_false.store(0, Ordering::SeqCst);

        let task = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::HighResAndLowRes);
    }

    /// Dispatches a drawing task to the background worker threads based on the specified scope.
    ///
    /// This method manages the distribution of rendering work to either the low-resolution
    /// preview pipeline, the high-resolution production pipeline, or both. It uses a
    /// condition variable pattern to wake up waiting worker threads after updating
    /// the atomic task counters.
    ///
    /// ### Arguments
    /// * `task` - The `DrawingTask` containing the parameters and data required for the render.
    /// * `scope` - A `TaskDispatch` enum determining which worker tiers (LowRes, HighRes, or Both)
    ///   should receive the task.
    ///
    /// ### Implementation Details
    /// The function uses an internal helper closure `notify` to:
    /// 1. Acquire the mutex lock on a task slot.
    /// 2. Inject the new task into the slot.
    /// 3. Signal the `Condvar` to wake up a blocked worker thread.
    fn dispatch_worker_task(&self, task: DrawingTask, scope: TaskDispatch) {
        if scope == TaskDispatch::LowRes || scope == TaskDispatch::HighResAndLowRes {
            self.drawing_tasks
                .low_res_task
                .task_count
                .fetch_add(1, Ordering::SeqCst);
            merge_into_slot(&self.drawing_tasks.low_res_task.task_request, task.clone());
        }

        if scope == TaskDispatch::HighRes || scope == TaskDispatch::HighResAndLowRes {
            self.drawing_tasks
                .high_res_task
                .task_count
                .fetch_add(1, Ordering::SeqCst);
            merge_into_slot(&self.drawing_tasks.high_res_task.task_request, task.clone());
        }

        if scope == TaskDispatch::Rois {
            merge_into_slot(&self.drawing_tasks.roi_task.task_request, task.clone());
        }
    }

    /// 1) Updates the Slint UI layer with the current viewport state and processed frame data.
    ///
    /// This function acts as the bridge between the internal processing pipeline and the
    /// UI thread, synchronizing the prepared image and positions.
    /// This is the first function to call for a full sync.
    ///
    /// ### Arguments
    /// * `pixel_buffer` - The raw RGB8 image data to be rendered in the UI.
    /// * `svg_histogram_data` - A collection of pre-calculated histogram points for SVG rendering.
    /// * `display_x` / `display_y` - The top-left offset of the current view within the global coordinate system.
    /// * `zoomed_w` / `zoomed_h` - The dimensions of the current zoomed viewport area.
    /// * `is_low_res` - A flag indicating if the provided buffer is a preview (proxy) or a full-resolution render.
    ///
    /// ### Returns
    /// * `Ok(())` if the state was successfully pushed to the UI components.
    /// * `Err(InternalErrors)` if the synchronization failed due to a locked resource or internal state error.
    pub fn sync_viewport_state_to_slint(
        &self,
        pixel_buffer: slint::SharedPixelBuffer<slint::Rgb8Pixel>,
        svg_histogram_data: Vec<HistogramData>,
        draw_x: f32,
        draw_y: f32,
        zoomed_w: f32,
        zoomed_h: f32,
        is_low_res: bool,
    ) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let view_state = ui.global::<ViewportSlintState>();
                if is_low_res {
                    view_state.set_display_x_ghost(draw_x);
                    view_state.set_display_y_ghost(draw_y);

                    view_state.set_tile_width_ghost(zoomed_w);
                    view_state.set_tile_height_ghost(zoomed_h);

                    ui.set_ghost_image(slint::Image::from_rgb8(pixel_buffer));
                } else {
                    view_state.set_display_x(draw_x);
                    view_state.set_display_y(draw_y);

                    view_state.set_tile_width(zoomed_w);
                    view_state.set_tile_height(zoomed_h);

                    let hist_state = ui.global::<HistogramState>();

                    ui.set_display_image(slint::Image::from_rgb8(pixel_buffer));

                    // Update the Histogram Visual
                    let model = Rc::new(VecModel::from(svg_histogram_data));
                    hist_state.set_histogram_svg_path(model.into());
                }
            }
        })
        .ok();

        // Only run the state update logic through the atomic guard wrapper.
        // If this is a high-res complete frame, we must pass it through our barrier engine
        // so that `self.high_res_is_ready` is stored as true inside our Rust runtime state.
        if !is_low_res {
            self.sync_high_res_ready_to_slint(true);
        }
    }

    /// Stores the raw breakpoint `ImageContainer` for rendering via the normal
    /// viewport worker path (histogram sliders apply just like any other channel).
    ///
    /// If the breakpoint toggle is already active a high-res redraw is triggered
    /// immediately so the new image appears without requiring a pan or zoom.
    #[allow(clippy::too_many_arguments)]
    pub fn set_breakpoint_channel(
        &self,
        image: ImageContainer,
        tile_offset_x: usize,
        tile_offset_y: usize,
        tile_width: usize,
        tile_height: usize,
        nr_bits: u8,
        channel_idx: Option<i32>,
    ) {
        {
            let mut ch = self.breakpoint_channel.write().unwrap();
            *ch = Some(BreakpointChannelData {
                image: Arc::new(image),
                tile_offset_x,
                tile_offset_y,
                tile_width,
                tile_height,
                nr_bits,
                channel_idx,
            });
        }
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<PipelinesPanelState>()
                    .set_has_breakpoint_image(true);
            }
        })
        .ok();

        // If the breakpoint view is already active, re-render immediately so
        // the updated image is visible without requiring a manual pan/zoom.
        if self.show_breakpoint.load(Ordering::Relaxed) {
            self.trigger_image_redraw();
        }
    }

    /// Switches the HighRes viewport worker between the original image and the
    /// breakpoint channel, then triggers a redraw so the change is immediate.
    pub fn set_show_breakpoint(&self, show: bool) {
        self.show_breakpoint.store(show, Ordering::Relaxed);
        self.trigger_image_redraw();
    }

    pub fn sync_high_res_ready_to_slint(&self, ready: bool) {
        let ui_weak = self.ui.clone();

        if ready {
            // 1. Increment your tracking counter safely
            self.high_res_posted_count.fetch_add(1, Ordering::SeqCst);

            // 2. Permanently lock the state to true.
            // Once this is set, no 'false' code paths below can bypass the barrier.
            self.high_res_is_ready.store(true, Ordering::SeqCst);

            // 3. Forward the true status to your Slint UI thread safely
            // Forward the false status to your Slint UI thread safely
            slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ViewportSlintState>().set_high_res_ready(true);
                    // roi_ready is managed solely by trigger_redraw_low_res (sets false)
                    // and sync_rois_to_slint_viewport (sets true). The LowRes image worker
                    // must not touch it, otherwise it races with the ROI worker and leaves
                    // roi_ready permanently false after a pan/zoom.
                }
            })
            .ok();
        } else {
            // BARRIER CHECK: If a 'true' was already set globally,
            // discard this 'false' completely. It's out-of-order or obsolete.
            if self.high_res_is_ready.load(Ordering::SeqCst) {
                return;
            }

            // If we passed the barrier, capture the snapshot safely
            let act_true = self.high_res_posted_count.load(Ordering::SeqCst);
            self.high_res_last_count_at_false
                .store(act_true, Ordering::SeqCst);

            // Forward the false status to your Slint UI thread safely
            slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ViewportSlintState>().set_high_res_ready(false);
                    // roi_ready is managed solely by trigger_redraw_low_res (sets false)
                    // and sync_rois_to_slint_viewport (sets true). The LowRes image worker
                    // must not touch it, otherwise it races with the ROI worker and leaves
                    // roi_ready permanently false after a pan/zoom.
                }
            })
            .ok();
        }
    }

    /// Updates the Slint UI layer by compositing all Regions of Interest (ROIs)
    /// into a single unified texture for viewport rendering.
    ///
    /// This function retrieves the active project's ROIs and synchronizes them
    /// with the current UI viewport state. It ensures that spatial annotations
    /// are consolidated into a consistent format suitable for Slint's rendering pipeline.
    ///
    /// ### Arguments
    /// * `&self` - Accesses the application state, specifically the project data and current viewport configuration.
    ///
    /// ### Returns
    /// * This function returns `()` on success.
    /// * Note: This function will silently return if no reference ROIs are currently
    ///   defined within the active project.
    pub fn sync_rois_to_slint_viewport(&self) {
        let project = self.app_state.get_project();

        let roi_transparency = (self
            .overlay_state
            .read()
            .expect("Failed to acquire read lock on viewport state")
            .roi_transparency
            * 255.0) as u8;

        // Guard: image must be loaded.
        let (full_img_width, full_img_height) = match project.get_selected_image_series() {
            Some(series) => (series.image_width, series.image_height),
            None => (0, 0),
        };
        if full_img_width == 0 || full_img_height == 0 {
            return;
        }

        // Read the current viewport transform.
        let (viewport_width, viewport_height, zoom, off_x, off_y) = {
            let s = self
                .viewport_state
                .read()
                .expect("Failed to acquire read lock on viewport state");
            (
                s.viewport_width,
                s.viewport_height,
                s.zoom,
                s.offset_x,
                s.offset_y,
            )
        };
        if viewport_width <= 0.0 || viewport_height <= 0.0 {
            return;
        }

        // The ROI image is positioned at (0,0) in the viewport and covers the whole
        // viewport (see viewport.slint Layer 3).  The buffer is therefore viewport-sized
        // and ROI pixels are mapped to screen coordinates directly.  This ensures the
        // overlay is always rendered at screen resolution - no matter the zoom level -
        // eliminating the blur that appeared when a fixed 1024-px buffer was upscaled.
        let buf_w = viewport_width as u32;
        let buf_h = viewport_height as u32;

        let selected_roi_id = project.get_selected_roi_id();
        let rois_option = project.get_rois();
        let Some(rois) = rois_option else {
            return;
        };

        let auto_rois = project.get_preview_rois();

        let hide_unclassified = project.hide_unclassified_rois();

        // Resolve project-level filtering/coloring decisions up front, so the
        // actual compositing pass (`composite_roi_instances`) is pure pixel math
        // with no project access - that's what makes it cheap to unit test.
        let mut instances: Vec<RoiDrawInstance> = Vec::new();
        for roi in rois.iter().chain(auto_rois.iter()) {
            if hide_unclassified && roi.object_class.is_empty() {
                continue;
            }

            // Skip ROIs whose every assigned class is hidden.
            let all_hidden = !roi.object_class.is_empty()
                && roi
                    .object_class
                    .iter()
                    .all(|c| !project.is_class_visible(c));
            if all_hidden {
                continue;
            }

            let color = if selected_roi_id.as_ref() == Some(&roi.id) {
                Color::from_argb_u8(0xfc, 0xe9, 0x03, roi_transparency)
            } else {
                get_colors_from_class(&project, roi_transparency, &roi.object_class)
            };

            instances.push(RoiDrawInstance {
                bbox: roi.bbox,
                mask: &roi.mask_data,
                color: slint::Rgba8Pixel {
                    r: color.red(),
                    g: color.green(),
                    b: color.blue(),
                    a: color.alpha(),
                },
            });
        }

        let pixels = composite_roi_instances(&instances, buf_w, buf_h, zoom, off_x, off_y);

        let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(buf_w, buf_h);
        buffer.make_mut_slice().copy_from_slice(&pixels);

        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let image = slint::Image::from_rgba8(buffer);
                ui.set_roi_image(image);
                ui.global::<ViewportSlintState>().set_roi_ready(true);
            }
        })
        .ok();
    }

    /// 2) Synchronizes the current zoom level and translation offsets to the Slint UI.
    ///
    /// This method updates the UI's internal coordinate system to ensure that the
    /// displayed image or canvas accurately reflects the user's interaction (e.g.,
    /// pinch-to-zoom or scroll-to-pan).
    ///
    /// ### Arguments
    /// * `zoom` - The magnification scale factor (where 1.0 is 100%).
    /// * `offset_x` - The horizontal translation offset from the origin.
    /// * `offset_y` - The vertical translation offset from the origin.
    ///
    /// ### Returns
    /// * `Ok(())` if the transformation parameters were successfully applied to the UI state.
    /// * `Err(InternalErrors)` if the UI properties could not be updated or the handle is invalid.
    pub fn sync_zoom_to_slint(&self, zoom: f32, offset_x: f32, offset_y: f32) {
        {
            let mut state = self
                .viewport_state
                .write()
                .expect("Failed to acquire write lock on viewport state");
            state.zoom = zoom;
            state.offset_x = offset_x;
            state.offset_y = offset_y;
        }
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let view_state = ui.global::<ViewportSlintState>();
                view_state.set_zoom_factor(zoom);
                view_state.set_offset_x(offset_x);
                view_state.set_offset_y(offset_y);
            }
        })
        .ok();

        self.sync_scale_bar_to_slint();
    }

    /// 3) Updates the navigator (minimap) state within the Slint UI.
    ///
    /// This function calculates and synchronizes the relationship between the full-sized
    /// source image and the current visible viewport. This is typically used to render
    /// a "navigation box" or thumbnail overlay that shows the user where they are
    /// zoomed in relative to the entire image.
    ///
    /// ### Arguments
    /// * `full_image_width` - The total horizontal resolution of the original source image.
    /// * `full_image_height` - The total vertical resolution of the original source image.
    /// * `viewport_width` - The width of the currently visible area in the main view.
    /// * `viewport_height` - The height of the currently visible area in the main view.
    /// * `offset_x` - The current horizontal scroll/pan position.
    /// * `offset_y` - The current vertical scroll/pan position.
    ///
    /// ### Returns
    /// * `Ok(())` if the navigator properties were successfully updated.
    /// * `Err(InternalErrors)` if the communication with the Slint component failed.
    pub fn sync_navigator_to_slint(
        &self,
        full_image_width: i64,
        full_image_height: i64,
        viewport_width: f32,
        viewport_height: f32,
        offset_x: f32,
        offset_y: f32,
    ) {
        let ui_weak = self.ui.clone();

        let zoom = self
            .viewport_state
            .read()
            .expect("Failed to acquire read lock on viewport state")
            .zoom
            .clone();

        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let full_w = full_image_width as f32;
                let full_h = full_image_height as f32;
                // Guard against division by zero: image not yet loaded or zoom not yet set
                if full_w <= 0.0 || full_h <= 0.0 || zoom <= 0.0 {
                    return;
                }

                let view_state = ui_ready.global::<ViewportSlintState>();

                let view_x_in_img = -offset_x / zoom;
                let view_y_in_img = -offset_y / zoom;
                view_state.set_nav_x(view_x_in_img / full_w);
                view_state.set_nav_y(view_y_in_img / full_h);
                view_state.set_nav_width(viewport_width / zoom / full_w);
                view_state.set_nav_height(viewport_height / zoom / full_h);
            }
        })
        .ok();
    }

    /// 4) Updates the scale bar state within the Slint UI.
    ///
    /// This function calculates and synchronizes the scale bar's visual representation
    /// based on the current zoom level and the physical size of the image.
    ///
    /// ### Returns
    /// * `Ok(())` if the scale bar properties were successfully updated.
    /// * `Err(InternalErrors)` if the communication with the Slint component failed.
    pub fn sync_scale_bar_to_slint(&self) {
        let ui_weak = self.ui.clone();
        let project = self.app_state.get_project();

        let pixel_sizes = project.get_pixel_sizes();

        // We must get UI data (filter text) on the UI thread or
        // keep a copy in Rust. Assuming we need to pull it from Slint:
        let zoom = self
            .viewport_state
            .read()
            .expect("Failed to acquire read lock on viewport state")
            .zoom
            .clone();
        let nanos_per_pixel_px = pixel_sizes.x;
        let target_screen_px = 150.0;
        // How many nanometers are currently in our 150px target?
        let nanos_at_target = (target_screen_px / zoom) * nanos_per_pixel_px;

        // Find the magnitude (power of 10)
        let exponent = nanos_at_target.log10().floor();
        let magnitude = 10.0f32.powf(exponent);

        // Find the leading digit (mantissa)
        let mantissa = nanos_at_target / magnitude;

        // Choose the step based on your 1, 2, 5 sequence
        let step_multiplier = if mantissa >= 5.0 {
            5.0
        } else if mantissa >= 2.0 {
            2.0
        } else {
            1.0
        };

        let scale_value_nanos = step_multiplier * magnitude;

        // Formatting logic (nm vs µm vs mm)
        let (display_val, unit) = if scale_value_nanos >= 1_000_000.0 {
            (scale_value_nanos / 1_000_000.0, "mm")
        } else if scale_value_nanos >= 1_000.0 {
            (scale_value_nanos / 1_000.0, "µm")
        } else {
            (scale_value_nanos, "nm")
        };

        // Convert back to screen pixels for Slint
        let final_bar_width_px = (scale_value_nanos / nanos_per_pixel_px) * zoom;

        // The final assignment goes into the event loop
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let view_state = ui_ready.global::<ViewportSlintState>();
                view_state.set_scale_bar_width(final_bar_width_px);
                view_state.set_scale_bar_text(format!("{} {}", display_val, unit).into());
            }
        })
        .ok();
    }
}

/// One ROI's pre-resolved render inputs. All project-level filtering and color
/// resolution (selection, class visibility, hide-unclassified) happens before this
/// is built, so [`composite_roi_instances`] is pure pixel math with no project
/// access - that's what makes it cheap to unit test independently of the Slint/UI
/// plumbing in [`ViewportController::sync_rois_to_slint_viewport`].
struct RoiDrawInstance<'a> {
    bbox: [u32; 4],
    mask: &'a bitvec::vec::BitVec<u64, bitvec::order::Lsb0>,
    color: slint::Rgba8Pixel,
}

/// Porter-Duff "over": blends `src` onto `dst` in place.
fn blend_pixel_over(dst: &mut slint::Rgba8Pixel, src: slint::Rgba8Pixel) {
    if dst.a == 0 {
        *dst = src;
        return;
    }
    let sa = src.a as f32 / 255.0;
    let da = dst.a as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    *dst = slint::Rgba8Pixel {
        r: ((src.r as f32 * sa + dst.r as f32 * da * (1.0 - sa)) / out_a) as u8,
        g: ((src.g as f32 * sa + dst.g as f32 * da * (1.0 - sa)) / out_a) as u8,
        b: ((src.b as f32 * sa + dst.b as f32 * da * (1.0 - sa)) / out_a) as u8,
        a: (out_a * 255.0) as u8,
    };
}

/// Composites pre-resolved ROI instances into a viewport-sized RGBA buffer.
///
/// Two perf shortcuts, both no-ops for ROIs whose screen footprint covers more
/// than one pixel - i.e. normal/zoomed-in viewing walks every mask pixel exactly
/// as before:
///
/// - **Viewport culling**: an instance whose bbox doesn't project onto the buffer
///   at all is skipped before touching its mask. Mask pixels are always within the
///   bbox, so an off-screen bbox guarantees every mask pixel is off-screen too -
///   this cannot change the rendered output, only skip work that would have been a
///   no-op anyway.
/// - **Sub-pixel collapse**: when the whole bbox's screen footprint fits within a
///   single pixel cell (e.g. viewing a whole-slide image zoomed out far enough
///   that a cell's mask is smaller than one screen pixel), the mask shape can't be
///   distinguished on screen anyway, so one pixel is stamped directly from the
///   bbox centre instead of walking every bit.
///
/// Without these, cost scales with total mask pixels across every ROI in the
/// project; with them it scales with the number of ROIs that are actually visible
/// and bigger than a pixel - which is what makes hundreds of thousands of ROIs in
/// a whole-slide project viable to render interactively.
fn composite_roi_instances(
    instances: &[RoiDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Vec<slint::Rgba8Pixel> {
    let pixel_count = (buf_w as usize) * (buf_h as usize);
    let mut pixels = vec![
        slint::Rgba8Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        };
        pixel_count
    ];
    // Instance map: tracks which ROI instance owns each screen pixel.
    // 0 = no ROI; n+1 = the ROI at `instances[n]`.
    // Used by the border pass to detect boundaries between same-colour adjacent instances.
    let mut instance_map = vec![0u32; pixel_count];

    for (idx, inst) in instances.iter().enumerate() {
        let instance_id = (idx + 1) as u32;
        let bbox = inst.bbox;
        let bbox_w = (bbox[2] - bbox[0] + 1) as usize;
        if bbox_w == 0 {
            continue;
        }

        // Screen-space footprint of the whole bbox. `+1` because bbox[2]/[3] are
        // inclusive max coordinates (mirrors the per-pixel `abs+1` edge below).
        let sx0 = bbox[0] as f32 * zoom + off_x;
        let sy0 = bbox[1] as f32 * zoom + off_y;
        let sx1 = (bbox[2] + 1) as f32 * zoom + off_x;
        let sy1 = (bbox[3] + 1) as f32 * zoom + off_y;

        // Viewport culling: bbox doesn't overlap the buffer at all.
        if sx1 <= 0.0 || sy1 <= 0.0 || sx0 >= buf_w as f32 || sy0 >= buf_h as f32 {
            continue;
        }

        // Sub-pixel collapse: the whole bbox fits within one pixel cell on screen.
        if sx1 - sx0 <= 1.0 && sy1 - sy0 <= 1.0 {
            let px = (((sx0 + sx1) * 0.5).max(0.0) as usize).min(buf_w as usize - 1);
            let py = (((sy0 + sy1) * 0.5).max(0.0) as usize).min(buf_h as usize - 1);
            let i = py * buf_w as usize + px;
            blend_pixel_over(&mut pixels[i], inst.color);
            instance_map[i] = instance_id;
            continue;
        }

        for bit_idx in inst.mask.iter_ones() {
            let local_x = (bit_idx % bbox_w) as f32;
            let local_y = (bit_idx / bbox_w) as f32;

            let abs_x = bbox[0] as f32 + local_x;
            let abs_y = bbox[1] as f32 + local_y;

            // Derive the screen rect from the image-pixel edges so adjacent pixels
            // are always exactly adjacent on screen - no gaps and no overlap.
            // Using (abs+1)*zoom for the far edge instead of abs*zoom+ceil(zoom)
            // is what eliminates the grid: ceil(zoom) > zoom for fractional zoom,
            // so the old approach created overlapping regions that composited twice.
            let x0 = ((abs_x * zoom + off_x).max(0.0) as usize).min(buf_w as usize);
            let y0 = ((abs_y * zoom + off_y).max(0.0) as usize).min(buf_h as usize);
            let x1 = (((abs_x + 1.0) * zoom + off_x).max(0.0) as usize).min(buf_w as usize);
            let y1 = (((abs_y + 1.0) * zoom + off_y).max(0.0) as usize).min(buf_h as usize);

            if x0 >= x1 || y0 >= y1 {
                continue; // entirely off-screen or zoomed-out pixel
            }

            for py in y0..y1 {
                for px in x0..x1 {
                    let i = py * buf_w as usize + px;
                    blend_pixel_over(&mut pixels[i], inst.color);
                    instance_map[i] = instance_id;
                }
            }
        }
    }

    // Border pass: make the outline of every ROI instance fully opaque.
    // A pixel is a border pixel if any 4-connected neighbour belongs to a
    // different instance (including the background, which has instance_id 0).
    // This correctly separates same-colour adjacent instances that rgb comparison
    // cannot distinguish.
    let bw = buf_w as usize;
    let bh = buf_h as usize;
    for by in 0..bh {
        for bx in 0..bw {
            let i = by * bw + bx;
            let inst = instance_map[i];
            if inst == 0 {
                continue;
            }
            let is_border = bx == 0
                || bx + 1 >= bw
                || by == 0
                || by + 1 >= bh
                || instance_map[i - 1] != inst
                || instance_map[i + 1] != inst
                || instance_map[i - bw] != inst
                || instance_map[i + bw] != inst;
            if is_border {
                pixels[i].a = 255;
            }
        }
    }

    pixels
}

#[cfg(test)]
mod composite_roi_instances_tests {
    use super::*;
    use bitvec::prelude::*;

    fn rgba(r: u8, g: u8, b: u8, a: u8) -> slint::Rgba8Pixel {
        slint::Rgba8Pixel { r, g, b, a }
    }

    fn full_mask(bbox: [u32; 4]) -> BitVec<u64, Lsb0> {
        let w = (bbox[2] - bbox[0] + 1) as usize;
        let h = (bbox[3] - bbox[1] + 1) as usize;
        bitvec![u64, Lsb0; 1; w * h]
    }

    #[test]
    fn single_pixel_roi_renders_at_correct_location_with_border_alpha() {
        // 3x3 bbox, only the centre bit set -> one isolated screen pixel.
        let bbox = [0u32, 0, 2, 2];
        let mut mask = bitvec![u64, Lsb0; 0; 9];
        mask.set(4, true); // local (1,1) in a 3-wide bbox
        let color = rgba(10, 20, 30, 128);
        let instances = [RoiDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_roi_instances(&instances, 5, 5, 1.0, 0.0, 0.0);

        for y in 0..5usize {
            for x in 0..5usize {
                let i = y * 5 + x;
                if x == 1 && y == 1 {
                    // Isolated pixel: every neighbour is background, so it's a
                    // border pixel and alpha is forced to 255.
                    assert_eq!(pixels[i], rgba(10, 20, 30, 255), "at ({x},{y})");
                } else {
                    assert_eq!(pixels[i].a, 0, "expected background at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn zoomed_in_block_has_no_gaps_and_distinguishes_border_from_interior_alpha() {
        // 2x2 fully-set mask, zoom=2 -> a gapless 4x4 screen block at (2,2)-(6,6)
        // inside an 8x8 buffer, with a 2x2 interior that isn't a border.
        let bbox = [1u32, 1, 2, 2];
        let mask = full_mask(bbox);
        let color = rgba(200, 0, 0, 128);
        let instances = [RoiDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_roi_instances(&instances, 8, 8, 2.0, 0.0, 0.0);

        for y in 0..8usize {
            for x in 0..8usize {
                let i = y * 8 + x;
                let in_block = (2..6).contains(&x) && (2..6).contains(&y);
                if !in_block {
                    assert_eq!(pixels[i].a, 0, "expected background at ({x},{y})");
                    continue;
                }
                let is_interior = (3..5).contains(&x) && (3..5).contains(&y);
                assert_eq!(pixels[i].r, 200, "at ({x},{y})");
                if is_interior {
                    // Not a border pixel: keeps the original blended alpha.
                    assert_eq!(pixels[i].a, 128, "interior alpha at ({x},{y})");
                } else {
                    assert_eq!(pixels[i].a, 255, "border alpha at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn roi_entirely_outside_viewport_is_culled_and_produces_no_pixels() {
        let bbox = [1000u32, 1000, 1001, 1001];
        let mask = full_mask(bbox);
        let instances = [RoiDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(255, 255, 255, 255),
        }];

        let pixels = composite_roi_instances(&instances, 8, 8, 1.0, 0.0, 0.0);

        assert!(pixels.iter().all(|p| p.a == 0));
    }

    #[test]
    fn roi_straddling_the_viewport_edge_is_not_over_culled() {
        // 3x3 fully-set mask anchored at the origin, but the buffer is only 2x2:
        // the bbox extends past the buffer, yet still overlaps it and must not be
        // culled - the visible 2x2 corner should render fully.
        let bbox = [0u32, 0, 2, 2];
        let mask = full_mask(bbox);
        let instances = [RoiDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(1, 2, 3, 200),
        }];

        let pixels = composite_roi_instances(&instances, 2, 2, 1.0, 0.0, 0.0);

        assert!(
            pixels.iter().all(|p| p.r == 1 && p.g == 2 && p.b == 3),
            "every pixel of the visible corner should be painted: {pixels:?}"
        );
    }

    #[test]
    fn far_zoomed_out_roi_collapses_to_exactly_one_pixel_instead_of_vanishing() {
        // A small (2x2) ROI viewed at zoom=0.01: its screen footprint
        // (2 * 0.01 = 0.02px) is far below one pixel. Without the collapse path
        // this would either vanish (every mask pixel rounds into the same
        // sub-pixel slot that the old per-pixel clipping logic could drop) or
        // require iterating mask bits for no visual gain; with it, exactly one
        // pixel must be painted so the ROI doesn't silently disappear.
        let bbox = [100u32, 100, 101, 101];
        let mask = full_mask(bbox);
        let color = rgba(50, 60, 70, 222);
        let instances = [RoiDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_roi_instances(&instances, 50, 50, 0.01, 0.0, 0.0);

        let painted: Vec<_> = pixels.iter().filter(|p| p.a != 0).collect();
        assert_eq!(
            painted.len(),
            1,
            "expected exactly one painted pixel, got {painted:?}"
        );
        assert_eq!(painted[0].r, 50);
        assert_eq!(painted[0].g, 60);
        assert_eq!(painted[0].b, 70);
        // Isolated single pixel -> border pass forces full opacity.
        assert_eq!(painted[0].a, 255);
    }

    #[test]
    fn overlapping_rois_blend_with_porter_duff_over() {
        let bbox = [0u32, 0, 0, 0]; // single-pixel bbox
        let mask = full_mask(bbox);
        let bottom = rgba(255, 0, 0, 128);
        let top = rgba(0, 0, 255, 128);
        let instances = [
            RoiDrawInstance {
                bbox,
                mask: &mask,
                color: bottom,
            },
            RoiDrawInstance {
                bbox,
                mask: &mask,
                color: top,
            },
        ];

        let pixels = composite_roi_instances(&instances, 3, 3, 1.0, 0.0, 0.0);

        let mut expected = bottom;
        blend_pixel_over(&mut expected, top);
        expected.a = 255; // isolated pixel -> border pass forces full opacity
        assert_eq!(pixels[0], expected);
    }
}

#[cfg(test)]
mod dispatch_slot_tests {
    use super::*;

    /// Reproduces the exact black-image race: a "new image" task is posted,
    /// then (before the worker's `wait_for_task` consumes it) a plain
    /// debounced-redraw task lands in the same slot. The slot must still
    /// carry `is_new_image`/`auto_adjust_if_not_set` when the worker finally
    /// takes it - a raw overwrite would have erased them here, skipping
    /// histogram computation and leaving the render black.
    #[test]
    fn a_pending_new_image_task_survives_a_later_plain_dispatch() {
        let pair: (Mutex<Option<DrawingTask>>, Condvar) = (Mutex::new(None), Condvar::new());

        let new_image_task = DrawingTask {
            auto_adjust_selected: false,
            auto_adjust_if_not_set: true,
            is_new_image: true,
            fit_to_screen: true,
            is_new_series: true,
        };
        merge_into_slot(&pair, new_image_task);

        // The worker hasn't run yet: a stray debounced pan/zoom redraw fires
        // and dispatches a plain task into the same slot.
        merge_into_slot(&pair, DrawingTask::default());

        // What the worker's `wait_for_task` would have taken.
        let taken = pair.0.lock().unwrap().take().expect("a task must be pending");
        assert!(
            taken.is_new_image,
            "the new-image request must not be dropped by the later plain dispatch"
        );
        assert!(taken.auto_adjust_if_not_set);
        assert!(taken.fit_to_screen);
        assert!(taken.is_new_series);
    }

    #[test]
    fn dispatch_into_an_empty_slot_is_unchanged() {
        let pair: (Mutex<Option<DrawingTask>>, Condvar) = (Mutex::new(None), Condvar::new());
        let task = DrawingTask {
            auto_adjust_selected: true,
            ..DrawingTask::default()
        };
        merge_into_slot(&pair, task);

        let taken = pair.0.lock().unwrap().take().expect("a task must be pending");
        assert!(taken.auto_adjust_selected);
        assert!(!taken.is_new_image);
    }

    #[test]
    fn a_consumed_slot_is_not_affected_by_a_stale_merge() {
        let pair: (Mutex<Option<DrawingTask>>, Condvar) = (Mutex::new(None), Condvar::new());
        merge_into_slot(
            &pair,
            DrawingTask {
                is_new_image: true,
                ..DrawingTask::default()
            },
        );
        // Worker consumes it.
        pair.0.lock().unwrap().take().expect("a task must be pending");

        // A later, unrelated plain dispatch must not resurrect the old flag.
        merge_into_slot(&pair, DrawingTask::default());
        let taken = pair.0.lock().unwrap().take().expect("a task must be pending");
        assert!(
            !taken.is_new_image,
            "a fresh dispatch into an empty (already-consumed) slot must not merge with stale state"
        );
    }
}
