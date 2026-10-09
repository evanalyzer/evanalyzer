use crate::ViewportState as ViewportSlintState;
use crate::editor::viewport_task::{DrawingTask, TaskDispatch};
use crate::helper::color_generators::get_colors_from_class;
use crate::{AppWindow, HistogramData, HistogramState, PipelinesPanelState, UiState};
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::ObjectClass;
use slint::{Color, ComponentHandle, VecModel};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// Which buffer the breakpoint preview renders. Mirrors the `int` values
/// used on the Slint side (`PipelinesPanelState.breakpoint-view-mode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BreakpointViewMode {
    #[default]
    Image = 0,
    Segmentation = 1,
    Instances = 2,
}

impl BreakpointViewMode {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Segmentation,
            2 => Self::Instances,
            _ => Self::Image,
        }
    }
}

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
    pub(crate) object_transparency: f32,
}

/// Raw breakpoint image data retained for re-rendering when histogram settings change.
pub struct BreakpointChannelData {
    pub image: Arc<ImageContainer>,
    /// Segmentation/instance label maps captured at the same breakpoint
    /// step, if the pipeline had produced them by then. `None` before
    /// `Threshold`/`ConnectedComponents`/`Watershed` respectively - the
    /// render loop falls back to `image` when the buffer the current
    /// `BreakpointViewMode` wants isn't available yet.
    pub segmentation: Option<Arc<ImageContainer>>,
    pub instances: Option<Arc<ImageContainer>>,
    pub tile_offset_x: usize,
    pub tile_offset_y: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    /// Original image bit depth — forwarded to `ReadContext.bit_depth` so the
    /// pixel-value HUD scales values correctly (e.g. ×65535 for 16-bit).
    pub nr_bits: u16,
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
    pub(crate) object_task: Arc<DrawingTaskContainer>,
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
    /// Which of `breakpoint_channel`'s buffers to render (image/segmentation/
    /// instances). Stored as the raw `BreakpointViewMode` discriminant.
    pub(crate) breakpoint_view_mode: Arc<AtomicU8>,
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
            object_task: Arc::new(DrawingTaskContainer {
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
                object_transparency: 0.8,
            })),
            drawing_tasks,
            breakpoint_channel: Arc::new(RwLock::new(None)),
            show_breakpoint: Arc::new(AtomicBool::new(false)),
            breakpoint_view_mode: Arc::new(AtomicU8::new(BreakpointViewMode::default() as u8)),
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
        self.dispatch_worker_task(task, TaskDispatch::Objects);
    }

    pub fn trigger_new_series_redraw(&self) {
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = true;
        task.auto_adjust_selected = false;
        task.is_new_image = false;
        task.fit_to_screen = true;
        task.is_new_series = true;
        self.dispatch_worker_task(task.clone(), TaskDispatch::HighResAndLowRes);
        self.dispatch_worker_task(task, TaskDispatch::Objects);
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

    pub fn trigger_image_redraw_objects(&self) {
        let task: DrawingTask = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::Objects);
    }

    pub fn trigger_redraw_low_res(&self) {
        //  RESET THE BARRIER HERE: Allow 'false' states to pass through again
        self.high_res_is_ready.store(false, Ordering::SeqCst);

        // Reset your counts if necessary
        self.high_res_posted_count.store(0, Ordering::SeqCst);
        self.high_res_last_count_at_false.store(0, Ordering::SeqCst);

        // Hide the object overlay immediately: it is rendered in screen-space so it
        // detaches visually from image objects the moment the viewport pans or zooms.
        // The debounce-triggered trigger_image_redraw_objects() will re-enable it once
        // the overlay has been re-composited at the new viewport position.
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ViewportSlintState>().set_object_ready(false);
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

        if scope == TaskDispatch::Objects {
            merge_into_slot(&self.drawing_tasks.object_task.task_request, task.clone());
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        segmentation: Option<ImageContainer>,
        instances: Option<ImageContainer>,
        tile_offset_x: usize,
        tile_offset_y: usize,
        tile_width: usize,
        tile_height: usize,
        nr_bits: u16,
        channel_idx: Option<i32>,
    ) {
        {
            let mut ch = self.breakpoint_channel.write().unwrap();
            *ch = Some(BreakpointChannelData {
                image: Arc::new(image),
                segmentation: segmentation.map(Arc::new),
                instances: instances.map(Arc::new),
                tile_offset_x,
                tile_offset_y,
                tile_width,
                tile_height,
                nr_bits,
                channel_idx,
            });
        }
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    /// Reads the currently selected breakpoint view mode (image/segmentation/
    /// instances).
    pub fn breakpoint_view_mode(&self) -> BreakpointViewMode {
        BreakpointViewMode::from_i32(self.breakpoint_view_mode.load(Ordering::Relaxed) as i32)
    }

    /// Sets which buffer the breakpoint preview renders, then triggers a
    /// redraw if the breakpoint view is currently active.
    pub fn set_breakpoint_view_mode(&self, mode: i32) {
        self.breakpoint_view_mode
            .store(BreakpointViewMode::from_i32(mode) as u8, Ordering::Relaxed);
        if self.show_breakpoint.load(Ordering::Relaxed) {
            self.trigger_image_redraw();
        }
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
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ViewportSlintState>().set_high_res_ready(true);
                    // object_ready is managed solely by trigger_redraw_low_res (sets false)
                    // and sync_objects_to_slint_viewport (sets true). The LowRes image worker
                    // must not touch it, otherwise it races with the object worker and leaves
                    // object_ready permanently false after a pan/zoom.
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
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ViewportSlintState>().set_high_res_ready(false);
                    // object_ready is managed solely by trigger_redraw_low_res (sets false)
                    // and sync_objects_to_slint_viewport (sets true). The LowRes image worker
                    // must not touch it, otherwise it races with the object worker and leaves
                    // object_ready permanently false after a pan/zoom.
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
    pub fn sync_objects_to_slint_viewport(&self) {
        let project = self.app_state.get_project();

        let object_transparency = (self
            .overlay_state
            .read()
            .expect("Failed to acquire read lock on viewport state")
            .object_transparency
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

        // The object image is positioned at (0,0) in the viewport and covers the whole
        // viewport (see viewport.slint Layer 3). The buffer is therefore viewport-sized
        // and object pixels are mapped to screen coordinates directly, so the overlay
        // is always rendered at screen resolution regardless of zoom.
        let buf_w = viewport_width as u32;
        let buf_h = viewport_height as u32;

        let selected_object_id = project.get_selected_object_id();
        let Some(objects) = project.get_objects() else {
            return;
        };
        let auto_objects = project.get_preview_objects();
        let hide_unclassified = project.hide_unclassified_objects();

        // Draw order: selected object > selected class > class list order
        // (first class in the list on top) > unclassified. Resolved once here.
        let z = ZOrder::new(&*project);

        // Resolve project-level filtering/coloring/ordering up front, so the
        // compositing pass (`composite_object_instances`) is pure pixel math
        // with no project access.
        let mut keyed: Vec<(u32, ObjectDrawInstance)> =
            Vec::with_capacity(objects.len() + auto_objects.len());

        for object in objects.iter().chain(auto_objects.iter()) {
            if hide_unclassified && object.object_class.is_empty() {
                continue;
            }

            // Skip ROIs whose every assigned class is hidden.
            let all_hidden = !object.object_class.is_empty()
                && object
                    .object_class
                    .iter()
                    .all(|c| !project.is_class_visible(c));
            if all_hidden {
                continue;
            }

            let is_selected = selected_object_id.as_ref() == Some(&object.id);

            let color = if is_selected {
                Color::from_argb_u8(0xfc, 0xe9, 0x03, object_transparency)
            } else {
                get_colors_from_class(&project, object_transparency, &object.object_class)
            };

            keyed.push((
                z.key(&object.object_class, is_selected),
                ObjectDrawInstance {
                    bbox: object.bbox,
                    mask: &object.mask_data,
                    color: slint::Rgba8Pixel {
                        r: color.red(),
                        g: color.green(),
                        b: color.blue(),
                        a: color.alpha(),
                    },
                },
            ));
        }

        // Stable sort: objects with equal key keep their original vector order.
        keyed.sort_by_key(|(k, _)| *k);
        let instances: Vec<ObjectDrawInstance> = keyed.into_iter().map(|(_, i)| i).collect();

        let pixels = composite_object_instances(&instances, buf_w, buf_h, zoom, off_x, off_y);

        let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(buf_w, buf_h);
        buffer.make_mut_slice().copy_from_slice(&pixels);

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let image = slint::Image::from_rgba8(buffer);
                ui.set_object_image(image);
                ui.global::<ViewportSlintState>().set_object_ready(true);
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let view_state = ui_ready.global::<ViewportSlintState>();
                view_state.set_scale_bar_width(final_bar_width_px);
                view_state.set_scale_bar_text(format!("{} {}", display_val, unit).into());
            }
        })
        .ok();
    }
}

/// One object's pre-resolved render inputs. All project-level filtering and color
/// resolution (selection, class visibility, hide-unclassified) happens before this
/// is built, so [`composite_object_instances`] is pure pixel math with no project
/// access - that's what makes it cheap to unit test independently of the Slint/UI
/// plumbing in [`ViewportController::sync_objects_to_slint_viewport`].
struct ObjectDrawInstance<'a> {
    bbox: [u32; 4],
    mask: &'a bitvec::vec::BitVec<u64, bitvec::order::Lsb0>,
    color: slint::Rgba8Pixel,
}
/// Porter-Duff "over": blends `src` onto `dst` in place.
fn blend_pixel_over(dst: &mut slint::Rgba8Pixel, src: slint::Rgba8Pixel) {
    if dst.a == 0 || src.a == 255 {
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

/// Composites pre-resolved instances (bottom -> top) into a viewport-sized RGBA buffer.
///
/// Pass 1 blends every instance's fill in order. Each instance is first rasterized
/// into a small scratch coverage buffer (its clipped screen footprint, padded by
/// 1px), so each screen pixel is blended once per object and the outline is derived
/// from the object's *own* footprint - independent of overlaps.
/// Pass 2 paints all outlines fully opaque in the same order, so an outline stays
/// visible even when another object lies on top of it.
///
/// Cost scales with visible, larger-than-a-pixel objects:
/// - off-screen bboxes are culled before touching their mask,
/// - bboxes within one screen pixel collapse to a single stamped pixel,
/// - only the visible sub-rectangle of each mask is walked.
fn composite_object_instances(
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Vec<slint::Rgba8Pixel> {
    let bw = buf_w as i32;
    let bh = buf_h as i32;
    let stride = buf_w as usize;
    let mut pixels = vec![
        slint::Rgba8Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        };
        stride * buf_h as usize
    ];
    if zoom <= 0.0 || buf_w == 0 || buf_h == 0 {
        return pixels;
    }

    // Scratch buffers reused across instances (no per-instance allocation).
    let mut coverage: Vec<u8> = Vec::new();
    let mut x_edges: Vec<(i32, i32)> = Vec::new();
    // Outline pixels of all instances, plus (end offset, colour) per instance.
    let mut outline_idx: Vec<u32> = Vec::new();
    let mut outline_runs: Vec<(usize, slint::Rgba8Pixel)> = Vec::with_capacity(instances.len());

    for inst in instances {
        let bbox = inst.bbox;
        let bbox_w = (bbox[2] - bbox[0] + 1) as usize;
        let bbox_h = (bbox[3] - bbox[1] + 1) as usize;
        if bbox_w == 0 || bbox_h == 0 {
            continue;
        }

        // Screen footprint of the whole bbox (bbox[2]/[3] are inclusive).
        let sx0 = bbox[0] as f32 * zoom + off_x;
        let sy0 = bbox[1] as f32 * zoom + off_y;
        let sx1 = (bbox[2] + 1) as f32 * zoom + off_x;
        let sy1 = (bbox[3] + 1) as f32 * zoom + off_y;

        // Viewport culling.
        if sx1 <= 0.0 || sy1 <= 0.0 || sx0 >= buf_w as f32 || sy0 >= buf_h as f32 {
            continue;
        }

        // Sub-pixel collapse: one stamped pixel (isolated -> outline pixel).
        if sx1 - sx0 <= 1.0 && sy1 - sy0 <= 1.0 {
            let px = (((sx0 + sx1) * 0.5).max(0.0) as usize).min(stride - 1);
            let py = (((sy0 + sy1) * 0.5).max(0.0) as usize).min(buf_h as usize - 1);
            let i = py * stride + px;
            blend_pixel_over(&mut pixels[i], inst.color);
            outline_idx.push(i as u32);
            outline_runs.push((outline_idx.len(), inst.color));
            continue;
        }

        // Scratch region = footprint clipped to the viewport, padded by 1px so a
        // viewport-clipped object doesn't get a fake outline at the clip edge.
        let rx0 = (sx0.floor() as i32).max(-1);
        let ry0 = (sy0.floor() as i32).max(-1);
        let rx1 = ((sx1.floor() as i32) + 1).min(bw + 1);
        let ry1 = ((sy1.floor() as i32) + 1).min(bh + 1);
        if rx1 <= rx0 || ry1 <= ry0 {
            continue;
        }
        let rw = (rx1 - rx0) as usize;
        let rh = (ry1 - ry0) as usize;
        coverage.clear();
        coverage.resize(rw * rh, 0);

        // Visible sub-rectangle of the bbox in local image-pixel coordinates.
        let local_range = |r0: i32, r1: i32, off: f32, origin: u32, len: usize| {
            let a = (((r0 as f32 - off) / zoom).floor() as i64 - origin as i64 - 1)
                .clamp(0, len as i64) as usize;
            let b = (((r1 as f32 - off) / zoom).ceil() as i64 - origin as i64 + 1)
                .clamp(0, len as i64) as usize;
            (a, b)
        };
        let (lx0, lx1) = local_range(rx0, rx1, off_x, bbox[0], bbox_w);
        let (ly0, ly1) = local_range(ry0, ry1, off_y, bbox[1], bbox_h);

        // Per-column screen edges, computed once (shared edges => no gaps/overlap;
        // at least one pixel wide so tiny masks at low zoom don't develop holes).
        x_edges.clear();
        x_edges.extend((lx0..lx1).map(|lx| {
            let ax = (bbox[0] as usize + lx) as f32;
            let a = (ax * zoom + off_x).floor() as i32;
            let b = (((ax + 1.0) * zoom + off_x).floor() as i32).max(a + 1);
            (a.max(rx0), b.min(rx1))
        }));

        for ly in ly0..ly1 {
            let ay = (bbox[1] as usize + ly) as f32;
            let y0f = (ay * zoom + off_y).floor() as i32;
            let y1f = (((ay + 1.0) * zoom + off_y).floor() as i32).max(y0f + 1);
            let (y0, y1) = (y0f.max(ry0), y1f.min(ry1));
            if y0 >= y1 {
                continue;
            }
            let row_start = ly * bbox_w;
            let Some(row_bits) = inst.mask.get(row_start + lx0..row_start + lx1) else {
                continue;
            };
            for rel in row_bits.iter_ones() {
                let (x0, x1) = x_edges[rel];
                if x0 >= x1 {
                    continue;
                }
                for py in y0..y1 {
                    let row = (py - ry0) as usize * rw;
                    coverage[row + (x0 - rx0) as usize..row + (x1 - rx0) as usize].fill(1);
                }
            }
        }

        // Resolve: blend the fill once per covered on-screen pixel and record
        // this object's own outline (covered pixel with an uncovered 4-neighbour).
        for sy in 0..rh {
            let py = ry0 + sy as i32;
            if py < 0 || py >= bh {
                continue;
            }
            for sx in 0..rw {
                let c = sy * rw + sx;
                if coverage[c] == 0 {
                    continue;
                }
                let px = rx0 + sx as i32;
                if px < 0 || px >= bw {
                    continue;
                }
                let i = py as usize * stride + px as usize;
                blend_pixel_over(&mut pixels[i], inst.color);

                let is_border = sx == 0
                    || sx + 1 == rw
                    || sy == 0
                    || sy + 1 == rh
                    || coverage[c - 1] == 0
                    || coverage[c + 1] == 0
                    || coverage[c - rw] == 0
                    || coverage[c + rw] == 0;
                if is_border {
                    outline_idx.push(i as u32);
                }
            }
        }
        outline_runs.push((outline_idx.len(), inst.color));
    }

    // Outline pass: opaque, in z-order, always on top of all fills.
    let mut start = 0;
    for (end, color) in outline_runs {
        let opaque = slint::Rgba8Pixel { a: 255, ..color };
        for &i in &outline_idx[start..end] {
            pixels[i as usize] = opaque;
        }
        start = end;
    }

    pixels
}

/// Draw-order key; instances are painted in ascending key order (bigger = on top).
/// 0 = unclassified/unknown class, 1..=class_count = class stack (rank 0, the
/// first class in the list, gets the biggest value), then selected class, then
/// the selected object.
fn z_key(
    is_selected_object: bool,
    in_selected_class: bool,
    best_rank: Option<usize>,
    class_count: usize,
) -> u32 {
    if is_selected_object {
        u32::MAX
    } else if in_selected_class {
        u32::MAX - 1
    } else {
        best_rank.map_or(0, |r| (class_count - r.min(class_count - 1)) as u32)
    }
}
/// Resolved draw/pick priority for the current project state. Built once per
/// redraw or click, then queried per object (no per-object project access).
pub(crate) struct ZOrder {
    rank: HashMap<ObjectClass, usize>, // class -> index in class list (0 = on top)
    class_count: usize,
    selected_class: ObjectClass,
}

impl ZOrder {
    pub(crate) fn new<P: ProjectExt + ?Sized>(project: &P) -> Self {
        let rank: HashMap<ObjectClass, usize> = project
            .get_object_classes()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();
        Self {
            class_count: rank.len().max(1),
            rank,
            selected_class: project.get_selected_object_class(),
        }
    }

    /// Bigger = further on top.
    pub(crate) fn key(&self, classes: &HashSet<ObjectClass>, is_selected_object: bool) -> u32 {
        let in_selected_class = matches!(self.selected_class, ObjectClass::Valid(_))
            && classes.contains(&self.selected_class);
        let best_rank = classes
            .iter()
            .filter_map(|c| self.rank.get(c))
            .min()
            .copied();
        z_key(
            is_selected_object,
            in_selected_class,
            best_rank,
            self.class_count,
        )
    }
}

#[cfg(test)]
mod composite_object_instances_tests {
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
    fn single_pixel_object_renders_at_correct_location_with_border_alpha() {
        // 3x3 bbox, only the centre bit set -> one isolated screen pixel.
        let bbox = [0u32, 0, 2, 2];
        let mut mask = bitvec![u64, Lsb0; 0; 9];
        mask.set(4, true); // local (1,1) in a 3-wide bbox
        let color = rgba(10, 20, 30, 128);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 5, 5, 1.0, 0.0, 0.0);

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
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 8, 8, 2.0, 0.0, 0.0);

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
    fn object_entirely_outside_viewport_is_culled_and_produces_no_pixels() {
        let bbox = [1000u32, 1000, 1001, 1001];
        let mask = full_mask(bbox);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(255, 255, 255, 255),
        }];

        let pixels = composite_object_instances(&instances, 8, 8, 1.0, 0.0, 0.0);

        assert!(pixels.iter().all(|p| p.a == 0));
    }

    #[test]
    fn object_straddling_the_viewport_edge_is_not_over_culled() {
        // 3x3 fully-set mask anchored at the origin, but the buffer is only 2x2:
        // the bbox extends past the buffer, yet still overlaps it and must not be
        // culled - the visible 2x2 corner should render fully.
        let bbox = [0u32, 0, 2, 2];
        let mask = full_mask(bbox);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color: rgba(1, 2, 3, 200),
        }];

        let pixels = composite_object_instances(&instances, 2, 2, 1.0, 0.0, 0.0);

        assert!(
            pixels.iter().all(|p| p.r == 1 && p.g == 2 && p.b == 3),
            "every pixel of the visible corner should be painted: {pixels:?}"
        );
    }

    #[test]
    fn far_zoomed_out_object_collapses_to_exactly_one_pixel_instead_of_vanishing() {
        // A small (2x2) object viewed at zoom=0.01: its screen footprint
        // (2 * 0.01 = 0.02px) is far below one pixel. Without the collapse path
        // this would either vanish (every mask pixel rounds into the same
        // sub-pixel slot that the old per-pixel clipping logic could drop) or
        // require iterating mask bits for no visual gain; with it, exactly one
        // pixel must be painted so the object doesn't silently disappear.
        let bbox = [100u32, 100, 101, 101];
        let mask = full_mask(bbox);
        let color = rgba(50, 60, 70, 222);
        let instances = [ObjectDrawInstance {
            bbox,
            mask: &mask,
            color,
        }];

        let pixels = composite_object_instances(&instances, 50, 50, 0.01, 0.0, 0.0);

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
    fn overlapping_objects_blend_with_porter_duff_over() {
        let bbox = [0u32, 0, 0, 0]; // single-pixel bbox
        let mask = full_mask(bbox);
        let bottom = rgba(255, 0, 0, 128);
        let top = rgba(0, 0, 255, 128);
        let instances = [
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: bottom,
            },
            ObjectDrawInstance {
                bbox,
                mask: &mask,
                color: top,
            },
        ];

        let pixels = composite_object_instances(&instances, 3, 3, 1.0, 0.0, 0.0);

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
        let taken = pair
            .0
            .lock()
            .unwrap()
            .take()
            .expect("a task must be pending");
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

        let taken = pair
            .0
            .lock()
            .unwrap()
            .take()
            .expect("a task must be pending");
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
        pair.0
            .lock()
            .unwrap()
            .take()
            .expect("a task must be pending");

        // A later, unrelated plain dispatch must not resurrect the old flag.
        merge_into_slot(&pair, DrawingTask::default());
        let taken = pair
            .0
            .lock()
            .unwrap()
            .take()
            .expect("a task must be pending");
        assert!(
            !taken.is_new_image,
            "a fresh dispatch into an empty (already-consumed) slot must not merge with stale state"
        );
    }
}

#[cfg(test)]
mod breakpoint_state_tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;
    use evanalyzer_app::images::ManagedImage;
    use evanalyzer_app::images::Point2d;
    use kornia_image::{Image, ImageSize};

    fn make_controller() -> ViewportController {
        ViewportController::new(slint::Weak::default(), test_ui_state())
    }

    fn gray_image() -> ImageContainer {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let image = Image::<f32, 1>::new(size, vec![0.0f32; 4]).unwrap();
        ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        })
    }

    // -- breakpoint_view_mode / set_breakpoint_view_mode -----------------------

    #[test]
    fn breakpoint_view_mode_defaults_to_image() {
        let controller = make_controller();
        assert_eq!(controller.breakpoint_view_mode(), BreakpointViewMode::Image);
    }

    #[test]
    fn set_breakpoint_view_mode_round_trips_every_valid_value() {
        let controller = make_controller();

        controller.set_breakpoint_view_mode(1);
        assert_eq!(
            controller.breakpoint_view_mode(),
            BreakpointViewMode::Segmentation
        );

        controller.set_breakpoint_view_mode(2);
        assert_eq!(
            controller.breakpoint_view_mode(),
            BreakpointViewMode::Instances
        );

        controller.set_breakpoint_view_mode(0);
        assert_eq!(controller.breakpoint_view_mode(), BreakpointViewMode::Image);
    }

    #[test]
    fn set_breakpoint_view_mode_out_of_range_falls_back_to_image() {
        let controller = make_controller();
        controller.set_breakpoint_view_mode(1);

        controller.set_breakpoint_view_mode(99);

        assert_eq!(controller.breakpoint_view_mode(), BreakpointViewMode::Image);
    }

    // -- set_show_breakpoint ------------------------------------------------------

    #[test]
    fn set_show_breakpoint_stores_the_given_flag() {
        let controller = make_controller();
        assert!(!controller.show_breakpoint.load(Ordering::Relaxed));

        controller.set_show_breakpoint(true);
        assert!(controller.show_breakpoint.load(Ordering::Relaxed));

        controller.set_show_breakpoint(false);
        assert!(!controller.show_breakpoint.load(Ordering::Relaxed));
    }

    // -- set_breakpoint_channel ----------------------------------------------------

    #[test]
    fn set_breakpoint_channel_stores_the_given_buffers_and_geometry() {
        let controller = make_controller();

        controller.set_breakpoint_channel(
            gray_image(),
            Some(gray_image()),
            None,
            10,
            20,
            2,
            2,
            8,
            Some(3),
        );

        let stored = controller.breakpoint_channel.read().unwrap();
        let data = stored.as_ref().expect("breakpoint channel must be set");
        assert!(data.segmentation.is_some());
        assert!(data.instances.is_none());
        assert_eq!(data.tile_offset_x, 10);
        assert_eq!(data.tile_offset_y, 20);
        assert_eq!(data.tile_width, 2);
        assert_eq!(data.tile_height, 2);
        assert_eq!(data.nr_bits, 8);
        assert_eq!(data.channel_idx, Some(3));
    }

    #[test]
    fn set_breakpoint_channel_overwrites_a_previous_value() {
        let controller = make_controller();
        controller.set_breakpoint_channel(gray_image(), None, None, 0, 0, 1, 1, 8, None);

        controller.set_breakpoint_channel(gray_image(), None, None, 5, 5, 1, 1, 16, Some(1));

        let stored = controller.breakpoint_channel.read().unwrap();
        let data = stored.as_ref().unwrap();
        assert_eq!(data.tile_offset_x, 5);
        assert_eq!(data.nr_bits, 16);
        assert_eq!(data.channel_idx, Some(1));
    }
}
