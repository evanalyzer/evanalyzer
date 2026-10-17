use crate::ViewportState as ViewportSlintState;
use crate::editor::viewport_cache::{ViewportCache, effective_cache_mb};
use crate::editor::viewport_task::{DrawingTask, TaskDispatch};
use crate::helper::color_generators::get_colors_from_class;
use crate::helper::size_formater::{BYTES_PER_MB, format_bytes};
use crate::{AppWindow, HistogramData, HistogramState, PipelinesPanelState, RetainedTile, UiState};
use evanalyzer_app::global::ViewportMemorySettings;
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::ObjectClass;
use slint::{Color, ComponentHandle, Model, VecModel};
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

/// Where a rendered buffer lies, in full-resolution image pixels. The
/// viewport places it with the live zoom/offset (see `viewport.slint`), so a
/// pan/zoom moves what is already on screen without waiting for a re-render.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ImageRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl ImageRect {
    /// The image area under the screen rectangle (`x`, `y`, `w`, `h`) of a
    /// buffer rendered at `zoom` and `offset_x`/`offset_y`.
    pub fn from_screen(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        zoom: f32,
        offset_x: f32,
        offset_y: f32,
    ) -> Self {
        if zoom <= 0.0 {
            return Self::default();
        }
        Self {
            x: (x - offset_x) / zoom,
            y: (y - offset_y) / zoom,
            w: w / zoom,
            h: h / zoom,
        }
    }

    /// Whether `other` lies within this rectangle (up to `eps` image pixels).
    fn contains(&self, other: &ImageRect, eps: f32) -> bool {
        other.x >= self.x - eps
            && other.y >= self.y - eps
            && other.x + other.w <= self.x + self.w + eps
            && other.y + other.h <= self.y + self.h + eps
    }
}

/// Bookkeeping of one high-res render in the tile history - the data
/// [`plan_tile_history`] decides on, without the image itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TileMeta {
    pub rect: ImageRect,
    /// Zoom it was rendered for - higher is sharper.
    pub zoom: f32,
    /// Content generation (see `ViewportController::content_generation`) it
    /// shows; only renders of the current one may be kept.
    pub generation: i32,
    /// Insertion order - the oldest is evicted first.
    pub seq: i32,
    pub bytes: u64,
}

/// Whether `top` hides `tile` entirely, at least as sharply.
fn covers(top: &TileMeta, tile: &TileMeta) -> bool {
    top.rect.contains(&tile.rect, 0.5) && tile.zoom <= top.zoom * 1.0001
}

/// Which renders the tile history keeps once the current high-res render
/// (`displaced`) is replaced by `current` (`None`: nothing replaced, just
/// re-check against `budget_bytes`), both of content `generation`.
///
/// Returns indexes in draw order (bottom to top, sharpest last): `0..len`
/// are `existing`, `existing.len()` is `displaced`. With `budget_bytes`
/// `None` (history off) nothing is kept. Renders of another generation show
/// outdated content and are dropped; so are renders `current` or
/// `displaced` hide entirely at least as sharply - e.g. the same tile
/// rendered again after a small pan; `current` joins the history itself
/// once replaced. Then the oldest go until the rest fits the budget.
pub(crate) fn plan_tile_history(
    existing: &[TileMeta],
    displaced: Option<TileMeta>,
    current: Option<TileMeta>,
    generation: i32,
    budget_bytes: Option<u64>,
) -> Vec<usize> {
    let Some(budget) = budget_bytes else {
        return Vec::new();
    };
    let hidden_by_current = |t: &TileMeta| current.is_some_and(|c| covers(&c, t));
    let displaced =
        displaced.filter(|d| d.generation == generation && d.rect.w > 0.0 && !hidden_by_current(d));
    let mut keep: Vec<(usize, TileMeta)> = existing
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, t)| t.generation == generation && !hidden_by_current(t))
        .filter(|(_, t)| displaced.is_none_or(|d| !covers(&d, t)))
        .collect();
    if let Some(d) = displaced {
        keep.push((existing.len(), d));
    }

    // Evict the oldest until the rest fits.
    keep.sort_by_key(|(_, t)| t.seq);
    let mut total: u64 = keep.iter().map(|(_, t)| t.bytes).sum();
    let mut first = 0;
    while total > budget && first < keep.len() {
        total -= keep[first].1.bytes;
        first += 1;
    }
    let mut keep = keep.split_off(first);

    keep.sort_by(|(_, a), (_, b)| a.zoom.total_cmp(&b.zoom).then(a.seq.cmp(&b.seq)));
    keep.into_iter().map(|(i, _)| i).collect()
}

fn retained_tile_meta(tile: &RetainedTile) -> TileMeta {
    TileMeta {
        rect: ImageRect {
            x: tile.img_x,
            y: tile.img_y,
            w: tile.img_w,
            h: tile.img_h,
        },
        zoom: tile.zoom,
        generation: tile.generation,
        seq: tile.seq,
        bytes: tile.bytes.max(0) as u64,
    }
}

/// Applies [`plan_tile_history`] to the Slint tile history, `displaced`
/// being the high-res render about to be replaced. UI thread only.
fn update_retained_tiles(
    view_state: &ViewportSlintState,
    displaced: Option<RetainedTile>,
    current: Option<TileMeta>,
    generation: i32,
    budget_bytes: Option<u64>,
) {
    let existing: Vec<RetainedTile> = view_state.get_retained_tiles().iter().collect();
    if existing.is_empty() && (displaced.is_none() || budget_bytes.is_none()) {
        return;
    }
    let metas: Vec<TileMeta> = existing.iter().map(retained_tile_meta).collect();
    let displaced = displaced.map(|mut d| {
        d.seq = existing
            .iter()
            .map(|t| t.seq)
            .max()
            .map_or(0, |s| s.wrapping_add(1));
        d
    });
    let plan = plan_tile_history(
        &metas,
        displaced.as_ref().map(retained_tile_meta),
        current,
        generation,
        budget_bytes,
    );
    let kept: Vec<RetainedTile> = plan
        .into_iter()
        .map(|i| match existing.get(i) {
            Some(tile) => tile.clone(),
            None => displaced
                .clone()
                .expect("index past `existing` is the displaced tile"),
        })
        .collect();
    view_state.set_retained_tiles(Rc::new(VecModel::from(kept)).into());
}

/// Hands freed heap memory back to the operating system. glibc keeps freed
/// chunks of its heap for reuse instead, so the process wouldn't shrink
/// after a cache is emptied. Elsewhere large blocks go back on their own.
fn return_freed_memory_to_os() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only releases free memory of the allocator.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// The tile history budget in bytes, `None` when the history is off.
fn tile_history_budget(settings: &ViewportMemorySettings) -> Option<u64> {
    settings
        .tile_history_enabled
        .then(|| settings.tile_history_budget_mb as u64 * BYTES_PER_MB)
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
    /// Compare (top bar, only with a breakpoint image): the original is
    /// rendered as the image and the breakpoint image as a second one over
    /// it, cut at the divider in the UI. Off: no second render.
    pub(crate) compare: Arc<AtomicBool>,
    pub(crate) high_res_posted_count: Arc<AtomicU64>,
    pub(crate) high_res_last_count_at_false: Arc<AtomicU64>,
    pub(crate) high_res_is_ready: AtomicBool,
    /// Bumped by every redraw request that changes what the image looks
    /// like (new image/series, channels, histogram, breakpoint view) - but
    /// not by pan/zoom/resize. Renders are tagged with it, so the tile
    /// history only ever keeps renders of what is shown now.
    pub(crate) content_generation: AtomicU64,
    /// This computer's viewport memory settings - see
    /// [`ViewportMemorySettings`].
    pub(crate) memory_settings: RwLock<ViewportMemorySettings>,
    /// Full-resolution size of the current image, for the navigator, which
    /// pan/zoom update without a render.
    pub(crate) full_image_size: Mutex<(i64, i64)>,
    /// The tile cache the viewport workers read through - limited by the
    /// memory settings, emptied for an analysis.
    pub(crate) viewport_cache: Arc<ViewportCache>,
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
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_cache: Arc<ViewportCache>,
    ) -> Self {
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
            compare: Arc::new(AtomicBool::new(false)),
            high_res_posted_count: Arc::new(AtomicU64::new(0)),
            high_res_last_count_at_false: Arc::new(AtomicU64::new(0)),
            high_res_is_ready: AtomicBool::new(false),
            content_generation: AtomicU64::new(0),
            memory_settings: RwLock::new(app_state.viewport_memory_settings()),
            full_image_size: Mutex::new((0, 0)),
            viewport_cache,
            app_state,
        }
    }

    /// The current content generation - see `content_generation`.
    pub(crate) fn content_generation(&self) -> u64 {
        self.content_generation.load(Ordering::SeqCst)
    }

    fn bump_content_generation(&self) {
        self.content_generation.fetch_add(1, Ordering::SeqCst);
    }

    /// The viewport memory settings in effect.
    pub fn memory_settings(&self) -> ViewportMemorySettings {
        self.memory_settings.read().expect("Poisoned").clone()
    }

    /// Applies new viewport memory settings: frees kept renders and cached
    /// tiles the new limits no longer allow right away and redraws the
    /// objects with the new margin.
    pub fn set_memory_settings(&self, settings: ViewportMemorySettings) {
        let margin_changed =
            self.memory_settings().object_overlay_margin_px != settings.object_overlay_margin_px;
        let budget = tile_history_budget(&settings);
        let (tile_mb, overview_mb) = effective_cache_mb(&settings);
        self.viewport_cache.set_limits(tile_mb, overview_mb);
        *self.memory_settings.write().expect("Poisoned") = settings;

        let generation = self.content_generation() as i32;
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                update_retained_tiles(
                    &ui.global::<ViewportSlintState>(),
                    None,
                    None,
                    generation,
                    budget,
                );
            }
        })
        .ok();
        if margin_changed {
            self.trigger_image_redraw_objects();
        }
    }

    /// Frees what the viewport keeps beyond what is on screen - the tile
    /// and overview caches and the kept earlier renders - if the settings
    /// ask for it. Called when an analysis starts on this computer; the
    /// caches fill up again as the viewport is used.
    pub fn release_preview_memory_for_analysis(&self) {
        if !self.memory_settings().free_caches_on_analysis {
            return;
        }
        let (tiles, overviews) = self.viewport_cache.held_bytes();
        self.viewport_cache.clear();
        log::info!(
            "Freed the viewport caches for the analysis: {} tiles, {} overviews",
            format_bytes(tiles as f64),
            format_bytes(overviews as f64)
        );
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ViewportSlintState>()
                    .set_retained_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
            }
            return_freed_memory_to_os();
        })
        .ok();
    }

    /// Lets a 'not ready' from the next low-res render of new content hide
    /// the high-res layer again - see `sync_high_res_ready_to_slint`.
    fn reset_high_res_barrier(&self) {
        self.high_res_is_ready.store(false, Ordering::SeqCst);
        self.high_res_posted_count.store(0, Ordering::SeqCst);
        self.high_res_last_count_at_false.store(0, Ordering::SeqCst);
    }

    pub fn trigger_new_image_redraw(&self) {
        self.bump_content_generation();
        self.reset_high_res_barrier();
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
        self.bump_content_generation();
        self.reset_high_res_barrier();
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = true;
        task.auto_adjust_selected = false;
        task.is_new_image = false;
        task.fit_to_screen = true;
        task.is_new_series = true;
        self.dispatch_worker_task(task.clone(), TaskDispatch::HighResAndLowRes);
        self.dispatch_worker_task(task, TaskDispatch::Objects);
    }

    /// Redraws after a display change (histogram, breakpoint view, ...).
    /// The overview is redrawn too: pan/zoom no longer re-render it, so it
    /// would otherwise keep showing the old look.
    pub fn trigger_image_redraw(&self) {
        self.bump_content_generation();
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = false;
        task.auto_adjust_selected = false;
        task.is_new_image = false;
        task.fit_to_screen = false;
        task.is_new_series = false;
        self.dispatch_worker_task(task, TaskDispatch::HighResAndLowRes);
    }

    pub fn trigger_image_redraw_with_auto_adjust(&self) {
        self.bump_content_generation();
        let mut task: DrawingTask = DrawingTask::default();
        task.auto_adjust_if_not_set = false;
        task.auto_adjust_selected = true;
        task.is_new_image = false;
        task.fit_to_screen = false;
        task.is_new_series = false;
        self.dispatch_worker_task(task, TaskDispatch::HighResAndLowRes);
    }

    pub fn trigger_image_redraw_objects(&self) {
        let task: DrawingTask = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::Objects);
    }

    /// Re-renders only the overview - while panning/zooming with the
    /// software renderer (Windows), whose overview covers just the viewport.
    pub fn trigger_redraw_low_res(&self) {
        let task = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::LowRes);
    }

    /// Redraws after a change of what is shown (channels, z/t stack, ...).
    pub fn trigger_redraw_low_res_and_high_res(&self) {
        self.bump_content_generation();
        let task = DrawingTask::default();
        self.dispatch_worker_task(task, TaskDispatch::HighResAndLowRes);
    }

    /// Re-renders once a pan/zoom/resize settled: the sharp tile and the
    /// objects for the new view. The content is unchanged, so the render
    /// being replaced may join the tile history. Elsewhere the overview
    /// covers the whole image and needs no re-render; the software renderer
    /// (Windows) gets a viewport-sized one, which does.
    pub fn trigger_viewport_redraw(&self) {
        let scope = if cfg!(target_os = "windows") {
            TaskDispatch::HighResAndLowRes
        } else {
            TaskDispatch::HighRes
        };
        self.dispatch_worker_task(DrawingTask::default(), scope);
        self.trigger_image_redraw_objects();
    }

    /// Updates the navigator from the current viewport transform - pan/zoom
    /// call this directly, since they no longer wait for a render.
    pub fn sync_navigator_from_viewport_state(&self) {
        let (full_w, full_h) = *self.full_image_size.lock().expect("Poisoned");
        let (vp_w, vp_h, off_x, off_y) = {
            let s = self.viewport_state.read().expect("Poisoned");
            (s.viewport_width, s.viewport_height, s.offset_x, s.offset_y)
        };
        self.sync_navigator_to_slint(full_w, full_h, vp_w, vp_h, off_x, off_y);
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
    /// * `rect` - Where the buffer lies, in full-resolution image pixels.
    /// * `render_zoom` - The zoom the buffer was rendered for.
    /// * `generation` - The content generation the buffer shows (see
    ///   `content_generation`), for the tile history.
    /// * `is_low_res` - A flag indicating if the provided buffer is a preview (proxy) or a full-resolution render.
    ///
    /// ### Returns
    /// * `Ok(())` if the state was successfully pushed to the UI components.
    /// * `Err(InternalErrors)` if the synchronization failed due to a locked resource or internal state error.
    pub fn sync_viewport_state_to_slint(
        &self,
        pixel_buffer: slint::SharedPixelBuffer<slint::Rgb8Pixel>,
        svg_histogram_data: Vec<HistogramData>,
        rect: ImageRect,
        render_zoom: f32,
        generation: u64,
        is_low_res: bool,
    ) {
        let budget = tile_history_budget(&self.memory_settings());
        let generation = generation as i32;
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let view_state = ui.global::<ViewportSlintState>();
                if is_low_res {
                    view_state.set_ghost_img_x(rect.x);
                    view_state.set_ghost_img_y(rect.y);
                    view_state.set_ghost_img_w(rect.w);
                    view_state.set_ghost_img_h(rect.h);

                    ui.set_ghost_image(slint::Image::from_rgb8(pixel_buffer));
                } else {
                    // The render being replaced may join the tile history.
                    let previous = ui.get_display_image();
                    let size = previous.size();
                    let displaced =
                        (size.width > 0 && view_state.get_high_res_ready()).then(|| RetainedTile {
                            image: previous,
                            img_x: view_state.get_hr_img_x(),
                            img_y: view_state.get_hr_img_y(),
                            img_w: view_state.get_hr_img_w(),
                            img_h: view_state.get_hr_img_h(),
                            zoom: view_state.get_hr_zoom(),
                            generation: view_state.get_hr_generation(),
                            seq: 0,
                            bytes: (size.width as u64 * size.height as u64 * 3).min(i32::MAX as u64)
                                as i32,
                        });
                    let current = TileMeta {
                        rect,
                        zoom: render_zoom,
                        generation,
                        seq: 0,
                        bytes: 0,
                    };
                    update_retained_tiles(
                        &view_state,
                        displaced,
                        Some(current),
                        generation,
                        budget,
                    );

                    view_state.set_hr_img_x(rect.x);
                    view_state.set_hr_img_y(rect.y);
                    view_state.set_hr_img_w(rect.w);
                    view_state.set_hr_img_h(rect.h);
                    view_state.set_hr_zoom(render_zoom);
                    view_state.set_hr_generation(generation);

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

        // If the breakpoint view is already active (or compared), re-render
        // immediately so the updated image is visible without requiring a
        // manual pan/zoom.
        if self.show_breakpoint.load(Ordering::Relaxed) || self.comparing() {
            self.trigger_image_redraw();
        }
    }

    /// Switches the HighRes viewport worker between the original image and the
    /// breakpoint channel, then triggers a redraw so the change is immediate.
    pub fn set_show_breakpoint(&self, show: bool) {
        self.show_breakpoint.store(show, Ordering::Relaxed);
        self.trigger_image_redraw();
    }

    /// Whether the breakpoint image is compared with the original.
    pub fn comparing(&self) -> bool {
        self.compare.load(Ordering::Relaxed)
    }

    /// Compare on or off. Off frees the second image at once.
    pub fn set_compare(&self, on: bool) {
        if self.compare.swap(on, Ordering::Relaxed) == on {
            return;
        }
        if !on {
            let ui_weak = self.ui.clone();
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    let view = ui.global::<ViewportSlintState>();
                    view.set_compare_ready(false);
                    view.set_compare_image(slint::Image::default());
                }
            })
            .ok();
        }
        self.trigger_image_redraw();
    }

    /// The breakpoint image of a compare, where it lies in the image.
    pub fn sync_compare_to_slint(
        &self,
        pixel_buffer: slint::SharedPixelBuffer<slint::Rgb8Pixel>,
        rect: ImageRect,
    ) {
        let compare = self.compare.clone();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            // Switched off meanwhile: not shown (and not kept).
            if !compare.load(Ordering::Relaxed) {
                return;
            }
            if let Some(ui) = ui_weak.upgrade() {
                let view = ui.global::<ViewportSlintState>();
                view.set_cmp_img_x(rect.x);
                view.set_cmp_img_y(rect.y);
                view.set_cmp_img_w(rect.w);
                view.set_cmp_img_h(rect.h);
                view.set_compare_image(slint::Image::from_rgb8(pixel_buffer));
                view.set_compare_ready(true);
            }
        })
        .ok();
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
        if self.show_breakpoint.load(Ordering::Relaxed) || self.comparing() {
            self.trigger_image_redraw();
        }
    }

    /// Shows (`true`, a high-res render arrived) or hides (`false`, the
    /// overview of new content arrived first) the high-res layer and the
    /// tile history. A `false` after a `true` for the same new content is
    /// out of date and ignored - `reset_high_res_barrier` re-arms it.
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
                    let view_state = ui.global::<ViewportSlintState>();
                    view_state.set_high_res_ready(false);
                    // New content: the kept renders show the previous one.
                    view_state
                        .set_retained_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
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

        // The overlay is rendered in screen space - at screen resolution
        // regardless of zoom - covering the viewport plus `margin` on every
        // side, so a pan doesn't uncover an empty border right away. It is
        // placed like the image layers (see viewport.slint Layer 3), so it
        // follows the image until the next render replaces it.
        let margin = self.memory_settings().object_overlay_margin_px as f32;
        let buf_w = (viewport_width + 2.0 * margin) as u32;
        let buf_h = (viewport_height + 2.0 * margin) as u32;
        let rect = ImageRect::from_screen(
            -margin,
            -margin,
            buf_w as f32,
            buf_h as f32,
            zoom,
            off_x,
            off_y,
        );

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
            if !is_object_drawn(&*project, hide_unclassified, &object.object_class) {
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

        // Composited straight into the buffer handed to Slint (starts out
        // transparent) - no second overlay-sized copy.
        let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(buf_w, buf_h);
        composite_object_instances_into(
            buffer.make_mut_slice(),
            &instances,
            buf_w,
            buf_h,
            zoom,
            off_x + margin,
            off_y + margin,
            zoom < 1.0,
        );

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let image = slint::Image::from_rgba8(buffer);
                ui.set_object_image(image);
                let view_state = ui.global::<ViewportSlintState>();
                view_state.set_obj_img_x(rect.x);
                view_state.set_obj_img_y(rect.y);
                view_state.set_obj_img_w(rect.w);
                view_state.set_obj_img_h(rect.h);
                view_state.set_object_ready(true);
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
/// - only the visible sub-rectangle of each mask is walked,
/// - zoomed out, mask columns sharing a screen column are checked as one run.
#[cfg(test)]
fn composite_object_instances(
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
) -> Vec<slint::Rgba8Pixel> {
    composite_object_instances_with(instances, buf_w, buf_h, zoom, off_x, off_y, zoom < 1.0)
}

/// [`composite_object_instances`], with the column runs (`use_runs`) chosen
/// by the caller - the tests check both ways produce the same pixels.
#[cfg(test)]
fn composite_object_instances_with(
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    use_runs: bool,
) -> Vec<slint::Rgba8Pixel> {
    let mut pixels = vec![
        slint::Rgba8Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        };
        buf_w as usize * buf_h as usize
    ];
    composite_object_instances_into(
        &mut pixels,
        instances,
        buf_w,
        buf_h,
        zoom,
        off_x,
        off_y,
        use_runs,
    );
    pixels
}

/// [`composite_object_instances`] into `pixels` - a transparent
/// `buf_w * buf_h` buffer.
#[allow(clippy::too_many_arguments)]
fn composite_object_instances_into(
    pixels: &mut [slint::Rgba8Pixel],
    instances: &[ObjectDrawInstance],
    buf_w: u32,
    buf_h: u32,
    zoom: f32,
    off_x: f32,
    off_y: f32,
    use_runs: bool,
) {
    let bw = buf_w as i32;
    let bh = buf_h as i32;
    let stride = buf_w as usize;
    debug_assert_eq!(pixels.len(), stride * buf_h as usize);
    if zoom <= 0.0 || buf_w == 0 || buf_h == 0 {
        return;
    }

    // Scratch buffers reused across instances (no per-instance allocation).
    let mut coverage: Vec<u8> = Vec::new();
    let mut x_edges: Vec<(i32, i32)> = Vec::new();
    // (first mask column, end mask column, first screen column, end screen
    // column) of each run of mask columns sharing their screen columns.
    let mut x_runs: Vec<(usize, usize, i32, i32)> = Vec::new();
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

        // Zoomed out, several neighbouring mask columns land on the same
        // screen column(s): group them into runs (mask columns rs..re ->
        // screen columns x0..x1), so each row checks a run with one
        // word-level `any()` instead of visiting every set bit. Exactly the
        // same coverage; zoomed in, every run would be a single column, so
        // walking the set bits stays cheaper there.
        if use_runs {
            x_runs.clear();
            for (rel, &(x0, x1)) in x_edges.iter().enumerate() {
                if x0 >= x1 {
                    continue;
                }
                match x_runs.last_mut() {
                    Some((_, re, rx0_, rx1_)) if *re == rel && (*rx0_, *rx1_) == (x0, x1) => {
                        *re = rel + 1
                    }
                    _ => x_runs.push((rel, rel + 1, x0, x1)),
                }
            }
        }

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
            if use_runs {
                for py in y0..y1 {
                    let row = (py - ry0) as usize * rw;
                    for &(rs, re, x0, x1) in &x_runs {
                        let cells = row + (x0 - rx0) as usize..row + (x1 - rx0) as usize;
                        if coverage[cells.clone()].iter().all(|&c| c != 0) {
                            continue;
                        }
                        if row_bits[rs..re].any() {
                            coverage[cells].fill(1);
                        }
                    }
                }
                continue;
            }
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
}

/// Whether an object with `classes` is drawn at all: not when it's
/// unclassified while unclassified objects are hidden, nor when every one of
/// its classes is hidden. Shared by the redraw and by click picking, so an
/// object that isn't drawn can never be picked either.
pub(crate) fn is_object_drawn<P: ProjectExt + ?Sized>(
    project: &P,
    hide_unclassified: bool,
    classes: &HashSet<ObjectClass>,
) -> bool {
    if classes.is_empty() {
        return !hide_unclassified;
    }
    classes.iter().any(|c| project.is_class_visible(c))
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
            .map(|(i, c)| (c.global(), i))
            .collect();
        Self {
            class_count: rank.len().max(1),
            rank,
            selected_class: project.get_selected_object_class(),
        }
    }

    /// Bigger = further on top.
    pub(crate) fn key(&self, classes: &HashSet<ObjectClass>, is_selected_object: bool) -> u32 {
        let in_selected_class = matches!(self.selected_class, ObjectClass::Global(_))
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

        // A single pixel is all outline: outlines are painted opaque in
        // z-order after every fill, so the top object's outline wins.
        assert_eq!(pixels[0], rgba(0, 0, 255, 255));
    }

    /// The zoomed-out column runs are only a shortcut: for any zoom and
    /// offset they must give exactly the pixels of walking every set bit.
    #[test]
    fn column_runs_produce_the_same_pixels_as_walking_every_bit() {
        let mut seed = 42u64;
        let mut rnd = |m: u32| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32) % m
        };
        let mut masks = Vec::new();
        let mut bboxes = Vec::new();
        for _ in 0..60 {
            let (w, h) = (1 + rnd(40), 1 + rnd(40));
            let (x, y) = (rnd(300), rnd(300));
            bboxes.push([x, y, x + w - 1, y + h - 1]);
            // Random fill, including empty and single-pixel masks.
            let density = rnd(4);
            let mask: BitVec<u64, Lsb0> = (0..w * h).map(|_| rnd(4) < density).collect();
            masks.push(mask);
        }
        let instances: Vec<ObjectDrawInstance> = bboxes
            .iter()
            .zip(&masks)
            .enumerate()
            .map(|(i, (bbox, mask))| ObjectDrawInstance {
                bbox: *bbox,
                mask,
                color: rgba((i * 37) as u8, (i * 91) as u8, 200, 100 + (i % 100) as u8),
            })
            .collect();

        for &zoom in &[0.07f32, 0.23, 0.5, 0.77, 0.999, 1.0, 1.5, 3.3] {
            for &(off_x, off_y) in &[(0.0f32, 0.0f32), (-17.3, 5.6), (11.9, -40.2)] {
                let bits = composite_object_instances_with(
                    &instances, 160, 120, zoom, off_x, off_y, false,
                );
                let runs =
                    composite_object_instances_with(&instances, 160, 120, zoom, off_x, off_y, true);
                assert!(
                    bits == runs,
                    "zoom {zoom}, offset ({off_x}, {off_y}): runs differ from bits"
                );
            }
        }
    }

    #[test]
    fn overlapping_interiors_blend_with_porter_duff_over_and_the_top_outline_wins() {
        let bbox = [1u32, 1, 3, 3]; // 3x3 -> the centre (2,2) is interior
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

        let pixels = composite_object_instances(&instances, 5, 5, 1.0, 0.0, 0.0);

        let mut expected = bottom;
        blend_pixel_over(&mut expected, top);
        assert_eq!(pixels[2 * 5 + 2], expected, "interior: both fills blended");
        assert_eq!(pixels[5 + 1], rgba(0, 0, 255, 255), "outline: top, opaque");
        assert_eq!(pixels[0].a, 0, "background stays empty");
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
        let ui_state = test_ui_state();
        ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
            Arc::new(ViewportCache::new(ui_state)),
        )
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

#[cfg(test)]
mod tile_history_tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> ImageRect {
        ImageRect { x, y, w, h }
    }

    fn tile(r: ImageRect, zoom: f32, seq: i32) -> TileMeta {
        TileMeta {
            rect: r,
            zoom,
            generation: 1,
            seq,
            bytes: 100,
        }
    }

    const PLENTY: Option<u64> = Some(1 << 30);

    #[test]
    fn a_screen_rectangle_maps_back_to_the_image_pixels_under_it() {
        // Zoom 2, image origin drawn at (-100, 40): screen x 0 shows image
        // x 50, and 200 screen pixels span 100 image pixels.
        let r = ImageRect::from_screen(0.0, 0.0, 200.0, 100.0, 2.0, -100.0, 40.0);
        assert_eq!(r, rect(50.0, -20.0, 100.0, 50.0));
        assert_eq!(
            ImageRect::from_screen(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            ImageRect::default(),
            "no zoom yet - no rectangle"
        );
    }

    #[test]
    fn with_the_history_off_nothing_is_kept() {
        let existing = [tile(rect(0.0, 0.0, 10.0, 10.0), 1.0, 0)];
        let displaced = tile(rect(50.0, 0.0, 10.0, 10.0), 1.0, 1);
        assert!(plan_tile_history(&existing, Some(displaced), None, 1, None).is_empty());
    }

    #[test]
    fn the_replaced_render_is_kept_and_the_sharpest_is_drawn_last() {
        let existing = [
            tile(rect(0.0, 0.0, 10.0, 10.0), 4.0, 0),
            tile(rect(100.0, 0.0, 10.0, 10.0), 1.0, 1),
        ];
        let displaced = tile(rect(200.0, 0.0, 10.0, 10.0), 2.0, 2);
        let current = tile(rect(300.0, 0.0, 10.0, 10.0), 2.0, 0);
        assert_eq!(
            plan_tile_history(&existing, Some(displaced), Some(current), 1, PLENTY),
            [1, 2, 0]
        );
    }

    #[test]
    fn renders_of_other_content_are_dropped() {
        let mut outdated = tile(rect(0.0, 0.0, 10.0, 10.0), 1.0, 0);
        outdated.generation = 0;
        let mut displaced = tile(rect(50.0, 0.0, 10.0, 10.0), 1.0, 1);
        displaced.generation = 0;
        assert!(plan_tile_history(&[outdated], Some(displaced), None, 1, PLENTY).is_empty());
    }

    #[test]
    fn a_render_hidden_at_least_as_sharply_is_dropped_a_sharper_one_stays() {
        let big = rect(0.0, 0.0, 100.0, 100.0);
        let inside = rect(10.0, 10.0, 20.0, 20.0);
        // The same tile rendered again after a small pan: no second copy.
        let displaced = tile(big, 1.0, 0);
        let current = tile(big, 1.0, 0);
        assert!(plan_tile_history(&[], Some(displaced), Some(current), 1, PLENTY).is_empty());

        // The replaced render hides a blurrier one but not a sharper one.
        let existing = [tile(inside, 0.5, 0), tile(inside, 3.0, 1)];
        let displaced = tile(big, 1.0, 2);
        assert_eq!(
            plan_tile_history(&existing, Some(displaced), None, 1, PLENTY),
            [2, 1]
        );
    }

    #[test]
    fn the_oldest_renders_go_first_until_the_rest_fits_the_budget() {
        let existing = [
            tile(rect(0.0, 0.0, 1.0, 1.0), 1.0, 5),
            tile(rect(10.0, 0.0, 1.0, 1.0), 1.0, 3),
            tile(rect(20.0, 0.0, 1.0, 1.0), 1.0, 4),
        ];
        let displaced = tile(rect(30.0, 0.0, 1.0, 1.0), 1.0, 6);
        // Room for two of the four (100 bytes each): the two newest.
        assert_eq!(
            plan_tile_history(&existing, Some(displaced), None, 1, Some(250)),
            [0, 3]
        );
        // A budget smaller than one render keeps none.
        assert!(plan_tile_history(&existing, Some(displaced), None, 1, Some(50)).is_empty());
    }
}
