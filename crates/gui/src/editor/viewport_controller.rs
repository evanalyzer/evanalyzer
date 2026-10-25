//! # viewport_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ViewportState as ViewportSlintState;
use crate::editor::object_render::{
    CHILD_RGB, DIMMED_RGB, ObjectDrawInstance, PARENT_RGB, SELECTED_RGB, ZOrder, draw_objects,
};
use crate::editor::results_overlay::{ResultsOverlay, draw_results_overlay};
use crate::editor::viewport_cache::{ViewportCache, effective_cache_mb};
use crate::editor::viewport_task::{DrawingTask, TaskDispatch};
use crate::helper::color_generators::get_colors_from_class;
use crate::helper::size_formater::{BYTES_PER_MB, format_bytes};
use crate::{AppWindow, HistogramData, HistogramState, PipelinesPanelState, RetainedTile, UiState};
use evanalyzer_app::global::ViewportMemorySettings;
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::images_settings::HistogramSettings;
use slint::{Color, ComponentHandle, Model, VecModel};
use std::collections::HashSet;
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

/// One tile's breakpoint image, kept for re-rendering when its histogram
/// window or the view mode changes. A preview captures up to the breakpoint
/// tiles setting of them (see `ViewportMemorySettings::breakpoint_tiles`).
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
    /// The breakpoint images of the last preview, one per captured tile -
    /// freed when the breakpoint is cleared, an analysis starts or another
    /// image opens (see [`Self::clear_breakpoint`]).
    pub(crate) breakpoint_tiles: Arc<RwLock<Vec<BreakpointChannelData>>>,
    /// Bumped whenever the compared picture changes (its tiles, histogram
    /// window, view mode, log scale, compare switched on): the worker
    /// renders it again only then - a pan or zoom just moves it.
    pub(crate) compare_epoch: AtomicU64,
    /// How often the compare was rendered (for the log and the tests).
    pub(crate) compare_renders: AtomicU64,
    /// Which of `breakpoint_tiles`' buffers to render (image/segmentation/
    /// instances). Stored as the raw `BreakpointViewMode` discriminant.
    pub(crate) breakpoint_view_mode: Arc<AtomicU8>,
    /// Compare (top bar, only with a breakpoint image): the original is
    /// rendered as the image and the breakpoint image as a second one over
    /// it, cut at the divider in the UI. Off: no second render.
    pub(crate) compare: Arc<AtomicBool>,
    /// Compare: the breakpoint image's histogram window - its own, not the
    /// channel's and not saved; `None`: auto-adjusted at its next render.
    pub(crate) breakpoint_histogram: RwLock<Option<HistogramSettings>>,
    /// Whose histogram the panel shows and changes: 0 the original's, 1
    /// the breakpoint image's (only while comparing).
    pub(crate) histogram_side: AtomicU8,
    /// The histogram's counts on a logarithmic scale (default) or linear.
    pub(crate) histogram_log: AtomicBool,
    /// The latest curve and value scale (2^bits) of each side.
    pub(crate) histograms: Mutex<[(Vec<HistogramData>, f32); 2]>,
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
    /// The results' split view: the analysis' objects, drawn instead of
    /// the project's (see [`ResultsOverlay`]).
    pub(crate) results_overlay: RwLock<Option<Arc<ResultsOverlay>>>,
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
            breakpoint_tiles: Arc::new(RwLock::new(Vec::new())),
            compare_epoch: AtomicU64::new(0),
            compare_renders: AtomicU64::new(0),
            breakpoint_view_mode: Arc::new(AtomicU8::new(BreakpointViewMode::default() as u8)),
            compare: Arc::new(AtomicBool::new(false)),
            breakpoint_histogram: RwLock::new(None),
            histogram_side: AtomicU8::new(0),
            histogram_log: AtomicBool::new(true),
            histograms: Mutex::new([(Vec::new(), 1.0), (Vec::new(), 1.0)]),
            high_res_posted_count: Arc::new(AtomicU64::new(0)),
            high_res_last_count_at_false: Arc::new(AtomicU64::new(0)),
            high_res_is_ready: AtomicBool::new(false),
            content_generation: AtomicU64::new(0),
            memory_settings: RwLock::new(app_state.viewport_memory_settings()),
            full_image_size: Mutex::new((0, 0)),
            viewport_cache,
            results_overlay: RwLock::new(None),
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
        // The breakpoint images always: up to ~190 MB a tile.
        self.clear_breakpoint();
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

    /// "Auto-scale all": every channel auto-adjusted, then redrawn.
    pub fn trigger_image_redraw_with_auto_adjust_all(&self) {
        self.bump_content_generation();
        let task = DrawingTask {
            auto_adjust_all: true,
            ..DrawingTask::default()
        };
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

                    ui.set_display_image(slint::Image::from_rgb8(pixel_buffer));
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

    /// The breakpoint image of one tile, replacing all others (one tile
    /// captured) - see [`Self::add_breakpoint_tile`].
    #[cfg(test)]
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
        self.add_breakpoint_tile(
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
            true,
        );
    }

    /// A tile's breakpoint image from a preview: `first` (the run's first)
    /// drops the last run's tiles, else it replaces the tile at the same
    /// place or joins them.
    pub fn add_breakpoint_tile(&self, tile: BreakpointChannelData, first: bool) {
        {
            let mut tiles = self.breakpoint_tiles.write().unwrap();
            if first {
                tiles.clear();
            }
            tiles.retain(|t| {
                (t.tile_offset_x, t.tile_offset_y) != (tile.tile_offset_x, tile.tile_offset_y)
            });
            tiles.push(tile);
        }
        self.bump_compare();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<PipelinesPanelState>()
                    .set_has_breakpoint_image(true);
            }
        })
        .ok();

        // Compared: re-render at once, so the new image shows without a
        // pan or zoom.
        if self.comparing() {
            self.trigger_image_redraw();
        }
    }

    /// No breakpoint images any more (the breakpoint cleared, an analysis
    /// starting): their RAM is freed - also the compare's render of them,
    /// at the next render.
    pub fn clear_breakpoint(&self) {
        let had = {
            let mut tiles = self.breakpoint_tiles.write().unwrap();
            let had = !tiles.is_empty();
            *tiles = Vec::new();
            had
        };
        if !had {
            return;
        }
        *self.breakpoint_histogram.write().unwrap() = None;
        self.bump_compare();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let view = ui.global::<ViewportSlintState>();
                view.set_compare_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
                view.set_compare_ready(false);
                ui.global::<PipelinesPanelState>()
                    .set_has_breakpoint_image(false);
            }
        })
        .ok();
        self.trigger_image_redraw();
    }

    /// See [`Self::compare_epoch`].
    pub(crate) fn compare_epoch(&self) -> u64 {
        self.compare_epoch.load(Ordering::SeqCst)
    }

    fn bump_compare(&self) {
        self.compare_epoch.fetch_add(1, Ordering::SeqCst);
    }

    pub fn clear_rendered_image(&self) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_display_image(slint::Image::default());
                ui.set_ghost_image(slint::Image::default());
                ui.global::<ViewportSlintState>()
                    .set_retained_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
            }
        })
        .ok();
    }

    /// Another project: nothing shown until an image is opened - the
    /// renders, the tile history, the objects, the breakpoint image and its
    /// compare are gone.
    pub fn clear_shown_image(&self) {
        self.bump_content_generation();
        *self.breakpoint_tiles.write().unwrap() = Vec::new();
        self.bump_compare();
        *self.breakpoint_histogram.write().unwrap() = None;
        self.compare.store(false, Ordering::Relaxed);
        *self.histograms.lock().expect("Poisoned") = [(Vec::new(), 1.0), (Vec::new(), 1.0)];
        if let Ok(mut active) = self.viewport_cache.active_high_res_data.write() {
            *active = None;
        }
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_display_image(slint::Image::default());
                ui.set_ghost_image(slint::Image::default());
                ui.set_object_image(slint::Image::default());
                let view = ui.global::<ViewportSlintState>();
                view.set_retained_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
                view.set_compare_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
                view.set_compare_ready(false);
                view.set_object_ready(false);
                ui.global::<HistogramState>()
                    .set_histogram_svg_path(Rc::new(VecModel::<HistogramData>::default()).into());
                ui.global::<PipelinesPanelState>()
                    .set_has_breakpoint_image(false);
                ui.global::<crate::ViewportObjectState>().set_compare(false);
            }
        })
        .ok();
    }

    /// The histogram curve and value scale (2^bits) of `side` (0 the
    /// original, 1 the breakpoint image) - shown if the panel shows that side.
    pub fn publish_histogram(&self, side: u8, curves: Vec<HistogramData>, value_scale: f32) {
        self.histograms.lock().expect("Poisoned")[side as usize] = (curves, value_scale);
        if self.histogram_side.load(Ordering::Relaxed) == side {
            self.show_histogram_curve();
        }
    }

    /// The shown side's curve and value scale to the panel.
    pub fn show_histogram_curve(&self) {
        let side = self.histogram_side.load(Ordering::Relaxed) as usize;
        let (curves, scale) = self.histograms.lock().expect("Poisoned")[side].clone();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let hist_state = ui.global::<HistogramState>();
                hist_state.set_value_scale(scale);
                hist_state.set_histogram_svg_path(Rc::new(VecModel::from(curves)).into());
            }
        })
        .ok();
    }

    /// The histogram's counts on a logarithmic scale.
    pub fn histogram_log(&self) -> bool {
        self.histogram_log.load(Ordering::Relaxed)
    }

    /// Log or linear counts - the curves are drawn anew.
    pub fn set_histogram_log(&self, log: bool) {
        if self.histogram_log.swap(log, Ordering::Relaxed) != log {
            self.bump_compare();
            self.trigger_image_redraw();
        }
    }

    /// Whose histogram the panel shows: 0 the original's, 1 the breakpoint
    /// image's.
    pub fn histogram_side(&self) -> u8 {
        self.histogram_side.load(Ordering::Relaxed)
    }

    pub fn set_histogram_side(&self, side: u8) {
        self.histogram_side.store(side.min(1), Ordering::Relaxed);
        self.show_histogram_curve();
    }

    /// The breakpoint image's histogram window, `None` before its first
    /// compare render.
    pub fn breakpoint_histogram(&self) -> Option<HistogramSettings> {
        self.breakpoint_histogram.read().unwrap().clone()
    }

    /// Sets it (`None`: auto-adjusted at the next render) and redraws.
    pub fn set_breakpoint_histogram(&self, settings: Option<HistogramSettings>) {
        *self.breakpoint_histogram.write().unwrap() = settings;
        self.bump_compare();
        if self.comparing() {
            self.trigger_image_redraw();
        }
    }

    /// The breakpoint image's value at full-image pixel (`x`, `y`), as the
    /// footer shows it: in the image's units ("128"), a label map's label
    /// ("label 3"); `None` outside it or without one.
    pub fn breakpoint_value_at(&self, x: f32, y: f32) -> Option<String> {
        let tiles = self.breakpoint_tiles.read().unwrap();
        let (px, py) = (x.floor() as i64, y.floor() as i64);
        let bp = tiles.iter().find(|t| {
            px >= t.tile_offset_x as i64
                && py >= t.tile_offset_y as i64
                && px < (t.tile_offset_x + t.tile_width) as i64
                && py < (t.tile_offset_y + t.tile_height) as i64
        })?;
        let idx =
            (py as usize - bp.tile_offset_y) * bp.tile_width + (px as usize - bp.tile_offset_x);
        let image = match self.breakpoint_view_mode() {
            BreakpointViewMode::Image => &bp.image,
            BreakpointViewMode::Segmentation => bp.segmentation.as_ref().unwrap_or(&bp.image),
            BreakpointViewMode::Instances => bp.instances.as_ref().unwrap_or(&bp.image),
        };
        match &**image {
            ImageContainer::U32(labels) => labels
                .data
                .as_slice()
                .get(idx)
                .map(|label| format!("label {label}")),
            ImageContainer::F32Gray(gray) => gray
                .as_slice()
                .get(idx)
                .map(|v| format!("{:.0}", v * 2f32.powi(bp.nr_bits as i32))),
            _ => None,
        }
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
        // On: rendered anew; off: the render is dropped at the next render.
        self.bump_compare();
        if !on {
            let ui_weak = self.ui.clone();
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    let view = ui.global::<ViewportSlintState>();
                    view.set_compare_ready(false);
                    view.set_compare_tiles(Rc::new(VecModel::<RetainedTile>::default()).into());
                }
            })
            .ok();
        }
        self.trigger_image_redraw();
    }

    /// The breakpoint images of a compare, each where it lies in the image
    /// (`(buffer, rect)`, image pixels): Slint moves them with a pan or zoom.
    pub fn sync_compare_to_slint(
        &self,
        tiles: Vec<(slint::SharedPixelBuffer<slint::Rgb8Pixel>, ImageRect)>,
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
                let tiles: Vec<RetainedTile> = tiles
                    .into_iter()
                    .map(|(buffer, rect)| RetainedTile {
                        image: slint::Image::from_rgb8(buffer),
                        img_x: rect.x,
                        img_y: rect.y,
                        img_w: rect.w,
                        img_h: rect.h,
                        ..Default::default()
                    })
                    .collect();
                view.set_compare_tiles(Rc::new(VecModel::from(tiles)).into());
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
        self.bump_compare();
        if self.comparing() {
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

        // The results' split view: the analysis' objects instead.
        let results_overlay = self.results_overlay.read().expect("Poisoned").clone();
        if let Some(overlay) = results_overlay {
            let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(buf_w, buf_h);
            draw_results_overlay(
                buffer.make_mut_slice(),
                &overlay,
                buf_w,
                buf_h,
                zoom,
                off_x + margin,
                off_y + margin,
                object_transparency as f32 / 255.0,
            );
            self.post_object_image(buffer, rect);
            return;
        }

        // Draw order: selected object > selected class > class list order
        // (first class in the list on top) > unclassified. Resolved once here.
        let z = ZOrder::new(&*project);

        // The selected object's family (Relate Objects' links, design canvas
        // "Object relations in the results", board 4): its children light
        // blue, its parents yellow, linked to it by lines; the rest dimmed.
        let family = selected_object_id.as_ref().and_then(|id| {
            let all = || objects.iter().chain(auto_objects.iter());
            let selected = all().find(|o| o.id == *id)?;
            let children: Vec<&evanalyzer_cfg::settings::object_settings::ObjectMetricSettings> =
                all()
                    .filter(|o| o.parents.values().any(|p| p == id))
                    .collect();
            let parents: Vec<&evanalyzer_cfg::settings::object_settings::ObjectMetricSettings> =
                all()
                    .filter(|o| selected.parents.values().any(|p| *p == o.id))
                    .collect();
            if children.is_empty() && parents.is_empty() {
                return None;
            }
            let centre = |o: &evanalyzer_cfg::settings::object_settings::ObjectMetricSettings| {
                (
                    (o.bbox[0] + o.bbox[2]) as f32 / 2.0 + 0.5,
                    (o.bbox[1] + o.bbox[3]) as f32 / 2.0 + 0.5,
                )
            };
            let links = children
                .iter()
                .chain(parents.iter())
                .map(|o| (centre(o), centre(selected)))
                .collect::<Vec<_>>();
            let children: Vec<_> = children.iter().map(|o| o.id.clone()).collect();
            let parents: Vec<_> = parents.iter().map(|o| o.id.clone()).collect();
            Some((children, parents, links))
        });

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

            let argb = |alpha: u8, [r, g, b]: [u8; 3]| Color::from_argb_u8(alpha, r, g, b);
            let relative = family.as_ref().map(|(children, parents, _)| {
                if children.contains(&object.id) {
                    argb(object_transparency, CHILD_RGB)
                } else if parents.contains(&object.id) {
                    argb(object_transparency, PARENT_RGB)
                } else {
                    argb(object_transparency / 3, DIMMED_RGB)
                }
            });
            let color = if is_selected {
                argb(object_transparency, SELECTED_RGB)
            } else if let Some(color) = relative {
                color
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

        // Composited straight into the buffer handed to Slint (starts out
        // transparent) - no second overlay-sized copy.
        let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(buf_w, buf_h);
        let links = family.as_ref().map_or(&[][..], |(_, _, links)| &links[..]);
        draw_objects(
            buffer.make_mut_slice(),
            keyed,
            links,
            buf_w,
            buf_h,
            zoom,
            off_x + margin,
            off_y + margin,
        );
        self.post_object_image(buffer, rect);
    }

    /// The results' split view's objects (`None`: the project's again),
    /// drawn right away - unless they are what is drawn already (e.g. only
    /// the selection changed: the crosshair shows it, not this layer).
    pub fn set_results_overlay(&self, overlay: Option<ResultsOverlay>) {
        {
            let mut current = self.results_overlay.write().expect("Poisoned");
            if current.as_deref() == overlay.as_ref() {
                return;
            }
            *current = overlay.map(Arc::new);
        }
        self.trigger_image_redraw_objects();
    }

    /// The current results overlay, if the split view shows one.
    pub fn results_overlay(&self) -> Option<Arc<ResultsOverlay>> {
        self.results_overlay.read().expect("Poisoned").clone()
    }

    /// The view now: `(zoom, offset_x, offset_y, viewport_w, viewport_h)`.
    pub fn view(&self) -> (f32, f32, f32, f32, f32) {
        let s = self.viewport_state.read().expect("Poisoned");
        (
            s.zoom,
            s.offset_x,
            s.offset_y,
            s.viewport_width,
            s.viewport_height,
        )
    }

    /// Hands a rendered object overlay (covering `rect` of the image) to
    /// the UI.
    fn post_object_image(
        &self,
        buffer: slint::SharedPixelBuffer<slint::Rgba8Pixel>,
        rect: ImageRect,
    ) {
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
        let changed = {
            let mut state = self
                .viewport_state
                .write()
                .expect("Failed to acquire write lock on viewport state");
            let changed =
                (state.zoom, state.offset_x, state.offset_y) != (zoom, offset_x, offset_y);
            state.zoom = zoom;
            state.offset_x = offset_x;
            state.offset_y = offset_y;
            changed
        };
        // The objects for this view: they may have been drawn for the one
        // before it (an image's objects can come before its first frame
        // fits or focuses the view) - left so, they'd sit off the image.
        if changed {
            self.trigger_image_redraw_objects();
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
            auto_adjust_all: false,
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
    use evanalyzer_cfg::core_types::ImageDimensionality;
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
            dimensionality: ImageDimensionality::Planar,
        })
    }

    /// The first frame fitting or focusing the view: the objects are drawn
    /// again for it - not when the view stays as it was.
    #[test]
    fn a_new_view_from_the_first_frame_redraws_the_objects() {
        let controller = make_controller();
        let object_slot = || {
            controller
                .drawing_tasks
                .object_task
                .task_request
                .0
                .lock()
                .unwrap()
                .take()
                .is_some()
        };
        controller.sync_zoom_to_slint(0.25, 10.0, 20.0);
        assert!(object_slot(), "another view: drawn again");
        controller.sync_zoom_to_slint(0.25, 10.0, 20.0);
        assert!(!object_slot(), "the same view: not");
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

        let stored = controller.breakpoint_tiles.read().unwrap();
        assert_eq!(stored.len(), 1);
        let data = stored.first().expect("breakpoint channel must be set");
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

        let stored = controller.breakpoint_tiles.read().unwrap();
        assert_eq!(stored.len(), 1, "replaced, not added");
        let data = stored.first().unwrap();
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
