use crate::{
    HistogramData, UiState,
    editor::{
        histogram_controller::HistogramController,
        viewport_cache::{ReadContext, ViewportCache, to_z_projection},
        viewport_controller::{DrawingTaskContainer, ViewportController},
        viewport_task::{DrawingTask, TaskDispatch},
    },
};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::HistogramSettings;
use evanalyzer_core::{ImageChannel, ImageContainer};
use log::{debug, info, warn};
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

pub struct ViewportWorker {
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) histogram_controller: Arc<HistogramController>,
    pub(crate) viewport_cache: Arc<ViewportCache>,
}

pub(crate) struct ChannelCtx<'a> {
    pub(crate) image_data: &'a [f32],
    pub(crate) histogram: HistogramSettings,
    pub(crate) color: [f32; 3],
    pub(crate) r_factor: f32,
    pub(crate) g_factor: f32,
    pub(crate) b_factor: f32,
    pub(crate) offset: f32,
    pub(crate) h_mult: f32,
    pub(crate) channel_idx: i32,
}

const NUM_BINS: usize = 512;

impl ViewportWorker {
    pub fn new(
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        histogram_controller: Arc<HistogramController>,
        viewport_cache: Arc<ViewportCache>,
    ) -> Self {
        Self {
            app_state,
            viewport_controller,
            histogram_controller,
            viewport_cache,
        }
    }

    pub(crate) fn start_worker(self: &Arc<Self>) {
        let configs = [
            ("HighResWorker", TaskDispatch::HighRes),
            ("LowResWorker", TaskDispatch::LowRes),
            ("ObjectWorker", TaskDispatch::Objects),
        ];

        for (name, dispatch) in configs {
            let self_handle = Arc::clone(self);
            std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    crate::helper::worker_supervisor::run_supervised(name, || {
                        self_handle.run_worker_loop(dispatch)
                    })
                })
                .expect("Failed to spawn viewport worker thread");
        }
    }

    fn run_worker_loop(self: &Arc<Self>, scope: TaskDispatch) -> ! {
        let ui_busy = Arc::new(AtomicBool::new(false));

        let mut buffer_pool = [
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
        ];
        let mut pool_idx = 0;

        // Viewport-sized buffer the native tile is composited into before being
        // handed to Slint.  See STEP 7b for why this screen-space step is needed.
        // Double-buffered for the same reason `buffer_pool` above is: a
        // `SharedPixelBuffer` is a ref-counted, copy-on-write `SharedVector`
        // under the hood, and the buffer sent to Slint last frame
        // (`pixel_buffer_to_send`) may still be held by the UI thread when
        // this thread starts the next frame - `make_mut_slice()` on a still-
        // shared buffer forces a full alloc+copy of the whole (multi-MB,
        // viewport-sized) buffer before returning, silently adding that cost
        // to every single frame. Alternating between two buffers makes it
        // very likely the "other" one has already been released by the time
        // it's reused.
        let mut screen_buffer_pool = [
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
        ];
        let mut screen_pool_idx = 0;

        let (version_tracker, drawing_task_container, is_low_res) = match scope {
            TaskDispatch::LowRes => (
                &self
                    .viewport_controller
                    .drawing_tasks
                    .low_res_task
                    .task_count,
                &self.viewport_controller.drawing_tasks.low_res_task,
                true,
            ),
            TaskDispatch::HighRes => (
                &self
                    .viewport_controller
                    .drawing_tasks
                    .high_res_task
                    .task_count,
                &self.viewport_controller.drawing_tasks.high_res_task,
                false,
            ),
            _ => (
                &self
                    .viewport_controller
                    .drawing_tasks
                    .object_task
                    .task_count,
                &self.viewport_controller.drawing_tasks.object_task,
                false,
            ),
        };

        loop {
            let mut task = wait_for_task(&drawing_task_container);
            let task_start = Instant::now();

            // --- object scope: simple path, no image processing ---
            if scope == TaskDispatch::Objects {
                self.viewport_controller.sync_objects_to_slint_viewport();
                continue;
            }

            // ----------------------------------------------------------------
            // STEP 1: Extract all needed data from project - drop lock immediately
            // This prevents holding the read lock during slow disk I/O or writes
            // ----------------------------------------------------------------
            let (
                has_image,
                series,
                visible_channels,
                z_stack,
                t_stack,
                hist_settings,
                selected_channel,
            ) = {
                let project = self.app_state.get_project();
                (
                    project.tmp_settings.current_image.is_some(),
                    project.get_selected_series_idx(),
                    project.get_image_channel_visibilities_vec(),
                    project.get_z_stack().cloned().unwrap_or_default(),
                    project.get_t_stack().cloned().unwrap_or_default(),
                    project.get_image_channel_histograms(),
                    project.get_selected_image_channel_idx(),
                )
            };

            if !has_image {
                //debug!("No current image");
                continue;
            }

            // ----------------------------------------------------------------
            // STEP 2: Extract viewport state - separate lock, acquired after
            // project lock is already dropped
            // ----------------------------------------------------------------
            let viewport_state = self
                .viewport_controller
                .viewport_state
                .read()
                .unwrap()
                .clone();

            let current_version = version_tracker.load(Ordering::SeqCst);

            // Whether the user has the breakpoint-image toggle active.
            let in_breakpoint_mode = self
                .viewport_controller
                .show_breakpoint
                .load(Ordering::Relaxed);
            // Both tiers skip disk I/O and render the in-memory breakpoint buffer
            // when in breakpoint mode - the LowRes ghost renders it in grayscale
            // (below) rather than being left showing the unrelated regular image.
            let show_bp = in_breakpoint_mode;

            // ----------------------------------------------------------------
            // STEP 3: Disk I/O (skipped in breakpoint mode)
            // ----------------------------------------------------------------
            // Whether the buffer this frame renders is a U32 label map
            // (Segmentation/Instances) rather than intensity data - set
            // below when `show_bp` picks one. Declared outside the `if` so
            // STEP 4 can branch on it after `read_result` is computed.
            let mut is_label_render = false;
            let read_result = if show_bp {
                match &*self.viewport_controller.breakpoint_channel.read().unwrap() {
                    Some(bp) => {
                        // Pick the buffer matching the user's selected view
                        // mode, falling back to the intensity image if the
                        // pipeline hadn't produced that buffer yet at this
                        // breakpoint step (e.g. Segmentation selected but the
                        // breakpoint is before `Threshold`).
                        let source_image = match self.viewport_controller.breakpoint_view_mode() {
                            crate::editor::viewport_controller::BreakpointViewMode::Image => {
                                bp.image.clone()
                            }
                            crate::editor::viewport_controller::BreakpointViewMode::Segmentation => {
                                bp.segmentation.clone().unwrap_or_else(|| bp.image.clone())
                            }
                            crate::editor::viewport_controller::BreakpointViewMode::Instances => {
                                bp.instances.clone().unwrap_or_else(|| bp.image.clone())
                            }
                        };
                        is_label_render = matches!(&*source_image, ImageContainer::U32(_));
                        let is_rgb = matches!(&*source_image, ImageContainer::F32Rgb(_));
                        let channel = ImageChannel {
                            image: source_image,
                            color: [1.0, 1.0, 1.0],
                            // Must match the channel the breakpointed pipeline
                            // actually started from - the render loop below
                            // looks up histogram/LUT settings by `c_stack`,
                            // and hardcoding 0 here made the breakpoint image
                            // render using an unrelated channel's histogram
                            // range whenever the pipeline started from any
                            // other channel (usually all-black, since the
                            // real data then falls outside that channel's
                            // configured min/max window). Irrelevant for
                            // label rendering (is_label_render bypasses the
                            // histogram lookup entirely), but harmless there.
                            c_stack: bp.channel_idx.unwrap_or(0),
                            name: "Breakpoint".to_string(),
                            is_rgb,
                            is_visible: true,
                        };
                        let prepared = ReadContext {
                            zoomed_w: bp.tile_width as f32 * viewport_state.zoom,
                            zoomed_h: bp.tile_height as f32 * viewport_state.zoom,
                            zoom: viewport_state.zoom,
                            draw_x: bp.tile_offset_x as f32 * viewport_state.zoom
                                + viewport_state.offset_x,
                            draw_y: bp.tile_offset_y as f32 * viewport_state.zoom
                                + viewport_state.offset_y,
                            offset_x: viewport_state.offset_x,
                            offset_y: viewport_state.offset_y,
                            read_off_x: bp.tile_offset_x,
                            read_off_y: bp.tile_offset_y,
                            res_idx: 0,
                            image_w: bp.tile_width,
                            image_h: bp.tile_height,
                            bit_depth: bp.nr_bits,
                            _nr_color_channels: if is_rgb { 3 } else { 1 },
                            viewport_width: viewport_state.viewport_width,
                            viewport_height: viewport_state.viewport_height,
                            full_image_w: bp.tile_offset_x + bp.tile_width,
                            full_image_h: bp.tile_offset_y + bp.tile_height,
                        };
                        Ok((Arc::new(vec![channel]), prepared))
                    }
                    None => Err(InternalErrors::ImageReadError(
                        "No breakpoint image captured yet".into(),
                    )),
                }
            } else {
                self.viewport_cache.read_image_tile_combined(
                    series,
                    to_z_projection(z_stack.z_projection.clone()),
                    z_stack.z_range.clone(),
                    t_stack.t_stack.clone(),
                    task.fit_to_screen,
                    task.is_new_image,
                    is_low_res,
                    &viewport_state,
                )
            };
            let read_duration = task_start.elapsed();
            let step4_7_start = Instant::now();

            // Cancel if a newer request came in during the slow disk read
            if version_tracker.load(Ordering::SeqCst) > current_version {
                continue;
            }

            // ----------------------------------------------------------------
            // STEP 4: Process loaded image
            // ----------------------------------------------------------------
            let mut pixel_buffer_to_send = None;
            let mut svg_hists_to_send = Vec::new();
            let mut render_info = None;

            if let Ok((render_src, prepared)) = read_result {
                if !is_low_res {
                    if let Ok(mut active) = self.viewport_cache.active_high_res_data.write() {
                        *active = Some((render_src.clone(), prepared.clone()));
                    }
                }

                // Resize buffer from pool if needed
                pool_idx = (pool_idx + 1) % buffer_pool.len();
                if buffer_pool[pool_idx].width() != prepared.image_w as u32
                    || buffer_pool[pool_idx].height() != prepared.image_h as u32
                {
                    buffer_pool[pool_idx] =
                        SharedPixelBuffer::new(prepared.image_w as u32, prepared.image_h as u32);
                }

                let master_slice = buffer_pool[pool_idx].make_mut_slice();
                master_slice.fill(Rgb8Pixel { r: 0, g: 0, b: 0 });

                // ------------------------------------------------------------
                // STEP 5-7: Render pixels + build histograms - pure CPU, no locks.
                //
                // Segmentation/Instances breakpoint views bypass all of this:
                // label IDs aren't intensities, so a histogram min/max window
                // is meaningless for them - `render_labels_to_rgb8` writes
                // `master_slice` directly instead.
                // ------------------------------------------------------------
                let mut channel_contexts: Vec<ChannelCtx> = Vec::new();
                let mut is_rgb = false;

                let all_hists: Vec<Vec<f32>> = if is_label_render {
                    if let Some(channel) = render_src.first() {
                        if let ImageContainer::U32(img) = &*channel.image {
                            render_labels_to_rgb8(img.data.as_slice(), master_slice);
                        }
                    }
                    Vec::new()
                } else {
                    // STEP 5: Auto-adjust - write lock acquired AFTER read lock
                    // dropped. Safe because we dropped the project read lock in
                    // STEP 1. Skipped in breakpoint mode to preserve the
                    // original histogram.
                    if !show_bp && (task.auto_adjust_if_not_set || task.auto_adjust_selected) {
                        for channel in render_src.iter() {
                            let idx = channel.c_stack;
                            if let Some(ch) = hist_settings.get(&idx) {
                                if (!ch.is_some() && task.auto_adjust_if_not_set)
                                    || (task.auto_adjust_selected && idx == selected_channel)
                                {
                                    let (min, max, min_range, max_range) =
                                        apply_auto_adjust(&channel.image, channel.is_rgb);
                                    debug!(
                                        "Auto-adjusting channel {} min={} max={} range=({},{})",
                                        idx, min, max, min_range, max_range
                                    );
                                    self.app_state
                                        .get_project_write()
                                        .set_image_histogram_settings_for_channel(
                                            idx, min, max, min_range, max_range,
                                        );
                                }
                            }
                        }
                        self.histogram_controller.sync_histogram_settings_to_slint();
                    } else if task.is_new_image || task.is_new_series {
                        self.histogram_controller.sync_histogram_settings_to_slint();
                    }

                    // STEP 6: Re-read histogram settings after potential write.
                    // Fresh read lock - safe because write lock was released above.
                    let hist_settings_fresh = self
                        .app_state
                        .get_project()
                        .get_image_channel_histograms()
                        .clone();

                    // Build channel contexts for rendering
                    channel_contexts = Vec::with_capacity(render_src.len());

                    for channel in render_src.iter() {
                        let idx = channel.c_stack;
                        if let Some(Some(histogram)) = hist_settings_fresh.get(&idx) {
                            let data_slice = match &*channel.image {
                                ImageContainer::F32Gray(img) => {
                                    is_rgb = false;
                                    Some(img.as_slice())
                                }
                                ImageContainer::F32Rgb(img) => {
                                    is_rgb = true;
                                    Some(img.as_slice())
                                }
                                _ => None,
                            };

                            if let Some(slice) = data_slice {
                                let inv_range = 1.0 / (histogram.max - histogram.min).max(0.001);
                                // In breakpoint mode the LowRes ghost is rendered in
                                // grayscale so it doesn't flash color during pan/zoom.
                                let color = if is_low_res && in_breakpoint_mode {
                                    [1.0f32, 1.0, 1.0]
                                } else {
                                    channel.color
                                };
                                channel_contexts.push(ChannelCtx {
                                    image_data: slice,
                                    histogram: (*histogram).clone(),
                                    color,
                                    r_factor: inv_range * color[0] * 255.0,
                                    g_factor: inv_range * color[1] * 255.0,
                                    b_factor: inv_range * color[2] * 255.0,
                                    offset: -histogram.min,
                                    h_mult: (NUM_BINS as f32 - 1.0)
                                        / (histogram.max_limit - histogram.min_limit)
                                            .max(f32::EPSILON),
                                    channel_idx: idx,
                                });
                            }
                        }
                    }

                    // STEP 7: render
                    prepare_image_channels_for_slint(
                        &channel_contexts,
                        master_slice,
                        NUM_BINS,
                        !is_low_res,
                        &visible_channels,
                        is_rgb,
                    )
                };
                let step4_7_duration = step4_7_start.elapsed();

                // ------------------------------------------------------------
                // STEP 7b: Composite into a viewport-sized, screen-space buffer
                // ------------------------------------------------------------
                // Windows-only workaround. The native tile buffer (image_w x
                // image_h) would otherwise be handed to Slint as a single
                // Image element positioned at draw_x and stretched to
                // zoomed_w/zoomed_h.  When zoomed/panned that element's
                // origin sits far off-screen (draw_x can be thousands of px
                // negative) and it is several thousand px wide.  The Slint
                // SOFTWARE renderer (Windows build) stores scene coordinates
                // as i16 and samples scaled images with an 8-bit fixed-point
                // step; the per-step rounding error gets multiplied by the
                // large off-screen offset, shifting the image by several
                // pixels - and by a different amount at every zoom level.
                // The GPU/Skia renderer (Linux/macOS) does not have this bug,
                // so it skips the workaround entirely below: composing a
                // *viewport-sized* buffer here means its cost tracks window
                // size, not the size of the image actually being displayed
                // (a small zoomed-out image in a huge window still pays for
                // the whole window) - both the CPU resample and the GPU
                // texture upload that follows scale with `vp_w * vp_h`.
                // Skia can scale/position the much smaller native tile
                // directly on the GPU, which is what `viewport.slint`'s
                // Image layers (image-fit: fill + explicit width/height)
                // were already designed to do - so on that path there is no
                // reason to pre-composite at all.
                if cfg!(target_os = "windows") {
                    let vp_w = prepared.viewport_width.max(1.0) as usize;
                    let vp_h = prepared.viewport_height.max(1.0) as usize;
                    screen_pool_idx = (screen_pool_idx + 1) % screen_buffer_pool.len();
                    if screen_buffer_pool[screen_pool_idx].width() as usize != vp_w
                        || screen_buffer_pool[screen_pool_idx].height() as usize != vp_h
                    {
                        screen_buffer_pool[screen_pool_idx] =
                            SharedPixelBuffer::new(vp_w as u32, vp_h as u32);
                    }
                    let composite_start = Instant::now();
                    let slice_start = Instant::now();
                    let native = buffer_pool[pool_idx].as_slice();
                    let screen = screen_buffer_pool[screen_pool_idx].make_mut_slice();
                    let slice_duration = slice_start.elapsed();
                    {
                        let img_w = prepared.image_w;
                        let img_h = prepared.image_h;
                        let inv_scale_x =
                            prepared.image_w as f32 / prepared.zoomed_w.max(f32::EPSILON);
                        let inv_scale_y =
                            prepared.image_h as f32 / prepared.zoomed_h.max(f32::EPSILON);
                        let draw_x = prepared.draw_x;
                        let draw_y = prepared.draw_y;
                        let black = Rgb8Pixel { r: 0, g: 0, b: 0 };
                        // Each output row is independent (nearest-neighbor
                        // resample, no cross-row state), so this is split
                        // across rows and run in parallel, chunked into a
                        // handful of multi-row groups per thread rather than
                        // one task per row (measured: one-row chunks added
                        // enough Rayon per-task scheduling overhead to cost
                        // 11-25ms/frame on a real ~3000x2000 viewport, on top
                        // of the actual memory-bound copy).
                        use rayon::prelude::*;
                        let rows_per_chunk = (vp_h / (rayon::current_num_threads() * 4)).max(1);
                        screen
                            .par_chunks_mut(vp_w * rows_per_chunk)
                            .enumerate()
                            .for_each(|(chunk_idx, rows)| {
                                let base_sy = chunk_idx * rows_per_chunk;
                                for (row_offset, row) in rows.chunks_mut(vp_w).enumerate() {
                                    let sy = base_sy + row_offset;
                                    let ty = (sy as f32 - draw_y) * inv_scale_y;
                                    if ty < 0.0 || ty >= img_h as f32 {
                                        row.fill(black);
                                        continue;
                                    }
                                    let ty_i = ty as usize * img_w;
                                    for (sx, out) in row.iter_mut().enumerate() {
                                        let tx = (sx as f32 - draw_x) * inv_scale_x;
                                        *out = if tx >= 0.0 && tx < img_w as f32 {
                                            native[ty_i + tx as usize]
                                        } else {
                                            black
                                        };
                                    }
                                }
                            });
                    }
                    let composite_duration = composite_start.elapsed();
                    info!(
                        "Viewport frame ({}): read {:?}, render(4-7) {:?} [{} channels], composite {:?} [make_mut_slice {:?}, resample {:?}] ({}x{} native -> {}x{} viewport, {} threads), total so far {:?}",
                        if is_low_res { "low-res" } else { "high-res" },
                        read_duration,
                        step4_7_duration,
                        channel_contexts.len(),
                        composite_duration,
                        slice_duration,
                        composite_duration.saturating_sub(slice_duration),
                        prepared.image_w,
                        prepared.image_h,
                        vp_w,
                        vp_h,
                        rayon::current_num_threads(),
                        task_start.elapsed()
                    );

                    // Display geometry is now screen-space: full viewport at
                    // (0,0). The logical transform (zoom/offset/full_image)
                    // in `prepared` is left untouched so sync_zoom, the
                    // navigator and the pixel-value HUD (which maps via
                    // active_high_res_data) keep working.
                    let mut display = prepared.clone();
                    display.draw_x = 0.0;
                    display.draw_y = 0.0;
                    display.zoomed_w = vp_w as f32;
                    display.zoomed_h = vp_h as f32;

                    pixel_buffer_to_send = Some(screen_buffer_pool[screen_pool_idx].clone());
                    render_info = Some(display);
                } else {
                    info!(
                        "Viewport frame ({}): read {:?}, render(4-7) {:?} [{} channels], no composite (direct GPU scale, {} threads), total so far {:?}",
                        if is_low_res { "low-res" } else { "high-res" },
                        read_duration,
                        step4_7_duration,
                        channel_contexts.len(),
                        rayon::current_num_threads(),
                        task_start.elapsed()
                    );

                    pixel_buffer_to_send = Some(buffer_pool[pool_idx].clone());
                    render_info = Some(prepared.clone());
                }

                if !is_low_res {
                    svg_hists_to_send = histogram_to_svg_fast(
                        &all_hists
                            .into_iter()
                            .zip(channel_contexts.iter().map(|c| c.color))
                            .collect(),
                        NUM_BINS,
                    );
                }
            } else if let Err(e) = read_result {
                warn!("Error reading image tile: {:?}", e);
            }

            // ----------------------------------------------------------------
            // STEP 8: Dispatch to UI thread
            // ----------------------------------------------------------------
            let busy = ui_busy.clone();
            busy.store(true, Ordering::SeqCst);

            if let (Some(pb), Some(info)) = (pixel_buffer_to_send, render_info) {
                self.viewport_controller.sync_viewport_state_to_slint(
                    pb,
                    svg_hists_to_send,
                    info.draw_x,
                    info.draw_y,
                    info.zoomed_w,
                    info.zoomed_h,
                    is_low_res,
                );

                if task.fit_to_screen {
                    self.viewport_controller.sync_zoom_to_slint(
                        info.zoom,
                        info.offset_x,
                        info.offset_y,
                    );
                }

                if is_low_res {
                    self.viewport_controller.sync_high_res_ready_to_slint(false);
                    self.viewport_controller.sync_navigator_to_slint(
                        info.full_image_w as i64,
                        info.full_image_h as i64,
                        info.viewport_width,
                        info.viewport_height,
                        info.offset_x,
                        info.offset_y,
                    );
                }
            }

            busy.store(false, Ordering::SeqCst);
            task.reset_job();
        }
    }
}

/// Waits for a drawing task to become available, blocking until one is posted.
fn wait_for_task(pair: &Arc<DrawingTaskContainer>) -> DrawingTask {
    let (lock, cvar) = &*pair.task_request;
    let mut task_slot = lock.lock().unwrap();
    while task_slot.is_none() {
        task_slot = cvar.wait(task_slot).unwrap();
    }
    task_slot.take().unwrap()
}

/// Applies auto-adjustment to an image using partial sorting (O(N) average).
///
/// Samples every 10th pixel and finds the 0.5th and 99.5th percentile values
/// to use as the display range, clipping extreme outliers.
///
/// # Returns
/// `(min, max, min_limit, max_limit)` - display range and histogram limits.
pub fn apply_auto_adjust(img: &ImageContainer, is_rgb: bool) -> (f32, f32, f32, f32) {
    if is_rgb {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let mut min = 0.0;
    let mut max = 1.0;

    if let ImageContainer::F32Gray(image) = img {
        let pixels = image.as_slice();
        if !pixels.is_empty() {
            let mut sample: Vec<f32> = pixels.iter().step_by(10).cloned().collect();
            let len = sample.len();

            let low_idx = (len as f32 * 0.005) as usize;
            let high_idx = ((len as f32 * 0.995) as usize).min(len - 1);

            sample.select_nth_unstable_by(low_idx, |a, b| a.total_cmp(b));
            min = sample[low_idx];

            sample.select_nth_unstable_by(high_idx, |a, b| a.total_cmp(b));
            max = sample[high_idx];
        }
    }

    (min, max, (min - 0.01).max(0.0), (max + 0.01).min(1.0))
}

/// Renders a `u32` label buffer (segmentation classes or instance IDs)
/// directly to RGB8, bypassing the histogram/LUT pipeline entirely - label
/// IDs aren't intensities, so a min/max window doesn't apply to them. `0`
/// (background) renders black; every other ID gets a deterministic,
/// visually-distinct color, so the same ID always renders the same color
/// across redraws (this fires on every pan/zoom).
fn render_labels_to_rgb8(labels: &[u32], dest_pixels: &mut [Rgb8Pixel]) {
    use rayon::prelude::*;
    let n = labels.len().min(dest_pixels.len());
    dest_pixels[..n]
        .par_iter_mut()
        .zip(labels[..n].par_iter())
        .for_each(|(px, &id)| {
            *px = label_to_rgb8(id);
        });
}

/// Maps a label ID to a color via golden-ratio hue stepping: consecutive IDs
/// land far apart on the hue wheel, so adjacent objects - which often get
/// consecutive IDs from connected-components/instance labeling - stay
/// visually distinguishable instead of blending into near-identical shades.
fn label_to_rgb8(id: u32) -> Rgb8Pixel {
    if id == 0 {
        return Rgb8Pixel { r: 0, g: 0, b: 0 };
    }
    const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_034;
    let hue = (id as f32 * GOLDEN_RATIO_CONJUGATE).fract();
    let (r, g, b) = hsv_to_rgb8(hue, 0.65, 0.95);
    Rgb8Pixel { r, g, b }
}

/// Minimal HSV -> RGB8 conversion (`h`, `s`, `v` in `0.0..=1.0`).
fn hsv_to_rgb8(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i64).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// Converts f32 image channels to RGB8 pixels, applying histogram brightness settings.
///
/// Processes pixels in parallel chunks using rayon. Also computes per-channel
/// histograms if `create_histogram` is true.
///
/// # Returns
/// Normalised per-channel histograms, or an empty vec if `create_histogram` is false.
pub(crate) fn prepare_image_channels_for_slint(
    channels: &[ChannelCtx],
    dest_pixels: &mut [Rgb8Pixel],
    num_bins: usize,
    create_histogram: bool,
    visible_channels: &Vec<i32>,
    _is_rgb: bool,
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;

    let expected_len = dest_pixels.len();
    // Resolved once, outside the pixel loop: which `channels` entries are
    // actually visible. `visible_channels.contains(...)` used to be called
    // per pixel per channel inside the hot loop below (O(pixels × channels²)
    // for a linear-scan `.contains()` over up to `channels²` comparisons);
    // this reduces that to one O(channels) pass up front, and the per-pixel
    // loop then only visits the (usually much smaller) visible subset
    // directly instead of walking every channel and skipping hidden ones.
    let visible_indices: Vec<usize> = channels
        .iter()
        .enumerate()
        .filter(|(_, ctx)| visible_channels.contains(&ctx.channel_idx))
        .map(|(i, _)| i)
        .collect();

    for &i in &visible_indices {
        let ctx = &channels[i];
        if ctx.image_data.len() != expected_len {
            panic!(
                "Memory alignment error: channel {} has {} pixels but destination expects {}. \
                Check tile clipping logic at image edges.",
                i,
                ctx.image_data.len(),
                expected_len
            );
        }
    }

    let chunk_size = 1024 * 4;
    let n_channels = channels.len();

    let raw_hists: Vec<Vec<u32>> = dest_pixels
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_hists = vec![vec![0u32; num_bins]; n_channels];
            let start_idx = chunk_idx * chunk_size;

            for (p_idx, pixel) in chunk.iter_mut().enumerate() {
                let global_idx = start_idx + p_idx;

                let mut r_acc = 0.0f32;
                let mut g_acc = 0.0f32;
                let mut b_acc = 0.0f32;

                for &c_idx in &visible_indices {
                    let ctx = &channels[c_idx];
                    let p = ctx.image_data[global_idx];

                    if create_histogram
                        && p >= ctx.histogram.min_limit
                        && p <= ctx.histogram.max_limit
                    {
                        let bin_idx = ((p - ctx.histogram.min_limit) * ctx.h_mult) as usize;
                        if bin_idx < num_bins {
                            local_hists[c_idx][bin_idx] += 1;
                        }
                    }

                    let val = (p + ctx.offset).max(0.0);
                    r_acc += val * ctx.r_factor;
                    g_acc += val * ctx.g_factor;
                    b_acc += val * ctx.b_factor;
                }

                pixel.r = r_acc.min(255.0) as u8;
                pixel.g = g_acc.min(255.0) as u8;
                pixel.b = b_acc.min(255.0) as u8;
            }
            local_hists
        })
        .reduce(
            || vec![vec![0u32; num_bins]; n_channels],
            |mut a, b| {
                for (ah, bh) in a.iter_mut().zip(b.iter()) {
                    for (av, bv) in ah.iter_mut().zip(bh.iter()) {
                        *av += bv;
                    }
                }
                a
            },
        );

    if !create_histogram {
        return vec![];
    }

    let mut final_hists = vec![vec![0.0f32; num_bins]; n_channels];
    for (c_idx, hist) in raw_hists.into_iter().enumerate() {
        for (bin_idx, count) in hist.into_iter().enumerate() {
            final_hists[c_idx][bin_idx] = count as f32;
        }
    }

    // Normalise each channel histogram to [0, 1]
    for hist in final_hists.iter_mut() {
        let max_v = hist.iter().cloned().fold(0.0f32, f32::max);
        if max_v > 0.0 {
            for v in hist.iter_mut() {
                *v /= max_v;
            }
        }
    }

    final_hists
}

/// Converts normalised histogram data into SVG path strings for Slint rendering.
///
/// Each histogram is rendered as a filled path from bottom-left to bottom-right,
/// with the curve following the histogram values scaled to a 100×100 viewBox.
pub(crate) fn histogram_to_svg_fast(
    histos: &Vec<(Vec<f32>, [f32; 3])>,
    bins: usize,
) -> Vec<HistogramData> {
    use std::fmt::Write;

    histos
        .iter()
        .filter(|(data, _)| !data.is_empty())
        .map(|(data, color)| {
            let mut path_data = String::with_capacity(data.len() * 20);
            write!(path_data, "M 0 100").unwrap();

            for (i, &val) in data.iter().enumerate() {
                let x = if bins > 1 {
                    (i as f32 / (bins - 1) as f32) * 100.0
                } else {
                    0.0
                };
                let y = (1.0 - val.clamp(0.0, 1.0)) * 100.0;

                if i == 0 {
                    write!(path_data, " L {:.2} {:.2}", x, y).unwrap();
                } else {
                    write!(path_data, " {:.2} {:.2}", x, y).unwrap();
                }
            }

            write!(path_data, " 100 100 Z").unwrap();

            HistogramData {
                color: slint::Color::from_rgb_f32(color[0], color[1], color[2]),
                path: path_data.into(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_core::{ImagePlane, ManagedImage};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::{Image, ImageSize};

    // -- hsv_to_rgb8 ------------------------------------------------------------

    #[test]
    fn hsv_to_rgb8_pure_colors_at_each_sixth_of_the_hue_wheel() {
        assert_eq!(hsv_to_rgb8(0.0, 1.0, 1.0), (255, 0, 0)); // red
        assert_eq!(hsv_to_rgb8(1.0 / 3.0, 1.0, 1.0), (0, 255, 0)); // green
        assert_eq!(hsv_to_rgb8(2.0 / 3.0, 1.0, 1.0), (0, 0, 255)); // blue
    }

    #[test]
    fn hsv_to_rgb8_zero_saturation_is_a_gray_shade() {
        let (r, g, b) = hsv_to_rgb8(0.5, 0.0, 0.5);
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert_eq!(r, 128);
    }

    #[test]
    fn hsv_to_rgb8_zero_value_is_black_regardless_of_hue() {
        assert_eq!(hsv_to_rgb8(0.3, 1.0, 0.0), (0, 0, 0));
    }

    // -- label_to_rgb8 / render_labels_to_rgb8 -----------------------------------

    #[test]
    fn label_to_rgb8_zero_is_always_black() {
        assert_eq!(label_to_rgb8(0), Rgb8Pixel { r: 0, g: 0, b: 0 });
    }

    #[test]
    fn label_to_rgb8_is_deterministic_for_the_same_id() {
        assert_eq!(label_to_rgb8(42), label_to_rgb8(42));
    }

    #[test]
    fn label_to_rgb8_distinct_ids_usually_render_distinct_colors() {
        // Not a strict guarantee for every possible pair, but consecutive
        // small IDs (the common case - connected-components output) must not
        // collide, otherwise adjacent objects would be indistinguishable.
        assert_ne!(label_to_rgb8(1), label_to_rgb8(2));
        assert_ne!(label_to_rgb8(2), label_to_rgb8(3));
    }

    #[test]
    fn render_labels_to_rgb8_maps_every_pixel_through_label_to_rgb8() {
        let labels = vec![0u32, 5, 5, 0];
        let mut dest = vec![Rgb8Pixel { r: 9, g: 9, b: 9 }; 4];

        render_labels_to_rgb8(&labels, &mut dest);

        assert_eq!(dest[0], Rgb8Pixel { r: 0, g: 0, b: 0 });
        assert_eq!(dest[1], label_to_rgb8(5));
        assert_eq!(dest[2], label_to_rgb8(5));
        assert_eq!(dest[3], Rgb8Pixel { r: 0, g: 0, b: 0 });
    }

    #[test]
    fn render_labels_to_rgb8_stops_at_the_shorter_of_the_two_slices() {
        let labels = vec![7u32, 7, 7];
        let mut dest = vec![Rgb8Pixel { r: 9, g: 9, b: 9 }; 1];

        // Must not panic when the buffers have mismatched lengths.
        render_labels_to_rgb8(&labels, &mut dest);

        assert_eq!(dest[0], label_to_rgb8(7));
    }

    // -- apply_auto_adjust --------------------------------------------------------

    fn gray_container(pixels: Vec<f32>) -> ImageContainer {
        let len = pixels.len();
        let image = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: len,
                height: 1,
            },
            pixels,
            CpuAllocator,
        )
        .unwrap();
        ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
        })
    }

    #[test]
    fn apply_auto_adjust_rgb_images_always_use_the_full_fixed_range() {
        let img = gray_container(vec![0.1, 0.9]);
        assert_eq!(apply_auto_adjust(&img, true), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn apply_auto_adjust_empty_gray_image_falls_back_to_the_default_range() {
        let img = gray_container(vec![]);
        assert_eq!(apply_auto_adjust(&img, false), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn apply_auto_adjust_uniform_image_collapses_min_and_max_to_the_same_value() {
        let img = gray_container(vec![0.5; 200]);
        let (min, max, min_limit, max_limit) = apply_auto_adjust(&img, false);
        assert_eq!(min, 0.5);
        assert_eq!(max, 0.5);
        assert_eq!(min_limit, 0.49);
        assert_eq!(max_limit, 0.51);
    }

    #[test]
    fn apply_auto_adjust_clips_extreme_outliers_out_of_the_display_range() {
        // 5000 pixels -> 500-element sample (every 10th pixel), giving
        // low_idx=2 / high_idx=497 (see `apply_auto_adjust`'s doc comment).
        // Only pixels at indices that are multiples of 10 land in the
        // sample, so outliers must be placed there to actually affect the
        // computed range - 2 outliers on each side (fewer than low_idx=2's
        // position) get skipped by the percentile window entirely.
        let mut pixels = vec![0.5f32; 5000];
        pixels[0] = -100.0;
        pixels[10] = -100.0;
        pixels[4980] = 100.0;
        pixels[4990] = 100.0;
        let img = gray_container(pixels);

        let (min, max, min_limit, max_limit) = apply_auto_adjust(&img, false);
        assert_eq!(
            min, 0.5,
            "the 2 low outliers must be clipped out of the range"
        );
        assert_eq!(
            max, 0.5,
            "the 2 high outliers must be clipped out of the range"
        );
        assert_eq!(min_limit, 0.49);
        assert_eq!(max_limit, 0.51);
    }

    #[test]
    fn apply_auto_adjust_limits_are_clamped_to_the_0_1_range() {
        let img = gray_container(vec![0.0; 100]);
        let (_, _, min_limit, _) = apply_auto_adjust(&img, false);
        assert_eq!(
            min_limit, 0.0,
            "min - 0.01 must clamp at 0.0, not go negative"
        );

        let img = gray_container(vec![1.0; 100]);
        let (_, _, _, max_limit) = apply_auto_adjust(&img, false);
        assert_eq!(max_limit, 1.0, "max + 0.01 must clamp at 1.0");
    }

    // -- prepare_image_channels_for_slint -----------------------------------------

    fn ctx_for(data: &[f32]) -> ChannelCtx<'_> {
        ChannelCtx {
            image_data: data,
            histogram: HistogramSettings {
                min: 0.0,
                max: 1.0,
                min_limit: 0.0,
                max_limit: 1.0,
            },
            color: [1.0, 0.0, 0.0],
            r_factor: 255.0,
            g_factor: 0.0,
            b_factor: 0.0,
            offset: 0.0,
            h_mult: 4.0, // num_bins (4) / (max_limit - min_limit)
            channel_idx: 0,
        }
    }

    #[test]
    fn prepare_image_channels_for_slint_maps_pixels_through_the_color_factors() {
        let data = [0.0f32, 0.9];
        let ctx = ctx_for(&data);
        let mut dest = vec![Rgb8Pixel { r: 9, g: 9, b: 9 }; 2];

        prepare_image_channels_for_slint(&[ctx], &mut dest, 4, false, &vec![0], false);

        assert_eq!(dest[0], Rgb8Pixel { r: 0, g: 0, b: 0 });
        // (0.9 + offset(0.0)) * r_factor(255.0) = 229.5 -> truncated to 229
        assert_eq!(dest[1], Rgb8Pixel { r: 229, g: 0, b: 0 });
    }

    #[test]
    fn prepare_image_channels_for_slint_without_histogram_flag_returns_no_histograms() {
        let data = [0.0f32, 0.9];
        let ctx = ctx_for(&data);
        let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; 2];

        let hists = prepare_image_channels_for_slint(&[ctx], &mut dest, 4, false, &vec![0], false);

        assert!(hists.is_empty());
    }

    #[test]
    fn prepare_image_channels_for_slint_builds_a_normalized_per_channel_histogram() {
        let data = [0.0f32, 0.9];
        let ctx = ctx_for(&data);
        let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; 2];

        let hists = prepare_image_channels_for_slint(&[ctx], &mut dest, 4, true, &vec![0], false);

        assert_eq!(hists.len(), 1);
        // p=0.0 -> bin 0; p=0.9 -> bin (0.9*4)=3 (truncated); bins 1,2 empty.
        // Normalized against the max count (1), so both hit bins read 1.0.
        assert_eq!(hists[0], vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn prepare_image_channels_for_slint_hidden_channels_are_excluded_from_the_output() {
        let data = [1.0f32];
        let ctx = ctx_for(&data);
        let mut dest = vec![Rgb8Pixel { r: 9, g: 9, b: 9 }; 1];

        // visible_channels doesn't include channel_idx 0, so it must be skipped entirely.
        prepare_image_channels_for_slint(&[ctx], &mut dest, 4, false, &vec![], false);

        assert_eq!(dest[0], Rgb8Pixel { r: 0, g: 0, b: 0 });
    }

    // -- histogram_to_svg_fast -----------------------------------------------------

    #[test]
    fn histogram_to_svg_fast_drops_empty_histograms() {
        let histos = vec![(vec![], [1.0, 0.0, 0.0]), (vec![0.5, 1.0], [0.0, 1.0, 0.0])];
        let result = histogram_to_svg_fast(&histos, 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn histogram_to_svg_fast_path_starts_and_ends_on_the_baseline() {
        let histos = vec![(vec![0.0, 1.0, 0.0], [1.0, 1.0, 1.0])];
        let result = histogram_to_svg_fast(&histos, 3);
        let path: String = result[0].path.to_string();

        assert!(path.starts_with("M 0 100"));
        assert!(path.ends_with("100 100 Z"));
    }

    #[test]
    fn histogram_to_svg_fast_a_full_bin_reaches_the_top_of_the_viewbox() {
        let histos = vec![(vec![1.0], [1.0, 1.0, 1.0])];
        let result = histogram_to_svg_fast(&histos, 1);
        let path: String = result[0].path.to_string();

        // val=1.0 -> y = (1.0 - 1.0) * 100.0 = 0.00 (top of a 100x100 viewBox)
        assert!(path.contains(" 0.00 0.00"), "path was: {path}");
    }
}
