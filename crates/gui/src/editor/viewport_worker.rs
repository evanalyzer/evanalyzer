//! # viewport_worker
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::{
    HistogramData, UiState,
    editor::{
        histogram_controller::HistogramController,
        view_channel::{SampleSlice, ViewChannel},
        viewport_cache::{ReadContext, ViewportCache, to_z_projection},
        viewport_controller::{DrawingTaskContainer, ImageRect, ViewportController, ViewportState},
        viewport_task::{DrawingTask, TaskDispatch},
    },
};
use evanalyzer_app::images::ImageChannel;
use evanalyzer_app::images::ImageContainer;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::settings::images_settings::HistogramSettings;
use log::{debug, warn};
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
    pub(crate) image_data: SampleSlice<'a>,
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

/// The breakpoint image, ready to render, and where it lies.

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
        let mut buffers = RenderBuffers::new();
        let drawing_task_container = self.task_container(scope);
        loop {
            let task = wait_for_task(drawing_task_container);
            self.render_task(scope, task, &mut buffers);
        }
    }

    /// The task slot this worker scope takes its drawing tasks from.
    fn task_container(&self, scope: TaskDispatch) -> &Arc<DrawingTaskContainer> {
        match scope {
            TaskDispatch::LowRes => &self.viewport_controller.drawing_tasks.low_res_task,
            TaskDispatch::HighRes => &self.viewport_controller.drawing_tasks.high_res_task,
            _ => &self.viewport_controller.drawing_tasks.object_task,
        }
    }

    /// Renders one drawing task and hands the result to the UI.
    fn render_task(&self, scope: TaskDispatch, mut task: DrawingTask, b: &mut RenderBuffers) {
        let drawing_task_container = self.task_container(scope);
        let version_tracker = &drawing_task_container.task_count;
        let is_low_res = scope == TaskDispatch::LowRes;
        let task_start = Instant::now();

        // --- object scope: simple path, no image processing ---
        if scope == TaskDispatch::Objects {
            self.viewport_controller.sync_objects_to_slint_viewport();
            return;
        }

        // Taken before any display settings are read: a change landing
        // mid-render then leaves this render tagged as outdated.
        let generation = self.viewport_controller.content_generation();

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
            auto_scale_off,
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
                // The user's setting: a new image's channels auto-scaled.
                !self
                    .viewport_controller
                    .memory_settings()
                    .auto_scale_new_images,
            )
        };

        if !has_image {
            //debug!("No current image");
            return;
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

        // Compare: the original here, the breakpoint image in a second
        // render after it (`render_compare`).
        let comparing = self.viewport_controller.comparing();

        // ----------------------------------------------------------------
        // STEP 3: Disk I/O
        // ----------------------------------------------------------------
        let read_result = self.viewport_cache.read_image_tile_combined(
            series,
            to_z_projection(z_stack.z_projection.clone()),
            z_stack.z_range.clone(),
            t_stack.t_stack.clone(),
            task.fit_to_screen,
            task.is_new_image,
            is_low_res,
            &viewport_state,
        );
        let read_duration = task_start.elapsed();
        let step4_7_start = Instant::now();

        // Cancel if a newer request came in during the slow disk read
        if version_tracker.load(Ordering::SeqCst) > current_version {
            return;
        }

        // ----------------------------------------------------------------
        // STEP 4: Process loaded image
        // ----------------------------------------------------------------
        let mut pixel_buffer_to_send = None;
        let mut svg_hists_to_send = Vec::new();
        let mut histogram_scale = 1.0;
        let mut render_info = None;

        if let Ok((render_src, prepared)) = read_result {
            if !is_low_res {
                if let Ok(mut active) = self.viewport_cache.active_high_res_data.write() {
                    *active = Some((render_src.clone(), prepared.clone()));
                }
            }

            // Resize buffer from pool if needed. A high-res buffer the tile
            // history may still hold is never written again (that would
            // copy it first) - a fresh one is taken instead.
            b.pool_idx = (b.pool_idx + 1) % b.buffer_pool.len();
            let tile_history_on = !is_low_res
                && self
                    .viewport_controller
                    .memory_settings()
                    .tile_history_enabled;
            if tile_history_on
                || b.buffer_pool[b.pool_idx].width() != prepared.image_w as u32
                || b.buffer_pool[b.pool_idx].height() != prepared.image_h as u32
            {
                b.buffer_pool[b.pool_idx] =
                    SharedPixelBuffer::new(prepared.image_w as u32, prepared.image_h as u32);
            }

            let master_slice = b.buffer_pool[b.pool_idx].make_mut_slice();
            master_slice.fill(Rgb8Pixel { r: 0, g: 0, b: 0 });

            // ------------------------------------------------------------
            // STEP 5-7: Render pixels + build histograms - pure CPU, no locks.
            // ------------------------------------------------------------
            let mut channel_contexts: Vec<ChannelCtx> = Vec::with_capacity(render_src.len());
            let mut is_rgb = false;

            let all_hists: Vec<Vec<f32>> = {
                // STEP 5: Auto-adjust - write lock acquired AFTER read lock
                // dropped. Safe because we dropped the project read lock in
                // STEP 1.
                if task.auto_adjust_if_not_set || task.auto_adjust_selected || task.auto_adjust_all
                {
                    for channel in render_src.iter() {
                        let idx = channel.c_stack;
                        if let Some(ch) = hist_settings.get(&idx) {
                            if (!ch.is_some() && task.auto_adjust_if_not_set)
                                || (task.auto_adjust_selected && idx == selected_channel)
                                || task.auto_adjust_all
                            {
                                // Auto-scale off: a channel without any
                                // scaling shows its full value range (the
                                // values are normalized to 0..1).
                                let (min, max, min_range, max_range) = if auto_scale_off
                                    && !task.auto_adjust_selected
                                    && !task.auto_adjust_all
                                {
                                    (0.0, 1.0, 0.0, 1.0)
                                } else {
                                    auto_adjust_range(channel.slice(), channel.is_rgb)
                                };
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
                let (hist_settings_fresh, display_colors, grayscale) = {
                    let project = self.app_state.get_project();
                    (
                        project.get_image_channel_histograms().clone(),
                        project.get_channel_display_colors(),
                        project.images.settings.grayscale,
                    )
                };

                // Build channel contexts for rendering

                for channel in render_src.iter() {
                    let idx = channel.c_stack;
                    if let Some(Some(histogram)) = hist_settings_fresh.get(&idx) {
                        // One value per pixel; interleaved RGB and label
                        // maps have no intensity window to render with.
                        let data_slice = channel.slice();
                        is_rgb = channel.is_rgb;

                        if let Some(slice) = data_slice {
                            let color = channel_display_color(
                                channel,
                                display_colors.get(&idx).copied(),
                                grayscale,
                            );
                            channel_contexts.push(channel_ctx(
                                slice,
                                (*histogram).clone(),
                                color,
                                idx,
                            ));
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
                b.screen_pool_idx = (b.screen_pool_idx + 1) % b.screen_buffer_pool.len();
                if tile_history_on
                    || b.screen_buffer_pool[b.screen_pool_idx].width() as usize != vp_w
                    || b.screen_buffer_pool[b.screen_pool_idx].height() as usize != vp_h
                {
                    b.screen_buffer_pool[b.screen_pool_idx] =
                        SharedPixelBuffer::new(vp_w as u32, vp_h as u32);
                }
                let composite_start = Instant::now();
                let slice_start = Instant::now();
                let native = b.buffer_pool[b.pool_idx].as_slice();
                let screen = b.screen_buffer_pool[b.screen_pool_idx].make_mut_slice();
                let slice_duration = slice_start.elapsed();
                composite_to_screen(native, &prepared, screen, vp_w, vp_h);
                let composite_duration = composite_start.elapsed();
                debug!(
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

                pixel_buffer_to_send = Some(b.screen_buffer_pool[b.screen_pool_idx].clone());
                render_info = Some(display);
            } else {
                debug!(
                    "Viewport frame ({}): read {:?}, render(4-7) {:?} [{} channels], no composite (direct GPU scale, {} threads), total so far {:?}",
                    if is_low_res { "low-res" } else { "high-res" },
                    read_duration,
                    step4_7_duration,
                    channel_contexts.len(),
                    rayon::current_num_threads(),
                    task_start.elapsed()
                );

                pixel_buffer_to_send = Some(b.buffer_pool[b.pool_idx].clone());
                render_info = Some(prepared.clone());
            }

            if !is_low_res {
                // The selected channel's only: the panel's black and white
                // lines are on its axis (each channel counts on its own).
                svg_hists_to_send = histogram_to_svg_fast(
                    &all_hists
                        .into_iter()
                        .zip(channel_contexts.iter())
                        .filter(|(_, c)| c.channel_idx == selected_channel)
                        .map(|(h, c)| (h, c.color))
                        .collect(),
                    NUM_BINS,
                    self.viewport_controller.histogram_log(),
                );
                histogram_scale = 2f32.powi(prepared.bit_depth as i32);
            }
        } else if let Err(e) = read_result {
            warn!("Error reading image tile: {:?}", e);
            // A new image that can't be read: not the last one's picture.
            if task.is_new_image || task.is_new_series {
                self.viewport_controller.clear_rendered_image();
            }
        }

        // ----------------------------------------------------------------
        // STEP 8: Dispatch to UI thread
        // ----------------------------------------------------------------
        let busy = b.ui_busy.clone();
        busy.store(true, Ordering::SeqCst);

        if let (Some(pb), Some(info)) = (pixel_buffer_to_send, render_info) {
            if !is_low_res {
                self.viewport_controller
                    .publish_histogram(0, svg_hists_to_send, histogram_scale);
            }
            self.viewport_controller.sync_viewport_state_to_slint(
                pb,
                ImageRect::from_screen(
                    info.draw_x,
                    info.draw_y,
                    info.zoomed_w,
                    info.zoomed_h,
                    info.zoom,
                    info.offset_x,
                    info.offset_y,
                ),
                info.zoom,
                generation,
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
                // Only new content hides the current high-res render (it
                // shows the previous image); after anything else it still
                // lines up with the overview and stays.
                if task.is_new_image || task.is_new_series {
                    self.viewport_controller.sync_high_res_ready_to_slint(false);
                }
                *self
                    .viewport_controller
                    .full_image_size
                    .lock()
                    .expect("Poisoned") = (info.full_image_w as i64, info.full_image_h as i64);
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

        if comparing && !is_low_res && version_tracker.load(Ordering::SeqCst) == current_version {
            self.render_compare(&viewport_state, &mut b.compare);
        } else if !comparing {
            // Compare off (or no breakpoint): its render's RAM is freed.
            b.compare = None;
        }

        busy.store(false, Ordering::SeqCst);
        task.reset_job();
    }

    /// Compare: the breakpoint images (one per captured tile), rendered
    /// like the breakpoint view (their own histogram window, white; label
    /// maps in their colours) and sent as the second image, each where its
    /// tile lies. Rendered only when the picture changes (see
    /// `ViewportController::compare_epoch`) - after a pan or zoom Slint just
    /// moves them; on Windows, which shows screen-sized frames, the kept
    /// renders are only composited again.
    fn render_compare(&self, viewport_state: &ViewportState, slot: &mut Option<CompareRender>) {
        let epoch = self.viewport_controller.compare_epoch();
        let fresh = slot.as_ref().is_some_and(|c| c.epoch == epoch);
        if !fresh {
            *slot = self.render_compare_tiles(epoch);
        }
        let Some(render) = slot.as_ref() else {
            return;
        };
        if cfg!(target_os = "windows") {
            let vp_w = viewport_state.viewport_width.max(1.0) as usize;
            let vp_h = viewport_state.viewport_height.max(1.0) as usize;
            let mut screen = SharedPixelBuffer::<Rgb8Pixel>::new(vp_w as u32, vp_h as u32);
            let dest = screen.make_mut_slice();
            for (native, tile) in &render.tiles {
                composite_into_screen(
                    native.as_slice(),
                    &tile_context(tile, viewport_state),
                    dest,
                    vp_w,
                    vp_h,
                );
            }
            let rect = ImageRect::from_screen(
                0.0,
                0.0,
                vp_w as f32,
                vp_h as f32,
                viewport_state.zoom,
                viewport_state.offset_x,
                viewport_state.offset_y,
            );
            self.viewport_controller
                .sync_compare_to_slint(vec![(screen, rect)]);
        } else if !fresh {
            self.viewport_controller.sync_compare_to_slint(
                render
                    .tiles
                    .iter()
                    .map(|(native, tile)| {
                        let rect = ImageRect {
                            x: tile.offset_x as f32,
                            y: tile.offset_y as f32,
                            w: tile.width as f32,
                            h: tile.height as f32,
                        };
                        (native.clone(), rect)
                    })
                    .collect(),
            );
        }
    }

    /// Renders every breakpoint tile in the buffer the view mode picks
    /// (falling back to the intensity image if the pipeline hadn't made
    /// that one yet at the breakpoint) - and the histogram of them all.
    /// `None` without breakpoint images.
    fn render_compare_tiles(&self, epoch: u64) -> Option<CompareRender> {
        use crate::editor::viewport_controller::BreakpointViewMode;
        let tiles = self.viewport_controller.breakpoint_tiles.read().unwrap();
        if tiles.is_empty() {
            return None;
        }
        self.viewport_controller
            .compare_renders
            .fetch_add(1, Ordering::Relaxed);
        let mode = self.viewport_controller.breakpoint_view_mode();
        let source = |bp: &crate::editor::viewport_controller::BreakpointChannelData| match mode {
            BreakpointViewMode::Image => bp.image.clone(),
            BreakpointViewMode::Segmentation => {
                bp.segmentation.clone().unwrap_or_else(|| bp.image.clone())
            }
            BreakpointViewMode::Instances => {
                bp.instances.clone().unwrap_or_else(|| bp.image.clone())
            }
        };
        let channel_of = |bp: &crate::editor::viewport_controller::BreakpointChannelData| {
            let image = source(bp);
            let is_rgb = matches!(&*image, ImageContainer::F32Rgb(_));
            ViewChannel::wrap(&ImageChannel {
                image,
                color: [1.0, 1.0, 1.0],
                // The channel the breakpointed pipeline started from: its
                // settings are looked up by `c_stack`.
                c_stack: bp.channel_idx.unwrap_or(0),
                name: "Breakpoint".to_string(),
                is_rgb,
                is_visible: true,
            })
        };

        // One window for all tiles: set, or auto-adjusted from the first.
        let histogram = self.viewport_controller.breakpoint_histogram().or_else(|| {
            let first = channel_of(&tiles[0]);
            if matches!(&*source(&tiles[0]), ImageContainer::U32(_)) {
                return None;
            }
            let (min, max, min_limit, max_limit) = auto_adjust_range(first.slice(), first.is_rgb);
            let auto = HistogramSettings {
                min,
                max,
                min_limit,
                max_limit,
            };
            *self
                .viewport_controller
                .breakpoint_histogram
                .write()
                .unwrap() = Some(auto.clone());
            // The panel shows it: its sliders.
            if self.viewport_controller.histogram_side() == 1 {
                self.histogram_controller.sync_histogram_settings_to_slint();
            }
            Some(auto)
        });

        let mut rendered = Vec::with_capacity(tiles.len());
        let mut bins: Vec<f32> = Vec::new();
        let mut color = [1.0f32; 3];
        let mut nr_bits = 8;
        for bp in tiles.iter() {
            nr_bits = bp.nr_bits;
            let mut native =
                SharedPixelBuffer::<Rgb8Pixel>::new(bp.tile_width as u32, bp.tile_height as u32);
            let slice = native.make_mut_slice();
            let channel = channel_of(bp);
            if let ImageContainer::U32(img) = &*source(bp) {
                // Label maps have no histogram.
                render_labels_to_rgb8(img.data.as_slice(), slice);
            } else if let (Some(data), Some(window)) = (channel.slice(), histogram.clone()) {
                let ctx = channel_ctx(data, window, channel.color, channel.c_stack);
                color = ctx.color;
                let hists = prepare_image_channels_for_slint(
                    &[ctx],
                    slice,
                    NUM_BINS,
                    true,
                    &vec![channel.c_stack],
                    channel.is_rgb,
                );
                for h in hists {
                    if bins.len() < h.len() {
                        bins.resize(h.len(), 0.0);
                    }
                    for (sum, v) in bins.iter_mut().zip(h) {
                        *sum += v;
                    }
                }
            }
            rendered.push((
                native,
                CompareTile {
                    offset_x: bp.tile_offset_x,
                    offset_y: bp.tile_offset_y,
                    width: bp.tile_width,
                    height: bp.tile_height,
                },
            ));
        }
        drop(tiles);
        let curves = if bins.is_empty() {
            Vec::new()
        } else {
            histogram_to_svg_fast(
                &vec![(bins, color)],
                NUM_BINS,
                self.viewport_controller.histogram_log(),
            )
        };
        self.viewport_controller
            .publish_histogram(1, curves, 2f32.powi(nr_bits as i32));
        Some(CompareRender {
            epoch,
            tiles: rendered,
        })
    }
}

/// The compare's render: each breakpoint tile drawn, at the compare epoch
/// it shows (see `ViewportController::compare_epoch`).
struct CompareRender {
    epoch: u64,
    tiles: Vec<(SharedPixelBuffer<Rgb8Pixel>, CompareTile)>,
}

/// Where a breakpoint tile lies in the image (full-image pixels).
struct CompareTile {
    offset_x: usize,
    offset_y: usize,
    width: usize,
    height: usize,
}

/// A tile's placement on screen in the current view (for compositing).
fn tile_context(tile: &CompareTile, view: &ViewportState) -> ReadContext {
    ReadContext {
        zoomed_w: tile.width as f32 * view.zoom,
        zoomed_h: tile.height as f32 * view.zoom,
        zoom: view.zoom,
        draw_x: tile.offset_x as f32 * view.zoom + view.offset_x,
        draw_y: tile.offset_y as f32 * view.zoom + view.offset_y,
        offset_x: view.offset_x,
        offset_y: view.offset_y,
        read_off_x: tile.offset_x,
        read_off_y: tile.offset_y,
        res_idx: 0,
        image_w: tile.width,
        image_h: tile.height,
        bit_depth: 8,
        _nr_color_channels: 3,
        viewport_width: view.viewport_width,
        viewport_height: view.viewport_height,
        full_image_w: tile.offset_x + tile.width,
        full_image_h: tile.offset_y + tile.height,
    }
}

/// Like [`composite_to_screen`], but writes only the screen pixels the
/// image covers - several tiles go into one frame.
fn composite_into_screen(
    native: &[Rgb8Pixel],
    prepared: &ReadContext,
    screen: &mut [Rgb8Pixel],
    vp_w: usize,
    vp_h: usize,
) {
    let (img_w, img_h) = (prepared.image_w, prepared.image_h);
    let inv_x = img_w as f32 / prepared.zoomed_w.max(f32::EPSILON);
    let inv_y = img_h as f32 / prepared.zoomed_h.max(f32::EPSILON);
    let x0 = prepared.draw_x.max(0.0) as usize;
    let x1 = ((prepared.draw_x + prepared.zoomed_w).ceil().max(0.0) as usize).min(vp_w);
    let y0 = prepared.draw_y.max(0.0) as usize;
    let y1 = ((prepared.draw_y + prepared.zoomed_h).ceil().max(0.0) as usize).min(vp_h);
    for sy in y0..y1 {
        let ty = ((sy as f32 - prepared.draw_y) * inv_y) as usize;
        if ty >= img_h {
            continue;
        }
        let row = &mut screen[sy * vp_w..(sy + 1) * vp_w];
        for (sx, out) in row.iter_mut().enumerate().take(x1).skip(x0) {
            let tx = ((sx as f32 - prepared.draw_x) * inv_x) as usize;
            if tx < img_w {
                *out = native[ty * img_w + tx];
            }
        }
    }
}

/// How a channel is drawn: its values through the histogram window, in
/// `color`.
fn channel_ctx<'a>(
    image_data: SampleSlice<'a>,
    histogram: HistogramSettings,
    color: [f32; 3],
    channel_idx: i32,
) -> ChannelCtx<'a> {
    let inv_range = 1.0 / (histogram.max - histogram.min).max(0.001);
    ChannelCtx {
        image_data,
        color,
        r_factor: inv_range * color[0] * 255.0,
        g_factor: inv_range * color[1] * 255.0,
        b_factor: inv_range * color[2] * 255.0,
        offset: -histogram.min,
        h_mult: (NUM_BINS as f32 - 1.0)
            / (histogram.max_limit - histogram.min_limit).max(f32::EPSILON),
        channel_idx,
        histogram,
    }
}

/// Windows: the native tile resampled into a viewport-sized buffer, in
/// screen space (see STEP 7b in `render_task`).
fn composite_to_screen(
    native: &[Rgb8Pixel],
    prepared: &ReadContext,
    screen: &mut [Rgb8Pixel],
    vp_w: usize,
    vp_h: usize,
) {
    let img_w = prepared.image_w;
    let img_h = prepared.image_h;
    let inv_scale_x = prepared.image_w as f32 / prepared.zoomed_w.max(f32::EPSILON);
    let inv_scale_y = prepared.image_h as f32 / prepared.zoomed_h.max(f32::EPSILON);
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

/// Per-worker-thread state reused from frame to frame.
struct RenderBuffers {
    ui_busy: Arc<AtomicBool>,
    buffer_pool: [SharedPixelBuffer<Rgb8Pixel>; 2],
    pool_idx: usize,
    screen_buffer_pool: [SharedPixelBuffer<Rgb8Pixel>; 2],
    screen_pool_idx: usize,
    /// The compare's render of the breakpoint images (none until a compare).
    compare: Option<CompareRender>,
}

impl RenderBuffers {
    fn new() -> Self {
        let ui_busy = Arc::new(AtomicBool::new(false));

        let buffer_pool = [
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
        ];
        let pool_idx = 0;

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
        let screen_buffer_pool = [
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
            SharedPixelBuffer::<Rgb8Pixel>::new(1, 1),
        ];
        let screen_pool_idx = 0;

        Self {
            ui_busy,
            buffer_pool,
            pool_idx,
            screen_buffer_pool,
            screen_pool_idx,
            compare: None,
        }
    }
}

/// The colour a channel is drawn in: the project's for it (its emission
/// wavelength's, a default colour without one, white in grayscale mode - see
/// `ProjectExt::get_channel_display_colors`), or the reader's colour when the
/// project knows the channel not. RGB images keep their fixed colours -
/// except in grayscale mode, where every channel is white.
fn channel_display_color(
    channel: &ViewChannel,
    project_color: Option<[f32; 3]>,
    grayscale: bool,
) -> [f32; 3] {
    match project_color {
        _ if grayscale => [1.0, 1.0, 1.0],
        Some(color) if !channel.is_rgb => color,
        _ => channel.color,
    }
}

/// The colour a channel's histogram curve is drawn in: the channel's own,
/// except that white (grayscale mode, a lone channel without colour) turns a
/// medium gray - white wouldn't show on the light histogram panel.
fn histogram_color(channel_color: [f32; 3]) -> [f32; 3] {
    if channel_color.iter().all(|c| *c > 0.9) {
        [0.45, 0.45, 0.45]
    } else {
        channel_color
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

/// The pixel loop of [`prepare_image_channels_for_slint`]: blends the
/// visible channels into `dest_pixels` and counts their histograms.
/// `value(k, i)` is visible channel `k`'s value at pixel `i` - generic, so
/// each sample type gets a loop of its own without a per-pixel type check.
fn render_pixels<F>(
    channels: &[ChannelCtx],
    visible_indices: &[usize],
    dest_pixels: &mut [Rgb8Pixel],
    num_bins: usize,
    create_histogram: bool,
    value: F,
) -> Vec<Vec<u32>>
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    use rayon::prelude::*;

    let chunk_size = 1024 * 4;
    let n_channels = channels.len();

    dest_pixels
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

                for (k, &c_idx) in visible_indices.iter().enumerate() {
                    let ctx = &channels[c_idx];
                    let p = value(k, global_idx);

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
        )
}

/// Applies auto-adjustment to an image using partial sorting (O(N) average).
///
/// Samples every 10th pixel and finds the 0.5th and 99.5th percentile values
/// to use as the display range, clipping extreme outliers.
///
/// # Returns
/// `(min, max, min_limit, max_limit)` - display range and histogram limits.
/// `samples`: a viewport channel's values, `None` when it isn't one value
/// per pixel - the full range is used then.
pub(crate) fn auto_adjust_range(
    samples: Option<SampleSlice<'_>>,
    is_rgb: bool,
) -> (f32, f32, f32, f32) {
    if is_rgb {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let mut min = 0.0;
    let mut max = 1.0;
    // The axis: past black and white, so the histogram doesn't end at the
    // white line (cutting off the brightest pixels, in fluorescence often
    // the signal) - but only as far as there are values, so a few hot
    // pixels don't squash the curve into a corner.
    let mut lowest = 0.0f32;
    let mut highest = 1.0f32;

    if let Some(pixels) = samples {
        if pixels.len() > 0 {
            let mut sample: Vec<f32> = (0..pixels.len())
                .step_by(10)
                .map(|i| pixels.get(i))
                .collect();
            let len = sample.len();

            let low_idx = (len as f32 * 0.005) as usize;
            let high_idx = ((len as f32 * 0.995) as usize).min(len - 1);

            sample.select_nth_unstable_by(low_idx, |a, b| a.total_cmp(b));
            min = sample[low_idx];

            sample.select_nth_unstable_by(high_idx, |a, b| a.total_cmp(b));
            max = sample[high_idx];

            lowest = sample.iter().copied().fold(f32::INFINITY, f32::min);
            highest = sample.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
    }

    let span = max - min;
    let low = (min - (0.1 * span).max(0.01)).max(lowest.min(min - 0.01));
    let high = (max + (0.5 * span).max(0.01)).min(highest.max(max + 0.01));
    (min, max, low.max(0.0), high.min(1.0))
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

    let n_channels = channels.len();

    // The sample type is resolved once per frame, not per pixel: when all
    // visible channels share one (the usual case), the loop below is
    // compiled for exactly that type.
    macro_rules! typed {
        ($variant:ident) => {
            visible_indices
                .iter()
                .map(|&i| match channels[i].image_data {
                    SampleSlice::$variant(d, inv) => Some((d, inv)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        };
    }
    let raw_hists = if let Some(data) = typed!(U8) {
        render_pixels(
            channels,
            &visible_indices,
            dest_pixels,
            num_bins,
            create_histogram,
            |k, i| data[k].0[i] as f32 * data[k].1,
        )
    } else if let Some(data) = typed!(U16) {
        render_pixels(
            channels,
            &visible_indices,
            dest_pixels,
            num_bins,
            create_histogram,
            |k, i| data[k].0[i] as f32 * data[k].1,
        )
    } else {
        render_pixels(
            channels,
            &visible_indices,
            dest_pixels,
            num_bins,
            create_histogram,
            |k, i| channels[visible_indices[k]].image_data.get(i),
        )
    };

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
///
/// `log`: the counts on a logarithmic scale (three decades below the
/// fullest bin) - the background's peak no longer flattens the rest.
pub(crate) fn histogram_to_svg_fast(
    histos: &Vec<(Vec<f32>, [f32; 3])>,
    bins: usize,
    log: bool,
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
                let val = val.clamp(0.0, 1.0);
                let height = if log {
                    (1.0 + 999.0 * val).ln() / 1000f32.ln()
                } else {
                    val
                };
                let y = (1.0 - height) * 100.0;

                if i == 0 {
                    write!(path_data, " L {:.2} {:.2}", x, y).unwrap();
                } else {
                    write!(path_data, " {:.2} {:.2}", x, y).unwrap();
                }
            }

            write!(path_data, " 100 100 Z").unwrap();

            let color = histogram_color(*color);
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
    use crate::helper::size_formater::BYTES_PER_MB;
    use evanalyzer_app::global::ViewportMemorySettings;
    use evanalyzer_app::images::ManagedImage;
    use evanalyzer_app::images::Point2d;
    use evanalyzer_cfg::core_types::ImageDimensionality;
    use evanalyzer_cfg::core_types::ImagePlane;
    use kornia_image::{Image, ImageSize};
    use slint::Model;

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

    fn apply_auto_adjust(img: &ImageContainer, is_rgb: bool) -> (f32, f32, f32, f32) {
        let samples = match img {
            ImageContainer::F32Gray(image) => Some(SampleSlice::F32(image.as_slice())),
            _ => None,
        };
        auto_adjust_range(samples, is_rgb)
    }

    /// The compare's first tile image (empty without one).
    fn compare_image(r: &Render) -> slint::Image {
        slint::Model::row_data(&r.view().get_compare_tiles(), 0)
            .map(|t| t.image)
            .unwrap_or_default()
    }

    fn gray_container(pixels: Vec<f32>) -> ImageContainer {
        let len = pixels.len();
        gray_container_sized(pixels, len, 1)
    }

    fn gray_container_sized(pixels: Vec<f32>, width: usize, height: usize) -> ImageContainer {
        let image = Image::<f32, 1>::new(ImageSize { width, height }, pixels).unwrap();
        ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            dimensionality: ImageDimensionality::Planar,
        })
    }

    #[test]
    fn auto_adjust_without_samples_keeps_the_default_window() {
        assert_eq!(auto_adjust_range(None, false), (0.0, 1.0, 0.0, 1.0));
        let labels = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(
                ImageSize {
                    width: 2,
                    height: 1,
                },
                vec![1, 2],
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        assert_eq!(apply_auto_adjust(&labels, false), (0.0, 1.0, 0.0, 1.0));
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
        // Nor the axis: a few outliers don't stretch it.
        assert_eq!(min_limit, 0.49);
        assert_eq!(max_limit, 0.51);
    }

    /// The histogram isn't cut at the bright end: the axis reaches the
    /// brightest pixels (in fluorescence often the signal), the white
    /// point stays at 99.5 %.
    #[test]
    fn the_axis_reaches_the_brightest_pixels() {
        let mut pixels = vec![0.1f32; 10_000];
        for i in (0..10_000).step_by(1000) {
            pixels[i] = 0.8; // bright signal, in the sample
        }
        let img = gray_container(pixels);
        let (_, max, _, max_limit) = apply_auto_adjust(&img, false);
        assert!(max_limit >= 0.8, "axis to {max_limit}");
        assert!(max <= max_limit);
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
            image_data: SampleSlice::F32(data),
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
    fn compact_and_float_samples_render_identical_pixels_and_histograms() {
        let inv = 1.0 / 255.0f32;
        let raw: Vec<u8> = (0..=255).collect();
        let floats: Vec<f32> = raw.iter().map(|&q| q as f32 * inv).collect();

        let render = |ctx: ChannelCtx| {
            let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; raw.len()];
            let hists =
                prepare_image_channels_for_slint(&[ctx], &mut dest, 16, true, &vec![0], false);
            (dest, hists)
        };
        let mut compact = ctx_for(&floats);
        compact.image_data = SampleSlice::U8(&raw, inv);

        assert_eq!(render(compact), render(ctx_for(&floats)));
    }

    #[test]
    fn sixteen_bit_samples_render_like_their_float_values() {
        let inv = 1.0 / 65535.0f32;
        let raw: Vec<u16> = (0..=64).map(|i| i * 1023).collect();
        let floats: Vec<f32> = raw.iter().map(|&q| q as f32 * inv).collect();

        let render = |ctx: ChannelCtx| {
            let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; raw.len()];
            let hists =
                prepare_image_channels_for_slint(&[ctx], &mut dest, 16, true, &vec![0], false);
            (dest, hists)
        };
        let mut compact = ctx_for(&floats);
        compact.image_data = SampleSlice::U16(&raw, inv);

        let (pixels, hists) = render(compact);
        assert_eq!((pixels.clone(), hists), render(ctx_for(&floats)));
        assert!(
            pixels.iter().any(|p| p.r > 200),
            "bright samples drawn bright"
        );
    }

    #[test]
    #[should_panic(expected = "Memory alignment error")]
    fn a_channel_not_matching_the_frame_size_is_a_loud_error() {
        let data = [0.0f32, 0.5, 1.0];
        let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; 2];
        prepare_image_channels_for_slint(&[ctx_for(&data)], &mut dest, 4, false, &vec![0], false);
    }

    #[test]
    fn a_hidden_misaligned_channel_is_never_looked_at() {
        let data = [0.0f32, 0.5, 1.0];
        let mut dest = vec![Rgb8Pixel { r: 0, g: 0, b: 0 }; 2];
        let hists =
            prepare_image_channels_for_slint(&[ctx_for(&data)], &mut dest, 4, true, &vec![], false);
        assert_eq!(hists, vec![vec![0.0; 4]], "an empty curve for it");
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

    /// Log scale: a bin at a thousandth of the fullest still shows (about a
    /// third up), the fullest reaches the top, an empty one the baseline.
    #[test]
    fn histogram_to_svg_fast_log_lifts_small_counts() {
        let histos = vec![(vec![0.0, 0.001, 1.0], [1.0, 0.0, 0.0])];
        let path = histogram_to_svg_fast(&histos, 3, true)[0].path.to_string();
        let ys: Vec<f32> = path
            .trim_start_matches("M 0 100 L ")
            .trim_end_matches(" 100 100 Z")
            .split(' ')
            .skip(1)
            .step_by(2)
            .map(|y| y.parse().unwrap())
            .collect();
        assert_eq!(ys[0], 100.0);
        assert!(
            (ys[1] - 100.0 * (1.0 - 2f32.ln() / 1000f32.ln())).abs() < 0.1,
            "{path}"
        );
        assert_eq!(ys[2], 0.0);
    }

    #[test]
    fn histogram_to_svg_fast_drops_empty_histograms() {
        let histos = vec![(vec![], [1.0, 0.0, 0.0]), (vec![0.5, 1.0], [0.0, 1.0, 0.0])];
        let result = histogram_to_svg_fast(&histos, 2, false);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn histogram_to_svg_fast_path_starts_and_ends_on_the_baseline() {
        let histos = vec![(vec![0.0, 1.0, 0.0], [1.0, 1.0, 1.0])];
        let result = histogram_to_svg_fast(&histos, 3, false);
        let path: String = result[0].path.to_string();

        assert!(path.starts_with("M 0 100"));
        assert!(path.ends_with("100 100 Z"));
    }

    #[test]
    fn histogram_to_svg_fast_a_full_bin_reaches_the_top_of_the_viewbox() {
        let histos = vec![(vec![1.0], [1.0, 1.0, 1.0])];
        let result = histogram_to_svg_fast(&histos, 1, false);
        let path: String = result[0].path.to_string();

        // val=1.0 -> y = (1.0 - 1.0) * 100.0 = 0.00 (top of a 100x100 viewBox)
        assert!(path.contains(" 0.00 0.00"), "path was: {path}");
    }

    // -- rendering real frames (UI updates applied via the test queue) -------

    use crate::editor::test_support::{
        project_with_fixture_image, test_ui_window, ui_state_with_window,
    };
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{AppWindow, ViewportState as ViewportSlintState};
    use evanalyzer_cfg::settings::images_settings::{ZStackHandling, ZStackSettings};
    use slint::ComponentHandle;

    struct Render {
        ui: AppWindow,
        app_state: Arc<UiState>,
        worker: Arc<ViewportWorker>,
        buffers: RenderBuffers,
    }

    fn render_fixture(project: evanalyzer_app::project::ProjectWithRuntime) -> Render {
        let ui = test_ui_window();
        let app_state = ui_state_with_window(&ui, project);
        let cache = Arc::new(ViewportCache::new(app_state.clone()));
        let viewport = Arc::new(ViewportController::new(
            ui.as_weak(),
            app_state.clone(),
            cache.clone(),
        ));
        {
            let mut vp = viewport.viewport_state.write().unwrap();
            vp.viewport_width = 400.0;
            vp.viewport_height = 300.0;
            vp.zoom = 1.0;
        }
        let histogram = Arc::new(HistogramController::new(
            ui.as_weak(),
            app_state.clone(),
            viewport.clone(),
        ));
        let worker = Arc::new(ViewportWorker::new(
            app_state.clone(),
            viewport,
            histogram,
            cache,
        ));
        Render {
            ui,
            app_state,
            worker,
            buffers: RenderBuffers::new(),
        }
    }

    impl Render {
        fn draw(&mut self, scope: TaskDispatch, task: DrawingTask) {
            self.worker.render_task(scope, task, &mut self.buffers);
            drain_ui_queue();
        }
        fn view(&self) -> ViewportSlintState<'_> {
            self.ui.global::<ViewportSlintState>()
        }
    }

    fn new_image_task() -> DrawingTask {
        DrawingTask {
            auto_adjust_if_not_set: true,
            is_new_image: true,
            fit_to_screen: true,
            ..Default::default()
        }
    }

    /// "Auto-scale new images" off (user setting): a new image's channels
    /// show their full value range, kept in the project.
    #[test]
    fn without_auto_scale_a_new_image_shows_the_full_range() {
        let mut r = render_fixture(project_with_fixture_image());
        let mut settings = r.worker.viewport_controller.memory_settings();
        settings.auto_scale_new_images = false;
        r.worker.viewport_controller.set_memory_settings(settings);
        r.draw(TaskDispatch::HighRes, new_image_task());
        let histograms = r.app_state.get_project().get_image_channel_histograms();
        let window = histograms
            .values()
            .flatten()
            .next()
            .expect("a window was kept");
        assert_eq!(
            (window.min, window.max, window.min_limit, window.max_limit),
            (0.0, 1.0, 0.0, 1.0)
        );
    }

    /// "Auto-scale all": every channel's window is auto-adjusted again -
    /// also one the user set.
    #[test]
    fn auto_scale_all_adjusts_every_channel() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let channels: Vec<i32> = r
            .app_state
            .get_project()
            .get_image_channel_histograms()
            .keys()
            .copied()
            .collect();
        for c in &channels {
            r.app_state
                .get_project_write()
                .set_image_histogram_settings_for_channel(*c, 0.2, 0.3, 0.1, 0.4);
        }
        r.draw(
            TaskDispatch::HighRes,
            DrawingTask {
                auto_adjust_all: true,
                ..Default::default()
            },
        );
        let histograms = r.app_state.get_project().get_image_channel_histograms();
        let adjusted: Vec<_> = histograms
            .values()
            .flatten()
            .filter(|w| (w.min, w.max, w.min_limit, w.max_limit) != (0.2, 0.3, 0.1, 0.4))
            .collect();
        assert_eq!(adjusted.len(), channels.len(), "{histograms:?}");
    }

    #[test]
    fn a_new_image_is_auto_adjusted_fitted_and_drawn() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let image = r.ui.get_display_image();
        assert!(image.size().width > 0 && image.size().height > 0);
        assert!(r.view().get_high_res_ready());
        assert!(r.view().get_hr_img_w() > 0.0);
        // Auto-adjust stored a histogram window for the image's channels.
        let histograms = r.app_state.get_project().get_image_channel_histograms();
        assert!(histograms.values().any(|h| h.is_some()), "{histograms:?}");
        // A second frame of the same image reuses the cache.
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert!(r.ui.get_display_image().size().width > 0);
    }

    #[test]
    fn the_low_res_ghost_and_navigator_are_drawn() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        r.draw(TaskDispatch::LowRes, new_image_task());
        assert!(r.ui.get_ghost_image().size().width > 0);
        assert!(r.view().get_ghost_img_w() > 0.0);
        // A low-res frame never takes back an already posted high-res one.
        assert!(r.view().get_high_res_ready());
    }

    /// Full-resolution size of the image as read from the file (the
    /// fixture project's metadata doesn't match the file) - known once an
    /// overview was drawn.
    fn full_image_size(r: &mut Render) -> (f32, f32) {
        r.draw(TaskDispatch::LowRes, DrawingTask::default());
        let (w, h) = *r.worker.viewport_controller.full_image_size.lock().unwrap();
        assert!(w > 0 && h > 0);
        (w as f32, h as f32)
    }

    fn set_view(r: &Render, zoom: f32, offset_x: f32, offset_y: f32) {
        let mut vp = r.worker.viewport_controller.viewport_state.write().unwrap();
        vp.zoom = zoom;
        vp.offset_x = offset_x;
        vp.offset_y = offset_y;
    }

    #[test]
    fn the_overview_covers_the_whole_image_in_image_pixels_wherever_the_view_is() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let (full_w, full_h) = full_image_size(&mut r);

        // Panned far and zoomed in: the overview still lies at the image
        // origin, image-sized - the viewport places it with the live
        // transform, so a pan needs no new overview.
        set_view(&r, 3.0, -5000.0, -4000.0);
        r.draw(TaskDispatch::LowRes, DrawingTask::default());

        let v = r.view();
        assert!(v.get_ghost_img_x().abs() < 0.01, "{}", v.get_ghost_img_x());
        assert!(v.get_ghost_img_y().abs() < 0.01, "{}", v.get_ghost_img_y());
        assert!(
            (v.get_ghost_img_w() - full_w).abs() < 0.5,
            "{} vs {full_w}",
            v.get_ghost_img_w()
        );
        assert!(
            (v.get_ghost_img_h() - full_h).abs() < 0.5,
            "{} vs {full_h}",
            v.get_ghost_img_h()
        );
    }

    #[test]
    fn the_high_res_rect_is_where_the_render_lies_in_the_image() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let (full_w, full_h) = full_image_size(&mut r);

        let v = r.view();
        // Fitted to the screen, the whole image is one tile.
        assert!(v.get_hr_img_x().abs() < 0.01);
        assert!(v.get_hr_img_y().abs() < 0.01);
        assert!(
            (v.get_hr_img_w() - full_w).abs() < 0.5,
            "{} vs {full_w}",
            v.get_hr_img_w()
        );
        assert!(
            (v.get_hr_img_h() - full_h).abs() < 0.5,
            "{} vs {full_h}",
            v.get_hr_img_h()
        );
        assert!(v.get_hr_zoom() > 0.0);
    }

    #[test]
    fn a_low_res_frame_of_new_content_hides_the_previous_high_res_render() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        assert!(r.view().get_high_res_ready());

        // A new image: its overview arrives before its sharp render.
        r.worker.viewport_controller.trigger_new_image_redraw();
        r.draw(TaskDispatch::LowRes, new_image_task());
        assert!(!r.view().get_high_res_ready());

        r.draw(TaskDispatch::HighRes, new_image_task());
        assert!(r.view().get_high_res_ready());

        // Any other overview (a pan, a histogram change) leaves it shown.
        r.draw(TaskDispatch::LowRes, DrawingTask::default());
        assert!(r.view().get_high_res_ready());
    }

    /// Renders the fixture at its fitted zoom, then 4x zoomed in, then at
    /// the fitted zoom again - returns the fitted zoom.
    fn zoom_in_and_out(r: &mut Render) -> f32 {
        r.draw(TaskDispatch::HighRes, new_image_task());
        let fitted = r
            .worker
            .viewport_controller
            .viewport_state
            .read()
            .unwrap()
            .zoom;
        set_view(r, fitted * 4.0, 0.0, 0.0);
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        set_view(r, fitted, 0.0, 0.0);
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        fitted
    }

    fn enable_tile_history(r: &Render) {
        r.worker
            .viewport_controller
            .set_memory_settings(ViewportMemorySettings {
                tile_history_enabled: true,
                ..Default::default()
            });
        drain_ui_queue();
    }

    #[test]
    fn without_the_tile_history_no_earlier_render_is_kept() {
        let mut r = render_fixture(project_with_fixture_image());
        zoom_in_and_out(&mut r);
        assert_eq!(r.view().get_retained_tiles().row_count(), 0);
    }

    #[test]
    fn the_tile_history_keeps_a_sharper_earlier_render_after_zooming_out() {
        let mut r = render_fixture(project_with_fixture_image());
        enable_tile_history(&r);
        let fitted = zoom_in_and_out(&mut r);

        let kept: Vec<_> = r.view().get_retained_tiles().iter().collect();
        assert!(
            kept.iter().any(|t| (t.zoom - fitted * 4.0).abs() < 1e-4),
            "zooms kept: {:?}",
            kept.iter().map(|t| t.zoom).collect::<Vec<_>>()
        );
        // The first, fitted render is hidden by the current one (same area,
        // same sharpness) and not kept twice.
        assert!(kept.iter().all(|t| (t.zoom - fitted).abs() > 1e-4));
    }

    #[test]
    fn a_display_change_drops_the_tile_history() {
        let mut r = render_fixture(project_with_fixture_image());
        enable_tile_history(&r);
        zoom_in_and_out(&mut r);
        assert!(r.view().get_retained_tiles().row_count() > 0);

        // E.g. a histogram change: the kept renders show the old look.
        r.worker.viewport_controller.trigger_image_redraw();
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(r.view().get_retained_tiles().row_count(), 0);
    }

    #[test]
    fn turning_the_tile_history_off_frees_the_kept_renders() {
        let mut r = render_fixture(project_with_fixture_image());
        enable_tile_history(&r);
        zoom_in_and_out(&mut r);
        assert!(r.view().get_retained_tiles().row_count() > 0);

        r.worker
            .viewport_controller
            .set_memory_settings(ViewportMemorySettings::default());
        drain_ui_queue();
        assert_eq!(r.view().get_retained_tiles().row_count(), 0);
    }

    #[test]
    fn an_analysis_start_empties_the_caches_unless_the_settings_say_otherwise() {
        let mut r = render_fixture(project_with_fixture_image());
        enable_tile_history(&r);
        zoom_in_and_out(&mut r);
        r.draw(TaskDispatch::LowRes, DrawingTask::default());
        let cache = r.worker.viewport_cache.clone();
        let (tiles, overviews) = cache.held_bytes();
        assert!(tiles > 0 && overviews > 0);

        r.worker
            .viewport_controller
            .set_memory_settings(ViewportMemorySettings {
                tile_history_enabled: true,
                free_caches_on_analysis: false,
                ..Default::default()
            });
        r.worker
            .viewport_controller
            .release_preview_memory_for_analysis();
        drain_ui_queue();
        assert_eq!(cache.held_bytes(), (tiles, overviews), "switched off: kept");

        r.worker
            .viewport_controller
            .set_memory_settings(ViewportMemorySettings {
                tile_history_enabled: true,
                ..Default::default()
            });
        r.worker
            .viewport_controller
            .release_preview_memory_for_analysis();
        drain_ui_queue();
        assert_eq!(cache.held_bytes(), (0, 0));
        assert_eq!(r.view().get_retained_tiles().row_count(), 0);
        // What is on screen stays.
        assert!(r.ui.get_display_image().size().width > 0);
        assert!(r.view().get_high_res_ready());
    }

    #[test]
    fn the_object_overlay_reaches_the_margin_past_every_viewport_edge() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let zoom = 2.0;
        let (off_x, off_y) = (-30.0, 12.0);
        set_view(&r, zoom, off_x, off_y);

        for margin in [0u32, 64] {
            r.worker
                .viewport_controller
                .set_memory_settings(ViewportMemorySettings {
                    object_overlay_margin_px: margin,
                    ..Default::default()
                });
            r.draw(TaskDispatch::Objects, DrawingTask::default());

            let size = r.ui.get_object_image().size();
            assert_eq!(
                (size.width, size.height),
                (400 + 2 * margin, 300 + 2 * margin),
                "margin {margin}"
            );
            // Its top-left corner lies `margin` screen pixels before the
            // viewport's, in image pixels.
            let v = r.view();
            let m = margin as f32;
            assert!((v.get_obj_img_x() - (-m - off_x) / zoom).abs() < 1e-3);
            assert!((v.get_obj_img_y() - (-m - off_y) / zoom).abs() < 1e-3);
            assert!((v.get_obj_img_w() - size.width as f32 / zoom).abs() < 1e-3);
            assert!(v.get_object_ready());
        }
    }

    #[test]
    fn every_z_projection_renders() {
        for handling in [
            ZStackHandling::MaxIntensity,
            ZStackHandling::MinIntensity,
            ZStackHandling::AvgIntensity,
            ZStackHandling::SumIntensity,
            ZStackHandling::TakeTheMiddle,
            ZStackHandling::AllStacks,
        ] {
            let mut project = project_with_fixture_image();
            project.settings.images.settings.z_stack = Some(ZStackSettings {
                z_projection: handling.clone(),
                ..Default::default()
            });
            let mut r = render_fixture(project);
            r.draw(TaskDispatch::HighRes, new_image_task());
            assert!(
                r.ui.get_display_image().size().width > 0,
                "{handling:?} drew nothing"
            );
        }
    }

    #[test]
    fn without_a_current_image_nothing_is_drawn() {
        let mut r = render_fixture(evanalyzer_app::project::ProjectWithRuntime::default());
        r.draw(TaskDispatch::HighRes, new_image_task());
        assert_eq!(r.ui.get_display_image().size().width, 0);
        // The objects scope only syncs objects, image or not.
        r.draw(TaskDispatch::Objects, DrawingTask::default());
    }

    #[test]
    fn a_compare_renders_each_view_of_the_breakpoint_image() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        r.worker.viewport_controller.set_compare(true);
        // Nothing captured yet: no second image.
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(compare_image(&r).size().width, 0);

        let labels = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(
                ImageSize {
                    width: 4,
                    height: 2,
                },
                vec![0, 1, 1, 2, 0, 2, 3, 3],
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        r.worker.viewport_controller.set_breakpoint_channel(
            gray_container(vec![0.1, 0.5, 0.9, 0.2, 0.3, 0.4, 0.6, 0.7]),
            Some(labels),
            None,
            0,
            0,
            8,
            1,
            8,
            Some(0),
        );
        drain_ui_queue();
        // Intensity view, then the label map, then instances (falls back to
        // the intensity image - none were captured).
        for mode in [0, 1, 2] {
            r.worker.viewport_controller.set_breakpoint_view_mode(mode);
            r.draw(TaskDispatch::HighRes, DrawingTask::default());
            assert!(compare_image(&r).size().width > 0, "mode {mode}");
        }
    }

    /// The footer's breakpoint value: at the full-image pixel, in the
    /// image's units - a label map's label; nothing outside it.
    #[test]
    fn the_breakpoint_value_is_read_at_the_full_image_pixel() {
        let r = render_fixture(project_with_fixture_image());
        let labels = ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1>::new(
                ImageSize {
                    width: 8,
                    height: 1,
                },
                vec![0, 1, 7, 2, 0, 2, 3, 3],
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        let vc = &r.worker.viewport_controller;
        vc.set_breakpoint_channel(
            gray_container(vec![0.1, 0.5, 0.5, 0.2, 0.3, 0.4, 0.6, 0.7]),
            Some(labels),
            None,
            2,
            3,
            8,
            1,
            8,
            Some(0),
        );
        // Pixel (4, 3) is the tile's third.
        assert_eq!(vc.breakpoint_value_at(4.5, 3.2).as_deref(), Some("128"));
        assert_eq!(vc.breakpoint_value_at(1.5, 3.2), None, "left of it");
        assert_eq!(vc.breakpoint_value_at(4.5, 4.2), None, "below it");
        vc.set_breakpoint_view_mode(1);
        assert_eq!(vc.breakpoint_value_at(4.5, 3.2).as_deref(), Some("label 7"));
    }

    /// Compare: the file as the image, the breakpoint image as the second
    /// one, where it lies - also with the breakpoint view on. Off: the
    /// second one is gone and not rendered again.
    #[test]
    fn a_compare_renders_the_file_and_the_breakpoint_image_side_by_side() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let file_size = r.ui.get_display_image().size();
        r.worker.viewport_controller.set_breakpoint_channel(
            gray_container(vec![0.1, 0.5, 0.9, 0.2, 0.3, 0.4, 0.6, 0.7]),
            None,
            None,
            2,
            3,
            8,
            1,
            8,
            Some(0),
        );
        r.worker.viewport_controller.set_compare(true);
        drain_ui_queue();

        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(r.ui.get_display_image().size(), file_size, "the file");
        let view = r.view();
        assert!(view.get_compare_ready());
        let tile = slint::Model::row_data(&view.get_compare_tiles(), 0).expect("the tile");
        assert_eq!(tile.image.size().width, 8);
        let rect = [tile.img_x, tile.img_y, tile.img_w, tile.img_h];
        for (got, want) in rect.into_iter().zip([2.0, 3.0, 8.0, 1.0]) {
            assert!((got - want).abs() < 1e-3, "{rect:?}");
        }

        r.worker.viewport_controller.set_compare(false);
        drain_ui_queue();
        assert!(!r.view().get_compare_ready());
        assert_eq!(compare_image(&r).size().width, 0);
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(compare_image(&r).size().width, 0);
        assert_eq!(r.ui.get_display_image().size(), file_size);
    }

    /// The compare is rendered once per picture: a pan only moves it, a
    /// new view mode renders it again.
    #[test]
    fn a_pan_moves_the_compared_breakpoint_image_without_rendering_it_again() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let vc = r.worker.viewport_controller.clone();
        vc.set_breakpoint_channel(
            gray_container(vec![0.1, 0.5, 0.9, 0.2]),
            None,
            None,
            0,
            0,
            4,
            1,
            8,
            Some(0),
        );
        vc.set_compare(true);
        drain_ui_queue();
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        let renders = || vc.compare_renders.load(Ordering::Relaxed);
        assert_eq!(renders(), 1);
        {
            let mut view = vc.viewport_state.write().unwrap();
            view.offset_x += 40.0;
            view.zoom *= 1.5;
        }
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(renders(), 1, "pan and zoom: not rendered again");
        vc.set_breakpoint_view_mode(1);
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        assert_eq!(renders(), 2, "another view: rendered again");
    }

    /// Several tiles: each shown where it lies, each one's values read;
    /// the next run's first tile replaces them; clearing frees them all.
    #[test]
    fn breakpoint_tiles_are_shown_side_by_side_and_freed_when_cleared() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let vc = r.worker.viewport_controller.clone();
        let tile =
            |x: usize, value: f32| crate::editor::viewport_controller::BreakpointChannelData {
                image: Arc::new(gray_container(vec![value; 4])),
                segmentation: None,
                instances: None,
                tile_offset_x: x,
                tile_offset_y: 0,
                tile_width: 4,
                tile_height: 1,
                nr_bits: 8,
                channel_idx: Some(0),
            };
        vc.add_breakpoint_tile(tile(0, 0.25), true);
        vc.add_breakpoint_tile(tile(4, 0.5), false);
        vc.set_compare(true);
        drain_ui_queue();
        r.draw(TaskDispatch::HighRes, DrawingTask::default());
        let tiles = r.view().get_compare_tiles();
        assert_eq!(slint::Model::row_count(&tiles), 2);
        let xs: Vec<f32> = slint::Model::iter(&tiles).map(|t| t.img_x).collect();
        assert_eq!(xs, [0.0, 4.0]);
        assert_eq!(vc.breakpoint_value_at(1.5, 0.5).as_deref(), Some("64"));
        assert_eq!(vc.breakpoint_value_at(5.5, 0.5).as_deref(), Some("128"));

        // The next preview's first tile: the last run's go.
        vc.add_breakpoint_tile(tile(8, 0.5), true);
        assert_eq!(vc.breakpoint_tiles.read().unwrap().len(), 1);

        vc.clear_breakpoint();
        drain_ui_queue();
        assert!(vc.breakpoint_tiles.read().unwrap().is_empty());
        assert!(
            !r.ui
                .global::<crate::PipelinesPanelState>()
                .get_has_breakpoint_image()
        );
        assert_eq!(slint::Model::row_count(&r.view().get_compare_tiles()), 0);
        assert_eq!(vc.breakpoint_value_at(1.5, 0.5), None);
    }

    // -- channel colours -------------------------------------------------------

    fn image_channel(is_rgb: bool, color: [f32; 3]) -> ImageChannel {
        ImageChannel {
            image: Arc::new(gray_container(vec![0.5])),
            color,
            c_stack: 0,
            name: "C".into(),
            is_rgb,
            is_visible: true,
        }
    }

    #[test]
    fn white_channels_get_a_gray_histogram_curve() {
        assert_eq!(histogram_color([1.0, 1.0, 1.0]), [0.45, 0.45, 0.45]);
        assert_eq!(histogram_color([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn channels_are_drawn_in_the_projects_colour() {
        let reader_red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        assert_eq!(
            channel_display_color(
                &ViewChannel::wrap(&image_channel(false, reader_red)),
                Some(green),
                false
            ),
            green,
            "the project's colour wins"
        );
        assert_eq!(
            channel_display_color(
                &ViewChannel::wrap(&image_channel(false, reader_red)),
                None,
                false
            ),
            reader_red,
            "a channel the project doesn't know: the reader's colour"
        );
        assert_eq!(
            channel_display_color(
                &ViewChannel::wrap(&image_channel(true, reader_red)),
                Some(green),
                false
            ),
            reader_red,
            "RGB images keep their colours"
        );
        for is_rgb in [false, true] {
            assert_eq!(
                channel_display_color(
                    &ViewChannel::wrap(&image_channel(is_rgb, reader_red)),
                    None,
                    true
                ),
                [1.0, 1.0, 1.0],
                "grayscale mode: white, RGB images too"
            );
        }
    }

    #[test]
    fn a_changed_wavelength_recolours_the_rendered_image() {
        let mut project = crate::editor::test_support::project_with_image_file(
            crate::editor::test_support::grayscale_image_path(),
        );
        project.set_global_emission_wavel_length(0, 532.0); // pure green
        let mut r = render_fixture(project);
        r.draw(TaskDispatch::HighRes, new_image_task());
        let pixels = r.ui.get_display_image().to_rgb8().expect("an RGB8 frame");
        let lit: Vec<_> = pixels.as_slice().iter().filter(|p| p.g > 0).collect();
        assert!(!lit.is_empty(), "something was drawn");
        assert!(
            lit.iter().all(|p| p.r == 0 && p.b == 0),
            "drawn in green only"
        );
    }

    // -- benchmark (ignored: needs a whole slide image) ---------------------

    /// Current and peak resident memory of this process, in MB (Linux).
    fn rss_mb() -> (f64, f64) {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let field = |name: &str| {
            status
                .lines()
                .find(|l| l.starts_with(name))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
                .map_or(0.0, |kb| kb / 1024.0)
        };
        (field("VmRSS:"), field("VmHWM:"))
    }

    /// Resets the peak resident memory (`VmHWM`) to the current one (Linux).
    fn reset_peak_rss() {
        let _ = std::fs::write("/proc/self/clear_refs", "5");
    }

    fn env_u32(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Viewport benchmark on a real or generated whole slide image (see
    /// `examples/make_test_slide.rs`): a 1920x1080 viewport opens the image,
    /// zooms in, pans away and back - each step is one settled re-render
    /// (what the 150 ms debounce triggers). Prints time, cache size, and the
    /// process's current and peak RAM per step.
    ///
    /// EVANALYZER_BENCH_SLIDE=<slide.tif> [BENCH_TILE_CACHE_MB=1024]
    /// [BENCH_OVERVIEW_CACHE_MB=256] [BENCH_TRIM=1] [BENCH_PAN_STEPS=8]
    /// [BENCH_VIEWPORT=1920x1080] [BENCH_SERIES=0]
    /// RAYON_NUM_THREADS=2
    /// LD_LIBRARY_PATH=$PWD/libs/libtorch/lib cargo test --release
    /// -p evanalyzer_gui bench_viewport -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_viewport_whole_slide() {
        let Ok(slide) = std::env::var("EVANALYZER_BENCH_SLIDE") else {
            eprintln!("EVANALYZER_BENCH_SLIDE not set - skipped");
            return;
        };
        // Explicit sizes, or this computer's automatic ones.
        let manual = std::env::var("BENCH_TILE_CACHE_MB").is_ok();
        let tile_mb = env_u32("BENCH_TILE_CACHE_MB", 1024);
        let overview_mb = env_u32("BENCH_OVERVIEW_CACHE_MB", 256);
        // BENCH_VIEWPORT=WxH, default a 1080p screen.
        let (vp_w, vp_h) = std::env::var("BENCH_VIEWPORT")
            .ok()
            .and_then(|v| {
                let (w, h) = v.split_once('x')?;
                Some((w.parse::<f32>().ok()?, h.parse::<f32>().ok()?))
            })
            .unwrap_or((1920.0, 1080.0));

        let (rss_start, _) = rss_mb();
        let mut r = render_fixture(crate::editor::test_support::project_with_image_file(
            slide.clone().into(),
        ));
        // The file's structure, and BENCH_SERIES to pick the series viewed.
        if let Ok(source) = r.app_state.get_image_source(&slide.clone().into()) {
            for (idx, series) in &source.meta().series {
                println!(
                    "series {idx}: c={} z={} t={}, levels: {}",
                    series.nr_c_stacks,
                    series.nr_z_stacks,
                    series.nr_t_stacks,
                    series
                        .resolutions
                        .iter()
                        .map(|(l, p)| format!(
                            "{l}:{}x{} {}bit tile {}x{}",
                            p.width, p.height, p.nr_bits, p.tile_width, p.tile_height
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        if let Ok(series) = std::env::var("BENCH_SERIES") {
            let series: i32 = series.parse().expect("BENCH_SERIES is a number");
            r.app_state.get_project_write().set_active_series(&series);
        }
        r.worker
            .viewport_controller
            .set_memory_settings(ViewportMemorySettings {
                cache_sizes_auto: !manual,
                tile_cache_mb: tile_mb,
                overview_cache_mb: overview_mb,
                ..Default::default()
            });
        drain_ui_queue();
        let (tile_mb, overview_mb) = crate::editor::viewport_cache::effective_cache_mb(
            &r.worker.viewport_controller.memory_settings(),
        );
        {
            let mut vp = r.worker.viewport_controller.viewport_state.write().unwrap();
            vp.viewport_width = vp_w;
            vp.viewport_height = vp_h;
        }

        println!(
            "\nslide {slide}\ncaches {tile_mb} / {overview_mb} MB, {} threads, viewport {vp_w}x{vp_h}",
            rayon::current_num_threads()
        );
        println!(
            "{:<28} {:>9} {:>11} {:>14} {:>9} {:>9} {:>7}",
            "step", "ms", "tile px", "cache MB", "RSS MB", "peak MB", "drawn"
        );
        let mut total = std::time::Duration::ZERO;
        let mut step = |r: &mut Render, name: &str, scope: &[TaskDispatch], task: DrawingTask| {
            reset_peak_rss();
            let start = Instant::now();
            for s in scope {
                r.draw(*s, task.clone());
            }
            let took = start.elapsed();
            total += took;
            // BENCH_TRIM=1: hand freed memory back first, so RSS shows what
            // is actually in use rather than what glibc keeps for reuse.
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            if std::env::var("BENCH_TRIM").is_ok() {
                // SAFETY: only releases free allocator memory.
                unsafe {
                    libc::malloc_trim(0);
                }
            }
            let size = r.ui.get_display_image().size();
            let (tiles, overviews) = r.worker.viewport_cache.held_bytes();
            let (rss, peak) = rss_mb();
            // Sanity check: share of non-black pixels in the sharp render.
            let drawn = r.ui.get_display_image().to_rgb8().map_or(0.0, |b| {
                let px = b.as_slice();
                px.iter().filter(|p| p.r | p.g | p.b != 0).count() as f64 / px.len().max(1) as f64
            });
            println!(
                "{:<28} {:>9.1} {:>11} {:>14} {:>9.0} {:>9.0} {:>6.0}%",
                name,
                took.as_secs_f64() * 1000.0,
                format!("{}x{}", size.width, size.height),
                format!(
                    "{:.0}+{:.0}",
                    tiles as f64 / BYTES_PER_MB as f64,
                    overviews as f64 / BYTES_PER_MB as f64
                ),
                rss - rss_start,
                peak - rss_start,
                drawn * 100.0
            );
        };

        // Open: fitted overview and sharp render.
        step(
            &mut r,
            "open (fit to screen)",
            &[TaskDispatch::LowRes, TaskDispatch::HighRes],
            new_image_task(),
        );
        let (channels, ctx) = r.worker.viewport_cache.get_image_references();
        println!(
            "channels: {:?}, bit depth {}",
            channels
                .iter()
                .map(|(c, img)| format!("{c}:{img:?}"))
                .collect::<Vec<_>>(),
            ctx.bit_depth
        );
        let (full_w, full_h) = *r.worker.viewport_controller.full_image_size.lock().unwrap();
        let (cx, cy) = (full_w as f32 / 2.0, full_h as f32 / 2.0);
        // BENCH_COMPARE=1: a 4096 x 4096 breakpoint tile at the centre,
        // compared - as after a breakpoint preview there.
        if std::env::var("BENCH_COMPARE").is_ok() {
            let side = 4096usize.min(full_w as usize).min(full_h as usize);
            let values: Vec<f32> = (0..side * side)
                .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0)
                .collect();
            r.worker.viewport_controller.set_breakpoint_channel(
                gray_container_sized(values, side, side),
                None,
                None,
                (cx as usize).saturating_sub(side / 2),
                (cy as usize).saturating_sub(side / 2),
                side,
                side,
                8,
                Some(0),
            );
            r.worker.viewport_controller.set_compare(true);
            drain_ui_queue();
            println!("compare: a {side} x {side} breakpoint tile at the centre");
        }
        let center = |r: &Render, zoom: f32, cx: f32, cy: f32| {
            set_view(r, zoom, vp_w / 2.0 - cx * zoom, vp_h / 2.0 - cy * zoom)
        };

        for zoom in [0.05f32, 0.25, 1.0] {
            center(&r, zoom, cx, cy);
            step(
                &mut r,
                &format!("zoom {zoom}"),
                &[TaskDispatch::HighRes],
                DrawingTask::default(),
            );
        }

        // Pan right in screen-sized steps at 100%, then back over the same
        // places (cache hits if they were kept).
        let pan = 700.0;
        let steps = env_u32("BENCH_PAN_STEPS", 8) as usize;
        // Along the image's longer side, so the tour stays on the image.
        let (dx, dy) = if full_w >= full_h {
            (pan, 0.0)
        } else {
            (0.0, pan)
        };
        for i in 1..=steps {
            center(&r, 1.0, cx + i as f32 * dx, cy + i as f32 * dy);
            step(
                &mut r,
                &format!("pan right {i}"),
                &[TaskDispatch::HighRes],
                DrawingTask::default(),
            );
        }
        for i in (0..steps).rev() {
            center(&r, 1.0, cx + i as f32 * dx, cy + i as f32 * dy);
            step(
                &mut r,
                &format!("pan back {i}"),
                &[TaskDispatch::HighRes],
                DrawingTask::default(),
            );
        }
        let (_, peak) = rss_mb();
        println!(
            "total {:.0} ms; process peak since start {:.0} MB",
            total.as_secs_f64() * 1000.0,
            peak - rss_start
        );
    }

    /// Leak check: 400 settled renders over and over the same tour (zoom
    /// levels, pans, compare on and off) - RAM must level off, not climb.
    /// `EVANALYZER_BENCH_SLIDE=<slide.tif> cargo test --release -p
    /// evanalyzer_gui --lib bench_soak_viewport -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_soak_viewport() {
        let Ok(slide) = std::env::var("EVANALYZER_BENCH_SLIDE") else {
            return;
        };
        let (rss_start, _) = rss_mb();
        let mut r = render_fixture(crate::editor::test_support::project_with_image_file(
            slide.into(),
        ));
        {
            let mut vp = r.worker.viewport_controller.viewport_state.write().unwrap();
            vp.viewport_width = 1920.0;
            vp.viewport_height = 1080.0;
        }
        r.draw(TaskDispatch::LowRes, new_image_task());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let (full_w, full_h) = *r.worker.viewport_controller.full_image_size.lock().unwrap();
        let (cx, cy) = (full_w as f32 / 2.0, full_h as f32 / 2.0);
        let trim = || {
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            unsafe {
                libc::malloc_trim(0);
            }
        };
        println!("{:>6} {:>9} {:>9}", "render", "RSS MB", "peak MB");
        let start = Instant::now();
        for i in 0..400usize {
            let zoom = [0.05f32, 0.25, 1.0][i % 3];
            let x = cx + ((i / 3) % 8) as f32 * 700.0;
            set_view(&r, zoom, 960.0 - x * zoom, 540.0 - cy * zoom);
            // Every 50th: compare on/off with a breakpoint tile, as a user
            // switching breakpoints would.
            if i % 50 == 25 {
                let side = 2048usize;
                r.worker.viewport_controller.set_breakpoint_channel(
                    gray_container_sized(vec![0.5; side * side], side, side),
                    None,
                    None,
                    cx as usize,
                    cy as usize,
                    side,
                    side,
                    8,
                    Some(0),
                );
                r.worker.viewport_controller.set_compare(true);
            } else if i % 50 == 35 {
                r.worker.viewport_controller.set_compare(false);
                r.worker.viewport_controller.clear_breakpoint();
            }
            drain_ui_queue();
            r.draw(TaskDispatch::HighRes, DrawingTask::default());
            if i % 50 == 49 {
                trim();
                let (rss, peak) = rss_mb();
                println!(
                    "{:>6} {:>9.0} {:>9.0}",
                    i + 1,
                    rss - rss_start,
                    peak - rss_start
                );
            }
        }
        println!("400 renders in {:.1} s", start.elapsed().as_secs_f64());
    }

    // -- worker threads and further frame kinds ---------------------------------

    #[test]
    fn started_workers_take_every_posted_task() {
        let app_state = crate::editor::test_support::test_ui_state();
        let cache = Arc::new(ViewportCache::new(app_state.clone()));
        let viewport = Arc::new(ViewportController::new(
            slint::Weak::default(),
            app_state.clone(),
            cache.clone(),
        ));
        let histogram = Arc::new(HistogramController::new(
            slint::Weak::default(),
            app_state.clone(),
            viewport.clone(),
        ));
        let worker = Arc::new(ViewportWorker::new(
            app_state,
            viewport.clone(),
            histogram,
            cache,
        ));
        worker.start_worker();

        // High-res, low-res and objects - one task per worker.
        viewport.trigger_new_image_redraw();

        let tasks = &viewport.drawing_tasks;
        let pending = || {
            [
                &tasks.high_res_task,
                &tasks.low_res_task,
                &tasks.object_task,
            ]
            .iter()
            .filter(|c| c.task_request.0.lock().unwrap().is_some())
            .count()
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while pending() > 0 {
            assert!(Instant::now() < deadline, "{} tasks never taken", pending());
            std::thread::yield_now();
        }
    }

    #[test]
    fn a_new_series_shows_its_histogram_settings_without_auto_adjusting() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(
            TaskDispatch::HighRes,
            DrawingTask {
                is_new_series: true,
                fit_to_screen: true,
                ..Default::default()
            },
        );
        assert!(r.ui.get_display_image().size().width > 0);
        let histograms = r.app_state.get_project().get_image_channel_histograms();
        assert!(histograms.values().all(|h| h.is_none()), "{histograms:?}");
    }

    #[test]
    fn a_compared_breakpoint_image_is_drawn_in_gray_through_its_own_window() {
        let mut r = render_fixture(project_with_fixture_image());
        r.draw(TaskDispatch::HighRes, new_image_task());
        let channel_window = r
            .app_state
            .get_project()
            .get_image_channel_histograms()
            .get(&0)
            .cloned()
            .flatten();
        r.worker.viewport_controller.set_breakpoint_channel(
            gray_container(vec![0.1, 0.5, 0.9, 0.2, 0.3, 0.4, 0.6, 1.0]),
            None,
            None,
            0,
            0,
            8,
            1,
            8,
            Some(0),
        );
        r.worker.viewport_controller.set_compare(true);
        r.worker.viewport_controller.set_histogram_side(1);
        drain_ui_queue();

        r.draw(TaskDispatch::HighRes, DrawingTask::default());

        let pixels = compare_image(&r).to_rgb8().expect("an RGB8 frame");
        assert_eq!(pixels.width(), 8);
        let pixels = pixels.as_slice();
        assert!(pixels.iter().any(|p| p.r > 0), "something was drawn");
        assert!(pixels.iter().all(|p| p.r == p.g && p.g == p.b), "in gray");
        // Its own window, auto-adjusted - the channel's stays as it was.
        assert!(
            r.worker
                .viewport_controller
                .breakpoint_histogram()
                .is_some()
        );
        let window =
            |h: Option<HistogramSettings>| h.map(|h| (h.min, h.max, h.min_limit, h.max_limit));
        assert_eq!(
            window(
                r.app_state
                    .get_project()
                    .get_image_channel_histograms()
                    .get(&0)
                    .cloned()
                    .flatten()
            ),
            window(channel_window)
        );
        // The panel shows the breakpoint's curve, in 8-bit values.
        assert_eq!(
            r.ui.global::<crate::HistogramState>()
                .get_histogram_svg_path()
                .row_count(),
            1
        );
        assert_eq!(
            r.ui.global::<crate::HistogramState>().get_value_scale(),
            256.0
        );
    }
}
