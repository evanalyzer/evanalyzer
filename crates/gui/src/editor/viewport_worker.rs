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
use log::{debug, warn};
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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
            ("RoiWorker", TaskDispatch::Rois),
        ];

        for (name, dispatch) in configs {
            let self_handle = Arc::clone(self);
            std::thread::Builder::new()
                .name(name.into())
                .spawn(move || self_handle.run_worker_loop(dispatch))
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
        let mut screen_buffer = SharedPixelBuffer::<Rgb8Pixel>::new(1, 1);

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
                &self.viewport_controller.drawing_tasks.roi_task.task_count,
                &self.viewport_controller.drawing_tasks.roi_task,
                false,
            ),
        };

        loop {
            let mut task = wait_for_task(&drawing_task_container);

            // --- ROI scope: simple path, no image processing ---
            if scope == TaskDispatch::Rois {
                self.viewport_controller.sync_rois_to_slint_viewport();
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
                debug!("No current image");
                continue;
            }

            debug!("Started heavy thread!");

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
            // HighRes skips disk I/O when in breakpoint mode.
            let show_bp = !is_low_res && in_breakpoint_mode;

            // ----------------------------------------------------------------
            // STEP 3: Disk I/O (skipped in breakpoint mode)
            // ----------------------------------------------------------------
            let read_result = if show_bp {
                match &*self.viewport_controller.breakpoint_channel.read().unwrap() {
                    Some(bp) => {
                        let is_rgb = matches!(&*bp.image, ImageContainer::F32Rgb(_));
                        let channel = ImageChannel {
                            image: bp.image.clone(),
                            color: [1.0, 1.0, 1.0],
                            c_stack: 0,
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
                // STEP 5: Auto-adjust - write lock acquired AFTER read lock dropped
                // Safe because we dropped the project read lock in STEP 1.
                // Skipped in breakpoint mode to preserve the original histogram.
                // ------------------------------------------------------------
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

                // ------------------------------------------------------------
                // STEP 6: Re-read histogram settings after potential write
                // Fresh read lock - safe because write lock was released above
                // ------------------------------------------------------------
                let hist_settings_fresh = self
                    .app_state
                    .get_project()
                    .get_image_channel_histograms()
                    .clone();

                // Build channel contexts for rendering
                let mut channel_contexts = Vec::with_capacity(render_src.len());
                let mut is_rgb = false;

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
                                    / (histogram.max_limit - histogram.min_limit).max(f32::EPSILON),
                                channel_idx: idx,
                            });
                        }
                    }
                }

                // ------------------------------------------------------------
                // STEP 7: Render pixels + build histograms - pure CPU, no locks
                // ------------------------------------------------------------
                let all_hists = prepare_image_channels_for_slint(
                    &channel_contexts,
                    master_slice,
                    NUM_BINS,
                    !is_low_res,
                    &visible_channels,
                    is_rgb,
                );

                // ------------------------------------------------------------
                // STEP 7b: Composite into a viewport-sized, screen-space buffer
                // ------------------------------------------------------------
                // The native tile buffer (image_w x image_h) would otherwise be
                // handed to Slint as a single Image element positioned at draw_x
                // and stretched to zoomed_w/zoomed_h.  When zoomed/panned that
                // element's origin sits far off-screen (draw_x can be thousands of
                // px negative) and it is several thousand px wide.  The Slint
                // SOFTWARE renderer (Windows build) stores scene coordinates as
                // i16 and samples scaled images with an 8-bit fixed-point step;
                // the per-step rounding error gets multiplied by the large
                // off-screen offset, shifting the image by several pixels - and by
                // a different amount at every zoom level.  The GPU/Skia renderer
                // (Linux/macOS) does not, which is why this only appears on
                // Windows.  By resampling the visible region into a viewport-sized
                // buffer drawn at (0,0) with scale 1:1, the renderer never scales
                // or offsets a large image, so the picture lines up with the
                // screen-space ROI overlay exactly on every platform.
                let vp_w = prepared.viewport_width.max(1.0) as usize;
                let vp_h = prepared.viewport_height.max(1.0) as usize;
                if screen_buffer.width() as usize != vp_w
                    || screen_buffer.height() as usize != vp_h
                {
                    screen_buffer = SharedPixelBuffer::new(vp_w as u32, vp_h as u32);
                }
                {
                    let img_w = prepared.image_w;
                    let img_h = prepared.image_h;
                    let inv_scale_x = prepared.image_w as f32 / prepared.zoomed_w.max(f32::EPSILON);
                    let inv_scale_y = prepared.image_h as f32 / prepared.zoomed_h.max(f32::EPSILON);
                    let draw_x = prepared.draw_x;
                    let draw_y = prepared.draw_y;
                    let native = buffer_pool[pool_idx].as_slice();
                    let screen = screen_buffer.make_mut_slice();
                    let black = Rgb8Pixel { r: 0, g: 0, b: 0 };
                    for sy in 0..vp_h {
                        let ty = (sy as f32 - draw_y) * inv_scale_y;
                        let row = sy * vp_w;
                        if ty < 0.0 || ty >= img_h as f32 {
                            screen[row..row + vp_w].fill(black);
                            continue;
                        }
                        let ty_i = ty as usize * img_w;
                        for sx in 0..vp_w {
                            let tx = (sx as f32 - draw_x) * inv_scale_x;
                            screen[row + sx] = if tx >= 0.0 && tx < img_w as f32 {
                                native[ty_i + tx as usize]
                            } else {
                                black
                            };
                        }
                    }
                }

                // Display geometry is now screen-space: full viewport at (0,0).
                // The logical transform (zoom/offset/full_image) in `prepared` is
                // left untouched so sync_zoom, the navigator and the pixel-value
                // HUD (which maps via active_high_res_data) keep working.
                let mut display = prepared.clone();
                display.draw_x = 0.0;
                display.draw_y = 0.0;
                display.zoomed_w = vp_w as f32;
                display.zoomed_h = vp_h as f32;

                pixel_buffer_to_send = Some(screen_buffer.clone());
                render_info = Some(display);

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
    for (i, ctx) in channels.iter().enumerate() {
        if !visible_channels.contains(&ctx.channel_idx) {
            continue;
        }
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

                for (c_idx, ctx) in channels.iter().enumerate() {
                    if !visible_channels.contains(&ctx.channel_idx) {
                        continue;
                    }

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
