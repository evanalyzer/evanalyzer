use crate::UiState;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::viewport_cache::{ViewportCache, effective_cache_mb, this_computer_ram_bytes};
use crate::editor::viewport_controller::ViewportController;
use crate::{
    AppWindow, ChannelInfo, ChannelState, ImageMetaData, ImagePixelInfo, IntensityProjection,
    ViewportMemoryState, ViewportState as ViewportSlintState,
};
use evanalyzer_app::global::ViewportMemorySettings;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::{
    TStackHandling, TStackSettings, ZStackHandling, ZStackSettings,
};
use log::warn;
use slint::{ComponentHandle, Model, Timer, TimerMode};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub(crate) struct ViewportImageController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) playback_timer: Timer,
    pub(crate) redraw_debounce_timer: Timer,
    /// Trailing timer for the pixel-info HUD throttle (see
    /// `sync_pixel_info_throttled`) - separate from `redraw_debounce_timer`
    /// so the two don't cancel each other.
    pub(crate) pixel_info_throttle_timer: Timer,
    pub(crate) last_pixel_info_sync: Mutex<Instant>,
    /// Trailing timer for the immediate low-res redraw throttle (see
    /// `trigger_redraw_low_res_throttled`) - separate from the other timers
    /// here so none of them cancel each other.
    pub(crate) low_res_throttle_timer: Timer,
    pub(crate) last_low_res_redraw: Mutex<Instant>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) viewport_cache: Arc<ViewportCache>,
    pub(crate) histogram_controller: Arc<HistogramController>,
    pub(crate) image_meta_controller: Arc<ImageMetaController>,
}

impl ViewportImageController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        viewport_cache: Arc<ViewportCache>,
        histogram_controller: Arc<HistogramController>,
        image_meta_controller: Arc<ImageMetaController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            playback_timer: Timer::default(),
            redraw_debounce_timer: Timer::default(),
            pixel_info_throttle_timer: Timer::default(),
            last_pixel_info_sync: Mutex::new(Instant::now()),
            low_res_throttle_timer: Timer::default(),
            last_low_res_redraw: Mutex::new(Instant::now()),
            viewport_controller,
            viewport_cache,
            histogram_controller,
            image_meta_controller,
        }
    }

    /// See `ViewportState.max_layer_extent` - i16 range in physical
    /// pixels, with room for a display scale factor of 2.
    const SOFTWARE_RENDERER_MAX_LAYER_EXTENT: f32 = 16000.0;

    /// Shows `settings` in the settings dialog's Viewport section.
    fn show_memory_settings(&self, settings: &ViewportMemorySettings) {
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ViewportMemoryState>();
            state.set_tile_history_enabled(settings.tile_history_enabled);
            state.set_tile_history_budget_mb(settings.tile_history_budget_mb as i32);
            state.set_object_overlay_margin_px(settings.object_overlay_margin_px as i32);
            // With automatic sizing the sizes in effect are shown (and kept as
            // the starting point if it's switched off).
            let (tile_mb, overview_mb) = effective_cache_mb(settings);
            let (auto_tile, auto_overview) =
                ViewportMemorySettings::auto_cache_mb(this_computer_ram_bytes());
            state.set_cache_sizes_auto(settings.cache_sizes_auto);
            state.set_auto_cache_note(
                format!(
                    "{:.0} GB RAM: {auto_tile} MB tiles, {auto_overview} MB overviews",
                    this_computer_ram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
                )
                .into(),
            );
            state.set_tile_cache_mb(tile_mb as i32);
            state.set_overview_cache_mb(overview_mb as i32);
            state.set_free_caches_on_analysis(settings.free_caches_on_analysis);
        }
    }

    /// Takes over what the settings dialog's Viewport section shows now -
    /// only what changed, so an Apply of other settings costs nothing here.
    fn apply_memory_settings_from_ui(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ViewportMemoryState>();
        let settings = ViewportMemorySettings {
            tile_history_enabled: state.get_tile_history_enabled(),
            tile_history_budget_mb: state.get_tile_history_budget_mb().max(0) as u32,
            object_overlay_margin_px: state.get_object_overlay_margin_px().max(0) as u32,
            tile_cache_mb: (state.get_tile_cache_mb().max(0) as u32)
                .max(ViewportMemorySettings::MIN_CACHE_MB),
            overview_cache_mb: (state.get_overview_cache_mb().max(0) as u32)
                .max(ViewportMemorySettings::MIN_CACHE_MB),
            free_caches_on_analysis: state.get_free_caches_on_analysis(),
            cache_sizes_auto: state.get_cache_sizes_auto(),
        };
        if settings != self.viewport_controller.memory_settings() {
            self.apply_memory_settings(settings);
        }
    }

    /// Applies `settings` to the viewport and saves them on this computer.
    fn apply_memory_settings(&self, settings: ViewportMemorySettings) {
        self.viewport_controller
            .set_memory_settings(settings.clone());
        self.app_state.save_viewport_memory_settings(settings);
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // The software renderer (Windows) stores scene coordinates as
            // i16 - layers scaled past that mid-gesture are hidden instead
            // of drawn misplaced (see `ViewportState.max_layer_extent`).
            if cfg!(target_os = "windows") {
                ui.global::<ViewportSlintState>()
                    .set_max_layer_extent(Self::SOFTWARE_RENDERER_MAX_LAYER_EXTENT);
            }

            // Viewport memory settings (Viewport section of the settings
            // dialog): shown when it opens, taken over on Apply.
            self.show_memory_settings(&self.viewport_controller.memory_settings());
            let manager = self.clone();
            ui.global::<ViewportMemoryState>().on_load(move || {
                manager.show_memory_settings(&manager.viewport_controller.memory_settings());
            });
            let manager = self.clone();
            ui.global::<ViewportMemoryState>().on_apply(move || {
                manager.apply_memory_settings_from_ui();
            });
            let manager = self.clone();
            ui.global::<ViewportMemoryState>()
                .on_reset_to_defaults(move || {
                    manager.show_memory_settings(&ViewportMemorySettings::default());
                });

            // Viewport size changed
            let manager = self.clone();
            ui.global::<ViewportSlintState>()
                .on_report_viewport_size(move |width, height| {
                    manager
                        .update_viewport_size_in_viewport_state(width, height)
                        .ok();
                });

            // Viewport moved (panned)
            let manager = self.clone();
            ui.global::<ViewportSlintState>().on_report_viewport_moved(
                move |offset_x, offset_y| {
                    manager
                        .update_viewport_position_in_viewport_state(offset_x, offset_y)
                        .ok();
                },
            );

            // Viewport zoomed (panned)
            let manager = self.clone();
            ui.global::<ViewportSlintState>().on_report_viewport_zoomed(
                move |zoom, offset_x, offset_y| {
                    manager
                        .update_viewport_zoom_in_viewport_state(zoom, offset_x, offset_y)
                        .ok();
                },
            );

            // Viewport mouse moved
            let manager = Arc::clone(self);
            ui.global::<ViewportSlintState>()
                .on_report_mouse_moved(move |x, y| {
                    manager.update_mouse_position_in_viewport_state(x, y);
                    manager.sync_pixel_info_throttled();
                });

            // Channel options changed
            let manager = Arc::clone(self);
            ui.global::<ChannelState>().on_channel_state_changed(
                move |channels_model,
                      _series,
                      z_stack,
                      t_stack,
                      projection,
                      _selected_channel_idx,
                      playback_speed| {
                    manager.update_channel_options_in_project(
                        channels_model.iter().collect(),
                        z_stack,
                        t_stack,
                        projection,
                        playback_speed as f32,
                    );
                },
            );

            // Active series changed
            let manager = Arc::clone(self);
            ui.global::<ChannelState>()
                .on_active_series_changed(move |active_series| {
                    manager.update_active_series_in_project(&active_series);
                });

            // Active channel changed
            let manager = Arc::clone(self);
            ui.global::<ChannelState>()
                .on_active_channel_changed(move |active_channel| {
                    manager.update_active_channel_in_project(&active_channel);
                });

            // Play button toggeled
            let manager = Arc::clone(self);
            ui.global::<ChannelState>()
                .on_toggle_play(move |_play_active| {
                    manager.handle_play_button_toggled();
                });
        }
    }

    /// Updates the internal viewport dimensions and transformation parameters.
    ///
    /// This method refreshes the stored state of the viewing area, typically called
    /// when the window is resized or when the layout changes. It ensures that
    /// subsequent coordinate calculations for panning and zooming remain accurate
    /// relative to the new container size.
    ///
    /// ### Arguments
    /// * `width` - The new width of the viewport container in logical pixels.
    /// * `height` - The new height of the viewport container in logical pixels.
    /// * `zoom` - The current magnification level to be preserved or updated.
    /// * `offset_x` - The current horizontal translation to be preserved or updated.
    /// * `offset_y` - The current vertical translation to be preserved or updated.
    ///
    /// ### Returns
    /// * `Ok(())` if the viewport state was successfully updated.
    /// * `Err(InternalErrors)` if the state lock is poisoned or internal bounds are invalid.
    pub fn update_viewport_size_in_viewport_state(
        self: &Arc<Self>,
        width: f32,
        height: f32,
    ) -> Result<(), InternalErrors> {
        let was_zero_sized = {
            let state = self
                .viewport_controller
                .viewport_state
                .read()
                .expect("Failed to acquire read lock on viewport state");
            state.viewport_width <= 0.0 || state.viewport_height <= 0.0
        };

        {
            let mut state = self
                .viewport_controller
                .viewport_state
                .write()
                .expect("Failed to acquire write lock on viewport state");
            state.viewport_width = width;
            state.viewport_height = height;
        }

        // If the viewport just became valid and an image is already waiting to be
        // displayed, re-fire a full fit-to-screen redraw (the initial one was
        // skipped because the viewport dimensions were 0 at that point).
        let image_pending = was_zero_sized
            && width > 0.0
            && height > 0.0
            && self
                .app_state
                .get_project()
                .tmp_settings
                .current_image
                .is_some();

        if image_pending {
            self.viewport_controller.trigger_new_image_redraw();
            return Ok(());
        }

        self.on_viewport_transform_changed();

        Ok(())
    }

    /// Delay after the last pan/zoom/resize event before the sharp tile and
    /// the objects are re-rendered for the new view. Until then the viewport
    /// moves the textures already on screen (see `viewport.slint`).
    const VIEWPORT_REDRAW_DEBOUNCE: Duration = Duration::from_millis(150);

    /// Common follow-up of a pan, zoom or resize: the layers already follow
    /// the new transform in Slint, so only the navigator is updated now and
    /// the re-render waits until the gesture settles. The software renderer
    /// (Windows) is the exception: its overview covers just the viewport,
    /// so it is still re-rendered while moving (throttled).
    fn on_viewport_transform_changed(self: &Arc<Self>) {
        self.viewport_controller
            .sync_navigator_from_viewport_state();

        if cfg!(target_os = "windows") {
            self.trigger_redraw_low_res_throttled();
        }

        self.redraw_debounce_timer.stop(); // Cancel any existing pending re-render
        let self_in = self.clone();
        self.redraw_debounce_timer.start(
            TimerMode::SingleShot,
            Self::VIEWPORT_REDRAW_DEBOUNCE,
            move || {
                self_in.viewport_controller.trigger_viewport_redraw();
            },
        );
    }

    /// Rate limit for [`Self::trigger_redraw_low_res_throttled`] (software
    /// renderer only - see `on_viewport_transform_changed`). A per-move cap
    /// is needed since raw mouse-move events fire at the input device's full
    /// poll rate (125-1000 Hz); 60/sec matches a typical display refresh
    /// rate.
    const LOW_RES_REDRAW_THROTTLE: Duration = Duration::from_millis(16);

    /// Throttled entry point for the "immediate" low-res redraw fired on
    /// every pan/zoom event - see [`Self::LOW_RES_REDRAW_THROTTLE`] for why
    /// this exists. Same leading+trailing-edge shape as
    /// `sync_pixel_info_throttled`: runs immediately if nothing has run in
    /// the last throttle window (so the first move still feels instant),
    /// otherwise coalesces further events into a single trailing redraw once
    /// the window elapses, so a fast drag doesn't leave a stale frame once
    /// it stops.
    fn trigger_redraw_low_res_throttled(self: &Arc<Self>) {
        let now = Instant::now();
        let should_run_now = {
            let mut last = self.last_low_res_redraw.lock().expect("Poisoned");
            if now.duration_since(*last) >= Self::LOW_RES_REDRAW_THROTTLE {
                *last = now;
                true
            } else {
                false
            }
        };

        if should_run_now {
            self.low_res_throttle_timer.stop();
            self.viewport_controller.trigger_redraw_low_res();
        } else {
            let self_in = self.clone();
            self.low_res_throttle_timer.stop();
            self.low_res_throttle_timer.start(
                TimerMode::SingleShot,
                Self::LOW_RES_REDRAW_THROTTLE,
                move || {
                    *self_in.last_low_res_redraw.lock().expect("Poisoned") = Instant::now();
                    self_in.viewport_controller.trigger_redraw_low_res();
                },
            );
        }
    }

    pub fn update_viewport_zoom_in_viewport_state(
        self: &Arc<Self>,
        zoom: f32,
        offset_x: f32,
        offset_y: f32,
    ) -> Result<(), InternalErrors> {
        {
            let mut state = self
                .viewport_controller
                .viewport_state
                .write()
                .expect("Failed to acquire write lock on viewport state");
            state.zoom = zoom;
            state.offset_x = offset_x;
            state.offset_y = offset_y;
        }
        self.viewport_controller.sync_scale_bar_to_slint();
        self.on_viewport_transform_changed();

        Ok(())
    }

    pub fn update_viewport_position_in_viewport_state(
        self: &Arc<Self>,
        offset_x: f32,
        offset_y: f32,
    ) -> Result<(), InternalErrors> {
        {
            let mut state = self
                .viewport_controller
                .viewport_state
                .write()
                .expect("Failed to acquire write lock on viewport state");
            state.offset_x = offset_x;
            state.offset_y = offset_y;
        }
        self.on_viewport_transform_changed();

        Ok(())
    }

    /// Updates the cached mouse cursor coordinates within the internal viewport state.
    ///
    /// This method records the current mouse position relative to the UI container,
    /// allowing other pipeline processes to access the latest cursor location for
    /// coordinate transformations, hover effects, or region-of-interest calculations.
    ///
    /// ### Arguments
    /// * `x` - The horizontal position of the mouse relative to the viewport origin.
    /// * `y` - The vertical position of the mouse relative to the viewport origin.
    pub fn update_mouse_position_in_viewport_state(&self, x: f32, y: f32) {
        let mut state = self
            .viewport_controller
            .viewport_state
            .write()
            .expect("Failed to acquire write lock on viewport state");
        state.mouse_pos_x = x;
        state.mouse_pos_y = y;
    }

    /// Rate limit for `sync_actual_mouse_position_information_to_slint`: ~30
    /// updates/sec is imperceptibly different from unthrottled for a pixel
    /// value readout, but caps the per-move cost (project/viewport-state
    /// locks, a `format!()` per visible channel, and an
    /// `invoke_from_event_loop` dispatch) that raw mouse-move events would
    /// otherwise trigger at the input device's full poll rate (commonly
    /// 125-1000 Hz).
    const PIXEL_INFO_THROTTLE: Duration = Duration::from_millis(33);

    /// Throttled entry point for the mouse-move pixel-info HUD. Runs
    /// immediately if nothing has run in the last [`Self::PIXEL_INFO_THROTTLE`]
    /// (so the first move after the cursor was still feels instant), otherwise
    /// coalesces further moves into a single trailing update once the window
    /// elapses - so a fast drag doesn't leave a stale readout once it stops.
    pub fn sync_pixel_info_throttled(self: &Arc<Self>) {
        let now = Instant::now();
        let should_run_now = {
            let mut last = self.last_pixel_info_sync.lock().expect("Poisoned");
            if now.duration_since(*last) >= Self::PIXEL_INFO_THROTTLE {
                *last = now;
                true
            } else {
                false
            }
        };

        if should_run_now {
            self.pixel_info_throttle_timer.stop();
            self.sync_actual_mouse_position_information_to_slint();
        } else {
            let self_in = self.clone();
            self.pixel_info_throttle_timer.stop();
            self.pixel_info_throttle_timer.start(
                TimerMode::SingleShot,
                Self::PIXEL_INFO_THROTTLE,
                move || {
                    *self_in.last_pixel_info_sync.lock().expect("Poisoned") = Instant::now();
                    self_in.sync_actual_mouse_position_information_to_slint();
                },
            );
        }
    }

    /// Synchronizes precise mouse interaction data and pixel-level information to the Slint UI.
    ///
    /// This method maps the current mouse cursor position from the UI's drawing coordinates
    /// back to the underlying image data. It is used to display live tooltips
    /// or status bars showing the pixel color values, coordinates, or channel data
    /// under the cursor.
    ///
    /// ### Arguments
    /// * `image_data` - A thread-safe reference to the raw image channels used for pixel sampling.
    /// * `draw_x` / `draw_y` - The current mouse coordinates relative to the UI drawing surface.
    /// * `image_w` / `image_h` - The dimensions of the source image being sampled.
    /// * `zoomed_w` / `zoomed_h` - The current dimensions of the rendered (zoomed) viewport.
    /// * `bit_depth` - The color depth of the image (e.g., 8, 10, or 16-bit), used for formatting value strings.
    ///
    /// ### Returns
    /// * `Ok(())` if the coordinate mapping was successful and the UI was updated.
    /// * `Err(InternalErrors)` if the coordinates fall outside valid bounds or image data is inaccessible.
    pub fn sync_actual_mouse_position_information_to_slint(&self) {
        let project = self.app_state.get_project();
        let pixel_sizes = project.get_pixel_sizes();

        let viewport_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Failed to acquire read lock on viewport state");

        let data_tmp = self
            .viewport_cache
            .active_high_res_data
            .read()
            .expect("Failed to acquire read lock on active high-res data");

        let Some((image_data, ctx)) = &*data_tmp else {
            // No active high-res data available, skip updating pixel info
            return;
        };

        // Calculate pixel coordinates locally (Fast Math)
        let local_x =
            (viewport_state.mouse_pos_x - ctx.draw_x) / (ctx.zoomed_w / ctx.image_w as f32);
        let local_y =
            (viewport_state.mouse_pos_y - ctx.draw_y) / (ctx.zoomed_h / ctx.image_h as f32);

        if local_x >= 0.0
            && local_x < ctx.image_w as f32
            && local_y >= 0.0
            && local_y < ctx.image_h as f32
        {
            let idx = (local_y as usize * ctx.image_w as usize) + local_x as usize;

            // Format the string right here
            let mut pixel_values = Vec::new();
            for channel in image_data.iter() {
                if !channel.is_rgb {
                    if let Some(raw_val) = channel.value_at(idx) {
                        let normalized_val = raw_val * 100.0;
                        let scaled_val = raw_val * 2.0f32.powf(ctx.bit_depth as f32);
                        pixel_values.push(format!(
                            "{}: {:.0} ({:.4}%)",
                            channel.name, scaled_val, normalized_val
                        ));
                    }
                }
            }

            // Calculate position
            let z_stack = match &project.get_z_stack() {
                Some(stack) => match &stack.z_range {
                    Some(range) => range.start().clone(),
                    None => 0,
                },
                _ => 0,
            };

            let pos = format!(
                "({:.1} {:.1} {:.1}) nm",
                local_x * pixel_sizes.x,
                local_y * pixel_sizes.y,
                (z_stack as f32 + 1.0) * pixel_sizes.z
            );

            let ui_weak = self.ui.clone();
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ImagePixelInfo>()
                        .set_pixel_value(pixel_values.join(" | ").into());
                    ui.global::<ImagePixelInfo>().set_mouse_pos(pos.into());
                }
            })
            .ok();
        }
    }

    /// Updates the image channel configurations and multi-dimensional playback settings within the project.
    ///
    /// This method modifies how the project interprets and displays image data across
    /// different dimensions (Z-stack and Time), handles intensity projections, and
    /// manages visual channel properties.
    ///
    /// ### Arguments
    /// * `channels` - A list of `ChannelInfo` objects defining the visibility, color, and mapping for each image channel.
    /// * `z_stack` - The index of the currently active focal plane in a 3D image stack.
    /// * `t_stack` - The index of the currently active time frame in a temporal sequence.
    /// * `projection` - The strategy used to flatten multi-layer data (e.g., Maximum Intensity Projection, Average, or None).
    /// * `playback_speed` - The rate at which the time series (T-stack) is played back in the UI.
    ///
    /// ### Returns
    /// * `Ok(())` if the project settings were successfully updated and propagated.
    /// * `Err(InternalErrors)` if the provided indices are out of range or the project state is immutable.
    pub fn update_channel_options_in_project(
        &self,
        channels: Vec<ChannelInfo>,
        z_stack: i32,
        t_stack: i32,
        projection: IntensityProjection,
        playback_speed: f32,
    ) {
        let mut project = self.app_state.get_project_write();
        let mut channel_visibility: BTreeMap<i32, bool> = BTreeMap::new();

        for ch in &channels {
            channel_visibility.insert(ch.idx, ch.active);
        }

        let (z_range, z_projection) = match projection {
            IntensityProjection::SingleStack => {
                (Some(z_stack..=z_stack), ZStackHandling::SingleStack)
            }
            IntensityProjection::AllStacks => (Some(z_stack..=z_stack), ZStackHandling::AllStacks),
            IntensityProjection::Max => (None, ZStackHandling::MaxIntensity),
            IntensityProjection::Min => (None, ZStackHandling::MinIntensity),
            IntensityProjection::Avg => (None, ZStackHandling::AvgIntensity),
            IntensityProjection::Sum => (None, ZStackHandling::SumIntensity),
            IntensityProjection::Middle => (None, ZStackHandling::TakeTheMiddle),
        };

        // Store the new settings to the project - in grayscale mode with
        // one visible channel at most; the list then shows that, too.
        let allowed = project.with_one_visible_channel(&channel_visibility);
        self.show_channel_visibilities(allowed.clone(), allowed != channel_visibility);

        project.set_global_preferences(&allowed);

        project.set_global_z_stack(&ZStackSettings {
            z_projection: z_projection,
            z_range: z_range,
        });

        project.set_global_t_stack(&TStackSettings {
            playback_speed: playback_speed as f32,
            t_stack,
            stack_handling: TStackHandling::AllStacks,
        });

        self.viewport_controller
            .trigger_redraw_low_res_and_high_res();
    }

    /// Shows how many channels are visible - and with `update_rows` which,
    /// when grayscale mode changed the user's choice.
    fn show_channel_visibilities(&self, visibilities: BTreeMap<i32, bool>, update_rows: bool) {
        let ui = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui.upgrade() else {
                return;
            };
            let visible = visibilities.values().filter(|visible| **visible).count();
            ui.global::<ChannelState>().set_active_count(visible as i32);

            if !update_rows {
                return;
            }

            let channels = ui.global::<ChannelState>().get_channels().clone();

            for row in 0..channels.row_count() {
                if let Some(mut channel) = channels.row_data(row)
                    && let Some(visible) = visibilities.get(&channel.idx)
                {
                    channel.active = *visible;
                    channels.set_row_data(row, channel);
                }
            }
            ui.global::<ChannelState>()
                .set_channels(std::rc::Rc::new(channels).into());
        })
        .ok();
    }

    /// Updates the currently selected image channel in the project and triggers a UI histogram refresh.
    ///
    /// This method sets the global channel focus within the project state and dispatches a
    /// background task to synchronize the histogram sliders for both high-resolution
    /// and low-resolution pipelines.
    ///
    /// ### Arguments
    /// * `selected_channel` - A reference to the integer index of the channel to be set as active.
    ///
    /// ### Side Effects
    /// * Updates the project's global state via `set_global_selected_channel`.
    /// * Dispatches a `DrawingTask` with the `UpdateHistoSliders` job to the worker threads
    ///   using `TaskDispatch::Both`.
    pub fn update_active_channel_in_project(&self, selected_channel: &i32) {
        {
            let mut project = self.app_state.get_project_write();
            project.set_global_selected_channel(selected_channel);
        } // write lock dropped before sync_histogram_settings_to_slint acquires read lock
        self.histogram_controller.sync_histogram_settings_to_slint();
        self.viewport_controller
            .trigger_redraw_low_res_and_high_res();
    }

    /// Sets a new active image series in the project and initializes the rendering pipeline for the new data.
    ///
    /// This method updates the project's current series index and dispatches a comprehensive
    /// background task to reset the viewport, recalculate histograms, and automatically
    /// adjust display settings to fit the new series to the screen.
    ///
    /// ### Arguments
    /// * `selected_series` - A reference to the index of the series to be activated.
    ///
    /// ### Side Effects
    /// * **State Update:** Updates the project's active series via `set_active_series`.
    /// * **Viewport Reset:** Flags the task as `is_new_series` and `fit_to_screen`, forcing
    ///   the UI to re-center and scale the image.
    /// * **Auto-Adjustment:** Sets `auto_adjust_if_not_set` to true, which typically triggers
    ///   automatic contrast or leveling if no previous settings exist.
    /// * **Task Dispatch:** Sends a `DrawingTask` containing both `UpdateImageViewport`
    ///   and `UpdateHistoSliders` jobs to all worker threads.
    pub fn update_active_series_in_project(&self, selected_series: &i32) {
        {
            let mut project = self.app_state.get_project_write();
            project.set_active_series(selected_series);
        } // write lock dropped before sync_image_meta_to_slint acquires read lock
        if self
            .image_meta_controller
            .sync_image_meta_to_slint()
            .is_err()
        {
            warn!("Failed to sync image metadata to Slint after series change");
        }
        self.viewport_controller.trigger_new_series_redraw();
    }

    /// Starts or stops timer-driven T-stack playback based on the Slint side's
    /// `is_play_active` state, which is already flipped by the time this runs.
    ///
    /// This must be the *only* place `on_toggle_play` is ever registered
    /// (see `attach_callbacks`) - it used to re-register a fresh
    /// `on_toggle_play` handler on every call instead of just running this
    /// logic directly, which replaced the handler installed by
    /// `attach_callbacks` before it had done anything, so the very first
    /// Play click of every session was silently swallowed and only took
    /// effect on the second click.
    pub fn handle_play_button_toggled(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            warn!("UI not available for play button toggle");
            return;
        };
        let state = ui.global::<ChannelState>();
        let is_playing = state.get_is_play_active();

        // Always stop the old timer first to clear previous intervals
        self.playback_timer.stop();

        if is_playing {
            let inner_handle = self.ui.clone();

            // Get speed from UI (e.g., 20.0)
            let hz = state.get_play_back_speed_hz() as f32;
            let millis = if hz > 0.0 {
                (1000.0 / hz) as u64
            } else {
                1000 // Default to 1Hz if something is wrong
            };

            self.playback_timer.start(
                TimerMode::Repeated,
                std::time::Duration::from_millis(millis),
                move || {
                    if let Some(ui_tick) = inner_handle.upgrade() {
                        let viewport = ui_tick.global::<ViewportSlintState>();
                        if viewport.get_high_res_ready() {}
                        let state = ui_tick.global::<ChannelState>();
                        let meta = ui_tick.global::<ImageMetaData>();

                        let current = state.get_selected_t_stack();
                        let total = meta.get_nr_t_stacks();

                        if total > 0 {
                            state.set_selected_t_stack((current + 1) % total);
                            state.invoke_trigger_channel_state_changed();
                        }
                    }
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::{project_with_one_image, test_ui_state_with_project};
    use crate::editor::viewport_cache::ViewportCache;

    fn make_controller() -> (Arc<UiState>, Arc<ViewportImageController>) {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let viewport_cache = Arc::new(ViewportCache::new(ui_state.clone()));
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_cache.clone(),
        ));
        let histogram_controller = Arc::new(HistogramController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_meta_controller = Arc::new(ImageMetaController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let controller = Arc::new(ViewportImageController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller,
            viewport_cache,
            histogram_controller,
            image_meta_controller,
        ));
        (ui_state, controller)
    }

    // -- update_viewport_size_in_viewport_state ----------------------------------

    #[test]
    fn update_viewport_size_stores_the_given_dimensions() {
        let (_, controller) = make_controller();

        controller
            .update_viewport_size_in_viewport_state(800.0, 600.0)
            .unwrap();

        let state = controller
            .viewport_controller
            .viewport_state
            .read()
            .unwrap();
        assert_eq!(state.viewport_width, 800.0);
        assert_eq!(state.viewport_height, 600.0);
    }

    // -- update_viewport_zoom_in_viewport_state ----------------------------------

    #[test]
    fn update_viewport_zoom_stores_zoom_and_offset() {
        let (_, controller) = make_controller();

        controller
            .update_viewport_zoom_in_viewport_state(2.5, 10.0, -5.0)
            .unwrap();

        let state = controller
            .viewport_controller
            .viewport_state
            .read()
            .unwrap();
        assert_eq!(state.zoom, 2.5);
        assert_eq!(state.offset_x, 10.0);
        assert_eq!(state.offset_y, -5.0);
    }

    // -- update_viewport_position_in_viewport_state ------------------------------

    #[test]
    fn update_viewport_position_stores_the_offset_and_leaves_zoom_untouched() {
        let (_, controller) = make_controller();
        controller
            .update_viewport_zoom_in_viewport_state(3.0, 0.0, 0.0)
            .unwrap();

        controller
            .update_viewport_position_in_viewport_state(42.0, -7.0)
            .unwrap();

        let state = controller
            .viewport_controller
            .viewport_state
            .read()
            .unwrap();
        assert_eq!(state.offset_x, 42.0);
        assert_eq!(state.offset_y, -7.0);
        assert_eq!(state.zoom, 3.0, "position update must not touch zoom");
    }

    // -- update_mouse_position_in_viewport_state ---------------------------------

    #[test]
    fn update_mouse_position_stores_the_given_coordinates() {
        let (_, controller) = make_controller();

        controller.update_mouse_position_in_viewport_state(123.0, 456.0);

        let state = controller
            .viewport_controller
            .viewport_state
            .read()
            .unwrap();
        assert_eq!(state.mouse_pos_x, 123.0);
        assert_eq!(state.mouse_pos_y, 456.0);
    }

    // -- update_channel_options_in_project ----------------------------------------

    fn channel(idx: i32, active: bool) -> ChannelInfo {
        ChannelInfo {
            name: "ch".into(),
            active,
            idx,
            color: slint::Color::from_rgb_u8(0, 0, 0),
            emission_wave_length: 0.0,
            wavelength_overridden: false,
        }
    }

    #[test]
    fn in_grayscale_switching_a_channel_on_hides_the_other() {
        let (ui_state, controller) = make_controller();
        controller.update_channel_options_in_project(
            vec![channel(0, true), channel(1, false)],
            0,
            0,
            IntensityProjection::SingleStack,
            1.0,
        );
        ui_state.get_project_write().set_grayscale(true);

        controller.update_channel_options_in_project(
            vec![channel(0, true), channel(1, true)],
            0,
            0,
            IntensityProjection::SingleStack,
            1.0,
        );

        let visibilities = ui_state.get_project().get_image_channel_visibilities();
        assert_eq!(visibilities.get(&0), Some(&false));
        assert_eq!(
            visibilities.get(&1),
            Some(&true),
            "the one just switched on"
        );
    }

    #[test]
    fn update_channel_options_stores_channel_visibility() {
        let (ui_state, controller) = make_controller();

        controller.update_channel_options_in_project(
            vec![channel(0, true), channel(1, false)],
            0,
            0,
            IntensityProjection::SingleStack,
            1.0,
        );

        let project = ui_state.get_project();
        let visibilities = project.get_image_channel_visibilities();
        assert_eq!(visibilities.get(&0), Some(&true));
        assert_eq!(visibilities.get(&1), Some(&false));
    }

    #[test]
    fn update_channel_options_maps_every_projection_kind_to_its_z_stack_handling() {
        let cases = [
            (
                IntensityProjection::SingleStack,
                ZStackHandling::SingleStack,
                true,
            ),
            (
                IntensityProjection::AllStacks,
                ZStackHandling::AllStacks,
                true,
            ),
            (
                IntensityProjection::Max,
                ZStackHandling::MaxIntensity,
                false,
            ),
            (
                IntensityProjection::Min,
                ZStackHandling::MinIntensity,
                false,
            ),
            (
                IntensityProjection::Avg,
                ZStackHandling::AvgIntensity,
                false,
            ),
            (
                IntensityProjection::Sum,
                ZStackHandling::SumIntensity,
                false,
            ),
            (
                IntensityProjection::Middle,
                ZStackHandling::TakeTheMiddle,
                false,
            ),
        ];

        for (projection, expected_handling, expects_range) in cases {
            let (ui_state, controller) = make_controller();

            controller.update_channel_options_in_project(vec![], 3, 0, projection, 1.0);

            let project = ui_state.get_project();
            let z = project.get_z_stack().expect("z-stack settings must be set");
            assert_eq!(z.z_projection, expected_handling);
            if expects_range {
                assert_eq!(z.z_range, Some(3..=3));
            } else {
                assert_eq!(z.z_range, None);
            }
        }
    }

    #[test]
    fn update_channel_options_stores_playback_speed_and_t_stack() {
        let (ui_state, controller) = make_controller();

        controller.update_channel_options_in_project(
            vec![],
            0,
            7,
            IntensityProjection::SingleStack,
            2.5,
        );

        let project = ui_state.get_project();
        let t = project.get_t_stack().expect("t-stack settings must be set");
        assert_eq!(t.t_stack, 7);
        assert_eq!(t.playback_speed, 2.5);
    }

    // -- update_active_channel_in_project ------------------------------------------

    #[test]
    fn update_active_channel_stores_the_selected_channel_index() {
        let (ui_state, controller) = make_controller();

        controller.update_active_channel_in_project(&1);

        let project = ui_state.get_project();
        assert_eq!(project.get_selected_image_channel_idx(), 1);
    }

    // -- update_active_series_in_project --------------------------------------------

    #[test]
    fn update_active_series_selects_the_series_for_every_image() {
        use evanalyzer_app::project::ImageEntryExt;
        let (ui_state, controller) = make_controller();

        controller.update_active_series_in_project(&2);

        let project = ui_state.get_project();
        assert_eq!(project.images.settings.selected_series, Some(2));
        for entry in project.images.list.values() {
            assert_eq!(entry.active_series(&project.images.settings), 2);
        }
    }

    // -- sync_pixel_info_throttled / sync_actual_mouse_position_information_to_slint

    #[test]
    fn sync_actual_mouse_position_information_to_slint_without_cached_high_res_data_is_a_no_op() {
        let (_, controller) = make_controller();
        // `viewport_cache.active_high_res_data` is `None` in a fresh cache -
        // must return early rather than panic.
        controller.sync_actual_mouse_position_information_to_slint();
    }

    #[test]
    fn sync_pixel_info_throttled_does_not_panic_on_first_or_repeated_calls() {
        let (_, controller) = make_controller();
        controller.sync_pixel_info_throttled();
        controller.sync_pixel_info_throttled();
    }

    // -- attach_callbacks (live AppWindow) --------------------------------------

    use crate::editor::test_support::test_ui_windows;

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
    ) -> (Arc<UiState>, Arc<ViewportImageController>) {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let viewport_cache = Arc::new(ViewportCache::new(ui_state.clone()));
        let histogram_controller = Arc::new(HistogramController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_meta_controller = Arc::new(ImageMetaController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let controller = Arc::new(ViewportImageController::new(
            ui,
            ui_state.clone(),
            viewport_controller,
            viewport_cache,
            histogram_controller,
            image_meta_controller,
        ));
        (ui_state, controller)
    }

    #[test]
    fn attach_callbacks_first_play_toggle_starts_the_timer_immediately() {
        // Regression test: `handle_play_button_toggled` used to re-register a
        // fresh `on_toggle_play` handler on every call instead of running its
        // playback logic directly - which replaced the handler
        // `attach_callbacks` had just installed before it ever started a
        // timer, so the *first* Play click of every session silently did
        // nothing (it took a second click to actually start playback).
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        let channel_state = ui.global::<ChannelState>();
        ui.global::<ImageMetaData>().set_nr_t_stacks(5);
        // Mirrors image_control_panel.slint: the property is flipped before
        // the callback fires.
        channel_state.set_is_play_active(true);
        channel_state.invoke_toggle_play(true);

        assert!(
            controller.playback_timer.running(),
            "the first Play toggle must start the timer immediately"
        );
    }

    #[test]
    fn attach_callbacks_toggling_play_off_stops_the_timer() {
        let (ui, _results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        let channel_state = ui.global::<ChannelState>();
        ui.global::<ImageMetaData>().set_nr_t_stacks(5);
        channel_state.set_is_play_active(true);
        channel_state.invoke_toggle_play(true);
        assert!(controller.playback_timer.running());

        channel_state.set_is_play_active(false);
        channel_state.invoke_toggle_play(false);
        assert!(
            !controller.playback_timer.running(),
            "toggling play off must stop the timer"
        );
    }
}
