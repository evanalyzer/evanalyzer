use crate::UiState;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::viewport_cache::ViewportCache;
use crate::editor::viewport_controller::ViewportController;
use crate::{
    AppWindow, ChannelInfo, ChannelState, ImageMetaData, ImagePixelInfo, IntensityProjection,
    ViewportState as ViewportSlintState,
};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::{
    TStackHandling, TStackSettings, ZStackHandling, ZStackSettings,
};
use evanalyzer_core::ImageContainer;
use log::warn;
use slint::{ComponentHandle, Model, Timer, TimerMode};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(crate) struct ViewportImageController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) playback_timer: Timer,
    pub(crate) redraw_debounce_timer: Timer,
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
            viewport_controller,
            viewport_cache,
            histogram_controller,
            image_meta_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
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
                    manager.sync_actual_mouse_position_information_to_slint();
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
                    manager.clone().handle_play_button_toggled();
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

        // Trigger Low-Res IMMEDIATELY for smoothness
        self.viewport_controller.trigger_redraw_low_res();

        // Debounce the High-Res update
        self.redraw_debounce_timer.stop(); // Cancel any existing pending high-res task
        let self_in = self.clone();
        self.redraw_debounce_timer.start(
            TimerMode::SingleShot,
            std::time::Duration::from_millis(150),
            move || {
                self_in.viewport_controller.trigger_redraw_low_res_and_high_res();
                self_in.viewport_controller.trigger_image_redraw_rois();
            },
        );

        Ok(())
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

        // Trigger Low-Res IMMEDIATELY for smoothness
        self.viewport_controller.trigger_redraw_low_res();

        // Debounce the High-Res update
        self.redraw_debounce_timer.stop(); // Cancel any existing pending high-res task
        let self_in = self.clone();
        self.redraw_debounce_timer.start(
            TimerMode::SingleShot,
            std::time::Duration::from_millis(150),
            move || {
                self_in.viewport_controller.trigger_redraw_low_res_and_high_res();
                self_in.viewport_controller.trigger_image_redraw_rois();
            },
        );

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
        // Trigger Low-Res IMMEDIATELY for smoothness
        self.viewport_controller.trigger_redraw_low_res();

        // Debounce the High-Res update
        self.redraw_debounce_timer.stop(); // Cancel any existing pending high-res task
        let self_in = self.clone();
        self.redraw_debounce_timer.start(
            TimerMode::SingleShot,
            std::time::Duration::from_millis(150),
            move || {
                self_in.viewport_controller.trigger_redraw_low_res_and_high_res();
                self_in.viewport_controller.trigger_image_redraw_rois();
            },
        );

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
                if let ImageContainer::F32Gray(img) = &*channel.image {
                    if let Some(&raw_val) = img.as_slice().get(idx) {
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
            slint::invoke_from_event_loop(move || {
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

        // Store the new settings to the project
        project.set_global_preferences(&channel_visibility);

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

    pub fn handle_play_button_toggled(self: Arc<Self>) {
        let ui_weak = self.ui.clone();
        let ui = ui_weak.upgrade().expect("Failed to upgrade UI handle");

        ui.global::<ChannelState>()
            .on_toggle_play(move |_play_active| {
                let Some(ui) = ui_weak.upgrade() else {
                    warn!("UI not available for play button toggle");
                    return;
                };
                let state = ui.global::<ChannelState>();
                let is_playing = state.get_is_play_active();

                // Always stop the old timer first to clear previous intervals
                self.playback_timer.stop();

                if is_playing {
                    let inner_handle = ui_weak.clone();

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
            });
    }
}
