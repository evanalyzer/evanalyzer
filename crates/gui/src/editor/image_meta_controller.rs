use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::color_from_rgb;
use crate::helper::size_formater::format_bits;
use crate::{AppWindow, ChannelInfo, ChannelState, ImageMetaData, IntensityProjection};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::extensions::utils::wavelength_to_rgb_float;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use log::warn;
use slint::{ComponentHandle, Model};
use std::sync::Arc;

pub struct ImageMetaController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl ImageMetaController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
        }
    }

    /// Attach UI callbacks related to image operations.
    ///
    /// This method registers handlers on the global ImagesListState (currently the
    /// `on_image_filter_text_changed` callback) so that UI-driven image filter actions
    /// are propagated to the background manager and the UI is refreshed on the
    /// Slint event loop.
    ///
    /// Behavior:
    /// - Clones required handles (UI and application state) so the closures can be
    ///   stored and invoked later.
    /// - The registered callback captures a worker/project manager and a weak UI
    ///   handle. It schedules work on the Slint event loop using
    ///   `slint::invoke_from_event_loop`.
    /// - Inside the event loop it attempts to upgrade the weak UI handle; if the
    ///   UI still exists it calls `update_image_list_in_sync` on the manager to
    ///   update the image list to reflect the applied filter.
    ///
    /// Notes:
    /// - The function is non-blocking from the caller's perspective; updates are
    ///   dispatched to the event loop.
    /// - If the UI has been dropped the callback is a no-op (the weak upgrade
    ///   fails). Any errors from scheduling are ignored via `.ok()`.
    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // Pixel sizes of image meta manually changed
            let manager = Arc::clone(self);
            ui.global::<ImageMetaData>().on_pixel_size_changed(
                move |pixel_size_x, pixel_size_y, pixel_size_z| {
                    manager.set_manual_pixel_sizes(pixel_size_x, pixel_size_y, pixel_size_z);
                },
            );

            // Reset manuel pixel size settings to image meta default
            let manager = Arc::clone(self);
            ui.global::<ImageMetaData>().on_reset_pixel_sizes(move || {
                manager.reset_manual_pixel_sizes();
            });
        }
    }

    /// Synchronizes image metadata and channel settings from the Rust backend to the Slint UI.
    ///
    /// This method extracts image-specific data (such as dimensions, color space,
    /// and channel configurations) and updates the corresponding Slint globals.
    ///
    /// # Errors
    /// Returns a [`InternalErrors`] if the image metadata cannot be retrieved from
    /// the current project state or if the data format is incompatible.
    ///
    /// # Threading
    /// The extraction logic runs on the caller's thread, while the UI property
    /// updates are dispatched to the main event loop via `slint::invoke_from_event_loop`.
    pub(crate) fn sync_image_meta_to_slint(&self) -> Result<(), InternalErrors> {
        // --- Extract everything from project first, then drop the lock ---
        let (
            image_path,
            selected_series,
            selected_channel,
            channel_visibilities,
            z_proj,
            hz,
            pixel_sizes,
        ) = {
            let project = self.app_state.get_project();

            let image_path = project.get_current_image_path_cloned();
            let selected_series = project.get_selected_series_idx();
            let selected_channel = project.get_selected_image_channel_idx();
            let channel_visibilities = project.get_image_channel_visibilities();

            let z_proj = project
                .get_z_stack()
                .map(|s| s.z_projection.clone())
                .unwrap_or(ZStackHandling::SingleStack);

            let hz = project
                .get_t_stack()
                .map(|s| s.playback_speed as i32)
                .unwrap_or(1);

            let pixel_sizes = project.get_pixel_sizes();

            (
                image_path,
                selected_series,
                selected_channel,
                channel_visibilities,
                z_proj,
                hz,
                pixel_sizes,
            )
        }; // ← lock dropped here

        let Some(path) = image_path else {
            warn!("No image path found in project, cannot sync metadata to UI.");
            return Ok(());
        };

        let reader = self.app_state.get_or_create_reader(&path)?;
        let image_meta = reader.get_image_meta().clone();
        let ui_weak = self.ui.clone();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                warn!("Cannot update image meta data - UI upgrade failed");
                return;
            };

            // --- Series info string ---
            let mut series_info_str = String::new();
            for i in 0..image_meta.series.len() as i32 {
                let Some(series_info) = image_meta.series.get(&i) else {
                    return;
                };
                let Some(pyramid_info) = series_info.resolutions.get(&0) else {
                    return;
                };
                series_info_str.push_str(&format!(
                    "{}: {}x{}\n",
                    i, pyramid_info.width, pyramid_info.height
                ));
            }
            series_info_str.truncate(series_info_str.trim_end_matches('\n').len());

            // --- Selected series ---
            let Some(series_info) = image_meta.series.get(&selected_series) else {
                return;
            };
            let Some(pyramid_info) = series_info.resolutions.get(&0) else {
                return;
            };

            // --- Channels ---
            let ch_state = ui.global::<ChannelState>();
            let mut channel_copy: Vec<ChannelInfo> = ch_state.get_channels().iter().collect();

            // Resize to match actual channel count
            while channel_copy.len() < series_info.channels.len() {
                channel_copy.push(ChannelInfo {
                    name: "Channel".into(),
                    active: true,
                    idx: channel_copy.len() as i32,
                    color: slint::Color::from_rgb_u8(255, 0, 0),
                });
            }
            channel_copy.truncate(series_info.channels.len());

            let channels: Vec<ChannelInfo> = series_info
                .channels
                .iter()
                .filter_map(|(idx, channel)| {
                    channel_copy.get(*idx as usize).map(|_| ChannelInfo {
                        name: channel.name.clone().into(),
                        active: *channel_visibilities.get(idx).unwrap_or(&true),
                        idx: *idx,
                        color: color_from_rgb(wavelength_to_rgb_float(
                            channel.emission_wave_length,
                        )),
                    })
                })
                .collect();

            ch_state.set_channels(std::rc::Rc::new(slint::VecModel::from(channels)).into());

            // --- Image meta ---
            let image_meta_ui = ui.global::<ImageMetaData>();
            image_meta_ui.set_image_name(image_meta.name.clone().into());
            image_meta_ui.set_dimensions_str(
                format!(
                    "{}x{}x{}",
                    series_info.nr_c_stacks, series_info.nr_z_stacks, series_info.nr_t_stacks
                )
                .into(),
            );
            image_meta_ui.set_nr_c_stacks(series_info.nr_c_stacks);
            image_meta_ui.set_nr_t_stacks(series_info.nr_t_stacks);
            image_meta_ui.set_nr_z_stacks(series_info.nr_z_stacks);
            image_meta_ui.set_nr_series(image_meta.series.len() as i32);
            image_meta_ui.set_nr_series_str(series_info_str.into());

            let bits = pyramid_info.width
                * pyramid_info.height
                * pyramid_info.nr_bits as u64
                * pyramid_info.color_channels as u64;

            image_meta_ui.set_storage_size(format_bits(bits).into());
            image_meta_ui
                .set_magnification(format!("x{}", image_meta.objective.magnification).into());
            image_meta_ui
                .set_size(format!("{}x{} px", pyramid_info.width, pyramid_info.height).into());
            image_meta_ui.set_pixel_type(format!("{} bits", pyramid_info.nr_bits).into());

            // --- Pixel size ---
            image_meta_ui.set_pixel_size_x(pixel_sizes.x);
            image_meta_ui.set_pixel_size_y(pixel_sizes.y);
            image_meta_ui.set_pixel_size_z(pixel_sizes.z);
            image_meta_ui.set_pixel_size_str(
                format!(
                    "{:.1}x{:.1}x{:.1} nm/px",
                    pixel_sizes.x, pixel_sizes.y, pixel_sizes.z
                )
                .into(),
            );

            // --- Playback + projection ---
            let state = ui.global::<ChannelState>();
            state.set_selected_series(selected_series);
            state.set_play_back_speed_hz(hz);
            state.set_selected_channel_index(selected_channel);
            state.set_intensity_projection(match z_proj {
                ZStackHandling::SingleStack => IntensityProjection::SingleStack,
                ZStackHandling::AllStacks => IntensityProjection::AllStacks,
                ZStackHandling::MaxIntensity => IntensityProjection::Max,
                ZStackHandling::MinIntensity => IntensityProjection::Min,
                ZStackHandling::AvgIntensity => IntensityProjection::Avg,
                ZStackHandling::SumIntensity => IntensityProjection::Sum,
                ZStackHandling::TakeTheMiddle => IntensityProjection::Middle,
            });
        }) {
            warn!("Failed to enqueue UI update: {:?}", e);
        }

        Ok(())
    }
    /// Manually updates the physical pixel dimensions (nm) for the current project.
    ///
    /// This method overrides any automatically detected metadata and establishes
    /// the new spatial calibration for the image pipeline. All future coordinate
    /// transforms, scale bar renderings, and volumetric calculations will
    /// reference these values.
    ///
    /// # Arguments
    /// * `px` - The physical width of a single pixel (X-axis).
    /// * `py` - The physical height of a single pixel (Y-axis).
    /// * `pz` - The physical depth/spacing between slices (Z-axis).
    ///
    /// # Errors
    /// Returns a [`InternalErrors`] if the provided values are non-positive (zero or negative)
    /// or if the project state is currently locked by another process.
    ///
    /// # Threading
    /// Updates are synchronous to the project state. It is the caller's responsibility
    /// to trigger a UI refresh (e.g., `sync_image_meta_to_slint` or `trigger_new_image_redraw`) after these
    /// values are successfully committed.
    pub(crate) fn set_manual_pixel_sizes(&self, px: f32, py: f32, pz: f32) {
        {
            let mut project = self.app_state.get_project_write();
            project.set_global_pixel_size_settings(px, py, pz);
        }
        self.sync_pixel_size_settings_to_slint();
        self.viewport_controller.sync_scale_bar_to_slint();
    }

    /// Resets pixel dimensions to their original values as defined in the image metadata.
    ///
    /// This method clears any manual overrides established by `pixel_sizes_manually_changed`
    /// and attempts to re-read the native spatial calibration (e.g., from EXIF, TIFF tags,
    /// or proprietary microscope headers).
    ///
    /// # Errors
    /// Returns a [`InternalErrors`] if the original metadata is missing, corrupted,
    /// or if the current project state cannot be accessed.
    ///
    /// # Side Effects
    /// Successful execution will likely invalidate current measurements or scale bars
    /// in the UI, requiring a subsequent call to `sync_image_meta_to_slint`.
    pub(crate) fn reset_manual_pixel_sizes(&self) {
        {
            let mut project = self.app_state.get_project_write();
            project.reset_global_pixel_size_settings();
        }
        self.sync_pixel_size_settings_to_slint();
        self.viewport_controller.sync_scale_bar_to_slint();
    }

    /// Synchronizes current pixel size settings from the shared state to the Slint UI properties.
    ///
    /// This resolves the priority (Global -> Local -> Default) and updates the UI
    /// via `slint::invoke_from_event_loop` to ensure thread safety.
    pub(crate) fn sync_pixel_size_settings_to_slint(&self) {
        let ui_weak = self.ui.clone();
        let project = self.app_state.get_project();
        let pixel_sizes = project.get_pixel_sizes();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                // Pixel size
                let px_size = format!(
                    "{:.1}x{:.1}x{:1} nm/px",
                    pixel_sizes.x, pixel_sizes.y, pixel_sizes.z
                );
                let image_meta_ui = ui_ready.global::<ImageMetaData>();
                image_meta_ui.set_pixel_size_x(pixel_sizes.x);
                image_meta_ui.set_pixel_size_y(pixel_sizes.y);
                image_meta_ui.set_pixel_size_z(pixel_sizes.z);
                image_meta_ui.set_pixel_size_str(px_size.into());
            } else {
                warn!(
                    "Failed to upgrade UI handle in sync_pixel_size_settings_to_slint, cannot update pixel size settings in UI!"
                );
            }
        }) {
            warn!(
                "Failed to enqueue UI update for pixel size settings: {:?}",
                e
            );
        }
    }
}
