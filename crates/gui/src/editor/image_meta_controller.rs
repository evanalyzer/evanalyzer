use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::color_from_rgb;
use crate::helper::size_formater::format_bits;
use crate::{
    AppWindow, ChannelInfo, ChannelState, ImageMetaData, IntensityProjection, SeriesInfo,
    WavelengthOption,
};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::utils::wavelength_to_rgb_float;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use log::warn;
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Emission wavelengths (nm) offered as colour tiles for a channel - spread
/// over the visible range the viewer can draw (see `wavelength_to_rgb_float`:
/// 420-700 nm, with exact pure blue/green/red at 450/532/635 nm).
const WAVELENGTH_OPTIONS: [f32; 11] = [
    420.0, 450.0, 470.0, 490.0, 532.0, 550.0, 570.0, 580.0, 600.0, 620.0, 635.0,
];

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

            // Channel colour chooser: the tiles, and setting/resetting a
            // channel's emission wavelength for the project.
            let options: Vec<WavelengthOption> = WAVELENGTH_OPTIONS
                .iter()
                .map(|&nm| WavelengthOption {
                    nm,
                    color: color_from_rgb(wavelength_to_rgb_float(nm)),
                })
                .collect();
            ui.global::<ImageMetaData>()
                .set_wavelength_options(ModelRc::new(VecModel::from(options)));

            let manager = Arc::clone(self);
            ui.global::<ImageMetaData>()
                .on_emission_wave_length_changed(move |channel_idx, nm| {
                    manager.set_channel_emission_wave_length(channel_idx, Some(nm));
                });

            let manager = Arc::clone(self);
            ui.global::<ImageMetaData>()
                .on_reset_emission_wave_length(move |channel_idx| {
                    manager.set_channel_emission_wave_length(channel_idx, None);
                });
        }
    }

    /// Sets (`Some`) or clears (`None`, back to the image's own value) the
    /// project's emission wavelength for one channel, then shows the new
    /// colour in the channel list and the viewer.
    pub(crate) fn set_channel_emission_wave_length(&self, channel_idx: i32, nm: Option<f32>) {
        {
            let mut project = self.app_state.get_project_write();
            match nm {
                Some(nm) => project.set_global_emission_wavel_length(channel_idx, nm),
                None => project.reset_global_emission_wavel_length(channel_idx),
            }
        }
        self.app_state.mark_dirty();
        if let Err(e) = self.sync_image_meta_to_slint() {
            warn!("Could not refresh the channel list: {e}");
        }
        self.viewport_controller.trigger_image_redraw();
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

        let image_meta = self.app_state.get_image_meta(&path)?;

        // Per channel: the project's emission wavelength (its own setting if
        // the user changed it, else the image's) and whether it was changed.
        let channel_wavelengths: BTreeMap<i32, (f32, bool)> = {
            let project = self.app_state.get_project();
            image_meta
                .series
                .get(&selected_series)
                .map(|series| {
                    series
                        .channels
                        .keys()
                        .map(|&idx| {
                            let overridden = project
                                .images
                                .settings
                                .channels
                                .get(&idx)
                                .is_some_and(|c| c.emission_wave_length.is_some());
                            (idx, (project.get_emission_wave_length(idx), overridden))
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let ui_weak = self.ui.clone();

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                warn!("Cannot update image meta data - UI upgrade failed");
                return;
            };

            // --- Per-series info (idx + first-resolution dimensions), for
            // the series picker dropdown. Built with filter_map rather than
            // the previous early-`return`-on-missing-entry loop, which used
            // to abort this whole closure (silently skipping every UI update
            // below it, not just the series list) the moment one series
            // lacked a resolution-0 entry.
            let series_list: Vec<SeriesInfo> = image_meta
                .series
                .iter()
                .filter_map(|(&idx, series_info)| {
                    series_info
                        .resolutions
                        .get(&0)
                        .map(|pyramid_info| SeriesInfo {
                            idx,
                            width: pyramid_info.width as i32,
                            height: pyramid_info.height as i32,
                        })
                })
                .collect();

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
                    emission_wave_length: 0.0,
                    wavelength_overridden: false,
                });
            }
            channel_copy.truncate(series_info.channels.len());

            let channels: Vec<ChannelInfo> = series_info
                .channels
                .iter()
                .filter_map(|(idx, channel)| {
                    let (project_nm, overridden) = channel_wavelengths
                        .get(idx)
                        .copied()
                        .unwrap_or((0.0, false));
                    // The project's value; the file's own as a last resort
                    // (e.g. a channel the project doesn't list yet).
                    let nm = if project_nm > 0.0 {
                        project_nm
                    } else {
                        channel.emission_wave_length
                    };
                    channel_copy.get(*idx as usize).map(|_| ChannelInfo {
                        name: channel.name.clone().into(),
                        active: *channel_visibilities.get(idx).unwrap_or(&true),
                        idx: *idx,
                        color: color_from_rgb(wavelength_to_rgb_float(nm)),
                        emission_wave_length: nm,
                        wavelength_overridden: overridden,
                    })
                })
                .collect();

            let is_rgb = pyramid_info.is_rgb
                && series_info.nr_c_stacks == 3
                && channels[0].name == "Red"
                && channels[1].name == "Green"
                && channels[2].name == "Blue";

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
            image_meta_ui.set_is_rgb(is_rgb);
            image_meta_ui.set_nr_c_stacks(series_info.nr_c_stacks);
            image_meta_ui.set_nr_t_stacks(series_info.nr_t_stacks);
            image_meta_ui.set_nr_z_stacks(series_info.nr_z_stacks);
            image_meta_ui.set_nr_series(image_meta.series.len() as i32);
            image_meta_ui
                .set_series_list(std::rc::Rc::new(slint::VecModel::from(series_list)).into());

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

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;
    use std::sync::Arc;

    fn make_controller() -> (Arc<UiState>, ImageMetaController) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let controller = ImageMetaController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller,
        );
        (ui_state, controller)
    }

    #[test]
    fn set_manual_pixel_sizes_writes_the_global_pixel_size_override() {
        let (ui_state, controller) = make_controller();

        controller.set_manual_pixel_sizes(1.5, 2.5, 3.5);

        let project = ui_state.get_project();
        let sizes = project.get_pixel_sizes();
        assert_eq!(sizes.x, 1.5);
        assert_eq!(sizes.y, 2.5);
        assert_eq!(sizes.z, 3.5);
    }

    #[test]
    fn reset_manual_pixel_sizes_clears_a_previously_set_override() {
        let (ui_state, controller) = make_controller();
        controller.set_manual_pixel_sizes(9.0, 9.0, 9.0);

        controller.reset_manual_pixel_sizes();

        // No image is loaded in this fixture project, so with the override
        // cleared `get_pixel_sizes` falls back to its hardcoded default.
        let project = ui_state.get_project();
        let sizes = project.get_pixel_sizes();
        assert_eq!(sizes.x, 1.0);
        assert_eq!(sizes.y, 1.0);
        assert_eq!(sizes.z, 1.0);
    }

    // -- sync_image_meta_to_slint --------------------------------------------------

    #[test]
    fn sync_image_meta_to_slint_without_a_current_image_returns_ok() {
        let (_, controller) = make_controller();

        assert!(controller.sync_image_meta_to_slint().is_ok());
    }

    #[test]
    fn sync_image_meta_to_slint_with_a_missing_image_file_returns_an_error() {
        let ui_state = crate::editor::test_support::test_ui_state_with_project(
            crate::editor::test_support::project_with_one_image(),
        );
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let controller =
            ImageMetaController::new(slint::Weak::default(), ui_state, viewport_controller);

        // `project_with_one_image()`'s "img.tif" doesn't exist on disk -
        // `get_image_meta` must surface that as an `Err`, not panic.
        assert!(controller.sync_image_meta_to_slint().is_err());
    }

    // -- with a window (UI updates applied via the test UI queue) ------------

    use crate::editor::test_support::{
        project_with_fixture_image, test_ui_state_with_project, test_ui_windows,
    };
    use crate::helper::ui_thread::drain_ui_queue;

    fn with_window(
        project: evanalyzer_app::project::ProjectWithRuntime,
    ) -> (AppWindow, Arc<UiState>, Arc<ImageMetaController>) {
        let (ui, _results_ui) = test_ui_windows();
        let ui_state = test_ui_state_with_project(project);
        let viewport_controller = Arc::new(ViewportController::new(ui.as_weak(), ui_state.clone()));
        let controller = Arc::new(ImageMetaController::new(
            ui.as_weak(),
            ui_state.clone(),
            viewport_controller,
        ));
        controller.attach_callbacks();
        (ui, ui_state, controller)
    }

    #[test]
    fn sync_image_meta_shows_the_real_images_dimensions_and_channels() {
        let (ui, ui_state, controller) = with_window(project_with_fixture_image());
        let meta = ui_state
            .get_image_meta(&crate::editor::test_support::fixture_image_path())
            .unwrap();
        let series = meta.series.get(&0).unwrap();

        controller.sync_image_meta_to_slint().unwrap();
        drain_ui_queue();

        let meta_ui = ui.global::<ImageMetaData>();
        assert_eq!(meta_ui.get_image_name(), meta.name.as_str());
        assert_eq!(
            meta_ui.get_dimensions_str(),
            format!(
                "{}x{}x{}",
                series.nr_c_stacks, series.nr_z_stacks, series.nr_t_stacks
            )
            .as_str()
        );
        assert_eq!(meta_ui.get_nr_series(), meta.series.len() as i32);
        assert_eq!(meta_ui.get_series_list().row_count(), meta.series.len());
        assert!(meta_ui.get_size().ends_with(" px"));
        assert!(meta_ui.get_pixel_type().ends_with(" bits"));
        // The fixture project overrides pixel sizes with 0.5 x 0.5 x 1.
        assert_eq!(meta_ui.get_pixel_size_x(), 0.5);
        assert_eq!(meta_ui.get_pixel_size_str(), "0.5x0.5x1.0 nm/px");

        let channels = ui.global::<ChannelState>().get_channels();
        assert_eq!(channels.row_count(), series.channels.len());
        // The fixture is an RGB image: three channels named Red/Green/Blue.
        assert!(meta_ui.get_is_rgb());
        assert_eq!(
            ui.global::<ChannelState>().get_intensity_projection(),
            IntensityProjection::SingleStack
        );
    }

    #[test]
    fn sync_image_meta_shows_every_z_projection_mode() {
        use evanalyzer_cfg::settings::images_settings::ZStackSettings;
        for (handling, shown) in [
            (ZStackHandling::AllStacks, IntensityProjection::AllStacks),
            (ZStackHandling::MaxIntensity, IntensityProjection::Max),
            (ZStackHandling::MinIntensity, IntensityProjection::Min),
            (ZStackHandling::AvgIntensity, IntensityProjection::Avg),
            (ZStackHandling::SumIntensity, IntensityProjection::Sum),
            (ZStackHandling::TakeTheMiddle, IntensityProjection::Middle),
        ] {
            let mut project = project_with_fixture_image();
            project.settings.images.settings.z_stack = Some(ZStackSettings {
                z_projection: handling,
                ..Default::default()
            });
            let (ui, _ui_state, controller) = with_window(project);
            controller.sync_image_meta_to_slint().unwrap();
            drain_ui_queue();
            assert_eq!(
                ui.global::<ChannelState>().get_intensity_projection(),
                shown
            );
        }
    }

    #[test]
    fn a_selected_series_the_image_lacks_leaves_the_ui_alone() {
        let mut project = project_with_fixture_image();
        project.images.settings.selected_series = Some(99);
        let (ui, _ui_state, controller) = with_window(project);
        ui.global::<ImageMetaData>()
            .set_image_name("untouched".into());
        controller.sync_image_meta_to_slint().unwrap();
        drain_ui_queue();
        assert_eq!(ui.global::<ImageMetaData>().get_image_name(), "untouched");
    }

    #[test]
    fn pixel_size_callbacks_override_and_reset_the_shown_sizes() {
        let (ui, ui_state, _controller) = with_window(project_with_fixture_image());
        let meta_ui = ui.global::<ImageMetaData>();

        meta_ui.invoke_pixel_size_changed(2.0, 3.0, 4.0);
        drain_ui_queue();
        assert_eq!(meta_ui.get_pixel_size_x(), 2.0);
        assert_eq!(meta_ui.get_pixel_size_z(), 4.0);
        assert_eq!(meta_ui.get_pixel_size_str(), "2.0x3.0x4 nm/px");
        assert_eq!(ui_state.get_project().get_pixel_sizes().y, 3.0);

        meta_ui.invoke_reset_pixel_sizes();
        drain_ui_queue();
        // Back to the image's own series settings (0.5 x 0.5 x 1).
        assert_eq!(meta_ui.get_pixel_size_x(), 0.5);
        assert_eq!(ui_state.get_project().get_pixel_sizes().x, 0.5);
    }

    // -- channel colour chooser ------------------------------------------------

    #[test]
    fn the_chooser_offers_one_coloured_tile_per_wavelength() {
        let (ui, _ui_state, _controller) = with_window(project_with_fixture_image());
        let options = ui.global::<ImageMetaData>().get_wavelength_options();
        assert_eq!(options.row_count(), WAVELENGTH_OPTIONS.len());
        let green = options.iter().find(|o| o.nm == 532.0).unwrap();
        assert_eq!(green.color, slint::Color::from_rgb_u8(0, 255, 0));
    }

    fn channel_row(ui: &AppWindow, idx: i32) -> ChannelInfo {
        ui.global::<ChannelState>()
            .get_channels()
            .iter()
            .find(|c| c.idx == idx)
            .unwrap()
    }

    #[test]
    fn choosing_a_channel_wavelength_overrides_it_and_reset_restores_it() {
        let project = crate::editor::test_support::project_with_image_file(
            crate::editor::test_support::grayscale_image_path(),
        );
        let (ui, ui_state, controller) = with_window(project);
        controller.sync_image_meta_to_slint().unwrap();
        drain_ui_queue();
        // The project's own value for channel 0 (from the image entry).
        let row = channel_row(&ui, 0);
        assert_eq!(row.emission_wave_length, 488.0);
        assert!(!row.wavelength_overridden);

        ui.global::<ImageMetaData>()
            .invoke_emission_wave_length_changed(0, 532.0);
        drain_ui_queue();
        let row = channel_row(&ui, 0);
        assert_eq!(row.emission_wave_length, 532.0);
        assert!(row.wavelength_overridden);
        assert_eq!(row.color, slint::Color::from_rgb_u8(0, 255, 0));
        assert_eq!(ui_state.get_project().get_emission_wave_length(0), 532.0);
        assert!(ui_state.is_dirty(), "saved with the project");

        ui.global::<ImageMetaData>()
            .invoke_reset_emission_wave_length(0);
        drain_ui_queue();
        let row = channel_row(&ui, 0);
        assert_eq!(row.emission_wave_length, 488.0);
        assert!(!row.wavelength_overridden);
    }
}
