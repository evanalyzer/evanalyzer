use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::{AppWindow, HistogramState};
use evanalyzer_app::extensions::project_ext::ProjectExt;
use log::warn;
use slint::ComponentHandle;
use std::sync::Arc;

pub struct HistogramController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl HistogramController {
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
            // Apply image filter
            let manager = Arc::clone(self);
            let debounce_timer = slint::Timer::default();
            ui.global::<HistogramState>().on_histogram_adjusted(
                move |min, max, min_limit, max_limit| {
                    let manager_in = manager.clone();
                    debounce_timer.start(
                        slint::TimerMode::SingleShot,
                        std::time::Duration::from_millis(5),
                        move || {
                            manager_in.update_histogram_settings_in_project(
                                min, max, min_limit, max_limit,
                            );
                            manager_in.viewport_controller.trigger_image_redraw();
                        },
                    );
                },
            );

            // Histogram auto adjust clicked^
            let manager = Arc::clone(self);
            ui.global::<HistogramState>()
                .on_auto_adjust_clicked(move || {
                    manager
                        .viewport_controller
                        .trigger_image_redraw_with_auto_adjust();
                });
        }
    }

    /// Updates the project's histogram scaling and clipping parameters.
    ///
    /// This method synchronizes the UI-driven histogram bounds with the underlying
    /// project state. It updates the active viewing window (min/max) and the
    /// hardware or data limits (min_limit/max_limit).
    ///
    /// # Arguments
    /// * `min` - The current lower bound of the visible histogram range.
    /// * `max` - The current upper bound of the visible histogram range.
    /// * `min_limit` - The absolute minimum intensity value possible for the dataset.
    /// * `max_limit` - The absolute maximum intensity value possible for the dataset.
    ///
    /// # Behavior
    /// 1. Persists the new bounds to the `AppState`.
    /// 2. Typically triggers a re-normalization of the Viewport image to ensure
    ///    the visual contrast matches the new histogram settings.
    ///
    /// # Threading
    /// Updates the state synchronously. If a high-resolution re-render is required
    /// based on these new settings, it should be dispatched to a background thread.
    pub fn update_histogram_settings_in_project(
        self: &Arc<Self>,
        min: f32,
        max: f32,
        min_limit: f32,
        max_limit: f32,
    ) {
        self.app_state
            .get_project_write()
            .set_image_histogram_settings_for_active_channel(min, max, min_limit, max_limit);
    }

    pub fn sync_histogram_settings_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();

        let histogram_settings = self
            .app_state
            .get_project()
            .get_histograms_from_selected_channel()
            .cloned();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let Some(task) = histogram_settings else {
                    warn!(
                        "No histogram settings found for the selected channel. Cannot sync to Slint."
                    );
                    return;
                };
                let range = (task.max_limit - task.min_limit).max(0.001);

                let hist_state = ui_ready.global::<HistogramState>();
                // Set the outer limits of the slider track
                hist_state.set_histogram_min_lim(task.min_limit);
                hist_state.set_histogram_max_lim(task.max_limit);

                // Round to 3 decimal places and convert to a Slint SharedString
                let min_txt = format!("{:.3}", task.min_limit);
                let max_txt = format!("{:.3}", task.max_limit);

                hist_state.set_histogram_min_lim_txt(min_txt.into());
                hist_state.set_histogram_max_lim_txt(max_txt.into());

                // Calculate relative handle positions (0.0 to 1.0) for the UI sliders
                // Formula: rel = (abs - min_lim) / range
                let min_rel = (task.min - task.min_limit) / range;
                let max_rel = (task.max - task.min_limit) / range;

                hist_state.set_histogram_min_val(min_rel);
                hist_state.set_histogram_max_val(max_rel);

                // Set absolute labels for text displays
                hist_state.set_histogram_min_val_abs(task.min);
                hist_state.set_histogram_max_val_abs(task.max);
            }
        }) {
            warn!("Failed to sync histogram settings to Slint: {}", e);
        }
    }
}
