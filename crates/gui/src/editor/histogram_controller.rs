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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::{project_with_one_image, test_ui_state, test_ui_state_with_project};

    fn make_controller(ui_state: Arc<UiState>) -> Arc<HistogramController> {
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        Arc::new(HistogramController::new(
            slint::Weak::default(),
            ui_state,
            viewport_controller,
        ))
    }

    #[test]
    fn update_histogram_settings_writes_the_active_channels_histogram() {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let controller = make_controller(ui_state.clone());

        controller.update_histogram_settings_in_project(0.1, 0.9, 0.0, 1.0);

        let project = ui_state.get_project();
        let hist = project
            .get_histograms_from_selected_channel()
            .expect("channel 0 must now have histogram settings");
        assert_eq!(hist.min, 0.1);
        assert_eq!(hist.max, 0.9);
        assert_eq!(hist.min_limit, 0.0);
        assert_eq!(hist.max_limit, 1.0);
    }

    #[test]
    fn update_histogram_settings_without_a_current_image_does_not_panic() {
        // No image is set as "current" in this fixture - the write must be a
        // silent no-op (matches `with_current_series_mut` returning `None`),
        // not a panic.
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());

        controller.update_histogram_settings_in_project(0.1, 0.9, 0.0, 1.0);

        let project = ui_state.get_project();
        assert!(project.get_histograms_from_selected_channel().is_none());
    }

    #[test]
    fn sync_histogram_settings_to_slint_does_not_panic_with_or_without_settings() {
        // Without a current image (no histogram settings to report) and with
        // one (the normal case) - the pre-`invoke_from_event_loop` half of
        // this method (project/state reads) must not panic either way, even
        // though the dead `ui` means the actual Slint push never runs.
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());
        controller.sync_histogram_settings_to_slint();

        let ui_state = test_ui_state_with_project(project_with_one_image());
        let controller = make_controller(ui_state.clone());
        controller.update_histogram_settings_in_project(0.0, 1.0, 0.0, 1.0);
        controller.sync_histogram_settings_to_slint();
    }
}
