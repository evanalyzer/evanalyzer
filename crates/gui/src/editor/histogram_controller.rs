//! # histogram_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::{AppWindow, HistogramState};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::settings::images_settings::HistogramSettings;
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
                            // Comparing, the breakpoint's side: its own window.
                            if manager_in.viewport_controller.histogram_side() == 1 {
                                manager_in
                                    .viewport_controller
                                    .set_breakpoint_histogram(Some(HistogramSettings {
                                        min,
                                        max,
                                        min_limit,
                                        max_limit,
                                    }));
                                return;
                            }
                            manager_in.update_histogram_settings_in_project(
                                min, max, min_limit, max_limit,
                            );
                            manager_in.viewport_controller.trigger_image_redraw();
                        },
                    );
                },
            );

            // Every channel auto-scaled.
            let manager = Arc::clone(self);
            ui.global::<HistogramState>()
                .on_auto_adjust_all_clicked(move || {
                    manager
                        .viewport_controller
                        .trigger_image_redraw_with_auto_adjust_all();
                });

            // The open image's scaling for every image
            let manager = Arc::clone(self);
            ui.global::<HistogramState>()
                .on_apply_to_all_images(move || {
                    manager
                        .app_state
                        .get_project_write()
                        .apply_histograms_to_all_images();
                    manager.app_state.mark_dirty();
                    manager.viewport_controller.trigger_image_redraw();
                });

            // Histogram auto adjust clicked^
            let manager = Arc::clone(self);
            ui.global::<HistogramState>()
                .on_auto_adjust_clicked(move || {
                    if manager.viewport_controller.histogram_side() == 1 {
                        // Auto-adjusted at its next render.
                        manager.viewport_controller.set_breakpoint_histogram(None);
                        return;
                    }
                    manager
                        .viewport_controller
                        .trigger_image_redraw_with_auto_adjust();
                });

            // Log or linear counts - the user's choice, kept.
            let log = !self.app_state.load_app_settings().histogram_linear;
            self.viewport_controller.set_histogram_log(log);
            ui.global::<HistogramState>().set_log_scale(log);
            let manager = Arc::clone(self);
            ui.global::<HistogramState>()
                .on_log_scale_changed(move |log| {
                    manager.viewport_controller.set_histogram_log(log);
                    manager
                        .app_state
                        .update_app_settings(|settings| settings.histogram_linear = !log);
                });

            // Comparing: the original's or the breakpoint's histogram.
            let manager = Arc::clone(self);
            ui.global::<HistogramState>().on_side_changed(move |side| {
                manager
                    .viewport_controller
                    .set_histogram_side(side.clamp(0, 1) as u8);
                manager.sync_histogram_settings_to_slint();
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
        // The breakpoint's side: its own window.
        let histogram_settings = if self.viewport_controller.histogram_side() == 1 {
            self.viewport_controller.breakpoint_histogram()
        } else {
            histogram_settings
        };

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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
    use crate::editor::test_support::{
        project_with_one_image, test_ui_state, test_ui_state_with_project,
    };

    fn make_controller(ui_state: Arc<UiState>) -> Arc<HistogramController> {
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
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

    // -- callbacks on a live window ---------------------------------------------

    use crate::editor::test_support::{test_ui_window, ui_state_with_window};
    use crate::editor::viewport_task::DrawingTask;
    use crate::helper::ui_thread::drain_ui_queue;

    fn advance(ms: u64) {
        i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_millis(ms));
    }

    fn attached() -> (AppWindow, Arc<UiState>, Arc<HistogramController>) {
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, project_with_one_image());
        let viewport_controller = Arc::new(ViewportController::new(
            ui.as_weak(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let controller = Arc::new(HistogramController::new(
            ui.as_weak(),
            ui_state.clone(),
            viewport_controller,
        ));
        controller.attach_callbacks();
        drain_ui_queue();
        take_redraw(&controller);
        (ui, ui_state, controller)
    }

    /// The image redraw waiting - and takes it.
    fn take_redraw(controller: &HistogramController) -> Option<DrawingTask> {
        let tasks = &controller.viewport_controller.drawing_tasks;
        tasks.low_res_task.task_request.0.lock().unwrap().take();
        tasks.high_res_task.task_request.0.lock().unwrap().take()
    }

    fn window(min: f32, max: f32) -> HistogramSettings {
        HistogramSettings {
            min,
            max,
            min_limit: 0.0,
            max_limit: 1.0,
        }
    }

    fn selected_window(ui_state: &UiState) -> Option<(f32, f32)> {
        ui_state
            .get_project()
            .get_histograms_from_selected_channel()
            .map(|h| (h.min, h.max))
    }

    /// Dragging the handles: only the last position is stored, once the
    /// drag pauses, and the image redrawn.
    #[test]
    fn an_adjusted_histogram_is_stored_after_the_debounce_and_redrawn() {
        let (ui, ui_state, controller) = attached();
        let state = ui.global::<HistogramState>();
        state.invoke_histogram_adjusted(0.1, 0.5, 0.0, 1.0);
        state.invoke_histogram_adjusted(0.2, 0.7, 0.0, 1.0);
        assert_eq!(selected_window(&ui_state), None, "not before the pause");
        assert!(take_redraw(&controller).is_none());

        advance(10);

        assert_eq!(selected_window(&ui_state), Some((0.2, 0.7)));
        let task = take_redraw(&controller).expect("redrawn");
        assert!(!task.auto_adjust_selected && !task.auto_adjust_all);
    }

    /// Comparing, the breakpoint's side: the window is the breakpoint
    /// image's own - the project's stays as it was.
    #[test]
    fn on_the_breakpoint_side_an_adjusted_histogram_is_the_breakpoints() {
        let (ui, ui_state, controller) = attached();
        ui.global::<HistogramState>().invoke_side_changed(1);
        ui.global::<HistogramState>()
            .invoke_histogram_adjusted(0.3, 0.6, 0.0, 1.0);
        advance(10);

        let breakpoint = controller.viewport_controller.breakpoint_histogram();
        assert_eq!(
            breakpoint.map(|h| (h.min, h.max, h.min_limit, h.max_limit)),
            Some((0.3, 0.6, 0.0, 1.0))
        );
        assert_eq!(selected_window(&ui_state), None);
    }

    #[test]
    fn auto_adjust_redraws_auto_adjusting_the_selected_channel() {
        let (ui, _, controller) = attached();
        ui.global::<HistogramState>().invoke_auto_adjust_clicked();
        let task = take_redraw(&controller).expect("redrawn");
        assert!(task.auto_adjust_selected);
        assert!(!task.auto_adjust_all);
    }

    /// The breakpoint's side: its window is dropped, to be auto-adjusted at
    /// its next render; the image's channels are left alone.
    #[test]
    fn auto_adjust_on_the_breakpoint_side_drops_the_breakpoints_window() {
        let (ui, _, controller) = attached();
        let vc = &controller.viewport_controller;
        vc.set_histogram_side(1);
        vc.set_breakpoint_histogram(Some(window(0.1, 0.2)));
        ui.global::<HistogramState>().invoke_auto_adjust_clicked();
        assert!(vc.breakpoint_histogram().is_none());
        assert!(
            take_redraw(&controller).is_none_or(|t| !t.auto_adjust_selected),
            "no auto-adjust of the image"
        );
    }

    #[test]
    fn auto_adjust_all_redraws_auto_adjusting_every_channel() {
        let (ui, _, controller) = attached();
        ui.global::<HistogramState>()
            .invoke_auto_adjust_all_clicked();
        let task = take_redraw(&controller).expect("redrawn");
        assert!(task.auto_adjust_all);
    }

    /// The open image's window becomes the project's: the image's own is
    /// removed, the project marked changed, the image redrawn.
    #[test]
    fn apply_to_all_images_moves_the_window_into_the_project() {
        let (ui, ui_state, controller) = attached();
        controller.update_histogram_settings_in_project(0.2, 0.4, 0.0, 1.0);
        ui_state.clear_dirty();
        ui.global::<HistogramState>().invoke_apply_to_all_images();

        let project = ui_state.get_project();
        let global = project.images.settings.channels[&0]
            .histogram
            .as_ref()
            .expect("the project's window");
        assert_eq!((global.min, global.max), (0.2, 0.4));
        let image_own = project
            .get_selected_image_series()
            .and_then(|s| s.channels[&0].histogram.clone());
        assert!(image_own.is_none(), "the image's own is removed");
        drop(project);
        assert!(ui_state.is_dirty());
        assert!(take_redraw(&controller).is_some());
    }

    /// The user's log/linear choice is applied at start and kept.
    #[test]
    fn the_log_scale_choice_is_applied_at_start_and_saved() {
        let (ui, ui_state, controller) = attached();
        let state = ui.global::<HistogramState>();
        let log = !ui_state.load_app_settings().histogram_linear;
        assert_eq!(state.get_log_scale(), log);
        assert_eq!(controller.viewport_controller.histogram_log(), log);

        state.invoke_log_scale_changed(!log);
        assert_eq!(controller.viewport_controller.histogram_log(), !log);
        assert_eq!(ui_state.load_app_settings().histogram_linear, log);
        assert!(take_redraw(&controller).is_some(), "curves drawn anew");

        // A new window starts with the saved choice.
        let (ui, controller) = attached_on(ui_state);
        assert_eq!(ui.global::<HistogramState>().get_log_scale(), !log);
        assert_eq!(controller.viewport_controller.histogram_log(), !log);
    }

    /// A second controller on `ui_state`'s settings, with a window of its
    /// own.
    fn attached_on(ui_state: Arc<UiState>) -> (AppWindow, Arc<HistogramController>) {
        let ui = test_ui_window();
        let viewport_controller = Arc::new(ViewportController::new(
            ui.as_weak(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let controller = Arc::new(HistogramController::new(
            ui.as_weak(),
            ui_state,
            viewport_controller,
        ));
        controller.attach_callbacks();
        (ui, controller)
    }

    /// Switching sides shows that side's window on the sliders; a side
    /// beyond the breakpoint's is the breakpoint's.
    #[test]
    fn switching_sides_shows_that_sides_window() {
        let (ui, _, controller) = attached();
        controller.update_histogram_settings_in_project(0.1, 0.9, 0.0, 1.0);
        controller
            .viewport_controller
            .set_breakpoint_histogram(Some(window(0.25, 0.5)));
        let state = ui.global::<HistogramState>();

        state.invoke_side_changed(7);
        drain_ui_queue();
        assert_eq!(controller.viewport_controller.histogram_side(), 1);
        assert_eq!(
            (
                state.get_histogram_min_val_abs(),
                state.get_histogram_max_val_abs()
            ),
            (0.25, 0.5)
        );

        state.invoke_side_changed(-3);
        drain_ui_queue();
        assert_eq!(controller.viewport_controller.histogram_side(), 0);
        assert_eq!(
            (
                state.get_histogram_min_val_abs(),
                state.get_histogram_max_val_abs()
            ),
            (0.1, 0.9)
        );
    }

    /// The sliders: the track from the limits, the handles relative to it,
    /// the labels absolute.
    #[test]
    fn sync_sets_the_track_handles_and_labels() {
        let (ui, _, controller) = attached();
        controller.update_histogram_settings_in_project(30.0, 70.0, 10.0, 110.0);
        controller.sync_histogram_settings_to_slint();
        drain_ui_queue();
        let state = ui.global::<HistogramState>();
        assert_eq!(state.get_histogram_min_lim(), 10.0);
        assert_eq!(state.get_histogram_max_lim(), 110.0);
        assert!((state.get_histogram_min_val() - 0.2).abs() < 1e-6);
        assert!((state.get_histogram_max_val() - 0.6).abs() < 1e-6);
        assert_eq!(state.get_histogram_min_val_abs(), 30.0);
        assert_eq!(state.get_histogram_max_val_abs(), 70.0);
    }

    /// Limits that are one value: no division by zero - the handles are
    /// placed as on a tiny track.
    #[test]
    fn sync_with_equal_limits_keeps_the_handles_finite() {
        let (ui, _, controller) = attached();
        controller.update_histogram_settings_in_project(0.5, 0.5, 0.5, 0.5);
        controller.sync_histogram_settings_to_slint();
        drain_ui_queue();
        let state = ui.global::<HistogramState>();
        assert_eq!(state.get_histogram_min_val(), 0.0);
        assert_eq!(state.get_histogram_max_val(), 0.0);
        assert_eq!(state.get_histogram_min_val_abs(), 0.5);
    }

    /// Nothing to show (the breakpoint not rendered yet): the sliders keep
    /// what they show.
    #[test]
    fn sync_without_a_window_leaves_the_sliders_alone() {
        let (ui, _, controller) = attached();
        let state = ui.global::<HistogramState>();
        state.set_histogram_min_val_abs(0.125);
        state.set_histogram_max_lim(42.0);
        controller.viewport_controller.set_histogram_side(1);
        controller.sync_histogram_settings_to_slint();
        drain_ui_queue();
        assert_eq!(state.get_histogram_min_val_abs(), 0.125);
        assert_eq!(state.get_histogram_max_lim(), 42.0);
    }
}
