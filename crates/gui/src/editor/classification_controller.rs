use crate::UiState;
use crate::editor::roi_list_controller::RoiListController;
use crate::editor::viewport_controller::ViewportController;
use crate::prelude::*;
use crate::{
    AppWindow, ClassItemData, ClassSettingsSlint, ClassificationSettingsState, ClassificationState,
};
use evanalyzer_cfg::AssignObjectClass;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::{
    Class, MeasurementChannels, MeasurementStatistics,
};
use indexmap::IndexMap;
use log::warn;
use slint::ModelRc;
use slint::{Color, Model};
use slint::{ComponentHandle, ModelNotify};
use std::rc::Rc;
use std::sync::Arc;

struct ClassificationModelBridge {
    app_state: Arc<UiState>,
    notify: ModelNotify,
}

pub struct ClassificationController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) roi_list_controller: Arc<RoiListController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl ClassificationController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        roi_list_controller: Arc<RoiListController>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            roi_list_controller,
            viewport_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // Selected class changed
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_selected(move |class_id| {
                    let obj_class = if class_id < 0 {
                        ObjectClass::Unset
                    } else {
                        AssignObjectClass!(class_id)
                    };
                    let mut project = manager.app_state.get_project_write();
                    project.set_selected_object_class(obj_class);
                });

            // Add / Update class
            let manager = self.clone();
            ui.global::<ClassificationSettingsState>()
                .on_apply_classification_settings(move |class_settings| {
                    manager.update_class_settings_in_project(class_settings);
                    manager.sync_classification_to_slint();
                    manager.roi_list_controller.sync_rois_to_slint();
                    manager.viewport_controller.trigger_image_redraw_rois();
                });

            // Auto add class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_auto_add_and_replace(move || {
                    let mut project = manager.app_state.get_project_write();
                    project.delete_all_classes();
                    project.auto_add_classes_based_on_image_meta();
                    manager.sync_classification_to_slint();
                    manager.roi_list_controller.sync_rois_to_slint();
                });

            // Auto add class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_auto_add_and_merge(move || {
                    let mut project = manager.app_state.get_project_write();
                    project.auto_add_classes_based_on_image_meta();
                    manager.sync_classification_to_slint();
                    manager.roi_list_controller.sync_rois_to_slint();
                });

            // Edit class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_edit(move |class_id| {
                    manager.sync_class_settings_to_class_edit_dialog_slint(AssignObjectClass!(
                        class_id
                    ));
                });

            // Delete class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_delete(move |class_id| {
                    let mut project = manager.app_state.get_project_write();
                    project
                        .classification
                        .delete_class(AssignObjectClass!(class_id));
                    project.set_selected_object_class(ObjectClass::Unset);
                    manager.sync_classification_to_slint();
                });

            // Move up
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_move_up(move |class_id| {
                    let mut project = manager.app_state.get_project_write();
                    project.classification.move_up(AssignObjectClass!(class_id));
                    manager.sync_classification_to_slint();
                });

            // Move down
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_move_down(move |class_id| {
                    let mut project = manager.app_state.get_project_write();
                    project
                        .classification
                        .move_down(AssignObjectClass!(class_id));
                    manager.sync_classification_to_slint();
                });

            // Toggle class visibility
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_visibility_toggled(move |class_id| {
                    manager
                        .app_state
                        .get_project_write()
                        .toggle_class_visibility(AssignObjectClass!(class_id));
                    manager.sync_classification_to_slint();
                    manager.viewport_controller.trigger_image_redraw_rois();
                });
        }
    }

    /// Synchronizes the current classification model state with the Slint UI.
    ///
    /// This method creates a thread-safe bridge between the internal application state
    /// and the Slint event loop. It wraps the controller's model bridge in a
    /// `ModelRc` and schedules an update on the UI thread.
    ///
    /// This is a non-blocking operation that ensures the UI reflects the most
    /// recent classification data without stalling the caller's thread.
    ///
    /// ### Thread Safety
    /// Since Slint's `ModelRc` requires an `Rc` (non-thread-safe) but the controller
    /// resides in an `Arc`, this method performs the necessary pointer wrapping
    /// within the Slint event loop context to bridge the two ownership models.
    ///
    /// ### Panics
    /// * This method does not panic but will log a `warn!` message if the
    ///   Slint event loop is unreachable or the update invocation fails.
    pub fn sync_classification_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let project = bridge_ptr.app_state.get_project();
                let total_visible: i32 = project
                    .classification
                    .classes
                    .iter()
                    .filter(|c| project.is_class_visible(&c.id))
                    .map(|c| project.count_rois_for_class(&c.id) as i32)
                    .sum();
                drop(project);

                let bridge = Rc::new(ClassificationModelBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                });
                let model_rc = ModelRc::new(bridge);
                let class_state = ui.global::<ClassificationState>();
                class_state.set_classes_list(model_rc);
                class_state.set_total_visible_objects(total_visible);
            }
        }) {
            warn!("Failed to sync classification to Slint: {}", e);
        }
    }

    /// Updates an existing class's configuration within the project based on the provided Slint settings.
    ///
    /// This method maps the UI-specific `ClassSettingsSlint` (which contains measurement
    /// parameters and style properties) into the internal `Class` model. It performs
    /// the necessary type conversions-specifically converting color values from
    /// Slint's representation to a standard `u32` format-and updates the corresponding
    /// `MeasurementChannels` map.
    ///
    /// ### Arguments
    /// * `class_settings` - The `ClassSettingsSlint` structure containing the updated
    ///   parameters for the class, such as display name, color, and various
    ///   measurement criteria.
    ///
    /// ### Thread Safety
    /// This method is intended to be called from the UI thread (via an `Arc<Self>`
    /// reference) and ensures data consistency by utilizing the underlying
    /// `update_class` method, which handles concurrent access via `RwLock`.
    pub fn update_class_settings_in_project(self: &Arc<Self>, class_settings: ClassSettingsSlint) {
        let mut project = self.app_state.get_project_write();

        let mut measures: IndexMap<MeasurementChannels, Vec<MeasurementStatistics>> =
            IndexMap::new();

        let new_class = Class {
            id: AssignObjectClass!(class_settings.class_id),
            color: class_settings.color.as_argb_encoded(),
            name: class_settings.name.into(),
            notes: class_settings.notes.into(),
            measure: measures,
        };

        if class_settings.class_id < 0 {
            // This is a new class, we have to enter a new class
            project.classification.add_class(new_class);
        } else {
            let ret = project.classification.update_class(new_class);
            if ret.is_err() {
                warn!("Could not update class!");
            }
        }
    }

    /// Synchronizes the current backend classification settings into the Slint UI state.
    ///
    /// This is typically called when a user selects a class for editing. It converts
    /// the internal Rust business models (likely from `evanalzer_core`) into the
    /// auto-generated `ClassSettingsSlint` struct used by the UI elements.
    pub fn sync_class_settings_to_class_edit_dialog_slint(self: &Arc<Self>, class_id: ObjectClass) {
        let ui_weak = self.ui.clone();
        let project = self.app_state.get_project();
        let class_cloned = project.classification.get_class(class_id).cloned();
        drop(project);

        let Some(class) = class_cloned else {
            return;
        };
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let r = ((class.color >> 16) & 0xff) as u8;
                let g = ((class.color >> 8) & 0xff) as u8;
                let b = (class.color & 0xff) as u8;

                let class_settings_state = ui.global::<ClassificationSettingsState>();
                let mut class_settings_slint: ClassSettingsSlint = ClassSettingsSlint::default();
                class_settings_slint.class_id = class.id.to_i32();
                class_settings_slint.color = Color::from_rgb_u8(r, g, b);
                class_settings_slint.name = class.name.clone().into();
                class_settings_slint.notes = class.notes.clone().into();

                class_settings_state.set_settings(class_settings_slint);
            }
        }) {
            warn!("Failed to sync classification to Slint: {}", e);
        }
    }
}

impl Model for ClassificationModelBridge {
    type Data = ClassItemData;

    /// Returns the total number of classification classes currently loaded.
    ///
    /// This method accesses the project settings via the shared application state.
    /// It locks the classification settings for reading to determine the count.
    ///
    /// ### Returns
    /// * `usize` - The count of classes in the underlying vector.
    ///
    /// ### Panics
    /// * If the internal classification read-lock is poisoned.
    fn row_count(&self) -> usize {
        let project = self.app_state.get_project();
        project.classification.classes.len()
    }

    /// Retrieves the display data for a specific classification row.
    ///
    /// This method performs an on-demand conversion of the internal `Class` data
    /// into a `ClassItemData` structure compatible with Slint. This includes
    /// parsing the hex-encoded color into a Slint-compatible `Color` object.
    ///
    /// ### Arguments
    /// * `row` - The zero-based index of the class to retrieve.
    ///
    /// ### Returns
    /// * `Some(ClassItemData)` if the row index is valid.
    /// * `None` if the index is out of bounds.
    ///
    /// ### Panics
    /// * If the internal classification read-lock is poisoned.
    fn row_data(&self, row: usize) -> Option<Self::Data> {
        let project = self.app_state.get_project();
        let classes = project.classification.classes.clone();
        classes.get(row).map(|c| {
            let r = ((c.color >> 16) & 0xff) as u8;
            let g = ((c.color >> 8) & 0xff) as u8;
            let b = (c.color & 0xff) as u8;
            let count = project.count_rois_for_class(&c.id) as i32;
            let visible = project.is_class_visible(&c.id);
            ClassItemData {
                id: c.id.to_i32(),
                name: (&c.name).into(),
                color: Color::from_rgb_u8(r, g, b),
                count,
                visible,
            }
        })
    }

    /// Returns a reference to the tracker that notifies Slint of data changes.
    ///
    /// This is essential for the UI to know when the underlying vector has
    /// been modified (e.g., added, deleted, or reordered).
    fn model_tracker(&self) -> &dyn slint::ModelTracker {
        &self.notify
    }
}
