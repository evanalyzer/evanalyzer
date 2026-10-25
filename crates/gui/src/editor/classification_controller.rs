//! # classification_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::object_list_controller::{ObjectListController, resolve_object_id};
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::color_from_hex;
use crate::{
    AppWindow, ClassItemData, ClassSettingsSlint, ClassificationSettingsState, ClassificationState,
    ObjectListState,
};
use evanalyzer_app::prelude::classification_ext::ClassificationExt;
use evanalyzer_app::project::{AnnotationsExt, ProjectExt};
use evanalyzer_app::project::{free_class_ids, next_class_id};
use evanalyzer_cfg::AssignObjectClass;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use log::{info, warn};
use slint::Model;
use slint::{ComponentHandle, ModelNotify};
use slint::{ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::sync::Arc;

struct ClassificationModelBridge {
    app_state: Arc<UiState>,
    notify: ModelNotify,
}

pub struct ClassificationController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl ClassificationController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        object_list_controller: Arc<ObjectListController>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            object_list_controller,
            viewport_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // Class operations on single objects (object list buttons). Here,
            // not in the object list controller: they change the class counts
            // too, and only this controller can refresh both.
            // Add class to object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_add_class(move |object_id, class_id| {
                    // A change of its own on the undo stack; only manual
                    // annotations change (pipeline objects are read-only).
                    manager.app_state.begin_undo_step();
                    let mut project = manager.app_state.get_project_write();
                    if let Some(id) = resolve_object_id(&project, object_id)
                        .filter(|id| project.is_manual_object(id.clone()))
                    {
                        // let class_id = project.get_selected_object_class();
                        if class_id >= 0 {
                            let class_id = ObjectClass::Global(class_id as u32);
                            project.add_class_to_object(id, class_id);
                        }
                    }
                    drop(project);
                    manager.app_state.mark_dirty();
                    manager
                        .object_list_controller
                        .sync_selected_object_to_slint(false);
                    manager.object_list_controller.sync_objects_to_slint();
                    // The object's classes changed: so do the class counts.
                    manager.sync_classification_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Rplace all classes
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_replace_class(move |object_id, class_id| {
                    // A change of its own on the undo stack; only manual
                    // annotations change (pipeline objects are read-only).
                    manager.app_state.begin_undo_step();
                    let mut project = manager.app_state.get_project_write();
                    if let Some(id) = resolve_object_id(&project, object_id)
                        .filter(|id| project.is_manual_object(id.clone()))
                    {
                        // let class_id = project.get_selected_object_class();
                        if class_id >= 0 {
                            let class_id = ObjectClass::Global(class_id as u32);
                            project.replace_classes_of_object(id, class_id);
                        }
                    }
                    drop(project);
                    manager.app_state.mark_dirty();
                    manager
                        .object_list_controller
                        .sync_selected_object_to_slint(false);
                    manager.object_list_controller.sync_objects_to_slint();
                    // The object's classes changed: so do the class counts.
                    manager.sync_classification_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Remove class from object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_remove_class(move |object_id, class_id| {
                    // A change of its own on the undo stack; only manual
                    // annotations change (pipeline objects are read-only).
                    manager.app_state.begin_undo_step();
                    let mut project = manager.app_state.get_project_write();
                    if let Some(id) = resolve_object_id(&project, object_id)
                        .filter(|id| project.is_manual_object(id.clone()))
                    {
                        let class_id = ObjectClass::Global(class_id as u32);
                        project.remove_class_from_object(id, &class_id);
                    }
                    drop(project);
                    manager.app_state.mark_dirty();
                    manager
                        .object_list_controller
                        .sync_selected_object_to_slint(false);
                    manager.object_list_controller.sync_objects_to_slint();
                    // The object's classes changed: so do the class counts.
                    manager.sync_classification_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Delete object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_delete(move |object_id| {
                    // A change of its own on the undo stack; only manual
                    // annotations change (pipeline objects are read-only).
                    manager.app_state.begin_undo_step();
                    let mut project = manager.app_state.get_project_write();
                    if let Some(id) = resolve_object_id(&project, object_id)
                        .filter(|id| project.is_manual_object(id.clone()))
                    {
                        project.delete_object(id);
                    }
                    project.set_selected_object(None);
                    drop(project);
                    manager.app_state.mark_dirty();
                    manager.object_list_controller.sync_objects_to_slint();
                    // The object's classes changed: so do the class counts.
                    manager.sync_classification_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

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
                    // Redraw the objects because class selection changes the z-order visibility
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Add / Update class
            let manager = self.clone();
            ui.global::<ClassificationSettingsState>()
                .on_apply_classification_settings(move |class_settings| {
                    let created = manager.update_class_settings_in_project(class_settings);
                    // The picker that asked for the new class picks it.
                    if let Some(class) = created
                        && let Some(ui) = manager.ui.upgrade()
                    {
                        let state = ui.global::<ClassificationSettingsState>();
                        state.set_last_created_id(class.to_i32());
                        state.set_created_count(state.get_created_count() + 1);
                    }
                    manager.sync_classification_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Auto add class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_auto_add_and_replace(move || {
                    let mut project = manager.app_state.get_project_write();
                    project.delete_all_classes();
                    project.auto_add_classes_based_on_image_meta();
                    manager.sync_classification_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
                });

            // Auto add class
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_auto_add_and_merge(move || {
                    let mut project = manager.app_state.get_project_write();
                    project.auto_add_classes_based_on_image_meta();
                    manager.sync_classification_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
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

            // Delete all classes
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_class_delete_all(move || {
                    let mut project = manager.app_state.get_project_write();
                    project.delete_all_classes();
                    project.set_selected_object_class(ObjectClass::Unset);
                    manager.sync_classification_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
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
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Toggle hide/show unclassified ROIs
            let manager = self.clone();
            ui.global::<ClassificationState>()
                .on_hide_unclassified_toggled(move || {
                    manager
                        .app_state
                        .get_project_write()
                        .toggle_hide_unclassified_objects();
                    manager.sync_classification_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
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

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let project = bridge_ptr.app_state.get_project();
                let total_visible: i32 = project
                    .classification
                    .classes()
                    .iter()
                    .filter(|c| project.is_class_visible(&c.global()))
                    .map(|c| project.count_objects_for_class(&c.global()) as i32)
                    .sum();
                let hide_unclassified = project.hide_unclassified_objects();
                // A new class's id: automatic, or one of the free ones.
                let new_id_options: Vec<SharedString> =
                    std::iter::once(format!("Auto ({})", next_class_id(&project.classification)))
                        .chain(
                            free_class_ids(&project.classification)
                                .into_iter()
                                .map(|id| id.to_string()),
                        )
                        .map(Into::into)
                        .collect();
                drop(project);
                ui.global::<ClassificationSettingsState>()
                    .set_new_id_options(ModelRc::new(VecModel::from(new_id_options)));

                let bridge = Rc::new(ClassificationModelBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                });
                let model_rc = ModelRc::new(bridge);
                let class_state = ui.global::<ClassificationState>();
                class_state.set_classes_list(model_rc);
                class_state.set_total_visible_objects(total_visible);
                class_state.set_hide_unclassified_objects(hide_unclassified);
                // Objects or classes changed: an open AI training dialog
                // follows (its training objects and class counts).
                ui.global::<crate::AiLearningState>()
                    .invoke_project_changed();
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
    /// Adds or updates the class; the id of a class it adds.
    pub fn update_class_settings_in_project(
        self: &Arc<Self>,
        class_settings: ClassSettingsSlint,
    ) -> Option<ObjectClass> {
        let mut project = self.app_state.get_project_write();

        let new_class = Class {
            // Negative for a new class - it gets `new_id`, or (negative too)
            // the next one from `add_class`.
            id: class_settings.class_id.max(0) as u32,
            color: class_settings.color.as_argb_encoded(),
            name: class_settings.name.into(),
            notes: class_settings.notes.into(),
        };

        if class_settings.class_id < 0 {
            if class_settings.new_id < 0 {
                return Some(project.classification.add_class(new_class));
            }
            return match project.classification.add_class_with_id(Class {
                id: class_settings.new_id as u32,
                ..new_class
            }) {
                Ok(class) => Some(class),
                Err(e) => {
                    warn!("Could not add the class: {e}");
                    None
                }
            };
        } else {
            if project.classification.update_class(new_class).is_err() {
                warn!("Could not update class!");
                return None;
            }
            // A new id: the class and everything referring to it.
            if class_settings.new_id >= 0 && class_settings.new_id != class_settings.class_id {
                match project
                    .change_class_id(class_settings.class_id as u32, class_settings.new_id as u32)
                {
                    Ok(change) => info!(
                        "Class {} is now {}: {} references and {} Extract Objects mappings changed",
                        class_settings.class_id,
                        class_settings.new_id,
                        change.references,
                        change.mappings_added
                    ),
                    Err(e) => warn!("Could not change the class id: {e}"),
                }
            }
        }
        None
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
        let free_ids = free_class_ids(&project.classification);
        drop(project);

        let Some(class) = class_cloned else {
            return;
        };
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let class_settings_state = ui.global::<ClassificationSettingsState>();
                let mut class_settings_slint: ClassSettingsSlint = ClassSettingsSlint::default();
                class_settings_slint.class_id = class.id as i32;
                class_settings_slint.new_id = class.id as i32;
                // Its own id first, then the free ones.
                let id_options: Vec<SharedString> = std::iter::once(class.id)
                    .chain(free_ids)
                    .map(|id| id.to_string().into())
                    .collect();
                class_settings_state.set_id_options(ModelRc::new(VecModel::from(id_options)));
                class_settings_state.set_id_index(0);
                class_settings_slint.color = color_from_hex(class.color);
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
        project.classification.classes().len()
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
        let classes = project.classification.classes().clone();
        classes.get(row).map(|c| {
            let count = project.count_objects_for_class(&c.global()) as i32;
            let visible = project.is_class_visible(&c.global());
            ClassItemData {
                id: c.id as i32,
                name: (&c.name).into(),
                color: color_from_hex(c.color),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;
    use slint::Color;

    fn make_controller() -> (Arc<UiState>, Arc<ClassificationController>) {
        make_controller_with_ui(slint::Weak::default())
    }

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
    ) -> (Arc<UiState>, Arc<ClassificationController>) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let controller = Arc::new(ClassificationController::new(
            ui,
            ui_state.clone(),
            object_list_controller,
            viewport_controller,
        ));
        (ui_state, controller)
    }

    fn settings(class_id: i32, name: &str) -> ClassSettingsSlint {
        ClassSettingsSlint {
            class_id,
            new_id: class_id,
            name: name.into(),
            notes: "".into(),
            color: Color::from_rgb_u8(10, 20, 30),
        }
    }

    #[test]
    fn a_class_the_dialog_adds_is_reported_for_the_picker_that_asked() {
        let ui = crate::editor::test_support::test_ui_window();
        let project = crate::editor::test_support::project_with_one_image();
        let (_ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();
        let dialog = ui.global::<ClassificationSettingsState>();

        dialog.invoke_apply_classification_settings(settings(-1, "Nuclei"));
        let first = dialog.get_last_created_id();
        assert!(first >= 0);
        assert_eq!(dialog.get_created_count(), 1);

        // Editing a class adds none.
        dialog.invoke_apply_classification_settings(settings(first, "Renamed"));
        assert_eq!(dialog.get_created_count(), 1);

        dialog.invoke_apply_classification_settings(settings(-1, "Spots"));
        assert_eq!(dialog.get_created_count(), 2);
        assert_ne!(dialog.get_last_created_id(), first);
    }

    #[test]
    fn update_class_settings_with_a_negative_id_adds_a_new_class() {
        let (ui_state, controller) = make_controller();

        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        let project = ui_state.get_project();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(project.classification.classes().len(), 2);
        assert_eq!(project.classification.classes()[1].id, 1);
        assert_eq!(project.classification.classes()[1].name, "Nuclei");
    }

    #[test]
    fn a_new_class_gets_the_chosen_free_id() {
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(ClassSettingsSlint {
            new_id: 4,
            ..settings(-1, "Nuclei")
        });
        let project = ui_state.get_project();
        assert_eq!(project.classification.classes()[1].id, 4);
        assert_eq!(project.classification.classes()[1].name, "Nuclei");
    }

    #[test]
    fn a_new_id_moves_the_references_selection_and_hidden_state_along() {
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        use evanalyzer_cfg::settings::pipeline_command_settings::ClassifyObjectsSettings;
        use evanalyzer_cfg::settings::pipeline_settings::{
            PipelineSettings, PipelineStepSettings, StepContent,
        };
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));
        {
            let mut project = ui_state.get_project_write();
            project.pipelines.push(PipelineSettings {
                id: evanalyzer_cfg::core_types::PipelineId(1),
                name: "P".into(),
                description: None,
                image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
                enabled: true,
                steps: vec![PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId(1),
                    enabled: true,
                    content: StepContent::Command(PipelineCommand::ClassifyObjects(
                        ClassifyObjectsSettings {
                            input_classes: vec![ObjectClass::Global(1)],
                            ..Default::default()
                        },
                    )),
                }],
            });
            project.set_selected_object_class(ObjectClass::Global(1));
            project.toggle_class_visibility(ObjectClass::Global(1));
        }

        controller.update_class_settings_in_project(ClassSettingsSlint {
            new_id: 7,
            ..settings(1, "Renamed")
        });

        let project = ui_state.get_project();
        let class = &project.classification.classes()[1];
        assert_eq!((class.id, class.name.as_str()), (7, "Renamed"));
        let Some(PipelineCommand::ClassifyObjects(classify)) =
            project.pipelines[0].steps[0].command()
        else {
            panic!("classify")
        };
        assert_eq!(classify.input_classes, [ObjectClass::Global(7)]);
        assert_eq!(project.get_selected_object_class(), ObjectClass::Global(7));
        assert!(
            !project.is_class_visible(&ObjectClass::Global(7)),
            "still hidden"
        );
    }

    #[test]
    fn the_dialog_offers_auto_or_free_ids_and_the_own_id_when_editing() {
        use crate::helper::ui_thread::drain_ui_queue;
        let mut project = crate::editor::test_support::project_with_one_image();
        project.classification.add_class(Class {
            id: 0,
            color: 0,
            name: "Nuclei".into(),
            notes: String::new(),
        });
        let ui = crate::editor::test_support::test_ui_window();
        let (ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();
        controller.sync_classification_to_slint();
        drain_ui_queue();
        let dialog = ui.global::<ClassificationSettingsState>();
        let options =
            |m: ModelRc<SharedString>| m.iter().map(|o| o.to_string()).collect::<Vec<_>>();

        // A new class: "Auto" first, then the free ids (0 and 1 are taken).
        let new_options = options(dialog.get_new_id_options());
        assert_eq!(new_options[0], "Auto (2)");
        assert_eq!(new_options[1], "2");
        dialog.invoke_reset_to_zero();
        assert_eq!(options(dialog.get_id_options()), new_options);

        // Editing class 1: its own id first.
        ui.global::<ClassificationState>().invoke_class_edit(1);
        drain_ui_queue();
        let edit_options = options(dialog.get_id_options());
        assert_eq!(&edit_options[..2], ["1", "2"]);
        assert_eq!(dialog.get_id_index(), 0);

        // Choosing 5 and pressing "Apply".
        let index = edit_options.iter().position(|o| o == "5").unwrap();
        dialog.set_id_index(index as i32);
        ui.global::<crate::GlobalAppState>()
            .set_active_dialog(crate::DialogType::ClassificationEditClassDialog);
        i_slint_backend_testing::ElementHandle::find_by_accessible_label(&ui, "Apply")
            .next()
            .expect("the dialog's apply button")
            .invoke_accessible_default_action();
        drain_ui_queue();
        assert_eq!(ui_state.get_project().classification.classes()[1].id, 5);
    }

    #[test]
    fn update_class_settings_with_an_existing_id_replaces_that_class_in_place() {
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        controller.update_class_settings_in_project(settings(1, "Renamed"));

        let project = ui_state.get_project();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(
            project.classification.classes().len(),
            2,
            "updating must not create a second class"
        );
        assert_eq!(project.classification.classes()[1].name, "Renamed");
    }

    #[test]
    fn update_class_settings_for_an_unknown_id_leaves_the_class_list_unchanged() {
        let (ui_state, controller) = make_controller();

        // No class with id 1 exists yet - `update_class` should fail (and be
        // logged), not panic or create one.
        controller.update_class_settings_in_project(settings(1, "Ghost"));

        let project = ui_state.get_project();
        // A fresh project always has exactly the auto-prepended Background
        // class and nothing else.
        assert_eq!(project.classification.classes().len(), 1);
    }

    #[test]
    fn update_class_settings_stores_the_given_color_as_argb() {
        let (ui_state, controller) = make_controller();
        let color = Color::from_rgb_u8(10, 20, 30);

        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        let project = ui_state.get_project();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(
            project.classification.classes()[1].color,
            color.as_argb_encoded()
        );
    }

    // -- ClassificationModelBridge (Model impl) --------------------------------

    fn bridge_for(ui_state: &Arc<UiState>) -> ClassificationModelBridge {
        ClassificationModelBridge {
            app_state: ui_state.clone(),
            notify: ModelNotify::default(),
        }
    }

    #[test]
    fn classification_model_bridge_row_count_matches_the_number_of_registered_classes() {
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "A"));
        controller.update_class_settings_in_project(settings(-1, "B"));

        let bridge = bridge_for(&ui_state);

        // Background, plus the two added classes.
        assert_eq!(bridge.row_count(), 3);
    }

    #[test]
    fn classification_model_bridge_row_data_decodes_the_hex_color_and_id() {
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(ClassSettingsSlint {
            class_id: -1,
            new_id: -1,
            name: "Nuclei".into(),
            notes: "".into(),
            color: Color::from_rgb_u8(0x11, 0x22, 0x33),
        });

        let bridge = bridge_for(&ui_state);
        // Row 0 is the auto-prepended Background class; Nuclei lands at row 1.
        let row = bridge.row_data(1).expect("one class was added");

        assert_eq!(row.id, 1); // add_class assigns id 1 to the first added class
        assert_eq!(row.name.as_str(), "Nuclei");
        assert_eq!(row.color, Color::from_rgb_u8(0x11, 0x22, 0x33));
    }

    #[test]
    fn classification_model_bridge_row_data_counts_objects_carrying_that_class() {
        let ui_state = crate::editor::test_support::test_ui_state_with_project(
            crate::editor::test_support::project_with_one_image(),
        );
        {
            let mut project = ui_state.get_project_write();
            project.classification.classes_mut().push(
                evanalyzer_cfg::settings::classification_settings::Class {
                    id: 1,
                    color: 0,
                    name: "Nuclei".to_string(),
                    notes: String::new(),
                },
            );
            let mut with_class =
                evanalyzer_cfg::settings::object_settings::ObjectMetricSettings::default();
            with_class.object_class.insert(ObjectClass::Global(1));
            project.add_object(&with_class);
            project.add_object(
                &evanalyzer_cfg::settings::object_settings::ObjectMetricSettings::default(),
            );
        }

        let bridge = bridge_for(&ui_state);
        // Row 0 is the auto-prepended Background class; Nuclei lands at row 1.
        let row = bridge.row_data(1).expect("one class registered");

        assert_eq!(row.count, 1);
    }

    #[test]
    fn classification_model_bridge_row_data_reflects_visibility_toggles() {
        let (ui_state, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        // Row 0 is the auto-prepended Background class; Nuclei lands at row 1.
        assert!(
            bridge_for(&ui_state).row_data(1).unwrap().visible,
            "classes are visible by default"
        );

        ui_state
            .get_project_write()
            .toggle_class_visibility(ObjectClass::Global(1));

        assert!(!bridge_for(&ui_state).row_data(1).unwrap().visible);
    }

    #[test]
    fn classification_model_bridge_row_data_out_of_range_returns_none() {
        let (ui_state, _controller) = make_controller();

        // A fresh project already has one row (Background) - row 1 is the
        // first genuinely out-of-range index.
        assert!(bridge_for(&ui_state).row_data(1).is_none());
    }

    // -- sync_classification_to_slint / sync_class_settings_to_class_edit_dialog_slint

    #[test]
    fn sync_classification_to_slint_does_not_panic_without_a_live_ui() {
        let (_, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        controller.sync_classification_to_slint();
    }

    #[test]
    fn sync_class_settings_to_class_edit_dialog_slint_does_not_panic_for_a_known_or_unknown_class()
    {
        let (_, controller) = make_controller();
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));

        controller.sync_class_settings_to_class_edit_dialog_slint(ObjectClass::Global(1));
        // Unknown class id - takes the early `return` branch.
        controller.sync_class_settings_to_class_edit_dialog_slint(ObjectClass::Global(99));
    }

    // -- attach_callbacks (live AppWindow) -----------------------------------------

    use crate::editor::test_support::{project_with_one_image, test_ui_window};
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

    #[test]
    fn attach_callbacks_class_selected_stores_the_selected_object_class() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<ClassificationState>().invoke_class_selected(3);
        assert_eq!(
            ui_state.get_project().get_selected_object_class(),
            ObjectClass::Global(3)
        );

        ui.global::<ClassificationState>().invoke_class_selected(-1);
        assert_eq!(
            ui_state.get_project().get_selected_object_class(),
            ObjectClass::Unset
        );
    }

    #[test]
    fn attach_callbacks_class_delete_removes_the_class_and_clears_selection() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));
        controller.attach_callbacks();

        ui.global::<ClassificationState>().invoke_class_delete(1);

        let project = ui_state.get_project();
        // Background can't be deleted (see `ClassificationExt::delete_class`),
        // so a fresh project falls back to exactly that one class.
        assert_eq!(project.classification.classes().len(), 1);
        assert_eq!(project.get_selected_object_class(), ObjectClass::Unset);
    }

    #[test]
    fn attach_callbacks_class_move_up_and_down_reorder_the_list() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.update_class_settings_in_project(settings(-1, "A"));
        controller.update_class_settings_in_project(settings(-1, "B"));
        controller.attach_callbacks();

        // classes()[0] is the auto-prepended Background class, which
        // `move_up`/`move_down` always keep pinned there - starting state is
        // [Background, A, B].
        ui.global::<ClassificationState>().invoke_class_move_up(2);
        assert_eq!(ui_state.get_project().classification.classes()[1].name, "B");

        ui.global::<ClassificationState>().invoke_class_move_down(2);
        assert_eq!(ui_state.get_project().classification.classes()[2].name, "B");
    }

    #[test]
    fn attach_callbacks_class_visibility_toggled_flips_visibility() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.update_class_settings_in_project(settings(-1, "Nuclei"));
        controller.attach_callbacks();

        ui.global::<ClassificationState>()
            .invoke_class_visibility_toggled(1);

        assert!(
            !ui_state
                .get_project()
                .is_class_visible(&ObjectClass::Global(1))
        );
    }

    #[test]
    fn attach_callbacks_hide_unclassified_toggled_flips_the_flag() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();
        let initial = ui_state.get_project().hide_unclassified_objects();

        ui.global::<ClassificationState>()
            .invoke_hide_unclassified_toggled();

        assert_eq!(ui_state.get_project().hide_unclassified_objects(), !initial);
    }

    #[test]
    fn attach_callbacks_apply_classification_settings_adds_a_class_via_the_wired_callback() {
        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<ClassificationSettingsState>()
            .invoke_apply_classification_settings(settings(-1, "Nuclei"));

        let project = ui_state.get_project();
        // classes()[0] is the auto-prepended Background class.
        assert_eq!(project.classification.classes().len(), 2);
        assert_eq!(project.classification.classes()[1].name, "Nuclei");
    }

    // ---- class operations on single objects ----

    fn object_with_class(id: u128, has_class: bool) -> ObjectMetricSettings {
        let mut object = ObjectMetricSettings {
            id: evanalyzer_cfg::core_types::ObjectId(id),
            segmentation_class: evanalyzer_cfg::core_types::SegmentationClass(1),
            ..Default::default()
        };
        if has_class {
            object.object_class.insert(ObjectClass::Global(1));
        }
        object
    }

    /// A classification controller on real windows, for `project`.
    fn classification_on_windows(
        ui: &AppWindow,
        project: evanalyzer_app::project::ProjectWithRuntime,
    ) -> (Arc<UiState>, Arc<ClassificationController>) {
        let ui_state = crate::editor::test_support::ui_state_with_window(ui, project);
        let weak = ui.as_weak();
        let viewport = Arc::new(ViewportController::new(
            weak.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list = Arc::new(ObjectListController::new(
            weak.clone(),
            ui_state.clone(),
            viewport.clone(),
        ));
        let controller = Arc::new(ClassificationController::new(
            weak,
            ui_state.clone(),
            object_list,
            viewport,
        ));
        (ui_state, controller)
    }

    fn with_nuclei_class(
        mut project: evanalyzer_app::project::ProjectWithRuntime,
    ) -> evanalyzer_app::project::ProjectWithRuntime {
        project.classification.classes_mut().push(Class {
            id: 1,
            color: 0,
            name: "Nuclei".to_string(),
            notes: String::new(),
        });
        project
    }

    #[test]
    fn adding_and_removing_a_class_on_an_object_updates_the_counts() {
        use crate::helper::ui_thread::drain_ui_queue;
        let mut project = with_nuclei_class(crate::editor::test_support::project_with_one_image());
        project.add_object(&object_with_class(1, false));
        let ui = crate::editor::test_support::test_ui_window();
        let (ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();
        ui_state
            .get_project_write()
            .set_selected_object_class(ObjectClass::Global(1));
        let total = || {
            ui.global::<ClassificationState>()
                .get_total_visible_objects()
        };

        ui.global::<ObjectListState>().invoke_object_add_class(1, 1);
        drain_ui_queue();
        assert_eq!(total(), 1);

        ui.global::<ObjectListState>()
            .invoke_object_remove_class(1, 1);
        drain_ui_queue();
        assert_eq!(total(), 0);
    }

    #[test]
    fn deleting_an_object_updates_the_counts() {
        use crate::helper::ui_thread::drain_ui_queue;
        let mut project = with_nuclei_class(crate::editor::test_support::project_with_one_image());
        project.add_object(&object_with_class(1, true));
        let ui = crate::editor::test_support::test_ui_window();
        let (_ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();
        controller.sync_classification_to_slint();
        drain_ui_queue();
        let total = || {
            ui.global::<ClassificationState>()
                .get_total_visible_objects()
        };
        assert_eq!(total(), 1);

        ui.global::<ObjectListState>().invoke_object_delete(1);
        drain_ui_queue();
        assert_eq!(total(), 0);
    }

    #[test]
    fn object_add_class_targets_the_manual_object_when_preview_objects_are_also_present() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, false)); // manual, unclassified
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, true)); // preview, combined index 1

        let ui = test_ui_window();
        let (ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();
        ui_state
            .get_project_write()
            .set_selected_object_class(ObjectClass::Global(1));

        // id 1 is the manual object even though a preview object also exists
        // in the combined list - regression coverage that the shared
        // resolution logic still picks the right one when both lists are
        // non-empty at once.
        ui.global::<ObjectListState>().invoke_object_add_class(1, 1);

        let project = ui_state.get_project();
        let manual_object = &project.get_objects().unwrap()[0];
        assert!(
            manual_object.object_class.contains(&ObjectClass::Global(1)),
            "the manual object should have had the class applied"
        );
    }

    #[test]
    fn object_delete_removes_the_manual_object_when_preview_objects_are_also_present() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, true)); // manual, combined index 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, true)); // preview, combined index 1

        let ui = test_ui_window();
        let (ui_state, controller) = classification_on_windows(&ui, project);
        controller.attach_callbacks();

        ui.global::<ObjectListState>().invoke_object_delete(1);

        let project = ui_state.get_project();
        assert!(project.get_objects().unwrap().is_empty());
        // The preview object (not touched by delete_object, which only ever
        // operates on the manual list) must be unaffected.
        assert_eq!(project.get_preview_objects().len(), 1);
    }
}
