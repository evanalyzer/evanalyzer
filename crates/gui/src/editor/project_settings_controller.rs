use crate::UiState;
use crate::{AppWindow, ProjectSettingsSlint, ProjectSettingsState, ResultsWindow};
use evanalyzer_app::global::UserInformation;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::plate_settings::{
    GroupingMode, PlateSettings, PlateSize, WellLayout,
};
use evanalyzer_cfg::settings::project_settings::TileMergeConnectivity;
use slint::{ComponentHandle, Model, ModelRc, SharedString};
use std::sync::{Arc, Mutex};

/// `ProjectSettingsState` is edited from two places: the Project Settings
/// dialog (on `AppWindow`) and the Results window's Matrix view settings
/// strip (on `ResultsWindow`, see `results_matrix_controller.rs`). Slint
/// gives each top-level window its own independent instance of every global
/// it references — `AppWindow` and `ResultsWindow` do **not** share one
/// `ProjectSettingsState` at runtime, even though both compile against the
/// same `.slint` global declaration. Every callback below is therefore
/// registered on *both* windows' instances, and `sync_project_settings_to_slint`
/// pushes to both, so the two stay in lockstep instead of silently diverging.
pub struct ProjectSettingsController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) results_ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
    // Called after the project's settings were (re)loaded into the dialog -
    // the results window follows the plate settings (wired in editor.rs).
    plate_settings_listener: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl ProjectSettingsController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        results_ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
    ) -> Self {
        Self {
            ui,
            results_ui,
            app_state,
            plate_settings_listener: Mutex::new(None),
        }
    }

    /// `f` runs whenever the project's settings were synced to the dialog
    /// (applied, cancelled, undone, a project loaded) - on the UI thread.
    pub fn on_plate_settings_changed(&self, f: impl Fn() + Send + Sync + 'static) {
        *self.plate_settings_listener.lock().expect("Poisoned") = Some(Box::new(f));
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        // Load user settings
        if let Some(_author) = &self.app_state.load_app_settings().author {
            self.sync_project_settings_to_slint();
        }

        // Attache callbacks
        if let Some(ui) = self.ui.upgrade() {
            let manager = self.clone();
            ui.global::<ProjectSettingsState>()
                .on_project_settings_changed(move |project_settings| {
                    manager.update_project_settings_in_project(&project_settings);
                    manager.sync_project_settings_to_slint();
                });

            let manager = self.clone();
            ui.global::<ProjectSettingsState>()
                .on_project_settings_canceled(move || {
                    manager.sync_project_settings_to_slint();
                });

            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_well_value_changed(move |index, value| {
                    if let Some(ui) = ui_weak.upgrade() {
                        let model = ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        set_well_value(&model, index, value);
                    }
                });

            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_well_dims_changed(move |rows, cols| {
                    if let Some(ui) = ui_weak.upgrade() {
                        let model = ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        resize_well_values(&model, rows, cols);
                    }
                });

            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_tile_merge_class_toggled(move |value| {
                    if let Some(ui) = ui_weak.upgrade() {
                        let model = ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .tile_merge_classes_to_not_merge_flags;
                        toggle_tile_merge_class(&model, &value);
                    }
                });
        }

        if let Some(results_ui) = self.results_ui.upgrade() {
            let manager = self.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_project_settings_changed(move |project_settings| {
                    manager.update_project_settings_in_project(&project_settings);
                    manager.sync_project_settings_to_slint();
                });

            let manager = self.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_project_settings_canceled(move || {
                    manager.sync_project_settings_to_slint();
                });

            let results_ui_weak = self.results_ui.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_well_value_changed(move |index, value| {
                    if let Some(results_ui) = results_ui_weak.upgrade() {
                        let model = results_ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        set_well_value(&model, index, value);
                    }
                });

            let results_ui_weak = self.results_ui.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_well_dims_changed(move |rows, cols| {
                    if let Some(results_ui) = results_ui_weak.upgrade() {
                        let model = results_ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        resize_well_values(&model, rows, cols);
                    }
                });

            let results_ui_weak = self.results_ui.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_tile_merge_class_toggled(move |value| {
                    if let Some(results_ui) = results_ui_weak.upgrade() {
                        let model = results_ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .tile_merge_classes_to_not_merge_flags;
                        toggle_tile_merge_class(&model, &value);
                    }
                });
        }
    }

    /// Synchronizes project configuration from the Slint UI settings dialog back to the internal project state.
    ///
    /// This function handles:
    /// 1. Author Metadata: Splitting the full name into first/last name and updating organization.
    /// 2. Grouping Logic: Converting UI dropdown indices into actual GroupingModes and Regex patterns.
    /// 3. Plate Geometry: Updating well dimensions and the flat-mapped image sequence order.
    pub fn update_project_settings_in_project(&self, project_settings: &ProjectSettingsSlint) {
        {
            let mut project = self.app_state.get_project_write();

            // Meta settings
            {
                self.app_state.update_app_settings(|settings| {
                    settings.author = Some(UserInformation {
                        full_name: project_settings.author_name.clone().into(),
                        organization: project_settings.organization_name.clone().into(),
                    });
                });

                project.meta.name = project_settings.project_name.clone().into();
            }

            // Plate settings
            project.plate = plate_settings_from_slint(project_settings, &project.plate);

            // Tile merging (docs/tile_merge_plan.md)
            {
                let tile_merge = &mut project.tile_merge;
                tile_merge.enabled = project_settings.tile_merge_enabled;
                tile_merge.classes_to_not_merge =
                    flags_to_classes(&project_settings.tile_merge_classes_to_not_merge_flags);
                tile_merge.connectivity =
                    index_to_connectivity(project_settings.tile_merge_connectivity);
                tile_merge.max_fragments_per_group =
                    project_settings.tile_merge_max_fragments_per_group.max(1) as u32;
            }
        }

        self.app_state.mark_dirty();
    }

    /// Synchronizes the current project state from the Rust backend to the Slint UI.
    ///
    /// This is typically called when:
    /// 1. A project is first loaded from disk.
    /// 2. Settings are reverted or reset to defaults.
    /// 3. An external event (like a hardware scan) changes the plate dimensions.
    pub fn sync_project_settings_to_slint(&self) {
        let project = self.app_state.get_project();
        let ui_handle = self.ui.clone();
        let results_ui_handle = self.results_ui.clone();

        let (author_name, organization) = {
            let addr = &*self.app_state.app_settings.lock().expect("Poisened");
            if let Some(usr) = &addr.author {
                (usr.full_name.clone(), usr.organization.clone())
            } else {
                ("".into(), "".into())
            }
        };

        let plate = project.plate.clone();
        let (well_auto, well_rows, well_cols) = match plate.well_layout {
            WellLayout::Auto => (true, 4, 4),
            WellLayout::Fixed { rows, cols } => (false, rows as i32, cols as i32),
        };
        let mut shown_order = plate.clone();
        shown_order.set_well_layout(WellLayout::Fixed {
            rows: well_rows as u32,
            cols: well_cols as u32,
        });
        let well_image_order: Vec<i32> = shown_order
            .well_image_order
            .iter()
            .map(|&v| v as i32)
            .collect();
        let regex = plate.grouping_regex.clone();
        let mode_index = grouping_mode_to_index(plate.grouping_mode);
        let plate_size_index = plate_size_to_index(plate.plate_size);
        let plate_size_labels: Vec<SharedString> = PlateSize::ALL
            .iter()
            .map(|size| size.label().into())
            .collect();

        let expirment_name = project.meta.name.clone();

        let (tile_merge_enabled, tile_merge_flags, tile_merge_connectivity, tile_merge_cap) = {
            let tile_merge = &project.tile_merge;
            (
                tile_merge.enabled,
                classes_to_flags(&tile_merge.classes_to_not_merge),
                connectivity_to_index(tile_merge.connectivity),
                tile_merge.max_fragments_per_group as i32,
            )
        };

        crate::helper::ui_thread::invoke_from_event_loop(move || {
            // Each window has its own independent `ProjectSettingsState`
            // instance (see the struct-level doc comment), so each needs its
            // own `ProjectSettingsSlint` value - in particular its own
            // `VecModel` for `well_values`, which can't be shared across them.
            let build_settings = || ProjectSettingsSlint {
                author_name: author_name.clone().into(),
                organization_name: organization.clone().into(),
                project_name: expirment_name.clone().into(),
                well_rows,
                well_columns: well_cols,
                well_auto,
                well_values: slint::ModelRc::from(std::rc::Rc::new(slint::VecModel::from(
                    well_image_order.clone(),
                ))),
                custom_regex: regex.clone().into(),
                grouping_mode: mode_index,
                plate_size_index,
                tile_merge_enabled,
                tile_merge_classes_to_not_merge_flags: slint::ModelRc::from(std::rc::Rc::new(
                    slint::VecModel::from(tile_merge_flags.clone()),
                )),
                tile_merge_connectivity,
                tile_merge_max_fragments_per_group: tile_merge_cap,
            };
            let labels = || {
                slint::ModelRc::from(std::rc::Rc::new(slint::VecModel::from(
                    plate_size_labels.clone(),
                )))
            };

            if let Some(ui) = ui_handle.upgrade() {
                let state = ui.global::<ProjectSettingsState>();
                state.set_plate_size_labels(labels());
                state.set_settings(build_settings());
            }
            if let Some(results_ui) = results_ui_handle.upgrade() {
                let state = results_ui.global::<ProjectSettingsState>();
                state.set_plate_size_labels(labels());
                state.set_settings(build_settings());
            }
        })
        .ok();

        // Not holding the project while the listener reads it.
        drop(project);
        if let Some(listener) = self
            .plate_settings_listener
            .lock()
            .expect("Poisoned")
            .as_ref()
        {
            listener();
        }
    }
}

/// Updates a single cell in the well-order model — shared by both windows'
/// `on_well_value_changed` handlers (see the struct-level doc comment).
fn set_well_value(model: &ModelRc<i32>, index: i32, value: i32) {
    if let Some(vec_model) = model.as_any().downcast_ref::<slint::VecModel<i32>>() {
        let idx = index as usize;
        if idx < vec_model.row_count() {
            vec_model.set_row_data(idx, value);
        }
    }
}

/// Resizes the well-order model to match new well row/col counts — shared by
/// both windows' `on_well_dims_changed` handlers (see the struct-level doc
/// comment).
fn resize_well_values(model: &ModelRc<i32>, rows: i32, cols: i32) {
    let new_size = (rows * cols).max(0) as usize;
    if let Some(vec_model) = model.as_any().downcast_ref::<slint::VecModel<i32>>() {
        let current = vec_model.row_count();
        if new_size > current {
            for i in current..new_size {
                vec_model.push((i + 1) as i32);
            }
        } else {
            while vec_model.row_count() > new_size {
                vec_model.remove(vec_model.row_count() - 1);
            }
        }
    }
}

/// Converts `classes_to_not_merge` into the 33-element ("1"/"0" for classes
/// 0-32) selection-flags shape `MultiClassDropdown` (shared with the
/// pipeline command editor) expects. Class IDs >= 33 have no slot in this
/// picker and are silently dropped, same as the pipeline editor's own
/// multi-class fields.
fn classes_to_flags(classes: &[ObjectClass]) -> Vec<slint::SharedString> {
    let mut flags = vec![slint::SharedString::from("0"); 33];
    for c in classes {
        if let Some(id) = c.to_u32() {
            if let Some(slot) = flags.get_mut(id as usize) {
                *slot = "1".into();
            }
        }
    }
    flags
}

/// Inverse of `classes_to_flags`.
fn flags_to_classes(flags: &slint::ModelRc<slint::SharedString>) -> Vec<ObjectClass> {
    flags
        .iter()
        .enumerate()
        .filter(|(_, f)| f.as_str() == "1")
        .map(|(i, _)| ObjectClass::Valid(i as u32))
        .collect()
}

/// Flips one class's exclusion flag in response to a "toggle:N" event from
/// the tile-merge exclude-classes `MultiClassDropdown` - shared by both
/// windows' `on_tile_merge_class_toggled` handlers (see the struct-level doc
/// comment).
fn toggle_tile_merge_class(model: &ModelRc<SharedString>, value: &str) {
    let Some(index_str) = value.strip_prefix("toggle:") else {
        return;
    };
    let Ok(index) = index_str.parse::<usize>() else {
        return;
    };
    if let Some(vec_model) = model
        .as_any()
        .downcast_ref::<slint::VecModel<slint::SharedString>>()
    {
        if index < vec_model.row_count() {
            let current = vec_model.row_data(index).unwrap_or_default();
            let flipped: slint::SharedString = if current == "1" { "0" } else { "1" }.into();
            vec_model.set_row_data(index, flipped);
        }
    }
}

fn index_to_connectivity(index: i32) -> TileMergeConnectivity {
    match index {
        0 => TileMergeConnectivity::FourConnected,
        _ => TileMergeConnectivity::EightConnected,
    }
}

fn connectivity_to_index(connectivity: TileMergeConnectivity) -> i32 {
    match connectivity {
        TileMergeConnectivity::FourConnected => 0,
        TileMergeConnectivity::EightConnected => 1,
    }
}

/// The dialog's plate fields as `PlateSettings`. `previous`: the project's
/// current settings - a well image order is kept for when a fixed well size
/// is chosen again.
fn plate_settings_from_slint(
    settings: &ProjectSettingsSlint,
    previous: &PlateSettings,
) -> PlateSettings {
    let mut plate = PlateSettings {
        grouping_mode: index_to_grouping_mode(settings.grouping_mode),
        grouping_regex: settings.custom_regex.to_string(),
        plate_size: index_to_plate_size(settings.plate_size_index),
        well_layout: previous.well_layout,
        well_image_order: previous.well_image_order.clone(),
    };
    if settings.well_auto {
        plate.set_well_layout(WellLayout::Auto);
    } else {
        plate.well_image_order = settings
            .well_values
            .iter()
            .map(|v| v.max(0) as u32)
            .collect();
        plate.set_well_layout(WellLayout::Fixed {
            rows: settings.well_rows.max(1) as u32,
            cols: settings.well_columns.max(1) as u32,
        });
    }
    plate
}

/// The grouping dropdown: 0 Auto, 1 Custom regex, 2 Folder.
fn index_to_grouping_mode(index: i32) -> GroupingMode {
    match index {
        1 => GroupingMode::Custom,
        2 => GroupingMode::Folder,
        _ => GroupingMode::Auto,
    }
}

fn grouping_mode_to_index(mode: GroupingMode) -> i32 {
    match mode {
        GroupingMode::Auto => 0,
        GroupingMode::Custom => 1,
        GroupingMode::Folder => 2,
    }
}

/// The plate size dropdown lists `PlateSize::ALL`.
fn index_to_plate_size(index: i32) -> PlateSize {
    usize::try_from(index)
        .ok()
        .and_then(|i| PlateSize::ALL.get(i).copied())
        .unwrap_or_default()
}

fn plate_size_to_index(size: PlateSize) -> i32 {
    PlateSize::ALL.iter().position(|s| *s == size).unwrap_or(0) as i32
}

#[cfg(test)]
mod tests {
    use slint::VecModel;

    use super::*;

    // -- grouping / plate size / well size mappings ---------------------------

    #[test]
    fn grouping_mode_index_round_trips_every_mode() {
        for mode in [
            GroupingMode::Auto,
            GroupingMode::Custom,
            GroupingMode::Folder,
        ] {
            assert_eq!(index_to_grouping_mode(grouping_mode_to_index(mode)), mode);
        }
        assert_eq!(index_to_grouping_mode(99), GroupingMode::Auto);
    }

    #[test]
    fn plate_size_index_round_trips_every_size_and_auto_is_first() {
        assert_eq!(index_to_plate_size(0), PlateSize::Auto);
        for size in PlateSize::ALL {
            assert_eq!(index_to_plate_size(plate_size_to_index(size)), size);
        }
        assert_eq!(index_to_plate_size(-1), PlateSize::Auto);
        assert_eq!(index_to_plate_size(99), PlateSize::Auto);
    }

    #[test]
    fn plate_settings_from_slint_reads_auto_and_a_fixed_well_size() {
        let mut settings = sample_settings();
        let plate = plate_settings_from_slint(&settings, &PlateSettings::default());
        assert_eq!(plate.grouping_mode, GroupingMode::Custom);
        assert_eq!(plate.grouping_regex, "^(x)");
        assert_eq!(plate.plate_size, PlateSize::Plate8x12);
        assert_eq!(plate.well_layout, WellLayout::Fixed { rows: 2, cols: 3 });
        assert_eq!(plate.well_image_order, vec![1, 2, 3, 6, 5, 4]);

        // Auto keeps the order for when a fixed size is chosen again.
        settings.well_auto = true;
        let auto = plate_settings_from_slint(&settings, &plate);
        assert_eq!(auto.well_layout, WellLayout::Auto);
        assert_eq!(auto.well_image_order, vec![1, 2, 3, 6, 5, 4]);
    }

    // -- update_project_settings_in_project / sync_project_settings_to_slint ------

    use crate::editor::test_support::test_ui_state;

    fn sample_settings() -> ProjectSettingsSlint {
        ProjectSettingsSlint {
            author_name: "Ada Lovelace".into(),
            organization_name: "Analytical Engines".into(),
            project_name: "Test Project".into(),
            well_rows: 2,
            well_columns: 3,
            well_auto: false,
            well_values: slint::ModelRc::new(slint::VecModel::from(vec![1, 2, 3, 6, 5, 4])),
            custom_regex: "^(x)".into(),
            grouping_mode: 1,
            plate_size_index: plate_size_to_index(PlateSize::Plate8x12),
            tile_merge_enabled: true,
            tile_merge_classes_to_not_merge_flags: slint::ModelRc::new(slint::VecModel::from(
                vec![slint::SharedString::from("0"); 33],
            )),
            tile_merge_connectivity: 1,
            tile_merge_max_fragments_per_group: 10000,
        }
    }

    fn make_controller(ui_state: Arc<UiState>) -> ProjectSettingsController {
        ProjectSettingsController::new(slint::Weak::default(), slint::Weak::default(), ui_state)
    }

    #[test]
    fn update_project_settings_in_project_writes_author_and_plate_fields() {
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());

        controller.update_project_settings_in_project(&sample_settings());

        let user_settings = &*ui_state.app_settings.lock().expect("Poisned");
        assert_eq!(
            user_settings.author.clone().unwrap().full_name,
            "Ada Lovelace"
        );
        assert_eq!(
            user_settings.author.clone().unwrap().organization,
            "Analytical Engines"
        );

        let project = ui_state.get_project();
        assert_eq!(project.meta.name, "Test Project");
        assert_eq!(
            project.plate.well_layout,
            WellLayout::Fixed { rows: 2, cols: 3 }
        );
        assert_eq!(project.plate.plate_size, PlateSize::Plate8x12);
        assert_eq!(project.plate.grouping_mode, GroupingMode::Custom);
    }

    #[test]
    fn update_project_well_values() {
        let sample_data: Vec<i32> = vec![10, 20, 30, 42, 100];
        let model: ModelRc<i32> = ModelRc::new(VecModel::from(sample_data));
        set_well_value(&model, 2, 99);
        assert_eq!(model.row_data(2).unwrap(), 99);
    }

    #[test]
    fn update_project_well_values_out_of_scope() {
        let sample_data: Vec<i32> = vec![10, 20, 30, 42, 100];
        let model: ModelRc<i32> = ModelRc::new(VecModel::from(sample_data));
        set_well_value(&model, 8, 99);
        assert_eq!(model.row_data(0).unwrap(), 10);
        assert_eq!(model.row_data(1).unwrap(), 20);
        assert_eq!(model.row_data(2).unwrap(), 30);
        assert_eq!(model.row_data(3).unwrap(), 42);
        assert_eq!(model.row_data(4).unwrap(), 100);
    }

    #[test]
    fn update_well_size() {
        let sample_data: Vec<i32> = vec![1, 2, 3, 4];
        let model: ModelRc<i32> = ModelRc::new(VecModel::from(sample_data));
        resize_well_values(&model, 2, 2);
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), 1);
        assert_eq!(model.row_data(1).unwrap(), 2);
        assert_eq!(model.row_data(2).unwrap(), 3);
        assert_eq!(model.row_data(3).unwrap(), 4);

        resize_well_values(&model, 3, 2);
        assert_eq!(model.row_count(), 6);
        assert_eq!(model.row_data(0).unwrap(), 1);
        assert_eq!(model.row_data(1).unwrap(), 2);
        assert_eq!(model.row_data(2).unwrap(), 3);
        assert_eq!(model.row_data(3).unwrap(), 4);
        assert_eq!(model.row_data(4).unwrap(), 5);
        assert_eq!(model.row_data(5).unwrap(), 6);

        resize_well_values(&model, 2, 2);
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), 1);
        assert_eq!(model.row_data(1).unwrap(), 2);
        assert_eq!(model.row_data(2).unwrap(), 3);
        assert_eq!(model.row_data(3).unwrap(), 4);
    }

    #[test]
    fn test_toggle_tile_merge_class() {
        let sample_data: Vec<String> = vec!["0".into(), "1".into(), "1".into(), "0".into()];
        let model: ModelRc<SharedString> = ModelRc::new(VecModel::from(
            sample_data
                .into_iter()
                .map(SharedString::from)
                .collect::<Vec<_>>(),
        ));
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), "0");
        assert_eq!(model.row_data(1).unwrap(), "1");
        assert_eq!(model.row_data(2).unwrap(), "1");
        assert_eq!(model.row_data(3).unwrap(), "0");

        toggle_tile_merge_class(&model, "toggle:1");
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), "0");
        assert_eq!(model.row_data(1).unwrap(), "0");
        assert_eq!(model.row_data(2).unwrap(), "1");
        assert_eq!(model.row_data(3).unwrap(), "0");

        toggle_tile_merge_class(&model, "toggle:1");
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), "0");
        assert_eq!(model.row_data(1).unwrap(), "1");
        assert_eq!(model.row_data(2).unwrap(), "1");
        assert_eq!(model.row_data(3).unwrap(), "0");

        toggle_tile_merge_class(&model, "toggle:0");
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), "1");
        assert_eq!(model.row_data(1).unwrap(), "1");
        assert_eq!(model.row_data(2).unwrap(), "1");
        assert_eq!(model.row_data(3).unwrap(), "0");
    }

    #[test]
    fn test_toggle_tile_abnormal_merge_class() {
        let sample_data: Vec<String> = vec!["0".into(), "1".into(), "1".into(), "0".into()];
        let model: ModelRc<SharedString> = ModelRc::new(VecModel::from(
            sample_data
                .into_iter()
                .map(SharedString::from)
                .collect::<Vec<_>>(),
        ));

        // Toggle out of range
        toggle_tile_merge_class(&model, "toggle:6");
        // Wrong syntax
        toggle_tile_merge_class(&model, "wrong_text:6");
        // Wrong text and bad number
        toggle_tile_merge_class(&model, "wrong_text:bad");
        // Wrong number
        toggle_tile_merge_class(&model, "toggle:bad");
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0).unwrap(), "0");
        assert_eq!(model.row_data(1).unwrap(), "1");
        assert_eq!(model.row_data(2).unwrap(), "1");
        assert_eq!(model.row_data(3).unwrap(), "0");
    }

    // -- tile merging -----------------------------------------------------

    #[test]
    fn classes_to_flags_sets_only_the_given_classes_and_drops_out_of_range_ids() {
        let flags = classes_to_flags(&[
            ObjectClass::Valid(1),
            ObjectClass::Valid(3),
            ObjectClass::Valid(99), // out of the 0-32 picker range, dropped
        ]);
        assert_eq!(flags.len(), 33);
        assert_eq!(flags[1], "1");
        assert_eq!(flags[3], "1");
        for (i, flag) in flags.iter().enumerate() {
            if i != 1 && i != 3 {
                assert_eq!(flag, "0", "index {i} must be unset");
            }
        }
    }

    #[test]
    fn flags_to_classes_is_the_inverse_of_classes_to_flags() {
        let classes = vec![ObjectClass::Valid(2), ObjectClass::Valid(5)];
        let flags = classes_to_flags(&classes);
        let model: slint::ModelRc<slint::SharedString> =
            slint::ModelRc::new(slint::VecModel::from(flags));
        assert_eq!(flags_to_classes(&model), classes);
    }

    #[test]
    fn connectivity_index_round_trips_both_variants() {
        for connectivity in [
            TileMergeConnectivity::FourConnected,
            TileMergeConnectivity::EightConnected,
        ] {
            assert_eq!(
                index_to_connectivity(connectivity_to_index(connectivity)),
                connectivity
            );
        }
    }

    #[test]
    fn update_project_settings_in_project_writes_tile_merge_fields() {
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());

        let mut settings = sample_settings();
        settings.tile_merge_enabled = true;
        settings.tile_merge_classes_to_not_merge_flags =
            slint::ModelRc::new(slint::VecModel::from(classes_to_flags(&[
                ObjectClass::Valid(1),
                ObjectClass::Valid(2),
            ])));
        settings.tile_merge_connectivity = 0;
        settings.tile_merge_max_fragments_per_group = 500;

        controller.update_project_settings_in_project(&settings);

        let project = ui_state.get_project();
        assert!(project.tile_merge.enabled);
        assert_eq!(
            project.tile_merge.classes_to_not_merge,
            vec![ObjectClass::Valid(1), ObjectClass::Valid(2)]
        );
        assert_eq!(
            project.tile_merge.connectivity,
            TileMergeConnectivity::FourConnected
        );
        assert_eq!(project.tile_merge.max_fragments_per_group, 500);
    }

    #[test]
    fn update_project_settings_in_project_clamps_max_fragments_per_group_to_at_least_one() {
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());

        let mut settings = sample_settings();
        settings.tile_merge_max_fragments_per_group = 0;

        controller.update_project_settings_in_project(&settings);

        assert_eq!(ui_state.get_project().tile_merge.max_fragments_per_group, 1);
    }

    #[test]
    fn update_project_settings_in_project_marks_the_project_dirty() {
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state.clone());

        controller.update_project_settings_in_project(&sample_settings());

        assert!(ui_state.is_dirty());
    }

    #[test]
    fn sync_project_settings_to_slint_does_not_panic_without_a_live_ui() {
        let ui_state = test_ui_state();
        let controller = make_controller(ui_state);
        controller.sync_project_settings_to_slint();
    }
}
