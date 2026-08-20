use crate::UiState;
use crate::{AppWindow, ProjectSettingsSlint, ProjectSettingsState, ResultsWindow};
use evanalyzer_cfg::settings::plate_settings::GroupingMode;
use slint::{ComponentHandle, Model};
use std::sync::Arc;

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
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
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
                        set_well_value(ui.global::<ProjectSettingsState>(), index, value);
                    }
                });

            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_well_dims_changed(move |rows, cols| {
                    if let Some(ui) = ui_weak.upgrade() {
                        resize_well_values(ui.global::<ProjectSettingsState>(), rows, cols);
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
                        set_well_value(results_ui.global::<ProjectSettingsState>(), index, value);
                    }
                });

            let results_ui_weak = self.results_ui.clone();
            results_ui
                .global::<ProjectSettingsState>()
                .on_well_dims_changed(move |rows, cols| {
                    if let Some(results_ui) = results_ui_weak.upgrade() {
                        resize_well_values(results_ui.global::<ProjectSettingsState>(), rows, cols);
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
                let meta = &mut project.metadata;
                let full_name: String = project_settings.author_name.clone().into();
                // This field only ever edits the primary author (authors[0]);
                // any co-authors past that (only settable today by
                // hand-editing the project file) are left untouched.
                match meta.authors.first_mut() {
                    Some(primary) => *primary = full_name,
                    None if !full_name.is_empty() => meta.authors.push(full_name),
                    None => {}
                }
                meta.author_organization = project_settings.organization_name.clone().into();
                meta.name = project_settings.project_name.clone().into();
            }

            // Plate settings
            {
                let plate = &mut project.plate;
                let (mode, regex) = index_to_grouping_mode(
                    project_settings.grouping_mode,
                    &project_settings.custom_regex.clone().into(),
                );
                plate.grouping_mode = mode;
                plate.grouping_regex = regex;

                let (plate_rows, plate_cols) = index_to_well_size(project_settings.well_size_index);
                plate.plate_rows = plate_rows;
                plate.plate_cols = plate_cols;

                plate.well_rows = project_settings.well_rows;
                plate.well_cols = project_settings.well_columns;
                plate.well_image_order = project_settings.well_values.iter().collect();
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
            let addr = &project.metadata;
            let full_name = addr.authors.first().cloned().unwrap_or_default();
            (full_name, addr.author_organization.clone())
        };

        let (plate_rows, plate_cols, well_rows, well_cols, well_image_order, regex, mode_index) = {
            let plate = &project.plate;
            let well_values: Vec<i32> = plate.well_image_order.clone();

            (
                plate.plate_rows,
                plate.plate_cols,
                plate.well_rows,
                plate.well_cols,
                well_values,
                plate.grouping_regex.clone(),
                grouping_mode_to_index(&plate.grouping_mode, &plate.grouping_regex.clone()),
            )
        };

        let expirment_name = project.metadata.name.clone();

        slint::invoke_from_event_loop(move || {
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
                well_values: slint::ModelRc::from(std::rc::Rc::new(slint::VecModel::from(
                    well_image_order.clone(),
                ))),
                custom_regex: regex.clone().into(),
                grouping_mode: mode_index,
                well_size_index: well_size_to_idx(plate_rows, plate_cols),
                plate_rows,
                plate_cols,
            };

            if let Some(ui) = ui_handle.upgrade() {
                ui.global::<ProjectSettingsState>()
                    .set_settings(build_settings());
            }
            if let Some(results_ui) = results_ui_handle.upgrade() {
                results_ui
                    .global::<ProjectSettingsState>()
                    .set_settings(build_settings());
            }
        })
        .ok();
    }
}

/// Updates a single cell in the well-order model — shared by both windows'
/// `on_well_value_changed` handlers (see the struct-level doc comment).
fn set_well_value(state: ProjectSettingsState<'_>, index: i32, value: i32) {
    let model = state.get_settings().well_values;
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
fn resize_well_values(state: ProjectSettingsState<'_>, rows: i32, cols: i32) {
    let model = state.get_settings().well_values;
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

fn index_to_grouping_mode(index: i32, regex: &String) -> (GroupingMode, String) {
    match index {
        0 => (GroupingMode::NoGrouping, "".into()),
        1 => (GroupingMode::FolderName, "".into()),
        2 => (GroupingMode::FileName, "(.*)_([0-9]*)".into()),
        3 => (GroupingMode::FileName, "((.)([0-9]+))_([0-9]+)".into()),
        _ => (GroupingMode::FileName, regex.into()),
    }
}

fn grouping_mode_to_index(mode: &GroupingMode, regex: &String) -> i32 {
    match mode {
        GroupingMode::NoGrouping => 0,
        GroupingMode::FolderName => 1,
        GroupingMode::FileName => match regex.as_str() {
            "(.*)_([0-9]*)" => 2,
            "((.)([0-9]+))_([0-9]+)" => 3,
            _ => 4,
        },
    }
}

/// Returns row and col
pub(crate) fn index_to_well_size(index: i32) -> (i32, i32) {
    match index {
        0 => (1, 1),
        1 => (2, 3),
        2 => (2, 4),
        3 => (2, 6),
        4 => (3, 4),
        5 => (3, 5),
        6 => (3, 6),
        7 => (4, 6),
        8 => (6, 8),
        9 => (8, 12),
        10 => (16, 24),
        11 => (32, 48),
        12 => (48, 72),
        _ => (1, 1),
    }
}

fn well_size_to_idx(row: i32, col: i32) -> i32 {
    for i in 0..=12 {
        if index_to_well_size(i) == (row, col) {
            return i;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- index_to_well_size / well_size_to_idx ---------------------------------

    #[test]
    fn index_to_well_size_covers_every_known_plate_format() {
        assert_eq!(index_to_well_size(0), (1, 1));
        assert_eq!(index_to_well_size(1), (2, 3));
        assert_eq!(index_to_well_size(9), (8, 12));
        assert_eq!(index_to_well_size(12), (48, 72));
    }

    #[test]
    fn index_to_well_size_out_of_range_falls_back_to_1x1() {
        assert_eq!(index_to_well_size(13), (1, 1));
        assert_eq!(index_to_well_size(-1), (1, 1));
    }

    #[test]
    fn well_size_to_idx_is_the_inverse_of_index_to_well_size_for_every_known_index() {
        for i in 0..=12 {
            let (rows, cols) = index_to_well_size(i);
            assert_eq!(well_size_to_idx(rows, cols), i);
        }
    }

    #[test]
    fn well_size_to_idx_of_an_unknown_dimension_pair_falls_back_to_zero() {
        assert_eq!(well_size_to_idx(7, 7), 0);
    }

    // -- index_to_grouping_mode / grouping_mode_to_index -----------------------

    #[test]
    fn index_to_grouping_mode_no_grouping() {
        let (mode, regex) = index_to_grouping_mode(0, &"ignored".to_string());
        assert_eq!(mode, GroupingMode::NoGrouping);
        assert_eq!(regex, "");
    }

    #[test]
    fn index_to_grouping_mode_folder_name() {
        let (mode, regex) = index_to_grouping_mode(1, &"ignored".to_string());
        assert_eq!(mode, GroupingMode::FolderName);
        assert_eq!(regex, "");
    }

    #[test]
    fn index_to_grouping_mode_file_name_presets_carry_their_fixed_regex() {
        let (mode, regex) = index_to_grouping_mode(2, &"ignored".to_string());
        assert_eq!(mode, GroupingMode::FileName);
        assert_eq!(regex, "(.*)_([0-9]*)");

        let (mode, regex) = index_to_grouping_mode(3, &"ignored".to_string());
        assert_eq!(mode, GroupingMode::FileName);
        assert_eq!(regex, "((.)([0-9]+))_([0-9]+)");
    }

    #[test]
    fn index_to_grouping_mode_custom_index_passes_through_the_given_regex() {
        let (mode, regex) = index_to_grouping_mode(4, &"my-custom-regex".to_string());
        assert_eq!(mode, GroupingMode::FileName);
        assert_eq!(regex, "my-custom-regex");
    }

    #[test]
    fn grouping_mode_to_index_round_trips_every_index_to_grouping_mode_case() {
        for (index, regex) in [
            (0, ""),
            (1, ""),
            (2, "(.*)_([0-9]*)"),
            (3, "((.)([0-9]+))_([0-9]+)"),
        ] {
            let (mode, produced_regex) = index_to_grouping_mode(index, &"unused".to_string());
            assert_eq!(produced_regex, regex);
            assert_eq!(grouping_mode_to_index(&mode, &produced_regex), index);
        }
    }

    #[test]
    fn grouping_mode_to_index_unrecognized_regex_maps_to_the_custom_index() {
        let index = grouping_mode_to_index(&GroupingMode::FileName, &"something-else".to_string());
        assert_eq!(index, 4);
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
            well_values: slint::ModelRc::new(slint::VecModel::from(vec![1, 2, 3, 4, 5, 6])),
            custom_regex: "".into(),
            grouping_mode: 0,
            well_size_index: 1,
            plate_rows: 0,
            plate_cols: 0,
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

        let project = ui_state.get_project();
        assert_eq!(project.metadata.authors, vec!["Ada Lovelace".to_string()]);
        assert_eq!(project.metadata.author_organization, "Analytical Engines");
        assert_eq!(project.metadata.name, "Test Project");
        assert_eq!(project.plate.well_rows, 2);
        assert_eq!(project.plate.well_cols, 3);
        // well_size_index=1 -> index_to_well_size(1) == (2, 3), see the test above.
        assert_eq!((project.plate.plate_rows, project.plate.plate_cols), (2, 3));
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
