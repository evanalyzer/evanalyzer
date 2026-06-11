use crate::UiState;
use crate::{AppWindow, ProjectSettingsSlint, ProjectSettingsState};
use evanalyzer_cfg::settings::plate_settings::GroupingMode;
use slint::{ComponentHandle, Model};
use std::sync::Arc;

pub struct ProjectSettingsController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
}

impl ProjectSettingsController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self { ui, app_state }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            let manager = self.clone();
            ui.global::<ProjectSettingsState>()
                .on_project_settings_changed(move |project_settings| {
                    manager.update_project_settings_in_project(&project_settings);
                });

            let manager = self.clone();
            ui.global::<ProjectSettingsState>()
                .on_project_settings_canceled(move || {
                    manager.sync_project_settings_to_slint();
                });

            // Update a single cell value in the well order model
            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_well_value_changed(move |index, value| {
                    if let Some(ui) = ui_weak.upgrade() {
                        let model = ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<i32>>()
                        {
                            let idx = index as usize;
                            if idx < vec_model.row_count() {
                                vec_model.set_row_data(idx, value);
                            }
                        }
                    }
                });

            // Resize well_values when well rows or columns change
            let ui_weak = self.ui.clone();
            ui.global::<ProjectSettingsState>()
                .on_well_dims_changed(move |rows, cols| {
                    if let Some(ui) = ui_weak.upgrade() {
                        let model = ui
                            .global::<ProjectSettingsState>()
                            .get_settings()
                            .well_values;
                        let new_size = (rows * cols).max(0) as usize;
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<i32>>()
                        {
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
                let mut parts = full_name.split_whitespace();
                meta.author_first_name = parts.next().unwrap_or("").into();
                meta.author_last_name = parts.next().unwrap_or("").into();
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

        let (author_name, organization) = {
            let addr = &project.metadata;
            let full_name = format!("{} {}", addr.author_first_name, addr.author_last_name)
                .trim()
                .to_string();
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
            if let Some(ui) = ui_handle.upgrade() {
                let model = std::rc::Rc::new(slint::VecModel::from(well_image_order));
                let settings = ProjectSettingsSlint {
                    author_name: author_name.into(),
                    organization_name: organization.into(),
                    project_name: expirment_name.into(),
                    well_rows: well_rows,
                    well_columns: well_cols,
                    well_values: slint::ModelRc::from(model),
                    custom_regex: regex.into(),
                    grouping_mode: mode_index,
                    well_size_index: well_size_to_idx(plate_rows, plate_cols),
                };

                ui.global::<ProjectSettingsState>().set_settings(settings);
            }
        })
        .ok();
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
fn index_to_well_size(index: i32) -> (i32, i32) {
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
