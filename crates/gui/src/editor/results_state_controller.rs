use crate::{MultiSelectItem, ResultsListState, ResultsState, UiState};
use evanalyzer_app::result::{self, ColumnEntry, ResultsGenerator};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_gui_slint::ResultsWindow;
use log::{error, info, warn};
use slint::{Color, ComponentHandle, ModelRc, VecModel};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
    result_generator: Mutex<Option<ResultsGenerator>>,
}

impl ResultsStateController {
    pub fn new(ui: slint::Weak<ResultsWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            result_generator: Mutex::new(None),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_refresh_clicked(move || {});

            let manager = self.clone();
            ui.global::<ResultsListState>()
                .on_open_folder_clicked(move || {});

            // -------- ResultsState (results_state.slint) --------
            // Prototypes only for now - wire up the real behavior next.

            // -- Navigation --
            ui.global::<ResultsState>()
                .on_rail_mode_selected(move |_mode| {});
            ui.global::<ResultsState>()
                .on_breadcrumb_nav(move |_index| {});

            // -- Global Z/T plane filter --
            ui.global::<ResultsState>().on_z_changed(move |_value| {});
            ui.global::<ResultsState>().on_t_changed(move |_value| {});

            // -- List view --
            ui.global::<ResultsState>()
                .on_list_image_filter_changed(move |_filter| {});
            ui.global::<ResultsState>()
                .on_list_class_selected(move |_key, _selected| {});
            ui.global::<ResultsState>()
                .on_list_columns_item_selected(move |_key, _selected| {});
            ui.global::<ResultsState>()
                .on_list_columns_select_all(move || {});
            ui.global::<ResultsState>()
                .on_list_columns_select_none(move || {});

            // -- Matrix / plate / well / object --
            ui.global::<ResultsState>()
                .on_matrix_value_clicked(move || {});
            ui.global::<ResultsState>()
                .on_matrix_aggregate_selected(move |_aggregate| {});
            ui.global::<ResultsState>()
                .on_matrix_class_selected(move |_key, _selected| {});
            ui.global::<ResultsState>()
                .on_matrix_regex_changed(move |_regex| {});
            ui.global::<ResultsState>()
                .on_matrix_color_scale_clicked(move || {});
            ui.global::<ResultsState>()
                .on_plate_cell_clicked(move |_key| {});
            ui.global::<ResultsState>()
                .on_well_field_clicked(move |_key| {});
            ui.global::<ResultsState>()
                .on_object_marker_clicked(move |_id| {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_plate(move || {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_well(move || {});

            // -- Charts --
            ui.global::<ResultsState>()
                .on_chart_kind_selected(move |_kind| {});
            ui.global::<ResultsState>()
                .on_chart_property_clicked(move || {});
            ui.global::<ResultsState>()
                .on_chart_class_selected(move |_key, _selected| {});

            // -- Colocalization --
            ui.global::<ResultsState>()
                .on_coloc_object_selected(move |_id| {});
            ui.global::<ResultsState>()
                .on_coloc_property_toggled(move |_property| {});

            // -- Export --
            ui.global::<ResultsState>()
                .on_export_dialog_open(move || {});
        }
    }

    pub fn open_database(&self, path: PathBuf) {
        info!("Opening database {:?}", path);
        let db = result::ResultsGenerator::open_database(path);
        match db {
            Ok(results) => {
                match results.get_object_classes() {
                    Ok(classes) => {
                        self.show_results_window();
                        self.set_object_classes_in_slint(&classes);
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };

                match results.get_available_columns() {
                    Ok(columns) => {
                        self.show_results_window();
                        self.set_columns_in_slint(&columns);
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };
                *self.result_generator.lock().expect("Poisned".into()) = Some(results);
            }
            Err(err) => {
                error!("{}", err);
            }
        }
    }

    pub fn show_results_window(&self) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                if let Err(e) = ui_ready.show() {
                    error!("Failed to show results window: {e}");
                }
            } else {
                warn!("Failed to upgrade UI handle in open_database, cannot show results window!");
            }
        })
        .ok();
    }

    pub fn set_object_classes_in_slint(&self, object_classes: &Vec<Class>) {
        let ui_weak = self.ui.clone();
        let items = class_filter_items(object_classes);
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_class_items(ModelRc::from(Rc::new(VecModel::from(items.clone()))));
                state
                    .set_matrix_class_items(ModelRc::from(Rc::new(VecModel::from(items.clone()))));
                state.set_chart_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_object_classes_in_slint, cannot update class filter options!"
                );
            }
        })
        .ok();
    }

    pub fn set_columns_in_slint(&self, columns: &Vec<ColumnEntry>) {
        let ui_weak = self.ui.clone();
        let items = column_items(columns);
        let groups = column_groups(columns);
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_columns(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_columns_groups(ModelRc::from(Rc::new(VecModel::from(groups))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_columns_in_slint, cannot update column filter options!"
                );
            }
        })
        .ok();
    }
}

// "All classes" (selected by default) followed by one entry per class from
// the open database's classification settings.
fn class_filter_items(object_classes: &[Class]) -> Vec<MultiSelectItem> {
    let mut items = vec![MultiSelectItem {
        key: "All classes".into(),
        value: "All classes".into(),
        color: Color::default(),
        group: "".into(),
        selected: true,
    }];
    items.extend(object_classes.iter().map(|class| MultiSelectItem {
        key: class.name.as_str().into(),
        value: class.name.as_str().into(),
        color: Color::default(),
        group: "".into(),
        selected: false,
    }));
    items
}

// All columns selected by default so the table shows every available column
// until the user narrows it down via the columns dropdown.
fn column_items(columns: &[ColumnEntry]) -> Vec<MultiSelectItem> {
    columns
        .iter()
        .map(|column| MultiSelectItem {
            key: column.name.as_str().into(),
            value: column.display_name.as_str().into(),
            color: Color::default(),
            group: column.group.as_str().into(),
            selected: true,
        })
        .collect()
}

// Distinct group names in first-seen order, so the dropdown renders sections
// in the same order the columns arrive in.
fn column_groups(columns: &[ColumnEntry]) -> Vec<slint::SharedString> {
    let mut groups: Vec<slint::SharedString> = Vec::new();
    for column in columns {
        let group: slint::SharedString = column.group.as_str().into();
        if !group.is_empty() && !groups.contains(&group) {
            groups.push(group);
        }
    }
    groups
}
