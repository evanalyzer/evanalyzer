use crate::{ResultsListState, ResultsState, UiState};
use evanalyzer_app::result::{self, ResultsGenerator};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_gui_slint::ResultsWindow;
use log::{error, info, warn};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
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
                .on_list_class_filter_selected(move |_class_name| {});
            ui.global::<ResultsState>()
                .on_list_column_toggled(move |_label| {});
            ui.global::<ResultsState>()
                .on_list_column_group_toggled(move |_group| {});
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
                .on_matrix_class_filter_selected(move |_class_name| {});
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
                .on_chart_class_filter_clicked(move || {});

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
            Ok(results) => match results.get_object_classes() {
                Ok(classes) => {
                    self.show_results_window();
                    self.set_object_classes_in_slint(&classes);
                    *self.result_generator.lock().expect("Poisned".into()) = Some(results);
                }
                Err(err) => {
                    error!("{}", err);
                }
            },
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
        let mut options: Vec<SharedString> = vec!["All classes".into()];
        options.extend(
            object_classes
                .iter()
                .map(|class| SharedString::from(class.name.as_str())),
        );
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ResultsState>()
                    .set_class_filter_options(ModelRc::from(Rc::new(VecModel::from(options))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_object_classes_in_slint, cannot update class filter options!"
                );
            }
        })
        .ok();
    }
}
