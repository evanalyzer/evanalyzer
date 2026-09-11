use crate::{MultiSelectItem, ResultRow, ResultsListState, ResultsState, UiState};
use evanalyzer_app::result::{
    self, Cell, Column, ColumnEntry, DatabaseResult, ImageEntry, ResultsGenerator,
};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_gui_slint::ResultsWindow;
use log::{error, info, warn};
use slint::{Color, ComponentHandle, ModelRc, VecModel};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

const LIST_PAGE_SIZE: i32 = 500;

#[derive(Default)]
struct PlaneFilter {
    pub selected_z_stack: u32,
    pub selected_t_stack: u32,
}

#[derive(Default)]
struct ListFilter {
    pub image_rel_path: Vec<PathBuf>,
    pub object_classes: Vec<ObjectClass>,
    pub columns: Vec<Column>,
}

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
    result_generator: Mutex<Option<ResultsGenerator>>,
    list_filter: Mutex<ListFilter>,
    plane_filter: Mutex<PlaneFilter>,
    list_page: Mutex<i32>,
    classes: Mutex<Vec<Class>>,
}

impl ResultsStateController {
    pub fn new(ui: slint::Weak<ResultsWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            result_generator: Mutex::new(None),
            list_filter: Mutex::new(ListFilter::default()),
            plane_filter: Mutex::new(PlaneFilter::default()),
            list_page: Mutex::new(0),
            classes: Mutex::new(Vec::new()),
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
            let manager = self.clone();
            ui.global::<ResultsState>().on_z_changed(move |value| {
                manager
                    .plane_filter
                    .lock()
                    .expect("Poisned")
                    .selected_z_stack = value as u32;
                manager.refresh_list();
            });
            let manager = self.clone();
            ui.global::<ResultsState>().on_t_changed(move |value| {
                manager
                    .plane_filter
                    .lock()
                    .expect("Poisned")
                    .selected_t_stack = value as u32;
                manager.refresh_list();
            });

            // -- List view --
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_image_selected(move |key, selected| {
                    let rel_path = PathBuf::from(key.to_string());
                    let mut list_filter = manager.list_filter.lock().expect("Poisened".into());
                    if selected {
                        if !list_filter.image_rel_path.contains(&rel_path) {
                            list_filter.image_rel_path.push(rel_path);
                        }
                    } else {
                        list_filter.image_rel_path.retain(|p| p != &rel_path);
                    }
                    drop(list_filter);
                    manager.refresh_list();
                });
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_class_selected(move |key, selected| {
                    let class_name = key.to_string();
                    let Some(object_class) = manager
                        .classes
                        .lock()
                        .expect("Poisened")
                        .iter()
                        .find(|class| class.name == class_name)
                        .map(|class| class.id)
                    else {
                        warn!("Unknown class selected: {class_name}");
                        return;
                    };
                    let mut list_filter = manager.list_filter.lock().expect("Poisened".into());
                    if selected {
                        if !list_filter.object_classes.contains(&object_class) {
                            list_filter.object_classes.push(object_class);
                        }
                    } else {
                        list_filter.object_classes.retain(|c| c != &object_class);
                    }
                    drop(list_filter);
                    manager.refresh_list();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_columns_item_selected(move |key, selected| {
                    let Some(column) = Column::from_key(key.as_str()) else {
                        warn!("Unknown column key selected: {key}");
                        return;
                    };
                    let mut list_filter = manager.list_filter.lock().expect("Poisened".into());
                    if selected {
                        if !list_filter.columns.contains(&column) {
                            list_filter.columns.push(column);
                        }
                    } else {
                        list_filter.columns.retain(|c| c != &column);
                    }
                    drop(list_filter);
                    manager.refresh_list();
                });

            ui.global::<ResultsState>()
                .on_list_columns_select_all(move || {});
            ui.global::<ResultsState>()
                .on_list_columns_select_none(move || {});

            let manager = self.clone();
            ui.global::<ResultsState>().on_list_next_page(move || {
                manager.change_list_page(1);
            });
            let manager = self.clone();
            ui.global::<ResultsState>().on_list_prev_page(move || {
                manager.change_list_page(-1);
            });

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
                        *self.classes.lock().expect("Poisened") = classes.clone();
                        self.set_object_classes_in_slint(&classes);
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };

                match results.get_available_columns() {
                    Ok(columns) => {
                        self.set_columns_in_slint(&columns);
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };
                match results.get_images() {
                    Ok(images) => {
                        self.set_images_in_slint(&images);
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };

                *self.plane_filter.lock().expect("Poisned") = PlaneFilter::default();
                self.set_max_z_and_t_stack_in_slint(
                    results.get_nr_of_t_stacks(),
                    results.get_nr_of_z_stacks(),
                );

                self.show_results_window();
                *self.result_generator.lock().expect("Poisned".into()) = Some(results);
                self.refresh_list();
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

    // Called whenever a filter (plane, image, class or column selection)
    // changes: the previous page number no longer means anything for the new
    // filter, so jump back to page 1.
    fn refresh_list(&self) {
        *self.list_page.lock().expect("Poisned") = 0;
        self.update_list_view();
    }

    // Called by the Next/Prev page callbacks. Negative results are clamped
    // to page 1; going past the last page is harmless (the query just comes
    // back empty), but the Next button is disabled once that would happen
    // (see `set_objects_list_in_slint`) so it shouldn't occur in practice.
    fn change_list_page(&self, delta: i32) {
        {
            let mut page = self.list_page.lock().expect("Poisned");
            *page = (*page + delta).max(0);
        }
        self.update_list_view();
    }

    pub fn update_list_view(&self) {
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            warn!("No database opened!");
            return;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = result::PlaneFilter {
            z_stack: plane_filter.selected_z_stack,
            t_stack: plane_filter.selected_t_stack,
        };
        drop(plane_filter);

        let list_filter = self.list_filter.lock().expect("Poisened".into());
        let images = if list_filter.image_rel_path.is_empty() {
            None
        } else {
            Some(
                list_filter
                    .image_rel_path
                    .iter()
                    .filter_map(|p| p.to_str().map(str::to_string))
                    .collect(),
            )
        };
        let object_classes = if list_filter.object_classes.is_empty() {
            None
        } else {
            Some(list_filter.object_classes.clone())
        };
        let columns = list_filter.columns.clone();
        drop(list_filter);

        let page = *self.list_page.lock().expect("Poisned");

        let Ok(result) = db.get_list(
            &evanalyzer_app::result::ListFilter {
                plane,
                images,
                object_classes,
                columns,
                page: result::Pagination {
                    limit: LIST_PAGE_SIZE,
                    offset: page * LIST_PAGE_SIZE,
                },
            },
            &result::View::List,
        ) else {
            warn!("Could not load results!");
            return;
        };
        self.set_objects_list_in_slint(&result);
    }

    // Rust never hands this more than one page of rows (see the LIST_PAGE_SIZE
    // comment on update_list_view), so building fresh VecModels here on every
    // filter change stays cheap no matter how large the underlying database is.
    pub fn set_objects_list_in_slint(&self, result: &DatabaseResult) {
        let ui_weak = self.ui.clone();
        let headers: Vec<slint::SharedString> = result
            .column_names
            .iter()
            .map(|name| name.as_str().into())
            .collect();
        let row_count = result.rows.len() as i32;
        let page_number = *self.list_page.lock().expect("Poisned") + 1;
        // A full page doesn't prove another page exists, but it's the only
        // signal we have without a separate COUNT(*) query — good enough to
        // gate the Next button until real pagination metadata exists.
        let has_next_page = row_count == LIST_PAGE_SIZE;
        // Plain, `Send`-safe data only: the `ModelRc`s that `ResultRow` and
        // the table properties need are `Rc`-based and can't cross the
        // `invoke_from_event_loop` closure boundary, so they're built below
        // once we're back on the UI thread.
        let row_cells: Vec<Vec<slint::SharedString>> = result
            .rows
            .iter()
            .map(|row| row.iter().map(cell_to_string).collect())
            .collect();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let rows: Vec<ResultRow> = row_cells
                    .into_iter()
                    .map(|cells| ResultRow {
                        cells: ModelRc::new(VecModel::from(cells)),
                    })
                    .collect();
                state.set_list_column_headers(ModelRc::from(Rc::new(VecModel::from(headers))));
                state.set_list_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
                state.set_list_row_count(row_count);
                state.set_list_page_number(page_number);
                state.set_list_has_next_page(has_next_page);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_objects_list_in_slint, cannot update results table!"
                );
            }
        })
        .ok();
    }

    pub fn set_max_z_and_t_stack_in_slint(&self, t_stack_max: u32, z_stack_max: u32) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_z_stack_max(z_stack_max.saturating_sub(1) as i32);
                state.set_t_stack_max(t_stack_max.saturating_sub(1) as i32);
                state.set_z_stack_active(z_stack_max > 1);
                state.set_t_stack_active(t_stack_max > 1);
                state.set_z_stack(0);
                state.set_t_stack(0);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_max_z_and_t_stack_in_slint, cannot update plane filter range!"
                );
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
                state.set_list_columns(ModelRc::from(Rc::new(VecModel::from(items.clone()))));
                state.set_matrix_column_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_columns_groups(ModelRc::from(Rc::new(VecModel::from(groups))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_columns_in_slint, cannot update column filter options!"
                );
            }
        })
        .ok();
    }

    pub fn set_images_in_slint(&self, images: &Vec<ImageEntry>) {
        let ui_weak = self.ui.clone();
        let items = image_items(images);
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
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
            key: column.key.as_key().into(),
            value: column.display_name.as_str().into(),
            color: Color::default(),
            group: column.group.as_str().into(),
            selected: false,
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

fn image_items(images: &[ImageEntry]) -> Vec<MultiSelectItem> {
    images
        .iter()
        .map(|image| MultiSelectItem {
            key: image.rel_path.to_str().unwrap_or_default().into(),
            value: image.name.as_str().into(),
            color: Color::default(),
            group: "".into(),
            selected: false,
        })
        .collect()
}

fn cell_to_string(cell: &Cell) -> slint::SharedString {
    match cell {
        Cell::String(value) => value.as_str().into(),
        Cell::Float(value) => format!("{value:.3}").into(),
        Cell::Integer(value) => value.to_string().into(),
        Cell::Class((name, _color)) => name.as_str().into(),
    }
}
