use crate::{
    BreadcrumbItem, MatrixCell, MatrixLevel, MultiSelectItem, ResultRow, ResultsListState,
    ResultsRailMode, ResultsState, UiState,
};
use evanalyzer_app::result::{
    self, Aggregation, Cell, CellValue, ColorScale, ColorSchema, Column, ColumnEntry,
    DatabaseResult, ImageEntry, ResultsGenerator, WellFilter,
};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_gui_slint::ResultsWindow;
use log::{error, info, warn};
use slint::{Color, ComponentHandle, Model, ModelRc, VecModel};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

const LIST_PAGE_SIZE: i32 = 500;

const DEFAULT_LIST_COLUMNS: [Column; 4] = [
    Column::ObjectId,
    Column::ImageName,
    Column::ObjectClass,
    Column::AreaSizeNm,
];

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

#[derive(Default)]
struct MatrixFilter {
    pub object_classe: ObjectClass,
    pub column: Column,
    pub aggregation: Aggregation,
    pub group_by_regex: String,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
}

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
    result_generator: Mutex<Option<ResultsGenerator>>,
    list_filter: Mutex<ListFilter>,
    matrix_filter: Mutex<Option<MatrixFilter>>,
    plane_filter: Mutex<PlaneFilter>,
    list_page: Mutex<i32>,
    list_page_cursors: Mutex<Vec<Option<String>>>,
    classes: Mutex<Vec<Class>>,
    images: Mutex<Vec<ImageEntry>>,
    available_columns: Mutex<Vec<ColumnEntry>>,
    // Last-rendered plate cells, by well key (e.g. "C12") — so
    // `on_plate_cell_clicked` can look up what to show in the sidebar
    // without re-querying the database.
    matrix_cells: Mutex<HashMap<String, MatrixCell>>,
    // Same idea as `matrix_cells`, one level down: last-rendered well-field
    // cells, by image name, for `on_well_cell_clicked`.
    well_cells: Mutex<HashMap<String, MatrixCell>>,
}

impl ResultsStateController {
    pub fn new(ui: slint::Weak<ResultsWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            result_generator: Mutex::new(None),
            list_filter: Mutex::new(ListFilter::default()),
            matrix_filter: Mutex::new(None),
            plane_filter: Mutex::new(PlaneFilter::default()),
            list_page: Mutex::new(0),
            list_page_cursors: Mutex::new(vec![None]),
            classes: Mutex::new(Vec::new()),
            images: Mutex::new(Vec::new()),
            available_columns: Mutex::new(Vec::new()),
            matrix_cells: Mutex::new(HashMap::new()),
            well_cells: Mutex::new(HashMap::new()),
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
            // The rail buttons already set `rail-mode` themselves
            // (results_common.slint) before firing this — only the
            // breadcrumb baseline needs resetting here, so switching away
            // from Matrix and back doesn't leave a stale "Well X" segment
            // or drop the user back into a well they were last looking at.
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_rail_mode_selected(move |mode| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_rail_mode_selected");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    let breadcrumb = if mode == ResultsRailMode::Matrix {
                        vec![
                            BreadcrumbItem { label: "All results".into() },
                            BreadcrumbItem { label: "Plate".into() },
                        ]
                    } else {
                        vec![BreadcrumbItem { label: "All results".into() }]
                    };
                    state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                    state.set_matrix_level(MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                });

            // Only two drill levels sit below the "All results"/"Plate"
            // base today (see on_open_well_clicked below) — navigating to
            // either of those base segments always means "back to the
            // plate grid", so a length check is enough without tracking
            // which matrix-level each breadcrumb segment represents.
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_breadcrumb_nav(move |index| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_breadcrumb_nav");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    let mut breadcrumb: Vec<BreadcrumbItem> = state.get_breadcrumb().iter().collect();
                    let keep = ((index as usize) + 1).min(breadcrumb.len());
                    breadcrumb.truncate(keep);
                    state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                    if keep <= 2 {
                        state.set_matrix_level(MatrixLevel::Plate);
                        state.set_active_well("".into());
                        state.set_active_well_has_value(false);
                        state.set_active_well_value("".into());
                    }
                });

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
                    let selected_count = list_filter.image_rel_path.len();
                    drop(list_filter);
                    manager.push_image_summary(selected_count);
                    manager.refresh_list();
                });
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_image_select_all(move || manager.select_all_images());
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_image_select_none(move || manager.select_none_images());

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
                    let selected_count = list_filter.object_classes.len();
                    drop(list_filter);
                    manager.push_class_summary(selected_count);
                    manager.refresh_list();
                });
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_class_select_all(move || manager.select_all_classes());
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_class_select_none(move || manager.select_none_classes());

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
                    let selected_count = list_filter.columns.len();
                    drop(list_filter);
                    manager.push_columns_summary(selected_count);
                    manager.refresh_list();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_columns_select_all(move || manager.select_all_columns());
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_columns_select_none(move || manager.select_none_columns());

            let manager = self.clone();
            ui.global::<ResultsState>().on_list_next_page(move || {
                manager.change_list_page(1);
            });
            let manager = self.clone();
            ui.global::<ResultsState>().on_list_prev_page(move || {
                manager.change_list_page(-1);
            });

            // -- Matrix / plate / well / object --
            // The COLUMN dropdown is single-select and its `item-selected`
            // handler (in results_matrix.slint) only forwards a no-argument
            // `matrix-value-clicked()` — the chosen key lives in
            // `matrix-column-items` itself, so it's read back here rather
            // than passed through the callback.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_matrix_value_clicked(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_matrix_value_clicked");
                        return;
                    };
                    let items = ui_ready.global::<ResultsState>().get_matrix_column_items();
                    let Some(item) = items.iter().find(|item| item.selected) else {
                        warn!("No column selected in matrix column picker");
                        return;
                    };
                    let Some(column) = Column::from_key(item.key.as_str()) else {
                        warn!("Unknown matrix column key selected: {}", item.key);
                        return;
                    };
                    manager.update_matrix_filter(|filter| filter.column = column);
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_aggregate_selected(move |aggregate| {
                    let aggregation = match aggregate.as_str() {
                        "Average" => Aggregation::Avg,
                        "Minimum" => Aggregation::Min,
                        "Maximum" => Aggregation::Max,
                        "Std Dev" | "Stddev" => Aggregation::Stddev,
                        "Sum" => Aggregation::Sum,
                        other => {
                            warn!("Unknown matrix aggregate selected: {other}");
                            return;
                        }
                    };
                    manager.update_matrix_filter(|filter| filter.aggregation = aggregation);
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_class_selected(move |key, selected| {
                    // Single-select: `MultiSelectDropdown.toggle()` only ever
                    // fires this with `selected == true` for the newly
                    // picked row, never `false` for the one it replaces.
                    if !selected {
                        return;
                    }
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
                    manager.update_matrix_filter(|filter| filter.object_classe = object_class);
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_regex_changed(move |regex| {
                    manager
                        .update_matrix_filter(|filter| filter.group_by_regex = regex.to_string());
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_matrix_color_schema_selected(move |key, selected| {
                    if !selected {
                        return;
                    }
                    let Some(schema) = color_schema_from_key(key.as_str()) else {
                        warn!("Unknown color schema selected: {key}");
                        return;
                    };
                    if let Some(ui_ready) = ui_weak.upgrade() {
                        let stops = color_scale_gradient_slint(&schema);
                        ui_ready
                            .global::<ResultsState>()
                            .set_matrix_color_scale_stops(ModelRc::from(Rc::new(VecModel::from(
                                stops,
                            ))));
                    } else {
                        warn!(
                            "Failed to upgrade UI handle in on_matrix_color_schema_selected, cannot update legend gradient!"
                        );
                    }
                    manager.update_matrix_filter(|filter| filter.color_schema = schema);
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_scale_set_auto(move || {
                    manager.update_matrix_filter(|filter| filter.color_scale = ColorScale::Auto);
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_scale_set_manual(move |min, max| {
                    manager.update_matrix_filter(|filter| {
                        filter.color_scale = ColorScale::Manual(min, max)
                    });
                    manager.update_matrix_view();
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_plate_cell_clicked(move |key| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_plate_cell_clicked");
                        return;
                    };
                    let cells = manager.matrix_cells.lock().expect("Poisned");
                    let Some(cell) = cells.get(key.as_str()) else {
                        warn!("Unknown well clicked: {key}");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    state.set_active_well(key);
                    state.set_active_well_has_value(cell.has_value);
                    state.set_active_well_value(cell.label.clone());
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_open_well_clicked(move |well| {
                    // The level switch itself already happened in
                    // results_matrix.slint's `clicked` handler — push the
                    // breadcrumb segment for it here and load its fields.
                    if let Some(ui_ready) = ui_weak.upgrade() {
                        let state = ui_ready.global::<ResultsState>();
                        let mut breadcrumb: Vec<BreadcrumbItem> =
                            state.get_breadcrumb().iter().collect();
                        breadcrumb.truncate(2);
                        breadcrumb.push(BreadcrumbItem {
                            label: format!("Well {well}").into(),
                        });
                        state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                    } else {
                        warn!("Failed to upgrade UI handle in on_open_well_clicked");
                    }
                    manager.update_well_view(&well);
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_well_cell_clicked(move |key| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_well_cell_clicked");
                        return;
                    };
                    let cells = manager.well_cells.lock().expect("Poisned");
                    let Some(cell) = cells.get(key.as_str()) else {
                        warn!("Unknown field clicked: {key}");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    state.set_active_well(key);
                    state.set_active_well_has_value(cell.has_value);
                    state.set_active_well_value(cell.label.clone());
                });

            // Not implemented yet (object-level image + marker rendering) —
            // only the level switch (results_matrix.slint's `clicked`
            // handler) and breadcrumb-back (on_breadcrumb_nav above) work
            // so far, same as `open-well-clicked` before `get_group_by_well`
            // existed.
            ui.global::<ResultsState>()
                .on_well_field_clicked(move |image| {
                    info!(
                        "Open image {image} requested, but object-level data isn't wired up yet"
                    );
                });
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
                let mut default_object_classes = Vec::new();
                match results.get_object_classes() {
                    Ok(classes) => {
                        default_object_classes = classes.iter().map(|class| class.id).collect();
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
                        *self.available_columns.lock().expect("Poisened") = columns;
                    }
                    Err(err) => {
                        error!("{}", err);
                    }
                };
                match results.get_images() {
                    Ok(images) => {
                        self.set_images_in_slint(&images);
                        *self.images.lock().expect("Poisened") = images;
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

                self.set_color_schemas_in_slint();
                *self.matrix_filter.lock().expect("Poisned") = None;

                *self.list_filter.lock().expect("Poisened") = ListFilter {
                    image_rel_path: Vec::new(),
                    object_classes: default_object_classes,
                    columns: DEFAULT_LIST_COLUMNS.to_vec(),
                };

                self.show_results_window();
                *self.result_generator.lock().expect("Poisned".into()) = Some(results);
                self.refresh_list();
                self.update_matrix_view();
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
        *self.list_page_cursors.lock().expect("Poisned") = vec![None];
        self.update_list_view();
    }

    // Every Matrix-view callback goes through this: `matrix_filter` starts
    // `None` (see `open_database`) and is lazily created here on whichever
    // control the user touches first, rather than springing into existence
    // fully-formed the moment a database opens.
    fn update_matrix_filter(&self, apply: impl FnOnce(&mut MatrixFilter)) {
        let mut matrix_filter = self.matrix_filter.lock().expect("Poisned");
        apply(matrix_filter.get_or_insert_with(MatrixFilter::default));
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

        let page = *self.list_page.lock().expect("Poisned") as usize;
        let cursor = self
            .list_page_cursors
            .lock()
            .expect("Poisned")
            .get(page)
            .cloned()
            .flatten();

        let Ok(result) = db.get_list(
            &evanalyzer_app::result::ListFilter {
                plane,
                images,
                object_classes,
                columns,
                page: result::Pagination {
                    limit: LIST_PAGE_SIZE,
                    after: cursor,
                },
            },
            &result::View::List,
        ) else {
            warn!("Could not load results!");
            return;
        };

        // Record the cursor for `page + 1` (the last row's `object_id`, from
        // `row_names` — see `get_list` — regardless of whether ObjectId is a
        // visible column) so Next can page forward from here. Only append,
        // never overwrite: revisiting a page via Prev/Next must not disturb
        // the cursor a later page already recorded.
        if let Some(last_id) = result.row_names.last() {
            let mut cursors = self.list_page_cursors.lock().expect("Poisned");
            if cursors.len() == page + 1 {
                cursors.push(Some(last_id.clone()));
            }
        }

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

    pub fn update_matrix_view(&self) {
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

        let matrix_filter_guard = self.matrix_filter.lock().expect("Poisned");
        // Nothing to aggregate on yet (the user hasn't touched the Matrix
        // view since the database was opened) — skip the query rather than
        // running one against `MatrixFilter::default()`'s arbitrary values.
        let Some(matrix_filter) = matrix_filter_guard.as_ref() else {
            return;
        };
        let value_caption = self.value_caption_for(matrix_filter);
        let is_manual_scale = matches!(matrix_filter.color_scale, ColorScale::Manual(..));

        let group_filter = result::PlateFilter {
            plane,
            grouping_regex: matrix_filter.group_by_regex.clone(),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // No UI to pick a fixed plate size yet — `None` has
            // `get_group_by_plate` auto-select the smallest standard
            // dimensions that fit the data.
            matrix_dimension: None,
        };
        drop(matrix_filter_guard);

        let Ok(result) = db.get_group_by_plate(&group_filter, &result::View::Heatmap) else {
            warn!("Could not load matrix results!");
            return;
        };
        self.set_matrix_in_slint(&result, value_caption, is_manual_scale);
    }

    // What the plate/well's values actually are, e.g. "Average Area [px]" —
    // shown in the sidebar for whichever well/field gets clicked, since
    // every cell in a given plate or well render shares this one
    // aggregation/column.
    fn value_caption_for(&self, matrix_filter: &MatrixFilter) -> String {
        let column_display_name = self
            .available_columns
            .lock()
            .expect("Poisned")
            .iter()
            .find(|entry| entry.key == matrix_filter.column)
            .map(|entry| entry.display_name.clone())
            .unwrap_or_else(|| matrix_filter.column.as_key());
        format!(
            "{} {}",
            aggregation_display_name(&matrix_filter.aggregation),
            column_display_name
        )
    }

    // Third drill level from the plate: the fields inside one well. Reuses
    // the current Matrix toolbar's column/aggregation/class/color settings
    // (the well level has no toolbar of its own — see results_matrix.slint)
    // scoped down to `well_id` via `WellFilter::group_name`.
    fn update_well_view(&self, well_id: &str) {
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

        let matrix_filter_guard = self.matrix_filter.lock().expect("Poisned");
        let Some(matrix_filter) = matrix_filter_guard.as_ref() else {
            warn!("No matrix filter set, cannot open well {well_id}");
            return;
        };
        let value_caption = self.value_caption_for(matrix_filter);

        let well_filter = WellFilter {
            plane,
            group_name: well_id.to_string(),
            grouping_regex: matrix_filter.group_by_regex.clone(),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // No UI to pick a well layout yet — `None` has
            // `get_group_by_well` assume the common 4x4 field grid.
            well_size: None,
            well_order: None,
        };
        drop(matrix_filter_guard);

        let Ok(result) = db.get_group_by_well(&well_filter, &result::View::Heatmap) else {
            warn!("Could not load well results for {well_id}!");
            return;
        };
        self.set_well_in_slint(&result, value_caption);
    }

    // `result` is the plate/well grid `get_group_by_plate(.., View::Heatmap)`
    // returns: `row_names`/`column_names` are the grid's two axes and
    // `rows[r][c]` the cell at that position — flattened here into the
    // row-major `plate-cells` array `PlateGrid` (results_matrix.slint)
    // indexes as `r * plate-cols + c`. Sparse plates (a row or column with no
    // objects at all) will shift the grid relative to `PlateGrid`'s
    // hardcoded A/B/C.../1/2/3... labels, since those come from index
    // position, not from `row_names`/`column_names` themselves — an accepted
    // limitation of this first pass, not something fixed here.
    pub fn set_matrix_in_slint(
        &self,
        result: &DatabaseResult,
        value_caption: String,
        is_manual_scale: bool,
    ) {
        let ui_weak = self.ui.clone();
        let rows = result.row_names.len() as i32;
        let cols = result.column_names.len() as i32;
        // `get_group_by_plate` already computed this range to color the
        // cells against — reuse it rather than re-deriving (and risking
        // disagreeing with) it from the cells themselves.
        let range_min = result.min;
        let range_max = result.max;

        let cells = flatten_grid_cells(result);

        // Cache by key for `on_plate_cell_clicked` to look up without
        // re-querying — this replaces whatever the previous matrix refresh
        // cached, so a click after a filter change can't show a stale value.
        *self.matrix_cells.lock().expect("Poisned") = cells
            .iter()
            .map(|cell| (cell.key.to_string(), cell.clone()))
            .collect();

        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected well (if any) belonged to the old
                // data — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_caption(value_caption.into());
                state.set_plate_rows(rows);
                state.set_plate_cols(cols);
                state.set_plate_min(range_min);
                state.set_plate_max(range_max);
                state.set_plate_cells(ModelRc::from(Rc::new(VecModel::from(cells))));
                state.set_matrix_scale_is_manual(is_manual_scale);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_matrix_in_slint, cannot update matrix view!"
                );
            }
        })
        .ok();
    }

    // `result` is the well grid `get_group_by_well(.., View::Heatmap)`
    // returns — same shape `set_matrix_in_slint` renders one level up, just
    // into `well-*` properties instead of `plate-*`, and cached into
    // `well_cells` instead of `matrix_cells` for `on_well_cell_clicked`.
    pub fn set_well_in_slint(&self, result: &DatabaseResult, value_caption: String) {
        let ui_weak = self.ui.clone();
        let rows = result.row_names.len() as i32;
        let cols = result.column_names.len() as i32;
        let range_min = result.min;
        let range_max = result.max;

        let cells = flatten_grid_cells(result);

        *self.well_cells.lock().expect("Poisned") = cells
            .iter()
            .map(|cell| (cell.key.to_string(), cell.clone()))
            .collect();

        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected field (if any) belonged to the
                // old well — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_caption(value_caption.into());
                state.set_well_rows(rows);
                state.set_well_cols(cols);
                state.set_well_min(range_min);
                state.set_well_max(range_max);
                state.set_well_fields(ModelRc::from(Rc::new(VecModel::from(cells))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_well_in_slint, cannot update well view!"
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
        let items = class_filter_items(object_classes, true);
        let matrix_items = class_filter_items(object_classes, false);
        let summary = list_summary(
            items.iter().filter(|item| item.selected).count(),
            items.len(),
            "Classes",
        );
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_class_items(ModelRc::from(Rc::new(VecModel::from(items.clone()))));
                state.set_matrix_class_items(ModelRc::from(Rc::new(VecModel::from(matrix_items))));
                state.set_chart_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_class_summary(summary);
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
        let mut items = column_items(columns, |key| DEFAULT_LIST_COLUMNS.contains(key));
        let groups = column_groups(columns);
        let summary = list_summary(
            items.iter().filter(|item| item.selected).count(),
            items.len(),
            "Columns",
        );
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_columns(ModelRc::from(Rc::new(VecModel::from(items.clone()))));
                // The Matrix view's column picker starts with nothing
                // checked — it's a single-column aggregation target, not a
                // multi-column display set like the List view's, so it
                // shouldn't inherit DEFAULT_LIST_COLUMNS' selections.
                for item in items.iter_mut() {
                    item.selected = false;
                }
                state.set_matrix_column_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_columns_groups(ModelRc::from(Rc::new(VecModel::from(groups))));
                state.set_list_columns_summary(summary);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_columns_in_slint, cannot update column filter options!"
                );
            }
        })
        .ok();
    }

    pub fn set_color_schemas_in_slint(&self) {
        let ui_weak = self.ui.clone();
        let items = color_schema_items();
        let summary = items
            .iter()
            .find(|item| item.selected)
            .map(|item| item.value.clone())
            .unwrap_or_default();
        let stops = color_scale_gradient_slint(&ColorSchema::default());
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_matrix_color_schema_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_matrix_color_scale_summary(summary);
                state.set_matrix_color_scale_stops(ModelRc::from(Rc::new(VecModel::from(stops))));
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_color_schemas_in_slint, cannot update color schema options!"
                );
            }
        })
        .ok();
    }

    pub fn set_images_in_slint(&self, images: &Vec<ImageEntry>) {
        let ui_weak = self.ui.clone();
        let items = image_items(images, false);
        let summary = list_summary(
            items.iter().filter(|item| item.selected).count(),
            items.len(),
            "Images",
        );
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_image_summary(summary);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_columns_in_slint, cannot update column filter options!"
                );
            }
        })
        .ok();
    }

    // Recompute+push a dropdown's "N of M ..." summary after a single item's
    // selection toggles — the initial `set_*_in_slint` calls above compute
    // the same text from the freshly-built item list, but a toggle only
    // touches one item, so it's cheaper to just recombine the new selected
    // count with the cached total than to rebuild the whole item list again.
    fn push_image_summary(&self, selected: usize) {
        let total = self.images.lock().expect("Poisened").len();
        let text = list_summary(selected, total, "Images");
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ResultsState>()
                    .set_list_image_summary(text);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the images summary!");
            }
        })
        .ok();
    }

    fn push_class_summary(&self, selected: usize) {
        let total = self.classes.lock().expect("Poisened").len();
        let text = list_summary(selected, total, "Classes");
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ResultsState>()
                    .set_list_class_summary(text);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the classes summary!");
            }
        })
        .ok();
    }

    fn push_columns_summary(&self, selected: usize) {
        let total = self.available_columns.lock().expect("Poisened").len();
        let text = list_summary(selected, total, "Columns");
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ResultsState>()
                    .set_list_columns_summary(text);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the columns summary!");
            }
        })
        .ok();
    }

    fn select_all_images(&self) {
        let images = self.images.lock().expect("Poisened");
        let items = image_items(&images, true);
        let total = images.len();
        let rel_paths = images.iter().map(|image| image.rel_path.clone()).collect();
        drop(images);
        self.list_filter.lock().expect("Poisened").image_rel_path = rel_paths;
        self.push_image_items(items, list_summary(total, total, "Images"));
        self.refresh_list();
    }

    fn select_none_images(&self) {
        let images = self.images.lock().expect("Poisened");
        let items = image_items(&images, false);
        let total = images.len();
        drop(images);
        // Sentinel: a real image's rel-path is never empty (schema
        // guarantees `image_rel_path` is `NOT NULL`), so this can't match
        // any real image — which is what "select none" needs, since an
        // empty `image_rel_path` list means "no filter" (all images) in
        // `update_list_view`, not "match nothing".
        self.list_filter.lock().expect("Poisened").image_rel_path = vec![PathBuf::new()];
        self.push_image_items(items, list_summary(0, total, "Images"));
        self.refresh_list();
    }

    fn push_image_items(&self, items: Vec<MultiSelectItem>, summary: slint::SharedString) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_image_summary(summary);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the images list!");
            }
        })
        .ok();
    }

    fn select_all_classes(&self) {
        let classes = self.classes.lock().expect("Poisened");
        let items = class_filter_items(&classes, true);
        let total = classes.len();
        let ids = classes.iter().map(|class| class.id).collect();
        drop(classes);
        self.list_filter.lock().expect("Poisened").object_classes = ids;
        self.push_class_items(items, list_summary(total, total, "Classes"));
        self.refresh_list();
    }

    fn select_none_classes(&self) {
        let classes = self.classes.lock().expect("Poisened");
        let items = class_filter_items(&classes, false);
        let total = classes.len();
        drop(classes);
        // Sentinel: `get_object_classes` never returns `ObjectClass::Unset`
        // (only real `Valid` ids), so this can't match any registered class
        // — needed because an empty `object_classes` list means "no filter"
        // (all classes) in `update_list_view`, not "match nothing".
        self.list_filter.lock().expect("Poisened").object_classes = vec![ObjectClass::Unset];
        self.push_class_items(items, list_summary(0, total, "Classes"));
        self.refresh_list();
    }

    // Only the List view's own class dropdown (`list-class-items`) — the
    // Matrix/Charts views' class dropdowns are separate, independent
    // selections (their own not-yet-wired-up callbacks), not something this
    // toggle should overwrite.
    fn push_class_items(&self, items: Vec<MultiSelectItem>, summary: slint::SharedString) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_class_summary(summary);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the class list!");
            }
        })
        .ok();
    }

    fn select_all_columns(&self) {
        let columns = self.available_columns.lock().expect("Poisened");
        let items = column_items(&columns, |_| true);
        let total = columns.len();
        let keys = columns.iter().map(|entry| entry.key.clone()).collect();
        drop(columns);
        self.list_filter.lock().expect("Poisened").columns = keys;
        self.push_columns_items(items, list_summary(total, total, "Columns"));
        self.refresh_list();
    }

    fn select_none_columns(&self) {
        let columns = self.available_columns.lock().expect("Poisened");
        let items = column_items(&columns, |_| false);
        let total = columns.len();
        drop(columns);
        self.list_filter.lock().expect("Poisened").columns = Vec::new();
        self.push_columns_items(items, list_summary(0, total, "Columns"));
        self.refresh_list();
    }

    // Only the List view's own columns dropdown — `matrix-column-items` is a
    // separate, single-column aggregation picker for the Matrix view.
    fn push_columns_items(&self, items: Vec<MultiSelectItem>, summary: slint::SharedString) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_columns(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_columns_summary(summary);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the columns list!");
            }
        })
        .ok();
    }
}

// Shared "N of M <noun>" text for the images/classes/columns dropdown pills
// (e.g. "3 of 12 Columns").
fn list_summary(selected: usize, total: usize, noun: &str) -> slint::SharedString {
    format!("{selected} of {total} {noun}").into()
}

// "All classes" (selected by default) followed by one entry per class from
// the open database's classification settings.
fn class_filter_items(object_classes: &[Class], selected: bool) -> Vec<MultiSelectItem> {
    object_classes
        .iter()
        .map(|class| MultiSelectItem {
            key: class.name.as_str().into(),
            value: class.name.as_str().into(),
            color: Color::default(),
            group: "".into(),
            selected,
        })
        .collect()
}

// `selected` decides each item's checked state by key — DEFAULT_LIST_COLUMNS
// membership for the initial population, or a constant true/false for
// "select all"/"select none".
fn column_items(
    columns: &[ColumnEntry],
    selected: impl Fn(&Column) -> bool,
) -> Vec<MultiSelectItem> {
    columns
        .iter()
        .map(|column| MultiSelectItem {
            key: column.key.as_key().into(),
            value: column.display_name.as_str().into(),
            color: Color::default(),
            group: column.group.as_str().into(),
            selected: selected(&column.key),
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

// Mirrors the strings `on_matrix_aggregate_selected` maps back from — kept
// as the single source of truth for that display text.
fn aggregation_display_name(aggregation: &Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "Average",
        Aggregation::Min => "Minimum",
        Aggregation::Max => "Maximum",
        Aggregation::Stddev => "Std Dev",
        Aggregation::Sum => "Sum",
    }
}

fn color_schemas() -> [(&'static str, ColorSchema); 2] {
    [
        ("Viridis", ColorSchema::Viridis),
        ("Excel", ColorSchema::Excel),
    ]
}

fn color_schema_items() -> Vec<MultiSelectItem> {
    color_schemas()
        .into_iter()
        .map(|(name, schema)| MultiSelectItem {
            key: name.into(),
            value: name.into(),
            color: Color::default(),
            group: "".into(),
            selected: schema == ColorSchema::default(),
        })
        .collect()
}

fn color_schema_from_key(key: &str) -> Option<ColorSchema> {
    color_schemas()
        .into_iter()
        .find(|(name, _)| *name == key)
        .map(|(_, schema)| schema)
}

fn image_items(images: &[ImageEntry], selected: bool) -> Vec<MultiSelectItem> {
    images
        .iter()
        .map(|image| MultiSelectItem {
            key: image.rel_path.to_str().unwrap_or_default().into(),
            value: image.name.as_str().into(),
            color: Color::default(),
            group: "".into(),
            selected,
        })
        .collect()
}

// Flattens a `get_group_by_plate`/`get_group_by_well` `View::Heatmap` result
// into the row-major `MatrixCell` array `PlateGrid`/`WellGrid`
// (results_matrix.slint) index as `r * cols + c` — shared by
// `set_matrix_in_slint` and `set_well_in_slint` since both grids render the
// exact same way, just against `plate-*`/`well-*` properties respectively.
fn flatten_grid_cells(result: &DatabaseResult) -> Vec<MatrixCell> {
    result
        .row_names
        .iter()
        .zip(result.rows.iter())
        .flat_map(|(row_key, row_cells)| {
            result
                .column_names
                .iter()
                .zip(row_cells.iter())
                .map(move |(col_key, cell)| {
                    let (has_value, value, label) = match &cell.value {
                        CellValue::Float(v) => (true, *v, format!("{v:.2}")),
                        _ => (false, 0.0, String::new()),
                    };
                    // Prefer the cell's own search key — the well id
                    // (`get_group_by_plate`) or image name
                    // (`get_group_by_well`) it actually grouped on — over
                    // reconstructing one from row+col, which only happens
                    // to work for the plate view (well id = row+col
                    // letters/numbers) and not the well view (row/col there
                    // are just grid position numbers, not identifiers).
                    let key = cell
                        .search_key
                        .as_ref()
                        .map(|(key, _)| key.clone())
                        .unwrap_or_else(|| format!("{row_key}{col_key}"));
                    MatrixCell {
                        key: key.into(),
                        value,
                        has_value,
                        label: label.into(),
                        color: bg_color_to_slint(cell.bg_color),
                    }
                })
        })
        .collect()
}

// Unpacks a `0xRRGGBB` cell background color (see `value_to_color` in
// evanalyzer_app, and `Class.color`/color_generators.rs for the same
// encoding elsewhere in this codebase) into a Slint `color`.
fn bg_color_to_slint(bg_color: u32) -> Color {
    Color::from_rgb_u8(
        ((bg_color >> 16) & 0xFF) as u8,
        ((bg_color >> 8) & 0xFF) as u8,
        (bg_color & 0xFF) as u8,
    )
}

// Samples `schema` the same way the plate cells are colored (see
// `value_to_color` in results_generator.rs), so the legend bar's gradient
// always matches what's on screen instead of reimplementing the
// interpolation a second time in Slint.
fn color_scale_gradient_slint(schema: &ColorSchema) -> Vec<Color> {
    result::color_scale_gradient(schema)
        .into_iter()
        .map(bg_color_to_slint)
        .collect()
}

fn cell_to_string(cell: &Cell) -> slint::SharedString {
    match &cell.value {
        CellValue::Empty => "".into(),
        CellValue::String(value) => value.as_str().into(),
        CellValue::Float(value) => format!("{value:.3}").into(),
        CellValue::Integer(value) => value.to_string().into(),
        CellValue::Class((name, _color)) => name.as_str().into(),
    }
}
