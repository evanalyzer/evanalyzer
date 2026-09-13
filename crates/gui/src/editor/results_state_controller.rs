use crate::editor::images_list_controller::ImagesListController;
use crate::{
    BreadcrumbItem, ChartBoxplotBox, ChartScatterPoint, ExportDialogState, MatrixCell,
    MatrixLevel, MultiSelectItem, ResultRow, ResultsChartKind2, ResultsListState, ResultsRailMode,
    ResultsState, UiState,
};
use evanalyzer_app::result::{
    self, Aggregation, BoxplotFilter, Cell, CellValue, ColorScale, ColorSchema, Column,
    ColumnEntry, DatabaseResult, ExportFormat, GroupedByImageFilter, HistogramFilter, ImageEntry,
    ImageHeatmapFilter, PlateDimensions, ResultCharts, ResultExport, ResultsGenerator,
    ScatterFilter, WellFilter, WellSize,
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
const CHART_HISTOGRAM_BINS: usize = 24;
const CHART_SCATTER_MAX_POINTS: usize = 2000;

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

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ListGroupBy {
    #[default]
    Objects,
    Images,
}

#[derive(Default)]
struct ListFilter {
    pub image_rel_path: Vec<PathBuf>,
    pub object_classes: Vec<ObjectClass>,
    pub columns: Vec<Column>,
    pub with_coloc_details: bool,
    pub group_by: ListGroupBy,
    /// Unused in Objects mode.
    pub aggregations: Vec<Aggregation>,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ChartKind {
    #[default]
    Histogram,
    Scatter,
    Boxplot,
}

#[derive(Default, Clone)]
struct ChartFilter {
    pub kind: ChartKind,
    /// Histogram/Boxplot's single measurement, Scatter's X axis.
    pub column: Column,
    /// Scatter's Y axis only.
    pub y_column: Column,
    /// `None` means no class filter (every object) — doesn't apply to
    /// Boxplot, which always groups by every class present regardless.
    pub object_class: Option<ObjectClass>,
}

#[derive(Default)]
struct MatrixFilter {
    pub object_classe: ObjectClass,
    pub column: Column,
    pub aggregation: Aggregation,
    pub group_by_regex: String,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    // Per-level grid-size overrides, one of which applies depending on
    // `matrix-level` — `None` keeps each level's own default (see
    // `update_matrix_view`/`update_well_view`/`update_image_heatmap_view`).
    pub plate_dimension: Option<PlateDimensions>,
    pub well_size: Option<WellSize>,
    pub square_size: Option<usize>,
}

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) _app_state: Arc<UiState>,
    result_generator: Mutex<Option<ResultsGenerator>>,
    list_filter: Mutex<ListFilter>,
    matrix_filter: Mutex<Option<MatrixFilter>>,
    chart_filter: Mutex<ChartFilter>,
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
    // Same idea again, one level further down: last-rendered image-heatmap
    // tile cells, by tile key (e.g. "R0C0"), for `on_image_heatmap_cell_clicked`.
    image_heatmap_cells: Mutex<HashMap<String, MatrixCell>>,
    // The well currently drilled into, if any — so a toolbar control change
    // (column/aggregate/class/regex/color) while at the Well level
    // refreshes that well's fields instead of the (hidden) plate grid. See
    // `refresh_active_matrix_view`.
    current_well: Mutex<Option<String>>,
    // Same idea one level further down: the image (by `image_rel_path`)
    // currently drilled into, if any, so a toolbar control change while at
    // the Object level refreshes that image's heatmap.
    current_image: Mutex<Option<String>>,
    // Last-rendered List rows' `(image_rel_path, bbox_px)`, by row index —
    // `DatabaseResult::row_locations`, cached the same way `matrix_cells`/
    // `well_cells`/`image_heatmap_cells` are, so `on_list_row_clicked` can
    // navigate to a clicked object's own image without re-querying the
    // database.
    list_row_locations: Mutex<Vec<(String, [u32; 4])>>,
    image_list_controller: Arc<ImagesListController>,
    // The path `result_generator`'s connection was opened from — so
    // `on_export_start_clicked` can open its *own* fresh connection for the
    // (potentially long-running) export, rather than holding
    // `result_generator`'s lock for the whole export and blocking every
    // other List/Matrix query on the UI thread for as long as it runs.
    db_path: Mutex<Option<PathBuf>>,
    // Whether the export dialog's defaults have already been seeded for the
    // currently-open database (see `populate_export_defaults`) — reset to
    // `false` on every `open_database` so a fresh database gets fresh
    // defaults the next time the dialog opens, but re-opening the dialog
    // against the *same* database leaves whatever the user already
    // configured alone. The configuration itself needs no Rust-side mirror:
    // every dialog control (checkboxes, text fields, the
    // MultiSelectDropdowns' own `items`) is two-way bound straight to
    // `ExportDialogState` in Slint, which retains it for the life of the
    // window — `on_export_start_clicked` reads it back from there directly.
    export_populated: Mutex<bool>,
}

impl ResultsStateController {
    pub fn new(
        ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
        image_list_controller: Arc<ImagesListController>,
    ) -> Self {
        Self {
            ui,
            _app_state: app_state.clone(),
            result_generator: Mutex::new(None),
            list_filter: Mutex::new(ListFilter::default()),
            matrix_filter: Mutex::new(None),
            chart_filter: Mutex::new(ChartFilter::default()),
            plane_filter: Mutex::new(PlaneFilter::default()),
            list_page: Mutex::new(0),
            list_page_cursors: Mutex::new(vec![None]),
            classes: Mutex::new(Vec::new()),
            images: Mutex::new(Vec::new()),
            available_columns: Mutex::new(Vec::new()),
            matrix_cells: Mutex::new(HashMap::new()),
            well_cells: Mutex::new(HashMap::new()),
            image_heatmap_cells: Mutex::new(HashMap::new()),
            current_well: Mutex::new(None),
            current_image: Mutex::new(None),
            list_row_locations: Mutex::new(Vec::new()),
            image_list_controller,
            db_path: Mutex::new(None),
            export_populated: Mutex::new(false),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            ui.global::<ResultsListState>()
                .on_refresh_clicked(move || {});

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
            let manager = self.clone();
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
                            BreadcrumbItem {
                                label: "All results".into(),
                            },
                            BreadcrumbItem {
                                label: "Plate".into(),
                            },
                        ]
                    } else {
                        vec![BreadcrumbItem {
                            label: "All results".into(),
                        }]
                    };
                    state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                    state.set_matrix_level(MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    *manager.current_well.lock().expect("Poisned") = None;
                    *manager.current_image.lock().expect("Poisned") = None;
                });

            // Three drill levels sit below the "All results"/"Plate" base
            // today: Plate -> "Well X" -> "{image}" (see on_open_well_clicked
            // and on_well_field_clicked below), so navigating to a breadcrumb
            // segment either lands back on the plate grid (kept <= 2) or
            // back on the well it names (kept == 3, re-entering whichever
            // well was last drilled into, per the reference behavior
            // described at the top of results_window.slint).
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>().on_breadcrumb_nav(move |index| {
                let Some(ui_ready) = ui_weak.upgrade() else {
                    warn!("Failed to upgrade UI handle in on_breadcrumb_nav");
                    return;
                };
                let state = ui_ready.global::<ResultsState>();
                // "All results" (segment 0) sits above the Matrix
                // drill-down entirely — it's the List view's home, not a
                // plate/well level, so clicking it switches tabs instead
                // of just truncating within Matrix.
                if index == 0 {
                    state.set_rail_mode(ResultsRailMode::List);
                    state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(vec![
                        BreadcrumbItem {
                            label: "All results".into(),
                        },
                    ]))));
                    state.set_matrix_level(MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    *manager.current_well.lock().expect("Poisned") = None;
                    *manager.current_image.lock().expect("Poisned") = None;
                    return;
                }
                let mut breadcrumb: Vec<BreadcrumbItem> = state.get_breadcrumb().iter().collect();
                let keep = ((index as usize) + 1).min(breadcrumb.len());
                breadcrumb.truncate(keep);
                state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                if keep <= 2 {
                    state.set_matrix_level(MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    *manager.current_well.lock().expect("Poisned") = None;
                    *manager.current_image.lock().expect("Poisned") = None;
                } else if keep == 3 {
                    state.set_matrix_level(MatrixLevel::Well);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    *manager.current_image.lock().expect("Poisned") = None;
                    if let Some(well_id) = manager.current_well.lock().expect("Poisned").clone() {
                        manager.update_well_view(&well_id);
                    }
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
                manager.refresh_charts();
            });
            let manager = self.clone();
            ui.global::<ResultsState>().on_t_changed(move |value| {
                manager
                    .plane_filter
                    .lock()
                    .expect("Poisned")
                    .selected_t_stack = value as u32;
                manager.refresh_list();
                manager.refresh_charts();
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
                    let classes = manager.classes.lock().expect("Poisened");
                    let Some(column) = Column::from_key(key.as_str(), &classes) else {
                        warn!("Unknown column key selected: {key}");
                        return;
                    };
                    drop(classes);
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
            ui.global::<ResultsState>()
                .on_list_group_by_selected(move |key| {
                    let group_by = match key.as_str() {
                        "images" => ListGroupBy::Images,
                        _ => ListGroupBy::Objects,
                    };
                    manager.list_filter.lock().expect("Poisened").group_by = group_by;
                    manager.refresh_list();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_aggregation_item_selected(move |key, selected| {
                    let Some(aggregation) = aggregation_from_key(key.as_str()) else {
                        warn!("Unknown aggregation key selected: {key}");
                        return;
                    };
                    let mut list_filter = manager.list_filter.lock().expect("Poisened".into());
                    if selected {
                        if !list_filter.aggregations.contains(&aggregation) {
                            list_filter.aggregations.push(aggregation);
                        }
                    } else {
                        list_filter.aggregations.retain(|a| a != &aggregation);
                    }
                    let selected_count = list_filter.aggregations.len();
                    drop(list_filter);
                    manager.push_aggregation_summary(selected_count);
                    manager.refresh_list();
                });
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_aggregation_select_all(move || manager.select_all_aggregations());
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_aggregation_select_none(move || manager.select_none_aggregations());

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_with_coloc_details_changed(move |enabled| {
                    manager
                        .list_filter
                        .lock()
                        .expect("Poisened")
                        .with_coloc_details = enabled;
                    manager.refresh_list();
                });

            let manager = self.clone();
            ui.global::<ResultsState>().on_list_next_page(move || {
                manager.change_list_page(1);
            });
            let manager = self.clone();
            ui.global::<ResultsState>().on_list_prev_page(move || {
                manager.change_list_page(-1);
            });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_list_row_clicked(move |index| {
                    let location = manager
                        .list_row_locations
                        .lock()
                        .expect("Poisned")
                        .get(index as usize)
                        .cloned();
                    let Some((rel_path, bbox_px)) = location else {
                        warn!("No location cached for clicked list row {index}");
                        return;
                    };
                    manager
                        .image_list_controller
                        .open_image_and_highlight_object(&PathBuf::from(rel_path), bbox_px, false);
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
                    let classes = manager.classes.lock().expect("Poisened");
                    let Some(column) = Column::from_key(item.key.as_str(), &classes) else {
                        warn!("Unknown matrix column key selected: {}", item.key);
                        return;
                    };
                    drop(classes);
                    // Count is a row tally, not a per-object measurement —
                    // averaging/summing/etc. it doesn't mean anything beyond
                    // the count itself, so the Aggregate picker is disabled
                    // while it's selected (see `aggregate_sql` in
                    // results_generator.rs, which ignores `Aggregation`
                    // entirely for `Column::Count` and always uses COUNT(*)).
                    ui_ready
                        .global::<ResultsState>()
                        .set_matrix_aggregate_enabled(!matches!(column, Column::Count));
                    manager.update_matrix_filter(|filter| filter.column = column);
                    manager.refresh_active_matrix_view();
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
                        "Median" => Aggregation::Median,
                        "Skewness" => Aggregation::Skewness,
                        other => {
                            warn!("Unknown matrix aggregate selected: {other}");
                            return;
                        }
                    };
                    manager.update_matrix_filter(|filter| filter.aggregation = aggregation);
                    manager.refresh_active_matrix_view();
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
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_regex_changed(move |regex| {
                    manager
                        .update_matrix_filter(|filter| filter.group_by_regex = regex.to_string());
                    manager.refresh_active_matrix_view();
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
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_scale_set_auto(move || {
                    manager.update_matrix_filter(|filter| filter.color_scale = ColorScale::Auto);
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_scale_set_manual(move |min, max| {
                    manager.update_matrix_filter(|filter| {
                        filter.color_scale = ColorScale::Manual(min, max)
                    });
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_plate_size_selected(move |key, selected| {
                    if !selected {
                        return;
                    }
                    let Some(dimension) = plate_dimension_from_key(key.as_str()) else {
                        warn!("Unknown plate size selected: {key}");
                        return;
                    };
                    manager.update_matrix_filter(|filter| filter.plate_dimension = dimension);
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_well_rows_changed(move |value| {
                    let Ok(rows) = value.trim().parse::<usize>() else {
                        warn!("Invalid well rows value: {value}");
                        return;
                    };
                    manager.update_matrix_filter(|filter| {
                        let mut size = filter.well_size.unwrap_or(WellSize { rows: 4, cols: 4 });
                        size.rows = rows.max(1);
                        filter.well_size = Some(size);
                    });
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_well_cols_changed(move |value| {
                    let Ok(cols) = value.trim().parse::<usize>() else {
                        warn!("Invalid well cols value: {value}");
                        return;
                    };
                    manager.update_matrix_filter(|filter| {
                        let mut size = filter.well_size.unwrap_or(WellSize { rows: 4, cols: 4 });
                        size.cols = cols.max(1);
                        filter.well_size = Some(size);
                    });
                    manager.refresh_active_matrix_view();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_square_size_selected(move |key, selected| {
                    if !selected {
                        return;
                    }
                    let Ok(size) = key.parse::<usize>() else {
                        warn!("Unknown square size selected: {key}");
                        return;
                    };
                    manager.update_matrix_filter(|filter| filter.square_size = Some(size));
                    manager.refresh_active_matrix_view();
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
                    *manager.current_well.lock().expect("Poisned") = Some(well.to_string());
                    *manager.current_image.lock().expect("Poisned") = None;
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

            // `well-field-clicked` fires with the clicked cell's key, which
            // is the image *name* (see `flatten_grid_cells`'s search-key
            // preference — `get_group_by_well`'s cells carry `(image_name,
            // image_rel_path)`), but `get_image_heatmap` needs the
            // `image_rel_path`. Reuses the already-loaded `images` list
            // (populated in `open_database`) to translate one to the other,
            // the same way `update_list_view`'s image filter does.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_well_field_clicked(move |image_name| {
                    let rel_path = manager
                        .images
                        .lock()
                        .expect("Poisened")
                        .iter()
                        .find(|image| image.name == image_name.as_str())
                        .map(|image| image.rel_path.to_string_lossy().into_owned());
                    let Some(rel_path) = rel_path else {
                        warn!("Unknown image clicked: {image_name}");
                        return;
                    };
                    if let Some(ui_ready) = ui_weak.upgrade() {
                        let state = ui_ready.global::<ResultsState>();
                        let mut breadcrumb: Vec<BreadcrumbItem> =
                            state.get_breadcrumb().iter().collect();
                        breadcrumb.truncate(3);
                        breadcrumb.push(BreadcrumbItem {
                            label: image_name.clone(),
                        });
                        state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(breadcrumb))));
                        state.set_active_image_name(image_name);
                    } else {
                        warn!("Failed to upgrade UI handle in on_well_field_clicked");
                    }
                    *manager.current_image.lock().expect("Poisned") = Some(rel_path.clone());
                    manager.update_image_heatmap_view(&rel_path);
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_image_heatmap_cell_clicked(move |key| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_heatmap_cell_clicked");
                        return;
                    };
                    let cells = manager.image_heatmap_cells.lock().expect("Poisned");
                    let Some(cell) = cells.get(key.as_str()) else {
                        warn!("Unknown heatmap tile clicked: {key}");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    state.set_active_well(key.clone());
                    state.set_active_well_has_value(cell.has_value);
                    state.set_active_well_value(cell.label.clone());
                    drop(cells);

                    // Paint the clicked tile's own bounds as a rectangle
                    // over the image it belongs to — unlike a list row's
                    // crosshair (`on_list_row_clicked`), the tile itself
                    // *is* the region of interest, see
                    // `ObjectHighlightBox::paint_as_rectangle`.
                    let Some(rel_path) = manager.current_image.lock().expect("Poisned").clone()
                    else {
                        warn!("No image open, cannot highlight heatmap tile {key}");
                        return;
                    };
                    let Some((row, col)) = parse_tile_key(&key) else {
                        warn!("Unrecognized heatmap tile key: {key}");
                        return;
                    };
                    let square_size = manager
                        .matrix_filter
                        .lock()
                        .expect("Poisned")
                        .as_ref()
                        .and_then(|filter| filter.square_size)
                        .unwrap_or(DEFAULT_SQUARE_SIZE)
                        as u32;
                    let xmin = col * square_size;
                    let ymin = row * square_size;
                    let bbox_px = [xmin, ymin, xmin + square_size - 1, ymin + square_size - 1];
                    manager
                        .image_list_controller
                        .open_image_and_highlight_object(&PathBuf::from(rel_path), bbox_px, true);
                });
            ui.global::<ResultsState>()
                .on_object_marker_clicked(move |_id| {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_plate(move || {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_well(move || {});

            // -- Charts --
            let manager = self.clone();
            ui.global::<ResultsState>().on_chart_kind_selected(move |kind| {
                let kind = match kind {
                    ResultsChartKind2::Histogram => ChartKind::Histogram,
                    ResultsChartKind2::Scatter => ChartKind::Scatter,
                    ResultsChartKind2::Boxplot => ChartKind::Boxplot,
                };
                manager.chart_filter.lock().expect("Poisened").kind = kind;
                manager.refresh_charts();
            });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_column_selected(move |key| {
                    let classes = manager.classes.lock().expect("Poisened");
                    let Some(column) = Column::from_key(key.as_str(), &classes) else {
                        warn!("Unknown chart column key selected: {key}");
                        return;
                    };
                    drop(classes);
                    manager.chart_filter.lock().expect("Poisened").column = column;
                    manager.refresh_charts();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_y_column_selected(move |key| {
                    let classes = manager.classes.lock().expect("Poisened");
                    let Some(column) = Column::from_key(key.as_str(), &classes) else {
                        warn!("Unknown chart Y column key selected: {key}");
                        return;
                    };
                    drop(classes);
                    manager.chart_filter.lock().expect("Poisened").y_column = column;
                    manager.refresh_charts();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_class_selected(move |key, selected| {
                    // Single-select, same as `on_matrix_class_selected`:
                    // only ever fires with `selected == true` for the
                    // newly picked row.
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
                    manager.chart_filter.lock().expect("Poisened").object_class = Some(object_class);
                    manager.refresh_charts();
                });

            // -- Export --
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>().on_export_dialog_open(move || {
                let Some(ui_ready) = ui_weak.upgrade() else {
                    warn!("Failed to upgrade UI handle in on_export_dialog_open");
                    return;
                };
                let already_populated = {
                    let mut populated = manager.export_populated.lock().expect("Poisened");
                    let was_populated = *populated;
                    *populated = true;
                    was_populated
                };
                if !already_populated {
                    manager.populate_export_defaults(&ui_ready);
                }
                ui_ready.global::<ExportDialogState>().set_active(true);
            });

            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_pick_output_dir(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_pick_output_dir");
                        return;
                    };
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        ui_ready
                            .global::<ExportDialogState>()
                            .set_output_dir(path.to_string_lossy().into_owned().into());
                    }
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_image_item_selected(move |_key, _selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_item_selected");
                        return;
                    };
                    let state = ui_ready.global::<ExportDialogState>();
                    let items = state.get_image_items();
                    let selected = items.iter().filter(|item| item.selected).count();
                    let total = manager.images.lock().expect("Poisened").len();
                    state.set_image_summary(image_summary_text(selected, total));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_image_select_all(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_select_all");
                        return;
                    };
                    let images = manager.images.lock().expect("Poisened");
                    let items = image_items(&images, true);
                    let total = images.len();
                    drop(images);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_image_summary(image_summary_text(total, total));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_image_select_none(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_select_none");
                        return;
                    };
                    let images = manager.images.lock().expect("Poisened");
                    let items = image_items(&images, false);
                    let total = images.len();
                    drop(images);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_image_summary(image_summary_text(0, total));
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_class_item_selected(move |_key, _selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_class_item_selected");
                        return;
                    };
                    let state = ui_ready.global::<ExportDialogState>();
                    let items = state.get_class_items();
                    let selected = items.iter().filter(|item| item.selected).count();
                    let total = manager.classes.lock().expect("Poisened").len();
                    state.set_class_summary(list_summary(selected, total, "Classes"));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_class_select_all(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_class_select_all");
                        return;
                    };
                    let classes = manager.classes.lock().expect("Poisened");
                    let items = class_filter_items(&classes, true);
                    let total = classes.len();
                    drop(classes);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_class_summary(list_summary(total, total, "Classes"));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_class_select_none(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_class_select_none");
                        return;
                    };
                    let classes = manager.classes.lock().expect("Poisened");
                    let items = class_filter_items(&classes, false);
                    let total = classes.len();
                    drop(classes);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_class_summary(list_summary(0, total, "Classes"));
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_column_item_selected(move |_key, _selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_column_item_selected");
                        return;
                    };
                    let state = ui_ready.global::<ExportDialogState>();
                    let items = state.get_column_items();
                    let selected = items.iter().filter(|item| item.selected).count();
                    let total = manager.available_columns.lock().expect("Poisened").len();
                    state.set_column_summary(list_summary(selected, total, "Columns"));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_column_select_all(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_column_select_all");
                        return;
                    };
                    let columns = manager.available_columns.lock().expect("Poisened");
                    let classes = manager.classes.lock().expect("Poisened");
                    let items = column_items(&columns, &classes, |_| true);
                    drop(classes);
                    let total = columns.len();
                    drop(columns);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_column_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_column_summary(list_summary(total, total, "Columns"));
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_column_select_none(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_column_select_none");
                        return;
                    };
                    let columns = manager.available_columns.lock().expect("Poisened");
                    let classes = manager.classes.lock().expect("Poisened");
                    let items = column_items(&columns, &classes, |_| false);
                    drop(classes);
                    let total = columns.len();
                    drop(columns);
                    let state = ui_ready.global::<ExportDialogState>();
                    state.set_column_items(ModelRc::from(Rc::new(VecModel::from(items))));
                    state.set_column_summary(list_summary(0, total, "Columns"));
                });

            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_aggregation_item_selected(move |_key, _selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_aggregation_item_selected");
                        return;
                    };
                    let state = ui_ready.global::<ExportDialogState>();
                    let items = state.get_aggregation_items();
                    let selected = items.iter().filter(|item| item.selected).count();
                    state.set_aggregation_summary(list_summary(
                        selected,
                        AGGREGATIONS.len(),
                        "Aggregations",
                    ));
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>().on_start_clicked(move || {
                let Some(ui_ready) = ui_weak.upgrade() else {
                    warn!("Failed to upgrade UI handle in on_start_clicked");
                    return;
                };
                let state = ui_ready.global::<ExportDialogState>();
                match manager.read_export_settings(&ui_ready) {
                    Ok(export) => {
                        state.set_is_exporting(true);
                        state.set_done(false);
                        state.set_has_error(false);
                        state.set_error_message("".into());
                        state.set_progress_message("Starting export…".into());
                        state.set_progress_current(0);
                        state.set_progress_total(1);
                        manager.run_export(export);
                    }
                    Err(message) => {
                        state.set_has_error(true);
                        state.set_error_message(message.into());
                    }
                }
            });

            ui.global::<ExportDialogState>()
                .on_close_clicked(move || {});
        }
    }

    pub fn open_database(&self, path: PathBuf) {
        info!("Opening database {:?}", path);
        let db = result::ResultsGenerator::open_database(path.clone());
        match db {
            Ok(results) => {
                *self.db_path.lock().expect("Poisened") = Some(path);
                *self.export_populated.lock().expect("Poisened") = false;
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
                self.set_grid_size_options_in_slint();
                *self.current_well.lock().expect("Poisned") = None;
                *self.current_image.lock().expect("Poisned") = None;

                *self.list_filter.lock().expect("Poisened") = ListFilter {
                    image_rel_path: Vec::new(),
                    object_classes: default_object_classes,
                    columns: DEFAULT_LIST_COLUMNS.to_vec(),
                    with_coloc_details: false,
                    group_by: ListGroupBy::Objects,
                    aggregations: vec![Aggregation::Avg],
                };

                // Charts default to the first aggregable column for both
                // Property (X) and Y — mirrors `matrix_filter`/`list_filter`
                // above being reset to fresh defaults for the newly opened
                // database rather than lingering on the previous one's
                // selections.
                let available_columns = self.available_columns.lock().expect("Poisened").clone();
                let classes_for_charts = self.classes.lock().expect("Poisened").clone();
                let default_chart_column = available_columns
                    .iter()
                    .map(|entry| entry.key.clone())
                    .find(is_chartable_column)
                    .unwrap_or_default();
                *self.chart_filter.lock().expect("Poisened") = ChartFilter {
                    kind: ChartKind::Histogram,
                    column: default_chart_column.clone(),
                    y_column: default_chart_column.clone(),
                    object_class: None,
                };
                let chart_column_items_vec =
                    chart_column_items(&available_columns, &classes_for_charts, &default_chart_column);
                let default_chart_column_label = default_chart_column.display_label(&classes_for_charts);

                // Matrix used to stay at `None` here until the user touched
                // one of its own toolbar controls (`update_matrix_filter`
                // lazily creates it) — meaning the very first time the
                // Plate view was shown, `update_matrix_view` saw `None` and
                // skipped the query entirely, leaving whatever `plate-cells`
                // happened to already hold on screen while
                // `matrix-column-summary` sat at its own unrelated .slint
                // default ("Count") — a label that never actually matched
                // what (if anything) was queried. Seeding it here with a
                // real default column, in sync with `matrix-column-items`'
                // own selection, means the very first query run after
                // opening a database is already for the same column the
                // toolbar displays as selected.
                let default_matrix_column = available_columns
                    .iter()
                    .map(|entry| entry.key.clone())
                    .find(is_matrix_column)
                    .unwrap_or_default();
                *self.matrix_filter.lock().expect("Poisned") = Some(MatrixFilter {
                    column: default_matrix_column.clone(),
                    ..MatrixFilter::default()
                });
                let matrix_eligible_columns: Vec<ColumnEntry> = available_columns
                    .iter()
                    .filter(|entry| is_matrix_column(&entry.key))
                    .cloned()
                    .collect();
                let matrix_column_items_vec = column_items(
                    &matrix_eligible_columns,
                    &classes_for_charts,
                    |key| *key == default_matrix_column,
                );
                let default_matrix_column_label =
                    default_matrix_column.display_label(&classes_for_charts);

                let ui_weak = self.ui.clone();
                slint::invoke_from_event_loop(move || {
                    if let Some(ui_ready) = ui_weak.upgrade() {
                        let state = ui_ready.global::<ResultsState>();
                        state.set_list_with_coloc_details(false);
                        state.set_list_group_by_summary("Objects".into());
                        state.set_list_group_by_items(ModelRc::from(Rc::new(VecModel::from(
                            group_by_items(false),
                        ))));
                        state.set_list_aggregate(list_summary(1, AGGREGATIONS.len(), "Aggregations"));
                        state.set_list_aggregation_items(ModelRc::from(Rc::new(VecModel::from(
                            aggregation_items(&[Aggregation::Avg]),
                        ))));

                        state.set_matrix_column_items(ModelRc::from(Rc::new(VecModel::from(
                            matrix_column_items_vec,
                        ))));
                        state.set_matrix_column_summary(default_matrix_column_label.into());
                        // Same "Count can't be aggregated" rule
                        // `on_matrix_value_clicked` applies on every later
                        // column change — applied here too so the very
                        // first render is consistent with it.
                        state.set_matrix_aggregate_enabled(!matches!(
                            default_matrix_column,
                            Column::Count
                        ));

                        state.set_chart_kind(ResultsChartKind2::Histogram);
                        state.set_chart_column_items(ModelRc::from(Rc::new(VecModel::from(
                            chart_column_items_vec.clone(),
                        ))));
                        state.set_chart_column_summary(default_chart_column_label.clone().into());
                        state.set_chart_y_column_items(ModelRc::from(Rc::new(VecModel::from(
                            chart_column_items_vec,
                        ))));
                        state.set_chart_y_column_summary(default_chart_column_label.into());
                        state.set_chart_class_summary("No Class".into());
                        state.set_chart_error("".into());
                    } else {
                        warn!(
                            "Failed to upgrade UI handle in open_database, cannot reset the coloc-details toggle and matrix aggregate state!"
                        );
                    }
                })
                .ok();

                self.show_results_window();
                *self.result_generator.lock().expect("Poisned".into()) = Some(results);
                self.refresh_list();
                self.update_matrix_view();
                self.refresh_charts();
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

    /// Runs whichever of `get_histogram`/`get_scatter`/`get_boxplot`
    /// matches `chart_filter.kind` and pushes the result into
    /// `ResultsState.chart-*` — called whenever anything the query depends
    /// on changes (chart kind, column(s), class filter, z/t plane, or a
    /// fresh database).
    fn refresh_charts(&self) {
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            return;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = result::PlaneFilter {
            z_stack: plane_filter.selected_z_stack,
            t_stack: plane_filter.selected_t_stack,
        };
        drop(plane_filter);

        let chart_filter = self.chart_filter.lock().expect("Poisened").clone();
        let object_classes = chart_filter.object_class.map(|class| vec![class]);
        let charts = ResultCharts {};

        match chart_filter.kind {
            ChartKind::Histogram => {
                match charts.paint_histogram(
                    db,
                    &HistogramFilter {
                        plane,
                        images: None,
                        object_classes,
                        column: chart_filter.column,
                        bins: CHART_HISTOGRAM_BINS,
                    },
                ) {
                    Ok(histogram) => self.push_histogram_in_slint(&histogram),
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
            ChartKind::Scatter => {
                match charts.paint_scatter(
                    db,
                    &ScatterFilter {
                        plane,
                        images: None,
                        object_classes,
                        x_column: chart_filter.column,
                        y_column: chart_filter.y_column,
                        max_points: Some(CHART_SCATTER_MAX_POINTS),
                    },
                ) {
                    Ok(scatter) => self.push_scatter_in_slint(&scatter),
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
            ChartKind::Boxplot => {
                match charts.paint_boxplot(
                    db,
                    &BoxplotFilter {
                        plane,
                        images: None,
                        // Boxplot always groups by every class present —
                        // the CLASS dropdown doesn't gate it the way it
                        // does Histogram/Scatter (see `ChartFilter::object_class`'s
                        // doc comment).
                        object_classes: None,
                        column: chart_filter.column,
                    },
                ) {
                    Ok(boxplot) => self.push_boxplot_in_slint(&boxplot.boxes),
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
        }
    }

    fn push_chart_error(&self, message: String) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_chart_error(message.into());
                state.set_chart_histogram_bins(ModelRc::from(Rc::new(VecModel::from(
                    Vec::<f32>::new(),
                ))));
                state.set_chart_scatter_points(ModelRc::from(Rc::new(VecModel::from(
                    Vec::<ChartScatterPoint>::new(),
                ))));
                state.set_chart_boxplot_boxes(ModelRc::from(Rc::new(VecModel::from(
                    Vec::<ChartBoxplotBox>::new(),
                ))));
            } else {
                warn!("Failed to upgrade UI handle, cannot show chart error!");
            }
        })
        .ok();
    }

    fn push_histogram_in_slint(&self, result: &result::HistogramResult) {
        let max_count = result.counts.iter().copied().max().unwrap_or(0).max(1);
        let bins: Vec<f32> = result
            .counts
            .iter()
            .map(|count| *count as f32 / max_count as f32)
            .collect();
        let object_count = result.counts.iter().sum::<u64>() as i32;
        let min_label = format_chart_value(result.min);
        let max_label = format_chart_value(result.max);

        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_chart_error("".into());
                state.set_chart_histogram_bins(ModelRc::from(Rc::new(VecModel::from(bins))));
                state.set_chart_histogram_min_label(min_label.into());
                state.set_chart_histogram_max_label(max_label.into());
                state.set_chart_histogram_object_count(object_count);
            } else {
                warn!("Failed to upgrade UI handle, cannot update histogram chart!");
            }
        })
        .ok();
    }

    fn push_scatter_in_slint(&self, result: &result::ScatterResult) {
        let x_range = (result.x_max - result.x_min).max(f64::EPSILON);
        let y_range = (result.y_max - result.y_min).max(f64::EPSILON);
        let points: Vec<ChartScatterPoint> = result
            .points
            .iter()
            .map(|point| ChartScatterPoint {
                x: ((point.x - result.x_min) / x_range) as f32,
                y: ((point.y - result.y_min) / y_range) as f32,
            })
            .collect();
        let shown_count = points.len() as i32;
        let total_count = result.total_object_count as i32;
        let x_min_label = format_chart_value(result.x_min);
        let x_max_label = format_chart_value(result.x_max);
        let y_min_label = format_chart_value(result.y_min);
        let y_max_label = format_chart_value(result.y_max);

        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_chart_error("".into());
                state.set_chart_scatter_points(ModelRc::from(Rc::new(VecModel::from(points))));
                state.set_chart_scatter_shown_count(shown_count);
                state.set_chart_scatter_total_count(total_count);
                state.set_chart_scatter_x_min_label(x_min_label.into());
                state.set_chart_scatter_x_max_label(x_max_label.into());
                state.set_chart_scatter_y_min_label(y_min_label.into());
                state.set_chart_scatter_y_max_label(y_max_label.into());
            } else {
                warn!("Failed to upgrade UI handle, cannot update scatter chart!");
            }
        })
        .ok();
    }

    fn push_boxplot_in_slint(&self, boxes: &[evanalyzer_app::result::BoxplotBox]) {
        // Every box normalized against the combined min/max across *all*
        // boxes, so they stay comparable to each other on one shared axis
        // rather than each box silently rescaling to its own range.
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for b in boxes {
            min = min.min(b.min).min(b.outliers.iter().copied().fold(b.min, f64::min));
            max = max.max(b.max).max(b.outliers.iter().copied().fold(b.max, f64::max));
        }
        if !min.is_finite() || !max.is_finite() {
            min = 0.0;
            max = 0.0;
        }
        let range = (max - min).max(f64::EPSILON);
        let norm = move |v: f64| ((v - min) / range) as f32;

        // Plain, `Send`-safe intermediate rows — `ChartBoxplotBox` itself
        // embeds a `ModelRc` (for `outliers`), and `ModelRc`s can't cross
        // the `invoke_from_event_loop` closure boundary (same reasoning as
        // `set_objects_list_in_slint`'s own row-building), so building the
        // real Slint struct happens after upgrading back onto the UI
        // thread, not here.
        let rows: Vec<(String, u32, f32, f32, f32, f32, f32, Vec<f32>, i32)> = boxes
            .iter()
            .map(|b| {
                (
                    b.label.clone(),
                    b.color,
                    norm(b.min),
                    norm(b.q1),
                    norm(b.median),
                    norm(b.q3),
                    norm(b.max),
                    b.outliers.iter().map(|v| norm(*v)).collect::<Vec<f32>>(),
                    b.object_count as i32,
                )
            })
            .collect();

        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let boxes: Vec<ChartBoxplotBox> = rows
                    .into_iter()
                    .map(
                        |(label, color, min, q1, median, q3, max, outliers, object_count)| {
                            ChartBoxplotBox {
                                label: label.into(),
                                color: bg_color_to_slint(color),
                                min,
                                q1,
                                median,
                                q3,
                                max,
                                outliers: ModelRc::from(Rc::new(VecModel::from(outliers))),
                                object_count,
                            }
                        },
                    )
                    .collect();
                state.set_chart_error("".into());
                state.set_chart_boxplot_boxes(ModelRc::from(Rc::new(VecModel::from(boxes))));
            } else {
                warn!("Failed to upgrade UI handle, cannot update boxplot chart!");
            }
        })
        .ok();
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
        let with_coloc_details = list_filter.with_coloc_details;
        let group_by = list_filter.group_by;
        let aggregations = list_filter.aggregations.clone();
        drop(list_filter);

        let page = *self.list_page.lock().expect("Poisned") as usize;
        let cursor = self
            .list_page_cursors
            .lock()
            .expect("Poisned")
            .get(page)
            .cloned()
            .flatten();

        let result = match group_by {
            ListGroupBy::Objects => db.get_object_list(&evanalyzer_app::result::ListFilter {
                plane,
                images,
                object_classes,
                columns,
                with_coloc_details,
                page: result::Pagination {
                    limit: LIST_PAGE_SIZE,
                    after: cursor,
                },
            }),
            // Non-aggregable columns (Object ID/Image/Class) don't mean
            // anything once rows are grouped by image — silently dropped
            // here, same as the export side's own grid views (see
            // `is_aggregable` in results_exporter.rs). `aggregations` comes
            // from the AGGREGATE dropdown (results_list.slint), one output
            // column per (selected column x selected aggregation) combo.
            ListGroupBy::Images => {
                let columns = columns.into_iter().filter(is_aggregable_column).collect();
                db.get_grouped_by_image(&GroupedByImageFilter {
                    plane,
                    images,
                    object_classes,
                    columns,
                    aggregation: aggregations,
                    page: result::Pagination {
                        limit: LIST_PAGE_SIZE,
                        after: cursor,
                    },
                })
            }
        };
        let Ok(result) = result else {
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
        // Compared against `source_object_count` (how many source rows the
        // page's query actually matched), not `rows.len()`/`row_count`:
        // `ListFilter::with_coloc_details` fan-out (see
        // `build_coloc_detail_rows` in results_generator.rs) can multiply
        // `rows.len()` past `LIST_PAGE_SIZE` even on the last page, or leave
        // it under `LIST_PAGE_SIZE` on a full page of objects with no
        // partners — only the source count means what this check needs.
        let has_next_page = result.source_object_count as i32 == LIST_PAGE_SIZE;
        // Plain, `Send`-safe data only: the `ModelRc`s that `ResultRow` and
        // the table properties need are `Rc`-based and can't cross the
        // `invoke_from_event_loop` closure boundary, so they're built below
        // once we're back on the UI thread.
        // Every `Cell` in a row carries the same `alternating_color` (see
        // `build_coloc_detail_rows`), so the first cell's flag speaks for
        // the whole row; an empty row (no columns selected) just isn't
        // alternated.
        let row_cells: Vec<(Vec<slint::SharedString>, bool)> = result
            .rows
            .iter()
            .map(|row| {
                let alternating = row.first().is_some_and(|cell| cell.alternating_color);
                (row.iter().map(cell_to_string).collect(), alternating)
            })
            .collect();
        *self.list_row_locations.lock().expect("Poisned") = result.row_locations.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let rows: Vec<ResultRow> = row_cells
                    .into_iter()
                    .map(|(cells, alternating)| ResultRow {
                        cells: ModelRc::new(VecModel::from(cells)),
                        alternating,
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
            // `None` (the "Auto" dropdown option) has `get_group_by_plate`
            // auto-select the smallest standard dimensions that fit the
            // data; otherwise the user's explicit PLATE SIZE choice.
            matrix_dimension: matrix_filter.plate_dimension,
        };
        drop(matrix_filter_guard);

        let result = match db.get_group_by_plate(&group_filter, &result::View::Heatmap) {
            Ok(result) => result,
            Err(err) => {
                error!("Could not load matrix results: {err}");
                return;
            }
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
            .unwrap_or_else(|| {
                matrix_filter
                    .column
                    .as_key(&self.classes.lock().expect("Poisened"))
            });
        format!(
            "{} {}",
            aggregation_display_name(&matrix_filter.aggregation),
            column_display_name
        )
    }

    // Every Matrix-toolbar control (column/aggregate/class/regex/color)
    // stays visible and live at the Well level too (see results_matrix.slint
    // — the well level reuses the exact same toolbar as the plate level,
    // not a separate one), so a change there must refresh whichever grid is
    // actually on screen rather than always the plate's.
    fn refresh_active_matrix_view(&self) {
        let Some(ui_ready) = self.ui.upgrade() else {
            warn!("Failed to upgrade UI handle in refresh_active_matrix_view");
            return;
        };
        let level = ui_ready.global::<ResultsState>().get_matrix_level();
        if level == MatrixLevel::Object {
            if let Some(image_rel_path) = self.current_image.lock().expect("Poisned").clone() {
                self.update_image_heatmap_view(&image_rel_path);
                return;
            }
        } else if level == MatrixLevel::Well {
            if let Some(well_id) = self.current_well.lock().expect("Poisned").clone() {
                self.update_well_view(&well_id);
                return;
            }
        }
        self.update_matrix_view();
    }

    // Third drill level from the plate: the fields inside one well. Reuses
    // the current Matrix toolbar's column/aggregation/class/color settings
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
        let is_manual_scale = matches!(matrix_filter.color_scale, ColorScale::Manual(..));

        let well_filter = WellFilter {
            plane,
            group_name: well_id.to_string(),
            grouping_regex: matrix_filter.group_by_regex.clone(),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // `None` has `get_group_by_well` assume the common 4x4 field
            // grid; otherwise the user's explicit ROWS/COLS choice.
            well_size: matrix_filter.well_size,
            well_order: None,
        };
        drop(matrix_filter_guard);

        let result = match db.get_group_by_well(&well_filter, &result::View::Heatmap) {
            Ok(result) => result,
            Err(err) => {
                error!("Could not load well results for {well_id}: {err}");
                return;
            }
        };
        self.set_well_in_slint(&result, value_caption, is_manual_scale);
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
    pub fn set_well_in_slint(
        &self,
        result: &DatabaseResult,
        value_caption: String,
        is_manual_scale: bool,
    ) {
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
                state.set_matrix_scale_is_manual(is_manual_scale);
            } else {
                warn!("Failed to upgrade UI handle in set_well_in_slint, cannot update well view!");
            }
        })
        .ok();
    }

    // Fourth (and last) drill level: a heatmap over one image's own pixels.
    // Reuses the current Matrix toolbar's column/aggregation/class/color
    // settings scoped down to `image_rel_path` via
    // `ImageHeatmapFilter::image_rel_path` — same pattern as
    // `update_well_view` one level up.
    fn update_image_heatmap_view(&self, image_rel_path: &str) {
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
            warn!("No matrix filter set, cannot open image {image_rel_path}");
            return;
        };
        let value_caption = self.value_caption_for(matrix_filter);
        let is_manual_scale = matches!(matrix_filter.color_scale, ColorScale::Manual(..));

        let image_filter = ImageHeatmapFilter {
            plane,
            image_rel_path: image_rel_path.to_string(),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // `None` has `get_image_heatmap` default to 256px tiles;
            // otherwise the user's explicit SQUARE SIZE choice.
            square_size: matrix_filter.square_size,
        };
        drop(matrix_filter_guard);

        let result = match db.get_image_heatmap(&image_filter, &result::View::Heatmap) {
            Ok(result) => result,
            Err(err) => {
                error!("Could not load image heatmap for {image_rel_path}: {err}");
                return;
            }
        };
        self.set_image_heatmap_in_slint(&result, value_caption, is_manual_scale);
    }

    // `result` is the image-heatmap grid `get_image_heatmap(.., View::Heatmap)`
    // returns — same shape `set_well_in_slint` renders one level up, just
    // into `image-heatmap-*` properties instead of `well-*`, and cached into
    // `image_heatmap_cells` instead of `well_cells` for
    // `on_image_heatmap_cell_clicked`.
    pub fn set_image_heatmap_in_slint(
        &self,
        result: &DatabaseResult,
        value_caption: String,
        is_manual_scale: bool,
    ) {
        let ui_weak = self.ui.clone();
        let rows = result.row_names.len() as i32;
        let cols = result.column_names.len() as i32;
        let range_min = result.min;
        let range_max = result.max;

        let cells = flatten_grid_cells(result);

        *self.image_heatmap_cells.lock().expect("Poisned") = cells
            .iter()
            .map(|cell| (cell.key.to_string(), cell.clone()))
            .collect();

        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected tile (if any) belonged to the old
                // image — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_caption(value_caption.into());
                state.set_image_heatmap_rows(rows);
                state.set_image_heatmap_cols(cols);
                state.set_image_heatmap_min(range_min);
                state.set_image_heatmap_max(range_max);
                state.set_image_heatmap_cells(ModelRc::from(Rc::new(VecModel::from(cells))));
                state.set_matrix_scale_is_manual(is_manual_scale);
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_image_heatmap_in_slint, cannot update image heatmap view!"
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
                state.set_list_class_items(ModelRc::from(Rc::new(VecModel::from(items))));
                // Charts' own class picker is single-select, same shape as
                // Matrix's (`matrix_items` — nothing selected by default,
                // meaning "no class filter" rather than "match nothing").
                state.set_matrix_class_items(ModelRc::from(Rc::new(VecModel::from(
                    matrix_items.clone(),
                ))));
                state.set_chart_class_items(ModelRc::from(Rc::new(VecModel::from(matrix_items))));
                state.set_chart_class_summary("No Class".into());
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
        let classes = self.classes.lock().expect("Poisened");
        let items = column_items(columns, &classes, |key| DEFAULT_LIST_COLUMNS.contains(key));
        drop(classes);
        let groups = column_groups(columns);
        let summary = list_summary(
            items.iter().filter(|item| item.selected).count(),
            items.len(),
            "Columns",
        );
        // Matrix view aggregates one column's value across every matched
        // object into a single well/field/tile cell — an object's own
        // identity (ID/Image/Class) isn't a value to aggregate, so those
        // three are excluded here even though they're normal List columns.
        // Also starts with nothing checked — it's a single-column
        // aggregation target, not a multi-column display set like the List
        // view's, so it shouldn't inherit DEFAULT_LIST_COLUMNS' selections.
        let matrix_items: Vec<MultiSelectItem> = items
            .iter()
            .filter(|item| {
                !matches!(
                    item.key.as_str(),
                    "object_id" | "image_name" | "object_class_name"
                )
            })
            .cloned()
            .map(|mut item| {
                item.selected = false;
                item
            })
            .collect();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_columns(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_matrix_column_items(ModelRc::from(Rc::new(VecModel::from(
                    matrix_items,
                ))));
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

    // Initial population for the three per-level grid-size dropdowns/inputs
    // (see the comment on `ResultsState.matrix-plate-size-items`) — called
    // once at `open_database`, mirroring `set_color_schemas_in_slint`.
    pub fn set_grid_size_options_in_slint(&self) {
        let ui_weak = self.ui.clone();
        let plate_items = plate_size_items(None);
        let plate_summary = plate_items
            .iter()
            .find(|item| item.selected)
            .map(|item| item.value.clone())
            .unwrap_or_default();
        let square_items = square_size_items(DEFAULT_SQUARE_SIZE);
        let square_summary = square_items
            .iter()
            .find(|item| item.selected)
            .map(|item| item.value.clone())
            .unwrap_or_default();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_matrix_plate_size_items(ModelRc::from(Rc::new(VecModel::from(
                    plate_items,
                ))));
                state.set_matrix_plate_size_summary(plate_summary);
                state.set_matrix_square_size_items(ModelRc::from(Rc::new(VecModel::from(
                    square_items,
                ))));
                state.set_matrix_square_size_summary(square_summary);
                state.set_matrix_well_rows("4".into());
                state.set_matrix_well_cols("4".into());
            } else {
                warn!(
                    "Failed to upgrade UI handle in set_grid_size_options_in_slint, cannot update grid size options!"
                );
            }
        })
        .ok();
    }

    pub fn set_images_in_slint(&self, images: &Vec<ImageEntry>) {
        let ui_weak = self.ui.clone();
        let items = image_items(images, false);
        let summary = image_summary_text(
            items.iter().filter(|item| item.selected).count(),
            items.len(),
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
        let text = image_summary_text(selected, total);
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

    fn push_aggregation_summary(&self, selected: usize) {
        let text = list_summary(selected, AGGREGATIONS.len(), "Aggregations");
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready.global::<ResultsState>().set_list_aggregate(text);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the aggregations summary!");
            }
        })
        .ok();
    }

    fn select_all_aggregations(&self) {
        let items = aggregation_items(&AGGREGATIONS);
        self.list_filter.lock().expect("Poisened").aggregations = AGGREGATIONS.to_vec();
        self.push_aggregation_items(
            items,
            list_summary(AGGREGATIONS.len(), AGGREGATIONS.len(), "Aggregations"),
        );
        self.refresh_list();
    }

    fn select_none_aggregations(&self) {
        let items = aggregation_items(&[]);
        self.list_filter.lock().expect("Poisened").aggregations = Vec::new();
        self.push_aggregation_items(items, list_summary(0, AGGREGATIONS.len(), "Aggregations"));
        self.refresh_list();
    }

    fn push_aggregation_items(&self, items: Vec<MultiSelectItem>, summary: slint::SharedString) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_aggregation_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_aggregate(summary);
            } else {
                warn!("Failed to upgrade UI handle, cannot update the aggregation items!");
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
        self.push_image_items(items, image_summary_text(total, total));
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
        self.push_image_items(items, image_summary_text(0, total));
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
        let classes = self.classes.lock().expect("Poisened");
        let items = column_items(&columns, &classes, |_| true);
        drop(classes);
        let total = columns.len();
        let keys = columns.iter().map(|entry| entry.key.clone()).collect();
        drop(columns);
        self.list_filter.lock().expect("Poisened").columns = keys;
        self.push_columns_items(items, list_summary(total, total, "Columns"));
        self.refresh_list();
    }

    fn select_none_columns(&self) {
        let columns = self.available_columns.lock().expect("Poisened");
        let classes = self.classes.lock().expect("Poisened");
        let items = column_items(&columns, &classes, |_| false);
        drop(classes);
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

    // Seeds `ExportDialogState` from whatever's currently selected in the
    // List/Matrix views — called once per opened database, the first time
    // the export dialog is opened against it (see `export_populated` and
    // `on_export_dialog_open`). Runs directly on the UI thread (it's called
    // from a Slint callback), so it pushes into `ExportDialogState` right
    // away rather than through `invoke_from_event_loop`.
    fn populate_export_defaults(&self, ui_ready: &ResultsWindow) {
        let rail_mode = ui_ready.global::<ResultsState>().get_rail_mode();
        let z_stack_max = ui_ready.global::<ResultsState>().get_z_stack_max().max(0) as u32;
        let t_stack_max = ui_ready.global::<ResultsState>().get_t_stack_max().max(0) as u32;

        let list_filter = self.list_filter.lock().expect("Poisened");
        let matrix_filter_guard = self.matrix_filter.lock().expect("Poisened");
        let classes = self.classes.lock().expect("Poisened");
        let images = self.images.lock().expect("Poisened");
        let available_columns = self.available_columns.lock().expect("Poisened");

        // `select_none_images`/`select_none_classes` store a sentinel (a
        // single entry that can't match any real image/class) rather than
        // a genuinely empty list, since empty means "no filter" (all) to
        // `update_list_view` — exporting is naturally "all" whenever
        // nothing meaningful was excluded, so the sentinel maps back to an
        // empty `ResultExport` list here rather than trying to preserve a
        // "selected literally nothing" state an export has no use for.
        let image_rel_paths: Vec<String> = if list_filter.image_rel_path == [PathBuf::new()] {
            Vec::new()
        } else {
            list_filter
                .image_rel_path
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect()
        };
        let object_classes: Vec<ObjectClass> = if list_filter.object_classes == [ObjectClass::Unset]
        {
            Vec::new()
        } else {
            list_filter.object_classes.clone()
        };
        let columns = if list_filter.columns.is_empty() {
            DEFAULT_LIST_COLUMNS.to_vec()
        } else {
            list_filter.columns.clone()
        };
        // Export always uses `ColorScale::Auto` (see `read_export_settings`)
        // - the dialog exposes no manual min/max override, so the current
        // Matrix view's own scale (which may be pinned) isn't mirrored here.
        let (aggregations, color_schema, grouping_regex, plate_dimension, well_size, square_size) =
            match matrix_filter_guard.as_ref() {
                Some(filter) => (
                    vec![filter.aggregation.clone()],
                    filter.color_schema.clone(),
                    filter.group_by_regex.clone(),
                    filter.plate_dimension,
                    filter.well_size,
                    filter.square_size,
                ),
                None => (
                    vec![Aggregation::default()],
                    ColorSchema::default(),
                    String::new(),
                    None,
                    None,
                    None,
                ),
            };
        let well_size = well_size.unwrap_or(WellSize { rows: 4, cols: 4 });
        let square_size = square_size.unwrap_or(DEFAULT_SQUARE_SIZE);

        let selected_images: std::collections::HashSet<&str> =
            image_rel_paths.iter().map(String::as_str).collect();
        let image_items_vec: Vec<MultiSelectItem> = images
            .iter()
            .map(|image| {
                let key = image.rel_path.to_str().unwrap_or_default();
                MultiSelectItem {
                    key: key.into(),
                    value: image.name.as_str().into(),
                    color: Color::default(),
                    group: "".into(),
                    selected: image_rel_paths.is_empty() || selected_images.contains(key),
                }
            })
            .collect();
        let image_summary = list_summary(
            if image_rel_paths.is_empty() {
                images.len()
            } else {
                image_rel_paths.len()
            },
            images.len(),
            "Images",
        );

        let class_items_vec: Vec<MultiSelectItem> = classes
            .iter()
            .map(|class| MultiSelectItem {
                key: class.name.as_str().into(),
                value: class.name.as_str().into(),
                color: bg_color_to_slint(class.color),
                group: "".into(),
                selected: object_classes.is_empty() || object_classes.contains(&class.id),
            })
            .collect();
        let class_summary = list_summary(
            if object_classes.is_empty() {
                classes.len()
            } else {
                object_classes.len()
            },
            classes.len(),
            "Classes",
        );

        let column_items_vec =
            column_items(&available_columns, &classes, |key| columns.contains(key));
        let column_groups_vec = column_groups(&available_columns);
        let column_summary = list_summary(columns.len(), available_columns.len(), "Columns");

        let aggregation_items_vec = aggregation_items(&aggregations);
        let aggregation_summary =
            list_summary(aggregations.len(), AGGREGATIONS.len(), "Aggregations");

        drop(available_columns);
        drop(images);
        drop(classes);
        drop(matrix_filter_guard);
        drop(list_filter);

        let state = ui_ready.global::<ExportDialogState>();
        state.set_output_dir("".into());
        state.set_z_start("0".into());
        state.set_z_end((z_stack_max + 1).to_string().into());
        state.set_t_start("0".into());
        state.set_t_end((t_stack_max + 1).to_string().into());

        state.set_image_items(ModelRc::from(Rc::new(VecModel::from(image_items_vec))));
        state.set_image_summary(image_summary);
        state.set_class_items(ModelRc::from(Rc::new(VecModel::from(class_items_vec))));
        state.set_class_summary(class_summary);
        state.set_column_items(ModelRc::from(Rc::new(VecModel::from(column_items_vec))));
        state.set_column_groups(ModelRc::from(Rc::new(VecModel::from(column_groups_vec))));
        state.set_column_summary(column_summary);

        state.set_grouping_regex(grouping_regex.into());
        state.set_aggregation_items(ModelRc::from(Rc::new(VecModel::from(
            aggregation_items_vec,
        ))));
        state.set_aggregation_summary(aggregation_summary);
        state.set_color_schema_items(ModelRc::from(Rc::new(VecModel::from(color_schema_items()))));
        state.set_color_schema_summary(
            color_schemas()
                .into_iter()
                .find(|(_, schema)| *schema == color_schema)
                .map(|(name, _)| name)
                .unwrap_or("Viridis")
                .into(),
        );
        state.set_plate_size_items(ModelRc::from(Rc::new(VecModel::from(plate_size_items(
            plate_dimension,
        )))));
        state.set_plate_size_summary(
            plate_dimensions()
                .into_iter()
                .find(|(_, dimension)| *dimension == plate_dimension)
                .map(|(name, _)| name)
                .unwrap_or("Auto")
                .into(),
        );
        state.set_well_rows(well_size.rows.to_string().into());
        state.set_well_cols(well_size.cols.to_string().into());
        state.set_square_size_items(ModelRc::from(Rc::new(VecModel::from(square_size_items(
            square_size,
        )))));
        state.set_square_size_summary(format!("{square_size} px").into());

        state.set_with_list_view(rail_mode == ResultsRailMode::List);
        state.set_with_list_coloc_details(false);
        state.set_with_list_one_file_per_image(false);
        state.set_with_grouped_by_image_list(false);
        state.set_with_plate_view(rail_mode == ResultsRailMode::Matrix);
        state.set_with_plates_and_wells_as_list(false);
        state.set_with_heatmap(false);

        state.set_is_exporting(false);
        state.set_done(false);
        state.set_has_error(false);
        state.set_error_message("".into());
        state.set_progress_message("".into());
        state.set_progress_current(0);
        state.set_progress_total(0);
    }

    // Reads `ExportDialogState`'s current values back into a `ResultExport`
    // at "Start Export" time — the single point where the dialog's Slint
    // state (the source of truth for everything a control lets the user
    // edit directly) gets turned into the typed request `start_export`
    // wants. `Err` for a setting that can't be parsed/resolved at all
    // (blank output folder, unrecognized dropdown key, non-numeric range
    // field) — shown in the dialog's error area rather than silently
    // falling back to a default the user didn't ask for.
    fn read_export_settings(&self, ui_ready: &ResultsWindow) -> Result<ResultExport, String> {
        let state = ui_ready.global::<ExportDialogState>();
        let classes = self.classes.lock().expect("Poisened");

        let output_dir = state.get_output_dir().to_string();
        if output_dir.is_empty() {
            return Err("Choose an output folder first.".to_string());
        }

        let parse_u32 = |label: &str, text: slint::SharedString| -> Result<u32, String> {
            text.trim()
                .parse::<u32>()
                .map_err(|_| format!("\"{text}\" isn't a valid {label}."))
        };
        let z_start = parse_u32("Z start", state.get_z_start())?;
        let z_end = parse_u32("Z end", state.get_z_end())?;
        let t_start = parse_u32("T start", state.get_t_start())?;
        let t_end = parse_u32("T end", state.get_t_end())?;
        let well_rows = parse_u32("well row count", state.get_well_rows())?.max(1) as usize;
        let well_cols = parse_u32("well column count", state.get_well_cols())?.max(1) as usize;

        let selected_keys = |items: ModelRc<MultiSelectItem>| -> Vec<slint::SharedString> {
            items
                .iter()
                .filter(|item| item.selected)
                .map(|item| item.key)
                .collect()
        };

        let image_rel_paths: Vec<String> = selected_keys(state.get_image_items())
            .into_iter()
            .map(|key| key.to_string())
            .collect();

        let mut object_classes = Vec::new();
        for key in selected_keys(state.get_class_items()) {
            let Some(class) = classes.iter().find(|class| class.name == key.as_str()) else {
                return Err(format!("Unknown class \"{key}\" selected for export."));
            };
            object_classes.push(class.id);
        }
        drop(classes);

        let available_columns = self.available_columns.lock().expect("Poisened");
        let class_list = self.classes.lock().expect("Poisened");
        let mut columns = Vec::new();
        for key in selected_keys(state.get_column_items()) {
            let Some(column) = Column::from_key(key.as_str(), &class_list) else {
                return Err(format!("Unknown column \"{key}\" selected for export."));
            };
            columns.push(column);
        }
        drop(class_list);
        drop(available_columns);

        let mut aggregations = Vec::new();
        for key in selected_keys(state.get_aggregation_items()) {
            let Some(aggregation) = aggregation_from_key(key.as_str()) else {
                return Err(format!(
                    "Unknown aggregation \"{key}\" selected for export."
                ));
            };
            aggregations.push(aggregation);
        }

        let color_schema_key = state
            .get_color_schema_items()
            .iter()
            .find(|item| item.selected)
            .map(|item| item.key)
            .unwrap_or_else(|| "Viridis".into());
        let Some(color_schema) = color_schema_from_key(color_schema_key.as_str()) else {
            return Err(format!("Unknown color schema \"{color_schema_key}\"."));
        };

        let plate_size_key = state
            .get_plate_size_items()
            .iter()
            .find(|item| item.selected)
            .map(|item| item.key)
            .unwrap_or_else(|| "Auto".into());
        let Some(plate_dimension) = plate_dimension_from_key(plate_size_key.as_str()) else {
            return Err(format!("Unknown plate size \"{plate_size_key}\"."));
        };

        let square_size_key = state
            .get_square_size_items()
            .iter()
            .find(|item| item.selected)
            .map(|item| item.key)
            .unwrap_or_else(|| DEFAULT_SQUARE_SIZE.to_string().into());
        let square_size = square_size_key
            .parse::<usize>()
            .map_err(|_| format!("Unknown square size \"{square_size_key}\"."))?;

        Ok(ResultExport {
            output_dir: PathBuf::from(output_dir),
            format: ExportFormat::XLSX,
            z_stacks: std::range::Range {
                start: z_start,
                end: z_end.max(z_start + 1),
            },
            t_stacks: std::range::Range {
                start: t_start,
                end: t_end.max(t_start + 1),
            },
            image_rel_paths,
            columns,
            color_schema,
            color_scale: ColorScale::Auto,
            grouping_regex: state.get_grouping_regex().to_string(),
            object_classes,
            aggregations,
            plate_dimension,
            well_size: Some(WellSize {
                rows: well_rows,
                cols: well_cols,
            }),
            well_order: None,
            square_size: Some(square_size),
            with_list_view: state.get_with_list_view(),
            with_list_coloc_details: state.get_with_list_coloc_details(),
            with_list_one_file_per_image: state.get_with_list_one_file_per_image(),
            with_grouped_by_image_list: state.get_with_grouped_by_image_list(),
            with_plate_view: state.get_with_plate_view(),
            with_plates_and_wells_as_list: state.get_with_plates_and_wells_as_list(),
            with_heatmap: state.get_with_heatmap(),
        })
    }

    // Runs `export` on a background thread against a *fresh* database
    // connection (see `db_path`'s doc comment for why not
    // `result_generator`), reporting progress/completion/error back to
    // `ExportDialogState` as it goes.
    fn run_export(self: &Arc<Self>, export: ResultExport) {
        let Some(path) = self.db_path.lock().expect("Poisened").clone() else {
            self.push_export_error("No database is open.".to_string());
            return;
        };
        let manager = self.clone();
        std::thread::spawn(move || {
            let database = match ResultsGenerator::open_database(path) {
                Ok(database) => database,
                Err(err) => {
                    manager.push_export_error(format!("Could not open database: {err}"));
                    return;
                }
            };
            let manager_for_progress = manager.clone();
            let result = export.start_export(&database, &mut |message, current, total| {
                manager_for_progress.push_export_progress(message.to_string(), current, total);
            });
            match result {
                Ok(()) => manager.push_export_done(),
                Err(err) => manager.push_export_error(err.to_string()),
            }
        });
    }

    fn push_export_progress(&self, message: String, current: usize, total: usize) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ExportDialogState>();
                state.set_progress_message(message.into());
                state.set_progress_current(current as i32);
                state.set_progress_total(total as i32);
            } else {
                warn!("Failed to upgrade UI handle, cannot update export progress!");
            }
        })
        .ok();
    }

    fn push_export_done(&self) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ExportDialogState>();
                state.set_is_exporting(false);
                state.set_done(true);
            } else {
                warn!("Failed to upgrade UI handle, cannot mark export as done!");
            }
        })
        .ok();
    }

    fn push_export_error(&self, message: String) {
        let ui_weak = self.ui.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ExportDialogState>();
                state.set_is_exporting(false);
                state.set_has_error(true);
                state.set_error_message(message.into());
            } else {
                warn!("Failed to upgrade UI handle, cannot report export error!");
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

// The IMAGES dropdown's own summary: unlike columns/classes, an empty
// selection here means "no image filter at all" (every image matches — see
// `update_list_view`'s `image_rel_path.is_empty()` check and
// `select_none_images`'s sentinel comment), not "match nothing". "0 of N
// Images" reads as the opposite of that, so it's shown as "All Images"
// instead — including right after a database opens, since `set_images_in_slint`
// starts with nothing selected.
fn image_summary_text(selected: usize, total: usize) -> slint::SharedString {
    if selected == 0 {
        "All Images".into()
    } else {
        list_summary(selected, total, "Images")
    }
}

// Chart axis-label formatting: whole numbers stay whole (most of these
// values, e.g. area/perimeter, are naturally integer-ish), everything else
// gets 2 decimal places rather than however many the raw f64 has.
fn format_chart_value(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.2}")
    }
}

// "All classes" (selected by default) followed by one entry per class from
// the open database's classification settings.
fn class_filter_items(object_classes: &[Class], selected: bool) -> Vec<MultiSelectItem> {
    object_classes
        .iter()
        .map(|class| MultiSelectItem {
            key: class.name.as_str().into(),
            value: class.name.as_str().into(),
            color: bg_color_to_slint(class.color),
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
    classes: &[Class],
    selected: impl Fn(&Column) -> bool,
) -> Vec<MultiSelectItem> {
    columns
        .iter()
        .map(|column| MultiSelectItem {
            key: column.key.as_key(classes).into(),
            value: column.display_name.as_str().into(),
            color: Color::default(),
            group: column.group.as_str().into(),
            selected: selected(&column.key),
        })
        .collect()
}

/// Whether `column` can be aggregated across an image's objects at all —
/// mirrors `is_aggregable` in results_exporter.rs (kept in sync with it by
/// inspecting the same variants), needed here because Images-mode reuses the
/// List view's own COLUMNS selection rather than a dedicated picker.
fn is_aggregable_column(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId
            | Column::ImageName
            | Column::ObjectClass
            | Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

/// Whether `column` is something `chart_value_expr`/`column_aggregate_expr`
/// (results_generator.rs) can resolve to a single per-object SQL
/// expression — the exact same set that function accepts, kept in sync by
/// inspecting the same variants. Unlike `is_aggregable_column`, `Count` is
/// also excluded here: it's a row tally (1 per object), not a per-object
/// measurement, so charting it doesn't mean anything the way it does for
/// Images-mode's own per-image counts.
fn is_chartable_column(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId
            | Column::ImageName
            | Column::ObjectClass
            | Column::Count
            | Column::IntensityAvg(_)
            | Column::IntensitySum(_)
            | Column::IntensityMin(_)
            | Column::IntensityMax(_)
    )
}

/// Whether `column` is a valid Matrix (Plate/Well/Heatmap) target — mirrors
/// `set_columns_in_slint`'s own `matrix_items` filter (an object's identity
/// isn't a value to aggregate; unlike `is_chartable_column`, `Count` stays
/// valid here — "how many objects in this well" is meaningful for a grid,
/// unlike for a histogram/scatter/boxplot).
fn is_matrix_column(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId | Column::ImageName | Column::ObjectClass
    )
}

/// The Charts toolbar's PROPERTY/X/Y column pickers — single-select, only
/// ever offering columns `is_chartable_column` accepts.
fn chart_column_items(
    columns: &[ColumnEntry],
    classes: &[Class],
    selected: &Column,
) -> Vec<MultiSelectItem> {
    let chartable: Vec<ColumnEntry> = columns
        .iter()
        .filter(|entry| is_chartable_column(&entry.key))
        .cloned()
        .collect();
    column_items(&chartable, classes, |key| key == selected)
}

// The GROUP BY dropdown's two fixed options, Objects always selected — used
// both to seed it on a fresh database and to reset it whenever `list_filter`
// itself is reset back to `ListGroupBy::Objects`.
fn group_by_items(selected_images: bool) -> Vec<MultiSelectItem> {
    vec![
        MultiSelectItem {
            key: "objects".into(),
            value: "Objects".into(),
            color: Color::default(),
            group: "".into(),
            selected: !selected_images,
        },
        MultiSelectItem {
            key: "images".into(),
            value: "Images".into(),
            color: Color::default(),
            group: "".into(),
            selected: selected_images,
        },
    ]
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
        Aggregation::Median => "Median",
        Aggregation::Skewness => "Skewness",
    }
}

const AGGREGATIONS: [Aggregation; 7] = [
    Aggregation::Avg,
    Aggregation::Min,
    Aggregation::Max,
    Aggregation::Stddev,
    Aggregation::Sum,
    Aggregation::Median,
    Aggregation::Skewness,
];

// The Export dialog's AGGREGATIONS picker is multi-select (unlike the
// Matrix toolbar's single-select one) — `selected` is a whole `Vec` to
// check each fixed option against, not one currently-chosen value.
fn aggregation_items(selected: &[Aggregation]) -> Vec<MultiSelectItem> {
    AGGREGATIONS
        .iter()
        .map(|aggregation| {
            let name = aggregation_display_name(aggregation);
            MultiSelectItem {
                key: name.into(),
                value: name.into(),
                color: Color::default(),
                group: "".into(),
                selected: selected.contains(aggregation),
            }
        })
        .collect()
}

fn aggregation_from_key(key: &str) -> Option<Aggregation> {
    AGGREGATIONS
        .iter()
        .find(|aggregation| aggregation_display_name(aggregation) == key)
        .cloned()
}

fn color_schemas() -> [(&'static str, ColorSchema); 11] {
    [
        ("Excel", ColorSchema::Excel),
        ("Viridis", ColorSchema::Viridis),
        ("Plasma", ColorSchema::Plasma),
        ("Inferno", ColorSchema::Inferno),
        ("Cividis", ColorSchema::Cividis),
        ("Coolwarm", ColorSchema::Coolwarm),
        ("Red-Blue", ColorSchema::RedBlue),
        ("YlGnBu", ColorSchema::YlGnBu),
        ("Haline", ColorSchema::Haline),
        ("Algae", ColorSchema::Algae),
        ("Thermal", ColorSchema::Thermal),
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

// PLATE SIZE dropdown vocabulary: "Auto" (`None`, `get_group_by_plate`
// auto-selects the smallest standard format that fits the data — see
// `best_matching_dimensions`) followed by every standard microplate well
// count `PlateDimensions` supports, smallest first.
fn plate_dimensions() -> [(&'static str, Option<PlateDimensions>); 8] {
    [
        ("Auto", None),
        ("6-well (2x3)", Some(PlateDimensions::PLate2x3)),
        ("12-well (3x4)", Some(PlateDimensions::Plate3x4)),
        ("24-well (4x6)", Some(PlateDimensions::Plate4x6)),
        ("48-well (6x8)", Some(PlateDimensions::Plate6x8)),
        ("96-well (8x12)", Some(PlateDimensions::Plate8x12)),
        ("384-well (16x24)", Some(PlateDimensions::Plate16x24)),
        ("1536-well (32x48)", Some(PlateDimensions::Plate32x48)),
    ]
}

fn plate_size_items(selected: Option<PlateDimensions>) -> Vec<MultiSelectItem> {
    plate_dimensions()
        .into_iter()
        .map(|(name, dimension)| MultiSelectItem {
            key: name.into(),
            value: name.into(),
            color: Color::default(),
            group: "".into(),
            selected: dimension == selected,
        })
        .collect()
}

// Returns `Some(dimension)` for a recognized key — note this is
// `Option<Option<PlateDimensions>>`: the outer `Option` is "was `key`
// recognized at all", the inner one is the dropdown item's own meaning
// ("Auto" -> `None`, an explicit size -> `Some(_)`).
fn plate_dimension_from_key(key: &str) -> Option<Option<PlateDimensions>> {
    plate_dimensions()
        .into_iter()
        .find(|(name, _)| *name == key)
        .map(|(_, dimension)| dimension)
}

// SQUARE SIZE dropdown vocabulary for the image heatmap — matches
// `ImageHeatmapFilter::square_size`'s own default (see `get_image_heatmap`).
const SQUARE_SIZES: [usize; 6] = [36, 48, 64, 128, 256, 1024];
const DEFAULT_SQUARE_SIZE: usize = 256;

/// Inverse of `get_image_heatmap`'s tile key format (`format!("R{row}C{col}")`
/// in results_generator.rs) — used by `on_image_heatmap_cell_clicked` to
/// recover which tile was clicked so its pixel bounds can be computed.
fn parse_tile_key(key: &str) -> Option<(u32, u32)> {
    let rest = key.strip_prefix('R')?;
    let (row, col) = rest.split_once('C')?;
    Some((row.parse().ok()?, col.parse().ok()?))
}

fn square_size_items(selected: usize) -> Vec<MultiSelectItem> {
    SQUARE_SIZES
        .iter()
        .map(|size| MultiSelectItem {
            key: size.to_string().into(),
            value: format!("{size} px").into(),
            color: Color::default(),
            group: "".into(),
            selected: *size == selected,
        })
        .collect()
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
        CellValue::Empty => "NaN".into(),
        CellValue::String(value) => value.as_str().into(),
        CellValue::Float(value) => format!("{value:.3}").into(),
        CellValue::Integer(value) => value.to_string().into(),
        CellValue::Class((name, _color)) => name.as_str().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppWindow;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::object_list_controller::ObjectListController;
    use crate::editor::test_support::{test_ui_state, test_ui_windows};
    use crate::editor::viewport_controller::ViewportController;

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
        results_ui: slint::Weak<ResultsWindow>,
    ) -> Arc<ResultsStateController> {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), ui_state.clone()));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let histogram_controller = Arc::new(HistogramController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_meta_controller = Arc::new(ImageMetaController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_list_controller = Arc::new(ImagesListController::new(
            ui,
            ui_state.clone(),
            viewport_controller,
            histogram_controller,
            image_meta_controller,
            object_list_controller,
        ));
        Arc::new(ResultsStateController::new(
            results_ui,
            ui_state,
            image_list_controller,
        ))
    }

    // Regression test for the export dialog's very first hurdle: for a long
    // time this failed to even *compile* (`ExportDialogState` had no public
    // Rust binding at all - it wasn't re-exported from main.slint's own
    // top-level `export { ... }` block, the same way every other dialog
    // global there is, see main.slint). Exercises the actual callback
    // wiring end to end against a real (headless) `ResultsWindow`, with no
    // database open - the state every dialog control's defaults must still
    // fall back to sanely (an empty images/classes list, no matrix filter
    // set yet).
    #[test]
    fn attach_callbacks_export_dialog_open_populates_defaults_and_activates() {
        let (ui, results_ui) = test_ui_windows();
        let controller = make_controller_with_ui(ui.as_weak(), results_ui.as_weak());
        controller.attach_callbacks();

        assert!(!results_ui.global::<ExportDialogState>().get_active());

        results_ui
            .global::<ResultsState>()
            .invoke_export_dialog_open();

        let state = results_ui.global::<ExportDialogState>();
        assert!(state.get_active(), "dialog should activate on open");
        // No matrix view has been rendered yet (`matrix_filter` starts
        // `None`) and the rail starts on List - see `populate_export_defaults`.
        assert!(state.get_with_list_view());
        assert!(!state.get_with_plate_view());
        assert!(!state.get_with_heatmap());
        // z/t default to the database's full range, exclusive end - starting
        // at 0, and past whatever `ResultsState.z-stack-max`/`t-stack-max`
        // are currently set to (their own default, before any real database
        // has been opened, is a temporary design-preview placeholder - see
        // that property's doc comment in results_state.slint - so this only
        // checks the *shape* of the range, not its exact placeholder bound).
        assert_eq!(state.get_z_start(), "0");
        assert_eq!(state.get_t_start(), "0");
        let z_end: i32 = state.get_z_end().parse().expect("z_end should be numeric");
        let t_end: i32 = state.get_t_end().parse().expect("t_end should be numeric");
        assert!(z_end >= 1);
        assert!(t_end >= 1);
        // No images/classes/columns registered without an open database.
        assert_eq!(state.get_image_items().row_count(), 0);
        assert_eq!(state.get_class_items().row_count(), 0);
    }

    // Re-opening the dialog after the user has already changed something
    // (here: unchecking "List view") must not clobber it - only the *first*
    // open after a database loads re-seeds the defaults (see
    // `export_populated`).
    #[test]
    fn attach_callbacks_export_dialog_reopen_keeps_user_changes() {
        let (ui, results_ui) = test_ui_windows();
        let controller = make_controller_with_ui(ui.as_weak(), results_ui.as_weak());
        controller.attach_callbacks();

        results_ui
            .global::<ResultsState>()
            .invoke_export_dialog_open();
        results_ui
            .global::<ExportDialogState>()
            .set_with_list_view(false);
        results_ui.global::<ExportDialogState>().set_active(false);

        results_ui
            .global::<ResultsState>()
            .invoke_export_dialog_open();

        let state = results_ui.global::<ExportDialogState>();
        assert!(state.get_active());
        assert!(
            !state.get_with_list_view(),
            "re-opening must not reset a setting the user already changed"
        );
    }

    // `on_start_clicked` with no output folder chosen must surface a
    // validation error rather than silently doing nothing or panicking
    // (there's no database open in this test either, so this also exercises
    // that `read_export_settings` fails fast before ever touching
    // `db_path`).
    #[test]
    fn attach_callbacks_export_start_without_output_dir_reports_an_error() {
        let (ui, results_ui) = test_ui_windows();
        let controller = make_controller_with_ui(ui.as_weak(), results_ui.as_weak());
        controller.attach_callbacks();

        results_ui
            .global::<ResultsState>()
            .invoke_export_dialog_open();
        results_ui
            .global::<ExportDialogState>()
            .invoke_start_clicked();

        let state = results_ui.global::<ExportDialogState>();
        assert!(state.get_has_error());
        assert!(!state.get_is_exporting());
        assert!(!state.get_error_message().is_empty());
    }
}
