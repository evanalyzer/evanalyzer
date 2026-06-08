use crate::editor::images_list_controller::ImagesListController;
use crate::{
    FilterItem, ResultsColumnDef, ResultsGroupBy, ResultsListState, ResultsRow, ResultsState,
    ResultsWindow, UiState,
};
use evanalyzer_app::result::{
    aggregate_rows, build_column_specs, discover_channels, to_display_row, AggFunc, ColumnSpec,
    DatabaseFilter, GroupBy, GroupConfig, ResultsExporter, ResultsLoader, RoiRow,
};
use log::warn;
use slint::{ComponentHandle, Model, SharedString};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const PAGE_SIZE: usize = 500;

pub struct ResultsTableController {
    pub(crate) ui: slint::Weak<ResultsWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) image_list_controller: Arc<ImagesListController>,
    pub(crate) path: Arc<Mutex<Option<PathBuf>>>,
    /// The per-ROI rows currently shown in the table, in display order. Indexed
    /// by `ResultsRow.roi_id - 1` to map a selected row back to its source ROI
    /// (image + bounding box). Empty while a grouped/aggregated view is active.
    pub(crate) displayed_rois: Arc<Mutex<Vec<RoiRow>>>,
    pub(crate) channels: Arc<Mutex<Vec<i32>>>,
    pub(crate) column_specs: Arc<Mutex<Vec<ColumnSpec>>>,
    pub(crate) current_page: Arc<Mutex<usize>>,
    pub(crate) all_loaded: Arc<Mutex<bool>>,
    pub(crate) image_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) class_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) group_config: Arc<Mutex<GroupConfig>>,
    pub(crate) image_search: Mutex<String>,
    pub(crate) class_search: Mutex<String>,
    pub(crate) column_search: Mutex<String>,
}

impl ResultsTableController {
    pub fn new(
        ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
        image_list_controller: Arc<ImagesListController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            image_list_controller,
            path: Arc::new(Mutex::new(None)),
            displayed_rois: Arc::new(Mutex::new(Vec::new())),
            channels: Arc::new(Mutex::new(Vec::new())),
            column_specs: Arc::new(Mutex::new(Vec::new())),
            current_page: Arc::new(Mutex::new(0)),
            all_loaded: Arc::new(Mutex::new(false)),
            image_filter: Arc::new(Mutex::new(None)),
            class_filter: Arc::new(Mutex::new(None)),
            group_config: Arc::new(Mutex::new(GroupConfig::default())),
            image_search: Mutex::new(String::new()),
            class_search: Mutex::new(String::new()),
            column_search: Mutex::new(String::new()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();

        macro_rules! cb {
            ($method:ident) => {{
                let this = Arc::clone(self);
                move || this.$method()
            }};
            ($method:ident, $arg:ty) => {{
                let this = Arc::clone(self);
                move |v: $arg| this.$method(v)
            }};
            ($method:ident, $arg1:ty, $arg2:ty) => {{
                let this = Arc::clone(self);
                move |a: $arg1, b: $arg2| this.$method(a, b)
            }};
        }

        state.on_image_filter_label_toggled(cb!(toggle_image_label, SharedString));
        state.on_image_filter_search_changed(cb!(image_search_changed, SharedString));
        state.on_image_select_all(cb!(image_select_all));
        state.on_image_clear_all(cb!(image_clear_all));

        state.on_class_filter_label_toggled(cb!(toggle_class_label, SharedString));
        state.on_class_filter_search_changed(cb!(class_search_changed, SharedString));
        state.on_class_select_all(cb!(class_select_all));
        state.on_class_clear_all(cb!(class_clear_all));

        state.on_column_label_toggled(cb!(toggle_column_label, SharedString));
        state.on_column_search_changed(cb!(column_search_changed, SharedString));
        state.on_column_select_all(cb!(column_select_all));
        state.on_column_clear_all(cb!(column_clear_all));
        state.on_column_filter_apply(cb!(column_filter_apply));

        state.on_sort_requested(cb!(on_sort_column_changed, SharedString, bool));

        state.on_roi_row_selected(cb!(on_roi_row_selected, i32));

        // --- group_apply: read group selection, reload (grouped or paginated) -
        {
            let this = Arc::clone(self);
            state.on_group_apply(move || {
                let Some(window) = this.ui.upgrade() else { return };
                let state = window.global::<ResultsState>();

                let config = GroupConfig {
                    group_by: map_group_by(state.get_group_by()),
                    regex: state.get_group_regex().to_string(),
                    aggs: selected_aggs(&state),
                };
                *this.group_config.lock().unwrap() = config;
                *this.current_page.lock().unwrap() = 0;
                *this.all_loaded.lock().unwrap() = false;

                state.set_loading_more(true);
                Self::spawn_reload(Arc::clone(&this));
            });
        }

        // --- filter_apply: read UI state on main thread, spawn DB reload ------
        {
            let this = Arc::clone(self);
            state.on_filter_apply(move || {
                let window = match this.ui.upgrade() {
                    Some(w) => w,
                    None => return,
                };
                let state = window.global::<ResultsState>();

                let img_model = state.get_filter_image_items();
                let cls_model = state.get_filter_class_items();
                let total_img = img_model.row_count();
                let total_cls = cls_model.row_count();

                let checked_img: Vec<String> = (0..total_img)
                    .filter_map(|i| {
                        img_model
                            .row_data(i)?
                            .checked
                            .then_some(img_model.row_data(i)?.label.to_string())
                    })
                    .collect();
                let checked_cls: Vec<String> = (0..total_cls)
                    .filter_map(|i| {
                        cls_model
                            .row_data(i)?
                            .checked
                            .then_some(cls_model.row_data(i)?.label.to_string())
                    })
                    .collect();

                let image_filter: Option<Vec<String>> =
                    (checked_img.len() < total_img).then_some(checked_img);
                let class_filter: Option<Vec<String>> =
                    (checked_cls.len() < total_cls).then_some(checked_cls);

                let is_filtered = image_filter.is_some() || class_filter.is_some();
                *this.image_filter.lock().unwrap() = image_filter;
                *this.class_filter.lock().unwrap() = class_filter;
                *this.current_page.lock().unwrap() = 0;

                state.set_filter_active(is_filtered);
                state.set_loading_more(true);

                Self::spawn_reload(Arc::clone(&this));
            });
        }

        // --- clear_all_filters ------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_clear_all_filters(move || {
                let window = match this.ui.upgrade() {
                    Some(w) => w,
                    None => return,
                };
                let state = window.global::<ResultsState>();

                *this.image_search.lock().unwrap() = String::new();
                *this.class_search.lock().unwrap() = String::new();

                let img = set_all_checked(&model_to_vec(&state.get_filter_image_items()), true);
                state.set_filter_image_active(false);
                state.set_filter_image_all_popup_checked(true);
                state.set_filter_image_items(to_model(img.clone()));
                state.set_filter_image_popup(to_model(img));

                let cls = set_all_checked(&model_to_vec(&state.get_filter_class_items()), true);
                state.set_filter_class_active(false);
                state.set_filter_class_all_popup_checked(true);
                state.set_filter_class_items(to_model(cls.clone()));
                state.set_filter_class_popup(to_model(cls));

                state.set_filter_active(false);
                state.set_loading_more(true);

                *this.image_filter.lock().unwrap() = None;
                *this.class_filter.lock().unwrap() = None;
                *this.current_page.lock().unwrap() = 0;

                Self::spawn_reload(Arc::clone(&this));
            });
        }

        // --- load_more_rows ---------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_load_more_rows(move || {
                if *this.all_loaded.lock().unwrap() {
                    return;
                }
                let arc = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_load_more(arc));
            });
        }

        // --- copy_to_clipboard ------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_copy_to_clipboard(move || {
                let Some(window) = this.ui.upgrade() else { return };
                let state = window.global::<ResultsState>();

                let cols: Vec<_> = (0..state.get_columns().row_count())
                    .filter_map(|i| state.get_columns().row_data(i))
                    .filter(|c| c.visible)
                    .collect();

                let rows_model = state.get_rows();
                let row_count = rows_model.row_count();

                let mut tsv = cols
                    .iter()
                    .map(|c| c.label.to_string())
                    .collect::<Vec<_>>()
                    .join("\t");
                tsv.push('\n');

                let specs = this.column_specs.lock().unwrap().clone();
                let visible_indices: Vec<usize> = specs
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.visible)
                    .map(|(i, _)| i)
                    .collect();

                for r in 0..row_count {
                    if let Some(row) = rows_model.row_data(r) {
                        let values: Vec<String> = visible_indices
                            .iter()
                            .filter_map(|&i| row.values.row_data(i).map(|v| v.to_string()))
                            .collect();
                        tsv.push_str(&values.join("\t"));
                        tsv.push('\n');
                    }
                }

                use copypasta::{ClipboardContext, ClipboardProvider};
                if let Ok(mut ctx) = ClipboardContext::new() {
                    let _ = ctx.set_contents(tsv);
                }
            });
        }

        // --- export_csv -------------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_csv(move || {
                let Some(path) = this.path.lock().unwrap().clone() else { return };
                let image_filter = this.image_filter.lock().unwrap().clone();
                let class_filter = this.class_filter.lock().unwrap().clone();
                let group = this.group_config.lock().unwrap().clone();
                let base_specs = this.column_specs.lock().unwrap().clone();

                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .set_file_name("results.csv")
                    .save_file()
                else {
                    return;
                };

                std::thread::spawn(move || {
                    let loader = Arc::new(ResultsLoader::new(&path));
                    let exporter = ResultsExporter::new(loader);
                    let filter = DatabaseFilter {
                        image_filter,
                        class_filter,
                        ..Default::default()
                    };
                    if let Err(e) =
                        exporter.export_to_csv(filter, &group, &base_specs, &export_path)
                    {
                        warn!("CSV export failed: {:?}", e);
                    }
                });
            });
        }

        // --- export_xlsx ------------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_xlsx(move || {
                let Some(path) = this.path.lock().unwrap().clone() else { return };
                let image_filter = this.image_filter.lock().unwrap().clone();
                let class_filter = this.class_filter.lock().unwrap().clone();
                let group = this.group_config.lock().unwrap().clone();
                let base_specs = this.column_specs.lock().unwrap().clone();

                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("Excel", &["xlsx"])
                    .set_file_name("results.xlsx")
                    .save_file()
                else {
                    return;
                };

                std::thread::spawn(move || {
                    let loader = Arc::new(ResultsLoader::new(&path));
                    let exporter = ResultsExporter::new(loader);
                    let filter = DatabaseFilter {
                        image_filter,
                        class_filter,
                        ..Default::default()
                    };
                    if let Err(e) =
                        exporter.export_to_xlsx(filter, &group, &base_specs, &export_path)
                    {
                        warn!("XLSX export failed: {:?}", e);
                    }
                });
            });
        }
    }

    // -------------------------------------------------------------------------
    // File loading
    // -------------------------------------------------------------------------

    pub fn load_from_file(self: &Arc<Self>, path: PathBuf) {
        if let Some(app_ui) = self.app_state.ui_handle.upgrade() {
            app_ui.global::<ResultsListState>().set_is_loading(true);
        }

        *self.path.lock().unwrap() = Some(path.clone());
        *self.current_page.lock().unwrap() = 0;
        *self.all_loaded.lock().unwrap() = false;
        *self.image_filter.lock().unwrap() = None;
        *self.class_filter.lock().unwrap() = None;
        *self.group_config.lock().unwrap() = GroupConfig::default();
        *self.image_search.lock().unwrap() = String::new();
        *self.class_search.lock().unwrap() = String::new();
        self.displayed_rois.lock().unwrap().clear();

        let ui = self.ui.clone();
        let app_ui = self.app_state.ui_handle.clone();
        let channels_arc = Arc::clone(&self.channels);
        let all_loaded_arc = Arc::clone(&self.all_loaded);
        let column_specs_arc = Arc::clone(&self.column_specs);
        let displayed_rois_arc = Arc::clone(&self.displayed_rois);

        std::thread::spawn(move || {
            let loader = ResultsLoader::new(&path);

            let first_page = loader.get_rois(DatabaseFilter {
                page_size: PAGE_SIZE,
                ..Default::default()
            });
            let img_names = loader.get_image_names();
            let cls_names = loader.get_class_names();

            match (first_page, img_names, cls_names) {
                (Ok(rois), Ok(img_names), Ok(cls_names)) => {
                    let channels = discover_channels(&rois);
                    let specs = build_column_specs(&channels);
                    let all_loaded = rois.len() < PAGE_SIZE;

                    *channels_arc.lock().unwrap() = channels;
                    *all_loaded_arc.lock().unwrap() = all_loaded;
                    *column_specs_arc.lock().unwrap() = specs.clone();
                    *displayed_rois_arc.lock().unwrap() = rois.clone();

                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app_ui) = app_ui.upgrade() {
                            app_ui.global::<ResultsListState>().set_is_loading(false);
                        }
                        if let Some(window) = ui.upgrade() {
                            let state = window.global::<ResultsState>();

                            let slint_rows: Vec<ResultsRow> = rois
                                .iter()
                                .enumerate()
                                .map(|(i, r)| to_slint_row(to_display_row(i, r, &specs)))
                                .collect();

                            let slint_cols: Vec<ResultsColumnDef> =
                                specs_to_slint_cols(&specs);
                            let visible_count =
                                specs.iter().filter(|c| c.visible).count() as i32;

                            let column_items: Vec<FilterItem> = specs
                                .iter()
                                .map(|c| FilterItem {
                                    label: c.label.as_str().into(),
                                    checked: c.visible,
                                })
                                .collect();

                            state.set_columns(slint::ModelRc::new(
                                slint::VecModel::from(slint_cols),
                            ));
                            state.set_visible_column_count(visible_count);
                            state.set_column_items(slint::ModelRc::new(
                                slint::VecModel::from(column_items.clone()),
                            ));
                            state.set_column_popup(slint::ModelRc::new(
                                slint::VecModel::from(column_items),
                            ));
                            state.set_column_popup_all_checked(true);

                            let image_items = names_to_filter_items(&img_names);
                            state.set_filter_image_items(slint::ModelRc::new(
                                slint::VecModel::from(image_items.clone()),
                            ));
                            state.set_filter_image_popup(slint::ModelRc::new(
                                slint::VecModel::from(image_items),
                            ));
                            state.set_filter_image_active(false);
                            state.set_filter_image_all_popup_checked(true);

                            let class_items = names_to_filter_items(&cls_names);
                            state.set_filter_class_items(slint::ModelRc::new(
                                slint::VecModel::from(class_items.clone()),
                            ));
                            state.set_filter_class_popup(slint::ModelRc::new(
                                slint::VecModel::from(class_items),
                            ));
                            state.set_filter_class_active(false);
                            state.set_filter_class_all_popup_checked(true);

                            state.set_filter_active(false);
                            state.set_group_active(false);
                            state.set_group_by(ResultsGroupBy::None);
                            state.set_group_regex(slint::SharedString::new());
                            state.set_all_rows_loaded(all_loaded);
                            state.set_loading_more(false);
                            state.set_rows(slint::ModelRc::new(slint::VecModel::from(
                                slint_rows,
                            )));
                            let _ = window.show();
                        }
                    });
                }
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                    warn!("Failed to load results: {:?}", e);
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app_ui) = app_ui.upgrade() {
                            app_ui.global::<ResultsListState>().set_is_loading(false);
                        }
                    });
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // Reload dispatch: grouped (aggregated) vs. paginated per-ROI view
    // -------------------------------------------------------------------------

    /// Spawns the appropriate background reload based on the active grouping.
    fn spawn_reload(this: Arc<Self>) {
        let grouped = !matches!(
            this.group_config.lock().unwrap().group_by,
            GroupBy::None
        );
        std::thread::spawn(move || {
            if grouped {
                Self::bg_reload_grouped(this);
            } else {
                Self::bg_reload_page0(this);
            }
        });
    }

    // -------------------------------------------------------------------------
    // Background: grouped/aggregated reload (one summary row per group)
    // -------------------------------------------------------------------------

    fn bg_reload_grouped(this: Arc<Self>) {
        let ui = this.ui.clone();
        let finish_loading = move |ui: slint::Weak<ResultsWindow>| {
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = ui.upgrade() {
                    w.global::<ResultsState>().set_loading_more(false);
                }
            });
        };

        let Some(path) = this.path.lock().unwrap().clone() else {
            finish_loading(ui);
            return;
        };

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let config = this.group_config.lock().unwrap().clone();
        // Per-ROI specs carry the column-visibility selection; only visible
        // metrics become grouped columns.
        let base_specs = this.column_specs.lock().unwrap().clone();

        let loader = ResultsLoader::new(&path);
        // Aggregation needs every matching row, so fetch all (page_size 0).
        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
        }) {
            Ok(rois) => {
                let (specs, display_rows) = aggregate_rows(&rois, &config, &base_specs);
                // Grouped view is never paginated.
                *this.all_loaded.lock().unwrap() = true;
                // Grouped rows aggregate many ROIs, so there is no single source
                // ROI to open/highlight when one is selected.
                this.displayed_rois.lock().unwrap().clear();

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let visible_count = specs.len() as i32;
                        let slint_rows: Vec<ResultsRow> =
                            display_rows.into_iter().map(to_slint_row).collect();

                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(
                            specs_to_slint_cols(&specs),
                        )));
                        state.set_visible_column_count(visible_count);
                        state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
                        state.set_all_rows_loaded(true);
                        state.set_loading_more(false);
                        state.set_group_active(true);
                    }
                });
            }
            Err(e) => {
                warn!("bg_reload_grouped failed: {:?}", e);
                finish_loading(ui);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Background: reload page 0 with current filters
    // -------------------------------------------------------------------------

    fn bg_reload_page0(this: Arc<Self>) {
        let Some(path) = this.path.lock().unwrap().clone() else {
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = this.ui.upgrade() {
                    w.global::<ResultsState>().set_loading_more(false);
                }
            });
            return;
        };

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let specs = this.column_specs.lock().unwrap().clone();
        let needs_intensities = specs.iter().any(|c| c.visible && c.id.starts_with("ch"));
        let ui = this.ui.clone();

        let loader = ResultsLoader::new(&path);
        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            page_size: PAGE_SIZE,
            page: 0,
            needs_intensities,
        }) {
            Ok(rois) => {
                let all_loaded = rois.len() < PAGE_SIZE;
                *this.all_loaded.lock().unwrap() = all_loaded;
                *this.displayed_rois.lock().unwrap() = rois.clone();

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let slint_rows: Vec<ResultsRow> = rois
                            .iter()
                            .enumerate()
                            .map(|(i, r)| to_slint_row(to_display_row(i, r, &specs)))
                            .collect();
                        let state = window.global::<ResultsState>();
                        // Restore the per-ROI columns (grouped mode may have replaced them).
                        let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(
                            specs_to_slint_cols(&specs),
                        )));
                        state.set_visible_column_count(visible_count);
                        state.set_group_active(false);
                        state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
                        state.set_all_rows_loaded(all_loaded);
                        state.set_loading_more(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_reload_page0 failed: {:?}", e);
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(w) = ui.upgrade() {
                        w.global::<ResultsState>().set_loading_more(false);
                    }
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // Background: append next page
    // -------------------------------------------------------------------------

    fn bg_load_more(this: Arc<Self>) {
        let Some(path) = this.path.lock().unwrap().clone() else {
            return;
        };

        let next_page = {
            let mut p = this.current_page.lock().unwrap();
            *p += 1;
            *p
        };

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let specs = this.column_specs.lock().unwrap().clone();
        let needs_intensities = specs.iter().any(|c| c.visible && c.id.starts_with("ch"));
        let ui = this.ui.clone();

        let _ = slint::invoke_from_event_loop({
            let ui = ui.clone();
            move || {
                if let Some(w) = ui.upgrade() {
                    w.global::<ResultsState>().set_loading_more(true);
                }
            }
        });

        let loader = ResultsLoader::new(&path);
        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            page_size: PAGE_SIZE,
            page: next_page,
            needs_intensities,
        }) {
            Ok(new_rois) => {
                let all_loaded = new_rois.len() < PAGE_SIZE;
                *this.all_loaded.lock().unwrap() = all_loaded;
                // Mirror the table append so display indices stay aligned with
                // `displayed_rois` (the next page is pushed after existing rows).
                this.displayed_rois
                    .lock()
                    .unwrap()
                    .extend(new_rois.iter().cloned());

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let model = state.get_rows();
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<ResultsRow>>()
                        {
                            let base = vec_model.row_count();
                            for (i, roi) in new_rois.iter().enumerate() {
                                vec_model.push(to_slint_row(to_display_row(base + i, roi, &specs)));
                            }
                        }
                        state.set_all_rows_loaded(all_loaded);
                        state.set_loading_more(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_load_more failed: {:?}", e);
                let mut p = this.current_page.lock().unwrap();
                if *p > 0 {
                    *p -= 1;
                }
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(w) = ui.upgrade() {
                        w.global::<ResultsState>().set_loading_more(false);
                    }
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sorting
    // -------------------------------------------------------------------------

    fn on_sort_column_changed(&self, column_id: SharedString, sort_ascending: bool) {
        println!("Sort: {}/{}", column_id, sort_ascending);
    }

    // -------------------------------------------------------------------------
    // Row selection: open the ROI's image and highlight its bounding box
    // -------------------------------------------------------------------------

    /// A per-ROI row was selected. Maps the display id back to the stored
    /// [`RoiRow`], then opens its source image in the editor and paints the
    /// ROI's bounding box. Grouped/aggregated rows have no source ROI, so the
    /// lookup misses and the selection is ignored.
    fn on_roi_row_selected(&self, roi_id: i32) {
        if roi_id < 1 {
            return;
        }
        let roi = {
            let rois = self.displayed_rois.lock().unwrap();
            match rois.get((roi_id - 1) as usize) {
                Some(roi) => roi.clone(),
                None => return,
            }
        };
        if roi.image_rel_path.is_empty() {
            warn!("Selected ROI has no image path; cannot open it");
            return;
        }
        let rel_path = PathBuf::from(&roi.image_rel_path);
        self.image_list_controller
            .open_image_and_highlight_roi(&rel_path, roi.bbox_px);
    }

    // -------------------------------------------------------------------------
    // Image filter popup management
    // -------------------------------------------------------------------------

    fn toggle_image_label(&self, label: SharedString) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_image_items());
        let current_popup = model_to_vec(&state.get_filter_image_popup());
        let items = toggle_item_by_label(&current, label.as_str());
        let popup = sync_popup_checked(&items, &current_popup);
        state.set_filter_image_active(any_unchecked(&items));
        state.set_filter_image_all_popup_checked(all_checked(&popup));
        state.set_filter_image_items(to_model(items));
        state.set_filter_image_popup(to_model(popup));
    }

    fn image_search_changed(&self, search: SharedString) {
        *self.image_search.lock().unwrap() = search.to_string();
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_image_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_image_all_popup_checked(all_checked(&popup));
        state.set_filter_image_popup(to_model(popup));
    }

    fn image_select_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.image_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_image_items());
        let items = set_checked_for_search(&current, &search, true);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_image_active(any_unchecked(&items));
        state.set_filter_image_all_popup_checked(all_checked(&popup));
        state.set_filter_image_items(to_model(items));
        state.set_filter_image_popup(to_model(popup));
    }

    fn image_clear_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.image_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_image_items());
        let items = set_checked_for_search(&current, &search, false);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_image_active(any_unchecked(&items));
        state.set_filter_image_all_popup_checked(all_checked(&popup));
        state.set_filter_image_items(to_model(items));
        state.set_filter_image_popup(to_model(popup));
    }

    // -------------------------------------------------------------------------
    // Class filter popup management
    // -------------------------------------------------------------------------

    fn toggle_class_label(&self, label: SharedString) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_class_items());
        let current_popup = model_to_vec(&state.get_filter_class_popup());
        let items = toggle_item_by_label(&current, label.as_str());
        let popup = sync_popup_checked(&items, &current_popup);
        state.set_filter_class_active(any_unchecked(&items));
        state.set_filter_class_all_popup_checked(all_checked(&popup));
        state.set_filter_class_items(to_model(items));
        state.set_filter_class_popup(to_model(popup));
    }

    fn class_search_changed(&self, search: SharedString) {
        *self.class_search.lock().unwrap() = search.to_string();
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_class_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_class_all_popup_checked(all_checked(&popup));
        state.set_filter_class_popup(to_model(popup));
    }

    fn class_select_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.class_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_class_items());
        let items = set_checked_for_search(&current, &search, true);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_class_active(any_unchecked(&items));
        state.set_filter_class_all_popup_checked(all_checked(&popup));
        state.set_filter_class_items(to_model(items));
        state.set_filter_class_popup(to_model(popup));
    }

    fn class_clear_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.class_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_class_items());
        let items = set_checked_for_search(&current, &search, false);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_class_active(any_unchecked(&items));
        state.set_filter_class_all_popup_checked(all_checked(&popup));
        state.set_filter_class_items(to_model(items));
        state.set_filter_class_popup(to_model(popup));
    }

    // -------------------------------------------------------------------------
    // Column-visibility popup management
    // -------------------------------------------------------------------------

    fn toggle_column_label(&self, label: SharedString) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_column_items());
        let current_popup = model_to_vec(&state.get_column_popup());
        let items = toggle_item_by_label(&current, label.as_str());
        let popup = sync_popup_checked(&items, &current_popup);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(items));
        state.set_column_popup(to_model(popup));
    }

    fn column_search_changed(&self, search: SharedString) {
        *self.column_search.lock().unwrap() = search.to_string();
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_column_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_popup(to_model(popup));
    }

    fn column_select_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let items = set_checked_for_search(&current, &search, true);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(items));
        state.set_column_popup(to_model(popup));
    }

    fn column_clear_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let items = set_checked_for_search(&current, &search, false);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(items));
        state.set_column_popup(to_model(popup));
    }

    /// Applies column-visibility selection: updates the stored `column_specs`
    /// and refreshes the view. In grouped mode this re-aggregates (so only the
    /// visible metrics appear as grouped columns); otherwise it updates
    /// `ResultsState.columns` / `visible_column_count` directly.
    fn column_filter_apply(self: &Arc<Self>) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();

        // Build label→checked map from the authoritative column_items list.
        let items = model_to_vec(&state.get_column_items());
        let visibility: BTreeMap<String, bool> = items
            .iter()
            .map(|i| (i.label.to_string(), i.checked))
            .collect();

        // Update the stored column specs (used by the next reload and to decide
        // which channel data to fetch).
        {
            let mut specs = self.column_specs.lock().unwrap();
            for spec in specs.iter_mut() {
                if let Some(&visible) = visibility.get(&spec.label) {
                    spec.visible = visible;
                }
            }
        }

        let grouped = !matches!(
            self.group_config.lock().unwrap().group_by,
            GroupBy::None
        );
        if grouped {
            // Re-aggregate so the grouped columns track the visible metrics.
            state.set_loading_more(true);
            Self::spawn_reload(Arc::clone(self));
        } else {
            let specs = self.column_specs.lock().unwrap().clone();
            let slint_cols = specs_to_slint_cols(&specs);
            let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
            state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
            state.set_visible_column_count(visible_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Slint type helpers
// ---------------------------------------------------------------------------

fn map_group_by(g: ResultsGroupBy) -> GroupBy {
    match g {
        ResultsGroupBy::None => GroupBy::None,
        ResultsGroupBy::Image => GroupBy::Image,
        ResultsGroupBy::Folder => GroupBy::Folder,
        ResultsGroupBy::Regex => GroupBy::Regex,
    }
}

/// Collects the ticked aggregate functions, in display order. Falls back to
/// `Avg` if the user unticked everything (a grouped view with no aggregate
/// would only show the key and count).
fn selected_aggs(state: &ResultsState) -> Vec<AggFunc> {
    let mut aggs = Vec::new();
    if state.get_group_agg_min() {
        aggs.push(AggFunc::Min);
    }
    if state.get_group_agg_max() {
        aggs.push(AggFunc::Max);
    }
    if state.get_group_agg_avg() {
        aggs.push(AggFunc::Avg);
    }
    if state.get_group_agg_median() {
        aggs.push(AggFunc::Median);
    }
    if state.get_group_agg_stdev() {
        aggs.push(AggFunc::Stdev);
    }
    if state.get_group_agg_sum() {
        aggs.push(AggFunc::Sum);
    }
    if aggs.is_empty() {
        aggs.push(AggFunc::Avg);
    }
    aggs
}

fn to_slint_row(row: evanalyzer_app::result::DisplayRow) -> ResultsRow {
    let values: Vec<SharedString> = row.values.into_iter().map(SharedString::from).collect();
    ResultsRow {
        roi_id: row.roi_id,
        values: slint::ModelRc::new(slint::VecModel::from(values)),
    }
}

fn specs_to_slint_cols(specs: &[ColumnSpec]) -> Vec<ResultsColumnDef> {
    specs
        .iter()
        .map(|c| ResultsColumnDef {
            id: c.id.as_str().into(),
            label: c.label.as_str().into(),
            visible: c.visible,
            filterable: c.filterable,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Filter model helpers
// ---------------------------------------------------------------------------

fn to_model(items: Vec<FilterItem>) -> slint::ModelRc<FilterItem> {
    slint::ModelRc::new(slint::VecModel::from(items))
}

fn model_to_vec(model: &slint::ModelRc<FilterItem>) -> Vec<FilterItem> {
    (0..model.row_count())
        .filter_map(|i| model.row_data(i))
        .collect()
}

fn toggle_item_by_label(items: &[FilterItem], label: &str) -> Vec<FilterItem> {
    items
        .iter()
        .map(|item| {
            let mut item = item.clone();
            if item.label.as_str() == label {
                item.checked = !item.checked;
            }
            item
        })
        .collect()
}

fn sync_popup_checked(items: &[FilterItem], popup: &[FilterItem]) -> Vec<FilterItem> {
    let lookup: BTreeMap<&str, bool> =
        items.iter().map(|i| (i.label.as_str(), i.checked)).collect();
    popup
        .iter()
        .map(|item| {
            let mut item = item.clone();
            if let Some(&checked) = lookup.get(item.label.as_str()) {
                item.checked = checked;
            }
            item
        })
        .collect()
}

fn filter_popup_by_search(items: &[FilterItem], search: &str) -> Vec<FilterItem> {
    let lower = search.to_lowercase();
    items
        .iter()
        .filter(|item| lower.is_empty() || item.label.to_lowercase().contains(&lower))
        .cloned()
        .collect()
}

fn set_checked_for_search(items: &[FilterItem], search: &str, checked: bool) -> Vec<FilterItem> {
    let lower = search.to_lowercase();
    items
        .iter()
        .map(|item| {
            let mut item = item.clone();
            if lower.is_empty() || item.label.to_lowercase().contains(&lower) {
                item.checked = checked;
            }
            item
        })
        .collect()
}

fn set_all_checked(items: &[FilterItem], checked: bool) -> Vec<FilterItem> {
    items
        .iter()
        .map(|item| {
            let mut item = item.clone();
            item.checked = checked;
            item
        })
        .collect()
}

fn any_unchecked(items: &[FilterItem]) -> bool {
    items.iter().any(|i| !i.checked)
}

fn all_checked(items: &[FilterItem]) -> bool {
    items.iter().all(|i| i.checked)
}

fn names_to_filter_items(names: &[String]) -> Vec<FilterItem> {
    names
        .iter()
        .map(|n| FilterItem { label: n.as_str().into(), checked: true })
        .collect()
}
