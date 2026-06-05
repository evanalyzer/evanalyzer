use crate::{
    FilterItem, ResultsColumnDef, ResultsListState, ResultsRow, ResultsState, ResultsWindow,
    UiState,
};
use evanalyzer_app::result::{
    build_column_specs, discover_channels, to_display_row, ColumnSpec, DatabaseFilter,
    ResultsExporter, ResultsLoader,
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
    pub(crate) path: Arc<Mutex<Option<PathBuf>>>,
    pub(crate) channels: Arc<Mutex<Vec<i32>>>,
    pub(crate) column_specs: Arc<Mutex<Vec<ColumnSpec>>>,
    pub(crate) current_page: Arc<Mutex<usize>>,
    pub(crate) all_loaded: Arc<Mutex<bool>>,
    pub(crate) image_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) class_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) image_search: Mutex<String>,
    pub(crate) class_search: Mutex<String>,
    pub(crate) column_search: Mutex<String>,
}

impl ResultsTableController {
    pub fn new(ui: slint::Weak<ResultsWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            path: Arc::new(Mutex::new(None)),
            channels: Arc::new(Mutex::new(Vec::new())),
            column_specs: Arc::new(Mutex::new(Vec::new())),
            current_page: Arc::new(Mutex::new(0)),
            all_loaded: Arc::new(Mutex::new(false)),
            image_filter: Arc::new(Mutex::new(None)),
            class_filter: Arc::new(Mutex::new(None)),
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

                let arc = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_reload_page0(arc));
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

                let arc = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_reload_page0(arc));
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
                    if let Err(e) = exporter.export_to_csv(filter, &export_path) {
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
                    if let Err(e) = exporter.export_to_xlsx(filter, &export_path) {
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
        *self.image_search.lock().unwrap() = String::new();
        *self.class_search.lock().unwrap() = String::new();

        let ui = self.ui.clone();
        let app_ui = self.app_state.ui_handle.clone();
        let channels_arc = Arc::clone(&self.channels);
        let all_loaded_arc = Arc::clone(&self.all_loaded);
        let column_specs_arc = Arc::clone(&self.column_specs);

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

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let slint_rows: Vec<ResultsRow> = rois
                            .iter()
                            .enumerate()
                            .map(|(i, r)| to_slint_row(to_display_row(i, r, &specs)))
                            .collect();
                        let state = window.global::<ResultsState>();
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

    /// Applies column-visibility selection: updates `ResultsState.columns`,
    /// `visible_column_count`, and the stored `column_specs` (used on the next
    /// DB reload to skip fetching unused channel data).
    fn column_filter_apply(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();

        // Build label→checked map from the authoritative column_items list.
        let items = model_to_vec(&state.get_column_items());
        let visibility: BTreeMap<String, bool> = items
            .iter()
            .map(|i| (i.label.to_string(), i.checked))
            .collect();

        // Update the stored column specs so future DB reloads use the right
        // fetch_intensities flag.
        {
            let mut specs = self.column_specs.lock().unwrap();
            for spec in specs.iter_mut() {
                if let Some(&visible) = visibility.get(&spec.label) {
                    spec.visible = visible;
                }
            }
            let specs_clone = specs.clone();
            let slint_cols = specs_to_slint_cols(&specs_clone);
            let visible_count = specs_clone.iter().filter(|c| c.visible).count() as i32;
            state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
            state.set_visible_column_count(visible_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Slint type helpers
// ---------------------------------------------------------------------------

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
