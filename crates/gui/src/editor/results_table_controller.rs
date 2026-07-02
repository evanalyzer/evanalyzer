use crate::editor::images_list_controller::ImagesListController;
use crate::{
    FilterItem, ResultsChartKind, ResultsColumnDef, ResultsGroupBy, ResultsListState, ResultsRow,
    ResultsState, ResultsWindow, UiState,
};
use evanalyzer_app::result::{
    aggregate_rows, build_coloc_detail_column_specs, build_column_specs, coloc_filter_label_any,
    coloc_filter_label_no, coloc_filter_label_with, compute_heatmap, compute_histogram,
    compute_scatter, discover_channels, flatten_coloc_rows, plottable_columns, render_heatmap,
    render_histogram, render_scatter, save_rendered_chart_png, sort_display_rows, to_display_row,
    AggFunc, ColorBy, ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, HeatmapMetric,
    RenderedChart, ResultsExporter, ResultsLoader, RoiRow,
};
use log::warn;
use slint::{ComponentHandle, Model, SharedString};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const PAGE_SIZE: usize = 500;
// Floors under the chart's actual on-screen size (see `on_chart_render_requested`),
// in case the window is reporting a degenerate size (e.g. not yet shown).
const CHART_WIDTH: u32 = 960;
const CHART_HEIGHT: u32 = 560;
const SCATTER_MAX_POINTS: usize = 5_000;

/// Sentinel shown as the first entry of the heatmap "Color by" picker —
/// picking it colors cells by ROI count instead of averaging a column.
const HEATMAP_METRIC_COUNT_LABEL: &str = "Count (ROIs per cell)";

/// Chart picks read off `ResultsState` on the UI thread before handing off to
/// `bg_render_chart` on a background thread — mirrors `GroupConfig`'s role for
/// `bg_reload_grouped`.
struct ChartRenderConfig {
    kind: ResultsChartKind,
    hist_column: String,
    scatter_x: String,
    scatter_y: String,
    color_by: ColorBy,
    bucket_count: usize,
    log_scale: bool,
    /// `HEATMAP_METRIC_COUNT_LABEL` or a column label to average.
    heatmap_metric: String,
    cell_size_px: f64,
    /// The chart area's current on-screen pixel size (a one-shot read of the
    /// window's size at the moment Plot was clicked — see
    /// `on_chart_render_requested`), so rendering matches the actual display
    /// size instead of a fixed resolution that goes blurry/tiny when scaled.
    render_width: u32,
    render_height: u32,
}

/// The most recently rendered chart's pixels — kept so the "Save chart"
/// button can write out exactly what's currently on screen without
/// re-running the query/render pass a second time.
pub(crate) struct LastChart {
    chart: RenderedChart,
    kind: ResultsChartKind,
}

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
    /// User-resized column widths (logical px), keyed by column id. Pure presentation
    /// state — applied on top of `column_specs`/grouped specs when building the Slint
    /// model, never touching `evanalyzer_app::ColumnSpec` (which CSV/XLSX export also
    /// uses, where a pixel width is meaningless). Columns absent here use the default.
    pub(crate) column_widths: Arc<Mutex<HashMap<String, f32>>>,
    pub(crate) current_page: Arc<Mutex<usize>>,
    pub(crate) all_loaded: Arc<Mutex<bool>>,
    pub(crate) image_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) class_filter: Arc<Mutex<Option<Vec<String>>>>,
    pub(crate) coloc_filter: Arc<Mutex<Option<Vec<String>>>>,
    /// Selected single time-frame/depth index, or `None` to show every frame
    /// (the default). `None` whenever the file has no t_stack/z_stack axis.
    pub(crate) t_stack_filter: Arc<Mutex<Option<i32>>>,
    pub(crate) z_stack_filter: Arc<Mutex<Option<i32>>>,
    pub(crate) group_config: Arc<Mutex<GroupConfig>>,
    /// Results-table column id the view is currently sorted by, or `None` for
    /// the default order. Re-applied on every reload (filter/group/page) the
    /// same way the active filters are, until the user picks another column.
    pub(crate) sort_column: Arc<Mutex<Option<String>>>,
    pub(crate) sort_ascending: Arc<Mutex<bool>>,
    pub(crate) image_search: Mutex<String>,
    pub(crate) class_search: Mutex<String>,
    pub(crate) coloc_search: Mutex<String>,
    pub(crate) column_search: Mutex<String>,
    /// Pixels of the most recently rendered chart — lets "Save chart" write
    /// out exactly what's on screen without re-rendering. `None` until the
    /// first successful Plot click.
    pub(crate) last_chart: Arc<Mutex<Option<LastChart>>>,
    /// True while the colocalization detail flat table is the active view.
    /// Cleared whenever `group_apply` fires (switching back to normal view).
    pub(crate) coloc_detail_mode: Arc<Mutex<bool>>,
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
            column_widths: Arc::new(Mutex::new(HashMap::new())),
            current_page: Arc::new(Mutex::new(0)),
            all_loaded: Arc::new(Mutex::new(false)),
            image_filter: Arc::new(Mutex::new(None)),
            class_filter: Arc::new(Mutex::new(None)),
            coloc_filter: Arc::new(Mutex::new(None)),
            t_stack_filter: Arc::new(Mutex::new(None)),
            z_stack_filter: Arc::new(Mutex::new(None)),
            group_config: Arc::new(Mutex::new(GroupConfig::default())),
            sort_column: Arc::new(Mutex::new(None)),
            sort_ascending: Arc::new(Mutex::new(true)),
            image_search: Mutex::new(String::new()),
            class_search: Mutex::new(String::new()),
            coloc_search: Mutex::new(String::new()),
            column_search: Mutex::new(String::new()),
            last_chart: Arc::new(Mutex::new(None)),
            coloc_detail_mode: Arc::new(Mutex::new(false)),
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

        state.on_coloc_filter_label_toggled(cb!(toggle_coloc_label, SharedString));
        state.on_coloc_filter_search_changed(cb!(coloc_search_changed, SharedString));
        state.on_coloc_select_all(cb!(coloc_select_all));
        state.on_coloc_clear_all(cb!(coloc_clear_all));

        state.on_column_label_toggled(cb!(toggle_column_label, SharedString));
        state.on_column_search_changed(cb!(column_search_changed, SharedString));
        state.on_column_select_all(cb!(column_select_all));
        state.on_column_clear_all(cb!(column_clear_all));
        state.on_column_filter_apply(cb!(column_filter_apply));
        state.on_column_width_changed(cb!(on_column_width_changed, SharedString, f32));
        state.on_column_group_toggle(cb!(column_group_toggle, SharedString));

        state.on_sort_requested(cb!(on_sort_column_changed, SharedString, bool));

        state.on_t_stack_changed(cb!(on_t_stack_changed, i32, bool));
        state.on_z_stack_changed(cb!(on_z_stack_changed, i32, bool));

        state.on_roi_row_selected(cb!(on_roi_row_selected, i32));

        // --- chart_render_requested: read picks on the UI thread, render in --
        // the background (mirrors group_apply's read-then-spawn split).
        {
            let this = Arc::clone(self);
            state.on_chart_render_requested(move || {
                let Some(window) = this.ui.upgrade() else { return };
                let state = window.global::<ResultsState>();

                // One-shot read of the window's current physical size, taken
                // right now rather than tracked reactively — a `changed
                // width/height` handler that wrote a Slint property back was
                // tried earlier and caused a property-recursion panic on
                // first layout. Reading `.window().size()` here is a plain
                // Rust call with no Slint binding involved, so it can't
                // re-enter the property graph. Subtracting the toolbar /
                // settings strip / status bar (+ their 1px dividers) leaves
                // the chart's own plotting area, so the bitmap renders at
                // (close to) its actual on-screen size instead of a fixed
                // resolution that wound up badly mismatched — too large
                // relative to the panel, which shrank the text and blurred
                // the downscale.
                let scale = window.window().scale_factor();
                let physical = window.window().size();
                const CHROME_HEIGHT_LOGICAL: f32 = 48.0 + 1.0 + 44.0 + 1.0 + 32.0 + 1.0;
                let chrome_height_physical = (CHROME_HEIGHT_LOGICAL * scale) as u32;
                let render_width = physical.width.max(CHART_WIDTH);
                let render_height =
                    physical.height.saturating_sub(chrome_height_physical).max(CHART_HEIGHT);

                let config = ChartRenderConfig {
                    kind: state.get_chart_kind(),
                    hist_column: state.get_chart_hist_column().to_string(),
                    scatter_x: state.get_chart_scatter_x().to_string(),
                    scatter_y: state.get_chart_scatter_y().to_string(),
                    color_by: match state.get_chart_color_by().as_str() {
                        "class" => ColorBy::Class,
                        "colocalized" => ColorBy::Colocalized,
                        _ => ColorBy::None,
                    },
                    bucket_count: state.get_chart_bucket_count().max(0) as usize,
                    log_scale: state.get_chart_log_scale(),
                    heatmap_metric: state.get_chart_heatmap_metric().to_string(),
                    cell_size_px: state.get_chart_cell_size_px().max(1) as f64,
                    render_width,
                    render_height,
                };

                state.set_chart_status("Rendering...".into());
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_render_chart(this, config));
            });
        }

        // --- chart_save_requested: write the last-rendered chart to disk ------
        {
            let this = Arc::clone(self);
            state.on_chart_save_requested(move || {
                let default_name = {
                    let guard = this.last_chart.lock().unwrap();
                    let Some(last) = guard.as_ref() else { return };
                    match last.kind {
                        ResultsChartKind::Histogram => "histogram.png",
                        ResultsChartKind::Scatter => "scatter.png",
                        ResultsChartKind::Heatmap => "heatmap.png",
                    }
                };

                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("PNG", &["png"])
                    .set_file_name(default_name)
                    .save_file()
                else {
                    return;
                };

                let this = Arc::clone(&this);
                std::thread::spawn(move || {
                    let guard = this.last_chart.lock().unwrap();
                    let Some(last) = guard.as_ref() else { return };
                    if let Err(e) = save_rendered_chart_png(&last.chart, &export_path) {
                        warn!("Chart PNG save failed: {:?}", e);
                    }
                });
            });
        }

        // --- chart_point_lookup: tap-to-inspect tooltip over the chart area ----
        {
            let this = Arc::clone(self);
            state.on_chart_point_lookup(move |mouse_x, mouse_y, area_width, area_height| {
                let guard = this.last_chart.lock().unwrap();
                let Some(last) = guard.as_ref() else { return SharedString::new() };
                let Some(tester) = last.chart.hit_test.as_ref() else { return SharedString::new() };

                let (bmp_w, bmp_h) = (last.chart.width as f32, last.chart.height as f32);
                if area_width <= 0.0 || area_height <= 0.0 || bmp_w <= 0.0 || bmp_h <= 0.0 {
                    return SharedString::new();
                }
                // Mirrors the Image's `image-fit: contain`: the bitmap is
                // scaled uniformly to fit inside the area and centered, so
                // letterbox bars can appear on two sides when the area's
                // aspect ratio doesn't match the bitmap's (e.g. the window
                // was resized after the last Plot click).
                let scale = (area_width / bmp_w).min(area_height / bmp_h);
                let offset_x = (area_width - bmp_w * scale) / 2.0;
                let offset_y = (area_height - bmp_h * scale) / 2.0;
                let bmp_x = (mouse_x - offset_x) / scale;
                let bmp_y = (mouse_y - offset_y) / scale;
                if bmp_x < 0.0 || bmp_y < 0.0 || bmp_x > bmp_w || bmp_y > bmp_h {
                    return SharedString::new();
                }

                match tester.hit_test(bmp_x as f64, bmp_y as f64) {
                    Some(s) => SharedString::from(s),
                    None => SharedString::new(),
                }
            });
        }

        // --- group_apply: read group selection, reload (grouped or paginated) -
        {
            let this = Arc::clone(self);
            state.on_group_apply(move || {
                let Some(window) = this.ui.upgrade() else { return };
                let state = window.global::<ResultsState>();

                // Switching back from coloc detail to normal view.
                *this.coloc_detail_mode.lock().unwrap() = false;

                let config = GroupConfig {
                    group_by: map_group_by(state.get_group_by()),
                    regex: state.get_group_regex().to_string(),
                    aggs: selected_aggs(&state),
                    split_colocalized: state.get_group_split_colocalized(),
                    group_by_class: state.get_group_by_class(),
                };
                *this.group_config.lock().unwrap() = config;
                *this.current_page.lock().unwrap() = 0;
                *this.all_loaded.lock().unwrap() = false;

                state.set_loading_more(true);
                Self::spawn_reload(Arc::clone(&this));
            });
        }

        // --- coloc_detail_requested: load the flat (source × partner) table ---
        {
            let this = Arc::clone(self);
            state.on_coloc_detail_requested(move || {
                *this.coloc_detail_mode.lock().unwrap() = true;
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_reload_coloc_detail(this));
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
                let coloc_model = state.get_filter_coloc_items();
                let total_img = img_model.row_count();
                let total_cls = cls_model.row_count();
                let total_coloc = coloc_model.row_count();

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
                let checked_coloc: Vec<String> = (0..total_coloc)
                    .filter_map(|i| {
                        coloc_model
                            .row_data(i)?
                            .checked
                            .then_some(coloc_model.row_data(i)?.label.to_string())
                    })
                    .collect();

                let image_filter: Option<Vec<String>> =
                    (checked_img.len() < total_img).then_some(checked_img);
                let class_filter: Option<Vec<String>> =
                    (checked_cls.len() < total_cls).then_some(checked_cls);
                let coloc_filter: Option<Vec<String>> =
                    (checked_coloc.len() < total_coloc).then_some(checked_coloc);

                let is_filtered =
                    image_filter.is_some() || class_filter.is_some() || coloc_filter.is_some();
                *this.image_filter.lock().unwrap() = image_filter;
                *this.class_filter.lock().unwrap() = class_filter;
                *this.coloc_filter.lock().unwrap() = coloc_filter;
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
                *this.coloc_search.lock().unwrap() = String::new();

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

                let coloc = set_all_checked(&model_to_vec(&state.get_filter_coloc_items()), true);
                state.set_filter_coloc_active(false);
                state.set_filter_coloc_all_popup_checked(true);
                state.set_filter_coloc_items(to_model(coloc.clone()));
                state.set_filter_coloc_popup(to_model(coloc));

                state.set_filter_active(false);
                state.set_loading_more(true);

                *this.image_filter.lock().unwrap() = None;
                *this.class_filter.lock().unwrap() = None;
                *this.coloc_filter.lock().unwrap() = None;
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
                let coloc_filter = this.coloc_filter.lock().unwrap().clone();
                let t_stack_filter = *this.t_stack_filter.lock().unwrap();
                let z_stack_filter = *this.z_stack_filter.lock().unwrap();
                let group = this.group_config.lock().unwrap().clone();
                let base_specs = this.column_specs.lock().unwrap().clone();
                let is_coloc_detail = *this.coloc_detail_mode.lock().unwrap();

                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .set_file_name(if is_coloc_detail { "coloc_detail.csv" } else { "results.csv" })
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
                        coloc_filter,
                        t_stack_filter,
                        z_stack_filter,
                        ..Default::default()
                    };
                    let result = if is_coloc_detail {
                        exporter.export_coloc_detail_to_csv(filter, &export_path)
                    } else {
                        exporter.export_to_csv(filter, &group, &base_specs, &export_path)
                    };
                    if let Err(e) = result {
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
                let coloc_filter = this.coloc_filter.lock().unwrap().clone();
                let t_stack_filter = *this.t_stack_filter.lock().unwrap();
                let z_stack_filter = *this.z_stack_filter.lock().unwrap();
                let group = this.group_config.lock().unwrap().clone();
                let base_specs = this.column_specs.lock().unwrap().clone();
                let is_coloc_detail = *this.coloc_detail_mode.lock().unwrap();

                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("Excel", &["xlsx"])
                    .set_file_name(if is_coloc_detail { "coloc_detail.xlsx" } else { "results.xlsx" })
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
                        coloc_filter,
                        t_stack_filter,
                        z_stack_filter,
                        ..Default::default()
                    };
                    let result = if is_coloc_detail {
                        exporter.export_coloc_detail_to_xlsx(filter, &export_path)
                    } else {
                        exporter.export_to_xlsx(filter, &group, &base_specs, &export_path)
                    };
                    if let Err(e) = result {
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
        *self.coloc_filter.lock().unwrap() = None;
        *self.t_stack_filter.lock().unwrap() = None;
        *self.z_stack_filter.lock().unwrap() = None;
        *self.group_config.lock().unwrap() = GroupConfig::default();
        *self.sort_column.lock().unwrap() = None;
        *self.sort_ascending.lock().unwrap() = true;
        *self.image_search.lock().unwrap() = String::new();
        *self.class_search.lock().unwrap() = String::new();
        *self.coloc_search.lock().unwrap() = String::new();
        self.displayed_rois.lock().unwrap().clear();

        let ui = self.ui.clone();
        let app_ui = self.app_state.ui_handle.clone();
        let channels_arc = Arc::clone(&self.channels);
        let all_loaded_arc = Arc::clone(&self.all_loaded);
        let column_specs_arc = Arc::clone(&self.column_specs);
        let column_widths_arc = Arc::clone(&self.column_widths);
        let displayed_rois_arc = Arc::clone(&self.displayed_rois);

        std::thread::spawn(move || {
            let loader = ResultsLoader::new(&path);

            let first_page = loader.get_rois(DatabaseFilter {
                page_size: PAGE_SIZE,
                ..Default::default()
            });
            let img_names = loader.get_image_names();
            let cls_names = loader.get_class_names();
            let coloc_partner_classes = loader.get_coloc_partner_class_names();
            // Non-fatal: a file with no time/depth axis (or a lookup error)
            // just means the frame steppers stay hidden.
            let t_range = loader.get_t_stack_range().unwrap_or(None);
            let z_range = loader.get_z_stack_range().unwrap_or(None);

            match (first_page, img_names, cls_names, coloc_partner_classes) {
                (Ok(rois), Ok(img_names), Ok(cls_names), Ok(coloc_partner_classes)) => {
                    let channels = discover_channels(&rois);
                    let specs = build_column_specs(&channels, &coloc_partner_classes);
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

                            let widths = column_widths_arc.lock().unwrap().clone();
                            let slint_cols: Vec<ResultsColumnDef> =
                                specs_to_slint_cols(&specs, &widths);
                            let visible_count =
                                specs.iter().filter(|c| c.visible).count() as i32;
                            let total_width: f32 = slint_cols
                                .iter()
                                .filter(|c| c.visible)
                                .map(|c| c.width)
                                .sum();

                            let column_items: Vec<FilterItem> = mark_group_headers(
                                specs
                                    .iter()
                                    .map(|c| FilterItem {
                                        label: c.label.as_str().into(),
                                        checked: c.visible,
                                        group: column_group(&c.id).into(),
                                        group_header: false,
                                        group_all_checked: false,
                                    })
                                    .collect(),
                            );

                            state.set_columns(slint::ModelRc::new(
                                slint::VecModel::from(slint_cols),
                            ));
                            state.set_visible_column_count(visible_count);
                            state.set_columns_total_width(total_width);
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

                            // "No" / "Yes (any class)" are always offered; the rest are the
                            // partner classes actually present in this file's coloc_json data,
                            // letting the user filter for e.g. "Colocalizes with Nucleus".
                            let mut coloc_labels = vec![
                                coloc_filter_label_no().to_string(),
                                coloc_filter_label_any().to_string(),
                            ];
                            coloc_labels.extend(
                                coloc_partner_classes.iter().map(|c| coloc_filter_label_with(c)),
                            );
                            let coloc_items = names_to_filter_items(&coloc_labels);
                            state.set_filter_coloc_items(slint::ModelRc::new(
                                slint::VecModel::from(coloc_items.clone()),
                            ));
                            state.set_filter_coloc_popup(slint::ModelRc::new(
                                slint::VecModel::from(coloc_items),
                            ));
                            state.set_filter_coloc_active(false);
                            state.set_filter_coloc_all_popup_checked(true);

                            state.set_filter_active(false);
                            state.set_group_active(false);
                            state.set_group_by(ResultsGroupBy::None);
                            state.set_group_regex(slint::SharedString::new());
                            state.set_sort_column_id(slint::SharedString::new());
                            state.set_sort_ascending(true);
                            state.set_chart_image(slint::Image::default());
                            state.set_chart_status("Pick a column and click Plot".into());
                            state.set_chart_hist_column(slint::SharedString::new());
                            state.set_chart_scatter_x(slint::SharedString::new());
                            state.set_chart_scatter_y(slint::SharedString::new());
                            state.set_chart_color_by(slint::SharedString::new());
                            state.set_chart_log_scale(false);
                            state.set_chart_heatmap_metric(slint::SharedString::new());
                            state.set_chart_cell_size_px(20);
                            state.set_chart_plottable_columns(slint::ModelRc::new(
                                slint::VecModel::from(plottable_column_labels(&specs)),
                            ));
                            state.set_chart_heatmap_metric_options(slint::ModelRc::new(
                                slint::VecModel::from(heatmap_metric_options(&specs)),
                            ));
                            state.set_t_stack_active(t_range.is_some());
                            state.set_t_stack_min(t_range.map_or(0, |(min, _)| min));
                            state.set_t_stack_max(t_range.map_or(0, |(_, max)| max));
                            state.set_selected_t_stack(t_range.map_or(0, |(min, _)| min));
                            state.set_t_stack_show_all(true);
                            state.set_z_stack_active(z_range.is_some());
                            state.set_z_stack_min(z_range.map_or(0, |(min, _)| min));
                            state.set_z_stack_max(z_range.map_or(0, |(_, max)| max));
                            state.set_selected_z_stack(z_range.map_or(0, |(min, _)| min));
                            state.set_z_stack_show_all(true);
                            state.set_all_rows_loaded(all_loaded);
                            state.set_loading_more(false);
                            state.set_rows(slint::ModelRc::new(slint::VecModel::from(
                                slint_rows,
                            )));
                            let _ = window.show();
                        }
                    });
                }
                (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
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
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();
        let config = this.group_config.lock().unwrap().clone();
        // Per-ROI specs carry the column-visibility selection; only visible
        // metrics become grouped columns.
        let base_specs = this.column_specs.lock().unwrap().clone();
        let sort_column = this.sort_column.lock().unwrap().clone();
        let sort_ascending = *this.sort_ascending.lock().unwrap();

        let loader = ResultsLoader::new(&path);
        // Aggregation needs every matching row, so fetch all (page_size 0).
        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(rois) => {
                let (specs, mut display_rows) = aggregate_rows(&rois, &config, &base_specs);
                // The grouped view is fully materialized in memory (unlike the
                // paginated per-ROI view), so sorting it is a plain in-memory
                // sort rather than another DB round-trip.
                if let Some(col) = &sort_column {
                    sort_display_rows(&mut display_rows, &specs, col, sort_ascending);
                }
                // Grouped view is never paginated.
                *this.all_loaded.lock().unwrap() = true;
                // Grouped rows aggregate many ROIs, so there is no single source
                // ROI to open/highlight when one is selected.
                this.displayed_rois.lock().unwrap().clear();
                let widths = this.column_widths.lock().unwrap().clone();

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let visible_count = specs.len() as i32;
                        let slint_rows: Vec<ResultsRow> =
                            display_rows.into_iter().map(to_slint_row).collect();
                        let slint_cols = specs_to_slint_cols(&specs, &widths);
                        let total_width: f32 = slint_cols
                            .iter()
                            .filter(|c| c.visible)
                            .map(|c| c.width)
                            .sum();

                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(
                            slint_cols,
                        )));
                        state.set_visible_column_count(visible_count);
                        state.set_columns_total_width(total_width);
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
    // Background: colocalization detail flat table (one row per source × partner)
    // -------------------------------------------------------------------------

    fn bg_reload_coloc_detail(this: Arc<Self>) {
        let ui = this.ui.clone();
        let finish = |ui: slint::Weak<ResultsWindow>| {
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = ui.upgrade() {
                    let state = w.global::<ResultsState>();
                    state.set_loading_more(false);
                    state.set_group_computing(false);
                }
            });
        };

        let Some(path) = this.path.lock().unwrap().clone() else {
            finish(ui);
            return;
        };

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();
        let widths = this.column_widths.lock().unwrap().clone();

        let loader = ResultsLoader::new(&path);
        let coloc_partner_classes = loader.get_coloc_partner_class_names().unwrap_or_default();

        // Load all ROIs in the same image/time-frame for partner property lookup.
        // Class and coloc filters are intentionally omitted so that partner ROIs from
        // any class are always available for the detail columns.
        let partner_lookup = match loader.get_rois(DatabaseFilter {
            image_filter: image_filter.clone(),
            class_filter: None,
            coloc_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(r) => r,
            Err(e) => {
                warn!("bg_reload_coloc_detail (partner lookup) failed: {:?}", e);
                finish(ui);
                return;
            }
        };

        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(source_rois) => {
                let channels = discover_channels(&partner_lookup);
                let specs = build_coloc_detail_column_specs(&channels, &coloc_partner_classes);
                let display_rows = flatten_coloc_rows(&source_rois, &partner_lookup, &specs);

                // Row selection doesn't apply to the flat coloc view.
                this.displayed_rois.lock().unwrap().clear();
                *this.all_loaded.lock().unwrap() = true;

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let slint_rows: Vec<ResultsRow> =
                            display_rows.into_iter().map(to_slint_row).collect();
                        let slint_cols = specs_to_slint_cols(&specs, &widths);
                        let visible_count = specs.len() as i32;
                        let total_width: f32 =
                            slint_cols.iter().filter(|c| c.visible).map(|c| c.width).sum();

                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
                        state.set_visible_column_count(visible_count);
                        state.set_columns_total_width(total_width);
                        state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
                        state.set_all_rows_loaded(true);
                        state.set_loading_more(false);
                        state.set_group_computing(false);
                        state.set_group_active(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_reload_coloc_detail failed: {:?}", e);
                finish(ui);
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
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();
        let specs = this.column_specs.lock().unwrap().clone();
        let needs_intensities = specs.iter().any(|c| c.visible && c.id.starts_with("ch"));
        let sort_column = this.sort_column.lock().unwrap().clone();
        let sort_ascending = *this.sort_ascending.lock().unwrap();
        let ui = this.ui.clone();

        let loader = ResultsLoader::new(&path);
        match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: 0,
            needs_intensities,
            sort_column,
            sort_ascending,
        }) {
            Ok(rois) => {
                let all_loaded = rois.len() < PAGE_SIZE;
                *this.all_loaded.lock().unwrap() = all_loaded;
                *this.displayed_rois.lock().unwrap() = rois.clone();
                let widths = this.column_widths.lock().unwrap().clone();

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
                        let slint_cols = specs_to_slint_cols(&specs, &widths);
                        let total_width: f32 = slint_cols
                            .iter()
                            .filter(|c| c.visible)
                            .map(|c| c.width)
                            .sum();
                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
                        state.set_visible_column_count(visible_count);
                        state.set_columns_total_width(total_width);
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
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();
        let specs = this.column_specs.lock().unwrap().clone();
        let needs_intensities = specs.iter().any(|c| c.visible && c.id.starts_with("ch"));
        let sort_column = this.sort_column.lock().unwrap().clone();
        let sort_ascending = *this.sort_ascending.lock().unwrap();
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
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: next_page,
            needs_intensities,
            sort_column,
            sort_ascending,
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

    fn on_sort_column_changed(self: &Arc<Self>, column_id: SharedString, sort_ascending: bool) {
        *self.sort_column.lock().unwrap() = Some(column_id.to_string());
        *self.sort_ascending.lock().unwrap() = sort_ascending;
        *self.current_page.lock().unwrap() = 0;
        *self.all_loaded.lock().unwrap() = false;

        let Some(window) = self.ui.upgrade() else { return };
        window.global::<ResultsState>().set_loading_more(true);

        Self::spawn_reload(Arc::clone(self));
    }

    // -------------------------------------------------------------------------
    // Frame navigation (time/depth stacks)
    // -------------------------------------------------------------------------

    fn on_t_stack_changed(self: &Arc<Self>, value: i32, show_all: bool) {
        *self.t_stack_filter.lock().unwrap() = if show_all { None } else { Some(value) };
        *self.current_page.lock().unwrap() = 0;
        *self.all_loaded.lock().unwrap() = false;

        let Some(window) = self.ui.upgrade() else { return };
        window.global::<ResultsState>().set_loading_more(true);

        Self::spawn_reload(Arc::clone(self));
    }

    fn on_z_stack_changed(self: &Arc<Self>, value: i32, show_all: bool) {
        *self.z_stack_filter.lock().unwrap() = if show_all { None } else { Some(value) };
        *self.current_page.lock().unwrap() = 0;
        *self.all_loaded.lock().unwrap() = false;

        let Some(window) = self.ui.upgrade() else { return };
        window.global::<ResultsState>().set_loading_more(true);

        Self::spawn_reload(Arc::clone(self));
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
    // Colocalized-column filter popup management
    // -------------------------------------------------------------------------

    fn toggle_coloc_label(&self, label: SharedString) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_coloc_items());
        let current_popup = model_to_vec(&state.get_filter_coloc_popup());
        let items = toggle_item_by_label(&current, label.as_str());
        let popup = sync_popup_checked(&items, &current_popup);
        state.set_filter_coloc_active(any_unchecked(&items));
        state.set_filter_coloc_all_popup_checked(all_checked(&popup));
        state.set_filter_coloc_items(to_model(items));
        state.set_filter_coloc_popup(to_model(popup));
    }

    fn coloc_search_changed(&self, search: SharedString) {
        *self.coloc_search.lock().unwrap() = search.to_string();
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_coloc_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_coloc_all_popup_checked(all_checked(&popup));
        state.set_filter_coloc_popup(to_model(popup));
    }

    fn coloc_select_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.coloc_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_coloc_items());
        let items = set_checked_for_search(&current, &search, true);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_coloc_active(any_unchecked(&items));
        state.set_filter_coloc_all_popup_checked(all_checked(&popup));
        state.set_filter_coloc_items(to_model(items));
        state.set_filter_coloc_popup(to_model(popup));
    }

    fn coloc_clear_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.coloc_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_filter_coloc_items());
        let items = set_checked_for_search(&current, &search, false);
        let popup = filter_popup_by_search(&items, &search);
        state.set_filter_coloc_active(any_unchecked(&items));
        state.set_filter_coloc_all_popup_checked(all_checked(&popup));
        state.set_filter_coloc_items(to_model(items));
        state.set_filter_coloc_popup(to_model(popup));
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
        state.set_column_items(to_model(mark_group_headers(items)));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    fn column_search_changed(&self, search: SharedString) {
        *self.column_search.lock().unwrap() = search.to_string();
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_column_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    fn column_select_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let items = set_checked_for_search(&current, &search, true);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(mark_group_headers(items)));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    fn column_clear_all(&self) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let items = set_checked_for_search(&current, &search, false);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(mark_group_headers(items)));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    /// Selects/clears every column in one popup section at once (e.g. all
    /// "Intensity" columns), scoped to the active search the same way
    /// `(Select All)` is — searching first, then toggling the group, only
    /// affects the rows currently visible.
    fn column_group_toggle(&self, group: SharedString) {
        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let currently_all_checked =
            group_all_checked_for_search(&current, group.as_str(), &search);
        let items =
            set_group_checked_for_search(&current, group.as_str(), &search, !currently_all_checked);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(mark_group_headers(items)));
        state.set_column_popup(to_model(mark_group_headers(popup)));
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

        let specs = self.column_specs.lock().unwrap().clone();
        state.set_chart_plottable_columns(slint::ModelRc::new(slint::VecModel::from(
            plottable_column_labels(&specs),
        )));
        state.set_chart_heatmap_metric_options(slint::ModelRc::new(slint::VecModel::from(
            heatmap_metric_options(&specs),
        )));

        let grouped = !matches!(
            self.group_config.lock().unwrap().group_by,
            GroupBy::None
        );
        if grouped {
            // Re-aggregate so the grouped columns track the visible metrics.
            state.set_loading_more(true);
            Self::spawn_reload(Arc::clone(self));
        } else {
            let widths = self.column_widths.lock().unwrap().clone();
            let slint_cols = specs_to_slint_cols(&specs, &widths);
            let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
            let total_width: f32 = slint_cols.iter().filter(|c| c.visible).map(|c| c.width).sum();
            state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
            state.set_visible_column_count(visible_count);
            state.set_columns_total_width(total_width);
        }
    }

    /// Live-updates one column's width as the user drags its header's resize handle.
    /// Operates directly on whatever `ResultsState.columns` currently holds — the per-ROI
    /// specs or a grouped/aggregated column list — so it works in both view modes without
    /// needing to know which is active. The chosen width is also remembered in
    /// `column_widths` so it survives the next reload/filter/group-apply.
    ///
    /// Mutates the existing `VecModel` row in place via `set_row_data` instead of pushing a
    /// brand-new model through `set_columns`: replacing the whole model on every drag-move
    /// event makes the `for col in ResultsState.columns` repeater recreate every header cell,
    /// which would reset the resize handle's own `dragging` flag mid-drag (the bug this fixes
    /// — the drag could only ever move 1px before getting stuck).
    fn on_column_width_changed(&self, col_id: SharedString, new_width: f32) {
        let clamped = new_width.max(40.0);
        self.column_widths
            .lock()
            .unwrap()
            .insert(col_id.to_string(), clamped);

        let Some(window) = self.ui.upgrade() else { return };
        let state = window.global::<ResultsState>();
        let model = state.get_columns();
        if let Some(vec_model) = model
            .as_any()
            .downcast_ref::<slint::VecModel<ResultsColumnDef>>()
        {
            for i in 0..vec_model.row_count() {
                if let Some(mut c) = vec_model.row_data(i) {
                    if c.id == col_id {
                        c.width = clamped;
                        vec_model.set_row_data(i, c);
                        break;
                    }
                }
            }
        }
        let total_width: f32 = (0..model.row_count())
            .filter_map(|i| model.row_data(i))
            .filter(|c| c.visible)
            .map(|c| c.width)
            .sum();
        state.set_columns_total_width(total_width);
    }

    // -------------------------------------------------------------------------
    // Chart view: histogram / scatter rendering
    // -------------------------------------------------------------------------

    /// Stashes a freshly rendered chart so a later "Save chart" click can
    /// write out exactly these pixels without re-rendering.
    fn cache_last_chart(&self, chart: &RenderedChart, kind: ResultsChartKind) {
        self.last_chart.lock().unwrap().replace(LastChart { chart: chart.clone(), kind });
    }

    /// Fetches every ROI matching the current filters (mirrors
    /// `bg_reload_grouped`'s "aggregation needs every matching row" fetch),
    /// computes the requested chart in `evanalyzer_app::result` (the same
    /// functions a future CLI export command would call), renders it with
    /// `plotters` into an RGB8 buffer, and pushes it to the UI as a
    /// `slint::Image`.
    fn bg_render_chart(this: Arc<Self>, config: ChartRenderConfig) {
        let ui = this.ui.clone();
        let report = move |status: String, chart: Option<RenderedChart>| {
            let ui = ui.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(window) = ui.upgrade() else { return };
                let state = window.global::<ResultsState>();
                if let Some(chart) = chart {
                    let mut buf =
                        slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(chart.width, chart.height);
                    {
                        let slice = buf.make_mut_slice();
                        for (px, rgb) in slice.iter_mut().zip(chart.rgb.chunks_exact(3)) {
                            *px = slint::Rgb8Pixel { r: rgb[0], g: rgb[1], b: rgb[2] };
                        }
                    }
                    state.set_chart_image(slint::Image::from_rgb8(buf));
                }
                state.set_chart_status(status.into());
            });
        };

        let Some(path) = this.path.lock().unwrap().clone() else {
            report("No results file loaded.".into(), None);
            return;
        };

        // Per-ROI specs carry the visible-column selection; chart axis picks
        // are resolved against them (label -> column id) below.
        let specs = this.column_specs.lock().unwrap().clone();
        let resolve_id =
            |label: &str| specs.iter().find(|c| c.label == label).map(|c| c.id.clone());

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();

        let loader = ResultsLoader::new(&path);
        // A chart summarizes every matching ROI, not just a loaded page.
        let rois = match loader.get_rois(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(rois) => rois,
            Err(e) => {
                warn!("bg_render_chart failed to load ROIs: {:?}", e);
                report("Failed to load data for the chart.".into(), None);
                return;
            }
        };

        if rois.is_empty() {
            report("No ROIs match the current filters.".into(), None);
            return;
        }

        match config.kind {
            ResultsChartKind::Histogram => {
                let Some(col_id) = resolve_id(&config.hist_column) else {
                    report("Pick a column to plot.".into(), None);
                    return;
                };
                let Some(data) = compute_histogram(
                    &rois,
                    &col_id,
                    &specs,
                    config.bucket_count.max(1),
                    config.log_scale,
                ) else {
                    report("No numeric data for this column.".into(), None);
                    return;
                };
                let status = if data.excluded_non_positive > 0 {
                    format!(
                        "Excluded {} value(s) <= 0 — can't be shown on a log scale.",
                        data.excluded_non_positive
                    )
                } else {
                    String::new()
                };
                match render_histogram(&data, config.render_width, config.render_height) {
                    Ok(chart) => {
                        this.cache_last_chart(&chart, config.kind);
                        report(status, Some(chart));
                    }
                    Err(e) => {
                        warn!("render_histogram failed: {:?}", e);
                        report("Failed to render the chart.".into(), None);
                    }
                }
            }
            ResultsChartKind::Scatter => {
                let (Some(x_id), Some(y_id)) =
                    (resolve_id(&config.scatter_x), resolve_id(&config.scatter_y))
                else {
                    report("Pick X and Y columns to plot.".into(), None);
                    return;
                };
                let Some(data) =
                    compute_scatter(&rois, &x_id, &y_id, config.color_by, &specs, SCATTER_MAX_POINTS)
                else {
                    report("No numeric data for these columns.".into(), None);
                    return;
                };
                let status = match data.sampled_from {
                    Some(total) => {
                        format!("Showing {} of {} points (sampled).", data.points.len(), total)
                    }
                    None => String::new(),
                };
                match render_scatter(&data, config.render_width, config.render_height) {
                    Ok(chart) => {
                        this.cache_last_chart(&chart, config.kind);
                        report(status, Some(chart));
                    }
                    Err(e) => {
                        warn!("render_scatter failed: {:?}", e);
                        report("Failed to render the chart.".into(), None);
                    }
                }
            }
            ResultsChartKind::Heatmap => {
                // The heatmap bins raw pixel centroids; overlaying more than
                // one image's coordinate space as-is would silently mix
                // unrelated images together rather than just look odd, so
                // require the user to filter down to a single image first.
                let image_count = rois
                    .iter()
                    .map(|r| r.image_name.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                if image_count > 1 {
                    report(
                        format!(
                            "Heatmap mixes pixel coordinates from {image_count} images — use the Image filter to pick a single image for a meaningful result."
                        ),
                        None,
                    );
                    return;
                }

                let metric = if config.heatmap_metric.is_empty()
                    || config.heatmap_metric == HEATMAP_METRIC_COUNT_LABEL
                {
                    HeatmapMetric::Count
                } else {
                    let Some(col_id) = resolve_id(&config.heatmap_metric) else {
                        report("Pick how to color the heatmap.".into(), None);
                        return;
                    };
                    HeatmapMetric::Average(col_id)
                };
                let Some(data) = compute_heatmap(&rois, &metric, &specs, config.cell_size_px)
                else {
                    report("No data to bin into a heatmap.".into(), None);
                    return;
                };
                match render_heatmap(&data, config.render_width, config.render_height) {
                    Ok(chart) => {
                        this.cache_last_chart(&chart, config.kind);
                        report(String::new(), Some(chart));
                    }
                    Err(e) => {
                        warn!("render_heatmap failed: {:?}", e);
                        report("Failed to render the chart.".into(), None);
                    }
                }
            }
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

/// Labels of the columns the chart view's axis/column pickers may offer —
/// visible numeric metrics only. Recomputed wherever `column_items` is, so
/// hiding a column also removes it from the chart pickers.
fn plottable_column_labels(specs: &[ColumnSpec]) -> Vec<SharedString> {
    plottable_columns(specs)
        .iter()
        .map(|c| c.label.as_str().into())
        .collect()
}

/// Options for the heatmap's "Color by" picker: the `Count` sentinel first,
/// then every plottable column label (averaged per cell when picked).
fn heatmap_metric_options(specs: &[ColumnSpec]) -> Vec<SharedString> {
    let mut options = vec![SharedString::from(HEATMAP_METRIC_COUNT_LABEL)];
    options.extend(plottable_column_labels(specs));
    options
}

fn specs_to_slint_cols(specs: &[ColumnSpec], widths: &HashMap<String, f32>) -> Vec<ResultsColumnDef> {
    specs
        .iter()
        .map(|c| ResultsColumnDef {
            id: c.id.as_str().into(),
            label: c.label.as_str().into(),
            visible: c.visible,
            filterable: c.filterable,
            width: widths.get(&c.id).copied().unwrap_or(100.0),
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
        .map(|n| FilterItem {
            label: n.as_str().into(),
            checked: true,
            group: SharedString::new(),
            group_header: false,
            group_all_checked: false,
        })
        .collect()
}

/// Popup section a column belongs to (e.g. all "Intensity" columns can be
/// switched on/off as one group instead of one column at a time), or `""` for
/// columns that don't belong to a section.
fn column_group(col_id: &str) -> &'static str {
    if col_id.starts_with("ch") {
        "Intensity"
    } else if col_id.starts_with("coloc_partner__") {
        "Colocalization"
    } else {
        ""
    }
}

/// Recomputes `group_header`/`group_all_checked` for an ordered list of
/// [`FilterItem`]s: the first item of each contiguous `group` run gets
/// `group_header = true` and `group_all_checked` set to whether every item
/// sharing that group is currently checked. Must be re-run any time the list
/// is rebuilt (toggle, search, select/clear-all, group toggle) since those
/// operations don't otherwise know about section boundaries.
fn mark_group_headers(items: Vec<FilterItem>) -> Vec<FilterItem> {
    let mut group_checked: BTreeMap<String, bool> = BTreeMap::new();
    for item in &items {
        if item.group.is_empty() {
            continue;
        }
        let entry = group_checked.entry(item.group.to_string()).or_insert(true);
        *entry &= item.checked;
    }

    let mut prev_group: Option<String> = None;
    items
        .into_iter()
        .map(|mut item| {
            let is_header =
                !item.group.is_empty() && prev_group.as_deref() != Some(item.group.as_str());
            item.group_header = is_header;
            item.group_all_checked = is_header
                && *group_checked.get(item.group.as_str()).unwrap_or(&false);
            prev_group = Some(item.group.to_string());
            item
        })
        .collect()
}

/// Whether every item belonging to `group` *and* matching `search` is
/// currently checked — scoped the same way `(Select All)` already is, so a
/// group toggle while searching only affects the rows the user can see.
fn group_all_checked_for_search(items: &[FilterItem], group: &str, search: &str) -> bool {
    let lower = search.to_lowercase();
    items
        .iter()
        .filter(|i| {
            i.group == group && (lower.is_empty() || i.label.to_lowercase().contains(&lower))
        })
        .all(|i| i.checked)
}

/// Sets `checked` on every item belonging to `group` *and* matching `search`,
/// leaving all other items untouched.
fn set_group_checked_for_search(
    items: &[FilterItem],
    group: &str,
    search: &str,
    checked: bool,
) -> Vec<FilterItem> {
    let lower = search.to_lowercase();
    items
        .iter()
        .map(|item| {
            let mut item = item.clone();
            if item.group == group
                && (lower.is_empty() || item.label.to_lowercase().contains(&lower))
            {
                item.checked = checked;
            }
            item
        })
        .collect()
}
