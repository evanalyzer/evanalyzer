use crate::editor::images_list_controller::ImagesListController;
use crate::editor::results_matrix_controller::agg_from_label;
use crate::{
    ExportBatchItem, FilterItem, ResultsChartKind, ResultsColumnDef, ResultsGroupBy,
    ResultsListState, ResultsRow, ResultsState, ResultsWindow, UiState,
};
use evanalyzer_app::result::{
    AggFunc, ColorBy, ColumnSpec, DEFAULT_WINDOW_PAGES, DatabaseFilter, EvictEdge, GroupBy,
    GroupConfig, HeatmapColorScheme, HeatmapMetric, HeatmapRange, ObjectRow, PageRowCounts,
    RenderedChart, ResultsExporter, ResultsLoader, RowWindow, aggregate_objects_sql,
    build_coloc_detail_column_specs, build_column_specs, coloc_filter_label_any,
    coloc_filter_label_no, coloc_filter_label_with, coloc_partner_ids, compute_heatmap,
    compute_histogram, compute_scatter, discover_channels, discover_coloc_detail_columns,
    flatten_coloc_rows, plottable_columns, render_heatmap, render_histogram, render_scatter,
    save_rendered_chart_png, sort_display_rows, to_display_row,
};
use evanalyzer_cfg::core_types::InternalErrors;
use log::warn;
use slint::{ComponentHandle, Model, SharedString};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const PAGE_SIZE: usize = 500;
// Floors under the chart's actual on-screen size (see `on_chart_render_requested`),
// in case the window is reporting a degenerate size (e.g. not yet shown).
const CHART_WIDTH: u32 = 960;
const CHART_HEIGHT: u32 = 560;
const SCATTER_MAX_POINTS: usize = 5_000;

/// Sentinel shown as the first entry of the heatmap "Color by" picker —
/// picking it colors cells by object count instead of averaging a column.
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
    heatmap_color_scheme: HeatmapColorScheme,
    heatmap_range: HeatmapRange,
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
    /// The window of currently-loaded per-object pages, used to map a selected
    /// row back to its source object (image + bounding box) by `object_id` —
    /// bounded to `DEFAULT_WINDOW_PAGES` pages, evicting the oldest page from
    /// the opposite scroll edge as new pages load, so scrolling through a huge
    /// file doesn't hold it all in memory. Empty while a grouped/aggregated or
    /// coloc-detail view is active (neither supports row selection).
    pub(crate) displayed_objects: Arc<Mutex<RowWindow>>,
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
    /// Column specs for the coloc-detail flat table — mirrors `column_specs`'
    /// role but for the `coloc_detail__*` column set built by
    /// `build_coloc_detail_column_specs`. Empty until the view is entered for
    /// the first time; from then on it persists visibility choices across
    /// re-entries into coloc-detail mode (see `bg_reload_coloc_detail`).
    pub(crate) coloc_detail_column_specs: Arc<Mutex<Vec<ColumnSpec>>>,
    /// Bounds the coloc-detail flat table's memory the same way `displayed_objects`
    /// bounds the normal table's, evicting the oldest source page's flattened
    /// rows once the window exceeds `DEFAULT_WINDOW_PAGES`. Lighter than
    /// `RowWindow` since this view never supports row selection — nothing
    /// ever needs to look a row up by id, only how many displayed rows each
    /// loaded source page produced.
    pub(crate) coloc_detail_page_rows: Arc<Mutex<PageRowCounts>>,
    /// Set while a background export is running; `export_cancel` flips it so
    /// the export loop can stop after its current file instead of a hard
    /// abort mid-write. `None` when no export is in flight.
    pub(crate) export_cancel_flag: Arc<Mutex<Option<Arc<AtomicBool>>>>,
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
            displayed_objects: Arc::new(Mutex::new(RowWindow::new(DEFAULT_WINDOW_PAGES))),
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
            coloc_detail_column_specs: Arc::new(Mutex::new(Vec::new())),
            coloc_detail_page_rows: Arc::new(Mutex::new(PageRowCounts::new(DEFAULT_WINDOW_PAGES))),
            export_cancel_flag: Arc::new(Mutex::new(None)),
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

        state.on_object_row_selected(cb!(on_object_row_selected, SharedString));

        // --- chart_render_requested: read picks on the UI thread, render in --
        // the background (mirrors group_apply's read-then-spawn split).
        {
            let this = Arc::clone(self);
            state.on_chart_render_requested(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
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
                let render_height = physical
                    .height
                    .saturating_sub(chrome_height_physical)
                    .max(CHART_HEIGHT);

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
                    heatmap_color_scheme: HeatmapColorScheme::from_label(
                        &state.get_chart_heatmap_color_scheme(),
                    ),
                    heatmap_range: if state.get_chart_heatmap_range_auto() {
                        HeatmapRange::Auto
                    } else {
                        HeatmapRange::Manual {
                            min: state.get_chart_heatmap_range_min() as f64,
                            max: state.get_chart_heatmap_range_max() as f64,
                        }
                    },
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
                let Some(last) = guard.as_ref() else {
                    return SharedString::new();
                };
                let Some(tester) = last.chart.hit_test.as_ref() else {
                    return SharedString::new();
                };

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
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
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
                *this.current_page.lock().unwrap() = 0;
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_reload_coloc_detail_page0(this));
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
                let is_coloc_detail = *arc.coloc_detail_mode.lock().unwrap();
                std::thread::spawn(move || {
                    if is_coloc_detail {
                        Self::bg_load_more_coloc_detail(arc);
                    } else {
                        Self::bg_load_more(arc);
                    }
                });
            });
        }

        // --- load_previous_rows: backfill a page evicted by scrolling forward -
        {
            let this = Arc::clone(self);
            state.on_load_previous_rows(move || {
                // Coloc-detail rows never populate `displayed_objects`/windowing
                // (no row selection in that view), so there's nothing to
                // backfill there yet.
                if *this.coloc_detail_mode.lock().unwrap() {
                    return;
                }
                let arc = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_load_previous(arc));
            });
        }

        // --- copy_to_clipboard ------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_copy_to_clipboard(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
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

        // --- export_dialog_open -------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_dialog_open(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();

                // Clear each checklist's filter box so a search left over
                // from a previous session doesn't silently hide freshly
                // seeded rows.
                state.set_export_class_search_text(SharedString::new());
                state.set_export_image_search_text(SharedString::new());
                state.set_export_column_search_text(SharedString::new());

                // Seed the dialog's checklist fresh from the known class
                // labels (same source as the table's own Class column
                // filter), all checked by default so exporting everything
                // needs no clicks.
                let labels = model_to_vec(&state.get_filter_class_items());
                let items: Vec<FilterItem> = labels
                    .into_iter()
                    .map(|i| FilterItem {
                        label: i.label,
                        checked: true,
                        group: SharedString::new(),
                        group_header: false,
                        group_all_checked: false,
                    })
                    .collect();
                apply_export_class_checklist(&state, items, "");

                // Same seeding for the image checklist, from the table's own
                // Image column filter labels.
                let image_labels = model_to_vec(&state.get_filter_image_items());
                let image_items: Vec<FilterItem> = image_labels
                    .into_iter()
                    .map(|i| FilterItem {
                        label: i.label,
                        checked: true,
                        group: SharedString::new(),
                        group_header: false,
                        group_all_checked: false,
                    })
                    .collect();
                apply_export_image_checklist(&state, image_items, "");
                state.set_export_each_image(false);

                // Default to the normal table's own columns; switching to
                // "Coloc details" re-seeds this from the coloc-detail specs.
                state.set_export_style("table".into());
                let specs = this.column_specs.lock().unwrap().clone();
                apply_export_column_checklist(&state, column_items_from_specs(&specs), "");
                state.set_export_has_intensity(has_intensity_columns(&specs));
                state.set_export_group_by("none".into());
                // Seed from the project's own saved regex (Project Settings /
                // Plate grouping) instead of leaving it blank, so it matches
                // what "Regex on name" already falls back to elsewhere.
                state.set_export_group_regex(
                    window
                        .global::<crate::ProjectSettingsState>()
                        .get_settings()
                        .custom_regex,
                );
                state.set_export_matrix_kind("plate".into());

                state.set_export_combo_name(SharedString::new());
                state.set_export_format("csv".into());
                state.set_export_batches(slint::ModelRc::new(slint::VecModel::from(Vec::<
                    ExportBatchItem,
                >::new(
                ))));
                state.set_export_status(SharedString::new());
                state.set_export_running(false);
                state.set_export_progress_current(0);
                state.set_export_progress_total(0);
                state.set_export_progress_fraction(0.0);
                state.set_export_dialog_active(true);
            });
        }

        // --- export_dialog_close ------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_dialog_close(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                window
                    .global::<ResultsState>()
                    .set_export_dialog_active(false);
            });
        }

        // --- export_class_item_toggled -------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_class_item_toggled(move |label: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items =
                    toggle_item_by_label(&model_to_vec(&state.get_export_class_items()), &label);
                let search = state.get_export_class_search_text().to_string();
                apply_export_class_checklist(&state, items, &search);
            });
        }

        // --- export_class_select_all / export_class_clear_all --------------------
        {
            let this = Arc::clone(self);
            state.on_export_class_select_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_class_items()), true);
                let search = state.get_export_class_search_text().to_string();
                apply_export_class_checklist(&state, items, &search);
            });
        }
        {
            let this = Arc::clone(self);
            state.on_export_class_clear_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_class_items()), false);
                let search = state.get_export_class_search_text().to_string();
                apply_export_class_checklist(&state, items, &search);
            });
        }

        // --- export_class_search_changed ------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_class_search_changed(move |search: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = model_to_vec(&state.get_export_class_items());
                state.set_export_class_displayed_items(to_model(filter_checklist_by_search(
                    &items,
                    search.as_str(),
                )));
            });
        }

        // --- export_image_item_toggled -------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_image_item_toggled(move |label: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items =
                    toggle_item_by_label(&model_to_vec(&state.get_export_image_items()), &label);
                let search = state.get_export_image_search_text().to_string();
                apply_export_image_checklist(&state, items, &search);
            });
        }

        // --- export_image_select_all / export_image_clear_all --------------------
        {
            let this = Arc::clone(self);
            state.on_export_image_select_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_image_items()), true);
                let search = state.get_export_image_search_text().to_string();
                apply_export_image_checklist(&state, items, &search);
            });
        }
        {
            let this = Arc::clone(self);
            state.on_export_image_clear_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_image_items()), false);
                let search = state.get_export_image_search_text().to_string();
                apply_export_image_checklist(&state, items, &search);
            });
        }

        // --- export_image_search_changed ------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_image_search_changed(move |search: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = model_to_vec(&state.get_export_image_items());
                state.set_export_image_displayed_items(to_model(filter_checklist_by_search(
                    &items,
                    search.as_str(),
                )));
            });
        }

        // --- export_style_selected ------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_style_selected(move |style: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                state.set_export_style(style.clone());
                // The column set is about to change entirely (table vs.
                // coloc-detail columns are unrelated) — a search typed
                // against the old list would otherwise silently hide every
                // freshly seeded row.
                state.set_export_column_search_text(SharedString::new());

                if style.as_str() == "matrix" {
                    // `export_group_by`'s "none"/"image" values (left over from
                    // "table" style) aren't valid choices for Matrix's own
                    // Folder/Regex picker.
                    let group_by = state.get_export_group_by().to_string();
                    if group_by != "folder" && group_by != "regex" {
                        state.set_export_group_by("folder".into());
                    }
                    // `matrix_agg_options`/`matrix_color_scheme_options` are
                    // always populated at startup by `ResultsMatrixController`
                    // (they don't depend on the loaded file), but
                    // `matrix_metric_options` is only populated once the live
                    // Matrix view has actually been opened this session — seed
                    // it here too so the dialog's Value picker isn't empty for
                    // a user who exports without ever visiting that view.
                    if state.get_matrix_metric_options().row_count() == 0 {
                        let specs = this.column_specs.lock().unwrap().clone();
                        let metric_options: Vec<SharedString> = plottable_columns(&specs)
                            .iter()
                            .map(|c| SharedString::from(c.label.as_str()))
                            .collect();
                        if state.get_matrix_metric().is_empty()
                            && let Some(first) = metric_options.first()
                        {
                            state.set_matrix_metric(first.clone());
                        }
                        state.set_matrix_metric_options(slint::ModelRc::new(
                            slint::VecModel::from(metric_options),
                        ));
                    }
                    if state.get_matrix_agg().is_empty() {
                        state.set_matrix_agg("Average".into());
                    }
                }

                if style.as_str() != "coloc_detail" {
                    let specs = this.column_specs.lock().unwrap().clone();
                    apply_export_column_checklist(&state, column_items_from_specs(&specs), "");
                    state.set_export_has_intensity(has_intensity_columns(&specs));
                    return;
                }

                // Coloc-detail's columns depend on which partner classes and
                // channels actually exist in the data, discovered via a DB
                // query — unlike the table's own columns, they can't just be
                // read off already-loaded state, and `coloc_detail_column_specs`
                // is only populated once the user has actually visited the
                // live Coloc Details table view (may still be empty here).
                // Seed from that cache if present (avoids a blank flash),
                // then always refresh from a fresh discovery in the
                // background so column selection works even if the user
                // never opened that view this session.
                let cached = this.coloc_detail_column_specs.lock().unwrap().clone();
                if !cached.is_empty() {
                    apply_export_column_checklist(&state, column_items_from_specs(&cached), "");
                    state.set_export_has_intensity(has_intensity_columns(&cached));
                } else {
                    apply_export_column_checklist(&state, Vec::new(), "");
                    state.set_export_has_intensity(false);
                }
                state.set_export_status("Loading coloc-detail columns...".into());

                let Some(path) = this.path.lock().unwrap().clone() else {
                    return;
                };
                let image_filter = this.image_filter.lock().unwrap().clone();
                let t_stack_filter = *this.t_stack_filter.lock().unwrap();
                let z_stack_filter = *this.z_stack_filter.lock().unwrap();

                let this = Arc::clone(&this);
                std::thread::spawn(move || {
                    let loader = ResultsLoader::new(&path);
                    let filter = DatabaseFilter {
                        image_filter,
                        t_stack_filter,
                        z_stack_filter,
                        ..Default::default()
                    };
                    let specs = match discover_coloc_detail_columns(&loader, &filter) {
                        Ok((channels, coloc_partner_classes)) => {
                            build_coloc_detail_column_specs(&channels, &coloc_partner_classes)
                        }
                        Err(e) => {
                            warn!("coloc-detail column discovery failed: {:?}", e);
                            Vec::new()
                        }
                    };

                    let ui = this.ui.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(window) = ui.upgrade() else { return };
                        let state = window.global::<ResultsState>();
                        // The user may have switched back to "table" while
                        // this discovery was running.
                        if state.get_export_style().as_str() != "coloc_detail" {
                            return;
                        }
                        let search = state.get_export_column_search_text().to_string();
                        apply_export_column_checklist(
                            &state,
                            column_items_from_specs(&specs),
                            &search,
                        );
                        state.set_export_has_intensity(has_intensity_columns(&specs));
                        state.set_export_status(SharedString::new());
                    });
                });
            });
        }

        // --- export_column_item_toggled -------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_column_item_toggled(move |label: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items =
                    toggle_item_by_label(&model_to_vec(&state.get_export_column_items()), &label);
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }

        // --- export_column_select_all / export_column_clear_all -------------------
        {
            let this = Arc::clone(self);
            state.on_export_column_select_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_column_items()), true);
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }
        {
            let this = Arc::clone(self);
            state.on_export_column_clear_all(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = set_all_checked(&model_to_vec(&state.get_export_column_items()), false);
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }

        // --- export_column_search_changed ------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_column_search_changed(move |search: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = model_to_vec(&state.get_export_column_items());
                state.set_export_column_displayed_items(to_model(filter_checklist_by_search(
                    &items,
                    search.as_str(),
                )));
            });
        }

        // --- export_column_group_toggle -------------------------------------------
        // Toggles every column in one section (e.g. "Intensity", "Coloc ClassA") at
        // once, mirroring the main toolbar's own Columns filter popup.
        {
            let this = Arc::clone(self);
            state.on_export_column_group_toggle(move |group: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let current = model_to_vec(&state.get_export_column_items());
                let currently_all_checked =
                    group_all_checked_for_search(&current, group.as_str(), "");
                let items = set_group_checked_for_search(
                    &current,
                    group.as_str(),
                    "",
                    !currently_all_checked,
                );
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }

        // --- export_column_intensity_preset ---------------------------------------
        // Quick picks for the "Intensity" section specifically: clear it, keep only
        // Avg+Sum (the two stats most exports actually want), or select every stat.
        {
            let this = Arc::clone(self);
            state.on_export_column_intensity_preset(move |preset: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = apply_intensity_preset(
                    model_to_vec(&state.get_export_column_items()),
                    preset.as_str(),
                );
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }

        // --- export_column_apply_group_to_all -------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_column_apply_group_to_all(move |group: SharedString| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let items = apply_coloc_group_template(
                    model_to_vec(&state.get_export_column_items()),
                    group.as_str(),
                );
                let search = state.get_export_column_search_text().to_string();
                apply_export_column_checklist(&state, items, &search);
            });
        }

        // --- export_add_batch ----------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_add_batch(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();

                let items = model_to_vec(&state.get_export_class_items());
                let selected: Vec<String> = items
                    .iter()
                    .filter(|i| i.checked)
                    .map(|i| i.label.to_string())
                    .collect();
                if selected.is_empty() {
                    state.set_export_status("Pick at least one class before adding.".into());
                    return;
                }
                let all_selected = selected.len() == items.len();
                let classes_label = if all_selected {
                    "All classes".to_string()
                } else {
                    selected.join(", ")
                };
                let classes: Vec<SharedString> = if all_selected {
                    Vec::new()
                } else {
                    selected.iter().map(SharedString::from).collect()
                };

                let image_items = model_to_vec(&state.get_export_image_items());
                let selected_images: Vec<String> = image_items
                    .iter()
                    .filter(|i| i.checked)
                    .map(|i| i.label.to_string())
                    .collect();
                if selected_images.is_empty() {
                    state.set_export_status("Pick at least one image before adding.".into());
                    return;
                }
                let all_images_selected = selected_images.len() == image_items.len();
                let images_label = if all_images_selected {
                    "All images".to_string()
                } else {
                    selected_images.join(", ")
                };

                let name = {
                    let n = state.get_export_combo_name().to_string();
                    let n = n.trim();
                    if !n.is_empty() {
                        n.to_string()
                    } else if all_selected {
                        "All".to_string()
                    } else {
                        selected.join("_")
                    }
                };

                let style = state.get_export_style().to_string();
                let is_matrix = style == "matrix";

                // Captured now (not read live at export time) so this batch's
                // columns stay correct even if the user later flips
                // style/columns to configure a different combination. Not
                // applicable to Matrix (it always writes a single value
                // column per cell — see `matrix_fields` below).
                let columns: Vec<SharedString> = if is_matrix {
                    Vec::new()
                } else {
                    let column_items = model_to_vec(&state.get_export_column_items());
                    let selected_columns: Vec<String> = column_items
                        .iter()
                        .filter(|i| i.checked)
                        .map(|i| i.label.to_string())
                        .collect();
                    if !column_items.is_empty() && selected_columns.is_empty() {
                        state.set_export_status("Pick at least one column before adding.".into());
                        return;
                    }
                    let all_columns_selected = selected_columns.len() == column_items.len();
                    if all_columns_selected {
                        Vec::new()
                    } else {
                        selected_columns.iter().map(SharedString::from).collect()
                    }
                };

                if is_matrix && state.get_matrix_metric().is_empty() {
                    state.set_export_status("Pick a value to color the matrix by.".into());
                    return;
                }

                let group_by = state.get_export_group_by().to_string();
                let group_regex = state.get_export_group_regex().to_string();
                let matrix_fields = matrix_batch_fields(&state, is_matrix);

                let mut batches = export_batches_to_vec(&state.get_export_batches());
                batches.push(ExportBatchItem {
                    name: name.into(),
                    classes: slint::ModelRc::new(slint::VecModel::from(classes)),
                    classes_label: classes_label.into(),
                    format: state.get_export_format(),
                    images: slint::ModelRc::new(slint::VecModel::from(
                        selected_images
                            .iter()
                            .map(SharedString::from)
                            .collect::<Vec<_>>(),
                    )),
                    images_label: images_label.into(),
                    each_image: !is_matrix && state.get_export_each_image(),
                    style_label: export_style_label(&style, &group_by).into(),
                    style: style.into(),
                    group_by: group_by.into(),
                    group_regex: group_regex.into(),
                    columns: slint::ModelRc::new(slint::VecModel::from(columns)),
                    matrix_metric: matrix_fields.metric.into(),
                    matrix_agg: matrix_fields.agg.into(),
                    matrix_color_scheme: matrix_fields.color_scheme.into(),
                    matrix_range_auto: matrix_fields.range_auto,
                    matrix_range_min: matrix_fields.range_min,
                    matrix_range_max: matrix_fields.range_max,
                    matrix_kind: matrix_fields.kind.into(),
                });
                state.set_export_batches(slint::ModelRc::new(slint::VecModel::from(batches)));
                state.set_export_combo_name(SharedString::new());
                state.set_export_status(SharedString::new());
            });
        }

        // --- export_remove_batch --------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_remove_batch(move |idx: i32| {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                let mut batches = export_batches_to_vec(&state.get_export_batches());
                if idx >= 0 && (idx as usize) < batches.len() {
                    batches.remove(idx as usize);
                }
                state.set_export_batches(slint::ModelRc::new(slint::VecModel::from(batches)));
            });
        }

        // --- export_run_all ---------------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_run_all(move || {
                let Some(window) = this.ui.upgrade() else { return };
                let state = window.global::<ResultsState>();

                let mut batches = export_batches_to_vec(&state.get_export_batches());
                if batches.is_empty() {
                    // Nothing queued — treat the checklist/name/format as one
                    // one-off export so a single export doesn't need the
                    // extra "Add" step.
                    let items = model_to_vec(&state.get_export_class_items());
                    let selected: Vec<String> =
                        items.iter().filter(|i| i.checked).map(|i| i.label.to_string()).collect();
                    if selected.is_empty() {
                        state.set_export_status("Pick at least one class to export.".into());
                        return;
                    }
                    let all_selected = selected.len() == items.len();
                    let classes: Vec<SharedString> = if all_selected {
                        Vec::new()
                    } else {
                        selected.iter().map(SharedString::from).collect()
                    };

                    let image_items = model_to_vec(&state.get_export_image_items());
                    let selected_images: Vec<String> = image_items
                        .iter()
                        .filter(|i| i.checked)
                        .map(|i| i.label.to_string())
                        .collect();
                    if selected_images.is_empty() {
                        state.set_export_status("Pick at least one image to export.".into());
                        return;
                    }

                    let name = {
                        let n = state.get_export_combo_name().to_string();
                        let n = n.trim().to_string();
                        if n.is_empty() { "results".to_string() } else { n }
                    };
                    let style = state.get_export_style().to_string();
                    let is_matrix = style == "matrix";
                    let group_by = state.get_export_group_by().to_string();
                    let group_regex = state.get_export_group_regex().to_string();

                    if is_matrix && state.get_matrix_metric().is_empty() {
                        state.set_export_status("Pick a value to color the matrix by.".into());
                        return;
                    }

                    let columns: Vec<SharedString> = if is_matrix {
                        Vec::new()
                    } else {
                        let column_items = model_to_vec(&state.get_export_column_items());
                        let selected_columns: Vec<String> = column_items
                            .iter()
                            .filter(|i| i.checked)
                            .map(|i| i.label.to_string())
                            .collect();
                        if !column_items.is_empty() && selected_columns.is_empty() {
                            state.set_export_status("Pick at least one column to export.".into());
                            return;
                        }
                        let all_columns_selected = selected_columns.len() == column_items.len();
                        if all_columns_selected {
                            Vec::new()
                        } else {
                            selected_columns.iter().map(SharedString::from).collect()
                        }
                    };
                    let matrix_fields = matrix_batch_fields(&state, is_matrix);

                    batches.push(ExportBatchItem {
                        name: name.into(),
                        classes: slint::ModelRc::new(slint::VecModel::from(classes)),
                        classes_label: SharedString::new(),
                        format: state.get_export_format(),
                        images: slint::ModelRc::new(slint::VecModel::from(
                            selected_images.iter().map(SharedString::from).collect::<Vec<_>>(),
                        )),
                        images_label: SharedString::new(),
                        each_image: !is_matrix && state.get_export_each_image(),
                        style_label: SharedString::new(),
                        style: style.into(),
                        group_by: group_by.into(),
                        group_regex: group_regex.into(),
                        columns: slint::ModelRc::new(slint::VecModel::from(columns)),
                        matrix_metric: matrix_fields.metric.into(),
                        matrix_agg: matrix_fields.agg.into(),
                        matrix_color_scheme: matrix_fields.color_scheme.into(),
                        matrix_range_auto: matrix_fields.range_auto,
                        matrix_range_min: matrix_fields.range_min,
                        matrix_range_max: matrix_fields.range_max,
                        matrix_kind: matrix_fields.kind.into(),
                    });
                }

                let Some(folder) = rfd::FileDialog::new().pick_folder() else { return };

                // `ExportBatchItem`'s array fields are Slint `ModelRc`s, which
                // aren't `Send` — read them out into a plain, thread-safe
                // struct here on the UI thread before handing off.
                let batches: Vec<PlannedExport> = batches
                    .into_iter()
                    .map(|batch| PlannedExport {
                        name: batch.name.to_string(),
                        classes: (0..batch.classes.row_count())
                            .filter_map(|i| batch.classes.row_data(i))
                            .map(|s| s.to_string())
                            .collect(),
                        images: (0..batch.images.row_count())
                            .filter_map(|i| batch.images.row_data(i))
                            .map(|s| s.to_string())
                            .collect(),
                        each_image: batch.each_image,
                        is_xlsx: batch.format.as_str() == "xlsx",
                        is_coloc_detail: batch.style.as_str() == "coloc_detail",
                        is_matrix: batch.style.as_str() == "matrix",
                        group_by: batch.group_by.to_string(),
                        group_regex: batch.group_regex.to_string(),
                        columns: (0..batch.columns.row_count())
                            .filter_map(|i| batch.columns.row_data(i))
                            .map(|s| s.to_string())
                            .collect(),
                        matrix_metric: batch.matrix_metric.to_string(),
                        matrix_agg: batch.matrix_agg.to_string(),
                        matrix_color_scheme: batch.matrix_color_scheme.to_string(),
                        matrix_range_auto: batch.matrix_range_auto,
                        matrix_range_min: batch.matrix_range_min as f64,
                        matrix_range_max: batch.matrix_range_max as f64,
                        matrix_kind: batch.matrix_kind.to_string(),
                    })
                    .collect();

                let Some(path) = this.path.lock().unwrap().clone() else { return };
                let coloc_filter = this.coloc_filter.lock().unwrap().clone();
                let t_stack_filter = *this.t_stack_filter.lock().unwrap();
                let z_stack_filter = *this.z_stack_filter.lock().unwrap();
                // The table-style column specs, reused as the template for
                // every "table" batch below (each batch clones it and applies
                // its own captured column selection on top). Coloc Details
                // batches don't need this — that exporter discovers its own
                // columns and only consults each batch's `columns` labels.
                // Matrix batches only use it to look up the picked metric's
                // `ColumnSpec`.
                let table_specs = this.column_specs.lock().unwrap().clone();
                // Plate grid dimensions — a project-wide setting (not
                // per-batch), read once here the same way `table_specs` is.
                let plate = this.app_state.get_project().plate.clone();
                let plate_rows = plate.plate_rows.max(1) as usize;
                let plate_cols = plate.plate_cols.max(1) as usize;
                let well_rows = plate.well_rows.max(1) as usize;
                let well_cols = plate.well_cols.max(1) as usize;
                let well_image_order = plate.well_image_order.clone();

                let total_files: usize = batches
                    .iter()
                    .map(|b| if b.each_image { b.images.len().max(1) } else { 1 })
                    .sum();
                state.set_export_running(true);
                state.set_export_progress_current(0);
                state.set_export_progress_total(total_files as i32);
                state.set_export_progress_fraction(0.0);
                state.set_export_status(format!("Exporting {total_files} file(s)...").into());

                let cancel_flag = Arc::new(AtomicBool::new(false));
                *this.export_cancel_flag.lock().unwrap() = Some(Arc::clone(&cancel_flag));

                let this = Arc::clone(&this);
                std::thread::spawn(move || {
                    let loader = Arc::new(ResultsLoader::new(&path));
                    let exporter = ResultsExporter::new(loader);
                    let mut used_paths = std::collections::HashSet::new();
                    let mut written = 0usize;
                    let mut completed = 0usize;
                    let mut failures = Vec::new();
                    let mut cancelled = false;

                    'outer: for batch in &batches {
                        let class_filter =
                            if batch.classes.is_empty() { None } else { Some(batch.classes.clone()) };
                        let is_xlsx = batch.is_xlsx;
                        let ext = if is_xlsx { "xlsx" } else { "csv" };

                        // Empty `columns` means "every column" (either the
                        // checklist was never seeded for this batch's style,
                        // or the user left every column checked) — treat
                        // that as "no filter" rather than hiding everything.
                        let visible_labels: Option<HashSet<String>> = if batch.columns.is_empty() {
                            None
                        } else {
                            Some(batch.columns.iter().cloned().collect())
                        };
                        let group = group_config_from_dialog(&batch.group_by, batch.group_regex.clone());
                        let mut base_specs = table_specs.clone();
                        if let Some(labels) = &visible_labels {
                            for spec in base_specs.iter_mut() {
                                spec.visible = labels.contains(&spec.label);
                            }
                        }

                        // One file per checked image (its name folded into
                        // the filename) when `each_image`, otherwise one file
                        // covering every image in `batch.images` together.
                        let per_file: Vec<(String, Option<Vec<String>>)> = if batch.each_image {
                            batch
                                .images
                                .iter()
                                .map(|img| {
                                    (format!("{}_{}", batch.name, image_stem(img)), Some(vec![img.clone()]))
                                })
                                .collect()
                        } else {
                            vec![(batch.name.clone(), Some(batch.images.clone()))]
                        };

                        for (file_label, image_filter) in per_file {
                            if cancel_flag.load(Ordering::Relaxed) {
                                cancelled = true;
                                break 'outer;
                            }

                            let out_path = unique_export_path(&folder, &file_label, ext, &mut used_paths);

                            let filter = DatabaseFilter {
                                image_filter,
                                class_filter: class_filter.clone(),
                                coloc_filter: coloc_filter.clone(),
                                t_stack_filter,
                                z_stack_filter,
                                ..Default::default()
                            };

                            let result = if batch.is_matrix {
                                match table_specs.iter().find(|c| c.label == batch.matrix_metric).cloned() {
                                    Some(metric_spec) => {
                                        let (matrix_group_by, matrix_regex) =
                                            match batch.group_by.as_str() {
                                                "regex" => (GroupBy::Regex, batch.group_regex.clone()),
                                                _ => (GroupBy::Folder, String::new()),
                                            };
                                        let agg = agg_from_label(&batch.matrix_agg);
                                        let scheme = HeatmapColorScheme::from_label(&batch.matrix_color_scheme);
                                        if batch.matrix_kind == "well" {
                                            if is_xlsx {
                                                exporter.export_well_matrices_to_xlsx(
                                                    filter,
                                                    matrix_group_by,
                                                    &matrix_regex,
                                                    agg,
                                                    &metric_spec,
                                                    plate_rows,
                                                    plate_cols,
                                                    well_rows,
                                                    well_cols,
                                                    &well_image_order,
                                                    scheme,
                                                    batch.matrix_range_auto,
                                                    batch.matrix_range_min,
                                                    batch.matrix_range_max,
                                                    &out_path,
                                                )
                                            } else {
                                                exporter.export_well_matrices_to_csv(
                                                    filter,
                                                    matrix_group_by,
                                                    &matrix_regex,
                                                    agg,
                                                    &metric_spec,
                                                    plate_rows,
                                                    plate_cols,
                                                    well_rows,
                                                    well_cols,
                                                    &well_image_order,
                                                    &folder,
                                                    &file_label,
                                                )
                                            }
                                        } else if is_xlsx {
                                            exporter.export_matrix_to_xlsx(
                                                filter,
                                                matrix_group_by,
                                                &matrix_regex,
                                                agg,
                                                &metric_spec,
                                                plate_rows,
                                                plate_cols,
                                                scheme,
                                                batch.matrix_range_auto,
                                                batch.matrix_range_min,
                                                batch.matrix_range_max,
                                                &out_path,
                                            )
                                        } else {
                                            exporter.export_matrix_to_csv(
                                                filter,
                                                matrix_group_by,
                                                &matrix_regex,
                                                agg,
                                                &metric_spec,
                                                plate_rows,
                                                plate_cols,
                                                &out_path,
                                            )
                                        }
                                    }
                                    None => Err(InternalErrors::Io(format!(
                                        "Matrix export: value column '{}' not found",
                                        batch.matrix_metric
                                    ))),
                                }
                            } else {
                                match (batch.is_coloc_detail, is_xlsx) {
                                (true, true) => exporter.export_coloc_detail_to_xlsx(
                                    filter,
                                    visible_labels.as_ref(),
                                    &out_path,
                                ),
                                (true, false) => exporter.export_coloc_detail_to_csv(
                                    filter,
                                    visible_labels.as_ref(),
                                    &out_path,
                                ),
                                (false, true) => {
                                    exporter.export_to_xlsx(filter, &group, &base_specs, &out_path)
                                }
                                (false, false) => {
                                    exporter.export_to_csv(filter, &group, &base_specs, &out_path)
                                }
                                }
                            };
                            match result {
                                Ok(()) => written += 1,
                                Err(e) => {
                                    warn!("Export of '{file_label}' failed: {:?}", e);
                                    failures.push(file_label.clone());
                                }
                            }

                            completed += 1;
                            let ui = this.ui.clone();
                            let progress_label = file_label.clone();
                            let _ = slint::invoke_from_event_loop(move || {
                                let Some(window) = ui.upgrade() else { return };
                                let state = window.global::<ResultsState>();
                                state.set_export_progress_current(completed as i32);
                                state.set_export_progress_fraction(
                                    completed as f32 / total_files.max(1) as f32,
                                );
                                state.set_export_status(
                                    format!("Exporting {completed} of {total_files}: {progress_label}...")
                                        .into(),
                                );
                            });
                        }
                    }

                    *this.export_cancel_flag.lock().unwrap() = None;

                    let status = if cancelled {
                        format!("Cancelled after {written} of {total_files} file(s).")
                    } else if failures.is_empty() {
                        format!("Exported {written} file(s) to {}", folder.display())
                    } else {
                        format!(
                            "Exported {written} of {total_files} file(s) — failed: {}",
                            failures.join(", ")
                        )
                    };

                    let ui = this.ui.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(window) = ui.upgrade() else { return };
                        let state = window.global::<ResultsState>();
                        state.set_export_running(false);
                        state.set_export_status(status.into());
                    });
                });
            });
        }

        // --- export_cancel -----------------------------------------------------
        {
            let this = Arc::clone(self);
            state.on_export_cancel(move || {
                if let Some(flag) = this.export_cancel_flag.lock().unwrap().as_ref() {
                    flag.store(true, Ordering::Relaxed);
                }
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                window
                    .global::<ResultsState>()
                    .set_export_status("Cancelling...".into());
            });
        }

        // --- export_as_displayed -----------------------------------------------
        // Immediate one-shot export matching the results table's own current
        // filters/style/grouping/columns — bypasses the batch queue.
        {
            let this = Arc::clone(self);
            state.on_export_as_displayed(move || {
                let Some(path) = this.path.lock().unwrap().clone() else {
                    return;
                };
                let is_coloc_detail = *this.coloc_detail_mode.lock().unwrap();

                let default_name = if is_coloc_detail {
                    "coloc_detail.csv"
                } else {
                    "results.csv"
                };
                let Some(export_path) = rfd::FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .add_filter("Excel", &["xlsx"])
                    .set_file_name(default_name)
                    .save_file()
                else {
                    return;
                };
                let is_xlsx = export_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("xlsx"));

                let image_filter = this.image_filter.lock().unwrap().clone();
                let class_filter = this.class_filter.lock().unwrap().clone();
                let coloc_filter = this.coloc_filter.lock().unwrap().clone();
                let t_stack_filter = *this.t_stack_filter.lock().unwrap();
                let z_stack_filter = *this.z_stack_filter.lock().unwrap();
                let group = this.group_config.lock().unwrap().clone();
                let base_specs = this.column_specs.lock().unwrap().clone();

                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();
                state.set_export_status("Exporting...".into());

                let this = Arc::clone(&this);
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
                    let result = match (is_coloc_detail, is_xlsx) {
                        (true, true) => {
                            exporter.export_coloc_detail_to_xlsx(filter, None, &export_path)
                        }
                        (true, false) => {
                            exporter.export_coloc_detail_to_csv(filter, None, &export_path)
                        }
                        (false, true) => {
                            exporter.export_to_xlsx(filter, &group, &base_specs, &export_path)
                        }
                        (false, false) => {
                            exporter.export_to_csv(filter, &group, &base_specs, &export_path)
                        }
                    };
                    let status = match result {
                        Ok(()) => format!("Exported to {}", export_path.display()),
                        Err(e) => {
                            warn!("export_as_displayed failed: {:?}", e);
                            "Export failed — see logs.".to_string()
                        }
                    };
                    let ui = this.ui.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(window) = ui.upgrade() else { return };
                        window
                            .global::<ResultsState>()
                            .set_export_status(status.into());
                    });
                });
            });
        }

        // --- export_add_batch_from_table ----------------------------------------
        // Queues a combination captured from the results table's own current
        // filters/style/grouping/columns, reusing the Name/Format fields
        // (the same ones "+ Add" reads) since those aren't part of "the
        // table's own state" the way filters/grouping/columns are.
        {
            let this = Arc::clone(self);
            state.on_export_add_batch_from_table(move || {
                let Some(window) = this.ui.upgrade() else {
                    return;
                };
                let state = window.global::<ResultsState>();

                let is_coloc_detail = *this.coloc_detail_mode.lock().unwrap();
                let group_cfg = this.group_config.lock().unwrap().clone();
                let (group_by, group_regex) =
                    match live_group_by_for_batch(is_coloc_detail, &group_cfg) {
                        Ok(pair) => pair,
                        Err(msg) => {
                            state.set_export_status(msg.into());
                            return;
                        }
                    };
                let style = if is_coloc_detail {
                    "coloc_detail"
                } else {
                    "table"
                };

                let class_filter = this.class_filter.lock().unwrap().clone();
                let classes: Vec<SharedString> = class_filter
                    .unwrap_or_default()
                    .iter()
                    .map(SharedString::from)
                    .collect();
                let classes_label = if classes.is_empty() {
                    "As displayed".to_string()
                } else {
                    classes
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                // Unlike `classes`, `images` must always be a concrete list
                // (never empty) — if the live table has no image filter set,
                // fall back to every known image name.
                let image_filter = this.image_filter.lock().unwrap().clone();
                let all_known_images: Vec<String> = model_to_vec(&state.get_filter_image_items())
                    .iter()
                    .map(|i| i.label.to_string())
                    .collect();
                let images: Vec<SharedString> =
                    images_from_live_filter(image_filter, &all_known_images)
                        .iter()
                        .map(SharedString::from)
                        .collect();
                let images_label = "As displayed".to_string();

                let specs = if is_coloc_detail {
                    this.coloc_detail_column_specs.lock().unwrap().clone()
                } else {
                    this.column_specs.lock().unwrap().clone()
                };
                let all_visible = specs.iter().all(|c| c.visible);
                let columns: Vec<SharedString> = if all_visible {
                    Vec::new()
                } else {
                    specs
                        .iter()
                        .filter(|c| c.visible)
                        .map(|c| SharedString::from(c.label.as_str()))
                        .collect()
                };

                let name = {
                    let n = state.get_export_combo_name().to_string();
                    let n = n.trim();
                    if !n.is_empty() {
                        n.to_string()
                    } else {
                        "AsDisplayed".to_string()
                    }
                };

                let mut batches = export_batches_to_vec(&state.get_export_batches());
                batches.push(ExportBatchItem {
                    name: name.into(),
                    classes: slint::ModelRc::new(slint::VecModel::from(classes)),
                    classes_label: classes_label.into(),
                    format: state.get_export_format(),
                    images: slint::ModelRc::new(slint::VecModel::from(images)),
                    images_label: images_label.into(),
                    each_image: false,
                    style_label: export_style_label(style, &group_by).into(),
                    style: style.into(),
                    group_by: group_by.into(),
                    group_regex: group_regex.into(),
                    columns: slint::ModelRc::new(slint::VecModel::from(columns)),
                    // Matrix export isn't reachable via this path — the "Add
                    // from table" button is disabled while `export_style ==
                    // "matrix"` (its live filters/grouping/columns don't map
                    // onto Matrix's Value/Aggregate/Folder-or-Regex picks).
                    matrix_metric: SharedString::new(),
                    matrix_agg: SharedString::new(),
                    matrix_color_scheme: SharedString::new(),
                    matrix_range_auto: true,
                    matrix_range_min: 0.0,
                    matrix_range_max: 1.0,
                    matrix_kind: "plate".into(),
                });
                state.set_export_batches(slint::ModelRc::new(slint::VecModel::from(batches)));
                state.set_export_combo_name(SharedString::new());
                state.set_export_status(SharedString::new());
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
        *self.coloc_detail_mode.lock().unwrap() = false;
        self.coloc_detail_column_specs.lock().unwrap().clear();
        self.coloc_detail_page_rows.lock().unwrap().reset();
        self.displayed_objects.lock().unwrap().reset();

        let ui = self.ui.clone();
        let app_ui = self.app_state.ui_handle.clone();
        let channels_arc = Arc::clone(&self.channels);
        let all_loaded_arc = Arc::clone(&self.all_loaded);
        let column_specs_arc = Arc::clone(&self.column_specs);
        let column_widths_arc = Arc::clone(&self.column_widths);
        let displayed_objects_arc = Arc::clone(&self.displayed_objects);

        std::thread::spawn(move || {
            let loader = ResultsLoader::new(&path);

            let first_page = loader.get_objects(DatabaseFilter {
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
                (Ok(objects), Ok(img_names), Ok(cls_names), Ok(coloc_partner_classes)) => {
                    let channels = discover_channels(&objects);
                    let specs = build_column_specs(&channels, &coloc_partner_classes);
                    let all_loaded = objects.len() < PAGE_SIZE;

                    *channels_arc.lock().unwrap() = channels;
                    *all_loaded_arc.lock().unwrap() = all_loaded;
                    *column_specs_arc.lock().unwrap() = specs.clone();
                    {
                        let mut window = displayed_objects_arc.lock().unwrap();
                        window.reset();
                        window.note_appended(0, &objects);
                    }

                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app_ui) = app_ui.upgrade() {
                            app_ui.global::<ResultsListState>().set_is_loading(false);
                        }
                        if let Some(window) = ui.upgrade() {
                            let state = window.global::<ResultsState>();

                            let slint_rows: Vec<ResultsRow> = objects
                                .iter()
                                .enumerate()
                                .map(|(i, r)| to_slint_row(to_display_row(i, r, &specs)))
                                .collect();

                            let widths = column_widths_arc.lock().unwrap().clone();
                            let slint_cols: Vec<ResultsColumnDef> =
                                specs_to_slint_cols(&specs, &widths);
                            let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
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

                            state.set_columns(slint::ModelRc::new(slint::VecModel::from(
                                slint_cols,
                            )));
                            state.set_visible_column_count(visible_count);
                            state.set_columns_total_width(total_width);
                            state.set_column_items(slint::ModelRc::new(slint::VecModel::from(
                                column_items.clone(),
                            )));
                            state.set_column_popup(slint::ModelRc::new(slint::VecModel::from(
                                column_items,
                            )));
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

                            let mut matrix_class_options =
                                vec![SharedString::from("All classes")];
                            matrix_class_options
                                .extend(cls_names.iter().map(SharedString::from));
                            state.set_matrix_class_options(slint::ModelRc::new(
                                slint::VecModel::from(matrix_class_options),
                            ));

                            // "No" / "Yes (any class)" are always offered; the rest are the
                            // partner classes actually present in this file's coloc_json data,
                            // letting the user filter for e.g. "Colocalizes with Nucleus".
                            let mut coloc_labels = vec![
                                coloc_filter_label_no().to_string(),
                                coloc_filter_label_any().to_string(),
                            ];
                            coloc_labels.extend(
                                coloc_partner_classes
                                    .iter()
                                    .map(|c| coloc_filter_label_with(c)),
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
                            state.set_chart_heatmap_color_scheme("Viridis".into());
                            state.set_chart_heatmap_range_auto(true);
                            state.set_chart_heatmap_range_min(0.0);
                            state.set_chart_heatmap_range_max(1.0);
                            state.set_chart_plottable_columns(slint::ModelRc::new(
                                slint::VecModel::from(plottable_column_labels(&specs)),
                            ));
                            state.set_chart_heatmap_metric_options(slint::ModelRc::new(
                                slint::VecModel::from(heatmap_metric_options(&specs)),
                            ));
                            state.set_chart_heatmap_color_scheme_options(slint::ModelRc::new(
                                slint::VecModel::from(heatmap_color_scheme_options()),
                            ));
                            // Matrix view's "Value" picker - same visible-numeric-
                            // columns source as the heatmap's, refreshed at the
                            // same time so it's populated before the user ever
                            // opens Matrix view.
                            state.set_matrix_metric_options(slint::ModelRc::new(
                                slint::VecModel::from(plottable_column_labels(&specs)),
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
                            state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
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
    // Reload dispatch: grouped (aggregated) vs. paginated per-object view
    // -------------------------------------------------------------------------

    /// Spawns the appropriate background reload based on the active view:
    /// coloc-detail flat table, grouped/aggregated, or plain paginated per-object.
    /// Checking `coloc_detail_mode` first means row filters (image/class/coloc)
    /// applied while the coloc-detail view is active reload *that* view instead
    /// of silently falling back to the normal table.
    pub(crate) fn spawn_reload(this: Arc<Self>) {
        let coloc_detail = *this.coloc_detail_mode.lock().unwrap();
        let grouped = !matches!(this.group_config.lock().unwrap().group_by, GroupBy::None);
        std::thread::spawn(move || {
            if coloc_detail {
                Self::bg_reload_coloc_detail_page0(this);
            } else if grouped {
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
                    let state = w.global::<ResultsState>();
                    state.set_loading_more(false);
                    state.set_group_computing(false);
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
        // Per-object specs carry the column-visibility selection; only visible
        // metrics become grouped columns.
        let base_specs = this.column_specs.lock().unwrap().clone();
        let sort_column = this.sort_column.lock().unwrap().clone();
        let sort_ascending = *this.sort_ascending.lock().unwrap();

        let loader = ResultsLoader::new(&path);
        // Aggregation is computed directly in DuckDB (see `aggregate_objects_sql`)
        // instead of fetching every matching row into Rust first.
        match aggregate_objects_sql(
            &loader,
            DatabaseFilter {
                object_id_filter: None,
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
            },
            &config,
            &base_specs,
        ) {
            Ok((specs, mut display_rows)) => {
                // The grouped view is fully materialized in memory (unlike the
                // paginated per-object view), so sorting it is a plain in-memory
                // sort rather than another DB round-trip.
                if let Some(col) = &sort_column {
                    sort_display_rows(&mut display_rows, &specs, col, sort_ascending);
                }
                // Grouped view is never paginated.
                *this.all_loaded.lock().unwrap() = true;
                // Grouped rows aggregate many ROIs, so there is no single source
                // object to open/highlight when one is selected.
                this.displayed_objects.lock().unwrap().reset();
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

                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
                        state.set_visible_column_count(visible_count);
                        state.set_columns_total_width(total_width);
                        state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
                        state.set_all_rows_loaded(true);
                        state.set_at_top_loaded(true);
                        state.set_loading_more(false);
                        state.set_group_active(true);
                        state.set_group_computing(false);
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

    /// Loads page 0 of the colocalization detail flat table. Unlike the old
    /// single-shot implementation, this never loads more than one page of
    /// source ROIs, nor more partner ROIs than that page actually references
    /// (via `DatabaseFilter::object_id_filter` — see `coloc_partner_ids`),
    /// instead of every object in the image regardless of relevance.
    fn bg_reload_coloc_detail_page0(this: Arc<Self>) {
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
        let base_filter = DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            t_stack_filter,
            z_stack_filter,
            ..Default::default()
        };
        let (channels, coloc_partner_classes) =
            match discover_coloc_detail_columns(&loader, &base_filter) {
                Ok(cols) => cols,
                Err(e) => {
                    warn!(
                        "bg_reload_coloc_detail_page0 (column discovery) failed: {:?}",
                        e
                    );
                    finish(ui);
                    return;
                }
            };

        let mut specs = build_coloc_detail_column_specs(&channels, &coloc_partner_classes);
        // Carry over the visibility the user already chose: on the very
        // first entry into coloc-detail mode this session, inherit from
        // the normal per-object column filter (columns like object_id/area_px/
        // circularity/ch* share the same ids in both specs); after that,
        // prefer whatever was chosen while already in coloc-detail mode
        // (e.g. via `column_filter_apply`), so a filter set here isn't
        // silently discarded on the next reload.
        {
            let prev = this.coloc_detail_column_specs.lock().unwrap();
            if prev.is_empty() {
                let base = this.column_specs.lock().unwrap();
                carry_over_visibility(&mut specs, &base);
            } else {
                carry_over_visibility(&mut specs, &prev);
            }
        }
        *this.coloc_detail_column_specs.lock().unwrap() = specs.clone();

        match loader.get_objects(DatabaseFilter {
            page_size: PAGE_SIZE,
            page: 0,
            needs_intensities: true,
            ..base_filter.clone()
        }) {
            Ok(source_page) => {
                let partner_page = match Self::fetch_coloc_partners(&loader, &source_page) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!(
                            "bg_reload_coloc_detail_page0 (partner fetch) failed: {:?}",
                            e
                        );
                        finish(ui);
                        return;
                    }
                };

                let all_loaded = source_page.len() < PAGE_SIZE;
                let display_rows = flatten_coloc_rows(&source_page, &partner_page, &specs);

                // Row selection doesn't apply to the flat coloc view.
                this.displayed_objects.lock().unwrap().reset();
                *this.all_loaded.lock().unwrap() = all_loaded;
                {
                    let mut page_rows = this.coloc_detail_page_rows.lock().unwrap();
                    page_rows.reset();
                    page_rows.note_appended(0, display_rows.len());
                }

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

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let slint_rows: Vec<ResultsRow> =
                            display_rows.into_iter().map(to_slint_row).collect();
                        let slint_cols = specs_to_slint_cols(&specs, &widths);
                        let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
                        let total_width: f32 = slint_cols
                            .iter()
                            .filter(|c| c.visible)
                            .map(|c| c.width)
                            .sum();

                        state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
                        state.set_visible_column_count(visible_count);
                        state.set_columns_total_width(total_width);
                        state.set_rows(slint::ModelRc::new(slint::VecModel::from(slint_rows)));
                        state.set_column_popup_all_checked(all_checked(&column_items));
                        state.set_column_items(slint::ModelRc::new(slint::VecModel::from(
                            column_items.clone(),
                        )));
                        state.set_column_popup(slint::ModelRc::new(slint::VecModel::from(
                            column_items,
                        )));
                        state.set_all_rows_loaded(all_loaded);
                        state.set_at_top_loaded(true);
                        state.set_loading_more(false);
                        state.set_group_computing(false);
                        state.set_group_active(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_reload_coloc_detail_page0 failed: {:?}", e);
                finish(ui);
            }
        }
    }

    /// Fetches exactly the colocalization partner ROIs referenced by
    /// `source_page`'s `coloc_json`, instead of every object in the image —
    /// `object_id_filter` alone, with no other filter fields, since an
    /// explicit id list is already unambiguous (adding the source-side
    /// image/class/coloc filters back on top would risk incorrectly
    /// excluding a partner that doesn't itself match those filters).
    fn fetch_coloc_partners(
        loader: &ResultsLoader,
        source_page: &[ObjectRow],
    ) -> Result<Vec<ObjectRow>, InternalErrors> {
        let ids = coloc_partner_ids(source_page);
        if ids.is_empty() {
            return Ok(vec![]);
        }
        loader.get_objects(DatabaseFilter {
            object_id_filter: Some(ids),
            page_size: 0,
            needs_intensities: true,
            ..Default::default()
        })
    }

    /// Appends the next page of the colocalization detail flat table.
    fn bg_load_more_coloc_detail(this: Arc<Self>) {
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
        // Column set was already fixed by page 0 — reused as-is so columns
        // never shift under the user while they scroll/load more.
        let specs = this.coloc_detail_column_specs.lock().unwrap().clone();
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
        match loader.get_objects(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            object_id_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: next_page,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(source_page) => {
                let partner_page = match Self::fetch_coloc_partners(&loader, &source_page) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("bg_load_more_coloc_detail (partner fetch) failed: {:?}", e);
                        let mut p = this.current_page.lock().unwrap();
                        if *p > 0 {
                            *p -= 1;
                        }
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(w) = ui.upgrade() {
                                w.global::<ResultsState>().set_loading_more(false);
                            }
                        });
                        return;
                    }
                };

                let all_loaded = source_page.len() < PAGE_SIZE;
                let display_rows = flatten_coloc_rows(&source_page, &partner_page, &specs);
                *this.all_loaded.lock().unwrap() = all_loaded;
                let evict = this
                    .coloc_detail_page_rows
                    .lock()
                    .unwrap()
                    .note_appended(next_page, display_rows.len());

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let model = state.get_rows();
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<ResultsRow>>()
                        {
                            // Offset each flattened row's positional id past the
                            // rows already loaded, so ids stay unique across
                            // pages (flatten_coloc_rows numbers each page from 1).
                            let base = vec_model.row_count() as i32;
                            for mut row in display_rows {
                                row.object_id_int += base;
                                vec_model.push(to_slint_row(row));
                            }
                            if let Some(EvictEdge {
                                from_front: true,
                                row_count,
                            }) = evict
                            {
                                for _ in 0..row_count {
                                    vec_model.remove(0);
                                }
                                state.set_at_top_loaded(false);
                                state.set_front_row_delta(row_count as i32);
                            }
                        }
                        state.set_all_rows_loaded(all_loaded);
                        state.set_loading_more(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_load_more_coloc_detail failed: {:?}", e);
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
        match loader.get_objects(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            object_id_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: 0,
            needs_intensities,
            sort_column,
            sort_ascending,
        }) {
            Ok(objects) => {
                let all_loaded = objects.len() < PAGE_SIZE;
                *this.all_loaded.lock().unwrap() = all_loaded;
                {
                    let mut window = this.displayed_objects.lock().unwrap();
                    window.reset();
                    window.note_appended(0, &objects);
                }
                let widths = this.column_widths.lock().unwrap().clone();

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let slint_rows: Vec<ResultsRow> = objects
                            .iter()
                            .enumerate()
                            .map(|(i, r)| to_slint_row(to_display_row(i, r, &specs)))
                            .collect();
                        let state = window.global::<ResultsState>();
                        // A full reload always starts fresh at the top.
                        state.set_at_top_loaded(true);
                        // Restore the per-object columns (grouped mode may have replaced them).
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
        match loader.get_objects(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            object_id_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: next_page,
            needs_intensities,
            sort_column,
            sort_ascending,
        }) {
            Ok(new_objects) => {
                let all_loaded = new_objects.len() < PAGE_SIZE;
                *this.all_loaded.lock().unwrap() = all_loaded;
                // Position is derived from the page number, not the model's
                // current length — required so ids stay unique/stable once
                // eviction can remove rows from the front (see `RowWindow`).
                let base = next_page * PAGE_SIZE;
                let evict = this
                    .displayed_objects
                    .lock()
                    .unwrap()
                    .note_appended(next_page, &new_objects);

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let model = state.get_rows();
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<ResultsRow>>()
                        {
                            // Append first, then evict — the model never
                            // observes rows disappearing before they
                            // conceptually existed.
                            for (i, object) in new_objects.iter().enumerate() {
                                vec_model.push(to_slint_row(to_display_row(
                                    base + i,
                                    object,
                                    &specs,
                                )));
                            }
                            if let Some(EvictEdge {
                                from_front: true,
                                row_count,
                            }) = evict
                            {
                                for _ in 0..row_count {
                                    vec_model.remove(0);
                                }
                                state.set_at_top_loaded(false);
                                state.set_front_row_delta(row_count as i32);
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
    // Background: prepend the page before the currently-loaded window
    // -------------------------------------------------------------------------

    /// Mirrors `bg_load_more`, for the opposite (top) scroll edge: fetches the
    /// page immediately before the loaded window's oldest page (backfilling
    /// one that scrolling forward previously evicted) and prepends it.
    fn bg_load_previous(this: Arc<Self>) {
        let Some(path) = this.path.lock().unwrap().clone() else {
            return;
        };

        let Some(oldest) = this.displayed_objects.lock().unwrap().oldest_loaded_page() else {
            return;
        };
        if oldest == 0 {
            // Nothing earlier to backfill — the window already reaches page 0.
            let ui = this.ui.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = ui.upgrade() {
                    w.global::<ResultsState>().set_at_top_loaded(true);
                }
            });
            return;
        }
        let prev_page = oldest - 1;

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
        match loader.get_objects(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            object_id_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: PAGE_SIZE,
            page: prev_page,
            needs_intensities,
            sort_column,
            sort_ascending,
        }) {
            Ok(prev_objects) => {
                let base = prev_page * PAGE_SIZE;
                let (evict, at_top) = {
                    let mut window = this.displayed_objects.lock().unwrap();
                    let evict = window.note_prepended(prev_page, &prev_objects);
                    (evict, window.oldest_loaded_page() == Some(0))
                };

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(window) = ui.upgrade() {
                        let state = window.global::<ResultsState>();
                        let model = state.get_rows();
                        if let Some(vec_model) =
                            model.as_any().downcast_ref::<slint::VecModel<ResultsRow>>()
                        {
                            // Insert in reverse so the final order is correct.
                            for (i, object) in prev_objects.iter().enumerate().rev() {
                                vec_model.insert(
                                    0,
                                    to_slint_row(to_display_row(base + i, object, &specs)),
                                );
                            }
                            if let Some(EvictEdge {
                                from_front: false,
                                row_count,
                            }) = evict
                            {
                                // Tail eviction needs no scroll compensation —
                                // removing from the end doesn't shift anything
                                // above the current viewport.
                                for _ in 0..row_count {
                                    vec_model.remove(vec_model.row_count() - 1);
                                }
                            }
                        }
                        state.set_front_row_delta(-(prev_objects.len() as i32));
                        state.set_at_top_loaded(at_top);
                        state.set_loading_more(false);
                    }
                });
            }
            Err(e) => {
                warn!("bg_load_previous failed: {:?}", e);
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

        let Some(window) = self.ui.upgrade() else {
            return;
        };
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

        let Some(window) = self.ui.upgrade() else {
            return;
        };
        window.global::<ResultsState>().set_loading_more(true);

        Self::spawn_reload(Arc::clone(self));
    }

    fn on_z_stack_changed(self: &Arc<Self>, value: i32, show_all: bool) {
        *self.z_stack_filter.lock().unwrap() = if show_all { None } else { Some(value) };
        *self.current_page.lock().unwrap() = 0;
        *self.all_loaded.lock().unwrap() = false;

        let Some(window) = self.ui.upgrade() else {
            return;
        };
        window.global::<ResultsState>().set_loading_more(true);

        Self::spawn_reload(Arc::clone(self));
    }

    // -------------------------------------------------------------------------
    // Row selection: open the object's image and highlight its bounding box
    // -------------------------------------------------------------------------

    /// A per-object row was selected. Looks the source object up by its stable
    /// `object_id` (not a positional array index — the loaded window can
    /// evict pages, so a position would silently go stale), then opens its
    /// source image in the editor and paints the object's bounding box.
    /// Grouped/aggregated/coloc-detail rows have no source object (`object_id`
    /// is `""`), so the lookup misses and the selection is ignored.
    fn on_object_row_selected(&self, object_id: SharedString) {
        if object_id.is_empty() {
            return;
        }
        let object = {
            let window = self.displayed_objects.lock().unwrap();
            match window.get(object_id.as_str()) {
                Some(object) => object.clone(),
                None => return,
            }
        };
        if object.image_rel_path.is_empty() {
            warn!("Selected object has no image path; cannot open it");
            return;
        }
        let rel_path = PathBuf::from(&object.image_rel_path);
        self.image_list_controller
            .open_image_and_highlight_object(&rel_path, object.bbox_px);
    }

    // -------------------------------------------------------------------------
    // Image filter popup management
    // -------------------------------------------------------------------------

    fn toggle_image_label(&self, label: SharedString) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_image_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_image_all_popup_checked(all_checked(&popup));
        state.set_filter_image_popup(to_model(popup));
    }

    fn image_select_all(&self) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_class_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_class_all_popup_checked(all_checked(&popup));
        state.set_filter_class_popup(to_model(popup));
    }

    fn class_select_all(&self) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_filter_coloc_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_filter_coloc_all_popup_checked(all_checked(&popup));
        state.set_filter_coloc_popup(to_model(popup));
    }

    fn coloc_select_all(&self) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();
        let current = model_to_vec(&state.get_column_items());
        let popup = filter_popup_by_search(&current, search.as_str());
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    fn column_select_all(&self) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();
        let search = self.column_search.lock().unwrap().clone();
        let current = model_to_vec(&state.get_column_items());
        let currently_all_checked = group_all_checked_for_search(&current, group.as_str(), &search);
        let items =
            set_group_checked_for_search(&current, group.as_str(), &search, !currently_all_checked);
        let popup = filter_popup_by_search(&items, &search);
        state.set_column_popup_all_checked(all_checked(&popup));
        state.set_column_items(to_model(mark_group_headers(items)));
        state.set_column_popup(to_model(mark_group_headers(popup)));
    }

    /// Applies column-visibility selection: updates the stored `column_specs`
    /// (or, while the coloc-detail view is active, `coloc_detail_column_specs`)
    /// and refreshes the view. In grouped or coloc-detail mode this reloads
    /// from the database (grouped re-aggregates so only visible metrics appear
    /// as grouped columns; coloc-detail re-flattens so the header/row mismatch
    /// bug — where a column-filter change silently reverted the view back to
    /// the normal table — can't happen, since the reload itself stays on the
    /// coloc-detail path). Otherwise it updates `ResultsState.columns` /
    /// `visible_column_count` directly.
    fn column_filter_apply(self: &Arc<Self>) {
        let Some(window) = self.ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();

        // Build label→checked map from the authoritative column_items list.
        let items = model_to_vec(&state.get_column_items());
        let visibility: BTreeMap<String, bool> = items
            .iter()
            .map(|i| (i.label.to_string(), i.checked))
            .collect();

        let coloc_detail = *self.coloc_detail_mode.lock().unwrap();

        // Update the stored column specs (used by the next reload and to decide
        // which channel data to fetch).
        let target = if coloc_detail {
            &self.coloc_detail_column_specs
        } else {
            &self.column_specs
        };
        {
            let mut specs = target.lock().unwrap();
            for spec in specs.iter_mut() {
                if let Some(&visible) = visibility.get(&spec.label) {
                    spec.visible = visible;
                }
            }
        }

        if coloc_detail {
            // Re-flatten so the coloc-detail rows/columns reload together and
            // stay in sync — the view never falls back to the normal table.
            state.set_loading_more(true);
            Self::spawn_reload(Arc::clone(self));
            return;
        }

        let specs = self.column_specs.lock().unwrap().clone();
        state.set_chart_plottable_columns(slint::ModelRc::new(slint::VecModel::from(
            plottable_column_labels(&specs),
        )));
        state.set_chart_heatmap_metric_options(slint::ModelRc::new(slint::VecModel::from(
            heatmap_metric_options(&specs),
        )));
        state.set_matrix_metric_options(slint::ModelRc::new(slint::VecModel::from(
            plottable_column_labels(&specs),
        )));

        let grouped = !matches!(self.group_config.lock().unwrap().group_by, GroupBy::None);
        if grouped {
            // Re-aggregate so the grouped columns track the visible metrics.
            state.set_loading_more(true);
            Self::spawn_reload(Arc::clone(self));
        } else {
            let widths = self.column_widths.lock().unwrap().clone();
            let slint_cols = specs_to_slint_cols(&specs, &widths);
            let visible_count = specs.iter().filter(|c| c.visible).count() as i32;
            let total_width: f32 = slint_cols
                .iter()
                .filter(|c| c.visible)
                .map(|c| c.width)
                .sum();
            state.set_columns(slint::ModelRc::new(slint::VecModel::from(slint_cols)));
            state.set_visible_column_count(visible_count);
            state.set_columns_total_width(total_width);
        }
    }

    /// Live-updates one column's width as the user drags its header's resize handle.
    /// Operates directly on whatever `ResultsState.columns` currently holds — the per-object
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

        let Some(window) = self.ui.upgrade() else {
            return;
        };
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
        self.last_chart.lock().unwrap().replace(LastChart {
            chart: chart.clone(),
            kind,
        });
    }

    /// Fetches every object matching the current filters (mirrors
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
                    let mut buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(
                        chart.width,
                        chart.height,
                    );
                    {
                        let slice = buf.make_mut_slice();
                        for (px, rgb) in slice.iter_mut().zip(chart.rgb.chunks_exact(3)) {
                            *px = slint::Rgb8Pixel {
                                r: rgb[0],
                                g: rgb[1],
                                b: rgb[2],
                            };
                        }
                    }
                    state.set_chart_image(slint::Image::from_rgb8(buf));
                    // Keep the Min/Max fields synced to whatever range the
                    // heatmap actually rendered with - in Auto mode this
                    // shows the freshly computed range instead of a stale
                    // one; in Manual mode it's just their own input echoed
                    // back, so this is a no-op either way.
                    if let Some((min, max)) = chart.heatmap_range {
                        state.set_chart_heatmap_range_min(min as f32);
                        state.set_chart_heatmap_range_max(max as f32);
                    }
                }
                state.set_chart_status(status.into());
            });
        };

        let Some(path) = this.path.lock().unwrap().clone() else {
            report("No results file loaded.".into(), None);
            return;
        };

        // Per-object specs carry the visible-column selection; chart axis picks
        // are resolved against them (label -> column id) below.
        let specs = this.column_specs.lock().unwrap().clone();
        let resolve_id = |label: &str| {
            specs
                .iter()
                .find(|c| c.label == label)
                .map(|c| c.id.clone())
        };

        let image_filter = this.image_filter.lock().unwrap().clone();
        let class_filter = this.class_filter.lock().unwrap().clone();
        let coloc_filter = this.coloc_filter.lock().unwrap().clone();
        let t_stack_filter = *this.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.z_stack_filter.lock().unwrap();

        let loader = ResultsLoader::new(&path);
        // A chart summarizes every matching object, not just a loaded page.
        let objects = match loader.get_objects(DatabaseFilter {
            image_filter,
            class_filter,
            coloc_filter,
            object_id_filter: None,
            t_stack_filter,
            z_stack_filter,
            page_size: 0,
            page: 0,
            needs_intensities: true,
            sort_column: None,
            sort_ascending: true,
        }) {
            Ok(objects) => objects,
            Err(e) => {
                warn!("bg_render_chart failed to load ROIs: {:?}", e);
                report("Failed to load data for the chart.".into(), None);
                return;
            }
        };

        if objects.is_empty() {
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
                    &objects,
                    &col_id,
                    &specs,
                    config.bucket_count.max(1),
                    config.log_scale,
                    config.color_by,
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
                let Some(data) = compute_scatter(
                    &objects,
                    &x_id,
                    &y_id,
                    config.color_by,
                    &specs,
                    SCATTER_MAX_POINTS,
                ) else {
                    report("No numeric data for these columns.".into(), None);
                    return;
                };
                let status = match data.sampled_from {
                    Some(total) => {
                        format!(
                            "Showing {} of {} points (sampled).",
                            data.points.len(),
                            total
                        )
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
                let image_count = objects
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
                let Some(data) = compute_heatmap(&objects, &metric, &specs, config.cell_size_px)
                else {
                    report("No data to bin into a heatmap.".into(), None);
                    return;
                };
                match render_heatmap(
                    &data,
                    config.heatmap_color_scheme,
                    config.heatmap_range,
                    config.render_width,
                    config.render_height,
                ) {
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
        object_id_int: row.object_id_int,
        object_id: row.object_id.into(),
        values: slint::ModelRc::new(slint::VecModel::from(values)),
        stripe: row.stripe,
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

/// Labels for the heatmap's "Colors" scheme picker, in `HeatmapColorScheme::all()` order.
fn heatmap_color_scheme_options() -> Vec<SharedString> {
    HeatmapColorScheme::all()
        .iter()
        .map(|s| SharedString::from(s.label()))
        .collect()
}

/// Seeds the export dialog's "Columns to export" checklist from a column
/// spec list — checked state mirrors each spec's current table visibility,
/// so by default the export matches what's shown, with room to adjust.
/// Grouped (via `export_column_group`/`mark_group_headers`) the same way the
/// main toolbar's own Columns filter popup groups channel/coloc columns, so
/// e.g. every "Coloc ClassA" or "Intensity" column can be toggled as one
/// section instead of one at a time.
fn has_intensity_columns(specs: &[ColumnSpec]) -> bool {
    specs.iter().any(|c| c.id.starts_with("ch"))
}

fn column_items_from_specs(specs: &[ColumnSpec]) -> Vec<FilterItem> {
    let items: Vec<FilterItem> = specs
        .iter()
        .map(|c| FilterItem {
            label: c.label.as_str().into(),
            checked: c.visible,
            group: export_column_group(&c.id).into(),
            group_header: false,
            group_all_checked: false,
        })
        .collect();
    mark_group_headers(items)
}

/// Section a column belongs to in the export dialog's "Columns to export"
/// checklist, or `""` for columns that don't belong to a section.
///
/// Unlike the main toolbar's `column_group` (which lumps every coloc-partner
/// column together under one "Colocalization" header), each partner class
/// gets its own group here — e.g. "Coloc ClassA", "Coloc ClassB" — since the
/// export dialog can show many partner classes at once and the user wants to
/// pick columns per class, not just colocalization-columns-as-a-whole.
fn export_column_group(col_id: &str) -> String {
    if let Some(rest) = col_id.strip_prefix("coloc_partner__") {
        return rest
            .rsplit_once("__")
            .map_or(String::new(), |(class, _)| format!("Coloc {class}"));
    }
    if let Some(rest) = col_id.strip_prefix("coloc_detail__") {
        return rest
            .rsplit_once("__")
            .map_or(String::new(), |(class, _)| format!("Coloc {class}"));
    }
    if col_id.starts_with("ch") {
        return "Intensity".to_string();
    }
    String::new()
}

/// Applies one of the Intensity section's quick-pick presets: `"none"`
/// unchecks every Intensity column, `"avg_sum"` checks only the Avg/Sum
/// stats (Min/Max off), anything else (`"all"`) checks every Intensity
/// column. Columns outside the Intensity group are left untouched.
fn apply_intensity_preset(items: Vec<FilterItem>, preset: &str) -> Vec<FilterItem> {
    items
        .into_iter()
        .map(|mut item| {
            if item.group.as_str() != "Intensity" {
                return item;
            }
            item.checked = match preset {
                "none" => false,
                "avg_sum" => item.label.ends_with("Avg (bit)") || item.label.ends_with("Sum (bit)"),
                _ => true,
            };
            item
        })
        .collect()
}

/// Copies `source_group`'s checked pattern onto every "Coloc *" group,
/// matching by field *position* within each group rather than by label
/// (labels differ per class, e.g. "Coloc ClassA object ID" vs "Coloc ClassB object
/// ID", but every class shares the same field layout/order by construction —
/// see `build_column_specs`/`build_coloc_detail_column_specs`). A no-op for
/// any group outside the "Coloc " family (e.g. "Intensity", which is always
/// a singleton section with nothing else to copy to).
fn apply_coloc_group_template(items: Vec<FilterItem>, source_group: &str) -> Vec<FilterItem> {
    if !source_group.starts_with("Coloc ") {
        return items;
    }
    let template: Vec<bool> = items
        .iter()
        .filter(|i| i.group.as_str() == source_group)
        .map(|i| i.checked)
        .collect();
    if template.is_empty() {
        return items;
    }

    let mut group_positions: HashMap<String, usize> = HashMap::new();
    items
        .into_iter()
        .map(|mut item| {
            if item.group.as_str().starts_with("Coloc ") {
                let pos = group_positions.entry(item.group.to_string()).or_insert(0);
                if let Some(&checked) = template.get(*pos) {
                    item.checked = checked;
                }
                *pos += 1;
            }
            item
        })
        .collect()
}

/// Maps the results table's own live `GroupConfig` onto the export batch's
/// simplified "none" | "image" | "regex" representation, for "Add from
/// table"/"Export as Displayed". Coloc Details ignores grouping entirely
/// (that view is always flat). Returns `Err` with a user-facing message if
/// the live grouping can't be represented (`GroupBy::Folder` isn't exposed
/// by the export dialog's own Group-by picker).
fn live_group_by_for_batch(
    is_coloc_detail: bool,
    group_cfg: &GroupConfig,
) -> Result<(String, String), &'static str> {
    if is_coloc_detail {
        return Ok(("none".to_string(), String::new()));
    }
    match group_cfg.group_by {
        GroupBy::None => Ok(("none".to_string(), String::new())),
        GroupBy::Image => Ok(("image".to_string(), String::new())),
        GroupBy::Regex => Ok(("regex".to_string(), group_cfg.regex.clone())),
        GroupBy::Folder => Err(
            "Folder grouping isn't supported in the batch queue yet — use \"Export as Displayed\" instead.",
        ),
    }
}

/// Resolves the concrete image list an "Add from table"/live-capture batch
/// should carry: the live filter's own list if one is set, otherwise every
/// known image name (since a batch's `images` field is never empty — see
/// `ExportBatchItem::images`).
fn images_from_live_filter(image_filter: Option<Vec<String>>, all_known: &[String]) -> Vec<String> {
    image_filter.unwrap_or_else(|| all_known.to_vec())
}

/// Display label for a queued combination's captured style/grouping, e.g.
/// "Table", "Table · Image", "Table · Regex", "Coloc Details".
fn export_style_label(style: &str, group_by: &str) -> String {
    if style == "coloc_detail" {
        return "Coloc Details".to_string();
    }
    if style == "matrix" {
        return match group_by {
            "regex" => "Matrix · Regex".to_string(),
            _ => "Matrix · Folder".to_string(),
        };
    }
    match group_by {
        "image" => "Table · Image".to_string(),
        "regex" => "Table · Regex".to_string(),
        _ => "Table".to_string(),
    }
}

/// A queued Matrix batch's Value/Aggregate/Colors/Range picks, captured from
/// the live `ResultsState.matrix_*` globals at "Add" time (see
/// `ExportBatchItem::matrix_metric` etc.) — not meaningful for "table"/
/// "coloc_detail" batches, which get [`MatrixBatchFields::default`] instead.
struct MatrixBatchFields {
    metric: String,
    agg: String,
    color_scheme: String,
    range_auto: bool,
    range_min: f32,
    range_max: f32,
    kind: String,
}

impl Default for MatrixBatchFields {
    fn default() -> Self {
        Self {
            metric: String::new(),
            agg: String::new(),
            color_scheme: String::new(),
            range_auto: true,
            range_min: 0.0,
            range_max: 1.0,
            kind: "plate".to_string(),
        }
    }
}

fn matrix_batch_fields(state: &ResultsState, is_matrix: bool) -> MatrixBatchFields {
    if !is_matrix {
        return MatrixBatchFields::default();
    }
    MatrixBatchFields {
        metric: state.get_matrix_metric().to_string(),
        agg: state.get_matrix_agg().to_string(),
        color_scheme: state.get_matrix_color_scheme().to_string(),
        range_auto: state.get_matrix_range_auto(),
        range_min: state.get_matrix_range_min(),
        range_max: state.get_matrix_range_max(),
        kind: state.get_export_matrix_kind().to_string(),
    }
}

/// Builds the `GroupConfig` for an export run from the dialog's own "Group
/// by" choice — mirrors the aggregate/split presets the main table's own
/// quick Group-by-Image / Group-by-Regex buttons use (avg + sum, split by
/// class), rather than exposing every aggregate-function checkbox in the
/// dialog too.
fn group_config_from_dialog(group_by: &str, regex: String) -> GroupConfig {
    match group_by {
        "image" => GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg, AggFunc::Sum],
            split_colocalized: false,
            group_by_class: true,
        },
        "regex" => GroupConfig {
            group_by: GroupBy::Regex,
            regex,
            aggs: vec![AggFunc::Avg, AggFunc::Sum],
            split_colocalized: false,
            group_by_class: true,
        },
        _ => GroupConfig::default(),
    }
}

fn specs_to_slint_cols(
    specs: &[ColumnSpec],
    widths: &HashMap<String, f32>,
) -> Vec<ResultsColumnDef> {
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

pub(crate) fn model_to_vec(model: &slint::ModelRc<FilterItem>) -> Vec<FilterItem> {
    (0..model.row_count())
        .filter_map(|i| model.row_data(i))
        .collect()
}

/// Number of checked items — the "N" in the export dialog's ChecklistBox
/// "N/M selected" count. Computed here because Slint's function bodies have
/// no loop or array-reduce construct (only `.length` is a builtin array
/// method), so this can't be computed in the component.
///
/// Counts every item, including one that's also serving as its group's
/// header banner (e.g. "Coloc w/ ClassA (#)", the first column pushed for
/// each partner class in `build_column_specs`, which `mark_group_headers`
/// reuses as the "Coloc ClassA" section header rather than inserting a
/// synthetic row) — that item is still a real, individually exportable
/// column with its own checkbox in `ChecklistBox`, not just a banner, so it
/// has to count like any other row. An earlier version excluded
/// `group_header` items here to match `ChecklistBox` not giving them their
/// own checkbox row at all, which made that first column of every group
/// effectively unselectable and invisible from the count.
fn checked_count(items: &[FilterItem]) -> i32 {
    items.iter().filter(|i| i.checked).count() as i32
}

/// The "M" counterpart to `checked_count`.
fn total_count(items: &[FilterItem]) -> i32 {
    items.len() as i32
}

/// Labels of up to the first `cap` checked items, in `items` order — backs
/// the export dialog's collapsed PickerField chip row. See `checked_count`
/// for why group-header items are included rather than skipped. Same "no
/// loop/array-reduce in Slint" reasoning as `checked_count` for why this is
/// computed here; a recursive `pure function` isn't a workaround either,
/// since Slint treats a function calling itself as a binding loop and
/// refuses to compile it.
fn checked_labels(items: &[FilterItem], cap: usize) -> Vec<SharedString> {
    items
        .iter()
        .filter(|i| i.checked)
        .take(cap)
        .map(|i| i.label.clone())
        .collect()
}

/// The export dialog ChecklistBox's search-filtered display list. Slint has
/// no string `.contains()` builtin (only `is-float`/`to-float`/`is-empty`/
/// `character-count`/`to-lowercase`/`to-uppercase`), so matching happens
/// here rather than in the component. Group-header rows are always kept
/// regardless of match, so a section banner never disappears while its
/// other members are being searched for.
fn filter_checklist_by_search(items: &[FilterItem], search: &str) -> Vec<FilterItem> {
    if search.is_empty() {
        return items.to_vec();
    }
    let lower = search.to_lowercase();
    items
        .iter()
        .filter(|i| i.group_header || i.label.to_lowercase().contains(&lower))
        .cloned()
        .collect()
}

/// Pushes a freshly mutated `items` list out to every property the export
/// dialog's ChecklistBox/PickerField instances read: the backing list
/// itself, the all-checked flag, the live count, the search-filtered
/// display list, and the collapsed field's chip labels. `search` should be
/// the checklist's current `*_search_text` (read from `state` at each call
/// site, since Slint owns that property).
fn apply_export_checklist(
    items: Vec<FilterItem>,
    search: &str,
    set_items: impl FnOnce(slint::ModelRc<FilterItem>),
    set_all_checked: impl FnOnce(bool),
    set_checked_count: impl FnOnce(i32),
    set_total_count: impl FnOnce(i32),
    set_displayed_items: impl FnOnce(slint::ModelRc<FilterItem>),
    set_chip_labels: impl FnOnce(slint::ModelRc<SharedString>),
) {
    set_all_checked(items.iter().all(|i| i.checked));
    set_checked_count(checked_count(&items));
    set_total_count(total_count(&items));
    set_displayed_items(to_model(filter_checklist_by_search(&items, search)));
    set_chip_labels(slint::ModelRc::new(slint::VecModel::from(checked_labels(
        &items, 4,
    ))));
    set_items(to_model(items));
}

fn apply_export_class_checklist(state: &ResultsState, items: Vec<FilterItem>, search: &str) {
    apply_export_checklist(
        items,
        search,
        |m| state.set_export_class_items(m),
        |b| state.set_export_class_all_checked(b),
        |c| state.set_export_class_checked_count(c),
        |t| state.set_export_class_total_count(t),
        |m| state.set_export_class_displayed_items(m),
        |m| state.set_export_class_chip_labels(m),
    );
}

fn apply_export_image_checklist(state: &ResultsState, items: Vec<FilterItem>, search: &str) {
    apply_export_checklist(
        items,
        search,
        |m| state.set_export_image_items(m),
        |b| state.set_export_image_all_checked(b),
        |c| state.set_export_image_checked_count(c),
        |t| state.set_export_image_total_count(t),
        |m| state.set_export_image_displayed_items(m),
        |m| state.set_export_image_chip_labels(m),
    );
}

/// Like the class/image wrappers, but also (re-)marks group headers first —
/// the only one of the three export checklists that's ever grouped
/// ("Intensity", "Coloc ClassA", ...), so `checked_count`/`total_count`/
/// `filter_checklist_by_search`'s `group_header` exclusion only matters
/// here. Safe to call even on already-marked items (`mark_group_headers`
/// only reads `group`, so it's idempotent).
fn apply_export_column_checklist(state: &ResultsState, items: Vec<FilterItem>, search: &str) {
    let items = mark_group_headers(items);
    apply_export_checklist(
        items,
        search,
        |m| state.set_export_column_items(m),
        |b| state.set_export_column_all_checked(b),
        |c| state.set_export_column_checked_count(c),
        |t| state.set_export_column_total_count(t),
        |m| state.set_export_column_displayed_items(m),
        |m| state.set_export_column_chip_labels(m),
    );
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
    let lookup: BTreeMap<&str, bool> = items
        .iter()
        .map(|i| (i.label.as_str(), i.checked))
        .collect();
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

fn export_batches_to_vec(model: &slint::ModelRc<ExportBatchItem>) -> Vec<ExportBatchItem> {
    (0..model.row_count())
        .filter_map(|i| model.row_data(i))
        .collect()
}

/// Plain, `Send`able copy of one queued `ExportBatchItem` — the Slint struct
/// itself holds a `ModelRc` (an `Rc`) in its `classes` field and so can't be
/// moved into the background export thread directly.
struct PlannedExport {
    name: String,
    /// Empty means "every class" (no filter applied).
    classes: Vec<String>,
    /// Image names in scope — always concrete (never "empty means all"),
    /// since `each_image` needs real names to iterate over.
    images: Vec<String>,
    /// When true, one output file is written per entry in `images` instead
    /// of one file covering all of them together.
    each_image: bool,
    is_xlsx: bool,
    is_coloc_detail: bool,
    is_matrix: bool,
    /// "none" | "image" | "regex" when `is_coloc_detail`/`is_matrix` are both
    /// false; "folder" | "regex" when `is_matrix`; ignored when
    /// `is_coloc_detail`.
    group_by: String,
    group_regex: String,
    /// Column labels to include; empty means "every column" (no filter
    /// applied). Ignored when `is_matrix`.
    columns: Vec<String>,
    /// The following six fields are only meaningful when `is_matrix`.
    matrix_metric: String,
    matrix_agg: String,
    matrix_color_scheme: String,
    matrix_range_auto: bool,
    matrix_range_min: f64,
    matrix_range_max: f64,
    /// "plate" (default) or "well" — see `ExportBatchItem.matrix_kind`.
    matrix_kind: String,
}

/// An image's display name with its own file extension stripped (e.g.
/// "image_01.tif" -> "image_01"), so per-image export filenames don't end up
/// with two extensions back to back. Falls back to the full name if it has
/// no extension to strip.
fn image_stem(image_name: &str) -> String {
    std::path::Path::new(image_name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| image_name.to_string())
}

/// Replaces characters that are illegal (or awkward) in a filename with `_`.
/// Falls back to `"export"` if nothing usable is left, so a combination named
/// e.g. "///" still produces a valid path.
fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim().to_string();
    // A name made up entirely of characters we just replaced (e.g. "///")
    // is as "nothing usable" as an empty one — fall back rather than
    // writing a file literally named "___".
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '_') {
        "export".to_string()
    } else {
        cleaned
    }
}

/// Builds `<folder>/<name>.<ext>`, appending " (2)", " (3)", ... if that path
/// is already in `used` (e.g. two combinations given the same name), and
/// records whichever path it returns into `used` so later calls don't
/// collide with it either.
fn unique_export_path(
    folder: &std::path::Path,
    name: &str,
    ext: &str,
    used: &mut std::collections::HashSet<PathBuf>,
) -> PathBuf {
    let base = sanitize_filename(name);
    let mut candidate = folder.join(format!("{base}.{ext}"));
    let mut n = 2;
    while used.contains(&candidate) {
        candidate = folder.join(format!("{base} ({n}).{ext}"));
        n += 1;
    }
    used.insert(candidate.clone());
    candidate
}

/// Copies visibility flags from `prev` onto `fresh` wherever both share a
/// column id. Ids present only in `fresh` (e.g. a newly appeared coloc
/// partner class) are left at their built-in default (visible).
fn carry_over_visibility(fresh: &mut [ColumnSpec], prev: &[ColumnSpec]) {
    let prev_visibility: HashMap<&str, bool> =
        prev.iter().map(|c| (c.id.as_str(), c.visible)).collect();
    for spec in fresh.iter_mut() {
        if let Some(&visible) = prev_visibility.get(spec.id.as_str()) {
            spec.visible = visible;
        }
    }
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
    if col_id.starts_with("coloc_detail__") {
        "Colocalization"
    } else if col_id.starts_with("ch") {
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
            item.group_all_checked =
                is_header && *group_checked.get(item.group.as_str()).unwrap_or(&false);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, label: &str, visible: bool) -> ColumnSpec {
        ColumnSpec {
            id: id.into(),
            label: label.into(),
            filterable: false,
            visible,
        }
    }

    #[test]
    fn map_group_by_mirrors_every_slint_variant() {
        assert_eq!(map_group_by(ResultsGroupBy::None), GroupBy::None);
        assert_eq!(map_group_by(ResultsGroupBy::Image), GroupBy::Image);
        assert_eq!(map_group_by(ResultsGroupBy::Folder), GroupBy::Folder);
        assert_eq!(map_group_by(ResultsGroupBy::Regex), GroupBy::Regex);
    }

    #[test]
    fn sync_popup_checked_carries_checked_state_over_by_label() {
        let items = vec![plain_item("A", true), plain_item("B", false)];
        let popup = vec![plain_item("B", true), plain_item("C", false)];
        let synced = sync_popup_checked(&items, &popup);
        assert_eq!(
            synced.len(),
            2,
            "popup's own item set is unchanged, only checked state moves"
        );
        assert!(
            !synced.iter().find(|i| i.label == "B").unwrap().checked,
            "B took items' false"
        );
        assert!(
            !synced.iter().find(|i| i.label == "C").unwrap().checked,
            "C has no match in items, unchanged"
        );
    }

    #[test]
    fn filter_popup_by_search_is_case_insensitive_substring() {
        let items = vec![
            plain_item("Alpha", true),
            plain_item("beta", true),
            plain_item("gamma", true),
        ];
        let filtered = filter_popup_by_search(&items, "ETA");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].label, "beta");
    }

    #[test]
    fn filter_popup_by_search_empty_search_returns_everything() {
        let items = vec![plain_item("Alpha", true), plain_item("beta", true)];
        assert_eq!(filter_popup_by_search(&items, "").len(), 2);
    }

    #[test]
    fn set_checked_for_search_only_touches_matching_items() {
        let items = vec![plain_item("Alpha", true), plain_item("beta", true)];
        let result = set_checked_for_search(&items, "alp", false);
        assert!(!result.iter().find(|i| i.label == "Alpha").unwrap().checked);
        assert!(
            result.iter().find(|i| i.label == "beta").unwrap().checked,
            "non-matching item untouched"
        );
    }

    #[test]
    fn set_all_checked_touches_every_item_regardless_of_search() {
        let items = vec![plain_item("Alpha", false), plain_item("beta", false)];
        let result = set_all_checked(&items, true);
        assert!(result.iter().all(|i| i.checked));
    }

    #[test]
    fn carry_over_visibility_matches_by_id_and_ignores_unknown_columns() {
        let mut fresh = vec![
            spec("object_id", "object ID", true),
            spec("ch0_avg_bit", "Ch0 Avg", true),
        ];
        let prev = vec![
            spec("ch0_avg_bit", "Ch0 Avg", false),
            spec("stale_id", "Gone Now", false),
        ];
        carry_over_visibility(&mut fresh, &prev);
        assert!(
            fresh[0].visible,
            "object_id has no match in prev, keeps its fresh default"
        );
        assert!(
            !fresh[1].visible,
            "ch0_avg_bit's prior hidden state carried over"
        );
    }

    #[test]
    fn any_unchecked_and_all_checked_are_complementary_for_non_empty_lists() {
        let all_true = vec![plain_item("A", true), plain_item("B", true)];
        assert!(!any_unchecked(&all_true));
        assert!(all_checked(&all_true));

        let mixed = vec![plain_item("A", true), plain_item("B", false)];
        assert!(any_unchecked(&mixed));
        assert!(!all_checked(&mixed));
    }

    #[test]
    fn all_checked_of_an_empty_list_is_vacuously_true() {
        assert!(all_checked(&[]));
        assert!(!any_unchecked(&[]));
    }

    #[test]
    fn names_to_filter_items_starts_every_item_checked_and_ungrouped() {
        let items = names_to_filter_items(&["A".to_string(), "B".to_string()]);
        assert_eq!(items.len(), 2);
        for item in &items {
            assert!(item.checked);
            assert_eq!(item.group, SharedString::new());
            assert!(!item.group_header);
        }
        assert_eq!(items[0].label, SharedString::from("A"));
    }

    #[test]
    fn column_group_buckets_by_id_prefix() {
        assert_eq!(
            column_group("coloc_detail__ClassA__object_id"),
            "Colocalization"
        );
        assert_eq!(column_group("ch0_avg_bit"), "Intensity");
        assert_eq!(
            column_group("coloc_partner__ClassA__count"),
            "Colocalization"
        );
        assert_eq!(column_group("object_id"), "");
    }

    #[test]
    fn group_all_checked_and_set_group_checked_for_search_are_scoped_to_group_and_search() {
        let items = vec![
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Min", true)
            },
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Max", false)
            },
            FilterItem {
                group: "Other".into(),
                ..plain_item("Something", false)
            },
        ];
        assert!(
            !group_all_checked_for_search(&items, "Intensity", ""),
            "Ch0 Max is unchecked"
        );

        // Search-scoped: only "Ch0 Min" matches "min", and it's already checked.
        assert!(group_all_checked_for_search(&items, "Intensity", "min"));

        let updated = set_group_checked_for_search(&items, "Intensity", "max", true);
        assert!(updated[0].checked, "Ch0 Min was already checked");
        assert!(
            updated[1].checked,
            "Ch0 Max matched the search and got checked"
        );
        assert!(!updated[2].checked, "Other group untouched");
    }

    #[test]
    fn plottable_column_labels_only_lists_visible_numeric_columns() {
        let specs = vec![
            spec("object_id", "object ID", true),  // not numeric
            spec("area_px", "Area (px)", true),    // numeric, visible
            spec("ch0_avg_bit", "Ch0 Avg", false), // numeric, hidden
        ];
        let labels = plottable_column_labels(&specs);
        assert_eq!(labels, vec![SharedString::from("Area (px)")]);
    }

    #[test]
    fn heatmap_metric_options_leads_with_the_count_sentinel() {
        let specs = vec![spec("area_px", "Area (px)", true)];
        let options = heatmap_metric_options(&specs);
        assert_eq!(options[0], SharedString::from(HEATMAP_METRIC_COUNT_LABEL));
        assert_eq!(options[1], SharedString::from("Area (px)"));
        assert_eq!(options.len(), 2);
    }

    #[test]
    fn heatmap_color_scheme_options_covers_every_scheme() {
        let options = heatmap_color_scheme_options();
        assert_eq!(options.len(), HeatmapColorScheme::all().len());
    }

    #[test]
    fn has_intensity_columns_checks_for_a_ch_prefixed_id() {
        assert!(has_intensity_columns(&[spec(
            "ch0_avg_bit",
            "Ch0 Avg",
            true
        )]));
        assert!(!has_intensity_columns(&[spec(
            "object_id",
            "object ID",
            true
        )]));
        assert!(!has_intensity_columns(&[]));
    }

    /// Simulates the exact chain `column_filter_apply` + `bg_reload_coloc_detail_page0`
    /// run through in the live app, without any Slint/GUI machinery: build the
    /// coloc-detail specs, hide one partner-class column by *label* (matching
    /// `column_filter_apply`'s own key), rebuild fresh specs (as a reload
    /// would, since `discover_coloc_detail_columns` re-runs every time) and
    /// carry the visibility choice over by *id* (matching
    /// `bg_reload_coloc_detail_page0`'s own key), then confirm the hidden
    /// column is still hidden and its flattened value is blank.
    #[test]
    fn column_filter_visibility_survives_a_coloc_detail_reload_round_trip() {
        let coloc_partner_classes = vec!["ClassA".to_string(), "ClassB".to_string()];
        let channels = vec![0];

        // 1. Initial build (mirrors the first `bg_reload_coloc_detail_page0`
        //    call): every column starts visible.
        let mut specs = build_coloc_detail_column_specs(&channels, &coloc_partner_classes);
        assert!(
            specs.iter().all(|c| c.visible),
            "sanity: every column starts visible"
        );

        // 2. Simulate `column_filter_apply`: the user unchecks ClassA's "object ID"
        //    column in the Columns popup. This mutation matches by *label*.
        let visibility: BTreeMap<String, bool> = [("Coloc ClassA object ID".to_string(), false)]
            .into_iter()
            .collect();
        for spec in specs.iter_mut() {
            if let Some(&visible) = visibility.get(&spec.label) {
                spec.visible = visible;
            }
        }
        assert!(
            !specs
                .iter()
                .find(|c| c.id == "coloc_detail__ClassA__object_id")
                .unwrap()
                .visible,
            "sanity: the mutation actually hid the column"
        );

        // 3. Simulate the next `bg_reload_coloc_detail_page0` call: it always
        //    rebuilds a *fresh* spec list (channels/classes are re-discovered
        //    every reload) and carries the previous visibility over by *id*.
        let mut fresh = build_coloc_detail_column_specs(&channels, &coloc_partner_classes);
        carry_over_visibility(&mut fresh, &specs);

        let carried = fresh
            .iter()
            .find(|c| c.id == "coloc_detail__ClassA__object_id")
            .unwrap();
        assert!(
            !carried.visible,
            "the hidden column must stay hidden across a reload, not reset to visible"
        );
        // An unrelated column (ClassB's object ID) must be unaffected.
        let unrelated = fresh
            .iter()
            .find(|c| c.id == "coloc_detail__ClassB__object_id")
            .unwrap();
        assert!(unrelated.visible);

        // 4. Confirm the value itself renders blank for the hidden column,
        //    using the real flatten/format path.
        let mut src = evanalyzer_app::result::ObjectRow {
            image_name: "img.tif".into(),
            image_rel_path: String::new(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: "00000000-0000-0000-0000-000000000001".into(),
            seg_class_name: None,
            seg_class_id: None,
            object_class_name: vec![],
            object_class_id: vec![],
            parent_id: None,
            children: vec![],
            track_id: 0,
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            centroid_x_nm: 0.0,
            centroid_y_nm: 0.0,
            area_px: 0,
            area_nm2: 0.0,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity: 0.0,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: String::new(),
            coloc_json: String::new(),
            bbox_px: [0, 0, 0, 0],
        };
        src.coloc_json = r#"{"ClassA":["00000000-0000-0000-0000-000000000002"]}"#.into();
        let mut partner = src.clone();
        partner.object_id = "00000000-0000-0000-0000-000000000002".into();

        let rows = flatten_coloc_rows(&[src], &[partner], &fresh);
        assert_eq!(rows.len(), 1);
        let col_idx = fresh
            .iter()
            .position(|c| c.id == "coloc_detail__ClassA__object_id")
            .unwrap();
        assert_eq!(
            rows[0].values[col_idx], "",
            "hidden column's value must render blank, not the partner's actual object_id"
        );
    }

    #[test]
    fn sanitize_filename_replaces_illegal_characters() {
        assert_eq!(sanitize_filename("Group A/B:C"), "Group A_B_C");
        assert_eq!(sanitize_filename("  padded  "), "padded");
    }

    #[test]
    fn sanitize_filename_falls_back_when_nothing_usable_remains() {
        assert_eq!(sanitize_filename("///"), "export");
        assert_eq!(sanitize_filename("   "), "export");
    }

    #[test]
    fn unique_export_path_appends_a_counter_on_collision() {
        let folder = std::path::Path::new("/tmp/exports");
        let mut used = std::collections::HashSet::new();

        let first = unique_export_path(folder, "GroupABC", "csv", &mut used);
        let second = unique_export_path(folder, "GroupABC", "csv", &mut used);
        let third = unique_export_path(folder, "GroupABC", "csv", &mut used);

        assert_eq!(first, folder.join("GroupABC.csv"));
        assert_eq!(second, folder.join("GroupABC (2).csv"));
        assert_eq!(third, folder.join("GroupABC (3).csv"));
    }

    #[test]
    fn unique_export_path_does_not_collide_across_different_formats() {
        let folder = std::path::Path::new("/tmp/exports");
        let mut used = std::collections::HashSet::new();

        let csv = unique_export_path(folder, "GroupABC", "csv", &mut used);
        let xlsx = unique_export_path(folder, "GroupABC", "xlsx", &mut used);

        assert_eq!(csv, folder.join("GroupABC.csv"));
        assert_eq!(
            xlsx,
            folder.join("GroupABC.xlsx"),
            "different extensions never collide"
        );
    }

    #[test]
    fn image_stem_strips_the_images_own_extension() {
        assert_eq!(image_stem("image_01.tif"), "image_01");
        assert_eq!(image_stem("scan.ome.tiff"), "scan.ome");
    }

    #[test]
    fn image_stem_falls_back_to_the_full_name_when_there_is_no_extension() {
        assert_eq!(image_stem("image_01"), "image_01");
        assert_eq!(image_stem(".hidden"), ".hidden");
    }

    #[test]
    fn column_items_from_specs_checked_state_mirrors_visibility() {
        let specs = vec![
            ColumnSpec {
                id: "object_id".into(),
                label: "object ID".into(),
                filterable: false,
                visible: true,
            },
            ColumnSpec {
                id: "area_px".into(),
                label: "Area (px²)".into(),
                filterable: false,
                visible: false,
            },
        ];
        let items = column_items_from_specs(&specs);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].label, SharedString::from("object ID"));
        assert!(items[0].checked);
        assert_eq!(items[1].label, SharedString::from("Area (px²)"));
        assert!(
            !items[1].checked,
            "hidden columns must start unchecked in the export dialog"
        );
    }

    #[test]
    fn export_column_group_buckets_intensity_and_per_partner_class_coloc_columns() {
        assert_eq!(export_column_group("ch0_min_bit"), "Intensity");
        assert_eq!(export_column_group("ch12_sum_bit"), "Intensity");
        assert_eq!(
            export_column_group("coloc_partner__ClassA__count"),
            "Coloc ClassA"
        );
        assert_eq!(
            export_column_group("coloc_partner__ClassB__ids"),
            "Coloc ClassB"
        );
        assert_eq!(
            export_column_group("coloc_detail__ClassA__ch0_avg_bit"),
            "Coloc ClassA"
        );
        assert_eq!(
            export_column_group("coloc_detail__ClassA__object_id"),
            "Coloc ClassA"
        );
        assert_eq!(export_column_group("object_id"), "");
        assert_eq!(export_column_group("area_px"), "");
    }

    #[test]
    fn column_items_from_specs_marks_group_headers_for_intensity_and_coloc_columns() {
        let specs = vec![
            ColumnSpec {
                id: "object_id".into(),
                label: "object ID".into(),
                filterable: false,
                visible: true,
            },
            ColumnSpec {
                id: "ch0_min_bit".into(),
                label: "Ch0 Min (bit)".into(),
                filterable: false,
                visible: true,
            },
            ColumnSpec {
                id: "ch0_max_bit".into(),
                label: "Ch0 Max (bit)".into(),
                filterable: false,
                visible: true,
            },
            ColumnSpec {
                id: "coloc_partner__ClassA__count".into(),
                label: "Coloc w/ ClassA (#)".into(),
                filterable: false,
                visible: true,
            },
        ];
        let items = column_items_from_specs(&specs);

        assert_eq!(
            items[0].group,
            SharedString::new(),
            "ungrouped column has no header"
        );
        assert!(!items[0].group_header);

        assert_eq!(items[1].group, SharedString::from("Intensity"));
        assert!(
            items[1].group_header,
            "first item of a group run is the header"
        );
        assert!(items[1].group_all_checked);

        assert_eq!(items[2].group, SharedString::from("Intensity"));
        assert!(
            !items[2].group_header,
            "second item of the same group is not a header"
        );

        assert_eq!(items[3].group, SharedString::from("Coloc ClassA"));
        assert!(items[3].group_header, "a new group starts a new header");
    }

    fn plain_item(label: &str, checked: bool) -> FilterItem {
        FilterItem {
            label: label.into(),
            checked,
            group: SharedString::new(),
            group_header: false,
            group_all_checked: false,
        }
    }

    #[test]
    fn checked_count_and_total_count_include_the_group_header_item_itself() {
        // A grouped list (mirrors the Columns checklist): item[0] is a plain
        // column, item[1] is the checked "Intensity" section header itself
        // (mark_group_headers reuses a real item as its group's header, it
        // doesn't insert a synthetic row), item[2] is another Intensity
        // column that's unchecked.
        let items = mark_group_headers(vec![
            plain_item("object ID", true),
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Min (bit)", true)
            },
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Max (bit)", false)
            },
        ]);
        assert!(items[1].group_header, "sanity: item[1] is the group header");

        // The header item is still a real, individually exportable column
        // (`ChecklistBox` gives it its own checkbox alongside the section
        // banner), so it counts like any other row: 3 real columns, 2 of
        // them checked (object ID and the header item, Ch0 Min).
        assert_eq!(total_count(&items), 3);
        assert_eq!(checked_count(&items), 2);
    }

    #[test]
    fn checked_count_and_total_count_agree_with_plain_ungrouped_lists() {
        // Classes/Images never set `group`, so every item stays
        // `group_header: false` and nothing gets excluded.
        let items = vec![
            plain_item("A", true),
            plain_item("B", false),
            plain_item("C", true),
        ];
        assert_eq!(checked_count(&items), 2);
        assert_eq!(total_count(&items), 3);
    }

    #[test]
    fn filter_checklist_by_search_is_case_insensitive_and_keeps_group_headers() {
        let items = mark_group_headers(vec![
            plain_item("Area", true),
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Min (bit)", true)
            },
            FilterItem {
                group: "Intensity".into(),
                ..plain_item("Ch0 Max (bit)", false)
            },
        ]);
        assert!(items[1].group_header, "sanity: item[1] is the group header");

        // "min" matches only "Ch0 Min (bit)" by label, but the Intensity
        // header (item[1], which happens to BE that same matching row here)
        // stays included either way; "Area" (no match, not a header) drops.
        let filtered = filter_checklist_by_search(&items, "min");
        let labels: Vec<&str> = filtered.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["Ch0 Min (bit)"]);

        // Empty search returns everything, unfiltered.
        assert_eq!(filter_checklist_by_search(&items, "").len(), 3);
    }

    #[test]
    fn filter_checklist_by_search_keeps_a_non_matching_group_header_so_the_section_still_shows() {
        // The group header here is "Area" itself (first item of the
        // "Misc" group) - it doesn't match "sum", but must stay so the
        // section banner still renders above the column that does match.
        let items = mark_group_headers(vec![
            FilterItem {
                group: "Misc".into(),
                ..plain_item("Area", true)
            },
            FilterItem {
                group: "Misc".into(),
                ..plain_item("Sum", true)
            },
        ]);
        assert!(items[0].group_header);

        let filtered = filter_checklist_by_search(&items, "sum");
        let labels: Vec<&str> = filtered.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["Area", "Sum"],
            "header row is kept even though its own label doesn't match"
        );
    }

    fn intensity_items() -> Vec<FilterItem> {
        let labels = [
            "Ch0 Min (bit)",
            "Ch0 Max (bit)",
            "Ch0 Avg (bit)",
            "Ch0 Sum (bit)",
        ];
        labels
            .iter()
            .map(|l| FilterItem {
                label: SharedString::from(*l),
                checked: true,
                group: "Intensity".into(),
                group_header: false,
                group_all_checked: false,
            })
            .collect()
    }

    #[test]
    fn apply_intensity_preset_none_unchecks_every_intensity_column() {
        let items = apply_intensity_preset(intensity_items(), "none");
        assert!(items.iter().all(|i| !i.checked));
    }

    #[test]
    fn apply_intensity_preset_avg_sum_checks_only_avg_and_sum() {
        let items = apply_intensity_preset(intensity_items(), "avg_sum");
        let checked: Vec<&str> = items
            .iter()
            .filter(|i| i.checked)
            .map(|i| i.label.as_str())
            .collect();
        assert_eq!(checked, ["Ch0 Avg (bit)", "Ch0 Sum (bit)"]);
    }

    #[test]
    fn apply_intensity_preset_all_checks_every_intensity_column() {
        let mut items = intensity_items();
        items[0].checked = false;
        let items = apply_intensity_preset(items, "all");
        assert!(items.iter().all(|i| i.checked));
    }

    #[test]
    fn apply_intensity_preset_leaves_other_groups_untouched() {
        let mut items = intensity_items();
        items.push(FilterItem {
            label: "Coloc w/ ClassA (#)".into(),
            checked: true,
            group: "Coloc ClassA".into(),
            group_header: false,
            group_all_checked: false,
        });
        let items = apply_intensity_preset(items, "none");
        assert!(
            items.last().unwrap().checked,
            "non-Intensity columns must be unaffected"
        );
    }

    fn coloc_item(label: &str, group: &str, checked: bool) -> FilterItem {
        FilterItem {
            label: SharedString::from(label),
            checked,
            group: SharedString::from(group),
            group_header: false,
            group_all_checked: false,
        }
    }

    #[test]
    fn apply_coloc_group_template_copies_checked_pattern_by_position_across_classes() {
        let items = vec![
            coloc_item("Coloc ClassA object ID", "Coloc ClassA", true),
            coloc_item("Coloc ClassA Area (px²)", "Coloc ClassA", false),
            coloc_item("Coloc ClassA Circularity", "Coloc ClassA", true),
            coloc_item("Coloc ClassB object ID", "Coloc ClassB", true),
            coloc_item("Coloc ClassB Area (px²)", "Coloc ClassB", true),
            coloc_item("Coloc ClassB Circularity", "Coloc ClassB", true),
        ];
        let items = apply_coloc_group_template(items, "Coloc ClassA");
        let class_b: Vec<bool> = items
            .iter()
            .filter(|i| i.group.as_str() == "Coloc ClassB")
            .map(|i| i.checked)
            .collect();
        assert_eq!(
            class_b,
            vec![true, false, true],
            "ClassB must mirror ClassA field-by-field"
        );
    }

    #[test]
    fn apply_coloc_group_template_is_a_noop_for_non_coloc_groups() {
        let items = intensity_items();
        let after = apply_coloc_group_template(items.clone(), "Intensity");
        let before: Vec<bool> = items.iter().map(|i| i.checked).collect();
        let after: Vec<bool> = after.iter().map(|i| i.checked).collect();
        assert_eq!(
            before, after,
            "Intensity is a singleton section — nothing to propagate to"
        );
    }

    #[test]
    fn apply_coloc_group_template_leaves_other_sections_untouched() {
        let mut items = vec![
            coloc_item("Coloc ClassA object ID", "Coloc ClassA", false),
            coloc_item("Coloc ClassB object ID", "Coloc ClassB", true),
        ];
        items.extend(intensity_items());
        let items = apply_coloc_group_template(items, "Coloc ClassA");
        assert!(
            items
                .iter()
                .filter(|i| i.group.as_str() == "Intensity")
                .all(|i| i.checked),
            "Intensity columns must be unaffected by a Coloc-group propagate"
        );
    }

    #[test]
    fn export_style_label_covers_every_combination() {
        assert_eq!(export_style_label("table", "none"), "Table");
        assert_eq!(export_style_label("table", "image"), "Table · Image");
        assert_eq!(export_style_label("table", "regex"), "Table · Regex");
        assert_eq!(export_style_label("coloc_detail", "none"), "Coloc Details");
        assert_eq!(
            export_style_label("coloc_detail", "image"),
            "Coloc Details",
            "grouping is ignored for Coloc Details, which is always flat"
        );
        assert_eq!(export_style_label("matrix", "folder"), "Matrix · Folder");
        assert_eq!(export_style_label("matrix", "regex"), "Matrix · Regex");
        assert_eq!(
            export_style_label("matrix", "none"),
            "Matrix · Folder",
            "an unrecognized/leftover group_by value falls back to Folder, matrix's own default"
        );
    }

    #[test]
    fn matrix_batch_fields_defaults_when_not_matrix() {
        let defaults = MatrixBatchFields::default();
        assert_eq!(defaults.metric, "");
        assert_eq!(defaults.agg, "");
        assert_eq!(defaults.color_scheme, "");
        assert!(defaults.range_auto);
        assert_eq!(defaults.range_min, 0.0);
        assert_eq!(defaults.range_max, 1.0);
        assert_eq!(defaults.kind, "plate");
    }

    #[test]
    fn live_group_by_for_batch_coloc_detail_ignores_grouping() {
        let group_cfg = GroupConfig {
            group_by: GroupBy::Image,
            regex: String::new(),
            aggs: vec![AggFunc::Avg],
            split_colocalized: false,
            group_by_class: false,
        };
        assert_eq!(
            live_group_by_for_batch(true, &group_cfg).unwrap(),
            ("none".to_string(), String::new())
        );
    }

    #[test]
    fn live_group_by_for_batch_maps_none_image_regex() {
        let none_cfg = GroupConfig::default();
        assert_eq!(
            live_group_by_for_batch(false, &none_cfg).unwrap(),
            ("none".to_string(), String::new())
        );

        let image_cfg = GroupConfig {
            group_by: GroupBy::Image,
            ..GroupConfig::default()
        };
        assert_eq!(
            live_group_by_for_batch(false, &image_cfg).unwrap(),
            ("image".to_string(), String::new())
        );

        let regex_cfg = GroupConfig {
            group_by: GroupBy::Regex,
            regex: "(.*)_ch\\d+".to_string(),
            ..GroupConfig::default()
        };
        assert_eq!(
            live_group_by_for_batch(false, &regex_cfg).unwrap(),
            ("regex".to_string(), "(.*)_ch\\d+".to_string())
        );
    }

    #[test]
    fn live_group_by_for_batch_rejects_folder_grouping() {
        let folder_cfg = GroupConfig {
            group_by: GroupBy::Folder,
            ..GroupConfig::default()
        };
        assert!(live_group_by_for_batch(false, &folder_cfg).is_err());
    }

    #[test]
    fn images_from_live_filter_falls_back_to_every_known_image_when_unset() {
        let known = vec!["image_01.tif".to_string(), "image_02.tif".to_string()];
        assert_eq!(images_from_live_filter(None, &known), known);
    }

    #[test]
    fn images_from_live_filter_uses_the_live_filter_when_set() {
        let known = vec!["image_01.tif".to_string(), "image_02.tif".to_string()];
        let filtered = vec!["image_01.tif".to_string()];
        assert_eq!(
            images_from_live_filter(Some(filtered.clone()), &known),
            filtered
        );
    }

    #[test]
    fn group_config_from_dialog_none_is_default_ungrouped() {
        let group = group_config_from_dialog("none", String::new());
        assert_eq!(group.group_by, GroupBy::None);
        assert_eq!(group.aggs, vec![AggFunc::Avg]);
        assert!(!group.group_by_class);
        assert!(!group.split_colocalized);
    }

    #[test]
    fn group_config_from_dialog_image_matches_the_table_quick_preset() {
        let group = group_config_from_dialog("image", String::new());
        assert_eq!(group.group_by, GroupBy::Image);
        assert_eq!(group.aggs, vec![AggFunc::Avg, AggFunc::Sum]);
        assert!(group.group_by_class);
        assert!(!group.split_colocalized);
    }

    #[test]
    fn group_config_from_dialog_regex_carries_the_pattern_through() {
        let group = group_config_from_dialog("regex", "(.*)_ch\\d+".to_string());
        assert_eq!(group.group_by, GroupBy::Regex);
        assert_eq!(group.regex, "(.*)_ch\\d+");
        assert_eq!(group.aggs, vec![AggFunc::Avg, AggFunc::Sum]);
        assert!(group.group_by_class);
    }

    // -- attach_callbacks (live ResultsWindow) -------------------------------------

    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::images_list_controller::ImagesListController;
    use crate::editor::object_list_controller::ObjectListController;
    use crate::editor::test_support::{test_ui_state, test_ui_windows};
    use crate::editor::viewport_controller::ViewportController;

    fn make_controller(
        results_ui: slint::Weak<ResultsWindow>,
    ) -> (Arc<UiState>, Arc<ResultsTableController>) {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_list_controller = Arc::new(ImagesListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(HistogramController::new(
                slint::Weak::default(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            Arc::new(ImageMetaController::new(
                slint::Weak::default(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            object_list_controller,
        ));
        let controller = Arc::new(ResultsTableController::new(
            results_ui,
            ui_state.clone(),
            image_list_controller,
        ));
        (ui_state, controller)
    }

    #[test]
    fn attach_callbacks_image_toggle_updates_checked_state_and_active_flag() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_filter_image_items(to_model(vec![plain_item("a", true), plain_item("b", true)]));
        state.set_filter_image_popup(to_model(vec![plain_item("a", true), plain_item("b", true)]));

        state.invoke_image_filter_label_toggled("a".into());

        let items = model_to_vec(&state.get_filter_image_items());
        assert!(!items.iter().find(|i| i.label == "a").unwrap().checked);
        assert!(items.iter().find(|i| i.label == "b").unwrap().checked);
        assert!(state.get_filter_image_active(), "one unchecked item makes the filter active");
    }

    #[test]
    fn attach_callbacks_image_select_all_and_clear_all_toggle_every_item() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_filter_image_items(to_model(vec![plain_item("a", false), plain_item("b", false)]));
        state.set_filter_image_popup(to_model(vec![plain_item("a", false), plain_item("b", false)]));

        state.invoke_image_select_all();
        let items = model_to_vec(&state.get_filter_image_items());
        assert!(items.iter().all(|i| i.checked));
        assert!(!state.get_filter_image_active());

        state.invoke_image_clear_all();
        let items = model_to_vec(&state.get_filter_image_items());
        assert!(items.iter().all(|i| !i.checked));
        assert!(state.get_filter_image_active());
    }

    #[test]
    fn attach_callbacks_class_toggle_and_select_all_mirror_the_image_filter_behavior() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_filter_class_items(to_model(vec![plain_item("A", true), plain_item("B", true)]));
        state.set_filter_class_popup(to_model(vec![plain_item("A", true), plain_item("B", true)]));

        state.invoke_class_filter_label_toggled("A".into());
        assert!(!model_to_vec(&state.get_filter_class_items())[0].checked);

        state.invoke_class_select_all();
        assert!(model_to_vec(&state.get_filter_class_items()).iter().all(|i| i.checked));
    }

    #[test]
    fn attach_callbacks_coloc_toggle_and_select_all_mirror_the_image_filter_behavior() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_filter_coloc_items(to_model(vec![plain_item("x", true)]));
        state.set_filter_coloc_popup(to_model(vec![plain_item("x", true)]));

        state.invoke_coloc_filter_label_toggled("x".into());
        assert!(!model_to_vec(&state.get_filter_coloc_items())[0].checked);

        state.invoke_coloc_select_all();
        assert!(model_to_vec(&state.get_filter_coloc_items())[0].checked);

        state.invoke_coloc_clear_all();
        assert!(!model_to_vec(&state.get_filter_coloc_items())[0].checked);
    }

    #[test]
    fn attach_callbacks_column_select_all_and_clear_all_toggle_visibility() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_column_items(to_model(vec![plain_item("Area", false), plain_item("Class", false)]));
        state.set_column_popup(to_model(vec![plain_item("Area", false), plain_item("Class", false)]));

        state.invoke_column_select_all();
        assert!(model_to_vec(&state.get_column_items()).iter().all(|i| i.checked));

        state.invoke_column_clear_all();
        assert!(model_to_vec(&state.get_column_items()).iter().all(|i| !i.checked));
    }

    #[test]
    fn attach_callbacks_image_search_changed_stores_the_search_text() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();
        let state = results_ui.global::<ResultsState>();
        state.set_filter_image_items(to_model(vec![plain_item("apple", false), plain_item("banana", false)]));

        state.invoke_image_filter_search_changed("app".into());

        assert_eq!(*controller.image_search.lock().unwrap(), "app");
        let popup = model_to_vec(&state.get_filter_image_popup());
        assert_eq!(popup.len(), 1);
        assert_eq!(popup[0].label, "apple");
    }

    #[test]
    fn attach_callbacks_column_width_changed_stores_the_width() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());
        controller.attach_callbacks();

        results_ui.global::<ResultsState>()
            .invoke_column_width_changed("area_px".into(), 120.0);

        assert_eq!(
            controller.column_widths.lock().unwrap().get("area_px").copied(),
            Some(120.0)
        );
    }
}
