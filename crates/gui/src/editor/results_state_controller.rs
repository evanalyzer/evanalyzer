use crate::editor::images_list_controller::ImagesListController;
use crate::editor::viewport_controller::ViewportController;

mod split;
use crate::{
    AppWindow, BreadcrumbItem, ChartBoxplotBox, ChartScatterPoint, ChartTimeLine,
    ExportDialogState, MatrixCell, MatrixLevel, MultiSelectItem, ResultRow, ResultsChartKind2,
    ResultsRailMode, ResultsState, UiState,
};
use evanalyzer_app::results::CellValue;
use evanalyzer_app::results::ColorScale;
use evanalyzer_app::results::ColorSchema;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::ColumnEntry;
use evanalyzer_app::results::DatabaseResult;
use evanalyzer_app::results::ExportFormat;
use evanalyzer_app::results::GroupedByImageFilter;
use evanalyzer_app::results::HistogramFilter;
use evanalyzer_app::results::ImageEntry;
use evanalyzer_app::results::ImageHeatmapFilter;
use evanalyzer_app::results::ResultExport;
use evanalyzer_app::results::ResultsSource;
use evanalyzer_app::results::ScatterFilter;
use evanalyzer_app::results::WellFilter;
use evanalyzer_app::results::WellSize;
use evanalyzer_app::results::{Aggregation, HistogramResult};
use evanalyzer_app::results::{BoxplotBox, Cell};
use evanalyzer_app::results::{BoxplotFilter, BoxplotGroup, ScatterResult};
use evanalyzer_app::results::{Grouping, PlateSize};
use evanalyzer_app::results::{
    TimeCourseExport, TimeCourseFilter, TimeCourseLayout, TimeCourseLines, TimeCourseNormalize,
    TimeCourseResult, TimeCourseStat,
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::plate_settings::{GroupingMode, PlateSettings, WellLayout};
use log::{error, info, warn};
use slint::{Color, ComponentHandle, Model, ModelRc, VecModel};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const LIST_PAGE_SIZE: i32 = 500;

type DbJob = Box<dyn FnOnce() + Send>;

/// Runs database work off the UI thread, one job at a time and in the order
/// it was queued, so a slow disk never freezes the window - every UI update
/// a job makes already goes through `invoke_from_event_loop`. A runner
/// thread is started when work arrives and ends once the queue is empty.
///
/// Tests run the jobs inline (`inline`), so they can check the outcome right
/// after triggering one.
struct DbWorker {
    /// Queued jobs, and whether a runner thread is draining them.
    queue: Mutex<(std::collections::VecDeque<DbJob>, bool)>,
    inline: AtomicBool,
    /// Jobs queued or running - drives `ResultsState.loading`.
    pending: AtomicUsize,
}

impl DbWorker {
    fn new() -> Self {
        Self {
            queue: Mutex::new((std::collections::VecDeque::new(), false)),
            inline: AtomicBool::new(cfg!(test)),
            pending: AtomicUsize::new(0),
        }
    }

    fn submit(self: &Arc<Self>, job: DbJob) {
        if self.inline.load(Ordering::SeqCst) {
            job();
            return;
        }
        let mut queue = self.queue.lock().expect("Poisoned");
        queue.0.push_back(job);
        if queue.1 {
            return;
        }
        queue.1 = true;
        drop(queue);
        let worker = self.clone();
        crate::helper::ui_thread::spawn(move || worker.run_queued());
    }

    fn run_queued(&self) {
        loop {
            let job = {
                let mut queue = self.queue.lock().expect("Poisoned");
                match queue.0.pop_front() {
                    Some(job) => job,
                    None => {
                        queue.1 = false;
                        return;
                    }
                }
            };
            // A failing job must not take the jobs queued after it down.
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                error!("A results database job panicked");
            }
        }
    }
}

const CHART_HISTOGRAM_BINS: usize = 24;
const CHART_SCATTER_MAX_POINTS: usize = 2000;

const DEFAULT_LIST_COLUMNS: [Column; 6] = [
    Column::ImageName,
    Column::ObjectId,
    Column::ObjectClass,
    Column::AreaSizePx,
    Column::AreaSizeNm,
    Column::Circularity,
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
    /// Classes side by side instead of one row per object/(image, class).
    pub transpond: bool,
    pub group_by: ListGroupBy,
    /// Unused in Objects mode.
    pub aggregations: Vec<Aggregation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ChartKind {
    #[default]
    Histogram,
    Scatter,
    Boxplot,
    OverTime,
}

#[derive(Clone)]
struct ChartFilter {
    pub kind: ChartKind,
    /// Histogram/Boxplot's single measurement, Scatter's X axis.
    pub column: Column,
    /// Scatter's Y axis only.
    pub y_column: Column,
    /// `None` means no class filter (every object) — doesn't apply to
    /// Boxplot, which always groups by every class present regardless.
    pub object_class: Option<ObjectClass>,
    /// Boxplot: a box per class, image, well, plate row or column.
    pub group_by: BoxplotGroup,
    /// Histogram: its bins, its counts on a log scale.
    pub bins: usize,
    pub log_counts: bool,
    /// Scatter: at most this many points drawn (`None`: all).
    pub max_points: Option<usize>,
    /// Over time: a line per class / image / well / track; median + IQR
    /// (else mean ± SD); in % of the first time point; the tracks faint
    /// behind.
    pub time_lines: TimeCourseLines,
    pub time_median: bool,
    pub time_relative: bool,
    pub time_tracks: bool,
}

impl Default for ChartFilter {
    fn default() -> Self {
        Self {
            kind: ChartKind::default(),
            column: Column::default(),
            y_column: Column::default(),
            object_class: None,
            group_by: BoxplotGroup::default(),
            bins: CHART_HISTOGRAM_BINS,
            log_counts: false,
            max_points: Some(CHART_SCATTER_MAX_POINTS),
            time_lines: TimeCourseLines::Class,
            time_median: false,
            time_relative: false,
            time_tracks: false,
        }
    }
}

/// The chart's / export dialog's "A line per" (0..3).
fn time_lines_of(index: i32) -> TimeCourseLines {
    match index {
        1 => TimeCourseLines::Image,
        2 => TimeCourseLines::Well,
        3 => TimeCourseLines::Track,
        _ => TimeCourseLines::Class,
    }
}

fn time_lines_index(lines: TimeCourseLines) -> i32 {
    match lines {
        TimeCourseLines::Image => 1,
        TimeCourseLines::Well | TimeCourseLines::Group => 2,
        TimeCourseLines::Track => 3,
        TimeCourseLines::Class => 0,
    }
}

/// The time course statistics the export dialog offers, in this order.
const TIME_STATS: [TimeCourseStat; 8] = [
    TimeCourseStat::Mean,
    TimeCourseStat::Sd,
    TimeCourseStat::Median,
    TimeCourseStat::Q1,
    TimeCourseStat::Q3,
    TimeCourseStat::Min,
    TimeCourseStat::Max,
    TimeCourseStat::Sum,
];

fn time_stat_items(selected: &[TimeCourseStat]) -> Vec<MultiSelectItem> {
    TIME_STATS
        .iter()
        .map(|stat| MultiSelectItem {
            key: stat.label().into(),
            value: stat.label().into(),
            color: Color::default(),
            group: "".into(),
            selected: selected.contains(stat),
        })
        .collect()
}

/// At most this many lines are drawn (a legend entry each).
const CHART_TIME_MAX_LINES: usize = 24;
/// At most this many faint tracks.
const CHART_TIME_MAX_TRACKS: usize = 400;

/// Colors of lines that aren't a class's.
const CHART_TIME_PALETTE: [u32; 10] = [
    0x2A6BDF, 0xC2410C, 0x0F9D8A, 0x9333EA, 0xCA8A04, 0xDB2777, 0x0891B2, 0x4D7C0F, 0x7C3AED,
    0xB45309,
];

/// A line of the "Over time" chart before it's a `ChartTimeLine`: label,
/// color, its values per time point (center, low, high).
type TimeSeries = (
    String,
    u32,
    Vec<(u32, Option<f64>, Option<f64>, Option<f64>)>,
);

/// The time course's lines: a series per line (and class) of `result`, in
/// their order - center and spread from the statistics `median` asks for
/// (median, Q1, Q3 - else mean, mean ∓ SD). Lines without any value (no
/// objects of that class at any time point) are left out; with several
/// classes left, an image's / well's / track's label names its class.
fn time_series(result: &TimeCourseResult, median: bool, classes: &[Class]) -> Vec<TimeSeries> {
    let mut series: Vec<((String, String, u32), TimeSeries)> = Vec::new();
    let mut index: std::collections::HashMap<(String, String, u32), usize> =
        std::collections::HashMap::new();
    for point in &result.points {
        let key = (point.line.clone(), point.image.clone(), point.class_id);
        let (center, low, high) = match median {
            true => (
                point.values.first().copied().flatten(),
                point.values.get(1).copied().flatten(),
                point.values.get(2).copied().flatten(),
            ),
            false => {
                let mean = point.values.first().copied().flatten();
                let sd = point.values.get(1).copied().flatten();
                (
                    mean,
                    mean.zip(sd).map(|(m, s)| m - s),
                    mean.zip(sd).map(|(m, s)| m + s),
                )
            }
        };
        if let Some(&at) = index.get(&key) {
            series[at].1.2.push((point.t, center, low, high));
            continue;
        }
        index.insert(key.clone(), series.len());
        let label = match (point.line.is_empty(), point.image.is_empty()) {
            (true, true) => point.class_name.clone(),
            (true, false) => point.image.clone(),
            (false, true) => point.line.clone(),
            (false, false) => format!("{} {}", point.line, point.image),
        };
        let color = match point.line.is_empty() && point.image.is_empty() {
            true => classes
                .iter()
                .find(|class| class.id == point.class_id)
                .map(|class| class.color)
                .filter(|color| *color != 0)
                .unwrap_or(CHART_TIME_PALETTE[0]),
            false => CHART_TIME_PALETTE[series.len() % CHART_TIME_PALETTE.len()],
        };
        series.push((key, (label, color, vec![(point.t, center, low, high)])));
    }
    series.retain(|(_, (_, _, points))| points.iter().any(|(_, center, _, _)| center.is_some()));
    let class_of = |key: &(String, String, u32)| key.2;
    let several_classes = series
        .iter()
        .any(|(key, _)| class_of(key) != series.first().map_or(0, |(k, _)| class_of(k)));
    series
        .into_iter()
        .map(|(key, (label, color, points))| {
            let of_class = key.0.is_empty() && key.1.is_empty();
            let label = match (
                several_classes && !of_class,
                classes.iter().find(|c| c.id == key.2),
            ) {
                (true, Some(class)) => format!("{label} · {}", class.name),
                _ => label,
            };
            (label, color, points)
        })
        .collect()
}

/// Path commands through `points` (x, y; `None` a gap), in the chart's
/// 1000 x 1000 box.
fn chart_path(points: impl Iterator<Item = Option<(f64, f64)>>) -> String {
    let mut path = String::new();
    let mut pen_down = false;
    for point in points {
        match point {
            Some((x, y)) => {
                path.push_str(&format!(
                    "{}{x:.1} {y:.1} ",
                    if pen_down { "L " } else { "M " }
                ));
                pen_down = true;
            }
            None => pen_down = false,
        }
    }
    path.trim_end().to_string()
}

/// The band between `low` and `high` - a closed shape per run of points
/// that have both.
fn chart_band(points: &[(f64, Option<f64>, Option<f64>)]) -> String {
    let mut band = String::new();
    let mut run: Vec<(f64, f64, f64)> = Vec::new();
    let close = |run: &mut Vec<(f64, f64, f64)>, band: &mut String| {
        if run.len() >= 2 {
            let up: Vec<String> = run
                .iter()
                .map(|(x, _, hi)| format!("{x:.1} {hi:.1}"))
                .collect();
            let down: Vec<String> = run
                .iter()
                .rev()
                .map(|(x, lo, _)| format!("{x:.1} {lo:.1}"))
                .collect();
            band.push_str(&format!("M {} L {} Z ", up.join(" L "), down.join(" L ")));
        }
        run.clear();
    };
    for (x, low, high) in points {
        match (low, high) {
            (Some(low), Some(high)) => run.push((*x, *low, *high)),
            _ => close(&mut run, &mut band),
        }
    }
    close(&mut run, &mut band);
    band.trim_end().to_string()
}

/// "Points drawn" (500 / 2 000 / 10 000 / all) as the scatter's limit.
fn scatter_points_limit(index: i32) -> Option<usize> {
    match index {
        0 => Some(500),
        2 => Some(10_000),
        3 => None,
        _ => Some(CHART_SCATTER_MAX_POINTS),
    }
}

#[derive(Default)]
struct MatrixFilter {
    pub object_classe: ObjectClass,
    pub column: Column,
    pub aggregation: Aggregation,
    pub color_schema: ColorSchema,
    pub color_scale: ColorScale,
    // Grouping, plate size and well layout: the project's plate settings,
    // kept in sync both ways (see `edit_plate_settings`/
    // `apply_project_plate_settings`).
    pub plate: PlateSettings,
    // The image heatmap's tile size - `None` keeps its default.
    pub square_size: Option<usize>,
}

pub struct ResultsStateController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) _app_state: Arc<UiState>,
    /// The open results database - on the server in remote mode.
    result_generator: Mutex<Option<Arc<dyn ResultsSource>>>,
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
    /// The scatter's points (0..1, as drawn) and their objects' image and
    /// box - for a click on a point.
    scatter_points: Mutex<Vec<(f32, f32, String, [u32; 4])>>,
    image_list_controller: Arc<ImagesListController>,
    // Whether the export dialog's defaults have already been seeded for the
    // currently-open database (see `populate_export_defaults`) — reset to
    // `false` on every `open_database` so a fresh database gets fresh
    // defaults the next time the dialog opens, but re-opening the dialog
    // against the *same* database leaves whatever the user already
    // configured alone. Most of the configuration needs no Rust-side
    // mirror: those dialog controls are two-way bound straight to
    // `ExportDialogState` in Slint, which retains it for the life of the
    // window — `on_export_start_clicked` reads it back from there directly.
    // The IMAGES picker is the exception, see `export_image_selection`.
    export_populated: Mutex<bool>,
    // The export dialog's IMAGES picker selection, by `rel_path`. Kept here
    // rather than in its Slint `image-items` because the picker's search
    // box narrows `image-items` down to the matching rows - reading the
    // selection back from there would silently drop every checked image the
    // current search hides.
    export_image_selection: Mutex<HashSet<PathBuf>>,
    // Set to a fresh flag whenever an export starts, so `on_cancel_clicked`
    // can signal the background export thread to stop at its next
    // per-image/per-class checkpoint (see `results_exporter::start_export`'s
    // `cancel` parameter). Cleared back to `None` once the export finishes
    // (successfully, with an error, or cancelled) — mirrors
    // `PipelinesController::pipeline_cancel_flag`'s pattern.
    export_cancel_flag: Mutex<Option<Arc<AtomicBool>>>,
    // Opening a database and toggling an image run here, off the UI thread.
    db_worker: Arc<DbWorker>,
    // The tab on screen (`ResultsState.rail-mode`), readable off the UI
    // thread. Only the visible tab queries the database: a hidden List or
    // Charts view is just marked stale (`list_stale`/`charts_stale`) and
    // refreshed when it's shown; Matrix always refreshes when it's shown.
    rail_mode: Mutex<ResultsRailMode>,
    list_stale: AtomicBool,
    charts_stale: AtomicBool,
    // Called after the results' plate view changed the project's plate settings,
    // so the project settings dialog shows them too (wired in editor.rs).
    plate_settings_written: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Results, split with the image - see [`split`].
    split: Mutex<split::SplitState>,
}

impl ResultsStateController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
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
            scatter_points: Mutex::new(Vec::new()),
            image_list_controller,
            export_populated: Mutex::new(false),
            export_image_selection: Mutex::new(HashSet::new()),
            export_cancel_flag: Mutex::new(None),
            db_worker: Arc::new(DbWorker::new()),
            rail_mode: Mutex::new(ResultsRailMode::List),
            list_stale: AtomicBool::new(false),
            charts_stale: AtomicBool::new(false),
            plate_settings_written: Mutex::new(None),
            split: Mutex::new(split::SplitState::default()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            self.attach_split_callbacks(&ui);

            // -------- ResultsState (api/results.slint) --------
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
                    manager.apply_matrix_aggregate_enabled(&ui_ready, MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    state.set_active_well_disabled(false);
                    *manager.current_well.lock().expect("Poisned") = None;
                    *manager.current_image.lock().expect("Poisned") = None;
                    manager.show_view(mode);
                    if mode == ResultsRailMode::Matrix {
                        manager.update_matrix_view();
                    }
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
                    manager.show_view(ResultsRailMode::List);
                    state.set_breadcrumb(ModelRc::from(Rc::new(VecModel::from(vec![
                        BreadcrumbItem {
                            label: "All results".into(),
                        },
                    ]))));
                    state.set_matrix_level(MatrixLevel::Plate);
                    manager.apply_matrix_aggregate_enabled(&ui_ready, MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    state.set_active_well_disabled(false);
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
                    manager.apply_matrix_aggregate_enabled(&ui_ready, MatrixLevel::Plate);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    state.set_active_well_disabled(false);
                    *manager.current_well.lock().expect("Poisned") = None;
                    *manager.current_image.lock().expect("Poisned") = None;
                    manager.update_matrix_view();
                } else if keep == 3 {
                    state.set_matrix_level(MatrixLevel::Well);
                    manager.apply_matrix_aggregate_enabled(&ui_ready, MatrixLevel::Well);
                    state.set_active_well("".into());
                    state.set_active_well_has_value(false);
                    state.set_active_well_value("".into());
                    state.set_active_well_disabled(false);
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
                .on_list_image_filter_changed(move |query| manager.filter_images(&query));

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
                        .map(|class| class.global())
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
            ui.global::<ResultsState>()
                .on_list_transpond_changed(move |enabled| {
                    manager.list_filter.lock().expect("Poisened").transpond = enabled;
                    // Rows and cursors mean something else in the other
                    // layout: start over at page 1.
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

            // A click in the scatter: the nearest point within 8 px - its
            // object on its image.
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_scatter_clicked(move |fx, fy, width, height| {
                    let hit = nearest_point(
                        &manager.scatter_points.lock().expect("Poisned"),
                        fx,
                        fy,
                        width,
                        height,
                        8.0,
                    );
                    let Some((rel_path, bbox_px)) = hit else {
                        return;
                    };
                    manager
                        .image_list_controller
                        .open_image_and_highlight_object(&PathBuf::from(&rel_path), bbox_px, false);
                    let name = std::path::Path::new(&rel_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or(rel_path);
                    manager.peek(-1, format!("Point in {name}"), false);
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
                    let rows = manager.list_row_locations.lock().expect("Poisned").len();
                    manager.peek(index, format!("Row {} of {rows}", index + 1), true);
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
                    manager.update_matrix_filter(|filter| filter.column = column);
                    let level = ui_ready.global::<ResultsState>().get_matrix_level();
                    manager.apply_matrix_aggregate_enabled(&ui_ready, level);
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
                    // Single-select: `InkMultiSelect.toggle()` only ever
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
                        .map(|class| class.global())
                    else {
                        warn!("Unknown class selected: {class_name}");
                        return;
                    };
                    manager.update_matrix_filter(|filter| filter.object_classe = object_class);
                    manager.refresh_active_matrix_view();
                });

            // Grouping, plate size and well size are the project's plate
            // settings: every change goes back to the project
            // (`edit_plate_settings`).
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_grouping_selected(move |key, selected| {
                    if !selected {
                        return;
                    }
                    let Some((_, mode)) = GROUPING_MODES.iter().find(|(name, _)| key == *name)
                    else {
                        warn!("Unknown grouping selected: {key}");
                        return;
                    };
                    // Switching to a custom regex starts from the pattern
                    // in use, rather than from nothing.
                    let in_use = manager.effective_grouping_regex();
                    manager.edit_plate_settings(|plate| {
                        if *mode == GroupingMode::Custom && plate.grouping_regex.trim().is_empty() {
                            plate.grouping_regex = in_use.unwrap_or_default();
                        }
                        plate.grouping_mode = *mode;
                    });
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_well_image_moved(move |from, to| {
                    let (Ok(from), Ok(to)) = (usize::try_from(from), usize::try_from(to)) else {
                        return;
                    };
                    let Some(ui) = ui_weak.upgrade() else {
                        return;
                    };
                    let state = ui.global::<ResultsState>();
                    let shown = (
                        state.get_well_rows().max(1) as u32,
                        state.get_well_cols().max(1) as u32,
                    );
                    manager.move_well_image(from, to, shown);
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_regex_changed(move |regex| {
                    manager.edit_plate_settings(|plate| {
                        plate.grouping_mode = GroupingMode::Custom;
                        plate.grouping_regex = regex.to_string();
                    });
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
                    let Some(size) = plate_size_from_key(key.as_str()) else {
                        warn!("Unknown plate size selected: {key}");
                        return;
                    };
                    manager.edit_plate_settings(|plate| plate.plate_size = size);
                });

            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_matrix_well_layout_selected(move |key, selected| {
                    if !selected {
                        return;
                    }
                    let layout = if key == WELL_LAYOUT_AUTO {
                        WellLayout::Auto
                    } else {
                        // Custom: the grid now shown, as a start.
                        let shown = ui_weak.upgrade().map(|ui| {
                            let state = ui.global::<ResultsState>();
                            (state.get_well_rows(), state.get_well_cols())
                        });
                        let (rows, cols) = shown.unwrap_or((4, 4));
                        WellLayout::Fixed {
                            rows: rows.max(1) as u32,
                            cols: cols.max(1) as u32,
                        }
                    };
                    manager.edit_plate_settings(|plate| {
                        if !matches!(
                            (plate.well_layout, layout),
                            (WellLayout::Fixed { .. }, WellLayout::Fixed { .. })
                        ) {
                            plate.set_well_layout(layout);
                        }
                    });
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_well_rows_changed(move |value| {
                    let Ok(rows) = value.trim().parse::<u32>() else {
                        warn!("Invalid well rows value: {value}");
                        return;
                    };
                    manager.edit_plate_settings(|plate| {
                        let cols = match plate.well_layout {
                            WellLayout::Fixed { cols, .. } => cols,
                            WellLayout::Auto => 4,
                        };
                        plate.set_well_layout(WellLayout::Fixed {
                            rows: rows.max(1),
                            cols,
                        });
                    });
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_matrix_well_cols_changed(move |value| {
                    let Ok(cols) = value.trim().parse::<u32>() else {
                        warn!("Invalid well cols value: {value}");
                        return;
                    };
                    manager.edit_plate_settings(|plate| {
                        let rows = match plate.well_layout {
                            WellLayout::Fixed { rows, .. } => rows,
                            WellLayout::Auto => 4,
                        };
                        plate.set_well_layout(WellLayout::Fixed {
                            rows,
                            cols: cols.max(1),
                        });
                    });
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
                    let Some(cell) = cells.get(key.as_str()).cloned() else {
                        warn!("Unknown well clicked: {key}");
                        return;
                    };
                    drop(cells);
                    let state = ui_ready.global::<ResultsState>();
                    // The card lists the well's images.
                    let images: Vec<MatrixCell> = manager
                        .query_well(&key)
                        .map(|(result, ..)| {
                            flatten_grid_cells(&result)
                                .into_iter()
                                .filter(|image| image.exists)
                                .collect()
                        })
                        .unwrap_or_default();
                    let names: Vec<String> =
                        images.iter().map(|image| image.key.to_string()).collect();
                    state.set_active_well_images(ModelRc::from(Rc::new(VecModel::from(images))));
                    state.set_active_well(key);
                    // Split: the well's images in the image pane.
                    if state.get_split() {
                        manager.split_show_well(names);
                    }
                    // `exists` (not `has_value`): the user must be able to
                    // select/open a well or field even when it has no
                    // number to show (an image with zero detected
                    // objects) - this drives both the detail card's
                    // "—"-vs-value fallback and the "Open well/Image"
                    // button's `enabled`.
                    state.set_active_well_has_value(cell.exists);
                    state.set_active_well_value(cell.label.clone());
                    state.set_active_well_disabled(cell.disabled);
                });

            // A row of the well card's images: leave it out / take it in.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_toggle_image_disabled(move |image_name, disable| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    let rel_path = manager
                        .images
                        .lock()
                        .expect("Poisened")
                        .iter()
                        .find(|image| image.name == image_name.as_str())
                        .map(|image| image.rel_path.to_string_lossy().into_owned());
                    let Some(rel_path) = rel_path else {
                        warn!("Unknown image to toggle: {image_name}");
                        return;
                    };
                    // The status bar's "1 disabled".
                    {
                        let mut images = manager.images.lock().expect("Poisened");
                        if let Some(image) = images
                            .iter_mut()
                            .find(|image| image.name == image_name.as_str())
                        {
                            image.disabled = disable;
                        }
                        state.set_analysis_summary(analysis_summary(&images).into());
                    }
                    // Shown right away.
                    let rows = state.get_active_well_images();
                    for i in 0..rows.row_count() {
                        if let Some(mut row) = rows.row_data(i)
                            && row.key == image_name
                        {
                            row.disabled = disable;
                            rows.set_row_data(i, row);
                        }
                    }
                    let well = state.get_active_well().to_string();
                    manager.run_db_job(move |manager| {
                        let toggled = {
                            let guard = manager.result_generator.lock().expect("Poisned");
                            match guard.as_ref() {
                                None => Err("No database opened!".to_string()),
                                Some(db) => db.enable_image(&rel_path, disable).map_err(|err| {
                                    format!("Could not toggle image {rel_path}: {err}")
                                }),
                            }
                        };
                        if let Err(err) = toggled {
                            error!("{err}");
                        }
                        // The plate and the well's images, from the database.
                        manager.refresh_matrix_view_at(MatrixLevel::Plate);
                        let images: Vec<MatrixCell> = manager
                            .query_well(&well)
                            .map(|(result, ..)| {
                                flatten_grid_cells(&result)
                                    .into_iter()
                                    .filter(|image| image.exists)
                                    .collect()
                            })
                            .unwrap_or_default();
                        // The plate's refresh closed the card: open it again,
                        // with the well's new value.
                        let cell = manager
                            .matrix_cells
                            .lock()
                            .expect("Poisned")
                            .get(well.as_str())
                            .cloned();
                        let ui_weak = manager.ui.clone();
                        crate::helper::ui_thread::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak.upgrade() {
                                let state = ui.global::<ResultsState>();
                                if let Some(cell) = cell {
                                    state.set_active_well_has_value(cell.exists);
                                    state.set_active_well_value(cell.label.clone());
                                }
                                state.set_active_well(well.into());
                                state.set_active_well_images(ModelRc::from(Rc::new(
                                    VecModel::from(images),
                                )));
                            }
                        })
                        .ok();
                    });
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
                    let Some(cell) = cells.get(key.as_str()).cloned() else {
                        warn!("Unknown field clicked: {key}");
                        return;
                    };
                    drop(cells);
                    let state = ui_ready.global::<ResultsState>();
                    // Split with the image: this image, right away.
                    if state.get_split() {
                        manager.split_show_image_named(&key);
                    }
                    state.set_active_well(key);
                    // `exists` (not `has_value`): the user must be able to
                    // select/open a well or field even when it has no
                    // number to show (an image with zero detected
                    // objects) - this drives both the detail card's
                    // "—"-vs-value fallback and the "Open well/Image"
                    // button's `enabled`.
                    state.set_active_well_has_value(cell.exists);
                    state.set_active_well_value(cell.label.clone());
                    state.set_active_well_disabled(cell.disabled);
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
                        manager.apply_matrix_aggregate_enabled(&ui_ready, MatrixLevel::Object);
                    } else {
                        warn!("Failed to upgrade UI handle in on_well_field_clicked");
                    }
                    *manager.current_image.lock().expect("Poisned") = Some(rel_path.clone());
                    manager.update_image_heatmap_view(&rel_path);
                });

            // Toggles the selected field's image disabled/enabled (see
            // `ResultsGenerator::enable_image`), then refreshes the well
            // grid so the heatmap reflects it immediately. A single image's
            // own aggregate is never affected by its own disabled flag (see
            // `get_group_by_well`), so `active-well-value`/`-has-value`
            // don't need updating here - only `-disabled` (drives the
            // detail card's crossed-out label) does, set synchronously
            // right away rather than only through the refresh below, since
            // `set_well_in_slint` (inside `refresh_active_matrix_view`)
            // always resets `active-well` to "" as part of that refresh via
            // a queued `invoke_from_event_loop` call - which the headless
            // test platform never drains, and which would otherwise close
            // the detail card for a moment even in a real app. The second
            // `invoke_from_event_loop` below re-queues behind that reset
            // (Slint runs queued UI updates in the order they were queued)
            // to restore the selection once it does drain, in a real app.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ResultsState>()
                .on_toggle_active_well_disabled(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_toggle_active_well_disabled");
                        return;
                    };
                    let state = ui_ready.global::<ResultsState>();
                    let image_name = state.get_active_well();
                    if image_name.is_empty() {
                        return;
                    }
                    let rel_path = manager
                        .images
                        .lock()
                        .expect("Poisened")
                        .iter()
                        .find(|image| image.name == image_name.as_str())
                        .map(|image| image.rel_path.to_string_lossy().into_owned());
                    let Some(rel_path) = rel_path else {
                        warn!("Unknown image to toggle: {image_name}");
                        return;
                    };
                    let disable = !state.get_active_well_disabled();
                    // Shown right away; the write and the refresh run on the
                    // database thread.
                    state.set_active_well_disabled(disable);
                    let level = state.get_matrix_level();
                    manager.run_db_job(move |manager| {
                        let toggled = {
                            let guard = manager.result_generator.lock().expect("Poisned");
                            match guard.as_ref() {
                                None => Err("No database opened!".to_string()),
                                Some(db) => db
                                    .enable_image(&rel_path, disable)
                                    .map_err(|err| format!("Could not toggle image {rel_path}: {err}")),
                            }
                        };
                        if let Err(err) = toggled {
                            error!("{err}");
                            let ui_weak = manager.ui.clone();
                            crate::helper::ui_thread::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.global::<ResultsState>().set_active_well_disabled(!disable);
                                }
                            })
                            .ok();
                            return;
                        }
                        manager.refresh_matrix_view_at(level);

                        let manager = manager.clone();
                        crate::helper::ui_thread::invoke_from_event_loop(move || {
                            let Some(ui_ready) = manager.ui.upgrade() else {
                                warn!(
                                    "Failed to upgrade UI handle re-selecting field after toggling disabled"
                                );
                                return;
                            };
                            let state = ui_ready.global::<ResultsState>();
                            let cells = manager.well_cells.lock().expect("Poisned");
                            if let Some(cell) = cells.get(image_name.as_str()) {
                                state.set_active_well(image_name.clone());
                                state.set_active_well_has_value(cell.exists);
                                state.set_active_well_value(cell.label.clone());
                                state.set_active_well_disabled(cell.disabled);
                            }
                        })
                        .ok();
                    });
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
                    let Some(cell) = cells.get(key.as_str()).cloned() else {
                        warn!("Unknown heatmap tile clicked: {key}");
                        return;
                    };
                    drop(cells);
                    let state = ui_ready.global::<ResultsState>();
                    // Split with the image: the tile on it, right away.
                    if state.get_split() {
                        manager.split_show_tile(&key);
                    }
                    state.set_active_well(key.clone());
                    // `exists` (not `has_value`): the user must be able to
                    // select/open a well or field even when it has no
                    // number to show (an image with zero detected
                    // objects) - this drives both the detail card's
                    // "—"-vs-value fallback and the "Open well/Image"
                    // button's `enabled`.
                    state.set_active_well_has_value(cell.exists);
                    state.set_active_well_value(cell.label.clone());
                    state.set_active_well_disabled(cell.disabled);
                });

            // "Show on image" (or a double click): the tile on its image.
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_image_heatmap_cell_open(move |key| {
                    // Split with the image: shown there (the split stays).
                    if manager
                        .ui
                        .upgrade()
                        .is_some_and(|ui| ui.global::<ResultsState>().get_split())
                    {
                        manager.split_show_tile(&key);
                        return;
                    }
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
                    let Some(bbox_px) = manager.tile_bbox(&key) else {
                        warn!("Unrecognized heatmap tile key: {key}");
                        return;
                    };
                    manager
                        .image_list_controller
                        .open_image_and_highlight_object(&PathBuf::from(rel_path), bbox_px, true);
                    manager.peek(-1, format!("Tile {key}"), false);
                });
            ui.global::<ResultsState>()
                .on_object_marker_clicked(move |_id| {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_plate(move || {});
            ui.global::<ResultsState>()
                .on_matrix_back_to_well(move || {});

            // -- Charts --
            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_kind_selected(move |kind| {
                    let kind = match kind {
                        ResultsChartKind2::Histogram => ChartKind::Histogram,
                        ResultsChartKind2::Scatter => ChartKind::Scatter,
                        ResultsChartKind2::Boxplot => ChartKind::Boxplot,
                        ResultsChartKind2::OverTime => ChartKind::OverTime,
                    };
                    manager.chart_filter.lock().expect("Poisened").kind = kind;
                    manager.refresh_charts();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_display_changed(move || {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let state = ui.global::<ResultsState>();
                    {
                        let mut filter = manager.chart_filter.lock().expect("Poisened");
                        filter.bins = state.get_chart_bins().clamp(5, 200) as usize;
                        filter.log_counts = state.get_chart_log_counts();
                        filter.max_points = scatter_points_limit(state.get_chart_points_index());
                    }
                    manager.refresh_charts();
                });

            let manager = self.clone();
            ui.global::<ResultsState>().on_chart_time_changed(move || {
                let Some(ui) = manager.ui.upgrade() else {
                    return;
                };
                let state = ui.global::<ResultsState>();
                {
                    let mut filter = manager.chart_filter.lock().expect("Poisened");
                    filter.time_lines = time_lines_of(state.get_chart_time_lines());
                    filter.time_median = state.get_chart_time_summary() == 1;
                    filter.time_relative = state.get_chart_time_relative();
                    filter.time_tracks = state.get_chart_time_tracks();
                }
                manager.refresh_charts();
            });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_time_export(move || manager.export_chart_time_course());

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_group_selected(move |group| {
                    manager.chart_filter.lock().expect("Poisened").group_by = match group {
                        1 => BoxplotGroup::Image,
                        2 => BoxplotGroup::Well,
                        3 => BoxplotGroup::WellRow,
                        4 => BoxplotGroup::WellColumn,
                        _ => BoxplotGroup::Class,
                    };
                    manager.refresh_charts();
                });

            let manager = self.clone();
            ui.global::<ResultsState>()
                .on_chart_box_clicked(move |label| manager.show_box_in_list(&label));

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
                        .map(|class| class.global())
                    else {
                        warn!("Unknown class selected: {class_name}");
                        return;
                    };
                    manager.chart_filter.lock().expect("Poisened").object_class =
                        Some(object_class);
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
            let browser = self._app_state.file_browser.clone();
            ui.global::<ExportDialogState>()
                .on_pick_output_dir(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_pick_output_dir");
                        return;
                    };
                    let current = ui_ready.global::<ExportDialogState>().get_output_dir();
                    let request = crate::FileRequest::open_folder("Choose export folder")
                        .start_in(current.as_str());
                    let ui_weak = ui_weak.clone();
                    browser.open(request, move |path| {
                        let (Some(path), Some(ui_ready)) = (path, ui_weak.upgrade()) else {
                            return;
                        };
                        ui_ready
                            .global::<ExportDialogState>()
                            .set_output_dir(path.to_string_lossy().into_owned().into());
                    });
                });

            // The IMAGES picker: every callback updates
            // `export_image_selection` (the real selection) and re-pushes
            // the rows matching the current search, see
            // `push_export_image_items`.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_image_item_selected(move |key, selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_item_selected");
                        return;
                    };
                    let rel_path = PathBuf::from(key.as_str());
                    {
                        let mut selection =
                            manager.export_image_selection.lock().expect("Poisened");
                        if selected {
                            selection.insert(rel_path);
                        } else {
                            selection.remove(&rel_path);
                        }
                    }
                    manager.push_export_image_summary(&ui_ready);
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_image_select_all(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_select_all");
                        return;
                    };
                    let all: HashSet<PathBuf> = manager
                        .images
                        .lock()
                        .expect("Poisened")
                        .iter()
                        .map(|image| image.rel_path.clone())
                        .collect();
                    *manager.export_image_selection.lock().expect("Poisened") = all;
                    manager.push_export_image_items(&ui_ready);
                });

            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_image_select_none(move || {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_select_none");
                        return;
                    };
                    manager
                        .export_image_selection
                        .lock()
                        .expect("Poisened")
                        .clear();
                    manager.push_export_image_items(&ui_ready);
                });

            // Search box in the export dialog's own IMAGES picker - only
            // narrows which rows show, never the selection.
            let ui_weak = self.ui.clone();
            let manager = self.clone();
            ui.global::<ExportDialogState>()
                .on_image_filter_changed(move |query| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        warn!("Failed to upgrade UI handle in on_image_filter_changed");
                        return;
                    };
                    ui_ready
                        .global::<ExportDialogState>()
                        .set_image_filter_text(query);
                    manager.push_export_image_items(&ui_ready);
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

            let ui_weak = self.ui.clone();
            ui.global::<ExportDialogState>()
                .on_time_stat_item_selected(move |_key, _selected| {
                    let Some(ui_ready) = ui_weak.upgrade() else {
                        return;
                    };
                    let state = ui_ready.global::<ExportDialogState>();
                    let selected = state
                        .get_time_stat_items()
                        .iter()
                        .filter(|i| i.selected)
                        .count();
                    state.set_time_stat_summary(list_summary(
                        selected,
                        TIME_STATS.len(),
                        "Statistics",
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

            let manager = self.clone();
            ui.global::<ExportDialogState>().on_cancel_clicked(move || {
                if let Some(flag) = manager
                    .export_cancel_flag
                    .lock()
                    .expect("Poisened")
                    .as_ref()
                {
                    flag.store(true, Ordering::Relaxed);
                }
            });
        }
    }

    /// `f` runs whenever the results' plate view changed the project's plate
    /// settings.
    pub fn on_plate_settings_written(&self, f: impl Fn() + Send + Sync + 'static) {
        *self.plate_settings_written.lock().expect("Poisoned") = Some(Box::new(f));
    }

    /// The plate settings the Matrix view uses: the project's, as last
    /// synced.
    fn current_plate_settings(&self) -> PlateSettings {
        match self.matrix_filter.lock().expect("Poisoned").as_ref() {
            Some(filter) => filter.plate.clone(),
            None => self._app_state.get_project().plate.clone(),
        }
    }

    /// The regex the current grouping uses on the open database - for Auto
    /// the pattern detected.
    fn effective_grouping_regex(&self) -> Option<String> {
        let grouping = grouping_of(&self.current_plate_settings());
        let guard = self.result_generator.lock().expect("Poisoned");
        let db = guard.as_ref()?;
        db.grouping_regex(&grouping)
            .inspect_err(|err| warn!("Could not resolve the grouping: {err}"))
            .ok()
    }

    /// Changes the plate settings from the results' plate view: in the Matrix
    /// view, in the project (marked unsaved, and shown in the project
    /// settings) and on screen.
    fn edit_plate_settings(&self, edit: impl FnOnce(&mut PlateSettings)) {
        let plate = {
            let mut guard = self.matrix_filter.lock().expect("Poisoned");
            let filter = guard.get_or_insert_with(MatrixFilter::default);
            let before = filter.plate.clone();
            edit(&mut filter.plate);
            if filter.plate == before {
                return;
            }
            filter.plate.clone()
        };
        let changed_project = {
            let mut project = self._app_state.get_project_write();
            let changed = project.plate != plate;
            project.plate = plate;
            changed
        };
        if changed_project {
            self._app_state.mark_dirty();
            if let Some(written) = self
                .plate_settings_written
                .lock()
                .expect("Poisoned")
                .as_ref()
            {
                written();
            }
        }
        self.push_plate_settings_in_slint();
        self.refresh_active_matrix_view();
    }

    /// Drag and drop in the well view: the images at grid positions `from`
    /// and `to` (row by row) swap places in the well's image order. A well
    /// fitted automatically becomes a fixed grid of the size `shown`, in
    /// number order - what Auto showed - first.
    fn move_well_image(&self, from: usize, to: usize, shown: (u32, u32)) {
        if from == to {
            return;
        }
        self.edit_plate_settings(|plate| {
            if plate.well_layout == WellLayout::Auto {
                let (rows, cols) = shown;
                plate.well_image_order = PlateSettings::default_image_order(rows, cols);
                plate.set_well_layout(WellLayout::Fixed { rows, cols });
            }
            let order = &mut plate.well_image_order;
            if from < order.len() && to < order.len() {
                order.swap(from, to);
            }
        });
    }

    /// The project's plate settings changed elsewhere (project settings
    /// dialog, undo, another project): show and use them.
    pub fn apply_project_plate_settings(&self) {
        let plate = self._app_state.get_project().plate.clone();
        let changed = match self.matrix_filter.lock().expect("Poisoned").as_mut() {
            Some(filter) if filter.plate != plate => {
                filter.plate = plate;
                true
            }
            _ => false,
        };
        if changed {
            self.push_plate_settings_in_slint();
            self.refresh_active_matrix_view();
        }
    }

    /// Shows the current plate settings in the Matrix toolbar: grouping
    /// (with the regex in use), plate size and well size.
    fn push_plate_settings_in_slint(&self) {
        let plate = self.current_plate_settings();
        let in_use = self.effective_grouping_regex().unwrap_or_default();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let summary = |items: &[MultiSelectItem]| {
                items
                    .iter()
                    .find(|item| item.selected)
                    .map(|item| item.value.clone())
                    .unwrap_or_default()
            };
            let grouping = grouping_items(plate.grouping_mode);
            state.set_matrix_grouping_summary(summary(&grouping));
            state.set_matrix_grouping_items(ModelRc::from(Rc::new(VecModel::from(grouping))));
            state.set_matrix_group_regex_custom(plate.grouping_mode == GroupingMode::Custom);
            if state.get_matrix_group_regex() != plate.grouping_regex.as_str() {
                state.set_matrix_group_regex(plate.grouping_regex.as_str().into());
            }
            state.set_matrix_group_regex_in_use(in_use.into());

            let sizes = plate_size_items(plate.plate_size);
            state.set_matrix_plate_size_summary(summary(&sizes));
            state.set_matrix_plate_size_items(ModelRc::from(Rc::new(VecModel::from(sizes))));

            let layouts = well_layout_items(plate.well_layout);
            state.set_matrix_well_layout_summary(summary(&layouts));
            state.set_matrix_well_layout_items(ModelRc::from(Rc::new(VecModel::from(layouts))));
            if let WellLayout::Fixed { rows, cols } = plate.well_layout {
                state.set_matrix_well_rows(rows.to_string().into());
                state.set_matrix_well_cols(cols.to_string().into());
            }
        })
        .ok();
    }

    /// Runs `job` on the database thread (see [`DbWorker`]), with
    /// `ResultsState.loading` set while any job is queued or running.
    fn run_db_job(self: &Arc<Self>, job: impl FnOnce(&Arc<Self>) + Send + 'static) {
        /// Counts the job as done even if it panics.
        struct Done(Arc<ResultsStateController>);
        impl Drop for Done {
            fn drop(&mut self) {
                if self.0.db_worker.pending.fetch_sub(1, Ordering::SeqCst) == 1 {
                    self.0.set_loading_in_slint(false);
                }
            }
        }
        if self.db_worker.pending.fetch_add(1, Ordering::SeqCst) == 0 {
            self.set_loading_in_slint(true);
        }
        let done = Done(self.clone());
        self.db_worker.submit(Box::new(move || {
            job(&done.0);
            drop(done);
        }));
    }

    /// The image heatmap's tile `key` ("R0C0") in image pixels
    /// (`[xmin, ymin, xmax, ymax]`, inclusive).
    fn tile_bbox(&self, key: &str) -> Option<[u32; 4]> {
        let (row, col) = parse_tile_key(key)?;
        let square_size = self
            .matrix_filter
            .lock()
            .expect("Poisned")
            .as_ref()
            .and_then(|filter| filter.square_size)
            .unwrap_or(DEFAULT_SQUARE_SIZE) as u32;
        let (xmin, ymin) = (col * square_size, row * square_size);
        Some([xmin, ymin, xmin + square_size - 1, ymin + square_size - 1])
    }

    /// A result's object shown on its image (`row`: the List row, -1 none):
    /// the window goes to the image, a bar on it says which (`label`) and
    /// leads back - `can_step`: ‹ › to the next row's (see `ResultsPeek`).
    fn peek(&self, row: i32, label: String, can_step: bool) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                state.set_list_selected_row(row);
                state.set_peek_label(label.into());
                state.set_peek_can_step(can_step);
                state.set_peek(true);
                ui.global::<crate::LayoutState>()
                    .set_mode(crate::WorkspaceMode::Images);
            }
        })
        .ok();
    }

    fn set_database_name_in_slint(&self, name: String) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_database_name(name.into());
            }
        })
        .ok();
    }

    fn set_loading_in_slint(&self, loading: bool) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_loading(loading);
            }
        })
        .ok();
    }

    /// Whether `mode` is the tab on screen. If not, its view is only marked
    /// `stale` - refreshed once the tab is shown - so a hidden view never
    /// queries the database.
    fn is_view_shown(&self, mode: ResultsRailMode, stale: &AtomicBool) -> bool {
        let shown = *self.rail_mode.lock().expect("Poisoned") == mode;
        if !shown {
            stale.store(true, Ordering::SeqCst);
        }
        shown
    }

    /// `mode` became the tab on screen: catch up on what changed while it
    /// was hidden.
    fn show_view(self: &Arc<Self>, mode: ResultsRailMode) {
        *self.rail_mode.lock().expect("Poisoned") = mode;
        let stale = match mode {
            ResultsRailMode::List => &self.list_stale,
            ResultsRailMode::Charts => &self.charts_stale,
            // Always refreshed when shown (`on_rail_mode_selected`).
            _ => return,
        };
        if stale.swap(false, Ordering::SeqCst) {
            self.run_db_job(move |manager| match mode {
                ResultsRailMode::List => manager.update_list_view(),
                _ => manager.refresh_charts(),
            });
        }
    }

    /// The Aggregate picker means nothing for `Column::Count` below the
    /// Plate level: a field (Well level) is a single image and a heatmap
    /// tile (Object level) is a region of a single image, so the count is
    /// just the count. At the Plate level Count is aggregated over the
    /// per-image counts of a well, so the picker stays enabled there.
    fn apply_matrix_aggregate_enabled(&self, ui: &AppWindow, level: MatrixLevel) {
        let is_count = self
            .matrix_filter
            .lock()
            .expect("Poisned")
            .as_ref()
            .is_some_and(|filter| matches!(filter.column, Column::Count));
        let below_plate = matches!(level, MatrixLevel::Well | MatrixLevel::Object);
        ui.global::<ResultsState>()
            .set_matrix_aggregate_enabled(!(is_count && below_plate));
    }

    /// Opens the results database at `path` on the database thread (see
    /// [`DbWorker`]); the window stays responsive meanwhile.
    pub fn open_database(self: &Arc<Self>, path: PathBuf) {
        self.run_db_job(move |manager| manager.open_database_now(path));
    }

    fn open_database_now(&self, path: PathBuf) {
        info!("Opening database {:?}", path);
        let db = self._app_state.backend().open_results(&path);
        match db {
            Ok(results) => {
                *self.export_populated.lock().expect("Poisened") = false;
                // Another analysis: nothing of the last one's split.
                *self.split.lock().expect("Poisoned") = split::SplitState::default();
                let run_warning = match results.run_status() {
                    Ok(status) => status.warning().unwrap_or_default(),
                    Err(err) => {
                        error!("Could not read how the analysis ended: {err}");
                        String::new()
                    }
                };
                if !run_warning.is_empty() {
                    warn!("{}: {run_warning}", path.display());
                }
                self.set_run_warning_in_slint(run_warning);
                self.set_database_name_in_slint(
                    path.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                );
                let mut default_object_classes = Vec::new();
                match results.get_object_classes() {
                    Ok(classes) => {
                        default_object_classes =
                            classes.iter().map(|class| class.global()).collect();
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
                        // The images panel's plate marks these as analysed.
                        self._app_state
                            .set_analysed(images.iter().map(|image| image.rel_path.clone()));
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
                    transpond: false,
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
                {
                    // Another analysis: what it shows anew, how it's drawn
                    // (bins, log, points) stays.
                    let mut filter = self.chart_filter.lock().expect("Poisened");
                    *filter = ChartFilter {
                        kind: ChartKind::Histogram,
                        column: default_chart_column.clone(),
                        y_column: default_chart_column.clone(),
                        object_class: None,
                        group_by: BoxplotGroup::Class,
                        ..filter.clone()
                    };
                }
                let chart_column_items_vec = chart_column_items(
                    &available_columns,
                    &classes_for_charts,
                    &default_chart_column,
                );
                let default_chart_column_label =
                    default_chart_column.display_label(&classes_for_charts);

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
                // Grouping, plate size and well layout start from the
                // project's plate settings.
                *self.matrix_filter.lock().expect("Poisned") = Some(MatrixFilter {
                    column: default_matrix_column.clone(),
                    plate: self._app_state.get_project().plate.clone(),
                    ..MatrixFilter::default()
                });
                let matrix_eligible_columns: Vec<ColumnEntry> = available_columns
                    .iter()
                    .filter(|entry| is_matrix_column(&entry.key))
                    .cloned()
                    .collect();
                let matrix_column_items_vec =
                    column_items(&matrix_eligible_columns, &classes_for_charts, |key| {
                        *key == default_matrix_column
                    });
                let default_matrix_column_label =
                    default_matrix_column.display_label(&classes_for_charts);

                let ui_weak = self.ui.clone();
                crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui_ready) = ui_weak.upgrade() {
                        let state = ui_ready.global::<ResultsState>();
                        state.set_list_with_coloc_details(false);
                        state.set_list_transpond(false);
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
                        state.set_matrix_aggregate_enabled(true);

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
                self.push_plate_settings_in_slint();
                // Only the tab on screen queries the database; the others
                // catch up when they're shown.
                self.refresh_list();
                self.update_matrix_view();
                self.refresh_charts();
            }
            Err(err) => {
                error!("{}", err);
            }
        }
    }

    /// Shows the results: the main window's Results mode.
    pub fn show_results_window(&self) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<crate::LayoutState>()
                    .set_mode(crate::WorkspaceMode::Results);
            } else {
                warn!("Failed to upgrade UI handle, cannot show the results");
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
        if !self.is_view_shown(ResultsRailMode::Charts, &self.charts_stale) {
            return;
        }
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            return;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = evanalyzer_app::results::PlaneFilter {
            z_stack: plane_filter.selected_z_stack,
            t_stack: plane_filter.selected_t_stack,
        };
        drop(plane_filter);

        let chart_filter = self.chart_filter.lock().expect("Poisened").clone();
        let object_classes = chart_filter.object_class.map(|class| vec![class]);

        match chart_filter.kind {
            ChartKind::Histogram => {
                match db.histogram(&HistogramFilter {
                    plane,
                    images: None,
                    object_classes,
                    column: chart_filter.column,
                    bins: chart_filter.bins.max(1),
                }) {
                    Ok(histogram) => {
                        self.push_histogram_in_slint(&histogram, chart_filter.log_counts)
                    }
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
            ChartKind::Scatter => {
                match db.scatter(&ScatterFilter {
                    plane,
                    images: None,
                    object_classes,
                    x_column: chart_filter.column,
                    y_column: chart_filter.y_column,
                    max_points: chart_filter.max_points,
                }) {
                    Ok(scatter) => self.push_scatter_in_slint(&scatter),
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
            ChartKind::Boxplot => {
                match db.boxplot(&BoxplotFilter {
                    plane,
                    images: None,
                    // Boxplot always groups by every class present —
                    // the CLASS dropdown doesn't gate it the way it
                    // does Histogram/Scatter (see `ChartFilter::object_class`'s
                    // doc comment).
                    object_classes: None,
                    column: chart_filter.column,
                    group_by: chart_filter.group_by,
                    grouping: self.current_grouping(),
                }) {
                    Ok(boxplot) => self.push_boxplot_in_slint(&boxplot.boxes),
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
            ChartKind::OverTime => {
                // The z-plane shown; every time point. A line per class:
                // every class - else the class picked.
                let filter = |lines, statistics: Vec<TimeCourseStat>| TimeCourseFilter {
                    lines,
                    grouping: self.current_grouping(),
                    columns: vec![chart_filter.column.clone()],
                    statistics,
                    normalize: match chart_filter.time_relative {
                        true => TimeCourseNormalize::RelativeToStart,
                        false => TimeCourseNormalize::None,
                    },
                    images: None,
                    object_classes: match lines {
                        TimeCourseLines::Class => None,
                        _ => object_classes.clone(),
                    },
                    t_stacks: None,
                    z_stacks: Some(std::range::Range {
                        start: plane.z_stack,
                        end: plane.z_stack + 1,
                    }),
                    // The faint tracks: only as many as are drawn.
                    max_tracks: Some(CHART_TIME_MAX_TRACKS),
                };
                let statistics = match chart_filter.time_median {
                    true => vec![
                        TimeCourseStat::Median,
                        TimeCourseStat::Q1,
                        TimeCourseStat::Q3,
                    ],
                    false => TimeCourseStat::DEFAULT.to_vec(),
                };
                let result = db.time_course(&filter(chart_filter.time_lines, statistics));
                let tracks = match chart_filter.time_tracks {
                    true => db
                        .time_course(&TimeCourseFilter {
                            // The tracks of the lines' class(es).
                            object_classes: object_classes.clone(),
                            ..filter(TimeCourseLines::Track, vec![TimeCourseStat::Mean])
                        })
                        .ok(),
                    false => None,
                };
                match result {
                    Ok(result) => {
                        self.push_time_course_in_slint(&result, tracks.as_ref(), &chart_filter)
                    }
                    Err(err) => self.push_chart_error(err.to_string()),
                }
            }
        }
    }

    /// The "Over time" chart: lines, bands, faint tracks as paths in a
    /// 1000 x 1000 box, the axes' ends, the legend.
    fn push_time_course_in_slint(
        &self,
        result: &TimeCourseResult,
        tracks: Option<&TimeCourseResult>,
        filter: &ChartFilter,
    ) {
        let classes = self.classes.lock().expect("Poisened").clone();
        let series = time_series(result, filter.time_median, &classes);
        let faint = tracks
            .map(|tracks| time_series(tracks, false, &classes))
            .unwrap_or_default();
        let times: Vec<u32> = result.points.iter().map(|p| p.t).collect();
        let (Some(t_min), Some(t_max)) = (times.iter().min().copied(), times.iter().max().copied())
        else {
            self.push_chart_error(
                "Nothing to show over time: no objects of this class.".to_string(),
            );
            return;
        };
        if t_min == t_max {
            self.push_chart_error(
                "Over time needs results with several time points - these have one.".to_string(),
            );
            return;
        }
        let shown = &series[..series.len().min(CHART_TIME_MAX_LINES)];
        let faint = &faint[..faint.len().min(CHART_TIME_MAX_TRACKS)];
        let values = shown
            .iter()
            .flat_map(|(_, _, points)| points.iter().flat_map(|(_, c, l, h)| [*c, *l, *h]))
            .chain(
                faint
                    .iter()
                    .flat_map(|(_, _, points)| points.iter().map(|(_, c, _, _)| *c)),
            )
            .flatten();
        let (mut low, mut high) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
        if !low.is_finite() {
            (low, high) = (0.0, 1.0);
        }
        if high <= low {
            (low, high) = (low - 1.0, high + 1.0);
        }
        let pad = (high - low) * 0.05;
        let (low, high) = (low - pad, high + pad);
        let x = |t: u32| (t - t_min) as f64 / (t_max - t_min) as f64 * 1000.0;
        let y = |v: f64| 1000.0 - (v - low) / (high - low) * 1000.0;
        let lines: Vec<(String, u32, String, String)> = shown
            .iter()
            .map(|(label, color, points)| {
                let line = chart_path(points.iter().map(|(t, c, _, _)| c.map(|c| (x(*t), y(c)))));
                let band = chart_band(
                    &points
                        .iter()
                        .map(|(t, _, l, h)| (x(*t), l.map(y), h.map(y)))
                        .collect::<Vec<_>>(),
                );
                (label.clone(), *color, line, band)
            })
            .collect();
        let faint_paths: Vec<String> = faint
            .iter()
            .map(|(_, _, points)| {
                chart_path(points.iter().map(|(t, c, _, _)| c.map(|c| (x(*t), y(c)))))
            })
            .collect();
        let unit = if filter.time_relative { " %" } else { "" };
        let (x_min, x_max) = match result.frame_interval_s {
            Some(interval) => {
                let (name, per) = match f64::from(t_max) * interval {
                    span if span >= 7200.0 => ("h", 3600.0),
                    span if span >= 120.0 => ("min", 60.0),
                    _ => ("s", 1.0),
                };
                (
                    format!(
                        "{} {name}",
                        format_chart_value(f64::from(t_min) * interval / per)
                    ),
                    format!(
                        "{} {name}",
                        format_chart_value(f64::from(t_max) * interval / per)
                    ),
                )
            }
            None => (format!("T {t_min}"), format!("T {t_max}")),
        };
        let column = filter.column.display_label(&classes);
        let title = format!(
            "{column} · {}{}",
            if filter.time_median {
                "median + IQR"
            } else {
                "mean ± SD"
            },
            if filter.time_relative {
                " · % of the first time point"
            } else {
                ""
            }
        );
        let note = match series.len() > shown.len() {
            true => format!("{} of {} lines shown", shown.len(), series.len()),
            false => String::new(),
        };
        let (y_min, y_max) = (
            format!("{}{unit}", format_chart_value(low)),
            format!("{}{unit}", format_chart_value(high)),
        );

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let data: Vec<ChartTimeLine> = lines
                    .into_iter()
                    .map(|(label, color, line, band)| ChartTimeLine {
                        label: label.into(),
                        color: bg_color_to_slint(color),
                        line: line.into(),
                        band: band.into(),
                    })
                    .collect();
                state.set_chart_error("".into());
                state.set_chart_time_data(ModelRc::from(Rc::new(VecModel::from(data))));
                state.set_chart_time_faint(ModelRc::from(Rc::new(VecModel::from(
                    faint_paths
                        .into_iter()
                        .map(slint::SharedString::from)
                        .collect::<Vec<_>>(),
                ))));
                state.set_chart_time_y_min(y_min.into());
                state.set_chart_time_y_max(y_max.into());
                state.set_chart_time_x_min(x_min.into());
                state.set_chart_time_x_max(x_max.into());
                state.set_chart_time_y_title(title.into());
                state.set_chart_time_note(note.into());
            }
        })
        .ok();
    }

    /// "Export data…" of the "Over time" chart: the export dialog with only
    /// the time course, as the chart has it - its value, lines, summary.
    fn export_chart_time_course(&self) {
        let Some(ui_ready) = self.ui.upgrade() else {
            return;
        };
        let populated =
            std::mem::replace(&mut *self.export_populated.lock().expect("Poisened"), true);
        if !populated {
            self.populate_export_defaults(&ui_ready);
        }
        let filter = self.chart_filter.lock().expect("Poisened").clone();
        let classes = self.classes.lock().expect("Poisened").clone();
        let state = ui_ready.global::<ExportDialogState>();
        state.set_with_list_view(false);
        state.set_with_grouped_by_image_list(false);
        state.set_with_plate_heatmap_view(false);
        state.set_with_well_heatmap_view(false);
        state.set_with_plate_list(false);
        state.set_with_well_list(false);
        state.set_with_heatmap(false);
        state.set_with_drawn_images(false);
        state.set_with_time_course(true);
        state.set_time_lines(time_lines_index(filter.time_lines));
        let statistics = match filter.time_median {
            true => vec![
                TimeCourseStat::Median,
                TimeCourseStat::Q1,
                TimeCourseStat::Q3,
            ],
            false => TimeCourseStat::DEFAULT.to_vec(),
        };
        state.set_time_stat_items(ModelRc::from(Rc::new(VecModel::from(time_stat_items(
            &statistics,
        )))));
        state.set_time_stat_summary(list_summary(
            statistics.len(),
            TIME_STATS.len(),
            "Statistics",
        ));
        state.set_time_relative(filter.time_relative);
        // The chart's value only.
        let key = filter.column.as_key(&classes);
        let items: Vec<MultiSelectItem> = state
            .get_column_items()
            .iter()
            .map(|item| MultiSelectItem {
                selected: item.key == key.as_str(),
                ..item
            })
            .collect();
        let total = items.len();
        state.set_column_items(ModelRc::from(Rc::new(VecModel::from(items))));
        state.set_column_summary(list_summary(1, total, "Columns"));
        state.set_active(true);
    }

    /// How images are put into wells now (the project's plate settings, as
    /// in the plate view).
    fn current_grouping(&self) -> evanalyzer_app::results::Grouping {
        self.matrix_filter
            .lock()
            .expect("Poisned")
            .as_ref()
            .map(|filter| grouping_of(&filter.plate))
            .unwrap_or_default()
    }

    /// The List view, filtered to the objects of the boxplot's box `label`
    /// (a class, an image, a well, a plate row or column).
    pub fn show_box_in_list(&self, label: &str) {
        let group = self.chart_filter.lock().expect("Poisened").group_by;
        let images = self.images.lock().expect("Poisened").clone();
        let total_images = images.len();
        let mut picked_images: Vec<PathBuf> = Vec::new();
        let mut picked_class: Option<ObjectClass> = None;
        match group {
            BoxplotGroup::Class => {
                picked_class = self
                    .classes
                    .lock()
                    .expect("Poisened")
                    .iter()
                    .find(|class| class.name == label)
                    .map(|class| class.global());
                if picked_class.is_none() {
                    return;
                }
            }
            BoxplotGroup::Image => {
                picked_images = images
                    .iter()
                    .filter(|image| image.name == label)
                    .map(|image| image.rel_path.clone())
                    .collect();
            }
            group => {
                let grouping = self.current_grouping();
                let regex = {
                    let db = self.result_generator.lock().expect("Poisened");
                    let Some(db) = db.as_ref() else {
                        return;
                    };
                    match db.grouping_regex(&grouping) {
                        Ok(regex) => regex,
                        Err(err) => {
                            warn!("No grouping for the box: {err}");
                            return;
                        }
                    }
                };
                let folder = matches!(grouping, evanalyzer_app::results::Grouping::Folder);
                let texts: Vec<String> = images
                    .iter()
                    .map(|image| {
                        if folder {
                            image.rel_path.to_string_lossy().into_owned()
                        } else {
                            image.name.clone()
                        }
                    })
                    .collect();
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let part = match group {
                    BoxplotGroup::Well => 1,
                    BoxplotGroup::WellRow => 2,
                    _ => 3,
                };
                picked_images = evanalyzer_app::images::texts_in_group(&refs, &regex, part, label)
                    .into_iter()
                    .map(|i| images[i].rel_path.clone())
                    .collect();
            }
        }

        // The List's filter and its dropdowns.
        {
            let mut list_filter = self.list_filter.lock().expect("Poisened");
            match picked_class {
                Some(class) => list_filter.object_classes = vec![class],
                None => list_filter.image_rel_path = picked_images.clone(),
            }
        }
        if let Some(class) = picked_class {
            let classes = self.classes.lock().expect("Poisened").clone();
            let mut items = class_filter_items(&classes, false);
            for item in &mut items {
                item.selected = item.key.as_str() == label;
            }
            let _ = class;
            self.push_class_items(items, list_summary(1, classes.len(), "Classes"));
        } else {
            let items =
                filtered_image_items(&images, |path| picked_images.iter().any(|p| p == path), "");
            self.push_image_items(items, image_summary_text(picked_images.len(), total_images));
        }
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>()
                    .set_rail_mode(ResultsRailMode::List);
            }
        })
        .ok();
        self.refresh_list();
    }

    fn push_chart_error(&self, message: String) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_chart_error(message.into());
                state.set_chart_histogram_bins(ModelRc::from(Rc::new(VecModel::from(
                    Vec::<f32>::new(),
                ))));
                state.set_chart_scatter_points(ModelRc::from(Rc::new(VecModel::from(Vec::<
                    ChartScatterPoint,
                >::new(
                )))));
                state.set_chart_boxplot_boxes(ModelRc::from(Rc::new(VecModel::from(Vec::<
                    ChartBoxplotBox,
                >::new(
                )))));
                state.set_chart_time_data(ModelRc::from(Rc::new(VecModel::from(Vec::<
                    ChartTimeLine,
                >::new(
                )))));
            } else {
                warn!("Failed to upgrade UI handle, cannot show chart error!");
            }
        })
        .ok();
    }

    fn push_histogram_in_slint(&self, result: &HistogramResult, log_counts: bool) {
        let max_count = result.counts.iter().copied().max().unwrap_or(0).max(1);
        let bins: Vec<f32> = result
            .counts
            .iter()
            .map(|count| histogram_bar(*count, max_count, log_counts))
            .collect();
        let object_count = result.counts.iter().sum::<u64>() as i32;
        let min_label = format_chart_value(result.min);
        let max_label = format_chart_value(result.max);

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    fn push_scatter_in_slint(&self, result: &ScatterResult) {
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
        *self.scatter_points.lock().expect("Poisned") = points
            .iter()
            .zip(&result.points)
            .map(|(drawn, point)| {
                (
                    drawn.x,
                    drawn.y,
                    result
                        .images
                        .get(point.image as usize)
                        .cloned()
                        .unwrap_or_default(),
                    point.bbox_px,
                )
            })
            .collect();
        let shown_count = points.len() as i32;
        let total_count = result.total_object_count as i32;
        let x_min_label = format_chart_value(result.x_min);
        let x_max_label = format_chart_value(result.x_max);
        let y_min_label = format_chart_value(result.y_min);
        let y_max_label = format_chart_value(result.y_max);

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    fn push_boxplot_in_slint(&self, boxes: &[BoxplotBox]) {
        // Every box normalized against the combined min/max across *all*
        // boxes, so they stay comparable to each other on one shared axis
        // rather than each box silently rescaling to its own range.
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for b in boxes {
            min = min
                .min(b.min)
                .min(b.outliers.iter().copied().fold(b.min, f64::min));
            max = max
                .max(b.max)
                .max(b.outliers.iter().copied().fold(b.max, f64::max));
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let boxes: Vec<ChartBoxplotBox> = rows
                    .into_iter()
                    .map(
                        |(label, color, min, q1, median, q3, max, outliers, object_count)| {
                            ChartBoxplotBox {
                                label: label.into(),
                                // A box that isn't a class (an image, a well, ...) has
                                // no color of its own.
                                color: bg_color_to_slint(if color == 0 { 0x5B5BD6 } else { color }),
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
        if !self.is_view_shown(ResultsRailMode::List, &self.list_stale) {
            return;
        }
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            warn!("No database opened!");
            return;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = evanalyzer_app::results::PlaneFilter {
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
        let transpond_table = list_filter.transpond;
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
            ListGroupBy::Objects => db.get_object_list(&evanalyzer_app::results::ListFilter {
                plane,
                images,
                object_classes,
                columns,
                with_coloc_details,
                page: evanalyzer_app::results::Pagination {
                    limit: LIST_PAGE_SIZE,
                    after: cursor,
                },
                transpond_table,
            }),
            // Non-aggregable columns (Object ID/Image/Class) don't mean
            // anything once rows are grouped by image — silently dropped
            // here, same as the export side's own grid views (see
            // `Column::is_aggregable`). `aggregations` comes
            // from the AGGREGATE dropdown (results_list.slint), one output
            // column per (selected column x selected aggregation) combo.
            ListGroupBy::Images => {
                let columns = columns.into_iter().filter(Column::is_aggregable).collect();
                db.get_grouped_by_image(&GroupedByImageFilter {
                    plane,
                    images,
                    object_classes,
                    columns,
                    aggregation: aggregations,
                    page: evanalyzer_app::results::Pagination {
                        limit: LIST_PAGE_SIZE,
                        after: cursor,
                    },
                    transpond_table,
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
        // Every `Cell` in a row carries the same `alternating_color`,
        // `disabled` and `failed` flag (see `build_coloc_detail_rows`/
        // `cell_for_column`/`get_grouped_by_image`), so the first cell's
        // flags speak for the whole row; an empty row (no columns selected)
        // just isn't alternated/disabled/failed.
        let row_cells: Vec<(Vec<slint::SharedString>, bool, bool, bool, i32, u32)> = result
            .rows
            .iter()
            .map(|row| {
                let alternating = row.first().is_some_and(|cell| cell.alternating_color);
                let disabled = row.first().is_some_and(|cell| cell.disabled);
                let failed = row.first().is_some_and(|cell| cell.failed);
                // The class column: a dot in its color before the name.
                let (class_column, class_color) = row
                    .iter()
                    .enumerate()
                    .find_map(|(i, cell)| match &cell.value {
                        CellValue::Class((_, color)) => Some((i as i32, *color)),
                        _ => None,
                    })
                    .unwrap_or((-1, 0));
                (
                    row.iter().map(cell_to_string).collect(),
                    alternating,
                    disabled,
                    failed,
                    class_column,
                    class_color,
                )
            })
            .collect();
        let mut column_widths = list_column_widths(
            &headers,
            row_cells.iter().map(|(cells, ..)| cells.as_slice()),
        );
        // Room for the class column's color dot.
        if let Some(column) = row_cells.iter().map(|row| row.4).find(|c| *c >= 0)
            && let Some(width) = column_widths.get_mut(column as usize)
        {
            *width += 14.0;
        }
        *self.list_row_locations.lock().expect("Poisned") = result.row_locations.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                let rows: Vec<ResultRow> = row_cells
                    .into_iter()
                    .map(|(cells, alternating, disabled, failed, class_column, class_color)| {
                        ResultRow {
                            cells: ModelRc::new(VecModel::from(cells)),
                            alternating,
                            disabled,
                            failed,
                            class_column,
                            class_color: bg_color_to_slint(class_color),
                        }
                    })
                    .collect();
                state.set_list_column_headers(ModelRc::from(Rc::new(VecModel::from(headers))));
                state.set_list_column_widths(ModelRc::from(Rc::new(VecModel::from(column_widths))));
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
        // Hidden: refreshed anyway whenever the Matrix tab is shown.
        if *self.rail_mode.lock().expect("Poisoned") != ResultsRailMode::Matrix {
            return;
        }
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            warn!("No database opened!");
            return;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = evanalyzer_app::results::PlaneFilter {
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

        let group_filter = evanalyzer_app::results::PlateFilter {
            plane,
            grouping: grouping_of(&matrix_filter.plate),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // `None` (the "Auto" dropdown option) has `get_group_by_plate`
            // auto-select the smallest standard dimensions that fit the
            // data; otherwise the user's explicit PLATE SIZE choice.
            plate_size: matrix_filter.plate.plate_size,
        };
        drop(matrix_filter_guard);

        let result =
            match db.get_group_by_plate(&group_filter, &evanalyzer_app::results::View::Heatmap) {
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
        self.refresh_matrix_view_at(level);
    }

    /// `refresh_active_matrix_view` for a known `level` - callable off the
    /// UI thread.
    fn refresh_matrix_view_at(&self, level: MatrixLevel) {
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
        if let Some((result, value_caption, is_manual_scale)) = self.query_well(well_id) {
            self.set_well_in_slint(&result, value_caption, is_manual_scale);
        }
    }

    /// The fields of well `well_id` with the Matrix settings: the grid, the
    /// value caption, whether the color scale is manual. `None` without a
    /// database or filter, or when the query fails (logged).
    fn query_well(&self, well_id: &str) -> Option<(DatabaseResult, String, bool)> {
        let Some(db) = &*self.result_generator.lock().expect("Poisened") else {
            warn!("No database opened!");
            return None;
        };

        let plane_filter = self.plane_filter.lock().expect("Poisned");
        let plane = evanalyzer_app::results::PlaneFilter {
            z_stack: plane_filter.selected_z_stack,
            t_stack: plane_filter.selected_t_stack,
        };
        drop(plane_filter);

        let matrix_filter_guard = self.matrix_filter.lock().expect("Poisned");
        let Some(matrix_filter) = matrix_filter_guard.as_ref() else {
            warn!("No matrix filter set, cannot open well {well_id}");
            return None;
        };
        let value_caption = self.value_caption_for(matrix_filter);
        let is_manual_scale = matches!(matrix_filter.color_scale, ColorScale::Manual(..));

        let well_filter = WellFilter {
            plane,
            group_name: well_id.to_string(),
            grouping: grouping_of(&matrix_filter.plate),
            aggregation: matrix_filter.aggregation.clone(),
            object_class: matrix_filter.object_classe,
            column: matrix_filter.column.clone(),
            color_schema: matrix_filter.color_schema.clone(),
            color_scale: matrix_filter.color_scale.clone(),
            // `None` has `get_group_by_well` assume the common 4x4 field
            // grid; otherwise the user's explicit ROWS/COLS choice.
            well_size: well_size_of(&matrix_filter.plate),
            well_order: well_order_of(&matrix_filter.plate),
        };
        drop(matrix_filter_guard);

        match db.get_group_by_well(&well_filter, &evanalyzer_app::results::View::Heatmap) {
            Ok(result) => Some((result, value_caption, is_manual_scale)),
            Err(err) => {
                error!("Could not load well results for {well_id}: {err}");
                None
            }
        }
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

        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected well (if any) belonged to the old
                // data — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_disabled(false);
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

        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected field (if any) belonged to the
                // old well — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_disabled(false);
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
        let plane = evanalyzer_app::results::PlaneFilter {
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

        let result =
            match db.get_image_heatmap(&image_filter, &evanalyzer_app::results::View::Heatmap) {
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

        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                // The previously selected tile (if any) belonged to the old
                // image — hide its card until the user clicks a new one.
                state.set_active_well("".into());
                state.set_active_well_has_value(false);
                state.set_active_well_value("".into());
                state.set_active_well_disabled(false);
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

    /// The incomplete-results banner; empty hides it.
    fn set_run_warning_in_slint(&self, warning: String) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_run_warning(warning.into());
            }
        })
        .ok();
    }

    pub fn set_max_z_and_t_stack_in_slint(&self, t_stack_max: u32, z_stack_max: u32) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        let square_items = square_size_items(DEFAULT_SQUARE_SIZE);
        let square_summary = square_items
            .iter()
            .find(|item| item.selected)
            .map(|item| item.value.clone())
            .unwrap_or_default();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_matrix_square_size_items(ModelRc::from(Rc::new(VecModel::from(
                    square_items,
                ))));
                state.set_matrix_square_size_summary(square_summary);
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
        let analysis = analysis_summary(images);
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ResultsState>();
                state.set_list_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
                state.set_list_image_summary(summary);
                state.set_analysis_summary(analysis.into());
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        // No image selected means no image filter (all images), see
        // `image_summary_text`.
        self.list_filter.lock().expect("Poisened").image_rel_path = Vec::new();
        self.push_image_items(items, image_summary_text(0, total));
        self.refresh_list();
    }

    fn push_image_items(&self, items: Vec<MultiSelectItem>, summary: slint::SharedString) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    // Narrows the IMAGES dropdown to images whose name contains `query`
    // (case-insensitive substring match), leaving the actual filter/
    // selection state (`list_filter.image_rel_path`) and the pill's own
    // summary text untouched - only which rows the open popup shows
    // changes. `InkMultiSelect`'s manual virtualization already keeps
    // scrolling the full list smooth (see its own file), so this exists to
    // let a user with thousands of images jump to one by typing instead of
    // scrolling to find it.
    fn filter_images(&self, query: &str) {
        let images = self.images.lock().expect("Poisened");
        let selected_paths = self
            .list_filter
            .lock()
            .expect("Poisened")
            .image_rel_path
            .clone();
        let items = filtered_image_items(
            &images,
            |path| selected_paths.iter().any(|p| p == path),
            query,
        );
        drop(images);

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ResultsState>()
                    .set_list_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
            } else {
                warn!("Failed to upgrade UI handle, cannot update the filtered images list!");
            }
        })
        .ok();
    }

    fn select_all_classes(&self) {
        let classes = self.classes.lock().expect("Poisened");
        let items = class_filter_items(&classes, true);
        let total = classes.len();
        let ids = classes.iter().map(|class| class.global()).collect();
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
    fn populate_export_defaults(&self, ui_ready: &AppWindow) {
        let rail_mode = ui_ready.global::<ResultsState>().get_rail_mode();
        let z_stack_max = ui_ready.global::<ResultsState>().get_z_stack_max().max(0) as u32;
        let t_stack_max = ui_ready.global::<ResultsState>().get_t_stack_max().max(0) as u32;

        let list_filter = self.list_filter.lock().expect("Poisened");
        let matrix_filter_guard = self.matrix_filter.lock().expect("Poisened");
        let classes = self.classes.lock().expect("Poisened");
        let images = self.images.lock().expect("Poisened");
        let available_columns = self.available_columns.lock().expect("Poisened");

        // `select_none_classes` stores a sentinel (a single entry that
        // can't match any real class) rather than a genuinely empty list,
        // since empty means "no filter" (all) to `update_list_view` —
        // exporting is naturally "all" whenever nothing meaningful was
        // excluded, so the sentinel maps back to an empty `ResultExport`
        // list here rather than trying to preserve a "selected literally
        // nothing" state an export has no use for.
        let image_rel_paths: Vec<String> = list_filter
            .image_rel_path
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
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
        let (aggregations, color_schema, square_size) = match matrix_filter_guard.as_ref() {
            Some(filter) => (
                vec![filter.aggregation.clone()],
                filter.color_schema.clone(),
                filter.square_size,
            ),
            None => (vec![Aggregation::default()], ColorSchema::default(), None),
        };
        let square_size = square_size.unwrap_or(DEFAULT_SQUARE_SIZE);

        // Seeds the export dialog's own image selection from whatever the
        // main List view's image filter currently has selected - except
        // when that's "all" (`image_rel_paths` empty, which is also what
        // "none selected" means), or when it's such a
        // large individual selection (> 50) that pre-checking every one of
        // them would make the dialog's own picker unwieldy to review/
        // change - in either case, default to just the first image rather
        // than a wall of checkmarks (or none at all) the user almost
        // certainly still needs to narrow down anyway.
        let select_individually = !image_rel_paths.is_empty()
            && image_rel_paths.len() <= 50
            && image_rel_paths.len() < images.len();

        let image_selection: HashSet<PathBuf> = if select_individually {
            image_rel_paths.iter().map(PathBuf::from).collect()
        } else {
            images
                .first()
                .map(|image| image.rel_path.clone())
                .into_iter()
                .collect()
        };

        let class_items_vec: Vec<MultiSelectItem> = classes
            .iter()
            .map(|class| MultiSelectItem {
                key: class.name.as_str().into(),
                value: class.name.as_str().into(),
                color: bg_color_to_slint(class.color),
                group: "".into(),
                selected: object_classes.is_empty() || object_classes.contains(&class.global()),
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
        state.set_format_items(ModelRc::from(Rc::new(VecModel::from(export_format_items(
            ExportFormat::XLSX,
        )))));
        state.set_format_summary("XLSX".into());
        state.set_z_all(true);
        state.set_t_all(true);
        state.set_z_start("0".into());
        state.set_z_end((z_stack_max + 1).to_string().into());
        state.set_t_start("0".into());
        state.set_t_end((t_stack_max + 1).to_string().into());

        *self.export_image_selection.lock().expect("Poisened") = image_selection;
        state.set_image_filter_text("".into());
        self.push_export_image_items(ui_ready);
        state.set_class_items(ModelRc::from(Rc::new(VecModel::from(class_items_vec))));
        state.set_class_summary(class_summary);
        state.set_column_items(ModelRc::from(Rc::new(VecModel::from(column_items_vec))));
        state.set_column_groups(ModelRc::from(Rc::new(VecModel::from(column_groups_vec))));
        state.set_column_summary(column_summary);

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
        state.set_square_size_items(ModelRc::from(Rc::new(VecModel::from(square_size_items(
            square_size,
        )))));
        state.set_square_size_summary(format!("{square_size} px").into());

        state.set_with_list_view(false);
        state.set_with_list_coloc_details(false);
        state.set_with_list_one_file_per_image(true);
        state.set_with_grouped_by_image_list(true);
        state.set_with_plate_heatmap_view(rail_mode == ResultsRailMode::Matrix);
        state.set_with_well_heatmap_view(rail_mode == ResultsRailMode::Matrix);
        state.set_with_plate_list(false);
        state.set_with_well_list(false);
        state.set_with_heatmap(false);
        state.set_with_time_course(false);
        state.set_time_lines(0);
        state.set_time_stat_items(ModelRc::from(Rc::new(VecModel::from(time_stat_items(
            &TimeCourseStat::DEFAULT,
        )))));
        state.set_time_stat_summary(list_summary(
            TimeCourseStat::DEFAULT.len(),
            TIME_STATS.len(),
            "Statistics",
        ));
        state.set_time_relative(false);
        state.set_time_layout(2);

        state.set_is_exporting(false);
        state.set_done(false);
        state.set_has_error(false);
        state.set_error_message("".into());
        state.set_progress_message("".into());
        state.set_progress_current(0);
        state.set_progress_total(0);
    }

    // Rebuilds the export dialog's IMAGES rows from `export_image_selection`,
    // showing only the images matching the picker's search text, and
    // refreshes its summary. Runs on the UI thread (called from Slint
    // callbacks), so it pushes into `ExportDialogState` directly.
    fn push_export_image_items(&self, ui_ready: &AppWindow) {
        let state = ui_ready.global::<ExportDialogState>();
        let query = state.get_image_filter_text();
        let images = self.images.lock().expect("Poisened");
        let selection = self.export_image_selection.lock().expect("Poisened");
        let items = filtered_image_items(&images, |path| selection.contains(path), &query);
        drop(selection);
        drop(images);
        state.set_image_items(ModelRc::from(Rc::new(VecModel::from(items))));
        self.push_export_image_summary(ui_ready);
    }

    fn push_export_image_summary(&self, ui_ready: &AppWindow) {
        let total = self.images.lock().expect("Poisened").len();
        let selected = self.export_image_selection.lock().expect("Poisened").len();
        ui_ready
            .global::<ExportDialogState>()
            .set_image_summary(image_summary_text(selected, total));
    }

    // Reads `ExportDialogState`'s current values back into a `ResultExport`
    // at "Start Export" time — the single point where the dialog's Slint
    // state (the source of truth for everything a control lets the user
    // edit directly) gets turned into the typed request `start_export`
    // wants. `Err` for a setting that can't be parsed/resolved at all
    // (blank output folder, unrecognized dropdown key, non-numeric range
    // field) — shown in the dialog's error area rather than silently
    // falling back to a default the user didn't ask for.
    /// "Images with objects drawn": where the project's images are, the
    /// series and the channels' colors (the database knows only the
    /// objects).
    fn drawn_images(&self) -> evanalyzer_app::results::DrawnImages {
        use evanalyzer_app::project::ProjectExt;
        let project = self._app_state.get_project();
        evanalyzer_app::results::DrawnImages {
            image_root: project.images.root.clone().unwrap_or_default(),
            series: project.images.settings.selected_series.unwrap_or(0),
            max_side: 4096,
            colors: project.get_channel_display_colors().into_iter().collect(),
        }
    }

    fn read_export_settings(&self, ui_ready: &AppWindow) -> Result<ResultExport, String> {
        let state = ui_ready.global::<ExportDialogState>();
        let classes = self.classes.lock().expect("Poisened");

        let output_dir = state.get_output_dir().to_string();
        if output_dir.is_empty() {
            return Err("Choose an output folder first.".to_string());
        }

        let file_prefix_tmp = state.get_output_file_prefix().to_string();
        let file_prefix = if file_prefix_tmp.is_empty() {
            None
        } else {
            Some(file_prefix_tmp)
        };

        let transpond_table = state.get_transpond_output_table();

        let parse_u32 = |label: &str, text: slint::SharedString| -> Result<u32, String> {
            text.trim()
                .parse::<u32>()
                .map_err(|_| format!("\"{text}\" isn't a valid {label}."))
        };
        // A range only when chosen - "All" leaves its (hidden) fields alone.
        let range =
            |all: bool, axis: &str, start: slint::SharedString, end: slint::SharedString| {
                if all {
                    return Ok(None);
                }
                let start = parse_u32(&format!("{axis} start"), start)?;
                let end = parse_u32(&format!("{axis} end"), end)?;
                Ok::<_, String>(Some(std::range::Range {
                    start,
                    end: end.max(start + 1),
                }))
            };
        let z_stacks = range(
            state.get_z_all(),
            "Z",
            state.get_z_start(),
            state.get_z_end(),
        )?;
        let t_stacks = range(
            state.get_t_all(),
            "T",
            state.get_t_start(),
            state.get_t_end(),
        )?;

        let selected_keys = |items: ModelRc<MultiSelectItem>| -> Vec<slint::SharedString> {
            items
                .iter()
                .filter(|item| item.selected)
                .map(|item| item.key)
                .collect()
        };

        // From `export_image_selection`, not the picker's `image-items`,
        // which only holds the rows the search box currently shows.
        let image_rel_paths: Vec<String> = self
            .export_image_selection
            .lock()
            .expect("Poisened")
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        // Empty means every image, as the picker's "All Images" says.

        let mut object_classes = Vec::new();
        for key in selected_keys(state.get_class_items()) {
            let Some(class) = classes.iter().find(|class| class.name == key.as_str()) else {
                return Err(format!("Unknown class \"{key}\" selected for export."));
            };
            object_classes.push(class.global());
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

        // Grouping, plate size and well layout: the project's plate
        // settings, the same the Matrix view shows.
        let plate = self.current_plate_settings();

        let square_size_key = state
            .get_square_size_items()
            .iter()
            .find(|item| item.selected)
            .map(|item| item.key)
            .unwrap_or_else(|| DEFAULT_SQUARE_SIZE.to_string().into());
        let square_size = square_size_key
            .parse::<usize>()
            .map_err(|_| format!("Unknown square size \"{square_size_key}\"."))?;

        let format_key = state
            .get_format_items()
            .iter()
            .find(|item| item.selected)
            .map(|item| item.key)
            .unwrap_or_else(|| "XLSX".into());
        let Some(format) = export_format_from_key(format_key.as_str()) else {
            return Err(format!("Unknown export format \"{format_key}\"."));
        };

        Ok(ResultExport {
            output_dir: PathBuf::from(output_dir),
            outputfile_prefix: file_prefix,
            format,
            z_stacks,
            t_stacks,
            image_rel_paths,
            columns,
            color_schema,
            color_scale: ColorScale::Auto,
            grouping: grouping_of(&plate),
            object_classes,
            aggregations,
            plate_size: plate.plate_size,
            well_size: well_size_of(&plate),
            well_order: well_order_of(&plate),
            square_size: Some(square_size),
            with_list_view: state.get_with_list_view(),
            with_list_coloc_details: state.get_with_list_coloc_details(),
            with_list_one_file_per_image: state.get_with_list_one_file_per_image(),
            with_grouped_by_image_list: state.get_with_grouped_by_image_list(),
            with_time_course: state.get_with_time_course(),
            time_course: TimeCourseExport {
                lines: time_lines_of(state.get_time_lines()),
                statistics: TIME_STATS
                    .iter()
                    .copied()
                    .filter(|stat| {
                        state
                            .get_time_stat_items()
                            .iter()
                            .any(|item| item.selected && item.key == stat.label())
                    })
                    .collect(),
                normalize: match state.get_time_relative() {
                    true => TimeCourseNormalize::RelativeToStart,
                    false => TimeCourseNormalize::None,
                },
                layout: match state.get_time_layout() {
                    0 => TimeCourseLayout::Long,
                    1 => TimeCourseLayout::Wide,
                    _ => TimeCourseLayout::Both,
                },
            },
            with_plate_view_heatmap: state.get_with_plate_heatmap_view(),
            with_well_view_heatmap: state.get_with_well_heatmap_view(),
            with_plate_view_list: state.get_with_plate_list(),
            with_well_view_list: state.get_with_well_list(),
            with_heatmap: state.get_with_heatmap(),
            transpond_table,
            drawn_images: state.get_with_drawn_images().then(|| self.drawn_images()),
        })
    }

    // Runs `export` on a background thread. The results source gives it its
    // own database connection (see `LocalResults::export`), so the
    // (potentially long-running) export doesn't block every other
    // List/Matrix query meanwhile.
    fn run_export(self: &Arc<Self>, export: ResultExport) {
        let Some(database) = self.result_generator.lock().expect("Poisened").clone() else {
            self.push_export_error("No database is open.".to_string());
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        *self.export_cancel_flag.lock().expect("Poisened") = Some(cancel.clone());
        let manager = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let manager_for_progress = manager.clone();
            let result = database.export(&export, &cancel, &mut |message, current, total| {
                manager_for_progress.push_export_progress(message.to_string(), current, total);
            });
            match result {
                Ok(()) => manager.push_export_done(),
                Err(InternalErrors::Cancelled) => manager.push_export_cancelled(),
                Err(err) => manager.push_export_error(err.to_string()),
            }
            *manager.export_cancel_flag.lock().expect("Poisened") = None;
        });
    }

    fn push_export_progress(&self, message: String, current: usize, total: usize) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    // A user-cancelled export is not an error - it's reported through
    // `progress_message` rather than `has_error`/`error_message`, so the
    // dialog doesn't flash red for something the user explicitly asked for.
    fn push_export_cancelled(&self) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                let state = ui_ready.global::<ExportDialogState>();
                state.set_is_exporting(false);
                state.set_done(false);
                state.set_has_error(false);
                state.set_progress_message("Export cancelled.".into());
            } else {
                warn!("Failed to upgrade UI handle, cannot report export cancellation!");
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
// `update_list_view`'s `image_rel_path.is_empty()` check), not "match
// nothing" - for the export dialog's picker too. "0 of N
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

/// Whether `column` is something `chart_value_expr`/`column_aggregate_expr`
/// (results_generator.rs) can resolve to a single per-object SQL
/// expression — the exact same set that function accepts, kept in sync by
/// inspecting the same variants. Unlike `Column::is_aggregable`, `Count` is
/// also excluded here: it's a row tally (1 per object), not a per-object
/// measurement, so charting it doesn't mean anything the way it does for
/// Images-mode's own per-image counts.
fn is_chartable_column(column: &Column) -> bool {
    !matches!(
        column,
        Column::ObjectId
            | Column::TrackId
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
        Column::ObjectId | Column::TrackId | Column::ImageName | Column::ObjectClass
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

// FORMAT dropdown vocabulary for the export dialog — mirrors
// `ExportFormat`'s three variants (see results_exporter.rs); "Parquet"
// ignores every other dialog setting (column/filter/grouping/plate/well/
// heatmap), which `export_dialog.slint` calls out with its own note text
// when selected.
fn export_format_options() -> [(&'static str, ExportFormat); 3] {
    [
        ("XLSX", ExportFormat::XLSX),
        ("CSV", ExportFormat::CSV),
        ("Parquet", ExportFormat::Parquet),
    ]
}

fn export_format_items(selected: ExportFormat) -> Vec<MultiSelectItem> {
    export_format_options()
        .into_iter()
        .map(|(name, format)| MultiSelectItem {
            key: name.into(),
            value: name.into(),
            color: Color::default(),
            group: "".into(),
            selected: format == selected,
        })
        .collect()
}

fn export_format_from_key(key: &str) -> Option<ExportFormat> {
    export_format_options()
        .into_iter()
        .find(|(name, _)| *name == key)
        .map(|(_, format)| format)
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

/// How the results queries group images for `plate`.
fn grouping_of(plate: &PlateSettings) -> Grouping {
    match plate.grouping_mode {
        GroupingMode::Auto => Grouping::Auto,
        GroupingMode::Custom => Grouping::Regex(plate.grouping_regex.clone()),
        GroupingMode::Folder => Grouping::Folder,
    }
}

/// The well grid for `plate` - `None` lets the query fit it (Auto).
fn well_size_of(plate: &PlateSettings) -> Option<WellSize> {
    match plate.well_layout {
        WellLayout::Auto => None,
        WellLayout::Fixed { rows, cols } => Some(WellSize {
            rows: rows.max(1) as usize,
            cols: cols.max(1) as usize,
        }),
    }
}

/// Which image goes where in a fixed well grid - `None` (number order) for
/// Auto or without an order.
fn well_order_of(plate: &PlateSettings) -> Option<Vec<u32>> {
    match plate.well_layout {
        WellLayout::Fixed { .. } if !plate.well_image_order.is_empty() => {
            Some(plate.well_image_order.clone())
        }
        _ => None,
    }
}

// GROUP BY dropdown vocabulary.
const GROUPING_MODES: [(&str, GroupingMode); 3] = [
    ("Auto", GroupingMode::Auto),
    ("Custom regex", GroupingMode::Custom),
    ("Folder", GroupingMode::Folder),
];

fn single_select_items<T: PartialEq>(
    choices: impl IntoIterator<Item = (String, T)>,
    selected: &T,
) -> Vec<MultiSelectItem> {
    choices
        .into_iter()
        .map(|(name, value)| MultiSelectItem {
            key: name.clone().into(),
            value: name.into(),
            color: Color::default(),
            group: "".into(),
            selected: value == *selected,
        })
        .collect()
}

fn grouping_items(selected: GroupingMode) -> Vec<MultiSelectItem> {
    single_select_items(
        GROUPING_MODES.map(|(name, mode)| (name.to_string(), mode)),
        &selected,
    )
}

// PLATE SIZE dropdown vocabulary: the same list as the project settings
// (`PlateSize::ALL`, keyed by its label).
fn plate_size_items(selected: PlateSize) -> Vec<MultiSelectItem> {
    single_select_items(PlateSize::ALL.map(|size| (size.label(), size)), &selected)
}

fn plate_size_from_key(key: &str) -> Option<PlateSize> {
    PlateSize::ALL.into_iter().find(|size| size.label() == key)
}

// WELL SIZE dropdown vocabulary: fitted to the images (Auto) or the ROWS/COLS
// fields (Custom).
const WELL_LAYOUT_AUTO: &str = "Auto";
const WELL_LAYOUT_CUSTOM: &str = "Custom";

fn well_layout_items(layout: WellLayout) -> Vec<MultiSelectItem> {
    let auto = matches!(layout, WellLayout::Auto);
    single_select_items(
        [
            (WELL_LAYOUT_AUTO.to_string(), true),
            (WELL_LAYOUT_CUSTOM.to_string(), false),
        ],
        &auto,
    )
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

/// `image_items`'s search-aware sibling - `filter_images`'s pure core,
/// pulled out as a standalone function so the actual filtering logic can be
/// unit-tested directly rather than through a Slint-dispatched callback
/// (see that test module's note on why - queued `invoke_from_event_loop`
/// pushes never run under the headless test platform's `init_no_event_loop`).
/// Keeps every image whose name contains `query` (case-insensitive, empty
/// query keeps everything), each carrying its *real* current selection
/// state from `is_selected` - unlike `image_items`'s uniform bool, since
/// narrowing the dropdown's visible rows must never look like it changed
/// which images are actually selected.
fn filtered_image_items(
    images: &[ImageEntry],
    is_selected: impl Fn(&Path) -> bool,
    query: &str,
) -> Vec<MultiSelectItem> {
    let query_lower = query.to_lowercase();
    images
        .iter()
        .filter(|image| query_lower.is_empty() || image.name.to_lowercase().contains(&query_lower))
        .map(|image| MultiSelectItem {
            key: image.rel_path.to_str().unwrap_or_default().into(),
            value: image.name.as_str().into(),
            color: Color::default(),
            group: "".into(),
            selected: is_selected(&image.rel_path),
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
                    // `exists` (a real well/field sits here, so it should
                    // be clickable/openable) is broader than `has_value`
                    // (there's a real number to show/color by) - an image
                    // with zero detected objects still has a `search_key`
                    // (see `get_group_by_plate`/`get_group_by_well`'s
                    // Heatmap arms) even though its `CellValue` is `Empty`.
                    let exists = cell.search_key.is_some();
                    let (has_value, value, label) = match &cell.value {
                        CellValue::Float(v) => (true, *v, format!("{v:.2}")),
                        _ if exists => (false, 0.0, "—".to_string()),
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
                        exists,
                        label: label.into(),
                        color: bg_color_to_slint(cell.bg_color),
                        disabled: cell.disabled,
                        any_disabled: cell.any_disabled,
                        failed: cell.failed,
                        any_failed: cell.any_failed,
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
    evanalyzer_app::results::color_scale_gradient(schema)
        .into_iter()
        .map(bg_color_to_slint)
        .collect()
}

fn cell_to_string(cell: &Cell) -> slint::SharedString {
    match &cell.value {
        CellValue::Empty => "NaN".into(),
        CellValue::String(value) => short_object_id(value).into(),
        CellValue::Float(value) => format!("{value:.3}").into(),
        CellValue::Integer(value) => value.to_string().into(),
        CellValue::Class((name, _color)) => name.as_str().into(),
    }
}

/// A histogram bar's height (0..1) for `count` of the fullest `max`:
/// linear, or on a log scale (`ln(1 + n) / ln(1 + max)`).
fn histogram_bar(count: u64, max: u64, log: bool) -> f32 {
    if log {
        ((1 + count) as f32).ln() / ((1 + max.max(1)) as f32).ln()
    } else {
        count as f32 / max.max(1) as f32
    }
}

/// An object id (a UUID, `01a1497c-887f-7d95-93c3-bfc847366b32`) as its
/// last block (`…bfc847366b32`): the leading blocks are a timestamp, the
/// same for the objects of one run - only the last tells them apart. Other
/// text as it is. (Exports keep the full id.)
fn short_object_id(value: &str) -> &str {
    let b = value.as_bytes();
    let is_uuid = b.len() == 36
        && [8, 13, 18, 23].iter().all(|&i| b[i] == b'-')
        && b.iter()
            .enumerate()
            .all(|(i, c)| [8, 13, 18, 23].contains(&i) || c.is_ascii_hexdigit());
    if is_uuid { &value[24..] } else { value }
}

/// One pixel width per `results_list.slint` column, each just wide enough
/// for the longer of that column's header or widest cell *on this page*
/// (Rust never holds more than one page of rows - see `LIST_PAGE_SIZE`),
/// clamped to `[MIN_LIST_COLUMN_WIDTH_PX, MAX_LIST_COLUMN_WIDTH_PX]`.
///
/// No real font-metrics measurement needed: every header/cell `Text` in
/// that table renders in `Theme.font-mono` ("IBM Plex Mono"), and monospace
/// fonts by definition give every character (regular or the header's bold
/// weight) the same fixed advance width - IBM Plex Mono's is documented as
/// exactly 0.6em. So a column's natural width is just `char_count *
/// font_size * 0.6`, no different from measuring real glyphs for this
/// specific font. A proportional font would need the real thing.
fn list_column_widths<'a>(
    headers: &[slint::SharedString],
    rows: impl Iterator<Item = &'a [slint::SharedString]> + Clone,
) -> Vec<f32> {
    const MONO_ADVANCE_EM: f32 = 0.6;
    const HEADER_FONT_SIZE_PX: f32 = 10.5;
    const BODY_FONT_SIZE_PX: f32 = 12.0;
    // Safety margin against elide-triggering rounding (cell padding,
    // sub-pixel layout rounding) - not meant to look like extra whitespace.
    const WIDTH_PADDING_PX: f32 = 8.0;
    const MIN_LIST_COLUMN_WIDTH_PX: f32 = 60.0;
    const MAX_LIST_COLUMN_WIDTH_PX: f32 = 260.0;

    headers
        .iter()
        .enumerate()
        .map(|(col, header)| {
            let header_px = header.chars().count() as f32 * HEADER_FONT_SIZE_PX * MONO_ADVANCE_EM;
            let widest_cell_px = rows
                .clone()
                .filter_map(|row| row.get(col))
                .map(|cell| cell.chars().count() as f32 * BODY_FONT_SIZE_PX * MONO_ADVANCE_EM)
                .fold(0.0f32, f32::max);
            (header_px.max(widest_cell_px) + WIDTH_PADDING_PX)
                .clamp(MIN_LIST_COLUMN_WIDTH_PX, MAX_LIST_COLUMN_WIDTH_PX)
        })
        .collect()
}

/// "81 images · 1 disabled" (the status bar's) - without the disabled
/// part when none is.
fn analysis_summary(images: &[ImageEntry]) -> String {
    let disabled = images.iter().filter(|image| image.disabled).count();
    let n = images.len();
    let images_text = if n == 1 {
        "1 image".to_string()
    } else {
        format!("{n} images")
    };
    if disabled == 0 {
        images_text
    } else {
        format!("{images_text} · {disabled} disabled")
    }
}
/// The point of `points` (x, y in 0..1 of an area `width` x `height` px,
/// y up) nearest to (`fx`, `fy`), if within `radius` px: its object's
/// image and box.
fn nearest_point(
    points: &[(f32, f32, String, [u32; 4])],
    fx: f32,
    fy: f32,
    width: f32,
    height: f32,
    radius: f32,
) -> Option<(String, [u32; 4])> {
    points
        .iter()
        .map(|(x, y, path, bbox)| {
            let dx = (x - fx) * width;
            let dy = (y - fy) * height;
            (dx * dx + dy * dy, path, bbox)
        })
        .filter(|(d2, _, _)| *d2 <= radius * radius)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, path, bbox)| (path.clone(), *bbox))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppWindow;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::object_list_controller::ObjectListController;
    use crate::editor::test_support::{test_ui_state, test_ui_window};
    use crate::editor::viewport_controller::ViewportController;

    fn make_controller_with_ui(ui: slint::Weak<AppWindow>) -> Arc<ResultsStateController> {
        make_controller_with_state(ui, test_ui_state())
    }

    fn make_controller_with_state(
        ui: slint::Weak<AppWindow>,
        ui_state: Arc<UiState>,
    ) -> Arc<ResultsStateController> {
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
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
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
            histogram_controller,
            image_meta_controller,
            object_list_controller.clone(),
            Arc::new(
                crate::editor::classification_controller::ClassificationController::new(
                    ui.clone(),
                    ui_state.clone(),
                    object_list_controller.clone(),
                    viewport_controller.clone(),
                ),
            ),
        ));
        Arc::new(ResultsStateController::new(
            ui,
            ui_state,
            image_list_controller,
        ))
    }

    // Regression test for the export dialog's very first hurdle: for a long
    // time this failed to even *compile* (`ExportDialogState` had no public
    // Rust binding at all - it wasn't re-exported from main.slint's own
    // top-level `export { ... }` block, the same way every other dialog
    // global there is, see main.slint). Exercises the actual callback
    // wiring end to end against a real (headless) `AppWindow`, with no
    // database open - the state every dialog control's defaults must still
    // fall back to sanely (an empty images/classes list, no matrix filter
    // set yet).
    #[test]
    fn attach_callbacks_export_dialog_open_populates_defaults_and_activates() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        assert!(!ui.global::<ExportDialogState>().get_active());

        ui.global::<ResultsState>().invoke_export_dialog_open();

        let state = ui.global::<ExportDialogState>();
        assert!(state.get_active(), "dialog should activate on open");
        // See `populate_export_defaults`: the Object list starts unchecked
        // (one-file-per-image and the Image/Grouped-by-image list start
        // checked instead), and the plate/well heatmap flags follow
        // whether the rail was on Matrix - no matrix view has been
        // rendered yet (`matrix_filter` starts `None`) and the rail starts
        // on List, so both start unchecked too.
        assert!(!state.get_with_list_view());
        assert!(state.get_with_list_one_file_per_image());
        assert!(state.get_with_grouped_by_image_list());
        assert!(!state.get_with_plate_heatmap_view());
        assert!(!state.get_with_well_heatmap_view());
        assert!(!state.get_with_plate_list());
        assert!(!state.get_with_well_list());
        assert!(!state.get_with_heatmap());
        // z/t default to the database's full range, exclusive end - starting
        // at 0, and past whatever `ResultsState.z-stack-max`/`t-stack-max`
        // are currently set to (their own default, before any real database
        // has been opened, is a temporary design-preview placeholder - see
        // that property's doc comment in api/results.slint - so this only
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
        // Format defaults to XLSX with all 3 options present.
        assert_eq!(state.get_format_summary(), "XLSX");
        assert_eq!(state.get_format_items().row_count(), 3);
        assert!(
            state
                .get_format_items()
                .iter()
                .any(|item| item.key == "XLSX" && item.selected)
        );
    }

    #[test]
    fn export_format_items_marks_only_the_selected_format() {
        let items = export_format_items(ExportFormat::CSV);
        let selected: Vec<&str> = items
            .iter()
            .filter(|item| item.selected)
            .map(|item| item.key.as_str())
            .collect();
        assert_eq!(selected, vec!["CSV"]);
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn export_format_from_key_round_trips_every_option_and_rejects_unknown_keys() {
        for (name, format) in export_format_options() {
            assert_eq!(export_format_from_key(name), Some(format));
        }
        assert_eq!(export_format_from_key("bogus"), None);
    }

    // Re-opening the dialog after the user has already changed something
    // (here: unchecking "List view") must not clobber it - only the *first*
    // open after a database loads re-seeds the defaults (see
    // `export_populated`).
    #[test]
    fn attach_callbacks_export_dialog_reopen_keeps_user_changes() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<ResultsState>().invoke_export_dialog_open();
        ui.global::<ExportDialogState>().set_with_list_view(false);
        ui.global::<ExportDialogState>().set_active(false);

        ui.global::<ResultsState>().invoke_export_dialog_open();

        let state = ui.global::<ExportDialogState>();
        assert!(state.get_active());
        assert!(
            !state.get_with_list_view(),
            "re-opening must not reset a setting the user already changed"
        );
    }

    // `on_start_clicked` with no output folder chosen must surface a
    // validation error rather than silently doing nothing or panicking
    // (there's no database open in this test either, so this also exercises
    // that `read_export_settings` fails fast before `run_export` ever
    // touches `result_generator`).
    #[test]
    fn attach_callbacks_export_start_without_output_dir_reports_an_error() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        ui.global::<ResultsState>().invoke_export_dialog_open();
        ui.global::<ExportDialogState>().invoke_start_clicked();

        let state = ui.global::<ExportDialogState>();
        assert!(state.get_has_error());
        assert!(!state.get_is_exporting());
        assert!(!state.get_error_message().is_empty());
    }

    #[test]
    fn on_cancel_clicked_does_nothing_when_no_export_is_running() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        assert!(controller.export_cancel_flag.lock().unwrap().is_none());
        // Must not panic even though no export ever set the flag.
        ui.global::<ExportDialogState>().invoke_cancel_clicked();
        assert!(controller.export_cancel_flag.lock().unwrap().is_none());
    }

    #[test]
    fn on_cancel_clicked_signals_the_flag_stored_for_the_running_export() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();

        // Simulate `run_export` having just started a background export -
        // it stores a fresh flag here before spawning the export thread.
        let flag = Arc::new(AtomicBool::new(false));
        *controller.export_cancel_flag.lock().unwrap() = Some(flag.clone());

        ui.global::<ExportDialogState>().invoke_cancel_clicked();

        assert!(
            flag.load(Ordering::Relaxed),
            "cancel-clicked must signal the stored flag"
        );
    }

    // -- Real-database fixture ---------------------------------------------
    //
    // `attach_callbacks`'s ~60 callbacks are almost entirely untested above
    // this point - every real one (as opposed to `on_refresh_clicked`-style
    // stubs) needs a real, opened `ResultsGenerator` to do anything but
    // early-return on "no database". This seeds a small but genuinely
    // plate/well/heatmap-shaped `.evadb` (schema kept in sync with
    // evanalyzer_core's `duckdb.rs` `CREATE_TABLES`, same duplication
    // reasoning as `crates/app/src/results/test_support.rs`/`crates/cli/src/
    // commands/test_support.rs`) so the tests below can open it for real via
    // `ResultsStateController::open_database` and exercise the callbacks
    // that only do anything once real query results exist.

    /// Well A1 has two fields (`A1_01.tif`, `A1_02.tif`), well A2 has one
    /// (`A2_01.tif`) - matching `DEFAULT_GROUPING_REGEX` - across two
    /// classes, with one object colocalizing with another so
    /// `with_coloc_details`-adjacent code has something real to resolve.
    fn seed_test_db(path: &std::path::Path) {
        let conn = duckdb::Connection::open(path).expect("open test db");
        conn.execute_batch(
            "CREATE TABLE objects (
                image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL,
                c_stack INTEGER, z_stack INTEGER, t_stack INTEGER,
                object_id UUID NOT NULL,
                seg_class_name VARCHAR, seg_class_id INTEGER,
                object_class_name VARCHAR, object_class_id INTEGER[],
                parent_id VARCHAR, children VARCHAR, track_id UBIGINT,
                centroid_x_px DOUBLE, centroid_y_px DOUBLE, centroid_x_nm DOUBLE, centroid_y_nm DOUBLE,
                bbox_xmin_px UINTEGER, bbox_ymin_px UINTEGER, bbox_xmax_px UINTEGER, bbox_ymax_px UINTEGER,
                bbox_xmin_nm DOUBLE, bbox_ymin_nm DOUBLE, bbox_xmax_nm DOUBLE, bbox_ymax_nm DOUBLE,
                area_px UBIGINT, area_nm2 DOUBLE, perimeter_px DOUBLE, perimeter_nm DOUBLE,
                circularity DOUBLE, solidity DOUBLE, aspect_ratio DOUBLE, roundness DOUBLE, compactness DOUBLE,
                major_axis_px DOUBLE, minor_axis_px DOUBLE, major_axis_nm DOUBLE, minor_axis_nm DOUBLE,
                major_axis_angle DOUBLE, eccentricity DOUBLE,
                feret_diameter_px DOUBLE, min_feret_px DOUBLE, feret_diameter_nm DOUBLE, min_feret_nm DOUBLE,
                touches_edge BOOLEAN,
                pixel_size_x_nm DOUBLE, pixel_size_y_nm DOUBLE, pixel_size_z_nm DOUBLE,
                image_bit_depth UTINYINT,
                intensity_sum_normalized DOUBLE[], intensity_sum_gray DOUBLE[],
                intensity_mean_normalized DOUBLE[], intensity_mean_gray DOUBLE[],
                intensity_min_normalized DOUBLE[], intensity_min_gray DOUBLE[],
                intensity_max_normalized DOUBLE[], intensity_max_gray DOUBLE[],
                coloc_partner_ids MAP(INTEGER, UUID[])
            );
            CREATE TABLE images (
                image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
                successful BOOLEAN NOT NULL DEFAULT true, error_message VARCHAR,
                disabled BOOLEAN NOT NULL DEFAULT false,
                width UINTEGER NOT NULL, height UINTEGER NOT NULL,
                c_stacks UINTEGER NOT NULL, z_stacks UINTEGER NOT NULL, t_stacks UINTEGER NOT NULL
            );
            CREATE TABLE classes (
                class_id INTEGER NOT NULL PRIMARY KEY, name VARCHAR NOT NULL, color UINTEGER
            );",
        )
        .expect("create test schema");

        // Channel 0 only: sum 1.0 (255 gray values), mean 0.5 (127), min 0,
        // max 1.0 (255).
        let ch0_intensities = "[1.0], [255.0], [0.5], [127.0], [0.0], [0.0], [1.0], [255.0]";
        let insert = |idx: usize,
                      image: &str,
                      class_name: &str,
                      class_id: i32,
                      area_px: u64,
                      centroid: (f64, f64),
                      // SQL map literal, partner class -> partner ids.
                      coloc_partner_ids: &str| {
            let object_id = format!("00000000-0000-0000-0000-{idx:012}");
            conn.execute(
                &format!("INSERT INTO objects (
                    image_name, image_rel_path, t_stack, z_stack, object_id, seg_class_name, seg_class_id,
                    object_class_name, object_class_id, track_id,
                    centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                    bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                    bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                    area_px, area_nm2, perimeter_px, perimeter_nm,
                    circularity, solidity, aspect_ratio, roundness, compactness,
                    major_axis_px, minor_axis_px, eccentricity, touches_edge,
                    pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                    intensity_sum_normalized, intensity_sum_gray, intensity_mean_normalized,
                    intensity_mean_gray, intensity_min_normalized, intensity_min_gray,
                    intensity_max_normalized, intensity_max_gray, coloc_partner_ids
                ) VALUES (
                    ?, ?, 0, 0, ?, ?, ?,
                    ?, ?, 0,
                    ?, ?, 0, 0,
                    0, 0, 10, 10,
                    0, 0, 0, 0,
                    ?, ?, 40, 40,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    10, 10, 1.0, false,
                    1.0, 1.0, 1.0,
                    {ch0_intensities}, {coloc_partner_ids}
                )"),
                duckdb::params![
                    image,
                    image,
                    object_id,
                    class_name,
                    class_id,
                    format!("[\"{class_name}\"]"),
                    format!("[{class_id}]"),
                    centroid.0,
                    centroid.1,
                    area_px,
                    area_px as f64,
                ],
            )
            .unwrap_or_else(|e| panic!("insert object {idx}: {e}"));
        };

        insert(
            0,
            "A1_01.tif",
            "ClassA",
            1,
            10,
            (10.0, 10.0),
            "MAP {2: ['00000000-0000-0000-0000-000000000002']}",
        );
        insert(1, "A1_02.tif", "ClassA", 1, 20, (60.0, 10.0), "MAP {}");
        insert(2, "A2_01.tif", "ClassB", 2, 30, (10.0, 10.0), "MAP {}");

        for image in ["A1_01.tif", "A1_02.tif", "A2_01.tif"] {
            conn.execute(
                "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
                 VALUES (?, ?, 100, 100, 1, 1, 1)",
                duckdb::params![image, image],
            )
            .unwrap_or_else(|e| panic!("insert image {image}: {e}"));
        }
        for (id, name) in [(1, "ClassA"), (2, "ClassB")] {
            conn.execute(
                "INSERT INTO classes (class_id, name, color) VALUES (?, ?, 0)",
                duckdb::params![id, name],
            )
            .unwrap_or_else(|e| panic!("insert class {name}: {e}"));
        }
    }

    /// Builds a real, headless `AppWindow` with
    /// callbacks attached and a real seeded `.evadb` already opened via
    /// `open_database` - the common setup every test below this point
    /// shares. Leaks the backing `TempDir` (only the file path is needed
    /// afterwards) - acceptable for a short-lived test process.
    fn controller_with_open_database() -> (AppWindow, Arc<ResultsStateController>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        std::mem::forget(dir);

        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.attach_callbacks();
        controller.open_database(path);
        (ui, controller)
    }

    /// "Open results" on a finished analysis' note opens that analysis'
    /// database - not the one the Results view had open before.
    #[test]
    fn open_results_opens_the_finished_analysis_not_the_one_open_before() {
        use crate::PipelineRunningState;
        use crate::editor::results_list_controller::ResultsListController;
        let dir = tempfile::tempdir().expect("tempdir");
        let earlier = dir.path().join("20261001T000000Z__earlier");
        let finished = dir.path().join("20261021T000000Z__finished");
        for (folder, name) in [(&earlier, "earlier"), (&finished, "finished")] {
            std::fs::create_dir_all(folder).unwrap();
            seed_test_db(&folder.join(format!("{name}.evadb")));
            // The run's settings beside it - not a database.
            std::fs::write(folder.join(format!("{name}.evaproj")), "{}").unwrap();
        }

        let ui = test_ui_window();
        let ui_state = test_ui_state();
        let results = make_controller_with_state(ui.as_weak(), ui_state.clone());
        results.attach_callbacks();
        let list = Arc::new(ResultsListController::new(
            ui.as_weak(),
            ui_state,
            results.clone(),
        ));
        list.attach_callbacks();

        results.open_database(earlier.join("earlier.evadb"));
        drain_ui_queue();
        assert_eq!(
            ui.global::<ResultsState>().get_database_name(),
            "earlier.evadb"
        );

        list.analysis_finished(&finished);
        ui.global::<PipelineRunningState>().invoke_open_results();
        drain_ui_queue();
        assert_eq!(
            ui.global::<ResultsState>().get_database_name(),
            "finished.evadb"
        );
    }

    #[test]
    fn a_clicked_box_shows_its_class_or_image_in_the_list() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();
        state.set_rail_mode(ResultsRailMode::Charts);

        // A class's box: the List shows that class.
        let class = controller.classes.lock().unwrap()[0].clone();
        controller.show_box_in_list(&class.name);
        drain_ui_queue();
        assert_eq!(state.get_rail_mode(), ResultsRailMode::List);
        assert_eq!(
            controller.list_filter.lock().unwrap().object_classes,
            vec![class.global()]
        );

        // An image's box: the List shows that image.
        controller.chart_filter.lock().unwrap().group_by = BoxplotGroup::Image;
        let image = {
            let images = controller.images.lock().unwrap();
            (images[1].name.clone(), images[1].rel_path.clone())
        };
        controller.show_box_in_list(&image.0);
        drain_ui_queue();
        assert_eq!(
            controller.list_filter.lock().unwrap().image_rel_path,
            vec![image.1]
        );
        let checked: Vec<bool> = state
            .get_list_image_items()
            .iter()
            .map(|i| i.selected)
            .collect();
        assert_eq!(
            checked.iter().filter(|c| **c).count(),
            1,
            "its box ticked in the list"
        );
    }

    #[test]
    fn open_database_populates_classes_columns_images_and_default_filters() {
        let (ui, controller) = controller_with_open_database();

        let classes = controller.classes.lock().unwrap();
        assert_eq!(classes.len(), 2);
        drop(classes);

        assert!(!controller.available_columns.lock().unwrap().is_empty());

        let images = controller.images.lock().unwrap();
        assert_eq!(images.len(), 3);
        drop(images);

        let list_filter = controller.list_filter.lock().unwrap();
        assert_eq!(list_filter.columns, DEFAULT_LIST_COLUMNS.to_vec());
        assert_eq!(
            list_filter.object_classes.len(),
            2,
            "every class selected by default"
        );
        drop(list_filter);

        assert!(controller.matrix_filter.lock().unwrap().is_some());
        assert!(controller.result_generator.lock().unwrap().is_some());
        assert!(!*controller.export_populated.lock().unwrap());

        // The List tab is on screen, so the list was queried as part of
        // opening - every object's location is cached. The hidden Matrix
        // tab wasn't, until it's shown: then both seeded wells get a cell.
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);
        assert!(controller.matrix_cells.lock().unwrap().is_empty());
        ui.global::<ResultsState>()
            .invoke_rail_mode_selected(ResultsRailMode::Matrix);
        let matrix_cells = controller.matrix_cells.lock().unwrap();
        assert!(matrix_cells.contains_key("A1"));
        assert!(matrix_cells.contains_key("A2"));
    }

    #[test]
    fn list_image_and_class_filters_narrow_the_cached_row_locations() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_list_image_selected("A1_01.tif".into(), true);
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 1);

        // "Select none" clears the image filter: no image selected means
        // all images.
        state.invoke_list_image_select_none();
        assert!(
            controller
                .list_filter
                .lock()
                .unwrap()
                .image_rel_path
                .is_empty()
        );
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);

        // Ticking one image after "Select none" filters to exactly that
        // one, and unticking it goes back to all.
        state.invoke_list_image_selected("A1_01.tif".into(), true);
        assert_eq!(
            controller.list_filter.lock().unwrap().image_rel_path,
            vec![PathBuf::from("A1_01.tif")]
        );
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 1);
        state.invoke_list_image_selected("A1_01.tif".into(), false);
        assert!(
            controller
                .list_filter
                .lock()
                .unwrap()
                .image_rel_path
                .is_empty()
        );
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);

        state.invoke_list_image_select_all();
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);

        // Deselect ClassB - only the two ClassA objects (both in well A1)
        // should remain.
        state.invoke_list_class_selected("ClassB".into(), false);
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 2);
        state.invoke_list_class_select_all();
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);
    }

    #[test]
    fn list_image_filter_changed_does_not_touch_the_actual_selection_or_row_cache() {
        // `filter_images`' own Slint push happens through
        // `invoke_from_event_loop`, which never actually runs under the
        // headless test platform's `init_no_event_loop` (see
        // `filtered_image_items`'s own doc comment) - so this only checks
        // what's independently observable: firing the callback doesn't
        // panic, and narrowing the dropdown's visible rows doesn't touch
        // the real query filter or the row cache it drives. The actual
        // filtering logic is unit-tested directly below instead.
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_list_image_filter_changed("A1".into());
        assert!(
            controller
                .list_filter
                .lock()
                .unwrap()
                .image_rel_path
                .is_empty()
        );
        assert_eq!(controller.list_row_locations.lock().unwrap().len(), 3);
    }

    #[test]
    fn export_dialog_open_seeds_image_selection_from_the_list_filter() {
        // Seeded images, alphabetically ("A1_01.tif" first) - see
        // `seed_test_db`.
        let (ui, controller) = controller_with_open_database();
        let list_state = ui.global::<ResultsState>();
        let export_state = ui.global::<ExportDialogState>();

        let selected_keys = |export_state: &ExportDialogState| -> Vec<String> {
            export_state
                .get_image_items()
                .iter()
                .filter(|item| item.selected)
                .map(|item| item.key.to_string())
                .collect()
        };

        // 1-50 images individually selected (here: 1, and deliberately not
        // the alphabetically-first one) -> the export dialog starts with
        // exactly those checked, not just the first.
        list_state.invoke_list_image_selected("A2_01.tif".into(), true);
        controller.populate_export_defaults(&ui);
        assert_eq!(selected_keys(&export_state), vec!["A2_01.tif".to_string()]);

        // Every image selected -> too many to usefully pre-check, defaults
        // back to just the first.
        list_state.invoke_list_image_select_all();
        controller.populate_export_defaults(&ui);
        assert_eq!(selected_keys(&export_state), vec!["A1_01.tif".to_string()]);

        // Nothing selected (= all images) -> same "just the
        // first" default, rather than a picker with nothing checked at all.
        list_state.invoke_list_image_select_none();
        controller.populate_export_defaults(&ui);
        assert_eq!(selected_keys(&export_state), vec!["A1_01.tif".to_string()]);
    }

    #[test]
    fn export_dialog_image_filter_changed_narrows_items_without_touching_selection() {
        let (ui, controller) = controller_with_open_database();
        let export_state = ui.global::<ExportDialogState>();
        controller.populate_export_defaults(&ui);

        // Starts with just "A1_01.tif" selected (the default - see the
        // test above), and, unlike the main List view's search box, this
        // one's own `on_image_filter_changed` handler runs synchronously
        // (no `invoke_from_event_loop`), so its effect is directly
        // observable here.
        export_state.invoke_image_filter_changed("A1".into());
        let keys: Vec<String> = export_state
            .get_image_items()
            .iter()
            .map(|item| item.key.to_string())
            .collect();
        assert_eq!(keys, vec!["A1_01.tif".to_string(), "A1_02.tif".to_string()]);
        assert!(
            export_state
                .get_image_items()
                .iter()
                .find(|item| item.key == "A1_01.tif")
                .unwrap()
                .selected,
            "the pre-existing selection must survive narrowing the view"
        );

        export_state.invoke_image_filter_changed("".into());
        assert_eq!(export_state.get_image_items().row_count(), 3);
    }

    // Clicks an export-dialog IMAGES row the way `InkMultiSelect`
    // does: flips the row in place, then fires `image-item-selected`.
    fn click_export_image(state: &ExportDialogState, key: &str, selected: bool) {
        let items = state.get_image_items();
        let row = (0..items.row_count())
            .find(|&i| items.row_data(i).unwrap().key == key)
            .unwrap_or_else(|| panic!("{key} is not shown in the picker"));
        let mut item = items.row_data(row).unwrap();
        item.selected = selected;
        items.set_row_data(row, item);
        state.invoke_image_item_selected(key.into(), selected);
    }

    fn exported_images(controller: &ResultsStateController, ui: &AppWindow) -> Vec<String> {
        let mut paths = controller
            .read_export_settings(ui)
            .expect("valid export settings")
            .image_rel_paths;
        paths.sort();
        paths
    }

    #[test]
    fn export_dialog_keeps_images_selected_under_an_earlier_search() {
        // Regression: the picker's selection used to be read back from its
        // search-narrowed rows, so images checked under one search were
        // dropped from the export (and the "N of M" count) by the next.
        let (ui, controller) = controller_with_open_database();
        controller.populate_export_defaults(&ui);
        let state = ui.global::<ExportDialogState>();
        state.set_output_dir("out".into());
        state.set_with_list_view(true);

        state.invoke_image_select_none();
        state.invoke_image_filter_changed("A1".into());
        click_export_image(&state, "A1_01.tif", true);
        click_export_image(&state, "A1_02.tif", true);
        state.invoke_image_filter_changed("A2".into());
        click_export_image(&state, "A2_01.tif", true);

        let all = vec!["A1_01.tif", "A1_02.tif", "A2_01.tif"];
        assert_eq!(state.get_image_summary(), image_summary_text(3, 3));
        assert_eq!(exported_images(&controller, &ui), all);

        // Clearing the search shows every checkmark again.
        state.invoke_image_filter_changed("".into());
        assert!(state.get_image_items().iter().all(|item| item.selected));
        assert_eq!(exported_images(&controller, &ui), all);

        // Unchecking a visible row leaves the hidden selections alone.
        state.invoke_image_filter_changed("A2".into());
        click_export_image(&state, "A2_01.tif", false);
        assert_eq!(
            exported_images(&controller, &ui),
            vec!["A1_01.tif", "A1_02.tif"]
        );
        assert_eq!(state.get_image_summary(), image_summary_text(2, 3));
    }

    #[test]
    fn export_dialog_select_all_and_none_keep_the_current_search() {
        let (ui, controller) = controller_with_open_database();
        controller.populate_export_defaults(&ui);
        let state = ui.global::<ExportDialogState>();
        state.set_output_dir("out".into());

        state.invoke_image_filter_changed("A1".into());
        state.invoke_image_select_all();
        let shown: Vec<String> = state
            .get_image_items()
            .iter()
            .map(|item| item.key.to_string())
            .collect();
        assert_eq!(shown, vec!["A1_01.tif", "A1_02.tif"]);
        assert_eq!(
            exported_images(&controller, &ui),
            vec!["A1_01.tif", "A1_02.tif", "A2_01.tif"]
        );

        state.invoke_image_select_none();
        assert_eq!(state.get_image_items().row_count(), 2);
        assert!(state.get_image_items().iter().all(|item| !item.selected));
        assert_eq!(state.get_image_summary(), image_summary_text(0, 3));
    }

    #[test]
    fn export_dialog_with_no_image_selected_exports_all_images() {
        let (ui, controller) = controller_with_open_database();
        controller.populate_export_defaults(&ui);
        let state = ui.global::<ExportDialogState>();
        state.set_output_dir("out".into());
        state.set_with_list_view(true);
        state.invoke_image_select_none();

        assert_eq!(state.get_image_summary(), "All Images");
        // An empty list is the exporter's "every image".
        assert!(exported_images(&controller, &ui).is_empty());
    }

    #[test]
    fn filtered_image_items_matches_by_name_case_insensitively_and_keeps_real_selection() {
        let images = vec![
            ImageEntry {
                name: "A1_01.tif".into(),
                rel_path: PathBuf::from("A1_01.tif"),
                disabled: false,
            },
            ImageEntry {
                name: "A1_02.tif".into(),
                rel_path: PathBuf::from("A1_02.tif"),
                disabled: false,
            },
            ImageEntry {
                name: "A2_01.tif".into(),
                rel_path: PathBuf::from("A2_01.tif"),
                disabled: false,
            },
        ];
        let selected_paths = vec![PathBuf::from("A1_01.tif")];
        let is_selected = |path: &Path| selected_paths.iter().any(|p| p == path);

        let matches = filtered_image_items(&images, is_selected, "a1");
        let keys: Vec<String> = matches.iter().map(|item| item.key.to_string()).collect();
        assert_eq!(keys, vec!["A1_01.tif".to_string(), "A1_02.tif".to_string()]);
        // The already-selected image keeps showing as selected even though
        // the dropdown was just narrowed, the not-yet-selected match doesn't.
        assert!(matches[0].selected);
        assert!(!matches[1].selected);

        // An empty query keeps every image.
        assert_eq!(filtered_image_items(&images, is_selected, "").len(), 3);
        // A query matching nothing returns an empty (not panicking) list.
        assert!(filtered_image_items(&images, is_selected, "zzz").is_empty());
    }

    #[test]
    fn list_column_widths_fits_the_longest_header_or_cell_and_caps_at_260px() {
        let headers: Vec<slint::SharedString> = vec!["ID".into(), "Name".into(), "Wide".into()];
        // Column 0: header "ID" (2 chars) is the widest thing in it.
        // Column 1: a body cell ("a_very_long_object_identifier", 30 chars)
        // is wider than its header ("Name", 4 chars).
        // Column 2: a cell long enough to blow past the 260px cap.
        let rows: Vec<Vec<slint::SharedString>> = vec![
            vec!["1".into(), "short".into(), "x".repeat(200).into()],
            vec![
                "2".into(),
                "a_very_long_object_identifier".into(),
                "y".into(),
            ],
        ];
        let widths = list_column_widths(&headers, rows.iter().map(|row| row.as_slice()));

        assert_eq!(widths.len(), 3);
        // Column 0 stays near its header's own natural width (well under
        // the cap, and clamped up to the 60px floor rather than a
        // barely-there sliver).
        assert_eq!(widths[0], 60.0);
        // Column 1 grew to fit the long cell, not just the short header.
        assert!(
            widths[1] > 100.0 && widths[1] < 260.0,
            "expected column 1 to fit its long cell without hitting the cap, got {}",
            widths[1]
        );
        // Column 2's 200-char cell is clamped at the max.
        assert_eq!(widths[2], 260.0);
    }

    #[test]
    fn list_columns_group_by_coloc_details_and_pagination_do_not_panic() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_list_columns_item_selected("area_px".into(), true);
        assert!(
            controller
                .list_filter
                .lock()
                .unwrap()
                .columns
                .contains(&Column::AreaSizePx)
        );
        state.invoke_list_columns_select_none();
        assert!(controller.list_filter.lock().unwrap().columns.is_empty());
        state.invoke_list_columns_select_all();
        assert!(!controller.list_filter.lock().unwrap().columns.is_empty());

        state.invoke_list_aggregation_item_selected("min".into(), true);
        state.invoke_list_aggregation_select_none();
        state.invoke_list_aggregation_select_all();

        state.invoke_list_with_coloc_details_changed(true);
        assert!(controller.list_filter.lock().unwrap().with_coloc_details);

        state.invoke_list_transpond_changed(true);
        assert!(controller.list_filter.lock().unwrap().transpond);
        state.invoke_list_transpond_changed(false);
        assert!(!controller.list_filter.lock().unwrap().transpond);

        state.invoke_list_group_by_selected("images".into());
        assert_eq!(
            controller.list_row_locations.lock().unwrap().len(),
            0,
            "grouped view has no per-object rows"
        );
        state.invoke_list_group_by_selected("objects".into());

        state.invoke_list_next_page();
        assert_eq!(*controller.list_page.lock().unwrap(), 1);
        state.invoke_list_prev_page();
        assert_eq!(*controller.list_page.lock().unwrap(), 0);
        // Clamped, not negative.
        state.invoke_list_prev_page();
        assert_eq!(*controller.list_page.lock().unwrap(), 0);

        // No real image file on disk to open, but this must not panic.
        state.invoke_list_row_clicked(0);
    }

    #[test]
    fn the_status_says_how_many_images_and_how_many_are_left_out() {
        let image = |name: &str, disabled| ImageEntry {
            name: name.into(),
            rel_path: name.into(),
            disabled,
        };
        assert_eq!(analysis_summary(&[image("a", false)]), "1 image");
        assert_eq!(
            analysis_summary(&[image("a", false), image("b", true), image("c", false)]),
            "3 images · 1 disabled"
        );
    }

    #[test]
    fn object_ids_are_shown_by_their_last_block_and_bars_can_be_log() {
        assert_eq!(
            short_object_id("01a1497c-887f-7d95-93c3-bfc847366b32"),
            "bfc847366b32"
        );
        assert_eq!(short_object_id("D15_03.vsi"), "D15_03.vsi");
        assert_eq!(histogram_bar(10, 100, false), 0.1);
        assert_eq!(histogram_bar(100, 100, true), 1.0);
        assert!(
            histogram_bar(1, 1000, true) > 0.09,
            "small counts still show"
        );
        assert_eq!(scatter_points_limit(3), None);
    }

    /// A click in the scatter opens the nearest point's object - only
    /// within 8 px of it.
    #[test]
    fn a_click_in_the_scatter_finds_the_nearest_point_close_by() {
        let points = vec![
            (0.1, 0.1, "a.tif".to_string(), [1, 1, 2, 2]),
            (0.5, 0.5, "b.tif".to_string(), [5, 5, 6, 6]),
        ];
        // 400 x 200 px: (0.51, 0.5) is 4 px from b.
        assert_eq!(
            nearest_point(&points, 0.51, 0.5, 400.0, 200.0, 8.0),
            Some(("b.tif".to_string(), [5, 5, 6, 6]))
        );
        // 0.1 of 400 px = 40 px away from both: nothing.
        assert_eq!(nearest_point(&points, 0.3, 0.1, 400.0, 200.0, 8.0), None);
    }

    /// A List row's object on its image: the window goes to the image, the
    /// row stays selected, the bar on the image says which (and steps).
    #[test]
    fn a_clicked_row_peeks_at_its_object_on_the_image() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();
        ui.global::<crate::LayoutState>()
            .set_mode(crate::WorkspaceMode::Results);
        let rows = controller.list_row_locations.lock().unwrap().len();
        assert!(rows > 1, "{rows} rows");

        state.invoke_list_row_clicked(1);
        drain_ui_queue();

        assert!(state.get_peek());
        assert!(state.get_peek_can_step());
        assert_eq!(state.get_list_selected_row(), 1);
        assert_eq!(state.get_peek_label(), format!("Row 2 of {rows}"));
        assert_eq!(
            ui.global::<crate::LayoutState>().get_mode(),
            crate::WorkspaceMode::Images
        );
    }

    #[test]
    fn matrix_toolbar_callbacks_update_the_filter_and_refresh_the_grid() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_matrix_value_clicked();
        state.invoke_matrix_aggregate_selected("Minimum".into());
        assert_eq!(
            controller
                .matrix_filter
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .aggregation,
            Aggregation::Min
        );

        state.invoke_matrix_class_selected("ClassB".into(), true);
        assert_eq!(
            controller
                .matrix_filter
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .object_classe,
            ObjectClass::Global(2)
        );

        state.invoke_matrix_regex_changed(r"^(([A-H])([0-9]{1,2}))_([0-9]+)\.".into());
        state.invoke_matrix_color_schema_selected("Viridis".into(), true);
        state.invoke_matrix_scale_set_manual(0.0, 100.0);
        assert!(matches!(
            controller
                .matrix_filter
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .color_scale,
            ColorScale::Manual(_, _)
        ));
        state.invoke_matrix_scale_set_auto();
        assert!(matches!(
            controller
                .matrix_filter
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .color_scale,
            ColorScale::Auto
        ));

        state.invoke_matrix_plate_size_selected("96 Well (8 x 12)".into(), true);
        state.invoke_matrix_well_rows_changed("6".into());
        state.invoke_matrix_well_cols_changed("6".into());
        let plate = controller
            .matrix_filter
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .plate
            .clone();
        assert_eq!(plate.plate_size, PlateSize::Plate8x12);
        assert_eq!(plate.well_layout, WellLayout::Fixed { rows: 6, cols: 6 });
        state.invoke_matrix_square_size_selected("128".into(), true);
        assert_eq!(
            controller
                .matrix_filter
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .square_size,
            Some(128)
        );
    }

    #[test]
    fn plate_well_and_image_heatmap_drill_down_all_the_way() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        state.invoke_plate_cell_clicked("A1".into());
        assert_eq!(state.get_active_well(), "A1");

        state.invoke_open_well_clicked("A1".into());
        assert_eq!(
            controller.current_well.lock().unwrap().as_deref(),
            Some("A1")
        );
        let well_cells = controller.well_cells.lock().unwrap();
        assert!(well_cells.contains_key("A1_01.tif"));
        assert!(well_cells.contains_key("A1_02.tif"));
        drop(well_cells);

        state.invoke_well_cell_clicked("A1_01.tif".into());
        assert_eq!(state.get_active_well(), "A1_01.tif");

        state.invoke_well_field_clicked("A1_01.tif".into());
        assert_eq!(
            controller.current_image.lock().unwrap().as_deref(),
            Some("A1_01.tif")
        );
        let heatmap_cells = controller.image_heatmap_cells.lock().unwrap();
        assert!(!heatmap_cells.is_empty());
        let tile_key = heatmap_cells.keys().next().cloned().unwrap();
        drop(heatmap_cells);

        state.invoke_image_heatmap_cell_clicked(tile_key.clone().into());
        assert_eq!(state.get_active_well(), tile_key);

        // Breadcrumb nav back to the plate level, and the plain stubs.
        state.invoke_breadcrumb_nav(0);
        assert_eq!(controller.current_well.lock().unwrap().as_deref(), None);
        state.invoke_matrix_back_to_plate();
        state.invoke_matrix_back_to_well();
        state.invoke_object_marker_clicked(0);
    }

    /// Regression test: disabling an image from the well grid, then
    /// navigating back to the plate grid via the breadcrumb, must show the
    /// well's now-updated aggregate (its disabled image's objects excluded)
    /// - previously `plate_cells` was left holding whatever was cached from
    /// before the well was even opened, since landing back on
    /// `MatrixLevel::Plate` via the breadcrumb never re-queried
    /// `get_group_by_plate`. The well itself must still never be flagged
    /// `disabled` though - only individual images are - so a well with one
    /// enabled and one disabled image keeps its normal heatmap color; it
    /// must instead be flagged `any_disabled`, the small-badge signal the
    /// plate view uses to mark a well that contains a disabled image
    /// without recoloring it.
    #[test]
    fn breadcrumb_nav_back_to_plate_refreshes_stale_plate_cells() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        // Count/Avg is the same (1.0) with or without A1_01 (one object per
        // image), so use Sum: 2 before disabling A1_01, 1 after.
        state.invoke_matrix_aggregate_selected("Sum".into());

        state.invoke_plate_cell_clicked("A1".into());
        let cells = controller.matrix_cells.lock().unwrap();
        let before = cells.get("A1").unwrap();
        assert!(!before.disabled, "a well is never itself flagged disabled");
        assert!(!before.any_disabled, "A1 has no disabled images yet");
        let value_before = before.value;
        drop(cells);

        state.invoke_open_well_clicked("A1".into());
        state.invoke_well_cell_clicked("A1_01.tif".into());
        state.invoke_toggle_active_well_disabled();

        // Breadcrumb: ["All results", "Plate", "Well A1"] - index 1 ("Plate")
        // truncates to the plate level.
        state.invoke_breadcrumb_nav(1);

        let cells = controller.matrix_cells.lock().unwrap();
        let after = cells.get("A1").unwrap();
        assert!(
            !after.disabled,
            "the well must still not be flagged disabled after refreshing"
        );
        assert!(
            after.any_disabled,
            "A1's cached plate cell must reflect that one of its images is now disabled, \
             so the plate view can badge it"
        );
        assert_ne!(
            after.value, value_before,
            "A1's cached plate cell must reflect the image just disabled, not stale pre-drill-down data"
        );
    }

    /// Times opening a real results file and toggling one of its images
    /// through the controller, including building the Slint models.
    /// `EVADB=<copy.evadb> cargo test --release -p evanalyzer_gui --lib
    /// bench_real_database -- --ignored --nocapture` (toggles an image, so
    /// use a copy).
    #[test]
    #[ignore]
    fn bench_real_database() {
        use std::time::Instant;
        let Ok(path) = std::env::var("EVADB") else {
            return;
        };
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.attach_callbacks();
        let state = ui.global::<ResultsState>();
        for i in 0..3 {
            let start = Instant::now();
            controller.open_database(PathBuf::from(&path));
            let opened = start.elapsed();
            drain_ui_queue();
            println!(
                "open #{i}: open_database {opened:.2?}, + UI queue {:.2?}, list rows {}",
                start.elapsed(),
                state.get_list_rows().row_count()
            );
        }
        let image_name = controller.images.lock().unwrap()[0].name.clone();
        let well = image_name.split('_').next().unwrap().to_string();
        state.invoke_plate_cell_clicked(well.clone().into());
        state.invoke_open_well_clicked(well.into());
        drain_ui_queue();
        for i in 0..4 {
            state.invoke_well_cell_clicked(image_name.clone().into());
            let start = Instant::now();
            state.invoke_toggle_active_well_disabled();
            let toggled = start.elapsed();
            drain_ui_queue();
            println!(
                "toggle #{i}: {toggled:.2?}, + UI queue {:.2?}",
                start.elapsed()
            );
        }
    }

    /// Times the results views on a real results file and the process's
    /// RAM (now / peak) after each: opening, List, Plate, Charts, and the
    /// split view (a row's objects read and drawn at 1920 x 1080).
    /// `EVADB=<file.evadb> cargo test --release -p evanalyzer_gui --lib
    /// bench_results_views -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_results_views() {
        use std::time::Instant;
        let Ok(path) = std::env::var("EVADB") else {
            return;
        };
        let rss = || {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            let kb = |key: &str| {
                status
                    .lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.0)
                    / 1024.0
            };
            (kb("VmRSS:"), kb("VmHWM:"))
        };
        let report = |name: &str, took: std::time::Duration| {
            let (now, peak) = rss();
            println!(
                "{name:<34} {:>9.1} ms {:>8.0} MB {:>8.0} MB",
                took.as_secs_f64() * 1000.0,
                now,
                peak
            );
        };
        println!("{:<34} {:>12} {:>11} {:>11}", "step", "time", "RSS", "peak");
        report("start", std::time::Duration::ZERO);

        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.attach_callbacks();
        let state = ui.global::<ResultsState>();

        let start = Instant::now();
        controller.open_database(PathBuf::from(&path));
        drain_ui_queue();
        report("open (List, first page)", start.elapsed());
        println!(
            "  {} rows on the page, {} images",
            state.get_list_rows().row_count(),
            controller.images.lock().unwrap().len()
        );

        for (name, mode) in [
            ("Plate", ResultsRailMode::Matrix),
            ("Charts (histogram)", ResultsRailMode::Charts),
            ("List again", ResultsRailMode::List),
        ] {
            let start = Instant::now();
            state.set_rail_mode(mode);
            state.invoke_rail_mode_selected(mode);
            drain_ui_queue();
            report(name, start.elapsed());
        }

        // Split: the first row's object on its image - its objects read
        // (the project doesn't hold the images: read straight from the db).
        let (rel_path, _) = controller.list_row_locations.lock().unwrap()[0].clone();
        let start = Instant::now();
        let objects = controller
            .result_generator
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .image_objects(&evanalyzer_app::results::ImageObjectsFilter {
                image_rel_path: rel_path.clone(),
                plane: evanalyzer_app::results::PlaneFilter {
                    z_stack: 0,
                    t_stack: 0,
                },
                value_column: None,
            })
            .unwrap();
        report(
            &format!("split: objects of one image ({})", objects.len()),
            start.elapsed(),
        );
        let overlay = crate::editor::results_overlay::ResultsOverlay {
            objects: objects.iter().map(|o| (o.bbox_px, [255, 0, 255])).collect(),
            selected: Some(0),
        };
        let (w, h) = (1920 + 512, 1080 + 512);
        let max_x = objects.iter().map(|o| o.bbox_px[2]).max().unwrap_or(1) as f32;
        let max_y = objects.iter().map(|o| o.bbox_px[3]).max().unwrap_or(1) as f32;
        for (name, zoom) in [
            ("fitted", (1920.0 / max_x).min(1080.0 / max_y)),
            ("zoom 1", 1.0f32),
        ] {
            let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(w, h);
            let start = Instant::now();
            crate::editor::results_overlay::draw_results_overlay(
                buffer.make_mut_slice(),
                &overlay,
                w,
                h,
                zoom,
                256.0,
                256.0,
            );
            report(
                &format!("split: draw the ellipses, {name}"),
                start.elapsed(),
            );
        }
    }

    /// Leak check: the results opened 15 times (`BENCH_ROUNDS`), each time List, Plate,
    /// Charts and the split view's selection - RAM must level off.
    /// `EVADB=<file.evadb> cargo test --release -p evanalyzer_gui --lib
    /// bench_soak_results -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_soak_results() {
        let Ok(path) = std::env::var("EVADB") else {
            return;
        };
        let rss = || {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            status
                .lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
                / 1024.0
        };
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.attach_callbacks();
        let state = ui.global::<ResultsState>();
        let start = rss();
        let rounds: usize = std::env::var("BENCH_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);
        // BENCH_PART=open: only reopening; =views: opened once, then only
        // the views and the split's selections.
        let part = std::env::var("BENCH_PART").unwrap_or_default();
        if part == "views" {
            controller.open_database(PathBuf::from(&path));
            drain_ui_queue();
        }
        for round in 1..=rounds {
            if part != "views" {
                controller.open_database(PathBuf::from(&path));
                drain_ui_queue();
            }
            if part == "open" {
                #[cfg(all(target_os = "linux", target_env = "gnu"))]
                unsafe {
                    libc::malloc_trim(0);
                }
                if round % 5 == 0 || round == 1 {
                    println!("round {round:>2}: RSS +{:.0} MB", rss() - start);
                }
                continue;
            }
            for mode in [
                ResultsRailMode::Matrix,
                ResultsRailMode::Charts,
                ResultsRailMode::List,
            ] {
                state.set_rail_mode(mode);
                state.invoke_rail_mode_selected(mode);
                drain_ui_queue();
            }
            state.set_split(true);
            for row in 0..20 {
                state.invoke_list_row_selected(row);
            }
            drain_ui_queue();
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            unsafe {
                libc::malloc_trim(0);
            }
            if round % 5 == 0 || round == 1 {
                println!("round {round:>2}: RSS +{:.0} MB", rss() - start);
            }
        }
    }

    #[test]
    fn toggle_active_well_disabled_flips_the_database_and_refreshes_the_well_grid() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_plate_cell_clicked("A1".into());
        state.invoke_open_well_clicked("A1".into());
        state.invoke_well_cell_clicked("A1_01.tif".into());
        assert_eq!(state.get_active_well(), "A1_01.tif");
        assert!(!state.get_active_well_disabled());

        state.invoke_toggle_active_well_disabled();
        assert!(
            state.get_active_well_disabled(),
            "the detail card must reflect the new disabled state without a re-click"
        );
        assert_eq!(
            state.get_active_well(),
            "A1_01.tif",
            "the same field must stay selected across the refresh the toggle triggers"
        );
        let db = controller.result_generator.lock().unwrap();
        let images = db.as_ref().unwrap().get_images().unwrap();
        drop(db);
        assert!(
            images
                .iter()
                .find(|image| image.name == "A1_01.tif")
                .unwrap()
                .disabled,
            "the underlying database must actually be updated"
        );

        // Toggling again flips it back.
        state.invoke_toggle_active_well_disabled();
        assert!(!state.get_active_well_disabled());
        let db = controller.result_generator.lock().unwrap();
        let images = db.as_ref().unwrap().get_images().unwrap();
        drop(db);
        assert!(
            !images
                .iter()
                .find(|image| image.name == "A1_01.tif")
                .unwrap()
                .disabled
        );
    }

    #[test]
    fn toggle_active_well_disabled_with_nothing_selected_does_not_panic() {
        let (ui, _controller) = controller_with_open_database();
        ui.global::<ResultsState>()
            .invoke_toggle_active_well_disabled();
    }

    #[test]
    fn well_cell_and_image_heatmap_cell_click_with_an_unknown_key_warns_but_does_not_panic() {
        let (ui, _controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();
        state.invoke_plate_cell_clicked("Z99".into());
        state.invoke_well_cell_clicked("does-not-exist.tif".into());
        state.invoke_well_field_clicked("does-not-exist.tif".into());
        state.invoke_image_heatmap_cell_clicked("R99C99".into());
    }

    #[test]
    fn chart_callbacks_switch_kind_and_axes_without_panicking() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();

        state.invoke_chart_column_selected("area_px".into());
        assert_eq!(
            controller.chart_filter.lock().unwrap().column,
            Column::AreaSizePx
        );

        state.invoke_chart_kind_selected(ResultsChartKind2::Scatter);
        state.invoke_chart_y_column_selected("area_px".into());
        assert_eq!(
            controller.chart_filter.lock().unwrap().kind,
            ChartKind::Scatter
        );

        state.invoke_chart_kind_selected(ResultsChartKind2::Boxplot);
        state.invoke_chart_class_selected("ClassA".into(), true);
        assert_eq!(
            controller.chart_filter.lock().unwrap().object_class,
            Some(ObjectClass::Global(1))
        );

        state.invoke_chart_kind_selected(ResultsChartKind2::Histogram);
    }

    #[test]
    fn export_dialog_select_all_and_select_none_callbacks_do_not_panic() {
        let (ui, _controller) = controller_with_open_database();
        ui.global::<ResultsState>().invoke_export_dialog_open();
        let state = ui.global::<ExportDialogState>();

        state.invoke_image_select_all();
        state.invoke_image_select_none();
        state.invoke_class_select_all();
        state.invoke_class_select_none();
        state.invoke_column_select_all();
        state.invoke_column_select_none();
        state.invoke_aggregation_item_selected("avg".into(), true);

        let image_key = state.get_image_items().iter().next().map(|item| item.key);
        if let Some(key) = image_key {
            state.invoke_image_item_selected(key, true);
        }
        let class_key = state.get_class_items().iter().next().map(|item| item.key);
        if let Some(key) = class_key {
            state.invoke_class_item_selected(key, true);
        }
        let column_key = state.get_column_items().iter().next().map(|item| item.key);
        if let Some(key) = column_key {
            state.invoke_column_item_selected(key, true);
        }
    }

    #[test]
    fn z_and_t_changed_update_the_plane_filter_and_refresh() {
        let (ui, controller) = controller_with_open_database();
        let state = ui.global::<ResultsState>();
        state.invoke_z_changed(2);
        state.invoke_t_changed(3);
        let plane = controller.plane_filter.lock().unwrap();
        assert_eq!(plane.selected_z_stack, 2);
        assert_eq!(plane.selected_t_stack, 3);
    }

    // -- what the results view shows (UI updates applied) --------------------

    use crate::editor::test_support::{choose_file, ui_state_with_window};
    use crate::helper::ui_thread::drain_ui_queue;

    // -- plate settings: shared with the project ------------------------------

    /// A database opened with `plate` as the project's plate settings, and
    /// the project settings dialog linked like in the app.
    fn opened_with_plate(
        plate: PlateSettings,
    ) -> (
        Opened,
        Arc<crate::editor::project_settings_controller::ProjectSettingsController>,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        ui_state.get_project_write().plate = plate;
        let controller = make_controller_with_state(ui.as_weak(), ui_state.clone());
        let project_settings = Arc::new(
            crate::editor::project_settings_controller::ProjectSettingsController::new(
                ui.as_weak(),
                ui_state,
            ),
        );
        crate::editor::link_plate_settings(&project_settings, &controller, None);
        controller.attach_callbacks();
        controller.open_database(path);
        drain_ui_queue();
        (
            Opened {
                ui,
                controller,
                _dir: dir,
            },
            project_settings,
        )
    }

    fn matrix_plate(o: &Opened) -> PlateSettings {
        o.controller
            .matrix_filter
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .plate
            .clone()
    }

    #[test]
    fn opening_results_uses_the_projects_plate_settings() {
        let plate = PlateSettings {
            grouping_mode: GroupingMode::Folder,
            plate_size: PlateSize::Plate8x12,
            well_layout: WellLayout::Fixed { rows: 3, cols: 3 },
            well_image_order: (1..=9).rev().collect(),
            ..PlateSettings::default()
        };
        let (o, _project_settings) = opened_with_plate(plate.clone());
        let s = o.state();

        assert_eq!(matrix_plate(&o), plate);
        assert_eq!(s.get_matrix_grouping_summary(), "Folder");
        assert!(!s.get_matrix_group_regex_custom());
        assert_eq!(s.get_matrix_plate_size_summary(), "96 Well (8 x 12)");
        assert_eq!(s.get_matrix_well_layout_summary(), "Custom");
        assert_eq!(s.get_matrix_well_rows(), "3");

        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        drain_ui_queue();
        assert_eq!(s.get_plate_rows(), 8, "the project's plate size");
        assert_eq!(s.get_plate_cols(), 12);
    }

    #[test]
    fn changes_in_the_results_window_go_to_the_project_and_its_settings_dialog() {
        // Kept alive, like the app's editor does.
        let (o, _project_settings) = opened_with_plate(PlateSettings::default());
        let s = o.state();
        let ui_state = o.controller._app_state.clone();
        assert!(!ui_state.is_dirty());

        s.invoke_matrix_plate_size_selected("384 Well (16 x 24)".into(), true);
        s.invoke_matrix_well_layout_selected("Custom".into(), true);
        s.invoke_matrix_well_cols_changed("3".into());
        drain_ui_queue();

        let project_plate = ui_state.get_project().plate.clone();
        assert_eq!(project_plate.plate_size, PlateSize::Plate16x24);
        assert!(matches!(
            project_plate.well_layout,
            WellLayout::Fixed { cols: 3, .. }
        ));
        assert_eq!(project_plate, matrix_plate(&o));
        assert!(ui_state.is_dirty(), "an unsaved change of the project");

        // The project settings dialog shows it too.
        let dialog = o.ui.global::<crate::ProjectSettingsState>().get_settings();
        assert_eq!(
            dialog.plate_size_index,
            PlateSize::ALL
                .iter()
                .position(|p| *p == PlateSize::Plate16x24)
                .unwrap() as i32
        );
        assert!(!dialog.well_auto);
        assert_eq!(dialog.well_columns, 3);
    }

    #[test]
    fn changes_in_the_project_settings_reach_the_results_window() {
        let (o, project_settings) = opened_with_plate(PlateSettings::default());
        let s = o.state();
        let ui_state = o.controller._app_state.clone();

        // As the project settings dialog's Apply does.
        ui_state.get_project_write().plate = PlateSettings {
            grouping_mode: GroupingMode::Custom,
            grouping_regex: r"^(([A-H])([0-9]+))_([0-9]+)".into(),
            plate_size: PlateSize::Plate4x6,
            ..PlateSettings::default()
        };
        project_settings.sync_project_settings_to_slint();
        drain_ui_queue();

        let plate = matrix_plate(&o);
        assert_eq!(plate.plate_size, PlateSize::Plate4x6);
        assert_eq!(s.get_matrix_grouping_summary(), "Custom regex");
        assert!(s.get_matrix_group_regex_custom());
        assert_eq!(s.get_matrix_group_regex(), r"^(([A-H])([0-9]+))_([0-9]+)");
        assert_eq!(s.get_matrix_plate_size_summary(), "24 Well (4 x 6)");
    }

    #[test]
    fn auto_grouping_shows_the_detected_regex_and_custom_starts_from_it() {
        let (o, _project_settings) = opened_with_plate(PlateSettings::default());
        let s = o.state();
        let detected = s.get_matrix_group_regex_in_use();
        assert!(!detected.is_empty(), "the detected pattern is shown");
        assert!(!s.get_matrix_group_regex_custom());

        s.invoke_matrix_grouping_selected("Custom regex".into(), true);
        drain_ui_queue();
        assert!(s.get_matrix_group_regex_custom());
        assert_eq!(s.get_matrix_group_regex(), detected);
        let project = o.controller._app_state.get_project().plate.clone();
        assert_eq!(project.grouping_mode, GroupingMode::Custom);
        assert_eq!(project.grouping_regex, detected.as_str());

        s.invoke_matrix_regex_changed("^(x)".into());
        drain_ui_queue();
        assert_eq!(
            o.controller._app_state.get_project().plate.grouping_regex,
            "^(x)"
        );
        assert_eq!(s.get_matrix_group_regex_in_use(), "^(x)");
    }

    /// What a double click on a tile (and the detail card's "Open" button)
    /// does: plate -> the well, well -> the image.
    #[test]
    fn open_cell_goes_one_level_deeper() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        drain_ui_queue();

        s.invoke_open_cell("A1".into());
        drain_ui_queue();
        assert_eq!(s.get_matrix_level(), MatrixLevel::Well);
        assert_eq!(
            o.controller.current_well.lock().unwrap().as_deref(),
            Some("A1")
        );
        assert!(s.get_well_fields().row_count() >= 2);

        s.invoke_open_cell("A1_01.tif".into());
        drain_ui_queue();
        assert_eq!(s.get_matrix_level(), MatrixLevel::Object);
        assert!(o.controller.current_image.lock().unwrap().is_some());
        assert!(s.get_image_heatmap_cells().row_count() > 0);

        // Nothing deeper than the image heatmap.
        s.invoke_open_cell("R0C0".into());
        drain_ui_queue();
        assert_eq!(s.get_matrix_level(), MatrixLevel::Object);
    }

    /// The image keys of the well grid, row by row ("" = no image there).
    fn well_grid_keys(s: &ResultsState) -> Vec<String> {
        s.get_well_fields()
            .iter()
            .map(|cell| {
                if cell.exists {
                    cell.key.to_string()
                } else {
                    String::new()
                }
            })
            .collect()
    }

    #[test]
    fn dropping_an_image_on_another_position_swaps_them_in_the_project() {
        let (o, _project_settings) = opened_with_plate(PlateSettings::default());
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        s.invoke_open_cell("A1".into());
        drain_ui_queue();
        assert_eq!(
            well_grid_keys(&s),
            vec!["A1_01.tif", "A1_02.tif"],
            "Auto: 1 x 2"
        );

        s.invoke_well_image_moved(0, 1);
        drain_ui_queue();
        let plate = o.controller._app_state.get_project().plate.clone();
        assert_eq!(
            plate.well_layout,
            WellLayout::Fixed { rows: 1, cols: 2 },
            "Auto becomes the grid that was shown"
        );
        assert_eq!(plate.well_image_order, vec![2, 1]);
        assert!(o.controller._app_state.is_dirty());
        assert_eq!(well_grid_keys(&s), vec!["A1_02.tif", "A1_01.tif"]);
        assert_eq!(s.get_matrix_well_layout_summary(), "Custom");

        // Back again; dropping on itself changes nothing.
        s.invoke_well_image_moved(1, 0);
        s.invoke_well_image_moved(1, 1);
        drain_ui_queue();
        assert_eq!(
            o.controller._app_state.get_project().plate.well_image_order,
            vec![1, 2]
        );
        assert_eq!(well_grid_keys(&s), vec!["A1_01.tif", "A1_02.tif"]);
    }

    /// The real gestures in the (headless) window: a double click on a plate
    /// tile opens the well, a plain click on a well tile selects it, and
    /// pressing on the first tile, moving to the second and releasing swaps
    /// them.
    #[test]
    fn dragging_and_double_clicking_tiles_in_the_window() {
        use i_slint_backend_testing::ElementHandle;
        use slint::LogicalPosition;
        use slint::platform::{PointerEventButton, WindowEvent};

        let (o, _project_settings) = opened_with_plate(PlateSettings::default());
        let s = o.state();
        let window = o.ui.window();
        window.set_size(slint::LogicalSize::new(1500.0, 900.0));
        s.set_rail_mode(ResultsRailMode::Matrix);
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        drain_ui_queue();

        let tiles = || -> Vec<(String, LogicalPosition)> {
            ElementHandle::find_by_element_type_name(&o.ui, "MatrixCellView")
                .filter(|tile| tile.size().width > 0.0)
                .map(|tile| {
                    let p = tile.absolute_position();
                    let size = tile.size();
                    (
                        tile.accessible_label().unwrap_or_default().to_string(),
                        LogicalPosition::new(p.x + size.width / 2.0, p.y + size.height / 2.0),
                    )
                })
                .collect()
        };
        let press = |at: LogicalPosition| {
            window.dispatch_event(WindowEvent::PointerMoved { position: at });
            window.dispatch_event(WindowEvent::PointerPressed {
                position: at,
                button: PointerEventButton::Left,
            });
        };
        let release = |at: LogicalPosition| {
            window.dispatch_event(WindowEvent::PointerReleased {
                position: at,
                button: PointerEventButton::Left,
            });
        };

        // Double click on the plate's first tile (A1).
        let plate_tiles = tiles();
        assert!(!plate_tiles.is_empty(), "plate tiles found in the window");
        let a1 = plate_tiles[0].1;
        press(a1);
        release(a1);
        i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_millis(50));
        press(a1);
        release(a1);
        drain_ui_queue();
        assert_eq!(
            s.get_matrix_level(),
            MatrixLevel::Well,
            "double click opened the well"
        );
        assert_eq!(
            o.controller.current_well.lock().unwrap().as_deref(),
            Some("A1")
        );
        i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_millis(1000));

        // A plain click on the second image selects it.
        let well_tiles = tiles();
        assert_eq!(well_tiles.len(), 2, "A1 has two images (Auto: 1 x 2)");
        let (first, second) = (well_tiles[0].1, well_tiles[1].1);
        press(second);
        release(second);
        drain_ui_queue();
        assert_eq!(s.get_active_well(), "A1_02.tif");
        i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_millis(1000));

        // Drag the first image onto the second.
        press(first);
        for step in 1..=10 {
            let t = step as f32 / 10.0;
            window.dispatch_event(WindowEvent::PointerMoved {
                position: LogicalPosition::new(
                    first.x + (second.x - first.x) * t,
                    first.y + (second.y - first.y) * t,
                ),
            });
        }
        release(second);
        drain_ui_queue();
        assert_eq!(
            o.controller._app_state.get_project().plate.well_image_order,
            vec![2, 1],
            "dropped: swapped"
        );
        assert_eq!(well_grid_keys(&s), vec!["A1_02.tif", "A1_01.tif"]);
    }

    #[test]
    fn the_export_uses_the_shared_plate_settings() {
        let plate = PlateSettings {
            grouping_mode: GroupingMode::Folder,
            plate_size: PlateSize::Plate6x8,
            well_layout: WellLayout::Fixed { rows: 2, cols: 2 },
            well_image_order: vec![2, 1, 3, 4],
            ..PlateSettings::default()
        };
        let (o, _project_settings) = opened_with_plate(plate);
        o.state().invoke_export_dialog_open();
        drain_ui_queue();
        o.ui.global::<ExportDialogState>()
            .set_output_dir("/tmp".into());
        let export = o.controller.read_export_settings(&o.ui).unwrap();
        assert_eq!(export.grouping, Grouping::Folder);
        assert_eq!(export.plate_size, PlateSize::Plate6x8);
        assert_eq!(export.well_size, Some(WellSize { rows: 2, cols: 2 }));
        assert_eq!(export.well_order, Some(vec![2, 1, 3, 4]));
    }

    struct Opened {
        ui: AppWindow,
        controller: Arc<ResultsStateController>,
        _dir: tempfile::TempDir,
    }

    /// An opened database with the results-window file browser wired up,
    /// every queued UI update applied.
    fn opened() -> Opened {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.attach_callbacks();
        controller.open_database(path);
        drain_ui_queue();
        Opened {
            ui,
            controller,
            _dir: dir,
        }
    }

    impl Opened {
        fn state(&self) -> ResultsState<'_> {
            self.ui.global::<ResultsState>()
        }
        fn export(&self) -> ExportDialogState<'_> {
            self.ui.global::<ExportDialogState>()
        }
    }

    #[test]
    fn opening_a_database_fills_every_results_panel() {
        let o = opened();
        let s = o.state();
        assert_eq!(
            s.get_list_rows().row_count(),
            3,
            "one row per seeded object"
        );
        assert!(s.get_list_column_headers().row_count() > 0);
        assert_eq!(
            s.get_list_column_widths().row_count(),
            s.get_list_column_headers().row_count()
        );
        assert_eq!(s.get_list_class_items().row_count(), 2);
        assert_eq!(s.get_list_image_items().row_count(), 3);
        assert_eq!(s.get_list_image_summary(), "All Images");
        assert!(s.get_list_columns().row_count() > 0);
        assert!(s.get_list_aggregation_items().row_count() > 0);
        assert!(s.get_matrix_column_items().row_count() > 0);
        assert!(s.get_matrix_class_items().row_count() > 0);
        assert!(s.get_matrix_color_schema_items().row_count() > 0);
        assert!(s.get_matrix_plate_size_items().row_count() > 0);
        assert!(s.get_matrix_square_size_items().row_count() > 0);
        assert!(s.get_chart_column_items().row_count() > 0);
        assert_eq!(s.get_chart_error(), "");

        // The hidden tabs' data is only queried once they're shown.
        assert_eq!(s.get_plate_cells().row_count(), 0);
        assert_eq!(s.get_chart_histogram_bins().row_count(), 0);
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        drain_ui_queue();
        assert!(s.get_plate_cells().row_count() > 0);
        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        drain_ui_queue();
        assert!(s.get_chart_histogram_bins().row_count() > 0);
    }

    /// A filter changed while its tab is hidden takes effect when the tab
    /// is shown again - without querying while hidden.
    #[test]
    fn a_hidden_list_catches_up_on_filter_changes_when_shown() {
        let o = opened();
        let s = o.state();
        assert_eq!(o.controller.list_row_locations.lock().unwrap().len(), 3);

        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        s.invoke_list_image_selected("A1_01.tif".into(), true);
        drain_ui_queue();
        assert_eq!(
            o.controller.list_row_locations.lock().unwrap().len(),
            3,
            "not queried while hidden"
        );

        s.invoke_rail_mode_selected(ResultsRailMode::List);
        drain_ui_queue();
        assert_eq!(o.controller.list_row_locations.lock().unwrap().len(), 1);
    }

    /// Outside tests, database work runs on a background thread: opening a
    /// database and toggling an image must still fill/refresh the window,
    /// and `loading` must be cleared once it's done.
    #[test]
    fn opening_and_toggling_run_on_the_database_thread() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.db_worker.inline.store(false, Ordering::SeqCst);
        controller.attach_callbacks();
        let s = ui.global::<ResultsState>();

        controller.open_database(path);
        drain_ui_queue();
        assert!(controller.result_generator.lock().unwrap().is_some());
        assert_eq!(s.get_list_rows().row_count(), 3);
        assert!(!s.get_loading(), "the status bar is back to Ready");
        assert_eq!(s.get_database_name(), "results.evadb");

        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        s.invoke_plate_cell_clicked("A1".into());
        s.invoke_open_well_clicked("A1".into());
        // Switched by the Slint UI itself.
        s.set_matrix_level(MatrixLevel::Well);
        // The well's grid lands (and resets the active field) first.
        drain_ui_queue();
        s.invoke_well_cell_clicked("A1_01.tif".into());
        s.invoke_toggle_active_well_disabled();
        assert!(s.get_active_well_disabled(), "shown right away");
        drain_ui_queue();
        let images = controller
            .result_generator
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get_images()
            .unwrap();
        let image = images.iter().find(|i| i.name == "A1_01.tif").unwrap();
        assert!(image.disabled, "written by the database thread");
        assert!(controller.well_cells.lock().unwrap()["A1_01.tif"].disabled);
        assert!(!s.get_loading());
        drop(dir);
    }

    #[test]
    fn drilling_down_shows_the_well_and_image_heatmap() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        s.invoke_plate_cell_clicked("A1".into());
        s.invoke_open_well_clicked("A1".into());
        drain_ui_queue();
        // (`matrix_level` itself is switched by the Slint UI.)
        assert!(s.get_well_fields().row_count() >= 2);

        s.invoke_well_field_clicked("A1_01.tif".into());
        drain_ui_queue();
        assert!(s.get_image_heatmap_cells().row_count() > 0);
        assert!(s.get_breadcrumb().row_count() >= 3);

        s.invoke_toggle_active_well_disabled();
        drain_ui_queue();

        s.invoke_breadcrumb_nav(0);
        drain_ui_queue();
        assert!(o.controller.current_well.lock().unwrap().is_none());
        assert!(s.get_plate_cells().row_count() > 0);
    }

    /// `opened()`'s database as a time series: every object again at T1,
    /// twice as large.
    fn opened_time_series() -> Opened {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        {
            let conn = duckdb::Connection::open(&path).unwrap();
            conn.execute_batch(
                "INSERT INTO objects SELECT * REPLACE (1 AS t_stack, gen_random_uuid() AS object_id, \
                     area_px * 2 AS area_px) FROM objects;
                 UPDATE images SET t_stacks = 2;",
            )
            .unwrap();
        }
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, Default::default());
        let controller = make_controller_with_state(ui.as_weak(), ui_state);
        controller.attach_callbacks();
        controller.open_database(path);
        drain_ui_queue();
        Opened {
            ui,
            controller,
            _dir: dir,
        }
    }

    /// Over time: a line per class with its legend label, a band where
    /// there's an SD (ClassA's two objects; not ClassB's one), the time
    /// points on the x axis; a line per image when asked.
    #[test]
    fn the_over_time_chart_draws_a_line_per_class_or_image() {
        let o = opened_time_series();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        s.set_chart_kind(ResultsChartKind2::OverTime);
        s.invoke_chart_kind_selected(ResultsChartKind2::OverTime);
        s.invoke_chart_column_selected("area_px".into());
        drain_ui_queue();
        assert_eq!(s.get_chart_error(), "");
        let lines: Vec<ChartTimeLine> = s.get_chart_time_data().iter().collect();
        let labels: Vec<String> = lines.iter().map(|l| l.label.to_string()).collect();
        assert_eq!(labels, ["ClassA", "ClassB"]);
        assert!(
            lines[0].line.starts_with("M 0.0 ") && lines[0].line.contains(" L 1000.0 "),
            "{}",
            lines[0].line
        );
        assert!(
            lines[0].band.starts_with('M') && lines[0].band.ends_with('Z'),
            "{}",
            lines[0].band
        );
        assert_eq!(lines[1].band, "", "one object: no SD");
        assert_eq!(s.get_chart_time_x_min(), "T 0");
        assert_eq!(s.get_chart_time_x_max(), "T 1");
        assert!(s.get_chart_time_y_title().contains("mean ± SD"));

        s.set_chart_time_lines(1);
        s.invoke_chart_time_changed();
        drain_ui_queue();
        // Only the image × class lines with objects, named with their class.
        let images: Vec<String> = s
            .get_chart_time_data()
            .iter()
            .map(|l| l.label.to_string())
            .collect();
        assert_eq!(
            images,
            [
                "A1_01.tif · ClassA",
                "A1_02.tif · ClassA",
                "A2_01.tif · ClassB"
            ]
        );

        s.invoke_chart_class_selected("ClassA".into(), true);
        drain_ui_queue();
        let images: Vec<String> = s
            .get_chart_time_data()
            .iter()
            .map(|l| l.label.to_string())
            .collect();
        assert_eq!(images, ["A1_01.tif", "A1_02.tif"], "the class picked");
    }

    /// One time point: nothing to draw over time - it says so.
    #[test]
    fn the_over_time_chart_needs_several_time_points() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        s.set_chart_kind(ResultsChartKind2::OverTime);
        s.invoke_chart_kind_selected(ResultsChartKind2::OverTime);
        drain_ui_queue();
        assert!(
            s.get_chart_error().contains("several time points"),
            "{}",
            s.get_chart_error()
        );
        assert_eq!(s.get_chart_time_data().row_count(), 0);
    }

    /// "Export data…": the export dialog with only the time course - the
    /// chart's value, lines, summary and normalization.
    #[test]
    fn exporting_the_over_time_chart_presets_the_dialog() {
        let o = opened_time_series();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        s.invoke_chart_kind_selected(ResultsChartKind2::OverTime);
        s.invoke_chart_column_selected("area_px".into());
        s.set_chart_time_lines(2);
        s.set_chart_time_summary(1);
        s.set_chart_time_relative(true);
        s.invoke_chart_time_changed();
        drain_ui_queue();

        s.invoke_chart_time_export();
        let export = o.export();
        assert!(export.get_active());
        assert!(export.get_with_time_course());
        assert!(!export.get_with_list_view() && !export.get_with_grouped_by_image_list());
        let selected: Vec<String> = export
            .get_column_items()
            .iter()
            .filter(|item| item.selected)
            .map(|item| item.key.to_string())
            .collect();
        assert_eq!(selected, ["area_px"]);

        export.set_output_dir("/tmp".into());
        let settings = o.controller.read_export_settings(&o.ui).expect("settings");
        assert!(settings.with_time_course);
        assert_eq!(settings.columns, [Column::AreaSizePx]);
        assert_eq!(settings.time_course.lines, TimeCourseLines::Well);
        assert_eq!(
            settings.time_course.statistics,
            [
                TimeCourseStat::Median,
                TimeCourseStat::Q1,
                TimeCourseStat::Q3
            ]
        );
        assert_eq!(
            settings.time_course.normalize,
            TimeCourseNormalize::RelativeToStart
        );
        assert_eq!(settings.time_course.layout, TimeCourseLayout::Both);
    }

    #[test]
    fn every_chart_kind_draws_into_the_window() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Charts);
        s.invoke_chart_kind_selected(ResultsChartKind2::Scatter);
        s.invoke_chart_y_column_selected("area_px".into());
        drain_ui_queue();
        assert!(s.get_chart_scatter_points().row_count() > 0);
        assert!(s.get_chart_scatter_total_count() > 0);

        s.invoke_chart_kind_selected(ResultsChartKind2::Boxplot);
        drain_ui_queue();
        assert!(s.get_chart_boxplot_boxes().row_count() > 0);

        s.invoke_chart_kind_selected(ResultsChartKind2::Histogram);
        s.invoke_chart_class_selected("ClassA".into(), true);
        drain_ui_queue();
        assert!(s.get_chart_histogram_object_count() > 0);
    }

    #[test]
    fn matrix_and_list_controls_refresh_the_window() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        s.invoke_matrix_aggregate_selected("Maximum".into());
        s.invoke_matrix_color_schema_selected("Viridis".into(), true);
        s.invoke_matrix_scale_set_manual(0.0, 10.0);
        s.invoke_matrix_scale_set_auto();
        drain_ui_queue();
        assert!(s.get_plate_cells().row_count() > 0);

        s.invoke_rail_mode_selected(ResultsRailMode::List);
        s.invoke_list_group_by_selected("images".into());
        s.invoke_list_aggregation_item_selected("min".into(), true);
        drain_ui_queue();
        assert!(s.get_list_rows().row_count() > 0);
        s.invoke_list_transpond_changed(true);
        drain_ui_queue();
        assert!(s.get_list_rows().row_count() > 0);
        s.invoke_list_group_by_selected("objects".into());
        s.invoke_list_class_select_none();
        drain_ui_queue();
        assert_eq!(s.get_list_rows().row_count(), 0, "no class selected");
        s.invoke_list_class_select_all();
        s.invoke_list_image_selected("A1_01.tif".into(), true);
        drain_ui_queue();
        assert!(s.get_list_image_summary().starts_with("1 of"));
    }

    #[test]
    fn grouped_by_image_keeps_a_selected_intensity_column() {
        let o = opened();
        let s = o.state();
        s.invoke_rail_mode_selected(ResultsRailMode::List);
        s.invoke_list_columns_item_selected("intensity_mean_gray_ch0".into(), true);
        drain_ui_queue();
        let has_intensity_header = |s: &ResultsState| {
            s.get_list_column_headers()
                .iter()
                .any(|h| h.contains("Avg Intensity (Ch 0)"))
        };
        assert!(has_intensity_header(&s), "object view shows the column");

        s.invoke_list_group_by_selected("images".into());
        drain_ui_queue();
        assert!(
            has_intensity_header(&s),
            "grouped by image keeps the column"
        );
    }

    fn start_export(o: &Opened, folder: &std::path::Path) {
        let export = o.export();
        o.state().invoke_export_dialog_open();
        drain_ui_queue();
        export.invoke_pick_output_dir();
        choose_file(&o.ui, folder);
        assert_eq!(export.get_output_dir(), folder.to_string_lossy().as_ref());
        export.invoke_start_clicked();
        drain_ui_queue();
    }

    #[test]
    fn export_writes_the_chosen_views_into_the_picked_folder() {
        let o = opened();
        let dir = tempfile::tempdir().unwrap();
        let export = o.export();
        o.state().invoke_export_dialog_open();
        export.set_with_list_view(true);
        export.set_with_grouped_by_image_list(true);
        export.set_output_file_prefix("run1".into());
        start_export(&o, dir.path());
        assert!(!export.get_has_error(), "{}", export.get_error_message());
        assert!(export.get_done());
        assert!(!export.get_is_exporting());
        let written: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(!written.is_empty());
        assert!(
            written.iter().all(|f| f.starts_with("run1_")),
            "{written:?}"
        );
    }

    /// Every z-plane and time point unless a range is chosen - the hidden
    /// range fields of "All" don't matter, even when they don't parse.
    #[test]
    fn export_covers_all_planes_unless_a_range_is_chosen() {
        let o = opened();
        let export = o.export();
        o.state().invoke_export_dialog_open();
        assert!(export.get_z_all() && export.get_t_all(), "All by default");
        export.set_output_dir("/tmp".into());
        export.set_t_start("x".into());
        let settings = o.controller.read_export_settings(&o.ui).expect("settings");
        assert!(settings.z_stacks.is_none() && settings.t_stacks.is_none());

        export.set_t_all(false);
        export.set_t_start("1".into());
        export.set_t_end("3".into());
        let settings = o.controller.read_export_settings(&o.ui).expect("settings");
        assert!(settings.z_stacks.is_none());
        let t = settings.t_stacks.expect("a range");
        assert_eq!((t.start, t.end), (1, 3));
    }

    #[test]
    fn export_reports_invalid_settings_instead_of_starting() {
        let o = opened();
        let export = o.export();
        o.state().invoke_export_dialog_open();
        export.set_output_dir("/tmp".into());
        let fields = ["Z start", "Z end", "T start", "T end"];
        for (broken, label) in fields.iter().enumerate() {
            o.state().invoke_export_dialog_open();
            let value = |i: usize| -> slint::SharedString {
                if i == broken { "x".into() } else { "1".into() }
            };
            export.set_z_all(false);
            export.set_t_all(false);
            export.set_z_start(value(0));
            export.set_z_end(value(1));
            export.set_t_start(value(2));
            export.set_t_end(value(3));
            export.invoke_start_clicked();
            assert!(export.get_has_error());
            assert!(
                export.get_error_message().contains(label),
                "{}",
                export.get_error_message()
            );
        }
    }

    #[test]
    fn export_without_an_open_database_reports_it() {
        let ui = test_ui_window();
        let controller = make_controller_with_ui(ui.as_weak());
        controller.run_export(ResultExport::default());
        drain_ui_queue();
        let export = ui.global::<ExportDialogState>();
        assert!(export.get_has_error());
        assert_eq!(export.get_error_message(), "No database is open.");
    }

    #[test]
    fn export_progress_cancel_and_failure_are_shown() {
        let o = opened();
        o.controller.push_export_progress("Working".into(), 2, 5);
        drain_ui_queue();
        let export = o.export();
        assert_eq!(export.get_progress_message(), "Working");
        assert_eq!(export.get_progress_current(), 2);
        assert_eq!(export.get_progress_total(), 5);

        export.set_is_exporting(true);
        o.controller.push_export_cancelled();
        drain_ui_queue();
        assert!(!export.get_is_exporting());
        assert_eq!(export.get_progress_message(), "Export cancelled.");

        // An export into a folder that can't be created fails visibly.
        let file = o._dir.path().join("a-file");
        std::fs::write(&file, "").unwrap();
        o.controller.run_export(ResultExport {
            output_dir: file.join("sub"),
            with_list_view: true,
            ..Default::default()
        });
        drain_ui_queue();
        assert!(export.get_has_error());
    }

    // -- unknown, deselected and out-of-range input -------------------------

    #[test]
    fn list_filters_ignore_unknown_entries_and_toggle_known_ones() {
        let o = opened();
        let state = o.state();
        let classes = |o: &Opened| {
            o.controller
                .list_filter
                .lock()
                .unwrap()
                .object_classes
                .clone()
        };

        o.controller
            .list_filter
            .lock()
            .unwrap()
            .object_classes
            .clear();
        state.invoke_list_class_selected("No such class".into(), true);
        assert!(classes(&o).is_empty());
        state.invoke_list_class_selected("ClassA".into(), true);
        state.invoke_list_class_selected("ClassA".into(), true);
        assert_eq!(classes(&o), vec![ObjectClass::Global(1)]);
        state.invoke_list_class_selected("ClassA".into(), false);
        assert!(classes(&o).is_empty());

        let columns = |o: &Opened| o.controller.list_filter.lock().unwrap().columns.clone();
        let before = columns(&o);
        state.invoke_list_columns_item_selected("no_such_column".into(), true);
        assert_eq!(columns(&o), before);
        state.invoke_list_columns_select_none();
        state.invoke_list_columns_item_selected("area_px".into(), true);
        state.invoke_list_columns_item_selected("area_px".into(), true);
        assert_eq!(columns(&o), vec![Column::AreaSizePx]);
        state.invoke_list_columns_item_selected("area_px".into(), false);
        assert!(columns(&o).is_empty());

        let aggregations = |o: &Opened| {
            o.controller
                .list_filter
                .lock()
                .unwrap()
                .aggregations
                .clone()
        };
        state.invoke_list_aggregation_select_none();
        state.invoke_list_aggregation_item_selected("no such aggregation".into(), true);
        assert!(aggregations(&o).is_empty());
        let key = aggregation_display_name(&AGGREGATIONS[0]);
        state.invoke_list_aggregation_item_selected(key.into(), true);
        state.invoke_list_aggregation_item_selected(key.into(), true);
        assert_eq!(aggregations(&o), vec![AGGREGATIONS[0].clone()]);
        state.invoke_list_aggregation_item_selected(key.into(), false);
        assert!(aggregations(&o).is_empty());

        // A row that isn't on the page.
        state.invoke_list_row_clicked(9999);
        drain_ui_queue();
    }

    fn matrix_filter_value<T>(o: &Opened, read: impl Fn(&MatrixFilter) -> T) -> T {
        read(o.controller.matrix_filter.lock().unwrap().as_ref().unwrap())
    }

    #[test]
    fn matrix_toolbar_picks_every_aggregate_and_ignores_unknown_entries() {
        let o = opened();
        let state = o.state();
        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        state.invoke_matrix_aggregate_selected("Sum".into());

        for (name, aggregation) in [
            ("Average", Aggregation::Avg),
            ("Maximum", Aggregation::Max),
            ("Std Dev", Aggregation::Stddev),
            ("Stddev", Aggregation::Stddev),
            ("Sum", Aggregation::Sum),
            ("Median", Aggregation::Median),
            ("Skewness", Aggregation::Skewness),
            ("Minimum", Aggregation::Min),
        ] {
            state.invoke_matrix_aggregate_selected(name.into());
            assert_eq!(
                matrix_filter_value(&o, |f| f.aggregation.clone()),
                aggregation
            );
        }
        state.invoke_matrix_aggregate_selected("Mode".into());
        assert_eq!(
            matrix_filter_value(&o, |f| f.aggregation.clone()),
            Aggregation::Min
        );

        let class = matrix_filter_value(&o, |f| f.object_classe.clone());
        state.invoke_matrix_class_selected("ClassB".into(), false);
        state.invoke_matrix_class_selected("No such class".into(), true);
        assert_eq!(matrix_filter_value(&o, |f| f.object_classe.clone()), class);

        let mode = matrix_filter_value(&o, |f| f.plate.grouping_mode);
        state.invoke_matrix_grouping_selected("Folder".into(), false);
        state.invoke_matrix_grouping_selected("By color".into(), true);
        assert_eq!(matrix_filter_value(&o, |f| f.plate.grouping_mode), mode);

        state.invoke_matrix_color_schema_selected("Viridis".into(), false);
        state.invoke_matrix_color_schema_selected("No such schema".into(), true);

        let plate = matrix_filter_value(&o, |f| f.plate.clone());
        state.invoke_matrix_plate_size_selected("96 Well (8 x 12)".into(), false);
        state.invoke_matrix_plate_size_selected("7 Well".into(), true);
        state.invoke_matrix_well_layout_selected("Custom".into(), false);
        state.invoke_matrix_well_rows_changed("many".into());
        state.invoke_matrix_well_cols_changed("-1".into());
        assert_eq!(matrix_filter_value(&o, |f| f.plate.clone()), plate);

        let square = matrix_filter_value(&o, |f| f.square_size);
        state.invoke_matrix_square_size_selected("128".into(), false);
        state.invoke_matrix_square_size_selected("big".into(), true);
        assert_eq!(matrix_filter_value(&o, |f| f.square_size), square);

        // A negative grid position (nothing dragged) is ignored.
        state.invoke_well_image_moved(-1, 0);
        assert_eq!(matrix_filter_value(&o, |f| f.plate.clone()), plate);
        drain_ui_queue();
    }

    #[test]
    fn well_rows_and_cols_typed_with_an_auto_layout_start_from_four() {
        let o = opened();
        let state = o.state();
        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        state.invoke_matrix_aggregate_selected("Sum".into());
        state.invoke_matrix_well_layout_selected(WELL_LAYOUT_AUTO.into(), true);
        assert_eq!(
            matrix_filter_value(&o, |f| f.plate.well_layout),
            WellLayout::Auto
        );

        state.invoke_matrix_well_rows_changed("3".into());
        assert_eq!(
            matrix_filter_value(&o, |f| f.plate.well_layout),
            WellLayout::Fixed { rows: 3, cols: 4 }
        );

        state.invoke_matrix_well_layout_selected(WELL_LAYOUT_AUTO.into(), true);
        state.invoke_matrix_well_cols_changed("2".into());
        assert_eq!(
            matrix_filter_value(&o, |f| f.plate.well_layout),
            WellLayout::Fixed { rows: 4, cols: 2 }
        );
        drain_ui_queue();
    }

    #[test]
    fn matrix_column_picker_without_a_known_selection_keeps_the_column() {
        let o = opened();
        let state = o.state();
        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        state.invoke_matrix_aggregate_selected("Sum".into());
        let column = matrix_filter_value(&o, |f| f.column.clone());

        let item = |key: &str, selected: bool| MultiSelectItem {
            key: key.into(),
            selected,
            ..Default::default()
        };
        state.set_matrix_column_items(ModelRc::new(VecModel::from(vec![item("area_px", false)])));
        state.invoke_matrix_value_clicked();
        state.set_matrix_column_items(ModelRc::new(VecModel::from(vec![item(
            "no_such_column",
            true,
        )])));
        state.invoke_matrix_value_clicked();

        assert_eq!(matrix_filter_value(&o, |f| f.column.clone()), column);
    }

    #[test]
    fn chart_pickers_ignore_unknown_and_deselected_entries() {
        let o = opened();
        let state = o.state();
        state.invoke_chart_column_selected("area_px".into());
        state.invoke_chart_y_column_selected("area_px".into());
        state.invoke_chart_class_selected("ClassA".into(), true);

        state.invoke_chart_column_selected("no_such_column".into());
        state.invoke_chart_y_column_selected("no_such_column".into());
        state.invoke_chart_class_selected("ClassB".into(), false);
        state.invoke_chart_class_selected("No such class".into(), true);

        let chart = o.controller.chart_filter.lock().unwrap().clone();
        assert_eq!(chart.column, Column::AreaSizePx);
        assert_eq!(chart.y_column, Column::AreaSizePx);
        assert_eq!(chart.object_class, Some(ObjectClass::Global(1)));
    }

    #[test]
    fn breadcrumb_back_to_the_well_shows_the_well_grid_again() {
        let o = opened();
        let state = o.state();
        state.invoke_rail_mode_selected(ResultsRailMode::Matrix);
        state.invoke_matrix_aggregate_selected("Sum".into());
        state.invoke_plate_cell_clicked("A1".into());
        state.invoke_open_well_clicked("A1".into());
        state.invoke_well_field_clicked("A1_01.tif".into());
        drain_ui_queue();

        // ["All results", "Plate", "Well A1", "Image"]: index 2 is the well.
        state.invoke_breadcrumb_nav(2);
        drain_ui_queue();

        assert_eq!(state.get_matrix_level(), MatrixLevel::Well);
        assert_eq!(o.controller.current_image.lock().unwrap().as_deref(), None);
        assert_eq!(
            o.controller.current_well.lock().unwrap().as_deref(),
            Some("A1")
        );
        assert_eq!(state.get_breadcrumb().row_count(), 3);
    }

    #[test]
    fn heatmap_tiles_without_an_open_image_or_with_a_bad_key_are_only_selected() {
        let o = opened();
        let state = o.state();
        o.controller
            .image_heatmap_cells
            .lock()
            .unwrap()
            .insert("not-a-tile".into(), MatrixCell::default());

        state.invoke_image_heatmap_cell_clicked("not-a-tile".into());
        assert_eq!(state.get_active_well(), "not-a-tile");

        *o.controller.current_image.lock().unwrap() = Some("A1_01.tif".into());
        state.invoke_image_heatmap_cell_clicked("not-a-tile".into());
        drain_ui_queue();
        assert_eq!(state.get_active_well(), "not-a-tile");
    }

    #[test]
    fn toggling_an_unknown_image_or_without_a_database_keeps_it_enabled() {
        let o = opened();
        let state = o.state();
        state.set_active_well("no-such-image.tif".into());
        state.invoke_toggle_active_well_disabled();
        assert!(!state.get_active_well_disabled());

        // The database went away: the toggle shown right away is undone.
        state.set_active_well("A1_01.tif".into());
        *o.controller.result_generator.lock().unwrap() = None;
        state.invoke_toggle_active_well_disabled();
        drain_ui_queue();
        assert!(!state.get_active_well_disabled());
    }

    #[test]
    fn cancelling_the_export_folder_picker_keeps_the_folder() {
        let o = opened();
        let export = o.export();
        o.state().invoke_export_dialog_open();
        export.set_output_dir("/kept".into());

        export.invoke_pick_output_dir();
        drain_ui_queue();
        o.ui.global::<crate::FileBrowserState>().invoke_cancel();
        drain_ui_queue();

        assert_eq!(export.get_output_dir(), "/kept");
    }

    #[test]
    fn export_reports_unknown_selections_instead_of_starting() {
        let o = opened();
        let export = o.export();
        let unknown = || {
            ModelRc::new(VecModel::from(vec![MultiSelectItem {
                key: "no such entry".into(),
                selected: true,
                ..Default::default()
            }]))
        };
        let cases: [(&str, &dyn Fn(&ExportDialogState)); 5] = [
            ("Unknown class", &|e| e.set_class_items(unknown())),
            ("Unknown column", &|e| e.set_column_items(unknown())),
            ("Unknown aggregation", &|e| {
                e.set_aggregation_items(unknown())
            }),
            ("Unknown color schema", &|e| {
                e.set_color_schema_items(unknown())
            }),
            ("Unknown export format", &|e| e.set_format_items(unknown())),
        ];
        o.state().invoke_export_dialog_open();
        drain_ui_queue();
        let originals = (
            export.get_class_items(),
            export.get_column_items(),
            export.get_aggregation_items(),
            export.get_color_schema_items(),
            export.get_format_items(),
        );
        for (message, break_it) in cases {
            export.set_class_items(originals.0.clone());
            export.set_column_items(originals.1.clone());
            export.set_aggregation_items(originals.2.clone());
            export.set_color_schema_items(originals.3.clone());
            export.set_format_items(originals.4.clone());
            export.set_output_dir("/tmp".into());
            break_it(&export);
            export.invoke_start_clicked();
            assert!(export.get_has_error(), "{message}");
            assert!(
                export.get_error_message().starts_with(message),
                "{}",
                export.get_error_message()
            );
            export.set_has_error(false);
        }
    }

    // -- the window closed / nothing to show ----------------------------------

    fn closed_window_controller() -> Arc<ResultsStateController> {
        crate::helper::ui_thread::fresh_test_queue();
        make_controller_with_ui(slint::Weak::default())
    }

    #[test]
    fn views_without_a_database_have_nothing_to_show() {
        let controller = closed_window_controller();
        *controller.rail_mode.lock().unwrap() = ResultsRailMode::Matrix;
        controller.update_matrix_view();
        controller.update_well_view("A1");
        controller.update_image_heatmap_view("A1_01.tif");
        *controller.current_image.lock().unwrap() = Some("A1_01.tif".into());
        controller.refresh_matrix_view_at(MatrixLevel::Object);
        *controller.current_well.lock().unwrap() = Some("A1".into());
        controller.refresh_matrix_view_at(MatrixLevel::Well);
        controller.refresh_active_matrix_view();

        *controller.rail_mode.lock().unwrap() = ResultsRailMode::List;
        controller.refresh_list();
        *controller.rail_mode.lock().unwrap() = ResultsRailMode::Charts;
        controller.refresh_charts();
        drain_ui_queue();

        assert!(controller.matrix_cells.lock().unwrap().is_empty());
        assert!(controller.well_cells.lock().unwrap().is_empty());
        assert!(controller.list_row_locations.lock().unwrap().is_empty());
    }

    #[test]
    fn results_for_a_closed_window_are_dropped_without_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("results.evadb");
        seed_test_db(&path);
        let controller = closed_window_controller();

        controller.open_database(path);
        drain_ui_queue();
        assert_eq!(controller.classes.lock().unwrap().len(), 2);
        assert!(!controller.images.lock().unwrap().is_empty());

        // No Matrix filter chosen: nothing to aggregate.
        *controller.matrix_filter.lock().unwrap() = None;
        controller.matrix_cells.lock().unwrap().clear();
        controller.well_cells.lock().unwrap().clear();
        *controller.rail_mode.lock().unwrap() = ResultsRailMode::Matrix;
        controller.update_matrix_view();
        controller.update_well_view("A1");
        controller.update_image_heatmap_view("A1_01.tif");
        assert!(controller.matrix_cells.lock().unwrap().is_empty());

        *controller.matrix_filter.lock().unwrap() = Some(MatrixFilter {
            aggregation: Aggregation::Sum,
            ..Default::default()
        });
        controller.update_matrix_view();
        controller.update_well_view("A1");
        controller.update_image_heatmap_view("A1_01.tif");
        controller.show_results_window();

        *controller.rail_mode.lock().unwrap() = ResultsRailMode::List;
        controller.refresh_list();
        controller.select_all_images();
        controller.select_none_images();
        controller.filter_images("A1");
        controller.select_all_classes();
        controller.select_none_classes();
        controller.select_all_columns();
        controller.select_none_columns();
        controller.select_all_aggregations();
        controller.select_none_aggregations();

        *controller.rail_mode.lock().unwrap() = ResultsRailMode::Charts;
        for kind in [ChartKind::Histogram, ChartKind::Scatter, ChartKind::Boxplot] {
            controller.chart_filter.lock().unwrap().kind = kind;
            controller.refresh_charts();
        }
        controller.push_chart_error("broken".into());
        controller.set_max_z_and_t_stack_in_slint(2, 3);
        controller.push_export_progress("Working".into(), 1, 2);
        controller.push_export_done();
        controller.push_export_error("failed".into());
        controller.push_export_cancelled();
        drain_ui_queue();

        assert!(controller.matrix_cells.lock().unwrap().contains_key("A1"));
        assert!(!controller.well_cells.lock().unwrap().is_empty());
    }

    /// Split with the image: the analysis' images registered in the
    /// project (else they can't be opened), the split on.
    fn split_controller() -> (AppWindow, Arc<ResultsStateController>) {
        use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
        let (ui, controller) = controller_with_open_database();
        {
            let mut project = controller._app_state.get_project_write();
            project.images.root = Some(PathBuf::from("/data/images"));
            for name in ["A1_01.tif", "A1_02.tif", "A2_01.tif"] {
                project.images.list.insert(
                    PathBuf::from(name),
                    ImageEntry {
                        rel_path: PathBuf::from(name),
                        file_size: 0,
                        selected_series: 0,
                        series: std::collections::BTreeMap::from([(0, SeriesSettings::default())]),
                    },
                );
            }
        }
        ui.global::<ResultsState>().set_split(true);
        (ui, controller)
    }

    #[test]
    fn split_a_row_shows_its_object_on_its_image_and_esc_lets_go() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        let (rel_path, bbox) = controller.list_row_locations.lock().unwrap()[1].clone();

        state.set_list_selected_row(1);
        state.invoke_list_row_selected(1);
        drain_ui_queue();

        let overlay = controller
            .viewport()
            .results_overlay()
            .expect("the analysis' objects on the image");
        assert_eq!(overlay.objects.len(), 1, "the one object of {rel_path}");
        assert_eq!(overlay.objects[0].0, bbox);
        assert_eq!(overlay.selected, Some(0));
        let sel = state.get_split_sel();
        assert!(sel.active);
        assert_eq!((sel.x, sel.y, sel.w, sel.h), (0.0, 0.0, 11.0, 11.0));
        assert!(sel.class_name.starts_with("Class"));
        assert_eq!(
            (state.get_split_pos(), state.get_split_total()),
            (2, 3),
            "row 2 of 3"
        );
        assert_eq!(state.get_split_title(), rel_path.as_str());

        let highlight = ui
            .global::<crate::ViewportObjectState>()
            .get_object_highlight();
        assert!(
            highlight.active && !highlight.paint_as_rectangle,
            "the yellow crosshair"
        );

        // Esc: nothing selected, the objects stay.
        state.invoke_split_clear();
        drain_ui_queue();
        assert!(!state.get_split_sel().active);
        assert!(
            !ui.global::<crate::ViewportObjectState>()
                .get_object_highlight()
                .active
        );
        assert_eq!(state.get_list_selected_row(), -1);
        assert_eq!(
            controller.viewport().results_overlay().unwrap().selected,
            None
        );

        // Leaving the split: the project's objects again.
        state.invoke_split_shown(false);
        assert!(controller.viewport().results_overlay().is_none());
    }

    #[test]
    fn split_a_click_on_an_object_selects_its_row_and_arrows_step_rows() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        state.invoke_list_row_selected(0);
        drain_ui_queue();
        state.invoke_split_clear();
        drain_ui_queue();

        // The object (0..10 px) under the click; zoom 1, no offset.
        let (zoom, off_x, off_y, ..) = controller.viewport().view();
        state.invoke_split_image_clicked(off_x + 5.0 * zoom, off_y + 5.0 * zoom);
        drain_ui_queue();
        assert_eq!(state.get_list_selected_row(), 0, "its row");
        assert!(state.get_split_sel().active);

        // Beside every object: nothing changes.
        state.invoke_split_image_clicked(off_x + 80.0 * zoom, off_y + 80.0 * zoom);
        drain_ui_queue();
        assert_eq!(state.get_list_selected_row(), 0);

        // ↓: the next row, on its own image.
        state.invoke_split_step(1);
        drain_ui_queue();
        assert_eq!(state.get_list_selected_row(), 1);
        let next = controller.list_row_locations.lock().unwrap()[1].0.clone();
        assert_eq!(
            controller.split.lock().unwrap().image.as_deref(),
            Some(next.as_str())
        );
        // At the end it stays.
        state.invoke_split_step(1);
        state.invoke_split_step(1);
        drain_ui_queue();
        assert_eq!(state.get_list_selected_row(), 2);
    }

    #[test]
    fn split_a_heatmap_tile_is_shown_on_the_split_image_and_the_split_stays() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        *controller.current_image.lock().unwrap() = Some("A1_01.tif".to_string());

        // A double click (or "Show on image"): no leaving for Images.
        state.invoke_image_heatmap_cell_open("R0C0".into());
        drain_ui_queue();
        assert_ne!(
            ui.global::<crate::LayoutState>().get_mode(),
            crate::WorkspaceMode::Images
        );
        assert!(!state.get_peek());
        assert_eq!(
            controller.split.lock().unwrap().image.as_deref(),
            Some("A1_01.tif")
        );
        let highlight = ui
            .global::<crate::ViewportObjectState>()
            .get_object_highlight();
        assert!(
            highlight.active && highlight.paint_as_rectangle,
            "the tile's box"
        );
        assert_eq!((highlight.x_px, highlight.y_px), (0.0, 0.0));
        assert!(!state.get_split_sel().active, "no object selected");

        // Selecting an object instead: the box becomes its crosshair, as a
        // list row opened in the full view.
        state.invoke_split_image_clicked(5.0, 5.0);
        drain_ui_queue();
        assert!(state.get_split_sel().active);
        let highlight = ui
            .global::<crate::ViewportObjectState>()
            .get_object_highlight();
        assert!(
            highlight.active && !highlight.paint_as_rectangle,
            "the crosshair"
        );
        assert_eq!((highlight.x_px, highlight.w_px), (0.0, 11.0));
    }

    #[test]
    fn split_settings_are_remembered_as_user_settings() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        state.invoke_split_changed(true);
        state.invoke_split_width_changed(512.0);
        state.invoke_split_follow_changed(false);
        let saved = controller._app_state.load_app_settings().results_split;
        assert!(saved.on);
        assert_eq!(saved.width_px, 512.0);
        assert!(!saved.follow);
        assert!(!saved.image_panel, "closed by default");
        state.invoke_split_image_panel_changed(true);
        assert!(
            controller
                ._app_state
                .load_app_settings()
                .results_split
                .image_panel
        );
    }

    /// Leaving the split view takes its marker off the image; "Open in
    /// Images" keeps it, with the bar back to the results.
    #[test]
    fn leaving_the_split_view_takes_its_marker_off_the_image() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        let marker = || {
            ui.global::<crate::ViewportObjectState>()
                .get_object_highlight()
                .active
        };
        state.invoke_split_shown(true);
        state.invoke_list_row_selected(1);
        drain_ui_queue();
        assert!(marker(), "the row's object marked");

        state.invoke_split_shown(false);
        assert!(!marker(), "gone with the split view");

        state.invoke_split_shown(true);
        state.invoke_list_row_selected(1);
        drain_ui_queue();
        state.set_list_selected_row(1);
        state.invoke_split_open_in_images();
        state.invoke_split_shown(false);
        assert!(marker(), "Open in Images keeps it");
        assert!(state.get_peek(), "with the bar back to the results");
        assert_eq!(state.get_peek_label(), "Row 2 of 3");
        let _ = controller;
    }

    /// A row's image opens slower than its objects are read: they wait for
    /// it (they were dropped, so a row's object never showed) and go on it
    /// once it is open.
    #[test]
    fn a_rows_objects_wait_for_its_image_to_open() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        state.invoke_split_shown(true);
        state.invoke_list_row_selected(0);
        drain_ui_queue();
        assert!(controller.viewport().results_overlay().is_some());
        let opened = controller
            ._app_state
            .get_project()
            .tmp_settings
            .current_image
            .clone();

        // Another image still in the viewer when the objects arrive.
        controller
            ._app_state
            .get_project_write()
            .tmp_settings
            .current_image = Some(PathBuf::from("/data/images/still-the-old-one.tif"));
        controller.split_push_overlay();
        assert!(
            controller.viewport().results_overlay().is_none(),
            "not drawn on the wrong image"
        );
        assert!(
            !controller.split.lock().unwrap().objects.is_empty(),
            "kept, not dropped"
        );

        // Now it is open: the objects and the selection go on it.
        controller
            ._app_state
            .get_project_write()
            .tmp_settings
            .current_image = opened;
        controller._app_state.image_opened();
        drain_ui_queue();
        let overlay = controller
            .viewport()
            .results_overlay()
            .expect("drawn once the image is open");
        assert_eq!(overlay.selected, Some(0));
        assert!(state.get_split_sel().active);

        // Split view away: an image opened elsewhere gets no results overlay.
        state.invoke_split_shown(false);
        controller._app_state.image_opened();
        assert!(controller.viewport().results_overlay().is_none());
    }

    /// Plate "Color by value" also when the overlay is pushed from the
    /// database thread (where the window can't be reached), as it is right
    /// after an image's objects are read.
    #[test]
    fn split_color_by_value_holds_when_pushed_from_the_database_thread() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        state.invoke_list_row_selected(0);
        drain_ui_queue();
        *controller.rail_mode.lock().unwrap() = ResultsRailMode::Matrix;
        {
            let mut split = controller.split.lock().unwrap();
            let mut objects = (*split.objects).clone();
            assert!(!objects.is_empty(), "row 0's objects");
            for object in &mut objects {
                object.value = Some(1.0);
            }
            split.objects = Arc::new(objects);
        }
        state.invoke_split_by_value_changed(false);
        let by_class = controller.viewport().results_overlay().unwrap().objects[0].1;
        state.invoke_split_by_value_changed(true);
        let by_value = controller.viewport().results_overlay().unwrap().objects[0].1;
        assert_ne!(by_value, by_class, "the heatmap's color, not the class'");

        std::thread::scope(|s| {
            s.spawn(|| controller.split_push_overlay());
        });
        assert_eq!(
            controller.viewport().results_overlay().unwrap().objects[0].1,
            by_value,
            "still by value"
        );
    }

    /// The card's mean is of the channel picked in the image controls.
    #[test]
    fn split_the_cards_mean_follows_the_picked_channel() {
        let (ui, controller) = split_controller();
        let state = ui.global::<ResultsState>();
        state.invoke_list_row_selected(0);
        drain_ui_queue();
        assert_eq!(state.get_split_sel().mean, "127", "channel 0's mean");
        state.invoke_split_channel_changed(5);
        drain_ui_queue();
        assert_eq!(state.get_split_sel().mean, "–", "no such channel: nothing");
        let _ = controller;
    }
}
