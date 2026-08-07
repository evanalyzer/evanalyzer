use crate::editor::project_settings_controller::{ProjectSettingsController, index_to_well_size};
use crate::editor::results_table_controller::{ResultsTableController, model_to_vec};
use crate::{
    FilterItem, ResultsGroupBy, ResultsMatrixCell, ResultsMatrixKind, ResultsState,
    ResultsViewMode, ResultsWindow, UiState,
};
use evanalyzer_app::result::{
    AggFunc, ColumnSpec, DatabaseFilter, GroupBy, GroupConfig, HeatmapColorScheme,
    PlateMatrixResult, ResultsLoader, compute_plate_matrix, compute_well_matrix, plottable_columns,
    resolve_range, suggest_regex,
};
use evanalyzer_cfg::settings::plate_settings::GroupingMode;
use log::warn;
use slint::{ComponentHandle, SharedString};
use std::sync::Arc;

/// Synthetic "Value:" option meaning "color by how many objects landed in
/// this well/image", not a real per-object column - unlike every other
/// entry (`plottable_columns`), it has no `ColumnSpec` of its own; `count`
/// is already computed by `aggregate_rows` for every group regardless of
/// which real metric is chosen, so `bg_compute_matrix` special-cases this
/// label to read that instead of an aggregated metric value.
const OBJECT_COUNT_METRIC_LABEL: &str = "Number of Objects";

/// Drives the Results window's Matrix (Plate/Well) view: which images belong
/// to which well/cell is always controlled by Matrix's own
/// `ResultsState.matrix_group_regex` (Matrix is always regex-grouped, unlike
/// the Table view's None/Image/Folder/Regex picker) and Matrix's own
/// `matrix_class_filter` - both intentionally decoupled from the Table
/// view's `group_by`/`group_regex`/`filter_class_items`, so switching
/// between Table and Matrix never clobbers either view's settings. Matrix
/// also never applies the Table's image filter - a matrix cell always
/// aggregates across every image in its well/group. Plate/well physical
/// dimensions come from the project's `PlateSettings`. Groups objects into
/// wells by reusing `evanalyzer_app::result::aggregate_rows` via
/// `compute_plate_matrix`/`compute_well_matrix` (exactly like the flat
/// grouped table and the heatmap chart reuse the same grouping/rendering
/// building blocks), and pushes a colored grid to `ResultsState`.
pub struct ResultsMatrixController {
    results_ui: slint::Weak<ResultsWindow>,
    app_state: Arc<UiState>,
    /// Reused for the current DB path, column specs, and the Table view's
    /// coloc/t-stack/z-stack filters (still shared - only the image and
    /// class filters are Matrix-specific, see this struct's doc comment).
    results_table_controller: Arc<ResultsTableController>,
    /// Reused so applying plate/well/regex settings from the Matrix view
    /// persists to the project and stays in sync with the Project Settings
    /// dialog (see that controller's struct-level doc comment).
    project_settings_controller: Arc<ProjectSettingsController>,
}

impl ResultsMatrixController {
    pub fn new(
        results_ui: slint::Weak<ResultsWindow>,
        app_state: Arc<UiState>,
        results_table_controller: Arc<ResultsTableController>,
        project_settings_controller: Arc<ProjectSettingsController>,
    ) -> Self {
        Self {
            results_ui,
            app_state,
            results_table_controller,
            project_settings_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(window) = self.results_ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();

        state.set_matrix_agg_options(slint::ModelRc::new(slint::VecModel::from(vec![
            SharedString::from("Min"),
            SharedString::from("Max"),
            SharedString::from("Average"),
            SharedString::from("Median"),
            SharedString::from("Std. dev."),
            SharedString::from("Sum"),
        ])));
        state.set_matrix_color_scheme_options(slint::ModelRc::new(slint::VecModel::from(
            color_scheme_labels(),
        )));

        {
            let this = Arc::clone(self);
            state.on_matrix_apply(move || {
                let Some(config) = Self::build_config(&this) else {
                    return;
                };
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_compute_matrix(this, None, config));
            });
        }
        {
            let this = Arc::clone(self);
            state.on_matrix_well_clicked(move |label: SharedString| {
                let Some(config) = Self::build_config(&this) else {
                    return;
                };
                let this = Arc::clone(&this);
                std::thread::spawn(move || {
                    Self::bg_compute_matrix(this, Some(label.to_string()), config)
                });
            });
        }
        {
            let this = Arc::clone(self);
            state.on_matrix_back_clicked(move || {
                let Some(config) = Self::build_config(&this) else {
                    return;
                };
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_compute_matrix(this, None, config));
            });
        }
        {
            let this = Arc::clone(self);
            state.on_matrix_autodetect_regex_requested(move || {
                let this = Arc::clone(&this);
                std::thread::spawn(move || Self::bg_autodetect_regex(this));
            });
        }
        {
            let this = Arc::clone(self);
            state.on_matrix_image_clicked(move |image_name: SharedString| {
                Self::jump_to_image_in_table(&this, image_name.to_string());
            });
        }
    }

    /// Well view only: filters the Table view down to exactly this
    /// field-of-view image, drops any grouping (so its individual objects
    /// are visible, not one aggregated row), and switches to Table view -
    /// mirrors the toolbar's own image-filter/`filter_apply` flow, just
    /// pre-seeded with a single known image instead of reading checkboxes.
    fn jump_to_image_in_table(this: &Arc<Self>, image_name: String) {
        let Some(window) = this.results_ui.upgrade() else {
            return;
        };
        let state = window.global::<ResultsState>();

        let items: Vec<FilterItem> = model_to_vec(&state.get_filter_image_items())
            .iter()
            .map(|i| FilterItem {
                label: i.label.clone(),
                checked: i.label.as_str() == image_name,
                group: i.group.clone(),
                group_header: i.group_header,
                group_all_checked: i.group_all_checked,
            })
            .collect();
        state.set_filter_image_items(slint::ModelRc::new(slint::VecModel::from(items.clone())));
        state.set_filter_image_popup(slint::ModelRc::new(slint::VecModel::from(items)));
        state.set_filter_image_active(true);
        state.set_filter_image_all_popup_checked(false);
        state.set_filter_active(true);

        state.set_group_by(ResultsGroupBy::None);
        state.set_group_active(false);
        state.set_view_mode(ResultsViewMode::Table);
        state.set_loading_more(true);

        *this.results_table_controller.image_filter.lock().unwrap() = Some(vec![image_name]);
        *this.results_table_controller.group_config.lock().unwrap() = GroupConfig::default();
        *this
            .results_table_controller
            .coloc_detail_mode
            .lock()
            .unwrap() = false;
        *this.results_table_controller.current_page.lock().unwrap() = 0;

        ResultsTableController::spawn_reload(Arc::clone(&this.results_table_controller));
    }

    /// Reads the current picks (Value/Aggregate/Colors/Range, and the
    /// toolbar's Group by/regex) and the visible-numeric-column list, then
    /// seeds the Value/Aggregate pickers if they're not set yet.
    ///
    /// **Must run on the UI thread** - Slint globals may only be read/written
    /// there, and this is called directly from a Slint callback (guaranteed
    /// to already be on the UI thread), never from a spawned background
    /// thread. `bg_compute_matrix` takes the plain `MatrixComputeConfig` this
    /// returns instead of touching `ResultsState` itself until it's ready to
    /// report back through `invoke_from_event_loop`.
    fn build_config(this: &Arc<Self>) -> Option<MatrixComputeConfig> {
        let window = this.results_ui.upgrade()?;
        let state = window.global::<ResultsState>();

        let specs = this
            .results_table_controller
            .column_specs
            .lock()
            .unwrap()
            .clone();
        let metric_options = plottable_labels(&specs);

        let mut metric_label = state.get_matrix_metric().to_string();
        if metric_label.is_empty() || !metric_options.iter().any(|m| m.as_str() == metric_label) {
            metric_label = metric_options
                .first()
                .map(|s| s.to_string())
                .unwrap_or_default();
        }
        let mut agg_label = state.get_matrix_agg().to_string();
        if agg_label.is_empty() {
            agg_label = "Average".to_string();
        }
        let color_scheme_label = state.get_matrix_color_scheme().to_string();
        let range_auto = state.get_matrix_range_auto();
        let range_min = state.get_matrix_range_min() as f64;
        let range_max = state.get_matrix_range_max() as f64;

        // Matrix is always grouped by regex, from its own decoupled
        // `matrix_group_regex` - never the Table view's `group_by`/
        // `group_regex` (see this controller's struct-level doc comment).
        let regex = state.get_matrix_group_regex().to_string();
        // "" (the "All classes" sentinel) means no class filter.
        let class_filter = state.get_matrix_class_filter().to_string();

        state.set_matrix_metric_options(slint::ModelRc::new(slint::VecModel::from(metric_options)));
        state.set_matrix_metric(metric_label.clone().into());
        state.set_matrix_agg(agg_label.clone().into());

        Some(MatrixComputeConfig {
            specs,
            metric_label,
            agg_label,
            color_scheme_label,
            range_auto,
            range_min,
            range_max,
            regex,
            class_filter,
        })
    }

    /// Computes the plate grid (`well_label: None`) or drills into one well's
    /// field-of-view grid (`well_label: Some(...)`), and pushes the result
    /// (or an explanatory status message) to `ResultsState`. `config` was
    /// already read from `ResultsState` on the UI thread by [`Self::build_config`],
    /// since this function itself must not touch any Slint global directly
    /// (it runs on a background thread), only through `invoke_from_event_loop`
    /// via `report_status`/`push_grid`.
    fn bg_compute_matrix(this: Arc<Self>, well_label: Option<String>, config: MatrixComputeConfig) {
        let ui = this.results_ui.clone();
        let report_status = move |status: String| {
            let ui = ui.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(window) = ui.upgrade() else { return };
                window
                    .global::<ResultsState>()
                    .set_matrix_status(status.into());
            });
        };

        // Keep the settings strip's plate/well controls fresh with whatever
        // is actually persisted, regardless of which window last edited them.
        this.project_settings_controller
            .sync_project_settings_to_slint();

        let MatrixComputeConfig {
            specs,
            metric_label,
            agg_label,
            color_scheme_label,
            range_auto,
            range_min,
            range_max,
            regex,
            class_filter,
        } = config;

        if regex.is_empty() {
            report_status(
                "Enter a regex to group wells (e.g. ^([A-Z]\\d+)_) or use Auto-detect.".into(),
            );
            return;
        }

        let Some(path) = this.results_table_controller.path.lock().unwrap().clone() else {
            report_status("No results file loaded.".into());
            return;
        };

        let plate = this.app_state.get_project().plate.clone();

        let is_object_count = metric_label == OBJECT_COUNT_METRIC_LABEL;
        let metric_spec = if is_object_count {
            // "Number of Objects" isn't a real per-object column - it's the
            // group's own size, which `aggregate_rows` computes for every
            // group regardless of which metric it's asked to aggregate. Any
            // visible numeric column works as that required-but-unused
            // grouping vehicle.
            plottable_columns(&specs).into_iter().next().cloned()
        } else {
            specs.iter().find(|c| c.label == metric_label).cloned()
        };
        let Some(metric_spec) = metric_spec else {
            report_status("Pick a value to color the matrix by.".into());
            return;
        };
        let agg = agg_from_label(&agg_label);
        let scheme = HeatmapColorScheme::from_label(&color_scheme_label);

        let loader = ResultsLoader::new(&path);
        // Matrix always aggregates across every image in its well/group - the
        // Table view's image filter never applies here (see this module's
        // doc comment). Class filtering uses Matrix's own decoupled
        // `matrix_class_filter` instead of the Table's `class_filter`.
        let class_filter = (!class_filter.is_empty()).then(|| vec![class_filter]);
        let coloc_filter = this
            .results_table_controller
            .coloc_filter
            .lock()
            .unwrap()
            .clone();
        let t_stack_filter = *this.results_table_controller.t_stack_filter.lock().unwrap();
        let z_stack_filter = *this.results_table_controller.z_stack_filter.lock().unwrap();

        let objects = match loader.get_objects(DatabaseFilter {
            image_filter: None,
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
                warn!("bg_compute_matrix failed to load ROIs: {:?}", e);
                report_status("Failed to load data for the matrix.".into());
                return;
            }
        };
        // Every processed image, regardless of object count or the current
        // class/coloc filter - lets a well/image with nothing matching still
        // render as an occupied-but-empty cell instead of vanishing outright.
        let all_images = match loader.get_images() {
            Ok(images) => images,
            Err(e) => {
                warn!("bg_compute_matrix failed to load the image list: {:?}", e);
                Vec::new()
            }
        };
        if objects.is_empty() && all_images.is_empty() {
            report_status("No ROIs match the current filters.".into());
            return;
        }

        let plate_rows = (plate.plate_rows.max(1)) as usize;
        let plate_cols = (plate.plate_cols.max(1)) as usize;

        if let Some(well_label) = well_label {
            let well_rows = (plate.well_rows.max(1)) as usize;
            let well_cols = (plate.well_cols.max(1)) as usize;
            match compute_well_matrix(
                &objects,
                &all_images,
                &regex,
                &well_label,
                agg,
                &metric_spec,
                well_rows,
                well_cols,
                &plate.well_image_order,
            ) {
                Some(result) => {
                    let plotted: Vec<(String, Option<f64>, usize, usize)> = result
                        .cells
                        .iter()
                        .map(|c| {
                            let occupied = !c.image_name.is_empty();
                            (
                                c.image_name.clone(),
                                cell_value(occupied, is_object_count, c.count, c.value),
                                c.count,
                                c.coloc_count,
                            )
                        })
                        .collect();
                    let values: Vec<f64> = plotted.iter().filter_map(|(_, v, ..)| *v).collect();
                    let (lo, hi) = resolve_range(&values, range_auto, range_min, range_max);
                    let cells: Vec<ResultsMatrixCell> = plotted
                        .into_iter()
                        .map(|(label, value, count, coloc_count)| {
                            matrix_cell(label, value, count, coloc_count, lo, hi, scheme)
                        })
                        .collect();
                    Self::push_grid(
                        &this,
                        ResultsMatrixKind::Well,
                        well_label,
                        result.rows,
                        result.cols,
                        cells,
                        String::new(),
                        (lo, hi),
                    );
                }
                None => {
                    report_status(format!(
                        "Well {well_label} has no sub-position data — check the regex has a 4th capture group."
                    ));
                }
            }
        } else {
            let mut plate_rows = plate_rows;
            let mut plate_cols = plate_cols;
            let plate_values = |result: &PlateMatrixResult| -> Vec<f64> {
                result
                    .cells
                    .iter()
                    .filter_map(|c| {
                        cell_value(!c.label.is_empty(), is_object_count, c.count, c.value)
                    })
                    .collect()
            };

            let mut result = compute_plate_matrix(
                &objects,
                &all_images,
                GroupBy::Regex,
                &regex,
                agg,
                &metric_spec,
                plate_rows,
                plate_cols,
            );
            let mut values: Vec<f64> = plate_values(&result);

            // Every well-shaped label decoded fine, but none of them fit the
            // configured plate size (e.g. still at the "1 Well" default) -
            // grow to the smallest preset that fits everything and retry,
            // instead of leaving the user staring at an empty grid.
            if values.is_empty()
                && let Some((need_rows, need_cols)) = result.required_span
                && let Some((new_rows, new_cols)) = fitting_preset(need_rows, need_cols)
            {
                plate_rows = new_rows;
                plate_cols = new_cols;
                {
                    let mut project = this.app_state.get_project_write();
                    project.plate.plate_rows = new_rows as i32;
                    project.plate.plate_cols = new_cols as i32;
                }
                this.app_state.mark_dirty();
                this.project_settings_controller
                    .sync_project_settings_to_slint();

                result = compute_plate_matrix(
                    &objects,
                    &all_images,
                    GroupBy::Regex,
                    &regex,
                    agg,
                    &metric_spec,
                    plate_rows,
                    plate_cols,
                );
                values = plate_values(&result);
            }

            let status = if values.is_empty() {
                match (
                    result.required_span,
                    result.group_count,
                    &result.sample_label,
                ) {
                    (Some((r, c)), ..) => format!(
                        "No wells fit — this data needs at least a {r} x {c} plate. Increase Plate size."
                    ),
                    (None, count, Some(sample)) if count > 0 => {
                        format!(
                            "Group by regex matched {count} group(s), but none look like well ids (e.g. \"D14\") - got \"{sample}\" instead. Capture group 1 should be just the well id, not the whole match - check for an extra wrapping parenthesis."
                        )
                    }
                    _ => "No wells matched — check Matrix's regex.".to_string(),
                }
            } else {
                String::new()
            };
            let (lo, hi) = resolve_range(&values, range_auto, range_min, range_max);
            let cells: Vec<ResultsMatrixCell> = result
                .cells
                .iter()
                .map(|c| {
                    let occupied = !c.label.is_empty();
                    matrix_cell(
                        c.label.clone(),
                        cell_value(occupied, is_object_count, c.count, c.value),
                        c.count,
                        c.coloc_count,
                        lo,
                        hi,
                        scheme,
                    )
                })
                .collect();
            Self::push_grid(
                &this,
                ResultsMatrixKind::Plate,
                String::new(),
                result.rows,
                result.cols,
                cells,
                status,
                (lo, hi),
            );
        }
    }

    /// `range` is the `(min, max)` this render actually colored with (from
    /// `resolve_range`) - pushed back into `matrix_range_min`/`max`
    /// unconditionally, same as the heatmap chart does for its own range
    /// fields, so they show the live auto-computed span while Auto is on and
    /// echo back a manual pick as a no-op otherwise; never stale either way.
    #[allow(clippy::too_many_arguments)]
    fn push_grid(
        this: &Arc<Self>,
        kind: ResultsMatrixKind,
        active_well: String,
        rows: usize,
        cols: usize,
        cells: Vec<ResultsMatrixCell>,
        status: String,
        range: (f64, f64),
    ) {
        let ui = this.results_ui.clone();
        let _ = slint::invoke_from_event_loop(move || {
            let Some(window) = ui.upgrade() else { return };
            let state = window.global::<ResultsState>();
            state.set_matrix_kind(kind);
            state.set_matrix_active_well(active_well.into());
            state.set_matrix_rows(rows as i32);
            state.set_matrix_cols(cols as i32);
            state.set_matrix_cells(slint::ModelRc::new(slint::VecModel::from(cells)));
            state.set_matrix_status(status.into());
            state.set_matrix_range_min(range.0 as f32);
            state.set_matrix_range_max(range.1 as f32);
        });
    }

    /// Suggests a regex from the current image names and, on success:
    /// - sets it on Matrix's own `matrix_group_regex` so it takes effect
    ///   immediately, and
    /// - persists it to the project's `PlateSettings` (mirroring what typing
    ///   directly into Matrix's own regex field does) and re-syncs the
    ///   Project Settings dialog, so it isn't lost the next time the project
    ///   is opened.
    fn bg_autodetect_regex(this: Arc<Self>) {
        let ui = this.results_ui.clone();
        let report_hint = move |hint: String| {
            let ui = ui.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(window) = ui.upgrade() else { return };
                window
                    .global::<ResultsState>()
                    .set_matrix_regex_hint(hint.into());
            });
        };

        let Some(path) = this.results_table_controller.path.lock().unwrap().clone() else {
            report_hint("No results file loaded.".to_string());
            return;
        };
        let loader = ResultsLoader::new(&path);
        let names = match loader.get_image_names() {
            Ok(names) => names,
            Err(e) => {
                warn!(
                    "matrix regex autodetect failed to load image names: {:?}",
                    e
                );
                report_hint("Failed to load image names.".to_string());
                return;
            }
        };

        match suggest_regex(&names) {
            Some(suggestion) => {
                let hint = format!(
                    "Matched {}/{} filenames",
                    suggestion.matched, suggestion.total
                );
                let pattern = suggestion.pattern;

                {
                    let mut project = this.app_state.get_project_write();
                    project.plate.grouping_mode = GroupingMode::FileName;
                    project.plate.grouping_regex = pattern.clone();
                }
                this.app_state.mark_dirty();
                this.project_settings_controller
                    .sync_project_settings_to_slint();

                let ui = this.results_ui.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(window) = ui.upgrade() else { return };
                    let state = window.global::<ResultsState>();
                    state.set_matrix_group_regex(pattern.into());
                    state.set_matrix_regex_hint(hint.into());
                });
            }
            None => {
                report_hint("No consistent pattern found — enter a regex manually.".to_string());
            }
        }
    }
}

/// UI picks read on the UI thread by [`ResultsMatrixController::build_config`]
/// and handed to the background thread, which must not touch `ResultsState`
/// itself (see that function's doc comment).
struct MatrixComputeConfig {
    specs: Vec<ColumnSpec>,
    metric_label: String,
    agg_label: String,
    color_scheme_label: String,
    range_auto: bool,
    range_min: f64,
    range_max: f64,
    regex: String,
    class_filter: String,
}

/// The smallest of the 13 plate-size presets (in list order, same as the
/// "Plate size" dropdown) whose `(rows, cols)` both cover `(need_rows,
/// need_cols)`, or `None` if even the largest preset (3456-well, 48x72)
/// isn't big enough.
fn fitting_preset(need_rows: usize, need_cols: usize) -> Option<(usize, usize)> {
    (0..=12).map(index_to_well_size).find_map(|(rows, cols)| {
        let (rows, cols) = (rows as usize, cols as usize);
        (rows >= need_rows && cols >= need_cols).then_some((rows, cols))
    })
}

pub(crate) fn agg_from_label(label: &str) -> AggFunc {
    match label {
        "Min" => AggFunc::Min,
        "Max" => AggFunc::Max,
        "Median" => AggFunc::Median,
        "Std. dev." => AggFunc::Stdev,
        "Sum" => AggFunc::Sum,
        _ => AggFunc::Avg,
    }
}

fn plottable_labels(specs: &[ColumnSpec]) -> Vec<SharedString> {
    let mut labels: Vec<SharedString> = vec![SharedString::from(OBJECT_COUNT_METRIC_LABEL)];
    labels.extend(
        plottable_columns(specs)
            .iter()
            .map(|c| SharedString::from(c.label.as_str())),
    );
    labels
}

fn color_scheme_labels() -> Vec<SharedString> {
    HeatmapColorScheme::all()
        .iter()
        .map(|s| SharedString::from(s.label()))
        .collect()
}

/// The value a cell should be colored/positioned by: for the
/// [`OBJECT_COUNT_METRIC_LABEL`] pseudo-metric, that's the cell's own object
/// count - well-defined (including zero) for any occupied cell, real or a
/// zero-object placeholder - rather than its (nonexistent) aggregated metric
/// value. For a real metric, unchanged: whatever `compute_plate_matrix`/
/// `compute_well_matrix` already computed.
fn cell_value(
    occupied: bool,
    is_object_count: bool,
    count: usize,
    value: Option<f64>,
) -> Option<f64> {
    if is_object_count {
        occupied.then_some(count as f64)
    } else {
        value
    }
}

/// Formats a cell's object/coloc-object count summary, e.g. `"12 obj"` or
/// `"12 obj · 3 coloc"` — the coloc half is only shown when there's something
/// to show, since most datasets never colocalize anything.
fn count_line(count: usize, coloc_count: usize) -> String {
    if coloc_count > 0 {
        format!("{count} obj \u{b7} {coloc_count} coloc")
    } else {
        format!("{count} obj")
    }
}

fn matrix_cell(
    label: String,
    value: Option<f64>,
    count: usize,
    coloc_count: usize,
    range_lo: f64,
    range_hi: f64,
    scheme: HeatmapColorScheme,
) -> ResultsMatrixCell {
    // A non-empty label means a well/image landed here — either with a real
    // metric value, or as a zero-object placeholder (see `all_images` on
    // `compute_plate_matrix`/`compute_well_matrix`) — as opposed to a
    // genuinely unoccupied slot, which arrives with an empty label.
    let occupied = !label.is_empty();
    let count_line: SharedString = if occupied {
        count_line(count, coloc_count).into()
    } else {
        SharedString::new()
    };
    match value {
        Some(v) => {
            let t = ((v - range_lo) / (range_hi - range_lo)).clamp(0.0, 1.0);
            let (r, g, b) = scheme.color_rgb(t);
            ResultsMatrixCell {
                label: label.into(),
                value: format!("{v:.1}").into(),
                count_line,
                color: slint::Color::from_rgb_u8(r, g, b),
                has_value: true,
                occupied: true,
            }
        }
        None => ResultsMatrixCell {
            label: label.into(),
            value: SharedString::new(),
            count_line,
            color: slint::Color::from_rgb_u8(0, 0, 0),
            has_value: false,
            occupied,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_preset_picks_smallest_covering_preset() {
        // Real-world case: a sparse plate with only rows D/G populated and
        // columns 2-19 needs at least a 7x19 span - the 96-well (8x12) preset
        // is too narrow (12 < 19), so it should skip to 384-well (16x24).
        assert_eq!(fitting_preset(7, 19), Some((16, 24)));
    }

    #[test]
    fn fitting_preset_exact_match() {
        assert_eq!(fitting_preset(8, 12), Some((8, 12)));
    }

    #[test]
    fn fitting_preset_none_beyond_largest() {
        assert_eq!(fitting_preset(100, 100), None);
    }
    // `resolve_range` itself is tested in `evanalyzer_app::results::plate_matrix`
    // (this module just re-uses it) - no need to duplicate those cases here.

    // -- plottable_labels ---------------------------------------------------------

    fn column(id: &str, label: &str, visible: bool) -> ColumnSpec {
        ColumnSpec {
            id: id.to_string(),
            label: label.to_string(),
            filterable: false,
            visible,
        }
    }

    #[test]
    fn plottable_labels_maps_each_plottable_columns_label_field() {
        let cols = vec![
            column("area_px", "Area (px)", true),
            column("image", "Image", true), // non-numeric, excluded upstream
        ];

        let labels: Vec<String> = plottable_labels(&cols)
            .iter()
            .map(|s| s.to_string())
            .collect();

        assert_eq!(
            labels,
            vec![
                OBJECT_COUNT_METRIC_LABEL.to_string(),
                "Area (px)".to_string()
            ]
        );
    }

    // -- color_scheme_labels --------------------------------------------------------

    #[test]
    fn color_scheme_labels_lists_every_scheme_in_declaration_order() {
        let labels: Vec<String> = color_scheme_labels()
            .iter()
            .map(|s| s.to_string())
            .collect();

        assert_eq!(
            labels,
            vec![
                "Viridis".to_string(),
                "Magma".to_string(),
                "Plasma".to_string(),
                "Grayscale".to_string(),
            ]
        );
    }

    // -- cell_value -------------------------------------------------------------------

    #[test]
    fn cell_value_for_a_real_metric_passes_the_aggregated_value_through_unchanged() {
        assert_eq!(cell_value(true, false, 5, Some(0.75)), Some(0.75));
        assert_eq!(cell_value(true, false, 5, None), None);
        assert_eq!(cell_value(false, false, 0, None), None);
    }

    #[test]
    fn cell_value_for_object_count_uses_count_for_any_occupied_cell_including_zero() {
        // A real well/image with objects.
        assert_eq!(cell_value(true, true, 5, Some(0.75)), Some(5.0));
        // A zero-object placeholder - still occupied, count is a real 0, not blank.
        assert_eq!(cell_value(true, true, 0, None), Some(0.0));
        // A genuinely unoccupied grid slot stays blank regardless of metric.
        assert_eq!(cell_value(false, true, 0, None), None);
    }

    // -- matrix_cell ----------------------------------------------------------------

    #[test]
    fn matrix_cell_with_a_value_formats_it_and_marks_has_value() {
        let cell = matrix_cell(
            "A1".to_string(),
            Some(42.567),
            12,
            3,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );

        assert_eq!(cell.label.as_str(), "A1");
        assert_eq!(cell.value.as_str(), "42.6");
        assert_eq!(cell.count_line.as_str(), "12 obj \u{b7} 3 coloc");
        assert!(cell.has_value);
        assert!(cell.occupied);
    }

    #[test]
    fn matrix_cell_count_line_omits_coloc_half_when_zero() {
        let cell = matrix_cell(
            "A1".to_string(),
            Some(1.0),
            12,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );
        assert_eq!(cell.count_line.as_str(), "12 obj");
    }

    #[test]
    fn matrix_cell_always_shows_one_decimal_even_for_metrics_the_table_view_shows_with_more() {
        // `matrix_cell`'s `{:.1}` is fixed regardless of which metric is
        // plotted, but the Table view (and CSV/XLSX export) format each
        // metric with its own `metric_precision` - 3 decimals for
        // Circularity, 2 for Area (nm²), 0 for a coloc partner count, only
        // Area (px²) and channel bit values happen to use 1. So for a
        // circularity average of 0.853 (what the Table view would display
        // verbatim, e.g. "0.853"), the on-screen Matrix cell shows "0.9" -
        // a real loss of the two least-significant digits, not just
        // cosmetic zero-padding (contrast `export_matrix_to_csv`, which
        // only ever *adds* trailing zeros, never drops real digits).
        // Documented here as current, intentional-or-not behavior - flag if
        // the Matrix grid should instead respect each metric's own
        // precision like every other view does.
        let cell = matrix_cell(
            "A1".to_string(),
            Some(0.853),
            1,
            0,
            0.0,
            1.0,
            HeatmapColorScheme::Grayscale,
        );

        assert_eq!(
            cell.value.as_str(),
            "0.9",
            "Matrix cell drops precision the Table view (\"0.853\") would show"
        );
    }

    #[test]
    fn matrix_cell_displayed_value_always_stays_within_the_expected_rounding_tolerance() {
        // Guardrail for the intentional precision loss documented above:
        // `matrix_cell` is *allowed* to round its input down to 1 decimal
        // (by design, kept as-is - see the test above), but the text it
        // shows must still be a faithful rounding of the value it was
        // given, not some unrelated/corrupted number. `{:.1}` rounds to the
        // nearest 0.1, so the reparsed displayed value can never be more
        // than half that (0.05) away from the source `f64`, regardless of
        // how many decimals the source itself carries (this covers every
        // real `metric_precision` case: 0 decimals for a coloc partner
        // count, 1 for Area (px²)/channel bit values, 2 for Area (nm²), 3
        // for Circularity). A future change to `matrix_cell` that reads the
        // wrong field, applies the wrong scale, or otherwise mangles the
        // value would push it outside this window and fail here even
        // though the exact-string test above only pins one example.
        const HALF_STEP: f64 = 0.05 + 1e-9; // rounding half-step + float slop
        let sources = [
            0.0,    // coloc partner count precision (0 decimals)
            7.0,    // "
            42.567, // Area (px²) / channel bit precision (1 decimal)
            -3.14,  // negative values must round the same way
            123.45, // Area (nm²) precision (2 decimals)
            0.853,  // Circularity precision (3 decimals)
            0.85,   // exactly at a rounding half-step (0.8 vs 0.9) - the
                    // tightest case HALF_STEP has to cover
        ];

        for &source in &sources {
            let cell = matrix_cell(
                "A1".to_string(),
                Some(source),
                1,
                0,
                source.min(0.0),
                source.max(1.0),
                HeatmapColorScheme::Viridis,
            );
            let displayed: f64 = cell.value.parse().unwrap_or_else(|e| {
                panic!("cell value {:?} must parse as a number: {e}", cell.value)
            });

            assert!(
                (displayed - source).abs() <= HALF_STEP,
                "source {source} rounded to {displayed}, which is more than one rounding \
                 half-step (±{HALF_STEP}) away - not a valid 1-decimal rounding"
            );
        }
    }

    #[test]
    fn matrix_cell_with_an_empty_label_is_a_genuinely_unoccupied_placeholder() {
        // An empty label only ever comes from `empty_plate_cell`/`empty_well_cell`
        // (no well/image landed in that grid slot at all) - unlike the
        // zero-object case below, this cell must stay fully blank and disabled.
        let cell = matrix_cell(
            String::new(),
            None,
            0,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Viridis,
        );

        assert_eq!(cell.label.as_str(), "");
        assert_eq!(cell.value.as_str(), "");
        assert_eq!(cell.count_line.as_str(), "");
        assert!(!cell.has_value);
        assert!(!cell.occupied);
        assert_eq!(cell.color, slint::Color::from_rgb_u8(0, 0, 0));
    }

    #[test]
    fn matrix_cell_with_a_label_but_no_value_is_occupied_but_uncolored() {
        // A real well/image that produced zero (matching) objects: `label` is
        // set (from `all_images`) but `value` is `None` - the cell must keep
        // its label/count_line and stay clickable, just without a heatmap color.
        let cell = matrix_cell(
            "A15".to_string(),
            None,
            0,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Viridis,
        );

        assert_eq!(cell.label.as_str(), "A15");
        assert_eq!(cell.value.as_str(), "");
        assert_eq!(cell.count_line.as_str(), "0 obj");
        assert!(!cell.has_value);
        assert!(cell.occupied);
        assert_eq!(cell.color, slint::Color::from_rgb_u8(0, 0, 0));
    }

    #[test]
    fn matrix_cell_clamps_values_outside_the_range_to_the_scale_endpoints() {
        let below = matrix_cell(
            "x".to_string(),
            Some(-50.0),
            1,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );
        let at_min = matrix_cell(
            "x".to_string(),
            Some(0.0),
            1,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );
        assert_eq!(
            below.color, at_min.color,
            "values below range_lo must clamp to t=0.0"
        );

        let above = matrix_cell(
            "x".to_string(),
            Some(500.0),
            1,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );
        let at_max = matrix_cell(
            "x".to_string(),
            Some(100.0),
            1,
            0,
            0.0,
            100.0,
            HeatmapColorScheme::Grayscale,
        );
        assert_eq!(
            above.color, at_max.color,
            "values above range_hi must clamp to t=1.0"
        );
    }

    // -- attach_callbacks (live ResultsWindow) -------------------------------------

    use crate::editor::images_list_controller::ImagesListController;
    use crate::editor::results_table_controller::ResultsTableController;
    use crate::editor::test_support::{test_ui_state, test_ui_windows};
    use slint::Model;

    fn make_controller(
        results_ui: slint::Weak<ResultsWindow>,
    ) -> (Arc<UiState>, Arc<ResultsMatrixController>) {
        let ui_state = test_ui_state();
        let viewport_controller =
            Arc::new(crate::editor::viewport_controller::ViewportController::new(
                slint::Weak::default(),
                ui_state.clone(),
            ));
        let object_list_controller = Arc::new(
            crate::editor::object_list_controller::ObjectListController::new(
                slint::Weak::default(),
                ui_state.clone(),
                viewport_controller.clone(),
            ),
        );
        let image_list_controller = Arc::new(ImagesListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(
                crate::editor::histogram_controller::HistogramController::new(
                    slint::Weak::default(),
                    ui_state.clone(),
                    viewport_controller.clone(),
                ),
            ),
            Arc::new(
                crate::editor::image_meta_controller::ImageMetaController::new(
                    slint::Weak::default(),
                    ui_state.clone(),
                    viewport_controller.clone(),
                ),
            ),
            object_list_controller,
        ));
        let results_table_controller = Arc::new(ResultsTableController::new(
            results_ui.clone(),
            ui_state.clone(),
            image_list_controller,
        ));
        let project_settings_controller = Arc::new(ProjectSettingsController::new(
            slint::Weak::default(),
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let controller = Arc::new(ResultsMatrixController::new(
            results_ui,
            ui_state.clone(),
            results_table_controller,
            project_settings_controller,
        ));
        (ui_state, controller)
    }

    #[test]
    fn attach_callbacks_populates_the_agg_and_color_scheme_option_lists() {
        let (_ui, results_ui) = test_ui_windows();
        let (_ui_state, controller) = make_controller(results_ui.as_weak());

        controller.attach_callbacks();

        let state = results_ui.global::<ResultsState>();
        let agg_options: Vec<String> = state
            .get_matrix_agg_options()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            agg_options,
            vec!["Min", "Max", "Average", "Median", "Std. dev.", "Sum"]
        );

        let color_options: Vec<String> = state
            .get_matrix_color_scheme_options()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            color_options,
            vec!["Viridis", "Magma", "Plasma", "Grayscale"]
        );
    }
}
