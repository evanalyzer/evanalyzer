//! The project's results views (`ProjectSettings::results_views`): each a
//! List, Plate, Charts or Tracks with its own settings, the same for every
//! analysis of the project. The view shown saves itself: after each refresh
//! its settings are written back (`remember_view`). Switching loads the
//! other view's settings and queries anew - no view is kept in memory.
//!
//! Settings name classes by id and columns by key (`Column::as_key`), so
//! they fit any analysis of the project. What an analysis lacks (a column of
//! a step it didn't have) is left out of the table and said so - and stays
//! in the view for the next analysis.

use super::*;
use evanalyzer_app::results::view_settings::{ChartView, ListView, PlateView, ViewSettings};
use evanalyzer_cfg::settings::project_settings::{ResultsView, ResultsViewKind};

/// What the view shown named but this analysis lacks - kept, so writing the
/// view back doesn't lose it. A single column stays while its stand-in
/// (what's shown instead) is still chosen.
#[derive(Default)]
pub(super) struct ViewUnavailable {
    list_classes: Vec<u32>,
    list_columns: Vec<String>,
    plate_column: Option<(String, String)>,
    chart_column: Option<(String, String)>,
    chart_y_column: Option<(String, String)>,
    plate_class: Option<u32>,
    chart_class: Option<u32>,
}

/// The view tabs as the bar shows them.
fn view_tabs(views: &[ResultsView], has_tracks: bool) -> Vec<crate::ResultsViewTab> {
    views
        .iter()
        .map(|view| crate::ResultsViewTab {
            id: view.id as i32,
            name: view.name.as_str().into(),
            kind: kind_label(view.kind).into(),
            available: view.kind != ResultsViewKind::Tracks || has_tracks,
            // "List", "Plate 2": the name says it already.
            show_kind: !view.name.to_uppercase().starts_with(kind_label(view.kind)),
        })
        .collect()
}

fn kind_label(kind: ResultsViewKind) -> &'static str {
    match kind {
        ResultsViewKind::List => "LIST",
        ResultsViewKind::Plate => "PLATE",
        ResultsViewKind::Charts => "CHART",
        ResultsViewKind::Tracks => "TRACKS",
    }
}

fn kind_of_key(key: &str) -> Option<ResultsViewKind> {
    match key {
        "list" => Some(ResultsViewKind::List),
        "plate" => Some(ResultsViewKind::Plate),
        "charts" => Some(ResultsViewKind::Charts),
        "tracks" => Some(ResultsViewKind::Tracks),
        _ => None,
    }
}

fn rail_mode_of(kind: ResultsViewKind) -> ResultsRailMode {
    match kind {
        ResultsViewKind::List => ResultsRailMode::List,
        ResultsViewKind::Plate => ResultsRailMode::Matrix,
        ResultsViewKind::Charts => ResultsRailMode::Charts,
        ResultsViewKind::Tracks => ResultsRailMode::Tracks,
    }
}

/// A name not yet used: "List", else "List 2", "List 3", ...
fn free_name(views: &[ResultsView], base: &str) -> String {
    (1..)
        .map(|n| match n {
            1 => base.to_string(),
            n => format!("{base} {n}"),
        })
        .find(|name| !views.iter().any(|view| view.name == *name))
        .expect("a free name")
}

fn chart_kind_key(kind: ChartKind) -> &'static str {
    match kind {
        ChartKind::Histogram => "histogram",
        ChartKind::Scatter => "scatter",
        ChartKind::Boxplot => "boxplot",
        ChartKind::OverTime => "overTime",
    }
}

fn chart_kind_of_key(key: &str) -> ChartKind {
    match key {
        "scatter" => ChartKind::Scatter,
        "boxplot" => ChartKind::Boxplot,
        "overTime" => ChartKind::OverTime,
        _ => ChartKind::Histogram,
    }
}

fn points_index(limit: Option<usize>) -> i32 {
    match limit {
        Some(500) => 0,
        Some(10_000) => 2,
        None => 3,
        Some(_) => 1,
    }
}

fn image_state_key(state: ImageState) -> &'static str {
    match state {
        ImageState::All => "all",
        ImageState::Enabled => "enabled",
        ImageState::Disabled => "disabled",
    }
}

fn image_state_of_key(key: &str) -> ImageState {
    match key {
        "enabled" => ImageState::Enabled,
        "disabled" => ImageState::Disabled,
        _ => ImageState::All,
    }
}

fn aggregation_label(aggregation: &Aggregation) -> &'static str {
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

fn class_id(class: ObjectClass) -> Option<u32> {
    match class {
        ObjectClass::Global(id) => Some(id),
        _ => None,
    }
}

impl ResultsStateController {
    pub(super) fn attach_views_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        // The project's views show before any analysis is opened.
        self.push_views_in_slint();
        // An undo or redo brought other views back (a deleted tab, a
        // rename): show them and the active one anew. Weak: `UiState`
        // keeps the listener, and this controller keeps `UiState`.
        let weak = Arc::downgrade(self);
        self._app_state.on_results_views_restored(move || {
            if let Some(manager) = weak.upgrade() {
                manager.run_db_job(|manager| manager.load_active_view());
            }
        });
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_view_selected(move |id| manager.select_view(id as u32));
        let manager = self.clone();
        state.on_view_new(move |key| manager.new_view(key.as_str()));
        let manager = self.clone();
        state.on_view_duplicate(move |id| manager.duplicate_view(id as u32));
        let manager = self.clone();
        state.on_view_show_as(move |id, key| {
            if let Some(kind) = kind_of_key(key.as_str()) {
                manager.show_view_as(id as u32, kind);
            }
        });
        let manager = self.clone();
        state.on_view_rename(move |id, name| manager.rename_view(id as u32, name.as_str()));
        let manager = self.clone();
        state.on_view_revert(move |id| manager.revert_view(id as u32));
        let manager = self.clone();
        state.on_view_delete(move |id| manager.delete_view(id as u32));
        let manager = self.clone();
        state.on_view_move(move |id, delta| manager.move_view(id as u32, delta));
    }

    /// The tabs, the active one, its kind and split - into the bar.
    pub(super) fn push_views_in_slint(&self) {
        let (views, active) = {
            let project = self._app_state.get_project();
            (
                project.results_views.views.clone(),
                project.results_views.active,
            )
        };
        let revertable: Vec<bool> = views
            .iter()
            .map(|view| {
                self._app_state
                    .saved_results_view(view.id)
                    .is_some_and(|saved| saved != *view)
            })
            .collect();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let tabs: Vec<crate::ResultsViewTab> = view_tabs(&views, state.get_has_tracks());
            let active_view = views.iter().find(|view| view.id == active);
            state.set_views(ModelRc::from(Rc::new(VecModel::from(tabs))));
            state.set_active_view(active as i32);
            state.set_views_revertable(ModelRc::from(Rc::new(VecModel::from(revertable))));
            if let Some(view) = active_view {
                state.set_rail_mode(rail_mode_of(view.kind));
                state.set_split(view.split);
            }
        })
        .ok();
    }

    /// The view shown.
    fn active_view(&self) -> Option<ResultsView> {
        let project = self._app_state.get_project();
        let views = &project.results_views;
        views
            .views
            .iter()
            .find(|view| view.id == views.active)
            .or_else(|| views.views.first())
            .cloned()
    }

    /// Opening results: the view shown with its settings, on what the
    /// filters were just reset to.
    pub(super) fn apply_active_view(&self) {
        let Some(view) = self.active_view() else {
            return;
        };
        let settings = ViewSettings::read(&view.settings);
        let missing = self.apply_view_settings(&settings);
        self.push_view_controls_in_slint();
        self.push_views_in_slint();
        self.push_view_missing(missing);
        // The split's state follows the view.
        let shown_split =
            view.split && matches!(view.kind, ResultsViewKind::List | ResultsViewKind::Plate);
        self.split.lock().expect("Poisoned").shown = shown_split;
        *self.rail_mode.lock().expect("Poisoned") = match view.kind {
            ResultsViewKind::List => ResultsRailMode::List,
            ResultsViewKind::Plate => ResultsRailMode::Matrix,
            ResultsViewKind::Charts => ResultsRailMode::Charts,
            ResultsViewKind::Tracks => ResultsRailMode::Tracks,
        };
    }

    /// Sets the filters from `settings` (on the reset ones); what this
    /// analysis lacks is kept aside (`view_unavailable`) and named.
    fn apply_view_settings(&self, settings: &ViewSettings) -> Vec<String> {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let has_class = |id: u32| {
            classes
                .iter()
                .any(|c| c.global() == ObjectClass::Global(id))
        };
        let resolve = |key: &str| column_for_key(key, &classes, &available);
        let mut missing: Vec<String> = Vec::new();
        let mut unavailable = ViewUnavailable::default();

        {
            let mut list = self.list_filter.lock().expect("Poisoned");
            let view = &settings.list;
            if let Some(ids) = &view.classes {
                list.object_classes = ids
                    .iter()
                    .filter(|id| has_class(**id))
                    .map(|id| ObjectClass::Global(*id))
                    .collect();
                unavailable.list_classes =
                    ids.iter().copied().filter(|id| !has_class(*id)).collect();
                missing.extend(
                    unavailable
                        .list_classes
                        .iter()
                        .map(|id| format!("class {id}")),
                );
            }
            if let Some(keys) = &view.columns {
                list.columns = keys.iter().filter_map(|key| resolve(key)).collect();
                unavailable.list_columns = keys
                    .iter()
                    .filter(|key| resolve(key).is_none())
                    .cloned()
                    .collect();
                missing.extend(unavailable.list_columns.iter().cloned());
            }
            if !view.aggregations.is_empty() {
                list.aggregations = view.aggregations.clone();
            }
            list.group_by = match view.rows_per_image {
                true => ListGroupBy::Images,
                false => ListGroupBy::Objects,
            };
            list.transpond = view.transposed;
            list.with_coloc_details = view.coloc_details;
            list.image_state = image_state_of_key(&view.images);
            list.children = view
                .children
                .iter()
                .map(|(p, c)| (ObjectClass::Global(*p), ObjectClass::Global(*c)))
                .collect();
            list.child_aggregations = view.child_aggregations.clone();
            list.parents = view.parents.iter().copied().map(ObjectClass::Global).collect();
        }

        {
            let mut guard = self.matrix_filter.lock().expect("Poisoned");
            let plate = guard.get_or_insert_with(MatrixFilter::default);
            let view = &settings.plate;
            if let Some(id) = view.class {
                match has_class(id) {
                    true => plate.object_classe = ObjectClass::Global(id),
                    false => {
                        unavailable.plate_class = Some(id);
                        missing.push(format!("class {id}"));
                    }
                }
            }
            if let Some(key) = &view.column {
                match resolve(key) {
                    Some(column) => plate.column = column,
                    None => {
                        unavailable.plate_column =
                            Some((key.clone(), plate.column.as_key(&classes)));
                        missing.push(key.clone());
                    }
                }
            }
            plate.aggregation = view.aggregation.clone();
            plate.color_schema = view.color_schema.clone();
            plate.color_scale = match view.color_scale {
                Some((min, max)) => ColorScale::Manual(min, max),
                None => ColorScale::Auto,
            };
            plate.square_size = view.square_size;
        }

        {
            let mut guard = self.chart_filter.lock().expect("Poisoned");
            // Through a plain reference: its fields borrowed apart below.
            let chart: &mut ChartFilter = &mut guard;
            let view = &settings.chart;
            chart.kind = chart_kind_of_key(&view.kind);
            for (key, column, slot) in [
                (
                    &view.column,
                    &mut chart.column,
                    &mut unavailable.chart_column,
                ),
                (
                    &view.y_column,
                    &mut chart.y_column,
                    &mut unavailable.chart_y_column,
                ),
            ] {
                if let Some(key) = key {
                    match resolve(key) {
                        Some(found) => *column = found,
                        None => {
                            *slot = Some((key.clone(), column.as_key(&classes)));
                            if !missing.contains(key) {
                                missing.push(key.clone());
                            }
                        }
                    }
                }
            }
            chart.object_class = match view.class {
                Some(id) if has_class(id) => Some(ObjectClass::Global(id)),
                Some(id) => {
                    unavailable.chart_class = Some(id);
                    missing.push(format!("class {id}"));
                    None
                }
                None => None,
            };
            chart.group_by = view.group_by;
            chart.bins = view.bins.clamp(5, 200);
            chart.log_counts = view.log_counts;
            chart.max_points = scatter_points_limit(view.points);
            chart.time_lines = view.time_lines;
            chart.time_median = view.time_median;
            chart.time_relative = view.time_relative;
            chart.time_tracks = view.time_tracks;
        }
        *self.view_unavailable.lock().expect("Poisoned") = Some(unavailable);
        missing.dedup();
        missing
    }

    /// The view's settings as the filters hold them now - with what this
    /// analysis lacks kept from the view.
    fn capture_view_settings(&self) -> Option<ViewSettings> {
        let guard = self.view_unavailable.lock().expect("Poisoned");
        let unavailable = guard.as_ref()?;
        let classes = self.classes.lock().expect("Poisoned").clone();
        let key = |column: &Column| column.as_key(&classes);
        let all_classes: Vec<u32> = classes
            .iter()
            .filter_map(|c| class_id(c.global()))
            .collect();

        let list = {
            let filter = self.list_filter.lock().expect("Poisoned");
            let mut shown: Vec<u32> = filter
                .object_classes
                .iter()
                .filter_map(|c| class_id(*c))
                .collect();
            let every = unavailable.list_classes.is_empty()
                && all_classes.iter().all(|id| shown.contains(id))
                && shown.len() == all_classes.len();
            shown.extend(unavailable.list_classes.iter().copied());
            let mut columns: Vec<String> = filter.columns.iter().map(key).collect();
            columns.extend(unavailable.list_columns.iter().cloned());
            let default_columns = unavailable.list_columns.is_empty()
                && filter.columns.as_slice() == DEFAULT_LIST_COLUMNS.as_slice();
            ListView {
                classes: (!every).then_some(shown),
                columns: (!default_columns).then_some(columns),
                aggregations: filter.aggregations.clone(),
                rows_per_image: filter.group_by == ListGroupBy::Images,
                transposed: filter.transpond,
                coloc_details: filter.with_coloc_details,
                images: image_state_key(filter.image_state).into(),
                children: filter
                    .children
                    .iter()
                    .filter_map(|(p, c)| Some((class_id(*p)?, class_id(*c)?)))
                    .collect(),
                child_aggregations: filter.child_aggregations.clone(),
                parents: filter.parents.iter().filter_map(|p| class_id(*p)).collect(),
            }
        };
        let kept = |slot: &Option<(String, String)>, now: String| match slot {
            Some((wanted, stand_in)) if *stand_in == now => wanted.clone(),
            _ => now,
        };
        let plate = {
            let guard = self.matrix_filter.lock().expect("Poisoned");
            let filter = guard.as_ref()?;
            PlateView {
                class: class_id(filter.object_classe).or(unavailable.plate_class),
                column: Some(kept(&unavailable.plate_column, key(&filter.column))),
                aggregation: filter.aggregation.clone(),
                color_schema: filter.color_schema.clone(),
                color_scale: match filter.color_scale {
                    ColorScale::Manual(min, max) => Some((min, max)),
                    ColorScale::Auto => None,
                },
                square_size: filter.square_size,
            }
        };
        let chart = {
            let filter = self.chart_filter.lock().expect("Poisoned");
            ChartView {
                kind: chart_kind_key(filter.kind).into(),
                column: Some(kept(&unavailable.chart_column, key(&filter.column))),
                y_column: Some(kept(&unavailable.chart_y_column, key(&filter.y_column))),
                class: filter
                    .object_class
                    .and_then(class_id)
                    .or(unavailable.chart_class),
                group_by: filter.group_by,
                bins: filter.bins,
                log_counts: filter.log_counts,
                points: points_index(filter.max_points),
                time_lines: filter.time_lines,
                time_median: filter.time_median,
                time_relative: filter.time_relative,
                time_tracks: filter.time_tracks,
            }
        };
        Some(ViewSettings { list, plate, chart })
    }

    /// Writes the filters into the view shown, if they changed - the
    /// project then has unsaved changes (not when the view was never shown
    /// before: its first settings are the defaults).
    pub(super) fn remember_view(&self) {
        let Some(settings) = self.capture_view_settings() else {
            return;
        };
        let Ok(value) = serde_json::to_value(&settings) else {
            return;
        };
        let changed = {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views;
            let active = views.active;
            let Some(view) = views.views.iter_mut().find(|view| view.id == active) else {
                return;
            };
            if view.settings == value {
                return;
            }
            let first = view.settings.is_null();
            view.settings = value;
            !first
        };
        if changed {
            self._app_state.mark_dirty();
            self.push_views_in_slint();
        }
    }

    /// The split shown or not: the view remembers it.
    pub(super) fn remember_view_split(&self, on: bool) {
        let changed = {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views;
            let active = views.active;
            match views.views.iter_mut().find(|view| view.id == active) {
                Some(view) if view.split != on => {
                    view.split = on;
                    true
                }
                _ => false,
            }
        };
        if changed {
            self._app_state.mark_dirty();
            self.push_views_in_slint();
        }
    }

    /// Shows view `id`: its settings into the filters and controls, its
    /// kind and split, its data queried.
    fn select_view(self: &Arc<Self>, id: u32) {
        {
            let mut project = self._app_state.get_project_write();
            if !project.results_views.views.iter().any(|view| view.id == id) {
                return;
            }
            project.results_views.active = id;
        }
        self.run_db_job(|manager| manager.load_active_view());
    }

    /// The active view anew: filters reset, its settings on them, queried.
    fn load_active_view(&self) {
        if self.result_generator.lock().expect("Poisoned").is_none() {
            self.push_views_in_slint();
            return;
        }
        self.reset_filters_to_defaults();
        self.apply_active_view();
        self.show_active_view_mode();
        // The export dialog starts from the view shown (its columns,
        // classes, aggregation) the next time it opens.
        *self.export_populated.lock().expect("Poisoned") = false;
        self.refresh_list();
        self.update_matrix_view();
        self.refresh_charts();
    }

    /// Shows the project's first view of `kind` (one is made if there is
    /// none) - its settings into the filters, nothing queried yet: the
    /// caller narrows them (e.g. a chart's box in the List) and shows it.
    pub(super) fn activate_first_view_of(&self, kind: ResultsViewKind) {
        let made = {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views;
            let active_kind = views
                .views
                .iter()
                .find(|view| view.id == views.active)
                .map(|view| view.kind);
            if active_kind == Some(kind) {
                return;
            }
            match views.views.iter().find(|view| view.kind == kind) {
                Some(view) => {
                    views.active = view.id;
                    false
                }
                None => {
                    let id = views.views.iter().map(|view| view.id).max().unwrap_or(0) + 1;
                    let base = match kind {
                        ResultsViewKind::List => "List",
                        ResultsViewKind::Plate => "Plate",
                        ResultsViewKind::Charts => "Charts",
                        ResultsViewKind::Tracks => "Tracks",
                    };
                    let name = free_name(&views.views, base);
                    views.views.push(ResultsView {
                        id,
                        name,
                        kind,
                        split: false,
                        settings: serde_json::Value::Null,
                    });
                    views.active = id;
                    true
                }
            }
        };
        if made {
            self._app_state.mark_dirty();
        }
        if self.result_generator.lock().expect("Poisoned").is_some() {
            self.reset_filters_to_defaults();
            self.apply_active_view();
        } else {
            self.push_views_in_slint();
        }
    }

    /// The view's layout and kind, as the bar's own switches would set
    /// them: their callbacks bring the rest (split, tracks) along.
    pub(super) fn show_active_view_mode(&self) {
        let Some(view) = self.active_view() else {
            return;
        };
        let split = view.split;
        let mode = rail_mode_of(view.kind);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                state.set_split(split);
                state.invoke_split_changed(split);
                state.set_rail_mode(mode);
                state.invoke_rail_mode_selected(mode);
            }
        })
        .ok();
    }

    /// The filters as a fresh analysis starts them (what `open_database`
    /// sets before the view's settings go on top).
    fn reset_filters_to_defaults(&self) {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let first = |accept: fn(&Column) -> bool| {
            available
                .iter()
                .map(|entry| entry.key.clone())
                .find(accept)
                .unwrap_or_default()
        };
        *self.list_filter.lock().expect("Poisoned") = ListFilter {
            image_rel_path: Vec::new(),
            image_state: ImageState::All,
            object_classes: classes.iter().map(|class| class.global()).collect(),
            columns: DEFAULT_LIST_COLUMNS.to_vec(),
            with_coloc_details: false,
            transpond: false,
            group_by: ListGroupBy::Objects,
            aggregations: vec![Aggregation::Avg],
            children: Vec::new(),
            child_aggregations: vec![Aggregation::Avg],
            parents: Vec::new(),
        };
        let chart_column = first(is_chartable_column);
        *self.chart_filter.lock().expect("Poisoned") = ChartFilter {
            column: chart_column.clone(),
            y_column: chart_column,
            ..ChartFilter::default()
        };
        let plate = self._app_state.get_project().plate.clone();
        *self.matrix_filter.lock().expect("Poisoned") = Some(MatrixFilter {
            column: first(is_matrix_column),
            plate,
            ..MatrixFilter::default()
        });
    }

    /// Every control of List, Plate and Charts as the filters are.
    pub(super) fn push_view_controls_in_slint(&self) {
        let classes = self.classes.lock().expect("Poisoned").clone();
        let available = self.available_columns.lock().expect("Poisoned").clone();
        let list = {
            let filter = self.list_filter.lock().expect("Poisoned");
            (
                filter.object_classes.clone(),
                filter.columns.clone(),
                filter.aggregations.clone(),
                filter.group_by,
                filter.transpond,
                filter.with_coloc_details,
                filter.image_state,
                (
                    filter.children.clone(),
                    filter.child_aggregations.clone(),
                    filter.parents.clone(),
                ),
            )
        };
        let (
            list_classes,
            list_columns,
            aggregations,
            group_by,
            transpond,
            coloc,
            image_state,
            (children, child_aggregations, parents),
        ) = list;
        let class_items: Vec<MultiSelectItem> = class_filter_items(&classes, false)
            .into_iter()
            .zip(&classes)
            .map(|(mut item, class)| {
                item.selected = list_classes.contains(&class.global());
                item
            })
            .collect();
        let class_summary = list_summary(
            class_items.iter().filter(|item| item.selected).count(),
            class_items.len(),
            "Classes",
        );
        // The children's and the parents' columns come from their own groups.
        let plain_columns: Vec<ColumnEntry> = available
            .iter()
            .filter(|entry| !matches!(entry.key, Column::Children(_) | Column::Parent(_)))
            .cloned()
            .collect();
        let column_items_list =
            column_items(&plain_columns, &classes, |key| list_columns.contains(key));
        let class_label = |class: ObjectClass| {
            classes
                .iter()
                .find(|c| c.global() == class)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("class {}", class.to_u32().unwrap_or_default()))
        };
        // The links Relate Objects made, from the results' columns.
        let children_items: Vec<MultiSelectItem> = available
            .iter()
            .filter_map(|entry| match &entry.key {
                Column::Children(c) if c.value == ChildValue::Count => {
                    let link = (c.parent_class, c.child_class);
                    Some(MultiSelectItem {
                        key: link_key(link).into(),
                        value: format!(
                            "{} in {}",
                            class_label(c.child_class),
                            class_label(c.parent_class)
                        )
                        .into(),
                        color: Color::default(),
                        group: "".into(),
                        selected: children.contains(&link),
                    })
                }
                _ => None,
            })
            .collect();
        let parent_items: Vec<MultiSelectItem> = available
            .iter()
            .filter_map(|entry| match &entry.key {
                Column::Parent(p) => Some(MultiSelectItem {
                    key: p.parent_class.to_u32()?.to_string().into(),
                    value: format!("Parent · {}", class_label(p.parent_class)).into(),
                    color: Color::default(),
                    group: "".into(),
                    selected: parents.contains(&p.parent_class),
                }),
                _ => None,
            })
            .collect();
        let children_summary = list_summary(children.len(), children_items.len(), "Children");
        let parent_summary = list_summary(parents.len(), parent_items.len(), "Linked objects");
        let child_aggregation_items = aggregation_items(&child_aggregations);
        let child_aggregate_summary =
            list_summary(child_aggregations.len(), AGGREGATIONS.len(), "Aggregations");
        let has_partners = available
            .iter()
            .any(|entry| matches!(entry.key, Column::ColocCount(_)));
        let columns_summary = list_summary(
            column_items_list
                .iter()
                .filter(|item| item.selected)
                .count(),
            column_items_list.len(),
            "Columns",
        );
        let aggregation_items_list = aggregation_items(&aggregations);
        let aggregate_summary =
            list_summary(aggregations.len(), AGGREGATIONS.len(), "Aggregations");

        let plate = self
            .matrix_filter
            .lock()
            .expect("Poisoned")
            .as_ref()
            .map(|filter| {
                (
                    filter.column.clone(),
                    filter.aggregation.clone(),
                    filter.object_classe,
                    filter.color_schema.clone(),
                    matches!(filter.color_scale, ColorScale::Manual(..)),
                )
            });
        let chart = self.chart_filter.lock().expect("Poisoned").clone();

        let matrix_columns: Vec<ColumnEntry> = available
            .iter()
            .filter(|entry| is_matrix_column(&entry.key))
            .cloned()
            .collect();
        let class_name = |class: ObjectClass| {
            classes
                .iter()
                .find(|c| c.global() == class)
                .map(|c| c.name.clone())
        };
        let plate_ui = plate.map(|(column, aggregation, class, schema, manual)| {
            let items = column_items(&matrix_columns, &classes, |key| *key == column);
            let class_items: Vec<MultiSelectItem> = class_filter_items(&classes, false)
                .into_iter()
                .zip(&classes)
                .map(|(mut item, c)| {
                    item.selected = c.global() == class;
                    item
                })
                .collect();
            let schema_items: Vec<MultiSelectItem> = color_schema_items()
                .into_iter()
                .map(|mut item| {
                    item.selected =
                        color_schema_from_key(item.key.as_str()).as_ref() == Some(&schema);
                    item
                })
                .collect();
            let schema_name = schema_items
                .iter()
                .find(|item| item.selected)
                .map(|item| item.value.clone())
                .unwrap_or_default();
            (
                items,
                column.display_label(&classes),
                aggregation_label(&aggregation),
                is_chartable_column(&column),
                class_items,
                class_name(class).unwrap_or_else(|| "No Class".into()),
                schema_items,
                schema_name,
                color_scale_gradient_slint(&schema),
                manual,
            )
        });
        let chart_items = chart_column_items(&available, &classes, &chart.column);
        let chart_y_items = chart_column_items(&available, &classes, &chart.y_column);
        let chart_class_items: Vec<MultiSelectItem> = class_filter_items(&classes, false)
            .into_iter()
            .zip(&classes)
            .map(|(mut item, c)| {
                item.selected = chart.object_class == Some(c.global());
                item
            })
            .collect();
        let chart_ui = (
            chart.column.display_label(&classes),
            chart.y_column.display_label(&classes),
            chart
                .object_class
                .and_then(class_name)
                .unwrap_or_else(|| "No Class".into()),
        );

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let model = |items: Vec<MultiSelectItem>| ModelRc::from(Rc::new(VecModel::from(items)));
            state.set_list_class_items(model(class_items));
            state.set_list_class_summary(class_summary);
            state.set_list_columns(model(column_items_list));
            state.set_list_columns_summary(columns_summary);
            state.set_list_aggregation_items(model(aggregation_items_list));
            state.set_list_aggregate(aggregate_summary);
            let rows = match (group_by, coloc) {
                (ListGroupBy::Images, _) => ("images", "Images"),
                (ListGroupBy::Objects, true) => ("pairs", "Overlapping pairs"),
                (ListGroupBy::Objects, false) => ("objects", "Objects"),
            };
            state.set_list_group_by_items(model(group_by_items(rows.0, has_partners)));
            state.set_list_group_by_summary(rows.1.into());
            state.set_list_transpond(transpond);
            state.set_list_with_coloc_details(coloc);
            state.set_list_children_items(model(children_items));
            state.set_list_children_summary(children_summary.into());
            state.set_list_child_aggregation_items(model(child_aggregation_items));
            state.set_list_child_aggregate(child_aggregate_summary.into());
            state.set_list_parent_items(model(parent_items));
            state.set_list_parent_summary(parent_summary.into());
            let state_items: Vec<MultiSelectItem> = image_state_items()
                .into_iter()
                .map(|mut item| {
                    item.selected = item.key.as_str() == image_state_key(image_state);
                    item
                })
                .collect();
            let state_summary = state_items
                .iter()
                .find(|item| item.selected)
                .map(|item| item.value.clone())
                .unwrap_or_default();
            state.set_list_image_state_items(model(state_items));
            state.set_list_image_state_summary(state_summary);

            if let Some((
                items,
                column_label,
                aggregate,
                per_object,
                class_items,
                class_label,
                schema_items,
                schema_name,
                stops,
                manual,
            )) = plate_ui
            {
                state.set_matrix_column_items(model(items));
                state.set_matrix_column_summary(column_label.into());
                state.set_matrix_aggregate(aggregate.into());
                state.set_split_values(per_object);
                state.set_matrix_class_items(model(class_items));
                state.set_matrix_class_summary(class_label.into());
                state.set_matrix_color_schema_items(model(schema_items));
                state.set_matrix_color_scale_summary(schema_name);
                state.set_matrix_color_scale_stops(ModelRc::from(Rc::new(VecModel::from(stops))));
                state.set_matrix_scale_is_manual(manual);
            }

            state.set_chart_kind(match chart.kind {
                ChartKind::Histogram => ResultsChartKind2::Histogram,
                ChartKind::Scatter => ResultsChartKind2::Scatter,
                ChartKind::Boxplot => ResultsChartKind2::Boxplot,
                ChartKind::OverTime => ResultsChartKind2::OverTime,
            });
            state.set_chart_column_items(model(chart_items));
            state.set_chart_column_summary(chart_ui.0.into());
            state.set_chart_y_column_items(model(chart_y_items));
            state.set_chart_y_column_summary(chart_ui.1.into());
            state.set_chart_class_items(model(chart_class_items));
            state.set_chart_class_summary(chart_ui.2.into());
            state.set_chart_group(match chart.group_by {
                BoxplotGroup::Image => 1,
                BoxplotGroup::Well => 2,
                BoxplotGroup::WellRow => 3,
                BoxplotGroup::WellColumn => 4,
                _ => 0,
            });
            state.set_chart_bins(chart.bins as i32);
            state.set_chart_log_counts(chart.log_counts);
            state.set_chart_points_index(points_index(chart.max_points));
            state.set_chart_time_lines(time_lines_index(chart.time_lines));
            state.set_chart_time_summary(chart.time_median as i32);
            state.set_chart_time_relative(chart.time_relative);
            state.set_chart_time_tracks(chart.time_tracks);
        })
        .ok();
    }

    /// The note over the results: what the view names but this analysis
    /// lacks ("" none).
    fn push_view_missing(&self, missing: Vec<String>) {
        let text = match missing.len() {
            0 => String::new(),
            _ => format!(
                "Not in these results, left out of the view: {} - the view keeps them for other results.",
                missing.join(", ")
            ),
        };
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_view_missing(text.into());
            }
        })
        .ok();
    }

    /// A new view of `key` ("list", "plate", "charts", "tracks") or a copy
    /// of the one shown ("copy") - shown right away.
    fn new_view(self: &Arc<Self>, key: &str) {
        let active = self.active_view();
        let id = {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views;
            let id = views.views.iter().map(|view| view.id).max().unwrap_or(0) + 1;
            let view = match (key, active) {
                ("copy", Some(active)) => ResultsView {
                    id,
                    name: free_name(&views.views, &format!("{} copy", active.name)),
                    ..active
                },
                (key, _) => {
                    let Some(kind) = kind_of_key(key) else {
                        return;
                    };
                    let base = match kind {
                        ResultsViewKind::List => "List",
                        ResultsViewKind::Plate => "Plate",
                        ResultsViewKind::Charts => "Charts",
                        ResultsViewKind::Tracks => "Tracks",
                    };
                    ResultsView {
                        id,
                        name: free_name(&views.views, base),
                        kind,
                        split: false,
                        settings: serde_json::Value::Null,
                    }
                }
            };
            views.views.push(view);
            id
        };
        self._app_state.mark_dirty();
        self.select_view(id);
    }

    fn duplicate_view(self: &Arc<Self>, id: u32) {
        {
            let mut project = self._app_state.get_project_write();
            project.results_views.active = id;
        }
        self.new_view("copy");
    }

    /// View `id` shows its settings as `kind` (List, Plate, ...) - each
    /// kind's settings stay, for switching back.
    fn show_view_as(self: &Arc<Self>, id: u32, kind: ResultsViewKind) {
        let changed = {
            let mut project = self._app_state.get_project_write();
            match project
                .results_views
                .views
                .iter_mut()
                .find(|view| view.id == id)
            {
                Some(view) if view.kind != kind => {
                    view.kind = kind;
                    true
                }
                _ => false,
            }
        };
        if changed {
            self._app_state.mark_dirty();
            self.select_view(id);
        }
    }

    fn rename_view(&self, id: u32, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        {
            let mut project = self._app_state.get_project_write();
            match project
                .results_views
                .views
                .iter_mut()
                .find(|view| view.id == id)
            {
                Some(view) if view.name != name => view.name = name.to_string(),
                _ => return,
            }
        }
        self._app_state.mark_dirty();
        self.push_views_in_slint();
    }

    /// View `id` as the project was last saved.
    fn revert_view(self: &Arc<Self>, id: u32) {
        let Some(saved) = self._app_state.saved_results_view(id) else {
            return;
        };
        {
            let mut project = self._app_state.get_project_write();
            let Some(view) = project
                .results_views
                .views
                .iter_mut()
                .find(|view| view.id == id)
            else {
                return;
            };
            *view = saved;
        }
        self._app_state.mark_dirty();
        let active = self._app_state.get_project().results_views.active;
        match active == id {
            true => self.select_view(id),
            false => self.push_views_in_slint(),
        }
    }

    /// Moves view `id` one place left (`delta` -1) or right (1).
    fn move_view(&self, id: u32, delta: i32) {
        {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views.views;
            let Some(index) = views.iter().position(|view| view.id == id) else {
                return;
            };
            let target = index as i64 + delta as i64;
            if target < 0 || target >= views.len() as i64 {
                return;
            }
            views.swap(index, target as usize);
        }
        self._app_state.mark_dirty();
        self.push_views_in_slint();
    }

    /// Removes view `id` - not the last one; the one shown goes to its
    /// neighbour.
    fn delete_view(self: &Arc<Self>, id: u32) {
        let next = {
            let mut project = self._app_state.get_project_write();
            let views = &mut project.results_views;
            if views.views.len() <= 1 {
                return;
            }
            let Some(index) = views.views.iter().position(|view| view.id == id) else {
                return;
            };
            views.views.remove(index);
            (views.active == id).then(|| views.views[index.min(views.views.len() - 1)].id)
        };
        self._app_state.mark_dirty();
        match next {
            Some(next) => self.select_view(next),
            None => self.push_views_in_slint(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults are what an empty view stores; a stored view reads back.
    #[test]
    fn view_settings_read_back_and_default() {
        assert!(ViewSettings::read(&serde_json::Value::Null) == ViewSettings::default());
        let mut settings = ViewSettings::default();
        settings.list.columns = Some(vec![
            "area_px".into(),
            "measurement:quality_01/noise".into(),
        ]);
        settings.list.rows_per_image = true;
        settings.plate.color_scale = Some((0.0, 1.0));
        settings.chart.kind = "scatter".into();
        let value = serde_json::to_value(&settings).unwrap();
        let back: ViewSettings = serde_json::from_value(value).unwrap();
        assert!(back == settings);
        // An older view without a section: that section's defaults.
        let partial: ViewSettings =
            serde_json::from_value(serde_json::json!({"list": {"transposed": true}})).unwrap();
        assert!(partial.list.transposed && partial.list.aggregations == vec![Aggregation::Avg]);
    }

    /// Settings the GUI can't read any more break only their section:
    /// unknown fields are ignored, a field of another type resets its
    /// section, the other sections stay.
    #[test]
    fn unreadable_settings_reset_only_their_section() {
        let stored = serde_json::json!({
            "list": {"columns": 5, "transposed": true},
            "plate": {"aggregation": "Median", "aNewSettingFromLater": [1, 2]},
            "chart": "not even an object",
            "somethingElse": {}
        });
        let settings = ViewSettings::read(&stored);
        assert!(
            settings.list == ListView::default(),
            "list unreadable: defaults"
        );
        assert!(
            settings.plate.aggregation == Aggregation::Median,
            "plate kept"
        );
        assert!(settings.chart == ChartView::default());
    }

    #[test]
    fn new_views_get_free_names() {
        let views = evanalyzer_cfg::settings::project_settings::ResultsViews::default().views;
        assert_eq!(free_name(&views, "List"), "List 2");
        assert_eq!(free_name(&views, "QC"), "QC");
    }
}
