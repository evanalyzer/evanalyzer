//! # tracks_all
//!
//! Results › Tracks, Full: every track as a table (filters, sortable) and
//! the selected one's details. The rows come from one summary query; their
//! area curves from their images' tracks, read for the rows in view (an
//! image at a time, a few kept). "Show in image" goes to the split layout.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::tracks::{TRACK_PALETTE, area_curve};
use super::*;
use crate::{MultiSelectItem, TrackAllRow, TrackColumnHead, TrackDetail, TrackFrame};
use evanalyzer_app::global::SpeedUnit;
use evanalyzer_app::results::{
    ImageTracks, ImageTracksFilter, MotionKind, TrackMotion, TrackPoint, TrackSummary,
    TrackSummaryFilter, drift_at,
};
use slint::{Model, ModelNotify, ModelTracker};
use std::cell::RefCell;
use std::collections::VecDeque;

/// Images whose tracks are kept for the curves (~1 MB each).
const CURVE_IMAGES_KEPT: usize = 16;
/// At most this many points of an image's tracks for its curves.
const CURVE_MAX_POINTS: usize = 1_000_000;

/// An image's tracks on a z-plane, by its name.
type CurveKey = (String, u32);

thread_local! {
    /// The table on screen - its rows are told when their curves arrive.
    static TABLE: RefCell<Option<Rc<TracksAllModel>>> = const { RefCell::new(None) };
}

#[derive(Default)]
pub(super) struct TracksAllState {
    rows: Arc<Vec<TrackSummary>>,
    /// The table's rows: indices into `rows`.
    order: Arc<Vec<usize>>,
    /// Index into `rows`.
    selected: Option<usize>,
    sort: i32,
    /// Bumped by every read: a late answer is dropped.
    generation: u64,
    /// The filters' choices: the class ids and image paths after "All".
    class_ids: Vec<u32>,
    image_paths: Vec<String>,
    /// Images' tracks for the curves, the oldest first.
    curves: Arc<Mutex<VecDeque<(CurveKey, Arc<ImageTracks>)>>>,
    loading: HashSet<CurveKey>,
    /// The motion columns shown (indices into `TrackMotion::VALUES`), in
    /// order - `None`: the view's default ones.
    columns: Option<Vec<usize>>,
}

impl TracksAllState {
    /// The motion columns shown.
    pub(super) fn columns(&self) -> Vec<usize> {
        self.columns.clone().unwrap_or_else(default_columns)
    }

    /// Sets the columns from a view's ids ("vcl", ...) - unknown ones
    /// (a later version's) are left out; `None`: the default ones.
    pub(super) fn set_column_ids(&mut self, ids: Option<&[String]>) {
        self.columns = ids.map(|ids| ids.iter().filter_map(|id| motion_index(id)).collect());
    }

    /// The columns as a view stores them: their ids, `None` for the default.
    pub(super) fn column_ids(&self) -> Option<Vec<String>> {
        self.columns.as_ref().map(|columns| {
            columns
                .iter()
                .map(|&i| TrackMotion::VALUES[i].0.to_string())
                .collect()
        })
    }
}

/// The index of Measure Track Motion's value `id` in `TrackMotion::VALUES`.
fn motion_index(id: &str) -> Option<usize> {
    TrackMotion::VALUES.iter().position(|(v, _, _)| *v == id)
}

fn default_columns() -> Vec<usize> {
    evanalyzer_app::results::view_settings::TracksView::DEFAULT_COLUMNS
        .iter()
        .filter_map(|id| motion_index(id))
        .collect()
}

/// Sort key of motion column `i` (0: track, 1: time points).
const MOTION_SORT: i32 = 10;

/// The table's order: by image and track, else descending by time points
/// or by a motion value (`MOTION_SORT` + its index; none last).
fn order_of(rows: &[TrackSummary], sort: i32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let key = |track: &TrackSummary| -> f64 {
        match sort {
            1 => Some(f64::from(track.points)),
            _ => track.motion.get((sort - MOTION_SORT).max(0) as usize),
        }
        .unwrap_or(f64::NEG_INFINITY)
    };
    match sort {
        1 | MOTION_SORT.. => order.sort_by(|&a, &b| key(&rows[b]).total_cmp(&key(&rows[a]))),
        _ => order.sort_by(|&a, &b| {
            (&rows[a].image, rows[a].track_id).cmp(&(&rows[b].image, rows[b].track_id))
        }),
    }
    order
}

/// "412 tracks · 9 864 objects · median 31 time points".
fn summary_of(rows: &[TrackSummary]) -> String {
    if rows.is_empty() {
        return "No tracks".to_string();
    }
    let objects: u64 = rows.iter().map(|t| u64::from(t.points)).sum();
    let mut lengths: Vec<u32> = rows.iter().map(|t| t.points).collect();
    lengths.sort_unstable();
    format!(
        "{} tracks · {objects} objects · median {} time points",
        rows.len(),
        lengths[lengths.len() / 2]
    )
}

/// How the table shows lengths and speeds: µm (and the speed unit asked
/// for) when every track has a pixel size and frame interval, else px.
#[derive(Clone, Copy)]
struct Shown {
    um: bool,
    speed: SpeedUnit,
}

impl Shown {
    fn new(rows: &[TrackSummary], speed: SpeedUnit) -> Self {
        let um = !rows.is_empty()
            && rows
                .iter()
                .all(|t| t.um_per_px.is_some() && t.s_per_frame.is_some());
        Self {
            um,
            speed: if um { speed } else { SpeedUnit::PxPerFrame },
        }
    }

    /// A length in the unit shown - "–" for none (not measured).
    fn length(&self, track: &TrackSummary, px: Option<f64>) -> String {
        match (px, self.um, track.um_per_px) {
            (None, ..) => "–".to_string(),
            (Some(px), true, Some(um)) => format_chart_value(px * um),
            (Some(px), ..) => format_chart_value(px),
        }
    }

    fn speed(&self, track: &TrackSummary) -> String {
        track
            .speed_in(self.speed)
            .map_or_else(|| "–".to_string(), format_chart_value)
    }

    fn length_unit(&self) -> &'static str {
        if self.um { "µm" } else { "px" }
    }

    /// The speed unit's time: "frame", "min" or "s".
    fn time_unit(&self) -> &'static str {
        match self.speed {
            SpeedUnit::PxPerFrame => "frame",
            SpeedUnit::UmPerMin => "min",
            SpeedUnit::UmPerS => "s",
        }
    }

    /// Motion column `i`'s head: its name and unit ("VCL µm/min").
    fn column_head(&self, i: usize) -> String {
        let (_, name, kind) = TrackMotion::VALUES[i];
        let unit = match kind {
            MotionKind::Length => self.length_unit().to_string(),
            MotionKind::Speed => self.speed_unit().to_string(),
            MotionKind::Ratio => String::new(),
            MotionKind::Angle => "°".to_string(),
            MotionKind::AngularSpeed => format!("°/{}", self.time_unit()),
            MotionKind::Frequency => format!("/{}", self.time_unit()),
            // In px² per frame unless a µm speed unit is picked
            // (`TrackMotion::in_units`), even with the pixel size known.
            MotionKind::Diffusion => match self.speed {
                SpeedUnit::PxPerFrame => "px²/frame".to_string(),
                _ => format!("µm²/{}", self.time_unit()),
            },
        };
        match unit.is_empty() {
            true => name.to_string(),
            false => format!("{name} {unit}"),
        }
    }

    /// Motion column `i` of `track` in the unit shown - "–" for none.
    fn column_cell(&self, i: usize, track: &TrackSummary) -> String {
        let value = match TrackMotion::VALUES[i].2 {
            MotionKind::Length => return self.length(track, track.motion.get(i)),
            _ => track
                .motion
                .in_units(i, self.speed, track.um_per_px, track.s_per_frame),
        };
        let Some(value) = value else {
            return "–".to_string();
        };
        match (TrackMotion::VALUES[i].0, TrackMotion::VALUES[i].2) {
            ("turning", _) => format!("{value:+.1}"),
            (_, MotionKind::Angle) => format!("{value:.1}"),
            (_, MotionKind::Ratio) => format!("{value:.2}"),
            _ => format_chart_value(value),
        }
    }

    fn speed_unit(&self) -> &'static str {
        match self.speed {
            SpeedUnit::PxPerFrame => "px/frame",
            SpeedUnit::UmPerMin => "µm/min",
            SpeedUnit::UmPerS => "µm/s",
        }
    }
}

/// A track's positions from its first, minus the image's stored drift (as
/// Measure Track Motion measured them), px.
fn path_from_start(points: &[TrackPoint], drift: &[(u32, f64, f64)]) -> Vec<(f64, f64)> {
    let corrected: Vec<(f64, f64)> = points
        .iter()
        .map(|p| {
            let (dx, dy) = drift_at(drift, p.t);
            (f64::from(p.x) - dx, f64::from(p.y) - dy)
        })
        .collect();
    let Some(&(x0, y0)) = corrected.first() else {
        return Vec::new();
    };
    corrected.iter().map(|(x, y)| (x - x0, y - y0)).collect()
}

/// `path` (from [`path_from_start`]) in 0 0 100 100 with its start at
/// 50 50, the farthest it got (in x or y) at the edge - and that distance,
/// px. Empty for none.
fn path_commands(path: &[(f64, f64)]) -> (String, f64) {
    let extent = path
        .iter()
        .map(|(x, y)| x.abs().max(y.abs()))
        .fold(0.0, f64::max);
    if path.is_empty() {
        return (String::new(), 0.0);
    }
    let scale = if extent > 0.0 { 48.0 / extent } else { 0.0 };
    let commands = path
        .iter()
        .enumerate()
        .map(|(k, (x, y))| {
            format!(
                "{}{:.2} {:.2} ",
                if k == 0 { "M" } else { "L" },
                50.0 + x * scale,
                50.0 + y * scale
            )
        })
        .collect();
    (commands, extent)
}

/// The MSD points (lag, MSD) log-log in 0 0 100 60, and the slopes 1 and 2
/// through the first point (as two lines) - empty for fewer than two.
fn msd_commands(curve: &[(usize, f64)]) -> (String, String) {
    if curve.len() < 2 {
        return (String::new(), String::new());
    }
    let xs: Vec<f64> = curve.iter().map(|(lag, _)| (*lag as f64).ln()).collect();
    let ys: Vec<f64> = curve.iter().map(|(_, msd)| msd.ln()).collect();
    let (x0, x1) = (xs[0], xs[xs.len() - 1]);
    let slope = |a: f64, x: f64| ys[0] + a * (x - x0);
    let (lo, hi) = ys
        .iter()
        .copied()
        .chain([slope(1.0, x1), slope(2.0, x1)])
        .fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
    let px = |x: f64| (x - x0) / (x1 - x0).max(1e-12) * 100.0;
    let py = |y: f64| 58.0 - (y - lo) / (hi - lo).max(1e-12) * 56.0;
    let line = xs
        .iter()
        .zip(&ys)
        .enumerate()
        .map(|(k, (x, y))| {
            format!(
                "{}{:.2} {:.2} ",
                if k == 0 { "M" } else { "L" },
                px(*x),
                py(*y)
            )
        })
        .collect();
    let refs = [1.0, 2.0]
        .iter()
        .map(|a| {
            format!(
                "M{:.2} {:.2} L{:.2} {:.2} ",
                px(x0),
                py(ys[0]),
                px(x1),
                py(slope(*a, x1))
            )
        })
        .collect();
    (line, refs)
}

/// `value` and its unit - "–" alone for none.
fn with_unit(value: String, unit: &str) -> String {
    if value == "–" {
        value
    } else {
        format!("{value} {unit}")
    }
}

/// The table: rows made when shown - there can be many thousands.
struct TracksAllModel {
    rows: Arc<Vec<TrackSummary>>,
    order: Arc<Vec<usize>>,
    curves: Arc<Mutex<VecDeque<(CurveKey, Arc<ImageTracks>)>>>,
    shown: Shown,
    /// The motion columns (indices into `TrackMotion::VALUES`).
    columns: Vec<usize>,
    notify: ModelNotify,
}

impl TracksAllModel {
    fn curve(&self, track: &TrackSummary) -> String {
        let curves = self.curves.lock().expect("Poisoned");
        let key = (track.image.clone(), track.z);
        curves
            .iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, tracks)| {
                let i = tracks
                    .tracks
                    .binary_search_by_key(&track.track_id, |t| t.track_id)
                    .ok()?;
                Some(area_curve(&tracks.tracks[i], track.first_t, track.last_t))
            })
            .unwrap_or_default()
    }
}

impl Model for TracksAllModel {
    type Data = TrackAllRow;

    fn row_count(&self) -> usize {
        self.order.len()
    }

    fn row_data(&self, row: usize) -> Option<TrackAllRow> {
        let track = &self.rows[*self.order.get(row)?];
        Some(TrackAllRow {
            id: format!("#{}", track.track_id).into(),
            color: bg_color_to_slint(
                TRACK_PALETTE[(track.track_id % TRACK_PALETTE.len() as u64) as usize],
            ),
            class_name: track.class_name.clone().into(),
            image: track.image.clone().into(),
            span: format!("{}–{}", track.first_t, track.last_t).into(),
            points: track.points.to_string().into(),
            cells: ModelRc::from(Rc::new(VecModel::from(
                self.columns
                    .iter()
                    .map(|&i| slint::SharedString::from(self.shown.column_cell(i, track)))
                    .collect::<Vec<_>>(),
            ))),
            area: track
                .values
                .first()
                .copied()
                .flatten()
                .map_or_else(|| "–".to_string(), format_chart_value)
                .into(),
            curve: self.curve(track).into(),
        })
    }

    fn model_tracker(&self) -> &dyn ModelTracker {
        &self.notify
    }
}

impl ResultsStateController {
    pub(super) fn attach_tracks_all_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_tracks_all_filter_changed(move || manager.tracks_all_load());
        let manager = self.clone();
        state.on_tracks_all_sort_changed(move |sort| {
            manager.tracks_all.lock().expect("Poisoned").sort = sort;
            manager.tracks_all_push_table();
        });
        let manager = self.clone();
        state.on_tracks_all_row_clicked(move |row| manager.tracks_all_select(row));
        let manager = self.clone();
        state.on_tracks_all_rows_shown(move |first, last| {
            manager.tracks_all_read_curves(first.max(0) as usize, last.max(0) as usize)
        });
        let manager = self.clone();
        state.on_track_show_in_image(move || manager.tracks_all_show_in_image());
        let manager = self.clone();
        state.on_tracks_all_column_picked(move |id, on| {
            manager.tracks_all_pick_column(&id, on);
        });
    }

    /// A motion column ticked or not: shown in the order of
    /// `TrackMotion::VALUES`, and the view remembers it.
    fn tracks_all_pick_column(&self, id: &str, on: bool) {
        let Some(index) = motion_index(id) else {
            return;
        };
        {
            let mut all = self.tracks_all.lock().expect("Poisoned");
            let mut columns = all.columns();
            columns.retain(|&i| i != index);
            if on {
                columns.push(index);
                columns.sort_unstable();
            }
            all.columns = Some(columns);
        }
        self.tracks_all_push_table();
        self.remember_view();
    }

    /// Full shown: the filters' choices, then the tracks.
    pub(super) fn tracks_all_show(self: &Arc<Self>) {
        let classes: Vec<(u32, String)> = self
            .classes
            .lock()
            .expect("Poisoned")
            .iter()
            .map(|class| (class.id, class.name.clone()))
            .collect();
        let images: Vec<(String, String)> = self
            .images
            .lock()
            .expect("Poisoned")
            .iter()
            .map(|image| {
                (
                    image.name.clone(),
                    image.rel_path.to_string_lossy().into_owned(),
                )
            })
            .collect();
        {
            let mut state = self.tracks_all.lock().expect("Poisoned");
            state.class_ids = classes.iter().map(|(id, _)| *id).collect();
            state.image_paths = images.iter().map(|(_, rel)| rel.clone()).collect();
        }
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ResultsState>();
            let names = |all: &str, names: Vec<String>| -> ModelRc<slint::SharedString> {
                let items: Vec<slint::SharedString> = std::iter::once(all.to_string())
                    .chain(names)
                    .map(Into::into)
                    .collect();
                ModelRc::from(Rc::new(VecModel::from(items)))
            };
            state.set_tracks_all_classes(names(
                "All classes",
                classes.into_iter().map(|(_, name)| name).collect(),
            ));
            state.set_tracks_all_images(names(
                "All images",
                images.into_iter().map(|(name, _)| name).collect(),
            ));
        }
        self.tracks_all_load();
    }

    /// The tracks the filters ask for (on the database thread).
    fn tracks_all_load(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ResultsState>();
        let (class_ids, image_paths) = {
            let all = self.tracks_all.lock().expect("Poisoned");
            (all.class_ids.clone(), all.image_paths.clone())
        };
        let pick = |index: i32| (index > 0).then(|| index as usize - 1);
        let filter = TrackSummaryFilter {
            columns: vec![Column::AreaSizePx],
            statistics: vec![evanalyzer_app::results::TimeCourseStat::Mean],
            object_classes: pick(state.get_tracks_all_class())
                .and_then(|i| class_ids.get(i))
                .map(|id| vec![ObjectClass::Global(*id)]),
            images: pick(state.get_tracks_all_image())
                .and_then(|i| image_paths.get(i))
                .map(|rel| vec![rel.clone()]),
            min_points: state
                .get_tracks_all_min()
                .trim()
                .parse()
                .unwrap_or(1)
                .max(1),
            from_start: state.get_tracks_all_from_start(),
            no_edge: state.get_tracks_all_no_edge(),
            ..TrackSummaryFilter::default()
        };
        let generation = {
            let mut all = self.tracks_all.lock().expect("Poisoned");
            all.generation += 1;
            all.generation
        };
        self.run_db_job(move |manager| {
            let rows = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.track_summary(&filter),
                }
            };
            let rows = rows.unwrap_or_else(|err| {
                error!("Could not read the tracks: {err}");
                Vec::new()
            });
            {
                let mut all = manager.tracks_all.lock().expect("Poisoned");
                if all.generation != generation {
                    return;
                }
                all.rows = Arc::new(rows);
                all.selected = None;
            }
            manager.tracks_all_push_table();
            manager.tracks_all_push_detail();
        });
    }

    /// The table in its order, how many, the units.
    pub(super) fn tracks_all_push_table(&self) {
        let speed = self._app_state.load_app_settings().results_split.speed_unit;
        let (rows, order, curves, row, columns) = {
            let mut all = self.tracks_all.lock().expect("Poisoned");
            let order = Arc::new(order_of(&all.rows, all.sort));
            all.order = order.clone();
            let row = all
                .selected
                .and_then(|i| order.iter().position(|&o| o == i));
            (
                all.rows.clone(),
                order,
                all.curves.clone(),
                row,
                all.columns(),
            )
        };
        let shown = Shown::new(&rows, speed);
        let summary = summary_of(&rows);
        let heads: Vec<TrackColumnHead> = columns
            .iter()
            .map(|&i| TrackColumnHead {
                label: shown.column_head(i).into(),
                key: MOTION_SORT + i as i32,
            })
            .collect();
        let column_items: Vec<MultiSelectItem> = TrackMotion::VALUES
            .iter()
            .enumerate()
            .map(|(i, (id, name, _))| MultiSelectItem {
                key: (*id).into(),
                value: (*name).into(),
                color: slint::Color::default(),
                group: "".into(),
                selected: columns.contains(&i),
            })
            .collect();
        let columns_summary = format!("{} of {}", columns.len(), TrackMotion::VALUES.len());
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_tracks_all_summary(summary.into());
            state.set_tracks_length_unit(shown.length_unit().into());
            state.set_tracks_all_speed_unit(shown.speed_unit().into());
            state.set_tracks_all_um(shown.um);
            state.set_tracks_all_heads(ModelRc::from(Rc::new(VecModel::from(heads))));
            state.set_tracks_all_column_items(ModelRc::from(Rc::new(VecModel::from(column_items))));
            state.set_tracks_all_columns_summary(columns_summary.into());
            let model = Rc::new(TracksAllModel {
                rows,
                order,
                curves,
                shown,
                columns,
                notify: ModelNotify::default(),
            });
            TABLE.with(|table| *table.borrow_mut() = Some(model.clone()));
            state.set_tracks_all_rows(ModelRc::from(model));
            state.set_tracks_all_selected(row.map_or(-1, |r| r as i32));
        })
        .ok();
    }

    /// Rows `first..=last` in view: their images' tracks, for the curves.
    fn tracks_all_read_curves(self: &Arc<Self>, first: usize, last: usize) {
        let wanted: Vec<CurveKey> = {
            let all = self.tracks_all.lock().expect("Poisoned");
            let curves = all.curves.lock().expect("Poisoned");
            let mut wanted: Vec<CurveKey> = Vec::new();
            for &i in all
                .order
                .iter()
                .skip(first)
                .take(last.saturating_sub(first) + 1)
            {
                let key = (all.rows[i].image.clone(), all.rows[i].z);
                if !wanted.contains(&key)
                    && !all.loading.contains(&key)
                    && !curves.iter().any(|(k, _)| *k == key)
                {
                    wanted.push(key);
                }
            }
            wanted
        };
        for key in wanted {
            self.tracks_all_read_image(key);
        }
    }

    fn tracks_all_read_image(self: &Arc<Self>, key: CurveKey) {
        let rel_path = self
            .images
            .lock()
            .expect("Poisoned")
            .iter()
            .find(|image| image.name == key.0)
            .map(|image| image.rel_path.to_string_lossy().into_owned());
        let Some(rel_path) = rel_path else {
            return;
        };
        self.tracks_all
            .lock()
            .expect("Poisoned")
            .loading
            .insert(key.clone());
        let filter = ImageTracksFilter {
            image_rel_path: rel_path,
            z_stack: key.1,
            max_points: CURVE_MAX_POINTS,
        };
        self.run_db_job(move |manager| {
            let tracks = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.image_tracks(&filter),
                }
            };
            let tracks = Arc::new(tracks.unwrap_or_default());
            {
                let mut all = manager.tracks_all.lock().expect("Poisoned");
                all.loading.remove(&key);
                let mut curves = all.curves.lock().expect("Poisoned");
                curves.push_back((key.clone(), tracks));
                while curves.len() > CURVE_IMAGES_KEPT {
                    curves.pop_front();
                }
            }
            // Its rows show their curves; the selected one its details.
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                TABLE.with(|table| {
                    if let Some(model) = table.borrow().as_ref() {
                        for (row, &i) in model.order.iter().enumerate() {
                            let track = &model.rows[i];
                            if track.image == key.0 && track.z == key.1 {
                                model.notify.row_changed(row);
                            }
                        }
                    }
                });
            })
            .ok();
            manager.tracks_all_push_detail();
        });
    }

    /// Table row `row` selected: its details (its image's tracks read for
    /// its time points, when not yet).
    fn tracks_all_select(self: &Arc<Self>, row: i32) {
        let key = {
            let mut all = self.tracks_all.lock().expect("Poisoned");
            let Some(&i) = all.order.get(row.max(0) as usize) else {
                return;
            };
            all.selected = Some(i);
            let track = &all.rows[i];
            (track.image.clone(), track.z)
        };
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ResultsState>().set_tracks_all_selected(row);
        }
        let loaded = {
            let all = self.tracks_all.lock().expect("Poisoned");
            let curves = all.curves.lock().expect("Poisoned");
            curves.iter().any(|(k, _)| *k == key) || all.loading.contains(&key)
        };
        if !loaded {
            self.tracks_all_read_image(key);
        }
        self.tracks_all_push_detail();
    }

    /// The selected track's numbers, area over time and time points.
    pub(super) fn tracks_all_push_detail(&self) {
        let speed = self._app_state.load_app_settings().results_split.speed_unit;
        let (track, points, drift, shown) = {
            let all = self.tracks_all.lock().expect("Poisoned");
            let shown = Shown::new(&all.rows, speed);
            let Some(track) = all.selected.and_then(|i| all.rows.get(i)).cloned() else {
                drop(all);
                let ui_weak = self.ui.clone();
                crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let state = ui.global::<ResultsState>();
                        state.set_track_detail(TrackDetail::default());
                        state.set_track_frames(ModelRc::default());
                    }
                })
                .ok();
                return;
            };
            let curves = all.curves.lock().expect("Poisoned");
            let (points, drift) = curves
                .iter()
                .find(|(k, _)| *k == (track.image.clone(), track.z))
                .and_then(|(_, tracks)| {
                    let i = tracks
                        .tracks
                        .binary_search_by_key(&track.track_id, |t| t.track_id)
                        .ok()?;
                    Some((tracks.tracks[i].points.clone(), tracks.drift.clone()))
                })
                .unwrap_or_default();
            (track, points, drift, shown)
        };
        // How it moved: its path from its start and the MSD points, both
        // without the drift - as the step measured them.
        let path = path_from_start(&points, &drift);
        let (path_commands, extent) = path_commands(&path);
        let msd = evanalyzer_app::results::msd_curve(
            &points
                .iter()
                .zip(&path)
                .map(|(p, (x, y))| evanalyzer_app::results::MotionPoint {
                    t: p.t as i32,
                    x: *x,
                    y: *y,
                })
                .collect::<Vec<_>>(),
        );
        let (msd_line, msd_ref) = msd_commands(&msd);
        let msd_label = match (msd.first(), msd.last()) {
            (Some(first), Some(last)) if msd.len() >= 2 => format!(
                "α {} · lags {}–{}",
                track
                    .motion
                    .msd_alpha
                    .map_or_else(|| "–".to_string(), |a| format!("{a:.2}")),
                first.0,
                last.0
            ),
            _ => String::new(),
        };
        let (lo, hi) = points.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.area_px), hi.max(p.area_px))
        });
        let span = track.last_t.saturating_sub(track.first_t).max(1) as f32;
        let range = (hi - lo).max(1e-6);
        let curve: String = points
            .iter()
            .enumerate()
            .map(|(k, p)| {
                format!(
                    "{}{:.1} {:.1} ",
                    if k == 0 { "M" } else { "L" },
                    (p.t - track.first_t) as f32 / span * 100.0,
                    38.0 - (p.area_px - lo) / range * 36.0
                )
            })
            .collect();
        let detail = TrackDetail {
            active: true,
            id: format!("#{}", track.track_id).into(),
            class_name: track.class_name.clone().into(),
            image: track.image.clone().into(),
            span: format!("{}–{} ({})", track.first_t, track.last_t, track.points).into(),
            moved: with_unit(
                shown.length(&track, track.motion.displacement),
                shown.length_unit(),
            )
            .into(),
            speed: with_unit(shown.speed(&track), shown.speed_unit()).into(),
            straightness: track
                .straightness()
                .map_or_else(|| "–".to_string(), |s| format!("{s:.2}"))
                .into(),
            turning: track
                .motion
                .turning
                .map_or_else(|| "–".to_string(), |a| format!("{a:+.1}° / time point"))
                .into(),
            max_distance: with_unit(
                shown.length(&track, track.motion.max_distance),
                shown.length_unit(),
            )
            .into(),
            msd_alpha: track
                .motion
                .msd_alpha
                .map_or_else(|| "–".to_string(), |a| format!("{a:.2}"))
                .into(),
            path: path_commands.into(),
            path_scale: if extent > 0.0 {
                format!(
                    "± {} {}",
                    shown.length(&track, Some(extent)),
                    shown.length_unit()
                )
            } else {
                String::new()
            }
            .into(),
            msd: msd_line.into(),
            msd_ref: msd_ref.into(),
            msd_label: msd_label.into(),
            curve: curve.into(),
            area_min: if points.is_empty() {
                String::new()
            } else {
                format_chart_value(f64::from(lo))
            }
            .into(),
            area_max: if points.is_empty() {
                String::new()
            } else {
                format_chart_value(f64::from(hi))
            }
            .into(),
            t_first: track.first_t.to_string().into(),
            t_last: track.last_t.to_string().into(),
        };
        let frames: Vec<TrackFrame> = points
            .iter()
            .map(|p| TrackFrame {
                t: p.t.to_string().into(),
                x: format!("{:.1}", p.x).into(),
                y: format!("{:.1}", p.y).into(),
                area: format_chart_value(f64::from(p.area_px)).into(),
            })
            .collect();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                state.set_track_detail(detail);
                state.set_track_frames(ModelRc::from(Rc::new(VecModel::from(frames))));
            }
        })
        .ok();
    }

    /// "Show in image": the split layout, at the selected track.
    fn tracks_all_show_in_image(self: &Arc<Self>) {
        let target = {
            let all = self.tracks_all.lock().expect("Poisoned");
            all.selected
                .and_then(|i| all.rows.get(i))
                .map(|t| (t.image.clone(), t.track_id))
        };
        let Some((image, track_id)) = target else {
            return;
        };
        let rel_path = self
            .images
            .lock()
            .expect("Poisoned")
            .iter()
            .find(|i| i.name == image)
            .map(|i| i.rel_path.to_string_lossy().into_owned());
        let Some(rel_path) = rel_path else {
            return;
        };
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ResultsState>();
            state.set_split(true);
            self.remember_view_split(true);
        }
        self.tracks_show_track(rel_path, track_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSD D's header names the unit its values are in: px²/frame with
    /// px/frame picked - also when the pixel size is known (lengths are in
    /// µm then, MSD D isn't converted) - else µm² per the speed's time.
    #[test]
    fn the_msd_header_names_the_unit_of_its_values() {
        let msd = TrackMotion::VALUES
            .iter()
            .position(|(id, ..)| *id == "msd_d")
            .unwrap();
        let head = |um, speed| Shown { um, speed }.column_head(msd);
        assert_eq!(head(true, SpeedUnit::PxPerFrame), "MSD D px²/frame");
        assert_eq!(head(false, SpeedUnit::PxPerFrame), "MSD D px²/frame");
        assert_eq!(head(true, SpeedUnit::UmPerMin), "MSD D µm²/min");
        assert_eq!(head(true, SpeedUnit::UmPerS), "MSD D µm²/s");

        // The values: px²/frame stay as measured, µm²/min convert.
        let motion = TrackMotion {
            msd_d: Some(2.0),
            ..Default::default()
        };
        assert_eq!(
            motion.in_units(msd, SpeedUnit::PxPerFrame, Some(0.1), Some(30.0)),
            Some(2.0)
        );
        let per_min = motion
            .in_units(msd, SpeedUnit::UmPerMin, Some(0.1), Some(30.0))
            .unwrap();
        // 2 px²/frame · 0.01 µm²/px² · 2 frames/min = 0.04 µm²/min.
        assert!((per_min - 0.04).abs() < 1e-12, "{per_min}");
    }

    /// A track with Measure Track Motion's path length and displacement,
    /// its VCL and LIN from them.
    fn track(id: u64, image: &str, points: u32, path: f64, moved: f64) -> TrackSummary {
        let frames = f64::from(points.saturating_sub(1));
        let mut motion = TrackMotion::default();
        motion.path_length = Some(path);
        motion.displacement = Some(moved);
        motion.vcl = (frames > 0.0).then(|| path / frames);
        motion.lin = (path > 0.0).then(|| moved / path);
        TrackSummary {
            image: image.into(),
            track_id: id,
            first_t: 0,
            last_t: points.saturating_sub(1),
            points,
            motion,
            ..TrackSummary::default()
        }
    }

    /// By image and track; by time points or a motion value (most first,
    /// none last).
    #[test]
    fn the_table_is_sorted_as_asked() {
        let rows = [
            track(2, "b.tif", 3, 10.0, 5.0),
            track(1, "b.tif", 5, 8.0, 8.0),
            track(9, "a.tif", 1, 0.0, 0.0),
        ];
        assert_eq!(order_of(&rows, 0), [2, 1, 0]);
        assert_eq!(order_of(&rows, 1), [1, 0, 2]);
        // Moved, speed (VCL), straightness (LIN): `MOTION_SORT` + index.
        let key = |id: &str| MOTION_SORT + motion_index(id).unwrap() as i32;
        assert_eq!(order_of(&rows, key("displacement")), [1, 0, 2]);
        assert_eq!(order_of(&rows, key("vcl")), [0, 1, 2]);
        assert_eq!(order_of(&rows, key("lin")), [1, 0, 2]);
    }

    #[test]
    fn the_summary_counts_tracks_objects_and_the_median_length() {
        let rows = [
            track(1, "a", 3, 0.0, 0.0),
            track(2, "a", 5, 0.0, 0.0),
            track(3, "a", 9, 0.0, 0.0),
        ];
        assert_eq!(
            summary_of(&rows),
            "3 tracks · 17 objects · median 5 time points"
        );
        assert_eq!(summary_of(&[]), "No tracks");
    }

    /// The path from its start takes the stored drift away; drawn with its
    /// start in the middle, as far as it got at the edge.
    #[test]
    fn the_path_from_its_start_is_without_the_drift() {
        let points: Vec<TrackPoint> = [(0, 10.0, 10.0), (1, 14.0, 10.0), (2, 18.0, 13.0)]
            .iter()
            .map(|&(t, x, y)| TrackPoint {
                t,
                x,
                y,
                area_px: 1.0,
            })
            .collect();
        let drift = [(0, 0.0, 0.0), (1, 2.0, 0.0), (2, 4.0, 0.0)];
        let path = path_from_start(&points, &drift);
        assert_eq!(path, [(0.0, 0.0), (2.0, 0.0), (4.0, 3.0)]);
        let (commands, extent) = path_commands(&path);
        assert_eq!(extent, 4.0);
        assert_eq!(commands, "M50.00 50.00 L74.00 50.00 L98.00 86.00 ");
        assert_eq!(path_commands(&[]), (String::new(), 0.0));
    }

    /// The MSD points log-log, with the slopes 1 and 2 through the first:
    /// a directed track (MSD ∝ lag²) lies on the slope 2.
    #[test]
    fn the_msd_plot_shows_the_points_and_the_slopes() {
        let directed: Vec<(usize, f64)> = [1usize, 2, 4, 8]
            .iter()
            .map(|&l| (l, (l * l) as f64))
            .collect();
        let (line, refs) = msd_commands(&directed);
        assert!(
            line.starts_with("M0.00 58.00") && line.ends_with("L100.00 2.00 "),
            "{line}"
        );
        assert!(
            refs.contains("L100.00 2.00"),
            "slope 2 ends where the points do: {refs}"
        );
        assert_eq!(msd_commands(&directed[..1]), (String::new(), String::new()));
    }

    /// µm and the speed unit only when every track can show them.
    #[test]
    fn lengths_are_in_um_only_when_every_track_can_be() {
        let mut a = track(1, "a", 3, 10.0, 4.0);
        a.um_per_px = Some(0.5);
        a.s_per_frame = Some(60.0);
        let b = track(2, "a", 3, 10.0, 4.0);
        let shown = Shown::new(&[a.clone()], SpeedUnit::UmPerMin);
        assert_eq!(
            (shown.length_unit(), shown.length(&a, Some(4.0))),
            ("µm", "2".to_string())
        );
        assert_eq!(shown.length(&a, None), "–");
        assert_eq!(shown.speed(&a), "2.50");
        let mixed = Shown::new(&[a, b], SpeedUnit::UmPerMin);
        assert_eq!(
            (mixed.length_unit(), mixed.speed_unit()),
            ("px", "px/frame")
        );
    }
}
