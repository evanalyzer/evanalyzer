//! Results › Tracks, Full: every track as a table (filters, sortable) and
//! the selected one's details. The rows come from one summary query; their
//! area curves from their images' tracks, read for the rows in view (an
//! image at a time, a few kept). "Show in image" goes to the split layout.

use super::tracks::{TRACK_PALETTE, area_curve};
use super::*;
use crate::{TrackAllRow, TrackDetail, TrackFrame};
use evanalyzer_app::global::SpeedUnit;
use evanalyzer_app::results::{ImageTracks, ImageTracksFilter, TrackSummary, TrackSummaryFilter};
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
}

/// The table's order: by image and track, else descending by time points,
/// distance moved, speed or straightness (none last).
fn order_of(rows: &[TrackSummary], sort: i32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let key = |track: &TrackSummary| -> f64 {
        match sort {
            1 => f64::from(track.points),
            2 => track.displacement_px,
            3 => track.speed_px_per_frame().unwrap_or(f64::NEG_INFINITY),
            _ => track.straightness().unwrap_or(f64::NEG_INFINITY),
        }
    };
    match sort {
        1..=4 => order.sort_by(|&a, &b| key(&rows[b]).total_cmp(&key(&rows[a]))),
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

    fn length(&self, track: &TrackSummary, px: f64) -> String {
        match (self.um, track.um_per_px) {
            (true, Some(um)) => format_chart_value(px * um),
            _ => format_chart_value(px),
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

    fn speed_unit(&self) -> &'static str {
        match self.speed {
            SpeedUnit::PxPerFrame => "px/frame",
            SpeedUnit::UmPerMin => "µm/min",
            SpeedUnit::UmPerS => "µm/s",
        }
    }
}

/// The table: rows made when shown - there can be many thousands.
struct TracksAllModel {
    rows: Arc<Vec<TrackSummary>>,
    order: Arc<Vec<usize>>,
    curves: Arc<Mutex<VecDeque<(CurveKey, Arc<ImageTracks>)>>>,
    shown: Shown,
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
            moved: self.shown.length(track, track.displacement_px).into(),
            speed: self.shown.speed(track).into(),
            straightness: track
                .straightness()
                .map_or_else(|| "–".to_string(), |s| format!("{s:.2}"))
                .into(),
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
        let (rows, order, curves, row) = {
            let mut all = self.tracks_all.lock().expect("Poisoned");
            let order = Arc::new(order_of(&all.rows, all.sort));
            all.order = order.clone();
            let row = all
                .selected
                .and_then(|i| order.iter().position(|&o| o == i));
            (all.rows.clone(), order, all.curves.clone(), row)
        };
        let shown = Shown::new(&rows, speed);
        let summary = summary_of(&rows);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_tracks_all_summary(summary.into());
            state.set_tracks_length_unit(shown.length_unit().into());
            state.set_tracks_speed_unit(shown.speed_unit().into());
            state.set_tracks_um(shown.um);
            let model = Rc::new(TracksAllModel {
                rows,
                order,
                curves,
                shown,
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
        let (track, points, shown) = {
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
            let points = curves
                .iter()
                .find(|(k, _)| *k == (track.image.clone(), track.z))
                .and_then(|(_, tracks)| {
                    let i = tracks
                        .tracks
                        .binary_search_by_key(&track.track_id, |t| t.track_id)
                        .ok()?;
                    Some(tracks.tracks[i].points.clone())
                })
                .unwrap_or_default();
            (track, points, shown)
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
            moved: format!(
                "{} {}",
                shown.length(&track, track.displacement_px),
                shown.length_unit()
            )
            .into(),
            speed: format!("{} {}", shown.speed(&track), shown.speed_unit()).into(),
            straightness: track
                .straightness()
                .map_or_else(|| "–".to_string(), |s| format!("{s:.2}"))
                .into(),
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
            self._app_state
                .update_app_settings(|s| s.results_split.on = true);
        }
        self.tracks_show_track(rel_path, track_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(id: u64, image: &str, points: u32, path: f64, moved: f64) -> TrackSummary {
        TrackSummary {
            image: image.into(),
            track_id: id,
            first_t: 0,
            last_t: points.saturating_sub(1),
            points,
            path_px: path,
            displacement_px: moved,
            ..TrackSummary::default()
        }
    }

    /// By image and track; by time points, distance, speed, straightness
    /// (most first, none last).
    #[test]
    fn the_table_is_sorted_as_asked() {
        let rows = [
            track(2, "b.tif", 3, 10.0, 5.0),
            track(1, "b.tif", 5, 8.0, 8.0),
            track(9, "a.tif", 1, 0.0, 0.0),
        ];
        assert_eq!(order_of(&rows, 0), [2, 1, 0]);
        assert_eq!(order_of(&rows, 1), [1, 0, 2]);
        assert_eq!(order_of(&rows, 2), [1, 0, 2]);
        assert_eq!(order_of(&rows, 3), [0, 1, 2]);
        assert_eq!(order_of(&rows, 4), [1, 0, 2]);
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

    /// µm and the speed unit only when every track can show them.
    #[test]
    fn lengths_are_in_um_only_when_every_track_can_be() {
        let mut a = track(1, "a", 3, 10.0, 4.0);
        a.um_per_px = Some(0.5);
        a.s_per_frame = Some(60.0);
        let b = track(2, "a", 3, 10.0, 4.0);
        let shown = Shown::new(&[a.clone()], SpeedUnit::UmPerMin);
        assert_eq!(
            (shown.length_unit(), shown.length(&a, 4.0)),
            ("µm", "2".to_string())
        );
        assert_eq!(shown.speed(&a), "2.50");
        let mixed = Shown::new(&[a, b], SpeedUnit::UmPerMin);
        assert_eq!(
            (mixed.length_unit(), mixed.speed_unit()),
            ("px", "px/frame")
        );
    }
}
