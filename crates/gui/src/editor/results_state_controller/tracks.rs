//! Results › Tracks (with tracked objects): one image's tracks as a table
//! beside the image (the split pane), their paths as trails on it up to
//! the time point shown. A row selects its track - its object on the image
//! (at the track's first time point when it isn't there now) - and a click
//! on an object selects its track.

use super::*;
use crate::editor::results_overlay::TrailsOverlay;
use crate::{TrackCard, TrackRow};
use evanalyzer_app::global::{SpeedUnit, TrailMode};
use evanalyzer_app::results::{ImageTracks, ImageTracksFilter, TrackPath};
use slint::{Model, ModelNotify, ModelTracker};

/// At most this many points of an image's tracks (~16 MB): beyond, a
/// sample of its tracks.
const MAX_POINTS: usize = 1_000_000;

/// Track colors on the image: light enough on dark images.
const TRACK_PALETTE: [u32; 10] = [
    0x5B9BFF, 0xFF8A4C, 0x2DD4BF, 0xC084FC, 0xFACC15, 0xF472B6, 0x38BDF8, 0xA3E635, 0xFB7185,
    0x4ADE80,
];

/// A track's numbers for the table and its card.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct TrackStats {
    first_t: u32,
    last_t: u32,
    points: usize,
    /// From its first point to its last, in pixels.
    displacement_px: f64,
    /// Its path's length over its time points - `None` with one point.
    speed_px_per_frame: Option<f64>,
}

impl TrackStats {
    pub(super) fn of(track: &TrackPath) -> Self {
        let first = track.points.first().copied().unwrap_or_default();
        let last = track.points.last().copied().unwrap_or_default();
        let length: f64 = track
            .points
            .windows(2)
            .map(|pair| f64::from(pair[1].x - pair[0].x).hypot(f64::from(pair[1].y - pair[0].y)))
            .sum();
        let frames = last.t.saturating_sub(first.t);
        Self {
            first_t: first.t,
            last_t: last.t,
            points: track.points.len(),
            displacement_px: f64::from(last.x - first.x).hypot(f64::from(last.y - first.y)),
            speed_px_per_frame: (frames > 0).then(|| length / f64::from(frames)),
        }
    }
}

/// How px and px/frame become the unit shown: µm per pixel and seconds per
/// frame, when both are known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Units {
    um_per_px: Option<f64>,
    s_per_frame: Option<f64>,
    unit: SpeedUnit,
}

impl Units {
    pub(super) fn new(tracks: &ImageTracks, unit: SpeedUnit) -> Self {
        Self {
            um_per_px: tracks
                .pixel_size_nm
                .filter(|v| *v > 0.0)
                .map(|nm| nm / 1000.0),
            s_per_frame: tracks.frame_interval_s.filter(|v| *v > 0.0),
            unit,
        }
    }

    /// µm units can be shown.
    pub(super) fn um_available(&self) -> bool {
        self.um_per_px.is_some() && self.s_per_frame.is_some()
    }

    /// The unit shown: px/frame when µm can't be.
    pub(super) fn shown(&self) -> SpeedUnit {
        match self.um_available() {
            true => self.unit,
            false => SpeedUnit::PxPerFrame,
        }
    }

    pub(super) fn speed_label(&self) -> &'static str {
        match self.shown() {
            SpeedUnit::PxPerFrame => "px/frame",
            SpeedUnit::UmPerMin => "µm/min",
            SpeedUnit::UmPerS => "µm/s",
        }
    }

    pub(super) fn speed(&self, px_per_frame: f64) -> f64 {
        match (self.shown(), self.um_per_px, self.s_per_frame) {
            (SpeedUnit::UmPerMin, Some(um), Some(s)) => px_per_frame * um / s * 60.0,
            (SpeedUnit::UmPerS, Some(um), Some(s)) => px_per_frame * um / s,
            _ => px_per_frame,
        }
    }

    /// A distance: µm with a pixel size (and µm units), else px.
    pub(super) fn distance(&self, px: f64) -> String {
        match (self.shown(), self.um_per_px) {
            (SpeedUnit::PxPerFrame, _) | (_, None) => format!("{} px", format_chart_value(px)),
            (_, Some(um)) => format!("{} µm", format_chart_value(px * um)),
        }
    }
}

/// The table's order: by track, by time points (most first), by speed
/// (fastest first; no speed last).
pub(super) fn table_order(tracks: &ImageTracks, stats: &[TrackStats], sort: i32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tracks.tracks.len()).collect();
    match sort {
        1 => order.sort_by_key(|&i| {
            (
                std::cmp::Reverse(stats[i].points),
                tracks.tracks[i].track_id,
            )
        }),
        2 => order.sort_by(|&a, &b| {
            let speed = |i: usize| stats[i].speed_px_per_frame.unwrap_or(f64::NEG_INFINITY);
            speed(b)
                .total_cmp(&speed(a))
                .then(tracks.tracks[a].track_id.cmp(&tracks.tracks[b].track_id))
        }),
        _ => order.sort_by_key(|&i| tracks.tracks[i].track_id),
    }
    order
}

/// A track's color: its own (by its id) or its class's.
fn track_rgb(track: &TrackPath, by_track: bool, class_colors: &HashMap<i32, u32>) -> u32 {
    match by_track {
        true => TRACK_PALETTE[(track.track_id % TRACK_PALETTE.len() as u64) as usize],
        false => class_colors
            .get(&track.class_id)
            .copied()
            .filter(|color| *color != 0)
            .unwrap_or(0xFFFFFF),
    }
}

/// The area over the track's time points as path commands in 0 0 100 20.
fn area_curve(track: &TrackPath, first_t: u32, last_t: u32) -> String {
    let (lo, hi) = track
        .points
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.area_px), hi.max(p.area_px))
        });
    let span = last_t.saturating_sub(first_t).max(1) as f32;
    let range = (hi - lo).max(1e-6);
    let mut commands = String::new();
    for (i, point) in track.points.iter().enumerate() {
        use std::fmt::Write;
        let x = (point.t - first_t) as f32 / span * 100.0;
        let y = 18.0 - (point.area_px - lo) / range * 16.0;
        let _ = write!(commands, "{}{x:.1} {y:.1} ", if i == 0 { "M" } else { "L" });
    }
    commands
}

/// The tracks table: rows made when shown (an image can have many
/// thousands of tracks).
struct TracksModel {
    tracks: Arc<ImageTracks>,
    stats: Arc<Vec<TrackStats>>,
    order: Arc<Vec<usize>>,
    colors: Vec<u32>,
    units: Units,
    notify: ModelNotify,
}

impl Model for TracksModel {
    type Data = TrackRow;

    fn row_count(&self) -> usize {
        self.order.len()
    }

    fn row_data(&self, row: usize) -> Option<TrackRow> {
        let &i = self.order.get(row)?;
        let (track, stats) = (&self.tracks.tracks[i], self.stats[i]);
        Some(TrackRow {
            id: format!("#{}", track.track_id).into(),
            color: bg_color_to_slint(self.colors[i]),
            span: format!("{}–{}", stats.first_t, stats.last_t).into(),
            frames: stats.points.to_string().into(),
            speed: stats
                .speed_px_per_frame
                .map_or_else(
                    || "–".to_string(),
                    |s| format_chart_value(self.units.speed(s)),
                )
                .into(),
            curve: area_curve(track, stats.first_t, stats.last_t).into(),
        })
    }

    fn model_tracker(&self) -> &dyn ModelTracker {
        &self.notify
    }
}

#[derive(Default)]
pub(super) struct TracksState {
    /// The image whose tracks these are (`image_rel_path`).
    pub(super) image: Option<String>,
    pub(super) tracks: Arc<ImageTracks>,
    stats: Arc<Vec<TrackStats>>,
    /// The table's rows: indices into `tracks.tracks`.
    order: Arc<Vec<usize>>,
    /// Index into `tracks.tracks`.
    pub(super) selected: Option<usize>,
    /// Bumped by every image loaded: a late answer for another is dropped.
    generation: u64,
    /// The table's sort (`ResultsState.tracks-sort`).
    sort: i32,
}

impl ResultsStateController {
    pub(super) fn attach_tracks_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let settings = self._app_state.load_app_settings().results_split;
        state.set_trail_mode(match settings.trail {
            TrailMode::Off => 0,
            TrailMode::Frames => 1,
            TrailMode::Whole => 2,
        });
        state.set_trail_frames(settings.trail_frames.max(1) as i32);
        state.set_tracks_only_selected(settings.only_selected_track);
        state.set_tracks_by_track(settings.color_by_track);
        state.set_tracks_unit(match settings.speed_unit {
            SpeedUnit::PxPerFrame => 0,
            SpeedUnit::UmPerMin => 1,
            SpeedUnit::UmPerS => 2,
        });

        let manager = self.clone();
        state.on_track_row_clicked(move |row| {
            let index = manager
                .tracks
                .lock()
                .expect("Poisoned")
                .order
                .get(row.max(0) as usize)
                .copied();
            if let Some(index) = index {
                manager.tracks_select(index, true);
            }
        });
        let manager = self.clone();
        state.on_tracks_image_step(move |delta| manager.tracks_image_step(delta));
        let manager = self.clone();
        state.on_tracks_sort_changed(move |sort| {
            manager.tracks.lock().expect("Poisoned").sort = sort;
            manager.tracks_push_table();
        });
        let manager = self.clone();
        state.on_tracks_unit_changed(move |unit| {
            manager._app_state.update_app_settings(|s| {
                s.results_split.speed_unit = match unit {
                    1 => SpeedUnit::UmPerMin,
                    2 => SpeedUnit::UmPerS,
                    _ => SpeedUnit::PxPerFrame,
                }
            });
            manager.tracks_push_table();
            manager.tracks_push_card();
        });
        let manager = self.clone();
        state.on_trail_changed(move |mode, frames| {
            manager._app_state.update_app_settings(|s| {
                s.results_split.trail = match mode {
                    0 => TrailMode::Off,
                    2 => TrailMode::Whole,
                    _ => TrailMode::Frames,
                };
                s.results_split.trail_frames = frames.max(1) as u32;
            });
            manager.split_push_overlay();
        });
        let manager = self.clone();
        state.on_tracks_only_selected_changed(move |on| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.only_selected_track = on);
            manager.split_push_overlay();
        });
        let manager = self.clone();
        state.on_tracks_by_track_changed(move |on| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.color_by_track = on);
            manager.tracks_push_table();
            manager.split_push_overlay();
        });
    }

    pub(super) fn in_tracks(&self) -> bool {
        *self.rail_mode.lock().expect("Poisoned") == ResultsRailMode::Tracks
    }

    /// Tracks shown: the split's image (else the first analysed one) and
    /// its tracks.
    pub(super) fn tracks_show(self: &Arc<Self>) {
        let image = self
            .split
            .lock()
            .expect("Poisoned")
            .image
            .clone()
            .or_else(|| {
                self.images
                    .lock()
                    .expect("Poisoned")
                    .first()
                    .map(|image| image.rel_path.to_string_lossy().into_owned())
            });
        if let Some(image) = image {
            self.tracks_show_image(image);
        }
    }

    fn tracks_show_image(self: &Arc<Self>, rel_path: String) {
        self.split_show_image(rel_path.clone(), None, None);
        let loaded =
            self.tracks.lock().expect("Poisoned").image.as_deref() == Some(rel_path.as_str());
        if loaded {
            self.tracks_push_table();
            self.tracks_push_card();
            self.split_push_overlay();
        } else {
            self.tracks_load(rel_path);
        }
    }

    /// ‹ ›: the previous / next analysed image (round).
    fn tracks_image_step(self: &Arc<Self>, delta: i32) {
        let current = self.tracks.lock().expect("Poisoned").image.clone();
        let next = {
            let images = self.images.lock().expect("Poisoned");
            if images.is_empty() {
                return;
            }
            let at = current
                .and_then(|rel| {
                    images
                        .iter()
                        .position(|image| image.rel_path.to_string_lossy() == rel)
                })
                .unwrap_or(0) as i64;
            let n = images.len() as i64;
            images[(at + delta as i64).rem_euclid(n) as usize]
                .rel_path
                .to_string_lossy()
                .into_owned()
        };
        self.tracks_show_image(next);
    }

    /// Reads `rel_path`'s tracks on the plane's z (on the database thread).
    pub(super) fn tracks_load(self: &Arc<Self>, rel_path: String) {
        let generation = {
            let mut tracks = self.tracks.lock().expect("Poisoned");
            tracks.generation += 1;
            *tracks = TracksState {
                image: Some(rel_path.clone()),
                generation: tracks.generation,
                sort: tracks.sort,
                ..TracksState::default()
            };
            tracks.generation
        };
        self.tracks_push_table();
        self.tracks_push_card();
        let filter = ImageTracksFilter {
            image_rel_path: rel_path.clone(),
            z_stack: self.plane_filter.lock().expect("Poisoned").selected_z_stack,
            max_points: MAX_POINTS,
        };
        self.run_db_job(move |manager| {
            let result = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.image_tracks(&filter),
                }
            };
            let tracks = match result {
                Ok(tracks) => tracks,
                Err(err) => {
                    error!("Could not read the tracks of {rel_path}: {err}");
                    ImageTracks::default()
                }
            };
            log::debug!(
                "Tracks: {} of {} in {rel_path}",
                tracks.tracks.len(),
                tracks.total_tracks
            );
            let stats: Vec<TrackStats> = tracks.tracks.iter().map(TrackStats::of).collect();
            {
                let mut state = manager.tracks.lock().expect("Poisoned");
                if state.generation != generation {
                    return;
                }
                state.tracks = Arc::new(tracks);
                state.stats = Arc::new(stats);
            }
            manager.tracks_push_table();
            manager.tracks_objects_loaded();
            manager.split_push_selection(false);
            manager.split_push_overlay();
        });
    }

    fn class_colors(&self) -> HashMap<i32, u32> {
        self.classes
            .lock()
            .expect("Poisoned")
            .iter()
            .map(|class| (class.id as i32, class.color))
            .collect()
    }

    fn units(&self, tracks: &ImageTracks) -> Units {
        Units::new(
            tracks,
            self._app_state.load_app_settings().results_split.speed_unit,
        )
    }

    /// The table in its order, the image's name and how many tracks.
    pub(super) fn tracks_push_table(&self) {
        let ui_weak = self.ui.clone();
        let by_track = self
            ._app_state
            .load_app_settings()
            .results_split
            .color_by_track;
        let class_colors = self.class_colors();
        let (image, tracks, stats, order, selected) = {
            let mut state = self.tracks.lock().expect("Poisoned");
            let order = Arc::new(table_order(&state.tracks, &state.stats, state.sort));
            state.order = order.clone();
            (
                state.image.clone(),
                state.tracks.clone(),
                state.stats.clone(),
                order,
                state.selected,
            )
        };
        let units = self.units(&tracks);
        let colors: Vec<u32> = tracks
            .tracks
            .iter()
            .map(|track| track_rgb(track, by_track, &class_colors))
            .collect();
        let row = selected.and_then(|i| order.iter().position(|&o| o == i));
        let image_name = image
            .as_deref()
            .map(|rel| {
                Path::new(rel)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel.to_string())
            })
            .unwrap_or_default();
        let summary = match (tracks.tracks.len(), tracks.total_tracks) {
            (_, 0) => "No tracks".to_string(),
            (shown, total) if shown < total => {
                format!("{shown} of {total} tracks (a sample - too many to draw all)")
            }
            (_, 1) => "1 track".to_string(),
            (_, total) => format!("{total} tracks"),
        };
        let um = units.um_available();
        let label = units.speed_label();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_tracks_image(image_name.into());
            state.set_tracks_summary(summary.into());
            state.set_tracks_um(um);
            state.set_tracks_speed_unit(label.into());
            state.set_track_rows(ModelRc::new(TracksModel {
                tracks,
                stats,
                order,
                colors,
                units,
                notify: ModelNotify::default(),
            }));
            state.set_track_selected(row.map_or(-1, |r| r as i32));
        })
        .ok();
    }

    /// The selected track's card.
    pub(super) fn tracks_push_card(&self) {
        let (tracks, stats, selected) = {
            let state = self.tracks.lock().expect("Poisoned");
            (state.tracks.clone(), state.stats.clone(), state.selected)
        };
        let units = self.units(&tracks);
        let card = match selected.and_then(|i| Some((tracks.tracks.get(i)?, stats.get(i)?))) {
            None => TrackCard::default(),
            Some((track, stats)) => TrackCard {
                active: true,
                id: format!("#{}", track.track_id).into(),
                span: format!("{}–{} ({})", stats.first_t, stats.last_t, stats.points).into(),
                displacement: units.distance(stats.displacement_px).into(),
                speed: stats
                    .speed_px_per_frame
                    .map_or_else(
                        || "–".to_string(),
                        |s| {
                            format!(
                                "{} {}",
                                format_chart_value(units.speed(s)),
                                units.speed_label()
                            )
                        },
                    )
                    .into(),
            },
        };
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_track_card(card);
            }
        })
        .ok();
    }

    /// Track `index` selected: its row, its card, its object on the image -
    /// at its first time point when it isn't on this one (`fly`: flown to).
    pub(super) fn tracks_select(self: &Arc<Self>, index: usize, fly: bool) {
        let (track_id, first_t, row) = {
            let mut state = self.tracks.lock().expect("Poisoned");
            let Some(track) = state.tracks.tracks.get(index) else {
                return;
            };
            let (id, first_t) = (track.track_id, track.points.first().map_or(0, |p| p.t));
            state.selected = Some(index);
            (id, first_t, state.order.iter().position(|&o| o == index))
        };
        self.tracks_push_card();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>()
                    .set_track_selected(row.map_or(-1, |r| r as i32));
            }
        })
        .ok();
        let at = {
            let mut split = self.split.lock().expect("Poisoned");
            let at = split
                .objects
                .iter()
                .position(|o| o.track_id == Some(track_id));
            split.selected = at;
            at
        };
        match at {
            Some(_) => {
                self.split_push_selection(fly);
                self.split_push_overlay();
            }
            // Not on this time point: to its first (its object selected
            // once that plane's objects are read).
            None => {
                self.split_push_selection(false);
                self.split_push_overlay();
                self.split_show_time_point(first_t);
            }
        }
    }

    /// The split's objects were read: the selected track's is selected.
    pub(super) fn tracks_objects_loaded(&self) {
        if !self.in_tracks() {
            return;
        }
        let track_id = {
            let state = self.tracks.lock().expect("Poisoned");
            state
                .selected
                .and_then(|i| state.tracks.tracks.get(i))
                .map(|track| track.track_id)
        };
        if let Some(track_id) = track_id {
            let mut split = self.split.lock().expect("Poisoned");
            split.selected = split
                .objects
                .iter()
                .position(|o| o.track_id == Some(track_id));
        }
    }

    /// A click on the image selected object `object`: its track.
    pub(super) fn tracks_object_clicked(self: &Arc<Self>, track_id: Option<u64>) {
        let index = track_id.and_then(|id| {
            let state = self.tracks.lock().expect("Poisoned");
            state
                .tracks
                .tracks
                .binary_search_by_key(&id, |track| track.track_id)
                .ok()
        });
        match index {
            Some(index) => self.tracks_select(index, false),
            None => {
                self.tracks.lock().expect("Poisoned").selected = None;
                self.tracks_push_card();
                if let Some(ui) = self.ui.upgrade() {
                    ui.global::<ResultsState>().set_track_selected(-1);
                }
            }
        }
    }

    /// The selected track's row (from 1; 0 none) and how many rows.
    pub(super) fn tracks_position(&self) -> (i32, i32) {
        let state = self.tracks.lock().expect("Poisoned");
        let row = state
            .selected
            .and_then(|i| state.order.iter().position(|&o| o == i));
        (row.map_or(0, |r| r as i32 + 1), state.order.len() as i32)
    }

    /// ‹ ›: the previous / next track in the table.
    pub(super) fn tracks_step(self: &Arc<Self>, delta: i32) {
        let next = {
            let state = self.tracks.lock().expect("Poisoned");
            let n = state.order.len();
            if n == 0 {
                return;
            }
            let row = state
                .selected
                .and_then(|i| state.order.iter().position(|&o| o == i));
            let next = match row {
                None if delta < 0 => n - 1,
                None => 0,
                Some(r) => (r as i64 + delta as i64).rem_euclid(n as i64) as usize,
            };
            state.order[next]
        };
        self.tracks_select(next, true);
    }

    /// The trails for the overlay: of the split's image, up to time point
    /// `t` - `None` without tracks, off, or another view.
    pub(super) fn tracks_trails(&self, image: Option<&str>, t: u32) -> Option<Arc<TrailsOverlay>> {
        if !self.in_tracks() {
            return None;
        }
        let settings = self._app_state.load_app_settings().results_split;
        let frames = match settings.trail {
            TrailMode::Off => return None,
            TrailMode::Frames => Some(settings.trail_frames.max(1)),
            TrailMode::Whole => None,
        };
        let (tracks, selected) = {
            let state = self.tracks.lock().expect("Poisoned");
            if state.image.as_deref() != image || state.tracks.tracks.is_empty() {
                return None;
            }
            (state.tracks.clone(), state.selected)
        };
        let class_colors = self.class_colors();
        let colors = tracks
            .tracks
            .iter()
            .map(|track| {
                let rgb = track_rgb(track, settings.color_by_track, &class_colors);
                [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
            })
            .collect();
        Some(Arc::new(TrailsOverlay {
            tracks,
            colors,
            t,
            frames,
            selected,
            only_selected: settings.only_selected_track,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::results::TrackPoint;

    fn track(id: u64, points: &[(u32, f32, f32)]) -> TrackPath {
        TrackPath {
            track_id: id,
            class_id: 1,
            points: points
                .iter()
                .map(|&(t, x, y)| TrackPoint {
                    t,
                    x,
                    y,
                    area_px: 10.0 + t as f32,
                })
                .collect(),
        }
    }

    /// Speed: the path's length over its time points; displacement: first
    /// to last point.
    #[test]
    fn a_tracks_stats_are_from_its_path() {
        let stats = TrackStats::of(&track(1, &[(2, 0.0, 0.0), (3, 3.0, 4.0), (5, 3.0, 0.0)]));
        assert_eq!((stats.first_t, stats.last_t, stats.points), (2, 5, 3));
        assert_eq!(stats.displacement_px, 3.0);
        assert_eq!(stats.speed_px_per_frame, Some(9.0 / 3.0));
        assert_eq!(
            TrackStats::of(&track(2, &[(4, 1.0, 1.0)])).speed_px_per_frame,
            None
        );
    }

    /// µm/min and µm/s from the pixel size and frame interval; without
    /// either, px/frame whatever was picked.
    #[test]
    fn speeds_are_in_the_unit_picked_when_it_can_be() {
        let tracks = ImageTracks {
            pixel_size_nm: Some(500.0),
            frame_interval_s: Some(30.0),
            ..ImageTracks::default()
        };
        let per_min = Units::new(&tracks, SpeedUnit::UmPerMin);
        assert_eq!(per_min.speed(2.0), 2.0 * 0.5 / 30.0 * 60.0);
        assert_eq!(per_min.speed_label(), "µm/min");
        assert_eq!(
            Units::new(&tracks, SpeedUnit::UmPerS).speed(2.0),
            2.0 * 0.5 / 30.0
        );
        assert_eq!(per_min.distance(4.0), "2 µm");
        let unknown = Units::new(&ImageTracks::default(), SpeedUnit::UmPerS);
        assert!(!unknown.um_available());
        assert_eq!(
            (unknown.speed(2.0), unknown.speed_label()),
            (2.0, "px/frame")
        );
        assert_eq!(unknown.distance(4.0), "4 px");
    }

    /// By id; by time points (most first); by speed (fastest first, those
    /// without a speed last).
    #[test]
    fn the_table_is_sorted_as_asked() {
        let tracks = ImageTracks {
            tracks: vec![
                track(1, &[(0, 0.0, 0.0), (1, 1.0, 0.0)]),
                track(2, &[(0, 0.0, 0.0), (1, 5.0, 0.0), (2, 10.0, 0.0)]),
                track(3, &[(0, 0.0, 0.0)]),
            ],
            ..ImageTracks::default()
        };
        let stats: Vec<TrackStats> = tracks.tracks.iter().map(TrackStats::of).collect();
        assert_eq!(table_order(&tracks, &stats, 0), [0, 1, 2]);
        assert_eq!(table_order(&tracks, &stats, 1), [1, 0, 2]);
        assert_eq!(table_order(&tracks, &stats, 2), [1, 0, 2]);
    }

    /// The table's rows: id, span, time points, speed and the area curve -
    /// made only when asked for.
    #[test]
    fn table_rows_are_made_from_the_tracks() {
        let tracks = Arc::new(ImageTracks {
            tracks: vec![track(7, &[(1, 0.0, 0.0), (3, 6.0, 8.0)])],
            total_tracks: 1,
            ..ImageTracks::default()
        });
        let stats = Arc::new(tracks.tracks.iter().map(TrackStats::of).collect::<Vec<_>>());
        let model = TracksModel {
            units: Units::new(&tracks, SpeedUnit::PxPerFrame),
            tracks,
            stats,
            order: Arc::new(vec![0]),
            colors: vec![0xFF0000],
            notify: ModelNotify::default(),
        };
        assert_eq!(model.row_count(), 1);
        let row = model.row_data(0).unwrap();
        assert_eq!(row.id, "#7");
        assert_eq!(row.span, "1–3");
        assert_eq!(row.frames, "2");
        assert_eq!(row.speed, "5");
        assert_eq!(row.curve, "M0.0 18.0 L100.0 2.0 ");
        assert!(model.row_data(1).is_none());
    }
}
