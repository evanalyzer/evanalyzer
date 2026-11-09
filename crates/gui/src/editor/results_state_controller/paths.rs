//! # paths
//!
//! Results › Paths: every track the filters pick drawn from its start (an
//! even sample of `MAX_PATHS` beyond), and the motion map - a dot per track
//! at its straightness (LIN) and its turning or MSD α, as Measure Track
//! Motion stored them. A click on the map selects the nearest track: its
//! ring, its path on top, its values. Nothing is computed here but where
//! things are drawn.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::tracks::TRACK_PALETTE;
use super::*;
use crate::{PathsDot, PathsLine};
use evanalyzer_app::results::view_settings::PathsView;
use evanalyzer_app::results::{
    TrackFromStart, TrackPaths, TrackPathsFilter, TrackSummary, TrackSummaryFilter,
};

/// At most this many paths are drawn.
const MAX_PATHS: usize = 2000;
/// At most this many dots are on the motion map.
const MAX_DOTS: usize = 5000;
/// Paths' viewbox: 1000 × 1000, the start in the middle, the farthest
/// point this far from it.
const PATH_RADIUS: f64 = 480.0;

#[derive(Default)]
pub(super) struct PathsState {
    /// The view's settings (stored in the project).
    pub(super) view: PathsView,
    /// The filters' choices: the class ids and image paths after "All".
    class_ids: Vec<u32>,
    image_paths: Vec<String>,
    /// Bumped by every read: a late answer is dropped.
    generation: u64,
    rows: Arc<Vec<TrackSummary>>,
    paths: Arc<TrackPaths>,
    /// The dots shown: (index into `rows`, x, y), 0..1.
    dots: Vec<(usize, f64, f64)>,
    /// Index into `rows`.
    selected: Option<usize>,
}

/// A track's color: by track, by well / group (its place among `groups`)
/// or by class.
fn color_of(color_by: &str, track_id: u64, group: &str, class_id: i32, groups: &[String]) -> u32 {
    let index = match color_by {
        "group" => groups.iter().position(|g| g == group).unwrap_or(0),
        "class" => class_id.max(0) as usize,
        _ => track_id as usize,
    };
    TRACK_PALETTE[index % TRACK_PALETTE.len()]
}

/// The paths as one command string per color (0 0 1000 1000, each from
/// the middle, the farthest point `PATH_RADIUS` from it) and how far that
/// is, px.
fn path_lines(
    paths: &[TrackFromStart],
    color: impl Fn(&TrackFromStart) -> u32,
) -> (Vec<(u32, String)>, f64) {
    let extent = paths
        .iter()
        .flat_map(|t| t.points.iter())
        .map(|(x, y)| f64::from(x.abs().max(y.abs())))
        .fold(0.0, f64::max);
    let scale = if extent > 0.0 {
        PATH_RADIUS / extent
    } else {
        0.0
    };
    let mut lines: Vec<(u32, String)> = Vec::new();
    for track in paths.iter().filter(|t| t.points.len() > 1) {
        let c = color(track);
        let i = match lines.iter().position(|(lc, _)| *lc == c) {
            Some(i) => i,
            None => {
                lines.push((c, String::new()));
                lines.len() - 1
            }
        };
        lines[i].1.push_str(&path_commands(&track.points, scale));
    }
    (lines, extent)
}

/// One track's path in the paths' viewbox at `scale` (px → viewbox).
fn path_commands(points: &[(f32, f32)], scale: f64) -> String {
    points
        .iter()
        .enumerate()
        .map(|(k, (x, y))| {
            format!(
                "{}{:.1} {:.1} ",
                if k == 0 { "M" } else { "L" },
                500.0 + f64::from(*x) * scale,
                500.0 + f64::from(*y) * scale
            )
        })
        .collect()
}

/// The motion map's y range: turning symmetric around 0 (±10°, ±30°,
/// ±90° or ±180°, the smallest that holds every value), MSD α from 0 to 2
/// or the largest value.
fn y_range(y_axis: &str, values: impl Iterator<Item = f64>) -> (f64, f64) {
    let max = values.fold(0.0f64, |m, v| m.max(v.abs()));
    match y_axis {
        "msd_alpha" => (0.0, max.max(2.0).ceil()),
        _ => {
            let bound = [10.0, 30.0, 90.0, 180.0]
                .into_iter()
                .find(|b| max <= *b)
                .unwrap_or(180.0);
            (-bound, bound)
        }
    }
}

/// The dots: a track with a straightness and a `y_axis` value, an even
/// sample of `MAX_DOTS` beyond - (row, x, y) in 0..1 of `range`.
fn map_dots(rows: &[TrackSummary], y_axis: &str, range: (f64, f64)) -> Vec<(usize, f64, f64)> {
    let y_of = |t: &TrackSummary| match y_axis {
        "msd_alpha" => t.motion.msd_alpha,
        _ => t.motion.turning,
    };
    let with: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].motion.lin.is_some() && y_of(&rows[i]).is_some())
        .collect();
    let keep = with.len().min(MAX_DOTS);
    (0..keep)
        .map(|k| with[k * with.len() / keep])
        .map(|i| {
            let t = &rows[i];
            let x = t.motion.lin.unwrap_or(0.0).clamp(0.0, 1.0);
            let y = ((y_of(t).unwrap_or(0.0) - range.0) / (range.1 - range.0)).clamp(0.0, 1.0);
            (i, x, y)
        })
        .collect()
}

/// The dot nearest to (x, y) on a plot of `w` × `h` px - within 12 px.
fn nearest_dot(dots: &[(usize, f64, f64)], x: f64, y: f64, w: f64, h: f64) -> Option<usize> {
    dots.iter()
        .map(|&(row, dx, dy)| (row, ((dx - x) * w).hypot((dy - y) * h)))
        .filter(|(_, d)| *d <= 12.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(row, _)| row)
}

impl ResultsStateController {
    pub(super) fn attach_paths_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_paths_filter_changed(move || {
            manager.paths_read_filters();
            manager.remember_view();
            manager.paths_load();
        });
        let manager = self.clone();
        state.on_paths_map_clicked(move |x, y, w, h| {
            {
                let mut paths = manager.paths.lock().expect("Poisoned");
                paths.selected = nearest_dot(
                    &paths.dots,
                    f64::from(x),
                    f64::from(y),
                    f64::from(w),
                    f64::from(h),
                );
            }
            manager.paths_push();
        });
    }

    /// Paths shown: the filters' choices and the view's settings, then the
    /// tracks.
    pub(super) fn paths_show(self: &Arc<Self>) {
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
        let view = {
            let mut state = self.paths.lock().expect("Poisoned");
            state.class_ids = classes.iter().map(|(id, _)| *id).collect();
            state.image_paths = images.iter().map(|(_, rel)| rel.clone()).collect();
            state.view.clone()
        };
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ResultsState>();
            let names = |all: &str, names: Vec<String>| -> ModelRc<slint::SharedString> {
                let items: Vec<slint::SharedString> = std::iter::once(all.to_string())
                    .chain(names)
                    .map(Into::into)
                    .collect();
                ModelRc::from(Rc::new(VecModel::from(items)))
            };
            let class_index = view
                .class
                .and_then(|id| classes.iter().position(|(c, _)| *c == id))
                .map_or(0, |i| i as i32 + 1);
            let image_index = view
                .image
                .as_ref()
                .and_then(|rel| images.iter().position(|(_, r)| r == rel))
                .map_or(0, |i| i as i32 + 1);
            state.set_paths_classes(names(
                "All classes",
                classes.into_iter().map(|(_, name)| name).collect(),
            ));
            state.set_paths_images(names(
                "All images",
                images.into_iter().map(|(name, _)| name).collect(),
            ));
            state.set_paths_class(class_index);
            state.set_paths_image(image_index);
            state.set_paths_min(view.min_points.to_string().into());
            state.set_paths_from_start(view.from_start);
            state.set_paths_no_edge(view.no_edge);
            state.set_paths_color_by(match view.color_by.as_str() {
                "group" => 1,
                "class" => 2,
                _ => 0,
            });
            state.set_paths_y_axis(i32::from(view.y_axis == "msd_alpha"));
        }
        self.paths_load();
    }

    /// The controls into the view's settings.
    fn paths_read_filters(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ResultsState>();
        let mut paths = self.paths.lock().expect("Poisoned");
        let pick = |index: i32| (index > 0).then(|| index as usize - 1);
        let class = pick(state.get_paths_class()).and_then(|i| paths.class_ids.get(i).copied());
        let image = pick(state.get_paths_image()).and_then(|i| paths.image_paths.get(i).cloned());
        paths.view = PathsView {
            class,
            image,
            min_points: state.get_paths_min().trim().parse().unwrap_or(1).max(1),
            from_start: state.get_paths_from_start(),
            no_edge: state.get_paths_no_edge(),
            color_by: match state.get_paths_color_by() {
                1 => "group",
                2 => "class",
                _ => "track",
            }
            .into(),
            y_axis: match state.get_paths_y_axis() {
                1 => "msd_alpha",
                _ => "turning",
            }
            .into(),
        };
    }

    /// The tracks the view asks for: their rows (the map) and a sample of
    /// their paths (on the database thread).
    fn paths_load(self: &Arc<Self>) {
        let (view, generation) = {
            let mut paths = self.paths.lock().expect("Poisoned");
            paths.generation += 1;
            (paths.view.clone(), paths.generation)
        };
        let summary = TrackSummaryFilter {
            columns: Vec::new(),
            object_classes: view.class.map(|id| vec![ObjectClass::Global(id)]),
            images: view.image.clone().map(|rel| vec![rel]),
            min_points: view.min_points.max(1),
            from_start: view.from_start,
            no_edge: view.no_edge,
            ..TrackSummaryFilter::default()
        };
        self.run_db_job(move |manager| {
            let answer = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.track_summary(&summary).and_then(|rows| {
                        let paths = db.track_paths(&TrackPathsFilter {
                            summary: summary.clone(),
                            max_tracks: MAX_PATHS,
                        })?;
                        Ok((rows, paths))
                    }),
                }
            };
            let (rows, paths) = answer.unwrap_or_else(|err| {
                error!("Could not read the tracks' paths: {err}");
                (Vec::new(), TrackPaths::default())
            });
            {
                let mut state = manager.paths.lock().expect("Poisoned");
                if state.generation != generation {
                    return;
                }
                state.rows = Arc::new(rows);
                state.paths = Arc::new(paths);
                state.selected = None;
            }
            manager.paths_push();
        });
    }

    /// The paths, the map and the selected track into the window.
    fn paths_push(&self) {
        let (view, rows, paths, selected) = {
            let state = self.paths.lock().expect("Poisoned");
            (
                state.view.clone(),
                state.rows.clone(),
                state.paths.clone(),
                state.selected,
            )
        };
        let mut groups: Vec<String> = rows.iter().map(|t| t.group.clone()).collect();
        groups.sort();
        groups.dedup();
        let color = |track_id: u64, group: &str, class_id: i32| {
            color_of(&view.color_by, track_id, group, class_id, &groups)
        };
        let (lines, extent) =
            path_lines(&paths.tracks, |t| color(t.track_id, &t.group, t.class_id));
        let scale = if extent > 0.0 {
            PATH_RADIUS / extent
        } else {
            0.0
        };
        // Lengths in µm when every drawn track has a pixel size.
        let um = (!paths.tracks.is_empty())
            .then(|| {
                paths
                    .tracks
                    .iter()
                    .map(|t| t.um_per_px)
                    .collect::<Option<Vec<f64>>>()
            })
            .flatten()
            .and_then(|sizes| sizes.into_iter().reduce(f64::max));
        let extent_label = match (extent > 0.0, um) {
            (false, _) => String::new(),
            (true, Some(um)) => format!(
                "± {} µm",
                format_chart_value((extent * um * 10.0).round() / 10.0)
            ),
            (true, None) => format!("± {} px", format_chart_value(extent.round())),
        };
        let header = match (paths.total_tracks, paths.tracks.len()) {
            (0, _) => String::new(),
            (total, drawn) if drawn < total => format!("{total} tracks · {drawn} drawn"),
            (total, _) => format!("{total} tracks"),
        };

        let y_of = |t: &TrackSummary| match view.y_axis.as_str() {
            "msd_alpha" => t.motion.msd_alpha,
            _ => t.motion.turning,
        };
        let range = y_range(&view.y_axis, rows.iter().filter_map(y_of));
        let dots = map_dots(&rows, &view.y_axis, range);
        let measured = rows.iter().any(|t| t.motion.is_measured());
        let map_header = match (measured, dots.len(), rows.len()) {
            (false, ..) => {
                "no Measure Track Motion values - add the step after tracking".to_string()
            }
            (true, shown, all) if shown < all => format!("{shown} of {all} tracks"),
            (true, shown, _) => format!("{shown} tracks"),
        };
        let (y_title, y_fmt): (&str, fn(f64) -> String) = match view.y_axis.as_str() {
            "msd_alpha" => ("MSD α", |v| format!("{v:.1}")),
            _ => ("Turning °", |v| format!("{v:+.0}")),
        };
        let selected_track = selected.and_then(|i| rows.get(i));
        let highlight = selected_track
            .and_then(|t| {
                paths
                    .tracks
                    .iter()
                    .find(|p| p.track_id == t.track_id && p.image == t.image)
            })
            .map(|p| path_commands(&p.points, scale))
            .unwrap_or_default();
        let selected_dot = selected.and_then(|i| dots.iter().find(|d| d.0 == i).copied());
        let caption = selected_track
            .map(|t| {
                let value =
                    |v: Option<f64>, f: &dyn Fn(f64) -> String| v.map_or("–".to_string(), f);
                format!(
                    "#{} · {} · LIN {} · turning {} · MSD α {}",
                    t.track_id,
                    t.image,
                    value(t.motion.lin, &|v| format!("{v:.2}")),
                    value(t.motion.turning, &|v| format!("{v:+.1}°")),
                    value(t.motion.msd_alpha, &|v| format!("{v:.2}")),
                )
            })
            .unwrap_or_default();
        let empty = match rows.is_empty() {
            true => "No tracks for these filters".to_string(),
            false => String::new(),
        };
        {
            let mut state = self.paths.lock().expect("Poisoned");
            state.dots = dots.clone();
        }
        let line_items: Vec<PathsLine> = lines
            .into_iter()
            .map(|(c, commands)| PathsLine {
                commands: commands.into(),
                color: bg_color_to_slint(c),
            })
            .collect();
        let dot_items: Vec<PathsDot> = dots
            .iter()
            .map(|&(i, x, y)| {
                let t = &rows[i];
                PathsDot {
                    x: x as f32,
                    y: y as f32,
                    color: bg_color_to_slint(color(t.track_id, &t.group, t.class_id)),
                }
            })
            .collect();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            state.set_paths_empty(empty.into());
            state.set_paths_lines(ModelRc::from(Rc::new(VecModel::from(line_items))));
            state.set_paths_highlight(highlight.into());
            state.set_paths_extent(extent_label.into());
            state.set_paths_header(header.into());
            state.set_paths_dots(ModelRc::from(Rc::new(VecModel::from(dot_items))));
            state.set_paths_map_header(map_header.into());
            state.set_paths_y_title(y_title.into());
            state.set_paths_y_max(y_fmt(range.1).into());
            state.set_paths_y_min(y_fmt(range.0).into());
            state.set_paths_has_selected(selected_dot.is_some());
            if let Some((_, x, y)) = selected_dot {
                state.set_paths_selected_x(x as f32);
                state.set_paths_selected_y(y as f32);
            }
            state.set_paths_selected(caption.into());
        })
        .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_app::results::TrackMotion;

    fn path(track_id: u64, group: &str, points: &[(f32, f32)]) -> TrackFromStart {
        TrackFromStart {
            image: "a.tif".into(),
            group: group.into(),
            class_id: 1,
            track_id,
            um_per_px: None,
            points: points.to_vec(),
        }
    }

    /// One path per color; the farthest point of any track at the edge of
    /// the 1000 × 1000 box, every track from its middle.
    #[test]
    fn paths_share_one_scale_and_a_line_per_color() {
        let tracks = [
            path(1, "A", &[(0.0, 0.0), (10.0, 0.0)]),
            path(11, "A", &[(0.0, 0.0), (0.0, -5.0)]),
            path(2, "B", &[(0.0, 0.0)]),
        ];
        let (lines, extent) =
            path_lines(&tracks, |t| color_of("track", t.track_id, &t.group, 1, &[]));
        assert_eq!(extent, 10.0);
        // Tracks 1 and 11 share a color (10 colors); track 2 has one point.
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].1,
            "M500.0 500.0 L980.0 500.0 M500.0 500.0 L500.0 260.0 "
        );
        let groups = ["A".to_string(), "B".to_string()];
        assert_ne!(
            color_of("group", 1, "A", 1, &groups),
            color_of("group", 1, "B", 1, &groups)
        );
    }

    /// The map's range: turning symmetric (the smallest step holding every
    /// value), MSD α from 0; dots in 0..1 of it, only tracks with both
    /// values; a click finds the nearest dot within 12 px.
    #[test]
    fn the_motion_map_places_each_measured_track() {
        assert_eq!(y_range("turning", [5.0, -25.0].into_iter()), (-30.0, 30.0));
        assert_eq!(y_range("turning", [500.0].into_iter()), (-180.0, 180.0));
        assert_eq!(y_range("msd_alpha", [1.2].into_iter()), (0.0, 2.0));
        let row = |lin: Option<f64>, turning: Option<f64>| {
            let mut motion = TrackMotion::default();
            motion.lin = lin;
            motion.turning = turning;
            TrackSummary {
                motion,
                ..TrackSummary::default()
            }
        };
        let rows = [
            row(Some(0.5), Some(15.0)),
            row(None, Some(1.0)),
            row(Some(1.0), Some(-30.0)),
        ];
        let dots = map_dots(&rows, "turning", (-30.0, 30.0));
        assert_eq!(dots, [(0, 0.5, 0.75), (2, 1.0, 0.0)]);
        assert_eq!(nearest_dot(&dots, 0.51, 0.74, 400.0, 300.0), Some(0));
        assert_eq!(nearest_dot(&dots, 0.2, 0.2, 400.0, 300.0), None);
    }
}
