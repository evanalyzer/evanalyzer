//! Results, split with the image (board Z): the results left, the image
//! right. A List row or an object on the image selects it - the other one
//! follows; Plate shows a well's images, one at a time, with a filmstrip.
//! The image is the one viewer; its objects here are the analysis' (see
//! [`ResultsOverlay`]), not the project's.

use super::*;
use crate::editor::results_overlay::{ResultsOverlay, focus_view, object_at};
use crate::{SplitObject, SplitThumb, ViewportState as ViewportSlintState};
use evanalyzer_app::results::{ImageObjectsFilter, ResultObject};
use slint::{Rgb8Pixel, SharedPixelBuffer};

/// Longest side of a filmstrip thumbnail's pyramid level.
const THUMB_SIDE: u64 = 256;
/// Images without a level this small get no thumbnail (too much to read).
const THUMB_MAX_READ_SIDE: u64 = 4096;
/// A thumbnail is kept this size (the strip's tile, twice for HiDPI).
const THUMB_KEPT: (u32, u32) = (184, 112);
/// Thumbnails kept at most (about 60 KB each).
const THUMBS_KEPT: usize = 384;
/// The fly to an object.
const FLY_MS: f32 = 400.0;

#[derive(Default)]
pub(super) struct SplitState {
    /// The image shown (`image_rel_path`) and its objects, by `object_id`.
    pub(super) image: Option<String>,
    pub(super) objects: Arc<Vec<ResultObject>>,
    /// Index into `objects`.
    pub(super) selected: Option<usize>,
    /// A region shown instead of an object (an image heatmap's tile), in
    /// image pixels: a yellow box, flown to.
    region: Option<[u32; 4]>,
    /// Bumped by every image shown: a late answer for another is dropped.
    generation: u64,
    /// Plate: the well's images, `(name, rel_path)`.
    well_images: Vec<(String, String)>,
    /// The channel of the card's mean.
    channel: usize,
    /// The split view is on screen (Results, List or Plate, split).
    pub(super) shown: bool,
    /// Left with "Open in Images": its marker stays on the image (with the
    /// way back to the results) - else it goes with the split view.
    keep_marker: bool,
    /// Plate "Color by value": the objects in the heatmap's colors. Kept
    /// here - the overlay is also pushed from the database thread, where
    /// the window can't be reached.
    by_value: bool,
    /// Thumbnails made so far, by `rel_path`.
    thumbs: HashMap<String, SharedPixelBuffer<Rgb8Pixel>>,
}

impl ResultsStateController {
    pub(super) fn attach_split_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();

        // The split's image open now: its objects and the selected one on it.
        let weak = Arc::downgrade(self);
        self._app_state.on_image_opened(move || {
            if let Some(manager) = weak.upgrade() {
                manager.split_image_opened();
            }
        });

        // The viewer on another plane (its T slider, play): the split's
        // objects and the results of that plane - one time point for both.
        let weak = Arc::downgrade(self);
        self._app_state.on_plane_changed(move |z, t| {
            if let Some(manager) = weak.upgrade() {
                manager.split_plane_changed(z, t);
            }
        });

        // As the user left it (a user setting).
        let settings = self._app_state.load_app_settings().results_split;
        state.set_split(settings.on);
        state.set_split_w(settings.width_px.max(0.0));
        state.set_split_follow(settings.follow);
        state.set_split_image_panel(settings.image_panel);
        state.set_split_card_folded(settings.card_folded);
        let manager = self.clone();
        state.on_split_card_folded_changed(move |folded| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.card_folded = folded);
        });

        let manager = self.clone();
        state.on_split_changed(move |on| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.on = on);
            // Tracks: one image's beside it, or every track.
            if manager.in_tracks() {
                match on {
                    true => manager.tracks_show(),
                    false => manager.tracks_all_show(),
                }
            }
            // A row already selected: shown right away.
            if on && let Some(ui) = manager.ui.upgrade() {
                let row = ui.global::<ResultsState>().get_list_selected_row();
                if row >= 0 && *manager.rail_mode.lock().expect("Poisoned") == ResultsRailMode::List
                {
                    manager.split_select_row(row);
                }
            }
        });
        let manager = self.clone();
        state.on_split_width_changed(move |w| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.width_px = w);
        });
        let manager = self.clone();
        state.on_split_follow_changed(move |follow| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.follow = follow);
        });
        let manager = self.clone();
        state.on_split_image_panel_changed(move |open| {
            manager
                ._app_state
                .update_app_settings(|s| s.results_split.image_panel = open);
        });
        // The split shown or not (mode, tab, Full/Split): its objects on
        // the image, or the project's again.
        let manager = self.clone();
        state.on_split_shown(move |shown| {
            let keep_marker = {
                let mut split = manager.split.lock().expect("Poisoned");
                split.shown = shown;
                std::mem::take(&mut split.keep_marker)
            };
            if shown {
                manager.split_push_overlay();
            } else {
                manager.viewport().set_results_overlay(None);
                *manager
                    .viewport()
                    .viewport_cache
                    .focus
                    .lock()
                    .expect("Poisoned") = None;
                // Its marker goes with it (it stayed on the image after
                // leaving the results) - unless it was opened in Images.
                if !keep_marker && let Some(ui) = manager.ui.upgrade() {
                    ui.global::<crate::ViewportObjectState>()
                        .set_object_highlight(crate::ObjectHighlightBox::default());
                }
            }
        });
        let manager = self.clone();
        state.on_split_by_value_changed(move |on| {
            manager.split.lock().expect("Poisoned").by_value = on;
            manager.split_push_overlay();
        });
        let manager = self.clone();
        state.on_split_channel_changed(move |channel| {
            manager.split.lock().expect("Poisoned").channel = channel.max(0) as usize;
            manager.split_push_selection(false);
        });
        let manager = self.clone();
        state.on_list_row_selected(move |row| manager.split_select_row(row));
        let manager = self.clone();
        state.on_split_step(move |delta| manager.split_step(delta));
        let manager = self.clone();
        state.on_split_clear(move || {
            {
                let mut split = manager.split.lock().expect("Poisoned");
                split.selected = None;
                split.region = None;
            }
            if let Some(ui) = manager.ui.upgrade() {
                ui.global::<ResultsState>().set_list_selected_row(-1);
            }
            manager.split_push_selection(false);
            manager.split_push_overlay();
        });
        let manager = self.clone();
        state.on_split_image_clicked(move |x, y| manager.split_image_clicked(x, y));
        let manager = self.clone();
        state.on_split_show_image(move |name| {
            let rel_path = manager
                .split
                .lock()
                .expect("Poisoned")
                .well_images
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .map(|(_, rel)| rel.clone());
            if let Some(rel_path) = rel_path {
                manager.split_show_image(rel_path, None, None);
            }
        });
        let manager = self.clone();
        // In Images with its marker - and the bar back to the results, as
        // "Show on image" has it.
        state.on_split_open_in_images(move || {
            let Some(ui) = manager.ui.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let row = state.get_list_selected_row();
            let rows = manager.list_row_locations.lock().expect("Poisoned").len();
            let label = if manager.in_plate() || row < 0 {
                state.get_split_title().to_string()
            } else {
                format!("Row {} of {rows}", row + 1)
            };
            manager.split.lock().expect("Poisoned").keep_marker = true;
            state.set_peek_label(label.into());
            state.set_peek_can_step(!manager.in_plate() && row >= 0);
            state.set_peek(true);
            ui.global::<crate::LayoutState>()
                .set_mode(crate::WorkspaceMode::Images);
        });
    }

    pub(super) fn viewport(&self) -> &Arc<ViewportController> {
        &self.image_list_controller.viewport_controller
    }

    fn in_plate(&self) -> bool {
        *self.rail_mode.lock().expect("Poisoned") == ResultsRailMode::Matrix
    }

    /// Plate, a well clicked: its images under the image, the first shown.
    pub(super) fn split_show_well(self: &Arc<Self>, image_names: Vec<String>) {
        let rel_paths: HashMap<String, String> = self
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
        let well_images: Vec<(String, String)> = image_names
            .into_iter()
            .filter_map(|name| rel_paths.get(&name).map(|rel| (name, rel.clone())))
            .collect();
        let first = well_images.first().map(|(_, rel)| rel.clone());
        self.split.lock().expect("Poisoned").well_images = well_images.clone();
        self.split_push_thumbs();
        self.split_make_thumbs(well_images.into_iter().map(|(_, rel)| rel).collect());
        if let Some(first) = first {
            self.split_show_image(first, None, None);
        }
    }

    /// The image heatmap's tile `key` on the image drilled into.
    pub(super) fn split_show_tile(self: &Arc<Self>, key: &str) {
        let rel_path = self.current_image.lock().expect("Poisoned").clone();
        if let (Some(rel_path), Some(bbox)) = (rel_path, self.tile_bbox(key)) {
            self.split_show_image(rel_path, None, Some(bbox));
        }
    }

    /// A well's image (the well level's field `name`), whole.
    pub(super) fn split_show_image_named(self: &Arc<Self>, name: &str) {
        let rel_path = self
            .images
            .lock()
            .expect("Poisoned")
            .iter()
            .find(|image| image.name == name)
            .map(|image| image.rel_path.to_string_lossy().into_owned());
        if let Some(rel_path) = rel_path {
            self.split_show_image(rel_path, None, None);
        }
    }

    /// A List row selected: its object on its image - a row per image: the
    /// whole image with its objects.
    fn split_select_row(self: &Arc<Self>, row: i32) {
        let location = self
            .list_row_locations
            .lock()
            .expect("Poisoned")
            .get(row.max(0) as usize)
            .cloned();
        let Some((rel_path, bbox)) = location.filter(|_| row >= 0) else {
            return;
        };
        self.split_show_image(rel_path, bbox, None);
    }

    /// ‹ › and ↑ ↓: List the next row, Plate the next object of the image
    /// (round).
    fn split_step(self: &Arc<Self>, delta: i32) {
        if self.in_tracks() {
            self.tracks_step(delta);
            return;
        }
        if self.in_plate() {
            let index = {
                let mut split = self.split.lock().expect("Poisoned");
                let n = split.objects.len();
                if n == 0 {
                    return;
                }
                let next = match split.selected {
                    None if delta < 0 => n - 1,
                    None => 0,
                    Some(i) => (i as i64 + delta as i64).rem_euclid(n as i64) as usize,
                };
                split.selected = Some(next);
                next
            };
            let _ = index;
            self.split_push_selection(true);
            self.split_push_overlay();
            return;
        }
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let rows = self.list_row_locations.lock().expect("Poisoned").len() as i32;
        if rows == 0 {
            return;
        }
        let state = ui.global::<ResultsState>();
        let current = state.get_list_selected_row();
        let row = if current < 0 {
            if delta < 0 { rows - 1 } else { 0 }
        } else {
            (current + delta).clamp(0, rows - 1)
        };
        state.set_list_selected_row(row);
        self.split_select_row(row);
    }

    /// A click on the image: the object there, and its row (if on this
    /// page).
    fn split_image_clicked(self: &Arc<Self>, x: f32, y: f32) {
        let Some(overlay) = self.viewport().results_overlay() else {
            return;
        };
        let (zoom, off_x, off_y, ..) = self.viewport().view();
        let Some(index) = object_at(&overlay, x, y, zoom, off_x, off_y) else {
            return;
        };
        let (rel_path, bbox, track_id) = {
            let mut split = self.split.lock().expect("Poisoned");
            split.selected = Some(index);
            (
                split.image.clone().unwrap_or_default(),
                split.objects[index].bbox_px,
                split.objects[index].track_id,
            )
        };
        if self.in_tracks() {
            self.tracks_object_clicked(track_id);
            self.split_push_selection(false);
            self.split_push_overlay();
            return;
        }
        let row = self
            .list_row_locations
            .lock()
            .expect("Poisoned")
            .iter()
            .position(|(rel, b)| *rel == rel_path && *b == Some(bbox))
            .map_or(-1, |row| row as i32);
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ResultsState>().set_list_selected_row(row);
        }
        self.split_push_selection(false);
        self.split_push_overlay();
    }

    /// The object with this box on the image shown.
    fn split_select_bbox(self: &Arc<Self>, bbox: [u32; 4], fly: bool) {
        {
            let mut split = self.split.lock().expect("Poisoned");
            split.selected = split.objects.iter().position(|o| o.bbox_px == bbox);
        }
        self.split_push_selection(fly);
        self.split_push_overlay();
    }

    /// Shows `rel_path` in the image pane, and then selects the object
    /// with box `select` - or marks `region` (a heatmap tile): opened at
    /// that when following, its objects read off the UI thread.
    pub(super) fn split_show_image(
        self: &Arc<Self>,
        rel_path: String,
        select: Option<[u32; 4]>,
        region: Option<[u32; 4]>,
    ) {
        let generation = {
            let mut split = self.split.lock().expect("Poisoned");
            // Asked for from the split view: it is on screen.
            split.shown = true;
            split.region = region;
            if split.image.as_deref() == Some(rel_path.as_str()) && !split.objects.is_empty() {
                if select.is_none() {
                    split.selected = None;
                }
                drop(split);
                match select {
                    Some(bbox) => self.split_select_bbox(bbox, true),
                    None => {
                        self.split_push_selection(region.is_some());
                        self.split_push_overlay();
                    }
                }
                return;
            }
            split.generation += 1;
            split.image = Some(rel_path.clone());
            split.objects = Arc::new(Vec::new());
            split.selected = None;
            split.generation
        };
        self.viewport().set_results_overlay(None);
        self.split_push_selection(false);

        // Not in the viewer yet: opened, at the object when following.
        // The series it is shown in (its own choice, else the project's):
        // the card's channel names are that series'.
        let (absolute, series) = {
            use evanalyzer_app::project::{ImageEntryExt, ProjectExt};
            let project = self._app_state.get_project();
            let absolute = project.get_image_absolute_path_from_relative(&PathBuf::from(&rel_path));
            let series = absolute
                .as_ref()
                .and_then(|path| project.find_image(path))
                .map(|image| image.active_series(&project.images.settings))
                .unwrap_or_else(|| project.images.settings.selected_series.unwrap_or(0));
            (absolute, series)
        };
        let in_viewer = self.split_in_viewer(&rel_path);
        let follow = self.split_follow();
        if !in_viewer {
            *self
                .viewport()
                .viewport_cache
                .focus
                .lock()
                .expect("Poisoned") = match (&absolute, select.or(region)) {
                (Some(path), Some(bbox)) if follow => Some((path.clone(), bbox.map(|v| v as f32))),
                _ => None,
            };
            self.image_list_controller
                .open_new_image_from_rel_path(&PathBuf::from(&rel_path));
        }

        let plane = {
            let plane = self.plane_filter.lock().expect("Poisoned");
            evanalyzer_app::results::PlaneFilter {
                z_stack: plane.selected_z_stack,
                t_stack: plane.selected_t_stack,
            }
        };
        let value_column = self
            .matrix_filter
            .lock()
            .expect("Poisoned")
            .as_ref()
            .map(|filter| filter.column.clone())
            // A value of each object (not, say, the image's count).
            .filter(is_chartable_column);
        let filter = ImageObjectsFilter {
            image_rel_path: rel_path.clone(),
            plane,
            value_column,
        };
        self.run_db_job(move |manager| {
            let objects = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.image_objects(&filter),
                }
            };
            let objects = match objects {
                Ok(objects) => {
                    log::debug!("Split view: {} objects of {rel_path}", objects.len());
                    objects
                }
                Err(err) => {
                    error!("Could not read the objects of {rel_path}: {err}");
                    Vec::new()
                }
            };
            let channels = absolute
                .as_ref()
                .and_then(|path| manager._app_state.get_image_source(path).ok())
                .and_then(|source| {
                    let meta = source.meta();
                    let info = meta
                        .series
                        .get(&series)
                        .or_else(|| meta.series.values().next())?;
                    Some(
                        info.channels
                            .values()
                            .map(|c| c.name.clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .unwrap_or_default();
            {
                let mut split = manager.split.lock().expect("Poisoned");
                if split.generation != generation {
                    return;
                }
                split.selected =
                    select.and_then(|bbox| objects.iter().position(|o| o.bbox_px == bbox));
                split.objects = Arc::new(objects);
            }
            manager.tracks_objects_loaded();
            manager.split_push_channels(channels);
            // Already in the viewer: flown to (else opened there).
            manager.split_push_selection(in_viewer);
            manager.split_push_overlay();
        });
    }

    /// An image was opened: the split's own (it was being opened for a row
    /// or a well) - its objects and the selection go on it now.
    fn split_image_opened(&self) {
        let (shown, image) = {
            let split = self.split.lock().expect("Poisoned");
            (split.shown, split.image.clone())
        };
        if shown
            && image
                .as_deref()
                .is_some_and(|rel| self.split_in_viewer(rel))
        {
            self.split_push_overlay();
            self.split_push_selection(false);
        }
    }

    /// The viewer shows the image `rel_path` (of the analysis).
    fn split_in_viewer(&self, rel_path: &str) -> bool {
        use evanalyzer_app::project::ProjectExt;
        let project = self._app_state.get_project();
        let absolute = project.get_image_absolute_path_from_relative(&PathBuf::from(rel_path));
        absolute.is_some() && project.tmp_settings.current_image == absolute
    }

    fn split_follow(&self) -> bool {
        self._app_state.load_app_settings().results_split.follow
    }

    /// The image's objects on it, in their class's color (or, Plate
    /// "Color by value", the heatmap's).
    pub(super) fn split_push_overlay(&self) {
        let by_value = self.split.lock().expect("Poisoned").by_value && self.in_plate();
        let (objects, image) = {
            let split = self.split.lock().expect("Poisoned");
            (split.objects.clone(), split.image.clone())
        };
        // Its image isn't the one in the viewer (yet): opening it takes
        // longer than reading its objects. They are kept and drawn once it
        // is open (`split_image_opened`) - not dropped, which left a row's
        // object unshown and the image unclickable.
        if image
            .as_deref()
            .is_some_and(|rel| !self.split_in_viewer(rel))
        {
            log::debug!("Split view: {image:?} not open yet - its objects wait for it");
            self.viewport().set_results_overlay(None);
            return;
        }
        let t = self.plane_filter.lock().expect("Poisoned").selected_t_stack;
        let trails = self.tracks_trails(image.as_deref(), t);
        if objects.is_empty() && trails.is_none() {
            self.viewport().set_results_overlay(None);
            return;
        }
        let class_colors: HashMap<i32, u32> = self
            .classes
            .lock()
            .expect("Poisoned")
            .iter()
            .map(|class| (class.id as i32, class.color))
            .collect();
        // No value of each object (e.g. the plate shows a count): by class.
        let by_value = by_value && objects.iter().any(|o| o.value.is_some());
        let gradient = by_value.then(|| {
            let schema = self
                .matrix_filter
                .lock()
                .expect("Poisoned")
                .as_ref()
                .map(|filter| filter.color_schema.clone())
                .unwrap_or_default();
            evanalyzer_app::results::color_scale_gradient(&schema)
        });
        let (lo, hi) = objects
            .iter()
            .filter_map(|o| o.value)
            .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let overlay = ResultsOverlay {
            objects: objects
                .iter()
                .map(|object| {
                    let rgb = match (&gradient, object.value) {
                        (Some(stops), Some(v)) if !stops.is_empty() => {
                            let t = if hi > lo { (v - lo) / (hi - lo) } else { 1.0 };
                            gradient_at(stops, t)
                        }
                        (Some(_), None) => 0x8A8A93,
                        _ => class_colors
                            .get(&object.class_id)
                            .copied()
                            .unwrap_or(0xFFFFFF),
                    };
                    (
                        object.bbox_px,
                        [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8],
                    )
                })
                .collect(),
            trails,
        };
        self.viewport().set_results_overlay(Some(overlay));
    }

    /// The viewer shows plane `(z, t)`: with the split on screen and its
    /// image in the viewer, the results follow (the Results T / Z) and the
    /// image's objects of that plane are read.
    fn split_plane_changed(self: &Arc<Self>, z: Option<u32>, t: u32) {
        let image = {
            let split = self.split.lock().expect("Poisoned");
            match split.shown {
                true => split.image.clone(),
                false => None,
            }
        };
        let Some(image) = image.filter(|rel| self.split_in_viewer(rel)) else {
            return;
        };
        let z_changed = {
            let mut plane = self.plane_filter.lock().expect("Poisoned");
            // A projection: the results stay on their Z plane.
            let z = z.unwrap_or(plane.selected_z_stack);
            let changed = (plane.selected_z_stack, plane.selected_t_stack) != (z, t);
            let z_changed = plane.selected_z_stack != z;
            plane.selected_z_stack = z;
            plane.selected_t_stack = t;
            if changed {
                let ui_weak = self.ui.clone();
                crate::helper::ui_thread::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let state = ui.global::<ResultsState>();
                        state.set_z_stack(z as i32);
                        state.set_t_stack(t as i32);
                    }
                })
                .ok();
            }
            changed.then_some(z_changed)
        };
        if let Some(z_changed) = z_changed {
            self.refresh_list();
            if z_changed && self.in_tracks() {
                self.tracks_load(image.clone());
            }
        }
        self.split_reload_objects(image, false);
    }

    /// The plate shows another value: "Color by value" colors by it - if it
    /// is a value of each object (not, say, a count), else the button is
    /// off - and the image shown is read again with it.
    pub(super) fn split_value_column_changed(self: &Arc<Self>) {
        let per_object = self
            .matrix_filter
            .lock()
            .expect("Poisoned")
            .as_ref()
            .is_some_and(|filter| is_chartable_column(&filter.column));
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ResultsState>().set_split_values(per_object);
            }
        })
        .ok();
        let image = {
            let split = self.split.lock().expect("Poisoned");
            match split.shown {
                true => split.image.clone(),
                false => None,
            }
        };
        if let Some(image) = image.filter(|rel| self.split_in_viewer(rel)) {
            self.split_reload_objects(image, true);
        }
    }

    /// The image's objects again: another plane (`keep_selected` false: the
    /// selected track's, if any, selected - else none) or another value
    /// (the same objects, the selected one stays).
    fn split_reload_objects(self: &Arc<Self>, rel_path: String, keep_selected: bool) {
        let generation = {
            let mut split = self.split.lock().expect("Poisoned");
            split.generation += 1;
            split.generation
        };
        let plane = {
            let plane = self.plane_filter.lock().expect("Poisoned");
            evanalyzer_app::results::PlaneFilter {
                z_stack: plane.selected_z_stack,
                t_stack: plane.selected_t_stack,
            }
        };
        let value_column = self
            .matrix_filter
            .lock()
            .expect("Poisoned")
            .as_ref()
            .map(|filter| filter.column.clone())
            .filter(is_chartable_column);
        let filter = ImageObjectsFilter {
            image_rel_path: rel_path.clone(),
            plane,
            value_column,
        };
        self.run_db_job(move |manager| {
            let objects = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.image_objects(&filter),
                }
            };
            let objects = objects.unwrap_or_else(|err| {
                error!("Could not read the objects of {rel_path}: {err}");
                Vec::new()
            });
            {
                let mut split = manager.split.lock().expect("Poisoned");
                if split.generation != generation {
                    return;
                }
                let selected_id = split
                    .selected
                    .filter(|_| keep_selected)
                    .and_then(|i| split.objects.get(i))
                    .map(|o| o.object_id.clone());
                split.selected =
                    selected_id.and_then(|id| objects.iter().position(|o| o.object_id == id));
                split.objects = Arc::new(objects);
            }
            manager.tracks_objects_loaded();
            manager.split_push_selection(false);
            manager.split_push_overlay();
        });
    }

    /// The viewer to time point `t` (the split's objects follow, see
    /// `split_plane_changed`).
    pub(super) fn split_show_time_point(&self, t: u32) {
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let channels = ui.global::<crate::ChannelState>();
            let last = ui.global::<crate::ImageMetaData>().get_nr_t_stacks() - 1;
            let t = (t as i32).clamp(0, last.max(0));
            if channels.get_selected_t_stack() != t {
                channels.set_selected_t_stack(t);
                channels.invoke_trigger_channel_state_changed();
            }
        })
        .ok();
    }

    /// The selected object's ring and card, "n / total" - and, `fly`, the
    /// view to it.
    pub(super) fn split_push_selection(&self, fly: bool) {
        let (title_image, object, total, index, channel, region) = {
            let split = self.split.lock().expect("Poisoned");
            let object = split.selected.and_then(|i| split.objects.get(i).cloned());
            (
                split.image.clone(),
                object,
                split.objects.len(),
                split.selected,
                split.channel,
                split.region,
            )
        };
        // The selected object: the viewer's yellow crosshair, as a list row
        // opened in the full view; a tile (no object): its box.
        let mark =
            |[x0, y0, x1, y1]: [u32; 4], paint_as_rectangle: bool| crate::ObjectHighlightBox {
                x_px: x0 as f32,
                y_px: y0 as f32,
                w_px: (x1.saturating_sub(x0) + 1) as f32,
                h_px: (y1.saturating_sub(y0) + 1) as f32,
                active: true,
                paint_as_rectangle,
            };
        let highlight = match (&object, region) {
            (Some(o), _) => mark(o.bbox_px, false),
            (None, Some(bbox)) => mark(bbox, true),
            _ => crate::ObjectHighlightBox::default(),
        };
        let classes: HashMap<i32, (String, u32)> = self
            .classes
            .lock()
            .expect("Poisoned")
            .iter()
            .map(|class| (class.id as i32, (class.name.clone(), class.color)))
            .collect();
        let rows = self.list_row_locations.lock().expect("Poisoned").len();
        let plate = self.in_plate();
        // Tracks: the selected track of the image's tracks.
        let track_pos = self.in_tracks().then(|| self.tracks_position());
        let image_name = title_image
            .as_deref()
            .map(|rel| {
                Path::new(rel)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel.to_string())
            })
            .unwrap_or_default();
        let well_image = self
            .split
            .lock()
            .expect("Poisoned")
            .well_images
            .iter()
            .find(|(_, rel)| Some(rel) == title_image.as_ref())
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        let sel = match &object {
            None => SplitObject::default(),
            Some(o) => {
                let (class_name, color) = classes
                    .get(&o.class_id)
                    .cloned()
                    .unwrap_or_else(|| ("Unclassified".to_string(), 0x8A8A93));
                SplitObject {
                    active: true,
                    x: o.bbox_px[0] as f32,
                    y: o.bbox_px[1] as f32,
                    w: (o.bbox_px[2].saturating_sub(o.bbox_px[0]) + 1) as f32,
                    h: (o.bbox_px[3].saturating_sub(o.bbox_px[1]) + 1) as f32,
                    class_name: class_name.into(),
                    class_color: bg_color_to_slint(color),
                    id: short_object_id(&o.object_id).into(),
                    area: format!("{} px", o.area_px).into(),
                    mean: o
                        .means
                        .get(channel)
                        .map(|m| format_chart_value(*m as f64))
                        .unwrap_or_else(|| "–".to_string())
                        .into(),
                    position: format!("{:.0}, {:.0}", o.centroid_px[0], o.centroid_px[1]).into(),
                }
            }
        };
        let fly_to = (fly && self.split_follow())
            .then_some(())
            .and(object.as_ref().map(|o| o.bbox_px).or(region));
        let reduce_motion = self.viewport().memory_settings().reduce_motion;
        let full = *self.viewport().full_image_size.lock().expect("Poisoned");
        let (_, _, _, v_w, v_h) = self.viewport().view();
        let target = fly_to.map(|bbox| {
            focus_view(
                v_w,
                v_h,
                full.0 as f32,
                full.1 as f32,
                bbox.map(|v| v as f32),
            )
        });

        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<ResultsState>();
            let well = state.get_active_well();
            state.set_split_title(
                if plate && !well.is_empty() && title_image.is_some() {
                    format!("Well {well} · {image_name}")
                } else {
                    image_name
                }
                .into(),
            );
            state.set_split_image_key(well_image.into());
            if let Some((pos, total)) = track_pos {
                state.set_split_pos(pos);
                state.set_split_total(total);
            } else if plate {
                state.set_split_pos(index.map_or(0, |i| i as i32 + 1));
                state.set_split_total(total as i32);
            } else {
                let row = state.get_list_selected_row();
                state.set_split_pos(if row >= 0 { row + 1 } else { 0 });
                state.set_split_total(rows as i32);
            }
            state.set_split_sel(sel);
            ui.global::<crate::ViewportObjectState>()
                .set_object_highlight(highlight);
            if let Some(target) = target
                && full.0 > 0
            {
                fly_to_view(&ui, target, (v_w, v_h), reduce_motion);
            }
        })
        .ok();
    }

    fn split_push_channels(&self, channels: Vec<String>) {
        let channel = self.split.lock().expect("Poisoned").channel;
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                let names: Vec<slint::SharedString> =
                    channels.into_iter().map(Into::into).collect();
                state.set_split_channel(channel.min(names.len().saturating_sub(1)) as i32);
                state.set_split_channels(ModelRc::from(Rc::new(VecModel::from(names))));
            }
        })
        .ok();
    }

    /// The filmstrip: the well's images, with the thumbnails made so far.
    fn split_push_thumbs(&self) {
        let thumbs: Vec<(String, Option<SharedPixelBuffer<Rgb8Pixel>>)> = {
            let split = self.split.lock().expect("Poisoned");
            split
                .well_images
                .iter()
                .map(|(name, rel)| (name.clone(), split.thumbs.get(rel).cloned()))
                .collect()
        };
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let thumbs: Vec<SplitThumb> = thumbs
                    .into_iter()
                    .map(|(key, buffer)| SplitThumb {
                        key: key.into(),
                        image: buffer.map(slint::Image::from_rgb8).unwrap_or_default(),
                    })
                    .collect();
                ui.global::<ResultsState>()
                    .set_split_thumbs(ModelRc::from(Rc::new(VecModel::from(thumbs))));
            }
        })
        .ok();
    }

    /// Makes the missing thumbnails, off the UI thread, from each image's
    /// smallest pyramid level; the filmstrip shows each as it's done.
    fn split_make_thumbs(self: &Arc<Self>, rel_paths: Vec<String>) {
        let missing: Vec<String> = {
            let split = self.split.lock().expect("Poisoned");
            rel_paths
                .into_iter()
                .filter(|rel| !split.thumbs.contains_key(rel))
                .collect()
        };
        if missing.is_empty() {
            return;
        }
        let (colors, series, root) = {
            use evanalyzer_app::project::ProjectExt;
            let project = self._app_state.get_project();
            (
                project
                    .get_channel_display_colors()
                    .into_iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
                project.images.settings.selected_series.unwrap_or(0),
                project.images.root.clone().unwrap_or_default(),
            )
        };
        let manager = self.clone();
        std::thread::spawn(move || {
            for rel in missing {
                let Some(buffer) =
                    make_thumb(&manager._app_state, &root.join(&rel), series, &colors)
                else {
                    continue;
                };
                let wanted = {
                    let mut split = manager.split.lock().expect("Poisoned");
                    if split.thumbs.len() >= THUMBS_KEPT {
                        split.thumbs.clear();
                    }
                    split.thumbs.insert(rel.clone(), buffer);
                    split.well_images.iter().any(|(_, r)| *r == rel)
                };
                if wanted {
                    manager.split_push_thumbs();
                }
            }
        });
    }
}

/// `stops` (the heatmap's gradient) at `t` (0…1), as 0xRRGGBB.
fn gradient_at(stops: &[u32], t: f64) -> u32 {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let i = (t.floor() as usize).min(stops.len() - 1);
    let j = (i + 1).min(stops.len() - 1);
    let f = t - i as f64;
    let mix = |shift: u32| {
        let a = ((stops[i] >> shift) & 0xFF) as f64;
        let b = ((stops[j] >> shift) & 0xFF) as f64;
        ((a + (b - a) * f).round() as u32) << shift
    };
    mix(16) | mix(8) | mix(0)
}

/// A filmstrip thumbnail of the image at `path`: its smallest level that
/// is at least [`THUMB_SIDE`] (or the smallest there is), channels in
/// their colors.
fn make_thumb(
    app_state: &UiState,
    path: &PathBuf,
    series: i32,
    colors: &std::collections::BTreeMap<i32, [f32; 3]>,
) -> Option<SharedPixelBuffer<Rgb8Pixel>> {
    let source = app_state.get_image_source(path).ok()?;
    let meta = source.meta();
    let (series, info) = meta
        .series
        .get_key_value(&series)
        .or_else(|| meta.series.iter().next())?;
    let (level, size) = info
        .resolutions
        .iter()
        .rev()
        .find(|(_, r)| r.width.max(r.height) >= THUMB_SIDE)
        .or_else(|| info.resolutions.iter().next_back())
        .map(|(l, r)| (*l, (r.width, r.height)))?;
    if size.0.max(size.1) > THUMB_MAX_READ_SIDE {
        return None;
    }
    let (width, height) = (size.0 as usize, size.1 as usize);
    let channels = source
        .read_tile(&evanalyzer_app::images::TileRequest {
            series: *series,
            resolution_idx: level,
            z_projection: evanalyzer_cfg::core_types::ZProjection::None,
            z_range: Some(0..=0),
            t_stack: 0,
            tile: evanalyzer_cfg::core_types::ImageTile {
                offset_x: 0,
                offset_y: 0,
                width,
                height,
            },
        })
        .ok()?;
    // Kept small: the strip's tile, cropped to its shape.
    let picture =
        evanalyzer_app::results::colorize_thumbnail(&channels, colors, (width, height), THUMB_KEPT);
    Some(SharedPixelBuffer::clone_from_slice(
        picture.as_raw(),
        picture.width(),
        picture.height(),
    ))
}

thread_local! {
    /// The fly running now (one at a time; a new one replaces it).
    static FLY: std::cell::RefCell<Option<slint::Timer>> = const { std::cell::RefCell::new(None) };
}

/// Moves the view to `(zoom, offset_x, offset_y)` in [`FLY_MS`] (eased),
/// or at once with reduced motion; the image is rendered for the end.
/// `(v_w, v_h)`: the viewport's size.
fn fly_to_view(
    ui: &AppWindow,
    (zoom, off_x, off_y): (f32, f32, f32),
    (v_w, v_h): (f32, f32),
    reduce_motion: bool,
) {
    let land = move |view: &ViewportSlintState| {
        view.set_zoom_factor(zoom);
        view.set_offset_x(off_x);
        view.set_offset_y(off_y);
        view.invoke_report_viewport_zoomed(zoom, off_x, off_y);
    };
    FLY.with(|fly| fly.borrow_mut().take());
    let view = ui.global::<ViewportSlintState>();
    let (z0, x0, y0) = (
        view.get_zoom_factor(),
        view.get_offset_x(),
        view.get_offset_y(),
    );
    if reduce_motion || z0 <= 0.0 {
        land(&view);
        return;
    }
    // The image point in the middle of the view moves straight, the zoom
    // changes evenly on a log scale.
    let (cx0, cy0) = ((v_w / 2.0 - x0) / z0, (v_h / 2.0 - y0) / z0);
    let (cx1, cy1) = ((v_w / 2.0 - off_x) / zoom, (v_h / 2.0 - off_y) / zoom);
    let start = std::time::Instant::now();
    let ui_weak = ui.as_weak();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16),
        move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let view = ui.global::<ViewportSlintState>();
            let t = (start.elapsed().as_secs_f32() * 1000.0 / FLY_MS).min(1.0);
            if t >= 1.0 {
                land(&view);
                // Stops this timer (dropped after this tick).
                FLY.with(|fly| fly.borrow_mut().take());
                return;
            }
            let e = 1.0 - (1.0 - t).powi(3);
            let z = z0 * (zoom / z0).powf(e);
            let (cx, cy) = (cx0 + (cx1 - cx0) * e, cy0 + (cy1 - cy0) * e);
            view.set_zoom_factor(z);
            view.set_offset_x(v_w / 2.0 - cx * z);
            view.set_offset_y(v_h / 2.0 - cy * z);
        },
    );
    FLY.with(|fly| *fly.borrow_mut() = Some(timer));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gradient_is_sampled_between_its_stops() {
        let stops = [0x000000, 0xFF0000, 0xFFFFFF];
        assert_eq!(gradient_at(&stops, 0.0), 0x000000);
        assert_eq!(gradient_at(&stops, 0.5), 0xFF0000);
        assert_eq!(gradient_at(&stops, 1.0), 0xFFFFFF);
        assert_eq!(gradient_at(&stops, 0.25), 0x800000);
        assert_eq!(gradient_at(&stops, 7.0), 0xFFFFFF, "clamped");
    }
}
