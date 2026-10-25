//! # viewport_object_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::AppWindow;
use crate::MarkerData;
use crate::PointSlint;
use crate::ToolState;
use crate::UiState;
use crate::ViewportObjectState;
use crate::editor::annotations_controller::AnnotationController;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::object_render::{ZOrder, pick};
use crate::editor::viewport_cache::ViewportCache;
use crate::editor::viewport_controller::ViewportController;
use crate::editor::viewport_controller::is_object_drawn;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use evanalyzer_app::images::{Erased, annotation_from_mask, erase_mask, stroke_mask, union_mask};
use evanalyzer_app::project::ObjectExt;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_cfg::core_types::{ImagePlane, ObjectClass, ObjectId};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use log::warn;
use slint::ComponentHandle;
use slint::Model;
use slint::ModelRc;
use slint::VecModel;
use std::sync::Arc;

/// Reads the two corner points a rectangle/oval-drag gesture is expected to
/// provide, without panicking if fewer were ever supplied. Currently the
/// Slint side always seeds both slots before the paint-finished callback
/// fires (`viewport.slint`), but that's a UI-side contract with no
/// corresponding guarantee here - a future change to how drag interactions
/// are handled (e.g. an interrupted drag) could violate it.
fn rect_corners(points: &ModelRc<PointSlint>) -> Option<(PointSlint, PointSlint)> {
    Some((points.row_data(0)?, points.row_data(1)?))
}

pub struct ViewPortObjectController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) viewport_cache: Arc<ViewportCache>,
    pub(crate) image_list_controller: Arc<ImagesListController>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
    /// Measures what was drawn, on the image file.
    pub(crate) annotation_controller: Arc<AnnotationController>,
    /// The brush or eraser stroke being drawn, in viewport coordinates.
    stroke: std::sync::Mutex<Vec<(f32, f32)>>,
}

impl ViewPortObjectController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        viewport_cache: Arc<ViewportCache>,
        image_list_controller: Arc<ImagesListController>,
        object_list_controller: Arc<ObjectListController>,
        classification_controller: Arc<ClassificationController>,
        annotation_controller: Arc<AnnotationController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
            viewport_cache,
            image_list_controller,
            object_list_controller,
            classification_controller,
            annotation_controller,
            stroke: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // object painting finished
            let manager = self.clone();
            ui.global::<ViewportObjectState>().on_object_paint_finished(
                move |points, tool_state, nr_of_polygon_points| {
                    match tool_state {
                        ToolState::Move => return,
                        ToolState::Select => return,
                        ToolState::PaintMarker => return,
                        // Strokes come through `brush_stroke`.
                        ToolState::Brush | ToolState::Eraser => return,
                        ToolState::PaintRectangle => manager.add_object_from_rect(&points),
                        ToolState::PaintOval => manager.add_oval_from_rect(&points),
                        ToolState::PaintPolygon => {
                            manager.add_polygon_from_rect(&points, nr_of_polygon_points)
                        }
                    };
                    manager.viewport_controller.trigger_image_redraw_objects();
                    manager.image_list_controller.sync_image_list_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
                    manager
                        .classification_controller
                        .sync_classification_to_slint();
                },
            );

            // Brush / eraser: the stroke's points as the mouse moves (0 down,
            // 1 move, 2 up); painted or erased on release.
            let manager = self.clone();
            let ui_weak = self.ui.clone();
            ui.global::<ViewportObjectState>()
                .on_brush_stroke(move |x, y, phase| {
                    manager.stroke_point(x, y, phase);
                    if phase != 2 {
                        return;
                    }
                    let Some(ui) = ui_weak.upgrade() else {
                        return;
                    };
                    let tool = ui.global::<ViewportObjectState>().get_act_tool_state();
                    let size = ui.global::<crate::LayoutState>().get_brush_size() as f32;
                    if manager.finish_stroke(tool, size) {
                        manager.viewport_controller.trigger_image_redraw_objects();
                        manager.image_list_controller.sync_image_list_to_slint();
                        manager.object_list_controller.sync_objects_to_slint();
                        manager
                            .classification_controller
                            .sync_classification_to_slint();
                    }
                });

            // In viewport clicked
            let manager = self.clone();
            ui.global::<ViewportObjectState>()
                .on_viewport_clicked(move |clicked_x, clicked_y| {
                    manager.find_object_from_clicked_coordinates(clicked_x, clicked_y);
                    manager
                        .object_list_controller
                        .sync_selected_object_to_slint(true);
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // object transparency
            let manager = self.clone();
            let debounce_timer = slint::Timer::default();
            ui.global::<ViewportObjectState>()
                .on_object_transparency_changed(move |transparency| {
                    let manager_in = manager.clone();
                    debounce_timer.start(
                        slint::TimerMode::SingleShot,
                        std::time::Duration::from_millis(5),
                        move || {
                            manager_in
                                .viewport_controller
                                .overlay_state
                                .write()
                                .expect("Poisoned")
                                .object_transparency = transparency;
                            manager_in
                                .viewport_controller
                                .trigger_image_redraw_objects();
                        },
                    );
                });

            // Marker placed (left-click in PaintMarker mode)
            let manager = self.clone();
            ui.global::<ViewportObjectState>()
                .on_marker_placed(move |screen_x, screen_y| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let (img_x, img_y) = {
                        let vp = manager
                            .viewport_controller
                            .viewport_state
                            .read()
                            .expect("Poisoned");
                        (
                            (screen_x - vp.offset_x) / vp.zoom,
                            (screen_y - vp.offset_y) / vp.zoom,
                        )
                    };
                    let label = manager.read_intensity_at_screen(screen_x, screen_y);
                    let state = ui.global::<ViewportObjectState>();
                    let current = state.get_markers();
                    let mut vec: Vec<MarkerData> = (0..current.row_count())
                        .filter_map(|i| current.row_data(i))
                        .collect();
                    vec.push(MarkerData {
                        image_x: img_x,
                        image_y: img_y,
                        label: label.into(),
                    });
                    state.set_markers(ModelRc::new(VecModel::from(vec)));
                });

            // Marker remove-at (right-click in PaintMarker mode)
            let manager = self.clone();
            ui.global::<ViewportObjectState>()
                .on_marker_remove_at(move |screen_x, screen_y| {
                    let Some(ui) = manager.ui.upgrade() else {
                        return;
                    };
                    let zoom = manager
                        .viewport_controller
                        .viewport_state
                        .read()
                        .expect("Poisoned")
                        .zoom;
                    let offset_x = manager
                        .viewport_controller
                        .viewport_state
                        .read()
                        .expect("Poisoned")
                        .offset_x;
                    let offset_y = manager
                        .viewport_controller
                        .viewport_state
                        .read()
                        .expect("Poisoned")
                        .offset_y;
                    let state = ui.global::<ViewportObjectState>();
                    let current = state.get_markers();
                    let threshold = 12.0_f32;
                    let closest = (0..current.row_count())
                        .filter_map(|i| current.row_data(i).map(|m| (i, m)))
                        .map(|(i, m)| {
                            let mx = m.image_x * zoom + offset_x;
                            let my = m.image_y * zoom + offset_y;
                            let dist = ((mx - screen_x).powi(2) + (my - screen_y).powi(2)).sqrt();
                            (i, dist)
                        })
                        .filter(|(_, d)| *d <= threshold)
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some((idx, _)) = closest {
                        let mut vec: Vec<MarkerData> = (0..current.row_count())
                            .filter_map(|i| current.row_data(i))
                            .collect();
                        vec.remove(idx);
                        state.set_markers(ModelRc::new(VecModel::from(vec)));
                    }
                });
        }
    }

    fn read_intensity_at_screen(&self, screen_x: f32, screen_y: f32) -> String {
        let data_tmp = self
            .viewport_cache
            .active_high_res_data
            .read()
            .expect("Poisoned");
        let Some((image_data, ctx)) = &*data_tmp else {
            return String::new();
        };
        let local_x = (screen_x - ctx.draw_x) / (ctx.zoomed_w / ctx.image_w as f32);
        let local_y = (screen_y - ctx.draw_y) / (ctx.zoomed_h / ctx.image_h as f32);
        if local_x < 0.0
            || local_x >= ctx.image_w as f32
            || local_y < 0.0
            || local_y >= ctx.image_h as f32
        {
            return String::new();
        }
        let idx = (local_y as usize * ctx.image_w) + local_x as usize;
        let mut values = Vec::new();
        for channel in image_data.iter() {
            if !channel.is_rgb {
                if let Some(raw_val) = channel.value_at(idx) {
                    let scaled = crate::editor::view_channel::raw_value(raw_val, ctx.bit_depth);
                    values.push(format!("{}: {:.0}", channel.name, scaled));
                }
            }
        }
        values.join(" | ")
    }

    pub fn find_object_from_clicked_coordinates(&self, click_x: f32, click_y: f32) {
        let vp = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();
        let fx = (click_x - vp.offset_x) / vp.zoom;
        let fy = (click_y - vp.offset_y) / vp.zoom;

        // `as u32` saturates negatives to 0, which would pick objects at the image
        // origin when clicking outside the image on the left/top.
        let clicked_object_id = if fx < 0.0 || fy < 0.0 {
            None
        } else {
            let (x1, y1) = (fx as u32, fy as u32);
            let project = self.app_state.get_project();
            let z = ZOrder::new(&*project);
            let hide_unclassified = project.hide_unclassified_objects();

            // As drawn (`object_render::pick`); objects that aren't drawn
            // (hidden class, hidden unclassified) can't be picked.
            let objects: Vec<_> = project
                .get_objects()
                .into_iter()
                .flatten()
                .chain(project.get_preview_objects())
                .filter(|o| is_object_drawn(&*project, hide_unclassified, &o.object_class))
                .collect();
            pick(
                objects
                    .iter()
                    .map(|o| (z.key(&o.object_class, false), o.is_part_of(x1, y1))),
            )
            .map(|i| objects[i].id.clone())
        };

        let mut project = self.app_state.get_project_write();
        project.set_selected_object(clicked_object_id);
    }

    pub fn add_object_from_rect(&self, points: &ModelRc<PointSlint>) {
        let Some((p0, p1)) = rect_corners(points) else {
            warn!(
                "add_object_from_rect: expected 2 points, got {}",
                points.row_count()
            );
            return;
        };
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();

        let x1 = (p0.x - view_port_state.offset_x) / (view_port_state.zoom);
        let y1 = (p0.y - view_port_state.offset_y) / (view_port_state.zoom);
        let x2 = (p1.x - view_port_state.offset_x) / (view_port_state.zoom);
        let y2 = (p1.y - view_port_state.offset_y) / (view_port_state.zoom);

        // Create mask
        let min_x = x1.min(x2) as u32;
        let max_x = x1.max(x2) as u32;
        let min_y = y1.min(y2) as u32;
        let max_y = y1.max(y2) as u32;

        let bbox = [min_x, min_y, max_x, max_y];

        let mut mask_data: BitVec<u64, Lsb0> = BitVec::new();
        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;
        mask_data.resize(width * height, false);
        mask_data.fill(true);

        self.add_to_object_list(mask_data, bbox);
    }

    pub fn add_oval_from_rect(&self, points: &ModelRc<PointSlint>) {
        let Some((p0, p1)) = rect_corners(points) else {
            warn!(
                "add_oval_from_rect: expected 2 points, got {}",
                points.row_count()
            );
            return;
        };
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();

        let x1 = (p0.x - view_port_state.offset_x) / (view_port_state.zoom);
        let y1 = (p0.y - view_port_state.offset_y) / (view_port_state.zoom);
        let x2 = (p1.x - view_port_state.offset_x) / (view_port_state.zoom);
        let y2 = (p1.y - view_port_state.offset_y) / (view_port_state.zoom);

        let min_x = x1.min(x2) as u32;
        let max_x = x1.max(x2) as u32;
        let min_y = y1.min(y2) as u32;
        let max_y = y1.max(y2) as u32;

        let bbox = [min_x, min_y, max_x, max_y];

        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;

        let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, width * height);

        let width = (max_x - min_x + 1) as i32;
        let height = (max_y - min_y + 1) as i32;

        // Semi-axes span from pixel 0 to pixel (width-1), so center and radius
        // are both (dim-1)/2 - this makes the ellipse symmetric: pixels at both
        // ends map to exactly ±1 in normalized coordinates.
        let rx = (width as f64 - 1.0) / 2.0;
        let ry = (height as f64 - 1.0) / 2.0;
        let cx = rx;
        let cy = ry;

        for y in 0..height {
            for x in 0..width {
                let dx = (x as f64 - cx) / rx;
                let dy = (y as f64 - cy) / ry;
                if (dx * dx) + (dy * dy) <= (1.0 + 1e-6) {
                    mask_data.set((y * width + x) as usize, true);
                }
            }
        }

        self.add_to_object_list(mask_data, bbox);
    }

    pub fn add_polygon_from_rect(&self, points: &ModelRc<PointSlint>, nr_of_points: i32) {
        if nr_of_points < 3 {
            return;
        }

        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();

        let vertices: Vec<(f32, f32)> = (0..nr_of_points as usize)
            .filter_map(|i| points.row_data(i))
            .map(|p| {
                let x = (p.x - view_port_state.offset_x) / view_port_state.zoom;
                let y = (p.y - view_port_state.offset_y) / view_port_state.zoom;
                (x, y)
            })
            .collect();

        if vertices.len() < 3 {
            return;
        }

        let min_x = vertices
            .iter()
            .map(|p| p.0)
            .fold(f32::INFINITY, f32::min)
            .max(0.0) as u32;
        let max_x = vertices
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max) as u32;
        let min_y = vertices
            .iter()
            .map(|p| p.1)
            .fold(f32::INFINITY, f32::min)
            .max(0.0) as u32;
        let max_y = vertices
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max) as u32;

        if max_x < min_x || max_y < min_y {
            return;
        }

        let bbox = [min_x, min_y, max_x, max_y];
        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;

        let mut mask_data: BitVec<u64, Lsb0> = BitVec::repeat(false, width * height);

        let n = vertices.len();
        for row in 0..height {
            let y = min_y as f32 + row as f32;
            let mut intersections: Vec<f32> = Vec::new();

            for i in 0..n {
                let (x1, y1) = vertices[i];
                let (x2, y2) = vertices[(i + 1) % n];
                // One endpoint strictly above scanline, the other at or below (even-odd fill)
                if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                    let t = (y - y1) / (y2 - y1);
                    intersections.push(x1 + t * (x2 - x1));
                }
            }

            intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut i = 0;
            while i + 1 < intersections.len() {
                let x_start = (intersections[i].ceil() as u32).max(min_x);
                let x_end = (intersections[i + 1].floor() as u32).min(max_x);
                for x in x_start..=x_end {
                    let col = (x - min_x) as usize;
                    mask_data.set(row * width + col, true);
                }
                i += 2;
            }
        }

        self.add_to_object_list(mask_data, bbox);
    }

    /// A point of the brush or eraser stroke (`phase` 0 starts it).
    pub fn stroke_point(&self, x: f32, y: f32, phase: i32) {
        let mut stroke = self.stroke.lock().expect("Poisoned");
        if phase == 0 {
            stroke.clear();
        }
        // Points closer than a screen pixel add nothing.
        if let Some(&(lx, ly)) = stroke.last()
            && (lx - x).abs() < 1.0
            && (ly - y).abs() < 1.0
        {
            return;
        }
        stroke.push((x, y));
    }

    /// Paints (`Brush`: into the selected hand-drawn object, else a new
    /// object of the selected class) or erases (`Eraser`: from the hand-drawn
    /// objects) the stroke, `brush_size` screen pixels wide - one undo step.
    /// Whether something changed.
    pub fn finish_stroke(&self, tool: ToolState, brush_size: f32) -> bool {
        let points = std::mem::take(&mut *self.stroke.lock().expect("Poisoned"));
        let view = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();
        if points.is_empty() || view.zoom <= 0.0 {
            return false;
        }
        let image_points: Vec<(f32, f32)> = points
            .iter()
            .map(|(x, y)| {
                (
                    (x - view.offset_x) / view.zoom,
                    (y - view.offset_y) / view.zoom,
                )
            })
            .collect();
        let (_, read_context) = self.viewport_cache.get_image_references();
        let Some((mask, bbox)) = stroke_mask(
            &image_points,
            brush_size / 2.0 / view.zoom,
            read_context.full_image_w,
            read_context.full_image_h,
        ) else {
            return false;
        };
        self.app_state.begin_undo_step();
        match tool {
            ToolState::Brush => {
                if self.paint_into_selected(&mask, bbox) {
                    return true;
                }
                let before = self
                    .app_state
                    .get_project()
                    .get_objects()
                    .map_or(0, |o| o.len());
                self.add_to_object_list(mask, bbox);
                self.app_state
                    .get_project()
                    .get_objects()
                    .map_or(0, |o| o.len())
                    != before
            }
            ToolState::Eraser => self.erase(&mask, bbox),
            _ => false,
        }
    }

    /// Adds the stroke to the selected object, if it is a hand-drawn one
    /// (measured again, keeping its id and classes). Whether it was.
    fn paint_into_selected(&self, stroke: &BitVec<u64, Lsb0>, stroke_bbox: [u32; 4]) -> bool {
        let (_, read_context) = self.viewport_cache.get_image_references();
        if read_context.full_image_w == 0 {
            return false; // no image shown
        }
        let painted = {
            let project = self.app_state.get_project();
            let Some(selected) = project.get_selected_object() else {
                return false;
            };
            let Some(index) = project
                .get_objects()
                .unwrap_or(&[])
                .iter()
                .position(|object| object.id == selected.id)
            else {
                return false;
            };
            let object = &project.get_objects().unwrap_or(&[])[index];
            let (mask, bbox) = union_mask(&object.mask_data, object.bbox, stroke, stroke_bbox);
            let class = object
                .object_class
                .iter()
                .next()
                .copied()
                .unwrap_or(ObjectClass::Unset);
            // On the plane it was drawn on; measured on the file (below).
            let mut painted = annotation_from_mask(
                read_context.full_image_w,
                read_context.full_image_h,
                mask,
                bbox,
                plane_of(object),
                class,
            );
            painted.id = object.id.clone();
            painted.segmentation_class = object.segmentation_class;
            painted.object_class = object.object_class.clone();
            painted.exclude_from_training = object.exclude_from_training;
            (index, painted)
        };
        let id = painted.1.id.clone();
        let mut project = self.app_state.get_project_write();
        if let Some(series) = project.get_selected_image_series_mut() {
            series.objects[painted.0] = painted.1;
        }
        drop(project);
        self.app_state.mark_dirty();
        self.annotation_controller.measure_in_background(vec![id]);
        true
    }

    /// Takes the pixels of `stroke` out of the hand-drawn objects of the open
    /// image: an object left empty is deleted, the others measured again
    /// (keeping their id and classes).
    fn erase(&self, stroke: &BitVec<u64, Lsb0>, stroke_bbox: [u32; 4]) -> bool {
        let (_, read_context) = self.viewport_cache.get_image_references();
        if read_context.full_image_w == 0 {
            return false; // no image shown
        }
        let changes: Vec<(usize, Option<_>)> = {
            let project = self.app_state.get_project();
            project
                .get_objects()
                .unwrap_or(&[])
                .iter()
                .enumerate()
                .filter_map(|(i, object)| {
                    match erase_mask(&object.mask_data, object.bbox, stroke, stroke_bbox) {
                        Erased::Untouched => None,
                        Erased::Empty => Some((i, None)),
                        Erased::Rest(mask, bbox) => {
                            let class = object
                                .object_class
                                .iter()
                                .next()
                                .copied()
                                .unwrap_or(ObjectClass::Unset);
                            // Measured on the file (below).
                            let mut rest = annotation_from_mask(
                                read_context.full_image_w,
                                read_context.full_image_h,
                                mask,
                                bbox,
                                plane_of(object),
                                class,
                            );
                            rest.id = object.id.clone();
                            rest.segmentation_class = object.segmentation_class;
                            rest.object_class = object.object_class.clone();
                            rest.exclude_from_training = object.exclude_from_training;
                            Some((i, Some(rest)))
                        }
                    }
                })
                .collect()
        };
        if changes.is_empty() {
            return false;
        }
        let changed: Vec<ObjectId> = changes
            .iter()
            .filter_map(|(_, rest)| rest.as_ref().map(|r| r.id.clone()))
            .collect();
        let mut project = self.app_state.get_project_write();
        if let Some(series) = project.get_selected_image_series_mut() {
            // Backwards, so the indices stay right while removing.
            for (i, rest) in changes.into_iter().rev() {
                match rest {
                    Some(rest) => series.objects[i] = rest,
                    None => {
                        series.objects.remove(i);
                    }
                }
            }
        }
        drop(project);
        self.app_state.mark_dirty();
        self.annotation_controller.measure_in_background(changed);
        true
    }

    fn add_to_object_list(&self, mask_data: BitVec<u64, Lsb0>, bbox: [u32; 4]) {
        // Each drawn object is an undo step of its own.
        self.app_state.begin_undo_step();
        let (_, read_context) = self.viewport_cache.get_image_references();
        if read_context.full_image_w == 0 {
            return; // no image shown
        }
        let (object_class, plane) = {
            let project = self.app_state.get_project();
            (project.get_selected_object_class(), shown_plane(&project))
        };

        if object_class == ObjectClass::Unset {
            // Tell the user to select an object class
            return;
        }

        // On the plane shown; measured on the image file (all channels,
        // full resolution), not on the pixels shown at this zoom.
        let object = annotation_from_mask(
            read_context.full_image_w,
            read_context.full_image_h,
            mask_data,
            bbox,
            plane,
            object_class,
        );
        self.app_state.get_project_write().add_object(&object);
        self.app_state.mark_dirty();
        self.annotation_controller
            .measure_in_background(vec![object.id.clone()]);
    }
}

/// The plane of an annotation (it is edited and measured on it).
fn plane_of(object: &ObjectMetricSettings) -> ImagePlane {
    ImagePlane {
        z: object.z_stack,
        c: object.c_stack,
        t: object.t_stack,
    }
}

/// The plane the viewer shows: the first slice of the z range, the frame,
/// the selected channel.
fn shown_plane(project: &ProjectWithRuntime) -> ImagePlane {
    ImagePlane {
        z: project
            .get_z_stack()
            .and_then(|z| z.z_range.as_ref().map(|r| *r.start()))
            .unwrap_or(0),
        c: project.get_selected_image_channel_idx(),
        t: project.get_t_stack().map_or(0, |t| t.t_stack),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::histogram_controller::HistogramController;
    use crate::editor::image_meta_controller::ImageMetaController;
    use crate::editor::test_support::{project_with_one_image, test_ui_state_with_project};
    use crate::editor::viewport_cache::{ReadContext, ViewportCache};
    use bitvec::prelude::*;
    use evanalyzer_app::images::ImageChannel;
    use evanalyzer_app::images::ManagedImage;
    use evanalyzer_app::images::Point2d;
    use evanalyzer_app::prelude::classification_ext::ClassificationExt;
    use evanalyzer_cfg::core_types::ImageDimensionality;
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use kornia_image::Image;

    fn make_controller() -> (
        Arc<UiState>,
        Arc<ViewPortObjectController>,
        Arc<ViewportCache>,
    ) {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let (controller, viewport_cache) = controller_for(slint::Weak::default(), &ui_state);
        (ui_state, controller, viewport_cache)
    }

    /// A controller with all its collaborators, on `ui` (`Weak::default()`
    /// for none).
    fn controller_for(
        ui: slint::Weak<AppWindow>,
        ui_state: &Arc<UiState>,
    ) -> (Arc<ViewPortObjectController>, Arc<ViewportCache>) {
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let viewport_cache = Arc::new(ViewportCache::new(ui_state.clone()));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_list_controller = Arc::new(ImagesListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
            Arc::new(HistogramController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
            Arc::new(ImageMetaController::new(
                ui.clone(),
                ui_state.clone(),
                viewport_controller.clone(),
            )),
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
        let classification_controller = Arc::new(ClassificationController::new(
            ui.clone(),
            ui_state.clone(),
            object_list_controller.clone(),
            viewport_controller.clone(),
        ));
        let annotation_controller = Arc::new(AnnotationController::new(
            ui.clone(),
            ui_state.clone(),
            object_list_controller.clone(),
            classification_controller.clone(),
            viewport_controller.clone(),
        ));
        let controller = Arc::new(ViewPortObjectController::new(
            ui,
            ui_state.clone(),
            viewport_controller,
            viewport_cache.clone(),
            image_list_controller,
            object_list_controller,
            classification_controller,
            annotation_controller,
        ));
        (controller, viewport_cache)
    }

    // -- find_object_from_clicked_coordinates --------------------------------------

    fn object_with_bbox(id: u128, bbox: [u32; 4]) -> ObjectMetricSettings {
        let width = (bbox[2] - bbox[0] + 1) as usize;
        let height = (bbox[3] - bbox[1] + 1) as usize;
        ObjectMetricSettings {
            id: ObjectId(id),
            bbox,
            mask_data: bitvec![u64, Lsb0; 1; width * height],
            ..Default::default()
        }
    }

    #[test]
    fn clicking_inside_an_objects_mask_selects_it() {
        let (ui_state, controller, _) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            // Unclassified objects are hidden (and so not pickable) by default.
            project.toggle_hide_unclassified_objects();
            project.add_object(&object_with_bbox(1, [10, 10, 15, 15]));
        }
        // Default viewport state: zoom=1.0, offset=(0,0) - click coordinates
        // map 1:1 to image coordinates.

        controller.find_object_from_clicked_coordinates(12.0, 12.0);

        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(1))
        );
    }

    #[test]
    fn clicking_outside_every_objects_bbox_clears_the_selection() {
        let (ui_state, controller, _) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            project.add_object(&object_with_bbox(1, [10, 10, 15, 15]));
            project.set_selected_object(Some(ObjectId(1)));
        }

        controller.find_object_from_clicked_coordinates(0.0, 0.0);

        assert_eq!(ui_state.get_project().get_selected_object_id(), None);
    }

    /// Two registered classes - the first in the list is drawn (and picked)
    /// on top - and two objects of them overlapping at (12, 12): `1` of the
    /// top class `a`, `2` of class `b`, added after it.
    fn two_overlapping_classified_objects(ui_state: &Arc<UiState>) -> (ObjectClass, ObjectClass) {
        let mut project = ui_state.get_project_write();
        let mut class = |name: &str| {
            project.classification.add_class(Class {
                id: 0,
                color: 0xff0000,
                name: name.into(),
                notes: String::new(),
            })
        };
        let (a, b) = (class("a"), class("b"));
        let mut top = object_with_bbox(1, [10, 10, 15, 15]);
        top.object_class.insert(a);
        let mut below = object_with_bbox(2, [11, 11, 16, 16]);
        below.object_class.insert(b);
        project.add_object(&top);
        project.add_object(&below);
        (a, b)
    }

    #[test]
    fn clicking_overlapping_objects_selects_the_one_drawn_on_top() {
        let (ui_state, controller, _) = make_controller();
        two_overlapping_classified_objects(&ui_state);

        controller.find_object_from_clicked_coordinates(12.0, 12.0);

        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(1)),
            "the first class in the list is on top, although its object came first"
        );
    }

    #[test]
    fn clicking_overlapping_objects_prefers_the_selected_class() {
        let (ui_state, controller, _) = make_controller();
        let (_, b) = two_overlapping_classified_objects(&ui_state);
        ui_state.get_project_write().set_selected_object_class(b);

        controller.find_object_from_clicked_coordinates(12.0, 12.0);

        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(2))
        );
    }

    /// An object that isn't drawn must not take the click from the visible
    /// one underneath.
    #[test]
    fn clicking_never_selects_an_object_of_a_hidden_class() {
        let (ui_state, controller, _) = make_controller();
        let (a, _) = two_overlapping_classified_objects(&ui_state);
        ui_state.get_project_write().toggle_class_visibility(a);

        controller.find_object_from_clicked_coordinates(12.0, 12.0);
        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(2)),
            "the visible object underneath is selected"
        );

        // Only the hidden object at (10, 10): nothing to select.
        controller.find_object_from_clicked_coordinates(10.0, 10.0);
        assert_eq!(ui_state.get_project().get_selected_object_id(), None);
    }

    #[test]
    fn clicking_never_selects_a_hidden_unclassified_object() {
        let (ui_state, controller, _) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            assert!(project.hide_unclassified_objects(), "the default");
            project.add_object(&object_with_bbox(1, [10, 10, 15, 15]));
        }

        controller.find_object_from_clicked_coordinates(12.0, 12.0);
        assert_eq!(ui_state.get_project().get_selected_object_id(), None);

        ui_state
            .get_project_write()
            .toggle_hide_unclassified_objects();
        controller.find_object_from_clicked_coordinates(12.0, 12.0);
        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(1))
        );
    }

    // -- add_object_from_rect ------------------------------------------------------

    fn points(pairs: &[(f32, f32)]) -> ModelRc<PointSlint> {
        let items: Vec<PointSlint> = pairs.iter().map(|&(x, y)| PointSlint { x, y }).collect();
        ModelRc::new(VecModel::from(items))
    }

    /// Seeds `viewport_cache.active_high_res_data` with a single flat 20x20
    /// grayscale channel and a matching `ReadContext` (zoom=1, no tile
    /// offset), so `add_to_object_list`'s pixel-by-pixel intensity sampling
    /// has real, in-bounds data to read.
    fn seed_image_cache(viewport_cache: &ViewportCache) {
        let size = kornia_image::ImageSize {
            width: 20,
            height: 20,
        };
        let image = Image::<f32, 1>::new(size, vec![0.5f32; 20 * 20]).unwrap();
        let container = evanalyzer_app::images::ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        });
        let channel = ImageChannel {
            image: Arc::new(container),
            color: [1.0, 1.0, 1.0],
            is_visible: true,
            c_stack: 0,
            name: "ch0".to_string(),
            is_rgb: false,
        };
        *viewport_cache.active_high_res_data.write().unwrap() = Some((
            Arc::new(vec![crate::editor::view_channel::ViewChannel::wrap(
                &channel,
            )]),
            ReadContext {
                zoom: 1.0,
                image_w: 20,
                image_h: 20,
                full_image_w: 20,
                full_image_h: 20,
                bit_depth: 8,
                ..Default::default()
            },
        ));
    }

    #[test]
    fn add_object_from_rect_with_no_selected_class() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_object_from_rect(&points(&[(2.0, 2.0), (5.0, 5.0)]));

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        // No class was selected, we do not allow object creation without class
        assert_eq!(objects.len(), 0);
    }

    #[test]
    fn add_object_from_rect_creates_an_object_spanning_the_two_corner_points() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        let new_class = controller
            .app_state
            .get_project_write()
            .classification
            .add_class(Class {
                id: 1,
                color: 0xff0000,
                name: "cl1".into(),
                notes: "note".into(),
            });

        controller
            .app_state
            .get_project_write()
            .set_selected_object_class(new_class);

        controller.add_object_from_rect(&points(&[(2.0, 2.0), (5.0, 5.0)]));

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].object_class.contains(&new_class), true);
        assert_eq!(objects[0].object_class.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 5, 5]);
        // A rectangle mask fills every pixel in its bbox.
        assert_eq!(objects[0].area, 4 * 4);
    }

    /// A controller on a seeded 20 x 20 image with class "cl1" selected.
    fn controller_with_a_class() -> (Arc<UiState>, Arc<ViewPortObjectController>) {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);
        let class = controller
            .app_state
            .get_project_write()
            .classification
            .add_class(Class {
                id: 1,
                color: 0xff0000,
                name: "cl1".into(),
                notes: "".into(),
            });
        controller
            .app_state
            .get_project_write()
            .set_selected_object_class(class);
        (ui_state, controller)
    }

    fn stroke(controller: &ViewPortObjectController, points: &[(f32, f32)]) {
        for (i, (x, y)) in points.iter().enumerate() {
            controller.stroke_point(*x, *y, if i == 0 { 0 } else { 1 });
        }
    }

    #[test]
    fn a_brush_stroke_paints_a_new_object_as_wide_as_the_brush() {
        let (ui_state, controller) = controller_with_a_class();

        stroke(&controller, &[(4.0, 10.0), (12.0, 10.0)]);
        assert!(controller.finish_stroke(ToolState::Brush, 4.0));

        let project = ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1);
        // Radius 2 around the line from x 4 to 12 at y 10.
        assert_eq!(objects[0].bbox, [2, 8, 14, 12]);
        assert_eq!(objects[0].object_class.len(), 1);
    }

    #[test]
    fn the_eraser_cuts_objects_and_deletes_what_it_empties() {
        let (ui_state, controller) = controller_with_a_class();
        // Two rectangles: 2..9 x 2..5 and 12..13 x 12..13.
        controller.add_object_from_rect(&points(&[(2.0, 2.0), (9.0, 5.0)]));
        controller.add_object_from_rect(&points(&[(12.0, 12.0), (13.0, 13.0)]));
        let (first_id, first_area) = {
            let project = ui_state.get_project();
            let objects = project.get_objects().unwrap();
            (objects[0].id.clone(), objects[0].area)
        };

        // Wipe the left half of the first, and all of the second.
        stroke(&controller, &[(2.0, 2.0), (2.0, 5.0)]);
        assert!(controller.finish_stroke(ToolState::Eraser, 8.0));
        stroke(&controller, &[(12.5, 12.5)]);
        assert!(controller.finish_stroke(ToolState::Eraser, 10.0));

        let project = ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1, "the emptied one is gone");
        assert_eq!(objects[0].id, first_id, "the cut one keeps its id");
        assert!(objects[0].area < first_area);
        assert!(
            objects[0].bbox[0] > 2,
            "its box shrank: {:?}",
            objects[0].bbox
        );
    }

    #[test]
    fn a_brush_stroke_paints_into_the_selected_hand_drawn_object() {
        let (ui_state, controller) = controller_with_a_class();
        controller.add_object_from_rect(&points(&[(2.0, 2.0), (4.0, 4.0)]));
        let id = ui_state.get_project().get_objects().unwrap()[0].id.clone();
        ui_state
            .get_project_runtime_write()
            .set_selected_object(Some(id.clone()));

        stroke(&controller, &[(10.0, 3.0)]);
        assert!(controller.finish_stroke(ToolState::Brush, 2.0));

        let project = ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1, "no new object");
        assert_eq!(objects[0].id, id);
        assert_eq!(objects[0].bbox, [2, 2, 11, 4]);
        assert!(objects[0].area > 9);
    }

    #[test]
    fn each_stroke_is_an_undo_step_of_its_own() {
        let (ui_state, controller) = controller_with_a_class();
        stroke(&controller, &[(3.0, 3.0)]);
        controller.finish_stroke(ToolState::Brush, 2.0);
        stroke(&controller, &[(12.0, 12.0)]);
        controller.finish_stroke(ToolState::Brush, 2.0);
        assert_eq!(ui_state.get_project().get_objects().unwrap().len(), 2);

        assert!(ui_state.undo());
        assert_eq!(ui_state.get_project().get_objects().unwrap().len(), 1);
        assert!(ui_state.undo());
        assert_eq!(
            ui_state.get_project().get_objects().map_or(0, |o| o.len()),
            0
        );
    }

    #[test]
    fn erasing_where_there_is_nothing_changes_nothing() {
        let (_ui_state, controller) = controller_with_a_class();
        controller.add_object_from_rect(&points(&[(2.0, 2.0), (4.0, 4.0)]));
        stroke(&controller, &[(15.0, 15.0)]);
        assert!(!controller.finish_stroke(ToolState::Eraser, 2.0));
    }

    #[test]
    fn painting_an_annotation_updates_the_class_counts_right_away() {
        use crate::editor::test_support::{test_ui_window, ui_state_with_window};
        use crate::helper::ui_thread::drain_ui_queue;
        use crate::{ClassificationState, ToolState, ViewportObjectState};
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, project_with_one_image());
        let nuclei = {
            use evanalyzer_app::prelude::classification_ext::ClassificationExt;
            let mut project = ui_state.get_project_write();
            let id = project.classification.add_class(
                evanalyzer_cfg::settings::classification_settings::Class {
                    name: "Nuclei".into(),
                    ..Default::default()
                },
            );
            project.set_selected_object_class(id);
            id
        };
        assert!(matches!(
            nuclei,
            evanalyzer_cfg::core_types::ObjectClass::Global(_)
        ));
        let (controller, viewport_cache) = controller_for(ui.as_weak(), &ui_state);
        seed_image_cache(&viewport_cache);
        controller.attach_callbacks();
        let total = || {
            ui.global::<ClassificationState>()
                .get_total_visible_objects()
        };
        assert_eq!(total(), 0);

        ui.global::<ViewportObjectState>()
            .invoke_object_paint_finished(
                points(&[(2.0, 2.0), (5.0, 5.0)]),
                ToolState::PaintRectangle,
                0,
            );
        drain_ui_queue();

        assert_eq!(total(), 1, "counted without switching tabs");
    }

    #[test]
    fn add_object_from_rect_without_cached_image_data_adds_nothing() {
        // No `seed_image_cache` call - `viewport_cache` is empty, matching
        // the state right after opening the app before any tile has loaded.
        let (ui_state, controller, _) = make_controller();

        controller.add_object_from_rect(&points(&[(2.0, 2.0), (5.0, 5.0)]));

        let project = ui_state.get_project();
        assert_eq!(project.get_objects().map(|o| o.len()), Some(0));
    }

    #[test]
    fn add_object_from_rect_with_fewer_than_two_points_is_a_no_op_not_a_panic() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_object_from_rect(&points(&[]));
        controller.add_object_from_rect(&points(&[(2.0, 2.0)]));

        let project = ui_state.get_project();
        assert_eq!(project.get_objects().map(|o| o.len()), Some(0));
    }

    // -- add_oval_from_rect / add_polygon_from_rect (no cached data) --------------

    #[test]
    fn add_oval_from_rect_without_cached_image_data_adds_nothing() {
        let (ui_state, controller, _) = make_controller();

        controller.add_oval_from_rect(&points(&[(2.0, 2.0), (8.0, 8.0)]));

        let project = ui_state.get_project();
        assert_eq!(project.get_objects().map(|o| o.len()), Some(0));
    }

    #[test]
    fn add_oval_from_rect_with_fewer_than_two_points_is_a_no_op_not_a_panic() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_oval_from_rect(&points(&[]));
        controller.add_oval_from_rect(&points(&[(2.0, 2.0)]));

        let project = ui_state.get_project();
        assert_eq!(project.get_objects().map(|o| o.len()), Some(0));
    }

    #[test]
    fn add_polygon_from_rect_with_fewer_than_three_points_is_a_no_op() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_polygon_from_rect(&points(&[(2.0, 2.0), (5.0, 5.0)]), 2);

        let project = ui_state.get_project();
        assert_eq!(project.get_objects().map(|o| o.len()), Some(0));
    }

    #[test]
    fn add_polygon_without_object_class_must_fail() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_polygon_from_rect(&points(&[(2.0, 2.0), (10.0, 2.0), (6.0, 10.0)]), 3);

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        assert_eq!(objects.len(), 0);
    }

    #[test]
    fn add_polygon_from_rect_creates_an_object_covering_the_triangle_bbox() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        let new_class = controller
            .app_state
            .get_project_write()
            .classification
            .add_class(Class {
                id: 1,
                color: 0xff0000,
                name: "cl1".into(),
                notes: "note".into(),
            });

        controller
            .app_state
            .get_project_write()
            .set_selected_object_class(new_class);

        controller.add_polygon_from_rect(&points(&[(2.0, 2.0), (10.0, 2.0), (6.0, 10.0)]), 3);

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 10, 10]);
        assert_eq!(objects[0].object_class.contains(&new_class), true);
        assert_eq!(objects[0].object_class.len(), 1);
        assert!(objects[0].area > 0, "the triangle interior must be filled");
        // Measured on the triangle's own pixels, not its whole bounding box.
        assert_eq!(objects[0].area, objects[0].mask_data.count_ones());
        assert!(objects[0].area < 9 * 9);
        // On the plane shown (first slice, first frame, the selected channel).
        assert_eq!(
            (objects[0].z_stack, objects[0].c_stack, objects[0].t_stack),
            (0, project.get_selected_image_channel_idx(), 0)
        );
    }

    // -- attach_callbacks (live AppWindow) ------------------------------------------

    struct Live {
        ui: AppWindow,
        ui_state: Arc<UiState>,
        controller: Arc<ViewPortObjectController>,
    }

    /// A controller on a live window with its callbacks attached, an image
    /// in the cache and - with `class` - an object class selected to paint.
    fn live_controller(class: bool) -> Live {
        use crate::editor::test_support::{test_ui_window, ui_state_with_window};
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, project_with_one_image());
        if class {
            let mut project = ui_state.get_project_write();
            let id = project.classification.add_class(Class {
                name: "Nuclei".into(),
                ..Default::default()
            });
            project.set_selected_object_class(id);
        }
        let (controller, viewport_cache) = controller_for(ui.as_weak(), &ui_state);
        seed_image_cache(&viewport_cache);
        // Drawn 1:1 at the origin.
        if let Some((_, ctx)) = &mut *viewport_cache.active_high_res_data.write().unwrap() {
            ctx.zoomed_w = 20.0;
            ctx.zoomed_h = 20.0;
        }
        controller.attach_callbacks();
        Live {
            ui,
            ui_state,
            controller,
        }
    }

    impl Live {
        fn object_state(&self) -> ViewportObjectState<'_> {
            self.ui.global::<ViewportObjectState>()
        }

        fn object_count(&self) -> usize {
            self.ui_state
                .get_project()
                .get_objects()
                .map_or(0, |o| o.len())
        }

        fn markers(&self) -> Vec<MarkerData> {
            self.object_state().get_markers().iter().collect()
        }
    }

    #[test]
    fn finishing_a_gesture_with_a_non_painting_tool_adds_no_object() {
        let live = live_controller(true);
        for tool in [
            ToolState::Move,
            ToolState::Select,
            ToolState::PaintMarker,
            ToolState::Brush,
            ToolState::Eraser,
        ] {
            live.object_state().invoke_object_paint_finished(
                points(&[(2.0, 2.0), (5.0, 5.0), (2.0, 8.0)]),
                tool,
                3,
            );
        }
        crate::helper::ui_thread::drain_ui_queue();
        assert_eq!(live.object_count(), 0);
    }

    #[test]
    fn painting_an_oval_adds_an_elliptic_object() {
        let live = live_controller(true);

        live.object_state().invoke_object_paint_finished(
            points(&[(2.0, 2.0), (12.0, 8.0)]),
            ToolState::PaintOval,
            0,
        );
        crate::helper::ui_thread::drain_ui_queue();

        let project = live.ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 12, 8]);
        // The ends of both axes are part of the ellipse.
        assert!(objects[0].is_part_of(2, 5) && objects[0].is_part_of(12, 5));
        assert!(objects[0].is_part_of(7, 2) && objects[0].is_part_of(7, 8));
        assert!(!objects[0].is_part_of(2, 2), "corner");
    }

    #[test]
    fn painting_a_polygon_adds_an_object_of_its_outline() {
        let live = live_controller(true);

        live.object_state().invoke_object_paint_finished(
            points(&[(2.0, 2.0), (10.0, 2.0), (10.0, 10.0), (2.0, 10.0)]),
            ToolState::PaintPolygon,
            4,
        );
        crate::helper::ui_thread::drain_ui_queue();

        let project = live.ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 10, 10]);
    }

    #[test]
    fn a_polygon_with_fewer_points_than_announced_is_a_no_op() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);
        let class = ui_state
            .get_project_write()
            .classification
            .add_class(Class {
                name: "c".into(),
                ..Default::default()
            });
        ui_state
            .get_project_write()
            .set_selected_object_class(class);

        controller.add_polygon_from_rect(&points(&[(2.0, 2.0), (10.0, 2.0)]), 3);

        assert_eq!(
            ui_state.get_project().get_objects().map(|o| o.len()),
            Some(0)
        );
    }

    #[test]
    fn an_oval_with_a_selected_class_is_added_without_a_window() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);
        let class = ui_state
            .get_project_write()
            .classification
            .add_class(Class {
                name: "c".into(),
                ..Default::default()
            });
        ui_state
            .get_project_write()
            .set_selected_object_class(class);

        controller.add_oval_from_rect(&points(&[(8.0, 8.0), (4.0, 4.0)]));

        let project = ui_state.get_project();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [4, 4, 8, 8], "corners in any order");
    }

    #[test]
    fn clicking_in_the_viewport_selects_the_object_under_the_cursor() {
        let live = live_controller(false);
        {
            let mut project = live.ui_state.get_project_write();
            project.toggle_hide_unclassified_objects();
            project.add_object(&object_with_bbox(1, [10, 10, 15, 15]));
        }

        live.object_state().invoke_viewport_clicked(12.0, 12.0);
        crate::helper::ui_thread::drain_ui_queue();
        assert_eq!(
            live.ui_state.get_project().get_selected_object_id(),
            Some(ObjectId(1))
        );

        live.object_state().invoke_viewport_clicked(30.0, 30.0);
        crate::helper::ui_thread::drain_ui_queue();
        assert_eq!(live.ui_state.get_project().get_selected_object_id(), None);
    }

    #[test]
    fn clicking_left_of_or_above_the_image_never_picks_an_object_at_its_origin() {
        let (ui_state, controller, _) = make_controller();
        {
            let mut project = ui_state.get_project_write();
            project.toggle_hide_unclassified_objects();
            project.add_object(&object_with_bbox(1, [0, 0, 5, 5]));
            project.set_selected_object(Some(ObjectId(1)));
        }

        controller.find_object_from_clicked_coordinates(-3.0, 2.0);
        assert_eq!(ui_state.get_project().get_selected_object_id(), None);

        controller.find_object_from_clicked_coordinates(2.0, -3.0);
        assert_eq!(ui_state.get_project().get_selected_object_id(), None);
    }

    #[test]
    fn clicking_without_a_current_image_selects_nothing() {
        let ui_state = crate::editor::test_support::test_ui_state();
        let (controller, _) = controller_for(slint::Weak::default(), &ui_state);

        controller.find_object_from_clicked_coordinates(1.0, 1.0);

        assert_eq!(ui_state.get_project().get_selected_object_id(), None);
    }

    #[test]
    fn a_transparency_change_reaches_the_overlay_after_a_short_debounce() {
        let live = live_controller(false);
        let transparency = || {
            live.controller
                .viewport_controller
                .overlay_state
                .read()
                .unwrap()
                .object_transparency
        };
        let before = transparency();

        live.object_state().invoke_object_transparency_changed(0.3);
        live.object_state().invoke_object_transparency_changed(0.25);
        assert_eq!(transparency(), before, "debounced");

        i_slint_backend_testing::mock_elapsed_time(std::time::Duration::from_millis(10));
        assert_eq!(transparency(), 0.25, "the last value wins");
    }

    #[test]
    fn a_placed_marker_is_labelled_with_the_intensities_under_it() {
        let live = live_controller(false);

        live.object_state().invoke_marker_placed(3.0, 4.0);

        let markers = live.markers();
        assert_eq!(markers.len(), 1);
        assert_eq!((markers[0].image_x, markers[0].image_y), (3.0, 4.0));
        // 0.5 of an 8-bit range (0..255): 127.5.
        assert_eq!(markers[0].label, "ch0: 128");
    }

    #[test]
    fn a_marker_is_placed_in_image_coordinates_of_the_current_view() {
        let live = live_controller(false);
        {
            let mut vp = live
                .controller
                .viewport_controller
                .viewport_state
                .write()
                .unwrap();
            vp.zoom = 2.0;
            vp.offset_x = 10.0;
            vp.offset_y = -6.0;
        }

        live.object_state().invoke_marker_placed(30.0, 14.0);

        let markers = live.markers();
        assert_eq!((markers[0].image_x, markers[0].image_y), (10.0, 10.0));
    }

    #[test]
    fn a_marker_placed_outside_the_image_has_no_label() {
        let live = live_controller(false);

        live.object_state().invoke_marker_placed(50.0, 3.0);
        live.object_state().invoke_marker_placed(3.0, -1.0);

        let markers = live.markers();
        assert_eq!(markers.len(), 2);
        assert!(markers.iter().all(|m| m.label.is_empty()));
    }

    #[test]
    fn a_marker_placed_before_any_image_was_drawn_has_no_label() {
        let ui = crate::editor::test_support::test_ui_window();
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let (controller, _) = controller_for(ui.as_weak(), &ui_state);
        controller.attach_callbacks();

        ui.global::<ViewportObjectState>()
            .invoke_marker_placed(3.0, 4.0);

        let markers: Vec<MarkerData> = ui
            .global::<ViewportObjectState>()
            .get_markers()
            .iter()
            .collect();
        assert_eq!(markers.len(), 1);
        assert!(markers[0].label.is_empty());
    }

    #[test]
    fn removing_at_a_point_drops_the_closest_marker_within_reach_only() {
        let live = live_controller(false);
        live.object_state().invoke_marker_placed(3.0, 3.0);
        live.object_state().invoke_marker_placed(10.0, 3.0);
        live.object_state().invoke_marker_placed(40.0, 40.0);

        // Nothing within 12 px.
        live.object_state().invoke_marker_remove_at(100.0, 100.0);
        assert_eq!(live.markers().len(), 3);

        // Both of the first two are within reach - the closer one goes.
        live.object_state().invoke_marker_remove_at(8.0, 3.0);
        let left: Vec<(f32, f32)> = live
            .markers()
            .iter()
            .map(|m| (m.image_x, m.image_y))
            .collect();
        assert_eq!(left, vec![(3.0, 3.0), (40.0, 40.0)]);
    }

    #[test]
    fn markers_are_removed_where_they_are_shown_in_the_current_view() {
        let live = live_controller(false);
        live.object_state().invoke_marker_placed(3.0, 3.0);
        {
            let mut vp = live
                .controller
                .viewport_controller
                .viewport_state
                .write()
                .unwrap();
            vp.zoom = 4.0;
            vp.offset_x = 100.0;
            vp.offset_y = 50.0;
        }

        // The marker's old screen position is far from where it is now.
        live.object_state().invoke_marker_remove_at(3.0, 3.0);
        assert_eq!(live.markers().len(), 1);

        live.object_state().invoke_marker_remove_at(112.0, 62.0);
        assert!(live.markers().is_empty());
    }
}
