use crate::AppWindow;
use crate::MarkerData;
use crate::PointSlint;
use crate::ToolState;
use crate::UiState;
use crate::ViewportObjectState;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::viewport_cache::ViewportCache;
use crate::editor::viewport_controller::ViewportController;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use evanalyzer_app::extensions::object_ext::ObjectExt;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_core::{ImageContainer, Object};
use kornia_image::ImageSize;
use slint::ComponentHandle;
use slint::Model;
use slint::ModelRc;
use slint::VecModel;
use std::sync::Arc;

pub struct ViewPortObjectController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) viewport_cache: Arc<ViewportCache>,
    pub(crate) image_list_controller: Arc<ImagesListController>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
}

impl ViewPortObjectController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        viewport_cache: Arc<ViewportCache>,
        image_list_controller: Arc<ImagesListController>,
        object_list_controller: Arc<ObjectListController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
            viewport_cache,
            image_list_controller,
            object_list_controller,
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
                        ToolState::PaintRectangle => manager.add_object_from_rect(&points),
                        ToolState::PaintOval => manager.add_oval_from_rect(&points),
                        ToolState::PaintPolygon => {
                            manager.add_polygon_from_rect(&points, nr_of_polygon_points)
                        }
                    };
                    manager.viewport_controller.trigger_image_redraw_objects();
                    manager.image_list_controller.sync_image_list_to_slint();
                    manager.object_list_controller.sync_objects_to_slint();
                },
            );

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
            if let ImageContainer::F32Gray(img) = &*channel.image {
                if let Some(&raw_val) = img.as_slice().get(idx) {
                    let scaled = raw_val * 2.0_f32.powf(ctx.bit_depth as f32);
                    values.push(format!("{}: {:.0}", channel.name, scaled));
                }
            }
        }
        values.join(" | ")
    }

    pub fn find_object_from_clicked_coordinates(&self, click_x: f32, click_y: f32) {
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();
        let x1 = ((click_x - view_port_state.offset_x) / (view_port_state.zoom)) as u32;
        let y1 = ((click_y - view_port_state.offset_y) / (view_port_state.zoom)) as u32;

        let clicked_object_id = {
            let project = self.app_state.get_project();
            let objects = project.get_objects();
            let preview_objects = project.get_preview_objects();

            let mut found_id = None;
            if let Some(objects_some) = objects {
                for object in objects_some.iter().chain(preview_objects) {
                    if object.is_part_of(x1, y1) {
                        found_id = Some(object.id.clone());
                        break;
                    }
                }
            }
            found_id
        };

        let mut project = self.app_state.get_project_write();
        project.set_selected_object(clicked_object_id);
    }

    pub fn add_object_from_rect(&self, points: &ModelRc<PointSlint>) {
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();

        let x1 =
            (points.row_data(0).unwrap().x - view_port_state.offset_x) / (view_port_state.zoom);
        let y1 =
            (points.row_data(0).unwrap().y - view_port_state.offset_y) / (view_port_state.zoom);
        let x2 =
            (points.row_data(1).unwrap().x - view_port_state.offset_x) / (view_port_state.zoom);
        let y2 =
            (points.row_data(1).unwrap().y - view_port_state.offset_y) / (view_port_state.zoom);

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
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();

        let x1 =
            (points.row_data(0).unwrap().x - view_port_state.offset_x) / (view_port_state.zoom);
        let y1 =
            (points.row_data(0).unwrap().y - view_port_state.offset_y) / (view_port_state.zoom);
        let x2 =
            (points.row_data(1).unwrap().x - view_port_state.offset_x) / (view_port_state.zoom);
        let y2 =
            (points.row_data(1).unwrap().y - view_port_state.offset_y) / (view_port_state.zoom);

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

    fn add_to_object_list(&self, mask_data: BitVec<u64, Lsb0>, bbox: [u32; 4]) {
        let (data_tmp, read_context) = self.viewport_cache.get_image_references();
        let (idx, object_class) = {
            let project = self.app_state.get_project();
            (
                project.get_selected_image_channel_idx(),
                project.get_selected_object_class(),
            )
        };

        if let Some((_, selected_channel)) = data_tmp.get(idx as usize) {
            let object = Object::from_mask(
                &ImageSize {
                    width: read_context.full_image_w,
                    height: read_context.full_image_h,
                },
                mask_data,
                bbox,
                selected_channel,
                data_tmp.as_slice(),
                object_class,
            );
            self.app_state
                .get_project_write()
                .add_object(&object.to_object_settings());
            self.app_state.mark_dirty();
        }
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
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use evanalyzer_core::{ImageChannel, ManagedImage};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::Image;
    use kornia_image::allocator::CpuAllocator;

    fn make_controller() -> (
        Arc<UiState>,
        Arc<ViewPortObjectController>,
        Arc<ViewportCache>,
    ) {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
        ));
        let viewport_cache = Arc::new(ViewportCache::new(ui_state.clone()));
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
            object_list_controller.clone(),
        ));
        let controller = Arc::new(ViewPortObjectController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller,
            viewport_cache.clone(),
            image_list_controller,
            object_list_controller,
        ));
        (ui_state, controller, viewport_cache)
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
        let image =
            Image::<f32, 1, CpuAllocator>::new(size, vec![0.5f32; 20 * 20], CpuAllocator).unwrap();
        let container = ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
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
            Arc::new(vec![channel]),
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
    fn add_object_from_rect_creates_an_object_spanning_the_two_corner_points() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_object_from_rect(&points(&[(2.0, 2.0), (5.0, 5.0)]));

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 5, 5]);
        // A rectangle mask fills every pixel in its bbox.
        assert_eq!(objects[0].area, 4 * 4);
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

    // -- add_oval_from_rect / add_polygon_from_rect (no cached data) --------------

    #[test]
    fn add_oval_from_rect_without_cached_image_data_adds_nothing() {
        let (ui_state, controller, _) = make_controller();

        controller.add_oval_from_rect(&points(&[(2.0, 2.0), (8.0, 8.0)]));

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
    fn add_polygon_from_rect_creates_an_object_covering_the_triangle_bbox() {
        let (ui_state, controller, viewport_cache) = make_controller();
        seed_image_cache(&viewport_cache);

        controller.add_polygon_from_rect(&points(&[(2.0, 2.0), (10.0, 2.0), (6.0, 10.0)]), 3);

        let project = ui_state.get_project();
        let objects = project
            .get_objects()
            .expect("current series must have objects");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].bbox, [2, 2, 10, 10]);
        assert!(objects[0].area > 0, "the triangle interior must be filled");
    }
}
