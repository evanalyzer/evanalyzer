use crate::AppWindow;
use crate::MarkerData;
use crate::PointSlint;
use crate::ToolState;
use crate::UiState;
use crate::ViewportRoiState;
use crate::editor::images_list_controller::ImagesListController;
use crate::editor::roi_list_controller::RoiListController;
use crate::editor::viewport_cache::ViewportCache;
use crate::editor::viewport_controller::ViewportController;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::extensions::roi_ext::RoiExt;
use evanalyzer_core::{ImageContainer, Roi};
use kornia_image::ImageSize;
use slint::ComponentHandle;
use slint::Model;
use slint::ModelRc;
use slint::VecModel;
use std::sync::Arc;

pub struct ViewPortRoiController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) viewport_cache: Arc<ViewportCache>,
    pub(crate) image_list_controller: Arc<ImagesListController>,
    pub(crate) roi_list_controller: Arc<RoiListController>,
}

impl ViewPortRoiController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        viewport_cache: Arc<ViewportCache>,
        image_list_controller: Arc<ImagesListController>,
        roi_list_controller: Arc<RoiListController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
            viewport_cache,
            image_list_controller,
            roi_list_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // ROI painting finished
            let manager = self.clone();
            ui.global::<ViewportRoiState>().on_roi_paint_finished(
                move |points, tool_state, nr_of_polygon_points| {
                    match tool_state {
                        ToolState::Move => return,
                        ToolState::Select => return,
                        ToolState::PaintMarker => return,
                        ToolState::PaintRectangle => manager.add_roi_from_rect(&points),
                        ToolState::PaintOval => manager.add_oval_from_rect(&points),
                        ToolState::PaintPolygon => {
                            manager.add_polygon_from_rect(&points, nr_of_polygon_points)
                        }
                    };
                    manager.viewport_controller.trigger_image_redraw_rois();
                    manager.image_list_controller.sync_image_list_to_slint();
                },
            );

            // In viewport clicked
            let manager = self.clone();
            ui.global::<ViewportRoiState>()
                .on_viewport_clicked(move |clicked_x, clicked_y| {
                    manager.find_roi_from_clicked_coordinates(clicked_x, clicked_y);
                    manager.roi_list_controller.sync_selected_roi_to_slint(true);
                    manager.viewport_controller.trigger_image_redraw_rois();
                });

            // ROI transparency
            let manager = self.clone();
            let debounce_timer = slint::Timer::default();
            ui.global::<ViewportRoiState>()
                .on_roi_transparency_changed(move |transparency| {
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
                                .roi_transparency = transparency;
                            manager_in.viewport_controller.trigger_image_redraw_rois();
                        },
                    );
                });

            // Marker placed (left-click in PaintMarker mode)
            let manager = self.clone();
            ui.global::<ViewportRoiState>()
                .on_marker_placed(move |screen_x, screen_y| {
                    let Some(ui) = manager.ui.upgrade() else { return };
                    let (img_x, img_y) = {
                        let vp = manager.viewport_controller.viewport_state.read().expect("Poisoned");
                        ((screen_x - vp.offset_x) / vp.zoom, (screen_y - vp.offset_y) / vp.zoom)
                    };
                    let label = manager.read_intensity_at_screen(screen_x, screen_y);
                    let state = ui.global::<ViewportRoiState>();
                    let current = state.get_markers();
                    let mut vec: Vec<MarkerData> = (0..current.row_count())
                        .filter_map(|i| current.row_data(i))
                        .collect();
                    vec.push(MarkerData { image_x: img_x, image_y: img_y, label: label.into() });
                    state.set_markers(ModelRc::new(VecModel::from(vec)));
                });

            // Marker remove-at (right-click in PaintMarker mode)
            let manager = self.clone();
            ui.global::<ViewportRoiState>()
                .on_marker_remove_at(move |screen_x, screen_y| {
                    let Some(ui) = manager.ui.upgrade() else { return };
                    let zoom = manager.viewport_controller.viewport_state.read().expect("Poisoned").zoom;
                    let offset_x = manager.viewport_controller.viewport_state.read().expect("Poisoned").offset_x;
                    let offset_y = manager.viewport_controller.viewport_state.read().expect("Poisoned").offset_y;
                    let state = ui.global::<ViewportRoiState>();
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
        let data_tmp = self.viewport_cache.active_high_res_data.read().expect("Poisoned");
        let Some((image_data, ctx)) = &*data_tmp else { return String::new() };
        let local_x = (screen_x - ctx.draw_x) / (ctx.zoomed_w / ctx.image_w as f32);
        let local_y = (screen_y - ctx.draw_y) / (ctx.zoomed_h / ctx.image_h as f32);
        if local_x < 0.0 || local_x >= ctx.image_w as f32 || local_y < 0.0 || local_y >= ctx.image_h as f32 {
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

    pub fn find_roi_from_clicked_coordinates(&self, click_x: f32, click_y: f32) {
        let view_port_state = self
            .viewport_controller
            .viewport_state
            .read()
            .expect("Poisoned")
            .clone();
        let x1 = ((click_x - view_port_state.offset_x) / (view_port_state.zoom)) as u32;
        let y1 = ((click_y - view_port_state.offset_y) / (view_port_state.zoom)) as u32;

        let clicked_roi_id = {
            let project = self.app_state.get_project();
            let rois = project.get_rois();
            let preview_rois = project.get_preview_rois();

            let mut found_id = None;
            if let Some(rois_some) = rois {
                for roi in rois_some.iter().chain(preview_rois) {
                    if roi.is_part_of(x1, y1) {
                        found_id = Some(roi.id.clone());
                        break;
                    }
                }
            }
            found_id
        };

        let mut project = self.app_state.get_project_write();
        project.set_selected_roi(clicked_roi_id);
    }

    pub fn add_roi_from_rect(&self, points: &ModelRc<PointSlint>) {
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

        self.add_to_roi_list(mask_data, bbox);
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

        self.add_to_roi_list(mask_data, bbox);
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

        self.add_to_roi_list(mask_data, bbox);
    }

    fn add_to_roi_list(&self, mask_data: BitVec<u64, Lsb0>, bbox: [u32; 4]) {
        let (data_tmp, read_context) = self.viewport_cache.get_image_references();
        let (idx, object_class) = {
            let project = self.app_state.get_project();
            (
                project.get_selected_image_channel_idx(),
                project.get_selected_object_class(),
            )
        };

        if let Some((_, selected_channel)) = data_tmp.get(idx as usize) {
            let roi = Roi::from_mask(
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
                .add_roi(&roi.to_roi_settings());
            self.app_state.mark_dirty();
        }
    }
}
