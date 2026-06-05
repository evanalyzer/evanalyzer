use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::get_colors_from_class;
use crate::{AppWindow, RoiItemDataSlint, RoiListState};
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use evanalyzer_app::ProjectWithRuntime;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::core_types::{ObjectClass, SegmentationClass};
use evanalyzer_cfg::settings::images_settings::PixelSizeSettings;
use evanalyzer_cfg::settings::roi_settings::RoiSettings;
use log::warn;
use slint::{Color, Model};
use slint::{ComponentHandle, ModelNotify};
use slint::{ModelRc, SharedString};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
struct RoiModalBridge {
    app_state: Arc<UiState>,
    notify: ModelNotify,
    /// Count of ROIs per segmentation class – precomputed at bridge creation.
    label_counts: HashMap<SegmentationClass, i32>,
}

pub struct RoiListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl RoiListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state: app_state.clone(),
            viewport_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // On ROI selected
            let manager = self.clone();
            ui.global::<RoiListState>().on_roi_selected(move |roi_id| {
                let mut project = manager.app_state.get_project_write();
                let selected = if roi_id > 0 {
                    let row = (roi_id - 1) as usize;
                    let manual_len = project.get_rois().map(|r| r.len()).unwrap_or(0);
                    if row < manual_len {
                        project.get_rois().and_then(|r| r.get(row)).map(|r| r.id.clone())
                    } else {
                        project.get_preview_rois().get(row - manual_len).map(|r| r.id.clone())
                    }
                } else {
                    None
                };
                project.set_selected_roi(selected);
                drop(project);
                manager.sync_selected_roi_to_slint(false);
                manager.viewport_controller.trigger_image_redraw_rois();
            });

            // Add class to ROI
            let manager = self.clone();
            ui.global::<RoiListState>().on_roi_add_class(move |roi_id| {
                let mut project = manager.app_state.get_project_write();
                if roi_id > 0 {
                    let class_id = project.get_selected_object_class();
                    let obj_id = project.get_rois()
                        .and_then(|r| r.get((roi_id - 1) as usize))
                        .map(|r| r.id.clone());
                    if let Some(id) = obj_id {
                        project.add_class_to_roi(id, class_id);
                    }
                }
                manager.sync_selected_roi_to_slint(false);
                manager.sync_rois_to_slint();
                manager.viewport_controller.trigger_image_redraw_rois();
            });

            // Remove class from ROI
            let manager = self.clone();
            ui.global::<RoiListState>()
                .on_roi_remove_class(move |roi_id, class_id| {
                    let mut project = manager.app_state.get_project_write();
                    if roi_id > 0 {
                        let class_id = ObjectClass::Valid(class_id as u32);
                        let obj_id = project.get_rois()
                            .and_then(|r| r.get((roi_id - 1) as usize))
                            .map(|r| r.id.clone());
                        if let Some(id) = obj_id {
                            project.remove_class_from_roi(id, &class_id);
                        }
                    }
                    manager.sync_selected_roi_to_slint(false);
                    manager.sync_rois_to_slint();
                    manager.viewport_controller.trigger_image_redraw_rois();
                });

            // Delete ROI
            let manager = self.clone();
            ui.global::<RoiListState>().on_roi_delete(move |roi_id| {
                let mut project = manager.app_state.get_project_write();
                let obj_id = project.get_rois()
                    .and_then(|r| r.get((roi_id - 1) as usize))
                    .map(|r| r.id.clone());
                if let Some(id) = obj_id {
                    project.delete_roi(id);
                }
                project.set_selected_roi(None);
                drop(project);
                manager.app_state.mark_dirty();
                manager.sync_rois_to_slint();
                manager.viewport_controller.trigger_image_redraw_rois();
            });
        }
    }

    pub fn sync_rois_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let label_counts = precompute_label_counts(&bridge_ptr.app_state);
                let bridge = Rc::new(RoiModalBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                    label_counts,
                });
                let model_rc = ModelRc::new(bridge);
                ui.global::<RoiListState>().set_roi_list(model_rc);
            }
        }) {
            warn!("Failed to sync ROIs to Slint: {}", e);
        }
    }

    pub fn sync_selected_roi_to_slint(self: &Arc<Self>, scroll_to: bool) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let class_state = ui.global::<RoiListState>();
                let project = bridge_ptr.app_state.get_project();
                if let Some(roi) = project.get_selected_roi() {
                    let preview_rois = project.get_preview_rois();
                    let label_count_from_preview = preview_rois
                        .iter()
                        .filter(|r| r.segmentation_class == roi.segmentation_class)
                        .count() as i32;
                    let label_count = project
                        .get_rois()
                        .map(|rois| {
                            rois.iter()
                                .filter(|r| r.segmentation_class == roi.segmentation_class)
                                .count() as i32
                        })
                        .unwrap_or(0)
                        + label_count_from_preview;
                    let manual_len = project.get_rois().map(|r| r.len()).unwrap_or(0);
                    let index = project
                        .get_rois()
                        .and_then(|rois| rois.iter().position(|r| r.id == roi.id))
                        .map(|i| i as i32)
                        .unwrap_or_else(|| {
                            preview_rois
                                .iter()
                                .position(|r| r.id == roi.id)
                                .map(|i| (manual_len + i) as i32)
                                .unwrap_or(-1)
                        });
                    class_state.set_selected_roi(roi_rust_to_roi_slint(
                        &roi,
                        &project,
                        label_count,
                        true,
                        index,
                    ));
                    if scroll_to {
                        class_state.set_scroll_to_roi_index(index);
                    }
                } else {
                    class_state.set_selected_roi(RoiItemDataSlint::default());
                }
            }
        }) {
            warn!("Failed to sync ROI selection to Slint: {}", e);
        }
    }
}

impl Model for RoiModalBridge {
    type Data = RoiItemDataSlint;

    fn row_count(&self) -> usize {
        let project = self.app_state.get_project();
        let manual = project.get_rois().map(|r| r.len()).unwrap_or(0);
        let preview = project.get_preview_rois().len();
        manual + preview
    }

    fn row_data(&self, row: usize) -> Option<Self::Data> {
        let project = self.app_state.get_project();
        let manual_len = project.get_rois().map(|r| r.len()).unwrap_or(0);

        if row < manual_len {
            project.get_rois()?.get(row).map(|roi| {
                let count = *self.label_counts.get(&roi.segmentation_class).unwrap_or(&0);
                roi_rust_to_roi_slint(roi, &project, count, false, row as i32)
            })
        } else {
            let preview_rois = project.get_preview_rois();
            preview_rois.get(row - manual_len).map(|roi| {
                let count = *self.label_counts.get(&roi.segmentation_class).unwrap_or(&0);
                roi_rust_to_roi_slint(roi, &project, count, false, row as i32)
            })
        }
    }

    fn model_tracker(&self) -> &dyn slint::ModelTracker {
        &self.notify
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn precompute_label_counts(app_state: &Arc<UiState>) -> HashMap<SegmentationClass, i32> {
    let project = app_state.get_project();
    let mut counts: HashMap<SegmentationClass, i32> = HashMap::new();
    if let Some(rois) = project.get_rois() {
        for roi in rois {
            *counts.entry(roi.segmentation_class).or_insert(0) += 1;
        }
    }
    for roi in project.get_preview_rois().iter() {
        *counts.entry(roi.segmentation_class).or_insert(0) += 1;
    }
    counts
}

/// Compute perimeter from RoiSettings mask using ImageJ's algorithm (same as Roi::get_perimeter).
fn get_perimeter(roi: &RoiSettings) -> f32 {
    let [x_min, y_min, x_max, y_max] = roi.bbox;
    // bbox[2]/[3] are INCLUSIVE - width = xmax - xmin + 1
    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;

    if width == 0 || height == 0 || roi.area == 0 {
        return 0.0;
    }

    let mask: &BitVec<u64, Lsb0> = &roi.mask_data;
    let mut perimeter = 0.0f32;
    const SQRT2: f32 = std::f32::consts::SQRT_2;

    for y in 0..height {
        for x in 0..width {
            if !mask.get(y * width + x).map(|b| *b).unwrap_or(false) {
                continue;
            }
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let neighbor_inside = nx >= 0
                        && nx < width as i32
                        && ny >= 0
                        && ny < height as i32
                        && mask
                            .get(ny as usize * width + nx as usize)
                            .map(|b| *b)
                            .unwrap_or(false);
                    if !neighbor_inside {
                        perimeter += if dx == 0 || dy == 0 { 1.0 } else { SQRT2 };
                    }
                }
            }
        }
    }
    perimeter / 2.0
}

fn format_circularity(roi: &RoiSettings) -> SharedString {
    if roi.area == 0 {
        return "".into();
    }
    let p = get_perimeter(roi);
    if p <= 0.0 {
        return "".into();
    }
    let c = (4.0 * std::f32::consts::PI * roi.area as f32) / (p * p);
    format!("{:.2}", c.min(1.0)).into()
}

fn format_intensities_per_channel(roi: &RoiSettings) -> (Vec<SharedString>, Vec<SharedString>) {
    let Some(&max_ch) = roi.intensities.keys().max() else {
        return (Vec::new(), Vec::new());
    };
    let len = (max_ch + 1) as usize;
    let mut sums: Vec<SharedString> = vec![SharedString::default(); len];
    let mut avgs: Vec<SharedString> = vec![SharedString::default(); len];
    let area = roi.area as f64;
    for (channel_id, intensities) in &roi.intensities {
        if *channel_id >= 0 {
            let i = *channel_id as usize;
            sums[i] = format!("{:.1}", intensities.sum_intensity).into();
            if area > 0.0 {
                avgs[i] = format!("{:.1}", intensities.sum_intensity / area).into();
            }
        }
    }
    (sums, avgs)
}

fn format_area_nm2(area_px: usize, pixel_sizes: &PixelSizeSettings) -> SharedString {
    let area_nm2 = area_px as f64 * pixel_sizes.x as f64 * pixel_sizes.y as f64;
    if area_nm2 >= 1_000_000.0 {
        format!("{:.2} M", area_nm2 / 1_000_000.0).into()
    } else if area_nm2 >= 1_000.0 {
        format!("{:.1} k", area_nm2 / 1_000.0).into()
    } else {
        format!("{:.1}", area_nm2).into()
    }
}

fn roi_rust_to_roi_slint(
    roi: &RoiSettings,
    project: &ProjectWithRuntime,
    label_count: i32,
    full_metrics: bool,
    row_index: i32,
) -> RoiItemDataSlint {
    let mut class_names_vec: Vec<SharedString> = Vec::new();
    let mut class_colors_vec: Vec<Color> = Vec::new();
    let mut class_ids_vec: Vec<i32> = Vec::new();
    let mut display_name = String::new();

    for class in &roi.object_class {
        let (class_name, (r, g, b)) = match project.get_class_from_id(class) {
            Some(class_data) => {
                let r = ((class_data.color >> 16) & 0xff) as u8;
                let g = ((class_data.color >> 8) & 0xff) as u8;
                let b = (class_data.color & 0xff) as u8;
                (class_data.name.clone(), (r, g, b))
            }
            _ => ("Unclassified".to_string(), (0xff, 0, 0)),
        };

        if display_name.is_empty() {
            display_name = class_name.clone();
        } else {
            display_name.push(',');
            display_name.push_str(&class_name);
        }

        class_names_vec.push(class_name.into());
        class_colors_vec.push(Color::from_rgb_u8(r, g, b));
        class_ids_vec.push(class.to_i32());
    }

    let display_color = get_colors_from_class(project, 255, &roi.object_class);
    let pixel_sizes = project.get_pixel_sizes();

    RoiItemDataSlint {
        id: row_index + 1,
        name: "Annotation".into(),
        display_name: display_name.into(),
        display_color,
        class_names: ModelRc::new(slint::VecModel::from(class_names_vec)).into(),
        class_colors: ModelRc::new(slint::VecModel::from(class_colors_vec)).into(),
        class_ids: ModelRc::new(slint::VecModel::from(class_ids_vec)).into(),
        label_id: roi.segmentation_class.as_u32() as i32,
        label_name: roi.segmentation_class.to_string().into(),
        label_count,
        area: roi.area as i32,
        area_nm2: if full_metrics && roi.area > 0 {
            format_area_nm2(roi.area, &pixel_sizes)
        } else {
            "".into()
        },
        intensities: {
            let (sums, _) = if full_metrics {
                format_intensities_per_channel(roi)
            } else {
                (Vec::new(), Vec::new())
            };
            ModelRc::new(slint::VecModel::from(sums)).into()
        },
        intensity_avgs: {
            let (_, avgs) = if full_metrics {
                format_intensities_per_channel(roi)
            } else {
                (Vec::new(), Vec::new())
            };
            ModelRc::new(slint::VecModel::from(avgs)).into()
        },
        circularity: if full_metrics {
            format_circularity(roi)
        } else {
            "-".into()
        },
    }
}
