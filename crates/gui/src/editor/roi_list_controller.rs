use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::get_colors_from_class;
use crate::{AppWindow, RoiItemDataSlint, RoiListState};
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
    /// Indices into the combined (manual ++ preview) ROI list that should actually
    /// be displayed - precomputed at bridge creation, same pattern as `label_counts`.
    /// Lets `row_count`/`row_data` filter out unclassified ROIs (when hidden) while
    /// the `id` passed to Slint still reflects the true underlying position, so the
    /// existing selection/edit callbacks (which index into the unfiltered list) keep
    /// working unchanged.
    visible_rows: Vec<usize>,
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
                let visible_rows = compute_visible_rows(&bridge_ptr.app_state);
                let bridge = Rc::new(RoiModalBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                    label_counts,
                    visible_rows,
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
        self.visible_rows.len()
    }

    fn row_data(&self, row: usize) -> Option<Self::Data> {
        // `row` is a position in the *filtered* list; map it back to the true
        // position in the underlying (manual ++ preview) list, which `id` must
        // keep reflecting so the existing selection/edit callbacks (indexed
        // against the unfiltered list) keep working unchanged.
        let underlying_row = *self.visible_rows.get(row)?;

        let project = self.app_state.get_project();
        let manual_len = project.get_rois().map(|r| r.len()).unwrap_or(0);

        if underlying_row < manual_len {
            project.get_rois()?.get(underlying_row).map(|roi| {
                let count = *self.label_counts.get(&roi.segmentation_class).unwrap_or(&0);
                roi_rust_to_roi_slint(roi, &project, count, false, underlying_row as i32)
            })
        } else {
            let preview_rois = project.get_preview_rois();
            preview_rois
                .get(underlying_row - manual_len)
                .map(|roi| {
                    let count = *self.label_counts.get(&roi.segmentation_class).unwrap_or(&0);
                    roi_rust_to_roi_slint(roi, &project, count, false, underlying_row as i32)
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

/// Indices into the combined (manual ++ preview) ROI list that should be shown in
/// the list panel, respecting `hide_unclassified_rois`.
fn compute_visible_rows(app_state: &Arc<UiState>) -> Vec<usize> {
    let project = app_state.get_project();
    let hide_unclassified = project.hide_unclassified_rois();
    let manual = project.get_rois().unwrap_or(&[]);
    let preview = project.get_preview_rois();

    manual
        .iter()
        .chain(preview.iter())
        .enumerate()
        .filter(|(_, roi)| !hide_unclassified || !roi.object_class.is_empty())
        .map(|(i, _)| i)
        .collect()
}

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

fn format_circularity(roi: &RoiSettings) -> SharedString {
    if roi.area == 0 {
        return "".into();
    }
    // Perimeter is precomputed once during extraction (Roi::get_perimeter) and carried
    // on RoiSettings, so the ROI list no longer re-walks the mask boundary here.
    let p = roi.perimeter;
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

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::roi_settings::IntensitySettings;
    use indexmap::IndexMap;

    fn roi_with_area_and_perimeter(area: usize, perimeter: f32) -> RoiSettings {
        RoiSettings { area, perimeter, ..Default::default() }
    }

    #[test]
    fn format_circularity_is_empty_for_a_zero_area_roi() {
        assert_eq!(format_circularity(&roi_with_area_and_perimeter(0, 10.0)), "");
    }

    #[test]
    fn format_circularity_is_empty_when_perimeter_is_not_positive() {
        assert_eq!(format_circularity(&roi_with_area_and_perimeter(100, 0.0)), "");
        assert_eq!(format_circularity(&roi_with_area_and_perimeter(100, -1.0)), "");
    }

    #[test]
    fn format_circularity_of_a_perfect_circle_is_one() {
        // area = pi * r^2, perimeter = 2 * pi * r -> circularity should be ~1.0.
        let r = 10.0f32;
        let area = (std::f32::consts::PI * r * r) as usize;
        let perimeter = 2.0 * std::f32::consts::PI * r;
        let result = format_circularity(&roi_with_area_and_perimeter(area, perimeter));
        assert_eq!(result, "1.00");
    }

    #[test]
    fn format_circularity_is_clamped_to_one_for_measurement_noise() {
        // A tiny/jagged mask can push the raw formula slightly above 1.0 -
        // the result must still clamp to "1.00", not overshoot.
        let result = format_circularity(&roi_with_area_and_perimeter(1000, 1.0));
        assert_eq!(result, "1.00");
    }

    #[test]
    fn format_intensities_per_channel_is_empty_with_no_channel_data() {
        let roi = RoiSettings::default();
        let (sums, avgs) = format_intensities_per_channel(&roi);
        assert!(sums.is_empty());
        assert!(avgs.is_empty());
    }

    #[test]
    fn format_intensities_per_channel_sizes_the_vec_to_the_highest_channel_id() {
        let mut roi = RoiSettings { area: 10, ..Default::default() };
        let mut intensities = IndexMap::new();
        intensities.insert(0, IntensitySettings { sum_intensity: 100.0, ..Default::default() });
        intensities.insert(2, IntensitySettings { sum_intensity: 50.0, ..Default::default() });
        roi.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&roi);
        // max channel id is 2, so the vecs must be length 3 (0, 1, 2).
        assert_eq!(sums.len(), 3);
        assert_eq!(avgs.len(), 3);
        assert_eq!(sums[0], "100.0");
        assert_eq!(sums[1], "", "no data for channel 1 - left as the default empty string");
        assert_eq!(sums[2], "50.0");
        assert_eq!(avgs[0], "10.0"); // 100 / area(10)
        assert_eq!(avgs[2], "5.0"); // 50 / area(10)
    }

    #[test]
    fn format_intensities_per_channel_leaves_avg_empty_when_area_is_zero() {
        let mut roi = RoiSettings { area: 0, ..Default::default() };
        let mut intensities = IndexMap::new();
        intensities.insert(0, IntensitySettings { sum_intensity: 100.0, ..Default::default() });
        roi.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&roi);
        assert_eq!(sums[0], "100.0");
        assert_eq!(avgs[0], "", "can't average over a zero-area ROI");
    }

    #[test]
    fn format_area_nm2_picks_the_right_unit_suffix() {
        let unit_px = PixelSizeSettings { x: 1.0, y: 1.0, z: 1.0 };
        assert_eq!(format_area_nm2(500, &unit_px), "500.0");
        assert_eq!(format_area_nm2(1500, &unit_px), "1.5 k");
        assert_eq!(format_area_nm2(2_500_000, &unit_px), "2.50 M");
    }

    #[test]
    fn format_area_nm2_scales_by_pixel_size() {
        let px = PixelSizeSettings { x: 2.0, y: 3.0, z: 1.0 };
        // 100 px * 2.0 * 3.0 = 600.0 nm^2
        assert_eq!(format_area_nm2(100, &px), "600.0");
    }

    #[test]
    fn format_area_nm2_boundary_is_inclusive_of_the_next_unit() {
        let unit_px = PixelSizeSettings { x: 1.0, y: 1.0, z: 1.0 };
        assert_eq!(format_area_nm2(1000, &unit_px), "1.0 k");
        assert_eq!(format_area_nm2(1_000_000, &unit_px), "1.00 M");
    }
}
