use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::get_colors_from_class;
use crate::{AppWindow, ObjectItemDataSlint, ObjectListState};
use evanalyzer_app::ProjectWithRuntime;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::core_types::{ObjectClass, SegmentationClass};
use evanalyzer_cfg::settings::images_settings::PixelSizeSettings;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use log::warn;
use slint::{Color, Model};
use slint::{ComponentHandle, ModelNotify};
use slint::{ModelRc, SharedString};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
struct ObjectModalBridge {
    app_state: Arc<UiState>,
    notify: ModelNotify,
    /// Count of ROIs per segmentation class – precomputed at bridge creation.
    label_counts: HashMap<SegmentationClass, i32>,
    /// Indices into the combined (manual ++ preview) object list that should actually
    /// be displayed - precomputed at bridge creation, same pattern as `label_counts`.
    /// Lets `row_count`/`row_data` filter out unclassified ROIs (when hidden) while
    /// the `id` passed to Slint still reflects the true underlying position, so the
    /// existing selection/edit callbacks (which index into the unfiltered list) keep
    /// working unchanged.
    visible_rows: Vec<usize>,
}

pub struct ObjectListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl ObjectListController {
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
            // On object selected
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_selected(move |object_id| {
                    let mut project = manager.app_state.get_project_write();
                    let selected = if object_id > 0 {
                        let row = (object_id - 1) as usize;
                        let manual_len = project.get_objects().map(|r| r.len()).unwrap_or(0);
                        if row < manual_len {
                            project
                                .get_objects()
                                .and_then(|r| r.get(row))
                                .map(|r| r.id.clone())
                        } else {
                            project
                                .get_preview_objects()
                                .get(row - manual_len)
                                .map(|r| r.id.clone())
                        }
                    } else {
                        None
                    };
                    project.set_selected_object(selected);
                    drop(project);
                    manager.sync_selected_object_to_slint(false);
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Add class to object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_add_class(move |object_id| {
                    let mut project = manager.app_state.get_project_write();
                    if object_id > 0 {
                        let class_id = project.get_selected_object_class();
                        let obj_id = project
                            .get_objects()
                            .and_then(|r| r.get((object_id - 1) as usize))
                            .map(|r| r.id.clone());
                        if let Some(id) = obj_id {
                            project.add_class_to_object(id, class_id);
                        }
                    }
                    manager.sync_selected_object_to_slint(false);
                    manager.sync_objects_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Remove class from object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_remove_class(move |object_id, class_id| {
                    let mut project = manager.app_state.get_project_write();
                    if object_id > 0 {
                        let class_id = ObjectClass::Valid(class_id as u32);
                        let obj_id = project
                            .get_objects()
                            .and_then(|r| r.get((object_id - 1) as usize))
                            .map(|r| r.id.clone());
                        if let Some(id) = obj_id {
                            project.remove_class_from_object(id, &class_id);
                        }
                    }
                    manager.sync_selected_object_to_slint(false);
                    manager.sync_objects_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // Delete object
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_object_delete(move |object_id| {
                    let mut project = manager.app_state.get_project_write();
                    let obj_id = project
                        .get_objects()
                        .and_then(|r| r.get((object_id - 1) as usize))
                        .map(|r| r.id.clone());
                    if let Some(id) = obj_id {
                        project.delete_object(id);
                    }
                    project.set_selected_object(None);
                    drop(project);
                    manager.app_state.mark_dirty();
                    manager.sync_objects_to_slint();
                    manager.viewport_controller.trigger_image_redraw_objects();
                });
        }
    }

    pub fn sync_objects_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();

        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let label_counts = precompute_label_counts(&bridge_ptr.app_state);
                let visible_rows = compute_visible_rows(&bridge_ptr.app_state);
                let bridge = Rc::new(ObjectModalBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                    label_counts,
                    visible_rows,
                });
                let model_rc = ModelRc::new(bridge);
                ui.global::<ObjectListState>().set_object_list(model_rc);
            }
        }) {
            warn!("Failed to sync ROIs to Slint: {}", e);
        }
    }

    pub fn sync_selected_object_to_slint(self: &Arc<Self>, scroll_to: bool) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let class_state = ui.global::<ObjectListState>();
                let project = bridge_ptr.app_state.get_project();
                if let Some(object) = project.get_selected_object() {
                    let preview_objects = project.get_preview_objects();
                    let label_count_from_preview = preview_objects
                        .iter()
                        .filter(|r| r.segmentation_class == object.segmentation_class)
                        .count() as i32;
                    let label_count = project
                        .get_objects()
                        .map(|objects| {
                            objects
                                .iter()
                                .filter(|r| r.segmentation_class == object.segmentation_class)
                                .count() as i32
                        })
                        .unwrap_or(0)
                        + label_count_from_preview;
                    let manual_len = project.get_objects().map(|r| r.len()).unwrap_or(0);
                    let index = project
                        .get_objects()
                        .and_then(|objects| objects.iter().position(|r| r.id == object.id))
                        .map(|i| i as i32)
                        .unwrap_or_else(|| {
                            preview_objects
                                .iter()
                                .position(|r| r.id == object.id)
                                .map(|i| (manual_len + i) as i32)
                                .unwrap_or(-1)
                        });
                    class_state.set_selected_object(object_rust_to_object_slint(
                        &object,
                        &project,
                        label_count,
                        true,
                        index,
                    ));
                    if scroll_to {
                        class_state.set_scroll_to_object_index(index);
                    }
                } else {
                    class_state.set_selected_object(ObjectItemDataSlint::default());
                }
            }
        }) {
            warn!("Failed to sync object selection to Slint: {}", e);
        }
    }
}

impl Model for ObjectModalBridge {
    type Data = ObjectItemDataSlint;

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
        let manual_len = project.get_objects().map(|r| r.len()).unwrap_or(0);

        if underlying_row < manual_len {
            project.get_objects()?.get(underlying_row).map(|object| {
                let count = *self
                    .label_counts
                    .get(&object.segmentation_class)
                    .unwrap_or(&0);
                object_rust_to_object_slint(object, &project, count, false, underlying_row as i32)
            })
        } else {
            let preview_objects = project.get_preview_objects();
            preview_objects
                .get(underlying_row - manual_len)
                .map(|object| {
                    let count = *self
                        .label_counts
                        .get(&object.segmentation_class)
                        .unwrap_or(&0);
                    object_rust_to_object_slint(
                        object,
                        &project,
                        count,
                        false,
                        underlying_row as i32,
                    )
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

/// Indices into the combined (manual ++ preview) object list that should be shown in
/// the list panel, respecting `hide_unclassified_objects`.
fn compute_visible_rows(app_state: &Arc<UiState>) -> Vec<usize> {
    let project = app_state.get_project();
    let hide_unclassified = project.hide_unclassified_objects();
    let manual = project.get_objects().unwrap_or(&[]);
    let preview = project.get_preview_objects();

    manual
        .iter()
        .chain(preview.iter())
        .enumerate()
        .filter(|(_, object)| !hide_unclassified || !object.object_class.is_empty())
        .map(|(i, _)| i)
        .collect()
}

fn precompute_label_counts(app_state: &Arc<UiState>) -> HashMap<SegmentationClass, i32> {
    let project = app_state.get_project();
    let mut counts: HashMap<SegmentationClass, i32> = HashMap::new();
    if let Some(objects) = project.get_objects() {
        for object in objects {
            *counts.entry(object.segmentation_class).or_insert(0) += 1;
        }
    }
    for object in project.get_preview_objects().iter() {
        *counts.entry(object.segmentation_class).or_insert(0) += 1;
    }
    counts
}

fn format_circularity(object: &ObjectMetricSettings) -> SharedString {
    if object.area == 0 {
        return "".into();
    }
    // Perimeter is precomputed once during extraction (Object::get_perimeter) and carried
    // on ObjectMetricSettings, so the object list no longer re-walks the mask boundary here.
    let p = object.perimeter;
    if p <= 0.0 {
        return "".into();
    }
    let c = (4.0 * std::f32::consts::PI * object.area as f32) / (p * p);
    format!("{:.2}", c.min(1.0)).into()
}

fn format_intensities_per_channel(
    object: &ObjectMetricSettings,
) -> (Vec<SharedString>, Vec<SharedString>) {
    let Some(&max_ch) = object.intensities.keys().max() else {
        return (Vec::new(), Vec::new());
    };
    let len = (max_ch + 1) as usize;
    let mut sums: Vec<SharedString> = vec![SharedString::default(); len];
    let mut avgs: Vec<SharedString> = vec![SharedString::default(); len];
    let area = object.area as f64;
    for (channel_id, intensities) in &object.intensities {
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

fn object_rust_to_object_slint(
    object: &ObjectMetricSettings,
    project: &ProjectWithRuntime,
    label_count: i32,
    full_metrics: bool,
    row_index: i32,
) -> ObjectItemDataSlint {
    let mut class_names_vec: Vec<SharedString> = Vec::new();
    let mut class_colors_vec: Vec<Color> = Vec::new();
    let mut class_ids_vec: Vec<i32> = Vec::new();
    let mut display_name = String::new();

    for class in &object.object_class {
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

    let display_color = get_colors_from_class(project, 255, &object.object_class);
    let pixel_sizes = project.get_pixel_sizes();

    ObjectItemDataSlint {
        id: row_index + 1,
        name: "Annotation".into(),
        display_name: display_name.into(),
        display_color,
        class_names: ModelRc::new(slint::VecModel::from(class_names_vec)).into(),
        class_colors: ModelRc::new(slint::VecModel::from(class_colors_vec)).into(),
        class_ids: ModelRc::new(slint::VecModel::from(class_ids_vec)).into(),
        label_id: object.segmentation_class.as_u32() as i32,
        label_name: object.segmentation_class.to_string().into(),
        label_count,
        area: object.area as i32,
        area_nm2: if full_metrics && object.area > 0 {
            format_area_nm2(object.area, &pixel_sizes)
        } else {
            "".into()
        },
        intensities: {
            let (sums, _) = if full_metrics {
                format_intensities_per_channel(object)
            } else {
                (Vec::new(), Vec::new())
            };
            ModelRc::new(slint::VecModel::from(sums)).into()
        },
        intensity_avgs: {
            let (_, avgs) = if full_metrics {
                format_intensities_per_channel(object)
            } else {
                (Vec::new(), Vec::new())
            };
            ModelRc::new(slint::VecModel::from(avgs)).into()
        },
        circularity: if full_metrics {
            format_circularity(object)
        } else {
            "-".into()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::object_settings::IntensitySettings;
    use indexmap::IndexMap;

    fn object_with_area_and_perimeter(area: usize, perimeter: f32) -> ObjectMetricSettings {
        ObjectMetricSettings {
            area,
            perimeter,
            ..Default::default()
        }
    }

    #[test]
    fn format_circularity_is_empty_for_a_zero_area_object() {
        assert_eq!(
            format_circularity(&object_with_area_and_perimeter(0, 10.0)),
            ""
        );
    }

    #[test]
    fn format_circularity_is_empty_when_perimeter_is_not_positive() {
        assert_eq!(
            format_circularity(&object_with_area_and_perimeter(100, 0.0)),
            ""
        );
        assert_eq!(
            format_circularity(&object_with_area_and_perimeter(100, -1.0)),
            ""
        );
    }

    #[test]
    fn format_circularity_of_a_perfect_circle_is_one() {
        // area = pi * r^2, perimeter = 2 * pi * r -> circularity should be ~1.0.
        let r = 10.0f32;
        let area = (std::f32::consts::PI * r * r) as usize;
        let perimeter = 2.0 * std::f32::consts::PI * r;
        let result = format_circularity(&object_with_area_and_perimeter(area, perimeter));
        assert_eq!(result, "1.00");
    }

    #[test]
    fn format_circularity_is_clamped_to_one_for_measurement_noise() {
        // A tiny/jagged mask can push the raw formula slightly above 1.0 -
        // the result must still clamp to "1.00", not overshoot.
        let result = format_circularity(&object_with_area_and_perimeter(1000, 1.0));
        assert_eq!(result, "1.00");
    }

    #[test]
    fn format_intensities_per_channel_is_empty_with_no_channel_data() {
        let object = ObjectMetricSettings::default();
        let (sums, avgs) = format_intensities_per_channel(&object);
        assert!(sums.is_empty());
        assert!(avgs.is_empty());
    }

    #[test]
    fn format_intensities_per_channel_sizes_the_vec_to_the_highest_channel_id() {
        let mut object = ObjectMetricSettings {
            area: 10,
            ..Default::default()
        };
        let mut intensities = IndexMap::new();
        intensities.insert(
            0,
            IntensitySettings {
                sum_intensity: 100.0,
                ..Default::default()
            },
        );
        intensities.insert(
            2,
            IntensitySettings {
                sum_intensity: 50.0,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&object);
        // max channel id is 2, so the vecs must be length 3 (0, 1, 2).
        assert_eq!(sums.len(), 3);
        assert_eq!(avgs.len(), 3);
        assert_eq!(sums[0], "100.0");
        assert_eq!(
            sums[1], "",
            "no data for channel 1 - left as the default empty string"
        );
        assert_eq!(sums[2], "50.0");
        assert_eq!(avgs[0], "10.0"); // 100 / area(10)
        assert_eq!(avgs[2], "5.0"); // 50 / area(10)
    }

    #[test]
    fn format_intensities_per_channel_leaves_avg_empty_when_area_is_zero() {
        let mut object = ObjectMetricSettings {
            area: 0,
            ..Default::default()
        };
        let mut intensities = IndexMap::new();
        intensities.insert(
            0,
            IntensitySettings {
                sum_intensity: 100.0,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&object);
        assert_eq!(sums[0], "100.0");
        assert_eq!(avgs[0], "", "can't average over a zero-area object");
    }

    #[test]
    fn format_area_nm2_picks_the_right_unit_suffix() {
        let unit_px = PixelSizeSettings {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        assert_eq!(format_area_nm2(500, &unit_px), "500.0");
        assert_eq!(format_area_nm2(1500, &unit_px), "1.5 k");
        assert_eq!(format_area_nm2(2_500_000, &unit_px), "2.50 M");
    }

    #[test]
    fn format_area_nm2_scales_by_pixel_size() {
        let px = PixelSizeSettings {
            x: 2.0,
            y: 3.0,
            z: 1.0,
        };
        // 100 px * 2.0 * 3.0 = 600.0 nm^2
        assert_eq!(format_area_nm2(100, &px), "600.0");
    }

    #[test]
    fn format_area_nm2_boundary_is_inclusive_of_the_next_unit() {
        let unit_px = PixelSizeSettings {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        assert_eq!(format_area_nm2(1000, &unit_px), "1.0 k");
        assert_eq!(format_area_nm2(1_000_000, &unit_px), "1.00 M");
    }

    // -- compute_visible_rows / precompute_label_counts -----------------------

    use crate::editor::test_support::{project_with_one_image, test_ui_state_with_project};
    use std::collections::HashSet;

    fn object_with_class(id: u128, seg_class: u32, has_class: bool) -> ObjectMetricSettings {
        let mut object_class = HashSet::new();
        if has_class {
            object_class.insert(ObjectClass::Valid(1));
        }
        ObjectMetricSettings {
            id: evanalyzer_cfg::core_types::ObjectId(id),
            segmentation_class: SegmentationClass(seg_class),
            object_class,
            ..Default::default()
        }
    }

    #[test]
    fn compute_visible_rows_hides_unclassified_objects_by_default() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, false)); // unclassified
        project.add_object(&object_with_class(2, 1, true)); // classified
        let ui_state = test_ui_state_with_project(project);

        assert!(
            ui_state.get_project().hide_unclassified_objects(),
            "hide_unclassified_objects defaults to true"
        );
        assert_eq!(compute_visible_rows(&ui_state), vec![1]);
    }

    #[test]
    fn compute_visible_rows_includes_every_row_once_hiding_is_toggled_off() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, false));
        project.add_object(&object_with_class(2, 1, true));
        project.toggle_hide_unclassified_objects();
        let ui_state = test_ui_state_with_project(project);

        assert_eq!(compute_visible_rows(&ui_state), vec![0, 1]);
    }

    #[test]
    fn compute_visible_rows_chains_preview_objects_after_manual_ones() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true)); // manual, index 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, 1, true)); // preview, index 1
        let ui_state = test_ui_state_with_project(project);

        assert_eq!(compute_visible_rows(&ui_state), vec![0, 1]);
    }

    #[test]
    fn precompute_label_counts_counts_per_segmentation_class_across_manual_and_preview() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true));
        project.add_object(&object_with_class(2, 1, true));
        project.add_object(&object_with_class(3, 2, true));
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(4, 1, true));
        let ui_state = test_ui_state_with_project(project);

        let counts = precompute_label_counts(&ui_state);

        assert_eq!(counts.get(&SegmentationClass(1)), Some(&3));
        assert_eq!(counts.get(&SegmentationClass(2)), Some(&1));
    }

    #[test]
    fn precompute_label_counts_is_empty_for_a_project_with_no_objects() {
        let ui_state = test_ui_state_with_project(project_with_one_image());

        assert!(precompute_label_counts(&ui_state).is_empty());
    }

    // -- object_rust_to_object_slint -------------------------------------------

    fn class(
        id: u32,
        name: &str,
        color: u32,
    ) -> evanalyzer_cfg::settings::classification_settings::Class {
        evanalyzer_cfg::settings::classification_settings::Class {
            id: ObjectClass::Valid(id),
            color,
            name: name.to_string(),
            notes: String::new(),
        }
    }

    #[test]
    fn object_rust_to_object_slint_resolves_a_registered_classs_name() {
        let mut project = ProjectWithRuntime::default();
        project
            .classification
            .classes_mut()
            .push(class(1, "Nuclei", 0x112233));
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Valid(1));

        let slint_obj = object_rust_to_object_slint(&object, &project, 3, true, 0);

        assert_eq!(slint_obj.display_name.as_str(), "Nuclei");
        assert_eq!(slint_obj.label_count, 3);
        let names: Vec<String> = slint_obj
            .class_names
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(names, vec!["Nuclei".to_string()]);
    }

    #[test]
    fn object_rust_to_object_slint_falls_back_to_unclassified_for_an_unregistered_class() {
        let project = ProjectWithRuntime::default(); // no classes registered
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Valid(99));

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 0);

        assert_eq!(slint_obj.display_name.as_str(), "Unclassified");
    }

    #[test]
    fn object_rust_to_object_slint_joins_multiple_class_names_with_a_comma() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "A", 0));
        project.classification.classes_mut().push(class(2, "B", 0));
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Valid(1));
        object.object_class.insert(ObjectClass::Valid(2));

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 0);

        // HashSet iteration order isn't guaranteed - just check both names
        // appear, comma-joined, rather than asserting an exact order.
        let dn = slint_obj.display_name.to_string();
        assert!(
            dn.contains('A') && dn.contains('B') && dn.contains(','),
            "got {dn:?}"
        );
    }

    #[test]
    fn object_rust_to_object_slint_id_is_the_row_index_plus_one() {
        let project = ProjectWithRuntime::default();
        let object = ObjectMetricSettings::default();

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 41);

        assert_eq!(slint_obj.id, 42);
    }

    #[test]
    fn object_rust_to_object_slint_partial_metrics_blanks_area_and_circularity() {
        let project = ProjectWithRuntime::default();
        let object = object_with_area_and_perimeter(500, 10.0);

        let full = object_rust_to_object_slint(&object, &project, 0, true, 0);
        let partial = object_rust_to_object_slint(&object, &project, 0, false, 0);

        assert!(!full.area_nm2.is_empty());
        assert_eq!(partial.area_nm2.as_str(), "");
        assert_eq!(partial.circularity.as_str(), "-");
    }

    // -- ObjectModalBridge (Model impl) ----------------------------------------

    fn bridge_for(ui_state: &Arc<UiState>) -> ObjectModalBridge {
        ObjectModalBridge {
            app_state: ui_state.clone(),
            notify: ModelNotify::default(),
            label_counts: precompute_label_counts(ui_state),
            visible_rows: compute_visible_rows(ui_state),
        }
    }

    #[test]
    fn object_modal_bridge_row_count_matches_the_filtered_row_count() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, false)); // hidden by default
        project.add_object(&object_with_class(2, 1, true));
        let ui_state = test_ui_state_with_project(project);

        let bridge = bridge_for(&ui_state);

        assert_eq!(bridge.row_count(), 1);
    }

    #[test]
    fn object_modal_bridge_row_data_maps_filtered_rows_back_to_the_underlying_position() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, false)); // filtered out, underlying row 0
        project.add_object(&object_with_class(2, 1, true)); // kept, underlying row 1
        let ui_state = test_ui_state_with_project(project);
        let bridge = bridge_for(&ui_state);

        // Row 0 of the *filtered* list is underlying row 1 - its `id` (which
        // encodes `row_index + 1` against the unfiltered list) must reflect
        // that, not the filtered position.
        let row0 = bridge.row_data(0).expect("one visible row");
        assert_eq!(row0.id, 2);
        assert!(bridge.row_data(1).is_none());
    }

    #[test]
    fn object_modal_bridge_row_data_reads_past_the_manual_list_from_preview_objects() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true)); // manual, underlying row 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, 1, true)); // preview, underlying row 1
        let ui_state = test_ui_state_with_project(project);
        let bridge = bridge_for(&ui_state);

        assert_eq!(bridge.row_count(), 2);
        assert_eq!(bridge.row_data(0).unwrap().id, 1);
        assert_eq!(bridge.row_data(1).unwrap().id, 2);
    }

    #[test]
    fn object_modal_bridge_row_data_out_of_range_returns_none() {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let bridge = bridge_for(&ui_state);

        assert!(bridge.row_data(0).is_none());
    }
}
