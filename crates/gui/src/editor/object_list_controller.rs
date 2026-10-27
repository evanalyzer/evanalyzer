//! # object_list_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::{color_from_hex, get_colors_from_class};
use crate::{AppWindow, ObjectItemDataSlint, ObjectListState};
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_app::project::{AnnotationsExt, ObjectExt, ProjectExt};
use evanalyzer_cfg::core_types::{ObjectId, SegmentationClass, max_gray_value};
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
    /// Bit depth of the current image - precomputed at bridge creation so
    /// `row_data` doesn't look up the image metadata once per row.
    nr_of_bits: Option<u16>,
}

/// Resolves a Slint 1-based `object_id` (as sent by every `ObjectListState`
/// callback) into the actual `ObjectId` it refers to, branching between the
/// manual object list and the live-preview object list exactly like
/// `ObjectModalBridge`'s row indexing does - `object_id` is a position into
/// the *combined* (manual ++ preview) list, not the manual list alone, so
/// every callback must resolve it this way rather than indexing straight
/// into `get_objects()`.
pub(crate) fn resolve_object_id(project: &ProjectWithRuntime, object_id: i32) -> Option<ObjectId> {
    if object_id <= 0 {
        return None;
    }
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
                    // A selection is no edit: no undo step.
                    let mut project = manager.app_state.get_project_runtime_write();
                    let selected = resolve_object_id(&project, object_id);
                    project.set_selected_object(selected);
                    drop(project);
                    manager.sync_selected_object_to_slint(false);
                    manager.viewport_controller.trigger_image_redraw_objects();
                });

            // The class filter or the search changed
            let manager = self.clone();
            ui.global::<ObjectListState>()
                .on_filter_changed(move || manager.sync_objects_to_slint());
        }

        // Another plane or frame: its objects are listed.
        let manager = Arc::downgrade(self);
        self.app_state.on_plane_changed(move |_, _| {
            let Some(manager) = manager.upgrade() else {
                return;
            };
            let any_object = {
                let project = manager.app_state.get_project();
                !project.get_objects().unwrap_or(&[]).is_empty()
                    || !project.get_preview_objects().is_empty()
            };
            if any_object {
                manager.sync_objects_to_slint();
            }
        });
    }

    pub fn sync_objects_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();

        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let label_counts = precompute_label_counts(&bridge_ptr.app_state);
                let state = ui.global::<ObjectListState>();
                let filter = ObjectFilter {
                    source: ObjectSource::from_slint(state.get_filter_source()),
                    ..ObjectFilter::parse(state.get_filter_class(), &state.get_filter_text())
                };
                let visible_rows = compute_visible_rows(&bridge_ptr.app_state, &filter);
                let nr_of_bits = current_image_nr_of_bits(&bridge_ptr.app_state);
                let bridge = Rc::new(ObjectModalBridge {
                    app_state: bridge_ptr.app_state.clone(),
                    notify: ModelNotify::default(),
                    label_counts,
                    visible_rows,
                    nr_of_bits,
                });
                let model_rc = ModelRc::new(bridge);
                ui.global::<ObjectListState>().set_object_list(model_rc);
                // The preview's measurements go with its objects.
                crate::editor::measurements_panel::sync(&ui, &bridge_ptr.app_state);
            }
        }) {
            warn!("Failed to sync ROIs to Slint: {}", e);
        }
    }

    pub fn sync_selected_object_to_slint(self: &Arc<Self>, scroll_to: bool) {
        let ui_weak = self.ui.clone();
        let bridge_ptr = self.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let class_state = ui.global::<ObjectListState>();
                // Before taking the project lock - this takes it itself.
                let nr_of_bits = current_image_nr_of_bits(&bridge_ptr.app_state);
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
                    let (parents, children) = relations_text(&object, &preview_objects, &project);
                    class_state.set_selected_object(ObjectItemDataSlint {
                        editable: project.is_manual_object(object.id.clone()),
                        parents: parents.into(),
                        children: children.into(),
                        ..object_rust_to_object_slint(
                            &object,
                            &project,
                            label_count,
                            true,
                            index,
                            nr_of_bits,
                        )
                    });
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
                // A manual annotation; preview objects are read-only.
                ObjectItemDataSlint {
                    editable: true,
                    ..object_rust_to_object_slint(
                        object,
                        &project,
                        count,
                        false,
                        underlying_row as i32,
                        self.nr_of_bits,
                    )
                }
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
                        self.nr_of_bits,
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
/// the list panel, respecting `hide_unclassified_objects` and `filter`.
fn compute_visible_rows(app_state: &Arc<UiState>, filter: &ObjectFilter) -> Vec<usize> {
    let project = app_state.get_project();
    let hide_unclassified = project.hide_unclassified_objects();
    let manual = project.get_objects().unwrap_or(&[]);
    let preview = project.get_preview_objects();
    let shown = project.get_shown_plane();

    manual
        .iter()
        .chain(preview.iter())
        .enumerate()
        .filter(|(i, _)| filter.source.shows(*i < manual.len()))
        // The objects of the plane (or z-projection) and frame shown.
        .filter(|(_, object)| object.mask_on_plane(shown).is_some())
        .filter(|(_, object)| !hide_unclassified || !object.object_class.is_empty())
        .filter(|(_, object)| filter.matches(object, &project))
        .map(|(i, _)| i)
        .collect()
}

/// What the objects list / table shows: manual annotations and/or
/// detections, a class (`None` all) and the search terms - all must match.
#[derive(Debug, Default, PartialEq)]
struct ObjectFilter {
    source: ObjectSource,
    class: Option<u32>,
    terms: Vec<ObjectTerm>,
}

/// Where the objects shown come from (`ObjectListState.filter-source`).
#[derive(Debug, Default, PartialEq, Clone, Copy)]
enum ObjectSource {
    #[default]
    Both,
    /// The manual annotations only.
    Manual,
    /// The pipeline's detections (preview objects) only.
    Detections,
}

impl ObjectSource {
    /// 0 both, 1 manual annotations, 2 detections.
    fn from_slint(source: i32) -> Self {
        match source {
            1 => ObjectSource::Manual,
            2 => ObjectSource::Detections,
            _ => ObjectSource::Both,
        }
    }

    fn shows(self, manual: bool) -> bool {
        match self {
            ObjectSource::Both => true,
            ObjectSource::Manual => manual,
            ObjectSource::Detections => !manual,
        }
    }
}

#[derive(Debug, PartialEq)]
enum ObjectTerm {
    /// "12": the object's id (its segmentation label).
    Id(u32),
    /// "area > 50" (`>`, `>=`, `<`, `<=`, `=`).
    Area(Compare, f64),
    /// "circ < 0.6" / "circularity …".
    Circularity(Compare, f64),
    /// Anything else: part of a class name (any case).
    Text(String),
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Compare {
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Equal,
}

impl Compare {
    fn holds(self, value: f64, limit: f64) -> bool {
        match self {
            Compare::Greater => value > limit,
            Compare::GreaterOrEqual => value >= limit,
            Compare::Less => value < limit,
            Compare::LessOrEqual => value <= limit,
            Compare::Equal => (value - limit).abs() < 1e-9,
        }
    }
}

impl ObjectFilter {
    /// `class` -1 for all; `text` the search ("area > 50, nuclei").
    fn parse(class: i32, text: &str) -> Self {
        let terms = text
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ObjectTerm::parse)
            .collect();
        Self {
            source: ObjectSource::Both,
            class: u32::try_from(class).ok(),
            terms,
        }
    }

    fn matches(&self, object: &ObjectMetricSettings, project: &ProjectWithRuntime) -> bool {
        if let Some(class) = self.class
            && !object
                .object_class
                .contains(&evanalyzer_cfg::core_types::ObjectClass::Global(class))
        {
            return false;
        }
        self.terms.iter().all(|term| term.matches(object, project))
    }
}

impl ObjectTerm {
    fn parse(term: &str) -> Self {
        let lower = term.to_lowercase();
        if let Ok(id) = lower.parse::<u32>() {
            return ObjectTerm::Id(id);
        }
        for (name, make) in [
            (
                "circularity",
                ObjectTerm::Circularity as fn(Compare, f64) -> ObjectTerm,
            ),
            ("circ", ObjectTerm::Circularity),
            ("area", ObjectTerm::Area),
        ] {
            if let Some(rest) = lower.strip_prefix(name) {
                let rest = rest.trim_start();
                for (op, compare) in [
                    (">=", Compare::GreaterOrEqual),
                    ("<=", Compare::LessOrEqual),
                    (">", Compare::Greater),
                    ("<", Compare::Less),
                    ("=", Compare::Equal),
                ] {
                    if let Some(number) = rest.strip_prefix(op)
                        && let Ok(limit) = number.trim().parse::<f64>()
                    {
                        return make(compare, limit);
                    }
                }
            }
        }
        ObjectTerm::Text(lower)
    }

    fn matches(&self, object: &ObjectMetricSettings, project: &ProjectWithRuntime) -> bool {
        match self {
            ObjectTerm::Id(id) => object.segmentation_class.as_u32() == *id,
            ObjectTerm::Area(compare, limit) => compare.holds(object.area as f64, *limit),
            ObjectTerm::Circularity(compare, limit) => {
                object.area > 0
                    && object.perimeter > 0.0
                    && compare.holds(
                        ((4.0 * std::f64::consts::PI * object.area as f64)
                            / (object.perimeter as f64 * object.perimeter as f64))
                            .min(1.0),
                        *limit,
                    )
            }
            ObjectTerm::Text(text) => object.object_class.iter().any(|class| {
                project
                    .get_class_from_id(class)
                    .is_some_and(|c| c.name.to_lowercase().contains(text.as_str()))
            }),
        }
    }
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

/// Bit depth of the current image's selected series (resolution 0) - the
/// same value the pipeline normalized the object intensities with.
/// `None` when no image is open or its metadata can't be read.
fn current_image_nr_of_bits(app_state: &UiState) -> Option<u16> {
    let (path, series) = {
        let project = app_state.get_project();
        (
            project.get_current_image_path_cloned()?,
            project.get_selected_series_idx(),
        )
    };
    let source = app_state.get_image_source(&path).ok()?;
    source
        .meta()
        .series
        .get(&series)?
        .resolutions
        .get(&0)
        .map(|pyramid| pyramid.nr_bits)
}

/// Per-channel sum and average intensity as gray values (normalized value
/// times the bit depth's max, as in ImageJ/Fiji). Channels with data show
/// "-" when the bit depth is unknown or implausible, rather than a
/// misleading normalized number.
fn format_intensities_per_channel(
    object: &ObjectMetricSettings,
    nr_of_bits: Option<u16>,
) -> (Vec<SharedString>, Vec<SharedString>) {
    let bit_max = nr_of_bits.and_then(max_gray_value);
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
            let Some(bit_max) = bit_max else {
                sums[i] = "-".into();
                avgs[i] = "-".into();
                continue;
            };
            sums[i] = format!("{:.1}", intensities.sum_intensity * bit_max).into();
            if area > 0.0 {
                avgs[i] = format!("{:.1}", intensities.avg_intensity * bit_max).into();
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

/// "CY5 · 2, FITC · 1": the classes `object` colocalizes with and how many
/// of their objects - "" for none.
fn coloc_text(object: &ObjectMetricSettings, project: &ProjectWithRuntime) -> String {
    object
        .colocalized_with
        .iter()
        .filter(|(_, partners)| !partners.is_empty())
        .map(|(class, partners)| {
            let name = project
                .get_class_from_id(class)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Class {}", class.to_i32()));
            format!("{name} · {}", partners.len())
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A class's name, or "Class N" for one the project doesn't have.
fn class_name(
    class: &evanalyzer_cfg::core_types::ObjectClass,
    project: &ProjectWithRuntime,
) -> String {
    project
        .get_class_from_id(class)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| format!("Class {}", class.to_i32()))
}

/// The object card's Relate Objects lines: the classes of the objects
/// `object` belongs to ("Cell, Region"), and how many objects of `others`
/// belong to it, per their class ("Spot · 7").
fn relations_text(
    object: &ObjectMetricSettings,
    others: &[ObjectMetricSettings],
    project: &ProjectWithRuntime,
) -> (String, String) {
    let parents = object
        .parents
        .keys()
        .map(|class| class_name(class, project))
        .collect::<Vec<_>>()
        .join(", ");
    let mut children: Vec<(String, usize)> = Vec::new();
    for child in others
        .iter()
        .filter(|o| o.parents.values().any(|p| *p == object.id))
    {
        let mut classes: Vec<&evanalyzer_cfg::core_types::ObjectClass> =
            child.object_class.iter().collect();
        classes.sort();
        let name = classes
            .first()
            .map_or_else(|| "Unclassified".to_string(), |c| class_name(c, project));
        match children.iter_mut().find(|(n, _)| *n == name) {
            Some((_, count)) => *count += 1,
            None => children.push((name, 1)),
        }
    }
    let children = children
        .iter()
        .map(|(name, count)| format!("{name} · {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    (parents, children)
}

fn object_rust_to_object_slint(
    object: &ObjectMetricSettings,
    project: &ProjectWithRuntime,
    label_count: i32,
    full_metrics: bool,
    row_index: i32,
    nr_of_bits: Option<u16>,
) -> ObjectItemDataSlint {
    let mut class_names_vec: Vec<SharedString> = Vec::new();
    let mut class_colors_vec: Vec<Color> = Vec::new();
    let mut class_ids_vec: Vec<i32> = Vec::new();
    let mut display_name = String::new();

    for class in &object.object_class {
        let (class_name, class_color) = match project.get_class_from_id(class) {
            Some(class_data) => (class_data.name.clone(), color_from_hex(class_data.color)),
            _ => ("Unclassified".to_string(), color_from_hex(0xff0000)),
        };

        if display_name.is_empty() {
            display_name = class_name.clone();
        } else {
            display_name.push(',');
            display_name.push_str(&class_name);
        }

        class_names_vec.push(class_name.into());
        class_colors_vec.push(class_color);
        class_ids_vec.push(class.to_i32());
    }

    let display_color = get_colors_from_class(project, 255, &object.object_class);
    let pixel_sizes = project.get_pixel_sizes();
    let (sums, avgs) = format_intensities_per_channel(object, nr_of_bits);

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
        // Set by the caller, which knows the list the object is in.
        editable: false,
        label_count,
        area: object.area as i32,
        area_nm2: if full_metrics && object.area > 0 {
            format_area_nm2(object.area, &pixel_sizes)
        } else {
            "".into()
        },
        // The means and the circularity fill the objects table's columns, so
        // every row has them; the sums only the selected object. Cheap: the
        // model builds only the rows in view.
        intensities: ModelRc::new(slint::VecModel::from(if full_metrics {
            sums
        } else {
            Vec::new()
        }))
        .into(),
        intensity_avgs: ModelRc::new(slint::VecModel::from(avgs)).into(),
        circularity: format_circularity(object),
        coloc_with: if full_metrics {
            coloc_text(object, project).into()
        } else {
            "".into()
        },
        // Set by the caller, which has the other objects.
        parents: "".into(),
        children: "".into(),
        bbox_x: object.bbox[0] as f32,
        bbox_y: object.bbox[1] as f32,
        bbox_w: (object.bbox[2].saturating_sub(object.bbox[0]) + 1) as f32,
        bbox_h: (object.bbox[3].saturating_sub(object.bbox[1]) + 1) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectClass;
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
        let (sums, avgs) = format_intensities_per_channel(&object, Some(16));
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
                sum_intensity: 2.0,
                avg_intensity: 0.2,
                ..Default::default()
            },
        );
        intensities.insert(
            2,
            IntensitySettings {
                sum_intensity: 1.0,
                avg_intensity: 0.1,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&object, Some(8));
        // max channel id is 2, so the vecs must be length 3 (0, 1, 2).
        assert_eq!(sums.len(), 3);
        assert_eq!(avgs.len(), 3);
        assert_eq!(sums[0], "510.0"); // 2.0 * 255
        assert_eq!(
            sums[1], "",
            "no data for channel 1 - left as the default empty string"
        );
        assert_eq!(sums[2], "255.0");
        assert_eq!(avgs[0], "51.0"); // 0.2 * 255
        assert_eq!(avgs[2], "25.5");
    }

    #[test]
    fn format_intensities_per_channel_scales_by_the_bit_depths_max() {
        let mut object = ObjectMetricSettings {
            area: 1,
            ..Default::default()
        };
        let mut intensities = IndexMap::new();
        intensities.insert(
            0,
            IntensitySettings {
                sum_intensity: 1.0,
                avg_intensity: 1.0,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&object, Some(16));
        assert_eq!(sums[0], "65535.0");
        assert_eq!(avgs[0], "65535.0");
        let (sums, _) = format_intensities_per_channel(&object, Some(12));
        assert_eq!(sums[0], "4095.0");
    }

    #[test]
    fn format_intensities_per_channel_shows_a_dash_without_a_valid_bit_depth() {
        let mut object = ObjectMetricSettings {
            area: 1,
            ..Default::default()
        };
        let mut intensities = IndexMap::new();
        intensities.insert(
            0,
            IntensitySettings {
                sum_intensity: 0.5,
                avg_intensity: 0.5,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        for nr_of_bits in [None, Some(0), Some(64)] {
            let (sums, avgs) = format_intensities_per_channel(&object, nr_of_bits);
            assert_eq!(sums[0], "-", "bit depth {nr_of_bits:?}");
            assert_eq!(avgs[0], "-", "bit depth {nr_of_bits:?}");
        }
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
                sum_intensity: 2.0,
                ..Default::default()
            },
        );
        object.intensities = intensities;

        let (sums, avgs) = format_intensities_per_channel(&object, Some(8));
        assert_eq!(sums[0], "510.0");
        assert_eq!(avgs[0], "", "can't average over a zero-area object");
    }

    #[test]
    fn format_area_nm2_picks_the_right_unit_suffix() {
        let unit_px = PixelSizeSettings {
            x: 1.0,
            y: 1.0,
            z: None,
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
            z: None,
        };
        // 100 px * 2.0 * 3.0 = 600.0 nm^2
        assert_eq!(format_area_nm2(100, &px), "600.0");
    }

    #[test]
    fn format_area_nm2_boundary_is_inclusive_of_the_next_unit() {
        let unit_px = PixelSizeSettings {
            x: 1.0,
            y: 1.0,
            z: None,
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
            object_class.insert(ObjectClass::Global(1));
        }
        ObjectMetricSettings {
            id: evanalyzer_cfg::core_types::ObjectId(id),
            segmentation_class: SegmentationClass(seg_class),
            object_class,
            ..Default::default()
        }
    }

    #[test]
    fn the_search_reads_ids_comparisons_and_class_names() {
        let filter = ObjectFilter::parse(-1, "12, area > 50, circ<=0.6, Nuc");
        assert_eq!(filter.class, None);
        assert_eq!(
            filter.terms,
            vec![
                ObjectTerm::Id(12),
                ObjectTerm::Area(Compare::Greater, 50.0),
                ObjectTerm::Circularity(Compare::LessOrEqual, 0.6),
                ObjectTerm::Text("nuc".into()),
            ]
        );
        assert_eq!(ObjectFilter::parse(3, " ").class, Some(3));
        assert!(ObjectFilter::parse(3, " ").terms.is_empty());
    }

    #[test]
    fn the_filter_shows_the_objects_of_a_class_matching_the_search() {
        let project = ProjectWithRuntime::default();
        let mut big = object_with_class(1, 7, true);
        big.area = 100;
        let mut small = object_with_class(2, 8, true);
        small.area = 10;
        let unclassified = object_with_class(3, 9, false);

        let shows = |filter: &ObjectFilter| {
            [&big, &small, &unclassified]
                .iter()
                .map(|o| filter.matches(o, &project))
                .collect::<Vec<_>>()
        };
        assert_eq!(shows(&ObjectFilter::parse(-1, "")), [true, true, true]);
        assert_eq!(shows(&ObjectFilter::parse(1, "")), [true, true, false]);
        assert_eq!(
            shows(&ObjectFilter::parse(-1, "area >= 50")),
            [true, false, false]
        );
        assert_eq!(shows(&ObjectFilter::parse(-1, "8")), [false, true, false]);
        assert_eq!(
            shows(&ObjectFilter::parse(1, "area < 50")),
            [false, true, false]
        );
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
        assert_eq!(
            compute_visible_rows(&ui_state, &ObjectFilter::default()),
            vec![1]
        );
    }

    #[test]
    fn compute_visible_rows_includes_every_row_once_hiding_is_toggled_off() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, false));
        project.add_object(&object_with_class(2, 1, true));
        project.toggle_hide_unclassified_objects();
        let ui_state = test_ui_state_with_project(project);

        assert_eq!(
            compute_visible_rows(&ui_state, &ObjectFilter::default()),
            vec![0, 1]
        );
    }

    #[test]
    fn compute_visible_rows_shows_only_manual_annotations_or_only_detections() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true)); // manual, index 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, 1, true)); // detection, index 1
        let ui_state = test_ui_state_with_project(project);

        let rows = |source: i32| {
            let filter = ObjectFilter {
                source: ObjectSource::from_slint(source),
                ..ObjectFilter::default()
            };
            compute_visible_rows(&ui_state, &filter)
        };
        assert_eq!(rows(0), vec![0, 1]);
        assert_eq!(rows(1), vec![0]);
        assert_eq!(rows(2), vec![1]);
    }

    /// A volume's object (z-planes 2 to 3) is listed on its planes, a
    /// plane's object (z-plane 2) on its plane only.
    #[test]
    fn compute_visible_rows_lists_objects_on_their_planes() {
        use evanalyzer_cfg::core_types::ImageDimensionality;
        use evanalyzer_cfg::settings::images_settings::{ZStackHandling, ZStackSettings};
        let rows = |z: i32| {
            let mut project = project_with_one_image();
            let mut planar = object_with_class(1, 1, true); // index 0
            planar.z_stack = 2;
            project.add_object(&planar);
            let mut volume = object_with_class(2, 1, true); // index 1
            // A 1-pixel box, set on both planes.
            volume.bbox = [0, 0, 0, 0];
            volume.mask_data = bitvec::bitvec![u64, bitvec::order::Lsb0; 1; 2];
            volume.z_stack = 2;
            volume.dimensionality = ImageDimensionality::Volume { depth: 2 };
            project.add_object(&volume);
            project.set_global_z_stack(&ZStackSettings {
                z_projection: ZStackHandling::Volume3d,
                z_range: Some(z..=z),
            });
            compute_visible_rows(
                &test_ui_state_with_project(project),
                &ObjectFilter::default(),
            )
        };
        assert_eq!(rows(1), Vec::<usize>::new());
        assert_eq!(rows(2), vec![0, 1]);
        assert_eq!(rows(3), vec![1]);
        assert_eq!(rows(4), Vec::<usize>::new());
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

        assert_eq!(
            compute_visible_rows(&ui_state, &ObjectFilter::default()),
            vec![0, 1]
        );
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
            id,
            color,
            name: name.to_string(),
            notes: String::new(),
        }
    }

    /// The object card's "Coloc with": each class and how many of its
    /// objects - for the selected object only.
    #[test]
    fn the_selected_object_says_what_it_colocalizes_with() {
        let mut project = ProjectWithRuntime::default();
        project
            .classification
            .classes_mut()
            .push(class(2, "CY5", 0x112233));
        let mut object = ObjectMetricSettings::default();
        object.colocalized_with.insert(
            ObjectClass::Global(2),
            vec![Default::default(), Default::default()],
        );
        object
            .colocalized_with
            .insert(ObjectClass::Global(3), vec![]);

        let selected = object_rust_to_object_slint(&object, &project, 1, true, 0, Some(16));
        assert_eq!(selected.coloc_with.as_str(), "CY5 · 2");
        let row = object_rust_to_object_slint(&object, &project, 1, false, 0, Some(16));
        assert_eq!(row.coloc_with.as_str(), "", "rows don't need it");
    }

    /// The object card names the cell a spot belongs to, and the cell its
    /// spots per class.
    #[test]
    fn the_selected_object_names_its_parent_and_counts_its_children() {
        let mut project = ProjectWithRuntime::default();
        project
            .classification
            .classes_mut()
            .push(class(1, "Cell", 0x112233));
        project
            .classification
            .classes_mut()
            .push(class(2, "Spot", 0x445566));
        let cell = ObjectMetricSettings {
            id: evanalyzer_cfg::core_types::ObjectId(1),
            object_class: [ObjectClass::Global(1)].into(),
            ..Default::default()
        };
        let spot = |id| {
            let mut spot = ObjectMetricSettings {
                id: evanalyzer_cfg::core_types::ObjectId(id),
                object_class: [ObjectClass::Global(2)].into(),
                ..Default::default()
            };
            spot.parents.insert(ObjectClass::Global(1), cell.id.clone());
            spot
        };
        let others = vec![cell.clone(), spot(2), spot(3)];
        assert_eq!(
            relations_text(&cell, &others, &project),
            (String::new(), "Spot · 2".to_string())
        );
        assert_eq!(
            relations_text(&spot(2), &others, &project),
            ("Cell".to_string(), String::new())
        );
    }

    #[test]
    fn object_rust_to_object_slint_resolves_a_registered_classs_name() {
        let mut project = ProjectWithRuntime::default();
        project
            .classification
            .classes_mut()
            .push(class(1, "Nuclei", 0x112233));
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Global(1));

        let slint_obj = object_rust_to_object_slint(&object, &project, 3, true, 0, Some(16));

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
        object.object_class.insert(ObjectClass::Global(99));

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 0, Some(16));

        assert_eq!(slint_obj.display_name.as_str(), "Unclassified");
    }

    #[test]
    fn object_rust_to_object_slint_joins_multiple_class_names_with_a_comma() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes_mut().push(class(1, "A", 0));
        project.classification.classes_mut().push(class(2, "B", 0));
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Global(1));
        object.object_class.insert(ObjectClass::Global(2));

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 0, Some(16));

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

        let slint_obj = object_rust_to_object_slint(&object, &project, 0, true, 41, Some(16));

        assert_eq!(slint_obj.id, 42);
    }

    #[test]
    fn a_list_row_has_the_table_columns_but_not_the_selected_objects_extras() {
        let project = ProjectWithRuntime::default();
        let mut object = object_with_area_and_perimeter(500, 10.0);
        object.intensities.insert(
            0,
            IntensitySettings {
                sum_intensity: 0.5,
                avg_intensity: 0.25,
                ..Default::default()
            },
        );

        let full = object_rust_to_object_slint(&object, &project, 0, true, 0, Some(16));
        let row = object_rust_to_object_slint(&object, &project, 0, false, 0, Some(16));

        // The table's columns: on every row, as on the selected object.
        assert_eq!(row.circularity, full.circularity);
        assert!(!row.circularity.is_empty());
        assert_eq!(
            row.intensity_avgs.row_data(0),
            full.intensity_avgs.row_data(0)
        );
        assert_eq!(row.area, 500);
        // The selected object's extras.
        assert!(!full.area_nm2.is_empty());
        assert_eq!(row.area_nm2.as_str(), "");
        assert_eq!(full.intensities.row_count(), 1);
        assert_eq!(row.intensities.row_count(), 0);
    }

    // -- ObjectModalBridge (Model impl) ----------------------------------------

    fn bridge_for(ui_state: &Arc<UiState>) -> ObjectModalBridge {
        ObjectModalBridge {
            app_state: ui_state.clone(),
            notify: ModelNotify::default(),
            label_counts: precompute_label_counts(ui_state),
            visible_rows: compute_visible_rows(ui_state, &ObjectFilter::default()),
            nr_of_bits: Some(16),
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

    // -- resolve_object_id -----------------------------------------------------

    #[test]
    fn resolve_object_id_returns_none_for_a_non_positive_id() {
        let ui_state = test_ui_state_with_project(project_with_one_image());
        let project = ui_state.get_project();
        assert_eq!(resolve_object_id(&project, 0), None);
        assert_eq!(resolve_object_id(&project, -1), None);
    }

    #[test]
    fn resolve_object_id_resolves_a_manual_object_by_its_combined_index() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true));
        project.add_object(&object_with_class(2, 1, true));
        let ui_state = test_ui_state_with_project(project);
        let project = ui_state.get_project();

        assert_eq!(
            resolve_object_id(&project, 1),
            Some(evanalyzer_cfg::core_types::ObjectId(1))
        );
        assert_eq!(
            resolve_object_id(&project, 2),
            Some(evanalyzer_cfg::core_types::ObjectId(2))
        );
    }

    #[test]
    fn resolve_object_id_resolves_a_preview_object_past_the_end_of_the_manual_list() {
        // This is the exact bug the audit flagged: `object_add_class`,
        // `object_remove_class`, and `object_delete` used to index straight
        // into the manual-only list with the *combined* index, so any id
        // past the manual list's length silently resolved to `None` instead
        // of the matching preview object.
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true)); // manual, combined index 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, 1, true)); // preview, combined index 1
        let ui_state = test_ui_state_with_project(project);
        let project = ui_state.get_project();

        assert_eq!(
            resolve_object_id(&project, 2),
            Some(evanalyzer_cfg::core_types::ObjectId(2)),
            "id 2 (combined index 1) is the preview object - must not resolve to None"
        );
    }

    #[test]
    fn resolve_object_id_out_of_range_past_every_list_returns_none() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true));
        let ui_state = test_ui_state_with_project(project);
        let project = ui_state.get_project();

        assert_eq!(resolve_object_id(&project, 5), None);
    }

    // -- attach_callbacks (live AppWindow) --------------------------------------
    //
    // `object_list_controller.rs` was flagged (see the production-readiness
    // audit) as one of the files where a callback-wiring bug shipped
    // specifically because it wasn't exercised through `test_ui_window()` -
    // only through pure-function tests above. These tests close that gap for
    // the four `ObjectListState` callbacks.

    use crate::editor::test_support::test_ui_window;
    use crate::editor::viewport_controller::ViewportController;

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
        project: ProjectWithRuntime,
    ) -> (Arc<UiState>, ObjectListController) {
        let ui_state = test_ui_state_with_project(project);
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        (
            ui_state.clone(),
            ObjectListController::new(ui, ui_state, viewport_controller),
        )
    }

    #[test]
    fn attach_callbacks_object_selected_resolves_a_preview_object() {
        let mut project = project_with_one_image();
        project.add_object(&object_with_class(1, 1, true)); // manual, combined index 0
        project
            .tmp_settings
            .preview_objects
            .push(object_with_class(2, 1, true)); // preview, combined index 1

        let ui = test_ui_window();
        let (ui_state, controller) = make_controller_with_ui(ui.as_weak(), project);
        let controller = Arc::new(controller);
        controller.attach_callbacks();

        // Selecting id 2 (the preview object) through the real Slint callback
        // must resolve to the preview object's id, not silently select nothing.
        ui.global::<ObjectListState>().invoke_object_selected(2);

        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(evanalyzer_cfg::core_types::ObjectId(2))
        );
    }
}
