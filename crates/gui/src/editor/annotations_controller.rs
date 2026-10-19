//! The manual-annotation actions (`AnnotationState`): copying pipeline
//! (preview) objects into the manual annotations, deleting manual
//! annotations in bulk, GeoJSON export and import.

use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::object_list_controller::{ObjectListController, resolve_object_id};
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::color_from_hex;
use crate::{
    AnnotationClassRow, AnnotationState, AppWindow, DialogType, FileRequest, GlobalAppState,
    WarningState,
};
use evanalyzer_app::project::{
    AnnotationScope, AnnotationsExt, ExportReport, ImportReport, ProjectExt, ProjectWithRuntime,
    apply_import, export_geojson, measure_on_source, measure_pending, read_geojson,
};
use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct AnnotationController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
    pub(crate) viewport_controller: Arc<ViewportController>,
}

impl AnnotationController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        object_list_controller: Arc<ObjectListController>,
        classification_controller: Arc<ClassificationController>,
        viewport_controller: Arc<ViewportController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            object_list_controller,
            classification_controller,
            viewport_controller,
        }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AnnotationState>();

        let m = self.clone();
        state.on_open_copy_dialog_for(move |row_id| m.open_copy_dialog_for(row_id));

        let m = self.clone();
        state.on_open_copy_dialog(move || m.open_copy_dialog());

        let m = self.clone();
        state.on_copy_all_to_manual(move |target_class| m.copy_all_to_manual(target_class));

        let m = self.clone();
        state.on_open_delete_dialog(move || m.open_delete_dialog());

        let m = self.clone();
        state.on_delete_selection_changed(move |all_images| m.delete_selection_changed(all_images));

        let m = self.clone();
        state.on_delete_manual(move |all_images| m.delete_manual(all_images));

        let m = self.clone();
        state.on_export_geojson(move || m.export_geojson());

        let m = self.clone();
        state.on_import_geojson(move |replace| m.import_geojson(replace));

        let m = self.clone();
        state.on_remeasure(move || m.remeasure());
    }

    /// Measures these manual annotations of the open image on its file, in
    /// the background (just drawn or changed with the brush): every channel
    /// at full resolution, whatever the zoom. A shape changed meanwhile
    /// keeps its values (its own measurement follows).
    pub fn measure_in_background(self: &Arc<Self>, ids: Vec<ObjectId>) {
        if ids.is_empty() {
            return;
        }
        let snapshot = {
            let project = self.app_state.get_project();
            let (Some(path), Some(rel_path)) = (
                project.get_current_image_path_cloned(),
                project.get_current_rel_image_path_cloned(),
            ) else {
                return;
            };
            let objects: Vec<ObjectMetricSettings> = project
                .get_objects()
                .unwrap_or_default()
                .iter()
                .filter(|o| ids.contains(&o.id))
                .cloned()
                .collect();
            (path, rel_path, project.get_selected_series_idx(), objects)
        };
        let m = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let (path, rel_path, series, mut objects) = snapshot;
            let measured = m
                .app_state
                .get_image_source(&path)
                .and_then(|source| measure_on_source(source.as_ref(), series, &mut objects));
            if let Err(e) = measured {
                log::warn!("Annotations of {} not measured: {e}", path.display());
                return;
            }
            // Measured values, not an edit of their own: no undo step.
            store_measurements(
                &mut m.app_state.get_project_runtime_write(),
                &rel_path,
                series,
                objects,
            );
            m.object_list_controller.sync_objects_to_slint();
            m.object_list_controller
                .sync_selected_object_to_slint(false);
        });
    }

    /// Measures every manual annotation of the project again on its image
    /// (e.g. ones from before measuring was exact), in the background.
    fn remeasure(self: &Arc<Self>) {
        let images: Vec<(PathBuf, PathBuf, i32, Vec<ObjectMetricSettings>)> = {
            let project = self.app_state.get_project();
            let Some(root) = project.settings.images.root.clone() else {
                return;
            };
            project
                .settings
                .images
                .list
                .iter()
                .flat_map(|(rel, image)| {
                    let root = root.clone();
                    image
                        .series
                        .iter()
                        .filter(|(_, s)| !s.objects.is_empty())
                        .map(move |(series, s)| {
                            (root.join(rel), rel.clone(), *series, s.objects.clone())
                        })
                })
                .collect()
        };
        let m = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let (mut measured, mut failed) = (0, Vec::new());
            let backend = m.app_state.backend();
            for (path, rel_path, series, mut objects) in images {
                // Opened for this only - not kept like the viewer's image.
                let result = backend
                    .open_image(&path)
                    .and_then(|source| measure_on_source(source.as_ref(), series, &mut objects));
                match result {
                    Ok(()) => {
                        measured += objects.len();
                        store_measurements(
                            &mut m.app_state.get_project_runtime_write(),
                            &rel_path,
                            series,
                            objects,
                        );
                    }
                    Err(e) => {
                        log::warn!("Annotations of {} not measured: {e}", path.display());
                        failed.push(rel_path);
                    }
                }
            }
            m.refresh();
            let mut message = format!(
                "{} measured on their images.",
                plural(measured, "annotation", "annotations")
            );
            if !failed.is_empty() {
                message.push_str(&format!(
                    "\nNot measured (image file not readable): {}.",
                    plural(failed.len(), "image", "images")
                ));
            }
            m.show_message("Annotations measured", message, true);
        });
    }

    /// Shows the copy dialog for one pipeline object: its own classes, or
    /// another one.
    fn open_copy_dialog_for(&self, row_id: i32) {
        let Some(ui) = self.ui.upgrade() else { return };
        let rows = {
            let project = self.app_state.get_project();
            let Some(object) = resolve_object_id(&project, row_id)
                .and_then(|id| project.get_preview_objects().iter().find(|o| o.id == id))
            else {
                return;
            };
            let mut classes: Vec<(ObjectClass, usize)> =
                object.object_class.iter().map(|c| (c.clone(), 1)).collect();
            classes.sort_by_key(|(c, _)| c.to_i32());
            class_rows(&project, &classes)
        };
        let state = ui.global::<AnnotationState>();
        state.set_pipeline_classes(ModelRc::new(VecModel::from(rows)));
        state.set_copy_single(row_id);
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::CopyToManual);
    }

    /// Fills the copy dialog with the classes of the open image's pipeline
    /// objects and shows it.
    fn open_copy_dialog(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let rows = {
            let project = self.app_state.get_project();
            class_rows(&project, &project.preview_object_classes())
        };
        let state = ui.global::<AnnotationState>();
        state.set_pipeline_classes(ModelRc::new(VecModel::from(rows)));
        state.set_copy_single(-1);
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::CopyToManual);
    }

    /// Copies the pipeline objects of the ticked classes; `target_class`:
    /// the class the copies get, -1 their own.
    fn copy_all_to_manual(self: &Arc<Self>, target_class: i32) {
        let Some(ui) = self.ui.upgrade() else { return };
        let target = (target_class >= 0).then(|| class_from_row_id(target_class));
        let single = ui.global::<AnnotationState>().get_copy_single();
        if single >= 0 {
            ui.global::<AnnotationState>().set_copy_single(-1);
            return self.copy_one_to_manual(single, target);
        }
        let classes = ticked(&ui.global::<AnnotationState>().get_pipeline_classes());
        if classes.is_empty() {
            return; // nothing ticked - not "all"
        }
        self.app_state.begin_undo_step();
        let mut project = self.app_state.get_project_write();
        let ids = project.preview_object_ids(&classes);
        let copied = project.copy_preview_objects_to_manual(&ids, target);
        drop(project);
        log::info!(
            "Copied {} pipeline objects to the manual annotations",
            copied.len()
        );
        self.refresh();
    }

    /// One pipeline object copied, with its classes or `target`; the copy
    /// is selected.
    fn copy_one_to_manual(self: &Arc<Self>, row_id: i32, target: Option<ObjectClass>) {
        self.app_state.begin_undo_step();
        let mut project = self.app_state.get_project_write();
        let Some(id) = resolve_object_id(&project, row_id) else {
            return;
        };
        if let Some(copy) = project.copy_preview_objects_to_manual(&[id], target).pop() {
            project.set_selected_object(Some(copy));
        }
        drop(project);
        self.refresh();
    }

    /// Fills the delete dialog (the open image's classes and count) and
    /// shows it.
    fn open_delete_dialog(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let (rows, (objects, images)) = {
            let project = self.app_state.get_project();
            (
                class_rows(
                    &project,
                    &project.manual_annotation_classes(AnnotationScope::CurrentImage),
                ),
                project.count_manual_annotations(AnnotationScope::CurrentImage, &[]),
            )
        };
        let state = ui.global::<AnnotationState>();
        state.set_manual_classes(ModelRc::new(VecModel::from(rows)));
        state.set_delete_count(objects as i32);
        state.set_delete_summary(
            delete_summary(objects, images, AnnotationScope::CurrentImage).into(),
        );
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::DeleteManualAnnotations);
    }

    /// Ticks or scope changed in the delete dialog: the classes and counts
    /// of the scope (ticks kept), what Delete would delete.
    fn delete_selection_changed(&self, all_images: bool) {
        let Some(ui) = self.ui.upgrade() else { return };
        let state = ui.global::<AnnotationState>();
        let rows = state.get_manual_classes();
        let unticked: Vec<i32> = rows.iter().filter(|r| !r.checked).map(|r| r.id).collect();
        let project = self.app_state.get_project();
        let mut new_rows = class_rows(
            &project,
            &project.manual_annotation_classes(scope(all_images)),
        );
        for row in &mut new_rows {
            row.checked = !unticked.contains(&row.id);
        }
        let classes: Vec<ObjectClass> = new_rows
            .iter()
            .filter(|r| r.checked)
            .map(|r| class_from_row_id(r.id))
            .collect();
        let (objects, images) = if classes.is_empty() {
            (0, 0)
        } else {
            project.count_manual_annotations(scope(all_images), &classes)
        };
        drop(project);
        if rows.row_count() != new_rows.len()
            || rows
                .iter()
                .zip(&new_rows)
                .any(|(a, b)| a.id != b.id || a.count != b.count)
        {
            state.set_manual_classes(ModelRc::new(VecModel::from(new_rows)));
        }
        state.set_delete_count(objects as i32);
        state.set_delete_summary(delete_summary(objects, images, scope(all_images)).into());
    }

    /// Deletes the manual annotations of the ticked classes, on the open
    /// image or all images.
    fn delete_manual(self: &Arc<Self>, all_images: bool) {
        let Some(ui) = self.ui.upgrade() else { return };
        let classes = ticked(&ui.global::<AnnotationState>().get_manual_classes());
        if classes.is_empty() {
            return; // nothing ticked - not "all"
        }
        self.app_state.begin_undo_step();
        let mut project = self.app_state.get_project_write();
        let deleted = project.delete_manual_annotations(scope(all_images), &classes);
        project.set_selected_object(None);
        drop(project);
        log::info!("Deleted {deleted} manual annotations");
        self.refresh();
    }

    /// Picks a folder, writes every image's manual annotations there.
    fn export_geojson(self: &Arc<Self>) {
        if !self.local_only() {
            return;
        }
        let mut request = FileRequest::open_folder("Export annotations to");
        if let Some(dir) = self.project_folder() {
            request = request.start_in(dir);
        }
        let m = self.clone();
        self.app_state.file_browser.open(request, move |folder| {
            let Some(folder) = folder else { return };
            let result = export_geojson(&m.app_state.get_project(), &folder);
            match result {
                Ok(report) => m.show_message(
                    "Annotations exported",
                    export_message(&report, &folder),
                    true,
                ),
                Err(e) => m.show_message("Export failed", e, false),
            }
        });
    }

    /// Picks a folder, reads its .geojson files into the matching images:
    /// added, or replacing their manual annotations. Reads and measures on
    /// a worker thread (measuring reads the images), then puts them in.
    fn import_geojson(self: &Arc<Self>, replace: bool) {
        if !self.local_only() {
            return;
        }
        let mut request = FileRequest::open_folder("Import annotations from");
        if let Some(dir) = self.project_folder() {
            request = request.start_in(dir);
        }
        let m = self.clone();
        self.app_state.file_browser.open(request, move |folder| {
            let Some(folder) = folder else { return };
            let m = m.clone();
            crate::helper::ui_thread::spawn(move || {
                let read = read_geojson(&m.app_state.get_project(), &folder);
                let mut pending = match read {
                    Ok(pending) => pending,
                    Err(e) => return m.show_message("Import failed", e, false),
                };
                let backend = m.app_state.backend();
                measure_pending(&mut pending, |path, series, objects| {
                    backend
                        .open_image(path)
                        .and_then(|source| measure_on_source(source.as_ref(), series, objects))
                        .map_err(|e| e.to_string())
                });
                m.app_state.begin_undo_step();
                let report = apply_import(&mut m.app_state.get_project_write(), pending, replace);
                if report.objects > 0 || replace {
                    m.refresh();
                }
                m.show_message("Annotations imported", import_message(&report), true);
            });
        });
    }

    /// After the manual annotations changed: the lists, the class counts,
    /// the image.
    fn refresh(&self) {
        self.app_state.mark_dirty();
        self.object_list_controller.sync_objects_to_slint();
        self.object_list_controller
            .sync_selected_object_to_slint(true);
        self.classification_controller
            .sync_classification_to_slint();
        self.viewport_controller.trigger_image_redraw_objects();
    }

    /// The info (or warning) dialog with `message`.
    fn show_message(&self, title: &str, message: String, info: bool) {
        let ui = self.ui.clone();
        let title = title.to_string();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(info);
                warning.set_title(title.into());
                warning.set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        });
    }

    /// The folder of the project file, where the file browser starts.
    fn project_folder(&self) -> Option<std::path::PathBuf> {
        let project = self.app_state.get_project();
        let file = project.tmp_settings.current_project.as_ref()?;
        file.parent().map(|p| p.to_path_buf())
    }

    /// GeoJSON files are read and written on this computer: not while the
    /// project is on a server.
    fn local_only(&self) -> bool {
        if self.app_state.backend().is_remote() {
            self.show_message(
                "Not on a server",
                "Annotations are exported and imported on this computer - not while working on a server.".into(),
                false,
            );
            return false;
        }
        true
    }
}

/// Puts measured intensities into the annotations they belong to - if
/// they still have the shape that was measured.
fn store_measurements(
    project: &mut ProjectWithRuntime,
    rel_path: &Path,
    series: i32,
    measured: Vec<ObjectMetricSettings>,
) {
    let Some(series) = project
        .settings
        .images
        .list
        .get_mut(rel_path)
        .and_then(|image| image.series.get_mut(&series))
    else {
        return;
    };
    for m in measured {
        if let Some(object) = series
            .objects
            .iter_mut()
            .find(|o| o.id == m.id && o.bbox == m.bbox && o.mask_data == m.mask_data)
        {
            object.intensities = m.intensities;
        }
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn export_message(report: &ExportReport, folder: &std::path::Path) -> String {
    if report.objects == 0 {
        return "There are no manual annotations to export.".into();
    }
    format!(
        "{} written to {} in {}.",
        plural(report.objects, "annotation", "annotations"),
        plural(report.files, "file", "files"),
        folder.display()
    )
}

fn import_message(report: &ImportReport) -> String {
    let mut lines = vec![format!(
        "{} added to {}.",
        plural(report.objects, "annotation", "annotations"),
        plural(report.images, "image", "images")
    )];
    if report.duplicates > 0 {
        lines.push(format!(
            "{} already there (the same pixels).",
            plural(report.duplicates, "annotation was", "annotations were")
        ));
    }
    if !report.created_classes.is_empty() {
        lines.push(format!(
            "New classes: {}.",
            report.created_classes.join(", ")
        ));
    }
    if report.skipped_features > 0 {
        lines.push(format!(
            "{} skipped (points, lines or empty shapes).",
            plural(report.skipped_features, "feature", "features")
        ));
    }
    if !report.unmeasured_images.is_empty() {
        lines.push(format!(
            "Not measured (image file not readable): {} - their intensities are missing.",
            plural(report.unmeasured_images.len(), "image", "images")
        ));
    }
    if !report.unmatched_files.is_empty() {
        let names: Vec<String> = report
            .unmatched_files
            .iter()
            .take(5)
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .collect();
        let more = report.unmatched_files.len().saturating_sub(names.len());
        lines.push(format!(
            "No image in the project for {}: {}{}.",
            plural(report.unmatched_files.len(), "file", "files"),
            names.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        ));
    }
    lines.join("\n")
}

/// `AnnotationClassRow::id` of a class: its id, -1 for none.
fn class_from_row_id(id: i32) -> ObjectClass {
    if id < 0 {
        ObjectClass::Unset
    } else {
        ObjectClass::Global(id as u32)
    }
}

/// Rows for `classes` (with their counts), all ticked.
fn class_rows(
    project: &ProjectWithRuntime,
    classes: &[(ObjectClass, usize)],
) -> Vec<AnnotationClassRow> {
    classes
        .iter()
        .map(|(class, count)| {
            let (name, color) = match project.get_class_from_id(class) {
                Some(c) if *class != ObjectClass::Unset => {
                    (c.name.clone(), color_from_hex(c.color))
                }
                _ => ("Unclassified".to_string(), color_from_hex(0x888888)),
            };
            AnnotationClassRow {
                id: class.to_i32(),
                name: name.into(),
                color,
                count: *count as i32,
                checked: true,
            }
        })
        .collect()
}

/// The ticked classes of `rows`.
fn ticked(rows: &ModelRc<AnnotationClassRow>) -> Vec<ObjectClass> {
    rows.iter()
        .filter(|row| row.checked)
        .map(|row| class_from_row_id(row.id))
        .collect()
}

/// What Delete would delete: "312 annotations on 14 images".
fn delete_summary(objects: usize, images: usize, scope: AnnotationScope) -> String {
    let annotations = if objects == 1 {
        "annotation"
    } else {
        "annotations"
    };
    match scope {
        AnnotationScope::CurrentImage => format!("{objects} {annotations} on this image."),
        AnnotationScope::AllImages => {
            let on = if images == 1 { "image" } else { "images" };
            format!("{objects} {annotations} on {images} {on}.")
        }
    }
}

fn scope(all_images: bool) -> AnnotationScope {
    if all_images {
        AnnotationScope::AllImages
    } else {
        AnnotationScope::CurrentImage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_import_message_lists_what_happened() {
        let report = ImportReport {
            images: 2,
            objects: 34,
            duplicates: 1,
            created_classes: vec!["Tumor".into()],
            unmatched_files: vec!["x/a.geojson".into()],
            skipped_features: 3,
            unmeasured_images: vec![],
        };
        assert_eq!(
            import_message(&report),
            "34 annotations added to 2 images.\n1 annotation was already there (the same pixels).\n\
             New classes: Tumor.\n3 features skipped (points, lines or empty shapes).\n\
             No image in the project for 1 file: a.geojson."
        );
        assert_eq!(
            export_message(
                &ExportReport {
                    files: 1,
                    objects: 1
                },
                std::path::Path::new("/d")
            ),
            "1 annotation written to 1 file in /d."
        );
    }

    #[test]
    fn measurements_go_only_to_annotations_still_of_the_measured_shape() {
        use bitvec::prelude::*;
        use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
        use evanalyzer_cfg::settings::object_settings::IntensitySettings;

        let shape = |id: u128, bits: BitVec<u64, Lsb0>| ObjectMetricSettings {
            id: ObjectId(id),
            bbox: [0, 0, 1, 0],
            mask_data: bits,
            ..Default::default()
        };
        let mut project = ProjectWithRuntime::default();
        let mut entry = ImageEntry::default();
        entry.series.insert(
            0,
            SeriesSettings {
                objects: vec![
                    shape(1, bitvec![u64, Lsb0; 1, 1]),
                    shape(2, bitvec![u64, Lsb0; 1, 0]),
                ],
                ..Default::default()
            },
        );
        project
            .settings
            .images
            .list
            .insert(PathBuf::from("a.tif"), entry);

        let mut measured_1 = shape(1, bitvec![u64, Lsb0; 1, 1]);
        // Object 2 was measured before a brush stroke changed it.
        let mut measured_2 = shape(2, bitvec![u64, Lsb0; 1, 1]);
        for m in [&mut measured_1, &mut measured_2] {
            m.intensities.insert(
                0,
                IntensitySettings {
                    sum_intensity: 7.0,
                    ..Default::default()
                },
            );
        }
        store_measurements(
            &mut project,
            Path::new("a.tif"),
            0,
            vec![measured_1, measured_2],
        );
        let objects = &project.settings.images.list[Path::new("a.tif")].series[&0].objects;
        assert_eq!(objects[0].intensities[&0].sum_intensity, 7.0);
        assert!(
            objects[1].intensities.is_empty(),
            "changed since: its own measurement follows"
        );
    }

    #[test]
    fn row_ids_map_back_to_classes() {
        assert_eq!(class_from_row_id(-1), ObjectClass::Unset);
        assert_eq!(class_from_row_id(3), ObjectClass::Global(3));
        assert_eq!(
            class_from_row_id(ObjectClass::Global(7).to_i32()),
            ObjectClass::Global(7)
        );
    }

    #[test]
    fn the_delete_summary_says_how_many_and_where() {
        assert_eq!(
            delete_summary(1, 1, AnnotationScope::CurrentImage),
            "1 annotation on this image."
        );
        assert_eq!(
            delete_summary(312, 14, AnnotationScope::AllImages),
            "312 annotations on 14 images."
        );
        assert_eq!(
            delete_summary(0, 0, AnnotationScope::AllImages),
            "0 annotations on 0 images."
        );
    }

    #[test]
    fn only_ticked_rows_count() {
        let rows = ModelRc::new(VecModel::from(vec![
            AnnotationClassRow {
                id: 1,
                checked: true,
                ..Default::default()
            },
            AnnotationClassRow {
                id: 2,
                checked: false,
                ..Default::default()
            },
            AnnotationClassRow {
                id: -1,
                checked: true,
                ..Default::default()
            },
        ]));
        assert_eq!(
            ticked(&rows),
            vec![ObjectClass::Global(1), ObjectClass::Unset]
        );
    }
}
