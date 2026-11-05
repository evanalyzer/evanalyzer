//! # annotations_controller
//!
//! The manual-annotation actions (`AnnotationState`): copying pipeline
//! (preview) objects into the manual annotations, deleting manual
//! annotations in bulk, GeoJSON export and import.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::object_list_controller::{ObjectListController, resolve_object_id};
use crate::editor::viewport_controller::ViewportController;
use crate::helper::color_generators::color_from_hex;
use crate::{
    AnnotationClassRow, AnnotationState, AppWindow, DialogType, FileRequest, GlobalAppState,
    ImportClassRow, WarningState,
};
use evanalyzer_app::project::ImageListExt;
use evanalyzer_app::project::{
    AnnotationScope, AnnotationsExt, ClassTarget, ImportReport, PendingImport, PipelineObjects,
    ProjectExt, ProjectWithRuntime, apply_import, default_class_target, export_image_geojson,
    image_annotation_classes, measure_on_source, measure_pending, read_image_geojson,
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
    /// A .geojson file read for the open image, waiting for the user's
    /// class mapping (the import dialog).
    pending_import: std::sync::Mutex<Option<ReadImport>>,
}

/// A file read for an image, and what the import dialog's targets mean.
struct ReadImport {
    pending: PendingImport,
    /// The project classes the targets 3 + k stand for.
    classes: Vec<u32>,
}

/// The import dialog's fixed targets, before the project's classes.
const TARGET_NEW: i32 = 0;
const TARGET_NONE: i32 = 1;
const TARGET_SKIP: i32 = 2;

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
            pending_import: std::sync::Mutex::new(None),
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
        state.on_open_export_dialog(move || m.open_export_dialog());

        let m = self.clone();
        state.on_export_geojson(move || m.export_geojson());

        let m = self.clone();
        state.on_pick_import_file(move || m.pick_import_file());

        let m = self.clone();
        state.on_apply_import(move |replace| m.apply_import(replace));

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

    /// The open image (its path in the project) and the series shown, its
    /// name for the dialogs - or a message that none is open.
    fn open_image(&self) -> Option<(PathBuf, i32, String)> {
        let project = self.app_state.get_project();
        let Some(rel_path) = project.get_current_relative_path() else {
            drop(project);
            self.show_message(
                "No image open",
                "Open the image whose annotations to export or import first.".into(),
                true,
            );
            return None;
        };
        let name = rel_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Some((rel_path, project.get_selected_series_idx(), name))
    }

    /// The export dialog: the open image's manual annotations by class,
    /// all ticked.
    fn open_export_dialog(self: &Arc<Self>) {
        let Some((rel_path, _, name)) = self.open_image() else {
            return;
        };
        let Some(ui) = self.ui.upgrade() else { return };
        let rows = {
            let project = self.app_state.get_project();
            // Its manual annotations and the pipeline objects shown on it.
            let classes: Vec<(ObjectClass, usize)> =
                image_annotation_classes(&project, &rel_path, project.get_preview_objects())
                    .into_iter()
                    .map(|(id, count)| (id.map_or(ObjectClass::Unset, ObjectClass::Global), count))
                    .collect();
            class_rows(&project, &classes)
        };
        let state = ui.global::<AnnotationState>();
        state.set_geojson_image(name.into());
        state.set_export_classes(ModelRc::new(VecModel::from(rows)));
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::ExportAnnotations);
    }

    /// Asks where to save (named after the image, changeable), writes the
    /// open image's annotations of the ticked classes there.
    fn export_geojson(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else { return };
        let Some((rel_path, series, name)) = self.open_image() else {
            return;
        };
        let classes: std::collections::HashSet<Option<u32>> =
            ticked(&ui.global::<AnnotationState>().get_export_classes())
                .into_iter()
                .map(|c| match c {
                    ObjectClass::Global(id) => Some(id),
                    _ => None,
                })
                .collect();
        if classes.is_empty() {
            return self.show_message(
                "Nothing exported",
                "Tick the classes to export.".into(),
                true,
            );
        }
        let stem = Path::new(&name)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name);
        let mut request = FileRequest::save_file("Export annotations as")
            .filter("GeoJSON", &["geojson"])
            .file_name(&format!("{stem}.geojson"));
        if let Some(dir) = self.project_folder() {
            request = request.start_in(dir);
        }
        let m = self.clone();
        self.app_state.file_browser.open(request, move |file| {
            let Some(file) = file else { return };
            let file = with_geojson_extension(file);
            let m = m.clone();
            // Written through the backend: on a server, over the network.
            crate::helper::ui_thread::spawn(move || {
                let backend = m.app_state.backend();
                // A snapshot: the project isn't locked while the file is written.
                let (project, pipeline) = {
                    let project = m.app_state.get_project();
                    (
                        ProjectWithRuntime {
                            settings: project.settings.clone(),
                            tmp_settings: Default::default(),
                        },
                        project.get_preview_objects().clone(),
                    )
                };
                let pipeline = PipelineObjects {
                    objects: &pipeline,
                    series,
                };
                match export_image_geojson(
                    &project,
                    backend.files(),
                    &rel_path,
                    &classes,
                    &file,
                    pipeline,
                ) {
                    Ok(count) => m.show_message(
                        "Annotations exported",
                        format!(
                            "{} written to {}.",
                            plural(count, "object", "objects"),
                            file.display()
                        ),
                        true,
                    ),
                    Err(e) => m.show_message("Export failed", e, false),
                }
            });
        });
    }

    /// Asks for a .geojson file, reads it for the open image, then shows
    /// the import dialog: the file's classes, each mapped to its project
    /// class of the same name, else a new one.
    fn pick_import_file(self: &Arc<Self>) {
        let Some((rel_path, series, name)) = self.open_image() else {
            return;
        };
        let mut request = FileRequest::open_file("Import annotations from")
            .filter("GeoJSON", &["geojson", "json"]);
        if let Some(dir) = self.project_folder() {
            request = request.start_in(dir);
        }
        let m = self.clone();
        self.app_state.file_browser.open(request, move |file| {
            let Some(file) = file else { return };
            let m = m.clone();
            crate::helper::ui_thread::spawn(move || {
                let backend = m.app_state.backend();
                let read = read_image_geojson(
                    &m.app_state.get_project(),
                    backend.files(),
                    &file,
                    &rel_path,
                    series,
                );
                let pending = match read {
                    Ok(pending) => pending,
                    Err(e) => return m.show_message("Import failed", e, false),
                };
                let ImportTargets {
                    rows,
                    targets,
                    colors,
                    classes,
                } = {
                    let project = m.app_state.get_project();
                    import_rows(&project, &pending)
                };
                let shapes = pending.shapes() as i32;
                *m.pending_import.lock().expect("Poisoned") = Some(ReadImport { pending, classes });
                let file_name = file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let ui = m.ui.clone();
                let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                    let Some(ui) = ui.upgrade() else { return };
                    let state = ui.global::<AnnotationState>();
                    state.set_geojson_image(name.into());
                    state.set_import_file(file_name.into());
                    state.set_import_shapes(shapes);
                    state.set_import_classes(ModelRc::new(VecModel::from(rows)));
                    state.set_import_targets(ModelRc::new(VecModel::from(targets)));
                    state.set_import_target_colors(ModelRc::new(VecModel::from(colors)));
                    ui.global::<GlobalAppState>()
                        .set_active_dialog(DialogType::ImportAnnotations);
                });
            });
        });
    }

    /// The read file into the open image with the dialog's class mapping:
    /// measured on the image (a worker thread - it reads the image), then
    /// added - or replacing its manual annotations.
    fn apply_import(self: &Arc<Self>, replace: bool) {
        let Some(ui) = self.ui.upgrade() else { return };
        let Some(read) = self.pending_import.lock().expect("Poisoned").take() else {
            return;
        };
        let mapping: std::collections::HashMap<String, ClassTarget> = ui
            .global::<AnnotationState>()
            .get_import_classes()
            .iter()
            .map(|row| (row.name.to_string(), target_of(row.target, &read.classes)))
            .collect();
        let m = self.clone();
        crate::helper::ui_thread::spawn(move || {
            let backend = m.app_state.backend();
            let mut pending = read.pending;
            measure_pending(&mut pending, |path, series, objects| {
                backend
                    .open_image(path)
                    .and_then(|source| measure_on_source(source.as_ref(), series, objects))
                    .map_err(|e| e.to_string())
            });
            m.app_state.begin_undo_step();
            let report = {
                let mut project = m.app_state.get_project_write();
                apply_import(&mut project, pending, replace, &mapping)
            };
            if report.objects > 0 || replace || !report.created_classes.is_empty() {
                m.refresh();
            }
            m.show_message("Annotations imported", import_message(&report), true);
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

/// The import dialog's rows (the file's class names, each with its
/// default target), its targets, the project classes behind them.
fn import_rows(project: &ProjectWithRuntime, pending: &PendingImport) -> ImportTargets {
    let classes: Vec<(u32, String, u32)> = project
        .settings
        .classification
        .classes()
        .iter()
        .filter(|c| c.id != 0)
        .map(|c| (c.id, c.name.clone(), c.color))
        .collect();
    let mut targets: Vec<slint::SharedString> =
        vec!["New class".into(), "No class".into(), "Don't import".into()];
    targets.extend(
        classes
            .iter()
            .map(|(_, name, _)| slint::SharedString::from(name.as_str())),
    );
    // The project classes' colors beside their names; none for the first three.
    let mut colors = vec![slint::Color::from_argb_u8(0, 0, 0, 0); 3];
    colors.extend(classes.iter().map(|(_, _, color)| color_from_hex(*color)));
    let ids: Vec<u32> = classes.iter().map(|(id, _, _)| *id).collect();
    let rows = pending
        .file_classes()
        .into_iter()
        .map(|class| {
            let target = match default_class_target(project, &class.name) {
                ClassTarget::Class(id) => ids
                    .iter()
                    .position(|i| *i == id)
                    .map_or(TARGET_NEW, |k| 3 + k as i32),
                ClassTarget::NoClass => TARGET_NONE,
                ClassTarget::Skip => TARGET_SKIP,
                ClassTarget::New => TARGET_NEW,
            };
            ImportClassRow {
                name: class.name.into(),
                count: class.count as i32,
                color: color_from_hex(class.color.unwrap_or(0x888888)),
                target,
            }
        })
        .collect();
    ImportTargets {
        rows,
        targets,
        colors,
        classes: ids,
    }
}

/// The import dialog's content: [`import_rows`].
struct ImportTargets {
    rows: Vec<ImportClassRow>,
    targets: Vec<slint::SharedString>,
    colors: Vec<slint::Color>,
    /// The project classes the targets 3 + k stand for.
    classes: Vec<u32>,
}

/// A target of the import dialog as a mapping.
fn target_of(target: i32, classes: &[u32]) -> ClassTarget {
    match target {
        TARGET_NONE => ClassTarget::NoClass,
        TARGET_SKIP => ClassTarget::Skip,
        t if t >= 3 => classes
            .get((t - 3) as usize)
            .map_or(ClassTarget::New, |id| ClassTarget::Class(*id)),
        _ => ClassTarget::New,
    }
}

/// A name typed without `.geojson` gets it.
fn with_geojson_extension(file: PathBuf) -> PathBuf {
    if file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("geojson") || e.eq_ignore_ascii_case("json"))
    {
        file
    } else {
        let mut name = file.into_os_string();
        name.push(".geojson");
        PathBuf::from(name)
    }
}

fn import_message(report: &ImportReport) -> String {
    let mut lines = vec![format!(
        "{} added to the image.",
        plural(report.objects, "annotation", "annotations")
    )];
    if report.skipped_by_class > 0 {
        lines.push(format!(
            "{} not imported (their class is set to \"Don't import\").",
            plural(report.skipped_by_class, "shape", "shapes")
        ));
    }
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
        lines.push(
            "Not measured (the image file can't be read) - their intensities are missing.".into(),
        );
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
    use evanalyzer_app::project::ImageListExt;

    #[test]
    fn the_import_message_lists_what_happened() {
        let report = ImportReport {
            objects: 34,
            duplicates: 1,
            created_classes: vec!["Tumor".into()],
            skipped_features: 3,
            skipped_by_class: 2,
            unmeasured_images: vec![],
        };
        assert_eq!(
            import_message(&report),
            "34 annotations added to the image.\n2 shapes not imported (their class is set to \"Don't import\").\n\
             1 annotation was already there (the same pixels).\nNew classes: Tumor.\n\
             3 features skipped (points, lines or empty shapes)."
        );
    }

    /// The import dialog: each class name of the file defaults to the
    /// project class of that name, else a new one; its targets map back.
    #[test]
    fn the_import_dialog_maps_file_classes_onto_the_projects() {
        use evanalyzer_cfg::settings::classification_settings::Class;
        use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
        let mut project = ProjectWithRuntime::default();
        let mut entry = ImageEntry::default();
        entry.series.insert(
            0,
            SeriesSettings {
                image_width: 20,
                image_height: 20,
                ..Default::default()
            },
        );
        project
            .settings
            .images
            .list
            .insert(PathBuf::from("a.tif"), entry);
        let nuclei = 5;
        project.settings.classification.classes_mut().push(Class {
            id: nuclei,
            color: 0x3080f0,
            name: "Nuclei".into(),
            notes: String::new(),
        });
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("qupath.geojson");
        let shape = |class: Option<&str>| {
            let mut properties = serde_json::json!({ "objectType": "annotation" });
            if let Some(name) = class {
                properties["classification"] = serde_json::json!({ "name": name });
            }
            serde_json::json!({
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": [[[1, 1], [4, 1], [4, 3], [1, 3], [1, 1]]] },
                "properties": properties
            })
        };
        std::fs::write(
            &file,
            serde_json::json!([shape(Some("nuclei")), shape(Some("Tumor")), shape(None)])
                .to_string(),
        )
        .unwrap();
        let pending = read_image_geojson(
            &project,
            &evanalyzer_app::fs::LocalFileSystem::default(),
            &file,
            Path::new("a.tif"),
            0,
        )
        .unwrap();

        let ImportTargets {
            rows,
            targets,
            colors,
            classes,
        } = import_rows(&project, &pending);
        let names: Vec<(String, i32)> = rows
            .iter()
            .map(|r| (r.name.to_string(), r.target))
            .collect();
        assert_eq!(
            names,
            vec![
                ("nuclei".into(), 3),
                ("Tumor".into(), TARGET_NEW),
                (String::new(), TARGET_NONE)
            ]
        );
        assert_eq!(targets[3], "Nuclei");
        assert_eq!(
            colors[3],
            color_from_hex(0x3080f0),
            "beside its name, its colour"
        );
        assert_eq!(colors[0].alpha(), 0, "none for New class");
        assert_eq!(target_of(3, &classes), ClassTarget::Class(nuclei));
        assert_eq!(target_of(TARGET_SKIP, &classes), ClassTarget::Skip);
        assert_eq!(target_of(TARGET_NONE, &classes), ClassTarget::NoClass);
        assert_eq!(target_of(TARGET_NEW, &classes), ClassTarget::New);
    }

    #[test]
    fn a_typed_export_name_gets_the_geojson_extension() {
        assert_eq!(
            with_geojson_extension("/d/cells".into()),
            PathBuf::from("/d/cells.geojson")
        );
        assert_eq!(
            with_geojson_extension("/d/cells.GeoJSON".into()),
            PathBuf::from("/d/cells.GeoJSON")
        );
        assert_eq!(
            with_geojson_extension("/d/a.tif".into()),
            PathBuf::from("/d/a.tif.geojson")
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
        let objects = &project
            .settings
            .images
            .list
            .get(Path::new("a.tif"))
            .unwrap()
            .series[&0]
            .objects;
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

    // --- The controller on a window --------------------------------------

    use crate::editor::test_support::{
        choose_file, fixture_image_path, project_with_fixture_image, project_with_one_image,
        test_ui_window, ui_state_with_window,
    };
    use crate::editor::viewport_cache::ViewportCache;
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{FileBrowserState, UiState};
    use bitvec::prelude::*;
    use evanalyzer_cfg::settings::classification_settings::Class;

    const NUCLEI: ObjectClass = ObjectClass::Global(5);
    const TUMOR: ObjectClass = ObjectClass::Global(6);

    /// A 2x2 object at x, of `classes`.
    fn object(id: u128, classes: &[ObjectClass], x: u32) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: ObjectId(id),
            object_class: classes.iter().cloned().collect(),
            bbox: [x, 0, x + 1, 1],
            mask_data: bitvec![u64, Lsb0; 1, 1, 1, 1],
            area: 4,
            ..Default::default()
        }
    }

    /// `project` with the classes Nuclei (5) and Tumor (6), its image 20x20.
    fn with_classes(mut project: ProjectWithRuntime) -> ProjectWithRuntime {
        for (id, name, color) in [(5, "Nuclei", 0x3080f0), (6, "Tumor", 0xf03030)] {
            project.settings.classification.classes_mut().push(Class {
                id,
                color,
                name: name.into(),
                notes: String::new(),
            });
        }
        for image in project.settings.images.list.values_mut() {
            for series in image.series.values_mut() {
                series.image_width = 20;
                series.image_height = 20;
            }
        }
        project
    }

    /// The one-image project with these manual annotations and pipeline
    /// objects.
    fn project_with(
        manual: Vec<ObjectMetricSettings>,
        preview: Vec<ObjectMetricSettings>,
    ) -> ProjectWithRuntime {
        let mut project = with_classes(project_with_one_image());
        project.get_selected_image_series_mut().unwrap().objects = manual;
        project.tmp_settings.preview_objects = preview;
        project
    }

    /// The controller with its collaborators on a headless window, its
    /// callbacks attached.
    fn controller(
        project: ProjectWithRuntime,
    ) -> (AppWindow, Arc<UiState>, Arc<AnnotationController>) {
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, project);
        let weak = ui.as_weak();
        let viewport = Arc::new(ViewportController::new(
            weak.clone(),
            ui_state.clone(),
            Arc::new(ViewportCache::new(ui_state.clone())),
        ));
        let object_list = Arc::new(ObjectListController::new(
            weak.clone(),
            ui_state.clone(),
            viewport.clone(),
        ));
        let classification = Arc::new(ClassificationController::new(
            weak.clone(),
            ui_state.clone(),
            object_list.clone(),
            viewport.clone(),
        ));
        let controller = Arc::new(AnnotationController::new(
            weak,
            ui_state.clone(),
            object_list,
            classification,
            viewport,
        ));
        controller.attach_callbacks();
        (ui, ui_state, controller)
    }

    fn manual_objects(ui_state: &UiState) -> Vec<ObjectMetricSettings> {
        ui_state
            .get_project()
            .get_objects()
            .unwrap_or_default()
            .to_vec()
    }

    fn rows(model: ModelRc<AnnotationClassRow>) -> Vec<(i32, String, i32)> {
        model
            .iter()
            .map(|r| (r.id, r.name.to_string(), r.count))
            .collect()
    }

    /// Unticks the row of `class` in `model`.
    fn untick(model: &ModelRc<AnnotationClassRow>, class: &ObjectClass) {
        let i = model.iter().position(|r| r.id == class.to_i32()).unwrap();
        let mut row = model.row_data(i).unwrap();
        row.checked = false;
        model.set_row_data(i, row);
    }

    /// The message dialog shown: title, message, info (not a warning).
    fn message(ui: &AppWindow) -> Option<(String, String, bool)> {
        (ui.global::<GlobalAppState>().get_active_dialog() == DialogType::Warning).then(|| {
            let w = ui.global::<WarningState>();
            (
                w.get_title().to_string(),
                w.get_message().to_string(),
                w.get_info(),
            )
        })
    }

    /// The copy dialog lists the pipeline objects' classes, most first,
    /// unclassified by name.
    #[test]
    fn the_copy_dialog_lists_the_pipeline_classes_by_count() {
        let (ui, _ui_state, _c) = controller(project_with(
            vec![],
            vec![
                object(1, &[NUCLEI], 0),
                object(2, &[NUCLEI], 3),
                object(3, &[], 6),
            ],
        ));
        let state = ui.global::<AnnotationState>();
        state.set_copy_single(4);
        state.invoke_open_copy_dialog();
        assert_eq!(
            rows(state.get_pipeline_classes()),
            vec![(5, "Nuclei".into(), 2), (-1, "Unclassified".into(), 1)]
        );
        assert!(state.get_pipeline_classes().iter().all(|r| r.checked));
        assert_eq!(state.get_copy_single(), -1, "all, not one");
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::CopyToManual
        );
    }

    /// Copying copies the pipeline objects of the ticked classes only, with
    /// their own classes, as one undo step.
    #[test]
    fn copying_takes_only_the_ticked_classes() {
        let (ui, ui_state, _c) = controller(project_with(
            vec![],
            vec![
                object(1, &[NUCLEI], 0),
                object(2, &[TUMOR], 3),
                object(3, &[], 6),
            ],
        ));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_copy_dialog();
        untick(&state.get_pipeline_classes(), &TUMOR);
        state.invoke_copy_all_to_manual(-1);
        drain_ui_queue();

        let copied = manual_objects(&ui_state);
        let mut classes: Vec<_> = copied.iter().map(|o| o.object_class.clone()).collect();
        classes.sort();
        assert_eq!(classes, vec![[].into(), [NUCLEI].into()]);
        assert!(
            copied
                .iter()
                .all(|o| o.id != ObjectId(1) && o.id != ObjectId(3))
        );
        assert_eq!(
            ui_state.get_project().get_preview_objects().len(),
            3,
            "the originals stay"
        );
        assert!(ui_state.is_dirty());
        assert!(ui_state.undo());
        assert!(manual_objects(&ui_state).is_empty(), "undone in one step");
    }

    /// A target class: every copy gets it instead of its own.
    #[test]
    fn copies_get_the_target_class() {
        let (ui, ui_state, _c) = controller(project_with(
            vec![],
            vec![object(1, &[NUCLEI], 0), object(2, &[], 3)],
        ));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_copy_dialog();
        state.invoke_copy_all_to_manual(TUMOR.to_i32());
        let classes: Vec<_> = manual_objects(&ui_state)
            .into_iter()
            .map(|o| o.object_class)
            .collect();
        assert_eq!(classes, vec![[TUMOR].into(), [TUMOR].into()]);
    }

    /// Nothing ticked copies nothing - not "all".
    #[test]
    fn copying_with_nothing_ticked_copies_nothing() {
        let (ui, ui_state, _c) = controller(project_with(vec![], vec![object(1, &[NUCLEI], 0)]));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_copy_dialog();
        untick(&state.get_pipeline_classes(), &NUCLEI);
        state.invoke_copy_all_to_manual(-1);
        assert!(manual_objects(&ui_state).is_empty());
        assert!(!ui_state.is_dirty());
    }

    /// The dialog for one pipeline object (its row after the manual ones):
    /// its classes; copying copies only it and selects the copy.
    #[test]
    fn copying_one_pipeline_object_selects_its_copy() {
        let (ui, ui_state, _c) = controller(project_with(
            vec![object(10, &[], 10)],
            vec![object(1, &[NUCLEI], 0), object(2, &[TUMOR, NUCLEI], 3)],
        ));
        let state = ui.global::<AnnotationState>();
        // Row 1 is the manual annotation, 2 and 3 the pipeline objects.
        state.invoke_open_copy_dialog_for(3);
        assert_eq!(
            rows(state.get_pipeline_classes()),
            vec![(5, "Nuclei".into(), 1), (6, "Tumor".into(), 1)]
        );
        assert_eq!(state.get_copy_single(), 3);
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::CopyToManual
        );

        state.invoke_copy_all_to_manual(-1);
        assert_eq!(state.get_copy_single(), -1, "reset for the next dialog");
        let objects = manual_objects(&ui_state);
        assert_eq!(objects.len(), 2);
        let copy = &objects[1];
        assert_eq!(copy.object_class, [NUCLEI, TUMOR].into());
        assert_eq!(copy.bbox, [3, 0, 4, 1]);
        assert_eq!(
            ui_state.get_project().get_selected_object_id(),
            Some(copy.id.clone())
        );
    }

    /// One object copied to a target class gets only that class.
    #[test]
    fn copying_one_pipeline_object_to_a_class() {
        let (ui, ui_state, _c) = controller(project_with(vec![], vec![object(1, &[NUCLEI], 0)]));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_copy_dialog_for(1);
        state.invoke_copy_all_to_manual(TUMOR.to_i32());
        assert_eq!(manual_objects(&ui_state)[0].object_class, [TUMOR].into());
    }

    /// A row that is no pipeline object (a manual one, or none) opens no
    /// dialog.
    #[test]
    fn the_copy_dialog_for_a_row_that_is_no_pipeline_object_stays_closed() {
        let (ui, _ui_state, _c) = controller(project_with(
            vec![object(10, &[], 10)],
            vec![object(1, &[NUCLEI], 0)],
        ));
        let state = ui.global::<AnnotationState>();
        for row in [0, 1, 3] {
            state.invoke_open_copy_dialog_for(row);
            assert_ne!(
                ui.global::<GlobalAppState>().get_active_dialog(),
                DialogType::CopyToManual,
                "row {row}"
            );
        }
    }

    /// The delete dialog: the open image's classes and count; another
    /// scope recounts, keeping what was unticked; Delete deletes those of
    /// the ticked classes there.
    #[test]
    fn deleting_takes_the_ticked_classes_of_the_scope() {
        let mut project = project_with(
            vec![
                object(1, &[NUCLEI], 0),
                object(2, &[NUCLEI], 3),
                object(3, &[TUMOR], 6),
            ],
            vec![],
        );
        let mut other = project
            .settings
            .images
            .list
            .get(Path::new("img.tif"))
            .unwrap()
            .clone();
        other.rel_path = PathBuf::from("b.tif");
        other.series.get_mut(&0).unwrap().objects =
            vec![object(4, &[NUCLEI], 0), object(5, &[], 3)];
        project
            .settings
            .images
            .list
            .insert(PathBuf::from("b.tif"), other);
        let (ui, ui_state, _c) = controller(project);
        let state = ui.global::<AnnotationState>();

        state.invoke_open_delete_dialog();
        assert_eq!(
            rows(state.get_manual_classes()),
            vec![(5, "Nuclei".into(), 2), (6, "Tumor".into(), 1)]
        );
        assert_eq!(state.get_delete_count(), 3);
        assert_eq!(state.get_delete_summary(), "3 annotations on this image.");
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::DeleteManualAnnotations
        );

        untick(&state.get_manual_classes(), &TUMOR);
        state.invoke_delete_selection_changed(false);
        assert_eq!(state.get_delete_summary(), "2 annotations on this image.");

        state.invoke_delete_selection_changed(true);
        assert_eq!(
            rows(state.get_manual_classes()),
            vec![
                (5, "Nuclei".into(), 3),
                (-1, "Unclassified".into(), 1),
                (6, "Tumor".into(), 1)
            ]
        );
        let ticks: Vec<bool> = state
            .get_manual_classes()
            .iter()
            .map(|r| r.checked)
            .collect();
        assert_eq!(ticks, vec![true, true, false], "Tumor stays unticked");
        assert_eq!(state.get_delete_count(), 4);
        assert_eq!(state.get_delete_summary(), "4 annotations on 2 images.");

        ui_state
            .get_project_runtime_write()
            .set_selected_object(Some(ObjectId(3)));
        state.invoke_delete_manual(true);
        let project = ui_state.get_project();
        let left: Vec<Vec<ObjectId>> = project
            .settings
            .images
            .list
            .values()
            .map(|i| i.series[&0].objects.iter().map(|o| o.id.clone()).collect())
            .collect();
        assert_eq!(left, vec![vec![ObjectId(3)], vec![]]);
        assert_eq!(project.get_selected_object_id(), None);
        drop(project);
        assert!(ui_state.is_dirty());
    }

    /// Nothing ticked: zero to delete, and Delete deletes nothing.
    #[test]
    fn deleting_with_nothing_ticked_deletes_nothing() {
        let (ui, ui_state, _c) = controller(project_with(vec![object(1, &[NUCLEI], 0)], vec![]));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_delete_dialog();
        untick(&state.get_manual_classes(), &NUCLEI);
        state.invoke_delete_selection_changed(false);
        assert_eq!(state.get_delete_count(), 0);
        assert_eq!(state.get_delete_summary(), "0 annotations on this image.");
        state.invoke_delete_manual(false);
        assert_eq!(manual_objects(&ui_state).len(), 1);
        assert!(!ui_state.is_dirty());
    }

    /// Export and import need an open image: a message says so.
    #[test]
    fn geojson_without_an_open_image_says_to_open_one() {
        let mut project = project_with(vec![object(1, &[NUCLEI], 0)], vec![]);
        project.rest_current_image_path();
        let (ui, _ui_state, _c) = controller(project);
        let state = ui.global::<AnnotationState>();
        for action in 0..3 {
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::None);
            match action {
                0 => state.invoke_open_export_dialog(),
                1 => state.invoke_export_geojson(),
                _ => state.invoke_pick_import_file(),
            }
            drain_ui_queue();
            let (title, text, info) = message(&ui).expect("a message");
            assert_eq!(title, "No image open");
            assert!(text.starts_with("Open the image"));
            assert!(info);
            assert!(!ui.global::<FileBrowserState>().get_visible());
        }
    }

    /// The export dialog: the image's name and the classes of its
    /// annotations and pipeline objects, by class id, unclassified last.
    #[test]
    fn the_export_dialog_lists_the_images_classes() {
        let (ui, _ui_state, _c) = controller(project_with(
            vec![object(1, &[TUMOR], 0), object(2, &[], 3)],
            vec![object(3, &[NUCLEI], 6), object(4, &[TUMOR], 9)],
        ));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_export_dialog();
        assert_eq!(state.get_geojson_image(), "img.tif");
        assert_eq!(
            rows(state.get_export_classes()),
            vec![
                (5, "Nuclei".into(), 1),
                (6, "Tumor".into(), 2),
                (-1, "Unclassified".into(), 1)
            ]
        );
        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::ExportAnnotations
        );
    }

    /// Nothing ticked: no file dialog, a message.
    #[test]
    fn exporting_with_nothing_ticked_asks_for_classes() {
        let (ui, _ui_state, _c) = controller(project_with(vec![object(1, &[NUCLEI], 0)], vec![]));
        let state = ui.global::<AnnotationState>();
        state.invoke_open_export_dialog();
        untick(&state.get_export_classes(), &NUCLEI);
        state.invoke_export_geojson();
        drain_ui_queue();
        assert!(!ui.global::<FileBrowserState>().get_visible());
        assert_eq!(
            message(&ui),
            Some((
                "Nothing exported".into(),
                "Tick the classes to export.".into(),
                true
            ))
        );
    }

    /// The save dialog starts in the project's folder, named after the
    /// image; the ticked classes' annotations and pipeline objects are
    /// written, a message says how many and where.
    #[test]
    fn exporting_writes_the_ticked_classes_to_the_chosen_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project_with(
            vec![object(1, &[NUCLEI], 0), object(2, &[TUMOR], 3)],
            vec![object(3, &[], 6)],
        );
        project.tmp_settings.current_project = Some(dir.path().join("p.evaproj"));
        let (ui, _ui_state, _c) = controller(project);
        let state = ui.global::<AnnotationState>();
        state.invoke_open_export_dialog();
        untick(&state.get_export_classes(), &TUMOR);
        state.invoke_export_geojson();
        drain_ui_queue();
        let browser = ui.global::<FileBrowserState>();
        assert!(browser.get_visible());
        assert_eq!(browser.get_file_name(), "img.geojson");
        assert_eq!(
            PathBuf::from(browser.get_current_path().as_str()),
            dir.path()
        );

        // Typed without the extension.
        choose_file(&ui, &dir.path().join("out"));
        let file = dir.path().join("out.geojson");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).expect("written")).unwrap();
        let types: Vec<&str> = json["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["properties"]["objectType"].as_str().unwrap())
            .collect();
        assert_eq!(types, vec!["annotation", "detection"]);
        assert_eq!(
            message(&ui),
            Some((
                "Annotations exported".into(),
                format!("2 objects written to {}.", file.display()),
                true
            ))
        );
    }

    /// An open image no longer in the project (removed meanwhile): the
    /// export fails with a warning that says so.
    #[test]
    fn exporting_an_image_gone_from_the_project_warns() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project_with(vec![], vec![]);
        project.set_current_image_path(&PathBuf::from("gone.tif"));
        let (ui, _ui_state, _c) = controller(project);
        let state = ui.global::<AnnotationState>();
        state.set_export_classes(ModelRc::new(VecModel::from(vec![AnnotationClassRow {
            id: -1,
            checked: true,
            ..Default::default()
        }])));
        state.invoke_export_geojson();
        choose_file(&ui, &dir.path().join("gone.geojson"));
        assert_eq!(
            message(&ui),
            Some((
                "Export failed".into(),
                "gone.tif is not an image of the project".into(),
                false
            ))
        );
        assert!(!dir.path().join("gone.geojson").exists());
    }

    /// Writes a QuPath-like file: a 4x3 shape per class of `classes`
    /// (`None`: none), side by side.
    fn geojson_file(dir: &Path, classes: &[Option<&str>]) -> PathBuf {
        let shape = |(i, class): (usize, &Option<&str>)| {
            let x = 1 + 4 * i as i32;
            let mut properties = serde_json::json!({ "objectType": "annotation" });
            if let Some(name) = class {
                properties["classification"] = serde_json::json!({ "name": name });
            }
            serde_json::json!({
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": [[[x, 1], [x + 3, 1], [x + 3, 3], [x, 3], [x, 1]]] },
                "properties": properties
            })
        };
        let file = dir.join("qupath.geojson");
        let features: Vec<_> = classes.iter().enumerate().map(shape).collect();
        std::fs::write(&file, serde_json::json!(features).to_string()).unwrap();
        file
    }

    /// Picking a file shows the import dialog: the file's classes with
    /// their default targets, the targets; applying adds the shapes with
    /// the chosen mapping and says what happened.
    #[test]
    fn importing_maps_the_files_classes_and_adds_the_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let file = geojson_file(
            dir.path(),
            &[Some("Nuclei"), Some("Nuclei"), Some("Stroma"), None],
        );
        let (ui, ui_state, _c) = controller(project_with(vec![object(9, &[TUMOR], 15)], vec![]));
        let state = ui.global::<AnnotationState>();
        state.invoke_pick_import_file();
        drain_ui_queue();
        choose_file(&ui, &file);

        assert_eq!(
            ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::ImportAnnotations
        );
        assert_eq!(state.get_geojson_image(), "img.tif");
        assert_eq!(state.get_import_file(), "qupath.geojson");
        assert_eq!(state.get_import_shapes(), 4);
        let targets: Vec<String> = state
            .get_import_targets()
            .iter()
            .map(|t| t.to_string())
            .collect();
        assert_eq!(
            targets,
            vec!["New class", "No class", "Don't import", "Nuclei", "Tumor"]
        );
        assert_eq!(state.get_import_target_colors().row_count(), 5);
        let classes = state.get_import_classes();
        let mut found: Vec<(String, i32, i32)> = classes
            .iter()
            .map(|r| (r.name.to_string(), r.count, r.target))
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec![
                (String::new(), 1, TARGET_NONE),
                ("Nuclei".into(), 2, 3),
                ("Stroma".into(), 1, TARGET_NEW)
            ]
        );

        // Stroma's shapes not imported, the unclassified one to Tumor.
        for i in 0..classes.row_count() {
            let mut row = classes.row_data(i).unwrap();
            row.target = match row.name.as_str() {
                "Stroma" => TARGET_SKIP,
                "" => 4,
                _ => row.target,
            };
            classes.set_row_data(i, row);
        }
        state.invoke_apply_import(false);
        drain_ui_queue();

        let objects = manual_objects(&ui_state);
        let mut classes: Vec<_> = objects.iter().map(|o| o.object_class.clone()).collect();
        classes.sort();
        assert_eq!(
            classes,
            vec![
                [NUCLEI].into(),
                [NUCLEI].into(),
                [TUMOR].into(),
                [TUMOR].into()
            ],
            "the one there kept, 3 added"
        );
        assert!(ui_state.is_dirty());
        let (title, text, info) = message(&ui).expect("a message");
        assert_eq!(title, "Annotations imported");
        assert!(info);
        assert!(
            text.starts_with("3 annotations added to the image."),
            "{text}"
        );
        assert!(text.contains("1 shape not imported"), "{text}");
        assert!(text.contains("Not measured"), "no image file: {text}");

        // Applied once: a second Apply has nothing to apply.
        state.invoke_apply_import(false);
        drain_ui_queue();
        assert_eq!(manual_objects(&ui_state).len(), 4);
    }

    /// Replacing: the image's manual annotations give way to the file's;
    /// an unknown class becomes a new project class.
    #[test]
    fn importing_with_replace_replaces_the_annotations_and_makes_new_classes() {
        let dir = tempfile::tempdir().unwrap();
        let file = geojson_file(dir.path(), &[Some("Stroma")]);
        let (ui, ui_state, _c) = controller(project_with(vec![object(9, &[TUMOR], 15)], vec![]));
        let state = ui.global::<AnnotationState>();
        state.invoke_pick_import_file();
        drain_ui_queue();
        choose_file(&ui, &file);
        state.invoke_apply_import(true);
        drain_ui_queue();

        let project = ui_state.get_project();
        let stroma = project
            .settings
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Stroma")
            .expect("a new class")
            .global();
        let objects = project.get_objects().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].object_class, [stroma].into());
        drop(project);
        let (_, text, _) = message(&ui).expect("a message");
        assert!(text.contains("New classes: Stroma."), "{text}");
    }

    /// A file that isn't GeoJSON: a warning, no import dialog, nothing to
    /// apply.
    #[test]
    fn importing_a_file_that_is_not_geojson_warns() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("broken.geojson");
        std::fs::write(&file, "not json").unwrap();
        let (ui, ui_state, _c) = controller(project_with(vec![object(9, &[TUMOR], 15)], vec![]));
        let state = ui.global::<AnnotationState>();
        state.invoke_pick_import_file();
        drain_ui_queue();
        choose_file(&ui, &file);
        let (title, text, info) = message(&ui).expect("a message");
        assert_eq!(title, "Import failed");
        assert!(text.contains("not GeoJSON"), "{text}");
        assert!(!info);
        state.invoke_apply_import(true);
        drain_ui_queue();
        assert_eq!(manual_objects(&ui_state).len(), 1, "nothing replaced");
    }

    /// The fixture image's project, with `objects` as manual annotations.
    fn fixture_project_with(objects: Vec<ObjectMetricSettings>) -> ProjectWithRuntime {
        let mut project = project_with_fixture_image();
        project.get_selected_image_series_mut().unwrap().objects = objects;
        project
    }

    /// Measuring in the background fills in the intensities of the
    /// annotations asked for, and only theirs.
    #[test]
    fn annotations_are_measured_in_the_background() {
        let (_ui, ui_state, c) = controller(fixture_project_with(vec![
            object(1, &[], 0),
            object(2, &[], 3),
        ]));
        c.measure_in_background(vec![ObjectId(1)]);
        drain_ui_queue();
        let objects = manual_objects(&ui_state);
        assert!(!objects[0].intensities.is_empty(), "measured");
        assert!(objects[1].intensities.is_empty(), "not asked for");
        assert!(!ui_state.is_dirty(), "measured values are no edit");
    }

    /// No ids, or no open image: nothing measured.
    #[test]
    fn measuring_nothing_or_without_an_open_image_does_nothing() {
        let (_ui, ui_state, c) = controller(fixture_project_with(vec![object(1, &[], 0)]));
        c.measure_in_background(vec![]);
        drain_ui_queue();
        ui_state
            .get_project_runtime_write()
            .rest_current_image_path();
        c.measure_in_background(vec![ObjectId(1)]);
        drain_ui_queue();
        assert!(
            manual_objects_of_fixture(&ui_state)[0]
                .intensities
                .is_empty()
        );
    }

    /// The fixture image's annotations, open or not.
    fn manual_objects_of_fixture(ui_state: &UiState) -> Vec<ObjectMetricSettings> {
        let name = PathBuf::from(fixture_image_path().file_name().unwrap());
        ui_state
            .get_project()
            .settings
            .images
            .list
            .get(&name)
            .unwrap()
            .series[&0]
            .objects
            .clone()
    }

    /// Measuring again measures every image's annotations and says how
    /// many; an image whose file can't be read is named in the message.
    #[test]
    fn remeasuring_measures_every_image_and_counts_the_unreadable() {
        let mut project = fixture_project_with(vec![object(1, &[], 0), object(2, &[], 3)]);
        let mut missing = project
            .settings
            .images
            .list
            .values()
            .next()
            .unwrap()
            .clone();
        missing.rel_path = PathBuf::from("missing.tif");
        missing.series.get_mut(&0).unwrap().objects = vec![object(3, &[], 0)];
        project
            .settings
            .images
            .list
            .insert(PathBuf::from("missing.tif"), missing);
        let (ui, ui_state, _c) = controller(project);
        ui.global::<AnnotationState>().invoke_remeasure();
        drain_ui_queue();

        assert!(
            manual_objects_of_fixture(&ui_state)
                .iter()
                .all(|o| !o.intensities.is_empty())
        );
        assert_eq!(
            message(&ui),
            Some((
                "Annotations measured".into(),
                "2 annotations measured on their images.\n\
                 Not measured (image file not readable): 1 image."
                    .into(),
                true
            ))
        );
    }

    /// A project without images (no image folder): nothing to measure, no
    /// message.
    #[test]
    fn remeasuring_a_project_without_images_does_nothing() {
        let mut project = project_with(vec![object(1, &[], 0)], vec![]);
        project.settings.images.root = None;
        let (ui, ui_state, _c) = controller(project);
        ui.global::<AnnotationState>().invoke_remeasure();
        drain_ui_queue();
        assert_eq!(message(&ui), None);
        assert!(manual_objects(&ui_state)[0].intensities.is_empty());
    }

    #[test]
    fn counts_are_spelled_singular_or_plural() {
        assert_eq!(plural(1, "image", "images"), "1 image");
        assert_eq!(plural(0, "image", "images"), "0 images");
        assert_eq!(plural(2, "image", "images"), "2 images");
    }

    /// A target past the project's classes (gone meanwhile): a new class.
    #[test]
    fn an_import_target_past_the_classes_makes_a_new_one() {
        assert_eq!(target_of(5, &[7, 8]), ClassTarget::New);
        assert_eq!(target_of(4, &[7, 8]), ClassTarget::Class(8));
        assert_eq!(target_of(-3, &[7]), ClassTarget::New);
    }

    #[test]
    fn the_import_message_notes_unmeasured_images() {
        let report = ImportReport {
            objects: 1,
            duplicates: 2,
            created_classes: vec![],
            skipped_features: 1,
            skipped_by_class: 0,
            unmeasured_images: vec![PathBuf::from("a.tif")],
        };
        assert_eq!(
            import_message(&report),
            "1 annotation added to the image.\n2 annotations were already there (the same pixels).\n\
             1 feature skipped (points, lines or empty shapes).\n\
             Not measured (the image file can't be read) - their intensities are missing."
        );
    }
}
