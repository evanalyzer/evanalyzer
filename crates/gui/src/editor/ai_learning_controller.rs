use crate::UiState;
use crate::prelude::*;
use crate::{
    AiLearningState, AiTrainingSettingsSlint, AppWindow, ChannelState, DialogType, FeatureRowSlint,
    GlobalAppState, ObjectMetricRowSlint, TrainingImageRowSlint, TrainingObjectRowSlint,
};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use log::warn;
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Fixed candidate list of object-classifier metrics (see `object.rs`'s
/// measurement methods). `object_class` (the training label itself) and
/// centroid/position are deliberately not offered here - see the dialog's
/// own note about label leakage / location-based overfitting.
const OBJECT_METRICS: &[(&str, i32)] = &[
    ("Area", 0),
    ("Perimeter", 1),
    ("Circularity", 2),
    ("Solidity", 3),
    ("Aspect Ratio", 4),
    ("Roundness", 5),
    ("Compactness", 6),
    ("Feret Diameter", 7),
    ("Min Feret Diameter", 8),
    ("Ellipse Major Axis", 9),
    ("Ellipse Minor Axis", 10),
    ("Ellipse Angle", 11),
    ("Eccentricity", 12),
    ("Touches Edge", 13),
];

const INTENSITY_METRIC_BASE: i32 = 100;
const INTENSITY_STATS: &[(&str, i32)] = &[
    ("Sum Intensity", 0),
    ("Min Intensity", 1),
    ("Max Intensity", 2),
    ("Avg Intensity", 3),
];

pub struct AiLearningController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
}

impl AiLearningController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self { ui, app_state }
    }

    pub fn attach_callbacks(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        // Opened from the toolbar button - refresh everything from the
        // current project before showing the dialog, so it never shows
        // stale data from a previous open.
        let manager = self.clone();
        ui.global::<AiLearningState>().on_open_requested(move || {
            manager.sync_class_names_to_slint();
            manager.sync_object_metrics_to_slint();
            manager.sync_training_images_to_slint();
            manager.sync_training_objects_to_slint();
            if let Some(ui) = manager.ui.upgrade() {
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::AiLearning);
            }
        });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_add_feature_row_clicked(move || {
                manager.add_feature_row(default_feature_row());
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_remove_feature_row_clicked(move |index| {
                manager.remove_feature_row(index);
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_add_gaussian_scales_clicked(move |csv| {
                manager.add_gaussian_scales(&csv);
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_image_selected(move |index| {
                manager.toggle_image_selected(index);
                manager.sync_training_objects_to_slint();
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_assign_object_class(move |row_index, class_index| {
                manager.assign_object_class(row_index, class_index);
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_browse_existing_model_clicked(move || {
                manager.browse_existing_model();
            });

        let manager = self.clone();
        ui.global::<AiLearningState>().on_train_clicked(
            move |settings, feature_rows, object_metrics, training_images, training_objects| {
                manager.train(
                    settings,
                    feature_rows,
                    object_metrics,
                    training_images,
                    training_objects,
                );
            },
        );
    }

    // -- Feature row editing (pixel classifier) --------------------------

    fn add_feature_row(self: &Arc<Self>, row: FeatureRowSlint) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<FeatureRowSlint> = state.get_feature_rows().iter().collect();
        rows.push(row);
        state.set_feature_rows(ModelRc::new(VecModel::from(rows)));
    }

    fn remove_feature_row(self: &Arc<Self>, index: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<FeatureRowSlint> = state.get_feature_rows().iter().collect();
        if index >= 0 && (index as usize) < rows.len() {
            rows.remove(index as usize);
        }
        state.set_feature_rows(ModelRc::new(VecModel::from(rows)));
    }

    fn add_gaussian_scales(self: &Arc<Self>, csv: &str) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<FeatureRowSlint> = state.get_feature_rows().iter().collect();
        for part in csv.split(',') {
            let Ok(sigma) = part.trim().parse::<f32>() else {
                continue;
            };
            rows.push(FeatureRowSlint {
                sigma,
                ..default_feature_row()
            });
        }
        state.set_feature_rows(ModelRc::new(VecModel::from(rows)));
    }

    // -- Training data source / labeling -----------------------------------

    fn toggle_image_selected(self: &Arc<Self>, index: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<TrainingImageRowSlint> = state.get_training_images().iter().collect();
        if let Some(row) = rows.get_mut(index as usize) {
            row.selected = !row.selected;
        }
        state.set_training_images(ModelRc::new(VecModel::from(rows)));
    }

    fn assign_object_class(self: &Arc<Self>, row_index: i32, class_index: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        let Some(row) = rows.get_mut(row_index as usize) else {
            return;
        };

        let mut project = self.app_state.get_project_write();
        let Some(class_id) = project
            .classification
            .classes
            .get(class_index as usize)
            .map(|c| c.id)
        else {
            return;
        };

        let image_path = PathBuf::from(row.image_path.as_str());
        let object_id = row.object_id;
        if let Some(entry) = project.images.list.get_mut(&image_path) {
            let series_idx = entry.selected_series;
            if let Some(series) = entry.series.get_mut(&series_idx) {
                if let Some(obj) = series
                    .objects
                    .iter_mut()
                    .find(|o| o.id.0 as i32 == object_id)
                {
                    // Adds to the object's class set rather than replacing it,
                    // matching how the rest of the app treats `object_class`
                    // (see the Object List panel's add/remove-class UI) - a
                    // wrong label is removed the same way, not overwritten here.
                    obj.object_class.insert(class_id);
                }
            }
        }
        drop(project);

        row.assigned_class_index = class_index;
        state.set_training_objects(ModelRc::new(VecModel::from(rows)));
    }

    fn browse_existing_model(self: &Arc<Self>) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("AI Classifier Model", &["bincode", "model"])
            .pick_file()
        else {
            return;
        };
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut settings = state.get_settings();
        settings.loaded_model_path = path.to_string_lossy().to_string().into();
        if settings.model_name.is_empty() {
            if let Some(stem) = path.file_stem() {
                settings.model_name = stem.to_string_lossy().to_string().into();
            }
        }
        state.set_settings(settings);
    }

    // -- Training entry point ------------------------------------------------

    /// TODO: not implemented yet. Needs to: build a `FeatureSpec`/object
    /// feature matrix from the selected `feature_rows`/`object_metrics`,
    /// gather labeled samples from `training_objects` (skipping rows with
    /// `assigned_class_index == -1`), call into
    /// `evanalyzer_core::ai_learning::pixel::{train_random_forest, train_knn}`
    /// (and the future MLP backend) based on `settings.algorithm`, then
    /// persist the result under `<project>/models/<settings.model_name>`.
    fn train(
        self: &Arc<Self>,
        settings: AiTrainingSettingsSlint,
        _feature_rows: ModelRc<FeatureRowSlint>,
        _object_metrics: ModelRc<ObjectMetricRowSlint>,
        _training_images: ModelRc<TrainingImageRowSlint>,
        _training_objects: ModelRc<TrainingObjectRowSlint>,
    ) {
        warn!(
            "AI classifier training requested (mode={}, algorithm={}, model_name={}) - training is not wired up yet",
            settings.mode, settings.algorithm, settings.model_name
        );
    }

    // -- Sync project state -> Slint -----------------------------------------

    pub fn sync_class_names_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let project = app_state.get_project();
            let names: Vec<slint::SharedString> = project
                .classification
                .classes
                .iter()
                .map(|c| c.name.clone().into())
                .collect();
            drop(project);
            ui.global::<AiLearningState>()
                .set_class_names(ModelRc::new(VecModel::from(names)));
        }) {
            warn!("Failed to sync AI classifier class names to Slint: {}", e);
        }
    }

    pub fn sync_object_metrics_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            let mut rows: Vec<ObjectMetricRowSlint> = OBJECT_METRICS
                .iter()
                .map(|(label, id)| ObjectMetricRowSlint {
                    label: (*label).into(),
                    metric_id: *id,
                    channel_index: -1,
                    selected: false,
                })
                .collect();

            let channels = ui.global::<ChannelState>().get_channels();
            for (ch_idx, ch) in channels.iter().enumerate() {
                for (stat_name, stat_offset) in INTENSITY_STATS {
                    rows.push(ObjectMetricRowSlint {
                        label: format!("{} - {}", ch.name, stat_name).into(),
                        metric_id: INTENSITY_METRIC_BASE + stat_offset,
                        channel_index: ch_idx as i32,
                        selected: false,
                    });
                }
            }

            ui.global::<AiLearningState>()
                .set_object_metrics(ModelRc::new(VecModel::from(rows)));
        }) {
            warn!(
                "Failed to sync AI classifier object metrics to Slint: {}",
                e
            );
        }
    }

    pub fn sync_training_images_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let project = app_state.get_project();
            let current_path = project.get_current_image_path_cloned();

            let rows: Vec<TrainingImageRowSlint> = project
                .images
                .list
                .iter()
                .filter(|(path, _)| Some((*path).clone()) != current_path)
                .map(|(path, entry)| {
                    let annotated = entry
                        .series
                        .get(&entry.selected_series)
                        .map(|s| {
                            s.objects
                                .iter()
                                .filter(|o| !o.object_class.is_empty())
                                .count()
                        })
                        .unwrap_or(0);
                    TrainingImageRowSlint {
                        name: path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default()
                            .into(),
                        path: path.to_string_lossy().to_string().into(),
                        selected: false,
                        annotated_object_count: annotated as i32,
                    }
                })
                .collect();
            drop(project);

            ui.global::<AiLearningState>()
                .set_training_images(ModelRc::new(VecModel::from(rows)));
        }) {
            warn!(
                "Failed to sync AI classifier training images to Slint: {}",
                e
            );
        }
    }

    pub fn sync_training_objects_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<AiLearningState>();
            let selected_other_images: Vec<PathBuf> = state
                .get_training_images()
                .iter()
                .filter(|row| row.selected)
                .map(|row| PathBuf::from(row.path.as_str()))
                .collect();

            let project = app_state.get_project();
            let classes = project.classification.classes.clone();
            let current_path = project.get_current_image_path_cloned();

            let mut rows = Vec::new();
            if let Some(path) = &current_path {
                if let Some(objects) = project.get_objects() {
                    push_object_rows(&mut rows, path, objects, &classes);
                }
            }
            for path in &selected_other_images {
                if let Some(entry) = project.images.list.get(path) {
                    if let Some(series) = entry.series.get(&entry.selected_series) {
                        push_object_rows(&mut rows, path, &series.objects, &classes);
                    }
                }
            }
            drop(project);

            state.set_training_objects(ModelRc::new(VecModel::from(rows)));
        }) {
            warn!(
                "Failed to sync AI classifier training objects to Slint: {}",
                e
            );
        }
    }
}

fn default_feature_row() -> FeatureRowSlint {
    FeatureRowSlint {
        filter_type: 1, // Gaussian Blur
        sigma: 1.0,
        kernel_size: 3,
        structure_tensor_mode: 0,
        hessian_mode: 0,
        rank_stat: 0,
        rank_radius: 2.0,
        pre_blur_enabled: false,
        pre_blur_sigma: 1.0,
    }
}

fn push_object_rows(
    rows: &mut Vec<TrainingObjectRowSlint>,
    path: &Path,
    objects: &[ObjectMetricSettings],
    classes: &[Class],
) {
    for obj in objects {
        let assigned_class_index = obj
            .object_class
            .iter()
            .find_map(|oc| classes.iter().position(|c| &c.id == oc))
            .map(|i| i as i32)
            .unwrap_or(-1);
        rows.push(TrainingObjectRowSlint {
            object_id: obj.id.0 as i32,
            label: format!("Object #{} (area {} px)", obj.id.0, obj.area).into(),
            assigned_class_index,
            image_path: path.to_string_lossy().to_string().into(),
        });
    }
}
