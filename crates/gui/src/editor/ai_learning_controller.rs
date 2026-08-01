use crate::UiState;
use crate::prelude::*;
use crate::{
    AiLearningState, AiTrainingSettingsSlint, AppWindow, ChannelState, DialogType, FeatureRowSlint,
    GlobalAppState, ObjectMetricRowSlint, TrainingImageRowSlint, TrainingObjectRowSlint,
    WarningState,
};
use evanalyzer_app::ai_learning::{PixelTrainingParams, TrainingJob};
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_pixel_settings::{
    AiLearningPixelFeatureSettings, PreprocessingSteps,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings, KNNAlgorithmName,
    KNNWeightFunction, KnnSettings, MlpActivation, MlpSettings, RandomForestSettings,
    SplitCriterion,
};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    EdgeDetectionSobelSettings, FiltersHessianHessianModeSettings,
    FiltersRankFilterRankFilterTypeSettings, FiltersStructureTensorTensorModeSettings,
    GaussianBlurSettings, HessianSettings, LaplacianSettings, RankFilterSettings,
    StructureTensorSettings,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_core::TrainingProgressEvent;
use log::{info, warn};
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
            manager.sync_channel_names_to_slint();
            manager.sync_object_metrics_to_slint();
            manager.sync_training_images_to_slint();
            manager.sync_training_objects_to_slint();
            if let Some(ui) = manager.ui.upgrade() {
                ui.global::<AiLearningState>().set_training_error("".into());
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
            .classes()
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

    /// Builds an `AiLearningSettings` from the dialog's state, gathers
    /// training data from the project (every object with an assigned class -
    /// see `evanalyzer_app::ai_learning::build_training_job`'s doc comment;
    /// `training_images`/`training_objects` aren't needed here since labels
    /// already live on the project's objects via `assign_object_class`, not
    /// in Slint-only state), then runs training on a background thread and
    /// saves the result under `<project>/models/<settings.model_name>`.
    fn train(
        self: &Arc<Self>,
        settings: AiTrainingSettingsSlint,
        feature_rows: ModelRc<FeatureRowSlint>,
        object_metrics: ModelRc<ObjectMetricRowSlint>,
        _training_images: ModelRc<TrainingImageRowSlint>,
        _training_objects: ModelRc<TrainingObjectRowSlint>,
    ) {
        let model_name = settings.model_name.trim().to_string();
        if model_name.is_empty() {
            self.set_training_error("Enter a model name before training.");
            return;
        }

        let project = self.app_state.get_project();
        let Some(project_path) = project.tmp_settings.current_project.clone() else {
            drop(project);
            self.set_training_error("Save the project before training a classifier.");
            return;
        };
        let project_dir = project_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let project_settings = project.settings.clone();
        drop(project);

        let ai_settings = build_ai_learning_settings(
            &settings,
            &feature_rows.iter().collect::<Vec<_>>(),
            &object_metrics.iter().collect::<Vec<_>>(),
            &project_settings,
        );

        // Z-stack handling isn't a dialog setting - it reuses the project's
        // own global z-stack projection (the same one the viewport's channel
        // controls set), not a separate picker.
        let z_stack_handling = project_settings
            .images
            .settings
            .z_stack
            .as_ref()
            .map(|z| z.z_projection.clone())
            .unwrap_or_default();
        let pixel_params = PixelTrainingParams {
            channel: settings.pixel_channel,
            z_stack_handling,
            ..Default::default()
        };

        let job = match evanalyzer_app::ai_learning::build_training_job(
            &project_settings,
            ai_settings,
            pixel_params,
        ) {
            Ok(job) => job,
            Err(e) => {
                self.set_training_error(&e.to_string());
                return;
            }
        };

        let has_training_data = match &job {
            TrainingJob::Pixel(j) => !j.images.is_empty(),
            TrainingJob::Object(j) => !j.objects.is_empty(),
        };
        if !has_training_data {
            self.set_training_error(
                "No labeled training data found - assign a class to at least one object before training.",
            );
            return;
        }

        // Every pre-flight check passed - close the dialog and hand off to
        // the background worker. Failures from here on (job errors, save
        // errors) happen well after the dialog is gone, so they're reported
        // via the app-wide warning dialog instead of the inline banner.
        info!("Starting classifier training ('{model_name}')");
        self.close_dialog();
        let manager = self.clone();
        std::thread::spawn(move || {
            let (handle, rx, _cancel) = job.run_async();
            for event in rx {
                log_training_progress(&event);
            }

            let result = match handle.join() {
                Ok(result) => result,
                Err(_) => Err(evanalyzer_cfg::core_types::InternalErrors::Internal(
                    "Training worker crashed unexpectedly".to_string(),
                )),
            };

            match result {
                Ok(classifier) => {
                    match evanalyzer_app::ai_learning::save_trained_model(
                        &classifier,
                        &project_dir,
                        &model_name,
                    ) {
                        Ok(path) => {
                            info!("Classifier training completed: {}", path.display());
                            manager.show_info(
                                "Training complete",
                                &format!("Model saved to {}", path.display()),
                            );
                        }
                        Err(e) => {
                            warn!("Failed to save trained classifier: {e}");
                            manager.show_warning(
                                "Training completed, but saving failed",
                                &e.to_string(),
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!("Classifier training failed: {e}");
                    manager.show_warning("Training failed", &e.to_string());
                }
            }
        });
    }

    // -- Notifications ---------------------------------------------------

    /// Shows a pre-flight validation error inline in the dialog itself
    /// (`training_error`) instead of the app-wide warning dialog - swapping
    /// to `DialogType::Warning` would replace this dialog outright, so the
    /// user would lose their in-progress settings just to be told to fix one
    /// of them. Only used for checks that run before the dialog would
    /// otherwise close; see `train`.
    fn set_training_error(self: &Arc<Self>, message: &str) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        ui.global::<AiLearningState>()
            .set_training_error(message.into());
    }

    fn close_dialog(self: &Arc<Self>) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);
    }

    /// Shows the generic warning dialog (error style), mirroring
    /// `ProjectController::show_warning` - there's no dedicated training
    /// progress/result UI yet, so training completion and failures both
    /// surface through this shared dialog.
    fn show_warning(self: &Arc<Self>, title: &str, message: &str) {
        self.show_warning_with_style(title, message, false);
    }

    fn show_info(self: &Arc<Self>, title: &str, message: &str) {
        self.show_warning_with_style(title, message, true);
    }

    fn show_warning_with_style(self: &Arc<Self>, title: &str, message: &str, info: bool) {
        let title = title.to_owned();
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let warning = ui.global::<WarningState>();
                warning.set_info(info);
                warning.set_title(title.into());
                warning.set_message(message.into());
                ui.global::<GlobalAppState>()
                    .set_active_dialog(DialogType::Warning);
            }
        }) {
            warn!("Failed to show warning dialog: {e}");
        }
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
                .classes()
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

    /// Channel names for the pixel-classifier's channel picker - index
    /// matches how `train()` reads back `settings.pixel_channel`, and the
    /// same position-based convention `sync_object_metrics_to_slint` already
    /// uses for its per-channel intensity metric rows.
    pub fn sync_channel_names_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let names: Vec<slint::SharedString> = ui
                .global::<ChannelState>()
                .get_channels()
                .iter()
                .map(|c| c.name.clone())
                .collect();
            ui.global::<AiLearningState>()
                .set_channel_names(ModelRc::new(VecModel::from(names)));
        }) {
            warn!("Failed to sync AI classifier channel names to Slint: {}", e);
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
            let classes = project.classification.classes().clone();
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
        rank_threshold: 50.0,
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

// -- Slint state -> AiLearningSettings ---------------------------------------

/// `object_metrics`' `metric_id` encoding mirrors `OBJECT_METRICS`/
/// `INTENSITY_METRIC_BASE`/`INTENSITY_STATS` above (`sync_object_metrics_to_slint`
/// is what originally assigned these ids).
fn build_ai_learning_settings(
    settings: &AiTrainingSettingsSlint,
    feature_rows: &[FeatureRowSlint],
    object_metrics: &[ObjectMetricRowSlint],
    project: &ProjectSettings,
) -> AiLearningSettings {
    let backend = match settings.algorithm {
        0 => AiLearningBackendSettings::RandomForest(RandomForestSettings {
            criterion: match settings.rf_criterion {
                1 => SplitCriterion::Entropy,
                _ => SplitCriterion::Gini,
            },
            max_depth: (settings.rf_max_depth > 0).then_some(settings.rf_max_depth as u16),
            min_samples_leaf: settings.rf_min_samples_leaf.max(1) as usize,
            min_samples_split: settings.rf_min_samples_split.max(2) as usize,
            n_trees: settings.rf_n_trees.max(1) as u16,
            m: (settings.rf_max_features > 0).then_some(settings.rf_max_features as usize),
            keep_samples: settings.rf_out_of_bag_eval,
            seed: settings.rf_seed.max(0) as u64,
        }),
        1 => AiLearningBackendSettings::Knn(KnnSettings {
            algorithm: match settings.knn_search_algorithm {
                0 => KNNAlgorithmName::LinearSearch,
                _ => KNNAlgorithmName::CoverTree,
            },
            weight: match settings.knn_weight {
                1 => KNNWeightFunction::Distance,
                _ => KNNWeightFunction::Uniform,
            },
            k: settings.knn_k.max(1) as usize,
        }),
        _ => AiLearningBackendSettings::Mlp(MlpSettings {
            hidden_layers: parse_hidden_layers(&settings.mlp_hidden_layers),
            activation: match settings.mlp_activation {
                1 => MlpActivation::Sigmoid,
                2 => MlpActivation::Tanh,
                _ => MlpActivation::Relu,
            },
            epochs: settings.mlp_max_iterations.max(1) as usize,
            learning_rate: settings.mlp_learning_rate as f64,
            batch_size: settings.mlp_batch_size.max(1) as usize,
            seed: settings.mlp_seed.max(0) as u64,
        }),
    };

    let classifier = if settings.mode == 0 {
        AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings {
                channels: feature_rows
                    .iter()
                    .map(feature_row_to_preprocessing_steps)
                    .collect(),
            },
            class_labels: evanalyzer_app::ai_learning::pixel_class_labels_from_project(project),
        }
    } else {
        AiLearningClassifierSettings::Object {
            feature_spec: AiLearningObjectFeatureSettings {
                metrics: object_metrics
                    .iter()
                    .filter(|m| m.selected)
                    .map(object_metric_row_to_metric)
                    .collect(),
            },
            class_labels: evanalyzer_app::ai_learning::object_class_labels_from_project(project),
        }
    };

    AiLearningSettings {
        metadata: MetaData {
            name: settings.model_name.to_string(),
            ..Default::default()
        },
        backend,
        classifier,
    }
}

fn parse_hidden_layers(csv: &str) -> Vec<usize> {
    csv.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect()
}

/// Converts one feature row into its preprocessing chain. `filter_type == 0`
/// (Raw) yields an empty chain - see `AiLearningPixelFeatureSettings`'s doc
/// comment for why that means "unmodified pixel value."
///
/// Laplacian/Hessian-of-Gaussian pre-blur reuses `row.kernel_size` for the
/// blur step too - the dialog has no separate pre-blur kernel-size field.
fn feature_row_to_preprocessing_steps(row: &FeatureRowSlint) -> Vec<PreprocessingSteps> {
    let pre_blur = |steps: &mut Vec<PreprocessingSteps>| {
        if row.pre_blur_enabled {
            steps.push(PreprocessingSteps::GaussianBlur(GaussianBlurSettings {
                kernel_size: row.kernel_size.max(3) as usize,
                sigma: row.pre_blur_sigma,
            }));
        }
    };

    match row.filter_type {
        1 => vec![PreprocessingSteps::GaussianBlur(GaussianBlurSettings {
            kernel_size: row.kernel_size.max(3) as usize,
            sigma: row.sigma,
        })],
        2 => vec![PreprocessingSteps::EdgeDetectionSobel(
            EdgeDetectionSobelSettings {
                kernel_size: row.kernel_size.max(3) as usize,
            },
        )],
        3 => {
            let mut steps = Vec::new();
            pre_blur(&mut steps);
            steps.push(PreprocessingSteps::Laplacian(LaplacianSettings {
                kernel_size: row.kernel_size.max(3) as usize,
            }));
            steps
        }
        4 => vec![PreprocessingSteps::StructureTensor(
            StructureTensorSettings {
                mode: match row.structure_tensor_mode {
                    1 => FiltersStructureTensorTensorModeSettings::EigenvaluesY,
                    2 => FiltersStructureTensorTensorModeSettings::Coherence,
                    _ => FiltersStructureTensorTensorModeSettings::EigenvaluesX,
                },
                kernel_size: row.kernel_size.max(3) as usize,
                sigma: row.sigma,
            },
        )],
        5 => {
            let mut steps = Vec::new();
            pre_blur(&mut steps);
            steps.push(PreprocessingSteps::Hessian(HessianSettings {
                mode: match row.hessian_mode {
                    1 => FiltersHessianHessianModeSettings::EigenvaluesX,
                    2 => FiltersHessianHessianModeSettings::EigenvaluesY,
                    _ => FiltersHessianHessianModeSettings::Determinant,
                },
            }));
            steps
        }
        6 => vec![PreprocessingSteps::RankFilter(RankFilterSettings {
            radius: row.rank_radius as f64,
            filter_type: match row.rank_stat {
                1 => FiltersRankFilterRankFilterTypeSettings::Median,
                2 => FiltersRankFilterRankFilterTypeSettings::Min,
                3 => FiltersRankFilterRankFilterTypeSettings::Max,
                4 => FiltersRankFilterRankFilterTypeSettings::Outliers(row.rank_threshold),
                _ => FiltersRankFilterRankFilterTypeSettings::Mean,
            },
        })],
        _ => Vec::new(), // 0 = Raw, and any unrecognized index
    }
}

/// `row.metric_id` encoding mirrors `OBJECT_METRICS`/`INTENSITY_METRIC_BASE`/
/// `INTENSITY_STATS` above.
fn object_metric_row_to_metric(row: &ObjectMetricRowSlint) -> ObjectMetric {
    if row.metric_id >= INTENSITY_METRIC_BASE {
        let channel = row.channel_index;
        return match row.metric_id - INTENSITY_METRIC_BASE {
            1 => ObjectMetric::IntensityMin(channel),
            2 => ObjectMetric::IntensityMax(channel),
            3 => ObjectMetric::IntensityAvg(channel),
            _ => ObjectMetric::IntensitySum(channel),
        };
    }
    match row.metric_id {
        1 => ObjectMetric::Perimeter,
        2 => ObjectMetric::Circularity,
        3 => ObjectMetric::Solidity,
        4 => ObjectMetric::AspectRatio,
        5 => ObjectMetric::Roundness,
        6 => ObjectMetric::Compactness,
        7 => ObjectMetric::FeretDiameter,
        8 => ObjectMetric::MinFeretDiameter,
        9 => ObjectMetric::EllipseMajor,
        10 => ObjectMetric::EllipseMinor,
        11 => ObjectMetric::EllipseAngle,
        12 => ObjectMetric::Eccentricity,
        13 => ObjectMetric::TouchesEdge,
        _ => ObjectMetric::Area,
    }
}

fn log_training_progress(event: &TrainingProgressEvent) {
    match event {
        TrainingProgressEvent::Started { total } => info!("Training started: {total} item(s)"),
        TrainingProgressEvent::ImageTilesScheduled {
            image_index,
            total_tiles,
        } => info!("Image {image_index}: {total_tiles} tile(s) scheduled"),
        TrainingProgressEvent::TileProcessed {
            image_index,
            tile_index,
            total_tiles,
        } => info!("Image {image_index}: tile {tile_index}/{total_tiles} processed"),
        TrainingProgressEvent::ItemCompleted { index, total } => {
            info!("Training progress: {index}/{total}")
        }
        TrainingProgressEvent::ImageFailed { path } => {
            warn!("Training image failed to load: {}", path.display())
        }
        TrainingProgressEvent::ObjectSkipped { index, reason } => {
            warn!("Object {index} skipped: {reason}")
        }
        TrainingProgressEvent::Training => info!("Fitting model..."),
        TrainingProgressEvent::Finished => info!("Training finished"),
    }
}
