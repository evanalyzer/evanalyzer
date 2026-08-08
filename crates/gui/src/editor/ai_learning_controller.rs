use crate::UiState;
use crate::prelude::*;
use crate::{
    AiLearningState, AiTrainingSettingsSlint, AppWindow, ChannelState, ClassSelectionRowSlint,
    DialogType, FeatureRowSlint, GlobalAppState, ObjectMetricRowSlint, TrainingImageRowSlint,
    TrainingObjectRowSlint,
};
use evanalyzer_app::ai_learning::{PixelTrainingParams, TrainingJob};
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_pixel_settings::{
    AiLearningPixelFeatureSettings, PreprocessingSteps,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings, KNNAlgorithmName,
    KNNDistanceMetric, KNNWeightFunction, KnnSettings, MlpActivation, MlpSettings,
    RandomForestSettings, SplitCriterion,
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
    /// The in-flight training job's cancel flag, if any - set right before
    /// spawning the background thread in `train`, read by the Cancel
    /// button's handler. Mirrors `PipelinesController::pipeline_cancel_flag`.
    training_cancel_flag: std::sync::Mutex<Option<Arc<std::sync::atomic::AtomicBool>>>,
}

impl AiLearningController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            training_cancel_flag: std::sync::Mutex::new(None),
        }
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
            manager.sync_class_selection_to_slint();
            if let Some(ui) = manager.ui.upgrade() {
                let state = ui.global::<AiLearningState>();
                // Don't blank a still-live progress banner: the background
                // job keeps running even if the dialog was closed and
                // reopened mid-training (`training_in_progress` only ever
                // flips back to false in `train`'s terminal-outcome handler).
                if !state.get_training_in_progress() {
                    state.set_training_status("".into());
                    state.set_training_status_is_error(false);
                }
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
            .on_toggle_object_excluded(move |row_index| {
                manager.toggle_object_excluded(row_index);
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_browse_existing_model_clicked(move || {
                manager.browse_existing_model();
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_cancel_training_clicked(move || {
                if let Some(flag) = manager.training_cancel_flag.lock().unwrap().as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });

        let manager = self.clone();
        ui.global::<AiLearningState>().on_train_clicked(
            move |settings,
                  feature_rows,
                  object_metrics,
                  training_images,
                  training_objects,
                  class_selection| {
                manager.train(
                    settings,
                    feature_rows,
                    object_metrics,
                    training_images,
                    training_objects,
                    class_selection,
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

    /// Flips `exclude_from_training` on one training object - the label
    /// itself (`object_class`) is left untouched, only whether it
    /// contributes a training sample the next time `train` runs.
    fn toggle_object_excluded(self: &Arc<Self>, row_index: i32) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<AiLearningState>();
        let mut rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        let Some(row) = rows.get_mut(row_index as usize) else {
            return;
        };

        let image_path = PathBuf::from(row.image_path.as_str());
        let object_id = row.object_id;
        let mut project = self.app_state.get_project_write();
        let mut excluded = row.excluded;
        if let Some(entry) = project.images.list.get_mut(&image_path) {
            let series_idx = entry.selected_series;
            if let Some(series) = entry.series.get_mut(&series_idx)
                && let Some(obj) = series
                    .objects
                    .iter_mut()
                    .find(|o| o.id.0 as i32 == object_id)
            {
                obj.exclude_from_training = !obj.exclude_from_training;
                excluded = obj.exclude_from_training;
            }
        }
        drop(project);

        row.excluded = excluded;
        state.set_training_objects(ModelRc::new(VecModel::from(rows)));
    }

    /// Loads a previously trained `.evamodel` and repopulates the dialog from
    /// it - algorithm/hyperparameters, feature selection (feature rows for a
    /// pixel classifier, metric/class checkboxes for an object classifier)
    /// and the model name - so retraining with more labeled data doesn't
    /// require re-entering the whole configuration by hand.
    fn browse_existing_model(self: &Arc<Self>) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "AI Classifier Model",
                &[evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS],
            )
            .pick_file()
        else {
            return;
        };
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        let saved = match evanalyzer_core::load_classifier_from_file(&path) {
            Ok(saved) => saved,
            Err(e) => {
                self.set_training_status(
                    &format!("Could not load '{}': {e}", path.display()),
                    true,
                );
                return;
            }
        };

        let state = ui.global::<AiLearningState>();
        let mut settings = state.get_settings();
        settings.loaded_model_path = path.to_string_lossy().to_string().into();
        settings.model_name = if saved.settings.metadata.name.trim().is_empty() {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
                .into()
        } else {
            saved.settings.metadata.name.clone().into()
        };
        apply_loaded_backend_settings(&mut settings, &saved.settings.backend);

        match &saved.settings.classifier {
            AiLearningClassifierSettings::Pixel { feature_spec, .. } => {
                settings.mode = 0;
                let rows: Vec<FeatureRowSlint> = feature_spec
                    .channels
                    .iter()
                    .map(|steps| preprocessing_steps_to_feature_row(steps))
                    .collect();
                state.set_feature_rows(ModelRc::new(VecModel::from(rows)));
            }
            AiLearningClassifierSettings::Object {
                feature_spec,
                class_labels,
            } => {
                settings.mode = 1;

                let mut metric_rows: Vec<ObjectMetricRowSlint> =
                    state.get_object_metrics().iter().collect();
                for row in &mut metric_rows {
                    row.selected = feature_spec
                        .metrics
                        .contains(&object_metric_row_to_metric(row));
                }
                state.set_object_metrics(ModelRc::new(VecModel::from(metric_rows)));

                let project = self.app_state.get_project();
                let classes = project.classification.classes().clone();
                drop(project);
                let mut class_rows: Vec<ClassSelectionRowSlint> =
                    state.get_class_selection().iter().collect();
                for row in &mut class_rows {
                    let class_id = classes.get(row.class_index as usize).map(|c| c.id);
                    row.selected =
                        class_id.is_some_and(|id| class_labels.iter().any(|l| l.class == id));
                }
                state.set_class_selection(ModelRc::new(VecModel::from(class_rows)));
            }
        }

        state.set_settings(settings);
        self.set_training_status(
            &format!("Loaded settings from '{}'.", path.display()),
            false,
        );
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
        class_selection: ModelRc<ClassSelectionRowSlint>,
    ) {
        let model_name = settings.model_name.trim().to_string();
        if model_name.is_empty() {
            self.set_training_status("Enter a model name before training.", true);
            return;
        }

        let project = self.app_state.get_project();
        let Some(project_path) = project.tmp_settings.current_project.clone() else {
            drop(project);
            self.set_training_status("Save the project before training a classifier.", true);
            return;
        };
        let project_dir = project_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let project_settings = project.settings.clone();
        drop(project);

        let selected_classes: std::collections::HashSet<ObjectClass> = class_selection
            .iter()
            .filter(|c| c.selected)
            .filter_map(|c| {
                project_settings
                    .classification
                    .classes()
                    .get(c.class_index as usize)
                    .map(|class| class.id)
            })
            .collect();

        let ai_settings = build_ai_learning_settings(
            &settings,
            &feature_rows.iter().collect::<Vec<_>>(),
            &object_metrics.iter().collect::<Vec<_>>(),
            &selected_classes,
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
                self.set_training_status(&e.to_string(), true);
                return;
            }
        };

        let has_training_data = match &job {
            TrainingJob::Pixel(j) => !j.images.is_empty(),
            TrainingJob::Object(j) => !j.objects.is_empty(),
        };
        if !has_training_data {
            self.set_training_status(
                "No labeled training data found - assign a class to at least one object before training.",
                true,
            );
            return;
        }

        // Every pre-flight check passed - hand off to the background worker.
        // The dialog is never closed here (or on completion below) so the
        // user can watch it finish and immediately retrain if they want to.
        info!("Starting classifier training ('{model_name}')");
        self.set_training_status("Training started...", false);
        self.set_training_in_progress(true);
        let manager = self.clone();
        std::thread::spawn(move || {
            let (handle, rx, cancel) = job.run_async();
            *manager.training_cancel_flag.lock().unwrap() = Some(cancel);

            // Captured here (rather than re-derived after `handle.join()`,
            // which only returns the `SavedClassifier`/error, not the
            // backend stats) so the final banner below can report it - see
            // `describe_training_progress`'s doc comment for why `Finished`
            // doesn't update the live banner itself.
            let mut stats: Option<evanalyzer_core::TrainingStats> = None;
            for event in rx {
                if let TrainingProgressEvent::Finished { stats: s } = event {
                    stats = Some(s);
                    continue;
                }
                if let Some(line) = describe_training_progress(&event) {
                    manager.set_training_status(&line, false);
                }
            }

            let result = match handle.join() {
                Ok(result) => result,
                Err(panic_payload) => {
                    let msg = crate::helper::worker_supervisor::panic_message(&panic_payload);
                    Err(evanalyzer_cfg::core_types::InternalErrors::Internal(
                        format!("Training worker crashed: {msg}"),
                    ))
                }
            };

            let (message, is_error) = match result {
                Ok(classifier) => {
                    let stats_line = stats
                        .as_ref()
                        .map(format_training_stats)
                        .unwrap_or_default();
                    match evanalyzer_app::ai_learning::save_trained_model(
                        &classifier,
                        &project_dir,
                        &model_name,
                    ) {
                        Ok(path) => {
                            info!("Classifier training completed: {}", path.display());
                            (
                                format!(
                                    "Training complete. {stats_line} Model saved to {}",
                                    path.display()
                                ),
                                false,
                            )
                        }
                        Err(e) => {
                            warn!("Failed to save trained classifier: {e}");
                            (
                                format!(
                                    "Training completed ({stats_line}), but saving failed: {e}"
                                ),
                                true,
                            )
                        }
                    }
                }
                Err(evanalyzer_cfg::core_types::InternalErrors::Cancelled) => {
                    info!("Classifier training cancelled by user");
                    ("Training cancelled.".to_string(), false)
                }
                Err(e) => {
                    warn!("Classifier training failed: {e}");
                    (format!("Training failed: {e}"), true)
                }
            };
            manager.set_training_status(&message, is_error);
            manager.set_training_in_progress(false);
        });
    }

    // -- Notifications ---------------------------------------------------

    /// Shows a pre-flight validation error, training-started notice, or
    /// training result inline in the dialog itself (`training_status`)
    /// instead of the app-wide warning dialog - swapping to
    /// `DialogType::Warning` would replace this dialog outright (only one
    /// dialog can be active at a time), hiding it and losing the user's
    /// in-progress settings. The dialog is never closed automatically,
    /// including when training finishes, so the user can see the result and
    /// retrain immediately. Callable from any thread (used from both the
    /// synchronous pre-flight checks and the background training thread).
    fn set_training_status(self: &Arc<Self>, message: &str, is_error: bool) {
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<AiLearningState>();
                state.set_training_status(message.into());
                state.set_training_status_is_error(is_error);
            }
        }) {
            warn!("Failed to update training status: {e}");
        }
    }

    /// Toggles the Train/Cancel button pair and the banner's in-progress
    /// color (`AiLearningState.training_in_progress`) - set to `true` right
    /// before spawning the background job in `train`, and back to `false` on
    /// every terminal outcome (success, cancelled, or error) so a failure
    /// never leaves the Train button stuck disabled.
    fn set_training_in_progress(self: &Arc<Self>, in_progress: bool) {
        let ui_weak = self.ui.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<AiLearningState>()
                    .set_training_in_progress(in_progress);
            }
        }) {
            warn!("Failed to update training_in_progress: {e}");
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

    /// Class checklist for the "Classes to Train" section, shared by both
    /// pixel- and object-classifier training - one row per project class,
    /// pre-checked for classes that already have at least one labeled object
    /// (a convenient starting point, not a restriction: the user can
    /// check/uncheck freely either way).
    pub fn sync_class_selection_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let project = app_state.get_project();
            let used = evanalyzer_app::ai_learning::used_object_classes(&project.settings);
            let rows: Vec<ClassSelectionRowSlint> = project
                .classification
                .classes()
                .iter()
                .enumerate()
                .map(|(i, c)| ClassSelectionRowSlint {
                    class_index: i as i32,
                    name: c.name.clone().into(),
                    selected: used.contains(&c.id),
                })
                .collect();
            drop(project);
            ui.global::<AiLearningState>()
                .set_class_selection(ModelRc::new(VecModel::from(rows)));
        }) {
            warn!(
                "Failed to sync AI classifier class selection to Slint: {}",
                e
            );
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
            // `images.list` is keyed by path *relative* to `images.root` -
            // `get_current_image_path_cloned` returns the *absolute* path
            // (see its doc comment vs. `get_current_rel_image_path_cloned`'s),
            // so comparing against it directly here would never match,
            // letting the current image sneak into "other images" too.
            let current_path = project.get_current_rel_image_path_cloned();

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
            // Must be the *relative* path - it becomes `TrainingObjectRowSlint::image_path`,
            // which `assign_object_class`/`toggle_object_excluded` look up
            // directly in `images.list` (keyed by relative path). Using the
            // absolute path here (`get_current_image_path_cloned`) made that
            // lookup silently fail for every object on the current image -
            // clicking either control appeared to do nothing.
            let current_path = project.get_current_rel_image_path_cloned();

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
            excluded: obj.exclude_from_training,
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
    selected_classes: &std::collections::HashSet<ObjectClass>,
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
            distance: match settings.knn_distance {
                1 => KNNDistanceMetric::Manhattan,
                2 => KNNDistanceMetric::Minkowski {
                    p: settings.knn_minkowski_p.max(1) as u16,
                },
                3 => KNNDistanceMetric::Cosine,
                4 => KNNDistanceMetric::Hamming,
                _ => KNNDistanceMetric::Euclidean,
            },
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
            // A cleared/invalid (<= 0) field falls back to burn's own
            // AdamConfig default rather than reintroducing the division by
            // zero `MlpSettings::default()`'s own doc comment warns about.
            epsilon: (settings.mlp_epsilon > 0.0)
                .then_some(settings.mlp_epsilon as f64)
                .unwrap_or(1e-5),
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
            class_labels: evanalyzer_app::ai_learning::pixel_class_labels_from_project(
                project,
                selected_classes,
            ),
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
            class_labels: evanalyzer_app::ai_learning::object_class_labels_from_project(
                project,
                selected_classes,
            ),
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

fn hessian_mode_to_row(mode: &FiltersHessianHessianModeSettings) -> i32 {
    match mode {
        FiltersHessianHessianModeSettings::EigenvaluesX => 1,
        FiltersHessianHessianModeSettings::EigenvaluesY => 2,
        FiltersHessianHessianModeSettings::Determinant => 0,
    }
}

/// The inverse of [`feature_row_to_preprocessing_steps`], used to repopulate
/// the dialog's feature-row editor from a loaded model's `feature_spec` (see
/// `browse_existing_model`). Any chain shape that dialog itself would never
/// produce (e.g. a model saved outside this GUI) falls back to Raw rather
/// than guessing at intent.
fn preprocessing_steps_to_feature_row(steps: &[PreprocessingSteps]) -> FeatureRowSlint {
    match steps {
        [] => FeatureRowSlint {
            filter_type: 0,
            ..default_feature_row()
        },
        [PreprocessingSteps::GaussianBlur(s)] => FeatureRowSlint {
            filter_type: 1,
            kernel_size: s.kernel_size as i32,
            sigma: s.sigma,
            ..default_feature_row()
        },
        [PreprocessingSteps::EdgeDetectionSobel(s)] => FeatureRowSlint {
            filter_type: 2,
            kernel_size: s.kernel_size as i32,
            ..default_feature_row()
        },
        [PreprocessingSteps::Laplacian(s)] => FeatureRowSlint {
            filter_type: 3,
            kernel_size: s.kernel_size as i32,
            pre_blur_enabled: false,
            ..default_feature_row()
        },
        [
            PreprocessingSteps::GaussianBlur(blur),
            PreprocessingSteps::Laplacian(s),
        ] => FeatureRowSlint {
            filter_type: 3,
            kernel_size: s.kernel_size as i32,
            pre_blur_enabled: true,
            pre_blur_sigma: blur.sigma,
            ..default_feature_row()
        },
        [PreprocessingSteps::StructureTensor(s)] => FeatureRowSlint {
            filter_type: 4,
            kernel_size: s.kernel_size as i32,
            sigma: s.sigma,
            structure_tensor_mode: match s.mode {
                FiltersStructureTensorTensorModeSettings::EigenvaluesY => 1,
                FiltersStructureTensorTensorModeSettings::Coherence => 2,
                FiltersStructureTensorTensorModeSettings::EigenvaluesX => 0,
            },
            ..default_feature_row()
        },
        [PreprocessingSteps::Hessian(s)] => FeatureRowSlint {
            filter_type: 5,
            pre_blur_enabled: false,
            hessian_mode: hessian_mode_to_row(&s.mode),
            ..default_feature_row()
        },
        [
            PreprocessingSteps::GaussianBlur(blur),
            PreprocessingSteps::Hessian(s),
        ] => FeatureRowSlint {
            filter_type: 5,
            pre_blur_enabled: true,
            pre_blur_sigma: blur.sigma,
            hessian_mode: hessian_mode_to_row(&s.mode),
            ..default_feature_row()
        },
        [PreprocessingSteps::RankFilter(s)] => {
            let (rank_stat, rank_threshold) = match &s.filter_type {
                FiltersRankFilterRankFilterTypeSettings::Median => {
                    (1, default_feature_row().rank_threshold)
                }
                FiltersRankFilterRankFilterTypeSettings::Min => {
                    (2, default_feature_row().rank_threshold)
                }
                FiltersRankFilterRankFilterTypeSettings::Max => {
                    (3, default_feature_row().rank_threshold)
                }
                FiltersRankFilterRankFilterTypeSettings::Outliers(t) => (4, *t),
                FiltersRankFilterRankFilterTypeSettings::Mean => {
                    (0, default_feature_row().rank_threshold)
                }
            };
            FeatureRowSlint {
                filter_type: 6,
                rank_radius: s.radius as f32,
                rank_stat,
                rank_threshold,
                ..default_feature_row()
            }
        }
        _ => FeatureRowSlint {
            filter_type: 0,
            ..default_feature_row()
        },
    }
}

/// Copies a loaded model's backend hyperparameters onto the dialog's
/// settings, including which algorithm/mode is selected. See
/// `browse_existing_model`.
fn apply_loaded_backend_settings(
    settings: &mut AiTrainingSettingsSlint,
    backend: &AiLearningBackendSettings,
) {
    match backend {
        AiLearningBackendSettings::RandomForest(s) => {
            settings.algorithm = 0;
            settings.rf_n_trees = s.n_trees as i32;
            settings.rf_max_depth = s.max_depth.map(|d| d as i32).unwrap_or(0);
            settings.rf_min_samples_split = s.min_samples_split as i32;
            settings.rf_min_samples_leaf = s.min_samples_leaf as i32;
            settings.rf_criterion = match s.criterion {
                SplitCriterion::Entropy => 1,
                SplitCriterion::Gini | SplitCriterion::ClassificationError => 0,
            };
            settings.rf_max_features = s.m.map(|m| m as i32).unwrap_or(0);
            settings.rf_seed = s.seed as i32;
            settings.rf_out_of_bag_eval = s.keep_samples;
        }
        AiLearningBackendSettings::Knn(s) => {
            settings.algorithm = 1;
            settings.knn_k = s.k as i32;
            settings.knn_search_algorithm = match s.algorithm {
                KNNAlgorithmName::CoverTree => 1,
                KNNAlgorithmName::LinearSearch => 0,
            };
            settings.knn_weight = match s.weight {
                KNNWeightFunction::Distance => 1,
                KNNWeightFunction::Uniform => 0,
            };
            settings.knn_distance = match s.distance {
                KNNDistanceMetric::Manhattan => 1,
                KNNDistanceMetric::Minkowski { .. } => 2,
                KNNDistanceMetric::Cosine => 3,
                KNNDistanceMetric::Hamming => 4,
                KNNDistanceMetric::Euclidean => 0,
            };
            // Only meaningful for Minkowski, but always set (rather than left
            // stale from a previous load) so a Euclidean/Manhattan/etc. model
            // doesn't inherit whatever `p` a previously loaded Minkowski model
            // left behind.
            settings.knn_minkowski_p = match s.distance {
                KNNDistanceMetric::Minkowski { p } => p as i32,
                _ => 3,
            };
        }
        AiLearningBackendSettings::Mlp(s) => {
            settings.algorithm = 2;
            settings.mlp_hidden_layers = s
                .hidden_layers
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
                .into();
            settings.mlp_activation = match s.activation {
                MlpActivation::Sigmoid => 1,
                MlpActivation::Tanh => 2,
                MlpActivation::Relu => 0,
            };
            settings.mlp_max_iterations = s.epochs as i32;
            settings.mlp_learning_rate = s.learning_rate as f32;
            settings.mlp_batch_size = s.batch_size as i32;
            settings.mlp_seed = s.seed as i32;
            settings.mlp_epsilon = s.epsilon as f32;
        }
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

/// Logs `event` (same as the old `log_training_progress`) and, for every
/// variant except `Finished` (see `train`'s doc comment on why that one's
/// banner text is built separately, once the save outcome is also known),
/// returns the line to live-update the dialog's `training_status` banner
/// with.
fn describe_training_progress(event: &TrainingProgressEvent) -> Option<String> {
    match event {
        TrainingProgressEvent::Started { total } => {
            info!("Training started: {total} item(s)");
            Some(format!("Training started - {total} item(s) to process..."))
        }
        TrainingProgressEvent::ImageTilesScheduled {
            image_index,
            total_tiles,
        } => {
            info!("Image {image_index}: {total_tiles} tile(s) scheduled");
            Some(format!(
                "Image {}: scanning {total_tiles} tile(s)...",
                image_index + 1
            ))
        }
        TrainingProgressEvent::TileProcessed {
            image_index,
            tile_index,
            total_tiles,
        } => {
            info!("Image {image_index}: tile {tile_index}/{total_tiles} processed");
            Some(format!(
                "Image {}: tile {}/{total_tiles}",
                image_index + 1,
                tile_index + 1
            ))
        }
        TrainingProgressEvent::ItemCompleted { index, total } => {
            info!("Training progress: {index}/{total}");
            Some(format!("Processed {index}/{total} item(s)..."))
        }
        TrainingProgressEvent::ImageFailed { path } => {
            warn!("Training image failed to load: {}", path.display());
            Some(format!("Warning: failed to read {}", path.display()))
        }
        TrainingProgressEvent::ObjectSkipped { index, reason } => {
            warn!("Object {index} skipped: {reason}");
            Some(format!("Warning: object {index} skipped ({reason})"))
        }
        TrainingProgressEvent::Training => {
            info!("Fitting model...");
            Some("Fitting model...".to_string())
        }
        TrainingProgressEvent::Epoch {
            epoch,
            total_epochs,
            train_loss,
            val_loss,
        } => {
            let line = match val_loss {
                Some(v) => format!(
                    "Epoch {}/{total_epochs} - train loss {train_loss:.4}, validation loss {v:.4}",
                    epoch + 1
                ),
                None => format!(
                    "Epoch {}/{total_epochs} - train loss {train_loss:.4}",
                    epoch + 1
                ),
            };
            info!("{line}");
            Some(line)
        }
        TrainingProgressEvent::Finished { stats } => {
            info!("Training finished: {stats:?}");
            None
        }
    }
}

/// Formats `TrainingStats` for the dialog's post-training banner - one line
/// per backend, since each carries different numbers.
fn format_training_stats(stats: &evanalyzer_core::TrainingStats) -> String {
    use evanalyzer_core::TrainingStats;
    match stats {
        TrainingStats::RandomForest { n_trees, n_samples } => {
            format!("{n_trees} tree(s), trained on {n_samples} sample(s).")
        }
        TrainingStats::Knn { k, n_samples } => {
            format!("k={k}, trained on {n_samples} sample(s).")
        }
        TrainingStats::Mlp {
            epochs_run,
            total_epochs,
            final_train_loss,
            final_val_loss,
            best_val_loss,
            best_val_epoch,
        } => {
            let mut line = format!(
                "{epochs_run}/{total_epochs} epoch(s), final train loss {final_train_loss:.4}"
            );
            let (Some(final_val), Some(best_val), Some(best_epoch)) =
                (final_val_loss, best_val_loss, best_val_epoch)
            else {
                line.push('.');
                return line;
            };
            line.push_str(&format!(", validation loss {final_val:.4}"));
            // A validation loss that ended up meaningfully worse than the
            // best epoch seen, well before the end of training, is the
            // classic overfitting signature - flag it rather than making
            // the user compare the numbers themselves. 5%/20% are rough
            // thresholds, not derived from anything - just "clearly worse"
            // and "stopped improving with training to spare".
            let epochs_since_best = epochs_run.saturating_sub(*best_epoch);
            let regressed = *final_val > *best_val * 1.05;
            let early = *epochs_run > 0 && epochs_since_best as f64 > *total_epochs as f64 * 0.2;
            if regressed && early {
                line.push_str(&format!(
                    " - possible overfitting: best validation loss {best_val:.4} was at epoch {}, {epochs_since_best} epoch(s) before the end.",
                    *best_epoch + 1
                ));
            } else {
                line.push('.');
            }
            line
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_hidden_layers -----------------------------------------------

    #[test]
    fn parse_hidden_layers_parses_a_comma_separated_list() {
        assert_eq!(parse_hidden_layers("64, 32, 16"), vec![64, 32, 16]);
    }

    #[test]
    fn parse_hidden_layers_drops_zero_and_unparsable_entries() {
        assert_eq!(parse_hidden_layers("64, 0, x, 32"), vec![64, 32]);
    }

    #[test]
    fn parse_hidden_layers_returns_empty_for_blank_input() {
        assert_eq!(parse_hidden_layers(""), Vec::<usize>::new());
    }

    // -- feature_row_to_preprocessing_steps / preprocessing_steps_to_feature_row --

    fn row(filter_type: i32) -> FeatureRowSlint {
        FeatureRowSlint {
            filter_type,
            ..default_feature_row()
        }
    }

    #[test]
    fn raw_row_round_trips_through_an_empty_step_chain() {
        let steps = feature_row_to_preprocessing_steps(&row(0));
        assert!(steps.is_empty());
        assert_eq!(preprocessing_steps_to_feature_row(&steps).filter_type, 0);
    }

    #[test]
    fn gaussian_blur_row_round_trips_kernel_size_and_sigma() {
        let mut r = row(1);
        r.kernel_size = 7;
        r.sigma = 2.5;

        let steps = feature_row_to_preprocessing_steps(&r);
        assert!(matches!(
            steps.as_slice(),
            [PreprocessingSteps::GaussianBlur(_)]
        ));

        let back = preprocessing_steps_to_feature_row(&steps);
        assert_eq!(back.filter_type, 1);
        assert_eq!(back.kernel_size, 7);
        assert_eq!(back.sigma, 2.5);
    }

    #[test]
    fn sobel_row_round_trips_kernel_size() {
        let mut r = row(2);
        r.kernel_size = 5;

        let steps = feature_row_to_preprocessing_steps(&r);
        let back = preprocessing_steps_to_feature_row(&steps);
        assert_eq!(back.filter_type, 2);
        assert_eq!(back.kernel_size, 5);
    }

    #[test]
    fn laplacian_without_pre_blur_round_trips_as_a_single_step() {
        let mut r = row(3);
        r.kernel_size = 5;
        r.pre_blur_enabled = false;

        let steps = feature_row_to_preprocessing_steps(&r);
        assert!(matches!(
            steps.as_slice(),
            [PreprocessingSteps::Laplacian(_)]
        ));

        let back = preprocessing_steps_to_feature_row(&steps);
        assert_eq!(back.filter_type, 3);
        assert_eq!(back.kernel_size, 5);
        assert!(!back.pre_blur_enabled);
    }

    #[test]
    fn laplacian_of_gaussian_round_trips_both_steps() {
        let mut r = row(3);
        r.kernel_size = 5;
        r.pre_blur_enabled = true;
        r.pre_blur_sigma = 1.5;

        let steps = feature_row_to_preprocessing_steps(&r);
        assert!(matches!(
            steps.as_slice(),
            [
                PreprocessingSteps::GaussianBlur(_),
                PreprocessingSteps::Laplacian(_)
            ]
        ));

        let back = preprocessing_steps_to_feature_row(&steps);
        assert_eq!(back.filter_type, 3);
        assert_eq!(back.kernel_size, 5);
        assert!(back.pre_blur_enabled);
        assert_eq!(back.pre_blur_sigma, 1.5);
    }

    #[test]
    fn structure_tensor_row_round_trips_mode_kernel_and_sigma() {
        for mode in [0, 1, 2] {
            let mut r = row(4);
            r.kernel_size = 9;
            r.sigma = 3.0;
            r.structure_tensor_mode = mode;

            let steps = feature_row_to_preprocessing_steps(&r);
            let back = preprocessing_steps_to_feature_row(&steps);
            assert_eq!(back.filter_type, 4, "mode {mode}");
            assert_eq!(back.kernel_size, 9, "mode {mode}");
            assert_eq!(back.sigma, 3.0, "mode {mode}");
            assert_eq!(back.structure_tensor_mode, mode, "mode {mode}");
        }
    }

    #[test]
    fn hessian_without_pre_blur_round_trips_mode() {
        for mode in [0, 1, 2] {
            let mut r = row(5);
            r.hessian_mode = mode;
            r.pre_blur_enabled = false;

            let steps = feature_row_to_preprocessing_steps(&r);
            assert!(matches!(steps.as_slice(), [PreprocessingSteps::Hessian(_)]));

            let back = preprocessing_steps_to_feature_row(&steps);
            assert_eq!(back.filter_type, 5, "mode {mode}");
            assert_eq!(back.hessian_mode, mode, "mode {mode}");
            assert!(!back.pre_blur_enabled, "mode {mode}");
        }
    }

    #[test]
    fn hessian_of_gaussian_round_trips_mode_and_pre_blur_sigma() {
        let mut r = row(5);
        r.hessian_mode = 2;
        r.pre_blur_enabled = true;
        r.pre_blur_sigma = 0.8;
        r.kernel_size = 5; // used for the pre-blur step, not Hessian itself

        let steps = feature_row_to_preprocessing_steps(&r);
        assert!(matches!(
            steps.as_slice(),
            [
                PreprocessingSteps::GaussianBlur(_),
                PreprocessingSteps::Hessian(_)
            ]
        ));

        let back = preprocessing_steps_to_feature_row(&steps);
        assert_eq!(back.filter_type, 5);
        assert_eq!(back.hessian_mode, 2);
        assert!(back.pre_blur_enabled);
        assert_eq!(back.pre_blur_sigma, 0.8);
    }

    #[test]
    fn rank_filter_round_trips_every_stat_including_outliers_threshold() {
        for (stat, threshold) in [(0, 50.0), (1, 50.0), (2, 50.0), (3, 50.0), (4, 12.5)] {
            let mut r = row(6);
            r.rank_radius = 3.0;
            r.rank_stat = stat;
            r.rank_threshold = threshold;

            let steps = feature_row_to_preprocessing_steps(&r);
            assert!(matches!(
                steps.as_slice(),
                [PreprocessingSteps::RankFilter(_)]
            ));

            let back = preprocessing_steps_to_feature_row(&steps);
            assert_eq!(back.filter_type, 6, "stat {stat}");
            assert_eq!(back.rank_radius, 3.0, "stat {stat}");
            assert_eq!(back.rank_stat, stat, "stat {stat}");
            if stat == 4 {
                assert_eq!(back.rank_threshold, threshold, "stat {stat}");
            }
        }
    }

    #[test]
    fn preprocessing_steps_to_feature_row_falls_back_to_raw_for_an_unrecognized_shape() {
        // Two Sobel steps in a row is not a shape this dialog would ever
        // produce - must degrade to Raw rather than panicking or guessing.
        let steps = vec![
            PreprocessingSteps::EdgeDetectionSobel(EdgeDetectionSobelSettings { kernel_size: 3 }),
            PreprocessingSteps::EdgeDetectionSobel(EdgeDetectionSobelSettings { kernel_size: 3 }),
        ];
        assert_eq!(preprocessing_steps_to_feature_row(&steps).filter_type, 0);
    }

    // -- object_metric_row_to_metric ----------------------------------------

    fn metric_row(metric_id: i32, channel_index: i32) -> ObjectMetricRowSlint {
        ObjectMetricRowSlint {
            label: "".into(),
            metric_id,
            channel_index,
            selected: false,
        }
    }

    #[test]
    fn object_metric_row_to_metric_maps_non_intensity_ids() {
        assert_eq!(
            object_metric_row_to_metric(&metric_row(0, -1)),
            ObjectMetric::Area
        );
        assert_eq!(
            object_metric_row_to_metric(&metric_row(12, -1)),
            ObjectMetric::Eccentricity
        );
    }

    #[test]
    fn object_metric_row_to_metric_maps_intensity_ids_with_their_channel() {
        assert_eq!(
            object_metric_row_to_metric(&metric_row(INTENSITY_METRIC_BASE + 1, 2)),
            ObjectMetric::IntensityMin(2)
        );
        assert_eq!(
            object_metric_row_to_metric(&metric_row(INTENSITY_METRIC_BASE, 0)),
            ObjectMetric::IntensitySum(0)
        );
    }

    // -- apply_loaded_backend_settings --------------------------------------

    fn default_settings() -> AiTrainingSettingsSlint {
        AiTrainingSettingsSlint::default()
    }

    #[test]
    fn apply_loaded_backend_settings_populates_random_forest_fields() {
        let mut settings = default_settings();
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::RandomForest(RandomForestSettings {
                criterion: SplitCriterion::Entropy,
                max_depth: Some(12),
                min_samples_leaf: 3,
                min_samples_split: 4,
                n_trees: 80,
                m: Some(5),
                keep_samples: false,
                seed: 7,
            }),
        );

        assert_eq!(settings.algorithm, 0);
        assert_eq!(settings.rf_n_trees, 80);
        assert_eq!(settings.rf_max_depth, 12);
        assert_eq!(settings.rf_min_samples_leaf, 3);
        assert_eq!(settings.rf_min_samples_split, 4);
        assert_eq!(settings.rf_criterion, 1);
        assert_eq!(settings.rf_max_features, 5);
        assert_eq!(settings.rf_seed, 7);
        assert!(!settings.rf_out_of_bag_eval);
    }

    #[test]
    fn apply_loaded_backend_settings_maps_an_unlimited_depth_to_zero() {
        let mut settings = default_settings();
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::RandomForest(RandomForestSettings {
                max_depth: None,
                ..RandomForestSettings::default()
            }),
        );
        assert_eq!(
            settings.rf_max_depth, 0,
            "None (unlimited) must round-trip to the dialog's 0-means-unlimited sentinel"
        );
    }

    #[test]
    fn apply_loaded_backend_settings_populates_knn_fields() {
        let mut settings = default_settings();
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::Knn(KnnSettings {
                algorithm: KNNAlgorithmName::CoverTree,
                weight: KNNWeightFunction::Distance,
                k: 9,
                distance: KNNDistanceMetric::Manhattan,
            }),
        );

        assert_eq!(settings.algorithm, 1);
        assert_eq!(settings.knn_k, 9);
        assert_eq!(settings.knn_search_algorithm, 1);
        assert_eq!(settings.knn_weight, 1);
        assert_eq!(settings.knn_distance, 1);
    }

    #[test]
    fn apply_loaded_backend_settings_maps_every_knn_distance_metric_to_its_dropdown_index() {
        let cases = [
            (KNNDistanceMetric::Euclidean, 0),
            (KNNDistanceMetric::Manhattan, 1),
            (KNNDistanceMetric::Minkowski { p: 7 }, 2),
            (KNNDistanceMetric::Cosine, 3),
            (KNNDistanceMetric::Hamming, 4),
        ];
        for (distance, expected_index) in cases {
            let mut settings = default_settings();
            apply_loaded_backend_settings(
                &mut settings,
                &AiLearningBackendSettings::Knn(KnnSettings {
                    distance: distance.clone(),
                    ..Default::default()
                }),
            );
            assert_eq!(
                settings.knn_distance, expected_index,
                "{distance:?} should map to dropdown index {expected_index}"
            );
        }
    }

    #[test]
    fn apply_loaded_backend_settings_populates_the_minkowski_order_only_for_minkowski() {
        let mut settings = default_settings();
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::Knn(KnnSettings {
                distance: KNNDistanceMetric::Minkowski { p: 7 },
                ..Default::default()
            }),
        );
        assert_eq!(settings.knn_minkowski_p, 7);

        // Loading a non-Minkowski model afterwards must reset it (not leave
        // the previous model's `p` behind for the now-hidden control).
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::Knn(KnnSettings {
                distance: KNNDistanceMetric::Euclidean,
                ..Default::default()
            }),
        );
        assert_eq!(settings.knn_minkowski_p, 3);
    }

    #[test]
    fn build_ai_learning_settings_maps_every_knn_dropdown_index_to_its_distance_metric() {
        let cases = [
            (0, KNNDistanceMetric::Euclidean),
            (1, KNNDistanceMetric::Manhattan),
            (2, KNNDistanceMetric::Minkowski { p: 7 }),
            (3, KNNDistanceMetric::Cosine),
            (4, KNNDistanceMetric::Hamming),
        ];
        for (dropdown_index, expected_distance) in cases {
            let settings = AiTrainingSettingsSlint {
                algorithm: 1,
                knn_distance: dropdown_index,
                knn_minkowski_p: 7,
                ..default_settings()
            };
            let ai_settings = build_ai_learning_settings(
                &settings,
                &[],
                &[],
                &std::collections::HashSet::new(),
                &ProjectSettings::default(),
            );
            let AiLearningBackendSettings::Knn(knn) = &ai_settings.backend else {
                panic!("expected a Knn backend for dropdown index {dropdown_index}");
            };
            assert_eq!(
                knn.distance, expected_distance,
                "dropdown index {dropdown_index} should map to {expected_distance:?}"
            );
        }
    }

    #[test]
    fn apply_loaded_backend_settings_populates_mlp_fields() {
        let mut settings = default_settings();
        apply_loaded_backend_settings(
            &mut settings,
            &AiLearningBackendSettings::Mlp(MlpSettings {
                hidden_layers: vec![64, 32],
                activation: MlpActivation::Tanh,
                epochs: 150,
                learning_rate: 0.01,
                batch_size: 16,
                seed: 3,
                epsilon: 0.0002,
            }),
        );

        assert_eq!(settings.algorithm, 2);
        assert_eq!(settings.mlp_hidden_layers, "64, 32");
        assert_eq!(settings.mlp_activation, 2);
        assert_eq!(settings.mlp_max_iterations, 150);
        assert_eq!(settings.mlp_learning_rate, 0.01);
        assert_eq!(settings.mlp_batch_size, 16);
        assert_eq!(settings.mlp_seed, 3);
        assert_eq!(settings.mlp_epsilon, 0.0002f32);
    }

    // -- Callback-driven Slint state mutations ------------------------------
    //
    // These handlers mutate `AiLearningState` directly (no
    // `invoke_from_event_loop`, unlike the `sync_*_to_slint` methods above),
    // so their effect is observable synchronously right after invoking the
    // wired callback - see `crate::editor::test_support::test_ui_windows`'s
    // doc comment for why this headless harness doesn't need a real display.

    use crate::editor::test_support::{
        project_with_one_image, test_ui_state_with_project, test_ui_windows,
    };
    use evanalyzer_app::extensions::project_ext::ProjectExt;
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::classification_settings::Class;

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
    ) -> Arc<AiLearningController> {
        Arc::new(AiLearningController::new(ui, app_state))
    }

    #[test]
    fn attach_callbacks_add_feature_row_appends_a_default_row() {
        let (ui, _results_ui) = test_ui_windows();
        let controller =
            make_controller_with_ui(ui.as_weak(), crate::editor::test_support::test_ui_state());
        controller.attach_callbacks();

        ui.global::<AiLearningState>()
            .invoke_add_feature_row_clicked();

        let rows: Vec<FeatureRowSlint> = ui
            .global::<AiLearningState>()
            .get_feature_rows()
            .iter()
            .collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].filter_type, 1); // Gaussian Blur, per default_feature_row
    }

    #[test]
    fn attach_callbacks_remove_feature_row_drops_the_given_index() {
        let (ui, _results_ui) = test_ui_windows();
        let controller =
            make_controller_with_ui(ui.as_weak(), crate::editor::test_support::test_ui_state());
        controller.attach_callbacks();
        let state = ui.global::<AiLearningState>();
        state.set_feature_rows(ModelRc::new(VecModel::from(vec![
            FeatureRowSlint {
                filter_type: 0,
                ..default_feature_row()
            },
            FeatureRowSlint {
                filter_type: 2,
                ..default_feature_row()
            },
        ])));

        state.invoke_remove_feature_row_clicked(0);

        let rows: Vec<FeatureRowSlint> = state.get_feature_rows().iter().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].filter_type, 2);
    }

    #[test]
    fn attach_callbacks_remove_feature_row_out_of_range_is_a_no_op() {
        let (ui, _results_ui) = test_ui_windows();
        let controller =
            make_controller_with_ui(ui.as_weak(), crate::editor::test_support::test_ui_state());
        controller.attach_callbacks();
        let state = ui.global::<AiLearningState>();
        state.set_feature_rows(ModelRc::new(VecModel::from(vec![default_feature_row()])));

        state.invoke_remove_feature_row_clicked(99);

        assert_eq!(state.get_feature_rows().iter().count(), 1);
    }

    #[test]
    fn attach_callbacks_add_gaussian_scales_parses_a_csv_list_and_skips_unparsable_entries() {
        let (ui, _results_ui) = test_ui_windows();
        let controller =
            make_controller_with_ui(ui.as_weak(), crate::editor::test_support::test_ui_state());
        controller.attach_callbacks();

        ui.global::<AiLearningState>()
            .invoke_add_gaussian_scales_clicked("1, x, 2.5, , 4".into());

        let rows: Vec<FeatureRowSlint> = ui
            .global::<AiLearningState>()
            .get_feature_rows()
            .iter()
            .collect();
        let sigmas: Vec<f32> = rows.iter().map(|r| r.sigma).collect();
        assert_eq!(sigmas, vec![1.0, 2.5, 4.0]);
        assert!(
            rows.iter().all(|r| r.filter_type == 1),
            "each scale is a GaussianBlur row"
        );
    }

    #[test]
    fn attach_callbacks_toggle_image_selected_flips_only_the_given_row() {
        let (ui, _results_ui) = test_ui_windows();
        let controller =
            make_controller_with_ui(ui.as_weak(), crate::editor::test_support::test_ui_state());
        controller.attach_callbacks();
        let state = ui.global::<AiLearningState>();
        state.set_training_images(ModelRc::new(VecModel::from(vec![
            TrainingImageRowSlint {
                name: "a".into(),
                path: "a.tif".into(),
                selected: false,
                annotated_object_count: 0,
            },
            TrainingImageRowSlint {
                name: "b".into(),
                path: "b.tif".into(),
                selected: false,
                annotated_object_count: 0,
            },
        ])));

        state.invoke_toggle_image_selected(1);

        let rows: Vec<TrainingImageRowSlint> = state.get_training_images().iter().collect();
        assert!(!rows[0].selected);
        assert!(rows[1].selected);
    }

    #[test]
    fn attach_callbacks_assign_object_class_labels_the_project_object_and_updates_the_row() {
        let (ui, _results_ui) = test_ui_windows();
        let mut project = project_with_one_image();
        project.classification.classes_mut().push(Class {
            id: evanalyzer_cfg::core_types::ObjectClass::Valid(1),
            name: "Cell".into(),
            ..Default::default()
        });
        let object_id = ObjectId(1);
        project.add_object(&ObjectMetricSettings {
            id: object_id.clone(),
            ..Default::default()
        });
        let ui_state = test_ui_state_with_project(project);
        let controller = make_controller_with_ui(ui.as_weak(), ui_state.clone());
        controller.attach_callbacks();
        let state = ui.global::<AiLearningState>();
        state.set_training_objects(ModelRc::new(VecModel::from(vec![TrainingObjectRowSlint {
            object_id: object_id.0 as i32,
            label: "Object #1".into(),
            assigned_class_index: -1,
            image_path: "img.tif".into(),
            excluded: false,
        }])));

        // classes()[0] is the auto-prepended Background class - "Cell" is at index 1.
        state.invoke_assign_object_class(0, 1);

        let project = ui_state.get_project();
        let obj = &project.images.list[std::path::Path::new("img.tif")].series[&0].objects[0];
        assert!(
            obj.object_class
                .contains(&evanalyzer_cfg::core_types::ObjectClass::Valid(1))
        );
        drop(project);

        let rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        assert_eq!(rows[0].assigned_class_index, 1);
    }

    #[test]
    fn attach_callbacks_toggle_object_excluded_flips_the_flag_on_the_project_object() {
        let (ui, _results_ui) = test_ui_windows();
        let mut project = project_with_one_image();
        let object_id = ObjectId(1);
        project.add_object(&ObjectMetricSettings {
            id: object_id.clone(),
            ..Default::default()
        });
        let ui_state = test_ui_state_with_project(project);
        let controller = make_controller_with_ui(ui.as_weak(), ui_state.clone());
        controller.attach_callbacks();
        let state = ui.global::<AiLearningState>();
        state.set_training_objects(ModelRc::new(VecModel::from(vec![TrainingObjectRowSlint {
            object_id: object_id.0 as i32,
            label: "Object #1".into(),
            assigned_class_index: -1,
            image_path: "img.tif".into(),
            excluded: false,
        }])));

        state.invoke_toggle_object_excluded(0);

        let project = ui_state.get_project();
        let obj = &project.images.list[std::path::Path::new("img.tif")].series[&0].objects[0];
        assert!(obj.exclude_from_training);
        drop(project);

        let rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        assert!(rows[0].excluded);

        // Toggling again flips it back.
        state.invoke_toggle_object_excluded(0);
        let project = ui_state.get_project();
        assert!(
            !project.images.list[std::path::Path::new("img.tif")].series[&0].objects[0]
                .exclude_from_training
        );
    }

    #[test]
    fn attach_callbacks_assign_object_class_with_an_unknown_row_index_is_a_no_op() {
        let (ui, _results_ui) = test_ui_windows();
        let ui_state = crate::editor::test_support::test_ui_state();
        let controller = make_controller_with_ui(ui.as_weak(), ui_state);
        controller.attach_callbacks();

        // No panic with an empty training_objects list.
        ui.global::<AiLearningState>()
            .invoke_assign_object_class(0, 0);
        ui.global::<AiLearningState>()
            .invoke_toggle_object_excluded(0);
    }
}
