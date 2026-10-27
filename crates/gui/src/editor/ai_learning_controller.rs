//! # ai_learning_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-07-31
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::UiState;
use crate::editor::ai_learning_features::{
    BRIGHTNESS_STATS, ObjectFeatureTable, PIXEL_FEATURES, PixelFeatureTable, Preset, SHAPE_METRICS,
    size_label,
};
use crate::{
    AiLearningState, AiTrainingSettingsSlint, AppWindow, BrightnessRowSlint, ChannelChipSlint,
    ChannelState, ClassSelectionRowSlint, DialogType, FeatureGridRowSlint, GlobalAppState,
    ToggleRowSlint, TrainingImageRowSlint, TrainingObjectRowSlint,
};
use evanalyzer_app::ai_learning::CancelHandle;
use evanalyzer_app::ai_learning::PixelTrainingParams;
use evanalyzer_app::ai_learning::TrainingRequest;
use evanalyzer_app::project::ImageEntryExt;
use evanalyzer_app::project::ImageListExt;
use evanalyzer_app::project::ProjectExt;
use evanalyzer_cfg::core_types::ObjectClass;
use evanalyzer_cfg::core_types::TrainingProgressEvent;
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings, KNNAlgorithmName,
    KNNDistanceMetric, KNNWeightFunction, KnnSettings, MlpActivation, MlpSettings,
    RandomForestSettings, SplitCriterion,
};
use evanalyzer_cfg::settings::classification_settings::Class;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use log::{info, warn};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A class with fewer examples than this gets a hint (it still trains).
const FEW_EXAMPLES: i32 = 5;

/// The feature tables the panel shows, kept here (Slint gets copies).
struct Features {
    pixel: PixelFeatureTable,
    /// (channel name, read by the pixel classifier).
    channels: Vec<(String, bool)>,
    object: ObjectFeatureTable,
    /// The object preset chosen and not edited since: applied again when
    /// the channels change (its brightness columns depend on them), and
    /// what the panel shows as chosen - without channels "Recommended"
    /// and "Shape only" are the same table.
    object_preset: Option<Preset>,
}

pub struct AiLearningController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    /// The in-flight training job's cancel flag, if any - set right before
    /// spawning the background thread in `train`, read by the Cancel
    /// button's handler. Mirrors `PipelinesController::pipeline_cancel_flag`.
    training_cancel_flag: std::sync::Mutex<Option<CancelHandle>>,
    features: std::sync::Mutex<Features>,
    /// The model last trained and saved, with what it learned - for "Add to
    /// pipeline".
    last_model: std::sync::Mutex<Option<(PathBuf, AiLearningClassifierSettings)>>,
}

impl AiLearningController {
    pub fn new(ui: slint::Weak<AppWindow>, app_state: Arc<UiState>) -> Self {
        Self {
            ui,
            app_state,
            training_cancel_flag: std::sync::Mutex::new(None),
            last_model: std::sync::Mutex::new(None),
            features: std::sync::Mutex::new(Features {
                pixel: PixelFeatureTable::preset(Preset::Recommended),
                channels: Vec::new(),
                object: ObjectFeatureTable::preset(Preset::Recommended, 0),
                object_preset: Some(Preset::Recommended),
            }),
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
            manager.sync_all_to_slint(false);
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

        // Objects, classes or channels changed while the dialog is open (an
        // object drawn, a class assigned, the image's channels loaded): its
        // lists follow, the user's ticks stay.
        let manager = self.clone();
        ui.global::<AiLearningState>().on_project_changed(move || {
            let open = manager.ui.upgrade().is_some_and(|ui| {
                ui.global::<GlobalAppState>().get_active_dialog() == DialogType::AiLearning
            });
            if open {
                manager.sync_all_to_slint(true);
            }
        });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_pixel_channel(move |i| {
                manager.edit_features(|f| {
                    if let Some(c) = f.channels.get_mut(i as usize) {
                        c.1 = !c.1;
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_feature_cell(move |row, col| {
                manager.edit_features(|f| {
                    if let Some(r) = f.pixel.rows.get_mut(row as usize) {
                        if col < 0 {
                            r.raw = !r.raw;
                        } else if let Some(cell) = r.cells.get_mut(col as usize) {
                            *cell = !*cell;
                        }
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_set_feature_option(move |row, option| {
                manager.edit_features(|f| {
                    if let Some(r) = f.pixel.rows.get_mut(row as usize) {
                        r.option = option.max(0) as usize;
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_apply_pixel_preset(move |i| {
                if let Some(preset) = preset_of(i) {
                    manager.edit_features(|f| f.pixel = PixelFeatureTable::preset(preset));
                }
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_remove_feature_size(move |col| {
                manager.edit_features(|f| {
                    f.pixel.remove_size(col.max(0) as usize);
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_add_feature_size(move |text| {
                let size = text.trim().replace(',', ".").parse::<f32>();
                manager.edit_features(|f| {
                    if let Ok(size) = size {
                        f.pixel.add_size(size);
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_shape_metric(move |i| {
                manager.edit_features(|f| {
                    f.object_preset = None;
                    if let Some(on) = f.object.shape.get_mut(i as usize) {
                        *on = !*on;
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_brightness_cell(move |row, col| {
                manager.edit_features(|f| {
                    f.object_preset = None;
                    if let Some(on) = f
                        .object
                        .brightness
                        .get_mut(row as usize)
                        .and_then(|r| r.get_mut(col as usize))
                    {
                        *on = !*on;
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_brightness_row(move |row| {
                manager.edit_features(|f| {
                    f.object_preset = None;
                    if let Some(r) = f.object.brightness.get_mut(row as usize) {
                        let all = r.iter().all(|on| *on);
                        r.fill(!all);
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_toggle_brightness_column(move |col| {
                manager.edit_features(|f| {
                    f.object_preset = None;
                    let col = col as usize;
                    let all = f
                        .object
                        .brightness
                        .iter()
                        .all(|r| r.get(col) == Some(&true));
                    for r in &mut f.object.brightness {
                        if let Some(on) = r.get_mut(col) {
                            *on = !all;
                        }
                    }
                });
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_apply_object_preset(move |i| {
                if let Some(preset) = preset_of(i) {
                    manager.edit_features(|f| {
                        let channels = f.channels.len();
                        f.object = ObjectFeatureTable::preset(preset, channels);
                        f.object_preset = Some(preset);
                    });
                }
            });
        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_class_selection_changed(move || {
                manager.sync_readiness_to_slint();
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
            .on_add_to_pipeline_clicked(move || {
                manager.add_to_pipeline();
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_browse_existing_model_clicked(move || {
                manager.browse_existing_model();
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_cancel_training_clicked(move || {
                if let Some(cancel) = manager.training_cancel_flag.lock().unwrap().as_ref() {
                    cancel.cancel();
                }
            });

        let manager = self.clone();
        ui.global::<AiLearningState>()
            .on_train_clicked(move |settings, class_selection| {
                manager.train(settings, class_selection);
            });
    }

    // -- Features ----------------------------------------------------------

    /// Changes the feature tables and shows them again.
    fn edit_features(self: &Arc<Self>, edit: impl FnOnce(&mut Features)) {
        edit(&mut self.features.lock().expect("Poisoned"));
        self.push_features_to_slint();
    }

    /// The feature tables as Slint models, with the presets they match and
    /// the feature count.
    fn push_features_to_slint(self: &Arc<Self>) {
        // Plain copies over to the UI thread (Slint models aren't `Send`).
        let (pixel, channels, object, object_preset) = {
            let f = self.features.lock().expect("Poisoned");
            let object_preset = f
                .object_preset
                .or_else(|| f.object.matches(f.channels.len()));
            (
                f.pixel.clone(),
                f.channels.clone(),
                f.object.clone(),
                object_preset,
            )
        };
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<AiLearningState>();
            // The chips get the channels' colors.
            let colors: Vec<slint::Color> = ui
                .global::<ChannelState>()
                .get_channels()
                .iter()
                .map(|c| c.color)
                .collect();
            let chips: Vec<ChannelChipSlint> = channels
                .iter()
                .enumerate()
                .map(|(i, (name, on))| ChannelChipSlint {
                    name: name.clone().into(),
                    color: colors.get(i).copied().unwrap_or_default(),
                    selected: *on,
                })
                .collect();
            let grid: Vec<FeatureGridRowSlint> = pixel
                .rows
                .iter()
                .zip(PIXEL_FEATURES.iter())
                .map(
                    |(row, (label, hint, has_raw, options))| FeatureGridRowSlint {
                        label: (*label).into(),
                        hint: (*hint).into(),
                        has_raw: *has_raw,
                        raw: row.raw,
                        cells: ModelRc::new(VecModel::from(row.cells.clone())),
                        options: ModelRc::new(VecModel::from(
                            options
                                .iter()
                                .map(|o| slint::SharedString::from(*o))
                                .collect::<Vec<_>>(),
                        )),
                        option: row.option as i32,
                    },
                )
                .collect();
            let sizes: Vec<slint::SharedString> =
                pixel.sizes.iter().map(|s| size_label(*s).into()).collect();
            let shape: Vec<ToggleRowSlint> = SHAPE_METRICS
                .iter()
                .zip(&object.shape)
                .map(|((label, _), on)| ToggleRowSlint {
                    label: (*label).into(),
                    selected: *on,
                })
                .collect();
            let brightness: Vec<BrightnessRowSlint> = BRIGHTNESS_STATS
                .iter()
                .zip(&object.brightness)
                .map(|((label, hint, _), cells)| BrightnessRowSlint {
                    label: (*label).into(),
                    hint: (*hint).into(),
                    cells: ModelRc::new(VecModel::from(cells.clone())),
                })
                .collect();
            let selected = channels.iter().filter(|c| c.1).count();
            let per_channel = pixel.count();
            let count = if state.get_settings().mode == 0 {
                format!("{per_channel} × {selected} ch = {}", per_channel * selected)
            } else {
                object.count().to_string()
            };
            state.set_pixel_channels(ModelRc::new(VecModel::from(chips)));
            state.set_feature_sizes(ModelRc::new(VecModel::from(sizes)));
            state.set_feature_grid(ModelRc::new(VecModel::from(grid)));
            state.set_pixel_preset(preset_index(pixel.matches()));
            state.set_shape_metrics(ModelRc::new(VecModel::from(shape)));
            state.set_brightness_rows(ModelRc::new(VecModel::from(brightness)));
            state.set_object_preset(preset_index(object_preset));
            state.set_ring_in_use(object.to_metrics().iter().any(ObjectMetric::uses_ring));
            state.set_feature_count(count.into());
            let legend: Vec<String> = channels
                .iter()
                .enumerate()
                .map(|(i, (name, _))| format!("{i} {name}"))
                .collect();
            state.set_channel_legend(legend.join(" · ").into());
        }) {
            warn!("Failed to sync AI classifier features to Slint: {}", e);
        }
    }

    /// The image's channels into the feature tables; `keep`: the channels
    /// the pixel classifier reads stay (by name), else the channel shown.
    fn sync_features_to_slint(self: &Arc<Self>, keep: bool) {
        let ui_weak = self.ui.clone();
        let manager = self.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let channel_state = ui.global::<ChannelState>();
            let names: Vec<String> = channel_state
                .get_channels()
                .iter()
                .map(|c| c.name.to_string())
                .collect();
            let shown = channel_state.get_selected_channel_index().max(0) as usize;
            {
                let mut f = manager.features.lock().expect("Poisoned");
                let before = std::mem::take(&mut f.channels);
                f.channels = names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let on = if keep && !before.is_empty() {
                            before.iter().any(|(n, on)| n == name && *on)
                        } else {
                            i == shown.min(names.len().saturating_sub(1))
                        };
                        (name.clone(), on)
                    })
                    .collect();
                // None of the channels chosen before is there (another
                // image, renamed): the one shown.
                if !f.channels.iter().any(|c| c.1)
                    && let Some(c) = f.channels.get_mut(shown.min(names.len().saturating_sub(1)))
                {
                    c.1 = true;
                }
                f.object = match f.object_preset {
                    Some(preset) => ObjectFeatureTable::preset(preset, names.len()),
                    None => f.object.with_channels(names.len()),
                };
            }
            manager.push_features_to_slint();
        }) {
            warn!("Failed to sync AI classifier channels to Slint: {}", e);
        }
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
            .map(|c| c.global())
        else {
            return;
        };

        let image_path = PathBuf::from(row.image_path.as_str());
        let object_id = row.object_id;
        let project_series = project.images.settings.selected_series;
        if let Some(entry) = project.images.list.get_mut(&image_path) {
            let series_idx = entry.series_for(project_series);
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
        let project_series = project.images.settings.selected_series;
        if let Some(entry) = project.images.list.get_mut(&image_path) {
            let series_idx = entry.series_for(project_series);
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
        let request = crate::FileRequest::open_file("Load AI classifier model").filter(
            "AI classifier model",
            &[evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS],
        );
        let this = Arc::clone(self);
        self.app_state.file_browser.open(request, move |path| {
            if let Some(path) = path {
                this.load_existing_model(path);
            }
        });
    }

    fn load_existing_model(self: &Arc<Self>, path: PathBuf) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };

        let loaded = match evanalyzer_app::ai_learning::load_classifier_settings(
            self.app_state.backend().files(),
            &path,
        ) {
            Ok(loaded) => loaded,
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
        settings.model_name = if loaded.meta.name.trim().is_empty() {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
                .into()
        } else {
            loaded.meta.name.clone().into()
        };
        apply_loaded_backend_settings(&mut settings, &loaded.backend);

        match &loaded.classifier {
            AiLearningClassifierSettings::Pixel { feature_spec, .. } => {
                settings.mode = 0;
                let table = PixelFeatureTable::from_recipe(&feature_spec.channels);
                let mut f = self.features.lock().expect("Poisoned");
                if let Some(table) = table {
                    f.pixel = table;
                } else {
                    warn!(
                        "'{}': features not from this panel - kept the table",
                        path.display()
                    );
                }
                if !feature_spec.image_channels.is_empty() {
                    for (i, channel) in f.channels.iter_mut().enumerate() {
                        channel.1 = feature_spec.image_channels.contains(&(i as i32));
                    }
                }
            }
            AiLearningClassifierSettings::Object {
                feature_spec,
                class_labels,
            } => {
                settings.mode = 1;

                {
                    let mut f = self.features.lock().expect("Poisoned");
                    let channels = f.channels.len();
                    f.object = ObjectFeatureTable::from_metrics(&feature_spec.metrics, channels);
                    f.object_preset = None;
                }
                settings.ring_width = feature_spec.ring_width as i32;

                let project = self.app_state.get_project();
                let classes = project.classification.classes().clone();
                drop(project);
                let mut class_rows: Vec<ClassSelectionRowSlint> =
                    state.get_class_selection().iter().collect();
                for row in &mut class_rows {
                    let class_id = classes.get(row.class_index as usize).map(|c| c.global());
                    row.selected =
                        class_id.is_some_and(|id| class_labels.iter().any(|l| l.class == id));
                }
                state.set_class_selection(ModelRc::new(VecModel::from(class_rows)));
            }
        }

        state.set_settings(settings);
        self.push_features_to_slint();
        self.set_training_status(
            &format!("Loaded settings from '{}'.", path.display()),
            false,
        );
    }

    // -- Training entry point ------------------------------------------------

    /// Builds an `AiLearningSettings` from the dialog's state, gathers
    /// training data from the project (every object with an assigned class -
    /// see `evanalyzer_app::backend::local::training::start_training`'s doc comment;
    /// `training_images`/`training_objects` aren't needed here since labels
    /// already live on the project's objects via `assign_object_class`, not
    /// in Slint-only state), then runs training on a background thread and
    /// saves the result under `<project>/models/<settings.model_name>`.
    fn train(
        self: &Arc<Self>,
        settings: AiTrainingSettingsSlint,
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
                    .map(|class| class.global())
            })
            .collect();

        let (recipe, image_channels, image_channel_names, metrics) = {
            let f = self.features.lock().expect("Poisoned");
            let (image_channels, image_channel_names): (Vec<i32>, Vec<String>) = f
                .channels
                .iter()
                .enumerate()
                .filter(|(_, c)| c.1)
                .map(|(i, c)| (i as i32, c.0.clone()))
                .unzip();
            (
                f.pixel.to_recipe(),
                image_channels,
                image_channel_names,
                f.object.to_metrics(),
            )
        };
        if settings.mode == 0 && (recipe.is_empty() || image_channels.is_empty()) {
            self.set_training_status("Tick at least one feature and one channel.", true);
            return;
        }
        if settings.mode == 1 && metrics.is_empty() {
            self.set_training_status("Tick at least one feature.", true);
            return;
        }
        let ai_settings = build_ai_learning_settings(
            &settings,
            AiLearningPixelFeatureSettings {
                channels: recipe,
                image_channels: image_channels.clone(),
                image_channel_names,
            },
            metrics,
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
            // Not read: the features name their channels.
            channel: image_channels.first().copied().unwrap_or(0),
            z_stack_handling,
        };

        let classifier_settings = ai_settings.classifier.clone();
        let training = match self.app_state.backend().start_training(TrainingRequest {
            project: project_settings,
            settings: ai_settings,
            pixel_params,
            save_to: Some(evanalyzer_app::ai_learning::ModelDestination {
                project_dir: project_dir.clone(),
                model_name: model_name.clone(),
            }),
        }) {
            Ok(training) => training,
            Err(e) => {
                self.set_training_status(&e.to_string(), true);
                return;
            }
        };

        // Every pre-flight check passed - hand off to the background worker.
        // The dialog is never closed here (or on completion below) so the
        // user can watch it finish and immediately retrain if they want to.
        info!("Starting classifier training ('{model_name}')");
        self.set_can_add_to_pipeline(false);
        self.set_training_status("Training started...", false);
        self.set_training_in_progress(true);
        *self.training_cancel_flag.lock().unwrap() = Some(training.cancel_handle());
        let manager = self.clone();
        crate::helper::ui_thread::spawn(move || {
            // Captured here (rather than re-derived after `wait()`,
            // which only returns the `SavedClassifier`/error, not the
            // backend stats) so the final banner below can report it - see
            // `describe_training_progress`'s doc comment for why `Finished`
            // doesn't update the live banner itself.
            let mut stats: Option<evanalyzer_cfg::core_types::TrainingStats> = None;
            for event in training.events() {
                if let TrainingProgressEvent::Finished { stats: s } = event {
                    stats = Some(s);
                    continue;
                }
                if let Some(line) = describe_training_progress(&event) {
                    manager.set_training_status(&line, false);
                }
            }

            let result = training.wait();

            let (message, is_error) = match result {
                Ok(classifier) => {
                    let stats_line = stats
                        .as_ref()
                        .map(format_training_stats)
                        .unwrap_or_default();
                    match evanalyzer_app::ai_learning::save_trained_model(
                        manager.app_state.backend().files(),
                        &classifier,
                        &project_dir,
                        &model_name,
                    ) {
                        Ok(path) => {
                            info!("Classifier training completed: {}", path.display());
                            manager.app_state.model_saved();
                            let used = evanalyzer_app::ai_learning::model_is_used(
                                &manager.app_state.get_project().settings,
                                &path,
                            );
                            *manager.last_model.lock().expect("Poisoned") =
                                Some((path.clone(), classifier_settings.clone()));
                            manager.set_can_add_to_pipeline(!used);
                            let preview = if used {
                                " The preview shows it with auto preview on."
                            } else {
                                " No pipeline uses it yet - \"Add to pipeline\" adds one that does."
                            };
                            (
                                format!(
                                    "Training complete. {stats_line} Model saved to {}.{preview}",
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
    /// Adds a pipeline applying the model last trained (see
    /// `pipeline_for_model`) - one undo step; the pipelines panel shows and
    /// selects it, the auto preview runs it.
    fn add_to_pipeline(self: &Arc<Self>) {
        let Some((model, classifier)) = self.last_model.lock().expect("Poisoned").clone() else {
            return;
        };
        self.app_state.begin_undo_step();
        let (id, name) = {
            let mut project = self.app_state.get_project_write();
            let id = project.pipelines.iter().map(|p| p.id.0).max().unwrap_or(0) + 1;
            let pipeline = evanalyzer_app::ai_learning::pipeline_for_model(
                &project,
                &model,
                &classifier,
                evanalyzer_cfg::core_types::PipelineId(id),
            );
            let name = pipeline.name.clone();
            project.add_pipeline(pipeline);
            (id, name)
        };
        self.app_state.mark_dirty();
        self.app_state.pipeline_added(id);
        self.set_can_add_to_pipeline(false);
        self.set_training_status(
            &format!(
                "Added the pipeline '{name}' - the preview shows its result with auto preview on."
            ),
            false,
        );
    }

    fn set_can_add_to_pipeline(self: &Arc<Self>, can: bool) {
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<AiLearningState>().set_can_add_to_pipeline(can);
            }
        }) {
            warn!("Failed to update can_add_to_pipeline: {e}");
        }
    }

    fn set_training_status(self: &Arc<Self>, message: &str, is_error: bool) {
        let message = message.to_owned();
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<AiLearningState>()
                    .set_training_in_progress(in_progress);
            }
        }) {
            warn!("Failed to update training_in_progress: {e}");
        }
    }

    // -- Sync project state -> Slint -----------------------------------------

    /// Every list of the dialog from the current project; `keep`: the rows
    /// the user ticked stay ticked (a refresh while open), else the
    /// defaults (opening it).
    fn sync_all_to_slint(self: &Arc<Self>, keep: bool) {
        self.sync_class_names_to_slint();
        self.sync_channel_names_to_slint();
        self.sync_features_to_slint(keep);
        self.sync_training_images_to_slint(keep);
        self.sync_training_objects_to_slint();
        self.sync_class_selection_to_slint(keep);
    }

    pub fn sync_class_names_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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
    pub fn sync_class_selection_to_slint(self: &Arc<Self>, keep: bool) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let project = app_state.get_project();
            let used = evanalyzer_app::ai_learning::used_object_classes(&project.settings);
            // Hand-drawn objects per class, and the images that have any.
            let mut counts: std::collections::HashMap<_, i32> = std::collections::HashMap::new();
            let mut annotated_images = 0;
            let mut examples = 0;
            let mut current_annotated = false;
            let current = project.get_current_rel_image_path_cloned();
            for (path, image) in project.images.list.iter() {
                let mut any = false;
                for series in image.series.values() {
                    for object in &series.objects {
                        for class in &object.object_class {
                            *counts.entry(*class).or_default() += 1;
                            any = true;
                        }
                        examples += i32::from(!object.object_class.is_empty());
                    }
                }
                annotated_images += i32::from(any);
                current_annotated |= any && current.as_ref() == Some(path);
            }
            let summary = examples_summary(annotated_images, current_annotated, examples);
            // Ticked by the user, by class name (kept on a refresh).
            let ticked: Option<std::collections::HashSet<slint::SharedString>> = keep.then(|| {
                ui.global::<AiLearningState>()
                    .get_class_selection()
                    .iter()
                    .filter(|row| row.selected)
                    .map(|row| row.name)
                    .collect()
            });
            let rows: Vec<ClassSelectionRowSlint> = project
                .classification
                .classes()
                .iter()
                .enumerate()
                .map(|(i, c)| ClassSelectionRowSlint {
                    class_index: i as i32,
                    name: c.name.clone().into(),
                    selected: match &ticked {
                        Some(ticked) => ticked.contains(c.name.as_str()),
                        None => used.contains(&c.global()),
                    },
                    color: slint::Color::from_argb_encoded(c.color | 0xff00_0000),
                    count: counts.get(&c.global()).copied().unwrap_or(0),
                })
                .collect();
            drop(project);
            let state = ui.global::<AiLearningState>();
            let (text, tone) = readiness(&rows, annotated_images);
            state.set_class_selection(ModelRc::new(VecModel::from(rows)));
            state.set_annotated_image_count(annotated_images);
            state.set_examples_summary(summary.into());
            state.set_readiness_text(text.into());
            state.set_readiness_tone(tone);
        }) {
            warn!(
                "Failed to sync AI classifier class selection to Slint: {}",
                e
            );
        }
    }

    /// The footer's hint again (a class ticked or unticked).
    fn sync_readiness_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        let manager = self.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let state = ui.global::<AiLearningState>();
            let rows: Vec<ClassSelectionRowSlint> = state.get_class_selection().iter().collect();
            let (text, tone) = readiness(&rows, state.get_annotated_image_count());
            state.set_readiness_text(text.into());
            state.set_readiness_tone(tone);
            // Pixel or object classifier: the feature count says which.
            manager.push_features_to_slint();
        }) {
            warn!("Failed to sync AI classifier readiness to Slint: {}", e);
        }
    }

    /// Channel names for the pixel-classifier's channel picker - index
    /// matches how `train()` reads back `settings.pixel_channel`, and the
    /// same position-based convention `sync_object_metrics_to_slint` already
    /// uses for its per-channel intensity metric rows.
    pub fn sync_channel_names_to_slint(self: &Arc<Self>) {
        let ui_weak = self.ui.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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

    pub fn sync_training_images_to_slint(self: &Arc<Self>, keep: bool) {
        let ui_weak = self.ui.clone();
        let app_state = self.app_state.clone();
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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
            let ticked: std::collections::HashSet<slint::SharedString> = if keep {
                ui.global::<AiLearningState>()
                    .get_training_images()
                    .iter()
                    .filter(|row| row.selected)
                    .map(|row| row.path)
                    .collect()
            } else {
                Default::default()
            };

            let rows: Vec<TrainingImageRowSlint> = project
                .images
                .list
                .iter()
                .filter(|(path, _)| Some((*path).clone()) != current_path)
                .map(|(path, entry)| {
                    let annotated = entry
                        .series
                        .get(&entry.active_series(&project.images.settings))
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
                        selected: ticked.contains(path.to_string_lossy().as_ref()),
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
        if let Err(e) = crate::helper::ui_thread::invoke_from_event_loop(move || {
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
                    let active = entry.active_series(&project.images.settings);
                    if let Some(series) = entry.series.get(&active) {
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

/// The panel's preset index (0 simple, 1 recommended, 2 all) as a preset.
fn preset_of(index: i32) -> Option<Preset> {
    match index {
        0 => Some(Preset::Simple),
        1 => Some(Preset::Recommended),
        2 => Some(Preset::All),
        _ => None,
    }
}

/// A preset as the panel's index; -1: none (edited by hand).
fn preset_index(preset: Option<Preset>) -> i32 {
    match preset {
        Some(Preset::Simple) => 0,
        Some(Preset::Recommended) => 1,
        Some(Preset::All) => 2,
        None => -1,
    }
}

/// "This image + 2 more · 94": where the examples come from.
fn examples_summary(images: i32, current: bool, examples: i32) -> String {
    match (images, current) {
        (0, _) => "none yet".into(),
        (1, true) => format!("This image · {examples}"),
        (n, true) => format!("This image + {} more · {examples}", n - 1),
        (n, false) => format!("{n} image{} · {examples}", if n == 1 { "" } else { "s" }),
    }
}

/// The footer's hint for the ticked classes: (text, tone - 0 can't train
/// yet, 1 trains but weak, 2 ready).
fn readiness(rows: &[ClassSelectionRowSlint], images: i32) -> (String, i32) {
    let ticked: Vec<&ClassSelectionRowSlint> = rows.iter().filter(|r| r.selected).collect();
    let with_examples = ticked.iter().filter(|r| r.count > 0).count();
    if with_examples < 2 {
        return (
            "Not yet: draw examples of at least 2 classes (and tick them).".into(),
            0,
        );
    }
    if let Some(empty) = ticked.iter().find(|r| r.count == 0) {
        return (
            format!("{} has no examples - draw some or untick it.", empty.name),
            1,
        );
    }
    if let Some(few) = ticked.iter().find(|r| r.count < FEW_EXAMPLES) {
        return (
            format!(
                "{} has {} example{} - it trains, but draw ~{FEW_EXAMPLES} or more for a steadier model.",
                few.name,
                few.count,
                if few.count == 1 { "" } else { "s" }
            ),
            1,
        );
    }
    let examples: i32 = ticked.iter().map(|r| r.count).sum();
    (
        format!(
            "Ready: {} classes, {examples} examples from {images} image{}.",
            ticked.len(),
            if images == 1 { "" } else { "s" }
        ),
        2,
    )
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
            .find_map(|oc| classes.iter().position(|c| &c.global() == oc))
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
    pixel_features: AiLearningPixelFeatureSettings,
    object_metrics: Vec<ObjectMetric>,
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
            feature_spec: pixel_features,
            input_color: Default::default(),
            class_labels: evanalyzer_app::ai_learning::pixel_class_labels_from_project(
                project,
                selected_classes,
            ),
        }
    } else {
        AiLearningClassifierSettings::Object {
            feature_spec: AiLearningObjectFeatureSettings {
                metrics: object_metrics,
                ring_width: settings.ring_width.max(1) as u32,
            },
            class_labels: evanalyzer_app::ai_learning::object_class_labels_from_project(
                project,
                selected_classes,
            ),
        }
    };

    AiLearningSettings {
        schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
        meta: MetaData {
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
            log::debug!("Image {image_index}: {total_tiles} tile(s) scheduled");
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
            log::debug!("Image {image_index}: tile {tile_index}/{total_tiles} processed");
            Some(format!(
                "Image {}: tile {}/{total_tiles}",
                image_index + 1,
                tile_index + 1
            ))
        }
        TrainingProgressEvent::ItemCompleted { index, total } => {
            log::debug!("Training progress: {index}/{total}");
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
fn format_training_stats(stats: &evanalyzer_cfg::core_types::TrainingStats) -> String {
    use evanalyzer_cfg::core_types::TrainingStats;
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
    use evanalyzer_app::project::ImageListExt;

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
                Default::default(),
                Vec::new(),
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
    // wired callback - see `crate::editor::test_support::test_ui_window`'s
    // doc comment for why this headless harness doesn't need a real display.

    use crate::editor::test_support::{
        project_with_one_image, test_ui_state_with_project, test_ui_window,
    };
    use evanalyzer_app::project::ProjectExt;
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::classification_settings::Class;

    fn make_controller_with_ui(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
    ) -> Arc<AiLearningController> {
        Arc::new(AiLearningController::new(ui, app_state))
    }

    #[test]
    fn attach_callbacks_toggle_image_selected_flips_only_the_given_row() {
        let ui = test_ui_window();
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
        let ui = test_ui_window();
        let mut project = project_with_one_image();
        project.classification.classes_mut().push(Class {
            id: 1,
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
        let obj = &project
            .images
            .list
            .get(std::path::Path::new("img.tif"))
            .unwrap()
            .series[&0]
            .objects[0];
        assert!(
            obj.object_class
                .contains(&evanalyzer_cfg::core_types::ObjectClass::Global(1))
        );
        drop(project);

        let rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        assert_eq!(rows[0].assigned_class_index, 1);
    }

    #[test]
    fn attach_callbacks_toggle_object_excluded_flips_the_flag_on_the_project_object() {
        let ui = test_ui_window();
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
        let obj = &project
            .images
            .list
            .get(std::path::Path::new("img.tif"))
            .unwrap()
            .series[&0]
            .objects[0];
        assert!(obj.exclude_from_training);
        drop(project);

        let rows: Vec<TrainingObjectRowSlint> = state.get_training_objects().iter().collect();
        assert!(rows[0].excluded);

        // Toggling again flips it back.
        state.invoke_toggle_object_excluded(0);
        let project = ui_state.get_project();
        assert!(
            !project
                .images
                .list
                .get(std::path::Path::new("img.tif"))
                .unwrap()
                .series[&0]
                .objects[0]
                .exclude_from_training
        );
    }

    #[test]
    fn attach_callbacks_assign_object_class_with_an_unknown_row_index_is_a_no_op() {
        let ui = test_ui_window();
        let ui_state = crate::editor::test_support::test_ui_state();
        let controller = make_controller_with_ui(ui.as_weak(), ui_state);
        controller.attach_callbacks();

        // No panic with an empty training_objects list.
        ui.global::<AiLearningState>()
            .invoke_assign_object_class(0, 0);
        ui.global::<AiLearningState>()
            .invoke_toggle_object_excluded(0);
    }

    // -- progress / stats texts ----------------------------------------------

    #[test]
    fn describe_training_progress_has_a_line_for_every_event_but_finished() {
        use evanalyzer_cfg::core_types::TrainingStats;
        let line = |e: TrainingProgressEvent| describe_training_progress(&e);
        assert_eq!(
            line(TrainingProgressEvent::Started { total: 3 }).unwrap(),
            "Training started - 3 item(s) to process..."
        );
        assert_eq!(
            line(TrainingProgressEvent::ImageTilesScheduled {
                image_index: 0,
                total_tiles: 4
            })
            .unwrap(),
            "Image 1: scanning 4 tile(s)..."
        );
        assert_eq!(
            line(TrainingProgressEvent::TileProcessed {
                image_index: 1,
                tile_index: 2,
                total_tiles: 4
            })
            .unwrap(),
            "Image 2: tile 3/4"
        );
        assert_eq!(
            line(TrainingProgressEvent::ItemCompleted { index: 2, total: 5 }).unwrap(),
            "Processed 2/5 item(s)..."
        );
        assert!(
            line(TrainingProgressEvent::ImageFailed {
                path: "/x/a.tif".into()
            })
            .unwrap()
            .contains("failed to read /x/a.tif")
        );
        assert_eq!(
            line(TrainingProgressEvent::ObjectSkipped {
                index: 7,
                reason: "ambiguous".into()
            })
            .unwrap(),
            "Warning: object 7 skipped (ambiguous)"
        );
        assert_eq!(
            line(TrainingProgressEvent::Training).unwrap(),
            "Fitting model..."
        );
        assert_eq!(
            line(TrainingProgressEvent::Epoch {
                epoch: 0,
                total_epochs: 10,
                train_loss: 0.5,
                val_loss: Some(0.25)
            })
            .unwrap(),
            "Epoch 1/10 - train loss 0.5000, validation loss 0.2500"
        );
        assert_eq!(
            line(TrainingProgressEvent::Epoch {
                epoch: 9,
                total_epochs: 10,
                train_loss: 0.125,
                val_loss: None
            })
            .unwrap(),
            "Epoch 10/10 - train loss 0.1250"
        );
        assert!(
            line(TrainingProgressEvent::Finished {
                stats: TrainingStats::Knn { k: 3, n_samples: 9 }
            })
            .is_none()
        );
    }

    #[test]
    fn format_training_stats_per_backend_and_flags_overfitting() {
        use evanalyzer_cfg::core_types::TrainingStats;
        assert_eq!(
            format_training_stats(&TrainingStats::RandomForest {
                n_trees: 10,
                n_samples: 50
            }),
            "10 tree(s), trained on 50 sample(s)."
        );
        assert_eq!(
            format_training_stats(&TrainingStats::Knn {
                k: 5,
                n_samples: 20
            }),
            "k=5, trained on 20 sample(s)."
        );
        let mlp = |final_val: Option<f32>, best_val: Option<f32>, best_epoch: Option<usize>| {
            format_training_stats(&TrainingStats::Mlp {
                epochs_run: 100,
                total_epochs: 100,
                final_train_loss: 0.1,
                final_val_loss: final_val,
                best_val_loss: best_val,
                best_val_epoch: best_epoch,
            })
        };
        assert_eq!(
            mlp(None, None, None),
            "100/100 epoch(s), final train loss 0.1000."
        );
        assert_eq!(
            mlp(Some(0.2), Some(0.2), Some(99)),
            "100/100 epoch(s), final train loss 0.1000, validation loss 0.2000."
        );
        let overfit = mlp(Some(0.5), Some(0.2), Some(10));
        assert!(overfit.contains("possible overfitting"), "{overfit}");
        assert!(overfit.contains("epoch 11"), "{overfit}");
    }

    // -- dialog sync and training, with a window -----------------------------

    use crate::editor::test_support::{choose_file, ui_state_with_window};
    use crate::helper::ui_thread::drain_ui_queue;

    const CELL: ObjectClass = ObjectClass::Global(1);
    const NUCLEUS: ObjectClass = ObjectClass::Global(2);

    /// One image with labeled objects of two classes (areas well apart),
    /// plus a second, unselected image with one labeled object.
    fn labeled_project() -> evanalyzer_app::project::ProjectWithRuntime {
        let mut project = project_with_one_image();
        for (id, name) in [(CELL, "Cell"), (NUCLEUS, "Nucleus")] {
            project.classification.classes_mut().push(Class {
                id: id.to_u32().unwrap(),
                name: name.into(),
                ..Default::default()
            });
        }
        for i in 0..6u32 {
            let class = if i % 2 == 0 { CELL } else { NUCLEUS };
            project.add_object(&ObjectMetricSettings {
                id: ObjectId((i + 1).into()),
                area: if class == CELL {
                    10 + i as usize
                } else {
                    1000 + i as usize
                },
                object_class: [class].into(),
                ..Default::default()
            });
        }
        let mut other = project
            .images
            .list
            .get(std::path::Path::new("img.tif"))
            .unwrap()
            .clone();
        other.rel_path = PathBuf::from("other.tif");
        for series in other.series.values_mut() {
            series.objects.truncate(1);
        }
        project
            .images
            .list
            .insert(PathBuf::from("other.tif"), other);
        project
    }

    struct Dialog {
        ui: AppWindow,
        ui_state: Arc<UiState>,
        controller: Arc<AiLearningController>,
    }

    fn opened(project: evanalyzer_app::project::ProjectWithRuntime) -> Dialog {
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, project);
        let controller = make_controller_with_ui(ui.as_weak(), ui_state.clone());
        controller.attach_callbacks();
        ui.global::<ChannelState>()
            .set_channels(ModelRc::new(VecModel::from(vec![crate::ChannelInfo {
                name: "DAPI".into(),
                active: true,
                idx: 0,
                color: slint::Color::from_rgb_u8(0, 0, 255),
                emission_wave_length: 461.0,
                wavelength_overridden: false,
            }])));
        ui.global::<AiLearningState>().invoke_open_requested();
        drain_ui_queue();
        Dialog {
            ui,
            ui_state,
            controller,
        }
    }

    impl Dialog {
        fn state(&self) -> AiLearningState<'_> {
            self.ui.global::<AiLearningState>()
        }

        fn train(&self) {
            let s = self.state();
            s.invoke_train_clicked(s.get_settings(), s.get_class_selection());
            drain_ui_queue();
        }
    }

    #[test]
    fn opening_the_dialog_fills_it_from_the_project() {
        let d = opened(labeled_project());
        let s = d.state();
        assert_eq!(
            d.ui.global::<GlobalAppState>().get_active_dialog(),
            DialogType::AiLearning
        );
        let names: Vec<String> = s.get_class_names().iter().map(|n| n.to_string()).collect();
        assert!(names.contains(&"Cell".to_string()) && names.contains(&"Nucleus".to_string()));
        assert_eq!(s.get_channel_names().row_count(), 1);
        // The feature tables: the one channel, read by the pixel classifier;
        // the brightness table has a column for it.
        let chips: Vec<ChannelChipSlint> = s.get_pixel_channels().iter().collect();
        assert_eq!(chips.len(), 1);
        assert!(chips[0].selected);
        assert_eq!(s.get_feature_grid().row_count(), PIXEL_FEATURES.len());
        assert_eq!(s.get_pixel_preset(), 1, "recommended");
        assert_eq!(s.get_feature_count(), "10 × 1 ch = 10");
        assert_eq!(s.get_shape_metrics().row_count(), SHAPE_METRICS.len());
        assert!(
            s.get_brightness_rows()
                .iter()
                .all(|r| r.cells.row_count() == 1)
        );
        // The current image is not offered as "another" image.
        let images: Vec<TrainingImageRowSlint> = s.get_training_images().iter().collect();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].name, "other.tif");
        assert_eq!(images[0].annotated_object_count, 1);
        assert_eq!(s.get_training_objects().row_count(), 6);
        // Classes with labeled objects start checked.
        let selected: Vec<String> = s
            .get_class_selection()
            .iter()
            .filter(|r| r.selected)
            .map(|r| r.name.to_string())
            .collect();
        assert_eq!(selected, ["Cell", "Nucleus"]);
        assert_eq!(s.get_training_status(), "");
    }

    #[test]
    fn selecting_another_image_adds_its_objects() {
        let d = opened(labeled_project());
        d.state().invoke_toggle_image_selected(0);
        drain_ui_queue();
        assert_eq!(d.state().get_training_objects().row_count(), 7);
        d.state().invoke_toggle_image_selected(0);
        drain_ui_queue();
        assert_eq!(d.state().get_training_objects().row_count(), 6);
    }

    /// An object drawn while the dialog is open: the training objects and
    /// the class counts follow at once (the user's class ticks stay).
    #[test]
    fn an_object_drawn_while_open_shows_up_in_the_dialog() {
        let d = opened(labeled_project());
        let s = d.state();
        assert_eq!(s.get_training_objects().row_count(), 6);
        // The user unticks "Nucleus" (only "Cell" left).
        let rows: Vec<ClassSelectionRowSlint> = s
            .get_class_selection()
            .iter()
            .map(|mut r| {
                r.selected = r.name == "Cell";
                r
            })
            .collect();
        s.set_class_selection(ModelRc::new(VecModel::from(rows)));
        let count = |s: &AiLearningState<'_>, name: &str| {
            s.get_class_selection()
                .iter()
                .find(|r| r.name == name)
                .unwrap()
                .count
        };
        let cells = count(&s, "Cell");

        d.ui_state
            .get_project_write()
            .add_object(&ObjectMetricSettings {
                id: ObjectId(100),
                area: 12,
                object_class: [CELL].into(),
                ..Default::default()
            });
        // What every object edit does: the class counts are refreshed.
        s.invoke_project_changed();
        drain_ui_queue();

        assert_eq!(s.get_training_objects().row_count(), 7);
        assert_eq!(count(&s, "Cell"), cells + 1);
        let selected: Vec<String> = s
            .get_class_selection()
            .iter()
            .filter(|r| r.selected)
            .map(|r| r.name.to_string())
            .collect();
        assert_eq!(selected, ["Cell"], "the user's ticks stay");
    }

    /// Opened before the image's channels were known: the channels and the
    /// brightness columns appear once they are - and what the user ticked
    /// stays.
    #[test]
    fn channels_arriving_while_open_fill_the_feature_tables() {
        let ui = test_ui_window();
        let ui_state = ui_state_with_window(&ui, labeled_project());
        let controller = make_controller_with_ui(ui.as_weak(), ui_state.clone());
        controller.attach_callbacks();
        let s = ui.global::<AiLearningState>();
        s.invoke_open_requested();
        drain_ui_queue();
        assert_eq!(s.get_pixel_channels().row_count(), 0);
        assert!(
            s.get_brightness_rows()
                .iter()
                .all(|r| r.cells.row_count() == 0)
        );

        let channel = |name: &str, idx: i32| crate::ChannelInfo {
            name: name.into(),
            active: true,
            idx,
            color: slint::Color::from_rgb_u8(0, 0, 255),
            emission_wave_length: 461.0,
            wavelength_overridden: false,
        };
        ui.global::<ChannelState>()
            .set_channels(ModelRc::new(VecModel::from(vec![channel("DAPI", 0)])));
        s.invoke_project_changed();
        drain_ui_queue();
        assert_eq!(s.get_pixel_channels().row_count(), 1);
        // The user ticks Sum for DAPI.
        s.invoke_toggle_brightness_cell(6, 0);
        drain_ui_queue();
        assert!(
            s.get_brightness_rows()
                .row_data(6)
                .unwrap()
                .cells
                .row_data(0)
                .unwrap()
        );

        ui.global::<ChannelState>()
            .set_channels(ModelRc::new(VecModel::from(vec![
                channel("DAPI", 0),
                channel("FITC", 1),
            ])));
        s.invoke_project_changed();
        drain_ui_queue();
        let chips: Vec<ChannelChipSlint> = s.get_pixel_channels().iter().collect();
        assert_eq!(chips.len(), 2);
        assert!(
            chips[0].selected && !chips[1].selected,
            "the user's channel stays"
        );
        let sum: Vec<bool> = s
            .get_brightness_rows()
            .row_data(6)
            .unwrap()
            .cells
            .iter()
            .collect();
        assert_eq!(
            sum,
            [true, true],
            "a new channel gets the first channel's ticks"
        );
    }

    /// Clicks in the pixel table change the features; a preset puts them
    /// back; the count follows the channels.
    #[test]
    fn the_pixel_table_follows_clicks_presets_and_channels() {
        let d = opened(labeled_project());
        let s = d.state();
        s.invoke_toggle_feature_cell(2, 0); // Gaussian gradient · 1 px off
        drain_ui_queue();
        assert_eq!(s.get_feature_count(), "9 × 1 ch = 9");
        assert_eq!(s.get_pixel_preset(), -1, "edited: no preset");
        s.invoke_set_feature_option(6, 2); // Rank filter: min
        s.invoke_add_feature_size("3".into());
        drain_ui_queue();
        assert_eq!(s.get_feature_sizes().row_count(), 6);
        assert_eq!(s.get_feature_grid().row_data(6).unwrap().option, 2);
        s.invoke_apply_pixel_preset(2);
        drain_ui_queue();
        assert_eq!(s.get_pixel_preset(), 2);
        assert_eq!(s.get_feature_count(), "31 × 1 ch = 31");
        s.invoke_toggle_pixel_channel(0);
        drain_ui_queue();
        assert_eq!(s.get_feature_count(), "31 × 0 ch = 0");
    }

    /// The object presets: "Recommended" ticks mean, median and std dev for
    /// every channel and is shown as chosen.
    #[test]
    fn the_object_presets_tick_their_features() {
        let d = opened(labeled_project());
        let s = d.state();
        let mut settings = s.get_settings();
        settings.mode = 1;
        s.set_settings(settings);
        s.invoke_class_selection_changed();
        drain_ui_queue();
        // Opening: Recommended, with the channel that arrived after it.
        assert_eq!(s.get_object_preset(), 1);
        assert_eq!(s.get_feature_count(), "10");
        s.invoke_apply_object_preset(0);
        drain_ui_queue();
        assert_eq!(s.get_object_preset(), 0);
        s.invoke_apply_object_preset(1);
        drain_ui_queue();
        assert_eq!(s.get_object_preset(), 1);
        let mean: Vec<bool> = s
            .get_brightness_rows()
            .row_data(0)
            .unwrap()
            .cells
            .iter()
            .collect();
        assert_eq!(mean, [true]);
        assert_eq!(s.get_feature_count(), "10");
    }

    /// The channels get other names (another image): none of the chosen
    /// ones is there any more - the channel shown is read, never none.
    #[test]
    fn renamed_channels_still_leave_one_to_read() {
        let d = opened(labeled_project());
        let s = d.state();
        let channel = |name: &str, idx: i32| crate::ChannelInfo {
            name: name.into(),
            active: true,
            idx,
            color: slint::Color::from_rgb_u8(0, 0, 255),
            emission_wave_length: 461.0,
            wavelength_overridden: false,
        };
        d.ui.global::<ChannelState>()
            .set_channels(ModelRc::new(VecModel::from(vec![
                channel("Channel 0", 0),
                channel("Channel 1", 1),
            ])));
        s.invoke_project_changed();
        drain_ui_queue();
        let chips: Vec<ChannelChipSlint> = s.get_pixel_channels().iter().collect();
        assert_eq!(chips.iter().filter(|c| c.selected).count(), 1, "{chips:?}");
        assert_eq!(s.get_feature_count(), "10 × 1 ch = 10");
    }

    #[test]
    fn the_footer_says_whether_it_can_train() {
        let row = |name: &str, count: i32, selected: bool| ClassSelectionRowSlint {
            class_index: 0,
            name: name.into(),
            selected,
            color: Default::default(),
            count,
        };
        assert_eq!(readiness(&[row("Cell", 9, true)], 1).1, 0, "one class");
        assert_eq!(
            readiness(&[row("Cell", 9, true), row("Spot", 9, false)], 1).1,
            0,
            "the other unticked"
        );
        let (text, tone) = readiness(&[row("Cell", 9, true), row("Spot", 3, true)], 2);
        assert_eq!(tone, 1);
        assert!(text.starts_with("Spot has 3 examples"), "{text}");
        let (text, tone) = readiness(
            &[
                row("Cell", 9, true),
                row("Spot", 0, true),
                row("Bg", 4, true),
            ],
            2,
        );
        assert_eq!(tone, 1);
        assert!(text.starts_with("Spot has no examples"), "{text}");
        let (text, tone) = readiness(&[row("Cell", 9, true), row("Spot", 6, true)], 2);
        assert_eq!(tone, 2);
        assert_eq!(text, "Ready: 2 classes, 15 examples from 2 images.");
        assert_eq!(examples_summary(0, false, 0), "none yet");
        assert_eq!(examples_summary(3, true, 94), "This image + 2 more · 94");
        assert_eq!(examples_summary(1, false, 4), "1 image · 4");
    }

    /// While the dialog is closed, changes don't touch its lists.
    #[test]
    fn a_closed_dialog_is_not_refreshed() {
        let d = opened(labeled_project());
        d.ui.global::<GlobalAppState>()
            .set_active_dialog(DialogType::None);
        d.ui_state
            .get_project_write()
            .add_object(&ObjectMetricSettings {
                id: ObjectId(100),
                object_class: [CELL].into(),
                ..Default::default()
            });
        d.state().invoke_project_changed();
        drain_ui_queue();
        assert_eq!(d.state().get_training_objects().row_count(), 6);
    }

    #[test]
    fn reopening_during_training_keeps_the_progress_banner() {
        let d = opened(labeled_project());
        d.state().set_training_in_progress(true);
        d.state().set_training_status("Fitting model...".into());
        d.state().invoke_open_requested();
        drain_ui_queue();
        assert_eq!(d.state().get_training_status(), "Fitting model...");
    }

    #[test]
    fn training_needs_a_model_name_and_a_saved_project() {
        let d = opened(labeled_project());
        let mut settings = d.state().get_settings();
        settings.model_name = "  ".into();
        d.state().set_settings(settings);
        d.train();
        assert!(d.state().get_training_status().contains("model name"));
        assert!(d.state().get_training_status_is_error());

        let mut settings = d.state().get_settings();
        settings.model_name = "m".into();
        d.state().set_settings(settings);
        d.train();
        assert!(d.state().get_training_status().contains("Save the project"));
    }

    fn set_up_object_training(d: &Dialog, dir: &std::path::Path) {
        d.ui_state.get_project_write().tmp_settings.current_project =
            Some(dir.join("study.evaproj"));
        let s = d.state();
        let mut settings = s.get_settings();
        settings.model_name = "cells".into();
        settings.mode = 1;
        settings.algorithm = 0;
        settings.rf_n_trees = 5;
        s.set_settings(settings);
        // Area only.
        let mut f = d.controller.features.lock().unwrap();
        f.object.shape = SHAPE_METRICS
            .iter()
            .map(|(name, _)| *name == "Area")
            .collect();
        for row in &mut f.object.brightness {
            row.fill(false);
        }
    }

    #[test]
    fn training_an_object_classifier_saves_it_and_it_can_be_loaded_back() {
        let d = opened(labeled_project());
        let dir = tempfile::tempdir().unwrap();
        set_up_object_training(&d, dir.path());
        d.train();
        let s = d.state();
        let status = s.get_training_status().to_string();
        assert!(status.starts_with("Training complete."), "{status}");
        assert!(status.contains("tree(s)"), "{status}");
        assert!(
            status.contains("No pipeline uses it yet"),
            "no step uses it yet: {status}"
        );
        assert!(!s.get_training_status_is_error());
        assert!(!s.get_training_in_progress());
        let model = dir.path().join("models").join(format!(
            "cells.{}",
            evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS
        ));
        assert!(model.exists(), "{model:?}");

        // Load it back: settings repopulate from the file.
        let mut settings = s.get_settings();
        settings.mode = 0;
        settings.model_name = "".into();
        s.set_settings(settings);
        s.invoke_apply_object_preset(2); // everything ticked
        drain_ui_queue();
        s.invoke_browse_existing_model_clicked();
        choose_file(&d.ui, &model);
        let settings = s.get_settings();
        assert_eq!(settings.mode, 1);
        assert_eq!(settings.model_name, "cells", "falls back to the file name");
        assert_eq!(settings.loaded_model_path, model.to_string_lossy().as_ref());
        assert!(s.get_training_status().starts_with("Loaded settings"));
        drain_ui_queue();
        // Back to what the model has: Area only.
        let shape: Vec<ToggleRowSlint> = s.get_shape_metrics().iter().collect();
        assert!(shape[0].selected, "Area");
        assert!(shape[1..].iter().all(|m| !m.selected));
        assert!(
            s.get_brightness_rows()
                .iter()
                .all(|r| r.cells.iter().all(|on| !on))
        );
    }

    /// After training, a model no step applies yet: "Add to pipeline" adds
    /// a pipeline with an AI object classifier step on that model - one
    /// undo step - and the offer goes away.
    #[test]
    fn add_to_pipeline_adds_a_pipeline_applying_the_trained_model() {
        let d = opened(labeled_project());
        let dir = tempfile::tempdir().unwrap();
        set_up_object_training(&d, dir.path());
        d.train();
        let s = d.state();
        drain_ui_queue();
        assert!(s.get_can_add_to_pipeline(), "{}", s.get_training_status());
        let before = d.ui_state.get_project().settings.pipelines.len();

        s.invoke_add_to_pipeline_clicked();
        drain_ui_queue();
        let project = d.ui_state.get_project();
        let pipelines = &project.settings.pipelines;
        assert_eq!(pipelines.len(), before + 1);
        let added = pipelines.last().unwrap();
        assert_eq!(added.name, "AI · cells");
        use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
        let Some(PipelineCommand::AiObjectClassifier(step)) = added.steps[0].command() else {
            panic!("an object classifier step")
        };
        assert!(step.model_path.ends_with(format!(
            "cells.{}",
            evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS
        )));
        assert!(evanalyzer_app::ai_learning::model_is_used(
            &project.settings,
            &step.model_path
        ));
        drop(project);
        assert!(!s.get_can_add_to_pipeline());
        assert!(
            s.get_training_status()
                .starts_with("Added the pipeline 'AI · cells'")
        );
    }

    #[test]
    fn loading_a_broken_model_reports_it() {
        let d = opened(labeled_project());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!(
            "bad.{}",
            evanalyzer_cfg::EVANALYZER_TRAINED_AI_MODELS
        ));
        std::fs::write(&path, "nope").unwrap();
        d.state().invoke_browse_existing_model_clicked();
        choose_file(&d.ui, &path);
        assert!(
            d.state()
                .get_training_status()
                .starts_with("Could not load")
        );
        assert!(d.state().get_training_status_is_error());
    }

    #[test]
    fn training_without_labels_for_the_chosen_classes_fails_with_a_message() {
        let d = opened(labeled_project());
        let dir = tempfile::tempdir().unwrap();
        set_up_object_training(&d, dir.path());
        let none: Vec<ClassSelectionRowSlint> = d
            .state()
            .get_class_selection()
            .iter()
            .map(|mut r| {
                r.selected = false;
                r
            })
            .collect();
        d.state()
            .set_class_selection(ModelRc::new(VecModel::from(none)));
        d.train();
        assert!(d.state().get_training_status_is_error());
        assert!(!d.state().get_training_in_progress());
    }

    #[test]
    fn cancel_training_cancels_the_running_job() {
        let d = opened(labeled_project());
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        *d.controller.training_cancel_flag.lock().unwrap() =
            Some(CancelHandle::with_callback(move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst)
            }));
        d.state().invoke_cancel_training_clicked();
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn build_ai_learning_settings_maps_the_forest_and_mlp_fields() {
        let project = ProjectSettings::default();
        let mut s = default_settings();
        s.algorithm = 0;
        s.rf_criterion = 1;
        s.rf_max_depth = 0;
        s.rf_max_features = 3;
        let built = build_ai_learning_settings(
            &s,
            Default::default(),
            Vec::new(),
            &Default::default(),
            &project,
        );
        let AiLearningBackendSettings::RandomForest(rf) = built.backend else {
            panic!("forest");
        };
        assert!(matches!(rf.criterion, SplitCriterion::Entropy));
        assert_eq!(rf.max_depth, None, "0 means unlimited");
        assert_eq!(rf.m, Some(3));

        s.algorithm = 2;
        s.mlp_hidden_layers = "8, 4".into();
        for (index, activation) in [
            (1, MlpActivation::Sigmoid),
            (2, MlpActivation::Tanh),
            (0, MlpActivation::Relu),
        ] {
            s.mlp_activation = index;
            s.mlp_epsilon = 0.0;
            let built = build_ai_learning_settings(
                &s,
                Default::default(),
                Vec::new(),
                &Default::default(),
                &project,
            );
            let AiLearningBackendSettings::Mlp(mlp) = built.backend else {
                panic!("mlp");
            };
            assert_eq!(mlp.activation, activation);
            assert_eq!(mlp.hidden_layers, [8, 4]);
            assert_eq!(mlp.epsilon, 1e-5, "invalid epsilon falls back");
            assert!(matches!(
                built.classifier,
                AiLearningClassifierSettings::Pixel { .. }
            ));
        }
    }
}
