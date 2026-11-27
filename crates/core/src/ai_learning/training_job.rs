//! # training_job
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-07-31
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ai_learning::model::{
    self, CURRENT_SAVED_CLASSIFIER_VERSION, Classifier, SavedClassifier,
};
use evanalyzer_cfg::core_types::{InternalErrors, TrainingProgressEvent, TrainingStats};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningSettings, KNNAlgorithmName, KNNDistanceMetric,
    KNNWeightFunction, MlpActivation, SplitCriterion,
};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_nn::classifier::knn::{KnnDistance, KnnSearch, KnnWeight};
use evanalyzer_nn::classifier::mlp::MlpActivation as NnMlpActivation;
use evanalyzer_nn::classifier::rtree::SplitCriterion as NnSplitCriterion;
use evanalyzer_nn::classifier::{
    Classifier as _, Knn, KnnParams, Mlp, MlpParams, RandomForest, RandomForestParams, Samples,
};
use evanalyzer_nn::train::{TrainControl, TrainDevice, TrainProgress, TrainStats};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// One image contributing labeled training samples to a `training::pixel::PixelTrainingJob`.
///
/// `labeled_objects` pairs each manually-painted, labeled Object (the settings
/// DTO already held by the project/GUI layer - not core's internal `Object`,
/// which stays private to this crate) with its resolved training class. This
/// job converts each one to an `Object` internally (to reuse its bbox/mask)
/// and walks only the pixels inside its mask - passing already-flattened
/// per-pixel coordinates instead would balloon memory for large painted
/// regions (a compact bbox+bitmask is far smaller than one tuple per masked
/// pixel).
#[derive(Debug, Clone)]
pub struct TrainingImage {
    pub path: PathBuf,
    pub series: i32,
    pub labeled_objects: Vec<ObjectMetricSettings>,
}

/// Fits `rows` (row-major, `n_features` values per sample) and `labels`
/// (dense indices into the model's `class_labels`, `n_classes` the total
/// count) with whichever classifier `backend` selects (`evanalyzer_nn`).
/// Shared by `PixelTrainingJob`/`ObjectTrainingJob`.
///
/// `progress`/`cancel` are used by the MLP (an epoch event, a check per
/// epoch); the random forest and KNN fit in one blocking call.
pub fn fit_classifier(
    backend: &AiLearningBackendSettings,
    rows: &[f32],
    n_features: usize,
    labels: &[u32],
    n_classes: usize,
    progress: &Sender<TrainingProgressEvent>,
    cancel: &Arc<AtomicBool>,
) -> Result<(Classifier, TrainingStats), InternalErrors> {
    // As before the move: the message the training panel shows.
    if labels.is_empty() {
        return Err(InternalErrors::Internal(
            "cannot train on zero samples".to_string(),
        ));
    }
    let samples = Samples {
        rows,
        n_features,
        labels,
        n_classes,
    };
    let report = |p: TrainProgress| {
        if let TrainProgress::Epoch {
            epoch,
            total_epochs,
            train_loss,
            val_loss,
            ..
        } = p
        {
            let _ = progress.send(TrainingProgressEvent::Epoch {
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                held_back: None,
            });
        }
    };
    let control = |seed: u64| TrainControl {
        progress: &report,
        cancel,
        seed,
        device: TrainDevice::Cpu,
        cores: rayon::current_num_threads(),
    };
    let (classifier, stats) = match backend {
        AiLearningBackendSettings::RandomForest(s) => {
            let params = RandomForestParams {
                criterion: match s.criterion {
                    SplitCriterion::Gini => NnSplitCriterion::Gini,
                    SplitCriterion::Entropy => NnSplitCriterion::Entropy,
                    SplitCriterion::ClassificationError => NnSplitCriterion::ClassificationError,
                },
                max_depth: s.max_depth,
                min_samples_leaf: s.min_samples_leaf,
                min_samples_split: s.min_samples_split,
                n_trees: s.n_trees,
                m: s.m,
            };
            let (model, stats) = RandomForest::train(&samples, &params, &control(s.seed))
                .map_err(model::nn_error)?;
            (Classifier::RandomForest(model), stats)
        }
        AiLearningBackendSettings::Knn(s) => {
            let params = KnnParams {
                search: match s.algorithm {
                    KNNAlgorithmName::LinearSearch => KnnSearch::Linear,
                    KNNAlgorithmName::CoverTree => KnnSearch::CoverTree,
                },
                weight: match s.weight {
                    KNNWeightFunction::Uniform => KnnWeight::Uniform,
                    KNNWeightFunction::Distance => KnnWeight::Distance,
                },
                k: s.k,
                distance: match s.distance {
                    KNNDistanceMetric::Euclidean => KnnDistance::Euclidean,
                    KNNDistanceMetric::Manhattan => KnnDistance::Manhattan,
                    KNNDistanceMetric::Cosine => KnnDistance::Cosine,
                    KNNDistanceMetric::Hamming => KnnDistance::Hamming,
                    KNNDistanceMetric::Minkowski { p } => KnnDistance::Minkowski { p },
                },
            };
            let (model, stats) =
                Knn::train(&samples, &params, &control(0)).map_err(model::nn_error)?;
            (Classifier::Knn(model), stats)
        }
        AiLearningBackendSettings::Mlp(s) => {
            let params = MlpParams {
                hidden_layers: s.hidden_layers.clone(),
                activation: match s.activation {
                    MlpActivation::Relu => NnMlpActivation::Relu,
                    MlpActivation::Sigmoid => NnMlpActivation::Sigmoid,
                    MlpActivation::Tanh => NnMlpActivation::Tanh,
                },
                epochs: s.epochs,
                learning_rate: s.learning_rate,
                batch_size: s.batch_size,
                epsilon: s.epsilon,
            };
            let (model, stats) =
                Mlp::train(&samples, &params, &control(s.seed)).map_err(model::nn_error)?;
            (Classifier::Mlp(model), stats)
        }
    };
    Ok((classifier, training_stats(stats)))
}

/// nn's training statistics as the progress events carry them.
fn training_stats(stats: TrainStats) -> TrainingStats {
    match stats {
        TrainStats::RandomForest { n_trees, n_samples } => {
            TrainingStats::RandomForest { n_trees, n_samples }
        }
        TrainStats::Knn { k, n_samples } => TrainingStats::Knn { k, n_samples },
        TrainStats::Epochs {
            epochs_run,
            total_epochs,
            final_train_loss,
            final_val_loss,
            best_val_loss,
            best_val_epoch,
        } => TrainingStats::Mlp {
            epochs_run,
            total_epochs,
            final_train_loss,
            final_val_loss,
            best_val_loss,
            best_val_epoch,
        },
    }
}

pub(crate) fn finish(settings: AiLearningSettings, classifier: Classifier) -> SavedClassifier {
    SavedClassifier {
        version: CURRENT_SAVED_CLASSIFIER_VERSION,
        classifier,
        settings,
    }
}

/// Runs `run` on a background thread, mirroring `JobExecutor::run_async`'s
/// exact shape (progress channel + shared cancel flag) so the GUI can wire
/// training up the same way it already wires up pipeline execution. Shared by
/// `PixelTrainingJob::run_async`/`ObjectTrainingJob::run_async` to dedupe the
/// channel/flag/thread-spawn boilerplate between them.
pub(crate) fn spawn_training_job<J, R>(
    job: J,
    run: impl FnOnce(&J, Sender<TrainingProgressEvent>, Arc<AtomicBool>) -> Result<R, InternalErrors>
    + Send
    + 'static,
) -> (
    JoinHandle<Result<R, InternalErrors>>,
    Receiver<TrainingProgressEvent>,
    Arc<AtomicBool>,
)
where
    J: Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);
    let handle = std::thread::spawn(move || run(&job, tx, cancel_clone));
    (handle, rx, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningClassifierSettings, KnnSettings, MlpSettings, RandomForestSettings,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;

    fn two_cluster_dataset() -> (Vec<f32>, Vec<u32>) {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.extend([jitter, jitter]);
            labels.push(0);
            rows.extend([10.0 + jitter, 10.0 + jitter]);
            labels.push(1);
        }
        (rows, labels)
    }

    fn no_op_progress() -> (Sender<TrainingProgressEvent>, Arc<AtomicBool>) {
        (
            std::sync::mpsc::channel().0,
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn fit_classifier_dispatches_to_random_forest() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::RandomForest(RandomForestSettings::default());
        let (progress, cancel) = no_op_progress();
        let (classifier, stats) =
            fit_classifier(&backend, &rows, 2, &labels, 2, &progress, &cancel).unwrap();
        assert!(matches!(classifier, Classifier::RandomForest(_)));
        assert!(matches!(
            stats,
            TrainingStats::RandomForest { n_samples: 30, .. }
        ));
    }

    #[test]
    fn fit_classifier_dispatches_to_knn() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::Knn(KnnSettings {
            k: 3,
            ..Default::default()
        });
        let (progress, cancel) = no_op_progress();
        let (classifier, stats) =
            fit_classifier(&backend, &rows, 2, &labels, 2, &progress, &cancel).unwrap();
        assert!(matches!(classifier, Classifier::Knn(_)));
        assert!(matches!(stats, TrainingStats::Knn { k: 3, .. }));
    }

    #[test]
    fn fit_classifier_dispatches_to_mlp() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::Mlp(MlpSettings {
            hidden_layers: vec![4],
            epochs: 5,
            learning_rate: 0.01,
            ..Default::default()
        });
        let (progress, cancel) = no_op_progress();
        let (classifier, stats) =
            fit_classifier(&backend, &rows, 2, &labels, 2, &progress, &cancel).unwrap();
        assert!(matches!(classifier, Classifier::Mlp(_)));
        assert!(matches!(
            stats,
            TrainingStats::Mlp {
                total_epochs: 5,
                ..
            }
        ));
    }

    #[test]
    fn finish_stamps_the_current_version_and_carries_settings_through() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::RandomForest(RandomForestSettings::default());
        let (progress, cancel) = no_op_progress();
        let (classifier, _stats) =
            fit_classifier(&backend, &rows, 2, &labels, 2, &progress, &cancel).unwrap();
        let settings = AiLearningSettings {
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: MetaData {
                name: "test".into(),
                ..Default::default()
            },
            backend,
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings::default(),
                class_labels: vec![],
            },
        };

        let saved = finish(settings, classifier);

        assert_eq!(saved.version, CURRENT_SAVED_CLASSIFIER_VERSION);
        assert_eq!(saved.settings.meta.name, "test");
    }

    #[test]
    fn spawn_training_job_runs_the_closure_and_streams_progress() {
        let (handle, rx, _cancel) = spawn_training_job(42u32, |job, progress, cancel| {
            let _ = progress.send(TrainingProgressEvent::Started {
                total: *job as usize,
            });
            let (rows, labels) = ([0.0, 1.0], [0u32, 1]);
            let (classifier, stats) = fit_classifier(
                &AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                &rows,
                1,
                &labels,
                2,
                &progress,
                &cancel,
            )?;
            let _ = progress.send(TrainingProgressEvent::Finished { stats });
            Ok(finish(
                AiLearningSettings {
                    schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                    meta: MetaData::default(),
                    backend: AiLearningBackendSettings::RandomForest(
                        RandomForestSettings::default(),
                    ),
                    classifier: AiLearningClassifierSettings::Object {
                        feature_spec: AiLearningObjectFeatureSettings::default(),
                        class_labels: vec![],
                    },
                },
                classifier,
            ))
        });

        let events: Vec<TrainingProgressEvent> = rx.iter().collect();
        assert!(matches!(
            events[0],
            TrainingProgressEvent::Started { total: 42 }
        ));
        assert!(matches!(
            events.last(),
            Some(TrainingProgressEvent::Finished { .. })
        ));

        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }
}
