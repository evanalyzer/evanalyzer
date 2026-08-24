use crate::ai_learning::model::{
    self, CURRENT_SAVED_CLASSIFIER_VERSION, Classifier, SavedClassifier,
};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningBackendSettings, AiLearningSettings,
};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
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
pub struct TrainingImage {
    pub path: PathBuf,
    pub series: i32,
    pub labeled_objects: Vec<ObjectMetricSettings>,
}

/// Progress reported by both `training::pixel::PixelTrainingJob::run` and
/// `training::object::ObjectTrainingJob::run`. The pixel job is the only one
/// that reads images tile-by-tile, hence the `Image*`/`Tile*` variants; the
/// object job (already-computed metrics, no image I/O) only ever reports
/// `Started`, `ItemCompleted`, `ObjectSkipped`, `Training` and `Finished`.
#[derive(Debug)]
pub enum TrainingProgressEvent {
    Started {
        total: usize,
    },
    ImageTilesScheduled {
        image_index: usize,
        total_tiles: usize,
    },
    TileProcessed {
        image_index: usize,
        tile_index: usize,
        total_tiles: usize,
    },
    ItemCompleted {
        index: usize,
        total: usize,
    },
    ImageFailed {
        path: PathBuf,
    },
    /// An object's `object_class` set matched zero or more than one of the
    /// model's configured `class_labels` - ambiguous, so it's excluded from
    /// training rather than guessed at.
    ObjectSkipped {
        index: usize,
        reason: String,
    },
    Training,
    /// One MLP training epoch finished. Only ever sent by the `Mlp` backend
    /// (`model::mlp::fit_mlp`) — `RandomForest`/`Knn` fit in one blocking
    /// smartcore call with no per-iteration hook to report from, so they
    /// only ever emit the surrounding `Training`/`Finished` events.
    ///
    /// Sending is throttled (see `fit_mlp`'s `report_every`) rather than sent
    /// for every epoch of a large `epochs` count, but the final epoch is
    /// always sent so the GUI's last-seen value matches `Finished`'s stats.
    Epoch {
        epoch: usize,
        total_epochs: usize,
        train_loss: f32,
        /// `None` when the dataset was too small for a held-out split (see
        /// `model::mlp::train_val_split`) - there's then no generalization
        /// signal, only the training-loss curve.
        val_loss: Option<f32>,
    },
    Finished {
        stats: TrainingStats,
    },
}

/// Backend-specific summary reported once, alongside `TrainingProgressEvent::Finished`,
/// for the GUI's post-training results banner.
#[derive(Debug)]
pub enum TrainingStats {
    RandomForest {
        n_trees: usize,
        n_samples: usize,
    },
    Knn {
        k: usize,
        n_samples: usize,
    },
    Mlp {
        /// Equal to `total_epochs` unless training was cancelled mid-run -
        /// cancellation surfaces as `InternalErrors::Cancelled` from `run()`
        /// though, so in practice this only ever reaches the GUI as
        /// `total_epochs`; kept distinct from it for when partial-result
        /// reporting on cancel is added.
        epochs_run: usize,
        total_epochs: usize,
        final_train_loss: f32,
        /// `None` when the dataset was too small for a held-out split.
        final_val_loss: Option<f32>,
        /// The lowest validation loss seen at any epoch, and which epoch it
        /// was at - lets the GUI point out overfitting concretely ("best
        /// epoch was 210 of 300; final validation loss is N% worse") instead
        /// of guessing at a threshold itself. `None` alongside `final_val_loss`.
        best_val_loss: Option<f32>,
        best_val_epoch: Option<usize>,
    },
}

/// Fits `rows`/`labels` (dense indices into the model's `class_labels`, with
/// `n_classes` the total count) using whichever backend `settings.backend`
/// selects. Shared by `PixelTrainingJob`/`ObjectTrainingJob` so both jobs
/// dispatch identically once their own feature extraction is done.
///
/// `progress`/`cancel` are only actually used by the `Mlp` backend (to emit
/// per-epoch `TrainingProgressEvent::Epoch` and to let the epoch loop bail
/// out early) - `RandomForest`/`Knn` fit in one blocking smartcore call with
/// no hook to report progress or interrupt mid-fit from, so they take them
/// only to keep this dispatch's signature uniform for its three callers.
pub(crate) fn fit_classifier(
    backend: &AiLearningBackendSettings,
    rows: &[Vec<f32>],
    labels: &[usize],
    n_classes: usize,
    progress: &Sender<TrainingProgressEvent>,
    cancel: &Arc<AtomicBool>,
) -> Result<(Classifier, TrainingStats), InternalErrors> {
    match backend {
        AiLearningBackendSettings::RandomForest(s) => {
            let classifier = model::random_forest::fit_random_forest(rows, labels, s)?;
            let stats = TrainingStats::RandomForest {
                n_trees: s.n_trees as usize,
                n_samples: rows.len(),
            };
            Ok((classifier, stats))
        }
        AiLearningBackendSettings::Knn(s) => {
            let classifier = model::knn::fit_knn(rows, labels, s)?;
            let stats = TrainingStats::Knn {
                k: s.k,
                n_samples: rows.len(),
            };
            Ok((classifier, stats))
        }
        AiLearningBackendSettings::Mlp(s) => {
            model::mlp::fit_mlp(rows, labels, n_classes, s, progress, cancel)
        }
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
pub(crate) fn spawn_training_job<J>(
    job: J,
    run: impl FnOnce(
        &J,
        Sender<TrainingProgressEvent>,
        Arc<AtomicBool>,
    ) -> Result<SavedClassifier, InternalErrors>
    + Send
    + 'static,
) -> (
    JoinHandle<Result<SavedClassifier, InternalErrors>>,
    Receiver<TrainingProgressEvent>,
    Arc<AtomicBool>,
)
where
    J: Send + 'static,
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

    fn two_cluster_dataset() -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![0.0 + jitter, 0.0 + jitter]);
            labels.push(0);
            rows.push(vec![10.0 + jitter, 10.0 + jitter]);
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
            fit_classifier(&backend, &rows, &labels, 2, &progress, &cancel).unwrap();
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
            fit_classifier(&backend, &rows, &labels, 2, &progress, &cancel).unwrap();
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
            fit_classifier(&backend, &rows, &labels, 2, &progress, &cancel).unwrap();
        assert!(matches!(classifier, Classifier::Mlp { .. }));
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
            fit_classifier(&backend, &rows, &labels, 2, &progress, &cancel).unwrap();
        let settings = AiLearningSettings {
            schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: MetaData {
                name: "test".into(),
                ..Default::default()
            },
            backend,
            classifier: AiLearningClassifierSettings::Object {
                feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
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
            let (rows, labels) = ([vec![0.0], vec![1.0]], [0usize, 1]);
            let (classifier, stats) = fit_classifier(
                &AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                &rows,
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
                        feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
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
