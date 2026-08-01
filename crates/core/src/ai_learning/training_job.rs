use crate::ai_learning::model::{self, CURRENT_SAVED_CLASSIFIER_VERSION, Classifier, SavedClassifier};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{AiLearningBackendSettings, AiLearningSettings};
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
    Finished,
}

/// Fits `rows`/`labels` (dense indices into the model's `class_labels`, with
/// `n_classes` the total count) using whichever backend `settings.backend`
/// selects. Shared by `PixelTrainingJob`/`ObjectTrainingJob` so both jobs
/// dispatch identically once their own feature extraction is done.
pub(crate) fn fit_classifier(
    backend: &AiLearningBackendSettings,
    rows: &[Vec<f32>],
    labels: &[usize],
    n_classes: usize,
) -> Result<Classifier, InternalErrors> {
    match backend {
        AiLearningBackendSettings::RandomForest(s) => {
            model::random_forest::fit_random_forest(rows, labels, s)
        }
        AiLearningBackendSettings::Knn(s) => model::knn::fit_knn(rows, labels, s),
        AiLearningBackendSettings::Mlp(s) => model::mlp::fit_mlp(rows, labels, n_classes, s),
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
    run: impl FnOnce(&J, Sender<TrainingProgressEvent>, Arc<AtomicBool>) -> Result<SavedClassifier, InternalErrors>
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
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningClassifierSettings, KnnSettings, MlpSettings, RandomForestSettings,
    };
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
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

    #[test]
    fn fit_classifier_dispatches_to_random_forest() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::RandomForest(RandomForestSettings::default());
        let classifier = fit_classifier(&backend, &rows, &labels, 2).unwrap();
        assert!(matches!(classifier, Classifier::RandomForest(_)));
    }

    #[test]
    fn fit_classifier_dispatches_to_knn() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::Knn(KnnSettings {
            k: 3,
            ..Default::default()
        });
        let classifier = fit_classifier(&backend, &rows, &labels, 2).unwrap();
        assert!(matches!(classifier, Classifier::Knn(_)));
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
        let classifier = fit_classifier(&backend, &rows, &labels, 2).unwrap();
        assert!(matches!(classifier, Classifier::Mlp { .. }));
    }

    #[test]
    fn finish_stamps_the_current_version_and_carries_settings_through() {
        let (rows, labels) = two_cluster_dataset();
        let backend = AiLearningBackendSettings::RandomForest(RandomForestSettings::default());
        let classifier = fit_classifier(&backend, &rows, &labels, 2).unwrap();
        let settings = AiLearningSettings {
            metadata: MetaData {
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
        assert_eq!(saved.settings.metadata.name, "test");
    }

    #[test]
    fn spawn_training_job_runs_the_closure_and_streams_progress() {
        let (handle, rx, _cancel) = spawn_training_job(42u32, |job, progress, _cancel| {
            let _ = progress.send(TrainingProgressEvent::Started { total: *job as usize });
            let (rows, labels) = ([vec![0.0], vec![1.0]], [0usize, 1]);
            let classifier = fit_classifier(
                &AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                &rows,
                &labels,
                2,
            )?;
            let _ = progress.send(TrainingProgressEvent::Finished);
            Ok(finish(
                AiLearningSettings {
                    metadata: MetaData::default(),
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
        assert!(matches!(events[0], TrainingProgressEvent::Started { total: 42 }));
        assert!(matches!(events.last(), Some(TrainingProgressEvent::Finished)));

        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }
}
