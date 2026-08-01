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
