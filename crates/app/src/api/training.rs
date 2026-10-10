//! Classifier training as front ends see it: the request parameters and the
//! handle to a running training (events, cancel, trained model).

use super::CancelHandle;
use evanalyzer_cfg::core_types::{InternalErrors, TrainingProgressEvent};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_core::SavedClassifier;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;

/// Extra parameters `PixelTrainingJob` needs that aren't part of the portable
/// `AiLearningSettings` model descriptor - object training needs none of
/// these, since it reads no images (see `ObjectTrainingJob`'s doc comment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelTrainingParams {
    pub channel: i32,
    pub t_stack: i32,
    pub z_stack_handling: ZStackHandling,
}

impl Default for PixelTrainingParams {
    fn default() -> Self {
        Self {
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandling::SingleStack,
        }
    }
}

/// What a training run learns from: labeled images (pixel classifier) or
/// labeled objects (object classifier), with how many of them were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingItems {
    Images(usize),
    Objects(usize),
}

impl TrainingItems {
    pub fn count(&self) -> usize {
        match self {
            TrainingItems::Images(n) | TrainingItems::Objects(n) => *n,
        }
    }
}

#[derive(Debug)]
pub enum StartTrainingError {
    NoTrainingData,
    Failed(InternalErrors),
}

impl std::fmt::Display for StartTrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartTrainingError::NoTrainingData => write!(
                f,
                "No labeled training data found - assign a class to at least one object before training."
            ),
            StartTrainingError::Failed(internal_errors) => write!(f, "{internal_errors}"),
        }
    }
}

impl std::error::Error for StartTrainingError {}

/// A trained classifier, ready for [`save_trained_model`]. Opaque so front
/// ends never depend on `evanalyzer_core`'s model representation.
pub struct TrainedClassifier(pub(crate) SavedClassifier);

impl TrainedClassifier {
    /// Serialized form for sending a model trained elsewhere (e.g. on a
    /// server) - the same JSON the model file stores.
    pub fn to_bytes(&self) -> Result<Vec<u8>, InternalErrors> {
        serde_json::to_vec(&self.0)
            .map_err(|e| InternalErrors::Internal(format!("failed to serialize classifier: {e}")))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InternalErrors> {
        serde_json::from_slice(bytes)
            .map(TrainedClassifier)
            .map_err(|e| InternalErrors::Internal(format!("failed to read classifier: {e}")))
    }
}

/// Blocks until a training run is over and yields the model - a local run
/// joins its thread, a remote one waits for the server's final message.
pub type TrainingCompletion = Box<dyn FnOnce() -> Result<TrainedClassifier, InternalErrors> + Send>;

/// A classifier training run, wherever it runs. Drain
/// [`events`](Self::events) until it closes, then call [`wait`](Self::wait)
/// for the trained model - saving it is up to the caller
/// ([`save_trained_model`]).
pub struct RunningTraining {
    events: Receiver<TrainingProgressEvent>,
    cancel: CancelHandle,
    items: TrainingItems,
    completion: TrainingCompletion,
}

impl RunningTraining {
    /// Assembles a training run executed by some other backend (e.g. on a
    /// server). The `events` channel must close once training is over.
    pub fn from_parts(
        events: Receiver<TrainingProgressEvent>,
        cancel: CancelHandle,
        items: TrainingItems,
        completion: TrainingCompletion,
    ) -> Self {
        Self {
            events,
            cancel,
            items,
            completion,
        }
    }

    /// Progress events, in order. The channel closes once the training
    /// thread exits, so `for event in training.events()` ends by itself.
    pub fn events(&self) -> &Receiver<TrainingProgressEvent> {
        &self.events
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        self.cancel.clone()
    }

    pub fn items(&self) -> TrainingItems {
        self.items
    }

    /// Blocks until training finishes. A panic in a local training thread
    /// is returned as `InternalErrors::Internal`, a cancel as
    /// `InternalErrors::Cancelled`.
    pub fn wait(self) -> Result<TrainedClassifier, InternalErrors> {
        (self.completion)()
    }
}
