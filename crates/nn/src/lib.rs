//! # evanalyzer_nn
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Every machine-learning algorithm of EVAnalyzer, behind plain-data
//! interfaces (slices in, `Vec`s out - no burn or smartcore types):
//!
//! - [`model`] - neural networks run on an image ([`Model`]): U-Net,
//!   StarDist, Cellpose-SAM; their training ([`Trainable`]) later.
//! - [`classifier`] - trained classifiers on feature rows ([`Classifier`]):
//!   MLP, KNN, random forest. Object and pixel classifiers are the same
//!   here; where the rows come from is core's.
//! - [`train`] - what every training shares: progress, cancel, seed.
//! - [`runtime`] - backend and device, chosen once per process on the first
//!   model run; nothing is set up before.
//! - [`pieces`] - large tiles through a network in pieces.
//! - [`weights`] - the model file format (safetensors + metadata).
//!
//! The only crate depending on burn and smartcore. Does not depend on
//! `evanalyzer_cfg`: core maps its settings to the types here.

pub mod cache;
pub mod classifier;
pub mod errors;
pub mod model;
pub mod pieces;
pub mod runtime;
pub mod train;
pub mod weights;

pub use classifier::{Classifier, Predictions, Samples, TrainedClassifier};
pub use errors::NnError;
pub use model::{Image, Model, Trainable};
pub use pieces::Pieces;
pub use runtime::{DeviceSelection, EngineConfig, EngineInfo};
pub use train::{TrainControl, TrainProgress, TrainStats};

use std::path::Path;

/// Runs model `M` from `file` on `image`. The engine is set up with
/// `config` on the first call (later calls ignore it); the model is loaded
/// once per device and kept until [`release_models`] (or the file changes).
/// Blocks while every device is busy.
pub fn infer<M: Model>(
    file: &Path,
    image: Image<'_>,
    settings: &M::Settings,
    config: impl FnOnce() -> EngineConfig,
) -> Result<M::Output, NnError> {
    let _ = (file, image, settings, config);
    todo!("P3: runtime + cache")
}

/// Lets go of every loaded model on every device (they load again on their
/// next run). Nothing to do when no model ran yet.
pub fn release_models() {
    todo!("P3: cache")
}
