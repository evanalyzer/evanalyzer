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
//! - [`model`] - neural networks run on an image ([`Model`]: pre-processing,
//!   the network, post-processing): U-Net, StarDist, Cellpose-SAM; their
//!   training ([`Trainable`]) later.
//! - [`classifier`] - trained classifiers on feature rows ([`Classifier`]):
//!   MLP, KNN, random forest. Object and pixel classifiers are the same
//!   here; where the rows come from is core's.
//! - [`train`] - what every training shares: progress, cancel, seed.
//! - [`runtime`] - backend and device, chosen once per process on the first
//!   model run; nothing is set up before.
//! - [`pieces`] - large tiles through a network in pieces.
//! - [`weights`] - the model file format (safetensors + metadata).
//! - `libs` - building blocks several models share (percentiles).
//!
//! The only crate depending on burn and smartcore. Does not depend on
//! `evanalyzer_cfg`: core maps its settings to the types here.

pub mod classifier;
pub mod errors;
mod libs;
pub mod model;
pub mod pieces;
pub mod runtime;
pub mod test_support;
pub mod train;
pub mod weights;

/// burn's device (`Device::flex()` = the CPU) - users of nn need no burn.
pub use burn::tensor::Device;
pub use classifier::{Classifier, Predictions, Samples, TrainedClassifier};
pub use errors::NnError;
pub use model::{Image, Model, Normalization, Trainable, Written};
pub use pieces::Pieces;
pub use runtime::{
    DeviceInfo, DeviceSelection, EngineConfig, EngineInfo, GpuInfo, configure_gpu, gpus,
};
pub use train::{TrainControl, TrainProgress, TrainStats};

use std::path::Path;
use std::sync::OnceLock;

/// The engine, set up on the first model run (`runtime`).
static ENGINE: OnceLock<runtime::Engine> = OnceLock::new();

fn engine(config: impl FnOnce() -> EngineConfig) -> &'static runtime::Engine {
    ENGINE.get_or_init(|| runtime::Engine::new(&config()))
}

/// Runs model `M` from `file` on `image` - pre-processing, the network in
/// an engine slot, post-processing into `out` (one value per pixel; see
/// [`Model`]). The engine is set up with `config` on the first call (later
/// calls ignore it); the model is loaded once per device and kept until
/// [`release_models`] (or the file changes). Blocks while every device is
/// busy.
pub fn run<M: Model>(
    file: &Path,
    image: Image<'_>,
    params: &M::Params,
    out: &mut [u32],
    config: impl FnOnce() -> EngineConfig,
) -> Result<Written, NnError> {
    engine(config).run::<M>(file, image, params, out)
}

/// What the engine runs on - `None` before the first model run.
pub fn engine_info() -> Option<EngineInfo> {
    ENGINE.get().map(runtime::Engine::info)
}

/// Lets go of every loaded model on every device (they load again on their
/// next run). Nothing to do when no model ran yet.
pub fn release_models() {
    if let Some(engine) = ENGINE.get() {
        engine.release_models();
    }
}
