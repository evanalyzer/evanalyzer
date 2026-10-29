//! # evanalyzer_ai
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Runs AI models (Cellpose, StarDist, U-Net, YOLOv5) on an
//! [`InferenceEngine`], set up on the first model run.
//!
//! - [`api`] - what the analysis sees: [`Request`]s in, [`Response`]s out,
//!   no framework types; the C interface of an engine plugin.
//! - [`torch`] (feature `torch`) - the libtorch engine. The app does not
//!   link it: it is this crate built as a cdylib (`cargo build-ai-plugin`)
//!   and loaded on the first model run.
//! - `runtime` - the plugin loader (feature `torch-plugin`) and the CUDA
//!   driver probe.
//!
//! Which engine runs the models is decided once per process (see
//! [`infer`]): `torch` linked in (tests, and inside the plugin itself) >
//! the plugin (`torch-plugin`, the app) > none - every model run then says
//! the build has no AI model support. Another framework (e.g. Burn) is
//! another module with an [`InferenceEngine`] and one more arm in `start`.

pub mod api;
mod runtime;
#[cfg(feature = "torch")]
pub mod torch;

pub use api::{
    Detection, DeviceSelection, EngineConfig, Image, InferenceEngine, Request, Response, Task,
    box_iou,
};
use std::sync::OnceLock;

static ENGINE: OnceLock<Result<Box<dyn InferenceEngine>, String>> = OnceLock::new();

/// Runs `request` on `image`, setting up the engine with `config` on the
/// first call (later calls ignore it). Blocks while every device is busy.
/// Errors are the messages shown to the user.
pub fn infer(
    request: &Request,
    image: Image<'_>,
    config: impl FnOnce() -> EngineConfig,
) -> Result<Response, String> {
    let engine = ENGINE
        .get_or_init(|| {
            let engine = start(&config());
            match &engine {
                Ok(engine) => {
                    let info = engine.info();
                    log::info!("AI engine: {} on {:?}", info.name, info.devices);
                }
                Err(e) => log::warn!("No AI engine: {e}"),
            }
            engine
        })
        .as_ref()
        .map_err(Clone::clone)?;
    engine.infer(request, image)
}

/// The AI plugin library to try first (`--ai-plugin`; for development and
/// tests - a package finds its own). Once, before the first model run;
/// `false` if one was already chosen or this build loads no plugin.
#[allow(unused_variables)]
pub fn set_plugin_path(path: std::path::PathBuf) -> bool {
    #[cfg(feature = "torch-plugin")]
    return runtime::plugin::set_path(path);
    #[allow(unreachable_code)]
    false
}

/// Lets go of every AI model loaded so far - each takes hundreds of MB up
/// to GBs, on every device, and stayed until the app ended. The next run
/// loads them again. Nothing to do when no model ran yet.
pub fn release_models() {
    if let Some(Ok(engine)) = ENGINE.get() {
        engine.release_models();
    }
}

/// Whether a CUDA GPU can run models: the NVIDIA driver finds one new enough
/// for this build, and the build can use it. Slow (loads the driver) - not
/// for a UI startup path.
pub fn cuda_is_available() -> bool {
    if runtime::cuda_probe::usable_gpus().is_empty() {
        return false;
    }
    #[cfg(feature = "torch")]
    return true;
    #[cfg(all(feature = "torch-plugin", not(feature = "torch")))]
    return runtime::plugin::cuda_plugin_installed();
    #[allow(unreachable_code)]
    false
}

#[allow(unused_variables)]
fn start(config: &EngineConfig) -> Result<Box<dyn InferenceEngine>, String> {
    #[cfg(feature = "torch")]
    return Ok(Box::new(torch::TorchEngine::new(config)));
    #[cfg(all(feature = "torch-plugin", not(feature = "torch")))]
    return runtime::plugin::load(config);
    #[allow(unreachable_code)]
    Err("This build has no AI model support (it was built without the `libtorch` feature).".into())
}

#[cfg(all(test, not(any(feature = "torch", feature = "torch-plugin"))))]
mod tests {
    use super::*;

    #[test]
    fn a_build_without_an_engine_says_so() {
        let request = Request {
            model: "model.pt".into(),
            task: Task::StarDist,
        };
        let image = Image {
            width: 1,
            height: 1,
            channels: 1,
            data: &[0.0],
        };
        let config = || EngineConfig {
            devices: DeviceSelection::Cpu,
            gpu_slots: 1,
            cores: 1,
        };
        let err = infer(&request, image, config).unwrap_err();
        assert!(err.contains("no AI model support"), "{err}");
    }
}
