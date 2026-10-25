//! # torch
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! AI inference with libtorch (feature `torch`) - the only module that
//! links it. Built into the plugin library the app loads on the first AI
//! step (see [`ffi`]), so the ~280 MB and the half second libtorch needs at
//! startup are only paid when a model runs. Built once against the CPU
//! libtorch and once against the CUDA one; the app picks the plugin that
//! fits the computer.
//!
//! [`TorchEngine`] is the [`InferenceEngine`] behind it: every model kind
//! of [`Task`] turned into tensor work, queued on the configured devices.

mod cellpose;
pub mod ffi;
mod scheduler;
mod stardist;
mod unet;
mod yolov5;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

use crate::api::{EngineConfig, EngineInfo, Image, InferenceEngine, Request, Response, Task};
use scheduler::TorchScheduler;

/// Errors are the messages shown to the user.
type Result<T> = std::result::Result<T, String>;

/// Runs the models of [`Task`] with libtorch.
pub struct TorchEngine {
    scheduler: TorchScheduler,
}

impl TorchEngine {
    /// Picks the devices `config` asks for (the CPU when there is no GPU).
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            scheduler: TorchScheduler::from_config(config),
        }
    }
}

impl InferenceEngine for TorchEngine {
    fn info(&self) -> EngineInfo {
        let cuda = if tch::Cuda::is_available() {
            "CUDA"
        } else {
            "CPU"
        };
        EngineInfo {
            name: format!("libtorch ({cuda})"),
            devices: self.scheduler.devices(),
        }
    }

    fn infer(&self, request: &Request, image: Image<'_>) -> Result<Response> {
        let expected = image.width * image.height * image.channels;
        if image.data.len() < expected || image.channels == 0 {
            return Err(format!(
                "AI inference: the image has {} values, {expected} expected",
                image.data.len()
            ));
        }
        let scheduler = &self.scheduler;
        let model = request.model.as_path();
        match request.task {
            Task::UNet {
                foreground_channel,
                softmax,
                boundary_channel,
            } => unet::run(
                scheduler,
                model,
                foreground_channel,
                softmax,
                boundary_channel,
                image,
            ),
            Task::Cellpose {
                net_width,
                net_height,
                input_channels,
                replicate_gray_channel,
            } => cellpose::run(
                scheduler,
                model,
                net_width,
                net_height,
                input_channels,
                replicate_gray_channel,
                image,
            ),
            Task::StarDist => stardist::run(scheduler, model, image),
            Task::Yolov5 {
                width,
                height,
                window_overlap,
                confidence_threshold,
                iou_threshold,
                mask_threshold,
            } => yolov5::run(
                scheduler,
                model,
                width,
                height,
                window_overlap,
                confidence_threshold,
                iou_threshold,
                mask_threshold,
                image,
            ),
        }
    }
}
