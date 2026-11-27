//! # train
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Training a SpotNet from clicked spots and training areas.

use crate::{
    NnError, TrainControl, TrainStats,
    model::{
        Dataset,
        spotnet::network::{SpotNet, SpotNetConfig},
    },
};
use burn::Tensor;

/// One clicked spot: its centre in px (a pixel's centre is +0.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spot {
    pub x: f32,
    pub y: f32,
}

/// What the user drew on one training image.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotLabels {
    pub width: usize,
    pub height: usize,
    pub spots: Vec<Spot>,
    /// Per pixel: inside a training area. Spots outside are ignored.
    pub area: Vec<bool>,
}

/// Settings of one training.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotNetTrainParams {
    pub epochs: usize,
    /// Patches per epoch (an image gives many patches).
    pub patches_per_epoch: usize,
    pub batch_size: usize,
    /// Side of a square patch in px, e.g. 128.
    pub patch_size: usize,
    pub learning_rate: f64,
    /// Width of the Gaussian hills in px, e.g. 1.5.
    pub sigma: f32,
}

/// Trains a new SpotNet on `data`.
pub fn train(
    data: &dyn Dataset<SpotLabels>,
    config: &SpotNetConfig,
    params: &SpotNetTrainParams,
    control: &TrainControl<'_>,
) -> Result<(SpotNet, TrainStats), NnError> {
    // 1. Prepare:   device = autodiff CPU, model = SpotNet::new(config, &device),
    //               optimizer = Adam.
    // 2. Per epoch:
    //    a. pick `patches_per_epoch` random patches inside the training areas
    //    b. per batch of `batch_size` patches:
    //         image patch → input tensor  [B, in_channels, P, P]
    //         gaussian_map patch → target [B, 1, P, P]
    //         area_mask patch → mask      [B, 1, P, P]
    //         prediction = model.forward(input)
    //         loss = masked_loss(prediction, target, mask)
    //         grads = loss.backward() → optimizer step
    //    c. report the epoch's loss via control.progress, check control.is_cancelled()
    // 3. Return model.valid() (the model without autodiff) and the stats.
    todo!()
}

/// How far the prediction is from the perfect answer, only where mask = 1.
fn masked_loss(prediction: Tensor<4>, target: Tensor<4>, mask: Tensor<4>) -> Tensor<1> {
    todo!()
}
