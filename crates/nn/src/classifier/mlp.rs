//! # mlp
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Multi-layer perceptron (burn). Weights as safetensors with our own tensor
//! names (`layer{i}.weight`, `layer{i}.bias`), never burn's record formats -
//! a burn upgrade can't break saved files. P2: moved from core's
//! `ai_learning/model/mlp.rs`.

use crate::classifier::{Classifier, Predictions, Samples};
use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};

/// The activation between hidden layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlpActivation {
    #[default]
    Relu,
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlpParams {
    /// Neurons per hidden layer.
    pub hidden_layers: Vec<usize>,
    pub activation: MlpActivation,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    /// Adam's epsilon.
    pub epsilon: f64,
}

pub struct Mlp {
    // P2: layer sizes + the burn module.
}

impl Classifier for Mlp {
    type Params = MlpParams;

    fn train(
        samples: &Samples<'_>,
        params: &MlpParams,
        control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError> {
        let _ = (samples, params, control);
        todo!("P2")
    }

    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        let _ = (rows, n_features);
        todo!("P2")
    }
}
