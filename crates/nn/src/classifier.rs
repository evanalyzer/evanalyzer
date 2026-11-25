//! # classifier
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Trained classifiers on feature rows: MLP, KNN, random forest. Object and
//! pixel classifiers are the same here - feature rows + labels. Where the
//! rows come from (object measurements, filter responses at annotated
//! pixels) needs images, objects and filters, and is core's.
//!
//! They run on the CPU, outside the model cache (core keeps one per run).

use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};

pub mod knn;
pub mod mlp;
pub mod rtree;

pub use knn::{Knn, KnnParams};
pub use mlp::{Mlp, MlpParams};
pub use rtree::{RandomForest, RandomForestParams};

/// Training samples: `n_samples` rows of `n_features` values, one block,
/// row-major - no `Vec<Vec<f32>>`.
#[derive(Debug, Clone, Copy)]
pub struct Samples<'a> {
    pub rows: &'a [f32],
    pub n_features: usize,
    /// One per row: a dense class index `0..n_classes`.
    pub labels: &'a [u32],
    pub n_classes: usize,
}

impl Samples<'_> {
    pub fn n_samples(&self) -> usize {
        self.labels.len()
    }

    /// The checks every classifier's `train` makes first.
    pub fn validate(&self) -> Result<(), NnError> {
        if self.labels.is_empty() {
            return Err(NnError::InvalidInput("no training samples".into()));
        }
        if self.n_features == 0 || self.rows.len() != self.labels.len() * self.n_features {
            return Err(NnError::InvalidInput(format!(
                "{} values are not {} rows of {} features",
                self.rows.len(),
                self.labels.len(),
                self.n_features
            )));
        }
        if let Some(&label) = self.labels.iter().find(|&&l| l as usize >= self.n_classes) {
            return Err(NnError::InvalidInput(format!(
                "label {label} with only {} classes",
                self.n_classes
            )));
        }
        Ok(())
    }
}

/// A class per row and how certain it is (0..=1): share of trees voting for
/// it (random forest), of neighbours (KNN), softmax probability (MLP).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Predictions {
    pub classes: Vec<u32>,
    pub certainty: Vec<f32>,
}

/// A classifier trained on feature rows.
pub trait Classifier: Sized + Send + Sync {
    /// Hyperparameters (the seed is [`TrainControl::seed`]).
    type Params;

    fn train(
        samples: &Samples<'_>,
        params: &Self::Params,
        control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError>;

    /// Class + certainty for each of the `rows.len() / n_features` rows.
    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError>;
}

/// A trained classifier as stored in a `.evamodel` file - which one comes
/// from the file, so an enum. P2: serde with today's names and layout for the
/// forest and KNN (their files keep loading); the MLP as safetensors bytes.
pub enum TrainedClassifier {
    RandomForest(RandomForest),
    Knn(Knn),
    Mlp(Mlp),
}

impl TrainedClassifier {
    pub fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        match self {
            Self::RandomForest(model) => model.predict(rows, n_features),
            Self::Knn(model) => model.predict(rows, n_features),
            Self::Mlp(model) => model.predict(rows, n_features),
        }
    }
}
