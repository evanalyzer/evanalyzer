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
use smartcore::linalg::basic::matrix::DenseMatrix;

mod config;
pub mod knn;
pub mod mlp;
pub mod rtree;

pub use config::TrainedClassifier;
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

    /// The class of each row only - cheaper where the certainty costs extra
    /// (KNN: a second neighbour search).
    fn predict_classes(&self, rows: &[f32], n_features: usize) -> Result<Vec<u32>, NnError> {
        Ok(self.predict(rows, n_features)?.classes)
    }
}

/// The number of rows in `rows` of `n_features` values - an error when they
/// don't divide.
pub(crate) fn row_count(rows: &[f32], n_features: usize) -> Result<usize, NnError> {
    if n_features == 0 || !rows.len().is_multiple_of(n_features) {
        return Err(NnError::InvalidInput(format!(
            "{} values are no rows of {n_features} features",
            rows.len()
        )));
    }
    Ok(rows.len() / n_features)
}

/// smartcore's matrix of the row-major `rows`, stored column-major as
/// smartcore's own `from_2d_array` does: a row-major matrix made the
/// forest's prediction 23 % slower (smartcore reads it column by column).
pub(crate) fn dense_matrix(rows: &[f32], n_features: usize) -> Result<DenseMatrix<f32>, NnError> {
    let n = row_count(rows, n_features)?;
    let mut columns = vec![0.0; rows.len()];
    for (r, row) in rows.chunks_exact(n_features.max(1)).enumerate() {
        for (c, &value) in row.iter().enumerate() {
            columns[c * n + r] = value;
        }
    }
    DenseMatrix::new(n, n_features, columns, true).map_err(|e| NnError::Internal(e.to_string()))
}

impl TrainedClassifier {
    pub fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        match self {
            Self::RandomForest(model) => model.predict(rows, n_features),
            Self::Knn(model) => model.predict(rows, n_features),
            Self::Mlp(model) => model.predict(rows, n_features),
        }
    }

    pub fn predict_classes(&self, rows: &[f32], n_features: usize) -> Result<Vec<u32>, NnError> {
        match self {
            Self::RandomForest(model) => model.predict_classes(rows, n_features),
            Self::Knn(model) => model.predict_classes(rows, n_features),
            Self::Mlp(model) => model.predict_classes(rows, n_features),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
