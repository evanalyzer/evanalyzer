//! # knn
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//! k-nearest neighbours (smartcore). Moved from core's
//! `ai_learning/model/knn.rs` (2026-11-25) with its tests.

use crate::classifier::{Classifier, Predictions, Samples, dense_matrix, row_count};
use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::cosine::Cosine;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::metrics::distance::hamming::Hamming;
use smartcore::metrics::distance::manhattan::Manhattan;
use smartcore::metrics::distance::minkowski::Minkowski;
use smartcore::neighbors::KNNWeightFunction;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

/// How the neighbours are searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnnSearch {
    #[default]
    Linear,
    CoverTree,
}

/// How the k neighbours are weighted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnnWeight {
    /// All equally.
    #[default]
    Uniform,
    /// By the inverse of their distance.
    Distance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnnDistance {
    #[default]
    Euclidean,
    Manhattan,
    Cosine,
    Hamming,
    /// Order `p` (at least 1).
    Minkowski {
        p: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KnnParams {
    pub search: KnnSearch,
    pub weight: KnnWeight,
    pub k: usize,
    pub distance: KnnDistance,
}

mod config;

pub use config::Knn;
use config::Model;

/// The smartcore parameters shared by every distance (`algorithm`, `weight`,
/// `k`) - `with_distance` swaps in the distance and, with it, the type.
fn base_params(params: &KnnParams) -> KNNClassifierParameters<f32, Euclidian<f32>> {
    KNNClassifierParameters::default()
        .with_k(params.k)
        .with_algorithm(match params.search {
            KnnSearch::Linear => KNNAlgorithmName::LinearSearch,
            KnnSearch::CoverTree => KNNAlgorithmName::CoverTree,
        })
        .with_weight(match params.weight {
            KnnWeight::Uniform => KNNWeightFunction::Uniform,
            KnnWeight::Distance => KNNWeightFunction::Distance,
        })
}

fn fit<D>(
    x: &DenseMatrix<f32>,
    y: &Vec<usize>,
    params: KNNClassifierParameters<f32, D>,
) -> Result<Model<D>, NnError>
where
    D: smartcore::metrics::distance::Distance<Vec<f32>>,
{
    KNNClassifier::fit(x, y, params).map_err(|e| NnError::Internal(e.to_string()))
}

/// Runs `$body` with `$m` bound to the model of whichever distance.
macro_rules! each_model {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            Knn::Euclidean($m) => $body,
            Knn::Manhattan($m) => $body,
            Knn::Cosine($m) => $body,
            Knn::Hamming($m) => $body,
            Knn::Minkowski($m) => $body,
        }
    };
}

impl Classifier for Knn {
    type Params = KnnParams;

    fn train(
        samples: &Samples<'_>,
        params: &KnnParams,
        _control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError> {
        samples.validate()?;
        let x = dense_matrix(samples.rows, samples.n_features)?;
        let y: Vec<usize> = samples.labels.iter().map(|&l| l as usize).collect();
        let base = base_params(params);
        let model = match params.distance {
            KnnDistance::Euclidean => {
                Knn::Euclidean(fit(&x, &y, base.with_distance(Euclidian::new()))?)
            }
            KnnDistance::Manhattan => {
                Knn::Manhattan(fit(&x, &y, base.with_distance(Manhattan::new()))?)
            }
            KnnDistance::Cosine => Knn::Cosine(fit(&x, &y, base.with_distance(Cosine::new()))?),
            KnnDistance::Hamming => Knn::Hamming(fit(&x, &y, base.with_distance(Hamming::new()))?),
            KnnDistance::Minkowski { p } => {
                Knn::Minkowski(fit(&x, &y, base.with_distance(Minkowski::new(p)))?)
            }
        };
        let stats = TrainStats::Knn {
            k: params.k,
            n_samples: samples.n_samples(),
        };
        Ok((model, stats))
    }

    /// The class and the neighbours' share of it (weighted like the
    /// prediction).
    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        if row_count(rows, n_features)? == 0 {
            return Ok(Predictions::default());
        }
        let x = dense_matrix(rows, n_features)?;
        let (classes, proba) = each_model!(self, m => m
            .predict(&x)
            .and_then(|c| Ok((c, m.predict_proba(&x)?))))
        .map_err(|e| NnError::Internal(e.to_string()))?;
        let certainty = proba
            .iter()
            .map(|row| row.iter().copied().fold(0.0, f64::max) as f32)
            .collect();
        Ok(Predictions {
            classes: classes.into_iter().map(|c| c as u32).collect(),
            certainty,
        })
    }

    /// The class only - without the second neighbour search the certainty
    /// takes.
    fn predict_classes(&self, rows: &[f32], n_features: usize) -> Result<Vec<u32>, NnError> {
        if row_count(rows, n_features)? == 0 {
            return Ok(Vec::new());
        }
        let x = dense_matrix(rows, n_features)?;
        let classes =
            each_model!(self, m => m.predict(&x)).map_err(|e| NnError::Internal(e.to_string()))?;
        Ok(classes.into_iter().map(|c| c as u32).collect())
    }
}

#[cfg(test)]
mod tests;
