//! # config
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! **KNN's part of the `.evamodel` format**: smartcore's fitted classifier
//! per distance in smartcore's own serde form - a smartcore upgrade that
//! changes its form changes the files too.
//!
//! Any change here (a field, a variant, a name, a serde attribute) changes
//! what saved `.evamodel` files contain: core's
//! `CURRENT_SAVED_CLASSIFIER_VERSION` goes up with it (older files are
//! refused - no backwards compatibility). The test
//! `classifier::tests::the_file_layout_is_pinned` guards it.

use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::cosine::Cosine;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::metrics::distance::hamming::Hamming;
use smartcore::metrics::distance::manhattan::Manhattan;
use smartcore::metrics::distance::minkowski::Minkowski;
use smartcore::neighbors::knn_classifier::KNNClassifier;

pub(super) type Model<D> = KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, D>;

/// A fitted KNN model, one variant per [`KnnDistance`](super::KnnDistance): smartcore's
/// classifier is generic over its distance, so each is its own type.
#[derive(Debug, Serialize, Deserialize)]
pub enum Knn {
    Euclidean(Model<Euclidian<f32>>),
    Manhattan(Model<Manhattan<f32>>),
    Cosine(Model<Cosine<f32>>),
    Hamming(Model<Hamming<f32>>),
    Minkowski(Model<Minkowski<f32>>),
}
