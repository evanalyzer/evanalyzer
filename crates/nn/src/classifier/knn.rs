//! # knn
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! K nearest neighbours (smartcore). P2: moved from core's
//! `ai_learning/model/knn.rs`.

use crate::classifier::{Classifier, Predictions, Samples};
use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnnParams {
    pub search: KnnSearch,
    pub weight: KnnWeight,
    pub k: usize,
    pub distance: KnnDistance,
}

pub struct Knn {
    // P2: the smartcore model per distance (core's `KnnModel`).
}

impl Classifier for Knn {
    type Params = KnnParams;

    fn train(
        samples: &Samples<'_>,
        params: &KnnParams,
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
