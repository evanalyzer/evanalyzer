//! # rtree
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Random forest (smartcore's trees; class and certainty in one pass).
//! P2: moved from core's `ai_learning/model/random_forest.rs`.

use crate::classifier::{Classifier, Predictions, Samples};
use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};

/// The function to measure the quality of a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitCriterion {
    #[default]
    Gini,
    Entropy,
    ClassificationError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomForestParams {
    pub criterion: SplitCriterion,
    pub max_depth: Option<u16>,
    /// The minimum number of samples at a leaf.
    pub min_samples_leaf: usize,
    /// The minimum number of samples to split a node.
    pub min_samples_split: usize,
    pub n_trees: u16,
    /// Features tried per split, `None` = smartcore's default.
    pub m: Option<usize>,
}

pub struct RandomForest {
    // P2: the trees (core's `Forest`).
}

impl Classifier for RandomForest {
    type Params = RandomForestParams;

    fn train(
        samples: &Samples<'_>,
        params: &RandomForestParams,
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
