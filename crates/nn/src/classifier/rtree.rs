//! # rtree
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Random forest: smartcore's trees, class and certainty (share of trees)
//! in one pass. Moved from core's `ai_learning/model/random_forest.rs`
//! (2026-11-25) with its tests.

use crate::classifier::{Classifier, Predictions, Samples, dense_matrix, row_count};
use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};
use rayon::prelude::*;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::SplitCriterion as SmartcoreSplitCriterion;

type ForestModel = RandomForestClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>;

mod config;

pub use config::RandomForest;

/// Rows per chunk of a prediction: a chunk holds every tree's votes for its
/// rows only.
const VOTE_CHUNK_ROWS: usize = 4096;

/// The function to measure the quality of a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitCriterion {
    #[default]
    Gini,
    Entropy,
    ClassificationError,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

fn to_smartcore_params(params: &RandomForestParams, seed: u64) -> RandomForestClassifierParameters {
    RandomForestClassifierParameters {
        criterion: match params.criterion {
            SplitCriterion::Gini => SmartcoreSplitCriterion::Gini,
            SplitCriterion::Entropy => SmartcoreSplitCriterion::Entropy,
            SplitCriterion::ClassificationError => SmartcoreSplitCriterion::ClassificationError,
        },
        max_depth: params.max_depth,
        min_samples_leaf: params.min_samples_leaf,
        min_samples_split: params.min_samples_split,
        n_trees: params.n_trees,
        m: params.m,
        // Out-of-bag samples: never kept, see `RandomForest`.
        keep_samples: false,
        seed,
    }
}

impl RandomForest {
    /// The trees of smartcore's `model` (through serde, the only public way
    /// to them).
    fn from_smartcore(model: &ForestModel) -> Result<Self, NnError> {
        serde_json::to_vec(model)
            .and_then(|bytes| serde_json::from_slice(&bytes))
            .map_err(|e| NnError::Internal(format!("failed to read the forest's trees: {e}")))
    }
}

impl Classifier for RandomForest {
    type Params = RandomForestParams;

    fn train(
        samples: &Samples<'_>,
        params: &RandomForestParams,
        control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError> {
        samples.validate()?;
        let x = dense_matrix(samples.rows, samples.n_features)?;
        let y: Vec<usize> = samples.labels.iter().map(|&l| l as usize).collect();
        let model = RandomForestClassifier::fit(&x, &y, to_smartcore_params(params, control.seed))
            .map_err(|e| NnError::Internal(e.to_string()))?;
        let stats = TrainStats::RandomForest {
            n_trees: params.n_trees as usize,
            n_samples: samples.n_samples(),
        };
        Ok((Self::from_smartcore(&model)?, stats))
    }

    /// For every row, the class most trees vote for and the share of trees
    /// voting for it (0..=1) - the forest's certainty, as ilastik shows it.
    /// The class is exactly smartcore's `predict`'s: it counts the same votes
    /// and gives a tie to the class that sorts first.
    ///
    /// Each tree runs once, in chunks of [`VOTE_CHUNK_ROWS`] rows: a chunk
    /// holds every tree's votes for its rows only, so the memory doesn't
    /// grow with the image, and the chunks run in parallel.
    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        let n_rows = row_count(rows, n_features)?;
        let trees = &self.trees;
        if trees.is_empty() {
            // No trees, no votes: every row the first class, as smartcore.
            let first = self
                .classes
                .as_ref()
                .and_then(|classes| classes.first())
                .ok_or_else(|| NnError::InvalidModelFile("the forest has no classes".into()))?;
            return Ok(Predictions {
                classes: vec![*first as u32; n_rows],
                certainty: vec![0.0; n_rows],
            });
        }
        if n_rows == 0 {
            return Ok(Predictions::default());
        }
        let chunks = rows
            .par_chunks(VOTE_CHUNK_ROWS * n_features)
            .map(|chunk| {
                let x = dense_matrix(chunk, n_features)?;
                let chunk_rows = chunk.len() / n_features;
                let votes = trees
                    .iter()
                    .map(|tree| tree.predict(&x))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| NnError::Internal(e.to_string()))?;
                let n_labels = votes.iter().flatten().max().map_or(0, |max| max + 1);
                let mut counts = vec![0u32; n_labels];
                Ok((0..chunk_rows)
                    .map(|row| {
                        counts.fill(0);
                        for tree_votes in &votes {
                            counts[tree_votes[row]] += 1;
                        }
                        // The first of the most voted labels, like smartcore.
                        let (class, &most) = counts
                            .iter()
                            .enumerate()
                            .rev()
                            .max_by_key(|(_, count)| **count)
                            .unwrap_or((0, &0));
                        (class as u32, most as f32 / trees.len() as f32)
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, NnError>>()?;
        let (classes, certainty) = chunks.into_iter().flatten().unzip();
        Ok(Predictions { classes, certainty })
    }
}

#[cfg(test)]
mod tests;
