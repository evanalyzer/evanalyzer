//! # random_forest
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{RandomForestSettings, SplitCriterion};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, SplitCriterion as SmartcoreSplitCriterion,
};
use std::sync::OnceLock;

type ForestModel = RandomForestClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>;
type TreeModel = DecisionTreeClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>;

/// A fitted random forest. Saved exactly as smartcore's own model
/// (`transparent`); `trees` is only a lazily filled copy of its trees for
/// [`Forest::vote_share`], since smartcore keeps them private.
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Forest {
    pub model: ForestModel,
    #[serde(skip)]
    trees: OnceLock<Vec<TreeModel>>,
}

impl Forest {
    pub fn new(model: ForestModel) -> Self {
        Self {
            model,
            trees: OnceLock::new(),
        }
    }

    /// The trees, read once through serde (the only public way to them).
    fn trees(&self) -> Result<&[TreeModel], InternalErrors> {
        #[derive(Deserialize)]
        struct Trees {
            trees: Option<Vec<TreeModel>>,
        }
        if let Some(trees) = self.trees.get() {
            return Ok(trees);
        }
        let read = serde_json::to_value(&self.model)
            .and_then(serde_json::from_value::<Trees>)
            .map_err(|e| {
                InternalErrors::Internal(format!("failed to read the forest's trees: {e}"))
            })?;
        let _ = self.trees.set(read.trees.unwrap_or_default());
        Ok(self.trees.get().map(Vec::as_slice).unwrap_or_default())
    }

    /// For every row of `features`, the class most trees vote for and the
    /// share of trees voting for it (0..=1) - the forest's certainty, as
    /// ilastik shows it. The class is exactly `predict`'s: smartcore counts
    /// the same votes and gives a tie to the class that sorts first.
    ///
    /// Each tree runs once (not `predict` and then every tree again), in
    /// chunks of [`VOTE_CHUNK_ROWS`] rows: a chunk holds every tree's votes
    /// for its rows only, so the memory doesn't grow with the image, and
    /// the chunks run in parallel.
    pub fn predict_with_share(
        &self,
        features: &[Vec<f32>],
    ) -> Result<(Vec<usize>, Vec<f32>), InternalErrors> {
        let trees = self.trees()?;
        if trees.is_empty() {
            // No trees, no votes.
            let x = to_dense_matrix(features)?;
            let classes = self
                .model
                .predict(&x)
                .map_err(|e| InternalErrors::Internal(e.to_string()))?;
            return Ok((classes, vec![0.0; features.len()]));
        }
        let chunks = features
            .par_chunks(VOTE_CHUNK_ROWS)
            .map(|chunk| {
                let x = to_dense_matrix(chunk)?;
                let votes = trees
                    .iter()
                    .map(|tree| tree.predict(&x))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| InternalErrors::Internal(e.to_string()))?;
                let n_labels = votes.iter().flatten().max().map_or(0, |max| max + 1);
                let mut counts = vec![0u32; n_labels];
                Ok((0..chunk.len())
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
                        (class, most as f32 / trees.len() as f32)
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, InternalErrors>>()?;
        Ok(chunks.into_iter().flatten().unzip())
    }
}

/// Rows per chunk in [`Forest::predict_with_share`]: every tree's votes for
/// them are held at once (trees × rows labels).
const VOTE_CHUNK_ROWS: usize = 4096;

fn to_smartcore_params(settings: &RandomForestSettings) -> RandomForestClassifierParameters {
    RandomForestClassifierParameters {
        criterion: match settings.criterion {
            SplitCriterion::Gini => SmartcoreSplitCriterion::Gini,
            SplitCriterion::Entropy => SmartcoreSplitCriterion::Entropy,
            SplitCriterion::ClassificationError => SmartcoreSplitCriterion::ClassificationError,
        },
        max_depth: settings.max_depth,
        min_samples_leaf: settings.min_samples_leaf,
        min_samples_split: settings.min_samples_split,
        n_trees: settings.n_trees,
        m: settings.m,
        keep_samples: settings.keep_samples,
        seed: settings.seed,
    }
}

/// `labels` are dense indices into the owning job's `class_labels` — see
/// `ai_learning::utils::validate_training_data`'s doc comment.
pub fn fit_random_forest(
    rows: &[Vec<f32>],
    labels: &[usize],
    settings: &RandomForestSettings,
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.to_vec();
    let model = RandomForestClassifier::fit(&x, &y, to_smartcore_params(settings))
        .map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok(Classifier::RandomForest(Forest::new(model)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters (label 0 near 0.0, label 1 near 10.0 on
    /// every feature) - trivial for any working classifier to fit exactly,
    /// so a wrong prediction here means the settings -> smartcore-params
    /// wiring (or the dense-matrix/label plumbing) is broken, not that the
    /// model just didn't generalize.
    fn two_cluster_dataset() -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![0.0 + jitter, 0.0 + jitter]);
            labels.push(0);
            rows.push(vec![10.0 + jitter, 10.0 + jitter]);
            labels.push(1);
        }
        (rows, labels)
    }

    #[test]
    fn fit_random_forest_separates_two_well_separated_clusters() {
        let (rows, labels) = two_cluster_dataset();
        let settings = RandomForestSettings {
            n_trees: 10,
            max_depth: Some(5),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: 42,
            ..Default::default()
        };

        let classifier = fit_random_forest(&rows, &labels, &settings).unwrap();

        let predictions = classifier
            .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);
    }

    #[test]
    fn the_vote_share_is_the_share_of_trees_agreeing() {
        let (rows, labels) = two_cluster_dataset();
        let settings = RandomForestSettings {
            n_trees: 20,
            seed: 7,
            ..Default::default()
        };
        let Classifier::RandomForest(forest) =
            fit_random_forest(&rows, &labels, &settings).unwrap()
        else {
            unreachable!()
        };
        let (classes, share) = forest
            .predict_with_share(&[vec![0.05, 0.05], vec![10.05, 10.05], vec![5.0, 5.0]])
            .unwrap();

        assert_eq!(&classes[..2], &[0, 1]);

        assert_eq!(&share[..2], &[1.0, 1.0], "clear cases: every tree agrees");
        assert!((0.5..=1.0).contains(&share[2]));
        // Each share is a whole number of trees.
        assert!(share.iter().all(|s| (s * 20.0).fract() == 0.0));
    }

    #[test]
    fn a_saved_forest_reads_its_trees_after_loading() {
        let (rows, labels) = two_cluster_dataset();
        let settings = RandomForestSettings {
            n_trees: 5,
            ..Default::default()
        };
        let fitted = fit_random_forest(&rows, &labels, &settings).unwrap();
        let loaded: Classifier =
            serde_json::from_slice(&serde_json::to_vec(&fitted).unwrap()).unwrap();
        let Classifier::RandomForest(forest) = loaded else {
            unreachable!()
        };
        assert_eq!(
            forest.predict_with_share(&[vec![0.05, 0.05]]).unwrap(),
            (vec![0], vec![1.0])
        );
    }

    /// The class of every row is `predict`'s, also across several chunks and
    /// with ties: an even number of trees on noisy, overlapping classes
    /// (labels 0, 2 and 5 - not 0..n, as when a class has no examples).
    #[test]
    fn predict_with_share_gives_the_same_classes_as_predict() {
        let mut seed = 12345u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let label_of = [0usize, 2, 5];
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..600 {
            let class = i % 3;
            rows.push(vec![next() + class as f32 * 0.2, next()]);
            labels.push(label_of[class]);
        }
        let settings = RandomForestSettings {
            n_trees: 6,
            max_depth: Some(4),
            seed: 3,
            ..Default::default()
        };
        let classifier = fit_random_forest(&rows, &labels, &settings).unwrap();
        let Classifier::RandomForest(forest) = &classifier else {
            unreachable!()
        };
        let pixels: Vec<Vec<f32>> = (0..VOTE_CHUNK_ROWS * 2 + 77)
            .map(|_| vec![next() * 1.4, next()])
            .collect();

        let (classes, share) = forest.predict_with_share(&pixels).unwrap();

        let x = to_dense_matrix(&pixels).unwrap();
        assert_eq!(classes, forest.model.predict(&x).unwrap());
        assert!(
            share.iter().any(|&s| s == 0.5),
            "the data has ties (3 of 6 trees)"
        );
        assert!(
            share
                .iter()
                .all(|&s| (1.0 / 3.0..=1.0).contains(&s) && (s * 6.0).fract() == 0.0)
        );
    }

    /// Speed and peak RAM of a prediction with certainty, as the pixel
    /// classifier step makes it for one 512² tile (20 features, 50 trees,
    /// 3 classes). Run with
    /// `cargo test --release -p evanalyzer_core --lib bench_predict_with_certainty -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_predict_with_certainty() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut sample = |class: usize| -> Vec<f32> {
            (0..20)
                .map(|f| next() + if f % 4 == 0 { class as f32 * 0.5 } else { 0.0 })
                .collect()
        };
        let labels: Vec<usize> = (0..5_000).map(|i| i % 3).collect();
        let rows: Vec<Vec<f32>> = labels.iter().map(|&c| sample(c)).collect();
        let settings = RandomForestSettings {
            n_trees: 50,
            seed: 1,
            ..Default::default()
        };
        let classifier = fit_random_forest(&rows, &labels, &settings).unwrap();
        let pixels: Vec<Vec<f32>> = (0..512 * 512).map(|i| sample(i % 3)).collect();

        let peak_kb = || {
            std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("VmHWM:"))
                        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                })
                .unwrap_or(0)
        };
        let before = peak_kb();
        let mut times = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let (classes, certainty) = classifier.predict_with_certainty(&pixels).unwrap();
            times.push(start.elapsed().as_secs_f64());
            assert_eq!(classes.len(), certainty.len());
        }
        times.sort_by(f64::total_cmp);
        let plain = std::time::Instant::now();
        let Classifier::RandomForest(forest) = &classifier else {
            unreachable!()
        };
        forest
            .model
            .predict(&to_dense_matrix(&pixels).unwrap())
            .unwrap();
        println!(
            "512² with certainty: median {:.2} s (min {:.2}, max {:.2}); smartcore predict (class only) {:.2} s; peak RSS {} MB (+{} MB)",
            times[1],
            times[0],
            times[2],
            plain.elapsed().as_secs_f64(),
            peak_kb() / 1024,
            peak_kb().saturating_sub(before) / 1024
        );
    }

    #[test]
    fn fit_random_forest_rejects_mismatched_rows_and_labels() {
        let rows = vec![vec![0.0], vec![1.0]];
        let labels = [0usize];
        let err = fit_random_forest(&rows, &labels, &RandomForestSettings::default()).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }
}
