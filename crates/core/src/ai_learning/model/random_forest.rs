use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{RandomForestSettings, SplitCriterion};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::arrays::Array;
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

    /// For every row, the share of trees voting for the most voted class
    /// (0..=1) - the forest's certainty, as ilastik shows it.
    pub fn vote_share(&self, x: &DenseMatrix<f32>) -> Result<Vec<f32>, InternalErrors> {
        let trees = self.trees()?;
        let votes = trees
            .iter()
            .map(|tree| tree.predict(x))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| InternalErrors::Internal(e.to_string()))?;
        let Some(first) = votes.first() else {
            // No trees, no votes.
            return Ok(vec![0.0; x.shape().0]);
        };
        let classes = votes.iter().flatten().max().map_or(0, |max| max + 1);
        let mut counts = vec![0u32; classes];
        Ok((0..first.len())
            .map(|row| {
                counts.fill(0);
                for tree_votes in &votes {
                    counts[tree_votes[row]] += 1;
                }
                counts.iter().copied().max().unwrap_or(0) as f32 / trees.len() as f32
            })
            .collect())
    }
}

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
        let x = to_dense_matrix(&[vec![0.05, 0.05], vec![10.05, 10.05], vec![5.0, 5.0]]).unwrap();

        let share = forest.vote_share(&x).unwrap();

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
        let x = to_dense_matrix(&[vec![0.05, 0.05]]).unwrap();
        assert_eq!(forest.vote_share(&x).unwrap(), vec![1.0]);
    }

    #[test]
    fn fit_random_forest_rejects_mismatched_rows_and_labels() {
        let rows = vec![vec![0.0], vec![1.0]];
        let labels = [0usize];
        let err = fit_random_forest(&rows, &labels, &RandomForestSettings::default()).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }
}
