use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{RandomForestSettings, SplitCriterion};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::tree::decision_tree_classifier::SplitCriterion as SmartcoreSplitCriterion;

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
    Ok(Classifier::RandomForest(model))
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
    fn fit_random_forest_rejects_mismatched_rows_and_labels() {
        let rows = vec![vec![0.0], vec![1.0]];
        let labels = [0usize];
        let err = fit_random_forest(&rows, &labels, &RandomForestSettings::default()).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }
}
