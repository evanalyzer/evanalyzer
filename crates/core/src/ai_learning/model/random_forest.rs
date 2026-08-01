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
