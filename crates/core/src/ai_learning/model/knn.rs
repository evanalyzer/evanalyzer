use super::mlp::{MlpArchitecture, predict_mlp};
use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use std::path::Path;

pub fn fit_knn(
    rows: &[Vec<f32>],
    labels: &[SegmentationClass],
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.iter().map(|item| item.as_usize()).collect();
    let model = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default())
        .map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok(Classifier::Knn(model))
}
