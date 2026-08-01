use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{
    KNNAlgorithmName as CfgKNNAlgorithmName, KNNWeightFunction as CfgKNNWeightFunction, KnnSettings,
};
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::KNNWeightFunction;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

fn to_smartcore_params(settings: &KnnSettings) -> KNNClassifierParameters<f32, Euclidian<f32>> {
    KNNClassifierParameters::default()
        .with_k(settings.k)
        .with_algorithm(match settings.algorithm {
            CfgKNNAlgorithmName::LinearSearch => KNNAlgorithmName::LinearSearch,
            CfgKNNAlgorithmName::CoverTree => KNNAlgorithmName::CoverTree,
        })
        .with_weight(match settings.weight {
            CfgKNNWeightFunction::Uniform => KNNWeightFunction::Uniform,
            CfgKNNWeightFunction::Distance => KNNWeightFunction::Distance,
        })
}

/// `labels` are dense indices into the owning job's `class_labels` — see
/// `ai_learning::utils::validate_training_data`'s doc comment.
pub fn fit_knn(
    rows: &[Vec<f32>],
    labels: &[usize],
    settings: &KnnSettings,
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.to_vec();
    let model = KNNClassifier::fit(&x, &y, to_smartcore_params(settings))
        .map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok(Classifier::Knn(model))
}
