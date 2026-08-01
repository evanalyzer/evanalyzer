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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters - see `random_forest`'s test module doc
    /// comment for why this is a wiring smoke test, not a generalization one.
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
    fn fit_knn_separates_two_well_separated_clusters() {
        let (rows, labels) = two_cluster_dataset();
        let settings = KnnSettings {
            k: 3,
            algorithm: CfgKNNAlgorithmName::CoverTree,
            weight: CfgKNNWeightFunction::Uniform,
        };

        let classifier = fit_knn(&rows, &labels, &settings).unwrap();

        let predictions = classifier
            .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);
    }

    #[test]
    fn fit_knn_rejects_zero_samples() {
        let settings = KnnSettings {
            k: 3,
            ..Default::default()
        };
        let err = fit_knn(&[], &[], &settings).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }
}
