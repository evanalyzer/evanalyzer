use crate::ai_learning::model::{Classifier, KnnModel};
use crate::ai_learning::utils::{to_dense_matrix, validate_training_data};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{
    KNNAlgorithmName as CfgKNNAlgorithmName, KNNDistanceMetric,
    KNNWeightFunction as CfgKNNWeightFunction, KnnSettings,
};
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::cosine::Cosine;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::metrics::distance::hamming::Hamming;
use smartcore::metrics::distance::manhattan::Manhattan;
use smartcore::metrics::distance::minkowski::Minkowski;
use smartcore::neighbors::KNNWeightFunction;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

/// Builds the smartcore parameters shared by every distance metric
/// (`algorithm`/`weight`/`k`) - `with_distance` below swaps in the concrete
/// distance instance and, with it, the parameters' own type.
fn base_params(settings: &KnnSettings) -> KNNClassifierParameters<f32, Euclidian<f32>> {
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
///
/// Dispatches on `settings.distance` to fit against the matching smartcore
/// distance type - see `KnnModel`'s doc comment for why this can't be a
/// single generic function the way `base_params` is.
pub fn fit_knn(
    rows: &[Vec<f32>],
    labels: &[usize],
    settings: &KnnSettings,
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.to_vec();

    let model = match &settings.distance {
        KNNDistanceMetric::Euclidean => KnnModel::Euclidean(fit_with_distance(
            &x,
            &y,
            base_params(settings).with_distance(Euclidian::new()),
        )?),
        KNNDistanceMetric::Manhattan => KnnModel::Manhattan(fit_with_distance(
            &x,
            &y,
            base_params(settings).with_distance(Manhattan::new()),
        )?),
        KNNDistanceMetric::Cosine => KnnModel::Cosine(fit_with_distance(
            &x,
            &y,
            base_params(settings).with_distance(Cosine::new()),
        )?),
        KNNDistanceMetric::Hamming => KnnModel::Hamming(fit_with_distance(
            &x,
            &y,
            base_params(settings).with_distance(Hamming::new()),
        )?),
        KNNDistanceMetric::Minkowski { p } => KnnModel::Minkowski(fit_with_distance(
            &x,
            &y,
            base_params(settings).with_distance(Minkowski::new(*p)),
        )?),
    };
    Ok(Classifier::Knn(model))
}

fn fit_with_distance<D>(
    x: &DenseMatrix<f32>,
    y: &Vec<usize>,
    params: KNNClassifierParameters<f32, D>,
) -> Result<KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, D>, InternalErrors>
where
    D: smartcore::metrics::distance::Distance<Vec<f32>>,
{
    KNNClassifier::fit(x, y, params).map_err(|e| InternalErrors::Internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_learning::model::KnnModel;

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
            distance: KNNDistanceMetric::Euclidean,
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

    /// Every non-default distance metric must actually be reachable end to
    /// end (settings -> `fit_knn` -> the matching `KnnModel` variant) and
    /// still able to separate two well-separated clusters - not just compile.
    #[test]
    fn fit_knn_separates_clusters_under_every_distance_metric() {
        let (rows, labels) = two_cluster_dataset();
        let metrics_and_expected_variant: Vec<(KNNDistanceMetric, fn(&KnnModel) -> bool)> = vec![
            (KNNDistanceMetric::Euclidean, |m| {
                matches!(m, KnnModel::Euclidean(_))
            }),
            (KNNDistanceMetric::Manhattan, |m| {
                matches!(m, KnnModel::Manhattan(_))
            }),
            (KNNDistanceMetric::Minkowski { p: 3 }, |m| {
                matches!(m, KnnModel::Minkowski(_))
            }),
        ];

        for (distance, is_expected_variant) in metrics_and_expected_variant {
            let settings = KnnSettings {
                k: 3,
                algorithm: CfgKNNAlgorithmName::CoverTree,
                weight: CfgKNNWeightFunction::Uniform,
                distance: distance.clone(),
            };

            let classifier = fit_knn(&rows, &labels, &settings)
                .unwrap_or_else(|e| panic!("fit_knn failed for {distance:?}: {e:?}"));

            let Classifier::Knn(model) = &classifier else {
                panic!("expected a Knn classifier for {distance:?}");
            };
            assert!(
                is_expected_variant(model),
                "{distance:?} produced the wrong KnnModel variant"
            );

            let predictions = classifier
                .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
                .unwrap();
            assert_eq!(
                predictions,
                vec![0, 1],
                "distance metric {distance:?} failed to separate two well-separated clusters"
            );
        }
    }

    /// Cosine distance measures the *angle* between vectors, not their
    /// position or magnitude - `two_cluster_dataset`'s two clusters both lie
    /// near the same ray from the origin (only their distance from it
    /// differs), so they're indistinguishable by angle and it isn't a
    /// meaningful test fixture for this metric. Uses two clusters near
    /// orthogonal directions instead, at a mix of magnitudes so a
    /// position-based metric (e.g. Euclidean) couldn't separate them by
    /// magnitude alone.
    #[test]
    fn fit_knn_with_cosine_distance_separates_clusters_by_angle_not_magnitude() {
        let rows = vec![
            vec![1.0, 0.05],
            vec![2.0, 0.1],
            vec![0.5, 0.02],
            vec![0.05, 1.0],
            vec![0.1, 2.0],
            vec![0.02, 0.5],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let settings = KnnSettings {
            k: 3,
            algorithm: CfgKNNAlgorithmName::CoverTree,
            weight: CfgKNNWeightFunction::Uniform,
            distance: KNNDistanceMetric::Cosine,
        };

        let classifier = fit_knn(&rows, &labels, &settings).unwrap();
        let Classifier::Knn(KnnModel::Cosine(_)) = &classifier else {
            panic!("expected KnnModel::Cosine");
        };

        // Same direction as cluster 0, but a magnitude neither cluster's
        // training points used - only separable by angle, not distance.
        let predictions = classifier.predict(&[vec![10.0, 0.5]]).unwrap();
        assert_eq!(predictions, vec![0]);
    }

    /// Hamming distance counts differing positions, so it's exercised
    /// separately with 0/1-valued rows instead of `two_cluster_dataset`'s
    /// continuous coordinates, which it isn't a meaningful metric for.
    #[test]
    fn fit_knn_with_hamming_distance_separates_binary_clusters() {
        let rows = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let settings = KnnSettings {
            k: 3,
            algorithm: CfgKNNAlgorithmName::LinearSearch,
            weight: CfgKNNWeightFunction::Uniform,
            distance: KNNDistanceMetric::Hamming,
        };

        let classifier = fit_knn(&rows, &labels, &settings).unwrap();
        let Classifier::Knn(KnnModel::Hamming(_)) = &classifier else {
            panic!("expected KnnModel::Hamming");
        };

        let predictions = classifier
            .predict(&[vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);
    }
}
