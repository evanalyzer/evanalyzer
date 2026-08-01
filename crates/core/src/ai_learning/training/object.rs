use crate::ai_learning::model::knn::fit_knn;
use crate::ai_learning::model::mlp;
use crate::ai_learning::model::random_forest::fit_random_forest;
use crate::{Object, ai_learning::model::Classifier};
use evanalyzer_cfg::{
    core_types::{InternalErrors, SegmentationClass},
    settings::ai_learning_object_settings::{AiLearningObjectFeatureSettings, ObjectMetric},
};

/// Assembles one feature vector for `object`, in `spec.metrics` order.
pub fn compute_object_features(
    object: &Object,
    spec: &AiLearningObjectFeatureSettings,
) -> Vec<f32> {
    spec.metrics
        .iter()
        .map(|metric| match metric {
            ObjectMetric::Area => object.area as f32,
            ObjectMetric::Perimeter => object.get_perimeter(),
            ObjectMetric::Circularity => object.circularity(),
            ObjectMetric::Solidity => object.get_solidity(),
            ObjectMetric::AspectRatio => object.get_aspect_ratio(),
            ObjectMetric::Roundness => object.get_roundness(object.get_perimeter()),
            ObjectMetric::Compactness => object.get_compactness(object.get_perimeter()),
            ObjectMetric::FeretDiameter => object.get_feret_diameter(),
            ObjectMetric::MinFeretDiameter => object.get_min_feret_diameter(),
            ObjectMetric::EllipseMajor => object.get_ellipse().major,
            ObjectMetric::EllipseMinor => object.get_ellipse().minor,
            ObjectMetric::EllipseAngle => object.get_ellipse().angle,
            ObjectMetric::Eccentricity => object.get_ellipse().eccentricity,
            ObjectMetric::TouchesEdge => {
                if object.touches_edge {
                    1.0
                } else {
                    0.0
                }
            }
            ObjectMetric::IntensitySum(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.sum_intensity as f32)
                .unwrap_or(0.0),
            ObjectMetric::IntensityMin(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.min_intensity)
                .unwrap_or(0.0),
            ObjectMetric::IntensityMax(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.max_intensity)
                .unwrap_or(0.0),
            ObjectMetric::IntensityAvg(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.avg_intensity)
                .unwrap_or(0.0),
        })
        .collect()
}

fn build_rows(objects: &[&Object], spec: &AiLearningObjectFeatureSettings) -> Vec<Vec<f32>> {
    objects
        .iter()
        .map(|o| compute_object_features(o, spec))
        .collect()
}

/// Trains a Random Forest object classifier from labeled objects. `labels[i]`
/// is the class index for `objects[i]`.
pub fn train_random_forest(
    objects: &[&Object],
    spec: &AiLearningObjectFeatureSettings,
    labels: &[SegmentationClass],
) -> Result<Classifier, InternalErrors> {
    fit_random_forest(&build_rows(objects, spec), labels)
}

/// Trains a K-Nearest-Neighbors object classifier from labeled objects.
pub fn train_knn(
    objects: &[&Object],
    spec: &AiLearningObjectFeatureSettings,
    labels: &[SegmentationClass],
) -> Result<Classifier, InternalErrors> {
    fit_knn(&build_rows(objects, spec), labels)
}

/// Trains a small feed-forward (MLP) object classifier from labeled objects.
pub fn train_mlp(
    objects: &[&Object],
    spec: &AiLearningObjectFeatureSettings,
    labels: &[SegmentationClass],
    hidden_layers: &[usize],
    epochs: usize,
    learning_rate: f64,
) -> Result<Classifier, InternalErrors> {
    mlp::fit_mlp(
        &build_rows(objects, spec),
        labels,
        hidden_layers,
        epochs,
        learning_rate,
    )
}
