pub mod knn;
pub mod mlp;
pub mod random_forest;

use crate::ai_learning::model::mlp::MlpArchitecture;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::cosine::Cosine;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::metrics::distance::hamming::Hamming;
use smartcore::metrics::distance::manhattan::Manhattan;
use smartcore::metrics::distance::minkowski::Minkowski;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::path::Path;

pub const CURRENT_SAVED_CLASSIFIER_VERSION: u32 = 1;

/// A fitted KNN model, one variant per [`KNNDistanceMetric`](evanalyzer_cfg::settings::ai_learning_settings::KNNDistanceMetric).
///
/// smartcore's `KNNClassifier<TX, TY, X, Y, D>` is generic over its distance
/// type `D` - a compile-time choice, unlike `algorithm`/`weight`/`k`, which
/// are runtime fields of `KNNClassifierParameters` regardless of `D`. Since
/// the distance metric is a runtime setting here (`KnnSettings::distance`),
/// this enum is what makes that possible: each variant fixes one concrete
/// `D`, and `fit_knn`/`Classifier::predict` match on it the same way
/// `Classifier` itself dispatches across backends.
///
/// SERIALIZATION-CRITICAL: stored inside every saved classifier model.
#[derive(Debug, Serialize, Deserialize)]
pub enum KnnModel {
    Euclidean(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Euclidian<f32>>),
    Manhattan(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Manhattan<f32>>),
    Cosine(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Cosine<f32>>),
    Hamming(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Hamming<f32>>),
    Minkowski(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Minkowski<f32>>),
}

/// The trained model, one variant per backend. This — plus `predict` below —
/// is the single interface the rest of the app (pixel/object training,
/// inference Commands) uses regardless of which algorithm was chosen; nothing
/// outside this file needs to know smartcore's or burn's own APIs.
///
/// SERIALIZATION-CRITICAL: stored inside every saved classifier model.
#[derive(Debug, Serialize, Deserialize)]
pub enum Classifier {
    RandomForest(RandomForestClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>),
    Knn(KnnModel),
    Mlp {
        architecture: MlpArchitecture,
        /// Weights recorded via `BinBytesRecorder<FullPrecisionSettings>` —
        /// an in-memory byte blob, not a filesystem path, so it can be
        /// embedded directly in `SavedClassifier`.
        weights: Vec<u8>,
    },
}

/// The full artifact saved to `<project>/models/<name>`: the trained weights
/// plus everything needed to reproduce its input features and interpret its
/// output — `AiLearningSettings` (`evanalyzer_cfg`) carries the backend
/// hyperparameters, the feature recipe, the class labels (with the real IDs
/// `Classifier::predict`'s indices refer to — not display names, since
/// `evanalyzer_cfg::PixelClassLabel`/`ObjectClassLabel` already snapshot the
/// name alongside the stable ID for portability), and descriptive metadata.
/// `Classifier`/`AiLearningSettings` used to be two independently-evolved,
/// overlapping descriptions of the same model (`FeatureRecipe` here duplicated
/// what `AiLearningSettings::classifier` already covers); this replaces that
/// duplication with the one `evanalyzer_cfg` type, which is also the only one
/// of the two schema-able (`JsonSchema`) and reusable outside the `ai` feature.
///
/// SERIALIZATION-CRITICAL: `version` is the escape hatch for future format
/// changes — never change the meaning of an existing field for a given
/// version, bump `version` instead.
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedClassifier {
    pub version: u32,
    pub classifier: Classifier,
    pub settings: AiLearningSettings,
}

/// Serializes as JSON (not bincode — bincode's development has ceased; JSON
/// also matches this codebase's existing settings-serialization convention,
/// see the generated `...Settings` types' `JsonSchema` derives).
pub fn save_to_file(classifier: &SavedClassifier, path: &Path) -> Result<(), InternalErrors> {
    let json = serde_json::to_vec(classifier)
        .map_err(|e| InternalErrors::Internal(format!("failed to serialize classifier: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| InternalErrors::Internal(format!("failed to create models dir: {e}")))?;
    }
    std::fs::write(path, json)
        .map_err(|e| InternalErrors::Internal(format!("failed to write model file: {e}")))
}

pub fn load_from_file(path: &Path) -> Result<SavedClassifier, InternalErrors> {
    let bytes = std::fs::read(path)
        .map_err(|e| InternalErrors::Internal(format!("failed to read model file: {e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| InternalErrors::Internal(format!("failed to deserialize classifier: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningClassifierSettings, RandomForestSettings,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;

    fn sample_saved_classifier() -> SavedClassifier {
        let rows = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let labels = [0usize, 0, 1, 1];
        let classifier =
            random_forest::fit_random_forest(&rows, &labels, &RandomForestSettings::default())
                .unwrap();
        SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                metadata: MetaData {
                    name: "test-model".into(),
                    ..Default::default()
                },
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                    class_labels: vec![],
                },
            },
        }
    }

    #[test]
    fn save_then_load_round_trips_metadata_and_predictions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models").join("test.evamodel");
        let saved = sample_saved_classifier();

        save_to_file(&saved, &path).unwrap();
        let loaded = load_from_file(&path).unwrap();

        assert_eq!(loaded.version, CURRENT_SAVED_CLASSIFIER_VERSION);
        assert_eq!(loaded.settings.metadata.name, "test-model");

        let before = saved.classifier.predict(&[vec![0.5, 0.5]]).unwrap();
        let after = loaded.classifier.predict(&[vec![0.5, 0.5]]).unwrap();
        assert_eq!(
            before, after,
            "a round-tripped model must predict identically"
        );
    }

    #[test]
    fn save_to_file_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("nested")
            .join("models")
            .join("test.evamodel");
        assert!(!path.parent().unwrap().exists());

        save_to_file(&sample_saved_classifier(), &path).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn load_from_file_errors_for_a_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist.evamodel");
        assert!(load_from_file(&missing).is_err());
    }

    #[test]
    fn load_from_file_errors_for_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.evamodel");
        std::fs::write(&path, b"{ not valid json").unwrap();
        assert!(load_from_file(&path).is_err());
    }
}
