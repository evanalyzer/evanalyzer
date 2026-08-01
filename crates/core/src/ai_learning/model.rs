pub mod knn;
pub mod mlp;
pub mod random_forest;

use crate::ai_learning::model::mlp::MlpArchitecture;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::path::Path;

pub const CURRENT_SAVED_CLASSIFIER_VERSION: u32 = 1;

/// The trained model, one variant per backend. This — plus `predict` below —
/// is the single interface the rest of the app (pixel/object training,
/// inference Commands) uses regardless of which algorithm was chosen; nothing
/// outside this file needs to know smartcore's or burn's own APIs.
///
/// SERIALIZATION-CRITICAL: stored inside every saved classifier model.
#[derive(Debug, Serialize, Deserialize)]
pub enum Classifier {
    RandomForest(RandomForestClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>),
    Knn(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Euclidian<f32>>),
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
