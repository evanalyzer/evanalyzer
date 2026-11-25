//! # model
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-07-31
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningSettings;
use evanalyzer_nn::NnError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The trained model: a random forest, KNN or MLP (`evanalyzer_nn`, whose
/// `classifier/**/config.rs` files define what is saved of each).
pub use evanalyzer_nn::TrainedClassifier as Classifier;

/// 2 (2026-11-25): the classifiers moved to `evanalyzer_nn` (MLP weights as
/// safetensors). Older files are refused - no backwards compatibility.
pub const CURRENT_SAVED_CLASSIFIER_VERSION: u32 = 2;

/// The full artifact saved to `<project>/models/<name>`: the trained model
/// plus everything needed to reproduce its input features and interpret its
/// output - `AiLearningSettings` (`evanalyzer_cfg`) carries the backend
/// hyperparameters, the feature recipe, the class labels (with the real IDs
/// the predicted class indices refer to) and descriptive metadata.
///
/// SERIALIZATION-CRITICAL: a change of what a file holds (here or in nn's
/// `config.rs` files) bumps [`CURRENT_SAVED_CLASSIFIER_VERSION`].
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedClassifier {
    pub version: u32,
    pub classifier: Classifier,
    pub settings: AiLearningSettings,
}

/// The classifiers' errors in core's terms.
pub(crate) fn nn_error(e: NnError) -> InternalErrors {
    match e {
        NnError::Cancelled => InternalErrors::Cancelled,
        NnError::InvalidInput(m) => InternalErrors::InvalidArgument(m),
        other => InternalErrors::Internal(other.to_string()),
    }
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
    /// Only the version: an older file can't be read as this one.
    #[derive(Deserialize)]
    struct Version {
        version: u32,
    }
    let version = serde_json::from_slice::<Version>(&bytes)
        .map_err(|e| InternalErrors::Internal(format!("failed to deserialize classifier: {e}")))?
        .version;
    if version != CURRENT_SAVED_CLASSIFIER_VERSION {
        return Err(InternalErrors::InvalidArgument(format!(
            "the classifier {} was saved by {} version of EVAnalyzer - train it again",
            path.display(),
            if version < CURRENT_SAVED_CLASSIFIER_VERSION { "an older" } else { "a newer" }
        )));
    }
    serde_json::from_slice(&bytes)
        .map_err(|e| InternalErrors::Internal(format!("failed to deserialize classifier: {e}")))
}

/// Tests' shortcuts in the shapes the tests were written in (a vector per
/// row, `usize` classes).
#[cfg(test)]
pub(crate) mod test_support {
    use super::Classifier;
    use evanalyzer_cfg::core_types::InternalErrors;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, RandomForestSettings,
    };

    /// A random forest of `settings` on `rows`/`labels`.
    pub(crate) fn fit_random_forest(
        rows: &[Vec<f32>],
        labels: &[usize],
        settings: &RandomForestSettings,
    ) -> Result<Classifier, InternalErrors> {
        let n_classes = labels.iter().max().map_or(1, |m| m + 1);
        try_fit(&AiLearningBackendSettings::RandomForest(settings.clone()), rows, labels, n_classes)
    }

    /// The classifier `backend` selects on `rows`/`labels`.
    pub(crate) fn try_fit(
        backend: &AiLearningBackendSettings,
        rows: &[Vec<f32>],
        labels: &[usize],
        n_classes: usize,
    ) -> Result<Classifier, InternalErrors> {
        let n_features = rows.first().map_or(0, Vec::len);
        let labels: Vec<u32> = labels.iter().map(|&l| l as u32).collect();
        let (progress, _rx) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        crate::ai_learning::training_job::fit_classifier(
            backend,
            &rows.concat(),
            n_features,
            &labels,
            n_classes,
            &progress,
            &cancel,
        )
        .map(|(classifier, _)| classifier)
    }

    /// The class of each row.
    pub(crate) fn predict(classifier: &Classifier, rows: &[Vec<f32>]) -> Vec<usize> {
        predict_with_certainty(classifier, rows).0
    }

    /// The class of each row and its certainty.
    pub(crate) fn predict_with_certainty(
        classifier: &Classifier,
        rows: &[Vec<f32>],
    ) -> (Vec<usize>, Vec<f32>) {
        let n_features = rows.first().map_or(1, Vec::len);
        let p = classifier.predict(&rows.concat(), n_features).unwrap();
        (p.classes.into_iter().map(|c| c as usize).collect(), p.certainty)
    }
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
            test_support::fit_random_forest(&rows, &labels, &RandomForestSettings::default())
                .unwrap();
        SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData {
                    name: "test-model".into(),
                    ..Default::default()
                },
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings::default(),
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
        assert_eq!(loaded.settings.meta.name, "test-model");

        let before = test_support::predict(&saved.classifier, &[vec![0.5, 0.5]]);
        let after = test_support::predict(&loaded.classifier, &[vec![0.5, 0.5]]);
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
