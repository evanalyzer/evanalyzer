use crate::ai_learning::model::{Classifier, mlp::predict_mlp};
use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use smartcore::linalg::basic::matrix::DenseMatrix;

// -- Shared training helpers, backend-agnostic over how `rows`/`labels` were
// gathered (pixel samples or object feature vectors) --------------------
pub fn to_dense_matrix(rows: &[Vec<f32>]) -> Result<DenseMatrix<f32>, InternalErrors> {
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    DenseMatrix::from_2d_array(&row_refs).map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// Shared by every backend's `fit_*`/`mlp::fit_mlp` so they all validate
/// identically before touching smartcore/burn.
pub(crate) fn validate_training_data(
    rows: &[Vec<f32>],
    labels: &[SegmentationClass],
) -> Result<(), InternalErrors> {
    if rows.len() != labels.len() {
        return Err(InternalErrors::Internal(
            "sample rows and labels must have the same length".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(InternalErrors::Internal(
            "cannot train on zero samples".to_string(),
        ));
    }
    Ok(())
}

impl Classifier {
    /// Predicts a class index (into the owning `SavedClassifier::settings`'s
    /// nested `class_labels`) for each row of `features`.
    pub fn predict(&self, features: &[Vec<f32>]) -> Result<Vec<usize>, InternalErrors> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Classifier::RandomForest(model) => {
                let x = to_dense_matrix(features)?;
                model
                    .predict(&x)
                    .map_err(|e| InternalErrors::Internal(e.to_string()))
            }
            Classifier::Knn(model) => {
                let x = to_dense_matrix(features)?;
                model
                    .predict(&x)
                    .map_err(|e| InternalErrors::Internal(e.to_string()))
            }
            Classifier::Mlp {
                architecture,
                weights,
            } => predict_mlp(architecture, weights, features),
        }
    }
}
