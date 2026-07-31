use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Int, Tensor, TensorData};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, SegmentationClass};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use std::path::Path;

use super::object_settings::ObjectFeatureSpec;
use super::pixel_settings::FeatureSpec;

/// The MLP backend used for both training and inference. Autodiff-wrapped even
/// at inference time (simpler, at the cost of unnecessary gradient-tracking
/// overhead on predict) — splitting out a pure-inference backend without the
/// Autodiff wrapper is a reasonable follow-up optimization, not done here.
type MlpBackend = Autodiff<NdArray<f32>>;

fn mlp_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

/// A small feed-forward classifier: `Linear` layers separated by a ReLU,
/// sized by `MlpArchitecture`. Verified against burn 0.21's real API (module
/// definition, forward pass, `BinBytesRecorder` save/load) rather than
/// assumed, since burn's API has shifted significantly across versions.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device, n_in: usize, hidden: &[usize], n_out: usize) -> Self {
        let mut sizes = vec![n_in];
        sizes.extend_from_slice(hidden);
        sizes.push(n_out);
        let layers = sizes
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();
        Self {
            layers,
            activation: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.layers.len();
        self.layers.iter().enumerate().fold(x, |x, (i, l)| {
            let x = l.forward(x);
            if i + 1 < n {
                self.activation.forward(x)
            } else {
                x
            }
        })
    }
}

/// Reconstructible shape of a trained `Mlp` — `Linear` layer weights alone
/// (via `BinBytesRecorder`) can't be loaded without first building a model of
/// the exact same shape, so this travels alongside the recorded weight bytes.
///
/// SERIALIZATION-CRITICAL: part of every saved MLP classifier — never rename
/// or remove these fields once a model has been saved with them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpArchitecture {
    pub n_in: usize,
    pub hidden: Vec<usize>,
    pub n_out: usize,
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
    Knn(KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Euclidian<f32>>),
    Mlp {
        architecture: MlpArchitecture,
        /// Weights recorded via `BinBytesRecorder<FullPrecisionSettings>` —
        /// an in-memory byte blob, not a filesystem path, so it can be
        /// embedded directly in `SavedClassifier`.
        weights: Vec<u8>,
    },
}

impl Classifier {
    /// Predicts a class index (into the owning `SavedClassifier::class_labels`)
    /// for each row of `features`.
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

/// TODO: the exact tensor -> Vec<usize> extraction (argmax + data readout)
/// below is written from general burn API knowledge, not verified against
/// 0.21.0 the way training/save/load were — worth a quick check against
/// `Tensor::argmax`/`TensorData` conversion methods before relying on it.
fn predict_mlp(
    architecture: &MlpArchitecture,
    weights: &[u8],
    features: &[Vec<f32>],
) -> Result<Vec<usize>, InternalErrors> {
    let device = mlp_device();
    let model = Mlp::<MlpBackend>::new(
        &device,
        architecture.n_in,
        &architecture.hidden,
        architecture.n_out,
    );

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(weights.to_vec(), &device)
        .map_err(|e| InternalErrors::Internal(format!("failed to load MLP weights: {e:?}")))?;
    let model = model.load_record(record);

    let n_rows = features.len();
    let n_in = architecture.n_in;
    let flat: Vec<f32> = features.iter().flatten().copied().collect();
    let x = Tensor::<MlpBackend, 2>::from_data(TensorData::new(flat, [n_rows, n_in]), &device);

    let out = model.forward(x);
    let preds = out.argmax(1);
    let preds_data = preds
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|e| InternalErrors::Internal(format!("failed to read MLP predictions: {e:?}")))?;
    Ok(preds_data.into_iter().map(|v| v as usize).collect())
}

/// Which feature recipe a `SavedClassifier` uses to build its input vectors,
/// bundled with the real class IDs `Classifier::predict`'s indices refer to —
/// not display names. Names get resolved from the current project's class
/// list at display time (`ObjectClass`/`SegmentationClass` are stable IDs;
/// renaming a class in the project must not break an already-saved model,
/// which a `Vec<String>` of display names would).
///
/// Pixel classification writes to `segmentation_class` (matching `Threshold`'s
/// own `object_class_id: SegmentationClass`), so pixel labels are
/// `SegmentationClass`; object classification writes to `object_class`, so
/// object labels are `ObjectClass` - genuinely different types, hence two
/// variants rather than one shared field.
///
/// SERIALIZATION-CRITICAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureRecipe {
    Pixel {
        feature_spec: FeatureSpec,
        class_labels: Vec<SegmentationClass>,
    },
    Object {
        feature_spec: ObjectFeatureSpec,
        class_labels: Vec<ObjectClass>,
    },
}

/// The full artifact saved to `<project>/models/<name>`: the trained model
/// plus which features it expects and what its predicted indices mean.
///
/// SERIALIZATION-CRITICAL: `version` is the escape hatch for future format
/// changes — never change the meaning of an existing field for a given
/// version, bump `version` instead.
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedClassifier {
    pub version: u32,
    pub classifier: Classifier,
    pub feature_recipe: FeatureRecipe,
}

pub const CURRENT_SAVED_CLASSIFIER_VERSION: u32 = 1;

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

// -- Shared training helpers, backend-agnostic over how `rows`/`labels` were
// gathered (pixel samples or object feature vectors) --------------------

fn to_dense_matrix(rows: &[Vec<f32>]) -> Result<DenseMatrix<f32>, InternalErrors> {
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    DenseMatrix::from_2d_array(&row_refs).map_err(|e| InternalErrors::Internal(e.to_string()))
}

fn validate_training_data(rows: &[Vec<f32>], labels: &[usize]) -> Result<(), InternalErrors> {
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

pub fn fit_random_forest(
    rows: &[Vec<f32>],
    labels: &[usize],
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.to_vec();
    let model = RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters::default())
        .map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok(Classifier::RandomForest(model))
}

pub fn fit_knn(rows: &[Vec<f32>], labels: &[usize]) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let x = to_dense_matrix(rows)?;
    let y = labels.to_vec();
    let model = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default())
        .map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok(Classifier::Knn(model))
}

/// Trains a small feed-forward classifier via a manual training loop (no
/// `Learner`/dashboard — appropriate for a model this size, per burn's own
/// "Custom Training Loop" guidance) using Adam and cross-entropy loss.
pub fn fit_mlp(
    rows: &[Vec<f32>],
    labels: &[usize],
    hidden_layers: &[usize],
    epochs: usize,
    learning_rate: f64,
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let n_in = rows[0].len();
    let n_out = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_in == 0 || n_out == 0 {
        return Err(InternalErrors::Internal(
            "cannot train MLP with zero input features or output classes".to_string(),
        ));
    }

    let device = mlp_device();
    let n_rows = rows.len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let x = Tensor::<MlpBackend, 2>::from_data(TensorData::new(flat, [n_rows, n_in]), &device);
    let y_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    let y = Tensor::<MlpBackend, 1, Int>::from_data(TensorData::new(y_i64, [n_rows]), &device);

    let mut model = Mlp::<MlpBackend>::new(&device, n_in, hidden_layers, n_out);
    let mut optim = AdamConfig::new().init();

    for _ in 0..epochs {
        let out = model.forward(x.clone());
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out, y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(learning_rate, model, grads);
    }

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let weights = recorder
        .record(model.into_record(), ())
        .map_err(|e| InternalErrors::Internal(format!("failed to record MLP weights: {e:?}")))?;

    Ok(Classifier::Mlp {
        architecture: MlpArchitecture {
            n_in,
            hidden: hidden_layers.to_vec(),
            n_out,
        },
        weights,
    })
}
