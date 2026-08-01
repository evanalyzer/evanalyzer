use crate::ai_learning::model::Classifier;
use crate::ai_learning::utils::validate_training_data;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Int, Tensor, TensorData};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::MlpSettings;
use serde::{Deserialize, Serialize};

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

/// The tensor -> `Vec<usize>` extraction (argmax + data readout) is covered
/// by `fit_mlp_separates_two_well_separated_clusters` below, which round-trips
/// a real trained model through this function against burn 0.21.
pub(crate) fn predict_mlp(
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

/// Trains a small feed-forward classifier via a manual training loop (no
/// `Learner`/dashboard — appropriate for a model this size, per burn's own
/// "Custom Training Loop" guidance) using Adam and cross-entropy loss.
///
/// `labels` are dense indices into the owning job's `class_labels` (see
/// `ai_learning::utils::validate_training_data`'s doc comment) — `n_classes`
/// sizes the output layer explicitly instead of inferring it from
/// `max(labels) + 1`, which silently undersizes the network if the
/// highest-index class happens to have zero training samples.
///
/// TODO: `settings.activation`/`batch_size`/`seed` are not wired in yet —
/// training always uses ReLU, full-batch gradient descent, and no seed
/// control. Only `hidden_layers`, `epochs` and `learning_rate` are used.
pub fn fit_mlp(
    rows: &[Vec<f32>],
    labels: &[usize],
    n_classes: usize,
    settings: &MlpSettings,
) -> Result<Classifier, InternalErrors> {
    validate_training_data(rows, labels)?;
    let n_in = rows[0].len();
    if n_in == 0 || n_classes == 0 {
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

    let mut model = Mlp::<MlpBackend>::new(&device, n_in, &settings.hidden_layers, n_classes);
    let mut optim = AdamConfig::new().init();

    for _ in 0..settings.epochs {
        let out = model.forward(x.clone());
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out, y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(settings.learning_rate, model, grads);
    }

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let weights = recorder
        .record(model.into_record(), ())
        .map_err(|e| InternalErrors::Internal(format!("failed to record MLP weights: {e:?}")))?;

    Ok(Classifier::Mlp {
        architecture: MlpArchitecture {
            n_in,
            hidden: settings.hidden_layers.clone(),
            n_out: n_classes,
        },
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters - see `model::random_forest`'s test
    /// module doc comment for why this is a wiring smoke test, not a
    /// generalization one. Also the first real exercise of `predict_mlp`'s
    /// argmax/tensor-readout path against burn 0.21 - see its own doc
    /// comment, which flagged that path as unverified.
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
    fn fit_mlp_separates_two_well_separated_clusters() {
        let (rows, labels) = two_cluster_dataset();
        let settings = MlpSettings {
            hidden_layers: vec![4],
            epochs: 300,
            learning_rate: 0.05,
            ..Default::default()
        };

        let classifier = fit_mlp(&rows, &labels, 2, &settings).unwrap();

        let predictions = classifier
            .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);
    }

    #[test]
    fn fit_mlp_rejects_zero_samples() {
        let err = fit_mlp(&[], &[], 2, &MlpSettings::default()).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }

    #[test]
    fn fit_mlp_rejects_zero_classes() {
        let (rows, labels) = two_cluster_dataset();
        let err = fit_mlp(&rows, &labels, 0, &MlpSettings::default()).unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero"));
    }
}
