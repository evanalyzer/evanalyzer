use crate::ai_learning::model::Classifier;
use crate::ai_learning::training_job::{TrainingProgressEvent, TrainingStats};
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

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

/// Deterministically shuffles `0..n` (xorshift64, keyed by `seed` — reusing
/// `MlpSettings::seed`, previously unused, see this module's earlier TODO)
/// and splits off the first `val_fraction` as a held-out validation set, the
/// rest as training indices.
///
/// This is a plain random split, not stratified by class and not aware of
/// which rows came from the same source image/object — for `PixelTrainingJob`
/// in particular, neighboring pixels end up on both sides of the split, so
/// the resulting validation loss is a rough in-run generalization signal for
/// the training-progress banner, not a rigorous held-out evaluation.
fn train_val_split(n: usize, seed: u64, val_fraction: f64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed.max(1);
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    let n_val = ((n as f64) * val_fraction).round() as usize;
    let val = indices[..n_val].to_vec();
    let train = indices[n_val..].to_vec();
    (train, val)
}

/// Below this many total samples, `fit_mlp` skips the train/val split
/// entirely (trains on everything, reports no validation loss) rather than
/// carve out a validation set too small to mean anything.
const MIN_SAMPLES_FOR_VAL_SPLIT: usize = 25;
const VAL_FRACTION: f64 = 0.2;

fn gather_rows(rows: &[Vec<f32>], indices: &[usize]) -> Vec<Vec<f32>> {
    indices.iter().map(|&i| rows[i].clone()).collect()
}

fn gather_labels(labels: &[usize], indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| labels[i]).collect()
}

fn to_tensors(
    rows: &[Vec<f32>],
    labels: &[usize],
    n_in: usize,
    device: &NdArrayDevice,
) -> (Tensor<MlpBackend, 2>, Tensor<MlpBackend, 1, Int>) {
    let n_rows = rows.len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let x = Tensor::<MlpBackend, 2>::from_data(TensorData::new(flat, [n_rows, n_in]), device);
    let y_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    let y = Tensor::<MlpBackend, 1, Int>::from_data(TensorData::new(y_i64, [n_rows]), device);
    (x, y)
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
/// When there are enough samples (`MIN_SAMPLES_FOR_VAL_SPLIT`), a fraction
/// (`VAL_FRACTION`) is held out via `train_val_split` and never trained on —
/// its loss each epoch is what lets the caller (and, via `progress`, the GUI)
/// see overfitting as it happens (training loss still falling while
/// validation loss climbs), not just whether training loss is converging.
/// The saved model is fit only on the remaining training rows in that case,
/// trading a bit of final model quality for a genuine generalization signal.
///
/// `progress` gets one `TrainingProgressEvent::Epoch` roughly every
/// `total_epochs / 200` epochs (always including the last), and `cancel` is
/// checked every epoch so a long MLP run can actually be interrupted — unlike
/// `fit_random_forest`/`fit_knn`, which fit in one blocking call with no
/// per-iteration hook for either.
///
/// TODO: `settings.activation`/`batch_size` are not wired in yet — training
/// always uses ReLU and full-batch gradient descent. `settings.seed` is used
/// for the train/val split above but not yet for weight initialization
/// (burn's `LinearConfig::init` doesn't take a seed directly), so two runs
/// with the same seed still get different starting weights.
pub fn fit_mlp(
    rows: &[Vec<f32>],
    labels: &[usize],
    n_classes: usize,
    settings: &MlpSettings,
    progress: &Sender<TrainingProgressEvent>,
    cancel: &Arc<AtomicBool>,
) -> Result<(Classifier, TrainingStats), InternalErrors> {
    validate_training_data(rows, labels)?;
    let n_in = rows[0].len();
    if n_in == 0 || n_classes == 0 {
        return Err(InternalErrors::Internal(
            "cannot train MLP with zero input features or output classes".to_string(),
        ));
    }

    let device = mlp_device();
    let n = rows.len();
    let (train_idx, val_idx) = if n >= MIN_SAMPLES_FOR_VAL_SPLIT {
        train_val_split(n, settings.seed, VAL_FRACTION)
    } else {
        ((0..n).collect(), Vec::new())
    };

    let (x, y) = to_tensors(
        &gather_rows(rows, &train_idx),
        &gather_labels(labels, &train_idx),
        n_in,
        &device,
    );
    let val_tensors = (!val_idx.is_empty()).then(|| {
        to_tensors(
            &gather_rows(rows, &val_idx),
            &gather_labels(labels, &val_idx),
            n_in,
            &device,
        )
    });

    let mut model = Mlp::<MlpBackend>::new(&device, n_in, &settings.hidden_layers, n_classes);
    let mut optim = AdamConfig::new().init();

    // Cap how many `Epoch` events get sent for a long run - the GUI only
    // needs a smooth-looking curve, not one message per epoch.
    let report_every = (settings.epochs / 200).max(1);

    let mut final_train_loss = 0.0f32;
    let mut final_val_loss: Option<f32> = None;
    let mut best_val_loss: Option<f32> = None;
    let mut best_val_epoch: Option<usize> = None;
    let mut epochs_run = 0usize;

    for epoch in 0..settings.epochs {
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }

        let out = model.forward(x.clone());
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out, y.clone());
        let train_loss = loss.clone().into_scalar();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(settings.learning_rate, model, grads);

        let val_loss = val_tensors.as_ref().map(|(x_val, y_val)| {
            let out_val = model.forward(x_val.clone());
            let loss_val = CrossEntropyLossConfig::new()
                .init(&out_val.device())
                .forward(out_val, y_val.clone());
            loss_val.into_scalar()
        });

        if let Some(v) = val_loss
            && best_val_loss.is_none_or(|best| v < best)
        {
            best_val_loss = Some(v);
            best_val_epoch = Some(epoch);
        }

        epochs_run = epoch + 1;
        final_train_loss = train_loss;
        final_val_loss = val_loss;

        let is_last = epoch + 1 == settings.epochs;
        if epoch % report_every == 0 || is_last {
            let _ = progress.send(TrainingProgressEvent::Epoch {
                epoch,
                total_epochs: settings.epochs,
                train_loss,
                val_loss,
            });
        }
    }

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let weights = recorder
        .record(model.into_record(), ())
        .map_err(|e| InternalErrors::Internal(format!("failed to record MLP weights: {e:?}")))?;

    let classifier = Classifier::Mlp {
        architecture: MlpArchitecture {
            n_in,
            hidden: settings.hidden_layers.clone(),
            n_out: n_classes,
        },
        weights,
    };
    let stats = TrainingStats::Mlp {
        epochs_run,
        total_epochs: settings.epochs,
        final_train_loss,
        final_val_loss,
        best_val_loss,
        best_val_epoch,
    };
    Ok((classifier, stats))
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

    fn no_op_progress() -> (Sender<TrainingProgressEvent>, Arc<AtomicBool>) {
        (std::sync::mpsc::channel().0, Arc::new(AtomicBool::new(false)))
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

        let (progress, cancel) = no_op_progress();
        let (classifier, stats) =
            fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap();

        let predictions = classifier
            .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);

        // 30 samples clears MIN_SAMPLES_FOR_VAL_SPLIT, so a validation split
        // must have happened.
        let TrainingStats::Mlp {
            epochs_run,
            total_epochs,
            final_val_loss,
            best_val_loss,
            best_val_epoch,
            ..
        } = stats
        else {
            panic!("expected Mlp stats");
        };
        assert_eq!(epochs_run, 300);
        assert_eq!(total_epochs, 300);
        assert!(final_val_loss.is_some());
        assert!(best_val_loss.is_some());
        assert!(best_val_epoch.is_some());
    }

    #[test]
    fn fit_mlp_rejects_zero_samples() {
        let (progress, cancel) = no_op_progress();
        let err = fit_mlp(&[], &[], 2, &MlpSettings::default(), &progress, &cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }

    #[test]
    fn fit_mlp_rejects_zero_classes() {
        let (rows, labels) = two_cluster_dataset();
        let (progress, cancel) = no_op_progress();
        let err = fit_mlp(&rows, &labels, 0, &MlpSettings::default(), &progress, &cancel)
            .unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero"));
    }

    #[test]
    fn fit_mlp_skips_the_val_split_below_the_minimum_sample_count() {
        // Only 4 samples - well under MIN_SAMPLES_FOR_VAL_SPLIT - so every
        // row must be used for training and no validation loss reported.
        let rows = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let labels = vec![0, 0, 1, 1];
        let settings = MlpSettings {
            hidden_layers: vec![2],
            epochs: 5,
            learning_rate: 0.05,
            ..Default::default()
        };
        let (progress, cancel) = no_op_progress();
        let (_classifier, stats) =
            fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap();

        let TrainingStats::Mlp {
            final_val_loss,
            best_val_loss,
            ..
        } = stats
        else {
            panic!("expected Mlp stats");
        };
        assert!(final_val_loss.is_none());
        assert!(best_val_loss.is_none());
    }

    #[test]
    fn fit_mlp_stops_early_and_returns_cancelled_when_the_flag_is_set() {
        let (rows, labels) = two_cluster_dataset();
        let settings = MlpSettings {
            hidden_layers: vec![4],
            epochs: 1000,
            learning_rate: 0.05,
            ..Default::default()
        };
        let (progress, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(true));

        let err = fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Cancelled));
    }

    #[test]
    fn fit_mlp_reports_epoch_progress_including_the_final_epoch() {
        let (rows, labels) = two_cluster_dataset();
        let settings = MlpSettings {
            hidden_layers: vec![4],
            epochs: 10,
            learning_rate: 0.05,
            ..Default::default()
        };
        let (progress, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap();
        let events: Vec<_> = rx.try_iter().collect();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::Epoch { epoch: 9, total_epochs: 10, .. })),
            "the last epoch must always be reported even if it doesn't land on report_every"
        );
    }
}
