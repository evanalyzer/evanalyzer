use crate::ai_learning::model::Classifier;
use crate::ai_learning::training_job::{TrainingProgressEvent, TrainingStats};
use crate::ai_learning::utils::validate_training_data;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::{Int, Tensor, TensorData};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::ai_learning_settings::{MlpActivation, MlpSettings};
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

/// A small feed-forward classifier: `Linear` layers separated by a
/// configurable activation, sized by `MlpArchitecture`. Verified against burn
/// 0.21's real API (module definition, forward pass, `BinBytesRecorder`
/// save/load) rather than assumed, since burn's API has shifted significantly
/// across versions.
///
/// The activation isn't a field: burn's `nn::{Relu, Sigmoid, Tanh}` are all
/// separate zero-field `Module` types (not one type with a runtime choice),
/// and it's not a learnable parameter anyway, so it's simplest to just pass
/// `MlpActivation` into `forward` and dispatch to the matching
/// `burn::tensor::activation` free function - see `MlpArchitecture::activation`
/// for how a saved model remembers which one it was trained with.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
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
        Self { layers }
    }

    fn forward(&self, x: Tensor<B, 2>, activation: MlpActivation) -> Tensor<B, 2> {
        let n = self.layers.len();
        self.layers.iter().enumerate().fold(x, |x, (i, l)| {
            let x = l.forward(x);
            if i + 1 < n {
                match activation {
                    MlpActivation::Relu => relu(x),
                    MlpActivation::Sigmoid => sigmoid(x),
                    MlpActivation::Tanh => tanh(x),
                }
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
    /// The activation the model was *trained* with - `Mlp::forward` needs it
    /// at inference time too (see `Mlp`'s doc comment for why it isn't a
    /// module field), and using a different one than training would silently
    /// produce garbage predictions rather than an error. `#[serde(default)]`
    /// so a model saved before this field existed still loads (as
    /// `MlpActivation::Relu`, matching the hardcoded activation those models
    /// were actually trained with).
    #[serde(default)]
    pub activation: MlpActivation,
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

    let out = model.forward(x, architecture.activation);
    let preds = out.argmax(1);
    let preds_data = preds
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|e| InternalErrors::Internal(format!("failed to read MLP predictions: {e:?}")))?;
    Ok(preds_data.into_iter().map(|v| v as usize).collect())
}

/// Deterministically shuffles `0..n` (xorshift64, keyed by `seed`).
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed.max(1);
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

/// Splits off the first `val_fraction` of a `shuffled_indices(n, seed)` as a
/// held-out validation set, the rest as training indices.
///
/// This is a plain random split, not stratified by class and not aware of
/// which rows came from the same source image/object — for `PixelTrainingJob`
/// in particular, neighboring pixels end up on both sides of the split, so
/// the resulting validation loss is a rough in-run generalization signal for
/// the training-progress banner, not a rigorous held-out evaluation.
fn train_val_split(n: usize, seed: u64, val_fraction: f64) -> (Vec<usize>, Vec<usize>) {
    let indices = shuffled_indices(n, seed);
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
/// Each epoch reshuffles the training rows (deterministically, keyed by
/// `settings.seed` and the epoch number - see `shuffled_indices`) and steps
/// the optimizer once per `settings.batch_size`-sized chunk, so an "epoch" is
/// still one full pass over the training data but is now `⌈n / batch_size⌉`
/// gradient updates instead of one full-batch update. `train_loss` reported
/// per epoch is the sample-count-weighted mean loss across that epoch's
/// batches (the last batch is often smaller than the rest).
///
/// TODO: `settings.seed` is used for the train/val split and per-epoch batch
/// shuffling above but not yet for weight initialization (burn's
/// `LinearConfig::init` doesn't take a seed directly), so two runs with the
/// same seed still get different starting weights.
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

    let val_tensors = (!val_idx.is_empty()).then(|| {
        to_tensors(
            &gather_rows(rows, &val_idx),
            &gather_labels(labels, &val_idx),
            n_in,
            &device,
        )
    });

    let mut model = Mlp::<MlpBackend>::new(&device, n_in, &settings.hidden_layers, n_classes);
    let mut optim = AdamConfig::new()
        .with_epsilon(settings.epsilon as f32)
        .init();
    let batch_size = settings.batch_size.max(1);

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

        // Reshuffle which rows land in which batch every epoch (a different,
        // but still deterministic-for-this-seed, order each time) rather
        // than cycling through the same fixed batches every pass.
        let epoch_order = shuffled_indices(
            train_idx.len(),
            settings.seed.wrapping_add(epoch as u64 + 1),
        );

        let mut loss_sum = 0.0f32;
        let mut samples_seen = 0usize;
        for batch in epoch_order.chunks(batch_size) {
            let batch_row_idx: Vec<usize> = batch.iter().map(|&local| train_idx[local]).collect();
            let (x_batch, y_batch) = to_tensors(
                &gather_rows(rows, &batch_row_idx),
                &gather_labels(labels, &batch_row_idx),
                n_in,
                &device,
            );

            let out = model.forward(x_batch, settings.activation);
            let loss = CrossEntropyLossConfig::new()
                .init(&out.device())
                .forward(out, y_batch);
            let batch_loss = loss.clone().into_scalar();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(settings.learning_rate, model, grads);

            loss_sum += batch_loss * batch_row_idx.len() as f32;
            samples_seen += batch_row_idx.len();
        }
        let train_loss = loss_sum / samples_seen.max(1) as f32;

        let val_loss = val_tensors.as_ref().map(|(x_val, y_val)| {
            let out_val = model.forward(x_val.clone(), settings.activation);
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
            activation: settings.activation,
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
        (
            std::sync::mpsc::channel().0,
            Arc::new(AtomicBool::new(false)),
        )
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

    /// Every activation must be reachable end to end (settings -> training
    /// -> `MlpArchitecture` -> inference) and still able to separate two
    /// well-separated clusters - not just compile. Also covers that
    /// `predict_mlp` replays the *trained* activation rather than a
    /// hardcoded one (a mismatch here would still often "work" for such an
    /// easy dataset, but `architecture.activation` is asserted directly so a
    /// regression can't hide behind that).
    #[test]
    fn fit_mlp_separates_clusters_under_every_activation() {
        let (rows, labels) = two_cluster_dataset();
        for activation in [
            MlpActivation::Relu,
            MlpActivation::Sigmoid,
            MlpActivation::Tanh,
        ] {
            let settings = MlpSettings {
                hidden_layers: vec![4],
                epochs: 300,
                learning_rate: 0.05,
                activation,
                ..Default::default()
            };
            let (progress, cancel) = no_op_progress();
            let (classifier, _stats) = fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel)
                .unwrap_or_else(|e| panic!("fit_mlp failed for {activation:?}: {e:?}"));

            let Classifier::Mlp { architecture, .. } = &classifier else {
                panic!("expected an Mlp classifier for {activation:?}");
            };
            assert_eq!(
                architecture.activation, activation,
                "the saved architecture must remember which activation it was trained with"
            );

            let predictions = classifier
                .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
                .unwrap();
            assert_eq!(
                predictions,
                vec![0, 1],
                "activation {activation:?} failed to separate two well-separated clusters"
            );
        }
    }

    /// A `batch_size` that doesn't evenly divide the training set (leaving a
    /// smaller final batch each epoch) must still train and predict
    /// correctly - the main risk being an off-by-one in the
    /// `chunks(batch_size)` loop or the sample-count-weighted loss average.
    #[test]
    fn fit_mlp_trains_correctly_with_a_batch_size_that_does_not_evenly_divide_the_data() {
        let (rows, labels) = two_cluster_dataset();
        assert_eq!(rows.len(), 30, "test assumes the fixture's known size");
        let settings = MlpSettings {
            hidden_layers: vec![4],
            epochs: 300,
            learning_rate: 0.05,
            batch_size: 7, // 30 rows / 7 = 4 batches of [7, 7, 7, 3] once val split removes some
            ..Default::default()
        };

        let (progress, cancel) = no_op_progress();
        let (classifier, _stats) =
            fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap();

        let predictions = classifier
            .predict(&[vec![0.05, 0.05], vec![10.05, 10.05]])
            .unwrap();
        assert_eq!(predictions, vec![0, 1]);
    }

    /// A batch size of 0 (e.g. an unpopulated/default settings value) must
    /// be treated as 1 (pure per-sample SGD), not panic in `chunks(0)`.
    #[test]
    fn fit_mlp_treats_a_zero_batch_size_as_one_instead_of_panicking() {
        let (rows, labels) = two_cluster_dataset();
        let settings = MlpSettings {
            hidden_layers: vec![4],
            epochs: 5,
            learning_rate: 0.05,
            batch_size: 0,
            ..Default::default()
        };
        let (progress, cancel) = no_op_progress();
        let result = fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel);
        assert!(result.is_ok());
    }

    /// Adam divides its update by `sqrt(v_hat) + epsilon`, so a large enough
    /// epsilon should visibly suppress how much the loss moves over a fixed,
    /// small number of epochs compared to a tiny one. If `settings.epsilon`
    /// were silently ignored (e.g. always falling back to burn's own
    /// `AdamConfig` default), both runs would train identically and this
    /// would fail to observe any difference.
    #[test]
    fn fit_mlp_a_large_epsilon_suppresses_training_progress_relative_to_a_tiny_one() {
        let (rows, labels) = two_cluster_dataset();

        let run = |epsilon: f64| {
            let settings = MlpSettings {
                hidden_layers: vec![4],
                epochs: 20,
                learning_rate: 0.05,
                seed: 1,
                epsilon,
                ..Default::default()
            };
            let (progress, cancel) = no_op_progress();
            let (_classifier, stats) =
                fit_mlp(&rows, &labels, 2, &settings, &progress, &cancel).unwrap();
            let TrainingStats::Mlp {
                final_train_loss, ..
            } = stats
            else {
                panic!("expected Mlp stats");
            };
            final_train_loss
        };

        let tiny_epsilon_loss = run(1e-8);
        let large_epsilon_loss = run(10.0);

        // Cross-entropy loss for a random 2-class init starts around ln(2) ≈
        // 0.69 - a large epsilon should keep it much closer to that starting
        // point than a tiny one does after the same 20 epochs.
        let initial_loss = 2f32.ln();
        assert!(
            (large_epsilon_loss - initial_loss).abs() < (tiny_epsilon_loss - initial_loss).abs(),
            "large epsilon ({large_epsilon_loss}) should have moved less from the \
             initial loss ({initial_loss}) than a tiny epsilon ({tiny_epsilon_loss}) did"
        );
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
        let err = fit_mlp(
            &rows,
            &labels,
            0,
            &MlpSettings::default(),
            &progress,
            &cancel,
        )
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
            events.iter().any(|e| matches!(
                e,
                TrainingProgressEvent::Epoch {
                    epoch: 9,
                    total_epochs: 10,
                    ..
                }
            )),
            "the last epoch must always be reported even if it doesn't land on report_every"
        );
    }
}
