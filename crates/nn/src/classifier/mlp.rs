//! # mlp
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//! A small feed-forward classifier (burn): `Linear` layers with an
//! activation between them, trained with Adam on cross-entropy. Moved from
//! core's `ai_learning/model/mlp.rs` (burn 0.21, 2026-11-25) with the same
//! training algorithm and its tests.
//!
//! Saved as `config.rs` describes: [`MlpArchitecture`] plus the weights as
//! safetensors bytes (D4 of the burn migration plan: never in burn's own
//! formats).

use crate::classifier::{Classifier, Predictions, Samples, row_count};
use crate::errors::NnError;
use crate::train::{SeededUniform, TrainControl, TrainProgress, TrainStats};
use crate::weights::{ModelFile, ModelMetadata, NamedTensor};
use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::{Device, Int, Tensor, TensorData};
use rayon::prelude::*;
use std::sync::OnceLock;

/// Rows per block of a prediction - a block's layer outputs are what is
/// held at once (rows x widest layer floats), as in the random forest.
const PREDICT_CHUNK_ROWS: usize = 4096;

/// Below this many samples there is no train/validation split (training on
/// everything, no validation loss) - a smaller validation set means nothing.
const MIN_SAMPLES_FOR_VAL_SPLIT: usize = 25;
const VAL_FRACTION: f64 = 0.2;

#[derive(Debug, Clone, PartialEq)]
pub struct MlpParams {
    /// Neurons per hidden layer.
    pub hidden_layers: Vec<usize>,
    pub activation: MlpActivation,
    pub epochs: usize,
    pub learning_rate: f64,
    /// Rows per gradient step (0 = 1).
    pub batch_size: usize,
    /// Adam's epsilon (0 makes Adam divide 0 by 0).
    pub epsilon: f64,
}

impl Default for MlpParams {
    /// No layers, no epochs - and Adam's epsilon 1e-5, as the settings'
    /// default.
    fn default() -> Self {
        Self {
            hidden_layers: Vec::new(),
            activation: MlpActivation::default(),
            epochs: 0,
            learning_rate: 0.0,
            batch_size: 0,
            epsilon: 1e-5,
        }
    }
}

impl MlpArchitecture {
    /// `(inputs, outputs)` of each layer.
    fn layer_sizes(&self) -> Vec<(usize, usize)> {
        let mut sizes = vec![self.n_in];
        sizes.extend_from_slice(&self.hidden);
        sizes.push(self.n_out);
        sizes.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// Refuses sizes no trained MLP has (the file is untrusted).
    fn check(&self) -> Result<(), NnError> {
        let ok = |n: usize| (1..=1 << 16).contains(&n);
        if !ok(self.n_in)
            || !ok(self.n_out)
            || self.hidden.len() > 32
            || !self.hidden.iter().all(|&n| ok(n))
        {
            return Err(NnError::InvalidModelFile(format!(
                "MLP layer sizes out of range: {self:?}"
            )));
        }
        Ok(())
    }
}

mod config;

use config::{ARCH, ARCH_VERSION, bias_name, weight_name};
pub use config::{Mlp, MlpActivation, MlpArchitecture};

/// The network: `Linear` layers, the activation between them.
#[derive(Module, Debug)]
pub(super) struct Network {
    layers: Vec<Linear>,
}

impl Network {
    /// Fresh layers, initialized as burn's `Linear` default (Kaiming uniform,
    /// gain 1/sqrt(3): weights and biases uniform in ±1/sqrt(inputs)) - but
    /// from a generator of its own seeded with `seed` ([`SeededUniform`]),
    /// not burn's global one: whatever else draws random numbers meanwhile,
    /// the same seed gives the same model.
    fn new(architecture: &MlpArchitecture, seed: u64, device: &Device) -> Self {
        let mut random = SeededUniform::new(seed);
        let mut uniform = |n: usize, bound: f32| random.values(n, bound);
        let layers = architecture
            .layer_sizes()
            .into_iter()
            .map(|(n_in, n_out)| {
                let bound = 1.0 / (n_in as f32).sqrt();
                let weight = uniform(n_in * n_out, bound);
                let bias = uniform(n_out, bound);
                Linear {
                    weight: Param::from_tensor(Tensor::from_data(
                        TensorData::new(weight, [n_in, n_out]),
                        device,
                    )),
                    bias: Some(Param::from_tensor(Tensor::from_data(
                        TensorData::new(bias, [n_out]),
                        device,
                    ))),
                }
            })
            .collect();
        Self { layers }
    }

    /// Raw class scores.
    fn forward(&self, x: Tensor<2>, activation: MlpActivation) -> Tensor<2> {
        let n = self.layers.len();
        self.layers.iter().enumerate().fold(x, |x, (i, layer)| {
            let x = layer.forward(x);
            if i + 1 == n {
                return x;
            }
            match activation {
                MlpActivation::Relu => relu(x),
                MlpActivation::Sigmoid => sigmoid(x),
                MlpActivation::Tanh => tanh(x),
            }
        })
    }

    /// The weights as `layer{i}.weight` `[out, in]` and `layer{i}.bias`.
    fn tensors(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let weight = layer.weight.val().transpose();
            out.push(NamedTensor {
                name: weight_name(i),
                shape: weight.dims().to_vec(),
                values: weight.into_data().try_to_vec::<f32>().expect("f32 weights"),
            });
            if let Some(bias) = &layer.bias {
                let bias = bias.val();
                out.push(NamedTensor {
                    name: bias_name(i),
                    shape: bias.dims().to_vec(),
                    values: bias.into_data().try_to_vec::<f32>().expect("f32 weights"),
                });
            }
        }
        out
    }

    /// A network of `architecture` with the weights of `bytes` (see the
    /// module docs), every shape checked.
    fn from_bytes(
        architecture: &MlpArchitecture,
        bytes: &[u8],
        device: &Device,
    ) -> Result<Self, NnError> {
        architecture.check()?;
        let file = ModelFile::parse(bytes)
            .map_err(|e| NnError::InvalidModelFile(format!("MLP weights: {e}")))?;
        file.expect_arch(ARCH, ARCH_VERSION)?;
        let read = |name: &str, shape: &[usize]| -> Result<Vec<f32>, NnError> {
            let view = file
                .tensors
                .tensor(name)
                .map_err(|_| NnError::InvalidModelFile(format!("the tensor {name} is missing")))?;
            if view.shape() != shape || view.dtype() != safetensors::Dtype::F32 {
                return Err(NnError::InvalidModelFile(format!(
                    "the tensor {name} is {:?} {:?}, F32 {shape:?} expected",
                    view.dtype(),
                    view.shape()
                )));
            }
            Ok(view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        let mut layers = Vec::new();
        for (i, (n_in, n_out)) in architecture.layer_sizes().into_iter().enumerate() {
            let weight = read(&weight_name(i), &[n_out, n_in])?;
            let bias = read(&bias_name(i), &[n_out])?;
            let weight =
                Tensor::<2>::from_data(TensorData::new(weight, [n_out, n_in]), device).transpose();
            let bias = Tensor::<1>::from_data(TensorData::new(bias, [n_out]), device);
            layers.push(Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            });
        }
        Ok(Self { layers })
    }
}

/// The CPU device predictions and training run on.
fn cpu() -> Device {
    Device::flex()
}

impl Mlp {
    fn network(&self) -> Result<&Network, NnError> {
        if let Some(network) = self.network.get() {
            return Ok(network);
        }
        let network = Network::from_bytes(&self.architecture, &self.weights, &cpu())?;
        Ok(self.network.get_or_init(|| network))
    }

    /// Raw class scores of a block of rows.
    fn scores(&self, network: &Network, rows: &[f32]) -> Tensor<2> {
        let n = rows.len() / self.architecture.n_in;
        let x = Tensor::<2>::from_data(
            TensorData::new(rows.to_vec(), [n, self.architecture.n_in]),
            &cpu(),
        );
        network.forward(x, self.architecture.activation)
    }

    fn check_rows(&self, rows: &[f32], n_features: usize) -> Result<usize, NnError> {
        if n_features != self.architecture.n_in {
            return Err(NnError::InvalidInput(format!(
                "the MLP takes {} features, {n_features} given",
                self.architecture.n_in
            )));
        }
        row_count(rows, n_features)
    }
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

/// The first `val_fraction` of `shuffled_indices(n, seed)` held out for
/// validation, the rest for training. A plain random split, not stratified
/// and not aware of which rows came from the same image or object - a rough
/// in-run generalization signal for the progress banner.
fn train_val_split(n: usize, seed: u64, val_fraction: f64) -> (Vec<usize>, Vec<usize>) {
    let indices = shuffled_indices(n, seed);
    let n_val = ((n as f64) * val_fraction).round() as usize;
    (indices[n_val..].to_vec(), indices[..n_val].to_vec())
}

/// Rows and labels of `indices` as tensors on `device`.
fn to_tensors(
    samples: &Samples<'_>,
    indices: &[usize],
    device: &Device,
) -> (Tensor<2>, Tensor<1, Int>) {
    let nf = samples.n_features;
    let mut x = Vec::with_capacity(indices.len() * nf);
    for &i in indices {
        x.extend_from_slice(&samples.rows[i * nf..(i + 1) * nf]);
    }
    let y: Vec<i64> = indices.iter().map(|&i| samples.labels[i] as i64).collect();
    (
        Tensor::<2>::from_data(TensorData::new(x, [indices.len(), nf]), device),
        Tensor::<1, Int>::from_data(TensorData::new(y, [indices.len()]), device),
    )
}

impl Classifier for Mlp {
    type Params = MlpParams;

    /// A manual training loop (burn's "custom training loop") with Adam and
    /// cross-entropy:
    ///
    /// - with enough samples ([`MIN_SAMPLES_FOR_VAL_SPLIT`]) a fraction
    ///   ([`VAL_FRACTION`]) is held out and never trained on - its loss each
    ///   epoch shows overfitting as it happens;
    /// - each epoch reshuffles the training rows (keyed by the seed and the
    ///   epoch) and steps once per `batch_size` rows; its `train_loss` is
    ///   the sample-weighted mean over its batches;
    /// - the seed also seeds the weight initialization (`Network::new`, its
    ///   own generator): the same seed and data give the same model;
    /// - progress roughly every `epochs / 200` epochs (always the last),
    ///   cancel checked every epoch.
    ///
    /// The output layer has `samples.n_classes` outputs, not
    /// `max(label) + 1` (a class without samples would shrink it).
    fn train(
        samples: &Samples<'_>,
        params: &MlpParams,
        control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError> {
        samples.validate()?;
        let (n_in, n_classes) = (samples.n_features, samples.n_classes);
        if n_classes == 0 {
            return Err(NnError::InvalidInput(
                "cannot train an MLP without output classes".into(),
            ));
        }
        let architecture = MlpArchitecture {
            n_in,
            hidden: params.hidden_layers.clone(),
            n_out: n_classes,
            activation: params.activation,
        };
        architecture.check()?;
        let seed = control.seed;
        let device = cpu().autodiff();
        let n = samples.n_samples();
        let (train_idx, val_idx) = if n >= MIN_SAMPLES_FOR_VAL_SPLIT {
            train_val_split(n, seed, VAL_FRACTION)
        } else {
            ((0..n).collect(), Vec::new())
        };
        let val = (!val_idx.is_empty()).then(|| to_tensors(samples, &val_idx, &device));

        let mut model = Network::new(&architecture, seed, &device);
        let mut optim = AdamConfig::new().with_epsilon(params.epsilon as f32).init();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let batch_size = params.batch_size.max(1);
        let report_every = (params.epochs / 200).max(1);

        let (mut final_train_loss, mut final_val_loss) = (0.0f32, None);
        let (mut best_val_loss, mut best_val_epoch): (Option<f32>, Option<usize>) = (None, None);
        let mut epochs_run = 0;
        for epoch in 0..params.epochs {
            if control.is_cancelled() {
                return Err(NnError::Cancelled);
            }
            let order = shuffled_indices(train_idx.len(), seed.wrapping_add(epoch as u64 + 1));
            let (mut loss_sum, mut seen) = (0.0f32, 0usize);
            for batch in order.chunks(batch_size) {
                let rows: Vec<usize> = batch.iter().map(|&local| train_idx[local]).collect();
                let (x, y) = to_tensors(samples, &rows, &device);
                let loss = loss_fn.forward(model.forward(x, params.activation), y);
                let batch_loss: f32 = loss.clone().into_scalar();
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(params.learning_rate, model, grads);
                loss_sum += batch_loss * rows.len() as f32;
                seen += rows.len();
            }
            let train_loss = loss_sum / seen.max(1) as f32;
            let val_loss = val.as_ref().map(|(x, y)| {
                let loss = loss_fn.forward(model.forward(x.clone(), params.activation), y.clone());
                loss.into_scalar::<f32>()
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
            if epoch % report_every == 0 || epoch + 1 == params.epochs {
                (control.progress)(TrainProgress::Epoch {
                    epoch,
                    total_epochs: params.epochs,
                    train_loss,
                    val_loss,
                });
            }
        }

        let network = model.valid();
        let metadata = ModelMetadata {
            arch: ARCH.into(),
            arch_version: ARCH_VERSION,
            config: serde_json::to_string(&architecture).expect("an architecture serializes"),
            preprocessing: None,
            info: Default::default(),
        };
        let weights = crate::weights::to_bytes(&network.tensors(), &metadata)?;
        let mlp = Self {
            architecture,
            weights,
            network: OnceLock::new(),
        };
        let stats = TrainStats::Epochs {
            epochs_run,
            total_epochs: params.epochs,
            final_train_loss,
            final_val_loss,
            best_val_loss,
            best_val_epoch,
        };
        Ok((mlp, stats))
    }

    /// The class and its softmax probability, in blocks of
    /// [`PREDICT_CHUNK_ROWS`] in parallel.
    fn predict(&self, rows: &[f32], n_features: usize) -> Result<Predictions, NnError> {
        if self.check_rows(rows, n_features)? == 0 {
            return Ok(Predictions::default());
        }
        let network = self.network()?;
        let blocks = rows
            .par_chunks(PREDICT_CHUNK_ROWS * n_features)
            .map(|block| {
                let scores = self.scores(network, block);
                let classes = read_classes(scores.clone().argmax(1))?;
                let certainty = softmax(scores, 1)
                    .max_dim(1)
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| NnError::Internal(format!("MLP certainty: {e:?}")))?;
                Ok((classes, certainty))
            })
            .collect::<Result<Vec<_>, NnError>>()?;
        let mut out = Predictions::default();
        for (classes, certainty) in blocks {
            out.classes.extend(classes);
            out.certainty.extend(certainty);
        }
        Ok(out)
    }

    fn predict_classes(&self, rows: &[f32], n_features: usize) -> Result<Vec<u32>, NnError> {
        if self.check_rows(rows, n_features)? == 0 {
            return Ok(Vec::new());
        }
        let network = self.network()?;
        let blocks = rows
            .par_chunks(PREDICT_CHUNK_ROWS * n_features)
            .map(|block| read_classes(self.scores(network, block).argmax(1)))
            .collect::<Result<Vec<_>, NnError>>()?;
        Ok(blocks.concat())
    }
}

fn read_classes(argmax: Tensor<2, Int>) -> Result<Vec<u32>, NnError> {
    let values = argmax
        .into_data()
        .convert::<i64>()
        .try_to_vec::<i64>()
        .map_err(|e| NnError::Internal(format!("MLP predictions: {e:?}")))?;
    Ok(values.into_iter().map(|v| v as u32).collect())
}

#[cfg(test)]
mod tests;
