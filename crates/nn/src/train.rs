//! # train
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! What every training shares - classifiers and (later) the networks:
//! progress, cancel, seed, device. Plain types: core maps them to cfg's
//! `TrainingProgressEvent` / `TrainingStats`.

use std::sync::atomic::{AtomicBool, Ordering};

/// How a training is run and watched.
pub struct TrainControl<'a> {
    /// Called with every progress step (throttled by the training).
    pub progress: &'a (dyn Fn(TrainProgress) + Sync),
    /// Set to stop: the training ends with `NnError::Cancelled`.
    pub cancel: &'a AtomicBool,
    /// The same seed and data give the same model.
    pub seed: u64,
    pub device: TrainDevice,
    /// CPU cores the training may use.
    pub cores: usize,
}

impl TrainControl<'_> {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Where a training runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainDevice {
    #[default]
    Cpu,
    /// This GPU (index); the CPU when it doesn't exist.
    Gpu(usize),
}

/// A step of a running training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainProgress {
    /// An epoch finished (the last one is always reported).
    Epoch {
        epoch: usize,
        total_epochs: usize,
        train_loss: f32,
        /// `None` without a held-out split (too few samples).
        val_loss: Option<f32>,
    },
    /// A batch of the current epoch finished - for trainings with long
    /// epochs (the networks).
    Batch { batch: usize, total_batches: usize },
}

/// A finished training, for the result card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainStats {
    RandomForest {
        n_trees: usize,
        n_samples: usize,
    },
    Knn {
        k: usize,
        n_samples: usize,
    },
    /// Trained in epochs: the MLP and the networks.
    Epochs {
        epochs_run: usize,
        total_epochs: usize,
        final_train_loss: f32,
        /// `None` without a held-out split.
        final_val_loss: Option<f32>,
        /// The lowest validation loss and its epoch - shows overfitting.
        best_val_loss: Option<f32>,
        best_val_epoch: Option<usize>,
    },
}

/// Uniform random start weights from a generator of their own (xorshift64),
/// seeded per training - not burn's global one: whatever else draws random
/// numbers meanwhile (other trainings, parallel tests), the same seed gives
/// the same model.
pub(crate) struct SeededUniform {
    state: u64,
}

impl SeededUniform {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1),
        }
    }

    /// `n` values uniform in `±bound`.
    pub(crate) fn values(&mut self, n: usize, bound: f32) -> Vec<f32> {
        (0..n).map(|_| (2.0 * self.unit() - 1.0) * bound).collect()
    }

    /// A random index in `0..n` (`n > 0`).
    pub(crate) fn index(&mut self, n: usize) -> usize {
        ((self.unit() as f64 * n as f64) as usize).min(n - 1)
    }

    /// The next value uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}
