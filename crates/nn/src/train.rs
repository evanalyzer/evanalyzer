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
