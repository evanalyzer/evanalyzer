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

/// What every network's training is set up with (each model adds its own,
/// e.g. SpotNet's target width).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NetworkTrainParams {
    pub epochs: usize,
    /// Patches per epoch (an image gives many patches).
    pub patches_per_epoch: usize,
    pub batch_size: usize,
    /// Side of a square patch in px, e.g. 128.
    pub patch_size: usize,
    pub learning_rate: f64,
    /// The random changes of every patch.
    pub augment: crate::augment::AugmentParams,
    /// What the training checks itself on.
    pub validation: Validation,
    /// Stop after this many epochs without a better check; `None` = run
    /// every epoch. Needs validation.
    pub patience: Option<usize>,
}

impl NetworkTrainParams {
    /// Refuses settings no training can run with.
    pub fn check(&self) -> Result<(), crate::NnError> {
        let bad = |what: &str| Err(crate::NnError::InvalidInput(format!("training: {what}")));
        if self.epochs == 0 {
            return bad("at least one epoch");
        }
        if self.patches_per_epoch == 0 || self.batch_size == 0 || self.patch_size == 0 {
            return bad("patches per epoch, batch size and patch size must be at least 1");
        }
        if !(self.learning_rate > 0.0) {
            return bad("the learning rate must be above 0");
        }
        if self.patience == Some(0) {
            return bad("the patience must be at least 1 epoch");
        }
        if let Validation::Auto { fraction } = self.validation
            && !(fraction > 0.0 && fraction < 1.0)
        {
            return bad("the share held back for validation must be between 0 and 1");
        }
        Ok(())
    }
}

/// A network's check on held-back areas - the same for every network: how
/// many of the annotated objects (spots, cells) it found there, and how
/// many it found wrongly.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeldBackScore {
    /// The training loss there (lower is better).
    pub loss: f32,
    /// Annotated objects in the held-back areas.
    pub annotated: usize,
    /// Objects the network found there.
    pub found: usize,
    /// Found objects matched to an annotated one.
    pub matched: usize,
}

impl HeldBackScore {
    /// How many of the found objects are real (1 when none was found).
    pub fn precision(&self) -> f32 {
        if self.found == 0 {
            1.0
        } else {
            self.matched as f32 / self.found as f32
        }
    }

    /// How many of the annotated objects were found (1 when there are none).
    pub fn recall(&self) -> f32 {
        if self.annotated == 0 {
            1.0
        } else {
            self.matched as f32 / self.annotated as f32
        }
    }

    /// The harmonic mean of precision and recall.
    pub fn f1(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Found objects that match none annotated.
    pub fn wrongly_found(&self) -> usize {
        self.found - self.matched
    }
}

/// How a network's training was checked.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HeldBack {
    /// The score of the kept model on the held-back areas.
    Scored { epoch: usize, score: HeldBackScore },
    /// Switched off.
    Off,
    /// `Validation::Auto` without drawn validation areas and with fewer than
    /// [`MIN_IMAGES_FOR_AUTO_SPLIT`] images: nothing to hold back.
    TooFewImages { images: usize },
}

/// A finished network training: the kept model (the best epoch's when
/// checked, else the last), its curve and its check.
#[derive(Debug, Clone)]
pub struct Trained<M> {
    pub model: M,
    pub stats: TrainStats,
    pub held_back: HeldBack,
}

/// Which annotations a network's training checks itself on - never trained
/// on, they show whether the model learned spots (or cells) in general or
/// only its training areas by heart, and pick the best epoch.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Validation {
    /// The validation areas the user drew, if there are any. Without: this
    /// share of the images with training areas is held back (from
    /// [`MIN_IMAGES_FOR_AUTO_SPLIT`] images on; with fewer the training runs
    /// unchecked and says so).
    Auto { fraction: f32 },
    /// Only the validation areas the user drew - refused without any.
    Areas,
    /// No check: every epoch runs, the last model is kept.
    Off,
}

impl Default for Validation {
    fn default() -> Self {
        Self::Auto { fraction: 0.2 }
    }
}

/// Below this many images with training areas `Validation::Auto` holds none
/// back: a fifth of 4 images is a single image - too little to judge by, and
/// a loss for the training.
pub const MIN_IMAGES_FOR_AUTO_SPLIT: usize = 5;

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
    pub(crate) fn unit(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}
