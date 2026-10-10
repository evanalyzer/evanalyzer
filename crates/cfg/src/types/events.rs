//! Progress events streamed by `evanalyzer_core`'s classifier training jobs.
//!
//! Lives here rather than in core so front ends (GUI/CLI, and later a remote
//! client) can consume them through `evanalyzer_app` without depending on the
//! engine crate, and so they can be serialized across a transport.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Progress reported by both `evanalyzer_core::PixelTrainingJob::run` and
/// `evanalyzer_core::ObjectTrainingJob::run`. The pixel job is the only one
/// that reads images tile-by-tile, hence the `Image*`/`Tile*` variants; the
/// object job (already-computed metrics, no image I/O) only ever reports
/// `Started`, `ItemCompleted`, `ObjectSkipped`, `Training` and `Finished`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingProgressEvent {
    Started {
        total: usize,
    },
    ImageTilesScheduled {
        image_index: usize,
        total_tiles: usize,
    },
    TileProcessed {
        image_index: usize,
        tile_index: usize,
        total_tiles: usize,
    },
    ItemCompleted {
        index: usize,
        total: usize,
    },
    ImageFailed {
        path: PathBuf,
    },
    /// An object's `object_class` set matched zero or more than one of the
    /// model's configured `class_labels` - ambiguous, so it's excluded from
    /// training rather than guessed at.
    ObjectSkipped {
        index: usize,
        reason: String,
    },
    Training,
    /// One MLP training epoch finished. Only ever sent by the `Mlp` backend
    /// (core's `fit_mlp`) — `RandomForest`/`Knn` fit in one blocking
    /// smartcore call with no per-iteration hook to report from, so they
    /// only ever emit the surrounding `Training`/`Finished` events.
    ///
    /// Sending is throttled (see `fit_mlp`'s `report_every`) rather than sent
    /// for every epoch of a large `epochs` count, but the final epoch is
    /// always sent so the GUI's last-seen value matches `Finished`'s stats.
    Epoch {
        epoch: usize,
        total_epochs: usize,
        train_loss: f32,
        /// `None` when the dataset was too small for a held-out split (see
        /// core's `train_val_split`) - there's then no generalization
        /// signal, only the training-loss curve.
        val_loss: Option<f32>,
    },
    Finished {
        stats: TrainingStats,
    },
}

/// Backend-specific summary reported once, alongside `TrainingProgressEvent::Finished`,
/// for the GUI's post-training results banner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingStats {
    RandomForest {
        n_trees: usize,
        n_samples: usize,
    },
    Knn {
        k: usize,
        n_samples: usize,
    },
    Mlp {
        /// Equal to `total_epochs` unless training was cancelled mid-run -
        /// cancellation surfaces as `InternalErrors::Cancelled` from `run()`
        /// though, so in practice this only ever reaches the GUI as
        /// `total_epochs`; kept distinct from it for when partial-result
        /// reporting on cancel is added.
        epochs_run: usize,
        total_epochs: usize,
        final_train_loss: f32,
        /// `None` when the dataset was too small for a held-out split.
        final_val_loss: Option<f32>,
        /// The lowest validation loss seen at any epoch, and which epoch it
        /// was at - lets the GUI point out overfitting concretely ("best
        /// epoch was 210 of 300; final validation loss is N% worse") instead
        /// of guessing at a threshold itself. `None` alongside `final_val_loss`.
        best_val_loss: Option<f32>,
        best_val_epoch: Option<usize>,
    },
}
