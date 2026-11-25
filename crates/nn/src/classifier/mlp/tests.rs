//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::classifier::TrainedClassifier;
use crate::classifier::tests::{samples, two_clusters, with_control};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

/// Trainings one at a time (they use all cores; the initialization has its
/// own generator - `the_same_seed_gives_the_same_model`).
static TRAIN_LOCK: Mutex<()> = Mutex::new(());

fn params(epochs: usize) -> MlpParams {
    MlpParams {
        hidden_layers: vec![4],
        epochs,
        learning_rate: 0.05,
        ..Default::default()
    }
}

fn train(
    rows: &[f32],
    labels: &[u32],
    params: &MlpParams,
    seed: u64,
) -> Result<(Mlp, TrainStats), NnError> {
    let s = samples(rows, labels, rows.len() / labels.len().max(1));
    with_control(seed, |c| Mlp::train(&s, params, c))
}

const CLUSTER_PIXELS: [f32; 4] = [0.05, 0.05, 10.05, 10.05];

#[test]
fn it_separates_two_well_separated_clusters() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let (mlp, stats) = train(&rows, &labels, &params(300), 1).unwrap();
    let p = mlp.predict(&CLUSTER_PIXELS, 2).unwrap();
    assert_eq!(p.classes, [0, 1]);
    assert_eq!(mlp.predict_classes(&CLUSTER_PIXELS, 2).unwrap(), p.classes);
    assert!(
        p.certainty.iter().all(|c| (0.5..=1.0).contains(c)),
        "{:?}",
        p.certainty
    );
    // 30 samples: a validation split happened.
    let TrainStats::Epochs {
        epochs_run,
        total_epochs,
        final_val_loss,
        best_val_loss,
        best_val_epoch,
        ..
    } = stats
    else {
        panic!("epoch stats expected");
    };
    assert_eq!((epochs_run, total_epochs), (300, 300));
    assert!(final_val_loss.is_some() && best_val_loss.is_some() && best_val_epoch.is_some());
}

/// Every activation is trained, remembered and used again at prediction.
#[test]
fn every_activation_separates_the_clusters() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    for activation in [
        MlpActivation::Relu,
        MlpActivation::Sigmoid,
        MlpActivation::Tanh,
    ] {
        let (mlp, _) = train(
            &rows,
            &labels,
            &MlpParams {
                activation,
                ..params(300)
            },
            1,
        )
        .unwrap();
        assert_eq!(mlp.architecture.activation, activation);
        assert_eq!(
            mlp.predict_classes(&CLUSTER_PIXELS, 2).unwrap(),
            [0, 1],
            "{activation:?}"
        );
    }
}

/// A batch size that doesn't divide the training rows (a smaller last
/// batch) trains as well; 0 counts as 1.
#[test]
fn uneven_and_zero_batch_sizes_train() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let (mlp, _) = train(
        &rows,
        &labels,
        &MlpParams {
            batch_size: 7,
            ..params(300)
        },
        1,
    )
    .unwrap();
    assert_eq!(mlp.predict_classes(&CLUSTER_PIXELS, 2).unwrap(), [0, 1]);
    assert!(
        train(
            &rows,
            &labels,
            &MlpParams {
                batch_size: 0,
                ..params(5)
            },
            0
        )
        .is_ok()
    );
}

/// Adam divides by `sqrt(v) + epsilon`: a large epsilon moves the loss
/// less from its start (ln 2) in 20 epochs than a tiny one - so the setting
/// reaches the optimizer.
#[test]
fn a_large_epsilon_suppresses_training_progress() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let loss = |epsilon: f64| {
        let (_, stats) = train(
            &rows,
            &labels,
            &MlpParams {
                epsilon,
                ..params(20)
            },
            1,
        )
        .unwrap();
        let TrainStats::Epochs {
            final_train_loss, ..
        } = stats
        else {
            unreachable!()
        };
        final_train_loss
    };
    let (tiny, large) = (loss(1e-8), loss(10.0));
    let start = 2f32.ln();
    assert!(
        (large - start).abs() < (tiny - start).abs(),
        "large {large}, tiny {tiny}"
    );
}

/// The same seed and data give the same model.
#[test]
fn the_same_seed_gives_the_same_model() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let (a, _) = train(&rows, &labels, &params(30), 5).unwrap();
    let (b, _) = train(&rows, &labels, &params(30), 5).unwrap();
    let (c, _) = train(&rows, &labels, &params(30), 6).unwrap();
    assert_eq!(a.weights, b.weights);
    assert_ne!(a.weights, c.weights);
}

/// In blocks across block borders: the same as in one pass; saved and
/// loaded: the same again.
#[test]
fn blocks_and_a_saved_copy_predict_the_same() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let (mlp, _) = train(&rows, &labels, &params(50), 3).unwrap();
    let pixels: Vec<f32> = (0..PREDICT_CHUNK_ROWS * 2 + 7)
        .flat_map(|i| {
            let v = (i % 211) as f32 / 21.0;
            [v, 10.0 - v]
        })
        .collect();
    let p = mlp.predict(&pixels, 2).unwrap();
    let network = mlp.network().unwrap();
    let one_pass = mlp.scores(network, &pixels);
    assert_eq!(read_classes(one_pass.clone().argmax(1)).unwrap(), p.classes);
    let certainty = softmax(one_pass, 1)
        .max_dim(1)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(certainty, p.certainty);
    assert_eq!(mlp.predict_classes(&pixels, 2).unwrap(), p.classes);
    assert!(p.classes.contains(&0) && p.classes.contains(&1));

    let json = serde_json::to_vec(&TrainedClassifier::Mlp(mlp)).unwrap();
    let loaded: TrainedClassifier = serde_json::from_slice(&json).unwrap();
    assert_eq!(loaded.predict(&pixels, 2).unwrap(), p);
}

/// Bytes that are no weights of ours, wrong shapes and wrong feature
/// counts are refused.
#[test]
fn wrong_weights_are_refused() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let (mlp, _) = train(&rows, &labels, &params(2), 1).unwrap();
    let garbage = Mlp {
        weights: vec![7; 64],
        ..mlp_copy(&mlp)
    };
    assert!(matches!(
        garbage.predict(&CLUSTER_PIXELS, 2),
        Err(NnError::InvalidModelFile(_))
    ));
    let wider = Mlp {
        architecture: MlpArchitecture {
            hidden: vec![5],
            ..mlp.architecture.clone()
        },
        ..mlp_copy(&mlp)
    };
    assert!(matches!(
        wider.predict(&CLUSTER_PIXELS, 2),
        Err(NnError::InvalidModelFile(_))
    ));
    assert!(matches!(
        mlp.predict(&[1.0, 2.0, 3.0], 3),
        Err(NnError::InvalidInput(_))
    ));
}

fn mlp_copy(mlp: &Mlp) -> Mlp {
    Mlp {
        architecture: mlp.architecture.clone(),
        weights: mlp.weights.clone(),
        network: OnceLock::new(),
    }
}

#[test]
fn zero_samples_or_classes_are_refused() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert!(train(&[], &[], &params(1), 0).is_err());
    let (rows, labels) = two_clusters();
    let s = Samples {
        n_classes: 0,
        ..samples(&rows, &labels, 2)
    };
    assert!(with_control(0, |c| Mlp::train(&s, &params(1), c)).is_err());
}

/// Below the minimum: every row trains, no validation loss.
#[test]
fn few_samples_have_no_validation_split() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (_, stats) = train(
        &[0.0, 0.1, 10.0, 10.1],
        &[0, 0, 1, 1],
        &MlpParams {
            hidden_layers: vec![2],
            ..params(5)
        },
        0,
    )
    .unwrap();
    let TrainStats::Epochs {
        final_val_loss,
        best_val_loss,
        ..
    } = stats
    else {
        unreachable!()
    };
    assert!(final_val_loss.is_none() && best_val_loss.is_none());
}

#[test]
fn a_set_cancel_flag_stops_the_training() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let s = samples(&rows, &labels, 2);
    let cancel = AtomicBool::new(true);
    let progress = |_: TrainProgress| {};
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed: 0,
        device: crate::train::TrainDevice::Cpu,
        cores: 1,
    };
    assert!(matches!(
        Mlp::train(&s, &params(1000), &control),
        Err(NnError::Cancelled)
    ));
}

/// The last epoch is always reported, whatever the reporting interval.
#[test]
fn the_last_epoch_is_reported() {
    let _guard = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (rows, labels) = two_clusters();
    let s = samples(&rows, &labels, 2);
    let epochs = Mutex::new(Vec::new());
    let progress = |p: TrainProgress| {
        if let TrainProgress::Epoch {
            epoch,
            total_epochs,
            ..
        } = p
        {
            epochs.lock().unwrap().push((epoch, total_epochs));
        }
    };
    let cancel = AtomicBool::new(false);
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed: 0,
        device: crate::train::TrainDevice::Cpu,
        cores: 1,
    };
    Mlp::train(&s, &params(10), &control).unwrap();
    assert!(epochs.lock().unwrap().contains(&(9, 10)));
}
