//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::classifier::mlp::MlpActivation;
use crate::train::{TrainControl, TrainDevice, TrainProgress};
use std::sync::atomic::AtomicBool;

/// Runs `f` with a training control of `seed` (no progress, never
/// cancelled, the CPU).
pub(crate) fn with_control<T>(seed: u64, f: impl FnOnce(&TrainControl<'_>) -> T) -> T {
    let cancel = AtomicBool::new(false);
    let progress = |_: TrainProgress| {};
    let control = TrainControl {
        progress: &progress,
        cancel: &cancel,
        seed,
        device: TrainDevice::Cpu,
        cores: 1,
    };
    f(&control)
}

/// Two well-separated clusters (label 0 near 0, label 1 near 10 on both
/// features) - trivial for any working classifier, so a wrong prediction
/// means the wiring (params, matrix, labels) is broken, not that the model
/// didn't generalize.
pub(crate) fn two_clusters() -> (Vec<f32>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for i in 0..15 {
        let jitter = (i % 3) as f32 * 0.1;
        rows.extend([jitter, jitter]);
        labels.push(0);
        rows.extend([10.0 + jitter, 10.0 + jitter]);
        labels.push(1);
    }
    (rows, labels)
}

pub(crate) fn samples<'a>(rows: &'a [f32], labels: &'a [u32], n_features: usize) -> Samples<'a> {
    Samples {
        rows,
        n_features,
        labels,
        n_classes: labels.iter().max().map_or(0, |m| *m as usize + 1),
    }
}

#[test]
fn samples_are_validated() {
    let ok = samples(&[0.0, 1.0], &[0], 2);
    assert!(ok.validate().is_ok());
    assert!(samples(&[], &[], 2).validate().is_err(), "no samples");
    assert!(
        samples(&[0.0, 1.0, 2.0], &[0], 2).validate().is_err(),
        "ragged"
    );
    let label_too_big = Samples {
        n_classes: 1,
        ..samples(&[0.0], &[3], 1)
    };
    assert!(label_too_big.validate().is_err());
    assert!(matches!(
        row_count(&[0.0; 5], 2),
        Err(NnError::InvalidInput(_))
    ));
    assert_eq!(row_count(&[0.0; 6], 2).unwrap(), 3);
}

/// The JSON layout of saved classifiers (the `config.rs` files): variant
/// names, fields, the MLP's weight tensors and metadata.
#[test]
fn the_file_layout_is_pinned() {
    let (rows, labels) = two_clusters();
    let s = samples(&rows, &labels, 2);
    let forest = with_control(1, |c| {
        RandomForest::train(
            &s,
            &RandomForestParams {
                n_trees: 2,
                ..Default::default()
            },
            c,
        )
    })
    .unwrap()
    .0;
    let json = serde_json::to_string(&TrainedClassifier::RandomForest(forest)).unwrap();
    assert!(
        json.starts_with(r#"{"RandomForest":{"trees":[{"#),
        "{}",
        &json[..60]
    );
    assert!(json.contains(r#""classes":[0,1]"#));

    let knn = with_control(1, |c| {
        Knn::train(
            &s,
            &KnnParams {
                k: 3,
                ..Default::default()
            },
            c,
        )
    })
    .unwrap()
    .0;
    let json = serde_json::to_string(&TrainedClassifier::Knn(knn)).unwrap();
    assert!(
        json.starts_with(r#"{"Knn":{"Euclidean":{"#),
        "{}",
        &json[..40]
    );

    let mlp = with_control(1, |c| {
        let params = MlpParams {
            hidden_layers: vec![4],
            epochs: 1,
            learning_rate: 0.01,
            batch_size: 8,
            ..Default::default()
        };
        Mlp::train(&s, &params, c)
    })
    .unwrap()
    .0;
    // The weights: our safetensors with the MLP's names and metadata.
    let file = crate::weights::ModelFile::parse(&mlp.weights).unwrap();
    assert_eq!(
        (file.metadata.arch.as_str(), file.metadata.arch_version),
        ("mlp", 1)
    );
    assert_eq!(
        file.metadata.config,
        r#"{"n_in":2,"hidden":[4],"n_out":2,"activation":"RELU"}"#
    );
    let mut names: Vec<(String, Vec<usize>)> = file
        .tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            ("layer0.bias".to_string(), vec![4]),
            ("layer0.weight".to_string(), vec![4, 2]),
            ("layer1.bias".to_string(), vec![2]),
            ("layer1.weight".to_string(), vec![2, 4]),
        ]
    );
    let json = serde_json::to_string(&TrainedClassifier::Mlp(mlp)).unwrap();
    assert!(
        json.starts_with(r#"{"Mlp":{"architecture":{"n_in":2,"hidden":[4],"n_out":2,"activation":"RELU"},"weights":["#),
        "{}",
        &json[..100]
    );
    let back: TrainedClassifier = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, TrainedClassifier::Mlp(_)));
}

/// `tests/fixtures/classifiers.json`, recorded once from core's classifiers
/// before the move (`crates/core` test `write_nn_fixtures`, 2026-11-25):
/// training data, settings (core's names) and the old predictions on fixed
/// probe rows. Trained here on the same data with the same settings:
///
/// - forests and KNNs give exactly the old classes and certainties
///   (smartcore is deterministic);
/// - the MLP is at least as accurate as the old one (within 2 points -
///   burn 0.21 and 0.22 draw other random numbers, so not the same
///   weights).
#[test]
fn classifiers_give_the_old_results() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/classifiers.json");
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let ints = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let n_features = fixture["n_features"].as_u64().unwrap() as usize;
    let (rows, labels) = (
        floats(&fixture["train_rows"]),
        ints(&fixture["train_labels"]),
    );
    let s = Samples {
        rows: &rows,
        n_features,
        labels: &labels,
        n_classes: fixture["n_classes"].as_u64().unwrap() as usize,
    };
    let probes = floats(&fixture["probes"]);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let settings = &case["settings"];
        let model = if name == "forest" {
            let params = RandomForestParams {
                criterion: match settings["criterion"].as_str().unwrap() {
                    "GINI" => rtree::SplitCriterion::Gini,
                    "ENTROPY" => rtree::SplitCriterion::Entropy,
                    _ => rtree::SplitCriterion::ClassificationError,
                },
                max_depth: settings["maxDepth"].as_u64().map(|d| d as u16),
                min_samples_leaf: settings["minSamplesLeaf"].as_u64().unwrap() as usize,
                min_samples_split: settings["minSamplesSplit"].as_u64().unwrap() as usize,
                n_trees: settings["nTrees"].as_u64().unwrap() as u16,
                m: settings["m"].as_u64().map(|m| m as usize),
            };
            let seed = settings["seed"].as_u64().unwrap();
            TrainedClassifier::RandomForest(
                with_control(seed, |c| RandomForest::train(&s, &params, c))
                    .unwrap()
                    .0,
            )
        } else {
            let distance = match &settings["distance"] {
                serde_json::Value::String(d) if d == "EUCLIDEAN" => knn::KnnDistance::Euclidean,
                serde_json::Value::String(d) if d == "MANHATTAN" => knn::KnnDistance::Manhattan,
                serde_json::Value::String(d) if d == "COSINE" => knn::KnnDistance::Cosine,
                serde_json::Value::String(d) if d == "HAMMING" => knn::KnnDistance::Hamming,
                other => knn::KnnDistance::Minkowski {
                    p: other["MINKOWSKI"]["p"].as_u64().unwrap() as u16,
                },
            };
            let params = KnnParams {
                search: match settings["algorithm"].as_str().unwrap() {
                    "COVER_TREE" => knn::KnnSearch::CoverTree,
                    _ => knn::KnnSearch::Linear,
                },
                weight: match settings["weight"].as_str().unwrap() {
                    "DISTANCE" => knn::KnnWeight::Distance,
                    _ => knn::KnnWeight::Uniform,
                },
                k: settings["k"].as_u64().unwrap() as usize,
                distance,
            };
            TrainedClassifier::Knn(with_control(0, |c| Knn::train(&s, &params, c)).unwrap().0)
        };
        let p = model.predict(&probes, n_features).unwrap();
        assert_eq!(p.classes, ints(&case["classes"]), "{name}: classes");
        assert_eq!(p.certainty, floats(&case["certainty"]), "{name}: certainty");
        assert_eq!(
            model.predict_classes(&probes, n_features).unwrap(),
            p.classes,
            "{name}"
        );
    }

    let mlp = &fixture["mlp"];
    let params = MlpParams {
        hidden_layers: mlp["hidden_layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
        activation: MlpActivation::Relu,
        epochs: mlp["epochs"].as_u64().unwrap() as usize,
        learning_rate: mlp["learning_rate"].as_f64().unwrap(),
        batch_size: mlp["batch_size"].as_u64().unwrap() as usize,
        epsilon: mlp["epsilon"].as_f64().unwrap(),
    };
    let (new, _) = with_control(mlp["seed"].as_u64().unwrap(), |c| {
        Mlp::train(&s, &params, c)
    })
    .unwrap();
    let (test_rows, test_labels) = (floats(&mlp["test_rows"]), ints(&mlp["test_labels"]));
    let predicted = new.predict_classes(&test_rows, n_features).unwrap();
    let correct = predicted
        .iter()
        .zip(&test_labels)
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = correct as f64 / test_labels.len() as f64;
    let old_accuracy = mlp["old_accuracy"].as_f64().unwrap();
    eprintln!("MLP accuracy: new {accuracy:.3}, old {old_accuracy:.3}");
    assert!(
        accuracy >= old_accuracy - 0.02,
        "new {accuracy:.3} vs old {old_accuracy:.3}"
    );
}
