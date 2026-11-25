//! # write_nn_fixtures
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Records what this (old, pre-move) implementation does, for `crates/nn`'s
//! comparison test (`classifier::tests::classifiers_give_the_old_results`):
//! fixed training data, a forest and a KNN per distance with their settings
//! and the old predictions (class + certainty) on fixed probe rows, and the
//! old MLP's accuracy on held-out rows. nn trains its own classifiers on
//! the same data and settings and compares. No model files - old files are
//! not read (no backwards compatibility). Temporary: goes with the old
//! implementation (the fixture stays).
//!
//!   cargo test -p evanalyzer_core --lib write_nn_fixtures -- --ignored

use super::{Classifier, knn::fit_knn, mlp::fit_mlp, random_forest::fit_random_forest};
use evanalyzer_cfg::settings::ai_learning_settings::{
    KNNAlgorithmName, KNNDistanceMetric, KNNWeightFunction, KnnSettings, MlpActivation,
    MlpSettings, RandomForestSettings,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const N_FEATURES: usize = 4;
const N_CLASSES: usize = 3;

/// Three overlapping noisy classes over 4 features (xorshift, fixed seed):
/// `n` rows and their labels.
fn data(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for i in 0..n {
        let class = i % N_CLASSES;
        let c = class as f32;
        rows.push(vec![
            c * 0.6 + next(),
            (2.0 - c) * 0.4 + next(),
            next() * 2.0,
            c * 0.3 + next() * 0.5,
        ]);
        labels.push(class);
    }
    (rows, labels)
}

fn case(
    name: &str,
    settings: serde_json::Value,
    classifier: &Classifier,
    probes: &[Vec<f32>],
) -> serde_json::Value {
    let (classes, certainty) = classifier.predict_with_certainty(probes).unwrap();
    json!({ "name": name, "settings": settings, "classes": classes, "certainty": certainty })
}

#[test]
#[ignore = "writes crates/nn/tests/fixtures/classifiers.json"]
fn write_nn_fixtures() {
    let (rows, labels) = data(450, 0x5EED_1234);
    let (probes, _) = data(300, 0xABCD_EF01);
    let (test_rows, test_labels) = data(900, 0x0DD5_EED5);

    let forest_settings = RandomForestSettings {
        n_trees: 8,
        max_depth: Some(6),
        min_samples_leaf: 1,
        min_samples_split: 2,
        seed: 42,
        ..Default::default()
    };
    let forest = fit_random_forest(&rows, &labels, &forest_settings).unwrap();
    let mut cases = vec![case(
        "forest",
        serde_json::to_value(&forest_settings).unwrap(),
        &forest,
        &probes,
    )];
    for (name, distance, algorithm, weight) in [
        (
            "knn-euclidean",
            KNNDistanceMetric::Euclidean,
            KNNAlgorithmName::CoverTree,
            KNNWeightFunction::Uniform,
        ),
        (
            "knn-manhattan",
            KNNDistanceMetric::Manhattan,
            KNNAlgorithmName::LinearSearch,
            KNNWeightFunction::Distance,
        ),
        (
            "knn-cosine",
            KNNDistanceMetric::Cosine,
            KNNAlgorithmName::CoverTree,
            KNNWeightFunction::Uniform,
        ),
        (
            "knn-minkowski",
            KNNDistanceMetric::Minkowski { p: 3 },
            KNNAlgorithmName::LinearSearch,
            KNNWeightFunction::Uniform,
        ),
    ] {
        let settings = KnnSettings {
            algorithm,
            weight,
            k: 5,
            distance,
        };
        let knn = fit_knn(&rows, &labels, &settings).unwrap();
        cases.push(case(
            name,
            serde_json::to_value(&settings).unwrap(),
            &knn,
            &probes,
        ));
    }

    let mlp_settings = MlpSettings {
        hidden_layers: vec![16],
        activation: MlpActivation::Relu,
        epochs: 60,
        learning_rate: 0.01,
        batch_size: 32,
        seed: 7,
        epsilon: 1e-5,
    };
    let (progress, _rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let (mlp, _) = fit_mlp(&rows, &labels, N_CLASSES, &mlp_settings, &progress, &cancel).unwrap();
    let predicted = mlp.predict(&test_rows).unwrap();
    let correct = predicted
        .iter()
        .zip(&test_labels)
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = correct as f64 / test_labels.len() as f64;

    let fixture = json!({
        "about": "written by crates/core ai_learning::model::write_nn_fixtures (the classifiers before the move to nn)",
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "train_rows": rows.concat(),
        "train_labels": labels,
        "probes": probes.concat(),
        "cases": cases,
        "mlp": {
            "hidden_layers": mlp_settings.hidden_layers,
            "epochs": mlp_settings.epochs,
            "learning_rate": mlp_settings.learning_rate,
            "batch_size": mlp_settings.batch_size,
            "seed": mlp_settings.seed,
            "epsilon": mlp_settings.epsilon,
            "test_rows": test_rows.concat(),
            "test_labels": test_labels,
            "old_accuracy": accuracy,
        },
    });
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../nn/tests/fixtures");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("classifiers.json"),
        serde_json::to_vec(&fixture).unwrap(),
    )
    .unwrap();
    println!("old MLP accuracy {accuracy:.3}");
}
