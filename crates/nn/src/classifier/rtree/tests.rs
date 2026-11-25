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

fn train(rows: &[f32], labels: &[u32], params: &RandomForestParams, seed: u64) -> RandomForest {
    let s = Samples {
        n_classes: labels.iter().max().map_or(1, |m| *m as usize + 1),
        ..samples(rows, labels, rows.len() / labels.len().max(1))
    };
    with_control(seed, |c| RandomForest::train(&s, params, c))
        .unwrap()
        .0
}

/// smartcore's own forest of the same samples and seed.
fn smartcore(rows: &[f32], labels: &[u32], params: &RandomForestParams, seed: u64) -> ForestModel {
    let y: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
    let x = dense_matrix(rows, rows.len() / labels.len()).unwrap();
    RandomForestClassifier::fit(&x, &y, to_smartcore_params(params, seed)).unwrap()
}

fn as_usize(classes: &[u32]) -> Vec<usize> {
    classes.iter().map(|&c| c as usize).collect()
}

#[test]
fn it_separates_two_well_separated_clusters() {
    let (rows, labels) = two_clusters();
    let params = RandomForestParams {
        n_trees: 10,
        max_depth: Some(5),
        min_samples_leaf: 1,
        min_samples_split: 2,
        ..Default::default()
    };
    let forest = train(&rows, &labels, &params, 42);
    assert_eq!(
        forest
            .predict_classes(&[0.05, 0.05, 10.05, 10.05], 2)
            .unwrap(),
        [0, 1]
    );
}

#[test]
fn the_certainty_is_the_share_of_trees_agreeing() {
    let (rows, labels) = two_clusters();
    let params = RandomForestParams {
        n_trees: 20,
        ..Default::default()
    };
    let forest = train(&rows, &labels, &params, 7);
    let p = forest
        .predict(&[0.05, 0.05, 10.05, 10.05, 5.0, 5.0], 2)
        .unwrap();
    assert_eq!(&p.classes[..2], &[0, 1]);
    assert_eq!(
        &p.certainty[..2],
        &[1.0, 1.0],
        "clear cases: every tree agrees"
    );
    assert!((0.5..=1.0).contains(&p.certainty[2]));
    assert!(
        p.certainty.iter().all(|s| (s * 20.0).fract() == 0.0),
        "whole trees"
    );
}

#[test]
fn a_saved_forest_reads_its_trees_after_loading() {
    let (rows, labels) = two_clusters();
    let forest = train(
        &rows,
        &labels,
        &RandomForestParams {
            n_trees: 5,
            ..Default::default()
        },
        0,
    );
    let json = serde_json::to_vec(&TrainedClassifier::RandomForest(forest)).unwrap();
    let loaded: TrainedClassifier = serde_json::from_slice(&json).unwrap();
    let p = loaded.predict(&[0.05, 0.05], 2).unwrap();
    assert_eq!((p.classes, p.certainty), (vec![0], vec![1.0]));
}

/// A forest of no trees gives every row its first class, as smartcore's
/// own `predict`, with no certainty.
#[test]
fn a_forest_without_trees_predicts_its_first_class() {
    let (rows, labels) = two_clusters();
    let params = RandomForestParams::default();
    assert_eq!(params.n_trees, 0);
    let ours = train(&rows, &labels, &params, 0);
    let theirs = smartcore(&rows, &labels, &params, 0);
    let pixels = [0.05, 0.05, 10.05, 10.05];
    let p = ours.predict(&pixels, 2).unwrap();
    assert_eq!(
        as_usize(&p.classes),
        theirs.predict(&dense_matrix(&pixels, 2).unwrap()).unwrap()
    );
    assert_eq!(p.certainty, [0.0, 0.0]);
}

/// The class of every row is smartcore's `predict`'s, also across several
/// chunks and with ties: an even number of trees on noisy, overlapping
/// classes (labels 0, 2 and 5 - not 0..n, as when a class has no examples).
#[test]
fn the_classes_are_smartcores() {
    let mut seed = 12345u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let label_of = [0u32, 2, 5];
    let (mut rows, mut labels) = (Vec::new(), Vec::new());
    for i in 0..600 {
        let class = i % 3;
        rows.extend([next() + class as f32 * 0.2, next()]);
        labels.push(label_of[class]);
    }
    let params = RandomForestParams {
        n_trees: 6,
        max_depth: Some(4),
        ..Default::default()
    };
    let forest = train(&rows, &labels, &params, 3);
    let theirs = smartcore(&rows, &labels, &params, 3);
    let pixels: Vec<f32> = (0..(VOTE_CHUNK_ROWS * 2 + 77) * 2)
        .map(|i| if i % 2 == 0 { next() * 1.4 } else { next() })
        .collect();
    let p = forest.predict(&pixels, 2).unwrap();
    assert_eq!(
        as_usize(&p.classes),
        theirs.predict(&dense_matrix(&pixels, 2).unwrap()).unwrap()
    );
    assert!(
        p.certainty.iter().any(|&s| s == 0.5),
        "the data has ties (3 of 6 trees)"
    );
    assert!(
        p.certainty
            .iter()
            .all(|&s| (1.0 / 3.0..=1.0).contains(&s) && (s * 6.0).fract() == 0.0)
    );
    assert_eq!(forest.predict_classes(&pixels, 2).unwrap(), p.classes);
}

#[test]
fn bad_samples_are_refused() {
    let params = RandomForestParams::default();
    let mismatched = samples(&[0.0, 1.0], &[0], 1);
    assert!(with_control(0, |c| RandomForest::train(&mismatched, &params, c)).is_err());
    let none = samples(&[], &[], 1);
    let err = with_control(0, |c| RandomForest::train(&none, &params, c))
        .err()
        .unwrap();
    assert!(err.to_string().contains("no training samples"), "{err}");
}

/// Every split criterion reaches smartcore as itself and fits the clusters.
#[test]
fn every_split_criterion_is_passed_on_and_fits() {
    let (rows, labels) = two_clusters();
    for (criterion, expected) in [
        (SplitCriterion::Gini, "Gini"),
        (SplitCriterion::Entropy, "Entropy"),
        (SplitCriterion::ClassificationError, "ClassificationError"),
    ] {
        let params = RandomForestParams {
            criterion,
            n_trees: 5,
            max_depth: Some(3),
            min_samples_leaf: 2,
            min_samples_split: 3,
            m: Some(1),
        };
        let sc = to_smartcore_params(&params, 11);
        assert_eq!(format!("{:?}", sc.criterion), expected);
        assert_eq!(
            (
                sc.max_depth,
                sc.min_samples_leaf,
                sc.min_samples_split,
                sc.n_trees,
                sc.m,
                sc.seed
            ),
            (Some(3), 2, 3, 5, Some(1), 11)
        );
        assert!(!sc.keep_samples);
        let forest = train(&rows, &labels, &params, 11);
        assert_eq!(
            forest
                .predict_classes(&[0.05, 0.05, 10.05, 10.05], 2)
                .unwrap(),
            [0, 1],
            "{expected}"
        );
    }
}

/// No trees: every row the first class with no certainty - or an error
/// without classes.
#[test]
fn a_forest_without_trees_predicts_its_first_class_or_fails() {
    let forest: RandomForest = serde_json::from_str(r#"{"trees":[],"classes":[3,4]}"#).unwrap();
    let p = forest.predict(&[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(
        (p.classes, p.certainty),
        (vec![3, 3, 3], vec![0.0, 0.0, 0.0])
    );
    for json in [
        r#"{"trees":[],"classes":null}"#,
        r#"{"trees":[],"classes":[]}"#,
    ] {
        let forest: RandomForest = serde_json::from_str(json).unwrap();
        let err = forest.predict(&[1.0], 1).unwrap_err();
        assert!(err.to_string().contains("no classes"), "{json}: {err}");
    }
}

#[test]
fn no_rows_give_nothing_and_ragged_rows_fail() {
    let (rows, labels) = two_clusters();
    let forest = train(
        &rows,
        &labels,
        &RandomForestParams {
            n_trees: 3,
            ..Default::default()
        },
        5,
    );
    assert_eq!(forest.predict(&[], 2).unwrap(), Predictions::default());
    assert!(forest.predict(&[0.0, 0.0, 1.0], 2).is_err());
}

/// Speed and peak RAM of a prediction with certainty, as the pixel
/// classifier step makes it for one 512² tile (20 features, 50 trees, 3
/// classes). `cargo test -p evanalyzer_nn --lib bench_forest -- --ignored --nocapture`
/// (debug build - compare like with like).
#[test]
#[ignore]
fn bench_forest_predict_with_certainty() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut sample = |class: usize, out: &mut Vec<f32>| {
        out.extend((0..20).map(|f| next() + if f % 4 == 0 { class as f32 * 0.5 } else { 0.0 }));
    };
    let labels: Vec<u32> = (0..5_000).map(|i| i % 3).collect();
    let mut rows = Vec::new();
    for &c in &labels {
        sample(c as usize, &mut rows);
    }
    let forest = train(
        &rows,
        &labels,
        &RandomForestParams {
            n_trees: 50,
            ..Default::default()
        },
        1,
    );
    let mut pixels = Vec::new();
    for i in 0..512 * 512 {
        sample(i % 3, &mut pixels);
    }
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let p = forest.predict(&pixels, 20).unwrap();
        times.push(start.elapsed().as_secs_f64());
        assert_eq!(p.classes.len(), 512 * 512);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "512² with certainty: median {:.2} s (min {:.2}, max {:.2})",
        times[1], times[0], times[2]
    );
}
