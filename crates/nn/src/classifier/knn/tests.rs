//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::classifier::tests::{samples, two_clusters, with_control};

fn train(rows: &[f32], labels: &[u32], params: &KnnParams) -> Knn {
    let s = samples(rows, labels, rows.len() / labels.len().max(1));
    with_control(0, |c| Knn::train(&s, params, c)).unwrap().0
}

fn params(search: KnnSearch, distance: KnnDistance) -> KnnParams {
    KnnParams {
        search,
        weight: KnnWeight::Uniform,
        k: 3,
        distance,
    }
}

#[test]
fn it_separates_two_well_separated_clusters() {
    let (rows, labels) = two_clusters();
    let knn = train(
        &rows,
        &labels,
        &params(KnnSearch::CoverTree, KnnDistance::Euclidean),
    );
    assert_eq!(
        knn.predict_classes(&[0.05, 0.05, 10.05, 10.05], 2).unwrap(),
        [0, 1]
    );
}

/// The certainty is the neighbours' share of the class: all 3 inside a
/// cluster, 2 of 3 between them.
#[test]
fn the_certainty_is_the_share_of_neighbours() {
    let rows = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    let knn = train(
        &rows,
        &[0, 0, 0, 1, 1, 1],
        &params(KnnSearch::Linear, KnnDistance::Euclidean),
    );
    let p = knn.predict(&[1.0, 6.4], 1).unwrap();
    // 6.4: nearest are 10 and 11 (class 1), then 2 (class 0).
    assert_eq!(p.classes, [0, 1]);
    assert_eq!(p.certainty[0], 1.0);
    assert!(
        (p.certainty[1] - 2.0 / 3.0).abs() < 1e-6,
        "{:?}",
        p.certainty
    );
    assert_eq!(knn.predict_classes(&[1.0, 6.4], 1).unwrap(), p.classes);
}

#[test]
fn zero_samples_are_refused() {
    let none = samples(&[], &[], 1);
    assert!(
        with_control(0, |c| Knn::train(
            &none,
            &KnnParams {
                k: 3,
                ..Default::default()
            },
            c
        ))
        .is_err()
    );
}

/// Every distance reaches smartcore as its own variant and still separates
/// two well-separated clusters.
#[test]
fn every_distance_separates_the_clusters() {
    let (rows, labels) = two_clusters();
    let cases: [(KnnDistance, fn(&Knn) -> bool); 3] = [
        (KnnDistance::Euclidean, |m| matches!(m, Knn::Euclidean(_))),
        (KnnDistance::Manhattan, |m| matches!(m, Knn::Manhattan(_))),
        (KnnDistance::Minkowski { p: 3 }, |m| {
            matches!(m, Knn::Minkowski(_))
        }),
    ];
    for (distance, is_variant) in cases {
        let knn = train(&rows, &labels, &params(KnnSearch::CoverTree, distance));
        assert!(is_variant(&knn), "{distance:?}");
        assert_eq!(
            knn.predict_classes(&[0.05, 0.05, 10.05, 10.05], 2).unwrap(),
            [0, 1],
            "{distance:?}"
        );
    }
}

/// Cosine measures the angle, not position or magnitude: two clusters near
/// orthogonal directions at mixed magnitudes.
#[test]
fn cosine_separates_by_angle_not_magnitude() {
    let rows = [
        1.0, 0.05, 2.0, 0.1, 0.5, 0.02, 0.05, 1.0, 0.1, 2.0, 0.02, 0.5,
    ];
    let knn = train(
        &rows,
        &[0, 0, 0, 1, 1, 1],
        &params(KnnSearch::CoverTree, KnnDistance::Cosine),
    );
    assert!(matches!(knn, Knn::Cosine(_)));
    // Cluster 0's direction at a magnitude no training point had.
    assert_eq!(knn.predict_classes(&[10.0, 0.5], 2).unwrap(), [0]);
}

/// Hamming counts differing positions: 0/1 rows.
#[test]
fn hamming_separates_binary_clusters() {
    let rows = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let knn = train(
        &rows,
        &[0, 0, 0, 1, 1, 1],
        &params(KnnSearch::Linear, KnnDistance::Hamming),
    );
    assert!(matches!(knn, Knn::Hamming(_)));
    assert_eq!(
        knn.predict_classes(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3)
            .unwrap(),
        [0, 1]
    );
}
