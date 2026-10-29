//! # bench_tracking
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark (not part of the test suite) for the tracking steps: runs one
// tracker on a results file (e.g. from `make_time_series`) and reports wall
// time, peak RSS (`/proc/self/status`'s `VmHWM`, Linux-only) and how many
// of the links in the file's tracks before the run the tracker found again
// (the file's tracks are overwritten - run it on a copy).
//
// Usage: cargo run --release -p evanalyzer_core --example bench_tracking -- \
//     <copy.evadb> --tracker nn|lap|kalman|overlap [--class N] [--distance PX] \
//     [--merge-split 1] [--no-truth 1]
//
// `--no-truth 1` skips the links check, for the tracker's own peak RSS.
// `--mark-divisions N` then makes a file with lineage for the results
// benchmarks: every N-th track (by id) gets the track before it as parent.

use evanalyzer_cfg::core_types::{ImageChannelIdx, ObjectClass, SizeUnits};
use evanalyzer_core::{
    DatasetAlgorithm, DatasetContext, DuckDbExporter, MergeSplitAmount, ObjectFilter,
    PipelineResultExporter, TrackAssignment, TrackKalman, TrackLap, TrackNearestNeighbor,
    TrackOverlap,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn peak_rss_mb() -> Option<f64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmHWM:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<f64>().ok())
                .map(|kb| kb / 1024.0)
        })
}

/// (image, frame, track) -> object, for the objects of `class`.
fn links(
    store: &dyn evanalyzer_core::DatasetStore,
    class: ObjectClass,
) -> HashMap<(String, i32, u64), u128> {
    store
        .get_objects(&ObjectFilter {
            object_class: Some(class),
            ..Default::default()
        })
        .unwrap()
        .into_iter()
        .filter_map(|o| Some(((o.image_rel_path, o.t_stack, o.track_id?), o.id.0)))
        .collect()
}

/// The links (object in frame t -> object in t + 1 of the same track).
fn link_set(tracks: &HashMap<(String, i32, u64), u128>) -> std::collections::HashSet<(u128, u128)> {
    tracks
        .iter()
        .filter_map(|((image, t, track), &a)| {
            let b = tracks.get(&(image.clone(), t + 1, *track))?;
            Some((a, *b))
        })
        .collect()
}

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: bench_tracking <copy.evadb> --tracker ..."),
    );
    let tracker = arg("--tracker").unwrap_or_else(|| "lap".into());
    let class = ObjectClass::Global(arg("--class").and_then(|c| c.parse().ok()).unwrap_or(1));
    let distance: f32 = arg("--distance")
        .and_then(|d| d.parse().ok())
        .unwrap_or(10.0);
    let merge_split = arg("--merge-split").as_deref() == Some("1");

    let exporter = DuckDbExporter::open_existing(&path).expect("open results");
    let store = exporter.dataset_store().unwrap();
    let check = arg("--no-truth").as_deref() != Some("1");
    let truth = if check {
        link_set(&links(store, class))
    } else {
        Default::default()
    };
    let rss_before = peak_rss_mb().unwrap_or(0.0);

    let step: Box<dyn DatasetAlgorithm> = match tracker.as_str() {
        "nn" => Box::new(TrackNearestNeighbor {
            object_class: class,
            unit: SizeUnits::Pixels,
            max_distance: distance,
        }),
        "lap" => Box::new(TrackLap {
            object_class: class,
            unit: SizeUnits::Pixels,
            max_distance: distance,
            max_gap_frames: 2,
            max_gap_distance: distance * 1.5,
            merging: merge_split,
            splitting: merge_split,
            merge_split_amount: MergeSplitAmount::Area,
            channel: ImageChannelIdx(0),
        }),
        "kalman" => Box::new(TrackKalman {
            object_class: class,
            unit: SizeUnits::Pixels,
            initial_max_distance: distance,
            max_distance: distance / 2.0,
            max_gap_frames: 2,
            max_gap_distance: distance,
            position_noise: 0.5,
            velocity_noise: 0.5,
            merging: merge_split,
            splitting: merge_split,
            merge_split_amount: MergeSplitAmount::Intensity,
            channel: ImageChannelIdx(0),
        }),
        "overlap" => Box::new(TrackOverlap {
            object_class: class,
            min_overlap: 0.3,
            max_gap_frames: 2,
            merging: merge_split,
            splitting: merge_split,
            merge_split_amount: MergeSplitAmount::Area,
            channel: ImageChannelIdx(0),
        }),
        other => panic!("unknown tracker {other}"),
    };
    let mut ctx = DatasetContext {
        store,
        progress: None,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let start = Instant::now();
    step.execute(&mut ctx).expect("tracking");
    let elapsed = start.elapsed();
    let rss = peak_rss_mb().unwrap_or(0.0);
    // Counted after the peak was read: reading every object takes RAM too.
    let objects = links(store, class).len();

    let found = if check {
        link_set(&links(store, class))
    } else {
        Default::default()
    };
    let correct = truth.intersection(&found).count();
    println!(
        "{tracker}: {objects} objects in {:.2} s ({:.0} objects/s), peak RSS {rss:.0} MB \
         ({rss_before:.0} MB before, reading the truth), links: {correct} of {} true found, \
         {} wrong",
        elapsed.as_secs_f64(),
        objects as f64 / elapsed.as_secs_f64(),
        truth.len(),
        found.len() - correct,
    );

    if let Some(every) = arg("--mark-divisions").and_then(|n| n.parse::<u64>().ok()) {
        let objects = store
            .get_objects(&ObjectFilter {
                object_class: Some(class),
                ..Default::default()
            })
            .unwrap();
        let marked: Vec<TrackAssignment> = objects
            .into_iter()
            .filter_map(|o| {
                let track = o.track_id?;
                (track % every == 0 && track > 1).then(|| TrackAssignment {
                    object_id: o.id,
                    track_id: track,
                    parent_track_ids: vec![track - 1],
                })
            })
            .collect();
        store.set_track_ids(&marked).unwrap();
        println!(
            "marked every {every}th track as a daughter ({} objects)",
            marked.len()
        );
    }
}
