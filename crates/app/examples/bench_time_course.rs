//! # bench_time_course
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark (not part of the test suite) of the time course - the "Over
// time" chart's query and its export - against a time series results
// database (make one with core's `make_time_series` example). Each query is
// the median of `--iters` runs after a warm-up run; each export runs once.
// Peak RSS at the end (`VmHWM`, Linux). `--case N` runs only the N-th query
// or export (in the order printed, from 0): its own peak RSS. The last two
// exports are the List (a row per object, CSV and XLSX); case 13 is
// Results › Tracks' query: one image's tracks (median of `--iters`); case
// 14 the tracks export (a row per track), 15 every track's row kept
// (Results › Tracks, Full).
//
// `--with-parents`: track lines with their parents' history (results with
// lineage; the other lines are the same either way).
//
// Usage: cargo run --release -p evanalyzer_app --example bench_time_course -- <db.evadb> [--iters N] [--case N] [--with-parents]

use evanalyzer_app::backend::Backend;
use evanalyzer_app::backend::local::LocalBackend;
use evanalyzer_app::exports::{ExportRun, ScriptPermission};
use evanalyzer_app::results::{
    Column, Grouping, ImageTracksFilter, LocalResultsGenerator, TimeCourseFilter, TimeCourseLines,
    TimeCourseNormalize, TimeCourseStat, TrackHistory, TrackSummaryFilter,
};

/// A project export (`run_export`) of `entry` into `out_dir`.
fn run_export(
    results: &dyn evanalyzer_app::results::ResultsSource,
    entry: serde_json::Value,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    let export = serde_json::from_value(serde_json::json!({
        "id": 1, "name": "bench", "entries": [entry]
    }))
    .map_err(|e| e.to_string())?;
    let report = results
        .run_export(
            &ExportRun {
                export,
                project_dir: out_dir.to_path_buf(),
                results_dir: out_dir.to_path_buf(),
                output: Some(out_dir.to_path_buf()),
                scripts: ScriptPermission::Skip,
                interpreters: Default::default(),
                drawing: None,
            },
            &AtomicBool::new(false),
            &mut |_, _, _| {},
        )
        .map_err(|e| e.to_string())?;
    match report.all_written() {
        true => Ok(()),
        false => Err(format!("{report:?}")),
    }
}
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmHWM:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let db_path = PathBuf::from(
        args.get(1)
            .expect("usage: bench_time_course <db.evadb> [--iters N]"),
    );
    let iters: usize = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let case: Option<usize> = args
        .iter()
        .position(|a| a == "--case")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    let runs = |n: usize| case.is_none_or(|c| c == n);
    let history = match args.iter().any(|a| a == "--with-parents") {
        true => TrackHistory::WithParents,
        false => TrackHistory::Own,
    };

    let start = Instant::now();
    let database = LocalResultsGenerator::open_database(db_path.clone()).expect("open database");
    println!(
        "open: {:.1} ms, RSS {:.0} MB",
        start.elapsed().as_secs_f64() * 1000.0,
        peak_rss_mb()
    );

    let filter = |lines, statistics: &[TimeCourseStat], normalize| TimeCourseFilter {
        lines,
        grouping: Grouping::Auto,
        columns: vec![Column::AreaSizePx],
        statistics: statistics.to_vec(),
        normalize,
        images: None,
        object_classes: None,
        t_stacks: None,
        z_stacks: None,
        max_tracks: None,
        history,
    };
    let mean_sd = &TimeCourseStat::DEFAULT[..];
    let median_iqr = &[
        TimeCourseStat::Median,
        TimeCourseStat::Q1,
        TimeCourseStat::Q3,
    ][..];
    let queries = [
        (
            "chart: line per class, mean ± SD",
            filter(TimeCourseLines::Class, mean_sd, TimeCourseNormalize::None),
        ),
        (
            "chart: line per class, median + IQR",
            filter(
                TimeCourseLines::Class,
                median_iqr,
                TimeCourseNormalize::None,
            ),
        ),
        (
            "chart: line per class, relative",
            filter(
                TimeCourseLines::Class,
                mean_sd,
                TimeCourseNormalize::RelativeToStart,
            ),
        ),
        (
            "chart: line per image",
            filter(TimeCourseLines::Image, mean_sd, TimeCourseNormalize::None),
        ),
        (
            "chart: line per well",
            filter(TimeCourseLines::Well, mean_sd, TimeCourseNormalize::None),
        ),
        (
            "chart: faint tracks (400 sampled)",
            TimeCourseFilter {
                max_tracks: Some(400),
                ..filter(
                    TimeCourseLines::Track,
                    &[TimeCourseStat::Mean],
                    TimeCourseNormalize::None,
                )
            },
        ),
        (
            "all tracks (as the export)",
            filter(
                TimeCourseLines::Track,
                &[TimeCourseStat::Mean],
                TimeCourseNormalize::None,
            ),
        ),
    ];
    println!("\n{:<42} {:>10} {:>10}", "query", "median ms", "points");
    for (n, (label, filter)) in queries.iter().enumerate() {
        if !runs(n) {
            continue;
        }
        let mut times = Vec::new();
        let mut points = 0;
        for i in 0..=iters {
            let start = Instant::now();
            let result = database.time_course(filter).expect("time course");
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            points = result.points.len();
        }
        times.sort_by(f64::total_cmp);
        println!("{label:<42} {:>10.1} {points:>10}", times[times.len() / 2]);
    }

    let out_root = std::env::temp_dir().join(format!(
        "evanalyzer_bench_time_course_{}",
        std::process::id()
    ));
    let results = LocalBackend::default()
        .open_results(&db_path)
        .expect("open results");
    let columns = [
        "area_px",
        "intensity_mean_gray_ch0",
        "intensity_mean_gray_ch1",
    ];
    let with_parents = history == TrackHistory::WithParents;
    let mut n_export = queries.len();
    let mut export =
        // `None`: the objects - a row per object; else the time course's
        // lines and layout.
        |label: &str, format: &str, time_course: Option<(&str, &str)>| {
            n_export += 1;
            if !runs(n_export - 1) {
                return;
            }
            let out_dir = out_root.join(label.replace(' ', "_"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let entry = match time_course {
                None => serde_json::json!({ "kind": "DATA", "outputs": ["OBJECTS"],
                    "format": format, "columns": columns, "images": "ALL" }),
                Some((lines, layout)) => serde_json::json!({ "kind": "DATA",
                    "outputs": ["TIME_COURSE"], "format": format, "columns": columns,
                    "aggregations": ["AVG", "STDDEV"], "images": "ALL",
                    "timeCourse": { "lines": lines, "layout": layout,
                        "withParents": with_parents } }),
            };
            let start = Instant::now();
            if let Err(e) = run_export(results.as_ref(), entry, &out_dir) {
                println!("{label:<42} failed: {e}");
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let (files, bytes) = std::fs::read_dir(&out_dir)
                .unwrap()
                .flatten()
                .fold((0, 0u64), |(n, b), e| {
                    (n + 1, b + e.metadata().map_or(0, |m| m.len()))
                });
            println!(
                "{label:<42} {elapsed:>10.1} {files:>6} files {:>8.2} MB",
                bytes as f64 / 1e6
            );
        };
    println!("\n{:<42} {:>10}", "export (3 values, mean + SD)", "ms");
    export(
        "CSV long+wide, line per class",
        "CSV",
        Some(("CLASS", "BOTH")),
    );
    export(
        "CSV long+wide, line per well",
        "CSV",
        Some(("WELL", "BOTH")),
    );
    export(
        "XLSX long+wide, line per image",
        "XLSX",
        Some(("IMAGE", "BOTH")),
    );
    export("CSV long, line per track", "CSV", Some(("TRACK", "LONG")));
    export("Objects CSV, a row per object", "CSV", None);
    export("Objects XLSX, a row per object", "XLSX", None);
    let _ = std::fs::remove_dir_all(&out_root);

    if runs(n_export) {
        let image = database.get_images().expect("images")[0]
            .rel_path
            .to_string_lossy()
            .into_owned();
        let filter = ImageTracksFilter {
            image_rel_path: image,
            z_stack: 0,
            max_points: 1_000_000,
        };
        let mut times = Vec::new();
        let mut tracks = None;
        for i in 0..=iters {
            let start = Instant::now();
            let result = database.image_tracks(&filter).expect("tracks");
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            tracks = Some(result);
        }
        times.sort_by(f64::total_cmp);
        let tracks = tracks.expect("ran");
        let points: usize = tracks.tracks.iter().map(|t| t.points.len()).sum();
        println!(
            "\n{:<42} {:>10.1} {:>6} tracks {points:>8} points",
            "Tracks: one image's tracks",
            times[times.len() / 2],
            tracks.tracks.len()
        );
    }

    // Case 14: a row per track, every image - CSV, the 3 values' means.
    if runs(n_export + 1) {
        let out_dir = out_root.join("tracks");
        std::fs::create_dir_all(&out_dir).unwrap();
        let start = Instant::now();
        run_export(
            results.as_ref(),
            serde_json::json!({ "kind": "DATA", "outputs": ["TRACKS"], "columns": columns,
                "images": "ALL" }),
            &out_dir,
        )
        .expect("tracks export");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let csv = std::fs::read_to_string(out_dir.join("tracks.csv")).unwrap();
        println!(
            "\n{:<42} {elapsed:>10.1} {:>8} tracks {:>8.2} MB",
            "Tracks CSV, a row per track",
            csv.lines().count() - 1,
            csv.len() as f64 / 1e6
        );
    }
    // Case 15: Tracks, Full - every track's row kept (the area's mean).
    if runs(n_export + 2) {
        let filter = TrackSummaryFilter {
            columns: vec![Column::AreaSizePx],
            ..Default::default()
        };
        let mut times = Vec::new();
        let mut n = 0;
        for i in 0..=iters {
            let start = Instant::now();
            let rows = database.track_summary(&filter).expect("tracks");
            if i > 0 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            n = rows.len();
        }
        times.sort_by(f64::total_cmp);
        println!(
            "\n{:<42} {:>10.1} {n:>8} tracks",
            "Tracks, Full: every track's row",
            times[times.len() / 2]
        );
    }
    let _ = std::fs::remove_dir_all(&out_root);
    println!("\npeak RSS {:.0} MB", peak_rss_mb());
}
