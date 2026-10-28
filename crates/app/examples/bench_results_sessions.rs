//! # bench_results_sessions
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark: several results files opened one after another in one process,
// as the results window does when the user switches between analyses - each
// opened, read (a grouped-by-image export: every object once), and closed
// again before the next. Reports RSS after each file is closed: what the
// earlier files still hold.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_results_sessions -- \
//     <out dir> <a.evadb> <b.evadb> ...

use evanalyzer_app::results::{
    Aggregation, Column, ExportFormat, LocalResultsGenerator, ResultExport,
};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn status_mb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        / 1024
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(&args[1]);
    println!("start: rss={} MB", status_mb("VmRSS:"));
    for (i, db) in args[2..].iter().enumerate() {
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let start = Instant::now();
        {
            let database =
                LocalResultsGenerator::open_database(PathBuf::from(db)).expect("open database");
            let export = ResultExport {
                output_dir: out.clone(),
                format: ExportFormat::CSV,
                columns: vec![Column::Count, Column::AreaSizePx, Column::IntensityAvg(0)],
                aggregations: vec![Aggregation::Avg, Aggregation::Median],
                with_grouped_by_image_list: true,
                ..Default::default()
            };
            export
                .start_export(&database, &AtomicBool::new(false), &mut |_, _, _| {})
                .expect("export");
            println!(
                "file {}: open + read {:.1}s, rss while open={} MB",
                i + 1,
                start.elapsed().as_secs_f64(),
                status_mb("VmRSS:")
            );
        }
        // Give a closing database a moment (its checkpoint).
        std::thread::sleep(std::time::Duration::from_millis(500));
        let wal = std::fs::metadata(format!("{db}.wal"))
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "file {}: closed - rss={} MB, open files={}, wal={} MB",
            i + 1,
            status_mb("VmRSS:"),
            std::fs::read_dir("/proc/self/fd")
                .map(|d| d.count())
                .unwrap_or(0),
            wal / 1_000_000
        );
    }
    println!("peak_rss={} MB", status_mb("VmHWM:"));
}
