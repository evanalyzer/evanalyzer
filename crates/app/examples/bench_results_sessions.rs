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
// `--together`: every file open at once instead (the results window and a
// running analysis), each read in full (the list as CSV: every object's
// columns) - RSS and DuckDB's memory with all of them open.
//
// `--max-ram MB` first: the app's RAM limit (as `evanalyzer --max-ram`) -
// a smaller computer for DuckDB's limit.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_results_sessions -- \
//     <out dir> [--max-ram MB] [--together] <a.evadb> <b.evadb> ...

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

/// DuckDB's memory of `db` (open), in MB.
fn duckdb_mb(db: &str) -> i64 {
    evanalyzer_core::open_results_database(std::path::Path::new(db))
        .and_then(|conn| {
            conn.query_row(
                "SELECT sum(memory_usage_bytes)::BIGINT FROM duckdb_memory()",
                [],
                |r| r.get::<_, i64>(0),
            )
            .map_err(|e| evanalyzer_cfg::core_types::InternalErrors::Io(e.to_string()))
        })
        .unwrap_or(0)
        / 1_000_000
}

fn together(out: &std::path::Path, dbs: &[String]) {
    let databases: Vec<_> = dbs
        .iter()
        .map(|db| LocalResultsGenerator::open_database(PathBuf::from(db)).expect("open database"))
        .collect();
    let start = Instant::now();
    for (i, database) in databases.iter().enumerate() {
        let _ = std::fs::remove_dir_all(out);
        std::fs::create_dir_all(out).unwrap();
        let export = ResultExport {
            output_dir: out.to_path_buf(),
            format: ExportFormat::CSV,
            columns: vec![
                Column::AreaSizePx,
                Column::IntensityAvg(0),
                Column::Circularity,
            ],
            with_list_view: true,
            ..Default::default()
        };
        export
            .start_export(database, &AtomicBool::new(false), &mut |_, _, _| {})
            .expect("export");
        let duckdb: Vec<String> = dbs[..=i]
            .iter()
            .map(|db| duckdb_mb(db).to_string())
            .collect();
        println!(
            "file {} read: rss={} MB, duckdb per file=[{}] MB",
            i + 1,
            status_mb("VmRSS:"),
            duckdb.join(", ")
        );
    }
    println!(
        "all read in {:.1}s, peak_rss={} MB",
        start.elapsed().as_secs_f64(),
        status_mb("VmHWM:")
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.get(2).is_some_and(|a| a == "--max-ram") {
        let mb: u64 = args[3].parse().expect("--max-ram MB");
        evanalyzer_app::backends::local::set_resource_limits(
            evanalyzer_app::backends::local::ResourceLimits {
                max_threads: None,
                max_ram_mb: Some(mb),
            },
        );
        args.drain(2..4);
    }
    let out = PathBuf::from(&args[1]);
    println!("start: rss={} MB", status_mb("VmRSS:"));
    if args[2] == "--together" {
        together(&out, &args[3..]);
        return;
    }
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
