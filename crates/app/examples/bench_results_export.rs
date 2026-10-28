//! # bench_results_export
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark: one export of every image of an existing `.evadb` (e.g. made
// with core's `bench_export --images 4000 --objects-per-image 1000`) through
// `ResultExport::start_export`, as the results window starts it. Reports
// wall time and peak RSS - one export per process, so the peak is its own.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_results_export -- \
//     <path.evadb> <out dir> --mode list_csv|list_xlsx|grouped_xlsx|grouped_csv

use evanalyzer_app::results::{
    Aggregation, Column, ExportFormat, LocalResultsGenerator, ResultExport,
};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn arg_str(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn peak_rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        / 1024
}

fn main() {
    let db = PathBuf::from(std::env::args().nth(1).expect("path.evadb"));
    let out = PathBuf::from(std::env::args().nth(2).expect("out dir"));
    let mode = arg_str("--mode", "list_csv");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    let database = LocalResultsGenerator::open_database(db).expect("open database");
    let (format, list, grouped) = match mode.as_str() {
        "list_csv" => (ExportFormat::CSV, true, false),
        "list_xlsx" => (ExportFormat::XLSX, true, false),
        "grouped_csv" => (ExportFormat::CSV, false, true),
        "grouped_xlsx" => (ExportFormat::XLSX, false, true),
        other => panic!("unknown mode {other}"),
    };
    let export = ResultExport {
        output_dir: out.clone(),
        format,
        columns: vec![
            Column::Count,
            Column::AreaSizePx,
            Column::IntensityAvg(0),
            Column::Circularity,
        ],
        aggregations: vec![Aggregation::Avg, Aggregation::Median],
        with_list_view: list,
        with_list_one_file_per_image: false,
        with_grouped_by_image_list: grouped,
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut steps = 0usize;
    let result = export.start_export(&database, &cancel, &mut |_msg, _cur, _total| steps += 1);
    let size: u64 = std::fs::read_dir(&out)
        .unwrap()
        .flatten()
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum();
    println!(
        "{mode}: result={:?} time={:.1}s peak_rss={} MB output={} MB progress_calls={steps}",
        result.err(),
        start.elapsed().as_secs_f64(),
        peak_rss_mb(),
        size / 1_000_000
    );
}
