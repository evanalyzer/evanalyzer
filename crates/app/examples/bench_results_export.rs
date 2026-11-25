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
// a project export (`ResultsSource::run_export`: OBJECTS / IMAGES / PLATE),
// or the raw tables as Parquet (`export_parquet`). Reports
// wall time and peak RSS - one export per process, so the peak is its own.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_results_export -- \
//     <path.evadb> <out dir> --mode list_csv|list_xlsx|grouped_xlsx|grouped_csv|plate|parquet

use evanalyzer_app::backend::Backend;
use evanalyzer_app::backend::local::LocalBackend;
use evanalyzer_app::exports::{ExportRun, ScriptPermission};
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

    let results = LocalBackend::default()
        .open_results(&db)
        .expect("open results");
    let (output, format) = match mode.as_str() {
        "list_csv" => ("OBJECTS", "CSV"),
        "list_xlsx" => ("OBJECTS", "XLSX"),
        "grouped_csv" => ("IMAGES", "CSV"),
        "grouped_xlsx" => ("IMAGES", "XLSX"),
        "plate" => ("PLATE", "XLSX"),
        "parquet" => ("", ""),
        other => panic!("unknown mode {other}"),
    };
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut steps = 0usize;
    let result = if output.is_empty() {
        // The whole objects table as Parquet (DuckDB's parquet extension).
        results.export_parquet(&out, &cancel, &mut |_, _, _| steps += 1)
    } else {
        let export = serde_json::from_value(serde_json::json!({
            "id": 1, "name": "bench", "entries": [{ "kind": "DATA", "outputs": [output],
                "format": format, "images": "INCLUDED",
                "columns": ["count", "area_px", "intensity_mean_gray_ch0", "circularity"],
                "aggregations": ["AVG", "MEDIAN"] }]
        }))
        .expect("an export");
        results
            .run_export(
                &ExportRun {
                    export,
                    project_dir: out.clone(),
                    results_dir: out.clone(),
                    output: Some(out.clone()),
                    scripts: ScriptPermission::Skip,
                    interpreters: Default::default(),
                    drawing: None,
                },
                &cancel,
                &mut |_, _, _| steps += 1,
            )
            .and_then(|report| match report.all_written() {
                true => Ok(()),
                false => Err(evanalyzer_cfg::core_types::InternalErrors::Internal(
                    format!("{report:?}"),
                )),
            })
    };
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
