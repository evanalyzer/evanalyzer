//! # bench_measurements
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Throwaway benchmark (not part of the test suite): how fast the results
// read per-object measurement values (the `measurements` table, one row
// per value) compared with a plain `objects` column (Area) - plate, per
// image, the object list and a chart, each the median of `--iters` runs
// after one warm-up run.
//
// `--prepare K` first adds K measurements with a value per object to the
// file (a copy - it is changed): `bench_k/value_0` .. `bench_k/value_{K-1}`,
// in `measurements` and in `objects.measurement_values`.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_measurements -- <copy.evadb> [--prepare K] [--iters N] [--only <query name>]

use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::results::Aggregation;
use evanalyzer_app::results::ColorScale;
use evanalyzer_app::results::ColorSchema;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::GroupedByImageFilter;
use evanalyzer_app::results::HistogramFilter;
use evanalyzer_app::results::ListFilter;
use evanalyzer_app::results::LocalResultsGenerator;
use evanalyzer_app::results::MeasurementColumn;
use evanalyzer_app::results::Pagination;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_app::results::PlateFilter;
use evanalyzer_app::results::View;
use evanalyzer_app::results::{Grouping, PlateSize};
use evanalyzer_cfg::core_types::ObjectClass;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Peak resident set size since process start, in MB (Linux).
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

fn time<T>(iters: usize, mut run: impl FnMut() -> T) -> (Duration, T) {
    let mut last = run();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        last = run();
        times.push(start.elapsed());
    }
    times.sort();
    (times[times.len() / 2], last)
}

fn report(name: &str, (median, rows): (Duration, usize)) {
    println!(
        "{name:<44} {:>10.1} ms  ({rows} rows)  peak RSS {:.0} MB",
        median.as_secs_f64() * 1e3,
        peak_rss_mb()
    );
}

/// Adds `k` per-object measurements (deterministic values from the
/// object's area) to the file at `path`.
fn prepare(path: &str, k: usize) {
    let conn = duckdb::Connection::open(path).expect("open for prepare");
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS measurement_info (
            key VARCHAR NOT NULL PRIMARY KEY, measurement VARCHAR NOT NULL,
            step VARCHAR NOT NULL, unit VARCHAR NOT NULL, target VARCHAR NOT NULL,
            range_min DOUBLE, range_max DOUBLE, aggregatable BOOLEAN NOT NULL,
            object_index INTEGER);
         ALTER TABLE objects ADD COLUMN IF NOT EXISTS measurement_values DOUBLE[];
         CREATE TABLE IF NOT EXISTS measurements (
            image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL,
            z_stack INTEGER NOT NULL, t_stack INTEGER NOT NULL, region VARCHAR NOT NULL,
            object_id UUID, related_object_id UUID, object_class_name VARCHAR,
            object_class_id INTEGER[], step VARCHAR NOT NULL, measurement VARCHAR NOT NULL,
            unit VARCHAR NOT NULL, key VARCHAR NOT NULL, value DOUBLE);",
    )
    .expect("create tables");
    let start = Instant::now();
    for i in 0..k {
        conn.execute_batch(&format!(
            "INSERT OR IGNORE INTO measurement_info VALUES
                ('bench_k/value_{i}', 'Value {i}', 'bench_k', 'None', 'Object', NULL, NULL, true,
                 {index});
             INSERT INTO measurements
             SELECT image_name, image_rel_path, z_stack, t_stack, 'object', object_id, NULL,
                    object_class_name, object_class_id, 'bench_k', 'Value {i}', 'None',
                    'bench_k/value_{i}', area_px * {f} FROM objects;",
            f = (i + 1) as f64 * 0.5,
            index = i + 1
        ))
        .expect("insert values");
    }
    // ... and on each object, as the analysis writes them (an UPDATE keeps
    // the old row versions: the file grows more than a fresh run's would).
    conn.execute_batch(&format!(
        "UPDATE objects SET measurement_values = [{}]",
        (0..k)
            .map(|i| format!("area_px * {}", (i + 1) as f64 * 0.5))
            .collect::<Vec<_>>()
            .join(", ")
    ))
    .expect("values on the objects");
    let rows: i64 = conn
        .query_row("SELECT count(*) FROM measurements", [], |r| r.get(0))
        .unwrap();
    println!(
        "prepared {k} measurement(s): {rows} rows in {:.1} s",
        start.elapsed().as_secs_f64()
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_measurements <copy.evadb> [--prepare K] [--iters N]");
    if let Some(k) = arg_value("--prepare").and_then(|v| v.parse().ok()) {
        prepare(&path, k);
        return;
    }
    let iters = arg_value("--iters")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5usize)
        .max(1);
    let only = arg_value("--only");
    let wanted = |name: &str| only.as_deref().is_none_or(|only| name.contains(only));

    let generator = LocalResultsGenerator::open_database(PathBuf::from(&path)).expect("open db");
    let source = LocalBackend::default()
        .open_results(std::path::Path::new(&path))
        .expect("open results");
    let plane = PlaneFilter {
        z_stack: 0,
        t_stack: 0,
    };
    let measurement = Column::Measurement(MeasurementColumn {
        key: "bench_k/value_0".into(),
        step: "bench_k".into(),
        name: "Value 0".into(),
        per_object: true,
    });
    println!("{path}: {iters} iters");

    for (label, column) in [("area", Column::AreaSizePx), ("measurement", measurement)] {
        let plate = PlateFilter {
            plane: plane.clone(),
            grouping: Grouping::Auto,
            aggregation: Aggregation::Avg,
            object_class: ObjectClass::Unset,
            column: column.clone(),
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            plate_size: PlateSize::Auto,
        };
        let name = format!("plate Avg {label}");
        if wanted(&name) {
            report(
                &name,
                time(iters, || {
                    generator
                        .get_group_by_plate(&plate, &View::List)
                        .unwrap()
                        .rows
                        .len()
                }),
            );
        }
        let name = format!("per-image Avg {label} (first page, 500)");
        if wanted(&name) {
            let filter = GroupedByImageFilter {
                plane: plane.clone(),
                images: None,
                object_classes: None,
                columns: vec![column.clone()],
                aggregation: vec![Aggregation::Avg],
                page: Pagination {
                    limit: 500,
                    after: None,
                },
                transpond_table: false,
            };
            report(
                &name,
                time(iters, || {
                    generator.get_grouped_by_image(&filter).unwrap().rows.len()
                }),
            );
        }
        let name = format!("object list {label} (first page, 500)");
        if wanted(&name) {
            let filter = ListFilter {
                plane: plane.clone(),
                images: None,
                object_classes: None,
                columns: vec![Column::AreaSizePx, column.clone()],
                with_coloc_details: false,
                related_to: None,
                page: Pagination {
                    limit: 500,
                    after: None,
                },
                transpond_table: false,
            };
            report(
                &name,
                time(iters, || {
                    source.get_object_list(&filter).unwrap().rows.len()
                }),
            );
        }
        let name = format!("histogram {label}");
        if wanted(&name) {
            let filter = HistogramFilter {
                plane: plane.clone(),
                images: None,
                object_classes: None,
                column: column.clone(),
                bins: 50,
            };
            report(
                &name,
                time(iters, || source.histogram(&filter).unwrap().counts.len()),
            );
        }
    }
}
