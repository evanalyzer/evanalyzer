//! # bench_track_motion
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Benchmark (not part of the test suite) of Measure Track Motion on a
// tracked results database (make one with `make_time_series`): runs the
// step once and prints its time, the rows it wrote and the peak RSS
// (`VmHWM`, Linux). Writes into the file - use a copy.
//
// Usage: cargo run --release -p evanalyzer_core --example bench_track_motion -- \
//     <copy.evadb> [--class N] [--drift-class N] [--id ID] [--no-write]

use evanalyzer_cfg::core_types::{InternalErrors, MeasurementInfo};
use evanalyzer_cfg::core_types::{ObjectClass, SizeUnits};
use evanalyzer_core::{
    DatasetAlgorithm, DatasetContext, DatasetStore, DuckDbExporter, MeasureTrackMotion,
    ObjectFilter, PipelineResultExporter, StoredObject,
};
use evanalyzer_core::{DatasetValue, TrackAssignment};

/// `--no-write`: reads as the step does, writes nothing - the step's own
/// time and RAM without the database's write.
struct NoWrite<'a>(&'a dyn DatasetStore);

impl DatasetStore for NoWrite<'_> {
    fn get_objects(&self, filter: &ObjectFilter) -> Result<Vec<StoredObject>, InternalErrors> {
        self.0.get_objects(filter)
    }
    fn image_rel_paths(&self, class: ObjectClass) -> Result<Vec<String>, InternalErrors> {
        self.0.image_rel_paths(class)
    }
    fn set_track_ids(&self, _: &[TrackAssignment]) -> Result<(), InternalErrors> {
        Ok(())
    }
    fn max_track_id_except(&self, class: ObjectClass) -> Result<u64, InternalErrors> {
        self.0.max_track_id_except(class)
    }
    fn write_measurements(
        &self,
        _: &str,
        _: &[MeasurementInfo],
        values: &[DatasetValue],
    ) -> Result<(), InternalErrors> {
        println!("(not written: {} values)", values.len());
        Ok(())
    }
}
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

fn arg_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

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
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: bench_track_motion <copy.evadb> [--class N] [--drift-class N]"),
    );
    let class = ObjectClass::Global(arg("--class").and_then(|c| c.parse().ok()).unwrap_or(1));
    let drift = arg("--drift-class")
        .and_then(|c| c.parse().ok())
        .map_or(ObjectClass::Unset, ObjectClass::Global);
    let exporter = DuckDbExporter::open_existing(&path).expect("open results");
    let rss_open = peak_rss_mb();
    let step = MeasureTrackMotion {
        id: arg("--id").unwrap_or_else(|| "motion_01".into()),
        objects: class,
        min_step: 1.0,
        unit: SizeUnits::Pixels,
        average_over: 2,
        drift_reference: drift,
    };
    let no_write = NoWrite(exporter.dataset_store().unwrap());
    let mut ctx = DatasetContext {
        store: match arg_flag("--no-write") {
            true => &no_write,
            false => exporter.dataset_store().unwrap(),
        },
        progress: None,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let start = Instant::now();
    step.execute(&mut ctx).expect("measure track motion");
    let took = start.elapsed();
    drop(ctx);
    drop(exporter);
    let conn = duckdb::Connection::open(&path).unwrap();
    let rows: i64 = conn
        .query_row(
            "SELECT count(*) FROM measurements WHERE step = ?",
            [arg("--id").unwrap_or_else(|| "motion_01".into())],
            |r| r.get(0),
        )
        .unwrap();
    println!(
        "Measure Track Motion: {:.0} ms, {rows} rows, peak RSS {:.0} MB (after opening {:.0} MB)",
        took.as_secs_f64() * 1e3,
        peak_rss_mb(),
        rss_open
    );
}
