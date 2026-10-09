// Times the plate-view queries (`get_group_by_plate`, `get_group_by_plate_multi`)
// per column kind - Count (aggregated over per-image counts), a plain
// per-object column and an intensity - against a real `.evadb` file.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_plate_count -- <path.evadb> [--iters N]

use evanalyzer_app::results::{
    Aggregation, ColorScale, ColorSchema, Column, LocalResultsGenerator, PlaneFilter, PlateFilter,
    PlateFilterMulti, View,
};
use evanalyzer_cfg::core_types::ObjectClass;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn time<T>(iters: usize, mut f: impl FnMut() -> T) -> (Duration, Duration, T) {
    let mut last = f(); // warm-up
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        last = f();
        samples.push(start.elapsed());
    }
    samples.sort();
    (samples[samples.len() / 2], samples[0], last)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_plate_count <path.evadb> [--iters N]");
    let iters = arg_value("--iters", 10);
    let generator = LocalResultsGenerator::open_database(PathBuf::from(&path)).expect("open db");
    let plane = PlaneFilter {
        z_stack: 0,
        t_stack: 0,
    };
    let aggregations = vec![
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
        Aggregation::Sum,
    ];
    let columns = [
        ("Count", Column::Count),
        ("AreaSizePx", Column::AreaSizePx),
        ("IntensityAvg(0)", Column::IntensityAvg(0)),
    ];
    let classes = [
        ("Unset", ObjectClass::Unset),
        ("Valid(1)", ObjectClass::Valid(1)),
    ];

    println!("iters={iters} (median / min)");
    for (class_name, class) in classes {
        for (name, column) in &columns {
            let filter = PlateFilter {
                plane: plane.clone(),
                grouping_regex: String::new(),
                aggregation: Aggregation::Avg,
                object_class: class,
                column: column.clone(),
                color_schema: ColorSchema::default(),
                color_scale: ColorScale::default(),
                matrix_dimension: None,
            };
            let (median, min, result) = time(iters, || {
                generator
                    .get_group_by_plate(&filter, &View::Heatmap)
                    .expect("plate")
            });
            println!(
                "get_group_by_plate       {class_name:9} {name:16} {median:>10.2?} / {min:>10.2?}  rows={}",
                result.rows.len()
            );
        }
    }

    for (label, cols) in [
        ("Count", vec![Column::Count]),
        ("AreaSizePx", vec![Column::AreaSizePx]),
        (
            "Count+Area+Int",
            columns.iter().map(|(_, c)| c.clone()).collect(),
        ),
    ] {
        let filter = PlateFilterMulti {
            plane: plane.clone(),
            grouping_regex: String::new(),
            aggregation: aggregations.clone(),
            object_class: vec![ObjectClass::Unset],
            column: cols,
            color_schema: ColorSchema::default(),
            color_scale: ColorScale::default(),
            matrix_dimension: None,
        };
        let (median, min, result) = time(iters, || {
            generator
                .get_group_by_plate_multi(&filter, &View::List)
                .expect("plate multi")
        });
        println!(
            "get_group_by_plate_multi {label:26} {median:>10.2?} / {min:>10.2?}  results={}",
            result.len()
        );
    }
}
