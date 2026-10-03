// Throwaway benchmark (not part of the test suite), run against a real,
// large `.evadb` file before and after a change to how the results queries
// treat images without objects (Count/Sum 0 for analysed images, empty for
// failed/disabled/not-analysed ones, per-image rows for images with no
// objects). Times every query that change touches, each as the median of
// `--iters` runs after one warm-up run:
//   - plate, plate_multi, well, wells_for_plate, wells_for_plate_multi
//   - the per-image table (normal and transposed): the GUI's first page
//     (500 rows), and every page walked via keyset pagination at the
//     export's page size (20000)
//   - the image heatmap of the first image
//
// Usage: cargo run --release -p evanalyzer_app --example bench_empty_groups -- <path.evadb> [--iters N]

use evanalyzer_app::results::Aggregation;
use evanalyzer_app::results::ColorScale;
use evanalyzer_app::results::ColorSchema;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::GroupedByImageFilter;
use evanalyzer_app::results::ImageHeatmapFilter;
use evanalyzer_app::results::LocalResultsGenerator;
use evanalyzer_app::results::Pagination;
use evanalyzer_app::results::PlaneFilter;
use evanalyzer_app::results::PlateFilter;
use evanalyzer_app::results::PlateFilterMulti;
use evanalyzer_app::results::View;
use evanalyzer_app::results::WellFilter;
use evanalyzer_app::results::WellsBatchFilter;
use evanalyzer_app::results::WellsBatchFilterMulti;
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

/// Peak resident set size ("high water mark") since process start, in MB.
/// Linux-only (`/proc/self/status`); 0.0 elsewhere.
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

/// Median of `iters` timed runs, after one untimed warm-up run. Returns the
/// median and whatever the last run reported (a row count, for sanity).
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
        "{name:<42} {:>10.1} ms  ({rows} rows)",
        median.as_secs_f64() * 1e3
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_empty_groups <path.evadb> [--iters N]");
    let iters = arg_value("--iters", 5).max(1);
    let generator = LocalResultsGenerator::open_database(PathBuf::from(&path)).expect("open db");

    let plane = PlaneFilter {
        z_stack: 0,
        t_stack: 0,
    };
    let images = generator.get_images().expect("images");
    let first_image = images
        .first()
        .expect("at least one image")
        .rel_path
        .to_string_lossy()
        .into_owned();
    println!("{path}: {} images, {iters} iters", images.len());

    let plate = |column: Column, aggregation: Aggregation| PlateFilter {
        plane: plane.clone(),
        grouping_regex: String::new(),
        aggregation,
        object_class: ObjectClass::Unset,
        column,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        matrix_dimension: None,
    };
    report(
        "plate Count",
        time(iters, || {
            generator
                .get_group_by_plate(&plate(Column::Count, Aggregation::Avg), &View::List)
                .unwrap()
                .rows
                .len()
        }),
    );
    report(
        "plate Avg area",
        time(iters, || {
            generator
                .get_group_by_plate(&plate(Column::AreaSizePx, Aggregation::Avg), &View::List)
                .unwrap()
                .rows
                .len()
        }),
    );

    let plate_multi = PlateFilterMulti {
        plane: plane.clone(),
        grouping_regex: String::new(),
        aggregation: vec![Aggregation::Avg, Aggregation::Sum],
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::Count, Column::AreaSizePx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        matrix_dimension: None,
    };
    report(
        "plate_multi Count+area x Avg+Sum",
        time(iters, || {
            generator
                .get_group_by_plate_multi(&plate_multi, &View::List)
                .unwrap()
                .len()
        }),
    );

    let first_well = generator
        .get_group_by_plate(&plate(Column::Count, Aggregation::Avg), &View::List)
        .unwrap()
        .row_names
        .first()
        .cloned()
        .unwrap_or_default();
    let well = WellFilter {
        plane: plane.clone(),
        group_name: first_well,
        grouping_regex: String::new(),
        aggregation: Aggregation::Sum,
        object_class: ObjectClass::Unset,
        column: Column::Count,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    report(
        "well Count",
        time(iters, || {
            generator
                .get_group_by_well(&well, &View::List)
                .unwrap()
                .rows
                .len()
        }),
    );

    let wells = WellsBatchFilter {
        plane: plane.clone(),
        grouping_regex: String::new(),
        aggregation: Aggregation::Sum,
        object_class: ObjectClass::Unset,
        column: Column::AreaSizePx,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    report(
        "wells_for_plate Sum area",
        time(iters, || {
            generator
                .get_wells_for_plate(&wells, &View::List)
                .unwrap()
                .len()
        }),
    );

    let wells_multi = WellsBatchFilterMulti {
        plane: plane.clone(),
        grouping_regex: String::new(),
        aggregation: vec![Aggregation::Avg, Aggregation::Sum],
        object_class: vec![ObjectClass::Unset],
        column: vec![Column::Count, Column::AreaSizePx],
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        well_size: None,
        well_order: None,
    };
    report(
        "wells_for_plate_multi Count+area x Avg+Sum",
        time(iters, || {
            generator
                .get_wells_for_plate_multi(&wells_multi, &View::List)
                .unwrap()
                .len()
        }),
    );

    for transposed in [false, true] {
        let by_image = |after: Option<String>, limit: i32| GroupedByImageFilter {
            plane: plane.clone(),
            images: None,
            object_classes: None,
            columns: vec![Column::Count, Column::AreaSizePx],
            aggregation: vec![Aggregation::Avg, Aggregation::Sum],
            page: Pagination { limit, after },
            transpond_table: transposed,
        };
        let label = if transposed { "transposed" } else { "normal" };
        report(
            &format!("per-image {label}, GUI first page (500)"),
            time(iters, || {
                generator
                    .get_grouped_by_image(&by_image(None, 500))
                    .unwrap()
                    .rows
                    .len()
            }),
        );
        report(
            &format!("per-image {label}, export (pages of 20000)"),
            time(iters, || {
                let mut total = 0;
                let mut after = None;
                loop {
                    let page = generator
                        .get_grouped_by_image(&by_image(after, 20_000))
                        .unwrap();
                    total += page.rows.len();
                    match page.row_names.last() {
                        Some(last) if page.rows.len() == 20_000 => after = Some(last.clone()),
                        _ => break,
                    }
                }
                total
            }),
        );
    }

    let heatmap = ImageHeatmapFilter {
        plane: plane.clone(),
        image_rel_path: first_image,
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column: Column::Count,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        square_size: None,
    };
    report(
        "image heatmap Count",
        time(iters, || {
            generator
                .get_image_heatmap(&heatmap, &View::Heatmap)
                .unwrap()
                .rows
                .len()
        }),
    );
    println!("peak RSS {:.0} MB", peak_rss_mb());
}
