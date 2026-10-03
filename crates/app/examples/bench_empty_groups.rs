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
// Usage: cargo run --release -p evanalyzer_app --example bench_empty_groups -- <path.evadb> [--iters N] [--only <query name>]

use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::results::Aggregation;
use evanalyzer_app::results::BoxplotFilter;
use evanalyzer_app::results::ColorScale;
use evanalyzer_app::results::ColorSchema;
use evanalyzer_app::results::Column;
use evanalyzer_app::results::GroupedByImageFilter;
use evanalyzer_app::results::HistogramFilter;
use evanalyzer_app::results::ImageHeatmapFilter;
use evanalyzer_app::results::ListFilter;
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
    // `--only <text>`: run just the queries whose name contains it - in a
    // fresh process each, so `peak RSS` is that query's own peak.
    let only: Option<String> = {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == "--only")
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let wanted = |name: &str| only.as_deref().is_none_or(|only| name.contains(only));
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
    if wanted("plate Count") {
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
    }
    if wanted("plate Avg area") {
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
    }

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
    if wanted("plate_multi Count+area x Avg+Sum") {
        report(
            "plate_multi Count+area x Avg+Sum",
            time(iters, || {
                generator
                    .get_group_by_plate_multi(&plate_multi, &View::List)
                    .unwrap()
                    .len()
            }),
        );
    }

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
    if wanted("well Count") {
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
    }

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
    if wanted("wells_for_plate Sum area") {
        report(
            "wells_for_plate Sum area",
            time(iters, || {
                generator
                    .get_wells_for_plate(&wells, &View::List)
                    .unwrap()
                    .len()
            }),
        );
    }

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
    if wanted("wells_for_plate_multi Count+area x Avg+Sum") {
        report(
            "wells_for_plate_multi Count+area x Avg+Sum",
            time(iters, || {
                generator
                    .get_wells_for_plate_multi(&wells_multi, &View::List)
                    .unwrap()
                    .len()
            }),
        );
    }

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
        if wanted(&format!("per-image {label}, GUI first page (500)")) {
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
        }
        if wanted(&format!("per-image {label}, export (pages of 20000)")) {
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
    if wanted("image heatmap Count") {
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
    }
    // Charts and the transposed object list, through the same
    // `ResultsSource` the GUI uses (the chart code isn't exported on its own).
    let source = LocalBackend::default()
        .open_results(std::path::Path::new(&path))
        .expect("open results");
    if wanted("boxplot area, every class") {
        report(
            "boxplot area, every class",
            time(iters, || {
                source
                    .boxplot(&BoxplotFilter {
                        plane: plane.clone(),
                        images: None,
                        object_classes: None,
                        column: Column::AreaSizePx,
                    })
                    .unwrap()
                    .boxes
                    .len()
            }),
        );
    }
    if wanted("histogram area, one class") {
        report(
            "histogram area, one class",
            time(iters, || {
                source
                    .histogram(&HistogramFilter {
                        plane: plane.clone(),
                        images: None,
                        object_classes: Some(vec![ObjectClass::Valid(4)]),
                        column: Column::AreaSizePx,
                        bins: 50,
                    })
                    .unwrap()
                    .counts
                    .len()
            }),
        );
    }
    if wanted("object list transposed, GUI first page (500)") {
        report(
            "object list transposed, GUI first page (500)",
            time(iters, || {
                source
                    .get_object_list(&ListFilter {
                        plane: plane.clone(),
                        images: None,
                        object_classes: None,
                        columns: vec![Column::AreaSizePx],
                        with_coloc_details: false,
                        page: Pagination {
                            limit: 500,
                            after: None,
                        },
                        transpond_table: true,
                    })
                    .unwrap()
                    .rows
                    .len()
            }),
        );
    }

    // Intensity and colocalization columns. `get_available_columns` scans
    // for the classes that colocalize; a fresh generator each run, so its
    // cache doesn't hide that.
    if wanted("available columns") {
        report(
            "available columns",
            time(iters, || {
                LocalResultsGenerator::open_database(PathBuf::from(&path))
                    .unwrap()
                    .get_available_columns()
                    .unwrap()
                    .len()
            }),
        );
    }
    // Only looked up when a query below needs it - it scans every object's
    // colocalization data, which would inflate the peak RSS of a `--only`
    // run of an unrelated query.
    let needs_columns = [
        "object list intensities",
        "object list coloc",
        "plate coloc",
        "available",
    ]
    .iter()
    .any(|name| {
        only.as_deref()
            .is_none_or(|only| name.contains(only) || only.contains(name))
    });
    let available = if needs_columns {
        generator.get_available_columns().unwrap()
    } else {
        Vec::new()
    };
    let coloc_column = available
        .iter()
        .find(|c| matches!(c.key, Column::ColocCount(_)))
        .map(|c| c.key.clone());
    let intensity_columns: Vec<Column> = available
        .iter()
        .filter(|c| {
            matches!(
                c.key,
                Column::IntensityAvg(_)
                    | Column::IntensitySum(_)
                    | Column::IntensityMin(_)
                    | Column::IntensityMax(_)
            )
        })
        .map(|c| c.key.clone())
        .collect();
    let list_page = |columns: Vec<Column>, with_coloc_details| ListFilter {
        plane: plane.clone(),
        images: None,
        object_classes: None,
        columns,
        with_coloc_details,
        page: Pagination {
            limit: 500,
            after: None,
        },
        transpond_table: false,
    };
    if wanted("object list intensities + coloc count") {
        let mut columns = intensity_columns.clone();
        columns.extend(coloc_column.clone());
        report(
            "object list intensities + coloc count (500)",
            time(iters, || {
                source
                    .get_object_list(&list_page(columns.clone(), false))
                    .unwrap()
                    .rows
                    .len()
            }),
        );
    }
    if wanted("object list coloc details") {
        let mut columns = vec![Column::AreaSizePx];
        columns.extend(coloc_column.clone());
        report(
            "object list coloc details (500)",
            time(iters, || {
                source
                    .get_object_list(&list_page(columns.clone(), true))
                    .unwrap()
                    .rows
                    .len()
            }),
        );
    }
    if wanted("plate coloc count") {
        if let Some(coloc) = &coloc_column {
            let filter = plate(coloc.clone(), Aggregation::Avg);
            match generator.get_group_by_plate(&filter, &View::List) {
                Ok(_) => report(
                    "plate coloc count",
                    time(iters, || {
                        generator
                            .get_group_by_plate(&filter, &View::List)
                            .unwrap()
                            .rows
                            .len()
                    }),
                ),
                Err(e) => println!("{:<42} n/a ({e})", "plate coloc count"),
            }
        }
    }
    if wanted("plate intensity avg") {
        let probe = generator.get_group_by_plate(
            &plate(Column::IntensityAvg(0), Aggregation::Avg),
            &View::List,
        );
        match probe {
            Ok(_) => report(
                "plate intensity avg",
                time(iters, || {
                    generator
                        .get_group_by_plate(
                            &plate(Column::IntensityAvg(0), Aggregation::Avg),
                            &View::List,
                        )
                        .unwrap()
                        .rows
                        .len()
                }),
            ),
            Err(e) => println!("{:<42} n/a ({e})", "plate intensity avg"),
        }
    }

    println!("peak RSS {:.0} MB", peak_rss_mb());
}
