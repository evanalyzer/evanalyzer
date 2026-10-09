// Replays what the results window does against a real `.evadb`, timing
// every step: opening the database (`open_database` in
// results_state_controller.rs - metadata, the first list page, the plate
// grid and the default histogram) and toggling an image's disabled flag
// (`on_toggle_active_well_disabled` - the update, then the well/plate grid
// refresh).
//
// The toggle writes to the database, so point it at a copy.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_open_results -- <copy.evadb> [--iters N]

use evanalyzer_app::backends::Backend;
use evanalyzer_app::backends::local::LocalBackend;
use evanalyzer_app::results::{
    Aggregation, ColorScale, ColorSchema, Column, HistogramFilter, ListFilter, Pagination,
    PlaneFilter, PlateFilter, ResultsSource, View, WellFilter,
};
use evanalyzer_cfg::core_types::ObjectClass;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Bytes this process has read from disk so far (Linux only; 0 elsewhere)
/// - what a cold start on a slow disk pays for.
fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|io| {
            io.lines()
                .find_map(|l| l.strip_prefix("read_bytes: ").map(str::to_string))
        })
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

struct Steps(Vec<(&'static str, Duration, u64)>);

impl Steps {
    fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let (start, bytes) = (Instant::now(), read_bytes());
        let result = f();
        self.0.push((name, start.elapsed(), read_bytes() - bytes));
        result
    }

    fn print(&self, title: &str) {
        let total: Duration = self.0.iter().map(|(_, d, _)| *d).sum();
        let mb = |b: u64| b as f64 / 1e6;
        let total_mb: u64 = self.0.iter().map(|(_, _, b)| *b).sum();
        println!(
            "-- {title}: {total:.2?}, {:.1} MB read from disk",
            mb(total_mb)
        );
        for (name, duration, bytes) in &self.0 {
            println!("   {name:28} {duration:>10.2?} {:>8.1} MB", mb(*bytes));
        }
    }
}

fn plane() -> PlaneFilter {
    PlaneFilter {
        z_stack: 0,
        t_stack: 0,
    }
}

fn open_sequence(backend: &LocalBackend, path: &PathBuf) -> (Arc<dyn ResultsSource>, Steps) {
    let mut s = Steps(Vec::new());
    let db = s.time("open_results", || backend.open_results(path).expect("open"));
    s.time("run_status", || db.run_status().ok());
    let classes = s.time("get_object_classes", || db.get_object_classes().unwrap());
    let columns = s.time("get_available_columns", || {
        db.get_available_columns().unwrap()
    });
    s.time("get_images", || db.get_images().unwrap());
    s.time("get_nr_of_t/z_stacks", || {
        (db.get_nr_of_t_stacks(), db.get_nr_of_z_stacks())
    });
    s.time("get_object_list (page 1)", || {
        db.get_object_list(&ListFilter {
            plane: plane(),
            images: None,
            object_classes: Some(classes.iter().map(|c| c.id).collect()),
            columns: vec![
                Column::ObjectId,
                Column::ImageName,
                Column::ObjectClass,
                Column::AreaSizeNm,
            ],
            with_coloc_details: false,
            page: Pagination {
                limit: 500,
                after: None,
            },
            transpond_table: false,
        })
        .unwrap()
    });
    let matrix_column = columns
        .iter()
        .map(|c| c.key.clone())
        .find(|c| {
            !matches!(
                c,
                Column::ObjectId | Column::ImageName | Column::ObjectClass
            )
        })
        .unwrap_or_default();
    s.time("get_group_by_plate", || {
        db.get_group_by_plate(&plate_filter(matrix_column.clone()), &View::Heatmap)
            .unwrap()
    });
    let chart_column = columns
        .iter()
        .map(|c| c.key.clone())
        .find(|c| {
            !matches!(
                c,
                Column::ObjectId
                    | Column::ImageName
                    | Column::ObjectClass
                    | Column::Count
                    | Column::IntensityAvg(_)
                    | Column::IntensitySum(_)
                    | Column::IntensityMin(_)
                    | Column::IntensityMax(_)
            )
        })
        .unwrap_or_default();
    s.time("histogram", || {
        db.histogram(&HistogramFilter {
            plane: plane(),
            images: None,
            object_classes: None,
            column: chart_column,
            bins: 24,
        })
        .unwrap()
    });
    (db, s)
}

fn plate_filter(column: Column) -> PlateFilter {
    PlateFilter {
        plane: plane(),
        grouping_regex: String::new(),
        aggregation: Aggregation::Avg,
        object_class: ObjectClass::Unset,
        column,
        color_schema: ColorSchema::default(),
        color_scale: ColorScale::default(),
        matrix_dimension: None,
    }
}

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: bench_open_results <copy.evadb> [--iters N]"),
    );
    let iters = arg_value("--iters", 3);
    let backend =
        LocalBackend::restricted_to(&[path.parent().unwrap().to_path_buf()]).expect("backend");

    // The first open in this process pays for opening the file; later
    // ones reuse the cached connection (see `shared_connection`).
    let (db, steps) = open_sequence(&backend, &path);
    steps.print("open (first in process)");
    for i in 0..iters {
        let (_, steps) = open_sequence(&backend, &path);
        steps.print(&format!("open (again #{i})"));
    }

    let images = db.get_images().unwrap();
    let image = images.first().expect("an image");
    let rel_path = image.rel_path.to_string_lossy().into_owned();
    let well = image.name.split('_').next().unwrap_or_default().to_string();
    let column = Column::Count;
    for i in 0..iters * 2 {
        let disable = i % 2 == 0;
        let mut s = Steps(Vec::new());
        s.time("enable_image", || {
            db.enable_image(&rel_path, disable).unwrap()
        });
        s.time("get_group_by_well", || {
            db.get_group_by_well(
                &WellFilter {
                    plane: plane(),
                    group_name: well.clone(),
                    grouping_regex: String::new(),
                    aggregation: Aggregation::Avg,
                    object_class: ObjectClass::Unset,
                    column: column.clone(),
                    color_schema: ColorSchema::default(),
                    color_scale: ColorScale::default(),
                    well_size: None,
                    well_order: None,
                },
                &View::Heatmap,
            )
            .unwrap()
        });
        s.time("get_group_by_plate", || {
            db.get_group_by_plate(&plate_filter(column.clone()), &View::Heatmap)
                .unwrap()
        });
        s.print(&format!("toggle {rel_path} disabled={disable}"));
    }
}
