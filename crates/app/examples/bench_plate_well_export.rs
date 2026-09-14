// Throwaway benchmark harness (not part of the test suite) — seeds a
// synthetic, plate-shaped `.evadb` file at a chosen scale and times
// `ResultExport::start_export` for the plate/well grid and flat-list
// exports, reporting wall time and peak RSS (from `/proc/self/status`'s
// `VmHWM`, Linux-only). Meant to be run against two checkouts of the
// library (before/after a change to the plate/well export path) with the
// same `--scale`/`--classes`/`--columns`/`--aggregations` args, so the
// numbers are comparable.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_plate_well_export -- \
//     [--scale N] [--classes N] [--columns N] [--aggregations N] [--mode view|flat|both]

use duckdb::{Connection, params};
use evanalyzer_app::result::{Aggregation, Column, ExportFormat, ResultExport, ResultsGenerator};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn arg_value(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_str(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Peak resident set size ("high water mark") since process start, in MB.
/// Linux-only (`/proc/self/status`) - returns 0.0 elsewhere/on any read
/// failure, which is fine for a throwaway local benchmark.
fn peak_rss_mb() -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().split_whitespace().next())
        .and_then(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Every column the schema's `objects` table needs a value for, in exact
/// declaration order - mirrors `evanalyzer_core::storage::duckdb`'s real
/// `Appender` insert and `results::test_support`'s test schema, so the
/// plate/well queries under benchmark see the same shape of table a real
/// `.evadb` file has.
const CREATE_TABLES: &str = "
    CREATE TABLE objects (
        image_name           VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL,
        c_stack               INTEGER, z_stack INTEGER, t_stack INTEGER,
        object_id             UUID NOT NULL,
        seg_class_name        VARCHAR, seg_class_id INTEGER,
        object_class_name     VARCHAR, object_class_id VARCHAR,
        parent_id              VARCHAR, children VARCHAR, track_id UBIGINT,
        centroid_x_px DOUBLE, centroid_y_px DOUBLE, centroid_x_nm DOUBLE, centroid_y_nm DOUBLE,
        bbox_xmin_px UINTEGER, bbox_ymin_px UINTEGER, bbox_xmax_px UINTEGER, bbox_ymax_px UINTEGER,
        bbox_xmin_nm DOUBLE, bbox_ymin_nm DOUBLE, bbox_xmax_nm DOUBLE, bbox_ymax_nm DOUBLE,
        area_px UBIGINT, area_nm2 DOUBLE, perimeter_px DOUBLE, perimeter_nm DOUBLE,
        circularity DOUBLE, solidity DOUBLE, aspect_ratio DOUBLE, roundness DOUBLE, compactness DOUBLE,
        major_axis_px DOUBLE, minor_axis_px DOUBLE, major_axis_nm DOUBLE, minor_axis_nm DOUBLE,
        major_axis_angle DOUBLE, eccentricity DOUBLE,
        feret_diameter_px DOUBLE, min_feret_px DOUBLE, feret_diameter_nm DOUBLE, min_feret_nm DOUBLE,
        touches_edge BOOLEAN,
        pixel_size_x_nm DOUBLE, pixel_size_y_nm DOUBLE, pixel_size_z_nm DOUBLE,
        image_bit_depth UTINYINT,
        intensities_json JSON, coloc_json JSON
    );
    CREATE TABLE images (
        image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
        successful BOOLEAN NOT NULL DEFAULT true, error_message VARCHAR,
        disabled BOOLEAN NOT NULL DEFAULT false,
        width UINTEGER NOT NULL, height UINTEGER NOT NULL,
        c_stacks UINTEGER NOT NULL, z_stacks UINTEGER NOT NULL, t_stacks UINTEGER NOT NULL
    );
    CREATE TABLE classes (
        class_id INTEGER NOT NULL PRIMARY KEY, name VARCHAR NOT NULL, color UINTEGER
    );";

/// Seeds an `n_classes`-class, plate-shaped database: an 8x12 (96-well)
/// plate, 4 fields per well (384 images), `objects_per_image_class`
/// objects per (image, class) pair - so total object count is
/// `384 * n_classes * objects_per_image_class`. Bulk-loaded via `Appender`
/// (same mechanism the real per-image exporter in evanalyzer_core uses),
/// not row-by-row `execute`, so seeding a million-row table takes seconds
/// rather than minutes.
fn seed_db(path: &PathBuf, n_classes: usize, objects_per_image_class: usize) -> usize {
    let conn = Connection::open(path).expect("open db");
    conn.execute_batch(CREATE_TABLES).expect("create schema");

    let rows = "ABCDEFGH";
    let cols = 1..=12;
    let fields = 1..=4;

    for (class_id, class_name) in (1..=n_classes).map(|id| (id, format!("Class{id}"))) {
        conn.execute(
            "INSERT INTO classes (class_id, name, color) VALUES (?, ?, ?)",
            params![class_id as i32, class_name, 0u32],
        )
        .expect("insert class");
    }

    let mut object_count = 0usize;
    {
        let mut app = conn.appender("objects").expect("objects appender");
        let mut idx: u64 = 0;
        for row in rows.chars() {
            for col in cols.clone() {
                for field in fields.clone() {
                    let image_name = format!("{row}{col}_{field:02}.tif");
                    let image_rel_path = image_name.clone();
                    for class_id in 1..=n_classes {
                        let object_class_id_json = format!("[{class_id}]");
                        for i in 0..objects_per_image_class {
                            let object_id = format!("{idx:032x}");
                            let object_id =
                                format!("{}-{}-{}-{}-{}", &object_id[0..8], &object_id[8..12], &object_id[12..16], &object_id[16..20], &object_id[20..32]);
                            // Values vary a bit per object so aggregations
                            // (avg/min/max/stddev/median/skewness) aren't
                            // degenerate on constant input.
                            let area_px = 100 + (i % 50) as u64;
                            let perimeter_px = 30.0 + (i % 20) as f64;
                            let circularity = 0.5 + (i % 10) as f64 / 20.0;
                            let solidity = 0.6 + (i % 10) as f64 / 25.0;
                            let eccentricity = 0.1 + (i % 10) as f64 / 15.0;

                            app.append_row(params![
                                &image_name,           // image_name
                                &image_rel_path,       // image_rel_path
                                0i32,                  // c_stack
                                0i32,                  // z_stack
                                0i32,                  // t_stack
                                &object_id,            // object_id
                                "",                    // seg_class_name
                                0i32,                  // seg_class_id
                                "",                    // object_class_name
                                &object_class_id_json, // object_class_id
                                None::<String>,        // parent_id
                                "[]",                  // children
                                0u64,                  // track_id
                                0.0,                   // centroid_x_px
                                0.0,                   // centroid_y_px
                                0.0,                   // centroid_x_nm
                                0.0,                   // centroid_y_nm
                                0u32,                  // bbox_xmin_px
                                0u32,                  // bbox_ymin_px
                                0u32,                  // bbox_xmax_px
                                0u32,                  // bbox_ymax_px
                                0.0,                   // bbox_xmin_nm
                                0.0,                   // bbox_ymin_nm
                                0.0,                   // bbox_xmax_nm
                                0.0,                   // bbox_ymax_nm
                                area_px,               // area_px
                                area_px as f64,        // area_nm2
                                perimeter_px,          // perimeter_px
                                perimeter_px,          // perimeter_nm
                                circularity,           // circularity
                                solidity,              // solidity
                                1.0,                   // aspect_ratio
                                0.5,                   // roundness
                                0.5,                   // compactness
                                0.0,                   // major_axis_px
                                0.0,                   // minor_axis_px
                                0.0,                   // major_axis_nm
                                0.0,                   // minor_axis_nm
                                0.0,                   // major_axis_angle
                                eccentricity,          // eccentricity
                                0.0,                   // feret_diameter_px
                                0.0,                   // min_feret_px
                                0.0,                   // feret_diameter_nm
                                0.0,                   // min_feret_nm
                                false,                 // touches_edge
                                1.0,                   // pixel_size_x_nm
                                1.0,                   // pixel_size_y_nm
                                1.0,                   // pixel_size_z_nm
                                8u8,                   // image_bit_depth
                                "{}",                  // intensities_json
                                "{}",                  // coloc_json
                            ])
                            .expect("append object row");
                            idx += 1;
                            object_count += 1;
                        }
                    }
                }
            }
        }
        // Appender flushes to disk on drop.
    }

    for row in rows.chars() {
        for col in cols.clone() {
            for field in fields.clone() {
                let image_name = format!("{row}{col}_{field:02}.tif");
                conn.execute(
                    "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) VALUES (?, ?, 100, 100, 1, 1, 1)",
                    params![&image_name, &image_name],
                )
                .expect("insert image");
            }
        }
    }

    object_count
}

fn main() {
    let n_classes = arg_value("--classes", 3);
    let n_columns = arg_value("--columns", 5);
    let n_aggregations = arg_value("--aggregations", 7);
    let objects_per_image_class = arg_value("--scale", 900);
    let mode = arg_str("--mode", "both");

    let all_columns = [
        Column::AreaSizePx,
        Column::PerimeterPx,
        Column::Circularity,
        Column::Solidity,
        Column::Eccentricity,
    ];
    let all_aggregations = [
        Aggregation::Avg,
        Aggregation::Min,
        Aggregation::Max,
        Aggregation::Stddev,
        Aggregation::Sum,
        Aggregation::Median,
        Aggregation::Skewness,
    ];
    let columns: Vec<Column> = all_columns.iter().take(n_columns.max(1)).cloned().collect();
    let aggregations: Vec<Aggregation> = all_aggregations
        .iter()
        .take(n_aggregations.max(1))
        .cloned()
        .collect();

    let db_path = std::env::temp_dir().join(format!(
        "evanalyzer_bench_plate_well_{}.evadb",
        std::process::id()
    ));
    let out_root = std::env::temp_dir().join(format!(
        "evanalyzer_bench_plate_well_out_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_root).unwrap();

    println!("== Seeding ==");
    let seed_start = Instant::now();
    let object_count = seed_db(&db_path, n_classes, objects_per_image_class);
    println!(
        "objects: {object_count}   classes: {n_classes}   columns: {}   aggregations: {}   \
         seed time: {:.2}s   rss after seed: {:.1} MB",
        columns.len(),
        aggregations.len(),
        seed_start.elapsed().as_secs_f64(),
        peak_rss_mb(),
    );

    let database = ResultsGenerator::open_database(db_path.clone()).expect("open database");

    let base = ResultExport {
        z_stacks: (0..1).into(),
        t_stacks: (0..1).into(),
        columns: columns.clone(),
        aggregations: aggregations.clone(),
        format: ExportFormat::XLSX,
        ..Default::default()
    };

    let run = |label: &str, mut export: ResultExport| {
        let out_dir = out_root.join(label);
        let _ = std::fs::remove_dir_all(&out_dir);
        std::fs::create_dir_all(&out_dir).unwrap();
        export.output_dir = out_dir.clone();

        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        let result = export.start_export(&database, &cancel, &mut |_msg, _cur, _total| {});
        let elapsed = start.elapsed();

        match result {
            Ok(()) => println!(
                "{label:<30} {:>10.3}s   peak rss: {:>8.1} MB",
                elapsed.as_secs_f64(),
                peak_rss_mb(),
            ),
            Err(err) => println!("{label:<30} FAILED: {err}"),
        }
    };

    println!("\n== Export timings ==");
    if mode == "view" || mode == "both" {
        run(
            "plate_and_well_view",
            ResultExport {
                with_plate_view_heatmap: true,
                with_well_view_heatmap: true,
                ..base.clone()
            },
        );
    }
    if mode == "flat" || mode == "both" {
        run(
            "plate_and_well_flat_list",
            ResultExport {
                with_plate_view_list: true,
                with_well_view_list: true,
                ..base.clone()
            },
        );
    }
    if mode == "list_xlsx" || mode == "list" || mode == "both" {
        run(
            "list_xlsx",
            ResultExport {
                with_list_view: true,
                format: ExportFormat::XLSX,
                ..base.clone()
            },
        );
    }
    if mode == "list_csv" || mode == "list" || mode == "both" {
        run(
            "list_csv",
            ResultExport {
                with_list_view: true,
                format: ExportFormat::CSV,
                ..base.clone()
            },
        );
    }

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&out_root);
}
