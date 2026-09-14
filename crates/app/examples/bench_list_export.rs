// Throwaway benchmark: one image with a large object count, to check
// List-XLSX/CSV export time and peak RSS when a single image has "millions
// of rows" - the scenario `stream_list_pages`'s per-page fetch bounds on
// the query side, but the XLSX *writer* side needed its own fix
// (`add_worksheet_with_constant_memory`) to stay bounded too.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_list_export -- [--rows N] [--mode xlsx|csv|both]

use duckdb::{Connection, params};
use evanalyzer_app::result::{Column, ExportFormat, ResultExport, ResultsGenerator};
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

/// Seeds ONE image with `rows` objects, all in class 1.
fn seed_db(path: &PathBuf, rows: usize) {
    let conn = Connection::open(path).expect("open db");
    conn.execute_batch(CREATE_TABLES).expect("create schema");
    conn.execute(
        "INSERT INTO classes (class_id, name, color) VALUES (1, 'ClassA', 0)",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) VALUES ('big.tif', 'big.tif', 100, 100, 1, 1, 1)",
        [],
    )
    .unwrap();

    let mut app = conn.appender("objects").expect("objects appender");
    for i in 0..rows {
        let object_id = format!("{i:032x}");
        let object_id = format!(
            "{}-{}-{}-{}-{}",
            &object_id[0..8],
            &object_id[8..12],
            &object_id[12..16],
            &object_id[16..20],
            &object_id[20..32]
        );
        let area_px = 100 + (i % 50) as u64;
        let perimeter_px = 30.0 + (i % 20) as f64;
        let circularity = 0.5 + (i % 10) as f64 / 20.0;
        let solidity = 0.6 + (i % 10) as f64 / 25.0;
        let eccentricity = 0.1 + (i % 10) as f64 / 15.0;

        app.append_row(params![
            "big.tif",
            "big.tif",
            0i32,
            0i32,
            0i32,
            &object_id,
            "",
            0i32,
            "",
            "[1]",
            None::<String>,
            "[]",
            0u64,
            0.0,
            0.0,
            0.0,
            0.0,
            0u32,
            0u32,
            0u32,
            0u32,
            0.0,
            0.0,
            0.0,
            0.0,
            area_px,
            area_px as f64,
            perimeter_px,
            perimeter_px,
            circularity,
            solidity,
            1.0,
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            eccentricity,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            1.0,
            1.0,
            1.0,
            8u8,
            "{}",
            "{}",
        ])
        .expect("append row");
    }
}

fn main() {
    let rows = arg_value("--rows", 500_000);
    let mode = arg_str("--mode", "both");

    let db_path = std::env::temp_dir().join(format!(
        "evanalyzer_bench_list_{}.evadb",
        std::process::id()
    ));
    let out_root = std::env::temp_dir().join(format!(
        "evanalyzer_bench_list_out_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_root).unwrap();

    println!("== Seeding ==");
    let start = Instant::now();
    seed_db(&db_path, rows);
    println!(
        "rows: {rows}   seed time: {:.2}s   rss after seed: {:.1} MB",
        start.elapsed().as_secs_f64(),
        peak_rss_mb(),
    );

    let database = ResultsGenerator::open_database(db_path.clone()).expect("open database");
    let columns = vec![
        Column::AreaSizePx,
        Column::PerimeterPx,
        Column::Circularity,
        Column::Solidity,
        Column::Eccentricity,
    ];

    let run = |label: &str, format: ExportFormat| {
        let out_dir = out_root.join(label);
        let _ = std::fs::remove_dir_all(&out_dir);
        std::fs::create_dir_all(&out_dir).unwrap();

        let export = ResultExport {
            output_dir: out_dir,
            format,
            columns: columns.clone(),
            with_list_view: true,
            z_stacks: (0..1).into(),
            t_stacks: (0..1).into(),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        let result = export.start_export(&database, &cancel, &mut |_msg, _cur, _total| {});
        let elapsed = start.elapsed();
        match result {
            Ok(()) => println!(
                "{label:<12} {:>10.3}s   peak rss: {:>8.1} MB",
                elapsed.as_secs_f64(),
                peak_rss_mb(),
            ),
            Err(err) => println!("{label:<12} FAILED: {err}"),
        }
    };

    println!("\n== Export timings ==");
    if mode == "xlsx" || mode == "both" {
        run("list_xlsx", ExportFormat::XLSX);
    }
    if mode == "csv" || mode == "both" {
        run("list_csv", ExportFormat::CSV);
    }

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&out_root);
}
