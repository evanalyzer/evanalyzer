// Throwaway benchmark: N images with a modest object count each (matching
// the "1000 images, ~1GB total" scenario reported against
// `with_list_one_file_per_image`), to find out where the ~1s/image cost
// actually goes - data volume per file is tiny, so the suspicion is fixed
// per-image query/file overhead, not row-writing cost.
//
// Usage: cargo run --release -p evanalyzer_app --example bench_one_file_per_image -- \
//     [--images N] [--objects-per-image N] [--mode xlsx|csv]

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

fn seed_db(
    path: &PathBuf,
    n_images: usize,
    objects_per_image: usize,
    background_images: usize,
    background_objects_per_image: usize,
) {
    let conn = Connection::open(path).expect("open db");
    conn.execute_batch(CREATE_TABLES).expect("create schema");
    conn.execute(
        "INSERT INTO classes (class_id, name, color) VALUES (1, 'ClassA', 0)",
        [],
    )
    .unwrap();

    let total_images = n_images + background_images;
    {
        let mut app = conn.appender("images").expect("images appender");
        for img in 0..total_images {
            let name = image_name(img, n_images);
            app.append_row(params![
                &name,
                &name,
                true,
                None::<String>,
                false,
                100u32,
                100u32,
                1u32,
                1u32,
                1u32
            ])
            .unwrap();
        }
    }

    let mut app = conn.appender("objects").expect("objects appender");
    let mut idx: u64 = 0;
    // Background (unrelated, not exported) images seeded FIRST, so the
    // export-target images aren't the very first rows in the table -
    // closer to a real project where the export covers a subset of a much
    // larger, already-populated database.
    for img in n_images..total_images {
        let name = image_name(img, n_images);
        for i in 0..background_objects_per_image {
            seed_object(&mut app, &name, idx, i);
            idx += 1;
        }
    }
    for img in 0..n_images {
        let name = image_name(img, n_images);
        for i in 0..objects_per_image {
            seed_object(&mut app, &name, idx, i);
            idx += 1;
        }
    }
}

/// `img_00000.tif`.. for the `n_images` export-target images, `bg_00000.tif`..
/// for anything past that (background/unrelated images) - so the two are
/// trivially distinguishable if needed, though the benchmark only ever
/// exports the `img_*` ones.
fn image_name(img: usize, n_images: usize) -> String {
    if img < n_images {
        format!("img_{img:05}.tif")
    } else {
        format!("bg_{:05}.tif", img - n_images)
    }
}

fn seed_object(app: &mut duckdb::Appender, name: &str, idx: u64, i: usize) {
    let object_id = format!("{idx:032x}");
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
        name,
        name,
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

fn main() {
    let n_images = arg_value("--images", 1000);
    let objects_per_image = arg_value("--objects-per-image", 250);
    let background_images = arg_value("--background-images", 0);
    let background_objects_per_image = arg_value("--background-objects-per-image", 250);
    let mode = arg_str("--mode", "xlsx");

    let db_path = std::env::temp_dir().join(format!(
        "evanalyzer_bench_ofpi_{}.evadb",
        std::process::id()
    ));
    let out_dir = std::env::temp_dir().join(format!(
        "evanalyzer_bench_ofpi_out_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("== Seeding ==");
    let start = Instant::now();
    seed_db(
        &db_path,
        n_images,
        objects_per_image,
        background_images,
        background_objects_per_image,
    );
    println!(
        "images: {n_images}   objects/image: {objects_per_image}   background images: {background_images}   \
         total objects: {}   seed time: {:.2}s",
        n_images * objects_per_image + background_images * background_objects_per_image,
        start.elapsed().as_secs_f64(),
    );

    let database = ResultsGenerator::open_database(db_path.clone()).expect("open database");
    let columns = vec![
        Column::AreaSizePx,
        Column::PerimeterPx,
        Column::Circularity,
        Column::Solidity,
        Column::Eccentricity,
    ];
    let format = if mode == "csv" {
        ExportFormat::CSV
    } else {
        ExportFormat::XLSX
    };
    // Export only the `img_*` target images - the `bg_*` ones exist purely
    // to make the underlying `objects` table bigger than what's exported,
    // matching a real project (a subset export against a much larger
    // database).
    let image_rel_paths: Vec<String> = (0..n_images).map(|img| image_name(img, n_images)).collect();

    let export = ResultExport {
        output_dir: out_dir.clone(),
        format,
        columns,
        image_rel_paths,
        with_list_view: true,
        with_list_one_file_per_image: true,
        z_stacks: (0..1).into(),
        t_stacks: (0..1).into(),
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let mut last_report = Instant::now();
    let start = Instant::now();
    let result = export.start_export(&database, &cancel, &mut |msg, cur, total| {
        if last_report.elapsed().as_secs_f64() > 2.0 {
            println!("  {cur}/{total}: {msg}");
            last_report = Instant::now();
        }
    });
    let elapsed = start.elapsed();
    let total_bytes: u64 = std::fs::read_dir(&out_dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    let nr_files = std::fs::read_dir(&out_dir).map(|it| it.count()).unwrap_or(0);

    match result {
        Ok(()) => println!(
            "\n{mode:<6} {:>10.3}s total   {:.4}s/image   {nr_files} files   {:.1} MB   peak rss: {:.1} MB",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / n_images as f64,
            total_bytes as f64 / 1_000_000.0,
            peak_rss_mb(),
        ),
        Err(err) => println!("FAILED: {err}"),
    }

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&out_dir);
}
