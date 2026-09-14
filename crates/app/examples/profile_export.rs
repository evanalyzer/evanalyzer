// Throwaway profiling harness (not part of the test suite) — times each
// `ResultExport::start_export` shape against a real, production-scale
// `.evadb` file, to find out which export flavor is actually slow.
//
// Usage: cargo run --release -p evanalyzer_app --example profile_export -- <path/to/db.evadb>

use evanalyzer_app::result::{Column, ExportFormat, ResultExport, ResultsGenerator};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn main() {
    let db_path = std::env::args()
        .nth(1)
        .expect("usage: profile_export <path/to/db.evadb>");
    let db_path = PathBuf::from(db_path);

    let database = ResultsGenerator::open_database(db_path.clone()).expect("open database");

    let images = database.get_images().expect("get_images");
    let nr_images = images.len();
    let nr_c_stacks = database.get_nr_of_c_stacks();
    println!("== Database ==");
    println!("path: {}", db_path.display());
    println!("images: {nr_images}");
    println!("c_stacks (channels): {nr_c_stacks}");

    let columns: Vec<Column> = std::iter::once(Column::AreaSizePx)
        .chain((0..nr_c_stacks).map(Column::IntensityAvg))
        .collect();

    let out_root = std::env::temp_dir().join(format!(
        "evanalyzer_profile_export_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_root).unwrap();

    let base = ResultExport {
        z_stacks: (0..1).into(),
        t_stacks: (0..1).into(),
        columns: columns.clone(),
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

        let size: u64 = std::fs::read_dir(&out_dir)
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
                "{label:<45} {:>10.3}s   {nr_files:>5} files   {:>10.2} MB",
                elapsed.as_secs_f64(),
                size as f64 / 1_000_000.0
            ),
            Err(err) => println!("{label:<45} FAILED: {err}"),
        }
    };

    println!("\n== Export timings ==");

    run(
        "list_xlsx_single_file",
        ResultExport {
            with_list_view: true,
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "list_csv_single_file",
        ResultExport {
            with_list_view: true,
            format: ExportFormat::CSV,
            ..base.clone()
        },
    );

    run(
        "list_xlsx_one_file_per_image",
        ResultExport {
            with_list_view: true,
            with_list_one_file_per_image: true,
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "list_csv_one_file_per_image",
        ResultExport {
            with_list_view: true,
            with_list_one_file_per_image: true,
            format: ExportFormat::CSV,
            ..base.clone()
        },
    );

    run(
        "grouped_by_image_xlsx",
        ResultExport {
            with_grouped_by_image_list: true,
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "plate_and_well_view_xlsx",
        ResultExport {
            with_plate_view: true,
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "plate_and_well_flat_list_xlsx",
        ResultExport {
            with_plates_and_wells_as_list: true,
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "image_heatmap_xlsx",
        ResultExport {
            with_heatmap: true,
            square_size: Some(256),
            format: ExportFormat::XLSX,
            ..base.clone()
        },
    );

    run(
        "parquet_dump",
        ResultExport {
            format: ExportFormat::Parquet,
            ..base.clone()
        },
    );

    println!("\noutput left under {}", out_root.display());
}
