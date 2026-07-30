use crate::args::{
    ChartKind, ColorByKind, ExportArgs, ExportCommand, HeatmapArgs, HistogramArgs, ScatterArgs,
    TableExportArgs,
};
use crate::commands::common::{build_database_filter, build_group_config, discover_columns};
use evanalyzer_app::result::{
    ColorBy, HeatmapColorScheme, HeatmapMetric, HeatmapRange, ResultsExporter, ResultsLoader,
    compute_heatmap, compute_histogram, compute_scatter, save_heatmap_png, save_histogram_png,
    save_scatter_png,
};
use evanalyzer_cfg::core_types::InternalErrors;
use std::sync::Arc;

pub fn run(args: ExportArgs) -> Result<(), InternalErrors> {
    match args.command {
        ExportCommand::Csv(table) => export_table(table, true),
        ExportCommand::Xlsx(table) => export_table(table, false),
        ExportCommand::Chart(chart) => match chart.kind {
            ChartKind::Histogram(a) => export_histogram(a),
            ChartKind::Scatter(a) => export_scatter(a),
            ChartKind::Heatmap(a) => export_heatmap(a),
        },
    }
}

fn export_table(args: TableExportArgs, csv: bool) -> Result<(), InternalErrors> {
    let loader = Arc::new(ResultsLoader::new(&args.db));
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let group = build_group_config(&args.group);

    let exporter = ResultsExporter::new(loader);
    if csv {
        exporter.export_to_csv(filter, &group, &specs, &args.out)?;
    } else {
        exporter.export_to_xlsx(filter, &group, &specs, &args.out)?;
    }
    println!("Exported to {}", args.out.display());
    Ok(())
}

fn export_histogram(args: HistogramArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let objects = loader.get_objects(filter)?;

    let data = compute_histogram(
        &objects,
        &args.column,
        &specs,
        args.buckets,
        args.log_scale,
        to_color_by(args.color_by),
    )
    .ok_or_else(|| column_not_found_error(&args.column, &args.db))?;
    save_histogram_png(&data, args.width, args.height, &args.out)?;
    println!("Saved histogram to {}", args.out.display());
    Ok(())
}

fn export_scatter(args: ScatterArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let objects = loader.get_objects(filter)?;

    let data =
        compute_scatter(&objects, &args.x, &args.y, to_color_by(args.color_by), &specs, args.max_points)
        .ok_or_else(|| column_not_found_error(&format!("{} / {}", args.x, args.y), &args.db))?;
    save_scatter_png(&data, args.width, args.height, &args.out)?;
    println!("Saved scatter plot to {}", args.out.display());
    Ok(())
}

fn export_heatmap(args: HeatmapArgs) -> Result<(), InternalErrors> {
    let loader = ResultsLoader::new(&args.db);
    let specs = discover_columns(&loader)?;
    let filter = build_database_filter(&args.filter, true, 0, 0);
    let objects = loader.get_objects(filter)?;

    let metric = if args.metric == "count" {
        HeatmapMetric::Count
    } else {
        HeatmapMetric::Average(args.metric.clone())
    };

    let data = compute_heatmap(&objects, &metric, &specs, args.cell_size)
        .ok_or_else(|| column_not_found_error(&args.metric, &args.db))?;
    let scheme = HeatmapColorScheme::from_label(&args.color_scheme);
    // `requires` on the arg definitions guarantees these are either both
    // present or both absent.
    let range = match (args.range_min, args.range_max) {
        (Some(min), Some(max)) => HeatmapRange::Manual { min, max },
        _ => HeatmapRange::Auto,
    };
    save_heatmap_png(&data, scheme, range, args.width, args.height, &args.out)?;
    println!("Saved heatmap to {}", args.out.display());
    Ok(())
}

fn to_color_by(kind: ColorByKind) -> ColorBy {
    match kind {
        ColorByKind::None => ColorBy::None,
        ColorByKind::Class => ColorBy::Class,
        ColorByKind::Colocalized => ColorBy::Colocalized,
    }
}

fn column_not_found_error(column: &str, db: &std::path::Path) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "No data for column '{column}': check it exists and matches the active filter \
         (see `evanalyzer cli columns --db {}`)",
        db.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{FilterArgs, GroupArgs};
    use std::path::PathBuf;

    // -----------------------------------------------------------------
    // Test-database fixture
    // -----------------------------------------------------------------
    //
    // `evanalyzer_app` has an equivalent seeding helper for its own
    // `ResultsLoader`/`ResultsExporter` tests
    // (`crates/app/src/results/test_support.rs`), but it's `pub(crate)` and
    // `#[cfg(test)]`-only there, so it isn't reachable from this crate. This
    // mirrors just enough of that schema (matching the column list
    // `evanalyzer_core::storage::duckdb`'s `get_objects` query selects) to
    // exercise the `export_*` commands end to end against a real DuckDB file.

    /// A row's worth of intensity data for channel 0, in the same JSON shape
    /// `evanalyzer_core::storage::duckdb::intensities_to_json` writes.
    const CH0_INTENSITIES_JSON: &str = r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#;

    fn create_schema(conn: &duckdb::Connection) {
        conn.execute_batch(
            "CREATE TABLE objects (
                image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL,
                c_stack INTEGER, z_stack INTEGER, t_stack INTEGER,
                object_id UUID NOT NULL,
                seg_class_name VARCHAR, seg_class_id INTEGER,
                object_class_name VARCHAR, object_class_id VARCHAR,
                parent_id VARCHAR, children VARCHAR, track_id UBIGINT,
                centroid_x_px DOUBLE, centroid_y_px DOUBLE,
                centroid_x_nm DOUBLE, centroid_y_nm DOUBLE,
                bbox_xmin_px UINTEGER, bbox_ymin_px UINTEGER,
                bbox_xmax_px UINTEGER, bbox_ymax_px UINTEGER,
                bbox_xmin_nm DOUBLE, bbox_ymin_nm DOUBLE,
                bbox_xmax_nm DOUBLE, bbox_ymax_nm DOUBLE,
                area_px UBIGINT, area_nm2 DOUBLE,
                perimeter_px DOUBLE, perimeter_nm DOUBLE,
                circularity DOUBLE, solidity DOUBLE, aspect_ratio DOUBLE,
                roundness DOUBLE, compactness DOUBLE,
                major_axis_px DOUBLE, minor_axis_px DOUBLE,
                major_axis_nm DOUBLE, minor_axis_nm DOUBLE,
                major_axis_angle DOUBLE, eccentricity DOUBLE,
                feret_diameter_px DOUBLE, min_feret_px DOUBLE,
                feret_diameter_nm DOUBLE, min_feret_nm DOUBLE,
                touches_edge BOOLEAN,
                pixel_size_x_nm DOUBLE, pixel_size_y_nm DOUBLE, pixel_size_z_nm DOUBLE,
                image_bit_depth UTINYINT,
                intensities_json JSON, coloc_json JSON
            )",
        )
        .expect("create schema");
    }

    /// One test object: image/class/area/circularity/centroid, chosen so
    /// histogram/scatter/heatmap each have distinct, real values to plot.
    struct Row {
        image: &'static str,
        object_id: &'static str,
        class_name: &'static str,
        class_id: i32,
        area_px: u64,
        circularity: f64,
        centroid_x_px: f64,
        centroid_y_px: f64,
    }

    fn insert(conn: &duckdb::Connection, row: &Row) {
        conn.execute(
            "INSERT INTO objects (
                image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                object_class_name, object_class_id, track_id,
                centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                area_px, area_nm2, perimeter_px, perimeter_nm,
                circularity, solidity, aspect_ratio, roundness, compactness,
                major_axis_px, minor_axis_px, touches_edge,
                pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                intensities_json, coloc_json
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, 0,
                ?, ?, 0, 0,
                0, 0, 10, 10,
                ?, ?, 40, 40,
                ?, 1.0, 1.0, 1.0, 1.0,
                10, 10, false,
                1.0, 1.0, 1.0,
                ?, '{}'
            )",
            duckdb::params![
                row.image,
                row.image,
                row.object_id,
                row.class_name,
                row.class_id,
                format!("[\"{}\"]", row.class_name),
                format!("[{}]", row.class_id),
                row.centroid_x_px,
                row.centroid_y_px,
                row.area_px,
                row.area_px as f64,
                row.circularity,
                CH0_INTENSITIES_JSON,
            ],
        )
        .unwrap_or_else(|e| panic!("insert object {}: {e}", row.object_id));
    }

    /// A temp-dir-backed DuckDB results file seeded with four objects across
    /// two images/classes, with distinct `area_px`/`circularity`/centroid
    /// values so histogram/scatter/heatmap all have something real to plot.
    struct TestDb {
        _dir: tempfile::TempDir,
        path: PathBuf,
    }

    impl TestDb {
        fn seeded() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("results.evadb");
            let conn = duckdb::Connection::open(&path).expect("open test db");
            create_schema(&conn);
            insert(
                &conn,
                &Row {
                    image: "img1.tif",
                    object_id: "00000000-0000-0000-0000-000000000001",
                    class_name: "ClassA",
                    class_id: 1,
                    area_px: 100,
                    circularity: 0.9,
                    centroid_x_px: 10.0,
                    centroid_y_px: 10.0,
                },
            );
            insert(
                &conn,
                &Row {
                    image: "img1.tif",
                    object_id: "00000000-0000-0000-0000-000000000002",
                    class_name: "ClassA",
                    class_id: 1,
                    area_px: 200,
                    circularity: 0.7,
                    centroid_x_px: 20.0,
                    centroid_y_px: 20.0,
                },
            );
            insert(
                &conn,
                &Row {
                    image: "img2.tif",
                    object_id: "00000000-0000-0000-0000-000000000003",
                    class_name: "ClassB",
                    class_id: 2,
                    area_px: 300,
                    circularity: 0.5,
                    centroid_x_px: 500.0,
                    centroid_y_px: 500.0,
                },
            );
            insert(
                &conn,
                &Row {
                    image: "img2.tif",
                    object_id: "00000000-0000-0000-0000-000000000004",
                    class_name: "ClassB",
                    class_id: 2,
                    area_px: 400,
                    circularity: 0.3,
                    centroid_x_px: 520.0,
                    centroid_y_px: 520.0,
                },
            );
            drop(conn);
            Self { _dir: dir, path }
        }
    }

    fn assert_png(path: &std::path::Path) {
        let bytes = std::fs::read(path).expect("read png back");
        assert!(!bytes.is_empty(), "png file is empty");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG file");
    }

    // -----------------------------------------------------------------
    // export_table
    // -----------------------------------------------------------------

    #[test]
    fn export_table_writes_a_csv_file_with_the_expected_rows() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.csv");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            },
            true,
        )
        .expect("csv export should succeed");

        let content = std::fs::read_to_string(&out).expect("read csv back");
        let mut lines = content.lines();
        let header = lines.next().expect("header row present");
        assert!(header.contains("object ID"), "header: {header}");
        assert!(header.contains("Area"), "header: {header}");
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), 4, "expected 4 data rows, got: {body:?}");
        assert!(content.contains("ClassA"));
        assert!(content.contains("ClassB"));
        assert!(content.contains("100"));
        assert!(content.contains("400"));
    }

    #[test]
    fn export_table_writes_an_xlsx_file_with_the_expected_rows() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("out.xlsx");

        export_table(
            TableExportArgs {
                db: db.path.clone(),
                out: out.clone(),
                filter: FilterArgs::default(),
                group: GroupArgs::default(),
            },
            false,
        )
        .expect("xlsx export should succeed");

        let bytes = std::fs::read(&out).expect("read xlsx back");
        // XLSX files are zip archives - "PK\x03\x04" is the local-file-header magic.
        assert!(bytes.len() > 4, "xlsx file is too small: {} bytes", bytes.len());
        assert_eq!(&bytes[..4], b"PK\x03\x04", "not a zip/xlsx file");
    }

    // -----------------------------------------------------------------
    // export_histogram
    // -----------------------------------------------------------------

    #[test]
    fn export_histogram_saves_a_png_for_an_existing_column() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("hist.png");

        export_histogram(HistogramArgs {
            db: db.path.clone(),
            out: out.clone(),
            column: "area_px".into(),
            buckets: 4,
            log_scale: false,
            color_by: ColorByKind::None,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        })
        .expect("histogram export should succeed");

        assert_png(&out);
    }

    #[test]
    fn export_histogram_errors_for_a_nonexistent_column() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("hist.png");

        let result = export_histogram(HistogramArgs {
            db: db.path.clone(),
            out,
            column: "does_not_exist".into(),
            buckets: 4,
            log_scale: false,
            color_by: ColorByKind::None,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        });

        assert!(result.is_err(), "expected an unknown column to be rejected");
    }

    // -----------------------------------------------------------------
    // export_scatter
    // -----------------------------------------------------------------

    #[test]
    fn export_scatter_saves_a_png_for_existing_columns() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("scatter.png");

        export_scatter(ScatterArgs {
            db: db.path.clone(),
            out: out.clone(),
            x: "area_px".into(),
            y: "circularity".into(),
            color_by: ColorByKind::Class,
            max_points: 1000,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        })
        .expect("scatter export should succeed");

        assert_png(&out);
    }

    #[test]
    fn export_scatter_errors_for_a_nonexistent_column() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("scatter.png");

        let result = export_scatter(ScatterArgs {
            db: db.path.clone(),
            out,
            x: "does_not_exist".into(),
            y: "area_px".into(),
            color_by: ColorByKind::None,
            max_points: 1000,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        });

        assert!(result.is_err(), "expected an unknown column to be rejected");
    }

    // -----------------------------------------------------------------
    // export_heatmap
    // -----------------------------------------------------------------

    #[test]
    fn export_heatmap_saves_a_png_for_an_existing_metric() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("heatmap.png");

        export_heatmap(HeatmapArgs {
            db: db.path.clone(),
            out: out.clone(),
            metric: "area_px".into(),
            cell_size: 256.0,
            color_scheme: "viridis".into(),
            range_min: None,
            range_max: None,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        })
        .expect("heatmap export should succeed");

        assert_png(&out);
    }

    #[test]
    fn export_heatmap_saves_a_png_for_the_count_metric() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("heatmap_count.png");

        export_heatmap(HeatmapArgs {
            db: db.path.clone(),
            out: out.clone(),
            metric: "count".into(),
            cell_size: 256.0,
            color_scheme: "viridis".into(),
            range_min: None,
            range_max: None,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        })
        .expect("heatmap export should succeed");

        assert_png(&out);
    }

    #[test]
    fn export_heatmap_errors_for_a_nonexistent_metric() {
        let db = TestDb::seeded();
        let out_dir = tempfile::tempdir().expect("tempdir");
        let out = out_dir.path().join("heatmap.png");

        let result = export_heatmap(HeatmapArgs {
            db: db.path.clone(),
            out,
            metric: "does_not_exist".into(),
            cell_size: 256.0,
            color_scheme: "viridis".into(),
            range_min: None,
            range_max: None,
            width: 400,
            height: 300,
            filter: FilterArgs::default(),
        });

        assert!(result.is_err(), "expected an unknown metric to be rejected");
    }

    #[test]
    fn to_color_by_maps_every_kind_to_its_matching_variant() {
        assert!(matches!(to_color_by(ColorByKind::None), ColorBy::None));
        assert!(matches!(to_color_by(ColorByKind::Class), ColorBy::Class));
        assert!(matches!(
            to_color_by(ColorByKind::Colocalized),
            ColorBy::Colocalized
        ));
    }

    #[test]
    fn column_not_found_error_names_the_column_and_db_path_in_the_message() {
        let err = column_not_found_error("area", std::path::Path::new("/tmp/results.evadb"));

        let InternalErrors::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("area"));
        assert!(msg.contains("/tmp/results.evadb"));
    }
}
