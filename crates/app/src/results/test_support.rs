//! Shared `#[cfg(test)]` fixtures for the `results` module's tests -
//! `results_exporter` and `results_loader` both need a real DuckDB file to
//! test against end-to-end, so the fixture lives here once instead of being
//! copy-pasted into both.

use std::path::Path;

/// A row's worth of intensity data for channel 0, in the same JSON shape
/// `evanalyzer_core::storage::duckdb::intensities_to_json` writes - mean
/// 127.0, sum 255.0 (scaled values).
pub(crate) const CH0_INTENSITIES_JSON: &str = r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#;

/// Builds a minimal `rois` table directly via SQL (matching the schema
/// `evanalyzer_core::storage::duckdb::CREATE_TABLES` writes - that constant
/// isn't public, so this mirrors it by hand) and inserts two ROIs from two
/// different images/classes. Lets tests exercise `ResultsLoader`/
/// `ResultsExporter` end-to-end against a real DuckDB file instead of
/// mocking the reader.
pub(crate) fn seed_results_db(path: &Path) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    conn.execute_batch(
        "CREATE TABLE rois (
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
        );
        CREATE TABLE coloc_stats (
            image VARCHAR NOT NULL, source_class VARCHAR NOT NULL, target_class VARCHAR NOT NULL,
            n_colocalized UBIGINT, avg_targets_per_roi DOUBLE, total_source_rois UBIGINT
        );",
    )
    .expect("create schema");

    let insert = |image: &str, object_id: &str, class_name: &str, class_id: i32, area_px: u64| {
        conn.execute(
            "INSERT INTO rois (
                image_name, image_rel_path, object_id, seg_class_name, seg_class_id,
                object_class_name, object_class_id, track_id,
                centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                area_px, area_nm2, perimeter_px, perimeter_nm,
                circularity, solidity, aspect_ratio, roundness, compactness,
                major_axis_px, minor_axis_px, touches_edge,
                pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                intensities_json, coloc_json
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, 0,
                0, 0, 0, 0,
                0, 0, 10, 10,
                0, 0, 0, 0,
                ?, ?, 40, 40,
                1.0, 1.0, 1.0, 1.0, 1.0,
                10, 10, false,
                1.0, 1.0, 1.0,
                ?, '{}'
            )",
            duckdb::params![
                image,
                image,
                object_id,
                class_name,
                class_id,
                format!("[\"{class_name}\"]"),
                format!("[{class_id}]"),
                area_px,
                area_px as f64,
                CH0_INTENSITIES_JSON,
            ],
        )
        .unwrap_or_else(|e| panic!("insert roi {object_id}: {e}"));
    };

    insert("img1.tif", "00000000-0000-0000-0000-000000000001", "ClassA", 1, 100);
    insert("img2.tif", "00000000-0000-0000-0000-000000000002", "ClassB", 2, 200);
}
