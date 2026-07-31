//! Shared `#[cfg(test)]` fixtures for the `results` module's tests -
//! `results_exporter` and `results_loader` both need a real DuckDB file to
//! test against end-to-end, so the fixture lives here once instead of being
//! copy-pasted into both.

use std::path::Path;

/// A row's worth of intensity data for channel 0, in the same JSON shape
/// `evanalyzer_core::storage::duckdb::intensities_to_json` writes - mean
/// 127.0, sum 255.0 (scaled values).
pub(crate) const CH0_INTENSITIES_JSON: &str = r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#;

/// Creates the `objects`/`coloc_stats` schema (matching the schema
/// `evanalyzer_core::storage::duckdb::CREATE_TABLES` writes - that constant
/// isn't public, so this mirrors it by hand) on an already-open connection.
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
        );
        CREATE TABLE coloc_stats (
            image VARCHAR NOT NULL, source_class VARCHAR NOT NULL, target_class VARCHAR NOT NULL,
            n_colocalized UBIGINT, avg_targets_per_object DOUBLE, total_source_objects UBIGINT
        );",
    )
    .expect("create schema");
}

/// Builds a minimal `objects` table and inserts two objects from two different
/// images/classes. Lets tests exercise `ResultsLoader`/`ResultsExporter`
/// end-to-end against a real DuckDB file instead of mocking the reader.
pub(crate) fn seed_results_db(path: &Path) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_schema(&conn);

    let insert = |image: &str, object_id: &str, class_name: &str, class_id: i32, area_px: u64| {
        conn.execute(
            "INSERT INTO objects (
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
        .unwrap_or_else(|e| panic!("insert object {object_id}: {e}"));
    };

    insert(
        "img1.tif",
        "00000000-0000-0000-0000-000000000001",
        "ClassA",
        1,
        100,
    );
    insert(
        "img2.tif",
        "00000000-0000-0000-0000-000000000002",
        "ClassB",
        2,
        200,
    );
}

/// Builds a deterministic, order-decodable `object_id` for the bulk fixtures
/// below: fixed zero prefix, a `group` nibble-block to keep different row
/// families from colliding when seeded into the same file, and `idx` as the
/// trailing 12 hex digits. Because every id in a group has the same width,
/// lexicographic (`ORDER BY object_id`) order matches numeric `idx` order -
/// letting pagination-boundary tests assert exact row order, not just set
/// membership.
fn make_object_id(group: u16, idx: usize) -> String {
    format!("00000000-0000-0000-{group:04x}-{idx:012x}")
}

/// Builds a two-object `objects` table shaped for Matrix (Plate) export
/// tests: image "A1_01.tif" (area 10, folder "A1") and image "B2_01.tif"
/// (area 20, folder "B2"). Both the folder name and the `^([A-Z]\d+)_`
/// prefix of the image name decode to the same well ids ("A1" -> row 0/col
/// 0, "B2" -> row 1/col 1), so the same fixture drives both Folder-mode and
/// Regex-mode grouping tests with the same expected 2x2 grid.
pub(crate) fn seed_plate_results_db(path: &Path) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_schema(&conn);

    let insert = |image: &str, rel_path: &str, object_id: &str, area_px: u64| {
        conn.execute(
            "INSERT INTO objects (
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
                ?, ?, ?, 'ClassA', 1,
                '[\"ClassA\"]', '[1]', 0,
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
                rel_path,
                object_id,
                area_px,
                area_px as f64,
                CH0_INTENSITIES_JSON
            ],
        )
        .unwrap_or_else(|e| panic!("insert object {object_id}: {e}"));
    };

    insert(
        "A1_01.tif",
        "A1/A1_01.tif",
        "00000000-0000-0000-0000-0000000000a1",
        10,
    );
    insert(
        "B2_01.tif",
        "B2/B2_01.tif",
        "00000000-0000-0000-0000-0000000000b2",
        20,
    );
}

/// Inverse of [`make_object_id`] - recovers `idx` from the trailing 12 hex
/// digits, regardless of `group`.
pub(crate) fn decode_object_id_idx(object_id: &str) -> usize {
    usize::from_str_radix(&object_id[object_id.len() - 12..], 16)
        .unwrap_or_else(|e| panic!("decode idx from test object id {object_id:?}: {e}"))
}

const LARGE_ROW_GROUP: u16 = 1;
const LARGE_SOURCE_GROUP: u16 = 2;
const LARGE_PARTNER_GROUP: u16 = 3;

pub(crate) fn large_source_object_id(idx: usize) -> String {
    make_object_id(LARGE_SOURCE_GROUP, idx)
}

pub(crate) fn large_partner_object_id(idx: usize) -> String {
    make_object_id(LARGE_PARTNER_GROUP, idx)
}

/// Inserts `sql`'s accumulated multi-row `INSERT` in bounded-size batches
/// instead of one enormous statement, then clears it for the next batch.
fn flush_batch(conn: &duckdb::Connection, sql: &mut String, prefix: &str) {
    conn.execute_batch(sql).expect("bulk insert batch");
    sql.clear();
    sql.push_str(prefix);
}

const INSERT_PREFIX: &str = "INSERT INTO objects (
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
) VALUES ";

const BATCH_ROWS: usize = 1000;

/// Seeds `n` plain (non-colocalized) objects, cycling across three images and
/// two classes, with a unique, order-decodable `object_id` (see
/// [`make_object_id`]) and `area_px` set to the row's index - so a
/// pagination test can assert the exported `Area` column is exactly
/// `0..n`, in order, with nothing missing or duplicated across an
/// export-page boundary.
pub(crate) fn seed_large_results_db(path: &Path, n: usize) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_schema(&conn);

    let images = ["img1.tif", "img2.tif", "img3.tif"];
    let classes = [("ClassA", 1), ("ClassB", 2)];

    let mut sql = String::from(INSERT_PREFIX);
    for idx in 0..n {
        if !sql.ends_with(' ') {
            sql.push(',');
        }
        let image = images[idx % images.len()];
        let (class_name, class_id) = classes[idx % classes.len()];
        let object_id = make_object_id(LARGE_ROW_GROUP, idx);
        sql.push_str(&format!(
            "('{image}','{image}','{object_id}','{class_name}',{class_id},\
              '[\"{class_name}\"]','[{class_id}]',0,\
              0,0,0,0,\
              0,0,10,10,\
              0,0,0,0,\
              {idx},{idx}.0,40,40,\
              1.0,1.0,1.0,1.0,1.0,\
              10,10,false,\
              1.0,1.0,1.0,\
              '{CH0_INTENSITIES_JSON}','{{}}')"
        ));
        if (idx + 1) % BATCH_ROWS == 0 {
            flush_batch(&conn, &mut sql, INSERT_PREFIX);
        }
    }
    if !sql.ends_with(' ') {
        flush_batch(&conn, &mut sql, INSERT_PREFIX);
    }
}

/// Seeds `n_sources` "ClassA" objects, each colocalized with exactly one
/// "ClassB" partner from a pool of `n_partners` (`partner_idx = idx %
/// n_partners`, so partners are reused once `n_sources > n_partners`), plus
/// the `n_partners` "ClassB" objects themselves. Both families use
/// order-decodable `object_id`s (see [`make_object_id`]) so a
/// coloc-detail pagination test can, for every exported row, decode the
/// source and partner indices straight from the "object ID" / "Coloc ClassB
/// object ID" columns and assert `partner_idx == source_idx % n_partners` -
/// proving both correct row coverage *and* correct partner resolution
/// across an export-page boundary.
pub(crate) fn seed_large_coloc_db(path: &Path, n_sources: usize, n_partners: usize) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_schema(&conn);

    let mut sql = String::from(INSERT_PREFIX);
    let mut row_in_batch = 0usize;
    let push_row =
        |conn: &duckdb::Connection, sql: &mut String, row_in_batch: &mut usize, row: String| {
            if !sql.ends_with(' ') {
                sql.push(',');
            }
            sql.push_str(&row);
            *row_in_batch += 1;
            if *row_in_batch % BATCH_ROWS == 0 {
                flush_batch(conn, sql, INSERT_PREFIX);
            }
        };

    for idx in 0..n_partners {
        let object_id = large_partner_object_id(idx);
        push_row(
            &conn,
            &mut sql,
            &mut row_in_batch,
            format!(
                "('img1.tif','img1.tif','{object_id}','ClassB',2,\
                  '[\"ClassB\"]','[2]',0,\
                  0,0,0,0,\
                  0,0,10,10,\
                  0,0,0,0,\
                  {idx},{idx}.0,40,40,\
                  1.0,1.0,1.0,1.0,1.0,\
                  10,10,false,\
                  1.0,1.0,1.0,\
                  '{CH0_INTENSITIES_JSON}','{{}}')"
            ),
        );
    }
    for idx in 0..n_sources {
        let object_id = large_source_object_id(idx);
        let partner_id = large_partner_object_id(idx % n_partners);
        let coloc_json = format!("{{\"ClassB\":[\"{partner_id}\"]}}");
        push_row(
            &conn,
            &mut sql,
            &mut row_in_batch,
            format!(
                "('img1.tif','img1.tif','{object_id}','ClassA',1,\
                  '[\"ClassA\"]','[1]',0,\
                  0,0,0,0,\
                  0,0,10,10,\
                  0,0,0,0,\
                  {idx},{idx}.0,40,40,\
                  1.0,1.0,1.0,1.0,1.0,\
                  10,10,false,\
                  1.0,1.0,1.0,\
                  '{CH0_INTENSITIES_JSON}','{coloc_json}')"
            ),
        );
    }
    if !sql.ends_with(' ') {
        flush_batch(&conn, &mut sql, INSERT_PREFIX);
    }
}
