//! Test-only scratch-directory helper shared by command test modules.
//!
//! Hand-rolled instead of pulling in the `tempfile` crate (not currently a
//! dependency of this crate) - all a command test needs is "a project file
//! on disk nobody else is using", which `std::env::temp_dir()` plus a unique
//! subdirectory name already gives us.
#![cfg(test)]

use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TempProjectFile {
    dir: PathBuf,
    pub(crate) path: PathBuf,
}

impl TempProjectFile {
    pub(crate) fn new(settings: &ProjectSettings) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "evanalyzer_cli_test_{}_{n}_{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("project.evaproj");
        std::fs::write(&path, serde_json::to_string(settings).unwrap()).unwrap();
        Self { dir, path }
    }
}

impl Drop for TempProjectFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A row's worth of channel-0 intensity data, in the same JSON shape
/// `evanalyzer_core::storage::duckdb::intensities_to_json` writes. Mirrors
/// `evanalyzer_app`'s own `CH0_INTENSITIES_JSON` test fixture
/// (`crates/app/src/results/test_support.rs`) - duplicated here because that
/// helper is `pub(crate)` and `#[cfg(test)]`-only there, so it isn't reachable
/// from this crate.
const CH0_INTENSITIES_JSON: &str = r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#;

fn create_results_schema(conn: &duckdb::Connection) {
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
        CREATE TABLE images (
            image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
            successful BOOLEAN NOT NULL DEFAULT true, error_message VARCHAR,
            disabled BOOLEAN NOT NULL DEFAULT false,
            width UINTEGER NOT NULL, height UINTEGER NOT NULL,
            c_stacks UINTEGER NOT NULL, z_stacks UINTEGER NOT NULL, t_stacks UINTEGER NOT NULL
        );
        CREATE TABLE classes (
            class_id INTEGER NOT NULL PRIMARY KEY, name VARCHAR NOT NULL, color UINTEGER
        );",
    )
    .expect("create schema");
}

/// Builds a minimal `objects` table for `view`/`export` command tests: two
/// objects from two different images/classes, at two different (but
/// contiguous, 0-indexed — real z/t stacks always are)
/// `t_stack`/`z_stack` planes, so a full-range export
/// (`ResultsGenerator::get_nr_of_z_stacks`/`get_nr_of_t_stacks`, both of
/// which return the *max* stack index, not a count) actually has to sweep
/// more than one plane to see both objects, and
/// channel-0 intensities (so `--channels` has something to discover).
pub(crate) fn seed_view_results_db(path: &Path) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_results_schema(&conn);

    let insert = |image: &str,
                  object_id: &str,
                  class_name: &str,
                  class_id: i32,
                  area_px: u64,
                  t_stack: i32,
                  z_stack: i32| {
        conn.execute(
            "INSERT INTO objects (
                image_name, image_rel_path, t_stack, z_stack,
                object_id, seg_class_name, seg_class_id,
                object_class_name, object_class_id, track_id,
                centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
                bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
                bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
                area_px, area_nm2, perimeter_px, perimeter_nm,
                circularity, solidity, aspect_ratio, roundness, compactness,
                major_axis_px, minor_axis_px, eccentricity, touches_edge,
                pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
                intensities_json, coloc_json
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, 0,
                0, 0, 0, 0,
                0, 0, 10, 10,
                0, 0, 0, 0,
                ?, ?, 40, 40,
                1.0, 1.0, 1.0, 1.0, 1.0,
                10, 10, 1.0, false,
                1.0, 1.0, 1.0,
                ?, '{}'
            )",
            duckdb::params![
                image,
                image,
                t_stack,
                z_stack,
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
        0,
        0,
    );
    insert(
        "img2.tif",
        "00000000-0000-0000-0000-000000000002",
        "ClassB",
        2,
        200,
        1,
        1,
    );

    // Mirrors what `DuckDbExporter::finalize_image` would have written for
    // these two images: one measured channel (`CH0_INTENSITIES_JSON`), and
    // 2 z/t stacks each - matching the two distinct (t_stack, z_stack) planes
    // `insert` above used (img1 @ (0,0), img2 @ (1,1)).
    for image in ["img1.tif", "img2.tif"] {
        conn.execute(
            "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
             VALUES (?, ?, 100, 100, 1, 2, 2)",
            duckdb::params![image, image],
        )
        .unwrap_or_else(|e| panic!("insert image {image}: {e}"));
    }

    for (id, name) in [(1, "ClassA"), (2, "ClassB")] {
        conn.execute(
            "INSERT INTO classes (class_id, name, color) VALUES (?, ?, 0)",
            duckdb::params![id, name],
        )
        .unwrap_or_else(|e| panic!("insert class {name}: {e}"));
    }
}

/// Same JSON shape as [`CH0_INTENSITIES_JSON`] but keyed `"0".."n_channels-1"`,
/// for tests asserting the CLI reports one intensity column group per real
/// image channel (`evanalyzer_app`'s `ResultsGenerator::get_nr_of_c_stacks`)
/// rather than a fixed count regardless of how many channels the image
/// actually has.
fn intensities_json_for_channels(n_channels: u32) -> String {
    let entries: Vec<String> = (0..n_channels)
        .map(|ch| {
            format!(
                "\"{ch}\":{{\"sum_raw\":1.0,\"sum_scaled\":255.0,\"mean_raw\":0.5,\"mean_scaled\":127.0,\
                 \"median_raw\":0.5,\"median_scaled\":127.0,\"std_raw\":0.1,\"std_scaled\":25.5,\
                 \"min_raw\":0.0,\"min_scaled\":0.0,\"max_raw\":1.0,\"max_scaled\":255.0}}"
            )
        })
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// Builds a single-image, single-object results DB whose object carries
/// intensity data for `n_channels` channels - unlike [`seed_view_results_db`]
/// (always channel 0 only), for tests that need a specific real channel
/// count. This `images` table (via `create_results_schema`) predates
/// `c_stacks`/`z_stacks`/`t_stacks`, so opening it through
/// `ResultsGenerator::open_database` exercises the same backfill migration a
/// real pre-upgrade `.evadb` would.
pub(crate) fn seed_multi_channel_results_db(path: &Path, n_channels: u32) {
    let conn = duckdb::Connection::open(path).expect("open test db");
    create_results_schema(&conn);

    conn.execute(
        "INSERT INTO objects (
            image_name, image_rel_path, t_stack, z_stack,
            object_id, seg_class_name, seg_class_id,
            object_class_name, object_class_id, track_id,
            centroid_x_px, centroid_y_px, centroid_x_nm, centroid_y_nm,
            bbox_xmin_px, bbox_ymin_px, bbox_xmax_px, bbox_ymax_px,
            bbox_xmin_nm, bbox_ymin_nm, bbox_xmax_nm, bbox_ymax_nm,
            area_px, area_nm2, perimeter_px, perimeter_nm,
            circularity, solidity, aspect_ratio, roundness, compactness,
            major_axis_px, minor_axis_px, eccentricity, touches_edge,
            pixel_size_x_nm, pixel_size_y_nm, pixel_size_z_nm,
            intensities_json, coloc_json
        ) VALUES (
            'img1.tif', 'img1.tif', 0, 0,
            '00000000-0000-0000-0000-000000000001', 'ClassA', 1,
            '[\"ClassA\"]', '[1]', 0,
            0, 0, 0, 0,
            0, 0, 10, 10,
            0, 0, 0, 0,
            100, 100.0, 40, 40,
            1.0, 1.0, 1.0, 1.0, 1.0,
            10, 10, 1.0, false,
            1.0, 1.0, 1.0,
            ?, '{}'
        )",
        duckdb::params![intensities_json_for_channels(n_channels)],
    )
    .expect("insert object");

    conn.execute(
        "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
         VALUES ('img1.tif', 'img1.tif', 100, 100, ?, 1, 1)",
        duckdb::params![n_channels],
    )
    .expect("insert image");

    conn.execute(
        "INSERT INTO classes (class_id, name, color) VALUES (1, 'ClassA', 0)",
        [],
    )
    .expect("insert class");
}

/// A scratch directory holding a seeded results DuckDB file, cleaned up on
/// drop. Mirrors [`TempProjectFile`]'s manual temp-dir approach rather than
/// pulling in `tempfile` for project files, but the `duckdb` dev-dependency
/// this needs is already pulled in for `evanalyzer_app`-style DB fixtures, so
/// `tempfile` is used here for brevity.
pub(crate) struct TempResultsDb {
    _dir: tempfile::TempDir,
    pub(crate) path: PathBuf,
}

impl TempResultsDb {
    /// Creates a fresh temp directory, seeds `results.evadb` in it via
    /// [`seed_view_results_db`], and returns the handle (keep it alive for as
    /// long as the path is needed - dropping it deletes the directory).
    pub(crate) fn seeded() -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("results.evadb");
        seed_view_results_db(&path);
        Self { _dir: dir, path }
    }

    /// Same as [`Self::seeded`], but the one seeded object carries intensity
    /// data for `n_channels` channels (see [`seed_multi_channel_results_db`]).
    pub(crate) fn with_channels(n_channels: u32) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("results.evadb");
        seed_multi_channel_results_db(&path, n_channels);
        Self { _dir: dir, path }
    }
}
