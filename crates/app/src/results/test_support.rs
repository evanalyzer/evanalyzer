//! Shared test-only fixtures for `results_generator`/`results_exporter`/
//! `results_charts` tests — a schema kept in sync with evanalyzer_core's
//! `duckdb.rs` `CREATE_TABLES`, plus a small builder for objects carrying
//! just the fields a given test actually varies, defaulting everything else
//! to an inert value. Centralized here (rather than each test module
//! hand-rolling its own INSERT, as `crates/cli/src/commands/test_support.rs`
//! and `results_exporter.rs`'s own earlier fixture both did) so the three
//! sibling `results_*` test suites can't quietly drift onto different
//! schemas.
#![cfg(test)]

use duckdb::Connection;
use std::path::Path;

/// A row's worth of single-channel (channel 0) intensity data, in the same
/// JSON shape `evanalyzer_core::storage::duckdb::intensities_to_json` writes.
pub(super) const CH0_INTENSITIES_JSON: &str = r#"{"0":{"sum_raw":1.0,"sum_scaled":255.0,"mean_raw":0.5,"mean_scaled":127.0,"median_raw":0.5,"median_scaled":127.0,"std_raw":0.1,"std_scaled":25.5,"min_raw":0.0,"min_scaled":0.0,"max_raw":1.0,"max_scaled":255.0}}"#;

fn create_schema(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE objects (
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
        );",
    )
    .expect("create test schema");
}

/// One object to seed via [`seed_db`]. Every field defaults to an inert
/// value (see [`ObjectSpec::new`]), so a test only ever sets what it's
/// actually varying — `area_px` (mapped to `Column::AreaSizePx`, plain
/// UBIGINT SQL column `area_px`) is the deliberate go-to numeric measurement
/// for chart/aggregate tests, since it needs no JSON parsing to control or
/// verify.
pub(super) struct ObjectSpec {
    pub(super) image: &'static str,
    pub(super) class_name: &'static str,
    pub(super) class_id: i32,
    pub(super) area_px: u64,
    pub(super) z_stack: i32,
    pub(super) t_stack: i32,
    pub(super) intensities_json: String,
    pub(super) coloc_json: String,
    pub(super) centroid_x_px: f64,
    pub(super) centroid_y_px: f64,
    /// Real pixel size of `image` in the `images` table — every object
    /// belonging to the same `image` must agree (the fixture uses whichever
    /// spec for that image it seeds last), so only set this on a test whose
    /// image heatmap needs a specific grid size.
    pub(super) image_width: u32,
    pub(super) image_height: u32,
}

impl ObjectSpec {
    pub(super) fn new(image: &'static str, class_name: &'static str, class_id: i32, area_px: u64) -> Self {
        Self {
            image,
            class_name,
            class_id,
            area_px,
            z_stack: 0,
            t_stack: 0,
            intensities_json: "{}".to_string(),
            coloc_json: "{}".to_string(),
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            image_width: 100,
            image_height: 100,
        }
    }

    pub(super) fn at_plane(mut self, z_stack: i32, t_stack: i32) -> Self {
        self.z_stack = z_stack;
        self.t_stack = t_stack;
        self
    }

    pub(super) fn with_intensities(mut self, json: &str) -> Self {
        self.intensities_json = json.to_string();
        self
    }

    pub(super) fn with_coloc(mut self, json: &str) -> Self {
        self.coloc_json = json.to_string();
        self
    }

    pub(super) fn at_centroid(mut self, x_px: f64, y_px: f64) -> Self {
        self.centroid_x_px = x_px;
        self.centroid_y_px = y_px;
        self
    }

    pub(super) fn with_image_size(mut self, width: u32, height: u32) -> Self {
        self.image_width = width;
        self.image_height = height;
        self
    }
}

/// Creates `path`'s schema and seeds one row per `objects` entry (object id
/// deterministically derived from its index), one `images` row per distinct
/// image name seen (real per-image `c_stacks`/`z_stacks`/`t_stacks`,
/// computed from the seeded objects themselves - the highest channel key
/// across every seeded `intensities_json` for that image, `+1`, and the
/// highest `z_stack`/`t_stack` used by any seeded object overall, matching
/// what a real `DuckDbExporter::finalize_image` call would have written),
/// and one `classes` row per distinct `(class_id, class_name)` pair seen.
pub(super) fn seed_db(path: &Path, objects: &[ObjectSpec]) {
    let conn = Connection::open(path).expect("open test db");
    create_schema(&conn);

    let mut images_seen: Vec<&str> = Vec::new();
    let mut classes_seen: Vec<(i32, &str)> = Vec::new();
    let mut max_channel_by_image: std::collections::HashMap<&str, i64> =
        std::collections::HashMap::new();
    let mut max_z_by_image: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
    let mut max_t_by_image: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
    let mut image_size: std::collections::HashMap<&str, (u32, u32)> = std::collections::HashMap::new();

    for (idx, spec) in objects.iter().enumerate() {
        let object_id = format!("00000000-0000-0000-0000-{idx:012}");
        conn.execute(
            "INSERT INTO objects (
                image_name, image_rel_path, t_stack, z_stack, object_id, seg_class_name, seg_class_id,
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
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, 0,
                ?, ?, 0, 0,
                0, 0, 10, 10,
                0, 0, 0, 0,
                ?, ?, 40, 40,
                1.0, 1.0, 1.0, 1.0, 1.0,
                10, 10, 1.0, false,
                1.0, 1.0, 1.0,
                ?, ?
            )",
            duckdb::params![
                spec.image,
                spec.image,
                spec.t_stack,
                spec.z_stack,
                object_id,
                spec.class_name,
                spec.class_id,
                format!("[\"{}\"]", spec.class_name),
                format!("[{}]", spec.class_id),
                spec.centroid_x_px,
                spec.centroid_y_px,
                spec.area_px,
                spec.area_px as f64,
                spec.intensities_json,
                spec.coloc_json,
            ],
        )
        .unwrap_or_else(|e| panic!("insert object {idx}: {e}"));

        if !images_seen.contains(&spec.image) {
            images_seen.push(spec.image);
        }
        image_size.insert(spec.image, (spec.image_width, spec.image_height));
        if !classes_seen
            .iter()
            .any(|(id, name)| *id == spec.class_id && *name == spec.class_name)
        {
            classes_seen.push((spec.class_id, spec.class_name));
        }

        let channel_max: i64 = parse_json_object_keys(&spec.intensities_json)
            .into_iter()
            .filter_map(|k| k.parse::<i64>().ok())
            .max()
            .unwrap_or(-1);
        let entry = max_channel_by_image.entry(spec.image).or_insert(-1);
        *entry = (*entry).max(channel_max);
        let z_entry = max_z_by_image.entry(spec.image).or_insert(0);
        *z_entry = (*z_entry).max(spec.z_stack);
        let t_entry = max_t_by_image.entry(spec.image).or_insert(0);
        *t_entry = (*t_entry).max(spec.t_stack);
    }

    for image in &images_seen {
        let c_stacks = (max_channel_by_image.get(image).copied().unwrap_or(-1) + 1).max(1);
        let z_stacks = max_z_by_image.get(image).copied().unwrap_or(0) + 1;
        let t_stacks = max_t_by_image.get(image).copied().unwrap_or(0) + 1;
        let (width, height) = image_size.get(image).copied().unwrap_or((100, 100));
        conn.execute(
            "INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, z_stacks, t_stacks) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![image, image, width, height, c_stacks, z_stacks, t_stacks],
        )
        .unwrap_or_else(|e| panic!("insert image {image}: {e}"));
    }

    for (id, name) in &classes_seen {
        conn.execute(
            "INSERT INTO classes (class_id, name, color) VALUES (?, ?, 0)",
            duckdb::params![id, name],
        )
        .unwrap_or_else(|e| panic!("insert class {name}: {e}"));
    }
}

/// Extracts a flat JSON object's top-level keys without pulling in a JSON
/// crate — good enough for the simple `{"0":{...},"1":{...}}` shape
/// `intensities_to_json` produces (used only to compute a fixture's real
/// `c_stacks` above, not to parse arbitrary JSON): tracks brace depth and
/// only treats a quoted string at depth 1 (directly inside the outer
/// object, not one of its nested values) immediately followed by `:` as a
/// key.
fn parse_json_object_keys(json: &str) -> Vec<String> {
    let chars: Vec<char> = json.chars().collect();
    let mut keys = Vec::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' => {
                depth += 1;
                i += 1;
            }
            '}' => {
                depth -= 1;
                i += 1;
            }
            '"' if depth == 1 => {
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j] != '"' {
                    j += 1;
                }
                let key: String = chars[start..j].iter().collect();
                let mut k = j + 1;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if k < chars.len() && chars[k] == ':' {
                    keys.push(key);
                }
                i = j + 1;
            }
            _ => i += 1,
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_object_keys_finds_only_top_level_keys() {
        let keys = parse_json_object_keys(CH0_INTENSITIES_JSON);
        assert_eq!(keys, vec!["0"]);
    }

    #[test]
    fn parse_json_object_keys_handles_multiple_channels() {
        let json = r#"{"0":{"a":1},"1":{"a":2},"2":{"a":3}}"#;
        assert_eq!(parse_json_object_keys(json), vec!["0", "1", "2"]);
    }

    #[test]
    fn parse_json_object_keys_empty_object_has_no_keys() {
        assert!(parse_json_object_keys("{}").is_empty());
    }
}
