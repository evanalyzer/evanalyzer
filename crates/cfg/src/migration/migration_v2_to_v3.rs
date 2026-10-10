//! Migrates a [`ProjectSettings`](crate::settings::project_settings::ProjectSettings)
//! document from schema version 2 to version 3.
//!
//! Version 3 reworks the plate settings (`plate`) so the project's values can
//! drive the results window directly:
//!
//! - `groupingMode` `NO_GROUPING`/`FOLDER_NAME`/`FILE_NAME` becomes
//!   `AUTO`/`FOLDER`/`CUSTOM`. The two built-in file name presets become
//!   `AUTO`, which detects the pattern itself; any other regex stays `CUSTOM`.
//! - `plateRows`/`plateCols` become `plateSize`. The old default 1 x 1 (and
//!   any size without a fixed format) becomes `AUTO`.
//! - `wellRows`/`wellCols` become `wellLayout`. The untouched old default (a
//!   4 x 4 grid in number order) becomes `AUTO`, anything else
//!   `{"FIXED": {"rows", "cols"}}`.
//! - `wellImageOrder` keeps its values (negative ones are dropped).

use serde_json::{Map, Value, json};

/// The two file name presets the old grouping dropdown offered.
const OLD_FILE_NAME_PRESETS: [&str; 2] = ["(.*)_([0-9]*)", "((.)([0-9]+))_([0-9]+)"];

fn int(plate: &Map<String, Value>, key: &str) -> i64 {
    plate.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn migrate_plate(plate: &mut Map<String, Value>) {
    let regex = plate
        .get("groupingRegex")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mode = match plate.get("groupingMode").and_then(Value::as_str) {
        Some("FOLDER_NAME") => "FOLDER",
        Some("FILE_NAME")
            if !regex.is_empty() && !OLD_FILE_NAME_PRESETS.contains(&regex.as_str()) =>
        {
            "CUSTOM"
        }
        _ => "AUTO",
    };
    plate.insert("groupingMode".into(), json!(mode));
    plate.insert("groupingRegex".into(), json!(regex));

    let (plate_rows, plate_cols) = (int(plate, "plateRows"), int(plate, "plateCols"));
    let plate_size = match (plate_rows, plate_cols) {
        (2, 3)
        | (2, 4)
        | (2, 6)
        | (3, 4)
        | (3, 5)
        | (3, 6)
        | (4, 6)
        | (6, 8)
        | (8, 12)
        | (16, 24)
        | (32, 48)
        | (48, 72) => format!("PLATE{plate_rows}X{plate_cols}"),
        _ => "AUTO".to_string(),
    };
    plate.insert("plateSize".into(), json!(plate_size));

    let order: Vec<u32> = plate
        .get("wellImageOrder")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_i64)
                .filter_map(|v| u32::try_from(v).ok())
                .collect()
        })
        .unwrap_or_default();
    let (well_rows, well_cols) = (int(plate, "wellRows"), int(plate, "wellCols"));
    let untouched_default =
        well_rows == 4 && well_cols == 4 && order == (1..=16).collect::<Vec<u32>>();
    let well_layout = if well_rows > 0 && well_cols > 0 && !untouched_default {
        json!({ "FIXED": { "rows": well_rows, "cols": well_cols } })
    } else {
        json!("AUTO")
    };
    plate.insert("wellLayout".into(), well_layout);
    plate.insert("wellImageOrder".into(), json!(order));

    for old in ["plateRows", "plateCols", "wellRows", "wellCols"] {
        plate.remove(old);
    }
}

pub fn migrate_from_v2_to_v3(raw: &mut Value) {
    if let Some(Value::Object(plate)) = raw.get_mut("plate") {
        migrate_plate(plate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::plate_settings::{GroupingMode, PlateSettings, PlateSize, WellLayout};

    fn migrated(plate: Value) -> PlateSettings {
        let mut raw = json!({ "plate": plate });
        migrate_from_v2_to_v3(&mut raw);
        serde_json::from_value(raw["plate"].clone()).expect("a valid v3 plate")
    }

    #[test]
    fn the_old_defaults_become_auto() {
        let plate = migrated(json!({
            "groupingMode": "NO_GROUPING", "groupingRegex": "",
            "plateCols": 1, "plateRows": 1, "wellCols": 4, "wellRows": 4,
            "wellImageOrder": (1..=16).collect::<Vec<u32>>(),
        }));
        assert_eq!(plate.grouping_mode, GroupingMode::Auto);
        assert_eq!(plate.plate_size, PlateSize::Auto);
        assert_eq!(plate.well_layout, WellLayout::Auto);
    }

    #[test]
    fn chosen_values_are_kept() {
        let plate = migrated(json!({
            "groupingMode": "FILE_NAME", "groupingRegex": "^(([A-H])([0-9]+))-([0-9]+)",
            "plateCols": 12, "plateRows": 8, "wellCols": 3, "wellRows": 2,
            "wellImageOrder": [3, 2, 1, -1, 6, 5, 4],
        }));
        assert_eq!(plate.grouping_mode, GroupingMode::Custom);
        assert_eq!(plate.grouping_regex, "^(([A-H])([0-9]+))-([0-9]+)");
        assert_eq!(plate.plate_size, PlateSize::Plate8x12);
        assert_eq!(plate.well_layout, WellLayout::Fixed { rows: 2, cols: 3 });
        assert_eq!(plate.well_image_order, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn folders_and_the_old_file_name_presets_map_to_the_new_modes() {
        let folder = migrated(json!({ "groupingMode": "FOLDER_NAME", "groupingRegex": "" }));
        assert_eq!(folder.grouping_mode, GroupingMode::Folder);
        for preset in OLD_FILE_NAME_PRESETS {
            let plate = migrated(json!({ "groupingMode": "FILE_NAME", "groupingRegex": preset }));
            assert_eq!(plate.grouping_mode, GroupingMode::Auto, "{preset}");
        }
    }
}
