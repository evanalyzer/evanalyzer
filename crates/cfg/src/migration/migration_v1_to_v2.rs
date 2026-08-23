//! Migrates a [`ProjectSettings`](crate::settings::project_settings::ProjectSettings)
//! document from schema version 1 to version 2.
//!
//! Version 2 fixes two related generator bugs that both changed how
//! internally-tagged enums serialize:
//!
//! 1. Every enum's `"type"` discriminator (every pipeline step's command,
//!    plus any nested rich-enum field, e.g. `IlluminationCorrection.smoothing`)
//!    was written in camelCase (e.g. `"gaussianBlur"`) instead of
//!    SCREAMING_SNAKE_CASE (`"GAUSSIAN_BLUR"`) - inconsistent with every
//!    *plain* enum's values, which were always SCREAMING_SNAKE_CASE.
//! 2. `ThresholdMethod::Otsu` gained a nested `classes` field (two-/
//!    three-class support), which makes serde treat the *whole*
//!    `ThresholdMethod` enum as internally-tagged (`{"type": "TRIANGLE"}`)
//!    instead of a bare string (`"TRIANGLE"`) - every variant's wire shape
//!    changed, not just Otsu's, since serde can't mix "bare string for unit
//!    variants" and "object for data-carrying variants" within one derived
//!    enum.
//!
//! Both walk the whole document rather than a fixed path, since either kind
//! of value can appear at any depth (nested inside pipelines/steps/commands).

use serde_json::Value;

/// Converts a camelCase identifier (as `rename_all = "camelCase"` used to
/// produce for every internally-tagged enum's `"type"` discriminator) into
/// the SCREAMING_SNAKE_CASE serde now writes: inserts `_` before every
/// non-initial uppercase letter, then uppercases the whole thing. Operating
/// on the *old* camelCase string (rather than the original Rust PascalCase
/// identifier, which isn't available at migration time) works because the
/// only difference between the two is the very first letter's case, and that
/// position never affects word-boundary detection either way.
fn camel_case_to_screaming_snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_uppercase());
    }
    out
}

/// Walks the whole document and re-cases every `"type"` string that looks
/// like the old camelCase form (starts with a lowercase letter - the new
/// form never does).
fn migrate_command_type_tags_to_screaming_snake_case(raw: &mut Value) {
    match raw {
        Value::Object(map) => {
            if let Some(Value::String(ty)) = map.get_mut("type") {
                if ty.starts_with(|c: char| c.is_ascii_lowercase()) {
                    *ty = camel_case_to_screaming_snake_case(ty);
                }
            }
            for value in map.values_mut() {
                migrate_command_type_tags_to_screaming_snake_case(value);
            }
        }
        Value::Array(arr) => {
            for value in arr.iter_mut() {
                migrate_command_type_tags_to_screaming_snake_case(value);
            }
        }
        _ => {}
    }
}

/// Walks the whole document for any `"thresholds"` array (however deep it's
/// nested inside pipelines/steps) and rewrites each entry's bare-string
/// `method` into the new tagged-object shape; `"OTSU"` maps to the new
/// two-class case (the only kind version-1 documents could express).
fn migrate_threshold_method_to_tagged_shape(raw: &mut Value) {
    match raw {
        Value::Object(map) => {
            if let Some(Value::Array(thresholds)) = map.get_mut("thresholds") {
                for entry in thresholds.iter_mut() {
                    let Some(entry) = entry.as_object_mut() else {
                        continue;
                    };
                    let Some(Value::String(name)) = entry.get("method") else {
                        continue;
                    };
                    let new_method = if name == "OTSU" {
                        serde_json::json!({ "type": "OTSU", "classes": { "type": "TWO" } })
                    } else {
                        serde_json::json!({ "type": name })
                    };
                    entry.insert("method".to_string(), new_method);
                }
            }
            for value in map.values_mut() {
                migrate_threshold_method_to_tagged_shape(value);
            }
        }
        Value::Array(arr) => {
            for value in arr.iter_mut() {
                migrate_threshold_method_to_tagged_shape(value);
            }
        }
        _ => {}
    }
}

/// Migrates `raw` in place from version 1 to version 2 - the single entry
/// point `versioning::PROJECT_MIGRATIONS` wires in for this step.
pub fn migrate_from_v1_to_v2(raw: &mut Value) {
    migrate_command_type_tags_to_screaming_snake_case(raw);
    migrate_threshold_method_to_tagged_shape(raw);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_case_to_screaming_snake_case_covers_one_two_and_three_word_names() {
        assert_eq!(camel_case_to_screaming_snake_case("threshold"), "THRESHOLD");
        assert_eq!(
            camel_case_to_screaming_snake_case("gaussianBlur"),
            "GAUSSIAN_BLUR"
        );
        assert_eq!(
            camel_case_to_screaming_snake_case("illuminationCorrection"),
            "ILLUMINATION_CORRECTION"
        );
    }

    #[test]
    fn migrate_command_type_tags_rewrites_every_camel_case_type_key_regardless_of_nesting() {
        let mut raw = serde_json::json!({
            "pipelines": [{
                "steps": [{
                    "enabled": true,
                    "command": {
                        "type": "illuminationCorrection",
                        "method": "REGULAR",
                        "smoothing": { "type": "fitPolynomial" }
                    }
                }]
            }]
        });
        migrate_command_type_tags_to_screaming_snake_case(&mut raw);
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["type"],
            "ILLUMINATION_CORRECTION"
        );
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["smoothing"]["type"],
            "FIT_POLYNOMIAL"
        );
        // Already-plain SCREAMING_SNAKE_CASE values (not under a "type" key)
        // must be left alone - this migration only touches "type" tags.
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["method"],
            "REGULAR"
        );
    }

    #[test]
    fn migrate_command_type_tags_is_a_no_op_on_an_already_migrated_document() {
        let mut raw = serde_json::json!({ "command": { "type": "ILLUMINATION_CORRECTION" } });
        let before = raw.clone();
        migrate_command_type_tags_to_screaming_snake_case(&mut raw);
        assert_eq!(raw, before);
    }

    #[test]
    fn migrate_threshold_method_wraps_bare_strings_and_special_cases_otsu() {
        let mut raw = serde_json::json!({
            "thresholds": [
                { "method": "TRIANGLE", "minThreshold": 0.0 },
                { "method": "OTSU", "minThreshold": 0.0 },
            ]
        });
        migrate_threshold_method_to_tagged_shape(&mut raw);
        assert_eq!(
            raw["thresholds"][0]["method"],
            serde_json::json!({"type": "TRIANGLE"})
        );
        assert_eq!(
            raw["thresholds"][1]["method"],
            serde_json::json!({"type": "OTSU", "classes": {"type": "TWO"}})
        );
    }

    #[test]
    fn migrate_threshold_method_finds_thresholds_arrays_at_any_depth() {
        let mut raw = serde_json::json!({
            "pipelines": [{
                "steps": [{
                    "command": {
                        "type": "THRESHOLD",
                        "thresholds": [{ "method": "MANUAL" }]
                    }
                }]
            }]
        });
        migrate_threshold_method_to_tagged_shape(&mut raw);
        assert_eq!(
            raw["pipelines"][0]["steps"][0]["command"]["thresholds"][0]["method"],
            serde_json::json!({"type": "MANUAL"})
        );
    }

    #[test]
    fn migrate_threshold_method_is_a_no_op_on_an_already_migrated_document() {
        let mut raw = serde_json::json!({
            "thresholds": [{ "method": { "type": "TRIANGLE" } }]
        });
        let before = raw.clone();
        migrate_threshold_method_to_tagged_shape(&mut raw);
        assert_eq!(raw, before);
    }

    // The full end-to-end path (a real version-1 document - no `schemaVersion`,
    // camelCase command tags, a bare-string `ThresholdMethod` including a
    // plain `"OTSU"` - loaded through `load_project_settings`) is covered by
    // `example_project_deserializes` in `lib.rs` against the real
    // `tests/fixtures/example.improj` fixture, rather than a hand-built
    // document here: `ProjectSettings`/`PipelineSettings` have enough other
    // required fields that a minimal synthetic JSON risks failing for
    // reasons unrelated to this migration.
}
