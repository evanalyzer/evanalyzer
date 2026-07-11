//! Coverage tests for the generated `pipeline_command` module
//! (`crates/cfg/src/modules/pipeline_command.rs`). That file is marked
//! `// @generated - do not edit by hand`, so tests live here instead of
//! inline, where they survive regeneration.
//!
//! The module is ~31 `PipelineCommand` variants, each with near-identical
//! `name`/`category`/`allowed_next`/`to_summary`/`to_parameters`/
//! `apply_param_change` match arms. Hand-writing one test per variant would
//! just be ~31 copies of the same shape, so these walk every variant via
//! `all_command_meta`/`default_command` in a loop instead.

use evanalyzer_cfg::settings::pipeline_command::{
    all_command_meta, default_command, CommandCategory,
};

#[test]
fn all_command_meta_ids_are_contiguous_and_match_default_command() {
    let metas = all_command_meta();
    assert!(!metas.is_empty());
    for (idx, meta) in metas.iter().enumerate() {
        assert_eq!(meta.id, idx as i32, "CommandMeta ids must be contiguous from 0");
        assert!(
            default_command(meta.id).is_some(),
            "default_command({}) should exist for every id in all_command_meta()",
            meta.id
        );
        assert!(!meta.name.is_empty());
        assert!(!meta.summary.is_empty());
    }
}

#[test]
fn default_command_returns_none_out_of_range() {
    let metas = all_command_meta();
    assert!(default_command(-1).is_none());
    assert!(default_command(metas.len() as i32).is_none());
}

#[test]
fn every_command_exercises_name_category_summary_and_parameters_without_panicking() {
    for meta in all_command_meta() {
        let mut cmd = default_command(meta.id).expect("id from all_command_meta must be valid");

        // `name()`/`category()` should agree with the CommandMeta entry.
        assert_eq!(cmd.name(), meta.name, "id {}", meta.id);
        assert_eq!(*cmd.category(), meta.category, "id {}: {}", meta.id, meta.name);

        // Exercises the `allowed_next` match arm; every `CommandCategory` is
        // valid, so there's nothing further to assert.
        let _ = cmd.allowed_next();

        // Must not panic for any variant's default settings.
        let _ = cmd.to_summary();

        // Round-trip every top-level parameter through `apply_param_change`:
        // feed each parameter its own current `value` back in. This exercises
        // the corresponding match arm for essentially every field on every
        // command, and re-fetching afterwards checks the round trip actually
        // lands the same value back (catches a parser that silently
        // drops/mis-parses a value).
        let params_before = cmd.to_parameters();
        for p in &params_before {
            cmd.apply_param_change(&p.name, &p.value);
        }
        let params_after = cmd.to_parameters();
        assert_eq!(
            params_before.len(),
            params_after.len(),
            "id {}: to_parameters() length changed across a same-value round trip",
            meta.id
        );
        for (before, after) in params_before.iter().zip(params_after.iter()) {
            assert_eq!(before.name, after.name, "id {}", meta.id);
            assert_eq!(
                before.value, after.value,
                "id {} param {:?}: value changed after applying its own current value back",
                meta.id, before.name
            );
        }
    }
}

#[test]
fn apply_param_change_ignores_unknown_param_names() {
    // Every variant's `apply_param_change` falls through to a no-op for a
    // name it doesn't recognize - exercises that fallback path and confirms
    // it doesn't panic or mutate anything.
    for meta in all_command_meta() {
        let mut cmd = default_command(meta.id).expect("valid id");
        let before = cmd.to_parameters();
        cmd.apply_param_change("this_param_does_not_exist", "123");
        let after = cmd.to_parameters();
        assert_eq!(
            before.len(),
            after.len(),
            "id {}: unknown param name should be a no-op",
            meta.id
        );
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.value, a.value, "id {}: unknown param name mutated {:?}", meta.id, b.name);
        }
    }
}

#[test]
fn add_and_remove_group_item_do_not_panic_for_any_command() {
    for meta in all_command_meta() {
        let mut cmd = default_command(meta.id).expect("valid id");
        // Only `Threshold`'s "thresholds" param actually does anything here;
        // every other variant's `add_group_item`/`remove_group_item` is a
        // no-op. Calling it for every param name on every command still
        // exercises each variant's match arm and confirms none of them
        // panic, including the no-op ones.
        for p in cmd.to_parameters() {
            cmd.add_group_item(&p.name);
        }
        // Removing at index 0 repeatedly, including once the group is
        // already empty again, confirms `remove_group_item` bounds-checks
        // `idx` rather than panicking via an unchecked `Vec::remove`.
        for p in cmd.to_parameters() {
            cmd.remove_group_item(&p.name, 0);
            cmd.remove_group_item(&p.name, 0);
        }
    }
}

#[test]
fn threshold_group_items_round_trip_through_nested_apply_param_change() {
    // `Threshold` is the only command with a `ParamType::Group` parameter
    // (verified by grep - it's the sole `param_type: ParamType::Group` in
    // the generated file), addressed with a dotted `thresholds.{idx}.{field}`
    // path. This is the one bit of `apply_param_change` the generic
    // top-level round trip above can't reach.
    let mut cmd = default_command(25).expect("id 25 is Threshold");
    assert_eq!(cmd.name(), "Threshold");

    cmd.add_group_item("thresholds");
    let params = cmd.to_parameters();
    let thresholds_param = params.iter().find(|p| p.name == "thresholds").unwrap();
    assert_eq!(thresholds_param.groups.len(), 1, "add_group_item should have added one row");

    let nested_before = &thresholds_param.groups[0];
    assert!(!nested_before.is_empty(), "the new threshold row should have its own fields");

    for nested in nested_before {
        let dotted = format!("thresholds.0.{}", nested.name);
        cmd.apply_param_change(&dotted, &nested.value);
    }

    let params_after = cmd.to_parameters();
    let nested_after = &params_after.iter().find(|p| p.name == "thresholds").unwrap().groups[0];
    for (before, after) in nested_before.iter().zip(nested_after.iter()) {
        assert_eq!(before.name, after.name);
        assert_eq!(
            before.value, after.value,
            "threshold row field {:?} changed after applying its own current value back",
            before.name
        );
    }

    // Actually change a value through the nested path (not just round-trip
    // the same one), to confirm the mutation really lands.
    cmd.apply_param_change("thresholds.0.min_threshold", "42");
    let params_changed = cmd.to_parameters();
    let nested_changed = &params_changed.iter().find(|p| p.name == "thresholds").unwrap().groups[0];
    let min_threshold = nested_changed.iter().find(|p| p.name == "min_threshold").unwrap();
    assert_eq!(min_threshold.value, "42");

    cmd.remove_group_item("thresholds", 0);
    let params_removed = cmd.to_parameters();
    let thresholds_removed = params_removed.iter().find(|p| p.name == "thresholds").unwrap();
    assert!(thresholds_removed.groups.is_empty(), "remove_group_item should have removed the row");
}

#[test]
fn command_category_helpers_cover_every_variant() {
    let categories = [
        CommandCategory::Preprocess,
        CommandCategory::Segment,
        CommandCategory::Object,
        CommandCategory::Measure,
        CommandCategory::Classify,
    ];
    for cat in categories {
        // These are `#[allow(dead_code)]` (unused outside tests currently),
        // which is exactly why they were at 0% coverage.
        let order = cat.display_order();
        assert!(order <= 4);
        let _ = cat.allowed_after();
        let _ = cat.suggested_next();
    }
}
