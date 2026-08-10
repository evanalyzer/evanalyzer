//! Coverage tests for the generated `pipeline_command` module
//! (`crates/cfg/src/modules/pipeline_command.rs`). That file is marked
//! `// @generated - do not edit by hand`, so tests live here instead of
//! inline, where they survive regeneration.
//!
//! The module is ~34 `PipelineCommand` variants, each with near-identical
//! `name`/`category`/`allowed_next`/`to_summary`/`to_parameters`/
//! `apply_param_change` match arms. Hand-writing one test per variant would
//! just be ~33 copies of the same shape, so these walk every variant via
//! `all_command_meta`/`default_command` in a loop instead.
//!
//! Tests that need one *specific* command look it up by name via [`id_of`]
//! rather than hardcoding its numeric id - ids are assigned alphabetically
//! by struct name at generation time, so a hardcoded id silently points at
//! the wrong command the moment a new command is added earlier in that
//! ordering (see [`id_of`]'s doc comment).

use evanalyzer_cfg::settings::pipeline_command::{
    CommandCategory, PipelineCommand, all_command_meta, default_command,
};

/// Resolves a command's *current* numeric id by its display name via
/// `all_command_meta()`, instead of the tests hardcoding a literal id.
///
/// `PipelineCommand` variants are id-ordered alphabetically by struct name
/// (`pipeline_commands_generator.rs` sorts them at generation time), so
/// inserting a single new command shifts every later id by one - hardcoded
/// ids throughout this file silently pointed at the wrong command the moment
/// that happened (a prior version of this file that shipped before
/// `FillHoles` existed broke exactly this way). Looking the id up by name
/// every time makes the whole file immune to that class of bug permanently.
fn id_of(name: &str) -> i32 {
    all_command_meta()
        .into_iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("no command named {name:?} in all_command_meta()"))
        .id
}

fn param_value(cmd: &PipelineCommand, name: &str) -> String {
    cmd.to_parameters()
        .into_iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("missing parameter '{name}'"))
        .value
}

fn param_value_nested(cmd: &PipelineCommand, group_name: &str, idx: usize, field: &str) -> String {
    cmd.to_parameters()
        .into_iter()
        .find(|p| p.name == group_name)
        .unwrap_or_else(|| panic!("missing group parameter '{group_name}'"))
        .groups
        .get(idx)
        .unwrap_or_else(|| panic!("missing group row {idx} in '{group_name}'"))
        .iter()
        .find(|p| p.name == field)
        .unwrap_or_else(|| panic!("missing field '{field}' in group row {idx}"))
        .value
        .clone()
}

#[test]
fn all_command_meta_ids_are_contiguous_and_match_default_command() {
    let metas = all_command_meta();
    assert!(!metas.is_empty());
    for (idx, meta) in metas.iter().enumerate() {
        assert_eq!(
            meta.id, idx as i32,
            "CommandMeta ids must be contiguous from 0"
        );
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
        assert_eq!(
            *cmd.category(),
            meta.category,
            "id {}: {}",
            meta.id,
            meta.name
        );

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
            assert_eq!(
                b.value, a.value,
                "id {}: unknown param name mutated {:?}",
                meta.id, b.name
            );
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
    let mut cmd = default_command(id_of("Threshold")).expect("Threshold must exist");
    assert_eq!(cmd.name(), "Threshold");

    cmd.add_group_item("thresholds");
    let params = cmd.to_parameters();
    let thresholds_param = params.iter().find(|p| p.name == "thresholds").unwrap();
    assert_eq!(
        thresholds_param.groups.len(),
        1,
        "add_group_item should have added one row"
    );

    let nested_before = &thresholds_param.groups[0];
    assert!(
        !nested_before.is_empty(),
        "the new threshold row should have its own fields"
    );

    for nested in nested_before {
        let dotted = format!("thresholds.0.{}", nested.name);
        cmd.apply_param_change(&dotted, &nested.value);
    }

    let params_after = cmd.to_parameters();
    let nested_after = &params_after
        .iter()
        .find(|p| p.name == "thresholds")
        .unwrap()
        .groups[0];
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
    let nested_changed = &params_changed
        .iter()
        .find(|p| p.name == "thresholds")
        .unwrap()
        .groups[0];
    let min_threshold = nested_changed
        .iter()
        .find(|p| p.name == "min_threshold")
        .unwrap();
    assert_eq!(min_threshold.value, "42");

    cmd.remove_group_item("thresholds", 0);
    let params_removed = cmd.to_parameters();
    let thresholds_removed = params_removed
        .iter()
        .find(|p| p.name == "thresholds")
        .unwrap();
    assert!(
        thresholds_removed.groups.is_empty(),
        "remove_group_item should have removed the row"
    );
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

// ---------------------------------------------------------------------
// The tests above exercise every variant generically by round-tripping
// each parameter's *own current default value* back through
// `apply_param_change`. That's enough to prove the getter/setter for a
// field agree and don't panic, but for fields backed by a multi-arm
// match (dropdowns), an `Option`-like class field (`ObjClass`), or a
// list field (`MultiObjClass`), the *default* value only ever exercises
// one arm/branch. The tests below feed genuinely different values (and,
// for numeric fields, deliberately unparsable ones) to reach the
// remaining arms/branches of `apply_param_change`/`to_parameters`.
// ---------------------------------------------------------------------

#[test]
fn command_category_display_order_is_sequential_by_pipeline_stage() {
    assert_eq!(CommandCategory::Preprocess.display_order(), 0);
    assert_eq!(CommandCategory::Segment.display_order(), 1);
    assert_eq!(CommandCategory::Object.display_order(), 2);
    assert_eq!(CommandCategory::Measure.display_order(), 3);
    assert_eq!(CommandCategory::Classify.display_order(), 4);
}

#[test]
fn command_category_allowed_after_matches_expected_predecessors() {
    assert_eq!(
        CommandCategory::Preprocess.allowed_after(),
        &[CommandCategory::Preprocess]
    );
    assert_eq!(
        CommandCategory::Segment.allowed_after(),
        &[CommandCategory::Preprocess, CommandCategory::Segment]
    );
    assert_eq!(
        CommandCategory::Object.allowed_after(),
        &[CommandCategory::Segment, CommandCategory::Object]
    );
    assert_eq!(
        CommandCategory::Measure.allowed_after(),
        &[CommandCategory::Object, CommandCategory::Measure]
    );
    assert_eq!(
        CommandCategory::Classify.allowed_after(),
        &[CommandCategory::Measure, CommandCategory::Classify]
    );
}

#[test]
fn command_category_suggested_next_advances_and_terminates_at_classify() {
    assert_eq!(
        CommandCategory::Preprocess.suggested_next(),
        CommandCategory::Segment
    );
    assert_eq!(
        CommandCategory::Segment.suggested_next(),
        CommandCategory::Object
    );
    assert_eq!(
        CommandCategory::Object.suggested_next(),
        CommandCategory::Measure
    );
    assert_eq!(
        CommandCategory::Measure.suggested_next(),
        CommandCategory::Classify
    );
    // Classify is the terminal stage: it suggests itself rather than looping
    // back or panicking.
    assert_eq!(
        CommandCategory::Classify.suggested_next(),
        CommandCategory::Classify
    );
}

#[test]
fn allowed_next_returns_expected_categories_for_every_variant() {
    use CommandCategory::*;
    let expected: [(&str, &[CommandCategory]); 34] = [
        ("AI Object Classifier", &[Classify]),
        ("Blur", &[Segment, Preprocess]),
        ("AI Cellpose Segmentation", &[Measure]),
        ("ClassifyObjects", &[Classify]),
        ("Colocalization", &[Classify]),
        ("ColorFilterCommand", &[Segment, Preprocess]),
        ("ConnectedComponents", &[Object, Measure]),
        ("DistanceTransform", &[Segment, Preprocess]),
        ("EdgeDetectionCanny", &[Segment, Preprocess]),
        ("EdgeDetectionSobel", &[Segment, Preprocess]),
        ("EnhanceContrast", &[Segment, Preprocess]),
        ("ExtractObjects", &[Classify]),
        // Not `default_next_for_category("Object")` (`[Measure, Object]`) -
        // pinned to just `[Object]`: `Measure`-stage commands like
        // ExtractObjects expect instance IDs from ConnectedComponents/
        // Watershed, which haven't run yet right after FillHoles.
        ("FillHoles", &[Object]),
        ("GaussianBlur", &[Segment, Preprocess]),
        ("Hessian", &[Segment, Preprocess]),
        ("ImageCache", &[Segment, Preprocess]),
        ("ImageMath", &[Segment, Preprocess]),
        ("IntensityTransformation", &[Segment, Preprocess]),
        ("Laplacian", &[Segment, Preprocess]),
        ("MedianSubtract", &[Segment, Preprocess]),
        ("MorphologicalCommand", &[Segment, Preprocess]),
        ("ObjectMath", &[Classify]),
        ("AI Pixel Classifier", &[Object]),
        ("RankFilter", &[Segment, Preprocess]),
        ("RollingBall", &[Segment, Preprocess]),
        ("SaveImage", &[Segment, Preprocess]),
        ("AI Stardist Segmentation", &[Measure]),
        ("StructureTensor", &[Segment, Preprocess]),
        // Not the default `[Segment]` for `Threshold`'s own category -
        // pinned to `[Object]` so `object`-category instance labeling
        // (ConnectedComponents) is suggested right after it.
        ("Threshold", &[Object]),
        ("TransformObjects", &[Classify]),
        ("AI UNet Segmentation", &[Object]),
        ("Voronoi", &[Classify]),
        ("Watershed", &[Measure]),
        ("WeightedDeviation", &[Segment, Preprocess]),
    ];
    assert_eq!(
        expected.len(),
        all_command_meta().len(),
        "this table must list every PipelineCommand variant - a command was \
         added/removed without updating it"
    );
    for (name, categories) in expected {
        let id = id_of(name);
        let cmd = default_command(id).unwrap();
        assert_eq!(cmd.name(), name, "id_of({name:?}) returned the wrong id");
        assert_eq!(
            cmd.allowed_next(),
            categories,
            "{name}: unexpected allowed_next()"
        );
    }
}

#[test]
fn to_summary_is_empty_for_variants_without_a_custom_summary() {
    for name in [
        "AI Cellpose Segmentation",
        "ImageCache",
        "SaveImage",
        "Threshold",
        "Watershed",
    ] {
        let cmd = default_command(id_of(name)).unwrap();
        assert_eq!(cmd.to_summary(), "", "{name}");
    }
}

#[test]
fn to_summary_formats_connected_components_min_size() {
    // `min_size` is an `i32`; `{:.3}` precision formatting has no effect on
    // integers (only on floats), so the summary is deliberately unpadded.
    let mut cmd = default_command(id_of("ConnectedComponents")).unwrap();
    cmd.apply_param_change("min_size", "12");
    assert_eq!(cmd.to_summary(), "Min Size: 12");
}

#[test]
fn to_summary_formats_blur_kernel_size() {
    // `kernel_size` is a `usize`; `{:.3}` precision formatting has no effect
    // on integers (only on floats), so the summary is deliberately unpadded.
    let mut cmd = default_command(id_of("Blur")).unwrap();
    cmd.apply_param_change("kernel_size", "5");
    assert_eq!(cmd.to_summary(), "Kernel size: 5");
}

#[test]
fn to_summary_formats_gaussian_blur_kernel_and_sigma() {
    let mut cmd = default_command(id_of("GaussianBlur")).unwrap();
    cmd.apply_param_change("kernel_size", "7");
    cmd.apply_param_change("sigma", "1.5");
    assert_eq!(cmd.to_summary(), "Kernel Size: 7 · Sigma: 1.500");
}

#[test]
fn to_summary_formats_classify_objects_criteria() {
    let mut cmd = default_command(id_of("ClassifyObjects")).unwrap();
    cmd.apply_param_change("min_area", "10");
    cmd.apply_param_change("min_eccentricity", "0.1");
    cmd.apply_param_change("max_eccentricity", "0.9");
    cmd.apply_param_change("allow_edge_touching", "true");
    assert_eq!(
        cmd.to_summary(),
        "Min Area: 10.000 · Min Eccentricity: 0.100 · Max Eccentricity: 0.900 · Allow Edge Touching: true"
    );
}

#[test]
fn to_summary_formats_object_math_operation() {
    let mut cmd = default_command(id_of("ObjectMath")).unwrap();
    cmd.apply_param_change("operation", "Xor");
    assert_eq!(cmd.to_summary(), "Operation: Xor");
}

#[test]
fn to_summary_formats_transform_objects_function() {
    let mut cmd = default_command(id_of("TransformObjects")).unwrap();
    cmd.apply_param_change("function", "Shrink");
    assert_eq!(cmd.to_summary(), "Function: Shrink");
}

#[test]
fn apply_param_change_dropdown_fields_cycle_through_every_option() {
    // Every dropdown/enum-valued top-level field, driven straight from
    // `to_parameters()`'s own `options` list so this doesn't drift from the
    // generated code. Exercises every match arm, not just whichever one the
    // default settings happen to start on.
    let dropdown_fields: &[(&str, &str)] = &[
        ("Hessian", "mode"),
        ("ImageCache", "mode"),
        ("ImageMath", "operand"),
        ("IntensityTransformation", "mode"),
        ("MorphologicalCommand", "op"),
        ("MorphologicalCommand", "kernel_shape"),
        // RankFilter's `filter_type` is deliberately excluded here: its
        // options include "Outliers", a data-carrying arm that
        // `apply_param_change` can't reconstruct from a bare string (see
        // `apply_param_change_rank_filter_ignores_unknown_filter_type_value`
        // below), so cycling through every option would fail here.
        ("ObjectMath", "operation"),
        ("RollingBall", "ball_type"),
        ("SaveImage", "source"),
        ("StructureTensor", "mode"),
        ("ClassifyObjects", "match_handling"),
        ("TransformObjects", "function"),
        ("AI UNet Segmentation", "output_mode"),
    ];
    for &(name, field) in dropdown_fields {
        let mut cmd = default_command(id_of(name)).unwrap();
        let options = cmd
            .to_parameters()
            .into_iter()
            .find(|p| p.name == field)
            .unwrap_or_else(|| panic!("{name}: missing parameter '{field}'"))
            .options;
        assert!(
            !options.is_empty(),
            "{name}: '{field}' has no options to cycle through"
        );
        for option in &options {
            cmd.apply_param_change(field, option);
            assert_eq!(
                param_value(&cmd, field),
                *option,
                "{name}: setting '{field}' to {option:?} didn't stick"
            );
        }
    }
}

#[test]
fn apply_param_change_unit_fields_toggle_both_variants() {
    // SizeUnits fields.
    for (name, field) in [
        ("ClassifyObjects", "size_unit"),
        ("Colocalization", "size_unit"),
        ("ObjectMath", "size_unit"),
        ("Voronoi", "unit"),
    ] {
        let mut cmd = default_command(id_of(name)).unwrap();
        cmd.apply_param_change(field, "nm");
        assert_eq!(param_value(&cmd, field), "nm", "{name}: {field}=nm");
        cmd.apply_param_change(field, "px");
        assert_eq!(param_value(&cmd, field), "px", "{name}: {field}=px");
    }

    // PixelUnits, nested inside a Threshold group entry.
    let mut cmd = default_command(id_of("Threshold")).unwrap();
    cmd.add_group_item("thresholds");
    for unit in ["bit", "%", "rel"] {
        cmd.apply_param_change("thresholds.0.unit", unit);
        let params = cmd.to_parameters();
        let nested = &params
            .iter()
            .find(|p| p.name == "thresholds")
            .unwrap()
            .groups[0];
        let value = &nested.iter().find(|p| p.name == "unit").unwrap().value;
        assert_eq!(value, unit);
    }
}

#[test]
fn apply_param_change_obj_class_fields_support_unset_and_valid_ids() {
    // One ObjClass field from each variant that has one, covering the
    // `"-1"` (Unset) branch and the `value.parse::<u32>()` (Valid) branch.
    let obj_class_fields: &[(&str, &str)] = &[
        ("ClassifyObjects", "output_class"),
        ("ClassifyObjects", "overlapping_with"),
        ("Colocalization", "class_for_overlapping_areas"),
        ("ObjectMath", "input_class"),
        ("ObjectMath", "other_class"),
        ("ObjectMath", "output_class"),
        ("TransformObjects", "input_class"),
        ("TransformObjects", "output_class"),
        ("Voronoi", "centers"),
        ("Voronoi", "mask"),
        ("Voronoi", "output_class"),
    ];
    for &(name, field) in obj_class_fields {
        let mut cmd = default_command(id_of(name)).unwrap();
        cmd.apply_param_change(field, "7");
        assert_eq!(param_value(&cmd, field), "7", "{name}: {field}=7");
        cmd.apply_param_change(field, "-1");
        assert_eq!(param_value(&cmd, field), "-1", "{name}: {field}=-1 (Unset)");
        // A value that doesn't parse as u32 must leave the field untouched.
        cmd.apply_param_change(field, "-1");
        cmd.apply_param_change(field, "not_a_number");
        assert_eq!(
            param_value(&cmd, field),
            "-1",
            "{name}: {field} garbage input should be a no-op"
        );
    }
}

#[test]
fn apply_param_change_multi_obj_class_fields_support_comma_lists_and_toggle() {
    let multi_obj_class_fields: &[(&str, &str)] = &[
        ("ClassifyObjects", "input_classes"),
        ("Colocalization", "classes_to_coloc"),
        // Colocalization.filter_classes is `#[cmdsmeta(visible = false)]`, so it
        // never reaches `to_parameters()` - not tested here.
        ("Colocalization", "exclude_classes"),
        ("ObjectMath", "other_filter_classes"),
        ("Voronoi", "center_filter_classes"),
        ("Voronoi", "mask_filter_classes"),
    ];
    for &(name, field) in multi_obj_class_fields {
        let mut cmd = default_command(id_of(name)).unwrap();

        // Comma-separated list, including a blank and a non-numeric entry
        // that must be silently dropped.
        cmd.apply_param_change(field, "1,,x,2,3");
        assert_eq!(
            param_value(&cmd, field),
            "1,2,3",
            "{name}: {field} comma list"
        );

        // `toggle:` removes an id already present...
        cmd.apply_param_change(field, "toggle:2");
        assert_eq!(
            param_value(&cmd, field),
            "1,3",
            "{name}: {field} toggle off"
        );

        // ...and adds one that's absent.
        cmd.apply_param_change(field, "toggle:9");
        assert_eq!(
            param_value(&cmd, field),
            "1,3,9",
            "{name}: {field} toggle on"
        );
    }
}

#[test]
fn apply_param_change_transform_objects_switches_through_every_function_kind() {
    let mut cmd = default_command(id_of("TransformObjects")).unwrap();

    cmd.apply_param_change("function", "Scale");
    cmd.apply_param_change("function.factor", "2.5");
    assert_eq!(param_value(&cmd, "function.factor"), "2.5");

    cmd.apply_param_change("function", "Snap Area");
    cmd.apply_param_change("function.extra_size", "4");
    cmd.apply_param_change("function.unit", "nm");
    assert_eq!(param_value(&cmd, "function.extra_size"), "4");
    assert_eq!(param_value(&cmd, "function.unit"), "nm");
    cmd.apply_param_change("function.unit", "px");
    assert_eq!(param_value(&cmd, "function.unit"), "px");

    cmd.apply_param_change("function", "Min Circle");
    cmd.apply_param_change("function.min_diameter", "6");
    cmd.apply_param_change("function.unit", "nm");
    assert_eq!(param_value(&cmd, "function.min_diameter"), "6");
    assert_eq!(param_value(&cmd, "function.unit"), "nm");
    cmd.apply_param_change("function.unit", "px");
    assert_eq!(param_value(&cmd, "function.unit"), "px");

    cmd.apply_param_change("function", "Draw Circle");
    cmd.apply_param_change("function.diameter", "12");
    cmd.apply_param_change("function.unit", "nm");
    assert_eq!(param_value(&cmd, "function.diameter"), "12");
    assert_eq!(param_value(&cmd, "function.unit"), "nm");
    cmd.apply_param_change("function.unit", "px");
    assert_eq!(param_value(&cmd, "function.unit"), "px");

    cmd.apply_param_change("function", "Fitting Ellipse");
    cmd.apply_param_change("function.scale", "1.8");
    assert_eq!(param_value(&cmd, "function.scale"), "1.8");

    cmd.apply_param_change("function", "Expand");
    cmd.apply_param_change("function.margin", "3");
    cmd.apply_param_change("function.unit", "nm");
    assert_eq!(param_value(&cmd, "function.margin"), "3");
    assert_eq!(param_value(&cmd, "function.unit"), "nm");
    cmd.apply_param_change("function.unit", "px");
    assert_eq!(param_value(&cmd, "function.unit"), "px");

    cmd.apply_param_change("function", "Shrink");
    cmd.apply_param_change("function.margin", "1.5");
    cmd.apply_param_change("function.unit", "nm");
    assert_eq!(param_value(&cmd, "function.margin"), "1.5");
    assert_eq!(param_value(&cmd, "function.unit"), "nm");
    cmd.apply_param_change("function.unit", "px");
    assert_eq!(param_value(&cmd, "function.unit"), "px");

    // An unrecognized function name must leave the current variant as-is.
    cmd.apply_param_change("function", "not-a-real-function");
    assert_eq!(param_value(&cmd, "function"), "Shrink");
}

#[test]
fn apply_param_change_dropdown_fields_ignore_unknown_values() {
    // The flip side of `apply_param_change_dropdown_fields_cycle_through_every_option`:
    // an unrecognized value must fall through to the match's `_ => s.field.clone()`
    // arm and leave the field untouched, for every dropdown field that has one.
    let dropdown_fields: &[(&str, &str, &str)] = &[
        ("Hessian", "mode", "Determinant"),
        ("ImageCache", "mode", "Store"),
        ("ImageMath", "operand", "Add"),
        ("IntensityTransformation", "mode", "Manual"),
        ("MorphologicalCommand", "op", "Erode"),
        ("MorphologicalCommand", "kernel_shape", "Ellipse"),
        ("ObjectMath", "operation", "Or"),
        ("RollingBall", "ball_type", "Paraboloid"),
        ("SaveImage", "source", "Instance Map"),
        ("StructureTensor", "mode", "Coherence"),
        (
            "AI UNet Segmentation",
            "output_mode",
            "Independent Channels",
        ),
    ];
    for &(name, field, known_value) in dropdown_fields {
        let mut cmd = default_command(id_of(name)).unwrap();
        cmd.apply_param_change(field, known_value);
        assert_eq!(
            param_value(&cmd, field),
            known_value,
            "{name}: {field} setup"
        );
        cmd.apply_param_change(field, "this-is-not-a-real-option");
        assert_eq!(
            param_value(&cmd, field),
            known_value,
            "{name}: unknown '{field}' value should be a no-op",
        );
    }
}

#[test]
fn threshold_add_group_item_clones_the_previous_entry_instead_of_resetting_to_default() {
    let mut cmd = default_command(id_of("Threshold")).unwrap();

    // First call on an empty list falls back to a fresh default entry.
    cmd.add_group_item("thresholds");
    cmd.apply_param_change("thresholds.0.min_threshold", "77");

    // A second call, with a non-empty list, clones the *last* entry rather
    // than pushing another default - so the new row inherits "77" too.
    cmd.add_group_item("thresholds");
    let params = cmd.to_parameters();
    let thresholds_param = params.iter().find(|p| p.name == "thresholds").unwrap();
    assert_eq!(thresholds_param.groups.len(), 2);
    let second_row_min = thresholds_param.groups[1]
        .iter()
        .find(|p| p.name == "min_threshold")
        .unwrap();
    assert_eq!(
        second_row_min.value, "77",
        "new row should clone the previous entry's values"
    );
}

#[test]
fn apply_param_change_threshold_entry_cycles_through_method_options() {
    let mut cmd = default_command(id_of("Threshold")).unwrap();
    cmd.add_group_item("thresholds");
    let methods = {
        let params = cmd.to_parameters();
        let nested = &params
            .iter()
            .find(|p| p.name == "thresholds")
            .unwrap()
            .groups[0];
        nested
            .iter()
            .find(|p| p.name == "method")
            .unwrap()
            .options
            .clone()
    };
    assert!(
        methods.len() > 5,
        "expected many threshold methods, got {methods:?}"
    );
    for method in &methods {
        cmd.apply_param_change("thresholds.0.method", method);
        let params = cmd.to_parameters();
        let nested = &params
            .iter()
            .find(|p| p.name == "thresholds")
            .unwrap()
            .groups[0];
        let value = &nested.iter().find(|p| p.name == "method").unwrap().value;
        assert_eq!(value, method);
    }
}

#[test]
fn apply_param_change_threshold_entry_ignores_malformed_or_out_of_range_paths() {
    // The `thresholds.{idx}.{field}` dotted path is parsed by hand (split on
    // '.', then `idx.parse::<usize>()`, then `Vec::get_mut(idx)`); malformed
    // or out-of-range paths must be silently ignored rather than panicking.
    let mut cmd = default_command(id_of("Threshold")).unwrap();
    cmd.add_group_item("thresholds");
    cmd.apply_param_change("thresholds.0.min_threshold", "5");
    assert_eq!(
        param_value_nested(&cmd, "thresholds", 0, "min_threshold"),
        "5"
    );

    // Out-of-range index.
    cmd.apply_param_change("thresholds.99.min_threshold", "999");
    // Unparsable index.
    cmd.apply_param_change("thresholds.not_a_number.min_threshold", "999");
    // Missing nested field name (no second '.' segment).
    cmd.apply_param_change("thresholds.0", "999");

    assert_eq!(
        param_value_nested(&cmd, "thresholds", 0, "min_threshold"),
        "5",
        "malformed/out-of-range threshold paths must not mutate any entry"
    );
}

#[test]
fn apply_param_change_rank_filter_switches_to_and_edits_a_tuple_variant() {
    // "Outliers" carries data (`Outliers(f32)`): switching to it constructs
    // the variant with its payload type's own default (`f32::default()`),
    // and the payload itself is then editable via the synthetic
    // "filter_type.0" field - the tuple-variant counterpart to a rich enum's
    // own named sibling fields.
    let mut cmd = default_command(id_of("RankFilter")).unwrap();
    cmd.apply_param_change("filter_type", "Median");
    assert_eq!(param_value(&cmd, "filter_type"), "Median");

    cmd.apply_param_change("filter_type", "Outliers");
    assert_eq!(param_value(&cmd, "filter_type"), "Outliers");
    assert_eq!(
        param_value(&cmd, "filter_type.0").parse::<f32>().unwrap(),
        0.0
    );

    cmd.apply_param_change("filter_type.0", "2.5");
    assert_eq!(
        param_value(&cmd, "filter_type.0").parse::<f32>().unwrap(),
        2.5
    );
    // Switching to a plain unit variant and back must not resurrect the old payload.
    cmd.apply_param_change("filter_type", "Median");
    cmd.apply_param_change("filter_type", "Outliers");
    assert_eq!(
        param_value(&cmd, "filter_type.0").parse::<f32>().unwrap(),
        0.0
    );
}

#[test]
fn apply_param_change_colocalization_multiplicity_switches_to_and_edits_multi_for() {
    // "Multi coloc only for selected" carries data
    // (`MultiFor(Vec<ObjectClass>)`): switching to it constructs the variant
    // with an empty class list (same mechanism as the RankFilter test above,
    // just for a `Vec<ObjectClass>` payload instead of a bare `f32`), and the
    // list is then editable via the synthetic "multiplicity.0" MultiObjClass
    // field, using the same comma-list/toggle: syntax every other
    // MultiObjClass field supports.
    let mut cmd = default_command(id_of("Colocalization")).unwrap();
    assert_eq!(param_value(&cmd, "multiplicity"), "No multi coloc (1:1)");

    cmd.apply_param_change("multiplicity", "Multi coloc only for selected");
    assert_eq!(
        param_value(&cmd, "multiplicity"),
        "Multi coloc only for selected"
    );
    assert_eq!(param_value(&cmd, "multiplicity.0"), "");

    cmd.apply_param_change("multiplicity.0", "1,2");
    assert_eq!(param_value(&cmd, "multiplicity.0"), "1,2");

    cmd.apply_param_change("multiplicity.0", "toggle:1");
    assert_eq!(param_value(&cmd, "multiplicity.0"), "2");

    // Switching away and back must not resurrect the old class list.
    cmd.apply_param_change("multiplicity", "Allow multi coloc");
    cmd.apply_param_change("multiplicity", "Multi coloc only for selected");
    assert_eq!(param_value(&cmd, "multiplicity.0"), "");
}

#[test]
fn apply_param_change_numeric_fields_ignore_unparsable_values() {
    // One numeric field per distinct parse target type (usize/i32/f32/f64)
    // to reach the `Err` side of `apply_param_change`'s `if let Ok(v) = ...`
    // guards, which a same-value round trip can never hit.
    let mut cmd = default_command(id_of("Blur")).unwrap(); // kernel_size: usize
    cmd.apply_param_change("kernel_size", "9");
    cmd.apply_param_change("kernel_size", "not_a_number");
    assert_eq!(param_value(&cmd, "kernel_size"), "9");

    let mut cmd = default_command(id_of("ConnectedComponents")).unwrap(); // min_size: i32
    cmd.apply_param_change("min_size", "15");
    cmd.apply_param_change("min_size", "not_a_number");
    assert_eq!(param_value(&cmd, "min_size"), "15");

    let mut cmd = default_command(id_of("DistanceTransform")).unwrap(); // threshold: f32
    cmd.apply_param_change("threshold", "0.3");
    cmd.apply_param_change("threshold", "not_a_number");
    assert_eq!(param_value(&cmd, "threshold"), "0.3");

    let mut cmd = default_command(id_of("MedianSubtract")).unwrap(); // radius: f64
    cmd.apply_param_change("radius", "12.5");
    cmd.apply_param_change("radius", "not_a_number");
    assert_eq!(param_value(&cmd, "radius"), "12.5");
}

#[test]
fn apply_param_change_text_and_path_fields_accept_arbitrary_strings() {
    let mut cmd = default_command(id_of("SaveImage")).unwrap(); // name: Text
    cmd.apply_param_change("name", "output_cell");
    assert_eq!(param_value(&cmd, "name"), "output_cell");

    let mut cmd = default_command(id_of("AI Cellpose Segmentation")).unwrap(); // model_path: FilePath
    cmd.apply_param_change("model_path", "/models/cellpose.pt");
    assert_eq!(param_value(&cmd, "model_path"), "/models/cellpose.pt");
}
