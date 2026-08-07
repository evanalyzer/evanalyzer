//! Coverage tests for the generated `pipeline_command` module
//! (`crates/cfg/src/modules/pipeline_command.rs`). That file is marked
//! `// @generated - do not edit by hand`, so tests live here instead of
//! inline, where they survive regeneration.
//!
//! The module is ~33 `PipelineCommand` variants, each with near-identical
//! `name`/`category`/`allowed_next`/`to_summary`/`to_parameters`/
//! `apply_param_change` match arms. Hand-writing one test per variant would
//! just be ~32 copies of the same shape, so these walk every variant via
//! `all_command_meta`/`default_command` in a loop instead.

use evanalyzer_cfg::settings::pipeline_command::{
    CommandCategory, PipelineCommand, all_command_meta, default_command,
};

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
    let mut cmd = default_command(27).expect("id 27 is Threshold");
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
    let expected: [(&str, &[CommandCategory]); 33] = [
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
        ("Threshold", &[Object]),
        ("TransformObjects", &[Classify]),
        ("AI UNet Segmentation", &[Object]),
        ("Voronoi", &[Classify]),
        ("Watershed", &[Measure]),
        ("WeightedDeviation", &[Segment, Preprocess]),
    ];
    for (id, (name, categories)) in expected.iter().enumerate() {
        let cmd = default_command(id as i32).unwrap();
        assert_eq!(
            cmd.name(),
            *name,
            "id {id} name mismatch (test table is out of sync)"
        );
        assert_eq!(
            cmd.allowed_next(),
            *categories,
            "id {id} ({name}): unexpected allowed_next()"
        );
    }
}

#[test]
fn to_summary_is_empty_for_variants_without_a_custom_summary() {
    for id in [2, 14, 24, 27, 31] {
        let cmd = default_command(id).unwrap();
        assert_eq!(cmd.to_summary(), "", "id {id} ({})", cmd.name());
    }
}

#[test]
fn to_summary_formats_connected_components_min_size() {
    // `min_size` is an `i32`; `{:.3}` precision formatting has no effect on
    // integers (only on floats), so the summary is deliberately unpadded.
    let mut cmd = default_command(6).unwrap();
    cmd.apply_param_change("min_size", "12");
    assert_eq!(cmd.to_summary(), "Min Size: 12");
}

#[test]
fn to_summary_formats_blur_kernel_size() {
    // `kernel_size` is a `usize`; `{:.3}` precision formatting has no effect
    // on integers (only on floats), so the summary is deliberately unpadded.
    let mut cmd = default_command(1).unwrap();
    cmd.apply_param_change("kernel_size", "5");
    assert_eq!(cmd.to_summary(), "Kernel size: 5");
}

#[test]
fn to_summary_formats_gaussian_blur_kernel_and_sigma() {
    let mut cmd = default_command(12).unwrap();
    cmd.apply_param_change("kernel_size", "7");
    cmd.apply_param_change("sigma", "1.5");
    assert_eq!(cmd.to_summary(), "Kernel Size: 7 · Sigma: 1.500");
}

#[test]
fn to_summary_formats_classify_objects_criteria() {
    let mut cmd = default_command(3).unwrap();
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
    let mut cmd = default_command(20).unwrap();
    cmd.apply_param_change("operation", "Xor");
    assert_eq!(cmd.to_summary(), "Operation: Xor");
}

#[test]
fn to_summary_formats_transform_objects_function() {
    let mut cmd = default_command(28).unwrap();
    cmd.apply_param_change("function", "Shrink");
    assert_eq!(cmd.to_summary(), "Function: Shrink");
}

#[test]
fn apply_param_change_dropdown_fields_cycle_through_every_option() {
    // Every dropdown/enum-valued top-level field, driven straight from
    // `to_parameters()`'s own `options` list so this doesn't drift from the
    // generated code. Exercises every match arm, not just whichever one the
    // default settings happen to start on.
    let dropdown_fields: &[(i32, &str)] = &[
        (13, "mode"),         // Hessian
        (14, "mode"),         // ImageCache
        (15, "operand"),      // ImageMath
        (16, "mode"),         // IntensityTransformation
        (19, "op"),           // MorphologicalCommand
        (19, "kernel_shape"), // MorphologicalCommand
        // RankFilter's `filter_type` is deliberately excluded here: its
        // options include "Outliers", a data-carrying arm that
        // `apply_param_change` can't reconstruct from a bare string (see
        // `apply_param_change_rank_filter_ignores_unknown_filter_type_value`
        // below), so cycling through every option would fail here.
        (20, "operation"),     // ObjectMath
        (23, "ball_type"),     // RollingBall
        (24, "source"),        // SaveImage
        (26, "mode"),          // StructureTensor
        (3, "match_handling"), // ClassifyObjects
        (28, "function"),      // TransformObjects
        (29, "output_mode"),   // UNet
    ];
    for &(id, field) in dropdown_fields {
        let mut cmd = default_command(id).unwrap();
        let options = cmd
            .to_parameters()
            .into_iter()
            .find(|p| p.name == field)
            .unwrap_or_else(|| panic!("id {id}: missing parameter '{field}'"))
            .options;
        assert!(
            !options.is_empty(),
            "id {id}: '{field}' has no options to cycle through"
        );
        for option in &options {
            cmd.apply_param_change(field, option);
            assert_eq!(
                param_value(&cmd, field),
                *option,
                "id {id} ({}): setting '{field}' to {option:?} didn't stick",
                cmd.name()
            );
        }
    }
}

#[test]
fn apply_param_change_unit_fields_toggle_both_variants() {
    // SizeUnits fields.
    for (id, field) in [
        (3, "size_unit"),  // ClassifyObjects
        (4, "size_unit"),  // Colocalization
        (20, "size_unit"), // ObjectMath
        (30, "unit"),      // Voronoi
    ] {
        let mut cmd = default_command(id).unwrap();
        cmd.apply_param_change(field, "nm");
        assert_eq!(param_value(&cmd, field), "nm", "id {id}: {field}=nm");
        cmd.apply_param_change(field, "px");
        assert_eq!(param_value(&cmd, field), "px", "id {id}: {field}=px");
    }

    // PixelUnits, nested inside a Threshold group entry.
    let mut cmd = default_command(27).unwrap();
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
    let obj_class_fields: &[(i32, &str)] = &[
        (3, "output_class"),                // ClassifyObjects
        (3, "overlapping_with"),            // ClassifyObjects
        (4, "class_for_overlapping_areas"), // Colocalization
        (20, "input_class"),                // ObjectMath
        (20, "other_class"),                // ObjectMath
        (20, "output_class"),               // ObjectMath
        (28, "input_class"),                // TransformObjects
        (28, "output_class"),               // TransformObjects
        (30, "centers"),                    // Voronoi
        (30, "mask"),                       // Voronoi
        (30, "output_class"),               // Voronoi
    ];
    for &(id, field) in obj_class_fields {
        let mut cmd = default_command(id).unwrap();
        cmd.apply_param_change(field, "7");
        assert_eq!(param_value(&cmd, field), "7", "id {id}: {field}=7");
        cmd.apply_param_change(field, "-1");
        assert_eq!(
            param_value(&cmd, field),
            "-1",
            "id {id}: {field}=-1 (Unset)"
        );
        // A value that doesn't parse as u32 must leave the field untouched.
        cmd.apply_param_change(field, "-1");
        cmd.apply_param_change(field, "not_a_number");
        assert_eq!(
            param_value(&cmd, field),
            "-1",
            "id {id}: {field} garbage input should be a no-op"
        );
    }
}

#[test]
fn apply_param_change_multi_obj_class_fields_support_comma_lists_and_toggle() {
    let multi_obj_class_fields: &[(i32, &str)] = &[
        (3, "input_classes"),    // ClassifyObjects
        (4, "classes_to_coloc"), // Colocalization
        // Colocalization.filter_classes is `#[cmdsmeta(visible = false)]`, so it
        // never reaches `to_parameters()` - not tested here.
        (4, "exclude_classes"),        // Colocalization
        (20, "other_filter_classes"),  // ObjectMath
        (30, "center_filter_classes"), // Voronoi
        (30, "mask_filter_classes"),   // Voronoi
    ];
    for &(id, field) in multi_obj_class_fields {
        let mut cmd = default_command(id).unwrap();

        // Comma-separated list, including a blank and a non-numeric entry
        // that must be silently dropped.
        cmd.apply_param_change(field, "1,,x,2,3");
        assert_eq!(
            param_value(&cmd, field),
            "1,2,3",
            "id {id}: {field} comma list"
        );

        // `toggle:` removes an id already present...
        cmd.apply_param_change(field, "toggle:2");
        assert_eq!(
            param_value(&cmd, field),
            "1,3",
            "id {id}: {field} toggle off"
        );

        // ...and adds one that's absent.
        cmd.apply_param_change(field, "toggle:9");
        assert_eq!(
            param_value(&cmd, field),
            "1,3,9",
            "id {id}: {field} toggle on"
        );
    }
}

#[test]
fn apply_param_change_transform_objects_switches_through_every_function_kind() {
    let mut cmd = default_command(28).unwrap(); // TransformObjects

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
    let dropdown_fields: &[(i32, &str, &str)] = &[
        (13, "mode", "Determinant"),                 // Hessian
        (14, "mode", "Store"),                       // ImageCache
        (15, "operand", "Add"),                      // ImageMath
        (16, "mode", "Manual"),                      // IntensityTransformation
        (19, "op", "Erode"),                         // MorphologicalCommand
        (19, "kernel_shape", "Ellipse"),             // MorphologicalCommand
        (20, "operation", "Or"),                     // ObjectMath
        (23, "ball_type", "Paraboloid"),             // RollingBall
        (24, "source", "Instance Map"),              // SaveImage
        (26, "mode", "Coherence"),                   // StructureTensor
        (29, "output_mode", "Independent Channels"), // UNet
    ];
    for &(id, field, known_value) in dropdown_fields {
        let mut cmd = default_command(id).unwrap();
        cmd.apply_param_change(field, known_value);
        assert_eq!(
            param_value(&cmd, field),
            known_value,
            "id {id}: {field} setup"
        );
        cmd.apply_param_change(field, "this-is-not-a-real-option");
        assert_eq!(
            param_value(&cmd, field),
            known_value,
            "id {id} ({}): unknown '{field}' value should be a no-op",
            cmd.name()
        );
    }
}

#[test]
fn threshold_add_group_item_clones_the_previous_entry_instead_of_resetting_to_default() {
    let mut cmd = default_command(27).unwrap(); // Threshold

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
    let mut cmd = default_command(27).unwrap(); // Threshold
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
    let mut cmd = default_command(27).unwrap(); // Threshold
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
    let mut cmd = default_command(22).unwrap(); // RankFilter
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
    let mut cmd = default_command(4).unwrap(); // Colocalization
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
    let mut cmd = default_command(1).unwrap(); // Blur.kernel_size: usize
    cmd.apply_param_change("kernel_size", "9");
    cmd.apply_param_change("kernel_size", "not_a_number");
    assert_eq!(param_value(&cmd, "kernel_size"), "9");

    let mut cmd = default_command(6).unwrap(); // ConnectedComponents.min_size: i32
    cmd.apply_param_change("min_size", "15");
    cmd.apply_param_change("min_size", "not_a_number");
    assert_eq!(param_value(&cmd, "min_size"), "15");

    let mut cmd = default_command(7).unwrap(); // DistanceTransform.threshold: f32
    cmd.apply_param_change("threshold", "0.3");
    cmd.apply_param_change("threshold", "not_a_number");
    assert_eq!(param_value(&cmd, "threshold"), "0.3");

    let mut cmd = default_command(18).unwrap(); // MedianSubtract.radius: f64
    cmd.apply_param_change("radius", "12.5");
    cmd.apply_param_change("radius", "not_a_number");
    assert_eq!(param_value(&cmd, "radius"), "12.5");
}

#[test]
fn apply_param_change_text_and_path_fields_accept_arbitrary_strings() {
    let mut cmd = default_command(24).unwrap(); // SaveImage.name: Text
    cmd.apply_param_change("name", "output_cell");
    assert_eq!(param_value(&cmd, "name"), "output_cell");

    let mut cmd = default_command(2).unwrap(); // Cellpose.model_path: FilePath
    cmd.apply_param_change("model_path", "/models/cellpose.pt");
    assert_eq!(param_value(&cmd, "model_path"), "/models/cellpose.pt");
}
