//! # parameter_visibility
//!
//! `ParameterDef::default_value` / `advanced` / `option_advanced` as generated from
//! `#[cmdsmeta(visibility = ...)]`.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::{all_command_meta, default_command};

fn all_params(params: &[ParameterDef], out: &mut Vec<ParameterDef>) {
    for p in params {
        out.push(p.clone());
        for group in &p.groups {
            all_params(group, out);
        }
    }
}

#[test]
fn a_fresh_command_has_every_setting_at_its_default() {
    for meta in all_command_meta() {
        let command = default_command(meta.id).unwrap();
        let mut params = Vec::new();
        all_params(&command.to_parameters(), &mut params);
        for p in params {
            if p.default_value.is_empty() {
                continue; // unknown (groups, tuple-variant payloads)
            }
            assert_eq!(
                p.value, p.default_value,
                "{} / {}: value and default differ in a fresh command",
                meta.name, p.name
            );
        }
    }
}

#[test]
fn hidden_dropdown_options_are_not_offered() {
    let classify = all_command_meta()
        .into_iter()
        .find(|m| m.name == "ClassifyObjects")
        .unwrap();
    let params = default_command(classify.id).unwrap().to_parameters();
    let handling = params.iter().find(|p| p.name == "match_handling").unwrap();
    assert!(
        !handling
            .options
            .iter()
            .any(|o| o == "Remove class on match")
    );
    assert!(
        !handling
            .options
            .iter()
            .any(|o| o == "Remove class on mismatch")
    );
    assert!(handling.options.iter().any(|o| o == "Add class on match"));
}

#[test]
fn option_flags_match_the_options() {
    for meta in all_command_meta() {
        let mut params = Vec::new();
        all_params(
            &default_command(meta.id).unwrap().to_parameters(),
            &mut params,
        );
        for p in params {
            assert!(
                p.option_advanced.is_empty() || p.option_advanced.len() == p.options.len(),
                "{} / {}: {} option flags for {} options",
                meta.name,
                p.name,
                p.option_advanced.len(),
                p.options.len()
            );
        }
    }
}

fn basic_and_advanced(command_name: &str) -> (Vec<String>, Vec<String>) {
    let meta = all_command_meta()
        .into_iter()
        .find(|m| m.name == command_name)
        .unwrap_or_else(|| panic!("no command {command_name}"));
    let mut params = Vec::new();
    all_params(
        &default_command(meta.id).unwrap().to_parameters(),
        &mut params,
    );
    let (advanced, basic): (Vec<_>, Vec<_>) = params.into_iter().partition(|p| p.advanced);
    (
        basic.into_iter().map(|p| p.name).collect(),
        advanced.into_iter().map(|p| p.name).collect(),
    )
}

#[test]
fn classify_objects_shows_its_basic_settings() {
    let (basic, advanced) = basic_and_advanced("ClassifyObjects");
    assert_eq!(
        basic,
        [
            "input_classes",
            "match_handling",
            "output_class",
            "size_unit",
            "min_size",
            "max_size",
            "min_circularity",
            "max_circularity",
            "allow_edge_touching",
            "intensity_filters"
        ]
    );
    assert!(advanced.contains(&"origin_segmentation".to_string()));
    assert!(!advanced.contains(&"min_circularity".to_string()));
}

#[test]
fn the_color_filter_shows_only_the_hue() {
    let (basic, advanced) = basic_and_advanced("Color Filter");
    assert_eq!(basic, ["range.min_h", "range.max_h"]);
    assert_eq!(advanced.len(), 4);
}

/// The command named `name` as fresh, by its display name.
fn command_named(name: &str) -> evanalyzer_cfg::settings::pipeline_command::PipelineCommand {
    let meta = all_command_meta()
        .into_iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("no command {name}"));
    default_command(meta.id).unwrap()
}

fn param<'a>(params: &'a [ParameterDef], name: &str) -> Option<&'a ParameterDef> {
    params.iter().find(|p| p.name == name)
}

#[test]
fn grow_regions_memory_shows_its_memory_slot() {
    let mut command = command_named("Grow Regions");
    command.apply_param_change("growth_channel", "Memory");
    let params = command.to_parameters();
    let slot = param(&params, "growth_channel.0").expect("a memory slot picker");
    assert_eq!(slot.param_type, ParamType::ImageAddress);
    assert_eq!(slot.value, "memory:1");
    assert_eq!(slot.options, ["memory", "1", "memory"]);

    command.apply_param_change("growth_channel.0", "cache:3");
    let params = command.to_parameters();
    assert_eq!(param(&params, "growth_channel.0").unwrap().value, "cache:3");
    // Not a memory slot: ignored.
    command.apply_param_change("growth_channel.0", "channel:2");
    let params = command.to_parameters();
    assert_eq!(param(&params, "growth_channel.0").unwrap().value, "cache:3");
}

#[test]
fn threshold_value_source_memory_shows_its_memory_slot() {
    let mut command = command_named("Threshold");
    command.add_group_item("thresholds");
    command.apply_param_change("thresholds.0.value_source", "Memory");
    command.apply_param_change("thresholds.0.value_source.0", "memory:4");
    let params = command.to_parameters();
    let entry = &param(&params, "thresholds").unwrap().groups[0];
    let slot = param(entry, "value_source.0").expect("a memory slot picker");
    assert_eq!(slot.param_type, ParamType::ImageAddress);
    assert_eq!(slot.value, "memory:4");
}
