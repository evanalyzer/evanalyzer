//! `ParameterDef::default_value` / `advanced` / `option_advanced` as generated from
//! `#[cmdsmeta(visibility = ...)]`.

use evanalyzer_cfg::settings::parameter_def::ParameterDef;
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
            "min_area",
            "max_area",
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
    let (basic, advanced) = basic_and_advanced("ColorFilterCommand");
    assert_eq!(basic, ["range.min_h", "range.max_h"]);
    assert_eq!(advanced.len(), 4);
}
