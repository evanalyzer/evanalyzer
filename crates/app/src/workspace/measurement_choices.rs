//! # measurement_choices
//!
//! Measurement settings in the editor (`ParamType::Measurement`, e.g. an
//! "Exclude Image" rule's measurement): picked from the measurements the
//! project's steps make (each step's `measurement_outputs`), shown by label
//! ("quality_01 · Focus score (channel 0)"), stored by key.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::workspace::ProjectWithRuntime;
use crate::workspace::module_insights::resolved_commands;
use evanalyzer_cfg::core_types::measurement_label;
use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;

/// One measurement to pick.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementChoice {
    pub label: String,
    pub key: String,
}

/// The measurements the project's steps make - every pipeline, steps in
/// modules too - in the pipelines' order.
pub fn measurement_choices(project: &ProjectWithRuntime) -> Vec<MeasurementChoice> {
    let mut choices: Vec<MeasurementChoice> = Vec::new();
    for pipeline in &project.pipelines {
        for step in &pipeline.steps {
            for resolved in resolved_commands(step, &project.modules) {
                for info in evanalyzer_core::measurement_outputs(&resolved.command) {
                    if !choices.iter().any(|c| c.key == info.key) {
                        choices.push(MeasurementChoice {
                            label: measurement_label(&info),
                            key: info.key,
                        });
                    }
                }
            }
        }
    }
    choices
}

/// A step's settings as the editor shows them: each measurement setting a
/// dropdown of `choices`' labels, its value the chosen one's label. A key
/// no step measures any more stays as it is (an option of its own), so it
/// isn't lost unseen.
pub fn show_measurement_choices(parameters: &mut [ParameterDef], choices: &[MeasurementChoice]) {
    for parameter in parameters {
        for group in &mut parameter.groups {
            show_measurement_choices(group, choices);
        }
        if parameter.param_type != ParamType::Measurement {
            continue;
        }
        let mut options: Vec<String> = choices.iter().map(|c| c.label.clone()).collect();
        match choices.iter().find(|c| c.key == parameter.value) {
            Some(choice) => parameter.value = choice.label.clone(),
            None if !parameter.value.is_empty() => options.insert(0, parameter.value.clone()),
            None => {}
        }
        parameter.param_type = ParamType::Dropdown;
        parameter.options = options;
    }
}

/// What to store for setting `path` of `command` when the editor reports
/// `text`: for a measurement setting the key of the label picked, else
/// `text` as it is.
pub fn measurement_setting_text(
    command: &PipelineCommand,
    path: &str,
    text: &str,
    choices: &[MeasurementChoice],
) -> String {
    let parameters = command.to_parameters();
    let mut parts = path.splitn(3, '.');
    let parameter = match (parts.next(), parts.next(), parts.next()) {
        (Some(list), Some(index), Some(field)) => index.parse::<usize>().ok().and_then(|i| {
            parameters
                .iter()
                .find(|p| p.name == list)?
                .groups
                .get(i)?
                .iter()
                .find(|p| p.name == field)
                .cloned()
        }),
        _ => parameters.into_iter().find(|p| p.name == path),
    };
    match parameter {
        Some(p) if p.param_type == ParamType::Measurement => choices
            .iter()
            .find(|c| c.label == text)
            .map_or_else(|| text.to_string(), |c| c.key.clone()),
        _ => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::pipeline_settings::{
        PipelineSettings, PipelineStepSettings, StepContent,
    };

    fn command(key: &str) -> PipelineCommand {
        PipelineCommand::default_for_key(key).unwrap()
    }

    /// A pipeline with Image Quality (quality_01, channel 0) and Exclude
    /// Image with one rule.
    fn project() -> (ProjectWithRuntime, PipelineCommand) {
        let mut quality = command("measure_image_quality");
        *quality.measurement_id_mut().unwrap() = "quality_01".into();
        let mut exclude = command("exclude_image");
        let PipelineCommand::ExcludeImage(settings) = &mut exclude else {
            unreachable!()
        };
        settings.rules.push(Default::default());
        let mut project = ProjectWithRuntime::default();
        project.pipelines.push(PipelineSettings {
            id: evanalyzer_cfg::core_types::PipelineId(1),
            name: String::new(),
            description: None,
            image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
            enabled: true,
            steps: [quality, exclude.clone()]
                .into_iter()
                .map(|c| PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: StepContent::Command(c),
                })
                .collect(),
        });
        (project, exclude)
    }

    #[test]
    fn the_projects_measurements_by_label() {
        let (project, _) = project();
        let choices = measurement_choices(&project);
        assert_eq!(choices.len(), 5, "Image Quality's five values");
        assert_eq!(
            choices[0],
            MeasurementChoice {
                label: "quality_01 · Focus score (channel 0)".into(),
                key: "quality_01/focus_score".into(),
            }
        );
    }

    /// A rule's measurement: a dropdown by label; picking one stores its
    /// key; a key nobody measures stays visible.
    #[test]
    fn a_rule_picks_by_label_and_stores_the_key() {
        let (project, mut exclude) = project();
        let choices = measurement_choices(&project);
        let path = "rules.0.measurement";
        let label = "quality_01 · Noise σ (channel 0)";
        let key = measurement_setting_text(&exclude, path, label, &choices);
        assert_eq!(key, "quality_01/noise");
        exclude.apply_param_change(path, &key);

        let mut parameters = exclude.to_parameters();
        show_measurement_choices(&mut parameters, &choices);
        let rule = &parameters
            .iter()
            .find(|p| p.name == "rules")
            .unwrap()
            .groups[0];
        let measurement = rule.iter().find(|p| p.name == "measurement").unwrap();
        assert_eq!(measurement.param_type, ParamType::Dropdown);
        assert_eq!(measurement.value, label);
        assert_eq!(measurement.options.len(), 5);

        // Gone from the project: shown by its key.
        exclude.apply_param_change(path, "old_01/noise");
        let mut parameters = exclude.to_parameters();
        show_measurement_choices(&mut parameters, &choices);
        let rule = &parameters
            .iter()
            .find(|p| p.name == "rules")
            .unwrap()
            .groups[0];
        let measurement = rule.iter().find(|p| p.name == "measurement").unwrap();
        assert_eq!(measurement.value, "old_01/noise");
        assert_eq!(measurement.options[0], "old_01/noise");

        // Other settings: the text as it is.
        assert_eq!(
            measurement_setting_text(&exclude, "rules.0.value", "0.15", &choices),
            "0.15"
        );
    }
}
