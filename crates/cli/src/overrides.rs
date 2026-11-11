//! # overrides
//!
//! `--set "Pipeline.step.setting=value"`: changes one setting of a step
//! for this run, without editing the project file. The step is its `id` in
//! the project file, the setting its name there (`minSize`) or as the
//! editor names it (`min_size`, a list entry's `thresholds.0.method`, a
//! module step's parameter key). A value is set as the editor sets it and
//! read back, so a value the setting doesn't take is an error instead of
//! being ignored. The run's copy of the project (`<job>.evaproj` next to
//! its results) records the values it ran with.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-11
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_app::modules::{set_module_parameter, step_setting, step_settings};
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_app::settings_of::{
    apply_command_param, measurement_choices, measurement_setting_text,
};
use evanalyzer_cfg::closest_name;
use evanalyzer_cfg::core_types::{InternalErrors, ModuleParamKey};
use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_settings::{PipelineStepSettings, StepContent};

/// Applies every `--set` to `project` - each a "Pipeline.step.setting=value"
/// - and returns a line per change ("pipeline “Nuclei”, step 2 (WATERSHED):
/// min_size 50 → 100"). Every override that doesn't apply is reported in
/// one error, none is applied then.
pub fn apply_overrides(
    project: &mut ProjectWithRuntime,
    overrides: &[String],
) -> Result<Vec<String>, InternalErrors> {
    if overrides.is_empty() {
        return Ok(Vec::new());
    }
    let unchanged = project.settings.clone();
    let mut lines = Vec::new();
    let mut problems = Vec::new();
    for text in overrides {
        match apply_override(project, text) {
            Ok(line) => lines.push(line),
            Err(problem) => problems.push(format!("--set \"{text}\": {problem}")),
        }
    }
    if !problems.is_empty() {
        project.settings = unchanged;
        return Err(InternalErrors::InvalidArgument(problems.join("\n")));
    }
    Ok(lines)
}

/// Applies one "Pipeline.step.setting=value".
fn apply_override(project: &mut ProjectWithRuntime, text: &str) -> Result<String, String> {
    let (path, value) = text
        .split_once('=')
        .ok_or("expected Pipeline.step.setting=value")?;
    let (path, value) = (path.trim(), value.trim());

    // The longest pipeline name the path starts with, followed by a dot -
    // a name may hold dots itself.
    let pipeline_index = project
        .pipelines
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            path.strip_prefix(p.name.as_str())
                .is_some_and(|rest| rest.starts_with('.'))
        })
        .max_by_key(|(_, p)| p.name.len())
        .map(|(i, _)| i)
        .ok_or_else(|| {
            let names: Vec<String> = project
                .pipelines
                .iter()
                .map(|p| format!("“{}”", p.name))
                .collect();
            format!(
                "no pipeline it starts with - the project has {}",
                if names.is_empty() {
                    "none".to_string()
                } else {
                    names.join(", ")
                }
            )
        })?;
    let pipeline_name = project.pipelines[pipeline_index].name.clone();
    let rest = &path[pipeline_name.len() + 1..];
    let (step_text, setting) = rest
        .split_once('.')
        .ok_or("expected Pipeline.step.setting=value")?;
    let step_id: u32 = step_text.parse().map_err(|_| {
        format!("`{step_text}` is not a step number (its `id` in the project file)")
    })?;

    let definitions = project.modules.clone();
    let choices = measurement_choices(project);
    let pipeline = &mut project.pipelines[pipeline_index];
    let Some(step) = pipeline.steps.iter_mut().find(|s| s.id.0 == step_id) else {
        let steps: Vec<String> = pipeline
            .steps
            .iter()
            .map(|s| format!("{} {}", s.id, step_kind(s)))
            .collect();
        return Err(format!(
            "pipeline “{pipeline_name}” has no step {step_id} - its steps (id and command): {}",
            steps.join(", ")
        ));
    };
    let place = format!(
        "pipeline “{pipeline_name}”, step {step_id} ({})",
        step_kind(step)
    );

    // The setting, by its path as the editor names it or as the project
    // file spells it (case and `_` aside: minSize = min_size).
    let settings = step_settings(&definitions, step);
    let wanted = normalized(setting);
    let Some((path, before, _)) = settings.iter().find(|(p, ..)| normalized(p) == wanted) else {
        let names: Vec<&str> = settings.iter().map(|(p, ..)| p.as_str()).collect();
        let suggestion = closest_name(&wanted, names.iter().copied())
            .map(|near| format!(" - did you mean `{near}`?"))
            .unwrap_or_default();
        return Err(format!("{place} has no setting `{setting}`{suggestion}"));
    };
    let (path, before) = (path.clone(), before.clone());
    let value = choice_for(&before, value);
    let value = match before.param_type {
        ParamType::Toggle => toggle_value(&value).ok_or_else(|| {
            format!(
                "{place}: `{value}` is no value for {path} - {}",
                what_it_takes(&before)
            )
        })?,
        _ => value,
    };
    let value = value.as_str();

    match &mut step.content {
        StepContent::Command(command) => {
            let text = measurement_setting_text(command, &path, value, &choices);
            apply_command_param(command, &path, &text);
        }
        StepContent::Module(instance) => {
            set_module_parameter(
                instance,
                &definitions,
                &ModuleParamKey::from(path.as_str()),
                value,
            )
            .map_err(|e| format!("{place}: {e}"))?;
        }
    }

    let after = step_setting(&definitions, step, &path)
        .map(|(p, _)| p.value)
        .unwrap_or_default();
    if !same_value(&after, value) && after == before.value {
        return Err(format!(
            "{place}: `{value}` is no value for {path} - {}",
            what_it_takes(&before)
        ));
    }
    Ok(format!("{place}: {path} {} → {after}", before.value))
}

/// What a step runs, for messages: its command's `type`, or its module.
fn step_kind(step: &PipelineStepSettings) -> String {
    match &step.content {
        StepContent::Command(command) => serde_json::to_value(command)
            .ok()
            .and_then(|v| v.get("type")?.as_str().map(str::to_string))
            .unwrap_or_else(|| "command".into()),
        StepContent::Module(instance) => format!("module {}", instance.key),
    }
}

/// A setting's path compared case- and `_`-blind, so the project file's
/// `minSize` finds the editor's `min_size`.
fn normalized(path: &str) -> String {
    path.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

/// For a setting with a choice of values, the choice `value` names - as
/// the editor shows it ("Distance Map") or as the project file spells it
/// ("DISTANCE_MAP", case, spaces, `_` and `-` aside); else `value`.
fn choice_for(setting: &ParameterDef, value: &str) -> String {
    let choices = matches!(
        setting.param_type,
        ParamType::Dropdown | ParamType::PixelUnits | ParamType::SizeUnits
    );
    let plain = |text: &str| -> String {
        text.chars()
            .filter(|c| !matches!(c, ' ' | '_' | '-'))
            .flat_map(char::to_lowercase)
            .collect()
    };
    if !choices || setting.options.iter().any(|o| o == value) {
        return value.to_string();
    }
    setting
        .options
        .iter()
        .find(|o| plain(o) == plain(value))
        .cloned()
        .unwrap_or_else(|| value.to_string())
}

/// An on/off value as the editor sets it ("true"/"false"); `None` for
/// anything else - checked before setting, since an on/off setting may
/// take any other text as off.
fn toggle_value(value: &str) -> Option<String> {
    match value.to_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some("true".into()),
        "false" | "no" | "off" | "0" => Some("false".into()),
        _ => None,
    }
}

/// Whether a setting now showing `shown` holds `wanted`: the same text
/// (case aside) or the same number ("100" = "100.0").
fn same_value(shown: &str, wanted: &str) -> bool {
    if shown.trim().eq_ignore_ascii_case(wanted.trim()) {
        return true;
    }
    match (shown.trim().parse::<f64>(), wanted.trim().parse::<f64>()) {
        (Ok(a), Ok(b)) => (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0),
        _ => false,
    }
}

/// What a setting takes, for the message about a value it didn't.
fn what_it_takes(setting: &ParameterDef) -> String {
    match setting.param_type {
        ParamType::Number | ParamType::Slider | ParamType::Spinner => {
            format!("it takes a number (it stays {})", setting.value)
        }
        ParamType::Toggle => "it takes true or false".into(),
        _ if !setting.options.is_empty() => {
            format!("it takes one of: {}", setting.options.join(", "))
        }
        _ => format!("it stays {}", setting.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{ImageAddress, PipelineId, StepId};
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    fn step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn pipeline(id: u32, name: &str, steps: Vec<PipelineStepSettings>) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(id),
            name: name.into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps,
        }
    }

    /// "Varroa detection" (a threshold with one entry as step 3, a
    /// watershed as step 9) and "Varroa detection.v2" (a find maxima as
    /// step 1) - a pipeline name another one starts with, holding a dot.
    fn project() -> ProjectWithRuntime {
        let mut threshold = PipelineCommand::default_for_key("threshold").unwrap();
        if let PipelineCommand::Threshold(settings) = &mut threshold {
            settings.thresholds.push(Default::default());
        }
        let mut project = ProjectWithRuntime::default();
        project.pipelines = vec![
            pipeline(
                1,
                "Varroa detection",
                vec![
                    step(3, threshold),
                    step(9, PipelineCommand::default_for_key("watershed").unwrap()),
                ],
            ),
            pipeline(
                2,
                "Varroa detection.v2",
                vec![step(
                    1,
                    PipelineCommand::default_for_key("find_maxima").unwrap(),
                )],
            ),
        ];
        project
    }

    fn set(project: &mut ProjectWithRuntime, overrides: &[&str]) -> Result<Vec<String>, String> {
        let overrides: Vec<String> = overrides.iter().map(|o| o.to_string()).collect();
        apply_overrides(project, &overrides).map_err(|e| e.to_string())
    }

    /// The value of setting `path` of step `step_id` of pipeline `index`.
    fn value(project: &ProjectWithRuntime, index: usize, step_id: u32, path: &str) -> String {
        let step = project.pipelines[index]
            .steps
            .iter()
            .find(|s| s.id.0 == step_id)
            .unwrap();
        step_setting(&project.modules, step, path).unwrap().0.value
    }

    /// The report's example: the setting by its project file name.
    #[test]
    fn a_setting_is_set_by_its_project_file_name() {
        let mut project = project();
        let lines = set(&mut project, &["Varroa detection.9.minObjectSize=100"]).unwrap();
        assert_eq!(
            lines,
            ["pipeline “Varroa detection”, step 9 (WATERSHED): min_object_size 0 → 100"]
        );
        assert_eq!(value(&project, 0, 9, "min_object_size"), "100");
    }

    /// The editor's names work too, a list entry's included, and dropdown
    /// values as the project file spells them.
    #[test]
    fn editor_names_list_entries_and_file_spelt_choices_are_set() {
        let mut project = project();
        set(
            &mut project,
            &[
                "Varroa detection.9.smoothing_sigma=1.5",
                "Varroa detection.9.seedSource=INTENSITY",
                "Varroa detection.3.thresholds.0.method=Otsu",
                "Varroa detection.3.thresholds.0.minThreshold=200",
            ],
        )
        .unwrap();
        assert_eq!(value(&project, 0, 9, "smoothing_sigma"), "1.5");
        assert_eq!(value(&project, 0, 9, "seed_source"), "Intensity");
        assert_eq!(value(&project, 0, 3, "thresholds.0.method"), "Otsu");
        assert_eq!(value(&project, 0, 3, "thresholds.0.min_threshold"), "200");
    }

    /// The longest pipeline name the path starts with wins: a dot in a
    /// name doesn't cut it short.
    #[test]
    fn a_pipeline_name_with_a_dot_is_found() {
        let mut project = project();
        set(&mut project, &["Varroa detection.v2.1.excludeAtEdges=yes"]).unwrap();
        assert_eq!(value(&project, 1, 1, "exclude_at_edges"), "true");
        // Also when it is on: "maybe" doesn't turn it off.
        let err = set(
            &mut project,
            &["Varroa detection.v2.1.excludeAtEdges=maybe"],
        )
        .unwrap_err();
        assert!(err.contains("it takes true or false"), "{err}");
        assert_eq!(value(&project, 1, 1, "exclude_at_edges"), "true");
    }

    /// Every override that doesn't apply is reported at once, with what
    /// would be right, and none is applied.
    #[test]
    fn every_problem_is_reported_and_nothing_is_applied() {
        let mut project = project();
        let err = set(
            &mut project,
            &[
                "Varroa detection.9.minObjectSize=100",
                "Bees.1.minSize=3",
                "Varroa detection.7.minSize=3",
                "Varroa detection.9.minObjectSiz=3",
                "Varroa detection.9.smoothingSigma=soft",
                "Varroa detection.9.seedSource=Random",
                "Varroa detection.v2.1.excludeAtEdges=maybe",
                "Varroa detection.9.minObjectSize",
            ],
        )
        .unwrap_err();
        for expected in [
            "--set \"Bees.1.minSize=3\": no pipeline it starts with - the project has \
             “Varroa detection”, “Varroa detection.v2”",
            "has no step 7 - its steps (id and command): 3 THRESHOLD, 9 WATERSHED",
            "has no setting `minObjectSiz` - did you mean `min_object_size`?",
            "`soft` is no value for smoothing_sigma - it takes a number (it stays 0)",
            "`Random` is no value for seed_source - it takes one of: Distance Map, Intensity",
            "`maybe` is no value for exclude_at_edges - it takes true or false",
            "--set \"Varroa detection.9.minObjectSize\": expected Pipeline.step.setting=value",
        ] {
            assert!(err.contains(expected), "missing {expected:?} in:\n{err}");
        }
        assert_eq!(err.lines().count(), 7, "{err}");
        assert_eq!(
            value(&project, 0, 9, "min_object_size"),
            "0",
            "nothing applied"
        );
    }

    /// A module step's parameter, by its key: set through the module, so
    /// the step runs with it; a value it doesn't take is an error.
    #[test]
    fn a_module_parameter_is_set_by_its_key() {
        use evanalyzer_cfg::settings::modules::{
            ModuleDefinition, ModuleInstance, ModuleParameterDef, ParamBinding, ParamValue,
        };
        let mut project = project();
        project.modules.push(ModuleDefinition {
            key: "SEGMENT".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(
                1,
                PipelineCommand::default_for_key("watershed").unwrap(),
            )],
            parameters: vec![ModuleParameterDef {
                key: ModuleParamKey::from("min_size"),
                label: "Min size".into(),
                description: String::new(),
                section: String::new(),
                default: ParamValue::Float(0.0),
                bindings: vec![ParamBinding {
                    step: StepId(1),
                    param: "min_object_size".into(),
                }],
            }],
            origin: None,
        });
        project.pipelines[0].steps.push(PipelineStepSettings {
            id: StepId(12),
            enabled: true,
            content: StepContent::Module(ModuleInstance {
                key: "SEGMENT".into(),
                values: Default::default(),
            }),
        });

        let lines = set(&mut project, &["Varroa detection.12.minSize=40"]).unwrap();
        assert_eq!(
            lines,
            ["pipeline “Varroa detection”, step 12 (module SEGMENT): min_size 0 → 40"]
        );
        assert_eq!(value(&project, 0, 12, "min_size"), "40");

        let err = set(&mut project, &["Varroa detection.12.minSize=big"]).unwrap_err();
        assert!(err.contains("`big` is no value for min_size"), "{err}");
        assert_eq!(value(&project, 0, 12, "min_size"), "40");
    }

    /// Setting the value a setting already has is fine.
    #[test]
    fn setting_the_current_value_is_no_error() {
        let mut project = project();
        let lines = set(&mut project, &["Varroa detection.9.minObjectSize=0"]).unwrap();
        assert_eq!(
            lines,
            ["pipeline “Varroa detection”, step 9 (WATERSHED): min_object_size 0 → 0"]
        );
    }
}
