//! Expanding command groups into plain commands before a job runs.
//!
//! A pipeline step is a command or an instance of a command group (see
//! `evanalyzer_cfg::settings::command_groups`). The executor only knows
//! commands, so the job generator replaces each group instance by its
//! definition's commands, with the instance's parameter values written into
//! the inner parameters they're bound to.

use evanalyzer_cfg::{
    core_types::{GroupParamId, InternalErrors, StepId},
    settings::{
        command_groups::{GroupDefinition, GroupInstance, ParamValue},
        parameter_def::ParameterDef,
        pipeline_command::PipelineCommand,
        pipeline_settings::{PipelineStepSettings, StepContent},
    },
};

/// How deep groups may nest - deeper is almost certainly a mistake.
const MAX_GROUP_DEPTH: usize = 8;

/// The commands `step` runs, in order, each with the id a breakpoint
/// targets: a command step is itself; a group is its definition's enabled
/// commands with the instance's values applied - the last one carries the
/// group step's id (a breakpoint on a group stops after the whole group),
/// the others [`StepId::UNASSIGNED`] (inner steps can't be targeted).
pub fn expand_step(
    step: &PipelineStepSettings,
    definitions: &[GroupDefinition],
) -> Result<Vec<(StepId, PipelineCommand)>, InternalErrors> {
    match &step.content {
        StepContent::Command(command) => Ok(vec![(step.id, command.clone())]),
        StepContent::Group(instance) => {
            let commands = expand_group(instance, definitions, &mut Vec::new())?;
            let last = commands.len().saturating_sub(1);
            Ok(commands
                .into_iter()
                .enumerate()
                .map(|(i, command)| {
                    let id = if i == last {
                        step.id
                    } else {
                        StepId::UNASSIGNED
                    };
                    (id, command)
                })
                .collect())
        }
    }
}

/// The commands of `instance`'s definition with its values applied.
/// `open` holds the keys of the groups being expanded around this one.
fn expand_group(
    instance: &GroupInstance,
    definitions: &[GroupDefinition],
    open: &mut Vec<String>,
) -> Result<Vec<PipelineCommand>, InternalErrors> {
    let definition = definitions
        .iter()
        .find(|d| d.key == instance.key)
        .ok_or_else(|| group_error(&instance.key, "isn't defined in this project".into()))?;
    if open.contains(&instance.key) {
        return Err(group_error(&instance.key, "contains itself".into()));
    }
    if open.len() >= MAX_GROUP_DEPTH {
        return Err(group_error(
            &instance.key,
            format!("is nested deeper than {MAX_GROUP_DEPTH} groups"),
        ));
    }

    // The steps with this instance's values written into what they're
    // bound to - the instance's value, else the definition's default.
    let mut steps = definition.steps.clone();
    for parameter in &definition.parameters {
        let value = instance
            .values
            .get(&parameter.id)
            .unwrap_or(&parameter.default);
        for binding in &parameter.bindings {
            let step = steps
                .iter_mut()
                .find(|s| s.id == binding.step)
                .ok_or_else(|| {
                    group_error(
                        &definition.key,
                        format!(
                            "parameter \"{}\" is bound to step {}, which the group doesn't have",
                            parameter.label, binding.step
                        ),
                    )
                })?;
            set_bound_value(step, &binding.param, value).map_err(|why| {
                group_error(
                    &definition.key,
                    format!(
                        "parameter \"{}\" can't set {} of step {}: {why}",
                        parameter.label, binding.param, binding.step
                    ),
                )
            })?;
        }
    }

    open.push(instance.key.clone());
    let mut commands = Vec::new();
    for step in steps.iter().filter(|s| s.enabled) {
        match &step.content {
            StepContent::Command(command) => commands.push(command.clone()),
            StepContent::Group(inner) => commands.extend(expand_group(inner, definitions, open)?),
        }
    }
    open.pop();
    Ok(commands)
}

/// Writes `value` into `param` of `step`: a command's parameter (checked
/// to have taken it - `apply_param_change` ignores what it can't use), or
/// for a nested group, the value of its group parameter `param` (its id).
fn set_bound_value(
    step: &mut PipelineStepSettings,
    param: &str,
    value: &ParamValue,
) -> Result<(), String> {
    match &mut step.content {
        StepContent::Command(command) => {
            let text = value_text(value);
            if find_param(&command.to_parameters(), param).is_none() {
                return Err("the command has no such parameter".into());
            }
            command.apply_param_change(param, &text);
            let now = find_param(&command.to_parameters(), param)
                .map(|p| p.value.clone())
                .unwrap_or_default();
            if same_value(&now, &text) {
                Ok(())
            } else {
                Err(format!("it doesn't accept {text:?}"))
            }
        }
        StepContent::Group(inner) => {
            let id = param
                .parse()
                .map(GroupParamId)
                .map_err(|_| "a group's parameters are bound by their id".to_string())?;
            inner.values.insert(id, value.clone());
            Ok(())
        }
    }
}

/// `value` in the text form `apply_param_change` takes: numbers and words as
/// written, a list comma separated (e.g. classes "1,3").
fn value_text(value: &ParamValue) -> String {
    match value {
        ParamValue::Bool(b) => b.to_string(),
        ParamValue::Int(i) => i.to_string(),
        ParamValue::Float(f) => f.to_string(),
        ParamValue::Text(t) => t.clone(),
        ParamValue::List(items) => items.iter().map(value_text).collect::<Vec<_>>().join(","),
    }
}

/// The parameter at `path` - a top-level name ("class_out", also a variant's
/// field like "function.factor"), or a list entry's field
/// ("thresholds.2.min").
fn find_param<'a>(params: &'a [ParameterDef], path: &str) -> Option<&'a ParameterDef> {
    if let Some(p) = params.iter().find(|p| p.name == path) {
        return Some(p);
    }
    let mut parts = path.splitn(3, '.');
    let (list, index, field) = (parts.next()?, parts.next()?, parts.next()?);
    let index: usize = index.parse().ok()?;
    let entry = params.iter().find(|p| p.name == list)?.groups.get(index)?;
    find_param(entry, field)
}

/// Whether two parameter values mean the same - numbers by value ("4" and
/// "4.0"), everything else as written.
fn same_value(a: &str, b: &str) -> bool {
    match (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a.trim() == b.trim(),
    }
}

fn group_error(key: &str, what: String) -> InternalErrors {
    InternalErrors::InvalidArgument(format!("Command group {key}: {what}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectClass;
    use evanalyzer_cfg::settings::command_groups::{GroupParameterDef, ParamBinding, ParamKind};
    use evanalyzer_cfg::settings::pipeline_command_settings::{BlurSettings, VoronoiSettings};
    use std::collections::BTreeMap;

    fn command_step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn group_step(id: u32, key: &str, values: &[(u32, ParamValue)]) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Group(GroupInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (GroupParamId(*k), v.clone()))
                    .collect::<BTreeMap<_, _>>(),
            }),
        }
    }

    fn parameter(id: u32, default: ParamValue, bindings: &[(u32, &str)]) -> GroupParameterDef {
        GroupParameterDef {
            id: GroupParamId(id),
            label: format!("Parameter {id}"),
            description: String::new(),
            default,
            kind: ParamKind::Exposed,
            bindings: bindings
                .iter()
                .map(|(step, param)| ParamBinding {
                    step: StepId(*step),
                    param: (*param).into(),
                })
                .collect(),
        }
    }

    fn definition(
        key: &str,
        steps: Vec<PipelineStepSettings>,
        parameters: Vec<GroupParameterDef>,
    ) -> GroupDefinition {
        GroupDefinition {
            key: key.into(),
            meta: Default::default(),
            steps,
            parameters,
            origin: None,
        }
    }

    /// Blur, then Voronoi - "Cell class" set in both places.
    fn cell_reconstruction() -> GroupDefinition {
        definition(
            "CELL_RECONSTRUCTION",
            vec![
                command_step(1, PipelineCommand::Blur(BlurSettings::default())),
                command_step(2, PipelineCommand::Voronoi(VoronoiSettings::default())),
            ],
            vec![
                parameter(1, ParamValue::Int(3), &[(1, "kernel_size")]),
                parameter(
                    2,
                    ParamValue::Int(1),
                    &[(2, "centers"), (2, "output_class")],
                ),
            ],
        )
    }

    fn voronoi(command: &PipelineCommand) -> &VoronoiSettings {
        let PipelineCommand::Voronoi(v) = command else {
            panic!("expected Voronoi");
        };
        v
    }

    #[test]
    fn a_command_step_is_just_itself() {
        let step = command_step(5, PipelineCommand::Blur(BlurSettings::default()));
        let commands = expand_step(&step, &[]).unwrap();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].0, StepId(5));
    }

    #[test]
    fn a_group_runs_its_commands_with_the_instances_values_and_defaults() {
        let step = group_step(9, "CELL_RECONSTRUCTION", &[(2, ParamValue::Int(4))]);
        let commands = expand_step(&step, &[cell_reconstruction()]).unwrap();

        assert_eq!(commands.len(), 2);
        let PipelineCommand::Blur(blur) = &commands[0].1 else {
            panic!("expected Blur");
        };
        assert_eq!(blur.kernel_size, 3, "the definition's default");
        // One group parameter, two inner parameters - always the same value.
        assert_eq!(voronoi(&commands[1].1).centers, ObjectClass::Global(4));
        assert_eq!(voronoi(&commands[1].1).output_class, ObjectClass::Global(4));
        // A breakpoint on the group stops after it.
        assert_eq!(commands[0].0, StepId::UNASSIGNED);
        assert_eq!(commands[1].0, StepId(9));
    }

    #[test]
    fn disabled_inner_steps_are_left_out() {
        let mut group = cell_reconstruction();
        group.steps[0].enabled = false;
        let commands = expand_step(&group_step(9, "CELL_RECONSTRUCTION", &[]), &[group]).unwrap();
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0].1, PipelineCommand::Voronoi(_)));
    }

    #[test]
    fn groups_nest_and_an_outer_parameter_can_set_an_inner_groups_parameter() {
        let outer = definition(
            "OUTER",
            vec![
                group_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            // Sets the inner group's "Cell class" (its parameter 2).
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let commands = expand_step(
            &group_step(4, "OUTER", &[]),
            &[cell_reconstruction(), outer],
        )
        .unwrap();
        assert_eq!(commands.len(), 3);
        assert_eq!(voronoi(&commands[1].1).centers, ObjectClass::Global(7));
        assert_eq!(commands[2].0, StepId(4));
    }

    #[test]
    fn mistakes_are_reported_with_the_group_and_what_is_wrong() {
        let error = |definitions: &[GroupDefinition], step: PipelineStepSettings| {
            expand_step(&step, definitions).unwrap_err().to_string()
        };

        let err = error(&[], group_step(1, "MISSING", &[]));
        assert!(
            err.contains("MISSING") && err.contains("isn't defined"),
            "{err}"
        );

        let looping = definition("LOOP", vec![group_step(1, "LOOP", &[])], vec![]);
        let err = error(&[looping], group_step(1, "LOOP", &[]));
        assert!(err.contains("contains itself"), "{err}");

        let mut unknown_step = cell_reconstruction();
        unknown_step.parameters[0].bindings[0].step = StepId(99);
        let err = error(&[unknown_step], group_step(1, "CELL_RECONSTRUCTION", &[]));
        assert!(err.contains("step 99"), "{err}");

        let mut unknown_param = cell_reconstruction();
        unknown_param.parameters[0].bindings[0].param = "no_such".into();
        let err = error(&[unknown_param], group_step(1, "CELL_RECONSTRUCTION", &[]));
        assert!(err.contains("no such parameter"), "{err}");

        // A value the parameter can't take is an error, not silently ignored.
        let bad_value = group_step(
            1,
            "CELL_RECONSTRUCTION",
            &[(1, ParamValue::Text("big".into()))],
        );
        let err = error(&[cell_reconstruction()], bad_value);
        assert!(
            err.contains("doesn't accept") && err.contains("big"),
            "{err}"
        );
    }

    #[test]
    fn values_become_the_text_commands_take() {
        assert_eq!(value_text(&ParamValue::Bool(true)), "true");
        assert_eq!(value_text(&ParamValue::Float(4.5)), "4.5");
        assert_eq!(
            value_text(&ParamValue::List(vec![
                ParamValue::Int(1),
                ParamValue::Int(3)
            ])),
            "1,3"
        );
        assert!(same_value("4", "4.0"));
        assert!(!same_value("nm", "px"));
    }
}
