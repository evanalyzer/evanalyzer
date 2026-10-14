//! Expanding command groups into plain commands before a job runs.
//!
//! A pipeline step is a command or an instance of a command group (see
//! `evanalyzer_cfg::settings::command_groups`). The executor only knows
//! commands, so the job generator replaces each group instance by its
//! definition's commands, with the instance's parameter values written into
//! the inner parameters they're bound to.

use crate::pipeline::scope::{Scope, ScopeId};
use evanalyzer_cfg::{
    core_types::{GroupParamId, InternalErrors, ObjectClass, StepId},
    settings::{
        command_groups::{GroupDefinition, GroupInstance, ParamValue},
        parameter_def::ParameterDef,
        pipeline_command::{CommandCategory, PipelineCommand},
        pipeline_settings::{PipelineStepSettings, StepContent},
    },
};

/// How deep groups may nest - deeper is almost certainly a mistake.
const MAX_GROUP_DEPTH: usize = 8;

/// One command a pipeline step runs.
#[derive(Debug, Clone)]
pub struct ExpandedCommand {
    /// What a breakpoint targets - see [`expand_step`].
    pub step_id: StepId,
    /// The scope the command runs in: its group instance's, else `Global`.
    pub scope: Scope,
    pub command: PipelineCommand,
}

/// Hands out a new [`ScopeId`] per group instance - one counter per job, so
/// no two instances share a scope.
#[derive(Debug, Default)]
pub struct ScopeCounter(u32);

impl ScopeCounter {
    fn next(&mut self) -> Scope {
        self.0 += 1;
        Scope::Scoped(ScopeId(self.0))
    }
}

/// The commands `step` runs, in order, each with the id a breakpoint
/// targets and the scope it runs in: a command step is itself, in the
/// global scope; a group is its definition's enabled commands with the
/// instance's values applied, in a new scope from `scopes` (a nested group
/// in one of its own) - the last one carries the group step's id (a
/// breakpoint on a group stops after the whole group), the others
/// [`StepId::UNASSIGNED`] (inner steps can't be targeted).
pub fn expand_step(
    step: &PipelineStepSettings,
    definitions: &[GroupDefinition],
    scopes: &mut ScopeCounter,
) -> Result<Vec<ExpandedCommand>, InternalErrors> {
    match &step.content {
        StepContent::Command(command) => {
            if let Some(class) = command.local_classes().first() {
                return Err(InternalErrors::InvalidArgument(format!(
                    "uses the group class {}, which only exists inside a command group",
                    class.to_text()
                )));
            }
            Ok(vec![ExpandedCommand {
                step_id: step.id,
                scope: Scope::Global,
                command: command.clone(),
            }])
        }
        StepContent::Group(instance) => {
            let commands = expand_group(instance, definitions, scopes, &mut Vec::new())?;
            let last = commands.len().saturating_sub(1);
            Ok(commands
                .into_iter()
                .enumerate()
                .map(|(i, (scope, command))| ExpandedCommand {
                    step_id: if i == last {
                        step.id
                    } else {
                        StepId::UNASSIGNED
                    },
                    scope,
                    command,
                })
                .collect())
        }
    }
}

/// The commands of `instance`'s definition with its values applied, each
/// with the scope it runs in - the instance's own, new one from `scopes`.
/// `open` holds the keys of the groups being expanded around this one.
fn expand_group(
    instance: &GroupInstance,
    definitions: &[GroupDefinition],
    scopes: &mut ScopeCounter,
    open: &mut Vec<String>,
) -> Result<Vec<(Scope, PipelineCommand)>, InternalErrors> {
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

    let scope = scopes.next();
    open.push(instance.key.clone());
    let mut commands = Vec::new();
    for step in steps.iter().filter(|s| s.enabled) {
        match &step.content {
            StepContent::Command(command) => {
                let undeclared = command
                    .local_classes()
                    .into_iter()
                    .find(|class| !definition.classes.iter().any(|c| c.local() == *class));
                if let Some(class) = undeclared {
                    return Err(group_error(
                        &definition.key,
                        format!(
                            "step {} uses the group class {}, which the group doesn't define",
                            step.id,
                            class.to_text()
                        ),
                    ));
                }
                if matches!(command.category(), CommandCategory::Dataset) {
                    if let Some(class) = command.local_classes().first() {
                        return Err(group_error(
                            &definition.key,
                            format!(
                                "step {} is a dataset step using the group class {} - dataset \
                                 steps run on stored objects, which only have project classes",
                                step.id,
                                class.to_text()
                            ),
                        ));
                    }
                }
                commands.push((scope, command.clone()))
            }
            StepContent::Group(inner) => {
                commands.extend(expand_group(inner, definitions, scopes, open)?)
            }
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
            // The inner group would read it as one of its own local classes.
            if let Some(local) = local_class_in(value) {
                return Err(format!(
                    "the group class {} can't be passed into a nested group",
                    local.to_text()
                ));
            }
            inner.values.insert(id, value.clone());
            Ok(())
        }
    }
}

/// The first local class (`"L1"`) in `value`, if any.
fn local_class_in(value: &ParamValue) -> Option<ObjectClass> {
    match value {
        ParamValue::Text(text) => {
            ObjectClass::from_text(text).filter(|c| matches!(c, ObjectClass::Local(_)))
        }
        ParamValue::List(items) => items.iter().find_map(local_class_in),
        _ => None,
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
    use evanalyzer_cfg::core_types::SegmentationClass;
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::command_groups::{GroupParameterDef, ParamBinding};
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        BlurSettings, ExtractClassMappingSettings, ExtractObjectsSettings, SimpleTrackerSettings,
        VoronoiSettings,
    };
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
            classes: vec![],
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

    /// Expands `step` with a fresh scope counter, like one job does.
    fn expand(
        step: &PipelineStepSettings,
        definitions: &[GroupDefinition],
    ) -> Result<Vec<ExpandedCommand>, InternalErrors> {
        expand_step(step, definitions, &mut ScopeCounter::default())
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
        let commands = expand(&step, &[]).unwrap();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].step_id, StepId(5));
    }

    #[test]
    fn a_group_runs_its_commands_with_the_instances_values_and_defaults() {
        let step = group_step(9, "CELL_RECONSTRUCTION", &[(2, ParamValue::Int(4))]);
        let commands = expand(&step, &[cell_reconstruction()]).unwrap();

        assert_eq!(commands.len(), 2);
        let PipelineCommand::Blur(blur) = &commands[0].command else {
            panic!("expected Blur");
        };
        assert_eq!(blur.kernel_size, 3, "the definition's default");
        // One group parameter, two inner parameters - always the same value.
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(4)
        );
        assert_eq!(
            voronoi(&commands[1].command).output_class,
            ObjectClass::Global(4)
        );
        // A breakpoint on the group stops after it.
        assert_eq!(commands[0].step_id, StepId::UNASSIGNED);
        assert_eq!(commands[1].step_id, StepId(9));
    }

    #[test]
    fn disabled_inner_steps_are_left_out() {
        let mut group = cell_reconstruction();
        group.steps[0].enabled = false;
        let commands = expand(&group_step(9, "CELL_RECONSTRUCTION", &[]), &[group]).unwrap();
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0].command, PipelineCommand::Voronoi(_)));
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
        let commands = expand(
            &group_step(4, "OUTER", &[]),
            &[cell_reconstruction(), outer],
        )
        .unwrap();
        assert_eq!(commands.len(), 3);
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(7)
        );
        assert_eq!(commands[2].step_id, StepId(4));
    }

    #[test]
    fn mistakes_are_reported_with_the_group_and_what_is_wrong() {
        let error = |definitions: &[GroupDefinition], step: PipelineStepSettings| {
            expand(&step, definitions).unwrap_err().to_string()
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

    fn local_class(id: u32) -> Class {
        Class {
            id,
            name: format!("Local {id}"),
            ..Default::default()
        }
    }

    fn voronoi_with_center(class: ObjectClass) -> PipelineCommand {
        PipelineCommand::Voronoi(VoronoiSettings {
            centers: class,
            ..Default::default()
        })
    }

    #[test]
    fn a_local_class_outside_a_group_is_refused() {
        let step = command_step(4, voronoi_with_center(ObjectClass::Local(1)));
        let error = expand(&step, &[]).unwrap_err().to_string();
        assert!(error.contains("group class L1"), "{error}");
    }

    #[test]
    fn extracting_into_a_local_class_outside_a_group_is_refused() {
        let extract = PipelineCommand::ExtractObjects(ExtractObjectsSettings {
            class_mapping: vec![ExtractClassMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class: ObjectClass::Local(1),
            }],
            ..Default::default()
        });
        let error = expand(&command_step(1, extract), &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("group class L1"), "{error}");
    }

    #[test]
    fn a_group_may_only_use_the_local_classes_it_defines() {
        let undeclared = definition(
            "SEEDS",
            vec![command_step(1, voronoi_with_center(ObjectClass::Local(2)))],
            vec![],
        );
        let error = expand(&group_step(1, "SEEDS", &[]), &[undeclared])
            .unwrap_err()
            .to_string();
        assert!(error.contains("step 1 uses the group class L2"), "{error}");

        // Declared, and set through a group parameter in its text form.
        let mut declared = definition(
            "SEEDS",
            vec![command_step(1, voronoi_with_center(ObjectClass::Global(4)))],
            vec![parameter(
                1,
                ParamValue::Text("L2".into()),
                &[(1, "centers")],
            )],
        );
        declared.classes = vec![local_class(2)];
        let commands = expand(&group_step(1, "SEEDS", &[]), &[declared]).unwrap();
        assert_eq!(voronoi(&commands[0].command).centers, ObjectClass::Local(2));
    }

    fn scope(n: u32) -> Scope {
        Scope::Scoped(ScopeId(n))
    }

    #[test]
    fn each_group_instance_runs_in_a_scope_of_its_own() {
        let mut scopes = ScopeCounter::default();
        let plain = command_step(1, PipelineCommand::Blur(BlurSettings::default()));
        let group = group_step(2, "CELL_RECONSTRUCTION", &[]);
        let definitions = [cell_reconstruction()];

        let commands = expand_step(&plain, &definitions, &mut scopes).unwrap();
        assert_eq!(commands[0].scope, Scope::Global, "outside groups");

        // Two instances of one group, as in two pipelines of a job.
        let first = expand_step(&group, &definitions, &mut scopes).unwrap();
        let second = expand_step(&group, &definitions, &mut scopes).unwrap();
        assert!(first.iter().all(|c| c.scope == scope(1)));
        assert!(second.iter().all(|c| c.scope == scope(2)));
    }

    #[test]
    fn a_nested_group_has_its_own_scope() {
        let outer = definition(
            "OUTER",
            vec![
                group_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            vec![],
        );
        let commands = expand(
            &group_step(4, "OUTER", &[]),
            &[cell_reconstruction(), outer],
        )
        .unwrap();
        let scopes: Vec<Scope> = commands.iter().map(|c| c.scope).collect();
        assert_eq!(scopes, [scope(2), scope(2), scope(1)]);
    }

    #[test]
    fn a_local_class_cant_be_passed_into_a_nested_group() {
        let mut outer = definition(
            "OUTER",
            vec![group_step(1, "CELL_RECONSTRUCTION", &[])],
            // The outer group's local class 1 into the inner "Cell class".
            vec![parameter(1, ParamValue::Text("L1".into()), &[(1, "2")])],
        );
        outer.classes = vec![local_class(1)];
        let error = expand(
            &group_step(4, "OUTER", &[]),
            &[cell_reconstruction(), outer],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("can't be passed into a nested group"),
            "{error}"
        );
    }

    #[test]
    fn a_dataset_step_in_a_group_may_not_use_local_classes() {
        let mut tracker = definition(
            "TRACK",
            vec![command_step(
                1,
                PipelineCommand::SimpleTracker(SimpleTrackerSettings {
                    object_class: ObjectClass::Local(1),
                    ..Default::default()
                }),
            )],
            vec![],
        );
        tracker.classes = vec![local_class(1)];
        let error = expand(&group_step(1, "TRACK", &[]), &[tracker])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("dataset step using the group class L1"),
            "{error}"
        );
    }
}
