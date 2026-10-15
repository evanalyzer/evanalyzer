//! Expanding command groups into plain commands before a job runs.
//!
//! A pipeline step is a command or an instance of a command group (see
//! `evanalyzer_cfg::settings::command_groups`). The executor only knows
//! commands, so the job generator replaces each group instance by its
//! definition's commands, with the instance's parameter values written into
//! the inner parameters they're bound to.

use crate::pipeline::scope::{Scope, ScopeId};
use evanalyzer_cfg::{
    core_types::{GroupParamKey, InternalErrors, ObjectClass, StepId},
    settings::{
        command_groups::{
            GroupDefinition, GroupInstance, GroupParameterDef, ParamBinding, ParamValue,
        },
        parameter_def::ParameterDef,
        pipeline_command::{CommandCategory, PipelineCommand},
        pipeline_command_settings::{
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings,
            ObjectClassifyObjectsClassifyMatchHandlingSettings,
        },
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
            if let Some(slot) = group_memory_slots(command).first() {
                return Err(InternalErrors::InvalidArgument(format!(
                    "uses the group memory slot {slot}, which only exists inside a command group"
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

    let steps = steps_with_values(instance, definition)?;

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
                if let Some(why) = mixed_classes(command) {
                    return Err(group_error(
                        &definition.key,
                        format!("step {} {why}", step.id),
                    ));
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

/// `definition`'s steps with `instance`'s values written into what they're
/// bound to - the instance's value, else the definition's default.
fn steps_with_values(
    instance: &GroupInstance,
    definition: &GroupDefinition,
) -> Result<Vec<PipelineStepSettings>, InternalErrors> {
    let mut steps = definition.steps.clone();
    for parameter in &definition.parameters {
        let value = instance
            .values
            .get(&parameter.key)
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
    Ok(steps)
}

/// `instance`'s group's steps with the instance's values applied - what
/// "Ungroup" puts into the pipeline in its place.
pub fn group_steps_with_values(
    instance: &GroupInstance,
    definitions: &[GroupDefinition],
) -> Result<Vec<PipelineStepSettings>, InternalErrors> {
    steps_with_values(instance, definition_of(instance, definitions)?)
}

/// The parameters a group card shows for `instance`: each group parameter
/// as the inner parameter it's bound to (its first binding) - same type,
/// options and range, so the card gets the same control as the command -
/// with the instance's value, the group parameter's id as `name`, its label
/// and description, and its default.
pub fn group_parameters(
    instance: &GroupInstance,
    definitions: &[GroupDefinition],
) -> Result<Vec<ParameterDef>, InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let steps = steps_with_values(instance, definition)?;
    definition
        .parameters
        .iter()
        .map(|parameter| {
            let binding = parameter.bindings.first().ok_or_else(|| {
                group_error(
                    &definition.key,
                    format!("parameter \"{}\" isn't bound to anything", parameter.label),
                )
            })?;
            let step = steps
                .iter()
                .find(|s| s.id == binding.step)
                .ok_or_else(|| missing_step(definition, parameter, binding))?;
            let bound = match &step.content {
                StepContent::Command(command) => {
                    find_param(&crate::algos::command_parameters(command), &binding.param).cloned()
                }
                StepContent::Group(inner) => group_parameters(inner, definitions)?
                    .into_iter()
                    .find(|p| p.name == binding.param),
            };
            let mut shown = bound.ok_or_else(|| {
                group_error(
                    &definition.key,
                    format!(
                        "parameter \"{}\" is bound to {} of step {}, which doesn't exist",
                        parameter.label, binding.param, binding.step
                    ),
                )
            })?;
            shown.name = parameter.key.to_string();
            shown.display_name = parameter.label.clone();
            if !parameter.description.is_empty() {
                shown.description = parameter.description.clone();
            }
            shown.default_value = value_text(&parameter.default);
            // Exposed on purpose - never hidden behind "advanced".
            shown.advanced = false;
            Ok(shown)
        })
        .collect()
}

/// Sets `instance`'s value of group parameter `key` from `text` - a value as
/// the card's control reports it (`apply_param_change`'s format, e.g. a
/// dropdown label or "toggle:3"). It's applied to a copy of the bound
/// command and read back, so the value is the one the command took.
pub fn set_group_parameter(
    instance: &mut GroupInstance,
    definitions: &[GroupDefinition],
    key: &GroupParamKey,
    text: &str,
) -> Result<(), InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let parameter = definition
        .parameters
        .iter()
        .find(|p| &p.key == key)
        .ok_or_else(|| group_error(&definition.key, format!("has no parameter {key}")))?;
    let binding = parameter.bindings.first().ok_or_else(|| {
        group_error(
            &definition.key,
            format!("parameter \"{}\" isn't bound to anything", parameter.label),
        )
    })?;
    let mut step = steps_with_values(instance, definition)?
        .into_iter()
        .find(|s| s.id == binding.step)
        .ok_or_else(|| missing_step(definition, parameter, binding))?;
    let value = match &mut step.content {
        StepContent::Command(command) => {
            crate::algos::apply_command_param(command, &binding.param, text);
            let now = find_param(&crate::algos::command_parameters(command), &binding.param)
                .map(|p| p.value.clone())
                .unwrap_or_default();
            value_from_text(&now)
        }
        StepContent::Group(inner) => {
            let inner_key = GroupParamKey::from(binding.param.as_str());
            set_group_parameter(inner, definitions, &inner_key, text)?;
            let inner_definition = definition_of(inner, definitions)?;
            inner.values.get(&inner_key).cloned().unwrap_or_else(|| {
                inner_definition
                    .parameters
                    .iter()
                    .find(|p| p.key == inner_key)
                    .map_or(ParamValue::Text(String::new()), |p| p.default.clone())
            })
        }
    };
    instance.values.insert(key.clone(), value);
    Ok(())
}

fn definition_of<'a>(
    instance: &GroupInstance,
    definitions: &'a [GroupDefinition],
) -> Result<&'a GroupDefinition, InternalErrors> {
    definitions
        .iter()
        .find(|d| d.key == instance.key)
        .ok_or_else(|| group_error(&instance.key, "isn't defined in this project".into()))
}

fn missing_step(
    definition: &GroupDefinition,
    parameter: &GroupParameterDef,
    binding: &ParamBinding,
) -> InternalErrors {
    group_error(
        &definition.key,
        format!(
            "parameter \"{}\" is bound to step {}, which the group doesn't have",
            parameter.label, binding.step
        ),
    )
}

/// A parameter's text value (`ParameterDef::value`) as a typed group value -
/// the reverse of [`value_text`]: `true`/`false`, whole and decimal numbers,
/// comma-separated lists (e.g. classes "1,3"), else text.
pub fn value_from_text(text: &str) -> ParamValue {
    if text.contains(',') {
        return ParamValue::List(text.split(',').map(|t| value_from_text(t.trim())).collect());
    }
    match text {
        "true" => ParamValue::Bool(true),
        "false" => ParamValue::Bool(false),
        _ => {
            if let Ok(int) = text.parse::<i64>() {
                ParamValue::Int(int)
            } else if let Ok(float) = text.parse::<f64>() {
                ParamValue::Float(float)
            } else {
                ParamValue::Text(text.to_string())
            }
        }
    }
}

/// Why `command` would give objects local and project classes at once, if
/// it would - see `Object::add_object_class`, which refuses it at run time.
/// Only adding a class to existing objects can mix: reclassifying replaces
/// the classes (the object moves to the new class's scope), and new objects
/// get a single class.
fn mixed_classes(command: &PipelineCommand) -> Option<String> {
    let (inputs, outputs): (&[ObjectClass], Vec<ObjectClass>) = match command {
        PipelineCommand::ClassifyObjects(s)
            if matches!(
                s.match_handling,
                ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch
                    | ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfNotMatch
            ) =>
        {
            (&s.input_classes, vec![s.output_class])
        }
        PipelineCommand::AiObjectClassifier(s)
            if matches!(
                s.match_handling,
                ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch
            ) =>
        {
            (
                &s.input_classes,
                s.segmentation_mapping
                    .iter()
                    .map(|m| m.output_class)
                    .collect(),
            )
        }
        _ => return None,
    };
    let is_local = |class: &ObjectClass| matches!(class, ObjectClass::Local(_));
    for output in outputs.iter().filter(|c| **c != ObjectClass::Unset) {
        if inputs.is_empty() {
            if is_local(output) {
                return Some(format!(
                    "adds the group class {} to all objects, also the project's - an object \
                     can't have both; select the group classes it applies to",
                    output.to_text()
                ));
            }
            continue;
        }
        for input in inputs.iter().filter(|c| **c != ObjectClass::Unset) {
            if is_local(input) != is_local(output) {
                let (adding, to) = if is_local(output) {
                    ("group class", "project class")
                } else {
                    ("project class", "group class")
                };
                return Some(format!(
                    "adds the {adding} {} to objects of the {to} {} - an object can't have \
                     both; use \"Reclassify\" to move them to the {adding}",
                    output.to_text(),
                    input.to_text()
                ));
            }
        }
    }
    None
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
            if find_param(&crate::algos::command_parameters(command), param).is_none() {
                return Err("the command has no such parameter".into());
            }
            crate::algos::apply_command_param(command, param, &text);
            let now = find_param(&crate::algos::command_parameters(command), param)
                .map(|p| p.value.clone())
                .unwrap_or_default();
            if same_value(&now, &text) {
                Ok(())
            } else {
                Err(format!("it doesn't accept {text:?}"))
            }
        }
        StepContent::Group(inner) => {
            let key = GroupParamKey::from(param);
            // The inner group would read it as one of its own local classes.
            if let Some(local) = local_class_in(value) {
                return Err(format!(
                    "the group class {} can't be passed into a nested group",
                    local.to_text()
                ));
            }
            if group_memory_in(value) {
                return Err(
                    "a group memory slot can't be passed into a nested group - it has its own"
                        .into(),
                );
            }
            inner.values.insert(key, value.clone());
            Ok(())
        }
    }
}

/// The group memory slots (`MemoryId::GroupContext`) `command` reads or writes.
/// Found in its file form, where such a slot is always `{"memory":
/// {"groupContext": n}}` - whichever setting holds it (an image address, a
/// threshold's value source, ...).
fn group_memory_slots(command: &PipelineCommand) -> Vec<u64> {
    fn collect(value: &serde_json::Value, slots: &mut Vec<u64>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(slot) = map
                    .get("memory")
                    .and_then(|memory| memory.get("groupContext"))
                    .and_then(serde_json::Value::as_u64)
                {
                    slots.push(slot);
                }
                map.values().for_each(|v| collect(v, slots));
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| collect(v, slots)),
            _ => {}
        }
    }
    let mut slots = Vec::new();
    if let Ok(value) = serde_json::to_value(command) {
        collect(&value, &mut slots);
    }
    slots
}

/// Whether `value` holds a group memory slot in its text form (`"group:2"`).
fn group_memory_in(value: &ParamValue) -> bool {
    match value {
        ParamValue::Text(text) => text.starts_with("group:"),
        ParamValue::List(items) => items.iter().any(group_memory_in),
        _ => false,
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
    use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::command_groups::{GroupParameterDef, ParamBinding};
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        AiObjectClassifierSettings, BlurSettings, ClassificationMappingSettings,
        ClassifyObjectsSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
        ImageCacheSettings, MathImageCacheImageCacheModeSettings, SimpleTrackerSettings,
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
                    .map(|(k, v)| (GroupParamKey(k.to_string()), v.clone()))
                    .collect::<BTreeMap<_, _>>(),
            }),
        }
    }

    fn parameter(id: u32, default: ParamValue, bindings: &[(u32, &str)]) -> GroupParameterDef {
        GroupParameterDef {
            key: GroupParamKey(id.to_string()),
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

    // ---- Mixing local and project classes ----

    fn classify(
        input: Vec<ObjectClass>,
        handling: ObjectClassifyObjectsClassifyMatchHandlingSettings,
        output: ObjectClass,
    ) -> PipelineCommand {
        PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
            input_classes: input,
            match_handling: handling,
            output_class: output,
            ..Default::default()
        })
    }

    /// Expands a group with local classes 1 and 2 running `command`.
    fn in_group(command: PipelineCommand) -> Result<Vec<ExpandedCommand>, InternalErrors> {
        let mut group = definition("MIX", vec![command_step(3, command)], vec![]);
        group.classes = vec![local_class(1), local_class(2)];
        expand(&group_step(1, "MIX", &[]), &[group])
    }

    #[test]
    fn adding_a_class_of_the_other_kind_to_existing_objects_is_refused() {
        use ObjectClassifyObjectsClassifyMatchHandlingSettings::*;
        let local = ObjectClass::Local(1);
        let global = ObjectClass::Global(5);

        let error = in_group(classify(vec![local], AddOutputClassIfMatch, global))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("step 3 adds the project class 5 to objects of the group class L1")
                && error.contains("Reclassify"),
            "{error}"
        );
        let error = in_group(classify(vec![global], AddOutputClassIfNotMatch, local))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("adds the group class L1 to objects of the project class 5"),
            "{error}"
        );
        // No input classes: every object the command sees, the project's too.
        let error = in_group(classify(vec![], AddOutputClassIfMatch, local))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("to all objects, also the project's"),
            "{error}"
        );
    }

    #[test]
    fn reclassifying_and_adding_a_class_of_the_same_kind_are_fine() {
        use ObjectClassifyObjectsClassifyMatchHandlingSettings::*;
        let (l1, l2, global) = (
            ObjectClass::Local(1),
            ObjectClass::Local(2),
            ObjectClass::Global(5),
        );
        for command in [
            classify(vec![l1], ReclassifyIfMatch, global),
            classify(vec![global], ReclassifyIfNotMatch, l1),
            classify(vec![], ReclassifyIfMatch, l1),
            classify(vec![l1], AddOutputClassIfMatch, l2),
            classify(vec![], AddOutputClassIfMatch, global),
            classify(vec![l1], RemoveInputClassIfMatch, global),
        ] {
            assert!(in_group(command).is_ok());
        }
    }

    #[test]
    fn the_ai_object_classifier_is_checked_per_output_class() {
        let ai = |handling| {
            PipelineCommand::AiObjectClassifier(AiObjectClassifierSettings {
                input_classes: vec![ObjectClass::Local(1)],
                segmentation_mapping: vec![
                    ClassificationMappingSettings {
                        object_class: ObjectClass::Global(1),
                        output_class: ObjectClass::Local(2),
                    },
                    ClassificationMappingSettings {
                        object_class: ObjectClass::Global(2),
                        output_class: ObjectClass::Global(7),
                    },
                ],
                match_handling: handling,
                ..Default::default()
            })
        };
        let error = in_group(ai(
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch,
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("adds the project class 7"), "{error}");
        assert!(
            in_group(ai(
                ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch
            ))
            .is_ok()
        );
    }

    // ---- Group memory slots ----

    fn store_in(address: ImageAddress) -> PipelineCommand {
        PipelineCommand::ImageCache(ImageCacheSettings {
            mode: MathImageCacheImageCacheModeSettings::Store,
            address,
        })
    }

    #[test]
    fn a_group_memory_slot_outside_a_group_is_refused() {
        let group_slot = ImageAddress::Memory(MemoryId::GroupContext(2));
        let error = expand(&command_step(1, store_in(group_slot)), &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("group memory slot 2"), "{error}");

        let group = definition("SLOTS", vec![command_step(1, store_in(group_slot))], vec![]);
        assert!(expand(&group_step(1, "SLOTS", &[]), &[group]).is_ok());
    }

    #[test]
    fn a_group_memory_slot_cant_be_passed_into_a_nested_group() {
        let inner = definition(
            "INNER",
            vec![command_step(
                1,
                store_in(ImageAddress::Memory(MemoryId::PipelineContext(1))),
            )],
            vec![parameter(
                1,
                ParamValue::Text("memory:1".into()),
                &[(1, "address")],
            )],
        );
        let outer = definition(
            "OUTER",
            vec![group_step(1, "INNER", &[])],
            vec![parameter(
                1,
                ParamValue::Text("group:1".into()),
                &[(1, "1")],
            )],
        );
        let error = expand(&group_step(4, "OUTER", &[]), &[inner, outer])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("memory slot can't be passed into a nested group"),
            "{error}"
        );
    }

    // ---- The group card's parameters ----

    fn instance(key: &str, values: &[(u32, ParamValue)]) -> GroupInstance {
        let StepContent::Group(instance) = group_step(1, key, values).content else {
            unreachable!()
        };
        instance
    }

    #[test]
    fn a_group_parameter_shows_as_the_parameter_it_is_bound_to() {
        let definitions = [cell_reconstruction()];
        let shown = group_parameters(
            &instance("CELL_RECONSTRUCTION", &[(2, ParamValue::Int(4))]),
            &definitions,
        )
        .unwrap();

        assert_eq!(shown.len(), 2);
        let blur = &shown[0];
        assert_eq!(
            (blur.name.as_str(), blur.display_name.as_str()),
            ("1", "Parameter 1")
        );
        assert_eq!(blur.value, "3", "the definition's default");
        let blur_kernel = find_param(
            &PipelineCommand::Blur(BlurSettings::default()).to_parameters(),
            "kernel_size",
        )
        .unwrap()
        .clone();
        assert_eq!(
            blur.param_type, blur_kernel.param_type,
            "the command's control"
        );
        assert_eq!((blur.min, blur.max), (blur_kernel.min, blur_kernel.max));

        let cell_class = &shown[1];
        assert_eq!(cell_class.value, "4", "the instance's value");
        assert_eq!(cell_class.default_value, "1", "the group's default");
    }

    #[test]
    fn setting_a_group_parameter_stores_what_the_command_took() {
        let definitions = [cell_reconstruction()];
        let mut cells = instance("CELL_RECONSTRUCTION", &[]);

        set_group_parameter(&mut cells, &definitions, &GroupParamKey::from("1"), "5").unwrap();
        set_group_parameter(&mut cells, &definitions, &GroupParamKey::from("2"), "L1").unwrap();
        assert_eq!(cells.values[&GroupParamKey::from("1")], ParamValue::Int(5));
        assert_eq!(
            cells.values[&GroupParamKey::from("2")],
            ParamValue::Text("L1".into())
        );

        // Text the command doesn't take leaves the value as it was.
        set_group_parameter(&mut cells, &definitions, &GroupParamKey::from("1"), "big").unwrap();
        assert_eq!(cells.values[&GroupParamKey::from("1")], ParamValue::Int(5));

        assert!(
            set_group_parameter(&mut cells, &definitions, &GroupParamKey::from("9"), "1").is_err()
        );
    }

    #[test]
    fn a_parameter_bound_to_a_nested_groups_parameter_works_the_same() {
        let outer = definition(
            "OUTER",
            vec![group_step(1, "CELL_RECONSTRUCTION", &[])],
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let definitions = [cell_reconstruction(), outer];
        let mut outer = instance("OUTER", &[]);

        let shown = group_parameters(&outer, &definitions).unwrap();
        assert_eq!(shown[0].value, "7");

        set_group_parameter(&mut outer, &definitions, &GroupParamKey::from("1"), "8").unwrap();
        assert_eq!(outer.values[&GroupParamKey::from("1")], ParamValue::Int(8));
        let commands = expand(
            &group_step(1, "OUTER", &[(1, ParamValue::Int(8))]),
            &definitions,
        )
        .unwrap();
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(8)
        );
    }
}
