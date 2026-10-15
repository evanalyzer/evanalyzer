//! Expanding components into plain commands before a job runs.
//!
//! A pipeline step is a command or an instance of a component (see
//! `evanalyzer_cfg::settings::components`). The executor only knows
//! commands, so the job generator replaces each component instance by its
//! definition's commands, with the instance's parameter values written into
//! the inner parameters they're bound to.

use crate::pipeline::scope::{Scope, ScopeId};
use evanalyzer_cfg::{
    core_types::{ComponentParamKey, InternalErrors, MemorySlot, ObjectClass, StepId},
    settings::{
        components::{
            ComponentDefinition, ComponentInstance, ComponentParameterDef, ParamBinding, ParamValue,
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
use std::collections::{BTreeMap, BTreeSet};

/// How deep components may nest - deeper is almost certainly a mistake.
const MAX_COMPONENT_DEPTH: usize = 8;

/// One command a pipeline step runs.
#[derive(Debug, Clone)]
pub struct ExpandedCommand {
    /// What a breakpoint targets - see [`expand_step`].
    pub step_id: StepId,
    /// The scope the command runs in: its component instance's, else `Global`.
    pub scope: Scope,
    /// The component uses the command belongs to, outermost first - each
    /// ends after its last command (see [`ScopeCounter::end_of`]).
    pub uses: Vec<UseId>,
    pub command: PipelineCommand,
}

/// One use of a component in a job, numbered at expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UseId(pub u32);

/// What goes when a component use ends.
#[derive(Debug, Clone, PartialEq)]
pub enum UseEnd {
    /// A use in a pipeline: its scope - its objects and memory slots.
    Scope(Scope),
    /// A use inside another component: it runs in the scope of the use in the
    /// pipeline around it (so it can work on that one's classes), with
    /// classes and memory slots of its own numbered apart - those go.
    Nested {
        scope: Scope,
        classes: Vec<ObjectClass>,
        slots: Vec<MemorySlot>,
    },
}

/// Numbers the component uses of a job - one counter per job, so no two
/// uses in pipelines share a scope - and remembers what ends with each.
#[derive(Debug, Default)]
pub struct ScopeCounter {
    last: u32,
    ends: BTreeMap<UseId, UseEnd>,
}

impl ScopeCounter {
    fn next(&mut self, end: impl FnOnce(UseId) -> UseEnd) -> UseId {
        self.last += 1;
        let id = UseId(self.last);
        self.ends.insert(id, end(id));
        id
    }

    /// What goes when use `id` ends.
    pub fn end_of(&self, id: UseId) -> Option<&UseEnd> {
        self.ends.get(&id)
    }
}

/// The classes and memory slots a use in a pipeline hands out to the
/// components nested in it: they run in its scope, so each gets numbers no
/// other class or slot of the scope has.
struct Frame {
    scope: Scope,
    next_class: u32,
    next_slot: MemorySlot,
}

impl Frame {
    /// For a use of `definition` in a pipeline, in `scope`: the numbers after
    /// its own.
    fn new(scope: Scope, definition: &ComponentDefinition) -> Self {
        let (classes, slots) = own_numbers(definition);
        Self {
            scope,
            next_class: classes.last().map_or(1, |n| n + 1),
            next_slot: slots.last().map_or(1, |n| n + 1),
        }
    }
}

/// The commands `step` runs, in order, each with the id a breakpoint
/// targets, the scope it runs in and the component uses it belongs to: a
/// command step is itself, in the global scope; a component is its
/// definition's enabled commands with the instance's values applied, in a
/// new scope from `scopes` - a component nested in it in that scope too, its
/// own classes and memory slots numbered apart (see [`UseEnd::Nested`]).
/// The last command carries the component step's id (a breakpoint on a
/// component stops after the whole component), the others
/// [`StepId::UNASSIGNED`] (inner steps can't be targeted).
pub fn expand_step(
    step: &PipelineStepSettings,
    definitions: &[ComponentDefinition],
    scopes: &mut ScopeCounter,
) -> Result<Vec<ExpandedCommand>, InternalErrors> {
    match &step.content {
        StepContent::Command(command) => {
            if let Some(class) = command.local_classes().first() {
                return Err(only_inside_class(*class));
            }
            if let Some(slot) = component_memory_slots(command).first() {
                return Err(only_inside_slot(*slot));
            }
            Ok(vec![ExpandedCommand {
                step_id: step.id,
                scope: Scope::Global,
                uses: Vec::new(),
                command: command.clone(),
            }])
        }
        StepContent::Component(instance) => {
            // A pipeline has no component classes or memory to hand in.
            for value in instance.values.values() {
                if let Some(class) = local_class_in(value) {
                    return Err(only_inside_class(class));
                }
                if let Some(slot) = component_slot_in(value) {
                    return Err(only_inside_slot(slot as u64));
                }
            }
            let commands = expand_instance(
                instance,
                definitions,
                scopes,
                None,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            let last = commands.len().saturating_sub(1);
            Ok(commands
                .into_iter()
                .enumerate()
                .map(|(i, (scope, uses, command))| ExpandedCommand {
                    step_id: if i == last {
                        step.id
                    } else {
                        StepId::UNASSIGNED
                    },
                    scope,
                    uses,
                    command,
                })
                .collect())
        }
    }
}

fn only_inside_class(class: ObjectClass) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "uses the component class {}, which only exists inside a component",
        class.to_text()
    ))
}

fn only_inside_slot(slot: u64) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "uses the component memory slot {slot}, which only exists inside a component"
    ))
}

/// The commands of `instance`'s definition with its values applied, each
/// with the scope it runs in and the uses it belongs to. `around` is the
/// frame of the use in the pipeline this one is nested in - `None` for that
/// use itself, which gets a new scope from `scopes`. `open` holds the keys
/// of the components being expanded around this one, `uses` their uses.
fn expand_instance(
    instance: &ComponentInstance,
    definitions: &[ComponentDefinition],
    scopes: &mut ScopeCounter,
    around: Option<&mut Frame>,
    open: &mut Vec<String>,
    uses: &mut Vec<UseId>,
) -> Result<Vec<(Scope, Vec<UseId>, PipelineCommand)>, InternalErrors> {
    let raw = definition_of(instance, definitions)?;
    if open.contains(&instance.key) {
        return Err(component_error(&instance.key, "contains itself".into()));
    }
    if open.len() >= MAX_COMPONENT_DEPTH {
        return Err(component_error(
            &instance.key,
            format!("is nested deeper than {MAX_COMPONENT_DEPTH} components"),
        ));
    }
    check_own_classes(raw)?;

    // In a pipeline: a scope of its own. Nested: the scope around it, its
    // classes and slots numbered apart - the values it's given are in the
    // numbers of the scope already.
    let mut own_frame;
    let (frame, definition) = match around {
        None => {
            let id = scopes.next(|id| UseEnd::Scope(Scope::Scoped(ScopeId(id.0))));
            own_frame = Frame::new(Scope::Scoped(ScopeId(id.0)), raw);
            uses.push(id);
            (&mut own_frame, std::borrow::Cow::Borrowed(raw))
        }
        Some(frame) => {
            let (renumbered, classes, slots) = renumber(raw, frame);
            let scope = frame.scope;
            let id = scopes.next(|_| UseEnd::Nested {
                scope,
                classes,
                slots,
            });
            uses.push(id);
            (frame, std::borrow::Cow::Owned(renumbered))
        }
    };

    let steps = steps_with_values(instance, &definition)?;

    open.push(instance.key.clone());
    let mut commands = Vec::new();
    for step in steps.iter().filter(|s| s.enabled) {
        match &step.content {
            StepContent::Command(command) => {
                if matches!(command.category(), CommandCategory::Dataset) {
                    if let Some(class) = command.local_classes().first() {
                        return Err(component_error(
                            &definition.key,
                            format!(
                                "step {} is a dataset step using the component class {} - dataset \
                                 steps run on stored objects, which only have project classes",
                                step.id,
                                class.to_text()
                            ),
                        ));
                    }
                }
                if let Some(why) = mixed_classes(command) {
                    return Err(component_error(
                        &definition.key,
                        format!("step {} {why}", step.id),
                    ));
                }
                commands.push((frame.scope, uses.clone(), command.clone()))
            }
            StepContent::Component(inner) => commands.extend(expand_instance(
                inner,
                definitions,
                scopes,
                Some(&mut *frame),
                open,
                uses,
            )?),
        }
    }
    open.pop();
    uses.pop();
    Ok(commands)
}

/// Refuses a component class `definition` uses without defining it - in a
/// command, a value it gives a component inside it, or a parameter's default.
fn check_own_classes(definition: &ComponentDefinition) -> Result<(), InternalErrors> {
    let defined = |class: &ObjectClass| definition.classes.iter().any(|c| c.local() == *class);
    for step in &definition.steps {
        let used: Vec<ObjectClass> = match &step.content {
            StepContent::Command(command) => command.local_classes(),
            StepContent::Component(inner) => {
                inner.values.values().filter_map(local_class_in).collect()
            }
        };
        if let Some(class) = used.into_iter().find(|c| !defined(c)) {
            return Err(component_error(
                &definition.key,
                format!(
                    "step {} uses the component class {}, which the component doesn't define",
                    step.id,
                    class.to_text()
                ),
            ));
        }
    }
    for parameter in &definition.parameters {
        if let Some(class) = local_class_in(&parameter.default).filter(|c| !defined(c)) {
            return Err(component_error(
                &definition.key,
                format!(
                    "parameter \"{}\" starts with the component class {}, which the component \
                     doesn't define",
                    parameter.label,
                    class.to_text()
                ),
            ));
        }
    }
    Ok(())
}

/// The component classes (their numbers) and memory slots `definition` has:
/// the classes it defines, the slots its steps and parameters use - sorted.
pub fn own_numbers(definition: &ComponentDefinition) -> (Vec<u32>, Vec<MemorySlot>) {
    let classes: BTreeSet<u32> = definition.classes.iter().map(|c| c.id).collect();
    let mut slots: BTreeSet<MemorySlot> = BTreeSet::new();
    for step in &definition.steps {
        match &step.content {
            StepContent::Command(command) => slots.extend(
                component_memory_slots(command)
                    .into_iter()
                    .map(|s| s as MemorySlot),
            ),
            StepContent::Component(inner) => {
                slots.extend(inner.values.values().filter_map(component_slot_in))
            }
        }
    }
    slots.extend(
        definition
            .parameters
            .iter()
            .filter_map(|p| component_slot_in(&p.default)),
    );
    (classes.into_iter().collect(), slots.into_iter().collect())
}

/// `definition` with its classes and memory slots numbered from `frame`'s
/// next free ones - and the classes and slots they became.
fn renumber(
    definition: &ComponentDefinition,
    frame: &mut Frame,
) -> (ComponentDefinition, Vec<ObjectClass>, Vec<MemorySlot>) {
    let (own_classes, own_slots) = own_numbers(definition);
    let classes: BTreeMap<u32, u32> = own_classes
        .into_iter()
        .map(|n| {
            frame.next_class += 1;
            (n, frame.next_class - 1)
        })
        .collect();
    let slots: BTreeMap<MemorySlot, MemorySlot> = own_slots
        .into_iter()
        .map(|n| {
            frame.next_slot += 1;
            (n, frame.next_slot - 1)
        })
        .collect();

    let renumbered = renumbered(definition, &classes, &slots);
    (
        renumbered,
        classes.values().map(|n| ObjectClass::Local(*n)).collect(),
        slots.values().copied().collect(),
    )
}

/// `definition` with its component classes and memory slots renumbered as
/// `classes` and `slots` say (old → new) - wherever they're written: its
/// classes, commands, the values it gives components inside it, its
/// parameters' defaults.
pub fn renumbered(
    definition: &ComponentDefinition,
    classes: &BTreeMap<u32, u32>,
    slots: &BTreeMap<MemorySlot, MemorySlot>,
) -> ComponentDefinition {
    let mut renumbered = definition.clone();
    for class in &mut renumbered.classes {
        if let Some(new) = classes.get(&class.id) {
            class.id = *new;
        }
    }
    for step in &mut renumbered.steps {
        match &mut step.content {
            StepContent::Command(command) => {
                if let Ok(mut value) = serde_json::to_value(&*command) {
                    renumber_json(&mut value, classes, slots);
                    if let Ok(done) = serde_json::from_value(value) {
                        *command = done;
                    }
                }
            }
            StepContent::Component(inner) => {
                for value in inner.values.values_mut() {
                    renumber_value(value, classes, slots);
                }
            }
        }
    }
    for parameter in &mut renumbered.parameters {
        renumber_value(&mut parameter.default, classes, slots);
    }
    renumbered
}

/// Renumbers the component classes (`{"LOCAL": n}`) and memory slots
/// (`{"componentContext": n}`) of a command in its file form.
fn renumber_json(
    value: &mut serde_json::Value,
    classes: &BTreeMap<u32, u32>,
    slots: &BTreeMap<MemorySlot, MemorySlot>,
) {
    match value {
        serde_json::Value::Object(map) => {
            if map.len() == 1 {
                let (key, number) = map.iter_mut().next().expect("one entry");
                let table = match key.as_str() {
                    "LOCAL" => Some(classes),
                    "componentContext" => Some(slots),
                    _ => None,
                };
                if let (Some(table), Some(n)) = (table, number.as_u64())
                    && let Some(new) = table.get(&(n as u32))
                {
                    *number = serde_json::Value::from(*new);
                    return;
                }
            }
            map.values_mut()
                .for_each(|v| renumber_json(v, classes, slots));
        }
        serde_json::Value::Array(items) => items
            .iter_mut()
            .for_each(|v| renumber_json(v, classes, slots)),
        _ => {}
    }
}

/// Renumbers the component classes ("L1") and memory slots ("component:1")
/// of a component parameter's value.
fn renumber_value(
    value: &mut ParamValue,
    classes: &BTreeMap<u32, u32>,
    slots: &BTreeMap<MemorySlot, MemorySlot>,
) {
    match value {
        ParamValue::Text(text) => {
            if let Some(ObjectClass::Local(n)) = ObjectClass::from_text(text)
                && let Some(new) = classes.get(&n)
            {
                *text = ObjectClass::Local(*new).to_text();
            } else if let Some(n) = component_slot_text(text)
                && let Some(new) = slots.get(&n)
            {
                *text = format!("component:{new}");
            }
        }
        ParamValue::List(items) => items
            .iter_mut()
            .for_each(|v| renumber_value(v, classes, slots)),
        _ => {}
    }
}

/// `definition`'s steps with `instance`'s values written into what they're
/// bound to - the instance's value, else the definition's default.
fn steps_with_values(
    instance: &ComponentInstance,
    definition: &ComponentDefinition,
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
                    component_error(
                        &definition.key,
                        format!(
                            "parameter \"{}\" is bound to step {}, which the component doesn't have",
                            parameter.label, binding.step
                        ),
                    )
                })?;
            set_bound_value(step, &binding.param, value).map_err(|why| {
                component_error(
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

/// `instance`'s component's steps with the instance's values applied - what
/// "Expand component" puts into the pipeline in its place.
pub fn component_steps_with_values(
    instance: &ComponentInstance,
    definitions: &[ComponentDefinition],
) -> Result<Vec<PipelineStepSettings>, InternalErrors> {
    steps_with_values(instance, definition_of(instance, definitions)?)
}

/// The parameters a component card shows for `instance`: each component parameter
/// as the inner parameter it's bound to (its first binding) - same type,
/// options and range, so the card gets the same control as the command -
/// with the instance's value, the component parameter's id as `name`, its label
/// and description, and its default.
pub fn component_parameters(
    instance: &ComponentInstance,
    definitions: &[ComponentDefinition],
) -> Result<Vec<ParameterDef>, InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let steps = steps_with_values(instance, definition)?;
    definition
        .parameters
        .iter()
        .map(|parameter| {
            let binding = parameter.bindings.first().ok_or_else(|| {
                component_error(
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
                StepContent::Component(inner) => component_parameters(inner, definitions)?
                    .into_iter()
                    .find(|p| p.name == binding.param),
            };
            let mut shown = bound.ok_or_else(|| {
                component_error(
                    &definition.key,
                    format!(
                        "parameter \"{}\" is bound to {} of step {}, which doesn't exist",
                        parameter.label, binding.param, binding.step
                    ),
                )
            })?;
            shown.name = parameter.key.to_string();
            shown.display_name = parameter.label.clone();
            shown.section = parameter.section.clone();
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

/// Sets `instance`'s value of component parameter `key` from `text` - a value as
/// the card's control reports it (`apply_param_change`'s format, e.g. a
/// dropdown label or "toggle:3"). It's applied to a copy of the bound
/// command and read back, so the value is the one the command took.
pub fn set_component_parameter(
    instance: &mut ComponentInstance,
    definitions: &[ComponentDefinition],
    key: &ComponentParamKey,
    text: &str,
) -> Result<(), InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let parameter = definition
        .parameters
        .iter()
        .find(|p| &p.key == key)
        .ok_or_else(|| component_error(&definition.key, format!("has no parameter {key}")))?;
    let binding = parameter.bindings.first().ok_or_else(|| {
        component_error(
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
        StepContent::Component(inner) => {
            let inner_key = ComponentParamKey::from(binding.param.as_str());
            set_component_parameter(inner, definitions, &inner_key, text)?;
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
    instance: &ComponentInstance,
    definitions: &'a [ComponentDefinition],
) -> Result<&'a ComponentDefinition, InternalErrors> {
    definitions
        .iter()
        .find(|d| d.key == instance.key)
        .ok_or_else(|| component_error(&instance.key, "isn't defined in this project".into()))
}

fn missing_step(
    definition: &ComponentDefinition,
    parameter: &ComponentParameterDef,
    binding: &ParamBinding,
) -> InternalErrors {
    component_error(
        &definition.key,
        format!(
            "parameter \"{}\" is bound to step {}, which the component doesn't have",
            parameter.label, binding.step
        ),
    )
}

/// A parameter's text value (`ParameterDef::value`) as a typed component value -
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
                    "adds the component class {} to all objects, also the project's - an object \
                     can't have both; select the component classes it applies to",
                    output.to_text()
                ));
            }
            continue;
        }
        for input in inputs.iter().filter(|c| **c != ObjectClass::Unset) {
            if is_local(input) != is_local(output) {
                let (adding, to) = if is_local(output) {
                    ("component class", "project class")
                } else {
                    ("project class", "component class")
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
/// for a nested component, the value of its component parameter `param` (its id).
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
        StepContent::Component(inner) => {
            // A component class or memory slot too: the nested component
            // runs in this one's scope (see `UseEnd::Nested`).
            let key = ComponentParamKey::from(param);
            inner.values.insert(key, value.clone());
            Ok(())
        }
    }
}

/// The component memory slots (`MemoryId::ComponentContext`) `command` reads or writes.
/// Found in its file form, where such a slot is always `{"memory":
/// {"componentContext": n}}` - whichever setting holds it (an image address, a
/// threshold's value source, ...).
fn component_memory_slots(command: &PipelineCommand) -> Vec<u64> {
    fn collect(value: &serde_json::Value, slots: &mut Vec<u64>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(slot) = map
                    .get("memory")
                    .and_then(|memory| memory.get("componentContext"))
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

/// The first component memory slot in `value`, in its text form
/// (`"component:2"`), if any.
fn component_slot_in(value: &ParamValue) -> Option<MemorySlot> {
    match value {
        ParamValue::Text(text) => component_slot_text(text),
        ParamValue::List(items) => items.iter().find_map(component_slot_in),
        _ => None,
    }
}

/// The slot of a component memory slot's text form (`"component:2"`).
fn component_slot_text(text: &str) -> Option<MemorySlot> {
    text.trim().strip_prefix("component:")?.trim().parse().ok()
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

fn component_error(key: &str, what: String) -> InternalErrors {
    InternalErrors::InvalidArgument(format!("Component {key}: {what}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectClass;
    use evanalyzer_cfg::core_types::SegmentationClass;
    use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::components::{ComponentParameterDef, ParamBinding};
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

    fn component_step(id: u32, key: &str, values: &[(u32, ParamValue)]) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (ComponentParamKey(k.to_string()), v.clone()))
                    .collect::<BTreeMap<_, _>>(),
            }),
        }
    }

    fn parameter(id: u32, default: ParamValue, bindings: &[(u32, &str)]) -> ComponentParameterDef {
        ComponentParameterDef {
            key: ComponentParamKey(id.to_string()),
            label: format!("Parameter {id}"),
            description: String::new(),
            section: String::new(),
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
        parameters: Vec<ComponentParameterDef>,
    ) -> ComponentDefinition {
        ComponentDefinition {
            key: key.into(),
            meta: Default::default(),
            classes: vec![],
            steps,
            parameters,
            origin: None,
        }
    }

    /// Blur, then Voronoi - "Cell class" set in both places.
    fn cell_reconstruction() -> ComponentDefinition {
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
        definitions: &[ComponentDefinition],
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
    fn a_component_runs_its_commands_with_the_instances_values_and_defaults() {
        let step = component_step(9, "CELL_RECONSTRUCTION", &[(2, ParamValue::Int(4))]);
        let commands = expand(&step, &[cell_reconstruction()]).unwrap();

        assert_eq!(commands.len(), 2);
        let PipelineCommand::Blur(blur) = &commands[0].command else {
            panic!("expected Blur");
        };
        assert_eq!(blur.kernel_size, 3, "the definition's default");
        // One component parameter, two inner parameters - always the same value.
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(4)
        );
        assert_eq!(
            voronoi(&commands[1].command).output_class,
            ObjectClass::Global(4)
        );
        // A breakpoint on the component stops after it.
        assert_eq!(commands[0].step_id, StepId::UNASSIGNED);
        assert_eq!(commands[1].step_id, StepId(9));
    }

    #[test]
    fn disabled_inner_steps_are_left_out() {
        let mut component = cell_reconstruction();
        component.steps[0].enabled = false;
        let commands =
            expand(&component_step(9, "CELL_RECONSTRUCTION", &[]), &[component]).unwrap();
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0].command, PipelineCommand::Voronoi(_)));
    }

    #[test]
    fn components_nest_and_an_outer_parameter_can_set_an_inner_components_parameter() {
        let outer = definition(
            "OUTER",
            vec![
                component_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            // Sets the inner component's "Cell class" (its parameter 2).
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let commands = expand(
            &component_step(4, "OUTER", &[]),
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
    fn mistakes_are_reported_with_the_component_and_what_is_wrong() {
        let error = |definitions: &[ComponentDefinition], step: PipelineStepSettings| {
            expand(&step, definitions).unwrap_err().to_string()
        };

        let err = error(&[], component_step(1, "MISSING", &[]));
        assert!(
            err.contains("MISSING") && err.contains("isn't defined"),
            "{err}"
        );

        let looping = definition("LOOP", vec![component_step(1, "LOOP", &[])], vec![]);
        let err = error(&[looping], component_step(1, "LOOP", &[]));
        assert!(err.contains("contains itself"), "{err}");

        let mut unknown_step = cell_reconstruction();
        unknown_step.parameters[0].bindings[0].step = StepId(99);
        let err = error(
            &[unknown_step],
            component_step(1, "CELL_RECONSTRUCTION", &[]),
        );
        assert!(err.contains("step 99"), "{err}");

        let mut unknown_param = cell_reconstruction();
        unknown_param.parameters[0].bindings[0].param = "no_such".into();
        let err = error(
            &[unknown_param],
            component_step(1, "CELL_RECONSTRUCTION", &[]),
        );
        assert!(err.contains("no such parameter"), "{err}");

        // A value the parameter can't take is an error, not silently ignored.
        let bad_value = component_step(
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
    fn a_local_class_outside_a_component_is_refused() {
        let step = command_step(4, voronoi_with_center(ObjectClass::Local(1)));
        let error = expand(&step, &[]).unwrap_err().to_string();
        assert!(error.contains("component class L1"), "{error}");
    }

    #[test]
    fn extracting_into_a_local_class_outside_a_component_is_refused() {
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
        assert!(error.contains("component class L1"), "{error}");
    }

    #[test]
    fn a_component_may_only_use_the_local_classes_it_defines() {
        let undeclared = definition(
            "SEEDS",
            vec![command_step(1, voronoi_with_center(ObjectClass::Local(2)))],
            vec![],
        );
        let error = expand(&component_step(1, "SEEDS", &[]), &[undeclared])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("step 1 uses the component class L2"),
            "{error}"
        );

        // Declared, and set through a component parameter in its text form.
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
        let commands = expand(&component_step(1, "SEEDS", &[]), &[declared]).unwrap();
        assert_eq!(voronoi(&commands[0].command).centers, ObjectClass::Local(2));
    }

    fn scope(n: u32) -> Scope {
        Scope::Scoped(ScopeId(n))
    }

    #[test]
    fn each_component_instance_runs_in_a_scope_of_its_own() {
        let mut scopes = ScopeCounter::default();
        let plain = command_step(1, PipelineCommand::Blur(BlurSettings::default()));
        let component = component_step(2, "CELL_RECONSTRUCTION", &[]);
        let definitions = [cell_reconstruction()];

        let commands = expand_step(&plain, &definitions, &mut scopes).unwrap();
        assert_eq!(commands[0].scope, Scope::Global, "outside components");

        // Two instances of one component, as in two pipelines of a job.
        let first = expand_step(&component, &definitions, &mut scopes).unwrap();
        let second = expand_step(&component, &definitions, &mut scopes).unwrap();
        assert!(first.iter().all(|c| c.scope == scope(1)));
        assert!(second.iter().all(|c| c.scope == scope(2)));
    }

    #[test]
    fn a_nested_component_runs_in_the_scope_of_the_use_around_it() {
        let outer = definition(
            "OUTER",
            vec![
                component_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            vec![],
        );
        let mut scopes = ScopeCounter::default();
        let commands = expand_step(
            &component_step(4, "OUTER", &[]),
            &[cell_reconstruction(), outer],
            &mut scopes,
        )
        .unwrap();
        let scopes_used: Vec<Scope> = commands.iter().map(|c| c.scope).collect();
        assert_eq!(scopes_used, [scope(1), scope(1), scope(1)]);
        let uses: Vec<Vec<UseId>> = commands.iter().map(|c| c.uses.clone()).collect();
        assert_eq!(
            uses,
            [
                vec![UseId(1), UseId(2)],
                vec![UseId(1), UseId(2)],
                vec![UseId(1)]
            ]
        );
        assert_eq!(scopes.end_of(UseId(1)), Some(&UseEnd::Scope(scope(1))));
        assert_eq!(
            scopes.end_of(UseId(2)),
            Some(&UseEnd::Nested {
                scope: scope(1),
                classes: vec![],
                slots: vec![],
            })
        );
    }

    #[test]
    fn a_local_class_is_passed_into_a_nested_component_whose_own_are_numbered_apart() {
        // "Centers" is a parameter; L1 is the inner component's own class.
        let mut inner = definition(
            "INNER",
            vec![command_step(
                1,
                PipelineCommand::Voronoi(VoronoiSettings {
                    output_class: ObjectClass::Local(1),
                    ..Default::default()
                }),
            )],
            vec![parameter(1, ParamValue::Int(1), &[(1, "centers")])],
        );
        inner.classes = vec![local_class(1)];
        // The outer component's L1 as the inner "Centers"; L2 its own too.
        let mut outer = definition(
            "OUTER",
            vec![
                component_step(1, "INNER", &[]),
                command_step(2, voronoi_with_center(ObjectClass::Local(2))),
            ],
            vec![parameter(1, ParamValue::Text("L1".into()), &[(1, "1")])],
        );
        outer.classes = vec![local_class(1), local_class(2)];

        let mut scopes = ScopeCounter::default();
        let commands = expand_step(
            &component_step(4, "OUTER", &[]),
            &[inner, outer],
            &mut scopes,
        )
        .unwrap();
        let nested = voronoi(&commands[0].command);
        assert_eq!(nested.centers, ObjectClass::Local(1), "the outer one's L1");
        assert_eq!(
            nested.output_class,
            ObjectClass::Local(3),
            "the inner one's own L1, after the outer one's classes"
        );
        assert_eq!(voronoi(&commands[1].command).centers, ObjectClass::Local(2));
        assert!(commands.iter().all(|c| c.scope == scope(1)));
        assert_eq!(
            scopes.end_of(UseId(2)),
            Some(&UseEnd::Nested {
                scope: scope(1),
                classes: vec![ObjectClass::Local(3)],
                slots: vec![],
            }),
            "ending the nested use removes only its own class"
        );
    }

    #[test]
    fn two_uses_of_a_nested_component_get_classes_of_their_own() {
        let mut inner = definition(
            "INNER",
            vec![command_step(1, voronoi_with_center(ObjectClass::Local(1)))],
            vec![],
        );
        inner.classes = vec![local_class(1)];
        let outer = definition(
            "OUTER",
            vec![
                component_step(1, "INNER", &[]),
                component_step(2, "INNER", &[]),
            ],
            vec![],
        );
        let commands = expand(&component_step(4, "OUTER", &[]), &[inner, outer]).unwrap();
        let centers: Vec<ObjectClass> = commands
            .iter()
            .map(|c| voronoi(&c.command).centers)
            .collect();
        assert_eq!(centers, [ObjectClass::Local(1), ObjectClass::Local(2)]);
    }

    #[test]
    fn a_component_class_a_pipeline_hands_in_is_refused() {
        let error = expand(
            &component_step(
                1,
                "CELL_RECONSTRUCTION",
                &[(2, ParamValue::Text("L1".into()))],
            ),
            &[cell_reconstruction()],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("component class L1"), "{error}");
    }

    #[test]
    fn a_dataset_step_in_a_component_may_not_use_local_classes() {
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
        let error = expand(&component_step(1, "TRACK", &[]), &[tracker])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("dataset step using the component class L1"),
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

    /// Expands a component with local classes 1 and 2 running `command`.
    fn in_component(command: PipelineCommand) -> Result<Vec<ExpandedCommand>, InternalErrors> {
        let mut component = definition("MIX", vec![command_step(3, command)], vec![]);
        component.classes = vec![local_class(1), local_class(2)];
        expand(&component_step(1, "MIX", &[]), &[component])
    }

    #[test]
    fn adding_a_class_of_the_other_kind_to_existing_objects_is_refused() {
        use ObjectClassifyObjectsClassifyMatchHandlingSettings::*;
        let local = ObjectClass::Local(1);
        let global = ObjectClass::Global(5);

        let error = in_component(classify(vec![local], AddOutputClassIfMatch, global))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("step 3 adds the project class 5 to objects of the component class L1")
                && error.contains("Reclassify"),
            "{error}"
        );
        let error = in_component(classify(vec![global], AddOutputClassIfNotMatch, local))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("adds the component class L1 to objects of the project class 5"),
            "{error}"
        );
        // No input classes: every object the command sees, the project's too.
        let error = in_component(classify(vec![], AddOutputClassIfMatch, local))
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
            assert!(in_component(command).is_ok());
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
        let error = in_component(ai(
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch,
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("adds the project class 7"), "{error}");
        assert!(
            in_component(ai(
                ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch
            ))
            .is_ok()
        );
    }

    // ---- Component memory slots ----

    fn store_in(address: ImageAddress) -> PipelineCommand {
        PipelineCommand::ImageCache(ImageCacheSettings {
            mode: MathImageCacheImageCacheModeSettings::Store,
            address,
        })
    }

    #[test]
    fn a_component_memory_slot_outside_a_component_is_refused() {
        let component_slot = ImageAddress::Memory(MemoryId::ComponentContext(2));
        let error = expand(&command_step(1, store_in(component_slot)), &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("component memory slot 2"), "{error}");

        let component = definition(
            "SLOTS",
            vec![command_step(1, store_in(component_slot))],
            vec![],
        );
        assert!(expand(&component_step(1, "SLOTS", &[]), &[component]).is_ok());
    }

    #[test]
    fn a_component_memory_slot_is_passed_into_a_nested_component() {
        // Stores in its "address"; slot 1 also its own.
        let inner = definition(
            "INNER",
            vec![
                command_step(
                    1,
                    store_in(ImageAddress::Memory(MemoryId::PipelineContext(1))),
                ),
                command_step(
                    2,
                    store_in(ImageAddress::Memory(MemoryId::ComponentContext(1))),
                ),
            ],
            vec![parameter(
                1,
                ParamValue::Text("memory:1".into()),
                &[(1, "address")],
            )],
        );
        let outer = definition(
            "OUTER",
            vec![component_step(1, "INNER", &[])],
            vec![parameter(
                1,
                ParamValue::Text("component:1".into()),
                &[(1, "1")],
            )],
        );
        let mut scopes = ScopeCounter::default();
        let commands = expand_step(
            &component_step(4, "OUTER", &[]),
            &[inner, outer],
            &mut scopes,
        )
        .unwrap();
        let address = |command: &PipelineCommand| match command {
            PipelineCommand::ImageCache(settings) => settings.address,
            _ => panic!("image cache"),
        };
        assert_eq!(
            address(&commands[0].command),
            ImageAddress::Memory(MemoryId::ComponentContext(1)),
            "the outer one's slot"
        );
        assert_eq!(
            address(&commands[1].command),
            ImageAddress::Memory(MemoryId::ComponentContext(2)),
            "the inner one's own, after the outer one's"
        );
        assert_eq!(
            scopes.end_of(UseId(2)),
            Some(&UseEnd::Nested {
                scope: scope(1),
                classes: vec![],
                slots: vec![2],
            })
        );
    }

    // ---- The component card's parameters ----

    fn instance(key: &str, values: &[(u32, ParamValue)]) -> ComponentInstance {
        let StepContent::Component(instance) = component_step(1, key, values).content else {
            unreachable!()
        };
        instance
    }

    #[test]
    fn a_component_parameter_shows_as_the_parameter_it_is_bound_to() {
        let definitions = [cell_reconstruction()];
        let shown = component_parameters(
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
        assert_eq!(cell_class.default_value, "1", "the component's default");
    }

    #[test]
    fn setting_a_component_parameter_stores_what_the_command_took() {
        let definitions = [cell_reconstruction()];
        let mut cells = instance("CELL_RECONSTRUCTION", &[]);

        set_component_parameter(&mut cells, &definitions, &ComponentParamKey::from("1"), "5")
            .unwrap();
        set_component_parameter(
            &mut cells,
            &definitions,
            &ComponentParamKey::from("2"),
            "L1",
        )
        .unwrap();
        assert_eq!(
            cells.values[&ComponentParamKey::from("1")],
            ParamValue::Int(5)
        );
        assert_eq!(
            cells.values[&ComponentParamKey::from("2")],
            ParamValue::Text("L1".into())
        );

        // Text the command doesn't take leaves the value as it was.
        set_component_parameter(
            &mut cells,
            &definitions,
            &ComponentParamKey::from("1"),
            "big",
        )
        .unwrap();
        assert_eq!(
            cells.values[&ComponentParamKey::from("1")],
            ParamValue::Int(5)
        );

        assert!(
            set_component_parameter(&mut cells, &definitions, &ComponentParamKey::from("9"), "1")
                .is_err()
        );
    }

    #[test]
    fn a_parameter_bound_to_a_nested_components_parameter_works_the_same() {
        let outer = definition(
            "OUTER",
            vec![component_step(1, "CELL_RECONSTRUCTION", &[])],
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let definitions = [cell_reconstruction(), outer];
        let mut outer = instance("OUTER", &[]);

        let shown = component_parameters(&outer, &definitions).unwrap();
        assert_eq!(shown[0].value, "7");

        set_component_parameter(&mut outer, &definitions, &ComponentParamKey::from("1"), "8")
            .unwrap();
        assert_eq!(
            outer.values[&ComponentParamKey::from("1")],
            ParamValue::Int(8)
        );
        let commands = expand(
            &component_step(1, "OUTER", &[(1, ParamValue::Int(8))]),
            &definitions,
        )
        .unwrap();
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(8)
        );
    }
}
