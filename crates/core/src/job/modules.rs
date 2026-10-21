//! Expanding modules into plain commands before a job runs.
//!
//! A pipeline step is a command or an instance of a module (see
//! `evanalyzer_cfg::settings::modules`). The executor only knows
//! commands, so the job generator replaces each module instance by its
//! definition's commands, with the instance's parameter values written into
//! the inner parameters they're bound to.

use crate::pipeline::scope::{Scope, ScopeId};
use evanalyzer_cfg::{
    core_types::{InternalErrors, MemorySlot, ModuleParamKey, ObjectClass, StepId},
    settings::{
        modules::{ModuleDefinition, ModuleInstance, ModuleParameterDef, ParamBinding, ParamValue},
        parameter_def::{ParamType, ParameterDef},
        pipeline_command::{CommandCategory, PipelineCommand},
        pipeline_command_settings::{
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings,
            ObjectClassifyObjectsClassifyMatchHandlingSettings,
        },
        pipeline_settings::{PipelineStepSettings, StepContent},
    },
};
use std::collections::{BTreeMap, BTreeSet};

/// How deep modules may nest - deeper is almost certainly a mistake.
const MAX_MODULE_DEPTH: usize = 8;

/// One command a pipeline step runs.
#[derive(Debug, Clone)]
pub struct ExpandedCommand {
    /// What a breakpoint targets - see [`expand_step`].
    pub step_id: StepId,
    /// The scope the command runs in: its module instance's, else `Global`.
    pub scope: Scope,
    /// The module uses the command belongs to, outermost first - each
    /// ends after its last command (see [`ScopeCounter::end_of`]).
    pub uses: Vec<UseId>,
    pub command: PipelineCommand,
}

/// One use of a module in a job, numbered at expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UseId(pub u32);

/// What goes when a module use ends.
#[derive(Debug, Clone, PartialEq)]
pub enum UseEnd {
    /// A use in a pipeline: its scope - its objects and memory slots.
    Scope(Scope),
    /// A use inside another module: it runs in the scope of the use in the
    /// pipeline around it (so it can work on that one's classes), with
    /// classes and memory slots of its own numbered apart - those go.
    Nested {
        scope: Scope,
        classes: Vec<ObjectClass>,
        slots: Vec<MemorySlot>,
    },
}

/// Numbers the module uses of a job - one counter per job, so no two
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
/// modules nested in it: they run in its scope, so each gets numbers no
/// other class or slot of the scope has.
struct Frame {
    scope: Scope,
    next_class: u32,
    next_slot: MemorySlot,
}

impl Frame {
    /// For a use of `definition` in a pipeline, in `scope`: the numbers after
    /// its own.
    fn new(scope: Scope, definition: &ModuleDefinition) -> Self {
        let (classes, slots) = own_numbers(definition);
        Self {
            scope,
            next_class: classes.last().map_or(1, |n| n + 1),
            next_slot: slots.last().map_or(1, |n| n + 1),
        }
    }
}

/// The commands `step` runs, in order, each with the id a breakpoint
/// targets, the scope it runs in and the module uses it belongs to: a
/// command step is itself, in the global scope; a module is its
/// definition's enabled commands with the instance's values applied, in a
/// new scope from `scopes` - a module nested in it in that scope too, its
/// own classes and memory slots numbered apart (see [`UseEnd::Nested`]).
/// The last command carries the module step's id (a breakpoint on a
/// module stops after the whole module), the others
/// [`StepId::UNASSIGNED`] (inner steps can't be targeted).
pub fn expand_step(
    step: &PipelineStepSettings,
    definitions: &[ModuleDefinition],
    scopes: &mut ScopeCounter,
) -> Result<Vec<ExpandedCommand>, InternalErrors> {
    match &step.content {
        StepContent::Command(command) => {
            if let Some(class) = command.local_classes().first() {
                return Err(only_inside_class(*class));
            }
            if let Some(slot) = module_memory_slots(command).first() {
                return Err(only_inside_slot(*slot));
            }
            Ok(vec![ExpandedCommand {
                step_id: step.id,
                scope: Scope::Global,
                uses: Vec::new(),
                command: command.clone(),
            }])
        }
        StepContent::Module(instance) => {
            // A pipeline has no module classes or memory to hand in.
            for value in instance.values.values() {
                if let Some(class) = local_class_in(value) {
                    return Err(only_inside_class(class));
                }
                if let Some(slot) = module_slot_in(value) {
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
        "uses the module class {}, which only exists inside a module",
        class.to_text()
    ))
}

fn only_inside_slot(slot: u64) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "uses the module memory slot {slot}, which only exists inside a module"
    ))
}

/// The commands of `instance`'s definition with its values applied, each
/// with the scope it runs in and the uses it belongs to. `around` is the
/// frame of the use in the pipeline this one is nested in - `None` for that
/// use itself, which gets a new scope from `scopes`. `open` holds the keys
/// of the modules being expanded around this one, `uses` their uses.
fn expand_instance(
    instance: &ModuleInstance,
    definitions: &[ModuleDefinition],
    scopes: &mut ScopeCounter,
    around: Option<&mut Frame>,
    open: &mut Vec<String>,
    uses: &mut Vec<UseId>,
) -> Result<Vec<(Scope, Vec<UseId>, PipelineCommand)>, InternalErrors> {
    let raw = definition_of(instance, definitions)?;
    if open.contains(&instance.key) {
        return Err(module_error(&instance.key, "contains itself".into()));
    }
    if open.len() >= MAX_MODULE_DEPTH {
        return Err(module_error(
            &instance.key,
            format!("is nested deeper than {MAX_MODULE_DEPTH} modules"),
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
                        return Err(module_error(
                            &definition.key,
                            format!(
                                "step {} is a dataset step using the module class {} - dataset \
                                 steps run on stored objects, which only have project classes",
                                step.id,
                                class.to_text()
                            ),
                        ));
                    }
                }
                if let Some(why) = mixed_classes(command) {
                    return Err(module_error(
                        &definition.key,
                        format!("step {} {why}", step.id),
                    ));
                }
                commands.push((frame.scope, uses.clone(), command.clone()))
            }
            StepContent::Module(inner) => commands.extend(expand_instance(
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

/// Refuses a module class `definition` uses without defining it - in a
/// command, a value it gives a module inside it, or a parameter's default.
fn check_own_classes(definition: &ModuleDefinition) -> Result<(), InternalErrors> {
    let defined = |class: &ObjectClass| definition.classes.iter().any(|c| c.local() == *class);
    for step in &definition.steps {
        let used: Vec<ObjectClass> = match &step.content {
            StepContent::Command(command) => command.local_classes(),
            StepContent::Module(inner) => {
                inner.values.values().filter_map(local_class_in).collect()
            }
        };
        if let Some(class) = used.into_iter().find(|c| !defined(c)) {
            return Err(module_error(
                &definition.key,
                format!(
                    "step {} uses the module class {}, which the module doesn't define",
                    step.id,
                    class.to_text()
                ),
            ));
        }
    }
    for parameter in &definition.parameters {
        if let Some(class) = local_class_in(&parameter.default).filter(|c| !defined(c)) {
            return Err(module_error(
                &definition.key,
                format!(
                    "parameter \"{}\" starts with the module class {}, which the module \
                     doesn't define",
                    parameter.label,
                    class.to_text()
                ),
            ));
        }
    }
    Ok(())
}

/// The module classes (their numbers) and memory slots `definition` has:
/// the classes it defines, the slots its steps and parameters use - sorted.
pub fn own_numbers(definition: &ModuleDefinition) -> (Vec<u32>, Vec<MemorySlot>) {
    let classes: BTreeSet<u32> = definition.classes.iter().map(|c| c.id).collect();
    let mut slots: BTreeSet<MemorySlot> = BTreeSet::new();
    for step in &definition.steps {
        match &step.content {
            StepContent::Command(command) => slots.extend(
                module_memory_slots(command)
                    .into_iter()
                    .map(|s| s as MemorySlot),
            ),
            StepContent::Module(inner) => {
                slots.extend(inner.values.values().filter_map(module_slot_in))
            }
        }
    }
    slots.extend(
        definition
            .parameters
            .iter()
            .filter_map(|p| module_slot_in(&p.default)),
    );
    (classes.into_iter().collect(), slots.into_iter().collect())
}

/// `definition` with its classes and memory slots numbered from `frame`'s
/// next free ones - and the classes and slots they became.
fn renumber(
    definition: &ModuleDefinition,
    frame: &mut Frame,
) -> (ModuleDefinition, Vec<ObjectClass>, Vec<MemorySlot>) {
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

/// `definition` with its module classes and memory slots renumbered as
/// `classes` and `slots` say (old → new) - wherever they're written: its
/// classes, commands, the values it gives modules inside it, its
/// parameters' defaults.
pub fn renumbered(
    definition: &ModuleDefinition,
    classes: &BTreeMap<u32, u32>,
    slots: &BTreeMap<MemorySlot, MemorySlot>,
) -> ModuleDefinition {
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
            StepContent::Module(inner) => {
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

/// Renumbers the module classes (`{"LOCAL": n}`) and memory slots
/// (`{"moduleContext": n}`) of a command in its file form.
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
                    "moduleContext" => Some(slots),
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

/// Renumbers the module classes ("L1") and memory slots ("module:1")
/// of a module parameter's value.
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
            } else if let Some(n) = module_slot_text(text)
                && let Some(new) = slots.get(&n)
            {
                *text = format!("module:{new}");
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
    instance: &ModuleInstance,
    definition: &ModuleDefinition,
) -> Result<Vec<PipelineStepSettings>, InternalErrors> {
    let mut steps = definition.steps.clone();
    // A class list several parameters set gets all their classes (in
    // parameter order, each once): (step, setting) → classes.
    let mut lists: Vec<((StepId, String), Vec<String>)> = Vec::new();
    for parameter in &definition.parameters {
        let value = own_value(instance, parameter);
        for binding in &parameter.bindings {
            if is_class_list(&definition.steps, binding) {
                let target = (binding.step, binding.param.clone());
                let index = match lists.iter().position(|(t, _)| *t == target) {
                    Some(index) => index,
                    None => {
                        lists.push((target, Vec::new()));
                        lists.len() - 1
                    }
                };
                for class in value_text(value).split(',').map(str::trim) {
                    let class = class.to_string();
                    if !class.is_empty()
                        && class != ObjectClass::Unset.to_text()
                        && !lists[index].1.contains(&class)
                    {
                        lists[index].1.push(class);
                    }
                }
                continue;
            }
            let step = steps
                .iter_mut()
                .find(|s| s.id == binding.step)
                .ok_or_else(|| {
                    module_error(
                        &definition.key,
                        format!(
                            "parameter \"{}\" is bound to step {}, which the module doesn't have",
                            parameter.label, binding.step
                        ),
                    )
                })?;
            set_bound_value(step, &binding.param, value).map_err(|why| {
                module_error(
                    &definition.key,
                    format!(
                        "parameter \"{}\" can't set {} of step {}: {why}",
                        parameter.label, binding.param, binding.step
                    ),
                )
            })?;
        }
    }
    for ((step_id, path), classes) in lists {
        let step = steps
            .iter_mut()
            .find(|s| s.id == step_id)
            .expect("found by is_class_list");
        set_bound_value(step, &path, &ParamValue::Text(classes.join(","))).map_err(|why| {
            module_error(
                &definition.key,
                format!("its parameters can't set {path} of step {step_id}: {why}"),
            )
        })?;
    }
    Ok(steps)
}

/// The value `instance` gives `parameter`: its own, else the default.
fn own_value<'a>(
    instance: &'a ModuleInstance,
    parameter: &'a ModuleParameterDef,
) -> &'a ParamValue {
    instance
        .values
        .get(&parameter.key)
        .unwrap_or(&parameter.default)
}

/// Whether `binding` sets a class list of a command among `steps` - a
/// setting every parameter bound to it adds its classes to.
fn is_class_list(steps: &[PipelineStepSettings], binding: &ParamBinding) -> bool {
    steps
        .iter()
        .find(|s| s.id == binding.step)
        .and_then(|s| s.command())
        .and_then(|command| {
            find_param(&crate::algos::command_parameters(command), &binding.param).map(|p| {
                matches!(
                    p.param_type,
                    ParamType::MultiObjClass | ParamType::MultiSegClass
                )
            })
        })
        .unwrap_or(false)
}

/// `instance`'s module's steps with the instance's values applied - what
/// "Expand module" puts into the pipeline in its place.
pub fn module_steps_with_values(
    instance: &ModuleInstance,
    definitions: &[ModuleDefinition],
) -> Result<Vec<PipelineStepSettings>, InternalErrors> {
    steps_with_values(instance, definition_of(instance, definitions)?)
}

/// The parameters a module card shows for `instance`: each module parameter
/// as the inner parameter it's bound to (its first binding) - same type,
/// options and range, so the card gets the same control as the command -
/// with the instance's value, the module parameter's id as `name`, its label
/// and description, and its default.
pub fn module_parameters(
    instance: &ModuleInstance,
    definitions: &[ModuleDefinition],
) -> Result<Vec<ParameterDef>, InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let steps = steps_with_values(instance, definition)?;
    definition
        .parameters
        .iter()
        .map(|parameter| {
            let binding = parameter.bindings.first().ok_or_else(|| {
                module_error(
                    &definition.key,
                    format!("parameter \"{}\" isn't bound to anything", parameter.label),
                )
            })?;
            let step = steps
                .iter()
                .find(|s| s.id == binding.step)
                .ok_or_else(|| missing_step(definition, parameter, binding))?;
            let bound = match &step.content {
                // A class list shows the parameter's own classes - not those
                // other parameters add to it.
                StepContent::Command(command) if is_class_list(&definition.steps, binding) => {
                    let mut own = command.clone();
                    crate::algos::apply_command_param(
                        &mut own,
                        &binding.param,
                        &value_text(own_value(instance, parameter)),
                    );
                    find_param(&crate::algos::command_parameters(&own), &binding.param).cloned()
                }
                StepContent::Command(command) => {
                    find_param(&crate::algos::command_parameters(command), &binding.param).cloned()
                }
                StepContent::Module(inner) => module_parameters(inner, definitions)?
                    .into_iter()
                    .find(|p| p.name == binding.param),
            };
            let mut shown = bound.ok_or_else(|| {
                module_error(
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

/// Sets `instance`'s value of module parameter `key` from `text` - a value as
/// the card's control reports it (`apply_param_change`'s format, e.g. a
/// dropdown label or "toggle:3"). It's applied to a copy of the bound
/// command and read back, so the value is the one the command took.
pub fn set_module_parameter(
    instance: &mut ModuleInstance,
    definitions: &[ModuleDefinition],
    key: &ModuleParamKey,
    text: &str,
) -> Result<(), InternalErrors> {
    let definition = definition_of(instance, definitions)?;
    let parameter = definition
        .parameters
        .iter()
        .find(|p| &p.key == key)
        .ok_or_else(|| module_error(&definition.key, format!("has no parameter {key}")))?;
    let binding = parameter.bindings.first().ok_or_else(|| {
        module_error(
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
            // From the parameter's own value - a class list may hold other
            // parameters' classes too.
            crate::algos::apply_command_param(
                command,
                &binding.param,
                &value_text(own_value(instance, parameter)),
            );
            crate::algos::apply_command_param(command, &binding.param, text);
            let now = find_param(&crate::algos::command_parameters(command), &binding.param)
                .map(|p| p.value.clone())
                .unwrap_or_default();
            value_from_text(&now)
        }
        StepContent::Module(inner) => {
            let inner_key = ModuleParamKey::from(binding.param.as_str());
            set_module_parameter(inner, definitions, &inner_key, text)?;
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
    instance: &ModuleInstance,
    definitions: &'a [ModuleDefinition],
) -> Result<&'a ModuleDefinition, InternalErrors> {
    definitions
        .iter()
        .find(|d| d.key == instance.key)
        .ok_or_else(|| module_error(&instance.key, "isn't defined in this project".into()))
}

fn missing_step(
    definition: &ModuleDefinition,
    parameter: &ModuleParameterDef,
    binding: &ParamBinding,
) -> InternalErrors {
    module_error(
        &definition.key,
        format!(
            "parameter \"{}\" is bound to step {}, which the module doesn't have",
            parameter.label, binding.step
        ),
    )
}

/// A parameter's text value (`ParameterDef::value`) as a typed module value -
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
                    "adds the module class {} to all objects, also the project's - an object \
                     can't have both; select the module classes it applies to",
                    output.to_text()
                ));
            }
            continue;
        }
        for input in inputs.iter().filter(|c| **c != ObjectClass::Unset) {
            if is_local(input) != is_local(output) {
                let (adding, to) = if is_local(output) {
                    ("module class", "project class")
                } else {
                    ("project class", "module class")
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
/// for a nested module, the value of its module parameter `param` (its id).
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
        StepContent::Module(inner) => {
            // A module class or memory slot too: the nested module
            // runs in this one's scope (see `UseEnd::Nested`).
            let key = ModuleParamKey::from(param);
            inner.values.insert(key, value.clone());
            Ok(())
        }
    }
}

/// The module memory slots (`MemoryId::ModuleContext`) `command` reads or writes.
/// Found in its file form, where such a slot is always `{"memory":
/// {"moduleContext": n}}` - whichever setting holds it (an image address, a
/// threshold's value source, ...).
fn module_memory_slots(command: &PipelineCommand) -> Vec<u64> {
    fn collect(value: &serde_json::Value, slots: &mut Vec<u64>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(slot) = map
                    .get("memory")
                    .and_then(|memory| memory.get("moduleContext"))
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

/// The first module memory slot in `value`, in its text form
/// (`"module:2"`), if any.
fn module_slot_in(value: &ParamValue) -> Option<MemorySlot> {
    match value {
        ParamValue::Text(text) => module_slot_text(text),
        ParamValue::List(items) => items.iter().find_map(module_slot_in),
        _ => None,
    }
}

/// The slot of a module memory slot's text form (`"module:2"`).
fn module_slot_text(text: &str) -> Option<MemorySlot> {
    text.trim().strip_prefix("module:")?.trim().parse().ok()
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

fn module_error(key: &str, what: String) -> InternalErrors {
    InternalErrors::InvalidArgument(format!("Module {key}: {what}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ObjectClass;
    use evanalyzer_cfg::core_types::SegmentationClass;
    use evanalyzer_cfg::core_types::{ImageAddress, MemoryId};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::modules::{ModuleParameterDef, ParamBinding};
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

    fn module_step(id: u32, key: &str, values: &[(u32, ParamValue)]) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Module(ModuleInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (ModuleParamKey(k.to_string()), v.clone()))
                    .collect::<BTreeMap<_, _>>(),
            }),
        }
    }

    fn parameter(id: u32, default: ParamValue, bindings: &[(u32, &str)]) -> ModuleParameterDef {
        ModuleParameterDef {
            key: ModuleParamKey(id.to_string()),
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
        parameters: Vec<ModuleParameterDef>,
    ) -> ModuleDefinition {
        ModuleDefinition {
            key: key.into(),
            meta: Default::default(),
            classes: vec![],
            steps,
            parameters,
            origin: None,
        }
    }

    /// Blur, then Voronoi - "Cell class" set in both places.
    fn cell_reconstruction() -> ModuleDefinition {
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
        definitions: &[ModuleDefinition],
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
    fn a_module_runs_its_commands_with_the_instances_values_and_defaults() {
        let step = module_step(9, "CELL_RECONSTRUCTION", &[(2, ParamValue::Int(4))]);
        let commands = expand(&step, &[cell_reconstruction()]).unwrap();

        assert_eq!(commands.len(), 2);
        let PipelineCommand::Blur(blur) = &commands[0].command else {
            panic!("expected Blur");
        };
        assert_eq!(blur.kernel_size, 3, "the definition's default");
        // One module parameter, two inner parameters - always the same value.
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(4)
        );
        assert_eq!(
            voronoi(&commands[1].command).output_class,
            ObjectClass::Global(4)
        );
        // A breakpoint on the module stops after it.
        assert_eq!(commands[0].step_id, StepId::UNASSIGNED);
        assert_eq!(commands[1].step_id, StepId(9));
    }

    #[test]
    fn disabled_inner_steps_are_left_out() {
        let mut module = cell_reconstruction();
        module.steps[0].enabled = false;
        let commands = expand(&module_step(9, "CELL_RECONSTRUCTION", &[]), &[module]).unwrap();
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0].command, PipelineCommand::Voronoi(_)));
    }

    #[test]
    fn modules_nest_and_an_outer_parameter_can_set_an_inner_modules_parameter() {
        let outer = definition(
            "OUTER",
            vec![
                module_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            // Sets the inner module's "Cell class" (its parameter 2).
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let commands = expand(
            &module_step(4, "OUTER", &[]),
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
    fn mistakes_are_reported_with_the_module_and_what_is_wrong() {
        let error = |definitions: &[ModuleDefinition], step: PipelineStepSettings| {
            expand(&step, definitions).unwrap_err().to_string()
        };

        let err = error(&[], module_step(1, "MISSING", &[]));
        assert!(
            err.contains("MISSING") && err.contains("isn't defined"),
            "{err}"
        );

        let looping = definition("LOOP", vec![module_step(1, "LOOP", &[])], vec![]);
        let err = error(&[looping], module_step(1, "LOOP", &[]));
        assert!(err.contains("contains itself"), "{err}");

        let mut unknown_step = cell_reconstruction();
        unknown_step.parameters[0].bindings[0].step = StepId(99);
        let err = error(&[unknown_step], module_step(1, "CELL_RECONSTRUCTION", &[]));
        assert!(err.contains("step 99"), "{err}");

        let mut unknown_param = cell_reconstruction();
        unknown_param.parameters[0].bindings[0].param = "no_such".into();
        let err = error(&[unknown_param], module_step(1, "CELL_RECONSTRUCTION", &[]));
        assert!(err.contains("no such parameter"), "{err}");

        // A value the parameter can't take is an error, not silently ignored.
        let bad_value = module_step(
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
    fn a_local_class_outside_a_module_is_refused() {
        let step = command_step(4, voronoi_with_center(ObjectClass::Local(1)));
        let error = expand(&step, &[]).unwrap_err().to_string();
        assert!(error.contains("module class L1"), "{error}");
    }

    #[test]
    fn extracting_into_a_local_class_outside_a_module_is_refused() {
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
        assert!(error.contains("module class L1"), "{error}");
    }

    #[test]
    fn a_module_may_only_use_the_local_classes_it_defines() {
        let undeclared = definition(
            "SEEDS",
            vec![command_step(1, voronoi_with_center(ObjectClass::Local(2)))],
            vec![],
        );
        let error = expand(&module_step(1, "SEEDS", &[]), &[undeclared])
            .unwrap_err()
            .to_string();
        assert!(error.contains("step 1 uses the module class L2"), "{error}");

        // Declared, and set through a module parameter in its text form.
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
        let commands = expand(&module_step(1, "SEEDS", &[]), &[declared]).unwrap();
        assert_eq!(voronoi(&commands[0].command).centers, ObjectClass::Local(2));
    }

    fn scope(n: u32) -> Scope {
        Scope::Scoped(ScopeId(n))
    }

    #[test]
    fn each_module_instance_runs_in_a_scope_of_its_own() {
        let mut scopes = ScopeCounter::default();
        let plain = command_step(1, PipelineCommand::Blur(BlurSettings::default()));
        let module = module_step(2, "CELL_RECONSTRUCTION", &[]);
        let definitions = [cell_reconstruction()];

        let commands = expand_step(&plain, &definitions, &mut scopes).unwrap();
        assert_eq!(commands[0].scope, Scope::Global, "outside modules");

        // Two instances of one module, as in two pipelines of a job.
        let first = expand_step(&module, &definitions, &mut scopes).unwrap();
        let second = expand_step(&module, &definitions, &mut scopes).unwrap();
        assert!(first.iter().all(|c| c.scope == scope(1)));
        assert!(second.iter().all(|c| c.scope == scope(2)));
    }

    /// One parameter can set a single class (one value) and add its class to a
    /// class list several parameters set - the list gets all their classes.
    #[test]
    fn a_class_list_several_parameters_set_gets_all_their_classes() {
        let classify = |output: ObjectClass, inputs: Vec<ObjectClass>| {
            PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                input_classes: inputs,
                output_class: output,
                ..Default::default()
            })
        };
        let definition = definition(
            "COLOC",
            vec![
                command_step(1, classify(ObjectClass::Global(1), vec![])),
                command_step(2, classify(ObjectClass::Global(2), vec![])),
                command_step(
                    3,
                    classify(ObjectClass::Unset, vec![ObjectClass::Global(9)]),
                ),
            ],
            vec![
                parameter(
                    1,
                    ParamValue::Int(4),
                    &[(1, "output_class"), (3, "input_classes")],
                ),
                parameter(
                    2,
                    ParamValue::Int(5),
                    &[(2, "output_class"), (3, "input_classes")],
                ),
                // Only on the list, its own classes.
                parameter(3, ParamValue::Int(7), &[(3, "input_classes")]),
            ],
        );
        let definitions = [definition];
        let classes = |commands: &[ExpandedCommand]| -> Vec<(ObjectClass, Vec<ObjectClass>)> {
            commands
                .iter()
                .map(|c| match &c.command {
                    PipelineCommand::ClassifyObjects(s) => {
                        (s.output_class, s.input_classes.clone())
                    }
                    _ => panic!("classify"),
                })
                .collect()
        };

        let commands = expand(
            &module_step(1, "COLOC", &[(2, ParamValue::Int(6))]),
            &definitions,
        )
        .unwrap();
        let g = ObjectClass::Global;
        assert_eq!(
            classes(&commands),
            [
                (g(4), vec![]),
                (g(6), vec![]),
                (ObjectClass::Unset, vec![g(4), g(6), g(7)])
            ],
            "single classes one value each, the list all of them in parameter order"
        );

        // Each card shows the parameter's own value, not the list's.
        let mut instance = instance("COLOC", &[(2, ParamValue::Int(6))]);
        let shown = module_parameters(&instance, &definitions).unwrap();
        assert_eq!(shown[2].value, "7");

        // Changing one changes only its part of the list.
        set_module_parameter(&mut instance, &definitions, &ModuleParamKey::from("1"), "8").unwrap();
        let commands = expand_step(
            &PipelineStepSettings {
                id: StepId(1),
                enabled: true,
                content: StepContent::Module(instance),
            },
            &definitions,
            &mut ScopeCounter::default(),
        )
        .unwrap();
        assert_eq!(classes(&commands)[2].1, [g(8), g(6), g(7)]);
    }

    #[test]
    fn a_nested_module_runs_in_the_scope_of_the_use_around_it() {
        let outer = definition(
            "OUTER",
            vec![
                module_step(1, "CELL_RECONSTRUCTION", &[]),
                command_step(2, PipelineCommand::Blur(BlurSettings::default())),
            ],
            vec![],
        );
        let mut scopes = ScopeCounter::default();
        let commands = expand_step(
            &module_step(4, "OUTER", &[]),
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
    fn a_local_class_is_passed_into_a_nested_module_whose_own_are_numbered_apart() {
        // "Centers" is a parameter; L1 is the inner module's own class.
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
        // The outer module's L1 as the inner "Centers"; L2 its own too.
        let mut outer = definition(
            "OUTER",
            vec![
                module_step(1, "INNER", &[]),
                command_step(2, voronoi_with_center(ObjectClass::Local(2))),
            ],
            vec![parameter(1, ParamValue::Text("L1".into()), &[(1, "1")])],
        );
        outer.classes = vec![local_class(1), local_class(2)];

        let mut scopes = ScopeCounter::default();
        let commands =
            expand_step(&module_step(4, "OUTER", &[]), &[inner, outer], &mut scopes).unwrap();
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
    fn two_uses_of_a_nested_module_get_classes_of_their_own() {
        let mut inner = definition(
            "INNER",
            vec![command_step(1, voronoi_with_center(ObjectClass::Local(1)))],
            vec![],
        );
        inner.classes = vec![local_class(1)];
        let outer = definition(
            "OUTER",
            vec![module_step(1, "INNER", &[]), module_step(2, "INNER", &[])],
            vec![],
        );
        let commands = expand(&module_step(4, "OUTER", &[]), &[inner, outer]).unwrap();
        let centers: Vec<ObjectClass> = commands
            .iter()
            .map(|c| voronoi(&c.command).centers)
            .collect();
        assert_eq!(centers, [ObjectClass::Local(1), ObjectClass::Local(2)]);
    }

    #[test]
    fn a_module_class_a_pipeline_hands_in_is_refused() {
        let error = expand(
            &module_step(
                1,
                "CELL_RECONSTRUCTION",
                &[(2, ParamValue::Text("L1".into()))],
            ),
            &[cell_reconstruction()],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("module class L1"), "{error}");
    }

    #[test]
    fn a_dataset_step_in_a_module_may_not_use_local_classes() {
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
        let error = expand(&module_step(1, "TRACK", &[]), &[tracker])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("dataset step using the module class L1"),
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

    /// Expands a module with local classes 1 and 2 running `command`.
    fn in_module(command: PipelineCommand) -> Result<Vec<ExpandedCommand>, InternalErrors> {
        let mut module = definition("MIX", vec![command_step(3, command)], vec![]);
        module.classes = vec![local_class(1), local_class(2)];
        expand(&module_step(1, "MIX", &[]), &[module])
    }

    #[test]
    fn adding_a_class_of_the_other_kind_to_existing_objects_is_refused() {
        use ObjectClassifyObjectsClassifyMatchHandlingSettings::*;
        let local = ObjectClass::Local(1);
        let global = ObjectClass::Global(5);

        let error = in_module(classify(vec![local], AddOutputClassIfMatch, global))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("step 3 adds the project class 5 to objects of the module class L1")
                && error.contains("Reclassify"),
            "{error}"
        );
        let error = in_module(classify(vec![global], AddOutputClassIfNotMatch, local))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("adds the module class L1 to objects of the project class 5"),
            "{error}"
        );
        // No input classes: every object the command sees, the project's too.
        let error = in_module(classify(vec![], AddOutputClassIfMatch, local))
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
            assert!(in_module(command).is_ok());
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
        let error = in_module(ai(
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch,
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("adds the project class 7"), "{error}");
        assert!(
            in_module(ai(
                ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch
            ))
            .is_ok()
        );
    }

    // ---- Module memory slots ----

    fn store_in(address: ImageAddress) -> PipelineCommand {
        PipelineCommand::ImageCache(ImageCacheSettings {
            mode: MathImageCacheImageCacheModeSettings::Store,
            address,
        })
    }

    #[test]
    fn a_module_memory_slot_outside_a_module_is_refused() {
        let module_slot = ImageAddress::Memory(MemoryId::ModuleContext(2));
        let error = expand(&command_step(1, store_in(module_slot)), &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("module memory slot 2"), "{error}");

        let module = definition(
            "SLOTS",
            vec![command_step(1, store_in(module_slot))],
            vec![],
        );
        assert!(expand(&module_step(1, "SLOTS", &[]), &[module]).is_ok());
    }

    #[test]
    fn a_module_memory_slot_is_passed_into_a_nested_module() {
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
                    store_in(ImageAddress::Memory(MemoryId::ModuleContext(1))),
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
            vec![module_step(1, "INNER", &[])],
            vec![parameter(
                1,
                ParamValue::Text("module:1".into()),
                &[(1, "1")],
            )],
        );
        let mut scopes = ScopeCounter::default();
        let commands =
            expand_step(&module_step(4, "OUTER", &[]), &[inner, outer], &mut scopes).unwrap();
        let address = |command: &PipelineCommand| match command {
            PipelineCommand::ImageCache(settings) => settings.address,
            _ => panic!("image cache"),
        };
        assert_eq!(
            address(&commands[0].command),
            ImageAddress::Memory(MemoryId::ModuleContext(1)),
            "the outer one's slot"
        );
        assert_eq!(
            address(&commands[1].command),
            ImageAddress::Memory(MemoryId::ModuleContext(2)),
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

    // ---- The module card's parameters ----

    fn instance(key: &str, values: &[(u32, ParamValue)]) -> ModuleInstance {
        let StepContent::Module(instance) = module_step(1, key, values).content else {
            unreachable!()
        };
        instance
    }

    #[test]
    fn a_module_parameter_shows_as_the_parameter_it_is_bound_to() {
        let definitions = [cell_reconstruction()];
        let shown = module_parameters(
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
        assert_eq!(cell_class.default_value, "1", "the module's default");
    }

    #[test]
    fn setting_a_module_parameter_stores_what_the_command_took() {
        let definitions = [cell_reconstruction()];
        let mut cells = instance("CELL_RECONSTRUCTION", &[]);

        set_module_parameter(&mut cells, &definitions, &ModuleParamKey::from("1"), "5").unwrap();
        set_module_parameter(&mut cells, &definitions, &ModuleParamKey::from("2"), "L1").unwrap();
        assert_eq!(cells.values[&ModuleParamKey::from("1")], ParamValue::Int(5));
        assert_eq!(
            cells.values[&ModuleParamKey::from("2")],
            ParamValue::Text("L1".into())
        );

        // Text the command doesn't take leaves the value as it was.
        set_module_parameter(&mut cells, &definitions, &ModuleParamKey::from("1"), "big").unwrap();
        assert_eq!(cells.values[&ModuleParamKey::from("1")], ParamValue::Int(5));

        assert!(
            set_module_parameter(&mut cells, &definitions, &ModuleParamKey::from("9"), "1")
                .is_err()
        );
    }

    #[test]
    fn a_parameter_bound_to_a_nested_modules_parameter_works_the_same() {
        let outer = definition(
            "OUTER",
            vec![module_step(1, "CELL_RECONSTRUCTION", &[])],
            vec![parameter(1, ParamValue::Int(7), &[(1, "2")])],
        );
        let definitions = [cell_reconstruction(), outer];
        let mut outer = instance("OUTER", &[]);

        let shown = module_parameters(&outer, &definitions).unwrap();
        assert_eq!(shown[0].value, "7");

        set_module_parameter(&mut outer, &definitions, &ModuleParamKey::from("1"), "8").unwrap();
        assert_eq!(outer.values[&ModuleParamKey::from("1")], ParamValue::Int(8));
        let commands = expand(
            &module_step(1, "OUTER", &[(1, ParamValue::Int(8))]),
            &definitions,
        )
        .unwrap();
        assert_eq!(
            voronoi(&commands[1].command).centers,
            ObjectClass::Global(8)
        );
    }
}
