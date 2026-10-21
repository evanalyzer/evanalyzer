//! "Create component": turns consecutive steps of a pipeline into a component
//! - a new definition in the project, and a step using it in their place.
//!
//! [`plan_component`] looks at the project classes the steps use and suggests
//! a role for each ([`ClassRole`]); [`create_component`] builds the component with the
//! roles the user confirmed. With every class a parameter or fixed, the component
//! runs exactly the commands the steps ran before.

use chrono::Utc;
use evanalyzer_cfg::{
    core_types::{
        ComponentParamKey, ImageAddress, InternalErrors, MemoryId, ObjectClass, PipelineId,
        SegmentationClass, StepId,
    },
    settings::{
        classification_settings::Class,
        components::{
            ComponentDefinition, ComponentInstance, ComponentParameterDef, ParamBinding, ParamValue,
        },
        meta_data::MetaData,
        parameter_def::{ParamType, ParameterDef},
        pipeline_command::PipelineCommand,
        pipeline_command_settings::ExtractClassMappingSettings,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
    },
};
use evanalyzer_core::{ExpandedCommand, ScopeCounter, expand_step};
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

use crate::workspace::extensions::utils::assign_step_ids;

/// What a project class becomes in the new component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassRole {
    /// A class of the component's own (`L1`, `L2`, ...): internal, each use of
    /// the component has its own, and its objects don't reach the results.
    Local,
    /// A component parameter: each use of the component picks the class (starting
    /// with this one).
    Parameter,
    /// Stays written into the component's steps as it is.
    Fixed,
}

/// A project class the steps going into the component use.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentPlanClass {
    pub class: ObjectClass,
    /// The class's name in the project ("Class 5" if it has none).
    pub name: String,
    /// Used by other steps, pipelines or hand annotations too.
    pub used_elsewhere: bool,
    pub suggested: ClassRole,
}

/// What making a component of some steps would involve - see [`plan_component`].
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentPlan {
    pub classes: Vec<ComponentPlanClass>,
    /// Image channels the steps load (Load Image Channel): exposed as component
    /// parameters.
    pub channels: BTreeSet<u32>,
    /// The pipeline's memory slots only these steps use: they become the
    /// component's own (each use of the component has its own).
    pub component_slots: BTreeSet<u32>,
    /// In a component's steps: its memory slots its other steps use too -
    /// each becomes a parameter of the new component, the outer one hands
    /// the slot in.
    pub handed_slots: BTreeSet<u32>,
}

/// The project classes steps `range` of `list` use, each with a suggested
/// role: used elsewhere too → [`ClassRole::Parameter`]; only here →
/// [`ClassRole::Local`] - except a class the steps only know as a
/// segmentation class with no Extract Objects among them to turn it into
/// objects ([`ClassRole::Fixed`]: making it local would change nothing).
///
/// In a component's steps (a component inside a component), the outer
/// component's own classes and memory slots the steps use go into the new
/// one - refused if other steps of the outer component use them too: the
/// new component can't be handed them (yet). Settings the outer component's
/// parameters set stay set by them, through new parameters of the new one.
pub fn plan_component(
    project: &ProjectSettings,
    list: impl Into<StepList>,
    range: RangeInclusive<usize>,
) -> Result<ComponentPlan, InternalErrors> {
    let list = list.into();
    let steps = selected_steps(project, &list, &range)?;
    let bound = bound_settings(project, &list, steps);
    let elsewhere = classes_used_elsewhere(project, &list, &range);
    let extracts = steps
        .iter()
        .filter_map(|s| s.command())
        .any(|c| matches!(c, PipelineCommand::ExtractObjects(_)));

    // Classes in settings the outer component's parameters set: set from
    // outside, so they're parameters of the new component.
    let mut set_from_outside: BTreeSet<ObjectClass> = BTreeSet::new();
    let mut used: BTreeSet<ObjectClass> = BTreeSet::new();
    for step in steps {
        match &step.content {
            StepContent::Command(command) => {
                // The outer component's own classes too, inside a component.
                let mut classes = command.object_classes();
                if matches!(list, StepList::Component(_)) {
                    classes.extend(command.local_classes());
                }
                for class in classes {
                    let paths = class_paths(command, class);
                    if paths
                        .iter()
                        .any(|(path, _)| bound.contains(&(step.id, path.clone())))
                    {
                        set_from_outside.insert(class);
                    }
                    if paths.is_empty()
                        || paths
                            .iter()
                            .any(|(path, _)| !bound.contains(&(step.id, path.clone())))
                    {
                        used.insert(class);
                    }
                }
            }
            // A nested component's classes are written inside its own
            // definition - they can't be rewritten here.
            StepContent::Component(_) => {}
        }
    }
    let nested: BTreeSet<ObjectClass> = steps
        .iter()
        .filter(|s| matches!(s.content, StepContent::Component(_)))
        .flat_map(|s| commands_of(project, s))
        .flat_map(|c| c.object_classes())
        .collect();

    // The outer component's classes its other steps use too: handed to the new
    // component (a parameter), not moved into it.
    let mut used_by_others: BTreeSet<ObjectClass> = BTreeSet::new();
    let mut handed_slots: BTreeSet<u32> = BTreeSet::new();
    if let StepList::Component(key) = &list {
        let outer = definition(project, key)?;
        let others = || {
            outer
                .steps
                .iter()
                .enumerate()
                .filter(|(index, _)| !range.contains(index))
                .filter_map(|(_, step)| step.command())
        };
        for (_, step) in outer
            .steps
            .iter()
            .enumerate()
            .filter(|(index, _)| !range.contains(index))
        {
            match &step.content {
                StepContent::Command(command) => used_by_others.extend(command.local_classes()),
                StepContent::Component(instance) => {
                    used_by_others.extend(instance.values.values().filter_map(local_class_of))
                }
            }
        }
        let moved_slots: BTreeSet<u32> = steps
            .iter()
            .filter_map(|s| s.command())
            .flat_map(component_slots)
            .collect();
        let other_slots: BTreeSet<u32> = others().flat_map(component_slots).collect();
        for slot in moved_slots.intersection(&other_slots) {
            // Handed in through a setting the slot is written in - where a step
            // uses it otherwise (e.g. a threshold's value source), it can't be.
            let fixed = steps.iter().find(|step| {
                step.command().is_some_and(|command| {
                    memory_slots(command, "componentContext").contains(slot)
                        && slot_paths(command, *slot).len() < slot_uses(command, *slot)
                })
            });
            if let Some(step) = fixed {
                return Err(InternalErrors::InvalidArgument(format!(
                    "component memory slot {slot} is also used by other steps of \"{}\", and \
                     step {} uses it in a way that can't be handed to a component inside it - \
                     take those steps too",
                    outer.meta.name, step.id
                )));
            }
            handed_slots.insert(*slot);
        }
    }

    let classes = used
        .into_iter()
        .map(|class| {
            let used_elsewhere = elsewhere.contains(&class)
                || nested.contains(&class)
                || set_from_outside.contains(&class)
                || used_by_others.contains(&class);
            let only_segmentation = steps.iter().filter_map(|s| s.command()).all(|c| {
                class_paths(c, class)
                    .iter()
                    .all(|(_, k)| *k == PathKind::Segmentation)
            });
            let suggested = if used_elsewhere {
                ClassRole::Parameter
            } else if only_segmentation && !extracts {
                ClassRole::Fixed
            } else {
                ClassRole::Local
            };
            ComponentPlanClass {
                class,
                name: match &list {
                    StepList::Component(key) if matches!(class, ObjectClass::Local(_)) => {
                        definition(project, key)
                            .map(|outer| local_class_name(outer, class))
                            .unwrap_or_else(|_| class.to_text())
                    }
                    _ => class_name(project, class),
                },
                used_elsewhere,
                suggested,
            }
        })
        .collect();

    let channels = steps
        .iter()
        .filter(|s| !bound.contains(&(s.id, "channel".to_string())))
        .filter_map(|s| s.command())
        .filter_map(|c| match c {
            PipelineCommand::LoadChannel(s) => Some(s.channel.0),
            _ => None,
        })
        .collect();
    Ok(ComponentPlan {
        classes,
        channels,
        component_slots: match list {
            StepList::Pipeline(pipeline_id) => slots_only_used_by(project, pipeline_id, &range),
            // An outer component's slots go with the steps as they are.
            StepList::Component(_) => BTreeSet::new(),
        },
        handed_slots,
    })
}

/// Turns steps `range` of `list` into a component named `name`: a new
/// definition in the project and one step using it in their place. `roles`
/// says what each class of [`plan_component`] becomes (a class left out stays
/// [`ClassRole::Fixed`]); loaded channels become parameters. Returns the new
/// definition's key.
///
/// In a component's steps, the outer component's classes the steps use
/// become the new one's (only [`ClassRole::Local`] - see [`plan_component`]),
/// and its parameters setting the steps' settings set them through new
/// parameters of the new component. The outer component runs exactly what
/// it ran before.
pub fn create_component(
    project: &mut ProjectSettings,
    list: impl Into<StepList>,
    range: RangeInclusive<usize>,
    name: &str,
    roles: &[(ObjectClass, ClassRole)],
) -> Result<String, InternalErrors> {
    let list = list.into();
    selected_steps(project, &list, &range)?;
    let name = name.trim();
    if name.is_empty() {
        return Err(InternalErrors::InvalidArgument(
            "a component needs a name".into(),
        ));
    }
    match list {
        StepList::Pipeline(pipeline_id) => {
            create_in_pipeline(project, pipeline_id, range, name, roles)
        }
        StepList::Component(outer) => create_in_component(project, &outer, range, name, roles),
    }
}

fn create_in_pipeline(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    range: RangeInclusive<usize>,
    name: &str,
    roles: &[(ObjectClass, ClassRole)],
) -> Result<String, InternalErrors> {
    let key = unique_key(project, name);
    let component_slots = slots_only_used_by(project, pipeline_id, &range);
    let names = class_names(project, None, roles);

    let pipeline = project
        .pipelines
        .iter_mut()
        .find(|p| p.id == pipeline_id)
        .expect("checked by selected_steps");
    let steps: Vec<PipelineStepSettings> = pipeline.steps.drain(range.clone()).collect();
    let mut definition = component_of_steps(
        &key,
        name,
        steps,
        roles,
        &names,
        &BTreeSet::new(),
        &mut Vec::new(),
    )?;

    // The pipeline's slots only these steps use become the component's own.
    for step in &mut definition.steps {
        if let StepContent::Command(command) = &mut step.content {
            *command = slots_to_component(command, &component_slots)?;
        }
    }

    let start = *range.start();
    pipeline.steps.insert(start, component_step(&key));
    assign_step_ids(&mut pipeline.steps);
    project.components.push(definition);
    Ok(key)
}

fn create_in_component(
    project: &mut ProjectSettings,
    outer_key: &str,
    range: RangeInclusive<usize>,
    name: &str,
    roles: &[(ObjectClass, ClassRole)],
) -> Result<String, InternalErrors> {
    // Refuses what can't go into the new component.
    let plan = plan_component(
        project,
        StepList::Component(outer_key.to_string()),
        range.clone(),
    )?;
    // The outer component's classes only these steps use, made the new one's:
    // they move.
    let moved_classes: BTreeSet<u32> = plan
        .classes
        .iter()
        .filter(|c| !c.used_elsewhere && roles.contains(&(c.class, ClassRole::Local)))
        .filter_map(|c| match c.class {
            ObjectClass::Local(n) => Some(n),
            _ => None,
        })
        .collect();
    let key = unique_key(project, name);
    let names = class_names(project, Some(definition(project, outer_key)?), roles);
    let bound = {
        let steps = selected_steps(project, &StepList::Component(outer_key.into()), &range)?;
        bound_settings(project, &StepList::Component(outer_key.into()), steps)
    };

    // Built on a copy - only taken if the outer component still runs.
    let mut components = project.components.clone();
    let outer = components
        .iter_mut()
        .find(|d| d.key == outer_key)
        .expect("checked by selected_steps");
    let steps: Vec<PipelineStepSettings> = outer.steps.drain(range.clone()).collect();
    let moved: BTreeSet<StepId> = steps.iter().map(|s| s.id).collect();
    let mut handed_in = Vec::new();
    let mut definition =
        component_of_steps(&key, name, steps, roles, &names, &bound, &mut handed_in)?;
    // A slot the outer component's other steps use too: a parameter, empty in
    // the new component, the outer one's step hands its slot in.
    for slot in &plan.handed_slots {
        let mut bindings = Vec::new();
        for step in &mut definition.steps {
            let id = step.id;
            if let Some(command) = step.command_mut() {
                for path in slot_paths(command, *slot) {
                    if bound.contains(&(id, path.clone())) {
                        continue;
                    }
                    evanalyzer_core::apply_command_param(
                        command,
                        &path,
                        &ImageAddress::None.to_param_value(),
                    );
                    bindings.push(ParamBinding {
                        step: id,
                        param: path,
                    });
                }
            }
        }
        if bindings.is_empty() {
            continue;
        }
        let label = format!("Memory slot {slot}");
        let parameter = new_parameter_key(&definition, &label);
        definition.parameters.push(ComponentParameterDef {
            key: parameter.clone(),
            label,
            description: String::new(),
            section: String::new(),
            default: ParamValue::Text(ImageAddress::None.to_param_value()),
            bindings,
        });
        handed_in.push((
            parameter,
            ParamValue::Text(
                ImageAddress::Memory(MemoryId::ComponentContext(*slot)).to_param_value(),
            ),
        ));
    }
    // Its classes went with the steps.
    outer.classes.retain(|c| !moved_classes.contains(&c.id));

    let start = *range.start();
    let mut uses_new = component_step(&key);
    if let StepContent::Component(instance) = &mut uses_new.content {
        instance.values = handed_in.into_iter().collect();
    }
    outer.steps.insert(start, uses_new);
    crate::workspace::extensions::utils::assign_component_step_ids(outer)?;
    let step = outer.steps[start].id;

    // A parameter of the outer component setting moved settings sets them
    // through a parameter of the new one, with its label and default.
    for parameter in &mut outer.parameters {
        if !parameter.bindings.iter().any(|b| moved.contains(&b.step)) {
            continue;
        }
        let inner_key = new_parameter_key(&definition, &parameter.label);
        let (inside, outside): (Vec<ParamBinding>, Vec<ParamBinding>) = parameter
            .bindings
            .iter()
            .cloned()
            .partition(|b| moved.contains(&b.step));
        definition.parameters.push(ComponentParameterDef {
            key: inner_key.clone(),
            label: parameter.label.clone(),
            description: parameter.description.clone(),
            section: parameter.section.clone(),
            default: parameter.default.clone(),
            bindings: inside,
        });
        let through = ParamBinding {
            step,
            param: inner_key.to_string(),
        };
        // Where the first moved one was: the first binding gives the card's
        // control.
        let first_inside = parameter
            .bindings
            .iter()
            .position(|b| moved.contains(&b.step))
            .expect("checked above");
        let before = parameter.bindings[..first_inside]
            .iter()
            .filter(|b| !moved.contains(&b.step))
            .count();
        let mut bindings = outside;
        bindings.insert(before, through);
        parameter.bindings = bindings;
    }
    components.push(definition);

    // The outer component must run as before - e.g. a value of its own class
    // can't be handed to the new component.
    let probe = component_step(outer_key);
    let runs = |definitions: &[ComponentDefinition]| {
        expand_step(&probe, definitions, &mut ScopeCounter::default()).map(|_| ())
    };
    if runs(&project.components).is_ok()
        && let Err(e) = runs(&components)
    {
        return Err(InternalErrors::InvalidArgument(format!(
            "these steps can't become a component: {e}"
        )));
    }
    project.components = components;
    Ok(key)
}

/// A step using component `key` at its defaults - to be numbered.
fn component_step(key: &str) -> PipelineStepSettings {
    PipelineStepSettings {
        id: StepId::UNASSIGNED,
        enabled: true,
        content: StepContent::Component(ComponentInstance {
            key: key.to_string(),
            values: Default::default(),
        }),
    }
}

/// Each class of `roles` with its name and colour - a local class of
/// `outer` (the component the steps are in) as `outer` has it.
fn class_names(
    project: &ProjectSettings,
    outer: Option<&ComponentDefinition>,
    roles: &[(ObjectClass, ClassRole)],
) -> Vec<(ObjectClass, String, u32)> {
    roles
        .iter()
        .map(|(class, _)| match (class, outer) {
            (ObjectClass::Global(n), _) => (
                *class,
                class_name(project, *class),
                project
                    .classification
                    .classes()
                    .iter()
                    .find(|c| c.id == *n)
                    .map_or(0x808080, |c| c.color),
            ),
            (ObjectClass::Local(n), Some(outer)) => (
                *class,
                local_class_name(outer, *class),
                outer
                    .classes
                    .iter()
                    .find(|c| c.id == *n)
                    .map_or(0x808080, |c| c.color),
            ),
            _ => (*class, class.to_text(), 0x808080),
        })
        .collect()
}

/// The component named `name` made of `steps`, with `roles` applied to their
/// classes and the loaded channels as parameters - the settings in `bound`
/// (step, path) left alone: a parameter of the component the steps came
/// from sets them.
fn component_of_steps(
    key: &str,
    name: &str,
    mut steps: Vec<PipelineStepSettings>,
    roles: &[(ObjectClass, ClassRole)],
    names: &[(ObjectClass, String, u32)],
    bound: &BTreeSet<(StepId, String)>,
    handed_in: &mut Vec<(ComponentParamKey, ParamValue)>,
) -> Result<ComponentDefinition, InternalErrors> {
    let mut definition = ComponentDefinition {
        key: key.to_string(),
        meta: MetaData {
            name: name.to_string(),
            creation_time: Utc::now(),
            ..Default::default()
        },
        classes: Vec::new(),
        steps: Vec::new(),
        parameters: Vec::new(),
        origin: None,
    };

    // An outer component's classes made the new one's keep their numbers -
    // the project's classes made local get the free ones.
    let kept: BTreeSet<u32> = roles
        .iter()
        .filter_map(|(class, role)| match (class, role) {
            (ObjectClass::Local(n), ClassRole::Local) => Some(*n),
            _ => None,
        })
        .collect();
    let mut free = (1..).filter(|n| !kept.contains(n));

    for ((class, role), (_, class_name, color)) in roles.iter().zip(names) {
        if let ObjectClass::Local(n) = class {
            match role {
                // In the new component it would be a class it doesn't have.
                ClassRole::Fixed => {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "the component class {class_name} can't stay as it is - make it a \
                         parameter or a class of the new component"
                    )));
                }
                ClassRole::Local => {
                    definition.classes.push(Class {
                        id: *n,
                        color: *color,
                        name: class_name.clone(),
                        notes: String::new(),
                    });
                    continue;
                }
                // Handed in, like a project class.
                ClassRole::Parameter => {}
            }
        }
        match role {
            ClassRole::Fixed => {}
            ClassRole::Local => {
                let local = ObjectClass::Local(free.next().expect("some number is free"));
                // Made by a segmentation step of these: Extract Objects maps it.
                let segmented_here = steps.iter().filter_map(|s| s.command()).any(|c| {
                    class_paths(c, *class)
                        .iter()
                        .any(|(_, kind)| *kind == PathKind::Segmentation)
                });
                for step in &mut steps {
                    let id = step.id;
                    if let StepContent::Command(command) = &mut step.content {
                        make_local(command, *class, local, segmented_here, |path| {
                            bound.contains(&(id, path.to_string()))
                        })?;
                    }
                }
                definition.classes.push(Class {
                    id: local_number(local),
                    color: *color,
                    name: class_name.clone(),
                    notes: String::new(),
                });
            }
            ClassRole::Parameter => {
                let mut bindings: Vec<(bool, ParamBinding)> = steps
                    .iter()
                    .filter_map(|s| s.command().map(|c| (s.id, c)))
                    .flat_map(|(id, command)| {
                        // Only the object-class side: a segmentation class
                        // with the class's number is the component's own
                        // business (as for a local class).
                        class_paths(command, *class)
                            .into_iter()
                            .filter(|(_, kind)| *kind == PathKind::Single)
                            .filter(move |(param, _)| !bound.contains(&(id, param.clone())))
                            .map(move |(param, _)| {
                                let list = setting_at(command, &param)
                                    .is_some_and(|(s, _)| s.param_type == ParamType::MultiObjClass);
                                (list, ParamBinding { step: id, param })
                            })
                    })
                    .collect();
                // A single class setting first: the card shows its control,
                // which can set the class lists too (see `same_kind`).
                bindings.sort_by_key(|(list, _)| *list);
                let bindings: Vec<ParamBinding> =
                    bindings.into_iter().map(|(_, binding)| binding).collect();
                if bindings.is_empty() {
                    continue;
                }
                let key = new_parameter_key(&definition, class_name);
                let default = if matches!(class, ObjectClass::Local(_)) {
                    // The outer component's class means nothing in the new one
                    // on its own: unset there, the outer one's step using it
                    // hands it in.
                    for binding in &bindings {
                        if let Some(command) = steps
                            .iter_mut()
                            .find(|s| s.id == binding.step)
                            .and_then(|s| s.command_mut())
                        {
                            evanalyzer_core::apply_command_param(
                                command,
                                &binding.param,
                                &ObjectClass::Unset.to_text(),
                            );
                        }
                    }
                    handed_in.push((key.clone(), ParamValue::Text(class.to_text())));
                    ParamValue::Text(ObjectClass::Unset.to_text())
                } else {
                    ParamValue::Text(class.to_text())
                };
                definition.parameters.push(ComponentParameterDef {
                    key,
                    label: class_name.clone(),
                    description: String::new(),
                    section: String::new(),
                    default,
                    bindings,
                });
            }
        }
    }

    // Each channel loaded becomes a parameter, bound to every step loading it.
    let mut channels: Vec<(u32, Vec<StepId>)> = Vec::new();
    for step in &steps {
        if bound.contains(&(step.id, "channel".to_string())) {
            continue;
        }
        if let Some(PipelineCommand::LoadChannel(s)) = step.command() {
            match channels.iter_mut().find(|(c, _)| *c == s.channel.0) {
                Some((_, ids)) => ids.push(step.id),
                None => channels.push((s.channel.0, vec![step.id])),
            }
        }
    }
    let several = channels.len() > 1;
    for (channel, ids) in channels {
        let label = if several {
            format!("Channel {channel}")
        } else {
            "Channel".into()
        };
        let key = new_parameter_key(&definition, &label);
        definition.parameters.push(ComponentParameterDef {
            key,
            label,
            description: String::new(),
            section: String::new(),
            default: ParamValue::Int(channel as i64),
            bindings: ids
                .into_iter()
                .map(|step| ParamBinding {
                    step,
                    param: "channel".into(),
                })
                .collect(),
        });
    }

    definition.steps = steps;
    Ok(definition)
}

/// The settings (step id, path) of `steps` that parameters of the component
/// `list` is set - none for a pipeline's steps.
fn bound_settings(
    project: &ProjectSettings,
    list: &StepList,
    steps: &[PipelineStepSettings],
) -> BTreeSet<(StepId, String)> {
    let StepList::Component(key) = list else {
        return BTreeSet::new();
    };
    let Ok(outer) = definition(project, key) else {
        return BTreeSet::new();
    };
    outer
        .parameters
        .iter()
        .flat_map(|p| p.bindings.iter())
        .filter(|b| steps.iter().any(|s| s.id == b.step))
        .map(|b| (b.step, b.param.clone()))
        .collect()
}

fn definition<'a>(
    project: &'a ProjectSettings,
    key: &str,
) -> Result<&'a ComponentDefinition, InternalErrors> {
    project
        .components
        .iter()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))
}

/// The component class (`"L1"`) a parameter value names, if it names one.
fn local_class_of(value: &ParamValue) -> Option<ObjectClass> {
    match value {
        ParamValue::Text(text) => {
            ObjectClass::from_text(text).filter(|c| matches!(c, ObjectClass::Local(_)))
        }
        _ => None,
    }
}

/// A local class's name as component `outer` has it ("L2" if it has none).
fn local_class_name(outer: &ComponentDefinition, class: ObjectClass) -> String {
    match class {
        ObjectClass::Local(n) => outer
            .classes
            .iter()
            .find(|c| c.id == n)
            .map(|c| c.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| class.to_text()),
        other => other.to_text(),
    }
}

/// Drops the values component steps hold for parameters their component no longer
/// has (hidden again, removed in the component settings, its step or script
/// param gone): a run never reads them, they'd only clutter the file.
/// Returns how many went.
pub fn prune_component_values(project: &mut ProjectSettings) -> usize {
    let parameters: std::collections::HashMap<String, BTreeSet<ComponentParamKey>> = project
        .components
        .iter()
        .map(|d| {
            (
                d.key.clone(),
                d.parameters.iter().map(|p| p.key.clone()).collect(),
            )
        })
        .collect();
    let mut pruned = 0;
    let steps = project
        .pipelines
        .iter_mut()
        .flat_map(|p| p.steps.iter_mut())
        .chain(
            project
                .components
                .iter_mut()
                .flat_map(|d| d.steps.iter_mut()),
        );
    for step in steps {
        if let StepContent::Component(instance) = &mut step.content
            && let Some(keys) = parameters.get(&instance.key)
        {
            let before = instance.values.len();
            instance.values.retain(|key, _| keys.contains(key));
            pruned += before - instance.values.len();
        }
    }
    pruned
}

/// How often the project uses component `key`: steps in its pipelines and in
/// other components.
pub fn component_uses(project: &ProjectSettings, key: &str) -> usize {
    project
        .pipelines
        .iter()
        .flat_map(|p| p.steps.iter())
        .chain(project.components.iter().flat_map(|d| d.steps.iter()))
        .filter(|step| matches!(&step.content, StepContent::Component(i) if i.key == key))
        .count()
}

/// Whether a step using component `inserted` would make component `container`
/// contain itself: `inserted` is `container`, or contains it - at any depth.
/// Such a component can't run ("contains itself"), so it isn't offered.
pub fn would_contain_itself(project: &ProjectSettings, container: &str, inserted: &str) -> bool {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut next = vec![inserted];
    while let Some(key) = next.pop() {
        if key == container {
            return true;
        }
        if !seen.insert(key) {
            continue;
        }
        if let Some(definition) = project.components.iter().find(|d| d.key == key) {
            next.extend(
                definition
                    .steps
                    .iter()
                    .filter_map(|step| match &step.content {
                        StepContent::Component(instance) => Some(instance.key.as_str()),
                        StepContent::Command(_) => None,
                    }),
            );
        }
    }
    false
}

/// The keys of the components no pipeline uses - not even through another
/// component (a component only an unused component uses is unused too).
pub fn unused_components(project: &ProjectSettings) -> Vec<String> {
    let mut used: Vec<&str> = Vec::new();
    let mut next: Vec<&PipelineStepSettings> = project
        .pipelines
        .iter()
        .flat_map(|p| p.steps.iter())
        .collect();
    while let Some(step) = next.pop() {
        if let StepContent::Component(instance) = &step.content
            && !used.contains(&instance.key.as_str())
        {
            used.push(&instance.key);
            if let Some(definition) = project.components.iter().find(|d| d.key == instance.key) {
                next.extend(definition.steps.iter());
            }
        }
    }
    project
        .components
        .iter()
        .filter(|d| !used.contains(&d.key.as_str()))
        .map(|d| d.key.clone())
        .collect()
}

/// Removes component `key` from the project - refused while a pipeline uses it.
pub fn remove_component(project: &mut ProjectSettings, key: &str) -> Result<(), InternalErrors> {
    if !unused_components(project).iter().any(|k| k == key) {
        return Err(InternalErrors::InvalidArgument(format!(
            "component {key} is still used - remove the steps using it first"
        )));
    }
    project.components.retain(|d| d.key != key);
    Ok(())
}

/// Removes every component no pipeline uses; returns their keys.
pub fn remove_unused_components(project: &mut ProjectSettings) -> Vec<String> {
    let unused = unused_components(project);
    project.components.retain(|d| !unused.contains(&d.key));
    unused
}

/// A list of steps: a pipeline's, or a component definition's (a component
/// nested in it is one of its steps).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepList {
    Pipeline(PipelineId),
    /// The steps of the component with this key.
    Component(String),
}

impl From<PipelineId> for StepList {
    fn from(id: PipelineId) -> Self {
        StepList::Pipeline(id)
    }
}

impl StepList {
    fn steps_mut<'a>(
        &self,
        project: &'a mut ProjectSettings,
    ) -> Option<&'a mut Vec<PipelineStepSettings>> {
        match self {
            StepList::Pipeline(id) => project
                .pipelines
                .iter_mut()
                .find(|p| p.id == *id)
                .map(|p| &mut p.steps),
            StepList::Component(key) => project
                .components
                .iter_mut()
                .find(|d| &d.key == key)
                .map(|d| &mut d.steps),
        }
    }

    fn step_mut<'a>(
        &self,
        project: &'a mut ProjectSettings,
        step_index: usize,
    ) -> Result<&'a mut PipelineStepSettings, InternalErrors> {
        self.steps_mut(project)
            .and_then(|steps| steps.get_mut(step_index))
            .ok_or_else(|| {
                InternalErrors::InvalidArgument(match self {
                    StepList::Pipeline(id) => {
                        format!("pipeline {id} has no step {}", step_index + 1)
                    }
                    StepList::Component(key) => {
                        format!("component {key} has no step {}", step_index + 1)
                    }
                })
            })
    }
}

/// What "Edit in place" copies: the component used by other steps too.
#[derive(Debug, Clone, PartialEq)]
pub struct InPlaceCopy {
    /// The component's name.
    pub component: String,
    /// The other steps using it - they keep it.
    pub other_uses: usize,
    /// It came from the library: the copy gets no library updates.
    pub from_library: bool,
    /// The copy's name unless one is chosen: "Name (2)", the next free one.
    pub suggested_name: String,
}

/// What "Edit in place" on step `step_index` of `list` (a component step)
/// does: `None` when no other step uses its component - edited directly -
/// else the copy it makes ([`component_in_place_for_step`] decides the same).
pub fn in_place_copy(
    project: &mut ProjectSettings,
    list: &StepList,
    step_index: usize,
) -> Result<Option<InPlaceCopy>, InternalErrors> {
    let key = match &list.step_mut(project, step_index)?.content {
        StepContent::Component(instance) => instance.key.clone(),
        StepContent::Command(_) => {
            return Err(InternalErrors::InvalidArgument(format!(
                "step {} isn't a component",
                step_index + 1
            )));
        }
    };
    let uses = component_uses(project, &key);
    if uses <= 1 {
        return Ok(None);
    }
    let definition = project
        .components
        .iter()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?;
    Ok(Some(InPlaceCopy {
        component: definition.meta.name.clone(),
        other_uses: uses - 1,
        from_library: definition.origin.is_some(),
        suggested_name: next_copy_name(project, &definition.meta.name),
    }))
}

/// "Name (2)" - the first "Name (n)" no component of `project` is called.
fn next_copy_name(project: &ProjectSettings, base: &str) -> String {
    (2..)
        .map(|n| format!("{base} ({n})"))
        .find(|name| copy_name_problem(project, name).is_none())
        .expect("some number is free")
}

/// Why `name` can't name a new component of `project` - empty, or another
/// component's (ignoring case and surrounding spaces) - or `None`.
pub fn copy_name_problem(project: &ProjectSettings, name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() {
        return Some("Give the copy a name".to_string());
    }
    project
        .components
        .iter()
        .any(|d| d.meta.name.trim().to_lowercase() == name.to_lowercase())
        .then(|| format!("A component is called “{name}” already"))
}

/// Gives step `step_index` of `list` (a component step) a copy of its
/// component - named `name`, else "Name (2)", with a key of its own - so
/// changes to it don't affect the component's other uses. Returns the copy's
/// key. In a component's steps that's a change of that component - so of all
/// its uses.
pub fn clone_component_for_step(
    project: &mut ProjectSettings,
    list: &StepList,
    step_index: usize,
    name: Option<&str>,
) -> Result<String, InternalErrors> {
    let key = match &list.step_mut(project, step_index)?.content {
        StepContent::Component(instance) => instance.key.clone(),
        StepContent::Command(_) => {
            return Err(InternalErrors::InvalidArgument(format!(
                "step {} isn't a component",
                step_index + 1
            )));
        }
    };
    let mut copy = project
        .components
        .iter()
        .find(|d| d.key == key)
        .cloned()
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?;
    let name = match name {
        Some(name) => {
            if let Some(problem) = copy_name_problem(project, name) {
                return Err(InternalErrors::InvalidArgument(problem));
            }
            name.trim().to_string()
        }
        None => next_copy_name(project, &copy.meta.name),
    };
    copy.key = unique_key(project, &name);
    copy.meta.name = name;
    copy.meta.creation_time = Utc::now();
    // Not the library file's version any more.
    copy.origin = None;
    let new_key = copy.key.clone();
    project.components.push(copy);
    if let StepContent::Component(instance) = &mut list.step_mut(project, step_index)?.content {
        instance.key = new_key.clone();
    }
    Ok(new_key)
}

/// "Edit in place": step `step_index` of `list` (a component step) gets a
/// component of its own that holds its current values - the inner steps
/// with the step's values applied, each parameter's default the step's
/// value - so opened it shows what this step runs, and changes reach no
/// other step. Used by other steps too: a copy (named `copy_name`, else
/// "Name (2)"; a key of its own, no library origin); used only here: the
/// definition itself, `copy_name` unused. The step's own values are cleared
/// (its definition holds them now - kept, they would override every edit
/// made inside). Returns the key.
pub fn component_in_place_for_step(
    project: &mut ProjectSettings,
    list: &StepList,
    step_index: usize,
    copy_name: Option<&str>,
) -> Result<String, InternalErrors> {
    let instance = match &list.step_mut(project, step_index)?.content {
        StepContent::Component(instance) => instance.clone(),
        StepContent::Command(_) => {
            return Err(InternalErrors::InvalidArgument(format!(
                "step {} isn't a component",
                step_index + 1
            )));
        }
    };
    let steps = evanalyzer_core::component_steps_with_values(&instance, &project.components)?;
    let key = if component_uses(project, &instance.key) > 1 {
        clone_component_for_step(project, list, step_index, copy_name)?
    } else {
        instance.key.clone()
    };
    let definition = project
        .components
        .iter_mut()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?;
    definition.steps = steps;
    for parameter in &mut definition.parameters {
        if let Some(value) = instance.values.get(&parameter.key) {
            parameter.default = value.clone();
        }
    }
    if let StepContent::Component(own) = &mut list.step_mut(project, step_index)?.content {
        own.values.clear();
    }
    Ok(key)
}

/// "Expand component": replaces step `step_index` of `list` (a component step) by
/// its component's steps, with the step's values applied. The definition
/// stays in the project.
///
/// In a pipeline, the component's own classes become new project classes
/// (same name and colour) and its memory slots pipeline slots - both only
/// exist inside a component. In a component's steps they become that
/// component's (numbered after its own), and its parameters setting the
/// expanded component's parameters set the settings those set.
pub fn expand_component(
    project: &mut ProjectSettings,
    list: impl Into<StepList>,
    step_index: usize,
) -> Result<(), InternalErrors> {
    let list = list.into();
    let StepContent::Component(instance) = &list.step_mut(project, step_index)?.content else {
        return Err(InternalErrors::InvalidArgument(format!(
            "step {} isn't a component",
            step_index + 1
        )));
    };
    let instance = instance.clone();
    match list {
        StepList::Pipeline(pipeline_id) => {
            expand_in_pipeline(project, pipeline_id, step_index, &instance)
        }
        StepList::Component(outer) => expand_in_component(project, &outer, step_index, &instance),
    }
}

fn expand_in_pipeline(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    step_index: usize,
    instance: &ComponentInstance,
) -> Result<(), InternalErrors> {
    let steps = evanalyzer_core::component_steps_with_values(instance, &project.components)?;
    let definition = definition(project, &instance.key)?.clone();

    // Each local class → a new project class.
    let mut next_id = project
        .classification
        .classes()
        .iter()
        .map(|c| c.id)
        .max()
        .unwrap_or(0)
        + 1;
    let mut to_global = std::collections::BTreeMap::new();
    for class in &definition.classes {
        project.classification.classes_mut().push(Class {
            id: next_id,
            color: class.color,
            name: class.name.clone(),
            notes: class.notes.clone(),
        });
        to_global.insert(class.id, next_id);
        next_id += 1;
    }

    let mut steps: Vec<PipelineStepSettings> = steps
        .into_iter()
        .map(|mut step| {
            step.id = StepId::UNASSIGNED;
            match &mut step.content {
                StepContent::Command(command) => {
                    *command = outside_the_component(command, &to_global)?;
                }
                // What it hands a component inside it too.
                StepContent::Component(inner) => {
                    for value in inner.values.values_mut() {
                        value_outside_the_component(value, &to_global);
                    }
                }
            }
            Ok(step)
        })
        .collect::<Result<_, InternalErrors>>()?;
    let pipeline = project
        .pipelines
        .iter_mut()
        .find(|p| p.id == pipeline_id)
        .expect("checked by expand_component");
    pipeline
        .steps
        .splice(step_index..=step_index, steps.drain(..));
    assign_step_ids(&mut pipeline.steps);
    Ok(())
}

fn expand_in_component(
    project: &mut ProjectSettings,
    outer_key: &str,
    step_index: usize,
    instance: &ComponentInstance,
) -> Result<(), InternalErrors> {
    let inner = definition(project, &instance.key)?.clone();
    let outer = definition(project, outer_key)?;

    // The inner component's classes and slots, numbered after the outer one's.
    let (outer_classes, outer_slots) = evanalyzer_core::own_numbers(outer);
    let (inner_classes, inner_slots) = evanalyzer_core::own_numbers(&inner);
    let renumber = |own: Vec<u32>, taken: &[u32]| -> std::collections::BTreeMap<u32, u32> {
        let first = taken.iter().max().map_or(1, |n| n + 1);
        own.into_iter().zip(first..).collect()
    };
    let classes = renumber(inner_classes, &outer_classes);
    let slots = renumber(inner_slots, &outer_slots);
    // Renumbered before the step's values go in - those are the outer one's.
    let mut definitions = project.components.clone();
    for definition in &mut definitions {
        if definition.key == inner.key {
            *definition = evanalyzer_core::renumbered(&inner, &classes, &slots);
        }
    }
    let steps = evanalyzer_core::component_steps_with_values(instance, &definitions)?;
    let old_ids: Vec<StepId> = steps.iter().map(|s| s.id).collect();

    let mut components = project.components.clone();
    let outer = components
        .iter_mut()
        .find(|d| d.key == outer_key)
        .expect("found above");
    outer
        .classes
        .extend(inner.classes.iter().map(|class| Class {
            id: classes[&class.id],
            ..class.clone()
        }));
    let expanded = outer.steps[step_index].id;
    let count = steps.len();
    outer.steps.splice(
        step_index..=step_index,
        steps.into_iter().map(|mut step| {
            step.id = StepId::UNASSIGNED;
            step
        }),
    );
    crate::workspace::extensions::utils::assign_component_step_ids(outer)?;
    let new_id: std::collections::BTreeMap<StepId, StepId> = old_ids
        .into_iter()
        .zip(
            outer.steps[step_index..step_index + count]
                .iter()
                .map(|s| s.id),
        )
        .collect();

    // A parameter setting the expanded component's parameter sets what that set.
    for parameter in &mut outer.parameters {
        parameter.bindings = parameter
            .bindings
            .iter()
            .flat_map(|binding| {
                if binding.step != expanded {
                    return vec![binding.clone()];
                }
                inner
                    .parameters
                    .iter()
                    .find(|p| p.key.to_string() == binding.param)
                    .map(|p| {
                        p.bindings
                            .iter()
                            .filter_map(|b| {
                                Some(ParamBinding {
                                    step: *new_id.get(&b.step)?,
                                    param: b.param.clone(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
    }
    outer.parameters.retain(|p| !p.bindings.is_empty());

    // The outer component must run as before.
    let probe = component_step(outer_key);
    let runs = |definitions: &[ComponentDefinition]| {
        expand_step(&probe, definitions, &mut ScopeCounter::default()).map(|_| ())
    };
    if runs(&project.components).is_ok()
        && let Err(e) = runs(&components)
    {
        return Err(InternalErrors::InvalidArgument(format!(
            "the component can't be expanded here: {e}"
        )));
    }
    project.components = components;
    Ok(())
}

/// A value a component's step hands a component inside it, as it reads outside
/// the component: its class a project class (see [`outside_the_component`]),
/// its memory slot a pipeline slot.
fn value_outside_the_component(
    value: &mut ParamValue,
    to_global: &std::collections::BTreeMap<u32, u32>,
) {
    match value {
        ParamValue::Text(text) => {
            if let Some(ObjectClass::Local(n)) = ObjectClass::from_text(text) {
                *text = ObjectClass::Global(to_global.get(&n).copied().unwrap_or(n)).to_text();
            } else if let Some(evanalyzer_cfg::core_types::ImageAddress::Memory(
                evanalyzer_cfg::core_types::MemoryId::ComponentContext(slot),
            )) = evanalyzer_cfg::core_types::ImageAddress::from_param_value(text)
            {
                *text = evanalyzer_cfg::core_types::ImageAddress::Memory(
                    evanalyzer_cfg::core_types::MemoryId::PipelineContext(slot),
                )
                .to_param_value();
            }
        }
        ParamValue::List(items) => items
            .iter_mut()
            .for_each(|v| value_outside_the_component(v, to_global)),
        _ => {}
    }
}

/// `command` as it reads outside its component: its local classes the project
/// classes `to_global` names, its component memory slots pipeline slots.
/// Rewritten in its file form, so every setting holding them is covered.
fn outside_the_component(
    command: &PipelineCommand,
    to_global: &std::collections::BTreeMap<u32, u32>,
) -> Result<PipelineCommand, InternalErrors> {
    fn rewrite(value: &mut serde_json::Value, to_global: &std::collections::BTreeMap<u32, u32>) {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1 {
                    if let Some(local) = map.get("LOCAL").and_then(|v| v.as_u64()) {
                        let global = to_global
                            .get(&(local as u32))
                            .copied()
                            .unwrap_or(local as u32);
                        *value = serde_json::json!({ "GLOBAL": global });
                        return;
                    }
                    if let Some(slot) = map.remove("componentContext") {
                        map.insert("pipelineContext".into(), slot);
                        return;
                    }
                }
                map.values_mut().for_each(|v| rewrite(v, to_global));
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(|v| rewrite(v, to_global)),
            _ => {}
        }
    }
    let mut value =
        serde_json::to_value(command).map_err(|e| InternalErrors::Internal(e.to_string()))?;
    rewrite(&mut value, to_global);
    serde_json::from_value(value).map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// Exposes setting `path` of step `step_id` as a parameter of `definition`
/// (`label`, its current `value` as default) - or, if it is exposed, hides
/// it again: the binding goes, and the parameter once nothing is bound to
/// it. Returns whether it's exposed now.
pub fn toggle_exposed(
    definition: &mut ComponentDefinition,
    step_id: StepId,
    path: &str,
    label: &str,
    value: &str,
) -> bool {
    let bound = |b: &ParamBinding| b.step == step_id && b.param == path;
    if definition
        .parameters
        .iter()
        .any(|p| p.bindings.iter().any(bound))
    {
        for parameter in &mut definition.parameters {
            parameter.bindings.retain(|b| !bound(b));
        }
        definition.parameters.retain(|p| !p.bindings.is_empty());
        return false;
    }
    definition.parameters.push(ComponentParameterDef {
        key: new_parameter_key(definition, label),
        label: label.to_string(),
        description: String::new(),
        section: String::new(),
        default: evanalyzer_core::value_from_text(value),
        bindings: vec![ParamBinding {
            step: step_id,
            param: path.to_string(),
        }],
    });
    true
}

/// A key for a new parameter of `definition` labelled `label`: the label in
/// `snake_case` ("Min Size" → `min_size`), with `_2`, `_3` … if taken. When
/// its section or label changes, `rekey_parameters` gives it the key they
/// make - and the instances' values, which are keyed by it, follow.
pub fn new_parameter_key(definition: &ComponentDefinition, label: &str) -> ComponentParamKey {
    let mut base = String::new();
    for c in label.trim().chars() {
        if c.is_alphanumeric() {
            base.extend(c.to_lowercase());
        } else if !base.is_empty() && !base.ends_with('_') {
            base.push('_');
        }
    }
    let base = base.trim_end_matches('_');
    let base = match base {
        "" => "parameter".to_string(),
        b if b.starts_with(|c: char| c.is_ascii_digit()) => format!("p_{b}"),
        b => b.to_string(),
    };
    let taken = |key: &str| definition.parameters.iter().any(|p| p.key.as_str() == key);
    let mut key = base.clone();
    let mut n = 2;
    while taken(&key) {
        key = format!("{base}_{n}");
        n += 1;
    }
    ComponentParamKey(key)
}

/// Renames parameter `key` of `definition` and sets its help text. Its key
/// stays until `rekey_parameters`. Returns whether the parameter exists.
pub fn edit_parameter(
    definition: &mut ComponentDefinition,
    key: &ComponentParamKey,
    label: &str,
    description: &str,
) -> bool {
    let Some(parameter) = definition.parameters.iter_mut().find(|p| &p.key == key) else {
        return false;
    };
    parameter.label = label.to_string();
    parameter.description = description.to_string();
    true
}

/// Puts parameter `key` of `definition` in section `section` on the component
/// card (blank: no section). Returns whether the parameter exists.
pub fn set_parameter_section(
    definition: &mut ComponentDefinition,
    key: &ComponentParamKey,
    section: &str,
) -> bool {
    let Some(parameter) = definition.parameters.iter_mut().find(|p| &p.key == key) else {
        return false;
    };
    parameter.section = section.trim().to_string();
    true
}

/// Moves parameter `key` of `definition` `delta` places (-1 up, +1 down) -
/// the order on the component card. Returns whether it moved.
pub fn move_parameter(
    definition: &mut ComponentDefinition,
    key: &ComponentParamKey,
    delta: i32,
) -> bool {
    let Some(from) = definition.parameters.iter().position(|p| &p.key == key) else {
        return false;
    };
    let to = from as i64 + delta as i64;
    if to < 0 || to >= definition.parameters.len() as i64 || to == from as i64 {
        return false;
    }
    let parameter = definition.parameters.remove(from);
    definition.parameters.insert(to as usize, parameter);
    true
}

/// Stops exposing parameter `key` of `definition`: the settings it set keep
/// the component's own values. Returns whether it existed.
pub fn remove_parameter(definition: &mut ComponentDefinition, key: &ComponentParamKey) -> bool {
    let before = definition.parameters.len();
    definition.parameters.retain(|p| &p.key != key);
    definition.parameters.len() != before
}

/// The parameter of `definition` that sets `path` of step `step_id`, if any.
pub fn parameter_for_setting(
    definition: &ComponentDefinition,
    step_id: StepId,
    path: &str,
) -> Option<ComponentParamKey> {
    definition
        .parameters
        .iter()
        .find(|p| {
            p.bindings
                .iter()
                .any(|b| b.step == step_id && b.param == path)
        })
        .map(|p| p.key.clone())
}

/// The setting at `path` of `command` - a top-level one ("min_area") or one
/// of a list entry ("thresholds.1.min_threshold") - and a label for it as a
/// component parameter ("Thresholds 2 · Min Threshold").
pub fn setting_at(command: &PipelineCommand, path: &str) -> Option<(ParameterDef, String)> {
    fn find(params: &[ParameterDef], path: &str) -> Option<(ParameterDef, String)> {
        if let Some(p) = params.iter().find(|p| p.name == path) {
            return Some((p.clone(), label_of(p)));
        }
        let mut parts = path.splitn(3, '.');
        let (list, index, field) = (parts.next()?, parts.next()?, parts.next()?);
        let index: usize = index.parse().ok()?;
        let list = params.iter().find(|p| p.name == list)?;
        let (setting, inner) = find(list.groups.get(index)?, field)?;
        Some((
            setting,
            format!("{} {} · {inner}", label_of(list), index + 1),
        ))
    }
    fn label_of(p: &ParameterDef) -> String {
        if p.display_name.is_empty() {
            p.name.clone()
        } else {
            p.display_name.clone()
        }
    }
    find(&evanalyzer_core::command_parameters(command), path)
}

/// The setting at `path` of `step` and a label for it, as [`setting_at`] - for
/// a component step, its component parameter `path` (as its card shows it,
/// components looked up in `definitions`).
pub fn step_setting(
    definitions: &[ComponentDefinition],
    step: &PipelineStepSettings,
    path: &str,
) -> Option<(ParameterDef, String)> {
    match &step.content {
        StepContent::Command(command) => setting_at(command, path),
        StepContent::Component(instance) => {
            evanalyzer_core::component_parameters(instance, definitions)
                .ok()?
                .into_iter()
                .find(|p| p.name == path)
                .map(|p| {
                    let label = if p.display_name.is_empty() {
                        p.name.clone()
                    } else {
                        p.display_name.clone()
                    };
                    (p, label)
                })
        }
    }
}

/// Every setting of `step` a component parameter could set: (path, the
/// setting, a label) - a command's, list entries' too ("thresholds.1.min"),
/// or a component step's parameters (components looked up in `definitions`).
pub fn step_settings(
    definitions: &[ComponentDefinition],
    step: &PipelineStepSettings,
) -> Vec<(String, ParameterDef, String)> {
    fn walk(
        params: &[ParameterDef],
        prefix: &str,
        label_prefix: &str,
        out: &mut Vec<(String, ParameterDef, String)>,
    ) {
        for param in params {
            let path = format!("{prefix}{}", param.name);
            let label = if param.display_name.is_empty() {
                param.name.clone()
            } else {
                param.display_name.clone()
            };
            match param.param_type {
                ParamType::Group => {
                    for (index, entry) in param.groups.iter().enumerate() {
                        walk(
                            entry,
                            &format!("{path}.{index}."),
                            &format!("{label_prefix}{label} {} · ", index + 1),
                            out,
                        );
                    }
                }
                ParamType::Label => {}
                _ => out.push((path, param.clone(), format!("{label_prefix}{label}"))),
            }
        }
    }
    let mut out = Vec::new();
    match &step.content {
        StepContent::Command(command) => walk(
            &evanalyzer_core::command_parameters(command),
            "",
            "",
            &mut out,
        ),
        StepContent::Component(instance) => {
            for param in
                evanalyzer_core::component_parameters(instance, definitions).unwrap_or_default()
            {
                let label = if param.display_name.is_empty() {
                    param.name.clone()
                } else {
                    param.display_name.clone()
                };
                out.push((param.name.clone(), param, label));
            }
        }
    }
    out
}

/// The settings of component `key`'s steps its parameter `parameter` could
/// also set: (step, path, label) - those with its control (see
/// [`compatible_parameters`]) no parameter sets yet, and class lists it
/// doesn't set yet (one several parameters set gets all their classes).
pub fn bindable_settings(
    definitions: &[ComponentDefinition],
    key: &str,
    parameter: &ComponentParamKey,
) -> Vec<(StepId, String, String)> {
    let Some(definition) = definitions.iter().find(|d| d.key == key) else {
        return Vec::new();
    };
    let instance = ComponentInstance {
        key: key.to_string(),
        values: Default::default(),
    };
    let Ok(shown) = evanalyzer_core::component_parameters(&instance, definitions) else {
        return Vec::new();
    };
    let Some(shown) = definition
        .parameters
        .iter()
        .zip(shown)
        .find(|(p, _)| &p.key == parameter)
        .map(|(_, shown)| shown)
    else {
        return Vec::new();
    };
    definition
        .steps
        .iter()
        .flat_map(|step| {
            step_settings(definitions, step)
                .into_iter()
                .filter(|(path, setting, _)| {
                    let list = matches!(
                        setting.param_type,
                        ParamType::MultiObjClass | ParamType::MultiSegClass
                    );
                    let sets_it = |p: &ComponentParameterDef| {
                        p.bindings
                            .iter()
                            .any(|b| b.step == step.id && &b.param == path)
                    };
                    same_kind(&shown, setting)
                        && if list {
                            !definition
                                .parameters
                                .iter()
                                .any(|p| &p.key == parameter && sets_it(p))
                        } else {
                            !definition.parameters.iter().any(sets_it)
                        }
                })
                .map(|(path, _, label)| (step.id, path, label))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Parameter `key` of `definition` no longer sets `path` of step `step_id` -
/// the setting keeps the value it shows. The parameter goes once it sets
/// nothing. Returns whether something changed.
pub fn remove_binding(
    definition: &mut ComponentDefinition,
    key: &ComponentParamKey,
    step_id: StepId,
    path: &str,
) -> bool {
    let Some(parameter) = definition.parameters.iter_mut().find(|p| &p.key == key) else {
        return false;
    };
    let before = parameter.bindings.len();
    parameter
        .bindings
        .retain(|b| !(b.step == step_id && b.param == path));
    if parameter.bindings.len() == before {
        return false;
    }
    definition.parameters.retain(|p| !p.bindings.is_empty());
    true
}

/// The parameters of component `key` a setting like `setting` could be added
/// to: those with the same control - type, and a dropdown's options.
pub fn compatible_parameters(
    definitions: &[ComponentDefinition],
    key: &str,
    setting: &ParameterDef,
) -> Vec<(ComponentParamKey, String)> {
    let Some(definition) = definitions.iter().find(|d| d.key == key) else {
        return Vec::new();
    };
    let instance = ComponentInstance {
        key: key.to_string(),
        values: Default::default(),
    };
    let Ok(shown) = evanalyzer_core::component_parameters(&instance, definitions) else {
        return Vec::new();
    };
    definition
        .parameters
        .iter()
        .zip(shown)
        .filter(|(_, shown)| same_kind(shown, setting))
        .map(|(parameter, _)| (parameter.key.clone(), parameter.label.clone()))
        .collect()
}

/// Whether a component parameter shown with the control of setting `shown`
/// (its first binding) can also set `other`: the same type and, for a type
/// whose options are its choices (a dropdown, units, file types), the same
/// options - or `shown` is a single object class and `other` a class list
/// (the value is then a list of one). Not the other way round: a list picks
/// several classes, a single class setting takes one. A segmentation class
/// is its own kind: a segmentation's number, not a project class. The
/// options of a class list or an image source are its current value (the
/// selected classes, the source pre-split), so they don't make a kind.
pub(crate) fn same_kind(shown: &ParameterDef, other: &ParameterDef) -> bool {
    if shown.param_type == ParamType::ObjClass
        && matches!(
            other.param_type,
            ParamType::ObjClass | ParamType::MultiObjClass
        )
    {
        return true;
    }
    let options_are_choices = matches!(
        shown.param_type,
        ParamType::Dropdown | ParamType::PixelUnits | ParamType::SizeUnits | ParamType::FilePath
    );
    shown.param_type == other.param_type && (!options_are_choices || shown.options == other.options)
}

/// Adds setting `path` of step `step_id` to parameter `key` of
/// `definition`: the parameter sets it too from now on.
pub fn add_binding(
    definition: &mut ComponentDefinition,
    key: &ComponentParamKey,
    step_id: StepId,
    path: &str,
) -> Result<(), InternalErrors> {
    let parameter = definition
        .parameters
        .iter_mut()
        .find(|p| &p.key == key)
        .ok_or_else(|| {
            InternalErrors::InvalidArgument(format!("the component has no parameter {key}"))
        })?;
    let binding = ParamBinding {
        step: step_id,
        param: path.to_string(),
    };
    if !parameter.bindings.contains(&binding) {
        parameter.bindings.push(binding);
    }
    // The setting now shows the value it gets: the parameter's.
    definition.steps = apply_defaults(definition);
    Ok(())
}

/// `definition`'s steps with each parameter's default written into the
/// settings it sets - what a use of the component without values of its own
/// runs. A bound setting's own value never runs: the parameter's replaces
/// it. Unchanged if a default can't be applied (the checks report that).
pub fn apply_defaults(definition: &ComponentDefinition) -> Vec<PipelineStepSettings> {
    let instance = ComponentInstance {
        key: definition.key.clone(),
        values: Default::default(),
    };
    evanalyzer_core::component_steps_with_values(&instance, std::slice::from_ref(definition))
        .unwrap_or_else(|_| definition.steps.clone())
}

/// Setting `path` of step `step_id` of `definition` was edited (in the
/// opened component). If a parameter sets it, its new value becomes the
/// parameter's default - else the edit would be lost, the parameter's value
/// replaces it at every run - and every setting the parameter sets shows
/// it. Returns whether a parameter sets it.
pub fn bound_setting_edited(
    definition: &mut ComponentDefinition,
    definitions: &[ComponentDefinition],
    step_id: StepId,
    path: &str,
) -> bool {
    let Some(key) = parameter_for_setting(definition, step_id, path) else {
        return false;
    };
    // A class list several parameters set holds their classes - none of
    // them takes it as its default.
    let setting_parameters = definition
        .parameters
        .iter()
        .filter(|p| {
            p.bindings
                .iter()
                .any(|b| b.step == step_id && b.param == path)
        })
        .count();
    if setting_parameters > 1 {
        return false;
    }
    let Some(value) = definition
        .steps
        .iter()
        .find(|s| s.id == step_id)
        .and_then(|step| step_setting(definitions, step, path))
        .map(|(setting, _)| setting.value)
    else {
        return false;
    };
    if let Some(parameter) = definition.parameters.iter_mut().find(|p| p.key == key) {
        parameter.default = evanalyzer_core::value_from_text(&value);
    }
    definition.steps = apply_defaults(definition);
    true
}

/// The settings of step `step_id` that are component parameters.
pub fn exposed_settings(definition: &ComponentDefinition, step_id: StepId) -> Vec<String> {
    definition
        .parameters
        .iter()
        .flat_map(|p| p.bindings.iter())
        .filter(|b| b.step == step_id)
        .map(|b| b.param.clone())
        .collect()
}

/// After entry `index` of list setting `list` of step `step_id` was
/// removed: bindings to that entry go (and a parameter left without
/// bindings), bindings to the entries after it move up one
/// ("thresholds.3.min" → "thresholds.2.min") - so they keep setting the
/// same entry.
pub fn list_entry_removed(
    definition: &mut ComponentDefinition,
    step_id: StepId,
    list: &str,
    index: usize,
) {
    let prefix = format!("{list}.");
    // The entry number of a binding into `list`, and the rest of its path.
    let entry_of = |param: &str| -> Option<(usize, String)> {
        let (number, rest) = param.strip_prefix(&prefix)?.split_once('.')?;
        Some((number.parse().ok()?, rest.to_string()))
    };
    for parameter in &mut definition.parameters {
        parameter
            .bindings
            .retain(|b| b.step != step_id || entry_of(&b.param).is_none_or(|(n, _)| n != index));
        for binding in &mut parameter.bindings {
            if binding.step != step_id {
                continue;
            }
            if let Some((n, rest)) = entry_of(&binding.param)
                && n > index
            {
                binding.param = format!("{list}.{}.{rest}", n - 1);
            }
        }
    }
    definition.parameters.retain(|p| !p.bindings.is_empty());
}

/// Removes step `index` of `definition` and every binding to it (and a
/// parameter left without bindings) - so the component still works.
pub fn remove_component_step(definition: &mut ComponentDefinition, index: usize) {
    if index >= definition.steps.len() {
        return;
    }
    let removed = definition.steps.remove(index).id;
    for parameter in &mut definition.parameters {
        parameter.bindings.retain(|b| b.step != removed);
    }
    definition.parameters.retain(|p| !p.bindings.is_empty());
}

/// How a setting refers to a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathKind {
    /// A single class setting (e.g. "output_class"), or a class list with
    /// only this class.
    Single,
    /// One entry of a class list with other classes too (e.g.
    /// "input_classes" = 1, 3) - a component parameter can't set one entry.
    Multi,
    /// A segmentation class with the class's number (a Threshold's class -
    /// the same number by convention).
    Segmentation,
}

/// Whether a setting of `command` writes or reads segmentation class
/// `number`.
pub(crate) fn uses_segmentation_class(command: &PipelineCommand, number: u32) -> bool {
    class_paths(command, ObjectClass::Global(number))
        .iter()
        .any(|(_, kind)| *kind == PathKind::Segmentation)
}

/// Where `command`'s settings refer to `class`: the setting's path (as
/// `apply_param_change` takes it, "list.2.field" inside list entries) and how.
fn class_paths(command: &PipelineCommand, class: ObjectClass) -> Vec<(String, PathKind)> {
    fn walk(
        params: &[ParameterDef],
        prefix: &str,
        class: ObjectClass,
        out: &mut Vec<(String, PathKind)>,
    ) {
        let text = class.to_text();
        for param in params {
            let path = format!("{prefix}{}", param.name);
            match param.param_type {
                ParamType::ObjClass if param.value.trim() == text => {
                    out.push((path, PathKind::Single))
                }
                ParamType::MultiObjClass => {
                    let items: Vec<&str> = param.value.split(',').map(str::trim).collect();
                    if items == [text.as_str()] {
                        // Only this class: set like a single class.
                        out.push((path, PathKind::Single));
                    } else if items.contains(&text.as_str()) {
                        out.push((path, PathKind::Multi));
                    }
                }
                ParamType::SegClass => {
                    if let ObjectClass::Global(n) = class
                        && param.value.trim() == n.to_string()
                    {
                        out.push((path, PathKind::Segmentation));
                    }
                }
                ParamType::Group => {
                    for (index, entry) in param.groups.iter().enumerate() {
                        walk(entry, &format!("{path}.{index}."), class, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(
        &evanalyzer_core::command_parameters(command),
        "",
        class,
        &mut out,
    );
    out
}

/// Rewrites every reference of `command` to project class `class` to the
/// component's class `local`. If `segmented_here` (a step going into the component makes the
/// segmentation class with the class's number), an Extract Objects maps that
/// segmentation class to `local`, so its objects get the component's class.
/// Settings `skip` (by path) says are left alone.
fn make_local(
    command: &mut PipelineCommand,
    class: ObjectClass,
    local: ObjectClass,
    segmented_here: bool,
    skip: impl Fn(&str) -> bool,
) -> Result<(), InternalErrors> {
    for (path, kind) in class_paths(command, class) {
        if skip(&path) {
            continue;
        }
        let value = match kind {
            PathKind::Segmentation => continue,
            PathKind::Single => local.to_text(),
            PathKind::Multi => {
                let current = find_value(&evanalyzer_core::command_parameters(command), &path)
                    .unwrap_or_default();
                current
                    .split(',')
                    .map(|item| {
                        if item.trim() == class.to_text() {
                            local.to_text()
                        } else {
                            item.trim().to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        evanalyzer_core::apply_command_param(command, &path, &value);
        if find_value(&evanalyzer_core::command_parameters(command), &path).as_deref()
            != Some(value.as_str())
        {
            return Err(InternalErrors::InvalidArgument(format!(
                "{} doesn't take the component class {} for {path}",
                command.name(),
                local.to_text()
            )));
        }
    }
    if let (true, PipelineCommand::ExtractObjects(extract), ObjectClass::Global(n)) =
        (segmented_here, command, class)
    {
        let segmentation_class = SegmentationClass(n);
        match extract
            .class_mapping
            .iter_mut()
            .find(|m| m.segmentation_class == segmentation_class)
        {
            // Already mapped elsewhere - that class decides.
            Some(_) => {}
            None => extract.class_mapping.push(ExtractClassMappingSettings {
                segmentation_class,
                object_class: local,
            }),
        }
    }
    Ok(())
}

fn find_value(params: &[ParameterDef], path: &str) -> Option<String> {
    if let Some(p) = params.iter().find(|p| p.name == path) {
        return Some(p.value.clone());
    }
    let mut parts = path.splitn(3, '.');
    let (list, index, field) = (parts.next()?, parts.next()?, parts.next()?);
    let index: usize = index.parse().ok()?;
    let entry = params.iter().find(|p| p.name == list)?.groups.get(index)?;
    find_value(entry, field)
}

fn local_number(class: ObjectClass) -> u32 {
    match class {
        ObjectClass::Local(n) => n,
        _ => unreachable!("only local classes are numbered here"),
    }
}

/// Steps `range` of `list` - an error if the list or the steps don't exist.
fn selected_steps<'a>(
    project: &'a ProjectSettings,
    list: &StepList,
    range: &RangeInclusive<usize>,
) -> Result<&'a [PipelineStepSettings], InternalErrors> {
    let (steps, name) = match list {
        StepList::Pipeline(id) => {
            let pipeline = project
                .pipelines
                .iter()
                .find(|p| p.id == *id)
                .ok_or_else(|| InternalErrors::InvalidArgument(format!("no pipeline {id}")))?;
            (&pipeline.steps, format!("pipeline \"{}\"", pipeline.name))
        }
        StepList::Component(key) => {
            let definition = definition(project, key)?;
            (
                &definition.steps,
                format!("component \"{}\"", definition.meta.name),
            )
        }
    };
    steps.get(range.clone()).ok_or_else(|| {
        InternalErrors::InvalidArgument(format!(
            "{name} has no steps {} to {}",
            range.start() + 1,
            range.end() + 1
        ))
    })
}

/// The commands `step` runs (a component's with its values) - none if it can't
/// be expanded.
fn commands_of(project: &ProjectSettings, step: &PipelineStepSettings) -> Vec<PipelineCommand> {
    commands_with(&project.components, step)
}

/// [`commands_of`] with these component definitions.
fn commands_with(
    definitions: &[ComponentDefinition],
    step: &PipelineStepSettings,
) -> Vec<PipelineCommand> {
    expand_step(step, definitions, &mut ScopeCounter::default())
        .map(|commands| {
            commands
                .into_iter()
                .map(|ExpandedCommand { command, .. }| command)
                .collect()
        })
        .unwrap_or_default()
}

/// The pipeline memory slots (`MemoryId::PipelineContext`) `command` reads
/// or writes.
fn pipeline_slots(command: &PipelineCommand) -> BTreeSet<u32> {
    memory_slots(command, "pipelineContext")
}

/// Where `command`'s settings name component memory slot `slot` as an
/// image address (`"component:2"`): their paths, as [`class_paths`] gives
/// them.
fn slot_paths(command: &PipelineCommand, slot: u32) -> Vec<String> {
    fn walk(params: &[ParameterDef], prefix: &str, text: &str, out: &mut Vec<String>) {
        for param in params {
            let path = format!("{prefix}{}", param.name);
            match param.param_type {
                ParamType::ImageAddress if param.value.trim() == text => out.push(path),
                ParamType::Group => {
                    for (index, entry) in param.groups.iter().enumerate() {
                        walk(entry, &format!("{path}.{index}."), text, out);
                    }
                }
                _ => {}
            }
        }
    }
    let text = ImageAddress::Memory(MemoryId::ComponentContext(slot)).to_param_value();
    let mut out = Vec::new();
    walk(
        &evanalyzer_core::command_parameters(command),
        "",
        &text,
        &mut out,
    );
    out
}

/// How often `command` names component memory slot `slot` - in any setting.
fn slot_uses(command: &PipelineCommand, slot: u32) -> usize {
    fn count(value: &serde_json::Value, slot: u64) -> usize {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1
                    && map.get("componentContext").and_then(|v| v.as_u64()) == Some(slot)
                {
                    return 1;
                }
                map.values().map(|v| count(v, slot)).sum()
            }
            serde_json::Value::Array(items) => items.iter().map(|v| count(v, slot)).sum(),
            _ => 0,
        }
    }
    serde_json::to_value(command).map_or(0, |value| count(&value, slot as u64))
}

/// The component memory slots (`MemoryId::ComponentContext`) `command` reads
/// or writes.
fn component_slots(command: &PipelineCommand) -> BTreeSet<u32> {
    memory_slots(command, "componentContext")
}

/// The memory slots of kind `kind` (`MemoryId`'s serialized name) `command`
/// reads or writes.
fn memory_slots(command: &PipelineCommand, kind: &str) -> BTreeSet<u32> {
    fn walk(value: &serde_json::Value, kind: &str, out: &mut BTreeSet<u32>) {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1
                    && let Some(slot) = map.get(kind).and_then(|v| v.as_u64())
                {
                    out.insert(slot as u32);
                    return;
                }
                map.values().for_each(|v| walk(v, kind, out));
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| walk(v, kind, out)),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    if let Ok(value) = serde_json::to_value(command) {
        walk(&value, kind, &mut out);
    }
    out
}

/// The memory slots of pipeline `pipeline_id` only its steps `range` use -
/// not its other steps, nor a component among the steps (whose definition can't
/// be rewritten here).
fn slots_only_used_by(
    project: &ProjectSettings,
    pipeline_id: PipelineId,
    range: &RangeInclusive<usize>,
) -> BTreeSet<u32> {
    let Some(pipeline) = project.pipelines.iter().find(|p| p.id == pipeline_id) else {
        return BTreeSet::new();
    };
    let mut inside = BTreeSet::new();
    let mut elsewhere = BTreeSet::new();
    for (index, step) in pipeline.steps.iter().enumerate() {
        let target = match (&step.content, range.contains(&index)) {
            (StepContent::Command(_), true) => &mut inside,
            _ => &mut elsewhere,
        };
        for command in commands_of(project, step) {
            target.extend(pipeline_slots(&command));
        }
    }
    inside.difference(&elsewhere).copied().collect()
}

/// `command` with the pipeline memory slots `slots` as component slots.
fn slots_to_component(
    command: &PipelineCommand,
    slots: &BTreeSet<u32>,
) -> Result<PipelineCommand, InternalErrors> {
    fn rewrite(value: &mut serde_json::Value, slots: &BTreeSet<u32>) {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1
                    && let Some(slot) = map.get("pipelineContext").and_then(|v| v.as_u64())
                    && slots.contains(&(slot as u32))
                {
                    *value = serde_json::json!({ "componentContext": slot });
                    return;
                }
                map.values_mut().for_each(|v| rewrite(v, slots));
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(|v| rewrite(v, slots)),
            _ => {}
        }
    }
    if slots.is_empty() {
        return Ok(command.clone());
    }
    let mut value =
        serde_json::to_value(command).map_err(|e| InternalErrors::Internal(e.to_string()))?;
    rewrite(&mut value, slots);
    serde_json::from_value(value).map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// The classes the project uses outside steps `range` of `list`: in other
/// steps and pipelines (inside components too), in hand annotations, in the
/// tile merge settings.
fn classes_used_elsewhere(
    project: &ProjectSettings,
    list: &StepList,
    range: &RangeInclusive<usize>,
) -> BTreeSet<ObjectClass> {
    // A component's steps: every use of the component, without them (nor the
    // bindings to them).
    let without: Option<Vec<ComponentDefinition>> = match list {
        StepList::Pipeline(_) => None,
        StepList::Component(key) => {
            let mut definitions = project.components.clone();
            if let Some(outer) = definitions.iter_mut().find(|d| &d.key == key) {
                let removed: BTreeSet<StepId> = outer
                    .steps
                    .get(range.clone())
                    .unwrap_or_default()
                    .iter()
                    .map(|s| s.id)
                    .collect();
                outer.steps.retain(|s| !removed.contains(&s.id));
                for parameter in &mut outer.parameters {
                    parameter.bindings.retain(|b| !removed.contains(&b.step));
                }
                outer.parameters.retain(|p| !p.bindings.is_empty());
            }
            Some(definitions)
        }
    };
    let definitions = without.as_deref().unwrap_or(&project.components);
    let mut classes = BTreeSet::new();
    for pipeline in &project.pipelines {
        for (index, step) in pipeline.steps.iter().enumerate() {
            if *list == StepList::Pipeline(pipeline.id) && range.contains(&index) {
                continue;
            }
            for command in commands_with(definitions, step) {
                classes.extend(command.object_classes());
            }
        }
    }
    for image in project.images.list.values() {
        for series in image.series.values() {
            for object in &series.objects {
                classes.extend(object.object_class.iter().copied());
            }
        }
    }
    classes.extend(project.tile_merge.classes_to_not_merge.iter().copied());
    classes
}

fn class_name(project: &ProjectSettings, class: ObjectClass) -> String {
    match class {
        ObjectClass::Global(n) => project
            .classification
            .classes()
            .iter()
            .find(|c| c.id == n)
            .map(|c| c.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| format!("Class {n}")),
        other => other.to_text(),
    }
}

/// A key for a component named `name` that no component of the project has:
/// SCREAMING_SNAKE_CASE of the name, numbered if taken ("CELLS_2").
fn unique_key(project: &ProjectSettings, name: &str) -> String {
    let mut base: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    base = base.trim_matches('_').to_string();
    while base.contains("__") {
        base = base.replace("__", "_");
    }
    if base.is_empty() {
        base = "COMPONENT".into();
    }
    let taken = |key: &str| project.components.iter().any(|d| d.key == key);
    if !taken(&base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|key| !taken(key))
        .expect("some number is free")
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{ImageAddress, ImageChannelIdx, SizeUnits};
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        ClassifyObjectsSettings, ExtractObjectsSettings, LoadChannelSettings,
        ObjectClassifyObjectsClassifyMatchHandlingSettings, ThresholdEntrySettings,
        ThresholdSettings, VoronoiSettings,
    };
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    const NUCLEI: ObjectClass = ObjectClass::Global(1);
    const SEEDS: ObjectClass = ObjectClass::Global(2);
    const CELLS: ObjectClass = ObjectClass::Global(3);

    fn step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn threshold(class: u32) -> PipelineCommand {
        PipelineCommand::Threshold(ThresholdSettings {
            thresholds: vec![ThresholdEntrySettings {
                segmentation_class: SegmentationClass(class),
                ..Default::default()
            }],
        })
    }

    fn classify(input: ObjectClass, output: ObjectClass) -> PipelineCommand {
        PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
            input_classes: vec![input],
            match_handling: ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch,
            output_class: output,
            size_unit: SizeUnits::Pixels,
            min_area: 50.0,
            ..Default::default()
        })
    }

    fn pipeline(id: u32, steps: Vec<PipelineStepSettings>) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(id),
            name: format!("Pipeline {id}"),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps,
        }
    }

    /// Pipeline 1: nuclei → seeds (only used here) → Voronoi cells; pipeline
    /// 2 reads the cells.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        for (id, name) in [(1, "Nuclei"), (2, "Seeds"), (3, "Cells")] {
            project.classification.classes_mut().push(Class {
                id,
                color: 0xFF0000,
                name: name.into(),
                notes: String::new(),
            });
        }
        project.pipelines = vec![
            pipeline(
                1,
                vec![
                    step(1, threshold(1)),
                    step(2, PipelineCommand::ConnectedComponents(Default::default())),
                    step(3, PipelineCommand::ExtractObjects(Default::default())),
                    step(4, classify(NUCLEI, SEEDS)),
                    step(
                        5,
                        PipelineCommand::Voronoi(VoronoiSettings {
                            centers: SEEDS,
                            output_class: CELLS,
                            ..Default::default()
                        }),
                    ),
                ],
            ),
            pipeline(2, vec![step(1, classify(CELLS, ObjectClass::Global(4)))]),
        ];
        project
    }

    fn roles(plan: &ComponentPlan) -> Vec<(ObjectClass, ClassRole)> {
        plan.classes
            .iter()
            .map(|c| (c.class, c.suggested))
            .collect()
    }

    fn json(commands: Vec<PipelineCommand>) -> serde_json::Value {
        serde_json::to_value(commands).unwrap()
    }

    fn commands_of_step(
        project: &ProjectSettings,
        pipeline: usize,
        index: usize,
    ) -> Vec<PipelineCommand> {
        commands_of(project, &project.pipelines[pipeline].steps[index])
    }

    #[test]
    fn classes_used_elsewhere_are_suggested_as_parameters_the_others_as_local() {
        let plan = plan_component(&project(), PipelineId(1), 2..=4).unwrap();
        assert_eq!(
            roles(&plan),
            [
                (NUCLEI, ClassRole::Parameter), // the Threshold before makes it
                (SEEDS, ClassRole::Local),      // only used by these steps
                (CELLS, ClassRole::Parameter),  // pipeline 2 reads it
            ]
        );
        assert_eq!(plan.classes[1].name, "Seeds");
        assert!(!plan.classes[1].used_elsewhere);
    }

    #[test]
    fn creating_a_component_puts_the_steps_into_a_definition_and_a_step_using_it_in_their_place() {
        let mut project = project();
        let plan = plan_component(&project, PipelineId(1), 2..=4).unwrap();
        let key = create_component(
            &mut project,
            PipelineId(1),
            2..=4,
            "Cell reconstruction",
            &roles(&plan),
        )
        .unwrap();

        assert_eq!(key, "CELL_RECONSTRUCTION");
        let steps = &project.pipelines[0].steps;
        assert_eq!(steps.len(), 3);
        let StepContent::Component(instance) = &steps[2].content else {
            panic!("a component step in their place");
        };
        assert_eq!(instance.key, key);
        let definition = &project.components[0];
        assert_eq!(definition.meta.name, "Cell reconstruction");
        assert_eq!(definition.steps.len(), 3);
        assert_eq!(definition.classes.len(), 1);
        assert_eq!(
            (
                definition.classes[0].id,
                definition.classes[0].name.as_str()
            ),
            (1, "Seeds")
        );
        let labels: Vec<&str> = definition
            .parameters
            .iter()
            .map(|p| p.label.as_str())
            .collect();
        assert_eq!(labels, ["Nuclei", "Cells"]);

        // What runs: the seeds are the component's own now.
        let commands = commands_of_step(&project, 0, 2);
        let PipelineCommand::ClassifyObjects(classify) = &commands[1] else {
            panic!("classify")
        };
        assert_eq!(
            (classify.input_classes.as_slice(), classify.output_class),
            (&[NUCLEI][..], ObjectClass::Local(1))
        );
        let PipelineCommand::Voronoi(voronoi) = &commands[2] else {
            panic!("voronoi")
        };
        assert_eq!(
            (voronoi.centers, voronoi.output_class),
            (ObjectClass::Local(1), CELLS)
        );
    }

    #[test]
    fn without_local_classes_the_component_runs_exactly_what_the_steps_ran() {
        let mut project = project();
        let before: Vec<PipelineCommand> = project.pipelines[0].steps[2..=4]
            .iter()
            .filter_map(|s| s.command().cloned())
            .collect();
        let plan = plan_component(&project, PipelineId(1), 2..=4).unwrap();
        let roles: Vec<_> = plan
            .classes
            .iter()
            .map(|c| (c.class, ClassRole::Parameter))
            .collect();
        create_component(&mut project, PipelineId(1), 2..=4, "Cells", &roles).unwrap();

        assert_eq!(json(commands_of_step(&project, 0, 2)), json(before));
        // Each use can pick other classes: a parameter per class.
        assert_eq!(project.components[0].parameters.len(), 3);
    }

    #[test]
    fn a_class_segmented_and_extracted_inside_becomes_local_via_the_class_mapping() {
        let mut project = project();
        project.pipelines.push(pipeline(
            3,
            vec![
                step(
                    1,
                    PipelineCommand::LoadChannel(LoadChannelSettings {
                        channel: ImageChannelIdx(1),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    }),
                ),
                step(2, threshold(7)),
                step(3, PipelineCommand::ConnectedComponents(Default::default())),
                step(
                    4,
                    PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                        max_objects_before_fail: 100_000,
                        ..Default::default()
                    }),
                ),
            ],
        ));
        let plan = plan_component(&project, PipelineId(3), 0..=3).unwrap();
        assert_eq!(roles(&plan), [(ObjectClass::Global(7), ClassRole::Local)]);
        assert_eq!(plan.channels, BTreeSet::from([1]));

        create_component(&mut project, PipelineId(3), 0..=3, "Spots", &roles(&plan)).unwrap();
        let commands = commands_of_step(&project, 2, 0);
        let PipelineCommand::ExtractObjects(extract) = &commands[3] else {
            panic!("extract")
        };
        assert_eq!(extract.class_mapping.len(), 1);
        assert_eq!(extract.class_mapping[0].object_class, ObjectClass::Local(1));
        let channel = &project.components[0].parameters[0];
        assert_eq!(
            (channel.label.as_str(), &channel.default),
            ("Channel", &ParamValue::Int(1))
        );
    }

    #[test]
    fn a_class_only_segmented_here_without_extracting_stays_fixed() {
        let mut project = project();
        project
            .pipelines
            .push(pipeline(3, vec![step(1, threshold(8))]));
        let plan = plan_component(&project, PipelineId(3), 0..=0).unwrap();
        assert_eq!(roles(&plan), [(ObjectClass::Global(8), ClassRole::Fixed)]);
    }

    #[test]
    fn keys_are_unique_and_mistakes_are_refused() {
        let mut project = project();
        let none: [(ObjectClass, ClassRole); 0] = [];
        assert_eq!(
            create_component(&mut project, PipelineId(1), 1..=1, "Cells!", &none).unwrap(),
            "CELLS"
        );
        assert_eq!(
            create_component(&mut project, PipelineId(1), 1..=1, "cells", &none).unwrap(),
            "CELLS_2"
        );
        assert!(create_component(&mut project, PipelineId(1), 0..=0, "  ", &none).is_err());
        assert!(create_component(&mut project, PipelineId(1), 0..=9, "Too many", &none).is_err());
        assert!(plan_component(&project, PipelineId(9), 0..=0).is_err());
    }

    // ---- Open, Expand, expose ----

    /// `project()` with steps 3-5 of pipeline 1 made a component - "Seeds" internal.
    fn with_component() -> ProjectSettings {
        let mut project = project();
        let plan = plan_component(&project, PipelineId(1), 2..=4).unwrap();
        create_component(
            &mut project,
            PipelineId(1),
            2..=4,
            "Cell reconstruction",
            &roles(&plan),
        )
        .unwrap();
        project
    }

    #[test]
    fn expanding_puts_the_steps_back_with_the_components_classes_as_project_classes() {
        let mut project = with_component();
        expand_component(&mut project, PipelineId(1), 2).unwrap();

        let steps = &project.pipelines[0].steps;
        assert_eq!(steps.len(), 5);
        let seeds = project
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Seeds" && c.id != 2)
            .expect("a new project class for the component's \"Seeds\"");
        let PipelineCommand::ClassifyObjects(classify) = steps[3].command().unwrap() else {
            panic!("classify")
        };
        assert_eq!(classify.output_class, ObjectClass::Global(seeds.id));
        let PipelineCommand::Voronoi(voronoi) = steps[4].command().unwrap() else {
            panic!("voronoi")
        };
        assert_eq!(
            (voronoi.centers, voronoi.output_class),
            (ObjectClass::Global(seeds.id), CELLS)
        );
        let ids: BTreeSet<StepId> = steps.iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 5, "every step its own id");
        assert_eq!(project.components.len(), 1, "the definition stays");
    }

    #[test]
    fn expanding_turns_component_memory_slots_into_pipeline_slots() {
        use evanalyzer_cfg::core_types::MemoryId;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ImageCacheSettings, MathImageCacheImageCacheModeSettings,
        };
        let mut project = project();
        project.components.push(ComponentDefinition {
            key: "STORE".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(
                1,
                PipelineCommand::ImageCache(ImageCacheSettings {
                    mode: MathImageCacheImageCacheModeSettings::Store,
                    address: ImageAddress::Memory(MemoryId::ComponentContext(2)),
                }),
            )],
            parameters: vec![],
            origin: None,
        });
        project.pipelines[1].steps.push(PipelineStepSettings {
            id: StepId(9),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: "STORE".into(),
                values: Default::default(),
            }),
        });
        expand_component(&mut project, PipelineId(2), 1).unwrap();
        let PipelineCommand::ImageCache(cache) = project.pipelines[1].steps[1].command().unwrap()
        else {
            panic!("image cache")
        };
        assert_eq!(
            cache.address,
            ImageAddress::Memory(MemoryId::PipelineContext(2))
        );
    }

    /// Steps storing to slot 2 and reading it back: slot 2 becomes the
    /// component's. Slot 3 is read after the steps too, so it stays the
    /// pipeline's.
    #[test]
    fn slots_only_the_chosen_steps_use_become_component_slots() {
        use evanalyzer_cfg::core_types::MemoryId;
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ImageCacheSettings, MathImageCacheImageCacheModeSettings,
        };
        let cache = |mode, slot| {
            PipelineCommand::ImageCache(ImageCacheSettings {
                mode,
                address: ImageAddress::Memory(MemoryId::PipelineContext(slot)),
            })
        };
        let mut project = project();
        project.pipelines[0].steps = vec![
            step(1, cache(MathImageCacheImageCacheModeSettings::Store, 2)),
            step(2, cache(MathImageCacheImageCacheModeSettings::Load, 2)),
            step(3, cache(MathImageCacheImageCacheModeSettings::Store, 3)),
            step(4, cache(MathImageCacheImageCacheModeSettings::Load, 3)),
        ];
        let plan = plan_component(&project, PipelineId(1), 0..=2).unwrap();
        assert_eq!(plan.component_slots, BTreeSet::from([2]));

        create_component(&mut project, PipelineId(1), 0..=2, "Cache", &[]).unwrap();
        let addresses: Vec<ImageAddress> = project.components[0]
            .steps
            .iter()
            .map(|s| match s.command() {
                Some(PipelineCommand::ImageCache(c)) => c.address.clone(),
                _ => panic!("image cache"),
            })
            .collect();
        assert_eq!(
            addresses,
            [
                ImageAddress::Memory(MemoryId::ComponentContext(2)),
                ImageAddress::Memory(MemoryId::ComponentContext(2)),
                ImageAddress::Memory(MemoryId::PipelineContext(3)),
            ]
        );
        assert!(
            crate::workspace::component_checks::check_components(&project)
                .iter()
                .all(|p| p.severity != crate::workspace::component_checks::Severity::Error),
            "the component is valid"
        );
    }

    #[test]
    fn only_this_step_gets_a_copy_of_the_component() {
        let mut project = with_component();
        let mut second_use = project.pipelines[0].steps[2].clone();
        second_use.id = StepId(1);
        project.pipelines[1].steps.push(second_use);
        assert_eq!(component_uses(&project, "CELL_RECONSTRUCTION"), 2);

        let key =
            clone_component_for_step(&mut project, &StepList::Pipeline(PipelineId(2)), 1, None).unwrap();
        assert_eq!(key, "CELL_RECONSTRUCTION_2");
        assert_eq!(project.components[1].meta.name, "Cell reconstruction (2)");
        assert_eq!(
            component_uses(&project, "CELL_RECONSTRUCTION"),
            1,
            "the other use keeps it"
        );
        assert_eq!(component_uses(&project, &key), 1);
    }

    /// "Edit in place": the step's own component holds what it runs - its
    /// values become the component's - so it runs the same; the other use
    /// keeps the original. Used only here: no copy, the definition itself.
    #[test]
    fn edit_in_place_gives_the_step_a_component_with_its_values() {
        let runs = |project: &ProjectSettings, pipeline: usize, step: usize| {
            let StepContent::Component(instance) = &project.pipelines[pipeline].steps[step].content else {
                panic!("a component step");
            };
            serde_json::to_string(
                &evanalyzer_core::component_steps_with_values(instance, &project.components).unwrap(),
            )
            .unwrap()
        };
        let mut project = with_component();
        // The step has a value of its own for each parameter.
        let StepContent::Component(instance) = &mut project.pipelines[0].steps[2].content else {
            panic!("a component step");
        };
        let parameters = project.components[0].parameters.clone();
        assert!(!parameters.is_empty(), "the fixture's component has parameters");
        for p in &parameters {
            instance.values.insert(p.key.clone(), p.default.clone());
        }
        let mut second_use = project.pipelines[0].steps[2].clone();
        second_use.id = StepId(1);
        project.pipelines[1].steps.push(second_use);
        let before = runs(&project, 0, 2);
        let other_before = runs(&project, 1, 1);

        let key = component_in_place_for_step(&mut project, &StepList::Pipeline(PipelineId(1)), 2, None).unwrap();
        assert_eq!(key, "CELL_RECONSTRUCTION_2", "used twice: a copy");
        assert_eq!(runs(&project, 0, 2), before, "the step runs as before");
        assert_eq!(runs(&project, 1, 1), other_before, "the other use is untouched");
        let StepContent::Component(own) = &project.pipelines[0].steps[2].content else {
            panic!("a component step");
        };
        assert!(own.values.is_empty(), "its values are its component's now");
        assert_eq!(component_uses(&project, "CELL_RECONSTRUCTION"), 1);

        // The other use, now alone with its component: no copy.
        let count = project.components.len();
        let key = component_in_place_for_step(&mut project, &StepList::Pipeline(PipelineId(2)), 1, None).unwrap();
        assert_eq!(key, "CELL_RECONSTRUCTION");
        assert_eq!(project.components.len(), count, "no copy");
        assert_eq!(runs(&project, 1, 1), other_before);
    }

    /// "Edit in place" twice on the same step: the first makes its copy, the
    /// second finds the step alone with it - no second copy, nothing changes.
    #[test]
    fn edit_in_place_twice_makes_one_copy() {
        let mut project = with_component();
        let mut second_use = project.pipelines[0].steps[2].clone();
        second_use.id = StepId(1);
        project.pipelines[1].steps.push(second_use);
        let list = StepList::Pipeline(PipelineId(1));

        let first = component_in_place_for_step(&mut project, &list, 2, None).unwrap();
        let after_first = serde_json::to_value((&project.pipelines, &project.components)).unwrap();
        let second = component_in_place_for_step(&mut project, &list, 2, None).unwrap();

        assert_eq!(first, "CELL_RECONSTRUCTION_2");
        assert_eq!(second, first, "the step's own copy again");
        assert_eq!(project.components.len(), 2, "the original and one copy");
        assert_eq!(
            serde_json::to_value((&project.pipelines, &project.components)).unwrap(),
            after_first,
            "nothing changed"
        );
    }

    /// "Edit in place" asks before copying: whether it copies (another step
    /// uses the component), what the copy is called unless renamed, and
    /// which names it can't have.
    #[test]
    fn edit_in_place_says_when_it_copies_and_takes_the_chosen_name() {
        let mut project = with_component();
        let list = StepList::Pipeline(PipelineId(1));
        let name = project.components[0].meta.name.clone();
        assert_eq!(in_place_copy(&mut project, &list, 2).unwrap(), None, "the only use");
        assert!(in_place_copy(&mut project, &list, 0).is_err(), "not a component step");

        let mut second_use = project.pipelines[0].steps[2].clone();
        second_use.id = StepId(1);
        project.pipelines[1].steps.push(second_use);
        let copy = in_place_copy(&mut project, &list, 2).unwrap().expect("a copy");
        assert_eq!(copy.component, name);
        assert_eq!(copy.other_uses, 1);
        assert!(!copy.from_library);
        assert_eq!(copy.suggested_name, format!("{name} (2)"));

        assert!(copy_name_problem(&project, "  ").is_some(), "empty");
        assert!(
            copy_name_problem(&project, &format!(" {} ", name.to_uppercase())).is_some(),
            "taken, ignoring case and spaces"
        );
        assert_eq!(copy_name_problem(&project, "Strict nuclei"), None);
        assert!(
            component_in_place_for_step(&mut project, &list, 2, Some(&name)).is_err(),
            "a taken name is refused"
        );
        assert_eq!(project.components.len(), 1, "nothing made");

        component_in_place_for_step(&mut project, &list, 2, Some(" Strict nuclei ")).unwrap();
        assert_eq!(project.components.len(), 2);
        assert_eq!(project.components[1].meta.name, "Strict nuclei");
        assert_eq!(in_place_copy(&mut project, &list, 2).unwrap(), None, "alone with its copy");
    }

    /// Pipeline 1 made the component "Outer": local Nuclei (L1, extracted in
    /// step 3) and Seeds (L2), the parameter "Cells" sets the Voronoi's
    /// output class (step 5).
    fn with_outer_component() -> ProjectSettings {
        let mut project = project();
        let plan = plan_component(&project, PipelineId(1), 0..=4).unwrap();
        create_component(&mut project, PipelineId(1), 0..=4, "Outer", &roles(&plan)).unwrap();
        project
    }

    /// What a step using component `key` with `values` runs, as JSON - scopes
    /// left out.
    fn runs(
        project: &ProjectSettings,
        key: &str,
        values: &[(&str, ParamValue)],
    ) -> Vec<serde_json::Value> {
        let mut step = component_step(key);
        if let StepContent::Component(instance) = &mut step.content {
            instance.values = values
                .iter()
                .map(|(k, v)| (ComponentParamKey::from(*k), v.clone()))
                .collect();
        }
        expand_step(&step, &project.components, &mut ScopeCounter::default())
            .unwrap()
            .into_iter()
            .map(|c| serde_json::to_value(c.command).unwrap())
            .collect()
    }

    fn outer(project: &ProjectSettings) -> &ComponentDefinition {
        project
            .components
            .iter()
            .find(|d| d.key == "OUTER")
            .unwrap()
    }

    #[test]
    fn a_component_made_inside_a_component_runs_what_its_steps_ran() {
        let mut project = with_outer_component();
        let before = runs(
            &project,
            "OUTER",
            &[("cells", ParamValue::Text("4".into()))],
        );
        let list = StepList::Component("OUTER".into());

        let plan = plan_component(&project, list.clone(), 2..=4).unwrap();
        let planned: Vec<(ObjectClass, ClassRole)> = roles(&plan);
        // Segmentation class 1 comes from the Threshold staying outside (only a
        // segmentation class here: no parameter comes of it). Cells is set by
        // the outer parameter - not a class of the steps' own.
        assert_eq!(
            planned,
            [
                (ObjectClass::Global(1), ClassRole::Parameter),
                (ObjectClass::Local(1), ClassRole::Local),
                (ObjectClass::Local(2), ClassRole::Local)
            ],
        );
        let names: Vec<&str> = plan.classes.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Nuclei", "Nuclei", "Seeds"]);

        let key = create_component(&mut project, list, 2..=4, "Inner", &planned).unwrap();
        let inner = project.components.iter().find(|d| d.key == key).unwrap();
        assert_eq!(inner.steps.len(), 3);
        let inner_classes: Vec<(u32, &str)> = inner
            .classes
            .iter()
            .map(|c| (c.id, c.name.as_str()))
            .collect();
        assert_eq!(inner_classes, [(1, "Nuclei"), (2, "Seeds")], "same numbers");
        assert_eq!(inner.parameters.len(), 1);
        assert_eq!(inner.parameters[0].label, "Cells");
        assert_eq!(
            inner.parameters[0].bindings,
            [ParamBinding {
                step: StepId(5),
                param: "output_class".into()
            }]
        );

        let outer = outer(&project);
        assert!(outer.classes.is_empty(), "the classes went with the steps");
        assert_eq!(outer.steps.len(), 3);
        let StepContent::Component(instance) = &outer.steps[2].content else {
            panic!("a step using the new component");
        };
        assert_eq!(instance.key, key);
        assert_eq!(
            outer.parameters[0].bindings,
            [ParamBinding {
                step: outer.steps[2].id,
                param: inner.parameters[0].key.to_string()
            }],
            "the outer parameter sets the new one's"
        );

        assert_eq!(
            runs(
                &project,
                "OUTER",
                &[("cells", ParamValue::Text("4".into()))]
            ),
            before,
            "the outer component runs what it ran"
        );
    }

    #[test]
    fn an_outer_parameter_on_any_setting_of_the_steps_still_sets_it() {
        let mut project = with_outer_component();
        let outer_definition = project
            .components
            .iter_mut()
            .find(|d| d.key == "OUTER")
            .unwrap();
        // Step 4 (classify) min_area as "Min size", next to "Cells".
        assert!(toggle_exposed(
            outer_definition,
            StepId(4),
            "min_area",
            "Min size",
            "50"
        ));
        let min_size = outer_definition.parameters[1].key.to_string();
        let values = [(min_size.as_str(), ParamValue::Float(99.0))];
        let before = runs(&project, "OUTER", &values);

        let list = StepList::Component("OUTER".into());
        let plan = plan_component(&project, list.clone(), 2..=4).unwrap();
        create_component(&mut project, list, 2..=4, "Inner", &roles(&plan)).unwrap();

        let after = runs(&project, "OUTER", &values);
        assert_eq!(after, before);
        assert!(
            after.iter().any(|c| c["minArea"] == 99.0),
            "the value reaches the classify step: {after:?}"
        );
        let outer = outer(&project);
        let labels: Vec<&str> = outer.parameters.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, ["Cells", "Min size"], "the card stays the same");
    }

    #[test]
    fn an_outer_components_class_its_other_steps_use_too_is_handed_in() {
        let mut project = with_outer_component();
        let before = runs(
            &project,
            "OUTER",
            &[("cells", ParamValue::Text("4".into()))],
        );
        let list = StepList::Component("OUTER".into());
        // Nuclei (L1) comes from step 3, which stays in "Outer"; Seeds (L2)
        // only these steps use.
        let plan = plan_component(&project, list.clone(), 3..=4).unwrap();
        assert_eq!(
            roles(&plan),
            [
                (ObjectClass::Local(1), ClassRole::Parameter),
                (ObjectClass::Local(2), ClassRole::Local)
            ]
        );

        let key = create_component(&mut project, list, 3..=4, "Inner", &roles(&plan)).unwrap();
        let inner = project.components.iter().find(|d| d.key == key).unwrap();
        let inner_classes: Vec<u32> = inner.classes.iter().map(|c| c.id).collect();
        assert_eq!(inner_classes, [2], "Seeds moved");
        let nuclei = inner
            .parameters
            .iter()
            .find(|p| p.label == "Nuclei")
            .unwrap();
        assert_eq!(
            nuclei.default,
            ParamValue::Text("-1".into()),
            "unset on its own"
        );
        let StepContent::Component(uses_inner) = &outer(&project).steps[3].content else {
            panic!("the step using the new component")
        };
        assert_eq!(
            uses_inner.values[&nuclei.key],
            ParamValue::Text("L1".into()),
            "the outer component hands its class in"
        );
        let outer_classes: Vec<u32> = outer(&project).classes.iter().map(|c| c.id).collect();
        assert_eq!(outer_classes, [1], "Nuclei stays the outer one's");

        assert_eq!(
            runs(
                &project,
                "OUTER",
                &[("cells", ParamValue::Text("4".into()))]
            ),
            before.iter().cloned().collect::<Vec<_>>(),
            "the outer component runs what it ran"
        );
    }

    #[test]
    fn expanding_a_component_inside_a_component_runs_what_it_ran() {
        let mut project = with_outer_component();
        let values = [("cells", ParamValue::Text("4".into()))];
        let before = runs(&project, "OUTER", &values);
        let list = StepList::Component("OUTER".into());
        // Seeds (L2) moves into "Inner", Nuclei (L1) is handed in.
        let plan = plan_component(&project, list.clone(), 3..=4).unwrap();
        create_component(&mut project, list.clone(), 3..=4, "Inner", &roles(&plan)).unwrap();
        assert_eq!(outer(&project).steps.len(), 4);

        expand_component(&mut project, list, 3).unwrap();
        let outer_definition = outer(&project);
        assert_eq!(outer_definition.steps.len(), 5, "the steps back in place");
        let classes: Vec<(u32, &str)> = outer_definition
            .classes
            .iter()
            .map(|c| (c.id, c.name.as_str()))
            .collect();
        assert_eq!(
            classes,
            [(1, "Nuclei"), (2, "Seeds")],
            "Inner's Seeds, numbered after the outer one's own"
        );
        assert_eq!(
            outer_definition.parameters[0].bindings,
            [ParamBinding {
                step: outer_definition.steps[4].id,
                param: "output_class".into()
            }],
            "\"Cells\" sets the Voronoi's class again"
        );
        assert_eq!(runs(&project, "OUTER", &values), before);
    }

    #[test]
    fn expanding_a_component_into_the_pipeline_makes_what_it_hands_in_project_classes() {
        let mut project = with_outer_component();
        let list = StepList::Component("OUTER".into());
        let plan = plan_component(&project, list.clone(), 3..=4).unwrap();
        create_component(&mut project, list, 3..=4, "Inner", &roles(&plan)).unwrap();

        expand_component(&mut project, PipelineId(1), 0).unwrap();
        let nuclei = project
            .classification
            .classes()
            .iter()
            .find(|c| c.name == "Nuclei" && c.id != 1)
            .map(|c| c.id)
            .expect("Outer's Nuclei, a project class now");
        let pipeline = &project.pipelines[0];
        let StepContent::Component(uses_inner) = &pipeline.steps[3].content else {
            panic!("the step using Inner")
        };
        let handed: Vec<&ParamValue> = uses_inner.values.values().collect();
        assert!(
            handed.contains(&&ParamValue::Text(nuclei.to_string())),
            "{handed:?}"
        );
        assert!(
            handed.iter().all(|v| local_class_of(v).is_none()),
            "no component class left in the pipeline: {handed:?}"
        );
        for step in &pipeline.steps {
            expand_step(step, &project.components, &mut ScopeCounter::default())
                .expect("every step runs in the pipeline");
        }
    }

    fn image_cache(
        mode: evanalyzer_cfg::settings::pipeline_command_settings::MathImageCacheImageCacheModeSettings,
        slot: u32,
    ) -> PipelineCommand {
        PipelineCommand::ImageCache(
            evanalyzer_cfg::settings::pipeline_command_settings::ImageCacheSettings {
                mode,
                address: ImageAddress::Memory(MemoryId::ComponentContext(slot)),
            },
        )
    }

    /// "Slots": stores in its slot 1, blurs, loads slot 1 back.
    fn with_slots_component(last: PipelineCommand) -> ProjectSettings {
        use evanalyzer_cfg::settings::pipeline_command_settings::MathImageCacheImageCacheModeSettings as Mode;
        let mut project = project();
        project.components.push(ComponentDefinition {
            key: "SLOTS".into(),
            meta: MetaData::default(),
            classes: vec![],
            steps: vec![
                step(1, image_cache(Mode::Store, 1)),
                step(2, PipelineCommand::Blur(Default::default())),
                step(3, last),
            ],
            parameters: vec![],
            origin: None,
        });
        project
    }

    #[test]
    fn an_outer_components_memory_slot_its_other_steps_use_too_is_handed_in() {
        use evanalyzer_cfg::settings::pipeline_command_settings::MathImageCacheImageCacheModeSettings as Mode;
        let mut project = with_slots_component(image_cache(Mode::Load, 1));
        let before = runs(&project, "SLOTS", &[]);
        let list = StepList::Component("SLOTS".into());
        let plan = plan_component(&project, list.clone(), 2..=2).unwrap();
        assert_eq!(plan.handed_slots, BTreeSet::from([1]));

        let key = create_component(&mut project, list, 2..=2, "Load", &roles(&plan)).unwrap();
        let inner = project.components.iter().find(|d| d.key == key).unwrap();
        assert_eq!(inner.parameters.len(), 1);
        let slot = &inner.parameters[0];
        assert_eq!(slot.label, "Memory slot 1");
        assert_eq!(
            slot.default,
            ParamValue::Text("none".into()),
            "empty on its own"
        );
        assert_eq!(
            slot.bindings,
            [ParamBinding {
                step: StepId(3),
                param: "address".into()
            }]
        );
        let outer = project
            .components
            .iter()
            .find(|d| d.key == "SLOTS")
            .unwrap();
        let StepContent::Component(uses_inner) = &outer.steps[2].content else {
            panic!("the step using the new component")
        };
        assert_eq!(
            uses_inner.values[&slot.key],
            ParamValue::Text("component:1".into()),
            "the outer component hands its slot in"
        );
        assert_eq!(
            runs(&project, "SLOTS", &[]),
            before,
            "the outer component runs what it ran"
        );
    }

    #[test]
    fn a_shared_slot_a_step_uses_other_than_as_an_address_is_refused() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            SegmentationThresholdThresholdValueSourceSettings, ThresholdEntrySettings,
            ThresholdSettings,
        };
        let threshold = PipelineCommand::Threshold(ThresholdSettings {
            thresholds: vec![ThresholdEntrySettings {
                value_source: SegmentationThresholdThresholdValueSourceSettings::Memory(
                    MemoryId::ComponentContext(1),
                ),
                ..Default::default()
            }],
        });
        let project = with_slots_component(threshold);
        let error = plan_component(&project, StepList::Component("SLOTS".into()), 2..=2)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("step 3 uses it in a way that can't be handed"),
            "{error}"
        );
    }

    #[test]
    fn a_parameter_can_also_set_the_settings_with_its_control_and_lose_one_again() {
        let mut project = with_outer_component();
        let key = ComponentParamKey::from("cells");
        let bindable = bindable_settings(&project.components, "OUTER", &key);
        let paths: Vec<(StepId, &str)> =
            bindable.iter().map(|(s, p, _)| (*s, p.as_str())).collect();
        assert!(paths.contains(&(StepId(4), "output_class")), "{paths:?}");
        assert!(
            paths.contains(&(StepId(4), "input_classes")),
            "a class list takes a single class too: {paths:?}"
        );
        assert!(
            !paths.contains(&(StepId(5), "output_class")),
            "already set by it"
        );
        assert!(
            !paths.iter().any(|(_, p)| *p == "min_area"),
            "a number isn't a class"
        );

        let outer = project
            .components
            .iter_mut()
            .find(|d| d.key == "OUTER")
            .unwrap();
        add_binding(outer, &key, StepId(4), "output_class").unwrap();
        assert!(remove_binding(outer, &key, StepId(5), "output_class"));
        assert_eq!(outer.parameters[0].bindings.len(), 1, "the other stays");
        assert!(
            !remove_binding(outer, &key, StepId(5), "output_class"),
            "gone already"
        );
        assert!(remove_binding(outer, &key, StepId(4), "output_class"));
        assert!(outer.parameters.is_empty(), "setting nothing, it goes");
    }

    #[test]
    fn a_class_list_another_parameter_sets_can_be_set_too_a_single_class_cant() {
        let mut project = with_outer_component();
        let outer = project
            .components
            .iter_mut()
            .find(|d| d.key == "OUTER")
            .unwrap();
        // "Cells" sets the Voronoi's class; "Seeds class" a classify's input list.
        assert!(toggle_exposed(
            outer,
            StepId(4),
            "input_classes",
            "Seeds class",
            "L1"
        ));
        let cells = ComponentParamKey::from("cells");
        let seeds = outer.parameters[1].key.clone();

        let bindable = |project: &ProjectSettings, key: &ComponentParamKey| {
            bindable_settings(&project.components, "OUTER", key)
                .into_iter()
                .map(|(step, path, _)| (step, path))
                .collect::<Vec<_>>()
        };
        assert!(
            bindable(&project, &cells).contains(&(StepId(4), "input_classes".into())),
            "a class list \"Seeds class\" sets: \"Cells\" can add its class"
        );
        assert!(
            !bindable(&project, &seeds).contains(&(StepId(4), "input_classes".into())),
            "not twice by the same parameter"
        );
        assert!(
            !bindable(&project, &seeds).contains(&(StepId(5), "output_class".into())),
            "a single class \"Cells\" sets: one value only"
        );

        // Both on the list: editing it there changes neither default.
        let outer = project
            .components
            .iter_mut()
            .find(|d| d.key == "OUTER")
            .unwrap();
        add_binding(outer, &cells, StepId(4), "input_classes").unwrap();
        let defaults: Vec<ParamValue> =
            outer.parameters.iter().map(|p| p.default.clone()).collect();
        assert!(!bound_setting_edited(
            outer,
            &[],
            StepId(4),
            "input_classes"
        ));
        let after: Vec<ParamValue> = outer.parameters.iter().map(|p| p.default.clone()).collect();
        assert_eq!(after, defaults);
    }

    #[test]
    fn an_outer_components_class_cant_stay_as_it_is_in_the_new_one() {
        let mut project = with_outer_component();
        let list = StepList::Component("OUTER".into());
        let error = create_component(
            &mut project,
            list,
            2..=4,
            "Inner",
            &[
                (ObjectClass::Local(1), ClassRole::Fixed),
                (ObjectClass::Local(2), ClassRole::Local),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("Nuclei can't stay as it is"), "{error}");
        assert_eq!(outer(&project).steps.len(), 5, "nothing changed");
    }

    #[test]
    fn an_outer_class_with_the_number_a_new_local_class_gets_stays() {
        let mut project = with_outer_component();
        // Seeds (L2) comes from step 4, which stays: steps 5 only uses Cells
        // (set by the outer parameter) - and a project class, made local in
        // the new component as L1.
        let outer_definition = project
            .components
            .iter_mut()
            .find(|d| d.key == "OUTER")
            .unwrap();
        outer_definition.steps.push(step(
            6,
            classify(ObjectClass::Global(3), ObjectClass::Global(9)),
        ));
        let list = StepList::Component("OUTER".into());
        let plan = plan_component(&project, list.clone(), 5..=5).unwrap();
        let roles: Vec<(ObjectClass, ClassRole)> = plan
            .classes
            .iter()
            .map(|c| (c.class, ClassRole::Local))
            .collect();
        let key = create_component(&mut project, list, 5..=5, "Inner", &roles).unwrap();
        let inner = project.components.iter().find(|d| d.key == key).unwrap();
        assert!(inner.classes.iter().any(|c| c.id == 1));
        let outer_classes: Vec<u32> = outer(&project).classes.iter().map(|c| c.id).collect();
        assert_eq!(
            outer_classes,
            [1, 2],
            "the outer one's own L1 isn't the new one's"
        );
    }

    #[test]
    fn a_component_containing_another_at_any_depth_would_contain_itself_in_it() {
        let mut project = with_component();
        let uses = |key: &str| PipelineStepSettings {
            id: StepId(1),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: key.into(),
                values: Default::default(),
            }),
        };
        let mut middle = project.components[0].clone();
        middle.key = "MIDDLE".into();
        middle.steps = vec![uses("CELL_RECONSTRUCTION")];
        let mut top = project.components[0].clone();
        top.key = "TOP".into();
        top.steps = vec![uses("MIDDLE")];
        project.components.extend([middle, top]);

        let contains = |container, inserted| would_contain_itself(&project, container, inserted);
        assert!(contains("CELL_RECONSTRUCTION", "CELL_RECONSTRUCTION"));
        assert!(contains("CELL_RECONSTRUCTION", "MIDDLE"));
        assert!(contains("CELL_RECONSTRUCTION", "TOP"), "two levels down");
        assert!(!contains("TOP", "CELL_RECONSTRUCTION"));
        assert!(!contains("TOP", "MIDDLE"));
        assert!(!contains("TOP", "UNKNOWN"));
    }

    #[test]
    fn a_component_nested_in_another_gets_a_copy_for_its_step_there() {
        let mut project = with_component();
        // "Outer" uses the component; the pipeline does too.
        let mut outer = project.components[0].clone();
        outer.key = "OUTER".into();
        outer.meta.name = "Outer".into();
        outer.classes.clear();
        outer.parameters.clear();
        outer.steps = vec![project.pipelines[0].steps[2].clone()];
        project.components.push(outer);
        assert_eq!(component_uses(&project, "CELL_RECONSTRUCTION"), 2);

        let key = clone_component_for_step(&mut project, &StepList::Component("OUTER".into()), 0, None)
            .unwrap();
        let outer = project
            .components
            .iter()
            .find(|d| d.key == "OUTER")
            .unwrap();
        assert!(matches!(&outer.steps[0].content, StepContent::Component(i) if i.key == key));
        assert_eq!(
            component_uses(&project, "CELL_RECONSTRUCTION"),
            1,
            "the pipeline keeps it"
        );

        let error = clone_component_for_step(&mut project, &StepList::Component("OUTER".into()), 5, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("component OUTER has no step 6"), "{error}");
    }

    #[test]
    fn a_setting_is_exposed_and_hidden_again() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        let before = definition.parameters.len();

        assert!(toggle_exposed(
            definition,
            StepId(4),
            "min_area",
            "Minimum size",
            "50"
        ));
        assert_eq!(definition.parameters.len(), before + 1);
        let added = definition.parameters.last().unwrap();
        assert_eq!(
            (added.label.as_str(), &added.default),
            ("Minimum size", &ParamValue::Int(50))
        );
        assert!(exposed_settings(definition, StepId(4)).contains(&"min_area".to_string()));

        assert!(!toggle_exposed(
            definition,
            StepId(4),
            "min_area",
            "Minimum size",
            "50"
        ));
        assert_eq!(
            definition.parameters.len(),
            before,
            "nothing bound to it any more"
        );
    }

    #[test]
    fn a_parameter_is_renamed_described_and_moved() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        toggle_exposed(definition, StepId(4), "min_area", "Minimum size", "50");
        let ids = |d: &ComponentDefinition| {
            d.parameters
                .iter()
                .map(|p| p.key.clone())
                .collect::<Vec<_>>()
        };
        let before = ids(definition);
        let last = before.last().unwrap().clone();

        assert!(edit_parameter(
            definition,
            &last,
            "Min. area",
            "Smaller objects are dropped"
        ));
        let edited = definition.parameters.last().unwrap();
        assert_eq!(
            (edited.label.as_str(), edited.description.as_str()),
            ("Min. area", "Smaller objects are dropped")
        );
        assert!(!edit_parameter(
            definition,
            &ComponentParamKey::from("999"),
            "x",
            ""
        ));

        assert!(set_parameter_section(definition, &last, "  Filtering "));
        assert_eq!(definition.parameters.last().unwrap().section, "Filtering");
        assert!(set_parameter_section(definition, &last, "   "));
        assert_eq!(definition.parameters.last().unwrap().section, "");
        assert!(!set_parameter_section(
            definition,
            &ComponentParamKey::from("999"),
            "x"
        ));

        assert!(move_parameter(definition, &last, -1));
        let mut expected = before.clone();
        let n = expected.len();
        expected.swap(n - 1, n - 2);
        assert_eq!(ids(definition), expected);
        assert!(move_parameter(definition, &last, 1));
        assert_eq!(ids(definition), before);
        assert!(!move_parameter(definition, &last, 1), "already last");
        assert!(!move_parameter(definition, &before[0], -1), "already first");
    }

    #[test]
    fn a_parameter_is_found_by_its_setting_and_removed() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        toggle_exposed(definition, StepId(4), "min_area", "Minimum size", "50");
        let id = parameter_for_setting(definition, StepId(4), "min_area").unwrap();
        assert_eq!(
            parameter_for_setting(definition, StepId(4), "max_area"),
            None
        );

        assert!(remove_parameter(definition, &id));
        assert_eq!(
            parameter_for_setting(definition, StepId(4), "min_area"),
            None
        );
        assert!(!remove_parameter(definition, &id), "gone already");
    }

    /// Threshold → Extract Objects (Seg 1 → class 1) → Classify, class 1 as
    /// a parameter: the parameter is the class the objects get - the
    /// segmentation class stays the component's own, and the check is happy.
    #[test]
    fn a_class_parameter_sets_the_object_classes_not_the_segmentation_class() {
        use evanalyzer_cfg::settings::pipeline_command_settings::ExtractClassMappingSettings;
        let mut project = project();
        project.pipelines[0].steps = vec![
            step(1, threshold(1)),
            step(
                2,
                PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                    class_mapping: vec![ExtractClassMappingSettings {
                        segmentation_class: SegmentationClass(1),
                        object_class: NUCLEI,
                    }],
                    ..Default::default()
                }),
            ),
            step(3, classify(NUCLEI, ObjectClass::Unset)),
        ];
        create_component(
            &mut project,
            PipelineId(1),
            0..=2,
            "Detect",
            &[(NUCLEI, ClassRole::Parameter)],
        )
        .unwrap();

        let definition = &project.components[0];
        let class_parameters: Vec<&ComponentParameterDef> = definition
            .parameters
            .iter()
            .filter(|p| p.label == "Nuclei")
            .collect();
        assert_eq!(class_parameters.len(), 1);
        let bound: Vec<&str> = class_parameters[0]
            .bindings
            .iter()
            .map(|b| b.param.as_str())
            .collect();
        assert_eq!(
            bound,
            ["class_mapping.0.object_class", "input_classes"],
            "not the Threshold's or the mapping's segmentation class"
        );
        let problems =
            crate::workspace::component_checks::check_component(&project, &definition.key);
        assert!(
            !problems
                .iter()
                .any(|p| p.severity == crate::workspace::component_checks::Severity::Error),
            "{:?}",
            problems.iter().map(|p| &p.message).collect::<Vec<_>>()
        );

        // Shown as the list (several classes) it couldn't set the mapping's
        // single class.
        let id = class_parameters[0].key.clone();
        let definition = &mut project.components[0];
        let parameter = definition
            .parameters
            .iter_mut()
            .find(|p| p.key == id)
            .unwrap();
        parameter.bindings.reverse();
        let key = definition.key.clone();
        let problems = crate::workspace::component_checks::check_component(&project, &key);
        assert!(
            problems
                .iter()
                .any(|p| p.message.contains("different kinds of setting")),
            "{:?}",
            problems.iter().map(|p| &p.message).collect::<Vec<_>>()
        );
    }

    /// A setting a parameter sets: editing it in the opened component changes the
    /// parameter's default (what runs), and the other settings it sets
    /// follow.
    #[test]
    fn editing_a_bound_setting_changes_the_parameters_default() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        toggle_exposed(definition, StepId(4), "min_area", "Minimum size", "50");
        let id = parameter_for_setting(definition, StepId(4), "min_area").unwrap();
        let step = definition
            .steps
            .iter_mut()
            .find(|s| s.id == StepId(4))
            .unwrap();
        step.command_mut()
            .unwrap()
            .apply_param_change("min_area", "80");

        assert!(bound_setting_edited(definition, &[], StepId(4), "min_area"));
        let parameter = definition.parameters.iter().find(|p| p.key == id).unwrap();
        assert_eq!(parameter.default, ParamValue::Int(80));
        // What runs for a use without values of its own.
        let instance = ComponentInstance {
            key: definition.key.clone(),
            values: Default::default(),
        };
        let steps = evanalyzer_core::component_steps_with_values(
            &instance,
            std::slice::from_ref(&*definition),
        )
        .unwrap();
        let classify = steps.iter().find(|s| s.id == StepId(4)).unwrap();
        let Some(PipelineCommand::ClassifyObjects(c)) = classify.command() else {
            panic!("classify")
        };
        assert_eq!(c.min_area, 80.0);

        assert!(
            !bound_setting_edited(definition, &[], StepId(4), "max_area"),
            "not set by a parameter"
        );
    }

    #[test]
    fn parameters_get_readable_unique_keys_that_stay_when_relabelled() {
        let mut project = with_component();
        let keys = |d: &ComponentDefinition| {
            d.parameters
                .iter()
                .map(|p| p.key.to_string())
                .collect::<Vec<_>>()
        };
        // Creating it: the class parameters are named after their classes.
        assert!(keys(&project.components[0]).contains(&"cells".to_string()));

        let definition = &mut project.components[0];
        toggle_exposed(definition, StepId(4), "min_area", "Min. Size (px)", "50");
        toggle_exposed(definition, StepId(4), "max_area", "Min. Size (px)", "500");
        assert!(keys(definition).ends_with(&["min_size_px".into(), "min_size_px_2".into()]));

        let key = ComponentParamKey::from("min_size_px");
        assert!(edit_parameter(definition, &key, "Smallest", ""));
        assert!(
            keys(definition).contains(&"min_size_px".to_string()),
            "the key stays"
        );

        let empty = ComponentDefinition {
            parameters: vec![],
            ..definition.clone()
        };
        assert_eq!(new_parameter_key(&empty, "  ").as_str(), "parameter");
        assert_eq!(
            new_parameter_key(&empty, "2nd class").as_str(),
            "p_2nd_class"
        );
    }

    #[test]
    fn values_for_parameters_a_component_no_longer_has_are_dropped() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        toggle_exposed(definition, StepId(4), "min_area", "Minimum size", "50");
        let key = parameter_for_setting(definition, StepId(4), "min_area").unwrap();
        let StepContent::Component(instance) = &mut project.pipelines[0].steps[2].content else {
            panic!("component step")
        };
        instance.values.insert(key.clone(), ParamValue::Int(80));
        instance.values.insert("cells".into(), ParamValue::Int(3));
        assert_eq!(prune_component_values(&mut project), 0, "both exist");

        remove_parameter(&mut project.components[0], &key);
        assert_eq!(prune_component_values(&mut project), 1);
        let StepContent::Component(instance) = &project.pipelines[0].steps[2].content else {
            panic!("component step")
        };
        assert!(!instance.values.contains_key(&key));
        assert!(
            instance
                .values
                .contains_key(&ComponentParamKey::from("cells")),
            "kept"
        );
    }

    #[test]
    fn removing_a_step_of_a_component_removes_the_bindings_to_it() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        // "Cells" is bound to the Voronoi (step 5) only.
        let index = definition
            .steps
            .iter()
            .position(|s| s.id == StepId(5))
            .unwrap();
        remove_component_step(definition, index);
        let labels: Vec<&str> = definition
            .parameters
            .iter()
            .map(|p| p.label.as_str())
            .collect();
        assert_eq!(labels, ["Nuclei"]);
    }

    // ---- List-entry settings, more settings for one parameter ----

    #[test]
    fn a_setting_of_a_list_entry_is_found_with_a_readable_label() {
        let (setting, label) = setting_at(&threshold(1), "thresholds.0.min_threshold").unwrap();
        assert_eq!(setting.name, "min_threshold");
        assert!(label.starts_with("Thresholds 1 · "), "{label}");
        assert!(setting_at(&threshold(1), "thresholds.5.min_threshold").is_none());
    }

    #[test]
    fn a_list_entry_setting_exposed_is_set_by_the_components_parameter() {
        let mut project = project();
        create_component(&mut project, PipelineId(1), 0..=0, "Detect", &[]).unwrap();
        let definition = &mut project.components[0];
        assert!(toggle_exposed(
            definition,
            StepId(1),
            "thresholds.0.min_threshold",
            "Minimum",
            "0"
        ));
        let id = definition.parameters[0].key.clone();
        let StepContent::Component(instance) = &mut project.pipelines[0].steps[0].content else {
            panic!("component step")
        };
        instance.values.insert(id, ParamValue::Int(1200));

        let commands = commands_of_step(&project, 0, 0);
        let PipelineCommand::Threshold(threshold) = &commands[0] else {
            panic!("threshold")
        };
        assert_eq!(threshold.thresholds[0].min_threshold, 1200.0);
    }

    #[test]
    fn a_second_setting_can_join_a_parameter_with_the_same_control() {
        let mut project = with_component();
        let key = project.components[0].key.clone();
        let voronoi = project.components[0]
            .steps
            .iter()
            .find(|s| s.id == StepId(5))
            .and_then(|s| s.command().cloned())
            .unwrap();
        let (mask, _) = setting_at(&voronoi, "mask").unwrap();

        // "Cells" (a class) fits, "Nuclei" (a class list) doesn't.
        let candidates = compatible_parameters(&project.components, &key, &mask);
        let labels: Vec<&str> = candidates.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(labels, ["Cells"]);

        add_binding(
            &mut project.components[0],
            &candidates[0].0,
            StepId(5),
            "mask",
        )
        .unwrap();
        let commands = commands_of_step(&project, 0, 2);
        let PipelineCommand::Voronoi(voronoi) = &commands[2] else {
            panic!("voronoi")
        };
        assert_eq!(voronoi.mask, CELLS, "set by the parameter too");
        assert!(
            add_binding(
                &mut project.components[0],
                &ComponentParamKey::from("99"),
                StepId(5),
                "mask"
            )
            .is_err()
        );
    }

    #[test]
    fn removing_a_list_entry_keeps_the_bindings_on_their_entries() {
        let mut definition = ComponentDefinition {
            key: "T".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![],
            parameters: vec![],
            origin: None,
        };
        let bind = |definition: &mut ComponentDefinition, step: u32, path: &str| {
            toggle_exposed(definition, StepId(step), path, path, "1");
        };
        bind(&mut definition, 1, "thresholds.0.min_threshold");
        bind(&mut definition, 1, "thresholds.1.min_threshold");
        bind(&mut definition, 1, "thresholds.2.method.classes");
        bind(&mut definition, 1, "thresholds_other.1.x"); // another list
        bind(&mut definition, 2, "thresholds.2.min_threshold"); // another step

        list_entry_removed(&mut definition, StepId(1), "thresholds", 1);

        let bound: Vec<(u32, String)> = definition
            .parameters
            .iter()
            .flat_map(|p| p.bindings.iter())
            .map(|b| (b.step.0, b.param.clone()))
            .collect();
        assert_eq!(
            bound,
            [
                (1, "thresholds.0.min_threshold".to_string()),
                (1, "thresholds.1.method.classes".to_string()),
                (1, "thresholds_other.1.x".to_string()),
                (2, "thresholds.2.min_threshold".to_string()),
            ]
        );
        assert_eq!(
            definition.parameters.len(),
            4,
            "the removed entry's parameter went"
        );
    }

    // ---- Removing components ----

    #[test]
    fn only_components_no_pipeline_reaches_count_as_unused_and_can_be_removed() {
        let mut project = with_component(); // CELL_RECONSTRUCTION, used by pipeline 1
        let definition = |key: &str, steps: Vec<PipelineStepSettings>| ComponentDefinition {
            key: key.into(),
            meta: Default::default(),
            classes: vec![],
            steps,
            parameters: vec![],
            origin: None,
        };
        let uses = |key: &str| PipelineStepSettings {
            id: StepId(1),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: key.into(),
                values: Default::default(),
            }),
        };
        // OUTER is used by nothing; INNER only by OUTER.
        project
            .components
            .push(definition("OUTER", vec![uses("INNER")]));
        project.components.push(definition("INNER", vec![]));
        assert_eq!(unused_components(&project), ["OUTER", "INNER"]);

        assert!(
            remove_component(&mut project, "CELL_RECONSTRUCTION").is_err(),
            "still used"
        );
        remove_component(&mut project, "INNER").unwrap();
        assert_eq!(remove_unused_components(&mut project), ["OUTER"]);
        let keys: Vec<&str> = project.components.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["CELL_RECONSTRUCTION"]);
    }

    #[test]
    fn only_component_steps_can_be_expanded_or_get_a_copy() {
        let mut project = with_component();

        let expand = expand_component(&mut project, PipelineId(1), 0)
            .unwrap_err()
            .to_string();
        let copy = clone_component_for_step(&mut project, &StepList::Pipeline(PipelineId(1)), 0, None)
            .unwrap_err()
            .to_string();

        assert!(expand.contains("step 1 isn't a component"), "{expand}");
        assert!(copy.contains("step 1 isn't a component"), "{copy}");
    }

    #[test]
    fn a_missing_pipeline_step_is_named_in_the_error() {
        let mut project = with_component();

        let past_the_end = expand_component(&mut project, PipelineId(1), 9)
            .unwrap_err()
            .to_string();
        let no_pipeline = expand_component(&mut project, PipelineId(7), 0)
            .unwrap_err()
            .to_string();

        assert!(past_the_end.contains("has no step 10"), "{past_the_end}");
        assert!(no_pipeline.contains("has no step 1"), "{no_pipeline}");
    }

    #[test]
    fn a_step_whose_component_is_missing_gets_no_copy() {
        let mut project = with_component();
        project.components.clear();

        let error = clone_component_for_step(&mut project, &StepList::Pipeline(PipelineId(1)), 2, None)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("no component CELL_RECONSTRUCTION"),
            "{error}"
        );
    }

    #[test]
    fn a_name_without_letters_or_digits_gets_a_generic_numbered_key() {
        let mut project = ProjectSettings::default();
        assert_eq!(unique_key(&project, " -- "), "COMPONENT");

        let mut taken = with_component().components.remove(0);
        taken.key = "COMPONENT".into();
        project.components.push(taken);

        assert_eq!(unique_key(&project, "!!"), "COMPONENT_2");
        assert_eq!(unique_key(&project, "a  b--c"), "A_B_C", "runs collapse");
    }

    #[test]
    fn moving_or_unbinding_an_unknown_parameter_changes_nothing() {
        let mut project = with_component();
        let definition = &mut project.components[0];
        let before = definition.parameters.len();
        let unknown = ComponentParamKey::from("999");

        assert!(!move_parameter(definition, &unknown, 1));
        assert!(!remove_binding(
            definition,
            &unknown,
            StepId(1),
            "output_class"
        ));
        assert_eq!(definition.parameters.len(), before);
    }
}
