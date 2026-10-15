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
        ComponentParamKey, InternalErrors, ObjectClass, PipelineId, SegmentationClass, StepId,
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
}

/// The project classes steps `range` of pipeline `pipeline_id` use, each with
/// a suggested role: used elsewhere too → [`ClassRole::Parameter`]; only
/// here → [`ClassRole::Local`] - except a class the steps only know as a
/// segmentation class with no Extract Objects among them to turn it into
/// objects ([`ClassRole::Fixed`]: making it local would change nothing).
pub fn plan_component(
    project: &ProjectSettings,
    pipeline_id: PipelineId,
    range: RangeInclusive<usize>,
) -> Result<ComponentPlan, InternalErrors> {
    let steps = selected_steps(project, pipeline_id, &range)?;
    let elsewhere = classes_used_elsewhere(project, pipeline_id, &range);
    let extracts = steps
        .iter()
        .filter_map(|s| s.command())
        .any(|c| matches!(c, PipelineCommand::ExtractObjects(_)));

    let mut used: BTreeSet<ObjectClass> = BTreeSet::new();
    for step in steps {
        match &step.content {
            StepContent::Command(command) => used.extend(command.object_classes()),
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

    let classes = used
        .into_iter()
        .map(|class| {
            let used_elsewhere = elsewhere.contains(&class) || nested.contains(&class);
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
                name: class_name(project, class),
                used_elsewhere,
                suggested,
            }
        })
        .collect();

    let channels = steps
        .iter()
        .filter_map(|s| s.command())
        .filter_map(|c| match c {
            PipelineCommand::LoadChannel(s) => Some(s.channel.0),
            _ => None,
        })
        .collect();
    Ok(ComponentPlan {
        classes,
        channels,
        component_slots: slots_only_used_by(project, pipeline_id, &range),
    })
}

/// Turns steps `range` of pipeline `pipeline_id` into a component named
/// `name`: a new definition in the project and one step using it in their
/// place. `roles` says what each class of [`plan_component`] becomes (a class
/// left out stays [`ClassRole::Fixed`]); loaded channels become parameters.
/// Returns the new definition's key.
pub fn create_component(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    range: RangeInclusive<usize>,
    name: &str,
    roles: &[(ObjectClass, ClassRole)],
) -> Result<String, InternalErrors> {
    selected_steps(project, pipeline_id, &range)?;
    let name = name.trim();
    if name.is_empty() {
        return Err(InternalErrors::InvalidArgument(
            "a component needs a name".into(),
        ));
    }
    let key = unique_key(project, name);
    let component_slots = slots_only_used_by(project, pipeline_id, &range);
    let names: Vec<(ObjectClass, String, u32)> = roles
        .iter()
        .map(|(class, _)| {
            let color = match class {
                ObjectClass::Global(n) => project
                    .classification
                    .classes()
                    .iter()
                    .find(|c| c.id == *n)
                    .map_or(0x808080, |c| c.color),
                _ => 0x808080,
            };
            (*class, class_name(project, *class), color)
        })
        .collect();

    let pipeline = project
        .pipelines
        .iter_mut()
        .find(|p| p.id == pipeline_id)
        .expect("checked by selected_steps");
    let mut steps: Vec<PipelineStepSettings> = pipeline.steps.drain(range.clone()).collect();

    let mut definition = ComponentDefinition {
        key: key.clone(),
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

    let mut next_local = 1;
    for ((class, role), (_, class_name, color)) in roles.iter().zip(&names) {
        match role {
            ClassRole::Fixed => {}
            ClassRole::Local => {
                let local = ObjectClass::Local(next_local);
                next_local += 1;
                // Made by a segmentation step of these: Extract Objects maps it.
                let segmented_here = steps.iter().filter_map(|s| s.command()).any(|c| {
                    class_paths(c, *class)
                        .iter()
                        .any(|(_, kind)| *kind == PathKind::Segmentation)
                });
                for step in &mut steps {
                    if let StepContent::Command(command) = &mut step.content {
                        make_local(command, *class, local, segmented_here)?;
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
                definition.parameters.push(ComponentParameterDef {
                    key,
                    label: class_name.clone(),
                    description: String::new(),
                    section: String::new(),
                    default: ParamValue::Text(class.to_text()),
                    bindings,
                });
            }
        }
    }

    // Each channel loaded becomes a parameter, bound to every step loading it.
    let mut channels: Vec<(u32, Vec<StepId>)> = Vec::new();
    for step in &steps {
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

    // The pipeline's slots only these steps use become the component's own.
    for step in &mut steps {
        if let StepContent::Command(command) = &mut step.content {
            *command = slots_to_component(command, &component_slots)?;
        }
    }

    definition.steps = steps;
    let start = *range.start();
    pipeline.steps.insert(
        start,
        PipelineStepSettings {
            id: StepId::UNASSIGNED,
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: key.clone(),
                values: Default::default(),
            }),
        },
    );
    assign_step_ids(&mut pipeline.steps);
    project.components.push(definition);
    Ok(key)
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

/// "Only this step": gives step `step_index` of pipeline `pipeline_id` (a
/// component step) a copy of its component - "Name (2)" with a key of its own - so
/// changes to it don't affect the component's other uses. Returns the copy's key.
pub fn clone_component_for_step(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    step_index: usize,
) -> Result<String, InternalErrors> {
    let key = component_key_of(project, pipeline_id, step_index)?;
    let mut copy = project
        .components
        .iter()
        .find(|d| d.key == key)
        .cloned()
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?;
    let base = copy.meta.name.clone();
    let name = (2..)
        .map(|n| format!("{base} ({n})"))
        .find(|name| !project.components.iter().any(|d| d.meta.name == *name))
        .expect("some number is free");
    copy.key = unique_key(project, &name);
    copy.meta.name = name;
    copy.meta.creation_time = Utc::now();
    // Not the library file's version any more.
    copy.origin = None;
    let new_key = copy.key.clone();
    project.components.push(copy);
    if let StepContent::Component(instance) =
        &mut step_mut(project, pipeline_id, step_index)?.content
    {
        instance.key = new_key.clone();
    }
    Ok(new_key)
}

/// "Expand component": replaces step `step_index` of pipeline `pipeline_id` (a component
/// step) by its component's steps, with the step's values applied. The component's
/// own classes become new project classes (same name and colour) and its
/// memory slots pipeline slots - both only exist inside a component. The
/// definition stays in the project.
pub fn expand_component(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    step_index: usize,
) -> Result<(), InternalErrors> {
    let key = component_key_of(project, pipeline_id, step_index)?;
    let StepContent::Component(instance) = &step_mut(project, pipeline_id, step_index)?.content
    else {
        unreachable!("checked by component_key_of");
    };
    let instance = instance.clone();
    let steps = evanalyzer_core::component_steps_with_values(&instance, &project.components)?;
    let definition = project
        .components
        .iter()
        .find(|d| d.key == key)
        .expect("expanded above")
        .clone();

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
            if let StepContent::Command(command) = &mut step.content {
                *command = outside_the_component(command, &to_global)?;
            }
            Ok(step)
        })
        .collect::<Result<_, InternalErrors>>()?;
    let pipeline = project
        .pipelines
        .iter_mut()
        .find(|p| p.id == pipeline_id)
        .expect("checked by component_key_of");
    pipeline
        .steps
        .splice(step_index..=step_index, steps.drain(..));
    assign_step_ids(&mut pipeline.steps);
    Ok(())
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
/// `snake_case` ("Min Size" → `min_size`), with `_2`, `_3` … if taken. It
/// stays when the label changes later - the instances' values are keyed by
/// it.
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

/// Renames parameter `key` of `definition` (its key stays) and sets its
/// help text. Returns whether the parameter exists.
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
/// (its first binding) can also set `other`: the same type and options -
/// or `shown` is a single object class and `other` a class list (the value
/// is then a list of one). Not the other way round: a list picks several
/// classes, a single class setting takes one. A segmentation class is its
/// own kind: a segmentation's number, not a project class.
pub(crate) fn same_kind(shown: &ParameterDef, other: &ParameterDef) -> bool {
    if shown.param_type == ParamType::ObjClass
        && matches!(
            other.param_type,
            ParamType::ObjClass | ParamType::MultiObjClass
        )
    {
        return true;
    }
    shown.param_type == other.param_type && shown.options == other.options
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
    step_id: StepId,
    path: &str,
) -> bool {
    let Some(key) = parameter_for_setting(definition, step_id, path) else {
        return false;
    };
    let Some(value) = definition
        .steps
        .iter()
        .find(|s| s.id == step_id)
        .and_then(|s| s.command())
        .and_then(|command| setting_at(command, path))
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

fn step_mut(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    step_index: usize,
) -> Result<&mut PipelineStepSettings, InternalErrors> {
    project
        .pipelines
        .iter_mut()
        .find(|p| p.id == pipeline_id)
        .and_then(|p| p.steps.get_mut(step_index))
        .ok_or_else(|| {
            InternalErrors::InvalidArgument(format!(
                "pipeline {pipeline_id} has no step {}",
                step_index + 1
            ))
        })
}

/// The key of the component step `step_index` of pipeline `pipeline_id` uses.
fn component_key_of(
    project: &mut ProjectSettings,
    pipeline_id: PipelineId,
    step_index: usize,
) -> Result<String, InternalErrors> {
    match &step_mut(project, pipeline_id, step_index)?.content {
        StepContent::Component(instance) => Ok(instance.key.clone()),
        StepContent::Command(_) => Err(InternalErrors::InvalidArgument(format!(
            "step {} isn't a component",
            step_index + 1
        ))),
    }
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
fn make_local(
    command: &mut PipelineCommand,
    class: ObjectClass,
    local: ObjectClass,
    segmented_here: bool,
) -> Result<(), InternalErrors> {
    for (path, kind) in class_paths(command, class) {
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

/// Steps `range` of the pipeline - an error if the pipeline or the steps
/// don't exist.
fn selected_steps<'a>(
    project: &'a ProjectSettings,
    pipeline_id: PipelineId,
    range: &RangeInclusive<usize>,
) -> Result<&'a [PipelineStepSettings], InternalErrors> {
    let pipeline = project
        .pipelines
        .iter()
        .find(|p| p.id == pipeline_id)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no pipeline {pipeline_id}")))?;
    pipeline.steps.get(range.clone()).ok_or_else(|| {
        InternalErrors::InvalidArgument(format!(
            "pipeline \"{}\" has no steps {} to {}",
            pipeline.name,
            range.start() + 1,
            range.end() + 1
        ))
    })
}

/// The commands `step` runs (a component's with its values) - none if it can't
/// be expanded.
fn commands_of(project: &ProjectSettings, step: &PipelineStepSettings) -> Vec<PipelineCommand> {
    expand_step(step, &project.components, &mut ScopeCounter::default())
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
    fn walk(value: &serde_json::Value, out: &mut BTreeSet<u32>) {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1
                    && let Some(slot) = map.get("pipelineContext").and_then(|v| v.as_u64())
                {
                    out.insert(slot as u32);
                    return;
                }
                map.values().for_each(|v| walk(v, out));
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    if let Ok(value) = serde_json::to_value(command) {
        walk(&value, &mut out);
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

/// The classes the project uses outside steps `range` of pipeline
/// `pipeline_id`: in other steps and pipelines (inside components too), in hand
/// annotations, in the tile merge settings.
fn classes_used_elsewhere(
    project: &ProjectSettings,
    pipeline_id: PipelineId,
    range: &RangeInclusive<usize>,
) -> BTreeSet<ObjectClass> {
    let mut classes = BTreeSet::new();
    for pipeline in &project.pipelines {
        for (index, step) in pipeline.steps.iter().enumerate() {
            if pipeline.id == pipeline_id && range.contains(&index) {
                continue;
            }
            for command in commands_of(project, step) {
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
    fn expanding_puts_the_steps_back_with_the_groups_classes_as_project_classes() {
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
    fn expanding_turns_group_memory_slots_into_pipeline_slots() {
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

        let key = clone_component_for_step(&mut project, PipelineId(2), 1).unwrap();
        assert_eq!(key, "CELL_RECONSTRUCTION_2");
        assert_eq!(project.components[1].meta.name, "Cell reconstruction (2)");
        assert_eq!(
            component_uses(&project, "CELL_RECONSTRUCTION"),
            1,
            "the other use keeps it"
        );
        assert_eq!(component_uses(&project, &key), 1);
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

        assert!(bound_setting_edited(definition, StepId(4), "min_area"));
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
            !bound_setting_edited(definition, StepId(4), "max_area"),
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
}
