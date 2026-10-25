//! # module_insights
//!
//! What a project's steps and modules do, worked out from their settings:
//! the commands a pipeline step runs with its values applied, the classes a
//! command creates and the channels it reads, what a module parameter
//! sets - for the checks (`module_checks`) and the editor's "Sets …"
//! lines - and the parameter keys that follow section and label.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-16
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::workspace::module_editing::step_setting;
use evanalyzer_cfg::{
    core_types::{ImageAddress, ModuleParamKey, ObjectClass, StepId},
    settings::{
        modules::{ModuleDefinition, ModuleInstance, ModuleParameterDef, ParamValue},
        parameter_def::{ParamType, ParameterDef},
        pipeline_command::PipelineCommand,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
    },
};
use std::collections::BTreeMap;

/// How deep modules are followed - as deep as a run expands them.
const MAX_DEPTH: usize = 8;

/// A command a pipeline step runs, with the step's values applied.
#[derive(Debug, Clone)]
pub struct ResolvedCommand {
    /// The ids of the steps it is, from the pipeline step's module
    /// inwards: empty for a command step, one id for a command of the
    /// module, more for one nested in it.
    pub steps: Vec<StepId>,
    pub command: PipelineCommand,
}

/// The enabled commands `step` runs (none if it's disabled), modules
/// opened with their values - also nested ones. A module whose values
/// don't work is skipped: the checks report that on their own.
pub fn resolved_commands(
    step: &PipelineStepSettings,
    definitions: &[ModuleDefinition],
) -> Vec<ResolvedCommand> {
    fn walk(
        instance: &ModuleInstance,
        definitions: &[ModuleDefinition],
        prefix: &mut Vec<StepId>,
        out: &mut Vec<ResolvedCommand>,
    ) {
        if prefix.len() >= MAX_DEPTH {
            return;
        }
        let Ok(steps) = evanalyzer_core::module_steps_with_values(instance, definitions) else {
            return;
        };
        for step in steps.iter().filter(|s| s.enabled) {
            prefix.push(step.id);
            match &step.content {
                StepContent::Command(command) => out.push(ResolvedCommand {
                    steps: prefix.clone(),
                    command: command.clone(),
                }),
                StepContent::Module(inner) => walk(inner, definitions, prefix, out),
            }
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    if !step.enabled {
        return out;
    }
    match &step.content {
        StepContent::Command(command) => out.push(ResolvedCommand {
            steps: Vec::new(),
            command: command.clone(),
        }),
        StepContent::Module(instance) => walk(instance, definitions, &mut Vec::new(), &mut out),
    }
    out
}

/// The settings of `command` that give the objects it creates their class:
/// (path, class) - also those set to no class. Commands that only change
/// the classes of existing objects aren't creating any.
pub fn created_classes(command: &PipelineCommand) -> Vec<(String, ObjectClass)> {
    // Writing into the input class changes objects in place.
    let new_unless_in_place = |output: ObjectClass, input: ObjectClass| {
        if output == input {
            ObjectClass::Unset
        } else {
            output
        }
    };
    match command {
        PipelineCommand::ExtractObjects(s) => s
            .class_mapping
            .iter()
            .enumerate()
            .map(|(i, m)| (format!("class_mapping.{i}.object_class"), m.object_class))
            .collect(),
        PipelineCommand::Voronoi(s) => vec![("output_class".into(), s.output_class)],
        PipelineCommand::Propagation(s) => vec![("output_class".into(), s.output_class)],
        PipelineCommand::ObjectMath(s) => vec![(
            "output_class".into(),
            new_unless_in_place(s.output_class, s.input_class),
        )],
        PipelineCommand::TransformObjects(s) => vec![(
            "output_class".into(),
            new_unless_in_place(s.output_class, s.input_class),
        )],
        PipelineCommand::LoadAnnotatedObjects(s) => vec![("output_class".into(), s.output_class)],
        PipelineCommand::ColocalizeObjects(s) => vec![(
            "class_for_overlapping_areas".into(),
            s.class_for_overlapping_areas,
        )],
        _ => Vec::new(),
    }
}

/// The classes of objects `command` works on: (path, class) - the classes
/// it selects its input objects by. Not the classes it only filters the
/// result by.
pub fn read_classes(command: &PipelineCommand) -> Vec<(String, ObjectClass)> {
    let list = |path: &str, classes: &[ObjectClass]| -> Vec<(String, ObjectClass)> {
        classes.iter().map(|c| (path.to_string(), *c)).collect()
    };
    let mut read = match command {
        PipelineCommand::ClassifyObjects(s) => {
            let mut read = list("input_classes", &s.input_classes);
            read.push(("overlapping_with".into(), s.overlapping_with));
            read
        }
        PipelineCommand::AiObjectClassifier(s) => list("input_classes", &s.input_classes),
        PipelineCommand::ColocalizeObjects(s) => list("classes_to_coloc", &s.classes_to_coloc),
        PipelineCommand::RelateObjects(s) => {
            let mut read = list("child_classes", &s.child_classes);
            read.extend(list("parent_classes", &s.parent_classes));
            read
        }
        PipelineCommand::ObjectMath(s) => vec![
            ("input_class".into(), s.input_class),
            ("other_class".into(), s.other_class),
        ],
        PipelineCommand::Voronoi(s) => vec![("centers".into(), s.centers), ("mask".into(), s.mask)],
        PipelineCommand::Propagation(s) => vec![("seeds".into(), s.seeds), ("mask".into(), s.mask)],
        PipelineCommand::TransformObjects(s) => vec![("input_class".into(), s.input_class)],
        _ => Vec::new(),
    };
    read.retain(|(_, c)| *c != ObjectClass::Unset);
    read
}

/// The classes `command` gives objects - creating them ([`created_classes`])
/// or classifying existing ones - and whether it may give any class (it
/// loads annotations with the classes they were given).
pub fn assigned_classes(command: &PipelineCommand) -> (Vec<ObjectClass>, bool) {
    let mut assigned: Vec<ObjectClass> = created_classes(command)
        .into_iter()
        .map(|(_, c)| c)
        .collect();
    let mut any = false;
    match command {
        PipelineCommand::ClassifyObjects(s) => assigned.push(s.output_class),
        PipelineCommand::ColocalizeObjects(s) => assigned.push(s.mark_as),
        PipelineCommand::AiObjectClassifier(s) => {
            assigned.extend(s.segmentation_mapping.iter().map(|m| m.output_class))
        }
        PipelineCommand::Script(s) => assigned.extend(s.classes.iter().copied()),
        PipelineCommand::LoadAnnotatedObjects(s) => any = s.keep_annotated_classes,
        // Without a mapping, each segmentation class's objects get the
        // project class with the same number: any class.
        PipelineCommand::ExtractObjects(s) => any = s.class_mapping.is_empty(),
        _ => {}
    }
    assigned.retain(|c| *c != ObjectClass::Unset);
    (assigned, any)
}

/// The image channels `command` reads: (path, channel) - a channel setting,
/// or an image source set to a channel.
pub fn channels_read(command: &PipelineCommand) -> Vec<(String, i32)> {
    fn walk(params: &[ParameterDef], prefix: &str, out: &mut Vec<(String, i32)>) {
        for param in params {
            let path = format!("{prefix}{}", param.name);
            match param.param_type {
                ParamType::ImageChannel => {
                    if let Ok(channel) = param.value.trim().parse() {
                        out.push((path, channel));
                    }
                }
                ParamType::ImageAddress => {
                    if let Some(ImageAddress::Channel(channel)) =
                        ImageAddress::from_param_value(&param.value)
                    {
                        out.push((path, channel));
                    }
                }
                ParamType::Group => {
                    for (index, entry) in param.groups.iter().enumerate() {
                        walk(entry, &format!("{path}.{index}."), out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(&evanalyzer_core::command_parameters(command), "", &mut out);
    out
}

/// The parameters of `definition` that set setting `path` of the command
/// at `steps` (as [`ResolvedCommand::steps`]) - through the modules nested
/// in it too.
pub fn parameters_setting(
    definition: &ModuleDefinition,
    definitions: &[ModuleDefinition],
    steps: &[StepId],
    path: &str,
) -> Vec<ModuleParamKey> {
    let Some((first, rest)) = steps.split_first() else {
        return Vec::new();
    };
    // Inside a module nested at `first`: its parameters that set it.
    let inner: Vec<ModuleParamKey> = if rest.is_empty() {
        Vec::new()
    } else {
        match definition
            .steps
            .iter()
            .find(|s| s.id == *first)
            .map(|s| &s.content)
        {
            Some(StepContent::Module(instance)) => definitions
                .iter()
                .find(|d| d.key == instance.key)
                .map(|d| parameters_setting(d, definitions, rest, path))
                .unwrap_or_default(),
            _ => return Vec::new(),
        }
    };
    definition
        .parameters
        .iter()
        .filter(|p| {
            p.bindings.iter().any(|b| {
                b.step == *first
                    && if rest.is_empty() {
                        b.param == path
                    } else {
                        inner.iter().any(|k| k.as_str() == b.param)
                    }
            })
        })
        .map(|p| p.key.clone())
        .collect()
}

/// Whether `parameter` of `definition` gives the objects a command creates
/// their class (see [`created_classes`]) - also through a module nested in it.
pub fn creates_objects(
    definition: &ModuleDefinition,
    definitions: &[ModuleDefinition],
    parameter: &ModuleParameterDef,
) -> bool {
    fn creates(
        definition: &ModuleDefinition,
        definitions: &[ModuleDefinition],
        parameter: &ModuleParameterDef,
        depth: usize,
    ) -> bool {
        depth < MAX_DEPTH
            && parameter.bindings.iter().any(|binding| {
                match definition
                    .steps
                    .iter()
                    .find(|s| s.id == binding.step)
                    .map(|s| &s.content)
                {
                    Some(StepContent::Command(command)) => created_classes(command)
                        .iter()
                        .any(|(path, _)| *path == binding.param),
                    Some(StepContent::Module(inner)) => definitions
                        .iter()
                        .find(|d| d.key == inner.key)
                        .and_then(|d| {
                            d.parameters
                                .iter()
                                .find(|p| p.key.as_str() == binding.param)
                                .map(|p| creates(d, definitions, p, depth + 1))
                        })
                        .unwrap_or(false),
                    None => false,
                }
            })
    }
    creates(definition, definitions, parameter, 0)
}

/// The classes a parameter value names: a class number ("3", "L2"), or a
/// list of them. Not "no class".
pub fn classes_in(value: &ParamValue) -> Vec<ObjectClass> {
    let classes = match value {
        ParamValue::Int(n) => u32::try_from(*n)
            .map(|n| vec![ObjectClass::Global(n)])
            .unwrap_or_default(),
        ParamValue::Text(text) => text.split(',').filter_map(ObjectClass::from_text).collect(),
        ParamValue::List(items) => items.iter().flat_map(classes_in).collect(),
        ParamValue::Bool(_) | ParamValue::Float(_) => Vec::new(),
    };
    classes
        .into_iter()
        .filter(|c| *c != ObjectClass::Unset)
        .collect()
}

/// What each parameter of module `key` sets, for its card: "Extract
/// Objects › Class Mapping 1 · Object Class (step 7) · Classify Objects ›
/// Overlapping With (steps 12, 13)" - a setting set in several steps once.
pub fn parameter_wiring(
    definitions: &[ModuleDefinition],
    key: &str,
) -> Vec<(ModuleParamKey, String)> {
    let Some(definition) = definitions.iter().find(|d| d.key == key) else {
        return Vec::new();
    };
    definition
        .parameters
        .iter()
        .map(|parameter| {
            // Setting label → the steps it's set in, in binding order.
            let mut targets: Vec<(String, Vec<StepId>)> = Vec::new();
            for binding in &parameter.bindings {
                let Some(step) = definition.steps.iter().find(|s| s.id == binding.step) else {
                    continue;
                };
                let owner = match &step.content {
                    StepContent::Command(command) => command.name().to_string(),
                    StepContent::Module(inner) => definitions
                        .iter()
                        .find(|d| d.key == inner.key)
                        .map_or_else(|| inner.key.clone(), |d| d.meta.name.clone()),
                };
                let setting = step_setting(definitions, step, &binding.param)
                    .map_or_else(|| binding.param.clone(), |(_, label)| label);
                let label = format!("{owner} › {setting}");
                match targets.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, steps)) => steps.push(step.id),
                    None => targets.push((label, vec![step.id])),
                }
            }
            let text = targets
                .into_iter()
                .map(|(label, steps)| {
                    let ids = steps
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    if steps.len() == 1 {
                        format!("{label} (step {ids})")
                    } else {
                        format!("{label} (steps {ids})")
                    }
                })
                .collect::<Vec<_>>()
                .join(" · ");
            (parameter.key.clone(), text)
        })
        .collect()
}

/// For setting `path` of step `step_id` of `definition`: which parameter
/// sets it - "Set by “Output class” (Tetraspeck)", and how many other
/// settings it sets too - or `None`.
pub fn setting_wiring(
    definition: &ModuleDefinition,
    step_id: StepId,
    path: &str,
) -> Option<String> {
    let parameter = definition.parameters.iter().find(|p| {
        p.bindings
            .iter()
            .any(|b| b.step == step_id && b.param == path)
    })?;
    let mut text = format!("Set by “{}”", parameter.label);
    if !parameter.section.is_empty() {
        text.push_str(&format!(" ({})", parameter.section));
    }
    match parameter.bindings.len() - 1 {
        0 => {}
        1 => text.push_str(" - which sets 1 other setting too"),
        n => text.push_str(&format!(" - which sets {n} other settings too")),
    }
    Some(text)
}

/// The channels of a project's images, as far as their metadata says.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageChannels {
    /// The fewest channels an image has.
    pub fewest: usize,
    /// How many images have that few.
    pub images_with_fewest: usize,
    /// How many images the metadata tells the channels of.
    pub images: usize,
    /// The channel names of one with the fewest, in channel order.
    pub names: Vec<String>,
}

impl ImageChannels {
    /// "the images have only channels 0–2 (CY7, CY5, FITC)" - or "3 of 81
    /// images have …" when not all have that few.
    pub fn describe(&self) -> String {
        let which = if self.images_with_fewest == self.images {
            "the images have".to_string()
        } else {
            format!("{} of {} images have", self.images_with_fewest, self.images)
        };
        let names = if self.names.iter().all(|n| n.is_empty()) {
            String::new()
        } else {
            format!(" ({})", self.names.join(", "))
        };
        match self.fewest {
            1 => format!("{which} only channel 0{names}"),
            n => format!("{which} only channels 0–{}{names}", n - 1),
        }
    }
}

/// The channels of `project`'s images - of the series a run reads - or
/// `None` without images whose channels are known.
pub fn image_channels(project: &ProjectSettings) -> Option<ImageChannels> {
    let counts: Vec<(usize, Vec<String>)> = project
        .images
        .list
        .values()
        .filter_map(|entry| {
            let series = project
                .images
                .settings
                .selected_series
                .unwrap_or(entry.selected_series);
            let series = entry.series.get(&series)?;
            Some((
                series.channels.len(),
                series.channels.values().map(|c| c.name.clone()).collect(),
            ))
        })
        .filter(|(count, _)| *count > 0)
        .collect();
    let fewest = counts.iter().map(|(count, _)| *count).min()?;
    Some(ImageChannels {
        fewest,
        images_with_fewest: counts.iter().filter(|(c, _)| *c == fewest).count(),
        images: counts.len(),
        names: counts
            .iter()
            .find(|(c, _)| *c == fewest)
            .map(|(_, names)| names.clone())
            .unwrap_or_default(),
    })
}

/// `text` as a key: lower case, words joined by `_` ("Min Size" → `min_size`).
pub(crate) fn snake_key(text: &str) -> String {
    let mut key = String::new();
    for c in text.trim().chars() {
        if c.is_alphanumeric() {
            key.extend(c.to_lowercase());
        } else if !key.is_empty() && !key.ends_with('_') {
            key.push('_');
        }
    }
    let key = key.trim_end_matches('_');
    match key {
        "" => "parameter".to_string(),
        k if k.starts_with(|c: char| c.is_ascii_digit()) => format!("p_{k}"),
        k => k.to_string(),
    }
}

/// The key a parameter labelled `label` in section `section` has: both in
/// `snake_case` ("Tetraspeck", "Output class" → `tetraspeck_output_class`),
/// so a value in a project file says what it is.
pub fn parameter_key(section: &str, label: &str) -> String {
    snake_key(&format!("{section} {label}"))
}

/// Gives the parameters of module `key` the keys their sections and
/// labels make ([`parameter_key`], `_2` … for a second one alike) - and
/// every use of them too: the values of the module's steps in pipelines
/// and modules, and what modules using it bind. Returns how many
/// parameters got a new key.
pub fn rekey_parameters(project: &mut ProjectSettings, key: &str) -> usize {
    let Some(definition) = project.modules.iter().find(|d| d.key == key) else {
        return 0;
    };
    let mut taken: Vec<String> = Vec::new();
    let mut renames: BTreeMap<ModuleParamKey, ModuleParamKey> = BTreeMap::new();
    for parameter in &definition.parameters {
        let base = parameter_key(&parameter.section, &parameter.label);
        let mut new = base.clone();
        let mut n = 2;
        while taken.contains(&new) {
            new = format!("{base}_{n}");
            n += 1;
        }
        taken.push(new.clone());
        if new != parameter.key.as_str() {
            renames.insert(parameter.key.clone(), ModuleParamKey(new));
        }
    }
    if renames.is_empty() {
        return 0;
    }
    let rename = |old: &ModuleParamKey| renames.get(old).cloned().unwrap_or_else(|| old.clone());

    let renamed_values = |steps: &mut Vec<PipelineStepSettings>| {
        for step in steps {
            if let StepContent::Module(instance) = &mut step.content
                && instance.key == key
            {
                instance.values = std::mem::take(&mut instance.values)
                    .into_iter()
                    .map(|(k, v)| (rename(&k), v))
                    .collect();
            }
        }
    };
    for pipeline in &mut project.pipelines {
        renamed_values(&mut pipeline.steps);
    }
    for definition in &mut project.modules {
        if definition.key == key {
            for parameter in &mut definition.parameters {
                parameter.key = rename(&parameter.key);
            }
        }
        // A module using this one: its bindings name the parameters.
        let uses: Vec<StepId> = definition
            .steps
            .iter()
            .filter(|s| matches!(&s.content, StepContent::Module(i) if i.key == key))
            .map(|s| s.id)
            .collect();
        for parameter in &mut definition.parameters {
            for binding in &mut parameter.bindings {
                if uses.contains(&binding.step) {
                    binding.param = rename(&ModuleParamKey::from(binding.param.as_str())).0;
                }
            }
        }
        renamed_values(&mut definition.steps);
    }
    renames.len()
}

/// The settings that are a size: an area on planes (2D), a volume on
/// volumes (3D).
const SIZE_SETTINGS: [&str; 6] = [
    "min_size",
    "max_size",
    "min_intersection_size",
    "min_overlap_size",
    "min_coloc_size",
    "min_object_size",
];

/// The size settings changed from their default in the enabled pipelines'
/// steps, as "Pipeline › Command: Setting" - an area set for planes is no
/// volume and the other way round, so they need a look when the z-stacks
/// switch between planes (2D) and volumes (3D).
pub fn size_settings_set(project: &ProjectSettings) -> Vec<String> {
    let mut found = Vec::new();
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        for step in &pipeline.steps {
            for resolved in resolved_commands(step, &project.modules) {
                for parameter in resolved.command.to_parameters() {
                    if SIZE_SETTINGS.contains(&parameter.name.as_str())
                        && parameter.value != parameter.default_value
                    {
                        found.push(format!(
                            "{} › {}: {}",
                            pipeline.name.trim(),
                            resolved.command.name(),
                            parameter.display_name
                        ));
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{PipelineId, SegmentationClass};
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use evanalyzer_cfg::settings::modules::ParamBinding;
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        BlurSettings, ExtractClassMappingSettings, ExtractObjectsSettings, LoadChannelSettings,
    };
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    /// The size settings changed from their default, by pipeline and
    /// command - not the ones left alone, not a disabled pipeline's.
    #[test]
    fn the_size_settings_set_are_named_with_their_pipeline_and_command() {
        use evanalyzer_cfg::settings::pipeline_command_settings::ClassifyObjectsSettings;
        let pipeline = |id: u32, name: &str, enabled: bool, min_size: f32| PipelineSettings {
            id: PipelineId(id),
            name: name.into(),
            description: None,
            image_source: ImageAddress::None,
            enabled,
            steps: vec![step(
                1,
                PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                    min_size,
                    ..Default::default()
                }),
            )],
        };
        let mut project = ProjectSettings::default();
        project.pipelines = vec![
            pipeline(1, "Cells", true, 50.0),
            pipeline(2, "Spots", true, 0.0),
            pipeline(3, "Off", false, 50.0),
        ];
        assert_eq!(
            size_settings_set(&project),
            ["Cells › ClassifyObjects: Min Size"]
        );
    }

    fn step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn uses(id: u32, key: &str, values: &[(&str, ParamValue)]) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Module(ModuleInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (ModuleParamKey::from(*k), v.clone()))
                    .collect(),
            }),
        }
    }

    fn parameter(
        key: &str,
        section: &str,
        label: &str,
        default: ParamValue,
        bindings: &[(u32, &str)],
    ) -> ModuleParameterDef {
        ModuleParameterDef {
            key: ModuleParamKey::from(key),
            label: label.into(),
            description: String::new(),
            section: section.into(),
            default,
            bindings: bindings
                .iter()
                .map(|(step, param)| ParamBinding {
                    step: StepId(*step),
                    param: param.to_string(),
                })
                .collect(),
        }
    }

    fn extract() -> PipelineCommand {
        PipelineCommand::ExtractObjects(ExtractObjectsSettings {
            class_mapping: vec![ExtractClassMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class: ObjectClass::Global(1),
            }],
            ..Default::default()
        })
    }

    /// DETECT: loads channel `channel` (parameter "Channel") and extracts
    /// class `class` (parameter "Output class"). TWO: DETECT twice, its
    /// parameters as "Spot" and "Bead" sections.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.modules.push(ModuleDefinition {
            key: "DETECT".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![
                step(
                    1,
                    PipelineCommand::LoadChannel(LoadChannelSettings::default()),
                ),
                step(2, extract()),
            ],
            parameters: vec![
                parameter(
                    "channel",
                    "",
                    "Channel",
                    ParamValue::Int(0),
                    &[(1, "channel")],
                ),
                parameter(
                    "output_class",
                    "",
                    "Output class",
                    ParamValue::Int(1),
                    &[(2, "class_mapping.0.object_class")],
                ),
            ],
            origin: None,
        });
        project.modules.push(ModuleDefinition {
            key: "TWO".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![
                uses(1, "DETECT", &[]),
                uses(2, "DETECT", &[]),
                step(3, PipelineCommand::Blur(BlurSettings::default())),
            ],
            parameters: vec![
                parameter(
                    "a",
                    "Spot",
                    "Output class",
                    ParamValue::Int(2),
                    &[(1, "output_class")],
                ),
                parameter(
                    "b",
                    "Bead",
                    "Output class",
                    ParamValue::Int(3),
                    &[(2, "output_class")],
                ),
                parameter(
                    "c",
                    "Bead",
                    "Channel",
                    ParamValue::Int(1),
                    &[(2, "channel")],
                ),
                parameter("d", "", "Size", ParamValue::Int(3), &[(3, "kernel_size")]),
            ],
            origin: None,
        });
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "p".into(),
            description: None,
            image_source: Default::default(),
            enabled: true,
            steps: vec![uses(1, "TWO", &[("b", ParamValue::Int(5))])],
        });
        project
    }

    #[test]
    fn a_module_step_runs_its_commands_with_its_values_also_nested_ones() {
        let project = project();
        let resolved = resolved_commands(&project.pipelines[0].steps[0], &project.modules);
        let steps: Vec<Vec<u32>> = resolved
            .iter()
            .map(|r| r.steps.iter().map(|s| s.0).collect())
            .collect();
        assert_eq!(
            steps,
            [vec![1, 1], vec![1, 2], vec![2, 1], vec![2, 2], vec![3]]
        );
        let created: Vec<ObjectClass> = resolved
            .iter()
            .flat_map(|r| created_classes(&r.command))
            .map(|(_, class)| class)
            .collect();
        assert_eq!(
            created,
            [ObjectClass::Global(2), ObjectClass::Global(5)],
            "the step's value 5, else the default"
        );
        let channels: Vec<i32> = resolved
            .iter()
            .flat_map(|r| channels_read(&r.command))
            .map(|(_, channel)| channel)
            .collect();
        assert_eq!(channels, [0, 1]);

        let mut disabled = project.pipelines[0].steps[0].clone();
        disabled.enabled = false;
        assert!(resolved_commands(&disabled, &project.modules).is_empty());
    }

    #[test]
    fn the_parameters_setting_a_nested_setting_are_found_through_the_module_in_between() {
        let project = project();
        let two = &project.modules[1];
        let found = |steps: &[u32], path: &str| -> Vec<String> {
            let steps: Vec<StepId> = steps.iter().map(|s| StepId(*s)).collect();
            parameters_setting(two, &project.modules, &steps, path)
                .into_iter()
                .map(|k| k.0)
                .collect()
        };
        assert_eq!(found(&[2, 2], "class_mapping.0.object_class"), ["b"]);
        assert_eq!(found(&[2, 1], "channel"), ["c"]);
        assert_eq!(
            found(&[1, 1], "channel"),
            Vec::<String>::new(),
            "not exposed"
        );
        assert_eq!(found(&[3], "kernel_size"), ["d"]);

        let creating: Vec<&str> = two
            .parameters
            .iter()
            .filter(|p| creates_objects(two, &project.modules, p))
            .map(|p| p.key.as_str())
            .collect();
        assert_eq!(creating, ["a", "b"]);
    }

    #[test]
    fn a_parameters_wiring_names_each_setting_once_with_its_steps() {
        let mut project = project();
        project.modules[0].meta.name = "Detect".into();
        project.modules[1].parameters[0]
            .bindings
            .push(ParamBinding {
                step: StepId(2),
                param: "output_class".into(),
            });
        let wiring = parameter_wiring(&project.modules, "TWO");
        assert_eq!(wiring[0].1, "Detect › Output class (steps 1, 2)");
        assert_eq!(wiring[3].1, "Blur › Kernel size (step 3)");
        assert_eq!(
            setting_wiring(&project.modules[1], StepId(2), "output_class").as_deref(),
            Some("Set by “Output class” (Spot) - which sets 1 other setting too")
        );
        assert_eq!(
            setting_wiring(&project.modules[1], StepId(3), "kernel_size").as_deref(),
            Some("Set by “Size”")
        );
        assert_eq!(
            setting_wiring(&project.modules[1], StepId(3), "nothing"),
            None
        );
    }

    #[test]
    fn class_values_are_numbers_or_lists_of_them_never_no_class() {
        assert_eq!(classes_in(&ParamValue::Int(3)), [ObjectClass::Global(3)]);
        assert_eq!(
            classes_in(&ParamValue::Text("1, L2,-1".into())),
            [ObjectClass::Global(1), ObjectClass::Local(2)]
        );
        assert_eq!(classes_in(&ParamValue::Int(-1)), []);
        assert_eq!(
            classes_in(&ParamValue::List(vec![ParamValue::Int(4)])),
            [ObjectClass::Global(4)]
        );
    }

    #[test]
    fn the_image_channels_are_those_of_the_image_with_the_fewest() {
        let mut project = project();
        assert_eq!(image_channels(&project), None);
        let image = |names: &[&str]| {
            let mut series = SeriesSettings::default();
            for (i, name) in names.iter().enumerate() {
                series.channels.insert(
                    i as i32,
                    evanalyzer_cfg::settings::images_settings::ChannelSettings {
                        name: name.to_string(),
                        ..Default::default()
                    },
                );
            }
            ImageEntry {
                series: [(0, series)].into(),
                ..Default::default()
            }
        };
        project
            .images
            .list
            .insert("a".into(), image(&["CY7", "CY5", "FITC"]));
        let all = image_channels(&project).unwrap();
        assert_eq!(all.fewest, 3);
        assert_eq!(
            all.describe(),
            "the images have only channels 0–2 (CY7, CY5, FITC)"
        );
        project.images.list.insert("b".into(), image(&["DAPI"]));
        project
            .images
            .list
            .insert("c".into(), image(&["A", "B", "C", "D"]));
        assert_eq!(
            image_channels(&project).unwrap().describe(),
            "1 of 3 images have only channel 0 (DAPI)"
        );
    }

    #[test]
    fn keys_follow_section_and_label_and_every_use_follows_them() {
        let mut project = project();
        assert_eq!(
            parameter_key("Tetraspeck", "Output class"),
            "tetraspeck_output_class"
        );
        assert_eq!(parameter_key("", "Min Size"), "min_size");

        // DETECT's keys are already its labels'.
        assert_eq!(rekey_parameters(&mut project, "DETECT"), 0);
        project.modules[0].parameters[1].label = "Class".into();
        assert_eq!(rekey_parameters(&mut project, "DETECT"), 1);
        assert_eq!(project.modules[0].parameters[1].key.as_str(), "class");
        // TWO binds DETECT's parameter: by its new key now - and still works.
        let bound: Vec<&str> = project.modules[1]
            .parameters
            .iter()
            .flat_map(|p| p.bindings.iter().map(|b| b.param.as_str()))
            .collect();
        assert_eq!(bound, ["class", "class", "channel", "kernel_size"]);

        assert_eq!(rekey_parameters(&mut project, "TWO"), 4);
        let keys: Vec<&str> = project.modules[1]
            .parameters
            .iter()
            .map(|p| p.key.as_str())
            .collect();
        assert_eq!(
            keys,
            [
                "spot_output_class",
                "bead_output_class",
                "bead_channel",
                "size"
            ]
        );
        let StepContent::Module(instance) = &project.pipelines[0].steps[0].content else {
            panic!("a module step");
        };
        assert_eq!(
            instance
                .values
                .keys()
                .map(|k| k.as_str())
                .collect::<Vec<_>>(),
            ["bead_output_class"],
            "the pipeline's value moved along"
        );
        let resolved = resolved_commands(&project.pipelines[0].steps[0], &project.modules);
        let created: Vec<ObjectClass> = resolved
            .iter()
            .flat_map(|r| created_classes(&r.command))
            .map(|(_, class)| class)
            .collect();
        assert_eq!(created, [ObjectClass::Global(2), ObjectClass::Global(5)]);

        // Two alike: the second gets `_2`.
        project.modules[1].parameters[1].section = "Spot".into();
        rekey_parameters(&mut project, "TWO");
        assert_eq!(
            project.modules[1].parameters[1].key.as_str(),
            "spot_output_class_2"
        );
    }
}
