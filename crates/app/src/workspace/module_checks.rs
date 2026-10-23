//! Checking a project's modules - on load and in the editor, so
//! mistakes show where they are instead of only when a run starts.
//!
//! Most checks are the expansion itself (`evanalyzer_core::expand_step`):
//! what a run would refuse, reported per module and per step using it. The
//! rest are things expansion can't see: duplicate keys, parameters binding
//! settings of different kinds, and warnings for what works but probably
//! isn't meant. [`check_project`] adds what only the whole project shows: a
//! step's commands with its values against the images' channels, the
//! classification and the other steps.

use crate::workspace::extensions::classification_ext::ClassificationExt;
use crate::workspace::module_editing::{remove_binding, same_kind, setting_at};
use crate::workspace::module_insights::{
    ResolvedCommand, assigned_classes, channels_read, classes_in, created_classes, creates_objects,
    image_channels, parameters_setting, read_classes, resolved_commands,
};
use evanalyzer_cfg::{
    core_types::{
        ImageAddress, ModuleParamKey, ObjectClass, PipelineId, StepId, is_valid_measurement_id,
    },
    settings::{
        classification_settings::Class,
        images_settings::{ZStackHandling, ZStackSettings},
        modules::{ModuleDefinition, ModuleInstance},
        pipeline_command::PipelineCommand,
        pipeline_command_settings::MathLoadChannelChannelZProjectionSettings,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
    },
};
use evanalyzer_core::{ScopeCounter, expand_step};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A run refuses it.
    Error,
    /// Works, but probably isn't meant.
    Warning,
}

/// One problem found by [`check_modules`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleProblem {
    pub severity: Severity,
    /// The module it's about, if any.
    pub module: Option<String>,
    /// The pipeline step it's about, if any: (pipeline, step id) - the step
    /// id [`StepId::UNASSIGNED`] for the pipeline itself (its image source).
    pub step: Option<(PipelineId, StepId)>,
    pub message: String,
    /// The module parameters it's about: of the step's module for a
    /// problem at a step, else of `module`.
    pub parameters: Vec<ModuleParamKey>,
    /// What fixes it with one click, if anything does.
    pub fix: Option<ProblemFix>,
    /// Where inside the step's module it is ("LoadChannel (its step
    /// 9)"), for those who look into modules (debug mode); empty if
    /// the message says it all.
    pub detail: String,
}

/// A fix [`apply_fix`] makes.
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemFix {
    /// Adds project class `id` to the classification.
    AddClass(u32),
    /// Parameter `parameter` of module `module` stops setting `path`
    /// of step `step` (labelled `label`).
    RemoveBinding {
        module: String,
        parameter: ModuleParamKey,
        step: StepId,
        path: String,
        label: String,
    },
}

impl ProblemFix {
    /// What the fix's button says.
    pub fn label(&self) -> String {
        match self {
            ProblemFix::AddClass(id) => format!("Add class {id}"),
            ProblemFix::RemoveBinding { label, .. } => format!("Stop setting {label}"),
        }
    }
}

/// Makes `fix` in `project`. Returns whether something changed.
pub fn apply_fix(project: &mut ProjectSettings, fix: &ProblemFix) -> bool {
    match fix {
        ProblemFix::AddClass(id) => project
            .classification
            .add_class_with_id(Class {
                id: *id,
                color: class_color(*id),
                name: format!("Class {id}"),
                notes: String::new(),
            })
            .is_ok(),
        ProblemFix::RemoveBinding {
            module,
            parameter,
            step,
            path,
            ..
        } => project
            .modules
            .iter_mut()
            .find(|d| &d.key == module)
            .is_some_and(|d| remove_binding(d, parameter, *step, path)),
    }
}

/// A color for a class added by a fix - one of a few distinct ones.
fn class_color(id: u32) -> u32 {
    const COLORS: [u32; 8] = [
        0xe6194b, 0x3cb44b, 0xffe119, 0x4363d8, 0xf58231, 0x911eb4, 0x46f0f0, 0xf032e6,
    ];
    COLORS[id as usize % COLORS.len()]
}

impl ModuleProblem {
    fn error(module: Option<&str>, step: Option<(PipelineId, StepId)>, message: String) -> Self {
        Self {
            severity: Severity::Error,
            module: module.map(str::to_string),
            step,
            message,
            parameters: Vec::new(),
            fix: None,
            detail: String::new(),
        }
    }

    fn inside(mut self, detail: String) -> Self {
        self.detail = detail;
        self
    }

    fn about(mut self, parameters: Vec<ModuleParamKey>) -> Self {
        self.parameters = parameters;
        self
    }

    fn fixed_by(mut self, fix: ProblemFix) -> Self {
        self.fix = Some(fix);
        self
    }

    fn warning(module: Option<&str>, step: Option<(PipelineId, StepId)>, message: String) -> Self {
        Self {
            severity: Severity::Warning,
            ..Self::error(module, step, message)
        }
    }
}

/// Every problem of the project's modules and the steps using them.
pub fn check_modules(project: &ProjectSettings) -> Vec<ModuleProblem> {
    let mut problems = Vec::new();

    let mut seen: Vec<&str> = Vec::new();
    for definition in &project.modules {
        if seen.contains(&definition.key.as_str()) {
            problems.push(ModuleProblem::error(
                Some(&definition.key),
                None,
                format!(
                    "Two modules have the key {} - steps can't tell which they use",
                    definition.key
                ),
            ));
        }
        seen.push(&definition.key);
    }

    let mut broken: Vec<&str> = Vec::new();
    for definition in &project.modules {
        let found = check_definition(project, definition);
        if found.iter().any(|p| p.severity == Severity::Error) {
            broken.push(&definition.key);
        }
        problems.extend(found);
    }

    for pipeline in &project.pipelines {
        for step in &pipeline.steps {
            let at = Some((pipeline.id, step.id));
            match &step.content {
                StepContent::Module(instance) => {
                    if !project.modules.iter().any(|d| d.key == instance.key) {
                        problems.push(ModuleProblem::error(
                            Some(&instance.key),
                            at,
                            format!("Module {} isn't defined in this project", instance.key),
                        ));
                    } else if !broken.contains(&instance.key.as_str()) {
                        // The module itself is fine: what's wrong are this
                        // step's values.
                        if let Err(e) =
                            expand_step(step, &project.modules, &mut ScopeCounter::default())
                        {
                            problems.push(ModuleProblem::error(
                                Some(&instance.key),
                                at,
                                e.to_string(),
                            ));
                        }
                    }
                }
                StepContent::Command(command) => {
                    if let Err(e) =
                        expand_step(step, &project.modules, &mut ScopeCounter::default())
                    {
                        problems.push(ModuleProblem::error(None, at, e.to_string()));
                    }
                    problems.extend(missing_created_classes(project, command).into_iter().map(
                        |(message, id, _)| {
                            ModuleProblem::warning(None, at, message)
                                .fixed_by(ProblemFix::AddClass(id))
                        },
                    ));
                }
            }
        }
    }
    problems
}

/// The problems of module `key` alone (the module, not the steps using it).
pub fn check_module(project: &ProjectSettings, key: &str) -> Vec<ModuleProblem> {
    project
        .modules
        .iter()
        .find(|d| d.key == key)
        .map(|definition| check_definition(project, definition))
        .unwrap_or_default()
}

/// What's wrong with pipeline step `step` - a module step whose module or
/// values don't work, a command a run would refuse - or `None`.
pub fn step_problem(project: &ProjectSettings, step: &PipelineStepSettings) -> Option<String> {
    if let StepContent::Module(instance) = &step.content
        && let Some(problem) = check_module(project, &instance.key)
            .into_iter()
            .find(|p| p.severity == Severity::Error)
    {
        return Some(problem.message);
    }
    expand_step(step, &project.modules, &mut ScopeCounter::default())
        .err()
        .map(|e| e.to_string())
}

/// Every problem of the project: [`check_modules`]'s, and what the
/// steps of the enabled pipelines do with their values - the classes they create (of the
/// classification, twice, by two parameters of one module step), the
/// channels they read (of the images), an overlap with the class itself.
pub fn check_project(project: &ProjectSettings) -> Vec<ModuleProblem> {
    let mut problems = check_modules(project);
    problems.extend(check_steps(project));
    problems.extend(check_measurement_ids(project));
    problems
}

/// The measurement steps' IDs in what runs: each valid, none twice (a
/// name starts its measurements' keys). A module used twice names its
/// measurement the same both times unless the name is its parameter.
fn check_measurement_ids(project: &ProjectSettings) -> Vec<ModuleProblem> {
    let mut problems = Vec::new();
    // (name, its step, where it is as a message says it, the module's key)
    let mut uses: Vec<(String, (PipelineId, StepId), String, Option<String>)> = Vec::new();
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        for (position, step) in pipeline.steps.iter().enumerate() {
            let module = match &step.content {
                StepContent::Module(instance) => Some(instance.key.clone()),
                StepContent::Command(_) => None,
            };
            for command in resolved_commands(step, &project.modules) {
                let Some(name) = command.command.measurement_id() else {
                    continue;
                };
                let at = (pipeline.id, step.id);
                if !is_valid_measurement_id(name) {
                    problems.push(ModuleProblem::error(
                        None,
                        Some(at),
                        format!(
                            "The measurement ID “{name}” may only have letters, digits and _ - \
                             it names the measurements' columns"
                        ),
                    ));
                }
                let place = format!("step {} of “{}”", position + 1, pipeline.name);
                uses.push((name.to_string(), at, place, module.clone()));
            }
        }
    }
    for group in same_names(&uses, |(name, ..)| name.as_str()) {
        for &i in &group {
            let (name, at, _, module) = &uses[i];
            let others: Vec<&str> = group
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| uses[j].2.as_str())
                .collect();
            let message = match module {
                Some(key)
                    if group
                        .iter()
                        .any(|&j| j != i && uses[j].3.as_ref() == Some(key)) =>
                {
                    format!(
                        "The module “{key}” is used more than once and names its measurement \
                         “{name}” each time - make the name a module parameter, so each use \
                         gets its own"
                    )
                }
                _ => format!(
                    "The measurement ID “{name}” is also used by {} - each measurement \
                     step needs its own",
                    others.join(", ")
                ),
            };
            problems.push(ModuleProblem::error(None, Some(*at), message));
        }
    }
    problems
}

/// The groups (indexes into `items`) whose names are the same, case aside -
/// only names used more than once.
fn same_names<T>(items: &[T], name: impl Fn(&T) -> &str) -> Vec<Vec<usize>> {
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (i, item) in items.iter().enumerate() {
        groups.entry(name(item).to_lowercase()).or_default().push(i);
    }
    groups.into_values().filter(|g| g.len() > 1).collect()
}

/// The problems at pipeline step `step_id` of pipeline `pipeline` among
/// `problems` (as [`check_project`] finds them).
pub fn problems_at(
    problems: &[ModuleProblem],
    pipeline: PipelineId,
    step_id: StepId,
) -> Vec<&ModuleProblem> {
    problems
        .iter()
        .filter(|p| p.step == Some((pipeline, step_id)))
        .collect()
}

/// A class as a message names it: `3 “Nuclei”`, or `3` if the project
/// doesn't have it.
fn class_label(project: &ProjectSettings, class: ObjectClass) -> String {
    match class {
        ObjectClass::Global(n) => project
            .classification
            .classes()
            .iter()
            .find(|c| c.id == n)
            .filter(|c| !c.name.is_empty())
            .map_or_else(|| n.to_string(), |c| format!("{n} “{}”", c.name)),
        other => other.to_text(),
    }
}

/// A module parameter as a message names it: “Output class” (Tetraspeck).
fn parameter_label(definition: &ModuleDefinition, key: &ModuleParamKey) -> String {
    definition
        .parameters
        .iter()
        .find(|p| &p.key == key)
        .map_or_else(
            || key.to_string(),
            |p| match p.section.as_str() {
                "" => format!("“{}”", p.label),
                section => format!("“{}” ({section})", p.label),
            },
        )
}

/// Whether a run analyzes several z-planes one by one (the project's or an
/// image's z-stack setting), not one projection or one plane.
fn analyzes_planes_one_by_one(project: &ProjectSettings) -> bool {
    let one_by_one = |z: &ZStackSettings| match z.z_projection {
        ZStackHandling::AllStacks => true,
        ZStackHandling::SingleStack => z.z_range.as_ref().is_some_and(|r| r.end() > r.start()),
        _ => false,
    };
    project
        .images
        .settings
        .z_stack
        .as_ref()
        .is_some_and(one_by_one)
        || project
            .images
            .list
            .values()
            .flat_map(|entry| entry.series.values())
            .any(|series| series.z_stack.as_ref().is_some_and(one_by_one))
}

/// Names joined for a message: "A", "A and B", "A, B and C".
fn joined(names: Vec<String>) -> String {
    match names.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// Who a message about a command of a pipeline step says does something:
/// the parameters of the step's module setting it (“Channel” (Bead)),
/// else the command (a command step) or the module ("“Spots” … inside").
struct Subject {
    /// The parameters, named - empty if none sets it.
    parameters: String,
    /// The command's or module's name.
    name: String,
    /// The command is inside a module.
    inside: bool,
}

impl Subject {
    /// "{the parameters} is {value}" - or "{name} {verb}[ inside]".
    fn says(&self, is: &str, verb: &str) -> String {
        if !self.parameters.is_empty() {
            let be = if self.parameters.contains(" and ") {
                "are"
            } else {
                "is"
            };
            format!("{} {be} {is}", self.parameters)
        } else if self.inside {
            format!("“{}” {verb} inside", self.name)
        } else {
            format!("{} {verb}", self.name)
        }
    }
}

/// What [`check_project`] adds to [`check_modules`] - for the enabled
/// pipelines, the ones a run runs.
fn check_steps(project: &ProjectSettings) -> Vec<ModuleProblem> {
    let mut problems = Vec::new();
    let channels = image_channels(project);
    let planes_one_by_one = analyzes_planes_one_by_one(project);
    // Each project class created: by which step (pipeline, step), a label
    // for it, and the step's parameters giving it.
    let mut creators: Vec<(u32, (PipelineId, StepId), String, String)> = Vec::new();
    // The project classes some step of an enabled pipeline gives objects -
    // `None` if one may give any (annotations loaded with their classes).
    let mut assigned: Option<Vec<u32>> = Some(Vec::new());
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        for step in &pipeline.steps {
            for command in resolved_commands(step, &project.modules) {
                let (classes, any) = assigned_classes(&command.command);
                if any {
                    assigned = None;
                }
                if let Some(assigned) = &mut assigned {
                    assigned.extend(classes.into_iter().filter_map(|c| match c {
                        ObjectClass::Global(n) => Some(n),
                        _ => None,
                    }));
                }
            }
        }
    }

    // A disabled pipeline doesn't run: nothing it does can go wrong.
    for pipeline in project.pipelines.iter().filter(|p| p.enabled) {
        if let (Some(channels), ImageAddress::Channel(channel)) = (&channels, pipeline.image_source)
            && channel as usize >= channels.fewest
        {
            problems.push(ModuleProblem::error(
                None,
                Some((pipeline.id, StepId::UNASSIGNED)),
                format!(
                    "The pipeline “{}” itself starts from channel {channel} (its image source - \
                     ✎ edits it), but {}",
                    pipeline.name,
                    channels.describe()
                ),
            ));
        }

        // This pipeline's classes worked on that no step gives any object.
        struct Unmade {
            class: u32,
            /// The steps working on it, in order - the warning is at the first.
            steps: Vec<Option<(PipelineId, StepId)>>,
            /// "ClassifyObjects works on class 15" - as the first step says it.
            says: String,
            keys: Vec<ModuleParamKey>,
            detail: String,
        }
        let mut unmade: Vec<Unmade> = Vec::new();

        for step in &pipeline.steps {
            let at = Some((pipeline.id, step.id));
            let instance = match &step.content {
                StepContent::Module(instance) => Some(instance),
                StepContent::Command(_) => None,
            };
            let definition = instance.and_then(|i| project.modules.iter().find(|d| d.key == i.key));
            let resolved = resolved_commands(step, &project.modules);
            // The step's parameters setting `path` of `command`.
            let setting = |command: &ResolvedCommand, path: &str| match definition {
                Some(definition) => {
                    parameters_setting(definition, &project.modules, &command.steps, path)
                }
                None => Vec::new(),
            };
            let subject = |command: &ResolvedCommand, keys: &[ModuleParamKey]| Subject {
                parameters: match definition {
                    Some(d) => joined(keys.iter().map(|k| parameter_label(d, k)).collect()),
                    None => String::new(),
                },
                name: definition.map_or_else(
                    || command.command.name().to_string(),
                    |d| d.meta.name.clone(),
                ),
                inside: definition.is_some(),
            };
            // Where inside the module: "ClassifyObjects (its step 14)",
            // "(its step 8 › 7)" nested. Empty for a command step.
            let detail = |command: &ResolvedCommand| match command.steps.as_slice() {
                [] => String::new(),
                steps => format!(
                    "{} (its step {})",
                    command.command.name(),
                    steps
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(" › ")
                ),
            };
            // "the pipeline “Coloc” at #8 (Colocalization)" - the step's id
            // as its card shows it, so it isn't read as a step inside a
            // module.
            let this_step = format!(
                "the pipeline “{}” at #{} ({})",
                pipeline.name.trim(),
                step.id,
                match definition {
                    Some(d) => format!("“{}”", d.meta.name),
                    None => step.command().map_or("", |c| c.name()).to_string(),
                }
            );

            for command in &resolved {
                if let Some(channels) = &channels {
                    for (path, channel) in channels_read(&command.command) {
                        if channel as usize >= channels.fewest {
                            let keys = setting(command, &path);
                            let who = subject(command, &keys);
                            problems.push(
                                ModuleProblem::error(
                                    None,
                                    at,
                                    format!(
                                        "{}, but {}",
                                        who.says(
                                            &format!("channel {channel}"),
                                            &format!("reads channel {channel}")
                                        ),
                                        channels.describe()
                                    ),
                                )
                                .about(keys)
                                .inside(detail(command)),
                            );
                        }
                    }
                }

                // Its own projection while each z-plane is analyzed: the
                // pipeline runs per plane on the same image - repeated objects.
                if planes_one_by_one
                    && let PipelineCommand::LoadChannel(load) = &command.command
                    && load.projection != MathLoadChannelChannelZProjectionSettings::Auto
                {
                    let keys = setting(command, "projection");
                    let who = subject(command, &keys);
                    problems.push(
                        ModuleProblem::warning(
                            None,
                            at,
                            format!(
                                "{}, but the project analyzes the z-planes one by one: the \
                                 pipeline then runs once per plane on the same image, so its \
                                 objects repeat for every plane - use either one-by-one \
                                 z-planes or a projection, not both",
                                who.says(
                                    "set to a z-projection of its own",
                                    "loads its channel with a z-projection of its own"
                                )
                            ),
                        )
                        .about(keys)
                        .inside(detail(command)),
                    );
                }

                if let PipelineCommand::ClassifyObjects(classify) = &command.command
                    && classify.overlapping_with != ObjectClass::Unset
                    && classify.input_classes.contains(&classify.overlapping_with)
                {
                    let mut keys = setting(command, "overlapping_with");
                    keys.extend(setting(command, "input_classes"));
                    keys.dedup();
                    let class = class_label(project, classify.overlapping_with);
                    let who = subject(command, &keys);
                    let message = if who.parameters.is_empty() {
                        format!(
                            "{} for class {class} objects overlapping other class {class} \
                             objects - “Overlapping with” is usually another class",
                            who.says("", "looks")
                        )
                    } else {
                        format!(
                            "{} - a step then looks for class {class} objects overlapping \
                             other class {class} objects; usually another class is meant",
                            who.says(&format!("class {class}"), "")
                        )
                    };
                    problems.push(
                        ModuleProblem::warning(None, at, message)
                            .about(keys)
                            .inside(detail(command)),
                    );
                }

                // Classes worked on that no step gives any object - one
                // warning per class and pipeline, at the first step.
                if let Some(assigned) = &assigned {
                    for (path, class) in read_classes(&command.command) {
                        let ObjectClass::Global(n) = class else {
                            continue;
                        };
                        if assigned.contains(&n) {
                            continue;
                        }
                        match unmade.iter_mut().find(|u| u.class == n) {
                            Some(first) => {
                                if !first.steps.contains(&at) {
                                    first.steps.push(at);
                                }
                            }
                            None => {
                                let keys = setting(command, &path);
                                unmade.push(Unmade {
                                    class: n,
                                    steps: vec![at],
                                    says: subject(command, &keys).says(
                                        &format!(
                                            "class {}",
                                            class_label(project, ObjectClass::Global(n))
                                        ),
                                        &format!(
                                            "works on class {}",
                                            class_label(project, ObjectClass::Global(n))
                                        ),
                                    ),
                                    keys,
                                    detail: detail(command),
                                })
                            }
                        }
                    }
                }

                for (path, class) in created_classes(&command.command) {
                    let ObjectClass::Global(n) = class else {
                        continue;
                    };
                    let keys = setting(command, &path);
                    let who = subject(command, &keys);
                    match creators
                        .iter_mut()
                        .find(|(c, s, _, _)| *c == n && *s == (pipeline.id, step.id))
                    {
                        Some((_, _, _, named)) => {
                            if named.is_empty() {
                                named.clone_from(&who.parameters);
                            }
                        }
                        None => creators.push((
                            n,
                            (pipeline.id, step.id),
                            match path.as_str() {
                                "class_for_overlapping_areas" => {
                                    format!(
                                        "{}, for its overlap areas)",
                                        this_step.strip_suffix(')').unwrap_or(&this_step)
                                    )
                                }
                                _ => this_step.clone(),
                            },
                            who.parameters.clone(),
                        )),
                    }
                    // A command step's are reported by `check_modules`.
                    if definition.is_some()
                        && !project.classification.classes().iter().any(|c| c.id == n)
                        && !problems.iter().any(|p: &ModuleProblem| {
                            p.step == at && p.fix == Some(ProblemFix::AddClass(n))
                        })
                    {
                        problems.push(
                            ModuleProblem::warning(
                                None,
                                at,
                                format!(
                                    "{}, which the project doesn't have",
                                    who.says(
                                        &format!("class {n}"),
                                        &format!("creates objects of class {n}")
                                    )
                                ),
                            )
                            .about(keys)
                            .inside(detail(command))
                            .fixed_by(ProblemFix::AddClass(n)),
                        );
                    }
                }
            }

            // Two parameters of one module step creating the same class.
            if let (Some(instance), Some(definition)) = (instance, definition) {
                problems.extend(colliding_parameters(project, definition, instance, at));
            }
        }

        for u in unmade {
            let also = match u.steps.len() - 1 {
                0 => String::new(),
                1 => " - 1 more step of this pipeline works on it too".to_string(),
                n => format!(" - {n} more steps of this pipeline work on it too"),
            };
            problems.push(
                ModuleProblem::warning(
                    None,
                    u.steps[0],
                    format!(
                        "{}, but no step gives any object that class - there's nothing to \
                         work on{also}",
                        u.says
                    ),
                )
                .about(u.keys)
                .inside(u.detail),
            );
        }
    }

    // The same class created by different steps.
    for (n, at, _, named) in &creators {
        let others: Vec<String> = creators
            .iter()
            .filter(|(m, other, _, _)| m == n && other != at)
            .map(|(_, _, label, _)| label.clone())
            .collect();
        if others.is_empty() {
            continue;
        }
        let class = class_label(project, ObjectClass::Global(*n));
        let message = if named.is_empty() {
            format!(
                "Class {class} is also given to new objects by {} - objects of both end up in \
                 one class and can't be told apart",
                joined(others)
            )
        } else {
            format!(
                "{named} is class {class} - but {} gives new objects that class too; objects \
                 of both end up in one class and can't be told apart",
                joined(others)
            )
        };
        problems.push(ModuleProblem::warning(None, Some(*at), message));
    }
    problems
}

/// Warnings for the parameters of `instance` (a step at `at`) that create
/// objects of the same class - their objects can't be told apart after.
fn colliding_parameters(
    project: &ProjectSettings,
    definition: &ModuleDefinition,
    instance: &ModuleInstance,
    at: Option<(PipelineId, StepId)>,
) -> Vec<ModuleProblem> {
    // Class → the parameters creating it, in parameter order.
    let mut by_class: Vec<(ObjectClass, Vec<ModuleParamKey>)> = Vec::new();
    for parameter in &definition.parameters {
        if !creates_objects(definition, &project.modules, parameter) {
            continue;
        }
        let value = instance
            .values
            .get(&parameter.key)
            .unwrap_or(&parameter.default);
        for class in classes_in(value) {
            match by_class.iter_mut().find(|(c, _)| *c == class) {
                Some((_, keys)) => keys.push(parameter.key.clone()),
                None => by_class.push((class, vec![parameter.key.clone()])),
            }
        }
    }
    by_class
        .into_iter()
        .filter(|(_, keys)| keys.len() > 1)
        .map(|(class, keys)| {
            let names: Vec<String> = keys
                .iter()
                .map(|k| parameter_label(definition, k))
                .collect();
            let names = joined(names);
            ModuleProblem::warning(
                Some(&definition.key),
                at,
                format!(
                    "{names} both create objects of class {} - their objects can't be told \
                     apart after; give each its own class",
                    class_label(project, class)
                ),
            )
            .about(keys)
        })
        .collect()
}

fn check_definition(
    project: &ProjectSettings,
    definition: &ModuleDefinition,
) -> Vec<ModuleProblem> {
    let key = Some(definition.key.as_str());
    let mut problems = Vec::new();
    let error = |message: String| ModuleProblem::error(key, None, message);
    let named = |message: String| format!("Module {}: {message}", definition.key);

    let mut ids: Vec<StepId> = Vec::new();
    for step in &definition.steps {
        if ids.contains(&step.id) {
            problems.push(error(named(format!(
                "two steps have the id {} - its parameters can't tell which they set",
                step.id
            ))));
        }
        ids.push(step.id);
    }

    for parameter in &definition.parameters {
        if parameter.bindings.is_empty() {
            problems.push(
                ModuleProblem::warning(
                    key,
                    None,
                    named(format!("parameter \"{}\" sets nothing", parameter.label)),
                )
                .about(vec![parameter.key.clone()]),
            );
            continue;
        }
        // What kind of setting each binding sets - all the same, or the
        // parameter's one value can't fit them all.
        let mut kinds = Vec::new();
        for binding in &parameter.bindings {
            let Some(step) = definition.steps.iter().find(|s| s.id == binding.step) else {
                // Expansion reports a missing step.
                continue;
            };
            if let Some(command) = step.command() {
                match setting_at(command, &binding.param) {
                    Some((setting, label)) => kinds.push((setting, label, binding)),
                    None => problems.push(error(named(format!(
                        "parameter \"{}\" is bound to {} of step {}, which {} doesn't have",
                        parameter.label,
                        binding.param,
                        binding.step,
                        command.name()
                    )))),
                }
            }
        }
        if let Some((first_setting, first, _)) = kinds.first()
            && let Some((_, other, binding)) = kinds
                .iter()
                .find(|(setting, _, _)| !same_kind(first_setting, setting))
        {
            problems.push(
                error(named(format!(
                    "parameter \"{}\" sets {first} and {other}, which are different kinds of \
                     setting",
                    parameter.label
                )))
                .about(vec![parameter.key.clone()])
                .fixed_by(ProblemFix::RemoveBinding {
                    module: definition.key.clone(),
                    parameter: parameter.key.clone(),
                    step: binding.step,
                    path: binding.param.clone(),
                    label: format!("{other} (step {})", binding.step),
                }),
            );
        }
    }

    // The module at its defaults: what a run would refuse.
    let at_defaults = PipelineStepSettings {
        id: StepId::UNASSIGNED,
        enabled: true,
        content: StepContent::Module(ModuleInstance {
            key: definition.key.clone(),
            values: Default::default(),
        }),
    };
    if let Err(e) = expand_step(&at_defaults, &project.modules, &mut ScopeCounter::default()) {
        problems.push(error(e.to_string()));
    }

    for step in &definition.steps {
        let Some(command) = step.command() else {
            continue;
        };
        if let PipelineCommand::Colocalization(coloc) = command
            && let Some(local) = std::iter::once(&coloc.objects)
                .chain(&coloc.overlapping)
                .find(|c| matches!(c, ObjectClass::Local(_)))
        {
            problems.push(ModuleProblem::warning(
                key,
                None,
                named(format!(
                    "step {} colocalizes with the module class {} - its links are dropped when \
                     the module is done, so the results have no colocalization counts for it",
                    step.id,
                    local.to_text()
                )),
            ));
        }
        if let PipelineCommand::RelateObjects(relate) = command
            && let Some(local) = relate
                .parent_classes
                .iter()
                .find(|c| matches!(c, ObjectClass::Local(_)))
        {
            problems.push(ModuleProblem::warning(
                key,
                None,
                named(format!(
                    "step {} links to parents of the module class {} - those links are dropped \
                     when the module is done, so the results have no children per parent for it",
                    step.id,
                    local.to_text()
                )),
            ));
        }
        problems.extend(missing_created_classes(project, command).into_iter().map(
            |(message, id, path)| {
                ModuleProblem::warning(key, None, named(message))
                    .about(parameters_setting(
                        definition,
                        &project.modules,
                        &[step.id],
                        &path,
                    ))
                    .fixed_by(ProblemFix::AddClass(id))
            },
        ));
    }
    problems
}

/// The project classes `command` creates objects of that the project
/// doesn't have: a message, the class, and the setting giving it.
fn missing_created_classes(
    project: &ProjectSettings,
    command: &PipelineCommand,
) -> Vec<(String, u32, String)> {
    let mut missing: Vec<(String, u32, String)> = Vec::new();
    for (path, class) in created_classes(command) {
        let ObjectClass::Global(n) = class else {
            continue;
        };
        if project.classification.classes().iter().any(|c| c.id == n)
            || missing.iter().any(|(_, m, _)| *m == n)
        {
            continue;
        }
        let message = match (command, path.split('.').nth(1)) {
            (PipelineCommand::ExtractObjects(extract), Some(index)) => {
                let segmentation = index
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| extract.class_mapping.get(i))
                    .map_or(0, |m| m.segmentation_class.0);
                format!(
                    "Extract Objects maps segmentation class {segmentation} to class {n}, which \
                     the project doesn't have"
                )
            }
            _ => format!(
                "{} creates objects of class {n}, which the project doesn't have",
                command.name()
            ),
        };
        missing.push((message, n, path));
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_used_more_than_once_are_grouped_whatever_their_case() {
        let names = [
            "pearson_01",
            "texture_01",
            "Pearson_01",
            "x",
            "PEARSON_01",
            "X",
        ];
        assert_eq!(
            same_names(&names, |n| n),
            [vec![0, 2, 4], vec![3, 5]],
            "texture_01 is used once"
        );
    }
    use crate::workspace::module_editing::{ClassRole, create_module, plan_module};
    use evanalyzer_cfg::core_types::{ModuleParamKey, SegmentationClass};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::modules::{ModuleParameterDef, ParamBinding, ParamValue};
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        BlurSettings, ColocalizationSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
        RelateObjectsSettings, VoronoiSettings,
    };
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    fn step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn uses(id: u32, key: &str, values: &[(u32, ParamValue)]) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Module(ModuleInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (ModuleParamKey(k.to_string()), v.clone()))
                    .collect(),
            }),
        }
    }

    fn blur() -> PipelineCommand {
        PipelineCommand::Blur(BlurSettings::default())
    }

    fn pearson(name: &str) -> PipelineCommand {
        let mut command =
            PipelineCommand::default_for_key("pearson_correlation_coefficient").unwrap();
        *command.measurement_id_mut().unwrap() = name.into();
        command
    }

    fn nuclei(steps: Vec<PipelineStepSettings>) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(1),
            name: "Nuclei".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps,
        }
    }

    fn messages_at(problems: &[ModuleProblem], step: u32) -> Vec<&str> {
        problems
            .iter()
            .filter(|p| p.step == Some((PipelineId(1), StepId(step))))
            .map(|p| p.message.as_str())
            .collect()
    }

    /// Each measurement step needs an ID of its own (case aside) made of
    /// letters, digits and _.
    #[test]
    fn a_measurement_id_used_twice_or_not_valid_is_reported_at_its_steps() {
        let mut project = ProjectSettings::default();
        project.pipelines.push(nuclei(vec![
            step(1, pearson("coloc")),
            step(2, pearson("COLOC")),
            step(3, pearson("bad name")),
        ]));
        let problems = check_project(&project);
        assert_eq!(
            messages_at(&problems, 1),
            [
                "The measurement ID “coloc” is also used by step 2 of “Nuclei” - each \
              measurement step needs its own"
            ]
        );
        assert!(messages_at(&problems, 2)[0].contains("also used by step 1 of “Nuclei”"));
        assert!(messages_at(&problems, 3)[0].contains("may only have letters, digits and _"));
    }

    /// A module used twice names its measurement the same both times - the
    /// message says how each use gets its own.
    #[test]
    fn a_module_measuring_used_twice_is_told_to_make_the_name_a_parameter() {
        let mut project = ProjectSettings::default();
        project.modules.push(ModuleDefinition {
            key: "COLOC".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(1, pearson("coloc"))],
            parameters: vec![],
            origin: None,
        });
        project
            .pipelines
            .push(nuclei(vec![uses(4, "COLOC", &[]), uses(9, "COLOC", &[])]));
        let problems = check_project(&project);
        for at in [4, 9] {
            let [message] = messages_at(&problems, at)[..] else {
                panic!("step {at}: one problem, got {problems:?}");
            };
            assert!(
                message.contains("make the name a module parameter"),
                "{message}"
            );
        }
    }

    /// Module SMOOTH (a blur, its kernel size parameter 1) used by pipeline 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.modules.push(ModuleDefinition {
            key: "SMOOTH".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(1, blur())],
            parameters: vec![ModuleParameterDef {
                key: ModuleParamKey::from("1"),
                label: "Size".into(),
                description: String::new(),
                section: String::new(),
                default: ParamValue::Int(3),
                bindings: vec![ParamBinding {
                    step: StepId(1),
                    param: "kernel_size".into(),
                }],
            }],
            origin: None,
        });
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "p".into(),
            description: None,
            image_source: Default::default(),
            enabled: true,
            steps: vec![uses(1, "SMOOTH", &[])],
        });
        project
    }

    fn messages(problems: &[ModuleProblem]) -> Vec<String> {
        problems.iter().map(|p| p.message.clone()).collect()
    }

    #[test]
    fn a_working_project_has_no_problems() {
        assert_eq!(check_modules(&project()), []);
    }

    #[test]
    fn a_missing_module_and_a_value_that_doesnt_fit_are_reported_at_the_step() {
        let mut project = project();
        project.pipelines[0].steps.push(uses(2, "MISSING", &[]));
        project.pipelines[0]
            .steps
            .push(uses(3, "SMOOTH", &[(1, ParamValue::Text("big".into()))]));
        let problems = check_modules(&project);
        assert_eq!(problems.len(), 2, "{:?}", messages(&problems));
        assert_eq!(problems[0].step, Some((PipelineId(1), StepId(2))));
        assert!(problems[0].message.contains("isn't defined"));
        assert_eq!(problems[1].step, Some((PipelineId(1), StepId(3))));
        assert!(
            problems[1].message.contains("doesn't accept \"big\""),
            "{}",
            problems[1].message
        );
        assert!(step_problem(&project, &project.pipelines[0].steps[2]).is_some());
        assert!(step_problem(&project, &project.pipelines[0].steps[0]).is_none());
    }

    #[test]
    fn a_broken_module_is_reported_once_not_at_every_step_using_it() {
        let mut project = project();
        project.modules[0].parameters[0].bindings[0].param = "no_such".into();
        project.pipelines[0].steps.push(uses(2, "SMOOTH", &[]));
        let problems = check_modules(&project);
        assert!(
            problems.iter().all(|p| p.step.is_none()),
            "{:?}",
            messages(&problems)
        );
        assert!(problems.iter().any(|p| p.message.contains("no_such")));
    }

    #[test]
    fn duplicate_keys_and_steps_and_a_parameter_of_two_kinds_are_errors() {
        let mut project = project();
        let mut copy = project.modules[0].clone();
        copy.steps.push(step(
            1,
            PipelineCommand::Voronoi(VoronoiSettings::default()),
        ));
        copy.parameters[0].bindings.push(ParamBinding {
            step: StepId(1),
            param: "output_class".into(),
        });
        project.modules.push(copy);
        let text = messages(&check_modules(&project)).join("\n");
        assert!(text.contains("Two modules have the key SMOOTH"), "{text}");
        assert!(text.contains("two steps have the id 1"), "{text}");
    }

    #[test]
    fn a_parameter_binding_a_number_and_a_class_is_an_error() {
        let mut project = project();
        let definition = &mut project.modules[0];
        definition.steps.push(step(
            2,
            PipelineCommand::Voronoi(VoronoiSettings::default()),
        ));
        definition.parameters[0].bindings.push(ParamBinding {
            step: StepId(2),
            param: "output_class".into(),
        });
        let text = messages(&check_module(&project, "SMOOTH")).join("\n");
        assert!(text.contains("different kinds of setting"), "{text}");
    }

    #[test]
    fn warnings_for_local_colocalization_unbound_parameters_and_missing_mapped_classes() {
        let mut project = project();
        project.classification.classes_mut().push(Class {
            id: 1,
            ..Default::default()
        });
        let definition = &mut project.modules[0];
        definition.classes.push(Class {
            id: 1,
            ..Default::default()
        });
        definition.steps.push(step(
            2,
            PipelineCommand::Colocalization(ColocalizationSettings {
                objects: ObjectClass::Global(1),
                overlapping: vec![ObjectClass::Local(1)],
                ..Default::default()
            }),
        ));
        definition.steps.push(step(
            3,
            PipelineCommand::RelateObjects(RelateObjectsSettings {
                child_classes: vec![ObjectClass::Global(1)],
                parent_classes: vec![ObjectClass::Local(1)],
                ..Default::default()
            }),
        ));
        definition.parameters.push(ModuleParameterDef {
            key: ModuleParamKey::from("2"),
            label: "Nothing".into(),
            description: String::new(),
            section: String::new(),
            default: ParamValue::Int(1),
            bindings: vec![],
        });
        project.pipelines[0].steps.push(step(
            2,
            PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                class_mapping: vec![ExtractClassMappingSettings {
                    segmentation_class: SegmentationClass(1),
                    object_class: ObjectClass::Global(9),
                }],
                ..Default::default()
            }),
        ));
        let problems = check_modules(&project);
        assert!(
            problems.iter().all(|p| p.severity == Severity::Warning),
            "{:?}",
            messages(&problems)
        );
        let text = messages(&problems).join("\n");
        assert!(
            text.contains("colocalizes with the module class L1"),
            "{text}"
        );
        assert!(
            text.contains("links to parents of the module class L1"),
            "{text}"
        );
        assert!(text.contains("\"Nothing\" sets nothing"), "{text}");
        assert!(
            text.contains("to class 9, which the project doesn't have"),
            "{text}"
        );
    }

    #[test]
    fn a_module_made_by_create_module_passes_the_checks() {
        let mut project = ProjectSettings::default();
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "p".into(),
            description: None,
            image_source: Default::default(),
            enabled: true,
            steps: vec![step(1, blur()), step(2, blur())],
        });
        let plan = plan_module(&project, PipelineId(1), 0..=1).unwrap();
        let roles: Vec<(ObjectClass, ClassRole)> = plan
            .classes
            .iter()
            .map(|c| (c.class, c.suggested))
            .collect();
        create_module(&mut project, PipelineId(1), 0..=1, "Twice", &roles).unwrap();
        assert_eq!(check_modules(&project), []);
    }

    /// One parameter setting the input classes of a classify step and the
    /// classes of a colocalization: both class lists, whatever they hold.
    #[test]
    fn class_lists_holding_different_classes_are_one_kind() {
        use evanalyzer_cfg::settings::pipeline_command_settings::ClassifyObjectsSettings;
        let mut project = project();
        let definition = &mut project.modules[0];
        definition.steps = vec![
            step(
                1,
                PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                    input_classes: vec![ObjectClass::Global(1)],
                    ..Default::default()
                }),
            ),
            step(
                2,
                PipelineCommand::Colocalization(ColocalizationSettings {
                    overlapping: vec![
                        ObjectClass::Global(1),
                        ObjectClass::Global(2),
                        ObjectClass::Global(3),
                    ],
                    ..Default::default()
                }),
            ),
        ];
        definition.parameters[0].default = ParamValue::Int(1);
        definition.parameters[0].bindings = vec![
            ParamBinding {
                step: StepId(1),
                param: "input_classes".into(),
            },
            ParamBinding {
                step: StepId(2),
                param: "overlapping".into(),
            },
        ];
        let text = messages(&check_module(&project, "SMOOTH")).join("\n");
        assert!(!text.contains("different kinds"), "{text}");
    }

    #[test]
    fn a_parameter_of_two_kinds_can_stop_setting_the_other() {
        let mut project = project();
        let definition = &mut project.modules[0];
        definition.steps.push(step(
            2,
            PipelineCommand::Voronoi(VoronoiSettings::default()),
        ));
        definition.parameters[0].bindings.push(ParamBinding {
            step: StepId(2),
            param: "output_class".into(),
        });
        let problems = check_module(&project, "SMOOTH");
        let fix = problems
            .iter()
            .find_map(|p| p.fix.clone())
            .expect("a fix for the second binding");
        assert!(fix.label().starts_with("Stop setting"), "{}", fix.label());
        assert!(apply_fix(&mut project, &fix));
        assert_eq!(project.modules[0].parameters[0].bindings.len(), 1);
        assert_eq!(check_module(&project, "SMOOTH"), []);
    }

    /// The project's mistakes in small: SPOTS detects spots (its "Spot"
    /// parameters) and beads ("Bead"), and removes spots on beads.
    fn spots_project() -> ProjectSettings {
        use evanalyzer_cfg::core_types::ImageChannelIdx;
        use evanalyzer_cfg::settings::images_settings::{
            ChannelSettings, ImageEntry, SeriesSettings,
        };
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ClassifyObjectsSettings, LoadChannelSettings,
        };
        let extract = |class: u32| {
            PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                class_mapping: vec![ExtractClassMappingSettings {
                    segmentation_class: SegmentationClass(1),
                    object_class: ObjectClass::Global(class),
                }],
                ..Default::default()
            })
        };
        let parameter =
            |key: &str, section: &str, label: &str, default: i64, bindings: &[(u32, &str)]| {
                ModuleParameterDef {
                    key: ModuleParamKey::from(key),
                    label: label.into(),
                    description: String::new(),
                    section: section.into(),
                    default: ParamValue::Int(default),
                    bindings: bindings
                        .iter()
                        .map(|(step, param)| ParamBinding {
                            step: StepId(*step),
                            param: param.to_string(),
                        })
                        .collect(),
                }
            };
        let mut project = ProjectSettings::default();
        for (id, name) in [(1, "Spot"), (2, "Bead")] {
            project.classification.classes_mut().push(Class {
                id,
                name: name.into(),
                ..Default::default()
            });
        }
        let mut series = SeriesSettings::default();
        for (channel, name) in ["CY5", "FITC"].iter().enumerate() {
            series.channels.insert(
                channel as i32,
                ChannelSettings {
                    name: name.to_string(),
                    ..Default::default()
                },
            );
        }
        project.images.list.insert(
            "a.vsi".into(),
            ImageEntry {
                series: [(0, series)].into(),
                ..Default::default()
            },
        );
        let mut definition = ModuleDefinition {
            key: "SPOTS".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![
                step(
                    1,
                    PipelineCommand::LoadChannel(LoadChannelSettings {
                        channel: ImageChannelIdx(0),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    }),
                ),
                step(2, extract(1)),
                step(
                    3,
                    PipelineCommand::LoadChannel(LoadChannelSettings {
                        channel: ImageChannelIdx(1),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    }),
                ),
                step(4, extract(2)),
                step(
                    5,
                    PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                        input_classes: vec![ObjectClass::Global(1)],
                        overlapping_with: ObjectClass::Global(2),
                        ..Default::default()
                    }),
                ),
            ],
            parameters: vec![
                parameter("spot_channel", "Spot", "Channel", 0, &[(1, "channel")]),
                parameter(
                    "spot_class",
                    "Spot",
                    "Output class",
                    1,
                    &[(2, "class_mapping.0.object_class"), (5, "input_classes")],
                ),
                parameter("bead_channel", "Bead", "Channel", 1, &[(3, "channel")]),
                parameter(
                    "bead_class",
                    "Bead",
                    "Output class",
                    2,
                    &[(4, "class_mapping.0.object_class"), (5, "overlapping_with")],
                ),
            ],
            origin: None,
        };
        definition.meta.name = "Spots".into();
        project.modules.push(definition);
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "Spots".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![uses(1, "SPOTS", &[])],
        });
        project
    }

    fn set(project: &mut ProjectSettings, step: usize, key: &str, value: i64) {
        let StepContent::Module(instance) = &mut project.pipelines[0].steps[step].content else {
            panic!("a module step");
        };
        instance
            .values
            .insert(ModuleParamKey::from(key), ParamValue::Int(value));
    }

    /// A Load Image Channel step with its own projection while the project
    /// analyzes each z-plane: a warning at the step, for a command step and
    /// inside a module; none with the default (one plane) or a projection.
    #[test]
    fn a_projection_of_its_own_while_planes_are_analyzed_one_by_one_warns() {
        use evanalyzer_cfg::core_types::ImageChannelIdx;
        use evanalyzer_cfg::settings::images_settings::{ZStackHandling, ZStackSettings};
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            LoadChannelSettings, MathLoadChannelChannelZProjectionSettings as Projection,
        };
        let mut project = spots_project();
        project.pipelines.push(PipelineSettings {
            id: PipelineId(2),
            name: "Nuclei".into(),
            description: None,
            image_source: ImageAddress::None,
            enabled: true,
            steps: vec![step(
                7,
                PipelineCommand::LoadChannel(LoadChannelSettings {
                    channel: ImageChannelIdx(0),
                    projection: Projection::MaxIntensity,
                    z_index: 0,
                }),
            )],
        });
        let warnings = |project: &ProjectSettings| -> Vec<ModuleProblem> {
            check_project(project)
                .into_iter()
                .filter(|p| p.severity == Severity::Warning && p.message.contains("one by one"))
                .collect()
        };
        let z_stack = |z_projection, z_range| {
            Some(ZStackSettings {
                z_projection,
                z_range,
            })
        };

        assert_eq!(warnings(&project), [], "the default: one plane");
        project.images.settings.z_stack = z_stack(ZStackHandling::MaxIntensity, None);
        assert_eq!(warnings(&project), [], "the project projects too");
        project.images.settings.z_stack = z_stack(ZStackHandling::SingleStack, Some(2..=2));
        assert_eq!(warnings(&project), [], "one plane");

        for one_by_one in [
            z_stack(ZStackHandling::AllStacks, None),
            z_stack(ZStackHandling::SingleStack, Some(0..=3)),
        ] {
            project.images.settings.z_stack = one_by_one;
            let found = warnings(&project);
            assert_eq!(found.len(), 1, "{:?}", messages(&found));
            assert_eq!(found[0].step, Some((PipelineId(2), StepId(7))));
            assert!(
                found[0]
                    .message
                    .starts_with("LoadChannel loads its channel with a z-projection"),
                "{}",
                found[0].message
            );
        }

        // Inside a module: at the step using it.
        let PipelineCommand::LoadChannel(load) = (match &mut project.modules[0].steps[0].content {
            StepContent::Command(command) => command,
            _ => panic!("a command"),
        }) else {
            panic!("LoadChannel");
        };
        load.projection = Projection::None;
        load.z_index = 1;
        let found = warnings(&project);
        assert_eq!(found.len(), 2, "{:?}", messages(&found));
        let inside = found
            .iter()
            .find(|p| p.step == Some((PipelineId(1), StepId(1))))
            .unwrap();
        assert!(
            inside.message.starts_with("“Spots” loads"),
            "{}",
            inside.message
        );
        assert_eq!(inside.detail, "LoadChannel (its step 1)");
    }

    #[test]
    fn the_spots_project_at_its_defaults_has_no_problems() {
        assert_eq!(
            messages(&check_project(&spots_project())),
            Vec::<String>::new()
        );
    }

    /// Beads given the spots' class: the spots on beads are then compared
    /// with spots, and beads count as spots.
    #[test]
    fn two_parameters_creating_one_class_are_named_at_the_step() {
        let mut project = spots_project();
        set(&mut project, 0, "bead_class", 1);
        let problems = check_project(&project);
        let collision = problems
            .iter()
            .find(|p| p.message.contains("both create objects of class 1 “Spot”"))
            .unwrap_or_else(|| panic!("{:?}", messages(&problems)));
        assert_eq!(collision.step, Some((PipelineId(1), StepId(1))));
        assert!(
            collision
                .message
                .starts_with("“Output class” (Spot) and “Output class” (Bead)"),
            "{}",
            collision.message
        );
        assert_eq!(
            collision.parameters,
            [
                ModuleParamKey::from("spot_class"),
                ModuleParamKey::from("bead_class")
            ]
        );
        let overlap = problems
            .iter()
            .find(|p| {
                p.message
                    .contains("overlapping other class 1 “Spot” objects")
            })
            .unwrap_or_else(|| panic!("{:?}", messages(&problems)));
        assert!(
            overlap
                .message
                .starts_with("“Output class” (Bead) and “Output class” (Spot) are class 1"),
            "{}",
            overlap.message
        );
        assert_eq!(overlap.detail, "ClassifyObjects (its step 5)");
        assert_eq!(
            overlap.parameters,
            [
                ModuleParamKey::from("bead_class"),
                ModuleParamKey::from("spot_class")
            ]
        );
    }

    #[test]
    fn a_channel_the_images_dont_have_is_an_error_at_the_parameter_setting_it() {
        let mut project = spots_project();
        set(&mut project, 0, "bead_channel", 3);
        project.pipelines[0].image_source = ImageAddress::Channel(2);
        let problems = check_project(&project);
        let errors: Vec<&ModuleProblem> = problems
            .iter()
            .filter(|p| p.severity == Severity::Error)
            .collect();
        assert_eq!(errors.len(), 2, "{:?}", messages(&problems));
        assert_eq!(
            errors[0].message,
            "The pipeline “Spots” itself starts from channel 2 (its image source - ✎ edits \
             it), but the images have only channels 0–1 (CY5, FITC)"
        );
        assert_eq!(errors[0].step, Some((PipelineId(1), StepId::UNASSIGNED)));
        assert_eq!(
            errors[1].message,
            "“Channel” (Bead) is channel 3, but the images have only channels 0–1 (CY5, FITC)",
            "named as the step's card shows it"
        );
        assert_eq!(
            errors[1].detail, "LoadChannel (its step 3)",
            "for debug mode"
        );
        assert_eq!(errors[1].parameters, [ModuleParamKey::from("bead_channel")]);
    }

    #[test]
    fn a_class_created_by_two_steps_is_reported_at_both() {
        let mut project = spots_project();
        project.pipelines[0].steps.push(step(
            2,
            PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                class_mapping: vec![ExtractClassMappingSettings {
                    segmentation_class: SegmentationClass(1),
                    object_class: ObjectClass::Global(2),
                }],
                ..Default::default()
            }),
        ));
        let problems = check_project(&project);
        let at: Vec<Option<(PipelineId, StepId)>> = problems
            .iter()
            .filter(|p| p.message.contains("“Bead”") && p.message.contains("one class"))
            .map(|p| p.step)
            .collect();
        assert_eq!(
            at,
            [
                Some((PipelineId(1), StepId(1))),
                Some((PipelineId(1), StepId(2)))
            ],
            "{:?}",
            messages(&problems)
        );
        assert!(
            problems.iter().any(|p| p.message
                == "“Output class” (Bead) is class 2 “Bead” - but the pipeline “Spots” at #2 \
                    (ExtractObjects) gives new objects that class too; objects of both end up \
                    in one class and can't be told apart"),
            "the module step names its parameter: {:?}",
            messages(&problems)
        );
        assert!(
            problems.iter().any(|p| p.message
                == "Class 2 “Bead” is also given to new objects by the pipeline “Spots” at #1 \
                    (“Spots”) - objects of both end up in one class and can't be told apart"),
            "{:?}",
            messages(&problems)
        );
    }

    #[test]
    fn a_class_a_step_creates_and_the_project_lacks_can_be_added() {
        let mut project = spots_project();
        set(&mut project, 0, "bead_class", 7);
        let problems = check_project(&project);
        let missing = problems
            .iter()
            .find(|p| p.fix == Some(ProblemFix::AddClass(7)))
            .unwrap_or_else(|| panic!("{:?}", messages(&problems)));
        assert_eq!(missing.parameters, [ModuleParamKey::from("bead_class")]);
        assert_eq!(missing.fix.as_ref().unwrap().label(), "Add class 7");
        assert!(apply_fix(&mut project, missing.fix.as_ref().unwrap()));
        assert!(project.classification.classes().iter().any(|c| c.id == 7));
        assert!(
            !check_project(&project)
                .iter()
                .any(|p| p.fix == Some(ProblemFix::AddClass(7)))
        );
        assert!(
            !apply_fix(&mut project, &ProblemFix::AddClass(7)),
            "added already"
        );
    }

    #[test]
    fn a_class_worked_on_that_no_step_gives_objects_is_reported() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ClassifyObjectsSettings, LoadAnnotatedObjectsSettings,
        };
        let classify = |input: u32, output: ObjectClass| {
            PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                input_classes: vec![ObjectClass::Global(input)],
                output_class: output,
                ..Default::default()
            })
        };
        let mut project = spots_project();
        project.classification.classes_mut().push(Class {
            id: 9,
            name: "Dim".into(),
            ..Default::default()
        });
        project.pipelines[0]
            .steps
            .push(step(2, classify(9, ObjectClass::Unset)));
        let unmade = |project: &ProjectSettings| -> Vec<String> {
            check_project(project)
                .into_iter()
                .filter(|p| p.message.contains("nothing to work on"))
                .map(|p| p.message)
                .collect()
        };
        assert_eq!(
            unmade(&project),
            [
                "ClassifyObjects works on class 9 “Dim”, but no step gives any object that \
              class - there's nothing to work on"
            ]
        );

        // A second step working on it: still one warning, at the first.
        project.pipelines[0]
            .steps
            .push(step(3, classify(9, ObjectClass::Unset)));
        let grouped = unmade(&project);
        assert_eq!(grouped.len(), 1, "{grouped:?}");
        assert!(
            grouped[0].ends_with("- 1 more step of this pipeline works on it too"),
            "{}",
            grouped[0]
        );
        project.pipelines[0].steps.pop();

        // Given by a step of a disabled pipeline: still nothing when run.
        project.pipelines.push(PipelineSettings {
            id: PipelineId(2),
            name: "Off".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: false,
            steps: vec![step(1, classify(1, ObjectClass::Global(9)))],
        });
        assert_eq!(unmade(&project).len(), 1);
        // Enabled: spots are classified as class 9.
        project.pipelines[1].enabled = true;
        assert_eq!(unmade(&project), Vec::<String>::new());

        // Annotations loaded with their classes may bring any class.
        project.pipelines[1].steps = vec![step(
            1,
            PipelineCommand::LoadAnnotatedObjects(LoadAnnotatedObjectsSettings {
                keep_annotated_classes: true,
                ..Default::default()
            }),
        )];
        assert_eq!(unmade(&project), Vec::<String>::new());
    }

    /// Without a class mapping, Extract Objects gives each segmentation
    /// class's objects the project class with the same number - nothing to
    /// warn about, and the classes it may make are not "never made".
    #[test]
    fn an_extract_objects_without_a_class_mapping_is_fine() {
        use evanalyzer_cfg::settings::pipeline_command_settings::ClassifyObjectsSettings;
        let mut project = spots_project();
        project.classification.classes_mut().push(Class {
            id: 9,
            name: "Nuclei".into(),
            ..Default::default()
        });
        project.pipelines[0].steps.push(step(
            2,
            PipelineCommand::ExtractObjects(ExtractObjectsSettings {
                class_mapping: vec![],
                ..Default::default()
            }),
        ));
        // Class 9 is made only by the unmapped segmentation class 9.
        project.pipelines[0].steps.push(step(
            3,
            PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                input_classes: vec![ObjectClass::Global(9)],
                output_class: ObjectClass::Unset,
                ..Default::default()
            }),
        ));
        let problems = check_project(&project);
        assert!(
            !problems
                .iter()
                .any(|p| p.message.contains("creates no objects")),
            "{:?}",
            messages(&problems)
        );
        assert!(
            !problems
                .iter()
                .any(|p| p.message.contains("nothing to work on")),
            "{:?}",
            messages(&problems)
        );
    }

    /// A disabled pipeline doesn't run: its channel that doesn't exist, the
    /// classes it creates are no problem.
    #[test]
    fn a_disabled_pipeline_has_no_problems_of_a_run() {
        let mut project = spots_project();
        project.pipelines[0].image_source = ImageAddress::Channel(5);
        project.pipelines.push(project.pipelines[0].clone());
        project.pipelines[1].id = PipelineId(2);
        project.pipelines[0].enabled = false;
        let problems = check_project(&project);
        assert_eq!(problems.len(), 1, "{:?}", messages(&problems));
        assert_eq!(problems[0].step, Some((PipelineId(2), StepId::UNASSIGNED)));
        project.pipelines[1].enabled = false;
        assert_eq!(messages(&check_project(&project)), Vec::<String>::new());
    }
}
