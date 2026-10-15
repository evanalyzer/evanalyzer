//! Checking a project's components - on load and in the editor, so
//! mistakes show where they are instead of only when a run starts.
//!
//! Most checks are the expansion itself (`evanalyzer_core::expand_step`):
//! what a run would refuse, reported per component and per step using it. The
//! rest are things expansion can't see: duplicate keys, parameters binding
//! settings of different kinds, and warnings for what works but probably
//! isn't meant.

use crate::workspace::component_editing::{same_kind, setting_at};
use evanalyzer_cfg::{
    core_types::{ObjectClass, PipelineId, StepId},
    settings::{
        components::{ComponentDefinition, ComponentInstance},
        pipeline_command::PipelineCommand,
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

/// One problem found by [`check_components`].
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentProblem {
    pub severity: Severity,
    /// The component it's about, if any.
    pub component: Option<String>,
    /// The pipeline step it's about, if any: (pipeline, step id).
    pub step: Option<(PipelineId, StepId)>,
    pub message: String,
}

impl ComponentProblem {
    fn error(component: Option<&str>, step: Option<(PipelineId, StepId)>, message: String) -> Self {
        Self {
            severity: Severity::Error,
            component: component.map(str::to_string),
            step,
            message,
        }
    }

    fn warning(
        component: Option<&str>,
        step: Option<(PipelineId, StepId)>,
        message: String,
    ) -> Self {
        Self {
            severity: Severity::Warning,
            ..Self::error(component, step, message)
        }
    }
}

/// Every problem of the project's components and the steps using them.
pub fn check_components(project: &ProjectSettings) -> Vec<ComponentProblem> {
    let mut problems = Vec::new();

    let mut seen: Vec<&str> = Vec::new();
    for definition in &project.components {
        if seen.contains(&definition.key.as_str()) {
            problems.push(ComponentProblem::error(
                Some(&definition.key),
                None,
                format!(
                    "Two components have the key {} - steps can't tell which they use",
                    definition.key
                ),
            ));
        }
        seen.push(&definition.key);
    }

    let mut broken: Vec<&str> = Vec::new();
    for definition in &project.components {
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
                StepContent::Component(instance) => {
                    if !project.components.iter().any(|d| d.key == instance.key) {
                        problems.push(ComponentProblem::error(
                            Some(&instance.key),
                            at,
                            format!("Component {} isn't defined in this project", instance.key),
                        ));
                    } else if !broken.contains(&instance.key.as_str()) {
                        // The component itself is fine: what's wrong are this
                        // step's values.
                        if let Err(e) =
                            expand_step(step, &project.components, &mut ScopeCounter::default())
                        {
                            problems.push(ComponentProblem::error(
                                Some(&instance.key),
                                at,
                                e.to_string(),
                            ));
                        }
                    }
                }
                StepContent::Command(command) => {
                    if let Err(e) =
                        expand_step(step, &project.components, &mut ScopeCounter::default())
                    {
                        problems.push(ComponentProblem::error(None, at, e.to_string()));
                    }
                    problems.extend(
                        missing_mapped_classes(project, command)
                            .into_iter()
                            .map(|message| ComponentProblem::warning(None, at, message)),
                    );
                }
            }
        }
    }
    problems
}

/// The problems of component `key` alone (the component, not the steps using it).
pub fn check_component(project: &ProjectSettings, key: &str) -> Vec<ComponentProblem> {
    project
        .components
        .iter()
        .find(|d| d.key == key)
        .map(|definition| check_definition(project, definition))
        .unwrap_or_default()
}

/// What's wrong with pipeline step `step` - a component step whose component or
/// values don't work, a command a run would refuse - or `None`.
pub fn step_problem(project: &ProjectSettings, step: &PipelineStepSettings) -> Option<String> {
    if let StepContent::Component(instance) = &step.content
        && let Some(problem) = check_component(project, &instance.key)
            .into_iter()
            .find(|p| p.severity == Severity::Error)
    {
        return Some(problem.message);
    }
    expand_step(step, &project.components, &mut ScopeCounter::default())
        .err()
        .map(|e| e.to_string())
}

fn check_definition(
    project: &ProjectSettings,
    definition: &ComponentDefinition,
) -> Vec<ComponentProblem> {
    let key = Some(definition.key.as_str());
    let mut problems = Vec::new();
    let error = |message: String| ComponentProblem::error(key, None, message);
    let named = |message: String| format!("Component {}: {message}", definition.key);

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
            problems.push(ComponentProblem::warning(
                key,
                None,
                named(format!("parameter \"{}\" sets nothing", parameter.label)),
            ));
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
                    Some((setting, label)) => kinds.push((setting, label)),
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
        if let Some((first_setting, first)) = kinds.first()
            && let Some((_, other)) = kinds
                .iter()
                .find(|(setting, _)| !same_kind(first_setting, setting))
        {
            problems.push(error(named(format!(
                "parameter \"{}\" sets {first} and {other}, which are different kinds of setting",
                parameter.label
            ))));
        }
    }

    // The component at its defaults: what a run would refuse.
    let at_defaults = PipelineStepSettings {
        id: StepId::UNASSIGNED,
        enabled: true,
        content: StepContent::Component(ComponentInstance {
            key: definition.key.clone(),
            values: Default::default(),
        }),
    };
    if let Err(e) = expand_step(
        &at_defaults,
        &project.components,
        &mut ScopeCounter::default(),
    ) {
        problems.push(error(e.to_string()));
    }

    for step in &definition.steps {
        let Some(command) = step.command() else {
            continue;
        };
        if let PipelineCommand::Colocalization(coloc) = command
            && let Some(local) = coloc
                .classes_to_coloc
                .iter()
                .find(|c| matches!(c, ObjectClass::Local(_)))
        {
            problems.push(ComponentProblem::warning(
                key,
                None,
                named(format!(
                    "step {} colocalizes with the component class {} - its links are dropped when \
                     the component is done, so the results have no colocalization counts for it",
                    step.id,
                    local.to_text()
                )),
            ));
        }
        problems.extend(
            missing_mapped_classes(project, command)
                .into_iter()
                .map(|message| ComponentProblem::warning(key, None, named(message))),
        );
    }
    problems
}

/// An Extract Objects mapping to project classes the project doesn't have.
fn missing_mapped_classes(project: &ProjectSettings, command: &PipelineCommand) -> Vec<String> {
    let PipelineCommand::ExtractObjects(extract) = command else {
        return Vec::new();
    };
    extract
        .class_mapping
        .iter()
        .filter_map(|mapping| match mapping.object_class {
            ObjectClass::Global(n)
                if !project.classification.classes().iter().any(|c| c.id == n) =>
            {
                Some(format!(
                    "Extract Objects maps segmentation class {} to class {n}, which the project \
                     doesn't have",
                    mapping.segmentation_class.0
                ))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::component_editing::{ClassRole, create_component, plan_component};
    use evanalyzer_cfg::core_types::{ComponentParamKey, SegmentationClass};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::components::{ComponentParameterDef, ParamBinding, ParamValue};
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        BlurSettings, ColocalizationSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
        VoronoiSettings,
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
            content: StepContent::Component(ComponentInstance {
                key: key.into(),
                values: values
                    .iter()
                    .map(|(k, v)| (ComponentParamKey(k.to_string()), v.clone()))
                    .collect(),
            }),
        }
    }

    fn blur() -> PipelineCommand {
        PipelineCommand::Blur(BlurSettings::default())
    }

    /// Component SMOOTH (a blur, its kernel size parameter 1) used by pipeline 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.components.push(ComponentDefinition {
            key: "SMOOTH".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(1, blur())],
            parameters: vec![ComponentParameterDef {
                key: ComponentParamKey::from("1"),
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

    fn messages(problems: &[ComponentProblem]) -> Vec<String> {
        problems.iter().map(|p| p.message.clone()).collect()
    }

    #[test]
    fn a_working_project_has_no_problems() {
        assert_eq!(check_components(&project()), []);
    }

    #[test]
    fn a_missing_component_and_a_value_that_doesnt_fit_are_reported_at_the_step() {
        let mut project = project();
        project.pipelines[0].steps.push(uses(2, "MISSING", &[]));
        project.pipelines[0]
            .steps
            .push(uses(3, "SMOOTH", &[(1, ParamValue::Text("big".into()))]));
        let problems = check_components(&project);
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
    fn a_broken_component_is_reported_once_not_at_every_step_using_it() {
        let mut project = project();
        project.components[0].parameters[0].bindings[0].param = "no_such".into();
        project.pipelines[0].steps.push(uses(2, "SMOOTH", &[]));
        let problems = check_components(&project);
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
        let mut copy = project.components[0].clone();
        copy.steps.push(step(
            1,
            PipelineCommand::Voronoi(VoronoiSettings::default()),
        ));
        copy.parameters[0].bindings.push(ParamBinding {
            step: StepId(1),
            param: "output_class".into(),
        });
        project.components.push(copy);
        let text = messages(&check_components(&project)).join("\n");
        assert!(
            text.contains("Two components have the key SMOOTH"),
            "{text}"
        );
        assert!(text.contains("two steps have the id 1"), "{text}");
    }

    #[test]
    fn a_parameter_binding_a_number_and_a_class_is_an_error() {
        let mut project = project();
        let definition = &mut project.components[0];
        definition.steps.push(step(
            2,
            PipelineCommand::Voronoi(VoronoiSettings::default()),
        ));
        definition.parameters[0].bindings.push(ParamBinding {
            step: StepId(2),
            param: "output_class".into(),
        });
        let text = messages(&check_component(&project, "SMOOTH")).join("\n");
        assert!(text.contains("different kinds of setting"), "{text}");
    }

    #[test]
    fn warnings_for_local_colocalization_unbound_parameters_and_missing_mapped_classes() {
        let mut project = project();
        project.classification.classes_mut().push(Class {
            id: 1,
            ..Default::default()
        });
        let definition = &mut project.components[0];
        definition.classes.push(Class {
            id: 1,
            ..Default::default()
        });
        definition.steps.push(step(
            2,
            PipelineCommand::Colocalization(ColocalizationSettings {
                classes_to_coloc: vec![ObjectClass::Global(1), ObjectClass::Local(1)],
                ..Default::default()
            }),
        ));
        definition.parameters.push(ComponentParameterDef {
            key: ComponentParamKey::from("2"),
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
        let problems = check_components(&project);
        assert!(
            problems.iter().all(|p| p.severity == Severity::Warning),
            "{:?}",
            messages(&problems)
        );
        let text = messages(&problems).join("\n");
        assert!(
            text.contains("colocalizes with the component class L1"),
            "{text}"
        );
        assert!(text.contains("\"Nothing\" sets nothing"), "{text}");
        assert!(
            text.contains("to class 9, which the project doesn't have"),
            "{text}"
        );
    }

    #[test]
    fn a_component_made_by_create_component_passes_the_checks() {
        let mut project = ProjectSettings::default();
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "p".into(),
            description: None,
            image_source: Default::default(),
            enabled: true,
            steps: vec![step(1, blur()), step(2, blur())],
        });
        let plan = plan_component(&project, PipelineId(1), 0..=1).unwrap();
        let roles: Vec<(ObjectClass, ClassRole)> = plan
            .classes
            .iter()
            .map(|c| (c.class, c.suggested))
            .collect();
        create_component(&mut project, PipelineId(1), 0..=1, "Twice", &roles).unwrap();
        assert_eq!(check_components(&project), []);
    }
}
