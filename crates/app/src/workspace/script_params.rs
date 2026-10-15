//! A Script step's `params` list follows its script's `param` lines: the
//! script declares them, the list holds the values set on the step (and is
//! what group parameters bind to). Refreshed when the script is edited and
//! when a project is loaded, so a script edited by hand needs nothing else.

use evanalyzer_cfg::{
    core_types::StepId,
    settings::{
        command_groups::GroupDefinition,
        pipeline_command::PipelineCommand,
        pipeline_command_settings::{
            ScriptParamSettings, ScriptScriptExecutorScriptValueSettings as ValueSettings,
            ScriptSettings,
        },
        pipeline_settings::PipelineStepSettings,
        project_settings::ProjectSettings,
    },
};
use evanalyzer_core::{ParamKind, declared_script_params};

/// Where each entry of the old list went: `moved[old] = Some(new)`, or
/// `None` if its `param` is gone (or changed its type).
pub type Moved = Vec<Option<usize>>;

fn kind_of(value: &ValueSettings) -> ParamKind {
    match value {
        ValueSettings::Number(_) => ParamKind::Number,
        ValueSettings::Decimal(_) => ParamKind::Decimal,
        ValueSettings::Bool(_) => ParamKind::Bool,
        ValueSettings::Text(_) => ParamKind::Text,
        ValueSettings::Class(_) => ParamKind::Class,
        ValueSettings::Channel(_) => ParamKind::Channel,
    }
}

fn to_settings(value: &evanalyzer_core::ScriptValue) -> ValueSettings {
    use evanalyzer_core::ScriptValue as V;
    match value {
        V::Number(n) => ValueSettings::Number(*n),
        V::Decimal(x) => ValueSettings::Decimal(*x),
        V::Bool(b) => ValueSettings::Bool(*b),
        V::Text(t) => ValueSettings::Text(t.clone()),
        V::Class(c) => ValueSettings::Class(*c),
        V::Channel(c) => ValueSettings::Channel(*c),
    }
}

/// Brings `settings.params` in line with the script's `param` lines: one
/// entry per `param`, in their order, keeping the value of an entry with the
/// same name and type, else the default.
///
/// # Errors
/// The script's `param` lines are wrong (with the line) - the list stays.
pub fn sync_script_params(settings: &mut ScriptSettings) -> Result<Moved, String> {
    let decls = declared_script_params(&settings.source)?;
    let old = std::mem::take(&mut settings.params);
    let mut moved: Moved = vec![None; old.len()];
    settings.params = decls
        .iter()
        .enumerate()
        .map(|(new, decl)| {
            let kept = old
                .iter()
                .position(|p| p.name == decl.name && kind_of(&p.value) == decl.kind);
            if let Some(index) = kept {
                moved[index] = Some(new);
            }
            ScriptParamSettings {
                name: decl.name.clone(),
                value: kept.map_or_else(|| to_settings(&decl.default), |i| old[i].value.clone()),
            }
        })
        .collect();
    Ok(moved)
}

/// The bindings of `definition` to step `step_id`'s params follow
/// `moved`; a binding to a param that's gone goes, and a parameter with no
/// binding left too. Returns whether anything changed.
pub fn rebind_script_params(
    definition: &mut GroupDefinition,
    step_id: StepId,
    moved: &[Option<usize>],
) -> bool {
    let mut changed = false;
    for parameter in &mut definition.parameters {
        parameter.bindings.retain_mut(|binding| {
            if binding.step != step_id {
                return true;
            }
            let Some(rest) = binding.param.strip_prefix("params.") else {
                return true;
            };
            let (index, tail) = rest.split_once('.').unwrap_or((rest, ""));
            let Ok(index) = index.parse::<usize>() else {
                return true;
            };
            match moved.get(index).copied().flatten() {
                Some(new) if new == index => true,
                Some(new) => {
                    binding.param = if tail.is_empty() {
                        format!("params.{new}")
                    } else {
                        format!("params.{new}.{tail}")
                    };
                    changed = true;
                    true
                }
                None => {
                    changed = true;
                    false
                }
            }
        });
    }
    let before = definition.parameters.len();
    definition.parameters.retain(|p| !p.bindings.is_empty());
    changed || definition.parameters.len() != before
}

/// Syncs every Script step of the project (pipelines and groups); the
/// groups' bindings follow. Returns the problems found, one per step.
pub fn sync_project_script_params(project: &mut ProjectSettings) -> Vec<String> {
    let mut problems = Vec::new();
    let mut sync = |steps: &mut Vec<PipelineStepSettings>, place: &str| {
        let mut moves = Vec::new();
        for step in steps.iter_mut() {
            if let Some(PipelineCommand::Script(settings)) = step.command_mut() {
                match sync_script_params(settings) {
                    Ok(moved) => moves.push((step.id, moved)),
                    Err(e) => problems.push(format!("{place}, step {}: {e}", step.id)),
                }
            }
        }
        moves
    };
    for pipeline in &mut project.pipelines {
        sync(
            &mut pipeline.steps,
            &format!("Pipeline \"{}\"", pipeline.name),
        );
    }
    for definition in &mut project.command_groups {
        let place = format!("Command group \"{}\"", definition.meta.name);
        let moves = sync(&mut definition.steps, &place);
        for (step_id, moved) in moves {
            rebind_script_params(definition, step_id, &moved);
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::GroupParamId;
    use evanalyzer_cfg::core_types::{ImageChannelIdx, ObjectClass};
    use evanalyzer_cfg::settings::command_groups::{
        GroupInstance, GroupParameterDef, ParamBinding, ParamValue,
    };
    use evanalyzer_cfg::settings::pipeline_settings::StepContent;

    fn script(source: &str) -> ScriptSettings {
        ScriptSettings {
            classes: vec![],
            source: source.into(),
            params: vec![],
        }
    }

    fn names(settings: &ScriptSettings) -> Vec<&str> {
        settings.params.iter().map(|p| p.name.as_str()).collect()
    }

    #[test]
    fn the_list_follows_the_params_and_keeps_values_by_name() {
        let mut settings = script(
            "param rounds = 10;\nparam start = 800.0;\nparam cell: class = 2;\nparam input: channel = 1;",
        );
        assert_eq!(
            sync_script_params(&mut settings).unwrap(),
            Vec::<Option<usize>>::new()
        );
        assert_eq!(names(&settings), ["rounds", "start", "cell", "input"]);
        assert_eq!(
            settings.params[2].value,
            ValueSettings::Class(ObjectClass::Global(2))
        );
        assert_eq!(
            settings.params[3].value,
            ValueSettings::Channel(ImageChannelIdx(1))
        );
        settings.params[0].value = ValueSettings::Number(25);

        // Reordered, one removed, one new, one changed type.
        settings.source =
            "param extra = true;\nparam start = \"high\";\nparam rounds = 10;\nparam cell: class = 2;"
                .into();
        let moved = sync_script_params(&mut settings).unwrap();
        assert_eq!(names(&settings), ["extra", "start", "rounds", "cell"]);
        assert_eq!(
            settings.params[2].value,
            ValueSettings::Number(25),
            "kept by name"
        );
        assert_eq!(
            settings.params[1].value,
            ValueSettings::Text("high".into()),
            "a new type starts at its default"
        );
        assert_eq!(moved, [Some(2), None, Some(3), None]);

        // A broken script keeps the list.
        settings.source = "param = 1;".into();
        assert!(sync_script_params(&mut settings).is_err());
        assert_eq!(names(&settings), ["extra", "start", "rounds", "cell"]);
    }

    fn binding(step: u32, param: &str) -> ParamBinding {
        ParamBinding {
            step: StepId(step),
            param: param.into(),
        }
    }

    #[test]
    fn group_bindings_follow_their_params() {
        let mut definition = GroupDefinition {
            key: "G".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![],
            parameters: vec![
                GroupParameterDef {
                    id: GroupParamId(1),
                    label: "Rounds".into(),
                    description: String::new(),
                    default: ParamValue::Int(10),
                    bindings: vec![binding(3, "params.0.value.0")],
                },
                GroupParameterDef {
                    id: GroupParamId(2),
                    label: "Start".into(),
                    description: String::new(),
                    default: ParamValue::Float(800.0),
                    bindings: vec![binding(3, "params.1.value.0")],
                },
                GroupParameterDef {
                    id: GroupParamId(3),
                    label: "Other step".into(),
                    description: String::new(),
                    default: ParamValue::Int(1),
                    bindings: vec![binding(4, "params.0.value.0")],
                },
            ],
            origin: None,
        };
        assert!(rebind_script_params(
            &mut definition,
            StepId(3),
            &[Some(2), None]
        ));
        let bound: Vec<(&str, &str)> = definition
            .parameters
            .iter()
            .map(|p| (p.label.as_str(), p.bindings[0].param.as_str()))
            .collect();
        assert_eq!(
            bound,
            [
                ("Rounds", "params.2.value.0"),
                ("Other step", "params.0.value.0")
            ],
            "a removed param's parameter goes; another step's binding stays"
        );
    }

    /// A Script step in a group, its `rounds` param on the group card: the
    /// card's value is what the script gets.
    #[test]
    fn an_exposed_param_runs_with_the_cards_value() {
        let mut settings = script("param rounds = 10;");
        sync_script_params(&mut settings).unwrap();
        let mut project = ProjectSettings::default();
        project.command_groups.push(GroupDefinition {
            key: "G".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![PipelineStepSettings {
                id: StepId(1),
                enabled: true,
                content: StepContent::Command(PipelineCommand::Script(settings)),
            }],
            parameters: vec![GroupParameterDef {
                id: GroupParamId(1),
                label: "Rounds".into(),
                description: String::new(),
                default: ParamValue::Int(10),
                bindings: vec![binding(1, "params.0.value.0")],
            }],
            origin: None,
        });
        let instance = GroupInstance {
            key: "G".into(),
            values: [(GroupParamId(1), ParamValue::Int(42))].into(),
        };
        assert!(
            crate::workspace::group_checks::check_groups(&project).is_empty(),
            "{:?}",
            crate::workspace::group_checks::check_groups(&project)
        );
        let steps =
            evanalyzer_core::group_steps_with_values(&instance, &project.command_groups).unwrap();
        let Some(PipelineCommand::Script(script)) = steps[0].command() else {
            panic!("script")
        };
        assert_eq!(script.params[0].value, ValueSettings::Number(42));
    }
}
