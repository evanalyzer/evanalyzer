//! Command groups: a reusable command made of other commands (e.g. "Cell
//! reconstruction"), showing only its own parameters.
//!
//! A group is defined once in the project ([`GroupDefinition`]) and used in
//! pipelines as instances ([`GroupInstance`]) - changing the definition
//! changes every instance. An instance only holds what differs per use: its
//! parameter values. See VISION_GROUP.md.

use crate::{
    modules::pipeline_settings::PipelineStepSettings,
    types::ids::{GroupId, GroupParamId, StepId},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A group as defined in the project: its steps and the parameters it shows.
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GroupDefinition {
    /// Unique group key, prefere SCREAMING_SNAKE_CASE
    pub key: String,

    /// Group name
    pub name: String,

    /// Description of what the group is doing
    pub description: Option<String>,

    /// Counts up on every change of the definition - lets copies (other
    /// projects, the library) tell an older from a newer definition.
    #[serde(default)]
    pub version: u32,

    /// The steps which are executed in the given order
    pub steps: Vec<PipelineStepSettings>,

    /// The group's parameters: shown to the user ([`ParamKind::Exposed`])
    /// or set per instance behind the scenes ([`ParamKind::InternalClass`]).
    pub parameters: Vec<GroupParameterDef>,

    /// The library file this definition was inserted from, if any.
    #[serde(default)]
    pub origin: Option<GroupOrigin>,
}

/// One parameter of a group: what it sets inside the group, and its default.
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GroupParameterDef {
    /// Unique within the group definition.
    pub id: GroupParamId,

    /// Name shown to the user, e.g. "Cell class".
    pub label: String,

    /// Explanation shown to the user.
    #[serde(default)]
    pub description: String,

    /// The value an instance starts with - formatted like
    /// `ParameterDef::value` (what `PipelineCommand::apply_param_change`
    /// takes).
    pub default: String,

    pub kind: ParamKind,

    /// The inner parameters this one sets - all of the same type. Type,
    /// options and range are those of the bound parameters (not stored here,
    /// so they can't go stale when a command changes).
    pub bindings: Vec<ParamBinding>,
}

/// Whether the user sees a group parameter.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParamKind {
    /// Shown on the group card and set by the user.
    #[default]
    Exposed,
    /// A class only used inside the group (e.g. "Seed"): hidden, and each
    /// instance gets a class of its own, so two instances never share it.
    InternalClass,
}

/// An inner parameter a group parameter sets.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ParamBinding {
    /// The step of the group that owns the parameter.
    pub step: StepId,

    /// The parameter's path in that step's command, as `ParameterDef::name`
    /// gives it (e.g. "class_out", "method.classes"). A path into a list
    /// entry ("thresholds.2.min") breaks when entries are added or removed.
    pub param: String,
}

/// Where a definition came from: a group library file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GroupOrigin {
    /// The library file's name (`.evagroup`).
    pub library: String,

    /// The definition's version in that file when it was inserted.
    pub version: u32,
}

/// A use of a group in a pipeline: which definition, with which values.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GroupInstance {
    /// The group's definition in the project.
    pub definition: GroupId,

    /// This instance's parameter values. A parameter missing here has its
    /// definition's default (e.g. one the definition gained later).
    #[serde(default)]
    pub values: BTreeMap<GroupParamId, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::pipeline_command::PipelineCommand;
    use crate::settings::pipeline_command_settings::BlurSettings;

    fn definition() -> GroupDefinition {
        let step = StepId::generate();
        GroupDefinition {
            key: "CELL_RECONSTRUCTION".into(),
            name: "Cell reconstruction".into(),
            description: None,
            version: 3,
            steps: vec![PipelineStepSettings {
                id: step,
                enabled: true,
                command: PipelineCommand::Blur(BlurSettings::default()),
            }],
            parameters: vec![GroupParameterDef {
                id: GroupParamId(1),
                label: "Blur size".into(),
                description: String::new(),
                default: "3".into(),
                kind: ParamKind::Exposed,
                bindings: vec![ParamBinding {
                    step,
                    param: "kernel_size".into(),
                }],
            }],
            origin: Some(GroupOrigin {
                library: "cells.evagroup".into(),
                version: 3,
            }),
        }
    }

    #[test]
    fn a_definition_and_an_instance_survive_a_json_round_trip() {
        let definition = definition();
        let json = serde_json::to_string(&definition).unwrap();
        let back: GroupDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.steps[0].id, definition.steps[0].id);
        assert_eq!(
            back.parameters[0].bindings,
            definition.parameters[0].bindings
        );
        assert_eq!(back.origin, definition.origin);

        let instance = GroupInstance {
            definition: GroupId(1),
            values: BTreeMap::from([(GroupParamId(1), "5".to_string())]),
        };
        let json = serde_json::to_string(&instance).unwrap();
        assert_eq!(
            serde_json::from_str::<GroupInstance>(&json).unwrap(),
            instance
        );
    }

    #[test]
    fn optional_parts_may_be_missing() {
        let instance: GroupInstance = serde_json::from_str(r#"{"definition":1}"#).unwrap();
        assert!(instance.values.is_empty(), "every parameter at its default");

        let parameter: GroupParameterDef = serde_json::from_str(
            r#"{"id":1,"label":"Seed","default":"","kind":"INTERNAL_CLASS","bindings":[]}"#,
        )
        .unwrap();
        assert_eq!(parameter.kind, ParamKind::InternalClass);
    }
}
