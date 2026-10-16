// Components: a reusable command made of other commands (e.g. "Cell
// reconstruction"), showing only its own parameters.
//
// A component is defined once in the project ([`ComponentDefinition`]) and used in
// pipelines as instances ([`ComponentInstance`]) - changing the definition
// changes every instance. An instance only holds what differs per use: its
// parameter values. Classes, objects and memory slots of an instance live
// in its own scope (`ObjectClass::Local`, `MemoryId::ComponentContext`),
// dropped when the instance's last step ran.
// (Plain comments: build.rs include!s this file, where `//!` isn't allowed.)

use crate::{
    modules::{
        classification_settings::Class, meta_data::MetaData,
        pipeline_settings::PipelineStepSettings,
    },
    types::ids::{ComponentParamKey, StepId},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A component parameter's value - written as plain JSON (`4`, `4.5`, `true`,
/// `"PARABOLOID"`, `[1, 2]`). Typed by JSON kind only: what it means (a
/// class, a channel, a dropdown option) and which values are valid come
/// from the bound inner parameters.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
#[serde(untagged)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    /// Dropdown options, units, paths ...
    Text(String),
    /// Several values, e.g. a multi-class or multi-channel selection.
    List(Vec<ParamValue>),
}

/// A component as defined in the project: its steps and the parameters it shows.
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComponentDefinition {
    /// Unique component key, prefere SCREAMING_SNAKE_CASE
    pub key: String,

    /// Metadata of this component
    pub meta: MetaData,

    /// Local classes used by the component. These classes are scoped to the component only
    #[serde(default)]
    pub classes: Vec<Class>,

    /// The steps which are executed in the given order
    pub steps: Vec<PipelineStepSettings>,

    /// The component's parameters, shown on the component card.
    pub parameters: Vec<ComponentParameterDef>,

    /// The library file this definition was inserted from, if any. Not
    /// written when absent, so library files (`.evacomp`) never carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<ComponentOrigin>,
}

/// One parameter of a component: what it sets inside the component, and its default.
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComponentParameterDef {
    /// Unique within the component definition, made from its section and
    /// label (`"tetraspeck_output_class"`) so a project file says what a
    /// value is; the instances' values are keyed by it.
    pub key: ComponentParamKey,

    /// Name shown to the user, e.g. "Cell class".
    pub label: String,

    /// Explanation shown to the user.
    #[serde(default)]
    pub description: String,

    /// Section the parameter is shown in on the component card, e.g.
    /// "Detection"; empty: the unnamed first section.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub section: String,

    /// The value an instance starts with.
    pub default: ParamValue,

    /// The inner parameters this one sets - all of the same type. Type,
    /// options and range are those of the bound parameters (not stored here,
    /// so they can't go stale when a command changes).
    pub bindings: Vec<ParamBinding>,
}

/// An inner parameter a component parameter sets.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ParamBinding {
    /// The step of the component that owns the parameter.
    pub step: StepId,

    /// The parameter's path in that step's command, as `ParameterDef::name`
    /// gives it (e.g. "class_out", "method.classes"). A path into a list
    /// entry ("thresholds.2.min") breaks when entries are added or removed.
    pub param: String,
}

/// Where a definition came from: a component library file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComponentOrigin {
    /// If this is a copy form an existing component, this is the origin component key.
    /// This allows to reset the component default settings.
    pub origin_component_key: String,

    /// The library file's name (`.evacomp`).
    pub library: String,

    /// The definition's version in that file when it was inserted.
    pub version: u32,

    /// A fingerprint of the definition as inserted - if it differs now, the
    /// project changed the component, and an update from the library would
    /// replace those changes. Empty in files written before.
    #[serde(default)]
    pub content_hash: String,
}

/// A use of a component in a pipeline: which definition, with which values.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComponentInstance {
    /// The component's key in the project.
    pub key: String,

    /// This instance's parameter values. A parameter missing here has its
    /// definition's default (e.g. one the definition gained later).
    #[serde(default)]
    pub values: BTreeMap<ComponentParamKey, ParamValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(json: &str) -> ParamValue {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn values_are_plain_json_of_their_kind() {
        assert_eq!(value("true"), ParamValue::Bool(true));
        assert_eq!(value("4"), ParamValue::Int(4));
        assert_eq!(value("-2"), ParamValue::Int(-2));
        assert_eq!(value("4.5"), ParamValue::Float(4.5));
        assert_eq!(
            value(r#""PARABOLOID""#),
            ParamValue::Text("PARABOLOID".into())
        );
        assert_eq!(
            value("[1, 2]"),
            ParamValue::List(vec![ParamValue::Int(1), ParamValue::Int(2)])
        );
        // And written back the same way - no wrapper around the value.
        for json in ["true", "4", "4.5", r#""PARABOLOID""#, "[1,2]"] {
            assert_eq!(serde_json::to_string(&value(json)).unwrap(), json);
        }
    }

    #[test]
    fn an_instance_reads_naturally_and_survives_loading_through_a_json_value() {
        let json = r#"{"key":"CELL_RECONSTRUCTION","values":{"cell_class":3,"method":"PARABOLOID","radius":4.5}}"#;
        // Projects load through `serde_json::Value`.
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let instance: ComponentInstance = serde_json::from_value(raw).unwrap();
        assert_eq!(instance.values[&"cell_class".into()], ParamValue::Int(3));
        assert_eq!(
            instance.values[&"method".into()],
            ParamValue::Text("PARABOLOID".into())
        );
        assert_eq!(instance.values[&"radius".into()], ParamValue::Float(4.5));
        assert_eq!(serde_json::to_string(&instance).unwrap(), json);

        let no_values: ComponentInstance = serde_json::from_str(r#"{"key":"X"}"#).unwrap();
        assert!(
            no_values.values.is_empty(),
            "every parameter at its default"
        );
    }

    #[test]
    fn a_parameter_definition_has_a_typed_default() {
        let parameter: ComponentParameterDef = serde_json::from_str(
            r#"{"key":"blur_size","label":"Blur size","default":3,
                "bindings":[{"step":2,"param":"kernel_size"}]}"#,
        )
        .unwrap();
        assert_eq!(parameter.default, ParamValue::Int(3));
        assert_eq!(
            parameter.bindings,
            [ParamBinding {
                step: StepId(2),
                param: "kernel_size".into()
            }]
        );
    }

    #[test]
    fn a_definition_without_classes_needs_no_classes_entry() {
        let definition = ComponentDefinition {
            key: "BLUR".into(),
            meta: MetaData::default(),
            classes: vec![],
            steps: vec![],
            parameters: vec![],
            origin: None,
        };
        let mut json = serde_json::to_value(&definition).unwrap();
        json.as_object_mut().unwrap().remove("classes");
        let loaded: ComponentDefinition = serde_json::from_value(json).unwrap();
        assert!(loaded.classes.is_empty());
    }

    #[test]
    fn a_library_file_holds_the_component_flat_and_without_origin() {
        use crate::modules::templates::ComponentTemplate;
        let definition = |key: &str| ComponentDefinition {
            key: key.into(),
            meta: MetaData::default(),
            classes: vec![],
            steps: vec![],
            parameters: vec![],
            origin: None,
        };
        let template = ComponentTemplate {
            schema_version: 1,
            component: definition("SMOOTH"),
            uses: vec![definition("BLUR")],
        };
        let json = serde_json::to_value(&template).unwrap();
        let file = json.as_object().unwrap();
        assert!(!file.contains_key("component"), "no wrapper: {json}");
        assert_eq!(file["key"], "SMOOTH");
        assert!(file.contains_key("meta") && file.contains_key("steps"));
        assert!(!file.contains_key("origin"));
        assert!(!file["uses"][0].as_object().unwrap().contains_key("origin"));

        // Projects load through `serde_json::Value`, library files the same.
        let loaded: ComponentTemplate = serde_json::from_value(json).unwrap();
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.component.key, "SMOOTH");
        assert!(loaded.component.origin.is_none());
        assert_eq!(loaded.uses[0].key, "BLUR");
    }
}
