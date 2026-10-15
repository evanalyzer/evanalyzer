use crate::{
    core_types::ImageAddress, settings::components::ComponentInstance,
    settings::pipeline_command::PipelineCommand, types::ids::PipelineId, types::ids::StepId,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PipelineStepSettings {
    #[serde(default)]
    pub id: StepId,

    /// If enabled, the command is executed, if not enabled, this command is skiped during pipeline execution
    pub enabled: bool,

    /// What the step runs: one command, or a component.
    #[serde(flatten)]
    pub content: StepContent,
}

impl PipelineStepSettings {
    /// The step's command - `None` for a component.
    pub fn command(&self) -> Option<&PipelineCommand> {
        match &self.content {
            StepContent::Command(command) => Some(command),
            StepContent::Component(_) => None,
        }
    }

    /// The step's command, to change - `None` for a component.
    pub fn command_mut(&mut self) -> Option<&mut PipelineCommand> {
        match &mut self.content {
            StepContent::Command(command) => Some(command),
            StepContent::Component(_) => None,
        }
    }
}

/// What a pipeline step runs. In JSON the variant name is the key, next to
/// `id` and `enabled`: `"command": {...}` (as in files written before
/// components existed) or `"component": {...}`.
#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum StepContent {
    Command(PipelineCommand),
    Component(ComponentInstance),
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PipelineSettings {
    /// Unique pipeline ID
    pub id: PipelineId,

    /// Display name of the pipeline
    pub name: String,

    /// Description on the pipeline
    pub description: Option<String>,

    /// This is the image which is used as initial image for this pipeline.
    /// Use Scratchpad if this is just a object manipulation pipeline
    pub image_source: ImageAddress,

    /// Pipelines which are enabled are executed, pipelines, which are disabled are skipt during analysis.
    pub enabled: bool,

    /// The steps which are execute in the given order when the pipeline runs
    pub steps: Vec<PipelineStepSettings>,
}

#[cfg(test)]
mod step_content_tests {
    use super::*;

    #[test]
    fn a_step_written_before_components_existed_still_loads_as_a_command() {
        let json = r#"{"id":3,"enabled":true,"command":{"type":"BLUR","kernelSize":3}}"#;
        // Projects load through `serde_json::Value`.
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let step: PipelineStepSettings = serde_json::from_value(raw).unwrap();
        assert!(matches!(step.command(), Some(PipelineCommand::Blur(_))));
        assert_eq!(step.id, StepId(3));
        let written = serde_json::to_value(&step).unwrap();
        assert!(
            written.get("command").is_some(),
            "still written as `command`"
        );
        assert!(written.get("content").is_none());
    }

    #[test]
    fn a_component_step_loads_and_is_written_as_component() {
        let json = r#"{"enabled":true,"component":{"key":"CELL_RECONSTRUCTION","values":{"cell_class":3}}}"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let step: PipelineStepSettings = serde_json::from_value(raw).unwrap();
        let StepContent::Component(component) = &step.content else {
            panic!("expected a component step");
        };
        assert_eq!(component.key, "CELL_RECONSTRUCTION");
        assert!(step.command().is_none());
        assert_eq!(
            step.id,
            StepId::UNASSIGNED,
            "no id written: assigned on load"
        );
        let written = serde_json::to_value(&step).unwrap();
        assert_eq!(written["component"]["key"], "CELL_RECONSTRUCTION");
    }
}
