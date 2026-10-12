use crate::{
    core_types::ImageAddress, settings::pipeline_command::PipelineCommand,
    types::classes::ObjectClass, types::ids::PipelineId,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PipelineStepSettings {
    /// If enabled, the command is executed, if not enabled, this command is skiped during pipeline execution
    pub enabled: bool,

    /// The pipeline command to execute in this step
    pub command: PipelineCommand,
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

impl PipelineSettings {
    /// The object classes of every enabled step, each once - used by the
    /// GUI's pipeline focus mode to show only the selected pipeline's
    /// classes. See the generated [`PipelineCommand::object_classes`].
    pub fn object_classes(&self) -> BTreeSet<ObjectClass> {
        self.steps
            .iter()
            .filter(|step| step.enabled)
            .flat_map(|step| step.command.object_classes())
            .collect()
    }
}
