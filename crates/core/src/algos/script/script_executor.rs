use crate::algos::ImageAlgorithm;
use macros::CommandsMeta;

/// Runs a user-written Rhai script as a pipeline step.
// TODO: pick the real category/successors once scripts can sit at any stage.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "Preprocessing",
    next = "preprocessing,segment,instance_segmentation,measure,object"
)]
pub struct Script {}

impl ImageAlgorithm for Script {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::GlobalPipelineCache,
    ) -> Result<(), evanalyzer_cfg::core_types::InternalErrors> {
        todo!()
    }

    fn name(&self) -> &'static str {
        todo!()
    }

    fn cite(&self) -> Vec<&'static evanalyzer_cfg::core_types::CitationMetadata> {
        todo!()
    }

    fn execution_scope(&self) -> crate::algos::ExecutionScope {
        todo!()
    }
}
