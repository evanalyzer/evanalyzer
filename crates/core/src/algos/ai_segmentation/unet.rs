use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::{InternalErrors, PixelUnits, SegmentationClass};
use macros::CommandsMeta;

#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment")]
struct Unet {}

impl ImageAlgorithm for Unet {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        //ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Unet"
    }
}
