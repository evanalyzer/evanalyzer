use crate::{algos::ImageAlgorithm, dataset::DatasetAlgorithm};
use evanalyzer_cfg::{core_types::InternalErrors, settings::pipeline_command::PipelineCommand};

/// A configured pipeline step, ready to run - an image step (runs per tile
/// or per image, see `ExecutionScope`) or a dataset step (runs once after
/// every image has been analysed, on the objects of all of them).
pub enum PipelineStep {
    Image(Box<dyn ImageAlgorithm>),
    Dataset(Box<dyn DatasetAlgorithm>),
}

/// The runnable step for `cmd`.
///
/// Every command is an image step for now. Once the first dataset command
/// exists, the generator that writes `algos_from_config.rs` also has to
/// scan `dataset/` and route those commands to [`PipelineStep::Dataset`].
pub fn into_pipeline_step(cmd: PipelineCommand) -> Result<PipelineStep, InternalErrors> {
    super::algos_from_config::into_algorithm(cmd).map(PipelineStep::Image)
}
