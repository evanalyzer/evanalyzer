use std::path::PathBuf;

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::{InternalErrors, PixelUnits, SegmentationClass};
use macros::CommandsMeta;

/// Configuration for a single thresholding operation within a multi-threshold stack.
#[derive(CommandsMeta)]
pub struct SegmentationMapping {
    /// Segmentation class from the classifier model
    pub segmentation_class: SegmentationClass,

    /// The classification ID assigned to pixels falling the segmentation class from the model
    pub object_class_id: SegmentationClass,
}

/// A filter that segments an image into discrete classes based on intensity.
///
/// This supports "Multi-Otsu" style behavior by allowing a vector of
/// [`ThresholdSettings`]. Each pixel is evaluated against the settings to
/// determine which `object_class_id` it belongs to.
///
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment")]
pub struct PixelClassifier {
    /// Path to a TorchScript-exported Cellpose model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "evamodel")]
    pub model_path: PathBuf,

    /// Segmentation mapping list.
    ///
    /// Maps the segmentation class from the pixel classifier output to a segmentation class of the project
    pub segmentation_mapping: Vec<SegmentationMapping>,
}

impl ImageAlgorithm for PixelClassifier {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "PixelClassifier".into()
    }
}
