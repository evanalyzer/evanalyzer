//! # unet
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use macros::CommandsMeta;
use tch::{CModule, Device, Kind, Tensor};

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// Semantic segmentation using a pretrained U-Net exported as TorchScript.
///
/// The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,
/// same normalization as the rest of the pipeline) and return either a
/// `[1, 1, H, W]` tensor of per-pixel foreground probabilities (the model already
/// applies its final sigmoid) or a `[1, C, H, W]` tensor of per-class logits/scores
/// (e.g. background/foreground), in which case a softmax is applied over the
/// channel dimension and the last channel is taken as the foreground probability.
/// Runs on GPU automatically if CUDA is available in the linked libtorch build,
/// otherwise falls back to CPU.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment")]
pub struct UNet {
    /// Path to a TorchScript-exported U-Net model (`torch.jit.script`/`torch.jit.trace`).
    pub model_path: PathBuf,

    /// The class assigned to pixels whose predicted probability reaches
    /// `probability_threshold`. All other pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub object_class_id: SegmentationClass,

    /// Probability above which a pixel is classified as foreground.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,
}

impl ImageAlgorithm for UNet {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(&self.model_path, device).map_err(|e| {
            InternalErrors::Generic(format!(
                "Failed to load U-Net model from {}: {e}",
                self.model_path.display()
            ))
        })?;

        let (input_image, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        let input = Tensor::from_slice(input_image.as_slice())
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([1, 1, height as i64, width as i64]);

        let output = model
            .forward_ts(&[input])
            .map_err(|e| InternalErrors::Generic(format!("U-Net inference failed: {e}")))?;

        let channels = *output.size().get(1).unwrap_or(&1);
        let foreground = if channels > 1 {
            output.softmax(1, Kind::Float).narrow(1, channels - 1, 1)
        } else {
            output
        };

        let probabilities: Vec<f32> = foreground
            .f_to_device(Device::Cpu)
            .and_then(|out| out.f_reshape([(width * height) as i64]))
            .map_err(|e| InternalErrors::Generic(format!("U-Net inference failed: {e}")))
            .and_then(|out| {
                Vec::try_from(&out)
                    .map_err(|e| InternalErrors::Generic(format!("U-Net inference failed: {e}")))
            })?;

        let foreground_class = self.object_class_id.as_u32();
        let output_slice = segmentation_map.as_slice_mut();
        for (out_pixel, &probability) in output_slice.iter_mut().zip(probabilities.iter()) {
            *out_pixel = if probability >= self.probability_threshold {
                foreground_class
            } else {
                SegmentationClass::BACKGROUND.as_u32()
            };
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "UNet"
    }
}
