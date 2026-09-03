//! # unet
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use macros::CommandsMeta;
use tch::{CModule, Device, Kind, Tensor};

use crate::{
    algos::{ExecutionScope, ImageAlgorithm, ai_segmentation::model_cache::load_cached_model},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// How a multi-channel U-Net output should be turned into a single foreground
/// probability map. Ignored for single-channel outputs, which are always
/// treated as already-activated foreground probabilities.
pub enum UNetOutputMode {
    /// Channels are mutually-exclusive class scores (e.g. a background/foreground
    /// classification head). `softmax` is applied over the channel dimension
    /// first, then the channel at `foreground_channel` is taken as the
    /// foreground probability.
    SoftmaxClasses,
    /// Channels are independent, already-activated probability maps — e.g. a
    /// foreground-mask channel plus a separate boundary channel, as produced by
    /// boundary-aware models (mask + boundary heads are *not* mutually
    /// exclusive, so they must never be put through a softmax together). The
    /// channel at `foreground_channel` is used directly.
    IndependentChannels,
}

/// Semantic segmentation using a pretrained U-Net exported as TorchScript.
///
/// The model is expected to accept a `[1, 1, H, W]` float tensor (single-channel,
/// same normalization as the rest of the pipeline) and return either a
/// `[1, 1, H, W]` tensor of per-pixel foreground probabilities (the model already
/// applies its final sigmoid) or a `[1, C, H, W]` tensor with more than one
/// channel, in which case `output_mode` and `foreground_channel` decide how the
/// foreground probability is extracted (see [`UNetOutputMode`]). Runs on GPU
/// automatically if CUDA is available in the linked libtorch build, otherwise
/// falls back to CPU.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", display_name = "AI UNet Segmentation")]
pub struct UNet {
    /// Path to a TorchScript-exported U-Net model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels whose predicted probability reaches
    /// `probability_threshold`. All other pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub object_class_id: SegmentationClass,

    /// Probability above which a pixel is classified as foreground.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,

    /// How to interpret the model output when it has more than one channel.
    /// Ignored for single-channel outputs.
    #[cmdsmeta(default = UNetOutputMode::SoftmaxClasses)]
    pub output_mode: UNetOutputMode,

    /// Index of the channel holding the foreground probability, used only
    /// when the model output has more than one channel. Out-of-range values
    /// are clamped to the last available channel.
    ///
    /// * For `SoftmaxClasses`, this is typically the last channel (e.g. `1`
    ///   for a 2-class background/foreground head).
    /// * For `IndependentChannels`, this is whichever channel the model
    ///   dedicates to the foreground mask — commonly `0` for boundary-aware
    ///   models, which conventionally output mask before boundary.
    #[cmdsmeta(default = 1, min = 0, max = 16, step = 1)]
    pub foreground_channel: i32,

    /// Index of an optional **boundary** channel for boundary-aware models
    /// (e.g. bioimage.io's `affable-shark` / NucleiSegmentationBoundaryModel,
    /// which outputs mask in channel 0 and boundary in channel 1). Set to `-1`
    /// to disable.
    ///
    /// When enabled, a pixel is classified as foreground only where the
    /// foreground probability reaches `probability_threshold` **and** the
    /// boundary probability stays below `boundary_threshold`. This carves the
    /// predicted boundaries out as thin gaps, so a following `ConnectedComponents`
    /// separates touching objects directly — which is the whole point of a
    /// boundary model and the only way to split nuclei a plain mask merges.
    #[cmdsmeta(default = -1, min = -1, max = 16, step = 1)]
    pub boundary_channel: i32,

    /// Boundary probability at or above which a pixel is treated as an object
    /// boundary and excluded from the foreground. Only used when
    /// `boundary_channel` is enabled (>= 0). Lower values cut wider gaps
    /// (separate more aggressively); higher values cut thinner gaps.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub boundary_threshold: f32,
}

impl ImageAlgorithm for UNet {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let device = Device::cuda_if_available();
        let model = load_cached_model(&self.model_path, || {
            CModule::load_on_device(&self.model_path, device)
        })
        .map_err(|e| {
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
            let idx = (self.foreground_channel as i64).clamp(0, channels - 1);
            match self.output_mode {
                UNetOutputMode::SoftmaxClasses => output.softmax(1, Kind::Float).narrow(1, idx, 1),
                UNetOutputMode::IndependentChannels => output.narrow(1, idx, 1),
            }
        } else {
            output.shallow_clone()
        };

        let probabilities = Self::channel_to_vec(&foreground, width, height)?;

        // Optional boundary channel: boundary-aware models (e.g. affable-shark)
        // predict an independent boundary map. Subtracting it from the foreground
        // opens thin gaps between touching objects so they separate downstream.
        let boundary: Option<Vec<f32>> = if self.boundary_channel >= 0
            && channels > 1
            && (self.boundary_channel as i64) < channels
        {
            let b = output.narrow(1, self.boundary_channel as i64, 1);
            Some(Self::channel_to_vec(&b, width, height)?)
        } else {
            None
        };

        let foreground_class = self.object_class_id.as_u32();
        let output_slice = segmentation_map.as_slice_mut();
        output_slice.copy_from_slice(&Self::classify_pixels(
            &probabilities,
            boundary.as_deref(),
            self.probability_threshold,
            self.boundary_threshold,
            foreground_class,
        ));

        Ok(())
    }

    fn name(&self) -> &'static str {
        "UNet"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "ronneberger2015unet",
            title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            authors: &["Olaf Ronneberger", "Philipp Fischer", "Thomas Brox"],
            year: 2015,
            container: Some("Medical Image Computing and Computer-Assisted Intervention (MICCAI)"),
            doi: Some("10.1007/978-3-319-24574-4_28"),
            url: Some("https://doi.org/10.1007/978-3-319-24574-4_28"),
            pages: Some("234-241"),
        })
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl UNet {
    /// Moves a single-channel `[1, 1, H, W]` tensor to the CPU and flattens it
    /// into a `width * height` vector.
    fn channel_to_vec(
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, InternalErrors> {
        tensor
            .f_to_device(Device::Cpu)
            .and_then(|out| out.f_reshape([(width * height) as i64]))
            .map_err(|e| InternalErrors::Generic(format!("U-Net inference failed: {e}")))
            .and_then(|out| {
                Vec::try_from(&out)
                    .map_err(|e| InternalErrors::Generic(format!("U-Net inference failed: {e}")))
            })
    }

    /// Classifies each pixel as foreground/background from its probability
    /// and, for boundary-aware models, its boundary probability: foreground
    /// when the probability reaches `probability_threshold` and - if a
    /// boundary map is given - the boundary probability stays below
    /// `boundary_threshold`. This is the "carve thin gaps between touching
    /// objects" behaviour the `boundary_channel` doc comment above describes;
    /// it is the whole reason a boundary-aware model separates touching
    /// nuclei where a plain mask-only model can't.
    fn classify_pixels(
        probabilities: &[f32],
        boundary: Option<&[f32]>,
        probability_threshold: f32,
        boundary_threshold: f32,
        foreground_class: u32,
    ) -> Vec<u32> {
        probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let is_foreground = p >= probability_threshold
                    && boundary.map(|b| b[i] < boundary_threshold).unwrap_or(true);
                if is_foreground {
                    foreground_class
                } else {
                    SegmentationClass::BACKGROUND.as_u32()
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::ai_segmentation::test_support::trace_and_save_model;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    const FG: u32 = 7;
    fn bg() -> u32 {
        SegmentationClass::BACKGROUND.as_u32()
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, values, CpuAllocator)
                .unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    fn unet(model_path: PathBuf) -> UNet {
        UNet {
            model_path,
            object_class_id: SegmentationClass(FG),
            probability_threshold: 0.5,
            output_mode: UNetOutputMode::SoftmaxClasses,
            foreground_channel: 1,
            boundary_channel: -1,
            boundary_threshold: 0.5,
        }
    }

    // ---- execute() - real TorchScript load + inference, see `test_support` ----

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = unet(dir.path().join("missing.pt"));
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
    }

    #[test]
    fn execute_single_channel_output_is_used_directly_as_the_foreground_probability() {
        // A single-channel model output is documented as already-activated
        // foreground probabilities (the model applies its own sigmoid), used
        // directly with no softmax/channel-selection - so an identity model
        // (output == input) makes the input pixel values themselves the
        // probabilities under test.
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| x.shallow_clone());
        let cmd = UNet {
            output_mode: UNetOutputMode::SoftmaxClasses, // irrelevant for 1 channel
            ..unet(model_path)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.1, 0.9]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[bg(), FG]);
    }

    #[test]
    fn execute_softmax_classes_selects_the_foreground_channel_after_softmax() {
        // Two-channel logits built from the single input channel: a large
        // negative/positive spread so softmax saturates close to 0/1,
        // cleanly on either side of the default 0.5 threshold regardless of
        // float rounding.
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| {
            let bg_logit = x * -10.0;
            let fg_logit = x * 10.0;
            Tensor::cat(&[bg_logit, fg_logit], 1)
        });
        let cmd = UNet {
            output_mode: UNetOutputMode::SoftmaxClasses,
            foreground_channel: 1,
            ..unet(model_path)
        };
        // x = -1 -> fg softmax ~0 (background); x = 1 -> fg softmax ~1 (foreground).
        let mut ctx = gray_ctx(2, 1, vec![-1.0, 1.0]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[bg(), FG]);
    }

    #[test]
    fn execute_independent_channels_uses_the_selected_channel_without_softmax() {
        // Channel 0 carries the raw input value directly (no softmax), so an
        // already-in-[0,1] probability passed as input round-trips exactly -
        // unlike `SoftmaxClasses`, which would normalize a 1-channel-derived
        // pair of logits instead.
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| {
            let junk = x.zeros_like();
            Tensor::cat(&[x.shallow_clone(), junk], 1)
        });
        let cmd = UNet {
            output_mode: UNetOutputMode::IndependentChannels,
            foreground_channel: 0,
            ..unet(model_path)
        };
        let mut ctx = gray_ctx(2, 1, vec![0.1, 0.9]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[bg(), FG]);
    }

    #[test]
    fn execute_boundary_channel_carves_out_pixels_whose_boundary_probability_is_high() {
        // Channel 0 (foreground) saturates to 1.0 for any x >= 1 via clamp,
        // so both test pixels have an equally "definite" foreground signal;
        // channel 1 (boundary) keeps varying past that saturation point
        // (sigmoid(x - 5)), so the two pixels differ only in boundary
        // probability - isolating the boundary-carving behavior from the
        // foreground-threshold behavior already covered by
        // `classify_pixels_boundary_at_or_above_threshold_excludes_the_pixel` above.
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| {
            let fg = x.clamp(0.0, 1.0);
            let boundary = (x - 5.0).sigmoid();
            Tensor::cat(&[fg, boundary], 1)
        });
        let cmd = UNet {
            output_mode: UNetOutputMode::IndependentChannels,
            foreground_channel: 0,
            boundary_channel: 1,
            boundary_threshold: 0.5,
            ..unet(model_path)
        };
        // x=1: fg saturates to 1.0, boundary = sigmoid(-4) ~0 (open) -> foreground.
        // x=20: fg saturates to 1.0, boundary = sigmoid(15) ~1 (closed) -> background.
        let mut ctx = gray_ctx(2, 1, vec![1.0, 20.0]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(
            seg.as_slice(),
            &[FG, bg()],
            "equal foreground signal, but only the low-boundary pixel should survive"
        );
    }

    #[test]
    fn execute_foreground_channel_out_of_range_is_clamped_to_the_last_channel() {
        // 2-channel output, but foreground_channel points past the end - must
        // clamp to the last valid channel (index 1) instead of panicking.
        let (_dir, model_path) = trace_and_save_model(1, 1, 2, |x| {
            let bg_logit = x * -10.0;
            let fg_logit = x * 10.0;
            Tensor::cat(&[bg_logit, fg_logit], 1)
        });
        let cmd = UNet {
            output_mode: UNetOutputMode::SoftmaxClasses,
            foreground_channel: 99,
            ..unet(model_path)
        };
        let mut ctx = gray_ctx(1, 1, vec![1.0]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[FG]);
    }

    #[test]
    fn classify_pixels_without_a_boundary_map_uses_probability_alone() {
        let probs = [0.1, 0.5, 0.9];
        let result = UNet::classify_pixels(&probs, None, 0.5, 0.5, FG);
        assert_eq!(result, vec![bg(), FG, FG]);
    }

    #[test]
    fn classify_pixels_threshold_comparison_is_inclusive() {
        // Exactly at the probability threshold must count as foreground (`>=`).
        let probs = [0.5];
        let result = UNet::classify_pixels(&probs, None, 0.5, 0.5, FG);
        assert_eq!(result, vec![FG]);
    }

    #[test]
    fn classify_pixels_boundary_at_or_above_threshold_excludes_the_pixel() {
        // High foreground probability, but the boundary probability is
        // exactly at the boundary threshold - not *below* it, so it must be
        // excluded (this is the exact carving behaviour boundary-aware
        // models rely on to separate touching nuclei).
        let probs = [0.9];
        let boundary = [0.5];
        let result = UNet::classify_pixels(&probs, Some(&boundary), 0.5, 0.5, FG);
        assert_eq!(result, vec![bg()]);
    }

    #[test]
    fn classify_pixels_boundary_below_threshold_keeps_the_pixel_as_foreground() {
        let probs = [0.9];
        let boundary = [0.1];
        let result = UNet::classify_pixels(&probs, Some(&boundary), 0.5, 0.5, FG);
        assert_eq!(result, vec![FG]);
    }

    #[test]
    fn classify_pixels_low_probability_is_background_regardless_of_boundary() {
        let probs = [0.1];
        let boundary = [0.0]; // well below the boundary threshold
        let result = UNet::classify_pixels(&probs, Some(&boundary), 0.5, 0.5, FG);
        assert_eq!(
            result,
            vec![bg()],
            "a low foreground probability must stay background even with an open boundary"
        );
    }

    #[test]
    fn classify_pixels_handles_a_mixed_row_pixel_by_pixel() {
        let probs = [0.9, 0.9, 0.2, 0.9];
        let boundary = [0.1, 0.9, 0.1, 0.5];
        // px0: fg prob high, boundary open -> foreground
        // px1: fg prob high, boundary closed (>= threshold) -> background
        // px2: fg prob low -> background regardless of boundary
        // px3: fg prob high, boundary exactly at threshold (not < ) -> background
        let result = UNet::classify_pixels(&probs, Some(&boundary), 0.5, 0.5, FG);
        assert_eq!(result, vec![FG, bg(), bg(), bg()]);
    }
}
