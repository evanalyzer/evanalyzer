//! # unet
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use std::path::PathBuf;

use evanalyzer_ai_engine::{Image, Request, Response, Task};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use macros::CommandsMeta;

use crate::{
    ai_devices,
    algos::{ExecutionScope, ImageAlgorithm},
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
///
/// Any tile size works: the tile is mirror-padded to a multiple of 16 (U-Nets
/// halve the image 4 times and fail otherwise) plus a 16 px border (context
/// for the pixels at the tile edge, the "halo" bioimage.io models declare),
/// and the prediction is cropped back to the tile.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", display_name = "AI UNet Segmentation")]
pub struct UNet {
    /// Path to a TorchScript-exported U-Net model (`torch.jit.script`/`torch.jit.trace`).
    #[cmdsmeta(file_extensions = "pt,pth")]
    pub model_path: PathBuf,

    /// The class assigned to pixels whose predicted probability reaches
    /// `probability_threshold`. All other pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub segmentation_class: SegmentationClass,

    /// Probability above which a pixel is classified as foreground.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,

    /// How to interpret the model output when it has more than one channel.
    /// Ignored for single-channel outputs.
    #[cmdsmeta(default = UNetOutputMode::SoftmaxClasses, visibility = Advanced)]
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
    #[cmdsmeta(default = 1, min = 0, max = 16, step = 1, visibility = Advanced)]
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
    #[cmdsmeta(default = -1, min = -1, max = 16, step = 1, visibility = Advanced)]
    pub boundary_channel: i32,

    /// Boundary probability at or above which a pixel is treated as an object
    /// boundary and excluded from the foreground. Only used when
    /// `boundary_channel` is enabled (>= 0). Lower values cut wider gaps
    /// (separate more aggressively); higher values cut thinner gaps.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01, visibility = Advanced)]
    pub boundary_threshold: f32,
}

impl ImageAlgorithm for UNet {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (input_image, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        let request = Request {
            model: self.model_path.clone(),
            task: Task::UNet {
                foreground_channel: self.foreground_channel,
                softmax: matches!(self.output_mode, UNetOutputMode::SoftmaxClasses),
                boundary_channel: self.boundary_channel,
            },
        };
        let image = Image {
            width,
            height,
            channels: 1,
            data: input_image.as_slice(),
        };
        let Response::UNet {
            probabilities,
            boundary,
        } = ai_devices::infer(&request, image)?
        else {
            return Err(InternalErrors::Generic(
                "U-Net: the AI engine gave another kind of answer".into(),
            ));
        };

        let foreground_class = self.segmentation_class.as_u32();
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

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "ronneberger2015unet",
            title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            authors: &["Olaf Ronneberger", "Philipp Fischer", "Thomas Brox"],
            year: 2015,
            container: Some("Medical Image Computing and Computer-Assisted Intervention (MICCAI)"),
            doi: Some("10.1007/978-3-319-24574-4_28"),
            url: Some("https://doi.org/10.1007/978-3-319-24574-4_28"),
            pages: Some("234-241"),
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl UNet {
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
    #[cfg(feature = "libtorch-in-process")]
    use evanalyzer_ai_engine::torch::test_support::{tch::Tensor, trace_and_save_model};
    use kornia_image::{Image, ImageSize};

    const FG: u32 = 7;
    fn bg() -> u32 {
        SegmentationClass::BACKGROUND.as_u32()
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    fn unet(model_path: PathBuf) -> UNet {
        UNet {
            model_path,
            segmentation_class: SegmentationClass(FG),
            probability_threshold: 0.5,
            output_mode: UNetOutputMode::SoftmaxClasses,
            foreground_channel: 1,
            boundary_channel: -1,
            boundary_threshold: 0.5,
        }
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_crops_the_prediction_back_onto_an_odd_sized_tile() {
        // An identity model on a 37x23 tile: the padded run must be cropped
        // back exactly - every pixel, edges included, keeps its own value.
        let (_dir, model_path) = trace_and_save_model(1, 2, 2, |x| x * 1.0);
        let cmd = unet(model_path);
        let (w, h) = (37usize, 23usize);
        let values: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let on_edge = x == 0 || y == 0 || x == w - 1 || y == h - 1;
                if on_edge || (x * 7 + y * 3) % 5 == 0 {
                    0.9
                } else {
                    0.1
                }
            })
            .collect();
        let expected: Vec<u32> = values
            .iter()
            .map(|&v| if v >= 0.5 { FG } else { bg() })
            .collect();
        let mut ctx = gray_ctx(w, h, values);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert_eq!(
            ctx.get_segmentation_map().unwrap().as_slice(),
            expected.as_slice()
        );
    }

    #[cfg(feature = "libtorch-in-process")]
    #[test]
    fn execute_handles_tiles_smaller_than_the_padding() {
        // Too small to mirror 16 px: the edge pixels are repeated instead.
        let (_dir, model_path) = trace_and_save_model(1, 2, 2, |x| x * 1.0);
        let cmd = unet(model_path);
        let mut ctx = gray_ctx(3, 1, vec![0.9, 0.1, 0.9]);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert_eq!(
            ctx.get_segmentation_map().unwrap().as_slice(),
            &[FG, bg(), FG]
        );
    }

    // ---- real bioimage.io model ----

    /// Reads a little-endian float32 `.npy` file (the bioimage.io test tensors).
    fn read_npy_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e} - see the test's doc comment", path.display()));
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        bytes[10 + header_len..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[cfg(feature = "libtorch-in-process")]
    /// The bioimage.io `joyful-pig` boundary U-Net (LiveCellSegmentationBoundaryModel)
    /// on its own 512x512 test input, compared with its reference output -
    /// at the full size and on an odd 250x250 cutout, which crashed in the
    /// model before tiles were padded to a multiple of 16. Not checked into
    /// git (~116 MB): download it from bioimage.io into `tests/joyful-pig/`.
    /// Run with:
    ///   cargo test -p evanalyzer_core --features libtorch-in-process --lib real_joyful_pig -- --ignored
    #[test]
    #[ignore]
    fn real_joyful_pig_matches_its_reference_at_any_tile_size() {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/joyful-pig"
        ));
        let raw = read_npy_f32(&dir.join("test_input_0.npy"));
        let reference = read_npy_f32(&dir.join("test_output_0.npy"));
        let n = 512usize;
        // The model's declared preprocessing: zero mean, unit variance.
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        let std = (raw.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / raw.len() as f32).sqrt();
        let input: Vec<f32> = raw.iter().map(|v| (v - mean) / (std + 1e-6)).collect();
        let cmd = UNet {
            output_mode: UNetOutputMode::IndependentChannels,
            foreground_channel: 0,
            boundary_channel: 1,
            ..unet(dir.join("weights-torchscript.pt"))
        };
        let expected =
            UNet::classify_pixels(&reference[..n * n], Some(&reference[n * n..]), 0.5, 0.5, FG);

        // Share of pixels of a `w x h` region at (x0, y0), at least 16 px away
        // from the region's edges, that differ from the reference.
        let mismatch = |seg: &[u32], x0: usize, y0: usize, w: usize, h: usize| -> f64 {
            let (mut diff, mut total) = (0, 0);
            for y in 16..h - 16 {
                for x in 16..w - 16 {
                    total += 1;
                    if seg[y * w + x] != expected[(y0 + y) * n + x0 + x] {
                        diff += 1;
                    }
                }
            }
            diff as f64 / total as f64
        };

        let mut ctx = gray_ctx(n, n, input.clone());
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        let full = mismatch(ctx.get_segmentation_map().unwrap().as_slice(), 0, 0, n, n);

        let (x0, y0, w, h) = (100usize, 130usize, 250usize, 250usize);
        let cutout: Vec<f32> = (0..h)
            .flat_map(|y| input[(y0 + y) * n + x0..(y0 + y) * n + x0 + w].to_vec())
            .collect();
        let mut ctx = gray_ctx(w, h, cutout);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .expect("an odd-sized tile must run (it is padded to a multiple of 16)");
        let odd = mismatch(ctx.get_segmentation_map().unwrap().as_slice(), x0, y0, w, h);
        // Not exact: the model normalizes each input it gets (InstanceNorm),
        // so the mirrored border - and, for the cutout, seeing less of the
        // image - shifts its statistics a little and flips pixels near the
        // 0.5 threshold (measured: 0.5 % and 2.5 %).
        assert!(
            full < 0.01,
            "{:.2} % of the pixels differ at 512x512",
            full * 100.0
        );
        assert!(
            odd < 0.05,
            "{:.2} % of the pixels differ at 250x250",
            odd * 100.0
        );
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

    #[cfg(feature = "libtorch-in-process")]
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

    #[cfg(feature = "libtorch-in-process")]
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

    #[cfg(feature = "libtorch-in-process")]
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

    #[cfg(feature = "libtorch-in-process")]
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

    #[cfg(feature = "libtorch-in-process")]
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
