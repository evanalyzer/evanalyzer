//! # unet
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::PathBuf;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use evanalyzer_nn::Image;
use evanalyzer_nn::model::unet::{UNet as UNetModel, UNetSettings};
use macros::CommandsMeta;

use super::PieceSize;

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

/// Semantic segmentation with a pretrained U-Net (torch-em `UNet2d`, the
/// architecture of bioimage.io's U-Nets), from an EVAnalyzer model file
/// (`.safetensors`, converted once from the original - `crates/nn`).
///
/// The model gives one or more channels per pixel: with one, its
/// foreground probability; with more, `output_mode` and `foreground_channel`
/// decide how the foreground probability is taken. The model file says
/// whether the image is normalized first (zero mean, unit variance). Runs on
/// a GPU when there is one (convolutions in half precision there), else on
/// the CPU.
///
/// Any tile size works: the tile is mirror-padded to the network's multiple
/// (16 for depth 4) plus a 16 px border (context for the pixels at the tile
/// edge, the "halo" bioimage.io models declare), and the prediction is
/// cropped back to the tile.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", display_name = "AI UNet Segmentation")]
pub struct UNet {
    /// The U-Net model file (EVAnalyzer's `.safetensors`).
    #[cmdsmeta(file_extensions = "safetensors")]
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

    /// How large the pieces are that a large tile runs through the network
    /// in: smaller ones take less RAM, larger ones are faster.
    #[cmdsmeta(default = PieceSize::Auto, optional = true, visibility = Advanced)]
    pub piece_size: PieceSize,
}

/// What the network takes per pixel of the image it gets at once (see
/// `shared_ram_bytes`).
pub(crate) const NETWORK_BYTES_PER_PIXEL: f64 = 1650.0;

impl UNet {
    /// The pieces a `width x height` tile runs in (see [`PieceSize::pieces`]).
    fn pieces(&self, width: usize, height: usize) -> evanalyzer_nn::Pieces {
        self.piece_size
            .pieces(NETWORK_BYTES_PER_PIXEL, width, height)
    }
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

        let settings = UNetSettings {
            foreground_channel: self.foreground_channel.max(0) as usize,
            softmax: matches!(self.output_mode, UNetOutputMode::SoftmaxClasses),
            boundary_channel: usize::try_from(self.boundary_channel).ok(),
            pieces: self.pieces(width, height),
        };
        let image = Image {
            width,
            height,
            channels: 1,
            data: input_image.as_slice(),
        };
        let output = ai_devices::infer::<UNetModel>(&self.model_path, image, &settings)?;
        let (probabilities, boundary) = (output.probabilities, output.boundary);

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

    /// What the step keeps of the answer per worker: the foreground and
    /// boundary probabilities (8 bytes per pixel, `tests::tile_ram`).
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    /// One inference at a time on the CPU (the AI engine's one slot): the
    /// network's activations on the largest image it gets at once - the
    /// tile, or a piece of it (`piece_size`, [`evanalyzer_nn::pieces::largest_model_input`]).
    /// 1.65 KB per pixel: measured 1.61 KB for bioimage.io's
    /// `romantic-shark` (depth 4, 64 filters, InstanceNorm) on burn's CPU
    /// backend (`tests::tile_ram::measure_ai_tile_commands`, 2026-11-25;
    /// libtorch: 1.55 KB for `joyful-pig`, 0.67 / 1.3 KB for U-Nets of depth
    /// 4 from 32 / 64 filters - BENCHMARK.md "AI models: RAM per inference")
    /// - [`NETWORK_BYTES_PER_PIXEL`].
    /// On a GPU they are in its memory - planned anyway (the computer may
    /// run on its CPU). Plus the model's weights (one copy).
    fn shared_ram_bytes(&self, tile_width: usize, tile_height: usize) -> u64 {
        // Before the run it isn't known which of the two the model gets.
        let piece = self.pieces(tile_width, tile_height).largest();
        let pixels = evanalyzer_nn::pieces::largest_model_input(tile_width, piece)
            * evanalyzer_nn::pieces::largest_model_input(tile_height, piece);
        (NETWORK_BYTES_PER_PIXEL * pixels as f64) as u64 + super::model_bytes(&self.model_path)
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
    use kornia_image::ImageSize;

    const FG: u32 = 7;
    fn bg() -> u32 {
        SegmentationClass::BACKGROUND.as_u32()
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = kornia_image::Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
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
            piece_size: PieceSize::Auto,
        }
    }

    /// A set piece size is the same for every model and every computer;
    /// Auto gives a model seeing a limited neighbourhood what the RAM
    /// allows, one seeing the whole image 1024 - and the RAM plan takes the
    /// larger (the kind of model is known only in the run).
    #[test]
    fn a_set_piece_size_is_fixed_auto_follows_the_ram_where_it_may() {
        use evanalyzer_nn::Pieces;
        use evanalyzer_nn::pieces::{WHOLE_IMAGE_PIECE_SIDE, piece_side_for};
        let with = |piece_size| UNet {
            piece_size,
            ..unet(PathBuf::new())
        };
        let fixed = |side| PieceSize::Fixed { side };
        assert_eq!(with(fixed(512)).pieces(4096, 4096), Pieces::fixed(512));
        assert_eq!(with(fixed(100)).pieces(4096, 4096), Pieces::fixed(256));
        assert_eq!(with(fixed(99_999)).pieces(4096, 4096), Pieces::fixed(4096));
        let ram = crate::memory::SystemRam::now().total;
        let auto = with(PieceSize::Auto).pieces(4096, 4096);
        assert_eq!(
            auto.local_side,
            piece_side_for(NETWORK_BYTES_PER_PIXEL, ram, 4096, 4096)
        );
        assert_eq!(auto.whole_image_side, WHOLE_IMAGE_PIECE_SIDE);
        let per_pixel = NETWORK_BYTES_PER_PIXEL as u64;
        let piece = |p: u64| per_pixel * (p + 256).min(4096).pow(2);
        assert_eq!(with(fixed(1024)).shared_ram_bytes(4096, 4096), piece(1024));
        assert_eq!(with(fixed(2048)).shared_ram_bytes(4096, 4096), piece(2048));
        assert_eq!(
            with(fixed(4096)).shared_ram_bytes(4096, 4096),
            per_pixel * 4096 * 4096
        );
        assert_eq!(
            with(PieceSize::Auto).shared_ram_bytes(4096, 4096),
            piece(auto.largest() as u64)
        );
    }

    // ---- execute(): a real model file (a tiny random U-Net, nn's
    // `test_support`) through the engine ----

    /// The step gives what the model gives, thresholded: the probabilities
    /// of the model run directly, through `classify_pixels`.
    #[test]
    fn execute_runs_the_model_file_and_thresholds_its_probabilities() {
        use evanalyzer_nn::Model;
        use evanalyzer_nn::model::unet::Preprocessing;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unet.safetensors");
        let config = evanalyzer_nn::test_support::tiny_unet_config();
        std::fs::write(
            &path,
            evanalyzer_nn::test_support::unet_file(config, Preprocessing::None),
        )
        .unwrap();
        let (w, h) = (37, 23);
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0)
            .collect();
        let cmd = UNet {
            output_mode: UNetOutputMode::IndependentChannels,
            foreground_channel: 0,
            ..unet(path.clone())
        };
        let mut ctx = gray_ctx(w, h, pixels.clone());
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let model = UNetModel::load(&path, &evanalyzer_nn::Device::flex(), false).unwrap();
        let settings = UNetSettings {
            foreground_channel: 0,
            softmax: false,
            boundary_channel: None,
            pieces: cmd.pieces(w, h),
        };
        let image = Image {
            width: w,
            height: h,
            channels: 1,
            data: &pixels,
        };
        let direct = model.infer(image, &settings).unwrap();
        let expected = UNet::classify_pixels(&direct.probabilities, None, 0.5, 0.5, FG);
        // Up to float rounding: the engine runs on its own threads (another
        // summation order; this tiny random network's InstanceNorm on little
        // variance turns it into ~1e-4) - pixels at the threshold may land
        // either side.
        let ours = ctx.get_segmentation_map().unwrap().as_slice();
        for ((a, b), p) in ours.iter().zip(&expected).zip(&direct.probabilities) {
            assert!(
                a == b || (p - 0.5).abs() < 1e-3,
                "probability {p}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = unet(dir.path().join("missing.safetensors"));
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
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
