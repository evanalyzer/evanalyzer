//! # stardist
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::PathBuf;

use super::{InputNormalization, PieceSize};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use evanalyzer_nn::model::stardist::{StarDist as StarDistModel, StarDistParams};
use evanalyzer_nn::{Image, Model};
use macros::CommandsMeta;

use crate::{
    ai_devices,
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Instance segmentation with a pretrained StarDist 2D model, from an
/// EVAnalyzer model file (`.evanet`, converted once from StarDist's own,
/// e.g. `2D_versatile_fluo` - `crates/nn`).
///
/// For each grid cell the model gives the probability of an object's centre
/// and the distances to its boundary along `n_rays` equally spaced angles
/// (StarDist's star-convex polygons). On a grid coarser than the pixels the
/// polygons are scaled back to the image.
///
/// Per grid cell candidates above `probability_threshold` are converted to
/// star-convex polygons, then greedily filtered with non-maximum suppression
/// (polygons whose pixel-overlap ratio with a higher-scoring candidate exceeds
/// `nms_threshold` are discarded) before being rasterized into the pipeline's
/// segmentation and instance maps. Runs on a GPU when there is one
/// (convolutions in half precision there), else on the CPU.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "segment",
    next = "extract",
    display_name = "AI Stardist Segmentation"
)]
pub struct Stardist {
    /// The StarDist model file (EVAnalyzer's `.evanet`).
    #[cmdsmeta(file_extensions = NN_MODEL_EXTENSION)]
    pub model_path: PathBuf,

    /// The class assigned to pixels of every detected object. All other
    /// pixels are assigned `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub segmentation_class: SegmentationClass,

    /// Probability above which a grid cell is considered a candidate object center.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01)]
    pub probability_threshold: f32,

    /// Pixel-overlap ratio (intersection / union) above which a lower-scoring
    /// candidate polygon is suppressed in favor of an overlapping higher-scoring one.
    #[cmdsmeta(default = 0.3, min = 0.0, max = 1.0, step = 0.01, visibility = Advanced)]
    pub nms_threshold: f32,

    /// How large the pieces are that a large tile runs through the network
    /// in: smaller ones take less RAM, larger ones are faster.
    #[cmdsmeta(default = PieceSize::Auto, optional = true, visibility = Advanced)]
    pub piece_size: PieceSize,

    /// How the image is scaled before the model sees it: by default the
    /// tile's 1st and 99.8th percentile become 0 and 1, as StarDist's
    /// models were trained.
    #[cmdsmeta(default = InputNormalization::ModelDefault, optional = true, visibility = Advanced)]
    pub normalization: InputNormalization,
}

/// What the network takes per pixel of the image it gets at once (see
/// `shared_ram_bytes`) - nn's measured figure.
const NETWORK_BYTES_PER_PIXEL: f64 = <StarDistModel as Model>::NET_BYTES_PER_PIXEL;

impl Stardist {
    /// The pieces a `width x height` tile runs in (see [`PieceSize::pieces`]).
    fn pieces(&self, width: usize, height: usize) -> evanalyzer_nn::Pieces {
        self.piece_size
            .pieces(NETWORK_BYTES_PER_PIXEL, width, height)
    }
}

impl ImageAlgorithm for Stardist {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        let params = StarDistParams {
            normalization: self.normalization.to_nn(),
            pieces: self.pieces(width, height),
            probability_threshold: self.probability_threshold,
            nms_threshold: self.nms_threshold,
        };
        let image = Image {
            width,
            height,
            channels: 1,
            data: input_image.as_slice(),
        };
        ai_devices::run::<StarDistModel>(
            &self.model_path,
            image,
            &params,
            instance_map.as_slice_mut(),
        )?;
        super::fill_class(
            instance_map.as_slice(),
            segmentation_map.as_slice_mut(),
            self.segmentation_class,
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Stardist"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "schmidt2018stardist",
            title: "Cell Detection with Star-Convex Polygons",
            authors: &[
                "Uwe Schmidt",
                "Martin Weigert",
                "Coleman Broaddus",
                "Gene Myers",
            ],
            year: 2018,
            container: Some("Medical Image Computing and Computer-Assisted Intervention (MICCAI)"),
            doi: Some("10.1007/978-3-030-00934-2_30"),
            url: Some("https://doi.org/10.1007/978-3-030-00934-2_30"),
            pages: Some("265-273"),
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// What the step keeps per worker: the answer (probability and 32 rays
    /// per grid cell) and the candidates' polygons and bit masks - ~100
    /// bytes per pixel on cells of 24 px (`tests::tile_ram`, a well-fitting
    /// model's outputs); more candidates (a lower threshold, larger
    /// objects) take more.
    fn ram_per_tile_pixel(&self) -> f64 {
        <StarDistModel as Model>::POST_BYTES_PER_PIXEL
    }

    /// One inference at a time on the CPU (the AI engine's one slot): the
    /// network's activations on the largest image it gets at once - the
    /// tile, or a piece of it (`piece_size`, [`evanalyzer_nn::pieces::largest_model_input`]).
    /// 0.4 KB per pixel: measured 0.37 KB for the layout of StarDist's
    /// `2D_versatile_fluo` (BENCHMARK.md "AI models: RAM per inference" -
    /// [`NETWORK_BYTES_PER_PIXEL`]). On a GPU they are in its memory - planned anyway (the computer may
    /// run on its CPU). Plus the model's weights (one copy).
    fn shared_ram_bytes(&self, tile_width: usize, tile_height: usize) -> u64 {
        // Before the run it isn't known which of the two the model gets.
        let piece = self.pieces(tile_width, tile_height).largest();
        let pixels = evanalyzer_nn::pieces::largest_model_input(tile_width, piece)
            * evanalyzer_nn::pieces::largest_model_input(tile_height, piece);
        (NETWORK_BYTES_PER_PIXEL * pixels as f64) as u64 + super::model_bytes(&self.model_path)
    }
}

#[cfg(test)]
// Indices written as `row * width + column`, row 0 and column 0 included.
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn stardist(probability_threshold: f32, nms_threshold: f32) -> Stardist {
        Stardist {
            model_path: PathBuf::new(),
            segmentation_class: SegmentationClass(1),
            probability_threshold,
            nms_threshold,
            piece_size: PieceSize::Auto,
            normalization: InputNormalization::ModelDefault,
        }
    }

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    // ---- execute(): a real model file (a tiny random StarDist, nn's
    // `test_support`) through the engine ----

    /// The step runs the model file and writes objects: every instance
    /// pixel has the step's class, every class pixel an instance.
    #[test]
    fn execute_runs_the_model_file_and_writes_instances() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stardist.evanet");
        let config = evanalyzer_nn::test_support::tiny_stardist_config([2, 2]);
        std::fs::write(&path, evanalyzer_nn::test_support::stardist_file(&config)).unwrap();
        let (w, h) = (64, 48);
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0)
            .collect();
        let cmd = Stardist {
            model_path: path,
            ..stardist(0.0, 0.3)
        };
        let mut ctx = gray_ctx(w, h, pixels);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        let classes = ctx.get_segmentation_map().unwrap().as_slice().to_vec();
        let instances = ctx.get_instance_map().unwrap().as_slice().to_vec();
        assert!(
            instances.iter().any(|&i| i > 0),
            "threshold 0: candidates everywhere"
        );
        for (class, instance) in classes.iter().zip(&instances) {
            assert_eq!(*instance > 0, *class == 1);
        }
    }

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = Stardist {
            model_path: dir.path().join("missing.evanet"),
            ..stardist(0.5, 0.3)
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::Generic(_)));
    }
}
