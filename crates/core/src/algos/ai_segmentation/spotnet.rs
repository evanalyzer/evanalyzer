//! # spotnet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::path::PathBuf;
use std::sync::Arc;

use super::{InputNormalization, PieceSize};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, SegmentationClass,
};
use evanalyzer_nn::model::spotnet::{SpotNetModel, SpotNetParams};
use evanalyzer_nn::{Image, Model};
use macros::CommandsMeta;

use crate::{
    ai_devices,
    algos::{ExecutionScope, ImageAlgorithm, math::load_channel::missing_channel},
    image::ImageContainer,
    pipeline::{
        pipeline_cache::{CacheAddress, GlobalPipelineCache},
        pipeline_context::PipelineContext,
    },
};

/// Spot detection with a SpotNet model trained in EVAnalyzer (`.evanet`):
/// a small network that gives every pixel the probability of being a spot's
/// centre; the map's peaks above `threshold` become one-pixel objects, one
/// per spot - also for touching spots, which a threshold would merge.
///
/// The spots are found in the pipeline's image. A model trained with a
/// reference channel (a channel showing only the structures spots must not
/// be found on, e.g. serum) gets that channel too and compares the two
/// itself - tolerant to channels a few pixels apart. Runs on a GPU when
/// there is one, else on the CPU.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "segment",
    next = "extract",
    display_name = "AI SpotNet Spot Detection"
)]
pub struct Spotnet {
    /// The SpotNet model file (EVAnalyzer's `.evanet`), trained in the app.
    #[cmdsmeta(file_extensions = NN_MODEL_EXTENSIONS)]
    pub model_path: PathBuf,

    /// The class every found spot gets. All other pixels are assigned
    /// `SegmentationClass::BACKGROUND`.
    #[cmdsmeta(default = SegmentationClass(1))]
    pub segmentation_class: SegmentationClass,

    /// A peak of the spot probability counts as a spot from this on: lower
    /// finds fainter spots and more wrong ones.
    #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.01, summary = true)]
    pub threshold: f32,

    /// The second input of a model trained with a reference channel; none
    /// for a model of the spot channel alone.
    #[cmdsmeta(default = SpotReference::None)]
    pub reference: SpotReference,

    /// How large the pieces are that a large tile runs through the network
    /// in: smaller ones take less RAM, larger ones are faster.
    #[cmdsmeta(default = PieceSize::Auto, optional = true, visibility = Advanced)]
    pub piece_size: PieceSize,

    /// How the image is scaled before the model sees it: by default each
    /// channel's 1st and 99.8th percentile of the tile become 0 and 1, as
    /// the model was trained.
    #[cmdsmeta(default = InputNormalization::ModelDefault, optional = true, visibility = Advanced)]
    pub normalization: InputNormalization,
}

/// The second input of a [`Spotnet`] model.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum SpotReference {
    /// The model takes the spot channel alone.
    None,
    /// The model takes a reference channel as its second input.
    Channel {
        /// The channel showing only the structures (as it was read).
        channel: ImageChannelIdx,
    },
}

/// What the network takes per pixel of the image it gets at once (see
/// `shared_ram_bytes`) - nn's figure.
const NETWORK_BYTES_PER_PIXEL: f64 = <SpotNetModel as Model>::NET_BYTES_PER_PIXEL;

impl Spotnet {
    /// The pieces a `width x height` tile runs in (see [`PieceSize::pieces`]).
    fn pieces(&self, width: usize, height: usize) -> evanalyzer_nn::Pieces {
        self.piece_size
            .pieces(NETWORK_BYTES_PER_PIXEL, width, height)
    }
}

impl ImageAlgorithm for Spotnet {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        // The reference channel first: the maps below borrow the context.
        let reference = match self.reference {
            SpotReference::None => None,
            SpotReference::Channel { channel } => {
                let tile = ctx.image_meta.image_tile_info;
                let index = channel.0 as i32;
                let image: Arc<ImageContainer> = cache
                    .image_cache
                    .get(&CacheAddress::Channel((index, tile)))
                    .ok_or_else(|| missing_channel(cache, index, tile))?;
                Some((channel, image))
            }
        };
        let (input_image, segmentation_map, instance_map) =
            ctx.get_f32_gray_segmentation_and_instances_mut()?;
        let size = input_image.size();
        let (width, height) = (size.width, size.height);

        // The spot channel, then the reference: one plane after the other.
        let both: Vec<f32>;
        let (data, channels) = match &reference {
            None => (input_image.as_slice(), 1),
            Some((channel, image)) => {
                let ImageContainer::F32Gray(r) = image.as_ref() else {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "AI SpotNet Spot Detection: the reference channel {} must be a grey image",
                        channel.0
                    )));
                };
                let other = r.data.size();
                if (other.width, other.height) != (width, height) {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "AI SpotNet Spot Detection: the image ({width} x {height}) and the \
                         reference channel {} ({} x {}) differ in size",
                        channel.0, other.width, other.height
                    )));
                }
                both = [input_image.as_slice(), r.data.as_slice()].concat();
                (both.as_slice(), 2)
            }
        };

        let params = SpotNetParams {
            normalization: self.normalization.to_nn(),
            pieces: self.pieces(width, height),
            threshold: self.threshold,
        };
        let image = Image {
            width,
            height,
            channels,
            data,
        };
        ai_devices::run::<SpotNetModel>(
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
        "Spotnet"
    }

    /// EVAnalyzer's own network: nothing to cite.
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        Vec::new()
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// What the step keeps per worker: the probability map and the spot
    /// list (nn's figure), plus the two-channel input when a reference
    /// channel is given (4 bytes per pixel and channel).
    fn ram_per_tile_pixel(&self) -> f64 {
        let input = match self.reference {
            SpotReference::None => 0.0,
            SpotReference::Channel { .. } => 8.0,
        };
        <SpotNetModel as Model>::POST_BYTES_PER_PIXEL + input
    }

    /// One inference at a time on the CPU (the AI engine's one slot): the
    /// network's activations on the largest image it gets at once - the
    /// tile, or a piece of it ([`evanalyzer_nn::pieces::largest_model_input`]),
    /// plus the model's weights (one copy). nn's figure is an estimate until
    /// measured with `tests::tile_ram`.
    fn shared_ram_bytes(&self, tile_width: usize, tile_height: usize) -> u64 {
        let piece = self.pieces(tile_width, tile_height).largest();
        let pixels = evanalyzer_nn::pieces::largest_model_input(tile_width, piece)
            * evanalyzer_nn::pieces::largest_model_input(tile_height, piece);
        (NETWORK_BYTES_PER_PIXEL * pixels as f64) as u64 + super::model_bytes(&self.model_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ManagedImage;
    use kornia_image::{Image as KorniaImage, ImageSize};

    fn spotnet(model_path: PathBuf, threshold: f32, reference: SpotReference) -> Spotnet {
        Spotnet {
            model_path,
            segmentation_class: SegmentationClass(1),
            threshold,
            reference,
            piece_size: PieceSize::Auto,
            normalization: InputNormalization::ModelDefault,
        }
    }

    fn pixels(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0)
            .collect()
    }

    fn gray_ctx(width: usize, height: usize) -> PipelineContext {
        let img =
            KorniaImage::<f32, 1>::new(ImageSize { width, height }, pixels(width, height)).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    /// A tiny random SpotNet of `in_channels` in a temp dir.
    fn model_file(dir: &tempfile::TempDir, in_channels: usize) -> PathBuf {
        let path = dir.path().join("spots.evanet");
        std::fs::write(
            &path,
            evanalyzer_nn::test_support::spotnet_file(in_channels),
        )
        .unwrap();
        path
    }

    /// Every spot is one pixel of its own instance, and has the step's class.
    fn assert_one_pixel_spots(ctx: &PipelineContext) -> usize {
        let classes = ctx.get_segmentation_map().unwrap().as_slice().to_vec();
        let instances = ctx.get_instance_map().unwrap().as_slice().to_vec();
        for (class, instance) in classes.iter().zip(&instances) {
            assert_eq!(*instance > 0, *class == 1);
        }
        let mut ids: Vec<u32> = instances.iter().copied().filter(|&i| i > 0).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "an instance of more than one pixel");
        n
    }

    #[test]
    fn execute_runs_the_model_file_and_writes_one_pixel_spots() {
        let dir = tempfile::tempdir().unwrap();
        // Threshold 0: every peak of a random model's map is a spot.
        let cmd = spotnet(model_file(&dir, 1), 0.0, SpotReference::None);
        let mut ctx = gray_ctx(64, 48);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert!(
            assert_one_pixel_spots(&ctx) > 0,
            "threshold 0: peaks everywhere"
        );
    }

    #[test]
    fn a_threshold_of_one_finds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = spotnet(model_file(&dir, 1), 1.0, SpotReference::None);
        let mut ctx = gray_ctx(32, 32);
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        assert_eq!(assert_one_pixel_spots(&ctx), 0);
    }

    #[test]
    fn a_two_channel_model_gets_the_reference_channel() {
        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (40, 30);
        let mut ctx = gray_ctx(w, h);
        let mut cache = GlobalPipelineCache::default();
        let tile = ctx.image_meta.image_tile_info;
        let reference = KorniaImage::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            vec![0.2; w * h],
        )
        .unwrap();
        let reference = ImageContainer::F32Gray(ManagedImage {
            data: reference,
            tile_offset: Default::default(),
            plane: None,
            dimensionality: Default::default(),
        });
        cache
            .image_cache
            .insert(CacheAddress::Channel((3, tile)), Arc::new(reference));
        let reference = SpotReference::Channel {
            channel: ImageChannelIdx(3),
        };
        let cmd = spotnet(model_file(&dir, 2), 0.0, reference);
        cmd.execute(&mut ctx, &mut cache).unwrap();
        assert!(assert_one_pixel_spots(&ctx) > 0);
    }

    #[test]
    fn a_missing_reference_channel_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let reference = SpotReference::Channel {
            channel: ImageChannelIdx(5),
        };
        let cmd = spotnet(model_file(&dir, 2), 0.5, reference);
        let mut ctx = gray_ctx(16, 16);
        assert!(
            cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
                .is_err()
        );
    }

    #[test]
    fn a_model_trained_with_a_reference_needs_one() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = spotnet(model_file(&dir, 2), 0.5, SpotReference::None);
        let mut ctx = gray_ctx(16, 16);
        let error = cmd
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("2 channel"), "{error}");
    }

    /// A run with another kind of model file: the error says what it is and
    /// which step runs it.
    #[test]
    fn a_stardist_model_file_names_the_step_that_runs_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cells.evanet");
        let config = evanalyzer_nn::test_support::tiny_stardist_config([2, 2]);
        std::fs::write(&path, evanalyzer_nn::test_support::stardist_file(&config)).unwrap();
        let cmd = spotnet(path, 0.5, SpotReference::None);
        let mut ctx = gray_ctx(8, 8);
        let error = cmd
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("is a StarDist 2D model") && error.contains("AI Stardist Segmentation"),
            "{error}"
        );
    }
}
