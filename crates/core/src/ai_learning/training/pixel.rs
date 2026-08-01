use crate::ai_learning::model::SavedClassifier;
use crate::ai_learning::training_job::{self, TrainingImage, TrainingProgressEvent};
use crate::ai_learning::utils::{
    TILE_SIZE, bbox_overlaps_tile, masked_pixels_in_tile, resolve_z_projection, tile_grid,
};
use crate::algos::EdgeDetectionSobel;
use crate::algos::GaussianBlur;
use crate::algos::Hessian;
use crate::algos::ImageAlgorithm;
use crate::algos::Laplacian;
use crate::algos::RankFilter;
use crate::algos::StructureTensor;
use crate::image::{ImageContainer, ImageReader, ImageTile, ReadMode};
use crate::object::Object;
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
use evanalyzer_cfg::settings::ai_learning_pixel_settings::PreprocessingSteps;
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, PixelClassLabel,
};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use kornia_image::ImageSize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Computed feature channels for one image, in `FeatureSpec::channels` order.
pub struct FeatureBank {
    width: usize,
    height: usize,
    channels: Vec<Arc<ImageContainer>>,
}

impl FeatureBank {
    pub fn n_features(&self) -> usize {
        self.channels.len()
    }

    /// Feature vector for one pixel, one value per channel, in `channels` order.
    /// Assumes every channel is single-channel (grayscale-derived) output.
    pub fn feature_vector_at(&self, x: usize, y: usize) -> Vec<f32> {
        self.channels
            .iter()
            .map(|c| {
                let slice = c
                    .as_f32_slice()
                    .expect("feature channel must be an f32 image");
                slice[y * self.width + x]
            })
            .collect()
    }
}

/// Builds the feature bank for one image, reusing the exact same optimized
/// `ImageAlgorithm` Commands (and their Arc-shared, scratch-pad/swap buffer model)
/// used by the main pipeline — no separate/duplicated filter math.
///
/// `template` supplies the source image plus the `image_meta`/`output_path` needed
/// to construct fresh per-channel `PipelineContext`s. Each channel gets its own
/// context sharing the same source `Arc` (cheap refcount bump, no pixel copy), so
/// filters never step on each other's input.
pub fn compute_pixel_features(
    template: &PipelineContext,
    spec: &AiLearningPixelFeatureSettings,
) -> Result<FeatureBank, InternalErrors> {
    let size = template.image.size();
    let mut channels = Vec::with_capacity(spec.channels.len());

    for steps in &spec.channels {
        channels.push(compute_channel(template, steps)?);
    }

    Ok(FeatureBank {
        width: size.width,
        height: size.height,
        channels,
    })
}

fn fresh_ctx(template: &PipelineContext) -> Result<PipelineContext, InternalErrors> {
    PipelineContext::new_from_image(
        template.output_path.clone().unwrap_or_default(),
        template.image_meta.clone(),
        template.image.clone(),
    )
}

fn compute_channel(
    template: &PipelineContext,
    steps: &[PreprocessingSteps],
) -> Result<Arc<ImageContainer>, InternalErrors> {
    if steps.is_empty() {
        return Ok(template.image.clone());
    }

    let mut ctx = fresh_ctx(template)?;
    let mut cache = PipelineCache::default();
    for step in steps {
        match step {
            PreprocessingSteps::GaussianBlur(s) => {
                GaussianBlur::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::EdgeDetectionSobel(s) => {
                EdgeDetectionSobel::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Laplacian(s) => {
                Laplacian::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::StructureTensor(s) => {
                StructureTensor::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Hessian(s) => {
                Hessian::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::RankFilter(s) => {
                RankFilter::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
        }
    }
    Ok(ctx.image)
}

/// Trains a pixel classifier across a list of images, reading each one
/// tile-by-tile (never loading a full image into memory at once - the same
/// requirement whole-slide images already impose on the main pipeline) and
/// only fetching tiles whose bounds actually overlap a labeled object.
///
/// Unlike `JobExecutor` (which processes each tile independently and writes
/// results incrementally), this is a map-then-reduce shape: every tile's
/// features get folded into one accumulated `(rows, labels)` set, and the
/// actual model fit happens once, after every image has been scanned.
///
/// `settings.classifier` must be `AiLearningClassifierSettings::Pixel` -
/// `run` returns an error otherwise.
pub struct PixelTrainingJob {
    pub settings: AiLearningSettings,
    pub images: Vec<TrainingImage>,
    /// Which image channel this classifier trains on - pixel-classifier
    /// feature computation operates on a single channel (see
    /// `compute_pixel_features`'s doc comment); multi-channel images are the
    /// caller's responsibility to split beforehand.
    pub channel: i32,
    /// Which time frame to read, alongside `channel`. No multi-t-stack
    /// handling (unlike z) - a single scalar index; add a `TStackHandling`-
    /// style mode later if that's ever needed.
    pub t_stack: i32,
    /// How to handle z-stacks - mirrors `JobExecutor::prepare_z_stack_iterator`'s
    /// handling table. `SingleStack` reads just the first z-plane (no
    /// project-configurable z-range like the main pipeline supports - a
    /// deliberate simplification, since this job has no per-image
    /// `ZStackSettings` concept). `AllStacks` reads every z-plane and treats
    /// each one as its own training sample at the same (x, y) - so sample
    /// count scales with z-depth for that mode, worth knowing going in.
    pub z_stack_handling: ZStackHandling,
}

impl PixelTrainingJob {
    /// Runs synchronously on the calling thread - use `run_async` to run in
    /// the background the way pipeline execution does.
    pub fn run(
        &self,
        progress: Sender<TrainingProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<SavedClassifier, InternalErrors> {
        let AiLearningClassifierSettings::Pixel {
            feature_spec,
            class_labels,
        } = &self.settings.classifier
        else {
            return Err(InternalErrors::Internal(
                "PixelTrainingJob requires a Pixel classifier configuration".to_string(),
            ));
        };

        let _ = progress.send(TrainingProgressEvent::Started {
            total: self.images.len(),
        });

        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();

        for (image_index, training_image) in self.images.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }

            let labeled_objects: Vec<(Object, usize)> = training_image
                .labeled_objects
                .iter()
                .filter_map(|settings| {
                    let label = resolve_label(class_labels, settings.segmentation_class)?;
                    Some((Object::from_object_settings(settings.clone()), label))
                })
                .collect();

            if labeled_objects.is_empty() {
                let _ = progress.send(TrainingProgressEvent::ItemCompleted {
                    index: image_index,
                    total: self.images.len(),
                });
                continue;
            }

            let Ok(reader) = ImageReader::new(&training_image.path, ReadMode::Default) else {
                let _ = progress.send(TrainingProgressEvent::ImageFailed {
                    path: training_image.path.clone(),
                });
                continue;
            };

            let image_meta = reader.get_image_meta();
            let Some(series_info) = image_meta.series.get(&training_image.series) else {
                let _ = progress.send(TrainingProgressEvent::ImageFailed {
                    path: training_image.path.clone(),
                });
                continue;
            };
            let Some(pyramid) = series_info.resolutions.get(&0) else {
                let _ = progress.send(TrainingProgressEvent::ImageFailed {
                    path: training_image.path.clone(),
                });
                continue;
            };

            let full_width = pyramid.width as usize;
            let full_height = pyramid.height as usize;
            let is_rgb = pyramid.is_rgb;
            let nr_of_bits = pyramid.nr_bits;
            let pixel_sizes = series_info.pixel_sizes.clone();
            let full_image_size = ImageSize {
                width: full_width,
                height: full_height,
            };

            let (z_projection, z_range) =
                resolve_z_projection(&self.z_stack_handling, series_info.nr_z_stacks);

            let tiles = tile_grid(full_width, full_height, TILE_SIZE);
            let relevant_tiles: Vec<&ImageTile> = tiles
                .iter()
                .filter(|t| {
                    labeled_objects
                        .iter()
                        .any(|(o, _)| bbox_overlaps_tile(o.bbox, t))
                })
                .collect();

            let _ = progress.send(TrainingProgressEvent::ImageTilesScheduled {
                image_index,
                total_tiles: relevant_tiles.len(),
            });

            for (tile_index, tile) in relevant_tiles.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InternalErrors::Cancelled);
                }

                for z in z_range.clone() {
                    let loaded_channels = reader.read_image_tile_combined(
                        training_image.series,
                        0, // base resolution - pyramid levels beyond 0 not handled yet
                        z_projection.clone(),
                        &Some(z..=z),
                        self.t_stack,
                        Some(&vec![self.channel]),
                        tile,
                    )?;

                    let Some(channel_image) = loaded_channels
                        .into_iter()
                        .find(|c| c.c_stack == self.channel)
                    else {
                        continue;
                    };

                    let loaded_size = channel_image.image.size();
                    let tile_image_meta = PipelineImageMeta {
                        image_tile_info: ImageTile {
                            width: loaded_size.width,
                            height: loaded_size.height,
                            offset_x: tile.offset_x,
                            offset_y: tile.offset_y,
                        },
                        full_image_width: full_image_size,
                        is_rgb,
                        nr_of_bits,
                        pixel_sizes: pixel_sizes.clone(),
                    };

                    let ctx = PipelineContext::new_from_image(
                        PathBuf::new(),
                        tile_image_meta,
                        channel_image.image,
                    )?;

                    let bank = compute_pixel_features(&ctx, feature_spec)?;

                    for (object, label) in &labeled_objects {
                        for (x, y) in masked_pixels_in_tile(object, tile) {
                            let local_x = x - tile.offset_x;
                            let local_y = y - tile.offset_y;
                            rows.push(bank.feature_vector_at(local_x, local_y));
                            labels.push(*label);
                        }
                    }
                }

                let _ = progress.send(TrainingProgressEvent::TileProcessed {
                    image_index,
                    tile_index,
                    total_tiles: relevant_tiles.len(),
                });
            }

            let _ = progress.send(TrainingProgressEvent::ItemCompleted {
                index: image_index,
                total: self.images.len(),
            });
        }

        let _ = progress.send(TrainingProgressEvent::Training);
        let n_classes = class_labels.len();
        let classifier =
            training_job::fit_classifier(&self.settings.backend, &rows, &labels, n_classes)?;
        let _ = progress.send(TrainingProgressEvent::Finished);

        Ok(training_job::finish(self.settings.clone(), classifier))
    }

    /// Runs in a background thread, mirroring `JobExecutor::run_async`'s
    /// exact shape (progress channel + shared cancel flag) so the GUI can
    /// wire this up the same way it already wires up pipeline execution.
    pub fn run_async(
        self,
    ) -> (
        JoinHandle<Result<SavedClassifier, InternalErrors>>,
        Receiver<TrainingProgressEvent>,
        Arc<AtomicBool>,
    ) {
        training_job::spawn_training_job(self, Self::run)
    }
}

fn resolve_label(class_labels: &[PixelClassLabel], class: SegmentationClass) -> Option<usize> {
    class_labels.iter().position(|l| l.class == class)
}

