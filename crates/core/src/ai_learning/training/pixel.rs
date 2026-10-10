use crate::ai_learning::model::SavedClassifier;
use crate::ai_learning::training_job::{self, TrainingImage};
use crate::ai_learning::utils::{
    bbox_overlaps_tile, masked_pixels_in_tile, resolve_z_projection, tile_grid,
};
use crate::algos::EdgeDetectionSobel;
use crate::algos::GaussianBlur;
use crate::algos::Hessian;
use crate::algos::ImageAlgorithm;
use crate::algos::Laplacian;
use crate::algos::RankFilter;
use crate::algos::StructureTensor;
use crate::image::{ImageContainer, ImageReader, ImageTile, ManagedImage, ReadMode};
use crate::object::Object;
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::resources::MAX_TILE_SIZE;
use evanalyzer_cfg::core_types::TrainingProgressEvent;
use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
use evanalyzer_cfg::settings::ai_learning_pixel_settings::PreprocessingSteps;
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, PixelClassLabel, PixelInputColor,
};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use kornia_image::{Image, ImageSize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Computed feature channels for one image, in `compute_pixel_features`
/// order. Every channel is a single-value-per-pixel `F32Gray` image of the
/// bank's size - enforced when the bank is built, so `feature_vector_at`
/// can never silently index into interleaved RGB data instead.
pub struct FeatureBank {
    width: usize,
    height: usize,
    channels: Vec<Arc<ImageContainer>>,
}

impl FeatureBank {
    fn new(
        width: usize,
        height: usize,
        channels: Vec<Arc<ImageContainer>>,
    ) -> Result<Self, InternalErrors> {
        for channel in &channels {
            let ImageContainer::F32Gray(img) = channel.as_ref() else {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Gray feature channel".into(),
                    found: format!("{:?}", channel),
                });
            };
            if img.width() != width || img.height() != height {
                return Err(InternalErrors::FormatMismatch {
                    expected: format!("{width}x{height} feature channel"),
                    found: format!("{}x{}", img.width(), img.height()),
                });
            }
        }
        Ok(Self {
            width,
            height,
            channels,
        })
    }

    pub fn n_features(&self) -> usize {
        self.channels.len()
    }

    /// Feature vector for one pixel, one value per channel, in `channels` order.
    pub fn feature_vector_at(&self, x: usize, y: usize) -> Vec<f32> {
        self.channels
            .iter()
            .map(|c| match c.as_ref() {
                ImageContainer::F32Gray(img) => img.as_slice()[y * self.width + x],
                _ => unreachable!("FeatureBank::new only accepts F32Gray channels"),
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
///
/// A colour (`F32Rgb`) image is split into its R, G and B planes first and
/// every recipe entry is computed on each of them, so the feature order is
/// entry 0 (R, G, B), entry 1 (R, G, B), ... - three times as many features
/// as for a greyscale image. That's why a model only fits the image kind it
/// was trained on (see `PixelInputColor`).
pub fn compute_pixel_features(
    template: &PipelineContext,
    spec: &AiLearningPixelFeatureSettings,
) -> Result<FeatureBank, InternalErrors> {
    let size = template.image.size();
    let mut channels = Vec::new();

    match template.image.as_ref() {
        ImageContainer::F32Gray(_) => {
            for steps in &spec.channels {
                channels.push(compute_channel(template, steps)?);
            }
        }
        ImageContainer::F32Rgb(rgb) => {
            let planes = split_rgb(rgb)?
                .into_iter()
                .map(|plane| gray_ctx_from(template, plane))
                .collect::<Result<Vec<_>, _>>()?;
            for steps in &spec.channels {
                for plane in &planes {
                    channels.push(compute_channel(plane, steps)?);
                }
            }
        }
        ImageContainer::U32(_) => {
            return Err(InternalErrors::FormatMismatch {
                expected: "F32Gray or F32Rgb image for pixel features".into(),
                found: format!("{:?}", template.image),
            });
        }
    }

    FeatureBank::new(size.width, size.height, channels)
}

/// Splits an interleaved RGB image into its three colour planes, in R, G, B
/// order.
fn split_rgb(rgb: &ManagedImage<f32, 3>) -> Result<[Arc<ImageContainer>; 3], InternalErrors> {
    let size = rgb.size();
    let pixels = size.width * size.height;
    let mut planes = [
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
    ];
    for pixel in rgb.as_slice().chunks_exact(3) {
        for (plane, value) in planes.iter_mut().zip(pixel) {
            plane.push(*value);
        }
    }
    let to_container = |values: Vec<f32>| -> Result<Arc<ImageContainer>, InternalErrors> {
        Ok(Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(size, values).map_err(InternalErrors::from_kornia)?,
            tile_offset: rgb.tile_offset,
            plane: rgb.plane,
        })))
    };
    let [r, g, b] = planes;
    Ok([to_container(r)?, to_container(g)?, to_container(b)?])
}

/// A context like `template` but holding the greyscale `plane` as its image.
fn gray_ctx_from(
    template: &PipelineContext,
    plane: Arc<ImageContainer>,
) -> Result<PipelineContext, InternalErrors> {
    let mut image_meta = template.image_meta.clone();
    image_meta.is_rgb = false;
    PipelineContext::new_from_image(
        template.output_path.clone().unwrap_or_default(),
        image_meta,
        plane,
    )
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
    let mut cache = GlobalPipelineCache::default();
    for step in steps {
        match step {
            PreprocessingSteps::GaussianBlur(s) => {
                GaussianBlur::from(s.clone()).run(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::EdgeDetectionSobel(s) => {
                EdgeDetectionSobel::from(s.clone()).run(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Laplacian(s) => {
                Laplacian::from(s.clone()).run(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::StructureTensor(s) => {
                StructureTensor::from(s.clone()).run(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Hessian(s) => Hessian::from(s.clone()).run(&mut ctx, &mut cache)?,
            PreprocessingSteps::RankFilter(s) => {
                RankFilter::from(s.clone()).run(&mut ctx, &mut cache)?
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
    /// caller's responsibility to split beforehand. A colour image is read
    /// as one RGB channel (channel 0) and split into R, G and B by
    /// `compute_pixel_features` itself.
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
            ..
        } = &self.settings.classifier
        else {
            return Err(InternalErrors::Internal(
                "PixelTrainingJob requires a Pixel classifier configuration".to_string(),
            ));
        };

        let trained_input_color = self.training_input_color(class_labels)?;

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

            let tiles = tile_grid(full_width, full_height, MAX_TILE_SIZE);
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
        let (classifier, stats) = training_job::fit_classifier(
            &self.settings.backend,
            &rows,
            &labels,
            n_classes,
            &progress,
            &cancel,
        )?;
        let _ = progress.send(TrainingProgressEvent::Finished { stats });

        let mut settings = self.settings.clone();
        if let AiLearningClassifierSettings::Pixel { input_color, .. } = &mut settings.classifier {
            *input_color = trained_input_color;
        }
        Ok(training_job::finish(settings, classifier))
    }

    /// The image kind every training image that contributes samples has in
    /// common - what the model gets stamped with and is restricted to. A
    /// mixed set is refused up front: colour and greyscale images produce
    /// feature vectors of different lengths, which can't be trained into one
    /// model. Images that can't be opened are left to the training loop,
    /// which reports them as failed.
    fn training_input_color(
        &self,
        class_labels: &[PixelClassLabel],
    ) -> Result<PixelInputColor, InternalErrors> {
        let mut first: Option<(&TrainingImage, bool)> = None;
        for training_image in &self.images {
            let has_samples = training_image
                .labeled_objects
                .iter()
                .any(|o| resolve_label(class_labels, o.segmentation_class).is_some());
            if !has_samples {
                continue;
            }
            let Ok(reader) = ImageReader::new(&training_image.path, ReadMode::Default) else {
                continue;
            };
            let Some(is_rgb) = reader
                .get_image_meta()
                .series
                .get(&training_image.series)
                .and_then(|series| series.resolutions.get(&0))
                .map(|pyramid| pyramid.is_rgb)
            else {
                continue;
            };
            match first {
                None => first = Some((training_image, is_rgb)),
                Some((first_image, first_is_rgb)) if first_is_rgb != is_rgb => {
                    let kind = |rgb: bool| if rgb { "a colour" } else { "a greyscale" };
                    return Err(InternalErrors::InvalidArgument(format!(
                        "The training images mix colour and greyscale images: '{}' is {} \
                         image, '{}' is {} image. A pixel classifier is trained for one kind \
                         only - annotate either only colour or only greyscale images.",
                        first_image.path.display(),
                        kind(first_is_rgb),
                        training_image.path.display(),
                        kind(is_rgb),
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(match first {
            Some((_, true)) => PixelInputColor::Rgb,
            _ => PixelInputColor::Gray,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::ai_learning_settings::AiLearningBackendSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::RandomForestSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use kornia_image::{Image, ImageSize};

    fn gray_context(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    /// `values` interleaved as R, G, B per pixel.
    fn rgb_context(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 3>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test_rgb(img).unwrap()
    }

    // -- resolve_label ---------------------------------------------------------

    #[test]
    fn resolve_label_finds_the_matching_class() {
        let labels = vec![
            PixelClassLabel {
                class: SegmentationClass(5),
                name: "Cell".into(),
            },
            PixelClassLabel {
                class: SegmentationClass(6),
                name: "Background".into(),
            },
        ];
        assert_eq!(resolve_label(&labels, SegmentationClass(6)), Some(1));
    }

    #[test]
    fn resolve_label_is_none_for_an_unconfigured_class() {
        let labels = vec![PixelClassLabel {
            class: SegmentationClass(5),
            name: "Cell".into(),
        }];
        assert_eq!(resolve_label(&labels, SegmentationClass(99)), None);
    }

    // -- compute_pixel_features / FeatureBank -----------------------------------

    #[test]
    fn compute_pixel_features_an_empty_step_chain_is_the_raw_pixel_value() {
        let ctx = gray_context(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![]], // one raw channel, no preprocessing
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();

        assert_eq!(bank.n_features(), 1);
        assert_eq!(bank.feature_vector_at(0, 0), vec![1.0]);
        assert_eq!(bank.feature_vector_at(1, 0), vec![2.0]);
        assert_eq!(bank.feature_vector_at(0, 1), vec![3.0]);
        assert_eq!(bank.feature_vector_at(1, 1), vec![4.0]);
    }

    #[test]
    fn compute_pixel_features_produces_one_channel_per_spec_entry() {
        let ctx = gray_context(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![], vec![]], // two raw channels
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();

        assert_eq!(bank.n_features(), 2);
        assert_eq!(bank.feature_vector_at(0, 0), vec![1.0, 1.0]);
    }

    #[test]
    fn compute_pixel_features_runs_a_single_preprocessing_step() {
        use evanalyzer_cfg::settings::pipeline_command_settings::EdgeDetectionSobelSettings;

        // A flat image has zero gradient everywhere - Sobel output should be
        // all zeros, which is enough to prove the step actually ran (as
        // opposed to `compute_channel`'s empty-chain shortcut being hit by
        // mistake) without needing to hand-verify a specific kernel result.
        let ctx = gray_context(3, 3, vec![5.0; 9]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::EdgeDetectionSobel(
                EdgeDetectionSobelSettings { kernel_size: 3 },
            )]],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();

        assert_eq!(bank.n_features(), 1);
        assert_eq!(bank.feature_vector_at(1, 1), vec![0.0]);
    }

    #[test]
    fn compute_pixel_features_chains_multiple_steps_in_one_channel() {
        use evanalyzer_cfg::settings::pipeline_command_settings::GaussianBlurSettings;

        // Two GaussianBlur steps back to back on a non-flat image -
        // exercises `compute_channel`'s multi-step loop (every other test
        // here only ever runs zero or one step). The exact output value is
        // `GaussianBlur`'s own implementation detail (covered by its own
        // algorithm tests); this test's job is only to prove both steps
        // actually ran, checked by comparing against running the identical
        // step just once.
        #[rustfmt::skip]
        let values = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let step = || {
            PreprocessingSteps::GaussianBlur(GaussianBlurSettings {
                kernel_size: 3,
                sigma: 1.0,
            })
        };

        let once = AiLearningPixelFeatureSettings {
            channels: vec![vec![step()]],
        };
        let twice = AiLearningPixelFeatureSettings {
            channels: vec![vec![step(), step()]],
        };

        let bank_once = compute_pixel_features(&gray_context(4, 4, values.clone()), &once).unwrap();
        let bank_twice = compute_pixel_features(&gray_context(4, 4, values), &twice).unwrap();

        assert_eq!(bank_twice.n_features(), 1);
        assert_ne!(
            bank_once.feature_vector_at(2, 1),
            bank_twice.feature_vector_at(2, 1),
            "a second blur pass must further smooth the impulse, proving the chain didn't stop after the first step"
        );
    }

    #[test]
    fn compute_pixel_features_runs_a_laplacian_step() {
        use evanalyzer_cfg::settings::pipeline_command_settings::LaplacianSettings;

        // Must run without erroring - proves the Laplacian match arm in
        // `compute_channel` actually executes. The exact output value is
        // `Laplacian`'s own implementation detail (covered by its own
        // algorithm tests, see the earlier multi-step-chain test's comment
        // for why this test doesn't assert on it).
        let ctx = gray_context(3, 3, vec![5.0; 9]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::Laplacian(LaplacianSettings {
                kernel_size: 3,
            })]],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();
        assert_eq!(bank.n_features(), 1);
    }

    #[test]
    fn compute_pixel_features_runs_a_structure_tensor_step() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            FiltersStructureTensorTensorModeSettings, StructureTensorSettings,
        };

        let ctx = gray_context(3, 3, vec![5.0; 9]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::StructureTensor(
                StructureTensorSettings {
                    mode: FiltersStructureTensorTensorModeSettings::EigenvaluesX,
                    kernel_size: 3,
                    sigma: 1.0,
                },
            )]],
        };

        // Must run without erroring - proves the StructureTensor match arm
        // in `compute_channel` actually executes.
        let bank = compute_pixel_features(&ctx, &spec).unwrap();
        assert_eq!(bank.n_features(), 1);
    }

    #[test]
    fn compute_pixel_features_runs_a_hessian_step() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            FiltersHessianHessianModeSettings, HessianSettings,
        };

        let ctx = gray_context(3, 3, vec![5.0; 9]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::Hessian(HessianSettings {
                mode: FiltersHessianHessianModeSettings::Determinant,
            })]],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();
        assert_eq!(bank.n_features(), 1);
    }

    #[test]
    fn compute_pixel_features_runs_a_rank_filter_step() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            FiltersRankFilterRankFilterTypeSettings, RankFilterSettings,
        };

        let ctx = gray_context(3, 3, vec![5.0; 9]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::RankFilter(RankFilterSettings {
                radius: 1.0,
                filter_type: FiltersRankFilterRankFilterTypeSettings::Median,
            })]],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();
        assert_eq!(bank.n_features(), 1);
        // A flat image's median is the flat value itself.
        assert_eq!(bank.feature_vector_at(1, 1), vec![5.0]);
    }

    // -- compute_pixel_features on colour images --------------------------------

    #[test]
    fn compute_pixel_features_computes_every_entry_per_colour_in_entry_major_order() {
        use evanalyzer_cfg::settings::pipeline_command_settings::EdgeDetectionSobelSettings;

        // Flat colour planes (R=1, G=2, B=3), so entry 1's Sobel is zero on
        // every plane - the values alone pin which feature came from where.
        let ctx = rgb_context(3, 3, [1.0, 2.0, 3.0].repeat(9));
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![
                vec![],
                vec![PreprocessingSteps::EdgeDetectionSobel(
                    EdgeDetectionSobelSettings { kernel_size: 3 },
                )],
            ],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();

        assert_eq!(bank.n_features(), 6);
        assert_eq!(
            bank.feature_vector_at(1, 1),
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn compute_pixel_features_reads_each_pixel_of_an_rgb_image_not_interleaved_data() {
        // The original bug: indexing the interleaved RGB buffer as if it
        // were greyscale gave pixel 1 the value of pixel 0's green.
        let ctx = rgb_context(2, 1, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![]],
        };

        let bank = compute_pixel_features(&ctx, &spec).unwrap();

        assert_eq!(bank.feature_vector_at(0, 0), vec![0.1, 0.2, 0.3]);
        assert_eq!(bank.feature_vector_at(1, 0), vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn compute_pixel_features_runs_a_hessian_step_on_each_colour_of_an_rgb_image() {
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            FiltersHessianHessianModeSettings, HessianSettings,
        };

        // Hessian only supports greyscale input - on an RGB image it used
        // to fail with a format mismatch. Each colour plane gets a
        // different pattern; the RGB result must equal running the step on
        // each plane as its own greyscale image.
        let (w, h) = (5, 5);
        let plane = |f: fn(usize, usize) -> f32| -> Vec<f32> {
            (0..w * h).map(|i| f(i % w, i / w)).collect()
        };
        let r = plane(|x, y| (x * x + y) as f32);
        let g = plane(|x, y| (x * y) as f32);
        let b = plane(|x, y| (y * y) as f32 - x as f32);
        let interleaved: Vec<f32> = (0..w * h).flat_map(|i| [r[i], g[i], b[i]]).collect();
        let spec = AiLearningPixelFeatureSettings {
            channels: vec![vec![PreprocessingSteps::Hessian(HessianSettings {
                mode: FiltersHessianHessianModeSettings::Determinant,
            })]],
        };

        let rgb_bank = compute_pixel_features(&rgb_context(w, h, interleaved), &spec).unwrap();
        let per_plane: Vec<FeatureBank> = [r, g, b]
            .into_iter()
            .map(|p| compute_pixel_features(&gray_context(w, h, p), &spec).unwrap())
            .collect();

        assert_eq!(rgb_bank.n_features(), 3);
        for y in 0..h {
            for x in 0..w {
                let expected: Vec<f32> = per_plane
                    .iter()
                    .map(|bank| bank.feature_vector_at(x, y)[0])
                    .collect();
                assert_eq!(
                    rgb_bank.feature_vector_at(x, y),
                    expected,
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn feature_bank_refuses_a_non_greyscale_channel() {
        let rgb = rgb_context(1, 1, vec![1.0, 2.0, 3.0]).image;

        let err = FeatureBank::new(1, 1, vec![rgb]).err().unwrap();

        assert!(matches!(err, InternalErrors::FormatMismatch { .. }));
    }

    #[test]
    fn feature_bank_refuses_a_channel_of_the_wrong_size() {
        let small = gray_context(1, 1, vec![1.0]).image;

        let err = FeatureBank::new(2, 2, vec![small]).err().unwrap();

        assert!(matches!(err, InternalErrors::FormatMismatch { .. }));
    }

    // -- PixelTrainingJob::run (paths that need no image I/O) ------------------

    fn empty_pixel_job() -> PixelTrainingJob {
        PixelTrainingJob {
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData::default(),
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Pixel {
                    feature_spec: AiLearningPixelFeatureSettings { channels: vec![] },
                    input_color: Default::default(),
                    class_labels: vec![PixelClassLabel {
                        class: SegmentationClass(1),
                        name: "Cell".into(),
                    }],
                },
            },
            images: vec![],
            channel: 0,
            t_stack: 0,
            z_stack_handling: ZStackHandling::SingleStack,
        }
    }

    #[test]
    fn run_errors_for_an_object_classifier_configuration() {
        let mut job = empty_pixel_job();
        job.settings.classifier = AiLearningClassifierSettings::Object {
            feature_spec: evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings {
                metrics: vec![],
            },
            class_labels: vec![],
        };
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let err = job.run(tx, cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }

    #[test]
    fn run_with_no_images_fails_to_train_on_zero_samples() {
        let job = empty_pixel_job();
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let err = job.run(tx, cancel).unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero samples"));
    }

    #[test]
    fn run_returns_cancelled_when_the_flag_is_already_set_and_images_are_pending() {
        let mut job = empty_pixel_job();
        job.images.push(TrainingImage {
            path: PathBuf::from("does-not-exist.tif"),
            series: 0,
            labeled_objects: vec![],
        });
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(true));

        let err = job.run(tx, cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Cancelled));
    }

    #[test]
    fn run_skips_an_image_with_no_labeled_objects_without_touching_the_filesystem() {
        // `labeled_objects: vec![]` must be treated as "nothing to train from
        // in this image" and skipped *before* `ImageReader::new` is ever
        // called - proven here by pointing `path` at a file that doesn't
        // exist and getting the same "zero samples" error `run` gives for no
        // images at all, not an `ImageFailed`-driven one.
        let mut job = empty_pixel_job();
        job.images.push(TrainingImage {
            path: PathBuf::from("does-not-exist.tif"),
            series: 0,
            labeled_objects: vec![],
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let err = job.run(tx, cancel).unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero samples"));

        let events: Vec<_> = rx.iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::ItemCompleted { index: 0, .. })),
            "an image with no labeled objects still counts as processed"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::ImageFailed { .. })),
            "must be skipped before any file I/O is attempted, not reported as a failed read"
        );
    }

    #[test]
    fn run_skips_an_image_whose_objects_match_no_configured_class_label() {
        // Every object's `segmentation_class` fails to `resolve_label` -
        // `labeled_objects` collapses to empty the same way `vec![]` does
        // above, so this must also skip without any file I/O.
        use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

        let mut job = empty_pixel_job(); // class_labels only configures SegmentationClass(1)
        job.images.push(TrainingImage {
            path: PathBuf::from("does-not-exist.tif"),
            series: 0,
            labeled_objects: vec![ObjectMetricSettings {
                segmentation_class: SegmentationClass(99), // not in class_labels
                ..Default::default()
            }],
        });
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let err = job.run(tx, cancel).unwrap_err();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal, got a different variant");
        };
        assert!(msg.contains("zero samples"));
    }

    // -- PixelTrainingJob::run (real image I/O) ---------------------------
    //
    // Everything above deliberately avoids touching the filesystem (see
    // this section's sibling above). This one real end-to-end run - reading
    // an actual fixture through Bio-Formats, tiling it, computing features,
    // and fitting a classifier - is what exercises the rest of `run`'s body
    // (the tile grid / z-stack / `ImageReader` machinery around line
    // 197 onward) that no amount of synthetic-data unit testing reaches.

    use bitvec::prelude::*;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

    /// A `[x_min, y_min, x_max, y_max]` inclusive bbox, fully-filled mask -
    /// same construction as `ai_learning::utils::tests::square_object`, but
    /// building the settings type directly since `PixelTrainingJob::run`
    /// reconstructs an `Object` from `ObjectMetricSettings` itself.
    fn full_square_object_settings(
        id: u128,
        x_min: u32,
        y_min: u32,
        side: u32,
        segmentation_class: SegmentationClass,
    ) -> ObjectMetricSettings {
        let area = (side * side) as usize;
        ObjectMetricSettings {
            id: evanalyzer_cfg::core_types::ObjectId(id),
            segmentation_class,
            bbox: [x_min, y_min, x_min + side - 1, y_min + side - 1],
            mask_data: bitvec![u64, Lsb0; 1; area],
            area,
            ..Default::default()
        }
    }

    #[test]
    fn run_trains_end_to_end_against_a_real_image_fixture() {
        let fixture = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/multi-channel-4D-series.ome.tif"
        ));

        // Two small, non-overlapping, differently-classed regions in the
        // fixture's top-left corner - real pixel values, but the exact
        // values don't matter here (unlike `random_forest.rs`'s own fit
        // tests): this test's job is to prove the tile-reading/feature/fit
        // pipeline runs end to end, not that the model classifies well.
        let cell = full_square_object_settings(1, 0, 0, 2, SegmentationClass(1));
        let background = full_square_object_settings(2, 10, 10, 2, SegmentationClass(2));

        let mut job = empty_pixel_job();
        job.settings.classifier = AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings {
                channels: vec![vec![]], // raw pixel value
            },
            input_color: Default::default(),
            class_labels: vec![
                PixelClassLabel {
                    class: SegmentationClass(1),
                    name: "Cell".into(),
                },
                PixelClassLabel {
                    class: SegmentationClass(2),
                    name: "Background".into(),
                },
            ],
        };
        job.images.push(TrainingImage {
            path: fixture,
            series: 0,
            labeled_objects: vec![cell, background],
        });

        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let saved = job.run(tx, cancel).unwrap();

        let AiLearningClassifierSettings::Pixel { class_labels, .. } = &saved.settings.classifier
        else {
            panic!("expected a Pixel classifier configuration to round-trip through `finish`");
        };
        assert_eq!(class_labels.len(), 2);
        assert_eq!(trained_input_color(&saved), PixelInputColor::Gray);

        let events: Vec<_> = rx.iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::Training))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::Finished { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::ImageTilesScheduled { total_tiles, .. } if *total_tiles > 0)),
            "the fixture image must actually get tiled and read, not silently skipped"
        );
    }

    fn trained_input_color(saved: &SavedClassifier) -> PixelInputColor {
        let AiLearningClassifierSettings::Pixel { input_color, .. } = &saved.settings.classifier
        else {
            panic!("expected a Pixel classifier");
        };
        *input_color
    }

    /// A 16x16 8-bit TIFF: the left half one colour, the right half another,
    /// so the two classes of `two_class_job` are separable.
    fn write_rgb_tiff(path: &std::path::Path) {
        let img = image::RgbImage::from_fn(16, 16, |x, _| {
            if x < 8 {
                image::Rgb([200, 20, 20])
            } else {
                image::Rgb([20, 200, 20])
            }
        });
        img.save(path).unwrap();
    }

    fn write_gray_tiff(path: &std::path::Path) {
        let img =
            image::GrayImage::from_fn(16, 16, |x, _| image::Luma([if x < 8 { 30 } else { 220 }]));
        img.save(path).unwrap();
    }

    /// Two classes, one 2x2 annotation per class (left half / right half)
    /// on every image in `paths`.
    fn two_class_job(paths: &[PathBuf]) -> PixelTrainingJob {
        let mut job = empty_pixel_job();
        job.settings.classifier = AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings {
                channels: vec![vec![]],
            },
            input_color: Default::default(),
            class_labels: vec![
                PixelClassLabel {
                    class: SegmentationClass(1),
                    name: "Left".into(),
                },
                PixelClassLabel {
                    class: SegmentationClass(2),
                    name: "Right".into(),
                },
            ],
        };
        for path in paths {
            job.images.push(TrainingImage {
                path: path.clone(),
                series: 0,
                labeled_objects: vec![
                    full_square_object_settings(1, 2, 2, 2, SegmentationClass(1)),
                    full_square_object_settings(2, 12, 12, 2, SegmentationClass(2)),
                ],
            });
        }
        job
    }

    #[test]
    fn run_trains_on_colour_images_and_marks_the_model_as_rgb() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("colour.tif");
        write_rgb_tiff(&path);

        let (tx, _rx) = std::sync::mpsc::channel();
        let saved = two_class_job(&[path])
            .run(tx, Arc::new(AtomicBool::new(false)))
            .unwrap();

        assert_eq!(trained_input_color(&saved), PixelInputColor::Rgb);
    }

    #[test]
    fn run_refuses_a_mix_of_colour_and_greyscale_training_images() {
        let dir = tempfile::tempdir().unwrap();
        let colour = dir.path().join("colour.tif");
        let gray = dir.path().join("gray.tif");
        write_rgb_tiff(&colour);
        write_gray_tiff(&gray);

        let (tx, _rx) = std::sync::mpsc::channel();
        let err = two_class_job(&[colour, gray])
            .run(tx, Arc::new(AtomicBool::new(false)))
            .err()
            .unwrap();

        let InternalErrors::InvalidArgument(message) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            message.contains("mix colour and greyscale images")
                && message.contains("colour.tif")
                && message.contains("gray.tif"),
            "{message}"
        );
    }
}
