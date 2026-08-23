use crate::ai_learning::model::SavedClassifier;
use crate::ai_learning::training_job::{self, TrainingImage, TrainingProgressEvent};
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
use crate::image::{ImageContainer, ImageReader, ImageTile, ReadMode};
use crate::object::Object;
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use crate::resources::MAX_TILE_SIZE;
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

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::ai_learning_settings::AiLearningBackendSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::RandomForestSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn gray_context(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, values, CpuAllocator)
                .unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
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

    // -- PixelTrainingJob::run (paths that need no image I/O) ------------------

    fn empty_pixel_job() -> PixelTrainingJob {
        PixelTrainingJob {
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                metadata: MetaData::default(),
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Pixel {
                    feature_spec: AiLearningPixelFeatureSettings { channels: vec![] },
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
}
