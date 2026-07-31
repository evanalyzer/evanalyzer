use super::model::{self, CURRENT_SAVED_CLASSIFIER_VERSION, FeatureRecipe, SavedClassifier};
use super::pixel::compute_pixel_features;
use super::pixel_settings::FeatureSpec;
use crate::ZProjection;
use crate::image::{ImageReader, ImageTile, ReadMode};
use crate::object::Object;
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_context::PipelineContext;
use evanalyzer_cfg::core_types::{InternalErrors, SegmentationClass};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use kornia_image::ImageSize;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Matches `JobExecutor::analyze_image`'s fixed tile size
/// (`crates/core/src/job/job_executor.rs`) - not reused directly (that
/// constant is private to `JobExecutor`), but kept identical for consistency.
const TILE_SIZE: usize = 4096;

/// One image contributing labeled training samples to a `PixelTrainingJob`.
///
/// `labeled_objects` pairs each manually-painted, labeled Object (the settings
/// DTO already held by the project/GUI layer - not core's internal `Object`,
/// which stays private to this crate) with its resolved training class index.
/// This job converts each one to an `Object` internally (to reuse its
/// bbox/mask) and walks only the pixels inside its mask - passing already-
/// flattened per-pixel coordinates instead would balloon memory for large
/// painted regions (a compact bbox+bitmask is far smaller than one tuple per
/// masked pixel).
pub struct TrainingImage {
    pub path: PathBuf,
    pub series: i32,
    pub labeled_objects: Vec<(ObjectMetricSettings, usize)>,
}

pub enum TrainingBackend {
    RandomForest,
    Knn,
    Mlp {
        hidden_layers: Vec<usize>,
        epochs: usize,
        learning_rate: f64,
    },
}

pub enum TrainingProgressEvent {
    Started {
        total_images: usize,
    },
    ImageTilesScheduled {
        image_index: usize,
        total_tiles: usize,
    },
    TileProcessed {
        image_index: usize,
        tile_index: usize,
        total_tiles: usize,
    },
    ImageCompleted {
        index: usize,
        total: usize,
    },
    ImageFailed {
        path: PathBuf,
    },
    Training,
    Finished,
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
pub struct PixelTrainingJob {
    pub images: Vec<TrainingImage>,
    /// Which image channel this classifier trains on - pixel-classifier
    /// feature computation operates on a single channel (see
    /// `pixel::compute_pixel_features`'s doc comment); multi-channel images
    /// are the caller's responsibility to split beforehand.
    pub channel: i32,
    /// Which time frame to read, alongside `channel`. No multi-t-stack
    /// handling (unlike z) - a single scalar index, per how this was asked
    /// for; add a `TStackHandling`-style mode later if that's ever needed.
    pub t_stack: i32,
    /// How to handle z-stacks - mirrors `JobExecutor::prepare_z_stack_iterator`'s
    /// handling table. `SingleStack` reads just the first z-plane (no
    /// project-configurable z-range like the main pipeline supports - a
    /// deliberate simplification, since this job has no per-image
    /// `ZStackSettings` concept). `AllStacks` reads every z-plane and treats
    /// each one as its own training sample at the same (x, y) - so sample
    /// count scales with z-depth for that mode, worth knowing going in.
    pub z_stack_handling: ZStackHandling,
    pub feature_spec: FeatureSpec,
    pub backend: TrainingBackend,
    pub class_labels: Vec<SegmentationClass>,
}

impl PixelTrainingJob {
    /// Runs synchronously on the calling thread - use `run_async` to run in
    /// the background the way pipeline execution does.
    pub fn run(
        &self,
        progress: Sender<TrainingProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<SavedClassifier, InternalErrors> {
        let _ = progress.send(TrainingProgressEvent::Started {
            total_images: self.images.len(),
        });

        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();

        for (image_index, training_image) in self.images.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            if training_image.labeled_objects.is_empty() {
                let _ = progress.send(TrainingProgressEvent::ImageCompleted {
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

            let labeled_objects: Vec<(Object, usize)> = training_image
                .labeled_objects
                .iter()
                .map(|(settings, label)| (Object::from_object_settings(settings.clone()), *label))
                .collect();

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

                    let bank = compute_pixel_features(&ctx, &self.feature_spec)?;

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

            let _ = progress.send(TrainingProgressEvent::ImageCompleted {
                index: image_index,
                total: self.images.len(),
            });
        }

        let _ = progress.send(TrainingProgressEvent::Training);

        let classifier = match &self.backend {
            TrainingBackend::RandomForest => model::fit_random_forest(&rows, &labels)?,
            TrainingBackend::Knn => model::fit_knn(&rows, &labels)?,
            TrainingBackend::Mlp {
                hidden_layers,
                epochs,
                learning_rate,
            } => model::fit_mlp(&rows, &labels, hidden_layers, *epochs, *learning_rate)?,
        };

        let _ = progress.send(TrainingProgressEvent::Finished);

        Ok(SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            feature_recipe: FeatureRecipe::Pixel {
                feature_spec: self.feature_spec.clone(),
                class_labels: self.class_labels.clone(),
            },
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
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);
        let handle = std::thread::spawn(move || self.run(tx, cancel_clone));
        (handle, rx, cancel)
    }
}

/// Mirrors `JobExecutor::prepare_z_stack_iterator`'s handling table
/// (`crates/core/src/job/job_executor.rs`), minus the project-configurable
/// z-range `SingleStack` normally supports there - this job has no
/// per-image `ZStackSettings`, so `SingleStack` here always means "just the
/// first z-plane."
fn resolve_z_projection(
    handling: &ZStackHandling,
    nr_z_stacks: i32,
) -> (ZProjection, RangeInclusive<i32>) {
    match handling {
        ZStackHandling::SingleStack => (ZProjection::None, 0..=0),
        ZStackHandling::AllStacks => (ZProjection::None, 0..=(nr_z_stacks - 1)),
        ZStackHandling::MaxIntensity => (ZProjection::MaxIntensity, 0..=0),
        ZStackHandling::MinIntensity => (ZProjection::MinIntensity, 0..=0),
        ZStackHandling::AvgIntensity => (ZProjection::AvgIntensity, 0..=0),
        ZStackHandling::SumIntensity => (ZProjection::SumIntensity, 0..=0),
        ZStackHandling::TakeTheMiddle => (ZProjection::TakeTheMiddle, 0..=0),
    }
}

fn tile_grid(full_width: usize, full_height: usize, tile_size: usize) -> Vec<ImageTile> {
    let x_steps = full_width.div_ceil(tile_size);
    let y_steps = full_height.div_ceil(tile_size);
    let mut tiles = Vec::with_capacity(x_steps * y_steps);
    for y in 0..y_steps {
        for x in 0..x_steps {
            let offset_x = x * tile_size;
            let offset_y = y * tile_size;
            tiles.push(ImageTile {
                offset_x,
                offset_y,
                width: (full_width - offset_x).min(tile_size),
                height: (full_height - offset_y).min(tile_size),
            });
        }
    }
    tiles
}

fn bbox_overlaps_tile(bbox: [u32; 4], tile: &ImageTile) -> bool {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;
    x_min <= tile_x_max && x_max >= tile_x_min && y_min <= tile_y_max && y_max >= tile_y_min
}

/// (x, y) coordinates (in the image's full-resolution grid, `usize`) of every
/// pixel in `object`'s mask that falls within `tile`'s bounds.
fn masked_pixels_in_tile(object: &Object, tile: &ImageTile) -> Vec<(usize, usize)> {
    let [x_min, y_min, x_max, y_max] = object.bbox;
    let tile_x_min = tile.offset_x as u32;
    let tile_y_min = tile.offset_y as u32;
    let tile_x_max = tile.offset_x as u32 + tile.width as u32 - 1;
    let tile_y_max = tile.offset_y as u32 + tile.height as u32 - 1;

    let ix_min = x_min.max(tile_x_min);
    let iy_min = y_min.max(tile_y_min);
    let ix_max = x_max.min(tile_x_max);
    let iy_max = y_max.min(tile_y_max);

    if ix_min > ix_max || iy_min > iy_max {
        return Vec::new();
    }

    let mut samples = Vec::new();
    for y in iy_min..=iy_max {
        for x in ix_min..=ix_max {
            if object.is_part_of(x, y) {
                samples.push((x as usize, y as usize));
            }
        }
    }
    samples
}
