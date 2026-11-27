//! # deep_learning
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Training a deep-learning network (SpotNet; U-Net, StarDist, Cellpose
//! later) on the project's annotations - one job for every network, the
//! model a setting.
//!
//! Shared by every network:
//!
//! - the annotations grouped by the image plane they belong to (as an
//!   analysis uses them, `ObjectMetricSettingsExt::belongs_to`);
//! - the **samples**: the region around each training or validation area
//!   plus a halo of half the network's receptive field (overlapping ones
//!   joined) - not the whole image, so a whole slide with a few areas is
//!   read only there;
//! - each sample read when the network asks for it (`nn`'s `Dataset`: one
//!   at a time, never all in RAM), its areas rasterised into masks;
//! - progress, cancel, the stats and the trained model.
//!
//! Per network only the step from a sample's annotations to the network's
//! labels (SpotNet: its one-pixel annotations as spots).

use crate::ZProjection;
use crate::ai_learning::training_job::{self, TrainingImage};
use crate::ai_learning::utils::mask_pixels_in_tile;
use crate::image::{ImageReader, ImageTile, ReadMode};
use crate::object::annotation_plane::ObjectMetricSettingsExt;
use evanalyzer_cfg::core_types::{
    InternalErrors, NetworkCheck, ObjectClass, TrainingProgressEvent, TrainingStats,
};
use evanalyzer_cfg::settings::images_settings::ZStackHandling;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_nn::NnError;
use evanalyzer_nn::model::spotnet::{Spot, SpotLabels, SpotNetModel, SpotNetTrainParams};
use evanalyzer_nn::model::{Dataset, Model, OwnedImage, Trainable};
use evanalyzer_nn::train::{
    HeldBack, TrainControl, TrainDevice, TrainProgress, TrainStats, Trained,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Which network a job trains, with its settings.
#[derive(Debug, Clone, PartialEq)]
pub enum DeepLearningModel {
    SpotNet(SpotNetTrainParams),
}

/// The annotation classes a training reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepLearningClasses {
    /// What the network learns to find (SpotNet: its one-pixel
    /// annotations are the spots).
    pub objects: ObjectClass,
    /// The training areas: inside every object must be annotated; outside
    /// nothing is learned.
    pub area: ObjectClass,
    /// The validation areas the user drew - never trained on; `None`
    /// without a class for them.
    pub validation: Option<ObjectClass>,
}

/// Trains a deep-learning network on annotated images.
#[derive(Debug, Clone)]
pub struct DeepLearningTrainingJob {
    pub model: DeepLearningModel,
    /// A model file to fine-tune; `None`: new weights.
    pub start: Option<PathBuf>,
    pub images: Vec<TrainingImage>,
    pub classes: DeepLearningClasses,
    /// The image channels the network gets, in its input order (SpotNet:
    /// the spot channel, then the reference channel if the model has one).
    pub channels: Vec<i32>,
    /// How the analyses handle z-stacks: which plane each annotation is
    /// trained on (as the pixel classifier).
    pub z_stack_handling: ZStackHandling,
    /// The same images, annotations, settings and seed give the same model.
    pub seed: u64,
}

/// A trained model, ready to be saved.
pub trait TrainedModel: Send {
    /// Writes its model file (`.evanet`).
    fn save(&self, path: &Path) -> Result<(), InternalErrors>;
}

impl<M: Trainable> TrainedModel for M {
    fn save(&self, path: &Path) -> Result<(), InternalErrors> {
        Trainable::save(self, path).map_err(nn_error)
    }
}

/// A finished training: the model to save and what the panel shows.
pub struct TrainedNetwork {
    pub model: Box<dyn TrainedModel>,
    pub stats: TrainingStats,
}

impl std::fmt::Debug for TrainedNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainedNetwork")
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// One sample: a rectangle of one image plane and the annotations on it.
#[derive(Debug, Clone)]
struct Region {
    image: usize,
    /// The plane (`None`: the z-projection) and frame.
    z: Option<i32>,
    t: i32,
    tile: ImageTile,
    /// The annotations of the job's classes reaching into the rectangle.
    annotations: Vec<ObjectMetricSettings>,
}

impl DeepLearningTrainingJob {
    /// Runs synchronously on the calling thread - use [`Self::run_async`]
    /// to run in the background.
    pub fn run(
        &self,
        progress: Sender<TrainingProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<TrainedNetwork, InternalErrors> {
        if self.z_stack_handling == ZStackHandling::Volume3d {
            return Err(InternalErrors::InvalidArgument(
                "Training on z-stacks as volumes (3D) isn't supported yet".into(),
            ));
        }
        match &self.model {
            DeepLearningModel::SpotNet(params) => {
                let halo = self.spotnet_halo(params)?;
                self.train::<SpotNetModel>(params, halo, spot_labels, progress, &cancel)
            }
        }
    }

    /// Runs in a background thread (progress channel + shared cancel flag,
    /// as the classifier trainings).
    pub fn run_async(
        self,
    ) -> (
        JoinHandle<Result<TrainedNetwork, InternalErrors>>,
        Receiver<TrainingProgressEvent>,
        Arc<AtomicBool>,
    ) {
        training_job::spawn_training_job(self, |job, progress, cancel| job.run(progress, cancel))
    }

    /// Half the receptive field of the network that will be trained: the
    /// file's when fine-tuning (its architecture stays), else the settings'.
    fn spotnet_halo(&self, params: &SpotNetTrainParams) -> Result<usize, InternalErrors> {
        let network = match &self.start {
            Some(file) => {
                <SpotNetModel as Model>::header(file)
                    .map_err(nn_error)?
                    .settings
                    .network
            }
            None => params.network.clone(),
        };
        Ok(network.receptive_field() / 2)
    }

    /// The shared part: the samples, nn's training of `M`, the result.
    /// `labels` makes a sample's labels for `M` from its annotations.
    fn train<M: Trainable>(
        &self,
        params: &M::TrainParams,
        halo: usize,
        labels: fn(&Region, &DeepLearningClasses) -> M::Target,
        progress: Sender<TrainingProgressEvent>,
        cancel: &AtomicBool,
    ) -> Result<TrainedNetwork, InternalErrors> {
        let regions = self.regions(halo)?;
        if regions.is_empty() {
            return Err(InternalErrors::InvalidArgument(
                "No training area: draw at least one around annotated objects, in the class \
                 chosen as training area."
                    .into(),
            ));
        }
        let _ = progress.send(TrainingProgressEvent::Started {
            total: regions.len(),
        });
        let samples = Samples {
            job: self,
            regions: &regions,
            labels,
        };
        let _ = progress.send(TrainingProgressEvent::Training);
        let report = |p: TrainProgress| {
            if let TrainProgress::Epoch {
                epoch,
                total_epochs,
                train_loss,
                val_loss,
            } = p
            {
                let _ = progress.send(TrainingProgressEvent::Epoch {
                    epoch,
                    total_epochs,
                    train_loss,
                    val_loss,
                });
            }
        };
        let control = TrainControl {
            progress: &report,
            cancel,
            seed: self.seed,
            device: TrainDevice::Cpu,
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        };
        let trained = M::train(self.start.as_deref(), &samples, params, &control);
        if cancel.load(Ordering::Relaxed) {
            return Err(InternalErrors::Cancelled);
        }
        let Trained {
            model,
            stats,
            held_back,
        } = trained.map_err(nn_error)?;
        let stats = network_stats(stats, held_back);
        let _ = progress.send(TrainingProgressEvent::Finished {
            stats: stats.clone(),
        });
        Ok(TrainedNetwork {
            model: Box::new(model),
            stats,
        })
    }

    /// The samples: per image and plane, the rectangles around its
    /// training and validation areas plus `halo` (within the image),
    /// overlapping ones joined.
    fn regions(&self, halo: usize) -> Result<Vec<Region>, InternalErrors> {
        let per_z_plane = self.z_stack_handling.z_projection() == ZProjection::None;
        let mut regions = Vec::new();
        for (image, training_image) in self.images.iter().enumerate() {
            let planes = self.annotations_by_plane(&training_image.labeled_objects, per_z_plane);
            if planes.is_empty() {
                continue;
            }
            let (width, height) = image_size(training_image)?;
            for ((z, t), annotations) in planes {
                let areas: Vec<[usize; 4]> = annotations
                    .iter()
                    .filter(|a| self.is_area(a))
                    .map(|a| around(a.bbox, halo, width, height))
                    .collect();
                for [x0, y0, x1, y1] in join_overlapping(areas) {
                    let tile = ImageTile {
                        offset_x: x0,
                        offset_y: y0,
                        width: x1 - x0 + 1,
                        height: y1 - y0 + 1,
                    };
                    let annotations = annotations
                        .iter()
                        .filter(|a| overlaps(a.bbox, &tile))
                        .cloned()
                        .collect();
                    regions.push(Region {
                        image,
                        z,
                        t,
                        tile,
                        annotations,
                    });
                }
            }
        }
        Ok(regions)
    }

    /// The annotations of the job's classes by the plane they are trained
    /// on: `(Some(z), t)` plane by plane, `(None, t)` on the z-projection.
    fn annotations_by_plane(
        &self,
        annotations: &[ObjectMetricSettings],
        per_z_plane: bool,
    ) -> BTreeMap<(Option<i32>, i32), Vec<ObjectMetricSettings>> {
        let classes = self.classes;
        let mut planes: BTreeMap<(Option<i32>, i32), Vec<ObjectMetricSettings>> = BTreeMap::new();
        for a in annotations.iter().filter(|a| !a.exclude_from_training) {
            let ours = a.object_class.contains(&classes.objects)
                || a.object_class.contains(&classes.area)
                || classes
                    .validation
                    .is_some_and(|v| a.object_class.contains(&v));
            let z = per_z_plane.then_some(a.z_stack);
            if ours && a.belongs_to(z, a.t_stack) {
                planes.entry((z, a.t_stack)).or_default().push(a.clone());
            }
        }
        planes
    }

    /// A training or validation area.
    fn is_area(&self, a: &ObjectMetricSettings) -> bool {
        a.object_class.contains(&self.classes.area)
            || self
                .classes
                .validation
                .is_some_and(|v| a.object_class.contains(&v))
    }
}

/// The full-resolution size of a training image's series.
fn image_size(training_image: &TrainingImage) -> Result<(usize, usize), InternalErrors> {
    let cannot = || {
        InternalErrors::InvalidArgument(format!(
            "The training image '{}' can't be read",
            training_image.path.display()
        ))
    };
    let reader = ImageReader::new(&training_image.path, ReadMode::Default).map_err(|_| cannot())?;
    let meta = reader.get_image_meta();
    let pyramid = meta
        .series
        .get(&training_image.series)
        .and_then(|s| s.resolutions.get(&0))
        .ok_or_else(cannot)?;
    Ok((pyramid.width as usize, pyramid.height as usize))
}

/// `bbox` (`[x_min, y_min, x_max, y_max]`, inclusive) grown by `halo` on
/// every side, within a `width x height` image.
fn around(bbox: [u32; 4], halo: usize, width: usize, height: usize) -> [usize; 4] {
    let [x0, y0, x1, y1] = bbox.map(|v| v as usize);
    [
        x0.saturating_sub(halo),
        y0.saturating_sub(halo),
        (x1 + halo).min(width.saturating_sub(1)),
        (y1 + halo).min(height.saturating_sub(1)),
    ]
}

/// Rectangles (inclusive) with every overlapping pair joined into their
/// bounding box, until none overlap.
fn join_overlapping(mut rects: Vec<[usize; 4]>) -> Vec<[usize; 4]> {
    let touch = |a: &[usize; 4], b: &[usize; 4]| {
        a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
    };
    let mut joined = true;
    while joined {
        joined = false;
        'outer: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                if touch(&rects[i], &rects[j]) {
                    let b = rects.swap_remove(j);
                    let a = &mut rects[i];
                    *a = [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ];
                    joined = true;
                    break 'outer;
                }
            }
        }
    }
    rects.sort_unstable();
    rects
}

fn overlaps(bbox: [u32; 4], tile: &ImageTile) -> bool {
    crate::ai_learning::utils::bbox_overlaps_tile(bbox, tile)
}

/// The job's regions as nn's training samples - each read when asked for.
struct Samples<'a, T> {
    job: &'a DeepLearningTrainingJob,
    regions: &'a [Region],
    labels: fn(&Region, &DeepLearningClasses) -> T,
}

impl<T> Dataset<T> for Samples<'_, T> {
    fn len(&self) -> usize {
        self.regions.len()
    }

    /// The region's channels (in the job's order) and its labels.
    fn get(&self, index: usize) -> Result<(OwnedImage, T), NnError> {
        let region = &self.regions[index];
        let image =
            read_region(self.job, region).map_err(|e| NnError::InvalidInput(e.to_string()))?;
        Ok((image, (self.labels)(region, &self.job.classes)))
    }
}

/// The job's channels of `region`, one plane after the other.
fn read_region(
    job: &DeepLearningTrainingJob,
    region: &Region,
) -> Result<OwnedImage, InternalErrors> {
    let training_image = &job.images[region.image];
    let path = training_image.path.display();
    let reader = ImageReader::new(&training_image.path, ReadMode::Default).map_err(|_| {
        InternalErrors::InvalidArgument(format!("The training image '{path}' can't be read"))
    })?;
    let loaded = reader.read_image_tile_combined(
        training_image.series,
        0,
        job.z_stack_handling.z_projection(),
        &region.z.map(|z| z..=z),
        region.t,
        Some(&job.channels),
        &region.tile,
    )?;
    let (width, height) = (region.tile.width, region.tile.height);
    let mut data = Vec::with_capacity(job.channels.len() * width * height);
    for &channel in &job.channels {
        let Some(found) = loaded.iter().find(|l| l.c_stack == channel) else {
            return Err(InternalErrors::InvalidArgument(format!(
                "The training image '{path}' has no channel {channel}"
            )));
        };
        let pixels = match found.image.as_ref() {
            crate::image::ImageContainer::F32Gray(img) => img.as_slice(),
            _ => {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Channel {channel} of '{path}' is no grey image - a network is trained on \
                     grey channels"
                )));
            }
        };
        if pixels.len() != width * height {
            return Err(InternalErrors::Internal(format!(
                "Channel {channel} of '{path}': {} pixels read for a {width} x {height} region",
                pixels.len()
            )));
        }
        data.extend_from_slice(pixels);
    }
    Ok(OwnedImage {
        width,
        height,
        channels: job.channels.len(),
        data,
    })
}

/// `true` on the pixels of `region` covered by an annotation of `class`.
fn area_mask(region: &Region, class: ObjectClass) -> Vec<bool> {
    let tile = &region.tile;
    let mut mask = vec![false; tile.width * tile.height];
    for a in region
        .annotations
        .iter()
        .filter(|a| a.object_class.contains(&class))
    {
        for (x, y) in mask_pixels_in_tile(&a.mask_data, a.bbox, tile) {
            mask[(y - tile.offset_y) * tile.width + (x - tile.offset_x)] = true;
        }
    }
    mask
}

/// SpotNet's labels of a region: the one-pixel annotations of the objects
/// class as spots (at their pixel's centre), the training and validation
/// areas as masks.
fn spot_labels(region: &Region, classes: &DeepLearningClasses) -> SpotLabels {
    let tile = &region.tile;
    let spots = region
        .annotations
        .iter()
        .filter(|a| a.object_class.contains(&classes.objects))
        .filter(|a| a.bbox[0] == a.bbox[2] && a.bbox[1] == a.bbox[3])
        .filter(|a| overlaps(a.bbox, tile))
        .map(|a| Spot {
            x: (a.bbox[0] as usize - tile.offset_x) as f32 + 0.5,
            y: (a.bbox[1] as usize - tile.offset_y) as f32 + 0.5,
        })
        .collect();
    SpotLabels {
        width: tile.width,
        height: tile.height,
        spots,
        area: area_mask(region, classes.area),
        validation: classes.validation.map_or_else(
            || vec![false; tile.width * tile.height],
            |v| area_mask(region, v),
        ),
    }
}

/// nn's stats and check as the panel's.
fn network_stats(stats: TrainStats, held_back: HeldBack) -> TrainingStats {
    let (epochs_run, total_epochs, final_train_loss, final_val_loss) = match stats {
        TrainStats::Epochs {
            epochs_run,
            total_epochs,
            final_train_loss,
            final_val_loss,
            ..
        } => (epochs_run, total_epochs, final_train_loss, final_val_loss),
        _ => (0, 0, f32::NAN, None),
    };
    let check = match held_back {
        HeldBack::Scored { epoch, score } => NetworkCheck::Scored {
            epoch,
            annotated: score.annotated,
            found: score.found,
            matched: score.matched,
        },
        HeldBack::Off => NetworkCheck::Off,
        HeldBack::TooFewImages { images } => NetworkCheck::TooFewImages { images },
    };
    TrainingStats::Network {
        epochs_run,
        total_epochs,
        final_train_loss,
        final_val_loss,
        check,
    }
}

fn nn_error(e: NnError) -> InternalErrors {
    match e {
        NnError::Cancelled => InternalErrors::Cancelled,
        NnError::InvalidInput(m) => InternalErrors::InvalidArgument(m),
        other => InternalErrors::Generic(other.to_string()),
    }
}

#[cfg(test)]
mod tests;
