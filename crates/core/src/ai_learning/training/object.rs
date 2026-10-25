//! # object
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ai_learning::model::SavedClassifier;
use crate::ai_learning::training_job::{self, TrainingImage};
use crate::image::{ImageReader, ReadMode};
use crate::object::regions::{measure_on_planes, ring_of};
use crate::{Intensity, Object};
use evanalyzer_cfg::core_types::TrainingProgressEvent;
use evanalyzer_cfg::core_types::ZProjection;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel,
};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Assembles one feature vector for `object`, in `spec.metrics` order.
/// `ring` holds the intensities of the ring around it (see
/// [`ring_of`]) - empty when no ring feature is chosen. A channel that
/// wasn't measured counts as 0, the same in training and in the step.
pub fn compute_object_features(
    object: &Object,
    ring: &IndexMap<i32, Intensity>,
    spec: &AiLearningObjectFeatureSettings,
) -> Vec<f32> {
    let of = |channel: &i32, pick: fn(&Intensity) -> f64| -> f32 {
        object.intensities.get(channel).map_or(0.0, pick) as f32
    };
    let ring_avg = |channel: &i32| ring.get(channel).map_or(0.0, |i| i.avg_intensity) as f32;
    spec.metrics
        .iter()
        .map(|metric| match metric {
            ObjectMetric::Area => object.area as f32,
            ObjectMetric::Perimeter => object.get_perimeter(),
            ObjectMetric::Circularity => object.circularity(),
            ObjectMetric::Solidity => object.get_solidity(),
            ObjectMetric::AspectRatio => object.get_aspect_ratio(),
            ObjectMetric::Roundness => object.get_roundness(object.get_perimeter()),
            ObjectMetric::Compactness => object.get_compactness(object.get_perimeter()),
            ObjectMetric::FeretDiameter => object.get_feret_diameter(),
            ObjectMetric::MinFeretDiameter => object.get_min_feret_diameter(),
            ObjectMetric::EllipseMajor => object.get_ellipse().major,
            ObjectMetric::EllipseMinor => object.get_ellipse().minor,
            ObjectMetric::EllipseAngle => object.get_ellipse().angle,
            ObjectMetric::Eccentricity => object.get_ellipse().eccentricity,
            ObjectMetric::TouchesEdge => {
                if object.touches_edge {
                    1.0
                } else {
                    0.0
                }
            }
            ObjectMetric::IntensitySum(c) => of(c, |i| i.sum_intensity),
            ObjectMetric::IntensityMin(c) => of(c, |i| i.min_intensity),
            ObjectMetric::IntensityMax(c) => of(c, |i| i.max_intensity),
            ObjectMetric::IntensityAvg(c) => of(c, |i| i.avg_intensity),
            ObjectMetric::IntensityStd(c) => of(c, |i| i.std_intensity),
            ObjectMetric::IntensityMedian(c) => of(c, |i| i.median_intensity),
            ObjectMetric::IntensityLowerQuartile(c) => of(c, |i| i.lower_quartile_intensity),
            ObjectMetric::IntensityUpperQuartile(c) => of(c, |i| i.upper_quartile_intensity),
            ObjectMetric::IntensityMad(c) => of(c, |i| i.mad_intensity),
            ObjectMetric::RingIntensityAvg(c) => ring_avg(c),
            ObjectMetric::RingContrast(c) => of(c, |i| i.avg_intensity) - ring_avg(c),
        })
        .collect()
}

/// Trains an object classifier from the labeled objects of `images`. Like
/// `PixelTrainingJob`, it reads the images: every intensity feature is
/// measured on demand under each object (on its own z and t plane, every
/// channel at full resolution) and, for the ring features, in the ring
/// around it ([`ring_of`]) - the same measuring the AI Object Classifier
/// step does, so training and the pipeline see the same values. Only the
/// regions with objects are read; with shape features alone no image is
/// opened.
///
/// `settings.classifier` must be `AiLearningClassifierSettings::Object` -
/// `run` returns an error otherwise.
pub struct ObjectTrainingJob {
    pub settings: AiLearningSettings,
    /// The labeled objects, with the image each is on.
    pub images: Vec<TrainingImage>,
}

impl ObjectTrainingJob {
    /// How many objects the job trains on (before skipping any).
    pub fn object_count(&self) -> usize {
        self.images.iter().map(|i| i.labeled_objects.len()).sum()
    }

    /// Runs synchronously on the calling thread - use `run_async` to run in
    /// the background the way pipeline execution does.
    pub fn run(
        &self,
        progress: Sender<TrainingProgressEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<SavedClassifier, InternalErrors> {
        let AiLearningClassifierSettings::Object {
            feature_spec,
            class_labels,
        } = &self.settings.classifier
        else {
            return Err(InternalErrors::Internal(
                "ObjectTrainingJob requires an Object classifier configuration".to_string(),
            ));
        };
        let needs_images = feature_spec.metrics.iter().any(|m| m.channel().is_some());
        let needs_ring = feature_spec.metrics.iter().any(ObjectMetric::uses_ring);

        let total = self.object_count();
        let _ = progress.send(TrainingProgressEvent::Started { total });

        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(total);
        let mut labels: Vec<usize> = Vec::with_capacity(total);
        let mut index = 0;

        for image in &self.images {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }
            let objects = &image.labeled_objects;
            let measured = if needs_images {
                match measure_image(image, needs_ring.then_some(feature_spec.ring_width)) {
                    Ok(measured) => Some(measured),
                    Err(_) => {
                        let _ = progress.send(TrainingProgressEvent::ImageFailed {
                            path: image.path.clone(),
                        });
                        None
                    }
                }
            } else {
                Some(Measured::default())
            };

            for (k, object_settings) in objects.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InternalErrors::Cancelled);
                }
                let skip = |reason: String| {
                    let _ = progress.send(TrainingProgressEvent::ObjectSkipped { index, reason });
                };
                match (
                    resolve_label(class_labels, &object_settings.object_class),
                    &measured,
                ) {
                    (Err(reason), _) => skip(reason.to_string()),
                    (Ok(_), None) => skip("its image can't be read".to_string()),
                    (Ok(label), Some(measured)) => {
                        let mut object = Object::from_object_settings(object_settings.clone());
                        let no_ring = IndexMap::new();
                        let ring = if needs_images {
                            object.intensities = measured.objects[k].clone();
                            measured.rings.get(k).unwrap_or(&no_ring)
                        } else {
                            &no_ring
                        };
                        rows.push(compute_object_features(&object, ring, feature_spec));
                        labels.push(label);
                    }
                }
                let _ = progress.send(TrainingProgressEvent::ItemCompleted { index, total });
                index += 1;
            }
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

/// The intensities of an image's objects (in order) and of the rings
/// around them (empty without ring features).
#[derive(Default)]
struct Measured {
    objects: Vec<IndexMap<i32, Intensity>>,
    rings: Vec<IndexMap<i32, Intensity>>,
}

/// Measures the objects of `image` - and, with a `ring_width`, the rings
/// around them - on the image file.
fn measure_image(
    image: &TrainingImage,
    ring_width: Option<u32>,
) -> Result<Measured, InternalErrors> {
    let reader = ImageReader::new(&image.path, ReadMode::Default)?;
    let meta = reader.get_image_meta();
    let full = meta
        .series
        .get(&image.series)
        .and_then(|s| s.resolutions.get(&0))
        .ok_or_else(|| {
            InternalErrors::Generic(format!("The image has no series {}", image.series))
        })?;
    let (width, height) = (full.width as usize, full.height as usize);
    let objects = &image.labeled_objects;
    let mut masks = objects.clone();
    if let Some(ring_width) = ring_width {
        masks.extend(
            objects
                .iter()
                .map(|object| ring_of(object, ring_width, width, height)),
        );
    }
    let mut sums: Vec<IndexMap<i32, Intensity>> =
        measure_on_planes(&masks, width, height, |z, t, tile| {
            reader.read_image_tile_combined(
                image.series,
                0,
                ZProjection::None,
                &Some(z..=z),
                t,
                None,
                &tile,
            )
        })?
        .into_iter()
        .map(|sum| sum.finish())
        .collect();
    let rings = sums.split_off(objects.len());
    Ok(Measured {
        objects: sums,
        rings,
    })
}

/// An object's `object_class` set can contain multiple classes (colocalization,
/// multi-class assignment) - only unambiguous objects (exactly one class that's
/// also one of this model's configured `class_labels`) are usable as training
/// data for a single-label classifier.
fn resolve_label(
    class_labels: &[ObjectClassLabel],
    object_class: &HashSet<ObjectClass>,
) -> Result<usize, &'static str> {
    let mut matches = class_labels
        .iter()
        .enumerate()
        .filter(|(_, l)| object_class.contains(&l.class));
    let Some((index, _)) = matches.next() else {
        return Err("object's class does not match any of the model's configured class labels");
    };
    if matches.next().is_some() {
        return Err("object matches more than one of the model's configured class labels");
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_learning::model::Classifier;
    use crate::object::{Intensity, ObjectInit};
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningSettings, KnnSettings, PixelClassLabel,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

    // -- compute_object_features ---------------------------------------------

    #[test]
    fn compute_object_features_reads_geometry_and_intensity_metrics_in_order() {
        let mut intensities = IndexMap::new();
        intensities.insert(
            0,
            Intensity {
                sum_intensity: 40.0,
                min_intensity: 1.0,
                max_intensity: 9.0,
                avg_intensity: 5.0,
                ..Default::default()
            },
        );
        let object = Object::new(ObjectInit {
            area: 42,
            touches_edge: true,
            intensities,
            ..Default::default()
        });
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![
                ObjectMetric::Area,
                ObjectMetric::TouchesEdge,
                ObjectMetric::IntensityAvg(0),
                ObjectMetric::IntensitySum(0),
            ],
            ..Default::default()
        };

        let features = compute_object_features(&object, &IndexMap::new(), &spec);

        assert_eq!(features, vec![42.0, 1.0, 5.0, 40.0]);
    }

    #[test]
    fn compute_object_features_defaults_missing_intensity_channel_to_zero() {
        let object = Object::new(ObjectInit::default());
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![ObjectMetric::IntensityAvg(3)],
            ..Default::default()
        };

        assert_eq!(
            compute_object_features(&object, &IndexMap::new(), &spec),
            vec![0.0]
        );
    }

    // -- resolve_label ---------------------------------------------------------

    fn class_label(id: u32, name: &str) -> ObjectClassLabel {
        ObjectClassLabel {
            class: ObjectClass::Global(id),
            name: name.into(),
        }
    }

    #[test]
    fn resolve_label_matches_the_one_configured_class() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Global(2)]);
        assert_eq!(resolve_label(&labels, &object_class), Ok(1));
    }

    #[test]
    fn resolve_label_errors_when_no_class_matches() {
        let labels = vec![class_label(1, "Cell")];
        let object_class = HashSet::from([ObjectClass::Global(99)]);
        assert!(resolve_label(&labels, &object_class).is_err());
    }

    #[test]
    fn resolve_label_errors_when_multiple_classes_match() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Global(1), ObjectClass::Global(2)]);
        assert!(resolve_label(&labels, &object_class).is_err());
    }

    // -- ObjectTrainingJob::run -------------------------------------------------

    fn labeled_object(id: u128, class: ObjectClass, area: usize) -> ObjectMetricSettings {
        ObjectMetricSettings {
            id: ObjectId(id),
            object_class: HashSet::from([class]),
            area,
            ..Default::default()
        }
    }

    fn object_job(objects: Vec<ObjectMetricSettings>) -> ObjectTrainingJob {
        ObjectTrainingJob {
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData::default(),
                backend: AiLearningBackendSettings::Knn(KnnSettings {
                    k: 2,
                    ..Default::default()
                }),
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings {
                        metrics: vec![ObjectMetric::Area],
                        ..Default::default()
                    },
                    class_labels: vec![class_label(1, "Small"), class_label(2, "Big")],
                },
            },
            images: vec![TrainingImage {
                path: "/no/such/image.tif".into(),
                series: 0,
                labeled_objects: objects,
            }],
        }
    }

    #[test]
    fn run_trains_a_classifier_from_labeled_objects() {
        let job = object_job(vec![
            labeled_object(1, ObjectClass::Global(1), 1),
            labeled_object(2, ObjectClass::Global(1), 2),
            labeled_object(3, ObjectClass::Global(2), 100),
            labeled_object(4, ObjectClass::Global(2), 101),
        ]);
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let saved = job.run(tx, cancel).unwrap();

        assert!(matches!(saved.classifier, Classifier::Knn(_)));
    }

    #[test]
    fn run_skips_objects_with_an_unresolvable_class_and_reports_it() {
        let job = object_job(vec![
            labeled_object(1, ObjectClass::Global(1), 1),
            labeled_object(2, ObjectClass::Global(99), 2), // not in class_labels
            labeled_object(3, ObjectClass::Global(2), 100),
        ]);
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        job.run(tx, cancel).unwrap();
        let events: Vec<TrainingProgressEvent> = rx.try_iter().collect();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::ObjectSkipped { index: 1, .. }))
        );
    }

    /// Shape features need no image: the job above trains although its
    /// image doesn't exist. An intensity feature reads it - an image that
    /// can't be read is reported and its objects skipped.
    #[test]
    fn an_intensity_feature_reads_the_image_and_skips_the_objects_of_a_missing_one() {
        let mut job = object_job(vec![
            labeled_object(1, ObjectClass::Global(1), 1),
            labeled_object(2, ObjectClass::Global(2), 100),
        ]);
        let AiLearningClassifierSettings::Object { feature_spec, .. } =
            &mut job.settings.classifier
        else {
            unreachable!()
        };
        feature_spec.metrics = vec![ObjectMetric::IntensityMedian(0)];
        let (tx, rx) = std::sync::mpsc::channel();
        let result = job.run(tx, Arc::new(AtomicBool::new(false)));
        assert!(result.is_err(), "nothing left to train on");
        let events: Vec<TrainingProgressEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainingProgressEvent::ImageFailed { .. }))
        );
        let skipped: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                TrainingProgressEvent::ObjectSkipped { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(skipped, [0, 1]);
    }

    #[test]
    fn the_ring_features_are_the_rings_mean_and_the_objects_mean_minus_it() {
        let intensity = |avg_intensity| Intensity {
            avg_intensity,
            ..Default::default()
        };
        let object = Object::new(ObjectInit {
            intensities: IndexMap::from([(0, intensity(10.0))]),
            ..Default::default()
        });
        let ring = IndexMap::from([(0, intensity(4.0))]);
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![
                ObjectMetric::RingIntensityAvg(0),
                ObjectMetric::RingContrast(0),
                ObjectMetric::RingIntensityAvg(1),
            ],
            ..Default::default()
        };
        assert_eq!(
            compute_object_features(&object, &ring, &spec),
            [4.0, 6.0, 0.0]
        );
    }

    /// Training measures an annotation and its ring from the image file;
    /// the AI Object Classifier step from the pipeline's channel images.
    /// Both must give the same features, or the model learns other values
    /// than it is applied to.
    #[test]
    fn training_and_the_pipeline_step_compute_the_same_features() {
        use crate::algos::ai_object_features_of as features_of;
        use crate::image::ImageReader;
        use crate::pipeline::pipeline_cache::GlobalPipelineCache;
        use bitvec::prelude::{BitVec, Lsb0};
        use evanalyzer_cfg::core_types::{ImagePlane, ImageTile};
        use kornia_image::ImageSize;

        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/slice_Z0_C0_T0.tif"
        ));
        let reader = ImageReader::new(&path, ReadMode::Default).unwrap();
        let full = reader.get_image_meta().series[&0].resolutions[&0].clone();
        let size = ImageSize {
            width: full.width as usize,
            height: full.height as usize,
        };
        // A disc of radius 6 in the middle.
        let side = 13u32;
        let mask: BitVec<u64, Lsb0> = (0..side * side)
            .map(|k| {
                let (x, y) = ((k % side) as i32 - 6, (k / side) as i32 - 6);
                x * x + y * y <= 36
            })
            .collect();
        let (x0, y0) = (size.width as u32 / 2, size.height as u32 / 2);
        let annotation = Object::from_mask_geometry(
            &size,
            mask,
            [x0, y0, x0 + side - 1, y0 + side - 1],
            ImagePlane { z: 0, c: 0, t: 0 },
            HashSet::from([ObjectClass::Global(1)]),
        )
        .to_object_settings();
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![
                ObjectMetric::Area,
                ObjectMetric::IntensityAvg(0),
                ObjectMetric::IntensityMedian(0),
                ObjectMetric::IntensityStd(0),
                ObjectMetric::RingIntensityAvg(0),
                ObjectMetric::RingContrast(0),
            ],
            ring_width: 4,
        };

        // Training.
        let measured = measure_image(
            &TrainingImage {
                path: path.clone(),
                series: 0,
                labeled_objects: vec![annotation.clone()],
            },
            Some(spec.ring_width),
        )
        .unwrap();
        let mut trained = Object::from_object_settings(annotation.clone());
        trained.intensities = measured.objects[0].clone();
        let in_training = compute_object_features(&trained, &measured.rings[0], &spec);

        // The step: the channel images in the cache, the object measured there.
        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: size.width,
            height: size.height,
        };
        let mut cache = GlobalPipelineCache::default();
        for channel in reader
            .read_image_tile_combined(0, 0, ZProjection::None, &Some(0..=0), 0, None, &tile)
            .unwrap()
        {
            cache.add_to_channel_cache(channel.image, channel.c_stack, tile);
        }
        let mut objects = [Object::from_object_settings(annotation)];
        Object::measure_intensities_of(&mut objects, &cache);
        let in_step = features_of(&[&objects[0]], &spec, &size, &cache);

        assert!(
            in_training[4] > 0.0,
            "the ring was measured: {in_training:?}"
        );
        assert_eq!(in_training, in_step[0]);
    }

    /// The spread goes into an annotation and comes back out as the
    /// features.
    #[test]
    fn the_spread_round_trips_through_an_annotation() {
        let mut intensity = crate::Intensity {
            avg_intensity: 2.5,
            ..Default::default()
        };
        intensity.set_spread(&mut [1.0, 2.0, 3.0, 4.0]);
        let stored = intensity.to_settings();
        assert_eq!(stored.median_intensity, Some(3.0));
        let mut annotation = labeled_object(1, ObjectClass::Global(1), 4);
        annotation.intensities.insert(0, stored);
        let object = Object::from_object_settings(annotation);
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![
                ObjectMetric::IntensityMedian(0),
                ObjectMetric::IntensityLowerQuartile(0),
                ObjectMetric::IntensityUpperQuartile(0),
                ObjectMetric::IntensityMad(0),
            ],
            ..Default::default()
        };
        assert_eq!(
            compute_object_features(&object, &IndexMap::new(), &spec),
            [3.0, 2.0, 4.0, 1.0]
        );
        // Not measured: none in the project file.
        let unmeasured = crate::Intensity::default().to_settings();
        assert_eq!(unmeasured.std_intensity, None);
        let json = serde_json::to_string(&unmeasured).unwrap();
        assert!(!json.contains("stdIntensity"), "{json}");
    }

    #[test]
    fn run_returns_cancelled_when_the_flag_is_already_set() {
        let job = object_job(vec![labeled_object(1, ObjectClass::Global(1), 1)]);
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(true));

        let err = job.run(tx, cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Cancelled));
    }

    #[test]
    fn run_errors_for_a_pixel_classifier_configuration() {
        let mut job = object_job(vec![]);
        job.settings.classifier = AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings {
                channels: vec![],
                image_channels: Vec::new(),
                image_channel_names: Vec::new(),
            },
            input_color: Default::default(),
            class_labels: vec![PixelClassLabel {
                class: evanalyzer_cfg::core_types::SegmentationClass(1),
                name: "X".into(),
            }],
        };
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let err = job.run(tx, cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Internal(_)));
    }
}
