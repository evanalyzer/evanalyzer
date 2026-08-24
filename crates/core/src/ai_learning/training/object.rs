use crate::Object;
use crate::ai_learning::model::SavedClassifier;
use crate::ai_learning::training_job::{self, TrainingProgressEvent};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, AiLearningSettings, ObjectClassLabel,
};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Assembles one feature vector for `object`, in `spec.metrics` order.
pub fn compute_object_features(
    object: &Object,
    spec: &AiLearningObjectFeatureSettings,
) -> Vec<f32> {
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
            ObjectMetric::IntensitySum(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.sum_intensity as f32)
                .unwrap_or(0.0),
            ObjectMetric::IntensityMin(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.min_intensity)
                .unwrap_or(0.0),
            ObjectMetric::IntensityMax(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.max_intensity)
                .unwrap_or(0.0),
            ObjectMetric::IntensityAvg(channel) => object
                .intensities
                .get(channel)
                .map(|i| i.avg_intensity)
                .unwrap_or(0.0),
        })
        .collect()
}

/// Trains an object classifier from a flat list of already-labeled objects -
/// unlike `PixelTrainingJob`, no image I/O is involved: every `ObjectMetric`
/// is either pure mask geometry or an intensity statistic already computed
/// and stored on the object during segmentation (see `Object::intensities`),
/// so this just resolves each object's class and fits the model.
///
/// `settings.classifier` must be `AiLearningClassifierSettings::Object` -
/// `run` returns an error otherwise.
pub struct ObjectTrainingJob {
    pub settings: AiLearningSettings,
    pub objects: Vec<ObjectMetricSettings>,
}

impl ObjectTrainingJob {
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

        let _ = progress.send(TrainingProgressEvent::Started {
            total: self.objects.len(),
        });

        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(self.objects.len());
        let mut labels: Vec<usize> = Vec::with_capacity(self.objects.len());

        for (index, object_settings) in self.objects.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(InternalErrors::Cancelled);
            }

            match resolve_label(class_labels, &object_settings.object_class) {
                Ok(label) => {
                    let object = Object::from_object_settings(object_settings.clone());
                    rows.push(compute_object_features(&object, feature_spec));
                    labels.push(label);
                }
                Err(reason) => {
                    let _ = progress.send(TrainingProgressEvent::ObjectSkipped {
                        index,
                        reason: reason.to_string(),
                    });
                }
            }

            let _ = progress.send(TrainingProgressEvent::ItemCompleted {
                index,
                total: self.objects.len(),
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
    use indexmap::IndexMap;

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
                pixel_values: vec![],
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
        };

        let features = compute_object_features(&object, &spec);

        assert_eq!(features, vec![42.0, 1.0, 5.0, 40.0]);
    }

    #[test]
    fn compute_object_features_defaults_missing_intensity_channel_to_zero() {
        let object = Object::new(ObjectInit::default());
        let spec = AiLearningObjectFeatureSettings {
            metrics: vec![ObjectMetric::IntensityAvg(3)],
        };

        assert_eq!(compute_object_features(&object, &spec), vec![0.0]);
    }

    // -- resolve_label ---------------------------------------------------------

    fn class_label(id: u32, name: &str) -> ObjectClassLabel {
        ObjectClassLabel {
            class: ObjectClass::Valid(id),
            name: name.into(),
        }
    }

    #[test]
    fn resolve_label_matches_the_one_configured_class() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Valid(2)]);
        assert_eq!(resolve_label(&labels, &object_class), Ok(1));
    }

    #[test]
    fn resolve_label_errors_when_no_class_matches() {
        let labels = vec![class_label(1, "Cell")];
        let object_class = HashSet::from([ObjectClass::Valid(99)]);
        assert!(resolve_label(&labels, &object_class).is_err());
    }

    #[test]
    fn resolve_label_errors_when_multiple_classes_match() {
        let labels = vec![class_label(1, "Cell"), class_label(2, "Background")];
        let object_class = HashSet::from([ObjectClass::Valid(1), ObjectClass::Valid(2)]);
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
                    },
                    class_labels: vec![class_label(1, "Small"), class_label(2, "Big")],
                },
            },
            objects,
        }
    }

    #[test]
    fn run_trains_a_classifier_from_labeled_objects() {
        let job = object_job(vec![
            labeled_object(1, ObjectClass::Valid(1), 1),
            labeled_object(2, ObjectClass::Valid(1), 2),
            labeled_object(3, ObjectClass::Valid(2), 100),
            labeled_object(4, ObjectClass::Valid(2), 101),
        ]);
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        let saved = job.run(tx, cancel).unwrap();

        assert!(matches!(saved.classifier, Classifier::Knn(_)));
    }

    #[test]
    fn run_skips_objects_with_an_unresolvable_class_and_reports_it() {
        let job = object_job(vec![
            labeled_object(1, ObjectClass::Valid(1), 1),
            labeled_object(2, ObjectClass::Valid(99), 2), // not in class_labels
            labeled_object(3, ObjectClass::Valid(2), 100),
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

    #[test]
    fn run_returns_cancelled_when_the_flag_is_already_set() {
        let job = object_job(vec![labeled_object(1, ObjectClass::Valid(1), 1)]);
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(true));

        let err = job.run(tx, cancel).unwrap_err();
        assert!(matches!(err, InternalErrors::Cancelled));
    }

    #[test]
    fn run_errors_for_a_pixel_classifier_configuration() {
        let mut job = object_job(vec![]);
        job.settings.classifier = AiLearningClassifierSettings::Pixel {
            feature_spec: AiLearningPixelFeatureSettings { channels: vec![] },
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
