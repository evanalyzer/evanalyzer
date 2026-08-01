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

