//! # ai_object_classifier
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::pipeline::scope::Scope;
use std::collections::HashMap;
use std::path::PathBuf;

use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ModelClass, ObjectClass, ObjectId, SegmentationClass,
};
use evanalyzer_cfg::settings::ai_learning_object_settings::{
    AiLearningObjectFeatureSettings, ObjectMetric,
};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningClassifierSettings;
use indexmap::IndexMap;
use kornia_image::ImageSize;
use macros::CommandsMeta;

use crate::{
    ai_learning::model::load_from_file,
    ai_learning::training::object::compute_object_features,
    algos::{
        ExecutionScope, ImageAlgorithm, ai_segmentation::classifier_cache::load_cached_classifier,
    },
    object::{Intensity, Object, ObjectInit, regions::ring_mask},
    pipeline::{
        pipeline::{ObjectPredictionCapture, PredictedObjectClass, PredictionRequest},
        pipeline_cache::GlobalPipelineCache,
        pipeline_context::PipelineContext,
    },
};

/// Maps one class the model was trained to predict to a class ID meaningful
/// in this project. Model class IDs are a snapshot taken at training time
/// (see `evanalyzer_cfg::ObjectClassLabel`'s doc comment) and aren't
/// guaranteed to line up with this project's own `ObjectClass` IDs - this is
/// the bridge between the two.
#[derive(CommandsMeta)]
pub struct ClassificationMapping {
    /// Object class predicted by the classifier model.
    pub object_class: ModelClass,

    /// The project's own object class objects predicted as `object_class`
    /// are assigned.
    pub output_class: ObjectClass,
}

/// What to do with an object's class labels once the model has predicted (and
/// `segmentation_mapping` has remapped) a class for it.
#[derive(CommandsMeta)]
pub enum AiClassifyMatchHandling {
    #[cmdsmeta(display_name = "Add class on match")]
    AddOutputClassIfMatch,

    #[cmdsmeta(display_name = "Reclassify on match")]
    ReclassifyIfMatch,
}

/// Classifies objects with a model trained in the AI training dialog (an
/// `.evamodel` file).
///
/// Every object found so far that matches the input classes is scored on
/// its own, with the same features as in training - for the ring features
/// the ring around the object is measured here, as in training. The predicted class is
/// mapped to one of this project's classes and applied as set under
/// **Match handling** - the same input selection as **Classify Objects**,
/// but the class comes from the model's prediction instead of a fixed rule.
///
/// A predicted class without a mapping leaves the object untouched (like
/// unmapped classes of the pixel classifier stay background): map only the
/// classes you care about.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", display_name = "AI Object Classifier")]
pub struct AiObjectClassifier {
    /// Path to a trained object classifier model, saved from the AI training dialog.
    #[cmdsmeta(file_extensions = "evamodel")]
    pub model_path: PathBuf,

    /// Maps the model's predicted classes to this project's object classes.
    pub segmentation_mapping: Vec<ClassificationMapping>,

    /// Apply only to objects with this given segmentation class
    ///
    /// The segmentation class value is assigned to each pixel in the image
    /// after a Threshold, Pixel classifier or AI classifier.
    /// Leave empty to select the objects by their input classes only.
    /// With input classes too, an object must match both. Alone it also
    /// selects objects without a class left - the way to bring objects an
    /// earlier step removed back.
    #[cmdsmeta(visibility = Hidden)]
    pub origin_segmentation: Vec<SegmentationClass>,

    /// Restrict classification to objects that already carry one of these classes
    ///
    /// Only ROIs that have been assigned at least one of the listed classes by a prior
    /// pipeline step will be evaluated by the model.
    /// Leave empty to select the objects by their segmentation class only.
    /// With neither input classes nor a segmentation class the step does nothing.
    #[cmdsmeta(display_name = "Input Classes")]
    pub input_classes: Vec<ObjectClass>,

    /// What to do with object class labels after prediction
    ///
    /// - **AddOutputClassIfMatch** - append the mapped class alongside the object's existing classes.
    /// - **ReclassifyIfMatch** - clear every class the object carries and assign only the mapped class.
    #[cmdsmeta(default = AiClassifyMatchHandling::ReclassifyIfMatch)]
    pub match_handling: AiClassifyMatchHandling,

    /// Objects the model is less sure about than this are left as they are (0 = off).
    /// The certainty is the share of trees (random forest) or neighbours (KNN) voting for the class, or the probability of the class (MLP).
    #[cmdsmeta(
        default = 0.0,
        min = 0.0,
        max = 1.0,
        step = 0.05,
        optional = true,
        summary = true
    )]
    pub min_certainty: f32,
}

impl ImageAlgorithm for AiObjectClassifier {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let saved = load_cached_classifier(&self.model_path, || load_from_file(&self.model_path))?;

        let AiLearningClassifierSettings::Object {
            feature_spec,
            class_labels,
        } = &saved.settings.classifier
        else {
            return Err(InternalErrors::InvalidArgument(format!(
                "'{}' is not an object classifier model",
                self.model_path.display()
            )));
        };

        // Nothing selects objects: classifying "every object" would also
        // give objects removed by an earlier step a class again. The editor
        // warns about it (`module_checks`). After the model checks, so a
        // wrong model is still reported.
        if self.input_classes.is_empty() && self.origin_segmentation.is_empty() {
            return Ok(());
        }

        // `output_class == Unset` means "not mapped yet" (the UI defaults every
        // model class to Unset until the user picks a target) - drop those
        // entries so they're skipped exactly like a class with no entry at all,
        // rather than clearing an object's classes to assign it nothing.
        let mapping: HashMap<ObjectClass, ObjectClass> = self
            .segmentation_mapping
            .iter()
            .filter(|m| m.output_class != ObjectClass::Unset)
            .map(|m| (m.object_class, m.output_class))
            .collect();

        // Immutable pass: pick the objects to classify and compute their feature
        // vectors, up front - `cache.object_cache` can't be read (to compute
        // features) and written (to apply predictions) at the same time.
        let chosen: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|object| self.matches_input(object, ctx.scope))
            .collect();
        let ids: Vec<ObjectId> = chosen.iter().map(|object| object.id.clone()).collect();
        let rows = features_of(
            &chosen,
            feature_spec,
            &ctx.image_meta.full_image_width,
            cache,
        );

        if ids.is_empty() {
            return Ok(());
        }

        // The prediction overlay wants every scored object's class as
        // predicted (the minimum certainty is applied after it) and how
        // probable it is.
        let wants_prediction = ctx.prediction == PredictionRequest::Asked;
        let predicted = if self.min_certainty > 0.0 || wants_prediction {
            let (mut predicted, certainty) = saved.classifier.predict_with_certainty(&rows)?;
            check_prediction_count_matches(ids.len(), predicted.len())?;
            if wants_prediction {
                ctx.prediction =
                    object_prediction(&ids, &predicted, &certainty, class_labels, &mapping);
            }
            // Below the minimum certainty: left as it is (skipped below).
            for (class, certainty) in predicted.iter_mut().zip(certainty) {
                if certainty < self.min_certainty {
                    *class = usize::MAX;
                }
            }
            predicted
        } else {
            saved.classifier.predict(&rows)?
        };
        check_prediction_count_matches(ids.len(), predicted.len())?;

        for (id, class_idx) in ids.into_iter().zip(predicted) {
            let Some(model_class) = class_labels.get(class_idx).map(|label| label.class) else {
                continue;
            };
            let Some(&output_class) = mapping.get(&model_class) else {
                continue;
            };
            let Some(object) = cache.object_cache.get_mut(&id) else {
                continue;
            };
            match self.match_handling {
                AiClassifyMatchHandling::AddOutputClassIfMatch => {
                    object.add_object_class(output_class, ctx.scope)?;
                }
                AiClassifyMatchHandling::ReclassifyIfMatch => {
                    object.clear_object_classes();
                    object.add_object_class(output_class, ctx.scope)?;
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Ai Object Classifier"
    }

    /// Nothing: the learning backend (random forest, KNN, MLP) is only known
    /// from the model file, and citing the wrong method would mislead.
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        Vec::new()
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

/// The prediction overlay's capture: each scored object's class as
/// predicted (an index into the model's `class_labels`) and its
/// probability as a byte, the classes with the project class each maps to.
/// `None` for a model with more classes than a byte holds.
fn object_prediction(
    ids: &[ObjectId],
    predicted: &[usize],
    certainty: &[f32],
    class_labels: &[evanalyzer_cfg::settings::ai_learning_settings::ObjectClassLabel],
    mapping: &HashMap<ObjectClass, ObjectClass>,
) -> PredictionRequest {
    if class_labels.len() > usize::from(u8::MAX) {
        log::warn!(
            "prediction overlay: {} classes don't fit a byte - not shown",
            class_labels.len()
        );
        return PredictionRequest::None;
    }
    PredictionRequest::CapturedObjects(ObjectPredictionCapture {
        classes: class_labels
            .iter()
            .map(|label| PredictedObjectClass {
                name: label.name.clone(),
                object_class: mapping
                    .get(&label.class)
                    .copied()
                    .unwrap_or(ObjectClass::Unset),
            })
            .collect(),
        objects: ids
            .iter()
            .zip(predicted)
            .zip(certainty)
            .map(|((id, &class), &p)| {
                (
                    id.clone(),
                    class as u8,
                    (p.clamp(0.0, 1.0) * 255.0).round() as u8,
                )
            })
            .collect(),
    })
}

/// The feature rows of `objects` (of an image of `size`), as in training:
/// the objects' own intensities plus, for the ring features, the rings
/// around them measured from the channel images in `cache` like the
/// objects themselves.
pub(crate) fn features_of(
    objects: &[&Object],
    spec: &AiLearningObjectFeatureSettings,
    size: &ImageSize,
    cache: &GlobalPipelineCache,
) -> Vec<Vec<f32>> {
    let rings: Vec<IndexMap<i32, Intensity>> = if spec.metrics.iter().any(ObjectMetric::uses_ring) {
        let mut rings: Vec<Object> = objects
            .iter()
            .map(|object| {
                let (mask_data, bbox) = ring_mask(
                    &object.mask_data,
                    object.bbox,
                    spec.ring_width,
                    size.width,
                    size.height,
                );
                Object::new(ObjectInit {
                    bbox,
                    mask_data,
                    ..Default::default()
                })
            })
            .collect();
        Object::measure_intensities_of(&mut rings, cache);
        rings.into_iter().map(|ring| ring.intensities).collect()
    } else {
        Vec::new()
    };
    let no_ring = IndexMap::new();
    objects
        .iter()
        .enumerate()
        .map(|(i, object)| compute_object_features(object, rings.get(i).unwrap_or(&no_ring), spec))
        .collect()
}

/// Guards against `zip`ing `ids` with a `predicted` vector of a different
/// length - the same "wrong output, no error" shape as the mismatched-buffer
/// bug already fixed in `ImageMath`, if `Classifier::predict` ever returns
/// something other than exactly one prediction per input row, in order.
fn check_prediction_count_matches(
    ids_len: usize,
    predicted_len: usize,
) -> Result<(), InternalErrors> {
    if predicted_len != ids_len {
        return Err(InternalErrors::Generic(format!(
            "classifier returned {predicted_len} predictions for {ids_len} objects - refusing \
             to guess which prediction belongs to which object"
        )));
    }
    Ok(())
}

impl AiObjectClassifier {
    /// Whether `object` is in scope for this classifier: matches
    /// `origin_segmentation` (if any is configured) and carries at least one
    /// class listed in `input_classes` (if any is configured) - the same input
    /// selection `ClassifyObjects` applies before evaluating its criteria.
    fn matches_input(&self, object: &Object, scope: Scope) -> bool {
        (self.origin_segmentation.is_empty()
            || self
                .origin_segmentation
                .contains(&object.segmentation_class))
            && if self.input_classes.is_empty() {
                object.is_visible_in(scope)
            } else {
                object.is_of_any(&self.input_classes, scope)
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_learning::model::random_forest::fit_random_forest;
    use crate::ai_learning::model::{
        CURRENT_SAVED_CLASSIFIER_VERSION, SavedClassifier, save_to_file,
    };
    use crate::image::Point2d;
    use crate::object::ObjectInit;
    use crate::{ImageContainer, ImagePlane, ManagedImage};
    use bitvec::prelude::*;
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningSettings, ObjectClassLabel, PixelClassLabel,
        RandomForestSettings,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use kornia_image::{Image, ImageSize};

    fn reliable_rf_settings() -> RandomForestSettings {
        RandomForestSettings {
            n_trees: 10,
            max_depth: Some(5),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: 42,
            ..Default::default()
        }
    }

    /// Two well-separated single-feature (area) clusters: small objects (area
    /// near 1.0) -> label 0, large objects (area near 100.0) -> label 1.
    fn saved_object_classifier(class_labels: Vec<ObjectClassLabel>) -> SavedClassifier {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![1.0 + jitter]);
            labels.push(0);
            rows.push(vec![100.0 + jitter]);
            labels.push(1);
        }
        let classifier = fit_random_forest(&rows, &labels, &reliable_rf_settings()).unwrap();
        SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData::default(),
                backend: AiLearningBackendSettings::RandomForest(reliable_rf_settings()),
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings {
                        metrics: vec![evanalyzer_cfg::settings::ai_learning_object_settings::ObjectMetric::Area],
                        ..Default::default()
                    },
                    class_labels,
                },
            },
        }
    }

    fn saved_pixel_classifier() -> SavedClassifier {
        let classifier = fit_random_forest(
            &[vec![0.0], vec![1.0]],
            &[0, 1],
            &RandomForestSettings::default(),
        )
        .unwrap();
        SavedClassifier {
            version: CURRENT_SAVED_CLASSIFIER_VERSION,
            classifier,
            settings: AiLearningSettings {
                schema_version: evanalyzer_cfg::CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION,
                meta: MetaData::default(),
                backend: AiLearningBackendSettings::RandomForest(RandomForestSettings::default()),
                classifier: AiLearningClassifierSettings::Pixel {
                    feature_spec: AiLearningPixelFeatureSettings {
                        channels: vec![],
                        image_channels: Vec::new(),
                        image_channel_names: Vec::new(),
                    },
                    input_color: Default::default(),
                    class_labels: Vec::<PixelClassLabel>::new(),
                },
            },
        }
    }

    fn make_ctx() -> PipelineContext {
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let img = Image::<f32, 1>::new(size, vec![0.0f32]).unwrap();
        let managed = ManagedImage {
            data: img,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        };
        PipelineContext::new_from_image(
            PathBuf::default(),
            crate::pipeline::pipeline::PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                },
                full_image_width: size,
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: crate::image::PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::F32Gray(managed).into(),
        )
        .unwrap()
    }

    fn make_object(id: u128, area: usize, seg_class: SegmentationClass) -> Object {
        let bbox = [0u32, 0u32, 0u32, 0u32];
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, 1);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data,
            area,
            plane: ImagePlane::default(),
            segmentation_class: seg_class,
            ..Default::default()
        });
        object
            .add_object_class(
                ObjectClass::from_segmentation_class(seg_class),
                Scope::Global,
            )
            .unwrap();
        object
    }

    const SMALL_ID: u128 = 1;
    const LARGE_ID: u128 = 2;

    fn labels() -> Vec<ObjectClassLabel> {
        vec![
            ObjectClassLabel {
                class: ObjectClass::Global(5),
                name: "Small".into(),
            },
            ObjectClassLabel {
                class: ObjectClass::Global(6),
                name: "Large".into(),
            },
        ]
    }

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = AiObjectClassifier {
            model_path: dir.path().join("missing.evamodel"),
            segmentation_mapping: vec![],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };
        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();

        assert!(cmd.execute(&mut ctx, &mut cache).is_err());
    }

    #[test]
    fn execute_errors_when_the_saved_model_is_a_pixel_classifier_not_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pixel.evamodel");
        save_to_file(&saved_pixel_classifier(), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };
        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::InvalidArgument(_)));
    }

    #[test]
    fn execute_predicts_and_remaps_matching_objects() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            // Only the "Large" model class is mapped - "Small" (5) is
            // deliberately left unmapped, so a small object should be left
            // untouched entirely.
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Global(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::from_segmentation_class(SegmentationClass(1))],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let small = make_object(SMALL_ID, 1, SegmentationClass(1));
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(small.id.clone(), small);
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let small = cache.object_cache.get(&ObjectId(SMALL_ID)).unwrap();
        assert!(
            small.is_of(
                &ObjectClass::from_segmentation_class(SegmentationClass(1)),
                Scope::Global
            ),
            "unmapped prediction must leave the object's classes untouched"
        );

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            large.is_of(&ObjectClass::Global(42), Scope::Global),
            "mapped prediction must be applied"
        );
        assert_eq!(
            large.object_classes().len(),
            1,
            "ReclassifyIfMatch must clear the object's previous classes"
        );
    }

    /// A model the trees disagree on for mid-sized objects (area 50 was
    /// trained as both classes), sure about large ones.
    fn unsure_in_the_middle() -> SavedClassifier {
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![1.0 + jitter]);
            targets.push(0);
            rows.push(vec![100.0 + jitter]);
            targets.push(1);
            rows.push(vec![50.0]);
            targets.push(i % 2);
        }
        let mut saved = saved_object_classifier(labels());
        saved.classifier = fit_random_forest(&rows, &targets, &reliable_rf_settings()).unwrap();
        saved
    }

    /// Both classes mapped (Small → 41, Large → 42); every object of
    /// segmentation class 1 scored.
    fn classify_all(path: &std::path::Path, min_certainty: f32) -> AiObjectClassifier {
        AiObjectClassifier {
            model_path: path.to_path_buf(),
            segmentation_mapping: vec![
                ClassificationMapping {
                    object_class: ObjectClass::Global(5),
                    output_class: ObjectClass::Global(41),
                },
                ClassificationMapping {
                    object_class: ObjectClass::Global(6),
                    output_class: ObjectClass::Unset,
                },
            ],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::from_segmentation_class(SegmentationClass(1))],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty,
        }
    }

    /// An object the model is less sure about than the minimum certainty
    /// keeps its classes; a sure one is classified.
    #[test]
    fn objects_below_the_minimum_certainty_are_left_as_they_are() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&unsure_in_the_middle(), &path).unwrap();
        let run = |min_certainty: f32| {
            let mut cache = GlobalPipelineCache::default();
            for (id, area) in [(SMALL_ID, 1), (3, 50)] {
                let object = make_object(id, area, SegmentationClass(1));
                cache.object_cache.insert(object.id.clone(), object);
            }
            classify_all(&path, min_certainty)
                .execute(&mut make_ctx(), &mut cache)
                .unwrap();
            [SMALL_ID, 3].map(|id| {
                cache
                    .object_cache
                    .get(&ObjectId(id))
                    .unwrap()
                    .is_of(&ObjectClass::Global(41), Scope::Global)
            })
        };
        let (_, share) = {
            let crate::ai_learning::model::Classifier::RandomForest(forest) =
                &unsure_in_the_middle().classifier
            else {
                unreachable!()
            };
            forest.predict_with_share(&[vec![50.0]]).unwrap()
        };
        assert!(share[0] < 1.0, "the trees disagree at 50: {share:?}");

        assert_eq!(run(0.0)[0], true);
        let [small, middle] = run(1.0);
        assert!(small, "a sure object is classified");
        assert!(!middle, "an unsure one keeps its classes");
    }

    /// Asked for its prediction, the step captures every scored object's
    /// class as predicted (also below the minimum certainty) with its
    /// probability, and the model's classes with the project class each
    /// gets (Unset where not mapped). Not asked, nothing.
    #[test]
    fn the_object_prediction_is_captured_only_when_asked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();
        let cache = || {
            let mut cache = GlobalPipelineCache::default();
            for (id, area) in [(SMALL_ID, 1), (LARGE_ID, 100)] {
                let object = make_object(id, area, SegmentationClass(1));
                cache.object_cache.insert(object.id.clone(), object);
            }
            cache
        };

        let mut ctx = make_ctx();
        classify_all(&path, 0.0)
            .execute(&mut ctx, &mut cache())
            .unwrap();
        assert_eq!(ctx.prediction, PredictionRequest::None);

        let mut ctx = make_ctx();
        ctx.prediction = PredictionRequest::Asked;
        classify_all(&path, 1.0)
            .execute(&mut ctx, &mut cache())
            .unwrap();
        let PredictionRequest::CapturedObjects(mut capture) = ctx.prediction else {
            panic!("captured")
        };
        capture.objects.sort_by_key(|(id, _, _)| id.0);
        assert_eq!(
            capture,
            ObjectPredictionCapture {
                classes: vec![
                    PredictedObjectClass {
                        name: "Small".into(),
                        object_class: ObjectClass::Global(41),
                    },
                    PredictedObjectClass {
                        name: "Large".into(),
                        object_class: ObjectClass::Unset,
                    },
                ],
                objects: vec![(ObjectId(SMALL_ID), 0, 255), (ObjectId(LARGE_ID), 1, 255)],
            }
        );
    }

    #[test]
    fn execute_treats_an_unset_output_class_as_not_mapped() {
        // The UI defaults every model class row to `ObjectClass::Unset` until the
        // user actively picks a target - that must behave exactly like no
        // mapping entry at all (skip), not clear the object's classes to
        // assign it "nothing".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Unset,
            }],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::from_segmentation_class(SegmentationClass(1))],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            large.is_of(
                &ObjectClass::from_segmentation_class(SegmentationClass(1)),
                Scope::Global
            ),
            "an unmapped (Unset) row must leave the object's classes untouched, \
             not clear them via ReclassifyIfMatch"
        );
    }

    #[test]
    fn execute_add_output_class_if_match_keeps_existing_classes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Global(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::from_segmentation_class(SegmentationClass(1))],
            match_handling: AiClassifyMatchHandling::AddOutputClassIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(large.is_of(&ObjectClass::Global(42), Scope::Global));
        assert!(
            large.is_of(
                &ObjectClass::from_segmentation_class(SegmentationClass(1)),
                Scope::Global
            ),
            "AddOutputClassIfMatch must keep the object's original class"
        );
    }

    /// Neither input classes nor a segmentation class: the step does
    /// nothing - no object is classified, and one an earlier step removed
    /// (without a class left) stays removed.
    #[test]
    fn execute_without_input_classes_and_origin_segmentation_classifies_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        for match_handling in [
            AiClassifyMatchHandling::ReclassifyIfMatch,
            AiClassifyMatchHandling::AddOutputClassIfMatch,
        ] {
            let cmd = AiObjectClassifier {
                model_path: path.clone(),
                segmentation_mapping: vec![ClassificationMapping {
                    object_class: ObjectClass::Global(6),
                    output_class: ObjectClass::Global(42),
                }],
                origin_segmentation: vec![],
                input_classes: vec![],
                match_handling,
                min_certainty: 0.0,
            };

            let mut ctx = make_ctx();
            let mut cache = GlobalPipelineCache::default();
            let large = make_object(LARGE_ID, 100, SegmentationClass(1));
            let mut removed = make_object(3, 100, SegmentationClass(1));
            removed.clear_object_classes();
            cache.object_cache.insert(large.id.clone(), large);
            cache.object_cache.insert(removed.id.clone(), removed);

            cmd.execute(&mut ctx, &mut cache).unwrap();

            let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
            assert!(!large.is_of(&ObjectClass::Global(42), Scope::Global));
            assert!(large.is_of(
                &ObjectClass::from_segmentation_class(SegmentationClass(1)),
                Scope::Global
            ));
            assert!(
                cache
                    .object_cache
                    .get(&ObjectId(3))
                    .unwrap()
                    .object_classes()
                    .is_empty(),
                "a removed object stays removed"
            );
        }
    }

    #[test]
    fn execute_skips_objects_outside_origin_segmentation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Global(42),
            }],
            origin_segmentation: vec![SegmentationClass(9)],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            !large.is_of(&ObjectClass::Global(42), Scope::Global),
            "object from a non-matching segmentation class must not be classified"
        );
    }

    /// One of several `input_classes` is enough to be classified.
    #[test]
    fn execute_classifies_objects_carrying_any_of_the_input_classes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Global(42),
            }],
            origin_segmentation: vec![],
            // The object carries only the second one.
            input_classes: vec![
                ObjectClass::Global(99),
                ObjectClass::from_segmentation_class(SegmentationClass(1)),
            ],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            large.is_of(&ObjectClass::Global(42), Scope::Global),
            "an object carrying one of the input classes must be classified"
        );
    }

    #[test]
    fn execute_skips_objects_missing_input_classes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Global(6),
                output_class: ObjectClass::Global(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::Global(99)],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };

        let mut ctx = make_ctx();
        let mut cache = GlobalPipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            !large.is_of(&ObjectClass::Global(42), Scope::Global),
            "object missing a required input class must not be classified"
        );
    }

    #[test]
    fn name_returns_ai_object_classifier() {
        let cmd = AiObjectClassifier {
            model_path: PathBuf::new(),
            segmentation_mapping: vec![],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
            min_certainty: 0.0,
        };
        assert_eq!(cmd.name(), "Ai Object Classifier");
    }

    #[test]
    fn check_prediction_count_matches_accepts_equal_lengths() {
        assert!(check_prediction_count_matches(3, 3).is_ok());
        assert!(check_prediction_count_matches(0, 0).is_ok());
    }

    #[test]
    fn check_prediction_count_matches_rejects_a_shorter_or_longer_prediction_vector() {
        // This is the exact bug: `zip` silently truncates/misaligns instead
        // of erroring when `predict()` returns the wrong number of
        // predictions - the guard must catch both directions.
        assert!(check_prediction_count_matches(3, 2).is_err());
        assert!(check_prediction_count_matches(3, 4).is_err());
    }
}
