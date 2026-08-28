use std::collections::HashMap;
use std::path::PathBuf;

use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SegmentationClass,
};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningClassifierSettings;
use macros::CommandsMeta;

use crate::{
    ai_learning::model::load_from_file,
    ai_learning::training::object::compute_object_features,
    algos::{ExecutionScope, ImageAlgorithm, ai_segmentation::model_cache::load_cached_classifier},
    object::Object,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// Maps one class the model was trained to predict to a class ID meaningful
/// in this project. Model class IDs are a snapshot taken at training time
/// (see `evanalyzer_cfg::ObjectClassLabel`'s doc comment) and aren't
/// guaranteed to line up with this project's own `ObjectClass` IDs - this is
/// the bridge between the two.
#[derive(CommandsMeta)]
pub struct ClassificationMapping {
    /// Object class predicted by the classifier model.
    pub object_class: ObjectClass,

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

/// An object classifier trained via the app's AI training dialog (an
/// `.evamodel` file - see `ai_learning::training::object::ObjectTrainingJob`),
/// applied here as a pipeline classification step: every object already
/// present in `PipelineCache::object_cache` matching `origin_segmentation`/
/// `input_classes` is scored independently (reusing the same feature recipe
/// used at training time), then remapped through `segmentation_mapping` into
/// this project's own classes and applied via `match_handling` - the same
/// input selection `ClassifyObjects` uses, but the output class comes from
/// the model's prediction instead of a fixed rule.
///
/// Predicted classes with no matching `segmentation_mapping` entry leave the
/// object untouched, mirroring how `PixelClassifier` leaves unmapped
/// predictions as background - mapping only the classes you care about is a
/// deliberate simplification, not an oversight.
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
    /// If no seg class is selected the criteria are applied to all objects.
    #[cmdsmeta(visible = false)]
    pub origin_segmentation: Vec<SegmentationClass>,

    /// Restrict classification to objects that already carry one of these classes
    ///
    /// Only ROIs that have been assigned at least one of the listed classes by a prior
    /// pipeline step will be evaluated by the model. Leave empty to apply the model to
    /// every object regardless of its current class.
    #[cmdsmeta(display_name = "Input Classes")]
    pub input_classes: Vec<ObjectClass>,

    /// What to do with object class labels after prediction
    ///
    /// - **AddOutputClassIfMatch** - append the mapped class alongside the object's existing classes.
    /// - **ReclassifyIfMatch** - clear every class the object carries and assign only the mapped class.
    #[cmdsmeta(default = AiClassifyMatchHandling::ReclassifyIfMatch)]
    pub match_handling: AiClassifyMatchHandling,
}

impl ImageAlgorithm for AiObjectClassifier {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
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
        let (ids, rows): (Vec<ObjectId>, Vec<Vec<f32>>) = cache
            .object_cache
            .values()
            .filter(|object| self.matches_input(object))
            .map(|object| {
                (
                    object.id.clone(),
                    compute_object_features(object, feature_spec),
                )
            })
            .unzip();

        if ids.is_empty() {
            return Ok(());
        }

        let predicted = saved.classifier.predict(&rows)?;
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
                    object.add_object_class(output_class);
                }
                AiClassifyMatchHandling::ReclassifyIfMatch => {
                    object.object_class.clear();
                    object.add_object_class(output_class);
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Ai Object Classifier"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
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
    /// `origin_segmentation` (if any is configured) and carries every class
    /// listed in `input_classes` (if any is configured) - the same input
    /// selection `ClassifyObjects` applies before evaluating its criteria.
    fn matches_input(&self, object: &Object) -> bool {
        (self.origin_segmentation.is_empty()
            || self
                .origin_segmentation
                .contains(&object.segmentation_class))
            && (self.input_classes.is_empty() || object.has_object_classes(&self.input_classes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_learning::model::random_forest::fit_random_forest;
    use crate::ai_learning::model::{
        CURRENT_SAVED_CLASSIFIER_VERSION, SavedClassifier, save_to_file,
    };
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
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

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
                    feature_spec: AiLearningPixelFeatureSettings { channels: vec![] },
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
        let img = Image::<f32, 1, CpuAllocator>::new(size, vec![0.0f32], CpuAllocator).unwrap();
        let managed = ManagedImage {
            data: img,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
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
        object.add_object_class(ObjectClass::from_segmentation_class(seg_class));
        object
    }

    const SMALL_ID: u128 = 1;
    const LARGE_ID: u128 = 2;

    fn labels() -> Vec<ObjectClassLabel> {
        vec![
            ObjectClassLabel {
                class: ObjectClass::Valid(5),
                name: "Small".into(),
            },
            ObjectClassLabel {
                class: ObjectClass::Valid(6),
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
        };
        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();

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
        };
        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();

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
                object_class: ObjectClass::Valid(6),
                output_class: ObjectClass::Valid(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
        };

        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();
        let small = make_object(SMALL_ID, 1, SegmentationClass(1));
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(small.id.clone(), small);
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let small = cache.object_cache.get(&ObjectId(SMALL_ID)).unwrap();
        assert!(
            small.has_object_class(&ObjectClass::from_segmentation_class(SegmentationClass(1))),
            "unmapped prediction must leave the object's classes untouched"
        );

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            large.has_object_class(&ObjectClass::Valid(42)),
            "mapped prediction must be applied"
        );
        assert_eq!(
            large.object_class.len(),
            1,
            "ReclassifyIfMatch must clear the object's previous classes"
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
                object_class: ObjectClass::Valid(6),
                output_class: ObjectClass::Unset,
            }],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
        };

        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            large.has_object_class(&ObjectClass::from_segmentation_class(SegmentationClass(1))),
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
                object_class: ObjectClass::Valid(6),
                output_class: ObjectClass::Valid(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::AddOutputClassIfMatch,
        };

        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(large.has_object_class(&ObjectClass::Valid(42)));
        assert!(
            large.has_object_class(&ObjectClass::from_segmentation_class(SegmentationClass(1))),
            "AddOutputClassIfMatch must keep the object's original class"
        );
    }

    #[test]
    fn execute_skips_objects_outside_origin_segmentation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(&saved_object_classifier(labels()), &path).unwrap();

        let cmd = AiObjectClassifier {
            model_path: path,
            segmentation_mapping: vec![ClassificationMapping {
                object_class: ObjectClass::Valid(6),
                output_class: ObjectClass::Valid(42),
            }],
            origin_segmentation: vec![SegmentationClass(9)],
            input_classes: vec![],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
        };

        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            !large.has_object_class(&ObjectClass::Valid(42)),
            "object from a non-matching segmentation class must not be classified"
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
                object_class: ObjectClass::Valid(6),
                output_class: ObjectClass::Valid(42),
            }],
            origin_segmentation: vec![],
            input_classes: vec![ObjectClass::Valid(99)],
            match_handling: AiClassifyMatchHandling::ReclassifyIfMatch,
        };

        let mut ctx = make_ctx();
        let mut cache = PipelineCache::default();
        let large = make_object(LARGE_ID, 100, SegmentationClass(1));
        cache.object_cache.insert(large.id.clone(), large);

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let large = cache.object_cache.get(&ObjectId(LARGE_ID)).unwrap();
        assert!(
            !large.has_object_class(&ObjectClass::Valid(42)),
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
