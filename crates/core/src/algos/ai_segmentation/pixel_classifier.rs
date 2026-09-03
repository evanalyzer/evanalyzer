use std::collections::HashMap;
use std::path::PathBuf;

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SegmentationClass};
use evanalyzer_cfg::settings::ai_learning_settings::AiLearningClassifierSettings;
use macros::CommandsMeta;

use crate::{
    ai_learning::model::load_from_file,
    ai_learning::training::pixel::compute_pixel_features,
    algos::{ExecutionScope, ImageAlgorithm, ai_segmentation::model_cache::load_cached_classifier},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Maps one class the model was trained to predict to a class ID meaningful
/// in this project. Model class IDs are a snapshot taken at training time
/// (see `evanalyzer_cfg::PixelClassLabel`'s doc comment) and aren't
/// guaranteed to line up with this project's own `SegmentationClass` IDs -
/// this is the bridge between the two.
#[derive(CommandsMeta)]
pub struct SegmentationMapping {
    /// Segmentation class predicted by the classifier model.
    pub segmentation_class: SegmentationClass,

    /// The project's own segmentation class pixels predicted as
    /// `segmentation_class` are written as.
    pub object_class_id: SegmentationClass,
}

/// A pixel classifier trained via the app's AI training dialog
/// (an`.evamodel` file - see `ai_learning::training::pixel::PixelTrainingJob`),
/// applied here as a pipeline segmentation step: every pixel is classified
/// independently (reusing the same feature recipe used at training time),
/// then remapped through `segmentation_mapping` into this project's own
/// classes and written to the segmentation map - the same output shape
/// `Threshold` produces, so downstream steps (extraction, classification)
/// don't need to care which one ran.
///
/// Predicted classes with no matching `segmentation_mapping` entry are
/// written as `SegmentationClass::BACKGROUND`, mirroring how `Threshold`
/// leaves pixels outside every configured range as background - mapping
/// only the classes you care about is a deliberate simplification, not an
/// oversight.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", display_name = "AI Pixel Classifier")]
pub struct PixelClassifier {
    /// Path to a trained pixel classifier model, saved from the AI training dialog.
    #[cmdsmeta(file_extensions = "evamodel")]
    pub model_path: PathBuf,

    /// Maps the model's predicted classes to this project's segmentation classes.
    pub segmentation_mapping: Vec<SegmentationMapping>,
}

impl ImageAlgorithm for PixelClassifier {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let saved = load_cached_classifier(&self.model_path, || load_from_file(&self.model_path))?;

        let AiLearningClassifierSettings::Pixel {
            feature_spec,
            class_labels,
        } = &saved.settings.classifier
        else {
            return Err(InternalErrors::InvalidArgument(format!(
                "'{}' is not a pixel classifier model",
                self.model_path.display()
            )));
        };

        // Immutable borrow of `ctx` to compute features and run inference -
        // `bank`/`predicted` own their data, so this ends before the mutable
        // borrow below (writing the segmentation map) starts.
        let bank = compute_pixel_features(ctx, feature_spec)?;
        let size = ctx.image.size();
        let (width, height) = (size.width, size.height);

        let mut rows = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                rows.push(bank.feature_vector_at(x, y));
            }
        }
        let predicted = saved.classifier.predict(&rows)?;

        let mapping: HashMap<SegmentationClass, u32> = self
            .segmentation_mapping
            .iter()
            .map(|m| (m.segmentation_class, m.object_class_id.as_u32()))
            .collect();

        let (_, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;
        write_predictions(
            &predicted,
            class_labels,
            &mapping,
            segmentation_map.as_slice_mut(),
        );

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Pixel Classifier"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// Remaps each pixel's predicted class index (into `class_labels`) through
/// `mapping` into this project's segmentation classes, writing the result
/// into `seg_slice`. Unmapped or out-of-range predictions fall back to
/// `SegmentationClass::BACKGROUND` - see [`PixelClassifier`]'s doc comment.
fn write_predictions(
    predicted: &[usize],
    class_labels: &[evanalyzer_cfg::settings::ai_learning_settings::PixelClassLabel],
    mapping: &HashMap<SegmentationClass, u32>,
    seg_slice: &mut [u32],
) {
    for (out_pixel, &class_idx) in seg_slice.iter_mut().zip(predicted.iter()) {
        let model_class = class_labels
            .get(class_idx)
            .map(|label| label.class)
            .unwrap_or(SegmentationClass::BACKGROUND);
        *out_pixel = mapping
            .get(&model_class)
            .copied()
            .unwrap_or(SegmentationClass::BACKGROUND.as_u32());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_learning::model::random_forest::fit_random_forest;
    use crate::ai_learning::model::{
        CURRENT_SAVED_CLASSIFIER_VERSION, SavedClassifier, save_to_file,
    };
    use evanalyzer_cfg::settings::ai_learning_object_settings::AiLearningObjectFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_pixel_settings::AiLearningPixelFeatureSettings;
    use evanalyzer_cfg::settings::ai_learning_settings::{
        AiLearningBackendSettings, AiLearningClassifierSettings, AiLearningSettings,
        PixelClassLabel, RandomForestSettings,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, values, CpuAllocator)
                .unwrap();
        PipelineContext::new_from_image_test(img).unwrap()
    }

    /// Two well-separated single-feature clusters (raw pixel value near 0.0
    /// -> label 0, near 10.0 -> label 1) - same recipe as
    /// `random_forest.rs`'s own `two_cluster_dataset`, trivial for a working
    /// classifier to fit exactly.
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

    fn saved_pixel_classifier(class_labels: Vec<PixelClassLabel>) -> SavedClassifier {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![0.0 + jitter]);
            labels.push(0);
            rows.push(vec![10.0 + jitter]);
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
                classifier: AiLearningClassifierSettings::Pixel {
                    // A single raw (unfiltered) channel - the feature vector
                    // is just the pixel's own value, matching `saved_pixel_classifier`'s
                    // single-feature training rows above.
                    feature_spec: AiLearningPixelFeatureSettings {
                        channels: vec![vec![]],
                    },
                    class_labels,
                },
            },
        }
    }

    fn saved_object_classifier() -> SavedClassifier {
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
                classifier: AiLearningClassifierSettings::Object {
                    feature_spec: AiLearningObjectFeatureSettings { metrics: vec![] },
                    class_labels: vec![],
                },
            },
        }
    }

    #[test]
    fn execute_errors_when_the_model_path_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = PixelClassifier {
            model_path: dir.path().join("missing.evamodel"),
            segmentation_mapping: vec![],
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        assert!(cmd.execute(&mut ctx, &mut cache).is_err());
    }

    #[test]
    fn execute_errors_when_the_saved_model_is_an_object_classifier_not_pixel() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("object.evamodel");
        save_to_file(&saved_object_classifier(), &path).unwrap();

        let cmd = PixelClassifier {
            model_path: path,
            segmentation_mapping: vec![],
        };
        let mut ctx = gray_ctx(2, 2, vec![0.0; 4]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();
        assert!(matches!(err, InternalErrors::InvalidArgument(_)));
    }

    #[test]
    fn execute_predicts_and_remaps_every_pixel_into_the_segmentation_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        save_to_file(
            &saved_pixel_classifier(vec![
                PixelClassLabel {
                    class: SegmentationClass(5),
                    name: "Background".into(),
                },
                PixelClassLabel {
                    class: SegmentationClass(6),
                    name: "Cell".into(),
                },
            ]),
            &path,
        )
        .unwrap();

        let cmd = PixelClassifier {
            model_path: path,
            // Only the "Cell" model class is mapped - "Background" (5) is
            // deliberately left unmapped, exercising the same
            // falls-back-to-BACKGROUND behavior `write_predictions`'s own
            // unit tests cover, but now through the full `execute` path.
            segmentation_mapping: vec![SegmentationMapping {
                segmentation_class: SegmentationClass(6),
                object_class_id: SegmentationClass(42),
            }],
        };

        // Pixel values chosen to fall squarely in each training cluster.
        let mut ctx = gray_ctx(2, 1, vec![0.05, 10.05]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[0u32, 42u32]);
    }

    #[test]
    fn name_returns_pixel_classifier() {
        let cmd = PixelClassifier {
            model_path: PathBuf::new(),
            segmentation_mapping: vec![],
        };
        assert_eq!(cmd.name(), "Pixel Classifier");
    }

    fn labels() -> Vec<PixelClassLabel> {
        vec![
            PixelClassLabel {
                class: SegmentationClass(5),
                name: "Background".into(),
            },
            PixelClassLabel {
                class: SegmentationClass(6),
                name: "Cell".into(),
            },
        ]
    }

    #[test]
    fn write_predictions_remaps_through_the_configured_mapping() {
        let mapping = HashMap::from([(SegmentationClass(6), 42u32)]);
        let predicted = vec![0usize, 1, 1, 0];
        let mut seg = vec![0u32; 4];

        write_predictions(&predicted, &labels(), &mapping, &mut seg);

        // Index 0 -> model class 5 (Background), unmapped -> falls back to 0.
        // Index 1 -> model class 6 (Cell), mapped -> 42.
        assert_eq!(seg, vec![0, 42, 42, 0]);
    }

    #[test]
    fn write_predictions_falls_back_to_background_for_an_out_of_range_index() {
        // Defensive: a mismatched/hand-edited model file could carry a
        // shorter `class_labels` than the classifier's own output size.
        let mapping = HashMap::from([(SegmentationClass(6), 42u32)]);
        let predicted = vec![99usize];
        let mut seg = vec![7u32]; // stale value from a previous pass

        write_predictions(&predicted, &labels(), &mapping, &mut seg);

        assert_eq!(seg, vec![0]);
    }

    #[test]
    fn write_predictions_resets_stale_buffer_values_for_unmapped_classes() {
        let mapping = HashMap::new();
        let predicted = vec![0usize, 1];
        let mut seg = vec![9u32, 9u32]; // stale sentinel from a previous pass

        write_predictions(&predicted, &labels(), &mapping, &mut seg);

        assert_eq!(
            seg,
            vec![0, 0],
            "unmapped predictions must reset stale buffer values, not preserve them"
        );
    }
}
