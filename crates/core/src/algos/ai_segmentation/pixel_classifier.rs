//! # pixel_classifier
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ModelSegmentationClass, SegmentationClass,
};
use evanalyzer_cfg::settings::ai_learning_settings::{
    AiLearningClassifierSettings, PixelInputColor,
};
use macros::CommandsMeta;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{
    ai_learning::model::load_from_file,
    ai_learning::training::pixel::{compute_pixel_features, compute_pixel_features_on},
    algos::{
        ExecutionScope, ImageAlgorithm, ai_segmentation::classifier_cache::load_cached_classifier,
    },
    image::ImageContainer,
    pipeline::{
        pipeline::{PredictedClass, PredictionCapture, PredictionRequest},
        pipeline_cache::{CacheAddress, GlobalPipelineCache},
        pipeline_context::PipelineContext,
    },
};

/// Maps one class the model was trained to predict to a class ID meaningful
/// in this project. Model class IDs are a snapshot taken at training time
/// (see `evanalyzer_cfg::PixelClassLabel`'s doc comment) and aren't
/// guaranteed to line up with this project's own `SegmentationClass` IDs -
/// this is the bridge between the two.
#[derive(CommandsMeta)]
pub struct SegmentationMapping {
    /// The class the classifier model predicts.
    pub model_class: ModelSegmentationClass,

    /// The project's own segmentation class pixels predicted as
    /// `model_class` are written as.
    pub segmentation_class: SegmentationClass,
}

/// Segments the image with a pixel classifier trained in the AI training dialog.
///
/// [Preprocessing] -> [Pixel classifier] -> [Fill Holes] -> [Connected Components] -> [Watershed] -> [Extract Objects]
///
/// Train a classifier in EVAnalyzer's AI training dialog from your annotated objects; the training saves it as an `.evamodel` file, which this step applies.
/// The classifier assigns every pixel of the image to a class, producing a segmented image.
/// The segmentation mapping translates the model's classes into this project's segmentation classes.
///
/// Use a pixel classifier in place of a threshold step.
/// Like a threshold, it is usually followed by a Connected Components step and then an Extract Objects step.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", display_name = "AI Pixel Classifier")]
pub struct PixelClassifier {
    /// Path to a trained pixel classifier model, saved from the AI training dialog.
    #[cmdsmeta(file_extensions = "evamodel")]
    pub model_path: PathBuf,

    /// Maps the model's predicted classes to this project's segmentation classes.
    pub segmentation_mapping: Vec<SegmentationMapping>,

    /// Pixels the model is less sure about than this become background (0 = off).
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

impl ImageAlgorithm for PixelClassifier {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let saved = load_cached_classifier(&self.model_path, || load_from_file(&self.model_path))?;

        let AiLearningClassifierSettings::Pixel {
            feature_spec,
            input_color,
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
        // The image channels the model was trained on, from this tile's
        // channel images - or, when it names none, the image the step gets.
        let bank = if feature_spec.image_channels.is_empty() {
            check_input_color(*input_color, ctx.image.as_ref(), &self.model_path)?;
            compute_pixel_features(ctx, feature_spec)?
        } else {
            let tile = ctx.image_meta.image_tile_info.clone();
            let images = feature_spec
                .image_channels
                .iter()
                .map(|channel| {
                    cache
                        .image_cache
                        .get(&CacheAddress::Channel((*channel, tile.clone())))
                        .ok_or_else(|| {
                            InternalErrors::InvalidArgument(format!(
                                "'{}' needs image channel {channel}, which this image doesn't have",
                                self.model_path.display()
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for image in &images {
                check_input_color(*input_color, image.as_ref(), &self.model_path)?;
            }
            compute_pixel_features_on(ctx, feature_spec, &images)?
        };
        let size = ctx.image.size();
        let (width, height) = (size.width, size.height);

        let mut rows = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                rows.push(bank.feature_vector_at(x, y));
            }
        }
        let mapping: HashMap<SegmentationClass, u32> = self
            .segmentation_mapping
            .iter()
            .map(|m| (m.model_class, m.segmentation_class.as_u32()))
            .collect();

        // The prediction overlay wants every pixel's class as predicted (the
        // minimum certainty is applied after it) and how probable it is.
        let wants_prediction = ctx.prediction == PredictionRequest::Asked;
        let predicted = if self.min_certainty > 0.0 || wants_prediction {
            let (mut predicted, certainty) = saved.classifier.predict_with_certainty(&rows)?;
            if wants_prediction {
                ctx.prediction = prediction_capture(
                    &predicted,
                    &certainty,
                    class_labels,
                    &mapping,
                    width,
                    height,
                );
            }
            // Below the minimum certainty: background (`write_predictions`).
            for (class, certainty) in predicted.iter_mut().zip(certainty) {
                if certainty < self.min_certainty {
                    *class = usize::MAX;
                }
            }
            predicted
        } else {
            saved.classifier.predict(&rows)?
        };

        let segmentation_map = ctx.get_segmentation_map_mut()?;
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

    /// Nothing: the learning backend (random forest, KNN, MLP) is only known
    /// from the model file, and citing the wrong method would mislead.
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        Vec::new()
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// Refuses to apply a model to the other image kind than it was trained on:
/// greyscale and colour images produce feature vectors of different length
/// and meaning (see `compute_pixel_features`), so the result would be wrong
/// or the backend would fail with an unhelpful dimension error.
fn check_input_color(
    model_color: PixelInputColor,
    image: &ImageContainer,
    model_path: &Path,
) -> Result<(), InternalErrors> {
    let image_is_rgb = matches!(image, ImageContainer::F32Rgb(_));
    match (model_color, image_is_rgb) {
        (PixelInputColor::Gray, true) => Err(InternalErrors::InvalidArgument(format!(
            "The pixel classifier '{}' was trained on greyscale images; this image is in \
             colour. Retrain the model on colour images to use it here.",
            model_path.display()
        ))),
        (PixelInputColor::Rgb, false) => Err(InternalErrors::InvalidArgument(format!(
            "The pixel classifier '{}' was trained on colour images; this image is \
             greyscale. Retrain the model on greyscale images to use it here.",
            model_path.display()
        ))),
        _ => Ok(()),
    }
}

/// The prediction overlay's capture of a tile: each pixel's class (an index
/// into the model's `class_labels`, as predicted - before the minimum
/// certainty) and its probability as a byte. `Captured`, or `None` for a
/// model with more classes than a byte holds (no overlay then).
fn prediction_capture(
    predicted: &[usize],
    certainty: &[f32],
    class_labels: &[evanalyzer_cfg::settings::ai_learning_settings::PixelClassLabel],
    mapping: &HashMap<SegmentationClass, u32>,
    width: usize,
    height: usize,
) -> PredictionRequest {
    if class_labels.len() > usize::from(u8::MAX) {
        log::warn!(
            "prediction overlay: {} classes don't fit a byte - not shown",
            class_labels.len()
        );
        return PredictionRequest::None;
    }
    PredictionRequest::Captured(PredictionCapture {
        width,
        height,
        classes: class_labels
            .iter()
            .map(|label| PredictedClass {
                name: label.name.clone(),
                segmentation_class: mapping
                    .get(&label.class)
                    .map_or(SegmentationClass::BACKGROUND, |&class| {
                        SegmentationClass(class)
                    }),
            })
            .collect(),
        class_of_pixel: predicted.iter().map(|&class| class as u8).collect(),
        probability: certainty
            .iter()
            .map(|&p| (p.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
    })
}

/// Remaps each pixel's predicted class index (into `class_labels`) through
/// `mapping` into this project's segmentation classes, writing the result
/// into `seg_slice`. Unmapped or out-of-range predictions fall back to
/// `SegmentationClass::BACKGROUND` - see [`PixelClassifier`]'s doc comment;
/// `usize::MAX` marks a pixel below the minimum certainty.
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
        PixelClassLabel, PixelInputColor, RandomForestSettings,
    };
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use kornia_image::{Image, ImageSize};

    fn gray_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 1>::new(ImageSize { width, height }, values).unwrap();
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
                        image_channels: Vec::new(),
                        image_channel_names: Vec::new(),
                    },
                    input_color: PixelInputColor::Gray,
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
                    feature_spec: AiLearningObjectFeatureSettings::default(),
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
            min_certainty: 0.0,
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
            min_certainty: 0.0,
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
                model_class: SegmentationClass(6),
                segmentation_class: SegmentationClass(42),
            }],
            min_certainty: 0.0,
        };

        // Pixel values chosen to fall squarely in each training cluster.
        let mut ctx = gray_ctx(2, 1, vec![0.05, 10.05]);
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[0u32, 42u32]);
    }

    /// A model trained on image channel 1 reads channel 1 of the tile from
    /// the cache - not the image the step gets (here channel 0, the
    /// opposite values) - and names a channel the image lacks.
    #[test]
    fn a_model_for_an_image_channel_reads_that_channel() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.evamodel");
        let mut saved = saved_pixel_classifier(labels());
        if let AiLearningClassifierSettings::Pixel { feature_spec, .. } =
            &mut saved.settings.classifier
        {
            feature_spec.image_channels = vec![1];
        }
        save_to_file(&saved, &path).unwrap();
        let cmd = PixelClassifier {
            model_path: path,
            segmentation_mapping: vec![SegmentationMapping {
                model_class: SegmentationClass(6),
                segmentation_class: SegmentationClass(42),
            }],
            min_certainty: 0.0,
        };

        let mut ctx = gray_ctx(2, 1, vec![10.05, 0.05]);
        let tile = ctx.image_meta.image_tile_info.clone();
        let mut cache = GlobalPipelineCache::default();
        let channel_1 = gray_ctx(2, 1, vec![0.05, 10.05]).image.clone();
        cache.add_to_channel_cache(channel_1, 1, tile);
        cmd.execute(&mut ctx, &mut cache).unwrap();
        assert_eq!(
            ctx.get_segmentation_map().unwrap().as_slice(),
            &[0u32, 42u32]
        );

        let mut ctx = gray_ctx(2, 1, vec![0.0, 0.0]);
        let err = cmd
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap_err();
        assert!(err.to_string().contains("image channel 1"), "{err}");
    }

    /// Pixels the forest is unsure about become background once a minimum
    /// certainty is set; clear pixels (every tree agrees) stay.
    #[test]
    fn pixels_below_the_minimum_certainty_become_background() {
        let mut rows = Vec::new();
        let mut row_classes = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![0.0 + jitter]);
            row_classes.push(0);
            rows.push(vec![10.0 + jitter]);
            row_classes.push(1);
            // The middle is both classes, so the trees disagree near it.
            rows.push(vec![5.0]);
            row_classes.push(i % 2);
        }
        let mut saved = saved_pixel_classifier(labels());
        saved.classifier = fit_random_forest(&rows, &row_classes, &reliable_rf_settings()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let run = |min_certainty: f32| {
            let cmd = PixelClassifier {
                min_certainty,
                ..command_for(&saved, &dir)
            };
            let mut ctx = gray_ctx(3, 1, vec![0.05, 5.0, 10.05]);
            cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
                .unwrap();
            ctx.get_segmentation_map().unwrap().as_slice().to_vec()
        };
        let crate::ai_learning::model::Classifier::RandomForest(forest) = &saved.classifier else {
            unreachable!()
        };
        let (_, share) = forest
            .predict_with_share(&[vec![0.05], vec![5.0], vec![10.05]])
            .unwrap();
        assert_eq!(share[0], 1.0, "a clear pixel: every tree agrees");
        assert!(share[1] < 1.0 || share[2] < 1.0, "{share:?}");

        let off = run(0.0);
        assert_eq!(off[0], 41);
        assert!(
            off.iter().all(|&c| c != 0),
            "without a minimum every pixel gets a class"
        );
        for min in [0.85, 0.95, 1.0] {
            let expected: Vec<u32> = off
                .iter()
                .zip(&share)
                .map(|(&class, &share)| if share < min { 0 } else { class })
                .collect();
            assert_eq!(run(min), expected, "minimum {min}, shares {share:?}");
        }
    }

    #[test]
    fn name_returns_pixel_classifier() {
        let cmd = PixelClassifier {
            model_path: PathBuf::new(),
            segmentation_mapping: vec![],
            min_certainty: 0.0,
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

    // -- colour (RGB) models and images -----------------------------------------

    fn rgb_ctx(width: usize, height: usize, values: Vec<f32>) -> PipelineContext {
        let img = Image::<f32, 3>::new(ImageSize { width, height }, values).unwrap();
        PipelineContext::new_from_image_test_rgb(img).unwrap()
    }

    /// A colour model telling red (label 0) from cyan (label 1) pixels -
    /// one raw recipe entry, so three features (R, G, B) per pixel. The
    /// classes differ in every colour, so whichever feature a random
    /// forest split happens to pick separates them.
    fn saved_rgb_pixel_classifier() -> SavedClassifier {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            let jitter = (i % 3) as f32 * 0.1;
            rows.push(vec![10.0 + jitter, 0.0, 0.0]);
            labels.push(0);
            rows.push(vec![0.0, 10.0 + jitter, 10.0 + jitter]);
            labels.push(1);
        }
        let classifier = fit_random_forest(&rows, &labels, &reliable_rf_settings()).unwrap();
        let mut saved = saved_pixel_classifier(vec![
            PixelClassLabel {
                class: SegmentationClass(5),
                name: "Red".into(),
            },
            PixelClassLabel {
                class: SegmentationClass(6),
                name: "Cyan".into(),
            },
        ]);
        saved.classifier = classifier;
        if let AiLearningClassifierSettings::Pixel { input_color, .. } =
            &mut saved.settings.classifier
        {
            *input_color = PixelInputColor::Rgb;
        }
        saved
    }

    fn command_for(saved: &SavedClassifier, dir: &tempfile::TempDir) -> PixelClassifier {
        let path = dir.path().join("model.evamodel");
        save_to_file(saved, &path).unwrap();
        PixelClassifier {
            model_path: path,
            segmentation_mapping: vec![
                SegmentationMapping {
                    model_class: SegmentationClass(5),
                    segmentation_class: SegmentationClass(41),
                },
                SegmentationMapping {
                    model_class: SegmentationClass(6),
                    segmentation_class: SegmentationClass(42),
                },
            ],
            min_certainty: 0.0,
        }
    }

    /// Asked for its prediction (the prediction overlay), the step captures
    /// every pixel's class as predicted - also where the minimum certainty
    /// turns it into background - with its probability, and the model's
    /// classes with the project class each goes to (background when the
    /// step maps a class nowhere). Not asked, it captures nothing.
    #[test]
    fn the_prediction_is_captured_only_when_asked() {
        let saved = saved_pixel_classifier(labels());
        let dir = tempfile::tempdir().unwrap();
        let mut cache = GlobalPipelineCache::default();

        let mut ctx = gray_ctx(2, 1, vec![0.05, 10.05]);
        command_for(&saved, &dir)
            .execute(&mut ctx, &mut cache)
            .unwrap();
        assert_eq!(ctx.prediction, PredictionRequest::None);

        let mut cmd = PixelClassifier {
            min_certainty: 1.0,
            ..command_for(&saved, &dir)
        };
        cmd.segmentation_mapping.truncate(1);
        let mut ctx = gray_ctx(2, 1, vec![0.05, 10.05]);
        ctx.prediction = PredictionRequest::Asked;
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert_eq!(
            ctx.prediction,
            PredictionRequest::Captured(PredictionCapture {
                width: 2,
                height: 1,
                classes: vec![
                    PredictedClass {
                        name: "Background".into(),
                        segmentation_class: SegmentationClass(41),
                    },
                    PredictedClass {
                        name: "Cell".into(),
                        segmentation_class: SegmentationClass::BACKGROUND,
                    },
                ],
                class_of_pixel: vec![0, 1],
                probability: vec![255, 255],
            })
        );
    }

    /// The captured probability is the certainty as a byte, for every pixel
    /// - here pixels the trees disagree on.
    #[test]
    fn the_captured_probability_is_the_certainty_as_a_byte() {
        let mut rows = Vec::new();
        let mut row_classes = Vec::new();
        for i in 0..15 {
            rows.push(vec![0.0 + (i % 3) as f32 * 0.1]);
            row_classes.push(0);
            rows.push(vec![10.0 + (i % 3) as f32 * 0.1]);
            row_classes.push(1);
            rows.push(vec![5.0]);
            row_classes.push(i % 2);
        }
        let mut saved = saved_pixel_classifier(labels());
        saved.classifier = fit_random_forest(&rows, &row_classes, &reliable_rf_settings()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let pixels = vec![0.05, 5.0, 10.05];
        let mut ctx = gray_ctx(3, 1, pixels.clone());
        ctx.prediction = PredictionRequest::Asked;
        command_for(&saved, &dir)
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let (classes, certainty) = saved
            .classifier
            .predict_with_certainty(&pixels.iter().map(|&v| vec![v]).collect::<Vec<_>>())
            .unwrap();
        let PredictionRequest::Captured(capture) = ctx.prediction else {
            panic!("captured")
        };
        assert_eq!(
            capture.class_of_pixel,
            classes.iter().map(|&c| c as u8).collect::<Vec<_>>()
        );
        assert_eq!(
            capture.probability,
            certainty
                .iter()
                .map(|&p| (p * 255.0).round() as u8)
                .collect::<Vec<_>>()
        );
        assert!(capture.probability.iter().any(|&p| p < 255), "{capture:?}");
    }

    #[test]
    fn execute_applies_a_colour_model_to_a_colour_image() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = command_for(&saved_rgb_pixel_classifier(), &dir);
        let mut ctx = rgb_ctx(2, 1, vec![10.05, 0.0, 0.0, 0.0, 10.05, 10.05]);
        let mut cache = GlobalPipelineCache::default();

        cmd.execute(&mut ctx, &mut cache).unwrap();

        let seg = ctx.get_segmentation_map().unwrap();
        assert_eq!(seg.as_slice(), &[41u32, 42u32]);
    }

    #[test]
    fn execute_refuses_a_greyscale_model_on_a_colour_image() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = command_for(&saved_pixel_classifier(vec![]), &dir);
        let mut ctx = rgb_ctx(1, 1, vec![0.0, 0.0, 0.0]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();

        let InternalErrors::InvalidArgument(message) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            message.contains("was trained on greyscale images; this image is in colour"),
            "{message}"
        );
    }

    #[test]
    fn execute_refuses_a_colour_model_on_a_greyscale_image() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = command_for(&saved_rgb_pixel_classifier(), &dir);
        let mut ctx = gray_ctx(1, 1, vec![0.0]);
        let mut cache = GlobalPipelineCache::default();

        let err = cmd.execute(&mut ctx, &mut cache).unwrap_err();

        let InternalErrors::InvalidArgument(message) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            message.contains("was trained on colour images; this image is greyscale"),
            "{message}"
        );
    }

    #[test]
    fn a_model_file_without_input_color_loads_as_greyscale() {
        // Model files written before `input_color` existed.
        let mut json = serde_json::to_value(saved_pixel_classifier(vec![]).settings).unwrap();
        json["classifier"]["PIXEL"]
            .as_object_mut()
            .unwrap()
            .remove("input_color")
            .expect("field should be serialized under this name");

        let settings: AiLearningSettings = serde_json::from_value(json).unwrap();

        let AiLearningClassifierSettings::Pixel { input_color, .. } = settings.classifier else {
            panic!("expected a Pixel classifier");
        };
        assert_eq!(input_color, PixelInputColor::Gray);
    }
}
