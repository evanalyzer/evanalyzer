use crate::modules::pipeline_command_settings::{
    EdgeDetectionSobelSettings, GaussianBlurSettings, HessianSettings, LaplacianSettings,
    RankFilterSettings, StructureTensorSettings,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A single preprocessing step within one feature channel's chain (an inner
/// entry of `FeatureSpec::channels`). Each step reads the previous step's
/// output in the same context (scratch-pad/swap chaining, same as any
/// multi-step Command sequence), so e.g. Laplacian-of-Gaussian is simply
/// `vec![GaussianBlur(...), Laplacian(...)]` — pre-smoothing any filter is just
/// an earlier step, not a bespoke field.
///
/// Wraps the already-generated `...Settings` types instead of re-declaring
/// parallel fields/enums: they're already Serialize/Deserialize/JsonSchema,
/// already narrow (pure algorithm parameters, no pipeline-orchestration
/// baggage), and their `From<Settings> for Command` conversions already exist
/// in `algos_from_config.rs`, so this stays in sync automatically if a
/// filter's parameters ever change.
///
/// IMPORTANT: `GaussianBlurSettings::sigma` is clamped to [0.1, 5.0] by its
/// `From` conversion (matches the command's own `#[cmdsmeta]` bounds) — any
/// step using `GaussianBlur` inherits that ceiling. ilastik's standard scale
/// range goes up to sigma=10; decide deliberately if you need to go higher.
///
/// SERIALIZATION-CRITICAL: this type is stored inside every saved pixel
/// classifier model. Once any model has been saved, existing variants must
/// never be renamed, removed, or have their meaning changed — treat this the
/// same as a database migration, not a normal refactor. Adding a new variant
/// is safe; changing or deleting an existing one silently breaks every model
/// already saved with it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum PreprocessingSteps {
    GaussianBlur(GaussianBlurSettings),
    EdgeDetectionSobel(EdgeDetectionSobelSettings),
    Laplacian(LaplacianSettings),
    StructureTensor(StructureTensorSettings),
    Hessian(HessianSettings),
    RankFilter(RankFilterSettings),
}

/// The closed, versioned feature recipe for a pixel classifier: one entry per
/// output feature channel, each channel an ordered chain of preprocessing
/// steps. Bundled with the saved classifier so training and inference
/// reproduce identical inputs. An empty inner `Vec` is the raw, unmodified
/// pixel value for that channel.
///
/// SERIALIZATION-CRITICAL — see `PreprocessingSteps` doc comment above; the
/// same never-rename/never-remove rule applies to this type's own fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AiLearningPixelFeatureSettings {
    pub channels: Vec<Vec<PreprocessingSteps>>,
}

impl AiLearningPixelFeatureSettings {
    /// Convenience: same filter at several scales, e.g. gaussian_scales(&[1.0, 2.0, 4.0], 5).
    /// Values outside [0.1, 5.0] will be silently clamped by GaussianBlur's own
    /// `From<GaussianBlurSettings>` conversion — see the `PreprocessingSteps` doc comment.
    pub fn gaussian_scales(sigmas: &[f32], kernel_size: usize) -> Vec<Vec<PreprocessingSteps>> {
        sigmas
            .iter()
            .map(|&sigma| {
                vec![PreprocessingSteps::GaussianBlur(GaussianBlurSettings {
                    kernel_size,
                    sigma,
                })]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- gaussian_scales ------------------------------------------------

    #[test]
    fn gaussian_scales_builds_one_single_step_channel_per_sigma() {
        let channels = AiLearningPixelFeatureSettings::gaussian_scales(&[1.0, 2.0, 4.0], 5);

        assert_eq!(channels.len(), 3);
        for (channel, expected_sigma) in channels.iter().zip([1.0, 2.0, 4.0]) {
            assert_eq!(channel.len(), 1, "each channel is a single GaussianBlur step");
            let PreprocessingSteps::GaussianBlur(settings) = &channel[0] else {
                panic!("expected a GaussianBlur step, got {:?}", channel[0]);
            };
            assert_eq!(settings.sigma, expected_sigma);
            assert_eq!(settings.kernel_size, 5);
        }
    }

    #[test]
    fn gaussian_scales_returns_no_channels_for_an_empty_sigma_slice() {
        assert!(AiLearningPixelFeatureSettings::gaussian_scales(&[], 5).is_empty());
    }

    #[test]
    fn gaussian_scales_preserves_sigma_order() {
        let channels = AiLearningPixelFeatureSettings::gaussian_scales(&[4.0, 1.0, 2.0], 3);
        let sigmas: Vec<f32> = channels
            .iter()
            .map(|c| {
                let PreprocessingSteps::GaussianBlur(settings) = &c[0] else {
                    panic!("expected a GaussianBlur step");
                };
                settings.sigma
            })
            .collect();
        assert_eq!(sigmas, vec![4.0, 1.0, 2.0]);
    }

    // -- serde round trip -------------------------------------------------
    //
    // `PreprocessingSteps`/`AiLearningPixelFeatureSettings` are SERIALIZATION-
    // CRITICAL (embedded in every saved `.evamodel` file) but the settings
    // types they wrap don't derive `PartialEq` (generated code, not hand-
    // editable - see `pipeline_command_settings.rs`'s `@generated` header),
    // so round trips here are checked by re-serializing rather than `==`.

    #[test]
    fn preprocessing_steps_json_round_trips_for_every_variant() {
        let steps = vec![
            PreprocessingSteps::GaussianBlur(GaussianBlurSettings {
                kernel_size: 5,
                sigma: 1.5,
            }),
            PreprocessingSteps::EdgeDetectionSobel(EdgeDetectionSobelSettings { kernel_size: 3 }),
            PreprocessingSteps::Laplacian(LaplacianSettings { kernel_size: 5 }),
            PreprocessingSteps::RankFilter(RankFilterSettings {
                radius: 2.0,
                filter_type:
                    crate::modules::pipeline_command_settings::FiltersRankFilterRankFilterTypeSettings::Outliers(
                        12.5,
                    ),
            }),
        ];

        let json = serde_json::to_string(&steps).expect("PreprocessingSteps must serialize");
        let round_tripped: Vec<PreprocessingSteps> =
            serde_json::from_str(&json).expect("PreprocessingSteps must deserialize");
        let re_serialized =
            serde_json::to_string(&round_tripped).expect("round-tripped value must re-serialize");

        assert_eq!(
            json, re_serialized,
            "serializing, deserializing, then re-serializing must be a no-op"
        );
    }

    #[test]
    fn ai_learning_pixel_feature_settings_json_round_trips_multiple_channels() {
        let settings = AiLearningPixelFeatureSettings {
            channels: vec![
                vec![], // Raw / unmodified pixel value
                AiLearningPixelFeatureSettings::gaussian_scales(&[1.0, 2.0], 5)
                    .into_iter()
                    .next()
                    .unwrap(),
                vec![PreprocessingSteps::Hessian(HessianSettings {
                    mode:
                        crate::modules::pipeline_command_settings::FiltersHessianHessianModeSettings::EigenvaluesX,
                })],
            ],
        };

        let json = serde_json::to_string(&settings)
            .expect("AiLearningPixelFeatureSettings must serialize");
        let round_tripped: AiLearningPixelFeatureSettings =
            serde_json::from_str(&json).expect("AiLearningPixelFeatureSettings must deserialize");

        assert_eq!(round_tripped.channels.len(), 3);
        assert!(round_tripped.channels[0].is_empty());
        assert_eq!(round_tripped.channels[1].len(), 1);
        assert!(matches!(
            round_tripped.channels[2].as_slice(),
            [PreprocessingSteps::Hessian(_)]
        ));
    }

    #[test]
    fn an_empty_channel_list_round_trips_as_the_raw_recipe() {
        let settings = AiLearningPixelFeatureSettings { channels: vec![] };
        let json = serde_json::to_string(&settings).unwrap();
        let round_tripped: AiLearningPixelFeatureSettings = serde_json::from_str(&json).unwrap();
        assert!(round_tripped.channels.is_empty());
    }
}
