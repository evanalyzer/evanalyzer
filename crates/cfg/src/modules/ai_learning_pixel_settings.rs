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
