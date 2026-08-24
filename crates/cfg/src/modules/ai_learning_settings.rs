use crate::{
    core_types::{ObjectClass, SegmentationClass},
    modules::{
        ai_learning_object_settings::AiLearningObjectFeatureSettings,
        ai_learning_pixel_settings::AiLearningPixelFeatureSettings, meta_data::MetaData,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The function to measure the quality of a split.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SplitCriterion {
    #[default]
    Gini,
    Entropy,
    ClassificationError,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct RandomForestSettings {
    /// Split criteria to use when building a tree.
    pub criterion: SplitCriterion,
    /// Tree max depth.
    pub max_depth: Option<u16>,
    /// The minimum number of samples required to be at a leaf node.
    pub min_samples_leaf: usize,
    /// The minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// The number of trees in the forest.
    pub n_trees: u16,
    /// Number of random sample of predictors to use as split candidates.
    pub m: Option<usize>,
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub seed: u64,
}

/// Both, KNN classifier and regressor benefits from underlying search algorithms that helps to speed up queries.
/// `KNNAlgorithmName` maintains a list of supported search algorithms, see [KNN algorithms]
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KNNAlgorithmName {
    /// Heap Search algorithm
    LinearSearch,
    /// Cover Tree Search algorithm
    #[default]
    CoverTree,
}

/// Weight function that is used to determine estimated value.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KNNWeightFunction {
    /// All k nearest points are weighted equally
    #[default]
    Uniform,
    /// k nearest points are weighted by the inverse of their distance. Closer neighbors will have a greater influence than neighbors which are further away.
    Distance,
}

/// The distance function used to measure similarity between feature vectors
/// when finding a point's k nearest neighbors.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KNNDistanceMetric {
    /// Straight-line ("as the crow flies") distance. The standard choice for
    /// continuous, comparably-scaled features.
    #[default]
    Euclidean,
    /// Sum of absolute differences per feature ("taxicab" distance) - less
    /// sensitive to outliers in a single feature than Euclidean.
    Manhattan,
    /// 1 minus the cosine similarity between two vectors - measures the
    /// angle between them, ignoring magnitude. Useful when the direction of
    /// a feature vector matters more than its scale.
    Cosine,
    /// Number of positions at which two vectors differ. Intended for
    /// discrete/categorical-valued features, not continuous ones.
    Hamming,
    /// Generalization of Euclidean (`p = 2`) and Manhattan (`p = 1`) distance
    /// to an arbitrary order `p`.
    Minkowski {
        /// The order of the norm. Must be at least 1.
        p: u16,
    },
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct KnnSettings {
    /// backend search algorithm.
    pub algorithm: KNNAlgorithmName,
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
    /// distance function used to find a point's nearest neighbors. Default is `KNNDistanceMetric::Euclidean`.
    pub distance: KNNDistanceMetric,
}

/// The activation function applied between hidden layers.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MlpActivation {
    #[default]
    Relu,
    Sigmoid,
    Tanh,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct MlpSettings {
    /// Number of nodes in each hidden layer, in order (e.g. `[64, 32]` is two
    /// hidden layers). Input/output layer sizes are derived from the feature
    /// count and number of classes, not configured here.
    pub hidden_layers: Vec<usize>,
    /// Activation function between hidden layers.
    pub activation: MlpActivation,
    /// Number of training epochs (full passes over the training data).
    pub epochs: usize,
    /// Adam optimizer learning rate.
    pub learning_rate: f64,
    /// Samples per gradient update.
    pub batch_size: usize,
    /// Seed for weight initialization, for reproducible training runs.
    pub seed: u64,
    /// Adam optimizer epsilon - a small value added to the denominator of the
    /// parameter update for numerical stability. Rarely needs tuning; the
    /// default (`1e-5`, via `Default` below) matches burn's own `AdamConfig`
    /// default.
    pub epsilon: f64,
}

/// `#[derive(Default)]` would give `epsilon` (and every other numeric field)
/// a bare `0.0`/`0` - fine for most of these (a project always overwrites
/// them before training), but `epsilon: 0.0` is a real correctness hazard
/// specifically: burn's Adam divides by `sqrt(v_hat) + epsilon`, and early in
/// training `v_hat` can legitimately be exactly 0, so an unset epsilon can
/// produce a division by zero instead of just "a slightly worse default."
impl Default for MlpSettings {
    fn default() -> Self {
        Self {
            hidden_layers: Vec::new(),
            activation: MlpActivation::default(),
            epochs: 0,
            learning_rate: 0.0,
            batch_size: 0,
            seed: 0,
            epsilon: 1e-5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AiLearningBackendSettings {
    RandomForest(RandomForestSettings),
    Knn(KnnSettings),
    Mlp(MlpSettings),
}

/// One class a pixel classifier was trained to recognize: the stable ID
/// actually used to write predictions back to `segmentation_class`, paired
/// with the display name that class had *at training time*.
///
/// This is a snapshot, not a live reference into a project's class list.
/// `AiLearningSettings` plus the trained weights make up the full portable
/// model file - it may be shared, applied to a different project, or applied
/// long after the original class was renamed in the project it came from -
/// so the name has to travel with the model rather than being resolved from
/// project state that may not exist or may not match at inference time.
///
/// SERIALIZATION-CRITICAL: part of the saved model artifact.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PixelClassLabel {
    pub class: SegmentationClass,
    pub name: String,
}

/// Same as `PixelClassLabel`, for object classifiers - writes to
/// `object_class` instead of `segmentation_class`, hence the different ID
/// type. SERIALIZATION-CRITICAL.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ObjectClassLabel {
    pub class: ObjectClass,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AiLearningClassifierSettings {
    Pixel {
        feature_spec: AiLearningPixelFeatureSettings,
        /// Classes this model predicts, in the exact order its output
        /// indices refer to (predicted index 0 = class_labels[0], etc.) -
        /// order is load-bearing (it's what an MLP's output layer is
        /// structured around), not just informational.
        class_labels: Vec<PixelClassLabel>,
    },
    Object {
        feature_spec: AiLearningObjectFeatureSettings,
        class_labels: Vec<ObjectClassLabel>,
    },
}

/// This struct describes the AI model used for training.
/// This information is needed to build up the models
/// either for training or inference.
///
/// Together with the trained weights, this is the full portable trained
/// model - see `PixelClassLabel`/`ObjectClassLabel` for why class names are
/// snapshotted here rather than resolved from a live project.
///
/// SERIALIZATION-CRITICAL: never rename or remove an existing field once a
/// model has been saved with it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AiLearningSettings {
    /// On-disk format version of this settings document. Absent on files
    /// written before versioning was introduced, which `serde(default)`
    /// reads as `0` - see `CURRENT_AI_LEARNING_SETTINGS_SCHEMA_VERSION` and
    /// `evanalyzer_cfg::load_ai_learning_settings`.
    #[serde(default)]
    pub schema_version: u32,

    /// Name, description, author, creation time, category/tags - the same
    /// browsable-artifact metadata project templates already use, so a
    /// future "pick a model" UI can list/search/filter saved models the same
    /// way the template picker already does.
    #[serde(alias = "metadata")]
    pub meta: MetaData,

    /// Backend used for AI training (RandomForest, Knn, MLP, ...)
    pub backend: AiLearningBackendSettings,

    /// Whether a pixel or object classifier was trained, its feature recipe,
    /// and the classes it predicts.
    pub classifier: AiLearningClassifierSettings,
}
