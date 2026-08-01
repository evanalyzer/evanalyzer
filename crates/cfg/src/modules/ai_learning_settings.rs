use crate::{
    core_types::{ObjectClass, SegmentationClass},
    modules::{
        ai_learning_object_settings::AiLearningObjectFeatureSettings,
        ai_learning_pixel_settings::AiLearningPixelFeatureSettings,
        meta_data::MetaData,
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

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct KnnSettings {
    /// backend search algorithm.
    pub algorithm: KNNAlgorithmName,
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
}

/// The activation function applied between hidden layers.
#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MlpActivation {
    #[default]
    Relu,
    Sigmoid,
    Tanh,
}

#[derive(Serialize, Deserialize, Debug, Default, JsonSchema, Clone)]
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
    /// Name, description, author, creation time, category/tags - the same
    /// browsable-artifact metadata project templates already use, so a
    /// future "pick a model" UI can list/search/filter saved models the same
    /// way the template picker already does.
    pub metadata: MetaData,

    /// Backend used for AI training (RandomForest, Knn, MLP, ...)
    pub backend: AiLearningBackendSettings,

    /// Whether a pixel or object classifier was trained, its feature recipe,
    /// and the classes it predicts.
    pub classifier: AiLearningClassifierSettings,
}
