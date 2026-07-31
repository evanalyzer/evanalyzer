use super::pixel_settings::{FeatureSpec, PreprocessingSteps};
use crate::algos::EdgeDetectionSobel;
use crate::algos::GaussianBlur;
use crate::algos::Hessian;
use crate::algos::ImageAlgorithm;
use crate::algos::Laplacian;
use crate::algos::RankFilter;
use crate::algos::StructureTensor;
use crate::image::ImageContainer;
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use evanalyzer_cfg::core_types::InternalErrors;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use std::sync::Arc;

/// Computed feature channels for one image, in `FeatureSpec::channels` order.
pub struct FeatureBank {
    width: usize,
    height: usize,
    channels: Vec<Arc<ImageContainer>>,
}

impl FeatureBank {
    pub fn n_features(&self) -> usize {
        self.channels.len()
    }

    /// Feature vector for one pixel, one value per channel, in `channels` order.
    /// Assumes every channel is single-channel (grayscale-derived) output.
    pub fn feature_vector_at(&self, x: usize, y: usize) -> Vec<f32> {
        self.channels
            .iter()
            .map(|c| {
                let slice = c
                    .as_f32_slice()
                    .expect("feature channel must be an f32 image");
                slice[y * self.width + x]
            })
            .collect()
    }
}

/// Builds the feature bank for one image, reusing the exact same optimized
/// `ImageAlgorithm` Commands (and their Arc-shared, scratch-pad/swap buffer model)
/// used by the main pipeline — no separate/duplicated filter math.
///
/// `template` supplies the source image plus the `image_meta`/`output_path` needed
/// to construct fresh per-channel `PipelineContext`s. Each channel gets its own
/// context sharing the same source `Arc` (cheap refcount bump, no pixel copy), so
/// filters never step on each other's input.
pub fn compute_pixel_features(
    template: &PipelineContext,
    spec: &FeatureSpec,
) -> Result<FeatureBank, InternalErrors> {
    let size = template.image.size();
    let mut channels = Vec::with_capacity(spec.channels.len());

    for steps in &spec.channels {
        channels.push(compute_channel(template, steps)?);
    }

    Ok(FeatureBank {
        width: size.width,
        height: size.height,
        channels,
    })
}

fn fresh_ctx(template: &PipelineContext) -> Result<PipelineContext, InternalErrors> {
    PipelineContext::new_from_image(
        template.output_path.clone().unwrap_or_default(),
        template.image_meta.clone(),
        template.image.clone(),
    )
}

fn compute_channel(
    template: &PipelineContext,
    steps: &[PreprocessingSteps],
) -> Result<Arc<ImageContainer>, InternalErrors> {
    if steps.is_empty() {
        return Ok(template.image.clone());
    }

    let mut ctx = fresh_ctx(template)?;
    let mut cache = PipelineCache::default();
    for step in steps {
        match step {
            PreprocessingSteps::GaussianBlur(s) => {
                GaussianBlur::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::EdgeDetectionSobel(s) => {
                EdgeDetectionSobel::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Laplacian(s) => {
                Laplacian::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::StructureTensor(s) => {
                StructureTensor::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::Hessian(s) => {
                Hessian::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
            PreprocessingSteps::RankFilter(s) => {
                RankFilter::from(s.clone()).execute(&mut ctx, &mut cache)?
            }
        }
    }
    Ok(ctx.image)
}

/// Builds smartcore's `(X, y)` training data from labeled pixel samples.
///
/// `samples` are (x, y) pixel coordinates within the image `features` was
/// computed from, paired 1:1 with `labels` — e.g. gathered from the pixels
/// inside a user-labeled Object's mask (one row per masked pixel, label =
/// that object's assigned class). Shared by every backend's `train_*` fn so
/// they all see identical training data.
fn build_training_data(
    features: &FeatureBank,
    samples: &[(usize, usize)],
    labels: &[usize],
) -> Result<(DenseMatrix<f32>, Vec<usize>), InternalErrors> {
    if samples.len() != labels.len() {
        return Err(InternalErrors::Internal(
            "samples and labels must have the same length".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(InternalErrors::Internal(
            "cannot train on zero samples".to_string(),
        ));
    }

    let rows: Vec<Vec<f32>> = samples
        .iter()
        .map(|&(x, y)| features.feature_vector_at(x, y))
        .collect();

    let x = DenseMatrix::from_2d_vec(&rows).map_err(|e| InternalErrors::Internal(e.to_string()))?;
    Ok((x, labels.to_vec()))
}

/// Trains a Random Forest pixel classifier from a labeled feature bank.
///
/// Returns the raw smartcore model directly for now — a first, working
/// end-to-end path. Wrapping it behind a shared `Classifier`/`AiLearning`
/// abstraction (so KNN/MLP can slot in the same way) is follow-up work, not
/// yet done here.
pub fn train_random_forest(
    features: &FeatureBank,
    samples: &[(usize, usize)],
    labels: &[usize],
) -> Result<RandomForestClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>>, InternalErrors> {
    let (x, y) = build_training_data(features, samples, labels)?;

    RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters::default())
        .map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// Trains a K-Nearest-Neighbors pixel classifier from a labeled feature bank.
/// Uses smartcore's defaults (k=3, Euclidean distance, CoverTree search) —
/// see `KNNClassifierParameters` if these need to be exposed as options later.
pub fn train_knn(
    features: &FeatureBank,
    samples: &[(usize, usize)],
    labels: &[usize],
) -> Result<KNNClassifier<f32, usize, DenseMatrix<f32>, Vec<usize>, Euclidian<f32>>, InternalErrors>
{
    let (x, y) = build_training_data(features, samples, labels)?;

    KNNClassifier::fit(&x, &y, KNNClassifierParameters::default())
        .map_err(|e| InternalErrors::Internal(e.to_string()))
}
