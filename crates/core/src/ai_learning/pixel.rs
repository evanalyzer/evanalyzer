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

/// Trains a pixel classifier from a labeled feature bank.
/// TODO: signature will firm up once `ai_learning::model::Classifier` (the
/// RandomForest/Knn/Mlp backend enum) and `SavedClassifier` exist.
pub fn train(_features: &FeatureBank, _labels: &[usize]) -> Result<(), InternalErrors> {
    todo!()
}
