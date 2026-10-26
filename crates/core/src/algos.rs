//! # algos
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

#![allow(unused_imports)]

// Register algos
mod ai_segmentation;
mod filters;
mod flow_control;
pub(crate) mod libs;
mod math;
mod measure;
mod morphology;
mod object;
mod script;
mod segmentation;
mod spartial_transform;

pub use self::ai_segmentation::cellpose::Cellpose;
pub use self::ai_segmentation::pixel_classifier::PixelClassifier;
pub use self::ai_segmentation::pixel_classifier::SegmentationMapping;
pub use self::ai_segmentation::stardist::Stardist;
pub use self::ai_segmentation::unet::UNet;
pub use self::ai_segmentation::unet::UNetOutputMode;
pub use self::ai_segmentation::yolov5::YoloClassMapping;
pub use self::ai_segmentation::yolov5::Yolov5;

pub use self::filters::blur::Blur;
pub use self::filters::blur_gaussian::GaussianBlur;
pub use self::filters::color_deconvolution::{ColorDeconvolution, StainOutput, StainVectors};
pub use self::filters::color_filter::ColorFilterCommand;
pub use self::filters::color_filter::HsvRange;
pub use self::filters::edge_detection_canny::EdgeDetectionCanny;
pub use self::filters::edge_detection_sobel::EdgeDetectionSobel;
pub use self::filters::enhance_contrast::EnhanceContrast;
pub use self::filters::hessian::Hessian;
pub use self::filters::hessian::HessianMode;
pub use self::filters::illumination_correction::ApplyMethod;
pub use self::filters::illumination_correction::CorrectionMethod;
pub use self::filters::illumination_correction::IlluminationCorrection;
pub use self::filters::illumination_correction::SmoothingMethod;
pub use self::filters::intensity_transform::IntensityTransformMode;
pub use self::filters::intensity_transform::IntensityTransformation;
pub use self::filters::laplacian::Laplacian;
pub use self::filters::rank_filter::RankFilter;
pub use self::filters::rank_filter::RankFilterType;
pub use self::filters::rolling_ball::BallType;
pub use self::filters::rolling_ball::RollingBall;
pub use self::filters::structure_tensor::StructureTensor;
pub use self::filters::structure_tensor::TensorMode;
pub use self::filters::weighted_deviation::WeightedDeviation;
pub(crate) use self::flow_control::end_scope::EndScope;
pub use self::math::image_cache::ImageCache;
pub use self::math::image_cache::ImageCacheMode;
pub use self::math::image_math::ImageMath;
pub use self::math::image_math::Operand;
pub use self::math::load_channel::ChannelZProjection;
pub use self::math::load_channel::LoadChannel;
pub use self::math::median_subtract::MedianSubtract;
pub use self::math::save_image::ImageSource;
pub use self::math::save_image::SaveImage;
pub use self::measure::colocalization::CostesMethod;
pub use self::measure::exclude_image::{ExcludeImage, ExcludeRule, ExcludeWhen, RuleComparison};
pub use self::measure::image_area_occupied::MeasureImageAreaOccupied;
pub use self::measure::image_colocalization::MeasureImageColocalization;
pub use self::measure::image_intensity::MeasureImageIntensity;
pub use self::measure::image_quality::MeasureImageQuality;
pub use self::measure::image_texture::MeasureImageTexture;
pub use self::measure::object_colocalization::MeasureObjectColocalization;
pub use self::measure::object_distance_to_parent::MeasureObjectDistanceToParent;
pub use self::measure::object_neighbors::{MeasureObjectNeighbors, NeighborMethod};
pub use self::measure::object_shape::MeasureObjectShape;
pub use self::measure::object_texture::MeasureObjectTexture;
pub use self::measure::texture::TextureDirections;
pub use self::morphology::fill_holes::FillHoles;
pub use self::morphology::fill_object_holes::FillObjectHoles;
pub use self::morphology::morphological_transformation::KernelShapes;
pub use self::morphology::morphological_transformation::MorphOps;
pub use self::morphology::morphological_transformation::MorphologicalCommand;
pub use self::object::ai_object_classifier::AiClassifyMatchHandling;
pub use self::object::ai_object_classifier::AiObjectClassifier;
pub use self::object::ai_object_classifier::ClassificationMapping;
pub(crate) use self::object::ai_object_classifier::features_of as ai_object_features_of;
pub use self::object::classify_objects::ClassifyMatchHandling;
pub use self::object::classify_objects::ClassifyObjects;
pub use self::object::classify_objects::IntensityComparison;
pub use self::object::classify_objects::IntensityFilter;
pub use self::object::classify_objects::IntensityMetric;
pub use self::object::coloc_objects::ColocMultiplicity;
pub use self::object::coloc_objects::ColocalizeObjects;
pub use self::object::extract_objects::{ExtractClassMapping, ExtractObjects};
pub use self::object::load_annotaed_objects::{AnnotationUse, LoadAnnotatedObjects};
pub use self::object::object_math::ObjectMath;
pub use self::object::object_math::ObjectSetOperation;
pub use self::object::propagation::Propagation;
pub use self::object::relate_objects::{RelateObjects, RelateRule};
pub(crate) use self::object::tile_merge::{Connectivity, TileMerge, touches_tile_edge};
pub use self::object::transform_objects::TransformFunction;
pub use self::object::transform_objects::TransformObjects;
pub use self::object::voronoi::Voronoi;
pub use self::script::script_executor::{
    Script, apply_command_param, command_parameters, declared_script_params, script_syntax_error,
};
pub use self::script::script_params::{ParamDecl, ParamKind, ScriptValue};
pub use self::segmentation::connected_components::ConnectedComponents;
pub use self::segmentation::threshold::Averaging;
pub use self::segmentation::threshold::OtsuClasses;
pub use self::segmentation::threshold::OtsuMiddleClass;
pub use self::segmentation::threshold::Threshold;
pub use self::segmentation::threshold::ThresholdEntry;
pub use self::segmentation::threshold::ThresholdMethod;
pub use self::segmentation::threshold::ThresholdValueSource;
pub use self::segmentation::watershed::SeedSource;
pub use self::segmentation::watershed::Watershed;
pub use self::spartial_transform::edm::DistanceTransform;

use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;

use evanalyzer_cfg::core_types::CitationMetadata;
use evanalyzer_cfg::core_types::InternalErrors;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionScope {
    Tile,
    WholeImage,
}

pub trait ImageAlgorithm: Send + Sync {
    /// Execute image processing algorithm
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors>;
    fn name(&self) -> &'static str;
    fn cite(&self) -> Vec<&'static CitationMetadata>;
    fn execution_scope(&self) -> ExecutionScope;

    /// Override to `false` only for the rare command that reads `scratch_pad`
    /// as actual data (not just workspace) - e.g. `ImageMath` configured to
    /// use the scratch pad as its second operand - where retyping it ahead of
    /// time would destroy the data it's about to read.
    fn scratch_is_workspace(&self) -> bool {
        true
    }

    /// `true` for commands that read `GlobalPipelineCache::annotated_objects`
    /// (the user's hand-annotated objects). The executor only fills that, and
    /// only runs the whole-image phase without any segmented objects, when a
    /// job contains such a command.
    fn uses_annotated_objects(&self) -> bool {
        false
    }

    /// `true` for measurement steps: they measure every image, also one
    /// without objects - the executor then still runs the whole-image phase
    /// and exports the image's measurements.
    fn measures(&self) -> bool {
        false
    }

    /// What a measurement step measures - one entry per value, built with
    /// [`Outputs`](crate::pipeline::pipeline_measure::Outputs). The editor
    /// offers them (e.g. to "Exclude Image"), the step reports its values
    /// under them ([`Self::measurement_output`]).
    fn measurement_outputs(&self) -> Vec<evanalyzer_cfg::core_types::MeasurementInfo> {
        Vec::new()
    }

    /// Its output `id` (see [`Self::measurement_outputs`]) - the
    /// description its value is reported under.
    fn measurement_output(&self, id: &str) -> evanalyzer_cfg::core_types::MeasurementInfo {
        let suffix = format!("/{id}");
        self.measurement_outputs()
            .into_iter()
            .find(|output| output.key.ends_with(&suffix))
            .unwrap_or_else(|| panic!("{} measures no {id:?}", self.name()))
    }

    /// The channel this command reads with its own z-projection or z-plane
    /// (a `LoadChannel` step not on Auto) - the executor reads it beside the
    /// channels before the pipelines run. `None`: it reads none.
    fn channel_variant(&self) -> Option<crate::pipeline::pipeline_cache::ChannelVariant> {
        None
    }

    /// Entry point pipeline dispatchers should call instead of [`Self::execute`]
    /// directly: prepares `ctx.scratch_pad` (per [`Self::scratch_is_workspace`])
    /// and then runs the command.
    fn run(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.scratch_is_workspace() {
            ctx.prepare_scratch_matching_image()?;
        }
        let start = std::time::Instant::now();
        let result = self.execute(ctx, cache);
        log::debug!(
            "step {}: {:.1} ms",
            self.name(),
            start.elapsed().as_secs_f64() * 1e3
        );
        result
    }
}
