#![allow(unused_imports)]

// Register algos
mod classification;
mod filters;
mod math;
mod morphology;
mod segmentation;
mod spartial_transform;

pub use self::classification::classify_rois::ClassifyRois;
pub use self::classification::coloc_rois::Colocalization;
pub use self::classification::voronoi::Voronoi;
pub use self::classification::extract_rois::ExtractRois;
pub use self::filters::blur::Blur;
pub use self::filters::blur_gaussian::GaussianBlur;
pub use self::filters::color_filter::ColorFilterCommand;
pub use self::filters::color_filter::HsvRange;
pub use self::filters::edge_detection_canny::EdgeDetectionCanny;
pub use self::filters::edge_detection_sobel::EdgeDetectionSobel;
pub use self::filters::enhance_contrast::EnhanceContrast;
pub use self::filters::hessian::Hessian;
pub use self::filters::hessian::HessianMode;
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
pub use self::math::image_cache::ImageCache;
pub use self::math::image_cache::ImageCacheMode;
pub use self::math::image_math::ImageMath;
pub use self::math::image_math::Operand;
pub use self::math::median_subtract::MedianSubtract;
pub use self::math::save_image::ImageSource;
pub use self::math::save_image::SaveImage;
pub use self::morphology::morphological_transformation::KernelShapes;
pub use self::morphology::morphological_transformation::MorphOps;
pub use self::morphology::morphological_transformation::MorphologicalCommand;
pub use self::segmentation::connected_components::ConnectedComponents;
pub use self::segmentation::threshold::Threshold;
pub use self::segmentation::threshold::ThresholdEntry;
pub use self::segmentation::threshold::ThresholdMethod;
pub use self::segmentation::watershed::Watershed;
pub use self::spartial_transform::edm::DistanceTransform;

use crate::pipeline::pipeline_cache::PipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;

use evanalyzer_cfg::core_types::InternalErrors;

pub trait ImageAlgorithm: Send + Sync {
    /// Execute image processing algorithm
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors>;
    fn name(&self) -> &'static str;
}
