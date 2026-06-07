// @generated - do not edit by hand
use crate::algos::*;
use evanalyzer_cfg::settings::pipeline_command_settings::*;

// ============ ENUM FROM IMPLS ============

impl From<FiltersRollingBallBallTypeSettings> for BallType {
    fn from(_s: FiltersRollingBallBallTypeSettings) -> Self {
        match _s {
            FiltersRollingBallBallTypeSettings::Ball => BallType::Ball,
            FiltersRollingBallBallTypeSettings::Paraboloid => BallType::Paraboloid,
        }
    }
}

impl From<ClassificationClassifyRoisClassifyMatchHandlingSettings> for ClassifyMatchHandling {
    fn from(_s: ClassificationClassifyRoisClassifyMatchHandlingSettings) -> Self {
        match _s {
            ClassificationClassifyRoisClassifyMatchHandlingSettings::AddOutputClassIfMatch => ClassifyMatchHandling::AddOutputClassIfMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::AddOutputClassIfNotMatch => ClassifyMatchHandling::AddOutputClassIfNotMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveInputClassIfMatch => ClassifyMatchHandling::RemoveInputClassIfMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveInputClassIfNotMatch => ClassifyMatchHandling::RemoveInputClassIfNotMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveOutputClassIfMatch => ClassifyMatchHandling::RemoveOutputClassIfMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveOutputClassIfNotMatch => ClassifyMatchHandling::RemoveOutputClassIfNotMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveAllClassesIfMatch => ClassifyMatchHandling::RemoveAllClassesIfMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::RemoveAllClassesIfNotMatch => ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::ReclassifyIfMatch => ClassifyMatchHandling::ReclassifyIfMatch,
            ClassificationClassifyRoisClassifyMatchHandlingSettings::ReclassifyIfNotMatch => ClassifyMatchHandling::ReclassifyIfNotMatch,
        }
    }
}

impl From<FiltersHessianHessianModeSettings> for HessianMode {
    fn from(_s: FiltersHessianHessianModeSettings) -> Self {
        match _s {
            FiltersHessianHessianModeSettings::Determinant => HessianMode::Determinant,
            FiltersHessianHessianModeSettings::EigenvaluesX => HessianMode::EigenvaluesX,
            FiltersHessianHessianModeSettings::EigenvaluesY => HessianMode::EigenvaluesY,
        }
    }
}

impl From<MathImageCacheImageCacheModeSettings> for ImageCacheMode {
    fn from(_s: MathImageCacheImageCacheModeSettings) -> Self {
        match _s {
            MathImageCacheImageCacheModeSettings::Store => ImageCacheMode::Store,
            MathImageCacheImageCacheModeSettings::Load => ImageCacheMode::Load,
        }
    }
}

impl From<MathSaveImageImageSourceSettings> for ImageSource {
    fn from(_s: MathSaveImageImageSourceSettings) -> Self {
        match _s {
            MathSaveImageImageSourceSettings::Image => ImageSource::Image,
            MathSaveImageImageSourceSettings::InstanceMap => ImageSource::InstanceMap,
            MathSaveImageImageSourceSettings::SegmentationMask => ImageSource::SegmentationMask,
        }
    }
}

impl From<FiltersIntensityTransformIntensityTransformModeSettings> for IntensityTransformMode {
    fn from(_s: FiltersIntensityTransformIntensityTransformModeSettings) -> Self {
        match _s {
            FiltersIntensityTransformIntensityTransformModeSettings::Automatic => {
                IntensityTransformMode::Automatic
            }
            FiltersIntensityTransformIntensityTransformModeSettings::Manual => {
                IntensityTransformMode::Manual
            }
        }
    }
}

impl From<MorphologyMorphologicalTransformationKernelShapesSettings> for KernelShapes {
    fn from(_s: MorphologyMorphologicalTransformationKernelShapesSettings) -> Self {
        match _s {
            MorphologyMorphologicalTransformationKernelShapesSettings::Box => KernelShapes::Box,
            MorphologyMorphologicalTransformationKernelShapesSettings::Ellipse => {
                KernelShapes::Ellipse
            }
            MorphologyMorphologicalTransformationKernelShapesSettings::Cross => KernelShapes::Cross,
        }
    }
}

impl From<MorphologyMorphologicalTransformationMorphOpsSettings> for MorphOps {
    fn from(_s: MorphologyMorphologicalTransformationMorphOpsSettings) -> Self {
        match _s {
            MorphologyMorphologicalTransformationMorphOpsSettings::Dilate => MorphOps::Dilate,
            MorphologyMorphologicalTransformationMorphOpsSettings::Erode => MorphOps::Erode,
            MorphologyMorphologicalTransformationMorphOpsSettings::Open => MorphOps::Open,
            MorphologyMorphologicalTransformationMorphOpsSettings::Close => MorphOps::Close,
        }
    }
}

impl From<MathImageMathOperandSettings> for Operand {
    fn from(_s: MathImageMathOperandSettings) -> Self {
        match _s {
            MathImageMathOperandSettings::None => Operand::None,
            MathImageMathOperandSettings::Invert => Operand::Invert,
            MathImageMathOperandSettings::Add => Operand::Add,
            MathImageMathOperandSettings::Subtract => Operand::Subtract,
            MathImageMathOperandSettings::Multiply => Operand::Multiply,
            MathImageMathOperandSettings::Divide => Operand::Divide,
            MathImageMathOperandSettings::And => Operand::And,
            MathImageMathOperandSettings::Or => Operand::Or,
            MathImageMathOperandSettings::Xor => Operand::Xor,
            MathImageMathOperandSettings::Min => Operand::Min,
            MathImageMathOperandSettings::Max => Operand::Max,
            MathImageMathOperandSettings::Average => Operand::Average,
            MathImageMathOperandSettings::DifferenceType => Operand::DifferenceType,
        }
    }
}

impl From<FiltersRankFilterRankFilterTypeSettings> for RankFilterType {
    fn from(_s: FiltersRankFilterRankFilterTypeSettings) -> Self {
        match _s {
            FiltersRankFilterRankFilterTypeSettings::Median => RankFilterType::Median,
            FiltersRankFilterRankFilterTypeSettings::Min => RankFilterType::Min,
            FiltersRankFilterRankFilterTypeSettings::Max => RankFilterType::Max,
            FiltersRankFilterRankFilterTypeSettings::Mean => RankFilterType::Mean,
            FiltersRankFilterRankFilterTypeSettings::Outliers(v) => RankFilterType::Outliers(v),
        }
    }
}

impl From<FiltersStructureTensorTensorModeSettings> for TensorMode {
    fn from(_s: FiltersStructureTensorTensorModeSettings) -> Self {
        match _s {
            FiltersStructureTensorTensorModeSettings::EigenvaluesX => TensorMode::EigenvaluesX,
            FiltersStructureTensorTensorModeSettings::EigenvaluesY => TensorMode::EigenvaluesY,
            FiltersStructureTensorTensorModeSettings::Coherence => TensorMode::Coherence,
        }
    }
}

impl From<SegmentationThresholdThresholdMethodSettings> for ThresholdMethod {
    fn from(_s: SegmentationThresholdThresholdMethodSettings) -> Self {
        match _s {
            SegmentationThresholdThresholdMethodSettings::None => ThresholdMethod::None,
            SegmentationThresholdThresholdMethodSettings::Manual => ThresholdMethod::Manual,
            SegmentationThresholdThresholdMethodSettings::Li => ThresholdMethod::Li,
            SegmentationThresholdThresholdMethodSettings::MinError => ThresholdMethod::MinError,
            SegmentationThresholdThresholdMethodSettings::Triangle => ThresholdMethod::Triangle,
            SegmentationThresholdThresholdMethodSettings::Moments => ThresholdMethod::Moments,
            SegmentationThresholdThresholdMethodSettings::Huang => ThresholdMethod::Huang,
            SegmentationThresholdThresholdMethodSettings::Intermodes => ThresholdMethod::Intermodes,
            SegmentationThresholdThresholdMethodSettings::IsoData => ThresholdMethod::IsoData,
            SegmentationThresholdThresholdMethodSettings::MaxEntropy => ThresholdMethod::MaxEntropy,
            SegmentationThresholdThresholdMethodSettings::Mean => ThresholdMethod::Mean,
            SegmentationThresholdThresholdMethodSettings::Minimum => ThresholdMethod::Minimum,
            SegmentationThresholdThresholdMethodSettings::Otsu => ThresholdMethod::Otsu,
            SegmentationThresholdThresholdMethodSettings::Percentile => ThresholdMethod::Percentile,
            SegmentationThresholdThresholdMethodSettings::RenyiEntropy => {
                ThresholdMethod::RenyiEntropy
            }
            SegmentationThresholdThresholdMethodSettings::Shanbhag => ThresholdMethod::Shanbhag,
            SegmentationThresholdThresholdMethodSettings::Yen => ThresholdMethod::Yen,
        }
    }
}

// ============ STRUCT FROM IMPLS ============

impl From<BlurSettings> for Blur {
    fn from(_s: BlurSettings) -> Self {
        Blur {
            kernel_size: _s.kernel_size,
        }
    }
}

impl From<ClassifyRoisSettings> for ClassifyRois {
    fn from(_s: ClassifyRoisSettings) -> Self {
        ClassifyRois {
            origin_segmentation: _s
                .origin_segmentation
                .into_iter()
                .map(|v| v.into())
                .collect(),
            input_classes: _s.input_classes.into_iter().map(|v| v.into()).collect(),
            match_handling: ClassifyMatchHandling::from(_s.match_handling),
            output_class: _s.output_class,
            size_unit: _s.size_unit,
            min_area: _s.min_area.clamp(0.0, 2147483600.0),
            max_area: _s.max_area.clamp(0.0, 2147483600.0),
            min_circularity: _s.min_circularity.clamp(0.0, 1.0),
            max_circularity: _s.max_circularity.clamp(0.0, 1.0),
            min_solidity: _s.min_solidity.clamp(0.0, 1.0),
            max_solidity: _s.max_solidity.clamp(0.0, 1.0),
            min_aspect_ratio: _s.min_aspect_ratio.clamp(0.0, 2147483600.0),
            max_aspect_ratio: _s.max_aspect_ratio.clamp(0.0, 2147483600.0),
            min_eccentricity: _s.min_eccentricity.clamp(0.0, 1.0),
            max_eccentricity: _s.max_eccentricity.clamp(0.0, 1.0),
            min_feret: _s.min_feret.clamp(0.0, 2147483600.0),
            max_feret: _s.max_feret.clamp(0.0, 2147483600.0),
            allow_edge_touching: _s.allow_edge_touching,
        }
    }
}

impl From<ColocalizationSettings> for Colocalization {
    fn from(_s: ColocalizationSettings) -> Self {
        Colocalization {
            classes_to_coloc: _s.classes_to_coloc.into_iter().map(|v| v.into()).collect(),
            filter_classes: _s.filter_classes.into_iter().map(|v| v.into()).collect(),
            class_for_overlapping_areas: _s.class_for_overlapping_areas,
            allow_multi_object_coloc: _s.allow_multi_object_coloc,
            size_unit: _s.size_unit,
            min_coloc_area: _s.min_coloc_area,
        }
    }
}

impl From<ColorFilterCommandSettings> for ColorFilterCommand {
    fn from(_s: ColorFilterCommandSettings) -> Self {
        ColorFilterCommand {
            range: HsvRange::from(_s.range),
        }
    }
}

impl From<ConnectedComponentsSettings> for ConnectedComponents {
    fn from(_s: ConnectedComponentsSettings) -> Self {
        ConnectedComponents {}
    }
}

impl From<DistanceTransformSettings> for DistanceTransform {
    fn from(_s: DistanceTransformSettings) -> Self {
        DistanceTransform {
            threshold: _s.threshold,
            edges_are_background: _s.edges_are_background,
        }
    }
}

impl From<EdgeDetectionCannySettings> for EdgeDetectionCanny {
    fn from(_s: EdgeDetectionCannySettings) -> Self {
        EdgeDetectionCanny {
            kernel_size: _s.kernel_size,
            threshold_min: _s.threshold_min,
            threshold_max: _s.threshold_max,
        }
    }
}

impl From<EdgeDetectionSobelSettings> for EdgeDetectionSobel {
    fn from(_s: EdgeDetectionSobelSettings) -> Self {
        EdgeDetectionSobel {
            kernel_size: _s.kernel_size,
        }
    }
}

impl From<EnhanceContrastSettings> for EnhanceContrast {
    fn from(_s: EnhanceContrastSettings) -> Self {
        EnhanceContrast {
            saturated_pixels: _s.saturated_pixels,
            normalize: _s.normalize,
            equalize_histogram: _s.equalize_histogram,
        }
    }
}

impl From<ExtractRoisSettings> for ExtractRois {
    fn from(_s: ExtractRoisSettings) -> Self {
        ExtractRois {
            max_objects_before_fail: _s.max_objects_before_fail,
        }
    }
}

impl From<GaussianBlurSettings> for GaussianBlur {
    fn from(_s: GaussianBlurSettings) -> Self {
        GaussianBlur {
            kernel_size: _s.kernel_size,
            sigma: _s.sigma.clamp(0.1, 5.0),
        }
    }
}

impl From<HessianSettings> for Hessian {
    fn from(_s: HessianSettings) -> Self {
        Hessian {
            mode: HessianMode::from(_s.mode),
        }
    }
}

impl From<HsvRangeSettings> for HsvRange {
    fn from(_s: HsvRangeSettings) -> Self {
        HsvRange {
            min_h: _s.min_h,
            max_h: _s.max_h,
            min_s: _s.min_s,
            max_s: _s.max_s,
            min_v: _s.min_v,
            max_v: _s.max_v,
        }
    }
}

impl From<ImageCacheSettings> for ImageCache {
    fn from(_s: ImageCacheSettings) -> Self {
        ImageCache {
            mode: ImageCacheMode::from(_s.mode),
            address: _s.address,
        }
    }
}

impl From<ImageMathSettings> for ImageMath {
    fn from(_s: ImageMathSettings) -> Self {
        ImageMath {
            operand: Operand::from(_s.operand),
            second_image_address: _s.second_image_address,
            swap_operands: _s.swap_operands,
        }
    }
}

impl From<IntensityTransformationSettings> for IntensityTransformation {
    fn from(_s: IntensityTransformationSettings) -> Self {
        IntensityTransformation {
            mode: IntensityTransformMode::from(_s.mode),
            contrast: _s.contrast,
            brightness: _s.brightness,
        }
    }
}

impl From<LaplacianSettings> for Laplacian {
    fn from(_s: LaplacianSettings) -> Self {
        Laplacian {
            kernel_size: _s.kernel_size,
        }
    }
}

impl From<MedianSubtractSettings> for MedianSubtract {
    fn from(_s: MedianSubtractSettings) -> Self {
        MedianSubtract { radius: _s.radius }
    }
}

impl From<MorphologicalCommandSettings> for MorphologicalCommand {
    fn from(_s: MorphologicalCommandSettings) -> Self {
        MorphologicalCommand {
            op: MorphOps::from(_s.op),
            kernel_size: _s.kernel_size,
            kernel_shape: KernelShapes::from(_s.kernel_shape),
            use_grayscale: _s.use_grayscale,
        }
    }
}

impl From<RankFilterSettings> for RankFilter {
    fn from(_s: RankFilterSettings) -> Self {
        RankFilter {
            radius: _s.radius,
            filter_type: RankFilterType::from(_s.filter_type),
        }
    }
}

impl From<RollingBallSettings> for RollingBall {
    fn from(_s: RollingBallSettings) -> Self {
        RollingBall {
            radius: _s.radius.clamp(1.0, 64.0),
            ball_type: BallType::from(_s.ball_type),
            pre_smooth: _s.pre_smooth,
        }
    }
}

impl From<SaveImageSettings> for SaveImage {
    fn from(_s: SaveImageSettings) -> Self {
        SaveImage {
            path: _s.path,
            source: ImageSource::from(_s.source),
        }
    }
}

impl From<StructureTensorSettings> for StructureTensor {
    fn from(_s: StructureTensorSettings) -> Self {
        StructureTensor {
            mode: TensorMode::from(_s.mode),
            kernel_size: _s.kernel_size,
            sigma: _s.sigma,
        }
    }
}

impl From<ThresholdSettings> for Threshold {
    fn from(_s: ThresholdSettings) -> Self {
        Threshold {
            thresholds: _s.thresholds.into_iter().map(|v| v.into()).collect(),
        }
    }
}

impl From<ThresholdEntrySettings> for ThresholdEntry {
    fn from(_s: ThresholdEntrySettings) -> Self {
        ThresholdEntry {
            method: ThresholdMethod::from(_s.method),
            min_threshold: _s.min_threshold.clamp(0.0, 65535.0),
            max_threshold: _s.max_threshold.clamp(0.0, 65535.0),
            unit: _s.unit,
            object_class_id: _s.object_class_id,
        }
    }
}

impl From<VoronoiSettings> for Voronoi {
    fn from(_s: VoronoiSettings) -> Self {
        Voronoi {
            centers: _s.centers,
            center_filter_classes: _s
                .center_filter_classes
                .into_iter()
                .map(|v| v.into())
                .collect(),
            mask: _s.mask,
            mask_filter_classes: _s
                .mask_filter_classes
                .into_iter()
                .map(|v| v.into())
                .collect(),
            output_class: _s.output_class,
            unit: _s.unit,
            max_radius: _s.max_radius,
            exclude_areas_at_the_edges: _s.exclude_areas_at_the_edges,
            exclude_areas_with_no_center: _s.exclude_areas_with_no_center,
        }
    }
}

impl From<WatershedSettings> for Watershed {
    fn from(_s: WatershedSettings) -> Self {
        Watershed {
            maximum_finder_tolerance: _s.maximum_finder_tolerance.clamp(0.1, 1.0),
        }
    }
}

impl From<WeightedDeviationSettings> for WeightedDeviation {
    fn from(_s: WeightedDeviationSettings) -> Self {
        WeightedDeviation {
            kernel_size: _s.kernel_size,
            sigma: _s.sigma,
        }
    }
}

// ============ INTO ALGORITHM ============

use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;

pub fn into_algorithm(cmd: PipelineCommand) -> Result<Box<dyn ImageAlgorithm>, InternalErrors> {
    match cmd {
        PipelineCommand::Blur(settings) => Ok(Box::new(crate::algos::Blur::from(settings))),
        PipelineCommand::ClassifyRois(settings) => {
            Ok(Box::new(crate::algos::ClassifyRois::from(settings)))
        }
        PipelineCommand::Colocalization(settings) => {
            Ok(Box::new(crate::algos::Colocalization::from(settings)))
        }
        PipelineCommand::ColorFilterCommand(settings) => {
            Ok(Box::new(crate::algos::ColorFilterCommand::from(settings)))
        }
        PipelineCommand::ConnectedComponents(settings) => {
            Ok(Box::new(crate::algos::ConnectedComponents::from(settings)))
        }
        PipelineCommand::DistanceTransform(settings) => {
            Ok(Box::new(crate::algos::DistanceTransform::from(settings)))
        }
        PipelineCommand::EdgeDetectionCanny(settings) => {
            Ok(Box::new(crate::algos::EdgeDetectionCanny::from(settings)))
        }
        PipelineCommand::EdgeDetectionSobel(settings) => {
            Ok(Box::new(crate::algos::EdgeDetectionSobel::from(settings)))
        }
        PipelineCommand::EnhanceContrast(settings) => {
            Ok(Box::new(crate::algos::EnhanceContrast::from(settings)))
        }
        PipelineCommand::ExtractRois(settings) => {
            Ok(Box::new(crate::algos::ExtractRois::from(settings)))
        }
        PipelineCommand::GaussianBlur(settings) => {
            Ok(Box::new(crate::algos::GaussianBlur::from(settings)))
        }
        PipelineCommand::Hessian(settings) => Ok(Box::new(crate::algos::Hessian::from(settings))),
        PipelineCommand::ImageCache(settings) => {
            Ok(Box::new(crate::algos::ImageCache::from(settings)))
        }
        PipelineCommand::ImageMath(settings) => {
            Ok(Box::new(crate::algos::ImageMath::from(settings)))
        }
        PipelineCommand::IntensityTransformation(settings) => Ok(Box::new(
            crate::algos::IntensityTransformation::from(settings),
        )),
        PipelineCommand::Laplacian(settings) => {
            Ok(Box::new(crate::algos::Laplacian::from(settings)))
        }
        PipelineCommand::MedianSubtract(settings) => {
            Ok(Box::new(crate::algos::MedianSubtract::from(settings)))
        }
        PipelineCommand::MorphologicalCommand(settings) => {
            Ok(Box::new(crate::algos::MorphologicalCommand::from(settings)))
        }
        PipelineCommand::RankFilter(settings) => {
            Ok(Box::new(crate::algos::RankFilter::from(settings)))
        }
        PipelineCommand::RollingBall(settings) => {
            Ok(Box::new(crate::algos::RollingBall::from(settings)))
        }
        PipelineCommand::SaveImage(settings) => {
            Ok(Box::new(crate::algos::SaveImage::from(settings)))
        }
        PipelineCommand::StructureTensor(settings) => {
            Ok(Box::new(crate::algos::StructureTensor::from(settings)))
        }
        PipelineCommand::Threshold(settings) => {
            Ok(Box::new(crate::algos::Threshold::from(settings)))
        }
        PipelineCommand::Voronoi(settings) => Ok(Box::new(crate::algos::Voronoi::from(settings))),
        PipelineCommand::Watershed(settings) => {
            Ok(Box::new(crate::algos::Watershed::from(settings)))
        }
        PipelineCommand::WeightedDeviation(settings) => {
            Ok(Box::new(crate::algos::WeightedDeviation::from(settings)))
        }
    }
}
