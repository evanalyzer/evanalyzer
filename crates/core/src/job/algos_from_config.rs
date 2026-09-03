// @generated - do not edit by hand
use crate::algos::*;
use evanalyzer_cfg::settings::pipeline_command_settings::*;

// ============ ENUM FROM IMPLS ============

#[cfg(feature = "ai")]
impl From<ObjectAiObjectClassifierAiClassifyMatchHandlingSettings> for AiClassifyMatchHandling {
    fn from(_s: ObjectAiObjectClassifierAiClassifyMatchHandlingSettings) -> Self {
        match _s {
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::AddOutputClassIfMatch => {
                AiClassifyMatchHandling::AddOutputClassIfMatch
            }
            ObjectAiObjectClassifierAiClassifyMatchHandlingSettings::ReclassifyIfMatch => {
                AiClassifyMatchHandling::ReclassifyIfMatch
            }
        }
    }
}

impl From<FiltersIlluminationCorrectionApplyMethodSettings> for ApplyMethod {
    fn from(_s: FiltersIlluminationCorrectionApplyMethodSettings) -> Self {
        match _s {
            FiltersIlluminationCorrectionApplyMethodSettings::Divide => ApplyMethod::Divide,
            FiltersIlluminationCorrectionApplyMethodSettings::Subtract => ApplyMethod::Subtract,
        }
    }
}

impl From<SegmentationThresholdAveragingSettings> for Averaging {
    fn from(_s: SegmentationThresholdAveragingSettings) -> Self {
        match _s {
            SegmentationThresholdAveragingSettings::Mean => Averaging::Mean,
            SegmentationThresholdAveragingSettings::Median => Averaging::Median,
        }
    }
}

impl From<FiltersRollingBallBallTypeSettings> for BallType {
    fn from(_s: FiltersRollingBallBallTypeSettings) -> Self {
        match _s {
            FiltersRollingBallBallTypeSettings::Ball => BallType::Ball,
            FiltersRollingBallBallTypeSettings::Paraboloid => BallType::Paraboloid,
        }
    }
}

impl From<ObjectClassifyObjectsClassifyMatchHandlingSettings> for ClassifyMatchHandling {
    fn from(_s: ObjectClassifyObjectsClassifyMatchHandlingSettings) -> Self {
        match _s {
            ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch => {
                ClassifyMatchHandling::AddOutputClassIfMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfNotMatch => {
                ClassifyMatchHandling::AddOutputClassIfNotMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfMatch => {
                ClassifyMatchHandling::RemoveInputClassIfMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveInputClassIfNotMatch => {
                ClassifyMatchHandling::RemoveInputClassIfNotMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfMatch => {
                ClassifyMatchHandling::RemoveOutputClassIfMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveOutputClassIfNotMatch => {
                ClassifyMatchHandling::RemoveOutputClassIfNotMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfMatch => {
                ClassifyMatchHandling::RemoveAllClassesIfMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfNotMatch => {
                ClassifyMatchHandling::RemoveAllClassesIfNotMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch => {
                ClassifyMatchHandling::ReclassifyIfMatch
            }
            ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfNotMatch => {
                ClassifyMatchHandling::ReclassifyIfNotMatch
            }
        }
    }
}

impl From<ObjectColocObjectsColocMultiplicitySettings> for ColocMultiplicity {
    fn from(_s: ObjectColocObjectsColocMultiplicitySettings) -> Self {
        match _s {
            ObjectColocObjectsColocMultiplicitySettings::OneToOne => ColocMultiplicity::OneToOne,
            ObjectColocObjectsColocMultiplicitySettings::ManyToMany => {
                ColocMultiplicity::ManyToMany
            }
            ObjectColocObjectsColocMultiplicitySettings::MultiFor(v) => {
                ColocMultiplicity::MultiFor(v)
            }
        }
    }
}

impl From<FiltersIlluminationCorrectionCorrectionMethodSettings> for CorrectionMethod {
    fn from(_s: FiltersIlluminationCorrectionCorrectionMethodSettings) -> Self {
        match _s {
            FiltersIlluminationCorrectionCorrectionMethodSettings::Regular => {
                CorrectionMethod::Regular
            }
            FiltersIlluminationCorrectionCorrectionMethodSettings::Background => {
                CorrectionMethod::Background
            }
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

impl From<ObjectObjectMathObjectSetOperationSettings> for ObjectSetOperation {
    fn from(_s: ObjectObjectMathObjectSetOperationSettings) -> Self {
        match _s {
            ObjectObjectMathObjectSetOperationSettings::And => ObjectSetOperation::And,
            ObjectObjectMathObjectSetOperationSettings::Or => ObjectSetOperation::Or,
            ObjectObjectMathObjectSetOperationSettings::Xor => ObjectSetOperation::Xor,
            ObjectObjectMathObjectSetOperationSettings::Subtract => ObjectSetOperation::Subtract,
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

impl From<SegmentationThresholdOtsuClassesSettings> for OtsuClasses {
    fn from(_s: SegmentationThresholdOtsuClassesSettings) -> Self {
        match _s {
            SegmentationThresholdOtsuClassesSettings::Two => OtsuClasses::Two,
            SegmentationThresholdOtsuClassesSettings::Three { middle_class } => {
                OtsuClasses::Three {
                    middle_class: OtsuMiddleClass::from(middle_class),
                }
            }
        }
    }
}

impl From<SegmentationThresholdOtsuMiddleClassSettings> for OtsuMiddleClass {
    fn from(_s: SegmentationThresholdOtsuMiddleClassSettings) -> Self {
        match _s {
            SegmentationThresholdOtsuMiddleClassSettings::Foreground => OtsuMiddleClass::Foreground,
            SegmentationThresholdOtsuMiddleClassSettings::Background => OtsuMiddleClass::Background,
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

impl From<SegmentationWatershedSeedSourceSettings> for SeedSource {
    fn from(_s: SegmentationWatershedSeedSourceSettings) -> Self {
        match _s {
            SegmentationWatershedSeedSourceSettings::DistanceMap => SeedSource::DistanceMap,
            SegmentationWatershedSeedSourceSettings::Intensity => SeedSource::Intensity,
        }
    }
}

impl From<FiltersIlluminationCorrectionSmoothingMethodSettings> for SmoothingMethod {
    fn from(_s: FiltersIlluminationCorrectionSmoothingMethodSettings) -> Self {
        match _s {
            FiltersIlluminationCorrectionSmoothingMethodSettings::None => SmoothingMethod::None,
            FiltersIlluminationCorrectionSmoothingMethodSettings::Gaussian { sigma } => {
                SmoothingMethod::Gaussian {
                    sigma: sigma.clamp(0.1, 20.0),
                }
            }
            FiltersIlluminationCorrectionSmoothingMethodSettings::Median { radius } => {
                SmoothingMethod::Median { radius: radius }
            }
            FiltersIlluminationCorrectionSmoothingMethodSettings::FitPolynomial => {
                SmoothingMethod::FitPolynomial
            }
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
            SegmentationThresholdThresholdMethodSettings::Otsu { classes } => {
                ThresholdMethod::Otsu {
                    classes: OtsuClasses::from(classes),
                }
            }
            SegmentationThresholdThresholdMethodSettings::Percentile => ThresholdMethod::Percentile,
            SegmentationThresholdThresholdMethodSettings::RenyiEntropy => {
                ThresholdMethod::RenyiEntropy
            }
            SegmentationThresholdThresholdMethodSettings::Shanbhag => ThresholdMethod::Shanbhag,
            SegmentationThresholdThresholdMethodSettings::Yen => ThresholdMethod::Yen,
            SegmentationThresholdThresholdMethodSettings::RobustBackground {
                lower_outlier_fraction,
                upper_outlier_fraction,
                averaging_method,
                deviations_above_average,
            } => ThresholdMethod::RobustBackground {
                lower_outlier_fraction: lower_outlier_fraction,
                upper_outlier_fraction: upper_outlier_fraction,
                averaging_method: Averaging::from(averaging_method),
                deviations_above_average: deviations_above_average,
            },
        }
    }
}

impl From<SegmentationThresholdThresholdValueSourceSettings> for ThresholdValueSource {
    fn from(_s: SegmentationThresholdThresholdValueSourceSettings) -> Self {
        match _s {
            SegmentationThresholdThresholdValueSourceSettings::ActualImage => {
                ThresholdValueSource::ActualImage
            }
            SegmentationThresholdThresholdValueSourceSettings::RawImage => {
                ThresholdValueSource::RawImage
            }
            SegmentationThresholdThresholdValueSourceSettings::Memory(v) => {
                ThresholdValueSource::Memory(v)
            }
        }
    }
}

impl From<ObjectTransformObjectsTransformFunctionSettings> for TransformFunction {
    fn from(_s: ObjectTransformObjectsTransformFunctionSettings) -> Self {
        match _s {
            ObjectTransformObjectsTransformFunctionSettings::Scale { factor } => {
                TransformFunction::Scale {
                    factor: factor.clamp(0.0, 65535.0),
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::SnapArea { extra_size, unit } => {
                TransformFunction::SnapArea {
                    extra_size: extra_size.clamp(0.0, 65535.0),
                    unit: unit,
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::MinCircle { min_diameter, unit } => {
                TransformFunction::MinCircle {
                    min_diameter: min_diameter.clamp(0.0, 65535.0),
                    unit: unit,
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::DrawCircle { diameter, unit } => {
                TransformFunction::DrawCircle {
                    diameter: diameter.clamp(0.0, 65535.0),
                    unit: unit,
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::FittingEllipse { scale } => {
                TransformFunction::FittingEllipse {
                    scale: scale.clamp(0.0, 65535.0),
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::Expand { margin, unit } => {
                TransformFunction::Expand {
                    margin: margin.clamp(0.0, 65535.0),
                    unit: unit,
                }
            }
            ObjectTransformObjectsTransformFunctionSettings::Shrink { margin, unit } => {
                TransformFunction::Shrink {
                    margin: margin.clamp(0.0, 65535.0),
                    unit: unit,
                }
            }
        }
    }
}

#[cfg(feature = "ai")]
impl From<AiSegmentationUnetUNetOutputModeSettings> for UNetOutputMode {
    fn from(_s: AiSegmentationUnetUNetOutputModeSettings) -> Self {
        match _s {
            AiSegmentationUnetUNetOutputModeSettings::SoftmaxClasses => {
                UNetOutputMode::SoftmaxClasses
            }
            AiSegmentationUnetUNetOutputModeSettings::IndependentChannels => {
                UNetOutputMode::IndependentChannels
            }
        }
    }
}

// ============ STRUCT FROM IMPLS ============

#[cfg(feature = "ai")]
impl From<AiObjectClassifierSettings> for AiObjectClassifier {
    fn from(_s: AiObjectClassifierSettings) -> Self {
        AiObjectClassifier {
            model_path: _s.model_path,
            segmentation_mapping: _s
                .segmentation_mapping
                .into_iter()
                .map(|v| v.into())
                .collect(),
            origin_segmentation: _s
                .origin_segmentation
                .into_iter()
                .map(|v| v.into())
                .collect(),
            input_classes: _s.input_classes.into_iter().map(|v| v.into()).collect(),
            match_handling: AiClassifyMatchHandling::from(_s.match_handling),
        }
    }
}

impl From<BlurSettings> for Blur {
    fn from(_s: BlurSettings) -> Self {
        Blur {
            kernel_size: _s.kernel_size,
        }
    }
}

#[cfg(feature = "ai")]
impl From<CellposeSettings> for Cellpose {
    fn from(_s: CellposeSettings) -> Self {
        Cellpose {
            model_path: _s.model_path,
            object_class_id: _s.object_class_id,
            input_channels: _s.input_channels,
            probability_threshold: _s.probability_threshold.clamp(0.0, 1.0),
            flow_iterations: _s.flow_iterations,
            min_object_size: _s.min_object_size,
        }
    }
}

#[cfg(feature = "ai")]
impl From<ClassificationMappingSettings> for ClassificationMapping {
    fn from(_s: ClassificationMappingSettings) -> Self {
        ClassificationMapping {
            object_class: _s.object_class,
            output_class: _s.output_class,
        }
    }
}

impl From<ClassifyObjectsSettings> for ClassifyObjects {
    fn from(_s: ClassifyObjectsSettings) -> Self {
        ClassifyObjects {
            origin_segmentation: _s
                .origin_segmentation
                .into_iter()
                .map(|v| v.into())
                .collect(),
            input_classes: _s.input_classes.into_iter().map(|v| v.into()).collect(),
            match_handling: ClassifyMatchHandling::from(_s.match_handling),
            output_class: _s.output_class,
            overlapping_with: _s.overlapping_with,
            min_intersection_area: _s.min_intersection_area.clamp(0.0, 2147483600.0),
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
            multiplicity: ColocMultiplicity::from(_s.multiplicity),
            size_unit: _s.size_unit,
            min_coloc_area: _s.min_coloc_area,
            exclude_classes: _s.exclude_classes.into_iter().map(|v| v.into()).collect(),
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
        ConnectedComponents {
            min_size: _s.min_size,
        }
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

impl From<ExtractObjectsSettings> for ExtractObjects {
    fn from(_s: ExtractObjectsSettings) -> Self {
        ExtractObjects {
            max_objects_before_fail: _s.max_objects_before_fail,
        }
    }
}

impl From<FillHolesSettings> for FillHoles {
    fn from(_s: FillHolesSettings) -> Self {
        FillHoles {}
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

impl From<IlluminationCorrectionSettings> for IlluminationCorrection {
    fn from(_s: IlluminationCorrectionSettings) -> Self {
        IlluminationCorrection {
            method: CorrectionMethod::from(_s.method),
            block_size: _s.block_size,
            smoothing: SmoothingMethod::from(_s.smoothing),
            apply_method: ApplyMethod::from(_s.apply_method),
            rescale: _s.rescale,
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

impl From<ObjectMathSettings> for ObjectMath {
    fn from(_s: ObjectMathSettings) -> Self {
        ObjectMath {
            operation: ObjectSetOperation::from(_s.operation),
            input_class: _s.input_class,
            other_class: _s.other_class,
            other_filter_classes: _s
                .other_filter_classes
                .into_iter()
                .map(|v| v.into())
                .collect(),
            size_unit: _s.size_unit,
            min_overlap_area: _s.min_overlap_area,
            output_class: _s.output_class,
            keep_unmatched: _s.keep_unmatched,
        }
    }
}

#[cfg(feature = "ai")]
impl From<PixelClassifierSettings> for PixelClassifier {
    fn from(_s: PixelClassifierSettings) -> Self {
        PixelClassifier {
            model_path: _s.model_path,
            segmentation_mapping: _s
                .segmentation_mapping
                .into_iter()
                .map(|v| v.into())
                .collect(),
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
            name: _s.name,
            source: ImageSource::from(_s.source),
        }
    }
}

#[cfg(feature = "ai")]
impl From<SegmentationMappingSettings> for SegmentationMapping {
    fn from(_s: SegmentationMappingSettings) -> Self {
        SegmentationMapping {
            segmentation_class: _s.segmentation_class,
            object_class_id: _s.object_class_id,
        }
    }
}

#[cfg(feature = "ai")]
impl From<StardistSettings> for Stardist {
    fn from(_s: StardistSettings) -> Self {
        Stardist {
            model_path: _s.model_path,
            object_class_id: _s.object_class_id,
            probability_threshold: _s.probability_threshold.clamp(0.0, 1.0),
            nms_threshold: _s.nms_threshold.clamp(0.0, 1.0),
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
            value_source: ThresholdValueSource::from(_s.value_source),
        }
    }
}

impl From<TransformObjectsSettings> for TransformObjects {
    fn from(_s: TransformObjectsSettings) -> Self {
        TransformObjects {
            function: TransformFunction::from(_s.function),
            input_class: _s.input_class,
            output_class: _s.output_class,
        }
    }
}

#[cfg(feature = "ai")]
impl From<UNetSettings> for UNet {
    fn from(_s: UNetSettings) -> Self {
        UNet {
            model_path: _s.model_path,
            object_class_id: _s.object_class_id,
            probability_threshold: _s.probability_threshold.clamp(0.0, 1.0),
            output_mode: UNetOutputMode::from(_s.output_mode),
            foreground_channel: _s.foreground_channel,
            boundary_channel: _s.boundary_channel,
            boundary_threshold: _s.boundary_threshold.clamp(0.0, 1.0),
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
            maximum_finder_tolerance: _s.maximum_finder_tolerance.clamp(0.1, 20.0),
            smoothing_sigma: _s.smoothing_sigma.clamp(0.0, 10.0),
            min_object_size: _s.min_object_size,
            seed_source: SeedSource::from(_s.seed_source),
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
        #[cfg(feature = "ai")]
        PipelineCommand::AiObjectClassifier(settings) => {
            Ok(Box::new(crate::algos::AiObjectClassifier::from(settings)))
        }
        #[cfg(not(feature = "ai"))]
        PipelineCommand::AiObjectClassifier(_settings) => Err(InternalErrors::Generic(
            "This build was compiled without the ai feature; AiObjectClassifier is unavailable."
                .into(),
        )),
        PipelineCommand::Blur(settings) => Ok(Box::new(crate::algos::Blur::from(settings))),
        #[cfg(feature = "ai")]
        PipelineCommand::Cellpose(settings) => Ok(Box::new(crate::algos::Cellpose::from(settings))),
        #[cfg(not(feature = "ai"))]
        PipelineCommand::Cellpose(_settings) => Err(InternalErrors::Generic(
            "This build was compiled without the ai feature; Cellpose is unavailable.".into(),
        )),
        PipelineCommand::ClassifyObjects(settings) => {
            Ok(Box::new(crate::algos::ClassifyObjects::from(settings)))
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
        PipelineCommand::ExtractObjects(settings) => {
            Ok(Box::new(crate::algos::ExtractObjects::from(settings)))
        }
        PipelineCommand::FillHoles(settings) => {
            Ok(Box::new(crate::algos::FillHoles::from(settings)))
        }
        PipelineCommand::GaussianBlur(settings) => {
            Ok(Box::new(crate::algos::GaussianBlur::from(settings)))
        }
        PipelineCommand::Hessian(settings) => Ok(Box::new(crate::algos::Hessian::from(settings))),
        PipelineCommand::IlluminationCorrection(settings) => Ok(Box::new(
            crate::algos::IlluminationCorrection::from(settings),
        )),
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
        PipelineCommand::ObjectMath(settings) => {
            Ok(Box::new(crate::algos::ObjectMath::from(settings)))
        }
        #[cfg(feature = "ai")]
        PipelineCommand::PixelClassifier(settings) => {
            Ok(Box::new(crate::algos::PixelClassifier::from(settings)))
        }
        #[cfg(not(feature = "ai"))]
        PipelineCommand::PixelClassifier(_settings) => Err(InternalErrors::Generic(
            "This build was compiled without the ai feature; PixelClassifier is unavailable."
                .into(),
        )),
        PipelineCommand::RankFilter(settings) => {
            Ok(Box::new(crate::algos::RankFilter::from(settings)))
        }
        PipelineCommand::RollingBall(settings) => {
            Ok(Box::new(crate::algos::RollingBall::from(settings)))
        }
        PipelineCommand::SaveImage(settings) => {
            Ok(Box::new(crate::algos::SaveImage::from(settings)))
        }
        #[cfg(feature = "ai")]
        PipelineCommand::Stardist(settings) => Ok(Box::new(crate::algos::Stardist::from(settings))),
        #[cfg(not(feature = "ai"))]
        PipelineCommand::Stardist(_settings) => Err(InternalErrors::Generic(
            "This build was compiled without the ai feature; Stardist is unavailable.".into(),
        )),
        PipelineCommand::StructureTensor(settings) => {
            Ok(Box::new(crate::algos::StructureTensor::from(settings)))
        }
        PipelineCommand::Threshold(settings) => {
            Ok(Box::new(crate::algos::Threshold::from(settings)))
        }
        PipelineCommand::TransformObjects(settings) => {
            Ok(Box::new(crate::algos::TransformObjects::from(settings)))
        }
        #[cfg(feature = "ai")]
        PipelineCommand::UNet(settings) => Ok(Box::new(crate::algos::UNet::from(settings))),
        #[cfg(not(feature = "ai"))]
        PipelineCommand::UNet(_settings) => Err(InternalErrors::Generic(
            "This build was compiled without the ai feature; UNet is unavailable.".into(),
        )),
        PipelineCommand::Voronoi(settings) => Ok(Box::new(crate::algos::Voronoi::from(settings))),
        PipelineCommand::Watershed(settings) => {
            Ok(Box::new(crate::algos::Watershed::from(settings)))
        }
        PipelineCommand::WeightedDeviation(settings) => {
            Ok(Box::new(crate::algos::WeightedDeviation::from(settings)))
        }
    }
}
