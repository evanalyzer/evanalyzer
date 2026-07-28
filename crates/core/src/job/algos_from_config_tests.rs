//! Tests for `algos_from_config.rs`, kept in a *separate* file rather than a
//! nested `#[cfg(test)] mod tests` inside that file - `algos_from_config.rs`
//! is a real generated-code output (`crates/cfg/build/pipeline_commands_generator.rs`
//! writes it via a whole-file `write_if_changed`, keyed off changes anywhere
//! under `crates/core/src/algos`), so anything appended directly into it
//! would be silently wiped the next time that generator runs. This file is
//! untouched by the generator and still gets full access to `algos_from_config`'s
//! private items because it's a sibling module under the same parent (`job`).
use super::algos_from_config::into_algorithm;
use crate::algos::*;
use evanalyzer_cfg::core_types::{
    ImageAddress, ObjectClass, PixelUnits, SegmentationClass, SizeUnits,
};
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::*;
use std::path::PathBuf;

// ---- Enum conversions --------------------------------------------

#[test]
fn ball_type_ball_setting_converts_to_balltype_ball() {
    let result: BallType = FiltersRollingBallBallTypeSettings::Ball.into();
    assert!(result == BallType::Ball);
}

#[test]
fn ball_type_paraboloid_setting_converts_to_balltype_paraboloid() {
    let result: BallType = FiltersRollingBallBallTypeSettings::Paraboloid.into();
    assert!(result == BallType::Paraboloid);
}

#[test]
fn classify_match_handling_first_variant_converts() {
    let result: ClassifyMatchHandling =
        ClassificationClassifyObjectsClassifyMatchHandlingSettings::AddOutputClassIfMatch.into();
    assert!(matches!(
        result,
        ClassifyMatchHandling::AddOutputClassIfMatch
    ));
}

#[test]
fn classify_match_handling_middle_variant_converts() {
    let result: ClassifyMatchHandling =
        ClassificationClassifyObjectsClassifyMatchHandlingSettings::RemoveAllClassesIfMatch.into();
    assert!(matches!(
        result,
        ClassifyMatchHandling::RemoveAllClassesIfMatch
    ));
}

#[test]
fn classify_match_handling_last_variant_converts() {
    let result: ClassifyMatchHandling =
        ClassificationClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfNotMatch.into();
    assert!(matches!(
        result,
        ClassifyMatchHandling::ReclassifyIfNotMatch
    ));
}

#[test]
fn hessian_mode_determinant_setting_converts() {
    let result: HessianMode = FiltersHessianHessianModeSettings::Determinant.into();
    assert_eq!(result, HessianMode::Determinant);
}

#[test]
fn hessian_mode_eigenvaluesx_setting_converts() {
    let result: HessianMode = FiltersHessianHessianModeSettings::EigenvaluesX.into();
    assert_eq!(result, HessianMode::EigenvaluesX);
}

#[test]
fn hessian_mode_eigenvaluesy_setting_converts() {
    let result: HessianMode = FiltersHessianHessianModeSettings::EigenvaluesY.into();
    assert_eq!(result, HessianMode::EigenvaluesY);
}

#[test]
fn image_cache_mode_store_setting_converts() {
    let result: ImageCacheMode = MathImageCacheImageCacheModeSettings::Store.into();
    assert!(matches!(result, ImageCacheMode::Store));
}

#[test]
fn image_cache_mode_load_setting_converts() {
    let result: ImageCacheMode = MathImageCacheImageCacheModeSettings::Load.into();
    assert!(matches!(result, ImageCacheMode::Load));
}

#[test]
fn image_source_image_setting_converts() {
    let result: ImageSource = MathSaveImageImageSourceSettings::Image.into();
    assert!(result == ImageSource::Image);
}

#[test]
fn image_source_instance_map_setting_converts() {
    let result: ImageSource = MathSaveImageImageSourceSettings::InstanceMap.into();
    assert!(result == ImageSource::InstanceMap);
}

#[test]
fn image_source_segmentation_mask_setting_converts() {
    let result: ImageSource = MathSaveImageImageSourceSettings::SegmentationMask.into();
    assert!(result == ImageSource::SegmentationMask);
}

#[test]
fn intensity_transform_mode_automatic_setting_converts() {
    let result: IntensityTransformMode =
        FiltersIntensityTransformIntensityTransformModeSettings::Automatic.into();
    assert_eq!(result, IntensityTransformMode::Automatic);
}

#[test]
fn intensity_transform_mode_manual_setting_converts() {
    let result: IntensityTransformMode =
        FiltersIntensityTransformIntensityTransformModeSettings::Manual.into();
    assert_eq!(result, IntensityTransformMode::Manual);
}

#[test]
fn kernel_shapes_box_setting_converts() {
    let result: KernelShapes =
        MorphologyMorphologicalTransformationKernelShapesSettings::Box.into();
    assert!(matches!(result, KernelShapes::Box));
}

#[test]
fn kernel_shapes_ellipse_setting_converts() {
    let result: KernelShapes =
        MorphologyMorphologicalTransformationKernelShapesSettings::Ellipse.into();
    assert!(matches!(result, KernelShapes::Ellipse));
}

#[test]
fn kernel_shapes_cross_setting_converts() {
    let result: KernelShapes =
        MorphologyMorphologicalTransformationKernelShapesSettings::Cross.into();
    assert!(matches!(result, KernelShapes::Cross));
}

#[test]
fn morph_ops_dilate_setting_converts() {
    let result: MorphOps = MorphologyMorphologicalTransformationMorphOpsSettings::Dilate.into();
    assert!(matches!(result, MorphOps::Dilate));
}

#[test]
fn morph_ops_erode_setting_converts() {
    let result: MorphOps = MorphologyMorphologicalTransformationMorphOpsSettings::Erode.into();
    assert!(matches!(result, MorphOps::Erode));
}

#[test]
fn morph_ops_open_setting_converts() {
    let result: MorphOps = MorphologyMorphologicalTransformationMorphOpsSettings::Open.into();
    assert!(matches!(result, MorphOps::Open));
}

#[test]
fn morph_ops_close_setting_converts() {
    let result: MorphOps = MorphologyMorphologicalTransformationMorphOpsSettings::Close.into();
    assert!(matches!(result, MorphOps::Close));
}

#[test]
fn operand_none_setting_converts() {
    let result: Operand = MathImageMathOperandSettings::None.into();
    assert_eq!(result, Operand::None);
}

#[test]
fn operand_subtract_setting_converts() {
    let result: Operand = MathImageMathOperandSettings::Subtract.into();
    assert_eq!(result, Operand::Subtract);
}

#[test]
fn operand_difference_type_setting_converts() {
    let result: Operand = MathImageMathOperandSettings::DifferenceType.into();
    assert_eq!(result, Operand::DifferenceType);
}

#[test]
fn rank_filter_type_median_setting_converts() {
    let result: RankFilterType = FiltersRankFilterRankFilterTypeSettings::Median.into();
    assert_eq!(result, RankFilterType::Median);
}

#[test]
fn rank_filter_type_mean_setting_converts() {
    let result: RankFilterType = FiltersRankFilterRankFilterTypeSettings::Mean.into();
    assert_eq!(result, RankFilterType::Mean);
}

#[test]
fn rank_filter_type_outliers_setting_converts_and_preserves_value() {
    let result: RankFilterType = FiltersRankFilterRankFilterTypeSettings::Outliers(3.5).into();
    assert_eq!(result, RankFilterType::Outliers(3.5));
}

#[test]
fn object_set_operation_and_setting_converts() {
    let result: ObjectSetOperation = ClassificationObjectMathObjectSetOperationSettings::And.into();
    assert!(matches!(result, ObjectSetOperation::And));
}

#[test]
fn object_set_operation_or_setting_converts() {
    let result: ObjectSetOperation = ClassificationObjectMathObjectSetOperationSettings::Or.into();
    assert!(matches!(result, ObjectSetOperation::Or));
}

#[test]
fn object_set_operation_xor_setting_converts() {
    let result: ObjectSetOperation = ClassificationObjectMathObjectSetOperationSettings::Xor.into();
    assert!(matches!(result, ObjectSetOperation::Xor));
}

#[test]
fn object_set_operation_subtract_setting_converts() {
    let result: ObjectSetOperation =
        ClassificationObjectMathObjectSetOperationSettings::Subtract.into();
    assert!(matches!(result, ObjectSetOperation::Subtract));
}

#[test]
fn tensor_mode_eigenvaluesx_setting_converts() {
    let result: TensorMode = FiltersStructureTensorTensorModeSettings::EigenvaluesX.into();
    assert!(matches!(result, TensorMode::EigenvaluesX));
}

#[test]
fn tensor_mode_eigenvaluesy_setting_converts() {
    let result: TensorMode = FiltersStructureTensorTensorModeSettings::EigenvaluesY.into();
    assert!(matches!(result, TensorMode::EigenvaluesY));
}

#[test]
fn tensor_mode_coherence_setting_converts() {
    let result: TensorMode = FiltersStructureTensorTensorModeSettings::Coherence.into();
    assert!(matches!(result, TensorMode::Coherence));
}

#[test]
fn threshold_method_none_setting_converts() {
    let result: ThresholdMethod = SegmentationThresholdThresholdMethodSettings::None.into();
    assert!(matches!(result, ThresholdMethod::None));
}

#[test]
fn threshold_method_otsu_setting_converts() {
    let result: ThresholdMethod = SegmentationThresholdThresholdMethodSettings::Otsu.into();
    assert!(matches!(result, ThresholdMethod::Otsu));
}

#[test]
fn threshold_method_yen_setting_converts() {
    let result: ThresholdMethod = SegmentationThresholdThresholdMethodSettings::Yen.into();
    assert!(matches!(result, ThresholdMethod::Yen));
}

#[test]
fn transform_function_scale_setting_converts_and_clamps_factor() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::Scale { factor: 999999.0 }.into();
    match result {
        TransformFunction::Scale { factor } => assert_eq!(factor, 65535.0),
        _ => panic!("expected Scale variant"),
    }
}

#[test]
fn transform_function_scale_setting_clamps_negative_factor_to_zero() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::Scale { factor: -10.0 }.into();
    match result {
        TransformFunction::Scale { factor } => assert_eq!(factor, 0.0),
        _ => panic!("expected Scale variant"),
    }
}

#[test]
fn transform_function_snap_area_setting_converts_and_carries_unit() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::SnapArea {
            extra_size: 12.0,
            unit: SizeUnits::Pixels,
        }
        .into();
    match result {
        TransformFunction::SnapArea { extra_size, unit } => {
            assert_eq!(extra_size, 12.0);
            assert_eq!(unit, SizeUnits::Pixels);
        }
        _ => panic!("expected SnapArea variant"),
    }
}

#[test]
fn transform_function_min_circle_setting_converts() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::MinCircle {
            min_diameter: 5.0,
            unit: SizeUnits::NanoMeter,
        }
        .into();
    match result {
        TransformFunction::MinCircle { min_diameter, unit } => {
            assert_eq!(min_diameter, 5.0);
            assert_eq!(unit, SizeUnits::NanoMeter);
        }
        _ => panic!("expected MinCircle variant"),
    }
}

#[test]
fn transform_function_draw_circle_setting_converts() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::DrawCircle {
            diameter: 8.0,
            unit: SizeUnits::Pixels,
        }
        .into();
    match result {
        TransformFunction::DrawCircle { diameter, unit } => {
            assert_eq!(diameter, 8.0);
            assert_eq!(unit, SizeUnits::Pixels);
        }
        _ => panic!("expected DrawCircle variant"),
    }
}

#[test]
fn transform_function_fitting_ellipse_setting_converts() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse { scale: 2.0 }
            .into();
    match result {
        TransformFunction::FittingEllipse { scale } => assert_eq!(scale, 2.0),
        _ => panic!("expected FittingEllipse variant"),
    }
}

#[test]
fn transform_function_expand_setting_converts() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::Expand {
            margin: 3.0,
            unit: SizeUnits::NanoMeter,
        }
        .into();
    match result {
        TransformFunction::Expand { margin, unit } => {
            assert_eq!(margin, 3.0);
            assert_eq!(unit, SizeUnits::NanoMeter);
        }
        _ => panic!("expected Expand variant"),
    }
}

#[test]
fn transform_function_shrink_setting_converts() {
    let result: TransformFunction =
        ClassificationTransformObjectsTransformFunctionSettings::Shrink {
            margin: 4.0,
            unit: SizeUnits::Pixels,
        }
        .into();
    match result {
        TransformFunction::Shrink { margin, unit } => {
            assert_eq!(margin, 4.0);
            assert_eq!(unit, SizeUnits::Pixels);
        }
        _ => panic!("expected Shrink variant"),
    }
}

#[cfg(feature = "ai")]
#[test]
fn unet_output_mode_softmax_classes_setting_converts() {
    let result: UNetOutputMode = AiSegmentationUnetUNetOutputModeSettings::SoftmaxClasses.into();
    assert!(matches!(result, UNetOutputMode::SoftmaxClasses));
}

#[cfg(feature = "ai")]
#[test]
fn unet_output_mode_independent_channels_setting_converts() {
    let result: UNetOutputMode =
        AiSegmentationUnetUNetOutputModeSettings::IndependentChannels.into();
    assert!(matches!(result, UNetOutputMode::IndependentChannels));
}

// ---- Struct conversions --------------------------------------------

#[test]
fn blur_settings_convert_kernel_size() {
    let settings = BlurSettings { kernel_size: 9 };
    let result = Blur::from(settings);
    assert_eq!(result.kernel_size, 9);
}

#[cfg(feature = "ai")]
#[test]
fn cellpose_settings_convert_fields_and_clamp_probability_threshold() {
    let settings = CellposeSettings {
        model_path: PathBuf::from("model.pt"),
        object_class_id: SegmentationClass(7),
        input_channels: 3,
        probability_threshold: 1.5,
        flow_iterations: 100,
        min_object_size: 20,
        ..Default::default()
    };
    let result = Cellpose::from(settings);
    assert_eq!(result.model_path, PathBuf::from("model.pt"));
    assert_eq!(result.object_class_id, SegmentationClass(7));
    assert_eq!(result.input_channels, 3);
    assert_eq!(result.probability_threshold, 1.0);
    assert_eq!(result.flow_iterations, 100);
    assert_eq!(result.min_object_size, 20);
}

#[cfg(feature = "ai")]
#[test]
fn cellpose_settings_clamp_negative_probability_threshold_to_zero() {
    let settings = CellposeSettings {
        probability_threshold: -1.0,
        ..Default::default()
    };
    let result = Cellpose::from(settings);
    assert_eq!(result.probability_threshold, 0.0);
}

#[test]
fn classify_objects_settings_convert_identity_fields() {
    let settings = ClassifyObjectsSettings {
        origin_segmentation: vec![SegmentationClass(1), SegmentationClass(2)],
        input_classes: vec![ObjectClass::Valid(3)],
        output_class: ObjectClass::Valid(4),
        overlapping_with: ObjectClass::Valid(5),
        allow_edge_touching: false,
        ..Default::default()
    };
    let result = ClassifyObjects::from(settings);
    assert_eq!(
        result.origin_segmentation,
        vec![SegmentationClass(1), SegmentationClass(2)]
    );
    assert_eq!(result.input_classes, vec![ObjectClass::Valid(3)]);
    assert_eq!(result.output_class, ObjectClass::Valid(4));
    assert_eq!(result.overlapping_with, ObjectClass::Valid(5));
    assert!(!result.allow_edge_touching);
}

#[test]
fn classify_objects_settings_clamp_area_and_circularity_fields() {
    let settings = ClassifyObjectsSettings {
        min_area: -5.0,
        max_area: 9999999999.0,
        min_circularity: -0.5,
        max_circularity: 5.0,
        ..Default::default()
    };
    let result = ClassifyObjects::from(settings);
    assert_eq!(result.min_area, 0.0);
    assert_eq!(result.max_area, 2147483600.0);
    assert_eq!(result.min_circularity, 0.0);
    assert_eq!(result.max_circularity, 1.0);
}

#[test]
fn colocalization_settings_convert_all_fields() {
    let settings = ColocalizationSettings {
        classes_to_coloc: vec![ObjectClass::Valid(1), ObjectClass::Valid(2)],
        filter_classes: vec![ObjectClass::Valid(3)],
        class_for_overlapping_areas: ObjectClass::Valid(9),
        exclude_classes: vec![ObjectClass::Valid(4)],
        allow_multi_object_coloc: true,
        size_unit: SizeUnits::Pixels,
        min_coloc_area: 12.5,
    };
    let result = Colocalization::from(settings);
    assert_eq!(
        result.classes_to_coloc,
        vec![ObjectClass::Valid(1), ObjectClass::Valid(2)]
    );
    assert_eq!(result.filter_classes, vec![ObjectClass::Valid(3)]);
    assert_eq!(result.class_for_overlapping_areas, ObjectClass::Valid(9));
    assert_eq!(result.exclude_classes, vec![ObjectClass::Valid(4)]);
    assert!(result.allow_multi_object_coloc);
    assert_eq!(result.size_unit, SizeUnits::Pixels);
    assert_eq!(result.min_coloc_area, 12.5);
}

#[test]
fn color_filter_command_settings_convert_nested_hsv_range() {
    let settings = ColorFilterCommandSettings {
        range: HsvRangeSettings {
            min_h: 1.0,
            max_h: 2.0,
            min_s: 3.0,
            max_s: 4.0,
            min_v: 5.0,
            max_v: 6.0,
        },
    };
    let result = ColorFilterCommand::from(settings);
    assert_eq!(result.range.min_h, 1.0);
    assert_eq!(result.range.max_h, 2.0);
    assert_eq!(result.range.min_s, 3.0);
    assert_eq!(result.range.max_s, 4.0);
    assert_eq!(result.range.min_v, 5.0);
    assert_eq!(result.range.max_v, 6.0);
}

#[test]
fn connected_components_settings_convert_min_size_px() {
    let settings = ConnectedComponentsSettings { min_size: 42 };
    let result = ConnectedComponents::from(settings);
    assert_eq!(result.min_size, 42);
}

#[test]
fn distance_transform_settings_convert_fields() {
    let settings = DistanceTransformSettings {
        threshold: 0.75,
        edges_are_background: true,
    };
    let result = DistanceTransform::from(settings);
    assert_eq!(result.threshold, 0.75);
    assert!(result.edges_are_background);
}

#[test]
fn edge_detection_canny_settings_convert_fields() {
    let settings = EdgeDetectionCannySettings {
        kernel_size: 5,
        threshold_min: 0.1,
        threshold_max: 0.4,
    };
    let result = EdgeDetectionCanny::from(settings);
    assert_eq!(result.kernel_size, 5);
    assert_eq!(result.threshold_min, 0.1);
    assert_eq!(result.threshold_max, 0.4);
}

#[test]
fn edge_detection_sobel_settings_convert_kernel_size() {
    let settings = EdgeDetectionSobelSettings { kernel_size: 7 };
    let result = EdgeDetectionSobel::from(settings);
    assert_eq!(result.kernel_size, 7);
}

#[test]
fn enhance_contrast_settings_convert_fields() {
    let settings = EnhanceContrastSettings {
        saturated_pixels: 0.02,
        normalize: true,
        equalize_histogram: true,
    };
    let result = EnhanceContrast::from(settings);
    assert_eq!(result.saturated_pixels, 0.02);
    assert!(result.normalize);
    assert!(result.equalize_histogram);
}

#[test]
fn extract_objects_settings_convert_max_objects_before_fail() {
    let settings = ExtractObjectsSettings {
        max_objects_before_fail: 500,
    };
    let result = ExtractObjects::from(settings);
    assert_eq!(result.max_objects_before_fail, 500);
}

#[test]
fn gaussian_blur_settings_convert_kernel_size_and_clamp_sigma() {
    let settings = GaussianBlurSettings {
        kernel_size: 5,
        sigma: 100.0,
    };
    let result = GaussianBlur::from(settings);
    assert_eq!(result.kernel_size, 5);
    assert_eq!(result.sigma, 5.0);
}

#[test]
fn gaussian_blur_settings_clamp_low_sigma_to_minimum() {
    let settings = GaussianBlurSettings {
        kernel_size: 5,
        sigma: 0.0,
    };
    let result = GaussianBlur::from(settings);
    assert_eq!(result.sigma, 0.1);
}

#[test]
fn hessian_settings_convert_mode() {
    let settings = HessianSettings {
        mode: FiltersHessianHessianModeSettings::EigenvaluesY,
    };
    let result = Hessian::from(settings);
    assert_eq!(result.mode, HessianMode::EigenvaluesY);
}

#[test]
fn hsv_range_settings_convert_all_fields() {
    let settings = HsvRangeSettings {
        min_h: 10.0,
        max_h: 20.0,
        min_s: 0.1,
        max_s: 0.9,
        min_v: 0.2,
        max_v: 0.8,
    };
    let result = HsvRange::from(settings);
    assert_eq!(result.min_h, 10.0);
    assert_eq!(result.max_h, 20.0);
    assert_eq!(result.min_s, 0.1);
    assert_eq!(result.max_s, 0.9);
    assert_eq!(result.min_v, 0.2);
    assert_eq!(result.max_v, 0.8);
}

#[test]
fn image_cache_settings_convert_mode_and_address() {
    let settings = ImageCacheSettings {
        mode: MathImageCacheImageCacheModeSettings::Load,
        address: ImageAddress::Scratchpad,
    };
    let result = ImageCache::from(settings);
    assert!(matches!(result.mode, ImageCacheMode::Load));
    assert_eq!(result.address, ImageAddress::Scratchpad);
}

#[test]
fn image_math_settings_convert_all_fields() {
    let settings = ImageMathSettings {
        operand: MathImageMathOperandSettings::Multiply,
        second_image_address: ImageAddress::Channel(2),
        swap_operands: true,
    };
    let result = ImageMath::from(settings);
    assert_eq!(result.operand, Operand::Multiply);
    assert_eq!(result.second_image_address, ImageAddress::Channel(2));
    assert!(result.swap_operands);
}

#[test]
fn intensity_transformation_settings_convert_all_fields() {
    let settings = IntensityTransformationSettings {
        mode: FiltersIntensityTransformIntensityTransformModeSettings::Manual,
        contrast: 1.5,
        brightness: -0.2,
    };
    let result = IntensityTransformation::from(settings);
    assert_eq!(result.mode, IntensityTransformMode::Manual);
    assert_eq!(result.contrast, 1.5);
    assert_eq!(result.brightness, -0.2);
}

#[test]
fn laplacian_settings_convert_kernel_size() {
    let settings = LaplacianSettings { kernel_size: 3 };
    let result = Laplacian::from(settings);
    assert_eq!(result.kernel_size, 3);
}

#[test]
fn median_subtract_settings_convert_radius() {
    let settings = MedianSubtractSettings { radius: 15.0 };
    let result = MedianSubtract::from(settings);
    assert_eq!(result.radius, 15.0);
}

#[test]
fn morphological_command_settings_convert_all_fields() {
    let settings = MorphologicalCommandSettings {
        op: MorphologyMorphologicalTransformationMorphOpsSettings::Close,
        kernel_size: 5,
        kernel_shape: MorphologyMorphologicalTransformationKernelShapesSettings::Cross,
        use_grayscale: true,
    };
    let result = MorphologicalCommand::from(settings);
    assert!(matches!(result.op, MorphOps::Close));
    assert_eq!(result.kernel_size, 5);
    assert!(matches!(result.kernel_shape, KernelShapes::Cross));
    assert!(result.use_grayscale);
}

#[test]
fn rank_filter_settings_convert_all_fields() {
    let settings = RankFilterSettings {
        radius: 2.5,
        filter_type: FiltersRankFilterRankFilterTypeSettings::Outliers(1.2),
    };
    let result = RankFilter::from(settings);
    assert_eq!(result.radius, 2.5);
    assert_eq!(result.filter_type, RankFilterType::Outliers(1.2));
}

#[test]
fn object_math_settings_convert_all_fields() {
    let settings = ObjectMathSettings {
        operation: ClassificationObjectMathObjectSetOperationSettings::Subtract,
        input_class: ObjectClass::Valid(1),
        other_class: ObjectClass::Valid(2),
        other_filter_classes: vec![ObjectClass::Valid(3)],
        size_unit: SizeUnits::Pixels,
        min_overlap_area: 4.0,
        output_class: ObjectClass::Valid(5),
        keep_unmatched: false,
    };
    let result = ObjectMath::from(settings);
    assert!(matches!(result.operation, ObjectSetOperation::Subtract));
    assert_eq!(result.input_class, ObjectClass::Valid(1));
    assert_eq!(result.other_class, ObjectClass::Valid(2));
    assert_eq!(result.other_filter_classes, vec![ObjectClass::Valid(3)]);
    assert_eq!(result.size_unit, SizeUnits::Pixels);
    assert_eq!(result.min_overlap_area, 4.0);
    assert_eq!(result.output_class, ObjectClass::Valid(5));
    assert!(!result.keep_unmatched);
}

#[test]
fn rolling_ball_settings_convert_and_clamp_radius_to_max() {
    let settings = RollingBallSettings {
        radius: 999.0,
        ball_type: FiltersRollingBallBallTypeSettings::Paraboloid,
        pre_smooth: true,
    };
    let result = RollingBall::from(settings);
    assert_eq!(result.radius, 64.0);
    assert!(result.ball_type == BallType::Paraboloid);
    assert!(result.pre_smooth);
}

#[test]
fn rolling_ball_settings_clamp_radius_to_min() {
    let settings = RollingBallSettings {
        radius: 0.0,
        ..Default::default()
    };
    let result = RollingBall::from(settings);
    assert_eq!(result.radius, 1.0);
}

#[test]
fn save_image_settings_convert_all_fields() {
    let settings = SaveImageSettings {
        name: "output".to_string(),
        source: MathSaveImageImageSourceSettings::InstanceMap,
    };
    let result = SaveImage::from(settings);
    assert_eq!(result.name, "output");
    assert!(result.source == ImageSource::InstanceMap);
}

#[cfg(feature = "ai")]
#[test]
fn stardist_settings_convert_and_clamp_thresholds() {
    let settings = StardistSettings {
        model_path: PathBuf::from("stardist.pt"),
        object_class_id: SegmentationClass(3),
        probability_threshold: -0.5,
        nms_threshold: 2.0,
    };
    let result = Stardist::from(settings);
    assert_eq!(result.model_path, PathBuf::from("stardist.pt"));
    assert_eq!(result.object_class_id, SegmentationClass(3));
    assert_eq!(result.probability_threshold, 0.0);
    assert_eq!(result.nms_threshold, 1.0);
}

#[test]
fn structure_tensor_settings_convert_all_fields() {
    let settings = StructureTensorSettings {
        mode: FiltersStructureTensorTensorModeSettings::Coherence,
        kernel_size: 3,
        sigma: 1.5,
    };
    let result = StructureTensor::from(settings);
    assert!(matches!(result.mode, TensorMode::Coherence));
    assert_eq!(result.kernel_size, 3);
    assert_eq!(result.sigma, 1.5);
}

#[test]
fn threshold_settings_convert_vector_of_entries() {
    let settings = ThresholdSettings {
        thresholds: vec![
            ThresholdEntrySettings {
                method: SegmentationThresholdThresholdMethodSettings::Otsu,
                ..Default::default()
            },
            ThresholdEntrySettings {
                method: SegmentationThresholdThresholdMethodSettings::Manual,
                ..Default::default()
            },
        ],
    };
    let result = Threshold::from(settings);
    assert_eq!(result.thresholds.len(), 2);
    assert!(matches!(result.thresholds[0].method, ThresholdMethod::Otsu));
    assert!(matches!(
        result.thresholds[1].method,
        ThresholdMethod::Manual
    ));
}

#[test]
fn threshold_entry_settings_convert_and_clamp_thresholds() {
    let settings = ThresholdEntrySettings {
        method: SegmentationThresholdThresholdMethodSettings::Manual,
        min_threshold: -10.0,
        max_threshold: 999999.0,
        unit: PixelUnits::Percent,
        object_class_id: SegmentationClass(6),
    };
    let result = ThresholdEntry::from(settings);
    assert!(matches!(result.method, ThresholdMethod::Manual));
    assert_eq!(result.min_threshold, 0.0);
    assert_eq!(result.max_threshold, 65535.0);
    assert_eq!(result.unit, PixelUnits::Percent);
    assert_eq!(result.object_class_id, SegmentationClass(6));
}

#[test]
fn transform_objects_settings_convert_all_fields() {
    let settings = TransformObjectsSettings {
        function: ClassificationTransformObjectsTransformFunctionSettings::FittingEllipse {
            scale: 1.2,
        },
        input_class: ObjectClass::Valid(1),
        output_class: ObjectClass::Valid(2),
    };
    let result = TransformObjects::from(settings);
    match result.function {
        TransformFunction::FittingEllipse { scale } => assert_eq!(scale, 1.2),
        _ => panic!("expected FittingEllipse variant"),
    }
    assert_eq!(result.input_class, ObjectClass::Valid(1));
    assert_eq!(result.output_class, ObjectClass::Valid(2));
}

#[cfg(feature = "ai")]
#[test]
fn unet_settings_convert_and_clamp_probability_threshold() {
    let settings = UNetSettings {
        model_path: PathBuf::from("unet.pt"),
        object_class_id: SegmentationClass(2),
        probability_threshold: 3.0,
        output_mode: AiSegmentationUnetUNetOutputModeSettings::IndependentChannels,
        foreground_channel: 1,
        boundary_channel: 0,
        boundary_threshold: -1.0,
    };
    let result = UNet::from(settings);
    assert_eq!(result.model_path, PathBuf::from("unet.pt"));
    assert_eq!(result.object_class_id, SegmentationClass(2));
    assert_eq!(result.probability_threshold, 1.0);
    assert!(matches!(
        result.output_mode,
        UNetOutputMode::IndependentChannels
    ));
    assert_eq!(result.foreground_channel, 1);
    assert_eq!(result.boundary_channel, 0);
    assert_eq!(result.boundary_threshold, 0.0);
}

#[test]
fn voronoi_settings_convert_all_fields() {
    let settings = VoronoiSettings {
        centers: ObjectClass::Valid(1),
        center_filter_classes: vec![ObjectClass::Valid(2)],
        mask: ObjectClass::Valid(3),
        mask_filter_classes: vec![ObjectClass::Valid(4)],
        output_class: ObjectClass::Valid(5),
        unit: SizeUnits::Pixels,
        max_radius: 50.0,
        exclude_areas_at_the_edges: true,
        exclude_areas_with_no_center: true,
    };
    let result = Voronoi::from(settings);
    assert_eq!(result.centers, ObjectClass::Valid(1));
    assert_eq!(result.center_filter_classes, vec![ObjectClass::Valid(2)]);
    assert_eq!(result.mask, ObjectClass::Valid(3));
    assert_eq!(result.mask_filter_classes, vec![ObjectClass::Valid(4)]);
    assert_eq!(result.output_class, ObjectClass::Valid(5));
    assert_eq!(result.unit, SizeUnits::Pixels);
    assert_eq!(result.max_radius, 50.0);
    assert!(result.exclude_areas_at_the_edges);
    assert!(result.exclude_areas_with_no_center);
}

#[test]
fn watershed_settings_convert_and_clamp_tolerance_and_sigma() {
    let settings = WatershedSettings {
        maximum_finder_tolerance: 0.0,
        smoothing_sigma: 999.0,
        min_object_size: 30,
    };
    let result = Watershed::from(settings);
    assert_eq!(result.maximum_finder_tolerance, 0.1);
    assert_eq!(result.smoothing_sigma, 10.0);
    assert_eq!(result.min_object_size, 30);
}

#[test]
fn watershed_settings_clamp_tolerance_to_max() {
    let settings = WatershedSettings {
        maximum_finder_tolerance: 999.0,
        ..Default::default()
    };
    let result = Watershed::from(settings);
    assert_eq!(result.maximum_finder_tolerance, 20.0);
}

#[test]
fn weighted_deviation_settings_convert_all_fields() {
    let settings = WeightedDeviationSettings {
        kernel_size: 9,
        sigma: 3.0,
    };
    let result = WeightedDeviation::from(settings);
    assert_eq!(result.kernel_size, 9);
    assert_eq!(result.sigma, 3.0);
}

// ---- into_algorithm dispatch --------------------------------------------

#[test]
fn into_algorithm_blur_dispatches_to_blur_algorithm() {
    let result = into_algorithm(PipelineCommand::Blur(BlurSettings { kernel_size: 3 }))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Blur");
}

#[test]
fn into_algorithm_watershed_dispatches_to_watershed_algorithm() {
    let result = into_algorithm(PipelineCommand::Watershed(WatershedSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Watershed");
}

#[test]
fn into_algorithm_classify_objects_dispatches_to_classify_objects_algorithm() {
    let result = into_algorithm(PipelineCommand::ClassifyObjects(
        ClassifyObjectsSettings::default(),
    ))
    .expect("conversion should not fail");
    assert_eq!(result.name(), "ClassifyObjects");
}

#[test]
fn into_algorithm_threshold_dispatches_to_threshold_algorithm() {
    let result = into_algorithm(PipelineCommand::Threshold(ThresholdSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Threshold");
}

#[cfg(feature = "ai")]
#[test]
fn into_algorithm_cellpose_dispatches_to_cellpose_algorithm_when_ai_enabled() {
    let result = into_algorithm(PipelineCommand::Cellpose(CellposeSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Cellpose");
}

#[cfg(feature = "ai")]
#[test]
fn into_algorithm_stardist_dispatches_to_stardist_algorithm_when_ai_enabled() {
    let result = into_algorithm(PipelineCommand::Stardist(StardistSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Stardist");
}

#[cfg(feature = "ai")]
#[test]
fn into_algorithm_unet_dispatches_to_unet_algorithm_when_ai_enabled() {
    let result = into_algorithm(PipelineCommand::UNet(UNetSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "UNet");
}

#[test]
fn into_algorithm_voronoi_dispatches_to_voronoi_algorithm() {
    let result = into_algorithm(PipelineCommand::Voronoi(VoronoiSettings::default()))
        .expect("conversion should not fail");
    assert_eq!(result.name(), "Voronoi");
}
