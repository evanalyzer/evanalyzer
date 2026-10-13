//! Tests for the generated `PipelineCommand::object_classes()` (see
//! `build/pipeline_commands_generator.rs`) and `PipelineSettings::object_classes()`.
//! The generated file can't carry inline tests, so they live here.

use evanalyzer_cfg::core_types::{ImageAddress, ObjectClass, PipelineId, SegmentationClass};
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::*;
use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};

fn step(enabled: bool, command: PipelineCommand) -> PipelineStepSettings {
    PipelineStepSettings {
        id: evanalyzer_cfg::core_types::StepId::generate(),
        enabled,
        command,
    }
}

fn pipeline(steps: Vec<PipelineStepSettings>) -> PipelineSettings {
    PipelineSettings {
        id: PipelineId(1),
        name: "p".into(),
        description: None,
        image_source: ImageAddress::Channel(0),
        enabled: true,
        steps,
    }
}

#[test]
fn segmentation_classes_count_as_the_object_class_they_become() {
    let threshold = PipelineCommand::Threshold(ThresholdSettings {
        thresholds: vec![
            ThresholdEntrySettings {
                object_class_id: SegmentationClass(2),
                ..Default::default()
            },
            ThresholdEntrySettings {
                object_class_id: SegmentationClass(5),
                ..Default::default()
            },
        ],
    });
    assert_eq!(
        threshold.object_classes(),
        vec![ObjectClass::Valid(2), ObjectClass::Valid(5)]
    );
}

#[test]
fn object_commands_report_inputs_filters_and_outputs() {
    let coloc = PipelineCommand::Colocalization(ColocalizationSettings {
        classes_to_coloc: vec![ObjectClass::Valid(1), ObjectClass::Valid(2)],
        filter_classes: vec![ObjectClass::Valid(3)],
        exclude_classes: vec![ObjectClass::Valid(4)],
        class_for_overlapping_areas: ObjectClass::Valid(9),
        ..Default::default()
    });
    let mut classes = coloc.object_classes();
    classes.sort();
    assert_eq!(classes, [1, 2, 3, 4, 9].map(ObjectClass::Valid));
}

#[test]
fn save_image_reports_its_display_classes() {
    let save = PipelineCommand::SaveImage(SaveImageSettings {
        display_classes: vec![ObjectClass::Valid(2), ObjectClass::Valid(5)],
        ..Default::default()
    });
    assert_eq!(save.object_classes(), [2, 5].map(ObjectClass::Valid));
}

#[test]
fn classes_inside_an_enum_variant_are_reported() {
    let coloc = PipelineCommand::Colocalization(ColocalizationSettings {
        multiplicity: ObjectColocObjectsColocMultiplicitySettings::MultiFor(vec![
            ObjectClass::Valid(11),
        ]),
        ..Default::default()
    });
    assert_eq!(coloc.object_classes(), [ObjectClass::Valid(11)]);
}

#[test]
fn unset_background_and_manual_annotation_are_left_out() {
    let classify = PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
        origin_segmentation: vec![SegmentationClass(0), SegmentationClass::MANUAL_ANNOTATED],
        input_classes: vec![ObjectClass::Valid(7)],
        output_class: ObjectClass::Unset,
        overlapping_with: ObjectClass::Unset,
        ..Default::default()
    });
    assert_eq!(classify.object_classes(), vec![ObjectClass::Valid(7)]);
}

fn sorted(command: PipelineCommand) -> Vec<ObjectClass> {
    let mut classes = command.object_classes();
    classes.sort();
    classes
}

#[test]
fn ai_segmentation_reports_the_class_it_writes() {
    let cellpose = PipelineCommand::Cellpose(CellposeSettings {
        object_class_id: SegmentationClass(3),
        ..Default::default()
    });
    let stardist = PipelineCommand::Stardist(StardistSettings {
        object_class_id: SegmentationClass(4),
        ..Default::default()
    });
    let unet = PipelineCommand::UNet(UNetSettings {
        object_class_id: SegmentationClass(5),
        ..Default::default()
    });
    assert_eq!(sorted(cellpose), [ObjectClass::Valid(3)]);
    assert_eq!(sorted(stardist), [ObjectClass::Valid(4)]);
    assert_eq!(sorted(unet), [ObjectClass::Valid(5)]);
}

#[test]
fn classifier_and_yolo_mappings_report_their_target_classes() {
    let pixel = PipelineCommand::PixelClassifier(PixelClassifierSettings {
        segmentation_mapping: vec![
            // The model's own classes (1, 2) are `ModelSegmentationClass`.
            SegmentationMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class_id: SegmentationClass(6),
            },
            SegmentationMappingSettings {
                segmentation_class: SegmentationClass(2),
                object_class_id: SegmentationClass(7),
            },
        ],
        ..Default::default()
    });
    assert_eq!(sorted(pixel), [6, 7].map(ObjectClass::Valid));

    let yolo = PipelineCommand::Yolov5(Yolov5Settings {
        class_mapping: vec![YoloClassMappingSettings {
            model_class: 0,
            segmentation_class: SegmentationClass(8),
        }],
        ..Default::default()
    });
    assert_eq!(sorted(yolo), [ObjectClass::Valid(8)]);

    let ai = PipelineCommand::AiObjectClassifier(AiObjectClassifierSettings {
        origin_segmentation: vec![SegmentationClass(1)],
        input_classes: vec![ObjectClass::Valid(2)],
        // The model's own class (3) is a `ModelClass` - not a project class.
        segmentation_mapping: vec![ClassificationMappingSettings {
            object_class: ObjectClass::Valid(3),
            output_class: ObjectClass::Valid(9),
        }],
        ..Default::default()
    });
    assert_eq!(sorted(ai), [1, 2, 9].map(ObjectClass::Valid));
}

#[test]
fn object_commands_report_all_their_classes() {
    let load = PipelineCommand::LoadAnnotatedObjects(LoadAnnotatedObjectsSettings {
        input_classes: vec![ObjectClass::Valid(1)],
        output_class: ObjectClass::Valid(2),
        ..Default::default()
    });
    assert_eq!(sorted(load), [1, 2].map(ObjectClass::Valid));

    let math = PipelineCommand::ObjectMath(ObjectMathSettings {
        input_class: ObjectClass::Valid(1),
        other_class: ObjectClass::Valid(2),
        output_class: ObjectClass::Valid(3),
        other_filter_classes: vec![ObjectClass::Valid(4)],
        ..Default::default()
    });
    assert_eq!(sorted(math), [1, 2, 3, 4].map(ObjectClass::Valid));

    let voronoi = PipelineCommand::Voronoi(VoronoiSettings {
        centers: ObjectClass::Valid(1),
        mask: ObjectClass::Valid(2),
        output_class: ObjectClass::Valid(3),
        center_filter_classes: vec![ObjectClass::Valid(4)],
        mask_filter_classes: vec![ObjectClass::Valid(5)],
        ..Default::default()
    });
    assert_eq!(sorted(voronoi), [1, 2, 3, 4, 5].map(ObjectClass::Valid));
}

#[test]
fn image_filters_have_no_classes() {
    let blur = PipelineCommand::GaussianBlur(GaussianBlurSettings::default());
    assert!(blur.object_classes().is_empty());
}

#[test]
fn a_pipeline_collects_the_classes_of_its_enabled_steps_once() {
    let make = |id| {
        PipelineCommand::TransformObjects(TransformObjectsSettings {
            input_class: ObjectClass::Valid(id),
            output_class: ObjectClass::Valid(10),
            ..Default::default()
        })
    };
    let p = pipeline(vec![
        step(true, make(1)),
        step(true, make(2)),
        step(false, make(3)),
    ]);
    assert_eq!(
        p.object_classes().into_iter().collect::<Vec<_>>(),
        [1, 2, 10].map(ObjectClass::Valid)
    );
}
