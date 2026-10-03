//! Which object classes a pipeline works with - used by the GUI's pipeline
//! focus mode to show only the classes of the selected pipeline.
//!
//! Hand-written (not generated): every command decides explicitly. The match
//! in [`PipelineCommand::object_classes`] has no wildcard arm, so a new command
//! fails to compile until it is listed here.

use crate::modules::pipeline_command::PipelineCommand;
use crate::modules::pipeline_settings::PipelineSettings;
use crate::types::classes::{ObjectClass, SegmentationClass};
use std::collections::BTreeSet;

impl PipelineCommand {
    /// Every object class this command creates, reads or filters on.
    /// Segmentation classes count as the object class Extract Objects turns
    /// them into. `Unset`, Background and the manual-annotation marker are
    /// left out - they never name a class of the project.
    pub fn object_classes(&self) -> Vec<ObjectClass> {
        let mut classes: Vec<ObjectClass> = Vec::new();
        let seg = |c: &SegmentationClass, out: &mut Vec<ObjectClass>| {
            out.push(ObjectClass::from_segmentation_class(*c))
        };
        match self {
            // Segmentation: the class written into the segmentation map.
            PipelineCommand::Cellpose(s) => seg(&s.object_class_id, &mut classes),
            PipelineCommand::Stardist(s) => seg(&s.object_class_id, &mut classes),
            PipelineCommand::UNet(s) => seg(&s.object_class_id, &mut classes),
            PipelineCommand::Threshold(s) => {
                for entry in &s.thresholds {
                    seg(&entry.object_class_id, &mut classes);
                }
            }
            PipelineCommand::PixelClassifier(s) => {
                for mapping in &s.segmentation_mapping {
                    seg(&mapping.object_class_id, &mut classes);
                }
            }
            // An empty mapping writes model class i as i + 1 - how many
            // classes that is, only the model file knows.
            PipelineCommand::Yolov5(s) => {
                for mapping in &s.class_mapping {
                    seg(&mapping.segmentation_class, &mut classes);
                }
            }

            // Object commands: inputs, filters and outputs.
            PipelineCommand::AiObjectClassifier(s) => {
                for c in &s.origin_segmentation {
                    seg(c, &mut classes);
                }
                classes.extend(&s.input_classes);
                classes.extend(s.segmentation_mapping.iter().map(|m| m.output_class));
            }
            PipelineCommand::ClassifyObjects(s) => {
                for c in &s.origin_segmentation {
                    seg(c, &mut classes);
                }
                classes.extend(&s.input_classes);
                classes.extend([s.output_class, s.overlapping_with]);
            }
            PipelineCommand::Colocalization(s) => {
                classes.extend(&s.classes_to_coloc);
                classes.extend(&s.filter_classes);
                classes.extend(&s.exclude_classes);
                classes.push(s.class_for_overlapping_areas);
            }
            PipelineCommand::LoadAnnotatedObjects(s) => {
                classes.extend(&s.input_classes);
                classes.push(s.output_class);
            }
            PipelineCommand::ObjectMath(s) => {
                classes.extend([s.input_class, s.other_class, s.output_class]);
                classes.extend(&s.other_filter_classes);
            }
            PipelineCommand::TransformObjects(s) => {
                classes.extend([s.input_class, s.output_class]);
            }
            PipelineCommand::Voronoi(s) => {
                classes.extend([s.centers, s.mask, s.output_class]);
                classes.extend(&s.center_filter_classes);
                classes.extend(&s.mask_filter_classes);
            }

            // Image filters and steps that keep the classes they're given.
            PipelineCommand::Blur(_)
            | PipelineCommand::ColorFilterCommand(_)
            | PipelineCommand::ConnectedComponents(_)
            | PipelineCommand::DistanceTransform(_)
            | PipelineCommand::EdgeDetectionCanny(_)
            | PipelineCommand::EdgeDetectionSobel(_)
            | PipelineCommand::EnhanceContrast(_)
            | PipelineCommand::ExtractObjects(_)
            | PipelineCommand::FillHoles(_)
            | PipelineCommand::FillObjectHoles(_)
            | PipelineCommand::GaussianBlur(_)
            | PipelineCommand::Hessian(_)
            | PipelineCommand::IlluminationCorrection(_)
            | PipelineCommand::ImageCache(_)
            | PipelineCommand::ImageMath(_)
            | PipelineCommand::IntensityTransformation(_)
            | PipelineCommand::Laplacian(_)
            | PipelineCommand::MedianSubtract(_)
            | PipelineCommand::MorphologicalCommand(_)
            | PipelineCommand::RankFilter(_)
            | PipelineCommand::RollingBall(_)
            | PipelineCommand::SaveImage(_)
            | PipelineCommand::StructureTensor(_)
            | PipelineCommand::Watershed(_)
            | PipelineCommand::WeightedDeviation(_) => {}
        }
        classes.retain(|c| {
            !matches!(c, ObjectClass::Unset)
                && *c != ObjectClass::BACKGROUND
                && c.to_u32() != Some(SegmentationClass::MANUAL_ANNOTATED.0)
        });
        classes
    }
}

impl PipelineSettings {
    /// The object classes of every enabled step (see
    /// [`PipelineCommand::object_classes`]), each once.
    pub fn object_classes(&self) -> BTreeSet<ObjectClass> {
        self.steps
            .iter()
            .filter(|step| step.enabled)
            .flat_map(|step| step.command.object_classes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::pipeline_command_settings::*;
    use crate::modules::pipeline_settings::PipelineStepSettings;
    use crate::types::ids::{ImageAddress, PipelineId};

    fn step(enabled: bool, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings { enabled, command }
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
}
