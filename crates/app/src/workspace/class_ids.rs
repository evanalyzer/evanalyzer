//! A project class's id: the free ones to choose from, and changing it -
//! together with every reference to the class.

use crate::workspace::module_editing::{setting_at, uses_segmentation_class};
use evanalyzer_cfg::{
    core_types::{InternalErrors, MAX_CLASS_ID, ModuleParamKey, ObjectClass, SegmentationClass},
    settings::{
        classification_settings::ClassificationSettings,
        modules::{ModuleDefinition, ParamValue},
        parameter_def::ParamType,
        pipeline_command::PipelineCommand,
        pipeline_command_settings::ExtractClassMappingSettings,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
    },
};
use std::collections::{BTreeMap, BTreeSet};

/// The id "Auto" gives a new class: one above the highest, so a deleted
/// class's id isn't reused (old pipelines and results may still refer to
/// it) - unless it was the highest.
pub fn next_class_id(classification: &ClassificationSettings) -> u32 {
    classification
        .classes()
        .iter()
        .map(|c| c.id)
        .max()
        .map_or(1, |max| max + 1)
}

/// The ids no class has, from 1 to [`MAX_CLASS_ID`] - the ones the
/// editor's class choices offer - for choosing one by hand. 0 is the
/// background's.
pub fn free_class_ids(classification: &ClassificationSettings) -> Vec<u32> {
    let used: BTreeSet<u32> = classification.classes().iter().map(|c| c.id).collect();
    (1..=MAX_CLASS_ID).filter(|id| !used.contains(id)).collect()
}

/// What [`change_class_id`] changed besides the class itself.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ClassIdChange {
    /// References to the class that now name the new id: in commands, module
    /// parameter values, annotated objects, the tile merge settings.
    pub references: usize,
    /// Extract Objects steps that got a mapping for the segmentation class
    /// with the old number, so its objects still get the class.
    pub mappings_added: usize,
}

/// Gives project class `from` the id `to`, and every reference to it too:
/// commands of pipelines and modules, module parameters setting a class, the
/// classes of annotated objects, the tile merge settings. A segmentation
/// class with the old number stays as it is: Extract Objects steps without
/// a mapping for it (which gave its objects the class of the same number)
/// get one to the new id. Results of earlier runs keep the old id.
pub fn change_class_id(
    project: &mut ProjectSettings,
    from: u32,
    to: u32,
) -> Result<ClassIdChange, InternalErrors> {
    if from == to {
        return Ok(ClassIdChange::default());
    }
    if from == 0 || to == 0 {
        return Err(InternalErrors::InvalidArgument(
            "the background class keeps the id 0".into(),
        ));
    }
    let ids: BTreeSet<u32> = project
        .classification
        .classes()
        .iter()
        .map(|c| c.id)
        .collect();
    if !ids.contains(&from) {
        return Err(InternalErrors::InvalidArgument(format!(
            "there is no class {from}"
        )));
    }
    if ids.contains(&to) {
        return Err(InternalErrors::InvalidArgument(format!(
            "class {to} exists already"
        )));
    }

    let mut change = ClassIdChange::default();
    let segmented = all_commands(project).any(|c| uses_segmentation_class(c, from));
    let class_parameters = class_parameters(&project.modules);

    // Commands: every `{"GLOBAL": from}`.
    let pipelines = rewrite_classes(&project.pipelines, from, to, &mut change.references)?;
    let modules = rewrite_classes(&project.modules, from, to, &mut change.references)?;
    project.pipelines = pipelines;
    project.modules = modules;

    if segmented {
        let steps = project
            .pipelines
            .iter_mut()
            .flat_map(|p| p.steps.iter_mut())
            .chain(project.modules.iter_mut().flat_map(|g| g.steps.iter_mut()));
        for step in steps {
            if let Some(PipelineCommand::ExtractObjects(extract)) = step.command_mut()
                && !extract
                    .class_mapping
                    .iter()
                    .any(|m| m.segmentation_class.0 == from)
            {
                extract.class_mapping.push(ExtractClassMappingSettings {
                    segmentation_class: SegmentationClass(from),
                    object_class: ObjectClass::Global(to),
                });
                change.mappings_added += 1;
            }
        }
    }

    // Module parameters setting a class: their defaults and values.
    for definition in &mut project.modules {
        if let Some(ids) = class_parameters.get(&definition.key) {
            for parameter in &mut definition.parameters {
                if ids.contains(&parameter.key) {
                    change.references += rewrite_value(&mut parameter.default, from, to);
                }
            }
        }
    }
    let steps = project
        .pipelines
        .iter_mut()
        .flat_map(|p| p.steps.iter_mut())
        .chain(project.modules.iter_mut().flat_map(|g| g.steps.iter_mut()));
    for step in steps {
        if let StepContent::Module(instance) = &mut step.content
            && let Some(ids) = class_parameters.get(&instance.key)
        {
            for (id, value) in &mut instance.values {
                if ids.contains(id) {
                    change.references += rewrite_value(value, from, to);
                }
            }
        }
    }

    for class in &mut project.tile_merge.classes_to_not_merge {
        if *class == ObjectClass::Global(from) {
            *class = ObjectClass::Global(to);
            change.references += 1;
        }
    }

    // Annotated objects.
    for image in project.images.list.values_mut() {
        for series in image.series.values_mut() {
            for object in &mut series.objects {
                if object.object_class.remove(&ObjectClass::Global(from)) {
                    object.object_class.insert(ObjectClass::Global(to));
                    change.references += 1;
                }
                if object
                    .colocalized_with
                    .contains_key(&ObjectClass::Global(from))
                {
                    object.colocalized_with = std::mem::take(&mut object.colocalized_with)
                        .into_iter()
                        .map(|(class, ids)| {
                            if class == ObjectClass::Global(from) {
                                (ObjectClass::Global(to), ids)
                            } else {
                                (class, ids)
                            }
                        })
                        .collect();
                    change.references += 1;
                }
                if object.parents.contains_key(&ObjectClass::Global(from)) {
                    object.parents = std::mem::take(&mut object.parents)
                        .into_iter()
                        .map(|(class, id)| {
                            if class == ObjectClass::Global(from) {
                                (ObjectClass::Global(to), id)
                            } else {
                                (class, id)
                            }
                        })
                        .collect();
                    change.references += 1;
                }
            }
        }
    }

    if let Some(class) = project
        .classification
        .classes_mut()
        .iter_mut()
        .find(|c| c.id == from)
    {
        class.id = to;
    }
    Ok(change)
}

/// Every command of the project's pipelines and modules.
fn all_commands(project: &ProjectSettings) -> impl Iterator<Item = &PipelineCommand> {
    project
        .pipelines
        .iter()
        .flat_map(|p| p.steps.iter())
        .chain(project.modules.iter().flat_map(|g| g.steps.iter()))
        .filter_map(PipelineStepSettings::command)
}

/// Per module key, its parameters that set a class (their card shows a class
/// choice): those whose first setting is a class or a class list.
fn class_parameters(
    definitions: &[ModuleDefinition],
) -> BTreeMap<String, BTreeSet<ModuleParamKey>> {
    definitions
        .iter()
        .map(|definition| {
            let ids = definition
                .parameters
                .iter()
                .filter(|parameter| {
                    parameter.bindings.first().is_some_and(|binding| {
                        definition
                            .steps
                            .iter()
                            .find(|s| s.id == binding.step)
                            .and_then(PipelineStepSettings::command)
                            .and_then(|command| setting_at(command, &binding.param))
                            .is_some_and(|(setting, _)| {
                                matches!(
                                    setting.param_type,
                                    ParamType::ObjClass | ParamType::MultiObjClass
                                )
                            })
                    })
                })
                .map(|parameter| parameter.key.clone())
                .collect();
            (definition.key.clone(), ids)
        })
        .collect()
}

/// `items` with every project class `from` (`{"GLOBAL": from}`) as `to`;
/// counts the changes into `count`.
fn rewrite_classes<T: serde::Serialize + serde::de::DeserializeOwned>(
    items: &T,
    from: u32,
    to: u32,
    count: &mut usize,
) -> Result<T, InternalErrors> {
    fn rewrite(value: &mut serde_json::Value, from: u64, to: u32, count: &mut usize) {
        match value {
            serde_json::Value::Object(map) => {
                if map.len() == 1
                    && map.get("GLOBAL").and_then(serde_json::Value::as_u64) == Some(from)
                {
                    *value = serde_json::json!({ "GLOBAL": to });
                    *count += 1;
                    return;
                }
                map.values_mut().for_each(|v| rewrite(v, from, to, count));
            }
            serde_json::Value::Array(items) => {
                items.iter_mut().for_each(|v| rewrite(v, from, to, count))
            }
            _ => {}
        }
    }
    let mut value =
        serde_json::to_value(items).map_err(|e| InternalErrors::Internal(e.to_string()))?;
    rewrite(&mut value, from as u64, to, count);
    serde_json::from_value(value).map_err(|e| InternalErrors::Internal(e.to_string()))
}

/// A class parameter's value with class `from` as `to` (a number, its text,
/// or in a list); returns how many it changed.
fn rewrite_value(value: &mut ParamValue, from: u32, to: u32) -> usize {
    match value {
        ParamValue::Int(n) if *n == from as i64 => {
            *n = to as i64;
            1
        }
        ParamValue::Text(text) if text.trim() == from.to_string() => {
            *text = to.to_string();
            1
        }
        ParamValue::List(items) => items.iter_mut().map(|v| rewrite_value(v, from, to)).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::module_editing::{ClassRole, create_module};
    use evanalyzer_cfg::core_types::{ImageAddress, PipelineId, StepId};
    use evanalyzer_cfg::settings::classification_settings::Class;
    use evanalyzer_cfg::settings::pipeline_command_settings::{
        ClassifyObjectsSettings, ExtractObjectsSettings, ThresholdEntrySettings, ThresholdSettings,
    };
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    fn class(id: u32, name: &str) -> Class {
        Class {
            id,
            color: 0xFF0000,
            name: name.into(),
            notes: String::new(),
        }
    }

    fn step(id: u32, command: PipelineCommand) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(command),
        }
    }

    fn classify(input: u32, output: u32) -> PipelineCommand {
        PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
            input_classes: vec![ObjectClass::Global(input)],
            output_class: ObjectClass::Global(output),
            ..Default::default()
        })
    }

    /// Classes 1 "Nuclei" and 3 "Cells"; Threshold (segmentation class 3) →
    /// Extract Objects (no mapping) → Classify 3 into 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project
            .classification
            .classes_mut()
            .extend([class(1, "Nuclei"), class(3, "Cells")]);
        project.pipelines = vec![PipelineSettings {
            id: PipelineId(1),
            name: "P".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![
                step(
                    1,
                    PipelineCommand::Threshold(ThresholdSettings {
                        thresholds: vec![ThresholdEntrySettings {
                            segmentation_class: SegmentationClass(3),
                            ..Default::default()
                        }],
                    }),
                ),
                step(
                    2,
                    PipelineCommand::ExtractObjects(ExtractObjectsSettings::default()),
                ),
                step(3, classify(3, 1)),
            ],
        }];
        project
    }

    fn ids(project: &ProjectSettings) -> Vec<u32> {
        project
            .classification
            .classes()
            .iter()
            .map(|c| c.id)
            .collect()
    }

    #[test]
    fn free_ids_are_the_unused_ones_up_to_the_highest_choosable_id() {
        let project = project();
        // Background 0, 1 and 3 are taken.
        let free = free_class_ids(&project.classification);
        assert_eq!(free.first(), Some(&2));
        assert!(!free.contains(&0) && !free.contains(&1) && !free.contains(&3));
        assert_eq!(free.last(), Some(&MAX_CLASS_ID));
        assert_eq!(free.len(), 30);
        assert_eq!(next_class_id(&project.classification), 4);
    }

    #[test]
    fn a_new_id_reaches_commands_and_maps_the_segmentation_class() {
        let mut project = project();
        let change = change_class_id(&mut project, 3, 5).unwrap();
        assert_eq!(ids(&project), [0, 1, 5]);
        assert_eq!(
            change,
            ClassIdChange {
                references: 1,
                mappings_added: 1
            }
        );

        let steps = &project.pipelines[0].steps;
        let Some(PipelineCommand::Threshold(threshold)) = steps[0].command() else {
            panic!("threshold")
        };
        assert_eq!(
            threshold.thresholds[0].segmentation_class,
            SegmentationClass(3),
            "the segmentation keeps its number"
        );
        let Some(PipelineCommand::ExtractObjects(extract)) = steps[1].command() else {
            panic!("extract")
        };
        assert_eq!(extract.class_mapping.len(), 1);
        assert_eq!(
            extract.class_mapping[0].segmentation_class,
            SegmentationClass(3)
        );
        assert_eq!(
            extract.class_mapping[0].object_class,
            ObjectClass::Global(5)
        );
        let Some(PipelineCommand::ClassifyObjects(classify)) = steps[2].command() else {
            panic!("classify")
        };
        assert_eq!(classify.input_classes, [ObjectClass::Global(5)]);
        assert_eq!(
            classify.output_class,
            ObjectClass::Global(1),
            "other classes stay"
        );
    }

    #[test]
    fn an_explicit_mapping_is_rewritten_not_doubled() {
        let mut project = project();
        let Some(PipelineCommand::ExtractObjects(extract)) =
            project.pipelines[0].steps[1].command_mut()
        else {
            panic!("extract")
        };
        extract.class_mapping.push(ExtractClassMappingSettings {
            segmentation_class: SegmentationClass(3),
            object_class: ObjectClass::Global(3),
        });
        let change = change_class_id(&mut project, 3, 5).unwrap();
        assert_eq!(change.mappings_added, 0);
        let Some(PipelineCommand::ExtractObjects(extract)) =
            project.pipelines[0].steps[1].command()
        else {
            panic!("extract")
        };
        assert_eq!(extract.class_mapping.len(), 1);
        assert_eq!(
            extract.class_mapping[0].object_class,
            ObjectClass::Global(5)
        );
    }

    #[test]
    fn create_module_and_class_parameter_values_follow() {
        let mut project = project();
        create_module(
            &mut project,
            PipelineId(1),
            2..=2,
            "Reclassify",
            &[
                (ObjectClass::Global(3), ClassRole::Parameter),
                (ObjectClass::Global(1), ClassRole::Fixed),
            ],
        )
        .unwrap();
        let parameter = project.modules[0].parameters[0].key.clone();
        let StepContent::Module(instance) = &mut project.pipelines[0].steps[2].content else {
            panic!("module step")
        };
        instance
            .values
            .insert(parameter.clone(), ParamValue::Int(3));

        change_class_id(&mut project, 3, 5).unwrap();
        let definition = &project.modules[0];
        assert_eq!(
            definition.parameters[0].default,
            ParamValue::Text("5".into())
        );
        let StepContent::Module(instance) = &project.pipelines[0].steps[2].content else {
            panic!("module step")
        };
        assert_eq!(instance.values[&parameter], ParamValue::Int(5));
        let Some(PipelineCommand::ClassifyObjects(classify)) = definition.steps[0].command() else {
            panic!("classify")
        };
        assert_eq!(classify.input_classes, [ObjectClass::Global(5)]);
    }

    #[test]
    fn annotated_objects_and_the_tile_merge_settings_follow() {
        use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
        let mut project = project();
        project.tile_merge.classes_to_not_merge = vec![ObjectClass::Global(3)];
        let mut object = ObjectMetricSettings::default();
        object.object_class.insert(ObjectClass::Global(3));
        object
            .colocalized_with
            .insert(ObjectClass::Global(1), vec![]);
        object
            .colocalized_with
            .insert(ObjectClass::Global(3), vec![]);
        object
            .parents
            .insert(ObjectClass::Global(3), evanalyzer_cfg::core_types::ObjectId(7));
        let mut series = evanalyzer_cfg::settings::images_settings::SeriesSettings::default();
        series.objects.push(object);
        let mut image = evanalyzer_cfg::settings::images_settings::ImageEntry::default();
        image.series.insert(0, series);
        project.images.list.insert("a.tif".into(), image);

        change_class_id(&mut project, 3, 5).unwrap();
        assert_eq!(
            project.tile_merge.classes_to_not_merge,
            [ObjectClass::Global(5)]
        );
        let object = &project.images.list[0].series[&0].objects[0];
        assert!(object.object_class.contains(&ObjectClass::Global(5)));
        assert!(!object.object_class.contains(&ObjectClass::Global(3)));
        let keys: Vec<ObjectClass> = object.colocalized_with.keys().copied().collect();
        assert_eq!(
            keys,
            [ObjectClass::Global(1), ObjectClass::Global(5)],
            "same order"
        );
        assert_eq!(
            object.parents.get(&ObjectClass::Global(5)),
            Some(&evanalyzer_cfg::core_types::ObjectId(7))
        );
    }

    #[test]
    fn the_background_a_taken_id_and_an_unknown_class_are_refused() {
        let mut project = project();
        assert!(change_class_id(&mut project, 0, 7).is_err());
        assert!(change_class_id(&mut project, 3, 0).is_err());
        assert!(change_class_id(&mut project, 3, 1).is_err());
        assert!(change_class_id(&mut project, 9, 7).is_err());
        assert_eq!(ids(&project), [0, 1, 3], "nothing changed");
        assert_eq!(
            change_class_id(&mut project, 3, 3).unwrap(),
            ClassIdChange::default()
        );
    }
}
