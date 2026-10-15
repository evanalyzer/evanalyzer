//! The component library: components saved to `.evacomp` files (in the
//! templates folders, like project templates), inserted into a
//! project as a copy that lives on its own from then on.
//!
//! A copy remembers where it came from (`ComponentDefinition::origin`: file, key,
//! version, and a fingerprint of the definition as inserted), so a newer
//! version of the file shows as an update - and an update knows whether the
//! project changed the component since.

use crate::api::FileSystem;
use chrono::Utc;
use evanalyzer_cfg::{
    COMPONENT_TEMPLATE_EXTENSION, CURRENT_COMPONENT_TEMPLATE_SCHEMA_VERSION,
    core_types::{InternalErrors, PipelineId},
    settings::{
        components::{ComponentDefinition, ComponentOrigin},
        meta_data::MetaData,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
        templates::ComponentTemplate,
    },
};
use std::path::Path;

use crate::workspace::component_editing::{ClassRole, create_component, plan_component};

/// The keys of the components `steps` use - also the components those use inside -
/// each once, in the order first met.
fn used_keys(
    definitions: &[ComponentDefinition],
    steps: &[PipelineStepSettings],
    out: &mut Vec<String>,
) {
    for step in steps {
        if let StepContent::Component(instance) = &step.content
            && !out.contains(&instance.key)
        {
            out.push(instance.key.clone());
            if let Some(definition) = definitions.iter().find(|d| d.key == instance.key) {
                used_keys(definitions, &definition.steps, out);
            }
        }
    }
}

/// The definitions of the components `steps` use (also inside other components) -
/// what a template of them must carry along.
pub fn components_used_by(
    project: &ProjectSettings,
    steps: &[PipelineStepSettings],
) -> Vec<ComponentDefinition> {
    let mut keys = Vec::new();
    used_keys(&project.components, steps, &mut keys);
    keys.iter()
        .filter_map(|key| project.components.iter().find(|d| &d.key == key).cloned())
        .collect()
}

/// Adds the definitions the project doesn't have yet (by key) - after
/// inserting steps that use them (e.g. from a pipeline template). A component the
/// project already has stays as it is.
pub fn merge_component_definitions(
    project: &mut ProjectSettings,
    definitions: &[ComponentDefinition],
) {
    for definition in definitions {
        if !project.components.iter().any(|d| d.key == definition.key) {
            project.components.push(definition.clone());
        }
    }
}

/// A fingerprint of what `definition` does - its steps, parameters and
/// classes, not its name or where it came from. Stable across runs and
/// versions (FNV-1a over the file form), as it's saved in the project.
pub fn content_hash(definition: &ComponentDefinition) -> String {
    let content = serde_json::json!({
        "steps": definition.steps,
        "parameters": definition.parameters,
        "classes": definition.classes,
    });
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in content.to_string().bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Whether the project changed `definition` since it was inserted from the
/// library - an update would replace those changes. `false` for a component not
/// from the library.
pub fn modified_since_inserted(definition: &ComponentDefinition) -> bool {
    definition
        .origin
        .as_ref()
        .is_some_and(|o| !o.content_hash.is_empty() && o.content_hash != content_hash(definition))
}

/// The library file name of `path` ("cells.evacomp") - what `origin`
/// remembers.
pub fn library_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether library file `library` holds the component `definition` was inserted
/// from.
fn comes_from(
    definition: &ComponentDefinition,
    template: &ComponentTemplate,
    library: &str,
) -> bool {
    definition
        .origin
        .as_ref()
        .is_some_and(|o| o.library == library && o.origin_component_key == template.component.key)
}

fn origin_of(
    template: &ComponentTemplate,
    library: &str,
    definition: &ComponentDefinition,
) -> ComponentOrigin {
    ComponentOrigin {
        origin_component_key: template.component.key.clone(),
        library: library.to_string(),
        version: template.component.meta.version.unwrap_or(1),
        content_hash: content_hash(definition),
    }
}

/// The project's copy of the library component `template` (from file
/// `library`), if it has one.
pub fn project_copy<'a>(
    project: &'a ProjectSettings,
    template: &ComponentTemplate,
    library: &str,
) -> Option<&'a ComponentDefinition> {
    project
        .components
        .iter()
        .find(|d| comes_from(d, template, library))
}

/// The newer version of `definition` that one of the library files
/// `library` has: (index into `library`, its version) - `None` if it isn't
/// from the library or is up to date.
pub fn update_available(
    definition: &ComponentDefinition,
    library: &[(std::path::PathBuf, ComponentTemplate)],
) -> Option<(usize, u32)> {
    let origin = definition.origin.as_ref()?;
    library
        .iter()
        .enumerate()
        .find_map(|(index, (path, template))| {
            let version = template.component.meta.version.unwrap_or(1);
            (comes_from(definition, template, &library_name(path)) && version > origin.version)
                .then_some((index, version))
        })
}

/// "Save to library": writes component `key` of the project to `path` as a
/// `.evacomp` file with `meta` (name, version, ...) and the components it uses,
/// and links the project's component to the file - a newer version of it later
/// shows as an update.
pub fn save_component_template(
    project: &mut ProjectSettings,
    files: &dyn FileSystem,
    mut meta: MetaData,
    key: &str,
    path: &Path,
) -> Result<(), InternalErrors> {
    let definition = project
        .components
        .iter()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?
        .clone();
    meta.app_version = env!("CARGO_PKG_VERSION").to_string();
    let mut component = definition.clone();
    component.meta = meta;
    component.origin = None;
    let template = ComponentTemplate {
        schema_version: CURRENT_COMPONENT_TEMPLATE_SCHEMA_VERSION,
        uses: components_used_by(project, &definition.steps),
        component,
    };

    let mut path = path.to_path_buf();
    if path.extension().and_then(|e| e.to_str()) != Some(COMPONENT_TEMPLATE_EXTENSION) {
        path.set_extension(COMPONENT_TEMPLATE_EXTENSION);
    }
    let json = serde_json::to_string_pretty(&template)
        .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
    files.write_file(&path, json.as_bytes())?;

    let library = library_name(&path);
    if let Some(definition) = project.components.iter_mut().find(|d| d.key == key) {
        definition.meta.version = template.component.meta.version;
        definition.origin = Some(origin_of(&template, &library, definition));
    }
    Ok(())
}

/// "Save pipeline as component": writes all steps of pipeline `pipeline_id`
/// to `path` as a library component with `meta` - made as "Create component"
/// makes one, on a copy: the project doesn't change. Every project class the
/// steps use becomes a parameter (a class only segmented, never extracted,
/// stays as it is), so where the component is used its classes can be
/// mapped to that project's and their objects reach the results.
pub fn save_pipeline_as_component(
    project: &ProjectSettings,
    files: &dyn FileSystem,
    meta: MetaData,
    pipeline_id: PipelineId,
    path: &Path,
) -> Result<(), InternalErrors> {
    let pipeline = project
        .pipelines
        .iter()
        .find(|p| p.id == pipeline_id)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no pipeline {pipeline_id}")))?;
    if pipeline.steps.is_empty() {
        return Err(InternalErrors::InvalidArgument(
            "the pipeline has no steps".into(),
        ));
    }
    let range = 0..=pipeline.steps.len() - 1;
    let name = if meta.name.trim().is_empty() {
        pipeline.name.clone()
    } else {
        meta.name.clone()
    };

    let mut copy = project.clone();
    let roles: Vec<_> = plan_component(&copy, pipeline_id, range.clone())?
        .classes
        .iter()
        .map(|c| (c.class, ClassRole::Parameter))
        .collect();
    let key = create_component(&mut copy, pipeline_id, range, &name, &roles)?;
    save_component_template(&mut copy, files, MetaData { name, ..meta }, &key, path)
}

/// Inserts library component `template` (from file `library`) into the project:
/// a copy, with the components it uses the project doesn't have. A copy already
/// in the project is used as it is. Returns the key to use the component by.
pub fn insert_component_template(
    project: &mut ProjectSettings,
    template: &ComponentTemplate,
    library: &str,
) -> Result<String, InternalErrors> {
    if let Some(existing) = project_copy(project, template, library) {
        return Ok(existing.key.clone());
    }
    merge_component_definitions(project, &template.uses);
    let mut copy = template.component.clone();
    copy.meta.creation_time = Utc::now();
    if project.components.iter().any(|d| d.key == copy.key) {
        // The project has an unrelated component with that key.
        copy.key = (2..)
            .map(|n| format!("{}_{n}", template.component.key))
            .find(|key| !project.components.iter().any(|d| &d.key == key))
            .expect("some number is free");
    }
    copy.origin = Some(origin_of(template, library, &copy));
    let key = copy.key.clone();
    project.components.push(copy);
    Ok(key)
}

/// Replaces component `key` by library component `template` (a newer version from
/// file `library`): steps, parameters, classes, description. The key stays,
/// so the steps using it keep their values (by parameter id). Components the new
/// version uses inside and the project lacks are added.
pub fn update_from_library(
    project: &mut ProjectSettings,
    key: &str,
    template: &ComponentTemplate,
    library: &str,
) -> Result<(), InternalErrors> {
    merge_component_definitions(project, &template.uses);
    let definition = project
        .components
        .iter_mut()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no component {key}")))?;
    let mut updated = template.component.clone();
    updated.key = key.to_string();
    updated.origin = Some(origin_of(template, library, &updated));
    *definition = updated;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalFileSystem;
    use evanalyzer_cfg::core_types::{ComponentParamKey, PipelineId, StepId};
    use evanalyzer_cfg::settings::components::{
        ComponentInstance, ComponentParameterDef, ParamBinding, ParamValue,
    };
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    fn blur(id: u32, kernel_size: usize) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Command(PipelineCommand::Blur(BlurSettings { kernel_size })),
        }
    }

    fn uses(id: u32, key: &str) -> PipelineStepSettings {
        PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: StepContent::Component(ComponentInstance {
                key: key.into(),
                values: Default::default(),
            }),
        }
    }

    fn definition(key: &str, steps: Vec<PipelineStepSettings>) -> ComponentDefinition {
        ComponentDefinition {
            key: key.into(),
            meta: MetaData {
                name: key.to_lowercase(),
                ..Default::default()
            },
            classes: vec![],
            steps,
            parameters: vec![ComponentParameterDef {
                key: ComponentParamKey::from("1"),
                label: "Size".into(),
                description: String::new(),
                section: String::new(),
                default: ParamValue::Int(3),
                bindings: vec![ParamBinding {
                    step: StepId(1),
                    param: "kernel_size".into(),
                }],
            }],
            origin: None,
        }
    }

    /// A project with component SMOOTH (two blurs and the component INNER) used in
    /// pipeline 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.components = vec![
            definition("SMOOTH", vec![blur(1, 3), uses(2, "INNER")]),
            definition("INNER", vec![blur(1, 5)]),
        ];
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: "p".into(),
            description: None,
            image_source: Default::default(),
            enabled: true,
            steps: vec![uses(1, "SMOOTH")],
        });
        project
    }

    fn meta(version: u32) -> MetaData {
        MetaData {
            name: "Smooth".into(),
            version: Some(version),
            ..Default::default()
        }
    }

    fn saved(
        project: &mut ProjectSettings,
        dir: &Path,
        version: u32,
    ) -> (std::path::PathBuf, ComponentTemplate) {
        let path = dir.join("smooth.evacomp");
        save_component_template(
            project,
            &LocalFileSystem::default(),
            meta(version),
            "SMOOTH",
            &path,
        )
        .unwrap();
        let template: ComponentTemplate =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        (path, template)
    }

    #[test]
    fn saving_writes_the_component_with_the_components_it_uses_and_links_the_project_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        let (_, template) = saved(&mut project, dir.path(), 2);

        assert_eq!(template.component.key, "SMOOTH");
        assert_eq!(template.component.meta.version, Some(2));
        assert!(template.component.origin.is_none());
        let used: Vec<&str> = template.uses.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(used, ["INNER"]);

        let origin = project.components[0].origin.as_ref().unwrap();
        assert_eq!(
            (origin.library.as_str(), origin.version),
            ("smooth.evacomp", 2)
        );
        assert!(!modified_since_inserted(&project.components[0]));
    }

    fn saved_pipeline(
        project: &ProjectSettings,
        dir: &Path,
        meta: MetaData,
    ) -> Result<ComponentTemplate, InternalErrors> {
        let path = dir.join("pipeline.evacomp");
        save_pipeline_as_component(
            project,
            &LocalFileSystem::default(),
            meta,
            PipelineId(1),
            &path,
        )?;
        Ok(serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap())
    }

    #[test]
    fn saving_a_pipeline_writes_all_its_steps_as_a_component_and_leaves_the_project_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        project.pipelines[0].steps.push(blur(2, 7));
        let before = serde_json::to_value(&project).unwrap();

        let template = saved_pipeline(&project, dir.path(), MetaData::default()).unwrap();

        assert_eq!(
            template.component.meta.name, "p",
            "named after the pipeline"
        );
        assert_eq!(template.component.steps.len(), 2);
        assert!(template.component.origin.is_none());
        let mut used: Vec<&str> = template.uses.iter().map(|d| d.key.as_str()).collect();
        used.sort();
        assert_eq!(used, ["INNER", "SMOOTH"]);
        assert_eq!(serde_json::to_value(&project).unwrap(), before);
    }

    #[test]
    fn saving_a_pipeline_makes_its_classes_parameters() {
        use evanalyzer_cfg::core_types::ObjectClass;
        use evanalyzer_cfg::settings::classification_settings::Class;
        use evanalyzer_cfg::settings::pipeline_command_settings::ClassifyObjectsSettings;
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        for (id, name) in [(1, "Nuclei"), (2, "Big nuclei")] {
            project.classification.classes_mut().push(Class {
                id,
                color: 0xFF0000,
                name: name.into(),
                notes: String::new(),
            });
        }
        project.pipelines[0].steps = vec![PipelineStepSettings {
            id: StepId(1),
            enabled: true,
            content: StepContent::Command(PipelineCommand::ClassifyObjects(
                ClassifyObjectsSettings {
                    input_classes: vec![ObjectClass::Global(1)],
                    output_class: ObjectClass::Global(2),
                    ..Default::default()
                },
            )),
        }];

        let template = saved_pipeline(&project, dir.path(), meta(1)).unwrap();

        assert_eq!(template.component.meta.name, "Smooth");
        assert!(template.component.classes.is_empty(), "no local classes");
        let labels: Vec<&str> = template
            .component
            .parameters
            .iter()
            .map(|p| p.label.as_str())
            .collect();
        assert_eq!(labels, ["Nuclei", "Big nuclei"]);
    }

    #[test]
    fn saving_an_empty_or_unknown_pipeline_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        project.pipelines[0].steps.clear();
        assert!(saved_pipeline(&project, dir.path(), meta(1)).is_err());
        project.pipelines.clear();
        assert!(saved_pipeline(&project, dir.path(), meta(1)).is_err());
        assert!(!dir.path().join("pipeline.evacomp").exists());
    }

    #[test]
    fn inserting_copies_the_component_once_with_the_components_it_uses() {
        let dir = tempfile::tempdir().unwrap();
        let (path, template) = saved(&mut project(), dir.path(), 1);
        let library = library_name(&path);

        let mut other = ProjectSettings::default();
        let key = insert_component_template(&mut other, &template, &library).unwrap();
        assert_eq!(key, "SMOOTH");
        let keys: Vec<&str> = other.components.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["INNER", "SMOOTH"]);
        // Inserting again uses the copy.
        assert_eq!(
            insert_component_template(&mut other, &template, &library).unwrap(),
            "SMOOTH"
        );
        assert_eq!(other.components.len(), 2);

        // An unrelated component with that key: the copy gets its own.
        let mut clash = ProjectSettings::default();
        clash.components.push(definition("SMOOTH", vec![]));
        assert_eq!(
            insert_component_template(&mut clash, &template, &library).unwrap(),
            "SMOOTH_2"
        );
    }

    #[test]
    fn a_newer_library_version_is_an_update_that_keeps_the_key_and_knows_local_changes() {
        let dir = tempfile::tempdir().unwrap();
        let mut source = project();
        let (path, v1) = saved(&mut source, dir.path(), 1);
        let library = library_name(&path);
        let mut project = ProjectSettings::default();
        let key = insert_component_template(&mut project, &v1, &library).unwrap();
        let copy_of = |project: &ProjectSettings| {
            project
                .components
                .iter()
                .find(|d| d.key == key)
                .unwrap()
                .clone()
        };

        let mut v2 = v1.clone();
        v2.component.meta.version = Some(2);
        v2.component.steps[0] = blur(1, 9);
        let files = vec![(path.clone(), v2.clone())];
        assert_eq!(update_available(&copy_of(&project), &files), Some((0, 2)));
        assert_eq!(
            update_available(&copy_of(&project), &[(path.clone(), v1.clone())]),
            None
        );

        // A change in the project is noticed.
        project
            .components
            .iter_mut()
            .find(|d| d.key == key)
            .unwrap()
            .steps[0] = blur(1, 7);
        assert!(modified_since_inserted(&copy_of(&project)));

        update_from_library(&mut project, &key, &v2, &library).unwrap();
        let updated = copy_of(&project);
        assert_eq!(updated.key, key);
        let StepContent::Command(PipelineCommand::Blur(b)) = &updated.steps[0].content else {
            panic!("blur")
        };
        assert_eq!(b.kernel_size, 9);
        assert_eq!(updated.origin.as_ref().unwrap().version, 2);
        assert!(!modified_since_inserted(&updated));
        assert_eq!(update_available(&updated, &files), None);
    }

    #[test]
    fn a_template_of_steps_carries_the_components_they_use() {
        let project = project();
        let used: Vec<String> = components_used_by(&project, &project.pipelines[0].steps)
            .into_iter()
            .map(|d| d.key)
            .collect();
        assert_eq!(used, ["SMOOTH", "INNER"]);

        let mut other = ProjectSettings::default();
        other.components.push(definition("INNER", vec![blur(1, 1)]));
        merge_component_definitions(
            &mut other,
            &components_used_by(&project, &project.pipelines[0].steps),
        );
        let keys: Vec<&str> = other.components.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["INNER", "SMOOTH"], "the project's own INNER stays");
    }
}
