//! The module library: modules saved to `.evamodule` files (in the
//! templates folders, like project templates), inserted into a
//! project as a copy that lives on its own from then on.
//!
//! A copy remembers where it came from (`ModuleDefinition::origin`: file, key,
//! version, and a fingerprint of the definition as inserted), so a newer
//! version of the file shows as an update - and an update knows whether the
//! project changed the module since.

use crate::api::FileSystem;
use chrono::Utc;
use evanalyzer_cfg::{
    CURRENT_MODULE_TEMPLATE_SCHEMA_VERSION, MODULE_TEMPLATE_EXTENSION,
    core_types::{InternalErrors, PipelineId},
    settings::{
        meta_data::MetaData,
        modules::{ModuleDefinition, ModuleOrigin},
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
        templates::ModuleTemplate,
    },
};
use std::path::Path;

use crate::workspace::module_editing::{ClassRole, create_module, plan_module};

/// The keys of the modules `steps` use - also the modules those use inside -
/// each once, in the order first met.
fn used_keys(
    definitions: &[ModuleDefinition],
    steps: &[PipelineStepSettings],
    out: &mut Vec<String>,
) {
    for step in steps {
        if let StepContent::Module(instance) = &step.content
            && !out.contains(&instance.key)
        {
            out.push(instance.key.clone());
            if let Some(definition) = definitions.iter().find(|d| d.key == instance.key) {
                used_keys(definitions, &definition.steps, out);
            }
        }
    }
}

/// The definitions of the modules `steps` use (also inside other modules) -
/// what a template of them must carry along.
pub fn modules_used_by(
    project: &ProjectSettings,
    steps: &[PipelineStepSettings],
) -> Vec<ModuleDefinition> {
    let mut keys = Vec::new();
    used_keys(&project.modules, steps, &mut keys);
    keys.iter()
        .filter_map(|key| project.modules.iter().find(|d| &d.key == key).cloned())
        .collect()
}

/// Adds the definitions the project doesn't have yet (by key) - after
/// inserting steps that use them (e.g. from a pipeline template). A module the
/// project already has stays as it is.
pub fn merge_module_definitions(project: &mut ProjectSettings, definitions: &[ModuleDefinition]) {
    for definition in definitions {
        if !project.modules.iter().any(|d| d.key == definition.key) {
            project.modules.push(definition.clone());
        }
    }
}

/// What `definition` does - its steps, parameters and classes, not its name
/// or where it came from.
fn content_of(definition: &ModuleDefinition) -> serde_json::Value {
    serde_json::json!({
        "steps": definition.steps,
        "parameters": definition.parameters,
        "classes": definition.classes,
    })
}

/// A fingerprint of what `definition` does - its steps, parameters and
/// classes, and those of the modules it uses inside (found in
/// `definitions`), at any depth; not names or where they came from. Stable
/// across runs and versions (FNV-1a over the file form with its keys
/// sorted, so reordering fields in code doesn't change it), as it's saved in
/// the project.
pub fn content_hash(definition: &ModuleDefinition, definitions: &[ModuleDefinition]) -> String {
    let mut content = content_of(definition);
    let mut keys = Vec::new();
    used_keys(definitions, &definition.steps, &mut keys);
    let uses: Vec<serde_json::Value> = keys
        .iter()
        .filter_map(|key| definitions.iter().find(|d| &d.key == key))
        .map(content_of)
        .collect();
    if !uses.is_empty() {
        content["uses"] = serde_json::Value::Array(uses);
    }
    fingerprint(content)
}

/// FNV-1a over `content`'s JSON text with every object's keys sorted.
fn fingerprint(mut content: serde_json::Value) -> String {
    content.sort_all_objects();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in content.to_string().bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Whether the project changed `definition` since it was inserted from the
/// library - also a module it uses inside (in `definitions`). An update
/// would replace those changes. `false` for a module not from the library.
pub fn modified_since_inserted(
    definition: &ModuleDefinition,
    definitions: &[ModuleDefinition],
) -> bool {
    definition.origin.as_ref().is_some_and(|o| {
        !o.content_hash.is_empty() && o.content_hash != content_hash(definition, definitions)
    })
}

/// Adds the modules a library module uses inside (`uses`) so it runs
/// exactly the library's: one the project has with the same key and the same
/// content is shared; one with the same key but other content (changed in the
/// project, or unrelated) stays as it is - the library's is added under a key
/// of its own. Returns those new keys (library key → project key), already
/// used in what was added; see [`with_keys`].
fn add_used(
    project: &mut ProjectSettings,
    uses: &[ModuleDefinition],
) -> std::collections::BTreeMap<String, String> {
    let mut renamed: std::collections::BTreeMap<String, String> = Default::default();
    // A rename changes what uses the renamed one - again until nothing changes.
    loop {
        let mut changed = false;
        for definition in uses {
            if renamed.contains_key(&definition.key) {
                continue;
            }
            let wanted = with_keys(definition, &renamed);
            let clash = project
                .modules
                .iter()
                .find(|d| d.key == definition.key)
                .is_some_and(|ours| content_of(ours) != content_of(&wanted));
            if clash {
                let key = (2..)
                    .map(|n| format!("{}_{n}", definition.key))
                    .find(|key| {
                        !project.modules.iter().any(|d| &d.key == key)
                            && !renamed.values().any(|k| k == key)
                            && !uses.iter().any(|d| &d.key == key)
                    })
                    .expect("some number is free");
                renamed.insert(definition.key.clone(), key);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for definition in uses {
        let mut added = with_keys(definition, &renamed);
        if let Some(key) = renamed.get(&definition.key) {
            added.key = key.clone();
        }
        if !project.modules.iter().any(|d| d.key == added.key) {
            project.modules.push(added);
        }
    }
    renamed
}

/// `definition` with the modules it uses inside renamed as `renamed` says
/// (old key → new key).
fn with_keys(
    definition: &ModuleDefinition,
    renamed: &std::collections::BTreeMap<String, String>,
) -> ModuleDefinition {
    let mut definition = definition.clone();
    for step in &mut definition.steps {
        if let StepContent::Module(instance) = &mut step.content
            && let Some(key) = renamed.get(&instance.key)
        {
            instance.key = key.clone();
        }
    }
    definition
}

/// The library file name of `path` ("cells.evamodule") - what `origin`
/// remembers.
pub fn library_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether library file `library` holds the module `definition` was inserted
/// from.
fn comes_from(definition: &ModuleDefinition, template: &ModuleTemplate, library: &str) -> bool {
    definition
        .origin
        .as_ref()
        .is_some_and(|o| o.library == library && o.origin_module_key == template.module.key)
}

/// Where `definition` (in the project with `definitions`) came from: library
/// module `template` of file `library`.
fn origin_of(
    template: &ModuleTemplate,
    library: &str,
    definition: &ModuleDefinition,
    definitions: &[ModuleDefinition],
) -> ModuleOrigin {
    ModuleOrigin {
        origin_module_key: template.module.key.clone(),
        library: library.to_string(),
        version: template.module.meta.version.unwrap_or(1),
        content_hash: content_hash(definition, definitions),
    }
}

/// The project's copy of the library module `template` (from file
/// `library`), if it has one.
pub fn project_copy<'a>(
    project: &'a ProjectSettings,
    template: &ModuleTemplate,
    library: &str,
) -> Option<&'a ModuleDefinition> {
    project
        .modules
        .iter()
        .find(|d| comes_from(d, template, library))
}

/// The newer version of `definition` that one of the library files
/// `library` has: (index into `library`, its version) - `None` if it isn't
/// from the library or is up to date.
pub fn update_available(
    definition: &ModuleDefinition,
    library: &[(std::path::PathBuf, ModuleTemplate)],
) -> Option<(usize, u32)> {
    let origin = definition.origin.as_ref()?;
    library
        .iter()
        .enumerate()
        .find_map(|(index, (path, template))| {
            let version = template.module.meta.version.unwrap_or(1);
            (comes_from(definition, template, &library_name(path)) && version > origin.version)
                .then_some((index, version))
        })
}

/// "Save to library": writes module `key` of the project to `path` as a
/// `.evamodule` file with `meta` (name, version, ...) and the modules it uses,
/// and links the project's module to the file - a newer version of it later
/// shows as an update.
pub fn save_module_template(
    project: &mut ProjectSettings,
    files: &dyn FileSystem,
    mut meta: MetaData,
    key: &str,
    path: &Path,
) -> Result<(), InternalErrors> {
    let definition = project
        .modules
        .iter()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no module {key}")))?
        .clone();
    meta.app_version = env!("CARGO_PKG_VERSION").to_string();
    let mut module = definition.clone();
    module.meta = meta;
    module.origin = None;
    let template = ModuleTemplate {
        schema_version: CURRENT_MODULE_TEMPLATE_SCHEMA_VERSION,
        uses: modules_used_by(project, &definition.steps),
        module,
    };

    let mut path = path.to_path_buf();
    if path.extension().and_then(|e| e.to_str()) != Some(MODULE_TEMPLATE_EXTENSION) {
        path.set_extension(MODULE_TEMPLATE_EXTENSION);
    }
    let json = serde_json::to_string_pretty(&template)
        .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
    files.write_file(&path, json.as_bytes())?;

    let library = library_name(&path);
    let definitions = project.modules.clone();
    if let Some(definition) = project.modules.iter_mut().find(|d| d.key == key) {
        definition.meta.version = template.module.meta.version;
        definition.origin = Some(origin_of(&template, &library, definition, &definitions));
    }
    Ok(())
}

/// "Save pipeline as module": writes all steps of pipeline `pipeline_id`
/// to `path` as a library module with `meta` - made as "Create module"
/// makes one, on a copy: the project doesn't change. Every project class the
/// steps use becomes a parameter (a class only segmented, never extracted,
/// stays as it is), so where the module is used its classes can be
/// mapped to that project's and their objects reach the results.
pub fn save_pipeline_as_module(
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
    let roles: Vec<_> = plan_module(&copy, pipeline_id, range.clone())?
        .classes
        .iter()
        .map(|c| (c.class, ClassRole::Parameter))
        .collect();
    let key = create_module(&mut copy, pipeline_id, range, &name, &roles)?;
    save_module_template(&mut copy, files, MetaData { name, ..meta }, &key, path)
}

/// Inserts library module `template` (from file `library`) into the project:
/// a copy, with the modules it uses inside (see [`add_used`]). A copy
/// already in the project is used as it is. Returns the key to use the
/// module by.
pub fn insert_module_template(
    project: &mut ProjectSettings,
    template: &ModuleTemplate,
    library: &str,
) -> Result<String, InternalErrors> {
    if let Some(existing) = project_copy(project, template, library) {
        return Ok(existing.key.clone());
    }
    let renamed = add_used(project, &template.uses);
    let mut copy = with_keys(&template.module, &renamed);
    copy.meta.creation_time = Utc::now();
    if project.modules.iter().any(|d| d.key == copy.key) {
        // The project has an unrelated module with that key.
        copy.key = (2..)
            .map(|n| format!("{}_{n}", template.module.key))
            .find(|key| !project.modules.iter().any(|d| &d.key == key))
            .expect("some number is free");
    }
    copy.origin = Some(origin_of(template, library, &copy, &project.modules));
    let key = copy.key.clone();
    project.modules.push(copy);
    Ok(key)
}

/// Replaces module `key` by library module `template` (a newer version from
/// file `library`): steps, parameters, classes, description. The key stays,
/// so the steps using it keep their values (by parameter id). The modules the
/// new version uses inside come along (see [`add_used`]) - one the project
/// changed, or uses elsewhere differently, isn't touched.
pub fn update_from_library(
    project: &mut ProjectSettings,
    key: &str,
    template: &ModuleTemplate,
    library: &str,
) -> Result<(), InternalErrors> {
    if !project.modules.iter().any(|d| d.key == key) {
        return Err(InternalErrors::InvalidArgument(format!("no module {key}")));
    }
    let renamed = add_used(project, &template.uses);
    let mut updated = with_keys(&template.module, &renamed);
    updated.key = key.to_string();
    updated.origin = Some(origin_of(template, library, &updated, &project.modules));
    if let Some(definition) = project.modules.iter_mut().find(|d| d.key == key) {
        *definition = updated;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalFileSystem;
    use evanalyzer_cfg::core_types::{ModuleParamKey, PipelineId, StepId};
    use evanalyzer_cfg::settings::modules::{
        ModuleInstance, ModuleParameterDef, ParamBinding, ParamValue,
    };
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;

    #[test]
    fn the_fingerprint_does_not_depend_on_key_order() {
        let a = serde_json::json!({ "steps": [{ "x": 1, "y": 2 }], "classes": [] });
        let b = serde_json::json!({ "classes": [], "steps": [{ "y": 2, "x": 1 }] });
        assert_eq!(fingerprint(a.clone()), fingerprint(b));
        let c = serde_json::json!({ "steps": [{ "x": 1, "y": 3 }], "classes": [] });
        assert_ne!(fingerprint(a), fingerprint(c), "other content");
    }

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
            content: StepContent::Module(ModuleInstance {
                key: key.into(),
                values: Default::default(),
            }),
        }
    }

    fn definition(key: &str, steps: Vec<PipelineStepSettings>) -> ModuleDefinition {
        ModuleDefinition {
            key: key.into(),
            meta: MetaData {
                name: key.to_lowercase(),
                ..Default::default()
            },
            classes: vec![],
            steps,
            parameters: vec![ModuleParameterDef {
                key: ModuleParamKey::from("1"),
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

    /// A project with module SMOOTH (two blurs and the module INNER) used in
    /// pipeline 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.modules = vec![
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
    ) -> (std::path::PathBuf, ModuleTemplate) {
        let path = dir.join("smooth.evamodule");
        save_module_template(
            project,
            &LocalFileSystem::default(),
            meta(version),
            "SMOOTH",
            &path,
        )
        .unwrap();
        let template: ModuleTemplate =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        (path, template)
    }

    #[test]
    fn saving_writes_the_module_with_the_modules_it_uses_and_links_the_project_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        let (_, template) = saved(&mut project, dir.path(), 2);

        assert_eq!(template.module.key, "SMOOTH");
        assert_eq!(template.module.meta.version, Some(2));
        assert!(template.module.origin.is_none());
        let used: Vec<&str> = template.uses.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(used, ["INNER"]);

        let origin = project.modules[0].origin.as_ref().unwrap();
        assert_eq!(
            (origin.library.as_str(), origin.version),
            ("smooth.evamodule", 2)
        );
        assert!(!modified_since_inserted(
            &project.modules[0],
            &project.modules
        ));
    }

    fn saved_pipeline(
        project: &ProjectSettings,
        dir: &Path,
        meta: MetaData,
    ) -> Result<ModuleTemplate, InternalErrors> {
        let path = dir.join("pipeline.evamodule");
        save_pipeline_as_module(
            project,
            &LocalFileSystem::default(),
            meta,
            PipelineId(1),
            &path,
        )?;
        Ok(serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap())
    }

    #[test]
    fn saving_a_pipeline_writes_all_its_steps_as_a_module_and_leaves_the_project_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        project.pipelines[0].steps.push(blur(2, 7));
        let before = serde_json::to_value(&project).unwrap();

        let template = saved_pipeline(&project, dir.path(), MetaData::default()).unwrap();

        assert_eq!(template.module.meta.name, "p", "named after the pipeline");
        assert_eq!(template.module.steps.len(), 2);
        assert!(template.module.origin.is_none());
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

        assert_eq!(template.module.meta.name, "Smooth");
        assert!(template.module.classes.is_empty(), "no local classes");
        let labels: Vec<&str> = template
            .module
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
        assert!(!dir.path().join("pipeline.evamodule").exists());
    }

    #[test]
    fn inserting_copies_the_module_once_with_the_modules_it_uses() {
        let dir = tempfile::tempdir().unwrap();
        let (path, template) = saved(&mut project(), dir.path(), 1);
        let library = library_name(&path);

        let mut other = ProjectSettings::default();
        let key = insert_module_template(&mut other, &template, &library).unwrap();
        assert_eq!(key, "SMOOTH");
        let keys: Vec<&str> = other.modules.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["INNER", "SMOOTH"]);
        // Inserting again uses the copy.
        assert_eq!(
            insert_module_template(&mut other, &template, &library).unwrap(),
            "SMOOTH"
        );
        assert_eq!(other.modules.len(), 2);

        // An unrelated module with that key: the copy gets its own.
        let mut clash = ProjectSettings::default();
        clash.modules.push(definition("SMOOTH", vec![]));
        assert_eq!(
            insert_module_template(&mut clash, &template, &library).unwrap(),
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
        let key = insert_module_template(&mut project, &v1, &library).unwrap();
        let copy_of = |project: &ProjectSettings| {
            project
                .modules
                .iter()
                .find(|d| d.key == key)
                .unwrap()
                .clone()
        };

        let mut v2 = v1.clone();
        v2.module.meta.version = Some(2);
        v2.module.steps[0] = blur(1, 9);
        let files = vec![(path.clone(), v2.clone())];
        assert_eq!(update_available(&copy_of(&project), &files), Some((0, 2)));
        assert_eq!(
            update_available(&copy_of(&project), &[(path.clone(), v1.clone())]),
            None
        );

        // A change in the project is noticed.
        project
            .modules
            .iter_mut()
            .find(|d| d.key == key)
            .unwrap()
            .steps[0] = blur(1, 7);
        assert!(modified_since_inserted(
            &copy_of(&project),
            &project.modules
        ));

        update_from_library(&mut project, &key, &v2, &library).unwrap();
        let updated = copy_of(&project);
        assert_eq!(updated.key, key);
        let StepContent::Command(PipelineCommand::Blur(b)) = &updated.steps[0].content else {
            panic!("blur")
        };
        assert_eq!(b.kernel_size, 9);
        assert_eq!(updated.origin.as_ref().unwrap().version, 2);
        assert!(!modified_since_inserted(&updated, &project.modules));
        assert_eq!(update_available(&updated, &files), None);
    }

    /// SMOOTH v1 from the library in a fresh project.
    fn inserted(dir: &Path) -> (ProjectSettings, ModuleTemplate, String) {
        let (path, template) = saved(&mut project(), dir, 1);
        let library = library_name(&path);
        let mut project = ProjectSettings::default();
        insert_module_template(&mut project, &template, &library).unwrap();
        (project, template, library)
    }

    fn module<'a>(project: &'a ProjectSettings, key: &str) -> &'a ModuleDefinition {
        project.modules.iter().find(|d| d.key == key).unwrap()
    }

    fn inner_kernel(project: &ProjectSettings, key: &str) -> usize {
        let StepContent::Command(PipelineCommand::Blur(b)) = &module(project, key).steps[0].content
        else {
            panic!("blur")
        };
        b.kernel_size
    }

    #[test]
    fn a_change_to_a_module_used_inside_counts_as_a_change() {
        let dir = tempfile::tempdir().unwrap();
        let (mut project, _, _) = inserted(dir.path());
        let smooth = module(&project, "SMOOTH").clone();
        assert!(!modified_since_inserted(&smooth, &project.modules));

        project
            .modules
            .iter_mut()
            .find(|d| d.key == "INNER")
            .unwrap()
            .steps[0] = blur(1, 11);
        assert!(modified_since_inserted(&smooth, &project.modules));
    }

    #[test]
    fn an_update_leaves_a_changed_inner_module_alone_and_brings_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let (mut project, v1, library) = inserted(dir.path());
        // INNER changed here; v2 changes INNER too.
        project
            .modules
            .iter_mut()
            .find(|d| d.key == "INNER")
            .unwrap()
            .steps[0] = blur(1, 11);
        let mut v2 = v1.clone();
        v2.module.meta.version = Some(2);
        v2.uses[0].steps[0] = blur(1, 21);

        update_from_library(&mut project, "SMOOTH", &v2, &library).unwrap();

        assert_eq!(
            inner_kernel(&project, "INNER"),
            11,
            "the project's change stays"
        );
        let smooth = module(&project, "SMOOTH");
        let StepContent::Module(uses_inner) = &smooth.steps[1].content else {
            panic!("the step using the inner module")
        };
        assert_eq!(uses_inner.key, "INNER_2");
        assert_eq!(
            inner_kernel(&project, "INNER_2"),
            21,
            "the library's version"
        );
        assert!(!modified_since_inserted(smooth, &project.modules));
    }

    #[test]
    fn an_unchanged_inner_module_is_shared_on_update() {
        let dir = tempfile::tempdir().unwrap();
        let (mut project, v1, library) = inserted(dir.path());
        let mut v2 = v1.clone();
        v2.module.meta.version = Some(2);
        v2.module.steps[0] = blur(1, 9);
        update_from_library(&mut project, "SMOOTH", &v2, &library).unwrap();
        let keys: Vec<&str> = project.modules.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["INNER", "SMOOTH"], "no copy of an identical INNER");
    }

    #[test]
    fn inserting_next_to_an_unrelated_module_with_an_inner_ones_key_adds_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let (path, template) = saved(&mut project(), dir.path(), 1);
        let mut other = ProjectSettings::default();
        other.modules.push(definition("INNER", vec![blur(1, 99)]));

        let key = insert_module_template(&mut other, &template, &library_name(&path)).unwrap();
        assert_eq!(inner_kernel(&other, "INNER"), 99, "the project's own stays");
        let StepContent::Module(uses_inner) = &module(&other, &key).steps[1].content else {
            panic!("the step using the inner module")
        };
        assert_eq!(uses_inner.key, "INNER_2");
        assert_eq!(inner_kernel(&other, "INNER_2"), 5);
    }

    #[test]
    fn a_template_of_steps_carries_the_modules_they_use() {
        let project = project();
        let used: Vec<String> = modules_used_by(&project, &project.pipelines[0].steps)
            .into_iter()
            .map(|d| d.key)
            .collect();
        assert_eq!(used, ["SMOOTH", "INNER"]);

        let mut other = ProjectSettings::default();
        other.modules.push(definition("INNER", vec![blur(1, 1)]));
        merge_module_definitions(
            &mut other,
            &modules_used_by(&project, &project.pipelines[0].steps),
        );
        let keys: Vec<&str> = other.modules.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, ["INNER", "SMOOTH"], "the project's own INNER stays");
    }
}
