//! The command group library: groups saved to `.evagroup` files (in the
//! templates folders, like pipeline and project templates), inserted into a
//! project as a copy that lives on its own from then on.
//!
//! A copy remembers where it came from (`GroupDefinition::origin`: file, key,
//! version, and a fingerprint of the definition as inserted), so a newer
//! version of the file shows as an update - and an update knows whether the
//! project changed the group since.

use crate::api::FileSystem;
use chrono::Utc;
use evanalyzer_cfg::{
    CURRENT_GROUP_TEMPLATE_SCHEMA_VERSION, GROUP_TEMPLATE_EXTENSION,
    core_types::InternalErrors,
    settings::{
        command_groups::{GroupDefinition, GroupOrigin},
        meta_data::MetaData,
        pipeline_settings::{PipelineStepSettings, StepContent},
        project_settings::ProjectSettings,
        templates::GroupTemplate,
    },
};
use std::path::Path;

/// The keys of the groups `steps` use - also the groups those use inside -
/// each once, in the order first met.
fn used_keys(
    definitions: &[GroupDefinition],
    steps: &[PipelineStepSettings],
    out: &mut Vec<String>,
) {
    for step in steps {
        if let StepContent::Group(instance) = &step.content
            && !out.contains(&instance.key)
        {
            out.push(instance.key.clone());
            if let Some(definition) = definitions.iter().find(|d| d.key == instance.key) {
                used_keys(definitions, &definition.steps, out);
            }
        }
    }
}

/// The definitions of the groups `steps` use (also inside other groups) -
/// what a template of them must carry along.
pub fn groups_used_by(
    project: &ProjectSettings,
    steps: &[PipelineStepSettings],
) -> Vec<GroupDefinition> {
    let mut keys = Vec::new();
    used_keys(&project.command_groups, steps, &mut keys);
    keys.iter()
        .filter_map(|key| {
            project
                .command_groups
                .iter()
                .find(|d| &d.key == key)
                .cloned()
        })
        .collect()
}

/// Adds the definitions the project doesn't have yet (by key) - after
/// inserting steps that use them (e.g. from a pipeline template). A group the
/// project already has stays as it is.
pub fn merge_group_definitions(project: &mut ProjectSettings, definitions: &[GroupDefinition]) {
    for definition in definitions {
        if !project
            .command_groups
            .iter()
            .any(|d| d.key == definition.key)
        {
            project.command_groups.push(definition.clone());
        }
    }
}

/// A fingerprint of what `definition` does - its steps, parameters and
/// classes, not its name or where it came from. Stable across runs and
/// versions (FNV-1a over the file form), as it's saved in the project.
pub fn content_hash(definition: &GroupDefinition) -> String {
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
/// library - an update would replace those changes. `false` for a group not
/// from the library.
pub fn modified_since_inserted(definition: &GroupDefinition) -> bool {
    definition
        .origin
        .as_ref()
        .is_some_and(|o| !o.content_hash.is_empty() && o.content_hash != content_hash(definition))
}

/// The library file name of `path` ("cells.evagroup") - what `origin`
/// remembers.
pub fn library_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether library file `library` holds the group `definition` was inserted
/// from.
fn comes_from(definition: &GroupDefinition, template: &GroupTemplate, library: &str) -> bool {
    definition
        .origin
        .as_ref()
        .is_some_and(|o| o.library == library && o.origin_group_key == template.group.key)
}

fn origin_of(template: &GroupTemplate, library: &str, definition: &GroupDefinition) -> GroupOrigin {
    GroupOrigin {
        origin_group_key: template.group.key.clone(),
        library: library.to_string(),
        version: template.group.meta.version.unwrap_or(1),
        content_hash: content_hash(definition),
    }
}

/// The project's copy of the library group `template` (from file
/// `library`), if it has one.
pub fn project_copy<'a>(
    project: &'a ProjectSettings,
    template: &GroupTemplate,
    library: &str,
) -> Option<&'a GroupDefinition> {
    project
        .command_groups
        .iter()
        .find(|d| comes_from(d, template, library))
}

/// The newer version of `definition` that one of the library files
/// `library` has: (index into `library`, its version) - `None` if it isn't
/// from the library or is up to date.
pub fn update_available(
    definition: &GroupDefinition,
    library: &[(std::path::PathBuf, GroupTemplate)],
) -> Option<(usize, u32)> {
    let origin = definition.origin.as_ref()?;
    library
        .iter()
        .enumerate()
        .find_map(|(index, (path, template))| {
            let version = template.group.meta.version.unwrap_or(1);
            (comes_from(definition, template, &library_name(path)) && version > origin.version)
                .then_some((index, version))
        })
}

/// "Save to library": writes group `key` of the project to `path` as a
/// `.evagroup` file with `meta` (name, version, ...) and the groups it uses,
/// and links the project's group to the file - a newer version of it later
/// shows as an update.
pub fn save_group_template(
    project: &mut ProjectSettings,
    files: &dyn FileSystem,
    mut meta: MetaData,
    key: &str,
    path: &Path,
) -> Result<(), InternalErrors> {
    let definition = project
        .command_groups
        .iter()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no command group {key}")))?
        .clone();
    meta.app_version = env!("CARGO_PKG_VERSION").to_string();
    let mut group = definition.clone();
    group.meta = meta;
    group.origin = None;
    let template = GroupTemplate {
        schema_version: CURRENT_GROUP_TEMPLATE_SCHEMA_VERSION,
        uses: groups_used_by(project, &definition.steps),
        group,
    };

    let mut path = path.to_path_buf();
    if path.extension().and_then(|e| e.to_str()) != Some(GROUP_TEMPLATE_EXTENSION) {
        path.set_extension(GROUP_TEMPLATE_EXTENSION);
    }
    let json = serde_json::to_string_pretty(&template)
        .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
    files.write_file(&path, json.as_bytes())?;

    let library = library_name(&path);
    if let Some(definition) = project.command_groups.iter_mut().find(|d| d.key == key) {
        definition.meta.version = template.group.meta.version;
        definition.origin = Some(origin_of(&template, &library, definition));
    }
    Ok(())
}

/// Inserts library group `template` (from file `library`) into the project:
/// a copy, with the groups it uses the project doesn't have. A copy already
/// in the project is used as it is. Returns the key to use the group by.
pub fn insert_group_template(
    project: &mut ProjectSettings,
    template: &GroupTemplate,
    library: &str,
) -> Result<String, InternalErrors> {
    if let Some(existing) = project_copy(project, template, library) {
        return Ok(existing.key.clone());
    }
    merge_group_definitions(project, &template.uses);
    let mut copy = template.group.clone();
    copy.meta.creation_time = Utc::now();
    if project.command_groups.iter().any(|d| d.key == copy.key) {
        // The project has an unrelated group with that key.
        copy.key = (2..)
            .map(|n| format!("{}_{n}", template.group.key))
            .find(|key| !project.command_groups.iter().any(|d| &d.key == key))
            .expect("some number is free");
    }
    copy.origin = Some(origin_of(template, library, &copy));
    let key = copy.key.clone();
    project.command_groups.push(copy);
    Ok(key)
}

/// Replaces group `key` by library group `template` (a newer version from
/// file `library`): steps, parameters, classes, description. The key stays,
/// so the steps using it keep their values (by parameter id). Groups the new
/// version uses inside and the project lacks are added.
pub fn update_from_library(
    project: &mut ProjectSettings,
    key: &str,
    template: &GroupTemplate,
    library: &str,
) -> Result<(), InternalErrors> {
    merge_group_definitions(project, &template.uses);
    let definition = project
        .command_groups
        .iter_mut()
        .find(|d| d.key == key)
        .ok_or_else(|| InternalErrors::InvalidArgument(format!("no command group {key}")))?;
    let mut updated = template.group.clone();
    updated.key = key.to_string();
    updated.origin = Some(origin_of(template, library, &updated));
    *definition = updated;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalFileSystem;
    use evanalyzer_cfg::core_types::{GroupParamKey, PipelineId, StepId};
    use evanalyzer_cfg::settings::command_groups::{
        GroupInstance, GroupParameterDef, ParamBinding, ParamValue,
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
            content: StepContent::Group(GroupInstance {
                key: key.into(),
                values: Default::default(),
            }),
        }
    }

    fn definition(key: &str, steps: Vec<PipelineStepSettings>) -> GroupDefinition {
        GroupDefinition {
            key: key.into(),
            meta: MetaData {
                name: key.to_lowercase(),
                ..Default::default()
            },
            classes: vec![],
            steps,
            parameters: vec![GroupParameterDef {
                key: GroupParamKey::from("1"),
                label: "Size".into(),
                description: String::new(),
                default: ParamValue::Int(3),
                bindings: vec![ParamBinding {
                    step: StepId(1),
                    param: "kernel_size".into(),
                }],
            }],
            origin: None,
        }
    }

    /// A project with group SMOOTH (two blurs and the group INNER) used in
    /// pipeline 1.
    fn project() -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.command_groups = vec![
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
    ) -> (std::path::PathBuf, GroupTemplate) {
        let path = dir.join("smooth.evagroup");
        save_group_template(
            project,
            &LocalFileSystem::default(),
            meta(version),
            "SMOOTH",
            &path,
        )
        .unwrap();
        let template: GroupTemplate =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        (path, template)
    }

    #[test]
    fn saving_writes_the_group_with_the_groups_it_uses_and_links_the_project_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = project();
        let (_, template) = saved(&mut project, dir.path(), 2);

        assert_eq!(template.group.key, "SMOOTH");
        assert_eq!(template.group.meta.version, Some(2));
        assert!(template.group.origin.is_none());
        let used: Vec<&str> = template.uses.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(used, ["INNER"]);

        let origin = project.command_groups[0].origin.as_ref().unwrap();
        assert_eq!(
            (origin.library.as_str(), origin.version),
            ("smooth.evagroup", 2)
        );
        assert!(!modified_since_inserted(&project.command_groups[0]));
    }

    #[test]
    fn inserting_copies_the_group_once_with_the_groups_it_uses() {
        let dir = tempfile::tempdir().unwrap();
        let (path, template) = saved(&mut project(), dir.path(), 1);
        let library = library_name(&path);

        let mut other = ProjectSettings::default();
        let key = insert_group_template(&mut other, &template, &library).unwrap();
        assert_eq!(key, "SMOOTH");
        let keys: Vec<&str> = other
            .command_groups
            .iter()
            .map(|d| d.key.as_str())
            .collect();
        assert_eq!(keys, ["INNER", "SMOOTH"]);
        // Inserting again uses the copy.
        assert_eq!(
            insert_group_template(&mut other, &template, &library).unwrap(),
            "SMOOTH"
        );
        assert_eq!(other.command_groups.len(), 2);

        // An unrelated group with that key: the copy gets its own.
        let mut clash = ProjectSettings::default();
        clash.command_groups.push(definition("SMOOTH", vec![]));
        assert_eq!(
            insert_group_template(&mut clash, &template, &library).unwrap(),
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
        let key = insert_group_template(&mut project, &v1, &library).unwrap();
        let copy_of = |project: &ProjectSettings| {
            project
                .command_groups
                .iter()
                .find(|d| d.key == key)
                .unwrap()
                .clone()
        };

        let mut v2 = v1.clone();
        v2.group.meta.version = Some(2);
        v2.group.steps[0] = blur(1, 9);
        let files = vec![(path.clone(), v2.clone())];
        assert_eq!(update_available(&copy_of(&project), &files), Some((0, 2)));
        assert_eq!(
            update_available(&copy_of(&project), &[(path.clone(), v1.clone())]),
            None
        );

        // A change in the project is noticed.
        project
            .command_groups
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
    fn a_template_of_steps_carries_the_groups_they_use() {
        let project = project();
        let used: Vec<String> = groups_used_by(&project, &project.pipelines[0].steps)
            .into_iter()
            .map(|d| d.key)
            .collect();
        assert_eq!(used, ["SMOOTH", "INNER"]);

        let mut other = ProjectSettings::default();
        other
            .command_groups
            .push(definition("INNER", vec![blur(1, 1)]));
        merge_group_definitions(
            &mut other,
            &groups_used_by(&project, &project.pipelines[0].steps),
        );
        let keys: Vec<&str> = other
            .command_groups
            .iter()
            .map(|d| d.key.as_str())
            .collect();
        assert_eq!(keys, ["INNER", "SMOOTH"], "the project's own INNER stays");
    }
}
