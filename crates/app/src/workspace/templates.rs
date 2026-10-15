use crate::api::{Backend, FileSystem};
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::templates::{ComponentTemplate, ProjectTemplate};
use evanalyzer_cfg::{COMPONENT_TEMPLATE_EXTENSION, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use std::path::{Path, PathBuf};

/// Loads a single `ProjectTemplate` from an arbitrary path (e.g. one picked
/// via a file-open dialog), rather than scanning a whole templates folder
/// like [`load_project_templates`] does. Unlike that bulk scan - which
/// silently skips files that fail to parse, since one bad file in a folder
/// shouldn't block the rest - a single explicitly-opened file surfaces its
/// error so the caller can show it to the user.
pub fn load_project_template_from_file(
    files: &dyn FileSystem,
    path: &Path,
) -> Result<ProjectTemplate, InternalErrors> {
    let data = files.read_file(path)?;
    serde_json::from_slice(&data)
        .map_err(|e| InternalErrors::Internal(format!("Failed to parse project template: {e}")))
}

/// Loads all component library files (`.evacomp`, see `ComponentTemplate`)
/// found in the backend's user and app templates folders.
pub fn load_component_templates(backend: &dyn Backend) -> Vec<(PathBuf, ComponentTemplate)> {
    load_templates(backend, COMPONENT_TEMPLATE_EXTENSION)
}

/// Loads all `ProjectTemplate`s found in the backend's user and app
/// templates folders.
///
/// Files are matched by the [`PROJECT_FILE_TEMPLATE_EXTENSIONS`] extension. The
/// returned templates are paired with the path they were loaded from.
pub fn load_project_templates(backend: &dyn Backend) -> Vec<(PathBuf, ProjectTemplate)> {
    load_templates(backend, PROJECT_FILE_TEMPLATE_EXTENSIONS)
}

fn load_templates<T: serde::de::DeserializeOwned>(
    backend: &dyn Backend,
    extension: &str,
) -> Vec<(PathBuf, T)> {
    let mut templates = Vec::new();
    match backend.template_folders() {
        Ok(folders) => {
            for folder in [folders.user, folders.bundled] {
                load_templates_from_folder(backend.files(), &folder, extension, &mut templates);
            }
        }
        Err(e) => log::warn!("Could not locate the template folders: {e}"),
    }
    templates
}

fn load_templates_from_folder<T: serde::de::DeserializeOwned>(
    files: &dyn FileSystem,
    folder: &Path,
    extension: &str,
    out: &mut Vec<(PathBuf, T)>,
) {
    let Ok(entries) = files.list_dir(folder) else {
        return;
    };
    for entry in entries.into_iter().filter(|e| !e.is_dir) {
        let path = entry.path;
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let Some(data) = files
            .read_file(&path)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
        else {
            continue;
        };
        match serde_json::from_str::<T>(&data) {
            Ok(template) => out.push((path, template)),
            Err(e) => log::warn!("Failed to load template {}: {}", path.display(), e),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::LocalFileSystem;

    use super::*;
    use evanalyzer_cfg::core_types::{ImageAddress, PipelineId};
    use evanalyzer_cfg::settings::classification_settings::ClassificationSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
    use evanalyzer_cfg::settings::plate_settings::PlateSettings;

    fn project_template(name: &str) -> ProjectTemplate {
        ProjectTemplate {
            meta: MetaData {
                name: name.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn inline_pipeline(id: PipelineId, name: &str) -> PipelineSettings {
        PipelineSettings {
            name: name.into(),
            id,
            description: None,
            image_source: ImageAddress::Scratchpad,
            enabled: true,
            steps: vec![],
        }
    }

    fn write(dir: &Path, filename: &str, contents: &str) {
        std::fs::write(dir.join(filename), contents).unwrap();
    }

    #[test]
    fn returns_empty_for_a_folder_that_does_not_exist() {
        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(
            &LocalFileSystem::default(),
            Path::new("/does/not/exist/at/all"),
            PROJECT_FILE_TEMPLATE_EXTENSIONS,
            &mut out,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn loads_only_files_with_the_matching_extension() {
        let dir = tempfile::tempdir().unwrap();
        let json = serde_json::to_string(&project_template("Match")).unwrap();
        write(dir.path(), "a.evapt", &json);
        write(dir.path(), "b.txt", &json); // same content, wrong extension
        write(dir.path(), "c.evapt.bak", &json); // extension is "bak", not "evapt"

        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(
            &LocalFileSystem::default(),
            dir.path(),
            PROJECT_FILE_TEMPLATE_EXTENSIONS,
            &mut out,
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, dir.path().join("a.evapt"));
        assert_eq!(out[0].1.meta.name, "Match");
    }

    #[test]
    fn skips_malformed_json_but_still_loads_the_valid_files() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "good.evapt",
            &serde_json::to_string(&project_template("Good")).unwrap(),
        );
        write(dir.path(), "bad.evapt", "{ this is not valid json");

        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(
            &LocalFileSystem::default(),
            dir.path(),
            PROJECT_FILE_TEMPLATE_EXTENSIONS,
            &mut out,
        );

        assert_eq!(
            out.len(),
            1,
            "the malformed file must be skipped, not abort the whole folder"
        );
        assert_eq!(out[0].1.meta.name, "Good");
    }

    #[test]
    fn skips_a_file_that_matches_the_extension_but_is_not_readable_as_utf8() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("binary.evapt"), [0xFFu8, 0xFE, 0x00, 0x01]).unwrap();
        write(
            dir.path(),
            "good.evapt",
            &serde_json::to_string(&project_template("Good")).unwrap(),
        );

        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(
            &LocalFileSystem::default(),
            dir.path(),
            PROJECT_FILE_TEMPLATE_EXTENSIONS,
            &mut out,
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.meta.name, "Good");
    }

    #[test]
    fn loads_project_templates_with_their_own_extension() {
        let dir = tempfile::tempdir().unwrap();
        let template = ProjectTemplate {
            meta: MetaData {
                name: "Proj".into(),
                ..Default::default()
            },
            classification: ClassificationSettings::default(),
            plate: PlateSettings::default(),
            pipelines: vec![inline_pipeline(PipelineId(0), "Inner")],
            ..Default::default()
        };
        write(
            dir.path(),
            "a.evapt",
            &serde_json::to_string(&template).unwrap(),
        );

        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(
            &LocalFileSystem::default(),
            dir.path(),
            PROJECT_FILE_TEMPLATE_EXTENSIONS,
            &mut out,
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.meta.name, "Proj");
        assert_eq!(out[0].1.pipelines.len(), 1);
        assert_eq!(out[0].1.pipelines[0].name, "Inner");
    }
}
