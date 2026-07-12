use evanalyzer_cfg::settings::templates::{PipelineTemplate, ProjectTemplate};
use evanalyzer_cfg::{PIPELINE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use std::path::{Path, PathBuf};

/// Returns the directory where templates shipped with the application are stored.
///
/// This is the `templates` subfolder next to the application binary.
pub fn get_app_templates_folder() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(std::env::temp_dir);
    exe_dir.join("templates")
}

/// Returns the directory where user created pipeline and project templates are stored.
///
/// The folder (and its parents) is created if it does not exist yet.
pub fn get_user_templates_folder() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(std::env::temp_dir);
    let folder = base.join("evanalyzer").join("templates");
    let _ = std::fs::create_dir_all(&folder);
    folder
}

/// Loads all `PipelineTemplate`s found in the user and app templates folders.
///
/// Files are matched by the [`PIPELINE_EXTENSIONS`] extension. The returned
/// templates are paired with the path they were loaded from.
pub fn load_pipeline_templates() -> Vec<(PathBuf, PipelineTemplate)> {
    let mut templates = Vec::new();
    for folder in [get_user_templates_folder(), get_app_templates_folder()] {
        load_templates_from_folder(&folder, PIPELINE_EXTENSIONS, &mut templates);
    }
    templates
}

/// Loads all `ProjectTemplate`s found in the user and app templates folders.
///
/// Files are matched by the [`PROJECT_FILE_TEMPLATE_EXTENSIONS`] extension. The
/// returned templates are paired with the path they were loaded from.
pub fn load_project_templates() -> Vec<(PathBuf, ProjectTemplate)> {
    let mut templates = Vec::new();
    for folder in [get_user_templates_folder(), get_app_templates_folder()] {
        load_templates_from_folder(&folder, PROJECT_FILE_TEMPLATE_EXTENSIONS, &mut templates);
    }
    templates
}

fn load_templates_from_folder<T: serde::de::DeserializeOwned>(
    folder: &Path,
    extension: &str,
    out: &mut Vec<(PathBuf, T)>,
) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let Ok(data) = std::fs::read_to_string(&path) else {
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
    use super::*;
    use evanalyzer_cfg::settings::classification_settings::ClassificationSettings;
    use evanalyzer_cfg::settings::meta_data::MetaData;
    use evanalyzer_cfg::settings::plate_settings::PlateSettings;

    fn pipeline_template(name: &str) -> PipelineTemplate {
        PipelineTemplate {
            meta: MetaData { name: name.into(), ..Default::default() },
            pipeline_steps: vec![],
        }
    }

    fn write(dir: &Path, filename: &str, contents: &str) {
        std::fs::write(dir.join(filename), contents).unwrap();
    }

    #[test]
    fn returns_empty_for_a_folder_that_does_not_exist() {
        let mut out: Vec<(PathBuf, PipelineTemplate)> = Vec::new();
        load_templates_from_folder(Path::new("/does/not/exist/at/all"), PIPELINE_EXTENSIONS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn loads_only_files_with_the_matching_extension() {
        let dir = tempfile::tempdir().unwrap();
        let json = serde_json::to_string(&pipeline_template("Match")).unwrap();
        write(dir.path(), "a.evapipe", &json);
        write(dir.path(), "b.txt", &json); // same content, wrong extension
        write(dir.path(), "c.evapipe.bak", &json); // extension is "bak", not "evapipe"

        let mut out: Vec<(PathBuf, PipelineTemplate)> = Vec::new();
        load_templates_from_folder(dir.path(), PIPELINE_EXTENSIONS, &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, dir.path().join("a.evapipe"));
        assert_eq!(out[0].1.meta.name, "Match");
    }

    #[test]
    fn skips_malformed_json_but_still_loads_the_valid_files() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "good.evapipe", &serde_json::to_string(&pipeline_template("Good")).unwrap());
        write(dir.path(), "bad.evapipe", "{ this is not valid json");

        let mut out: Vec<(PathBuf, PipelineTemplate)> = Vec::new();
        load_templates_from_folder(dir.path(), PIPELINE_EXTENSIONS, &mut out);

        assert_eq!(out.len(), 1, "the malformed file must be skipped, not abort the whole folder");
        assert_eq!(out[0].1.meta.name, "Good");
    }

    #[test]
    fn skips_a_file_that_matches_the_extension_but_is_not_readable_as_utf8() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("binary.evapipe"), [0xFFu8, 0xFE, 0x00, 0x01]).unwrap();
        write(dir.path(), "good.evapipe", &serde_json::to_string(&pipeline_template("Good")).unwrap());

        let mut out: Vec<(PathBuf, PipelineTemplate)> = Vec::new();
        load_templates_from_folder(dir.path(), PIPELINE_EXTENSIONS, &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.meta.name, "Good");
    }

    #[test]
    fn loads_project_templates_with_their_own_extension() {
        let dir = tempfile::tempdir().unwrap();
        let template = ProjectTemplate {
            meta: MetaData { name: "Proj".into(), ..Default::default() },
            classification: ClassificationSettings::default(),
            plate: PlateSettings::default(),
            pipelines: vec![pipeline_template("Inner")],
        };
        write(dir.path(), "a.evapt", &serde_json::to_string(&template).unwrap());

        let mut out: Vec<(PathBuf, ProjectTemplate)> = Vec::new();
        load_templates_from_folder(dir.path(), PROJECT_FILE_TEMPLATE_EXTENSIONS, &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.meta.name, "Proj");
        assert_eq!(out[0].1.pipelines.len(), 1);
        assert_eq!(out[0].1.pipelines[0].meta.name, "Inner");
    }
}
