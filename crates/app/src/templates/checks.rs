//! # template_checks
//!
//! Whether a project template or a library module is usable - not only
//! readable: made into a project the way the app does it, the project's
//! checks find no error, a run's job can be made, and - given an image - a
//! preview of it runs. A module is used twice in a pipeline and twice inside
//! another module, with its default values, as users do.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::backend::Backend;
use crate::jobs::{PreviewRequest, PreviewViewport, StartPreviewError};
use crate::modules::checks::{Severity, check_project};
use crate::modules::insights::resolved_commands;
use crate::modules::library::insert_module_template;
use crate::project::ProjectWithRuntime;
use evanalyzer_cfg::core_types::{ImageAddress, InternalErrors, PipelineId, StepId};
use evanalyzer_cfg::settings::modules::{ModuleDefinition, ModuleInstance};
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_settings::{
    PipelineSettings, PipelineStepSettings, StepContent,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_cfg::settings::templates::{ModuleTemplate, ProjectTemplate};
use std::path::{Path, PathBuf};

/// What checking a template found.
#[derive(Debug, Default, PartialEq)]
pub struct TemplateCheck {
    /// What keeps it from being used - empty: usable.
    pub problems: Vec<String>,
    /// Why no preview ran, if none did although an image was given (e.g.
    /// it needs an AI model).
    pub not_run: Option<String>,
}

/// Checks project template `template`; with `image`, also runs a preview
/// of it on that image.
pub fn check_project_template(
    template: &ProjectTemplate,
    image: Option<&Path>,
    backend: &dyn Backend,
) -> TemplateCheck {
    let mut project = ProjectWithRuntime::default();
    project.apply_project_template(template);
    check_usable(project.settings, image, backend)
}

/// Checks library module `template` (from file `library`): used twice in a
/// pipeline and twice inside another module, with its default values; with
/// `image`, also runs a preview of that on the image.
pub fn check_module_template(
    template: &ModuleTemplate,
    library: &str,
    image: Option<&Path>,
    backend: &dyn Backend,
) -> TemplateCheck {
    let mut project = ProjectWithRuntime::default();
    let key = match insert_module_template(&mut project.settings, template, library) {
        Ok(key) => key,
        Err(e) => {
            return TemplateCheck {
                problems: vec![format!("can't be inserted into a project: {e}")],
                not_run: None,
            };
        }
    };
    let uses = |id: u32, key: &str| PipelineStepSettings {
        id: StepId(id),
        enabled: true,
        content: StepContent::Module(ModuleInstance {
            key: key.to_string(),
            values: Default::default(),
        }),
    };
    let wrapper = (1..)
        .map(|n| format!("TWICE_{n}"))
        .find(|k| !project.modules.iter().any(|d| &d.key == k))
        .expect("some key is free");
    project.modules.push(ModuleDefinition {
        key: wrapper.clone(),
        meta: Default::default(),
        classes: vec![],
        steps: vec![uses(1, &key), uses(2, &key)],
        parameters: vec![],
        origin: None,
    });
    project.add_pipeline(PipelineSettings {
        id: PipelineId(1),
        name: "Module check".into(),
        description: None,
        image_source: ImageAddress::Channel(0),
        enabled: true,
        steps: vec![uses(1, &key), uses(2, &key), uses(3, &wrapper)],
    });
    check_usable(project.settings, image, backend)
}

/// Whether `path` is validated as templates: a template or module file,
/// or a folder.
pub fn is_template_path(path: &Path) -> bool {
    path.is_dir() || is_module(path) || path.extension().is_some_and(|e| e == "evatemplate")
}

/// Checks template, module or folder `path` (`image`: to run a preview
/// on) and prints a line per file - for the command line (`project
/// validate`, the `check_templates` example in CI). Fails if one isn't
/// usable.
pub fn validate_templates(
    path: &Path,
    image: Option<&Path>,
    backend: &dyn Backend,
) -> Result<(), InternalErrors> {
    if let Some(image) = image
        && !image.is_file()
    {
        return Err(InternalErrors::InvalidArgument(format!(
            "--image {}: no such file",
            image.display()
        )));
    }
    let mut files = Vec::new();
    if path.is_dir() {
        collect(path, &mut files);
        files.sort();
    } else {
        files.push(path.to_path_buf());
    }
    if files.is_empty() {
        return Err(InternalErrors::InvalidArgument(format!(
            "no .evatemplate or .evamodule files under {}",
            path.display()
        )));
    }
    let folder = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    let mut failed = 0;
    for path in &files {
        let check = check_template_file(path, image, backend);
        let shown = path.strip_prefix(folder).unwrap_or(path).display();
        if check.problems.is_empty() {
            match &check.not_run {
                Some(why) => println!("ok       {shown} (no preview: {why})"),
                None => println!("ok       {shown}"),
            }
        } else {
            failed += 1;
            println!("FAILED   {shown}");
            for problem in &check.problems {
                println!("         - {problem}");
            }
        }
    }
    println!("{} file(s) checked, {failed} not usable", files.len());
    if failed > 0 {
        return Err(InternalErrors::InvalidArgument(format!(
            "{failed} template(s) not usable"
        )));
    }
    Ok(())
}

/// The check of one template file.
pub fn check_template_file(
    path: &Path,
    image: Option<&Path>,
    backend: &dyn Backend,
) -> TemplateCheck {
    let unreadable = |e: String| TemplateCheck {
        problems: vec![e],
        not_run: None,
    };
    let json = match std::fs::read_to_string(path) {
        Ok(json) => json,
        Err(e) => return unreadable(format!("can't be read: {e}")),
    };
    if is_module(path) {
        match serde_json::from_str::<ModuleTemplate>(&json) {
            Ok(template) => {
                let library = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                check_module_template(&template, &library, image, backend)
            }
            Err(e) => unreadable(format!("isn't a module: {e}")),
        }
    } else {
        match serde_json::from_str::<ProjectTemplate>(&json) {
            Ok(template) => check_project_template(&template, image, backend),
            Err(e) => unreadable(format!("isn't a project template: {e}")),
        }
    }
}

fn is_module(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "evamodule")
}

/// The template files under `folder`, `.git` left out.
fn collect(folder: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n != ".git") {
                collect(&path, out);
            }
        } else if is_module(&path) || path.extension().is_some_and(|e| e == "evatemplate") {
            out.push(path);
        }
    }
}

/// The checks of a project made from a template.
fn check_usable(
    mut settings: ProjectSettings,
    image: Option<&Path>,
    backend: &dyn Backend,
) -> TemplateCheck {
    let mut check = TemplateCheck::default();
    check.problems.extend(
        check_project(&settings)
            .into_iter()
            .filter(|p| p.severity == Severity::Error)
            .map(|p| match (&p.module, p.step) {
                (Some(module), _) => format!("module {module}: {}", p.message),
                (None, Some((pipeline, step))) => {
                    format!("pipeline {}, step {}: {}", pipeline.0, step.0, p.message)
                }
                (None, None) => p.message,
            }),
    );

    // A run's job, without images: what the job generator refuses.
    let dir = match CheckFolder::new() {
        Ok(dir) => dir,
        Err(e) => {
            check
                .problems
                .push(format!("no temporary folder for the check: {e}"));
            return check;
        }
    };
    let images = dir.path().join("images");
    if let Err(e) = std::fs::create_dir_all(&images) {
        check.problems.push(format!("no temporary folder: {e}"));
        return check;
    }
    let mut empty = settings.clone();
    empty.images.root = Some(images.clone());
    if let Err(e) =
        evanalyzer_core::generate_preview_job_from_project_settings(empty, dir.path().into())
    {
        check.problems.push(format!("a run refuses it: {e}"));
    }
    let Some(image) = image else {
        return check;
    };
    if !check.problems.is_empty() {
        return check;
    }
    if let Some(command) = ai_command(&settings) {
        check.not_run = Some(format!("needs an AI model ({command})"));
        return check;
    }

    // A preview on the image.
    let name = image.file_name().unwrap_or_else(|| "image.tif".as_ref());
    if let Err(e) = std::fs::copy(image, images.join(name)) {
        check
            .problems
            .push(format!("can't copy {}: {e}", image.display()));
        return check;
    }
    settings.images.root = Some(images);
    let mut project = ProjectWithRuntime {
        settings,
        ..Default::default()
    };
    project.scan_image_folder_and_add(backend);
    settings = project.settings;
    // An image with too few channels says nothing about the template.
    if let Err(e) = evanalyzer_core::check_pipeline_channels(&settings) {
        check.not_run = Some(format!("the image hasn't the channels it reads ({e})"));
        return check;
    }
    let preview = backend.start_preview(PreviewRequest {
        settings,
        project_path: dir.path().to_path_buf(),
        viewport: PreviewViewport {
            offset_x: 0.0,
            offset_y: 0.0,
            viewport_width: 256.0,
            viewport_height: 256.0,
            zoom: 1.0,
        },
        breakpoint: None,
        prediction: None,
        max_visible_tiles: None,
        breakpoint_tiles: 1,
        whole_image: false,
    });
    match preview {
        Ok(job) => {
            for _ in job.events().iter() {}
            if let Err(e) = job.wait() {
                check.problems.push(format!("a preview fails: {e:?}"));
            }
        }
        Err(StartPreviewError::Failed(e)) => {
            check.problems.push(format!("a preview refuses it: {e}"));
        }
        Err(e) => check.problems.push(format!("a preview refuses it: {e:?}")),
    }
    check
}

/// The first command of `settings` that needs an AI model, if any.
fn ai_command(settings: &ProjectSettings) -> Option<&'static str> {
    settings
        .pipelines
        .iter()
        .filter(|p| p.enabled)
        .flat_map(|p| p.steps.iter())
        .flat_map(|step| resolved_commands(step, &settings.modules))
        .find_map(|resolved| match resolved.command {
            PipelineCommand::AiObjectClassifier(_)
            | PipelineCommand::Cellpose(_)
            | PipelineCommand::PixelClassifier(_) => Some(resolved.command.key()),
            _ => None,
        })
}

/// A folder of its own for one check, removed with everything in it when
/// dropped.
struct CheckFolder(std::path::PathBuf);

impl CheckFolder {
    fn new() -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let name = format!(
            "evanalyzer-template-check-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for CheckFolder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalBackend;

    fn fixture_image() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../core/tests/multi-channel-4D-series.ome.tif")
    }

    /// A detection module as a library file has it: the Threshold's ID
    /// empty, its class a parameter - `measured` is the second step.
    fn detection_module(measured: serde_json::Value) -> ModuleTemplate {
        let meta =
            serde_json::to_value(evanalyzer_cfg::settings::meta_data::MetaData::default()).unwrap();
        serde_json::from_value(serde_json::json!({
            "schemaVersion": 4,
                "key": "EV_DETECTION",
                "meta": meta,
                "steps": [
                    {"id": 1, "enabled": true, "command": {"type": "THRESHOLD", "thresholds": [
                        {"method": {"type": "LI"}, "minThreshold": 0.0,
                         "maxThreshold": 65535.0, "unit": "BIT", "segmentationClass": 1,
                         "valueSource": "ACTUAL_IMAGE"}]}},
                    {"id": 2, "enabled": true, "command": measured},
                    {"id": 3, "enabled": true, "command": {"type": "CONNECTED_COMPONENTS", "minSize": 1}},
                    {"id": 4, "enabled": true, "command": {"type": "EXTRACT_OBJECTS",
                        "maxObjectsBeforeFail": 1000000,
                        "classMapping": [{"segmentationClass": 1, "objectClass": {"GLOBAL": 1}}]}}
                ],
                "parameters": [{"key": "evs", "label": "EVs", "default": "1",
                    "bindings": [{"step": 4, "param": "class_mapping.0.object_class"}]}],
            "uses": []
        }))
        .unwrap()
    }

    /// A module with a Threshold, used twice and inside another module,
    /// runs - what failed before measurements were named per use.
    #[test]
    fn a_module_with_a_threshold_is_usable_twice_and_nested() {
        let template = detection_module(serde_json::json!({"type": "BLUR", "kernelSize": 3}));
        let check = check_module_template(
            &template,
            "ev.evamodule",
            Some(&fixture_image()),
            &LocalBackend::default(),
        );
        assert_eq!(check, TemplateCheck::default());
    }

    /// A module whose two measurement steps have the same ID isn't usable.
    #[test]
    fn a_module_with_one_measurement_id_twice_is_not_usable() {
        let template = detection_module(
            serde_json::json!({"type": "THRESHOLD", "id": "threshold_01",
            "thresholds": [{"method": {"type": "MANUAL"}, "minThreshold": 100.0,
                "maxThreshold": 65535.0, "unit": "BIT", "segmentationClass": 1,
                "valueSource": "ACTUAL_IMAGE"}]}),
        );
        let mut template = template;
        // The first Threshold under the same ID as the second.
        if let Some(PipelineCommand::Threshold(t)) = template.module.steps[0].command_mut() {
            t.id = "threshold_01".into();
        }
        let check =
            check_module_template(&template, "ev.evamodule", None, &LocalBackend::default());
        assert!(
            check.problems.iter().any(|p| p.contains("more than once")),
            "{check:?}"
        );
    }
}
