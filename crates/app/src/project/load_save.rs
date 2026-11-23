//! # load_save
//!
//! A new project, loading and saving it, saving it as a template and
//! starting from one.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::files::FileSystem;
use crate::project::ProjectTmpSettings;
use crate::project::ProjectWithRuntime;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::meta_data::MetaData;
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use evanalyzer_cfg::settings::templates::ProjectTemplate;
use evanalyzer_cfg::{
    CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION, PROJECT_FILE_EXTENSIONS,
    PROJECT_FILE_TEMPLATE_EXTENSIONS,
};
use log::{info, warn};
use std::path::{Path, PathBuf};
use std::sync::Arc;

impl ProjectWithRuntime {
    pub fn save_project(&mut self, files: &dyn FileSystem) -> SaveProjectActions {
        // 1. Clone the path to release the borrow on 'self' immediately
        let Some(path) = self.tmp_settings.current_project.clone() else {
            return SaveProjectActions::PleaseSelectFile;
        };

        // 2. Now 'self' is free to be borrowed mutably by 'save_project_as'
        match self.save_project_as(files, &path) {
            Ok(_) => SaveProjectActions::Success,
            Err(_) => SaveProjectActions::Error,
        }
    }

    pub fn save_project_as(
        &mut self,
        files: &dyn FileSystem,
        path: &PathBuf,
    ) -> Result<(), InternalErrors> {
        let mut final_path = path.clone();

        // Check if the extension matches; if not, set it to evaproj
        if final_path.extension().and_then(|s| s.to_str()) != Some(PROJECT_FILE_EXTENSIONS) {
            final_path.set_extension(PROJECT_FILE_EXTENSIONS);
        }

        // Every file written by this build is stamped with the current
        // format version, regardless of what version it was loaded at (or
        // whether it was ever loaded from disk at all - e.g. a brand-new
        // project still has `schema_version: 0` from `Default`).
        self.settings.schema_version = evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION;
        self.settings.meta.app_version = env!("CARGO_PKG_VERSION").to_string();
        self.forget_hidden_classes_not_defined();

        // Every command's file-path field (AI model paths today - see
        // `relativize_file_paths`'s doc comment for why this isn't limited to
        // those specifically) is stored relative to the project's own
        // directory on disk so the project keeps working after the folder is
        // moved or copied elsewhere - but only in the serialized copy; the
        // in-memory settings keep absolute paths so the rest of the app
        // (pipeline preview, the model-info dialog, ...) doesn't need to
        // know about the project directory to resolve them.
        let mut on_disk_settings = self.settings.clone();
        if let Some(project_dir) = final_path.parent() {
            crate::project::utils::relativize_file_paths(
                &mut on_disk_settings.pipelines,
                project_dir,
            );
            crate::project::utils::relativize_module_file_paths(
                &mut on_disk_settings.modules,
                project_dir,
            );
        }

        let json = serde_json::to_string_pretty(&on_disk_settings)
            .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
        files.write_file(&final_path, json.as_bytes())?;

        let _ = self.tmp_settings.current_project.insert(final_path);
        Ok(())
    }

    /// Stores the actual project as template project
    pub fn save_project_as_template(
        &mut self,
        files: &dyn FileSystem,
        mut meta: MetaData,
        path: &PathBuf,
    ) -> Result<(), InternalErrors> {
        meta.app_version = env!("CARGO_PKG_VERSION").to_string();
        let template = ProjectTemplate {
            schema_version: CURRENT_PROJECT_TEMPLATE_SCHEMA_VERSION,
            meta,
            classification: self.classification.clone(),
            plate: self.plate.clone(),
            pipelines: self.pipelines.clone(),
            modules: self.modules.clone(),
        };

        let mut final_path = path.clone();
        if final_path.extension().and_then(|s| s.to_str()) != Some(PROJECT_FILE_TEMPLATE_EXTENSIONS)
        {
            final_path.set_extension(PROJECT_FILE_TEMPLATE_EXTENSIONS);
        }

        let json = serde_json::to_string_pretty(&template)
            .map_err(|e| InternalErrors::ParseError(e.to_string()))?;
        files.write_file(&final_path, json.as_bytes())?;
        Ok(())
    }

    /// Creates a new, empty project instance wrapped in an [`Arc`].
    ///
    /// This initializes a "blank slate" project with a `None` root and an empty
    /// image list. The resulting project is thread-safe and ready to be shared
    /// between the UI and background processing threads.
    ///
    /// # Returns
    /// An `Arc<Project>` containing default-initialized synchronization primitives
    /// (Mutexes/RwLocks) for immediate use in a multi-threaded context.
    pub fn new(&self) -> Arc<ProjectWithRuntime> {
        Arc::new(ProjectWithRuntime::default())
    }

    pub fn new_project(
        &mut self,
        files: &dyn FileSystem,
        path: &PathBuf,
    ) -> Result<ProjectWithRuntime, InternalErrors> {
        let mut project = ProjectWithRuntime::default();
        project.save_project_as(files, path)?;
        Ok(project)
    }

    /// Replaces the classification, plate and pipeline settings of this project
    /// with the ones from `template`. The image list and project path are kept.
    pub fn apply_project_template(&mut self, template: &ProjectTemplate) {
        self.classification = template.classification.clone();
        self.plate = template.plate.clone();
        self.pipelines = template
            .pipelines
            .iter()
            .cloned()
            .map(|mut pipeline| {
                super::utils::assign_step_ids(&mut pipeline.steps);
                pipeline
            })
            .collect();
        self.modules = template.modules.clone();
        for module in &mut self.modules {
            if let Err(e) = super::utils::assign_module_step_ids(module) {
                warn!("Project template {}: {e}", template.meta.name);
            }
        }
        super::utils::assign_measurement_ids(self);
    }
}

#[derive(Debug, PartialEq)]
pub enum SaveProjectActions {
    /// Everything went fine, image added.
    Success,

    /// No project file yet selected
    PleaseSelectFile,

    Error,
}

/// Reads a project file through `files` (the backend's - the server's in
/// remote mode).
pub fn load_project(
    files: &dyn FileSystem,
    path: &PathBuf,
) -> Result<ProjectWithRuntime, InternalErrors> {
    let data = read_text(files, path)?;
    let mut inner: ProjectSettings = evanalyzer_cfg::load_project_settings_from_str(&data)?;

    // A module definition's steps are a list of their own (see
    // `assign_module_step_ids`).
    for module in &mut inner.modules {
        super::utils::assign_module_step_ids(module)?;
    }
    // Steps written without an id (or copied within a hand-written file,
    // sharing one) get the next number of their pipeline.
    for pipeline in &mut inner.pipelines {
        let assigned = super::utils::assign_step_ids(&mut pipeline.steps);
        if assigned > 0 {
            info!(
                "Pipeline {}: {assigned} step(s) without an id of their own got one",
                pipeline.id
            );
        }
    }
    // Measurement steps written without a name get one.
    super::utils::assign_measurement_ids(&mut inner);

    // Reverse of the relativization `save_project_as` does on write - resolve
    // AI command `model_path`s back to absolute now, using this project
    // file's own directory, so the rest of the app can keep treating them as
    // plain absolute paths.
    if let Some(project_dir) = path.parent() {
        crate::project::utils::resolve_file_paths(&mut inner.pipelines, project_dir);
        crate::project::utils::resolve_module_file_paths(&mut inner.modules, project_dir);
    }

    // Values for parameters a module no longer has (e.g. a hand edit).
    crate::modules::editing::prune_module_values(&mut inner);

    // Module mistakes show in the editor too; here they're logged.
    for problem in crate::modules::checks::check_project(&inner) {
        warn!("{}", problem.message);
    }

    let mut project = ProjectWithRuntime {
        settings: inner,
        tmp_settings: ProjectTmpSettings::default(),
    };
    project.tmp_settings.current_project = Some(path.clone());
    Ok(project)
}

pub(super) fn read_text(files: &dyn FileSystem, path: &Path) -> Result<String, InternalErrors> {
    String::from_utf8(files.read_file(path)?).map_err(|e| {
        InternalErrors::ParseError(format!("'{}' is not a text file: {e}", path.display()))
    })
}
