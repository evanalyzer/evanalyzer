// app/src/project_owner.rs

use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass, ObjectId},
    settings::{project_settings::ProjectSettings, roi_settings::RoiSettings},
};
use evanalyzer_core::{ImageReader, ReadMode};
use std::collections::HashSet;
use std::{
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::extensions::project_ext::ProjectExt;

/// ProjectTmpSettings transient, never serialised
/// Lives alongside ProjectSettings but owned by evanalyzer_app.
#[derive(Debug, Default)]
pub struct ProjectTmpSettings {
    /// Path of the currently open project file (None = unsaved).
    pub current_project: Option<PathBuf>,
    /// Absolute path of the image active in the viewport.
    pub current_image: Option<PathBuf>,
    pub selected_object_class: ObjectClass,

    /// The regions of interest from the preview run
    pub preview_rois: Vec<RoiSettings>,

    pub selected_roi: Option<ObjectId>,

    /// Class IDs currently hidden from the viewport overlay.
    pub hidden_classes: HashSet<ObjectClass>,
}

/// ProjectWithRuntime pairs serialisable settings with runtime state.
/// Derefs to ProjectSettings so all existing field access still works.
pub struct ProjectWithRuntime {
    pub settings: ProjectSettings,
    pub tmp_settings: ProjectTmpSettings,
}

impl Default for ProjectWithRuntime {
    fn default() -> Self {
        Self {
            settings: ProjectSettings::default(),
            tmp_settings: ProjectTmpSettings::default(),
        }
    }
}

impl Deref for ProjectWithRuntime {
    type Target = ProjectSettings;
    fn deref(&self) -> &ProjectSettings {
        &self.settings
    }
}

impl DerefMut for ProjectWithRuntime {
    fn deref_mut(&mut self) -> &mut ProjectSettings {
        &mut self.settings
    }
}

/// ProjectOwner single writer, owns all resources
/// Not Clone there is only ever one owner
pub struct ProjectOwner {
    /// The shared project - handed out to GUI/CLI via AppHandle
    project: Arc<RwLock<ProjectWithRuntime>>,

    /// Current project file path - None if unsaved
    current_path: Mutex<Option<PathBuf>>,
}

impl ProjectOwner {
    pub fn new() -> Self {
        Self {
            project: Arc::new(RwLock::new(ProjectWithRuntime::default())),
            current_path: Mutex::new(None),
        }
    }

    /// Returns a lightweight cloneable handle for GUI/CLI.
    pub fn handle(&self) -> AppHandle {
        AppHandle {
            project: Arc::clone(&self.project),
            reader: Arc::new(Mutex::new(None)),
        }
    }

    /// Loads a project from disk replacing the current project.
    pub fn load_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        let project = crate::extensions::project_ext::load_project(path)
            .map_err(|_| InternalErrors::Internal("Could not open project".into()))?;
        *self.project.write().expect("Poisoned") = project;
        *self.current_path.lock().unwrap() = Some(path.clone());
        Ok(())
    }

    /// Saves the current project to disk.
    pub fn save_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        let project = self.project.read().expect("Poisoned");
        let content = serde_json::to_string_pretty(&project.settings)
            .map_err(|e| InternalErrors::Internal(e.to_string()))?;
        std::fs::write(path, content).map_err(|e| InternalErrors::Internal(e.to_string()))?;
        *self.current_path.lock().unwrap() = Some(path.clone());
        Ok(())
    }

    /// Returns the current project file path if saved.
    pub fn current_path(&self) -> Option<PathBuf> {
        self.current_path.lock().unwrap().clone()
    }
}

/// AppHandle lightweight, cloneable, handed to GUI/CLI
/// Only exposes what GUI/CLI need no pipeline, no owner concerns
#[derive(Clone)]
pub struct AppHandle {
    /// Shared reference to the project - same Arc as ProjectOwner
    project: Arc<RwLock<ProjectWithRuntime>>,

    /// Per-handle image reader cache
    reader: Arc<Mutex<Option<Arc<ImageReader>>>>,
}

impl AppHandle {
    /// Acquire a read guard.
    /// Multiple threads can hold read guards concurrently.
    /// Drop the guard before calling `get_project_write`.
    pub fn get_project(&self) -> RwLockReadGuard<'_, ProjectWithRuntime> {
        self.project.read().expect("Poisoned")
    }

    /// Acquire a write guard.
    /// Exclusive - blocks all readers until dropped.
    /// Never hold a read guard on the same thread when calling this.
    pub fn get_project_write(&self) -> RwLockWriteGuard<'_, ProjectWithRuntime> {
        self.project.write().expect("Poisoned")
    }

    /// Loads a project from disk replacing the current project.
    pub fn load_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        let project = crate::extensions::project_ext::load_project(path)
            .map_err(|_| InternalErrors::Internal("Could not open project".into()))?;
        *self.project.write().expect("Poisoned") = project;
        Ok(())
    }

    /// Returns or creates an image reader for the given path.
    /// Reuses the existing reader if the path has not changed.
    pub fn get_or_create_reader(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<ImageReader>, InternalErrors> {
        let mut reader_lock = self.reader.lock().unwrap();
        if let Some(ref r) = *reader_lock {
            if r.get_current_image_path() == new_path {
                return Ok(Arc::clone(r));
            }
        }
        let new_reader = Arc::new(ImageReader::new(new_path, ReadMode::SplitChannels)?);
        *reader_lock = Some(Arc::clone(&new_reader));
        Ok(new_reader)
    }
}

impl ProjectWithRuntime {
    pub fn get_preview_rois(&self) -> &Vec<RoiSettings> {
        return &self.tmp_settings.preview_rois;
    }

    pub fn set_selected_roi(&mut self, roi_id: Option<ObjectId>) {
        self.tmp_settings.selected_roi = roi_id;
    }

    pub fn get_selected_roi_id(&self) -> Option<ObjectId> {
        return self.tmp_settings.selected_roi.clone();
    }

    pub fn get_selected_roi(&self) -> Option<RoiSettings> {
        if let Some(series) = self.get_selected_image_series() {
            let id = self.get_selected_roi_id();
            if let Some(id_some) = id {
                let found = series
                    .rois
                    .iter()
                    .chain(self.get_preview_rois())
                    .find(|roi| roi.id == id_some)
                    .cloned();
                return found;
            } else {
                return self.get_selected_preview_roi_from();
            };
        }
        return self.get_selected_preview_roi_from();
    }

    pub fn get_selected_preview_roi_from(&self) -> Option<RoiSettings> {
        let id = self.get_selected_roi_id();
        if let Some(id_some) = id {
            let found = self
                .tmp_settings
                .preview_rois
                .iter()
                .find(|roi| roi.id == id_some)
                .cloned();
            return found;
        } else {
            return None;
        };
    }
}
