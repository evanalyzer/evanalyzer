// app/src/project_owner.rs

use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass, ObjectId},
    settings::{object_settings::ObjectMetricSettings, project_settings::ProjectSettings},
};
use evanalyzer_core::{ImageReader, ReadMode, recommended_reader_pool_size};
use std::collections::HashSet;
use std::{
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::extensions::project_ext::ProjectExt;

/// ProjectTmpSettings transient, never serialised
/// Lives alongside ProjectSettings but owned by evanalyzer_app.
#[derive(Debug)]
pub struct ProjectTmpSettings {
    /// Path of the currently open project file (None = unsaved).
    pub current_project: Option<PathBuf>,
    /// Absolute path of the image active in the viewport.
    pub current_image: Option<PathBuf>,
    pub selected_object_class: ObjectClass,

    /// The regions of interest from the preview run
    pub preview_objects: Vec<ObjectMetricSettings>,

    pub selected_object: Option<ObjectId>,

    /// Class IDs currently hidden from the viewport overlay.
    pub hidden_classes: HashSet<ObjectClass>,

    /// Hide ROIs that carry no object class at all (shown in red as
    /// "Unclassified" otherwise). Defaults to hidden - these are usually
    /// pipeline output the user hasn't classified yet, not something worth
    /// cluttering the list/viewport with by default.
    pub hide_unclassified_objects: bool,
}

impl Default for ProjectTmpSettings {
    fn default() -> Self {
        Self {
            current_project: None,
            current_image: None,
            selected_object_class: ObjectClass::default(),
            preview_objects: Vec::new(),
            selected_object: None,
            hidden_classes: HashSet::new(),
            hide_unclassified_objects: true,
        }
    }
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
            reader_pool: Arc::new(Mutex::new(None)),
        }
    }

    /// Loads a project from disk replacing the current project.
    pub fn load_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        let project = crate::extensions::project_ext::load_project(path)?;
        *self.project.write().expect("Poisoned") = project;
        *self.current_path.lock().unwrap() = Some(path.clone());
        Ok(())
    }

    /// Saves the current project to disk.
    pub fn save_project(&self, path: &PathBuf) -> Result<(), InternalErrors> {
        let mut project = self.project.write().expect("Poisoned");
        project.settings.schema_version = evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION;
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

/// A pool of independent readers open on the same image path (sized by
/// [`recommended_reader_pool_size`], which balances cores against available
/// RAM), so different channels/Z-slices can be read truly in parallel
/// instead of serializing through one `IFormatReader` (see
/// `BioFormatsWrapper.java`'s `synchronized(formatReader)` block - one
/// reader is safe, not concurrent).
///
/// Reader #1 pays the full `setId()` cost (BioFormats parses the file's
/// structure). The BioFormats wrapper wraps every reader in a `Memoizer`,
/// which caches that parsed structure to disk (a `.bfmemo` file next to the
/// image); every reader built after #1 reads that cache instead of
/// re-parsing, so building the rest of the pool is cheap.
pub struct ReaderPool {
    path: PathBuf,
    readers: Vec<Arc<ImageReader>>,
}

impl ReaderPool {
    pub fn readers(&self) -> &[Arc<ImageReader>] {
        &self.readers
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

    /// Per-handle reader pool cache, for callers that need to read multiple
    /// channels/Z-slices in parallel (see `ReaderPool`).
    reader_pool: Arc<Mutex<Option<Arc<ReaderPool>>>>,
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
        let project = crate::extensions::project_ext::load_project(path)?;
        *self.project.write().expect("Poisoned") = project;
        Ok(())
    }

    /// Imports an old (`.icproj`) project, replacing the current project.
    ///
    /// Returns the conversion warnings and the legacy project's configured
    /// image folder (if any), so the caller can surface the warnings and decide
    /// whether to scan that folder for images.
    pub fn import_legacy_project(
        &self,
        path: &PathBuf,
    ) -> Result<(Vec<String>, Option<String>), InternalErrors> {
        let (project, warnings, legacy_image_folder) =
            crate::extensions::project_ext::import_legacy_project(path).map_err(|e| {
                InternalErrors::Internal(format!("Could not import legacy project: {e}"))
            })?;
        *self.project.write().expect("Poisoned") = project;
        Ok((warnings, legacy_image_folder))
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

    /// Returns or creates a small pool of readers for the given path, for
    /// callers that read multiple channels/Z-slices in parallel (see
    /// [`ReaderPool`]). Reuses the existing pool if the path has not
    /// changed. Independent from [`Self::get_or_create_reader`]'s
    /// single-reader cache - a small amount of redundancy (both may end up
    /// holding an extra reader instance open on the same path) in exchange
    /// for not cross-wiring the two caches.
    pub fn get_or_create_reader_pool(
        &self,
        new_path: &PathBuf,
    ) -> Result<Arc<ReaderPool>, InternalErrors> {
        let mut pool_lock = self.reader_pool.lock().unwrap();
        if let Some(ref pool) = *pool_lock {
            if &pool.path == new_path {
                return Ok(Arc::clone(pool));
            }
        }

        let size = recommended_reader_pool_size();
        let mut readers = Vec::with_capacity(size);
        // Built sequentially and in this order: the first call pays the
        // full BioFormats parse cost and populates the Memoizer cache, so
        // every reader after it is cheap to construct (see `ReaderPool`'s
        // docs). Building them concurrently would risk every reader racing
        // to populate that cache at once, for no benefit.
        for _ in 0..size {
            readers.push(Arc::new(ImageReader::new(
                new_path,
                ReadMode::SplitChannels,
            )?));
        }

        let pool = Arc::new(ReaderPool {
            path: new_path.clone(),
            readers,
        });
        *pool_lock = Some(Arc::clone(&pool));
        Ok(pool)
    }
}

impl ProjectWithRuntime {
    pub fn get_preview_objects(&self) -> &Vec<ObjectMetricSettings> {
        return &self.tmp_settings.preview_objects;
    }

    pub fn set_selected_object(&mut self, object_id: Option<ObjectId>) {
        self.tmp_settings.selected_object = object_id;
    }

    pub fn get_selected_object_id(&self) -> Option<ObjectId> {
        return self.tmp_settings.selected_object.clone();
    }

    pub fn get_selected_object(&self) -> Option<ObjectMetricSettings> {
        if let Some(series) = self.get_selected_image_series() {
            let id = self.get_selected_object_id();
            if let Some(id_some) = id {
                let found = series
                    .objects
                    .iter()
                    .chain(self.get_preview_objects())
                    .find(|object| object.id == id_some)
                    .cloned();
                return found;
            } else {
                return self.get_selected_preview_object_from();
            };
        }
        return self.get_selected_preview_object_from();
    }

    pub fn get_selected_preview_object_from(&self) -> Option<ObjectMetricSettings> {
        let id = self.get_selected_object_id();
        if let Some(id_some) = id {
            let found = self
                .tmp_settings
                .preview_objects
                .iter()
                .find(|object| object.id == id_some)
                .cloned();
            return found;
        } else {
            return None;
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// Adds a single image (one series, no channels) to `owner`'s project and
    /// selects it, mirroring what `ProjectExt::add_image_to_list` +
    /// `set_current_image_path` would do for a real scanned image - built by
    /// hand here since that needs a real `ImageMeta` from actually reading a
    /// file off disk.
    fn seed_one_image(owner: &ProjectOwner, rel_path: &str) -> PathBuf {
        let handle = owner.handle();
        let mut project = handle.get_project_write();
        let path = PathBuf::from(rel_path);
        project.images.list.insert(
            path.clone(),
            ImageEntry {
                rel_path: path.clone(),
                file_size: 0,
                selected_series: 0,
                series: BTreeMap::from([(0, SeriesSettings::default())]),
            },
        );
        drop(project);
        handle.get_project_write().set_current_image_path(&path);
        path
    }

    #[test]
    fn current_path_is_none_for_a_fresh_owner() {
        let owner = ProjectOwner::new();
        assert_eq!(owner.current_path(), None);
    }

    #[test]
    fn save_project_writes_json_and_records_current_path() {
        let owner = ProjectOwner::new();
        let handle = owner.handle();
        handle.get_project_write().metadata.name = "My Project".into();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.evaproj");

        owner.save_project(&path).expect("save should succeed");

        assert_eq!(owner.current_path(), Some(path.clone()));
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("My Project"));
    }

    #[test]
    fn load_project_round_trips_what_save_project_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.evaproj");

        // Write with one owner...
        let writer = ProjectOwner::new();
        writer.handle().get_project_write().metadata.name = "Round Trip".into();
        writer.save_project(&path).unwrap();

        // ...and read back with a fresh one, as a real app restart would.
        let reader = ProjectOwner::new();
        reader.load_project(&path).expect("load should succeed");

        assert_eq!(reader.current_path(), Some(path));
        assert_eq!(reader.handle().get_project().metadata.name, "Round Trip");
    }

    #[test]
    fn load_project_errors_for_a_missing_file() {
        let owner = ProjectOwner::new();
        let missing = Path::new("/nonexistent/does-not-exist.evaproj").to_path_buf();
        assert!(owner.load_project(&missing).is_err());
        // A failed load must not clobber `current_path`.
        assert_eq!(owner.current_path(), None);
    }

    #[test]
    fn get_preview_objects_reflects_tmp_settings() {
        let owner = ProjectOwner::new();
        let handle = owner.handle();
        {
            let mut project = handle.get_project_write();
            project
                .tmp_settings
                .preview_objects
                .push(ObjectMetricSettings {
                    id: ObjectId(7),
                    ..Default::default()
                });
        }
        let project = handle.get_project();
        assert_eq!(project.get_preview_objects().len(), 1);
        assert_eq!(project.get_preview_objects()[0].id, ObjectId(7));
    }

    #[test]
    fn selected_object_prefers_the_series_object_over_a_same_id_preview_object() {
        let owner = ProjectOwner::new();
        let path = seed_one_image(&owner, "img.tif");
        let handle = owner.handle();
        {
            let mut project = handle.get_project_write();
            project.set_current_image_path(&path);
            project.add_object(&ObjectMetricSettings {
                id: ObjectId(1),
                area: 100,
                ..Default::default()
            });
            project
                .tmp_settings
                .preview_objects
                .push(ObjectMetricSettings {
                    id: ObjectId(1),
                    area: 999,
                    ..Default::default()
                });
            project.set_selected_object(Some(ObjectId(1)));
        }
        let project = handle.get_project();
        let selected = project.get_selected_object().expect("a object is selected");
        assert_eq!(
            selected.area, 100,
            "the real series object should win, not the preview one"
        );
    }

    #[test]
    fn selected_object_falls_back_to_a_preview_object_when_no_series_object_matches() {
        let owner = ProjectOwner::new();
        let handle = owner.handle();
        {
            let mut project = handle.get_project_write();
            project
                .tmp_settings
                .preview_objects
                .push(ObjectMetricSettings {
                    id: ObjectId(42),
                    area: 5,
                    ..Default::default()
                });
            project.set_selected_object(Some(ObjectId(42)));
        }
        let project = handle.get_project();
        let selected = project
            .get_selected_object()
            .expect("preview object should be found");
        assert_eq!(selected.area, 5);
    }

    #[test]
    fn selected_object_is_none_when_nothing_is_selected() {
        let owner = ProjectOwner::new();
        let handle = owner.handle();
        let project = handle.get_project();
        assert_eq!(project.get_selected_object_id(), None);
        assert!(project.get_selected_object().is_none());
    }
}
