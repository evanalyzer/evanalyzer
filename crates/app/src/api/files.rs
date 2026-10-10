//! File access on the machine the backend runs on - the file browser, and
//! every project/template/model file the UI reads or writes, goes through
//! this instead of `std::fs`, so it works the same against a remote server.

use evanalyzer_cfg::core_types::InternalErrors;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub trait FileSystem: Send + Sync {
    /// Starting points for browsing: the home folder and drives/filesystem
    /// root - or, on a restricted server, exactly its allowed folders.
    fn places(&self) -> Result<Vec<Place>, InternalErrors>;

    /// Entries of `dir`, folders first, then by name (case-insensitive).
    fn list_dir(&self, dir: &Path) -> Result<Vec<DirEntry>, InternalErrors>;

    /// `None` if nothing exists at `path`.
    fn stat(&self, path: &Path) -> Result<Option<DirEntry>, InternalErrors>;

    fn read_file(&self, path: &Path) -> Result<Vec<u8>, InternalErrors>;

    /// Writes atomically (temp file + rename), so a crash mid-write never
    /// leaves a truncated project file behind. Creates missing parent folders.
    fn write_file(&self, path: &Path, data: &[u8]) -> Result<(), InternalErrors>;

    fn create_dir_all(&self, path: &Path) -> Result<(), InternalErrors>;

    /// Moves a file or folder (both paths on the same machine).
    fn rename(&self, from: &Path, to: &Path) -> Result<(), InternalErrors>;

    /// Deletes a file, or a folder with everything in it. A missing path is
    /// not an error.
    fn remove_all(&self, path: &Path) -> Result<(), InternalErrors>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    /// Bytes; 0 for folders.
    pub size: u64,
    /// Last modification, seconds since the Unix epoch.
    pub modified: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaceKind {
    Home,
    Drive,
    Folder,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place {
    pub name: String,
    pub path: PathBuf,
    pub kind: PlaceKind,
}
