//! File access on the machine the backend runs on - the file browser, and
//! every project/template/model file the UI reads or writes, goes through
//! this instead of `std::fs`, so it works the same against a remote server.

use evanalyzer_cfg::core_types::InternalErrors;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

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

/// `std::fs`-backed file access, optionally confined to a set of folders
/// (what a server exposes to its clients).
#[derive(Debug, Default, Clone)]
pub struct LocalFileSystem {
    /// Canonicalized allowed folders, shown as places; `None` = unrestricted.
    roots: Option<Vec<PathBuf>>,
    /// Further allowed folders that aren't shown as places (e.g. the
    /// templates folders a server keeps reachable for its clients).
    hidden_roots: Vec<PathBuf>,
}

impl LocalFileSystem {
    /// Only paths inside `roots` (after resolving `..` and symlinks) are
    /// accessible. Fails if a root doesn't exist.
    pub fn restricted_to(roots: &[PathBuf]) -> Result<Self, InternalErrors> {
        Ok(Self {
            roots: Some(canonical_roots(roots)?),
            hidden_roots: Vec::new(),
        })
    }

    /// Additionally allows `folders` without listing them as places.
    pub fn also_allowing(mut self, folders: &[PathBuf]) -> Result<Self, InternalErrors> {
        self.hidden_roots.extend(canonical_roots(folders)?);
        Ok(self)
    }

    pub fn is_restricted(&self) -> bool {
        self.roots.is_some()
    }

    /// Returns `path` if it's accessible under the current restriction.
    /// Paths that don't exist yet (a file about to be saved) are checked via
    /// their nearest existing ancestor, so `..` can't escape a root.
    pub fn check(&self, path: &Path) -> Result<PathBuf, InternalErrors> {
        let Some(roots) = &self.roots else {
            return Ok(path.to_path_buf());
        };
        if !path.is_absolute() {
            return Err(denied(path));
        }
        let resolved = resolve_for_check(path).ok_or_else(|| denied(path))?;
        if roots
            .iter()
            .chain(&self.hidden_roots)
            .any(|root| resolved.starts_with(root))
        {
            Ok(path.to_path_buf())
        } else {
            Err(denied(path))
        }
    }
}

fn canonical_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, InternalErrors> {
    roots
        .iter()
        .map(|root| {
            root.canonicalize().map_err(|e| {
                InternalErrors::InvalidArgument(format!(
                    "Allowed folder '{}' is not accessible: {e}",
                    root.display()
                ))
            })
        })
        .collect()
}

fn denied(path: &Path) -> InternalErrors {
    InternalErrors::InvalidArgument(format!(
        "Access to '{}' is not allowed on this server",
        path.display()
    ))
}

/// Canonicalizes the longest existing prefix of `path` and appends the rest,
/// rejecting any `..` in that non-existing rest.
fn resolve_for_check(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        let name = existing.file_name()?.to_os_string();
        rest.push(name);
        existing = existing.parent()?.to_path_buf();
    }
    let mut resolved = existing.canonicalize().ok()?;
    for part in rest.into_iter().rev() {
        let part = PathBuf::from(part);
        if part
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return None;
        }
        resolved.push(part);
    }
    Some(resolved)
}

fn io_error(action: &str, path: &Path, e: std::io::Error) -> InternalErrors {
    InternalErrors::Io(format!("Could not {action} '{}': {e}", path.display()))
}

fn entry_for(path: &Path, metadata: &std::fs::Metadata) -> DirEntry {
    DirEntry {
        name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string()),
        path: path.to_path_buf(),
        is_dir: metadata.is_dir(),
        size: if metadata.is_dir() { 0 } else { metadata.len() },
        modified: metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64),
    }
}

impl FileSystem for LocalFileSystem {
    fn places(&self) -> Result<Vec<Place>, InternalErrors> {
        if let Some(roots) = &self.roots {
            return Ok(roots
                .iter()
                .map(|root| Place {
                    name: root
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| root.display().to_string()),
                    path: root.clone(),
                    kind: PlaceKind::Folder,
                })
                .collect());
        }
        let mut places = Vec::new();
        if let Some(home) = dirs::home_dir() {
            places.push(Place {
                name: "Home".into(),
                path: home,
                kind: PlaceKind::Home,
            });
        }
        for (name, dir) in [
            ("Desktop", dirs::desktop_dir()),
            ("Documents", dirs::document_dir()),
            ("Pictures", dirs::picture_dir()),
        ] {
            if let Some(dir) = dir.filter(|d| d.is_dir()) {
                places.push(Place {
                    name: name.into(),
                    path: dir,
                    kind: PlaceKind::Folder,
                });
            }
        }
        places.extend(drives());
        Ok(places)
    }

    fn list_dir(&self, dir: &Path) -> Result<Vec<DirEntry>, InternalErrors> {
        let dir = self.check(dir)?;
        let mut entries: Vec<DirEntry> = std::fs::read_dir(&dir)
            .map_err(|e| io_error("open folder", &dir, e))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                // Follows symlinks, so a link to a folder lists as a folder;
                // broken links are skipped.
                let metadata = std::fs::metadata(&path).ok()?;
                // A link pointing outside the allowed folders is hidden.
                self.check(&path).ok()?;
                Some(entry_for(&path, &metadata))
            })
            .collect();
        entries.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        Ok(entries)
    }

    fn stat(&self, path: &Path) -> Result<Option<DirEntry>, InternalErrors> {
        let path = self.check(path)?;
        match std::fs::metadata(&path) {
            Ok(metadata) => Ok(Some(entry_for(&path, &metadata))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_error("inspect", &path, e)),
        }
    }

    fn read_file(&self, path: &Path) -> Result<Vec<u8>, InternalErrors> {
        let path = self.check(path)?;
        std::fs::read(&path).map_err(|e| io_error("read", &path, e))
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> Result<(), InternalErrors> {
        let path = self.check(path)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| io_error("create folder", parent, e))?;
        }
        let file_name = path.file_name().ok_or_else(|| {
            InternalErrors::InvalidArgument(format!("'{}' is not a file path", path.display()))
        })?;
        let tmp = path.with_file_name(format!(".{}.tmp", file_name.to_string_lossy()));
        std::fs::write(&tmp, data).map_err(|e| io_error("write", &tmp, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            io_error("replace", &path, e)
        })
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), InternalErrors> {
        let path = self.check(path)?;
        std::fs::create_dir_all(&path).map_err(|e| io_error("create folder", &path, e))
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), InternalErrors> {
        let (from, to) = (self.check(from)?, self.check(to)?);
        std::fs::rename(&from, &to).map_err(|e| io_error("move", &from, e))
    }

    fn remove_all(&self, path: &Path) -> Result<(), InternalErrors> {
        let path = self.check(path)?;
        let result = match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(&path),
            Ok(_) => std::fs::remove_file(&path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => Err(e),
        };
        result.map_err(|e| io_error("delete", &path, e))
    }
}

#[cfg(windows)]
fn drives() -> Vec<Place> {
    (b'A'..=b'Z')
        .map(|letter| format!("{}:\\", letter as char))
        .filter(|root| Path::new(root).exists())
        .map(|root| Place {
            name: root.trim_end_matches('\\').to_string(),
            path: PathBuf::from(root),
            kind: PlaceKind::Drive,
        })
        .collect()
}

#[cfg(not(windows))]
fn drives() -> Vec<Place> {
    vec![Place {
        name: "Computer".into(),
        path: PathBuf::from("/"),
        kind: PlaceKind::Drive,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_dir_puts_folders_first_then_sorts_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.txt"), b"12345").unwrap();
        std::fs::write(dir.path().join("A.txt"), b"").unwrap();
        std::fs::create_dir(dir.path().join("zeta")).unwrap();

        let entries = LocalFileSystem::default().list_dir(dir.path()).unwrap();
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["zeta", "A.txt", "b.txt"]);
        assert!(entries[0].is_dir);
        assert_eq!(entries[2].size, 5);
        assert!(entries[2].modified.is_some());
    }

    #[test]
    fn write_file_creates_parents_and_replaces_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::default();
        let path = dir.path().join("nested/deeper/project.evaproj");
        fs.write_file(&path, b"first").unwrap();
        fs.write_file(&path, b"second").unwrap();
        assert_eq!(fs.read_file(&path).unwrap(), b"second");
        // No temp file is left behind.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(leftovers, ["project.evaproj"]);
    }

    #[test]
    fn rename_moves_and_remove_all_deletes_recursively() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::default();
        fs.write_file(&dir.path().join("scratch/a/b.txt"), b"x")
            .unwrap();
        fs.rename(
            &dir.path().join("scratch/a/b.txt"),
            &dir.path().join("out.txt"),
        )
        .unwrap();
        assert_eq!(fs.read_file(&dir.path().join("out.txt")).unwrap(), b"x");
        fs.remove_all(&dir.path().join("scratch")).unwrap();
        assert!(!dir.path().join("scratch").exists());
        fs.remove_all(&dir.path().join("scratch")).unwrap();
        fs.remove_all(&dir.path().join("out.txt")).unwrap();
        assert!(!dir.path().join("out.txt").exists());
    }

    #[test]
    fn stat_reports_missing_paths_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::default();
        assert!(fs.stat(&dir.path().join("missing")).unwrap().is_none());
        assert!(fs.stat(dir.path()).unwrap().unwrap().is_dir);
    }

    #[test]
    fn a_restricted_file_system_only_reaches_its_roots() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"x").unwrap();
        let fs = LocalFileSystem::restricted_to(&[allowed.path().to_path_buf()]).unwrap();

        fs.write_file(&allowed.path().join("new/file.txt"), b"ok")
            .unwrap();
        assert!(fs.read_file(&outside.path().join("secret.txt")).is_err());
        assert!(fs.list_dir(outside.path()).is_err());
        // `..` can't climb out, neither through existing nor new path parts.
        let escape = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap());
        assert!(fs.list_dir(&escape).is_err());
        assert!(
            fs.write_file(&allowed.path().join("new/../../x.txt"), b"no")
                .is_err()
        );
        assert!(fs.read_file(Path::new("relative.txt")).is_err());

        let places = fs.places().unwrap();
        assert_eq!(places.len(), 1);
        assert_eq!(places[0].path, allowed.path().canonicalize().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_a_root_is_neither_listed_nor_followed() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"x").unwrap();
        std::os::unix::fs::symlink(outside.path(), allowed.path().join("link")).unwrap();
        let fs = LocalFileSystem::restricted_to(&[allowed.path().to_path_buf()]).unwrap();

        assert!(fs.list_dir(allowed.path()).unwrap().is_empty());
        assert!(
            fs.read_file(&allowed.path().join("link/secret.txt"))
                .is_err()
        );
    }

    #[test]
    fn restricting_to_a_missing_folder_fails() {
        assert!(LocalFileSystem::restricted_to(&[PathBuf::from("/no/such/folder")]).is_err());
    }
}
