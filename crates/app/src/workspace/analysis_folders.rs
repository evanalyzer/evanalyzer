//! # analysis_folders
//!
//! An analysis's folder on the disk - `<project>/results/<run>/`, holding
//! its results file (`.evadb`), the snapshot of its settings (`.evaproj`)
//! and its control images (`images/`) - for deleting it completely: what is
//! in it (the question shows it) and the deletion. Through the backend's
//! file system, so the same on a server.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::api::FileSystem;
use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
use std::path::{Path, PathBuf};

/// The folder of the live preview (`results/preview/`): never an analysis.
const PREVIEW_FOLDER: &str = "preview";

/// An analysis's folder and what is in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisFolder {
    /// `<project>/results/<run>/`.
    pub dir: PathBuf,
    /// The analysis's name, as the list shows it (its results file's stem).
    pub name: String,
    /// The results file (with DuckDB's journal beside it).
    pub results_bytes: u64,
    /// The control images (`images/`).
    pub images: usize,
    pub images_bytes: u64,
    /// The snapshot of the settings it ran with (`.evaproj`).
    pub settings_bytes: u64,
    /// Everything else in the folder.
    pub other_bytes: u64,
}

impl AnalysisFolder {
    /// All of it - what deleting frees.
    pub fn total_bytes(&self) -> u64 {
        self.results_bytes + self.images_bytes + self.settings_bytes + self.other_bytes
    }
}

/// The folder of the analysis whose results file is `results_file`, in the
/// project at `project_dir` - and what is in it. Refuses anything that isn't
/// an analysis's own folder: a file other than a results file, one not
/// directly in a folder of `<project_dir>/results/`, the preview's folder.
pub fn analysis_folder(
    files: &dyn FileSystem,
    project_dir: &Path,
    results_file: &Path,
) -> Result<AnalysisFolder, String> {
    let refuse = || {
        format!(
            "{} is not an analysis of this project",
            results_file.display()
        )
    };
    if results_file.extension().and_then(|e| e.to_str()) != Some(RESULTS_FILE_EXTENSION) {
        return Err(refuse());
    }
    let dir = results_file.parent().ok_or_else(refuse)?;
    let folder = dir
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(refuse)?;
    if folder == PREVIEW_FOLDER
        || dir
            .parent()
            .map(Path::components)
            .map(Iterator::collect::<Vec<_>>)
            != Some(project_dir.join("results").components().collect())
    {
        return Err(refuse());
    }
    match files.stat(dir) {
        Ok(Some(entry)) if entry.is_dir => {}
        Ok(_) => return Err(format!("{} doesn't exist (any more)", dir.display())),
        Err(e) => return Err(e.to_string()),
    }
    let mut folder = AnalysisFolder {
        dir: dir.to_path_buf(),
        name: results_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(folder)
            .to_string(),
        results_bytes: 0,
        images: 0,
        images_bytes: 0,
        settings_bytes: 0,
        other_bytes: 0,
    };
    add_contents(files, dir, false, &mut folder)?;
    Ok(folder)
}

/// Adds the files under `dir` to `folder`'s parts; `images`: inside the
/// control images' folder.
fn add_contents(
    files: &dyn FileSystem,
    dir: &Path,
    images: bool,
    folder: &mut AnalysisFolder,
) -> Result<(), String> {
    for entry in files.list_dir(dir).map_err(|e| e.to_string())? {
        if entry.is_dir {
            let images = images || (dir == folder.dir && entry.name == "images");
            add_contents(files, &entry.path, images, folder)?;
            continue;
        }
        let name = entry.name.to_ascii_lowercase();
        if images {
            folder.images += 1;
            folder.images_bytes += entry.size;
        } else if name.ends_with(&format!(".{RESULTS_FILE_EXTENSION}"))
            || name.ends_with(&format!(".{RESULTS_FILE_EXTENSION}.wal"))
        {
            folder.results_bytes += entry.size;
        } else if name.ends_with(".evaproj") {
            folder.settings_bytes += entry.size;
        } else {
            folder.other_bytes += entry.size;
        }
    }
    Ok(())
}

/// Deletes `folder` with everything in it. What can't be deleted (a file
/// another program has open) stays: the error says how many files.
pub fn delete_analysis(files: &dyn FileSystem, folder: &AnalysisFolder) -> Result<(), String> {
    let failed = files.remove_all(&folder.dir).err();
    match files.stat(&folder.dir) {
        Ok(None) => Ok(()),
        _ => {
            let left = count_files(files, &folder.dir);
            Err(match (left, failed) {
                (0, Some(e)) => format!("Couldn't delete the analysis: {e}"),
                (0, None) => "Couldn't delete the analysis's folder".to_string(),
                (1, _) => {
                    "Couldn't delete 1 file - close the program using it and try again".to_string()
                }
                (n, _) => {
                    format!(
                        "Couldn't delete {n} files - close the program using them and try again"
                    )
                }
            })
        }
    }
}

/// The files left under `dir` (0 if it can't be read).
fn count_files(files: &dyn FileSystem, dir: &Path) -> usize {
    files.list_dir(dir).map_or(0, |entries| {
        entries
            .iter()
            .map(|e| match e.is_dir {
                true => count_files(files, &e.path),
                false => 1,
            })
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LocalFileSystem;
    use evanalyzer_cfg::core_types::InternalErrors;

    /// A project with one analysis: results file and journal, settings,
    /// 3 control images in a subfolder, a log - and the preview's folder.
    fn project() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("plate");
        let run = project.join("results").join("20261023T091000Z__serum");
        let write = |path: PathBuf, bytes: usize| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, vec![0u8; bytes]).unwrap();
        };
        write(run.join("serum.evadb"), 1000);
        write(run.join("serum.evadb.wal"), 24);
        write(run.join("serum.evaproj"), 300);
        for i in 0..3 {
            write(run.join("images").join("A01").join(format!("{i}.png")), 50);
        }
        write(run.join("notes.txt"), 7);
        write(
            project
                .join("results")
                .join("preview")
                .join("preview.evadb"),
            10,
        );
        (dir, project, run.join("serum.evadb"))
    }

    #[test]
    fn the_folder_says_what_is_in_it() {
        let (_dir, project, results) = project();
        let files = LocalFileSystem::default();
        let folder = analysis_folder(&files, &project, &results).unwrap();
        assert_eq!(folder.name, "serum");
        assert_eq!(folder.dir, results.parent().unwrap());
        assert_eq!(folder.results_bytes, 1024);
        assert_eq!((folder.images, folder.images_bytes), (3, 150));
        assert_eq!(folder.settings_bytes, 300);
        assert_eq!(folder.other_bytes, 7);
        assert_eq!(folder.total_bytes(), 1481);
    }

    #[test]
    fn deleting_removes_the_whole_folder_and_nothing_else() {
        let (_dir, project, results) = project();
        let files = LocalFileSystem::default();
        let folder = analysis_folder(&files, &project, &results).unwrap();
        delete_analysis(&files, &folder).unwrap();
        assert!(!folder.dir.exists());
        assert!(
            project
                .join("results")
                .join("preview")
                .join("preview.evadb")
                .exists()
        );
    }

    /// Only an analysis's own folder: not the preview's, not a file outside
    /// `results/<run>/`, not another kind of file, not another project's.
    #[test]
    fn anything_but_an_analysis_folder_is_refused() {
        let (dir, project, results) = project();
        let files = LocalFileSystem::default();
        let preview = project
            .join("results")
            .join("preview")
            .join("preview.evadb");
        assert!(analysis_folder(&files, &project, &preview).is_err());
        let settings = results.with_extension("evaproj");
        assert!(analysis_folder(&files, &project, &settings).is_err());
        let deeper = results.parent().unwrap().join("images").join("x.evadb");
        assert!(analysis_folder(&files, &project, &deeper).is_err());
        let in_results = project.join("results").join("loose.evadb");
        assert!(analysis_folder(&files, &project, &in_results).is_err());
        let other_project = dir.path().join("other");
        assert!(analysis_folder(&files, &other_project, &results).is_err());
        let gone = project
            .join("results")
            .join("20200101T000000Z__gone")
            .join("gone.evadb");
        assert!(analysis_folder(&files, &project, &gone).is_err());
    }

    /// The local files - but deleting only removes the files `keep` says
    /// go (folders stay), then fails if `fail`; `stat` fails if
    /// `stat_fails`.
    struct Stubborn {
        inner: LocalFileSystem,
        keep: fn(&Path) -> bool,
        fail: bool,
        stat_fails: bool,
    }

    impl Stubborn {
        fn new(keep: fn(&Path) -> bool, fail: bool) -> Self {
            Self {
                inner: LocalFileSystem::default(),
                keep,
                fail,
                stat_fails: false,
            }
        }

        fn remove_files(&self, dir: &Path) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    self.remove_files(&path);
                } else if !(self.keep)(&path) {
                    std::fs::remove_file(&path).unwrap();
                }
            }
        }
    }

    impl FileSystem for Stubborn {
        fn places(&self) -> Result<Vec<crate::api::Place>, InternalErrors> {
            self.inner.places()
        }
        fn list_dir(&self, dir: &Path) -> Result<Vec<crate::api::DirEntry>, InternalErrors> {
            self.inner.list_dir(dir)
        }
        fn stat(&self, path: &Path) -> Result<Option<crate::api::DirEntry>, InternalErrors> {
            if self.stat_fails {
                return Err(InternalErrors::Io("stat refused".into()));
            }
            self.inner.stat(path)
        }
        fn read_file(&self, path: &Path) -> Result<Vec<u8>, InternalErrors> {
            self.inner.read_file(path)
        }
        fn write_file(&self, path: &Path, data: &[u8]) -> Result<(), InternalErrors> {
            self.inner.write_file(path, data)
        }
        fn create_dir_all(&self, path: &Path) -> Result<(), InternalErrors> {
            self.inner.create_dir_all(path)
        }
        fn rename(&self, from: &Path, to: &Path) -> Result<(), InternalErrors> {
            self.inner.rename(from, to)
        }
        fn remove_all(&self, path: &Path) -> Result<(), InternalErrors> {
            self.remove_files(path);
            match self.fail {
                true => Err(InternalErrors::Io("access denied".into())),
                false => Ok(()),
            }
        }
    }

    fn delete_with(files: &Stubborn) -> Result<(), String> {
        let (_dir, project, results) = project();
        let folder = analysis_folder(&files.inner, &project, &results).unwrap();
        delete_analysis(files, &folder)
    }

    /// Files left behind (counted in subfolders too) are named in the
    /// error: one, or how many.
    #[test]
    fn files_left_behind_are_counted_in_the_error() {
        let png = |path: &Path| path.extension().is_some_and(|e| e == "png");
        let notes = |path: &Path| path.ends_with("notes.txt");

        assert_eq!(
            delete_with(&Stubborn::new(png, true)).unwrap_err(),
            "Couldn't delete 3 files - close the program using them and try again",
            "the 3 control images, in images/A01/"
        );
        assert_eq!(
            delete_with(&Stubborn::new(notes, false)).unwrap_err(),
            "Couldn't delete 1 file - close the program using it and try again"
        );
    }

    /// Every file went but the folder stayed: the error says why, if the
    /// file system said - else that the folder stayed.
    #[test]
    fn a_folder_left_behind_without_files_is_reported() {
        let nothing = |_: &Path| false;

        let with_reason = delete_with(&Stubborn::new(nothing, true)).unwrap_err();
        let without = delete_with(&Stubborn::new(nothing, false)).unwrap_err();

        assert!(
            with_reason.starts_with("Couldn't delete the analysis: ")
                && with_reason.contains("access denied"),
            "{with_reason}"
        );
        assert_eq!(without, "Couldn't delete the analysis's folder");
    }

    /// The folder can't be looked at: the file system's error.
    #[test]
    fn a_folder_that_cant_be_looked_at_is_an_error() {
        let (_dir, project, results) = project();
        let mut files = Stubborn::new(|_| false, false);
        files.stat_fails = true;

        let error = analysis_folder(&files, &project, &results).unwrap_err();

        assert!(error.contains("stat refused"), "{error}");
    }
}
