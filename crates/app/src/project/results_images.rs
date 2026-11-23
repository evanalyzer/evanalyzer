//! # results_images
//!
//! The images a results file names, found in the project - also when its
//! image folder changed since the run.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::files::FileSystem;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The image of `list` (paths relative to `root`) a results file names
/// `results_rel` (relative to `run_root`, the image folder of its run) -
/// see [`ResultsImageFinder::find`].
pub fn find_results_image(
    list: &evanalyzer_cfg::settings::images_settings::ImageList,
    root: Option<&Path>,
    results_rel: &Path,
    run_root: Option<&Path>,
) -> Option<PathBuf> {
    ResultsImageFinder::new(list, root, run_root).find(results_rel)
}

/// Finds the images a results file names in a project's list (paths
/// relative to `root`); the results' paths are relative to `run_root`, the
/// image folder of their run. For many images: the list's file names are
/// indexed once, at the first image not found directly.
pub struct ResultsImageFinder<'a> {
    list: &'a evanalyzer_cfg::settings::images_settings::ImageList,
    root: Option<&'a Path>,
    run_root: Option<&'a Path>,
    by_name: std::cell::OnceCell<HashMap<&'a std::ffi::OsStr, Vec<&'a PathBuf>>>,
}

impl<'a> ResultsImageFinder<'a> {
    pub fn new(
        list: &'a evanalyzer_cfg::settings::images_settings::ImageList,
        root: Option<&'a Path>,
        run_root: Option<&'a Path>,
    ) -> Self {
        Self {
            list,
            root,
            run_root,
            by_name: std::cell::OnceCell::new(),
        }
    }

    /// The project's image of `results_rel`:
    /// 1. the same file, when it lies in the project's folder: the image
    ///    folder moved up or down since the run;
    /// 2. the same path: the folder itself moved (or the run's is unknown);
    /// 3. the image with the same file name whose path ends the same way
    ///    the longest, when only one does: the folder was moved elsewhere.
    pub fn find(&self, results_rel: &Path) -> Option<PathBuf> {
        let list = &self.list.0;
        if let (Some(run_root), Some(root)) = (self.run_root, self.root)
            && let Ok(rel) = run_root.join(results_rel).strip_prefix(root)
            && list.contains_key(rel)
        {
            return Some(rel.to_path_buf());
        }
        if list.contains_key(results_rel) {
            return Some(results_rel.to_path_buf());
        }
        let name = results_rel.file_name()?;
        let by_name = self.by_name.get_or_init(|| {
            let mut by_name: HashMap<&std::ffi::OsStr, Vec<&PathBuf>> = HashMap::new();
            for path in list.keys() {
                if let Some(name) = path.file_name() {
                    by_name.entry(name).or_default().push(path);
                }
            }
            by_name
        });
        let wanted: Vec<_> = results_rel.components().rev().collect();
        let mut best: Option<&PathBuf> = None;
        let (mut best_len, mut ties) = (0, 0);
        for path in by_name.get(name)? {
            let len = path
                .components()
                .rev()
                .zip(&wanted)
                .take_while(|(a, b)| a == *b)
                .count();
            if len > best_len {
                (best, best_len, ties) = (Some(*path), len, 1);
            } else if len == best_len {
                ties += 1;
            }
        }
        best.filter(|_| ties == 1).cloned()
    }
}

/// The image folder the run of the results `results_db` analysed: from the
/// copy of the project each run keeps beside its results
/// (`<name>.evaproj` next to `<name>.evadb`). None without one (or
/// without a folder in it).
pub fn run_image_root(files: &dyn FileSystem, results_db: &Path) -> Option<PathBuf> {
    // Only the folder is read of the copy (all else is skipped).
    #[derive(serde::Deserialize)]
    struct Run {
        images: RunImages,
    }
    #[derive(serde::Deserialize)]
    struct RunImages {
        root: Option<PathBuf>,
    }
    let data = files
        .read_file(&results_db.with_extension("evaproj"))
        .ok()?;
    serde_json::from_slice::<Run>(&data).ok()?.images.root
}
