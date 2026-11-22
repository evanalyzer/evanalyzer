//! # image_folder
//!
//! The image folder: changing it, scanning it, adding images and their
//! metadata, finding an image by its path.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::results_images::*;

use crate::backend::Backend;
use crate::files::FileSystem;
use crate::images::ImageMeta;
use crate::images::image_list::ImageListExt;
use crate::project::ProjectWithRuntime;
use crate::project::utils::{get_relative_key, is_in_root};
use evanalyzer_cfg::settings::images_settings::SeriesSettings;
use evanalyzer_cfg::settings::images_settings::{ChannelSettings, ImageReview, PixelSizeSettings};
use evanalyzer_cfg::settings::images_settings::{ImageEntry, ImageSettings};
use human_sort::compare;
use log::{info, trace, warn};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

impl ProjectWithRuntime {
    /// Retrieves a shared pointer to an `ImageEntry` using an absolute file path.
    pub fn find_image(&self, absolute_path: &PathBuf) -> Option<&ImageEntry> {
        let relative = get_relative_key(absolute_path, self.images.root.as_ref())?;
        self.images.list.get(&relative)
    }

    pub fn find_image_mut(&mut self, absolute_path: &PathBuf) -> Option<&mut ImageEntry> {
        let relative = get_relative_key(absolute_path, self.images.root.as_ref())?;
        self.images.list.get_mut(&relative)
    }

    /// This project's image (its path in the list) that a results file
    /// names `results_rel` - relative to `run_root`, the image folder the
    /// analysis ran on (its run's project copy), which may not be this
    /// project's any more. See [`find_results_image`].
    pub fn find_results_image(
        &self,
        results_rel: &Path,
        run_root: Option<&Path>,
    ) -> Option<PathBuf> {
        find_results_image(
            &self.images.list,
            self.images.root.as_deref(),
            results_rel,
            run_root,
        )
    }

    pub fn get_image_absolute_path_from_relative(&self, path: &Path) -> Option<PathBuf> {
        // Check if the path exists in our list
        if self.images.list.contains_key(path) {
            Some(match self.images.root.as_ref() {
                Some(root) => root.join(path),
                None => path.to_path_buf(),
            })
        } else {
            None
        }
    }

    /// Updates the project's image root and performs a full synchronization.
    ///
    /// This method is destructive: it clears the current [`images.list`] and resets
    /// the [`current_image`] selection to ensure no "ghost" paths from the previous
    /// root remain in memory.
    ///
    /// # Arguments
    /// * `new_root` - The absolute path to the new directory containing the project images.
    ///
    /// # Note
    /// This should typically be followed by a background directory scan to
    /// repopulate the image list based on the new root.
    pub fn change_images_root(&mut self, new_root: &PathBuf) {
        // Clear the image list first (Prevent UI from accessing dead paths)

        self.images.list.clear();

        // Update the root

        self.images.root = Some(new_root.clone());

        // Reset actual selected image

        self.tmp_settings.current_image = None;
        self.tmp_settings.current_image_planes = None;

        info!(
            "Project root changed to {:?}. Image list cleared.",
            new_root
        );

        // Trigger a re-scan of the new directory
        //self.spawn_directory_scan(new_root);
    }

    /// Re-links an existing project to a new location on the filesystem.
    ///
    /// Use this when a project file has been moved or shared, and the image
    /// resources are now located at a different absolute path. Unlike
    /// `change_images_root`, this preserves the existing image metadata and list,
    /// only updating the base reference point.
    ///
    /// # Arguments
    /// * `new_root` - The new absolute path where the project's images are located.
    pub fn select_new_images_root_with_check(
        &mut self,
        files: &dyn FileSystem,
        new_root: &PathBuf,
    ) -> SelectNewProjectRootAction {
        if !self.does_project_image_exists_at_path(files, new_root) {
            return SelectNewProjectRootAction::ImageNotFound;
        }

        self.select_new_images_root(new_root);
        SelectNewProjectRootAction::Success
    }

    pub fn select_new_images_root(&mut self, new_root: &PathBuf) {
        {
            self.images.root = Some(new_root.clone());
        }
        info!("Project re-linked to new root: {:?}", new_root);

        // 3. Optional: Validation
        // You might want to trigger a "Check Integrity" scan here to verify
        // that the files actually exist at the new location.
    }

    pub fn does_project_images_exist(&self, files: &dyn FileSystem) -> bool {
        if let Some(root) = &self.images.root {
            return self.does_project_image_exists_at_path(files, root);
        };

        // No root set
        return true;
    }

    pub fn does_project_image_exists_at_path(
        &self,
        files: &dyn FileSystem,
        new_root: &PathBuf,
    ) -> bool {
        let sample_rel_path = { self.images.list.keys().next().cloned() };

        if let Some(rel) = sample_rel_path {
            let test_path = new_root.join(rel);
            if !matches!(files.stat(&test_path), Ok(Some(_))) {
                // Warn the user that the first image wasn't found here
                return false;
            }
        }
        return true;
    }

    pub fn is_image_part_of_the_root(&self, absolute_path: &Path) -> bool {
        if let Some(existing_root) = self.images.root.as_deref() {
            if is_in_root(absolute_path, existing_root) {
                return true;
            } else {
                return false;
            }
        } else {
            // No root yet set
            return false;
        }
    }

    /// Registers a new image and its associated metadata into the project.
    ///
    /// This method generates a relative path key based on the project's base directory
    /// and performs a batch-compatible insertion into the image map.
    ///
    /// ### Workflow
    /// 1. **Relativize:** Converts the absolute path to a project-relative `PathBuf`.
    /// 2. **Initialize:** Wraps the provided `ImageMeta` into a new `ImageEntry`.
    /// 3. **Lock & Insert:** Acquires a **write lock** to update the `IndexMap`.
    ///
    /// # Thread Safety
    /// This method requires a **unique write lock** on the `images` map.
    /// If multiple threads attempt to add images simultaneously, they will be serialized.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned by a previous thread failure.
    pub fn add_image_and_read_meta(
        &mut self,
        backend: &dyn Backend,
        absolute_path: &Path,
    ) -> ProjectAction {
        if self.is_supported_image(backend, absolute_path) {
            match backend.read_image_meta(absolute_path) {
                Ok(image_meta) => {
                    self.add_image(absolute_path, &image_meta);
                    return ProjectAction::Success;
                }
                Err(ex) => {
                    return ProjectAction::Failure(ex.to_string());
                }
            }
        } else {
            return ProjectAction::Failure("Unsupported device".into());
        }
    }

    pub fn add_image(&mut self, absolute_path: &Path, image_meta: &ImageMeta) -> ProjectAction {
        // Determine the effective root
        let effective_root = {
            if let Some(existing_root) = self.images.root.as_deref() {
                if is_in_root(absolute_path, existing_root) {
                    Some(existing_root.to_path_buf())
                } else {
                    // Case 2: Image is outside existing root.
                    // Decision: Do you ignore it, or force a root change?
                    warn!(
                        "Image {:?} is outside current root {:?}. Ignoring.",
                        absolute_path, existing_root
                    );
                    return ProjectAction::OutSideRootConflict {
                        image_path: absolute_path.into(),
                        current_root: existing_root.into(),
                    };
                }
            } else {
                None // Case 3: No root set yet
            }
        };

        // Handle the "No Root" case by setting it
        let final_root = match effective_root {
            Some(root) => root,
            None => {
                // Set the new root in the project settings
                let parent = absolute_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                info!("No root detected. Setting root to: {:?}", parent);

                self.images.root = Some(parent.clone());
                parent
            }
        };

        // Unified Success Path
        if let Some(relative_path) = get_relative_key(absolute_path, Some(&final_root)) {
            self.add_image_to_list(&relative_path, absolute_path, image_meta);
        } else {
            warn!("Failed to calculate relative path for {:?}", absolute_path);
        }
        ProjectAction::Success
    }

    /// Registers a new image into the project manifest.
    ///
    /// This method initializes default [`SeriesSettings`] and [`ChannelSettings`]
    /// based on the provided metadata. If the image (by relative path) already
    /// exists in the list, the operation is skipped to prevent duplicates.
    ///
    /// # Thread Safety
    /// Acquires a write lock on `self.images.list`. Will panic if the lock is poisoned.
    pub fn add_image_to_list(&mut self, rel_path: &Path, _abs_path: &Path, image_meta: &ImageMeta) {
        if self.images.list.contains_key(rel_path) {
            trace!("Image already exists at {:?}, skipping.", rel_path);
            return;
        }

        let mut file_size_approx_bytes: u64 = 0;
        //  Map Series and Channels
        let series_settings: BTreeMap<i32, SeriesSettings> = image_meta
            .series
            .iter()
            .map(|(&series_idx, data)| {
                let channels: BTreeMap<_, _> = data
                    .channels
                    .iter()
                    .map(|(ch_idx, data)| {
                        (
                            *ch_idx,
                            ChannelSettings {
                                name: data.name.clone(),
                                emission_wave_length: Some(data.emission_wave_length.clone()),
                                visible: None,
                                histogram: None,
                            },
                        )
                    })
                    .collect();

                let (image_width, image_height, nr_bits, rgb) = match data.resolutions.get(&0) {
                    Some(res) => (res.width, res.height, res.nr_bits, res.is_rgb),
                    None => (0, 0, 0, false), // Fallback if resolution 0 is missing
                };

                let pixel_sizes = data.pixel_sizes.clone();
                let settings = SeriesSettings {
                    selected_channel: None,
                    image_width: image_width,
                    image_height: image_height,
                    channels,
                    pixel_sizes: PixelSizeSettings {
                        x: pixel_sizes.px_size_x,
                        y: pixel_sizes.px_size_y,
                        // The reader's 0: unknown.
                        z: (pixel_sizes.px_size_z > 0.0).then_some(pixel_sizes.px_size_z),
                    },
                    z_stack: None,
                    t_stack: None,
                    objects: Vec::new(),
                    rgb,
                };

                file_size_approx_bytes += image_width * image_height * nr_bits as u64;

                (series_idx, settings)
            })
            .collect();

        let img = ImageEntry {
            review: Default::default(),
            rel_path: rel_path.to_path_buf(),
            file_size: file_size_approx_bytes,
            selected_series: 0,
            series: series_settings,
        };

        self.images.list.insert(rel_path.to_path_buf(), img);

        log::debug!(
            "Successfully added image to project: {}",
            rel_path.display()
        );
    }

    /// Describe this function.
    ///
    /// # Arguments
    ///
    /// - `path` (`PathBuf`) - Describe this parameter.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<ImageEntry>, InternalErrors>` - Describe the return value.
    ///
    /// # Errors
    ///
    /// Describe possible errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = scan_folder();
    /// ```
    ///
    pub fn scan_image_folder_and_add(&mut self, backend: &dyn Backend) {
        if let Some(root_folder) = self.settings.images.root.clone() {
            let found_images = collect_images_at_root(backend, &root_folder);
            self.apply_scanned_images(found_images);
        }
    }

    /// Replaces the project's image list with `found_images` (as returned by
    /// the free function [`collect_images_at_root`]). Unlike
    /// `scan_image_folder_and_add`, this does no filesystem I/O itself, so
    /// callers that already scanned off-lock only need to hold the project
    /// write guard for this call, not for the scan that produced its input.
    pub fn apply_scanned_images(&mut self, mut found_images: Vec<(PathBuf, ImageMeta)>) {
        // The user's notes, markers and left-outs stay with the images
        // found again.
        let reviews: Vec<(PathBuf, ImageReview)> = self
            .images
            .list
            .0
            .iter()
            .filter(|(_, image)| !image.review.is_empty())
            .map(|(path, image)| (path.clone(), image.review.clone()))
            .collect();
        self.images.list.clear();

        // Natural sort
        found_images.sort_by(|a, b| {
            let path_a = a.0.to_string_lossy();
            let path_b = b.0.to_string_lossy();
            compare(&path_a, &path_b)
        });

        let start = Instant::now();
        for (path, meta) in found_images {
            self.add_image(&mut &path, &meta);
        }
        for (path, review) in reviews {
            if let Some(image) = self.images.list.get_mut(&path) {
                image.review = review;
            }
        }
        // Folded folders that hold no image any more are forgotten.
        if !self.images.collapsed_folders.is_empty() {
            let folders: std::collections::HashSet<&Path> = self
                .images
                .list
                .0
                .keys()
                .map(|path| path.parent().unwrap_or(Path::new("")))
                .collect();
            let kept: std::collections::BTreeSet<PathBuf> = self
                .images
                .collapsed_folders
                .iter()
                .filter(|folder| folders.contains(folder.as_path()))
                .cloned()
                .collect();
            self.images.collapsed_folders = kept;
        }
        let duration = start.elapsed();
        info!("Added images {:?}", duration);
    }

    pub fn collect_images_parallel(
        &self,
        backend: &dyn Backend,
        dir: &Path,
    ) -> Vec<(PathBuf, ImageMeta)> {
        collect_images_at_root(backend, dir)
    }

    /// Whether `backend` can read `path` as an image (by its extension).
    pub fn is_supported_image(&self, backend: &dyn Backend, path: &Path) -> bool {
        is_supported_image_path(path, &backend.image_formats())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProjectAction {
    /// Everything went fine, image added.
    Success,
    /// The image is outside the root. The UI needs to ask the user what to do.
    OutSideRootConflict {
        image_path: PathBuf,
        current_root: PathBuf,
    },
    /// A legitimate failure (e.g., file doesn't exist).
    Failure(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum SelectNewProjectRootAction {
    /// Everything went fine, image added.
    Success,

    /// Image in the new project root not found
    ImageNotFound,
}

/// Recursively scans `dir` in parallel for supported image files and reads
/// each one's metadata, through `backend` (so in remote mode the server's
/// folder is scanned). Takes no project reference and holds no lock, so
/// callers on a project shared behind an `RwLock` (the GUI) should run this
/// *before* taking the write guard, then commit the result with
/// `ProjectWithRuntime::apply_scanned_images` - keeping the lock held only for the
/// fast in-memory part instead of for the whole scan (which reads every
/// image's metadata and can take seconds on a plate-sized folder).
pub fn collect_images_at_root(backend: &dyn Backend, dir: &Path) -> Vec<(PathBuf, ImageMeta)> {
    // 1. Check if the current directory itself is named "results"
    if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
        if name.eq_ignore_ascii_case("results") {
            return vec![]; // Skip this folder and everything inside it
        }
    }

    let Ok(entries) = backend.files().list_dir(dir) else {
        return vec![];
    };
    // The backend's formats - in remote mode the worker decides what it can
    // read, not the client's build.
    let formats = backend.image_formats();

    entries
        .into_par_iter()
        .flat_map(|entry| {
            if entry.is_dir {
                // The check happens again here for subdirectories
                collect_images_at_root(backend, &entry.path)
            } else if is_supported_image_path(&entry.path, &formats) {
                match backend.read_image_meta(&entry.path) {
                    Ok(meta) => vec![(entry.path, meta)],
                    Err(_) => vec![],
                }
            } else {
                vec![]
            }
        })
        .collect()
}

/// Whether `path`'s extension is one of `formats` (lowercase, no dot).
pub(super) fn is_supported_image_path(path: &Path, formats: &[String]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_lowercase();
            formats.iter().any(|format| *format == ext)
        })
        .unwrap_or(false)
}

/// The project's image settings: which images an analysis leaves out.
pub trait ImageSettingsExt {
    /// Whether an analysis leaves out the image at `rel_path` (relative to
    /// `root`): it lies right in a skipped folder.
    fn is_skipped(&self, rel_path: &Path) -> bool;
}

impl ImageSettingsExt for ImageSettings {
    fn is_skipped(&self, rel_path: &Path) -> bool {
        !self.skipped_folders.is_empty()
            && self
                .skipped_folders
                .contains(rel_path.parent().unwrap_or(Path::new("")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_images_right_in_a_skipped_folder_are_left_out() {
        let images = ImageSettings {
            skipped_folders: [PathBuf::from("A")].into_iter().collect(),
            ..Default::default()
        };
        assert!(images.is_skipped(Path::new("A/a.tif")));
        assert!(!images.is_skipped(Path::new("A/sub/s.tif")));
        assert!(!images.is_skipped(Path::new("top.tif")));
    }
}
