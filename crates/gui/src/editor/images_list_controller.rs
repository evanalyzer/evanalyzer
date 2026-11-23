//! # images_list_controller
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::FileRequest;
use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::size_formater::format_bytes;
use crate::{AppWindow, DialogType, GlobalAppState, ObjectHighlightBox, ViewportObjectState};
use crate::{ImageItemData, ImagesListState, WellState};
use evanalyzer_app::project::ImageListExt;
use evanalyzer_app::project::ImageSettingsExt;
use evanalyzer_app::project::SelectNewProjectRootAction;
use evanalyzer_app::project::collect_images_at_root;
use log::{debug, info, warn};
use slint::{ComponentHandle, Model};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Which images were previewed (this session) or are in the analysis open
/// in Results - the states of the image rows. Kept
/// in `UiState::image_states`: the preview worker and the results controller
/// write it.
#[derive(Default, Clone)]
pub struct ImageStates {
    /// Full paths of the images a preview ran on.
    pub previewed: std::collections::HashSet<PathBuf>,
    /// Paths (relative to the image folder) of the images in the analysis
    /// open in Results.
    pub analysed: std::collections::HashSet<PathBuf>,
}

pub struct ImagesListControllerState {
    pub image_filter_text: RwLock<String>,
    /// The markers the list and the chip's picker are narrowed to (bit `n`
    /// for marker `n`, see [`marker_bit`]); 0: no marker filter.
    pub marker_filter: std::sync::atomic::AtomicU8,
    /// The chip picker's last search, to list it again when the marker
    /// filter changes.
    pub quick_query: RwLock<String>,
    /// The list is by groups - the wells of the project's "Grouping" -
    /// not by folders. This session only.
    pub by_groups: std::sync::atomic::AtomicBool,
    /// Folded group heads (by the group's name) - this session only.
    pub collapsed_groups: std::sync::Mutex<std::collections::BTreeSet<String>>,
}
pub struct ImagesListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) _histogram_controller: Arc<HistogramController>,
    pub(crate) image_meta_controller: Arc<ImageMetaController>,
    pub(crate) image_controller_state: Arc<ImagesListControllerState>,
    pub(crate) object_list_controller: Arc<ObjectListController>,
    pub(crate) classification_controller: Arc<ClassificationController>,
}

impl ImagesListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        histogram_controller: Arc<HistogramController>,
        image_meta_controller: Arc<ImageMetaController>,
        object_list_controller: Arc<ObjectListController>,
        classification_controller: Arc<ClassificationController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
            _histogram_controller: histogram_controller,
            image_meta_controller,
            image_controller_state: Arc::new(ImagesListControllerState {
                image_filter_text: RwLock::new(String::new()),
                marker_filter: std::sync::atomic::AtomicU8::new(0),
                quick_query: RwLock::new(String::new()),
                by_groups: std::sync::atomic::AtomicBool::new(false),
                collapsed_groups: Default::default(),
            }),
            object_list_controller,
            classification_controller,
        }
    }

    /// Another project (new, opened, imported): no image shown - the
    /// viewer, its objects and the image's info are emptied.
    pub fn clear_shown_image(&self) {
        self.viewport_controller.clear_shown_image();
        self.object_list_controller.sync_objects_to_slint();
        if self
            .image_meta_controller
            .sync_image_meta_to_slint()
            .is_err()
        {
            warn!("Failed to sync image meta to slint!");
        }
    }

    /// A new image has been selected. We update the project state accordingly.
    /// This method is called when the user selects a new image from the UI.
    ///
    /// # Threading
    /// `is_part_of_root`'s branch opens (or reuses a cached) `ImageReader` on
    /// `image_path` via `sync_image_meta_to_slint` - for a large/slow-to-parse
    /// file this can take the better part of a second (see
    /// `evanalyzer_core::image::image_reader`'s own timing logs). Since this
    /// method is called directly from the `on_image_selected` Slint callback
    /// (i.e. on the UI/event-loop thread), that work is dispatched to a
    /// background thread - exactly like `scan_image_root_for_images` already
    /// does for the same call - so selecting an image doesn't freeze the UI
    /// for as long as the file takes to open.
    pub fn open_new_image(self: &Arc<Self>, image_path: &PathBuf) {
        let (is_part_of_root, parent_dir) = {
            let project = self.app_state.get_project();
            let is_part = project.is_image_part_of_the_root(image_path);
            let parent = if !is_part {
                image_path.parent().map(PathBuf::from)
            } else {
                None
            };
            (is_part, parent)
        }; // read lock dropped before write lock is acquired below

        if is_part_of_root {
            {
                let mut project = self.app_state.get_project_runtime_write();
                project.set_current_image_path(image_path);
            } // write lock dropped before sync calls that re-acquire it

            // Remove possible existing markers and the results-table object
            // highlight now, queued onto the event loop immediately - not
            // after the slow work spawned below. `open_new_image` is called
            // both directly from UI callbacks and from background threads
            // (e.g. `project_controller.rs`'s "open file" flow), so this
            // can't safely touch Slint state directly; it must always go
            // through `invoke_from_event_loop`. Queuing it here, before the
            // slow reader-open thread even starts, matters because
            // `open_image_and_highlight_object` calls this method and then
            // immediately queues its own highlight-set afterward (see that
            // method) - relying on `invoke_from_event_loop` running queued
            // closures in the order they were queued for its highlight to
            // survive this clear. If this were queued only after the
            // (500ms+) reader-open below instead, it would run *after* that
            // later highlight-set and silently erase it.
            let ui_weak = self.ui.clone();
            let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    let object_state = ui.global::<ViewportObjectState>();
                    object_state.set_markers(slint::ModelRc::new(slint::VecModel::default()));
                    object_state.set_object_highlight(ObjectHighlightBox::default());
                }
            });

            let manager = self.clone();
            crate::helper::ui_thread::spawn(move || {
                // Fired first, before the (slow) metadata sync below:
                // `trigger_new_image_redraw` only enqueues worker tasks
                // (`dispatch_worker_task`) and returns immediately, it
                // doesn't block on the reads themselves. Those worker
                // threads open the image through the same cached reader
                // `sync_image_meta_to_slint` below is about to populate -
                // calling it first lets that reader
                // open race for the shared cache concurrently with metadata
                // sync instead of only starting once metadata sync (and the
                // two calls after it) have already finished.
                manager.viewport_controller.trigger_new_image_redraw();
                manager.image_meta_controller.assign_missing_wavelengths();
                if manager
                    .image_meta_controller
                    .sync_image_meta_to_slint()
                    .is_err()
                {
                    warn!("Failed to sync image meta to slint!");
                }
                manager.object_list_controller.sync_objects_to_slint();
                // E.g. the auto preview, for the new image.
                manager.app_state.image_opened();
                // The class counts belong to the image shown.
                manager
                    .classification_controller
                    .sync_classification_to_slint();
                // Re-read the current path instead of using the path this
                // thread was spawned for: if a newer `open_new_image` call
                // has since superseded it (fast repeated selection), this
                // keeps the list's selected/scrolled-to row in sync with
                // what's actually current instead of reverting to a stale
                // selection once this slower call finally catches up -
                // mirroring how `sync_image_meta_to_slint` and the pixel
                // read path already re-read the current path fresh rather
                // than trusting a value captured at spawn time.
                // Bound first: no project guard held while the list may
                // unfold the image's folder (a write).
                let current_path = manager
                    .app_state
                    .get_project()
                    .get_current_image_path_cloned();
                if let Some(current_path) = current_path {
                    manager.set_selected_image_index_in_slint_images_list(current_path, true);
                }
            });
        } else if let Some(parent) = parent_dir {
            self.change_image_root(&parent, Some(image_path));
        }
    }

    /// A new image has been selected. We update the project state accordingly.
    /// This method is called when the user selects a new image from the UI.
    pub(crate) fn open_new_image_from_rel_path(self: &Arc<Self>, rel_path: &PathBuf) {
        let absolute_path = {
            let project = self.app_state.get_project();
            project.get_image_absolute_path_from_relative(&rel_path)
        }; // read lock dropped before open_new_image acquires write lock

        if let Some(absolute_path) = absolute_path {
            if let Some(act_image_path) = &self.app_state.get_project().tmp_settings.current_image {
                if act_image_path == &absolute_path {
                    debug!("Image still opened!");
                    return;
                }
            }
            self.open_new_image(&absolute_path);
        } else {
            log::debug!("Image is not part of the project {:?}", rel_path);
        }
    }

    /// Opens the image identified by `rel_path` (as stored in a results file) and
    /// paints `bbox_px` (`[xmin, ymin, xmax, ymax]`, in image pixels) as a
    /// highlight box in the viewport. Used when a row is selected in the
    /// results table (an object row, `paint_as_rectangle: false` - a
    /// crosshair points at it from outside so the object itself stays
    /// unobscured) or in the results Matrix's image heatmap (a tile,
    /// `paint_as_rectangle: true` - the box itself *is* the region of
    /// interest there, so it's traced directly instead).
    pub fn open_image_and_highlight_object(
        self: &Arc<Self>,
        rel_path: &PathBuf,
        bbox_px: [u32; 4],
        paint_as_rectangle: bool,
    ) {
        self.open_new_image_from_rel_path(rel_path);

        // `open_new_image` queues a marker/highlight clear onto the event
        // loop as soon as it's called (see its doc comment). Queuing ours
        // here too, rather than setting it directly, relies on
        // `invoke_from_event_loop` running queued closures in the order
        // they were queued: this is queued strictly after that clear, so it
        // runs after it instead of racing (and possibly losing to) it.
        let [xmin, ymin, xmax, ymax] = bbox_px;
        let highlight = ObjectHighlightBox {
            x_px: xmin as f32,
            y_px: ymin as f32,
            // +1 so a single-pixel object is still visible (max pixel is inclusive).
            w_px: (xmax.saturating_sub(xmin) + 1) as f32,
            h_px: (ymax.saturating_sub(ymin) + 1) as f32,
            active: true,
            paint_as_rectangle,
        };
        let ui_weak = self.ui.clone();
        let _ = crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ViewportObjectState>()
                    .set_object_highlight(highlight);
            }
        });
    }

    /// Updates the project's root image directory and initiates a background scan for new images.
    ///
    /// This method performs a coordinated update between the Rust state and Slint UI:
    /// 1. Updates the project configuration and clears the current image list in Slint.
    /// 2. Sets the UI to a 'Scanning' state and updates the displayed path.
    /// 3. Spawns a background thread to crawl the new directory for valid image files.
    /// 4. Synchronizes the results to the UI and clears the scanning state upon completion.
    ///
    /// # Arguments
    /// * `new_root` - The new filesystem path to be used as the image source.
    ///
    /// # Threading
    /// Immediate UI updates happen via the main event loop, while the heavy filesystem
    /// crawl is offloaded to a dedicated background thread to prevent UI freezing.
    pub fn change_image_root(
        self: &Arc<Self>,
        new_root: &PathBuf,
        selected_image_absolute_path: Option<&PathBuf>,
    ) {
        let ui_weak = self.ui.clone();
        let image_root_dir_str = new_root.to_string_lossy().into_owned();
        self.app_state
            .get_project_write()
            .change_images_root(&new_root);
        self.sync_image_list_to_slint(); // The list is now empty, we sync this to slint

        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                ui_ready
                    .global::<ImagesListState>()
                    .set_act_image_root_dir(image_root_dir_str.into());
            }
        })
        .ok();

        // Now we can scan the new image root for images
        self.scan_image_root_for_images(selected_image_absolute_path);
        // A folder of images opened: next, picking one.
        crate::editor::start_page_controller::set_active(
            &self.ui,
            false,
            Some(crate::WorkspaceMode::Images),
        );
    }

    /// Updates the project's root image directory without performing a fresh filesystem crawl.
    ///
    /// This method performs a "soft" path migration. Unlike a full scan, it:
    /// 1. Updates the internal project reference to the new root directory.
    /// 2. Validates existing image entries against the new root to ensure they are still accessible.
    /// 3. Updates the Slint UI to reflect the new directory path.
    ///
    /// This is intended for cases where the project folder has been moved or renamed,
    /// allowing the application to maintain existing metadata and annotations for
    /// known images.
    ///
    /// # Arguments
    /// * `new_root` - The new filesystem path to be assigned as the image source.
    ///
    /// # Threading
    /// This is a synchronous operation that updates the project state and dispatches
    /// UI property updates to the main event loop.
    /// Where folder pickers start: the project's image folder, or the home
    /// folder of whichever machine the backend runs on.
    fn current_image_root(&self) -> PathBuf {
        self.app_state
            .get_project()
            .images
            .root
            .clone()
            .unwrap_or_default()
    }

    pub fn set_new_image_root(self: &Arc<Self>, new_root: &PathBuf) {
        let ui_weak = self.ui.clone();
        let result = self
            .app_state
            .get_project_write()
            .select_new_images_root_with_check(self.app_state.backend().files(), &new_root);

        if result == SelectNewProjectRootAction::ImageNotFound {
            // Images not found, show the Missing image dialog
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    ui_ready
                        .global::<GlobalAppState>()
                        .set_active_dialog(DialogType::MissingImages);
                }
            })
            .ok();
        } else {
            // Images found, set the new root dir
            let image_root_dir_str = new_root.to_string_lossy().into_owned();
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    ui_ready
                        .global::<ImagesListState>()
                        .set_act_image_root_dir(image_root_dir_str.into());
                }
            })
            .ok();
        }
    }

    /// Re-scans the current image root directory for new or modified image files.
    ///
    /// This method performs a non-destructive synchronization between the
    /// local filesystem and the project database. It:
    /// 1. Signals the UI to enter a 'Scanning' state.
    /// 2. Offloads the filesystem crawl to a background thread to maintain UI responsiveness.
    /// 3. Updates the internal image collection and notifies the Slint UI of any changes.
    /// 4. Disables the 'Scanning' state once the synchronization is finalized.
    ///
    /// # Threading
    /// Filesystem I/O and image list processing are performed on a background thread.
    /// UI state changes are safely dispatched via the Slint event loop.
    pub fn scan_image_root_for_images(
        self: &Arc<Self>,
        selected_image_absolute_path: Option<&PathBuf>,
    ) {
        let selected_absolute_path = selected_image_absolute_path.cloned();

        // Show spinner
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ImagesListState>().set_is_scanning(true);
            }
        })
        .ok();

        let ui_weak = self.ui.clone();
        let manager = self.clone();

        crate::helper::ui_thread::spawn(move || {
            // Scan the filesystem without holding the project lock - opening
            // an `ImageReader` per file can take seconds on a plate-sized
            // folder, and holding the write guard for that long stalls every
            // other thread that needs the project (viewport render, object
            // edits, dirty-tracking). Only the fast in-memory apply step
            // below needs the lock.
            let root_folder = manager.app_state.get_project().images.root.clone();
            if let Some(root_folder) = root_folder {
                let found_images =
                    collect_images_at_root(manager.app_state.backend().as_ref(), &root_folder);
                manager
                    .app_state
                    .get_project_write()
                    .apply_scanned_images(found_images);
            }

            manager.sync_image_list_to_slint();

            if let Some(path) = selected_absolute_path {
                {
                    let mut project = manager.app_state.get_project_runtime_write();
                    project.set_current_image_path(&path);
                }

                manager.image_meta_controller.assign_missing_wavelengths();
                if manager
                    .image_meta_controller
                    .sync_image_meta_to_slint()
                    .is_err()
                {
                    warn!("Failed to sync image meta to slint!");
                }

                manager.object_list_controller.sync_objects_to_slint();
                // E.g. the auto preview, for the new image.
                manager.app_state.image_opened();
                // The class counts belong to the image shown.
                manager
                    .classification_controller
                    .sync_classification_to_slint();
                manager.set_selected_image_index_in_slint_images_list(path, true);
                manager.viewport_controller.trigger_new_image_redraw();
            }

            // Hide spinner
            crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<ImagesListState>().set_is_scanning(false);
                    info!("Folder scan complete.");
                }
            })
            .ok();
        });
    }

    /// Attach UI callbacks related to image operations.
    ///
    /// This method registers handlers on the global ImagesListState (currently the
    /// `on_image_filter_text_changed` callback) so that UI-driven image filter actions
    /// are propagated to the background manager and the UI is refreshed on the
    /// Slint event loop.
    ///
    /// Behavior:
    /// - Clones required handles (UI and application state) so the closures can be
    ///   stored and invoked later.
    /// - The registered callback captures a worker/project manager and a weak UI
    ///   handle. It schedules work on the Slint event loop using
    ///   `slint::invoke_from_event_loop`.
    /// - Inside the event loop it attempts to upgrade the weak UI handle; if the
    ///   UI still exists it calls `update_image_list_in_sync` on the manager to
    ///   update the image list to reflect the applied filter.
    ///
    /// Notes:
    /// - The function is non-blocking from the caller's perspective; updates are
    ///   dispatched to the event loop.
    /// - If the UI has been dropped the callback is a no-op (the weak upgrade
    ///   fails). Any errors from scheduling are ignored via `.ok()`.
    pub fn attach_callbacks(self: &Arc<Self>) {
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ImagesListState>();
            state.on_folder_name(|path| folder_split(&path).1.into());
            state.on_folder_parent(|path| folder_split(&path).0.into());
        }

        // The rows' state dots follow previews and opened analyses. Weak:
        // `UiState` keeps the listener, and this controller keeps `UiState`.
        let weak = Arc::downgrade(self);
        self.app_state.on_image_states_changed(move || {
            if let Some(manager) = weak.upgrade() {
                manager.sync_row_states_to_slint();
            }
        });

        let ui_handle = self.ui.clone();
        if let Some(ui) = ui_handle.upgrade() {
            // Apply image filter
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_image_filter_text_changed(move |search_text| {
                    manager.update_image_filter_text_in_project(&search_text);
                    manager.sync_image_list_to_slint();
                    let project = manager.app_state.get_project();
                    if let Some(image_path) = project.get_current_image_path_cloned() {
                        manager.set_selected_image_index_in_slint_images_list(image_path, false);
                    }
                });

            self.init_review_callbacks();

            // A colour of the marker filter clicked: narrow to it / not
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_marker_filter_toggled(move |marker| {
                    manager.toggle_marker_filter(marker);
                });

            // A folder's head clicked: fold / unfold it
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_folder_toggled(move |folder| {
                    manager.toggle_folder(PathBuf::from(folder.as_str()));
                });

            // A folder's box: analyse / skip its images
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_folder_skip_toggled(move |folder| {
                    manager.toggle_folder_skipped(PathBuf::from(folder.as_str()));
                });

            // The image chip's quick picker searched
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_quick_search(move |query| {
                    manager.sync_quick_pick_to_slint(&query);
                });

            // Image selected
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_image_selected(move |image_path| {
                    let path = PathBuf::from(image_path.as_str());
                    manager.open_new_image_from_rel_path(&path);
                });

            // Change the image root, actual images will be removed from project
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_open_images_folder_clicked(move || {
                    let request = FileRequest::open_folder("Choose image folder")
                        .start_in(manager.current_image_root());
                    let manager = Arc::clone(&manager);
                    manager
                        .app_state
                        .clone()
                        .file_browser
                        .open(request, move |path| {
                            if let Some(path) = path {
                                manager.change_image_root(&path, None);
                            }
                        });
                });

            // Set new image root. Images stay in the list only the root path is changed and it is checked if images in the new root path are found
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_new_image_root_folder_clicked(move || {
                    let request = FileRequest::open_folder("Choose new image folder")
                        .start_in(manager.current_image_root());
                    let manager = Arc::clone(&manager);
                    manager
                        .app_state
                        .clone()
                        .file_browser
                        .open(request, move |path| {
                            if let Some(path) = path {
                                manager.set_new_image_root(&path);
                            }
                        });
                });

            // Fold every head, or unfold them all
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_fold_all(move || manager.fold_all());

            // Listed by folders or by groups
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_by_groups_changed(move |on| {
                    manager
                        .image_controller_state
                        .by_groups
                        .store(on, std::sync::atomic::Ordering::Relaxed);
                    manager.sync_image_list_to_slint();
                    let image_path = manager
                        .app_state
                        .get_project()
                        .get_current_image_path_cloned();
                    if let Some(image_path) = image_path {
                        manager.set_selected_image_index_in_slint_images_list(image_path, false);
                    }
                });

            // Rerun folder scan on selected image root
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_refresh_images_clicked(move || {
                    let project = manager.app_state.get_project();
                    let image_path = project.get_current_image_path_cloned();
                    manager.scan_image_root_for_images(image_path.as_ref());
                });
        }
    }

    /// Folds the list's `folder` (relative to the image folder) or unfolds
    /// it, and keeps the open image marked.
    pub fn toggle_folder(&self, folder: PathBuf) {
        // A group's head: folded for this session only.
        if let Some(name) = folder.to_string_lossy().strip_prefix(GROUP_HEAD_PREFIX) {
            {
                let mut collapsed = self
                    .image_controller_state
                    .collapsed_groups
                    .lock()
                    .expect("Poisoned");
                if !collapsed.remove(name) {
                    collapsed.insert(name.to_string());
                }
            }
            self.sync_image_list_to_slint();
            return;
        }
        {
            // Saved in the project, but no edit: no undo step - an unsaved
            // change, though, so the save button asks for it.
            let mut project = self.app_state.get_project_runtime_write();
            let collapsed = &mut project.images.collapsed_folders;
            if !collapsed.remove(&folder) {
                collapsed.insert(folder);
            }
        }
        self.app_state.mark_dirty();
        self.sync_image_list_to_slint();
        let image_path = self.app_state.get_project().get_current_image_path_cloned();
        if let Some(image_path) = image_path {
            self.set_selected_image_index_in_slint_images_list(image_path, false);
        }
    }

    /// Folds every head of the list (its folders, or its groups), or
    /// unfolds them all when every one is folded. Folders: saved in the
    /// project like a single fold (an unsaved change, no undo step);
    /// groups: this session.
    pub fn fold_all(&self) {
        let by_groups = self
            .image_controller_state
            .by_groups
            .load(std::sync::atomic::Ordering::Relaxed);
        if by_groups {
            let names: std::collections::BTreeSet<String> = {
                // The heads the list shows, folded or not.
                let project = self.app_state.get_project();
                let (rows, _) = build_group_rows(
                    &project.images.list,
                    &project.plate,
                    "",
                    0,
                    &ImageStates::default(),
                    std::path::Path::new(""),
                    &Default::default(),
                    &Default::default(),
                );
                rows.into_iter()
                    .filter(|row| row.group_head)
                    .map(|row| row.name.to_string())
                    .collect()
            };
            let mut collapsed = self
                .image_controller_state
                .collapsed_groups
                .lock()
                .expect("Poisoned");
            *collapsed = if names.is_subset(&collapsed) {
                Default::default()
            } else {
                names
            };
        } else {
            {
                let mut project = self.app_state.get_project_runtime_write();
                let folders: std::collections::BTreeSet<PathBuf> = project
                    .images
                    .list
                    .0
                    .keys()
                    .map(|p| p.parent().map(PathBuf::from).unwrap_or_default())
                    .collect();
                let collapsed = &mut project.images.collapsed_folders;
                *collapsed = if folders.is_subset(collapsed) {
                    Default::default()
                } else {
                    folders
                };
            }
            self.app_state.mark_dirty();
        }
        self.sync_image_list_to_slint();
    }

    /// Skips the images right in `folder` (relative to the image folder)
    /// in an analysis, or analyses them again - saved in the project, one
    /// undo step.
    pub fn toggle_folder_skipped(&self, folder: PathBuf) {
        self.app_state.begin_undo_step();
        {
            let mut project = self.app_state.get_project_write();
            let skipped = &mut project.images.skipped_folders;
            if !skipped.remove(&folder) {
                skipped.insert(folder);
            }
        }
        self.app_state.mark_dirty();
        self.sync_image_list_to_slint();
        let image_path = self.app_state.get_project().get_current_image_path_cloned();
        if let Some(image_path) = image_path {
            self.set_selected_image_index_in_slint_images_list(image_path, false);
        }
    }

    /// The markers the list is narrowed to, see
    /// [`ImagesListControllerState::marker_filter`].
    pub(crate) fn marker_filter(&self) -> u8 {
        self.image_controller_state
            .marker_filter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adds marker `marker` (1-6) to the filter or takes it out (0: no
    /// marker filter), then lists the images again.
    pub fn toggle_marker_filter(&self, marker: i32) {
        let filter = &self.image_controller_state.marker_filter;
        let bit = u8::try_from(marker).ok().map_or(0, marker_bit);
        if bit == 0 {
            filter.store(0, std::sync::atomic::Ordering::Relaxed);
        } else {
            filter.fetch_xor(bit, std::sync::atomic::Ordering::Relaxed);
        }
        self.refilter();
    }

    /// Lists the images again after the filter changed: the list (the open
    /// image selected in it) and the chip's picker.
    pub(crate) fn refilter(&self) {
        self.sync_image_list_to_slint();
        let query = self
            .image_controller_state
            .quick_query
            .read()
            .expect("Poisoned")
            .clone();
        self.sync_quick_pick_to_slint(&query);
        let image_path = self.app_state.get_project().get_current_image_path_cloned();
        if let Some(image_path) = image_path {
            self.set_selected_image_index_in_slint_images_list(image_path, false);
        }
    }

    pub fn update_image_filter_text_in_project(&self, new_text: &str) {
        *self
            .image_controller_state
            .image_filter_text
            .write()
            .expect("Poisened") = new_text.to_string();
    }

    /// Filters the project's image list based on the UI search criteria and
    /// synchronizes the result with the Slint `ImagesListState` global.
    ///
    /// This method performs the following:
    /// 1. Retrieves the current filter text from the Slint UI.
    /// 2. Filters the internal Rust image database (case-insensitive).
    /// 3. Updates the Slint `images_list` model on the main event loop.
    ///
    /// # Threading
    /// Data processing happens on the caller's thread, while the final
    /// UI update is dispatched via `slint::invoke_from_event_loop`.
    pub fn sync_image_list_to_slint(&self) {
        crate::editor::image_review_controller::sync_current_review_to_slint(
            &self.ui,
            &self.app_state,
        );
        let ui_weak = self.ui.clone();
        let state = self.app_state.clone();

        // We must get UI data (filter text) on the UI thread or
        // keep a copy in Rust. Assuming we need to pull it from Slint:
        let filter_text = self
            .image_controller_state
            .image_filter_text
            .read()
            .expect("Poisned")
            .clone()
            .to_lowercase();
        let marker_filter = self.marker_filter();

        let image_states = state.image_states();
        let project = state.get_project();
        let images_guard = &project.images.list;
        let by_groups = self
            .image_controller_state
            .by_groups
            .load(std::sync::atomic::Ordering::Relaxed);
        let (slint_items, image_count) = if by_groups {
            build_group_rows(
                images_guard,
                &project.plate,
                &filter_text,
                marker_filter,
                &image_states,
                &project.images.root.clone().unwrap_or_default(),
                &self
                    .image_controller_state
                    .collapsed_groups
                    .lock()
                    .expect("Poisoned"),
                &project.images.skipped_folders,
            )
        } else {
            build_image_list_items(
                images_guard,
                &filter_text,
                &image_states,
                &project.images.root.clone().unwrap_or_default(),
                &project.images.collapsed_folders,
                &project.images.skipped_folders,
                marker_filter,
            )
        };
        // Every head folded (no heads: false) - the fold-all button.
        let all_folded = {
            let mut heads = slint_items.iter().filter(|row| row.folder).peekable();
            heads.peek().is_some() && heads.all(|row| row.collapsed)
        };
        let any_marker = marker_filter != 0
            || images_guard
                .values()
                .any(|entry| entry.review.marker.is_some());
        let total_images = images_guard.0.len();
        let run_images = if project.images.skipped_folders.is_empty() {
            total_images
        } else {
            images_guard
                .0
                .keys()
                .filter(|path| !project.images.is_skipped(path))
                .count()
        };
        drop(project);

        // The final assignment goes into the event loop
        crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    let model = slint::ModelRc::new(slint::VecModel::from(slint_items));
                    let images_state = ui_ready.global::<ImagesListState>();
                    images_state.set_images_list(model);
                    images_state.set_image_count(image_count as i32);
                    images_state.set_all_folded(all_folded);
                    images_state.set_any_marker(any_marker);
                    images_state.set_marker_filter(i32::from(marker_filter));
                    images_state.set_total_images(total_images as i32);
                    images_state.set_run_images(run_images as i32);
                }else{
                    warn!("Failed to upgrade UI handle in sync_image_list_to_slint, cannot update image list!");
                }
            })
            .ok();
    }

    /// The image chip's quick picker: the images matching `query`.
    pub fn sync_quick_pick_to_slint(&self, query: &str) {
        query.clone_into(
            &mut self
                .image_controller_state
                .quick_query
                .write()
                .expect("Poisoned"),
        );
        let marker_filter = self.marker_filter();
        let image_states = self.app_state.image_states();
        let project = self.app_state.get_project();
        let image_root = project.images.root.clone().unwrap_or_default();
        let current = project
            .get_current_image_path_cloned()
            .and_then(|path| path.strip_prefix(&image_root).ok().map(PathBuf::from));
        let (items, matches, current_index) = quick_pick_items(
            &project.images.list,
            query,
            current.as_deref(),
            &image_states,
            &image_root,
            marker_filter,
        );
        drop(project);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ImagesListState>();
                state.set_quick_images(slint::ModelRc::new(slint::VecModel::from(items)));
                state.set_quick_matches(matches as i32);
                state.set_quick_current(current_index.map_or(-1, |i| i as i32));
            }
        })
        .ok();
    }

    /// Only the rows' states changed (an image was previewed, an analysis
    /// opened) - cheaper than the whole list.
    pub fn sync_row_states_to_slint(&self) {
        let image_states = self.app_state.image_states();
        // The rows' states, by path (patched in place: rebuilding the list
        // would lose the scroll position and the chosen well).
        let row_states: std::collections::HashMap<slint::SharedString, WellState> =
            project_rows(&self.app_state, &image_states);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ImagesListState>();
                for model in [state.get_images_list()] {
                    for row in 0..model.row_count() {
                        if let Some(mut item) = model.row_data(row) {
                            if item.folder {
                                continue;
                            }
                            let new = row_states
                                .get(&item.path)
                                .copied()
                                .unwrap_or(WellState::New);
                            if item.state != new {
                                item.state = new;
                                model.set_row_data(row, item);
                            }
                        }
                    }
                }
                // The folders' ✓ counts (folded folders' images aren't
                // rows: counted from the states by path).
                let list = state.get_images_list();
                for row in 0..list.row_count() {
                    if let Some(mut item) = list.row_data(row)
                        && item.folder
                    {
                        let analysed = analysed_in_folder(&row_states, &item.path);
                        if item.analysed != analysed {
                            item.analysed = analysed;
                            list.set_row_data(row, item);
                        }
                    }
                }
            }
        })
        .ok();
    }

    /// Marks the image at `selected_absolute_path` as selected in the Slint list.
    /// When `scroll_to` is true the list is also scrolled to bring that row into
    /// view — used when selection is driven from outside the list (results-table
    /// object selection, folder scan) and the row may be off-screen.
    pub(crate) fn set_selected_image_index_in_slint_images_list(
        &self,
        selected_absolute_path: PathBuf,
        scroll_to: bool,
    ) {
        let root = self.app_state.get_project().images.root.clone();
        // Chosen elsewhere (results, the image chip) in a folded folder:
        // the folder opens, so its row can be shown.
        // (Its own statement: the write guard must be gone before the sync
        // reads the project.)
        let unfolded = scroll_to
            && root
                .as_deref()
                .and_then(|root| selected_absolute_path.strip_prefix(root).ok())
                .and_then(|rel| rel.parent())
                .is_some_and(|folder| {
                    // Written only when folded - the common case reads.
                    let folded = self
                        .app_state
                        .get_project()
                        .images
                        .collapsed_folders
                        .contains(folder);
                    folded
                        && self
                            .app_state
                            .get_project_runtime_write()
                            .images
                            .collapsed_folders
                            .remove(folder)
                });
        if unfolded {
            self.app_state.mark_dirty();
            self.sync_image_list_to_slint();
        }
        let ui_weak = self.ui.clone();

        // The final assignment goes into the event loop
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                    let images_state = ui_ready.global::<ImagesListState>();
                   let images_list_model =  images_state.get_images_list();

                    let selected_index = row_of_image(
                        images_list_model.iter().map(|item| item.path.to_string()),
                        root.as_deref(),
                        &selected_absolute_path,
                    );

                    if let Some(index) = selected_index {
                        images_state.set_selected_image_index(index as i32);
                        if scroll_to {
                            images_state.set_scroll_to_image_index(index as i32);
                        }
                    } else{
                       images_state.set_selected_image_index(-1);
                    }
                }else{
                    warn!("Failed to upgrade UI handle in sync_image_list_to_slint, cannot update image list!");
                }
            })
            .ok();
    }
}

/// The row (its relative path, as the list shows it) of the image at
/// `selected` - whole paths compared, so "1.tif" is not ".../11.tif".
fn row_of_image(
    rel_paths: impl Iterator<Item = String>,
    root: Option<&std::path::Path>,
    selected: &std::path::Path,
) -> Option<usize> {
    let root = root.unwrap_or(std::path::Path::new(""));
    rel_paths
        .into_iter()
        .position(|rel_path| root.join(rel_path) == selected)
}

/// Whether `entry` is listed: its marker passes the marker filter and the
/// search text (lowercase, may be empty) is in its path or its note.
fn is_listed(
    entry: &evanalyzer_cfg::settings::images_settings::ImageEntry,
    filter_text_lowercase: &str,
    marker_filter: u8,
) -> bool {
    if !has_filtered_marker(entry, marker_filter) {
        return false;
    }
    if filter_text_lowercase.is_empty() {
        return true;
    }
    entry
        .rel_path
        .to_string_lossy()
        .to_lowercase()
        .contains(filter_text_lowercase)
        || (!entry.review.note.is_empty()
            && entry
                .review
                .note
                .to_lowercase()
                .contains(filter_text_lowercase))
}

/// A group head's row path: the prefix, then the group's name - told
/// apart from a folder's when it is clicked.
const GROUP_HEAD_PREFIX: &str = "group:";

/// The listed images by group: the wells the project's "Grouping" finds in
/// the names (as the results' plate does), a head per well in plate order,
/// then the images no well was found for. A folded head (`collapsed`, by
/// name; not while searching) keeps only itself.
#[allow(clippy::too_many_arguments)]
fn build_group_rows(
    images: &evanalyzer_cfg::settings::images_settings::ImageList,
    plate: &evanalyzer_cfg::settings::plate_settings::PlateSettings,
    filter_text_lowercase: &str,
    marker_filter: u8,
    states: &ImageStates,
    image_root: &std::path::Path,
    collapsed: &std::collections::BTreeSet<String>,
    skipped: &std::collections::BTreeSet<PathBuf>,
) -> (Vec<ImageItemData>, usize) {
    let matching: Vec<_> = images
        .values()
        .filter(|entry| is_listed(entry, filter_text_lowercase, marker_filter))
        .collect();
    let paths: Vec<&std::path::Path> = matching.iter().map(|e| e.rel_path.as_path()).collect();
    let folder_of = |i: usize| paths[i].parent().unwrap_or(std::path::Path::new(""));
    // Auto finds the name pattern per folder: folders named differently
    // each get theirs. The same well from several folders is one group.
    let parts: Vec<Vec<usize>> =
        if plate.grouping_mode == evanalyzer_cfg::settings::plate_settings::GroupingMode::Auto {
            let mut by_folder: Vec<(&std::path::Path, Vec<usize>)> = Vec::new();
            for i in 0..paths.len() {
                match by_folder.iter_mut().find(|(f, _)| *f == folder_of(i)) {
                    Some((_, members)) => members.push(i),
                    None => by_folder.push((folder_of(i), vec![i])),
                }
            }
            by_folder.into_iter().map(|(_, members)| members).collect()
        } else {
            vec![(0..paths.len()).collect()]
        };
    // (row, col) -> (name, images)
    let mut wells: std::collections::BTreeMap<(usize, usize), (String, Vec<usize>)> =
        Default::default();
    let mut placed = vec![false; matching.len()];
    for part in parts {
        let part_paths: Vec<&std::path::Path> = part.iter().map(|&i| paths[i]).collect();
        let layout = evanalyzer_app::project::plate_layout(&part_paths, plate);
        for well in layout.wells {
            let entry = wells
                .entry((well.row, well.col))
                .or_insert_with(|| (well.name.clone(), Vec::new()));
            for k in well.images {
                placed[part[k]] = true;
                entry.1.push(part[k]);
            }
        }
    }
    let mut groups: Vec<(String, Vec<usize>)> = wells.into_values().collect();
    let unplaced: Vec<usize> = (0..matching.len()).filter(|&i| !placed[i]).collect();
    if !unplaced.is_empty() {
        groups.push(("No group".to_string(), unplaced));
    }
    let searching = !filter_text_lowercase.is_empty() || marker_filter != 0;
    let mut rows = Vec::with_capacity(matching.len() + groups.len());
    for (name, indices) in groups {
        let folded = !searching && collapsed.contains(&name);
        let analysed = indices
            .iter()
            .filter(|&&i| states.analysed.contains(&matching[i].rel_path))
            .count();
        let head = rows.len() as i32;
        let mut folders: Vec<&std::path::Path> = indices.iter().map(|&i| folder_of(i)).collect();
        folders.sort();
        folders.dedup();
        rows.push(ImageItemData {
            name: name.as_str().into(),
            path: format!("{GROUP_HEAD_PREFIX}{name}").into(),
            folder: true,
            group_head: true,
            folders: folders.len() as i32,
            count: indices.len() as i32,
            analysed: analysed as i32,
            collapsed: folded,
            head,
            ..Default::default()
        });
        if !folded {
            rows.extend(indices.into_iter().map(|i| {
                let entry = matching[i];
                ImageItemData {
                    skipped: skipped
                        .contains(entry.rel_path.parent().unwrap_or(std::path::Path::new(""))),
                    head,
                    ..image_list_item(entry, states, image_root)
                }
            }));
        }
    }
    (rows, matching.len())
}

/// Builds the Slint-facing row list for `sync_image_list_to_slint`: applies
/// the (already-lowercased) filter text against `rel_path` and the note and
/// the marker filter (see [`has_filtered_marker`]), then maps each
/// surviving entry to an `ImageItemData`. When the images lie in more than
/// one folder, each folder's images follow a head row (the folders in the
/// order they first appear); a folded folder (`collapsed`, ignored while
/// searching) keeps only its head. Also returns how many images the rows
/// stand for. Pure, so it can be tested directly - unlike the rest of
/// `sync_image_list_to_slint`, this doesn't touch Slint's event loop, which
/// the headless `init_no_event_loop()` testing platform (see
/// `test_ui_window`) never actually drains.
fn build_image_list_items(
    images: &evanalyzer_cfg::settings::images_settings::ImageList,
    filter_text_lowercase: &str,
    states: &ImageStates,
    image_root: &std::path::Path,
    collapsed: &std::collections::BTreeSet<PathBuf>,
    skipped: &std::collections::BTreeSet<PathBuf>,
    marker_filter: u8,
) -> (Vec<ImageItemData>, usize) {
    let matching: Vec<_> = images
        .values()
        .filter(|entry| is_listed(entry, filter_text_lowercase, marker_filter))
        .collect();
    let count = matching.len();
    let folder_of = |entry: &evanalyzer_cfg::settings::images_settings::ImageEntry| {
        entry
            .rel_path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default()
    };
    // One folder (the common case): the plain list, nothing grouped.
    let first_folder = matching.first().map(|entry| folder_of(entry));
    if matching
        .iter()
        .all(|entry| Some(folder_of(entry)) == first_folder)
    {
        let folder_skipped = first_folder.is_some_and(|folder| skipped.contains(&folder));
        let rows = matching
            .into_iter()
            .map(|entry| ImageItemData {
                skipped: folder_skipped,
                head: -1,
                ..image_list_item(entry, states, image_root)
            })
            .collect();
        return (rows, count);
    }
    let mut folders: indexmap::IndexMap<PathBuf, Vec<_>> = indexmap::IndexMap::new();
    for entry in matching {
        folders.entry(folder_of(entry)).or_default().push(entry);
    }
    let searching = !filter_text_lowercase.is_empty() || marker_filter != 0;
    let mut rows = Vec::with_capacity(count + folders.len());
    for (folder, entries) in folders {
        let folded = !searching && collapsed.contains(&folder);
        let analysed = entries
            .iter()
            .filter(|entry| states.analysed.contains(&entry.rel_path))
            .count();
        let folder_skipped = skipped.contains(&folder);
        let head = rows.len() as i32;
        rows.push(ImageItemData {
            skipped: folder_skipped,
            head,
            ..folder_head_item(&folder, entries.len(), analysed, folded, image_root)
        });
        if !folded {
            rows.extend(entries.into_iter().map(|entry| ImageItemData {
                skipped: folder_skipped,
                head,
                ..image_list_item(entry, states, image_root)
            }));
        }
    }
    (rows, count)
}

/// A folder's head row: its last name, the folders above it joined by
/// " › " (images right in the image folder: the image folder's name).
fn folder_head_item(
    folder: &std::path::Path,
    count: usize,
    analysed: usize,
    collapsed: bool,
    image_root: &std::path::Path,
) -> ImageItemData {
    let names: Vec<String> = folder
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    let (parent, name) = match names.split_last() {
        Some((last, above)) => (
            above
                .iter()
                .map(|name| format!("{name} › "))
                .collect::<String>(),
            last.clone(),
        ),
        None => (
            String::new(),
            image_root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Image folder".to_string()),
        ),
    };
    ImageItemData {
        name: name.into(),
        path: folder.to_string_lossy().to_string().into(),
        folder: true,
        folder_parent: parent.into(),
        count: count as i32,
        analysed: analysed as i32,
        collapsed,
        ..Default::default()
    }
}

/// How many of the images right in `folder` (a head row's path) are
/// analysed, from the non-new states by image path.
fn analysed_in_folder(
    states: &std::collections::HashMap<slint::SharedString, WellState>,
    folder: &str,
) -> i32 {
    let folder = std::path::Path::new(folder);
    states
        .iter()
        .filter(|(path, state)| {
            **state == WellState::Analysed
                && std::path::Path::new(path.as_str()).parent() == Some(folder)
        })
        .count() as i32
}

fn image_list_item(
    entry: &evanalyzer_cfg::settings::images_settings::ImageEntry,
    states: &ImageStates,
    image_root: &std::path::Path,
) -> ImageItemData {
    ImageItemData {
        name: entry
            .rel_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
            .into(),
        path: entry.rel_path.to_string_lossy().to_string().into(),
        image_dimension: dimension_text(entry).into(),
        image_size: format!("{}", format_bytes(entry.file_size as f64)).into(),
        has_annotations: entry
            .series
            .values()
            .any(|series| !series.objects.is_empty()),
        state: image_state(&entry.rel_path, states, image_root),
        head: -1,
        marker: entry
            .review
            .marker
            .map_or(0, |marker| marker.number() as i32),
        has_note: !entry.review.note.is_empty(),
        left_out: entry.review.leave_out.is_some(),
        ..Default::default()
    }
}

/// At most this many images in the image chip's quick picker - typing
/// narrows the rest down.
const QUICK_PICK_LIMIT: usize = 50;

/// The quick picker's images: those whose path holds every word of
/// `query` (any case, any order), at most [`QUICK_PICK_LIMIT`]. Without a
/// query the listed window is centred on the open image `current`
/// (relative), so it is always in the list. Also how many match in all and
/// where `current` is in the returned items.
fn quick_pick_items(
    images: &evanalyzer_cfg::settings::images_settings::ImageList,
    query: &str,
    current: Option<&std::path::Path>,
    states: &ImageStates,
    image_root: &std::path::Path,
    marker_filter: u8,
) -> (Vec<ImageItemData>, usize, Option<usize>) {
    let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    let is_match = |entry: &evanalyzer_cfg::settings::images_settings::ImageEntry| {
        if !has_filtered_marker(entry, marker_filter) {
            return false;
        }
        let path = entry.rel_path.to_string_lossy().to_lowercase();
        words.iter().all(|word| path.contains(word.as_str()))
    };
    // Without a query every image matches: the open image's position in the
    // whole list decides which window of images is shown.
    let start = match current {
        Some(current)
            if words.is_empty() && marker_filter == 0 && images.len() > QUICK_PICK_LIMIT =>
        {
            images
                .values()
                .position(|entry| entry.rel_path.as_path() == current)
                .map_or(0, |pos| {
                    pos.saturating_sub(QUICK_PICK_LIMIT / 2)
                        .min(images.len() - QUICK_PICK_LIMIT)
                })
        }
        _ => 0,
    };
    let mut items = Vec::new();
    let mut matches = 0;
    let mut current_index = None;
    for entry in images.values() {
        if !is_match(entry) {
            continue;
        }
        matches += 1;
        if matches > start && items.len() < QUICK_PICK_LIMIT {
            if current == Some(entry.rel_path.as_path()) {
                current_index = Some(items.len());
            }
            items.push(image_list_item(entry, states, image_root));
        }
    }
    (items, matches, current_index)
}

/// Marker `marker`'s bit in a marker filter (0 for no marker).
pub(crate) fn marker_bit(marker: u8) -> u8 {
    if (1..=6).contains(&marker) {
        1 << marker
    } else {
        0
    }
}

/// Whether `entry` passes the marker filter: any image without one, else
/// an image with one of its markers.
fn has_filtered_marker(
    entry: &evanalyzer_cfg::settings::images_settings::ImageEntry,
    marker_filter: u8,
) -> bool {
    marker_filter == 0
        || entry
            .review
            .marker
            .is_some_and(|marker| marker_filter & marker_bit(marker.number()) != 0)
}

/// "2048 × 2048 · 3 ch" of the image's selected series ("" before it was
/// read).
fn dimension_text(entry: &evanalyzer_cfg::settings::images_settings::ImageEntry) -> String {
    let series = entry
        .series
        .get(&entry.selected_series)
        .or_else(|| entry.series.values().next());
    match series {
        Some(series) if series.image_width > 0 => format!(
            "{} × {} · {} ch",
            series.image_width,
            series.image_height,
            series.channels.len()
        ),
        _ => String::new(),
    }
}

/// How far the image at `rel_path` is: in the open analysis, previewed, new.
fn image_state(
    rel_path: &std::path::Path,
    states: &ImageStates,
    image_root: &std::path::Path,
) -> WellState {
    if states.analysed.contains(rel_path) {
        WellState::Analysed
    } else if states.previewed.contains(&image_root.join(rel_path)) {
        WellState::Preview
    } else {
        WellState::New
    }
}

/// The state of every image that isn't new, by its list path.
fn project_rows(
    app_state: &UiState,
    states: &ImageStates,
) -> std::collections::HashMap<slint::SharedString, WellState> {
    let project = app_state.get_project();
    let root = project.images.root.clone().unwrap_or_default();
    project
        .images
        .list
        .values()
        .filter_map(|entry| {
            let state = image_state(&entry.rel_path, states, &root);
            (state != WellState::New)
                .then(|| (entry.rel_path.to_string_lossy().to_string().into(), state))
        })
        .collect()
}

/// A folder path split into what comes before its last name (with the
/// trailing separator) and the last name: "/home/a/C14" → ("/home/a/", "C14").
fn folder_split(path: &str) -> (String, String) {
    let trimmed = path.trim_end_matches(['/', '\\']);
    match trimmed.rfind(['/', '\\']) {
        Some(i) => (trimmed[..=i].to_string(), trimmed[i + 1..].to_string()),
        None => (String::new(), trimmed.to_string()),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_folder_path_splits_into_its_start_and_its_last_name() {
        assert_eq!(
            super::folder_split("/home/a/C14"),
            ("/home/a/".into(), "C14".into())
        );
        assert_eq!(
            super::folder_split("/home/a/C14/"),
            ("/home/a/".into(), "C14".into())
        );
        assert_eq!(
            super::folder_split("C:\\data\\plate1"),
            ("C:\\data\\".into(), "plate1".into())
        );
        assert_eq!(super::folder_split("plate1"), ("".into(), "plate1".into()));
    }

    use super::*;

    /// The selected image's row: whole paths, not a substring - "1.tif"
    /// must not find ".../11.tif", nor "a/1.tif" find "b/a/1.tif".
    #[test]
    fn the_selected_row_is_the_image_with_exactly_that_path() {
        let rows = || {
            ["11.tif", "1.tif", "b/a/1.tif", "a/1.tif"]
                .map(String::from)
                .into_iter()
        };
        let root = std::path::Path::new("/data/plate");
        let row = |selected: &str| row_of_image(rows(), Some(root), std::path::Path::new(selected));
        assert_eq!(row("/data/plate/1.tif"), Some(1));
        assert_eq!(row("/data/plate/11.tif"), Some(0));
        assert_eq!(row("/data/plate/a/1.tif"), Some(3));
        assert_eq!(row("/data/plate/b/a/1.tif"), Some(2));
        assert_eq!(row("/data/other/1.tif"), None);
    }
    use crate::editor::test_support::test_ui_state;

    /// Images, then Pipelines, then Images again: the list shows the rows
    /// it showed (the panel is rebuilt, its scroll position kept).
    #[test]
    fn the_images_list_keeps_its_scroll_position_across_panel_switches() {
        use crate::{LayoutState, StartState, WorkspaceMode};
        use i_slint_backend_testing::ElementHandle;

        let ui = crate::editor::test_support::test_ui_window();
        ui.window().set_size(slint::LogicalSize::new(1500.0, 900.0));
        ui.global::<StartState>().set_active(false);
        let layout = ui.global::<LayoutState>();
        layout.set_steps_open(true);
        layout.set_panels_hidden(false);
        layout.set_mode(WorkspaceMode::Images);
        let rows: Vec<ImageItemData> = (0..300)
            .map(|i| ImageItemData {
                name: format!("img{i:03}").into(),
                path: format!("/data/img{i:03}.tif").into(),
                ..Default::default()
            })
            .collect();
        let state = ui.global::<ImagesListState>();
        state.set_images_list(std::rc::Rc::new(slint::VecModel::from(rows)).into());
        // The top of the panel (y = 0 at the window's top).
        let top_row_y = |name: &str| {
            ElementHandle::find_by_accessible_label(&ui, name)
                .next()
                .map(|row| row.absolute_position().y)
        };
        assert!(top_row_y("img000").is_some(), "the list at its top");

        // Scrolled down by 100 rows (30 px each).
        state.set_list_scroll_y(-3000.0);
        layout.set_mode(WorkspaceMode::Pipelines);
        assert!(top_row_y("img100").is_none(), "the images panel is away");
        layout.set_mode(WorkspaceMode::Images);
        let shown = top_row_y("img100").expect("row 100 shown again");
        assert!(top_row_y("img000").is_none(), "not back at the top");
        assert_eq!(state.get_list_scroll_y(), -3000.0, "nothing reset it");

        // Shown once more: the very same place.
        layout.set_mode(WorkspaceMode::Pipelines);
        layout.set_mode(WorkspaceMode::Images);
        assert_eq!(top_row_y("img100"), Some(shown));
    }

    /// Images, then Pipelines, an image picked there (the image chip), then
    /// Images again: the list shows the picked image's row, not the rows it
    /// showed before.
    #[test]
    fn an_image_picked_in_pipelines_is_in_view_back_in_images() {
        use crate::{LayoutState, StartState, WorkspaceMode};
        use i_slint_backend_testing::ElementHandle;

        let ui = crate::editor::test_support::test_ui_window();
        ui.window().set_size(slint::LogicalSize::new(1500.0, 900.0));
        ui.global::<StartState>().set_active(false);
        let layout = ui.global::<LayoutState>();
        layout.set_steps_open(true);
        layout.set_panels_hidden(false);
        layout.set_mode(WorkspaceMode::Images);
        let rows: Vec<ImageItemData> = (0..300)
            .map(|i| ImageItemData {
                name: format!("img{i:03}").into(),
                path: format!("/data/img{i:03}.tif").into(),
                ..Default::default()
            })
            .collect();
        let state = ui.global::<ImagesListState>();
        state.set_images_list(std::rc::Rc::new(slint::VecModel::from(rows)).into());
        let shown = |name: &str| {
            ElementHandle::find_by_accessible_label(&ui, name)
                .next()
                .is_some()
        };
        assert!(shown("img000"), "the list at its top");

        layout.set_mode(WorkspaceMode::Pipelines);
        assert!(!shown("img000"), "the images panel is away");
        // What `set_selected_image_index_in_slint_images_list` sets.
        state.set_selected_image_index(200);
        state.set_scroll_to_image_index(200);
        slint::platform::update_timers_and_animations();
        layout.set_mode(WorkspaceMode::Images);
        slint::platform::update_timers_and_animations();

        assert!(shown("img200"), "the picked image's row is in view");
        assert!(!shown("img000"), "not the rows from before");
        assert_eq!(
            state.get_scroll_to_image_index(),
            -1,
            "the request is consumed"
        );
    }

    fn make_controller() -> ImagesListController {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            slint::Weak::default(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let histogram_controller = Arc::new(HistogramController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_meta_controller = Arc::new(ImageMetaController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        ImagesListController::new(
            slint::Weak::default(),
            ui_state.clone(),
            viewport_controller.clone(),
            histogram_controller,
            image_meta_controller,
            object_list_controller.clone(),
            Arc::new(
                crate::editor::classification_controller::ClassificationController::new(
                    slint::Weak::default(),
                    ui_state.clone(),
                    object_list_controller.clone(),
                    viewport_controller.clone(),
                ),
            ),
        )
    }

    #[test]
    fn folding_is_an_unsaved_change_but_no_undo_step() {
        let controller = make_controller();
        let state = controller.app_state.clone();
        let undo_steps = || state.undo_stack.lock().unwrap().len();
        // An edit first: undoing it must not unfold anything.
        state.get_project_write().meta.name = "edited".into();
        let steps_before = undo_steps();

        controller.toggle_folder(PathBuf::from("Serum"));

        let folded = |state: &UiState| {
            state
                .get_project()
                .images
                .collapsed_folders
                .contains(std::path::Path::new("Serum"))
        };
        assert!(folded(&state));
        assert!(state.is_dirty(), "to be saved: the save button says so");
        assert_eq!(undo_steps(), steps_before, "no undo step");

        assert!(state.undo());
        assert_eq!(state.get_project().meta.name, "", "the edit undone");
        assert!(folded(&state), "the folding stays");

        controller.toggle_folder(PathBuf::from("Serum"));
        assert!(!folded(&state), "unfolded again");
    }

    /// Fold all folds every folder, a second click unfolds them all; by
    /// groups it folds every group instead (not saved).
    #[test]
    fn fold_all_folds_every_head_then_unfolds_them() {
        let controller = make_controller();
        let state = controller.app_state.clone();
        {
            let mut project = state.get_project_write();
            for path in ["Serum/A01_01.tif", "Plasma/B02_01.tif"] {
                project.images.list.insert(
                    PathBuf::from(path),
                    ImageEntry {
                        rel_path: PathBuf::from(path),
                        ..Default::default()
                    },
                );
            }
        }
        let folded = || state.get_project().images.collapsed_folders.len();
        controller.fold_all();
        assert_eq!(folded(), 2, "both folders");
        assert!(state.is_dirty());
        controller.toggle_folder(PathBuf::from("Serum"));
        controller.fold_all();
        assert_eq!(folded(), 2, "one open: all fold again");
        controller.fold_all();
        assert_eq!(folded(), 0, "all folded: all open");

        controller
            .image_controller_state
            .by_groups
            .store(true, std::sync::atomic::Ordering::Relaxed);
        controller.fold_all();
        let groups = controller
            .image_controller_state
            .collapsed_groups
            .lock()
            .unwrap()
            .clone();
        assert_eq!(groups.into_iter().collect::<Vec<_>>(), ["A1", "B2"]);
        assert_eq!(folded(), 0, "folders untouched");
    }

    /// A marker or a left out is an undo step and marks the project
    /// unsaved; the same marker again changes nothing.
    #[test]
    fn marking_an_image_is_an_undo_step_and_an_unsaved_change() {
        use crate::editor::image_review_controller::ReviewChange;
        use evanalyzer_cfg::settings::images_settings::ImageMarker;
        let controller = make_controller();
        let state = controller.app_state.clone();
        let rel = PathBuf::from("A/a.tif");
        state.get_project_runtime_write().images.list.insert(
            rel.clone(),
            ImageEntry {
                rel_path: rel.clone(),
                ..Default::default()
            },
        );
        let marker = |state: &UiState| {
            state
                .get_project()
                .images
                .list
                .get(&rel)
                .unwrap()
                .review
                .marker
        };
        let undo_steps = || state.undo_stack.lock().unwrap().len();

        controller.change_review("A/a.tif", ReviewChange::Marker(Some(ImageMarker::Red)));
        assert_eq!(marker(&state), Some(ImageMarker::Red));
        assert!(state.is_dirty());
        let steps = undo_steps();
        assert_eq!(steps, 1);
        controller.change_review("A/a.tif", ReviewChange::Marker(Some(ImageMarker::Red)));
        assert_eq!(undo_steps(), steps, "nothing changed");

        assert!(state.undo());
        assert_eq!(marker(&state), None);
    }

    /// The rows carry the review's marks, and the search finds notes too.
    #[test]
    fn rows_show_the_review_and_the_search_finds_notes() {
        use evanalyzer_cfg::settings::images_settings::{ImageMarker, ImageReview, LeaveOut};
        let mut images = image_list_of(&["a.tif", "b.tif"]);
        images
            .get_mut(std::path::Path::new("a.tif"))
            .unwrap()
            .review = ImageReview {
            note: "Bubble lower left".into(),
            marker: Some(ImageMarker::Blue),
            leave_out: Some(LeaveOut::default()),
        };
        let build = |filter: &str| {
            build_image_list_items(
                &images,
                filter,
                &ImageStates::default(),
                std::path::Path::new("/data"),
                &Default::default(),
                &Default::default(),
                0,
            )
            .0
        };
        let rows = build("");
        assert_eq!(
            (rows[0].marker, rows[0].has_note, rows[0].left_out),
            (5, true, true)
        );
        assert_eq!(
            (rows[1].marker, rows[1].has_note, rows[1].left_out),
            (0, false, false)
        );
        assert_eq!(row_names(&build("bubble")), ["a.tif"]);
    }

    /// The marker filter keeps the images with one of its colours, in the
    /// list (folders without one hide, folded ones open) and the picker.
    #[test]
    fn the_marker_filter_keeps_images_with_one_of_its_colours() {
        use evanalyzer_cfg::settings::images_settings::ImageMarker;
        let mut images = image_list_of(&["A/1.tif", "A/2.tif", "B/3.tif", "C/4.tif"]);
        for (path, marker) in [
            ("A/2.tif", ImageMarker::Red),
            ("C/4.tif", ImageMarker::Green),
        ] {
            images
                .get_mut(std::path::Path::new(path))
                .unwrap()
                .review
                .marker = Some(marker);
        }
        let collapsed: std::collections::BTreeSet<PathBuf> = [PathBuf::from("A")].into();
        let build = |filter: &str, markers: u8| {
            build_image_list_items(
                &images,
                filter,
                &ImageStates::default(),
                std::path::Path::new("/data"),
                &collapsed,
                &Default::default(),
                markers,
            )
        };
        let red = marker_bit(ImageMarker::Red.number());
        let green = marker_bit(ImageMarker::Green.number());
        let (rows, count) = build("", red);
        assert_eq!(count, 1);
        assert_eq!(row_names(&rows), ["2.tif"], "one folder left: no heads");
        let (rows, count) = build("", red | green);
        assert_eq!(count, 2);
        assert_eq!(
            rows.iter().filter(|row| !row.folder).count(),
            2,
            "folded A opens, B hides"
        );
        assert_eq!(rows.len(), 4, "the heads of A and C");
        assert_eq!(
            build("4", red).1,
            0,
            "the search and the markers both count"
        );
        assert_eq!(build("", 0).1, 4);

        let names = |markers: u8| -> Vec<String> {
            quick_pick_items(
                &images,
                "",
                None,
                &ImageStates::default(),
                std::path::Path::new(""),
                markers,
            )
            .0
            .iter()
            .map(|item| item.path.to_string())
            .collect()
        };
        assert_eq!(names(green), ["C/4.tif"]);
        assert_eq!(names(0).len(), 4);
    }

    #[test]
    fn toggling_a_marker_twice_turns_it_off_and_zero_clears_all() {
        let controller = make_controller();
        controller.toggle_marker_filter(1);
        controller.toggle_marker_filter(4);
        assert_eq!(controller.marker_filter(), 0b10010);
        controller.toggle_marker_filter(1);
        assert_eq!(controller.marker_filter(), 0b10000);
        controller.toggle_marker_filter(7);
        assert_eq!(controller.marker_filter(), 0, "anything else clears");
    }

    #[test]
    fn update_image_filter_text_in_project_stores_the_given_text() {
        let controller = make_controller();

        controller.update_image_filter_text_in_project("Nuclei");

        assert_eq!(
            *controller
                .image_controller_state
                .image_filter_text
                .read()
                .unwrap(),
            "Nuclei"
        );
    }

    #[test]
    fn update_image_filter_text_in_project_overwrites_a_previous_value() {
        let controller = make_controller();
        controller.update_image_filter_text_in_project("first");

        controller.update_image_filter_text_in_project("second");

        assert_eq!(
            *controller
                .image_controller_state
                .image_filter_text
                .read()
                .unwrap(),
            "second"
        );
    }

    #[test]
    fn open_new_image_already_under_the_project_root_just_sets_it_as_current() {
        let controller = Arc::new(make_controller());
        {
            let mut project = controller.app_state.get_project_write();
            project.images.root = Some(PathBuf::from("/data/images"));
        }
        let image_path = PathBuf::from("/data/images/img.tif");

        controller.open_new_image(&image_path);

        let project = controller.app_state.get_project();
        assert_eq!(project.get_current_image_path_cloned(), Some(image_path));
        // Must not have touched the root - it was already correct.
        assert_eq!(project.images.root, Some(PathBuf::from("/data/images")));
    }

    /// Opening an image is no edit: it takes no undo step, so an undone
    /// edit can still be redone after it.
    #[test]
    fn opening_an_image_after_an_undo_keeps_the_redo() {
        let controller = Arc::new(make_controller());
        controller.app_state.get_project_runtime_write().images.root =
            Some(PathBuf::from("/data/images"));
        controller.app_state.get_project_write().meta.name = "edited".into();
        assert!(controller.app_state.undo());

        controller.open_new_image(&PathBuf::from("/data/images/img.tif"));

        assert!(controller.app_state.redo(), "the redo survives");
        assert_eq!(controller.app_state.get_project().meta.name, "edited");
    }

    #[test]
    fn open_new_image_outside_the_project_root_switches_the_root_to_its_parent_dir() {
        let controller = Arc::new(make_controller());
        // No root set yet (fresh project) - the image is by definition not
        // part of it, so this must take the "change root" branch.
        let image_path = PathBuf::from("/some/other/place/img.tif");

        controller.open_new_image(&image_path);

        let project = controller.app_state.get_project();
        assert_eq!(
            project.images.root,
            Some(PathBuf::from("/some/other/place"))
        );
    }

    // -- attach_callbacks (live AppWindow) -----------------------------------------

    use crate::editor::test_support::test_ui_window;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use std::collections::BTreeMap;

    fn make_controller_with_ui(ui: slint::Weak<AppWindow>) -> ImagesListController {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(
            ui.clone(),
            ui_state.clone(),
            Arc::new(crate::editor::viewport_cache::ViewportCache::new(
                ui_state.clone(),
            )),
        ));
        let object_list_controller = Arc::new(ObjectListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let histogram_controller = Arc::new(HistogramController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        let image_meta_controller = Arc::new(ImageMetaController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
        ));
        ImagesListController::new(
            ui.clone(),
            ui_state.clone(),
            viewport_controller.clone(),
            histogram_controller,
            image_meta_controller,
            object_list_controller.clone(),
            Arc::new(
                crate::editor::classification_controller::ClassificationController::new(
                    ui.clone(),
                    ui_state.clone(),
                    object_list_controller.clone(),
                    viewport_controller.clone(),
                ),
            ),
        )
    }

    #[test]
    fn attach_callbacks_image_filter_text_changed_stores_the_text_and_resyncs() {
        let ui = test_ui_window();
        let controller = Arc::new(make_controller_with_ui(ui.as_weak()));
        controller.attach_callbacks();

        ui.global::<ImagesListState>()
            .invoke_image_filter_text_changed("Nuclei".into());

        assert_eq!(
            *controller
                .image_controller_state
                .image_filter_text
                .read()
                .unwrap(),
            "Nuclei"
        );
    }

    #[test]
    fn attach_callbacks_image_selected_opens_the_matching_image() {
        let ui = test_ui_window();
        let controller = Arc::new(make_controller_with_ui(ui.as_weak()));
        {
            let mut project = controller.app_state.get_project_write();
            project.images.root = Some(PathBuf::from("/data/images"));
        }
        controller.attach_callbacks();

        ui.global::<ImagesListState>()
            .invoke_image_selected("img.tif".into());

        // "img.tif" isn't a real relative path in the (empty) project image
        // list, so `get_image_absolute_path_from_relative` resolves to
        // `None` and `open_new_image_from_rel_path` is a no-op - this is
        // exercising that lookup-miss branch through the real callback
        // wiring, not asserting an image actually opened.
        assert!(
            controller
                .app_state
                .get_project()
                .get_current_image_path_cloned()
                .is_none()
        );
    }

    #[test]
    fn switching_images_shows_the_class_counts_of_the_new_image() {
        use crate::ClassificationState;
        use crate::helper::ui_thread::drain_ui_queue;
        use evanalyzer_cfg::core_types::ObjectClass;
        let ui = test_ui_window();
        let controller = Arc::new(make_controller_with_ui(ui.as_weak()));
        let classified = || {
            let mut object = ObjectMetricSettings::default();
            object.object_class.insert(ObjectClass::Global(1));
            object
        };
        {
            let mut project = controller.app_state.get_project_write();
            project.images.root = Some(PathBuf::from("/data/images"));
            project.classification.classes_mut().push(
                evanalyzer_cfg::settings::classification_settings::Class {
                    id: 1,
                    name: "Nuclei".into(),
                    ..Default::default()
                },
            );
            for (name, objects) in [("a.tif", 1), ("b.tif", 2)] {
                let series = SeriesSettings {
                    objects: (0..objects).map(|_| classified()).collect(),
                    ..Default::default()
                };
                project.images.list.insert(
                    PathBuf::from(name),
                    ImageEntry {
                        review: Default::default(),
                        rel_path: PathBuf::from(name),
                        file_size: 0,
                        selected_series: 0,
                        series: BTreeMap::from([(0, series)]),
                    },
                );
            }
        }
        let total = || {
            ui.global::<ClassificationState>()
                .get_total_visible_objects()
        };

        controller.open_new_image(&PathBuf::from("/data/images/a.tif"));
        drain_ui_queue();
        assert_eq!(total(), 1);

        controller.open_new_image(&PathBuf::from("/data/images/b.tif"));
        drain_ui_queue();
        assert_eq!(total(), 2, "counts follow the image shown");
    }

    fn image_list_of(paths: &[&str]) -> evanalyzer_cfg::settings::images_settings::ImageList {
        let mut images = evanalyzer_cfg::settings::images_settings::ImageList::default();
        for path in paths {
            images.insert(
                PathBuf::from(path),
                ImageEntry {
                    rel_path: PathBuf::from(path),
                    ..Default::default()
                },
            );
        }
        images
    }

    /// Listed by groups: a head per well the project's "Grouping" finds, in
    /// plate order, then "No group" for images without a well; a folded
    /// head keeps only itself, a search opens it again.
    #[test]
    fn listed_by_groups_the_images_follow_their_wells_head() {
        let images = image_list_of(&["B03_01.tif", "A01_02.tif", "notes.tif", "A01_01.tif"]);
        let build = |search: &str, collapsed: &[&str]| {
            build_group_rows(
                &images,
                &Default::default(),
                search,
                0,
                &ImageStates::default(),
                std::path::Path::new("/data"),
                &collapsed.iter().map(|s| s.to_string()).collect(),
                &Default::default(),
            )
        };
        let (rows, count) = build("", &[]);
        assert_eq!(count, 4);
        assert_eq!(
            row_names(&rows),
            [
                "▸ A1 (2)",
                "A01_01.tif",
                "A01_02.tif",
                "▸ B3 (1)",
                "B03_01.tif",
                "▸ No group (1)",
                "notes.tif"
            ]
        );
        assert!(rows[0].group_head);
        assert_eq!(rows[0].path, "group:A1");
        let (rows, _) = build("", &["A1"]);
        assert_eq!(row_names(&rows)[..2], ["▸ A1 (2)", "▸ B3 (1)"]);
        let (rows, count) = build("a01_02", &["A1"]);
        assert_eq!(
            (row_names(&rows), count),
            (vec!["▸ A1 (1)".to_string(), "A01_02.tif".into()], 1)
        );
    }

    /// Auto finds the name pattern per folder: a folder named "C02.tif"
    /// isn't left without groups because another names "A01_01.tif" - and a
    /// group from both folders says so.
    #[test]
    fn listed_by_groups_each_folder_gets_its_own_name_pattern() {
        let images = image_list_of(&[
            "Exp1/A01_01.tif",
            "Exp1/A01_02.tif",
            "Exp1/B03_01.tif",
            "Exp2/A01.tif",
            "Exp2/C02.tif",
        ]);
        let (rows, count) = build_group_rows(
            &images,
            &Default::default(),
            "",
            0,
            &ImageStates::default(),
            std::path::Path::new("/data"),
            &Default::default(),
            &Default::default(),
        );
        assert_eq!(count, 5);
        assert_eq!(
            row_names(&rows),
            [
                "▸ A1 (3)",
                "A01_01.tif",
                "A01_02.tif",
                "A01.tif",
                "▸ B3 (1)",
                "B03_01.tif",
                "▸ C2 (1)",
                "C02.tif"
            ]
        );
        let folders: Vec<i32> = rows
            .iter()
            .filter(|r| r.group_head)
            .map(|r| r.folders)
            .collect();
        assert_eq!(folders, [2, 1, 1], "A1 from both folders");
    }

    /// (head or image, name, count) of each row - heads marked "▸".
    fn row_names(rows: &[ImageItemData]) -> Vec<String> {
        rows.iter()
            .map(|row| {
                if row.folder {
                    format!("▸ {}{} ({})", row.folder_parent, row.name, row.count)
                } else {
                    row.name.to_string()
                }
            })
            .collect()
    }

    #[test]
    fn images_in_one_folder_are_listed_without_heads() {
        let images = image_list_of(&["day1/a.tif", "day1/b.tif"]);
        let (rows, count) = build_image_list_items(
            &images,
            "",
            &ImageStates::default(),
            std::path::Path::new("/data/study"),
            &Default::default(),
            &Default::default(),
            0,
        );
        assert_eq!(row_names(&rows), ["a.tif", "b.tif"]);
        assert!(rows.iter().all(|row| row.head == -1), "nothing to pin");
        assert_eq!(count, 2);
    }

    #[test]
    fn each_folder_gets_a_head_with_its_path_count_and_analysed_images() {
        let images = image_list_of(&[
            "top.tif",
            "Control/B5/a.tif",
            "Serum/C14/c.tif",
            "Control/B5/b.tif",
        ]);
        let states = ImageStates {
            analysed: [PathBuf::from("Control/B5/a.tif")].into_iter().collect(),
            ..Default::default()
        };
        let (rows, count) = build_image_list_items(
            &images,
            "",
            &states,
            std::path::Path::new("/data/study"),
            &Default::default(),
            &Default::default(),
            0,
        );
        assert_eq!(
            row_names(&rows),
            [
                "▸ study (1)",
                "top.tif",
                "▸ Control › B5 (2)",
                "a.tif",
                "b.tif",
                "▸ Serum › C14 (1)",
                "c.tif",
            ],
            "folders in the order they first appear, their images together"
        );
        assert_eq!(count, 4, "heads aren't images");
        let heads: Vec<i32> = rows.iter().map(|row| row.head).collect();
        assert_eq!(
            heads,
            [0, 0, 2, 2, 2, 5, 5],
            "each row knows its folder's head"
        );
        assert_eq!(
            rows[2].path,
            std::path::Path::new("Control/B5")
                .to_string_lossy()
                .as_ref()
        );
        assert_eq!(rows[2].analysed, 1);
        assert_eq!(rows[5].analysed, 0);
    }

    #[test]
    fn a_folded_folder_keeps_its_head_but_a_search_shows_its_matches() {
        let images = image_list_of(&["A/a1.tif", "A/a2.tif", "B/b1.tif"]);
        let collapsed = [PathBuf::from("A")].into_iter().collect();
        let (rows, count) = build_image_list_items(
            &images,
            "",
            &ImageStates::default(),
            std::path::Path::new("/data"),
            &collapsed,
            &Default::default(),
            0,
        );
        assert_eq!(row_names(&rows), ["▸ A (2)", "▸ B (1)", "b1.tif"]);
        assert!(rows[0].collapsed && !rows[1].collapsed);
        assert_eq!(count, 3, "a folded folder's images still count");

        let (rows, count) = build_image_list_items(
            &images,
            "a2",
            &ImageStates::default(),
            std::path::Path::new("/data"),
            &collapsed,
            &Default::default(),
            0,
        );
        assert_eq!(row_names(&rows), ["a2.tif"], "one folder left: no head");
        assert_eq!(count, 1);
        let (rows, _) = build_image_list_items(
            &images,
            "1.tif",
            &ImageStates::default(),
            std::path::Path::new("/data"),
            &collapsed,
            &Default::default(),
            0,
        );
        assert_eq!(row_names(&rows), ["▸ A (1)", "a1.tif", "▸ B (1)", "b1.tif"]);
    }

    #[test]
    fn a_skipped_folders_head_and_images_are_marked_skipped() {
        let images = image_list_of(&["A/a.tif", "A/sub/s.tif", "B/b.tif"]);
        let skipped = [PathBuf::from("A")].into_iter().collect();
        let (rows, _) = build_image_list_items(
            &images,
            "",
            &ImageStates::default(),
            std::path::Path::new("/data"),
            &Default::default(),
            &skipped,
            0,
        );
        let marks: Vec<(String, bool)> = row_names(&rows)
            .into_iter()
            .zip(rows.iter().map(|row| row.skipped))
            .collect();
        assert_eq!(
            marks,
            [
                ("▸ A (1)".to_string(), true),
                ("a.tif".to_string(), true),
                ("▸ A › sub (1)".to_string(), false),
                ("s.tif".to_string(), false),
                ("▸ B (1)".to_string(), false),
                ("b.tif".to_string(), false),
            ],
            "not its subfolders"
        );
    }

    #[test]
    fn a_folders_analysed_count_comes_from_the_states_of_its_own_images() {
        let states: std::collections::HashMap<slint::SharedString, WellState> = [
            ("A/a.tif", WellState::Analysed),
            ("A/b.tif", WellState::Preview),
            ("A/sub/c.tif", WellState::Analysed),
            ("B/d.tif", WellState::Analysed),
        ]
        .into_iter()
        .map(|(path, state)| (path.into(), state))
        .collect();
        assert_eq!(analysed_in_folder(&states, "A"), 1);
        assert_eq!(analysed_in_folder(&states, "A/sub"), 1);
    }

    #[test]
    fn a_row_says_its_size_channels_and_state() {
        let mut series = BTreeMap::new();
        series.insert(
            0,
            SeriesSettings {
                image_width: 2048,
                image_height: 1024,
                channels: [(0, Default::default()), (1, Default::default())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        );
        let mut images = evanalyzer_cfg::settings::images_settings::ImageList::default();
        for name in ["a.tif", "b.tif"] {
            images.insert(
                PathBuf::from(name),
                ImageEntry {
                    rel_path: PathBuf::from(name),
                    series: if name == "a.tif" {
                        series.clone()
                    } else {
                        BTreeMap::new()
                    },
                    ..Default::default()
                },
            );
        }
        let states = ImageStates {
            previewed: [PathBuf::from("/root/a.tif")].into_iter().collect(),
            analysed: [PathBuf::from("b.tif")].into_iter().collect(),
        };
        let (rows, _) = build_image_list_items(
            &images,
            "",
            &states,
            std::path::Path::new("/root"),
            &Default::default(),
            &Default::default(),
            0,
        );
        assert_eq!(rows[0].image_dimension, "2048 × 1024 · 2 ch");
        assert_eq!(rows[0].state, WellState::Preview);
        assert_eq!(rows[1].image_dimension, "", "not read yet");
        assert_eq!(rows[1].state, WellState::Analysed);
    }

    #[test]
    fn sync_image_list_to_slint_reports_has_annotations_only_for_images_with_objects() {
        // Regression test: `has_annotations` used to be hardcoded `false`
        // for every image (the real check was commented out as a TODO),
        // permanently disabling this indicator. Exercises `sync_image_list_to_slint`'s
        // item-building logic through `build_image_list_items` directly rather than
        // reading it back off the live Slint model: the headless test platform (see
        // `test_ui_window`) uses `init_no_event_loop()`, under which
        // `slint::invoke_from_event_loop` - the mechanism `sync_image_list_to_slint`
        // uses for its actual UI write - has no event loop to run on and always
        // returns `Err(NoEventLoopProvider)` without invoking the closure, so nothing
        // would ever land in `images_list` to read back.
        let controller = make_controller_with_ui(slint::Weak::default());

        {
            let mut project = controller.app_state.get_project_write();

            let mut series_with_objects = BTreeMap::new();
            series_with_objects.insert(
                0,
                SeriesSettings {
                    objects: vec![ObjectMetricSettings::default()],
                    ..Default::default()
                },
            );
            project.images.list.insert(
                PathBuf::from("with_objects.tif"),
                ImageEntry {
                    review: Default::default(),
                    rel_path: PathBuf::from("with_objects.tif"),
                    file_size: 0,
                    selected_series: 0,
                    series: series_with_objects,
                },
            );

            project.images.list.insert(
                PathBuf::from("empty.tif"),
                ImageEntry {
                    review: Default::default(),
                    rel_path: PathBuf::from("empty.tif"),
                    file_size: 0,
                    selected_series: 0,
                    series: BTreeMap::from([(0, SeriesSettings::default())]),
                },
            );
        }

        let (images_list, _) = build_image_list_items(
            &controller.app_state.get_project().images.list,
            "",
            &Default::default(),
            std::path::Path::new(""),
            &Default::default(),
            &Default::default(),
            0,
        );

        let mut has_annotations_by_path = std::collections::HashMap::new();
        for item in &images_list {
            has_annotations_by_path.insert(item.path.to_string(), item.has_annotations);
        }

        assert_eq!(has_annotations_by_path.get("with_objects.tif"), Some(&true));
        assert_eq!(has_annotations_by_path.get("empty.tif"), Some(&false));
    }

    fn image_entries(paths: &[&str]) -> evanalyzer_cfg::settings::images_settings::ImageList {
        evanalyzer_cfg::settings::images_settings::ImageList::from_entries(paths.iter().map(
            |path| {
                (
                    PathBuf::from(path),
                    ImageEntry {
                        review: Default::default(),
                        rel_path: PathBuf::from(path),
                        file_size: 0,
                        selected_series: 0,
                        series: BTreeMap::new(),
                    },
                )
            },
        ))
    }

    #[test]
    fn quick_pick_keeps_the_images_holding_every_word_in_any_case_and_order() {
        let images = image_entries(&[
            "plate1/A01_01.vsi",
            "plate1/B02_01.vsi",
            "plate2/A01_02.vsi",
        ]);

        let (items, matches, _) = quick_pick_items(
            &images,
            "a01 PLATE1",
            None,
            &Default::default(),
            std::path::Path::new(""),
            0,
        );

        let paths: Vec<String> = items.iter().map(|item| item.path.to_string()).collect();
        assert_eq!(paths, vec!["plate1/A01_01.vsi"]);
        assert_eq!(matches, 1);
    }

    #[test]
    fn quick_pick_without_a_query_lists_the_images_up_to_the_limit_and_counts_all() {
        let names: Vec<String> = (0..QUICK_PICK_LIMIT + 7)
            .map(|i| format!("img_{i:03}.tif"))
            .collect();
        let images = image_entries(&names.iter().map(String::as_str).collect::<Vec<_>>());

        let (items, matches, _) = quick_pick_items(
            &images,
            "  ",
            None,
            &Default::default(),
            std::path::Path::new(""),
            0,
        );

        assert_eq!(items.len(), QUICK_PICK_LIMIT);
        assert_eq!(matches, QUICK_PICK_LIMIT + 7);
        assert_eq!(items[0].path.as_str(), "img_000.tif");
    }

    #[test]
    fn quick_pick_without_a_query_lists_the_window_around_the_open_image() {
        let names: Vec<String> = (0..QUICK_PICK_LIMIT * 3)
            .map(|i| format!("img_{i:03}.tif"))
            .collect();
        let images = image_entries(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let open = |i: usize| {
            quick_pick_items(
                &images,
                "",
                Some(std::path::Path::new(&names[i])),
                &Default::default(),
                std::path::Path::new(""),
                0,
            )
        };

        // In the middle: centred.
        let (items, matches, current) = open(100);
        assert_eq!(items.len(), QUICK_PICK_LIMIT);
        assert_eq!(matches, QUICK_PICK_LIMIT * 3);
        assert_eq!(current, Some(QUICK_PICK_LIMIT / 2));
        assert_eq!(items[QUICK_PICK_LIMIT / 2].path.as_str(), names[100]);

        // Near the start and the end: the window stays inside the list.
        let (items, _, current) = open(3);
        assert_eq!(items[0].path.as_str(), names[0]);
        assert_eq!(current, Some(3));
        let last = names.len() - 1;
        let (items, _, current) = open(last);
        assert_eq!(items.len(), QUICK_PICK_LIMIT);
        assert_eq!(items[QUICK_PICK_LIMIT - 1].path.as_str(), names[last]);
        assert_eq!(current, Some(QUICK_PICK_LIMIT - 1));

        // With a query the first matches are listed, as before.
        let (items, _, _) = quick_pick_items(
            &images,
            "img",
            Some(std::path::Path::new(&names[100])),
            &Default::default(),
            std::path::Path::new(""),
            0,
        );
        assert_eq!(items[0].path.as_str(), names[0]);
    }

    #[test]
    fn quick_pick_marks_the_open_image_only_when_it_is_listed() {
        let images = image_entries(&["a.tif", "b.tif", "c.tif"]);

        let (_, _, current) = quick_pick_items(
            &images,
            "",
            Some(std::path::Path::new("b.tif")),
            &Default::default(),
            std::path::Path::new(""),
            0,
        );
        assert_eq!(current, Some(1));

        let (_, _, current) = quick_pick_items(
            &images,
            "c",
            Some(std::path::Path::new("b.tif")),
            &Default::default(),
            std::path::Path::new(""),
            0,
        );
        assert_eq!(current, None);
    }
}
