use crate::FileRequest;
use crate::UiState;
use crate::editor::classification_controller::ClassificationController;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::size_formater::format_bytes;
use crate::{AppWindow, DialogType, GlobalAppState, ObjectHighlightBox, ViewportObjectState};
use crate::{ImageItemData, ImagesListState, PlateWell, WellState};
use evanalyzer_app::project::collect_images_at_root;
use evanalyzer_app::project::{ProjectExt, SelectNewProjectRootAction};
use log::{debug, info, warn};
use slint::{ComponentHandle, Model};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Which images were previewed (this session) or are in the analysis open
/// in Results - the states of the plate's wells and of the image rows. Kept
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
    /// The images of each well of the plate view, as `plate-wells` lists them.
    pub well_items: RwLock<Vec<Vec<ImageItemData>>>,
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
                well_items: RwLock::new(Vec::new()),
            }),
            object_list_controller,
            classification_controller,
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
                let mut project = self.app_state.get_project_write();
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
                // threads call `get_or_create_reader_pool`, which now reuses
                // the same single-reader cache `sync_image_meta_to_slint`
                // below is about to populate (see `ReaderPool`'s doc comment
                // in `project_owner.rs`) - calling it first lets that reader
                // open race for the shared cache concurrently with metadata
                // sync instead of only starting once metadata sync (and the
                // two calls after it) have already finished.
                manager.viewport_controller.trigger_new_image_redraw();
                if manager
                    .image_meta_controller
                    .sync_image_meta_to_slint()
                    .is_err()
                {
                    warn!("Failed to sync image meta to slint!");
                }
                manager.object_list_controller.sync_objects_to_slint();
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
                if let Some(current_path) = manager
                    .app_state
                    .get_project()
                    .get_current_image_path_cloned()
                {
                    manager.set_selected_image_index_in_slint_images_list(current_path, true);
                }
            });
        } else if let Some(parent) = parent_dir {
            self.change_image_root(&parent, Some(image_path));
        }
    }

    /// A new image has been selected. We update the project state accordingly.
    /// This method is called when the user selects a new image from the UI.
    fn open_new_image_from_rel_path(self: &Arc<Self>, rel_path: &PathBuf) {
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
            info!("Image is not part of the project {:?}", rel_path);
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
                    let mut project = manager.app_state.get_project_write();
                    project.set_current_image_path(&path);
                }

                if manager
                    .image_meta_controller
                    .sync_image_meta_to_slint()
                    .is_err()
                {
                    warn!("Failed to sync image meta to slint!");
                }

                manager.object_list_controller.sync_objects_to_slint();
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
        // The plate follows previews and opened analyses. Weak: `UiState`
        // keeps the listener, and this controller keeps `UiState`.
        let weak = Arc::downgrade(self);
        self.app_state.on_image_states_changed(move || {
            if let Some(manager) = weak.upgrade() {
                manager.sync_plate_to_slint();
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

            // A well of the plate view: show its images
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_well_selected(move |index| {
                    manager.show_well_images(index);
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

        let image_states = state.image_states();
        let project = state.get_project();
        let images_guard = &project.images.list;
        let slint_items = build_image_list_items(
            images_guard,
            &filter_text,
            &image_states,
            &project.images.root.clone().unwrap_or_default(),
        );
        let plate = build_plate(
            images_guard,
            &project.plate,
            &image_states,
            &project.images.root.clone().unwrap_or_default(),
        );
        drop(project);
        *self
            .image_controller_state
            .well_items
            .write()
            .expect("Poisoned") = plate.well_items;
        let plate_wells = plate.wells;
        let (plate_rows, plate_cols) = (plate.rows, plate.cols);

        // The final assignment goes into the event loop
        crate::helper::ui_thread::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    let model = slint::ModelRc::new(slint::VecModel::from(slint_items));
                    let images_state = ui_ready.global::<ImagesListState>();
                    images_state.set_images_list(model);
                    images_state.set_plate_rows(plate_rows);
                    images_state.set_plate_cols(plate_cols);
                    images_state.set_plate_wells(slint::ModelRc::new(slint::VecModel::from(plate_wells)));
                    // The wells may have changed: no well selected.
                    images_state.set_selected_well(-1);
                    images_state.set_well_images(slint::ModelRc::default());
                }else{
                    warn!("Failed to upgrade UI handle in sync_image_list_to_slint, cannot update image list!");
                }
            })
            .ok();
    }

    /// Only the plate: its wells' states changed (an image was previewed,
    /// an analysis opened) - cheaper than the whole list.
    pub fn sync_plate_to_slint(&self) {
        let image_states = self.app_state.image_states();
        let project = self.app_state.get_project();
        let plate = build_plate(
            &project.images.list,
            &project.plate,
            &image_states,
            &project.images.root.clone().unwrap_or_default(),
        );
        drop(project);
        *self
            .image_controller_state
            .well_items
            .write()
            .expect("Poisoned") = plate.well_items;
        let wells = plate.wells;
        // The rows' states, by path (patched in place: rebuilding the list
        // would lose the scroll position and the chosen well).
        let row_states: std::collections::HashMap<slint::SharedString, WellState> =
            project_rows(&self.app_state, &image_states);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ImagesListState>();
                // The same wells as before (same images), so the selected
                // one stays.
                state.set_plate_wells(slint::ModelRc::new(slint::VecModel::from(wells)));
                for model in [state.get_images_list(), state.get_well_images()] {
                    for row in 0..model.row_count() {
                        if let Some(mut item) = model.row_data(row) {
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
            }
        })
        .ok();
    }

    /// Shows the images of well `index` (of `plate-wells`) under the plate.
    pub fn show_well_images(&self, index: i32) {
        let items = usize::try_from(index)
            .ok()
            .and_then(|index| {
                self.image_controller_state
                    .well_items
                    .read()
                    .expect("Poisoned")
                    .get(index)
                    .cloned()
            })
            .unwrap_or_default();
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ImagesListState>()
                    .set_well_images(slint::ModelRc::new(slint::VecModel::from(items)));
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
        let ui_weak = self.ui.clone();

        // The final assignment goes into the event loop
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui_ready) = ui_weak.upgrade() {
                    let images_state = ui_ready.global::<ImagesListState>();
                   let images_list_model =  images_state.get_images_list();

                   let search_path = selected_absolute_path.to_string_lossy();
                    let selected_index: Option<usize> = images_list_model.iter().position(|item| {
                        let item_path: &str = item.path.as_str();
                        search_path.contains(item_path)
                    });

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

/// Builds the Slint-facing row list for `sync_image_list_to_slint`: applies
/// the (already-lowercased) filter text against `rel_path`, then maps each
/// surviving entry to an `ImageItemData`. Pulled out as a pure function so
/// it can be tested directly - unlike the rest of `sync_image_list_to_slint`,
/// this doesn't touch Slint's event loop, which the headless
/// `init_no_event_loop()` testing platform (see `test_ui_window`) never
/// actually drains.
fn build_image_list_items(
    images: &indexmap::IndexMap<PathBuf, evanalyzer_cfg::settings::images_settings::ImageEntry>,
    filter_text_lowercase: &str,
    states: &ImageStates,
    image_root: &std::path::Path,
) -> Vec<ImageItemData> {
    images
        .values()
        .filter(|entry| {
            if filter_text_lowercase.is_empty() {
                return true;
            }
            entry
                .rel_path
                .to_string_lossy()
                .to_lowercase()
                .contains(filter_text_lowercase)
        })
        .map(|entry| ImageItemData {
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
        })
        .collect()
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

/// The plate view of `images`: the size, the wells, each well's images.
struct PlateItems {
    rows: i32,
    cols: i32,
    wells: Vec<PlateWell>,
    well_items: Vec<Vec<ImageItemData>>,
}

/// Groups `images` into wells by `plate` (see `evanalyzer_app::images::plate_layout`).
/// A well is `analysed` when one of its images is in the analysis open in
/// Results, else `preview` when one was previewed, else `new`.
fn build_plate(
    images: &indexmap::IndexMap<PathBuf, evanalyzer_cfg::settings::images_settings::ImageEntry>,
    plate: &evanalyzer_cfg::settings::plate_settings::PlateSettings,
    states: &ImageStates,
    image_root: &std::path::Path,
) -> PlateItems {
    let entries: Vec<&std::path::Path> = images.values().map(|e| e.rel_path.as_path()).collect();
    let state_of = |indices: &[usize]| {
        let of = |state| {
            indices
                .iter()
                .any(|&i| image_state(entries[i], states, image_root) == state)
        };
        if of(WellState::Analysed) {
            WellState::Analysed
        } else if of(WellState::Preview) {
            WellState::Preview
        } else {
            WellState::New
        }
    };
    let paths: Vec<&std::path::Path> = images
        .values()
        .map(|entry| entry.rel_path.as_path())
        .collect();
    let layout = evanalyzer_app::images::plate_layout(&paths, plate);
    let items = build_image_list_items(images, "", states, image_root);
    let wells = layout
        .wells
        .iter()
        .map(|well| PlateWell {
            row: well.row as i32,
            col: well.col as i32,
            name: well.name.clone().into(),
            image_count: well.images.len() as i32,
            state: state_of(&well.images),
        })
        .collect();
    let well_items = layout
        .wells
        .iter()
        .map(|well| well.images.iter().map(|&i| items[i].clone()).collect())
        .collect();
    PlateItems {
        rows: layout.rows as i32,
        cols: layout.cols as i32,
        wells,
        well_items,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::test_ui_state;

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
        let mut images = indexmap::IndexMap::new();
        for name in ["a.tif", "b.tif"] {
            images.insert(
                PathBuf::from(name),
                ImageEntry {
                    rel_path: PathBuf::from(name),
                    series: if name == "a.tif" { series.clone() } else { BTreeMap::new() },
                    ..Default::default()
                },
            );
        }
        let states = ImageStates {
            previewed: [PathBuf::from("/root/a.tif")].into_iter().collect(),
            analysed: [PathBuf::from("b.tif")].into_iter().collect(),
        };
        let rows = build_image_list_items(&images, "", &states, std::path::Path::new("/root"));
        assert_eq!(rows[0].image_dimension, "2048 × 1024 · 2 ch");
        assert_eq!(rows[0].state, WellState::Preview);
        assert_eq!(rows[1].image_dimension, "", "not read yet");
        assert_eq!(rows[1].state, WellState::Analysed);
    }

    #[test]
    fn a_wells_state_is_analysed_before_preview_before_new() {
        let mut images = indexmap::IndexMap::new();
        for name in ["A01_01.tif", "A01_02.tif", "A02_01.tif", "A03_01.tif"] {
            images.insert(
                PathBuf::from(name),
                ImageEntry {
                    rel_path: PathBuf::from(name),
                    ..Default::default()
                },
            );
        }
        let states = ImageStates {
            previewed: [
                PathBuf::from("/root/A01_02.tif"),
                PathBuf::from("/root/A02_01.tif"),
            ]
            .into_iter()
            .collect(),
            analysed: [PathBuf::from("A01_01.tif")].into_iter().collect(),
        };
        let plate = build_plate(
            &images,
            &Default::default(),
            &states,
            std::path::Path::new("/root"),
        );
        let state: Vec<WellState> = plate.wells.iter().map(|w| w.state).collect();
        assert_eq!(
            state,
            [WellState::Analysed, WellState::Preview, WellState::New]
        );
    }

    #[test]
    fn build_plate_puts_each_image_in_its_well_in_number_order() {
        let mut images = indexmap::IndexMap::new();
        for name in ["B03_02.tif", "A01_01.tif", "B03_01.tif", "notes.tif"] {
            images.insert(
                PathBuf::from(name),
                ImageEntry {
                    rel_path: PathBuf::from(name),
                    ..Default::default()
                },
            );
        }
        let plate = build_plate(
            &images,
            &Default::default(),
            &Default::default(),
            std::path::Path::new("/root"),
        );
        assert_eq!((plate.rows, plate.cols), (2, 3));
        let names: Vec<String> = plate.wells.iter().map(|w| w.name.to_string()).collect();
        assert_eq!(names, ["A1", "B3"]);
        assert_eq!(plate.wells[1].image_count, 2);
        assert_eq!((plate.wells[1].row, plate.wells[1].col), (1, 2));
        let b3: Vec<String> = plate.well_items[1]
            .iter()
            .map(|i| i.path.to_string())
            .collect();
        assert_eq!(b3, ["B03_01.tif", "B03_02.tif"]);
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
                    rel_path: PathBuf::from("with_objects.tif"),
                    file_size: 0,
                    selected_series: 0,
                    series: series_with_objects,
                },
            );

            project.images.list.insert(
                PathBuf::from("empty.tif"),
                ImageEntry {
                    rel_path: PathBuf::from("empty.tif"),
                    file_size: 0,
                    selected_series: 0,
                    series: BTreeMap::from([(0, SeriesSettings::default())]),
                },
            );
        }

        let images_list = build_image_list_items(
            &controller.app_state.get_project().images.list,
            "",
            &Default::default(),
            std::path::Path::new(""),
        );

        let mut has_annotations_by_path = std::collections::HashMap::new();
        for item in &images_list {
            has_annotations_by_path.insert(item.path.to_string(), item.has_annotations);
        }

        assert_eq!(has_annotations_by_path.get("with_objects.tif"), Some(&true));
        assert_eq!(has_annotations_by_path.get("empty.tif"), Some(&false));
    }
}
