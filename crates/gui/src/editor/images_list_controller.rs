use crate::UiState;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::roi_list_controller::RoiListController;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::size_formater::format_bytes;
use crate::{AppWindow, DialogType, GlobalAppState, RoiHighlightBox, ViewportRoiState};
use crate::{ImageItemData, ImagesListState};
use evanalyzer_app::extensions::project_ext::{ProjectExt, SelectNewProjectRootAction};
use log::{debug, info, warn};
use slint::{ComponentHandle, Model};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

pub struct ImagesListControllerState {
    pub image_filter_text: RwLock<String>,
}
pub struct ImagesListController {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) app_state: Arc<UiState>,
    pub(crate) viewport_controller: Arc<ViewportController>,
    pub(crate) _histogram_controller: Arc<HistogramController>,
    pub(crate) image_meta_controller: Arc<ImageMetaController>,
    pub(crate) image_controller_state: Arc<ImagesListControllerState>,
    pub(crate) roi_list_controller: Arc<RoiListController>,
}

impl ImagesListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        histogram_controller: Arc<HistogramController>,
        image_meta_controller: Arc<ImageMetaController>,
        roi_list_controller: Arc<RoiListController>,
    ) -> Self {
        Self {
            ui,
            app_state,
            viewport_controller,
            _histogram_controller: histogram_controller,
            image_meta_controller,
            image_controller_state: Arc::new(ImagesListControllerState {
                image_filter_text: RwLock::new(String::new()),
            }),
            roi_list_controller,
        }
    }

    /// A new image has been selected. We update the project state accordingly.
    /// This method is called when the user selects a new image from the UI.
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
            if self
                .image_meta_controller
                .sync_image_meta_to_slint()
                .is_err()
            {
                warn!("Failed to sync image meta to slint!");
            }
            self.roi_list_controller.sync_rois_to_slint();
            self.set_selected_image_index_in_slint_images_list(image_path.clone());
            self.viewport_controller.trigger_new_image_redraw();
            // Remove possible existing markers and the results-table ROI highlight
            if let Some(ui) = self.ui.upgrade() {
                let roi_state = ui.global::<ViewportRoiState>();
                roi_state.set_markers(slint::ModelRc::new(slint::VecModel::default()));
                roi_state.set_roi_highlight(RoiHighlightBox::default());
            }
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
    /// highlight box in the viewport. Used when a ROI row is selected in the
    /// results table so the user can locate the object in its source image.
    pub fn open_image_and_highlight_roi(self: &Arc<Self>, rel_path: &PathBuf, bbox_px: [u32; 4]) {
        self.open_new_image_from_rel_path(rel_path);

        // `open_new_image` clears any previous highlight, so set ours afterwards.
        let [xmin, ymin, xmax, ymax] = bbox_px;
        let highlight = RoiHighlightBox {
            x_px: xmin as f32,
            y_px: ymin as f32,
            // +1 so a single-pixel ROI is still visible (max pixel is inclusive).
            w_px: (xmax.saturating_sub(xmin) + 1) as f32,
            h_px: (ymax.saturating_sub(ymin) + 1) as f32,
            active: true,
        };
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ViewportRoiState>().set_roi_highlight(highlight);
        }
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

        slint::invoke_from_event_loop(move || {
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
    pub fn set_new_image_root(self: &Arc<Self>, new_root: &PathBuf) {
        let ui_weak = self.ui.clone();
        let result = self
            .app_state
            .get_project_write()
            .select_new_images_root_with_check(&new_root);

        if result == SelectNewProjectRootAction::ImageNotFound {
            // Images not found, show the Missing image dialog
            slint::invoke_from_event_loop(move || {
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
            slint::invoke_from_event_loop(move || {
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
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<ImagesListState>().set_is_scanning(true);
            }
        })
        .ok();

        let ui_weak = self.ui.clone();
        let manager = self.clone();

        std::thread::spawn(move || {
            {
                let mut project = manager.app_state.get_project_write();
                project.scan_image_folder_and_add();
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

                manager.roi_list_controller.sync_rois_to_slint();
                manager.set_selected_image_index_in_slint_images_list(path);
                manager.viewport_controller.trigger_new_image_redraw();
            }

            // Hide spinner
            slint::invoke_from_event_loop(move || {
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
                        manager.set_selected_image_index_in_slint_images_list(image_path);
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
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        manager.change_image_root(&path, None);
                    }
                });

            // Set new image root. Images stay in the list only the root path is changed and it is checked if images in the new root path are found
            let manager = Arc::clone(self);
            ui.global::<ImagesListState>()
                .on_new_image_root_folder_clicked(move || {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        manager.set_new_image_root(&path);
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

        let images_guard = &state.get_project().images.list;

        let slint_items: Vec<ImageItemData> = images_guard
            .values()
            .filter(|entry| {
                if filter_text.is_empty() {
                    return true;
                }
                entry
                    .rel_path
                    .to_string_lossy()
                    .to_lowercase()
                    .contains(&filter_text)
            })
            .map(|entry| ImageItemData {
                name: entry
                    .rel_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
                    .into(),
                path: entry.rel_path.to_string_lossy().to_string().into(),
                image_dimension: "".into(),
                image_size: format!("{}", format_bytes(entry.file_size as f64)).into(),
                // TODO: /*!entry.rois.lock().expect("poisoned").is_empty(),*/
                has_annotations: false,
            })
            .collect();

        // The final assignment goes into the event loop
        slint::invoke_from_event_loop(move || {
                if let Some(ui_ready) = ui_weak.upgrade() {
                    let model = slint::ModelRc::new(slint::VecModel::from(slint_items));
                    ui_ready.global::<ImagesListState>().set_images_list(model);
                }else{
                    warn!("Failed to upgrade UI handle in sync_image_list_to_slint, cannot update image list!");
                }
            })
            .ok();
    }

    pub(crate) fn set_selected_image_index_in_slint_images_list(
        &self,
        selected_absolute_path: PathBuf,
    ) {
        let ui_weak = self.ui.clone();

        // The final assignment goes into the event loop
        slint::invoke_from_event_loop(move || {
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
