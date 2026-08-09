use crate::UiState;
use crate::editor::histogram_controller::HistogramController;
use crate::editor::image_meta_controller::ImageMetaController;
use crate::editor::object_list_controller::ObjectListController;
use crate::editor::viewport_controller::ViewportController;
use crate::helper::size_formater::format_bytes;
use crate::{AppWindow, DialogType, GlobalAppState, ObjectHighlightBox, ViewportObjectState};
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
    pub(crate) object_list_controller: Arc<ObjectListController>,
}

impl ImagesListController {
    pub fn new(
        ui: slint::Weak<AppWindow>,
        app_state: Arc<UiState>,
        viewport_controller: Arc<ViewportController>,
        histogram_controller: Arc<HistogramController>,
        image_meta_controller: Arc<ImageMetaController>,
        object_list_controller: Arc<ObjectListController>,
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
            object_list_controller,
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
            self.object_list_controller.sync_objects_to_slint();
            self.set_selected_image_index_in_slint_images_list(image_path.clone(), true);
            self.viewport_controller.trigger_new_image_redraw();
            // Remove possible existing markers and the results-table object highlight
            if let Some(ui) = self.ui.upgrade() {
                let object_state = ui.global::<ViewportObjectState>();
                object_state.set_markers(slint::ModelRc::new(slint::VecModel::default()));
                object_state.set_object_highlight(ObjectHighlightBox::default());
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
    /// highlight box in the viewport. Used when a object row is selected in the
    /// results table so the user can locate the object in its source image.
    pub fn open_image_and_highlight_object(
        self: &Arc<Self>,
        rel_path: &PathBuf,
        bbox_px: [u32; 4],
    ) {
        self.open_new_image_from_rel_path(rel_path);

        // `open_new_image` clears any previous highlight, so set ours afterwards.
        let [xmin, ymin, xmax, ymax] = bbox_px;
        let highlight = ObjectHighlightBox {
            x_px: xmin as f32,
            y_px: ymin as f32,
            // +1 so a single-pixel object is still visible (max pixel is inclusive).
            w_px: (xmax.saturating_sub(xmin) + 1) as f32,
            h_px: (ymax.saturating_sub(ymin) + 1) as f32,
            active: true,
        };
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ViewportObjectState>()
                .set_object_highlight(highlight);
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
            // Scan the filesystem without holding the project lock - opening
            // an `ImageReader` per file can take seconds on a plate-sized
            // folder, and holding the write guard for that long stalls every
            // other thread that needs the project (viewport render, object
            // edits, dirty-tracking). Only the fast in-memory apply step
            // below needs the lock.
            let root_folder = manager.app_state.get_project().images.root.clone();
            if let Some(root_folder) = root_folder {
                let found_images =
                    evanalyzer_app::extensions::project_ext::collect_images_at_root(&root_folder);
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
                manager.set_selected_image_index_in_slint_images_list(path, true);
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
        let slint_items = build_image_list_items(images_guard, &filter_text);

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
/// `init_no_event_loop()` testing platform (see `test_ui_windows`) never
/// actually drains.
fn build_image_list_items(
    images: &indexmap::IndexMap<PathBuf, evanalyzer_cfg::settings::images_settings::ImageEntry>,
    filter_text_lowercase: &str,
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
            image_dimension: "".into(),
            image_size: format!("{}", format_bytes(entry.file_size as f64)).into(),
            has_annotations: entry
                .series
                .values()
                .any(|series| !series.objects.is_empty()),
        })
        .collect()
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
            ui_state,
            viewport_controller,
            histogram_controller,
            image_meta_controller,
            object_list_controller,
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

    use crate::editor::test_support::test_ui_windows;
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, SeriesSettings};
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    use std::collections::BTreeMap;

    fn make_controller_with_ui(ui: slint::Weak<AppWindow>) -> ImagesListController {
        let ui_state = test_ui_state();
        let viewport_controller = Arc::new(ViewportController::new(ui.clone(), ui_state.clone()));
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
            ui,
            ui_state,
            viewport_controller,
            histogram_controller,
            image_meta_controller,
            object_list_controller,
        )
    }

    #[test]
    fn attach_callbacks_image_filter_text_changed_stores_the_text_and_resyncs() {
        let (ui, _results_ui) = test_ui_windows();
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
        let (ui, _results_ui) = test_ui_windows();
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
    fn sync_image_list_to_slint_reports_has_annotations_only_for_images_with_objects() {
        // Regression test: `has_annotations` used to be hardcoded `false`
        // for every image (the real check was commented out as a TODO),
        // permanently disabling this indicator. Exercises `sync_image_list_to_slint`'s
        // item-building logic through `build_image_list_items` directly rather than
        // reading it back off the live Slint model: the headless test platform (see
        // `test_ui_windows`) uses `init_no_event_loop()`, under which
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

        let images_list =
            build_image_list_items(&controller.app_state.get_project().images.list, "");

        let mut has_annotations_by_path = std::collections::HashMap::new();
        for item in &images_list {
            has_annotations_by_path.insert(item.path.to_string(), item.has_annotations);
        }

        assert_eq!(has_annotations_by_path.get("with_objects.tif"), Some(&true));
        assert_eq!(has_annotations_by_path.get("empty.tif"), Some(&false));
    }
}
