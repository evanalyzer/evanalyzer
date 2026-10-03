//! Shared test-only fixtures for controller unit tests.
//!
//! Controllers hold a `slint::Weak<AppWindow>` and talk to the live UI
//! through it, but every `sync_*_to_slint` method already has to tolerate
//! that handle being dead (the window was closed while a background task
//! was in flight) - `ui.upgrade()` returning `None` is a normal, handled
//! case, not a bug. That means a controller's *project-state* mutations can
//! be exercised and asserted on without ever constructing a real `AppWindow`
//! (which would need a Slint platform/backend): build the controller with a
//! dead `Weak::default()`, call the method, and inspect `UiState`'s project
//! afterwards. The Slint-sync half of the method still runs - it just
//! becomes a no-op, exactly like it does in production when the window is
//! gone.
use crate::{AppWindow, ResultsWindow, UiState};
use evanalyzer_app::project::ProjectExt;
use evanalyzer_app::project::ProjectOwner;
use evanalyzer_app::project::ProjectWithRuntime;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, ImageEntry, PixelSizeSettings, SeriesSettings,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Builds a real `UiState` backed by a fresh, empty, in-memory project - no
/// Slint window/platform required.
pub(crate) fn test_ui_state() -> Arc<UiState> {
    test_ui_state_with_project(ProjectWithRuntime::default())
}

/// Same as [`test_ui_state`], but seeded with `project` instead of an empty
/// default - see [`project_with_one_image`] for a ready-made single-image
/// fixture.
/// A settings file of its own for every test `UiState`, so a test saving a
/// preference never touches the user's real `settings.json` (nor another
/// test's).
pub(crate) fn temp_settings_file() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "evanalyzer-test-settings-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

pub(crate) fn test_ui_state_with_project(project: ProjectWithRuntime) -> Arc<UiState> {
    let owner = ProjectOwner::new();
    let handle = owner.handle();
    *handle.get_project_write() = project;
    let mut ui_state = UiState::new(handle, slint::Weak::default(), slint::Weak::default());
    ui_state.app_settings_file = temp_settings_file();
    Arc::new(ui_state)
}

/// A minimal project with one 2x2, 2-channel image set as "current" (i.e.
/// what `get_current_image_settings`/`with_current_series_mut` resolve to) -
/// mirrors the equivalent private helper in
/// `evanalyzer_app::workspace::extensions::project_ext`'s own test module, since
/// several controller methods only do anything when a "current image" is
/// set.
pub(crate) fn project_with_one_image() -> ProjectWithRuntime {
    let mut project = ProjectWithRuntime::default();
    let rel_path = PathBuf::from("img.tif");

    let channels = BTreeMap::from([
        (
            0,
            ChannelSettings {
                name: "Ch0".into(),
                emission_wave_length: Some(488.0),
                visible: None,
                histogram: None,
            },
        ),
        (
            1,
            ChannelSettings {
                name: "Ch1".into(),
                emission_wave_length: Some(561.0),
                visible: None,
                histogram: None,
            },
        ),
    ]);
    let series = SeriesSettings {
        selected_channel: None,
        image_width: 2,
        image_height: 2,
        channels,
        pixel_sizes: PixelSizeSettings {
            x: 0.5,
            y: 0.5,
            z: 1.0,
        },
        z_stack: None,
        t_stack: None,
        objects: Vec::new(),
    };
    project.images.list.insert(
        rel_path.clone(),
        ImageEntry {
            rel_path: rel_path.clone(),
            file_size: 16,
            selected_series: 0,
            series: BTreeMap::from([(0, series)]),
        },
    );
    project.set_current_image_path(&rel_path);
    project
}

/// The 4D multi-channel OME-TIFF fixture shared with the core tests.
pub(crate) fn fixture_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../core/tests/multi-channel-4D-series.ome.tif")
        .canonicalize()
        .expect("fixture image exists")
}

/// Like [`project_with_one_image`], but the current image is a real file
/// ([`fixture_image_path`]) that can be opened, read and rendered.
pub(crate) fn project_with_fixture_image() -> ProjectWithRuntime {
    project_with_image_file(fixture_image_path())
}

/// A single-channel grayscale image from the core test fixtures.
pub(crate) fn grayscale_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../core/tests/slice_Z0_C0_T0.tif")
        .canonicalize()
        .expect("fixture image exists")
}

/// Like [`project_with_fixture_image`], for the image file at `path`.
pub(crate) fn project_with_image_file(path: PathBuf) -> ProjectWithRuntime {
    let mut project = project_with_one_image();
    let file_name = PathBuf::from(path.file_name().unwrap());
    let mut entry = project
        .images
        .list
        .shift_remove(&PathBuf::from("img.tif"))
        .unwrap();
    entry.rel_path = file_name.clone();
    project.images.list.insert(file_name, entry);
    project.images.root = Some(path.parent().unwrap().to_path_buf());
    project.set_current_image_path(&path);
    project
}

/// Ensures the Slint headless testing platform is set up on the *calling
/// thread* before constructing a real `AppWindow`/`ResultsWindow`.
///
/// `i_slint_backend_testing::init_no_event_loop()` panics
/// ("platform already initialized") if called twice on the same OS thread,
/// and Rust's default test harness reuses a fixed pool of worker threads
/// across many `#[test]` functions rather than spawning one thread per test
/// - so calling the raw function directly from every test that needs a
/// window would intermittently panic depending on which worker thread
/// happened to pick up which test. The Slint platform context is itself
/// thread-local (see `i-slint-core`'s `GLOBAL_CONTEXT`), so gating the call
/// on a matching thread-local flag here correctly makes it idempotent for
/// the lifetime of whichever thread runs it.
pub(crate) fn ensure_slint_test_platform() {
    thread_local! {
        static INITIALIZED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    INITIALIZED.with(|initialized| {
        if !initialized.get() {
            i_slint_backend_testing::init_no_event_loop();
            initialized.set(true);
        }
    });
}

/// Builds real, headless `AppWindow`/`ResultsWindow` instances - unlike
/// [`test_ui_state`]'s dead `Weak::default()`, these support
/// `controller.attach_callbacks()` followed by firing the registered
/// callback through Slint's generated `ui.global::<X>().invoke_y(...)`
/// methods, and reading back whatever properties the callback set. No real
/// window system or renderer is involved (see
/// [`ensure_slint_test_platform`]), so this is safe and fast to call from
/// any number of unit tests.
pub(crate) fn test_ui_windows() -> (AppWindow, ResultsWindow) {
    ensure_slint_test_platform();
    crate::helper::ui_thread::fresh_test_queue();
    let ui =
        AppWindow::new().expect("AppWindow::new must succeed under the headless test platform");
    let results_ui = ResultsWindow::new()
        .expect("ResultsWindow::new must succeed under the headless test platform");
    (ui, results_ui)
}

/// A [`UiState`] wired to real windows (see [`test_ui_windows`]), with both
/// file browsers attached - so dialogs a controller opens can be driven with
/// [`choose_file`].
pub(crate) fn ui_state_with_windows(
    ui: &AppWindow,
    results_ui: &ResultsWindow,
    project: ProjectWithRuntime,
) -> Arc<UiState> {
    let owner = ProjectOwner::new();
    let handle = owner.handle();
    *handle.get_project_write() = project;
    use slint::ComponentHandle;
    let mut ui_state = UiState::new(handle, ui.as_weak(), results_ui.as_weak());
    ui_state.app_settings_file = temp_settings_file();
    let ui_state = Arc::new(ui_state);
    ui_state.file_browser.attach(ui);
    ui_state.results_file_browser.attach(results_ui);
    ui_state
}

/// Completes the file dialog currently open in `ui` with `path`, the way a
/// user would: typing a file to open, picking a folder, or entering a save
/// name (answering "replace" if the file exists). Applies all resulting UI
/// updates.
pub(crate) fn choose_file<W>(ui: &W, path: &std::path::Path)
where
    W: slint::ComponentHandle + 'static,
    for<'a> crate::FileBrowserState<'a>: slint::Global<'a, W>,
{
    use crate::helper::ui_thread::drain_ui_queue;
    use crate::{FileBrowserMode, FileBrowserState};
    drain_ui_queue();
    let browser = ui.global::<FileBrowserState>();
    assert!(browser.get_visible(), "no file dialog is open");
    let text = |p: &std::path::Path| slint::SharedString::from(p.to_string_lossy().as_ref());
    match browser.get_mode() {
        FileBrowserMode::OpenFile => browser.invoke_path_entered(text(path)),
        FileBrowserMode::OpenFolder => {
            browser.invoke_navigate(text(path));
            drain_ui_queue();
            browser.invoke_accept();
        }
        FileBrowserMode::SaveFile => {
            browser.invoke_navigate(text(path.parent().unwrap()));
            drain_ui_queue();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            browser.invoke_file_name_edited(name.into());
            browser.invoke_accept();
            drain_ui_queue();
            if !browser.get_overwrite_name().is_empty() {
                browser.invoke_overwrite_answered(true);
            }
        }
    }
    drain_ui_queue();
}

/// A [`FocusController`](crate::editor::focus_controller::FocusController)
/// with its collaborators, for tests building a `PipelinesController`.
pub(crate) fn test_focus_controller(
    ui: slint::Weak<crate::AppWindow>,
    ui_state: &Arc<UiState>,
    object_list: &Arc<crate::editor::object_list_controller::ObjectListController>,
    viewport: &Arc<crate::editor::viewport_controller::ViewportController>,
) -> Arc<crate::editor::focus_controller::FocusController> {
    let image_meta = Arc::new(
        crate::editor::image_meta_controller::ImageMetaController::new(
            ui.clone(),
            ui_state.clone(),
            viewport.clone(),
        ),
    );
    let classification = Arc::new(
        crate::editor::classification_controller::ClassificationController::new(
            ui.clone(),
            ui_state.clone(),
            object_list.clone(),
            viewport.clone(),
        ),
    );
    Arc::new(crate::editor::focus_controller::FocusController::new(
        ui,
        ui_state.clone(),
        image_meta,
        classification,
        viewport.clone(),
    ))
}
