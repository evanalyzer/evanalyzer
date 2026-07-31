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
use crate::UiState;
use evanalyzer_app::ProjectOwner;
use evanalyzer_app::ProjectWithRuntime;
use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, ImageEntry, PixelSizeSettings, SeriesSettings,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Builds a real `UiState` backed by a fresh, empty, in-memory project - no
/// JVM, no Slint window/platform required.
pub(crate) fn test_ui_state() -> Arc<UiState> {
    test_ui_state_with_project(ProjectWithRuntime::default())
}

/// Same as [`test_ui_state`], but seeded with `project` instead of an empty
/// default - see [`project_with_one_image`] for a ready-made single-image
/// fixture.
pub(crate) fn test_ui_state_with_project(project: ProjectWithRuntime) -> Arc<UiState> {
    let owner = ProjectOwner::new();
    let handle = owner.handle();
    *handle.get_project_write() = project;
    Arc::new(UiState::new(
        handle,
        slint::Weak::default(),
        slint::Weak::default(),
    ))
}

/// A minimal project with one 2x2, 2-channel image set as "current" (i.e.
/// what `get_current_image_settings`/`with_current_series_mut` resolve to) -
/// mirrors the equivalent private helper in
/// `evanalyzer_app::extensions::project_ext`'s own test module, since
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
                emission_wave_length: 488.0,
                visible: None,
                histogram: None,
            },
        ),
        (
            1,
            ChannelSettings {
                name: "Ch1".into(),
                emission_wave_length: 561.0,
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
