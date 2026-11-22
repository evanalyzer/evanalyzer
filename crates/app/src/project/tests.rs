//! # tests
//!
//! The project's tests: images, display, classes, objects, folders, saving.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::{image_folder::*, load_save::*, results_images::*};
use crate::images::ImageMeta;
use crate::project::ProjectWithRuntime;
use crate::project::classes::ClassificationExt;
use crate::project::images::GlobalImageSettingsExt;
use crate::project::images::ImageListExt;
use crate::project::utils::ASSIGNED_WAVELENGTHS;
use evanalyzer_cfg::core_types::{InternalErrors, ObjectId, PipelineId, ZProjection};
use evanalyzer_cfg::settings::images_settings::SeriesSettings;
use evanalyzer_cfg::settings::images_settings::{
    ChannelSettings, ImageReview, PixelSizeSettings, TStackSettings, ZStackHandling, ZStackSettings,
};
use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
use evanalyzer_cfg::settings::templates::ProjectTemplate;
use evanalyzer_cfg::{PROJECT_FILE_EXTENSIONS, PROJECT_FILE_TEMPLATE_EXTENSIONS};
use evanalyzer_cfg::{
    core_types::ObjectClass,
    settings::{
        classification_settings::Class, images_settings::ImageEntry,
        project_settings::ProjectSettings,
    },
};
use evanalyzer_cfg::{
    core_types::SegmentationClass, settings::object_settings::ObjectMetricSettings,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[test]
fn the_effective_frame_interval_is_the_projects_then_the_images_then_the_default() {
    use super::*;
    let mut project = ProjectWithRuntime {
        settings: Default::default(),
        tmp_settings: ProjectTmpSettings::default(),
    };
    assert_eq!(
        project.get_effective_frame_interval(Some(5.0)),
        (5.0, ConfigValueSource::Image)
    );
    assert_eq!(
        project.get_effective_frame_interval(None),
        (DEFAULT_FRAME_INTERVAL_S, ConfigValueSource::Default)
    );
    // Zero, negative or NaN mean "not set".
    assert_eq!(
        project.get_effective_frame_interval(Some(-1.0)).1,
        ConfigValueSource::Default
    );
    project.set_frame_interval(Some(f32::NAN));
    assert_eq!(
        project.get_effective_frame_interval(Some(3.0)),
        (3.0, ConfigValueSource::Image)
    );
    project.set_frame_interval(Some(0.0));
    assert_eq!(
        project.get_effective_frame_interval(None).1,
        ConfigValueSource::Default
    );
    project.set_frame_interval(Some(2.0));
    assert_eq!(
        project.get_effective_frame_interval(Some(5.0)),
        (2.0, ConfigValueSource::Project)
    );
}

#[test]
fn the_frame_interval_is_set_and_cleared_without_touching_the_rest() {
    use super::*;
    let mut project = ProjectWithRuntime {
        settings: Default::default(),
        tmp_settings: ProjectTmpSettings::default(),
    };
    assert_eq!(project.get_frame_interval(), None);

    project.set_frame_interval(Some(30.0));
    assert_eq!(project.get_frame_interval(), Some(30.0));

    project
        .settings
        .images
        .settings
        .t_stack
        .as_mut()
        .unwrap()
        .t_stack = 4;
    project.set_frame_interval(None);
    assert_eq!(project.get_frame_interval(), None);
    assert_eq!(
        project
            .settings
            .images
            .settings
            .t_stack
            .as_ref()
            .unwrap()
            .t_stack,
        4
    );
}

use crate::backend::LocalBackend;
use crate::files::LocalFileSystem;

use crate::project::utils::wavelength_to_rgb_u32;
use evanalyzer_cfg::core_types::ImageAddress;

fn fs() -> LocalFileSystem {
    Default::default()
}

fn backend() -> LocalBackend {
    Default::default()
}
use evanalyzer_cfg::settings::images_settings::TStackHandling;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings;

/// A project with one image ("img.tif") of one series (idx 0, 2x2px)
/// carrying two channels ("Ch0"/"Ch1"), selected as the current image -
/// i.e. everything `get_selected_image_series`/`get_current_image_*`
/// need to return `Some`.
// -- get_emission_wave_length -------------------------------------------

#[test]
fn emission_wave_length_comes_from_the_image_when_no_global_is_set() {
    let project = project_with_one_image();
    assert_eq!(project.get_emission_wave_length(0), 488.0);
    assert_eq!(project.get_emission_wave_length(1), 561.0);
}

#[test]
fn a_global_emission_wave_length_overrides_the_images_own() {
    let mut project = project_with_one_image();
    project
        .images
        .settings
        .set_global_emission_wavel_length(0, 405.0);
    assert_eq!(project.get_emission_wave_length(0), 405.0);
    assert_eq!(
        project.get_emission_wave_length(1),
        561.0,
        "other channels unaffected"
    );

    project
        .images
        .settings
        .reset_global_emission_wavel_length(0);
    assert_eq!(
        project.get_emission_wave_length(0),
        488.0,
        "back to the image's value"
    );
}

/// The project's visibility and z-projection for every image; an image
/// with its own display range is listed with it.
#[test]
fn drawn_images_show_the_images_as_the_viewer_does() {
    let mut project = project_with_one_image();
    project.set_global_preferences(&BTreeMap::from([(1, false)]));
    let (shown, own) = project.drawn_image_display();
    assert!(!shown.channels[&1].visible);
    assert!(own.is_empty(), "nothing of its own yet");

    project.set_image_histogram_settings_for_all_channels(0.1, 0.6, 0.0, 1.0);
    project.set_image_z_stack(&ZStackSettings {
        z_projection: ZStackHandling::MaxIntensity,
        z_range: None,
    });
    let (shown, own) = project.drawn_image_display();
    assert!(!shown.channels.contains_key(&0), "shown, auto-scaled");
    assert_eq!(shown.z_projection, ZProjection::None);
    let image = &own["img.tif"];
    assert_eq!(image.channels[&0].range, Some((0.1, 0.6)));
    assert!(!image.channels[&1].visible, "the project's visibility");
    assert_eq!(image.z_projection, ZProjection::MaxIntensity);
}

#[test]
fn a_global_channel_entry_without_a_wave_length_falls_back_to_the_image() {
    let mut project = project_with_one_image();
    // Only visibility is set globally for channel 0.
    project.set_global_preferences(&BTreeMap::from([(0, false)]));
    assert_eq!(project.get_emission_wave_length(0), 488.0);
}

#[test]
fn emission_wave_length_is_zero_when_neither_global_nor_image_knows_it() {
    let mut project = project_with_one_image();
    assert_eq!(
        project.get_emission_wave_length(7),
        0.0,
        "channel the image lacks"
    );
    project.with_current_series_mut(|series| {
        series.channels.get_mut(&1).unwrap().emission_wave_length = None;
    });
    assert_eq!(
        project.get_emission_wave_length(1),
        0.0,
        "image has no value"
    );

    let no_image = ProjectWithRuntime::default();
    assert_eq!(no_image.get_emission_wave_length(0), 0.0);
    let mut global_only = ProjectWithRuntime::default();
    global_only
        .images
        .settings
        .set_global_emission_wavel_length(2, 640.0);
    assert_eq!(global_only.get_emission_wave_length(2), 640.0);
}

fn project_with_one_image() -> ProjectWithRuntime {
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
            z: Some(1.0),
        },
        z_stack: None,
        t_stack: None,
        objects: Vec::new(),
        rgb: false,
    };
    project.images.list.insert(
        rel_path.clone(),
        ImageEntry {
            review: Default::default(),
            rel_path: rel_path.clone(),
            file_size: 16,
            selected_series: 0,
            series: BTreeMap::from([(0, series)]),
        },
    );
    project.set_current_image_path(&rel_path);
    project
}

fn class(id: u32, name: &str) -> Class {
    Class {
        id: id,
        name: name.into(),
        ..Default::default()
    }
}

// -- objects -----------------------------------------------------------

#[test]
fn add_get_delete_object_round_trip() {
    let mut project = project_with_one_image();
    assert_eq!(project.get_objects().unwrap().len(), 0);

    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        ..Default::default()
    });
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(2),
        ..Default::default()
    });
    assert_eq!(project.get_objects().unwrap().len(), 2);

    project.delete_object(ObjectId(1));
    let remaining = project.get_objects().unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, ObjectId(2));
}

#[test]
fn object_mutations_are_a_no_op_without_a_selected_image() {
    let mut project = ProjectWithRuntime::default();
    assert!(project.get_objects().is_none());
    // Must not panic even though there's nothing to add to.
    project.add_object(&ObjectMetricSettings::default());
    project.delete_object(ObjectId(1));
    assert!(project.get_objects().is_none());
}

#[test]
fn add_and_remove_class_from_object() {
    let mut project = project_with_one_image();
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        ..Default::default()
    });

    project.add_class_to_object(ObjectId(1), ObjectClass::Global(3));
    assert!(
        project.get_objects().unwrap()[0]
            .object_class
            .contains(&ObjectClass::Global(3))
    );

    project.remove_class_from_object(ObjectId(1), &ObjectClass::Global(3));
    assert!(
        !project.get_objects().unwrap()[0]
            .object_class
            .contains(&ObjectClass::Global(3))
    );
}

// -- Classes ----------------------------------------------------------

#[test]
fn get_class_from_id_finds_by_id_not_position() {
    let mut project = ProjectWithRuntime::default();
    project.classification.classes_mut().push(class(1, "A"));
    project.classification.classes_mut().push(class(2, "B"));

    let found = project
        .classification
        .get_class_from_id(&ObjectClass::Global(2))
        .unwrap();
    assert_eq!(found.name, "B");
    assert!(
        project
            .classification
            .get_class_from_id(&ObjectClass::Global(99))
            .is_none()
    );
}

#[test]
fn delete_all_classes_clears_the_list() {
    let mut project = ProjectWithRuntime::default();
    project.classification.classes_mut().push(class(1, "A"));
    project.classification.delete_all_classes();
    assert!(project.classification.classes().len() == 1);
    assert!(project.classification.classes()[0].global() == ObjectClass::BACKGROUND);
    assert!(project.classification.classes()[0].color == 0x97978F);
    assert!(project.classification.classes()[0].name == "Background");
}

#[test]
fn selected_object_class_round_trips() {
    let mut project = ProjectWithRuntime::default();
    assert_eq!(project.get_selected_object_class(), ObjectClass::default());
    project.set_selected_object_class(ObjectClass::Global(4));
    assert_eq!(project.get_selected_object_class(), ObjectClass::Global(4));
}

#[test]
fn toggle_class_visibility_is_a_flip_flop() {
    let mut project = ProjectWithRuntime::default();
    let c = ObjectClass::Global(1);
    assert!(
        project.is_class_visible(&c),
        "classes are visible by default"
    );

    project.toggle_class_visibility(c.clone());
    assert!(!project.is_class_visible(&c));

    project.toggle_class_visibility(c.clone());
    assert!(project.is_class_visible(&c));
}

#[test]
fn toggle_hide_unclassified_objects_is_a_flip_flop() {
    let mut project = ProjectWithRuntime::default();
    let initial = project.hide_unclassified_objects();
    project.toggle_hide_unclassified_objects();
    assert_eq!(project.hide_unclassified_objects(), !initial);
    project.toggle_hide_unclassified_objects();
    assert_eq!(project.hide_unclassified_objects(), initial);
}

#[test]
fn count_objects_for_class_counts_both_series_and_preview_objects() {
    let mut project = project_with_one_image();
    let target = ObjectClass::Global(5);

    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        object_class: [target.clone()].into(),
        ..Default::default()
    });
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(2),
        ..Default::default()
    });
    project
        .tmp_settings
        .preview_objects
        .push(ObjectMetricSettings {
            id: ObjectId(3),
            object_class: [target.clone()].into(),
            ..Default::default()
        });

    assert_eq!(project.count_objects_for_class(&target), 2);
    assert_eq!(
        project.count_objects_for_class(&ObjectClass::Global(999)),
        0
    );
}

// -- Pixel sizes: series-level, then global override, then reset ------

#[test]
fn pixel_sizes_default_to_one_and_an_unknown_z_when_nothing_is_set() {
    let project = ProjectWithRuntime::default();
    let px = project.get_pixel_sizes();
    assert_eq!((px.x, px.y, px.z), (1.0, 1.0, None));
}

#[test]
fn pixel_sizes_prefer_global_over_series_and_reset_falls_back() {
    let mut project = project_with_one_image();

    project.set_image_pixel_size_settings(0.1, 0.2, Some(0.3));
    let px = project.get_pixel_sizes();
    assert_eq!(
        (px.x, px.y, px.z),
        (0.1, 0.2, Some(0.3)),
        "falls back to the series value"
    );

    project
        .images
        .settings
        .set_global_pixel_size_settings(9.0, 9.0, Some(9.0));
    let px = project.get_pixel_sizes();
    assert_eq!(
        (px.x, px.y, px.z),
        (9.0, 9.0, Some(9.0)),
        "global overrides series"
    );

    project.images.settings.reset_global_pixel_size_settings();
    let px = project.get_pixel_sizes();
    assert_eq!(
        (px.x, px.y, px.z),
        (0.1, 0.2, Some(0.3)),
        "clearing global falls back to series again"
    );
}

// -- Channel visibility -------------------------------------------------

#[test]
fn channels_are_visible_by_default() {
    let project = project_with_one_image();
    let vis = project.get_image_channel_visibilities();
    assert_eq!(vis.get(&0), Some(&true));
    assert_eq!(vis.get(&1), Some(&true));
}

#[test]
fn set_image_preferences_overrides_per_channel_visibility() {
    let mut project = project_with_one_image();
    project.set_image_preferences(&BTreeMap::from([(0, false)]));

    let vis = project.get_image_channel_visibilities();
    assert_eq!(vis.get(&0), Some(&false));
    assert_eq!(
        vis.get(&1),
        Some(&true),
        "untouched channel keeps its default"
    );

    let visible_only = project.get_image_channel_visibilities_vec();
    assert!(!visible_only.contains(&0));
    assert!(visible_only.contains(&1));
}

/// [`project_with_one_image`] with a third channel and no wavelengths.
fn project_with_three_unnamed_channels() -> ProjectWithRuntime {
    let mut project = project_with_one_image();
    project.with_current_series_mut(|series| {
        series.channels.insert(2, series.channels[&0].clone());
        for channel in series.channels.values_mut() {
            channel.emission_wave_length = None;
            channel.name.clear();
        }
    });
    project
}

#[test]
fn channels_without_wavelengths_are_white_until_assigned_and_grayscale_makes_them_white() {
    let mut project = project_with_three_unnamed_channels();
    let white = [1.0, 1.0, 1.0];
    assert!(
        project
            .get_channel_display_colors()
            .values()
            .all(|c| *c == white)
    );

    // Assigned (the user setting): distinct colours, none white.
    project.assign_missing_wavelengths(&[(0, 0.0), (1, 0.0), (2, 0.0)]);
    let colors: Vec<[f32; 3]> = project.get_channel_display_colors().into_values().collect();
    assert!(colors.iter().all(|c| *c != white), "{colors:?}");
    assert!(colors[0] != colors[1] && colors[1] != colors[2] && colors[0] != colors[2]);

    project.set_grayscale(true);
    assert!(
        project
            .get_channel_display_colors()
            .values()
            .all(|c| *c == [1.0, 1.0, 1.0])
    );
}

#[test]
fn switching_grayscale_on_keeps_only_the_selected_channel_visible() {
    let mut project = project_with_three_unnamed_channels();
    project.set_global_preferences(&BTreeMap::from([(0, true), (1, true), (2, true)]));
    project.images.settings.set_global_selected_channel(&1);

    project.set_grayscale(true);

    assert_eq!(
        project.get_image_channel_visibilities(),
        BTreeMap::from([(0, false), (1, true), (2, false)])
    );
}

#[test]
fn in_grayscale_switching_a_channel_on_switches_the_other_off() {
    let mut project = project_with_three_unnamed_channels();
    project.set_global_preferences(&BTreeMap::from([(0, true), (1, false), (2, false)]));
    let requested = BTreeMap::from([(0, true), (1, false), (2, true)]);

    assert_eq!(
        project.with_one_visible_channel(&requested),
        requested,
        "colour mode"
    );

    project.set_grayscale(true);
    assert_eq!(
        project.with_one_visible_channel(&requested),
        BTreeMap::from([(0, false), (1, false), (2, true)]),
        "the newly switched on one wins"
    );
    let none = BTreeMap::from([(0, false), (1, false), (2, false)]);
    assert_eq!(
        project.with_one_visible_channel(&none),
        none,
        "none visible is fine"
    );
}

#[test]
fn auto_added_classes_of_unnamed_channels_get_names_and_distinct_colours() {
    let mut project = project_with_three_unnamed_channels();
    project.auto_add_classes_based_on_image_meta();
    // Besides the "Background" class every project has.
    let classes: Vec<_> = project
        .classification
        .classes()
        .iter()
        .filter(|c| c.name != "Background")
        .collect();
    let names: Vec<&str> = classes.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["Class 0", "Class 1", "Class 2"]);
    assert_eq!(
        classes.iter().map(|c| c.color).collect::<Vec<_>>(),
        [0xFF0000, 0x00FF00, 0x0000FF]
    );
}

#[test]
fn set_global_preferences_is_the_fallback_when_a_channel_has_no_local_override() {
    let mut project = project_with_one_image();
    project.set_global_preferences(&BTreeMap::from([(0, false)]));
    let vis = project.get_image_channel_visibilities();
    assert_eq!(vis.get(&0), Some(&false));

    // A local override still wins over the global one.
    project.set_image_preferences(&BTreeMap::from([(0, true)]));
    let vis = project.get_image_channel_visibilities();
    assert_eq!(vis.get(&0), Some(&true));
}

// -- Image root / path resolution ---------------------------------------

#[test]
fn get_image_absolute_path_from_relative_only_resolves_known_images() {
    let mut project = project_with_one_image();
    project.images.root = Some(PathBuf::from("/data/images"));

    assert_eq!(
        project.get_image_absolute_path_from_relative(Path::new("img.tif")),
        Some(PathBuf::from("/data/images/img.tif"))
    );
    assert_eq!(
        project.get_image_absolute_path_from_relative(Path::new("unknown.tif")),
        None
    );
}

#[test]
fn the_run_image_root_is_read_from_the_runs_project_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("plate.evadb");
    assert_eq!(run_image_root(&fs(), &db), None, "no copy");
    std::fs::write(
        dir.path().join("plate.evaproj"),
        r#"{"schemaVersion": 4, "images": {"root": "/data/run", "list": {}}, "pipelines": []}"#,
    )
    .unwrap();
    assert_eq!(run_image_root(&fs(), &db), Some(PathBuf::from("/data/run")));
}

#[test]
fn a_results_image_is_found_although_the_image_folder_changed() {
    use evanalyzer_cfg::settings::images_settings::{ImageEntry, ImageList};
    let list = |paths: &[&str]| {
        ImageList(std::sync::Arc::new(
            paths
                .iter()
                .map(|p| (PathBuf::from(p), std::sync::Arc::new(ImageEntry::default())))
                .collect(),
        ))
    };
    let find = |list: &ImageList, root: &str, rel: &str, run: Option<&str>| {
        find_results_image(
            list,
            Some(Path::new(root)),
            Path::new(rel),
            run.map(Path::new),
        )
    };
    let images = list(&["a.tif", "b.tif"]);
    // Analysed from the parent folder, the project now on `CMEVs`.
    assert_eq!(
        find(&images, "/d/CMEVs", "CMEVs/a.tif", Some("/d")),
        Some(PathBuf::from("a.tif"))
    );
    // The same folder: the path as it is.
    assert_eq!(
        find(&images, "/d", "b.tif", Some("/d")),
        Some(PathBuf::from("b.tif"))
    );
    // Analysed on a subfolder, the project now on the parent.
    let parent = list(&["CMEVs/a.tif", "Control/a.tif"]);
    assert_eq!(
        find(&parent, "/d", "a.tif", Some("/d/Control")),
        Some(PathBuf::from("Control/a.tif"))
    );
    // Moved elsewhere: the longest same ending - unless two share it.
    assert_eq!(
        find(&parent, "/new", "Control/a.tif", Some("/old")),
        Some(PathBuf::from("Control/a.tif"))
    );
    assert_eq!(
        find(&parent, "/new", "Plate1/Control/a.tif", Some("/old")),
        Some(PathBuf::from("Control/a.tif"))
    );
    assert_eq!(
        find(&parent, "/new", "a.tif", Some("/old")),
        None,
        "which a.tif?"
    );
    assert_eq!(
        find(&parent, "/new", "c.tif", None),
        None,
        "not in the project"
    );
}

#[test]
fn is_image_part_of_the_root_requires_a_root_to_be_set() {
    let mut project = ProjectWithRuntime::default();
    assert!(!project.is_image_part_of_the_root(Path::new("/data/images/img.tif")));

    project.images.root = Some(PathBuf::from("/data/images"));
    assert!(project.is_image_part_of_the_root(Path::new("/data/images/img.tif")));
    assert!(!project.is_image_part_of_the_root(Path::new("/other/img.tif")));
}

#[test]
fn change_images_root_clears_the_image_list_and_current_image() {
    let mut project = project_with_one_image();
    assert!(!project.images.list.is_empty());

    let new_root = PathBuf::from("/new/root");
    project.change_images_root(&new_root);

    assert!(project.images.list.is_empty());
    assert_eq!(project.images.root, Some(new_root));
    assert_eq!(project.get_current_image_path_cloned(), None);
}

#[test]
fn select_new_images_root_with_check_fails_when_the_sample_image_is_missing() {
    let mut project = project_with_one_image();
    let missing_root = PathBuf::from("/definitely/does/not/exist/anywhere");
    let action = project.select_new_images_root_with_check(&fs(), &missing_root);
    assert_eq!(action, SelectNewProjectRootAction::ImageNotFound);
    // A failed check must not have changed the root.
    assert_ne!(project.images.root, Some(missing_root));
}

#[test]
fn select_new_images_root_with_check_succeeds_when_the_sample_image_exists() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("img.tif"), b"fake image bytes").unwrap();

    let mut project = project_with_one_image();
    let action = project.select_new_images_root_with_check(&fs(), &dir.path().to_path_buf());
    assert_eq!(action, SelectNewProjectRootAction::Success);
    assert_eq!(project.images.root, Some(dir.path().to_path_buf()));
}

// -- Save / load round trip ----------------------------------------------

#[test]
fn save_project_without_a_path_asks_the_caller_to_pick_one() {
    let mut project = ProjectWithRuntime::default();
    assert_eq!(
        project.save_project(&fs()),
        SaveProjectActions::PleaseSelectFile
    );
}

#[test]
fn save_project_as_appends_the_extension_and_records_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = ProjectWithRuntime::default();
    project.meta.name = "Round Trip Project".into();

    // No extension given - save_project_as should append the project one.
    let requested = dir.path().join("myproject");
    project
        .save_project_as(&fs(), &requested)
        .expect("save should succeed");

    let expected = dir
        .path()
        .join(format!("myproject.{}", PROJECT_FILE_EXTENSIONS));
    assert!(
        expected.exists(),
        "file should be written with the project extension appended"
    );
    assert_eq!(project.tmp_settings.current_project, Some(expected.clone()));

    // save_project() (no path arg) should now reuse that recorded path.
    project.meta.name = "Changed After Save As".into();
    assert_eq!(project.save_project(&fs()), SaveProjectActions::Success);

    let written = std::fs::read_to_string(&expected).unwrap();
    assert!(written.contains("Changed After Save As"));
}

#[test]
fn loaded_project_round_trips_objects_and_classes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.evaproj");

    let mut project = project_with_one_image();
    project
        .classification
        .classes_mut()
        .push(class(1, "Nucleus"));
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        area: 1234,
        object_class: [ObjectClass::Global(1)].into(),
        ..Default::default()
    });
    project.save_project_as(&fs(), &path).unwrap();

    let loaded = load_project(&fs(), &path).expect("load should succeed");
    // classes()[0] is the auto-prepended Background class.
    assert_eq!(loaded.classification.classes().len(), 2);
    assert_eq!(loaded.classification.classes()[1].name, "Nucleus");
    // `load_project` doesn't select an image on its own - it only loads
    // `settings`, leaving `tmp_settings` at its defaults.
    assert_eq!(loaded.get_current_image_path_cloned(), None);
    let objects = &loaded.images.list.get(Path::new("img.tif")).unwrap().series[&0].objects;
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].area, 1234);
}

#[test]
fn save_project_as_stores_ai_model_paths_relative_to_the_project_dir() {
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;
    use evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.evaproj");
    let absolute_model_path = dir.path().join("models").join("nuclei.evamodel");

    let mut project = ProjectWithRuntime::default();
    let mut step_pipeline = pipeline(1);
    step_pipeline.steps.push(PipelineStepSettings {
        id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
        enabled: true,
        content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
            PipelineCommand::PixelClassifier(PixelClassifierSettings {
                model_path: absolute_model_path.clone(),
                ..Default::default()
            }),
        ),
    });
    project.add_pipeline(step_pipeline);
    project.save_project_as(&fs(), &path).unwrap();

    // On disk, the path is relative to the project directory - not the
    // absolute temp-dir path it was set from - so the project stays
    // valid if the folder is moved or copied elsewhere.
    let written = std::fs::read_to_string(&path).unwrap();
    assert!(
        written.contains("models/nuclei.evamodel")
            && !written.contains(&*absolute_model_path.to_string_lossy()),
        "expected a relative model path on disk, got: {written}"
    );

    // Reloading resolves it straight back to the original absolute path
    // - the rest of the app never has to know about relative storage.
    let loaded = load_project(&fs(), &path).expect("load should succeed");
    let PipelineCommand::PixelClassifier(settings) =
        loaded.pipelines[0].steps[0].command().unwrap()
    else {
        panic!("expected a PixelClassifier command");
    };
    assert_eq!(settings.model_path, absolute_model_path);
}

// -- Pipeline management ---------------------------------------------

fn pipeline(id: u32) -> PipelineSettings {
    PipelineSettings {
        id: PipelineId(id),
        name: "".into(),
        description: None,
        image_source: ImageAddress::default(),
        enabled: true,
        steps: vec![],
    }
}

#[test]
fn add_pipeline_appends_to_the_list() {
    let mut project = ProjectWithRuntime::default();
    project.add_pipeline(pipeline(1));
    project.add_pipeline(pipeline(2));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id).collect::<Vec<_>>(),
        vec![PipelineId(1), PipelineId(2)]
    );
}

#[test]
fn move_pipeline_up_and_down_swap_neighbors_and_are_edge_noops() {
    let mut project = ProjectWithRuntime::default();
    project.add_pipeline(pipeline(1));
    project.add_pipeline(pipeline(2));
    project.add_pipeline(pipeline(3));

    project.move_pipeline_up(PipelineId(2));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
        vec![2, 1, 3]
    );

    // Already first: no-op.
    project.move_pipeline_up(PipelineId(2));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
        vec![2, 1, 3]
    );

    project.move_pipeline_down(PipelineId(2));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    // Already last: no-op.
    project.move_pipeline_down(PipelineId(3));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    // Unknown id: no-op, no panic.
    project.move_pipeline_up(PipelineId(99));
    project.move_pipeline_down(PipelineId(99));
    assert_eq!(
        project.pipelines.iter().map(|p| p.id.0).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn enable_pipeline_toggles_only_the_matching_pipeline() {
    let mut project = ProjectWithRuntime::default();
    project.add_pipeline(pipeline(1));
    project.add_pipeline(pipeline(2));

    project.enable_pipeline(false, PipelineId(1));
    assert!(!project.pipelines[0].enabled);
    assert!(project.pipelines[1].enabled);

    // Unknown id: no panic, no effect.
    project.enable_pipeline(false, PipelineId(99));
}

#[test]
fn enable_pipeline_step_toggles_only_the_matching_step() {
    let mut project = ProjectWithRuntime::default();
    let mut p = pipeline(1);
    p.steps = vec![
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::Blur(Default::default()),
            ),
        },
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::Blur(Default::default()),
            ),
        },
    ];
    project.add_pipeline(p);

    project.enable_pipeline_step(false, PipelineId(1), 0);
    assert!(!project.pipelines[0].steps[0].enabled);
    assert!(project.pipelines[0].steps[1].enabled);

    // Unknown pipeline / out-of-range step: no panic.
    project.enable_pipeline_step(false, PipelineId(99), 0);
    project.enable_pipeline_step(false, PipelineId(1), 99);
}

#[test]
fn apply_project_template_replaces_classification_plate_and_pipelines() {
    let mut project = ProjectWithRuntime::default();
    project.classification.classes_mut().push(class(1, "Stale"));
    project.add_pipeline(pipeline(1));

    let template = evanalyzer_cfg::settings::templates::ProjectTemplate {
        meta: Default::default(),
        classification:
            evanalyzer_cfg::settings::classification_settings::ClassificationSettings::new_from_existing(
               vec![class(9, "Fresh")]),
        plate: Default::default(),
        pipelines: vec![PipelineSettings {
            id: PipelineId(0),
            enabled: true,
            name:"P1".into(),
            description: None,
            steps: vec![],
            image_source: ImageAddress::Scratchpad
        }],
        modules: vec![evanalyzer_cfg::settings::modules::ModuleDefinition {
            key: "CELL_RECONSTRUCTION".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![PipelineStepSettings {
                id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                enabled: true,
                content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(PipelineCommand::Blur(Default::default())),
            }],
            parameters: vec![],
            origin: None,
        }],
        ..Default::default()
    };
    project.apply_project_template(&template);
    assert_eq!(project.modules.len(), 1, "the definitions come along");
    assert_eq!(
        project.modules[0].steps[0].id,
        evanalyzer_cfg::core_types::StepId(1),
        "numbered within the definition"
    );

    // classes()[0] is the auto-prepended Background class.
    assert_eq!(project.classification.classes().len(), 2);
    assert_eq!(project.classification.classes()[1].name, "Fresh");
    assert_eq!(project.pipelines.len(), 1);
    assert_eq!(
        project.pipelines[0].id,
        PipelineId(0),
        "the template's own id is used as-is, not renumbered"
    );
    assert_eq!(project.pipelines[0].name, "P1");
}

// -- Image existence / root checks ------------------------------------

#[test]
fn does_project_images_exist_is_true_when_no_root_is_set() {
    let project = ProjectWithRuntime::default();
    assert!(project.does_project_images_exist(&fs()));
}

#[test]
fn does_project_images_exist_checks_the_first_listed_image_under_the_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("img.tif"), b"x").unwrap();

    let mut project = project_with_one_image();
    project.images.root = Some(dir.path().to_path_buf());
    assert!(project.does_project_images_exist(&fs()));

    // Point the root somewhere that doesn't have img.tif.
    let other_dir = tempfile::tempdir().unwrap();
    project.images.root = Some(other_dir.path().to_path_buf());
    assert!(!project.does_project_images_exist(&fs()));
}

#[test]
fn does_project_image_exists_at_path_is_true_for_an_empty_image_list() {
    let project = ProjectWithRuntime::default();
    assert!(project.does_project_image_exists_at_path(&fs(), &PathBuf::from("/anywhere")));
}

// -- add_image / add_image_to_list ------------------------------------

fn sample_image_meta(channel_name: &str, wavelength: f32) -> ImageMeta {
    let mut resolutions = BTreeMap::new();
    resolutions.insert(
        0,
        crate::images::PyramidInfo {
            width: 4,
            height: 4,
            nr_bits: 8,
            ..Default::default()
        },
    );
    let mut channels = BTreeMap::new();
    channels.insert(
        0,
        evanalyzer_core::ChannelInfo {
            name: channel_name.into(),
            emission_wave_length: wavelength,
            ..Default::default()
        },
    );
    let mut series = BTreeMap::new();
    series.insert(
        0,
        evanalyzer_core::ImageInfo {
            nr_c_stacks: 1,
            nr_z_stacks: 1,
            nr_t_stacks: 1,
            resolutions,
            channels,
            ..Default::default()
        },
    );
    ImageMeta {
        name: "meta".into(),
        series,
        ..Default::default()
    }
}

#[test]
fn add_image_sets_the_root_from_the_first_images_parent_when_unset() {
    let mut project = ProjectWithRuntime::default();
    let meta = sample_image_meta("Ch0", 488.0);
    let action = project.add_image(Path::new("/data/plate1/img.tif"), &meta);

    assert_eq!(action, ProjectAction::Success);
    assert_eq!(project.images.root, Some(PathBuf::from("/data/plate1")));
    assert!(project.images.list.contains_key(Path::new("img.tif")));
}

#[test]
fn add_image_accepts_a_path_inside_an_existing_root() {
    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(PathBuf::from("/data/plate1"));
    let meta = sample_image_meta("Ch0", 488.0);

    let action = project.add_image(Path::new("/data/plate1/well_A1/img.tif"), &meta);
    assert_eq!(action, ProjectAction::Success);
    assert!(
        project
            .images
            .list
            .contains_key(Path::new("well_A1/img.tif"))
    );
}

#[test]
fn add_image_outside_an_existing_root_is_a_conflict_and_does_not_add_it() {
    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(PathBuf::from("/data/plate1"));
    let meta = sample_image_meta("Ch0", 488.0);

    let action = project.add_image(Path::new("/other/img.tif"), &meta);
    assert!(matches!(action, ProjectAction::OutSideRootConflict { .. }));
    assert!(project.images.list.is_empty());
}

#[test]
fn add_image_to_list_skips_a_path_that_is_already_present() {
    let mut project = ProjectWithRuntime::default();
    let first = sample_image_meta("Ch0", 488.0);
    let second = sample_image_meta("ShouldNotOverwrite", 600.0);

    project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &first);
    project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &second);

    let entry = project.images.list.get(Path::new("img.tif")).unwrap();
    assert_eq!(
        entry.series[&0].channels[&0].name, "Ch0",
        "the second insert must be a no-op"
    );
}

#[test]
fn add_image_to_list_derives_series_and_channel_settings_from_the_metadata() {
    let mut project = ProjectWithRuntime::default();
    let meta = sample_image_meta("DAPI", 461.0);
    project.add_image_to_list(Path::new("img.tif"), Path::new("/abs/img.tif"), &meta);

    let entry = project.images.list.get(Path::new("img.tif")).unwrap();
    let series = &entry.series[&0];
    assert_eq!(series.image_width, 4);
    assert_eq!(series.image_height, 4);
    assert_eq!(series.channels[&0].name, "DAPI");
    assert_eq!(series.channels[&0].emission_wave_length, Some(461.0));
    assert_eq!(
        entry.file_size,
        4 * 4 * 8,
        "approximate size is width * height * bits"
    );
}

// -- Series / channel selection ---------------------------------------

#[test]
fn set_active_series_updates_the_current_images_selected_series() {
    let mut project = project_with_one_image();
    project.with_current_image_mut(|img| {
        img.series.insert(1, img.series[&0].clone());
    });
    project.set_active_series(&1);
    assert_eq!(project.get_selected_image_series_idx(), Some(1));
    assert!(project.get_selected_image_series().is_some());
}

#[test]
fn get_selected_image_series_mut_allows_editing_the_active_series() {
    let mut project = project_with_one_image();
    project.get_selected_image_series_mut().unwrap().image_width = 999;
    assert_eq!(
        project.get_selected_image_series().unwrap().image_width,
        999
    );
}

#[test]
fn selected_channel_prefers_local_over_global_over_zero() {
    let mut project = project_with_one_image();
    assert_eq!(
        project.get_selected_image_channel_idx(),
        0,
        "defaults to 0 with nothing set"
    );

    project.images.settings.set_global_selected_channel(&1);
    assert_eq!(
        project.get_selected_image_channel_idx(),
        1,
        "falls back to the global setting"
    );

    project.set_image_selected_channel(&0);
    assert_eq!(
        project.get_selected_image_channel_idx(),
        0,
        "a local override wins over global"
    );
    assert_eq!(project.get_selected_image_channel().unwrap().name, "Ch0");
}

// -- Z / T stack --------------------------------------------------------

#[test]
fn z_stack_prefers_local_over_global() {
    let mut project = project_with_one_image();
    assert!(project.get_z_stack().is_none());

    let global = ZStackSettings {
        z_projection: ZStackHandling::MaxIntensity,
        z_range: None,
    };
    project.images.settings.set_global_z_stack(&global);
    assert_eq!(
        project.get_z_stack().unwrap().z_projection,
        ZStackHandling::MaxIntensity
    );

    let local = ZStackSettings {
        z_projection: ZStackHandling::SumIntensity,
        z_range: None,
    };
    project.set_image_z_stack(&local);
    assert_eq!(
        project.get_z_stack().unwrap().z_projection,
        ZStackHandling::SumIntensity
    );
}

#[test]
fn t_stack_prefers_local_over_global() {
    let mut project = project_with_one_image();
    assert!(project.get_t_stack().is_none());

    let global = TStackSettings {
        stack_handling: TStackHandling::AllStacks,
        playback_speed: 1.0,
        t_stack: 0,
        frame_interval: None,
    };
    project.images.settings.set_global_t_stack(&global);
    assert_eq!(
        project.get_t_stack().unwrap().stack_handling,
        TStackHandling::AllStacks
    );

    let local = TStackSettings {
        stack_handling: TStackHandling::SingleStack,
        playback_speed: 1.0,
        t_stack: 3,
        frame_interval: None,
    };
    project.set_image_t_stack(&local);
    assert_eq!(
        project.get_t_stack().unwrap().stack_handling,
        TStackHandling::SingleStack
    );
}

// -- Histogram settings -------------------------------------------------

/// Linking a colour image's colours gives every channel the selected
/// one's range; unlinking keeps the ranges as they are.
#[test]
fn linking_the_colours_gives_all_channels_one_range() {
    let mut project = project_with_one_image();
    project.set_image_histogram_settings_for_channel(0, 0.1, 0.9, 0.0, 1.0);
    project.set_image_histogram_settings_for_channel(1, 0.3, 0.5, 0.0, 1.0);
    project.set_rgb_linked(false);
    assert!(project.images.settings.rgb_unlinked);
    assert_eq!(
        project.get_image_channel_histograms()[&1]
            .as_ref()
            .unwrap()
            .max,
        0.5
    );

    project.set_rgb_linked(true);
    assert!(!project.images.settings.rgb_unlinked);
    let selected = project
        .get_histograms_from_selected_channel()
        .unwrap()
        .clone();
    for window in project.get_image_channel_histograms().values() {
        let window = window.as_ref().unwrap();
        assert_eq!((window.min, window.max), (selected.min, selected.max));
    }

    project.set_image_histogram_settings_for_all_channels(0.2, 0.7, 0.0, 1.0);
    for window in project.get_image_channel_histograms().values() {
        let window = window.as_ref().unwrap();
        assert_eq!((window.min, window.max), (0.2, 0.7));
    }
}

#[test]
fn set_image_histogram_settings_for_channel_updates_only_that_channel() {
    let mut project = project_with_one_image();
    project.set_image_histogram_settings_for_channel(0, 0.1, 0.9, 0.0, 1.0);

    let hist = project.get_image_channel_histograms();
    assert_eq!(hist[&0].as_ref().unwrap().min, 0.1);
    assert_eq!(hist[&0].as_ref().unwrap().max, 0.9);
    assert!(hist[&1].is_none());
}

#[test]
fn set_image_histogram_settings_for_active_channel_targets_the_selected_channel() {
    let mut project = project_with_one_image();
    project.set_image_selected_channel(&1);
    project.set_image_histogram_settings_for_active_channel(0.2, 0.8, 0.0, 1.0);

    assert_eq!(
        project.get_histograms_from_selected_channel().unwrap().min,
        0.2
    );
    assert!(project.get_image_channel_histograms()[&0].is_none());
}

/// The histogram shows the scaling the image is drawn with: after "Use
/// for all images" the project's (no image has its own any more).
#[test]
fn the_histogram_shows_the_projects_scaling_after_use_for_all_images() {
    let mut project = project_with_one_image();
    project.set_image_histogram_settings_for_channel(0, 0.1, 0.9, 0.0, 1.0);
    project.apply_histograms_to_all_images();

    let shown = project
        .get_histograms_from_selected_channel()
        .expect("the project's scaling");
    assert_eq!((shown.min, shown.max), (0.1, 0.9));
    // The image's own still wins.
    project.set_image_histogram_settings_for_channel(0, 0.3, 0.6, 0.0, 1.0);
    let shown = project.get_histograms_from_selected_channel().unwrap();
    assert_eq!((shown.min, shown.max), (0.3, 0.6));
}

#[test]
fn the_open_images_scaling_becomes_every_images_scaling() {
    let mut project = project_with_one_image();
    project.set_image_histogram_settings_for_channel(0, 0.1, 0.9, 0.0, 1.0);

    project.apply_histograms_to_all_images();

    // The project's scaling now; no image keeps its own.
    let global = project.images.settings.channels[&0]
        .histogram
        .as_ref()
        .unwrap();
    assert_eq!((global.min, global.max), (0.1, 0.9));
    assert!(project.images.list.values().all(|image| {
        image
            .series
            .values()
            .all(|series| series.channels.values().all(|c| c.histogram.is_none()))
    }));
    // ... which the open image shows.
    assert_eq!(
        project.get_image_channel_histograms()[&0]
            .as_ref()
            .unwrap()
            .min,
        0.1
    );
    assert!(
        project
            .images
            .settings
            .channels
            .get(&1)
            .is_none_or(|c| c.histogram.is_none())
    );
}

#[test]
fn channels_without_a_wavelength_get_distinct_ones_for_the_project() {
    let mut project = project_with_one_image();
    // Channel 1 is red already (635 nm): the others skip red.
    assert!(project.assign_missing_wavelengths(&[(0, 0.0), (1, 635.0), (2, 0.0)]));
    let nm = |p: &ProjectWithRuntime, id: i32| {
        p.images
            .settings
            .channels
            .get(&id)
            .and_then(|c| c.emission_wave_length)
    };
    assert_eq!(nm(&project, 0), Some(520.0));
    assert_eq!(nm(&project, 1), None, "it has its own");
    assert_eq!(nm(&project, 2), Some(450.0));
    // Once there, nothing more to assign.
    assert!(!project.assign_missing_wavelengths(&[(0, 0.0), (1, 635.0), (2, 0.0)]));
    // A single channel stays white.
    let mut single = project_with_one_image();
    assert!(!single.assign_missing_wavelengths(&[(0, 0.0)]));
}

#[test]
fn a_channel_made_white_on_purpose_stays_white() {
    let mut project = project_with_one_image();
    // The user picked white (0) for channel 0.
    project
        .images
        .settings
        .set_global_emission_wavel_length(0, 0.0);
    assert!(project.assign_missing_wavelengths(&[(0, 0.0), (1, 0.0)]));
    let nm = |p: &ProjectWithRuntime, id: i32| {
        p.images
            .settings
            .channels
            .get(&id)
            .and_then(|c| c.emission_wave_length)
    };
    assert_eq!(nm(&project, 0), Some(0.0), "kept white");
    assert!(
        nm(&project, 1).is_some_and(|nm| nm > 1.0),
        "the other one gets a colour"
    );
    assert_eq!(project.get_emission_wave_length(0), 0.0);
}

#[test]
fn a_channel_without_a_wavelength_is_shown_white() {
    let project = project_with_one_image();
    let colors = project.channel_display_colors(&[(0, 0.0), (1, 0.0)]);
    assert_eq!(colors[&0], [1.0, 1.0, 1.0]);
    assert_eq!(colors[&1], [1.0, 1.0, 1.0]);
}

#[test]
fn set_image_histogram_settings_for_an_unknown_channel_does_not_panic() {
    let mut project = project_with_one_image();
    project.set_image_histogram_settings_for_channel(99, 0.0, 1.0, 0.0, 1.0);
    // No channel 99 exists - nothing to assert beyond "did not panic".
}

// -- pipeline focus ------------------------------------------------------

/// A pipeline on `source` whose Transform Objects step reads `input`
/// and writes `output`.
fn focus_test_pipeline(id: u32, source: ImageAddress, input: u32, output: u32) -> PipelineSettings {
    use evanalyzer_cfg::settings::pipeline_command_settings::TransformObjectsSettings;
    PipelineSettings {
        id: PipelineId(id),
        name: format!("p{id}"),
        description: None,
        image_source: source,
        enabled: true,
        steps: vec![PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::TransformObjects(TransformObjectsSettings {
                    input_class: ObjectClass::Global(input),
                    output_class: ObjectClass::Global(output),
                    ..Default::default()
                }),
            ),
        }],
    }
}

fn project_with_focus_pipelines() -> ProjectWithRuntime {
    let mut project = project_with_one_image();
    project
        .pipelines
        .push(focus_test_pipeline(1, ImageAddress::Channel(1), 3, 4));
    project
        .pipelines
        .push(focus_test_pipeline(2, ImageAddress::Scratchpad, 5, 6));
    project
}

#[test]
fn focus_shows_only_the_pipelines_channel_and_classes() {
    let mut project = project_with_focus_pipelines();
    assert!(project.focus_pipeline(PipelineId(1)));
    assert_eq!(project.focused_pipeline(), Some(PipelineId(1)));
    let channels = project.get_image_channel_visibilities();
    assert_eq!(channels.get(&0), Some(&false));
    assert_eq!(channels.get(&1), Some(&true));
    assert!(project.is_class_visible(&ObjectClass::Global(3)));
    assert!(project.is_class_visible(&ObjectClass::Global(4)));
    assert!(!project.is_class_visible(&ObjectClass::Global(5)));
}

#[test]
fn focus_shows_the_classes_used_inside_a_pipelines_modules() {
    let mut project = project_with_focus_pipelines();
    // Pipeline 3: only a module step, whose transform uses classes 7 → 8.
    let mut inner = focus_test_pipeline(9, ImageAddress::Channel(0), 7, 8).steps;
    inner[0].id = evanalyzer_cfg::core_types::StepId(1);
    project
        .modules
        .push(evanalyzer_cfg::settings::modules::ModuleDefinition {
            key: "INNER".into(),
            meta: Default::default(),
            classes: vec![],
            steps: inner,
            parameters: vec![],
            origin: None,
        });
    let mut pipeline = focus_test_pipeline(3, ImageAddress::Channel(0), 1, 1);
    pipeline.steps = vec![PipelineStepSettings {
        id: evanalyzer_cfg::core_types::StepId(1),
        enabled: true,
        content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Module(
            evanalyzer_cfg::settings::modules::ModuleInstance {
                key: "INNER".into(),
                values: Default::default(),
            },
        ),
    }];
    project.pipelines.push(pipeline);

    project.focus_pipeline(PipelineId(3));
    assert!(project.is_class_visible(&ObjectClass::Global(7)));
    assert!(project.is_class_visible(&ObjectClass::Global(8)));
    assert!(
        !project.is_class_visible(&ObjectClass::Global(3)),
        "pipeline 1's"
    );
}

#[test]
fn focus_on_a_pipeline_without_an_image_source_shows_the_channels_it_loads() {
    let mut project = project_with_focus_pipelines();
    let mut pipeline = focus_test_pipeline(3, ImageAddress::None, 7, 8);
    pipeline.steps.insert(
        0,
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::LoadChannel(
                    evanalyzer_cfg::settings::pipeline_command_settings::LoadChannelSettings {
                        channel: evanalyzer_cfg::core_types::ImageChannelIdx(1),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    },
                ),
            ),
        },
    );
    project.pipelines.push(pipeline);

    project.focus_pipeline(PipelineId(3));
    let channels = project.get_image_channel_visibilities();
    assert_eq!(channels.get(&0), Some(&false));
    assert_eq!(channels.get(&1), Some(&true), "the loaded channel");
}

#[test]
fn focus_on_a_scratchpad_pipeline_shows_every_channel() {
    let mut project = project_with_focus_pipelines();
    project.set_global_preferences(&BTreeMap::from([(0, false), (1, true)]));
    project.focus_pipeline(PipelineId(2));
    assert!(
        project
            .get_image_channel_visibilities()
            .values()
            .all(|&v| v)
    );
    assert!(project.is_class_visible(&ObjectClass::Global(6)));
    assert!(!project.is_class_visible(&ObjectClass::Global(3)));
}

#[test]
fn clearing_the_focus_restores_the_users_settings_exactly() {
    let mut project = project_with_focus_pipelines();
    project.set_global_preferences(&BTreeMap::from([(0, true), (1, false)]));
    project.toggle_class_visibility(ObjectClass::Global(7));
    let channels_before = project.get_image_channel_visibilities();

    project.focus_pipeline(PipelineId(1));
    // Changes while focused only touch the focus overlay ...
    project.set_global_preferences(&BTreeMap::from([(0, true), (1, true)]));
    project.toggle_class_visibility(ObjectClass::Global(3));
    assert!(
        project
            .get_image_channel_visibilities()
            .values()
            .all(|&v| v)
    );
    assert!(!project.is_class_visible(&ObjectClass::Global(3)));

    // ... so leaving the focus brings back what the user had.
    project.clear_pipeline_focus();
    assert_eq!(project.focused_pipeline(), None);
    assert_eq!(project.get_image_channel_visibilities(), channels_before);
    assert!(!project.is_class_visible(&ObjectClass::Global(7)));
    assert!(project.is_class_visible(&ObjectClass::Global(3)));
}

/// The channels shown, as indices.
fn shown_channels(project: &ProjectWithRuntime) -> Vec<i32> {
    project.get_image_channel_visibilities_vec()
}

/// Grayscale mode shows one channel at a time, also when a focused
/// pipeline reads several - here every channel (a scratchpad pipeline):
/// the selected one if the focus shows it.
#[test]
fn in_grayscale_a_focus_on_several_channels_shows_one() {
    let mut project = project_with_focus_pipelines();
    project.images.settings.set_global_selected_channel(&1);
    project.set_grayscale(true);
    project.focus_pipeline(PipelineId(2));
    assert_eq!(shown_channels(&project), [1]);
    // A selected channel the focus doesn't show: its first one.
    project.focus_pipeline(PipelineId(1));
    project.images.settings.set_global_selected_channel(&0);
    assert_eq!(shown_channels(&project), [1]);
}

/// Grayscale switched on while focused, then the focus left: the user's
/// own channels come back one at a time, too.
#[test]
fn grayscale_switched_on_while_focused_stays_one_channel_after_the_focus() {
    let mut project = project_with_focus_pipelines();
    project.set_global_preferences(&BTreeMap::from([(0, true), (1, true)]));
    project.images.settings.set_global_selected_channel(&0);
    project.focus_pipeline(PipelineId(2));
    project.set_grayscale(true);
    assert_eq!(shown_channels(&project), [0]);
    project.clear_pipeline_focus();
    assert_eq!(shown_channels(&project), [0]);
}

#[test]
fn focusing_an_unknown_pipeline_clears_the_focus() {
    let mut project = project_with_focus_pipelines();
    project.focus_pipeline(PipelineId(1));
    assert!(!project.focus_pipeline(PipelineId(99)));
    assert_eq!(project.focused_pipeline(), None);
}

#[test]
fn refresh_keeps_tweaks_until_the_pipelines_classes_change() {
    let mut project = project_with_focus_pipelines();
    project.focus_pipeline(PipelineId(1));
    project.toggle_class_visibility(ObjectClass::Global(3));

    // An edit that doesn't change the classes keeps the user's tweak.
    project.pipelines[0].name = "renamed".into();
    project.refresh_pipeline_focus();
    assert!(!project.is_class_visible(&ObjectClass::Global(3)));

    // A new class in the pipeline resets the focus to the new classes.
    project.pipelines[0] = focus_test_pipeline(1, ImageAddress::Channel(1), 3, 8);
    project.refresh_pipeline_focus();
    assert!(project.is_class_visible(&ObjectClass::Global(3)));
    assert!(project.is_class_visible(&ObjectClass::Global(8)));
    assert!(!project.is_class_visible(&ObjectClass::Global(4)));
}

#[test]
fn refresh_follows_a_changed_image_source_and_ends_for_a_deleted_pipeline() {
    let mut project = project_with_focus_pipelines();
    project.focus_pipeline(PipelineId(1));
    project.pipelines[0].image_source = ImageAddress::Channel(0);
    project.refresh_pipeline_focus();
    assert_eq!(
        project.get_image_channel_visibilities().get(&0),
        Some(&true)
    );
    assert_eq!(
        project.get_image_channel_visibilities().get(&1),
        Some(&false)
    );

    project.pipelines.remove(0);
    project.refresh_pipeline_focus();
    assert_eq!(project.focused_pipeline(), None);
}

// -- auto_add_classes_based_on_image_meta -------------------------------

/// Colour of the auto-added class called `name`.
fn auto_class_color(project: &ProjectWithRuntime, name: &str) -> u32 {
    project
        .classification
        .classes()
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no class {name}"))
        .color
}

/// Adds a second image whose single channel is called `channel_name`.
fn add_second_image(project: &mut ProjectWithRuntime, channel_name: &str) -> PathBuf {
    let rel_path = PathBuf::from("second.tif");
    let mut entry = project.images.list.values().next().unwrap().clone();
    entry.rel_path = rel_path.clone();
    let series = entry.series.get_mut(&0).unwrap();
    series.channels = BTreeMap::from([(
        0,
        ChannelSettings {
            name: channel_name.into(),
            emission_wave_length: Some(640.0),
            visible: None,
            histogram: None,
        },
    )]);
    project.images.list.insert(rel_path.clone(), entry);
    rel_path
}

#[test]
fn auto_add_classes_colours_classes_by_the_images_wavelength() {
    let mut project = project_with_one_image();
    project.auto_add_classes_based_on_image_meta();
    assert_eq!(
        auto_class_color(&project, "Ch0"),
        wavelength_to_rgb_u32(488.0)
    );
    assert_eq!(
        auto_class_color(&project, "Ch1"),
        wavelength_to_rgb_u32(561.0)
    );
}

#[test]
fn auto_add_classes_prefers_a_global_wavelength_over_the_images_own() {
    let mut project = project_with_one_image();
    project
        .images
        .settings
        .set_global_emission_wavel_length(0, 640.0);
    project.auto_add_classes_based_on_image_meta();
    assert_eq!(
        auto_class_color(&project, "Ch0"),
        wavelength_to_rgb_u32(640.0)
    );
    // Channels without a global value keep the image's own.
    assert_eq!(
        auto_class_color(&project, "Ch1"),
        wavelength_to_rgb_u32(561.0)
    );
}

#[test]
fn auto_add_classes_uses_the_selected_image_not_the_first() {
    let mut project = project_with_one_image();
    let second = add_second_image(&mut project, "Far red");
    project.set_current_image_path(&second);
    project.auto_add_classes_based_on_image_meta();

    let names: Vec<&str> = project
        .classification
        .classes()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        names.len(),
        2,
        "Background plus the second image's one channel"
    );
    assert!(names.contains(&"Far red"));
    assert_eq!(
        auto_class_color(&project, "Far red"),
        wavelength_to_rgb_u32(640.0)
    );
}

#[test]
fn auto_add_classes_falls_back_to_the_first_image_without_a_selection() {
    let mut project = project_with_one_image();
    add_second_image(&mut project, "Far red");
    project.rest_current_image_path();
    project
        .images
        .settings
        .set_global_emission_wavel_length(1, 405.0);
    project.auto_add_classes_based_on_image_meta();

    assert!(
        project
            .classification
            .classes()
            .iter()
            .all(|c| c.name != "Far red")
    );
    assert_eq!(
        auto_class_color(&project, "Ch0"),
        wavelength_to_rgb_u32(488.0)
    );
    // The global value applies without a selected image, too.
    assert_eq!(
        auto_class_color(&project, "Ch1"),
        wavelength_to_rgb_u32(405.0)
    );
}

#[test]
fn auto_add_classes_uses_the_images_selected_series() {
    let mut project = project_with_one_image();
    project.with_current_image_mut(|img| {
        let mut series = img.series[&0].clone();
        series.channels.retain(|id, _| *id == 0);
        series.channels.get_mut(&0).unwrap().name = "Series1".into();
        img.series.insert(1, series);
    });
    project.set_active_series(&1);
    project.auto_add_classes_based_on_image_meta();
    assert!(
        project
            .classification
            .classes()
            .iter()
            .any(|c| c.name == "Series1")
    );
    assert_eq!(project.classification.classes().len(), 2);
}

#[test]
fn auto_add_classes_creates_one_class_per_channel_of_the_first_image() {
    let mut project = project_with_one_image();
    // A fresh project always has exactly the auto-prepended Background
    // class and nothing else.
    assert_eq!(project.classification.classes().len(), 1);

    project.auto_add_classes_based_on_image_meta();

    let names: Vec<&str> = project
        .classification
        .classes()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    // Background, plus one class per channel.
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"Background"));
    assert!(names.contains(&"Ch0"));
    assert!(names.contains(&"Ch1"));
    // `add_class` (via `ClassificationExt`) assigns each new class a
    // fresh, unique id - the `ObjectClass::Unset` built in the collected
    // `Class` is only a placeholder that gets overwritten there, not the
    // final id. Background's own id is always `Valid(0)`, so all three
    // classes have a real `Valid` id here.
    let ids: Vec<u32> = project
        .classification
        .classes()
        .iter()
        .filter_map(|c| Some(c.id))
        .collect();
    assert_eq!(
        ids.len(),
        3,
        "Background and both new classes must have a real Valid id"
    );
}

// -- misc getters ---------------------------------------------------

#[test]
fn get_current_relative_path_and_find_image_round_trip() {
    let project = project_with_one_image();
    let rel = project.get_current_rel_image_path_cloned().unwrap();
    assert_eq!(rel, PathBuf::from("img.tif"));

    // `find_image` takes an *absolute* path and relativizes it against
    // `images.root`; with no root set, "absolute" and "relative" are the
    // same string here.
    assert!(project.find_image(&PathBuf::from("img.tif")).is_some());
    assert!(project.find_image(&PathBuf::from("missing.tif")).is_none());
}

#[test]
fn find_image_mut_allows_editing_and_rest_current_image_path_clears_the_selection() {
    let mut project = project_with_one_image();
    project
        .find_image_mut(&PathBuf::from("img.tif"))
        .unwrap()
        .file_size = 777;
    assert_eq!(
        project
            .images
            .list
            .get(Path::new("img.tif"))
            .unwrap()
            .file_size,
        777
    );

    assert!(project.get_current_image_path_cloned().is_some());
    project.rest_current_image_path();
    assert!(project.get_current_image_path_cloned().is_none());
    assert!(project.get_current_image_settings().is_none());
}

#[test]
fn current_image_settings_are_none_when_the_selected_path_is_not_in_the_list() {
    let mut project = project_with_one_image();
    project.set_current_image_path(&PathBuf::from("ghost.tif"));
    assert!(project.get_current_image_settings().is_none());
    assert!(project.get_current_image_settings_mut().is_none());
    assert!(project.with_current_image_mut(|_| ()).is_none());
}

#[test]
fn with_current_series_mut_uses_the_shown_series_when_the_selected_one_is_missing() {
    let mut project = project_with_one_image();
    let first = project
        .with_current_series_mut(|series| series.image_width)
        .expect("the image's series");
    // Series 99 the image lacks: its first one is shown - and changed.
    project.set_active_series(&99);
    assert_eq!(
        project.with_current_series_mut(|series| series.image_width),
        Some(first)
    );
}

// -- get_reference_object --------------------------------------------------

#[test]
fn get_reference_object_generates_a_grid_of_circular_objects() {
    let project = ProjectWithRuntime::default();
    let objects = project.get_reference_object().expect("always returns Some");
    assert_eq!(objects.len(), 30000);

    let first = &objects[0];
    assert_eq!(first.id, ObjectId(1));
    assert_eq!(first.area, 31415);
    assert_eq!(first.mask_data.len(), 5 * 5, "a 5x5 mask per object");
    assert_eq!(
        first.bbox[2] - first.bbox[0],
        5,
        "bbox width matches the object width"
    );
    assert_eq!(
        first.bbox[3] - first.bbox[1],
        5,
        "bbox height matches the object height"
    );
    assert_eq!(first.segmentation_class, SegmentationClass(1));

    let last = &objects[objects.len() - 1];
    assert_eq!(last.id, ObjectId(30000));
}

// -- get_selected_series_idx --------------------------------------------

/// A frame or plane kept from a bigger image (frame 13 of a time-lapse)
/// is shown as the image's last one - as the reader reads it - so the
/// objects analysed there (stamped with it) are listed and drawn. Before
/// the metadata is read, and for another image, the stored one counts.
#[test]
fn the_shown_plane_stays_within_the_current_images_frames_and_planes() {
    use crate::project::objects::ShownPlane;
    use evanalyzer_cfg::settings::images_settings::{TStackSettings, ZStackSettings};
    let mut project = ProjectWithRuntime::default();
    project.images.settings.set_global_t_stack(&TStackSettings {
        t_stack: 13,
        ..Default::default()
    });
    project.images.settings.z_stack = Some(ZStackSettings {
        z_projection: ZStackHandling::SingleStack,
        z_range: Some(4..=4),
    });
    project.set_current_image_path(&PathBuf::from("/d/single.vsi"));
    assert_eq!(project.get_shown_plane(), ShownPlane { z: Some(4), t: 13 });

    project.set_current_image_planes(1, 1);
    assert_eq!(project.get_shown_plane(), ShownPlane { z: Some(0), t: 0 });
    project.set_current_image_planes(10, 20);
    assert_eq!(project.get_shown_plane(), ShownPlane { z: Some(4), t: 13 });

    project.set_current_image_planes(1, 1);
    project.set_current_image_path(&PathBuf::from("/d/timelapse.tif"));
    assert_eq!(
        project.get_shown_plane(),
        ShownPlane { z: Some(4), t: 13 },
        "another image: its own counts, once read"
    );
}

/// The preview's objects (and a selection among them) go when another
/// image or series is shown - they were the last one's; the same image
/// again keeps them.
#[test]
fn another_image_or_series_forgets_the_preview_objects() {
    use evanalyzer_cfg::core_types::ObjectId;
    use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
    let mut project = ProjectWithRuntime::default();
    let preview = |project: &mut ProjectWithRuntime| {
        project.tmp_settings.preview_objects = vec![ObjectMetricSettings {
            id: ObjectId(7),
            ..Default::default()
        }];
        project.set_selected_object(Some(ObjectId(7)));
    };
    project.set_current_image_path(&PathBuf::from("/d/a.tif"));
    preview(&mut project);
    project.set_current_image_path(&PathBuf::from("/d/a.tif"));
    assert_eq!(
        project.get_preview_objects().len(),
        1,
        "the same image keeps them"
    );

    project.set_current_image_path(&PathBuf::from("/d/b.tif"));
    assert!(project.get_preview_objects().is_empty());
    assert_eq!(
        project.get_selected_object_id(),
        None,
        "the selection was one of them"
    );

    preview(&mut project);
    project.set_active_series(&0);
    assert_eq!(
        project.get_preview_objects().len(),
        1,
        "the same series keeps them"
    );
    project.set_active_series(&1);
    assert!(project.get_preview_objects().is_empty());
}

#[test]
fn get_selected_series_idx_defaults_to_zero_without_a_current_image() {
    let project = ProjectWithRuntime::default();
    assert_eq!(project.get_selected_series_idx(), 0);
}

#[test]
fn get_selected_series_idx_reflects_the_current_images_selected_series() {
    let mut project = project_with_one_image();
    project.with_current_image_mut(|img| {
        img.series.insert(1, img.series[&0].clone());
    });
    project.set_active_series(&1);
    assert_eq!(project.get_selected_series_idx(), 1);
}

// -- add_image_and_read_meta (reader-free branch only) --------------------

#[test]
fn add_image_and_read_meta_rejects_an_unsupported_extension_without_opening_a_reader() {
    let mut project = ProjectWithRuntime::default();
    let action = project.add_image_and_read_meta(&backend(), Path::new("notes.txt"));
    assert_eq!(action, ProjectAction::Failure("Unsupported device".into()));
    assert!(project.images.list.is_empty());
}

// -- is_supported_image / is_supported_image_path ------------------------

#[test]
fn is_supported_image_checks_the_extension_case_insensitively() {
    let project = ProjectWithRuntime::default();
    let backend = backend();
    assert!(project.is_supported_image(&backend, Path::new("a.tif")));
    assert!(project.is_supported_image(&backend, Path::new("a.TIF")));
    assert!(!project.is_supported_image(&backend, Path::new("a.txt")));
    assert!(!project.is_supported_image(&backend, Path::new("no_extension")));
}

#[test]
fn is_supported_image_path_matches_against_the_supported_formats_list() {
    let formats = vec!["czi".to_string(), "tif".to_string()];
    assert!(is_supported_image_path(Path::new("scan.czi"), &formats));
    assert!(is_supported_image_path(Path::new("scan.CZI"), &formats));
    assert!(!is_supported_image_path(Path::new("scan.pdf"), &formats));
    assert!(!is_supported_image_path(
        Path::new("no_extension"),
        &formats
    ));
    // Only the given formats count - a worker may accept others.
    assert!(!is_supported_image_path(Path::new("scan.vsi"), &formats));
}

// -- collect_images_at_root / collect_images_parallel --------------------

#[test]
fn collect_images_at_root_skips_a_folder_named_results_case_insensitively() {
    // The "results" check happens before any filesystem access, so this
    // is safe to call on paths that don't even exist.
    assert!(collect_images_at_root(&backend(), Path::new("/any/path/Results")).is_empty());
    assert!(collect_images_at_root(&backend(), Path::new("/any/path/RESULTS")).is_empty());
}

#[test]
fn collect_images_at_root_returns_empty_for_a_missing_directory() {
    assert!(
        collect_images_at_root(&backend(), Path::new("/definitely/does/not/exist/anywhere"))
            .is_empty()
    );
}

#[test]
fn collect_images_at_root_recurses_and_skips_unsupported_files_and_results_folders() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("doc.docx"), b"not an image").unwrap();
    let results = dir.path().join("results");
    std::fs::create_dir(&results).unwrap();
    // Even a supported extension inside "results" must never be opened -
    // if it were, this would fail trying to actually parse the file.
    std::fs::write(results.join("fake.tif"), b"not a real tiff").unwrap();

    assert!(collect_images_at_root(&backend(), dir.path()).is_empty());
}

#[test]
fn collect_images_parallel_delegates_to_collect_images_at_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"nope").unwrap();
    let project = ProjectWithRuntime::default();
    assert!(
        project
            .collect_images_parallel(&backend(), dir.path())
            .is_empty()
    );
}

// -- scan_image_folder_and_add / apply_scanned_images --------------------

#[test]
fn scan_image_folder_and_add_is_a_noop_without_a_root() {
    let mut project = ProjectWithRuntime::default();
    project.scan_image_folder_and_add(&backend());
    assert!(project.images.list.is_empty());
}

#[test]
fn scan_image_folder_and_add_finds_no_images_in_an_empty_root() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = ProjectWithRuntime::default();
    project.images.root = Some(dir.path().to_path_buf());
    project.scan_image_folder_and_add(&backend());
    assert!(project.images.list.is_empty());
}

#[test]
fn apply_scanned_images_clears_the_old_list_and_inserts_in_natural_sort_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = project_with_one_image();
    let meta = sample_image_meta("Ch0", 400.0);
    let found = vec![
        (dir.path().join("img10.tif"), meta.clone()),
        (dir.path().join("img2.tif"), meta),
    ];
    project.apply_scanned_images(found);

    assert!(
        !project.images.list.contains_key(Path::new("img.tif")),
        "the previous list must be cleared"
    );
    let keys: Vec<_> = project.images.list.keys().cloned().collect();
    assert_eq!(
        keys,
        vec![PathBuf::from("img2.tif"), PathBuf::from("img10.tif")],
        "natural sort orders img2 before img10"
    );
}

#[test]
fn a_rescan_forgets_folded_folders_without_images() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = project_with_one_image();
    project.images.root = Some(dir.path().to_path_buf());
    project.images.collapsed_folders = [PathBuf::from("A"), PathBuf::from("Gone")]
        .into_iter()
        .collect();
    let meta = sample_image_meta("Ch0", 400.0);
    project.apply_scanned_images(vec![
        (dir.path().join("A").join("a.tif"), meta.clone()),
        (dir.path().join("B").join("b.tif"), meta),
    ]);
    assert_eq!(
        project.images.collapsed_folders,
        [PathBuf::from("A")].into_iter().collect(),
        "A still holds an image, Gone none"
    );
}

#[test]
fn a_rescan_keeps_the_notes_of_the_images_found_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = project_with_one_image();
    project.images.root = Some(dir.path().to_path_buf());
    let meta = sample_image_meta("Ch0", 400.0);
    let found = vec![
        (dir.path().join("A").join("a.tif"), meta.clone()),
        (dir.path().join("A").join("b.tif"), meta),
    ];
    project.apply_scanned_images(found.clone());
    let review = ImageReview {
        note: "bubble".into(),
        ..Default::default()
    };
    project
        .images
        .list
        .get_mut(Path::new("A/a.tif"))
        .unwrap()
        .review = review.clone();
    project.apply_scanned_images(found);
    assert_eq!(
        project
            .images
            .list
            .get(Path::new("A/a.tif"))
            .unwrap()
            .review,
        review
    );
    assert!(
        project
            .images
            .list
            .get(Path::new("A/b.tif"))
            .unwrap()
            .review
            .is_empty()
    );
}

// -- save_project_as_template ---------------------------------------------

#[test]
fn save_project_as_template_writes_classification_plate_and_pipelines() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = ProjectWithRuntime::default();
    project.classification.classes_mut().push(class(1, "A"));
    let mut p = pipeline(1);
    p.name = "MyPipe".into();
    project.add_pipeline(p);

    let meta = evanalyzer_cfg::settings::meta_data::MetaData {
        name: "Tmpl".into(),
        ..Default::default()
    };
    let path = dir.path().join("out");
    project
        .save_project_as_template(&fs(), meta, &path)
        .expect("save should succeed");

    let expected = dir
        .path()
        .join(format!("out.{}", PROJECT_FILE_TEMPLATE_EXTENSIONS));
    assert!(expected.exists());
    let content = std::fs::read_to_string(&expected).unwrap();
    let parsed: ProjectTemplate = serde_json::from_str(&content).unwrap();
    // classes()[0] is the auto-prepended Background class.
    assert_eq!(parsed.classification.classes().len(), 2);
    assert_eq!(parsed.classification.classes()[1].name, "A");
    assert_eq!(parsed.pipelines.len(), 1);
    assert_eq!(parsed.pipelines[0].name, "MyPipe");
}

// -- new / new_project ----------------------------------------------------

#[test]
fn new_returns_an_arc_wrapped_default_project() {
    let project = ProjectWithRuntime::default();
    let fresh = project.new();
    // A fresh project always has exactly the auto-prepended Background
    // class and nothing else.
    assert_eq!(fresh.classification.classes().len(), 1);
    assert_eq!(
        fresh.classification.classes()[0].global(),
        ObjectClass::BACKGROUND
    );
    assert!(fresh.images.list.is_empty());
}

#[test]
fn new_project_creates_and_persists_a_fresh_default_project() {
    let dir = tempfile::tempdir().unwrap();
    let mut project = ProjectWithRuntime::default();
    let path = dir.path().join("newproj");

    let created = project.new_project(&fs(), &path).expect("should succeed");

    let expected = dir
        .path()
        .join(format!("newproj.{}", PROJECT_FILE_EXTENSIONS));
    assert!(expected.exists());
    assert_eq!(created.tmp_settings.current_project, Some(expected));
    // A fresh project always has exactly the auto-prepended Background
    // class and nothing else.
    assert_eq!(created.classification.classes().len(), 1);
    assert_eq!(
        created.classification.classes()[0].global(),
        ObjectClass::BACKGROUND
    );
}

// -- load_project schema version check ------------------------------------

#[test]
fn load_project_rejects_a_file_written_by_a_newer_schema_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("future.evaproj");

    let settings = ProjectSettings::default();
    let mut value = serde_json::to_value(&settings).unwrap();
    value["schemaVersion"] = serde_json::json!(evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION + 1);
    std::fs::write(&path, serde_json::to_string(&value).unwrap()).unwrap();

    match load_project(&fs(), &path) {
        Err(err) => assert!(err.to_string().contains("newer version")),
        Ok(_) => panic!("a newer schema version must be rejected"),
    }
}

// -- Less common paths -------------------------------------------------

fn fixture_tif() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../core/tests/slice_Z0_C0_T0.tif"
    ))
}

fn classes_of_first_object(project: &ProjectWithRuntime) -> Vec<ObjectClass> {
    project.get_objects().unwrap()[0]
        .object_class
        .iter()
        .copied()
        .collect()
}

/// Replacing an object's classes leaves it with only the new one.
#[test]
fn replace_classes_of_object_keeps_only_the_new_class() {
    let mut project = project_with_one_image();
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        ..Default::default()
    });
    project.add_class_to_object(ObjectId(1), ObjectClass::Global(1));
    project.add_class_to_object(ObjectId(1), ObjectClass::Global(2));

    project.replace_classes_of_object(ObjectId(1), ObjectClass::Global(3));

    assert_eq!(classes_of_first_object(&project), [ObjectClass::Global(3)]);
}

/// Class changes for an object id the image doesn't have change no
/// other object - and without an image they don't panic.
#[test]
fn class_changes_for_an_unknown_object_change_nothing() {
    let mut project = project_with_one_image();
    project.add_object(&ObjectMetricSettings {
        id: ObjectId(1),
        ..Default::default()
    });
    project.add_class_to_object(ObjectId(1), ObjectClass::Global(1));

    project.add_class_to_object(ObjectId(9), ObjectClass::Global(2));
    project.replace_classes_of_object(ObjectId(9), ObjectClass::Global(3));
    project.remove_class_from_object(ObjectId(9), &ObjectClass::Global(1));

    assert_eq!(classes_of_first_object(&project), [ObjectClass::Global(1)]);
    let mut empty = ProjectWithRuntime::default();
    empty.replace_classes_of_object(ObjectId(1), ObjectClass::Global(3));
    assert!(empty.get_objects().is_none());
}

/// An image entry without any series: channel preferences have nowhere
/// to go and change nothing.
#[test]
fn preferences_for_an_image_without_series_change_nothing() {
    let mut project = project_with_one_image();
    project
        .get_current_image_settings_mut()
        .unwrap()
        .series
        .clear();

    project.set_image_preferences(&BTreeMap::from([(0, false)]));

    assert!(
        project
            .get_current_image_settings()
            .unwrap()
            .series
            .is_empty()
    );
}

/// While a pipeline is focused, a channel switched off for the image
/// only changes the focus: the image's own settings stay, and leaving
/// the focus shows every channel again.
#[test]
fn while_focused_image_preferences_change_only_the_focus() {
    let mut project = project_with_focus_pipelines();
    project.focus_pipeline(PipelineId(1));

    project.set_image_preferences(&BTreeMap::from([(0, true), (1, false)]));

    assert_eq!(
        project.get_image_channel_visibilities(),
        BTreeMap::from([(0, true), (1, false)])
    );
    let series = project.get_selected_image_series().unwrap();
    assert!(series.channels.values().all(|c| c.visible.is_none()));
    project.clear_pipeline_focus();
    assert_eq!(
        project.get_image_channel_visibilities(),
        BTreeMap::from([(0, true), (1, true)])
    );
}

/// While focused, a class the focus hides is shown when toggled.
#[test]
fn toggling_a_hidden_class_while_focused_shows_it() {
    let mut project = project_with_focus_pipelines();
    project.focus_pipeline(PipelineId(1)); // shows classes 3 and 4
    assert!(!project.is_class_visible(&ObjectClass::Global(5)));

    project.toggle_class_visibility(ObjectClass::Global(5));

    assert!(project.is_class_visible(&ObjectClass::Global(5)));
    assert!(
        project.tmp_settings.hidden_classes.is_empty(),
        "the focus only"
    );
}

/// A class the focus shows keeps being shown under its new id.
#[test]
fn a_changed_class_id_stays_shown_in_the_focus() {
    let mut project = project_with_focus_pipelines();
    project.classification.classes_mut().push(class(3, "Three"));
    project.focus_pipeline(PipelineId(1)); // shows classes 3 and 4

    project.change_class_id(3, 30).unwrap();

    assert!(project.is_class_visible(&ObjectClass::Global(30)));
    assert!(!project.is_class_visible(&ObjectClass::Global(3)));
}

/// Resetting the global wave length of a channel the project has no
/// entry for leaves it unset: the image's own value shows.
#[test]
fn resetting_an_unset_global_wave_length_falls_back_to_the_image() {
    let mut project = project_with_one_image();

    project
        .images
        .settings
        .reset_global_emission_wavel_length(1);

    assert_eq!(
        project.images.settings.channels[&1].emission_wave_length,
        None
    );
    assert_eq!(project.get_emission_wave_length(1), 561.0);
}

/// Every assignable colour is already shown: a channel without one
/// gets one anyway - the palette starts over.
#[test]
fn a_missing_wavelength_is_assigned_even_when_every_colour_is_shown() {
    let mut project = ProjectWithRuntime::default();
    let mut channels: Vec<(i32, f32)> = ASSIGNED_WAVELENGTHS
        .iter()
        .enumerate()
        .map(|(i, nm)| (i as i32, *nm))
        .collect();
    let missing = channels.len() as i32;
    channels.push((missing, 0.0));

    assert!(project.assign_missing_wavelengths(&channels));

    assert_eq!(
        project.images.settings.channels[&missing].emission_wave_length,
        Some(ASSIGNED_WAVELENGTHS[0])
    );
    assert_eq!(project.images.settings.channels.len(), 1, "only it");
}

/// Grayscale with the selected channel hidden: the first visible one
/// stays visible.
#[test]
fn grayscale_keeps_the_first_visible_channel_when_the_selected_one_is_hidden() {
    let mut project = project_with_three_unnamed_channels();
    project.set_global_preferences(&BTreeMap::from([(0, false), (1, true), (2, true)]));
    project.images.settings.set_global_selected_channel(&0);

    project.set_grayscale(true);

    assert_eq!(
        project.get_image_channel_visibilities(),
        BTreeMap::from([(0, false), (1, true), (2, false)])
    );
}

/// Clearing the frame interval of a project without time settings adds
/// none.
#[test]
fn clearing_an_unset_frame_interval_adds_no_time_settings() {
    let mut project = ProjectWithRuntime::default();
    project.images.settings.t_stack = None;

    project.set_frame_interval(None);

    assert!(project.images.settings.t_stack.is_none());
}

/// Without an image root, a known image's path is the listed one.
#[test]
fn without_a_root_a_known_image_path_stays_as_listed() {
    let mut project = project_with_one_image();
    project.images.root = None;

    assert_eq!(
        project.get_image_absolute_path_from_relative(Path::new("img.tif")),
        Some(PathBuf::from("img.tif"))
    );
}

/// A readable image is added under its name, its folder the root.
#[test]
fn add_image_and_read_meta_adds_a_readable_image() {
    let mut project = ProjectWithRuntime::default();
    let path = fixture_tif();

    let action = project.add_image_and_read_meta(&backend(), &path);

    assert_eq!(action, ProjectAction::Success);
    assert_eq!(project.images.root.as_deref(), path.parent());
    let keys: Vec<&PathBuf> = project.images.list.keys().collect();
    assert_eq!(keys, [&PathBuf::from("slice_Z0_C0_T0.tif")]);
}

/// A file with an image extension that isn't an image: a failure, not
/// added.
#[test]
fn add_image_and_read_meta_reports_an_unreadable_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.tif");
    std::fs::write(&path, b"not a real tiff").unwrap();
    let mut project = ProjectWithRuntime::default();

    let action = project.add_image_and_read_meta(&backend(), &path);

    assert!(matches!(action, ProjectAction::Failure(_)), "{action:?}");
    assert!(project.images.list.is_empty());
}

/// Metadata without a full-resolution level: the series is added with
/// no size.
#[test]
fn add_image_to_list_without_a_full_resolution_has_no_size() {
    let mut project = ProjectWithRuntime::default();
    let mut meta = sample_image_meta("DAPI", 460.0);
    meta.series.get_mut(&0).unwrap().resolutions.clear();

    project.add_image_to_list(Path::new("a.tif"), Path::new("/x/a.tif"), &meta);

    let entry = project.images.list.get(Path::new("a.tif")).unwrap();
    let series = &entry.series[&0];
    assert_eq!((series.image_width, series.image_height), (0, 0));
    assert_eq!(entry.file_size, 0);
    assert_eq!(series.channels[&0].name, "DAPI");
}

/// A scan finds the readable image and skips a broken one with an
/// image extension.
#[test]
fn collect_images_at_root_skips_unreadable_images() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(fixture_tif(), dir.path().join("good.tif")).unwrap();
    std::fs::write(dir.path().join("broken.tif"), b"not a real tiff").unwrap();

    let found: Vec<PathBuf> = collect_images_at_root(&backend(), dir.path())
        .into_iter()
        .map(|(path, _)| path)
        .collect();

    assert_eq!(found, [dir.path().join("good.tif")]);
}

/// A project file that isn't text is refused with a parse error naming
/// the file.
#[test]
fn load_project_rejects_a_file_that_is_not_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binary.evaproj");
    std::fs::write(&path, [0xff, 0xfe, 0x00, 0x80]).unwrap();

    match load_project(&fs(), &path) {
        Err(InternalErrors::ParseError(message)) => {
            assert!(message.contains("is not a text file"), "{message}");
            assert!(message.contains("binary.evaproj"), "{message}");
        }
        Err(other) => panic!("a parse error, got {other:?}"),
        Ok(_) => panic!("a binary file must be rejected"),
    }
}

fn module_with_steps(ids: &[u32]) -> evanalyzer_cfg::settings::modules::ModuleDefinition {
    evanalyzer_cfg::settings::modules::ModuleDefinition {
        key: "TWICE".into(),
        meta: Default::default(),
        classes: vec![],
        steps: ids
            .iter()
            .map(|id| PipelineStepSettings {
                id: evanalyzer_cfg::core_types::StepId(*id),
                enabled: true,
                content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                    PipelineCommand::Blur(Default::default()),
                ),
            })
            .collect(),
        parameters: vec![],
        origin: None,
    }
}

/// A module with two steps of one id can't be loaded: its parameters
/// couldn't tell which step they set.
#[test]
fn load_project_rejects_a_module_with_two_steps_of_one_id() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("twice.evaproj");
    let mut settings = ProjectSettings::default();
    settings.modules.push(module_with_steps(&[1, 1]));
    std::fs::write(&path, serde_json::to_string(&settings).unwrap()).unwrap();

    match load_project(&fs(), &path) {
        Err(err) => assert!(err.to_string().contains("two steps have the id 1"), "{err}"),
        Ok(_) => panic!("two steps of one id must be rejected"),
    }
}

/// Module mistakes (a step using a module the project lacks) are only
/// logged on load: the project opens with the step as it is.
#[test]
fn load_project_opens_a_project_with_module_mistakes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mistakes.evaproj");
    let mut settings = ProjectSettings::default();
    let mut pipeline = pipeline(1);
    pipeline.steps.push(PipelineStepSettings {
        id: evanalyzer_cfg::core_types::StepId(1),
        enabled: true,
        content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Module(
            evanalyzer_cfg::settings::modules::ModuleInstance {
                key: "MISSING".into(),
                values: Default::default(),
            },
        ),
    });
    settings.pipelines.push(pipeline);
    std::fs::write(&path, serde_json::to_string(&settings).unwrap()).unwrap();

    let loaded = load_project(&fs(), &path).unwrap();

    let evanalyzer_cfg::settings::pipeline_settings::StepContent::Module(instance) =
        &loaded.pipelines[0].steps[0].content
    else {
        panic!("the module step");
    };
    assert_eq!(instance.key, "MISSING");
}

/// A template module with two steps of one id comes along unchanged
/// (only logged) - its steps aren't renumbered.
#[test]
fn a_template_module_with_two_steps_of_one_id_is_kept_as_it_is() {
    let mut project = ProjectWithRuntime::default();
    let template = evanalyzer_cfg::settings::templates::ProjectTemplate {
        modules: vec![module_with_steps(&[1, 1])],
        ..Default::default()
    };

    project.apply_project_template(&template);

    let ids: Vec<u32> = project.modules[0].steps.iter().map(|s| s.id.0).collect();
    assert_eq!(ids, [1, 1]);
}
