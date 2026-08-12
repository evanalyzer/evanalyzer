use evanalyzer_cfg::settings::parameter_def::ParamType;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_settings::PipelineSettings;
use pathdiff::diff_paths;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn get_relative_key(image_path: &Path, images_root: Option<&PathBuf>) -> Option<PathBuf> {
    match images_root {
        Some(root) => diff_paths(image_path, root),
        None => Some(image_path.to_path_buf()),
    }
}

/// Returns `(field_name, new_value)` for every top-level `ParamType::FilePath`
/// field on `command` whose current value passes `keep`, with `new_value`
/// computed by `rewrite`. Driven entirely by `to_parameters()`'s generic,
/// codegen-derived `ParamType` classification (any `PathBuf`-typed field
/// automatically becomes `ParamType::FilePath` - see
/// `crates/cfg/build/pipeline_commands_generator.rs`'s `"PathBuf" =>` arm),
/// not a hardcoded list of command/field names - see this module's doc
/// comment on [`relativize_file_paths`] for why that matters. Collecting into
/// a `Vec` first (rather than calling `apply_param_change` inline) sidesteps
/// borrowing `to_parameters()`'s `&self` and `apply_param_change`'s `&mut
/// self` at the same time.
fn file_path_changes(
    command: &PipelineCommand,
    keep: impl Fn(&Path) -> bool,
    rewrite: impl Fn(&Path) -> Option<PathBuf>,
) -> Vec<(String, PathBuf)> {
    command
        .to_parameters()
        .into_iter()
        .filter(|p| p.param_type == ParamType::FilePath && !p.value.is_empty())
        .filter_map(|p| {
            let path = PathBuf::from(&p.value);
            if !keep(&path) {
                return None;
            }
            rewrite(&path).map(|new_path| (p.name, new_path))
        })
        .collect()
}

/// Rewrites every `PathBuf`-typed command field in `pipelines` (Cellpose /
/// Stardist / UNet / PixelClassifier's `model_path` today - and, without any
/// further change here, whatever field a future command declares the same
/// way) to be relative to `project_dir`, so a saved project stays valid when
/// the project folder is moved or copied to another machine - mirrors
/// `get_relative_key`'s handling of image paths.
///
/// This is intentionally generic rather than matching specific command
/// variants/field names: adding a `pub some_path: PathBuf` field to any
/// `#[derive(CommandsMeta)]` struct already makes it a `ParamType::FilePath`
/// (that's what drives the "Browse…" button in the pipeline editor too), and
/// that alone is now enough to make it project-relative on save - no edit
/// needed here. See `README.md`'s "Adding a new pipeline command" section and
/// the `file_extensions` doc comment in `crates/core/macros/src/lib.rs`.
///
/// A path that can't be related to `project_dir` (e.g. a different drive on
/// Windows) is left absolute.
pub fn relativize_file_paths(pipelines: &mut [PipelineSettings], project_dir: &Path) {
    for pipeline in pipelines {
        for step in &mut pipeline.steps {
            let changes = file_path_changes(
                &step.command,
                |path| path.is_absolute(),
                |path| diff_paths(path, project_dir),
            );
            for (name, relative) in changes {
                step.command
                    .apply_param_change(&name, &relative.to_string_lossy());
            }
        }
    }
}

/// The inverse of [`relativize_file_paths`]: resolves every `PathBuf`-typed
/// command field against `project_dir` when it's stored as a relative path.
/// Already-absolute paths (e.g. projects saved before this existed) are left
/// untouched. Generic for the same reason `relativize_file_paths` is - see
/// its doc comment.
pub fn resolve_file_paths(pipelines: &mut [PipelineSettings], project_dir: &Path) {
    for pipeline in pipelines {
        for step in &mut pipeline.steps {
            let changes = file_path_changes(
                &step.command,
                |path| path.is_relative(),
                |path| Some(project_dir.join(path)),
            );
            for (name, absolute) in changes {
                step.command
                    .apply_param_change(&name, &absolute.to_string_lossy());
            }
        }
    }
}

pub fn get_file_size(path: &Path) -> std::io::Result<u64> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.len())
}

pub fn is_in_root(image_path: &Path, data_root: &Path) -> bool {
    // Basic check: Does the image path begin with the data_root string?
    image_path.starts_with(data_root)
}

pub fn wavelength_to_rgb_u32(wavelength: f32) -> u32 {
    let color = wavelength_to_rgb_float(wavelength);
    let ret_color: u32 = ((color[0] * 255.0) as u32) << 16
        | ((color[1] * 255.0) as u32) << 8
        | (color[2] * 255.0) as u32;
    ret_color
}

/// Converts a wavelength in nm to an RGB [f32; 3] color.
///
/// A wavelength outside the visible spectrum (or within it but near an edge,
/// e.g. deep violet/infrared) is clamped into the fully-saturated plateau
/// (420-700nm, where `factor` below is 1.0) rather than left to fade toward
/// black - a channel with an out-of-range emission wavelength should still
/// render as a clearly visible violet/red, not dim to invisible.
pub fn wavelength_to_rgb_float(wavelength: f32) -> [f32; 3] {
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);

    // Images without given emission wave length have default value 0.
    // In this case we show a grayscale value. Because of float we assume < 1
    if wavelength <= 1.0 {
        return [1.0, 1.0, 1.0];
    }

    let wavelength = wavelength.clamp(420.0, 700.0);

    // Pure red
    if wavelength == 635.0 {
        return [1.0, 0.0, 0.0];
    }

    // Pure green
    if wavelength == 532.0 {
        return [0.0, 1.0, 0.0];
    }

    // Pure blue
    if wavelength == 450.0 {
        return [0.0, 0.0, 1.0];
    }

    // Calculate base RGB components
    if (380.0..440.0).contains(&wavelength) {
        r = -(wavelength - 440.0) / (440.0 - 380.0);
        b = 1.0;
    } else if (440.0..490.0).contains(&wavelength) {
        g = (wavelength - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (490.0..510.0).contains(&wavelength) {
        g = 1.0;
        b = -(wavelength - 510.0) / (510.0 - 490.0);
    } else if (510.0..580.0).contains(&wavelength) {
        r = (wavelength - 510.0) / (580.0 - 510.0);
        g = 1.0;
    } else if (580.0..645.0).contains(&wavelength) {
        r = 1.0;
        g = -(wavelength - 645.0) / (645.0 - 580.0);
    } else if (645.0..781.0).contains(&wavelength) {
        r = 1.0;
    }

    // No fade-out factor needed here: `wavelength` was clamped to
    // 420.0..=700.0 above, which is exactly the plateau the original
    // fade-out curve used for full (1.0) intensity - every value reaching
    // this point is already fully saturated.
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- get_relative_key ----

    #[test]
    fn get_relative_key_returns_the_diff_from_root_when_a_root_is_given() {
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("/data/plate1/well_A1/img.tif");
        assert_eq!(
            get_relative_key(image, Some(&root)),
            Some(PathBuf::from("well_A1/img.tif"))
        );
    }

    #[test]
    fn get_relative_key_returns_the_full_path_when_there_is_no_root() {
        let image = Path::new("/data/plate1/img.tif");
        assert_eq!(get_relative_key(image, None), Some(image.to_path_buf()));
    }

    #[test]
    fn get_relative_key_diffs_via_parent_segments_even_across_unrelated_absolute_roots() {
        // Two absolute paths always produce a relative diff (via `..`
        // segments), even with nothing in common - `diff_paths` only
        // returns `None` on an absolute/relative mismatch (see the next
        // test), not for "unrelated" paths.
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("/other/img.tif");
        assert_eq!(
            get_relative_key(image, Some(&root)),
            Some(PathBuf::from("../../other/img.tif"))
        );
    }

    #[test]
    fn get_relative_key_returns_none_for_a_relative_image_path_against_an_absolute_root() {
        // `diff_paths` returns `None` exactly when the two paths' "is
        // absolute" status differs and the candidate path is the relative
        // one - it has no way to express a relative-to-root diff for a path
        // that isn't anchored anywhere itself.
        let root = PathBuf::from("/data/plate1");
        let image = Path::new("relative/img.tif");
        assert_eq!(get_relative_key(image, Some(&root)), None);
    }

    // ---- relativize_file_paths / resolve_file_paths ----

    fn pipeline_with(command: PipelineCommand) -> PipelineSettings {
        use evanalyzer_cfg::core_types::{ImageAddress, PipelineId};

        PipelineSettings {
            id: PipelineId(0),
            name: None,
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![
                evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings {
                    enabled: true,
                    command,
                },
            ],
        }
    }

    fn pixel_classifier_model_path(pipeline: &PipelineSettings) -> &PathBuf {
        let PipelineCommand::PixelClassifier(settings) = &pipeline.steps[0].command else {
            panic!("expected a PixelClassifier command");
        };
        &settings.model_path
    }

    #[test]
    fn relativize_file_paths_rewrites_an_absolute_path_under_the_project_dir() {
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;

        let mut pipelines = vec![pipeline_with(PipelineCommand::PixelClassifier(
            PixelClassifierSettings {
                model_path: PathBuf::from("/data/project/models/nuclei.evamodel"),
                ..Default::default()
            },
        ))];

        relativize_file_paths(&mut pipelines, Path::new("/data/project"));

        assert_eq!(
            pixel_classifier_model_path(&pipelines[0]),
            &PathBuf::from("models/nuclei.evamodel")
        );
    }

    #[test]
    fn relativize_file_paths_leaves_an_empty_path_untouched() {
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;

        let mut pipelines = vec![pipeline_with(PipelineCommand::PixelClassifier(
            PixelClassifierSettings::default(),
        ))];

        relativize_file_paths(&mut pipelines, Path::new("/data/project"));

        assert_eq!(pixel_classifier_model_path(&pipelines[0]), &PathBuf::new());
    }

    #[test]
    fn relativize_file_paths_ignores_commands_without_a_file_path_field() {
        use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;

        let mut pipelines = vec![pipeline_with(
            PipelineCommand::Blur(BlurSettings::default()),
        )];

        // Must not panic on a command variant with no `ParamType::FilePath` field.
        relativize_file_paths(&mut pipelines, Path::new("/data/project"));
    }

    #[test]
    fn relativize_file_paths_handles_stardist_and_unet_the_same_way_as_pixel_classifier() {
        // No command name is special-cased in the implementation - any
        // `PathBuf` field works the same way, demonstrated here with two
        // more of the four current FilePath-bearing commands (Cellpose is
        // covered separately by the round-trip test below).
        use evanalyzer_cfg::settings::pipeline_command_settings::{StardistSettings, UNetSettings};

        let mut pipelines = vec![
            pipeline_with(PipelineCommand::Stardist(StardistSettings {
                model_path: PathBuf::from("/data/project/models/stardist.pt"),
                ..Default::default()
            })),
            pipeline_with(PipelineCommand::UNet(UNetSettings {
                model_path: PathBuf::from("/data/project/models/unet.pt"),
                ..Default::default()
            })),
        ];

        relativize_file_paths(&mut pipelines, Path::new("/data/project"));

        let PipelineCommand::Stardist(stardist) = &pipelines[0].steps[0].command else {
            panic!("expected a Stardist command");
        };
        assert_eq!(stardist.model_path, PathBuf::from("models/stardist.pt"));

        let PipelineCommand::UNet(unet) = &pipelines[1].steps[0].command else {
            panic!("expected a UNet command");
        };
        assert_eq!(unet.model_path, PathBuf::from("models/unet.pt"));
    }

    #[test]
    fn resolve_file_paths_joins_a_relative_path_onto_the_project_dir() {
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;

        let mut pipelines = vec![pipeline_with(PipelineCommand::PixelClassifier(
            PixelClassifierSettings {
                model_path: PathBuf::from("models/nuclei.evamodel"),
                ..Default::default()
            },
        ))];

        resolve_file_paths(&mut pipelines, Path::new("/data/project"));

        assert_eq!(
            pixel_classifier_model_path(&pipelines[0]),
            &PathBuf::from("/data/project/models/nuclei.evamodel")
        );
    }

    #[test]
    fn resolve_file_paths_leaves_an_already_absolute_path_untouched() {
        use evanalyzer_cfg::settings::pipeline_command_settings::PixelClassifierSettings;

        // A project saved before this feature existed, or one whose model
        // lives outside the project directory - must not be rewritten.
        let mut pipelines = vec![pipeline_with(PipelineCommand::PixelClassifier(
            PixelClassifierSettings {
                model_path: PathBuf::from("/elsewhere/models/nuclei.evamodel"),
                ..Default::default()
            },
        ))];

        resolve_file_paths(&mut pipelines, Path::new("/data/project"));

        assert_eq!(
            pixel_classifier_model_path(&pipelines[0]),
            &PathBuf::from("/elsewhere/models/nuclei.evamodel")
        );
    }

    #[test]
    fn relativize_then_resolve_file_paths_round_trips() {
        use evanalyzer_cfg::settings::pipeline_command_settings::CellposeSettings;

        let original = PathBuf::from("/data/project/models/cyto.pt");
        let mut pipelines = vec![pipeline_with(PipelineCommand::Cellpose(CellposeSettings {
            model_path: original.clone(),
            ..Default::default()
        }))];
        let project_dir = Path::new("/data/project");

        relativize_file_paths(&mut pipelines, project_dir);
        resolve_file_paths(&mut pipelines, project_dir);

        let PipelineCommand::Cellpose(settings) = &pipelines[0].steps[0].command else {
            panic!("expected a Cellpose command");
        };
        assert_eq!(settings.model_path, original);
    }

    // ---- get_file_size ----

    #[test]
    fn get_file_size_returns_the_byte_length_of_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, [0u8; 42]).unwrap();
        assert_eq!(get_file_size(&path).unwrap(), 42);
    }

    #[test]
    fn get_file_size_errors_for_a_missing_file() {
        assert!(get_file_size(Path::new("/does/not/exist.bin")).is_err());
    }

    // ---- is_in_root ----

    #[test]
    fn is_in_root_true_for_a_path_under_the_root() {
        assert!(is_in_root(
            Path::new("/data/plate1/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    #[test]
    fn is_in_root_false_for_a_path_outside_the_root() {
        assert!(!is_in_root(
            Path::new("/other/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    #[test]
    fn is_in_root_false_for_a_sibling_directory_with_a_shared_prefix_string() {
        // `starts_with` is a path-component comparison, not a raw string
        // prefix, so "/data/plate10" must not be considered inside
        // "/data/plate1".
        assert!(!is_in_root(
            Path::new("/data/plate10/img.tif"),
            Path::new("/data/plate1")
        ));
    }

    // ---- wavelength_to_rgb_float / wavelength_to_rgb_u32 ----

    #[test]
    fn wavelength_at_or_below_one_is_treated_as_unset_and_returns_white() {
        assert_eq!(wavelength_to_rgb_float(0.0), [1.0, 1.0, 1.0]);
        assert_eq!(wavelength_to_rgb_float(1.0), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn wavelength_pure_colors_are_exact() {
        assert_eq!(wavelength_to_rgb_float(635.0), [1.0, 0.0, 0.0]);
        assert_eq!(wavelength_to_rgb_float(532.0), [0.0, 1.0, 0.0]);
        assert_eq!(wavelength_to_rgb_float(450.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn wavelength_outside_the_visible_spectrum_clamps_to_a_visible_edge_color_instead_of_black() {
        // Too far UV clamps to the same color as 420nm (the start of the
        // fully-saturated plateau) - visibly violet-blue, not black.
        assert_eq!(
            wavelength_to_rgb_float(300.0),
            wavelength_to_rgb_float(420.0)
        );
        assert_ne!(wavelength_to_rgb_float(300.0), [0.0, 0.0, 0.0]);

        // Too far infrared clamps to the same color as 700nm (the end of
        // the plateau) - visibly red, not black.
        assert_eq!(
            wavelength_to_rgb_float(800.0),
            wavelength_to_rgb_float(700.0)
        );
        assert_eq!(wavelength_to_rgb_float(800.0), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn wavelength_near_the_visible_edges_is_fully_saturated_not_dimmed() {
        // Previously these faded toward black approaching 380nm/780nm; now
        // every in-spectrum wavelength is clamped into the 420-700nm
        // full-intensity plateau, so even a near-edge wavelength like
        // 390nm or 770nm renders at full brightness rather than dim.
        assert_eq!(
            wavelength_to_rgb_float(390.0),
            wavelength_to_rgb_float(420.0)
        );
        assert_eq!(
            wavelength_to_rgb_float(770.0),
            wavelength_to_rgb_float(700.0)
        );
    }

    #[test]
    fn wavelength_to_rgb_u32_packs_each_float_channel_into_its_own_byte() {
        // Regression test: the blue byte was previously packed from
        // `color[0]` (red) instead of `color[2]` (blue), so a pure-blue
        // wavelength rendered as black (0x000000) instead of 0x0000FF.
        assert_eq!(wavelength_to_rgb_u32(635.0), 0x00FF0000, "pure red");
        assert_eq!(wavelength_to_rgb_u32(532.0), 0x0000FF00, "pure green");
        assert_eq!(wavelength_to_rgb_u32(450.0), 0x000000FF, "pure blue");
    }

    #[test]
    fn wavelength_to_rgb_u32_white_for_an_unset_wavelength() {
        assert_eq!(wavelength_to_rgb_u32(0.0), 0x00FFFFFF);
    }
}
