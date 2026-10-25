//! # utils
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::next_measurement_id;
use evanalyzer_cfg::core_types::{InternalErrors, StepId};
use evanalyzer_cfg::settings::modules::ModuleDefinition;
use evanalyzer_cfg::settings::parameter_def::ParamType;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_settings::{
    PipelineSettings, PipelineStepSettings, StepContent,
};
use evanalyzer_cfg::settings::project_settings::ProjectSettings;
use pathdiff::diff_paths;
use std::collections::HashSet;
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
        relativize_step_paths(&mut pipeline.steps, project_dir);
    }
}

/// [`relativize_file_paths`] for the steps of `modules`' definitions.
pub fn relativize_module_file_paths(modules: &mut [ModuleDefinition], project_dir: &Path) {
    for module in modules {
        relativize_step_paths(&mut module.steps, project_dir);
    }
}

fn relativize_step_paths(steps: &mut [PipelineStepSettings], project_dir: &Path) {
    // A module instance holds no paths itself - its definition's steps do.
    for command in steps.iter_mut().filter_map(|step| step.command_mut()) {
        let changes = file_path_changes(
            command,
            |path| path.is_absolute(),
            |path| diff_paths(path, project_dir),
        );
        for (name, relative) in changes {
            command.apply_param_change(&name, &relative.to_string_lossy());
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
        resolve_step_paths(&mut pipeline.steps, project_dir);
    }
}

/// [`resolve_file_paths`] for the steps of `modules`' definitions.
pub fn resolve_module_file_paths(modules: &mut [ModuleDefinition], project_dir: &Path) {
    for module in modules {
        resolve_step_paths(&mut module.steps, project_dir);
    }
}

fn resolve_step_paths(steps: &mut [PipelineStepSettings], project_dir: &Path) {
    for command in steps.iter_mut().filter_map(|step| step.command_mut()) {
        let changes = file_path_changes(
            command,
            |path| path.is_relative(),
            |path| Some(project_dir.join(path)),
        );
        for (name, absolute) in changes {
            command.apply_param_change(&name, &absolute.to_string_lossy());
        }
    }
}

/// Gives every step of one list (a pipeline, or a module definition) that has
/// no id yet - or one an earlier step of the list already has - the next
/// number after the list's highest. The first of two equal ids keeps it.
/// Call it after adding steps to a list. Returns how many got a new id.
pub fn assign_step_ids(steps: &mut [PipelineStepSettings]) -> usize {
    let mut next = steps.iter().map(|s| s.id.0).max().unwrap_or(0) + 1;
    let mut seen = std::collections::HashSet::new();
    let mut assigned = 0;
    for step in steps {
        if step.id == StepId::UNASSIGNED || !seen.insert(step.id) {
            step.id = StepId(next);
            seen.insert(step.id);
            next += 1;
            assigned += 1;
        }
    }
    assigned
}

/// Gives each measurement step without an ID the next free one
/// ("pearson_01", …) - in the pipelines and the module definitions: one set
/// of names for the project. A name already given stays, also one used
/// twice: the checks report that (`check_project`). How many got one.
pub fn assign_measurement_ids(project: &mut ProjectSettings) -> usize {
    let mut names: Vec<(&'static str, &mut String)> = project
        .pipelines
        .iter_mut()
        .flat_map(|p| p.steps.iter_mut())
        .chain(project.modules.iter_mut().flat_map(|m| m.steps.iter_mut()))
        .filter_map(|step| match &mut step.content {
            StepContent::Command(command) => {
                let prefix = command.measurement_id_prefix()?;
                command.measurement_id_mut().map(|name| (prefix, name))
            }
            StepContent::Module(_) => None,
        })
        .collect();
    fill_measurement_ids(&mut names)
}

/// The empty names of `names` (prefix, name) get the next free one of their
/// prefix; how many did.
fn fill_measurement_ids(names: &mut [(&str, &mut String)]) -> usize {
    let mut taken: HashSet<String> = names
        .iter()
        .filter(|(_, name)| !name.trim().is_empty())
        .map(|(_, name)| name.to_lowercase())
        .collect();
    let mut filled = 0;
    for (prefix, name) in names.iter_mut() {
        if name.trim().is_empty() {
            **name = next_measurement_id(prefix, &taken);
            taken.insert(name.to_lowercase());
            filled += 1;
        }
    }
    filled
}

/// Numbers the steps of a module definition that have no id yet, like
/// [`assign_step_ids`] - but two steps with the same id are an error, not
/// renumbered: the module's bindings refer to steps by id, so which of the
/// two one means can't be known. Returns how many got an id.
pub fn assign_module_step_ids(module: &mut ModuleDefinition) -> Result<usize, InternalErrors> {
    let mut seen = std::collections::HashSet::new();
    for step in &module.steps {
        if step.id != StepId::UNASSIGNED && !seen.insert(step.id) {
            return Err(InternalErrors::ParseError(format!(
                "Module {}: two steps have the id {} - its parameters can't tell \
                 which one they set",
                module.key, step.id
            )));
        }
    }
    Ok(assign_step_ids(&mut module.steps))
}

/// `steps` without their ids - copies to add to another list (a template's
/// steps, a duplicated step), which [`assign_step_ids`] then numbers there.
pub fn unassigned(mut steps: Vec<PipelineStepSettings>) -> Vec<PipelineStepSettings> {
    for step in &mut steps {
        step.id = StepId::UNASSIGNED;
    }
    steps
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
    rgb_to_u32(wavelength_to_rgb_float(wavelength))
}

/// `[r, g, b]` (0..1) as `0xRRGGBB`.
pub fn rgb_to_u32(color: [f32; 3]) -> u32 {
    ((color[0] * 255.0) as u32) << 16 | ((color[1] * 255.0) as u32) << 8 | (color[2] * 255.0) as u32
}

/// Colours for channels without an emission wavelength, by position - the
/// order Fiji/ImageJ uses for composite images, so it looks familiar:
/// red, green, blue, gray, cyan, magenta, yellow.
pub const DEFAULT_CHANNEL_COLORS: [[f32; 3]; 7] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
];

/// The display colour of each channel of a series, from `(channel,
/// emission wavelength in nm)` (`<= 1.0`: unknown): a known wavelength's
/// colour; otherwise - for an image with several channels - the next
/// [`DEFAULT_CHANNEL_COLORS`] entry no channel shows yet, so no two
/// channels look alike; a single channel without one stays white.
pub fn channel_colors(channels: &[(i32, f32)]) -> std::collections::BTreeMap<i32, [f32; 3]> {
    let mut colors: std::collections::BTreeMap<i32, [f32; 3]> = channels
        .iter()
        .filter(|(_, nm)| *nm > 1.0)
        .map(|(id, nm)| (*id, wavelength_to_rgb_float(*nm)))
        .collect();
    let unknown = channels
        .iter()
        .filter(|(_, nm)| *nm <= 1.0)
        .map(|(id, _)| *id);
    if channels.len() == 1 {
        colors.extend(unknown.map(|id| (id, [1.0, 1.0, 1.0])));
        return colors;
    }
    let free: Vec<[f32; 3]> = DEFAULT_CHANNEL_COLORS
        .iter()
        .filter(|color| !colors.values().any(|used| used == *color))
        .copied()
        .collect();
    // All seven taken by wavelengths: repeat the palette.
    let palette = if free.is_empty() {
        DEFAULT_CHANNEL_COLORS.to_vec()
    } else {
        free
    };
    let assigned: Vec<(i32, [f32; 3])> = unknown.zip(palette.iter().copied().cycle()).collect();
    colors.extend(assigned);
    colors
}

/// Wavelengths [nm] given to channels without one when an image is opened
/// (the user setting "Color channels without a wavelength"), in order: red,
/// green, blue, cyan, yellow, orange - distinct, and stored in the project
/// like a wavelength the user set.
pub const ASSIGNED_WAVELENGTHS: [f32; 6] = [635.0, 520.0, 450.0, 490.0, 580.0, 605.0];

/// The display colour of each channel: its wavelength's, white without one
/// (see [`ASSIGNED_WAVELENGTHS`] for colours given when an image opens).
pub fn wavelength_colors(channels: &[(i32, f32)]) -> std::collections::BTreeMap<i32, [f32; 3]> {
    channels
        .iter()
        .map(|(id, nm)| {
            (
                *id,
                if *nm > 1.0 {
                    wavelength_to_rgb_float(*nm)
                } else {
                    [1.0, 1.0, 1.0]
                },
            )
        })
        .collect()
}

/// A channel's name for display: its own, or "Channel N" (N its index,
/// from 0 like the channels) when the image has none.
pub fn channel_display_name(name: &str, channel: i32) -> String {
    if name.trim().is_empty() {
        format!("Channel {channel}")
    } else {
        name.to_string()
    }
}

/// The name of a class made from an image channel ("Add classes from
/// image"): the channel's own name, or "Class N" (N the channel's index,
/// from 0) when the image has none.
pub fn class_name_of_channel(name: &str, channel: i32) -> String {
    if name.trim().is_empty() {
        format!("Class {channel}")
    } else {
        name.to_string()
    }
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

    /// In a project: the pipelines' measurement steps, then the modules'.
    #[test]
    fn a_projects_measurement_steps_without_an_id_get_one() {
        use evanalyzer_cfg::core_types::PipelineId;
        let pearson = |name: &str| {
            let mut command =
                PipelineCommand::default_for_key("measure_image_colocalization").unwrap();
            *command.measurement_id_mut().unwrap() = name.into();
            PipelineStepSettings {
                id: StepId::UNASSIGNED,
                enabled: true,
                content: StepContent::Command(command),
            }
        };
        let mut project = ProjectSettings::default();
        project.pipelines.push(PipelineSettings {
            id: PipelineId(1),
            name: String::new(),
            description: None,
            image_source: evanalyzer_cfg::core_types::ImageAddress::Channel(0),
            enabled: true,
            steps: vec![pearson(""), pearson("coloc_01")],
        });
        project.modules.push(ModuleDefinition {
            key: "COLOC".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![pearson("")],
            parameters: vec![],
            origin: None,
        });

        assert_eq!(assign_measurement_ids(&mut project), 2);
        let name = |step: &PipelineStepSettings| {
            step.command()
                .and_then(|c| c.measurement_id())
                .unwrap()
                .to_string()
        };
        let names: Vec<String> = project.pipelines[0]
            .steps
            .iter()
            .chain(&project.modules[0].steps)
            .map(name)
            .collect();
        assert_eq!(names, ["coloc_02", "coloc_01", "coloc_03"]);
    }

    /// Empty names get the next free number of their kind; names already
    /// given stay - also one used twice (the checks report it).
    #[test]
    fn empty_measurement_ids_get_the_next_free_number() {
        let mut texts = [
            "".to_string(),
            "Pearson_02".into(),
            "".into(),
            "own".into(),
            "own".into(),
            " ".into(),
        ];
        let [a, b, c, d, e, f] = &mut texts;
        let mut names: Vec<(&str, &mut String)> = vec![
            ("pearson", a),
            ("pearson", b),
            ("pearson", c),
            ("texture", d),
            ("texture", e),
            ("texture", f),
        ];
        assert_eq!(fill_measurement_ids(&mut names), 3);
        assert_eq!(
            texts,
            [
                "pearson_01",
                "Pearson_02",
                "pearson_03",
                "own",
                "own",
                "texture_01"
            ]
        );
    }

    const RED: [f32; 3] = DEFAULT_CHANNEL_COLORS[0];
    const GREEN: [f32; 3] = DEFAULT_CHANNEL_COLORS[1];
    const BLUE: [f32; 3] = DEFAULT_CHANNEL_COLORS[2];
    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

    #[test]
    fn channels_without_a_wavelength_get_the_fiji_colours_in_order() {
        let colors = channel_colors(&[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]);
        assert_eq!(
            colors.values().copied().collect::<Vec<_>>(),
            [RED, GREEN, BLUE, WHITE]
        );
    }

    #[test]
    fn a_single_channel_without_a_wavelength_is_white() {
        assert_eq!(channel_colors(&[(0, 0.0)])[&0], WHITE);
    }

    #[test]
    fn known_wavelengths_keep_their_colour_and_the_others_skip_it() {
        // 635 nm is pure red: the unknown channels start with green.
        let colors = channel_colors(&[(0, 0.0), (1, 635.0), (2, 0.0)]);
        assert_eq!(colors[&1], RED);
        assert_eq!(colors[&0], GREEN);
        assert_eq!(colors[&2], BLUE);
    }

    #[test]
    fn more_channels_than_colours_repeat_the_palette() {
        let channels: Vec<(i32, f32)> = (0..9).map(|c| (c, 0.0)).collect();
        let colors = channel_colors(&channels);
        assert_eq!(colors[&7], RED);
        assert_eq!(colors[&8], GREEN);
    }

    #[test]
    fn unnamed_channels_are_numbered_from_one() {
        assert_eq!(channel_display_name("", 0), "Channel 0");
        assert_eq!(channel_display_name("  ", 2), "Channel 2");
        assert_eq!(class_name_of_channel("", 0), "Class 0");
        assert_eq!(class_name_of_channel("  ", 2), "Class 2");
        assert_eq!(class_name_of_channel("DAPI", 0), "DAPI");
        assert_eq!(channel_display_name("DAPI", 0), "DAPI");
    }

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
            name: "".into(),
            image_source: ImageAddress::Channel(0),
            enabled: true,
            steps: vec![
                evanalyzer_cfg::settings::pipeline_settings::PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                    enabled: true,
                    content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                        command,
                    ),
                },
            ],
            description: None,
        }
    }

    fn pixel_classifier_model_path(pipeline: &PipelineSettings) -> &PathBuf {
        let PipelineCommand::PixelClassifier(settings) = pipeline.steps[0].command().unwrap()
        else {
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

        let PipelineCommand::Stardist(stardist) = pipelines[0].steps[0].command().unwrap() else {
            panic!("expected a Stardist command");
        };
        assert_eq!(stardist.model_path, PathBuf::from("models/stardist.pt"));

        let PipelineCommand::UNet(unet) = pipelines[1].steps[0].command().unwrap() else {
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
    fn model_paths_in_module_definitions_are_stored_relative_too() {
        use evanalyzer_cfg::settings::pipeline_command_settings::CellposeSettings;

        let original = PathBuf::from("/data/project/models/cyto.pt");
        let mut modules = vec![ModuleDefinition {
            key: "SEGMENT".into(),
            meta: Default::default(),
            classes: vec![],
            steps: pipeline_with(PipelineCommand::Cellpose(CellposeSettings {
                model_path: original.clone(),
                ..Default::default()
            }))
            .steps,
            parameters: vec![],
            origin: None,
        }];
        let project_dir = Path::new("/data/project");

        relativize_module_file_paths(&mut modules, project_dir);
        let Some(PipelineCommand::Cellpose(stored)) = modules[0].steps[0].command() else {
            panic!("expected Cellpose");
        };
        assert_eq!(stored.model_path, PathBuf::from("models/cyto.pt"));

        resolve_module_file_paths(&mut modules, project_dir);
        let Some(PipelineCommand::Cellpose(loaded)) = modules[0].steps[0].command() else {
            panic!("expected Cellpose");
        };
        assert_eq!(loaded.model_path, original);
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

        let PipelineCommand::Cellpose(settings) = pipelines[0].steps[0].command().unwrap() else {
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

    #[test]
    fn steps_without_an_id_or_with_a_taken_one_get_the_next_number() {
        use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
        let step = |id| PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::Blur(BlurSettings::default()),
            ),
        };
        // A hand-written list: two without an id, a copied block with 3.
        let mut steps = vec![step(3), step(0), step(3), step(0), step(1)];
        assert_eq!(super::assign_step_ids(&mut steps), 3);
        let ids: Vec<u32> = steps.iter().map(|s| s.id.0).collect();
        assert_eq!(ids, [3, 4, 5, 6, 1], "the first 3 keeps its id");

        assert_eq!(super::assign_step_ids(&mut steps), 0, "nothing left to do");
        let copies = super::unassigned(steps.clone());
        assert!(copies.iter().all(|s| s.id == StepId::UNASSIGNED));
    }

    #[test]
    fn a_modules_missing_step_ids_are_assigned_but_a_duplicate_is_an_error() {
        use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
        let step = |id| PipelineStepSettings {
            id: StepId(id),
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::Blur(BlurSettings::default()),
            ),
        };
        let mut module = ModuleDefinition {
            key: "CELL_RECONSTRUCTION".into(),
            meta: Default::default(),
            classes: vec![],
            steps: vec![step(2), step(0)],
            parameters: Vec::new(),
            origin: None,
        };
        assert_eq!(super::assign_module_step_ids(&mut module).unwrap(), 1);
        assert_eq!(module.steps[1].id, StepId(3));

        module.steps.push(step(2));
        let err = super::assign_module_step_ids(&mut module)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("CELL_RECONSTRUCTION") && err.contains("id 2"),
            "{err}"
        );
    }
}
