use crate::{
    DuckDbExporter, MemoryExporter,
    algos::TileMerge,
    image::PixelSizes,
    job::job_executor::{JobExecutor, TILE_MERGE_PIPELINE_ID},
    pipeline::pipeline::{CorePipelineSettings, Pipeline},
    storage::PipelineResultExporter,
};
use chrono::Utc;
use evanalyzer_cfg::{PROJECT_FILE_EXTENSIONS, RESULTS_FILE_EXTENSION, core_types::ImageAddress};
use evanalyzer_cfg::{
    core_types::InternalErrors,
    settings::{
        images_settings::ImageEntry, object_settings::ObjectMetricSettings,
        project_settings::ProjectSettings,
    },
};
use log::{error, info, warn};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Generate a job just for preview
///
/// No output data are written to disk, results are just stored in memory -
/// the returned `Arc<Mutex<Vec<ObjectMetricSettings>>>` is that in-memory
/// store, filled in by `MemoryExporter::export` once the job's whole-image
/// phase (including `TileMerge`, if enabled) finishes for each image. The
/// caller must read *this* after the job completes to get the final,
/// correctly-merged object set - the per-tile `ProgressEvent::TileCompleted`
/// events streamed during the run carry each tile's own objects *before*
/// tile-merge has run, so accumulating those alone leaves cross-tile
/// fragments unmerged.
pub fn generate_preview_job_from_project_settings(
    config: ProjectSettings,
    project_path: PathBuf,
) -> Result<(JobExecutor, Arc<Mutex<Vec<ObjectMetricSettings>>>), InternalErrors> {
    check_series(&config)?;
    check_pipeline_channels(&config)?;
    let out_objects: Arc<Mutex<Vec<ObjectMetricSettings>>> = Arc::new(Mutex::new(vec![]));
    let memory_storage = Arc::new(Mutex::new(MemoryExporter {
        out_objects: out_objects.clone(),
    }));

    let output_path = project_path.join("results").join("preview");
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        error!("Failed to create preview output directory: {e}");
        return Err(InternalErrors::Io(format!("{e}")));
    }

    let job = generate_job_from_project_settings_intenal(
        config,
        project_path,
        output_path,
        memory_storage,
    )?;
    Ok((job, out_objects))
}

/// Generates a job for a full analysis.
///
/// `job_name` is used verbatim (after sanitizing for filesystem-illegal
/// characters) as both the results-subdirectory suffix and the `.evadb`
/// filename. Pass `None` (or an empty/whitespace-only string) to fall back to
/// a randomly generated two-word name, as before this parameter existed.
///
/// Also writes a full copy of `config` (the exact settings this run used,
/// images included) next to the `.evadb` file as `<job_name>.evaproj`, so a
/// user can reopen it later to see - or restore - exactly what produced
/// these results.
pub fn generate_analyze_job_from_project_settings(
    mut config: ProjectSettings,
    project_path: PathBuf,
    job_name: Option<String>,
) -> Result<JobExecutor, InternalErrors> {
    // Before anything is created on disk: a refused start leaves no empty
    // results folder behind.
    check_series(&config)?;
    check_pipeline_channels(&config)?;
    let class_names: std::collections::HashMap<_, _> = config
        .classification
        .classes()
        .iter()
        .filter_map(|c| {
            c.id.to_u32().map(|n| {
                (
                    evanalyzer_cfg::core_types::ObjectClass::Valid(n),
                    (c.name.clone(), c.color),
                )
            })
        })
        .collect();

    let now = Utc::now();
    let file_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let job_name = job_name
        .as_deref()
        .map(sanitize_job_name)
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| petname::petname(2, "_").expect("Problem in random job name generator"));
    let output_path = project_path
        .join("results")
        .join(format!("{file_date}__{job_name}"));
    info!("Creating output directory: {:?}", output_path);
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        error!("Failed to create output directory: {e}");
        return Err(InternalErrors::Io(format!("{e}")));
    }

    let db_out_name = output_path.join(format!("{job_name}.{RESULTS_FILE_EXTENSION}"));
    let database_storage = match DuckDbExporter::new(&db_out_name, class_names) {
        Ok(exp) => Arc::new(Mutex::new(exp)),
        Err(e) => {
            error!(
                "Failed to open result database {}: {e}",
                db_out_name.display()
            );
            return Err(e);
        }
    };

    config.meta.app_version = env!("CARGO_PKG_VERSION").to_string();
    write_project_snapshot(&config, &output_path, &job_name);

    generate_job_from_project_settings_intenal(config, project_path, output_path, database_storage)
}

/// Refuses to start when an enabled pipeline reads an image channel some of
/// the images don't have - such a pipeline would silently produce nothing
/// for them (the image reader skips channels an image lacks). Checked against
/// the channels stored for each image in the project; images without that
/// information are left to the run itself.
pub fn check_pipeline_channels(config: &ProjectSettings) -> Result<(), InternalErrors> {
    const LISTED: usize = 5;
    let mut problems = Vec::new();
    for pipeline in config.pipelines.iter().filter(|p| p.enabled) {
        let ImageAddress::Channel(channel) = pipeline.image_source else {
            continue;
        };
        let missing: Vec<String> = config
            .images
            .list
            .values()
            .filter_map(|image| {
                let active = active_series(image, config.images.settings.selected_series);
                let series = image.series.get(&active)?;
                if series.channels.is_empty() || series.channels.contains_key(&channel) {
                    return None;
                }
                let available: Vec<String> =
                    series.channels.keys().map(|c| c.to_string()).collect();
                Some(format!(
                    "{} (channels {})",
                    image.rel_path.display(),
                    available.join(", ")
                ))
            })
            .collect();
        if missing.is_empty() {
            continue;
        }
        let mut listed = missing
            .iter()
            .take(LISTED)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if missing.len() > LISTED {
            listed.push_str(&format!(" and {} more", missing.len() - LISTED));
        }
        let message = format!(
            "pipeline '{}' reads channel {channel}, which {} image(s) don't have: {listed}",
            pipeline.name,
            missing.len()
        );
        warn!("{message}");
        problems.push(message);
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(InternalErrors::InvalidArgument(format!(
        "Cannot start: {}. Change the pipeline's image source, or remove those images.",
        problems.join("; ")
    )))
}

/// The series that counts for `image`: the project-wide choice
/// `project_series` (`images.settings.selected_series`) if there is one,
/// else the image's own, else its first series. The one place this is
/// decided - analysis, preview, training and the GUI all ask here.
pub fn active_series(image: &ImageEntry, project_series: Option<i32>) -> i32 {
    if let Some(series) = project_series {
        return series;
    }
    match image.series.keys().next() {
        Some(first) if !image.series.contains_key(&image.selected_series) => *first,
        _ => image.selected_series,
    }
}

/// Refuses to start when the project-wide series (`images.settings
/// .selected_series`) is one some images don't
/// have: they would fail one by one, or - worse - be measured on another
/// series. Images whose series haven't been read yet are left to the run.
pub fn check_series(config: &ProjectSettings) -> Result<(), InternalErrors> {
    const LISTED: usize = 5;
    let Some(series) = config.images.settings.selected_series else {
        return Ok(());
    };
    let missing: Vec<String> = config
        .images
        .list
        .values()
        .filter(|image| !image.series.is_empty() && !image.series.contains_key(&series))
        .map(|image| {
            let available: Vec<String> = image.series.keys().map(|s| (s + 1).to_string()).collect();
            format!(
                "{} (series {})",
                image.rel_path.display(),
                available.join(", ")
            )
        })
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let mut listed = missing
        .iter()
        .take(LISTED)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if missing.len() > LISTED {
        listed.push_str(&format!(" and {} more", missing.len() - LISTED));
    }
    let message = format!(
        "Cannot start: series {} is selected, which {} image(s) don't have: {listed}. \
         Select another series, or remove those images.",
        series + 1,
        missing.len()
    );
    warn!("{message}");
    Err(InternalErrors::InvalidArgument(message))
}

/// Writes a full copy of `config` as `<job_name>.evaproj` next to the run's
/// `.evadb` file. Best-effort: a snapshot failure is logged, not propagated -
/// it must never fail an otherwise-successful analysis run.
fn write_project_snapshot(config: &ProjectSettings, output_path: &Path, job_name: &str) {
    let snapshot_path = output_path.join(format!("{job_name}.{PROJECT_FILE_EXTENSIONS}"));
    let json = match serde_json::to_string_pretty(config) {
        Ok(json) => json,
        Err(e) => {
            warn!("Failed to serialize project snapshot: {e}");
            return;
        }
    };
    if let Err(e) = std::fs::write(&snapshot_path, json) {
        warn!(
            "Failed to write project snapshot {}: {e}",
            snapshot_path.display()
        );
    }
}

/// Replaces filesystem-illegal characters with `_` and trims whitespace, so a
/// user-provided job name is always safe to use as a directory/file name.
/// Returns an empty string if nothing usable remains (e.g. the input was
/// blank or made up entirely of illegal characters) - the caller treats that
/// the same as "no name given" and falls back to a random one.
fn sanitize_job_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.chars().all(|c| c == '_') {
        String::new()
    } else {
        cleaned
    }
}

fn generate_job_from_project_settings_intenal(
    config: ProjectSettings,
    project_path: PathBuf,
    output_path: PathBuf,
    result_storage: Arc<Mutex<dyn PipelineResultExporter>>,
) -> Result<JobExecutor, InternalErrors> {
    let Some(image_base_path) = config.images.root else {
        return Err(InternalErrors::InvalidArgument(
            "No image base path set".into(),
        ));
    };

    let pixel_sizes = match &config.images.settings.pixel_sizes {
        Some(data) => Some(PixelSizes {
            px_size_x: data.x,
            px_size_y: data.y,
            px_size_z: data.z,
        }),
        None => None,
    };

    let mut job = JobExecutor::new(
        project_path,
        output_path,
        config.images.list,
        image_base_path,
        config.images.settings,
        result_storage,
        pixel_sizes,
    );

    // If tile merged is enabled, add this command as first post process pipeline.
    // This command must run before any other object command
    if config.tile_merge.enabled {
        let mut tile_merge_pipeline = Pipeline::new(
            TILE_MERGE_PIPELINE_ID,
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );
        tile_merge_pipeline.add_command(Box::new(TileMerge {
            classes_to_not_merge: config.tile_merge.classes_to_not_merge.clone(),
            connectivity: config.tile_merge.connectivity.into(),
            max_fragments_per_group: config.tile_merge.max_fragments_per_group,
        }));
        job.add_post_process_pipeline(tile_merge_pipeline);
    }

    // Now generate the pipelines from the user settings
    for pipeline_setting in &config.pipelines {
        if !pipeline_setting.enabled {
            continue;
        }

        let mut pipeline_pre_process = Pipeline::new(
            pipeline_setting.id.clone(),
            CorePipelineSettings {
                start_image: pipeline_setting.image_source,
            },
        );

        let mut pipeline_post_process = Pipeline::new(
            pipeline_setting.id.clone(),
            CorePipelineSettings {
                start_image: ImageAddress::Scratchpad,
            },
        );

        for step in &pipeline_setting.steps {
            if step.enabled {
                let step = super::algos_from_config::into_algorithm(step.command.clone())?;
                match step.execution_scope() {
                    crate::algos::ExecutionScope::Tile => pipeline_pre_process.add_command(step),
                    crate::algos::ExecutionScope::WholeImage => {
                        pipeline_post_process.add_command(step)
                    }
                }
            }
        }

        job.add_pre_process_pipeline(pipeline_pre_process);
        job.add_post_process_pipeline(pipeline_post_process);
    }

    Ok(job)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::{ImageAddress, PipelineId};
    use evanalyzer_cfg::settings::images_settings::PixelSizeSettings;
    use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
    use evanalyzer_cfg::settings::pipeline_command_settings::BlurSettings;
    use evanalyzer_cfg::settings::pipeline_settings::{PipelineSettings, PipelineStepSettings};

    fn blur_step(enabled: bool) -> PipelineStepSettings {
        PipelineStepSettings {
            enabled,
            command: PipelineCommand::Blur(BlurSettings::default()),
        }
    }

    fn pipeline(id: u32, enabled: bool, steps: Vec<PipelineStepSettings>) -> PipelineSettings {
        PipelineSettings {
            id: PipelineId(id),
            name: "".into(),
            description: None,
            image_source: ImageAddress::Channel(0),
            enabled,
            steps,
        }
    }

    fn project_with(root: Option<PathBuf>, pipelines: Vec<PipelineSettings>) -> ProjectSettings {
        let mut project = ProjectSettings::default();
        project.images.root = root;
        project.pipelines = pipelines;
        project
    }

    // ---- pipelines reading channels the images don't have ----

    /// An image with the series `keys` (empty = not read yet).
    fn image_with_series(
        name: &str,
        keys: &[i32],
    ) -> evanalyzer_cfg::settings::images_settings::ImageEntry {
        evanalyzer_cfg::settings::images_settings::ImageEntry {
            rel_path: PathBuf::from(name),
            series: keys.iter().map(|k| (*k, Default::default())).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn the_series_comes_from_the_project_then_the_image_then_its_first_one() {
        let mut image = image_with_series("a.vsi", &[0, 1, 2]);
        image.selected_series = 1;
        assert_eq!(active_series(&image, None), 1, "the image's own");
        assert_eq!(active_series(&image, Some(2)), 2, "the project's wins");
        assert_eq!(
            active_series(&image, Some(7)),
            7,
            "even one the image lacks - check_series refuses that"
        );

        let mut stale = image_with_series("b.vsi", &[3, 4]);
        stale.selected_series = 0;
        assert_eq!(active_series(&stale, None), 3, "its first series");
        let mut unread = image_with_series("c.vsi", &[]);
        unread.selected_series = 5;
        assert_eq!(active_series(&unread, None), 5, "series not read yet");
    }

    #[test]
    fn project_files_without_a_project_wide_series_still_load() {
        let settings: evanalyzer_cfg::settings::images_settings::GlobalImageSettings =
            serde_json::from_str(r#"{"channels":{}}"#).unwrap();
        assert_eq!(settings.selected_series, None);
    }

    #[test]
    fn a_project_wide_series_some_images_lack_refuses_the_start() {
        let mut project = ProjectSettings::default();
        for (name, keys) in [
            ("a.vsi", &[0, 1][..]),
            ("b.vsi", &[0][..]),
            ("c.vsi", &[][..]),
        ] {
            project
                .images
                .list
                .insert(PathBuf::from(name), image_with_series(name, keys));
        }
        assert!(check_series(&project).is_ok(), "no project-wide series");

        project.images.settings.selected_series = Some(0);
        assert!(check_series(&project).is_ok(), "all have series 1");

        project.images.settings.selected_series = Some(1);
        let error = check_series(&project).unwrap_err().to_string();
        assert!(error.contains("series 2 is selected"), "{error}");
        assert!(error.contains("b.vsi (series 1)"), "{error}");
        assert!(!error.contains("a.vsi"), "{error}");
        assert!(!error.contains("c.vsi"), "not read yet: {error}");
    }

    /// An image whose selected series has the given channels (empty = no
    /// channel information stored).
    fn image(
        name: &str,
        channels: &[i32],
    ) -> evanalyzer_cfg::settings::images_settings::ImageEntry {
        use evanalyzer_cfg::settings::images_settings::{
            ChannelSettings, ImageEntry, SeriesSettings,
        };
        let series = SeriesSettings {
            channels: channels
                .iter()
                .map(|&c| {
                    (
                        c,
                        ChannelSettings {
                            name: format!("C{c}"),
                            emission_wave_length: None,
                            visible: None,
                            histogram: None,
                        },
                    )
                })
                .collect(),
            ..Default::default()
        };
        ImageEntry {
            rel_path: PathBuf::from(name),
            file_size: 0,
            selected_series: 0,
            series: std::collections::BTreeMap::from([(0, series)]),
        }
    }

    fn named(id: u32, name: &str, source: ImageAddress, enabled: bool) -> PipelineSettings {
        PipelineSettings {
            name: name.into(),
            image_source: source,
            ..pipeline(id, enabled, vec![blur_step(true)])
        }
    }

    fn project_with_images(
        pipelines: Vec<PipelineSettings>,
        images: Vec<evanalyzer_cfg::settings::images_settings::ImageEntry>,
    ) -> ProjectSettings {
        let mut project = project_with(None, pipelines);
        for img in images {
            project.images.list.insert(img.rel_path.clone(), img);
        }
        project
    }

    #[test]
    fn a_pipeline_reading_a_missing_channel_is_refused_with_a_clear_message() {
        let project = project_with_images(
            vec![named(1, "Spots", ImageAddress::Channel(2), true)],
            vec![image("a.tif", &[0, 1, 2]), image("b.tif", &[0, 1])],
        );
        let err = check_pipeline_channels(&project).unwrap_err();
        let InternalErrors::InvalidArgument(message) = err else {
            panic!("expected InvalidArgument");
        };
        assert!(
            message.contains("pipeline 'Spots' reads channel 2"),
            "{message}"
        );
        assert!(message.contains("b.tif (channels 0, 1)"), "{message}");
        assert!(!message.contains("a.tif"), "a.tif has channel 2: {message}");
    }

    #[test]
    fn only_enabled_channel_pipelines_against_known_channels_are_checked() {
        use evanalyzer_cfg::core_types::MemoryId;
        let project = project_with_images(
            vec![
                named(1, "Off", ImageAddress::Channel(5), false),
                named(2, "Scratch", ImageAddress::Scratchpad, true),
                named(
                    3,
                    "Memory",
                    ImageAddress::Memory(MemoryId::PipelineContext(1)),
                    true,
                ),
                named(4, "Fine", ImageAddress::Channel(0), true),
            ],
            vec![image("a.tif", &[0]), image("unknown.tif", &[])],
        );
        assert!(check_pipeline_channels(&project).is_ok());
    }

    #[test]
    fn many_affected_images_are_summarized() {
        let images = (0..8)
            .map(|i| image(&format!("img{i}.tif"), &[0]))
            .collect();
        let project = project_with_images(
            vec![named(1, "Red", ImageAddress::Channel(1), true)],
            images,
        );
        let InternalErrors::InvalidArgument(message) =
            check_pipeline_channels(&project).unwrap_err()
        else {
            panic!("expected InvalidArgument");
        };
        assert!(message.contains("8 image(s)"), "{message}");
        assert!(message.contains("and 3 more"), "{message}");
    }

    #[test]
    fn a_refused_analysis_leaves_no_results_folder_behind() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with_images(
            vec![named(1, "Spots", ImageAddress::Channel(3), true)],
            vec![image("a.tif", &[0])],
        );
        project.images.root = Some(image_root.path().to_path_buf());

        assert!(
            generate_analyze_job_from_project_settings(
                project.clone(),
                project_dir.path().to_path_buf(),
                None
            )
            .is_err()
        );
        assert!(!project_dir.path().join("results").exists());
        assert!(
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .is_err(),
            "the preview is refused as well"
        );
    }

    #[test]
    fn missing_image_root_is_rejected_with_invalid_argument() {
        let dir = tempfile::tempdir().unwrap();
        let project = project_with(None, vec![]);

        let err = generate_preview_job_from_project_settings(project, dir.path().to_path_buf())
            .err()
            .expect("no image root must be an error");

        assert!(matches!(err, InternalErrors::InvalidArgument(_)));
    }

    #[test]
    fn preview_job_creates_the_results_preview_directory_and_forwards_paths() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(Some(image_root.path().to_path_buf()), vec![]);

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .expect("valid config with an image root must succeed");

        let expected_output = project_dir.path().join("results").join("preview");
        assert!(
            expected_output.is_dir(),
            "preview output directory must be created on disk"
        );
        assert_eq!(job.output_path, expected_output);
        assert_eq!(job.project_path, project_dir.path());
        assert_eq!(job.image_base_path, image_root.path());
        assert!(job.pipelines_pre_process.is_empty());
    }

    #[test]
    fn disabled_pipelines_are_skipped_entirely() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![
                pipeline(1, false, vec![blur_step(true)]),
                pipeline(2, true, vec![blur_step(true)]),
            ],
        );

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        assert_eq!(
            job.pipelines_pre_process.len(),
            1,
            "only the enabled pipeline must be added"
        );
        assert!(job.pipelines_pre_process.contains_key(&PipelineId(2)));
        assert!(!job.pipelines_pre_process.contains_key(&PipelineId(1)));
    }

    #[test]
    fn disabled_steps_within_an_enabled_pipeline_are_skipped() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(
                1,
                true,
                vec![blur_step(true), blur_step(false), blur_step(true)],
            )],
        );

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        let built = job
            .pipelines_pre_process
            .get(&PipelineId(1))
            .expect("pipeline 1 must exist");
        assert_eq!(
            built.commands.len(),
            2,
            "only the two enabled steps must become commands"
        );
    }

    #[test]
    fn pixel_size_override_is_forwarded_when_set() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with(Some(image_root.path().to_path_buf()), vec![]);
        project.images.settings.pixel_sizes = Some(PixelSizeSettings {
            x: 1.5,
            y: 2.5,
            z: 3.5,
        });

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        let sizes = job
            .override_pixel_sizes
            .expect("pixel size override must be forwarded");
        assert_eq!(sizes.px_size_x, 1.5);
        assert_eq!(sizes.px_size_y, 2.5);
        assert_eq!(sizes.px_size_z, 3.5);
    }

    #[test]
    fn pixel_size_override_is_none_when_unset() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(Some(image_root.path().to_path_buf()), vec![]);

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        assert!(job.override_pixel_sizes.is_none());
    }

    #[test]
    fn analyze_job_creates_a_timestamped_results_database() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![blur_step(true)])],
        );

        let job = generate_analyze_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
            None,
        )
        .expect("valid config with an image root must succeed");

        assert_eq!(job.pipelines_pre_process.len(), 1);
        assert!(
            job.output_path
                .starts_with(project_dir.path().join("results")),
            "the analyze job's output directory must live under <project>/results"
        );
        assert!(job.output_path.is_dir());

        let db_files: Vec<_> = std::fs::read_dir(&job.output_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension().and_then(|s| s.to_str())
                    == Some(evanalyzer_cfg::RESULTS_FILE_EXTENSION)
            })
            .collect();
        assert_eq!(db_files.len(), 1, "exactly one .evadb file must be created");
    }

    #[test]
    fn analyze_job_writes_a_project_snapshot_beside_the_database_stamped_with_the_app_version() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![blur_step(true)])],
        );

        let job = generate_analyze_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
            Some("snapshot_run".to_string()),
        )
        .unwrap();

        let snapshot_path = job.output_path.join("snapshot_run.evaproj");
        let content = std::fs::read_to_string(&snapshot_path)
            .unwrap_or_else(|e| panic!("expected a snapshot at {snapshot_path:?}: {e}"));
        let restored: ProjectSettings = serde_json::from_str(&content)
            .expect("the snapshot must deserialize back into ProjectSettings");

        assert_eq!(restored.meta.app_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(
            restored.pipelines.len(),
            1,
            "the snapshot must carry the pipelines this run actually used"
        );
    }

    #[test]
    fn analyze_job_uses_the_provided_job_name_for_the_output_folder_and_database() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(Some(image_root.path().to_path_buf()), vec![]);

        let job = generate_analyze_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
            Some("dose_response_1".to_string()),
        )
        .unwrap();

        let folder_name = job
            .output_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert!(
            folder_name.ends_with("__dose_response_1"),
            "the timestamp prefix must still be kept alongside the custom name, got {folder_name}"
        );
        assert!(job.output_path.join("dose_response_1.evadb").is_file());
    }

    #[test]
    fn analyze_job_sanitizes_a_job_name_with_path_separators() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(Some(image_root.path().to_path_buf()), vec![]);

        let job = generate_analyze_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
            Some("../../etc/passwd".to_string()),
        )
        .unwrap();

        // Must stay a single path segment directly under <project>/results,
        // never escape it via the sanitized name.
        assert_eq!(
            job.output_path.parent().unwrap(),
            project_dir.path().join("results")
        );
    }

    #[test]
    fn analyze_job_falls_back_to_a_random_name_for_blank_or_illegal_only_input() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();

        for blank in ["", "   ", "///"] {
            let project = project_with(Some(image_root.path().to_path_buf()), vec![]);
            let job = generate_analyze_job_from_project_settings(
                project,
                project_dir.path().to_path_buf(),
                Some(blank.to_string()),
            )
            .unwrap();
            let folder_name = job
                .output_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            assert!(
                !folder_name.ends_with("__"),
                "blank/illegal-only input {blank:?} must fall back to a random name, got {folder_name}"
            );
        }
    }

    #[test]
    fn sanitize_job_name_replaces_illegal_characters_and_trims() {
        assert_eq!(sanitize_job_name("well A1/rep 2"), "well A1_rep 2");
        assert_eq!(sanitize_job_name("  padded  "), "padded");
        assert_eq!(sanitize_job_name("///"), "");
        assert_eq!(sanitize_job_name(""), "");
    }
}
