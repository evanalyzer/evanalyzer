use crate::{
    DuckDbExporter, MemoryExporter,
    algos::{EndScope, ExecutionScope, TileMerge},
    dataset::DatasetAlgorithm,
    image::PixelSizes,
    job::{
        algos_from_config::into_pipeline_step,
        job_executor::{JobExecutor, TILE_MERGE_PIPELINE_ID},
        modules::{ExpandedCommand, ScopeCounter, UseId, expand_step},
        pipeline_step::PipelineStep,
    },
    pipeline::pipeline::{CorePipelineSettings, Pipeline},
    pipeline::scope::Scope,
    storage::PipelineResultExporter,
};
use chrono::Utc;
use evanalyzer_cfg::{
    PROJECT_FILE_EXTENSIONS, RESULTS_FILE_EXTENSION,
    core_types::{ImageAddress, is_valid_measurement_name},
};
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass, StepId},
    settings::{
        images_settings::ImageEntry,
        modules::ModuleDefinition,
        object_settings::ObjectMetricSettings,
        pipeline_command::{CommandCategory, PipelineCommand},
        pipeline_command_settings::MathImageCacheImageCacheModeSettings,
        pipeline_settings::{PipelineSettings, PipelineStepSettings},
        project_settings::ProjectSettings,
    },
};
use log::{error, info, warn};
use std::{
    collections::{BTreeMap, BTreeSet},
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
) -> Result<(JobExecutor, PreviewOutput), InternalErrors> {
    check_series(&config)?;
    check_pipeline_channels(&config)?;
    let output = PreviewOutput::default();
    let memory_storage = Arc::new(MemoryExporter {
        out_objects: output.objects.clone(),
        out_measurements: output.measurements.clone(),
    });

    let output_path = project_path.join("results").join("preview");
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        error!("Failed to create preview output directory: {e}");
        return Err(InternalErrors::Io(format!("{e}")));
    }

    let mut job = generate_job_from_project_settings_intenal(
        config,
        project_path,
        output_path,
        memory_storage,
    )?;
    // A preview analyses one frame of the visible tiles - too little for a
    // step over all images and frames, which would only show a misleading
    // result (e.g. a tracker with nothing to link).
    job.dataset_steps.clear();
    Ok((job, output))
}

/// What a preview job leaves in memory (see `MemoryExporter`): its final
/// objects and its measurements per plane - read once the job finished.
#[derive(Clone, Default)]
pub struct PreviewOutput {
    pub objects: Arc<Mutex<Vec<ObjectMetricSettings>>>,
    pub measurements: Arc<Mutex<Vec<evanalyzer_cfg::core_types::PipelineMeasurements>>>,
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
        .map(|c| (c.global(), (c.name.clone(), c.color)))
        .collect();

    let (output_path, job_name) = create_results_dir(&project_path, job_name)?;

    config.meta.app_version = env!("CARGO_PKG_VERSION").to_string();

    let db_out_name = output_path.join(format!("{job_name}.{RESULTS_FILE_EXTENSION}"));
    let database_storage = match DuckDbExporter::new(&db_out_name, class_names, &config.meta) {
        Ok(exp) => Arc::new(exp),
        Err(e) => {
            error!(
                "Failed to open result database {}: {e}",
                db_out_name.display()
            );
            return Err(e);
        }
    };

    write_project_snapshot(&config, &output_path, &job_name);

    generate_job_from_project_settings_intenal(config, project_path, output_path, database_storage)
}

/// Generates a job that runs only the dataset steps, on the objects of an
/// earlier analysis - the image phase is skipped.
///
/// The earlier analysis is never changed: its results folder (the one
/// holding `base_results_db`, which must lie in the project's `results`
/// folder) is copied to a new one, named like a full analysis's
/// (`job_name`, or a random name if `None`), and the dataset steps write
/// into that copy. The copy's `.evaproj` is this run's settings as usual;
/// the earlier analysis's is kept beside it as `<job_name>.base.evaproj`.
pub fn generate_dataset_job_from_project_settings(
    mut config: ProjectSettings,
    project_path: PathBuf,
    job_name: Option<String>,
    base_results_db: &Path,
) -> Result<JobExecutor, InternalErrors> {
    // Before anything is created on disk.
    if dataset_steps_from_project_settings(&config)?.is_empty() {
        return Err(InternalErrors::InvalidArgument(
            "The project has no enabled dataset steps to run".into(),
        ));
    }
    let base_dir = checked_base_results_dir(&project_path, base_results_db)?;
    let base_name = base_results_db
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| InternalErrors::InvalidArgument("Invalid results database name".into()))?
        .to_string();

    let (output_path, job_name) = create_results_dir(&project_path, job_name)?;
    let job = (|| {
        copy_results_folder(&base_dir, &base_name, &output_path, &job_name)?;
        config.meta.app_version = env!("CARGO_PKG_VERSION").to_string();
        let db = output_path.join(format!("{job_name}.{RESULTS_FILE_EXTENSION}"));
        let storage = Arc::new(DuckDbExporter::continue_run(&db, &config.meta)?);
        write_project_snapshot(&config, &output_path, &job_name);
        let mut job = generate_job_from_project_settings_intenal(
            config,
            project_path,
            output_path.clone(),
            storage,
        )?;
        job.skip_image_phase = true;
        Ok(job)
    })();
    if job.is_err() {
        // A failed start leaves no half-copied results behind.
        let _ = std::fs::remove_dir_all(&output_path);
    }
    job
}

/// Creates `<project_path>/results/<date>__<job_name>` for a new run; a
/// blank `job_name` gets a random one. Returns the folder and the name.
fn create_results_dir(
    project_path: &Path,
    job_name: Option<String>,
) -> Result<(PathBuf, String), InternalErrors> {
    let file_date = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
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
    Ok((output_path, job_name))
}

/// The folder of `base_results_db` - refused unless it is an existing
/// results database inside the project's `results` folder (on a server, the
/// path comes from a client).
fn checked_base_results_dir(
    project_path: &Path,
    base_results_db: &Path,
) -> Result<PathBuf, InternalErrors> {
    let refuse = |why: &str| {
        InternalErrors::InvalidArgument(format!(
            "Can't use {} as the basis: {why}",
            base_results_db.display()
        ))
    };
    if base_results_db.extension().and_then(|e| e.to_str()) != Some(RESULTS_FILE_EXTENSION) {
        return Err(refuse("not a results database"));
    }
    let db = base_results_db
        .canonicalize()
        .map_err(|_| refuse("it doesn't exist"))?;
    let results = project_path
        .join("results")
        .canonicalize()
        .map_err(|_| refuse("the project has no results"))?;
    match db.parent() {
        Some(dir) if dir.starts_with(&results) && dir != results => Ok(dir.to_path_buf()),
        _ => Err(refuse("it isn't one of this project's results")),
    }
}

/// Copies the files of the results folder `from` into `to`, renaming the
/// run's own files from `from_name` to `to_name` (`<name>.evadb`,
/// `<name>.evadb.wal`, ...). The earlier run's settings snapshot becomes
/// `<to_name>.base.<ext>`.
fn copy_results_folder(
    from: &Path,
    from_name: &str,
    to: &Path,
    to_name: &str,
) -> Result<(), InternalErrors> {
    let io = |e: std::io::Error| InternalErrors::Io(format!("Copying the results failed: {e}"));
    for entry in std::fs::read_dir(from).map_err(io)? {
        let entry = entry.map_err(io)?;
        if !entry.file_type().map_err(io)?.is_file() {
            continue;
        }
        let file_name = entry.file_name().to_string_lossy().to_string();
        let target = match file_name.strip_prefix(&format!("{from_name}.")) {
            Some(PROJECT_FILE_EXTENSIONS) => format!("{to_name}.base.{PROJECT_FILE_EXTENSIONS}"),
            Some(rest) => format!("{to_name}.{rest}"),
            None => file_name,
        };
        std::fs::copy(entry.path(), to.join(target)).map_err(io)?;
    }
    Ok(())
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
        for channel in pipeline_channels(pipeline, &config.modules) {
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
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(InternalErrors::InvalidArgument(format!(
        "Cannot start: {}. Change the pipeline's image source or load steps, or remove those \
         images.",
        problems.join("; ")
    )))
}

/// The image channels `pipeline` reads: its image source's and those its
/// steps load (Load Image Channel, Image Cache Load, Image Math's second
/// image) - also inside modules, with the instances' values. A module
/// that can't be expanded is skipped here; building the job reports it.
pub fn pipeline_channels(
    pipeline: &PipelineSettings,
    modules: &[ModuleDefinition],
) -> BTreeSet<i32> {
    let mut channels = BTreeSet::new();
    if let ImageAddress::Channel(channel) = pipeline.image_source {
        channels.insert(channel);
    }
    for step in pipeline.steps.iter().filter(|s| s.enabled) {
        let Ok(commands) = expand_step(step, modules, &mut ScopeCounter::default()) else {
            continue;
        };
        for ExpandedCommand { command, .. } in commands {
            channels.extend(loaded_channel(&command));
        }
    }
    channels
}

/// The project classes `pipeline`'s enabled steps use - also inside command
/// modules, with the instances' values (a module's own classes aren't project
/// classes and are left out). A module that can't be expanded is skipped.
pub fn pipeline_classes(
    pipeline: &PipelineSettings,
    modules: &[ModuleDefinition],
) -> BTreeSet<ObjectClass> {
    pipeline
        .steps
        .iter()
        .filter(|s| s.enabled)
        .filter_map(|step| expand_step(step, modules, &mut ScopeCounter::default()).ok())
        .flatten()
        .flat_map(|ExpandedCommand { command, .. }| command.object_classes())
        .collect()
}

/// The image channel `command` reads, if it reads one.
fn loaded_channel(command: &PipelineCommand) -> Option<i32> {
    match command {
        PipelineCommand::LoadChannel(s) => Some(s.channel.0 as i32),
        PipelineCommand::ImageCache(s)
            if matches!(s.mode, MathImageCacheImageCacheModeSettings::Load) =>
        {
            match s.address {
                ImageAddress::Channel(channel) => Some(channel),
                _ => None,
            }
        }
        PipelineCommand::ImageMath(s) => match s.second_image_address {
            ImageAddress::Channel(channel) => Some(channel),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `command` replaces the current image with a loaded one - what a
/// pipeline without an image source must start with.
fn loads_image(command: &PipelineCommand) -> bool {
    match command {
        PipelineCommand::LoadChannel(_) => true,
        PipelineCommand::ImageCache(s) => {
            matches!(s.mode, MathImageCacheImageCacheModeSettings::Load)
                && s.address != ImageAddress::None
        }
        _ => false,
    }
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
    result_storage: Arc<dyn PipelineResultExporter>,
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

    let class_colors = config
        .classification
        .classes()
        .iter()
        .map(|c| (c.global(), c.color))
        .collect();

    let mut job = JobExecutor::new(
        project_path,
        output_path,
        config.images.list,
        image_base_path,
        config.images.settings,
        result_storage,
        pixel_sizes,
        Arc::new(class_colors),
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
    let mut scopes = ScopeCounter::default();
    // The measurement steps' names, with their pipeline's: unique in the run.
    let mut measurement_names: Vec<(String, &str)> = Vec::new();
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

        // A pipeline without an image source must load one first.
        let mut needs_image = pipeline_setting.image_source == ImageAddress::None;
        // Each module use's last command, per phase (index in that list).
        let mut last_tile: BTreeMap<UseId, usize> = BTreeMap::new();
        let mut last_whole_image: BTreeMap<UseId, usize> = BTreeMap::new();
        for step in pipeline_setting.steps.iter().filter(|s| s.enabled) {
            // A module runs as its commands.
            for ExpandedCommand {
                step_id,
                scope,
                uses,
                command,
            } in step_commands(&pipeline_setting.name, step, &config.modules, &mut scopes)?
            {
                if let Some(name) = command.measurement_name() {
                    measurement_names.push((name.to_string(), pipeline_setting.name.as_str()));
                }
                let loads = loads_image(&command);
                match into_pipeline_step(command)? {
                    // Kept with its id: what a breakpoint targets.
                    PipelineStep::Image(algo) => match algo.execution_scope() {
                        ExecutionScope::Tile => {
                            if needs_image && !loads {
                                return Err(InternalErrors::InvalidArgument(format!(
                                    "Pipeline \"{}\" has no image source, so its first step \
                                     must load one (Load Image Channel) - \"{}\" needs an image",
                                    pipeline_setting.name,
                                    algo.name()
                                )));
                            }
                            needs_image = false;
                            pipeline_pre_process.add_step(step_id, scope, algo);
                            let index = pipeline_pre_process.commands.len() - 1;
                            last_tile.extend(uses.iter().map(|id| (*id, index)));
                        }
                        ExecutionScope::WholeImage => {
                            pipeline_post_process.add_step(step_id, scope, algo);
                            let index = pipeline_post_process.commands.len() - 1;
                            last_whole_image.extend(uses.iter().map(|id| (*id, index)));
                        }
                    },
                    // Runs once after every image, whatever its place in the pipeline.
                    PipelineStep::Dataset(step) => job.add_dataset_step(step),
                }
            }
        }

        end_uses(
            &mut pipeline_pre_process,
            &mut pipeline_post_process,
            &last_tile,
            &last_whole_image,
            &scopes,
        );
        job.add_pre_process_pipeline(pipeline_pre_process);
        job.add_post_process_pipeline(pipeline_post_process);
    }
    check_measurement_names(&measurement_names)?;

    Ok(job)
}

/// Refuses a run with a measurement name that isn't valid or is used twice
/// (`names`: each with its pipeline's name) - a name starts its
/// measurements' keys. The editor reports the same while editing.
fn check_measurement_names(names: &[(String, &str)]) -> Result<(), InternalErrors> {
    let mut seen = BTreeSet::new();
    for (name, pipeline) in names {
        if !is_valid_measurement_name(name) {
            return Err(InternalErrors::InvalidArgument(format!(
                "Pipeline \"{pipeline}\": the measurement name \"{name}\" may only have \
                 letters, digits and _"
            )));
        }
        if !seen.insert(name.to_lowercase()) {
            return Err(InternalErrors::InvalidArgument(format!(
                "Pipeline \"{pipeline}\": the measurement name \"{name}\" is used more than \
                 once - each measurement step needs its own (a module used twice: make the \
                 name its parameter)"
            )));
        }
    }
    Ok(())
}

/// Adds an [`EndScope`] right after each module use's last command
/// across both phases: in the whole-image list if the use has a command
/// there (it runs after every tile command), else in the tile list - where
/// it frees the use's objects tile by tile, before they're ever merged into
/// the whole image. A use nested in another ends before the one around it.
fn end_uses(
    tile: &mut Pipeline,
    whole_image: &mut Pipeline,
    last_tile: &BTreeMap<UseId, usize>,
    last_whole_image: &BTreeMap<UseId, usize>,
    scopes: &ScopeCounter,
) {
    let in_tiles: BTreeMap<UseId, usize> = last_tile
        .iter()
        .filter(|(id, _)| !last_whole_image.contains_key(id))
        .map(|(id, index)| (*id, *index))
        .collect();
    for (pipeline, ends, phase) in [
        (tile, &in_tiles, ExecutionScope::Tile),
        (whole_image, last_whole_image, ExecutionScope::WholeImage),
    ] {
        let mut ends: Vec<(UseId, usize)> = ends.iter().map(|(id, i)| (*id, *i)).collect();
        // Back to front, so the earlier positions stay valid; at one position
        // the outer use (numbered first) first - the nested one then goes in
        // before it.
        ends.sort_by_key(|(id, index)| (std::cmp::Reverse(*index), *id));
        for (id, index) in ends {
            let Some(end) = scopes.end_of(id) else {
                continue;
            };
            pipeline.commands.insert(
                index + 1,
                (
                    StepId::UNASSIGNED,
                    Scope::Global,
                    Box::new(EndScope {
                        end: end.clone(),
                        phase,
                    }),
                ),
            );
        }
    }
}

/// The commands `step` runs, with their breakpoint ids and scopes (see
/// [`expand_step`]) - a module's error says which pipeline and step it is.
fn step_commands(
    pipeline_name: &str,
    step: &PipelineStepSettings,
    definitions: &[ModuleDefinition],
    scopes: &mut ScopeCounter,
) -> Result<Vec<ExpandedCommand>, InternalErrors> {
    expand_step(step, definitions, scopes).map_err(|e| {
        InternalErrors::InvalidArgument(format!(
            "Pipeline \"{pipeline_name}\", step {}: {e}",
            step.id
        ))
    })
}

/// The enabled dataset steps of `config`'s enabled pipelines, in pipeline
/// and step order. Image steps aren't built, so an image step this build
/// can't run (e.g. an AI step in a build without the `libtorch` feature) doesn't matter.
pub(crate) fn dataset_steps_from_project_settings(
    config: &ProjectSettings,
) -> Result<Vec<Box<dyn DatasetAlgorithm>>, InternalErrors> {
    let mut steps = Vec::new();
    // Dataset steps have no scope: they only use project classes.
    let mut scopes = ScopeCounter::default();
    for pipeline in config.pipelines.iter().filter(|p| p.enabled) {
        for step in pipeline.steps.iter().filter(|s| s.enabled) {
            for ExpandedCommand { command, .. } in
                step_commands(&pipeline.name, step, &config.modules, &mut scopes)?
            {
                if !matches!(command.category(), CommandCategory::Dataset) {
                    continue;
                }
                if let PipelineStep::Dataset(step) = into_pipeline_step(command)? {
                    steps.push(step);
                }
            }
        }
    }
    Ok(steps)
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
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::Blur(BlurSettings::default()),
            ),
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

    fn load_channel_step(channel: u32) -> PipelineStepSettings {
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::LoadChannel(
                    evanalyzer_cfg::settings::pipeline_command_settings::LoadChannelSettings {
                        channel: evanalyzer_cfg::core_types::ImageChannelIdx(channel),
                        projection: evanalyzer_cfg::settings::pipeline_command_settings::MathLoadChannelChannelZProjectionSettings::Auto,
                        z_index: 0,
                    },
                ),
            ),
        }
    }

    #[test]
    fn a_pipelines_classes_are_its_enabled_steps_and_its_modules_project_classes() {
        use evanalyzer_cfg::core_types::ObjectClass;
        use evanalyzer_cfg::settings::pipeline_command_settings::TransformObjectsSettings;
        let transform = |input: ObjectClass, output: u32, enabled: bool| PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::TransformObjects(TransformObjectsSettings {
                    input_class: input,
                    output_class: ObjectClass::Global(output),
                    ..Default::default()
                }),
            ),
        };
        let mut pipeline = named(1, "Cells", ImageAddress::Channel(0), true);
        pipeline.steps = vec![
            transform(ObjectClass::Global(1), 10, true),
            transform(ObjectClass::Global(3), 11, false),
            module_step("INNER"),
        ];
        let mut inner = module("INNER", vec![transform(ObjectClass::Local(1), 12, true)]);
        inner.steps[0].id = evanalyzer_cfg::core_types::StepId(1);
        inner.classes = vec![evanalyzer_cfg::settings::classification_settings::Class {
            id: 1,
            ..Default::default()
        }];
        assert_eq!(
            pipeline_classes(&pipeline, &[inner]),
            BTreeSet::from([1, 10, 12].map(ObjectClass::Global)),
            "disabled steps and the module's own class left out"
        );
    }

    #[test]
    fn the_channels_a_pipeline_reads_include_its_load_steps_also_in_modules() {
        let mut pipeline = named(1, "Cells", ImageAddress::Channel(0), true);
        pipeline.steps = vec![load_channel_step(3), module_step("LOADS")];
        let modules = [module("LOADS", vec![load_channel_step(5)])];
        assert_eq!(
            pipeline_channels(&pipeline, &modules),
            BTreeSet::from([0, 3, 5])
        );
    }

    #[test]
    fn a_load_step_in_a_module_reading_a_missing_channel_is_refused() {
        let mut pipeline = named(1, "Cells", ImageAddress::None, true);
        pipeline.steps = vec![module_step("LOADS")];
        let mut project = project_with_images(
            vec![pipeline],
            vec![image("a.tif", &[0, 1]), image("b.tif", &[0, 1, 2])],
        );
        project
            .modules
            .push(module("LOADS", vec![load_channel_step(2)]));
        let message = check_pipeline_channels(&project).unwrap_err().to_string();
        assert!(
            message.contains("pipeline 'Cells' reads channel 2"),
            "{message}"
        );
        assert!(message.contains("a.tif (channels 0, 1)"), "{message}");
    }

    #[test]
    fn a_pipeline_without_an_image_source_must_load_one_first() {
        let image_root = tempfile::tempdir().unwrap();
        let project_dir = tempfile::tempdir().unwrap();
        let mut pipeline = named(1, "Cells", ImageAddress::None, true);
        pipeline.steps = vec![blur_step(true), load_channel_step(0)];
        let project = project_with(Some(image_root.path().to_path_buf()), vec![pipeline]);
        let Err(error) =
            generate_preview_job_from_project_settings(project.clone(), project_dir.path().into())
        else {
            panic!("a blur without an image must be refused");
        };
        let error = error.to_string();
        assert!(
            error.contains("has no image source") && error.contains("\"Blur\" needs an image"),
            "{error}"
        );

        let mut fine = project;
        fine.pipelines[0].steps = vec![load_channel_step(0), blur_step(true)];
        assert!(
            generate_preview_job_from_project_settings(fine, project_dir.path().into()).is_ok()
        );
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

    // ---- dataset steps ----

    fn tracker_step(enabled: bool) -> PipelineStepSettings {
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
            enabled,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(PipelineCommand::SimpleTracker(
                evanalyzer_cfg::settings::pipeline_command_settings::SimpleTrackerSettings::default(
                ),
            )),
        }
    }

    fn project_with_dataset_steps(root: PathBuf) -> ProjectSettings {
        project_with(
            Some(root),
            vec![
                pipeline(
                    1,
                    true,
                    vec![blur_step(true), tracker_step(true), tracker_step(false)],
                ),
                pipeline(2, false, vec![tracker_step(true)]),
            ],
        )
    }

    fn module_step(key: &str) -> PipelineStepSettings {
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(4),
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Module(
                evanalyzer_cfg::settings::modules::ModuleInstance {
                    key: key.into(),
                    values: Default::default(),
                },
            ),
        }
    }

    #[test]
    fn a_module_runs_as_its_commands() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![module_step("TWO_BLURS")])],
        );
        project
            .modules
            .push(evanalyzer_cfg::settings::modules::ModuleDefinition {
                key: "TWO_BLURS".into(),
                meta: Default::default(),
                classes: vec![],
                steps: vec![blur_step(true), blur_step(true)],
                parameters: vec![],
                origin: None,
            });

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().into()).unwrap();

        let commands = &job.pipelines_pre_process[&PipelineId(1)].commands;
        assert_eq!(
            command_names(&job.pipelines_pre_process[&PipelineId(1)]),
            ["Blur", "Blur", "End Scope"],
            "the module's two blurs, then its scope ends"
        );
        assert_eq!(
            commands[1].0,
            evanalyzer_cfg::core_types::StepId(4),
            "a breakpoint on the module stops after it"
        );
    }

    fn command_names(pipeline: &Pipeline) -> Vec<&'static str> {
        pipeline.commands.iter().map(|(_, _, c)| c.name()).collect()
    }

    fn module(key: &str, steps: Vec<PipelineStepSettings>) -> ModuleDefinition {
        ModuleDefinition {
            key: key.into(),
            meta: Default::default(),
            classes: vec![],
            steps,
            parameters: vec![],
            origin: None,
        }
    }

    fn classify_step() -> PipelineStepSettings {
        PipelineStepSettings {
            id: evanalyzer_cfg::core_types::StepId(9),
            enabled: true,
            content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(
                PipelineCommand::ClassifyObjects(Default::default()),
            ),
        }
    }

    /// The tile and whole-image command names of pipeline 1 of `project`.
    fn phases(project: ProjectSettings) -> (Vec<&'static str>, Vec<&'static str>) {
        let project_dir = tempfile::tempdir().unwrap();
        let (job, _) =
            generate_preview_job_from_project_settings(project, project_dir.path().into()).unwrap();
        (
            command_names(&job.pipelines_pre_process[&PipelineId(1)]),
            command_names(&job.pipelines_post_process[&PipelineId(1)]),
        )
    }

    #[test]
    fn a_modules_scope_ends_after_its_last_command_across_both_phases() {
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(
                1,
                true,
                vec![
                    module_step("BLUR_CLASSIFY"),
                    blur_step(true),
                    classify_step(),
                ],
            )],
        );
        project.modules.push(module(
            "BLUR_CLASSIFY",
            vec![blur_step(true), classify_step()],
        ));

        let (tiles, whole_image) = phases(project);
        // The blur after the module is a tile command, but the module's last
        // command (Classify) runs in the whole-image phase - it ends there.
        assert_eq!(tiles, ["Blur", "Blur"]);
        assert_eq!(
            whole_image,
            ["Classify Objects", "End Scope", "Classify Objects"]
        );
    }

    #[test]
    fn a_nested_module_ends_before_the_module_around_it() {
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![module_step("OUTER")])],
        );
        project.modules.push(module(
            "OUTER",
            vec![
                PipelineStepSettings {
                    id: evanalyzer_cfg::core_types::StepId(1),
                    ..module_step("INNER")
                },
                classify_step(),
            ],
        ));
        project.modules.push(module("INNER", vec![classify_step()]));

        let (_, whole_image) = phases(project);
        assert_eq!(
            whole_image,
            [
                "Classify Objects",
                "End Scope",
                "Classify Objects",
                "End Scope"
            ]
        );
    }

    #[test]
    fn a_pipeline_without_modules_gets_no_scope_end() {
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![blur_step(true), classify_step()])],
        );
        let (tiles, whole_image) = phases(project);
        assert_eq!(tiles, ["Blur"]);
        assert_eq!(whole_image, ["Classify Objects"]);
    }

    #[test]
    fn a_module_the_project_doesnt_define_is_refused_with_a_clear_error() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(
                1,
                true,
                vec![blur_step(true), module_step("MISSING")],
            )],
        );

        let err = generate_preview_job_from_project_settings(project, project_dir.path().into())
            .err()
            .expect("refused")
            .to_string();
        assert!(
            err.contains("Pipeline") && err.contains("step 4") && err.contains("MISSING"),
            "{err}"
        );
    }

    #[test]
    fn dataset_steps_are_collected_apart_from_the_image_pipelines() {
        let dir = tempfile::tempdir().unwrap();
        let project = project_with_dataset_steps(dir.path().to_path_buf());

        let job = generate_job_from_project_settings_intenal(
            project.clone(),
            dir.path().to_path_buf(),
            dir.path().to_path_buf(),
            Arc::new(MemoryExporter {
                out_objects: Arc::new(Mutex::new(Vec::new())),
                ..Default::default()
            }),
        )
        .unwrap();

        assert_eq!(
            job.dataset_steps.len(),
            1,
            "enabled ones of enabled pipelines"
        );
        assert_eq!(
            job.pipelines_pre_process[&PipelineId(1)].commands.len(),
            1,
            "blur"
        );
        assert_eq!(
            dataset_steps_from_project_settings(&project).unwrap().len(),
            1
        );
    }

    #[test]
    fn each_command_keeps_the_id_of_its_project_step() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let mut steps = vec![blur_step(true), blur_step(false), blur_step(true)];
        for (i, step) in steps.iter_mut().enumerate() {
            step.id = evanalyzer_cfg::core_types::StepId(i as u32 + 1);
        }
        let ids = [steps[0].id, steps[2].id];
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, steps)],
        );

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        let built: Vec<_> = job.pipelines_pre_process[&PipelineId(1)]
            .commands
            .iter()
            .map(|(id, _, _)| *id)
            .collect();
        assert_eq!(
            built, ids,
            "what a breakpoint targets, whatever the position"
        );
    }

    /// Two measurement steps named the same can't run: their values would
    /// mix in the results.
    #[test]
    fn a_job_with_two_measurement_steps_of_one_name_is_refused() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let pearson = || {
            let mut command =
                PipelineCommand::default_for_key("pearson_correlation_coefficient").unwrap();
            *command.measurement_name_mut().unwrap() = "coloc".into();
            PipelineStepSettings {
                id: evanalyzer_cfg::core_types::StepId::UNASSIGNED,
                enabled: true,
                content: evanalyzer_cfg::settings::pipeline_settings::StepContent::Command(command),
            }
        };
        let project = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![pearson(), pearson()])],
        );
        let Err(err) =
            generate_preview_job_from_project_settings(project, project_dir.path().into())
        else {
            panic!("refused");
        };
        assert!(
            err.to_string().contains("\"coloc\" is used more than once"),
            "{err}"
        );
    }

    #[test]
    fn a_run_with_a_measurement_name_used_twice_or_not_valid_is_refused() {
        let named = |names: &[&str]| -> Vec<(String, &'static str)> {
            names.iter().map(|n| (n.to_string(), "Nuclei")).collect()
        };
        assert!(check_measurement_names(&named(&["pearson_01", "texture_01"])).is_ok());
        let twice = check_measurement_names(&named(&["pearson_01", "PEARSON_01"]))
            .unwrap_err()
            .to_string();
        assert!(
            twice.contains("\"PEARSON_01\" is used more than once"),
            "{twice}"
        );
        let bad = check_measurement_names(&named(&["coloc nuclei"]))
            .unwrap_err()
            .to_string();
        assert!(bad.contains("only have letters, digits and _"), "{bad}");
        assert!(check_measurement_names(&named(&[""])).is_err(), "no name");
    }

    #[test]
    fn a_preview_runs_no_dataset_steps() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with_dataset_steps(image_root.path().to_path_buf());

        let (job, _out_objects) =
            generate_preview_job_from_project_settings(project, project_dir.path().to_path_buf())
                .unwrap();

        assert!(job.dataset_steps.is_empty());
    }

    /// An earlier analysis `results/<date>__base/` with three objects, its
    /// settings snapshot and one more file. Returns its database.
    fn earlier_analysis(project_dir: &Path) -> PathBuf {
        use crate::object::{Object, ObjectInit};
        use crate::pipeline::pipeline_cache::GlobalPipelineCache;
        use evanalyzer_cfg::core_types::ObjectId;

        let dir = project_dir.join("results").join("20260101T000000Z__base");
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join(format!("base.{RESULTS_FILE_EXTENSION}"));
        let exporter = DuckDbExporter::new(&db, Default::default(), &Default::default()).unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_rel_path = PathBuf::from("a.tif");
        cache.image_meta.nr_of_bits = 8;
        for id in 1..=3u128 {
            cache.object_cache.insert(
                ObjectId(id),
                Object::new(ObjectInit {
                    id: ObjectId(id),
                    area: 1,
                    bbox: [0, 0, 1, 1],
                    ..Default::default()
                }),
            );
        }
        exporter.export(&cache).unwrap();
        exporter.finish_run(&Ok(())).unwrap();
        drop(exporter);
        std::fs::write(dir.join(format!("base.{PROJECT_FILE_EXTENSIONS}")), "{}").unwrap();
        std::fs::write(dir.join("notes.txt"), "kept").unwrap();
        db
    }

    fn run_status(db: &Path) -> String {
        crate::storage::duckdb::shared_connection(db)
            .unwrap()
            .query_row("SELECT status FROM run", [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn a_dataset_only_run_works_on_a_renamed_copy_of_the_earlier_results() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let base_db = earlier_analysis(project_dir.path());
        let project = project_with_dataset_steps(image_root.path().to_path_buf());

        let mut job = generate_dataset_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
            Some("tracked".into()),
            &base_db,
        )
        .unwrap();
        assert!(job.skip_image_phase);
        assert_eq!(job.dataset_steps.len(), 1, "the project's tracker");

        // The tracker can't run yet - record what a dataset step sees instead.
        let seen = Arc::new(Mutex::new(Vec::new()));
        job.dataset_steps = vec![Box::new(crate::dataset::test_support::RecordingStep {
            seen: seen.clone(),
            fail: false,
        })];
        let copy = job.output_path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .unwrap();

        assert_eq!(*seen.lock().unwrap(), [3], "the earlier analysis's objects");
        let started = rx.into_iter().find_map(|e| match e {
            crate::ProgressEvent::Started { total } => Some(total),
            _ => None,
        });
        assert_eq!(started, Some(0), "no image analysed");

        let name = copy.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.ends_with("__tracked") && name != "20260101T000000Z__tracked");
        let new_db = copy.join(format!("tracked.{RESULTS_FILE_EXTENSION}"));
        assert_eq!(run_status(&new_db), "finished");
        assert!(
            copy.join(format!("tracked.{PROJECT_FILE_EXTENSIONS}"))
                .is_file()
        );
        assert_eq!(
            std::fs::read_to_string(copy.join(format!("tracked.base.{PROJECT_FILE_EXTENSIONS}")))
                .unwrap(),
            "{}",
            "the earlier run's settings"
        );
        assert!(copy.join("notes.txt").is_file());
        assert!(base_db.is_file(), "the earlier analysis is untouched");
        assert_eq!(run_status(&base_db), "finished");
    }

    #[test]
    fn a_dataset_only_run_is_refused_before_anything_is_created() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let base_db = earlier_analysis(project_dir.path());
        let results = project_dir.path().join("results");
        let runs = || std::fs::read_dir(&results).unwrap().count();
        let start = |project: ProjectSettings, base: &Path| {
            generate_dataset_job_from_project_settings(
                project,
                project_dir.path().to_path_buf(),
                None,
                base,
            )
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
        };

        let no_dataset_steps = project_with(
            Some(image_root.path().to_path_buf()),
            vec![pipeline(1, true, vec![blur_step(true)])],
        );
        assert!(start(no_dataset_steps, &base_db).contains("no enabled dataset steps"));

        let project = project_with_dataset_steps(image_root.path().to_path_buf());
        let elsewhere = image_root
            .path()
            .join(format!("other.{RESULTS_FILE_EXTENSION}"));
        std::fs::write(&elsewhere, "").unwrap();
        assert!(start(project.clone(), &elsewhere).contains("isn't one of this project's results"));
        let missing = results
            .join("x")
            .join(format!("x.{RESULTS_FILE_EXTENSION}"));
        assert!(start(project, &missing).contains("doesn't exist"));

        assert_eq!(runs(), 1, "only the earlier analysis");
    }

    /// A module written by hand into the project file - local
    /// classes, a module parameter bound to an inner setting, an instance with
    /// its value - loads like a saved project and runs: the objects the module
    /// extracts internally (`L1`) leave it reclassified into the project class
    /// the instance chose.
    #[test]
    fn a_hand_written_module_loads_and_runs() {
        use evanalyzer_cfg::core_types::{ObjectClass, SegmentationClass};
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ClassifyObjectsSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
            ObjectClassifyObjectsClassifyMatchHandlingSettings, ThresholdEntrySettings,
            ThresholdSettings,
        };
        use serde_json::json;
        let command = |command: PipelineCommand| serde_json::to_value(command).unwrap();
        let threshold = command(PipelineCommand::Threshold(ThresholdSettings {
            thresholds: vec![ThresholdEntrySettings {
                segmentation_class: SegmentationClass(1),
                ..Default::default()
            }],
        }));
        let components = command(PipelineCommand::ConnectedComponents(Default::default()));
        let extract = command(PipelineCommand::ExtractObjects(ExtractObjectsSettings {
            class_mapping: vec![ExtractClassMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class: ObjectClass::Local(1),
            }],
            ..Default::default()
        }));
        let reclassify = command(PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
            input_classes: vec![ObjectClass::Local(1)],
            match_handling: ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch,
            output_class: ObjectClass::Global(5),
            ..Default::default()
        }));

        let mut project = project_with(
            Some(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))),
            vec![],
        );
        let image = evanalyzer_cfg::settings::images_settings::ImageEntry {
            rel_path: PathBuf::from("multi-channel-4D-series.ome.tif"),
            series: std::collections::BTreeMap::from([(0, Default::default())]),
            ..Default::default()
        };
        project.images.list.insert(image.rel_path.clone(), image);
        let mut raw = serde_json::to_value(&project).unwrap();
        raw["schemaVersion"] = json!(evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION);
        // Written by hand: no step ids in the pipeline, the module's steps
        // numbered because the binding names them.
        raw["pipelines"] = json!([{
            "id": 1, "name": "Cells", "imageSource": {"channel": 0}, "enabled": true,
            "steps": [
                {"enabled": true, "command": threshold},
                {"enabled": true, "command": components},
                {"enabled": true, "module": {"key": "CELLS", "values": {"cell_class": 7}}}
            ]
        }]);
        raw["modules"] = json!([{
            "key": "CELLS",
            "meta": serde_json::to_value(evanalyzer_cfg::settings::meta_data::MetaData::default()).unwrap(),
            "classes": [{"id": 1, "color": "#FF0000", "name": "Candidates", "notes": ""}],
            "steps": [
                {"id": 1, "enabled": true, "command": extract},
                {"id": 2, "enabled": true, "command": reclassify}
            ],
            "parameters": [{
                "key": "cell_class", "label": "Cell class", "default": 5,
                "bindings": [{"step": 2, "param": "output_class"}]
            }]
        }]);

        let loaded =
            evanalyzer_cfg::load_project_settings(raw).expect("the hand-written project loads");
        let project_dir = tempfile::tempdir().unwrap();
        let (job, out_objects) =
            generate_preview_job_from_project_settings(loaded, project_dir.path().into()).unwrap();
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .unwrap();

        let objects = out_objects.objects.lock().unwrap();
        assert!(!objects.is_empty(), "the fixture image has objects");
        for object in objects.iter() {
            assert_eq!(
                object.object_class,
                std::collections::HashSet::from([ObjectClass::Global(7)]),
                "reclassified into the instance's class, nothing local left"
            );
        }
    }

    /// An outer module hands its own classes into a module nested in
    /// it: the inner one moves the outer one's L1 objects into a class of
    /// its own, then into the outer one's L2 - which survive the inner one's
    /// end and leave the outer one reclassified into the project class.
    #[test]
    fn an_outer_modules_classes_go_into_a_nested_module_and_back() {
        use evanalyzer_cfg::core_types::{ObjectClass, SegmentationClass};
        use evanalyzer_cfg::settings::pipeline_command_settings::{
            ClassifyObjectsSettings, ExtractClassMappingSettings, ExtractObjectsSettings,
            ObjectClassifyObjectsClassifyMatchHandlingSettings, ThresholdEntrySettings,
            ThresholdSettings,
        };
        use serde_json::json;
        let command = |command: PipelineCommand| serde_json::to_value(command).unwrap();
        let reclassify = |from: ObjectClass, to: ObjectClass| {
            command(PipelineCommand::ClassifyObjects(ClassifyObjectsSettings {
                input_classes: vec![from],
                match_handling:
                    ObjectClassifyObjectsClassifyMatchHandlingSettings::ReclassifyIfMatch,
                output_class: to,
                ..Default::default()
            }))
        };
        let threshold = command(PipelineCommand::Threshold(ThresholdSettings {
            thresholds: vec![ThresholdEntrySettings {
                segmentation_class: SegmentationClass(1),
                ..Default::default()
            }],
        }));
        let components = command(PipelineCommand::ConnectedComponents(Default::default()));
        let extract = command(PipelineCommand::ExtractObjects(ExtractObjectsSettings {
            class_mapping: vec![ExtractClassMappingSettings {
                segmentation_class: SegmentationClass(1),
                object_class: ObjectClass::Local(1),
            }],
            ..Default::default()
        }));

        let mut project = project_with(
            Some(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))),
            vec![],
        );
        let image = evanalyzer_cfg::settings::images_settings::ImageEntry {
            rel_path: PathBuf::from("multi-channel-4D-series.ome.tif"),
            series: std::collections::BTreeMap::from([(0, Default::default())]),
            ..Default::default()
        };
        project.images.list.insert(image.rel_path.clone(), image);
        let mut raw = serde_json::to_value(&project).unwrap();
        raw["schemaVersion"] = json!(evanalyzer_cfg::CURRENT_PROJECT_SCHEMA_VERSION);
        let meta =
            serde_json::to_value(evanalyzer_cfg::settings::meta_data::MetaData::default()).unwrap();
        raw["pipelines"] = json!([{
            "id": 1, "name": "Cells", "imageSource": {"channel": 0}, "enabled": true,
            "steps": [
                {"enabled": true, "command": threshold},
                {"enabled": true, "command": components},
                {"enabled": true, "module": {"key": "OUTER", "values": {"cell_class": 7}}}
            ]
        }]);
        raw["modules"] = json!([
            {
                "key": "OUTER", "meta": meta,
                "classes": [
                    {"id": 1, "color": "#FF0000", "name": "Candidates", "notes": ""},
                    {"id": 2, "color": "#00FF00", "name": "Kept", "notes": ""}
                ],
                "steps": [
                    {"id": 1, "enabled": true, "command": extract},
                    {"id": 2, "enabled": true, "module": {"key": "PASS_THROUGH"}},
                    {"id": 3, "enabled": true,
                     "command": reclassify(ObjectClass::Local(2), ObjectClass::Global(5))}
                ],
                "parameters": [
                    {"key": "from", "label": "From", "default": "L1",
                     "bindings": [{"step": 2, "param": "from"}]},
                    {"key": "to", "label": "To", "default": "L2",
                     "bindings": [{"step": 2, "param": "to"}]},
                    {"key": "cell_class", "label": "Cell class", "default": 5,
                     "bindings": [{"step": 3, "param": "output_class"}]}
                ]
            },
            {
                "key": "PASS_THROUGH", "meta": meta,
                "classes": [{"id": 1, "color": "#0000FF", "name": "Between", "notes": ""}],
                "steps": [
                    {"id": 1, "enabled": true,
                     "command": reclassify(ObjectClass::Global(5), ObjectClass::Local(1))},
                    {"id": 2, "enabled": true,
                     "command": reclassify(ObjectClass::Local(1), ObjectClass::Global(6))}
                ],
                "parameters": [
                    {"key": "from", "label": "From", "default": 5,
                     "bindings": [{"step": 1, "param": "input_classes"}]},
                    {"key": "to", "label": "To", "default": 6,
                     "bindings": [{"step": 2, "param": "output_class"}]}
                ]
            }
        ]);

        let loaded =
            evanalyzer_cfg::load_project_settings(raw).expect("the hand-written project loads");
        let project_dir = tempfile::tempdir().unwrap();
        let (job, out_objects) =
            generate_preview_job_from_project_settings(loaded, project_dir.path().into()).unwrap();
        let (tx, _rx) = std::sync::mpsc::channel();
        job.run(1, tx, Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .unwrap();

        let objects = out_objects.objects.lock().unwrap();
        assert!(
            !objects.is_empty(),
            "the objects survive the nested module's end"
        );
        for object in objects.iter() {
            assert_eq!(
                object.object_class,
                std::collections::HashSet::from([ObjectClass::Global(7)]),
                "through the nested module and back, nothing local left"
            );
        }
    }
}
