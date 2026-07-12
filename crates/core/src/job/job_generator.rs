use crate::{
    DuckDbExporter, MemoryExporter,
    image::PixelSizes,
    job::job_executor::JobExecutor,
    pipeline::pipeline::{CorePipelineSettings, Pipeline},
    storage::PipelineResultExporter,
};
use chrono::Utc;
use evanalyzer_cfg::RESULTS_FILE_EXTENSION;
use evanalyzer_cfg::{
    core_types::InternalErrors,
    settings::{project_settings::ProjectSettings, roi_settings::RoiSettings},
};
use log::{error, info};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// Generate a job just for preview
///
/// No output data are written to disk, results are just stored in memory
pub fn generate_preview_job_from_project_settings(
    config: ProjectSettings,
    project_path: PathBuf,
) -> Result<JobExecutor, InternalErrors> {
    let out_rois: Arc<Mutex<Vec<RoiSettings>>> = Arc::new(Mutex::new(vec![]));
    let memory_storage = Arc::new(Mutex::new(MemoryExporter {
        out_rois: out_rois.clone(),
    }));

    let output_path = project_path.join("results").join("preview");
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        error!("Failed to create preview output directory: {e}");
        return Err(InternalErrors::Io(format!("{e}")));
    }

    generate_job_from_project_settings_intenal(config, project_path, output_path, memory_storage)
}

/// Generates a job for a full analysis.
///
/// `job_name` is used verbatim (after sanitizing for filesystem-illegal
/// characters) as both the results-subdirectory suffix and the `.evadb`
/// filename. Pass `None` (or an empty/whitespace-only string) to fall back to
/// a randomly generated two-word name, as before this parameter existed.
pub fn generate_analyze_job_from_project_settings(
    config: ProjectSettings,
    project_path: PathBuf,
    job_name: Option<String>,
) -> Result<JobExecutor, InternalErrors> {
    let class_names: std::collections::HashMap<_, _> = config
        .classification
        .classes
        .iter()
        .filter_map(|c| {
            c.id.to_u32().map(|n| {
                (
                    evanalyzer_cfg::core_types::ObjectClass::Valid(n),
                    c.name.clone(),
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
        .unwrap_or_else(|| {
            petname::petname(2, "_").expect("Problem in random job name generator")
        });
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

    generate_job_from_project_settings_intenal(config, project_path, output_path, database_storage)
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
        .map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
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

    for pipeline_setting in &config.pipelines {
        if !pipeline_setting.enabled {
            continue;
        }

        let mut pipeline = Pipeline::new(
            pipeline_setting.id.clone(),
            CorePipelineSettings {
                start_image: pipeline_setting.image_source,
            },
        );

        for step in &pipeline_setting.steps {
            if step.enabled {
                pipeline.add_command(super::algos_from_config::into_algorithm(
                    step.command.clone(),
                )?);
            }
        }

        job.add_pipeline(pipeline);
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
            name: None,
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

        let job = generate_preview_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
        )
        .expect("valid config with an image root must succeed");

        let expected_output = project_dir.path().join("results").join("preview");
        assert!(expected_output.is_dir(), "preview output directory must be created on disk");
        assert_eq!(job.output_path, expected_output);
        assert_eq!(job.project_path, project_dir.path());
        assert_eq!(job.image_base_path, image_root.path());
        assert!(job.pipelines.is_empty());
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

        let job = generate_preview_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
        )
        .unwrap();

        assert_eq!(job.pipelines.len(), 1, "only the enabled pipeline must be added");
        assert!(job.pipelines.contains_key(&PipelineId(2)));
        assert!(!job.pipelines.contains_key(&PipelineId(1)));
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

        let job = generate_preview_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
        )
        .unwrap();

        let built = job.pipelines.get(&PipelineId(1)).expect("pipeline 1 must exist");
        assert_eq!(built.commands.len(), 2, "only the two enabled steps must become commands");
    }

    #[test]
    fn pixel_size_override_is_forwarded_when_set() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let mut project = project_with(Some(image_root.path().to_path_buf()), vec![]);
        project.images.settings.pixel_sizes = Some(PixelSizeSettings { x: 1.5, y: 2.5, z: 3.5 });

        let job = generate_preview_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
        )
        .unwrap();

        let sizes = job.override_pixel_sizes.expect("pixel size override must be forwarded");
        assert_eq!(sizes.px_size_x, 1.5);
        assert_eq!(sizes.px_size_y, 2.5);
        assert_eq!(sizes.px_size_z, 3.5);
    }

    #[test]
    fn pixel_size_override_is_none_when_unset() {
        let project_dir = tempfile::tempdir().unwrap();
        let image_root = tempfile::tempdir().unwrap();
        let project = project_with(Some(image_root.path().to_path_buf()), vec![]);

        let job = generate_preview_job_from_project_settings(
            project,
            project_dir.path().to_path_buf(),
        )
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

        assert_eq!(job.pipelines.len(), 1);
        assert!(
            job.output_path.starts_with(project_dir.path().join("results")),
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

        let folder_name = job.output_path.file_name().unwrap().to_string_lossy().into_owned();
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
        assert_eq!(job.output_path.parent().unwrap(), project_dir.path().join("results"));
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
            let folder_name = job.output_path.file_name().unwrap().to_string_lossy().into_owned();
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
