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

/// Generates a job for a full analysis
pub fn generate_analyze_job_from_project_settings(
    config: ProjectSettings,
    project_path: PathBuf,
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
    let job_name = petname::petname(2, "_").expect("Problem in random job name generator");
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
