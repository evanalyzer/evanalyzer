use crate::{
    image::PixelSizes,
    job::job_executor::JobExecutor,
    pipeline::pipeline::{CorePipelineSettings, Pipeline},
    storage::PipelineResultExporter,
};
use evanalyzer_cfg::{core_types::InternalErrors, settings::project_settings::ProjectSettings};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub fn generate_job_from_project_settings(
    config: ProjectSettings,
    project_path: PathBuf,
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
        config.images.list,
        image_base_path,
        config.images.settings,
        result_storage,
        pixel_sizes,
    );

    for pipeline_setting in &config.pipelines {
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
