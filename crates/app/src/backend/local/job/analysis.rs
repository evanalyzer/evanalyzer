use super::spawn_job;
use crate::api::RunningJob;
use evanalyzer_cfg::{core_types::InternalErrors, settings::project_settings::ProjectSettings};
use std::path::PathBuf;

/// Starts a full analysis run over every image in `settings`, writing a
/// results database under `<project_path>/results`.
///
/// `threads` overrides the worker count; `None` picks it from available CPU
/// cores *and* RAM, so a low-memory machine doesn't try to run as many
/// concurrent workers as it has cores. The per-worker RAM estimate is sized
/// to the images actually being analyzed, not a flat guess - see
/// `JobExecutor::estimate_ram_per_worker_bytes`.
pub(crate) fn start_analysis(
    settings: ProjectSettings,
    project_path: PathBuf,
    job_name: Option<String>,
    threads: Option<usize>,
) -> Result<RunningJob, InternalErrors> {
    let job = evanalyzer_core::generate_analyze_job_from_project_settings(
        settings,
        project_path,
        job_name,
    )?;
    let parallelism = threads.unwrap_or_else(|| {
        evanalyzer_core::recommended_parallelism(job.estimate_ram_per_worker_bytes())
    });
    Ok(spawn_job(job, parallelism, None))
}
