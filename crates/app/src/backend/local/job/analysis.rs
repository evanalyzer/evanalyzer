//! # analysis
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::spawn_job;
use crate::api::RunningJob;
use evanalyzer_cfg::{core_types::InternalErrors, settings::project_settings::ProjectSettings};
use std::path::PathBuf;

/// Starts a full analysis run over every image in `settings`, writing a
/// results database under `<project_path>/results` - or, with
/// `base_results`, a run of only the dataset steps on a copy of that earlier
/// analysis's results.
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
    base_results: Option<PathBuf>,
) -> Result<RunningJob, InternalErrors> {
    let job = match base_results {
        Some(base) => evanalyzer_core::generate_dataset_job_from_project_settings(
            settings,
            project_path,
            job_name,
            &base,
        )?,
        None => evanalyzer_core::generate_analyze_job_from_project_settings(
            settings,
            project_path,
            job_name,
        )?,
    };
    let parallelism = threads.unwrap_or_else(|| job.recommended_parallelism());
    Ok(spawn_job(job, parallelism, None))
}
