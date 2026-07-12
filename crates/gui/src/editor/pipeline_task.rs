use std::path::PathBuf;

use evanalyzer_cfg::{core_types::PipelineId, settings::project_settings::ProjectSettings};
use evanalyzer_core::BreakpointMode;

#[derive(Debug)]
pub struct PipelineTask {
    pub project_settings: ProjectSettings,
    pub project_path: PathBuf,
    pub preview: bool,
    /// Optional breakpoint: (pipeline_id, step_id, mode).
    pub breakpoint: Option<(PipelineId, i32, BreakpointMode)>,
    /// Optional user-chosen name for a full (non-preview) run, forwarded to
    /// `generate_analyze_job_from_project_settings`. Ignored for preview runs.
    pub job_name: Option<String>,
}

impl Default for PipelineTask {
    fn default() -> Self {
        Self {
            project_settings: ProjectSettings::default(),
            project_path: PathBuf::default(),
            preview: false,
            breakpoint: None,
            job_name: None,
        }
    }
}

impl PipelineTask {}
