use std::path::PathBuf;

use evanalyzer_cfg::{
    core_types::{BreakpointMode, PipelineId},
    settings::project_settings::ProjectSettings,
};

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
    /// Instead of starting an analysis, follow the one with this id that is
    /// already running on the server (started before this window existed,
    /// or before the connection dropped).
    pub attach: Option<String>,
}

impl Default for PipelineTask {
    fn default() -> Self {
        Self {
            project_settings: ProjectSettings::default(),
            project_path: PathBuf::default(),
            preview: false,
            breakpoint: None,
            job_name: None,
            attach: None,
        }
    }
}

impl PipelineTask {}
