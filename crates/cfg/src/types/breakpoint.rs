//! Breakpoint request for a preview run - part of what a front end sends to
//! `evanalyzer_app::job::start_preview`, so it lives here rather than in core
//! (see `events.rs` for the same reasoning on the response side).

use crate::types::ids::PipelineId;
use serde::{Deserialize, Serialize};

/// Controls pipeline behaviour when a breakpoint step is reached.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BreakpointMode {
    /// Stop the pipeline at this step and return the intermediate image.
    Stop,
    /// Capture the image at this step, then continue running the pipeline
    /// to completion.  The final results (ROIs, DB write) are produced
    /// normally; the captured image is sent as a side-channel preview.
    Snapshot,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BreakpointSettings {
    pub pipeline_id: PipelineId,
    pub pipeline_step_id: i32,
    pub mode: BreakpointMode,
}
