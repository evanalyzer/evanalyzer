//! # breakpoint
//!
//! Breakpoint request for a preview run - part of what a front end sends to
//! `evanalyzer_app::job::start_preview`, so it lives here rather than in core
//! (see `events.rs` for the same reasoning on the response side).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::types::ids::{PipelineId, StepId};
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
    /// The step to act at - its id, so the breakpoint stays on that step
    /// when steps are inserted, moved or removed above it.
    pub step_id: StepId,
    pub mode: BreakpointMode,
}
