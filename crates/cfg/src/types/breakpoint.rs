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

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BreakpointSettings {
    pub pipeline_id: PipelineId,
    /// The step to act at - its id, so the breakpoint stays on that step
    /// when steps are inserted, moved or removed above it.
    pub step_id: StepId,
    /// A step inside the module `step_id` is: its id in the module, then in
    /// a module nested there, and so on. Empty: `step_id` itself (on a
    /// module: after the whole module).
    #[serde(default)]
    pub inner: Vec<StepId>,
    pub mode: BreakpointMode,
}

impl BreakpointSettings {
    /// The id the pipeline stops at: `step_id`, or for a step inside a
    /// module the id the job gives that one command
    /// ([`StepId::INNER_BREAKPOINT`]).
    pub fn run_step(&self) -> StepId {
        if self.inner.is_empty() {
            self.step_id
        } else {
            StepId::INNER_BREAKPOINT
        }
    }
}

/// The pixel classifier step a preview run captures the prediction of, for
/// the prediction overlay (each pixel's class and how probable it is).
/// Independent of a breakpoint - both can be set.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct PredictionTarget {
    pub pipeline_id: PipelineId,
    pub step_id: StepId,
}
