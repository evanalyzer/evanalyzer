//! # jobs
//!
//! Analyses and previews: what is asked for ([`AnalysisRequest`],
//! [`PreviewRequest`]), a job running ([`RunningJob`]), and running them in
//! this process ([`local`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod local;
mod types;

pub use types::*;

pub use evanalyzer_core::check_pipeline_channels;
pub use evanalyzer_core::random_job_name;
pub use evanalyzer_core::{
    ObjectPredictionCapture, PredictedClass, PredictedObjectClass, PredictionCapture,
};
