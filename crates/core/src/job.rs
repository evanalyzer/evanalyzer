//! # job
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod algos_from_config;
#[cfg(test)]
mod algos_from_config_tests;
pub(crate) mod job_executor;
pub(crate) mod job_generator;
pub mod modules;
pub(crate) mod object_index;
pub(crate) mod object_scratch;
pub(crate) mod pipeline_step;
