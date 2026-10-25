//! # pipeline_step
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-15
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::{algos::ImageAlgorithm, dataset::DatasetAlgorithm};
use evanalyzer_cfg::core_types::CitationMetadata;

/// A configured pipeline step, ready to run - an image step (runs per tile
/// or per image, see `ExecutionScope`) or a dataset step (runs once after
/// every image has been analysed, on the objects of all of them). Built by
/// the generated `into_pipeline_step`.
pub enum PipelineStep {
    Image(Box<dyn ImageAlgorithm>),
    Dataset(Box<dyn DatasetAlgorithm>),
}

impl PipelineStep {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Image(algo) => algo.name(),
            Self::Dataset(algo) => algo.name(),
        }
    }

    pub fn cite(&self) -> Vec<&'static CitationMetadata> {
        match self {
            Self::Image(algo) => algo.cite(),
            Self::Dataset(algo) => algo.cite(),
        }
    }
}
