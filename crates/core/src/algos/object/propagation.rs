//! # Propagation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! Grows one region per seed object (typically a nucleus) through a channel
//! (typically a cell or membrane stain): every pixel goes to the seed it can
//! be reached from at the lowest cost, where a step costs more the more the
//! stain's brightness changes along it - so the border between two cells
//! falls on the dim gap between them, not halfway between the nuclei.
//! `regularization` (λ) weighs plain distance against the brightness change:
//! a large λ approaches a distance-only split, 0 one by brightness only.
//! Only pixels inside the optional mask are assigned.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use evanalyzer_cfg::core_types::{CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass};
use macros::CommandsMeta;

/// Grows a region around each seed object (e.g. a cell around its nucleus)
/// through a channel (e.g. the cell stain), so neighbouring regions meet
/// where the stain is dim. Each region is linked to its seed.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct Propagation {
    /// Objects carrying this class are the seeds (e.g. nuclei): each grows
    /// into one region.
    #[cmdsmeta(summary = true)]
    pub seeds: ObjectClass,

    /// The channel the regions grow through (e.g. the cell or membrane
    /// stain).
    #[cmdsmeta(summary = true)]
    pub growth_channel: ImageChannelIdx,

    /// Regions only cover pixels of objects carrying this class (e.g. the
    /// thresholded cell stain). `Unset`: the whole image.
    pub mask: ObjectClass,

    /// How much plain distance counts against the change in brightness
    /// (λ): larger values split more by distance, `0` by brightness only.
    #[cmdsmeta(default = 0.05, min = 0.0, step = 0.01)]
    pub regularization: f32,

    /// The class the grown regions get.
    pub output_class: ObjectClass,

    /// Discard regions that touch the image border.
    pub exclude_at_edges: bool,
}

impl ImageAlgorithm for Propagation {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        Err(InternalErrors::Generic(
            "Propagation isn't implemented yet".into(),
        ))
    }

    fn name(&self) -> &'static str {
        "Propagation"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&JONES_2005]
    }

    /// The seeds and channel of the whole image: a region may grow across
    /// tile borders.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

/// Where the method was published (CellProfiler's "Propagation").
const JONES_2005: CitationMetadata = CitationMetadata {
    cite_key: "jones2005voronoi",
    title: "Voronoi-Based Segmentation of Cells on Image Manifolds",
    authors: &["Thouis R. Jones", "Anne Carpenter", "Polina Golland"],
    year: 2005,
    container: Some("Computer Vision for Biomedical Image Applications (CVBIA), LNCS 3765"),
    doi: Some("10.1007/11569541_54"),
    url: Some("https://doi.org/10.1007/11569541_54"),
    pages: Some("535-543"),
};
