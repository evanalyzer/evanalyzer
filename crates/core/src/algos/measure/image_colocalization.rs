//! # Measure Image Colocalization
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How two channels go together over the whole image - the maths is in
//! [`super::colocalization`].

use super::colocalization::{CITATIONS, Coloc, CostesMethod, image_values};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::MeasurementRegion;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo,
};
use macros::CommandsMeta;

/// Measures how two channels go together over the whole image -
/// CellProfiler's MeasureColocalization, per image.
///
/// Pearson (-1 to 1) uses every pixel. Manders M1 is the part of channel
/// A's intensity where channel B is bright too (M2 the other way round),
/// "bright" meaning above the threshold - a percentage of the channel's
/// maximum, or Costes' automatic threshold. The overlap coefficient, K1/K2
/// and RWC (Manders weighted by how similar the two pixels' ranks are) use
/// the same pixels.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Image Colocalization")]
pub struct MeasureImageColocalization {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("coloc_01/...").
    #[cmdsmeta(measurement_id = "coloc", display_name = "ID", summary = true)]
    pub id: String,

    /// The first channel (A).
    #[cmdsmeta(summary = true)]
    pub channel_a: ImageChannelIdx,

    /// The second channel (B).
    #[cmdsmeta(default = ImageChannelIdx(1), summary = true)]
    pub channel_b: ImageChannelIdx,

    /// A pixel counts for Manders, the overlap and RWC when it is brighter
    /// than this percentage of its channel's maximum (in the image).
    #[cmdsmeta(
        default = 15.0,
        min = 0.0,
        max = 100.0,
        step = 1.0,
        display_name = "Threshold (% of maximum)"
    )]
    pub threshold_percent: f32,

    /// Manders with Costes' automatic thresholds, and how they are found.
    #[cmdsmeta(default = CostesMethod::Faster, display_name = "Costes thresholds")]
    pub costes: CostesMethod,
}

impl ImageAlgorithm for MeasureImageColocalization {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let coloc = self.coloc();
        let scale = coloc.scale(cache)?;
        let tiles = coloc.image_tiles(cache)?;
        let image = image_values(&tiles, coloc.threshold_percent as f64, coloc.costes, scale);
        for (info, (i, _)) in coloc
            .outputs(&self.id, false)
            .into_iter()
            .zip(coloc.values())
        {
            let value = image.values[i];
            cache.add_image_measurement(
                info,
                value
                    .is_finite()
                    .then_some((MeasurementRegion::Image, value)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Image Colocalization"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        CITATIONS.to_vec()
    }

    /// The tiles of the whole image, after they are all read.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        self.coloc().outputs(&self.id, false)
    }
}

impl MeasureImageColocalization {
    fn coloc(&self) -> Coloc {
        Coloc {
            channel_a: self.channel_a,
            channel_b: self.channel_b,
            threshold_percent: self.threshold_percent,
            costes: self.costes,
        }
    }
}
