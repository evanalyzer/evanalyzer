//! # Measure Image Texture
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Haralick's texture features of a channel over the whole image - the
//! maths is in [`super::texture`].

use super::texture::{HARALICK_1973, Texture, TextureDirections};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::MeasurementRegion;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo,
};
use macros::CommandsMeta;

/// Measures the texture of a channel over the whole image - Haralick's
/// features of the gray-level co-occurrence matrix, per image.
///
/// The pixel pairs the scale apart in a direction say how the gray levels
/// go together: e.g. contrast is high where neighbours differ a lot, the
/// angular second moment where few pairs repeat over and over (smooth or
/// regular), entropy where the pairs are all different.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Image Texture")]
pub struct MeasureImageTexture {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("texture_01/...").
    #[cmdsmeta(measurement_id = "texture", display_name = "ID", summary = true)]
    pub id: String,

    /// The channel measured.
    #[cmdsmeta(summary = true)]
    pub channel: ImageChannelIdx,

    /// How far apart the two pixels of a pair are, in pixels - about the
    /// size of the texture's grain.
    #[cmdsmeta(
        default = 3,
        min = 1.0,
        max = 1000.0,
        step = 1.0,
        display_name = "Scale [px]",
        summary = true
    )]
    pub scale: u32,

    /// The gray levels the 8-bit values are rescaled to (2 to 256): fewer
    /// are faster and less noisy, more see finer differences.
    #[cmdsmeta(
        default = 256,
        min = 2.0,
        max = 256.0,
        step = 1.0,
        display_name = "Gray levels"
    )]
    pub gray_levels: u32,

    /// One value per feature (the mean of the directions) or one per
    /// direction.
    #[cmdsmeta(default = TextureDirections::Mean, display_name = "Directions")]
    pub directions: TextureDirections,
}

impl ImageAlgorithm for MeasureImageTexture {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let texture = self.texture();
        let levels = texture.levels()?;
        let features = texture.image_features(cache, &levels)?;
        for (feature, value) in texture.values(&features).into_iter().enumerate() {
            cache.add_image_measurement(
                texture.output(&self.id, feature, false),
                value
                    .is_finite()
                    .then_some((MeasurementRegion::Image, value)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Image Texture"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::CELLPROFILER, &HARALICK_1973]
    }

    /// The tiles of the whole image, after they are all read.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        self.texture().outputs(&self.id, false)
    }
}

impl MeasureImageTexture {
    fn texture(&self) -> Texture {
        Texture {
            channel: self.channel,
            scale: self.scale,
            gray_levels: self.gray_levels,
            directions: self.directions,
        }
    }
}
