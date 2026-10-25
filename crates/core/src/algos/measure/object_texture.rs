//! # Measure Object Texture
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Haralick's texture features of a channel in each object of a class -
//! the maths is in [`super::texture`].

use super::texture::{HARALICK_1973, Texture, TextureDirections};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::ObjectValue;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// Measures the texture of a channel in each object of a class - Haralick's
/// features (CellProfiler's MeasureTexture, per object).
///
/// The same values as Measure Image Texture, of the object's pixels only.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Texture")]
pub struct MeasureObjectTexture {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("object_texture_01/...").
    #[cmdsmeta(measurement_id = "object_texture", display_name = "ID", summary = true)]
    pub id: String,

    /// The objects measured.
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

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

impl ImageAlgorithm for MeasureObjectTexture {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object texture: choose the objects".into(),
            ));
        }
        let texture = self.texture();
        let levels = texture.levels(cache)?;
        let objects: Vec<_> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.objects, ctx.scope))
            .collect();
        let per_object: Vec<(ObjectId, Vec<f64>)> = objects
            .par_iter()
            .map(|(id, object)| {
                let features = texture.object_features(object, cache, &levels);
                ((*id).clone(), texture.values(&features))
            })
            .collect();
        for feature in 0..texture.output_count() {
            let values: Vec<ObjectValue> = per_object
                .iter()
                .map(|(id, values)| ObjectValue {
                    object: id.clone(),
                    value: values[feature],
                    related: None,
                })
                .collect();
            cache.add_object_measurement(texture.output(&self.id, feature, true), values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Texture"
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
        self.texture().outputs(&self.id, true)
    }
}

impl MeasureObjectTexture {
    fn texture(&self) -> Texture {
        Texture {
            channel: self.channel,
            scale: self.scale,
            gray_levels: self.gray_levels,
            directions: self.directions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pipeline::PipelineImageMeta;

    /// Without objects to measure, Measure Object Texture says so.
    #[test]
    fn object_texture_needs_the_objects() {
        let step = MeasureObjectTexture {
            id: "o".into(),
            objects: ObjectClass::Unset,
            channel: ImageChannelIdx(0),
            scale: 3,
            gray_levels: 256,
            directions: TextureDirections::Mean,
        };
        let mut cache = GlobalPipelineCache::default();
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        assert!(step.execute(&mut ctx, &mut cache).is_err());
    }
}
