//! # Measure Object Colocalization
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How two channels go together in each object of a class - the maths is
//! in [`super::colocalization`].

use super::colocalization::{
    CITATIONS, Coloc, CostesMethod, SLOPE, costes_thresholds, object_values,
};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::ObjectValue;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, ObjectClass, ObjectId,
};
use macros::CommandsMeta;

/// Measures how two channels go together in each object of a class -
/// colocalization of every object as its own region.
///
/// The values of Measure Image Colocalization (but the slope) of the
/// object's pixels: the threshold is a percentage of the object's maximum,
/// RWC ranks the values of all the objects, and Costes uses the whole
/// image's Costes thresholds.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Colocalization")]
pub struct MeasureObjectColocalization {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("object_coloc_01/...").
    #[cmdsmeta(measurement_id = "object_coloc", display_name = "ID", summary = true)]
    pub id: String,

    /// The objects measured.
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

    /// The first channel (A).
    #[cmdsmeta(summary = true)]
    pub channel_a: ImageChannelIdx,

    /// The second channel (B).
    #[cmdsmeta(default = ImageChannelIdx(1), summary = true)]
    pub channel_b: ImageChannelIdx,

    /// A pixel counts for Manders, the overlap and RWC when it is brighter
    /// than this percentage of its channel's maximum (in the object).
    #[cmdsmeta(
        default = 15.0,
        min = 0.0,
        max = 100.0,
        step = 1.0,
        display_name = "Threshold (% of maximum)"
    )]
    pub threshold_percent: f32,

    /// Manders with Costes' automatic thresholds (the whole image's), and
    /// how they are found.
    #[cmdsmeta(default = CostesMethod::Faster, display_name = "Costes thresholds")]
    pub costes: CostesMethod,
}

impl ImageAlgorithm for MeasureObjectColocalization {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object colocalization: choose the objects".into(),
            ));
        }
        let coloc = self.coloc();
        // The whole image's Costes thresholds - the objects' use them.
        let costes_thresholds = if coloc.costes == CostesMethod::Off {
            (f64::NAN, f64::NAN)
        } else {
            let scale = coloc.scale(cache)?;
            let tiles = coloc.image_tiles(cache)?;
            costes_thresholds(&tiles, coloc.costes, scale)
        };
        // Each object's pixels are read when its values are computed - not
        // every object's at once (gigabytes on a whole slide).
        let objects: Vec<(&ObjectId, &Object)> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.objects, ctx.scope))
            .collect();
        let groups = coloc.tile_groups(
            &objects
                .iter()
                .map(|(_, object)| *object)
                .collect::<Vec<_>>(),
            cache,
        );
        let per_object = object_values(
            &groups,
            |i| coloc.object_pixels(objects[i].1, cache),
            coloc.threshold_percent as f64,
            costes_thresholds,
        );
        let objects: Vec<ObjectId> = objects.into_iter().map(|(id, _)| id.clone()).collect();
        for (info, (i, _)) in coloc
            .outputs(&self.id, true)
            .into_iter()
            .zip(coloc.values().filter(|(i, _)| *i != SLOPE))
        {
            cache.add_object_measurement(
                info,
                objects
                    .iter()
                    .zip(&per_object)
                    .map(|(object, values)| ObjectValue {
                        object: object.clone(),
                        value: values[i],
                        related: None,
                    }),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Colocalization"
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
        self.coloc().outputs(&self.id, true)
    }
}

impl MeasureObjectColocalization {
    fn coloc(&self) -> Coloc {
        Coloc {
            channel_a: self.channel_a,
            channel_b: self.channel_b,
            threshold_percent: self.threshold_percent,
            costes: self.costes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pipeline::PipelineImageMeta;

    /// Without objects to measure, Measure Object Colocalization says so.
    #[test]
    fn object_colocalization_needs_the_objects() {
        let step = MeasureObjectColocalization {
            id: "o".into(),
            objects: ObjectClass::Unset,
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes: CostesMethod::Off,
        };
        let mut cache = GlobalPipelineCache::default();
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        assert!(step.execute(&mut ctx, &mut cache).is_err());
    }
}
