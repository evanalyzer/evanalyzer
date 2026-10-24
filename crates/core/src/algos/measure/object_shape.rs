//! # Measure Object Shape
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! The shape values of CellProfiler's MeasureObjectSizeShape the objects
//! don't carry anyway (they have area, perimeter, Feret, axes, ...): the
//! convex area and CellProfiler's solidity, the Euler number and the radii -
//! a step of its own because the radii need a distance transform of every
//! object (~10 µs for a 24 px object), too much to pay in every analysis.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// For each object: its convex area and solidity as CellProfiler gives them
/// (the pixels whose centre lies in the convex hull; area / convex area),
/// its Euler number (1 - holes, 8-connected) and its radii (the distance of
/// its pixels to the nearest pixel outside it: maximum, mean, median).
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Shape")]
pub struct MeasureObjectShape {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("shape_01/...").
    #[cmdsmeta(measurement_id = "shape", display_name = "ID", summary = true)]
    pub id: String,

    /// The objects measured.
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,
}

/// The values: (id, name, unit).
const VALUES: [(&str, &str, MeasurementUnit); 6] = [
    ("convex_area", "Convex area", MeasurementUnit::SquarePixels),
    (
        "solidity",
        "Solidity (convex area)",
        MeasurementUnit::Fraction,
    ),
    ("euler_number", "Euler number", MeasurementUnit::None),
    ("max_radius", "Max radius", MeasurementUnit::Pixels),
    ("mean_radius", "Mean radius", MeasurementUnit::Pixels),
    ("median_radius", "Median radius", MeasurementUnit::Pixels),
];

impl ImageAlgorithm for MeasureObjectShape {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object shape: choose the objects".into(),
            ));
        }
        let objects: Vec<_> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.objects, ctx.scope))
            .collect();
        // Each object on its own: in parallel.
        let rows: Vec<[f64; VALUES.len()]> = objects
            .par_iter()
            .map(|(_, object)| {
                let shape = object.shape();
                let radii = shape.radii;
                [
                    shape.convex_area as f64,
                    object.area as f64 / shape.convex_area as f64,
                    shape.euler_number as f64,
                    radii.map_or(f64::NAN, |r| r.max),
                    radii.map_or(f64::NAN, |r| r.mean),
                    radii.map_or(f64::NAN, |r| r.median),
                ]
            })
            .collect();
        let per_output: Vec<Vec<ObjectValue>> = (0..VALUES.len())
            .map(|i| {
                objects
                    .iter()
                    .zip(&rows)
                    .map(|((id, _), row)| ObjectValue {
                        object: (*id).clone(),
                        value: row[i],
                        related: None,
                    })
                    .collect()
            })
            .collect();
        for ((id, ..), values) in VALUES.iter().zip(per_output) {
            cache.add_object_measurement(self.measurement_output(id), values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Shape"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![
            &CitationMetadata::CELLPROFILER,
            &FELZENSZWALB_2012,
            &GRAY_1971,
            &ANDREW_1979,
        ]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        VALUES
            .iter()
            .map(|(id, name, unit)| {
                let info = outputs.object(id, name, *unit);
                if *id == "solidity" {
                    info.range(0.0, 1.0)
                } else {
                    info
                }
            })
            .collect()
    }
}

/// The exact distance transform the radii come from.
const FELZENSZWALB_2012: CitationMetadata = CitationMetadata {
    cite_key: "felzenszwalb2012distance",
    title: "Distance Transforms of Sampled Functions",
    authors: &["Pedro F. Felzenszwalb", "Daniel P. Huttenlocher"],
    year: 2012,
    container: Some("Theory of Computing"),
    doi: Some("10.4086/toc.2012.v008a019"),
    url: Some("https://doi.org/10.4086/toc.2012.v008a019"),
    pages: Some("415-428"),
};

/// The Euler number from 2 x 2 pixel patterns (bit quads).
const GRAY_1971: CitationMetadata = CitationMetadata {
    cite_key: "gray1971local",
    title: "Local Properties of Binary Images in Two Dimensions",
    authors: &["Stephen B. Gray"],
    year: 1971,
    container: Some("IEEE Transactions on Computers"),
    doi: Some("10.1109/T-C.1971.223289"),
    url: Some("https://doi.org/10.1109/T-C.1971.223289"),
    pages: Some("551-561"),
};

/// The convex hull (monotone chain) of the convex area.
const ANDREW_1979: CitationMetadata = CitationMetadata {
    cite_key: "andrew1979another",
    title: "Another efficient algorithm for convex hulls in two dimensions",
    authors: &["A. M. Andrew"],
    year: 1979,
    container: Some("Information Processing Letters"),
    doi: Some("10.1016/0020-0190(79)90072-3"),
    url: Some("https://doi.org/10.1016/0020-0190(79)90072-3"),
    pages: Some("216-219"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Object, ObjectInit};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;

    /// A 3 x 3 ring (one hole): Euler number 0, convex area 9, radii 1.
    #[test]
    fn a_ring_has_a_hole_and_radius_one() {
        let mut mask = BitVec::<u64, Lsb0>::repeat(true, 9);
        mask.set(4, false);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(1),
            bbox: [10, 10, 12, 12],
            area: 8,
            mask_data: mask,
            ..Default::default()
        });
        object
            .add_object_class(ObjectClass::Global(1), Scope::Global)
            .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(ObjectId(1), object);
        let step = MeasureObjectShape {
            id: "s".into(),
            objects: ObjectClass::Global(1),
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        let value = |id: &str| match &cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == format!("s/{id}"))
            .unwrap()
            .values
        {
            MeasurementValues::Objects(v) => v[0].value,
            _ => panic!(),
        };
        assert_eq!(value("euler_number"), 0.0);
        assert_eq!(value("convex_area"), 9.0);
        assert_eq!(value("solidity"), 8.0 / 9.0);
        assert_eq!(value("max_radius"), 1.0);
        assert_eq!(value("median_radius"), 1.0);
    }
}
