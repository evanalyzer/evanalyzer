//! # Measure Distance To Parent
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! How far each child lies from the parent Relate Objects linked it to - the
//! distances of CellProfiler's RelateObjects ("Calculate child-parent
//! distances", `calculate_centroid_distances` / `calculate_minimum_distances`
//! in cellprofiler/modules/relateobjects.py, CellProfiler 4.2.8): from the
//! child's centre of mass to the parent's centre of mass, and to the
//! nearest pixel of the parent's edge.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass,
};
use macros::CommandsMeta;

/// Each child's distance to its parent of a class (linked by Relate
/// Objects): centre to centre, and from its centre to the parent's edge.
/// A child without a parent of the class has no value.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Distance To Parent")]
pub struct MeasureDistanceToParent {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("parent_distance_01/...").
    #[cmdsmeta(
        measurement_id = "parent_distance",
        display_name = "ID",
        summary = true
    )]
    pub id: String,

    /// The children measured (e.g. spots).
    #[cmdsmeta(summary = true)]
    pub children: ObjectClass,

    /// The class of their parent (as linked by Relate Objects, e.g. Cell).
    #[cmdsmeta(summary = true, display_name = "Parent")]
    pub parent_class: ObjectClass,
}

impl ImageAlgorithm for MeasureDistanceToParent {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.children == ObjectClass::Unset || self.parent_class == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure distance to parent: choose the children and the parent's class".into(),
            ));
        }
        let size = cache.image_meta.full_image_width;
        let image = (size.width as i64, size.height as i64);
        let (mut centroid, mut minimum) = (Vec::new(), Vec::new());
        for (id, child) in cache.object_cache.iter() {
            if !child.is_of(&self.children, ctx.scope) {
                continue;
            }
            let parent = child
                .parent(&self.parent_class)
                .and_then(|p| cache.object_cache.get(p).map(|o| (p, o)));
            let (c, m) = match parent {
                Some((_, parent)) => {
                    let center = center_of_mass(child);
                    let (px, py) = center_of_mass(parent);
                    (
                        ((center.0 - px).powi(2) + (center.1 - py).powi(2)).sqrt(),
                        distance_to_edge(center, parent, image),
                    )
                }
                None => (f64::NAN, f64::NAN),
            };
            let related = parent.map(|(p, _)| p.clone());
            centroid.push(ObjectValue {
                object: id.clone(),
                value: c,
                related: related.clone(),
            });
            minimum.push(ObjectValue {
                object: id.clone(),
                value: m,
                related,
            });
        }
        cache.add_object_measurement(self.measurement_output("centroid_distance"), centroid);
        cache.add_object_measurement(self.measurement_output("minimum_distance"), minimum);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Distance To Parent"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::CELLPROFILER, &CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        vec![
            outputs
                .object(
                    "centroid_distance",
                    "Distance to parent centre",
                    MeasurementUnit::Pixels,
                )
                .naming_objects(),
            outputs
                .object(
                    "minimum_distance",
                    "Distance to parent edge",
                    MeasurementUnit::Pixels,
                )
                .naming_objects(),
        ]
    }
}

/// The mean pixel position (x, y) of `object`'s mask, in f64.
fn center_of_mass(object: &Object) -> (f64, f64) {
    let [x0, y0, x1, _] = object.bbox;
    let w = (x1 - x0 + 1) as usize;
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
    for i in object.mask_data.iter_ones() {
        sx += (x0 as usize + i % w) as f64;
        sy += (y0 as usize + i / w) as f64;
        n += 1.0;
    }
    (sx / n, sy / n)
}

/// The distance from `point` to the nearest edge pixel of `parent`: a pixel
/// of it with a pixel beside it (4-neighbourhood, inside the image) that
/// isn't - scikit-image's `find_boundaries(mode="inner")` as CellProfiler
/// uses it (beyond the image edge counts as the same, its "reflect");
/// `image`: the image's width and height.
fn distance_to_edge(point: (f64, f64), parent: &Object, (width, height): (i64, i64)) -> f64 {
    let [x0, y0, x1, y1] = parent.bbox;
    let w = (x1 - x0 + 1) as usize;
    let inside = |x: i64, y: i64| {
        x >= x0 as i64
            && y >= y0 as i64
            && x <= x1 as i64
            && y <= y1 as i64
            && parent
                .mask_data
                .get((y - y0 as i64) as usize * w + (x - x0 as i64) as usize)
                .is_some_and(|b| *b)
    };
    let mut best = f64::INFINITY;
    for i in parent.mask_data.iter_ones() {
        let (x, y) = ((x0 as usize + i % w) as i64, (y0 as usize + i / w) as i64);
        let edge = [(0, -1), (-1, 0), (1, 0), (0, 1)].iter().any(|(dx, dy)| {
            let (nx, ny) = (x + dx, y + dy);
            // Off the image: as the pixel itself (reflected).
            let on_image = nx >= 0 && ny >= 0 && nx < width && ny < height;
            on_image && !inside(nx, ny)
        });
        if edge {
            let d = ((x as f64 - point.0).powi(2) + (y as f64 - point.1).powi(2)).sqrt();
            best = best.min(d);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;
    use kornia_image::ImageSize;

    const CELL: ObjectClass = ObjectClass::Global(1);
    const SPOT: ObjectClass = ObjectClass::Global(2);
    const SPOT_IDS: u128 = 1000;

    #[derive(serde::Deserialize)]
    struct Case {
        width: usize,
        height: usize,
        parents: Vec<u32>,
        children: Vec<u32>,
        parent_of_child: Vec<u32>,
        centroid: Vec<Option<f64>>,
        minimum: Vec<Option<f64>>,
    }

    fn objects(labels: &[u32], width: usize, class: ObjectClass, ids: u128) -> Vec<Object> {
        let max = labels.iter().copied().max().unwrap_or(0);
        (1..=max)
            .map(|label| {
                let pixels: Vec<(u32, u32)> = (0..labels.len())
                    .filter(|&p| labels[p] == label)
                    .map(|p| ((p % width) as u32, (p / width) as u32))
                    .collect();
                let x0 = pixels.iter().map(|p| p.0).min().unwrap();
                let y0 = pixels.iter().map(|p| p.1).min().unwrap();
                let x1 = pixels.iter().map(|p| p.0).max().unwrap();
                let y1 = pixels.iter().map(|p| p.1).max().unwrap();
                let w = (x1 - x0 + 1) as usize;
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
                for (x, y) in &pixels {
                    mask.set((y - y0) as usize * w + (x - x0) as usize, true);
                }
                let mut object = Object::new(ObjectInit {
                    id: ObjectId(ids + label as u128),
                    bbox: [x0, y0, x1, y1],
                    area: pixels.len(),
                    mask_data: mask,
                    ..Default::default()
                });
                object.add_object_class(class, Scope::Global).unwrap();
                object
            })
            .collect()
    }

    /// Both distances as CellProfiler's RelateObjects gives them (fixture
    /// from `tests/distance_to_parent_cellprofiler.py`); none without a
    /// parent; each names the parent.
    #[test]
    fn distances_are_cellprofilers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/distance_to_parent_cellprofiler.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let case = &cases[0];
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.full_image_width = ImageSize {
            width: case.width,
            height: case.height,
        };
        for o in objects(&case.parents, case.width, CELL, 0) {
            cache.object_cache.insert(o.id.clone(), o);
        }
        for (i, mut o) in objects(&case.children, case.width, SPOT, SPOT_IDS)
            .into_iter()
            .enumerate()
        {
            let parent = case.parent_of_child[i];
            o.set_parent(CELL, (parent > 0).then(|| ObjectId(parent as u128)));
            cache.object_cache.insert(o.id.clone(), o);
        }
        let step = MeasureDistanceToParent {
            id: "d".into(),
            children: SPOT,
            parent_class: CELL,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        for (key, expected) in [
            ("d/centroid_distance", &case.centroid),
            ("d/minimum_distance", &case.minimum),
        ] {
            let m = cache
                .measurements
                .measurements
                .iter()
                .find(|m| m.info.key == key)
                .unwrap();
            let MeasurementValues::Objects(values) = &m.values else {
                panic!()
            };
            assert_eq!(values.len(), expected.len());
            for v in values {
                let i = (v.object.0 - SPOT_IDS) as usize - 1;
                let parent = case.parent_of_child[i];
                match expected[i] {
                    Some(e) => {
                        assert!(
                            (v.value - e).abs() <= 1e-9 * e.max(1.0),
                            "{key} child {}: {} vs {e}",
                            i + 1,
                            v.value
                        );
                        assert_eq!(v.related, Some(ObjectId(parent as u128)));
                    }
                    None => {
                        assert!(v.value.is_nan(), "{key} child {}: {}", i + 1, v.value);
                        assert_eq!(v.related, None);
                    }
                }
            }
        }
    }
}
