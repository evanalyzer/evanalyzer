//! # Relate Objects
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! Links each child object (e.g. a spot) to the parent object it belongs to
//! (e.g. its cell): at most one parent per parent class, so a spot can
//! belong to a cell and to a tissue region at the same time. This is the
//! one step that links objects - the counts and means per parent are
//! computed from these links when the results are read, as in
//! CellProfiler's RelateObjects.
//!
//! A child belongs to the parent its centre of mass lies in, or to the
//! parent it overlaps most. Equal candidates go to the lower object id, so
//! the link doesn't depend on the order the objects are stored in.

use crate::algos::flow_control::end_scope::remove_where;
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::utils::spatial_grid::BboxGrid;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectClass, ObjectId};
use macros::CommandsMeta;

/// How a child is matched to a parent.
#[derive(CommandsMeta, Debug, PartialEq)]
pub enum RelateRule {
    /// The child belongs to the parent its centre of mass lies in - robust
    /// for small children such as spots in a cell.
    #[cmdsmeta(display_name = "Its centre is in the parent")]
    CentreInside,

    /// The child belongs to the parent it shares the most pixels with -
    /// for children whose centre can lie outside themselves (crescents, the
    /// overlap regions of Colocalization).
    #[cmdsmeta(display_name = "It overlaps the parent most")]
    LargestOverlap,
}

/// Links objects to the objects they belong to: each child (e.g. a spot)
/// gets at most one parent of each parent class (e.g. its cell).
///
/// The results then give per parent the number of its children and the
/// mean, median, ... of their measurements, and the parent's values next to
/// each child. A child already linked to a parent of one of these classes
/// is linked anew - or unlinked when no parent matches.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", display_name = "Relate Objects")]
pub struct RelateObjects {
    /// The objects to link to a parent (e.g. spots).
    #[cmdsmeta(summary = true, display_name = "Children")]
    pub child_classes: Vec<ObjectClass>,

    /// The classes of the objects they belong to (e.g. Cell); a child gets
    /// at most one parent of each.
    #[cmdsmeta(summary = true, display_name = "Parents")]
    pub parent_classes: Vec<ObjectClass>,

    /// How a child is matched to a parent.
    #[cmdsmeta(default = RelateRule::CentreInside, display_name = "A child belongs to the parent if")]
    pub rule: RelateRule,

    /// Remove the children that don't get a parent of every parent class
    /// (e.g. keep only the spots inside cells).
    #[cmdsmeta(display_name = "Remove children without a parent")]
    pub remove_without_parent: bool,
}

impl ImageAlgorithm for RelateObjects {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let real = |classes: &[ObjectClass]| classes.iter().any(|c| *c != ObjectClass::Unset);
        if !real(&self.child_classes) || !real(&self.parent_classes) {
            return Err(InternalErrors::InvalidArgument(
                "Relate Objects: choose the children's and the parents' classes".into(),
            ));
        }
        let scope = ctx.scope;
        let is_child = |object: &Object| self.child_classes.iter().any(|c| object.is_of(c, scope));
        let children: Vec<ObjectId> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| is_child(object))
            .map(|(id, _)| id.clone())
            .collect();
        if children.is_empty() {
            return Ok(());
        }

        // The links, worked out first and written after: the cache can't be
        // read and changed at the same time.
        let mut links: Vec<(ObjectId, ObjectClass, Option<ObjectId>)> = Vec::new();
        for parent_class in self
            .parent_classes
            .iter()
            .filter(|c| **c != ObjectClass::Unset)
        {
            let parents: Vec<ObjectId> = cache
                .object_cache
                .iter()
                .filter(|(_, object)| object.is_of(parent_class, scope))
                .map(|(id, _)| id.clone())
                .collect();
            let grid = BboxGrid::build(&parents, cache);
            for child_id in &children {
                let Some(child) = cache.object_cache.get(child_id) else {
                    continue;
                };
                let parent = self.parent_of(child_id, child, &grid, cache);
                links.push((child_id.clone(), *parent_class, parent));
            }
        }

        for (child_id, class, parent) in links {
            if let Some(child) = cache.object_cache.get_mut(&child_id) {
                child.set_parent(class, parent);
            }
        }

        if self.remove_without_parent {
            let wanted: Vec<ObjectClass> = self
                .parent_classes
                .iter()
                .filter(|c| **c != ObjectClass::Unset)
                .copied()
                .collect();
            remove_where(&mut cache.object_cache, |object| {
                is_child(object) && wanted.iter().any(|class| object.parent(class).is_none())
            });
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Relate Objects"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    /// Only objects are touched - no scratch image needed.
    fn scratch_is_workspace(&self) -> bool {
        false
    }
}

impl RelateObjects {
    /// The parent `child` belongs to among the objects in `grid`, if any.
    fn parent_of(
        &self,
        child_id: &ObjectId,
        child: &Object,
        grid: &BboxGrid,
        cache: &GlobalPipelineCache,
    ) -> Option<ObjectId> {
        let mut candidates: Vec<(ObjectId, &Object)> = grid
            .candidates(child.bbox)
            .into_iter()
            .filter(|id| id != child_id)
            .filter_map(|id| cache.object_cache.get(&id).map(|parent| (id, parent)))
            .collect();
        // Lowest id first: the tie-break.
        candidates.sort_by(|(a, _), (b, _)| a.cmp(b));
        match self.rule {
            RelateRule::CentreInside => {
                let (x, y) = child.get_centroid();
                let (x, y) = (x.round() as u32, y.round() as u32);
                candidates
                    .into_iter()
                    .find(|(_, parent)| parent.is_part_of(x, y))
                    .map(|(id, _)| id)
            }
            RelateRule::LargestOverlap => {
                let mut best: Option<(ObjectId, usize)> = None;
                for (id, parent) in candidates {
                    let area = child.overlap_area(parent);
                    if area > 0 && best.as_ref().is_none_or(|(_, best_area)| area > *best_area) {
                        best = Some((id, area));
                    }
                }
                best.map(|(id, _)| id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use kornia_image::ImageSize;

    const CELL: ObjectClass = ObjectClass::Global(1);
    const SPOT: ObjectClass = ObjectClass::Global(2);
    const REGION: ObjectClass = ObjectClass::Global(3);

    fn filled(id: u128, bbox: [u32; 4], class: ObjectClass) -> Object {
        let w = bbox[2] - bbox[0] + 1;
        let h = bbox[3] - bbox[1] + 1;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: bitvec![u64, Lsb0; 1; (w * h) as usize],
            area: (w * h) as usize,
            ..Default::default()
        });
        object.add_object_class(class, Scope::default()).unwrap();
        object
    }

    fn cache(objects: Vec<Object>) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn step(rule: RelateRule) -> RelateObjects {
        RelateObjects {
            child_classes: vec![SPOT],
            parent_classes: vec![CELL],
            rule,
            remove_without_parent: false,
        }
    }

    fn run(step: &RelateObjects, cache: &mut GlobalPipelineCache) -> Result<(), InternalErrors> {
        let mut meta = PipelineImageMeta::default();
        meta.full_image_width = ImageSize {
            width: 100,
            height: 100,
        };
        let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
        step.execute(&mut ctx, cache)
    }

    fn parent(cache: &GlobalPipelineCache, id: u128, class: ObjectClass) -> Option<u128> {
        cache
            .object_cache
            .get(&ObjectId(id))
            .unwrap()
            .parent(&class)
            .map(|p| p.0)
    }

    #[test]
    fn a_spot_belongs_to_the_cell_its_centre_is_in() {
        let mut cache = cache(vec![
            filled(1, [0, 0, 9, 9], CELL),
            filled(2, [20, 0, 29, 9], CELL),
            filled(10, [3, 3, 4, 4], SPOT),
            filled(11, [22, 5, 23, 6], SPOT),
            filled(12, [50, 50, 51, 51], SPOT),
        ]);
        run(&step(RelateRule::CentreInside), &mut cache).unwrap();
        assert_eq!(parent(&cache, 10, CELL), Some(1));
        assert_eq!(parent(&cache, 11, CELL), Some(2));
        assert_eq!(parent(&cache, 12, CELL), None, "outside every cell");
        assert!(
            cache
                .object_cache
                .get(&ObjectId(1))
                .unwrap()
                .parents()
                .is_empty(),
            "the cells get no parent"
        );
    }

    /// A spot on the border of two cells: its centre decides, or the
    /// bigger overlap.
    #[test]
    fn a_spot_on_a_border_gets_one_cell() {
        let objects = || {
            vec![
                filled(1, [0, 0, 9, 9], CELL),
                filled(2, [10, 0, 19, 9], CELL),
                // 3 columns in cell 1, 2 in cell 2; centre column 9 (cell 1).
                filled(10, [7, 4, 11, 5], SPOT),
            ]
        };
        let mut by_centre = cache(objects());
        run(&step(RelateRule::CentreInside), &mut by_centre).unwrap();
        assert_eq!(parent(&by_centre, 10, CELL), Some(1));

        let mut by_overlap = cache(objects());
        run(&step(RelateRule::LargestOverlap), &mut by_overlap).unwrap();
        assert_eq!(parent(&by_overlap, 10, CELL), Some(1));
    }

    #[test]
    fn equal_overlaps_go_to_the_lower_id() {
        let mut cache = cache(vec![
            filled(2, [10, 0, 19, 9], CELL),
            filled(1, [0, 0, 9, 9], CELL),
            // 2 columns in each cell.
            filled(10, [8, 4, 11, 5], SPOT),
        ]);
        run(&step(RelateRule::LargestOverlap), &mut cache).unwrap();
        assert_eq!(parent(&cache, 10, CELL), Some(1));
    }

    /// A crescent's centre of mass lies outside it - in the parent's hole.
    #[test]
    fn the_centre_rule_uses_the_centre_of_mass_not_the_box() {
        // An L: a full first column and a full last row of a 5×5 box. Its
        // box centre is (2, 2), its centre of mass (1.1, 2.9) → (1, 3).
        let mut mask = bitvec![u64, Lsb0; 0; 25];
        for i in 0..5 {
            mask.set(i * 5, true);
            mask.set(20 + i, true);
        }
        let l_shape = {
            let mut object = Object::new(ObjectInit {
                id: ObjectId(10),
                bbox: [0, 0, 4, 4],
                mask_data: mask,
                area: 9,
                ..Default::default()
            });
            object.add_object_class(SPOT, Scope::default()).unwrap();
            object
        };
        let mut cache = cache(vec![
            filled(1, [2, 2, 2, 2], CELL),
            filled(2, [1, 3, 1, 3], CELL),
            l_shape,
        ]);
        run(&step(RelateRule::CentreInside), &mut cache).unwrap();
        assert_eq!(parent(&cache, 10, CELL), Some(2));
    }

    #[test]
    fn a_child_gets_one_parent_of_each_class() {
        let mut cache = cache(vec![
            filled(1, [0, 0, 9, 9], CELL),
            filled(2, [0, 0, 49, 49], REGION),
            filled(10, [3, 3, 4, 4], SPOT),
        ]);
        let step = RelateObjects {
            parent_classes: vec![CELL, REGION],
            ..step(RelateRule::CentreInside)
        };
        run(&step, &mut cache).unwrap();
        assert_eq!(parent(&cache, 10, CELL), Some(1));
        assert_eq!(parent(&cache, 10, REGION), Some(2));
    }

    /// Relate owns its parent class: an earlier link of that class is
    /// replaced, or dropped when nothing matches; other classes stay.
    #[test]
    fn earlier_links_of_the_parent_class_are_replaced() {
        let mut spot = filled(10, [50, 50, 51, 51], SPOT);
        spot.set_parent(CELL, Some(ObjectId(1)));
        spot.set_parent(REGION, Some(ObjectId(7)));
        let mut cache = cache(vec![filled(1, [0, 0, 9, 9], CELL), spot]);
        run(&step(RelateRule::CentreInside), &mut cache).unwrap();
        assert_eq!(parent(&cache, 10, CELL), None);
        assert_eq!(parent(&cache, 10, REGION), Some(7), "not this step's");
    }

    #[test]
    fn children_without_a_parent_can_be_removed() {
        let mut cache = cache(vec![
            filled(1, [0, 0, 9, 9], CELL),
            filled(10, [3, 3, 4, 4], SPOT),
            filled(11, [50, 50, 51, 51], SPOT),
        ]);
        let step = RelateObjects {
            remove_without_parent: true,
            ..step(RelateRule::CentreInside)
        };
        run(&step, &mut cache).unwrap();
        assert!(cache.object_cache.get(&ObjectId(10)).is_some());
        assert!(cache.object_cache.get(&ObjectId(11)).is_none());
        assert!(
            cache.object_cache.get(&ObjectId(1)).is_some(),
            "parents stay"
        );
    }

    /// An object of both classes (e.g. all objects are "Object") is never
    /// its own parent.
    #[test]
    fn an_object_is_not_its_own_parent() {
        let mut cache = cache(vec![filled(1, [0, 0, 9, 9], CELL)]);
        let step = RelateObjects {
            child_classes: vec![CELL],
            ..step(RelateRule::LargestOverlap)
        };
        run(&step, &mut cache).unwrap();
        assert_eq!(parent(&cache, 1, CELL), None);
    }

    /// The objects of a label image (row by row, 0: background), with ids
    /// `first_id + label`.
    fn from_labels(
        labels: &[u32],
        width: usize,
        first_id: u128,
        class: ObjectClass,
    ) -> Vec<Object> {
        let mut pixels: std::collections::BTreeMap<u32, Vec<(u32, u32)>> = Default::default();
        for (i, &label) in labels.iter().enumerate() {
            if label > 0 {
                pixels
                    .entry(label)
                    .or_default()
                    .push(((i % width) as u32, (i / width) as u32));
            }
        }
        pixels
            .into_iter()
            .map(|(label, pixels)| {
                let x_min = pixels.iter().map(|p| p.0).min().unwrap();
                let y_min = pixels.iter().map(|p| p.1).min().unwrap();
                let x_max = pixels.iter().map(|p| p.0).max().unwrap();
                let y_max = pixels.iter().map(|p| p.1).max().unwrap();
                let w = (x_max - x_min + 1) as usize;
                let mut mask = bitvec![u64, Lsb0; 0; w * (y_max - y_min + 1) as usize];
                for (x, y) in &pixels {
                    mask.set((y - y_min) as usize * w + (x - x_min) as usize, true);
                }
                let mut object = Object::new(ObjectInit {
                    id: ObjectId(first_id + label as u128),
                    bbox: [x_min, y_min, x_max, y_max],
                    mask_data: mask,
                    area: pixels.len(),
                    ..Default::default()
                });
                object.add_object_class(class, Scope::default()).unwrap();
                object
            })
            .collect()
    }

    /// "It overlaps the parent most" gives each child the parent
    /// CellProfiler's RelateObjects gives it (`relate_labels`, its own code:
    /// fixture from `tests/relate_objects_cellprofiler.py`), including ties
    /// and children outside every parent.
    #[test]
    fn largest_overlap_matches_cellprofiler() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            width: usize,
            parents: Vec<u32>,
            children: Vec<u32>,
            parent_of_child: Vec<u32>,
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/relate_objects_cellprofiler.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(cases.len(), 7);
        const CHILD_IDS: u128 = 1000;
        let mut compared = 0;
        for case in cases {
            let mut objects = from_labels(&case.parents, case.width, 0, CELL);
            objects.extend(from_labels(&case.children, case.width, CHILD_IDS, SPOT));
            let mut cache = cache(objects);
            run(&step(RelateRule::LargestOverlap), &mut cache).unwrap();
            for (i, &expected) in case.parent_of_child.iter().enumerate() {
                let child = CHILD_IDS + i as u128 + 1;
                // A label painted over completely is no object.
                let Some(object) = cache.object_cache.get(&ObjectId(child)) else {
                    continue;
                };
                let got = object.parent(&CELL).map_or(0, |p| p.0 as u32);
                assert_eq!(got, expected, "{}: child {}", case.name, i + 1);
                compared += 1;
            }
        }
        assert!(compared > 200, "{compared} children compared");
    }

    /// `overlap_area` counts what `overlaps` would build, for every
    /// parent-child pair of a fixture case (irregular shapes).
    #[test]
    fn overlap_area_is_the_area_of_the_intersection() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/relate_objects_cellprofiler.json"
        );
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let labels =
            |key: &str| -> Vec<u32> { serde_json::from_value(cases[1][key].clone()).unwrap() };
        let parents = from_labels(&labels("parents"), 64, 0, CELL);
        let children = from_labels(&labels("children"), 64, 1000, SPOT);
        let mut overlapping = 0;
        for parent in &parents {
            for child in &children {
                let built = child.overlaps(parent).map_or(0, |o| o.area);
                assert_eq!(child.overlap_area(parent), built);
                assert_eq!(parent.overlap_area(child), built);
                overlapping += usize::from(built > 0);
            }
        }
        assert!(overlapping > 20, "{overlapping} overlapping pairs");
    }

    /// Peak RSS of this process in MB (Linux).
    fn peak_rss_mb() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
            })
            .map_or(0.0, |kb| kb / 1024.0)
    }

    /// Speed and peak RAM: a cell every 40 px on a `BENCH_SIZE`² image
    /// (default 8192: 42,025 cells) with 10 spots (3 × 3 px) each, one rule
    /// (`BENCH_RULE=overlap` or centre) - run in release, one per process:
    /// `BENCH_SIZE=8192 cargo test --release -p evanalyzer_core --lib bench_relate_objects -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_relate_objects() {
        let size: u32 = std::env::var("BENCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let rule = match std::env::var("BENCH_RULE").as_deref() {
            Ok("overlap") => RelateRule::LargestOverlap,
            _ => RelateRule::CentreInside,
        };
        let spacing = 40;
        let mut cache = GlobalPipelineCache::default();
        let mut id = 0u128;
        let mut spots = 0;
        for y in (0..size - spacing + 1).step_by(spacing as usize) {
            for x in (0..size - spacing + 1).step_by(spacing as usize) {
                id += 1;
                let cell = filled(id, [x, y, x + spacing - 1, y + spacing - 1], CELL);
                cache.object_cache.insert(cell.id.clone(), cell);
                for k in 0..10u32 {
                    // Along the diagonal, the last one across the border.
                    let (sx, sy) = (x + 2 + k * 4, y + 2 + k * 4);
                    id += 1;
                    spots += 1;
                    let spot = filled(id, [sx, sy, sx + 2, sy + 2], SPOT);
                    cache.object_cache.insert(spot.id.clone(), spot);
                }
            }
        }
        let rule_name = format!("{rule:?}");
        let before_mb = peak_rss_mb();
        let start = std::time::Instant::now();
        run(&step(rule), &mut cache).unwrap();
        let elapsed = start.elapsed();
        let linked = cache
            .object_cache
            .values()
            .filter(|o| o.parent(&CELL).is_some())
            .count();
        println!(
            "relate objects {size}², {} cells, {spots} spots, {rule_name}: {linked} linked in \
             {:.2} s, peak RSS {:.0} MB (objects before: {before_mb:.0} MB)",
            id as usize - spots,
            elapsed.as_secs_f64(),
            peak_rss_mb()
        );
        assert!(linked >= spots * 9 / 10);
    }

    #[test]
    fn without_classes_it_says_what_is_missing() {
        let mut cache = cache(vec![]);
        let step = RelateObjects {
            parent_classes: vec![ObjectClass::Unset],
            ..step(RelateRule::CentreInside)
        };
        assert!(run(&step, &mut cache).is_err());
    }
}
