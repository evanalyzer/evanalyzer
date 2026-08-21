//! # Spatial Grid Index
//!
//! Provides a uniform-grid spatial index over object bounding boxes, used to prune
//! pairwise-overlap scans (`Colocalization`, `ClassifyObjects`'s `overlapping_with`) from
//! O(N*M) down to roughly O(N+M) for the sparse, small-object layouts typical of bioimage
//! ROIs.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::pipeline::pipeline_cache::PipelineCache;
use evanalyzer_cfg::core_types::ObjectId;
use std::collections::{HashMap, HashSet};

/// Uniform grid over a set of objects' bounding boxes, keyed by each object's own
/// `ObjectId`. Used to cheaply narrow "which objects could possibly overlap this bbox"
/// down to a small candidate list before paying for the exact `Object::overlaps` mask
/// intersection.
///
/// Cell size is derived from the *indexed* set's own average object extent, so a query
/// against it returns a small, roughly constant-size candidate list regardless of how
/// many objects are indexed - the property that turns an O(N*M) full scan into an
/// O(N+M) one. Objects are inserted into every cell their bbox spans, and a query
/// returns every candidate whose bbox spans a shared cell with the query bbox. This is a
/// deliberate superset of the true overlapping set - false positives are expected and
/// cheap to filter out with the real `Object::overlaps` check the caller already
/// performs - but it can never produce a false negative: two intersecting bboxes always
/// share at least one grid cell under the same partition (the cell containing any point
/// in their intersection), regardless of where the grid lines fall.
pub(crate) struct BboxGrid {
    cell_size: u32,
    cells: HashMap<(i64, i64), Vec<ObjectId>>,
}

impl BboxGrid {
    pub(crate) fn build(ids: &[ObjectId], cache: &PipelineCache) -> Self {
        let mut sum_extent: u64 = 0;
        let mut count: u64 = 0;
        for id in ids {
            if let Some(object) = cache.object_cache.get(id) {
                let [x_min, y_min, x_max, y_max] = object.bbox;
                sum_extent += (x_max - x_min + 1) as u64 + (y_max - y_min + 1) as u64;
                count += 1;
            }
        }
        // Average object extent (width+height, averaged) as the cell size: most objects
        // then span a small, roughly constant number of cells regardless of how many
        // objects are indexed. Falls back to 1 for an empty set (never divides by zero,
        // since bbox extents are always >= 1).
        let cell_size = if count == 0 {
            1
        } else {
            ((sum_extent / (count * 2)).max(1)) as u32
        };

        let mut cells: HashMap<(i64, i64), Vec<ObjectId>> = HashMap::new();
        for id in ids {
            if let Some(object) = cache.object_cache.get(id) {
                for cell in Self::cell_range(object.bbox, cell_size) {
                    cells.entry(cell).or_default().push(id.clone());
                }
            }
        }

        Self { cell_size, cells }
    }

    fn cell_range(bbox: [u32; 4], cell_size: u32) -> impl Iterator<Item = (i64, i64)> {
        let [x_min, y_min, x_max, y_max] = bbox;
        let cx_min = (x_min / cell_size) as i64;
        let cx_max = (x_max / cell_size) as i64;
        let cy_min = (y_min / cell_size) as i64;
        let cy_max = (y_max / cell_size) as i64;
        (cx_min..=cx_max).flat_map(move |cx| (cy_min..=cy_max).map(move |cy| (cx, cy)))
    }

    /// Every indexed id whose bbox spans a grid cell in common with `bbox` - a
    /// deduplicated superset of the ids whose bbox truly intersects `bbox`. Never omits a
    /// true intersection (see struct docs); the caller is expected to filter the result
    /// with an exact overlap check.
    pub(crate) fn candidates(&self, bbox: [u32; 4]) -> Vec<ObjectId> {
        let mut seen: HashSet<&ObjectId> = HashSet::new();
        for cell in Self::cell_range(bbox, self.cell_size) {
            if let Some(ids) = self.cells.get(&cell) {
                seen.extend(ids.iter());
            }
        }
        seen.into_iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImagePlane, object::{Object, ObjectInit}};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectClass;

    fn make_object(id: u128, bbox: [u32; 4]) -> Object {
        let [x_min, y_min, x_max, y_max] = bbox;
        let w = (x_max - x_min + 1) as usize;
        let h = (y_max - y_min + 1) as usize;
        let area = w * h;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data,
            area,
            plane: ImagePlane::default(),
            ..Default::default()
        });
        object.add_object_class(ObjectClass::Valid(1));
        object
    }

    fn cache_with(objects: Vec<Object>) -> (PipelineCache, Vec<ObjectId>) {
        let mut cache = PipelineCache::default();
        let mut ids = Vec::new();
        for object in objects {
            ids.push(object.id.clone());
            cache.object_cache.insert(object.id.clone(), object);
        }
        (cache, ids)
    }

    /// Naive reference: two bboxes truly intersect (inclusive bounds) iff their ranges
    /// overlap on both axes.
    fn bboxes_intersect(a: [u32; 4], b: [u32; 4]) -> bool {
        a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
    }

    #[test]
    fn empty_index_returns_no_candidates() {
        let (cache, _) = cache_with(vec![]);
        let grid = BboxGrid::build(&[], &cache);
        assert!(grid.candidates([0, 0, 10, 10]).is_empty());
    }

    #[test]
    fn far_apart_objects_do_not_appear_as_candidates_for_each_other() {
        let a = make_object(1, [0, 0, 4, 4]);
        let b = make_object(2, [1000, 1000, 1004, 1004]);
        let (cache, ids) = cache_with(vec![a, b]);
        let grid = BboxGrid::build(&ids, &cache);

        assert_eq!(grid.candidates([0, 0, 4, 4]), vec![ObjectId(1)]);
        assert_eq!(grid.candidates([1000, 1000, 1004, 1004]), vec![ObjectId(2)]);
    }

    #[test]
    fn objects_touching_at_a_single_corner_pixel_are_found() {
        // A=[0,0,4,4], B=[4,4,8,8] - share exactly the pixel (4,4).
        let a = make_object(1, [0, 0, 4, 4]);
        let b = make_object(2, [4, 4, 8, 8]);
        let (cache, ids) = cache_with(vec![a, b]);
        let grid = BboxGrid::build(&ids, &cache);

        assert!(grid.candidates([0, 0, 4, 4]).contains(&ObjectId(2)));
        assert!(grid.candidates([4, 4, 8, 8]).contains(&ObjectId(1)));
    }

    #[test]
    fn adjacent_but_not_touching_objects_are_not_required_to_be_pruned_but_are_correctly_excludable()
    {
        // A=[0,0,3,3], B=[4,0,7,3] - share no pixel (A ends at x=3, B starts at x=4).
        let a = make_object(1, [0, 0, 3, 3]);
        let b = make_object(2, [4, 0, 7, 3]);
        assert!(!bboxes_intersect(a.bbox, b.bbox));
    }

    #[test]
    fn a_huge_outlier_object_is_still_found_by_a_small_query_bbox() {
        // Wildly different sizes in the same bucket: 50 tiny 2x2 objects plus one
        // 500x500 giant. The average-extent cell size is dominated by the giant, but the
        // giant must still be found by a query that only partially overlaps it, since it
        // spans every cell its own bbox covers.
        let mut objects = Vec::new();
        for i in 0..50u128 {
            objects.push(make_object(i, [i as u32 * 3, 0, i as u32 * 3 + 1, 1]));
        }
        objects.push(make_object(1000, [10, 10, 509, 509]));
        let (cache, ids) = cache_with(objects);
        let grid = BboxGrid::build(&ids, &cache);

        // Query bbox only touches the giant's bottom-right corner.
        let candidates = grid.candidates([505, 505, 520, 520]);
        assert!(
            candidates.contains(&ObjectId(1000)),
            "a query touching only a small corner of a huge indexed object must still find it"
        );
    }

    #[test]
    fn identical_and_nested_bboxes_are_found() {
        let outer = make_object(1, [0, 0, 100, 100]);
        let inner = make_object(2, [40, 40, 60, 60]);
        let duplicate = make_object(3, [0, 0, 100, 100]);
        let (cache, ids) = cache_with(vec![outer, inner, duplicate]);
        let grid = BboxGrid::build(&ids, &cache);

        let candidates = grid.candidates([40, 40, 60, 60]);
        assert!(candidates.contains(&ObjectId(1)));
        assert!(candidates.contains(&ObjectId(2)));
        assert!(candidates.contains(&ObjectId(3)));
    }

    /// Property-style fuzz check: for many randomized layouts of objects with widely
    /// varying sizes and positions (including dense clustering, which forces heavy
    /// per-cell bucketing), every pair whose bboxes truly intersect (per the naive
    /// `bboxes_intersect` reference) must be mutually discoverable via
    /// `BboxGrid::candidates` - i.e. the grid must never produce a false negative. This is
    /// the correctness guardrail for using the grid to prune `Colocalization` and
    /// `ClassifyObjects`'s overlap scans: both call sites rely on the grid never hiding a
    /// true candidate from the exact-overlap check that follows it.
    #[test]
    fn randomized_layouts_never_miss_a_true_bbox_intersection() {
        struct Rng(u32);
        impl Rng {
            fn next(&mut self) -> u32 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 17;
                self.0 ^= self.0 << 5;
                self.0
            }
            fn range(&mut self, n: u32) -> u32 {
                self.next() % n
            }
        }

        for seed in 1..20u32 {
            let mut rng = Rng(seed);
            let mut objects = Vec::new();
            let mut bboxes: Vec<(u128, [u32; 4])> = Vec::new();
            for i in 0..80u128 {
                // 1..40 px extents and 0..80 positions - deliberately overlapping ranges
                // so a good fraction of pairs truly intersect, including dense clusters.
                let w = 1 + rng.range(40);
                let h = 1 + rng.range(40);
                let x = rng.range(80);
                let y = rng.range(80);
                let bbox = [x, y, x + w, y + h];
                bboxes.push((i, bbox));
                objects.push(make_object(i, bbox));
            }
            let (cache, ids) = cache_with(objects);
            let grid = BboxGrid::build(&ids, &cache);

            for &(id_a, bbox_a) in &bboxes {
                for &(id_b, bbox_b) in &bboxes {
                    if id_a == id_b {
                        continue;
                    }
                    if bboxes_intersect(bbox_a, bbox_b) {
                        let candidates = grid.candidates(bbox_a);
                        assert!(
                            candidates.contains(&ObjectId(id_b)),
                            "seed {seed}: bbox {bbox_a:?} (id {id_a}) truly intersects \
                             {bbox_b:?} (id {id_b}) but the grid didn't return it as a candidate"
                        );
                    }
                }
            }
        }
    }
}
