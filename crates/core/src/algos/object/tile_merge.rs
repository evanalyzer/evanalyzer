//! Cross-tile object merging: reassembles an object that a tiled analysis
//! run split across an internal tile boundary (e.g. a whole organ/tissue
//! region on a whole-slide image) into one correct object.
//!
//! This is the whole-image algorithm that used to be a bespoke step in
//! `job_executor.rs` (`job::tile_merge::merge_pending_fragments`, driven by
//! per-tile fragment buffering). Now that every object carries its own
//! `source_tile` (see `Object::source_tile`), this can run as a plain
//! `ExecutionScope::WholeImage` `ImageAlgorithm` directly over
//! `cache.object_cache` - once every tile's `run_tile` pass has finished and
//! populated it - instead of `job_executor` needing to pre-detect and buffer
//! edge-touching fragments during the tile loop.
//!
//! Summary: groups every tile-edge-touching object by class/plane, finds
//! which ones from *different* tiles are actually touching across their
//! shared tile seam (a small variant of [`Object::overlaps`]'s windowed mask
//! scan), unions matched fragments via a union-find, and rebuilds each
//! merge-group as one [`Object`] through [`Object::new`] so its geometry
//! (perimeter, ellipse, ...) is recomputed correctly from the real merged
//! shape - removing the consumed fragments from `cache.object_cache` and
//! inserting the rebuilt object in their place. Groups of size 1 (an object
//! that touched a tile edge but had no real cross-tile neighbor) are left
//! exactly as they already sit in the cache - untouched, no fragment ever
//! needed rebuilding.
//!
//! This is a tiled/blocked variant of the classical two-pass connected-
//! component labeling algorithm (Rosenfeld & Pfaltz, 1966): each tile is
//! labeled independently, then object identity is reconciled across tile
//! boundaries via the same union-find idea, just run as one end-of-image
//! batch pass instead of incrementally online.

use crate::ImageTile;
use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::object::{Intensity, Object, ObjectInit};
use crate::pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext};
use crate::spatial_grid::BboxGrid;
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, ObjectClass, ObjectId};
use indexmap::IndexMap;
use macros::CommandsMeta;
use std::collections::HashMap;

/// 4- or 8-connected boundary adjacency, used to decide whether two
/// fragments from different tiles are actually touching. Deliberately its
/// own algo-native type, not `evanalyzer_cfg::settings::project_settings::
/// TileMergeConnectivity` (the project-settings enum) - this module never
/// references project settings or `job_executor` at all; converting from
/// the project-settings type is entirely `job_executor`'s job, at the one
/// call site that actually constructs a `TileMerge`.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Connectivity {
    FourConnected,
    EightConnected,
}

/// System-inserted, not a user-pickable `PipelineCommand` - `job_executor`
/// prepends this to a pipeline's whole-image command list itself, built
/// from the project-level `TileMergeSettings` (converted at that call site,
/// not in this module - see `Connectivity`'s own doc comment), not a
/// per-step setting. Whether to run it at all (`TileMergeSettings::enabled`)
/// is deliberately *not* a field here either - that's `job_executor`'s call
/// to make before it ever constructs a `TileMerge`, not something the
/// algorithm itself should know about. Stays `pub(crate)` rather than
/// `pub`, and deliberately still derives `CommandsMeta` (needed for the
/// `#[cmdsmeta(...)]` field attributes below to be legal at all, since
/// `cmdsmeta` is a derive-helper attribute): the `cfg` crate's build-time
/// generator only scans `pub` items, so `pub(crate)` already keeps this out
/// of the generated command registry regardless of `CommandsMeta` being
/// present.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", next = "object")]
pub(crate) struct TileMerge {
    /// Opt-out deny-list: objects carrying any of these classes are excluded
    /// from tile-edge merging. Empty (the default) means every class is
    /// merge-eligible - a non-expert user turning tile-merge on has no
    /// reason to know about tiles at all and just expects objects to be
    /// detected correctly, not to configure a class list first.
    pub classes_to_not_merge: Vec<ObjectClass>,

    /// 4- or 8-connected boundary adjacency used to decide whether two
    /// fragments from different tiles are actually touching.
    #[cmdsmeta(default = Connectivity::EightConnected)]
    pub connectivity: Connectivity,

    /// Safety cap on how many tile-edge fragments a single merge group (one
    /// real-world object's worth of fragments, before matching) may contain
    /// before the run fails loudly instead of building a pathological merge
    /// - mirrors `ExtractObjects.max_objects_before_fail`. Since every class
    /// is merge-eligible by default, this is the guard against a class that
    /// turns out to match thousands of small tile-edge-touching objects
    /// (add it to `classes_to_not_merge` instead), not just a misconfigured
    /// allow-list.
    pub max_fragments_per_group: u32,
}

impl ImageAlgorithm for TileMerge {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let groups = group_candidates(cache, &self.classes_to_not_merge);

        for (_, group_ids) in groups {
            if group_ids.len() > self.max_fragments_per_group as usize {
                return Err(InternalErrors::TooManyObjects(format!(
                    "Detected {} tile-edge object fragments in one merge group, limit is {}. \
                     Add the offending class to `classes_to_not_merge` if it matches far more \
                     tile-edge-touching objects than expected.",
                    group_ids.len(),
                    self.max_fragments_per_group
                )));
            }

            let grid = BboxGrid::build(&group_ids, cache);
            let local_index: HashMap<ObjectId, usize> = group_ids
                .iter()
                .enumerate()
                .map(|(local, id)| (id.clone(), local))
                .collect();

            let mut uf = UnionFind::new(group_ids.len());
            for (local_a, id_a) in group_ids.iter().enumerate() {
                let Some(object_a) = cache.object_cache.get(id_a) else {
                    continue;
                };
                // Expand the query bbox by one pixel: `BboxGrid`'s own no-false-
                // negative guarantee only covers *true* bbox intersection, but
                // fragments from adjacent (non-overlapping) tiles are at best
                // touching, never truly intersecting.
                let [x0, y0, x1, y1] = object_a.bbox;
                let query = [
                    x0.saturating_sub(1),
                    y0.saturating_sub(1),
                    x1.saturating_add(1),
                    y1.saturating_add(1),
                ];
                for candidate_id in grid.candidates(query) {
                    let Some(&local_b) = local_index.get(&candidate_id) else {
                        continue;
                    };
                    if local_b <= local_a {
                        continue; // dedupe unordered pairs and skip self
                    }
                    let Some(object_b) = cache.object_cache.get(&candidate_id) else {
                        continue;
                    };
                    if object_a.source_tile.offset_x == object_b.source_tile.offset_x
                        && object_a.source_tile.offset_y == object_b.source_tile.offset_y
                    {
                        continue; // same tile - not a cross-tile fragment pair
                    }
                    if fragments_are_adjacent(object_a, object_b, self.connectivity) {
                        uf.union(local_a, local_b);
                    }
                }
            }

            let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
            for local in 0..group_ids.len() {
                let root = uf.find(local);
                by_root.entry(root).or_default().push(local);
            }

            for locals in by_root.into_values() {
                // A group of one already sits correctly in `cache.object_cache` -
                // it touched a tile edge but had no real cross-tile neighbor
                // (e.g. it also happens to touch the full image edge
                // diagonally), so there's nothing to rebuild.
                if locals.len() == 1 {
                    continue;
                }

                let ids: Vec<ObjectId> = locals.iter().map(|&l| group_ids[l].clone()).collect();
                let objects: Vec<Object> = ids
                    .iter()
                    .filter_map(|id| cache.object_cache.get(id).cloned())
                    .collect();
                let object_refs: Vec<&Object> = objects.iter().collect();

                let geometry = union_merge(&object_refs);
                let intensities = merge_intensities(&object_refs, geometry.area);

                let mut object_class = std::collections::HashSet::new();
                for o in &object_refs {
                    object_class.extend(o.object_class.iter().copied());
                }

                let first = object_refs[0];
                let merged = Object::new(ObjectInit {
                    id: ObjectId::next(),
                    segmentation_class: first.segmentation_class,
                    object_class,
                    area: geometry.area,
                    bbox: geometry.bbox,
                    mask_data: geometry.mask_data,
                    touches_edge: geometry.touches_edge,
                    sum_x: geometry.sum_x,
                    sum_y: geometry.sum_y,
                    sum_x2: geometry.sum_x2,
                    sum_y2: geometry.sum_y2,
                    sum_xy: geometry.sum_xy,
                    intensities,
                    plane: first.plane,
                    ..Default::default()
                });

                for id in &ids {
                    cache.object_cache.remove(id);
                }
                cache.object_cache.insert(merged.id.clone(), merged);
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Tile Merge"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "rosenfeld1966sequential",
            title: "Sequential Operations in Digital Picture Processing",
            authors: &["Azriel Rosenfeld", "John L. Pfaltz"],
            year: 1966,
            container: Some("Journal of the ACM"),
            doi: Some("10.1145/321356.321357"),
            url: Some("https://doi.org/10.1145/321356.321357"),
            pages: Some("471-494"),
        })
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

/// True if `bbox` (absolute image coordinates, inclusive) reaches any edge of
/// `tile` (also absolute coordinates) - i.e. the object may have been cut off
/// by the tile boundary rather than genuinely ending there. `job_executor`
/// also uses this directly to decide what to buffer for the whole-image
/// phase in the first place, before this algorithm ever runs.
pub(crate) fn touches_tile_edge(bbox: [u32; 4], tile: &ImageTile) -> bool {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_max = tile.offset_x + tile.width.saturating_sub(1);
    let tile_y_max = tile.offset_y + tile.height.saturating_sub(1);
    x_min as usize == tile.offset_x
        || y_min as usize == tile.offset_y
        || x_max as usize == tile_x_max
        || y_max as usize == tile_y_max
}

/// Groups every object in `cache.object_cache` that touches its own
/// `source_tile`'s edge (and isn't in `classes_to_not_merge`) by its full
/// class set (sorted/deduped) plus plane - the merge-candidate pool.
/// Objects fully interior to their tile can never have a real cross-tile
/// neighbor, so they're never even candidates.
fn group_candidates(
    cache: &GlobalPipelineCache,
    classes_to_not_merge: &[ObjectClass],
) -> IndexMap<(Vec<ObjectClass>, crate::ImagePlane), Vec<ObjectId>> {
    let mut groups: IndexMap<(Vec<ObjectClass>, crate::ImagePlane), Vec<ObjectId>> =
        IndexMap::new();
    for (id, object) in cache.object_cache.iter() {
        if !touches_tile_edge(object.bbox, &object.source_tile) {
            continue;
        }
        let denied = object
            .object_class
            .iter()
            .any(|c| classes_to_not_merge.contains(c));
        if denied {
            continue;
        }
        let mut classes: Vec<_> = object.object_class.iter().copied().collect();
        classes.sort();
        classes.dedup();
        groups
            .entry((classes, object.plane))
            .or_default()
            .push(id.clone());
    }
    groups
}

/// Plain union-find (disjoint-set) over `0..n`, path-compressed on `find`,
/// unranked union - groups are small (one real object's worth of tile-edge
/// fragments) so the extra bookkeeping of union-by-rank isn't worth it.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// True if `a` and `b` have at least one foreground pixel within `connectivity`
/// of each other - "touching", not "overlapping" (fragments from different,
/// non-overlapping tiles never truly overlap). Scans only the small window
/// where such a pair could exist (each bbox dilated by one pixel, intersected),
/// not the full union bbox - the window is bounded by the length of the
/// shared tile seam, not the fragments' own extent, so this stays cheap even
/// for whole-organ-sized fragments.
fn fragments_are_adjacent(a: &Object, b: &Object, connectivity: Connectivity) -> bool {
    let [ax0, ay0, ax1, ay1] = a.bbox;
    let [bx0, by0, bx1, by1] = b.bbox;

    let x0 = ax0.max(bx0).saturating_sub(1);
    let x1 = ax1.min(bx1).saturating_add(1);
    let y0 = ay0.max(by0).saturating_sub(1);
    let y1 = ay1.min(by1).saturating_add(1);
    if x0 > x1 || y0 > y1 {
        return false;
    }

    const FOUR_CONNECTED: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    const EIGHT_CONNECTED: [(i64, i64); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let offsets: &[(i64, i64)] = match connectivity {
        Connectivity::FourConnected => &FOUR_CONNECTED,
        Connectivity::EightConnected => &EIGHT_CONNECTED,
    };

    for y in y0..=y1 {
        for x in x0..=x1 {
            if !a.is_part_of(x, y) {
                continue;
            }
            for (dx, dy) in offsets {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                if b.is_part_of(nx as u32, ny as u32) {
                    return true;
                }
            }
        }
    }
    false
}

/// Result of unioning a merge-group's fragment masks together.
struct UnionGeometry {
    bbox: [u32; 4],
    mask_data: BitVec<u64, Lsb0>,
    area: usize,
    touches_edge: bool,
    sum_x: u64,
    sum_y: u64,
    sum_x2: u64,
    sum_y2: u64,
    sum_xy: u64,
}

/// Unions `objects`' masks into one, offsetting each into the group's shared
/// bbox. `O(sum of fragment areas)`, not `O(union bbox area)` - deliberately
/// not implemented via `Object::combine_geometry`'s `BooleanOp::Or` (which
/// rasterizes the *entire* union bbox), since the whole point of tile-merge is
/// whole-organ-sized objects whose union bbox can span most of a whole-slide
/// image while the fragments themselves stay sparse.
fn union_merge(objects: &[&Object]) -> UnionGeometry {
    let mut x_min = u32::MAX;
    let mut y_min = u32::MAX;
    let mut x_max = 0u32;
    let mut y_max = 0u32;
    for o in objects {
        let [ox0, oy0, ox1, oy1] = o.bbox;
        x_min = x_min.min(ox0);
        y_min = y_min.min(oy0);
        x_max = x_max.max(ox1);
        y_max = y_max.max(oy1);
    }

    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, width * height);
    let mut area = 0usize;
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    let mut sum_x2 = 0u64;
    let mut sum_y2 = 0u64;
    let mut sum_xy = 0u64;

    for o in objects {
        let [ox0, oy0, ox1, oy1] = o.bbox;
        let ow = (ox1 - ox0 + 1) as usize;
        let oh = (oy1 - oy0 + 1) as usize;
        for ly in 0..oh {
            for lx in 0..ow {
                if !o.mask_data.get(ly * ow + lx).map(|b| *b).unwrap_or(false) {
                    continue;
                }
                let gx = ox0 + lx as u32;
                let gy = oy0 + ly as u32;
                let out_idx = (gy - y_min) as usize * width + (gx - x_min) as usize;
                // Fragments come from non-overlapping tiles so this shouldn't
                // double-count in practice, but guard anyway: `mask_data`'s
                // `Index` doesn't panic-check bounds the way `.get()` does,
                // and re-setting an already-true bit is harmless either way.
                if !mask[out_idx] {
                    mask.set(out_idx, true);
                    area += 1;
                    let (gx64, gy64) = (gx as u64, gy as u64);
                    sum_x += gx64;
                    sum_y += gy64;
                    sum_x2 += gx64 * gx64;
                    sum_y2 += gy64 * gy64;
                    sum_xy += gx64 * gy64;
                }
            }
        }
    }

    UnionGeometry {
        bbox: [x_min, y_min, x_max, y_max],
        mask_data: mask,
        area,
        touches_edge: objects.iter().any(|o| o.touches_edge),
        sum_x,
        sum_y,
        sum_x2,
        sum_y2,
        sum_xy,
    }
}

/// Additively merges a merge-group's per-channel intensities: `sum`/`min`/`max`
/// combine directly, `avg` is re-derived from the merged `sum`/`area` (the
/// per-fragment averages themselves can't just be averaged - fragments don't
/// necessarily have equal pixel counts).
fn merge_intensities(objects: &[&Object], merged_area: usize) -> IndexMap<i32, Intensity> {
    let mut merged: IndexMap<i32, Intensity> = IndexMap::new();
    for o in objects {
        for (&channel, intensity) in &o.intensities {
            let acc = merged.entry(channel).or_insert_with(|| Intensity {
                sum_intensity: 0.0,
                min_intensity: f32::MAX,
                max_intensity: f32::MIN,
                avg_intensity: 0.0,
                pixel_values: Vec::new(),
            });
            acc.sum_intensity += intensity.sum_intensity;
            acc.min_intensity = acc.min_intensity.min(intensity.min_intensity);
            acc.max_intensity = acc.max_intensity.max(intensity.max_intensity);
        }
    }
    let n = merged_area.max(1) as f64;
    for intensity in merged.values_mut() {
        intensity.avg_intensity = (intensity.sum_intensity / n) as f32;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImagePlane;
    use crate::object::ObjectInit;
    use evanalyzer_cfg::core_types::ObjectClass;

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    /// A fully-filled rectangular object `[x0,y0,x1,y1]` (inclusive), carrying
    /// `class`/`plane`/`source_tile`, with `touches_edge` set explicitly (as
    /// the real extraction path would have already computed it against the
    /// *full* image, not the tile it was cut from).
    fn rect_object(
        id: u128,
        bbox: [u32; 4],
        class: ObjectClass,
        plane: ImagePlane,
        source_tile: ImageTile,
        touches_edge: bool,
    ) -> Object {
        let [x0, y0, x1, y1] = bbox;
        let w = (x1 - x0 + 1) as usize;
        let h = (y1 - y0 + 1) as usize;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: BitVec::<u64, Lsb0>::repeat(true, w * h),
            area: w * h,
            touches_edge,
            plane,
            source_tile,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    fn tile_merge(connectivity: Connectivity) -> TileMerge {
        TileMerge {
            classes_to_not_merge: Vec::new(),
            connectivity,
            max_fragments_per_group: 100,
        }
    }

    fn cache_with(objects: Vec<Object>) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn dummy_ctx() -> crate::pipeline::pipeline_context::PipelineContext {
        crate::pipeline::pipeline_context::PipelineContext::new_from_image_test(
            kornia_image::Image::<f32, 1, kornia_tensor::CpuAllocator>::new(
                kornia_image::ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0.0f32],
                kornia_tensor::CpuAllocator,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn run(cache: &mut GlobalPipelineCache, algo: TileMerge) {
        algo.execute(&mut dummy_ctx(), cache).unwrap();
    }

    #[test]
    fn a_single_fragment_with_no_neighbor_is_left_unchanged() {
        let object = rect_object(
            1,
            [0, 0, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 10, 10),
            false,
        );
        let mut cache = cache_with(vec![object]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(cache.object_cache.len(), 1);
        let unchanged = cache.object_cache.get(&ObjectId(1)).unwrap();
        assert_eq!(unchanged.bbox, [0, 0, 9, 9]);
        assert_eq!(unchanged.area, 100);
    }

    /// Two fragments from horizontally-adjacent tiles, sharing a full-height
    /// seam (A ends at x=9, B starts at x=10) - the common case: an organ cut
    /// straight down the middle by a tile boundary.
    #[test]
    fn fragments_sharing_a_straight_tile_seam_merge_into_one_object() {
        let a = rect_object(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 10, 20),
            false,
        );
        let b = rect_object(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(10, 0, 10, 20),
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(
            cache.object_cache.len(),
            1,
            "the two fragments must merge into one object"
        );
        let merged = cache.object_cache.values().next().unwrap();
        assert_eq!(merged.bbox, [0, 0, 19, 19]);
        assert_eq!(merged.area, 400);
        assert!(
            !merged.touches_edge,
            "neither fragment touched the full image edge"
        );
        assert_ne!(
            merged.id,
            ObjectId(1),
            "the merged object must get a brand-new id"
        );
        assert_ne!(merged.id, ObjectId(2));
    }

    /// Same seam, but the two fragments come from the *same* tile (a
    /// pathological/synthetic case) - must never merge, since two objects
    /// extracted from one tile are always genuinely separate objects.
    #[test]
    fn fragments_from_the_same_tile_never_merge_even_if_touching() {
        let same_tile = tile(0, 0, 20, 20);
        let a = rect_object(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            same_tile.clone(),
            false,
        );
        let b = rect_object(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            same_tile,
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(
            cache.object_cache.len(),
            2,
            "same-tile fragments must stay separate objects"
        );
    }

    /// Fragments touching at exactly one pixel (a single-pixel corner touch,
    /// not a full seam) must still be found and merged.
    #[test]
    fn fragments_touching_at_a_single_pixel_merge() {
        let a = rect_object(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 5, 5),
            false,
        );
        let b = rect_object(
            2,
            [5, 5, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(5, 5, 5, 5),
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(cache.object_cache.len(), 1);
        assert_eq!(cache.object_cache.values().next().unwrap().area, 50);
    }

    /// The same single-pixel diagonal touch must NOT merge under
    /// 4-connectivity - a purely diagonal touch has no shared edge.
    #[test]
    fn diagonal_only_touch_does_not_merge_under_four_connectivity() {
        let a = rect_object(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 5, 5),
            false,
        );
        let b = rect_object(
            2,
            [5, 5, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(5, 5, 5, 5),
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::FourConnected));

        assert_eq!(
            cache.object_cache.len(),
            2,
            "a purely diagonal touch must not merge under 4-connectivity"
        );
    }

    /// Four fragments, one from each tile around a 4-tile corner, all meeting
    /// at a single point - the classic case that needs the transitive
    /// union-find, not just pairwise matching: TL-TR and TL-BL touch directly
    /// (shared edges), TL-BR only diagonally, but all four must end up in one
    /// group under 8-connectivity.
    #[test]
    fn four_tiles_meeting_at_one_corner_all_merge_into_one_object() {
        // Tiles: TL=[0,0]-[4,4], TR=[5,0]-[9,4], BL=[0,5]-[4,9], BR=[5,5]-[9,9].
        let class = ObjectClass::Valid(1);
        let plane = ImagePlane::default();
        let tl = rect_object(1, [0, 0, 4, 4], class, plane, tile(0, 0, 5, 5), false);
        let tr = rect_object(2, [5, 0, 9, 4], class, plane, tile(5, 0, 5, 5), false);
        let bl = rect_object(3, [0, 5, 4, 9], class, plane, tile(0, 5, 5, 5), false);
        let br = rect_object(4, [5, 5, 9, 9], class, plane, tile(5, 5, 5, 5), false);
        let mut cache = cache_with(vec![tl, tr, bl, br]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(
            cache.object_cache.len(),
            1,
            "all four corner fragments must transitively merge into one object"
        );
        let merged = cache.object_cache.values().next().unwrap();
        assert_eq!(merged.area, 100);
        assert_eq!(merged.bbox, [0, 0, 9, 9]);
    }

    /// Fragments far apart (not touching at all) must never merge.
    #[test]
    fn far_apart_fragments_do_not_merge() {
        let a = rect_object(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 5, 5),
            false,
        );
        let b = rect_object(
            2,
            [1000, 1000, 1004, 1004],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(1000, 1000, 5, 5),
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(cache.object_cache.len(), 2);
    }

    /// Every class merges by default - `classes_to_not_merge` is an opt-out
    /// deny-list, so a fragment carrying a denied class is never even
    /// considered a candidate.
    #[test]
    fn fragments_with_a_denied_class_are_left_unmerged_and_unremoved() {
        let object = rect_object(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(99),
            ImagePlane::default(),
            tile(0, 0, 5, 5),
            false,
        );
        let mut cache = cache_with(vec![object]);
        let mut denylisted = tile_merge(Connectivity::EightConnected);
        denylisted.classes_to_not_merge = vec![ObjectClass::Valid(99)];

        run(&mut cache, denylisted);

        assert_eq!(
            cache.object_cache.len(),
            1,
            "a denied-class object must be left exactly as it was, not dropped"
        );
    }

    /// A class that isn't on `classes_to_not_merge` merges without needing
    /// any explicit opt-in - the whole point of the deny-list default.
    #[test]
    fn a_class_not_on_the_deny_list_merges_by_default() {
        let a = rect_object(
            1,
            [0, 0, 4, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(0, 0, 5, 10),
            false,
        );
        let b = rect_object(
            2,
            [5, 0, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            tile(5, 0, 5, 10),
            false,
        );
        let mut cache = cache_with(vec![a, b]);
        let mut denylisted = tile_merge(Connectivity::EightConnected);
        denylisted.classes_to_not_merge = vec![ObjectClass::Valid(2)]; // unrelated class

        run(&mut cache, denylisted);

        assert_eq!(
            cache.object_cache.len(),
            1,
            "a class absent from classes_to_not_merge must still merge"
        );
    }

    /// Fragments on different `ImagePlane`s never merge, even if their masks
    /// touch - they're different channels/z-slices/timepoints, not pieces of
    /// the same real object.
    #[test]
    fn fragments_on_different_planes_do_not_merge() {
        let a = rect_object(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane { z: 0, c: 0, t: 0 },
            tile(0, 0, 10, 20),
            false,
        );
        let b = rect_object(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane { z: 0, c: 1, t: 0 },
            tile(10, 0, 10, 20),
            false,
        );
        let mut cache = cache_with(vec![a, b]);

        run(&mut cache, tile_merge(Connectivity::EightConnected));

        assert_eq!(cache.object_cache.len(), 2);
    }

    #[test]
    fn a_merge_group_over_the_fragment_cap_fails_loudly() {
        let mut objects = Vec::new();
        for i in 0..5u128 {
            let x0 = (i as u32) * 10;
            objects.push(rect_object(
                i + 1,
                [x0, 0, x0 + 4, 4],
                ObjectClass::Valid(1),
                ImagePlane::default(),
                tile(x0 as usize, 0, 5, 5),
                false,
            ));
        }
        let mut cache = cache_with(objects);
        let mut too_strict = tile_merge(Connectivity::EightConnected);
        too_strict.max_fragments_per_group = 4;

        let result = too_strict.execute(&mut dummy_ctx(), &mut cache);
        assert!(matches!(result, Err(InternalErrors::TooManyObjects(_))));
    }

    #[test]
    fn touches_tile_edge_detects_every_side() {
        let t = tile(100, 200, 50, 60);
        assert!(touches_tile_edge([100, 210, 110, 220], &t), "left edge");
        assert!(touches_tile_edge([120, 200, 130, 210], &t), "top edge");
        assert!(touches_tile_edge([140, 210, 149, 220], &t), "right edge");
        assert!(touches_tile_edge([120, 250, 130, 259], &t), "bottom edge");
        assert!(
            !touches_tile_edge([110, 210, 120, 220], &t),
            "fully interior bbox must not report a touch"
        );
    }
}
