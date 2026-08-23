//! Cross-tile object merging: reassembles an object that a tiled analysis
//! run split across an internal tile boundary (e.g. a whole organ/tissue
//! region on a whole-slide image) into one correct object.
//!
//! See `docs/tile_merge_plan.md` for the full design. Summary: `job_executor`
//! buffers per-tile object fragments that touch their own tile's edge (but
//! not the full image's edge) instead of exporting them immediately; once
//! every tile of an image has finished, [`merge_pending_fragments`] groups
//! those fragments by class/plane, finds which ones from *different* tiles
//! are actually touching across their shared tile seam (a small variant of
//! [`Object::overlaps`]'s windowed mask scan), unions matched fragments via
//! a union-find, and rebuilds each merge-group as one [`Object`] through
//! [`Object::new`] so its geometry (perimeter, ellipse, ...) is recomputed
//! correctly from the real merged shape.
//!
//! This is a tiled/blocked variant of the classical two-pass connected-
//! component labeling algorithm (Rosenfeld & Pfaltz, 1966): each tile is
//! labeled independently, then object identity is reconciled across tile
//! boundaries via the same union-find idea, just run as one end-of-image
//! batch pass instead of incrementally online.

use crate::ImageTile;
use crate::object::{Intensity, Object, ObjectInit};
use crate::spatial_grid::BboxGrid;
use bitvec::prelude::*;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectId},
    settings::project_settings::{TileMergeConnectivity, TileMergeSettings},
};
use indexmap::IndexMap;
use std::collections::HashMap;

/// One tile-edge-touching object fragment, buffered by `job_executor` instead
/// of being exported immediately, together with the tile it was extracted
/// from (needed to know which fragments come from *different* tiles - two
/// fragments from the *same* tile are never merge candidates, they're just
/// two genuinely separate objects that both happen to touch that tile's edge).
pub(crate) struct PendingFragment {
    pub(crate) object: Object,
    pub(crate) tile: ImageTile,
}

/// True if `bbox` (absolute image coordinates, inclusive) reaches any edge of
/// `tile` (also absolute coordinates) - i.e. the object may have been cut off
/// by the tile boundary rather than genuinely ending there. Deliberately not
/// a field on [`Object`]: unlike `Object::touches_edge` (checked once, at
/// extraction time, against the *full* image), this only matters transiently
/// at the tile-merge decision point in `job_executor`, against whichever
/// tile happened to produce the fragment.
pub(crate) fn touches_tile_edge(bbox: [u32; 4], tile: &ImageTile) -> bool {
    let [x_min, y_min, x_max, y_max] = bbox;
    let tile_x_max = tile.offset_x + tile.width.saturating_sub(1);
    let tile_y_max = tile.offset_y + tile.height.saturating_sub(1);
    x_min as usize == tile.offset_x
        || y_min as usize == tile.offset_y
        || x_max as usize == tile_x_max
        || y_max as usize == tile_y_max
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
fn fragments_are_adjacent(a: &Object, b: &Object, connectivity: TileMergeConnectivity) -> bool {
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
        TileMergeConnectivity::FourConnected => &FOUR_CONNECTED,
        TileMergeConnectivity::EightConnected => &EIGHT_CONNECTED,
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

/// Groups `pending` by its full class set (sorted/deduped, so fragments that
/// don't actually share a class combination are never merged into each
/// other) plus plane. Every class merges by default -
/// `settings.classes_to_not_merge` is an opt-out deny-list, so a fragment is
/// dropped here only if it carries one of those classes.  `job_executor` is
/// expected to only ever buffer already-eligible fragments, but this stays
/// defensive rather than assuming that.
fn group_fragments(
    pending: &[PendingFragment],
    settings: &TileMergeSettings,
) -> IndexMap<
    (
        Vec<evanalyzer_cfg::core_types::ObjectClass>,
        crate::ImagePlane,
    ),
    Vec<usize>,
> {
    let mut groups: IndexMap<
        (
            Vec<evanalyzer_cfg::core_types::ObjectClass>,
            crate::ImagePlane,
        ),
        Vec<usize>,
    > = IndexMap::new();
    for (i, frag) in pending.iter().enumerate() {
        let denied = frag
            .object
            .object_class
            .iter()
            .any(|c| settings.classes_to_not_merge.contains(c));
        if denied {
            continue;
        }
        let mut classes: Vec<_> = frag.object.object_class.iter().copied().collect();
        classes.sort();
        classes.dedup();
        groups
            .entry((classes, frag.object.plane))
            .or_default()
            .push(i);
    }
    groups
}

/// Matches and merges buffered tile-edge fragments, run once per image after
/// every tile has finished. Returns one [`Object`] per merge-group: groups of
/// size 1 (a fragment that touched a tile edge but had no real cross-tile
/// neighbor - e.g. it also happens to touch the full image edge diagonally)
/// are returned unchanged, groups of size > 1 are rebuilt as one merged
/// object via [`Object::new`] (which recomputes perimeter/ellipse/etc. from
/// the real merged shape).
///
/// # Errors
/// [`InternalErrors::TooManyObjects`] if a single merge group (before
/// matching) exceeds `settings.max_fragments_per_group` - guards against a
/// class that turns out to match far more tile-edge-touching objects than
/// expected (since every class merges by default) building a pathological
/// merge instead of failing loudly.
pub(crate) fn merge_pending_fragments(
    pending: Vec<PendingFragment>,
    settings: &TileMergeSettings,
) -> Result<Vec<Object>, InternalErrors> {
    if pending.is_empty() {
        return Ok(Vec::new());
    }

    let groups = group_fragments(&pending, settings);
    let mut merged_objects = Vec::with_capacity(pending.len());

    for (_, indices) in groups {
        if indices.len() > settings.max_fragments_per_group as usize {
            return Err(InternalErrors::TooManyObjects(format!(
                "Detected {} tile-edge object fragments in one merge group, limit is {}. \
                 Add the offending class to `classes_to_not_merge` if it matches far more \
                 tile-edge-touching objects than expected.",
                indices.len(),
                settings.max_fragments_per_group
            )));
        }

        // Throwaway cache + BboxGrid over just this group's fragments, so
        // pairing candidates is O(N) instead of O(N^2) - same pattern
        // `Colocalization`/`ClassifyObjects.overlapping_with` use.
        let mut cache = crate::pipeline::pipeline_cache::PipelineCache::default();
        let ids: Vec<ObjectId> = indices
            .iter()
            .map(|&i| pending[i].object.id.clone())
            .collect();
        for &i in &indices {
            cache
                .object_cache
                .insert(pending[i].object.id.clone(), pending[i].object.clone());
        }
        let grid = BboxGrid::build(&ids, &cache);
        let id_to_local: HashMap<ObjectId, usize> = indices
            .iter()
            .enumerate()
            .map(|(local, &i)| (pending[i].object.id.clone(), local))
            .collect();

        let mut uf = UnionFind::new(indices.len());
        for (local_a, &i) in indices.iter().enumerate() {
            let frag_a = &pending[i];
            // Expand the query bbox by one pixel: `BboxGrid`'s own no-false-
            // negative guarantee only covers *true* bbox intersection, but
            // fragments from adjacent (non-overlapping) tiles are at best
            // touching, never truly intersecting.
            let [x0, y0, x1, y1] = frag_a.object.bbox;
            let query = [
                x0.saturating_sub(1),
                y0.saturating_sub(1),
                x1.saturating_add(1),
                y1.saturating_add(1),
            ];
            for candidate_id in grid.candidates(query) {
                let Some(&local_b) = id_to_local.get(&candidate_id) else {
                    continue;
                };
                if local_b <= local_a {
                    continue; // dedupe unordered pairs and skip self
                }
                let frag_b = &pending[indices[local_b]];
                if frag_a.tile.offset_x == frag_b.tile.offset_x
                    && frag_a.tile.offset_y == frag_b.tile.offset_y
                {
                    continue; // same tile - not a cross-tile fragment pair
                }
                if fragments_are_adjacent(&frag_a.object, &frag_b.object, settings.connectivity) {
                    uf.union(local_a, local_b);
                }
            }
        }

        let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
        for local in 0..indices.len() {
            let root = uf.find(local);
            by_root.entry(root).or_default().push(local);
        }

        for locals in by_root.into_values() {
            if locals.len() == 1 {
                merged_objects.push(pending[indices[locals[0]]].object.clone());
                continue;
            }

            let objs: Vec<&Object> = locals
                .iter()
                .map(|&l| &pending[indices[l]].object)
                .collect();
            let geometry = union_merge(&objs);
            let intensities = merge_intensities(&objs, geometry.area);

            let mut object_class = std::collections::HashSet::new();
            for o in &objs {
                object_class.extend(o.object_class.iter().copied());
            }

            let first = objs[0];
            merged_objects.push(Object::new(ObjectInit {
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
            }));
        }
    }

    Ok(merged_objects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImagePlane;
    use evanalyzer_cfg::core_types::ObjectClass;

    fn tile(offset_x: usize, offset_y: usize, width: usize, height: usize) -> ImageTile {
        ImageTile {
            offset_x,
            offset_y,
            width,
            height,
        }
    }

    /// A fully-filled rectangular fragment `[x0,y0,x1,y1]` (inclusive),
    /// carrying `class` and `plane`, with `touches_edge` set explicitly (as
    /// the real extraction path would have already computed it against the
    /// *full* image, not the tile it was cut from).
    fn rect_fragment(
        id: u128,
        bbox: [u32; 4],
        class: ObjectClass,
        plane: ImagePlane,
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
            sum_x: 0,
            sum_y: 0,
            sum_x2: 0,
            sum_y2: 0,
            sum_xy: 0,
            plane,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    fn settings(connectivity: TileMergeConnectivity) -> TileMergeSettings {
        TileMergeSettings {
            enabled: true,
            classes_to_not_merge: Vec::new(),
            connectivity,
            max_fragments_per_group: 100,
        }
    }

    #[test]
    fn empty_pending_returns_no_objects() {
        let merged =
            merge_pending_fragments(vec![], &settings(TileMergeConnectivity::EightConnected))
                .unwrap();
        assert!(merged.is_empty());
    }

    #[test]
    fn a_single_fragment_with_no_neighbor_is_returned_unchanged() {
        let object = rect_fragment(
            1,
            [0, 0, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![PendingFragment {
            object,
            tile: tile(0, 0, 10, 10),
        }];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, [0, 0, 9, 9]);
        assert_eq!(merged[0].area, 100);
    }

    /// Two fragments from horizontally-adjacent tiles, sharing a full-height
    /// seam (A ends at x=9, B starts at x=10) - the common case: an organ cut
    /// straight down the middle by a tile boundary.
    #[test]
    fn fragments_sharing_a_straight_tile_seam_merge_into_one_object() {
        let a = rect_fragment(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 10, 20),
            },
            PendingFragment {
                object: b,
                tile: tile(10, 0, 10, 20),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(
            merged.len(),
            1,
            "the two fragments must merge into one object"
        );
        assert_eq!(merged[0].bbox, [0, 0, 19, 19]);
        assert_eq!(merged[0].area, 400);
        assert!(
            !merged[0].touches_edge,
            "neither fragment touched the full image edge"
        );
    }

    /// Same seam, but the two fragments come from the *same* tile (a
    /// pathological/synthetic case) - must never merge, since fragment
    /// buffering only ever happens per-tile in the real pipeline and two
    /// fragments from one tile are always genuinely separate objects.
    #[test]
    fn fragments_from_the_same_tile_never_merge_even_if_touching() {
        let a = rect_fragment(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let same_tile = tile(0, 0, 20, 20);
        let pending = vec![
            PendingFragment {
                object: a,
                tile: same_tile.clone(),
            },
            PendingFragment {
                object: b,
                tile: same_tile,
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(
            merged.len(),
            2,
            "same-tile fragments must stay separate objects"
        );
    }

    /// Fragments touching at exactly one pixel (a single-pixel corner touch,
    /// not a full seam) must still be found and merged.
    #[test]
    fn fragments_touching_at_a_single_pixel_merge() {
        let a = rect_fragment(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [5, 5, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 5, 5),
            },
            PendingFragment {
                object: b,
                tile: tile(5, 5, 5, 5),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].area, 50);
    }

    /// The same single-pixel diagonal touch must NOT merge under
    /// 4-connectivity - a purely diagonal touch has no shared edge.
    #[test]
    fn diagonal_only_touch_does_not_merge_under_four_connectivity() {
        let a = rect_fragment(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [5, 5, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 5, 5),
            },
            PendingFragment {
                object: b,
                tile: tile(5, 5, 5, 5),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::FourConnected))
                .unwrap();

        assert_eq!(
            merged.len(),
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
        let tl = rect_fragment(1, [0, 0, 4, 4], class, plane, false);
        let tr = rect_fragment(2, [5, 0, 9, 4], class, plane, false);
        let bl = rect_fragment(3, [0, 5, 4, 9], class, plane, false);
        let br = rect_fragment(4, [5, 5, 9, 9], class, plane, false);
        let pending = vec![
            PendingFragment {
                object: tl,
                tile: tile(0, 0, 5, 5),
            },
            PendingFragment {
                object: tr,
                tile: tile(5, 0, 5, 5),
            },
            PendingFragment {
                object: bl,
                tile: tile(0, 5, 5, 5),
            },
            PendingFragment {
                object: br,
                tile: tile(5, 5, 5, 5),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(
            merged.len(),
            1,
            "all four corner fragments must transitively merge into one object"
        );
        assert_eq!(merged[0].area, 100);
        assert_eq!(merged[0].bbox, [0, 0, 9, 9]);
    }

    /// Fragments far apart (not touching at all) must never merge.
    #[test]
    fn far_apart_fragments_do_not_merge() {
        let a = rect_fragment(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [1000, 1000, 1004, 1004],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 5, 5),
            },
            PendingFragment {
                object: b,
                tile: tile(1000, 1000, 5, 5),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(merged.len(), 2);
    }

    /// Every class merges by default - `classes_to_not_merge` is an opt-out
    /// deny-list, so a fragment carrying a denied class is dropped entirely
    /// by `group_fragments` (defensive fallback for a fragment that
    /// shouldn't have been buffered by `job_executor` in the first place).
    #[test]
    fn fragments_with_a_denied_class_are_dropped() {
        let object = rect_fragment(
            1,
            [0, 0, 4, 4],
            ObjectClass::Valid(99),
            ImagePlane::default(),
            false,
        );
        let pending = vec![PendingFragment {
            object,
            tile: tile(0, 0, 5, 5),
        }];

        let mut denylisted = settings(TileMergeConnectivity::EightConnected);
        denylisted.classes_to_not_merge = vec![ObjectClass::Valid(99)];

        let merged = merge_pending_fragments(pending, &denylisted).unwrap();

        assert!(merged.is_empty());
    }

    /// A class that isn't on `classes_to_not_merge` merges without needing
    /// any explicit opt-in - the whole point of the deny-list default.
    #[test]
    fn a_class_not_on_the_deny_list_merges_by_default() {
        let a = rect_fragment(
            1,
            [0, 0, 4, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let b = rect_fragment(
            2,
            [5, 0, 9, 9],
            ObjectClass::Valid(1),
            ImagePlane::default(),
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 5, 10),
            },
            PendingFragment {
                object: b,
                tile: tile(5, 0, 5, 10),
            },
        ];

        let mut denylisted = settings(TileMergeConnectivity::EightConnected);
        denylisted.classes_to_not_merge = vec![ObjectClass::Valid(2)]; // unrelated class

        let merged = merge_pending_fragments(pending, &denylisted).unwrap();

        assert_eq!(
            merged.len(),
            1,
            "a class absent from classes_to_not_merge must still merge"
        );
    }

    /// Fragments on different `ImagePlane`s never merge, even if their masks
    /// touch - they're different channels/z-slices/timepoints, not pieces of
    /// the same real object.
    #[test]
    fn fragments_on_different_planes_do_not_merge() {
        let a = rect_fragment(
            1,
            [0, 0, 9, 19],
            ObjectClass::Valid(1),
            ImagePlane { z: 0, c: 0, t: 0 },
            false,
        );
        let b = rect_fragment(
            2,
            [10, 0, 19, 19],
            ObjectClass::Valid(1),
            ImagePlane { z: 0, c: 1, t: 0 },
            false,
        );
        let pending = vec![
            PendingFragment {
                object: a,
                tile: tile(0, 0, 10, 20),
            },
            PendingFragment {
                object: b,
                tile: tile(10, 0, 10, 20),
            },
        ];

        let merged =
            merge_pending_fragments(pending, &settings(TileMergeConnectivity::EightConnected))
                .unwrap();

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn a_merge_group_over_the_fragment_cap_fails_loudly() {
        let mut pending = Vec::new();
        for i in 0..5u128 {
            let x0 = (i as u32) * 10;
            pending.push(PendingFragment {
                object: rect_fragment(
                    i + 1,
                    [x0, 0, x0 + 4, 4],
                    ObjectClass::Valid(1),
                    ImagePlane::default(),
                    false,
                ),
                tile: tile(x0 as usize, 0, 5, 5),
            });
        }
        let mut too_strict = settings(TileMergeConnectivity::EightConnected);
        too_strict.max_fragments_per_group = 4;

        let result = merge_pending_fragments(pending, &too_strict);
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
