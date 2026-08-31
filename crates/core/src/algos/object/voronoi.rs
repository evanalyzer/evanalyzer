//! # Voronoi Module
//!
//! Provides functionality for creating a Voronoi tessellation from segmented object centers.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-05-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module computes a Voronoi diagram from a set of seed objects (centers), dividing the
//! image plane into regions where each pixel belongs to the nearest seed. The resulting areas
//! can optionally be confined to a mask object and limited by a maximum expansion radius.
//! Each Voronoi region is labeled with the configured output class and linked back to its
//! originating center object.
//!
use crate::{
    ImagePlane,
    algos::{ExecutionScope, ImageAlgorithm},
    object::{Object, ObjectInit},
};
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SegmentationClass, SizeUnits,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// Computes a Voronoi tessellation from segmented seed objects.
///
/// Each seed center expands outward until it reaches another region, the optional mask
/// boundary, or the maximum radius. The resulting areas are stored as new ROIs labeled
/// with `output_class` and linked to their originating center object.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct Voronoi {
    /// Object class whose instances act as Voronoi seed points.
    pub centers: ObjectClass,

    /// Additional label filters applied to center objects before tessellation.
    ///
    /// Only center objects that carry all listed classes pass the filter.
    /// Leave empty to include all objects of `centers`.
    pub center_filter_classes: Vec<ObjectClass>,

    /// Object class used to spatially constrain the Voronoi areas.
    ///
    /// Each computed Voronoi region is intersected with the union of all mask objects,
    /// discarding pixels that fall outside the mask. Set to `Unset` to expand
    /// to the full image boundary instead.
    pub mask: ObjectClass,

    /// Additional label filters applied to mask objects.
    ///
    /// Only mask objects that carry all listed classes pass the filter.
    /// Leave empty to include all objects of `mask`.
    pub mask_filter_classes: Vec<ObjectClass>,

    /// Object class assigned to the resulting Voronoi region ROIs.
    pub output_class: ObjectClass,

    /// Unit in which `max_radius` is expressed (e.g. pixels, nm, µm).
    pub unit: SizeUnits,

    /// Maximum expansion radius for a Voronoi region.
    ///
    /// Pixels farther than this distance from the nearest seed center are excluded
    /// from the region. Use `0` or a negative value to disable the limit.
    pub max_radius: f32,

    /// Discard Voronoi regions that touch the image border.
    pub exclude_areas_at_the_edges: bool,

    /// Discard Voronoi regions whose originating center object was filtered out or missing.
    pub exclude_areas_with_no_center: bool,
}

/// 2D KD-tree over center seed points, for exact nearest-center queries.
///
/// Replaces an earlier uniform-grid approach: that grid's cell size was tuned to the
/// *average* center density across the whole image, which works fine for evenly
/// spread centers but degrades badly - `O(ring_radius²)` per query - for a pixel far
/// from any center when centers are unevenly distributed (e.g. densely clustered in
/// the middle of an image with empty space at the edges, which is the normal case
/// for real tissue, not an edge case). A KD-tree's partitioning adapts to the actual
/// point distribution instead of a single global density estimate, so query cost
/// stays well-behaved regardless of clustering.
///
/// Produces byte-for-byte identical results to a brute-force scan (see
/// `center_kdtree_matches_brute_force_for_many_random_centers`): the same squared
/// Euclidean distance is compared at every visited center, with the same "lowest
/// index wins" tie-break, and a subtree is only pruned when *no* point inside it
/// could possibly beat (or tie) the best candidate found so far - so pruning can
/// never change which center's index the query settles on.
struct CenterKdTree {
    root: Option<Box<KdNode>>,
}

struct KdNode {
    /// Index into the `centers` slice this node holds.
    idx: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

impl CenterKdTree {
    fn build(centers: &[(ObjectId, f64, f64)]) -> Self {
        let mut indices: Vec<usize> = (0..centers.len()).collect();
        Self {
            root: Self::build_recursive(centers, &mut indices, 0),
        }
    }

    /// Splits `indices` on the median of the current axis (x at even depth, y at
    /// odd), recursing into the two halves - a standard balanced KD-tree build.
    /// `select_nth_unstable_by` partitions around the median in expected `O(k)`
    /// per level instead of fully sorting, so the whole build is `O(n log n)`.
    fn build_recursive(
        centers: &[(ObjectId, f64, f64)],
        indices: &mut [usize],
        depth: usize,
    ) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }
        if indices.len() == 1 {
            return Some(Box::new(KdNode {
                idx: indices[0],
                left: None,
                right: None,
            }));
        }

        let axis = depth % 2;
        let coord = |i: usize| if axis == 0 { centers[i].1 } else { centers[i].2 };
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| coord(a).partial_cmp(&coord(b)).unwrap());
        let idx = indices[mid];

        let (left, rest) = indices.split_at_mut(mid);
        let right = &mut rest[1..];
        Some(Box::new(KdNode {
            idx,
            left: Self::build_recursive(centers, left, depth + 1),
            right: Self::build_recursive(centers, right, depth + 1),
        }))
    }

    /// Finds the center nearest to `(x, y)`, returning its index into `centers` and
    /// the squared distance - `None` only when there are no centers at all.
    /// `max_dist_sq` bounds the search the same way it always has: a subtree whose
    /// closest possible point is already farther than `max_dist_sq` is skipped,
    /// since nothing in it could pass the caller's own `dist_sq <= max_dist_sq`
    /// check either way - this is what keeps a capped `max_radius` cheap even when
    /// most of the image is empty.
    fn nearest(
        &self,
        centers: &[(ObjectId, f64, f64)],
        x: f64,
        y: f64,
        max_dist_sq: f64,
    ) -> Option<(usize, f64)> {
        let root = self.root.as_ref()?;
        let mut best: Option<(usize, f64)> = None;
        Self::query_recursive(root, centers, x, y, 0, max_dist_sq, &mut best);
        best
    }

    fn query_recursive(
        node: &KdNode,
        centers: &[(ObjectId, f64, f64)],
        x: f64,
        y: f64,
        depth: usize,
        max_dist_sq: f64,
        best: &mut Option<(usize, f64)>,
    ) {
        let (_, cx, cy) = &centers[node.idx];
        let dx = x - cx;
        let dy = y - cy;
        let dist_sq = dx * dx + dy * dy;
        // Same tie-break as the brute-force reference: strictly closer wins, and an
        // exact tie goes to the lower index.
        let better = match best {
            None => true,
            Some((best_idx, best_dist)) => {
                dist_sq < *best_dist || (dist_sq == *best_dist && node.idx < *best_idx)
            }
        };
        if better {
            *best = Some((node.idx, dist_sq));
        }

        let axis = depth % 2;
        let diff = if axis == 0 { x - cx } else { y - cy };
        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(near) = near {
            Self::query_recursive(near, centers, x, y, depth + 1, max_dist_sq, best);
        }

        // The far side can only contain a point closer than `diff` along this axis
        // alone, so its closest *possible* point is at distance `diff²`. Skip it
        // unless that could still beat the current best, or beat `max_dist_sq` (no
        // point exploring for a candidate the caller would reject as out of range
        // anyway). `<=`, not `<`, on the current-best bound: an exact tie on the
        // splitting plane could still hide a lower-index candidate on the far side.
        let plane_dist_sq = diff * diff;
        let bound = best.map_or(max_dist_sq, |(_, d)| d.min(max_dist_sq));
        if plane_dist_sq <= bound
            && let Some(far) = far
        {
            Self::query_recursive(far, centers, x, y, depth + 1, max_dist_sq, best);
        }
    }
}

impl ImageAlgorithm for Voronoi {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        // Voronoi is `ExecutionScope::WholeImage` - it runs once against the
        // complete object set, not per tile, so its pixel-assignment pass
        // covers the *whole* image. That must come from `full_image_size()`
        // (metadata, always correct) rather than `ctx.get_image_size()`/
        // `get_image_tile_offset()` (the size of `ctx.image`'s actual
        // buffer): Voronoi's `start_image` is recommended to be Scratchpad,
        // whose placeholder buffer is deliberately allocated as small as
        // possible (nothing here ever reads its pixel content, only object
        // data from `cache.object_cache`), so it can't be used as a stand-in
        // for "how big is the image."
        let full_size = ctx.full_image_size();
        let full_w = full_size.width as u32;
        let full_h = full_size.height as u32;
        let tile_w = full_w;
        let tile_h = full_h;
        let off_x = 0u32;
        let off_y = 0u32;

        if tile_w == 0 || tile_h == 0 {
            return Ok(());
        }

        // Convert max_radius to pixels squared for distance comparisons.
        // max_radius <= 0 means unlimited expansion.
        let px_sizes = ctx.pixel_sizes();
        let max_dist_sq: f64 = if self.max_radius > 0.0 {
            let radius_px = match self.unit {
                SizeUnits::Pixels => self.max_radius as f64,
                SizeUnits::NanoMeter => (self.max_radius / px_sizes.px_size_x) as f64,
            };
            radius_px * radius_px
        } else {
            f64::MAX
        };

        // --- Phase 1: Collect filtered center objects and their seed coordinates ---
        // Seed point is the bounding-box centre, matching the C++ reference implementation.
        let centers: Vec<(ObjectId, f64, f64)> = cache
            .object_cache
            .values()
            .filter(|object| {
                object.has_object_class(&self.centers)
                    && self
                        .center_filter_classes
                        .iter()
                        .all(|f| object.has_object_class(f))
            })
            .map(|object| {
                let [x_min, y_min, x_max, y_max] = object.bbox;
                let cx = (x_min + x_max) as f64 / 2.0;
                let cy = (y_min + y_max) as f64 / 2.0;
                (object.id.clone(), cx, cy)
            })
            .collect();

        if centers.is_empty() {
            return Ok(());
        }

        // --- Phase 2: Collect mask object references ---
        let has_mask = self.mask != ObjectClass::Unset;
        let mask_objects: Vec<&Object> = if has_mask {
            cache
                .object_cache
                .values()
                .filter(|r| {
                    r.has_object_class(&self.mask)
                        && self
                            .mask_filter_classes
                            .iter()
                            .all(|f| r.has_object_class(f))
                })
                .collect()
        } else {
            vec![]
        };

        // --- Phase 3: Assign each pixel to its nearest center (distance-transform Voronoi) ---
        // Simultaneously apply the mask constraint to avoid a second full-image scan.
        // A KD-tree over the centers turns the per-pixel nearest-center query into a
        // small number of comparisons instead of scanning every center - and unlike a
        // uniform grid, its cost doesn't depend on centers being evenly distributed
        // (see `CenterKdTree`'s doc comment).
        //
        // Runs in parallel over row-chunks of a flat `labels` buffer (nearest-center
        // index per pixel, `UNASSIGNED` for excluded pixels) rather than building a
        // `Vec<(u32, u32)>` of assigned pixel coordinates per center: for a
        // whole-slide-sized image that per-center-Vec approach costs several times
        // more memory (every pixel's coordinates stored explicitly, plus per-Vec
        // growth overhead) than one `u32` index per pixel. Each row is independent -
        // no pixel's assignment depends on any other's - so this is safe to split
        // across threads with no shared mutable state beyond each row's own slice.
        let n = centers.len();
        const UNASSIGNED: u32 = u32::MAX;
        let mut labels: Vec<u32> = vec![UNASSIGNED; tile_w as usize * tile_h as usize];
        let kdtree = CenterKdTree::build(&centers);

        labels
            .par_chunks_mut(tile_w as usize)
            .enumerate()
            .for_each(|(row, labels_row)| {
                let y = off_y + row as u32;
                for (col, label) in labels_row.iter_mut().enumerate() {
                    let x = off_x + col as u32;

                    // Skip pixels outside the mask when a mask is configured.
                    if has_mask && !mask_objects.iter().any(|mr| mr.is_part_of(x, y)) {
                        continue;
                    }

                    // Apply max_radius separately with <= so boundary pixels are included,
                    // matching the filled-ellipse behaviour of the C++ reference.
                    if let Some((nearest, dist_sq)) =
                        kdtree.nearest(&centers, x as f64, y as f64, max_dist_sq)
                        && dist_sq <= max_dist_sq
                    {
                        *label = nearest as u32;
                    }
                }
            });

        // --- Phase 4: Accumulate each center's extent/moments in one pass over `labels` ---
        let mut area = vec![0usize; n];
        let mut bbox = vec![[u32::MAX, u32::MAX, 0u32, 0u32]; n];
        let mut sum_x = vec![0u64; n];
        let mut sum_y = vec![0u64; n];
        let mut sum_x2 = vec![0u64; n];
        let mut sum_y2 = vec![0u64; n];
        let mut sum_xy = vec![0u64; n];

        for row in 0..tile_h {
            let y = off_y + row;
            let row_start = row as usize * tile_w as usize;
            for col in 0..tile_w {
                let label = labels[row_start + col as usize];
                if label == UNASSIGNED {
                    continue;
                }
                let i = label as usize;
                let x = off_x + col;
                area[i] += 1;
                sum_x[i] += x as u64;
                sum_y[i] += y as u64;
                sum_x2[i] += (x as u64) * (x as u64);
                sum_y2[i] += (y as u64) * (y as u64);
                sum_xy[i] += (x as u64) * (y as u64);
                let b = &mut bbox[i];
                b[0] = b[0].min(x);
                b[1] = b[1].min(y);
                b[2] = b[2].max(x);
                b[3] = b[3].max(y);
            }
        }

        // --- Phase 5: Build one object per center from its assigned pixels ---
        // Each region's plane comes from its own seeding center, not from
        // `ctx.image`: when Voronoi runs in a separate, Scratchpad-sourced
        // pipeline (the recommended setup for a pure object-manipulation step
        // with no pixel input), `ctx.image` is a freshly allocated buffer with
        // no plane metadata at all, while the center ROIs - read from
        // `cache.object_cache`, which does survive across pipelines - already carry
        // the correct z/c/t from whichever pipeline originally extracted them.
        let fallback_plane = ImagePlane {
            z: -1,
            c: -1,
            t: -1,
        };

        // Collect new ROIs before mutating the cache.
        let mut new_objects: Vec<Object> = Vec::new();

        for i in 0..n {
            if area[i] == 0 {
                continue;
            }

            let [x_min, y_min, x_max, y_max] = bbox[i];
            // bbox convention: bbox[2]/[3] are INCLUSIVE maximum pixel coordinates,
            // matching the convention used by extract_objects and the renderer.
            // The mask stride is therefore (bbox[2] - bbox[0] + 1).
            let w = (x_max - x_min + 1) as usize;
            let h = (y_max - y_min + 1) as usize;

            // Re-scan only this region's own bbox window (against the shared `labels`
            // buffer, not a per-center pixel list) to build its bbox-relative mask.
            let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, w * h);
            for ry in 0..h {
                let y = y_min + ry as u32;
                let row_start = (y - off_y) as usize * tile_w as usize;
                for rx in 0..w {
                    let x = x_min + rx as u32;
                    if labels[row_start + (x - off_x) as usize] == i as u32 {
                        mask_data.set(ry * w + rx, true);
                    }
                }
            }

            // With inclusive bbox: touching the right/bottom edge means the max pixel
            // is the last column/row of the *full* image (index full_w-1 / full_h-1) -
            // a tile boundary is not an image edge, the object may continue in the
            // neighboring tile.
            let touches_edge =
                x_min == 0 || y_min == 0 || x_max + 1 >= full_w || y_max + 1 >= full_h;

            if self.exclude_areas_at_the_edges && touches_edge {
                continue;
            }

            // Discard if the seeding center's bounding-box midpoint is no longer inside
            // the (potentially mask-clipped) Voronoi area.
            if self.exclude_areas_with_no_center {
                let (_, cx, cy) = &centers[i];
                let cx_u = *cx as u32;
                let cy_u = *cy as u32;
                let inside = cx_u >= x_min
                    && cx_u <= x_max
                    && cy_u >= y_min
                    && cy_u <= y_max
                    && mask_data
                        .get((cy_u - y_min) as usize * w + (cx_u - x_min) as usize)
                        .map(|b| *b)
                        .unwrap_or(false);
                if !inside {
                    continue;
                }
            }

            let (center_id, _, _) = &centers[i];
            let plane = cache
                .object_cache
                .get(center_id)
                .map(|center_object| center_object.plane)
                .unwrap_or(fallback_plane);
            let mut object = Object::new(ObjectInit {
                id: ObjectId::next(),
                segmentation_class: SegmentationClass::MANUAL_ANNOTATED,
                bbox: [x_min, y_min, x_max, y_max],
                mask_data,
                area: area[i],
                plane,
                touches_edge,
                sum_x: sum_x[i],
                sum_y: sum_y[i],
                sum_x2: sum_x2[i],
                sum_y2: sum_y2[i],
                sum_xy: sum_xy[i],
                parent_id: Some(center_id.clone()),
                ..Default::default()
            });
            object.add_object_class(self.output_class);
            // `Object::new` only derives geometry from the mask; Voronoi regions never
            // pass through `ExtractObjects` (the step that normally samples pixel data),
            // so they need their own intensity measurement pass here.
            object.intensities = object.measure_intensities(cache);
            new_objects.push(object);
        }

        // --- Phase 6: Insert the new ROIs into the cache ---
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Voronoi"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        Some(&CitationMetadata {
            cite_key: "voronoi1908nouvelles",
            title: "Nouvelles applications des paramètres continus à la théorie des formes quadratiques. Deuxième mémoire",
            authors: &["Georgy Voronoi"],
            year: 1908,
            container: Some("Journal für die reine und angewandte Mathematik"),
            doi: None,
            url: None,
            pages: Some("198-287"),
        })
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        ImageContainer, ImagePlane, ImageTile, ManagedImage,
        image::PixelSizes,
        pipeline::{
            pipeline::PipelineImageMeta, pipeline_cache::GlobalPipelineCache,
            pipeline_context::PipelineContext,
        },
    };
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{ObjectClass, ObjectId};
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    const CENTER_CLASS: ObjectClass = ObjectClass::Valid(1);
    const MASK_CLASS: ObjectClass = ObjectClass::Valid(2);
    const OUTPUT_CLASS: ObjectClass = ObjectClass::Valid(10);
    const ID_A: u128 = 100_000;
    const ID_B: u128 = 200_000;
    const ID_MASK: u128 = 300_000;

    fn make_ctx(width: usize, height: usize) -> PipelineContext {
        let size = ImageSize { width, height };
        let img =
            Image::<f32, 1, CpuAllocator>::new(size, vec![0.0f32; width * height], CpuAllocator)
                .unwrap();
        let managed = ManagedImage {
            data: img,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        };
        PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: size.width,
                    height: size.height,
                },
                full_image_width: size,
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::F32Gray(managed).into(),
        )
        .unwrap()
    }

    /// Builds a context for a tile that is smaller than the full image, at a
    /// nonzero offset - the multi-tile (whole-slide-image) case.
    fn make_tiled_ctx(
        tile_w: usize,
        tile_h: usize,
        full_w: usize,
        full_h: usize,
        off_x: usize,
        off_y: usize,
    ) -> PipelineContext {
        let tile_size = ImageSize {
            width: tile_w,
            height: tile_h,
        };
        let img = Image::<f32, 1, CpuAllocator>::new(
            tile_size,
            vec![0.0f32; tile_w * tile_h],
            CpuAllocator,
        )
        .unwrap();
        let managed = ManagedImage {
            data: img,
            tile_offset: Point2d { x: off_x, y: off_y },
            plane: None,
        };
        PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: off_x,
                    offset_y: off_y,
                    width: tile_size.width,
                    height: tile_size.height,
                },
                full_image_width: ImageSize {
                    width: full_w,
                    height: full_h,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::F32Gray(managed).into(),
        )
        .unwrap()
    }

    fn make_filled_object(id: u128, bbox: [u32; 4], class: ObjectClass) -> Object {
        let [x_min, y_min, x_max, y_max] = bbox;
        // bbox uses inclusive convention: width = xmax - xmin + 1
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
        object.add_object_class(class);
        object
    }

    fn default_voronoi() -> Voronoi {
        Voronoi {
            centers: CENTER_CLASS,
            center_filter_classes: vec![],
            mask: ObjectClass::Unset,
            mask_filter_classes: vec![],
            output_class: OUTPUT_CLASS,
            unit: SizeUnits::Pixels,
            max_radius: 0.0,
            exclude_areas_at_the_edges: false,
            exclude_areas_with_no_center: false,
        }
    }

    fn voronoi_objects(cache: &GlobalPipelineCache) -> Vec<&Object> {
        cache
            .object_cache
            .values()
            .filter(|r| r.has_object_class(&OUTPUT_CLASS))
            .collect()
    }

    fn run(v: &Voronoi, ctx: &mut PipelineContext, cache: &mut GlobalPipelineCache) {
        v.execute(ctx, cache).unwrap();
    }

    fn center_bbox(cx: u32, cy: u32) -> [u32; 4] {
        [cx - 1, cy - 1, cx + 1, cy + 1]
    }

    // --- Tests ---

    #[test]
    fn voronoi_regions_have_measured_intensities() {
        // Regression test: Voronoi regions never pass through ExtractObjects (the step
        // that normally samples pixel data), so without their own measurement pass
        // they'd be left with empty intensities.
        const CHANNEL: i32 = 0;
        const VALUE: f32 = 5.0;
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();

        let channel_img = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: 10,
                height: 10,
            },
            vec![VALUE; 100],
            CpuAllocator,
        )
        .unwrap();
        cache.add_to_channel_cache(
            std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: channel_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })),
            CHANNEL,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: 10,
                height: 10,
            },
        );

        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 100);
        let intensity = regions[0]
            .intensities
            .get(&CHANNEL)
            .expect("Voronoi region should have measured intensities for the channel");
        assert_eq!(intensity.sum_intensity, 100.0 * VALUE as f64);
        assert_eq!(intensity.avg_intensity, VALUE);
        assert_eq!(intensity.min_intensity, VALUE);
        assert_eq!(intensity.max_intensity, VALUE);
    }

    #[test]
    fn voronoi_covers_the_whole_image_even_when_ctxs_buffer_is_a_smaller_placeholder() {
        // Voronoi is `ExecutionScope::WholeImage` - it must size its pixel-assignment
        // pass from `full_image_size()` (metadata), not from `ctx.image`'s actual
        // buffer, since that buffer is only ever a placeholder for a Scratchpad-
        // sourced pipeline (nothing here reads its pixel content) and is deliberately
        // allocated much smaller than the real image to avoid a full-image-sized
        // allocation on every whole-image run. This constructs exactly that
        // situation - a 20x20 placeholder buffer inside a 500x500 image - and checks
        // the region still spans the *whole* image, not just the placeholder's bounds.
        const CHANNEL: i32 = 0;
        let tile_w = 20usize;
        let tile_h = 20usize;
        let off_x = 100usize;
        let off_y = 100usize;
        let full_w = 500usize;
        let full_h = 500usize;
        let mut ctx = make_tiled_ctx(tile_w, tile_h, full_w, full_h, off_x, off_y);
        let mut cache = GlobalPipelineCache::default();

        // Only a small patch of channel data is actually loaded (as it would be for
        // one tile's worth of pixels) - intensity sampling must still only sum over
        // the pixels it actually has, silently skipping the rest of the now-larger
        // region rather than panicking.
        let channel_img = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: tile_w,
                height: tile_h,
            },
            vec![7.0f32; tile_w * tile_h],
            CpuAllocator,
        )
        .unwrap();
        cache.add_to_channel_cache(
            std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: channel_img,
                tile_offset: Point2d { x: off_x, y: off_y },
                plane: None,
            })),
            CHANNEL,
            ImageTile {
                offset_x: off_x,
                offset_y: off_y,
                width: tile_w,
                height: tile_h,
            },
        );

        // A center near the middle of the placeholder tile, in absolute
        // (full-image) coordinates.
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(110, 110), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(
            regions[0].bbox,
            [0, 0, full_w as u32 - 1, full_h as u32 - 1],
            "the single center has no competitor, mask, or radius limit, so its region \
             must cover the entire image, not just the placeholder buffer's bounds"
        );
        assert_eq!(regions[0].area, full_w * full_h);
        assert_eq!(
            regions[0].intensities.get(&CHANNEL).unwrap().sum_intensity,
            (tile_w * tile_h) as f64 * 7.0,
            "intensity sampling must still only sum the pixels actually loaded, \
             silently skipping the rest of the region rather than panicking"
        );
    }

    #[test]
    fn no_centers_produces_no_output() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        run(&default_voronoi(), &mut ctx, &mut cache);
        assert!(voronoi_objects(&cache).is_empty());
    }

    #[test]
    fn single_center_covers_full_image() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 100);
    }

    #[test]
    fn two_centers_partition_image_without_overlap_or_gap() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(2, 5), CENTER_CLASS),
        );
        cache.object_cache.insert(
            ObjectId(ID_B),
            make_filled_object(ID_B, center_bbox(7, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 2);

        let total: usize = regions.iter().map(|r| r.area).sum();
        assert_eq!(total, 100);

        let mut areas: Vec<usize> = regions.iter().map(|r| r.area).collect();
        areas.sort_unstable();
        assert_eq!(areas, vec![50, 50]);
    }

    #[test]
    fn max_radius_limits_assigned_pixels() {
        let mut ctx = make_ctx(20, 20);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(10, 10), CENTER_CLASS),
        );

        run(
            &Voronoi {
                max_radius: 2.0,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);

        let expected: usize = (0u32..20)
            .flat_map(|y| (0u32..20).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let dx = x as f64 - 10.0;
                let dy = y as f64 - 10.0;
                dx * dx + dy * dy <= 4.0
            })
            .count();
        assert_eq!(regions[0].area, expected);
    }

    #[test]
    fn mask_clips_voronoi_region() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );
        cache.object_cache.insert(
            ObjectId(ID_MASK),
            make_filled_object(ID_MASK, [2, 2, 7, 7], MASK_CLASS), // inclusive [2,7] = 6 wide → 6×6=36
        );

        run(
            &Voronoi {
                mask: MASK_CLASS,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 36);
    }

    #[test]
    fn edge_exclusion_discards_border_touching_regions() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(
            &Voronoi {
                exclude_areas_at_the_edges: true,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        assert!(voronoi_objects(&cache).is_empty());
    }

    #[test]
    fn edge_exclusion_off_keeps_border_region() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        assert_eq!(voronoi_objects(&cache).len(), 1);
    }

    #[test]
    fn center_exclusion_discards_region_when_seed_outside_mask() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 4, 1, 6], CENTER_CLASS),
        );
        cache.object_cache.insert(
            ObjectId(ID_MASK),
            make_filled_object(ID_MASK, [2, 0, 4, 10], MASK_CLASS),
        );

        run(
            &Voronoi {
                mask: MASK_CLASS,
                exclude_areas_with_no_center: true,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        assert!(voronoi_objects(&cache).is_empty());
    }

    #[test]
    fn center_exclusion_off_keeps_region_even_when_seed_outside_mask() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 4, 1, 6], CENTER_CLASS),
        );
        cache.object_cache.insert(
            ObjectId(ID_MASK),
            make_filled_object(ID_MASK, [2, 0, 3, 9], MASK_CLASS), // inclusive [2,3]×[0,9] = 2×10=20
        );

        run(
            &Voronoi {
                mask: MASK_CLASS,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 20);
    }

    #[test]
    fn center_filter_class_excludes_unmatched_centers() {
        const FILTER_CLASS: ObjectClass = ObjectClass::Valid(3);
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();

        let object_a = make_filled_object(ID_A, center_bbox(2, 5), CENTER_CLASS);
        let mut object_b = make_filled_object(ID_B, center_bbox(7, 5), CENTER_CLASS);
        object_b.add_object_class(FILTER_CLASS);

        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(
            &Voronoi {
                center_filter_classes: vec![FILTER_CLASS],
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 100);
    }

    #[test]
    fn region_plane_comes_from_its_center_not_from_ctx_image() {
        // Simulates Voronoi running in a separate, Scratchpad-sourced pipeline:
        // `ctx.image` carries no plane metadata at all (`make_ctx` always builds
        // it with `plane: None`), while the center object - surviving in
        // `cache.object_cache` from an earlier pipeline - carries a real plane.
        let mut ctx = make_ctx(10, 10);
        let mut cache = GlobalPipelineCache::default();
        let center_plane = ImagePlane { z: 3, c: 1, t: 2 };
        let mut object_a = make_filled_object(ID_A, center_bbox(5, 5), CENTER_CLASS);
        object_a.plane = center_plane;
        cache.object_cache.insert(object_a.id.clone(), object_a);

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_objects(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(
            regions[0].plane, center_plane,
            "Voronoi region must inherit its center's plane, not ctx.image's (which is \
             None/unset when Voronoi runs from a Scratchpad-sourced pipeline)"
        );
    }

    /// Tiny deterministic LCG so the stress test below doesn't need a `rand` dependency.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    /// Asserts `kdtree.nearest(...)` agrees with a brute-force scan for every pixel
    /// of `img_w`x`img_h` against `centers`, including the exact tie-broken index and
    /// squared distance - not just "some center within range".
    fn assert_kdtree_matches_brute_force(
        label: &str,
        centers: &[(ObjectId, f64, f64)],
        img_w: u32,
        img_h: u32,
    ) {
        let kdtree = CenterKdTree::build(centers);

        for y in 0..img_h {
            for x in 0..img_w {
                let (kd_idx, kd_dist_sq) = kdtree
                    .nearest(centers, x as f64, y as f64, f64::MAX)
                    .expect("non-empty centers must yield a nearest match");

                let mut brute_dist_sq = f64::MAX;
                let mut brute_idx = usize::MAX;
                for (i, (_, cx, cy)) in centers.iter().enumerate() {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < brute_dist_sq {
                        brute_dist_sq = dist_sq;
                        brute_idx = i;
                    }
                }

                assert_eq!(
                    kd_idx, brute_idx,
                    "{label} pixel ({x},{y}): kd-tree picked center {kd_idx} but brute force picked {brute_idx}"
                );
                assert_eq!(kd_dist_sq, brute_dist_sq, "{label} pixel ({x},{y})");
            }
        }
    }

    /// `CenterKdTree::nearest` is a performance optimisation over a brute-force scan
    /// of every center; this checks it produces identical (index, tie-break) results
    /// to that brute-force scan across many random, roughly-uniform center layouts,
    /// not just the simple hand-picked layouts used by the other tests above.
    #[test]
    fn center_kdtree_matches_brute_force_for_many_random_centers() {
        let img_w = 80u32;
        let img_h = 60u32;
        let mut state = 0xC0FFEEu64;

        for trial in 0..20 {
            let n = 2 + (trial % 30);
            let centers: Vec<(ObjectId, f64, f64)> = (0..n)
                .map(|i| {
                    let cx = (next_rand(&mut state) % (img_w as u64 * 4)) as f64 / 4.0;
                    let cy = (next_rand(&mut state) % (img_h as u64 * 4)) as f64 / 4.0;
                    (ObjectId(i as u128), cx, cy)
                })
                .collect();

            assert_kdtree_matches_brute_force(&format!("trial {trial}"), &centers, img_w, img_h);
        }
    }

    /// Regression test for the pathology `CenterGrid` (the uniform-grid predecessor
    /// of `CenterKdTree`) had: a cell size tuned to the *average* density across the
    /// whole image degrades badly once centers are unevenly distributed - e.g. a
    /// dense cluster in the middle of the image with empty space at the edges, which
    /// is the normal shape of real tissue on a slide, not a hand-picked edge case.
    /// This doesn't assert on timing (too environment-dependent for a unit test),
    /// only that the result is still exactly correct for this layout - the KD-tree's
    /// actual performance advantage here was confirmed manually.
    #[test]
    fn center_kdtree_matches_brute_force_for_a_dense_cluster_with_empty_edges() {
        let img_w = 100u32;
        let img_h = 100u32;
        let mut state = 0xDEADBEEFu64;

        // Every center packed into a small region near the middle of the image,
        // leaving the rest of the (much larger) image empty - the layout that made
        // the old uniform-grid search's ring expansion blow up.
        let centers: Vec<(ObjectId, f64, f64)> = (0..200)
            .map(|i| {
                let cx = 45.0 + (next_rand(&mut state) % 40) as f64 / 4.0;
                let cy = 45.0 + (next_rand(&mut state) % 40) as f64 / 4.0;
                (ObjectId(i as u128), cx, cy)
            })
            .collect();

        assert_kdtree_matches_brute_force("dense cluster", &centers, img_w, img_h);
    }

    /// A fully-filled rectangular object carrying an explicit `source_tile` -
    /// what `TileMerge` needs to recognize a tile-edge fragment, unlike
    /// `make_filled_object` above (which leaves `source_tile` defaulted and
    /// is only ever used untiled in this file's other tests).
    fn tile_fragment(id: u128, bbox: [u32; 4], class: ObjectClass, source_tile: ImageTile) -> Object {
        let [x0, y0, x1, y1] = bbox;
        let w = (x1 - x0 + 1) as usize;
        let h = (y1 - y0 + 1) as usize;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: BitVec::<u64, Lsb0>::repeat(true, w * h),
            area: w * h,
            plane: ImagePlane::default(),
            source_tile,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    /// Composed-correctness regression: a real nucleus split across a tile
    /// boundary must seed exactly one Voronoi region once `TileMerge` has
    /// reconstructed it - not two, which would carve a spurious internal
    /// boundary straight through what should be one cell's territory.
    /// Proves the specific claim that Voronoi is correct for whole-slide
    /// (tiled) images, not just that it's `WholeImage`-scoped (already
    /// covered by `voronoi_regions_are_computed_once_across_the_whole_image_not_per_tile`
    /// in `pipeline.rs`, using a plain union as a tile-merge stand-in rather
    /// than running the real algorithm).
    #[test]
    fn tile_merged_object_produces_one_voronoi_region_not_a_spurious_split() {
        use crate::algos::{Connectivity, TileMerge};

        let tile_a = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: 10,
            height: 10,
        };
        let tile_b = ImageTile {
            offset_x: 10,
            offset_y: 0,
            width: 10,
            height: 10,
        };
        // One real nucleus, split by the tile boundary at x=10: fragment A
        // ends at x=9 (its tile's right edge), fragment B starts at x=10
        // (its tile's left edge).
        let objects = || {
            vec![
                tile_fragment(ID_A, [8, 4, 9, 5], CENTER_CLASS, tile_a.clone()),
                tile_fragment(ID_B, [10, 4, 11, 5], CENTER_CLASS, tile_b.clone()),
                // An unrelated, genuinely separate center, entirely interior
                // to its own tile - a fixed point the region count is
                // checked against.
                tile_fragment(300_000, [0, 0, 1, 1], CENTER_CLASS, tile_a.clone()),
            ]
        };

        // Sanity check: without tile-merge, the split fragments wrongly seed
        // two separate regions (the bug this test guards against) - proves
        // this scenario actually exercises it, not just that any layout
        // happens to produce two regions.
        let mut unmerged_ctx = make_ctx(20, 10);
        let mut unmerged_cache = GlobalPipelineCache::default();
        for o in objects() {
            unmerged_cache.object_cache.insert(o.id.clone(), o);
        }
        run(&default_voronoi(), &mut unmerged_ctx, &mut unmerged_cache);
        assert_eq!(
            voronoi_objects(&unmerged_cache).len(),
            3,
            "sanity check: unmerged split fragments must wrongly seed two regions (plus the far one)"
        );

        // Now merge first, exactly as the real whole-image pipeline order
        // does (TileMerge always runs before any other whole-image command).
        let mut ctx = make_ctx(20, 10);
        let mut cache = GlobalPipelineCache::default();
        for o in objects() {
            cache.object_cache.insert(o.id.clone(), o);
        }
        TileMerge {
            classes_to_not_merge: Vec::new(),
            connectivity: Connectivity::EightConnected,
            max_fragments_per_group: 100,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        assert_eq!(
            cache.object_cache.len(),
            2,
            "the two tile fragments must have merged into one object"
        );

        run(&default_voronoi(), &mut ctx, &mut cache);
        assert_eq!(
            voronoi_objects(&cache).len(),
            2,
            "after merging, the reconstructed object must seed exactly one Voronoi region, not two"
        );
    }
}
