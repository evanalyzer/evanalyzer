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
    algos::ImageAlgorithm,
    roi::{Roi, RoiInit},
};
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{
    InternalErrors, ObjectClass, ObjectId, SegmentationClass, SizeUnits,
};
use macros::CommandsMeta;

/// Computes a Voronoi tessellation from segmented seed objects.
///
/// Each seed center expands outward until it reaches another region, the optional mask
/// boundary, or the maximum radius. The resulting areas are stored as new ROIs labeled
/// with `output_class` and linked to their originating center object.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
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

/// Uniform spatial grid over center seed points.
///
/// Lets the per-pixel nearest-center query check only the centers in nearby cells
/// instead of scanning every center, without changing the result: the search expands
/// ring by ring until the guaranteed minimum distance to any unscanned cell exceeds
/// the best candidate found so far, so it always converges on the true nearest center.
struct CenterGrid {
    cell_size: f64,
    grid_w: usize,
    grid_h: usize,
    cells: Vec<Vec<u32>>,
}

impl CenterGrid {
    fn build(centers: &[(ObjectId, f64, f64)], img_w: u32, img_h: u32) -> Self {
        let cell_size = (img_w as f64 * img_h as f64 / centers.len() as f64)
            .sqrt()
            .max(1.0);
        let grid_w = ((img_w as f64 / cell_size).ceil() as usize).max(1);
        let grid_h = ((img_h as f64 / cell_size).ceil() as usize).max(1);

        let mut cells = vec![Vec::new(); grid_w * grid_h];
        for (i, &(_, cx, cy)) in centers.iter().enumerate() {
            let gx = ((cx / cell_size) as isize).clamp(0, grid_w as isize - 1) as usize;
            let gy = ((cy / cell_size) as isize).clamp(0, grid_h as isize - 1) as usize;
            cells[gy * grid_w + gx].push(i as u32);
        }

        Self {
            cell_size,
            grid_w,
            grid_h,
            cells,
        }
    }

    fn cell_of(&self, x: f64, y: f64) -> (isize, isize) {
        let gx = ((x / self.cell_size) as isize).clamp(0, self.grid_w as isize - 1);
        let gy = ((y / self.cell_size) as isize).clamp(0, self.grid_h as isize - 1);
        (gx, gy)
    }

    /// Updates `best_dist_sq`/`nearest` with any center in cell `(gx, gy)` closer to
    /// `(x, y)` than the current best. Ties keep the lower index, matching a
    /// brute-force scan in index order.
    fn scan_cell(
        &self,
        gx: isize,
        gy: isize,
        centers: &[(ObjectId, f64, f64)],
        x: f64,
        y: f64,
        best_dist_sq: &mut f64,
        nearest: &mut usize,
    ) {
        if gx < 0 || gy < 0 || gx as usize >= self.grid_w || gy as usize >= self.grid_h {
            return;
        }
        for &i in &self.cells[gy as usize * self.grid_w + gx as usize] {
            let i = i as usize;
            let (_, cx, cy) = &centers[i];
            let dx = x - cx;
            let dy = y - cy;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < *best_dist_sq || (dist_sq == *best_dist_sq && i < *nearest) {
                *best_dist_sq = dist_sq;
                *nearest = i;
            }
        }
    }

    /// Finds the center nearest to `(x, y)`, returning its index into `centers` and the
    /// squared distance. `max_dist_sq` allows the search to stop early once no
    /// unscanned cell could possibly hold a center within that bound.
    fn nearest(
        &self,
        centers: &[(ObjectId, f64, f64)],
        x: f64,
        y: f64,
        max_dist_sq: f64,
    ) -> Option<(usize, f64)> {
        let (px_gx, px_gy) = self.cell_of(x, y);
        let mut best_dist_sq = f64::MAX;
        let mut nearest = usize::MAX;
        let max_r = (self.grid_w + self.grid_h) as isize;

        let mut r: isize = 0;
        loop {
            if r == 0 {
                self.scan_cell(px_gx, px_gy, centers, x, y, &mut best_dist_sq, &mut nearest);
            } else {
                for gx in (px_gx - r)..=(px_gx + r) {
                    self.scan_cell(gx, px_gy - r, centers, x, y, &mut best_dist_sq, &mut nearest);
                    self.scan_cell(gx, px_gy + r, centers, x, y, &mut best_dist_sq, &mut nearest);
                }
                for gy in (px_gy - r + 1)..=(px_gy + r - 1) {
                    self.scan_cell(px_gx - r, gy, centers, x, y, &mut best_dist_sq, &mut nearest);
                    self.scan_cell(px_gx + r, gy, centers, x, y, &mut best_dist_sq, &mut nearest);
                }
            }

            // After fully scanning rings 0..=r, any unscanned cell is at Chebyshev grid
            // distance > r, so the closest point it could contain is at least r*cell_size
            // away - safe to stop once that bound can no longer beat the best match.
            let bound_sq = (r as f64 * self.cell_size).powi(2);
            if nearest != usize::MAX && best_dist_sq <= bound_sq {
                break;
            }
            // Once the bound alone exceeds max_dist_sq, anything still unscanned would be
            // excluded by the radius limit anyway, regardless of how "near" it is.
            if bound_sq > max_dist_sq {
                break;
            }
            r += 1;
            if r > max_r {
                break;
            }
        }

        (nearest != usize::MAX).then(|| (nearest, best_dist_sq))
    }
}

impl ImageAlgorithm for Voronoi {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        let img_size = ctx.full_image_size();
        let img_w = img_size.width as u32;
        let img_h = img_size.height as u32;

        if img_w == 0 || img_h == 0 {
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
            .roi_cache
            .values()
            .filter(|roi| {
                roi.has_object_class(&self.centers)
                    && self
                        .center_filter_classes
                        .iter()
                        .all(|f| roi.has_object_class(f))
            })
            .map(|roi| {
                let [x_min, y_min, x_max, y_max] = roi.bbox;
                let cx = (x_min + x_max) as f64 / 2.0;
                let cy = (y_min + y_max) as f64 / 2.0;
                (roi.id.clone(), cx, cy)
            })
            .collect();

        if centers.is_empty() {
            return Ok(());
        }

        // --- Phase 2: Collect mask ROI references ---
        let has_mask = self.mask != ObjectClass::Unset;
        let mask_rois: Vec<&Roi> = if has_mask {
            cache
                .roi_cache
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
        // A uniform spatial grid over the centers turns the per-pixel nearest-center
        // query into a small expanding-ring search instead of scanning every center,
        // which matters a lot once there are many seed objects.
        let n = centers.len();
        let mut center_pixels: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        let grid = CenterGrid::build(&centers, img_w, img_h);

        for y in 0..img_h {
            for x in 0..img_w {
                // Skip pixels outside the mask when a mask is configured.
                if has_mask && !mask_rois.iter().any(|mr| mr.is_part_of(x, y)) {
                    continue;
                }

                // Apply max_radius separately with <= so boundary pixels are included,
                // matching the filled-ellipse behaviour of the C++ reference.
                if let Some((nearest, dist_sq)) =
                    grid.nearest(&centers, x as f64, y as f64, max_dist_sq)
                {
                    if dist_sq <= max_dist_sq {
                        center_pixels[nearest].push((x, y));
                    }
                }
            }
        }

        // --- Phase 4: Build one ROI per center from its assigned pixel set ---
        let plane = ctx.image.plane().unwrap_or(ImagePlane {
            z: -1,
            c: -1,
            t: -1,
        });

        // Collect new ROIs before mutating the cache.
        let mut new_rois: Vec<Roi> = Vec::new();

        for (i, pixels) in center_pixels.iter().enumerate() {
            if pixels.is_empty() {
                continue;
            }

            let x_min = pixels.iter().map(|(x, _)| *x).min().unwrap();
            let y_min = pixels.iter().map(|(_, y)| *y).min().unwrap();
            // bbox convention: bbox[2]/[3] are INCLUSIVE maximum pixel coordinates,
            // matching the convention used by extract_rois and the renderer.
            // The mask stride is therefore (bbox[2] - bbox[0] + 1).
            let x_max = pixels.iter().map(|(x, _)| *x).max().unwrap();
            let y_max = pixels.iter().map(|(_, y)| *y).max().unwrap();
            let w = (x_max - x_min + 1) as usize;
            let h = (y_max - y_min + 1) as usize;

            let mut mask_data = BitVec::<u64, Lsb0>::repeat(false, w * h);
            let mut area = 0usize;
            let mut sum_x = 0u64;
            let mut sum_y = 0u64;
            let mut sum_x2 = 0u64;
            let mut sum_y2 = 0u64;
            let mut sum_xy = 0u64;

            for &(px, py) in pixels {
                let lx = (px - x_min) as usize;
                let ly = (py - y_min) as usize;
                mask_data.set(ly * w + lx, true);
                area += 1;
                sum_x += px as u64;
                sum_y += py as u64;
                sum_x2 += (px as u64) * (px as u64);
                sum_y2 += (py as u64) * (py as u64);
                sum_xy += (px as u64) * (py as u64);
            }

            // With inclusive bbox: touching the right/bottom edge means the max pixel
            // is the last column/row of the image (index img_w-1 / img_h-1).
            let touches_edge = x_min == 0 || y_min == 0 || x_max + 1 >= img_w || y_max + 1 >= img_h;

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
            let mut roi = Roi::new(RoiInit {
                id: ObjectId::next(),
                segmentation_class: SegmentationClass::MANUAL_ANNOTATED,
                bbox: [x_min, y_min, x_max, y_max],
                mask_data,
                area,
                plane,
                touches_edge,
                sum_x,
                sum_y,
                sum_x2,
                sum_y2,
                sum_xy,
                parent_id: Some(center_id.clone()),
                ..Default::default()
            });
            roi.add_object_class(self.output_class);
            new_rois.push(roi);
        }

        // --- Phase 5: Insert the new ROIs into the cache ---
        for roi in new_rois {
            cache.roi_cache.insert(roi.id.clone(), roi);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Voronoi"
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        ImageContainer, ImagePlane, ManagedImage,
        image::PixelSizes,
        pipeline::{
            pipeline::PipelineImageMeta, pipeline_cache::PipelineCache,
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
            ImageContainer::F32Gray(managed),
        )
        .unwrap()
    }

    fn make_filled_roi(id: u128, bbox: [u32; 4], class: ObjectClass) -> Roi {
        let [x_min, y_min, x_max, y_max] = bbox;
        // bbox uses inclusive convention: width = xmax - xmin + 1
        let w = (x_max - x_min + 1) as usize;
        let h = (y_max - y_min + 1) as usize;
        let area = w * h;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        let mut roi = Roi::new(RoiInit {
            id: ObjectId(id),
            bbox,
            mask_data,
            area,
            plane: ImagePlane::default(),
            ..Default::default()
        });
        roi.add_object_class(class);
        roi
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

    fn voronoi_rois(cache: &PipelineCache) -> Vec<&Roi> {
        cache
            .roi_cache
            .values()
            .filter(|r| r.has_object_class(&OUTPUT_CLASS))
            .collect()
    }

    fn run(v: &Voronoi, ctx: &mut PipelineContext, cache: &mut PipelineCache) {
        v.execute(ctx, cache).unwrap();
    }

    fn center_bbox(cx: u32, cy: u32) -> [u32; 4] {
        [cx - 1, cy - 1, cx + 1, cy + 1]
    }

    // --- Tests ---

    #[test]
    fn no_centers_produces_no_output() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        run(&default_voronoi(), &mut ctx, &mut cache);
        assert!(voronoi_rois(&cache).is_empty());
    }

    #[test]
    fn single_center_covers_full_image() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_rois(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 100);
    }

    #[test]
    fn two_centers_partition_image_without_overlap_or_gap() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(2, 5), CENTER_CLASS),
        );
        cache.roi_cache.insert(
            ObjectId(ID_B),
            make_filled_roi(ID_B, center_bbox(7, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        let regions = voronoi_rois(&cache);
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
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(10, 10), CENTER_CLASS),
        );

        run(
            &Voronoi {
                max_radius: 2.0,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_rois(&cache);
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
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );
        cache.roi_cache.insert(
            ObjectId(ID_MASK),
            make_filled_roi(ID_MASK, [2, 2, 7, 7], MASK_CLASS), // inclusive [2,7] = 6 wide → 6×6=36
        );

        run(
            &Voronoi {
                mask: MASK_CLASS,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_rois(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 36);
    }

    #[test]
    fn edge_exclusion_discards_border_touching_regions() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(
            &Voronoi {
                exclude_areas_at_the_edges: true,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        assert!(voronoi_rois(&cache).is_empty());
    }

    #[test]
    fn edge_exclusion_off_keeps_border_region() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, center_bbox(5, 5), CENTER_CLASS),
        );

        run(&default_voronoi(), &mut ctx, &mut cache);

        assert_eq!(voronoi_rois(&cache).len(), 1);
    }

    #[test]
    fn center_exclusion_discards_region_when_seed_outside_mask() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, [0, 4, 1, 6], CENTER_CLASS),
        );
        cache.roi_cache.insert(
            ObjectId(ID_MASK),
            make_filled_roi(ID_MASK, [2, 0, 4, 10], MASK_CLASS),
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

        assert!(voronoi_rois(&cache).is_empty());
    }

    #[test]
    fn center_exclusion_off_keeps_region_even_when_seed_outside_mask() {
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();
        cache.roi_cache.insert(
            ObjectId(ID_A),
            make_filled_roi(ID_A, [0, 4, 1, 6], CENTER_CLASS),
        );
        cache.roi_cache.insert(
            ObjectId(ID_MASK),
            make_filled_roi(ID_MASK, [2, 0, 3, 9], MASK_CLASS), // inclusive [2,3]×[0,9] = 2×10=20
        );

        run(
            &Voronoi {
                mask: MASK_CLASS,
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_rois(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 20);
    }

    #[test]
    fn center_filter_class_excludes_unmatched_centers() {
        const FILTER_CLASS: ObjectClass = ObjectClass::Valid(3);
        let mut ctx = make_ctx(10, 10);
        let mut cache = PipelineCache::default();

        let roi_a = make_filled_roi(ID_A, center_bbox(2, 5), CENTER_CLASS);
        let mut roi_b = make_filled_roi(ID_B, center_bbox(7, 5), CENTER_CLASS);
        roi_b.add_object_class(FILTER_CLASS);

        cache.roi_cache.insert(roi_a.id.clone(), roi_a);
        cache.roi_cache.insert(roi_b.id.clone(), roi_b);

        run(
            &Voronoi {
                center_filter_classes: vec![FILTER_CLASS],
                ..default_voronoi()
            },
            &mut ctx,
            &mut cache,
        );

        let regions = voronoi_rois(&cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 100);
    }

    /// Tiny deterministic LCG so the stress test below doesn't need a `rand` dependency.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    /// `CenterGrid::nearest` is a performance optimisation over a brute-force scan of
    /// every center; this checks it produces identical (index, tie-break) results to
    /// that brute-force scan across many random center layouts, not just the simple
    /// hand-picked layouts used by the other tests above.
    #[test]
    fn center_grid_matches_brute_force_for_many_random_centers() {
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

            let grid = CenterGrid::build(&centers, img_w, img_h);

            for y in 0..img_h {
                for x in 0..img_w {
                    let (gx, gy) = grid
                        .nearest(&centers, x as f64, y as f64, f64::MAX)
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
                        gx, brute_idx,
                        "trial {trial} pixel ({x},{y}): grid picked center {gx} but brute force picked {brute_idx}"
                    );
                    assert_eq!(gy, brute_dist_sq);
                }
            }
        }
    }
}
