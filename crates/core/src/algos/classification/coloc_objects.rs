//! # Colocalization Module
//!
//! Provides functionality for calculating spatial colocalization and overlap relationships
//! between different object classes across Regions of Interest (ROIs).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-05-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module implements spatial intersection algorithms that detect overlaps between
//! specified object classes, establishing relational links or creating new intersection
//! regions as distinct ROIs based on configuration settings.
//!
use crate::{
    algos::ImageAlgorithm,
    object::{Object, ObjectInit},
    spatial_grid::BboxGrid,
};
use evanalyzer_cfg::core_types::{
    InternalErrors, ObjectClass, ObjectId, SegmentationClass, SizeUnits,
};
use indexmap::IndexMap;
use log::{debug, info, warn};
use macros::CommandsMeta;
use std::sync::Arc;

/// How many partners an object may coloc with at once.
#[derive(CommandsMeta, Debug, PartialEq)]
pub enum ColocMultiplicity {
    /// Every object keeps only its single best-overlap match per other class -
    /// e.g. a small object sitting on the border of two larger ones picks
    /// exactly one (the larger overlap; ties go to the lower `ObjectId`).
    #[cmdsmeta(display_name = "No multi coloc (1:1)")]
    OneToOne,

    /// Every object may coloc with every overlapping partner meeting `min_coloc_area`.
    #[cmdsmeta(display_name = "Allow multi coloc")]
    ManyToMany,

    /// Only objects of these classes may coloc with more than one partner;
    /// every other class in `classes_to_coloc` is capped to its single
    /// best-overlap match. E.g. with `classes_to_coloc: [Cell, Spot]` and
    /// `MultiFor([Cell])`, a cell can coloc with any number of spots, but
    /// each spot colocs with exactly one cell (the one it overlaps most).
    #[cmdsmeta(display_name = "Multi coloc only for selected")]
    MultiFor(Vec<ObjectClass>),
}

/// Calculates spatial colocalization and intersections between specified object classes.
///
/// This command scans the object cache, groups objects by their designated classes,
/// and performs spatial overlap analysis. It records colocalization relationships
/// between intersecting entities and can optionally generate new child ROIs representing
/// the precise intersection regions.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
pub struct Colocalization {
    /// Theses are the classes the coloclization should be calculated for
    pub classes_to_coloc: Vec<ObjectClass>,

    /// Optional additional label filters.
    ///
    /// Only classes which matches all of these filters are used for coloc calculation
    #[cmdsmeta(visible = false)]
    pub filter_classes: Vec<ObjectClass>,

    /// Class of the overlapping area if needed
    ///
    /// If defined the overlapping coloc area is added as new object and labeled with this class
    pub class_for_overlapping_areas: ObjectClass,

    /// How many partners an object may coloc with at once.
    #[cmdsmeta(default = ColocMultiplicity::OneToOne)]
    pub multiplicity: ColocMultiplicity,

    /// Size unit for the minimum coloc area size
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub size_unit: SizeUnits,

    /// Minimum overlapping area size to count objects as coloc
    #[cmdsmeta(default = 0.0)]
    pub min_coloc_area: f32,

    /// Classes an object must NOT overlap to be considered colocalized.
    ///
    /// Exclude_classes is a blocklist — "even if an object matches everything else, throw it out if it also touches one of these classes."
    /// Concretely: you're looking for objects that overlap every class in classes_to_coloc (say Class 1 and Class 2).
    /// Without exclude_classes, any object satisfying that gets recorded as colocalized.
    /// With exclude_classes: an object that overlaps 1 and 2 but also touches Class 3 gets dropped entirely - no colocalization recorded for it at all, even though it passed the 1-and-2 check.
    /// So it's a "match A and B, but not C" filter
    ///
    /// Example: "cells colocalizing with both a nucleus stain and a membrane stain, but exclude any that also overlap a dead-cell marker."
    #[cmdsmeta(optional = true)]
    pub exclude_classes: Vec<ObjectClass>,
}

impl ImageAlgorithm for Colocalization {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        // If there aren't at least two classes, colocalization is impossible.
        if self.classes_to_coloc.len() < 2 {
            return Ok(());
        }

        let px_sizes = ctx.pixel_sizes();
        let pixel_area_nm2 = px_sizes.px_size_x * px_sizes.px_size_y;
        let min_area_px = self.size_unit.to_pixel(self.min_coloc_area, pixel_area_nm2);

        // --- PHASE 1: Group ObjectIds by their specific ObjectClass ---
        let mut class_buckets: std::collections::HashMap<ObjectClass, Vec<ObjectId>> =
            std::collections::HashMap::new();

        for target_class in &self.classes_to_coloc {
            let mut matched_ids = Vec::new();

            for object in cache.object_cache.values() {
                let passes_filter = self
                    .filter_classes
                    .iter()
                    .all(|f_class| object.has_object_class(f_class));
                if !passes_filter {
                    continue;
                }

                if object.has_object_class(target_class) {
                    matched_ids.push(object.id.clone());
                }
            }

            class_buckets.insert(target_class.clone(), matched_ids);
        }

        // Spatial index per class bucket, used below to prune the "does this anchor
        // overlap an object of this other class" scan down from a full bucket walk to a
        // small candidate list - see `BboxGrid` docs. Built once per class and reused
        // across every anchor, since a class's bucket (and its objects' positions) don't
        // change during this pass.
        let class_grids: std::collections::HashMap<ObjectClass, BboxGrid> = class_buckets
            .iter()
            .map(|(class, ids)| (*class, BboxGrid::build(ids, cache)))
            .collect();

        // ROIs carrying any of `exclude_classes` (subject to the same `filter_classes`).
        // An anchor overlapping one of these by at least `min_coloc_area` is dropped
        // entirely, regardless of how well it matches `classes_to_coloc`.
        let exclude_ids: Vec<ObjectId> = if self.exclude_classes.is_empty() {
            Vec::new()
        } else {
            cache
                .object_cache
                .values()
                .filter(|object| {
                    self.filter_classes
                        .iter()
                        .all(|f_class| object.has_object_class(f_class))
                        && self
                            .exclude_classes
                            .iter()
                            .any(|c| object.has_object_class(c))
                })
                .map(|object| object.id.clone())
                .collect()
        };
        let exclude_grid = BboxGrid::build(&exclude_ids, cache);

        let mut overlap_matches: std::collections::HashMap<
            ObjectId,
            IndexMap<ObjectClass, Vec<ObjectId>>,
        > = std::collections::HashMap::new();
        let mut new_objects: Vec<Object> = Vec::new();
        // Tracks which sorted sets of object IDs already produced an overlap object (avoids duplicates).
        let mut processed_combinations: std::collections::HashSet<Vec<ObjectId>> =
            std::collections::HashSet::new();

        // Precompute, for every object of a class that is capped to a single partner
        // (see `allows_multi`), its single best-overlap match per other class. Both
        // directions of a relation must agree on this pick: an unrestricted anchor
        // (e.g. a Cell under `MultiFor([Cell])`) may only list a capped partner (e.g.
        // a Spot) if that partner's own single pick points back at this exact anchor -
        // otherwise the same Spot would show up under every Cell it merely touches,
        // even though it only "colocs" with the one it overlaps most.
        let mut best_partner: std::collections::HashMap<(ObjectId, ObjectClass), ObjectId> =
            std::collections::HashMap::new();
        for (source_idx, source_class) in self.classes_to_coloc.iter().enumerate() {
            if self.allows_multi(source_class) {
                continue;
            }
            let source_ids = match class_buckets.get(source_class) {
                Some(b) => b,
                None => continue,
            };
            for (target_idx, target_class) in self.classes_to_coloc.iter().enumerate() {
                if target_idx == source_idx {
                    continue;
                }
                let target_grid = match class_grids.get(target_class) {
                    Some(g) => g,
                    None => continue,
                };
                for source_id in source_ids {
                    let source_object = match cache.object_cache.get(source_id) {
                        Some(o) => o,
                        None => continue,
                    };
                    let candidates = target_grid.candidates(source_object.bbox);
                    if let Some(best_id) = Self::best_overlap(
                        source_object,
                        source_id,
                        &candidates,
                        cache,
                        min_area_px,
                    ) {
                        best_partner.insert((source_id.clone(), *target_class), best_id);
                    }
                }
            }
        }

        // --- PHASE 2: For each object, check whether it overlaps at least one object from every
        //              other class. Only then is it considered colocalized. ---
        for (class_idx, anchor_class) in self.classes_to_coloc.iter().enumerate() {
            let anchor_ids = match class_buckets.get(anchor_class) {
                Some(b) => b.clone(),
                None => continue,
            };

            'anchor: for anchor_id in &anchor_ids {
                let anchor_object = match cache.object_cache.get(anchor_id) {
                    Some(r) => r,
                    None => continue,
                };

                // For every other class collect the IDs of ROIs that spatially overlap the anchor
                // with sufficient area.
                let mut class_matches: IndexMap<ObjectClass, Vec<ObjectId>> = IndexMap::new();

                for (other_idx, other_class) in self.classes_to_coloc.iter().enumerate() {
                    if other_idx == class_idx {
                        continue;
                    }
                    let other_grid = match class_grids.get(other_class) {
                        Some(g) => g,
                        None => continue 'anchor,
                    };
                    let other_candidates = other_grid.candidates(anchor_object.bbox);

                    // Compute the actual overlap and keep only entries meeting min_area_px.
                    // Each entry is (ObjectId, overlap_area_in_pixels).
                    let mut candidates: Vec<(ObjectId, usize)> = other_candidates
                        .iter()
                        .filter(|other_id| *other_id != anchor_id)
                        .filter_map(|other_id| {
                            cache
                                .object_cache
                                .get(other_id)
                                .and_then(|r| anchor_object.overlaps(r))
                                .filter(|intersection| intersection.area >= min_area_px)
                                .map(|intersection| (other_id.clone(), intersection.area))
                        })
                        .collect();

                    if !self.allows_multi(anchor_class) {
                        // Keep only the single best match (largest overlap area) per
                        // class - ties (equal overlap area, e.g. an object sitting
                        // exactly on the border between two equally-overlapping
                        // partners) go to the lower ObjectId, so the pick is
                        // deterministic rather than depending on cache iteration order.
                        candidates = best_partner
                            .get(&(anchor_id.clone(), *other_class))
                            .and_then(|best_id| {
                                candidates.iter().find(|(id, _)| id == best_id).cloned()
                            })
                            .into_iter()
                            .collect();
                    }

                    if !self.allows_multi(other_class) {
                        // `other_class` is itself capped to a single partner: only keep a
                        // candidate if this anchor is also that candidate's own single
                        // best match, so a capped object never ends up listed under more
                        // than one anchor.
                        candidates.retain(|(other_id, _)| {
                            best_partner.get(&(other_id.clone(), *anchor_class)) == Some(anchor_id)
                        });
                    }

                    let overlapping: Vec<ObjectId> =
                        candidates.into_iter().map(|(id, _)| id).collect();

                    if overlapping.is_empty() {
                        continue 'anchor;
                    }

                    class_matches.insert(*other_class, overlapping);
                }

                // The anchor satisfies classes_to_coloc, but drop it if it also overlaps
                // (by at least min_coloc_area) any object from an excluded class.
                if exclude_grid
                    .candidates(anchor_object.bbox)
                    .iter()
                    .any(|excl_id| {
                        excl_id != anchor_id
                            && cache
                                .object_cache
                                .get(excl_id)
                                .and_then(|r| anchor_object.overlaps(r))
                                .is_some_and(|intersection| intersection.area >= min_area_px)
                    })
                {
                    continue 'anchor;
                }

                // The anchor object overlaps with at least one object from every other class.
                for (cls, ids) in &class_matches {
                    overlap_matches
                        .entry(anchor_id.clone())
                        .or_default()
                        .entry(*cls)
                        .or_default()
                        .extend(ids.iter().cloned());
                }

                // Compute the N-way intersection area (chain of pairwise intersections).
                if let ObjectClass::Valid(overlap_class_id) = self.class_for_overlapping_areas {
                    let overlap_class = ObjectClass::Valid(overlap_class_id);
                    let mut states: Vec<(Object, Vec<ObjectId>)> = Vec::new();

                    let mut matches_iter = class_matches.iter();
                    if let Some((_, first_ids)) = matches_iter.next() {
                        // Seed: anchor ∩ each object from the first other class
                        for first_id in first_ids {
                            if let Some(first_object) = cache.object_cache.get(first_id) {
                                if let Some(intersection) = anchor_object.overlaps(first_object) {
                                    states.push((
                                        intersection,
                                        vec![anchor_id.clone(), first_id.clone()],
                                    ));
                                }
                            }
                        }

                        // Extend each state by intersecting with ROIs from remaining classes
                        for (_, next_ids) in matches_iter {
                            let mut next_states = Vec::new();
                            for (current_object, current_ids) in &states {
                                for next_id in next_ids {
                                    if let Some(next_object) = cache.object_cache.get(next_id) {
                                        if let Some(intersection) =
                                            current_object.overlaps(next_object)
                                        {
                                            let mut new_ids = current_ids.clone();
                                            new_ids.push(next_id.clone());
                                            next_states.push((intersection, new_ids));
                                        }
                                    }
                                }
                            }
                            states = next_states;
                        }

                        // Each unique set of contributing object IDs yields exactly one overlap object.
                        for (mut intersection, mut ids) in states {
                            ids.sort_by_key(|ObjectId(n)| *n);
                            if processed_combinations.insert(ids) {
                                intersection.add_object_class(overlap_class);
                                // `Object::overlaps()` only derives geometry from the parent masks
                                // and never sampled pixel data, and this object never passes
                                // through `ExtractObjects` (the step that normally measures
                                // intensities), so it needs its own measurement pass here.
                                intersection.intensities =
                                    intersection.measure_intensities(ctx, cache);
                                new_objects.push(intersection);
                            }
                        }
                    }
                }
            }
        }

        // --- PHASE 3: Write Back Results ---
        // A object carrying more than one of `classes_to_coloc` is anchored once per
        // class it belongs to (see PHASE 2), so the same partner id can be
        // recomputed and `.extend()`-ed into `found_matches` more than once for a
        // given target class. Dedup per class before writing back, mirroring
        // `Object::add_colocalizing_object`.
        for (object_id, mut found_matches) in overlap_matches {
            for ids in found_matches.values_mut() {
                ids.sort();
                ids.dedup();
            }
            if let Some(object) = cache.object_cache.get_mut(&object_id) {
                object.colocalized_with.extend(found_matches);
            }
        }
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Colocalization"
    }
}

impl Colocalization {
    /// Whether an anchor object of `anchor_class` may keep more than one
    /// overlapping partner per other class, or must be capped to its single
    /// best-overlap match (see [`ColocMultiplicity`]).
    fn allows_multi(&self, anchor_class: &ObjectClass) -> bool {
        match &self.multiplicity {
            ColocMultiplicity::OneToOne => false,
            ColocMultiplicity::ManyToMany => true,
            ColocMultiplicity::MultiFor(classes) => classes.contains(anchor_class),
        }
    }

    /// Finds the single overlapping object in `bucket` with the largest overlap
    /// area against `object` (>= `min_area_px`), ties broken toward the lower
    /// `ObjectId` for a deterministic pick independent of cache iteration order.
    fn best_overlap(
        object: &Object,
        exclude_id: &ObjectId,
        bucket: &[ObjectId],
        cache: &crate::pipeline::pipeline_cache::PipelineCache,
        min_area_px: usize,
    ) -> Option<ObjectId> {
        bucket
            .iter()
            .filter(|other_id| *other_id != exclude_id)
            .filter_map(|other_id| {
                cache
                    .object_cache
                    .get(other_id)
                    .and_then(|r| object.overlaps(r))
                    .filter(|intersection| intersection.area >= min_area_px)
                    .map(|intersection| (other_id.clone(), intersection.area))
            })
            .max_by(|(id_a, area_a), (id_b, area_b)| {
                area_a.cmp(area_b).then_with(|| id_b.cmp(id_a))
            })
            .map(|(id, _)| id)
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

    fn make_ctx() -> PipelineContext {
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let img = Image::<f32, 1, CpuAllocator>::new(size, vec![0.0f32], CpuAllocator).unwrap();
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

    fn make_filled_object(
        id: u128,
        bbox: [u32; 4],
        plane: ImagePlane,
        class: ObjectClass,
    ) -> Object {
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
            plane,
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    const CLASS_A: ObjectClass = ObjectClass::Valid(1);
    const CLASS_B: ObjectClass = ObjectClass::Valid(2);
    const CLASS_C: ObjectClass = ObjectClass::Valid(3);
    const CLASS_OVERLAP: ObjectClass = ObjectClass::Valid(99);

    // object IDs are set far above the ObjectId::next() counter range so that
    // counter-generated IDs from overlaps() never collide with test IDs in parallel runs.
    const ID_A: u128 = 100_000;
    const ID_B: u128 = 200_000;
    const ID_C: u128 = 300_000;

    fn run(coloc: &Colocalization, cache: &mut PipelineCache) {
        coloc.execute(&mut make_ctx(), cache).unwrap();
    }

    /// Manual repro/benchmark, originally written to demonstrate that `execute` scanned,
    /// for every anchor object, *every* object of every other class-to-coloc via a plain
    /// nested loop with no spatial index at all - O(class_a_count * class_b_count)
    /// regardless of how many pairs actually overlap. `MANY_TO_MANY` colocalization is
    /// exactly the production pipeline shape from a user report (a pipeline chaining 12
    /// sequential MANY_TO_MANY steps): a noisy/low-threshold channel producing tens of
    /// thousands of tiny spurious objects on one image could stall for over an hour.
    /// Fixed by `BboxGrid` (see `spatial_grid.rs`), which prunes each anchor's per-class
    /// scan down to a small candidate list instead of the whole bucket. Left in place as a
    /// regression benchmark - a future change to `execute` that accidentally drops back to
    /// a full scan should show up here as scaling roughly quadratically again instead of
    /// near-linearly. Not asserted against a baseline (timing isn't deterministic across
    /// machines), so it must still be read manually. Run manually with:
    ///   cargo test --release -p evanalyzer_core --features ai -- --ignored \
    ///     repro_many_to_many_coloc_object_count_scaling --nocapture
    #[test]
    #[ignore]
    fn repro_many_to_many_coloc_object_count_scaling() {
        // xorshift32 - deterministic, no external dep needed.
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

        // Small (3x3px), scattered, non-overlapping-within-class objects -
        // real EV/puncta detections are small and sparse; what's under test
        // is the class_a x class_b scan cost, not overlap resolution itself.
        fn scattered_objects(
            n: u32,
            class: ObjectClass,
            id_base: u128,
            span: u32,
            seed: u32,
        ) -> Vec<Object> {
            let mut rng = Rng(seed.max(1));
            let mut objects = Vec::with_capacity(n as usize);
            for i in 0..n {
                // Stride placement (not fully random) guarantees no
                // within-class overlap regardless of n, isolating the
                // cross-class scan cost from any O(n^2) contribution the
                // *input generation* itself might otherwise add.
                let cell = i;
                let x = (cell % span) * 4;
                let y = (cell / span) * 4;
                let jitter = rng.range(2);
                let x = x + jitter;
                let y = y + jitter;
                objects.push(make_filled_object(
                    id_base + i as u128,
                    [x, y, x + 2, y + 2],
                    ImagePlane::default(),
                    class,
                ));
            }
            objects
        }

        for &n in &[500u32, 2_000, 8_000, 20_000] {
            let span = (n as f64).sqrt().ceil() as u32 + 1;
            let mut cache = PipelineCache::default();
            for o in scattered_objects(n, CLASS_A, 100_000, span, 111) {
                cache.object_cache.insert(o.id.clone(), o);
            }
            for o in scattered_objects(n, CLASS_B, 900_000_000, span, 222) {
                cache.object_cache.insert(o.id.clone(), o);
            }

            let coloc = Colocalization {
                classes_to_coloc: vec![CLASS_A, CLASS_B],
                filter_classes: vec![],
                exclude_classes: vec![],
                class_for_overlapping_areas: CLASS_OVERLAP,
                multiplicity: ColocMultiplicity::ManyToMany,
                size_unit: SizeUnits::Pixels,
                min_coloc_area: 0.0,
            };

            let t0 = std::time::Instant::now();
            run(&coloc, &mut cache);
            let elapsed = t0.elapsed();

            println!(
                "class_a={n:6} class_b={n:6} ({:>12} pairs scanned) -> {elapsed:8.2?}",
                n as u64 * n as u64
            );
        }
    }

    #[test]
    fn fewer_than_two_classes_is_noop() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        cache.object_cache.insert(object.id.clone(), object);

        run(&coloc, &mut cache);

        assert!(
            cache
                .object_cache
                .values()
                .all(|r| r.colocalized_with.is_empty())
        );
    }

    #[test]
    fn overlapping_objects_are_colocalized() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        // object A: [0,0,4,4], object B: [2,2,6,6] → overlap [2,2,4,4]
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 6, 6], ImagePlane::default(), CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.contains_key(&CLASS_B),
            "A should coloc with B"
        );
        assert!(a.colocalized_with[&CLASS_B].contains(&ObjectId(ID_B)));

        let b = cache.object_cache.get(&ObjectId(ID_B)).unwrap();
        assert!(
            b.colocalized_with.contains_key(&CLASS_A),
            "B should coloc with A"
        );
        assert!(b.colocalized_with[&CLASS_A].contains(&ObjectId(ID_A)));
    }

    #[test]
    fn non_overlapping_objects_are_not_colocalized() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        // No shared bounding box region
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [10, 10, 14, 14], ImagePlane::default(), CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        assert!(
            cache
                .object_cache
                .values()
                .all(|r| r.colocalized_with.is_empty())
        );
    }

    #[test]
    fn objects_on_different_channels_are_colocalized() {
        // Cross-channel colocalization is the primary use case (e.g. DAPI channel vs GFP channel).
        // ROIs from different c-planes with matching XY extent must still colocalize.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let plane_ch0 = ImagePlane { z: 0, c: 0, t: 0 };
        let plane_ch1 = ImagePlane { z: 0, c: 1, t: 0 };
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], plane_ch0, CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 6, 6], plane_ch1, CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.contains_key(&CLASS_B),
            "cross-channel overlap must be detected"
        );
        let b = cache.object_cache.get(&ObjectId(ID_B)).unwrap();
        assert!(
            b.colocalized_with.contains_key(&CLASS_A),
            "cross-channel overlap must be detected"
        );
    }

    #[test]
    fn filter_class_excludes_objects_that_do_not_have_it() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![CLASS_C], // only ROIs also carrying CLASS_C participate
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        // object_a has CLASS_A but NOT CLASS_C → filtered out
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        // object_b has CLASS_B and CLASS_C → passes filter
        let mut object_b = make_filled_object(ID_B, [2, 2, 6, 6], ImagePlane::default(), CLASS_B);
        object_b.add_object_class(CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        // object_a is excluded from CLASS_A bucket so no colocalization occurs
        assert!(
            cache
                .object_cache
                .values()
                .all(|r| r.colocalized_with.is_empty())
        );
    }

    #[test]
    fn overlapping_area_is_added_as_new_object() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: CLASS_OVERLAP,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        // Inclusive bboxes: A covers [0,3]×[0,3], B covers [2,5]×[2,5] → overlap [2,3]×[2,3]=2×2=4 px
        let object_a = make_filled_object(ID_A, [0, 0, 3, 3], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 5, 5], ImagePlane::default(), CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        // Three ROIs total: the original two + the intersection object
        assert_eq!(
            cache.object_cache.len(),
            3,
            "intersection object should be added"
        );
        let overlap_object = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&CLASS_OVERLAP))
            .expect("intersection object tagged with CLASS_OVERLAP not found");
        assert_eq!(overlap_object.bbox, [2, 2, 3, 3]);
        assert_eq!(overlap_object.area, 4);
    }

    #[test]
    fn overlapping_area_object_has_measured_intensities() {
        // Regression test: `Object::overlaps()` cannot sample pixel data (it only sees the two
        // parent masks), so the intersection object needs its own intensity measurement pass.
        const CHANNEL: i32 = 0;
        const VALUE: f32 = 5.0;
        let size = ImageSize {
            width: 10,
            height: 10,
        };

        // ctx's own image content is irrelevant to intensity sampling; only its size/tile_offset matter.
        let ctx_image =
            Image::<f32, 1, CpuAllocator>::new(size, vec![0.0f32; 100], CpuAllocator).unwrap();
        let mut ctx = PipelineContext::new_from_image(
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
            ImageContainer::F32Gray(ManagedImage {
                data: ctx_image,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })
            .into(),
        )
        .unwrap();

        let mut cache = PipelineCache::default();
        // Constant-value channel so the expected sum/avg/min/max are trivial to compute.
        let channel_img =
            Image::<f32, 1, CpuAllocator>::new(size, vec![VALUE; 100], CpuAllocator).unwrap();
        cache.image_cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: channel_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })),
            CHANNEL,
        );

        // Inclusive bboxes: A covers [0,5]×[0,5], B covers [3,8]×[3,8] → overlap [3,5]×[3,5] = 3×3 = 9 px
        let object_a = make_filled_object(ID_A, [0, 0, 5, 5], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [3, 3, 8, 8], ImagePlane::default(), CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: CLASS_OVERLAP,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        coloc.execute(&mut ctx, &mut cache).unwrap();

        let overlap_object = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&CLASS_OVERLAP))
            .expect("intersection object tagged with CLASS_OVERLAP not found");
        assert_eq!(overlap_object.area, 9);
        let intensity = overlap_object
            .intensities
            .get(&CHANNEL)
            .expect("overlap object should have measured intensities for the channel");
        assert_eq!(intensity.sum_intensity, 9.0 * VALUE as f64);
        assert_eq!(intensity.avg_intensity, VALUE);
        assert_eq!(intensity.min_intensity, VALUE);
        assert_eq!(intensity.max_intensity, VALUE);
    }

    // --- 3-class tests ---

    #[test]
    fn three_class_all_overlap_full_colocalization() {
        // A=[0,0,6,6], B=[2,2,8,8], C=[4,4,10,10] - every pair overlaps.
        // Every object must be recorded as colocalized with both other classes.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 6, 6], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 8, 8], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [4, 4, 10, 10], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.contains_key(&CLASS_B),
            "A must coloc with B"
        );
        assert!(
            a.colocalized_with.contains_key(&CLASS_C),
            "A must coloc with C"
        );
        assert!(a.colocalized_with[&CLASS_B].contains(&ObjectId(ID_B)));
        assert!(a.colocalized_with[&CLASS_C].contains(&ObjectId(ID_C)));

        let b = cache.object_cache.get(&ObjectId(ID_B)).unwrap();
        assert!(
            b.colocalized_with.contains_key(&CLASS_A),
            "B must coloc with A"
        );
        assert!(
            b.colocalized_with.contains_key(&CLASS_C),
            "B must coloc with C"
        );
        assert!(b.colocalized_with[&CLASS_A].contains(&ObjectId(ID_A)));
        assert!(b.colocalized_with[&CLASS_C].contains(&ObjectId(ID_C)));

        let c = cache.object_cache.get(&ObjectId(ID_C)).unwrap();
        assert!(
            c.colocalized_with.contains_key(&CLASS_A),
            "C must coloc with A"
        );
        assert!(
            c.colocalized_with.contains_key(&CLASS_B),
            "C must coloc with B"
        );
        assert!(c.colocalized_with[&CLASS_A].contains(&ObjectId(ID_A)));
        assert!(c.colocalized_with[&CLASS_B].contains(&ObjectId(ID_B)));
    }

    #[test]
    fn three_class_nway_overlap_area_correct() {
        // Inclusive bboxes: A=[0,5]×[0,5], B=[2,7]×[2,7], C=[4,9]×[4,9]
        // 3-way intersection = [4,5]×[4,5] = 2×2 = 4 pixels, bbox [4,4,5,5].
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: CLASS_OVERLAP,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 5, 5], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 7, 7], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [4, 4, 9, 9], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        let overlap_objects: Vec<_> = cache
            .object_cache
            .values()
            .filter(|r| r.has_object_class(&CLASS_OVERLAP))
            .collect();
        assert_eq!(
            overlap_objects.len(),
            1,
            "exactly one 3-way overlap object should be added"
        );
        assert_eq!(
            overlap_objects[0].bbox,
            [4, 4, 5, 5],
            "3-way bbox should be [4,4,5,5]"
        );
        assert_eq!(overlap_objects[0].area, 4, "3-way area should be 4 pixels");
    }

    #[test]
    fn three_class_hub_spoke_only_hub_colocalized() {
        // A=[0,0,10,10] (hub), B=[1,1,4,4] (top-left), C=[6,6,9,9] (bottom-right)
        // A overlaps both B and C, but B and C don't overlap each other.
        // → only A is fully colocalized; B and C are not.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 10, 10], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [1, 1, 4, 4], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [6, 6, 9, 9], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.contains_key(&CLASS_B),
            "A overlaps both → should be colocalized"
        );
        assert!(
            a.colocalized_with.contains_key(&CLASS_C),
            "A overlaps both → should be colocalized"
        );

        let b = cache.object_cache.get(&ObjectId(ID_B)).unwrap();
        assert!(
            b.colocalized_with.is_empty(),
            "B doesn't overlap C → must not be colocalized"
        );

        let c = cache.object_cache.get(&ObjectId(ID_C)).unwrap();
        assert!(
            c.colocalized_with.is_empty(),
            "C doesn't overlap B → must not be colocalized"
        );
    }

    #[test]
    fn three_class_chain_only_middle_colocalized() {
        // A=[0,0,5,5], B=[3,3,8,8], C=[6,6,11,11]
        // A∩B=[3,3,5,5] ✓,  B∩C=[6,6,8,8] ✓,  A∩C=∅
        // → B colocalized (overlaps A and C); A and C are not.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 5, 5], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [3, 3, 8, 8], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [6, 6, 11, 11], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.is_empty(),
            "A doesn't overlap C → must not be colocalized"
        );

        let b = cache.object_cache.get(&ObjectId(ID_B)).unwrap();
        assert!(
            b.colocalized_with.contains_key(&CLASS_A),
            "B overlaps both → should be colocalized"
        );
        assert!(
            b.colocalized_with.contains_key(&CLASS_C),
            "B overlaps both → should be colocalized"
        );

        let c = cache.object_cache.get(&ObjectId(ID_C)).unwrap();
        assert!(
            c.colocalized_with.is_empty(),
            "C doesn't overlap A → must not be colocalized"
        );
    }

    #[test]
    fn multi_class_anchor_does_not_duplicate_partner_ids() {
        // X carries both CLASS_A and CLASS_B, so it is anchored twice (once
        // per bucket it belongs to). Both passes independently re-derive a
        // match against CLASS_C (via Y), which used to get `.extend()`-ed
        // into X's CLASS_C partner list twice — see the bug this guards
        // against, where the coloc-detail flat table showed an extra row for
        // the same object/partner pair.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();

        const ID_X: u128 = 400_000;
        const ID_W: u128 = 500_000;
        const ID_Z: u128 = 600_000;
        const ID_Y: u128 = 700_000;

        // X belongs to both A and B.
        let mut x = make_filled_object(ID_X, [0, 0, 5, 5], ImagePlane::default(), CLASS_A);
        x.add_object_class(CLASS_B);
        // W (class A) and Z (class B) each overlap X, so both of X's anchor
        // passes (as an A member and as a B member) find a partner for the
        // *other* target class and pass the "must overlap every other class"
        // check.
        let w = make_filled_object(ID_W, [3, 3, 8, 8], ImagePlane::default(), CLASS_A);
        let z = make_filled_object(ID_Z, [0, 0, 2, 2], ImagePlane::default(), CLASS_B);
        // Y (class C) overlaps X — this is the partner that both of X's
        // anchor passes independently rediscover for CLASS_C.
        let y = make_filled_object(ID_Y, [4, 4, 9, 9], ImagePlane::default(), CLASS_C);

        cache.object_cache.insert(x.id.clone(), x);
        cache.object_cache.insert(w.id.clone(), w);
        cache.object_cache.insert(z.id.clone(), z);
        cache.object_cache.insert(y.id.clone(), y);

        run(&coloc, &mut cache);

        let x = cache.object_cache.get(&ObjectId(ID_X)).unwrap();
        assert!(
            x.colocalized_with.contains_key(&CLASS_C),
            "X must coloc with C"
        );
        assert_eq!(
            x.colocalized_with[&CLASS_C],
            vec![ObjectId(ID_Y)],
            "Y must be recorded exactly once under CLASS_C, not duplicated"
        );
    }

    // --- exclude_classes tests ---

    #[test]
    fn exclude_classes_drops_anchor_that_also_overlaps_excluded_class() {
        // A=[0,0,6,6], B=[2,2,8,8] (A∩B → would colocalize), C=[4,4,10,10] (excluded,
        // overlaps both). A and B must NOT be colocalized because each also overlaps C.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![CLASS_C],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 6, 6], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 8, 8], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [4, 4, 10, 10], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        assert!(
            cache
                .object_cache
                .values()
                .all(|r| r.colocalized_with.is_empty()),
            "A and B both also overlap excluded class C, so neither should colocalize"
        );
    }

    #[test]
    fn exclude_classes_keeps_anchor_that_does_not_overlap_excluded_class() {
        // A=[0,0,4,4], B=[2,2,6,6] (A∩B → colocalize), C=[20,20,24,24] (excluded,
        // doesn't overlap anything). A and B should colocalize normally.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![CLASS_C],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 6, 6], ImagePlane::default(), CLASS_B);
        let object_c = make_filled_object(ID_C, [20, 20, 24, 24], ImagePlane::default(), CLASS_C);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);
        cache.object_cache.insert(object_c.id.clone(), object_c);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            a.colocalized_with.contains_key(&CLASS_B),
            "A and B don't overlap the excluded class, so they should still colocalize"
        );
    }

    #[test]
    fn exclude_classes_empty_by_default_does_not_change_behavior() {
        // Same geometry as `three_class_all_overlap_full_colocalization` but with B
        // also acting as the (empty) exclude list - leaving it empty must be a no-op.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::ManyToMany,
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], ImagePlane::default(), CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 6, 6], ImagePlane::default(), CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&coloc, &mut cache);

        let a = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(a.colocalized_with.contains_key(&CLASS_B));
    }

    // --- ColocMultiplicity::MultiFor ---

    #[test]
    fn multi_for_lets_listed_class_coloc_multiple_while_capping_the_other_to_its_single_best_match()
    {
        // Cells (CLASS_A, listed in MultiFor) may coloc with any number of
        // spots; spots (CLASS_B, not listed) must each pick exactly one cell -
        // the "count spots per cell" use case, where a spot sitting on the
        // border between two cells must not be double-counted under both. This
        // also means a cell's own partner list must agree with the spot's pick:
        // a spot that chose a different cell must not also show up here,
        // otherwise summing spots-per-cell across all cells would overcount.
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::MultiFor(vec![CLASS_A]),
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let mut cache = PipelineCache::default();

        const ID_CELL_A: u128 = 900_001; // lower id - wins area ties
        const ID_CELL_B: u128 = 900_002;
        const ID_SPOT_BORDER: u128 = 900_003; // straddles both cells, equal overlap area
        const ID_SPOT_IN_A: u128 = 900_004;
        const ID_SPOT_IN_B: u128 = 900_005;

        // Cell A: [0,9]x[0,9]; Cell B: [10,19]x[0,9] - adjacent, non-overlapping.
        let cell_a = make_filled_object(ID_CELL_A, [0, 0, 9, 9], ImagePlane::default(), CLASS_A);
        let cell_b = make_filled_object(ID_CELL_B, [10, 0, 19, 9], ImagePlane::default(), CLASS_A);
        // Border spot [8,11]x[4,5]: overlaps A in x=[8,9] (2 cols) and B in
        // x=[10,11] (2 cols), same y=[4,5] range both times -> 2x2=4 px each,
        // an exact tie.
        let spot_border = make_filled_object(
            ID_SPOT_BORDER,
            [8, 4, 11, 5],
            ImagePlane::default(),
            CLASS_B,
        );
        let spot_in_a =
            make_filled_object(ID_SPOT_IN_A, [2, 2, 3, 3], ImagePlane::default(), CLASS_B);
        let spot_in_b =
            make_filled_object(ID_SPOT_IN_B, [15, 2, 16, 3], ImagePlane::default(), CLASS_B);

        cache.object_cache.insert(cell_a.id.clone(), cell_a);
        cache.object_cache.insert(cell_b.id.clone(), cell_b);
        cache
            .object_cache
            .insert(spot_border.id.clone(), spot_border);
        cache.object_cache.insert(spot_in_a.id.clone(), spot_in_a);
        cache.object_cache.insert(spot_in_b.id.clone(), spot_in_b);

        run(&coloc, &mut cache);

        // Cell A (listed in MultiFor) sees every spot that picked it, including
        // the border spot (tie broken toward the lower ObjectId, Cell A).
        let cell_a = cache.object_cache.get(&ObjectId(ID_CELL_A)).unwrap();
        let mut a_spots = cell_a.colocalized_with[&CLASS_B].clone();
        a_spots.sort();
        let mut expected_a = vec![ObjectId(ID_SPOT_BORDER), ObjectId(ID_SPOT_IN_A)];
        expected_a.sort();
        assert_eq!(
            a_spots, expected_a,
            "Cell A must list every spot that picked it"
        );

        // Cell B only sees the spot that picked it (spot_in_b). The border spot
        // touches Cell B too, but it picked Cell A, so it must NOT also appear
        // here - otherwise the same spot would be double-counted across cells.
        let cell_b = cache.object_cache.get(&ObjectId(ID_CELL_B)).unwrap();
        let mut b_spots = cell_b.colocalized_with[&CLASS_B].clone();
        b_spots.sort();
        let expected_b = vec![ObjectId(ID_SPOT_IN_B)];
        assert_eq!(
            b_spots, expected_b,
            "Cell B must not list a spot that picked a different cell"
        );

        // The border spot overlaps both cells by the exact same area, so it
        // picks exactly one partner - not both - breaking the tie toward the
        // lower ObjectId (Cell A).
        let spot_border = cache.object_cache.get(&ObjectId(ID_SPOT_BORDER)).unwrap();
        assert_eq!(
            spot_border.colocalized_with[&CLASS_A],
            vec![ObjectId(ID_CELL_A)],
            "an equal-overlap tie must resolve to the lower ObjectId, not both cells"
        );

        // Spots overlapping only one cell are unaffected by the cap.
        let spot_in_a = cache.object_cache.get(&ObjectId(ID_SPOT_IN_A)).unwrap();
        assert_eq!(
            spot_in_a.colocalized_with[&CLASS_A],
            vec![ObjectId(ID_CELL_A)]
        );
        let spot_in_b = cache.object_cache.get(&ObjectId(ID_SPOT_IN_B)).unwrap();
        assert_eq!(
            spot_in_b.colocalized_with[&CLASS_A],
            vec![ObjectId(ID_CELL_B)]
        );
    }

    #[test]
    fn multi_for_two_independent_spot_channels_each_coloc_with_exactly_one_cell() {
        // 5 cells (CLASS_A) and two independent spot channels (CLASS_B, CLASS_C),
        // combined via two separate Colocalization runs against the *same* cells -
        // mirroring a real pipeline where each spot channel gets its own coloc step.
        // With `MultiFor([CLASS_A])`, cells may coloc with any number of spots, but
        // each spot must pick exactly one cell. Layout:
        //
        //   Cell 0 [0,9]x[0,9]      - adjacent to Cell 1 along x=9/10 (B border)
        //   Cell 1 [10,19]x[0,9]    - adjacent to Cell 0 (B border) and Cell 4 (C border)
        //   Cell 4 [10,19]x[10,19]  - adjacent to Cell 1 along y=9/10 (C border)
        //   Cell 2 [40,49]x[0,9]    - isolated, only touched by channel B
        //   Cell 3 [60,69]x[0,9]    - isolated, only touched by channel C
        //
        // IDs are chosen so ties resolve deterministically: Cell 0 beats Cell 1 for
        // the B border, and Cell 4 beats Cell 1 for the C border (lower ObjectId
        // wins a tie), leaving Cell 1 as the loser of both.
        const ID_CELL_0: u128 = 910_000;
        const ID_CELL_1: u128 = 910_100; // deliberately highest id - loses both border ties
        const ID_CELL_2: u128 = 910_002;
        const ID_CELL_3: u128 = 910_003;
        const ID_CELL_4: u128 = 910_004;

        let mut cache = PipelineCache::default();
        let cell_0 = make_filled_object(ID_CELL_0, [0, 0, 9, 9], ImagePlane::default(), CLASS_A);
        let cell_1 = make_filled_object(ID_CELL_1, [10, 0, 19, 9], ImagePlane::default(), CLASS_A);
        let cell_2 = make_filled_object(ID_CELL_2, [40, 0, 49, 9], ImagePlane::default(), CLASS_A);
        let cell_3 = make_filled_object(ID_CELL_3, [60, 0, 69, 9], ImagePlane::default(), CLASS_A);
        let cell_4 =
            make_filled_object(ID_CELL_4, [10, 10, 19, 19], ImagePlane::default(), CLASS_A);
        for cell in [cell_0, cell_1, cell_2, cell_3, cell_4] {
            cache.object_cache.insert(cell.id.clone(), cell);
        }

        // --- Channel B: 15 spots -------------------------------------------
        let mut next_b_id = 920_000u128;
        let mut make_b = |bbox: [u32; 4]| {
            next_b_id += 1;
            make_filled_object(next_b_id, bbox, ImagePlane::default(), CLASS_B)
        };

        let mut b_direct_0 = Vec::new();
        for bbox in [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]] {
            let spot = make_b(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            b_direct_0.push(id);
        }
        let mut b_direct_1 = Vec::new();
        for bbox in [[12, 1, 12, 1], [13, 2, 13, 2], [14, 3, 14, 3]] {
            let spot = make_b(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            b_direct_1.push(id);
        }
        let mut b_direct_2 = Vec::new();
        for bbox in [[41, 1, 41, 1], [42, 2, 42, 2], [43, 3, 43, 3]] {
            let spot = make_b(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            b_direct_2.push(id);
        }
        // Border spots straddling Cell 0 / Cell 1 (x=9/10): 2 px overlap on each
        // side - an exact tie.
        let mut b_border = Vec::new();
        for bbox in [[8, 5, 11, 5], [8, 6, 11, 6], [8, 7, 11, 7]] {
            let spot = make_b(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            b_border.push(id);
        }
        let mut b_none = Vec::new();
        for bbox in [[90, 90, 90, 90], [91, 91, 91, 91], [92, 92, 92, 92]] {
            let spot = make_b(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            b_none.push(id);
        }

        // --- Channel C: 15 spots -------------------------------------------
        let mut next_c_id = 930_000u128;
        let mut make_c = |bbox: [u32; 4]| {
            next_c_id += 1;
            make_filled_object(next_c_id, bbox, ImagePlane::default(), CLASS_C)
        };

        let mut c_direct_0 = Vec::new();
        for bbox in [[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]] {
            let spot = make_c(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            c_direct_0.push(id);
        }
        let mut c_direct_1 = Vec::new();
        for bbox in [[16, 4, 16, 4], [17, 5, 17, 5], [18, 6, 18, 6]] {
            let spot = make_c(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            c_direct_1.push(id);
        }
        let mut c_direct_3 = Vec::new();
        for bbox in [[61, 1, 61, 1], [62, 2, 62, 2], [63, 3, 63, 3]] {
            let spot = make_c(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            c_direct_3.push(id);
        }
        // Border spots straddling Cell 1 / Cell 4 (y=9/10): 2 px overlap on each
        // side - an exact tie.
        let mut c_border = Vec::new();
        for bbox in [[12, 8, 12, 11], [13, 8, 13, 11], [14, 8, 14, 11]] {
            let spot = make_c(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            c_border.push(id);
        }
        let mut c_none = Vec::new();
        for bbox in [[95, 95, 95, 95], [96, 96, 96, 96], [97, 97, 97, 97]] {
            let spot = make_c(bbox);
            let id = spot.id.clone();
            cache.object_cache.insert(id.clone(), spot);
            c_none.push(id);
        }

        // Two independent runs - one per spot channel - both against the same
        // cells, as in a pipeline that colocs each spot channel with the cell
        // mask separately.
        let coloc_b = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_B],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::MultiFor(vec![CLASS_A]),
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        let coloc_c = Colocalization {
            classes_to_coloc: vec![CLASS_A, CLASS_C],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            multiplicity: ColocMultiplicity::MultiFor(vec![CLASS_A]),
            min_coloc_area: 0.0,
            size_unit: SizeUnits::Pixels,
        };
        run(&coloc_b, &mut cache);
        run(&coloc_c, &mut cache);

        let coloc_count = |cache: &PipelineCache, id: u128, class: ObjectClass| -> usize {
            cache
                .object_cache
                .get(&ObjectId(id))
                .and_then(|o| o.colocalized_with.get(&class))
                .map(|v| v.len())
                .unwrap_or(0)
        };

        // Cell 2 and Cell 3 are the "clean" single-channel cells.
        assert_eq!(
            coloc_count(&cache, ID_CELL_2, CLASS_B),
            3,
            "Cell 2 gets exactly its 3 direct B spots"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_2, CLASS_C),
            0,
            "Cell 2 is never touched by channel C"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_3, CLASS_C),
            3,
            "Cell 3 gets exactly its 3 direct C spots"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_3, CLASS_B),
            0,
            "Cell 3 is never touched by channel B"
        );

        // Cell 0 wins the B border tie (lower id than Cell 1): 3 direct + 3 border = 6.
        assert_eq!(
            coloc_count(&cache, ID_CELL_0, CLASS_B),
            6,
            "Cell 0 wins the B border tie"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_0, CLASS_C),
            3,
            "Cell 0 gets its 3 direct C spots, no C border"
        );

        // Cell 1 loses both border ties, left with just its direct spots.
        assert_eq!(
            coloc_count(&cache, ID_CELL_1, CLASS_B),
            3,
            "Cell 1 loses the B border tie to Cell 0"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_1, CLASS_C),
            3,
            "Cell 1 loses the C border tie to Cell 4"
        );

        // Cell 4 has no direct spots of its own; it wins the C border tie
        // (lower id than Cell 1).
        assert_eq!(
            coloc_count(&cache, ID_CELL_4, CLASS_B),
            0,
            "Cell 4 is never touched by channel B"
        );
        assert_eq!(
            coloc_count(&cache, ID_CELL_4, CLASS_C),
            3,
            "Cell 4 wins the C border tie"
        );

        // Sums across all cells must equal the number of spots that found a
        // partner (12 of 15 in each channel - the other 3 touch nothing), not
        // more - which is exactly the double-counting bug this test guards
        // against.
        let sum_b: usize = [ID_CELL_0, ID_CELL_1, ID_CELL_2, ID_CELL_3, ID_CELL_4]
            .iter()
            .map(|id| coloc_count(&cache, *id, CLASS_B))
            .sum();
        let sum_c: usize = [ID_CELL_0, ID_CELL_1, ID_CELL_2, ID_CELL_3, ID_CELL_4]
            .iter()
            .map(|id| coloc_count(&cache, *id, CLASS_C))
            .sum();
        assert_eq!(
            sum_b, 12,
            "sum of B-colocs across all cells must equal the 12 assigned B spots"
        );
        assert_eq!(
            sum_c, 12,
            "sum of C-colocs across all cells must equal the 12 assigned C spots"
        );

        // Every assigned spot picks exactly one cell; the "nothing" spots pick none.
        for id in b_direct_0
            .iter()
            .chain(&b_direct_1)
            .chain(&b_direct_2)
            .chain(&b_border)
        {
            let spot = cache.object_cache.get(id).unwrap();
            assert_eq!(
                spot.colocalized_with[&CLASS_A].len(),
                1,
                "every colocalized B spot must pick exactly one cell"
            );
        }
        for id in &b_none {
            let spot = cache.object_cache.get(id).unwrap();
            assert!(
                spot.colocalized_with.is_empty(),
                "a B spot touching nothing must not be colocalized"
            );
        }
        for id in c_direct_0
            .iter()
            .chain(&c_direct_1)
            .chain(&c_direct_3)
            .chain(&c_border)
        {
            let spot = cache.object_cache.get(id).unwrap();
            assert_eq!(
                spot.colocalized_with[&CLASS_A].len(),
                1,
                "every colocalized C spot must pick exactly one cell"
            );
        }
        for id in &c_none {
            let spot = cache.object_cache.get(id).unwrap();
            assert!(
                spot.colocalized_with.is_empty(),
                "a C spot touching nothing must not be colocalized"
            );
        }

        // The B border spots must all resolve to Cell 0 - not split, not
        // double-booked with Cell 1.
        for id in &b_border {
            let spot = cache.object_cache.get(id).unwrap();
            assert_eq!(spot.colocalized_with[&CLASS_A], vec![ObjectId(ID_CELL_0)]);
        }
        // The C border spots must all resolve to Cell 4.
        for id in &c_border {
            let spot = cache.object_cache.get(id).unwrap();
            assert_eq!(spot.colocalized_with[&CLASS_A], vec![ObjectId(ID_CELL_4)]);
        }

        // Cell 1's B-list must be exactly its direct spots (the border spots it
        // merely touches, but didn't win, must be excluded).
        let cell_1 = cache.object_cache.get(&ObjectId(ID_CELL_1)).unwrap();
        let mut cell_1_b = cell_1.colocalized_with[&CLASS_B].clone();
        cell_1_b.sort();
        let mut expected_cell_1_b = b_direct_1.clone();
        expected_cell_1_b.sort();
        assert_eq!(
            cell_1_b, expected_cell_1_b,
            "Cell 1 must not list a B spot it merely touches but didn't win"
        );

        // Cell 1's C-list must be exactly its direct spots (the C border went to Cell 4).
        let mut cell_1_c = cell_1.colocalized_with[&CLASS_C].clone();
        cell_1_c.sort();
        let mut expected_cell_1_c = c_direct_1.clone();
        expected_cell_1_c.sort();
        assert_eq!(
            cell_1_c, expected_cell_1_c,
            "Cell 1 must not list a C spot it merely touches but didn't win"
        );
    }

    // --- BboxGrid integration cross-check ---

    /// End-to-end cross-check of the grid-pruned scan against an independent naive
    /// O(N*M) reference computed directly here with `Object::overlaps` (bypassing
    /// `BboxGrid` entirely) - across many randomized 3-class layouts with size and
    /// position variance (small/large mixes, dense clusters, near-misses, wide empty
    /// gaps). Restricted to `ManyToMany`, whose semantics are exactly "coloc with every
    /// overlapping partner meeting `min_coloc_area`" and so are simple enough to
    /// reimplement independently without duplicating `execute`'s own logic. This is the
    /// correctness guardrail for wiring `BboxGrid` into `execute` itself, complementing
    /// `spatial_grid::tests::randomized_layouts_never_miss_a_true_bbox_intersection`
    /// (which only checks the grid's candidate lists in isolation, not the full
    /// `Colocalization` pipeline built on top of them).
    #[test]
    fn randomized_many_to_many_layout_matches_naive_pairwise_reference() {
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

        let classes = [CLASS_A, CLASS_B, CLASS_C];

        for seed in 1..12u32 {
            let mut rng = Rng(seed * 104_729 + 1);
            let mut cache = PipelineCache::default();
            let mut ids_by_class: std::collections::HashMap<ObjectClass, Vec<ObjectId>> =
                std::collections::HashMap::new();
            let mut next_id: u128 = 1;
            for &class in &classes {
                for _ in 0..25 {
                    // Wide size and position ranges (1..20px extents over a 0..50
                    // canvas) so layouts include dense clusters, near-misses (adjacent
                    // but not touching), and isolated objects across seeds.
                    let w = 1 + rng.range(20);
                    let h = 1 + rng.range(20);
                    let x = rng.range(50);
                    let y = rng.range(50);
                    let object = make_filled_object(
                        next_id,
                        [x, y, x + w, y + h],
                        ImagePlane::default(),
                        class,
                    );
                    next_id += 1;
                    ids_by_class
                        .entry(class)
                        .or_default()
                        .push(object.id.clone());
                    cache.object_cache.insert(object.id.clone(), object);
                }
            }

            let coloc = Colocalization {
                classes_to_coloc: classes.to_vec(),
                filter_classes: vec![],
                exclude_classes: vec![],
                class_for_overlapping_areas: ObjectClass::Unset,
                multiplicity: ColocMultiplicity::ManyToMany,
                min_coloc_area: 0.0,
                size_unit: SizeUnits::Pixels,
            };
            run(&coloc, &mut cache);

            for &anchor_class in &classes {
                for anchor_id in &ids_by_class[&anchor_class] {
                    let anchor_object = cache.object_cache.get(anchor_id).unwrap();

                    // Raw per-other-class overlap lists, and whether the anchor overlaps
                    // *every* other class - `execute` only records an anchor as
                    // colocalized at all if it overlaps at least one object from every
                    // other class-to-coloc (see `three_class_hub_spoke_only_hub_colocalized`
                    // above), independent of `ManyToMany` allowing multiple partners
                    // *within* a class.
                    let mut raw: std::collections::HashMap<ObjectClass, Vec<ObjectId>> =
                        std::collections::HashMap::new();
                    let mut overlaps_every_other_class = true;
                    for &other_class in &classes {
                        if other_class == anchor_class {
                            continue;
                        }
                        let matches: Vec<ObjectId> = ids_by_class[&other_class]
                            .iter()
                            .filter(|other_id| {
                                cache
                                    .object_cache
                                    .get(*other_id)
                                    .and_then(|other| anchor_object.overlaps(other))
                                    .is_some()
                            })
                            .cloned()
                            .collect();
                        if matches.is_empty() {
                            overlaps_every_other_class = false;
                        }
                        raw.insert(other_class, matches);
                    }

                    for &other_class in &classes {
                        if other_class == anchor_class {
                            continue;
                        }
                        let mut expected = if overlaps_every_other_class {
                            raw[&other_class].clone()
                        } else {
                            Vec::new()
                        };
                        expected.sort();

                        let mut actual = anchor_object
                            .colocalized_with
                            .get(&other_class)
                            .cloned()
                            .unwrap_or_default();
                        actual.sort();

                        assert_eq!(
                            actual, expected,
                            "seed {seed}: anchor {anchor_id:?} ({anchor_class:?}) vs \
                             {other_class:?} - grid-pruned result diverged from the naive \
                             pairwise reference"
                        );
                    }
                }
            }
        }
    }
}
