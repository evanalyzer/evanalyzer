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
};
use evanalyzer_cfg::core_types::{
    InternalErrors, ObjectClass, ObjectId, SegmentationClass, SizeUnits,
};
use indexmap::IndexMap;
use log::{debug, info, warn};
use macros::CommandsMeta;
use std::sync::Arc;

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
    pub filter_classes: Vec<ObjectClass>,

    /// Class of the overlapping area if needed
    ///
    /// If defined the overlapping coloc area is added as new object and labeled with this class
    pub class_for_overlapping_areas: ObjectClass,

    /// Classes an object must NOT overlap to be considered colocalized.
    ///
    /// An object that otherwise satisfies `classes_to_coloc` is excluded entirely
    /// (no relation recorded, no intersection object created) if it also overlaps any
    /// object from one of these classes with at least `min_coloc_area`. Leave empty
    /// (the default) to disable this filter - exclusion is opt-in.
    ///
    /// Example: to find objects colocalizing with class 1 and 2 but *not* 3, set
    /// `classes_to_coloc: [1, 2]` and `exclude_classes: [3]`.
    #[cmdsmeta(optional = true)]
    pub exclude_classes: Vec<ObjectClass>,

    /// If set one object is allowed to coloc with more than one other object
    pub allow_multi_object_coloc: bool,

    // Size unit for the minimum coloc area size
    pub size_unit: SizeUnits,

    /// Minimum overlapping area size to count objects as coloc
    pub min_coloc_area: f32,
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

        let mut overlap_matches: std::collections::HashMap<
            ObjectId,
            IndexMap<ObjectClass, Vec<ObjectId>>,
        > = std::collections::HashMap::new();
        let mut new_objects: Vec<Object> = Vec::new();
        // Tracks which sorted sets of object IDs already produced an overlap object (avoids duplicates).
        let mut processed_combinations: std::collections::HashSet<Vec<ObjectId>> =
            std::collections::HashSet::new();

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
                    let other_bucket = match class_buckets.get(other_class) {
                        Some(b) => b,
                        None => continue 'anchor,
                    };

                    // Compute the actual overlap and keep only entries meeting min_area_px.
                    // Each entry is (ObjectId, overlap_area_in_pixels).
                    let mut candidates: Vec<(ObjectId, usize)> = other_bucket
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

                    if !self.allow_multi_object_coloc {
                        // Keep only the single best match (largest overlap area) per class.
                        if let Some(best_idx) = candidates
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, (_, area))| *area)
                            .map(|(idx, _)| idx)
                        {
                            let best = candidates.swap_remove(best_idx);
                            candidates = vec![best];
                        }
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
                if exclude_ids.iter().any(|excl_id| {
                    excl_id != anchor_id
                        && cache
                            .object_cache
                            .get(excl_id)
                            .and_then(|r| anchor_object.overlaps(r))
                            .is_some_and(|intersection| intersection.area >= min_area_px)
                }) {
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

    #[test]
    fn fewer_than_two_classes_is_noop() {
        let coloc = Colocalization {
            classes_to_coloc: vec![CLASS_A],
            filter_classes: vec![],
            exclude_classes: vec![],
            class_for_overlapping_areas: ObjectClass::Unset,
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
            allow_multi_object_coloc: true,
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
}
