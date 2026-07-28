//! # Classification Module
//!
//! Provides functionality for classifying ROIs based on morphological and intensity metrics.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-05-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module implements object classification algorithms that assign object classes
//! to regions of interest based on configurable criteria and machine learning models.

use crate::{
    algos::ImageAlgorithm,
    image::PixelSizes,
    object::Object, // ... other imports
};
use evanalyzer_cfg::core_types::{
    InternalErrors,
    ObjectClass::{self, Unset},
    ObjectId, PixelUnits, SegmentationClass, SizeUnits,
};
use log::{debug, info, warn};
use macros::CommandsMeta;

#[derive(CommandsMeta)]
pub enum ClassifyMatchHandling {
    #[cmdsmeta(display_name = "Add class on match")]
    AddOutputClassIfMatch,
    #[cmdsmeta(display_name = "Add class on mismatch")]
    AddOutputClassIfNotMatch,

    #[cmdsmeta(display_name = "Remove class on match", visible = false)]
    RemoveInputClassIfMatch,
    #[cmdsmeta(display_name = "Remove class on mismatch", visible = false)]
    RemoveInputClassIfNotMatch,

    #[cmdsmeta(display_name = "Remove output class on match")]
    RemoveOutputClassIfMatch,
    #[cmdsmeta(display_name = "Remove output class on mismatch")]
    RemoveOutputClassIfNotMatch,

    #[cmdsmeta(display_name = "Remove objects matching criteria")]
    RemoveAllClassesIfMatch,
    #[cmdsmeta(display_name = "Keep objects matching criteria")]
    RemoveAllClassesIfNotMatch,

    #[cmdsmeta(display_name = "Reclassify on match")]
    ReclassifyIfMatch,
    #[cmdsmeta(display_name = "Reclassify on mismatch")]
    ReclassifyIfNotMatch,
}

/// Classifies ROIs based on morphological and intensity features.
///
/// This command applies rule-based classification logic to assign object classes
/// to extracted ROIs. Classification is performed using configurable criteria
/// including area, shape descriptors, and intensity statistics.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
pub struct ClassifyObjects {
    /// Apply only to objects with this given segmentation class
    ///
    /// The segmentation class value is assigned to each pixel in the image
    /// after a Threshold, Pixel classifier or AI classifier.
    /// If no seg class is selected the criteria are applied to all objects.
    #[cmdsmeta(visible = false)]
    pub origin_segmentation: Vec<SegmentationClass>,

    /// Restrict classification to objects that already carry one of these classes
    ///
    /// Only ROIs that have been assigned at least one of the listed classes by a prior
    /// pipeline step will be evaluated against the morphological and intensity criteria below.
    /// Leave empty to apply the criteria to every object regardless of its current class.
    #[cmdsmeta(display_name = "Input Classes")]
    pub input_classes: Vec<ObjectClass>,

    /// What to do with object class labels after criteria evaluation
    ///
    /// Controls whether the output class is added or existing classes are removed,
    /// and whether the action is triggered on a criteria **match** or a **non-match**:
    ///
    /// - **AddOutputClassIfMatch** - append the output class to objects that pass the criteria.
    /// - **AddOutputClassIfNotMatch** - append the output class to objects that fail the criteria.
    /// - **RemoveInputClassIfMatch / NotMatch** - strip all input classes from matching / non-matching objects.
    /// - **RemoveOutputClassIfMatch / NotMatch** - strip the output class from matching / non-matching objects.
    /// - **RemoveAllClassesIfMatch / NotMatch** - clear every class label from matching / non-matching objects.
    #[cmdsmeta(default = ClassifyMatchHandling::RemoveAllClassesIfNotMatch)]
    pub match_handling: ClassifyMatchHandling,

    /// Class label assigned to (or removed from) objects by the chosen operation
    ///
    /// Used as the target class for `AddOutputClass*` and `RemoveOutputClass*` operations.
    /// Has no effect when the selected operation only manipulates input classes or clears all classes.
    #[cmdsmeta(default = ObjectClass::Unset, display_name = "Output Tag")]
    pub output_class: ObjectClass,

    /// Additional criterion: the object must intersect an object carrying this class
    ///
    /// If unset (the default) this filter is not applied. When set, an object only
    /// satisfies the overall criteria if it also overlaps at least one object carrying this
    /// class by at least `min_intersection_area`. Combine with e.g.
    /// `RemoveAllClassesIfMatch` to drop objects that intersect another class's objects,
    /// or `AddOutputClassIfMatch` to tag objects that do.
    #[cmdsmeta(default = ObjectClass::Unset, display_name = "Intersecting With")]
    pub overlapping_with: ObjectClass,

    /// Minimum intersection area with an `overlapping_with` object, in `size_unit`
    ///
    /// Has no effect while `overlapping_with` is Unset.
    #[cmdsmeta(default = 0, min = 0.0, max = 2147483648.0, summary = false)]
    pub min_intersection_area: f32,

    /// Unit to use for object extraction
    #[cmdsmeta(
        default = SizeUnits::NanoMeter,
    )]
    pub size_unit: SizeUnits,

    /// Minimum area size
    ///
    /// Minimum area size of the object in selected unit (px^2 or nm^2).
    #[cmdsmeta(default = 0, min = 0.0, max = 2147483648.0, summary = true)]
    pub min_area: f32,

    /// Maximum area size
    ///
    /// Maximum area size of the object in selected unit (px^2 or nm^2).
    #[cmdsmeta(default = 2147483648.0, min = 0.0, max = 2147483648.0, summary = false)]
    pub max_area: f32,

    /// Circularity range: 0 = elongated, 1 = perfect circle
    ///
    /// Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    /// A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    /// It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub min_circularity: f32,

    /// Circularity range: 0 = elongated, 1 = perfect circle
    ///
    /// Circularity (sometimes called Isoperimetric Quotient) measures how efficiently a shape encloses its area relative to the length of its perimeter.
    /// A circle is the mathematically perfect shape for maximizing area while minimizing perimeter.
    /// It is calculated with `4*Pi*AreaSize / Perimeter^2`
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub max_circularity: f32,

    /// Minimum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    /// Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    /// It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    /// often visualized as a rubber band stretched around the shape).
    ///
    /// Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    /// Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub min_solidity: f32,

    /// Maximum Solidity/Compactness: 0 = hollow, 1 = perfect convex
    ///
    /// Solidity is a structural metric used in shape analysis to measure how "solid" or compact an object is.
    /// It compares the actual area of an object to the area of its Convex Hull (the smallest convex polygon that can completely enclose the object,
    /// often visualized as a rubber band stretched around the shape).
    ///
    /// Solidity = 1.0: The object is perfectly convex (e.g., a perfect circle, a solid square, or an ellipse). It has no holes, indentations, or deep recesses.
    /// Solidity < 1.0: The object has irregular boundaries, deep "bays," protrusions, or internal holes. The lower the value, the more jagged or structurally fragmented the object is.
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = false)]
    pub max_solidity: f32,

    /// Minimum proportional relationship between an object's width and its height
    ///
    /// This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    /// The value is without unit in the range of 0 to MAX_F32
    #[cmdsmeta(default = 0.0, min = 0.0, max = 2147483648.0, summary = false)]
    pub min_aspect_ratio: f32,

    /// Maximum proportional relationship between an object's width and its height
    ///
    /// This value is calculated by the object bounding box with and height and is defined with `a = with/height`.
    /// The value is without unit in the range of 0 to MAX_F32
    #[cmdsmeta(
        default = 2147483648.0,
        min = 0.0,
        max = 2147483648.0,
        step = 1.0,
        summary = false
    )]
    pub max_aspect_ratio: f32,

    /// Eccentricity: 0 = perfect circle, 1 = line
    ///
    /// Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    /// It imagines the shape as an ellipse and measures how far apart its focal points are.
    /// It is calculated with `sqrt(1-(b/a)^2)`
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.1, summary = true)]
    pub min_eccentricity: f32,

    /// Eccentricity: 0 = perfect circle, 1 = line
    ///
    /// Eccentricity is a metric that measures how much a shape deviates from being a perfect circle.
    /// It imagines the shape as an ellipse and measures how far apart its focal points are.
    /// It is calculated with `sqrt(1-(b/a)^2)`
    #[cmdsmeta(default = 1.0, min = 0.0, max = 1.0, step = 0.1, summary = true)]
    pub max_eccentricity: f32,

    /// Feret diameter threshold
    ///
    /// The absolute shortest parallel distance across the object.
    /// This represents the minimum sieve size a particle could pass through.
    ///
    /// In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    /// It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    /// When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[cmdsmeta(default = 0, min = 0.0, max = 2147483648.0, summary = false, step = 1)]
    pub min_feret: f32,

    /// Maximum feret diameter threshold in selected unit (px or nm)
    ///
    /// The absolute longest distance across the object at any angle.
    /// Used to measure elongation or the maximum length of a particle.
    ///
    /// In image processing and particle size analysis, the Feret diameter (often called the caliper diameter) is a metric used to measure the size of an irregular object.
    /// It mimics the action of a slide caliper, measuring the distance between two parallel tangential lines bounding the object at a specific angle.
    /// When analyzing objects or particles, applying Feret diameter thresholds allows you to filter out noise, classify objects by shape, or isolate specific structures based on their directional length rather than their total area.
    #[cmdsmeta(
        default = 2147483648.0,
        min = 0.0,
        max = 2147483648.0,
        summary = false,
        step = 1
    )]
    pub max_feret: f32,

    /// Whether object can touch image edge
    #[cmdsmeta(default = true, summary = true)]
    pub allow_edge_touching: bool,
}

impl Default for ClassifyObjects {
    fn default() -> Self {
        Self {
            min_area: 0.0,
            max_area: 2147483648.0,
            min_circularity: 0.0,
            max_circularity: 1.0,
            min_solidity: 0.0,
            max_solidity: 1.0,
            min_aspect_ratio: 0.0,
            max_aspect_ratio: 2147483648.0,
            min_eccentricity: 0.0,
            max_eccentricity: 1.0,
            min_feret: 0.0,
            max_feret: 2147483648.0,
            allow_edge_touching: true,
            size_unit: SizeUnits::NanoMeter,
            origin_segmentation: vec![],
            output_class: Unset,
            overlapping_with: Unset,
            min_intersection_area: 0.0,
            input_classes: vec![],
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
        }
    }
}

impl ImageAlgorithm for ClassifyObjects {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        let px_size = ctx.pixel_sizes();
        let pixel_area_size_nm = px_size.px_size_x * px_size.px_size_y;
        let min_intersection_px = self
            .size_unit
            .to_pixel(self.min_intersection_area, pixel_area_size_nm);

        // ROIs carrying the class the overlap criterion checks against. Collected once
        // up front since evaluating it below needs to read other `object_cache` entries while
        // the entry under evaluation is also being read - the two can't interleave with the
        // mutation pass further down, which needs a mutable borrow of the same map.
        let overlap_candidates: Vec<ObjectId> = if self.overlapping_with == Unset {
            Vec::new()
        } else {
            cache
                .object_cache
                .values()
                .filter(|object| object.has_object_class(&self.overlapping_with))
                .map(|object| object.id.clone())
                .collect()
        };

        let evaluations: Vec<(ObjectId, bool)> = cache
            .object_cache
            .values()
            .filter(|object| {
                self.input_classes.is_empty() || object.has_object_classes(&self.input_classes)
            })
            .map(|object| {
                let matches = self.matches_criteria(object, px_size)
                    && self.matches_overlap(
                        object,
                        cache,
                        &overlap_candidates,
                        min_intersection_px,
                    );
                (object.id.clone(), matches)
            })
            .collect();

        for (id, matches) in evaluations {
            let Some(object) = cache.object_cache.get_mut(&id) else {
                continue;
            };
            match self.match_handling {
                ClassifyMatchHandling::AddOutputClassIfMatch => {
                    if matches {
                        object.add_object_class(self.output_class.clone());
                    }
                }
                ClassifyMatchHandling::AddOutputClassIfNotMatch => {
                    if !matches {
                        object.add_object_class(self.output_class.clone());
                    }
                }
                ClassifyMatchHandling::RemoveInputClassIfMatch => {
                    if matches {
                        for class in &self.input_classes {
                            object.remove_object_class(class);
                        }
                    }
                }
                ClassifyMatchHandling::RemoveInputClassIfNotMatch => {
                    if !matches {
                        for class in &self.input_classes {
                            object.remove_object_class(class);
                        }
                    }
                }
                ClassifyMatchHandling::RemoveOutputClassIfMatch => {
                    if matches {
                        object.remove_object_class(&self.output_class);
                    }
                }
                ClassifyMatchHandling::RemoveOutputClassIfNotMatch => {
                    if !matches {
                        object.remove_object_class(&self.output_class);
                    }
                }
                ClassifyMatchHandling::RemoveAllClassesIfMatch => {
                    if matches {
                        object.object_class.clear();
                    }
                }
                ClassifyMatchHandling::RemoveAllClassesIfNotMatch => {
                    if !matches {
                        object.object_class.clear();
                    }
                }
                ClassifyMatchHandling::ReclassifyIfMatch => {
                    if matches {
                        object.object_class.clear();
                        object.add_object_class(self.output_class.clone());
                    }
                }
                ClassifyMatchHandling::ReclassifyIfNotMatch => {
                    if !matches {
                        object.object_class.clear();
                        object.add_object_class(self.output_class.clone());
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ClassifyObjects"
    }
}

impl ClassifyObjects {
    /// Evaluates whether an object matches the classification criteria.
    ///
    /// # Arguments
    /// * `object` - The object to evaluate
    ///
    /// # Returns
    /// `true` if the object matches all criteria, `false` otherwise
    fn matches_criteria(&self, object: &Object, pixel_sizes: &PixelSizes) -> bool {
        let pixel_area_size_nm = pixel_sizes.px_size_x * pixel_sizes.px_size_y;
        let min_area_px = self.size_unit.to_pixel(self.min_area, pixel_area_size_nm);
        let max_area_px = self.size_unit.to_pixel(self.max_area, pixel_area_size_nm);

        // Check area
        if object.area < min_area_px || object.area > max_area_px {
            return false;
        }

        // Check edge touching
        if object.touches_edge && !self.allow_edge_touching {
            return false;
        }

        // Check circularity
        let circularity = object.circularity();
        if circularity < self.min_circularity || circularity > self.max_circularity {
            return false;
        }

        // Check solidity
        let solidity = object.get_solidity();
        if solidity < self.min_solidity || solidity > self.max_solidity {
            return false;
        }

        // Check aspect ratio
        let aspect_ratio = object.get_aspect_ratio();
        if aspect_ratio < self.min_aspect_ratio || aspect_ratio > self.max_aspect_ratio {
            return false;
        }

        // Check eccentricity (from ellipse fitting)
        let ellipse = object.get_ellipse();
        if ellipse.eccentricity < self.min_eccentricity
            || ellipse.eccentricity > self.max_eccentricity
        {
            return false;
        }

        // Check Feret diameter
        let feret = object.get_feret_diameter();
        if feret < self.min_feret || feret > self.max_feret {
            return false;
        }

        true
    }

    /// Checks the `overlapping_with` criterion: whether `object` intersects at least one of
    /// `candidates` (ROIs carrying the `overlapping_with` class) by at least
    /// `min_intersection_px`. Always true when `overlapping_with` is Unset (filter disabled).
    fn matches_overlap(
        &self,
        object: &Object,
        cache: &crate::pipeline::pipeline_cache::PipelineCache,
        candidates: &[ObjectId],
        min_intersection_px: usize,
    ) -> bool {
        if self.overlapping_with == Unset {
            return true;
        }

        candidates.iter().any(|other_id| {
            *other_id != object.id
                && cache
                    .object_cache
                    .get(other_id)
                    .and_then(|other| object.overlaps(other))
                    .is_some_and(|intersection| intersection.area >= min_intersection_px)
        })
    }
}

#[cfg(test)]
mod tests {
    use bitvec::vec;

    use super::*;
    use crate::{
        ImageContainer, ImagePlane, ManagedImage,
        image::PixelSizes,
        object::ObjectInit,
        pipeline::{
            pipeline::PipelineImageMeta, pipeline_cache::PipelineCache,
            pipeline_context::PipelineContext,
        },
    };
    use bitvec::prelude::*;
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;
    use std::path::PathBuf;

    #[test]
    fn test_classification_criteria_default() {
        let criteria = ClassifyObjects::default();
        assert_eq!(criteria.min_area, 0.0);
        assert_eq!(criteria.max_area, 2147483600.0);
        assert!(criteria.allow_edge_touching);
        assert_eq!(criteria.overlapping_with, ObjectClass::Unset);
        assert_eq!(criteria.min_intersection_area, 0.0);
    }

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

    const CLASS_A: ObjectClass = ObjectClass::Valid(1);
    const CLASS_B: ObjectClass = ObjectClass::Valid(2);
    const ID_A: u128 = 100_000;
    const ID_B: u128 = 200_000;

    fn run(cmd: &ClassifyObjects, cache: &mut PipelineCache) {
        cmd.execute(&mut make_ctx(), cache).unwrap();
    }

    #[test]
    fn overlapping_with_unset_by_default_does_not_filter() {
        // Two non-overlapping ROIs, no overlap criterion configured: the criteria (which are
        // otherwise wide open) must match both, exactly as before this feature existed.
        let cmd = ClassifyObjects {
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        let object_b = make_filled_object(ID_B, [10, 10, 14, 14], CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&cmd, &mut cache);

        assert!(
            cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A)
        );
        assert!(
            cache
                .object_cache
                .get(&ObjectId(ID_B))
                .unwrap()
                .has_object_class(&CLASS_B)
        );
    }

    #[test]
    fn overlapping_with_removes_class_from_non_intersecting_object() {
        // A doesn't overlap B, so with `overlapping_with: CLASS_B` set, A fails the criteria
        // and RemoveAllClassesIfNotMatch strips its class.
        let cmd = ClassifyObjects {
            overlapping_with: CLASS_B,
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        let object_b = make_filled_object(ID_B, [10, 10, 14, 14], CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&cmd, &mut cache);

        assert!(
            !cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A),
            "A doesn't intersect any CLASS_B object, so it should fail the criteria"
        );
    }

    #[test]
    fn overlapping_with_keeps_class_on_intersecting_object() {
        // A overlaps B, so with `overlapping_with: CLASS_B` set, A passes the criteria and
        // keeps its class.
        let cmd = ClassifyObjects {
            overlapping_with: CLASS_B,
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        // A=[0,4]x[0,4], B=[2,6]x[2,6] -> overlap [2,4]x[2,4]
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        let object_b = make_filled_object(ID_B, [2, 2, 6, 6], CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&cmd, &mut cache);

        assert!(
            cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A),
            "A intersects a CLASS_B object, so it should pass the criteria"
        );
    }

    #[test]
    fn min_intersection_area_rejects_too_small_an_overlap() {
        // A and B overlap by a single pixel ([4,4]x[4,4] = 1 px), but min_intersection_area
        // demands at least 4 - too small an overlap must not count as a match.
        let cmd = ClassifyObjects {
            overlapping_with: CLASS_B,
            min_intersection_area: 4.0,
            size_unit: SizeUnits::Pixels,
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        // A=[0,4]x[0,4], B=[4,8]x[4,8] -> overlap is the single pixel (4,4)
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        let object_b = make_filled_object(ID_B, [4, 4, 8, 8], CLASS_B);
        cache.object_cache.insert(object_a.id.clone(), object_a);
        cache.object_cache.insert(object_b.id.clone(), object_b);

        run(&cmd, &mut cache);

        assert!(
            !cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A),
            "overlap area (1px) is below min_intersection_area (4px), so A should fail"
        );
    }

    // -- match_handling: every other test in this file uses
    // `RemoveAllClassesIfNotMatch` - these cover the other 9 `ClassifyMatchHandling`
    // variants, each previously untested. `min_circularity: 2.0` is an
    // unreachable bound (circularity tops out at 1.0) used to force
    // `matches == false` for the "IfNotMatch" cases, since a lone filled
    // rectangular object otherwise passes the wide-open default criteria.

    #[test]
    fn add_output_class_if_match_adds_alongside_the_existing_class() {
        let cmd = ClassifyObjects {
            output_class: CLASS_B,
            match_handling: ClassifyMatchHandling::AddOutputClassIfMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        let object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            object.has_object_class(&CLASS_A),
            "original class must be kept, not replaced"
        );
        assert!(object.has_object_class(&CLASS_B));
    }

    #[test]
    fn add_output_class_if_not_match_only_fires_when_criteria_fail() {
        let cmd = ClassifyObjects {
            output_class: CLASS_B,
            min_circularity: 2.0,
            match_handling: ClassifyMatchHandling::AddOutputClassIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        assert!(
            cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_B)
        );
    }

    #[test]
    fn remove_input_class_if_match_strips_listed_input_classes_only() {
        let cmd = ClassifyObjects {
            input_classes: vec![CLASS_A],
            match_handling: ClassifyMatchHandling::RemoveInputClassIfMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let mut object = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        object.add_object_class(CLASS_B);
        cache.object_cache.insert(ObjectId(ID_A), object);
        run(&cmd, &mut cache);

        let object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(
            !object.has_object_class(&CLASS_A),
            "listed input class must be removed"
        );
        assert!(
            object.has_object_class(&CLASS_B),
            "class not in input_classes must survive"
        );
    }

    #[test]
    fn remove_input_class_if_not_match_only_fires_when_criteria_fail() {
        let cmd = ClassifyObjects {
            input_classes: vec![CLASS_A],
            min_circularity: 2.0,
            match_handling: ClassifyMatchHandling::RemoveInputClassIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        assert!(
            !cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A)
        );
    }

    #[test]
    fn remove_output_class_if_match_strips_only_the_output_class() {
        let cmd = ClassifyObjects {
            output_class: CLASS_A,
            match_handling: ClassifyMatchHandling::RemoveOutputClassIfMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let mut object = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        object.add_object_class(CLASS_B);
        cache.object_cache.insert(ObjectId(ID_A), object);
        run(&cmd, &mut cache);

        let object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(!object.has_object_class(&CLASS_A));
        assert!(
            object.has_object_class(&CLASS_B),
            "only the output class should be touched"
        );
    }

    #[test]
    fn remove_output_class_if_not_match_only_fires_when_criteria_fail() {
        let cmd = ClassifyObjects {
            output_class: CLASS_A,
            min_circularity: 2.0,
            match_handling: ClassifyMatchHandling::RemoveOutputClassIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        assert!(
            !cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A)
        );
    }

    #[test]
    fn remove_all_classes_if_match_clears_every_class() {
        let cmd = ClassifyObjects {
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let mut object = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        object.add_object_class(CLASS_B);
        cache.object_cache.insert(ObjectId(ID_A), object);
        run(&cmd, &mut cache);

        assert!(
            cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .object_class
                .is_empty()
        );
    }

    #[test]
    fn reclassify_if_match_replaces_every_class_with_the_output_class() {
        let cmd = ClassifyObjects {
            output_class: CLASS_B,
            match_handling: ClassifyMatchHandling::ReclassifyIfMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        let object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(!object.has_object_class(&CLASS_A));
        assert!(object.has_object_class(&CLASS_B));
        assert_eq!(
            object.object_class.len(),
            1,
            "reclassify should leave exactly the output class"
        );
    }

    #[test]
    fn reclassify_if_not_match_only_fires_when_criteria_fail() {
        let cmd = ClassifyObjects {
            output_class: CLASS_B,
            min_circularity: 2.0,
            match_handling: ClassifyMatchHandling::ReclassifyIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        cache.object_cache.insert(
            ObjectId(ID_A),
            make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A),
        );
        run(&cmd, &mut cache);

        let object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(!object.has_object_class(&CLASS_A));
        assert!(object.has_object_class(&CLASS_B));
    }

    #[test]
    fn overlapping_with_ignores_self_intersection() {
        // The overlap candidate bucket can legitimately include the object being evaluated
        // itself (e.g. `overlapping_with` set to one of the object's own classes). It must not
        // be allowed to "intersect itself" to pass the criteria.
        let cmd = ClassifyObjects {
            overlapping_with: CLASS_A,
            match_handling: ClassifyMatchHandling::RemoveAllClassesIfNotMatch,
            ..Default::default()
        };
        let mut cache = PipelineCache::default();
        let object_a = make_filled_object(ID_A, [0, 0, 4, 4], CLASS_A);
        cache.object_cache.insert(object_a.id.clone(), object_a);

        run(&cmd, &mut cache);

        assert!(
            !cache
                .object_cache
                .get(&ObjectId(ID_A))
                .unwrap()
                .has_object_class(&CLASS_A),
            "a lone object must not be considered as intersecting itself"
        );
    }
}
