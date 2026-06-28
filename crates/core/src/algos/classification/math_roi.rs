//! # ROI Math Module
//!
//! Provides boolean "ROI math" (AND / OR / XOR / SUBTRACT) between two object classes.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module computes a boolean set operation between each `input_class` ROI ("A")
//! and the union of its overlapping `other_class` ROIs ("B"). Unlike a pixel-level
//! binary-image operation (rasterize the whole class to one mask, apply the op, then
//! re-extract objects), this operates on object pairs: each result stays tied to
//! exactly the one `input_class` ROI it came from, and only the small bbox window
//! spanned by the operands is ever touched, regardless of image or tile size.
//!
use crate::{
    algos::ImageAlgorithm,
    roi::{BooleanOp, Roi, RoiInit},
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId, SizeUnits};
use macros::CommandsMeta;

/// Boolean set operation applied between an `input_class` ROI ("A") and the union of
/// its overlapping `other_class` ROIs ("B").
#[derive(CommandsMeta)]
pub enum RoiSetOperation {
    /// Intersection: pixels present in both A and B. With no overlapping B, the
    /// result is empty - there is nothing to keep regardless of `keep_unmatched`.
    And,
    /// Union: pixels present in A or B (or both). The result can extend beyond A's
    /// own bounding box into B's territory.
    Or,
    /// Symmetric difference: pixels present in exactly one of A, B (the overlap
    /// itself is excluded). Like `Or`, the result can extend beyond A's bbox.
    Xor,
    /// Set difference: pixels in A that are NOT in B (A \ B). The classic use case
    /// is deriving a cytoplasm-only region from a whole-cell mask and its nucleus.
    Subtract,
}

impl From<&RoiSetOperation> for BooleanOp {
    fn from(op: &RoiSetOperation) -> Self {
        match op {
            RoiSetOperation::And => BooleanOp::And,
            RoiSetOperation::Or => BooleanOp::Or,
            RoiSetOperation::Xor => BooleanOp::Xor,
            RoiSetOperation::Subtract => BooleanOp::Subtract,
        }
    }
}

/// Computes a boolean set operation between two object classes, object pair by
/// object pair.
///
/// When more than one `other_class` object overlaps a given input ROI, all of them
/// are unioned into a single "B" before the operation is applied, so the result
/// doesn't depend on the order they'd otherwise be combined in.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
pub struct RoiMath {
    /// Boolean set operation to apply
    #[cmdsmeta(summary = true)]
    pub operation: RoiSetOperation,

    /// ROIs carrying this class are the left-hand operand ("A").
    pub input_class: ObjectClass,

    /// ROIs carrying this class are the right-hand operand ("B").
    pub other_class: ObjectClass,

    /// Optional additional label filters applied to `other_class` objects.
    ///
    /// Only `other_class` objects that carry all listed classes are used.
    pub other_filter_classes: Vec<ObjectClass>,

    /// Size unit for `min_overlap_area`
    pub size_unit: SizeUnits,

    /// Minimum overlap area before an `other_class` object is treated as a partner
    /// of an input ROI; objects overlapping less than this are ignored.
    pub min_overlap_area: f32,

    /// If unset, the result replaces the input ROI in place.
    ///
    /// If set, a new ROI carrying this class is created for each input ROI instead,
    /// leaving the input ROI untouched.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub output_class: ObjectClass,

    /// When an input ROI has no qualifying overlapping partner: keep it unchanged in
    /// the output (true), or drop it entirely - no output for it at all - (false).
    ///
    /// Note this is a policy override, not the literal mathematical result: e.g. for
    /// `And`, the true result of "A and nothing" is empty, but `keep_unmatched = true`
    /// still leaves A untouched rather than emitting a zero-area ROI.
    #[cmdsmeta(default = true)]
    pub keep_unmatched: bool,
}

impl ImageAlgorithm for RoiMath {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.input_class == ObjectClass::Unset || self.other_class == ObjectClass::Unset {
            return Ok(());
        }

        let image_size = ctx.full_image_size();
        let px_sizes = ctx.pixel_sizes();
        let pixel_area_nm2 = px_sizes.px_size_x * px_sizes.px_size_y;
        let min_area_px = self.size_unit.to_pixel(self.min_overlap_area, pixel_area_nm2);
        let op = BooleanOp::from(&self.operation);

        let other_ids: Vec<ObjectId> = cache
            .roi_cache
            .values()
            .filter(|r| {
                r.has_object_class(&self.other_class)
                    && self
                        .other_filter_classes
                        .iter()
                        .all(|f| r.has_object_class(f))
            })
            .map(|r| r.id.clone())
            .collect();

        let input_ids: Vec<ObjectId> = cache
            .roi_cache
            .values()
            .filter(|r| r.has_object_class(&self.input_class))
            .map(|r| r.id.clone())
            .collect();

        let mut new_rois: Vec<Roi> = Vec::new();
        let mut to_remove: Vec<ObjectId> = Vec::new();
        let replace_in_place =
            self.output_class == ObjectClass::Unset || self.output_class == self.input_class;

        for input_id in &input_ids {
            let Some(input_roi) = cache.roi_cache.get(input_id) else {
                continue;
            };

            let overlapping: Vec<&Roi> = other_ids
                .iter()
                .filter(|oid| *oid != input_id)
                .filter_map(|oid| cache.roi_cache.get(oid))
                .filter(|o| {
                    input_roi
                        .overlaps(o)
                        .is_some_and(|i| i.area >= min_area_px)
                })
                .collect();

            if overlapping.is_empty() {
                if !self.keep_unmatched && replace_in_place {
                    to_remove.push(input_id.clone());
                }
                continue;
            }

            let geometry = input_roi.combine_geometry(&overlapping, op, image_size);
            let (segmentation_class, plane) = (input_roi.segmentation_class, input_roi.plane.clone());

            // Build the result as its own (not-yet-inserted) ROI so intensities can be
            // sampled against the *new* mask before mutating the cache - `cache` can't
            // be borrowed both mutably and immutably at once.
            let transformed = Roi::new(RoiInit {
                id: ObjectId::next(),
                segmentation_class,
                parent_id: Some(input_id.clone()),
                plane,
                bbox: geometry.bbox,
                area: geometry.area,
                mask_data: geometry.mask_data,
                touches_edge: geometry.touches_edge,
                sum_x: geometry.sum_x,
                sum_y: geometry.sum_y,
                sum_x2: geometry.sum_x2,
                sum_y2: geometry.sum_y2,
                sum_xy: geometry.sum_xy,
                ..Default::default()
            });
            let intensities = transformed.measure_intensities(ctx, cache);

            if replace_in_place {
                if let Some(roi) = cache.roi_cache.get_mut(input_id) {
                    roi.bbox = transformed.bbox;
                    roi.mask_data = transformed.mask_data.clone();
                    roi.area = transformed.area;
                    roi.touches_edge = transformed.touches_edge;
                    roi.sum_x = transformed.sum_x;
                    roi.sum_y = transformed.sum_y;
                    roi.sum_x2 = transformed.sum_x2;
                    roi.sum_y2 = transformed.sum_y2;
                    roi.sum_xy = transformed.sum_xy;
                    roi.intensities = intensities;
                    roi.finalize_geometry();
                }
            } else {
                let mut new_roi = transformed;
                new_roi.intensities = intensities;
                new_roi.add_object_class(self.output_class);
                new_rois.push(new_roi);
            }
        }

        for id in to_remove {
            cache.roi_cache.remove(&id);
        }
        for roi in new_rois {
            cache.roi_cache.insert(roi.id.clone(), roi);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "RoiMath"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::PixelSizes,
        pipeline::{
            pipeline::PipelineImageMeta, pipeline_cache::PipelineCache,
            pipeline_context::PipelineContext,
        },
        ImageContainer, ImagePlane, ManagedImage,
    };
    use bitvec::prelude::*;
    use kornia_apriltag::utils::Point2d;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    const ID_A: u128 = 100_000;

    fn make_ctx(size: ImageSize) -> PipelineContext {
        let img = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0.0f32; size.width * size.height],
            CpuAllocator,
        )
        .unwrap();
        let managed = ManagedImage {
            data: img,
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        };
        PipelineContext::new_from_image(
            std::path::PathBuf::default(),
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

    /// Like `make_ctx`, but also registers a constant-value channel-0 image in
    /// `cache`, so intensity measurement has something real to sample.
    fn make_ctx_with_channel(size: ImageSize, cache: &mut PipelineCache, value: f32) -> PipelineContext {
        let ctx = make_ctx(size);
        let img = Image::<f32, 1, CpuAllocator>::new(size, vec![value; size.width * size.height], CpuAllocator)
            .unwrap();
        cache.image_cache.image_meta = PipelineImageMeta {
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
        };
        cache.image_cache.add_to_channel_cache(
            std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })),
            0,
        );
        ctx
    }

    /// A filled square ROI, `side` pixels wide, with top-left corner at `(x, y)`.
    fn make_square_roi(id: u128, x: u32, y: u32, side: u32, class: ObjectClass) -> Roi {
        let area = (side * side) as usize;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        let mut roi = Roi::new(RoiInit {
            id: ObjectId(id),
            bbox: [x, y, x + side - 1, y + side - 1],
            mask_data,
            area,
            plane: ImagePlane::default(),
            ..Default::default()
        });
        roi.add_object_class(class);
        roi
    }

    const CELL: ObjectClass = ObjectClass::Valid(10);
    const NUCLEUS: ObjectClass = ObjectClass::Valid(11);
    const OUT: ObjectClass = ObjectClass::Valid(12);

    fn default_cmd(operation: RoiSetOperation, output_class: ObjectClass) -> RoiMath {
        RoiMath {
            operation,
            input_class: CELL,
            other_class: NUCLEUS,
            other_filter_classes: vec![],
            size_unit: SizeUnits::Pixels,
            min_overlap_area: 0.0,
            output_class,
            keep_unmatched: true,
        }
    }

    #[test]
    fn subtract_creates_cytoplasm_like_shape_excluding_the_nucleus() {
        let cmd = default_cmd(RoiSetOperation::Subtract, OUT);
        let mut cache = PipelineCache::default();
        // Cell: 10x10 at (10,10) -> bbox [10,10,19,19], area 100.
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL);
        // Nucleus: 4x4 at (13,13) -> bbox [13,13,16,16], area 16, fully inside the cell.
        let nucleus = make_square_roi(200_000, 13, 13, 4, NUCLEUS);
        cache.roi_cache.insert(cell.id.clone(), cell);
        cache.roi_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert_eq!(cache.roi_cache.len(), 3, "original cell and nucleus stay untouched");
        let cytoplasm = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("cytoplasm ROI not found");
        assert_eq!(cytoplasm.area, 100 - 16);
        assert!(!cytoplasm.is_part_of(14, 14), "nucleus region must be excluded");
        assert!(cytoplasm.is_part_of(10, 10), "cell region outside the nucleus must remain");
    }

    #[test]
    fn and_keeps_only_the_overlap() {
        let cmd = default_cmd(RoiSetOperation::And, OUT);
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL); // [10,10,19,19]
        let nucleus = make_square_roi(200_000, 13, 13, 4, NUCLEUS); // [13,13,16,16], inside cell
        cache.roi_cache.insert(cell.id.clone(), cell);
        cache.roi_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("AND result not found");
        assert_eq!(result.area, 16, "AND must equal the nucleus area (fully inside the cell)");
        assert_eq!(result.bbox, [13, 13, 16, 16]);
    }

    #[test]
    fn or_extends_beyond_the_input_bbox() {
        let cmd = default_cmd(RoiSetOperation::Or, OUT);
        let mut cache = PipelineCache::default();
        let a = make_square_roi(ID_A, 10, 10, 10, CELL); // [10,10,19,19], area 100
        let b = make_square_roi(200_000, 15, 15, 10, NUCLEUS); // [15,15,24,24], area 100, overlap 25
        cache.roi_cache.insert(a.id.clone(), a);
        cache.roi_cache.insert(b.id.clone(), b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("OR result not found");
        assert_eq!(result.bbox, [10, 10, 24, 24], "union bbox must cover both operands");
        assert_eq!(result.area, 100 + 100 - 25);
    }

    #[test]
    fn xor_excludes_the_overlap() {
        let cmd = default_cmd(RoiSetOperation::Xor, OUT);
        let mut cache = PipelineCache::default();
        let a = make_square_roi(ID_A, 10, 10, 10, CELL); // area 100
        let b = make_square_roi(200_000, 15, 15, 10, NUCLEUS); // area 100, overlap 25
        cache.roi_cache.insert(a.id.clone(), a);
        cache.roi_cache.insert(b.id.clone(), b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("XOR result not found");
        assert_eq!(result.area, 100 + 100 - 2 * 25);
        assert!(!result.is_part_of(17, 17), "the overlap region must be excluded");
    }

    #[test]
    fn and_with_no_overlap_keeps_input_unchanged_when_keep_unmatched_true() {
        // AND's true mathematical result with nothing is empty, but keep_unmatched is
        // a policy override: it means "don't touch this ROI at all", not "replace it
        // with the literal (empty) result".
        let cmd = default_cmd(RoiSetOperation::And, ObjectClass::Unset);
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL);
        let original_area = cell.area;
        cache.roi_cache.insert(cell.id.clone(), cell);
        // No nucleus at all.

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let cell = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(cell.area, original_area);
    }

    #[test]
    fn drops_unmatched_input_when_keep_unmatched_false() {
        let mut cmd = default_cmd(RoiSetOperation::Or, ObjectClass::Unset);
        cmd.keep_unmatched = false;
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL);
        cache.roi_cache.insert(cell.id.clone(), cell);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert!(
            cache.roi_cache.get(&ObjectId(ID_A)).is_none(),
            "unmatched input must be dropped entirely when keep_unmatched is false"
        );
    }

    #[test]
    fn in_place_replaces_input_when_output_class_unset() {
        let cmd = default_cmd(RoiSetOperation::Subtract, ObjectClass::Unset);
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL);
        let nucleus = make_square_roi(200_000, 13, 13, 4, NUCLEUS);
        cache.roi_cache.insert(cell.id.clone(), cell);
        cache.roi_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert_eq!(cache.roi_cache.len(), 2, "no new ROI in in-place mode");
        let cell = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(cell.area, 100 - 16);
    }

    #[test]
    fn unions_multiple_overlapping_partners_before_combining() {
        const DEBRIS: ObjectClass = ObjectClass::Valid(13);
        let mut cmd = default_cmd(RoiSetOperation::Subtract, ObjectClass::Unset);
        cmd.other_class = DEBRIS;
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL); // [10,10,19,19], area 100
        let debris_a = make_square_roi(200_001, 10, 10, 3, DEBRIS); // corner, area 9
        let debris_b = make_square_roi(200_002, 16, 16, 3, DEBRIS); // opposite corner, area 9
        cache.roi_cache.insert(cell.id.clone(), cell);
        cache.roi_cache.insert(debris_a.id.clone(), debris_a);
        cache.roi_cache.insert(debris_b.id.clone(), debris_b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let cell = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(cell.area, 100 - 9 - 9, "both debris blobs must be removed, not just one");
    }

    #[test]
    fn measures_intensities_on_the_result() {
        const CHANNEL: i32 = 0;
        const VALUE: f32 = 4.0;
        let cmd = default_cmd(RoiSetOperation::Subtract, OUT);
        let mut cache = PipelineCache::default();
        let cell = make_square_roi(ID_A, 10, 10, 10, CELL);
        let nucleus = make_square_roi(200_000, 13, 13, 4, NUCLEUS);
        cache.roi_cache.insert(cell.id.clone(), cell);
        cache.roi_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx_with_channel(
            ImageSize {
                width: 100,
                height: 100,
            },
            &mut cache,
            VALUE,
        );
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let cytoplasm = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("cytoplasm ROI not found");
        let intensity = cytoplasm
            .intensities
            .get(&CHANNEL)
            .expect("cytoplasm ROI must have measured intensities");
        assert_eq!(intensity.sum_intensity, cytoplasm.area as f64 * VALUE as f64);
    }

    #[test]
    fn name_is_roi_math() {
        let cmd = default_cmd(RoiSetOperation::And, ObjectClass::Unset);
        assert_eq!(cmd.name(), "RoiMath");
    }
}
