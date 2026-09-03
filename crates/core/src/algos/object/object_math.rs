//! # object Math Module
//!
//! Provides boolean "object math" (AND / OR / XOR / SUBTRACT) between two object classes.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-28
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module computes a boolean set operation between each `input_class` object ("A")
//! and the union of its overlapping `other_class` ROIs ("B"). Unlike a pixel-level
//! binary-image operation (rasterize the whole class to one mask, apply the op, then
//! re-extract objects), this operates on object pairs: each result stays tied to
//! exactly the one `input_class` object it came from, and only the small bbox window
//! spanned by the operands is ever touched, regardless of image or tile size.
//!
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    object::{BooleanOp, Object, ObjectInit},
};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SizeUnits,
};
use macros::CommandsMeta;

/// Boolean set operation applied between an `input_class` object ("A") and the union of
/// its overlapping `other_class` ROIs ("B").
#[derive(CommandsMeta)]
pub enum ObjectSetOperation {
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

impl From<&ObjectSetOperation> for BooleanOp {
    fn from(op: &ObjectSetOperation) -> Self {
        match op {
            ObjectSetOperation::And => BooleanOp::And,
            ObjectSetOperation::Or => BooleanOp::Or,
            ObjectSetOperation::Xor => BooleanOp::Xor,
            ObjectSetOperation::Subtract => BooleanOp::Subtract,
        }
    }
}

/// Computes a boolean set operation between two object classes, object pair by
/// object pair.
///
/// When more than one `other_class` object overlaps a given input object, all of them
/// are unioned into a single "B" before the operation is applied, so the result
/// doesn't depend on the order they'd otherwise be combined in.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct ObjectMath {
    /// Boolean set operation to apply
    #[cmdsmeta(summary = true)]
    pub operation: ObjectSetOperation,

    /// ROIs carrying this class are the left-hand operand ("A").
    pub input_class: ObjectClass,

    /// ROIs carrying this class are the right-hand operand ("B").
    pub other_class: ObjectClass,

    /// Optional additional label filters applied to `other_class` objects.
    ///
    /// Only `other_class` objects that carry all listed classes are used.
    pub other_filter_classes: Vec<ObjectClass>,

    /// Size unit for `min_overlap_area`
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub size_unit: SizeUnits,

    /// Minimum overlap area before an `other_class` object is treated as a partner
    /// of an input object; objects overlapping less than this are ignored.
    #[cmdsmeta(default = 2)]
    pub min_overlap_area: f32,

    /// If unset, the result replaces the input object in place.
    ///
    /// If set, a new object carrying this class is created for each input object instead,
    /// leaving the input object untouched.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub output_class: ObjectClass,

    /// When an input object has no qualifying overlapping partner: keep it unchanged in
    /// the output (true), or drop it entirely - no output for it at all - (false).
    ///
    /// Note this is a policy override, not the literal mathematical result: e.g. for
    /// `And`, the true result of "A and nothing" is empty, but `keep_unmatched = true`
    /// still leaves A untouched rather than emitting a zero-area object.
    #[cmdsmeta(default = true)]
    pub keep_unmatched: bool,
}

impl ImageAlgorithm for ObjectMath {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.input_class == ObjectClass::Unset || self.other_class == ObjectClass::Unset {
            return Ok(());
        }

        let image_size = ctx.full_image_size();
        let px_sizes = ctx.pixel_sizes();
        let pixel_area_nm2 = px_sizes.px_size_x * px_sizes.px_size_y;
        let min_area_px = self
            .size_unit
            .to_pixel(self.min_overlap_area, pixel_area_nm2);
        let op = BooleanOp::from(&self.operation);

        let other_ids: Vec<ObjectId> = cache
            .object_cache
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
            .object_cache
            .values()
            .filter(|r| r.has_object_class(&self.input_class))
            .map(|r| r.id.clone())
            .collect();

        let mut new_objects: Vec<Object> = Vec::new();
        let mut to_remove: Vec<ObjectId> = Vec::new();
        let replace_in_place =
            self.output_class == ObjectClass::Unset || self.output_class == self.input_class;

        for input_id in &input_ids {
            let Some(input_object) = cache.object_cache.get(input_id) else {
                continue;
            };

            let overlapping: Vec<&Object> = other_ids
                .iter()
                .filter(|oid| *oid != input_id)
                .filter_map(|oid| cache.object_cache.get(oid))
                .filter(|o| {
                    input_object
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

            let geometry = input_object.combine_geometry(&overlapping, op, image_size);
            let (segmentation_class, plane) =
                (input_object.segmentation_class, input_object.plane.clone());

            // Build the result as its own (not-yet-inserted) object so intensities can be
            // sampled against the *new* mask before mutating the cache - `cache` can't
            // be borrowed both mutably and immutably at once.
            let transformed = Object::new(ObjectInit {
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
            let intensities = transformed.measure_intensities(cache);

            if replace_in_place {
                if let Some(object) = cache.object_cache.get_mut(input_id) {
                    object.bbox = transformed.bbox;
                    object.mask_data = transformed.mask_data.clone();
                    object.area = transformed.area;
                    object.touches_edge = transformed.touches_edge;
                    object.sum_x = transformed.sum_x;
                    object.sum_y = transformed.sum_y;
                    object.sum_x2 = transformed.sum_x2;
                    object.sum_y2 = transformed.sum_y2;
                    object.sum_xy = transformed.sum_xy;
                    object.intensities = intensities;
                    object.finalize_geometry();
                }
            } else {
                let mut new_object = transformed;
                new_object.intensities = intensities;
                new_object.add_object_class(self.output_class);
                new_objects.push(new_object);
            }
        }

        for id in to_remove {
            cache.object_cache.remove(&id);
        }
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Object Math"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ImageContainer, ImagePlane, ImageTile, ManagedImage,
        image::PixelSizes,
        pipeline::{
            pipeline::PipelineImageMeta,
            pipeline_cache::{GlobalImageMeta, GlobalPipelineCache},
            pipeline_context::PipelineContext,
        },
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
            ImageContainer::F32Gray(managed).into(),
        )
        .unwrap()
    }

    /// Like `make_ctx`, but also registers a constant-value channel-0 image in
    /// `cache`, so intensity measurement has something real to sample.
    fn make_ctx_with_channel(
        size: ImageSize,
        cache: &mut GlobalPipelineCache,
        value: f32,
    ) -> PipelineContext {
        let ctx = make_ctx(size);
        let img = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![value; size.width * size.height],
            CpuAllocator,
        )
        .unwrap();
        cache.image_meta = GlobalImageMeta {
            full_image_width: size,
            is_rgb: false,
            nr_of_bits: 8,
            pixel_sizes: PixelSizes {
                px_size_x: 1.0,
                px_size_y: 1.0,
                px_size_z: 1.0,
            },
        };
        cache.add_to_channel_cache(
            std::sync::Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
            })),
            0,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: size.width,
                height: size.height,
            },
        );
        ctx
    }

    /// A filled square object, `side` pixels wide, with top-left corner at `(x, y)`.
    fn make_square_object(id: u128, x: u32, y: u32, side: u32, class: ObjectClass) -> Object {
        let area = (side * side) as usize;
        let mask_data = BitVec::<u64, Lsb0>::repeat(true, area);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x, y, x + side - 1, y + side - 1],
            mask_data,
            area,
            plane: ImagePlane::default(),
            ..Default::default()
        });
        object.add_object_class(class);
        object
    }

    const CELL: ObjectClass = ObjectClass::Valid(10);
    const NUCLEUS: ObjectClass = ObjectClass::Valid(11);
    const OUT: ObjectClass = ObjectClass::Valid(12);

    fn default_cmd(operation: ObjectSetOperation, output_class: ObjectClass) -> ObjectMath {
        ObjectMath {
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
        let cmd = default_cmd(ObjectSetOperation::Subtract, OUT);
        let mut cache = GlobalPipelineCache::default();
        // Cell: 10x10 at (10,10) -> bbox [10,10,19,19], area 100.
        let cell = make_square_object(ID_A, 10, 10, 10, CELL);
        // Nucleus: 4x4 at (13,13) -> bbox [13,13,16,16], area 16, fully inside the cell.
        let nucleus = make_square_object(200_000, 13, 13, 4, NUCLEUS);
        cache.object_cache.insert(cell.id.clone(), cell);
        cache.object_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert_eq!(
            cache.object_cache.len(),
            3,
            "original cell and nucleus stay untouched"
        );
        let cytoplasm = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("cytoplasm object not found");
        assert_eq!(cytoplasm.area, 100 - 16);
        assert!(
            !cytoplasm.is_part_of(14, 14),
            "nucleus region must be excluded"
        );
        assert!(
            cytoplasm.is_part_of(10, 10),
            "cell region outside the nucleus must remain"
        );
    }

    #[test]
    fn and_keeps_only_the_overlap() {
        let cmd = default_cmd(ObjectSetOperation::And, OUT);
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL); // [10,10,19,19]
        let nucleus = make_square_object(200_000, 13, 13, 4, NUCLEUS); // [13,13,16,16], inside cell
        cache.object_cache.insert(cell.id.clone(), cell);
        cache.object_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("AND result not found");
        assert_eq!(
            result.area, 16,
            "AND must equal the nucleus area (fully inside the cell)"
        );
        assert_eq!(result.bbox, [13, 13, 16, 16]);
    }

    #[test]
    fn or_extends_beyond_the_input_bbox() {
        let cmd = default_cmd(ObjectSetOperation::Or, OUT);
        let mut cache = GlobalPipelineCache::default();
        let a = make_square_object(ID_A, 10, 10, 10, CELL); // [10,10,19,19], area 100
        let b = make_square_object(200_000, 15, 15, 10, NUCLEUS); // [15,15,24,24], area 100, overlap 25
        cache.object_cache.insert(a.id.clone(), a);
        cache.object_cache.insert(b.id.clone(), b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("OR result not found");
        assert_eq!(
            result.bbox,
            [10, 10, 24, 24],
            "union bbox must cover both operands"
        );
        assert_eq!(result.area, 100 + 100 - 25);
    }

    #[test]
    fn xor_excludes_the_overlap() {
        let cmd = default_cmd(ObjectSetOperation::Xor, OUT);
        let mut cache = GlobalPipelineCache::default();
        let a = make_square_object(ID_A, 10, 10, 10, CELL); // area 100
        let b = make_square_object(200_000, 15, 15, 10, NUCLEUS); // area 100, overlap 25
        cache.object_cache.insert(a.id.clone(), a);
        cache.object_cache.insert(b.id.clone(), b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let result = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("XOR result not found");
        assert_eq!(result.area, 100 + 100 - 2 * 25);
        assert!(
            !result.is_part_of(17, 17),
            "the overlap region must be excluded"
        );
    }

    #[test]
    fn and_with_no_overlap_keeps_input_unchanged_when_keep_unmatched_true() {
        // AND's true mathematical result with nothing is empty, but keep_unmatched is
        // a policy override: it means "don't touch this object at all", not "replace it
        // with the literal (empty) result".
        let cmd = default_cmd(ObjectSetOperation::And, ObjectClass::Unset);
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL);
        let original_area = cell.area;
        cache.object_cache.insert(cell.id.clone(), cell);
        // No nucleus at all.

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let cell = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(cell.area, original_area);
    }

    #[test]
    fn drops_unmatched_input_when_keep_unmatched_false() {
        let mut cmd = default_cmd(ObjectSetOperation::Or, ObjectClass::Unset);
        cmd.keep_unmatched = false;
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL);
        cache.object_cache.insert(cell.id.clone(), cell);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert!(
            cache.object_cache.get(&ObjectId(ID_A)).is_none(),
            "unmatched input must be dropped entirely when keep_unmatched is false"
        );
    }

    #[test]
    fn in_place_replaces_input_when_output_class_unset() {
        let cmd = default_cmd(ObjectSetOperation::Subtract, ObjectClass::Unset);
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL);
        let nucleus = make_square_object(200_000, 13, 13, 4, NUCLEUS);
        cache.object_cache.insert(cell.id.clone(), cell);
        cache.object_cache.insert(nucleus.id.clone(), nucleus);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        assert_eq!(
            cache.object_cache.len(),
            2,
            "no new object in in-place mode"
        );
        let cell = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(cell.area, 100 - 16);
    }

    #[test]
    fn unions_multiple_overlapping_partners_before_combining() {
        const DEBRIS: ObjectClass = ObjectClass::Valid(13);
        let mut cmd = default_cmd(ObjectSetOperation::Subtract, ObjectClass::Unset);
        cmd.other_class = DEBRIS;
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL); // [10,10,19,19], area 100
        let debris_a = make_square_object(200_001, 10, 10, 3, DEBRIS); // corner, area 9
        let debris_b = make_square_object(200_002, 16, 16, 3, DEBRIS); // opposite corner, area 9
        cache.object_cache.insert(cell.id.clone(), cell);
        cache.object_cache.insert(debris_a.id.clone(), debris_a);
        cache.object_cache.insert(debris_b.id.clone(), debris_b);

        let mut ctx = make_ctx(ImageSize {
            width: 100,
            height: 100,
        });
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let cell = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(
            cell.area,
            100 - 9 - 9,
            "both debris blobs must be removed, not just one"
        );
    }

    #[test]
    fn measures_intensities_on_the_result() {
        const CHANNEL: i32 = 0;
        const VALUE: f32 = 4.0;
        let cmd = default_cmd(ObjectSetOperation::Subtract, OUT);
        let mut cache = GlobalPipelineCache::default();
        let cell = make_square_object(ID_A, 10, 10, 10, CELL);
        let nucleus = make_square_object(200_000, 13, 13, 4, NUCLEUS);
        cache.object_cache.insert(cell.id.clone(), cell);
        cache.object_cache.insert(nucleus.id.clone(), nucleus);

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
            .object_cache
            .values()
            .find(|r| r.has_object_class(&OUT))
            .expect("cytoplasm object not found");
        let intensity = cytoplasm
            .intensities
            .get(&CHANNEL)
            .expect("cytoplasm object must have measured intensities");
        assert_eq!(
            intensity.sum_intensity,
            cytoplasm.area as f64 * VALUE as f64
        );
    }

    #[test]
    fn name_is_object_math() {
        let cmd = default_cmd(ObjectSetOperation::And, ObjectClass::Unset);
        assert_eq!(cmd.name(), "Object Math");
    }
}
