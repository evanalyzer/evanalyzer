//! # Object Transform Module
//!
//! Provides functionality for reshaping Regions of Interest (ROIs): scaling them, snapping a
//! circle around them, or replacing them with a fitted ellipse.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-05-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! ## Overview
//! This module implements geometric transforms that change an object's shape and/or size while
//! keeping its bounding-box center fixed. Depending on the configured output class, the
//! transformed shape either replaces the input object in place or is added as a new, separate object.
//!
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    image::PixelSizes,
    object::{Object, ObjectInit, TransformedGeometry},
};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SizeUnits,
};
use kornia_image::ImageSize;
use macros::CommandsMeta;

/// Each variant carries only the parameters it actually uses, so there's no shared field whose
/// meaning shifts depending on which function is selected (e.g. a "factor" that's a unitless
/// multiplier for `Scale` but a length in `size_unit` for `SnapArea`).
#[derive(CommandsMeta)]
pub enum TransformFunction {
    /// Scale the object by the given scale factor.
    ///
    /// Shape keeps and center of the object keeps the same, it is just shrinked or expanded.
    Scale {
        /// Unitless scale factor
        #[cmdsmeta(default = 1.0, min = 0.0, max = 65535.0, step = 1.0)]
        factor: f32,
    },

    /// Draws a circle around the object which is `extra_size` bigger than the object's bounding box.
    SnapArea {
        /// Size added on top of the object's bounding-box diameter
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        extra_size: f32,
        /// Unit `extra_size` is expressed in
        #[cmdsmeta(default = SizeUnits::NanoMeter)]
        unit: SizeUnits,
    },

    /// Draws a circle around the object's bounding box, with `min_diameter` as the minimum diameter.
    MinCircle {
        /// Minimum circle diameter
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        min_diameter: f32,
        /// Unit `min_diameter` is expressed in
        #[cmdsmeta(default = SizeUnits::NanoMeter)]
        unit: SizeUnits,
    },

    /// Draws a circle with exactly `diameter` as diameter around the object.
    ///
    /// If `diameter` is 0, the object's bounding box is used as the diameter instead.
    DrawCircle {
        /// Circle diameter (0 = use the object's bounding box)
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        diameter: f32,
        /// Unit `diameter` is expressed in
        #[cmdsmeta(default = SizeUnits::NanoMeter)]
        unit: SizeUnits,
    },

    /// Replaces the object with the ellipse fitted to its mask.
    FittingEllipse {
        /// Unitless scale factor for the fitted ellipse
        #[cmdsmeta(default = 1.0, min = 0.0, max = 65535.0, step = 1.0)]
        scale: f32,
    },

    /// Grows the object outward by `margin`, following its actual contour (standard flat
    /// dilation with a disk structuring element) - unlike `Scale`, irregular shapes
    /// grow by a uniform margin instead of being stretched proportionally.
    Expand {
        /// Margin added on every side of the mask's contour
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        margin: f32,
        /// Unit `margin` is expressed in
        #[cmdsmeta(default = SizeUnits::NanoMeter)]
        unit: SizeUnits,
    },

    /// Shrinks the object inward by `margin`, following its actual contour (standard flat
    /// erosion with a disk structuring element).
    Shrink {
        /// Margin removed from every side of the mask's contour
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        margin: f32,
        /// Unit `margin` is expressed in
        #[cmdsmeta(default = SizeUnits::NanoMeter)]
        unit: SizeUnits,
    },
}

/// Transforms given ROIs and either replaces the old ones or creates new ones.
///
/// This command applies a geometric transform (scale, circle, fitted ellipse) to every object
/// carrying `input_class`. The transformed shape keeps the original object's bounding-box center.
/// If `output_class` is unset (or equal to `input_class`) the input object is replaced in place;
/// otherwise a new object carrying `output_class` is created alongside the untouched input object.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct TransformObjects {
    /// Geometric transform applied to each input object
    #[cmdsmeta(summary = true)]
    pub function: TransformFunction,

    /// ROIs carrying this class are the input to the transform
    pub input_class: ObjectClass,

    /// If unset, the transformed shape replaces the input object in place.
    ///
    /// If set, a new object carrying this class is created for each transformed input object instead,
    /// leaving the input object untouched.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub output_class: ObjectClass,
}

impl ImageAlgorithm for TransformObjects {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.input_class == ObjectClass::Unset {
            return Ok(());
        }

        let image_size = ctx.full_image_size();
        let px_sizes = ctx.pixel_sizes().clone();

        // Same decision for every matching object, so it's hoisted out of the loop.
        let replace_in_place =
            self.output_class == ObjectClass::Unset || self.output_class == self.input_class;

        let target_ids: Vec<ObjectId> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.has_object_class(&self.input_class))
            .map(|(id, _)| id.clone())
            .collect();

        let mut new_objects: Vec<Object> = Vec::new();

        for id in target_ids {
            let Some(object) = cache.object_cache.get(&id) else {
                continue;
            };
            let geometry = self.transform_geometry(object, image_size, &px_sizes);
            let (segmentation_class, parent_id, plane) = (
                object.segmentation_class,
                object.parent_id.clone(),
                object.plane.clone(),
            );

            // Build the transformed shape as its own (not-yet-inserted) object so
            // intensities can be sampled against the *new* mask via
            // `measure_intensities` before mutating the cache - `cache` can't be
            // borrowed both mutably (to update the cached object) and immutably (to read
            // channel data) at the same time.
            let transformed = Object::new(ObjectInit {
                id: ObjectId::next(),
                segmentation_class,
                parent_id: parent_id.clone(),
                plane: plane.clone(),
                bbox: geometry.bbox,
                area: geometry.area,
                mask_data: geometry.mask_data.clone(),
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
                if let Some(object) = cache.object_cache.get_mut(&id) {
                    apply_geometry(object, geometry);
                    object.intensities = intensities;
                }
            } else {
                let mut new_object = transformed;
                new_object.intensities = intensities;
                new_object.add_object_class(self.output_class);
                new_objects.push(new_object);
            }
        }

        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Transform Objects"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

impl TransformObjects {
    /// Computes the transformed mask geometry for a single object according to `self.function`.
    fn transform_geometry(
        &self,
        object: &Object,
        image_size: ImageSize,
        px_sizes: &PixelSizes,
    ) -> TransformedGeometry {
        // size_unit-based fields are 1-D lengths (radius/diameter), not areas, so average the
        // two pixel axes rather than multiplying them the way area-based commands do.
        let pixel_size_nm = (px_sizes.px_size_x + px_sizes.px_size_y) / 2.0;

        match &self.function {
            TransformFunction::Scale { factor } => {
                object.scaled_geometry(*factor, *factor, image_size)
            }
            TransformFunction::SnapArea { extra_size, unit } => {
                let extra = unit.to_pixel(*extra_size, pixel_size_nm) as f32;
                object.circle_geometry(bbox_max_dimension(object) + extra, image_size)
            }
            TransformFunction::MinCircle { min_diameter, unit } => {
                let min_diameter = unit.to_pixel(*min_diameter, pixel_size_nm) as f32;
                object.circle_geometry(bbox_max_dimension(object).max(min_diameter), image_size)
            }
            TransformFunction::DrawCircle { diameter, unit } => {
                let diameter = unit.to_pixel(*diameter, pixel_size_nm) as f32;
                let diameter = if diameter == 0.0 {
                    bbox_max_dimension(object)
                } else {
                    diameter
                };
                object.circle_geometry(diameter, image_size)
            }
            TransformFunction::FittingEllipse { scale } => {
                let scale = if *scale > 1.0 { *scale } else { 1.0 };
                object.fitting_ellipse_geometry(scale, image_size)
            }
            TransformFunction::Expand { margin, unit } => {
                let margin = unit.to_pixel(*margin, pixel_size_nm) as f32;
                object.dilated_geometry(margin, image_size)
            }
            TransformFunction::Shrink { margin, unit } => {
                let margin = unit.to_pixel(*margin, pixel_size_nm) as f32;
                object.eroded_geometry(margin, image_size)
            }
        }
    }
}

/// Overwrites `object`'s geometry fields with `geometry` and recomputes the derived metrics
/// (perimeter, fitted ellipse) that depend on them.
fn apply_geometry(object: &mut Object, geometry: TransformedGeometry) {
    object.bbox = geometry.bbox;
    object.mask_data = geometry.mask_data;
    object.area = geometry.area;
    object.touches_edge = geometry.touches_edge;
    object.sum_x = geometry.sum_x;
    object.sum_y = geometry.sum_y;
    object.sum_x2 = geometry.sum_x2;
    object.sum_y2 = geometry.sum_y2;
    object.sum_xy = geometry.sum_xy;
    object.finalize_geometry();
}

/// The longer side of the object's bounding box, in pixels.
fn bbox_max_dimension(object: &Object) -> f32 {
    let [x_min, y_min, x_max, y_max] = object.bbox;
    (x_max - x_min + 1).max(y_max - y_min + 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ImageContainer, ImagePlane, ImageTile, ManagedImage,
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

    const CLASS_IN: ObjectClass = ObjectClass::Valid(1);
    const CLASS_OUT: ObjectClass = ObjectClass::Valid(2);
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

    fn run(cmd: &TransformObjects, cache: &mut GlobalPipelineCache, image_size: ImageSize) {
        cmd.execute(&mut make_ctx(image_size), cache).unwrap();
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
            ImageTile::default(),
        );
        ctx
    }

    #[test]
    fn unset_input_class_is_noop() {
        let cmd = TransformObjects {
            function: TransformFunction::Scale { factor: 2.0 },
            input_class: ObjectClass::Unset,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 10, 10, 4, CLASS_IN);
        let original_bbox = object.bbox;
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        assert_eq!(
            cache.object_cache.get(&ObjectId(ID_A)).unwrap().bbox,
            original_bbox
        );
    }

    #[test]
    fn scale_up_replaces_object_in_place() {
        let cmd = TransformObjects {
            function: TransformFunction::Scale { factor: 2.0 },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        // 4x4 square centered at (12, 12) -> scaled by 2 should roughly double to ~8x8.
        let object = make_square_object(ID_A, 10, 10, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        assert_eq!(
            cache.object_cache.len(),
            1,
            "scale must not create a new object"
        );
        let scaled = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = scaled.bbox;
        let width = x_max - x_min + 1;
        let height = y_max - y_min + 1;
        assert!(width >= 7 && width <= 9, "expected ~8px wide, got {width}");
        assert!(
            height >= 7 && height <= 9,
            "expected ~8px tall, got {height}"
        );
        assert!(
            scaled.has_object_class(&CLASS_IN),
            "class is unchanged by a geometric transform"
        );
    }

    #[test]
    fn scale_with_output_class_creates_new_object_and_keeps_original() {
        let cmd = TransformObjects {
            function: TransformFunction::Scale { factor: 2.0 },
            input_class: CLASS_IN,
            output_class: CLASS_OUT,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 10, 10, 4, CLASS_IN);
        let original_bbox = object.bbox;
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        assert_eq!(cache.object_cache.len(), 2, "a new object must be added");
        let original = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(
            original.bbox, original_bbox,
            "input object must stay untouched"
        );

        let transformed = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&CLASS_OUT))
            .expect("transformed object tagged with CLASS_OUT not found");
        assert_ne!(transformed.bbox, original_bbox);
    }

    #[test]
    fn draw_circle_with_zero_factor_uses_bounding_box() {
        let cmd = TransformObjects {
            function: TransformFunction::DrawCircle {
                diameter: 0.0,
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 20, 20, 6, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let circle = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        // Diameter should track the original 6px bounding box, not collapse to ~0.
        assert!(diameter >= 5 && diameter <= 8, "got diameter {diameter}");
    }

    #[test]
    fn min_circle_never_shrinks_below_bounding_box() {
        let cmd = TransformObjects {
            function: TransformFunction::MinCircle {
                min_diameter: 1.0, // smaller than the 6px bounding box
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 20, 20, 6, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let circle = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        assert!(
            diameter >= 5,
            "min circle must not shrink below the bbox, got {diameter}"
        );
    }

    #[test]
    fn snap_area_grows_beyond_bounding_box() {
        let cmd = TransformObjects {
            function: TransformFunction::SnapArea {
                extra_size: 10.0,
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 20, 20, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let circle = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        // bbox diameter (4) + extra_size (10) = ~14
        assert!(
            diameter >= 12,
            "expected snap area to grow well beyond the bbox, got {diameter}"
        );
    }

    #[test]
    fn fitting_ellipse_replaces_square_with_round_shape() {
        let cmd = TransformObjects {
            function: TransformFunction::FittingEllipse { scale: 1.0 },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 20, 20, 8, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let ellipse_object = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        // A filled square's corners are now outside the fitted ellipse, so the area shrinks.
        assert!(ellipse_object.area > 0);
        assert!(
            ellipse_object.area < 64,
            "expected the ellipse to cut the square's corners"
        );
    }

    #[test]
    fn transformed_object_clipped_at_image_edge_touches_edge() {
        let cmd = TransformObjects {
            function: TransformFunction::Scale { factor: 5.0 },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        // Small image: scaling by 5 will push the shape past the image bounds.
        let object = make_square_object(ID_A, 4, 4, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 20,
                height: 20,
            },
        );

        let scaled = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(scaled.touches_edge);
        assert!(
            scaled.bbox[2] <= 19 && scaled.bbox[3] <= 19,
            "mask must stay inside the image"
        );
    }

    #[test]
    fn expand_grows_mask_by_margin_following_the_contour() {
        let cmd = TransformObjects {
            function: TransformFunction::Expand {
                margin: 2.0,
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        // 4x4 square at (20,20) -> bbox [20,20,23,23].
        let object = make_square_object(ID_A, 20, 20, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let expanded = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(
            expanded.bbox,
            [18, 18, 25, 25],
            "a disk margin of 2 must grow the bbox by exactly 2 on every side"
        );
        assert!(
            expanded.area > 16,
            "expanded area must exceed the original 4x4=16"
        );
    }

    #[test]
    fn shrink_shrinks_mask_by_margin_following_the_contour() {
        let cmd = TransformObjects {
            function: TransformFunction::Shrink {
                margin: 2.0,
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        // 8x8 square at (20,20) -> bbox [20,20,27,27].
        let object = make_square_object(ID_A, 20, 20, 8, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let shrunk = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(
            shrunk.bbox,
            [22, 22, 25, 25],
            "a disk margin of 2 must shrink the bbox by exactly 2 on every side"
        );
        assert!(shrunk.area > 0 && shrunk.area <= 16);
    }

    #[test]
    fn shrink_past_the_whole_mask_leaves_nothing() {
        let cmd = TransformObjects {
            function: TransformFunction::Shrink {
                margin: 10.0, // bigger than the 4x4 square itself
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 20, 20, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        run(
            &cmd,
            &mut cache,
            ImageSize {
                width: 100,
                height: 100,
            },
        );

        let shrunk = cache.object_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(shrunk.area, 0);
    }

    #[test]
    fn transform_recomputes_intensities_after_geometry_change() {
        // Regression test: applying a geometric transform must resample intensities
        // against the *new* mask - leaving stale (or, for a brand-new output object,
        // entirely empty/default) intensities was a dormant bug shared by every
        // TransformFunction variant, the same class of bug fixed for Voronoi.
        const CHANNEL: i32 = 0;
        const VALUE: f32 = 7.0;
        let cmd = TransformObjects {
            function: TransformFunction::Scale { factor: 2.0 },
            input_class: CLASS_IN,
            output_class: CLASS_OUT,
        };
        let mut cache = GlobalPipelineCache::default();
        let object = make_square_object(ID_A, 10, 10, 4, CLASS_IN);
        cache.object_cache.insert(object.id.clone(), object);

        let mut ctx = make_ctx_with_channel(
            ImageSize {
                width: 100,
                height: 100,
            },
            &mut cache,
            VALUE,
        );
        cmd.execute(&mut ctx, &mut cache).unwrap();

        let transformed = cache
            .object_cache
            .values()
            .find(|r| r.has_object_class(&CLASS_OUT))
            .expect("transformed object not found");
        let intensity = transformed
            .intensities
            .get(&CHANNEL)
            .expect("transformed object must have measured intensities, not be left empty");
        assert_eq!(
            intensity.sum_intensity,
            transformed.area as f64 * VALUE as f64
        );
    }
}
