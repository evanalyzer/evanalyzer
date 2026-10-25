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
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
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

    /// Grows the objects outward by `margin` without growing into each other: every
    /// background pixel within `margin` of an object goes to the nearest object (as
    /// CellProfiler's ExpandOrShrinkObjects "Expand objects by a specified number of
    /// pixels"). With a large margin the objects grow until they touch.
    ExpandWithoutOverlap {
        /// How far the objects grow at most
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
            .filter(|(_, object)| object.is_of(&self.input_class, ctx.scope))
            .map(|(id, _)| id.clone())
            .collect();

        let mut new_objects: Vec<Object> = Vec::new();

        // Expanding without overlap decides every object's new pixels together.
        let mut grown = match &self.function {
            TransformFunction::ExpandWithoutOverlap { margin, unit } => {
                let pixel_size_nm = (px_sizes.px_size_x + px_sizes.px_size_y) / 2.0;
                // Not `to_pixel`: a fractional distance counts (2.5 px reaches
                // the diagonal neighbours 2 px over and 1 px up).
                let margin = match unit {
                    SizeUnits::Pixels => *margin as f64,
                    SizeUnits::NanoMeter => (*margin / pixel_size_nm) as f64,
                };
                let (ids, objects): (Vec<ObjectId>, Vec<&Object>) = target_ids
                    .iter()
                    .filter_map(|id| cache.object_cache.get(id).map(|o| (id.clone(), o)))
                    .unzip();
                ids.into_iter()
                    .zip(expand_without_overlap(&objects, margin, image_size))
                    .collect()
            }
            _ => std::collections::HashMap::new(),
        };

        for id in target_ids {
            let Some(object) = cache.object_cache.get(&id) else {
                continue;
            };
            let geometry = match grown.remove(&id) {
                Some(geometry) => geometry,
                None => self.transform_geometry(object, image_size, &px_sizes),
            };
            let (segmentation_class, plane) = (object.segmentation_class, object.plane.clone());

            // Build the transformed shape as its own (not-yet-inserted) object so
            // intensities can be sampled against the *new* mask via
            // `measure_intensities` before mutating the cache - `cache` can't be
            // borrowed both mutably (to update the cached object) and immutably (to read
            // channel data) at the same time.
            let transformed = Object::new(ObjectInit {
                id: ObjectId::next(),
                segmentation_class,
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
            let intensities = transformed.measure_intensities(cache);

            if replace_in_place {
                if let Some(object) = cache.object_cache.get_mut(&id) {
                    apply_geometry(object, geometry);
                    object.intensities = intensities;
                }
            } else {
                let mut new_object = transformed;
                new_object.intensities = intensities;
                new_object.add_object_class(self.output_class, ctx.scope)?;
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

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        match self.function {
            // As CellProfiler's ExpandOrShrinkObjects, with the feature transform.
            TransformFunction::ExpandWithoutOverlap { .. } => vec![
                &CitationMetadata::DANMAYR,
                &CitationMetadata::CELLPROFILER,
                &CitationMetadata::MAURER_2003,
            ],
            _ => vec![&CitationMetadata::DANMAYR],
        }
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
            // Decided for all objects together (`expand_without_overlap`).
            TransformFunction::ExpandWithoutOverlap { .. } => TransformedGeometry {
                bbox: object.bbox,
                mask_data: object.mask_data.clone(),
                area: object.area,
                touches_edge: object.touches_edge,
                sum_x: object.sum_x,
                sum_y: object.sum_y,
                sum_x2: object.sum_x2,
                sum_y2: object.sum_y2,
                sum_xy: object.sum_xy,
            },
        }
    }
}

/// Side length of the blocks `expand_without_overlap` works in (plus the
/// margin around each), so a whole slide never needs one raster of its size.
const EXPAND_BLOCK: usize = 2048;

/// The grown shape of each of `objects` (same order): every background
/// pixel - covered by none of them - within `margin` pixels (Euclidean) of
/// an object pixel goes to the object of the nearest such pixel. As
/// CellProfiler's ExpandOrShrinkObjects (`expand` in
/// cellprofiler/library/functions/object_processing.py: SciPy's
/// `distance_transform_edt` with its indices, `distance <= margin`), with
/// the same feature transform and so the same pixel where two objects are
/// equally near. Where objects overlap, a pixel counts as the later one's.
///
/// Worked out in blocks of `EXPAND_BLOCK`², each with `margin` around it:
/// the nearest object pixel of a pixel that is within `margin` of one lies
/// in that border, so a block decides its pixels exactly as the whole image
/// would.
fn expand_without_overlap(
    objects: &[&Object],
    margin: f64,
    image_size: ImageSize,
) -> Vec<TransformedGeometry> {
    expand_without_overlap_in_blocks(objects, margin, image_size, EXPAND_BLOCK)
}

/// [`expand_without_overlap`] in blocks of `block`² pixels.
fn expand_without_overlap_in_blocks(
    objects: &[&Object],
    margin: f64,
    image_size: ImageSize,
    block: usize,
) -> Vec<TransformedGeometry> {
    let (width, height) = (image_size.width, image_size.height);
    let mut added: Vec<Vec<(u32, u32)>> = vec![Vec::new(); objects.len()];
    if margin > 0.0 && width > 0 && height > 0 && !objects.is_empty() {
        let halo = margin.ceil() as usize;
        for by in (0..height).step_by(block) {
            for bx in (0..width).step_by(block) {
                let window = [bx, by, (bx + block).min(width), (by + block).min(height)];
                expand_block(objects, margin, halo, window, [width, height], &mut added);
            }
        }
    }
    objects
        .iter()
        .zip(added)
        .map(|(object, pixels)| grown_geometry(object, &pixels, image_size))
        .collect()
}

/// `expand_without_overlap` for the pixels of `block` (`[x0, y0, x1, y1)`):
/// adds each background pixel within `margin` to its nearest object's
/// `added` pixels.
fn expand_block(
    objects: &[&Object],
    margin: f64,
    halo: usize,
    block: [usize; 4],
    [width, height]: [usize; 2],
    added: &mut [Vec<(u32, u32)>],
) {
    use crate::algos::libs::feature_transform::nearest_features;
    let [bx0, by0, bx1, by1] = block;
    let (wx0, wy0) = (bx0.saturating_sub(halo), by0.saturating_sub(halo));
    let (wx1, wy1) = ((bx1 + halo).min(width), (by1 + halo).min(height));
    let (ww, wh) = (wx1 - wx0, wy1 - wy0);
    // The window's labels: object index + 1, 0 for the background.
    let mut labels = vec![0u32; ww * wh];
    let mut any = false;
    for (index, object) in objects.iter().enumerate() {
        let [ox0, oy0, ox1, oy1] = object.bbox.map(|v| v as usize);
        if ox1 < wx0 || oy1 < wy0 || ox0 >= wx1 || oy0 >= wy1 {
            continue;
        }
        let ow = ox1 - ox0 + 1;
        for y in oy0.max(wy0)..=oy1.min(wy1 - 1) {
            for x in ox0.max(wx0)..=ox1.min(wx1 - 1) {
                if object
                    .mask_data
                    .get((y - oy0) * ow + (x - ox0))
                    .is_some_and(|b| *b)
                {
                    labels[(y - wy0) * ww + (x - wx0)] = index as u32 + 1;
                    any = true;
                }
            }
        }
    }
    if !any {
        return;
    }
    let ft = nearest_features(ww, wh, |x, y| labels[y * ww + x] != 0);
    for y in by0..by1 {
        for x in bx0..bx1 {
            let (lx, ly) = (x - wx0, y - wy0);
            if labels[ly * ww + lx] != 0 {
                continue;
            }
            let Some((r, c)) = ft.get(ly * ww + lx) else {
                continue;
            };
            let (dr, dc) = ((r - ly as i64) as f64, (c - lx as i64) as f64);
            if (dr * dr + dc * dc).sqrt() <= margin {
                let label = labels[r as usize * ww + c as usize];
                added[label as usize - 1].push((x as u32, y as u32));
            }
        }
    }
}

/// `object` with `added` pixels (outside it) set as well.
fn grown_geometry(
    object: &Object,
    added: &[(u32, u32)],
    image_size: ImageSize,
) -> TransformedGeometry {
    let [mut x0, mut y0, mut x1, mut y1] = object.bbox;
    for &(x, y) in added {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    let w = (x1 - x0 + 1) as usize;
    let mut mask =
        bitvec::vec::BitVec::<u64, bitvec::order::Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
    let [ox0, oy0, ox1, oy1] = object.bbox;
    let ow = (ox1 - ox0 + 1) as usize;
    let pixels = (oy0..=oy1)
        .flat_map(|y| (ox0..=ox1).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            object
                .mask_data
                .get((y - oy0) as usize * ow + (x - ox0) as usize)
                .is_some_and(|b| *b)
        })
        .chain(added.iter().copied());
    let mut geometry = TransformedGeometry::default();
    let (max_x, max_y) = (image_size.width as u32 - 1, image_size.height as u32 - 1);
    for (x, y) in pixels {
        mask.set((y - y0) as usize * w + (x - x0) as usize, true);
        let (xu, yu) = (x as u64, y as u64);
        geometry.area += 1;
        geometry.sum_x += xu;
        geometry.sum_y += yu;
        geometry.sum_x2 += xu * xu;
        geometry.sum_y2 += yu * yu;
        geometry.sum_xy += xu * yu;
        geometry.touches_edge |= x == 0 || y == 0 || x == max_x || y == max_y;
    }
    geometry.touches_edge |= object.touches_edge;
    geometry.bbox = [x0, y0, x1, y1];
    geometry.mask_data = mask;
    geometry
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
    use crate::image::Point2d;
    use crate::pipeline::scope::Scope;
    use crate::{
        ImageContainer, ImagePlane, ImageTile, ManagedImage,
        pipeline::{
            pipeline::PipelineImageMeta,
            pipeline_cache::{GlobalImageMeta, GlobalPipelineCache},
            pipeline_context::PipelineContext,
        },
    };
    use bitvec::prelude::*;
    use kornia_image::{Image, ImageSize};

    const CLASS_IN: ObjectClass = ObjectClass::Global(1);
    const CLASS_OUT: ObjectClass = ObjectClass::Global(2);
    const ID_A: u128 = 100_000;

    fn make_ctx(size: ImageSize) -> PipelineContext {
        let img = Image::<f32, 1>::new(size, vec![0.0f32; size.width * size.height]).unwrap();
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
        object.add_object_class(class, Scope::Global).unwrap();
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
        let img = Image::<f32, 1>::new(size, vec![value; size.width * size.height]).unwrap();
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
            scaled.is_of(&CLASS_IN, Scope::Global),
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
            .find(|r| r.is_of(&CLASS_OUT, Scope::Global))
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
            .find(|r| r.is_of(&CLASS_OUT, Scope::Global))
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

    /// One case of `tests/expand_objects_cellprofiler.json`.
    #[derive(serde::Deserialize)]
    struct ExpandCase {
        name: String,
        width: usize,
        height: usize,
        distance: f64,
        labels: Vec<u32>,
        expanded: Vec<u32>,
    }

    /// An object per label of a label image, with the label as its id.
    fn objects_from_labels(labels: &[u32], width: usize) -> Vec<Object> {
        let max = labels.iter().copied().max().unwrap_or(0);
        (1..=max)
            .filter_map(|label| {
                let pixels: Vec<(u32, u32)> = (0..labels.len())
                    .filter(|&p| labels[p] == label)
                    .map(|p| ((p % width) as u32, (p / width) as u32))
                    .collect();
                let x0 = pixels.iter().map(|p| p.0).min()?;
                let y0 = pixels.iter().map(|p| p.1).min()?;
                let x1 = pixels.iter().map(|p| p.0).max()?;
                let y1 = pixels.iter().map(|p| p.1).max()?;
                let w = (x1 - x0 + 1) as usize;
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
                for (x, y) in &pixels {
                    mask.set((y - y0) as usize * w + (x - x0) as usize, true);
                }
                Some(Object::new(ObjectInit {
                    id: ObjectId(label as u128),
                    bbox: [x0, y0, x1, y1],
                    area: pixels.len(),
                    mask_data: mask,
                    ..Default::default()
                }))
            })
            .collect()
    }

    /// Paints `geometries` (in order) into a label image, object i as `ids[i]`.
    fn paint(
        geometries: &[TransformedGeometry],
        ids: &[u32],
        width: usize,
        height: usize,
    ) -> Vec<u32> {
        let mut out = vec![0u32; width * height];
        for (geometry, id) in geometries.iter().zip(ids) {
            let [x0, y0, x1, y1] = geometry.bbox.map(|v| v as usize);
            let w = x1 - x0 + 1;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if geometry.mask_data[(y - y0) * w + (x - x0)] {
                        out[y * width + x] = *id;
                    }
                }
            }
        }
        out
    }

    /// The same pixels as CellProfiler's ExpandOrShrinkObjects (fixture from
    /// `tests/expand_objects_cellprofiler.py`) - also when worked out in
    /// small blocks, so the blocks' borders change nothing.
    #[test]
    fn expanding_without_overlap_gives_cellprofilers_pixels() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/expand_objects_cellprofiler.json"
        );
        let cases: Vec<ExpandCase> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(cases.len(), 12);
        for case in &cases {
            let objects = objects_from_labels(&case.labels, case.width);
            let refs: Vec<&Object> = objects.iter().collect();
            let ids: Vec<u32> = objects.iter().map(|o| o.id.0 as u32).collect();
            let size = ImageSize {
                width: case.width,
                height: case.height,
            };
            for block in [EXPAND_BLOCK, 7] {
                let grown = expand_without_overlap_in_blocks(&refs, case.distance, size, block);
                let got = paint(&grown, &ids, case.width, case.height);
                let wrong = (0..got.len())
                    .filter(|&p| got[p] != case.expanded[p])
                    .count();
                assert_eq!(
                    wrong, 0,
                    "{} (blocks of {block}): {wrong} pixels differ",
                    case.name
                );
                for (geometry, object) in grown.iter().zip(&objects) {
                    assert_eq!(
                        geometry.area,
                        geometry.mask_data.count_ones(),
                        "{}",
                        case.name
                    );
                    assert!(geometry.area >= object.area);
                }
            }
        }
    }

    /// Through the step: neighbours stop each other, the objects are
    /// replaced in place and measured again.
    #[test]
    fn expand_without_overlap_replaces_the_objects_grown() {
        // Two points 6 px apart on a row, grown by 4 px: they meet halfway.
        let size = ImageSize {
            width: 20,
            height: 5,
        };
        let mut labels = vec![0u32; 100];
        labels[2 * 20 + 7] = 1;
        labels[2 * 20 + 13] = 2;
        let mut cache = GlobalPipelineCache::default();
        for mut object in objects_from_labels(&labels, 20) {
            object.add_object_class(CLASS_IN, Scope::Global).unwrap();
            cache.object_cache.insert(object.id.clone(), object);
        }
        let step = TransformObjects {
            function: TransformFunction::ExpandWithoutOverlap {
                margin: 4.0,
                unit: SizeUnits::Pixels,
            },
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
        };
        step.execute(&mut make_ctx(size), &mut cache).unwrap();
        let a = cache.object_cache.get(&ObjectId(1)).unwrap();
        let b = cache.object_cache.get(&ObjectId(2)).unwrap();
        assert!(
            a.overlaps(b).is_none(),
            "neighbours don't grow into each other"
        );
        // On the row: a takes 8..=10 (and 3..=6), b 11/12..; the pixel at
        // 10 is 3 from a and 3 from b - SciPy's tie goes to a.
        assert!(a.is_part_of(10, 2) && !b.is_part_of(10, 2));
        assert!(b.is_part_of(11, 2) && !a.is_part_of(11, 2));
        assert!(a.is_part_of(3, 2) && !a.is_part_of(2, 2));
        assert_eq!(a.bbox[0], 3);
    }

    /// Speed and peak RAM of Expand Without Overlap: a 9 x 9 object every
    /// 40 px on a `BENCH_SIZE`² image (default 8192: 42,025 objects), grown
    /// by `BENCH_MARGIN` px (default 8) - release, one per process:
    /// `cargo test --release -p evanalyzer_core --lib bench_expand_without_overlap -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_expand_without_overlap() {
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (size, margin) = (env("BENCH_SIZE", 8192), env("BENCH_MARGIN", 8));
        let mut objects = Vec::new();
        let mut id = 1u128;
        for y in (0..size.saturating_sub(9)).step_by(40) {
            for x in (0..size.saturating_sub(9)).step_by(40) {
                objects.push(Object::new(ObjectInit {
                    id: ObjectId(id),
                    bbox: [x as u32, y as u32, x as u32 + 8, y as u32 + 8],
                    area: 81,
                    mask_data: BitVec::<u64, Lsb0>::repeat(true, 81),
                    ..Default::default()
                }));
                id += 1;
            }
        }
        let refs: Vec<&Object> = objects.iter().collect();
        let image = ImageSize {
            width: size,
            height: size,
        };
        let start = std::time::Instant::now();
        let grown = expand_without_overlap(&refs, margin as f64, image);
        let elapsed = start.elapsed();
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let area: usize = grown.iter().map(|g| g.area).sum();
        eprintln!(
            "BENCH expand without overlap: {} objects on {size}², margin {margin}: {elapsed:?}, {area} px, {hwm}",
            objects.len()
        );
    }
}
