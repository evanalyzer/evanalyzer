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
//! This module implements geometric transforms that change an ROI's shape and/or size while
//! keeping its bounding-box center fixed. Depending on the configured output class, the
//! transformed shape either replaces the input ROI in place or is added as a new, separate ROI.
//!
use crate::{
    algos::ImageAlgorithm,
    roi::{Roi, RoiInit, TransformedGeometry},
};
use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass, ObjectId, SizeUnits};
use kornia_image::ImageSize;
use macros::CommandsMeta;

#[derive(CommandsMeta)]
pub enum TransformFunction {
    /// Scale the ROI by the given scale factor.
    ///
    /// Shape keeps and center of the ROI keeps the same, it is just shrinked or expanded.
    Scale,

    /// Draws a circle around the ROI which is `size_factor` bigger than the ROI's bounding box.
    SnapArea,

    /// Draws a circle around the ROI's bounding box, with `size_factor` as the minimum diameter.
    MinCircle,

    /// Draws a circle with exactly `size_factor` as diameter around the ROI.
    ///
    /// If `size_factor` is 0, the ROI's bounding box is used as the diameter instead.
    DrawCircle,

    /// Replaces the ROI with the ellipse fitted to its mask.
    FittingEllipse,
}

/// Transforms given ROIs and either replaces the old ones or creates new ones.
///
/// This command applies a geometric transform (scale, circle, fitted ellipse) to every ROI
/// carrying `input_class`. The transformed shape keeps the original ROI's bounding-box center.
/// If `output_class` is unset (or equal to `input_class`) the input ROI is replaced in place;
/// otherwise a new ROI carrying `output_class` is created alongside the untouched input ROI.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "classify")]
pub struct TransformRois {
    /// Geometric transform applied to each input ROI
    #[cmdsmeta(summary = true)]
    pub function: TransformFunction,

    /// ROIs carrying this class are the input to the transform
    pub input_class: ObjectClass,

    /// If unset, the transformed shape replaces the input ROI in place.
    ///
    /// If set, a new ROI carrying this class is created for each transformed input ROI instead,
    /// leaving the input ROI untouched.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub output_class: ObjectClass,

    /// Unit for `size_factor` when it represents a length (snapArea / minCircle / drawCircle)
    ///
    /// Has no effect for scale and fittingEllipse, where `size_factor` is a unitless multiplier.
    #[cmdsmeta(default = SizeUnits::NanoMeter)]
    pub size_unit: SizeUnits,

    /// Factor based on the selected function
    ///
    /// - scale: unitless scale factor, value between 0.0 and 65535.0
    /// - snapArea: the snap area size in `size_unit`, added to the ROI's bounding box diameter
    /// - minCircle: the minimum circle diameter in `size_unit`
    /// - drawCircle: the circle diameter in `size_unit` (0 = use the ROI's bounding box)
    /// - fittingEllipse: unitless scale factor for the fitted ellipse (default = 1)
    #[cmdsmeta(default = 1.0, min = 0.0, max = 65535.0, step = 1.0, summary = true)]
    pub size_factor: f32,
}

impl ImageAlgorithm for TransformRois {
    fn execute(
        &self,
        ctx: &mut crate::pipeline::pipeline_context::PipelineContext,
        cache: &mut crate::pipeline::pipeline_cache::PipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.input_class == ObjectClass::Unset {
            return Ok(());
        }

        let image_size = ctx.full_image_size();
        let px_sizes = ctx.pixel_sizes();
        // size_factor is a 1-D length (radius/diameter) here, not an area, so average the two
        // axes rather than multiplying them the way area-based commands do.
        let pixel_size_nm = (px_sizes.px_size_x + px_sizes.px_size_y) / 2.0;

        // Same decision for every matching ROI, so it's hoisted out of the loop.
        let replace_in_place =
            self.output_class == ObjectClass::Unset || self.output_class == self.input_class;

        let target_ids: Vec<ObjectId> = cache
            .roi_cache
            .iter()
            .filter(|(_, roi)| roi.has_object_class(&self.input_class))
            .map(|(id, _)| id.clone())
            .collect();

        let mut new_rois: Vec<Roi> = Vec::new();

        for id in target_ids {
            let Some(roi) = cache.roi_cache.get(&id) else {
                continue;
            };
            let geometry = self.transform_geometry(roi, pixel_size_nm, image_size);

            if replace_in_place {
                if let Some(roi) = cache.roi_cache.get_mut(&id) {
                    apply_geometry(roi, geometry);
                }
            } else {
                let roi = cache.roi_cache.get(&id).expect("checked above");
                let mut new_roi = Roi::new(RoiInit {
                    id: ObjectId::next(),
                    segmentation_class: roi.segmentation_class,
                    parent_id: roi.parent_id.clone(),
                    plane: roi.plane.clone(),
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
                new_roi.add_object_class(self.output_class);
                new_rois.push(new_roi);
            }
        }

        for roi in new_rois {
            cache.roi_cache.insert(roi.id.clone(), roi);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "TransformRois"
    }
}

impl TransformRois {
    /// Computes the transformed mask geometry for a single ROI according to `self.function`.
    fn transform_geometry(
        &self,
        roi: &Roi,
        pixel_size_nm: f32,
        image_size: ImageSize,
    ) -> TransformedGeometry {
        match self.function {
            TransformFunction::Scale => {
                roi.scaled_geometry(self.size_factor, self.size_factor, image_size)
            }
            TransformFunction::SnapArea => {
                let extra = self.size_unit.to_pixel(self.size_factor, pixel_size_nm) as f32;
                roi.circle_geometry(bbox_max_dimension(roi) + extra, image_size)
            }
            TransformFunction::MinCircle => {
                let min_diameter = self.size_unit.to_pixel(self.size_factor, pixel_size_nm) as f32;
                roi.circle_geometry(bbox_max_dimension(roi).max(min_diameter), image_size)
            }
            TransformFunction::DrawCircle => {
                let diameter = self.size_unit.to_pixel(self.size_factor, pixel_size_nm) as f32;
                let diameter = if diameter == 0.0 {
                    bbox_max_dimension(roi)
                } else {
                    diameter
                };
                roi.circle_geometry(diameter, image_size)
            }
            TransformFunction::FittingEllipse => {
                let scale = if self.size_factor > 1.0 {
                    self.size_factor
                } else {
                    1.0
                };
                roi.fitting_ellipse_geometry(scale, image_size)
            }
        }
    }
}

/// Overwrites `roi`'s geometry fields with `geometry` and recomputes the derived metrics
/// (perimeter, fitted ellipse) that depend on them.
fn apply_geometry(roi: &mut Roi, geometry: TransformedGeometry) {
    roi.bbox = geometry.bbox;
    roi.mask_data = geometry.mask_data;
    roi.area = geometry.area;
    roi.touches_edge = geometry.touches_edge;
    roi.sum_x = geometry.sum_x;
    roi.sum_y = geometry.sum_y;
    roi.sum_x2 = geometry.sum_x2;
    roi.sum_y2 = geometry.sum_y2;
    roi.sum_xy = geometry.sum_xy;
    roi.finalize_geometry();
}

/// The longer side of the ROI's bounding box, in pixels.
fn bbox_max_dimension(roi: &Roi) -> f32 {
    let [x_min, y_min, x_max, y_max] = roi.bbox;
    (x_max - x_min + 1).max(y_max - y_min + 1) as f32
}

#[cfg(test)]
mod tests {
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
            ImageContainer::F32Gray(managed),
        )
        .unwrap()
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

    fn run(cmd: &TransformRois, cache: &mut PipelineCache, image_size: ImageSize) {
        cmd.execute(&mut make_ctx(image_size), cache).unwrap();
    }

    #[test]
    fn unset_input_class_is_noop() {
        let cmd = TransformRois {
            function: TransformFunction::Scale,
            input_class: ObjectClass::Unset,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 2.0,
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 10, 10, 4, CLASS_IN);
        let original_bbox = roi.bbox;
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        assert_eq!(cache.roi_cache.get(&ObjectId(ID_A)).unwrap().bbox, original_bbox);
    }

    #[test]
    fn scale_up_replaces_roi_in_place() {
        let cmd = TransformRois {
            function: TransformFunction::Scale,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 2.0,
        };
        let mut cache = PipelineCache::default();
        // 4x4 square centered at (12, 12) -> scaled by 2 should roughly double to ~8x8.
        let roi = make_square_roi(ID_A, 10, 10, 4, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        assert_eq!(cache.roi_cache.len(), 1, "scale must not create a new ROI");
        let scaled = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = scaled.bbox;
        let width = x_max - x_min + 1;
        let height = y_max - y_min + 1;
        assert!(width >= 7 && width <= 9, "expected ~8px wide, got {width}");
        assert!(
            height >= 7 && height <= 9,
            "expected ~8px tall, got {height}"
        );
        assert!(scaled.has_object_class(&CLASS_IN), "class is unchanged by a geometric transform");
    }

    #[test]
    fn scale_with_output_class_creates_new_roi_and_keeps_original() {
        let cmd = TransformRois {
            function: TransformFunction::Scale,
            input_class: CLASS_IN,
            output_class: CLASS_OUT,
            size_unit: SizeUnits::Pixels,
            size_factor: 2.0,
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 10, 10, 4, CLASS_IN);
        let original_bbox = roi.bbox;
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        assert_eq!(cache.roi_cache.len(), 2, "a new ROI must be added");
        let original = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        assert_eq!(original.bbox, original_bbox, "input ROI must stay untouched");

        let transformed = cache
            .roi_cache
            .values()
            .find(|r| r.has_object_class(&CLASS_OUT))
            .expect("transformed ROI tagged with CLASS_OUT not found");
        assert_ne!(transformed.bbox, original_bbox);
    }

    #[test]
    fn draw_circle_with_zero_factor_uses_bounding_box() {
        let cmd = TransformRois {
            function: TransformFunction::DrawCircle,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 0.0,
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 20, 20, 6, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        let circle = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        // Diameter should track the original 6px bounding box, not collapse to ~0.
        assert!(diameter >= 5 && diameter <= 8, "got diameter {diameter}");
    }

    #[test]
    fn min_circle_never_shrinks_below_bounding_box() {
        let cmd = TransformRois {
            function: TransformFunction::MinCircle,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 1.0, // smaller than the 6px bounding box
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 20, 20, 6, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        let circle = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        assert!(diameter >= 5, "min circle must not shrink below the bbox, got {diameter}");
    }

    #[test]
    fn snap_area_grows_beyond_bounding_box() {
        let cmd = TransformRois {
            function: TransformFunction::SnapArea,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 10.0,
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 20, 20, 4, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        let circle = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        let [x_min, y_min, x_max, y_max] = circle.bbox;
        let diameter = (x_max - x_min + 1).max(y_max - y_min + 1);
        // bbox diameter (4) + size_factor (10) = ~14
        assert!(diameter >= 12, "expected snap area to grow well beyond the bbox, got {diameter}");
    }

    #[test]
    fn fitting_ellipse_replaces_square_with_round_shape() {
        let cmd = TransformRois {
            function: TransformFunction::FittingEllipse,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 1.0,
        };
        let mut cache = PipelineCache::default();
        let roi = make_square_roi(ID_A, 20, 20, 8, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 100,
            height: 100,
        });

        let ellipse_roi = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        // A filled square's corners are now outside the fitted ellipse, so the area shrinks.
        assert!(ellipse_roi.area > 0);
        assert!(ellipse_roi.area < 64, "expected the ellipse to cut the square's corners");
    }

    #[test]
    fn transformed_roi_clipped_at_image_edge_touches_edge() {
        let cmd = TransformRois {
            function: TransformFunction::Scale,
            input_class: CLASS_IN,
            output_class: ObjectClass::Unset,
            size_unit: SizeUnits::Pixels,
            size_factor: 5.0,
        };
        let mut cache = PipelineCache::default();
        // Small image: scaling by 5 will push the shape past the image bounds.
        let roi = make_square_roi(ID_A, 4, 4, 4, CLASS_IN);
        cache.roi_cache.insert(roi.id.clone(), roi);

        run(&cmd, &mut cache, ImageSize {
            width: 20,
            height: 20,
        });

        let scaled = cache.roi_cache.get(&ObjectId(ID_A)).unwrap();
        assert!(scaled.touches_edge);
        assert!(scaled.bbox[2] <= 19 && scaled.bbox[3] <= 19, "mask must stay inside the image");
    }
}
