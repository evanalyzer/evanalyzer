//! # images
//!
//! Image work the UI does on pixels it already has.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{ImagePlane, ObjectClass};
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;
use evanalyzer_core::Object;
use kornia_image::ImageSize;
use std::collections::HashSet;

/// A manually annotated object from its painted `mask` (covering `bbox` =
/// `[x1, y1, x2, y2]` in full-image pixels, inclusive) on `plane` of an
/// image of `full_image_width` x `full_image_height`: shape, area and
/// outline from the mask; no intensities - they are measured on the image
/// file afterwards (`measure_on_source`), whatever the zoom it was drawn at.
pub fn annotation_from_mask(
    full_image_width: usize,
    full_image_height: usize,
    mask: BitVec<u64, Lsb0>,
    bbox: [u32; 4],
    plane: ImagePlane,
    object_class: ObjectClass,
) -> ObjectMetricSettings {
    let classes = if object_class == ObjectClass::Unset {
        HashSet::new()
    } else {
        HashSet::from([object_class])
    };
    Object::from_mask_geometry(
        &ImageSize {
            width: full_image_width,
            height: full_image_height,
        },
        mask,
        bbox,
        plane,
        classes,
    )
    .to_object_settings()
}

/// The pixels a brush stroke covers: every pixel within `radius` of the
/// polyline through `points` (full-image pixel coordinates), cut to the
/// image. The mask is relative to the bounding box `[x_min, y_min, x_max,
/// y_max]` (inclusive), as objects keep it. `None` without points or when
/// the stroke lies outside the image.
pub fn stroke_mask(
    points: &[(f32, f32)],
    radius: f32,
    image_width: usize,
    image_height: usize,
) -> Option<(BitVec<u64, Lsb0>, [u32; 4])> {
    if points.is_empty() || image_width == 0 || image_height == 0 {
        return None;
    }
    let r = radius.max(0.5);
    let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in points {
        lo_x = lo_x.min(x);
        lo_y = lo_y.min(y);
        hi_x = hi_x.max(x);
        hi_y = hi_y.max(y);
    }
    let x_min = (lo_x - r).floor().max(0.0);
    let y_min = (lo_y - r).floor().max(0.0);
    let x_max = (hi_x + r).ceil().min(image_width as f32 - 1.0);
    let y_max = (hi_y + r).ceil().min(image_height as f32 - 1.0);
    if x_max < x_min || y_max < y_min {
        return None;
    }
    let (x_min, y_min, x_max, y_max) = (x_min as u32, y_min as u32, x_max as u32, y_max as u32);
    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, width * height);

    // A single point is a segment of length 0.
    let segments: Vec<((f32, f32), (f32, f32))> = if points.len() == 1 {
        vec![(points[0], points[0])]
    } else {
        points.windows(2).map(|w| (w[0], w[1])).collect()
    };
    let r2 = r * r;
    for ((ax, ay), (bx, by)) in segments {
        let sx0 = ((ax.min(bx) - r).floor().max(x_min as f32)) as u32;
        let sx1 = ((ax.max(bx) + r).ceil().min(x_max as f32)) as u32;
        let sy0 = ((ay.min(by) - r).floor().max(y_min as f32)) as u32;
        let sy1 = ((ay.max(by) + r).ceil().min(y_max as f32)) as u32;
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        for y in sy0..=sy1 {
            for x in sx0..=sx1 {
                let (px, py) = (x as f32, y as f32);
                let t = if len2 > 0.0 {
                    (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let (cx, cy) = (ax + t * dx - px, ay + t * dy - py);
                if cx * cx + cy * cy <= r2 {
                    mask.set((y - y_min) as usize * width + (x - x_min) as usize, true);
                }
            }
        }
    }
    tighten(&mask, [x_min, y_min, x_max, y_max])
}

/// What erasing left of an object.
#[derive(Debug, PartialEq)]
pub enum Erased {
    /// The stroke didn't touch it.
    Untouched,
    /// Nothing is left.
    Empty,
    /// The rest, cut to its own bounding box.
    Rest(BitVec<u64, Lsb0>, [u32; 4]),
}

/// `mask` (relative to `bbox`) without the pixels of `stroke` (relative to
/// `stroke_bbox`).
pub fn erase_mask(
    mask: &BitVec<u64, Lsb0>,
    bbox: [u32; 4],
    stroke: &BitVec<u64, Lsb0>,
    stroke_bbox: [u32; 4],
) -> Erased {
    let x0 = bbox[0].max(stroke_bbox[0]);
    let y0 = bbox[1].max(stroke_bbox[1]);
    let x1 = bbox[2].min(stroke_bbox[2]);
    let y1 = bbox[3].min(stroke_bbox[3]);
    if x1 < x0 || y1 < y0 {
        return Erased::Untouched;
    }
    let width = (bbox[2] - bbox[0] + 1) as usize;
    let stroke_width = (stroke_bbox[2] - stroke_bbox[0] + 1) as usize;
    let mut rest = mask.clone();
    let mut changed = false;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = (y - bbox[1]) as usize * width + (x - bbox[0]) as usize;
            let j = (y - stroke_bbox[1]) as usize * stroke_width + (x - stroke_bbox[0]) as usize;
            if stroke.get(j).is_some_and(|b| *b) && rest.get(i).is_some_and(|b| *b) {
                rest.set(i, false);
                changed = true;
            }
        }
    }
    if !changed {
        return Erased::Untouched;
    }
    match tighten(&rest, bbox) {
        Some((mask, bbox)) => Erased::Rest(mask, bbox),
        None => Erased::Empty,
    }
}

/// The pixels of both masks (each relative to its bounding box), in the
/// box around both.
pub fn union_mask(
    a: &BitVec<u64, Lsb0>,
    a_bbox: [u32; 4],
    b: &BitVec<u64, Lsb0>,
    b_bbox: [u32; 4],
) -> (BitVec<u64, Lsb0>, [u32; 4]) {
    let bbox = [
        a_bbox[0].min(b_bbox[0]),
        a_bbox[1].min(b_bbox[1]),
        a_bbox[2].max(b_bbox[2]),
        a_bbox[3].max(b_bbox[3]),
    ];
    let width = (bbox[2] - bbox[0] + 1) as usize;
    let mut out = BitVec::<u64, Lsb0>::repeat(false, width * (bbox[3] - bbox[1] + 1) as usize);
    for (mask, mbox) in [(a, a_bbox), (b, b_bbox)] {
        let mwidth = (mbox[2] - mbox[0] + 1) as usize;
        for i in mask.iter_ones() {
            let x = (i % mwidth) as u32 + mbox[0] - bbox[0];
            let y = (i / mwidth) as u32 + mbox[1] - bbox[1];
            out.set(y as usize * width + x as usize, true);
        }
    }
    (out, bbox)
}

/// `mask` cut to the bounding box of its set pixels; `None` if none is set.
fn tighten(mask: &BitVec<u64, Lsb0>, bbox: [u32; 4]) -> Option<(BitVec<u64, Lsb0>, [u32; 4])> {
    let width = (bbox[2] - bbox[0] + 1) as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for i in mask.iter_ones() {
        let (x, y) = (i % width, i / width);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    if x0 == usize::MAX {
        return None;
    }
    let new_width = x1 - x0 + 1;
    let mut out = BitVec::<u64, Lsb0>::repeat(false, new_width * (y1 - y0 + 1));
    for i in mask.iter_ones() {
        let (x, y) = (i % width, i / width);
        out.set((y - y0) * new_width + (x - x0), true);
    }
    Some((
        out,
        [
            bbox[0] + x0 as u32,
            bbox[1] + y0 as u32,
            bbox[0] + x1 as u32,
            bbox[1] + y1 as u32,
        ],
    ))
}

#[cfg(test)]
mod stroke_tests {
    use super::*;

    #[test]
    fn a_single_point_is_a_disc() {
        let (mask, bbox) = stroke_mask(&[(10.0, 10.0)], 2.0, 100, 100).unwrap();
        assert_eq!(bbox, [8, 8, 12, 12]);
        // The disc of radius 2 has 13 pixels (centres within 2 of 10,10).
        assert_eq!(mask.count_ones(), 13);
    }

    #[test]
    fn a_line_covers_its_length_and_is_cut_to_the_image() {
        let (mask, bbox) = stroke_mask(&[(0.0, 5.0), (20.0, 5.0)], 1.0, 15, 10).unwrap();
        assert_eq!(bbox, [0, 4, 14, 6], "cut at the image's right edge");
        assert_eq!(mask.count_ones(), 15 * 3);
    }

    #[test]
    fn a_stroke_outside_the_image_is_nothing() {
        assert!(stroke_mask(&[(-50.0, -50.0)], 2.0, 10, 10).is_none());
        assert!(stroke_mask(&[], 2.0, 10, 10).is_none());
    }

    #[test]
    fn erasing_removes_the_strokes_pixels_and_tightens_the_box() {
        // A 4 x 2 object at x 10..13, y 0..1; the stroke covers x 10..11.
        let object = BitVec::<u64, Lsb0>::repeat(true, 8);
        let stroke = BitVec::<u64, Lsb0>::repeat(true, 4);
        match erase_mask(&object, [10, 0, 13, 1], &stroke, [10, 0, 11, 1]) {
            Erased::Rest(mask, bbox) => {
                assert_eq!(bbox, [12, 0, 13, 1]);
                assert_eq!(mask.count_ones(), 4);
            }
            other => panic!("expected a rest, got {other:?}"),
        }
    }

    #[test]
    fn a_union_holds_both_masks_in_the_box_around_both() {
        let a = BitVec::<u64, Lsb0>::repeat(true, 4);
        let b = BitVec::<u64, Lsb0>::repeat(true, 1);
        let (mask, bbox) = union_mask(&a, [0, 0, 1, 1], &b, [3, 3, 3, 3]);
        assert_eq!(bbox, [0, 0, 3, 3]);
        assert_eq!(mask.count_ones(), 5);
        assert!(mask[3 * 4 + 3]);
    }

    #[test]
    fn erasing_everything_or_nothing() {
        let object = BitVec::<u64, Lsb0>::repeat(true, 4);
        let stroke = BitVec::<u64, Lsb0>::repeat(true, 16);
        assert_eq!(
            erase_mask(&object, [1, 1, 2, 2], &stroke, [0, 0, 3, 3]),
            Erased::Empty
        );
        assert_eq!(
            erase_mask(&object, [1, 1, 2, 2], &stroke, [10, 10, 13, 13]),
            Erased::Untouched
        );
    }
}
