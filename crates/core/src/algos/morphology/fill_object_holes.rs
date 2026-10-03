//! # fill_object_holes
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;

/// Fills the holes inside every object, one object at a time.
///
/// [AI Cellpose / StarDist Segmentation | Watershed | Connected Components] -> [Fill Object Holes] -> [Extract Objects]
///
/// Works on the objects (instance map), so it has to come after a step that
/// creates objects and before Extract Objects. A pixel becomes part of an
/// object when it is enclosed by that object alone; a gap enclosed by several
/// touching objects together is left as it is. An object lying completely
/// inside another object's hole becomes part of the enclosing object. Filled
/// pixels get the class of the object they now belong to.
///
/// Use Fill Holes instead to fill holes in the segmentation map before the
/// objects are created (e.g. right after a Threshold).
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", next = "measure")]
pub struct FillObjectHoles {}

impl ImageAlgorithm for FillObjectHoles {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (Some(segmentation), Some(instances)) =
            (ctx.segmentation_map.as_mut(), ctx.instance_map.as_mut())
        else {
            return Err(InternalErrors::FormatMismatch {
                expected: "Objects (segmentation and instance map) - place Fill Object Holes \
                           after a step that creates objects, e.g. Connected Components, \
                           Watershed, Cellpose or StarDist"
                    .into(),
                found: "None (Buffer not initialized)".into(),
            });
        };
        let size = instances.size();
        Self::fill(
            segmentation.as_slice_mut(),
            instances.as_slice_mut(),
            size.width,
            size.height,
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Fill Object Holes"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl FillObjectHoles {
    /// Fills the holes of every object in `instances` and gives the filled
    /// pixels the object's class in `segmentation` (taken from its first pixel
    /// in row-major order). Afterwards the instance ids are renumbered to
    /// `1..=n` keeping their order - Extract Objects expects ids without gaps,
    /// and an object absorbed by an enclosing one leaves a gap.
    fn fill(segmentation: &mut [u32], instances: &mut [u32], width: usize, height: usize) {
        let max_id = instances.iter().copied().max().unwrap_or(0) as usize;
        if max_id == 0 {
            return;
        }
        let mut class_of = vec![None; max_id + 1];
        for (&id, &class) in instances.iter().zip(segmentation.iter()) {
            if id != 0 && class_of[id as usize].is_none() {
                class_of[id as usize] = Some(class);
            }
        }

        let before = instances.to_vec();
        fill_instance_holes(instances, width, height);
        for (i, (&old, &new)) in before.iter().zip(instances.iter()).enumerate() {
            if old != new {
                segmentation[i] = class_of[new as usize].unwrap_or(segmentation[i]);
            }
        }

        let mut present = vec![false; max_id + 1];
        for &id in instances.iter() {
            present[id as usize] = true;
        }
        let mut new_id = vec![0u32; max_id + 1];
        let mut next = 1;
        for id in 1..=max_id {
            if present[id] {
                new_id[id] = next;
                next += 1;
            }
        }
        for id in instances.iter_mut() {
            *id = new_id[*id as usize];
        }
    }
}

/// Fills the holes of every object of an instance map on its own, the way
/// Cellpose's `fill_holes_and_remove_small_masks` does: a pixel inside an
/// object's bounding box that cannot reach the box border through pixels of
/// *other* labels (4-connectivity) becomes part of the object - including
/// pixels of a smaller object enclosed by it. Unlike [`FillHoles`](crate::algos::FillHoles), a gap
/// enclosed by several touching objects is not a hole of any of them and
/// stays as it is.
///
/// Objects are filled in label order, so a later object enclosing an earlier
/// one takes its pixels. `labels` is a row-major `width * height` instance
/// map, `0` = background.
pub fn fill_instance_holes(labels: &mut [u32], width: usize, height: usize) {
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    if max_label == 0 {
        return;
    }
    // Bounding boxes taken once up front, like Cellpose's `find_objects`.
    let mut bbox = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; max_label + 1];
    for y in 0..height {
        for x in 0..width {
            let l = labels[y * width + x] as usize;
            if l != 0 {
                let b = &mut bbox[l];
                b[0] = b[0].min(y);
                b[1] = b[1].min(x);
                b[2] = b[2].max(y);
                b[3] = b[3].max(x);
            }
        }
    }

    let mut outside: Vec<bool> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for label in 1..=max_label as u32 {
        let [y0, x0, y1, x1] = bbox[label as usize];
        if y0 == usize::MAX {
            continue;
        }
        let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
        let at = |bx: usize, by: usize| (y0 + by) * width + x0 + bx;
        outside.clear();
        outside.resize(bw * bh, false);
        // Flood the non-object pixels reachable from the box border.
        for by in 0..bh {
            for bx in 0..bw {
                let on_border = by == 0 || bx == 0 || by == bh - 1 || bx == bw - 1;
                if on_border && labels[at(bx, by)] != label && !outside[by * bw + bx] {
                    outside[by * bw + bx] = true;
                    stack.push(by * bw + bx);
                }
            }
        }
        while let Some(i) = stack.pop() {
            let (bx, by) = (i % bw, i / bw);
            let mut visit = |nx: usize, ny: usize| {
                let j = ny * bw + nx;
                if !outside[j] && labels[at(nx, ny)] != label {
                    outside[j] = true;
                    stack.push(j);
                }
            };
            if bx > 0 {
                visit(bx - 1, by);
            }
            if bx + 1 < bw {
                visit(bx + 1, by);
            }
            if by > 0 {
                visit(bx, by - 1);
            }
            if by + 1 < bh {
                visit(bx, by + 1);
            }
        }
        for by in 0..bh {
            for bx in 0..bw {
                if !outside[by * bw + bx] {
                    labels[at(bx, by)] = label;
                }
            }
        }
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    // ---- fill_instance_holes ----

    #[test]
    fn fill_instance_holes_fills_a_hole_inside_one_object() {
        #[rustfmt::skip]
        let mut labels = vec![
            1, 1, 1,
            1, 0, 1,
            1, 1, 1,
        ];
        fill_instance_holes(&mut labels, 3, 3);
        assert_eq!(labels, vec![1; 9]);
    }

    #[test]
    fn fill_instance_holes_leaves_a_gap_enclosed_by_several_objects() {
        // The 0 is enclosed, but by two objects together - a hole of neither.
        #[rustfmt::skip]
        let mut labels = vec![
            1, 1, 2,
            1, 0, 2,
            1, 2, 2,
        ];
        let before = labels.clone();
        fill_instance_holes(&mut labels, 3, 3);
        assert_eq!(labels, before);
    }

    #[test]
    fn fill_instance_holes_lets_an_enclosing_object_take_an_enclosed_one() {
        // Like Cellpose's fill_voids: object 2 lies in object 1's hole.
        #[rustfmt::skip]
        let mut labels = vec![
            1, 1, 1, 1,
            1, 2, 0, 1,
            1, 1, 1, 1,
        ];
        fill_instance_holes(&mut labels, 4, 3);
        assert_eq!(labels, vec![1; 12]);
    }

    #[test]
    fn fill_instance_holes_keeps_a_gap_open_to_the_bounding_box_border() {
        // The notch at the top connects to the border: not a hole.
        #[rustfmt::skip]
        let mut labels = vec![
            1, 0, 1,
            1, 0, 1,
            1, 1, 1,
        ];
        let before = labels.clone();
        fill_instance_holes(&mut labels, 3, 3);
        assert_eq!(labels, before);
    }

    #[test]
    fn fill_instance_holes_uses_4_connectivity_for_the_outside() {
        // The center 0 touches the outside only diagonally: it is a hole.
        #[rustfmt::skip]
        let mut labels = vec![
            0, 1, 0,
            1, 0, 1,
            0, 1, 0,
        ];
        fill_instance_holes(&mut labels, 3, 3);
        assert_eq!(labels[4], 1);
        assert_eq!(labels[0], 0, "corners reach the border and stay background");
    }

    #[test]
    fn fill_instance_holes_on_an_empty_map_does_nothing() {
        let mut labels = vec![0; 4];
        fill_instance_holes(&mut labels, 2, 2);
        assert_eq!(labels, vec![0; 4]);
    }

    fn ctx_with(
        segmentation: Vec<u32>,
        instances: Vec<u32>,
        w: usize,
        h: usize,
    ) -> PipelineContext {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mut ctx =
            PipelineContext::new_from_image_test(Image::new(size, vec![0f32; w * h]).unwrap())
                .unwrap();
        ctx.segmentation_map = Some(Image::new(size, segmentation).unwrap());
        ctx.instance_map = Some(Image::new(size, instances).unwrap());
        ctx
    }

    fn run(ctx: &mut PipelineContext) {
        FillObjectHoles {}
            .execute(ctx, &mut GlobalPipelineCache::default())
            .unwrap();
    }

    #[test]
    fn fills_an_objects_hole_with_its_id_and_class() {
        #[rustfmt::skip]
        let instances = vec![
            1, 1, 1,
            1, 0, 1,
            1, 1, 1,
        ];
        let mut ctx = ctx_with(instances.iter().map(|&i| i * 3).collect(), instances, 3, 3);
        run(&mut ctx);
        assert_eq!(ctx.get_instance_map().unwrap().as_slice(), &[1; 9]);
        assert_eq!(ctx.get_segmentation_map().unwrap().as_slice(), &[3; 9]);
    }

    #[test]
    fn leaves_a_gap_between_touching_objects() {
        #[rustfmt::skip]
        let instances = vec![
            1, 1, 2,
            1, 0, 2,
            1, 2, 2,
        ];
        let segmentation: Vec<u32> = instances.iter().map(|&i| (i != 0) as u32).collect();
        let mut ctx = ctx_with(segmentation.clone(), instances.clone(), 3, 3);
        run(&mut ctx);
        assert_eq!(
            ctx.get_instance_map().unwrap().as_slice(),
            instances.as_slice()
        );
        assert_eq!(
            ctx.get_segmentation_map().unwrap().as_slice(),
            segmentation.as_slice()
        );
    }

    #[test]
    fn an_absorbed_object_leaves_no_gap_in_the_ids() {
        // Object 1 lies in object 2's hole and is absorbed; object 3 must be
        // renumbered to 2 so Extract Objects sees ids 1..=n.
        #[rustfmt::skip]
        let instances = vec![
            2, 2, 2, 0, 3,
            2, 1, 2, 0, 3,
            2, 2, 2, 0, 3,
        ];
        #[rustfmt::skip]
        let segmentation = vec![
            5, 5, 5, 0, 6,
            5, 7, 5, 0, 6,
            5, 5, 5, 0, 6,
        ];
        let mut ctx = ctx_with(segmentation, instances, 5, 3);
        run(&mut ctx);
        #[rustfmt::skip]
        let expected_instances = [
            1, 1, 1, 0, 2,
            1, 1, 1, 0, 2,
            1, 1, 1, 0, 2,
        ];
        assert_eq!(
            ctx.get_instance_map().unwrap().as_slice(),
            &expected_instances
        );
        // The absorbed pixel takes the enclosing object's class.
        assert_eq!(ctx.get_segmentation_map().unwrap().as_slice()[6], 5);
    }

    #[test]
    fn no_objects_changes_nothing() {
        let mut ctx = ctx_with(vec![0; 4], vec![0; 4], 2, 2);
        run(&mut ctx);
        assert_eq!(ctx.get_instance_map().unwrap().as_slice(), &[0; 4]);
    }

    #[test]
    fn errors_with_a_placement_hint_without_a_segmentation_map() {
        let mut ctx = ctx_with(vec![0; 4], vec![0; 4], 2, 2);
        ctx.segmentation_map = None;
        let err = FillObjectHoles {}
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap_err();
        assert!(
            matches!(err, InternalErrors::FormatMismatch { expected, .. } if expected.contains("Connected Components"))
        );
    }

    #[test]
    fn command_metadata() {
        let cmd = FillObjectHoles {};
        assert_eq!(cmd.name(), "Fill Object Holes");
        assert!(cmd.cite().is_none());
        assert!(matches!(cmd.execution_scope(), ExecutionScope::Tile));
    }
}
