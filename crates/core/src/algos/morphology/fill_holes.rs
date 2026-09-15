//! # fill_holes
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-10
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

/// Fills enclosed background holes in the segmentation map.
///
/// [Preprocessing] -> [Segment/Threshold] -> [Fill Holes] -> [Connected Components] -> [Watershed] -> [Extract Objects]
///
/// A direct port of ImageJ's `Process > Binary > Fill Holes` command
/// (`ij.plugin.filter.Binary.fill`, originally contributed by Gabriel
/// Landini): a background pixel counts as a "hole" - and is turned into
/// foreground - exactly when it cannot be reached from the image border by a
/// path of background pixels using 4-connectivity.
///
/// Which pixels count as a "hole" is decided the same way ImageJ does it:
/// every non-background pixel is "foreground" regardless of its actual
/// label/class value, so a background pocket enclosed by *any* mix of labels
/// is still a hole. Unlike ImageJ, each hole is then attributed back to the
/// class that actually encloses it - every enclosed background region is
/// grouped into a connected component, and that component is filled with
/// whichever label is most common among its immediately bordering pixels
/// (falling back to `FILL_VALUE` only in the degenerate case of a hole with
/// no foreground neighbor at all). This keeps multi-class segmentation maps
/// correct: a hole inside a class-2 object is filled with 2, not merged with
/// a fixed value that happens to collide with an unrelated class elsewhere
/// in the image.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "instance_segmentation", next = "instance_segmentation")]
pub struct FillHoles {}

impl ImageAlgorithm for FillHoles {
    /// Fills enclosed background holes in the segmentation map - see the
    /// struct docs for the exact algorithm and its ImageJ provenance.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (segmentation, scratch) = ctx.get_segmentation_map_u32_buf()?;
        let size = segmentation.size();
        Self::fill(
            segmentation.as_slice(),
            scratch.as_slice_mut(),
            size.width,
            size.height,
        );
        ctx.swap_scratch_with_segmentations()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Fill Holes"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl FillHoles {
    /// Fallback value for a filled hole that has no foreground neighbor to
    /// take a label from. Should not occur in practice - an enclosed hole is
    /// by definition surrounded by non-background pixels - but keeps `fill`
    /// total. See the struct docs for the normal (per-object) fill value.
    const FILL_VALUE: u32 = 1;

    /// Marks `idx` as reachable from the border ("outside") and pushes it
    /// onto the flood-fill stack, if it's background and not already marked.
    fn seed(idx: usize, input: &[u32], stack: &mut Vec<usize>, outside: &mut [bool]) {
        if input[idx] == 0 && !outside[idx] {
            outside[idx] = true;
            stack.push(idx);
        }
    }

    /// Part of the connected-component walk over one hole: if `nidx` is
    /// another enclosed-background pixel, claims it into the current
    /// component (reusing `outside` as the "already assigned to a hole" flag
    /// so a separate visited buffer isn't needed); if it's foreground,
    /// tallies its label so the component can later be filled with whichever
    /// label borders it most. `border_labels` is a flat `(label, count)`
    /// list rather than a hash map: a hole's border almost always touches
    /// only one or two distinct labels, so a linear scan beats hashing, and
    /// the caller reuses the same allocation across holes.
    fn visit_hole_neighbor(
        nidx: usize,
        input: &[u32],
        outside: &mut [bool],
        component: &mut Vec<usize>,
        border_labels: &mut Vec<(u32, u32)>,
    ) {
        if input[nidx] == 0 {
            if !outside[nidx] {
                outside[nidx] = true;
                component.push(nidx);
            }
        } else {
            let label = input[nidx];
            match border_labels.iter_mut().find(|(l, _)| *l == label) {
                Some((_, count)) => *count += 1,
                None => border_labels.push((label, 1)),
            }
        }
    }

    /// Runs the fill-holes algorithm described in the struct docs over a
    /// flat, row-major `width * height` buffer, writing the result to
    /// `output` (which may start as arbitrary/stale data - every pixel is
    /// always (re)written).
    fn fill(input: &[u32], output: &mut [u32], width: usize, height: usize) {
        output.copy_from_slice(input);
        if width == 0 || height == 0 {
            return;
        }

        // "Outside" = background reachable from the image border.
        let mut outside = vec![false; input.len()];
        let mut stack: Vec<usize> = Vec::new();

        // Seed every border pixel that is background - mirrors Binary.fill's
        // edge scan (left/right column, then top/bottom row).
        for y in 0..height {
            Self::seed(y * width, input, &mut stack, &mut outside);
            Self::seed(y * width + (width - 1), input, &mut stack, &mut outside);
        }
        for x in 0..width {
            Self::seed(x, input, &mut stack, &mut outside);
            Self::seed((height - 1) * width + x, input, &mut stack, &mut outside);
        }

        // 4-connected flood fill from every seed, matching `FloodFiller`.
        while let Some(idx) = stack.pop() {
            let x = idx % width;
            let y = idx / width;
            if x > 0 {
                Self::seed(idx - 1, input, &mut stack, &mut outside);
            }
            if x + 1 < width {
                Self::seed(idx + 1, input, &mut stack, &mut outside);
            }
            if y > 0 {
                Self::seed(idx - width, input, &mut stack, &mut outside);
            }
            if y + 1 < height {
                Self::seed(idx + width, input, &mut stack, &mut outside);
            }
        }

        // Background never reached from the border is an enclosed hole.
        // Group each hole into its own 4-connected component and fill it
        // with whichever label borders it most, so holes inside different
        // objects/classes are filled independently rather than all being
        // stamped with one fixed value. `component` and `border_labels` are
        // hoisted out of the loop and `.clear()`-ed between holes (retaining
        // their capacity) instead of being reallocated per hole.
        let mut component: Vec<usize> = Vec::new();
        let mut border_labels: Vec<(u32, u32)> = Vec::new();

        for start in 0..input.len() {
            if input[start] != 0 || outside[start] {
                continue;
            }

            outside[start] = true;
            component.clear();
            component.push(start);
            border_labels.clear();

            let mut head = 0;
            while head < component.len() {
                let idx = component[head];
                head += 1;
                let x = idx % width;
                let y = idx / width;
                if x > 0 {
                    Self::visit_hole_neighbor(
                        idx - 1,
                        input,
                        &mut outside,
                        &mut component,
                        &mut border_labels,
                    );
                }
                if x + 1 < width {
                    Self::visit_hole_neighbor(
                        idx + 1,
                        input,
                        &mut outside,
                        &mut component,
                        &mut border_labels,
                    );
                }
                if y > 0 {
                    Self::visit_hole_neighbor(
                        idx - width,
                        input,
                        &mut outside,
                        &mut component,
                        &mut border_labels,
                    );
                }
                if y + 1 < height {
                    Self::visit_hole_neighbor(
                        idx + width,
                        input,
                        &mut outside,
                        &mut component,
                        &mut border_labels,
                    );
                }
            }

            let fill_value = border_labels
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|&(label, _)| label)
                .unwrap_or(Self::FILL_VALUE);

            for &idx in component.iter() {
                output[idx] = fill_value;
            }
        }
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_fill_holes_name() {
        assert_eq!(FillHoles {}.name(), "Fill Holes");
    }

    /// The core algorithmic property that distinguishes a faithful ImageJ
    /// port from a "close enough" reimplementation: `FloodFiller.fill` is
    /// documented as 4-connected, not 8-connected.
    ///
    /// Fixture (`0` = background, `9` = foreground):
    /// ```text
    /// 0 9 0
    /// 9 H 9
    /// 0 9 0
    /// ```
    /// The center `H` is orthogonally sealed by the four `9`s, so a correct
    /// 4-connected flood fill can never reach it from a border corner - even
    /// though every corner touches `H` *diagonally*. It must be filled.
    ///
    /// An (incorrect) 8-connected flood fill would instead walk straight
    /// from any border corner (e.g. `(0,0)`) diagonally onto `H` and mark it
    /// "outside", leaving it unfilled - this test fails under that bug.
    #[test]
    fn test_fill_holes_uses_4_connectivity_like_imagejs_floodfiller() {
        #[rustfmt::skip]
        let input: Vec<u32> = vec![
            0, 9, 0,
            9, 0, 9,
            0, 9, 0,
        ];
        let mut output = vec![0u32; 9];

        FillHoles::fill(&input, &mut output, 3, 3);

        assert_eq!(
            output[1 * 3 + 1],
            9,
            "center is enclosed under 4-connectivity and must be filled with the label that encloses it"
        );
        // The four ring pixels must survive unchanged.
        for &idx in &[1usize, 3, 5, 7] {
            assert_eq!(input[idx], output[idx], "ring pixel must not be modified");
        }
        // The four border corners are background themselves (seeded
        // directly, not holes) and must stay background.
        for &idx in &[0usize, 2, 6, 8] {
            assert_eq!(output[idx], 0, "border corner must remain background");
        }
    }

    /// A ring with one straight (non-diagonal) gap in its wall: the interior
    /// is reachable from the border through that gap using plain
    /// 4-connectivity, so it must stay background - the mirror image of the
    /// enclosed case above.
    #[test]
    fn test_fill_holes_does_not_fill_a_hole_open_to_the_border() {
        // Same 3x3 "plus" ring as the enclosed test, but the top of the
        // ring (1,0) is removed, opening a straight path from the top
        // border edge (1,0) down into the center (1,1).
        #[rustfmt::skip]
        let input: Vec<u32> = vec![
            0, 0, 0,
            9, 0, 9,
            0, 9, 0,
        ];
        let mut output = vec![0u32; 9];

        FillHoles::fill(&input, &mut output, 3, 3);

        assert_eq!(
            output[1 * 3 + 1],
            0,
            "center is reachable from the border through the gap and must stay background"
        );
    }

    /// End-to-end test through `execute`/`PipelineContext`, using the same
    /// "plus"-ring shape as the connectivity test but on a real segmentation
    /// map, with a non-`1` label value to also verify existing labels are
    /// copied through unchanged rather than being overwritten.
    #[test]
    fn test_fill_holes_execute_fills_the_enclosed_hole_and_preserves_the_label()
    -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u32; 25];
        // A "plus" ring of label 9 orthogonally sealing the center (2,2).
        data[1 * 5 + 2] = 9; // (2,1)
        data[2 * 5 + 1] = 9; // (1,2)
        data[2 * 5 + 3] = 9; // (3,2)
        data[3 * 5 + 2] = 9; // (2,3)

        let mut ctx = PipelineContext::new_test::<F32Gray>(size)?;
        ctx.segmentation_map = Some(Image::<u32, 1, CpuAllocator>::new(
            size,
            data,
            CpuAllocator,
        )?);

        FillHoles {}.execute(&mut ctx, &mut GlobalPipelineCache::default())?;

        let labels = ctx.segmentation_map.as_ref().expect("no labels found");
        assert_eq!(
            *labels.get_pixel(2, 2, 0)?,
            9,
            "enclosed center must be filled with the label of the object that encloses it"
        );
        assert_eq!(
            *labels.get_pixel(2, 1, 0)?,
            9,
            "ring label must be preserved"
        );
        assert_eq!(
            *labels.get_pixel(0, 0, 0)?,
            0,
            "true background outside the ring must stay background"
        );
        Ok(())
    }

    /// Regression test for a multi-class segmentation map: two separate
    /// enclosed holes, belonging to two differently-labeled objects side by
    /// side, must each be filled with their *own* object's label rather than
    /// both collapsing onto whichever value happens to be first/fixed.
    ///
    /// Fixture (`.` = background, ring of `1`s enclosing hole `A`, ring of
    /// `2`s enclosing hole `B`):
    /// ```text
    /// . 1 . . 2 .
    /// 1 A 1 2 B 2
    /// . 1 . . 2 .
    /// ```
    #[test]
    fn test_fill_holes_fills_each_class_hole_with_its_own_label() {
        #[rustfmt::skip]
        let input: Vec<u32> = vec![
            0, 1, 0, 0, 2, 0,
            1, 0, 1, 2, 0, 2,
            0, 1, 0, 0, 2, 0,
        ];
        let mut output = vec![0u32; input.len()];

        FillHoles::fill(&input, &mut output, 6, 3);

        assert_eq!(
            output[1 * 6 + 1],
            1,
            "hole enclosed by class 1 must be filled with 1, not the other class"
        );
        assert_eq!(
            output[1 * 6 + 4],
            2,
            "hole enclosed by class 2 must be filled with 2, not the other class"
        );
    }
}
