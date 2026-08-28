//! # fill_holes
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::pipeline::pipeline_cache::PipelineCache;
use crate::{algos::{ExecutionScope, ImageAlgorithm}, pipeline::pipeline_context::PipelineContext};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;

/// Fills enclosed background holes in the segmentation map.
///
/// A direct port of ImageJ's `Process > Binary > Fill Holes` command
/// (`ij.plugin.filter.Binary.fill`, originally contributed by Gabriel
/// Landini): a background pixel counts as a "hole" - and is turned into
/// foreground - exactly when it cannot be reached from the image border by a
/// path of background pixels using 4-connectivity.
///
/// Like ImageJ's own command, this treats the image as strictly binary: every
/// non-background pixel is "foreground" regardless of its actual label/class
/// value, and a filled hole is stamped with a single fixed value rather than
/// inheriting whatever label happens to surround it. If the segmentation map
/// carries several distinct label values, holes are not attributed back to
/// the object that encloses them - only ImageJ's original background/
/// foreground distinction is reproduced here.
///
/// # Algorithm (matches `ij.process.FloodFiller.fill(x, y)`)
/// 1. Scan every pixel on the image border; for each one that is background
///    (`0`), flood-fill outward from it using 4-connectivity (up/down/left/
///    right only - diagonal neighbors are **not** considered connected),
///    marking every background pixel reached this way as "outside".
/// 2. Any background pixel never marked "outside" is enclosed and becomes
///    foreground. Every non-background pixel is copied through unchanged.
///
/// The 4-connectivity in step 1 is load-bearing, not an implementation
/// detail: a boundary that only touches itself diagonally (8-connected) does
/// **not** block this flood fill, exactly mirroring ImageJ's `FloodFiller`,
/// whose own docs specify a 4-connected fill.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "instance_segmentation", next = "instance_segmentation")]
pub struct FillHoles {}

impl ImageAlgorithm for FillHoles {
    /// Fills enclosed background holes in the segmentation map - see the
    /// struct docs for the exact algorithm and its ImageJ provenance.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
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
    /// The value written into every filled hole. Fixed rather than derived
    /// from the surrounding label - see the struct docs for why.
    const FILL_VALUE: u32 = 1;

    /// Marks `idx` as reachable from the border ("outside") and pushes it
    /// onto the flood-fill stack, if it's background and not already marked.
    fn seed(idx: usize, input: &[u32], stack: &mut Vec<usize>, outside: &mut [bool]) {
        if input[idx] == 0 && !outside[idx] {
            outside[idx] = true;
            stack.push(idx);
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
        for (idx, &v) in input.iter().enumerate() {
            if v == 0 && !outside[idx] {
                output[idx] = Self::FILL_VALUE;
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
            FillHoles::FILL_VALUE,
            "center is enclosed under 4-connectivity and must be filled"
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

        FillHoles {}.execute(&mut ctx, &mut PipelineCache::default())?;

        let labels = ctx.segmentation_map.as_ref().expect("no labels found");
        assert_eq!(
            *labels.get_pixel(2, 2, 0)?,
            FillHoles::FILL_VALUE,
            "enclosed center must be filled"
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
}
