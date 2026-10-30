//! # Row runs
//!
//! Regions built from their rows of pixels - for commands that label a
//! whole image piece by piece (Voronoi, Propagation) without holding a
//! label per pixel of the image.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use bitvec::prelude::*;
use rayon::prelude::*;

/// A row of pixels of one region: label (> 0), row, `[x0, x1)`.
pub(crate) type Run = (u32, u32, u32, u32);

/// One region from its runs, in image coordinates (bbox inclusive, mask
/// row-major over it), with the sums of its pixel coordinates.
#[derive(Default)]
pub(crate) struct RunRegion {
    pub bbox: [u32; 4],
    pub mask: BitVec<u64, Lsb0>,
    pub area: usize,
    pub sum_x: u64,
    pub sum_y: u64,
    pub sum_x2: u64,
    pub sum_y2: u64,
    pub sum_xy: u64,
}

/// Appends the runs of one row of labels (`labels[i]` is pixel `x0 + i`;
/// 0: none) to `runs`.
pub(crate) fn push_row_runs(
    runs: &mut Vec<Run>,
    y: u32,
    x0: u32,
    labels: impl Iterator<Item = u32>,
) {
    let mut run: Option<(u32, u32)> = None;
    let mut x = x0;
    for label in labels {
        match run {
            Some((l, _)) if l == label => {}
            _ => {
                if let Some((l, start)) = run.take()
                    && l != 0
                {
                    runs.push((l, y, start, x));
                }
                run = Some((label, x));
            }
        }
        x += 1;
    }
    if let Some((l, start)) = run
        && l != 0
    {
        runs.push((l, y, start, x));
    }
}

/// Per label `1..=count`, the region of its runs (`None`: no pixel). A
/// pixel is in one run only.
pub(crate) fn regions_from_runs(mut runs: Vec<Run>, count: usize) -> Vec<Option<RunRegion>> {
    runs.par_sort_unstable();
    let mut regions: Vec<Option<RunRegion>> = (0..count).map(|_| None).collect();
    for rows in runs.chunk_by(|a, b| a.0 == b.0) {
        let label = rows[0].0;
        let x_min = rows.iter().map(|r| r.2).min().unwrap_or(0);
        let x_max = rows.iter().map(|r| r.3 - 1).max().unwrap_or(0);
        let (y_min, y_max) = (rows[0].1, rows[rows.len() - 1].1);
        let w = (x_max - x_min + 1) as usize;
        let h = (y_max - y_min + 1) as usize;
        let mut region = RunRegion {
            bbox: [x_min, y_min, x_max, y_max],
            mask: BitVec::<u64, Lsb0>::repeat(false, w * h),
            ..Default::default()
        };
        // Σx and Σx² over 0..to.
        let sum_x = |to: u64| to * to.saturating_sub(1) / 2;
        let sum_x2 = |to: u64| to.saturating_sub(1) * to * (2 * to).saturating_sub(1) / 6;
        for &(_, y, x0, x1) in rows {
            let start = (y - y_min) as usize * w + (x0 - x_min) as usize;
            region.mask[start..start + (x1 - x0) as usize].fill(true);
            let (yu, n) = (y as u64, (x1 - x0) as u64);
            let sx = sum_x(x1 as u64) - sum_x(x0 as u64);
            region.area += n as usize;
            region.sum_x += sx;
            region.sum_y += yu * n;
            region.sum_x2 += sum_x2(x1 as u64) - sum_x2(x0 as u64);
            region.sum_y2 += yu * yu * n;
            region.sum_xy += yu * sx;
        }
        regions[label as usize - 1] = Some(region);
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Area, bounding box and coordinate sums from the rows, as summed
    /// pixel by pixel.
    #[test]
    fn regions_keep_the_sums_of_their_pixels() {
        let mut runs = Vec::new();
        push_row_runs(&mut runs, 2, 0, [0, 0, 0, 1, 1, 1, 0].into_iter());
        push_row_runs(&mut runs, 3, 0, [1, 1, 2, 2].into_iter());
        push_row_runs(&mut runs, 0, 9, [2].into_iter());
        assert_eq!(
            runs,
            vec![(1, 2, 3, 6), (1, 3, 0, 2), (2, 3, 2, 4), (2, 0, 9, 10)]
        );
        let regions = regions_from_runs(runs, 3);
        let one = regions[0].as_ref().unwrap();
        assert_eq!(one.bbox, [0, 2, 5, 3]);
        assert_eq!(one.area, 5);
        // Pixels (3,2) (4,2) (5,2) (0,3) (1,3).
        assert_eq!(one.sum_x, 3 + 4 + 5 + 1);
        assert_eq!(one.sum_y, 3 * 2 + 2 * 3);
        assert_eq!(one.sum_x2, 9 + 16 + 25 + 1);
        assert_eq!(one.sum_y2, 3 * 4 + 2 * 9);
        assert_eq!(one.sum_xy, 2 * (3 + 4 + 5) + 3);
        assert_eq!(
            one.mask.iter().by_vals().collect::<Vec<_>>(),
            [
                false, false, false, true, true, true, true, true, false, false, false, false
            ]
        );
        let two = regions[1].as_ref().unwrap();
        assert_eq!((two.bbox, two.area), ([2, 0, 9, 3], 3));
        assert!(regions[2].is_none());
    }
}
