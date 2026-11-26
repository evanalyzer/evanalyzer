//! # post
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist's post-processing steps: a star-convex polygon per grid cell
//! whose probability reaches the threshold, greedy non-maximum suppression
//! by pixel overlap, the kept polygons rasterized into instance labels.

use super::StarDistRaw;
use crate::errors::NnError;

/// `out` holds one label per pixel of the `width x height` tile.
pub(super) fn check_size(out: &[u32], width: usize, height: usize) -> Result<(), NnError> {
    if out.len() == width * height {
        return Ok(());
    }
    Err(NnError::InvalidInput(format!(
        "StarDist: {} labels for a {width}x{height} tile",
        out.len()
    )))
}

/// One star-convex polygon candidate, produced from a single grid cell.
#[cfg_attr(test, derive(Clone))]
pub(super) struct Candidate {
    score: f32,
    /// Inclusive pixel bounds `[min_x, min_y, max_x, max_y]`, clipped to the image.
    bbox: [i64; 4],
    /// Rasterized fill mask local to `bbox`, row-major, width = `bbox` width.
    mask: Bits,
    /// Number of `true` entries in `mask` (i.e. the polygon's pixel area).
    area: usize,
}

/// A candidate's mask, one bit per pixel: a tile has many candidates per
/// object (every grid cell above the threshold), with a byte per pixel
/// their masks took 386 bytes per tile pixel (6.5 GB on a 4096² tile,
/// `tests::tile_ram`).
#[derive(Debug, Clone, PartialEq)]
struct Bits {
    words: Vec<u64>,
    len: usize,
}

impl Bits {
    fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    fn set(&mut self, index: usize) {
        self.words[index / 64] |= 1 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        index < self.len && self.words[index / 64] & (1 << (index % 64)) != 0
    }

    fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(|i| self.get(i))
    }
}

impl std::ops::Index<usize> for Bits {
    type Output = bool;
    fn index(&self, index: usize) -> &bool {
        assert!(index < self.len, "bit {index} of {}", self.len);
        if self.get(index) { &true } else { &false }
    }
}

#[cfg(test)]
impl From<Vec<bool>> for Bits {
    fn from(values: Vec<bool>) -> Self {
        let mut bits = Self::new(values.len());
        for (i, _) in values.iter().enumerate().filter(|(_, v)| **v) {
            bits.set(i);
        }
        bits
    }
}

/// The candidates whose boxes touch each cell of a grid over the tile - for
/// finding a candidate's neighbours without looking at all of them.
struct CandidateGrid {
    /// Cell side (pixels).
    cell: i64,
    columns: usize,
    rows: usize,
    /// Per cell, the candidates touching it.
    cells: Vec<Vec<u32>>,
}

impl CandidateGrid {
    /// Cells of about a typical object's size: a candidate is in a few.
    const CELL: i64 = 64;

    fn new(candidates: &[Candidate]) -> Self {
        let (max_x, max_y) = candidates
            .iter()
            .fold((0, 0), |(x, y), c| (x.max(c.bbox[2]), y.max(c.bbox[3])));
        let cell = Self::CELL;
        let columns = (max_x / cell + 1) as usize;
        let rows = (max_y / cell + 1) as usize;
        let mut cells = vec![Vec::new(); columns * rows];
        for (index, c) in candidates.iter().enumerate() {
            for row in (c.bbox[1] / cell)..=(c.bbox[3] / cell) {
                for column in (c.bbox[0] / cell)..=(c.bbox[2] / cell) {
                    cells[row as usize * columns + column as usize].push(index as u32);
                }
            }
        }
        Self {
            cell,
            columns,
            rows,
            cells,
        }
    }

    /// The candidates in the cells `bbox` touches (one may come more than
    /// once - in several cells).
    fn near(&self, bbox: &[i64; 4]) -> impl Iterator<Item = usize> + '_ {
        let cell = |v: i64, n: usize| ((v.max(0) / self.cell) as usize).min(n - 1);
        let (x0, x1) = (cell(bbox[0], self.columns), cell(bbox[2], self.columns));
        let (y0, y1) = (cell(bbox[1], self.rows), cell(bbox[3], self.rows));
        (y0..=y1).flat_map(move |row| {
            (x0..=x1).flat_map(move |column| {
                self.cells[row * self.columns + column]
                    .iter()
                    .map(|&j| j as usize)
            })
        })
    }
}

/// Builds one star-convex polygon candidate per grid cell whose probability
/// reaches `probability_threshold`, rescaling coordinates back to image
/// resolution if the model predicts on a coarser grid than the input.
pub(super) fn candidates(
    raw: &StarDistRaw,
    width: usize,
    height: usize,
    probability_threshold: f32,
) -> Vec<Candidate> {
    let (prob_flat, dist_flat) = (&raw.prob, &raw.dist);
    let (grid_h, grid_w, n_rays) = (raw.grid_height, raw.grid_width, raw.n_rays);
    let scale_y = height as f32 / grid_h as f32;
    let scale_x = width as f32 / grid_w as f32;
    let dist_scale = (scale_y + scale_x) * 0.5;

    let angles: Vec<(f32, f32)> = (0..n_rays)
        .map(|k| {
            let angle = 2.0 * std::f32::consts::PI * k as f32 / n_rays as f32;
            (angle.cos(), angle.sin())
        })
        .collect();

    let mut candidates = Vec::new();
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let score = prob_flat[idx];
            // NaN fails every `<` comparison, so a NaN score would
            // otherwise sail past the threshold check below (`NaN <
            // threshold` is always false) and later reach the score sort
            // in `non_max_suppression`, which can't order it.
            if !score.is_finite() || score < probability_threshold {
                continue;
            }

            let cy = (gy as f32 + 0.5) * scale_y;
            let cx = (gx as f32 + 0.5) * scale_x;

            let polygon: Vec<(f32, f32)> = angles
                .iter()
                .enumerate()
                .map(|(k, (cos_a, sin_a))| {
                    let r = dist_flat[k * grid_h * grid_w + idx] * dist_scale;
                    (cx + r * cos_a, cy + r * sin_a)
                })
                .collect();

            if let Some(candidate) = build_candidate(score, &polygon, width, height) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

fn build_candidate(
    score: f32,
    polygon: &[(f32, f32)],
    width: usize,
    height: usize,
) -> Option<Candidate> {
    // A single degenerate ray distance (model output NaN/inf for one grid
    // cell, e.g. from a blank/degenerate input tile) poisons this whole
    // star-convex polygon. `f32::min`/`max` below silently ignore NaN
    // operands (so the bbox wouldn't catch it), but the raw polygon is
    // also fed to `rasterize_polygon`'s scanline sort, which can't order
    // a NaN crossing - reject the candidate outright instead of letting
    // a corrupted shape (or a panic) through.
    if polygon
        .iter()
        .any(|(x, y)| !x.is_finite() || !y.is_finite())
    {
        return None;
    }

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in polygon {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let bbox = [
        (min_x.floor() as i64).clamp(0, width as i64 - 1),
        (min_y.floor() as i64).clamp(0, height as i64 - 1),
        (max_x.ceil() as i64).clamp(0, width as i64 - 1),
        (max_y.ceil() as i64).clamp(0, height as i64 - 1),
    ];
    if bbox[2] < bbox[0] || bbox[3] < bbox[1] {
        return None;
    }

    let mask = rasterize_polygon(polygon, bbox);
    let area = mask.count_ones();
    if area == 0 {
        return None;
    }

    Some(Candidate {
        score,
        bbox,
        mask,
        area,
    })
}

/// Fills a simple (non-self-intersecting) polygon using an even-odd scanline
/// rasterizer, sampling at pixel centers. Star-convex polygons (rays cast from
/// a single interior center) are always simple, so this is exact.
fn rasterize_polygon(polygon: &[(f32, f32)], bbox: [i64; 4]) -> Bits {
    let [min_x, min_y, max_x, max_y] = bbox;
    let local_w = (max_x - min_x + 1) as usize;
    let local_h = (max_y - min_y + 1) as usize;
    let mut mask = Bits::new(local_w * local_h);

    let n = polygon.len();
    for row in 0..local_h {
        let py = min_y as f32 + row as f32 + 0.5;
        let mut crossings: Vec<f32> = Vec::new();
        for i in 0..n {
            let (x0, y0) = polygon[i];
            let (x1, y1) = polygon[(i + 1) % n];
            if (y0 <= py && y1 > py) || (y1 <= py && y0 > py) {
                let t = (py - y0) / (y1 - y0);
                crossings.push(x0 + t * (x1 - x0));
            }
        }
        // `build_candidate` already rejects any polygon with a non-finite
        // vertex, so `crossings` shouldn't contain NaN here - `unwrap_or`
        // is defense in depth against a future caller of this function
        // skipping that check, not the primary fix.
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for pair in crossings.chunks_exact(2) {
            let x_start = (pair[0].round() as i64 - min_x).clamp(0, local_w as i64 - 1);
            let x_end = (pair[1].round() as i64 - min_x).clamp(0, local_w as i64 - 1);
            for x in x_start..=x_end {
                mask.set(row * local_w + x as usize);
            }
        }
    }
    mask
}

fn bbox_overlaps(a: &[i64; 4], b: &[i64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Pixel-overlap ratio (intersection / union) between two candidates' masks.
fn iou(a: &Candidate, b: &Candidate) -> f32 {
    let min_x = a.bbox[0].max(b.bbox[0]);
    let min_y = a.bbox[1].max(b.bbox[1]);
    let max_x = a.bbox[2].min(b.bbox[2]);
    let max_y = a.bbox[3].min(b.bbox[3]);
    if max_x < min_x || max_y < min_y {
        return 0.0;
    }

    let a_w = a.bbox[2] - a.bbox[0] + 1;
    let b_w = b.bbox[2] - b.bbox[0] + 1;

    let mut intersection = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let a_idx = ((y - a.bbox[1]) * a_w + (x - a.bbox[0])) as usize;
            let b_idx = ((y - b.bbox[1]) * b_w + (x - b.bbox[0])) as usize;
            if a.mask.get(a_idx) && b.mask.get(b_idx) {
                intersection += 1;
            }
        }
    }
    if intersection == 0 {
        return 0.0;
    }

    let union = a.area + b.area - intersection;
    intersection as f32 / union as f32
}

/// Greedy NMS: candidates are visited in descending score order; a candidate
/// is kept unless it overlaps (by more than `nms_threshold`) a higher-scoring
/// candidate that was already kept. Returns the kept candidates, still sorted
/// by descending score (highest-scoring objects are painted first).
pub(super) fn non_max_suppression(
    candidates: Vec<Candidate>,
    nms_threshold: f32,
) -> Vec<Candidate> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    // `candidates` already skips non-finite scores before a
    // `Candidate` is ever built, so `unwrap_or` here is defense in depth
    // against a future caller constructing candidates some other way,
    // not the primary fix.
    order.sort_by(|&a, &b| {
        candidates[b]
            .score
            .partial_cmp(&candidates[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; candidates.len()];
    let mut keep_order = Vec::new();
    // Only candidates whose boxes overlap can suppress each other: a grid
    // of cells lists the candidates whose box touches each, so a kept
    // candidate is compared with its neighbours only - not with all
    // (590,000 candidates on a 2048² tile took 20 s). The same result:
    // which candidates one suppresses doesn't depend on the order they
    // are looked at in.
    let grid = CandidateGrid::new(&candidates);
    let mut seen = vec![usize::MAX; candidates.len()];

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep_order.push(i);
        for j in grid.near(&candidates[i].bbox) {
            if seen[j] == i {
                continue;
            }
            seen[j] = i;
            if j == i || suppressed[j] || !bbox_overlaps(&candidates[i].bbox, &candidates[j].bbox) {
                continue;
            }
            if iou(&candidates[i], &candidates[j]) > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    let mut slots: Vec<Option<Candidate>> = candidates.into_iter().map(Some).collect();
    keep_order
        .into_iter()
        .map(|i| slots[i].take().unwrap())
        .collect()
}

/// The kept candidates (by descending score) as instance labels into `out`:
/// candidate `k` gets label `k + 1`, the first to claim a pixel wins (one
/// covered entirely by higher-scoring ones keeps no pixel), every other
/// pixel 0. Returns the number of labels given.
pub(super) fn rasterize(kept: &[Candidate], width: usize, out: &mut [u32]) -> u32 {
    out.fill(0);
    for (instance_id, candidate) in (1u32..).zip(kept.iter()) {
        let [min_x, min_y, max_x, max_y] = candidate.bbox;
        let local_w = (max_x - min_x + 1) as usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let local_idx = (y - min_y) as usize * local_w + (x - min_x) as usize;
                if !candidate.mask.get(local_idx) {
                    continue;
                }
                let global_idx = y as usize * width + x as usize;
                if out[global_idx] == 0 {
                    out[global_idx] = instance_id;
                }
            }
        }
    }
    kept.len() as u32
}

#[cfg(test)]
// Indices written as `row * width + column`, row 0 and column 0 included.
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::*;

    /// A raw answer of `prob` and `dist` on a `grid_h x grid_w` grid.
    fn raw_of(
        prob: &[f32],
        dist: &[f32],
        grid_h: usize,
        grid_w: usize,
        n_rays: usize,
    ) -> StarDistRaw {
        StarDistRaw {
            prob: prob.to_vec(),
            dist: dist.to_vec(),
            grid_height: grid_h,
            grid_width: grid_w,
            n_rays,
        }
    }

    fn square(min_x: i64, min_y: i64, side: i64, score: f32) -> Candidate {
        let bbox = [min_x, min_y, min_x + side - 1, min_y + side - 1];
        let w = side as usize;
        Candidate {
            score,
            bbox,
            mask: vec![true; w * w].into(),
            area: w * w,
        }
    }

    /// The grid's suppression keeps exactly the candidates the old one did
    /// (every kept candidate against every candidate), on many overlapping
    /// squares of all sizes - some across several grid cells, some as large
    /// as the tile - and IoU thresholds from strict to lax.
    #[test]
    fn suppression_with_the_grid_keeps_what_comparing_all_kept() {
        let all_pairs = |candidates: &[Candidate], threshold: f32| -> Vec<[i64; 4]> {
            let mut order: Vec<usize> = (0..candidates.len()).collect();
            order.sort_by(|&a, &b| {
                candidates[b]
                    .score
                    .partial_cmp(&candidates[a].score)
                    .unwrap()
            });
            let mut suppressed = vec![false; candidates.len()];
            let mut kept = Vec::new();
            for &i in &order {
                if suppressed[i] {
                    continue;
                }
                kept.push(candidates[i].bbox);
                for &j in &order {
                    if j != i
                        && !suppressed[j]
                        && bbox_overlaps(&candidates[i].bbox, &candidates[j].bbox)
                        && iou(&candidates[i], &candidates[j]) > threshold
                    {
                        suppressed[j] = true;
                    }
                }
            }
            kept
        };
        let mut state = 12345u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % n
        };
        for round in 0..20 {
            let candidates: Vec<Candidate> = (0..400)
                .map(|k| {
                    let side = if k % 97 == 0 {
                        300
                    } else {
                        4 + next(60) as i64
                    };
                    let score = (next(1000) as f32 + k as f32 * 1e-4) / 1000.0;
                    square(next(400) as i64, next(400) as i64, side, score)
                })
                .collect();
            for threshold in [0.0, 0.1, 0.3, 0.7] {
                let expected = all_pairs(&candidates, threshold);
                let kept: Vec<[i64; 4]> = non_max_suppression(candidates.clone(), threshold)
                    .iter()
                    .map(|c| c.bbox)
                    .collect();
                assert_eq!(kept, expected, "round {round}, threshold {threshold}");
            }
        }
    }

    // ---- rasterize_polygon ----

    #[test]
    fn rasterize_polygon_fills_a_simple_square() {
        // A square strictly larger than the 4x4 bbox on every side, so every
        // sampled pixel center (0.5..3.5) is unambiguously interior - corners
        // landing exactly on a pixel-center sample line would instead
        // exercise the scanline rounding convention, which isn't what this
        // test is about.
        let polygon = [(-1.0, -1.0), (5.0, -1.0), (5.0, 5.0), (-1.0, 5.0)];
        let bbox = [0, 0, 3, 3];
        let mask = rasterize_polygon(&polygon, bbox);
        assert_eq!(mask.len(), 16);
        assert!(
            mask.iter().all(|b| b),
            "every pixel center lies inside the square"
        );
    }

    #[test]
    fn rasterize_polygon_leaves_pixels_outside_a_triangle_unfilled() {
        // Right triangle spanning a 4x4 bbox: only the lower-left half should fill.
        let polygon = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0)];
        let bbox = [0, 0, 3, 3];
        let mask = rasterize_polygon(&polygon, bbox);
        let local_w = 4usize;
        // Top-right corner pixel (row 0, col 3) is well outside the triangle.
        assert!(!mask[0 * local_w + 3]);
        // Bottom-left corner pixel (row 3, col 0) is well inside the triangle.
        assert!(mask[3 * local_w + 0]);
    }

    // ---- build_candidate ----

    #[test]
    fn build_candidate_clips_bbox_to_image_bounds() {
        let polygon = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
        let candidate = build_candidate(0.9, &polygon, 4, 4)
            .expect("a large square overlapping the image must produce a candidate");
        assert_eq!(
            candidate.bbox,
            [0, 0, 3, 3],
            "bbox must be clamped to the image dimensions"
        );
    }

    #[test]
    fn build_candidate_returns_none_for_a_zero_area_polygon() {
        // All three points on one horizontal line - zero area, no scanline crossing pair.
        let polygon = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        assert!(build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    #[test]
    fn build_candidate_returns_none_when_entirely_outside_the_image() {
        let polygon = [(-10.0, -10.0), (-6.0, -10.0), (-6.0, -6.0), (-10.0, -6.0)];
        assert!(build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    // ---- build_candidates ----

    #[test]
    fn build_candidates_only_keeps_grid_cells_above_the_probability_threshold() {
        // 1x2 grid; only the second cell clears the threshold.
        let prob_flat = [0.1, 0.9];
        // 4 rays, both cells given the same modest radius.
        let dist_flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let candidates = candidates(&raw_of(&prob_flat, &dist_flat, 1, 2, 4), 20, 10, 0.5);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].score, 0.9);
    }

    #[test]
    fn build_candidates_scales_ray_distance_by_the_grid_to_image_ratio() {
        // 1x1 grid over a 10x10 image (scale factor 10 in both axes) with a
        // unit ray distance - the resulting polygon should span roughly
        // radius-10 around the image center, clipped to the image bounds.
        let prob_flat = [1.0];
        let n_rays = 8;
        let dist_flat = vec![1.0f32; n_rays];
        let candidates = candidates(&raw_of(&prob_flat, &dist_flat, 1, 1, n_rays), 10, 10, 0.5);
        assert_eq!(candidates.len(), 1);
        let bbox = candidates[0].bbox;
        // The unscaled radius (1px) would collapse to a single pixel; scaled
        // by ~10x it must span a large fraction of the 10x10 image instead.
        assert!(
            bbox[2] - bbox[0] > 4,
            "bbox must reflect the scaled radius, not the raw 1px ray length"
        );
    }

    // ---- bbox_overlaps ----

    #[test]
    fn bbox_overlaps_true_for_touching_boxes() {
        assert!(bbox_overlaps(&[0, 0, 3, 3], &[3, 3, 6, 6]));
    }

    #[test]
    fn bbox_overlaps_false_for_disjoint_boxes() {
        assert!(!bbox_overlaps(&[0, 0, 3, 3], &[4, 4, 6, 6]));
    }

    // ---- iou ----

    #[test]
    fn iou_is_zero_for_disjoint_candidates() {
        let a = square(0, 0, 4, 0.9);
        let b = square(10, 10, 4, 0.5);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_is_zero_when_bboxes_overlap_but_the_masks_share_no_pixel() {
        // A occupies only the top-left 2x2 of its 4x4 bbox [0,0,3,3]; B
        // occupies only the bottom-right 2x2 of its 4x4 bbox [2,2,5,5]. The
        // two *bboxes* overlap (in [2,3]x[2,3]), but neither mask is filled
        // there - `iou`'s own bbox fast-path (checked by
        // `iou_is_zero_for_disjoint_candidates` above) can't catch this; only
        // the mask intersection loop finding zero true pixels does.
        let mut a_mask = vec![false; 16];
        for row in 0..2 {
            for col in 0..2 {
                a_mask[row * 4 + col] = true;
            }
        }
        let a = Candidate {
            score: 0.9,
            bbox: [0, 0, 3, 3],
            mask: a_mask.into(),
            area: 4,
        };
        let mut b_mask = vec![false; 16];
        for row in 2..4 {
            for col in 2..4 {
                b_mask[row * 4 + col] = true;
            }
        }
        let b = Candidate {
            score: 0.5,
            bbox: [2, 2, 5, 5],
            mask: b_mask.into(),
            area: 4,
        };
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_is_one_for_identical_candidates() {
        let a = square(0, 0, 4, 0.9);
        let b = square(0, 0, 4, 0.5);
        assert_eq!(iou(&a, &b), 1.0);
    }

    #[test]
    fn iou_computes_the_intersection_over_union_ratio() {
        // Two 4x4 squares (area 16 each) overlapping in a 2x4 strip (area 8).
        // union = 16 + 16 - 8 = 24, iou = 8/24.
        let a = square(0, 0, 4, 0.9);
        let b = square(2, 0, 4, 0.5);
        let iou = iou(&a, &b);
        assert!((iou - 8.0 / 24.0).abs() < 1e-6, "got {iou}");
    }

    // ---- non_max_suppress ----

    #[test]
    fn non_max_suppress_keeps_every_candidate_when_none_overlap() {
        let candidates = vec![
            square(0, 0, 2, 0.9),
            square(10, 10, 2, 0.8),
            square(20, 20, 2, 0.7),
        ];
        let kept = non_max_suppression(candidates, 0.3);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn non_max_suppress_drops_the_lower_scoring_of_two_heavily_overlapping_candidates() {
        // Identical boxes (iou = 1.0) - well above any positive threshold.
        let candidates = vec![square(0, 0, 4, 0.4), square(0, 0, 4, 0.9)];
        let kept = non_max_suppression(candidates, 0.3);
        assert_eq!(kept.len(), 1);
        assert_eq!(
            kept[0].score, 0.9,
            "the higher-scoring candidate must survive"
        );
    }

    #[test]
    fn non_max_suppress_keeps_both_when_overlap_is_below_threshold() {
        // 4x4 squares shifted by 3px overlap in a 1x4 strip: iou = 4/(16+16-4) = 4/28 ≈ 0.14.
        let candidates = vec![square(0, 0, 4, 0.9), square(3, 0, 4, 0.8)];
        let kept = non_max_suppression(candidates, 0.3);
        assert_eq!(
            kept.len(),
            2,
            "0.14 overlap must not exceed a 0.3 suppression threshold"
        );
    }

    #[test]
    fn non_max_suppress_returns_candidates_sorted_by_descending_score() {
        let candidates = vec![
            square(0, 0, 2, 0.3),
            square(50, 50, 2, 0.9),
            square(100, 100, 2, 0.6),
        ];
        let kept = non_max_suppression(candidates, 0.3);
        let scores: Vec<f32> = kept.iter().map(|c| c.score).collect();
        assert_eq!(scores, vec![0.9, 0.6, 0.3]);
    }

    // ---- NaN model output no longer panics ----
    //
    // A degenerate/blank tile or an incompatible model export can make the
    // model emit NaN. Fixed at three points: `candidates`
    // skips non-finite scores before a `Candidate` is even built,
    // `build_candidate` rejects any polygon with a non-finite vertex, and
    // both `partial_cmp(...).unwrap()` sort call sites were changed to
    // `unwrap_or(Ordering::Equal)` as defense in depth for callers that
    // bypass the two filters above (as these tests deliberately do, to
    // exercise that fallback directly).

    #[test]
    fn rasterize_polygon_does_not_panic_on_a_nan_vertex() {
        let polygon = [(0.0, 0.0), (f32::NAN, 4.0), (4.0, 4.0)];
        // No assertion on the resulting mask's shape - a NaN vertex makes the
        // polygon meaningless, but `rasterize_polygon` (unlike
        // `build_candidate`, which rejects this polygon outright) is a
        // low-level primitive that must at least not crash.
        let _ = rasterize_polygon(&polygon, [0, 0, 3, 3]);
    }

    #[test]
    fn non_max_suppress_does_not_panic_when_a_candidate_score_is_nan() {
        let candidates = vec![square(0, 0, 2, f32::NAN), square(10, 10, 2, 0.5)];
        let kept = non_max_suppression(candidates, 0.3);
        assert_eq!(kept.len(), 2, "non-overlapping candidates are both kept");
    }

    #[test]
    fn build_candidates_skips_a_nan_score_grid_cell() {
        // 1x2 grid: cell 0 has a NaN score (e.g. degenerate model output),
        // cell 1 clears the threshold normally.
        let prob_flat = [f32::NAN, 0.9];
        let dist_flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let candidates = candidates(&raw_of(&prob_flat, &dist_flat, 1, 2, 4), 20, 10, 0.5);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].score, 0.9);
    }

    #[test]
    fn build_candidate_returns_none_for_a_polygon_with_a_nan_vertex() {
        let polygon = [(1.0, 1.0), (f32::NAN, 5.0), (5.0, 5.0), (1.0, 5.0)];
        assert!(build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    #[test]
    fn build_candidate_returns_none_for_a_polygon_with_an_infinite_vertex() {
        let polygon = [(1.0, 1.0), (f32::INFINITY, 5.0), (5.0, 5.0), (1.0, 5.0)];
        assert!(build_candidate(0.9, &polygon, 10, 10).is_none());
    }

    // ---- rasterize ----

    #[test]
    fn rasterize_resets_stale_labels_when_no_candidate_survives() {
        let mut out = vec![9u32; 9];
        assert_eq!(rasterize(&[], 3, &mut out), 0);
        assert_eq!(out, vec![0; 9]);
    }

    #[test]
    fn rasterize_resets_stale_pixels_outside_every_candidate() {
        let mut out = vec![9u32; 16];
        assert_eq!(rasterize(&[square(0, 0, 2, 0.9)], 4, &mut out), 1);
        for y in 0..4 {
            for x in 0..4 {
                let expected = u32::from(x < 2 && y < 2);
                assert_eq!(out[y * 4 + x], expected, "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn rasterize_skips_the_false_cells_of_a_non_rectangular_mask() {
        let candidate = Candidate {
            score: 0.9,
            bbox: [0, 0, 1, 1],
            mask: vec![true, false, false, true].into(),
            area: 2,
        };
        let mut out = vec![0u32; 4];
        rasterize(&[candidate], 2, &mut out);
        assert_eq!(out, vec![1, 0, 0, 1], "only the diagonal cells are claimed");
    }

    #[test]
    fn rasterize_gives_a_contested_pixel_to_the_first_candidate() {
        let candidates = vec![square(0, 0, 2, 0.9), square(1, 0, 2, 0.5)];
        let mut out = vec![0u32; 9];
        assert_eq!(rasterize(&candidates, 3, &mut out), 2);
        assert_eq!(
            out[0 * 3 + 1],
            1,
            "the first candidate wins the contested pixel"
        );
        assert_eq!(out[1 * 3 + 1], 1);
        assert_eq!(out[0 * 3 + 0], 1);
        assert_eq!(out[0 * 3 + 2], 2);
    }
}
