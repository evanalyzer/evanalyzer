//! # post
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose's post-processing steps: the flows followed from every cell
//! pixel (Cellpose's dynamics), the pixels grouped into masks by where they
//! ended, masks whose shape doesn't match the predicted flows removed, the
//! masks scaled back to the image, small ones removed and the rest
//! renumbered.
//!
//! Portions ported from Cellpose, Copyright © 2020 Howard Hughes Medical
//! Institute, BSD-3-Clause - see the third-party notices (About > Licenses).

use super::{CellposeSamParams, CellposeSamRaw};
use crate::errors::NnError;

/// Cellpose scales the training flows by `5`, so the predicted flows are
/// divided by the same factor before integration to keep each Euler step
/// near one pixel.
const FLOW_SCALE: f32 = 5.0;

/// `out` holds one label per pixel of the image (`(width, height)`).
pub(super) fn check_size(out: &[u32], (width, height): (usize, usize)) -> Result<(), NnError> {
    if out.len() == width * height {
        return Ok(());
    }
    Err(NnError::InvalidInput(format!(
        "Cellpose: {} labels for a {width}x{height} image",
        out.len()
    )))
}

/// The pixels taking part in the dynamics: cell probability from
/// `threshold` on.
pub(super) fn cell_pixels(cell_prob: &[f32], threshold: f32) -> Vec<bool> {
    cell_prob.iter().map(|&p| p >= threshold).collect()
}

/// The masks at the network's size (`(width, height)`, `0` = background):
/// with `cellpose_dynamics` Cellpose's interpolated dynamics and its mask
/// building from seeds, else the integer dynamics and one object per sink.
pub(super) fn masks(
    raw: &CellposeSamRaw,
    cells: &[bool],
    (width, height): (usize, usize),
    params: &CellposeSamParams,
) -> Vec<u32> {
    let (fy, fx) = (&raw.flow_y, &raw.flow_x);
    if params.cellpose_dynamics {
        let ends = follow_flows_interpolated(fy, fx, cells, width, height, params.flow_iterations);
        masks_from_seeds(&ends, cells, width, height)
    } else {
        let ends = follow_flows(fy, fx, cells, width, height, params.flow_iterations);
        label_sinks(&ends, cells, width, height)
    }
}

/// Cellpose's flow-error check with `threshold` (0: none), see
/// [`remove_bad_flows`].
pub(super) fn remove_bad_flow_masks(
    labels: &mut [u32],
    raw: &CellposeSamRaw,
    (width, height): (usize, usize),
    threshold: f32,
) {
    if threshold > 0.0 {
        remove_bad_flows(labels, &raw.flow_y, &raw.flow_x, width, height, threshold);
    }
}

/// Nearest-neighbor resize of a label map (OpenCV's `INTER_NEAREST`, as used
/// by Cellpose to scale masks back up) from `from` to `to` (`(width,
/// height)`) into `out`: target pixel `d` takes source pixel
/// `floor(d * src / dst)`.
pub(super) fn resize_nearest(
    labels: &[u32],
    from: (usize, usize),
    to: (usize, usize),
    out: &mut [u32],
) {
    if from == to {
        out.copy_from_slice(labels);
        return;
    }
    let ((src_width, src_height), (dst_width, dst_height)) = (from, to);
    let src_x: Vec<usize> = (0..dst_width)
        .map(|x| (x * src_width / dst_width).min(src_width - 1))
        .collect();
    for (y, row) in out.chunks_exact_mut(dst_width).take(dst_height).enumerate() {
        let source = (y * src_height / dst_height).min(src_height - 1) * src_width;
        for (value, &x) in row.iter_mut().zip(&src_x) {
            *value = labels[source + x];
        }
    }
}

/// Removes objects smaller than `min_size` pixels (0 keeps every object)
/// and renumbers the others to `1..=n` in the order of their labels.
/// Returns `n`.
pub(super) fn remove_small_and_renumber(labels: &mut [u32], min_size: usize) -> u32 {
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    if max_label == 0 {
        return 0;
    }
    let mut sizes = vec![0usize; max_label + 1];
    for &label in labels.iter() {
        sizes[label as usize] += 1;
    }
    let mut remap = vec![0u32; max_label + 1];
    let mut next_id = 1u32;
    for label in 1..=max_label {
        if sizes[label] >= min_size.max(1) {
            remap[label] = next_id;
            next_id += 1;
        }
    }
    for label in labels.iter_mut() {
        *label = remap[*label as usize];
    }
    next_id - 1
}

/// Cellpose's dynamics (`dynamics.steps_interp`): every cell pixel takes
/// `flow_iterations` Euler steps along the flows divided by `FLOW_SCALE`,
/// sampled with bilinear interpolation. Like Cellpose, the flows are zero
/// outside the cell pixels, and the sampling reproduces its
/// `grid_sample(align_corners=False)` on `[0, L-1]`-normalized
/// coordinates: position `p` samples pixel `p * L / (L - 1) - 0.5`,
/// pixels outside the image count as zero. Returns each pixel's final
/// `(y, x)`, truncated to whole pixels (non-cell pixels keep their own).
pub(super) fn follow_flows_interpolated(
    flow_y: &[f32],
    flow_x: &[f32],
    is_cell: &[bool],
    width: usize,
    height: usize,
    flow_iterations: usize,
) -> Vec<(usize, usize)> {
    use rayon::prelude::*;
    let niter = flow_iterations.max(1);
    let masked = |flow: &[f32]| -> Vec<f32> {
        flow.iter()
            .zip(is_cell)
            .map(|(&f, &c)| if c { f / FLOW_SCALE } else { 0.0 })
            .collect()
    };
    let (fy, fx) = (masked(flow_y), masked(flow_x));
    // Sampling coordinate of position `p` along an axis of length `len`.
    let to_sample = |p: f32, len: usize| -> f32 {
        if len < 2 {
            0.0
        } else {
            p * len as f32 / (len - 1) as f32 - 0.5
        }
    };
    let bilinear = |field: &[f32], uy: f32, ux: f32| -> f32 {
        let (y0, x0) = (uy.floor(), ux.floor());
        let (ty, tx) = (uy - y0, ux - x0);
        let mut value = 0.0;
        for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
            for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                let (yy, xx) = (y0 as i64 + dy, x0 as i64 + dx);
                if yy >= 0 && xx >= 0 && (yy as usize) < height && (xx as usize) < width {
                    value += wy * wx * field[yy as usize * width + xx as usize];
                }
            }
        }
        value
    };
    let (max_y, max_x) = ((height - 1) as f32, (width - 1) as f32);
    (0..width * height)
        .into_par_iter()
        .map(|idx| {
            let (y, x) = (idx / width, idx % width);
            if !is_cell[idx] {
                return (y, x);
            }
            let (mut py, mut px) = (y as f32, x as f32);
            for _ in 0..niter {
                let (uy, ux) = (to_sample(py, height), to_sample(px, width));
                let step_y = bilinear(&fy, uy, ux);
                let step_x = bilinear(&fx, uy, ux);
                py = (py + step_y).clamp(0.0, max_y);
                px = (px + step_x).clamp(0.0, max_x);
            }
            (py as usize, px as usize)
        })
        .collect()
}

/// Cellpose's mask building (`dynamics.get_masks_torch`): a histogram of
/// where the cell pixels ended up is searched for peaks (local maxima in
/// a 5x5 window with more than 10 pixels); each peak grows for 5 steps
/// (3x3) into the histogram bins holding more than 2 pixels, within 5
/// pixels of the peak. A cell pixel takes the label of the peak region
/// it ended up in, or `0` if it reached none. Where regions overlap, the
/// peak with more pixels wins. Labels are `1..=n` in order of first
/// appearance. (Cellpose's removal of objects above 40 % of the image is
/// left to a size filter in the pipeline.)
pub(super) fn masks_from_seeds(
    final_positions: &[(usize, usize)],
    is_cell: &[bool],
    width: usize,
    height: usize,
) -> Vec<u32> {
    const PAD: usize = 20;
    let (hw, hh) = (width + 2 * PAD, height + 2 * PAD);
    let mut hist = vec![0u32; hw * hh];
    for (idx, &(y, x)) in final_positions.iter().enumerate() {
        if is_cell[idx] {
            hist[(y + PAD) * hw + x + PAD] += 1;
        }
    }
    let local_max = max_filter(&hist, hw, hh, 2);
    let mut seeds: Vec<usize> = (0..hw * hh)
        .filter(|&i| hist[i] > 10 && hist[i] == local_max[i])
        .collect();
    seeds.sort_by_key(|&i| hist[i]); // stable: row-major among equals
    if seeds.is_empty() {
        return vec![0; width * height];
    }

    // Grow every seed inside its 11x11 neighborhood; seeds never sit
    // closer than PAD to the histogram border, so the window always fits.
    const R: usize = 5;
    const SIDE: usize = 2 * R + 1;
    let mut region = vec![0u32; hw * hh];
    let mut mask = [0u32; SIDE * SIDE];
    for (k, &seed) in seeds.iter().enumerate() {
        let (sy, sx) = (seed / hw, seed % hw);
        let bin = |wy: usize, wx: usize| hist[(sy + wy - R) * hw + sx + wx - R];
        mask.fill(0);
        mask[R * SIDE + R] = 1;
        for _ in 0..5 {
            mask = max_filter(&mask, SIDE, SIDE, 1).try_into().unwrap();
            for wy in 0..SIDE {
                for wx in 0..SIDE {
                    if bin(wy, wx) <= 2 {
                        mask[wy * SIDE + wx] = 0;
                    }
                }
            }
        }
        for wy in 0..SIDE {
            for wx in 0..SIDE {
                if mask[wy * SIDE + wx] != 0 {
                    let i = (sy + wy - R) * hw + sx + wx - R;
                    region[i] = region[i].max(k as u32 + 1);
                }
            }
        }
    }

    let mut labels = vec![0u32; width * height];
    for (idx, &(y, x)) in final_positions.iter().enumerate() {
        if is_cell[idx] {
            labels[idx] = region[(y + PAD) * hw + x + PAD];
        }
    }
    renumber(&mut labels);
    labels
}

/// Maximum over a `(2 * radius + 1)`² window (cut off at the borders).
fn max_filter(values: &[u32], width: usize, height: usize, radius: usize) -> Vec<u32> {
    let mut rows = vec![0u32; values.len()];
    for y in 0..height {
        for x in 0..width {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius).min(width - 1));
            rows[y * width + x] = values[y * width + x0..=y * width + x1]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
        }
    }
    let mut out = vec![0u32; values.len()];
    for y in 0..height {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius).min(height - 1));
        for x in 0..width {
            out[y * width + x] = (y0..=y1).map(|yy| rows[yy * width + x]).max().unwrap_or(0);
        }
    }
    out
}

/// Renumbers labels to `1..=n` in order of first appearance (row-major).
fn renumber(labels: &mut [u32]) {
    let mut map = std::collections::HashMap::new();
    for l in labels.iter_mut() {
        if *l != 0 {
            let next = map.len() as u32 + 1;
            *l = *map.entry(*l).or_insert(next);
        }
    }
}

/// Cellpose's flow-error quality control (`dynamics.remove_bad_flow_masks`):
/// recomputes the flows each object's shape implies (`masks_to_flows`, a
/// diffusion from the object's center) and removes every object whose
/// mean squared difference to the predicted flows (divided by
/// `FLOW_SCALE`) is above `threshold`. Removed objects become background
/// (`0`); the surviving labels are left as they are.
fn remove_bad_flows(
    labels: &mut [u32],
    flow_y: &[f32],
    flow_x: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) {
    let (implied_y, implied_x) = masks_to_flows(labels, width, height);
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut error_sum = vec![0f64; max_label + 1];
    let mut count = vec![0usize; max_label + 1];
    for (i, &label) in labels.iter().enumerate() {
        if label == 0 {
            continue;
        }
        let dy = implied_y[i] - flow_y[i] as f64 / FLOW_SCALE as f64;
        let dx = implied_x[i] - flow_x[i] as f64 / FLOW_SCALE as f64;
        error_sum[label as usize] += dy * dy + dx * dx;
        count[label as usize] += 1;
    }
    let bad: Vec<bool> = (0..=max_label)
        .map(|l| count[l] > 0 && error_sum[l] / count[l] as f64 > threshold as f64)
        .collect();
    for label in labels.iter_mut() {
        if bad[*label as usize] {
            *label = 0;
        }
    }
}

/// Unit flow vectors (dY, dX) per pixel implied by the label map - a port
/// of Cellpose's `masks_to_flows_gpu`, kept faithful (including its
/// quirks) so the flow error matches Cellpose's. Background pixels get
/// `0`. Labels must be `1..=n` without gaps.
pub(super) fn masks_to_flows(labels: &[u32], width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;
    let n_labels = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut flow_y = vec![0f64; width * height];
    let mut flow_x = vec![0f64; width * height];
    if n_labels == 0 {
        return (flow_y, flow_x);
    }

    // Bounding boxes and mean positions, both relative to the box like
    // Cellpose's `find_objects` slices (the rounding of the mean depends on it).
    let mut bbox = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; n_labels + 1];
    for y in 0..height {
        for x in 0..width {
            let l = labels[y * width + x] as usize;
            if l == 0 {
                continue;
            }
            let b = &mut bbox[l];
            b[0] = b[0].min(y);
            b[1] = b[1].min(x);
            b[2] = b[2].max(y);
            b[3] = b[3].max(x);
        }
    }
    // Padded by one pixel on every side, so every neighbor index is valid.
    let pw = width + 2;
    let padded = |y: usize, x: usize| (y + 1) * pw + (x + 1);
    let mut centers = vec![0usize; n_labels + 1];
    let mut max_ext = 0usize;
    for l in 1..=n_labels {
        let [y0, x0, y1, x1] = bbox[l];
        if y0 == usize::MAX {
            continue;
        }
        let (mut sy, mut sx, mut n) = (0usize, 0usize, 0usize);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if labels[y * width + x] as usize == l {
                    sy += y - y0;
                    sx += x - x0;
                    n += 1;
                }
            }
        }
        let ym = (sy as f64 / n as f64).round_ties_even() as usize;
        let xm = (sx as f64 / n as f64).round_ties_even() as usize;
        let (mut cy, mut cx) = (y0 + ym, x0 + xm);
        if cy > y1 || cx > x1 || labels[cy * width + cx] as usize != l {
            // The mean lies outside the object: take its closest pixel,
            // the first one in row-major order on a tie.
            let mut best = usize::MAX;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if labels[y * width + x] as usize != l {
                        continue;
                    }
                    let d = (y - y0).abs_diff(ym).pow(2) + (x - x0).abs_diff(xm).pow(2);
                    if d < best {
                        best = d;
                        (cy, cx) = (y, x);
                    }
                }
            }
        }
        centers[l] = padded(cy, cx);
        max_ext = max_ext.max((y1 - y0 + 1) + (x1 - x0 + 1) + 2);
    }

    // Mask pixels with their 9 neighbors (center, up, down, left, right,
    // then the diagonals) and which of those belong to the same object.
    const OFFSETS: [(isize, isize); 9] = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];
    let label_at = |y: isize, x: isize| -> u32 {
        if y < 0 || x < 0 || y >= height as isize || x >= width as isize {
            0
        } else {
            labels[y as usize * width + x as usize]
        }
    };
    let mut pixels: Vec<(usize, usize)> = Vec::new();
    let mut neighbors: Vec<[usize; 9]> = Vec::new();
    let mut same: Vec<[bool; 9]> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let l = labels[y * width + x];
            if l == 0 {
                continue;
            }
            let mut nb = [0usize; 9];
            let mut sm = [false; 9];
            for (k, (dy, dx)) in OFFSETS.iter().enumerate() {
                let (ny, nx) = (y as isize + dy, x as isize + dx);
                nb[k] = ((ny + 1) as usize) * pw + (nx + 1) as usize;
                sm[k] = label_at(ny, nx) == l;
            }
            pixels.push((y, x));
            neighbors.push(nb);
            same.push(sm);
        }
    }

    // Diffusion from the centers: every step adds 1 at each center, then
    // replaces each pixel by the sum of its same-object neighbors / 9.
    let mut heat = vec![0f64; pw * (height + 2)];
    let mut next = vec![0f64; pixels.len()];
    for _ in 0..2 * max_ext {
        for &c in &centers[1..] {
            heat[c] += 1.0;
        }
        next.par_iter_mut().enumerate().for_each(|(i, v)| {
            let nb = &neighbors[i];
            let sm = &same[i];
            *v = (0..9).filter(|&k| sm[k]).map(|k| heat[nb[k]]).sum::<f64>() / 9.0;
        });
        for (i, nb) in neighbors.iter().enumerate() {
            heat[nb[0]] = next[i];
        }
    }

    // Normalized gradient. Like Cellpose, it reads the raw neighbor
    // values, also those of a touching object.
    for (i, &(y, x)) in pixels.iter().enumerate() {
        let nb = &neighbors[i];
        let dy = heat[nb[2]] - heat[nb[1]];
        let dx = heat[nb[4]] - heat[nb[3]];
        let norm = 1e-60 + (dy * dy + dx * dx).sqrt();
        flow_y[y * width + x] = dy / norm;
        flow_x[y * width + x] = dx / norm;
    }
    (flow_y, flow_x)
}

/// Advects each cell pixel along the flow field for `flow_iterations` Euler
/// steps (Cellpose's `steps2D`), sampling the flow at the current integer
/// position and clamping to the image bounds. Returns, for every pixel, the
/// flattened index of its final position (cell pixels converge to the sink
/// at their object's center; non-cell pixels keep their own index).
pub(super) fn follow_flows(
    flow_y: &[f32],
    flow_x: &[f32],
    is_cell: &[bool],
    width: usize,
    height: usize,
    flow_iterations: usize,
) -> Vec<usize> {
    let niter = flow_iterations.max(1);
    let max_x = width as f32 - 1.0;
    let max_y = height as f32 - 1.0;

    let mut final_pos = vec![0usize; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !is_cell[idx] {
                final_pos[idx] = idx;
                continue;
            }

            let (mut py, mut px) = (y as f32, x as f32);
            for _ in 0..niter {
                let sample = py.round() as usize * width + px.round() as usize;
                py = (py + flow_y[sample] / FLOW_SCALE).clamp(0.0, max_y);
                px = (px + flow_x[sample] / FLOW_SCALE).clamp(0.0, max_x);
            }
            final_pos[idx] = py.round() as usize * width + px.round() as usize;
        }
    }
    final_pos
}

/// Groups cell pixels into instances by their convergence sinks. A density
/// map of all final positions is built and its occupied cells are labeled
/// with 8-connected connected components; every cell pixel then inherits the
/// label of the sink it landed in. Returns a per-pixel instance label
/// (`0` = background).
pub(super) fn label_sinks(
    final_positions: &[usize],
    is_cell: &[bool],
    width: usize,
    height: usize,
) -> Vec<u32> {
    // Mark the sink cells (final positions of cell pixels).
    let mut is_sink = vec![false; width * height];
    for (idx, &pos) in final_positions.iter().enumerate() {
        if is_cell[idx] {
            is_sink[pos] = true;
        }
    }

    // 8-connected connected components over the sink cells.
    let mut sink_label = vec![0u32; width * height];
    let mut next_label = 1u32;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..(width * height) {
        if !is_sink[start] || sink_label[start] != 0 {
            continue;
        }
        sink_label[start] = next_label;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (cx, cy) = (p % width, p / width);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = cx as i64 + dx;
                    let ny = cy as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let np = ny as usize * width + nx as usize;
                    if is_sink[np] && sink_label[np] == 0 {
                        sink_label[np] = next_label;
                        stack.push(np);
                    }
                }
            }
        }
        next_label += 1;
    }

    // Propagate the sink label back to every cell pixel.
    let mut labels = vec![0u32; width * height];
    for (idx, &pos) in final_positions.iter().enumerate() {
        if is_cell[idx] {
            labels[idx] = sink_label[pos];
        }
    }
    labels
}

#[cfg(test)]
// Indices written as `row * width + column`, row 0 and column 0 included.
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::*;

    fn resize_nearest_vec(labels: &[u32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u32> {
        let mut out = vec![0u32; dw * dh];
        resize_nearest(labels, (sw, sh), (dw, dh), &mut out);
        out
    }

    // ---- follow_flows ----

    #[test]
    fn follow_flows_leaves_non_cell_pixels_at_their_own_index() {
        let flow_y = vec![1.0; 9];
        let flow_x = vec![1.0; 9];
        let is_cell = vec![false; 9];
        let final_pos = follow_flows(&flow_y, &flow_x, &is_cell, 3, 3, 10);
        assert_eq!(final_pos, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn follow_flows_converges_every_cell_pixel_to_a_single_sink() {
        // 1D row of 5 pixels; the flow field points every pixel toward x=2 at
        // exactly one pixel per Euler step (FLOW_SCALE cancels the /5 in
        // follow_flows), so a handful of iterations is enough to converge
        // from either end.
        let flow_y = vec![0.0; 5];
        let flow_x = vec![5.0, 5.0, 0.0, -5.0, -5.0];
        let is_cell = vec![true; 5];
        let final_pos = follow_flows(&flow_y, &flow_x, &is_cell, 5, 1, 10);
        assert_eq!(final_pos, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn follow_flows_clamps_trajectories_to_the_image_bounds() {
        // Every pixel pushed hard off the right/bottom edge must clamp to the
        // last valid row/column, not wrap or index out of bounds.
        let flow_y = vec![100.0; 4];
        let flow_x = vec![100.0; 4];
        let is_cell = vec![true; 4];
        let final_pos = follow_flows(&flow_y, &flow_x, &is_cell, 2, 2, 10);
        // width=2, height=2 -> bottom-right pixel is index 1*2+1 = 3.
        assert_eq!(final_pos, vec![3, 3, 3, 3]);
    }

    // ---- label_sinks ----

    #[test]
    fn label_sinks_gives_the_same_label_to_pixels_converging_to_adjacent_sinks() {
        // Two adjacent (8-connected) sink cells merge into one connected
        // component, so cell pixels converging to either must share a label.
        let final_positions = vec![0, 1];
        let is_cell = vec![true, true];
        let labels = label_sinks(&final_positions, &is_cell, 2, 1);
        assert_ne!(labels[0], 0);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn label_sinks_gives_different_labels_to_far_apart_sinks() {
        let final_positions = vec![0, 0, 2];
        let is_cell = vec![true, true, true];
        let labels = label_sinks(&final_positions, &is_cell, 3, 1);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[2], 0);
    }

    #[test]
    fn label_sinks_never_labels_a_non_cell_pixel() {
        let final_positions = vec![1, 1, 1];
        let is_cell = vec![false, true, true];
        let labels = label_sinks(&final_positions, &is_cell, 3, 1);
        assert_eq!(
            labels[0], 0,
            "non-cell pixels must stay unlabeled regardless of final_positions"
        );
        assert_ne!(labels[1], 0);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn label_sinks_returns_all_zero_when_no_pixel_is_a_cell() {
        let final_positions = vec![0, 1, 2, 3];
        let is_cell = vec![false, false, false, false];
        let labels = label_sinks(&final_positions, &is_cell, 2, 2);
        assert_eq!(labels, vec![0, 0, 0, 0]);
    }

    // ---- flow_threshold ----

    /// `cellpose.dynamics.masks_to_flows_gpu` (Cellpose 4.2.1) on this label
    /// map: two touching squares, a C shape whose mean lies outside it, a
    /// single pixel next to another object, and a 3x3 square.
    // Values as Cellpose printed them (0.707107 is not meant as 1/√2).
    #[allow(clippy::approx_constant)]
    mod reference {
        pub const LABELS: [u32; 108] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0,
            0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 5, 5, 5, 3, 0, 0, 0, 0, 0, 0, 0, 4, 5, 5, 5, 3, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 5, 5, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        pub const EXPECTED_Y: [f64; 108] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.974547,
            0.996067, 0.977018, 0.993298, 1.0, 0.895948, 0.0, 0.0, 0.0, 0.0, 0.0, 0.224183,
            0.707107, 0.978536, 0.891251, 0.962258, 1.0, 0.521388, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.088608, -0.206078, -0.707107, -0.300207, -0.417556, -1.0, -0.087281, 0.0, 0.0, 0.0,
            0.0, 0.0, -0.478715, -0.869184, -0.995737, -0.967643, -0.986389, -1.0, -0.716851, 0.0,
            0.0, 0.0, 0.0, 0.989622, -0.999995, -0.999999, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.707107, 1.0, 0.707107, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.707107, -1.0, -0.707107, -0.143741,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        pub const EXPECTED_X: [f64; 108] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.224183,
            -0.088608, -0.213157, 0.115583, 0.0, -0.444159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974547,
            0.707107, -0.206078, -0.45351, 0.272139, 0.0, -0.85332, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.996067, 0.978536, -0.707107, -0.953874, 0.908651, 0.0, -0.996184, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.87797, 0.494488, -0.092234, -0.252321, 0.164426, 0.0, -0.697226, 0.0, 0.0, 0.0,
            0.0, 0.143692, -0.003081, -0.00127, -0.000144, -2.4e-05, 0.0, 0.0, 0.0, 0.0, 0.707107,
            0.0, -0.707107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707107, 0.0, -0.707107, 0.989615, 1.0, -1.0, -1.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
    }

    #[test]
    fn masks_to_flows_matches_cellpose() {
        let (fy, fx) = masks_to_flows(&reference::LABELS, 12, 9);
        for i in 0..reference::LABELS.len() {
            assert!(
                (fy[i] - reference::EXPECTED_Y[i]).abs() < 1e-4
                    && (fx[i] - reference::EXPECTED_X[i]).abs() < 1e-4,
                "pixel ({}, {}): got ({}, {}), Cellpose has ({}, {})",
                i % 12,
                i / 12,
                fy[i],
                fx[i],
                reference::EXPECTED_Y[i],
                reference::EXPECTED_X[i]
            );
        }
    }

    #[test]
    fn remove_bad_flow_masks_keeps_objects_whose_flows_match_and_drops_the_rest() {
        let (w, h) = (12, 9);
        let (fy, fx) = masks_to_flows(&reference::LABELS, w, h);
        // Predicted flows (x FLOW_SCALE) that match every object exactly ...
        let mut pred_y: Vec<f32> = fy.iter().map(|v| (v * 5.0) as f32).collect();
        let mut pred_x: Vec<f32> = fx.iter().map(|v| (v * 5.0) as f32).collect();
        // ... except object 5, whose flows point the wrong way.
        for i in 0..w * h {
            if reference::LABELS[i] == 5 {
                pred_y[i] = -pred_y[i];
                pred_x[i] = -pred_x[i];
            }
        }
        let mut labels = reference::LABELS.to_vec();
        remove_bad_flows(&mut labels, &pred_y, &pred_x, w, h, 0.4);
        for i in 0..w * h {
            let expected = if reference::LABELS[i] == 5 {
                0
            } else {
                reference::LABELS[i]
            };
            assert_eq!(labels[i], expected, "pixel {i}");
        }
    }

    #[test]
    fn masks_to_flows_of_an_empty_label_map_is_zero() {
        let (fy, fx) = masks_to_flows(&[0; 6], 3, 2);
        assert!(fy.iter().chain(&fx).all(|&v| v == 0.0));
    }

    #[test]
    fn masks_from_seeds_drops_pixels_that_reach_no_crowded_spot() {
        // 12 pixels end at (5,5): a seed. 3 pixels end alone far away: no seed.
        let (w, h) = (20, 1);
        let mut final_positions: Vec<(usize, usize)> = (0..w).map(|x| (0, x)).collect();
        let is_cell = vec![true; w];
        for p in final_positions.iter_mut().take(12) {
            *p = (0, 5);
        }
        let labels = masks_from_seeds(&final_positions, &is_cell, w, h);
        assert!(labels[..12].iter().all(|&l| l == 1));
        assert!(labels[12..].iter().all(|&l| l == 0), "{labels:?}");
    }

    #[test]
    fn masks_from_seeds_without_any_crowded_spot_is_empty() {
        let final_positions: Vec<(usize, usize)> = (0..4).map(|x| (0, x)).collect();
        let labels = masks_from_seeds(&final_positions, &[true; 4], 4, 1);
        assert_eq!(labels, vec![0; 4]);
    }

    #[test]
    fn resize_labels_nearest_matches_opencv_inter_nearest() {
        // 2x2 -> 4x3: column d takes floor(d * 2 / 4), row d floor(d * 2 / 3).
        let labels = vec![1, 2, 3, 4];
        let out = resize_nearest_vec(&labels, 2, 2, 4, 3);
        assert_eq!(out, vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]);
        // Shrinking takes every other pixel.
        let out = resize_nearest_vec(&[1, 2, 3, 4], 4, 1, 2, 1);
        assert_eq!(out, vec![1, 3]);
    }

    // ---- remove_small_and_renumber ----

    #[test]
    fn small_objects_are_removed_and_the_rest_renumbered() {
        // label 1: 3 px (kept), label 2: 1 px (removed), label 3: 3 px (kept).
        let mut labels = vec![1, 1, 1, 2, 3, 3, 3, 0];
        assert_eq!(remove_small_and_renumber(&mut labels, 2), 2);
        assert_eq!(labels, vec![1, 1, 1, 0, 2, 2, 2, 0]);
    }

    #[test]
    fn min_size_zero_keeps_a_single_pixel_object() {
        let mut labels = vec![1, 0, 0];
        assert_eq!(remove_small_and_renumber(&mut labels, 0), 1);
        assert_eq!(labels, vec![1, 0, 0]);
        let mut empty = vec![0u32; 4];
        assert_eq!(remove_small_and_renumber(&mut empty, 0), 0);
        assert_eq!(empty, vec![0; 4]);
    }
}
