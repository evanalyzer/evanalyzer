//! # wand
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The wand: a brush that keeps to edges. Along the stroke, every brush
//! circle takes only the pixels that look like the one under the cursor and
//! are connected to it - so a stroke inside a nucleus stops at its border.
//!
//! "Look like" is decided on what the viewer shows (the visible channels,
//! each scaled to its 1-99 % range of the view, smoothed 3 × 3): each
//! pixel's color distance to the cursor's, split in two by Otsu's
//! threshold. Where the circle has no edge (the two halves differ by less
//! than [`MIN_CONTRAST`]) the whole circle is taken, as the brush does.

use crate::editor::view_channel::{SampleSlice, Samples, ViewChannel};
use crate::editor::viewport_cache::ReadContext;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use evanalyzer_app::images::ImageContainer;

/// Below this color distance (channels scaled to 0-1) between the two
/// halves of a circle, it has no edge to keep to.
const MIN_CONTRAST: f32 = 0.1;

/// Bins of the distance histogram Otsu's threshold is found on.
const BINS: usize = 64;

/// One value per pixel of the view: a gray channel, or one color of an RGB
/// one.
#[derive(Clone, Copy)]
enum Source<'a> {
    Gray(SampleSlice<'a>),
    Rgb(&'a [f32], usize),
}

impl Source<'_> {
    fn len(&self) -> usize {
        match self {
            Source::Gray(s) => s.len(),
            Source::Rgb(d, _) => d.len() / 3,
        }
    }

    fn get(&self, i: usize) -> f32 {
        match self {
            Source::Gray(s) => s.get(i),
            Source::Rgb(d, c) => d[i * 3 + c],
        }
    }
}

/// The visible channels of the view (all of them when none is marked
/// visible), an RGB channel as its three colors.
fn sources<'a>(channels: &'a [ViewChannel], visible: &[i32]) -> Vec<Source<'a>> {
    let any_visible = channels.iter().any(|c| visible.contains(&c.c_stack));
    let mut out = Vec::new();
    for channel in channels {
        if any_visible && !visible.contains(&channel.c_stack) {
            continue;
        }
        if let Some(slice) = channel.slice() {
            out.push(Source::Gray(slice));
        } else if let Samples::Image(image) = &channel.samples
            && let ImageContainer::F32Rgb(rgb) = &**image
        {
            let data = rgb.as_slice();
            out.extend((0..3).map(|c| Source::Rgb(data, c)));
        }
    }
    out
}

/// The 1st and 99th percentile of `source` (from at most ~64k samples).
fn robust_range(source: &Source) -> (f32, f32) {
    let len = source.len();
    let step = (len / 65_536).max(1);
    let mut values: Vec<f32> = (0..len)
        .step_by(step)
        .map(|i| source.get(i))
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return (0.0, 1.0);
    }
    values.sort_unstable_by(f32::total_cmp);
    let at = |q: f32| values[((values.len() - 1) as f32 * q).round() as usize];
    let (lo, hi) = (at(0.01), at(0.99));
    if hi - lo > 1e-6 { (lo, hi) } else { (lo, lo + 1.0) }
}

/// The pixels of a window of the view, as the wand compares them: per
/// source, scaled to its range and smoothed 3 × 3.
pub(crate) struct Features {
    /// Window position and size, in pixels of the view.
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
    planes: Vec<Vec<f32>>,
}

impl Features {
    /// `[x0, y0, x1, y1]` (inclusive, clamped to the view) of the view
    /// `view_w` × `view_h` made of `channels`. `None` without data.
    pub fn of_window(
        channels: &[ViewChannel],
        visible: &[i32],
        view_w: usize,
        view_h: usize,
        window: [usize; 4],
    ) -> Option<Self> {
        let sources = sources(channels, visible);
        if sources.is_empty() || view_w == 0 || view_h == 0 {
            return None;
        }
        if sources.iter().any(|s| s.len() != view_w * view_h) {
            return None;
        }
        let [x0, y0, x1, y1] = window;
        let (x1, y1) = (x1.min(view_w - 1), y1.min(view_h - 1));
        if x0 > x1 || y0 > y1 {
            return None;
        }
        let (width, height) = (x1 - x0 + 1, y1 - y0 + 1);
        let planes = sources
            .iter()
            .map(|source| {
                let (lo, hi) = robust_range(source);
                let scale = 1.0 / (hi - lo);
                let mut plane = Vec::with_capacity(width * height);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        // 3 × 3 mean, the view's edge repeated.
                        let mut sum = 0.0;
                        for ny in [y.saturating_sub(1), y, (y + 1).min(view_h - 1)] {
                            for nx in [x.saturating_sub(1), x, (x + 1).min(view_w - 1)] {
                                sum += source.get(ny * view_w + nx);
                            }
                        }
                        plane.push((sum / 9.0 - lo) * scale);
                    }
                }
                plane
            })
            .collect();
        Some(Self {
            x0,
            y0,
            width,
            height,
            planes,
        })
    }

    /// Features from planes already scaled - for tests.
    #[cfg(test)]
    fn from_planes(width: usize, height: usize, planes: Vec<Vec<f32>>) -> Self {
        Self {
            x0: 0,
            y0: 0,
            width,
            height,
            planes,
        }
    }

    fn distance(&self, a: usize, b: usize) -> f32 {
        self.planes
            .iter()
            .map(|p| (p[a] - p[b]) * (p[a] - p[b]))
            .sum::<f32>()
            .sqrt()
    }
}

/// Otsu's threshold of `values` (all in `0..=max`): the cut that best
/// separates them in two, and the two halves' means.
fn otsu(values: &[f32], max: f32) -> (f32, f32, f32) {
    let mut hist = [0usize; BINS];
    let bin = |v: f32| (((v / max) * BINS as f32) as usize).min(BINS - 1);
    for &v in values {
        hist[bin(v)] += 1;
    }
    let center = |b: usize| (b as f32 + 0.5) * max / BINS as f32;
    let total = values.len() as f32;
    let sum_all: f32 = (0..BINS).map(|b| hist[b] as f32 * center(b)).sum();
    let (mut best, mut best_cut) = (-1.0f32, 0);
    let (mut n0, mut sum0) = (0.0f32, 0.0f32);
    for cut in 0..BINS - 1 {
        n0 += hist[cut] as f32;
        sum0 += hist[cut] as f32 * center(cut);
        let n1 = total - n0;
        if n0 == 0.0 || n1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / n0, (sum_all - sum0) / n1);
        let between = n0 * n1 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            best_cut = cut;
        }
    }
    let threshold = (best_cut + 1) as f32 * max / BINS as f32;
    let (mut s0, mut c0, mut s1, mut c1) = (0.0, 0.0, 0.0, 0.0);
    for &v in values {
        if v < threshold {
            s0 += v;
            c0 += 1.0;
        } else {
            s1 += v;
            c1 += 1.0;
        }
    }
    let mean = |s: f32, c: f32| if c > 0.0 { s / c } else { 0.0 };
    (threshold, mean(s0, c0), mean(s1, c1))
}

/// The wand's pixels for a stroke through `points` (window pixels) with
/// circles of `radius` window pixels: a mask of the window, row by row.
pub(crate) fn wand_mask(features: &Features, points: &[(f32, f32)], radius: f32) -> BitVec<u64, Lsb0> {
    let (w, h) = (features.width, features.height);
    let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * h);
    if points.is_empty() || w == 0 || h == 0 {
        return mask;
    }
    let r = radius.max(1.0);
    let r2 = r * r;

    // Circles a third of the radius apart, also along fast strokes.
    let spacing = (r / 3.0).max(1.0);
    let mut centers = vec![points[0]];
    for pair in points.windows(2) {
        let ((ax, ay), (bx, by)) = (pair[0], pair[1]);
        let steps = ((bx - ax).hypot(by - ay) / spacing).ceil().max(1.0) as usize;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            centers.push((ax + (bx - ax) * t, ay + (by - ay) * t));
        }
    }

    let mut cells: Vec<usize> = Vec::new();
    let mut distances: Vec<f32> = Vec::new();
    let mut taken = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for (cx, cy) in centers {
        let (sx, sy) = (cx.floor(), cy.floor());
        if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 {
            continue;
        }
        let seed = sy as usize * w + sx as usize;
        let x_lo = (cx - r).floor().max(0.0) as usize;
        let x_hi = ((cx + r).ceil() as usize).min(w - 1);
        let y_lo = (cy - r).floor().max(0.0) as usize;
        let y_hi = ((cy + r).ceil() as usize).min(h - 1);
        let in_circle = |x: usize, y: usize| {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            dx * dx + dy * dy <= r2
        };

        cells.clear();
        distances.clear();
        for y in y_lo..=y_hi {
            for x in x_lo..=x_hi {
                if in_circle(x, y) {
                    let i = y * w + x;
                    cells.push(i);
                    distances.push(features.distance(i, seed));
                }
            }
        }
        let max = distances.iter().copied().fold(0.0f32, f32::max);
        let threshold = if max < MIN_CONTRAST {
            f32::INFINITY
        } else {
            let (t, near, far) = otsu(&distances, max);
            if far - near < MIN_CONTRAST { f32::INFINITY } else { t }
        };

        if threshold.is_infinite() {
            // No edge: the whole circle.
            for &i in &cells {
                mask.set(i, true);
            }
            continue;
        }
        // The pixels like the cursor's, connected to it (4-neighbours).
        let mut visited: Vec<usize> = Vec::new();
        stack.clear();
        stack.push(seed);
        taken[seed] = true;
        while let Some(i) = stack.pop() {
            visited.push(i);
            mask.set(i, true);
            let (x, y) = (i % w, i / w);
            let mut visit = |nx: usize, ny: usize| {
                let n = ny * w + nx;
                if !taken[n] && in_circle(nx, ny) && features.distance(n, seed) < threshold {
                    taken[n] = true;
                    stack.push(n);
                }
            };
            if x > x_lo {
                visit(x - 1, y);
            }
            if x < x_hi {
                visit(x + 1, y);
            }
            if y > y_lo {
                visit(x, y - 1);
            }
            if y < y_hi {
                visit(x, y + 1);
            }
        }
        for i in visited {
            taken[i] = false;
        }
    }
    mask
}

/// Where the view's pixels lie in the full image - [`ReadContext`] as it
/// was when the view was read.
pub(crate) struct ViewGrid {
    draw_x: f32,
    draw_y: f32,
    /// Screen pixels per view pixel.
    step_x: f32,
    step_y: f32,
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
}

impl ViewGrid {
    pub fn new(ctx: &ReadContext) -> Option<Self> {
        if ctx.image_w == 0 || ctx.image_h == 0 || ctx.zoom <= 0.0 {
            return None;
        }
        Some(Self {
            draw_x: ctx.draw_x,
            draw_y: ctx.draw_y,
            step_x: ctx.zoomed_w / ctx.image_w as f32,
            step_y: ctx.zoomed_h / ctx.image_h as f32,
            zoom: ctx.zoom,
            offset_x: ctx.offset_x,
            offset_y: ctx.offset_y,
        })
        .filter(|g| g.step_x > 0.0 && g.step_y > 0.0)
    }

    /// The view position of full-image point (`x`, `y`).
    pub fn to_view(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x * self.zoom + self.offset_x - self.draw_x) / self.step_x,
            (y * self.zoom + self.offset_y - self.draw_y) / self.step_y,
        )
    }

    /// The full-image position of view point (`x`, `y`).
    pub fn to_full(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x * self.step_x + self.draw_x - self.offset_x) / self.zoom,
            (y * self.step_y + self.draw_y - self.offset_y) / self.zoom,
        )
    }

    /// Full-image pixels per view pixel.
    pub fn full_per_view(&self) -> f32 {
        self.step_x / self.zoom
    }
}

/// The wand's `mask` of `features`' window as a mask of the full image
/// (`image_w` × `image_h`), relative to its bounding box `[x_min, y_min,
/// x_max, y_max]` (inclusive), as objects keep it. `None` when empty.
pub(crate) fn to_full_image(
    mask: &BitVec<u64, Lsb0>,
    features: &Features,
    grid: &ViewGrid,
    image_w: usize,
    image_h: usize,
) -> Option<(BitVec<u64, Lsb0>, [u32; 4])> {
    let w = features.width;
    let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (usize::MAX, usize::MAX, 0, 0);
    for i in mask.iter_ones() {
        let (x, y) = (i % w, i / w);
        lo_x = lo_x.min(x);
        lo_y = lo_y.min(y);
        hi_x = hi_x.max(x);
        hi_y = hi_y.max(y);
    }
    if lo_x == usize::MAX || image_w == 0 || image_h == 0 {
        return None;
    }
    let (fx0, fy0) = grid.to_full((features.x0 + lo_x) as f32, (features.y0 + lo_y) as f32);
    let (fx1, fy1) = grid.to_full((features.x0 + hi_x + 1) as f32, (features.y0 + hi_y + 1) as f32);
    let x_min = fx0.floor().max(0.0) as u32;
    let y_min = fy0.floor().max(0.0) as u32;
    let x_max = (fx1.ceil() as i64 - 1).min(image_w as i64 - 1);
    let y_max = (fy1.ceil() as i64 - 1).min(image_h as i64 - 1);
    if x_max < x_min as i64 || y_max < y_min as i64 {
        return None;
    }
    let (x_max, y_max) = (x_max as u32, y_max as u32);
    let width = (x_max - x_min + 1) as usize;
    let height = (y_max - y_min + 1) as usize;
    let mut out = BitVec::<u64, Lsb0>::repeat(false, width * height);
    let mut any = false;
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            // The view pixel the full pixel's centre falls in.
            let (vx, vy) = grid.to_view(x as f32 + 0.5, y as f32 + 0.5);
            let (vx, vy) = (vx.floor() - features.x0 as f32, vy.floor() - features.y0 as f32);
            if vx < 0.0 || vy < 0.0 || vx >= w as f32 || vy >= features.height as f32 {
                continue;
            }
            if mask[vy as usize * w + vx as usize] {
                out.set(((y - y_min) as usize) * width + (x - x_min) as usize, true);
                any = true;
            }
        }
    }
    any.then_some((out, [x_min, y_min, x_max, y_max]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `size` × `size` gray image: 0 left of column `edge`, 1 from it on.
    fn two_halves(size: usize, edge: usize) -> Features {
        let plane = (0..size * size)
            .map(|i| if i % size >= edge { 1.0 } else { 0.0 })
            .collect();
        Features::from_planes(size, size, vec![plane])
    }

    fn columns(mask: &BitVec<u64, Lsb0>, width: usize) -> (usize, usize) {
        let xs: Vec<usize> = mask.iter_ones().map(|i| i % width).collect();
        (*xs.iter().min().unwrap(), *xs.iter().max().unwrap())
    }

    #[test]
    fn the_wand_stops_at_an_edge() {
        let features = two_halves(40, 20);
        // A circle of radius 10 at x 15 reaches over the edge at x 20.
        let mask = wand_mask(&features, &[(15.5, 20.5)], 10.0);
        let (lo, hi) = columns(&mask, 40);
        assert_eq!(hi, 19, "nothing right of the edge");
        assert!(lo <= 6, "the circle's left part is taken: from {lo}");
    }

    #[test]
    fn on_the_bright_side_it_takes_the_bright_side() {
        let features = two_halves(40, 20);
        let mask = wand_mask(&features, &[(24.5, 20.5)], 10.0);
        let (lo, _) = columns(&mask, 40);
        assert_eq!(lo, 20);
    }

    #[test]
    fn without_an_edge_it_takes_the_whole_circle_like_the_brush() {
        // Faint noise only, far below MIN_CONTRAST.
        let plane = (0..30 * 30).map(|i| 0.5 + 0.01 * ((i * 7) % 3) as f32).collect();
        let features = Features::from_planes(30, 30, vec![plane]);
        let mask = wand_mask(&features, &[(15.0, 15.0)], 5.0);
        let expected = (0..30 * 30)
            .filter(|i| {
                let (dx, dy) = ((i % 30) as f32 + 0.5 - 15.0, (i / 30) as f32 + 0.5 - 15.0);
                dx * dx + dy * dy <= 25.0
            })
            .count();
        assert_eq!(mask.count_ones(), expected);
    }

    #[test]
    fn it_takes_only_pixels_connected_to_the_cursor() {
        // A bright spot (x 4-6) apart from the bright stripe the cursor is
        // on (x 12-17), dark between.
        let size = 20;
        let plane = (0..size * size)
            .map(|i| {
                let x = i % size;
                if (12..18).contains(&x) || (4..7).contains(&x) { 1.0 } else { 0.0 }
            })
            .collect();
        let features = Features::from_planes(size, size, vec![plane]);
        let mask = wand_mask(&features, &[(14.5, 10.5)], 10.0);
        let (lo, hi) = columns(&mask, size);
        assert_eq!((lo, hi), (12, 17));
    }

    #[test]
    fn a_stroke_follows_the_edge_along_its_length() {
        let features = two_halves(60, 30);
        let mask = wand_mask(&features, &[(25.5, 5.5), (25.5, 54.5)], 8.0);
        assert!(mask.iter_ones().all(|i| i % 60 < 30));
        // Rows along the stroke are covered without gaps.
        for y in 5..55 {
            assert!(mask[y * 60 + 25], "row {y}");
        }
    }

    #[test]
    fn colors_count_not_only_brightness() {
        // Red left, green right: the same brightness, different colors.
        let size = 30;
        let red = (0..size * size).map(|i| if i % size < 15 { 1.0 } else { 0.0 }).collect();
        let green = (0..size * size).map(|i| if i % size < 15 { 0.0 } else { 1.0 }).collect();
        let features = Features::from_planes(size, size, vec![red, green]);
        let mask = wand_mask(&features, &[(10.5, 15.5)], 8.0);
        assert!(mask.iter_ones().all(|i| i % size < 15));
    }

    fn grid(zoom: f32, view_px_per_full: f32) -> ViewGrid {
        // A view read at `view_px_per_full` view pixels per full pixel,
        // shown at `zoom` screen pixels per full pixel from (0, 0).
        ViewGrid {
            draw_x: 0.0,
            draw_y: 0.0,
            step_x: zoom / view_px_per_full,
            step_y: zoom / view_px_per_full,
            zoom,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    #[test]
    fn a_view_pixel_of_a_coarser_level_covers_its_full_pixels() {
        // Every view pixel covers 4 × 4 full pixels.
        let features = Features::from_planes(10, 10, vec![vec![0.0; 100]]);
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, 100);
        mask.set(2 * 10 + 3, true);
        let (full, bbox) = to_full_image(&mask, &features, &grid(0.25, 0.25), 1000, 1000).unwrap();
        assert_eq!(bbox, [12, 8, 15, 11]);
        assert_eq!(full.count_ones(), 16);
    }

    #[test]
    fn view_and_full_positions_map_back_and_forth() {
        let g = ViewGrid {
            draw_x: -37.0,
            draw_y: 12.5,
            step_x: 2.0,
            step_y: 2.0,
            zoom: 0.5,
            offset_x: -40.0,
            offset_y: 10.0,
        };
        let (vx, vy) = g.to_view(123.0, 456.0);
        let (fx, fy) = g.to_full(vx, vy);
        assert!((fx - 123.0).abs() < 1e-3 && (fy - 456.0).abs() < 1e-3);
        assert!((g.full_per_view() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn the_full_mask_is_cut_to_the_image() {
        let features = Features::from_planes(4, 4, vec![vec![0.0; 16]]);
        let mask = BitVec::<u64, Lsb0>::repeat(true, 16);
        let (_, bbox) = to_full_image(&mask, &features, &grid(1.0, 1.0), 3, 2).unwrap();
        assert_eq!(bbox, [0, 0, 2, 1]);
    }

    #[test]
    fn an_empty_mask_is_none() {
        let features = Features::from_planes(4, 4, vec![vec![0.0; 16]]);
        let mask = BitVec::<u64, Lsb0>::repeat(false, 16);
        assert!(to_full_image(&mask, &features, &grid(1.0, 1.0), 10, 10).is_none());
    }
}
