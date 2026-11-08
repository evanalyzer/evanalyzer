//! # skeletonize_objects
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Reduces each object to its 1-pixel-wide centre line (skeleton) - the
//! first step of filament analysis (neurites, vessels, fibres). Each object
//! gets one skeleton object; optionally its branch points and end points
//! become 1-pixel objects of their own. Spurs shorter than a minimum length
//! (the side branches a ragged outline leaves) are pruned.
//!
//! 1. **Thinning** (Guo & Hall 1989, two sub-iterations): border pixels are
//!    peeled off in parallel steps while the shape's connectivity and its
//!    end points are kept. Each step only looks at the current border
//!    pixels, so a large object costs its area once, not its area per step.
//! 2. **Thin to one pixel:** the 4-connected corners thinning leaves on
//!    diagonal lines (a staircase: exactly two 4-neighbours at a right angle)
//!    are removed, in row order, where the neighbours stay connected -
//!    otherwise every corner would count as a branch point.
//! 3. **Pruning:** from each end point along the line to its branch point;
//!    a spur shorter than `min_branch_length` is removed (never all arms of a
//!    branch point, the longest stays), then steps 1-2 again (no stub stays
//!    where it was attached). Repeated until nothing changes.
//! 4. **Graph:** end points have one neighbour; branch points are the
//!    8-connected groups of pixels with three or more, each one point (its
//!    pixel nearest the group's centre).
//!
//! The working grid is one bit per pixel of the object's box (plus a bit to
//! queue it), so a tissue-sized object stays within the whole-image RAM
//! budget. Parents (skeleton → its object, points → their skeleton) come
//! from Relate Objects, as everywhere.

use std::collections::{HashMap, HashSet};

use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, ObjectClass, ObjectId, SizeUnits,
};
use kornia_image::ImageSize;
use macros::CommandsMeta;

use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    object::{Object, ObjectInit},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Reduces each object to its 1-pixel-wide centre line (skeleton), for
/// filament analysis (neurites, vessels, fibres).
///
/// [Threshold] -> [Extract Objects] -> [Skeletonize Objects] -> [Relate Objects] -> [Measure Object Skeleton]
///
/// Each object of the input class gets one skeleton object; branch points
/// and end points can become 1-pixel objects of their own classes. Spurs
/// shorter than the minimum branch length are pruned. The input objects
/// stay as they are; link skeletons and points to them with Relate Objects.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", display_name = "Skeletonize Objects")]
pub struct SkeletonizeObjects {
    /// Objects carrying this class are skeletonized (e.g. the thresholded
    /// neurites).
    #[cmdsmeta(summary = true)]
    pub input_class: ObjectClass,

    /// The class the skeletons get (one per input object).
    #[cmdsmeta(summary = true)]
    pub skeleton_class: ObjectClass,

    /// The class the branch points get (one 1-pixel object each). Unset:
    /// none are created.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub branch_point_class: ObjectClass,

    /// The class the end points get (one 1-pixel object each). Unset: none
    /// are created.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub end_point_class: ObjectClass,

    /// Side branches shorter than this (from their end point to the branch
    /// point) are removed - the spurs a ragged outline leaves. 0: none are
    /// removed.
    #[cmdsmeta(default = 5.0, min = 0.0, max = 100000.0, step = 1.0, summary = true)]
    pub min_branch_length: f32,

    /// Unit `min_branch_length` is expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub length_unit: SizeUnits,
}

impl ImageAlgorithm for SkeletonizeObjects {
    /// Per input object (in the cache's order): thinning, one-pixel
    /// cleanup, pruning, then the skeleton and its points as new objects
    /// (intensities measured on their own pixels, like any object).
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.input_class == ObjectClass::Unset || self.skeleton_class == ObjectClass::Unset {
            return Ok(());
        }
        let full = ctx.full_image_size();
        let min_length = self.min_length_px(ctx);
        let scope = ctx.scope;

        let inputs: Vec<ObjectId> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.input_class, scope))
            .map(|(id, _)| id.clone())
            .collect();

        let mut new_objects = Vec::new();
        for id in inputs {
            let Some(object) = cache.object_cache.get(&id) else {
                continue;
            };
            let skeleton = skeletonize(object, min_length);
            if skeleton.pixels.is_empty() {
                continue;
            }
            let mut outputs = vec![(self.skeleton_class, skeleton.pixels)];
            if self.branch_point_class != ObjectClass::Unset {
                outputs.extend(
                    skeleton
                        .branch_points
                        .into_iter()
                        .map(|p| (self.branch_point_class, vec![p])),
                );
            }
            if self.end_point_class != ObjectClass::Unset {
                outputs.extend(
                    skeleton
                        .end_points
                        .into_iter()
                        .map(|p| (self.end_point_class, vec![p])),
                );
            }
            for (class, pixels) in outputs {
                let mut new_object = object_from_pixels(&pixels, object, full);
                new_object.intensities = new_object.measure_intensities(cache);
                new_object.add_object_class(class, scope)?;
                new_objects.push(new_object);
            }
        }
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Skeletonize Objects"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&GUO_HALL_1989]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

pub(crate) static GUO_HALL_1989: CitationMetadata = CitationMetadata {
    cite_key: "guo1989parallel",
    title: "Parallel Thinning with Two-Subiteration Algorithms",
    authors: &["Zicheng Guo", "Richard W. Hall"],
    year: 1989,
    container: Some("Communications of the ACM"),
    doi: Some("10.1145/62065.62074"),
    url: Some("https://doi.org/10.1145/62065.62074"),
    pages: Some("359-373"),
};

impl SkeletonizeObjects {
    /// `min_branch_length` in pixels; without a pixel size nanometres are
    /// taken as pixels.
    fn min_length_px(&self, ctx: &PipelineContext) -> f32 {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        match self.length_unit {
            SizeUnits::NanoMeter if pixel_size_nm > 0.0 => self.min_branch_length / pixel_size_nm,
            _ => self.min_branch_length,
        }
        .max(0.0)
    }
}

/// An object's skeleton in image coordinates: its pixels (row order), its
/// branch points and end points.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Skeleton {
    pub pixels: Vec<(u32, u32)>,
    pub branch_points: Vec<(u32, u32)>,
    pub end_points: Vec<(u32, u32)>,
}

/// The neighbours' offsets in the order P2 … P9 of the thinning papers:
/// N, NE, E, SE, S, SW, W, NW.
pub(crate) const RING: [(isize, isize); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

/// One bit per pixel of an object's box with a 1-pixel border of 0 around
/// (so every pixel of the object has all 8 neighbours in the grid).
pub(crate) struct Grid {
    pub(crate) width: usize,
    pub(crate) bits: BitVec<u64, Lsb0>,
}

impl Grid {
    pub(crate) fn of(object: &Object) -> Self {
        let [x0, y0, x1, y1] = object.bbox;
        let box_width = (x1 - x0 + 1) as usize;
        let width = box_width + 2;
        let height = (y1 - y0 + 1) as usize + 2;
        let mut bits = BitVec::repeat(false, width * height);
        for i in object.mask_data.iter_ones() {
            let (x, y) = (i % box_width, i / box_width);
            bits.set((y + 1) * width + x + 1, true);
        }
        Grid { width, bits }
    }

    pub(crate) fn get(&self, p: usize) -> bool {
        self.bits[p]
    }

    fn set(&mut self, p: usize, on: bool) {
        self.bits.set(p, on);
    }

    /// The index of `p`'s neighbour `k` (in [`RING`] order). Every set
    /// pixel is off the border, so its neighbours are in the grid.
    pub(crate) fn neighbour(&self, p: usize, k: usize) -> usize {
        let (dx, dy) = RING[k];
        (p as isize + dy * self.width as isize + dx) as usize
    }

    /// Which of `p`'s 8 neighbours are set, P2 … P9.
    fn ring(&self, p: usize) -> [bool; 8] {
        std::array::from_fn(|k| self.get(self.neighbour(p, k)))
    }

    pub(crate) fn neighbour_count(&self, p: usize) -> usize {
        self.ring(p).iter().filter(|&&on| on).count()
    }

    /// The set pixels, in row order.
    pub(crate) fn ones(&self) -> Vec<usize> {
        self.bits.iter_ones().collect()
    }
}

/// Guo & Hall's C(P): the number of 8-connected groups of set neighbours
/// (1: removing P keeps them connected).
fn connectivity(n: &[bool; 8]) -> u8 {
    let [p2, p3, p4, p5, p6, p7, p8, p9] = *n;
    (!p2 && (p3 || p4)) as u8
        + (!p4 && (p5 || p6)) as u8
        + (!p6 && (p7 || p8)) as u8
        + (!p8 && (p9 || p2)) as u8
}

/// Guo & Hall 1989 (their algorithm A1): may P be removed in sub-iteration
/// `first` / second? C(P) = 1 (connectivity kept), 2 ≤ N(P) ≤ 3 (not an end
/// point, a border pixel), and the sub-iteration's side condition.
fn removable(n: &[bool; 8], first: bool) -> bool {
    let [p2, p3, p4, p5, p6, p7, p8, p9] = *n;
    let n1 = (p9 || p2) as u8 + (p3 || p4) as u8 + (p5 || p6) as u8 + (p7 || p8) as u8;
    let n2 = (p2 || p3) as u8 + (p4 || p5) as u8 + (p6 || p7) as u8 + (p8 || p9) as u8;
    let n_min = n1.min(n2);
    let side = if first {
        (p6 || p7 || !p9) && p8
    } else {
        (p2 || p3 || !p5) && p4
    };
    connectivity(n) == 1 && (2..=3).contains(&n_min) && !side
}

/// Step 1: thinning in place. Only the border pixels are looked at; after
/// a sub-iteration the next border is the surviving one plus the set
/// neighbours of the removed pixels.
pub(crate) fn thin(grid: &mut Grid) {
    let mut queued: BitVec<u64, Lsb0> = BitVec::repeat(false, grid.bits.len());
    // Filtered while iterating: collecting every pixel first would take
    // 8 bytes per pixel of a tissue-sized object.
    let mut border: Vec<usize> = grid
        .bits
        .iter_ones()
        .filter(|&p| grid.ring(p).iter().any(|&on| !on))
        .collect();
    loop {
        let mut changed = false;
        for first in [true, false] {
            // Decided on the state before the sub-iteration (parallel).
            let removed: Vec<usize> = border
                .iter()
                .copied()
                .filter(|&p| grid.get(p) && removable(&grid.ring(p), first))
                .collect();
            if removed.is_empty() {
                continue;
            }
            changed = true;
            for &p in &removed {
                grid.set(p, false);
            }
            let mut next = Vec::with_capacity(border.len());
            let mut push = |p: usize, next: &mut Vec<usize>| {
                if grid.get(p) && !queued[p] {
                    queued.set(p, true);
                    next.push(p);
                }
            };
            for &p in &border {
                push(p, &mut next);
            }
            for &p in &removed {
                for k in 0..8 {
                    push(grid.neighbour(p, k), &mut next);
                }
            }
            for &p in &next {
                queued.set(p, false);
            }
            border = next;
        }
        if !changed {
            break;
        }
    }
}

/// Step 2: removes the corner pixels of 4-connected staircases - a pixel
/// with exactly two set 4-neighbours at a right angle, whose removal keeps
/// its neighbours connected - in row order, until none is left. A pixel
/// with three 4-neighbours (the centre of a T) stays, so do end points.
pub(crate) fn one_pixel_wide(grid: &mut Grid) {
    loop {
        let mut changed = false;
        for p in grid.ones() {
            let n = grid.ring(p);
            let [north, _, east, _, south, _, west, _] = n;
            let four = [north, east, south, west].iter().filter(|&&on| on).count();
            let corner = four == 2 && north != south;
            if corner && connectivity(&n) == 1 {
                grid.set(p, false);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// A side branch from an end point to its branch point: its pixels (the
/// branch point not included), its length and the branch point reached.
struct Spur {
    pixels: Vec<usize>,
    length: f32,
    junction: usize,
}

/// Follows the line from end point `start` to the first pixel with three
/// or more neighbours. `None`: the line ends in another end point (a lone
/// line, never pruned) or runs into a loop.
fn trace_spur(grid: &Grid, start: usize) -> Option<Spur> {
    let mut pixels = vec![start];
    let mut seen: HashSet<usize> = HashSet::from([start]);
    let mut current = start;
    let mut length = 0.0f32;
    loop {
        let step = (0..8)
            .map(|k| (k, grid.neighbour(current, k)))
            .find(|&(_, q)| grid.get(q) && !seen.contains(&q))?;
        let (k, next) = step;
        length += if k % 2 == 0 {
            1.0
        } else {
            std::f32::consts::SQRT_2
        };
        match grid.neighbour_count(next) {
            1 => return None,
            2 => {
                seen.insert(next);
                pixels.push(next);
                current = next;
            }
            _ => {
                return Some(Spur {
                    pixels,
                    length,
                    junction: next,
                });
            }
        }
    }
}

/// Step 3: removes spurs shorter than `min_length`, never all arms of a
/// branch point (its longest spur stays then); repeated until nothing
/// changes. Returns whether anything was removed.
fn prune(grid: &mut Grid, min_length: f32) -> bool {
    if min_length <= 0.0 {
        return false;
    }
    let mut any = false;
    loop {
        let spurs: Vec<Spur> = grid
            .ones()
            .into_iter()
            .filter(|&p| grid.neighbour_count(p) == 1)
            .filter_map(|p| trace_spur(grid, p))
            .filter(|spur| spur.length < min_length)
            .collect();
        if spurs.is_empty() {
            return any;
        }
        // The spurs per branch-point group.
        let groups = junction_groups(grid);
        let mut by_group: HashMap<usize, Vec<&Spur>> = HashMap::new();
        for spur in &spurs {
            by_group
                .entry(groups[&spur.junction])
                .or_default()
                .push(spur);
        }
        let spur_pixels: HashSet<usize> = spurs
            .iter()
            .flat_map(|s| s.pixels.iter().copied())
            .collect();
        let mut removed = false;
        let mut group_ids: Vec<usize> = by_group.keys().copied().collect();
        group_ids.sort_unstable();
        for group in group_ids {
            let mut members = by_group.remove(&group).unwrap();
            // An arm of the group that is no short spur: a pixel next to the
            // group, outside it, not on one of the short spurs.
            let keeps_an_arm = groups
                .iter()
                .filter(|&(_, &g)| g == group)
                .flat_map(|(&p, _)| (0..8).map(move |k| (p, k)))
                .map(|(p, k)| grid.neighbour(p, k))
                .any(|q| grid.get(q) && !groups.contains_key(&q) && !spur_pixels.contains(&q));
            if !keeps_an_arm {
                // All arms are short spurs: the longest stays (the first in
                // row order of the longest).
                members.sort_by(|a, b| {
                    b.length
                        .total_cmp(&a.length)
                        .then(a.pixels[0].cmp(&b.pixels[0]))
                });
                members.remove(0);
            }
            for spur in members {
                for &p in &spur.pixels {
                    grid.set(p, false);
                }
                removed = true;
            }
        }
        if !removed {
            return any;
        }
        any = true;
        // Where a spur met its branch point a 1-pixel stub can stay: thin
        // again.
        thin(grid);
        one_pixel_wide(grid);
    }
}

/// The branch-point pixels (three or more neighbours), each with the id of
/// its 8-connected group (the group's first pixel in row order).
pub(crate) fn junction_groups(grid: &Grid) -> HashMap<usize, usize> {
    let junctions: Vec<usize> = grid
        .ones()
        .into_iter()
        .filter(|&p| grid.neighbour_count(p) >= 3)
        .collect();
    let is_junction: HashSet<usize> = junctions.iter().copied().collect();
    let mut groups: HashMap<usize, usize> = HashMap::new();
    for &start in &junctions {
        if groups.contains_key(&start) {
            continue;
        }
        groups.insert(start, start);
        let mut stack = vec![start];
        while let Some(p) = stack.pop() {
            for k in 0..8 {
                let q = grid.neighbour(p, k);
                if is_junction.contains(&q) && !groups.contains_key(&q) {
                    groups.insert(q, start);
                    stack.push(q);
                }
            }
        }
    }
    groups
}

/// Steps 1-4 for one object, in image coordinates.
pub(crate) fn skeletonize(object: &Object, min_length: f32) -> Skeleton {
    let mut grid = Grid::of(object);
    thin(&mut grid);
    one_pixel_wide(&mut grid);
    prune(&mut grid, min_length);

    let [x0, y0, _, _] = object.bbox;
    let width = grid.width;
    let to_image = |p: usize| (x0 + (p % width) as u32 - 1, y0 + (p / width) as u32 - 1);

    let pixels = grid.ones();
    let end_points = pixels
        .iter()
        .copied()
        .filter(|&p| grid.neighbour_count(p) == 1)
        .map(to_image)
        .collect();

    // One point per branch-point group: its pixel nearest the group's
    // centre (the first in row order on a tie).
    let groups = junction_groups(&grid);
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&p, &group) in &groups {
        members.entry(group).or_default().push(p);
    }
    let mut group_ids: Vec<usize> = members.keys().copied().collect();
    group_ids.sort_unstable();
    let branch_points = group_ids
        .into_iter()
        .map(|group| {
            let mut pixels = members.remove(&group).unwrap();
            pixels.sort_unstable();
            to_image(nearest_to_centre(&pixels, width))
        })
        .collect();

    Skeleton {
        pixels: pixels.into_iter().map(to_image).collect(),
        branch_points,
        end_points,
    }
}

/// The pixel of `pixels` (row order) nearest their centroid; the first on
/// a tie.
fn nearest_to_centre(pixels: &[usize], width: usize) -> usize {
    let n = pixels.len() as f64;
    let (sx, sy) = pixels.iter().fold((0.0, 0.0), |(sx, sy), &p| {
        (sx + (p % width) as f64, sy + (p / width) as f64)
    });
    let (cx, cy) = (sx / n, sy / n);
    let distance = |p: usize| {
        let (dx, dy) = ((p % width) as f64 - cx, (p / width) as f64 - cy);
        dx * dx + dy * dy
    };
    let mut best = pixels[0];
    for &p in &pixels[1..] {
        if distance(p) < distance(best) {
            best = p;
        }
    }
    best
}

/// A new object of `pixels` (image coordinates, not empty) on `source`'s
/// plane, with its segmentation class.
fn object_from_pixels(pixels: &[(u32, u32)], source: &Object, full: ImageSize) -> Object {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for &(x, y) in pixels {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    let width = (x1 - x0 + 1) as usize;
    let mut mask_data = BitVec::repeat(false, width * (y1 - y0 + 1) as usize);
    let (mut sum_x, mut sum_y, mut sum_x2, mut sum_y2, mut sum_xy) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut touches_edge = false;
    for &(x, y) in pixels {
        mask_data.set((y - y0) as usize * width + (x - x0) as usize, true);
        let (xu, yu) = (x as u64, y as u64);
        sum_x += xu;
        sum_y += yu;
        sum_x2 += xu * xu;
        sum_y2 += yu * yu;
        sum_xy += xu * yu;
        touches_edge |=
            x == 0 || y == 0 || x as usize + 1 >= full.width || y as usize + 1 >= full.height;
    }
    Object::new(ObjectInit {
        id: ObjectId::next(),
        segmentation_class: source.segmentation_class,
        plane: source.plane.clone(),
        dimensionality: source.dimensionality,
        bbox: [x0, y0, x1, y1],
        area: pixels.len(),
        mask_data,
        touches_edge,
        sum_x,
        sum_y,
        sum_x2,
        sum_y2,
        sum_xy,
        ..Default::default()
    })
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An object from rows of '#' (set) and '.' at (`x0`, `y0`).
    fn object(rows: &[&str], x0: u32, y0: u32) -> Object {
        let pixels: Vec<(u32, u32)> = rows
            .iter()
            .enumerate()
            .flat_map(|(y, row)| {
                row.chars()
                    .enumerate()
                    .filter(|&(_, c)| c == '#')
                    .map(move |(x, _)| (x0 + x as u32, y0 + y as u32))
            })
            .collect();
        let full = ImageSize {
            width: 1000,
            height: 1000,
        };
        let template = Object::new(ObjectInit::default());
        object_from_pixels(&pixels, &template, full)
    }

    /// A filled rectangle `w` x `h` at (10, 10).
    fn rect(w: usize, h: usize) -> Object {
        let row = "#".repeat(w);
        let rows: Vec<&str> = (0..h).map(|_| row.as_str()).collect();
        object(&rows, 10, 10)
    }

    /// The number of 8-neighbours of `p` in `pixels`.
    fn neighbours(pixels: &[(u32, u32)], p: (u32, u32)) -> usize {
        pixels
            .iter()
            .filter(|&&q| q != p && q.0.abs_diff(p.0) <= 1 && q.1.abs_diff(p.1) <= 1)
            .count()
    }

    /// The 8-connected groups of `pixels`.
    fn components(pixels: &[(u32, u32)]) -> usize {
        let mut left: HashSet<(u32, u32)> = pixels.iter().copied().collect();
        let mut count = 0;
        while let Some(&start) = left.iter().next() {
            count += 1;
            left.remove(&start);
            let mut stack = vec![start];
            while let Some(p) = stack.pop() {
                let near: Vec<(u32, u32)> = left
                    .iter()
                    .copied()
                    .filter(|q| q.0.abs_diff(p.0) <= 1 && q.1.abs_diff(p.1) <= 1)
                    .collect();
                for q in near {
                    left.remove(&q);
                    stack.push(q);
                }
            }
        }
        count
    }

    #[test]
    fn a_thick_horizontal_bar_becomes_one_line() {
        let skeleton = skeletonize(&rect(30, 5), 0.0);
        // One row of pixels, roughly the bar's length, in the middle.
        let ys: HashSet<u32> = skeleton.pixels.iter().map(|p| p.1).collect();
        assert_eq!(ys, HashSet::from([12]));
        assert!(skeleton.pixels.len() >= 24, "{:?}", skeleton.pixels);
        assert_eq!(skeleton.end_points.len(), 2);
        assert!(skeleton.branch_points.is_empty());
    }

    #[test]
    fn a_two_by_two_square_keeps_one_pixel() {
        let skeleton = skeletonize(&rect(2, 2), 0.0);
        assert_eq!(skeleton.pixels.len(), 1);
    }

    #[test]
    fn a_thick_diagonal_line_stays_connected_and_one_pixel_wide() {
        // A diagonal band 2 px thick.
        let rows: Vec<String> = (0..20)
            .map(|y| {
                (0..22)
                    .map(|x| if x == y || x == y + 1 { '#' } else { '.' })
                    .collect()
            })
            .collect();
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        let skeleton = skeletonize(&object(&rows, 5, 5), 0.0);
        assert_eq!(components(&skeleton.pixels), 1);
        assert!(skeleton.pixels.len() >= 15, "{:?}", skeleton.pixels);
        for &p in &skeleton.pixels {
            assert!(
                neighbours(&skeleton.pixels, p) <= 2,
                "{p:?} in {:?}",
                skeleton.pixels
            );
        }
        assert_eq!(skeleton.end_points.len(), 2);
        assert!(skeleton.branch_points.is_empty());
    }

    #[test]
    fn a_thin_line_is_kept_as_it_is() {
        let skeleton = skeletonize(&object(&["##########"], 3, 4), 0.0);
        assert_eq!(skeleton.pixels, (3..13).map(|x| (x, 4)).collect::<Vec<_>>());
        assert_eq!(skeleton.end_points, vec![(3, 4), (12, 4)]);
    }

    /// A plus sign with 3-pixel-thick arms of 12 px around a centre.
    fn plus() -> Object {
        let rows: Vec<String> = (0..27)
            .map(|y| {
                (0..27)
                    .map(|x| {
                        let arm = |v: i32| (12..15).contains(&v);
                        if arm(x) || arm(y) { '#' } else { '.' }
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        object(&rows, 20, 20)
    }

    #[test]
    fn a_plus_gives_one_branch_point_and_four_end_points() {
        let skeleton = skeletonize(&plus(), 0.0);
        assert_eq!(components(&skeleton.pixels), 1);
        assert_eq!(skeleton.branch_points.len(), 1, "{:?}", skeleton.pixels);
        assert_eq!(skeleton.end_points.len(), 4);
        let (bx, by) = skeleton.branch_points[0];
        assert!(bx.abs_diff(33) <= 1 && by.abs_diff(33) <= 1, "{bx},{by}");
    }

    /// A horizontal line of 30 px with a side branch of `branch` px going
    /// down from x = 15.
    fn line_with_branch(branch: usize) -> Object {
        let mut rows = vec!["#".repeat(30)];
        for _ in 0..branch {
            rows.push(format!("{}#{}", ".".repeat(15), ".".repeat(14)));
        }
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        object(&rows, 50, 50)
    }

    #[test]
    fn a_short_spur_is_pruned() {
        let skeleton = skeletonize(&line_with_branch(3), 5.0);
        assert!(skeleton.branch_points.is_empty(), "{:?}", skeleton.pixels);
        assert_eq!(skeleton.end_points.len(), 2);
        let ys: HashSet<u32> = skeleton.pixels.iter().map(|p| p.1).collect();
        assert_eq!(ys, HashSet::from([50]));
    }

    #[test]
    fn a_long_branch_stays() {
        let skeleton = skeletonize(&line_with_branch(10), 5.0);
        assert_eq!(skeleton.branch_points.len(), 1);
        assert_eq!(skeleton.end_points.len(), 3);
        assert_eq!(skeleton.pixels.len(), 40);
    }

    #[test]
    fn without_pruning_a_short_spur_stays() {
        let skeleton = skeletonize(&line_with_branch(3), 0.0);
        assert_eq!(skeleton.branch_points.len(), 1);
        assert_eq!(skeleton.end_points.len(), 3);
    }

    #[test]
    fn pruning_never_removes_all_arms() {
        // A small plus: every arm shorter than the minimum length.
        let skeleton = skeletonize(
            &object(&["..#..", "..#..", "#####", "..#..", "..#.."], 0, 0),
            50.0,
        );
        assert!(!skeleton.pixels.is_empty());
        assert_eq!(components(&skeleton.pixels), 1);
        assert!(skeleton.pixels.len() >= 3, "{:?}", skeleton.pixels);
    }

    #[test]
    fn a_ring_keeps_its_hole_and_has_no_end_points() {
        let mut rows = Vec::new();
        for y in 0..20 {
            let row: String = (0..20)
                .map(|x| {
                    let edge = !(4..16).contains(&x) || !(4..16).contains(&y);
                    if edge { '#' } else { '.' }
                })
                .collect();
            rows.push(row);
        }
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        let skeleton = skeletonize(&object(&rows, 30, 30), 5.0);
        assert_eq!(components(&skeleton.pixels), 1);
        assert!(skeleton.end_points.is_empty(), "{:?}", skeleton.end_points);
        // A closed loop around the hole: every pixel has two neighbours.
        for &p in &skeleton.pixels {
            assert_eq!(neighbours(&skeleton.pixels, p), 2, "{p:?}");
        }
    }

    #[test]
    fn coordinates_are_in_the_image() {
        let skeleton = skeletonize(&object(&["#####"], 100, 200), 0.0);
        assert_eq!(skeleton.end_points, vec![(100, 200), (104, 200)]);
    }

    #[test]
    fn the_command_creates_the_skeleton_and_its_points() {
        use crate::F32Gray;
        let mut ctx = PipelineContext::new_test::<F32Gray>(ImageSize {
            width: 100,
            height: 100,
        })
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        let mut input = plus();
        input
            .add_object_class(ObjectClass::Global(1), ctx.scope)
            .unwrap();
        let input_id = input.id.clone();
        cache.object_cache.insert(input_id.clone(), input);
        SkeletonizeObjects {
            input_class: ObjectClass::Global(1),
            skeleton_class: ObjectClass::Global(2),
            branch_point_class: ObjectClass::Global(3),
            end_point_class: ObjectClass::Global(4),
            min_branch_length: 5.0,
            length_unit: SizeUnits::Pixels,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        let count = |class: u32| {
            cache
                .object_cache
                .iter()
                .filter(|(_, o)| o.is_of(&ObjectClass::Global(class), ctx.scope))
                .count()
        };
        assert_eq!((count(1), count(2), count(3), count(4)), (1, 1, 1, 4));
        // The input stays as it was.
        assert_eq!(
            cache.object_cache.get(&input_id).unwrap().area,
            27 * 3 * 2 - 9
        );
    }

    #[test]
    fn unset_point_classes_create_no_points() {
        use crate::F32Gray;
        let mut ctx = PipelineContext::new_test::<F32Gray>(ImageSize {
            width: 100,
            height: 100,
        })
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        let mut input = plus();
        input
            .add_object_class(ObjectClass::Global(1), ctx.scope)
            .unwrap();
        cache.object_cache.insert(input.id.clone(), input);
        SkeletonizeObjects {
            input_class: ObjectClass::Global(1),
            skeleton_class: ObjectClass::Global(2),
            branch_point_class: ObjectClass::Unset,
            end_point_class: ObjectClass::Unset,
            min_branch_length: 0.0,
            length_unit: SizeUnits::Pixels,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        assert_eq!(cache.object_cache.len(), 2);
    }
}

/// Random shapes (unions of thick lines and disks): the skeleton lies
/// inside the shape, with as many parts and holes.
#[cfg(test)]
mod topology {
    use super::*;
    use std::collections::HashSet;

    fn groups(pixels: &HashSet<(i64, i64)>, eight: bool) -> usize {
        let mut left = pixels.clone();
        let mut n = 0;
        while let Some(&s) = left.iter().next() {
            n += 1;
            left.remove(&s);
            let mut st = vec![s];
            while let Some((x, y)) = st.pop() {
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        if (dx, dy) == (0, 0) || (!eight && dx != 0 && dy != 0) {
                            continue;
                        }
                        if left.remove(&(x + dx, y + dy)) {
                            st.push((x + dx, y + dy));
                        }
                    }
                }
            }
        }
        n
    }
    /// The holes: the 4-connected background groups in the padded box, less
    /// the outside.
    fn holes(pixels: &HashSet<(i64, i64)>) -> usize {
        let (x0, x1) = (
            pixels.iter().map(|p| p.0).min().unwrap() - 1,
            pixels.iter().map(|p| p.0).max().unwrap() + 1,
        );
        let (y0, y1) = (
            pixels.iter().map(|p| p.1).min().unwrap() - 1,
            pixels.iter().map(|p| p.1).max().unwrap() + 1,
        );
        let mut bg = HashSet::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                if !pixels.contains(&(x, y)) {
                    bg.insert((x, y));
                }
            }
        }
        groups(&bg, false) - 1
    }

    #[test]
    fn random_shapes_keep_their_topology() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for case in 0..60 {
            let mut set: HashSet<(i64, i64)> = HashSet::new();
            let shapes = 1 + (rnd() * 5.0) as usize;
            for _ in 0..shapes {
                let (ax, ay, bx, by) = (rnd() * 60.0, rnd() * 60.0, rnd() * 60.0, rnd() * 60.0);
                let r = 0.7 + rnd() * 4.0;
                for y in 0..64i64 {
                    for x in 0..64i64 {
                        let (px, py) = (x as f64, y as f64);
                        let (dx, dy) = (bx - ax, by - ay);
                        let t = (((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9))
                            .clamp(0.0, 1.0);
                        let (qx, qy) = (ax + t * dx - px, ay + t * dy - py);
                        if qx * qx + qy * qy <= r * r {
                            set.insert((x, y));
                        }
                    }
                }
            }
            if set.is_empty() {
                continue;
            }
            let pixels: Vec<(u32, u32)> = {
                let mut v: Vec<_> = set
                    .iter()
                    .map(|&(x, y)| (x as u32 + 5, y as u32 + 5))
                    .collect();
                v.sort();
                v
            };
            let obj = object_from_pixels(
                &pixels,
                &Object::new(ObjectInit::default()),
                ImageSize {
                    width: 1000,
                    height: 1000,
                },
            );
            let sk = skeletonize(&obj, 0.0);
            let skset: HashSet<(i64, i64)> = sk
                .pixels
                .iter()
                .map(|&(x, y)| (x as i64 - 5, y as i64 - 5))
                .collect();
            assert!(
                skset.iter().all(|p| set.contains(p)),
                "case {case}: outside"
            );
            assert_eq!(
                groups(&skset, true),
                groups(&set, true),
                "case {case}: components"
            );
            assert_eq!(holes(&skset), holes(&set), "case {case}: holes");
            // Pruning keeps the parts too.
            let pruned = skeletonize(&obj, 5.0);
            let pruned: HashSet<(i64, i64)> = pruned
                .pixels
                .iter()
                .map(|&(x, y)| (x as i64 - 5, y as i64 - 5))
                .collect();
            assert_eq!(
                groups(&pruned, true),
                groups(&set, true),
                "case {case}: pruned components"
            );
        }
    }
}
