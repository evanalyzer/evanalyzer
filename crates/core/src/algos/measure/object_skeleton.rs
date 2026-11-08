//! # Measure Object Skeleton
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! The values of a filament's skeleton (from Skeletonize Objects): its
//! length, its end and branch points, its segments (the lines between
//! them), the longest path through it and - from the object it was made
//! of, its parent - its mean diameter.
//!
//! The skeleton is read as a graph: end points (one neighbour), branch
//! points (8-connected groups of pixels with three or more) and the lines
//! between them. A step to a 4-neighbour is 1 px long, to a diagonal one
//! √2 - a diagonal step that a 4-neighbour of both pixels bridges (the
//! corner of a branch point) isn't counted twice. The skeleton is thinned
//! once more first, so an object that isn't one (yet) is measured as its
//! skeleton (without pruning).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::algos::object::skeletonize_objects::{
    GUO_HALL_1989, Grid, RING, junction_groups, one_pixel_wide, thin,
};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::Object;
use crate::pipeline::pipeline_measure::{ObjectValue, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MeasurementInfo, MeasurementUnit, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// For each skeleton (Skeletonize Objects): its total length, the longest
/// path through it, its end points, branch points and segments (the lines
/// between them), their mean length and straightness (the straight
/// distance between a segment's ends / its length; loops don't count) and,
/// with "Diameter from", the mean diameter of the object it was made of
/// along it. Lengths in pixels.
///
/// [Skeletonize Objects] -> [Relate Objects] -> [Measure Object Skeleton]
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Object Skeleton")]
pub struct MeasureObjectSkeleton {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("skeleton_01/...").
    #[cmdsmeta(measurement_id = "skeleton", display_name = "ID", summary = true)]
    pub id: String,

    /// The skeletons measured (the skeleton class of Skeletonize Objects).
    #[cmdsmeta(summary = true)]
    pub objects: ObjectClass,

    /// The class of the objects the skeletons were made of (e.g. the
    /// neurites), linked to them as their parents by Relate Objects: the
    /// diameter is twice the distance from the skeleton to their outline.
    /// Unset: no diameter.
    #[cmdsmeta(default = ObjectClass::Unset)]
    pub diameter_from: ObjectClass,
}

/// The values: (id, name, unit), the diameter last.
const VALUES: [(&str, &str, MeasurementUnit); 8] = [
    ("length", "Total length", MeasurementUnit::Pixels),
    ("longest_path", "Longest path", MeasurementUnit::Pixels),
    ("end_points", "End points", MeasurementUnit::Count),
    ("branch_points", "Branch points", MeasurementUnit::Count),
    ("segments", "Segments", MeasurementUnit::Count),
    (
        "mean_segment_length",
        "Mean segment length",
        MeasurementUnit::Pixels,
    ),
    (
        "mean_straightness",
        "Mean straightness",
        MeasurementUnit::Fraction,
    ),
    ("mean_diameter", "Mean diameter", MeasurementUnit::Pixels),
];

impl ImageAlgorithm for MeasureObjectSkeleton {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.objects == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Measure object skeleton: choose the skeletons".into(),
            ));
        }
        let outputs = self.value_count();
        let objects: Vec<_> = cache
            .object_cache
            .iter()
            .filter(|(_, object)| object.is_of(&self.objects, ctx.scope))
            .collect();
        // The parents' rows once each (several skeletons can share one).
        let diameter_from = self.diameter_from;
        fn parent_of(object: &Object, class: ObjectClass) -> Option<&ObjectId> {
            (class != ObjectClass::Unset)
                .then(|| object.parent(&class))
                .flatten()
        }
        let mut parents: Vec<&ObjectId> = objects
            .iter()
            .filter_map(|(_, o)| parent_of(o, diameter_from))
            .collect();
        parents.sort_unstable();
        parents.dedup();
        let outlines: HashMap<&ObjectId, RowRuns> = parents
            .par_iter()
            .filter_map(|&id| cache.object_cache.get(id).map(|p| (id, RowRuns::of(p))))
            .collect();
        // Each skeleton on its own: in parallel.
        let rows: Vec<[f64; VALUES.len()]> = objects
            .par_iter()
            .map(|(_, object)| {
                measure(
                    object,
                    parent_of(object, diameter_from).and_then(|id| outlines.get(id)),
                )
            })
            .collect();
        let per_output: Vec<Vec<ObjectValue>> = (0..outputs)
            .map(|i| {
                objects
                    .iter()
                    .zip(&rows)
                    .map(|((id, _), row)| ObjectValue {
                        object: (*id).clone(),
                        value: row[i],
                        related: None,
                        position: None,
                    })
                    .collect()
            })
            .collect();
        for ((id, ..), values) in VALUES.iter().zip(per_output) {
            cache.add_object_measurement(self.measurement_output(id), values);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Object Skeleton"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR, &GUO_HALL_1989]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        VALUES[..self.value_count()]
            .iter()
            .map(|(id, name, unit)| {
                let info = outputs.object(id, name, *unit);
                if *id == "mean_straightness" {
                    info.range(0.0, 1.0)
                } else {
                    info
                }
            })
            .collect()
    }
}

impl MeasureObjectSkeleton {
    /// How many of [`VALUES`] are measured: the diameter only with
    /// "Diameter from".
    fn value_count(&self) -> usize {
        if self.diameter_from == ObjectClass::Unset {
            VALUES.len() - 1
        } else {
            VALUES.len()
        }
    }
}

/// The values of one skeleton, in [`VALUES`] order (NaN where there is none:
/// no segment with two ends, no parent).
fn measure(skeleton: &Object, parent: Option<&RowRuns>) -> [f64; VALUES.len()] {
    let mut grid = Grid::of(skeleton);
    thin(&mut grid);
    one_pixel_wide(&mut grid);
    let graph = SkeletonGraph::of(&grid);

    let segments = graph.segments.len();
    let mean_length = if segments > 0 {
        graph.segments.iter().map(|s| s.length).sum::<f64>() / segments as f64
    } else {
        f64::NAN
    };
    let open: Vec<f64> = graph
        .segments
        .iter()
        .filter_map(|s| s.straightness)
        .collect();
    let straightness = if open.is_empty() {
        f64::NAN
    } else {
        open.iter().sum::<f64>() / open.len() as f64
    };
    let diameter = parent.map_or(f64::NAN, |outline| {
        mean_diameter(&graph.pixels, &grid, skeleton, outline)
    });
    [
        graph.total_length,
        longest_path(&grid, &graph.pixels),
        graph.end_points as f64,
        graph.branch_points as f64,
        segments as f64,
        mean_length,
        straightness,
        diameter,
    ]
}

/// A line of the skeleton between two nodes (end or branch points), or a
/// loop without one.
struct Segment {
    length: f64,
    /// Straight distance between its ends / its length; `None` for a loop.
    straightness: Option<f64>,
}

/// A skeleton read as a graph.
struct SkeletonGraph {
    pixels: Vec<usize>,
    total_length: f64,
    end_points: usize,
    branch_points: usize,
    segments: Vec<Segment>,
}

/// The length of the step from `p` to its neighbour `k` ([`RING`] order).
fn step_length(k: usize) -> f64 {
    if k % 2 == 0 {
        1.0
    } else {
        std::f64::consts::SQRT_2
    }
}

/// Whether the diagonal step from `p` to its neighbour `k` is bridged by a
/// set 4-neighbour of both (then it isn't a step of its own).
fn bridged(grid: &Grid, p: usize, k: usize) -> bool {
    // The two 4-neighbours next to a diagonal in RING order.
    k % 2 == 1 && (grid.get(grid.neighbour(p, k - 1)) || grid.get(grid.neighbour(p, (k + 1) % 8)))
}

impl SkeletonGraph {
    fn of(grid: &Grid) -> Self {
        let pixels = grid.ones();
        let width = grid.width;

        // Every pair of neighbours once (from the one first in row order).
        let total_length = pixels
            .iter()
            .flat_map(|&p| (0..8).map(move |k| (p, k)))
            .filter(|&(p, k)| {
                let q = grid.neighbour(p, k);
                q > p && grid.get(q) && !bridged(grid, p, k)
            })
            .map(|(_, k)| step_length(k))
            .sum();

        let mut groups = junction_groups(grid);
        // A line pixel between two pixels of the same branch point belongs
        // to it (else it would be a tiny loop on the branch point).
        let absorbed: Vec<(usize, usize)> = pixels
            .iter()
            .copied()
            .filter(|p| !groups.contains_key(p) && grid.neighbour_count(*p) == 2)
            .filter_map(|p| {
                let near: Vec<usize> = (0..8)
                    .map(|k| grid.neighbour(p, k))
                    .filter(|&q| grid.get(q))
                    .collect();
                match (groups.get(&near[0]), groups.get(&near[1])) {
                    (Some(a), Some(b)) if a == b => Some((p, *a)),
                    _ => None,
                }
            })
            .collect();
        groups.extend(absorbed);

        let is_end = |p: usize| grid.neighbour_count(p) == 1;
        let is_node = |p: usize| is_end(p) || groups.contains_key(&p);
        let end_points = pixels.iter().filter(|&&p| is_end(p)).count();
        let branch_points = groups.values().collect::<HashSet<_>>().len();
        let position = |p: usize| ((p % width) as f64, (p / width) as f64);
        let distance = |a: usize, b: usize| {
            let ((ax, ay), (bx, by)) = (position(a), position(b));
            ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
        };
        // Each branch point's pixel nearest its group's centre: segments
        // are measured to it, so they add up to the total length.
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&p, &group) in &groups {
            members.entry(group).or_default().push(p);
        }
        let centres: HashMap<usize, usize> = members
            .into_iter()
            .map(|(group, mut pixels)| {
                pixels.sort_unstable();
                let n = pixels.len() as f64;
                let (sx, sy) = pixels.iter().fold((0.0, 0.0), |(sx, sy), &p| {
                    let (x, y) = position(p);
                    (sx + x, sy + y)
                });
                let (cx, cy) = (sx / n, sy / n);
                let off = |p: usize| {
                    let (x, y) = position(p);
                    (x - cx).powi(2) + (y - cy).powi(2)
                };
                let mut best = pixels[0];
                for &p in &pixels[1..] {
                    if off(p) < off(best) {
                        best = p;
                    }
                }
                (group, best)
            })
            .collect();
        // Where a segment ends: an end point, or its branch point's centre.
        let anchor = |p: usize| groups.get(&p).map_or(p, |g| centres[g]);
        let to_anchor = |p: usize| distance(p, anchor(p));

        // From each node along each line leaving it, to the next node.
        let mut segments = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        for &start in pixels.iter().filter(|&&p| is_node(p)) {
            for k in 0..8 {
                let first = grid.neighbour(start, k);
                if !grid.get(first) || bridged(grid, start, k) {
                    continue;
                }
                if is_node(first) {
                    // Two nodes next to each other: a segment of one step
                    // (once, and not inside a branch point).
                    let same = matches!(
                        (groups.get(&start), groups.get(&first)),
                        (Some(a), Some(b)) if a == b
                    );
                    if first > start && !same {
                        let length = to_anchor(start) + step_length(k) + to_anchor(first);
                        let straight = distance(anchor(start), anchor(first));
                        segments.push(Segment {
                            length,
                            straightness: Some((straight / length).min(1.0)),
                        });
                    }
                    continue;
                }
                if visited.contains(&first) {
                    continue;
                }
                visited.insert(first);
                let (mut previous, mut current) = (start, first);
                let mut length = to_anchor(start) + step_length(k);
                loop {
                    let next = (0..8)
                        .map(|k| (k, grid.neighbour(current, k)))
                        .find(|&(_, q)| q != previous && grid.get(q));
                    let Some((k, next)) = next else {
                        break; // Can't happen: a line pixel has two neighbours.
                    };
                    length += step_length(k);
                    if is_node(next) {
                        length += to_anchor(next);
                        let straight = distance(anchor(start), anchor(next));
                        segments.push(Segment {
                            length,
                            straightness: Some((straight / length).min(1.0)),
                        });
                        break;
                    }
                    visited.insert(next);
                    (previous, current) = (current, next);
                }
            }
        }

        // What is left are loops without a node: one segment each.
        let mut left: Vec<usize> = pixels
            .iter()
            .copied()
            .filter(|p| !is_node(*p) && !visited.contains(p))
            .collect();
        left.sort_unstable();
        for start in left {
            if visited.contains(&start) {
                continue;
            }
            let mut length = 0.0;
            let mut stack = vec![start];
            visited.insert(start);
            let mut members = vec![start];
            while let Some(p) = stack.pop() {
                for k in 0..8 {
                    let q = grid.neighbour(p, k);
                    if grid.get(q) && !visited.contains(&q) {
                        visited.insert(q);
                        members.push(q);
                        stack.push(q);
                    }
                }
            }
            for &p in &members {
                for k in 0..8 {
                    let q = grid.neighbour(p, k);
                    if q > p && grid.get(q) && !bridged(grid, p, k) {
                        length += step_length(k);
                    }
                }
            }
            segments.push(Segment {
                length,
                straightness: None,
            });
        }

        SkeletonGraph {
            pixels,
            total_length,
            end_points,
            branch_points,
            segments,
        }
    }
}

/// A distance in a [`BinaryHeap`] (smallest first).
#[derive(PartialEq)]
struct Nearest(f64, usize);

impl Eq for Nearest {}

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The distances along the skeleton from `start` to every pixel it reaches
/// (Dijkstra; steps of 1 and √2).
fn distances_from(grid: &Grid, start: usize) -> HashMap<usize, f64> {
    let mut best: HashMap<usize, f64> = HashMap::from([(start, 0.0)]);
    let mut heap = BinaryHeap::from([Nearest(0.0, start)]);
    while let Some(Nearest(d, p)) = heap.pop() {
        if d > best[&p] {
            continue;
        }
        for k in 0..8 {
            let q = grid.neighbour(p, k);
            if !grid.get(q) {
                continue;
            }
            let nd = d + step_length(k);
            if best.get(&q).is_none_or(|&old| nd < old) {
                best.insert(q, nd);
                heap.push(Nearest(nd, q));
            }
        }
    }
    best
}

/// The longest of the shortest paths between two skeleton pixels, of all
/// its parts: from a pixel (an end point if it has one) to the farthest
/// one, and from there to the farthest again - exact for a tree.
fn longest_path(grid: &Grid, pixels: &[usize]) -> f64 {
    let mut done: HashSet<usize> = HashSet::new();
    let mut longest = 0.0f64;
    for &p in pixels {
        if done.contains(&p) {
            continue;
        }
        let part = distances_from(grid, p);
        done.extend(part.keys().copied());
        // An end point of this part, else its first pixel.
        let mut members: Vec<usize> = part.keys().copied().collect();
        members.sort_unstable();
        let start = members
            .iter()
            .copied()
            .find(|&q| grid.neighbour_count(q) == 1)
            .unwrap_or(members[0]);
        let farthest = |from: &HashMap<usize, f64>| {
            let mut best = (start, 0.0f64);
            for &q in &members {
                let d = from[&q];
                if d > best.1 {
                    best = (q, d);
                }
            }
            best
        };
        let (far, _) = farthest(&distances_from(grid, start));
        let (_, length) = farthest(&distances_from(grid, far));
        longest = longest.max(length);
    }
    longest
}

/// The mean diameter of the parent (its rows `runs`) along the skeleton:
/// per skeleton pixel 2 × its distance to the nearest pixel not in the
/// parent - 1 (a line 3 pixels wide: 3). Skeleton pixels outside the parent
/// count as 0 wide.
fn mean_diameter(pixels: &[usize], grid: &Grid, skeleton: &Object, runs: &RowRuns) -> f64 {
    if pixels.is_empty() {
        return f64::NAN;
    }
    let [x0, y0, _, _] = skeleton.bbox;
    let sum: f64 = pixels
        .iter()
        .map(|&p| {
            let x = x0 as i64 + (p % grid.width) as i64 - 1;
            let y = y0 as i64 + (p / grid.width) as i64 - 1;
            (2.0 * runs.distance_outside(x, y) - 1.0).max(0.0)
        })
        .sum();
    sum / pixels.len() as f64
}

/// An object's set pixels as runs per row (image coordinates, inclusive).
struct RowRuns {
    y0: i64,
    rows: Vec<Vec<(i64, i64)>>,
}

impl RowRuns {
    fn of(object: &Object) -> Self {
        let [x0, y0, x1, y1] = object.bbox;
        let width = (x1 - x0 + 1) as usize;
        let mut rows = vec![Vec::new(); (y1 - y0 + 1) as usize];
        for (y, row) in rows.iter_mut().enumerate() {
            let mut start: Option<usize> = None;
            for x in 0..=width {
                let set = x < width && object.mask_data[y * width + x];
                match (set, start) {
                    (true, None) => start = Some(x),
                    (false, Some(s)) => {
                        row.push((x0 as i64 + s as i64, x0 as i64 + x as i64 - 1));
                        start = None;
                    }
                    _ => {}
                }
            }
        }
        RowRuns {
            y0: y0 as i64,
            rows,
        }
    }

    /// The horizontal distance from (x, y) to the nearest pixel of row y
    /// not in the object (0: (x, y) isn't in it).
    fn across(&self, x: i64, y: i64) -> i64 {
        let Some(row) = usize::try_from(y - self.y0)
            .ok()
            .and_then(|i| self.rows.get(i))
        else {
            return 0;
        };
        let i = row.partition_point(|&(_, end)| end < x);
        match row.get(i) {
            Some(&(start, end)) if start <= x => (x - start + 1).min(end - x + 1),
            _ => 0,
        }
    }

    /// The Euclidean distance from (x, y) to the nearest pixel not in the
    /// object: row by row away from y, stopping once a row can't be closer.
    fn distance_outside(&self, x: i64, y: i64) -> f64 {
        let mut best = self.across(x, y).pow(2);
        let mut dy = 1i64;
        while dy * dy < best {
            for row in [y - dy, y + dy] {
                let h = self.across(x, row);
                best = best.min(h * h + dy * dy);
            }
            dy += 1;
        }
        (best as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;

    /// An object from rows of '#' (set) and '.' at (`x0`, `y0`).
    fn object(id: u128, rows: &[&str], x0: u32, y0: u32) -> Object {
        let width = rows[0].len();
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, width * rows.len());
        let mut area = 0;
        for (y, row) in rows.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                if c == '#' {
                    mask.set(y * width + x, true);
                    area += 1;
                }
            }
        }
        Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x0, y0, x0 + width as u32 - 1, y0 + rows.len() as u32 - 1],
            area,
            mask_data: mask,
            ..Default::default()
        })
    }

    fn value(row: &[f64; VALUES.len()], id: &str) -> f64 {
        row[VALUES.iter().position(|v| v.0 == id).unwrap()]
    }

    #[test]
    fn a_straight_line() {
        let row = measure(&object(1, &["##########"], 5, 5), None);
        assert_eq!(value(&row, "length"), 9.0);
        assert_eq!(value(&row, "longest_path"), 9.0);
        assert_eq!(value(&row, "end_points"), 2.0);
        assert_eq!(value(&row, "branch_points"), 0.0);
        assert_eq!(value(&row, "segments"), 1.0);
        assert_eq!(value(&row, "mean_segment_length"), 9.0);
        assert_eq!(value(&row, "mean_straightness"), 1.0);
        assert!(value(&row, "mean_diameter").is_nan());
    }

    #[test]
    fn a_diagonal_line_counts_root_two_per_step() {
        let rows = ["#....", ".#...", "..#..", "...#.", "....#"];
        let row = measure(&object(1, &rows, 0, 0), None);
        let expected = 4.0 * std::f64::consts::SQRT_2;
        assert!((value(&row, "length") - expected).abs() < 1e-9);
        assert!((value(&row, "longest_path") - expected).abs() < 1e-9);
        assert_eq!(value(&row, "mean_straightness"), 1.0);
    }

    /// A T: a line of 21 px with a stem of 10 px down from its middle.
    fn tee() -> Object {
        let mut rows = vec!["#".repeat(21)];
        for _ in 0..10 {
            rows.push(format!("{}#{}", ".".repeat(10), ".".repeat(10)));
        }
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        object(1, &rows, 3, 3)
    }

    #[test]
    fn a_tee_has_one_branch_point_and_three_segments() {
        let row = measure(&tee(), None);
        assert_eq!(value(&row, "end_points"), 3.0);
        assert_eq!(value(&row, "branch_points"), 1.0);
        assert_eq!(value(&row, "segments"), 3.0);
        // 20 along the top, 10 down the stem.
        assert_eq!(value(&row, "length"), 30.0);
        // From one end of the top to the stem's end: 10 + 10.
        assert_eq!(value(&row, "longest_path"), 20.0);
        assert_eq!(value(&row, "mean_segment_length"), 10.0);
        assert_eq!(value(&row, "mean_straightness"), 1.0);
    }

    #[test]
    fn a_ring_is_one_segment_without_straightness() {
        let rows = [
            "..###..", ".#...#.", "#.....#", "#.....#", "#.....#", ".#...#.", "..###..",
        ];
        let row = measure(&object(1, &rows, 0, 0), None);
        assert_eq!(value(&row, "end_points"), 0.0);
        assert_eq!(value(&row, "branch_points"), 0.0);
        assert_eq!(value(&row, "segments"), 1.0);
        assert!(value(&row, "mean_straightness").is_nan());
        // 8 steps along the sides, 8 diagonal ones at the corners.
        let expected = 8.0 + 8.0 * std::f64::consts::SQRT_2;
        assert!(
            (value(&row, "length") - expected).abs() < 1e-9,
            "{}",
            value(&row, "length")
        );
    }

    #[test]
    fn a_curved_line_is_less_straight() {
        // An L: 10 right, 10 down - straight distance √200 over 20.
        let mut rows = vec!["#".repeat(11)];
        for _ in 0..10 {
            rows.push(format!("{}#", ".".repeat(10)));
        }
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        let row = measure(&object(1, &rows, 0, 0), None);
        // The corner pixel is a staircase step: cut diagonally (9 + √2 + 9).
        let straightness = value(&row, "mean_straightness");
        let expected = 200f64.sqrt() / (18.0 + std::f64::consts::SQRT_2);
        assert!((straightness - expected).abs() < 1e-9, "{straightness}");
    }

    #[test]
    fn the_diameter_comes_from_the_parent() {
        // A bar 5 px high: its centre line is 5 wide.
        let bar = object(2, &["##############"; 5], 0, 0);
        let skeleton = object(1, &["##########"], 2, 2);
        let row = measure(&skeleton, Some(&RowRuns::of(&bar)));
        assert_eq!(value(&row, "mean_diameter"), 5.0);
        let thin = object(3, &["##############"; 3], 0, 1);
        let row = measure(&skeleton, Some(&RowRuns::of(&thin)));
        assert_eq!(value(&row, "mean_diameter"), 3.0);
    }

    #[test]
    fn distance_outside_is_euclidean() {
        // A 21 x 21 square: its centre is 11 px from the nearest outside pixel.
        let square = object(1, &["#####################"; 21], 0, 0);
        let runs = RowRuns::of(&square);
        assert_eq!(runs.distance_outside(10, 10), 11.0);
        assert_eq!(runs.distance_outside(0, 0), 1.0);
        assert_eq!(runs.distance_outside(30, 30), 0.0);
        // A hole at (1, 1) is nearer to the centre (4, 4) of a 9 x 9 square
        // than its sides (5): diagonally √18.
        let mut rows = vec!["#".repeat(9); 9];
        rows[1].replace_range(1..2, ".");
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        let runs = RowRuns::of(&object(1, &rows, 0, 0));
        assert_eq!(runs.distance_outside(4, 4), (18f64).sqrt());
    }

    #[test]
    fn the_step_reports_its_values() {
        let mut skeleton = tee();
        skeleton
            .add_object_class(ObjectClass::Global(1), Scope::Global)
            .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(ObjectId(1), skeleton);
        let step = MeasureObjectSkeleton {
            id: "sk".into(),
            objects: ObjectClass::Global(1),
            diameter_from: ObjectClass::Unset,
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();
        let reported: Vec<&str> = cache
            .measurements
            .measurements
            .iter()
            .map(|m| m.info.key.as_str())
            .collect();
        // No diameter without "Diameter from".
        assert_eq!(reported.len(), VALUES.len() - 1);
        let value = |id: &str| match &cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == format!("sk/{id}"))
            .unwrap()
            .values
        {
            MeasurementValues::Objects(v) => v[0].value,
            _ => panic!(),
        };
        assert_eq!(value("branch_points"), 1.0);
        assert_eq!(value("length"), 30.0);
    }

    /// Skeletonize Objects, the skeleton linked to its object (as Relate
    /// Objects does), then this step: a plus of 3-pixel-thick arms.
    #[test]
    fn skeletonized_plus_is_measured_with_its_diameter() {
        use crate::algos::object::skeletonize_objects::SkeletonizeObjects;
        use evanalyzer_cfg::core_types::SizeUnits;
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
        let mut plus = object(1, &rows, 10, 10);
        plus.add_object_class(ObjectClass::Global(1), Scope::Global)
            .unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.object_cache.insert(ObjectId(1), plus);
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        ctx.image_meta.full_image_width = kornia_image::ImageSize {
            width: 100,
            height: 100,
        };
        SkeletonizeObjects {
            input_class: ObjectClass::Global(1),
            skeleton_class: ObjectClass::Global(2),
            branch_point_class: ObjectClass::Unset,
            end_point_class: ObjectClass::Unset,
            min_branch_length: 5.0,
            length_unit: SizeUnits::Pixels,
        }
        .execute(&mut ctx, &mut cache)
        .unwrap();
        let skeleton_id = cache
            .object_cache
            .iter()
            .find(|(_, o)| o.is_of(&ObjectClass::Global(2), Scope::Global))
            .map(|(id, _)| id.clone())
            .unwrap();
        cache
            .object_cache
            .get_mut(&skeleton_id)
            .unwrap()
            .set_parent(ObjectClass::Global(1), Some(ObjectId(1)));
        let skeleton = cache.object_cache.get(&skeleton_id).unwrap();
        let parent = RowRuns::of(cache.object_cache.get(&ObjectId(1)).unwrap());
        let row = measure(skeleton, Some(&parent));
        assert_eq!(value(&row, "end_points"), 4.0);
        assert_eq!(value(&row, "branch_points"), 1.0);
        assert_eq!(value(&row, "segments"), 4.0);
        // Arms of 12-13 px around the centre.
        let length = value(&row, "length");
        assert!((48.0..=54.0).contains(&length), "{length}");
        let diameter = value(&row, "mean_diameter");
        assert!((2.9..=3.6).contains(&diameter), "{diameter}");
    }
}
