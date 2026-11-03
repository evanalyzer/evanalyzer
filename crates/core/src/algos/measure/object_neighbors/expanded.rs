//! # Expand until adjacent, row by row
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! "Expand until adjacent" gives every pixel the label of its nearest object
//! pixel (the Euclidean feature transform, `feature_transform`), then counts
//! per grown object its neighbours and how much of its outline touches them.
//! Without a raster of the image: each column keeps its object pixels as
//! runs, so the feature transform's column pass (the nearest object pixel in
//! the column, the upper one when two are as near) is read off the runs for
//! the current row, and its row pass runs on that row alone - the same input
//! row by row as the whole-image transform, so exactly its labels. The
//! counts only look at a pixel's 3 × 3 neighbourhood: three grown rows at a
//! time. RAM: the runs (as the masks, per object) and a few rows.

use super::{Indexed, paint};
use crate::algos::libs::feature_transform::nearest_in_row;

/// Labels painted at a time (4 MB; tests: a few rows, to cross bands).
#[cfg(not(test))]
const BAND_PIXELS: usize = 1 << 20;
#[cfg(test)]
const BAND_PIXELS: usize = 64;

/// A column's pixels `start..=end` of one label.
struct Run {
    start: u32,
    end: u32,
    label: u32,
}

/// Each object's (number of neighbours, percent touching) after growing
/// `objects` until adjacent; `others`: the neighbours when they aren't the
/// objects themselves (as painted, not grown). An object left without
/// pixels (covered by later ones): (0, 0).
pub(super) fn values(
    objects: &Indexed,
    others: Option<&Indexed>,
    (width, height): (usize, usize),
) -> Vec<(f64, f64)> {
    let n = objects.objects.len();
    let mut perimeter = vec![0u64; n];
    let mut touching = vec![0u64; n];
    let mut found: Vec<Vec<u32>> = vec![Vec::new(); n];
    if width == 0 || height == 0 {
        return vec![(0.0, 0.0); n];
    }
    let runs = column_runs(objects, width, height);
    let mut grower = Grower::new(&runs);
    let mut painted = others.map(|others| Band::new(others, width, height));

    // The grown rows y - 1, y and y + 1.
    let mut above: Vec<u32> = vec![0; width];
    let mut row: Vec<u32> = vec![0; width];
    let mut below: Vec<u32> = vec![0; width];
    grower.row(0, &mut row);
    for y in 0..height {
        if y + 1 < height {
            grower.row(y + 1, &mut below);
        }
        let up = (y > 0).then_some(above.as_slice());
        let down = (y + 1 < height).then_some(below.as_slice());
        let other_rows = painted.as_mut().map(|band| band.rows(y));
        for x in 0..width {
            let label = row[x];
            if label == 0 {
                continue;
            }
            let i = label as usize - 1;
            let left = x.checked_sub(1);
            let right = (x + 1 < width).then_some(x + 1);
            // Any of the 8 neighbours (in the image) of another label.
            let differs = |r: &[u32]| {
                left.is_some_and(|l| r[l] != label)
                    || r[x] != label
                    || right.is_some_and(|r2| r[r2] != label)
            };
            let other_label = up.is_some_and(differs)
                || down.is_some_and(differs)
                || left.is_some_and(|l| row[l] != label)
                || right.is_some_and(|r| row[r] != label);

            // The neighbours: the labels a dilation by 1 (a cross) reaches.
            match &other_rows {
                None => {
                    if other_label {
                        let cross = [
                            up.map(|r| r[x]),
                            down.map(|r| r[x]),
                            left.map(|l| row[l]),
                            right.map(|r| row[r]),
                        ];
                        for neighbor in cross.into_iter().flatten() {
                            if neighbor != 0 && neighbor != label && !found[i].contains(&neighbor) {
                                found[i].push(neighbor);
                            }
                        }
                    }
                }
                Some((o_up, o_row, o_down)) => {
                    let cross = [
                        Some(o_row[x]),
                        o_up.map(|r| r[x]),
                        o_down.map(|r| r[x]),
                        left.map(|l| o_row[l]),
                        right.map(|r| o_row[r]),
                    ];
                    for neighbor in cross.into_iter().flatten() {
                        if neighbor != 0 && !found[i].contains(&neighbor) {
                            found[i].push(neighbor);
                        }
                    }
                }
            }

            // Percent touching: outline pixels (on the image border or next
            // to another label) within 1.5 px of a neighbour.
            let border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            if !(border || other_label) {
                continue;
            }
            perimeter[i] += 1;
            let touches = match &other_rows {
                // Every pixel has a label once grown: another label is a
                // neighbour.
                None => other_label,
                Some((o_up, o_row, o_down)) => {
                    let any = |r: &[u32]| {
                        left.is_some_and(|l| r[l] != 0)
                            || r[x] != 0
                            || right.is_some_and(|r2| r[r2] != 0)
                    };
                    any(o_row) || o_up.is_some_and(any) || o_down.is_some_and(any)
                }
            };
            if touches {
                touching[i] += 1;
            }
        }
        std::mem::swap(&mut above, &mut row);
        std::mem::swap(&mut row, &mut below);
    }
    (0..n)
        .map(|i| match perimeter[i] {
            0 => (0.0, 0.0),
            p => (found[i].len() as f64, touching[i] as f64 * 100.0 / p as f64),
        })
        .collect()
}

/// Each column's object pixels as runs of one label, top to bottom (later
/// objects over earlier ones, as painted).
fn column_runs(objects: &Indexed, width: usize, height: usize) -> Vec<Vec<Run>> {
    let rows_per_band = (BAND_PIXELS / width).max(1);
    let mut runs: Vec<Vec<Run>> = (0..width).map(|_| Vec::new()).collect();
    for y0 in (0..height).step_by(rows_per_band) {
        let y1 = (y0 + rows_per_band).min(height);
        let band = paint(objects, [0, y0, width, y1]);
        for (dy, labels) in band.chunks_exact(width).enumerate() {
            let y = (y0 + dy) as u32;
            for (column, &label) in runs.iter_mut().zip(labels) {
                if label == 0 {
                    continue;
                }
                match column.last_mut() {
                    Some(run) if run.end + 1 == y && run.label == label => run.end = y,
                    _ => column.push(Run {
                        start: y,
                        end: y,
                        label,
                    }),
                }
            }
        }
    }
    runs
}

/// The grown labels, row by row from the top.
struct Grower<'a> {
    runs: &'a [Vec<Run>],
    /// Per column: its first run not above the current row.
    next: Vec<usize>,
    /// The row's candidates: each column's nearest object pixel in it.
    line: Vec<[i32; 2]>,
    /// The candidates' labels, by column.
    labels: Vec<u32>,
    g: Vec<usize>,
    f: Vec<[i32; 2]>,
}

impl<'a> Grower<'a> {
    fn new(runs: &'a [Vec<Run>]) -> Self {
        let width = runs.len();
        Self {
            runs,
            next: vec![0; width],
            line: vec![[-1, 0]; width],
            labels: vec![0; width],
            g: Vec::with_capacity(width),
            f: Vec::with_capacity(width),
        }
    }

    /// Row `y`'s grown labels into `out` (rows in order, from the top;
    /// 0: there are no object pixels at all).
    fn row(&mut self, y: usize, out: &mut [u32]) {
        let y32 = y as u32;
        for (x, column) in self.runs.iter().enumerate() {
            let mut k = self.next[x];
            while k < column.len() && column[k].end < y32 {
                k += 1;
            }
            self.next[x] = k;
            let up = k.checked_sub(1).map(|j| &column[j]);
            let candidate = match (up, column.get(k)) {
                (_, Some(run)) if run.start <= y32 => Some((y32, run.label)),
                // The upper one when both are as near (the column pass).
                (Some(a), Some(b)) if y32 - a.end <= b.start - y32 => Some((a.end, a.label)),
                (_, Some(b)) => Some((b.start, b.label)),
                (Some(a), None) => Some((a.end, a.label)),
                (None, None) => None,
            };
            (self.line[x], self.labels[x]) = match candidate {
                Some((r, label)) => ([r as i32, x as i32], label),
                None => ([-1, 0], 0),
            };
        }
        nearest_in_row(&mut self.line, y, &mut self.g, &mut self.f);
        for (out, nearest) in out.iter_mut().zip(&self.line) {
            *out = match nearest[0] >= 0 {
                true => self.labels[nearest[1] as usize],
                false => 0,
            };
        }
    }
}

/// Painted labels of the neighbour objects, a band of rows at a time.
struct Band<'a> {
    objects: &'a Indexed<'a>,
    width: usize,
    height: usize,
    /// The band's first row.
    y0: usize,
    labels: Vec<u32>,
}

impl<'a> Band<'a> {
    fn new(objects: &'a Indexed<'a>, width: usize, height: usize) -> Self {
        Self {
            objects,
            width,
            height,
            y0: 0,
            labels: Vec::new(),
        }
    }

    /// Rows y - 1, y and y + 1 (`None`: outside the image).
    fn rows(&mut self, y: usize) -> (Option<&[u32]>, &[u32], Option<&[u32]>) {
        let first = y.saturating_sub(1);
        let last = (y + 1).min(self.height - 1);
        let rows_held = self.labels.len() / self.width;
        if first < self.y0 || last >= self.y0 + rows_held {
            let rows = (BAND_PIXELS / self.width).max(3);
            self.y0 = first;
            let y1 = (first + rows).min(self.height);
            self.labels = paint(self.objects, [0, first, self.width, y1]);
        }
        let at = |r: usize| {
            let start = (r - self.y0) * self.width;
            &self.labels[start..start + self.width]
        };
        (
            (y > 0).then(|| at(y - 1)),
            at(y),
            (y + 1 < self.height).then(|| at(y + 1)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Indexed, Patch, dilate, disk, is_outline, paint};
    use super::*;
    use crate::algos::libs::feature_transform::nearest_features;
    use crate::object::{Object, ObjectInit};
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::ObjectId;

    /// The code before (2026-11-03): the objects grown over one raster of
    /// the whole image, then each object's values on a window of it.
    mod before {
        use super::*;

        fn grown(objects: &Indexed, width: usize, height: usize) -> Vec<u32> {
            let labels = paint(objects, [0, 0, width, height]);
            let ft = nearest_features(width, height, |x, y| labels[y * width + x] != 0);
            (0..labels.len())
                .map(|i| match (labels[i], ft.get(i)) {
                    (0, Some((r, c))) => labels[r as usize * width + c as usize],
                    (l, _) => l,
                })
                .collect()
        }

        fn raster_extents(labels: &[u32], width: usize, n: usize) -> Vec<Option<[usize; 4]>> {
            let mut extents: Vec<Option<[usize; 4]>> = vec![None; n];
            for (i, l) in labels.iter().enumerate() {
                if *l == 0 {
                    continue;
                }
                let (x, y) = (i % width, i / width);
                let e = &mut extents[*l as usize - 1];
                *e = Some(match *e {
                    None => [x, y, x, y],
                    Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
                });
            }
            extents
        }

        pub(super) fn values(
            objects: &Indexed,
            others: Option<&Indexed>,
            (width, height): (usize, usize),
        ) -> Vec<(f64, f64)> {
            let n = objects.objects.len();
            let same = others.is_none();
            let raster = grown(objects, width, height);
            let extents = raster_extents(&raster, width, n);
            let reach = disk(1.0);
            let touch = disk(1.5);
            (0..n)
                .map(|index| {
                    let label = index as u32 + 1;
                    let Some([ex0, ey0, ex1, ey1]) = extents[index] else {
                        return (0.0, 0.0);
                    };
                    let window = [
                        ex0.saturating_sub(1),
                        ey0.saturating_sub(1),
                        (ex1 + 2).min(width),
                        (ey1 + 2).min(height),
                    ];
                    let [x0, y0, x1, y1] = window;
                    let patch = Patch {
                        window,
                        labels: (y0..y1)
                            .flat_map(|y| raster[y * width + x0..y * width + x1].iter().copied())
                            .collect(),
                    };
                    let npatch = others.map(|o| Patch {
                        window,
                        labels: paint(o, window),
                    });
                    let npatch = npatch.as_ref().unwrap_or(&patch);
                    let extended = dilate(&patch, |l| l == label, &reach, 1.0);
                    let found: std::collections::BTreeSet<u32> = extended
                        .iter()
                        .zip(&npatch.labels)
                        .filter(|(e, l)| **e && **l != 0 && !(same && **l == label))
                        .map(|(_, l)| *l)
                        .collect();
                    let near_others = if same {
                        dilate(&patch, |l| l != 0 && l != label, &touch, 1.5)
                    } else {
                        dilate(npatch, |l| l != 0, &touch, 1.5)
                    };
                    let (mut perimeter, mut touching) = (0usize, 0usize);
                    let pw = patch.width();
                    for (i, l) in patch.labels.iter().enumerate() {
                        let (x, y) = (window[0] + i % pw, window[1] + i / pw);
                        if *l != label || !is_outline(&patch, x, y, (width, height)) {
                            continue;
                        }
                        perimeter += 1;
                        if near_others[i] {
                            touching += 1;
                        }
                    }
                    (
                        found.len() as f64,
                        touching as f64 * 100.0 / perimeter as f64,
                    )
                })
                .collect()
        }
    }

    /// A pseudo-random generator (xorshift), the same numbers every run.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n.max(1) as u64) as usize
        }
    }

    /// Random objects (boxes with holes, single pixels, lattices of points
    /// for many equally near pixels), some overlapping.
    fn random_objects(rng: &mut Rng, width: usize, height: usize) -> Vec<Object> {
        let count = 1 + rng.below(25);
        let lattice = rng.below(4) == 0;
        let step = 2 + rng.below(6);
        (0..count)
            .filter_map(|i| {
                let (x0, y0) = if lattice {
                    (
                        (i * step) % width,
                        (i * step / width.max(1) * step) % height,
                    )
                } else {
                    (rng.below(width), rng.below(height))
                };
                let w = 1 + rng.below(if lattice { 1 } else { 8 }).min(width - x0 - 1);
                let h = 1 + rng.below(if lattice { 1 } else { 8 }).min(height - y0 - 1);
                let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * h);
                for p in 0..w * h {
                    mask.set(p, rng.below(5) != 0);
                }
                mask.set(0, true);
                let area = mask.count_ones();
                Some(Object::new(ObjectInit {
                    id: ObjectId(i as u128 + 1),
                    bbox: [
                        x0 as u32,
                        y0 as u32,
                        (x0 + w - 1) as u32,
                        (y0 + h - 1) as u32,
                    ],
                    area,
                    mask_data: mask,
                    ..Default::default()
                }))
            })
            .collect()
    }

    fn indexed(objects: &[Object]) -> Indexed<'_> {
        let ids: Vec<ObjectId> = objects.iter().map(|o| o.id.clone()).collect();
        Indexed::new(ids.iter().zip(objects).collect())
    }

    /// Row by row gives exactly the values of the whole-image raster -
    /// ties (lattices), overlaps, objects covered by others, the image
    /// border, neighbours of another class, bands of a few rows.
    #[test]
    fn row_by_row_gives_the_values_of_the_whole_image_raster() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..3000 {
            let (width, height) = (1 + rng.below(40), 1 + rng.below(40));
            let objects = random_objects(&mut rng, width, height);
            let objects = indexed(&objects);
            let spots = random_objects(&mut rng, width, height);
            let spots = indexed(&spots);
            for others in [None, Some(&spots)] {
                assert_eq!(
                    values(&objects, others, (width, height)),
                    before::values(&objects, others, (width, height)),
                    "case {case}: {width} x {height}, neighbours of another class: {}",
                    others.is_some()
                );
            }
        }
    }
}
