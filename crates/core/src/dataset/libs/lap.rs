//! # lap
//!
//! The linear assignment problem (LAP) of the tracking steps, sparse: rows
//! (e.g. the objects of frame `t`) are linked to columns (those of frame
//! `t + 1`) so that the sum of the costs is minimal, each row and column
//! used at most once. A row or column left alone costs its alternative
//! cost instead.
//!
//! ## Overview
//! As in Jaqaman et al. (2008, Fig. 1b), the problem becomes a square one:
//! the links, a "no link" entry for every row and column (their
//! alternative costs), and an auxiliary block with an entry for every
//! link, transposed, at cost 0 - it pairs the "no link" entries that a
//! taken link leaves free. Leaving everything unlinked is always possible,
//! so a solution always exists.
//!
//! Only the candidate links are stored. Rows and columns that no link
//! connects are separate problems, each solved on its own (in a tracking
//! problem mostly a few nearby objects), by shortest augmenting paths
//! (Jonker & Volgenant, 1987): Dijkstra over the sparse entries, with
//! potentials that keep the reduced costs non-negative.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A candidate link: row, column, cost (finite, at least 0).
pub(crate) type Link = (usize, usize, f64);

/// The column each row is linked to - `None`: the row isn't linked (it
/// costs its alternative cost). `row_alt` and `col_alt` are the rows' and
/// columns' alternative costs (at least 0).
pub(crate) fn solve(links: &[Link], row_alt: &[f64], col_alt: &[f64]) -> Vec<Option<usize>> {
    let mut assigned = vec![None; row_alt.len()];
    for component in components(links, row_alt.len(), col_alt.len()) {
        for (row, col) in solve_component(&component, row_alt, col_alt) {
            assigned[row] = Some(col);
        }
    }
    assigned
}

/// The links, split into groups that share no row or column - each
/// group's links in their input order.
fn components(links: &[Link], rows: usize, cols: usize) -> Vec<Vec<Link>> {
    // Union-find over rows (0..rows) and columns (rows..rows + cols).
    let mut parent: Vec<usize> = (0..rows + cols).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &(r, c, _) in links {
        let (a, b) = (find(&mut parent, r), find(&mut parent, rows + c));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut group_of_root = vec![usize::MAX; rows + cols];
    let mut groups: Vec<Vec<Link>> = Vec::new();
    for &link in links {
        let root = find(&mut parent, link.0);
        if group_of_root[root] == usize::MAX {
            group_of_root[root] = groups.len();
            groups.push(Vec::new());
        }
        groups[group_of_root[root]].push(link);
    }
    groups
}

/// The links taken among one group's (global row, column) pairs.
fn solve_component(links: &[Link], row_alt: &[f64], col_alt: &[f64]) -> Vec<(usize, usize)> {
    // Local indices, in order of first appearance.
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut local_row = std::collections::HashMap::new();
    let mut local_col = std::collections::HashMap::new();
    for &(r, c, _) in links {
        local_row.entry(r).or_insert_with(|| {
            rows.push(r);
            rows.len() - 1
        });
        local_col.entry(c).or_insert_with(|| {
            cols.push(c);
            cols.len() - 1
        });
    }
    let (nr, nc) = (rows.len(), cols.len());

    // The square matrix: rows 0..nr real, nr..nr+nc one "no link" per
    // column; columns 0..nc real, nc..nc+nr one "no link" per row.
    let mut matrix: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nr + nc];
    for &(r, c, cost) in links {
        let (i, j) = (local_row[&r], local_col[&c]);
        matrix[i].push((j, cost));
        matrix[nr + j].push((nc + i, 0.0));
    }
    for (i, &r) in rows.iter().enumerate() {
        matrix[i].push((nc + i, row_alt[r]));
    }
    for (j, &c) in cols.iter().enumerate() {
        matrix[nr + j].push((j, col_alt[c]));
    }

    let col_of_row = assign(&matrix);
    (0..nr)
        .filter_map(|i| {
            let j = col_of_row[i];
            (j < nc).then(|| (rows[i], cols[j]))
        })
        .collect()
}

/// Min-cost perfect assignment of a square sparse matrix (row -> its
/// entries) that has one: the column of each row.
fn assign(matrix: &[Vec<(usize, f64)>]) -> Vec<usize> {
    let n = matrix.len();
    let mut search = Search {
        matrix,
        u: vec![0.0; n],
        v: vec![0.0; n],
        row_of_col: vec![NONE; n],
        dist: vec![f64::INFINITY; n],
        pred: vec![NONE; n],
        done: vec![false; n],
        reached: Vec::new(),
        heap: BinaryHeap::new(),
    };
    let mut col_of_row = vec![NONE; n];
    for start in 0..n {
        search.augment(start, &mut col_of_row);
    }
    col_of_row
}

const NONE: usize = usize::MAX;

/// The state of the shortest augmenting path search. The reduced cost of
/// entry (i, j), `cost - u[i] - v[j]`, stays at least 0; it is 0 for the
/// entries assigned.
struct Search<'a> {
    matrix: &'a [Vec<(usize, f64)>],
    u: Vec<f64>,
    v: Vec<f64>,
    row_of_col: Vec<usize>,
    /// Per column, from the start row - reset after each search.
    dist: Vec<f64>,
    pred: Vec<usize>,
    done: Vec<bool>,
    /// The columns with a distance, to reset.
    reached: Vec<usize>,
    heap: BinaryHeap<Entry>,
}

impl Search<'_> {
    /// Assigns the free row `start`, along the cheapest path to a free
    /// column (Dijkstra on the reduced costs).
    fn augment(&mut self, start: usize, col_of_row: &mut [usize]) {
        self.relax(start, 0.0);
        let (sink, delta) = loop {
            let Some(Entry { dist, col }) = self.heap.pop() else {
                unreachable!("the matrix has a perfect assignment");
            };
            if self.done[col] || dist > self.dist[col] {
                continue;
            }
            self.done[col] = true;
            match self.row_of_col[col] {
                NONE => break (col, dist),
                row => self.relax(row, dist),
            }
        };

        // New potentials: assigned entries stay at 0, none goes below 0.
        self.u[start] += delta;
        for &col in &self.reached {
            if self.done[col] && col != sink {
                let shift = delta - self.dist[col];
                self.v[col] -= shift;
                self.u[self.row_of_col[col]] += shift;
            }
        }
        // Flip the path back to `start`.
        let mut col = sink;
        loop {
            let row = self.pred[col];
            let next = col_of_row[row];
            self.row_of_col[col] = row;
            col_of_row[row] = col;
            if row == start {
                break;
            }
            col = next;
        }

        for &col in &self.reached {
            self.dist[col] = f64::INFINITY;
            self.done[col] = false;
        }
        self.reached.clear();
        self.heap.clear();
    }

    /// Updates the distances of `row`'s columns, `row` being `base` away.
    fn relax(&mut self, row: usize, base: f64) {
        for &(col, cost) in &self.matrix[row] {
            if self.done[col] {
                continue;
            }
            // Rounding may push a reduced cost a hair below 0.
            let dist = base + (cost - self.u[row] - self.v[col]).max(0.0);
            if dist < self.dist[col] {
                if self.dist[col].is_infinite() {
                    self.reached.push(col);
                }
                self.dist[col] = dist;
                self.pred[col] = row;
                self.heap.push(Entry { dist, col });
            }
        }
    }
}

/// A heap entry: the smallest distance first, ties by column - so the
/// result doesn't depend on the heap's order.
struct Entry {
    dist: f64,
    col: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.col.cmp(&self.col))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cost of `assigned`: its links plus the alternative cost of every
    /// row and column left alone.
    fn total(links: &[Link], assigned: &[Option<usize>], row_alt: &[f64], col_alt: &[f64]) -> f64 {
        let mut col_used = vec![false; col_alt.len()];
        let mut sum = 0.0;
        for (r, c) in assigned.iter().enumerate() {
            match c {
                Some(c) => {
                    assert!(!col_used[*c], "column {c} used twice");
                    col_used[*c] = true;
                    sum += links
                        .iter()
                        .filter(|l| l.0 == r && l.1 == *c)
                        .map(|l| l.2)
                        .fold(f64::INFINITY, f64::min);
                }
                None => sum += row_alt[r],
            }
        }
        sum + col_used
            .iter()
            .zip(col_alt)
            .filter(|(used, _)| !**used)
            .map(|(_, alt)| alt)
            .sum::<f64>()
    }

    /// The lowest total cost over every assignment, by trying them all.
    fn brute_force(links: &[Link], row_alt: &[f64], col_alt: &[f64]) -> f64 {
        fn go(
            row: usize,
            links: &[Link],
            used: &mut Vec<bool>,
            row_alt: &[f64],
            col_alt: &[f64],
        ) -> f64 {
            if row == row_alt.len() {
                return used
                    .iter()
                    .zip(col_alt)
                    .filter(|(u, _)| !**u)
                    .map(|(_, a)| a)
                    .sum();
            }
            let mut best = row_alt[row] + go(row + 1, links, used, row_alt, col_alt);
            for &(r, c, cost) in links.iter().filter(|l| l.0 == row) {
                if !used[c] {
                    used[c] = true;
                    best = best.min(cost + go(r + 1, links, used, row_alt, col_alt));
                    used[c] = false;
                }
            }
            best
        }
        go(0, links, &mut vec![false; col_alt.len()], row_alt, col_alt)
    }

    #[test]
    fn the_global_optimum_beats_closest_pair_first() {
        // Row 0 is closest to column 0, but linking it there leaves row 1
        // without a partner: 0 -> 1 and 1 -> 0 cost less in total.
        let links = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0)];
        let assigned = solve(&links, &[10.0; 2], &[10.0; 2]);
        assert_eq!(assigned, [Some(1), Some(0)]);
    }

    #[test]
    fn a_link_dearer_than_its_alternatives_is_not_taken() {
        let links = [(0, 0, 5.0)];
        assert_eq!(solve(&links, &[2.0], &[2.0]), [None]);
        assert_eq!(solve(&links, &[3.0], &[3.0]), [Some(0)]);
        assert_eq!(solve(&[], &[1.0], &[1.0]), [None]);
    }

    #[test]
    fn random_problems_match_trying_every_assignment() {
        // A small deterministic generator (xorshift), so the test needs no crate.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let rows = (next() % 6) as usize + 1;
            let cols = (next() % 6) as usize + 1;
            let mut links = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    if next() % 3 != 0 {
                        // Integers too, so ties come up.
                        let cost = if next() % 2 == 0 {
                            (next() % 5) as f64
                        } else {
                            (next() % 1000) as f64 / 100.0
                        };
                        links.push((r, c, cost));
                    }
                }
            }
            let row_alt: Vec<f64> = (0..rows).map(|_| (next() % 600) as f64 / 100.0).collect();
            let col_alt: Vec<f64> = (0..cols).map(|_| (next() % 600) as f64 / 100.0).collect();
            let assigned = solve(&links, &row_alt, &col_alt);
            let got = total(&links, &assigned, &row_alt, &col_alt);
            let best = brute_force(&links, &row_alt, &col_alt);
            assert!(
                (got - best).abs() < 1e-9,
                "{got} != {best} for {links:?} {row_alt:?} {col_alt:?}"
            );
        }
    }
}
