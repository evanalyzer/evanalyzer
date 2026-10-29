//! # linking
//!
//! The two linking steps the LAP trackers share (Jaqaman et al., 2008):
//! objects are linked frame to frame into track segments, then the
//! segments are linked to each other - closing gaps, and capturing merges
//! and splits. Both steps are linear assignment problems ([`super::lap`]),
//! global in space; the second also in time.
//!
//! ## Overview
//! What a link costs - and how far apart partners may be - is the
//! tracker's ([`Linker`]): squared distance (Track LAP), squared distance
//! to the predicted position (Track Kalman), mask overlap (Track Overlap).
//!
//! Step 1 links the segments ending in frame `t` to the objects of frame
//! `t + 1`. Step 2 links every segment's end to another's start up to
//! `max_gap_frames` missing frames later (gap closing), a segment's end to
//! a middle point of another one in the next frame (merge), and a
//! segment's start to a middle point of another one in the frame before
//! (split). Merges and splits cost the pair's cost times the intensity
//! penalty of equations 5 and 6 of the paper.
//!
//! Not linking costs the alternative cost: 1.05 times the highest cost a
//! link may have (its cutoff). So any possible link beats no link on its
//! own, and the links compete with each other only. (The paper derives
//! its alternative costs and cutoffs from the data - its Supplementary
//! Note 3; here the cutoffs are the user's.)
//!
//! Tracks: a closed gap continues the track. A merge ends both tracks and
//! starts a new one with both as parents; a split ends the track and
//! starts two, the parent's continuation and the new object, each with the
//! track as parent.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-29
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::lap;
use crate::dataset::DatasetContext;
use crate::storage::stored_object::{ObjectFilter, StoredObject, TrackAssignment};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass, SizeUnits,
};
use macros::CommandsMeta;
use std::collections::{BTreeMap, HashMap};

pub(crate) const JAQAMAN_2008: CitationMetadata = CitationMetadata {
    cite_key: "jaqaman2008robust",
    title: "Robust single-particle tracking in live-cell time-lapse sequences",
    authors: &[
        "Khuloud Jaqaman",
        "Dinah Loerke",
        "Marcel Mettlen",
        "Hirotaka Kuwata",
        "Sergio Grinstein",
        "Sandra L. Schmid",
        "Gaudenz Danuser",
    ],
    year: 2008,
    container: Some("Nature Methods"),
    doi: Some("10.1038/nmeth.1237"),
    url: Some("https://doi.org/10.1038/nmeth.1237"),
    pages: Some("695-702"),
};

pub(crate) const JONKER_1987: CitationMetadata = CitationMetadata {
    cite_key: "jonker1987shortest",
    title: "A shortest augmenting path algorithm for dense and sparse linear assignment problems",
    authors: &["Roy Jonker", "Anton Volgenant"],
    year: 1987,
    container: Some("Computing"),
    doi: Some("10.1007/BF02278710"),
    url: Some("https://doi.org/10.1007/BF02278710"),
    pages: Some("325-340"),
};

/// What a tracker's links cost. Distances are in pixels of the image.
pub(crate) trait Linker {
    /// What a segment remembers of its objects (e.g. a motion filter).
    type State;

    fn start(&self, object: &StoredObject) -> Self::State;
    fn extend(&self, state: &mut Self::State, object: &StoredObject);

    /// Where a partner of the segment (`state`, ending in `last`) is
    /// looked for `frames` later: a centre and the radius around it in
    /// which the centroid of a partner lies (plus the partner's
    /// [`Self::extent`]). `closing`: step 2 (gap closing).
    fn search(
        &self,
        state: &Self::State,
        last: &StoredObject,
        frames: i32,
        closing: bool,
    ) -> ((f64, f64), f64);

    /// The cost of `object` continuing the segment `frames` later - `None`
    /// beyond the cutoff.
    fn cost(
        &self,
        state: &Self::State,
        last: &StoredObject,
        frames: i32,
        object: &StoredObject,
        closing: bool,
    ) -> Option<f64>;

    /// The highest cost `cost` returns (the cutoff).
    fn max_cost(&self, closing: bool) -> f64;

    /// The cost of `a` and `b` (one frame apart) being the same object
    /// before or after a merge or split, without the intensity penalty -
    /// at most `max_cost(false)`.
    fn pair_cost(&self, a: &StoredObject, b: &StoredObject) -> Option<f64>;

    /// How far from `a`'s centroid a `pair_cost` partner's lies (plus its
    /// extent).
    fn pair_radius(&self, a: &StoredObject) -> f64;

    /// How far an object reaches from its centroid, for the search - 0
    /// when only centroids count.
    fn extent(&self, _object: &StoredObject) -> f64 {
        0.0
    }
}

/// What step 2 does.
pub(crate) struct GapOptions {
    /// How many frames in a row an object may be missing (0: no gap
    /// closing).
    pub max_gap_frames: i32,
    pub merging: bool,
    pub splitting: bool,
    /// The amount a merge adds up and a split divides (the "intensity" of
    /// equation 6) - `None` when unknown: no penalty.
    pub amount: Box<dyn Fn(&StoredObject) -> Option<f64>>,
}

/// One object's track.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Linked {
    /// Index into the objects.
    pub object: usize,
    pub track_id: u64,
    pub parent_track_ids: Vec<u64>,
}

/// Tracks the objects of one image and z-plane (`frames`: frame -> indices
/// into `objects`), track ids from `*next_id` on (it is advanced).
pub(crate) fn track<L: Linker>(
    linker: &L,
    objects: &[StoredObject],
    frames: &BTreeMap<i32, Vec<usize>>,
    options: &GapOptions,
    next_id: &mut u64,
) -> Vec<Linked> {
    let segments = link_frames(linker, objects, frames);
    let links = link_segments(linker, objects, &segments, options);
    assemble(objects, &segments, &links, next_id)
}

/// Objects of consecutive frames, linked in step 1.
pub(crate) struct Segment<S> {
    pub objects: Vec<usize>,
    pub state: S,
}

/// Step 1: the segments, linking each frame's objects to the segments
/// ending in the frame before.
fn link_frames<L: Linker>(
    linker: &L,
    objects: &[StoredObject],
    frames: &BTreeMap<i32, Vec<usize>>,
) -> Vec<Segment<L::State>> {
    let alt = alternative_cost(linker.max_cost(false));
    let mut segments: Vec<Segment<L::State>> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut previous_t: Option<i32> = None;
    for (&t, current) in frames {
        if previous_t != Some(t - 1) {
            active.clear();
        }
        let queries: Vec<_> = active
            .iter()
            .map(|&s| {
                let segment = &segments[s];
                let last = &objects[*segment.objects.last().unwrap()];
                linker.search(&segment.state, last, 1, false)
            })
            .collect();
        let grid = Grid::new(linker, objects, current.iter().copied(), &queries);
        let mut links = Vec::new();
        for (row, (&s, query)) in active.iter().zip(&queries).enumerate() {
            let segment = &segments[s];
            let last = &objects[*segment.objects.last().unwrap()];
            for col in grid.near(*query) {
                let object = &objects[current[col]];
                if let Some(cost) = linker.cost(&segment.state, last, 1, object, false) {
                    links.push((row, col, cost));
                }
            }
        }
        let assigned = lap::solve(&links, &vec![alt; active.len()], &vec![alt; current.len()]);

        let mut continues = vec![None; current.len()];
        for (row, col) in assigned.iter().enumerate() {
            if let Some(col) = col {
                continues[*col] = Some(active[row]);
            }
        }
        active = current
            .iter()
            .zip(continues)
            .map(|(&o, continues)| match continues {
                Some(s) => {
                    segments[s].objects.push(o);
                    linker.extend(&mut segments[s].state, &objects[o]);
                    s
                }
                None => {
                    segments.push(Segment {
                        objects: vec![o],
                        state: linker.start(&objects[o]),
                    });
                    segments.len() - 1
                }
            })
            .collect();
        previous_t = Some(t);
    }
    segments
}

/// A link of step 2.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SegmentLink {
    /// The end of segment `from` to the start of `to`.
    Gap { from: usize, to: usize },
    /// The end of `from` into `to`'s object `at` (not its first).
    Merge { from: usize, to: usize, at: usize },
    /// The start of `to` out of `from`'s object `at` (not its last).
    Split { from: usize, at: usize, to: usize },
}

/// Step 2: the links between segments.
fn link_segments<L: Linker>(
    linker: &L,
    objects: &[StoredObject],
    segments: &[Segment<L::State>],
    options: &GapOptions,
) -> Vec<SegmentLink> {
    let merge_split = options.merging || options.splitting;
    if options.max_gap_frames <= 0 && !merge_split {
        return Vec::new();
    }
    let t = |o: usize| objects[o].t_stack;
    let first = |s: usize| segments[s].objects[0];
    let last = |s: usize| *segments[s].objects.last().unwrap();
    let n = segments.len();

    // Rows: 0..n the segments' ends, then the middle points splits come
    // out of. Columns: 0..n the starts, then the middle points merges go
    // into.
    let mut links = Vec::new();
    let mut row_points = Points::default();
    let mut col_points = Points::default();

    if options.max_gap_frames > 0 {
        let mut starts: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        let mut ends: HashMap<i32, Vec<usize>> = HashMap::new();
        for s in 0..n {
            starts.entry(t(first(s))).or_default().push(s);
            ends.entry(t(last(s))).or_default().push(s);
        }
        let none = Vec::new();
        for (&ts, starting) in &starts {
            // The ends 1 to max_gap + 1 frames before.
            let ending: Vec<(usize, i32)> = (1..=options.max_gap_frames + 1)
                .flat_map(|gap| {
                    let ending = ends.get(&(ts - gap)).unwrap_or(&none);
                    ending.iter().map(move |&s| (s, gap))
                })
                .collect();
            let queries: Vec<_> = ending
                .iter()
                .map(|&(s, gap)| linker.search(&segments[s].state, &objects[last(s)], gap, true))
                .collect();
            let firsts = starting.iter().map(|&s| first(s));
            let grid = Grid::new(linker, objects, firsts, &queries);
            for (&(from, gap), query) in ending.iter().zip(&queries) {
                for i in grid.near(*query) {
                    let to = starting[i];
                    let start = &objects[first(to)];
                    let state = &segments[from].state;
                    if let Some(cost) = linker.cost(state, &objects[last(from)], gap, start, true) {
                        links.push((from, to, cost));
                    }
                }
            }
        }
    }

    if merge_split {
        // Every object, by frame: (segment, index in it).
        let mut points: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
        for (s, segment) in segments.iter().enumerate() {
            for (k, &o) in segment.objects.iter().enumerate() {
                points.entry(t(o)).or_default().push((s, k));
            }
        }
        let amount = |o: usize| (options.amount)(&objects[o]);
        let empty = Vec::new();
        for (&frame, here) in &points {
            let next = points.get(&(frame + 1)).unwrap_or(&empty);
            let at = |(s, k): (usize, usize)| segments[s].objects[k];
            // Merges: an end here into a middle point (not a first) in the
            // next frame; splits: a start in the next frame out of a middle
            // point (not a last) here.
            let ends: Vec<usize> = here
                .iter()
                .filter(|&&(s, k)| options.merging && k + 1 == segments[s].objects.len())
                .map(|&(s, _)| s)
                .collect();
            let middles: Vec<(usize, usize)> = here
                .iter()
                .filter(|&&(s, k)| options.splitting && k + 1 < segments[s].objects.len())
                .copied()
                .collect();
            let queries: Vec<_> = ends
                .iter()
                .map(|&s| last(s))
                .chain(middles.iter().map(|&p| at(p)))
                .map(|o| (objects[o].centroid_px, linker.pair_radius(&objects[o])))
                .collect();
            let grid = Grid::new(linker, objects, next.iter().map(|&p| at(p)), &queries);
            for (q, query) in queries.iter().enumerate() {
                for i in grid.near(*query) {
                    let (s, k) = next[i];
                    if q < ends.len() {
                        let from = ends[q];
                        if k == 0 {
                            continue;
                        }
                        // rho = A_J(t) / (A_I(t-1) + A_J(t-1))
                        let (end, into, before) = (last(from), at((s, k)), at((s, k - 1)));
                        let rho = ratio(amount(into), amount(end), amount(before));
                        if let Some(cost) = linker.pair_cost(&objects[end], &objects[into]) {
                            let col = n + col_points.index_of((s, k));
                            links.push((from, col, cost * penalty(rho)));
                        }
                    } else {
                        let (from, at_k) = middles[q - ends.len()];
                        if k != 0 {
                            continue;
                        }
                        // rho = A_I(t-1) / (A_I(t) + A_J(t))
                        let (out_of, after, start) =
                            (at((from, at_k)), at((from, at_k + 1)), first(s));
                        let rho = ratio(amount(out_of), amount(after), amount(start));
                        if let Some(cost) = linker.pair_cost(&objects[out_of], &objects[start]) {
                            let row = n + row_points.index_of((from, at_k));
                            links.push((row, s, cost * penalty(rho)));
                        }
                    }
                }
            }
        }
    }

    let mut max_cost = 0.0f64;
    if options.max_gap_frames > 0 {
        max_cost = max_cost.max(linker.max_cost(true));
    }
    if merge_split {
        max_cost = max_cost.max(linker.max_cost(false));
    }
    let alt = alternative_cost(max_cost);
    let assigned = lap::solve(
        &links,
        &vec![alt; n + row_points.list.len()],
        &vec![alt; n + col_points.list.len()],
    );
    assigned
        .iter()
        .enumerate()
        .filter_map(|(row, col)| {
            let col = (*col)?;
            Some(match (row < n, col < n) {
                (true, true) => SegmentLink::Gap { from: row, to: col },
                (true, false) => {
                    let (to, at) = col_points.list[col - n];
                    SegmentLink::Merge { from: row, to, at }
                }
                (false, true) => {
                    let (from, at) = row_points.list[row - n];
                    SegmentLink::Split { from, at, to: col }
                }
                (false, false) => unreachable!("middle points are never linked to each other"),
            })
        })
        .collect()
}

/// Middle points (segment, index in it) numbered in order of appearance.
#[derive(Default)]
struct Points {
    list: Vec<(usize, usize)>,
    index: HashMap<(usize, usize), usize>,
}

impl Points {
    /// The number of `point`, a new one if it has none yet.
    fn index_of(&mut self, point: (usize, usize)) -> usize {
        *self.index.entry(point).or_insert_with(|| {
            self.list.push(point);
            self.list.len() - 1
        })
    }
}

/// `a / (b + c)` - `None` if an amount is unknown or the sum is 0.
fn ratio(a: Option<f64>, b: Option<f64>, c: Option<f64>) -> Option<f64> {
    let sum = b? + c?;
    if sum > 0.0 { Some(a? / sum) } else { None }
}

/// The intensity penalty of equation 5: rho above 1, rho^-2 below - so an
/// amount that is too small costs more than one too big. 1 when unknown.
fn penalty(rho: Option<f64>) -> f64 {
    match rho {
        Some(rho) if rho > 1.0 => rho,
        Some(rho) if rho > 0.0 => rho.powi(-2),
        // A merged or split object of nothing: as unlikely as can be.
        Some(_) => f64::MAX,
        None => 1.0,
    }
}

/// The cost of not linking, given the highest cost of a link - a bit above
/// it, and above 0 so that a link of cost 0 still wins.
fn alternative_cost(max_cost: f64) -> f64 {
    1.05 * max_cost + 1e-9
}

/// Builds the tracks from the segments and step 2's links.
fn assemble<S>(
    objects: &[StoredObject],
    segments: &[Segment<S>],
    links: &[SegmentLink],
    next_id: &mut u64,
) -> Vec<Linked> {
    // A merge or split cuts a segment into pieces: before and from `at`
    // (merge) or from `at + 1` (split).
    let mut cuts: Vec<Vec<usize>> = vec![Vec::new(); segments.len()];
    for link in links {
        match *link {
            SegmentLink::Merge { to, at, .. } => cuts[to].push(at),
            SegmentLink::Split { from, at, .. } => cuts[from].push(at + 1),
            SegmentLink::Gap { .. } => {}
        }
    }
    // Pieces: (segment, first index, end index), per segment in order.
    let mut pieces: Vec<(usize, usize, usize)> = Vec::new();
    let mut first_piece = Vec::with_capacity(segments.len());
    for (s, segment) in segments.iter().enumerate() {
        let cuts = &mut cuts[s];
        cuts.sort_unstable();
        cuts.dedup();
        first_piece.push(pieces.len());
        let mut from = 0;
        for &cut in cuts.iter().chain(std::iter::once(&segment.objects.len())) {
            pieces.push((s, from, cut));
            from = cut;
        }
    }
    // The piece of segment `s` holding its object `k`.
    let piece_of = |s: usize, k: usize| first_piece[s] + cuts[s].partition_point(|&c| c <= k);
    let last_piece = |s: usize| first_piece[s] + cuts[s].len();

    // A gap continues the track: pieces joined (union-find).
    let mut root: Vec<usize> = (0..pieces.len()).collect();
    fn find(root: &mut [usize], mut x: usize) -> usize {
        while root[x] != x {
            root[x] = root[root[x]];
            x = root[x];
        }
        x
    }
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];
    for link in links {
        match *link {
            SegmentLink::Gap { from, to } => {
                let (a, b) = (
                    find(&mut root, last_piece(from)),
                    find(&mut root, first_piece[to]),
                );
                root[b] = a;
            }
            SegmentLink::Merge { from, to, at } => {
                parents[piece_of(to, at)].push(last_piece(from));
            }
            SegmentLink::Split { from, at, to } => {
                parents[first_piece[to]].push(piece_of(from, at));
            }
        }
    }
    // The piece after a cut comes from the one before it.
    for (s, cuts) in cuts.iter().enumerate() {
        for &cut in cuts {
            parents[piece_of(s, cut)].push(piece_of(s, cut - 1));
        }
    }

    // Track ids in order of the tracks' first object (frame, then index).
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    let first_object = |p: usize| segments[pieces[p].0].objects[pieces[p].1];
    order.sort_by_key(|&p| (objects[first_object(p)].t_stack, first_object(p)));
    let mut track_of_root: HashMap<usize, u64> = HashMap::new();
    for &p in &order {
        let r = find(&mut root, p);
        track_of_root.entry(r).or_insert_with(|| {
            *next_id += 1;
            *next_id - 1
        });
    }
    let mut track_parents: HashMap<u64, Vec<u64>> = HashMap::new();
    for (p, parent_pieces) in parents.iter().enumerate() {
        let track = track_of_root[&find(&mut root, p)];
        for &q in parent_pieces {
            let parent = track_of_root[&find(&mut root, q)];
            if parent != track {
                track_parents.entry(track).or_default().push(parent);
            }
        }
    }
    for parents in track_parents.values_mut() {
        parents.sort_unstable();
        parents.dedup();
    }

    let mut linked = Vec::with_capacity(objects.len());
    for (p, &(s, from, to)) in pieces.iter().enumerate() {
        let track_id = track_of_root[&find(&mut root, p)];
        let parent_track_ids = track_parents.get(&track_id).cloned().unwrap_or_default();
        for &object in &segments[s].objects[from..to] {
            linked.push(Linked {
                object,
                track_id,
                parent_track_ids: parent_track_ids.clone(),
            });
        }
    }
    linked
}

/// The objects of a frame by centroid, in square cells - to find the
/// partners near a point without comparing with every object.
struct Grid {
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
    /// The largest extent of the objects, added to every search radius.
    extent: f64,
}

impl Grid {
    /// The grid of `members` (indices into `objects`; [`Self::near`]
    /// returns positions in this list), sized for `queries`.
    fn new<L: Linker>(
        linker: &L,
        objects: &[StoredObject],
        members: impl Iterator<Item = usize>,
        queries: &[((f64, f64), f64)],
    ) -> Self {
        let members: Vec<usize> = members.collect();
        let extent = members
            .iter()
            .map(|&o| linker.extent(&objects[o]))
            .fold(0.0, f64::max);
        let radius = queries.iter().map(|q| q.1).fold(0.0, f64::max) + extent;
        // A cell as big as the biggest search: a search covers few cells.
        let cell = if radius > 0.0 && radius.is_finite() {
            radius
        } else {
            1.0
        };
        let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, &o) in members.iter().enumerate() {
            cells
                .entry(cell_of(objects[o].centroid_px, cell))
                .or_default()
                .push(i);
        }
        Self {
            cell,
            cells,
            extent,
        }
    }

    /// The members whose centroid may lie within `radius` (plus their
    /// extent) of `centre` - a superset, in member order.
    fn near(&self, (centre, radius): ((f64, f64), f64)) -> Vec<usize> {
        let r = radius + self.extent;
        let (x0, y0) = cell_of((centre.0 - r, centre.1 - r), self.cell);
        let (x1, y1) = cell_of((centre.0 + r, centre.1 + r), self.cell);
        let mut found = Vec::new();
        for x in x0..=x1 {
            for y in y0..=y1 {
                if let Some(members) = self.cells.get(&(x, y)) {
                    found.extend_from_slice(members);
                }
            }
        }
        found.sort_unstable();
        found
    }
}

fn cell_of((x, y): (f64, f64), cell: f64) -> (i64, i64) {
    ((x / cell).floor() as i64, (y / cell).floor() as i64)
}

/// `objects` by image and z-plane, then frame (object indices) - the
/// stacks [`track`] tracks one by one.
pub(crate) fn stacks(objects: &[StoredObject]) -> BTreeMap<(&str, i32), BTreeMap<i32, Vec<usize>>> {
    let mut stacks: BTreeMap<(&str, i32), BTreeMap<i32, Vec<usize>>> = BTreeMap::new();
    for (i, object) in objects.iter().enumerate() {
        stacks
            .entry((&object.image_rel_path, object.z_stack))
            .or_default()
            .entry(object.t_stack)
            .or_default()
            .push(i);
    }
    stacks
}

/// What a merge adds up and a split divides - telling a true merge or
/// split from objects that only pass close by.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum MergeSplitAmount {
    /// The area: for objects that keep their size (e.g. cells).
    #[cmdsmeta(display_name = "Area")]
    Area,
    /// The summed intensity of `channel` (mean times area): for spots
    /// below the resolution limit (e.g. vesicles), whose brightness adds
    /// up but whose size doesn't.
    #[cmdsmeta(display_name = "Intensity")]
    Intensity,
}

impl MergeSplitAmount {
    /// The amount of an object - `None` where unknown (the channel wasn't
    /// measured).
    pub(crate) fn of(self, channel: ImageChannelIdx) -> Box<dyn Fn(&StoredObject) -> Option<f64>> {
        match self {
            Self::Area => Box::new(|o| Some(o.area_px as f64)),
            Self::Intensity => Box::new(move |o| {
                let mean = (*o.intensity_mean.get(channel.0 as usize)?)?;
                Some(mean * o.area_px as f64)
            }),
        }
    }
}

/// `value` (in `unit`) in pixels of `object`'s image - an error for a
/// length in nm when the image has no pixel size.
pub(crate) fn to_px(
    value: f32,
    unit: SizeUnits,
    object: &StoredObject,
    step: &str,
) -> Result<f64, InternalErrors> {
    match unit {
        SizeUnits::Pixels => Ok(value as f64),
        SizeUnits::NanoMeter => match object.pixel_size_nm {
            Some(nm_per_px) if nm_per_px > 0.0 => Ok(value as f64 / nm_per_px),
            _ => Err(InternalErrors::InvalidArgument(format!(
                "{step}: {} has no pixel size - give the distances in pixels",
                object.image_rel_path
            ))),
        },
    }
}

/// Tracks the objects of `class`, image by image - so only one image's
/// objects are in memory (and every object's new track, written at the
/// end). `linker` makes the linker of a stack from its
/// first object (for the image's pixel size). Track ids continue after
/// the other classes' tracks.
pub(crate) fn run<L: Linker>(
    ctx: &mut DatasetContext,
    class: ObjectClass,
    with_masks: bool,
    options: &GapOptions,
    linker: impl Fn(&StoredObject) -> Result<L, InternalErrors>,
) -> Result<(), InternalErrors> {
    let mut next_id = ctx.store.max_track_id_except(class)? + 1;
    let mut assignments = Vec::new();
    for image in ctx.store.image_rel_paths(class)? {
        if ctx.is_cancelled() {
            return Err(InternalErrors::Cancelled);
        }
        let objects = ctx.store.get_objects(&ObjectFilter {
            image_rel_path: Some(image),
            object_class: Some(class),
            with_masks,
            ..Default::default()
        })?;
        assignments.reserve(objects.len());
        for frames in stacks(&objects).values() {
            let Some(&first) = frames.values().next().and_then(|f| f.first()) else {
                continue;
            };
            let linker = linker(&objects[first])?;
            for linked in track(&linker, &objects, frames, options, &mut next_id) {
                assignments.push(TrackAssignment {
                    object_id: objects[linked.object].id.clone(),
                    track_id: linked.track_id,
                    parent_track_ids: linked.parent_track_ids,
                });
            }
        }
    }
    // One write: an update per image would rewrite the table every time.
    ctx.store.set_track_ids(&assignments)
}
