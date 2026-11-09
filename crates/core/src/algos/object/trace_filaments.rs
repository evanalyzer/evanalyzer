//! # trace_filaments
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Traces each neuron as a tree: from its soma along the neurites to their
//! tips (the end points of Skeletonize Objects), as the cheapest paths
//! through the image - bridging the dim stretches a threshold breaks, and
//! giving each tip to the soma that reaches it most cheaply (touching
//! neurons are told apart). The tree is split at its branch points into
//! segments, each a 1-pixel-wide object linked to the next segment towards
//! the soma, and all to their soma.
//!
//! Per soma, in a window of its box grown by the maximum reach:
//! 1. **Cost:** per pixel the price of crossing it - low along fibres. With
//!    "Follow fibres" from the channel's vesselness (Frangi, the widths
//!    given) - the cost tracers such as NeuronJ (Meijering et al. 2004)
//!    and SNT (Longair et al. 2011) use - else from its brightness.
//! 2. **Shortest paths** (Dijkstra 1959) from every soma in the window at
//!    once (its pixels the start): each pixel learns its cheapest way back
//!    and the soma at its end.
//! 3. **The tree:** from each of this soma's tips back along that way;
//!    paths that meet go on as one.
//! 4. **Pruning:** a branch shorter than the minimum branch length is
//!    removed, the branch point it left is no more one (its two segments
//!    become one); repeated until nothing changes.
//! 5. **Segments:** the tree between its branch points, root segments
//!    starting at the soma.
//!
//! Sets its segments' parents itself (the next segment, the soma) - the
//! one exception to "links come from Relate Objects" (see LIMITATIONS).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use bitvec::prelude::{BitVec, Lsb0};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, ImageTile, InternalErrors, ObjectClass, ObjectId, SizeUnits,
};
use kornia_image::ImageSize;
use macros::CommandsMeta;
use rayon::prelude::*;

use crate::algos::filters::vesselness::{sigmas_for_widths, vesselness_into};
use crate::algos::object::skeletonize_objects::object_from_pixels;
use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::object::Object;
use crate::pipeline::pipeline_cache::{
    CacheAddress, GlobalPipelineCache, channel_pixel_slice, sample_channel_pixel,
};
use crate::pipeline::pipeline_context::PipelineContext;

/// Traces each neuron as a tree from its soma to its tips (the end points
/// of Skeletonize Objects) along the cheapest paths - bridging dim gaps and
/// giving each tip to the soma reaching it most cheaply - and splits it
/// into segments at its branch points.
///
/// [Skeletonize Objects (end points)] -> [Trace Filaments] -> [Measure Filament Tree]
///
/// Each segment is a 1-pixel-wide object of the output class, linked to the
/// next segment towards the soma (a root segment: none) and to its soma.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", display_name = "Trace Filaments")]
pub struct TraceFilaments {
    /// The somata: each the root of one tree.
    #[cmdsmeta(summary = true)]
    pub seeds: ObjectClass,

    /// The tips to reach (e.g. the end points of Skeletonize Objects).
    #[cmdsmeta(summary = true)]
    pub targets: ObjectClass,

    /// The class the segments get.
    #[cmdsmeta(summary = true)]
    pub output_class: ObjectClass,

    /// The channel the neurites are in (their stain).
    pub channel: ImageChannelIdx,

    /// Follow tube-shaped structures (vesselness of the channel at the
    /// fibre widths) - else the brightness alone, which also follows blobs
    /// and debris.
    #[cmdsmeta(default = true)]
    pub follow_fibres: bool,

    /// The thinnest fibres to follow.
    #[cmdsmeta(default = 2.0, min = 0.5, max = 1000.0, step = 0.5)]
    pub min_width: f32,

    /// The widest fibres to follow.
    #[cmdsmeta(default = 6.0, min = 0.5, max = 1000.0, step = 0.5)]
    pub max_width: f32,

    /// How far from its soma a tip may be: the search window (soma box grown
    /// by this) - its RAM grows with its area (about 25 bytes per pixel,
    /// a window per thread).
    #[cmdsmeta(
        default = 300.0,
        min = 1.0,
        max = 100000.0,
        step = 10.0,
        summary = true
    )]
    pub max_reach: f32,

    /// Side branches shorter than this are removed.
    #[cmdsmeta(default = 5.0, min = 0.0, max = 100000.0, step = 1.0)]
    pub min_branch_length: f32,

    /// Unit of the widths, the reach and the branch length.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub unit: SizeUnits,
}

impl ImageAlgorithm for TraceFilaments {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.seeds == ObjectClass::Unset
            || self.targets == ObjectClass::Unset
            || self.output_class == ObjectClass::Unset
        {
            return Err(InternalErrors::InvalidArgument(
                "Trace filaments: choose the somata, the tips and the output class".into(),
            ));
        }
        let channel = ChannelWindow::of(cache, self.channel.0 as i32)?;
        let scope = ctx.scope;
        let px = self.to_px(ctx);
        let settings = Settings {
            sigmas: self
                .follow_fibres
                .then(|| sigmas_for_widths(px(self.min_width), px(self.max_width), 3)),
            reach: px(self.max_reach).max(1.0).ceil() as usize,
            min_branch: px(self.min_branch_length).max(0.0) as f64,
        };
        let mut somata: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.seeds, scope))
            .collect();
        // In image order: the same in every run.
        somata.sort_by(|a, b| a.position_cmp(b));
        let tips: Vec<(u32, u32)> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.targets, scope))
            .map(|o| {
                let (cx, cy) = o.get_centroid();
                (cx.round().max(0.0) as u32, cy.round().max(0.0) as u32)
            })
            .collect();
        let full = ctx.full_image_size();

        // Each soma in its own window: in parallel (the windows only read;
        // the result doesn't depend on the order they finish in).
        let traced: Vec<(usize, Vec<Segment>)> = (0..somata.len())
            .into_par_iter()
            .map(|index| {
                trace_one(index, &somata, &tips, &channel, cache, &settings)
                    .map(|segments| (index, segments))
            })
            .collect::<Result<_, _>>()?;

        // The segments as objects: parents the next segment and the soma.
        let mut new_objects = Vec::new();
        for (index, segments) in traced {
            let soma = somata[index];
            let mut objects: Vec<Object> = segments
                .iter()
                .map(|segment| object_from_pixels(&segment.pixels, soma, full))
                .collect();
            let ids: Vec<ObjectId> = objects.iter().map(|o| o.id.clone()).collect();
            for (object, segment) in objects.iter_mut().zip(&segments) {
                object.intensities = object.measure_intensities(cache);
                object.add_object_class(self.output_class, scope)?;
                object.set_parent(self.seeds, Some(soma.id.clone()));
                if let Some(parent) = segment.parent {
                    object.set_parent(self.output_class, Some(ids[parent].clone()));
                }
            }
            new_objects.extend(objects);
        }
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Trace Filaments"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut papers = vec![&DIJKSTRA_1959, &MEIJERING_2004, &LONGAIR_2011];
        if self.follow_fibres {
            papers.push(&FRANGI_1998);
        }
        papers
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

static DIJKSTRA_1959: CitationMetadata = CitationMetadata {
    cite_key: "dijkstra1959note",
    title: "A Note on Two Problems in Connexion with Graphs",
    authors: &["Edsger W. Dijkstra"],
    year: 1959,
    container: Some("Numerische Mathematik"),
    doi: Some("10.1007/BF01386390"),
    url: Some("https://doi.org/10.1007/BF01386390"),
    pages: Some("269-271"),
};

static MEIJERING_2004: CitationMetadata = CitationMetadata {
    cite_key: "meijering2004design",
    title: "Design and Validation of a Tool for Neurite Tracing and Analysis in Fluorescence Microscopy Images",
    authors: &[
        "E. Meijering",
        "M. Jacob",
        "J.-C. F. Sarria",
        "P. Steiner",
        "H. Hirling",
        "M. Unser",
    ],
    year: 2004,
    container: Some("Cytometry Part A"),
    doi: Some("10.1002/cyto.a.20022"),
    url: Some("https://doi.org/10.1002/cyto.a.20022"),
    pages: Some("167-176"),
};

static LONGAIR_2011: CitationMetadata = CitationMetadata {
    cite_key: "longair2011simple",
    title: "Simple Neurite Tracer: Open Source Software for Reconstruction, Visualization and Analysis of Neuronal Processes",
    authors: &["Mark H. Longair", "Dean A. Baker", "J. Douglas Armstrong"],
    year: 2011,
    container: Some("Bioinformatics"),
    doi: Some("10.1093/bioinformatics/btr390"),
    url: Some("https://doi.org/10.1093/bioinformatics/btr390"),
    pages: Some("2453-2454"),
};

static FRANGI_1998: CitationMetadata = CitationMetadata {
    cite_key: "frangi1998multiscale",
    title: "Multiscale Vessel Enhancement Filtering",
    authors: &[
        "Alejandro F. Frangi",
        "Wiro J. Niessen",
        "Koen L. Vincken",
        "Max A. Viergever",
    ],
    year: 1998,
    container: Some("Medical Image Computing and Computer-Assisted Intervention (MICCAI'98)"),
    doi: Some("10.1007/BFb0056195"),
    url: Some("https://doi.org/10.1007/BFb0056195"),
    pages: Some("130-137"),
};

impl TraceFilaments {
    /// A length in this command's unit to pixels; without a pixel size
    /// nanometres are taken as pixels.
    fn to_px(&self, ctx: &PipelineContext) -> impl Fn(f32) -> f32 + use<> {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        let nm = self.unit == SizeUnits::NanoMeter && pixel_size_nm > 0.0;
        move |v: f32| if nm { v / pixel_size_nm } else { v }
    }
}

/// What a trace needs, in pixels.
struct Settings {
    /// The vesselness scales ("Follow fibres"), `None`: brightness.
    sigmas: Option<Vec<f32>>,
    reach: usize,
    min_branch: f64,
}

/// `[x0, y0, x1, y1)` in image pixels.
type Rect = [usize; 4];

/// A channel's tiles in the cache and the part of the image they cover -
/// read window by window, one tile at a time.
struct ChannelWindow {
    channel: i32,
    tiles: Vec<ImageTile>,
    extent: Rect,
}

impl ChannelWindow {
    fn of(cache: &GlobalPipelineCache, channel: i32) -> Result<Self, InternalErrors> {
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(Self::no_channel(channel));
        }
        let extent = [
            tiles.iter().map(|t| t.offset_x).min().unwrap_or(0),
            tiles.iter().map(|t| t.offset_y).min().unwrap_or(0),
            tiles
                .iter()
                .map(|t| t.offset_x + t.width)
                .max()
                .unwrap_or(0),
            tiles
                .iter()
                .map(|t| t.offset_y + t.height)
                .max()
                .unwrap_or(0),
        ];
        Ok(Self {
            channel,
            tiles,
            extent,
        })
    }

    fn no_channel(channel: i32) -> InternalErrors {
        InternalErrors::InvalidArgument(format!(
            "Trace filaments: the image has no channel {channel}"
        ))
    }

    /// The normalized (0..1) values of `rect`, row by row (NaN where no
    /// tile was read or the pixel isn't a number).
    fn read(&self, cache: &GlobalPipelineCache, rect: Rect) -> Result<Vec<f32>, InternalErrors> {
        let [x0, y0, x1, y1] = rect;
        let width = x1 - x0;
        let mut values = vec![f32::NAN; width * (y1 - y0)];
        for tile in &self.tiles {
            let (tx0, ty0) = (tile.offset_x.max(x0), tile.offset_y.max(y0));
            let (tx1, ty1) = (
                (tile.offset_x + tile.width).min(x1),
                (tile.offset_y + tile.height).min(y1),
            );
            if tx0 >= tx1 || ty0 >= ty1 {
                continue;
            }
            let container = cache
                .image_cache
                .get(&CacheAddress::Channel((self.channel, *tile)))
                .ok_or_else(|| Self::no_channel(self.channel))?;
            let rgb = matches!(*container, crate::image::ImageContainer::F32Rgb(_));
            let slice = channel_pixel_slice(&container).unwrap_or_default();
            let len = if rgb { slice.len() / 3 } else { slice.len() };
            for y in ty0..ty1 {
                for x in tx0..tx1 {
                    let sample = (y - tile.offset_y) * tile.width + (x - tile.offset_x);
                    if sample >= len {
                        break;
                    }
                    values[(y - y0) * width + (x - x0)] = sample_channel_pixel(rgb, slice, sample);
                }
            }
        }
        Ok(values)
    }
}

/// A segment of a traced tree: its pixels (image coordinates, from the soma
/// side outwards) and the segment it continues (`None`: it starts at the
/// soma).
#[derive(Debug, Clone, PartialEq)]
struct Segment {
    pixels: Vec<(u32, u32)>,
    parent: Option<usize>,
}

/// The 8 neighbours (dx, dy).
const STEPS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// A pixel's cost (the price of crossing it) from its fibre-ness or
/// brightness `v` (0..1, the window's highest 1): 1 on the strongest, up to
/// 100 on nothing.
fn pixel_cost(v: f32) -> f32 {
    if v.is_finite() {
        1.0 / (v.clamp(0.0, 1.0) + 0.01)
    } else {
        f32::INFINITY
    }
}

/// A heap entry: the smallest distance first, the lower index on a tie.
#[derive(PartialEq)]
struct Reached(f32, usize);

impl Eq for Reached {}

impl Ord for Reached {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Reached {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The tree of `somata[index]`: steps 1-5 in its window.
fn trace_one(
    index: usize,
    somata: &[&Object],
    tips: &[(u32, u32)],
    channel: &ChannelWindow,
    cache: &GlobalPipelineCache,
    settings: &Settings,
) -> Result<Vec<Segment>, InternalErrors> {
    let soma = somata[index];
    let [bx0, by0, bx1, by1] = soma.bbox.map(|v| v as usize);
    let extent = channel.extent;
    let window: Rect = [
        bx0.saturating_sub(settings.reach).max(extent[0]),
        by0.saturating_sub(settings.reach).max(extent[1]),
        (bx1 + 1 + settings.reach).min(extent[2]),
        (by1 + 1 + settings.reach).min(extent[3]),
    ];
    let [wx0, wy0, wx1, wy1] = window;
    if wx1 <= wx0 || wy1 <= wy0 {
        return Ok(Vec::new());
    }
    let (width, height) = (wx1 - wx0, wy1 - wy0);
    let inside = |x: u32, y: u32| {
        (x as usize) >= wx0 && (x as usize) < wx1 && (y as usize) >= wy0 && (y as usize) < wy1
    };
    let at = |x: u32, y: u32| (y as usize - wy0) * width + (x as usize - wx0);

    // 1. Cost per pixel.
    let values = channel.read(cache, window)?;
    let fibre: Vec<f32> = match &settings.sigmas {
        Some(sigmas) => {
            let filled: Vec<f32> = values
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 })
                .collect();
            let mut out = vec![0.0f32; filled.len()];
            vesselness_into(&filled, width, height, sigmas, 0.5, 0.0, true, &mut out);
            out
        }
        None => values.clone(),
    };
    let top = fibre
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);
    let cost: Vec<f32> = fibre
        .iter()
        .zip(&values)
        .map(|(f, v)| {
            if !v.is_finite() {
                f32::INFINITY
            } else {
                pixel_cost(if top > 0.0 { f / top } else { 0.0 })
            }
        })
        .collect();
    drop(fibre);
    drop(values);

    // 2. Shortest paths from every soma in the window (label = its index).
    let mut distance = vec![f32::INFINITY; width * height];
    let mut label = vec![u32::MAX; width * height];
    // The step back towards the soma (index into STEPS), 8: a soma pixel.
    let mut back = vec![u8::MAX; width * height];
    let mut heap = BinaryHeap::new();
    for (i, other) in somata.iter().enumerate() {
        let [ox0, oy0, ox1, oy1] = other.bbox;
        if (ox1 as usize) < wx0
            || (ox0 as usize) >= wx1
            || (oy1 as usize) < wy0
            || (oy0 as usize) >= wy1
        {
            continue;
        }
        let w = (ox1 - ox0 + 1) as usize;
        for bit in other.mask_data.iter_ones() {
            let (x, y) = (ox0 + (bit % w) as u32, oy0 + (bit / w) as u32);
            if !inside(x, y) {
                continue;
            }
            let p = at(x, y);
            // Overlapping somata: the first in image order keeps the pixel.
            if distance[p] == 0.0 {
                continue;
            }
            distance[p] = 0.0;
            label[p] = i as u32;
            back[p] = 8;
            heap.push(Reached(0.0, p));
        }
    }
    while let Some(Reached(d, p)) = heap.pop() {
        if d > distance[p] {
            continue;
        }
        let (x, y) = ((p % width) as isize, (p / width) as isize);
        for (k, (dx, dy)) in STEPS.iter().enumerate() {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let q = ny as usize * width + nx as usize;
            let step = if *dx != 0 && *dy != 0 {
                std::f32::consts::SQRT_2
            } else {
                1.0
            };
            let nd = d + step * (cost[p] + cost[q]) / 2.0;
            if nd < distance[q] {
                distance[q] = nd;
                label[q] = label[p];
                // Back: the opposite step.
                back[q] = (7 - k) as u8;
                heap.push(Reached(nd, q));
            }
        }
    }
    drop(cost);

    // 3. The tree: from this soma's tips back to it.
    let own = index as u32;
    let mut on_tree: BitVec<u64, Lsb0> = BitVec::repeat(false, width * height);
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    let mut roots: Vec<usize> = Vec::new();
    let mut tip_pixels: Vec<usize> = tips
        .iter()
        .filter(|&&(x, y)| inside(x, y))
        .map(|&(x, y)| at(x, y))
        .filter(|&p| label[p] == own && distance[p].is_finite() && back[p] != 8)
        .collect();
    tip_pixels.sort_unstable();
    tip_pixels.dedup();
    for &tip in &tip_pixels {
        let mut p = tip;
        loop {
            if on_tree[p] {
                break;
            }
            on_tree.set(p, true);
            if back[p] == 8 {
                roots.push(p);
                break;
            }
            let (dx, dy) = STEPS[back[p] as usize];
            let (x, y) = ((p % width) as isize + dx, (p / width) as isize + dy);
            let q = y as usize * width + x as usize;
            parent_of.insert(p, q);
            p = q;
        }
    }
    if roots.is_empty() {
        return Ok(Vec::new());
    }
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tree_pixels: Vec<usize> = parent_of.keys().copied().collect();
    tree_pixels.sort_unstable();
    for &p in &tree_pixels {
        children.entry(parent_of[&p]).or_default().push(p);
    }
    for list in children.values_mut() {
        list.sort_unstable();
    }
    roots.sort_unstable();
    roots.dedup();

    // 5. (before 4: pruning works on segments) Segments between branch
    // points, from each root outwards.
    let mut segments: Vec<(Vec<usize>, Option<usize>)> = Vec::new();
    let mut stack: Vec<(usize, Option<usize>)> = roots.iter().map(|&r| (r, None)).collect();
    stack.reverse();
    while let Some((start, parent)) = stack.pop() {
        let mut pixels = vec![start];
        let mut p = start;
        loop {
            let next = children.get(&p).map(Vec::as_slice).unwrap_or(&[]);
            if next.len() == 1 {
                p = next[0];
                pixels.push(p);
                continue;
            }
            let me = segments.len();
            segments.push((pixels, parent));
            for &child in next.iter().rev() {
                stack.push((child, Some(me)));
            }
            break;
        }
    }

    // 4. Pruning: short leaf branches go, a branch point left with one
    // branch joins it to its parent.
    let to_xy = |p: usize| ((wx0 + p % width) as u32, (wy0 + p / width) as u32);
    let length = |pixels: &[usize]| -> f64 {
        pixels
            .windows(2)
            .map(|w| {
                let (a, b) = (to_xy(w[0]), to_xy(w[1]));
                if a.0 != b.0 && a.1 != b.1 {
                    std::f64::consts::SQRT_2
                } else {
                    1.0
                }
            })
            .sum()
    };
    let mut alive: Vec<bool> = vec![true; segments.len()];
    loop {
        let child_count = |i: usize, alive: &[bool], segments: &[(Vec<usize>, Option<usize>)]| {
            segments
                .iter()
                .enumerate()
                .filter(|(j, s)| alive[*j] && s.1 == Some(i))
                .count()
        };
        // A leaf branch of a branch point shorter than the minimum.
        let short = (0..segments.len()).find(|&i| {
            alive[i]
                && segments[i].1.is_some()
                && child_count(i, &alive, &segments) == 0
                && length(&segments[i].0) < settings.min_branch
        });
        if let Some(i) = short {
            alive[i] = false;
            continue;
        }
        // A segment left with one child: they become one.
        let single =
            (0..segments.len()).find(|&i| alive[i] && child_count(i, &alive, &segments) == 1);
        let Some(i) = single else {
            break;
        };
        let child = (0..segments.len())
            .find(|&j| alive[j] && segments[j].1 == Some(i))
            .unwrap();
        let tail = std::mem::take(&mut segments[child].0);
        segments[i].0.extend(tail);
        alive[child] = false;
        for s in segments.iter_mut() {
            if s.1 == Some(child) {
                s.1 = Some(i);
            }
        }
    }

    // Renumbered without the removed ones.
    let mut new_index = vec![usize::MAX; segments.len()];
    let mut kept = Vec::new();
    for (i, (pixels, parent)) in segments.into_iter().enumerate() {
        if !alive[i] {
            continue;
        }
        new_index[i] = kept.len();
        kept.push((pixels, parent));
    }
    Ok(kept
        .into_iter()
        .map(|(pixels, parent)| Segment {
            pixels: pixels.into_iter().map(to_xy).collect(),
            parent: parent.map(|p| new_index[p]),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ManagedImage, Point2d};
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use kornia_image::Image;
    use std::sync::Arc;

    const SIDE: usize = 120;
    const SOMA: ObjectClass = ObjectClass::Global(1);
    const TIP: ObjectClass = ObjectClass::Global(2);
    const SEGMENT: ObjectClass = ObjectClass::Global(3);

    /// A disk-shaped soma of radius `r` at (cx, cy).
    fn soma(id: u128, cx: u32, cy: u32, r: u32) -> Object {
        let (x0, y0) = (cx - r, cy - r);
        let side = 2 * r + 1;
        let mut mask = BitVec::<u64, Lsb0>::repeat(false, (side * side) as usize);
        let (mut area, mut sx, mut sy) = (0, 0u64, 0u64);
        for y in 0..side {
            for x in 0..side {
                let (dx, dy) = (x as i64 - r as i64, y as i64 - r as i64);
                if dx * dx + dy * dy <= (r * r) as i64 {
                    mask.set((y * side + x) as usize, true);
                    area += 1;
                    sx += (x0 + x) as u64;
                    sy += (y0 + y) as u64;
                }
            }
        }
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x0, y0, x0 + side - 1, y0 + side - 1],
            area,
            mask_data: mask,
            sum_x: sx,
            sum_y: sy,
            ..Default::default()
        });
        object.add_object_class(SOMA, Scope::Global).unwrap();
        object
    }

    fn tip(id: u128, x: u32, y: u32) -> Object {
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x, y, x, y],
            area: 1,
            mask_data: BitVec::repeat(true, 1),
            sum_x: x as u64,
            sum_y: y as u64,
            ..Default::default()
        });
        object.add_object_class(TIP, Scope::Global).unwrap();
        object
    }

    /// A cache with channel 0 = `pixel(x, y)` (one tile) and `objects`.
    fn cache_with(
        pixel: impl Fn(usize, usize) -> f32,
        objects: Vec<Object>,
    ) -> GlobalPipelineCache {
        let size = ImageSize {
            width: SIDE,
            height: SIDE,
        };
        let data: Vec<f32> = (0..SIDE * SIDE)
            .map(|p| pixel(p % SIDE, p / SIDE))
            .collect();
        let image = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::<f32, 1>::new(size, data).unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: Default::default(),
        }));
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            image,
            0,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: SIDE,
                height: SIDE,
            },
        );
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn command() -> TraceFilaments {
        TraceFilaments {
            seeds: SOMA,
            targets: TIP,
            output_class: SEGMENT,
            channel: ImageChannelIdx(0),
            follow_fibres: true,
            min_width: 2.0,
            max_width: 4.0,
            max_reach: 200.0,
            min_branch_length: 5.0,
            unit: SizeUnits::Pixels,
        }
    }

    fn run(command: &TraceFilaments, cache: &mut GlobalPipelineCache) -> Vec<Object> {
        let meta = PipelineImageMeta {
            full_image_width: ImageSize {
                width: SIDE,
                height: SIDE,
            },
            ..Default::default()
        };
        let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
        command.execute(&mut ctx, cache).unwrap();
        let mut segments: Vec<Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&SEGMENT, Scope::Global))
            .cloned()
            .collect();
        segments.sort_by(|a, b| a.position_cmp(b));
        segments
    }

    /// A line 3 px wide from (x0, y) to (x1, y), brightness `v`.
    fn hline(x: usize, y: usize, (x0, x1, yy): (usize, usize, usize), v: f32) -> f32 {
        if (x0..=x1).contains(&x) && y.abs_diff(yy) <= 1 {
            v
        } else {
            0.0
        }
    }

    fn vline(x: usize, y: usize, (xx, y0, y1): (usize, usize, usize), v: f32) -> f32 {
        if (y0..=y1).contains(&y) && x.abs_diff(xx) <= 1 {
            v
        } else {
            0.0
        }
    }

    #[test]
    fn a_neurite_with_a_dim_gap_is_traced_through_it() {
        // Soma at (20, 60), a neurite to (100, 60), dim (0.05) from 55 to 65.
        let pixel = |x, y| {
            let v = if (55..=65).contains(&x) { 0.05 } else { 0.8 };
            hline(x, y, (20, 100, 60), v).max(if x.abs_diff(20) + y.abs_diff(60) < 6 {
                0.9
            } else {
                0.0
            })
        };
        let mut cache = cache_with(pixel, vec![soma(1, 20, 60, 5), tip(2, 100, 60)]);
        let segments = run(&command(), &mut cache);
        assert_eq!(segments.len(), 1);
        let s = &segments[0];
        // From the soma's edge to the tip, along the line.
        assert!(s.is_part_of(100, 60));
        assert!(s.is_part_of(60, 60) || s.is_part_of(60, 59) || s.is_part_of(60, 61));
        assert_eq!(s.parent(&SOMA), Some(&ObjectId(1)));
        assert_eq!(s.parent(&SEGMENT), None);
    }

    /// A trunk from the soma to (60, 60), splitting into two branches to
    /// (100, 30) and (100, 90): a root segment and two children.
    #[test]
    fn a_branching_neurite_gives_a_tree_of_segments() {
        let pixel = |x: usize, y: usize| {
            let trunk = hline(x, y, (20, 60, 60), 0.8);
            let up = vline(x, y, (60, 30, 60), 0.8).max(hline(x, y, (60, 100, 30), 0.8));
            let down = vline(x, y, (60, 60, 90), 0.8).max(hline(x, y, (60, 100, 90), 0.8));
            trunk.max(up).max(down)
        };
        let mut cache = cache_with(
            pixel,
            vec![soma(1, 20, 60, 5), tip(2, 100, 30), tip(3, 100, 90)],
        );
        let segments = run(&command(), &mut cache);
        assert_eq!(segments.len(), 3, "{segments:?}");
        let roots: Vec<&Object> = segments
            .iter()
            .filter(|s| s.parent(&SEGMENT).is_none())
            .collect();
        assert_eq!(roots.len(), 1);
        let root = roots[0].id.clone();
        let children = segments
            .iter()
            .filter(|s| s.parent(&SEGMENT) == Some(&root))
            .count();
        assert_eq!(children, 2);
        assert!(
            segments
                .iter()
                .all(|s| s.parent(&SOMA) == Some(&ObjectId(1)))
        );
    }

    /// Two somata on one line: each tip goes to the soma nearer along it.
    #[test]
    fn touching_neurons_get_their_own_tips() {
        let pixel = |x: usize, y: usize| hline(x, y, (10, 110, 60), 0.8);
        let mut cache = cache_with(
            pixel,
            vec![
                soma(1, 30, 60, 4),
                soma(2, 90, 60, 4),
                tip(3, 10, 60),
                tip(4, 110, 60),
            ],
        );
        let segments = run(&command(), &mut cache);
        assert_eq!(segments.len(), 2);
        let of = |x: u32| {
            segments
                .iter()
                .find(|s| s.is_part_of(x, 60))
                .and_then(|s| s.parent(&SOMA).cloned())
        };
        assert_eq!(of(10), Some(ObjectId(1)));
        assert_eq!(of(110), Some(ObjectId(2)));
    }

    /// A short side branch is pruned: the trunk stays one segment.
    #[test]
    fn a_short_side_branch_is_pruned() {
        let pixel = |x: usize, y: usize| {
            hline(x, y, (20, 100, 60), 0.8).max(vline(x, y, (60, 60, 63), 0.8))
        };
        let mut cache = cache_with(
            pixel,
            vec![soma(1, 20, 60, 5), tip(2, 100, 60), tip(3, 60, 63)],
        );
        let segments = run(&command(), &mut cache);
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert!(segments[0].is_part_of(100, 60));
    }

    #[test]
    fn a_tip_beyond_the_reach_is_left_out() {
        let pixel = |x: usize, y: usize| hline(x, y, (20, 100, 60), 0.8);
        let mut cache = cache_with(pixel, vec![soma(1, 20, 60, 5), tip(2, 100, 60)]);
        let command = TraceFilaments {
            max_reach: 30.0,
            ..command()
        };
        assert!(run(&command, &mut cache).is_empty());
    }

    #[test]
    fn brightness_works_without_following_fibres() {
        let pixel = |x: usize, y: usize| hline(x, y, (20, 100, 60), 0.8);
        let mut cache = cache_with(pixel, vec![soma(1, 20, 60, 5), tip(2, 100, 60)]);
        let command = TraceFilaments {
            follow_fibres: false,
            ..command()
        };
        let segments = run(&command, &mut cache);
        assert_eq!(segments.len(), 1);
        // On the line all the way.
        let s = &segments[0];
        for x in (30..=100).step_by(10) {
            assert!((59..=61).any(|y| s.is_part_of(x, y)), "x {x} off the line");
        }
    }

    #[test]
    fn the_same_trace_every_run() {
        let pixel = |x: usize, y: usize| {
            hline(x, y, (20, 100, 60), 0.8).max(vline(x, y, (60, 20, 60), 0.6))
        };
        let objects = || vec![soma(1, 20, 60, 5), tip(2, 100, 60), tip(3, 60, 20)];
        let pixels = |segments: Vec<Object>| -> Vec<(Vec<u32>, usize)> {
            segments
                .into_iter()
                .map(|s| (s.bbox.to_vec(), s.area))
                .collect()
        };
        let a = pixels(run(&command(), &mut cache_with(pixel, objects())));
        let b = pixels(run(&command(), &mut cache_with(pixel, objects())));
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
    }
}
