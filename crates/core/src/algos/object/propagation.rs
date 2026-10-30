//! # Propagation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Grows one region per seed object (typically a nucleus) through a channel
//! (typically a cell or membrane stain): every pixel goes to the seed it can
//! be reached from at the lowest cost, where a step costs more the more the
//! stain's brightness changes along it - so the border between two cells
//! falls on the dim gap between them, not halfway between the nuclei.
//! `regularization` (λ) weighs plain distance against the brightness change:
//! a large λ approaches a distance-only split, 0 one by brightness only.
//! Only pixels inside the optional mask are assigned.
//!
//! The growth follows `centrosome/_propagate.pyx`: a fixed step cost, 8
//! neighbours and a fixed order of equally cheap pixels, so the same image
//! always gives the same regions.
//!
//! Portions ported from centrosome, Copyright © 2003 - 2019 Broad Institute,
//! Inc., BSD-3-Clause - see the third-party notices (About > Licenses).

use crate::ImageDimensionality;
use crate::algos::libs::row_runs::{Run, RunRegion, push_row_runs, regions_from_runs};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::{Object, ObjectInit};
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, ImageTile, InternalErrors, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Grows a region around each seed object through a channel.
///
/// E.g. a cell around its nucleus through the cell stain: every pixel goes
/// to the seed it is reached from at the lowest cost, a step costing more
/// the more the stain's brightness changes - so neighbouring cells meet
/// where the stain is dim, not halfway between their nuclei. Each region
/// keeps its seed's pixels (Relate Objects links it to its seed).
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object")]
pub struct Propagation {
    /// Objects carrying this class are the seeds (e.g. nuclei): each grows
    /// into one region.
    #[cmdsmeta(summary = true)]
    pub seeds: ObjectClass,

    /// The channel the regions grow through (e.g. the cell or membrane
    /// stain).
    #[cmdsmeta(summary = true)]
    pub growth_channel: ImageChannelIdx,

    /// Regions only cover pixels of objects carrying this class (e.g. the
    /// thresholded cell stain). `Unset`: the whole image.
    pub mask: ObjectClass,

    /// How much plain distance counts against the change in brightness
    /// (λ): larger values split more by distance, `0` by brightness only.
    #[cmdsmeta(default = 0.05, min = 0.0, step = 0.01)]
    pub regularization: f32,

    /// The class the grown regions get.
    pub output_class: ObjectClass,

    /// Discard regions that touch the image border.
    pub exclude_at_edges: bool,
}

impl ImageAlgorithm for Propagation {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.seeds == ObjectClass::Unset || self.output_class == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Propagation: choose the seeds' class and the output class".into(),
            ));
        }
        let channel = GrowthChannel::of(cache, self.growth_channel.0 as i32)?;
        let mut seeds: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.seeds, ctx.scope))
            .collect();
        if seeds.is_empty() {
            return Ok(());
        }
        // In image order (`Object::position_cmp`): equally cheap pixels go to
        // the lower label, the seed first in reading order - the same in every
        // run, not the order the tiles finished in.
        seeds.sort_by(|a, b| a.position_cmp(b));
        let masks: Option<Vec<&Object>> = (self.mask != ObjectClass::Unset).then(|| {
            cache
                .object_cache
                .values()
                .filter(|o| o.is_of(&self.mask, ctx.scope))
                .collect()
        });

        // Label `i + 1` is `seeds[i]`.
        let growth = Growth {
            channel: &channel,
            cache,
            seeds: ObjectsGrid::new(&seeds),
            masks: masks.as_deref().map(ObjectsGrid::new),
            weight: self.regularization as f64,
        };
        let full = ctx.full_image_size();
        // A small image in one piece: one window over all of it, no edge to
        // bound.
        let extent = channel.extent;
        let block = if rect_area(extent) <= ONE_PIECE_PIXELS {
            (extent[2] - extent[0]).max(extent[3] - extent[1])
        } else {
            BLOCK
        };
        let regions = growth.regions(block, HALO)?;
        let mut new_objects = Vec::new();
        for (seed, region) in seeds.iter().zip(regions) {
            let Some(region) = region else {
                continue;
            };
            // The image's border, not a tile's.
            let [x_min, y_min, x_max, y_max] = region.bbox.map(|v| v as usize);
            let touches_edge =
                x_min == 0 || y_min == 0 || x_max + 1 >= full.width || y_max + 1 >= full.height;
            if self.exclude_at_edges && touches_edge {
                continue;
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId::next(),
                segmentation_class: seed.segmentation_class,
                bbox: region.bbox,
                mask_data: region.mask,
                area: region.area,
                plane: seed.plane,
                touches_edge,
                sum_x: region.sum_x,
                sum_y: region.sum_y,
                sum_x2: region.sum_x2,
                sum_y2: region.sum_y2,
                sum_xy: region.sum_xy,
                ..Default::default()
            });
            object.add_object_class(self.output_class, ctx.scope)?;
            new_objects.push(object);
        }
        // The regions never pass ExtractObjects (which samples the pixels):
        // their intensities are measured here.
        for object in &mut new_objects {
            object.intensities = object.measure_intensities(cache);
        }
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Propagation"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&JONES_2005]
    }

    /// The seeds and channel of the whole image: a region may grow across
    /// tile borders.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

/// Where the method was published.
const JONES_2005: CitationMetadata = CitationMetadata {
    cite_key: "jones2005voronoi",
    title: "Voronoi-Based Segmentation of Cells on Image Manifolds",
    authors: &["Thouis R. Jones", "Anne Carpenter", "Polina Golland"],
    year: 2005,
    container: Some("Computer Vision for Biomedical Image Applications (CVBIA), LNCS 3765"),
    doi: Some("10.1007/11569541_54"),
    url: Some("https://doi.org/10.1007/11569541_54"),
    pages: Some("535-543"),
};

// ## Growing in windows
//
// The regions are grown block by block, each block in a window reaching
// `HALO` pixels past it - never the whole image at once (16 bytes and more
// per pixel: 15 GB for a 1-gigapixel slide). The result is the same as
// growing over the whole image, to the last pixel:
//
// A pixel goes to the seed it is reached from most cheaply (blocking by
// other regions never changes that: the cheapest path to a pixel only
// passes pixels of the same seed). A path from outside the window comes
// from a seed, so it costs at least the cheapest steps from the nearest seed
// to where it comes in, plus the cheapest way on from there - a second
// growth from the window's edge gives this bound `B` for every pixel. A
// pixel the window's seeds reach for less than its `B` has its cheapest path
// inside the window, and so do the pixels on that path: its label and cost -
// and with them the order equally cheap pixels are taken in - are the same
// as in the whole image.
//
// The pixels left (where no seed is near - the background without a mask
// - or seeds outside the window might be as near) are grown together
// afterwards: from the told pixels next to them, which go into the queue
// with their final cost and label, as they come out of it in the whole
// image. Only the pixels left take RAM then: their label and the channel's
// values, about 8 bytes each.
//
// The costs are summed in floating point as before; the bound holds there
// too, as adding a larger or equal number never gives a smaller sum.
//
// An image up to `ONE_PIECE_PIXELS` grows in one window over all of it, as
// before the windows: for it, overlapping windows in parallel would cost
// more RAM and time than they save.

/// The side of a block (pixels). Smaller in the library's tests: a test
/// slide then has blocks inside it, not only at its border (see
/// `tests::whole_image_ram`).
const BLOCK: usize = if cfg!(test) { 256 } else { 1024 };
/// How far a window reaches past its block.
const HALO: usize = if cfg!(test) { 64 } else { 128 };
/// Images up to this many pixels grow in one piece (~16 bytes per pixel,
/// 270 MB at most): the windows only pay off above it - they overlap and
/// run in parallel. Tiny in the library's tests, so their slides use windows.
const ONE_PIECE_PIXELS: usize = if cfg!(test) { 64 * 64 } else { 4096 * 4096 };
/// RAM per pixel of a window: values, label, cost, bound and the queues.
const WINDOW_BYTES_PER_PIXEL: u64 = 40;
/// RAM per pixel of a block with pixels left: values, label, queue.
const LEFT_BYTES_PER_PIXEL: u64 = 12;

/// A rectangle `[x0, y0, x1, y1)` in image pixels.
type Rect = [usize; 4];

fn rect_area([x0, y0, x1, y1]: Rect) -> usize {
    (x1 - x0) * (y1 - y0)
}

/// The growth channel's tiles and the part of the image they cover (the
/// whole image in a run, the visible part in a preview) - no pixels: a
/// window reads its own.
struct GrowthChannel {
    channel: i32,
    tiles: Vec<ImageTile>,
    extent: Rect,
}

impl GrowthChannel {
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
        InternalErrors::InvalidArgument(format!("Propagation: the image has no channel {channel}"))
    }

    /// The normalized (0..1) values of `rect`, row-major (0 where no tile
    /// was read), and which pixels a tile covered with a number. One tile
    /// at a time: none is held after it is copied.
    fn read(
        &self,
        cache: &GlobalPipelineCache,
        rect: Rect,
    ) -> Result<(Vec<f32>, BitVec<u64, Lsb0>), InternalErrors> {
        let [x0, y0, x1, y1] = rect;
        let width = x1 - x0;
        let mut values = vec![0.0f32; rect_area(rect)];
        let mut read = BitVec::<u64, Lsb0>::repeat(false, values.len());
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
                    let v = sample_channel_pixel(rgb, slice, sample);
                    let idx = (y - y0) * width + (x - x0);
                    // A pixel that isn't a number can't be grown into.
                    values[idx] = if v.is_finite() { v } else { 0.0 };
                    read.set(idx, v.is_finite());
                }
            }
        }
        Ok((values, read))
    }
}

/// Objects by the grid cells their bounding boxes cover, to find the ones
/// in a window.
struct ObjectsGrid<'a> {
    objects: &'a [&'a Object],
    cells: HashMap<(u32, u32), Vec<u32>>,
}

/// The side of an `ObjectsGrid` cell (pixels).
const GRID_CELL: u32 = 256;

impl<'a> ObjectsGrid<'a> {
    fn new(objects: &'a [&'a Object]) -> Self {
        let mut cells: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (i, object) in objects.iter().enumerate() {
            let [x0, y0, x1, y1] = object.bbox;
            if x1 < x0 || y1 < y0 {
                continue;
            }
            for cy in y0 / GRID_CELL..=y1 / GRID_CELL {
                for cx in x0 / GRID_CELL..=x1 / GRID_CELL {
                    cells.entry((cx, cy)).or_default().push(i as u32);
                }
            }
        }
        Self { objects, cells }
    }

    /// The indices of the objects whose bounding box meets `rect`, in
    /// order.
    fn in_rect(&self, [x0, y0, x1, y1]: Rect) -> Vec<usize> {
        let mut found = Vec::new();
        if x0 >= x1 || y0 >= y1 {
            return found;
        }
        let cell = GRID_CELL as usize;
        for cy in y0 / cell..=(y1 - 1) / cell {
            for cx in x0 / cell..=(x1 - 1) / cell {
                if let Some(ids) = self.cells.get(&(cx as u32, cy as u32)) {
                    found.extend(ids.iter().map(|&i| i as usize));
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found.retain(|&i| {
            let [ox0, oy0, ox1, oy1] = self.objects[i].bbox.map(|v| v as usize);
            ox0 < x1 && oy0 < y1 && ox1 >= x0 && oy1 >= y0
        });
        found
    }
}

impl ObjectsGrid<'_> {
    /// The least a path from any of the objects to pixel (`x`, `y`) costs if
    /// a straight step costs at least `straight` and a diagonal one
    /// `diagonal`: the cheapest mix of the two from the nearest bounding box
    /// (an octile distance; exact arithmetic, see [`ROUNDING`]). Searched up
    /// to `max_cells` grid cells away: with no object that near, the cost of
    /// that distance.
    fn least_cost_to(
        &self,
        x: usize,
        y: usize,
        straight: f64,
        diagonal: f64,
        max_cells: usize,
    ) -> f64 {
        let cell = GRID_CELL as usize;
        let (cx, cy) = (x / cell, y / cell);
        let cost_to = |bbox: [u32; 4]| {
            let [x0, y0, x1, y1] = bbox.map(|v| v as usize);
            let dx = x0.saturating_sub(x).max(x.saturating_sub(x1));
            let dy = y0.saturating_sub(y).max(y.saturating_sub(y1));
            let (long, short) = (dx.max(dy) as f64, dx.min(dy) as f64);
            (long - short) * straight + short * diagonal
        };
        let mut best = f64::INFINITY;
        for ring in 0..=max_cells {
            // Every cell of this ring and beyond is at least `ring - 1`
            // whole cells away.
            if best <= ring.saturating_sub(1) as f64 * cell as f64 * straight {
                break;
            }
            let (x0, y0) = (cx.saturating_sub(ring), cy.saturating_sub(ring));
            for gy in y0..=cy + ring {
                for gx in x0..=cx + ring {
                    let on_ring = gx.abs_diff(cx) == ring || gy.abs_diff(cy) == ring;
                    if !on_ring {
                        continue;
                    }
                    if let Some(ids) = self.cells.get(&(gx as u32, gy as u32)) {
                        for &i in ids {
                            best = best.min(cost_to(self.objects[i as usize].bbox));
                        }
                    }
                }
            }
        }
        best.min(max_cells as f64 * cell as f64 * straight)
    }
}

/// How much less a sum of step costs in floating point can be than the
/// exact sum: each addition rounds by at most 2^-53 of it, for 4 billion
/// steps (more than an image has pixels) at most 5·10^-7 together.
const ROUNDING: f64 = 1.0 - 1e-6;

/// How far (grid cells) a window's edge looks for the nearest seed.
const NEAREST_SEED_CELLS: usize = 64;

/// Calls `f` with the window index of each of `object`'s pixels inside
/// `window`.
fn for_each_pixel_in(object: &Object, window: Rect, mut f: impl FnMut(usize)) {
    let [x_min, y_min, x_max, y_max] = object.bbox.map(|v| v as usize);
    if x_max < x_min || y_max < y_min {
        return;
    }
    let [wx0, wy0, wx1, wy1] = window;
    let mask_width = x_max - x_min + 1;
    // bbox[2]/[3] are inclusive; the mask is row-major over the bbox.
    for y in y_min.max(wy0)..=y_max.min(wy1.saturating_sub(1)) {
        for x in x_min.max(wx0)..=x_max.min(wx1.saturating_sub(1)) {
            if object
                .mask_data
                .get((y - y_min) * mask_width + (x - x_min))
                .is_some_and(|b| *b)
            {
                f((y - wy0) * (wx1 - wx0) + (x - wx0));
            }
        }
    }
}

/// The inputs of one growth.
struct Growth<'a> {
    channel: &'a GrowthChannel,
    cache: &'a GlobalPipelineCache,
    seeds: ObjectsGrid<'a>,
    masks: Option<ObjectsGrid<'a>>,
    weight: f64,
}

/// What growing one block in its window told.
struct GrownBlock {
    core: Rect,
    /// The rows of the pixels it told.
    runs: Vec<Run>,
    /// The pixels it couldn't tell (`None`: none).
    left: Option<BitVec<u64, Lsb0>>,
    /// Told pixels a growth went on from (in the mask, reached), with their
    /// cost and label, next to pixels left - its own, or another block's
    /// across its border: (image index, cost, label).
    rim: Vec<(u64, f64, u32)>,
}

/// A block with pixels left, for growing them.
struct LeftBlock {
    core: Rect,
    left: BitVec<u64, Lsb0>,
    labels: Vec<u32>,
    /// The channel's values two pixels past the block (as far as a step's
    /// cost from a neighbour looks).
    values: Vec<f32>,
    around: Rect,
}

impl Growth<'_> {
    /// Per seed (label `i + 1`), the region it got (`None`: no pixel).
    /// `block` and `halo`: see [`BLOCK`], [`HALO`].
    fn regions(&self, block: usize, halo: usize) -> Result<Vec<Option<RunRegion>>, InternalErrors> {
        let extent = self.channel.extent;
        let cores: Vec<Rect> = Self::grid(extent, block).collect();
        let window_bytes = rect_area(Self::window(extent, [0, 0, block, block], halo)) as u64
            * WINDOW_BYTES_PER_PIXEL;
        let available = crate::memory::available_memory_bytes();
        // As many windows at once as fit in a quarter of the free RAM.
        let at_once =
            ((available / 4 / window_bytes.max(1)) as usize).clamp(1, rayon::current_num_threads());
        let mut grown: Vec<GrownBlock> = Vec::with_capacity(cores.len());
        for batch in cores.chunks(at_once) {
            grown.extend(
                batch
                    .par_iter()
                    .map(|core| self.grow_block(*core, halo))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let mut runs: Vec<Run> = grown
            .iter_mut()
            .flat_map(|b| std::mem::take(&mut b.runs))
            .collect();
        if grown.iter().any(|b| b.left.is_some()) {
            runs.extend(self.grow_left(grown, block)?);
        }
        Ok(regions_from_runs(runs, self.seeds.objects.len()))
    }

    /// `extent` in blocks of `side` pixels, starting at its corner.
    fn grid(extent: Rect, side: usize) -> impl Iterator<Item = Rect> {
        let [x0, y0, x1, y1] = extent;
        (y0..y1).step_by(side).flat_map(move |y| {
            (x0..x1)
                .step_by(side)
                .map(move |x| [x, y, (x + side).min(x1), (y + side).min(y1)])
        })
    }

    /// `core` grown by `halo` on every side, inside `extent`.
    fn window(extent: Rect, [x0, y0, x1, y1]: Rect, halo: usize) -> Rect {
        [
            x0.saturating_sub(halo).max(extent[0]),
            y0.saturating_sub(halo).max(extent[1]),
            (x1 + halo).min(extent[2]),
            (y1 + halo).min(extent[3]),
        ]
    }

    /// Grows the block `core` in a window `halo` pixels larger.
    fn grow_block(&self, core: Rect, halo: usize) -> Result<GrownBlock, InternalErrors> {
        let extent = self.channel.extent;
        let window = Self::window(extent, core, halo);
        let [wx0, wy0, wx1, wy1] = window;
        let (width, height) = (wx1 - wx0, wy1 - wy0);
        // The values one pixel past the window too: a step's cost looks at
        // the 3×3 neighbourhoods.
        let around = Self::window(extent, window, 1);
        let (values, read) = self.channel.read(self.cache, around)?;
        let frame = Frame {
            values: &values,
            stride: around[2] - around[0],
            offset: (wx0 - around[0], wy0 - around[1]),
            clamp: [
                extent[1] as isize - wy0 as isize,
                extent[0] as isize - wx0 as isize,
                extent[3] as isize - 1 - wy0 as isize,
                extent[2] as isize - 1 - wx0 as isize,
            ],
        };
        let read_at = |idx: usize| {
            let (x, y) = (idx % width, idx / width);
            read[(y + frame.offset.1) * frame.stride + x + frame.offset.0]
        };
        let allowed: BitVec<u64, Lsb0> = match &self.masks {
            None => (0..width * height).map(read_at).collect(),
            Some(masks) => {
                let mut allowed = BitVec::repeat(false, width * height);
                for i in masks.in_rect(window) {
                    for_each_pixel_in(masks.objects[i], window, |idx| {
                        if read_at(idx) {
                            allowed.set(idx, true);
                        }
                    });
                }
                allowed
            }
        };
        drop(read);
        let mut seed_pixels = Vec::new();
        for i in self.seeds.in_rect(window) {
            for_each_pixel_in(self.seeds.objects[i], window, |idx| {
                seed_pixels.push((idx, i as u32 + 1))
            });
        }
        let grown = grow(&frame, width, height, &allowed, &seed_pixels, self.weight);

        // The window's sides inside the image: where a path from outside
        // can come in.
        let open = [
            wy0 > extent[1],
            wx0 > extent[0],
            wy1 < extent[3],
            wx1 < extent[2],
        ];
        let [cx0, cy0, cx1, cy1] = core;
        let core_width = cx1 - cx0;
        let index = |x: usize, y: usize| (y - wy0) * width + (x - wx0);
        let bound = if open.contains(&true) {
            // Only up to the dearest pixel to tell.
            let mut limit = f64::NEG_INFINITY;
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    let idx = index(x, y);
                    if grown.labels[idx] != 0 && grown.costs[idx] > 0.0 {
                        limit = limit.max(grown.costs[idx]);
                    }
                }
            }
            // A path from outside comes from a seed: it costs at least the
            // way from the nearest seed to where it comes in - and a step.
            let (straight, diagonal) = (
                step_cost_without_change(1.0, self.weight),
                step_cost_without_change(2.0, self.weight),
            );
            let mut entries = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let edge = (open[0] && y == 0)
                        || (open[1] && x == 0)
                        || (open[2] && y + 1 == height)
                        || (open[3] && x + 1 == width);
                    if edge && allowed[y * width + x] {
                        let from_seed = self.seeds.least_cost_to(
                            wx0 + x,
                            wy0 + y,
                            straight,
                            diagonal,
                            NEAREST_SEED_CELLS,
                        );
                        entries.push((y * width + x, straight.max(from_seed * ROUNDING)));
                    }
                }
            }
            Some(bound(
                &frame,
                width,
                height,
                &allowed,
                &entries,
                self.weight,
                limit,
            ))
        } else {
            None
        };

        let mut runs = Vec::new();
        let mut left: Option<BitVec<u64, Lsb0>> = None;
        let mut told = BitVec::<u64, Lsb0>::repeat(false, rect_area(core));
        let mut row = Vec::with_capacity(core_width);
        for y in cy0..cy1 {
            row.clear();
            for x in cx0..cx1 {
                let idx = index(x, y);
                let (l, cost) = (grown.labels[idx], grown.costs[idx]);
                // A seed's pixel (cost 0, also outside the mask) and a pixel
                // outside the mask are what they are in every window.
                let is_told = match &bound {
                    None => true,
                    Some(_) if cost == 0.0 && l != 0 => true,
                    Some(_) if !allowed[idx] => true,
                    Some(b) => l != 0 && cost < b[idx],
                };
                let at = (y - cy0) * core_width + (x - cx0);
                if is_told {
                    told.set(at, true);
                } else {
                    left.get_or_insert_with(|| BitVec::repeat(false, rect_area(core)))
                        .set(at, true);
                }
                row.push(if is_told { l } else { 0 });
            }
            push_row_runs(&mut runs, y as u32, cx0 as u32, row.iter().copied());
        }
        // The told pixels a growth went on from next to the pixels left:
        // here, and on the block's border (next to another block's).
        let (ex0, ey0, ew) = (extent[0], extent[1], extent[2] - extent[0]);
        let mut rim = Vec::new();
        for y in cy0..cy1 {
            for x in cx0..cx1 {
                let (idx, at) = (index(x, y), (y - cy0) * core_width + (x - cx0));
                if !told[at] || grown.labels[idx] == 0 || !allowed[idx] {
                    continue;
                }
                let on_border = x == cx0 || y == cy0 || x + 1 == cx1 || y + 1 == cy1;
                let next_to_left = left.as_ref().is_some_and(|left| {
                    NEIGHBOURS.iter().any(|(dy, dx)| {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        nx >= cx0 as isize
                            && ny >= cy0 as isize
                            && nx < cx1 as isize
                            && ny < cy1 as isize
                            && left[(ny as usize - cy0) * core_width + (nx as usize - cx0)]
                    })
                });
                if on_border || next_to_left {
                    let image_index = ((y - ey0) * ew + (x - ex0)) as u64;
                    rim.push((image_index, grown.costs[idx], grown.labels[idx]));
                }
            }
        }
        Ok(GrownBlock {
            core,
            runs,
            left,
            rim,
        })
    }

    /// Grows the pixels the windows left, all together: from the told
    /// pixels next to them, in the order the whole image takes them (see
    /// "Growing in windows"). Their rows.
    fn grow_left(&self, grown: Vec<GrownBlock>, block: usize) -> Result<Vec<Run>, InternalErrors> {
        let extent = self.channel.extent;
        let [ex0, ey0, ex1, ey1] = extent;
        let ew = ex1 - ex0;
        let blocks_x = ew.div_ceil(block);
        let block_of = |x: usize, y: usize| ((y - ey0) / block) * blocks_x + (x - ex0) / block;

        let left_px: u64 = grown
            .iter()
            .filter(|b| b.left.is_some())
            .map(|b| rect_area(Self::window(extent, b.core, 2)) as u64)
            .sum();
        let needed = left_px * LEFT_BYTES_PER_PIXEL;
        let available = crate::memory::available_memory_bytes();
        if needed > available / 2 {
            return Err(InternalErrors::InvalidArgument(format!(
                "Propagation: the regions grow far from their seeds (e.g. over the \
                 background) - that needs about {} of RAM, {} are free. Set a mask (e.g. \
                 the thresholded cell stain) so the regions stop at the cells",
                crate::memory::format_binary_bytes(needed),
                crate::memory::format_binary_bytes(available),
            )));
        }

        let mut rims = Vec::new();
        let mut blocks: HashMap<usize, LeftBlock> = HashMap::new();
        for b in grown {
            rims.push(b.rim);
            let Some(left) = b.left else { continue };
            let around = Self::window(extent, b.core, 2);
            let (values, _) = self.channel.read(self.cache, around)?;
            blocks.insert(
                block_of(b.core[0], b.core[1]),
                LeftBlock {
                    core: b.core,
                    labels: vec![0; rect_area(b.core)],
                    left,
                    values,
                    around,
                },
            );
        }
        let is_left = |blocks: &HashMap<usize, LeftBlock>, x: usize, y: usize| {
            blocks.get(&block_of(x, y)).is_some_and(|b| {
                b.left[(y - b.core[1]) * (b.core[2] - b.core[0]) + (x - b.core[0])]
            })
        };
        let neighbours = |x: usize, y: usize| {
            NEIGHBOURS.iter().filter_map(move |(dy, dx)| {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                (nx >= ex0 as isize && ny >= ey0 as isize && nx < ex1 as isize && ny < ey1 as isize)
                    .then_some((nx as usize, ny as usize))
            })
        };
        let position = |idx: u64| (ex0 + (idx as usize) % ew, ey0 + (idx as usize) / ew);

        // The rim next to pixels left, as the whole image takes it: by cost,
        // label, then position.
        let mut queue: BinaryHeap<Reverse<(u64, u32, u64)>> = rims
            .into_iter()
            .flatten()
            .filter(|&(idx, _, _)| {
                let (x, y) = position(idx);
                neighbours(x, y).any(|(nx, ny)| is_left(&blocks, nx, ny))
            })
            .map(|(idx, cost, label)| Reverse((cost.to_bits(), label, idx)))
            .collect();
        // The cheapest cost pushed so far, of the pixels in the queue.
        let mut costs: HashMap<u64, f64> = HashMap::new();
        while let Some(Reverse((bits, label, idx))) = queue.pop() {
            let (x, y) = position(idx);
            if is_left(&blocks, x, y) {
                let b = blocks
                    .get_mut(&block_of(x, y))
                    .expect("a block with pixels left");
                let at = (y - b.core[1]) * (b.core[2] - b.core[0]) + (x - b.core[0]);
                if b.labels[at] != 0 {
                    continue;
                }
                b.labels[at] = label;
                costs.remove(&idx);
            }
            let base = f64::from_bits(bits);
            for (nx, ny) in neighbours(x, y) {
                let Some(b) = blocks.get(&block_of(nx, ny)) else {
                    continue;
                };
                let width = b.core[2] - b.core[0];
                let at = (ny - b.core[1]) * width + (nx - b.core[0]);
                if !b.left[at] || b.labels[at] != 0 {
                    continue;
                }
                // In the block's coordinates (its values reach two pixels
                // past it).
                let frame = Frame {
                    values: &b.values,
                    stride: b.around[2] - b.around[0],
                    offset: (b.core[0] - b.around[0], b.core[1] - b.around[1]),
                    clamp: [
                        ey0 as isize - b.core[1] as isize,
                        ex0 as isize - b.core[0] as isize,
                        ey1 as isize - 1 - b.core[1] as isize,
                        ex1 as isize - 1 - b.core[0] as isize,
                    ],
                };
                let local = |px: usize, py: usize| {
                    (
                        py as isize - b.core[1] as isize,
                        px as isize - b.core[0] as isize,
                    )
                };
                let cost = base + step_cost(&frame, local(x, y), local(nx, ny), self.weight);
                let next = ((ny - ey0) * ew + (nx - ex0)) as u64;
                if costs.get(&next).is_none_or(|&c| cost < c) {
                    costs.insert(next, cost);
                    queue.push(Reverse((cost.to_bits(), label, next)));
                }
            }
        }

        let mut runs = Vec::new();
        for b in blocks.into_values() {
            let width = b.core[2] - b.core[0];
            for y in b.core[1]..b.core[3] {
                let row = (y - b.core[1]) * width;
                push_row_runs(
                    &mut runs,
                    y as u32,
                    b.core[0] as u32,
                    (0..width).map(|x| {
                        if b.left[row + x] {
                            b.labels[row + x]
                        } else {
                            0
                        }
                    }),
                );
            }
        }
        Ok(runs)
    }
}

/// The channel's values around a window, as a step's cost reads them.
struct Frame<'a> {
    /// Row-major, at least one pixel past the window where the image goes
    /// on.
    values: &'a [f32],
    stride: usize,
    /// The window's pixel (0, 0) in `values` (column, row).
    offset: (usize, usize),
    /// The first and last row and column of the image, in window
    /// coordinates (row min, column min, row max, column max): pixels past
    /// them repeat the border's.
    clamp: [isize; 4],
}

impl Frame<'_> {
    /// The whole image as one window.
    fn whole(values: &[f32], width: usize, height: usize) -> Frame<'_> {
        Frame {
            values,
            stride: width,
            offset: (0, 0),
            clamp: [0, 0, height as isize - 1, width as isize - 1],
        }
    }

    fn at(&self, r: isize, c: isize) -> f64 {
        let r = r.clamp(self.clamp[0], self.clamp[2]);
        let c = c.clamp(self.clamp[1], self.clamp[3]);
        self.values[(r + self.offset.1 as isize) as usize * self.stride
            + (c + self.offset.0 as isize) as usize] as f64
    }
}

/// The labels and costs a growth gave.
struct Grown {
    labels: Vec<u32>,
    costs: Vec<f64>,
}

/// [`grow`] over a whole `width` × `height` image: the labels (the
/// centrosome reference tests' entry).
#[cfg(test)]
fn propagate(
    values: &[f32],
    width: usize,
    height: usize,
    allowed: &BitSlice<u64, Lsb0>,
    seeds: &[(usize, u32)],
    weight: f64,
) -> Vec<u32> {
    let frame = Frame::whole(values, width, height);
    grow(&frame, width, height, allowed, seeds, weight).labels
}

/// Grows the seeds (pixel index, label > 0; where seeds overlap, the first
/// wins) into the `allowed` pixels of a `width` × `height` window of
/// `frame`: each pixel gets the label that reaches it cheapest (see
/// [`step_cost`]); equally cheap ones go to the lower label, then the
/// earlier pixel - a total heap order, so ties never depend on insertion
/// order. Seed pixels keep their label, also outside `allowed`; only seeds
/// inside it grow. The labels and the cheapest costs (`INFINITY` where no
/// seed reached).
///
/// Per pixel: its label (4 bytes) and cheapest cost (8 - an `f64`, so
/// summed costs keep their precision and ties are exact).
fn grow(
    frame: &Frame,
    width: usize,
    height: usize,
    allowed: &BitSlice<u64, Lsb0>,
    seeds: &[(usize, u32)],
    weight: f64,
) -> Grown {
    // Queue entries hold the pixel as u32: 16 bytes instead of 24.
    assert!(
        width * height <= u32::MAX as usize,
        "Propagation: an image of more than 4 billion pixels"
    );
    let mut labels = vec![0u32; width * height];
    // The cheapest cost pushed so far; seeds cost 0.
    let mut costs = vec![f64::INFINITY; width * height];
    let mut first_seeds = Vec::with_capacity(seeds.len());
    for &(idx, label) in seeds {
        if costs[idx] != 0.0 {
            costs[idx] = 0.0;
            first_seeds.push((idx, label));
        }
    }
    // Costs are >= 0, whose bit patterns sort like the numbers.
    let mut queue: BinaryHeap<Reverse<(u64, u32, u32)>> = first_seeds
        .iter()
        .filter(|&&(idx, _)| allowed[idx])
        .map(|&(idx, label)| Reverse((0f64.to_bits(), label, idx as u32)))
        .collect();
    while let Some(Reverse((_, label, idx))) = queue.pop() {
        let idx = idx as usize;
        if labels[idx] != 0 {
            continue;
        }
        labels[idx] = label;
        let base = costs[idx];
        let (row, col) = ((idx / width) as isize, (idx % width) as isize);
        for (dr, dc) in NEIGHBOURS {
            let (r, c) = (row + dr, col + dc);
            if r < 0 || c < 0 || r >= height as isize || c >= width as isize {
                continue;
            }
            let next = r as usize * width + c as usize;
            if labels[next] != 0 || !allowed[next] {
                continue;
            }
            let cost = base + step_cost(frame, (row, col), (r, c), weight);
            if cost < costs[next] {
                costs[next] = cost;
                queue.push(Reverse((cost.to_bits(), label, next as u32)));
            }
        }
    }
    for &(idx, label) in &first_seeds {
        labels[idx] = label;
    }
    Grown { labels, costs }
}

const NEIGHBOURS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// The least a path coming in from outside the window costs to reach each
/// pixel: `entries` (the window's edge pixels a path can come in at, with
/// the least a path costs there) and on through the `allowed` pixels as a
/// growth goes. Grown only up to `limit`: past it the pixels keep
/// `INFINITY` (they cost more anyway).
fn bound(
    frame: &Frame,
    width: usize,
    height: usize,
    allowed: &BitSlice<u64, Lsb0>,
    entries: &[(usize, f64)],
    weight: f64,
    limit: f64,
) -> Vec<f64> {
    let mut costs = vec![f64::INFINITY; width * height];
    let mut done = BitVec::<u64, Lsb0>::repeat(false, width * height);
    let mut queue: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &(idx, start) in entries {
        if start < costs[idx] {
            costs[idx] = start;
            queue.push(Reverse((start.to_bits(), idx as u32)));
        }
    }
    while let Some(Reverse((bits, idx))) = queue.pop() {
        let idx = idx as usize;
        if done[idx] {
            continue;
        }
        if f64::from_bits(bits) > limit {
            break;
        }
        done.set(idx, true);
        let base = costs[idx];
        let (row, col) = ((idx / width) as isize, (idx % width) as isize);
        for (dr, dc) in NEIGHBOURS {
            let (r, c) = (row + dr, col + dc);
            if r < 0 || c < 0 || r >= height as isize || c >= width as isize {
                continue;
            }
            let next = r as usize * width + c as usize;
            if done[next] || !allowed[next] {
                continue;
            }
            let cost = base + step_cost(frame, (row, col), (r, c), weight);
            if cost < costs[next] {
                costs[next] = cost;
                queue.push(Reverse((cost.to_bits(), next as u32)));
            }
        }
    }
    // Not taken: dearer than `limit` (a pushed cost isn't the cheapest yet).
    for (idx, cost) in costs.iter_mut().enumerate() {
        if !done[idx] {
            *cost = f64::INFINITY;
        }
    }
    costs
}

/// A step's cost (see [`step_cost`]) without any change in brightness,
/// `manhattan` 1 straight, 2 diagonal - computed as `step_cost` does: no
/// step of that kind costs less.
fn step_cost_without_change(manhattan: f64, weight: f64) -> f64 {
    (0.0f64 * 0.0 + manhattan * weight * weight).sqrt()
}

/// The cost of the step between neighbours `a` and `b` (row,
/// column, in the window): the summed absolute difference of their 3×3
/// neighbourhoods (pixels past the image's border repeat the border's)
/// combined with their Manhattan distance weighted by `weight`²:
/// `sqrt(diff² + manhattan · weight²)`.
///
/// Summed in a fixed order (rows, then columns), so the result is the same
/// to the last bit every run - and in every window.
fn step_cost(frame: &Frame, a: (isize, isize), b: (isize, isize), weight: f64) -> f64 {
    let [r_min, c_min, r_max, c_max] = frame.clamp;
    let inside = |p: (isize, isize)| p.0 > r_min && p.1 > c_min && p.0 < r_max && p.1 < c_max;
    let mut diff = 0.0;
    if inside(a) && inside(b) {
        // Away from the border (nearly every pixel): no clamping.
        let w = frame.stride as isize;
        let index = |p: (isize, isize)| {
            ((p.0 + frame.offset.1 as isize) * w + p.1 + frame.offset.0 as isize) as usize
        };
        let (ia, ib) = (index(a), index(b));
        for dr in [-w, 0, w] {
            for dc in -1..=1 {
                let offset = dr + dc;
                let va = frame.values[ia.wrapping_add_signed(offset)] as f64;
                let vb = frame.values[ib.wrapping_add_signed(offset)] as f64;
                diff += (va - vb).abs();
            }
        }
    } else {
        for dr in -1..=1 {
            for dc in -1..=1 {
                diff += (frame.at(a.0 + dr, a.1 + dc) - frame.at(b.0 + dr, b.1 + dc)).abs();
            }
        }
    }
    let manhattan = ((a.0 - b.0).abs() + (a.1 - b.1).abs()) as f64;
    (diff * diff + manhattan * weight * weight).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::scope::Scope;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    /// A 10×10 image (`image[row][col]`), seeds at (row, col) with labels
    /// 1, 2, ...; the labels grown, as rows.
    fn grow(
        image: &[[f32; 10]; 10],
        seeds: &[(usize, usize)],
        allowed: &[bool],
        weight: f64,
    ) -> Vec<Vec<u32>> {
        let values: Vec<f32> = image.iter().flatten().copied().collect();
        let seeds: Vec<(usize, u32)> = seeds
            .iter()
            .enumerate()
            .map(|(i, (r, c))| (r * 10 + c, i as u32 + 1))
            .collect();
        let allowed: BitVec<u64, Lsb0> = allowed.iter().collect();
        propagate(&values, 10, 10, &allowed, &seeds, weight)
            .chunks(10)
            .map(|row| row.to_vec())
            .collect()
    }

    // The cases of centrosome's own tests (`test_propagate.py`).

    #[test]
    fn nothing_grows_without_seeds() {
        let labels = grow(&[[0.0; 10]; 10], &[], &[true; 100], 1.0);
        assert!(labels.iter().flatten().all(|&l| l == 0));
    }

    #[test]
    fn one_seed_takes_everything() {
        let labels = grow(&[[0.0; 10]; 10], &[(5, 5)], &[true; 100], 1.0);
        assert!(labels.iter().flatten().all(|&l| l == 1));
    }

    #[test]
    fn two_seeds_split_a_flat_image_halfway() {
        let labels = grow(&[[0.0; 10]; 10], &[(0, 5), (9, 5)], &[true; 100], 1.0);
        assert!(labels[..5].iter().flatten().all(|&l| l == 1));
        assert!(labels[5..].iter().flatten().all(|&l| l == 2));
    }

    /// A bright wall around the top left corner keeps the other seed out.
    #[test]
    fn a_bright_wall_is_a_border() {
        let mut image = [[0.0; 10]; 10];
        for col in 0..5 {
            image[5][col] = 1.0;
        }
        for row in image.iter_mut().take(6) {
            row[5] = 1.0;
        }
        let labels = grow(&image, &[(0, 0), (9, 0)], &[true; 100], 0.1);
        for (r, row) in labels.iter().enumerate() {
            for (c, &l) in row.iter().enumerate() {
                if r < 5 && c < 5 {
                    assert_eq!(l, 1, "({r}, {c})");
                }
                if r > 5 || c > 5 {
                    assert_eq!(l, 2, "({r}, {c})");
                }
            }
        }
    }

    /// Through a gap in the wall the first seed reaches past it.
    #[test]
    fn a_leaky_wall_lets_through() {
        let mut image = [[0.0; 10]; 10];
        for col in 1..4 {
            image[4][col] = 1.0;
        }
        for row in image.iter_mut().take(4) {
            row[4] = 1.0;
        }
        let labels = grow(&image, &[(0, 0), (9, 0)], &[true; 100], 0.1);
        for row in &labels[..4] {
            assert!(row[..4].iter().all(|&l| l == 1));
        }
        assert_eq!(labels[4][0], 1);
    }

    #[test]
    fn pixels_outside_the_mask_stay_empty() {
        let mut allowed = [true; 100];
        allowed[2 * 10 + 2] = false;
        allowed[7 * 10 + 2] = false;
        let labels = grow(&[[0.0; 10]; 10], &[(0, 5), (9, 5)], &allowed, 1.0);
        assert_eq!((labels[2][2], labels[7][2]), (0, 0));
        for (r, row) in labels.iter().enumerate() {
            for (c, &l) in row.iter().enumerate() {
                if (r, c) != (2, 2) && (r, c) != (7, 2) {
                    assert_eq!(l, if r < 5 { 1 } else { 2 }, "({r}, {c})");
                }
            }
        }
    }

    /// The cost formula, by hand: a flat image costs only distance.
    #[test]
    fn a_step_costs_brightness_change_and_distance() {
        let flat = vec![0.0f32; 9];
        let flat = Frame::whole(&flat, 3, 3);
        let side = step_cost(&flat, (1, 1), (1, 2), 0.5);
        let diagonal = step_cost(&flat, (1, 1), (2, 2), 0.5);
        assert!((side - 0.5).abs() < 1e-12);
        assert!((diagonal - (2.0f64).sqrt() * 0.5).abs() < 1e-12);
        // One bright pixel at (0, 0): it lies in a's window (1, 1) and in
        // b's (1, 0) window too, so only the step (1, 1) -> (2, 2) sees it.
        let mut spot = vec![0.0f32; 9];
        spot[0] = 1.0;
        let cost = step_cost(&Frame::whole(&spot, 3, 3), (1, 1), (2, 2), 0.0);
        assert!((cost - 1.0).abs() < 1e-12, "{cost}");
    }

    /// Pixel for pixel the reference result
    /// (`centrosome.propagate` 1.3.4 on Gaussian "cells" with noise; fixture
    /// `tests/propagation.json`).
    #[test]
    fn regions_match_reference() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            width: usize,
            height: usize,
            weight: f64,
            image: Vec<f64>,
            seeds: Vec<u32>,
            mask: Vec<bool>,
            expected: Vec<u32>,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/propagation.json");
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(fixture.cases.len(), 5);
        for case in fixture.cases {
            let values: Vec<f32> = case.image.iter().map(|&v| v as f32).collect();
            // The reference's seed image as a list of seed pixels.
            let seeds: Vec<(usize, u32)> = case
                .seeds
                .iter()
                .enumerate()
                .filter(|&(_, &label)| label > 0)
                .map(|(idx, &label)| (idx, label))
                .collect();
            let mask: BitVec<u64, Lsb0> = case.mask.iter().collect();
            let labels = propagate(&values, case.width, case.height, &mask, &seeds, case.weight);
            let differing = labels
                .iter()
                .zip(&case.expected)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differing, 0, "{}: {differing} pixels differ", case.name);
        }
    }

    /// The growth over the whole image at once, as it was before the
    /// windows (2026-10-29), kept as the reference the windows must match.
    mod reference {
        use bitvec::prelude::*;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        pub fn propagate(
            values: &[f32],
            width: usize,
            height: usize,
            allowed: &BitSlice<u64, Lsb0>,
            seeds: &[(usize, u32)],
            weight: f64,
        ) -> Vec<u32> {
            let mut labels = vec![0u32; values.len()];
            let mut costs = vec![f64::INFINITY; values.len()];
            let mut first_seeds = Vec::with_capacity(seeds.len());
            for &(idx, label) in seeds {
                if costs[idx] != 0.0 {
                    costs[idx] = 0.0;
                    first_seeds.push((idx, label));
                }
            }
            let mut queue: BinaryHeap<Reverse<(u64, u32, u32)>> = first_seeds
                .iter()
                .filter(|&&(idx, _)| allowed[idx])
                .map(|&(idx, label)| Reverse((0f64.to_bits(), label, idx as u32)))
                .collect();
            while let Some(Reverse((_, label, idx))) = queue.pop() {
                let idx = idx as usize;
                if labels[idx] != 0 {
                    continue;
                }
                labels[idx] = label;
                let base = costs[idx];
                let (row, col) = ((idx / width) as isize, (idx % width) as isize);
                for (dr, dc) in super::NEIGHBOURS {
                    let (r, c) = (row + dr, col + dc);
                    if r < 0 || c < 0 || r >= height as isize || c >= width as isize {
                        continue;
                    }
                    let next = r as usize * width + c as usize;
                    if labels[next] != 0 || !allowed[next] {
                        continue;
                    }
                    let cost = base + step_cost(values, width, height, (row, col), (r, c), weight);
                    if cost < costs[next] {
                        costs[next] = cost;
                        queue.push(Reverse((cost.to_bits(), label, next as u32)));
                    }
                }
            }
            for &(idx, label) in &first_seeds {
                labels[idx] = label;
            }
            labels
        }

        fn step_cost(
            values: &[f32],
            width: usize,
            height: usize,
            a: (isize, isize),
            b: (isize, isize),
            weight: f64,
        ) -> f64 {
            let (w, h) = (width as isize, height as isize);
            let inside = |p: (isize, isize)| p.0 >= 1 && p.1 >= 1 && p.0 < h - 1 && p.1 < w - 1;
            let mut diff = 0.0;
            if inside(a) && inside(b) {
                let (ia, ib) = ((a.0 * w + a.1) as usize, (b.0 * w + b.1) as usize);
                for dr in [-w, 0, w] {
                    for dc in -1..=1 {
                        let offset = dr + dc;
                        let va = values[ia.wrapping_add_signed(offset)] as f64;
                        let vb = values[ib.wrapping_add_signed(offset)] as f64;
                        diff += (va - vb).abs();
                    }
                }
            } else {
                let at = |r: isize, c: isize| {
                    let r = r.clamp(0, h - 1) as usize;
                    let c = c.clamp(0, w - 1) as usize;
                    values[r * width + c] as f64
                };
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        diff += (at(a.0 + dr, a.1 + dc) - at(b.0 + dr, b.1 + dc)).abs();
                    }
                }
            }
            let manhattan = ((a.0 - b.0).abs() + (a.1 - b.1).abs()) as f64;
            (diff * diff + manhattan * weight * weight).sqrt()
        }
    }

    /// A small deterministic random generator (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// A random object over `bbox` (about `fill` of its pixels set).
    fn blob(id: u128, bbox: [u32; 4], fill: f32, rng: &mut Rng) -> Object {
        let w = (bbox[2] - bbox[0] + 1) as usize;
        let h = (bbox[3] - bbox[1] + 1) as usize;
        let mut mask: BitVec<u64, Lsb0> = (0..w * h).map(|_| rng.unit() < fill).collect();
        mask.set(0, true);
        let area = mask.count_ones();
        Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: mask,
            area,
            ..Default::default()
        })
    }

    /// One random case grown in windows (`block`, `halo` on the first round)
    /// and over the whole image at once: the labels must be the same.
    fn compare_with_whole_image(seed: u64, block: usize, halo: usize, weight: f64, masked: bool) {
        let mut rng = Rng(seed);
        let (width, height) = (40 + rng.below(70), 40 + rng.below(70));
        // Cells: smooth bright blobs, noise, and now and then a NaN.
        let centres: Vec<(f32, f32)> = (0..6)
            .map(|_| (rng.below(width) as f32, rng.below(height) as f32))
            .collect();
        let values: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                let cells: f32 = centres
                    .iter()
                    .map(|(cx, cy)| (-((x - cx).powi(2) + (y - cy).powi(2)) / 60.0).exp())
                    .sum();
                if rng.below(500) == 0 {
                    f32::NAN
                } else {
                    (0.6 * cells + 0.2 * rng.unit()).min(1.0)
                }
            })
            .collect();
        // In tiles of 23 pixels, as the cache holds a slide.
        let mut cache = GlobalPipelineCache::default();
        let tile = 23;
        for ty in (0..height).step_by(tile) {
            for tx in (0..width).step_by(tile) {
                let (tw, th) = (tile.min(width - tx), tile.min(height - ty));
                let data: Vec<f32> = (0..tw * th)
                    .map(|i| values[(ty + i / tw) * width + tx + i % tw])
                    .collect();
                cache.add_to_channel_cache(
                    Arc::new(ImageContainer::F32Gray(ManagedImage {
                        data: Image::new(
                            ImageSize {
                                width: tw,
                                height: th,
                            },
                            data,
                        )
                        .unwrap(),
                        tile_offset: Point2d { x: tx, y: ty },
                        plane: None,
                        dimensionality: ImageDimensionality::Planar,
                    })),
                    1,
                    ImageTile {
                        offset_x: tx,
                        offset_y: ty,
                        width: tw,
                        height: th,
                    },
                );
            }
        }
        let random_bbox = |rng: &mut Rng, max: usize| {
            let (x, y) = (rng.below(width) as u32, rng.below(height) as u32);
            let (w, h) = (rng.below(max) as u32, rng.below(max) as u32);
            [
                x,
                y,
                (x + w).min(width as u32 - 1),
                (y + h).min(height as u32 - 1),
            ]
        };
        // Overlapping now and then: the first seed wins.
        let seeds: Vec<Object> = (0..3 + rng.below(12))
            .map(|i| {
                let bbox = random_bbox(&mut rng, 5);
                blob(i as u128 + 1, bbox, 0.7, &mut rng)
            })
            .collect();
        let masks: Vec<Object> = (0..4)
            .map(|i| {
                let bbox = random_bbox(&mut rng, 50);
                blob(100 + i, bbox, 0.9, &mut rng)
            })
            .collect();

        // The whole image at once.
        let mut read: BitVec<u64, Lsb0> = BitVec::repeat(false, width * height);
        let whole: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                read.set(i, v.is_finite());
                if v.is_finite() { *v } else { 0.0 }
            })
            .collect();
        let window = [0, 0, width, height];
        let allowed: BitVec<u64, Lsb0> = if masked {
            let mut allowed = BitVec::repeat(false, width * height);
            for mask in &masks {
                for_each_pixel_in(mask, window, |idx| {
                    if read[idx] {
                        allowed.set(idx, true)
                    }
                });
            }
            allowed
        } else {
            read.clone()
        };
        let mut seed_pixels = Vec::new();
        for (i, seed) in seeds.iter().enumerate() {
            for_each_pixel_in(seed, window, |idx| seed_pixels.push((idx, i as u32 + 1)));
        }
        let expected = reference::propagate(&whole, width, height, &allowed, &seed_pixels, weight);

        // In windows.
        let channel = GrowthChannel::of(&cache, 1).unwrap();
        let seed_refs: Vec<&Object> = seeds.iter().collect();
        let mask_refs: Vec<&Object> = masks.iter().collect();
        let growth = Growth {
            channel: &channel,
            cache: &cache,
            seeds: ObjectsGrid::new(&seed_refs),
            masks: masked.then(|| ObjectsGrid::new(&mask_refs)),
            weight,
        };
        let regions = growth.regions(block, halo).unwrap();
        let mut labels = vec![0u32; width * height];
        for (i, region) in regions.iter().enumerate() {
            let Some(region) = region else { continue };
            let [x0, y0, x1, _] = region.bbox.map(|v| v as usize);
            let w = x1 - x0 + 1;
            let mut area = 0;
            for bit in region.mask.iter_ones() {
                let idx = (y0 + bit / w) * width + x0 + bit % w;
                assert_eq!(labels[idx], 0, "case {seed}: a pixel in two regions");
                labels[idx] = i as u32 + 1;
                area += 1;
            }
            assert_eq!(region.area, area, "case {seed}: area");
        }
        let differing = labels.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "case {seed} ({width}x{height}, block {block}, halo {halo}, weight {weight}, \
             mask {masked}): {differing} pixels differ"
        );
    }

    /// Windows of every size give the labels of the whole image - with and
    /// without a mask, at any regularization (0: only the whole image can
    /// tell, the windows grow to it).
    #[test]
    fn windows_give_the_same_regions_as_the_whole_image() {
        for seed in 1..=60u64 {
            for (block, halo) in [(8, 2), (16, 4), (32, 1)] {
                for weight in [0.05, 0.01, 0.5, 0.0] {
                    compare_with_whole_image(seed, block, halo, weight, seed % 2 == 0);
                }
            }
        }
    }

    // The command on a cache.

    fn filled(id: u128, bbox: [u32; 4], class: ObjectClass) -> Object {
        let w = bbox[2] - bbox[0] + 1;
        let h = bbox[3] - bbox[1] + 1;
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox,
            mask_data: bitvec![u64, Lsb0; 1; (w * h) as usize],
            area: (w * h) as usize,
            ..Default::default()
        });
        object.add_object_class(class, Scope::default()).unwrap();
        object
    }

    const NUCLEUS: ObjectClass = ObjectClass::Global(1);
    const CELL: ObjectClass = ObjectClass::Global(2);
    const STAIN: ObjectClass = ObjectClass::Global(3);

    /// A 20×10 image with a bright stripe at column 12, two nuclei left and
    /// right of the middle.
    fn cache_with_two_nuclei() -> GlobalPipelineCache {
        let (width, height) = (20, 10);
        let mut values = vec![0.2f32; width * height];
        for row in 0..height {
            values[row * width + 12] = 1.0;
        }
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::new(ImageSize { width, height }, values).unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            1,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width,
                height,
            },
        );
        for object in [
            filled(1, [3, 4, 4, 5], NUCLEUS),
            filled(2, [16, 4, 17, 5], NUCLEUS),
        ] {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn step() -> Propagation {
        Propagation {
            seeds: NUCLEUS,
            growth_channel: ImageChannelIdx(1),
            mask: ObjectClass::Unset,
            regularization: 0.05,
            output_class: CELL,
            exclude_at_edges: false,
        }
    }

    fn run(step: &Propagation, cache: &mut GlobalPipelineCache) -> Result<(), InternalErrors> {
        let mut meta = PipelineImageMeta::default();
        meta.full_image_width = ImageSize {
            width: 20,
            height: 10,
        };
        let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
        step.execute(&mut ctx, cache)
    }

    fn cells(cache: &GlobalPipelineCache) -> Vec<&Object> {
        cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&CELL, Scope::default()))
            .collect()
    }

    /// The border falls on the bright stripe, not halfway (column 10);
    /// each cell holds its nucleus.
    #[test]
    fn cells_meet_at_the_stripe_and_hold_their_nucleus() {
        let mut cache = cache_with_two_nuclei();
        run(&step(), &mut cache).unwrap();
        let cells = cells(&cache);
        assert_eq!(cells.len(), 2);
        let left = cells.iter().min_by_key(|c| c.bbox[0]).unwrap();
        let right = cells.iter().max_by_key(|c| c.bbox[0]).unwrap();
        assert_eq!(left.area + right.area, 200, "every pixel is taken");
        assert!(
            left.bbox[2] >= 11,
            "left reaches past the middle: {:?}",
            left.bbox
        );
        assert!(
            right.bbox[0] >= 12,
            "right stops at the stripe: {:?}",
            right.bbox
        );
        assert!(left.is_part_of(3, 4) && right.is_part_of(17, 5));
        assert!(!left.intensities.is_empty(), "intensities measured");
    }

    #[test]
    fn cells_stay_inside_the_mask() {
        let mut cache = cache_with_two_nuclei();
        let stain = filled(9, [2, 3, 18, 6], STAIN);
        cache.object_cache.insert(stain.id.clone(), stain);
        run(
            &Propagation {
                mask: STAIN,
                ..step()
            },
            &mut cache,
        )
        .unwrap();
        let cells = cells(&cache);
        assert_eq!(cells.iter().map(|c| c.area).sum::<usize>(), 17 * 4);
        assert!(cells.iter().all(|c| c.bbox[1] >= 3 && c.bbox[3] <= 6));
    }

    #[test]
    fn cells_at_the_border_can_be_left_out() {
        let mut cache = cache_with_two_nuclei();
        run(
            &Propagation {
                exclude_at_edges: true,
                ..step()
            },
            &mut cache,
        )
        .unwrap();
        assert!(cells(&cache).is_empty(), "both reach the image border");
    }

    #[test]
    fn classes_must_be_chosen_and_the_channel_exist() {
        let mut cache = cache_with_two_nuclei();
        let unset = Propagation {
            output_class: ObjectClass::Unset,
            ..step()
        };
        assert!(run(&unset, &mut cache).is_err());
        let missing = Propagation {
            growth_channel: ImageChannelIdx(4),
            ..step()
        };
        let err = run(&missing, &mut cache).unwrap_err().to_string();
        assert!(err.contains("no channel 4"), "{err}");
    }

    /// Peak RSS of this process in MB (Linux).
    fn peak_rss_mb() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
            })
            .map_or(0.0, |kb| kb / 1024.0)
    }

    /// Speed and peak RAM on a `BENCH_SIZE`² image (default 4096) with a
    /// nucleus every 40 pixels in a cell stain - run in release, one size
    /// per process:
    /// `BENCH_SIZE=8192 cargo test --release -p evanalyzer_core --lib bench_propagation -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_propagation() {
        let size: usize = std::env::var("BENCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let spacing = 40usize;
        // Bright cells around each nucleus, dim gaps between them.
        let values: Vec<f32> = (0..size * size)
            .map(|i| {
                let (x, y) = (i % size, i / size);
                let dx = (x % spacing) as f32 - spacing as f32 / 2.0;
                let dy = (y % spacing) as f32 - spacing as f32 / 2.0;
                (-(dx * dx + dy * dy) / 200.0).exp()
            })
            .collect();
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 16;
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::new(
                    ImageSize {
                        width: size,
                        height: size,
                    },
                    values,
                )
                .unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            1,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: size,
                height: size,
            },
        );
        let mut id = 0;
        for cy in (spacing / 2..size).step_by(spacing) {
            for cx in (spacing / 2..size).step_by(spacing) {
                id += 1;
                let (x, y) = (cx as u32 - 3, cy as u32 - 3);
                let nucleus = filled(id, [x, y, x + 6, y + 6], NUCLEUS);
                cache.object_cache.insert(nucleus.id.clone(), nucleus);
            }
        }
        let before_mb = peak_rss_mb();
        let mut meta = PipelineImageMeta::default();
        meta.full_image_width = ImageSize {
            width: size,
            height: size,
        };
        let mut ctx = PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap();
        let start = std::time::Instant::now();
        step().execute(&mut ctx, &mut cache).unwrap();
        let elapsed = start.elapsed();
        let cells = cells(&cache).len();
        println!(
            "propagation {size}x{size} ({:.1} MP), {id} nuclei -> {cells} cells: {:.2} s, \
             peak RSS {:.0} MB (image + nuclei before: {before_mb:.0} MB)",
            (size * size) as f64 / 1e6,
            elapsed.as_secs_f64(),
            peak_rss_mb()
        );
        assert_eq!(cells, id as usize);
    }
}
