//! # Grow Regions
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Grows a region from each seed object (a maximum of Find Maxima, a
//! nucleus, an annotation) through a channel - a "magic wand" per seed.
//! Pixels join in a priority order from the seed outwards (seeded region
//! growing, Adams & Bischof 1994): the brightest first (Intensity, a
//! flooding from markers as in Vincent & Soille 1991), or the most similar
//! to the seed first (Similarity). Every seed has its own stop level -
//! relative to its own peak and background, a tolerance below its peak
//! (the support of the peak's h-dome, Vincent 1993), a fixed threshold, or
//! a multiple of the noise - so dim and bright objects are cut alike.
//!
//! ## How seeds compete
//! Each seed is flooded on its own, inside `max_distance` of it: a pixel's
//! score for a seed is the best path to it - the highest of the lowest
//! brightness along the path (Intensity), or the lowest of the largest
//! difference to the seed's value (Similarity). With Separate a pixel goes
//! to the seed scoring best on it (then the nearer one, then the first in
//! image order) and each region keeps the pixels connected to its seed - so
//! the border between two touching spots lies on the darkest line between
//! them. Defined per pixel from the seeds near it, the result doesn't
//! depend on how the image is split up: seeds whose reach doesn't meet are
//! grown apart, a large group of them in blocks with a margin, and a whole
//! slide never needs a buffer per pixel of the image.

use crate::algos::libs::channel_window::{ChannelWindow, Rect};
use crate::algos::libs::feature_transform::{NearestFeatures, nearest_features};
use crate::algos::libs::robust_stats::{background_and_noise_in_place, median_in_place};
use crate::algos::libs::row_runs::{Run, RunRegion, push_row_runs, regions_from_runs};
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::{Object, ObjectInit};
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass, ObjectId, PixelUnits, SizeUnits,
};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// Grows a region from each seed object through a channel, like a magic
/// wand: e.g. a spot's outline from its maximum, or a nucleus from a point
/// inside it.
///
/// [Find Maxima] -> [Extract Objects] -> [Grow Regions]
///
/// Every seed gets its own stop level (relative to its own peak and
/// background, so dim and bright objects keep the same shape); touching
/// regions are split on the darkest line between them, may overlap, or are
/// merged. Each region keeps its seed's pixels; Relate Objects links it to
/// its seed. Seeds 1024 px across or more (a tissue-sized object) don't
/// grow: their region is the seed.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "object", display_name = "Grow Regions")]
pub struct GrowRegions {
    /// Objects carrying this class are the seeds (e.g. the maxima of Find
    /// Maxima after Extract Objects, or nuclei): each grows into one region.
    #[cmdsmeta(summary = true)]
    pub seeds: ObjectClass,

    /// The channel the regions grow through (the raw spot or nucleus
    /// stain).
    #[cmdsmeta(summary = true)]
    pub growth_channel: ImageChannelIdx,

    /// Bright: objects brighter than their background (fluorescence).
    /// Dark: objects darker than their background (brightfield, stains) -
    /// everything below then counts the other way round.
    #[cmdsmeta(default = Polarity::Bright)]
    pub objects_are: Polarity,

    /// Which neighbouring pixel joins a region next.
    ///
    /// Intensity: the brightest - a region spreads downhill from its peak.
    /// For objects with a bright centre that fades out (spots, blobs).
    /// Similarity: the one closest to the seed's value (the median of the
    /// seed's pixels) - a region covers what looks like its seed. For evenly
    /// bright objects (nuclei, organelles); larger seeds give a steadier
    /// value than a single pixel.
    #[cmdsmeta(default = GrowthOrder::Intensity, summary = true)]
    pub order: GrowthOrder,

    /// Which pixels may join a region (the seed's own pixels always
    /// belong to it).
    ///
    /// Relative, Noise: the background is the median of the pixels just
    /// beyond `max_distance` around the seed (2 px wide ring), the noise
    /// their median absolute deviation × 1.4826.
    #[cmdsmeta(summary = true)]
    pub stop: StopRule,

    /// How far a region reaches at most from its seed (from the nearest
    /// seed pixel). Also where the background is measured: a little more
    /// than the largest object's radius.
    #[cmdsmeta(default = 10.0, min = 1.0, max = 1000.0, step = 1.0, summary = true)]
    pub max_distance: f32,

    /// Unit of `max_distance`.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub distance_unit: SizeUnits,

    /// What happens where regions of different seeds meet.
    pub when_regions_meet: Meeting,

    /// Regions only grow into pixels of objects carrying this class (e.g.
    /// the thresholded cells). `Unset`: everywhere.
    pub mask: ObjectClass,

    /// The class the grown regions get.
    pub output_class: ObjectClass,

    /// Discard regions that touch the image border.
    pub exclude_at_edges: bool,
}

/// Whether [`GrowRegions`]' objects are brighter or darker than their
/// background.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Polarity {
    /// Brighter than the background (fluorescence).
    Bright,
    /// Darker than the background (brightfield, stains).
    Dark,
}

/// Which neighbouring pixel joins a [`GrowRegions`] region next.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum GrowthOrder {
    /// The brightest: a region spreads downhill from its peak (spots,
    /// blobs).
    Intensity,
    /// The one closest to the seed's value: a region covers what looks like
    /// its seed (evenly bright objects).
    Similarity,
}

/// Which pixels may join a [`GrowRegions`] region. "Brighter"/"above"
/// count the other way round for dark objects.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum StopRule {
    /// A level between the background (0) and the seed (1): 0.5 cuts each
    /// region at half its height (full width at half maximum), so dim and
    /// bright objects keep the same shape. Intensity: pixels above the
    /// level, measured from the region's peak (the brightest pixel the seed
    /// reaches without going down). Similarity: pixels closer to
    /// the seed's value than (1 - fraction) of its contrast, on either side.
    /// A seed not above its background grows nothing.
    Relative {
        /// The level between the background (0) and the seed (1).
        #[cmdsmeta(default = 0.5, min = 0.0, max = 1.0, step = 0.05)]
        fraction: f32,
    },
    /// The classic magic wand. Intensity: pixels at most `tolerance` below
    /// the region's peak. Similarity: pixels at most `tolerance` from the
    /// seed's value, on either side.
    Tolerance {
        /// How far a pixel may lie from the peak or seed value.
        #[cmdsmeta(default = 10.0, min = 0.0, max = 65535.0, step = 1.0)]
        tolerance: f32,
        /// Unit of `tolerance` (bit: 0 - 255/65535, %: 0 - 100, rel: 0 - 1).
        #[cmdsmeta(default = PixelUnits::Bit)]
        unit: PixelUnits,
    },
    /// Pixels above one fixed level, the same for every seed.
    Threshold {
        /// The level a pixel must lie above.
        #[cmdsmeta(default = 0.0, min = 0.0, max = 65535.0, step = 1.0)]
        level: f32,
        /// Unit of `level` (bit: 0 - 255/65535, %: 0 - 100, rel: 0 - 1).
        #[cmdsmeta(default = PixelUnits::Bit)]
        unit: PixelUnits,
    },
    /// Pixels more than `noise_factor` times the noise above the
    /// background: a region covers what stands out from the noise around
    /// it - the same factor fits every exposure.
    Noise {
        /// How many times the noise a pixel must lie above the background.
        #[cmdsmeta(default = 3.0, min = 0.0, max = 100.0, step = 0.5)]
        noise_factor: f32,
    },
}

/// What [`GrowRegions`] does where regions of different seeds meet.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Meeting {
    /// Each pixel goes to one region: the border lies on the darkest line
    /// between them (touching spots counted apart).
    Separate,
    /// Every region grows as if it were alone: a pixel may belong to
    /// several (e.g. each spot's whole halo) - a region may then take in a
    /// neighbouring spot if the valley between them stays above its stop
    /// level.
    Overlap,
    /// Regions that meet over a shallow valley become one object (several
    /// maxima on one object); regions with a deep valley between them stay
    /// apart.
    Merge {
        /// The deepest valley merged, as a part of the dimmer region's
        /// contrast (peak - background): 0.2 merges regions whose border
        /// lies at most 20 % below the dimmer peak. Two objects only
        /// touching at their stop level never merge below 1 - the stop
        /// fraction.
        #[cmdsmeta(default = 0.2, min = 0.0, max = 1.0, step = 0.05)]
        max_valley: f32,
    },
}

/// The width (px) of the ring beyond `max_distance` the background is
/// measured in.
const RING: usize = 2;
/// Seeds this many pixels across or more (a tissue-sized object, never a
/// spot or nucleus) neither grow nor compete: their region is the seed.
/// Growing one would need its whole box at once - on a whole slide most of
/// the slide. Smaller in the library's tests.
const MAX_SEED_SIDE: usize = if cfg!(test) { 40 } else { 1024 };
/// A group of seeds whose values window has at most this many pixels grows
/// in one piece (~17 bytes per pixel, 18 MB at most); a larger one in
/// blocks. Measured on 200 × 2048² images with 12,000 seeds each (one group
/// over each image), 21 at once: 1024² took 20-30 % less time than one
/// piece up to 4096² (the workers' arrays then thrashed the shared caches)
/// and less than 512² (more floods redone at the blocks' margins). Tiny in
/// the library's tests, so their images use blocks.
const ONE_PIECE_PIXELS: usize = if cfg!(test) { 48 * 48 } else { 1024 * 1024 };
/// The side of a block (pixels) a large group of seeds is split into, by
/// where the seeds lie.
const BLOCK: usize = if cfg!(test) { 24 } else { 1024 };
/// A unit up to this many window pixels runs beside any other; larger ones
/// as many at once as fit in a quarter of the free RAM.
const SMALL_UNIT_PIXELS: usize = 512 * 512;
/// RAM per pixel of a unit's window: values, mask, score, distance, label.
const UNIT_BYTES_PER_PIXEL: u64 = 17;

impl ImageAlgorithm for GrowRegions {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.seeds == ObjectClass::Unset || self.output_class == ObjectClass::Unset {
            return Err(InternalErrors::InvalidArgument(
                "Grow regions: choose the seeds' class and the output class".into(),
            ));
        }
        let channel = ChannelWindow::of(cache, self.growth_channel.0 as i32, "Grow regions")?;
        let scope = ctx.scope;
        let mut seeds: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.seeds, scope))
            .collect();
        if seeds.is_empty() {
            return Ok(());
        }
        // In image order: ties go to the first seed in reading order, the
        // same in every run.
        seeds.sort_by(|a, b| a.position_cmp(b));
        let masks: Option<Vec<&Object>> = (self.mask != ObjectClass::Unset).then(|| {
            cache
                .object_cache
                .values()
                .filter(|o| o.is_of(&self.mask, scope))
                .collect()
        });
        let params = self.params(ctx);
        let grower = Grower::new(&channel, cache, &seeds, masks.as_deref(), params);
        let regions = grower.regions(ONE_PIECE_PIXELS, BLOCK)?;

        let full = ctx.full_image_size();
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
            object.add_object_class(self.output_class, scope)?;
            new_objects.push(object);
        }
        // The regions never pass Extract Objects (which samples the
        // pixels): their intensities are measured here.
        Object::measure_intensities_of(&mut new_objects, cache);
        for object in new_objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Grow Regions"
    }

    /// Seeded region growing (Adams & Bischof 1994) in the order of a
    /// flooding from markers (Vincent & Soille 1991); the tolerance below a
    /// peak is the support of its h-dome (Vincent 1993); the noise is the
    /// scaled median absolute deviation (Hampel 1974).
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut papers = vec![&ADAMS_1994];
        if self.order == GrowthOrder::Intensity {
            papers.push(&VINCENT_SOILLE_1991);
        }
        match self.stop {
            StopRule::Tolerance { .. } if self.order == GrowthOrder::Intensity => {
                papers.push(&VINCENT_1993)
            }
            StopRule::Noise { .. } => papers.push(&HAMPEL_1974),
            _ => {}
        }
        papers
    }

    /// A region may grow across tile borders.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }
}

static ADAMS_1994: CitationMetadata = CitationMetadata {
    cite_key: "adams1994seeded",
    title: "Seeded Region Growing",
    authors: &["Rolf Adams", "Leanne Bischof"],
    year: 1994,
    container: Some("IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    doi: Some("10.1109/34.295913"),
    url: Some("https://doi.org/10.1109/34.295913"),
    pages: Some("641-647"),
};

static VINCENT_SOILLE_1991: CitationMetadata = CitationMetadata {
    cite_key: "vincent1991watersheds",
    title: "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations",
    authors: &["Luc Vincent", "Pierre Soille"],
    year: 1991,
    container: Some("IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    doi: Some("10.1109/34.87344"),
    url: Some("https://doi.org/10.1109/34.87344"),
    pages: Some("583-598"),
};

static VINCENT_1993: CitationMetadata = CitationMetadata {
    cite_key: "vincent1993morphological",
    title: "Morphological Grayscale Reconstruction in Image Analysis: Applications and Efficient Algorithms",
    authors: &["Luc Vincent"],
    year: 1993,
    container: Some("IEEE Transactions on Image Processing"),
    doi: Some("10.1109/83.217222"),
    url: Some("https://doi.org/10.1109/83.217222"),
    pages: Some("176-201"),
};

static HAMPEL_1974: CitationMetadata = CitationMetadata {
    cite_key: "hampel1974influence",
    title: "The Influence Curve and Its Role in Robust Estimation",
    authors: &["Frank R. Hampel"],
    year: 1974,
    container: Some("Journal of the American Statistical Association"),
    doi: Some("10.1080/01621459.1974.10482962"),
    url: Some("https://doi.org/10.1080/01621459.1974.10482962"),
    pages: Some("383-393"),
};

impl GrowRegions {
    /// The settings in pixels and in the channel's normalized values
    /// (0..1), turned so that objects are bright.
    fn params(&self, ctx: &PipelineContext) -> Params {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        let reach = match self.distance_unit {
            SizeUnits::NanoMeter if pixel_size_nm > 0.0 => self.max_distance / pixel_size_nm,
            _ => self.max_distance,
        }
        .max(1.0);
        let bits = ctx.image_meta.nr_of_bits;
        let dark = self.objects_are == Polarity::Dark;
        let rule = match self.stop {
            StopRule::Relative { fraction } => Rule::Relative(fraction.clamp(0.0, 1.0)),
            StopRule::Tolerance { tolerance, unit } => {
                Rule::Tolerance(unit.to_relative(tolerance, bits).max(0.0))
            }
            StopRule::Threshold { level, unit } => {
                let level = unit.to_relative(level, bits);
                Rule::Threshold(if dark { 1.0 - level } else { level })
            }
            StopRule::Noise { noise_factor } => Rule::Noise(noise_factor.max(0.0)),
        };
        Params {
            dark,
            similarity: self.order == GrowthOrder::Similarity,
            rule,
            reach,
            meeting: self.when_regions_meet,
        }
    }
}

/// [`StopRule`] in normalized values, for bright objects.
#[derive(Clone, Copy, Debug)]
enum Rule {
    Relative(f32),
    Tolerance(f32),
    Threshold(f32),
    Noise(f32),
}

#[derive(Clone, Copy, Debug)]
struct Params {
    /// Values are turned (1 - v): dark objects become bright.
    dark: bool,
    similarity: bool,
    rule: Rule,
    /// `max_distance` in pixels.
    reach: f32,
    meeting: Meeting,
}

impl Params {
    /// Whether the background is needed (measured in the ring).
    fn needs_background(&self) -> bool {
        matches!(self.rule, Rule::Relative(_) | Rule::Noise(_))
            || matches!(self.meeting, Meeting::Merge { .. })
    }

    /// Whether a pixel of value `v` may join a region whose reference is
    /// `reference` (Intensity: the region's peak so far; Similarity: the
    /// seed's value), over `background` with `noise`.
    fn accepts(&self, v: f32, reference: f32, background: f32, noise: f32) -> bool {
        match (self.rule, self.similarity) {
            (Rule::Threshold(level), _) => v > level,
            (Rule::Noise(k), _) => v > background + k * noise,
            (Rule::Relative(f), false) => {
                reference > background && v >= background + f * (reference - background)
            }
            (Rule::Relative(f), true) => {
                (v - reference).abs() <= (1.0 - f) * (reference - background).abs()
            }
            (Rule::Tolerance(h), false) => v >= reference - h,
            (Rule::Tolerance(h), true) => (v - reference).abs() <= h,
        }
    }
}

/// `rect` grown by `by` on every side, inside `extent`.
fn grow_rect([x0, y0, x1, y1]: Rect, by: usize, extent: Rect) -> Rect {
    [
        x0.saturating_sub(by).max(extent[0]),
        y0.saturating_sub(by).max(extent[1]),
        (x1 + by).min(extent[2]),
        (y1 + by).min(extent[3]),
    ]
}

fn is_empty([x0, y0, x1, y1]: Rect) -> bool {
    x0 >= x1 || y0 >= y1
}

fn area([x0, y0, x1, y1]: Rect) -> usize {
    x1.saturating_sub(x0) * y1.saturating_sub(y0)
}

fn union(a: Rect, b: Rect) -> Rect {
    if is_empty(a) {
        return b;
    }
    if is_empty(b) {
        return a;
    }
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn intersects(a: Rect, b: Rect) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// Calls `f(x, y)` for each of `object`'s pixels inside `rect`.
fn for_each_pixel_in(object: &Object, rect: Rect, mut f: impl FnMut(usize, usize)) {
    let [x_min, y_min, x_max, y_max] = object.bbox.map(|v| v as usize);
    if x_max < x_min || y_max < y_min {
        return;
    }
    let [rx0, ry0, rx1, ry1] = rect;
    let mask_width = x_max - x_min + 1;
    for y in y_min.max(ry0)..=y_max.min(ry1.saturating_sub(1)) {
        for x in x_min.max(rx0)..=x_max.min(rx1.saturating_sub(1)) {
            if object
                .mask_data
                .get((y - y_min) * mask_width + (x - x_min))
                .is_some_and(|b| *b)
            {
                f(x, y);
            }
        }
    }
}

/// Rectangles by the grid cells they cover, to find the ones meeting
/// another.
struct RectGrid {
    cell: usize,
    cells: HashMap<(usize, usize), Vec<u32>>,
}

impl RectGrid {
    fn new(rects: &[Rect], cell: usize) -> Self {
        let cell = cell.max(1);
        let mut cells: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        for (i, r) in rects.iter().enumerate() {
            if is_empty(*r) {
                continue;
            }
            for cy in r[1] / cell..=(r[3] - 1) / cell {
                for cx in r[0] / cell..=(r[2] - 1) / cell {
                    cells.entry((cx, cy)).or_default().push(i as u32);
                }
            }
        }
        Self { cell, cells }
    }

    /// The indices of the rectangles that may meet `rect` (a superset), in
    /// order.
    fn near(&self, rect: Rect) -> Vec<usize> {
        let mut found = Vec::new();
        if is_empty(rect) {
            return found;
        }
        for cy in rect[1] / self.cell..=(rect[3] - 1) / self.cell {
            for cx in rect[0] / self.cell..=(rect[2] - 1) / self.cell {
                if let Some(ids) = self.cells.get(&(cx, cy)) {
                    found.extend(ids.iter().map(|&i| i as usize));
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }
}

/// Where a seed may grow.
struct SeedBoxes {
    /// The pixels its region may reach (its box grown by the reach).
    reach: Rect,
    /// `reach` with the background ring.
    ring: Rect,
}

/// A piece of work: the regions of `owned` seeds, from the floods of
/// `neighbours` (`owned` among them) - every seed that could take one of
/// their pixels.
struct Unit {
    owned: Vec<usize>,
    neighbours: Vec<usize>,
    /// Where the values are read: every neighbour's ring box.
    window: Rect,
    /// Where pixels are labelled: the owned seeds' reach and one pixel
    /// around it.
    labelled: Rect,
}

/// A unit's values (turned for dark objects) and which pixels the mask
/// allows.
struct Window {
    rect: Rect,
    values: Vec<f32>,
    allowed: Option<BitVec<u64, Lsb0>>,
}

impl Window {
    fn index(&self, x: usize, y: usize) -> usize {
        (y - self.rect[1]) * (self.rect[2] - self.rect[0]) + (x - self.rect[0])
    }
}

/// What a unit gives: its regions' rows, and per owned seed its region's
/// peak and background (for Merge).
#[derive(Default)]
struct UnitOutput {
    runs: Vec<Run>,
    /// (seed, peak, background) per owned seed with a region.
    peaks: Vec<(usize, f32, f32)>,
    /// (seed, other seed, pass value, other's pixel x, y): a region's pixel
    /// next to a pixel labelled for another seed, with the lower of the two
    /// values.
    contacts: Vec<(usize, usize, f32, u32, u32)>,
}

/// A flood's buffers, reused from seed to seed.
#[derive(Default)]
struct FloodScratch {
    state: Vec<u8>,
    queue: BinaryHeap<u64>,
    values: Vec<f32>,
    pixels: Vec<(usize, usize)>,
}

/// A seed's flood: its background.
struct FloodStats {
    background: f32,
}

/// The 8 neighbours.
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

/// The squared distance of a pixel to the nearest seed pixel.
enum SeedDistance {
    /// A one-pixel seed.
    Point(usize, usize),
    /// The nearest seed pixel of every pixel of `rect`.
    Map(NearestFeatures, Rect),
}

impl SeedDistance {
    fn of(seed: &Object, rect: Rect) -> Self {
        let [x0, y0, x1, y1] = seed.bbox.map(|v| v as usize);
        if seed.area <= 1 && x0 == x1 && y0 == y1 {
            return SeedDistance::Point(x0, y0);
        }
        let width = rect[2] - rect[0];
        let height = rect[3] - rect[1];
        let mut is_seed = BitVec::<u64, Lsb0>::repeat(false, width * height);
        for_each_pixel_in(seed, rect, |x, y| {
            is_seed.set((y - rect[1]) * width + (x - rect[0]), true);
        });
        let map = nearest_features(width, height, |x, y| is_seed[y * width + x]);
        SeedDistance::Map(map, rect)
    }

    fn squared(&self, x: usize, y: usize) -> f32 {
        let d2 = |dx: i64, dy: i64| (dx * dx + dy * dy) as f32;
        match self {
            SeedDistance::Point(sx, sy) => d2(x as i64 - *sx as i64, y as i64 - *sy as i64),
            SeedDistance::Map(map, rect) => {
                let w = rect[2] - rect[0];
                match map.get((y - rect[1]) * w + (x - rect[0])) {
                    Some((r, c)) => {
                        d2(x as i64 - rect[0] as i64 - c, y as i64 - rect[1] as i64 - r)
                    }
                    None => f32::INFINITY,
                }
            }
        }
    }
}

/// The inputs of one Grow Regions run.
struct Grower<'a> {
    channel: &'a ChannelWindow,
    cache: &'a GlobalPipelineCache,
    seeds: &'a [&'a Object],
    masks: Option<(&'a [&'a Object], RectGrid)>,
    boxes: Vec<SeedBoxes>,
    params: Params,
}

impl<'a> Grower<'a> {
    fn new(
        channel: &'a ChannelWindow,
        cache: &'a GlobalPipelineCache,
        seeds: &'a [&'a Object],
        masks: Option<&'a [&'a Object]>,
        params: Params,
    ) -> Self {
        let extent = channel.extent;
        let reach = params.reach.ceil() as usize;
        let boxes = seeds
            .iter()
            .map(|seed| {
                let [x0, y0, x1, y1] = seed.bbox.map(|v| v as usize);
                let own = [x0, y0, x1 + 1, y1 + 1];
                // Too large to grow (see `MAX_SEED_SIDE`): its region is
                // the seed (added in `regions`).
                let too_large = x1 >= x0 + MAX_SEED_SIDE || y1 >= y0 + MAX_SEED_SIDE;
                if x1 < x0 || y1 < y0 || !intersects(own, extent) || too_large {
                    let none = [0, 0, 0, 0];
                    return SeedBoxes {
                        reach: none,
                        ring: none,
                    };
                }
                SeedBoxes {
                    reach: grow_rect(own, reach, extent),
                    ring: grow_rect(own, reach + RING, extent),
                }
            })
            .collect();
        let masks = masks.map(|masks| {
            let rects: Vec<Rect> = masks
                .iter()
                .map(|m| {
                    let [x0, y0, x1, y1] = m.bbox.map(|v| v as usize);
                    if x1 < x0 || y1 < y0 {
                        [0, 0, 0, 0]
                    } else {
                        [x0, y0, x1 + 1, y1 + 1]
                    }
                })
                .collect();
            (masks, RectGrid::new(&rects, 256))
        });
        Self {
            channel,
            cache,
            seeds,
            masks,
            boxes,
            params,
        }
    }

    /// Per seed, its region (`None`: none). Merged regions are given to
    /// the first of their seeds. `one_piece` and `block`: see
    /// [`ONE_PIECE_PIXELS`], [`BLOCK`] - the result is the same for any.
    fn regions(
        &self,
        one_piece: usize,
        block: usize,
    ) -> Result<Vec<Option<RunRegion>>, InternalErrors> {
        let units = self.units(one_piece, block);
        let (small, large): (Vec<&Unit>, Vec<&Unit>) = units
            .iter()
            .partition(|u| area(u.window) <= SMALL_UNIT_PIXELS);
        let mut outputs: Vec<UnitOutput> = small
            .par_iter()
            .map(|u| self.grow_unit(u))
            .collect::<Result<_, _>>()?;
        if let Some(largest) = large.iter().map(|u| area(u.window)).max() {
            let available = crate::memory::available_memory_bytes();
            let bytes = largest as u64 * UNIT_BYTES_PER_PIXEL;
            let at_once =
                ((available / 4 / bytes.max(1)) as usize).clamp(1, rayon::current_num_threads());
            for batch in large.chunks(at_once) {
                outputs.extend(
                    batch
                        .par_iter()
                        .map(|u| self.grow_unit(u))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        let count = self.seeds.len();
        let mut runs: Vec<Run> = Vec::new();
        for output in &mut outputs {
            runs.append(&mut output.runs);
        }
        // The seeds too large to grow keep their own pixels.
        for (i, seed) in self.seeds.iter().enumerate() {
            let [x0, y0, x1, y1] = seed.bbox.map(|v| v as usize);
            if x1 >= x0 + MAX_SEED_SIDE || y1 >= y0 + MAX_SEED_SIDE {
                let width = x1 - x0 + 1;
                for y in y0..=y1 {
                    let row = &seed.mask_data[(y - y0) * width..(y - y0 + 1) * width];
                    push_row_runs(
                        &mut runs,
                        y as u32,
                        x0 as u32,
                        row.iter()
                            .by_vals()
                            .map(|b| if b { i as u32 + 1 } else { 0 }),
                    );
                }
            }
        }
        let Meeting::Merge { max_valley } = self.params.meeting else {
            return Ok(regions_from_runs(runs, count));
        };
        let regions = regions_from_runs(runs.clone(), count);
        let roots = self.merged(&regions, &outputs, max_valley);
        if roots.iter().enumerate().all(|(i, r)| *r == i) {
            return Ok(regions);
        }
        drop(regions);
        for run in &mut runs {
            run.0 = roots[run.0 as usize - 1] as u32 + 1;
        }
        Ok(regions_from_runs(runs, count))
    }

    /// Per seed the first seed of the regions it is merged with: regions
    /// meeting over a valley at most `max_valley` of the dimmer one's
    /// contrast deep.
    fn merged(
        &self,
        regions: &[Option<RunRegion>],
        outputs: &[UnitOutput],
        max_valley: f32,
    ) -> Vec<usize> {
        let mut peak = vec![(f32::NAN, 0.0f32); regions.len()];
        for output in outputs {
            for &(seed, p, bg) in &output.peaks {
                peak[seed] = (p, bg);
            }
        }
        // The highest pass between two regions, over the contacts whose
        // other pixel really is in the other region (a labelled pixel may
        // not be connected to its seed).
        let mut saddles: BTreeMap<(usize, usize), f32> = BTreeMap::new();
        for output in outputs {
            for &(a, b, pass, x, y) in &output.contacts {
                let Some(other) = &regions[b] else { continue };
                let [x0, y0, x1, y1] = other.bbox;
                if x < x0 || x > x1 || y < y0 || y > y1 {
                    continue;
                }
                let w = (x1 - x0 + 1) as usize;
                if !other.mask[(y - y0) as usize * w + (x - x0) as usize] {
                    continue;
                }
                let key = (a.min(b), a.max(b));
                let s = saddles.entry(key).or_insert(f32::NEG_INFINITY);
                *s = s.max(pass);
            }
        }
        let mut parent: Vec<usize> = (0..regions.len()).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for ((a, b), saddle) in saddles {
            let ((pa, ba), (pb, bb)) = (peak[a], peak[b]);
            if !pa.is_finite() || !pb.is_finite() {
                continue;
            }
            let (dimmer, background) = if pa <= pb { (pa, ba) } else { (pb, bb) };
            let contrast = dimmer - background;
            if contrast <= 0.0 || dimmer - saddle <= max_valley * contrast {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                // The first seed stands for the merged region.
                let (lo, hi) = (ra.min(rb), ra.max(rb));
                parent[hi] = lo;
            }
        }
        (0..regions.len()).map(|i| find(&mut parent, i)).collect()
    }

    /// The units: each seed alone (Overlap), or groups of seeds whose
    /// reach meets - one unit if its window is small enough, else one per
    /// block of the seeds lying in it.
    fn units(&self, one_piece: usize, block: usize) -> Vec<Unit> {
        let valid: Vec<usize> = (0..self.seeds.len())
            .filter(|&i| !is_empty(self.boxes[i].reach))
            .collect();
        if self.params.meeting == Meeting::Overlap {
            return valid
                .into_iter()
                .map(|i| Unit {
                    owned: vec![i],
                    neighbours: vec![i],
                    window: self.boxes[i].ring,
                    labelled: self.boxes[i].reach,
                })
                .collect();
        }
        let extent = self.channel.extent;
        // Reach grown by a pixel: regions this close may touch.
        let touch: Vec<Rect> = self
            .boxes
            .iter()
            .map(|b| {
                if is_empty(b.reach) {
                    b.reach
                } else {
                    grow_rect(b.reach, 1, extent)
                }
            })
            .collect();
        let reach: Vec<Rect> = self.boxes.iter().map(|b| b.reach).collect();
        let cell = (self.params.reach.ceil() as usize * 4).clamp(16, 1024);
        let touch_grid = RectGrid::new(&touch, cell);
        let mut parent: Vec<usize> = (0..self.seeds.len()).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for &i in &valid {
            for j in touch_grid.near(touch[i]) {
                if j > i && intersects(touch[i], touch[j]) {
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    if ri != rj {
                        parent[ri.max(rj)] = ri.min(rj);
                    }
                }
            }
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &i in &valid {
            groups.entry(find(&mut parent, i)).or_default().push(i);
        }
        let reach_grid = RectGrid::new(&reach, cell);
        let mut units = Vec::new();
        for members in groups.into_values() {
            let window = members
                .iter()
                .fold([0, 0, 0, 0], |w, &i| union(w, self.boxes[i].ring));
            if area(window) <= one_piece {
                let labelled = members
                    .iter()
                    .fold([0, 0, 0, 0], |w, &i| union(w, touch[i]));
                units.push(Unit {
                    owned: members.clone(),
                    neighbours: members,
                    window,
                    labelled,
                });
                continue;
            }
            // By the block the seed's box starts in.
            let mut blocks: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for &i in &members {
                let [sx, sy, ..] = self.seeds[i].bbox.map(|v| v as usize);
                blocks.entry((sy / block, sx / block)).or_default().push(i);
            }
            let root = find(&mut parent, members[0]);
            for owned in blocks.into_values() {
                let labelled = owned.iter().fold([0, 0, 0, 0], |w, &i| union(w, touch[i]));
                let neighbours: Vec<usize> = reach_grid
                    .near(labelled)
                    .into_iter()
                    .filter(|&j| intersects(reach[j], labelled) && find(&mut parent, j) == root)
                    .collect();
                let window = neighbours
                    .iter()
                    .fold([0, 0, 0, 0], |w, &j| union(w, self.boxes[j].ring));
                units.push(Unit {
                    owned,
                    neighbours,
                    window,
                    labelled,
                });
            }
        }
        units
    }

    /// Reads a unit's window: the values (turned for dark objects) and
    /// which pixels the mask allows.
    fn window(&self, rect: Rect) -> Result<Window, InternalErrors> {
        let mut values = self.channel.read(self.cache, rect)?;
        if self.params.dark {
            for v in &mut values {
                *v = 1.0 - *v;
            }
        }
        let allowed = self.masks.as_ref().map(|(masks, grid)| {
            let width = rect[2] - rect[0];
            let mut allowed = BitVec::<u64, Lsb0>::repeat(false, values.len());
            for i in grid.near(rect) {
                for_each_pixel_in(masks[i], rect, |x, y| {
                    allowed.set((y - rect[1]) * width + (x - rect[0]), true);
                });
            }
            allowed
        });
        Ok(Window {
            rect,
            values,
            allowed,
        })
    }

    /// Grows one unit.
    fn grow_unit(&self, unit: &Unit) -> Result<UnitOutput, InternalErrors> {
        let window = self.window(unit.window)?;
        let mut output = UnitOutput::default();
        if self.params.meeting == Meeting::Overlap {
            let i = unit.owned[0];
            let reach = self.boxes[i].reach;
            let width = reach[2] - reach[0];
            let mut grown = BitVec::<u64, Lsb0>::repeat(false, area(reach));
            self.flood(i, &window, &mut FloodScratch::default(), |x, y, _, _| {
                grown.set((y - reach[1]) * width + (x - reach[0]), true);
            });
            push_mask_runs(&mut output.runs, i, reach, &grown);
            return Ok(output);
        }

        // The best seed of every labelled pixel: score, squared distance,
        // seed + 1 (0: none).
        let labelled = unit.labelled;
        let lw = labelled[2] - labelled[0];
        let n = area(labelled);
        // Together: one cache line per pixel.
        let mut best = vec![(f32::NEG_INFINITY, f32::INFINITY, 0u32); n];
        let mut scratch = FloodScratch::default();
        let mut backgrounds: HashMap<usize, f32> = HashMap::new();
        for &j in &unit.neighbours {
            let stats = self.flood(j, &window, &mut scratch, |x, y, s, d2| {
                if x < labelled[0] || x >= labelled[2] || y < labelled[1] || y >= labelled[3] {
                    return;
                }
                let at = (y - labelled[1]) * lw + (x - labelled[0]);
                // Seeds come in order: on a full tie the earlier keeps it.
                let b = &mut best[at];
                if s > b.0 || (s == b.0 && d2 < b.1) {
                    *b = (s, d2, j as u32 + 1);
                }
            });
            backgrounds.insert(j, stats.background);
        }

        let merge = matches!(self.params.meeting, Meeting::Merge { .. });
        for &i in &unit.owned {
            let reach = self.boxes[i].reach;
            let rw = reach[2] - reach[0];
            let own = i as u32 + 1;
            let label_at = |x: usize, y: usize| best[(y - labelled[1]) * lw + (x - labelled[0])].2;
            // The pixels labelled for the seed and connected to it.
            let mut region = BitVec::<u64, Lsb0>::repeat(false, area(reach));
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for_each_pixel_in(self.seeds[i], reach, |x, y| {
                if label_at(x, y) == own {
                    region.set((y - reach[1]) * rw + (x - reach[0]), true);
                    stack.push((x, y));
                }
            });
            let mut peak = f32::NEG_INFINITY;
            while let Some((x, y)) = stack.pop() {
                let v = window.values[window.index(x, y)];
                if v.is_finite() {
                    peak = peak.max(v);
                }
                for (dx, dy) in STEPS {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < labelled[0] as isize
                        || ny < labelled[1] as isize
                        || nx >= labelled[2] as isize
                        || ny >= labelled[3] as isize
                    {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let l = label_at(nx, ny);
                    if l == own {
                        let at = (ny - reach[1]) * rw + (nx - reach[0]);
                        if !region[at] {
                            region.set(at, true);
                            stack.push((nx, ny));
                        }
                    } else if merge && l != 0 {
                        let w = window.values[window.index(nx, ny)];
                        let pass = if v.is_finite() && w.is_finite() {
                            v.min(w)
                        } else {
                            f32::NEG_INFINITY
                        };
                        output
                            .contacts
                            .push((i, l as usize - 1, pass, nx as u32, ny as u32));
                    }
                }
            }
            if region.any() {
                push_mask_runs(&mut output.runs, i, reach, &region);
                if merge {
                    let bg = backgrounds.get(&i).copied().unwrap_or(0.0);
                    output.peaks.push((i, peak, bg));
                }
            }
        }
        Ok(output)
    }

    /// Floods seed `i` inside its reach: calls `take(x, y, score, d²)` for
    /// each pixel it reaches and the stop rule lets in (its own pixels
    /// first, score +∞), in the order they join.
    fn flood(
        &self,
        i: usize,
        window: &Window,
        scratch: &mut FloodScratch,
        mut take: impl FnMut(usize, usize, f32, f32),
    ) -> FloodStats {
        let FloodScratch {
            state,
            queue,
            values: ring_values,
            pixels: seeds_px,
        } = scratch;
        let seed = self.seeds[i];
        let SeedBoxes { reach, ring } = self.boxes[i];
        let params = &self.params;
        let reach2 = params.reach * params.reach;
        let distance = SeedDistance::of(seed, ring);

        // The seed's values: its brightest and its median.
        let seed_values = &mut *ring_values;
        seed_values.clear();
        for_each_pixel_in(seed, reach, |x, y| {
            let v = window.values[window.index(x, y)];
            if v.is_finite() {
                seed_values.push(v);
            }
        });
        let brightest = seed_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        // The region's peak: the brightest pixel connected to the seed
        // through pixels at least as bright as the seed - its own hilltop
        // (a seed beside the peak finds it), never a brighter neighbour's
        // behind a valley.
        let mut peak = brightest;
        let seed_value = if params.similarity && !seed_values.is_empty() {
            median_in_place(seed_values)
        } else {
            f32::NAN
        };

        // The background and noise: the ring just beyond the reach.
        let (background, noise) = if params.needs_background() {
            let outer = params.reach + RING as f32;
            let outer2 = outer * outer;
            ring_values.clear();
            let mut push = |x: usize, y: usize| {
                let v = window.values[window.index(x, y)];
                if v.is_finite() {
                    ring_values.push(v);
                }
            };
            match distance {
                // Only the ring's pixels of each row: |dx| within
                // (reach² - dy², outer² - dy²].
                SeedDistance::Point(sx, sy) => {
                    for y in ring[1]..ring[3] {
                        let dy = y as i64 - sy as i64;
                        let dy2 = (dy * dy) as f32;
                        if dy2 > outer2 {
                            continue;
                        }
                        let max_dx = (outer2 - dy2).sqrt() as i64;
                        let x_from = (sx as i64 - max_dx - 1).max(ring[0] as i64) as usize;
                        let x_to = ((sx as i64 + max_dx + 2) as usize).min(ring[2]);
                        for x in x_from..x_to {
                            let dx = x as i64 - sx as i64;
                            let d2 = (dx * dx) as f32 + dy2;
                            if d2 > reach2 && d2 <= outer2 {
                                push(x, y);
                            }
                        }
                    }
                }
                SeedDistance::Map(..) => {
                    for y in ring[1]..ring[3] {
                        for x in ring[0]..ring[2] {
                            let d2 = distance.squared(x, y);
                            if d2 > reach2 && d2 <= outer2 {
                                push(x, y);
                            }
                        }
                    }
                }
            }
            // Without pixels around it (the seed fills the image): 0.
            background_and_noise_in_place(ring_values).unwrap_or((0.0, 0.0))
        } else {
            (0.0, 0.0)
        };

        // The reach with a border of one pixel (marked done), row-major:
        // a neighbour is an offset, never outside.
        let [rx0, ry0, rx1, ry1] = reach;
        let width = rx1 - rx0 + 2;
        let height = ry1 - ry0 + 2;
        // 0: not seen, 1: queued, 2: done.
        state.clear();
        state.resize(width * height, 0);
        for x in 0..width {
            state[x] = 2;
            state[(height - 1) * width + x] = 2;
        }
        for y in 0..height {
            state[y * width] = 2;
            state[y * width + width - 1] = 2;
        }
        let local = |x: usize, y: usize| (y - ry0 + 1) * width + (x - rx0 + 1);
        let stride = window.rect[2] - window.rect[0];
        // Pixel `at` of the grid: its image coordinates and window index.
        let at_xy = |at: usize| (rx0 + at % width - 1, ry0 + at / width - 1);
        let offsets: [(isize, isize, isize, isize); 8] =
            STEPS.map(|(dx, dy)| (dy * width as isize + dx, dy * stride as isize + dx, dx, dy));
        // Higher score first, then the lower pixel: the score's bits turned
        // so that unsigned order is its order, above the pixel's complement.
        let key = |score: f32, at: usize| {
            let bits = score.to_bits();
            let ordered = if bits >> 31 == 1 {
                !bits
            } else {
                bits | 0x8000_0000
            };
            ((ordered as u64) << 32) | (u32::MAX - at as u32) as u64
        };
        let unkey = |key: u64| {
            let ordered = (key >> 32) as u32;
            let bits = if ordered >> 31 == 1 {
                ordered & 0x7FFF_FFFF
            } else {
                !ordered
            };
            (f32::from_bits(bits), (u32::MAX - key as u32) as usize)
        };
        queue.clear();
        seeds_px.clear();
        for_each_pixel_in(seed, reach, |x, y| seeds_px.push((x, y)));
        for &(x, y) in seeds_px.iter() {
            state[local(x, y)] = 2;
        }
        let expand = |at: usize, s: f32, state: &mut [u8], queue: &mut BinaryHeap<u64>| {
            let (x, y) = at_xy(at);
            let wi = window.index(x, y) as isize;
            for (grid_step, window_step, dx, dy) in offsets {
                let n = (at as isize + grid_step) as usize;
                if state[n] != 0 {
                    continue;
                }
                let nwi = (wi + window_step) as usize;
                let v = window.values[nwi];
                let in_mask = window.allowed.as_ref().is_none_or(|m| m[nwi]);
                let far = || {
                    let (nx, ny) = ((x as isize + dx) as usize, (y as isize + dy) as usize);
                    distance.squared(nx, ny) > reach2
                };
                if !v.is_finite() || !in_mask || far() {
                    state[n] = 2;
                    continue;
                }
                // The best path's worst step: the lowest value
                // (Intensity) or the largest difference to the seed
                // (Similarity), as a score where higher is better.
                let own = if params.similarity {
                    -(v - seed_value).abs()
                } else {
                    v
                };
                state[n] = 1;
                queue.push(key(own.min(s), n));
            }
        };
        for &(x, y) in seeds_px.iter() {
            take(x, y, f32::INFINITY, 0.0);
            expand(local(x, y), f32::INFINITY, state, queue);
        }
        let can_grow = !params.similarity || seed_value.is_finite();
        while can_grow && let Some(top) = queue.pop() {
            let (s, at) = unkey(top);
            state[at] = 2;
            let (x, y) = at_xy(at);
            let v = window.values[window.index(x, y)];
            let reference = if params.similarity { seed_value } else { peak };
            if !params.accepts(v, reference, background, noise) {
                continue;
            }
            if s >= brightest {
                peak = peak.max(v);
            }
            take(x, y, s, distance.squared(x, y));
            expand(at, s, state, queue);
        }
        FloodStats { background }
    }
}

/// Appends the rows of `mask` (over `rect`, row-major) as runs of seed
/// `i`.
fn push_mask_runs(runs: &mut Vec<Run>, i: usize, rect: Rect, mask: &BitVec<u64, Lsb0>) {
    let width = rect[2] - rect[0];
    let label = i as u32 + 1;
    for y in rect[1]..rect[3] {
        let row = &mask[(y - rect[1]) * width..(y - rect[1] + 1) * width];
        if row.not_any() {
            continue;
        }
        push_row_runs(
            runs,
            y as u32,
            rect[0] as u32,
            row.iter().by_vals().map(|b| if b { label } else { 0 }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::scope::Scope;
    use evanalyzer_cfg::core_types::ImageTile;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    const SIDE: usize = 64;
    const SEED: ObjectClass = ObjectClass::Global(1);
    const MASK: ObjectClass = ObjectClass::Global(2);
    const REGION: ObjectClass = ObjectClass::Global(3);

    /// A one-pixel seed at (x, y).
    fn point(id: u128, x: u32, y: u32) -> Object {
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x, y, x, y],
            area: 1,
            mask_data: BitVec::repeat(true, 1),
            sum_x: x as u64,
            sum_y: y as u64,
            ..Default::default()
        });
        object.add_object_class(SEED, Scope::Global).unwrap();
        object
    }

    /// A rectangle object `[x0, y0, x1, y1]` (inclusive) of `class`.
    fn rect(id: u128, [x0, y0, x1, y1]: [u32; 4], class: ObjectClass) -> Object {
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let mut object = Object::new(ObjectInit {
            id: ObjectId(id),
            bbox: [x0, y0, x1, y1],
            area: (w * h) as usize,
            mask_data: BitVec::repeat(true, (w * h) as usize),
            ..Default::default()
        });
        object.add_object_class(class, Scope::Global).unwrap();
        object
    }

    /// A cache with channel 0 = `pixel(x, y)` in tiles of `tile` px, and
    /// `objects`.
    fn cache_with(
        pixel: impl Fn(usize, usize) -> f32,
        tile: usize,
        objects: Vec<Object>,
    ) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        for ty in (0..SIDE).step_by(tile) {
            for tx in (0..SIDE).step_by(tile) {
                let (w, h) = (tile.min(SIDE - tx), tile.min(SIDE - ty));
                let data: Vec<f32> = (0..w * h).map(|p| pixel(tx + p % w, ty + p / w)).collect();
                let image = Arc::new(ImageContainer::F32Gray(ManagedImage {
                    data: Image::<f32, 1>::new(
                        ImageSize {
                            width: w,
                            height: h,
                        },
                        data,
                    )
                    .unwrap(),
                    tile_offset: Point2d { x: tx, y: ty },
                    plane: None,
                    dimensionality: Default::default(),
                }));
                cache.add_to_channel_cache(
                    image,
                    0,
                    ImageTile {
                        offset_x: tx,
                        offset_y: ty,
                        width: w,
                        height: h,
                    },
                );
            }
        }
        for object in objects {
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn command() -> GrowRegions {
        GrowRegions {
            seeds: SEED,
            growth_channel: ImageChannelIdx(0),
            objects_are: Polarity::Bright,
            order: GrowthOrder::Intensity,
            stop: StopRule::Relative { fraction: 0.5 },
            max_distance: 10.0,
            distance_unit: SizeUnits::Pixels,
            when_regions_meet: Meeting::Separate,
            mask: ObjectClass::Unset,
            output_class: REGION,
            exclude_at_edges: false,
        }
    }

    fn ctx() -> PipelineContext {
        let meta = PipelineImageMeta {
            full_image_width: ImageSize {
                width: SIDE,
                height: SIDE,
            },
            nr_of_bits: 16,
            ..Default::default()
        };
        PipelineContext::without_image(std::path::PathBuf::new(), meta).unwrap()
    }

    fn run(command: &GrowRegions, cache: &mut GlobalPipelineCache) -> Vec<Object> {
        command.execute(&mut ctx(), cache).unwrap();
        let mut regions: Vec<Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&REGION, Scope::Global))
            .cloned()
            .collect();
        regions.sort_by(|a, b| a.position_cmp(b));
        regions
    }

    fn contains(object: &Object, x: u32, y: u32) -> bool {
        let [x0, y0, x1, y1] = object.bbox;
        if x < x0 || x > x1 || y < y0 || y > y1 {
            return false;
        }
        let w = (x1 - x0 + 1) as usize;
        object.mask_data[(y - y0) as usize * w + (x - x0) as usize]
    }

    /// A Gaussian spot of height `h` and sigma `s` at (cx, cy) over `bg`.
    fn spot(x: usize, y: usize, (cx, cy): (f32, f32), h: f32, s: f32) -> f32 {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        h * (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
    }

    /// The pixels of a spot at or above half its height: within
    /// σ·√(2 ln 2) of its centre.
    fn half_max_area(s: f32) -> usize {
        let r2 = 2.0 * 2f32.ln() * s * s;
        let mut n = 0;
        for y in -20i32..=20 {
            for x in -20i32..=20 {
                if ((x * x + y * y) as f32) <= r2 {
                    n += 1;
                }
            }
        }
        n
    }

    #[test]
    fn a_spot_is_cut_at_half_its_height() {
        let mut cache = cache_with(
            |x, y| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 2.5),
            SIDE,
            vec![point(1, 32, 32)],
        );
        let regions = run(&command(), &mut cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, half_max_area(2.5));
    }

    #[test]
    fn dim_and_bright_spots_keep_the_same_shape() {
        let image = |x: usize, y: usize| {
            0.05 + spot(x, y, (16.0, 32.0), 0.05, 2.5) + spot(x, y, (48.0, 32.0), 0.8, 2.5)
        };
        let mut cache = cache_with(image, SIDE, vec![point(1, 16, 32), point(2, 48, 32)]);
        let regions = run(&command(), &mut cache);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].area, regions[1].area);
        assert_eq!(regions[0].area, half_max_area(2.5));
    }

    #[test]
    fn a_seed_off_the_peak_finds_the_peak() {
        // Seed one pixel beside the peak: the level comes from the peak.
        let mut cache = cache_with(
            |x, y| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 2.5),
            SIDE,
            vec![point(1, 33, 32)],
        );
        let regions = run(&command(), &mut cache);
        assert_eq!(regions[0].area, half_max_area(2.5));
    }

    #[test]
    fn touching_spots_are_split_on_the_valley() {
        // Two spots 8 px apart, the right one brighter: the border lies in
        // the valley (nearer the dimmer spot), not halfway.
        let image = |x: usize, y: usize| {
            spot(x, y, (28.0, 32.0), 0.4, 3.0) + spot(x, y, (36.0, 32.0), 0.8, 3.0)
        };
        let mut command = command();
        command.stop = StopRule::Relative { fraction: 0.1 };
        let mut cache = cache_with(image, SIDE, vec![point(1, 28, 32), point(2, 36, 32)]);
        let regions = run(&command, &mut cache);
        assert_eq!(regions.len(), 2);
        // No pixel in both.
        for y in 0..SIDE as u32 {
            for x in 0..SIDE as u32 {
                assert!(!(contains(&regions[0], x, y) && contains(&regions[1], x, y)));
            }
        }
        // The valley on the centre row.
        let valley = (29..=35)
            .min_by(|&a, &b| image(a, 32).total_cmp(&image(b, 32)))
            .unwrap() as u32;
        let of = |x: u32| regions.iter().find(|r| contains(r, x, 32)).unwrap();
        let (dim, bright) = (of(28), of(36));
        assert!(!std::ptr::eq(dim, bright));
        assert!(contains(dim, valley - 1, 32));
        assert!(contains(bright, valley + 1, 32));
        // Nearer the dimmer spot than halfway (32).
        assert!(valley < 32);
    }

    #[test]
    fn a_bright_neighbour_does_not_shrink_a_dim_spot() {
        // A dim spot beside a bright one: the dim spot's level comes from
        // its own peak, so its far side is as without the neighbour.
        let dim = |x: usize, y: usize| 0.05 + spot(x, y, (24.0, 32.0), 0.2, 2.5);
        let both = |x: usize, y: usize| dim(x, y) + spot(x, y, (32.0, 32.0), 0.8, 2.5);
        let left_of = |r: &Object, at: u32| {
            (0..SIDE as u32)
                .flat_map(|y| (0..=at).map(move |x| (x, y)))
                .filter(|&(x, y)| contains(r, x, y))
                .count()
        };
        let mut alone = cache_with(dim, SIDE, vec![point(1, 24, 32)]);
        let alone = run(&command(), &mut alone);
        let mut cache = cache_with(both, SIDE, vec![point(1, 24, 32), point(2, 32, 32)]);
        let regions = run(&command(), &mut cache);
        let with = regions.iter().find(|r| contains(r, 24, 32)).unwrap();
        assert_eq!(left_of(with, 24), left_of(&alone[0], 24));
    }

    #[test]
    fn overlapping_regions_each_grow_as_if_alone() {
        let image = |x: usize, y: usize| {
            spot(x, y, (28.0, 32.0), 0.6, 3.0) + spot(x, y, (36.0, 32.0), 0.6, 3.0)
        };
        let mut command = command();
        command.stop = StopRule::Threshold {
            level: 0.05,
            unit: PixelUnits::Relative,
        };
        command.when_regions_meet = Meeting::Overlap;
        let mut cache = cache_with(image, SIDE, vec![point(1, 28, 32), point(2, 36, 32)]);
        let both = run(&command, &mut cache);
        assert_eq!(both.len(), 2);
        assert!(contains(&both[0], 32, 32) && contains(&both[1], 32, 32));
        let mut alone = cache_with(image, SIDE, vec![point(1, 28, 32)]);
        let alone = run(&command, &mut alone);
        assert_eq!(alone[0].area, both[0].area);
    }

    #[test]
    fn merge_joins_maxima_on_one_object_but_not_separate_spots() {
        let mut command = command();
        command.stop = StopRule::Relative { fraction: 0.3 };
        command.when_regions_meet = Meeting::Merge { max_valley: 0.2 };
        // Two maxima 4 px apart on one elongated object: a shallow dip.
        let blob = |x: usize, y: usize| {
            0.05 + spot(x, y, (30.0, 32.0), 0.5, 2.5) + spot(x, y, (34.0, 32.0), 0.5, 2.5)
        };
        let mut cache = cache_with(blob, SIDE, vec![point(1, 30, 32), point(2, 34, 32)]);
        let merged = run(&command, &mut cache);
        assert_eq!(merged.len(), 1);
        // Two spots 9 px apart: they touch, but over a deep valley.
        let pair = |x: usize, y: usize| {
            0.05 + spot(x, y, (27.0, 32.0), 0.5, 2.5) + spot(x, y, (36.0, 32.0), 0.5, 2.5)
        };
        let mut cache = cache_with(pair, SIDE, vec![point(1, 27, 32), point(2, 36, 32)]);
        assert_eq!(run(&command, &mut cache).len(), 2);
    }

    #[test]
    fn similarity_fills_an_even_object_but_not_a_bright_speck_beside_it() {
        // A flat nucleus (0.4) on 0.1, a bright speck (0.9) touching it.
        let image = |x: usize, y: usize| {
            if (40..=42).contains(&x) && (30..=32).contains(&y) {
                0.9
            } else if (24..=39).contains(&x) && (24..=39).contains(&y) {
                0.4
            } else {
                0.1
            }
        };
        let mut command = command();
        command.order = GrowthOrder::Similarity;
        command.max_distance = 20.0;
        let nucleus = rect(1, [30, 30, 32, 32], SEED);
        let mut cache = cache_with(image, SIDE, vec![nucleus]);
        let regions = run(&command, &mut cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 16 * 16);
        assert!(!contains(&regions[0], 40, 31));
    }

    #[test]
    fn dark_objects_grow_like_bright_ones_inverted() {
        let bright = |x: usize, y: usize| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 2.5);
        let mut command = command();
        command.objects_are = Polarity::Dark;
        let mut cache = cache_with(|x, y| 1.0 - bright(x, y), SIDE, vec![point(1, 32, 32)]);
        let regions = run(&command, &mut cache);
        assert_eq!(regions[0].area, half_max_area(2.5));
    }

    #[test]
    fn regions_stop_at_the_max_distance_and_the_mask() {
        // A wide plateau: only the distance stops it.
        let image = |x: usize, y: usize| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 30.0);
        let mut command = command();
        command.stop = StopRule::Threshold {
            level: 0.0,
            unit: PixelUnits::Relative,
        };
        command.max_distance = 3.0;
        let mut cache = cache_with(image, SIDE, vec![point(1, 32, 32)]);
        let regions = run(&command, &mut cache);
        // The pixels within 3 px: 29.
        assert_eq!(regions[0].area, 29);
        command.mask = MASK;
        let mut cache = cache_with(
            image,
            SIDE,
            vec![point(1, 32, 32), rect(2, [32, 0, 63, 63], MASK)],
        );
        let regions = run(&command, &mut cache);
        // The half of them at x >= 32: 7 + 5 + 5 + 1.
        assert_eq!(regions[0].area, 18);
        assert!(regions[0].bbox[0] >= 32);
    }

    #[test]
    fn a_seed_that_grows_nothing_keeps_its_pixels() {
        let mut cache = cache_with(|_, _| 0.2, SIDE, vec![point(1, 10, 10)]);
        let regions = run(&command(), &mut cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].area, 1);
    }

    #[test]
    fn a_seed_too_large_to_grow_keeps_its_pixels() {
        // 41 px across (the tests' limit is 40), on a bright image.
        let mut cache = cache_with(
            |_, _| 0.8,
            SIDE,
            vec![rect(1, [10, 10, 50, 12], SEED), point(2, 30, 40)],
        );
        let mut command = command();
        command.stop = StopRule::Threshold {
            level: 0.0,
            unit: PixelUnits::Relative,
        };
        let regions = run(&command, &mut cache);
        let large = regions.iter().find(|r| contains(r, 10, 10)).unwrap();
        assert_eq!((large.bbox, large.area), ([10, 10, 50, 12], 41 * 3));
        // The small one grows (here up to its 10 px reach).
        let small = regions.iter().find(|r| contains(r, 30, 40)).unwrap();
        assert!(small.area > 100);
    }

    #[test]
    fn regions_at_the_border_can_be_dropped() {
        let image = |x: usize, y: usize| 0.1 + spot(x, y, (1.0, 32.0), 0.5, 2.5);
        let mut command = command();
        command.exclude_at_edges = true;
        let mut cache = cache_with(image, SIDE, vec![point(1, 1, 32), point(2, 40, 40)]);
        let regions = run(&command, &mut cache);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].bbox[0], 40);
    }

    #[test]
    fn noise_and_tolerance_stop_rules() {
        let mut cache = cache_with(
            |x, y| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 2.5),
            SIDE,
            vec![point(1, 32, 32)],
        );
        let mut command = command();
        command.stop = StopRule::Tolerance {
            tolerance: 0.25,
            unit: PixelUnits::Relative,
        };
        // 0.25 below a peak of 0.6: the half maximum again.
        assert_eq!(run(&command, &mut cache)[0].area, half_max_area(2.5));
        // A flat background has no noise: everything above it joins, up to
        // the max distance.
        let mut cache = cache_with(
            |x, y| 0.1 + spot(x, y, (32.0, 32.0), 0.5, 2.5),
            SIDE,
            vec![point(1, 32, 32)],
        );
        command.stop = StopRule::Noise { noise_factor: 3.0 };
        command.max_distance = 6.0;
        let area = run(&command, &mut cache)[0].area;
        assert!(area > half_max_area(2.5) && area <= 113, "{area}");
    }

    /// Many seeds in tiles: the regions are the same however the seeds are
    /// split into units (one piece, or blocks with their margin).
    #[test]
    fn the_result_does_not_depend_on_the_split() {
        let centres: Vec<(u32, u32)> = (0..7)
            .flat_map(|j| (0..7).map(move |i| (5 + i * 9 + (j % 2) * 3, 4 + j * 9)))
            .filter(|&(x, y)| x < SIDE as u32 && y < SIDE as u32)
            .collect();
        let image = |x: usize, y: usize| {
            let mut v = 0.05 + 0.01 * (((x * 7 + y * 13) % 5) as f32);
            for (k, &(cx, cy)) in centres.iter().enumerate() {
                let h = 0.2 + 0.1 * (k % 5) as f32;
                v += spot(x, y, (cx as f32, cy as f32), h, 2.0);
            }
            v
        };
        for meeting in [
            Meeting::Separate,
            Meeting::Merge { max_valley: 0.3 },
            Meeting::Overlap,
        ] {
            let seeds: Vec<Object> = centres
                .iter()
                .enumerate()
                .map(|(k, &(x, y))| point(k as u128 + 1, x, y))
                .collect();
            let cache = cache_with(image, 20, seeds.clone());
            let mut command = command();
            command.stop = StopRule::Relative { fraction: 0.2 };
            command.max_distance = 6.0;
            command.when_regions_meet = meeting;
            let channel = ChannelWindow::of(&cache, 0, "test").unwrap();
            let mut refs: Vec<&Object> = seeds.iter().collect();
            refs.sort_by(|a, b| a.position_cmp(b));
            let grower = Grower::new(&channel, &cache, &refs, None, command.params(&ctx()));
            let summary = |regions: Vec<Option<RunRegion>>| -> Vec<_> {
                regions
                    .into_iter()
                    .map(|r| r.map(|r| (r.bbox, r.area, r.mask)))
                    .collect()
            };
            let whole = summary(grower.regions(usize::MAX, 1 << 20).unwrap());
            for block in [8, 13, 24] {
                assert!(grower.units(1, block).len() > 1);
                let split = summary(grower.regions(1, block).unwrap());
                assert!(whole == split, "{meeting:?}, block {block}");
            }
            assert!(whole.iter().flatten().count() >= centres.len() / 2);
        }
    }
}
