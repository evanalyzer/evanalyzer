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
//! The growth is a port of CellProfiler's (`centrosome/_propagate.pyx`,
//! BSD-3-Clause, Broad Institute): the same step cost, 8 neighbours and
//! order of equally cheap pixels, so the same image gives the same regions.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::object::{Object, ObjectInit};
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use crate::pipeline::scope::Scope;
use bitvec::prelude::*;
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass, ObjectId,
};
use macros::CommandsMeta;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

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
        let image = GrowthImage::read(cache, self.growth_channel.0 as i32)?;
        let seeds: Vec<&Object> = cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.seeds, ctx.scope))
            .collect();
        if seeds.is_empty() {
            return Ok(());
        }

        let allowed = self.allowed_pixels(&image, cache, ctx.scope);
        // Label `i + 1` is `seeds[i]`: a short list of their pixels, not a
        // label per pixel of the image.
        let mut seed_pixels = Vec::new();
        for (i, seed) in seeds.iter().enumerate() {
            image.for_each_pixel_of(seed, |idx| seed_pixels.push((idx, i as u32 + 1)));
        }
        let labels = propagate(
            &image.values,
            image.width,
            image.height,
            &allowed,
            &seed_pixels,
            self.regularization as f64,
        );

        let full = ctx.full_image_size();
        let regions = image.regions(&labels, seeds.len(), full.width, full.height);
        let mut new_objects = Vec::new();
        for (seed, region) in seeds.iter().zip(regions) {
            let Some(region) = region else {
                continue;
            };
            if self.exclude_at_edges && region.touches_edge {
                continue;
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId::next(),
                segmentation_class: seed.segmentation_class,
                bbox: region.bbox,
                mask_data: region.mask,
                area: region.area,
                plane: seed.plane,
                touches_edge: region.touches_edge,
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

impl Propagation {
    /// The pixels a region may grow into: read from the channel, and inside
    /// the mask's objects if a mask is set.
    fn allowed_pixels(
        &self,
        image: &GrowthImage,
        cache: &GlobalPipelineCache,
        scope: Scope,
    ) -> BitVec<u64, Lsb0> {
        if self.mask == ObjectClass::Unset {
            return image.read.clone();
        }
        let mut allowed = BitVec::repeat(false, image.values.len());
        for object in cache
            .object_cache
            .values()
            .filter(|o| o.is_of(&self.mask, scope))
        {
            image.for_each_pixel_of(object, |idx| allowed.set(idx, image.read[idx]));
        }
        allowed
    }
}

/// Where the method was published (CellProfiler's "Propagation").
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

/// The growth channel over the part of the image its tiles cover (the whole
/// image in a run, the visible part in a preview), row-major.
struct GrowthImage {
    x0: u32,
    y0: u32,
    width: usize,
    height: usize,
    /// Normalized (0..1) pixel values; 0 where no tile was read.
    values: Vec<f32>,
    /// A tile covered the pixel (one bit each).
    read: BitVec<u64, Lsb0>,
}

impl GrowthImage {
    fn read(cache: &GlobalPipelineCache, channel: i32) -> Result<Self, InternalErrors> {
        let no_channel = || {
            InternalErrors::InvalidArgument(format!(
                "Propagation: the image has no channel {channel}"
            ))
        };
        let tiles: Vec<_> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(no_channel());
        }
        let x0 = tiles.iter().map(|t| t.offset_x).min().unwrap_or(0);
        let y0 = tiles.iter().map(|t| t.offset_y).min().unwrap_or(0);
        let x1 = tiles
            .iter()
            .map(|t| t.offset_x + t.width)
            .max()
            .unwrap_or(0);
        let y1 = tiles
            .iter()
            .map(|t| t.offset_y + t.height)
            .max()
            .unwrap_or(0);
        let (width, height) = (x1 - x0, y1 - y0);
        let mut values = vec![0.0f32; width * height];
        let mut read = BitVec::<u64, Lsb0>::repeat(false, width * height);
        for tile in &tiles {
            let container = cache
                .image_cache
                .get(&CacheAddress::Channel((channel, tile.clone())))
                .ok_or_else(no_channel)?;
            let rgb = matches!(*container, crate::image::ImageContainer::F32Rgb(_));
            let slice = channel_pixel_slice(&container).unwrap_or_default();
            let len = if rgb { slice.len() / 3 } else { slice.len() };
            for ty in 0..tile.height {
                for tx in 0..tile.width {
                    let sample = ty * tile.width + tx;
                    if sample >= len {
                        break;
                    }
                    let v = sample_channel_pixel(rgb, slice, sample);
                    let idx = (tile.offset_y + ty - y0) * width + (tile.offset_x + tx - x0);
                    // A pixel that isn't a number can't be grown into.
                    values[idx] = if v.is_finite() { v } else { 0.0 };
                    read.set(idx, v.is_finite());
                }
            }
        }
        Ok(Self {
            x0: x0 as u32,
            y0: y0 as u32,
            width,
            height,
            values,
            read,
        })
    }

    /// Calls `f` with the index of each of `object`'s pixels inside the
    /// image.
    fn for_each_pixel_of(&self, object: &Object, mut f: impl FnMut(usize)) {
        let [x_min, y_min, x_max, y_max] = object.bbox;
        if x_max < x_min || y_max < y_min {
            return;
        }
        // bbox[2]/[3] are inclusive; the mask is row-major over the bbox.
        let mask_width = (x_max - x_min + 1) as usize;
        for (bit, set) in object.mask_data.iter().by_vals().enumerate() {
            if !set {
                continue;
            }
            let x = x_min + (bit % mask_width) as u32;
            let y = y_min + (bit / mask_width) as u32;
            if x > x_max || y > y_max || x < self.x0 || y < self.y0 {
                continue;
            }
            let (col, row) = ((x - self.x0) as usize, (y - self.y0) as usize);
            if col < self.width && row < self.height {
                f(row * self.width + col);
            }
        }
    }

    /// Per label `1..=count`, the region it got (`None`: no pixel).
    fn regions(
        &self,
        labels: &[u32],
        count: usize,
        full_width: usize,
        full_height: usize,
    ) -> Vec<Option<Region>> {
        let mut regions: Vec<Option<Region>> = (0..count).map(|_| None).collect();
        for (idx, &label) in labels.iter().enumerate() {
            if label == 0 {
                continue;
            }
            let x = self.x0 + (idx % self.width) as u32;
            let y = self.y0 + (idx / self.width) as u32;
            let region = regions[label as usize - 1].get_or_insert_with(|| Region {
                bbox: [x, y, x, y],
                ..Default::default()
            });
            let (xu, yu) = (x as u64, y as u64);
            region.area += 1;
            region.sum_x += xu;
            region.sum_y += yu;
            region.sum_x2 += xu * xu;
            region.sum_y2 += yu * yu;
            region.sum_xy += xu * yu;
            let b = &mut region.bbox;
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        }
        for (i, region) in regions.iter_mut().enumerate() {
            let Some(region) = region else {
                continue;
            };
            let [x_min, y_min, x_max, y_max] = region.bbox;
            let w = (x_max - x_min + 1) as usize;
            let h = (y_max - y_min + 1) as usize;
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * h);
            for ry in 0..h {
                let row = (y_min - self.y0) as usize + ry;
                for rx in 0..w {
                    let col = (x_min - self.x0) as usize + rx;
                    if labels[row * self.width + col] == i as u32 + 1 {
                        mask.set(ry * w + rx, true);
                    }
                }
            }
            region.mask = mask;
            // The image's border, not a tile's.
            region.touches_edge = x_min == 0
                || y_min == 0
                || x_max as usize + 1 >= full_width
                || y_max as usize + 1 >= full_height;
        }
        regions
    }
}

/// One grown region, in image coordinates (bbox inclusive, mask row-major
/// over it).
#[derive(Default)]
struct Region {
    bbox: [u32; 4],
    mask: BitVec<u64, Lsb0>,
    area: usize,
    sum_x: u64,
    sum_y: u64,
    sum_x2: u64,
    sum_y2: u64,
    sum_xy: u64,
    touches_edge: bool,
}

/// Grows the seeds (pixel index, label > 0; where seeds overlap, the first
/// wins) into the `allowed` pixels of a `width` × `height` image: each pixel
/// gets the label that reaches it cheapest (see [`step_cost`]); equally
/// cheap ones go to the lower label, then the earlier pixel - CellProfiler's
/// heap order. Seed pixels keep their label, also outside `allowed`; only
/// seeds inside it grow.
///
/// Per pixel: its label (4 bytes) and cheapest cost (8 - an `f64` like
/// CellProfiler's, so sums and ties come out the same).
fn propagate(
    values: &[f32],
    width: usize,
    height: usize,
    allowed: &BitSlice<u64, Lsb0>,
    seeds: &[(usize, u32)],
    weight: f64,
) -> Vec<u32> {
    // Queue entries hold the pixel as u32: 16 bytes instead of 24.
    assert!(
        values.len() <= u32::MAX as usize,
        "Propagation: an image of more than 4 billion pixels"
    );
    let mut labels = vec![0u32; values.len()];
    // The cheapest cost pushed so far; seeds cost 0.
    let mut costs = vec![f64::INFINITY; values.len()];
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

/// CellProfiler's cost of the step between neighbours `a` and `b` (row,
/// column): the summed absolute difference of their 3×3 neighbourhoods
/// (pixels past the border repeat the border's) combined with their
/// Manhattan distance weighted by `weight`²:
/// `sqrt(diff² + manhattan · weight²)`.
///
/// Summed in CellProfiler's order (rows, then columns), so the result is
/// the same to the last bit.
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
        // Away from the border (nearly every pixel): no clamping.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageTile;
    use crate::image::{ImageContainer, ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
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

    // The cases of CellProfiler's own tests (centrosome `test_propagate.py`).

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
        let side = step_cost(&flat, 3, 3, (1, 1), (1, 2), 0.5);
        let diagonal = step_cost(&flat, 3, 3, (1, 1), (2, 2), 0.5);
        assert!((side - 0.5).abs() < 1e-12);
        assert!((diagonal - (2.0f64).sqrt() * 0.5).abs() < 1e-12);
        // One bright pixel at (0, 0): it lies in a's window (1, 1) and in
        // b's (1, 0) window too, so only the step (1, 1) -> (2, 2) sees it.
        let mut spot = vec![0.0f32; 9];
        spot[0] = 1.0;
        let cost = step_cost(&spot, 3, 3, (1, 1), (2, 2), 0.0);
        assert!((cost - 1.0).abs() < 1e-12, "{cost}");
    }

    /// Pixel for pixel what CellProfiler's own code gives
    /// (`centrosome.propagate` 1.3.4 on Gaussian "cells" with noise; made by
    /// `tests/propagation_centrosome.py`).
    #[test]
    fn same_regions_as_cellprofiler() {
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
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/propagation_centrosome.json"
        );
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(fixture.cases.len(), 5);
        for case in fixture.cases {
            let values: Vec<f32> = case.image.iter().map(|&v| v as f32).collect();
            // CellProfiler's seed image as a list of seed pixels.
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
