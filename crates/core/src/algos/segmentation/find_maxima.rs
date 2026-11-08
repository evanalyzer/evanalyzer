//! # find_maxima
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Spot detection by local maxima: every pixel that is the brightest within
//! `radius` becomes one 1-pixel object. Meant for an image smoothed at the
//! spot size beforehand (Gaussian Blur, or a difference of two Gaussians):
//! each spot is then one peak, its height the spot's "quality". Takes the
//! place of Threshold + Connected Components + Watershed in a spot pipeline;
//! the size of a spot comes afterwards (Transform Objects › Draw Circle, or
//! region growing with Threshold + Propagation).
//!
//! Self-contained on purpose: shares no code with Watershed's seed search
//! (whose rules are pinned by its CellProfiler parity), so either can change
//! without the other.

use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, PixelUnits, SegmentationClass, SizeUnits,
};
use kornia_image::Image;
use macros::CommandsMeta;

use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// Finds spots as local maxima: each pixel that is the brightest within the
/// radius becomes one 1-pixel object.
///
/// [Preprocessing (Gaussian Blur)] -> [Find Maxima] -> [Extract Objects] -> [Transform Objects (Draw Circle)]
///
/// Writes the segmentation and the instance map (both replaced): the maxima
/// get the segmentation class and one instance each; the image stays as it
/// is. A plateau (equal neighbouring pixels) gives one maximum, its pixel
/// nearest the plateau's centre; of equal maxima within the radius the one
/// found first (row order) is kept. Flat regions (a background of 0) are
/// never maxima.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", next = "extract", display_name = "Find Maxima")]
pub struct FindMaxima {
    /// A pixel is a maximum only if no pixel within this radius is brighter.
    /// About half the spot diameter: smaller finds several maxima per spot,
    /// larger merges neighbouring spots into one.
    #[cmdsmeta(default = 2.0, min = 1.0, max = 1000.0, step = 0.5, summary = true)]
    pub radius: f32,

    /// Unit `radius` is expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub radius_unit: SizeUnits,

    /// How the quality cut-off is chosen. A maximum's quality is the image's
    /// value at it.
    ///
    /// Manual: maxima above `min_quality` are kept.
    /// Otsu: Otsu's method on the qualities of all maxima of the image
    /// splits spots from noise peaks; `min_quality` stays the lowest cut-off,
    /// so an image without spots doesn't turn its noise into spots.
    #[cmdsmeta(default = MaximaCutoff::Manual, summary = true)]
    pub cutoff: MaximaCutoff,

    /// Maxima with a quality at or below this value are dropped (with Otsu:
    /// the lowest cut-off).
    #[cmdsmeta(default = 0, min = 0, max = 65535, step = 1, summary = true)]
    pub min_quality: f32,

    /// Unit used for `min_quality`.
    ///
    /// bit: 0 - 255/65535
    /// %: 0 - 100.0
    /// rel: 0 - 1.0
    #[cmdsmeta(default = PixelUnits::Bit, visibility = Advanced)]
    pub quality_unit: PixelUnits,

    /// Only maxima on pixels of the current segmentation (any class, e.g.
    /// from a Threshold before) count. Brighter pixels outside it still
    /// suppress maxima within the radius.
    pub inside_segmentation: bool,

    /// Drop maxima closer to the image border than the radius.
    pub exclude_at_edges: bool,

    /// The segmentation class the maxima get.
    pub segmentation_class: SegmentationClass,
}

/// How [`FindMaxima`] chooses its quality cut-off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaximaCutoff {
    Manual,
    Otsu,
}

// Per-pixel mark bits of the search (one byte per pixel).
/// No pixel within the radius is brighter.
const PEAK: u8 = 1;
/// Already gathered into a plateau.
const VISITED: u8 = 2;
/// A kept maximum.
const KEPT: u8 = 4;

impl ImageAlgorithm for FindMaxima {
    /// 1. **Peaks:** pixels with no brighter pixel within `radius` (first
    ///    against the 8 neighbours inside the disk, which rejects nearly
    ///    every pixel, then the whole disk). Pixels outside the tile don't
    ///    count.
    /// 2. **Plateaus:** from each peak with a darker 8-neighbour (in row
    ///    order), the 8-connected pixels of equal value are one plateau. It's
    ///    a maximum only if every pixel of it is a peak; its pixel nearest the
    ///    plateau's centroid stands for it (first in row order on a tie).
    ///    Flat regions without a darker pixel next to them (a background of 0,
    ///    a flat image) are never maxima.
    /// 3. **Equal maxima within the radius:** a maximum is dropped if one
    ///    found before lies within the radius (only equal ones can: a
    ///    brighter one would have rejected it in step 1). With
    ///    `inside_segmentation` the maximum's pixel must lie on the
    ///    segmentation.
    /// 4. **Edges and quality:** the image border (not the tile's) and the
    ///    cut-off (`> cutoff`, like Threshold's lower bound).
    /// 5. **Output:** both maps cleared, then each maximum written with the
    ///    segmentation class and its own instance id (1, 2, … in row order).
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let radius = self.radius_px(ctx);
        let nr_of_bits = ctx.image_meta.nr_of_bits;
        let floor = self.quality_unit.to_relative(self.min_quality, nr_of_bits);
        let offset = ctx.get_image_tile_offset();
        let full = ctx.full_image_size();

        let image = ctx.get_f32_gray_image()?;
        let (width, height) = (image.width(), image.height());
        let mask = if self.inside_segmentation {
            Some(ctx.get_segmentation_map()?.as_slice())
        } else {
            None
        };
        let values = image.as_slice();

        let mut maxima = find_maxima(values, mask, width, height, radius);

        if self.exclude_at_edges {
            let (full_w, full_h) = (full.width as f32, full.height as f32);
            maxima.retain(|&p| {
                let x = (offset.x + p % width) as f32;
                let y = (offset.y + p / width) as f32;
                x >= radius
                    && y >= radius
                    && full_w - 1.0 - x >= radius
                    && full_h - 1.0 - y >= radius
            });
        }

        let cutoff = match self.cutoff {
            MaximaCutoff::Manual => floor,
            MaximaCutoff::Otsu => {
                let qualities: Vec<f32> = maxima.iter().map(|&p| values[p]).collect();
                otsu_cutoff(&qualities).map_or(floor, |otsu| otsu.max(floor))
            }
        };
        maxima.retain(|&p| values[p] > cutoff);

        let class = self.segmentation_class.as_u32();
        write_maps(ctx, &maxima, class)
    }

    fn name(&self) -> &'static str {
        "Find Maxima"
    }

    /// One mark byte per pixel, and up to 4 bytes per pixel of the largest
    /// plateau (a flat region as big as the tile at worst); measured by
    /// `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        5.0
    }

    /// The detection (local maxima within a radius, then a brightness
    /// cut-off) is the one Sbalzarini & Koumoutsakos 2005 describe for
    /// particles; Otsu 1979 for the automatic cut-off.
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut papers = vec![&SBALZARINI_2005];
        if self.cutoff == MaximaCutoff::Otsu {
            papers.push(&OTSU_1979);
        }
        papers
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

static SBALZARINI_2005: CitationMetadata = CitationMetadata {
    cite_key: "sbalzarini2005feature",
    title: "Feature Point Tracking and Trajectory Analysis for Video Imaging in Cell Biology",
    authors: &["Ivo F. Sbalzarini", "Petros Koumoutsakos"],
    year: 2005,
    container: Some("Journal of Structural Biology"),
    doi: Some("10.1016/j.jsb.2005.06.002"),
    url: Some("https://doi.org/10.1016/j.jsb.2005.06.002"),
    pages: Some("182-195"),
};

static OTSU_1979: CitationMetadata = CitationMetadata {
    cite_key: "otsu1979threshold",
    title: "A Threshold Selection Method from Gray-Level Histograms",
    authors: &["Nobuyuki Otsu"],
    year: 1979,
    container: Some("IEEE Transactions on Systems, Man, and Cybernetics"),
    doi: Some("10.1109/TSMC.1979.4310076"),
    url: Some("https://doi.org/10.1109/TSMC.1979.4310076"),
    pages: Some("62-66"),
};

impl FindMaxima {
    /// `radius` in pixels (fractional: 2.5 px reaches the pixels within
    /// 2.5). Without a pixel size, nanometres are taken as pixels. At least
    /// 1, so the 4 direct neighbours always count.
    fn radius_px(&self, ctx: &PipelineContext) -> f32 {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        let radius = match self.radius_unit {
            SizeUnits::NanoMeter if pixel_size_nm > 0.0 => self.radius / pixel_size_nm,
            _ => self.radius,
        };
        radius.max(1.0)
    }
}

/// The pixel indices (row order) of the local maxima of `values`: steps 1-3
/// of [`FindMaxima::execute`].
fn find_maxima(
    values: &[f32],
    mask: Option<&[u32]>,
    width: usize,
    height: usize,
    radius: f32,
) -> Vec<usize> {
    let disk = disk_offsets(radius);
    // The 8 neighbours inside the disk: the cheap first test.
    let near: Vec<(isize, isize)> = disk
        .iter()
        .copied()
        .filter(|&(dx, dy)| dx.abs() <= 1 && dy.abs() <= 1)
        .collect();
    let mut marks = vec![0u8; values.len()];

    // 1. Peaks.
    for y in 0..height {
        for x in 0..width {
            let p = y * width + x;
            let v = values[p];
            if !v.is_nan()
                && !brighter_within(values, width, height, x, y, v, &near)
                && !brighter_within(values, width, height, x, y, v, &disk)
            {
                marks[p] = PEAK;
            }
        }
    }

    // 2. Plateaus; 3. equal maxima within the radius.
    let mut kept: Vec<usize> = Vec::new();
    // The plateau's pixels, gathered breadth-first (u32: a tile has fewer
    // than 2³² pixels).
    let mut plateau: Vec<u32> = Vec::new();
    for start in 0..marks.len() {
        if marks[start] != PEAK || !darker_neighbour(values, width, height, start) {
            continue;
        }
        let value = values[start];
        plateau.clear();
        plateau.push(start as u32);
        marks[start] |= VISITED;
        let mut all_peaks = true;
        let mut next = 0;
        while next < plateau.len() {
            let p = plateau[next] as usize;
            next += 1;
            all_peaks &= marks[p] & PEAK != 0;
            let (x, y) = ((p % width) as isize, (p / width) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let q = ny as usize * width + nx as usize;
                    if marks[q] & VISITED == 0 && values[q] == value {
                        marks[q] |= VISITED;
                        plateau.push(q as u32);
                    }
                }
            }
        }
        if !all_peaks {
            continue;
        }
        let pick = nearest_to_centroid(&plateau, width);
        if mask.is_some_and(|m| m[pick] == 0) {
            continue;
        }
        let (x, y) = (pick % width, pick / width);
        let clashes = disk.iter().any(|&(dx, dy)| {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            nx >= 0
                && ny >= 0
                && (nx as usize) < width
                && (ny as usize) < height
                && marks[ny as usize * width + nx as usize] & KEPT != 0
        });
        if !clashes {
            marks[pick] |= KEPT;
            kept.push(pick);
        }
    }
    // Plateaus are found in the order of their first darker-bordered pixel,
    // not of their pick - back into row order.
    kept.sort_unstable();
    kept
}

/// Whether one of the 8 neighbours of `p` inside the tile is darker.
fn darker_neighbour(values: &[f32], width: usize, height: usize, p: usize) -> bool {
    let value = values[p];
    let (x, y) = ((p % width) as isize, (p / width) as isize);
    (-1..=1).any(|dy| {
        (-1..=1).any(|dx| {
            let (nx, ny) = (x + dx, y + dy);
            nx >= 0
                && ny >= 0
                && nx < width as isize
                && ny < height as isize
                && values[ny as usize * width + nx as usize] < value
        })
    })
}

/// Whether any of `offsets` around (`x`, `y`) inside the tile is brighter
/// than `value`.
fn brighter_within(
    values: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    value: f32,
    offsets: &[(isize, isize)],
) -> bool {
    offsets.iter().any(|&(dx, dy)| {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        nx >= 0
            && ny >= 0
            && (nx as usize) < width
            && (ny as usize) < height
            && values[ny as usize * width + nx as usize] > value
    })
}

/// The offsets (dx, dy) of the disk of `radius` around a pixel, without the
/// pixel itself (`dx² + dy² ≤ radius²`).
fn disk_offsets(radius: f32) -> Vec<(isize, isize)> {
    let reach = radius.floor() as isize;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            if (dx, dy) != (0, 0) && (dx * dx + dy * dy) as f32 <= r2 {
                offsets.push((dx, dy));
            }
        }
    }
    offsets
}

/// The pixel of `plateau` nearest its centroid; the first in row order on a
/// tie.
fn nearest_to_centroid(plateau: &[u32], width: usize) -> usize {
    let n = plateau.len() as f64;
    let (sx, sy) = plateau.iter().fold((0.0, 0.0), |(sx, sy), &p| {
        let p = p as usize;
        (sx + (p % width) as f64, sy + (p / width) as f64)
    });
    let (cx, cy) = (sx / n, sy / n);
    let distance = |p: usize| {
        let dx = (p % width) as f64 - cx;
        let dy = (p / width) as f64 - cy;
        dx * dx + dy * dy
    };
    let mut best = plateau[0] as usize;
    for &p in plateau {
        let p = p as usize;
        let (d, d_best) = (distance(p), distance(best));
        if d < d_best || (d == d_best && p < best) {
            best = p;
        }
    }
    best
}

/// Otsu's cut-off (Otsu 1979) over `qualities`, on 256 bins between their
/// lowest and highest value: the largest value of the last background bin
/// (kept are the values above it). `None` with fewer than two different
/// values - nothing to split.
fn otsu_cutoff(qualities: &[f32]) -> Option<f32> {
    let (lo, hi) = qualities
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &q| {
            (lo.min(q), hi.max(q))
        });
    if !(hi > lo) {
        return None;
    }
    const BINS: usize = 256;
    let scale = (BINS - 1) as f32 / (hi - lo);
    let bin = |q: f32| (((q - lo) * scale).round() as usize).min(BINS - 1);
    let mut histogram = [0u64; BINS];
    for &q in qualities {
        histogram[bin(q)] += 1;
    }
    let total = qualities.len() as f64;
    let sum_all: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut weight_bg, mut sum_bg) = (0.0f64, 0.0f64);
    let (mut best_k, mut best_var) = (0usize, -1.0f64);
    // The last bin as background would keep nothing.
    for (k, &count) in histogram.iter().enumerate().take(BINS - 1) {
        weight_bg += count as f64;
        sum_bg += k as f64 * count as f64;
        let weight_fg = total - weight_bg;
        if weight_bg == 0.0 || weight_fg == 0.0 {
            continue;
        }
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;
        let between = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);
        if between > best_var {
            best_var = between;
            best_k = k;
        }
    }
    // The largest value `bin` still rounds into `best_k`.
    let upper = lo + (best_k as f32 + 0.5) / scale;
    let cutoff = qualities
        .iter()
        .copied()
        .filter(|&q| q <= upper && bin(q) <= best_k)
        .fold(lo, f32::max);
    Some(cutoff)
}

/// Clears the segmentation and the instance map (created if missing) and
/// writes `maxima`: the segmentation `class`, instance ids 1, 2, … in order.
fn write_maps(
    ctx: &mut PipelineContext,
    maxima: &[usize],
    class: u32,
) -> Result<(), InternalErrors> {
    let size = ctx.image.size();
    let pixels = size.width * size.height;
    let blank = |map: &mut Option<Image<u32, 1>>| -> Result<(), InternalErrors> {
        match map {
            Some(map) if map.size() == size => map.as_slice_mut().fill(0),
            _ => {
                *map = Some(
                    Image::new(size, vec![0u32; pixels]).map_err(InternalErrors::from_kornia)?,
                )
            }
        }
        Ok(())
    };
    blank(&mut ctx.segmentation_map)?;
    blank(&mut ctx.instance_map)?;
    let segmentation = ctx
        .segmentation_map
        .as_mut()
        .expect("created above")
        .as_slice_mut();
    for &p in maxima {
        segmentation[p] = class;
    }
    let instances = ctx
        .instance_map
        .as_mut()
        .expect("created above")
        .as_slice_mut();
    for (id, &p) in maxima.iter().enumerate() {
        instances[p] = id as u32 + 1;
    }
    Ok(())
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
    use kornia_image::ImageSize;

    /// The 12 x 7 example of the design discussion (values / 100 as the
    /// relative image): spots A (3,3) = 90 and B (6,3) = 80 touching, a weak
    /// spot C (10,1) = 30.
    const EXAMPLE: [[u8; 12]; 7] = [
        [5, 10, 15, 20, 15, 15, 20, 15, 10, 5, 15, 5],
        [10, 25, 40, 50, 40, 35, 45, 40, 20, 10, 30, 10],
        [15, 40, 70, 80, 65, 55, 70, 60, 30, 10, 15, 5],
        [20, 50, 80, 90, 70, 60, 80, 65, 35, 10, 5, 5],
        [15, 40, 70, 75, 65, 50, 65, 55, 30, 10, 5, 5],
        [10, 25, 40, 45, 40, 30, 40, 35, 20, 5, 5, 5],
        [5, 10, 15, 20, 15, 10, 15, 10, 10, 5, 5, 5],
    ];

    fn command() -> FindMaxima {
        FindMaxima {
            radius: 2.0,
            radius_unit: SizeUnits::Pixels,
            cutoff: MaximaCutoff::Manual,
            min_quality: 0.0,
            quality_unit: PixelUnits::Relative,
            inside_segmentation: false,
            exclude_at_edges: false,
            segmentation_class: SegmentationClass(1),
        }
    }

    fn context(rows: &[Vec<f32>]) -> PipelineContext {
        let size = ImageSize {
            width: rows[0].len(),
            height: rows.len(),
        };
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.get_f32_gray_image_mut()
            .unwrap()
            .as_slice_mut()
            .copy_from_slice(&rows.concat());
        ctx
    }

    fn example() -> PipelineContext {
        let rows: Vec<Vec<f32>> = EXAMPLE
            .iter()
            .map(|row| row.iter().map(|&v| v as f32 / 100.0).collect())
            .collect();
        context(&rows)
    }

    /// (x, y, segmentation class, instance id) of every nonzero pixel.
    fn written(ctx: &PipelineContext) -> Vec<(usize, usize, u32, u32)> {
        let width = ctx.image.size().width;
        let segmentation = ctx.segmentation_map.as_ref().unwrap().as_slice();
        let instances = ctx.instance_map.as_ref().unwrap().as_slice();
        (0..segmentation.len())
            .filter(|&p| segmentation[p] != 0 || instances[p] != 0)
            .map(|p| (p % width, p / width, segmentation[p], instances[p]))
            .collect()
    }

    fn run(command: &FindMaxima, ctx: &mut PipelineContext) {
        command
            .execute(ctx, &mut GlobalPipelineCache::default())
            .unwrap();
    }

    #[test]
    fn the_design_example_keeps_a_and_b_and_drops_the_weak_spot() {
        let mut ctx = example();
        let command = FindMaxima {
            min_quality: 0.5,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1), (6, 3, 1, 2)]);
    }

    #[test]
    fn without_a_cutoff_the_weak_spot_is_kept_too() {
        let mut ctx = example();
        run(&command(), &mut ctx);
        assert_eq!(
            written(&ctx),
            vec![(10, 1, 1, 1), (3, 3, 1, 2), (6, 3, 1, 3)]
        );
    }

    #[test]
    fn a_larger_radius_merges_the_touching_spots() {
        let mut ctx = example();
        let command = FindMaxima {
            radius: 3.0,
            min_quality: 0.5,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1)]);
    }

    #[test]
    fn the_image_stays_unchanged() {
        let mut ctx = example();
        let before = ctx.get_f32_gray_image().unwrap().as_slice().to_vec();
        run(&command(), &mut ctx);
        assert_eq!(ctx.get_f32_gray_image().unwrap().as_slice(), &before[..]);
    }

    #[test]
    fn a_plateau_gives_its_middle_pixel() {
        let row = |v: [f32; 7]| v.to_vec();
        let mut ctx = context(&[
            row([0.0; 7]),
            row([0.0, 0.1, 0.9, 0.9, 0.9, 0.1, 0.0]),
            row([0.0; 7]),
        ]);
        run(&command(), &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 1, 1, 1)]);
    }

    #[test]
    fn a_two_pixel_plateau_gives_its_first_pixel() {
        let mut ctx = context(&[
            vec![0.0; 6],
            vec![0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
            vec![0.0; 6],
        ]);
        run(&command(), &mut ctx);
        assert_eq!(written(&ctx), vec![(2, 1, 1, 1)]);
    }

    #[test]
    fn equal_maxima_within_the_radius_keep_the_first() {
        // 0.9 at x = 1 and x = 3, a dip between: not one plateau, but
        // within the radius of each other.
        let mut ctx = context(&[
            vec![0.0; 7],
            vec![0.0, 0.9, 0.5, 0.9, 0.0, 0.0, 0.0],
            vec![0.0; 7],
        ]);
        run(&command(), &mut ctx);
        assert_eq!(written(&ctx), vec![(1, 1, 1, 1)]);
    }

    #[test]
    fn a_plateau_seeing_a_brighter_pixel_is_no_maximum() {
        // The plateau 0.5 0.5 at x = 1, 2: x = 2 is within 2 px of 0.9 at
        // x = 4, so the whole plateau is no maximum, also its pixel x = 1.
        let mut ctx = context(&[
            vec![0.0; 7],
            vec![0.0, 0.5, 0.5, 0.0, 0.9, 0.0, 0.0],
            vec![0.0; 7],
        ]);
        run(&command(), &mut ctx);
        assert_eq!(written(&ctx), vec![(4, 1, 1, 1)]);
    }

    #[test]
    fn exclude_at_edges_counts_the_image_border_not_the_tile_border() {
        use crate::image::{ImageContainer, Point2d};
        let mut ctx = example();
        if let ImageContainer::F32Gray(image) = std::sync::Arc::make_mut(&mut ctx.image) {
            image.tile_offset = Point2d { x: 50, y: 50 };
        }
        ctx.image_meta.full_image_width = ImageSize {
            width: 1000,
            height: 1000,
        };
        let command = FindMaxima {
            exclude_at_edges: true,
            ..command()
        };
        run(&command, &mut ctx);
        // C at the tile's top edge is far from the image's: kept.
        assert_eq!(
            written(&ctx),
            vec![(10, 1, 1, 1), (3, 3, 1, 2), (6, 3, 1, 3)]
        );
    }

    #[test]
    fn a_flat_image_has_no_maxima() {
        let mut ctx = context(&vec![vec![0.0; 5]; 5]);
        run(&command(), &mut ctx);
        assert!(written(&ctx).is_empty());
    }

    #[test]
    fn stale_maps_are_cleared() {
        let mut ctx = example();
        let size = ctx.image.size();
        let pixels = size.width * size.height;
        ctx.segmentation_map = Some(Image::new(size, vec![7u32; pixels]).unwrap());
        ctx.instance_map = Some(Image::new(size, vec![9u32; pixels]).unwrap());
        let command = FindMaxima {
            min_quality: 0.5,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1), (6, 3, 1, 2)]);
    }

    #[test]
    fn inside_segmentation_keeps_only_maxima_on_the_mask() {
        let mut ctx = example();
        let size = ctx.image.size();
        // Only the left half (A's side) is segmented.
        let mask: Vec<u32> = (0..size.width * size.height)
            .map(|p| u32::from(p % size.width < 5))
            .collect();
        ctx.segmentation_map = Some(Image::new(size, mask).unwrap());
        let command = FindMaxima {
            inside_segmentation: true,
            min_quality: 0.5,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1)]);
    }

    #[test]
    fn inside_segmentation_without_a_segmentation_fails() {
        let mut ctx = example();
        ctx.segmentation_map = None;
        let command = FindMaxima {
            inside_segmentation: true,
            ..command()
        };
        assert!(
            command
                .execute(&mut ctx, &mut GlobalPipelineCache::default())
                .is_err()
        );
    }

    #[test]
    fn exclude_at_edges_drops_maxima_near_the_border() {
        let mut ctx = example();
        let command = FindMaxima {
            exclude_at_edges: true,
            ..command()
        };
        run(&command, &mut ctx);
        // C at y = 1 is closer than 2 px to the border.
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1), (6, 3, 1, 2)]);
    }

    #[test]
    fn otsu_splits_spots_from_noise_peaks() {
        // Ten noise peaks of 0.10-0.14 and three spots of 0.8-0.9, every
        // peak 4 px from the next.
        let peaks = [
            0.10, 0.11, 0.12, 0.13, 0.14, 0.8, 0.10, 0.11, 0.85, 0.12, 0.13, 0.9, 0.14,
        ];
        let mut row = vec![0.0f32; peaks.len() * 4 + 1];
        for (i, &v) in peaks.iter().enumerate() {
            row[2 + i * 4] = v;
        }
        let blank = vec![0.0f32; row.len()];
        let mut ctx = context(&[blank.clone(), blank.clone(), row, blank.clone(), blank]);
        let command = FindMaxima {
            cutoff: MaximaCutoff::Otsu,
            ..command()
        };
        run(&command, &mut ctx);
        let xs: Vec<usize> = written(&ctx).iter().map(|w| w.0).collect();
        assert_eq!(xs, vec![2 + 5 * 4, 2 + 8 * 4, 2 + 11 * 4]);
    }

    #[test]
    fn otsu_keeps_min_quality_as_the_lowest_cutoff() {
        let mut ctx = example();
        let command = FindMaxima {
            cutoff: MaximaCutoff::Otsu,
            min_quality: 0.85,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1)]);
    }

    #[test]
    fn otsu_cutoff_needs_two_different_values() {
        assert_eq!(otsu_cutoff(&[]), None);
        assert_eq!(otsu_cutoff(&[0.5, 0.5]), None);
        assert_eq!(otsu_cutoff(&[0.1, 0.9]), Some(0.1));
    }

    #[test]
    fn radius_in_nanometres_uses_the_pixel_size() {
        let mut ctx = example();
        ctx.image_meta.pixel_sizes.px_size_x = 100.0;
        ctx.image_meta.pixel_sizes.px_size_y = 100.0;
        // 300 nm = 3 px: A and B merge as with `radius: 3.0`.
        let command = FindMaxima {
            radius: 300.0,
            radius_unit: SizeUnits::NanoMeter,
            min_quality: 0.5,
            ..command()
        };
        run(&command, &mut ctx);
        assert_eq!(written(&ctx), vec![(3, 3, 1, 1)]);
    }

    #[test]
    fn disk_offsets_follow_the_radius() {
        assert_eq!(disk_offsets(1.0).len(), 4);
        assert_eq!(disk_offsets(1.5).len(), 8);
        assert_eq!(disk_offsets(2.0).len(), 12);
    }
}
