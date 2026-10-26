//! # Texture
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-24
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Haralick's texture features - CellProfiler's MeasureTexture
//! (cellprofiler/modules/measuretexture.py, CellProfiler 4.2.8, which calls
//! mahotas.features.haralick), value for value: the channel as 8 bits
//! (its normalized 0..1 values, as CellProfiler loads the image, × 255),
//! rescaled to the gray levels; per direction
//! the symmetric co-occurrence matrix of the pixel pairs that far apart and
//! the 13 features of it - over the whole image (Measure Image Texture) or
//! per object of a class (Measure Object Texture), where the pixels outside
//! the object and the pixels at 0 are left out. An object with no pair in
//! one of the directions gets no values (CellProfiler stores 0 then).

use crate::ImageDimensionality;
use crate::algos::GlobalPipelineCache;
use crate::object::Object;
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use crate::pipeline::pipeline_measure::Outputs;
use crate::{ImageTile, image::ImageContainer};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, MeasurementUnit,
};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

/// Which values per feature: the mean of the four directions, or one per
/// direction.
#[derive(CommandsMeta, Debug, Clone, Copy, PartialEq)]
pub enum TextureDirections {
    /// The mean of the four directions - one value per feature.
    #[cmdsmeta(display_name = "Mean of the directions")]
    Mean,
    /// A value per direction (0°, 45°, 90°, 135°), as CellProfiler stores
    /// them - four times as many values.
    #[cmdsmeta(display_name = "Each direction")]
    Each,
}

/// The features, in mahotas' order: (id, name, range).
const FEATURES: [(&str, &str, Option<(f64, f64)>); 13] = [
    (
        "angular_second_moment",
        "Angular second moment",
        Some((0.0, 1.0)),
    ),
    ("contrast", "Contrast", None),
    ("correlation", "Correlation", Some((-1.0, 1.0))),
    ("variance", "Variance", None),
    (
        "inverse_difference_moment",
        "Inverse difference moment",
        Some((0.0, 1.0)),
    ),
    ("sum_average", "Sum average", None),
    ("sum_variance", "Sum variance", None),
    ("sum_entropy", "Sum entropy", None),
    ("entropy", "Entropy", None),
    ("difference_variance", "Difference variance", None),
    ("difference_entropy", "Difference entropy", None),
    (
        "info_meas_1",
        "Information measure of correlation 1",
        Some((-1.0, 0.0)),
    ),
    (
        "info_meas_2",
        "Information measure of correlation 2",
        Some((0.0, 1.0)),
    ),
];

/// The directions, in mahotas' order: (row, column) step and the angle
/// CellProfiler gives it.
const DIRECTIONS: [((isize, isize), u32); 4] =
    [((0, 1), 0), ((1, 1), 135), ((1, 0), 90), ((1, -1), 45)];

/// The 13 features of each direction.
pub(super) type Features = [[f64; 13]; 4];

/// Co-occurrence counts are indexed `a * LEVELS + b`.
const LEVELS: usize = 256;

/// The settings Measure Image Texture and Measure Object Texture share.
#[derive(Clone, Copy)]
pub(super) struct Texture {
    pub(super) channel: ImageChannelIdx,
    pub(super) scale: u32,
    pub(super) gray_levels: u32,
    pub(super) directions: TextureDirections,
}

impl Texture {
    /// The gray levels - checking the settings.
    pub(super) fn levels(&self) -> Result<Levels, InternalErrors> {
        if self.scale == 0 || !(2..=256).contains(&self.gray_levels) {
            return Err(InternalErrors::InvalidArgument(
                "Measure texture: the scale must be at least 1, the gray levels 2 to 256".into(),
            ));
        }
        Ok(Levels::new(self.gray_levels))
    }

    /// How many values per image (or per object).
    pub(super) fn output_count(&self) -> usize {
        match self.directions {
            TextureDirections::Mean => FEATURES.len(),
            TextureDirections::Each => FEATURES.len() * DIRECTIONS.len(),
        }
    }

    /// Every output of the step `step_id`.
    pub(super) fn outputs(&self, step_id: &str, per_object: bool) -> Vec<MeasurementInfo> {
        (0..self.output_count())
            .map(|i| self.output(step_id, i, per_object))
            .collect()
    }

    /// Output `i` (direction-major with [`TextureDirections::Each`]) of the
    /// step `step_id`, of the image or of the objects.
    pub(super) fn output(&self, step_id: &str, i: usize, per_object: bool) -> MeasurementInfo {
        let about = format!("channel {}, scale {}", self.channel, self.scale);
        let outputs = Outputs::of(step_id).about(about);
        let (id, name) = match self.directions {
            TextureDirections::Mean => {
                let (id, name, _) = FEATURES[i];
                (id.to_string(), name.to_string())
            }
            TextureDirections::Each => {
                let (id, name, _) = FEATURES[i % FEATURES.len()];
                let angle = DIRECTIONS[i / FEATURES.len()].1;
                (format!("{id}_{angle}"), format!("{name} {angle}°"))
            }
        };
        let info = if per_object {
            outputs.object(&id, &name, MeasurementUnit::None)
        } else {
            outputs.image(&id, &name, MeasurementUnit::None)
        };
        match FEATURES[i % FEATURES.len()].2 {
            Some((min, max)) => info.range(min, max),
            None => info,
        }
    }

    /// The values stored for `features` (NaN: none), in output order.
    pub(super) fn values(&self, features: &Option<Features>) -> Vec<f64> {
        let Some(features) = features else {
            return vec![f64::NAN; self.output_count()];
        };
        match self.directions {
            TextureDirections::Mean => (0..FEATURES.len())
                .map(|f| features.iter().map(|d| d[f]).sum::<f64>() / DIRECTIONS.len() as f64)
                .collect(),
            TextureDirections::Each => features.iter().flatten().copied().collect(),
        }
    }

    /// The features over the whole image (CellProfiler's run_image: every
    /// pixel, zeros included). The tiles are counted in parallel; pairs
    /// across a tile border take the neighbour from the tile it lies in.
    pub(super) fn image_features(
        &self,
        cache: &GlobalPipelineCache,
        levels: &Levels,
    ) -> Result<Option<Features>, InternalErrors> {
        let channel = self.channel.0 as i32;
        let tiles: Vec<(ImageTile, Arc<ImageContainer>)> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .filter_map(|(_, tile)| {
                let image = cache
                    .image_cache
                    .get(&CacheAddress::Channel((channel, tile)))?;
                Some((tile, image.clone()))
            })
            .collect();
        if tiles.is_empty() {
            return Err(InternalErrors::InvalidArgument(format!(
                "Measure texture: the image has no channel {channel}"
            )));
        }
        let level_at = |x: usize, y: usize| -> Option<u8> {
            tiles.iter().find_map(|(t, image)| {
                let inside = x >= t.offset_x
                    && y >= t.offset_y
                    && x < t.offset_x + t.width
                    && y < t.offset_y + t.height;
                inside.then(|| {
                    let rgb = matches!(**image, ImageContainer::F32Rgb(_));
                    let samples = channel_pixel_slice(image).unwrap_or_default();
                    let sample = (y - t.offset_y) * t.width + x - t.offset_x;
                    levels.of(sample_channel_pixel(rgb, samples, sample))
                })
            })
        };
        let distance = self.scale as isize;
        let counted = tiles
            .par_iter()
            .map(|(t, image)| {
                let rgb = matches!(**image, ImageContainer::F32Rgb(_));
                let samples = channel_pixel_slice(image).unwrap_or_default();
                let grid: Vec<u8> = (0..t.width * t.height)
                    .map(|i| levels.of(sample_channel_pixel(rgb, samples, i)))
                    .collect();
                let mut counts = Cooccurrences::new();
                for (d, ((dy, dx), _)) in DIRECTIONS.iter().enumerate() {
                    let (dy, dx) = (dy * distance, dx * distance);
                    for y in 0..t.height {
                        for x in 0..t.width {
                            let (x2, y2) = (x as isize + dx, y as isize + dy);
                            let b = if x2 >= 0
                                && y2 >= 0
                                && (x2 as usize) < t.width
                                && (y2 as usize) < t.height
                            {
                                Some(grid[y2 as usize * t.width + x2 as usize])
                            } else {
                                let (gx, gy) = (t.offset_x as isize + x2, t.offset_y as isize + y2);
                                (gx >= 0 && gy >= 0)
                                    .then(|| level_at(gx as usize, gy as usize))
                                    .flatten()
                            };
                            if let Some(b) = b {
                                counts.add(d, grid[y * t.width + x], b);
                            }
                        }
                    }
                }
                let max = grid.iter().copied().max().unwrap_or(0);
                (counts, max)
            })
            .reduce(
                || (Cooccurrences::new(), 0),
                |(mut a, max_a), (b, max_b)| {
                    a.merge(&b);
                    (a, max_a.max(max_b))
                },
            );
        let (counts, max) = counted;
        Ok(counts.features(max as usize + 1))
    }

    /// `object`'s features (CellProfiler's run_one: the pixels of its bounding
    /// box, those outside it at 0, the zeros left out).
    pub(super) fn object_features(
        &self,
        object: &Object,
        cache: &GlobalPipelineCache,
        levels: &Levels,
    ) -> Option<Features> {
        let channel = self.channel.0 as i32;
        let views = cache.resolve_channel_views_for_bbox(object.bbox);
        let tiles: Vec<(ImageTile, bool, &[f32])> = views
            .iter()
            .filter(|((c, _), _, _)| *c == channel)
            .filter_map(|((_, tile), rgb, container)| {
                Some((*tile, *rgb, channel_pixel_slice(container)?))
            })
            .collect();
        let [x0, y0, x1, y1] = object.bbox.map(|v| v as usize);
        let (width, height) = (x1 - x0 + 1, y1 - y0 + 1);
        let mut grid = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                if !object.mask_data.get(y * width + x).is_some_and(|b| *b) {
                    continue;
                }
                let (gx, gy) = (x0 + x, y0 + y);
                grid[y * width + x] = tiles
                    .iter()
                    .find_map(|(t, rgb, samples)| {
                        let inside = gx >= t.offset_x
                            && gy >= t.offset_y
                            && gx < t.offset_x + t.width
                            && gy < t.offset_y + t.height;
                        inside.then(|| {
                            levels.of(sample_channel_pixel(
                                *rgb,
                                samples,
                                (gy - t.offset_y) * t.width + gx - t.offset_x,
                            ))
                        })
                    })
                    .unwrap_or(0);
            }
        }
        let max = grid.iter().copied().max().unwrap_or(0) as usize;
        let distance = self.scale as isize;
        SCRATCH.with_borrow_mut(|counts| {
            for (d, ((dy, dx), _)) in DIRECTIONS.iter().enumerate() {
                let (dy, dx) = (dy * distance, dx * distance);
                for y in 0..height as isize {
                    let y2 = y + dy;
                    if y2 < 0 || y2 >= height as isize {
                        continue;
                    }
                    for x in 0..width as isize {
                        let x2 = x + dx;
                        if x2 < 0 || x2 >= width as isize {
                            continue;
                        }
                        let a = grid[(y * width as isize + x) as usize];
                        let b = grid[(y2 * width as isize + x2) as usize];
                        if a != 0 && b != 0 {
                            counts.add(d, a, b);
                        }
                    }
                }
            }
            let features = counts.features(max + 1);
            counts.clear();
            features
        })
    }
}

thread_local! {
    /// Each thread's co-occurrence counts for objects: allocated once, only
    /// the counted entries cleared after each object.
    static SCRATCH: RefCell<Cooccurrences> = RefCell::new(Cooccurrences::new());
}

/// A channel value → its gray level: the normalized (0..1) value the image
/// reader stores in 8 bits as scikit-image's img_as_ubyte rounds it, then
/// rescaled to the gray levels as rescale_intensity and a cast to uint8 do.
pub(super) struct Levels {
    rescaled: [u8; 256],
}

impl Levels {
    fn new(gray_levels: u32) -> Self {
        let mut rescaled = [0u8; 256];
        for (u, level) in rescaled.iter_mut().enumerate() {
            *level = if gray_levels == 256 {
                u as u8
            } else {
                (u as f64 / 255.0 * (gray_levels - 1) as f64) as u8
            };
        }
        Self { rescaled }
    }

    fn of(&self, value: f32) -> u8 {
        let byte = (value as f64 * 255.0).round_ties_even().clamp(0.0, 255.0);
        self.rescaled[byte as usize]
    }
}

/// The symmetric co-occurrence counts of each direction, with the entries
/// counted so far (for clearing and for summing over them only).
struct Cooccurrences {
    counts: [Vec<u64>; 4],
    counted: [Vec<u16>; 4],
}

impl Cooccurrences {
    fn new() -> Self {
        Self {
            counts: std::array::from_fn(|_| vec![0; LEVELS * LEVELS]),
            counted: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// The pair (a, b) in direction `d` - and (b, a), the matrix is symmetric.
    #[inline]
    fn add(&mut self, d: usize, a: u8, b: u8) {
        for index in [
            a as usize * LEVELS + b as usize,
            b as usize * LEVELS + a as usize,
        ] {
            let count = &mut self.counts[d][index];
            if *count == 0 {
                self.counted[d].push(index as u16);
            }
            *count += 1;
        }
    }

    fn merge(&mut self, other: &Cooccurrences) {
        for d in 0..DIRECTIONS.len() {
            for &index in &other.counted[d] {
                let count = &mut self.counts[d][index as usize];
                if *count == 0 {
                    self.counted[d].push(index);
                }
                *count += other.counts[d][index as usize];
            }
        }
    }

    fn clear(&mut self) {
        for d in 0..DIRECTIONS.len() {
            for &index in &self.counted[d] {
                self.counts[d][index as usize] = 0;
            }
            self.counted[d].clear();
        }
    }

    /// The features of each direction, for gray levels 0..`n` (none if a
    /// direction has no pair, as mahotas raises then).
    fn features(&self, n: usize) -> Option<Features> {
        let mut out = [[0.0; 13]; 4];
        for d in 0..DIRECTIONS.len() {
            out[d] = haralick(&self.counts[d], &self.counted[d], n)?;
        }
        Some(out)
    }
}

/// -Σ p log2 p over the p > 0.
fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&p| p > 0.0)
        .map(|p| p * p.log2())
        .sum::<f64>()
}

/// mahotas' haralick_features for one symmetric co-occurrence matrix of
/// gray levels 0..`n`, summing over the counted entries only. Like mahotas:
/// the difference variance is the variance of the n values of P(|i - j|),
/// the correlation 1 where a standard deviation is 0. HXY2 = HX + HY (the
/// entropy of px·py, as px and py sum to 1).
fn haralick(counts: &[u64], counted: &[u16], n: usize) -> Option<[f64; 13]> {
    let total: u64 = counted.iter().map(|&i| counts[i as usize]).sum();
    if total == 0 {
        return None;
    }
    let total = total as f64;
    let mut px = vec![0.0; n];
    let mut px_plus_y = vec![0.0; 2 * n];
    let mut px_minus_y = vec![0.0; n];
    for &index in counted {
        let (i, j) = (index as usize / LEVELS, index as usize % LEVELS);
        let p = counts[index as usize] as f64 / total;
        px[j] += p;
        px_plus_y[i + j] += p;
        px_minus_y[i.abs_diff(j)] += p;
    }
    let ux: f64 = px.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
    let vx = px
        .iter()
        .enumerate()
        .map(|(k, p)| (k * k) as f64 * p)
        .sum::<f64>()
        - ux * ux;
    let sx = vx.sqrt();

    let (mut asm, mut ij, mut idm, mut hxy, mut hxy1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &index in counted {
        let (i, j) = (index as usize / LEVELS, index as usize % LEVELS);
        let p = counts[index as usize] as f64 / total;
        asm += p * p;
        ij += (i * j) as f64 * p;
        idm += p / (1 + (i.abs_diff(j)).pow(2)) as f64;
        hxy -= p * p.log2();
        // mahotas takes log2(1) = 0 where px·py is 0.
        let cross = px[i] * px[j];
        if cross != 0.0 {
            hxy1 -= p * cross.log2();
        }
    }
    let contrast: f64 = px_minus_y
        .iter()
        .enumerate()
        .map(|(k, p)| (k * k) as f64 * p)
        .sum();
    let correlation = if sx == 0.0 {
        1.0
    } else {
        (ij - ux * ux) / sx / sx
    };
    let sum_average: f64 = px_plus_y
        .iter()
        .enumerate()
        .map(|(k, p)| k as f64 * p)
        .sum();
    let sum_variance = px_plus_y
        .iter()
        .enumerate()
        .map(|(k, p)| (k * k) as f64 * p)
        .sum::<f64>()
        - sum_average * sum_average;
    let sum_entropy = entropy(&px_plus_y);
    let mean_minus = px_minus_y.iter().sum::<f64>() / n as f64;
    let difference_variance = px_minus_y
        .iter()
        .map(|p| (p - mean_minus) * (p - mean_minus))
        .sum::<f64>()
        / n as f64;
    let difference_entropy = entropy(&px_minus_y);
    let hx = entropy(&px);
    let info_meas_1 = if hx == 0.0 {
        hxy - hxy1
    } else {
        (hxy - hxy1) / hx
    };
    let hxy2 = 2.0 * hx;
    let info_meas_2 = (1.0 - (-2.0 * (hxy2 - hxy)).exp()).max(0.0).sqrt();
    Some([
        asm,
        contrast,
        correlation,
        vx,
        idm,
        sum_average,
        sum_variance,
        sum_entropy,
        hxy,
        difference_variance,
        difference_entropy,
        info_meas_1,
        info_meas_2,
    ])
}

/// Haralick's texture features.
pub(super) const HARALICK_1973: CitationMetadata = CitationMetadata {
    cite_key: "haralick1973textural",
    title: "Textural Features for Image Classification",
    authors: &["Robert M. Haralick", "K. Shanmugam", "Its'Hak Dinstein"],
    year: 1973,
    container: Some("IEEE Transactions on Systems, Man, and Cybernetics"),
    doi: Some("10.1109/TSMC.1973.4309314"),
    url: Some("https://doi.org/10.1109/TSMC.1973.4309314"),
    pages: Some("610-621"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::{
        ImageAlgorithm, MeasureImageTexture, MeasureObjectTexture, PipelineContext,
    };
    use crate::image::{ManagedImage, Point2d};
    use crate::object::ObjectInit;
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use crate::pipeline::scope::Scope;
    use bitvec::prelude::*;
    use evanalyzer_cfg::core_types::{MeasurementTarget, ObjectClass, ObjectId};
    use kornia_image::{Image, ImageSize};

    const CELL: ObjectClass = ObjectClass::Global(1);

    /// One case of `tests/texture_cellprofiler.json`.
    #[derive(serde::Deserialize)]
    struct Case {
        width: usize,
        height: usize,
        scale: u32,
        gray_levels: u32,
        image: Vec<f32>,
        labels: Vec<u32>,
        image_features: Vec<Option<f64>>,
        object_features: Vec<Option<f64>>,
    }

    /// The 16-bit image as the image reader stores it - normalized to 0..1
    /// (÷ 65535), as CellProfiler loads it too - split into a left and a
    /// right tile, the labels as objects.
    fn cache_of(case: &Case) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 16;
        let split = case.width / 2;
        for (x0, x1) in [(0, split), (split, case.width)] {
            let samples: Vec<f32> = (0..case.height)
                .flat_map(|y| (x0..x1).map(move |x| case.image[y * case.width + x] / 65535.0))
                .collect();
            let size = ImageSize {
                width: x1 - x0,
                height: case.height,
            };
            let image = ImageContainer::F32Gray(ManagedImage {
                data: Image::<f32, 1>::from_size_slice(size, &samples).unwrap(),
                tile_offset: Point2d { x: x0, y: 0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            });
            let tile = ImageTile {
                offset_x: x0,
                offset_y: 0,
                width: x1 - x0,
                height: case.height,
            };
            cache.add_to_channel_cache(Arc::new(image), 0, tile);
        }
        let max = case.labels.iter().copied().max().unwrap_or(0);
        for label in 1..=max {
            let pixels: Vec<(u32, u32)> = (0..case.labels.len())
                .filter(|&p| case.labels[p] == label)
                .map(|p| ((p % case.width) as u32, (p / case.width) as u32))
                .collect();
            let x0 = pixels.iter().map(|p| p.0).min().unwrap();
            let y0 = pixels.iter().map(|p| p.1).min().unwrap();
            let x1 = pixels.iter().map(|p| p.0).max().unwrap();
            let y1 = pixels.iter().map(|p| p.1).max().unwrap();
            let w = (x1 - x0 + 1) as usize;
            let mut mask = BitVec::<u64, Lsb0>::repeat(false, w * (y1 - y0 + 1) as usize);
            for (x, y) in &pixels {
                mask.set((y - y0) as usize * w + (x - x0) as usize, true);
            }
            let mut object = Object::new(ObjectInit {
                id: ObjectId(label as u128),
                bbox: [x0, y0, x1, y1],
                area: pixels.len(),
                mask_data: mask,
                ..Default::default()
            });
            object.add_object_class(CELL, Scope::Global).unwrap();
            cache.object_cache.insert(object.id.clone(), object);
        }
        cache
    }

    fn close(got: f64, expected: Option<f64>, what: &str) {
        match expected {
            Some(e) => assert!(
                (got - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{what}: {got} vs {e}"
            ),
            None => assert!(got.is_nan(), "{what}: {got} vs none"),
        }
    }

    fn run(case: &Case, directions: TextureDirections) -> GlobalPipelineCache {
        let mut cache = cache_of(case);
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        let image = MeasureImageTexture {
            id: "texture_01".into(),
            channel: ImageChannelIdx(0),
            scale: case.scale,
            gray_levels: case.gray_levels,
            directions,
        };
        image.execute(&mut ctx, &mut cache).unwrap();
        let objects = MeasureObjectTexture {
            id: "object_texture_01".into(),
            objects: CELL,
            channel: ImageChannelIdx(0),
            scale: case.scale,
            gray_levels: case.gray_levels,
            directions,
        };
        objects.execute(&mut ctx, &mut cache).unwrap();
        cache
    }

    fn image_value(cache: &GlobalPipelineCache, key: &str) -> f64 {
        let m = cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == key)
            .unwrap();
        let MeasurementValues::Image(values) = &m.values else {
            panic!("{key}: per image")
        };
        values.first().map_or(f64::NAN, |v| v.1)
    }

    fn object_values(cache: &GlobalPipelineCache, key: &str) -> Vec<(usize, f64)> {
        let m = cache
            .measurements
            .measurements
            .iter()
            .find(|m| m.info.key == key)
            .unwrap();
        let MeasurementValues::Objects(values) = &m.values else {
            panic!("{key}: per object")
        };
        values
            .iter()
            .map(|v| (v.object.0 as usize, v.value))
            .collect()
    }

    fn cases() -> Vec<Case> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/texture_cellprofiler.json"
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// Every feature of every direction, over the image and per object, as
    /// CellProfiler's MeasureTexture gives it (fixture from
    /// `tests/texture_cellprofiler.py`) - at three scales and gray levels.
    #[test]
    fn values_are_cellprofilers() {
        let cases = cases();
        assert_eq!(cases.len(), 3);
        for case in &cases {
            let cache = run(case, TextureDirections::Each);
            let what = format!("scale {} levels {}", case.scale, case.gray_levels);
            for (d, (_, angle)) in DIRECTIONS.iter().enumerate() {
                for (f, (id, ..)) in FEATURES.iter().enumerate() {
                    let expected = case.image_features[d * 13 + f];
                    close(
                        image_value(&cache, &format!("texture_01/{id}_{angle}")),
                        expected,
                        &format!("{what}: image {id}_{angle}"),
                    );
                    let values = object_values(&cache, &format!("object_texture_01/{id}_{angle}"));
                    assert_eq!(values.len() * 52, case.object_features.len());
                    for (label, value) in values {
                        let expected = case.object_features[(label - 1) * 52 + d * 13 + f];
                        close(
                            value,
                            expected,
                            &format!("{what}: object {label} {id}_{angle}"),
                        );
                    }
                }
            }
        }
    }

    /// The mean of the directions: per feature, the mean of the four.
    #[test]
    fn mean_is_the_mean_of_the_directions() {
        let case = &cases()[0];
        let cache = run(case, TextureDirections::Mean);
        for (f, (id, ..)) in FEATURES.iter().enumerate() {
            let expected = (0..4)
                .map(|d| case.image_features[d * 13 + f].unwrap())
                .sum::<f64>()
                / 4.0;
            close(
                image_value(&cache, &format!("texture_01/{id}")),
                Some(expected),
                id,
            );
        }
    }

    /// 13 values; per direction 52 - of the image or of the objects.
    #[test]
    fn outputs_follow_the_settings() {
        let step = MeasureImageTexture {
            id: "t".into(),
            channel: ImageChannelIdx(0),
            scale: 3,
            gray_levels: 256,
            directions: TextureDirections::Mean,
        };
        assert_eq!(step.measurement_outputs().len(), 13);
        let each = MeasureImageTexture {
            directions: TextureDirections::Each,
            ..step
        };
        let outputs = each.measurement_outputs();
        assert_eq!(outputs.len(), 52);
        assert_eq!(outputs[13].key, "t/angular_second_moment_135");
        let objects = MeasureObjectTexture {
            id: "o".into(),
            objects: CELL,
            channel: ImageChannelIdx(0),
            scale: 3,
            gray_levels: 256,
            directions: TextureDirections::Each,
        };
        let outputs = objects.measurement_outputs();
        assert_eq!(outputs.len(), 52);
        assert_eq!(outputs[0].key, "o/angular_second_moment_0");
        assert!(
            outputs
                .iter()
                .all(|o| o.target == MeasurementTarget::Object)
        );
    }

    /// Speed and RAM: a textured 16-bit image in 4 tiles (`BENCH_SIZE`²,
    /// default 2048), a 21 × 21 object every 30 px, all 4 directions.
    ///
    /// `cargo test --release -p evanalyzer_core --lib bench_measure_texture -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_measure_texture() {
        let size: usize = std::env::var("BENCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let gray_levels: u32 = std::env::var("BENCH_LEVELS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let mut seed = 5u64;
        let mut noise = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 8000) as f32
        };
        let values: Vec<f32> = (0..size * size)
            .map(|i| {
                let (x, y) = (i % size, i / size);
                let signal = if x % 30 < 21 && y % 30 < 21 {
                    20000.0
                } else {
                    0.0
                };
                // Normalized, as the image reader stores it.
                (signal + noise()) / 65535.0
            })
            .collect();
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 16;
        let half = size / 2;
        for (x0, y0) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let samples: Vec<f32> = (y0..y0 + half)
                .flat_map(|y| {
                    (x0..x0 + half)
                        .map(|x| values[y * size + x])
                        .collect::<Vec<_>>()
                })
                .collect();
            let image = ImageContainer::F32Gray(ManagedImage {
                data: Image::<f32, 1>::from_size_slice(
                    ImageSize {
                        width: half,
                        height: half,
                    },
                    &samples,
                )
                .unwrap(),
                tile_offset: Point2d { x: x0, y: y0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            });
            let tile = ImageTile {
                offset_x: x0,
                offset_y: y0,
                width: half,
                height: half,
            };
            cache.add_to_channel_cache(Arc::new(image), 0, tile);
        }
        drop(values);
        let mut id = 1u128;
        for y in (0..size - 21).step_by(30) {
            for x in (0..size - 21).step_by(30) {
                let mut object = Object::new(ObjectInit {
                    id: ObjectId(id),
                    bbox: [x as u32, y as u32, x as u32 + 20, y as u32 + 20],
                    area: 441,
                    mask_data: BitVec::<u64, Lsb0>::repeat(true, 441),
                    ..Default::default()
                });
                object.add_object_class(CELL, Scope::Global).unwrap();
                cache.object_cache.insert(object.id.clone(), object);
                id += 1;
            }
        }
        let image = MeasureImageTexture {
            id: "t".into(),
            channel: ImageChannelIdx(0),
            scale: 3,
            gray_levels,
            directions: TextureDirections::Mean,
        };
        let objects = MeasureObjectTexture {
            id: "o".into(),
            objects: CELL,
            channel: ImageChannelIdx(0),
            scale: 3,
            gray_levels,
            directions: TextureDirections::Mean,
        };
        let steps: [(&str, &dyn ImageAlgorithm); 2] = [("image", &image), ("objects", &objects)];
        for (label, step) in steps {
            let mut ctx = PipelineContext::without_image(
                std::path::PathBuf::new(),
                PipelineImageMeta::default(),
            )
            .unwrap();
            cache.measurements.measurements.clear();
            let start = std::time::Instant::now();
            step.execute(&mut ctx, &mut cache).unwrap();
            println!(
                "{size}² {gray_levels} levels, {label}, {} objects: {:.1} ms",
                id - 1,
                start.elapsed().as_secs_f64() * 1e3
            );
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        println!(
            "{}",
            status.lines().find(|l| l.starts_with("VmHWM")).unwrap()
        );
    }
}
