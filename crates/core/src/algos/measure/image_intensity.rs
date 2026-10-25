//! # Measure Image Intensity
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::ImageDimensionality;
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use crate::pipeline::pipeline_measure::{MeasurementRegion, Outputs};
use crate::{ImageTile, image::ImageContainer};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, MeasurementTarget,
    MeasurementUnit, max_gray_value,
};
use macros::CommandsMeta;
use rayon::prelude::*;

/// Measures the brightness of a channel across the whole image.
///
/// Total, mean, standard deviation, minimum, maximum, median, lower and
/// upper quartile and median absolute deviation - in gray values, like the
/// objects' intensities. It uses every pixel of the image, the background
/// too.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Image Intensity")]
pub struct MeasureImageIntensity {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("intensity_01/...").
    #[cmdsmeta(measurement_id = "intensity", display_name = "ID", summary = true)]
    pub id: String,

    /// The channel to measure.
    #[cmdsmeta(summary = true)]
    pub channel: ImageChannelIdx,
}

/// Histogram bins above 16 bits: the median and quartiles of such an image
/// are exact to 1/65535 of its range.
const MAX_BINS: usize = 1 << 16;

impl ImageAlgorithm for MeasureImageIntensity {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let channel = self.channel.0 as i32;
        let nr_of_bits = cache.image_meta.nr_of_bits;
        let bit_max = max_gray_value(nr_of_bits).ok_or_else(|| {
            InternalErrors::InvalidArgument(format!(
                "Image intensity: implausible bit depth {nr_of_bits} (expected 1-32)"
            ))
        })?;
        let bins = Bins::for_bit_max(bit_max);
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(InternalErrors::InvalidArgument(format!(
                "Image intensity: the image has no channel {channel}"
            )));
        }
        let stats = tiles
            .par_iter()
            .try_fold(
                || IntensityStats::new(bins),
                |mut stats, tile| {
                    let pixels = cache
                        .image_cache
                        .get(&CacheAddress::Channel((channel, tile.clone())))
                        .ok_or_else(|| {
                            InternalErrors::InvalidArgument(format!(
                                "Image intensity: the image has no channel {channel}"
                            ))
                        })?;
                    stats.add_tile(&pixels);
                    Ok::<_, InternalErrors>(stats)
                },
            )
            .try_reduce(|| IntensityStats::new(bins), |a, b| Ok(a.merge(b)))?;

        let values = [
            ("total", stats.total()),
            ("mean", stats.mean()),
            ("std", stats.std()),
            ("min", stats.min()),
            ("max", stats.max()),
            ("median", stats.quantile(0.5)),
            ("lower_quartile", stats.quantile(0.25)),
            ("upper_quartile", stats.quantile(0.75)),
            ("mad", stats.median_absolute_deviation()),
        ];
        for (id, value) in values {
            // No value for an image without pixels.
            cache.add_image_measurement(
                self.measurement_output(id),
                value.map(|v| (MeasurementRegion::Image, v)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Image Intensity"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    /// The tiles of the whole image, after they are all read.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    /// In gray values.
    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id).about(format!("channel {}", self.channel));
        [
            ("total", "Total intensity"),
            ("mean", "Mean intensity"),
            ("std", "Standard deviation of intensity"),
            ("min", "Minimum intensity"),
            ("max", "Maximum intensity"),
            ("median", "Median intensity"),
            ("lower_quartile", "Lower quartile intensity"),
            ("upper_quartile", "Upper quartile intensity"),
            ("mad", "Median absolute deviation of intensity"),
        ]
        .into_iter()
        .map(|(id, name)| outputs.image(id, name, MeasurementUnit::Intensity))
        .collect()
    }
}

/// The histogram's bins: one per gray value up to 16 bits, else
/// `MAX_BINS` over the range.
#[derive(Clone, Copy, Debug)]
struct Bins {
    count: usize,
    /// Gray value of bin `i`: `i * gray_per_bin`.
    gray_per_bin: f64,
}

impl Bins {
    fn for_bit_max(bit_max: f64) -> Self {
        let count = (bit_max as usize + 1).min(MAX_BINS);
        Self {
            count,
            gray_per_bin: bit_max / (count - 1).max(1) as f64,
        }
    }

    /// The bin of a normalized (0..1) pixel value.
    fn of(&self, normalized: f32) -> usize {
        let last = (self.count - 1) as f64;
        (normalized as f64 * last).round().clamp(0.0, last) as usize
    }

    fn gray(&self, bin: usize) -> f64 {
        bin as f64 * self.gray_per_bin
    }
}

/// What the measurement needs over the pixels, in gray values: count, mean
/// and sum of squared deviations (combined exactly, Chan et al.), min, max
/// and a histogram for the median and quartiles.
#[derive(Clone, Debug)]
struct IntensityStats {
    bins: Bins,
    n: f64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    histogram: Vec<u64>,
}

impl IntensityStats {
    fn new(bins: Bins) -> Self {
        Self {
            bins,
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            histogram: vec![0; bins.count],
        }
    }

    /// Adds a tile's pixels (two passes: the mean, then the deviations -
    /// stable also for bright images). Pixels that aren't numbers are left
    /// out.
    fn add_tile(&mut self, tile: &ImageContainer) {
        let rgb = matches!(tile, ImageContainer::F32Rgb(_));
        let slice = channel_pixel_slice(tile).unwrap_or_default();
        let len = if rgb { slice.len() / 3 } else { slice.len() };
        let gray_max = self.bins.gray(self.bins.count - 1);
        let pixels = || {
            (0..len)
                .map(|i| sample_channel_pixel(rgb, slice, i))
                .filter(|v| v.is_finite())
        };
        let (mut n, mut sum) = (0.0, 0.0);
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in pixels() {
            let gray = v as f64 * gray_max;
            n += 1.0;
            sum += gray;
            min = min.min(gray);
            max = max.max(gray);
            self.histogram[self.bins.of(v)] += 1;
        }
        if n == 0.0 {
            return;
        }
        let mean = sum / n;
        let m2 = pixels().map(|v| (v as f64 * gray_max - mean).powi(2)).sum();
        self.merge_moments(n, mean, m2, min, max);
    }

    /// Adds the count, mean, sum of squared deviations, min and max of
    /// other pixels.
    fn merge_moments(&mut self, n: f64, mean: f64, m2: f64, min: f64, max: f64) {
        if n == 0.0 {
            return;
        }
        let total = self.n + n;
        let delta = mean - self.mean;
        self.m2 += m2 + delta * delta * self.n * n / total;
        self.mean += delta * n / total;
        self.n = total;
        self.min = self.min.min(min);
        self.max = self.max.max(max);
    }

    fn merge(mut self, other: Self) -> Self {
        self.merge_moments(other.n, other.mean, other.m2, other.min, other.max);
        for (a, b) in self.histogram.iter_mut().zip(&other.histogram) {
            *a += b;
        }
        self
    }

    fn has_pixels(&self) -> bool {
        self.n > 0.0
    }

    fn total(&self) -> Option<f64> {
        self.has_pixels().then(|| self.mean * self.n)
    }

    fn mean(&self) -> Option<f64> {
        self.has_pixels().then_some(self.mean)
    }

    /// Of the population (all pixels are measured, not a sample of them).
    fn std(&self) -> Option<f64> {
        self.has_pixels().then(|| (self.m2 / self.n).sqrt())
    }

    fn min(&self) -> Option<f64> {
        self.has_pixels().then_some(self.min)
    }

    fn max(&self) -> Option<f64> {
        self.has_pixels().then_some(self.max)
    }

    /// The `q` quantile (0..1), linearly interpolated between the two
    /// nearest pixels like NumPy's default.
    fn quantile(&self, q: f64) -> Option<f64> {
        let sorted: Vec<(f64, u64)> = self
            .histogram
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(bin, count)| (self.bins.gray(bin), *count))
            .collect();
        quantile_of_counts(&sorted, q)
    }

    /// The median of the pixels' distances to the median.
    fn median_absolute_deviation(&self) -> Option<f64> {
        let median = self.quantile(0.5)?;
        let mut deviations: Vec<(f64, u64)> = self
            .histogram
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(bin, count)| ((self.bins.gray(bin) - median).abs(), *count))
            .collect();
        deviations.sort_by(|a, b| a.0.total_cmp(&b.0));
        quantile_of_counts(&deviations, 0.5)
    }
}

/// The `q` quantile of values sorted ascending, each with how often it
/// occurs: linearly interpolated between the two nearest ranks.
fn quantile_of_counts(sorted: &[(f64, u64)], q: f64) -> Option<f64> {
    let n: u64 = sorted.iter().map(|(_, count)| count).sum();
    if n == 0 {
        return None;
    }
    let position = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let (lower, upper) = (position.floor() as u64, position.ceil() as u64);
    let at = |rank: u64| {
        let mut seen = 0;
        sorted
            .iter()
            .find(|(_, count)| {
                seen += count;
                seen > rank
            })
            .map(|(value, _)| *value)
            .expect("rank below the count")
    };
    let (low, high) = (at(lower), at(upper));
    Some(low + (high - low) * (position - lower as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    fn tile_image(values: &[f32]) -> Arc<ImageContainer> {
        Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: values.len(),
                    height: 1,
                },
                values.to_vec(),
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        }))
    }

    /// An 8-bit image whose channel 0 is `tiles` (gray values), one tile
    /// after the other.
    fn cache_with_gray_tiles(tiles: &[&[f32]]) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 8;
        let mut offset = 0;
        for gray in tiles {
            let normalized: Vec<f32> = gray.iter().map(|g| g / 255.0).collect();
            let tile = ImageTile {
                offset_x: offset,
                offset_y: 0,
                width: gray.len(),
                height: 1,
            };
            offset += gray.len();
            cache.add_to_channel_cache(tile_image(&normalized), 0, tile);
        }
        cache
    }

    fn run(
        step: &MeasureImageIntensity,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, cache)
    }

    fn values_by_key(cache: &GlobalPipelineCache) -> Vec<(String, Option<f64>)> {
        cache
            .measurements
            .measurements
            .iter()
            .map(|m| {
                let value = match &m.values {
                    MeasurementValues::Image(values) => values.first().map(|(_, v)| *v),
                    MeasurementValues::Objects(_) => panic!("an image measurement"),
                };
                (m.info.key.clone(), value)
            })
            .collect()
    }

    /// Every value over all tiles, against numbers worked out by hand
    /// (NumPy gives the same: `np.percentile(x, 25)` etc.).
    #[test]
    fn measures_every_value_over_all_tiles() {
        // 2, 4, 4, 4, 5, 5, 7, 9: mean 5, population std 2.
        let mut cache = cache_with_gray_tiles(&[&[2.0, 4.0, 4.0], &[4.0, 5.0, 5.0, 7.0, 9.0]]);
        let step = MeasureImageIntensity {
            id: "dapi".into(),
            channel: ImageChannelIdx(0),
        };
        run(&step, &mut cache).unwrap();

        let values = values_by_key(&cache);
        let expected = [
            ("total", 40.0),
            ("mean", 5.0),
            ("std", 2.0),
            ("min", 2.0),
            ("max", 9.0),
            ("median", 4.5),
            ("lower_quartile", 4.0),
            ("upper_quartile", 5.5),
            // |x - 4.5|: 0.5 0.5 0.5 0.5 0.5 2.5 2.5 4.5 -> 0.5
            ("mad", 0.5),
        ];
        assert_eq!(values.len(), expected.len());
        for ((key, value), (stat, want)) in values.iter().zip(expected) {
            assert_eq!(key, &format!("dapi/{stat}"));
            let value = value.unwrap();
            // Pixels are f32 (0..1): gray values come back to ~1e-7.
            assert!((value - want).abs() < 1e-4, "{stat}: {value} != {want}");
        }
        assert!(
            cache
                .measurements
                .measurements
                .iter()
                .all(|m| m.info.step == "dapi" && m.info.unit == MeasurementUnit::Intensity)
        );
    }

    /// Pixels that aren't numbers are left out; with none left there's no
    /// value.
    #[test]
    fn pixels_that_arent_numbers_are_left_out() {
        let mut cache = cache_with_gray_tiles(&[&[f32::NAN, 10.0, 20.0]]);
        let step = MeasureImageIntensity {
            id: "i".into(),
            channel: ImageChannelIdx(0),
        };
        run(&step, &mut cache).unwrap();
        let values = values_by_key(&cache);
        assert_eq!(values[1].0, "i/mean");
        assert!((values[1].1.unwrap() - 15.0).abs() < 1e-4);

        let mut cache = cache_with_gray_tiles(&[&[f32::NAN]]);
        run(&step, &mut cache).unwrap();
        assert!(values_by_key(&cache).iter().all(|(_, v)| v.is_none()));
    }

    #[test]
    fn a_channel_the_image_lacks_is_an_error() {
        let mut cache = cache_with_gray_tiles(&[&[1.0]]);
        let step = MeasureImageIntensity {
            id: "i".into(),
            channel: ImageChannelIdx(2),
        };
        let err = run(&step, &mut cache).unwrap_err().to_string();
        assert!(err.contains("no channel 2"), "{err}");
    }

    /// Above 16 bits the histogram has 65536 bins over the range.
    #[test]
    fn deep_images_bin_over_the_range() {
        let bins = Bins::for_bit_max(max_gray_value(32).unwrap());
        assert_eq!(bins.count, MAX_BINS);
        assert_eq!(bins.of(1.0), MAX_BINS - 1);
        assert_eq!(bins.gray(MAX_BINS - 1), u32::MAX as f64);
        let bins = Bins::for_bit_max(255.0);
        assert_eq!(
            (bins.count, bins.of(128.0 / 255.0), bins.gray(128)),
            (256, 128, 128.0)
        );
    }

    #[test]
    fn quantiles_interpolate_between_ranks() {
        let counts = [(1.0, 1), (3.0, 2), (10.0, 1)];
        assert_eq!(quantile_of_counts(&counts, 0.0), Some(1.0));
        assert_eq!(quantile_of_counts(&counts, 0.5), Some(3.0));
        assert_eq!(quantile_of_counts(&counts, 1.0), Some(10.0));
        // Position 0.25 * 3 = 0.75 between 1 and 3.
        assert_eq!(quantile_of_counts(&counts, 0.25), Some(2.5));
        assert_eq!(quantile_of_counts(&[], 0.5), None);
    }
}
