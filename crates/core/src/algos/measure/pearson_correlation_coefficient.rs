//! # Pearson correlation coefficient
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_cache::{CacheAddress, channel_pixel_slice, sample_channel_pixel};
use crate::pipeline::pipeline_measure::{
    Measurement, MeasurementRegion, MeasurementValues, Outputs,
};
use crate::{ImageTile, image::ImageContainer};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, MeasurementTarget,
    MeasurementUnit,
};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::sync::Arc;

/// Measures how two channels go together across the whole image.
///
/// The Pearson correlation coefficient, from -1 (one bright where the other
/// is dark) over 0 (unrelated) to 1 (bright together). It uses every pixel
/// of the image, the background too - a large dark background raises it.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Pearson Correlation")]
pub struct PearsonCorrelationCoefficient {
    /// The measurement's name - its values in the results start with it.
    #[cmdsmeta(measurement_name = "pearson", summary = true)]
    pub name: String,

    /// The first channel.
    #[cmdsmeta(summary = true)]
    pub channel_a: ImageChannelIdx,

    /// The second channel.
    #[cmdsmeta(default = ImageChannelIdx(1), summary = true)]
    pub channel_b: ImageChannelIdx,
}

impl ImageAlgorithm for PearsonCorrelationCoefficient {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let (a, b) = (self.channel_a.0 as i32, self.channel_b.0 as i32);
        // Every tile of channel A, with the same tile of channel B.
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(channel, _)| *channel == a)
            .map(|(_, tile)| tile)
            .collect();
        let sums = tiles
            .par_iter()
            .map(|tile| {
                let pixels = |channel| {
                    cache
                        .image_cache
                        .get(&CacheAddress::Channel((channel, tile.clone())))
                        .ok_or_else(|| {
                            InternalErrors::InvalidArgument(format!(
                                "Pearson correlation: the image has no channel {channel}"
                            ))
                        })
                };
                Ok::<_, InternalErrors>(Some(PairSums::of(&pixels(a)?, &pixels(b)?)))
            })
            .try_reduce(|| None, |x, y| Ok(PairSums::merge(x, y)))?;
        if tiles.is_empty() {
            return Err(InternalErrors::InvalidArgument(format!(
                "Pearson correlation: the image has no channel {a}"
            )));
        }

        let r = sums.and_then(|s| s.pearson());
        // No value if a channel doesn't vary (e.g. an empty image).
        cache.add_image_measurement(
            self.measurement_output("pearson"),
            r.map(|r| (MeasurementRegion::Image, r)),
        );
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Pearson correlation coefficient"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&PEARSON_1895]
    }

    /// The tiles of the whole image, after they are all read.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let about = format!("channel {} × channel {}", self.channel_a, self.channel_b);
        vec![
            Outputs::of(&self.name)
                .about(about)
                .image("pearson", "Pearson", MeasurementUnit::None)
                .range(-1.0, 1.0),
        ]
    }
}

/// Where the coefficient was first published.
const PEARSON_1895: CitationMetadata = CitationMetadata {
    cite_key: "pearson1895notes",
    title: "Notes on regression and inheritance in the case of two parents",
    authors: &["Karl Pearson"],
    year: 1895,
    container: Some("Proceedings of the Royal Society of London"),
    doi: Some("10.1098/rspl.1895.0041"),
    url: Some("https://doi.org/10.1098/rspl.1895.0041"),
    pages: Some("240-242"),
};

/// The sums Pearson needs over pixel pairs (x, y): their count, means and
/// sums of squared deviations and of co-deviations. Two of them combine
/// exactly (Chan et al.), so tiles are summed apart.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PairSums {
    n: f64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    co: f64,
}

impl PairSums {
    /// Over the pixels of two tiles of the same place (two passes: the
    /// means, then the deviations - stable also for bright images).
    fn of(a: &Arc<ImageContainer>, b: &Arc<ImageContainer>) -> Self {
        fn pixels(c: &ImageContainer) -> (bool, &[f32], usize) {
            let rgb = matches!(c, ImageContainer::F32Rgb(_));
            let slice = channel_pixel_slice(c).unwrap_or_default();
            let len = if rgb { slice.len() / 3 } else { slice.len() };
            (rgb, slice, len)
        }
        let ((rgb_a, xs, len_a), (rgb_b, ys, len_b)) = (pixels(a), pixels(b));
        let len = len_a.min(len_b);
        let at = |i: usize| {
            (
                sample_channel_pixel(rgb_a, xs, i) as f64,
                sample_channel_pixel(rgb_b, ys, i) as f64,
            )
        };
        let n = len as f64;
        let (sum_x, sum_y) = (0..len)
            .map(at)
            .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
        let (mean_x, mean_y) = if len > 0 {
            (sum_x / n, sum_y / n)
        } else {
            (0.0, 0.0)
        };
        let (m2_x, m2_y, co) = (0..len)
            .map(at)
            .fold((0.0, 0.0, 0.0), |(m2x, m2y, co), (x, y)| {
                let (dx, dy) = (x - mean_x, y - mean_y);
                (m2x + dx * dx, m2y + dy * dy, co + dx * dy)
            });
        Self {
            n,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            co,
        }
    }

    fn merge(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            (a, b) => return a.or(b),
        };
        let n = a.n + b.n;
        if n == 0.0 {
            return Some(a);
        }
        let (dx, dy) = (b.mean_x - a.mean_x, b.mean_y - a.mean_y);
        let weight = a.n * b.n / n;
        Some(Self {
            n,
            mean_x: a.mean_x + dx * b.n / n,
            mean_y: a.mean_y + dy * b.n / n,
            m2_x: a.m2_x + b.m2_x + dx * dx * weight,
            m2_y: a.m2_y + b.m2_y + dy * dy * weight,
            co: a.co + b.co + dx * dy * weight,
        })
    }

    /// The coefficient, `None` if a channel doesn't vary.
    fn pearson(&self) -> Option<f64> {
        (self.m2_x > 0.0 && self.m2_y > 0.0).then(|| self.co / (self.m2_x * self.m2_y).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use kornia_image::{Image, ImageSize};

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
        }))
    }

    fn pearson(x: &[f32], y: &[f32]) -> Option<f64> {
        PairSums::of(&tile_image(x), &tile_image(y)).pearson()
    }

    #[test]
    fn together_opposite_and_flat() {
        let x = [0.1, 0.4, 0.2, 0.9, 0.5];
        assert!((pearson(&x, &x).unwrap() - 1.0).abs() < 1e-12);
        let inverted: Vec<f32> = x.iter().map(|v| 1.0 - v).collect();
        assert!((pearson(&x, &inverted).unwrap() + 1.0).abs() < 1e-12);
        let scaled: Vec<f32> = x.iter().map(|v| 3.0 * v + 7.0).collect();
        assert!(
            (pearson(&x, &scaled).unwrap() - 1.0).abs() < 1e-6,
            "scale and offset don't matter"
        );
        assert_eq!(pearson(&x, &[0.3; 5]), None, "a channel that doesn't vary");
    }

    /// Tiles summed apart and combined give the coefficient of the whole.
    #[test]
    fn tiles_combine_to_the_whole_images_coefficient() {
        let x = [0.1, 0.4, 0.2, 0.9, 0.5, 0.7, 0.3];
        let y = [0.2, 0.3, 0.1, 0.8, 0.7, 0.4, 0.4];
        let whole = PairSums::of(&tile_image(&x), &tile_image(&y));
        let first = PairSums::of(&tile_image(&x[..3]), &tile_image(&y[..3]));
        let rest = PairSums::of(&tile_image(&x[3..]), &tile_image(&y[3..]));
        let merged = PairSums::merge(Some(first), Some(rest)).unwrap();
        assert!((merged.pearson().unwrap() - whole.pearson().unwrap()).abs() < 1e-12);
        assert_eq!(PairSums::merge(None, Some(whole)), Some(whole));
    }

    fn cache_with_channels(channels: &[(i32, [&[f32]; 2])]) -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        for (channel, tiles) in channels {
            for (i, values) in tiles.iter().enumerate() {
                let tile = ImageTile {
                    offset_x: i * values.len(),
                    offset_y: 0,
                    width: values.len(),
                    height: 1,
                };
                cache.add_to_channel_cache(tile_image(values), *channel, tile);
            }
        }
        cache
    }

    fn run(
        step: &PearsonCorrelationCoefficient,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, cache)
    }

    /// Over every tile of the image; its key starts with the step's name.
    #[test]
    fn measures_the_whole_image_under_the_steps_name() {
        let mut cache = cache_with_channels(&[
            (0, [&[0.1, 0.4, 0.2], &[0.9, 0.5, 0.7]]),
            (1, [&[0.2, 0.3, 0.1], &[0.8, 0.7, 0.4]]),
        ]);
        let step = PearsonCorrelationCoefficient {
            name: "nuclei_coloc".into(),
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
        };
        run(&step, &mut cache).unwrap();

        let [measurement] = &cache.measurements.measurements[..] else {
            panic!("one measurement");
        };
        assert_eq!(measurement.info.key, "nuclei_coloc/pearson");
        assert_eq!(measurement.info.step, "nuclei_coloc");
        let expected = pearson(
            &[0.1, 0.4, 0.2, 0.9, 0.5, 0.7],
            &[0.2, 0.3, 0.1, 0.8, 0.7, 0.4],
        );
        match &measurement.values {
            MeasurementValues::Image(values) => {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].0, MeasurementRegion::Image);
                assert!((values[0].1 - expected.unwrap()).abs() < 1e-9);
            }
            MeasurementValues::Objects(_) => panic!("an image measurement"),
        }
    }

    #[test]
    fn a_channel_the_image_lacks_is_an_error() {
        let mut cache = cache_with_channels(&[(0, [&[0.1, 0.4], &[0.9, 0.5]])]);
        let step = PearsonCorrelationCoefficient {
            name: "p".into(),
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(3),
        };
        let err = run(&step, &mut cache).unwrap_err().to_string();
        assert!(err.contains("no channel 3"), "{err}");
    }
}
