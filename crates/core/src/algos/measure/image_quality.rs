//! # Measure Image Quality
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Values to tell a usable image from a blurry, noisy or overexposed one.
//! Focus score, power log-log slope and the percent of pixels at the
//! maximum/minimum follow the established definitions (Bray et al. 2012),
//! so values stay comparable with other tools and published thresholds.
//! The noise is Immerkær's estimate.
//!
//! Parts of the calculations are ported from CellProfiler and centrosome,
//! Copyright © 2003 - 2026 Broad Institute, Inc., BSD-3-Clause - see the
//! third-party notices (About > Licenses).

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
use rustfft::{FftPlanner, num_complex::Complex};

/// Measures how usable a channel of the image is: sharp, not noisy, not
/// overexposed.
///
/// Focus score (the variance over the mean - low: blurry), power log-log
/// slope (how fast fine detail fades - more negative: blurrier), the
/// percent of pixels at the image's maximum and minimum (high: saturated or
/// clipped) - all by their standard definitions - and the noise (σ in gray
/// values, Immerkær's estimate). Use them with "Exclude Image" to leave bad
/// images out of the results.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Measure Image Quality")]
pub struct MeasureImageQuality {
    /// The step's ID, unique in the project - its values' keys start with
    /// it ("quality_01/...").
    #[cmdsmeta(measurement_id = "quality", display_name = "ID", summary = true)]
    pub id: String,

    /// The channel to measure.
    #[cmdsmeta(summary = true)]
    pub channel: ImageChannelIdx,
}

/// The power spectrum uses at most this many pixels per side, from the
/// image's middle (the whole image held as complex numbers, 16 bytes each).
const MAX_SPECTRUM_SIDE: usize = 4096;

impl ImageAlgorithm for MeasureImageQuality {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let channel = self.channel.0 as i32;
        let nr_of_bits = cache.image_meta.nr_of_bits;
        let bit_max = max_gray_value(nr_of_bits).ok_or_else(|| {
            InternalErrors::InvalidArgument(format!(
                "Image quality: implausible bit depth {nr_of_bits} (expected 1-32)"
            ))
        })?;
        let no_channel = || {
            InternalErrors::InvalidArgument(format!(
                "Image quality: the image has no channel {channel}"
            ))
        };
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(no_channel());
        }
        let pixels_of = |tile: &ImageTile| {
            cache
                .image_cache
                .get(&CacheAddress::Channel((channel, tile.clone())))
                .ok_or_else(no_channel)
        };
        let stats = tiles
            .par_iter()
            .map(|tile| {
                Ok::<_, InternalErrors>(TileStats::of(&Plane::of(&*pixels_of(tile)?, tile)))
            })
            .try_reduce(TileStats::default, |a, b| Ok(a.merge(b)))?;
        let slope = power_log_log_slope(&spectrum_window(&tiles, |t| pixels_of(t))?);

        let values = [
            ("focus_score", stats.focus_score()),
            ("power_log_log_slope", slope),
            ("percent_maximal", stats.percent(stats.n_max)),
            ("percent_minimal", stats.percent(stats.n_min)),
            ("noise", stats.noise().map(|sigma| sigma * bit_max)),
        ];
        for (id, value) in values {
            cache.add_image_measurement(
                self.measurement_output(id),
                value.map(|v| (MeasurementRegion::Image, v)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Measure Image Quality"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&BRAY_2012, &IMMERKAER_1996]
    }

    /// The tiles of the whole image, after they are all read.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id).about(format!("channel {}", self.channel));
        vec![
            outputs.image("focus_score", "Focus score", MeasurementUnit::None),
            outputs.image(
                "power_log_log_slope",
                "Power log-log slope",
                MeasurementUnit::None,
            ),
            outputs
                .image(
                    "percent_maximal",
                    "Pixels at the maximum",
                    MeasurementUnit::Percent,
                )
                .range(0.0, 100.0),
            outputs
                .image(
                    "percent_minimal",
                    "Pixels at the minimum",
                    MeasurementUnit::Percent,
                )
                .range(0.0, 100.0),
            outputs.image("noise", "Noise σ", MeasurementUnit::Intensity),
        ]
    }
}

/// The image quality measures: how they were chosen and tested.
const BRAY_2012: CitationMetadata = CitationMetadata {
    cite_key: "bray2012workflow",
    title: "Workflow and metrics for image quality control in large-scale high-content screens",
    authors: &[
        "Mark-Anthony Bray",
        "Adam N. Fraser",
        "Thomas P. Hasaka",
        "Anne E. Carpenter",
    ],
    year: 2012,
    container: Some("Journal of Biomolecular Screening"),
    doi: Some("10.1177/1087057111420292"),
    url: Some("https://doi.org/10.1177/1087057111420292"),
    pages: Some("266-274"),
};

/// The noise estimate.
const IMMERKAER_1996: CitationMetadata = CitationMetadata {
    cite_key: "immerkaer1996fast",
    title: "Fast Noise Variance Estimation",
    authors: &["John Immerkær"],
    year: 1996,
    container: Some("Computer Vision and Image Understanding"),
    doi: Some("10.1006/cviu.1996.0060"),
    url: Some("https://doi.org/10.1006/cviu.1996.0060"),
    pages: Some("300-302"),
};

/// One tile's pixels (normalized 0..1, RGB as luminance), row-major.
struct Plane {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

impl Plane {
    fn of(container: &ImageContainer, tile: &ImageTile) -> Self {
        let rgb = matches!(container, ImageContainer::F32Rgb(_));
        let slice = channel_pixel_slice(container).unwrap_or_default();
        let len = if rgb { slice.len() / 3 } else { slice.len() };
        let (width, height) = (tile.width, tile.height.min(len / tile.width.max(1)));
        let values = (0..width * height)
            .map(|i| sample_channel_pixel(rgb, slice, i) as f64)
            .collect();
        Self {
            width,
            height,
            values,
        }
    }
}

/// What the values need over the pixels: count, mean and sum of squared
/// deviations (combined exactly, Chan et al.), the extremes with how often
/// they occur, and Immerkær's sum over the tiles' inner pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct TileStats {
    n: f64,
    mean: f64,
    m2: f64,
    max: f64,
    n_max: f64,
    min: f64,
    n_min: f64,
    noise_sum: f64,
    noise_pixels: f64,
}

impl TileStats {
    fn of(plane: &Plane) -> Self {
        let values = &plane.values;
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let m2 = values.iter().map(|v| (v - mean).powi(2)).sum();
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        // Immerkær's mask, on every pixel with all 8 neighbours in the tile.
        let (w, h) = (plane.width, plane.height);
        let mut noise_sum = 0.0;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w - 1 {
                let at = |dx: isize, dy: isize| {
                    values[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
                };
                let response = at(-1, -1) - 2.0 * at(0, -1) + at(1, -1) - 2.0 * at(-1, 0)
                    + 4.0 * at(0, 0)
                    - 2.0 * at(1, 0)
                    + at(-1, 1)
                    - 2.0 * at(0, 1)
                    + at(1, 1);
                noise_sum += response.abs();
            }
        }
        Self {
            n,
            mean,
            m2,
            max,
            n_max: values.iter().filter(|v| **v == max).count() as f64,
            min,
            n_min: values.iter().filter(|v| **v == min).count() as f64,
            noise_sum,
            noise_pixels: (w.saturating_sub(2) * h.saturating_sub(2)) as f64,
        }
    }

    fn merge(self, other: Self) -> Self {
        if other.n == 0.0 {
            return self;
        }
        if self.n == 0.0 {
            return other;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        let extreme = |a: f64, na: f64, b: f64, nb: f64, better: fn(f64, f64) -> bool| {
            if a == b {
                (a, na + nb)
            } else if better(a, b) {
                (a, na)
            } else {
                (b, nb)
            }
        };
        let (max, n_max) = extreme(self.max, self.n_max, other.max, other.n_max, |a, b| a > b);
        let (min, n_min) = extreme(self.min, self.n_min, other.min, other.n_min, |a, b| a < b);
        Self {
            n,
            mean: self.mean + delta * other.n / n,
            m2: self.m2 + other.m2 + delta * delta * self.n * other.n / n,
            max,
            n_max,
            min,
            n_min,
            noise_sum: self.noise_sum + other.noise_sum,
            noise_pixels: self.noise_pixels + other.noise_pixels,
        }
    }

    /// The focus score: the variance over the mean (0 for a black image,
    /// where the ratio is undefined).
    fn focus_score(&self) -> Option<f64> {
        (self.n > 0.0).then(|| {
            if self.mean > 0.0 {
                self.m2 / (self.n * self.mean)
            } else {
                0.0
            }
        })
    }

    fn percent(&self, count: f64) -> Option<f64> {
        (self.n > 0.0).then(|| 100.0 * count / self.n)
    }

    /// Immerkær's σ (normalized): `sqrt(π/2) · Σ|mask| / (6 · pixels)`.
    fn noise(&self) -> Option<f64> {
        (self.noise_pixels > 0.0).then(|| {
            (std::f64::consts::PI / 2.0).sqrt() * self.noise_sum / (6.0 * self.noise_pixels)
        })
    }
}

/// The part of the image the power spectrum uses: what the tiles cover, at
/// most `MAX_SPECTRUM_SIDE` per side from its middle.
fn spectrum_window<P: AsRef<ImageContainer>>(
    tiles: &[ImageTile],
    pixels_of: impl Fn(&ImageTile) -> Result<P, InternalErrors>,
) -> Result<Plane, InternalErrors> {
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
    let (width, height) = (
        (x1 - x0).min(MAX_SPECTRUM_SIDE),
        (y1 - y0).min(MAX_SPECTRUM_SIDE),
    );
    let (wx0, wy0) = (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2);
    let mut values = vec![0.0; width * height];
    for tile in tiles {
        let (tx1, ty1) = (tile.offset_x + tile.width, tile.offset_y + tile.height);
        if tx1 <= wx0 || ty1 <= wy0 || tile.offset_x >= wx0 + width || tile.offset_y >= wy0 + height
        {
            continue;
        }
        let plane = Plane::of(pixels_of(tile)?.as_ref(), tile);
        for ty in 0..plane.height {
            let y = tile.offset_y + ty;
            if y < wy0 || y >= wy0 + height {
                continue;
            }
            for tx in 0..plane.width {
                let x = tile.offset_x + tx;
                if x >= wx0 && x < wx0 + width {
                    values[(y - wy0) * width + (x - wx0)] = plane.values[ty * plane.width + tx];
                }
            }
        }
    }
    Ok(Plane {
        width,
        height,
        values,
    })
}

/// The power log-log slope: the slope of log(power) over log(radius) of
/// the radial power spectrum, radii 2 .. min(side)/8 (the lowest
/// frequencies and the noisy highest ones left out). `None` for an image
/// without any variation (the spectrum is empty, so there is no slope).
///
/// No normalization of the image (e.g. dividing by its median absolute
/// deviation) is done: that scales every power alike and leaves the slope
/// as it is.
fn power_log_log_slope(plane: &Plane) -> Option<f64> {
    let (w, h) = (plane.width, plane.height);
    let values = &plane.values;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if values.is_empty() || min == max {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut data: Vec<Complex<f64>> = values.iter().map(|v| Complex::new(v - mean, 0.0)).collect();
    // 2D FFT: the rows, then the columns (as the rows of the transpose).
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(w).process(&mut data);
    let mut columns = vec![Complex::new(0.0, 0.0); w * h];
    for y in 0..h {
        for x in 0..w {
            columns[x * h + y] = data[y * w + x];
        }
    }
    drop(data);
    planner.plan_fft_forward(h).process(&mut columns);

    // The radius of frequency (row i, column j), with the frequencies
    // mirrored about the centre so negative ones fold onto positive ones:
    // min(i, h-1-i), min(j, w-1-j); labels floor(r) + 1 in 2 .. max_width.
    let max_width = (w.min(h) as f64 / 8.0).floor() as usize;
    if max_width <= 2 {
        return None;
    }
    let mut magnitude = vec![0.0; max_width];
    let mut power = vec![0.0; max_width];
    for x in 0..w {
        let cj = x.min(w - 1 - x) as f64;
        for y in 0..h {
            let ri = y.min(h - 1 - y) as f64;
            let label = (ri * ri + cj * cj).sqrt().floor() as usize + 1;
            if (2..max_width).contains(&label) {
                let mag = columns[x * h + y].norm();
                magnitude[label] += mag;
                power[label] += mag * mag;
            }
        }
    }
    let points: Vec<(f64, f64)> = (2..max_width)
        .filter(|&r| magnitude[r] > 0.0 && power[r] > 0.0)
        .map(|r| ((r as f64).ln(), power[r].ln()))
        .collect();
    if points.len() < 2 {
        return None;
    }
    // Least squares with an intercept: the slope.
    let n = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    Some(sxy / sxx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ManagedImage, Point2d};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use crate::pipeline::pipeline_measure::MeasurementValues;
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        width: usize,
        height: usize,
        image: Vec<f64>,
        focus_score: f64,
        power_log_log_slope: Option<f64>,
        percent_maximal: f64,
        percent_minimal: f64,
        noise: f64,
    }

    fn cases() -> Vec<Case> {
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/image_quality.json");
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        fixture.cases
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    /// The reference values for the same images (fixture
    /// `tests/image_quality.json`: sharp, blurred, clipped
    /// 8-bit, odd size); the noise against SciPy's convolution.
    #[test]
    fn values_match_reference() {
        let cases = cases();
        assert_eq!(cases.len(), 4);
        for case in cases {
            let plane = Plane {
                width: case.width,
                height: case.height,
                values: case.image.iter().map(|&v| v as f32 as f64).collect(),
            };
            let stats = TileStats::of(&plane);
            let name = &case.name;
            let focus = stats.focus_score().unwrap();
            assert!(close(focus, case.focus_score), "{name}: focus {focus}");
            let max = stats.percent(stats.n_max).unwrap();
            assert!(close(max, case.percent_maximal), "{name}: % max {max}");
            let min = stats.percent(stats.n_min).unwrap();
            assert!(close(min, case.percent_minimal), "{name}: % min {min}");
            let noise = stats.noise().unwrap();
            assert!(close(noise, case.noise), "{name}: noise {noise}");
            let slope = power_log_log_slope(&plane);
            let expected = case.power_log_log_slope.unwrap();
            assert!(
                (slope.unwrap() - expected).abs() < 1e-9,
                "{name}: slope {slope:?} != {expected}"
            );
        }
    }

    /// Blur lowers the focus score and makes the slope steeper.
    #[test]
    fn blur_shows() {
        let cases = cases();
        let [sharp, blurred] =
            ["sharp", "blurred"].map(|n| cases.iter().find(|c| c.name == n).unwrap());
        assert!(blurred.focus_score < sharp.focus_score);
        assert!(blurred.power_log_log_slope.unwrap() < sharp.power_log_log_slope.unwrap() - 1.0);
    }

    /// Tiles combine to the whole image's values (the noise skips each
    /// tile's border pixels).
    #[test]
    fn tiles_combine_to_the_whole() {
        let case = cases()
            .into_iter()
            .find(|c| c.name == "saturated_8_bit")
            .unwrap();
        let values: Vec<f64> = case.image.iter().map(|&v| v as f32 as f64).collect();
        let whole = TileStats::of(&Plane {
            width: case.width,
            height: case.height,
            values: values.clone(),
        });
        let half = case.height / 2;
        let part = |rows: std::ops::Range<usize>| {
            TileStats::of(&Plane {
                width: case.width,
                height: rows.len(),
                values: values[rows.start * case.width..rows.end * case.width].to_vec(),
            })
        };
        let merged = part(0..half).merge(part(half..case.height));
        assert!(close(
            merged.focus_score().unwrap(),
            whole.focus_score().unwrap()
        ));
        assert_eq!(merged.percent(merged.n_max), whole.percent(whole.n_max));
        assert_eq!(merged.percent(merged.n_min), whole.percent(whole.n_min));
        assert!(
            (merged.noise().unwrap() - whole.noise().unwrap()).abs() < 0.1 * whole.noise().unwrap()
        );
    }

    #[test]
    fn a_flat_image_has_no_slope() {
        let flat = Plane {
            width: 64,
            height: 64,
            values: vec![0.3; 64 * 64],
        };
        assert_eq!(power_log_log_slope(&flat), None);
        let stats = TileStats::of(&flat);
        // 0.3 isn't exact in binary: zero up to rounding.
        assert!(stats.focus_score().unwrap() < 1e-20);
        assert_eq!(stats.percent(stats.n_max), Some(100.0));
        assert_eq!(stats.noise(), Some(0.0));
    }

    /// The step: its values under its name, the noise in gray values.
    #[test]
    fn measures_a_channel_under_the_steps_name() {
        let case = cases().into_iter().find(|c| c.name == "sharp").unwrap();
        let mut cache = GlobalPipelineCache::default();
        cache.image_meta.nr_of_bits = 8;
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::new(
                    ImageSize {
                        width: case.width,
                        height: case.height,
                    },
                    case.image.iter().map(|&v| v as f32).collect(),
                )
                .unwrap(),
                tile_offset: Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: ImageDimensionality::Planar,
            })),
            2,
            ImageTile {
                offset_x: 0,
                offset_y: 0,
                width: case.width,
                height: case.height,
            },
        );
        let step = MeasureImageQuality {
            id: "qc".into(),
            channel: ImageChannelIdx(2),
        };
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, &mut cache).unwrap();

        let value = |key: &str| {
            let m = cache
                .measurements
                .measurements
                .iter()
                .find(|m| m.info.key == format!("qc/{key}"))
                .unwrap_or_else(|| panic!("{key}"));
            match &m.values {
                MeasurementValues::Image(v) => v[0].1,
                MeasurementValues::Objects(_) => panic!("an image measurement"),
            }
        };
        assert!(close(value("focus_score"), case.focus_score));
        assert!((value("power_log_log_slope") - case.power_log_log_slope.unwrap()).abs() < 1e-9);
        assert!(close(value("percent_maximal"), case.percent_maximal));
        assert!(close(value("noise"), case.noise * 255.0), "gray values");

        // What it reported is what the project offers for its settings
        // (`measurement_outputs` via the settings).
        let mut settings =
            evanalyzer_cfg::settings::pipeline_command_settings::MeasureImageQualitySettings::default();
        settings.id = "qc".into();
        settings.channel = ImageChannelIdx(2);
        let offered = crate::measurement_outputs(
            &evanalyzer_cfg::settings::pipeline_command::PipelineCommand::MeasureImageQuality(
                settings,
            ),
        );
        let reported: Vec<_> = cache
            .measurements
            .measurements
            .iter()
            .map(|m| m.info.clone())
            .collect();
        assert_eq!(reported, offered);

        let missing = MeasureImageQuality {
            id: "qc".into(),
            channel: ImageChannelIdx(5),
        };
        let err = missing
            .execute(&mut ctx, &mut cache)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no channel 5"), "{err}");
    }

    /// Larger than `MAX_SPECTRUM_SIDE`: the middle of it.
    #[test]
    fn the_spectrum_uses_the_middle_of_a_big_image() {
        let side = MAX_SPECTRUM_SIDE + 100;
        let tile = ImageTile {
            offset_x: 0,
            offset_y: 0,
            width: side,
            height: 10,
        };
        let values: Vec<f32> = (0..side * 10).map(|i| (i % side) as f32).collect();
        let container = Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: side,
                    height: 10,
                },
                values,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
            dimensionality: ImageDimensionality::Planar,
        }));
        let window = spectrum_window(&[tile], |_| Ok(container.clone())).unwrap();
        assert_eq!((window.width, window.height), (MAX_SPECTRUM_SIDE, 10));
        assert_eq!(window.values[0], 50.0, "starts 50 pixels in");
    }
}
