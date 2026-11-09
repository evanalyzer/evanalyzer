//! # vesselness
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-08
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Frangi et al. 1998's vesselness: how much each pixel looks like the
//! middle of a line (a fibre, neurite, vessel) of the given widths - bright
//! for lines, dark for the background, blobs and flat areas. A threshold
//! on it segments thin fibres a threshold on the raw image can't.
//!
//! Per scale σ: the Hessian (second derivatives of the image smoothed with a
//! Gaussian of σ, times σ²), its eigenvalues |λ1| ≤ |λ2|; a line has a small
//! λ1 (along it) and a large λ2 (across it, negative for a bright line).
//! V = exp(-R²/2β²) · (1 - exp(-S²/2c²)) with R = λ1/λ2 (0 for a line, 1 for
//! a blob) and S = √(λ1² + λ2²) (the structure's strength, against noise);
//! 0 where λ2 has the wrong sign. The result is the largest V over the
//! scales, 0..1.
//!
//! Self-contained (its own Gaussian derivative passes): the Hessian command
//! stays untouched.

use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors, SizeUnits};
use macros::CommandsMeta;
use rayon::prelude::*;
use std::sync::Arc;

use crate::algos::{ExecutionScope, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use crate::pipeline::pipeline_cache::GlobalPipelineCache;

/// Highlights lines (fibres, neurites, vessels) of the given widths, so a
/// threshold segments them (Frangi vesselness).
///
/// [Vesselness] -> [Threshold] -> [Connected Components] -> [Extract Objects] -> [Skeletonize Objects]
///
/// The result is 0..1: high in the middle of a line, low for the
/// background, blobs and flat areas.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing", display_name = "Vesselness")]
pub struct Vesselness {
    /// The width of the thinnest lines to find.
    #[cmdsmeta(default = 2.0, min = 0.5, max = 1000.0, step = 0.5, summary = true)]
    pub min_width: f32,

    /// The width of the widest lines to find.
    #[cmdsmeta(default = 6.0, min = 0.5, max = 1000.0, step = 0.5, summary = true)]
    pub max_width: f32,

    /// How many widths between the thinnest and the widest are tried
    /// (spaced evenly on a log scale). More: steadier response over the
    /// widths, slower.
    #[cmdsmeta(default = 3, min = 1, max = 10, step = 1)]
    pub width_steps: usize,

    /// Unit the widths are expressed in.
    #[cmdsmeta(default = SizeUnits::Pixels)]
    pub width_unit: SizeUnits,

    /// Bright lines on a dark background (fluorescence) or dark lines on a
    /// bright one (brightfield).
    #[cmdsmeta(default = LinePolarity::Bright)]
    pub polarity: LinePolarity,

    /// How strongly blobs are suppressed (Frangi's β): lower suppresses
    /// more, so only clearly elongated structures stay.
    #[cmdsmeta(default = 0.5, min = 0.05, max = 5.0, step = 0.05, visibility = Advanced)]
    pub blob_suppression: f32,

    /// Structures much weaker than this count as noise (Frangi's c, on the
    /// relative 0..1 scale of the image's second derivatives). 0:
    /// automatic - half the strongest structure in the image (on a whole
    /// slide: in each tile).
    #[cmdsmeta(default = 0.0, min = 0.0, max = 1.0, step = 0.001, visibility = Advanced)]
    pub noise_level: f32,
}

/// Whether [`Vesselness`] looks for bright or dark lines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinePolarity {
    Bright,
    Dark,
}

impl ImageAlgorithm for Vesselness {
    /// Per scale: Gaussian derivatives along the rows (smoothing, first and
    /// second derivative), then down the columns to the Hessian of each
    /// pixel - its vesselness kept where larger than at the scales before.
    /// The result goes into the scratch pad, which becomes the image.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let sigmas = self.sigmas(ctx);
        ctx.prepare_f32_gray_scratch()?;
        let (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) =
            (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad))
        else {
            return Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", ctx.image),
            });
        };
        let (width, height) = (input.width(), input.height());
        vesselness_into(
            input.data.as_slice(),
            width,
            height,
            &sigmas,
            self.blob_suppression,
            self.noise_level,
            self.polarity == LinePolarity::Bright,
            output.data.as_slice_mut(),
        );
        output.tile_offset = input.tile_offset;
        output.plane = input.plane;
        output.dimensionality = input.dimensionality;
        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Vesselness"
    }

    /// Three planes of row derivatives (4 bytes each); measured by
    /// `tests::tile_ram`.
    fn ram_per_tile_pixel(&self) -> f64 {
        12.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&FRANGI_1998]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

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

impl Vesselness {
    /// The scales σ (pixels) for the widths: σ = width / √6 - where the σ²-
    /// normalized response of a bar of that width peaks (a bar of width w
    /// has the spread w / √12; for a Gaussian line of spread s the response
    /// peaks at σ = √2·s). Log-spaced from the thinnest to the widest.
    fn sigmas(&self, ctx: &PipelineContext) -> Vec<f32> {
        let sizes = ctx.pixel_sizes();
        let pixel_size_nm = (sizes.px_size_x + sizes.px_size_y) / 2.0;
        let to_px = |w: f32| match self.width_unit {
            SizeUnits::NanoMeter if pixel_size_nm > 0.0 => w / pixel_size_nm,
            _ => w,
        };
        sigmas_for_widths(
            to_px(self.min_width),
            to_px(self.max_width),
            self.width_steps,
        )
    }
}

/// The scales σ for fibre widths `a`..`b` (pixels), `steps` of them
/// log-spaced: σ = width / √6 (see [`Vesselness`]'s widths).
pub(crate) fn sigmas_for_widths(a: f32, b: f32, steps: usize) -> Vec<f32> {
    let (lo, hi) = (a.min(b).max(0.5), a.max(b).max(0.5));
    let steps = steps.max(1);
    (0..steps)
        .map(|i| {
            let t = if steps == 1 {
                0.0
            } else {
                i as f32 / (steps - 1) as f32
            };
            let width = lo * (hi / lo).powf(t);
            width / 6f32.sqrt()
        })
        .collect()
}

/// The vesselness of `input` (`width` × `height`, 0..1) at the scales
/// `sigmas` into `out` (cleared first): the largest over the scales. With
/// `noise` 0 the noise level is automatic (half the strongest structure of
/// `input`). Also what Trace Filaments follows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vesselness_into(
    input: &[f32],
    width: usize,
    height: usize,
    sigmas: &[f32],
    blob_suppression: f32,
    noise: f32,
    bright: bool,
    out: &mut [f32],
) {
    out.fill(0.0);
    if width == 0 || height == 0 {
        return;
    }
    let params = Params {
        beta: blob_suppression.max(1e-3),
        noise,
        bright,
    };
    let mut rows = RowDerivatives::new(width * height);
    for &sigma in sigmas {
        let kernels = Kernels::new(sigma);
        rows.compute(input, width, &kernels);
        vesselness_at_scale(&rows, width, height, &kernels, sigma, params, out);
    }
}

/// Weaker structure than this (on the 0..1 scale) is rounding noise - a
/// flat image stays 0 instead of its noise being scaled up.
const MIN_STRUCTURE: f32 = 1e-5;

#[derive(Clone, Copy)]
struct Params {
    beta: f32,
    /// Frangi's c; 0: half the largest S of the scale.
    noise: f32,
    bright: bool,
}

/// The sampled Gaussian of σ and its first and second derivative, on
/// -radius ..= radius (radius = ⌈3σ⌉, at least 1).
struct Kernels {
    radius: usize,
    g0: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
}

impl Kernels {
    fn new(sigma: f32) -> Self {
        let radius = (3.0 * sigma).ceil().max(1.0) as usize;
        let s2 = sigma * sigma;
        let xs: Vec<f32> = (-(radius as isize)..=radius as isize)
            .map(|x| x as f32)
            .collect();
        let g: Vec<f32> = xs.iter().map(|x| (-x * x / (2.0 * s2)).exp()).collect();
        let norm: f32 = g.iter().sum();
        let g0: Vec<f32> = g.iter().map(|v| v / norm).collect();
        // The derivatives normalized like the continuous ones despite the
        // cut-off at 3σ: Σ g1·x = -1, Σ g2 = 0 and Σ g2·x² = 2 (so a ramp
        // and a parabola give their exact derivatives, a flat image 0).
        let mut g1: Vec<f32> = xs.iter().zip(&g0).map(|(x, v)| -x / s2 * v).collect();
        let m1: f32 = xs.iter().zip(&g1).map(|(x, g)| g * x).sum();
        g1.iter_mut().for_each(|g| *g /= -m1);
        let mut g2: Vec<f32> = xs
            .iter()
            .zip(&g0)
            .map(|(x, v)| (x * x - s2) / (s2 * s2) * v)
            .collect();
        let mean = g2.iter().sum::<f32>() / g2.len() as f32;
        g2.iter_mut().for_each(|g| *g -= mean);
        let m2: f32 = xs.iter().zip(&g2).map(|(x, g)| g * x * x).sum();
        g2.iter_mut().for_each(|g| *g *= 2.0 / m2);
        Kernels { radius, g0, g1, g2 }
    }
}

/// The image convolved along its rows with the Gaussian (`smooth`), its
/// first (`first`) and second derivative (`second`) - the image's edge
/// repeated outward.
struct RowDerivatives {
    smooth: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
}

impl RowDerivatives {
    fn new(pixels: usize) -> Self {
        RowDerivatives {
            smooth: vec![0.0; pixels],
            first: vec![0.0; pixels],
            second: vec![0.0; pixels],
        }
    }

    fn compute(&mut self, image: &[f32], width: usize, k: &Kernels) {
        let r = k.radius;
        let taps = 2 * r + 1;
        self.smooth
            .par_chunks_mut(width)
            .zip(self.first.par_chunks_mut(width))
            .zip(self.second.par_chunks_mut(width))
            .zip(image.par_chunks(width))
            .for_each(|(((s, f), d), row)| {
                for x in 0..width {
                    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
                    if x >= r && x + r < width {
                        // Inside: no edge to repeat.
                        let window = &row[x - r..x - r + taps];
                        for i in 0..taps {
                            let v = window[i];
                            a += k.g0[i] * v;
                            b += k.g1[i] * v;
                            c += k.g2[i] * v;
                        }
                    } else {
                        for i in 0..taps {
                            let at = (x + i).saturating_sub(r).min(width - 1);
                            let v = row[at];
                            a += k.g0[i] * v;
                            b += k.g1[i] * v;
                            c += k.g2[i] * v;
                        }
                    }
                    (s[x], f[x], d[x]) = (a, b, c);
                }
            });
    }
}

/// One row y of the Hessian · σ² (Ixx, Ixy, Iyy), down the columns of the
/// row derivatives - row by row, so the reads stay in memory order.
fn hessian_row(
    rows: &RowDerivatives,
    width: usize,
    height: usize,
    k: &Kernels,
    y: usize,
    scale: f32,
    [xx, xy, yy]: [&mut [f32]; 3],
) {
    xx.fill(0.0);
    xy.fill(0.0);
    yy.fill(0.0);
    let r = k.radius as isize;
    for (i, dy) in (-r..=r).enumerate() {
        let at = (y as isize + dy).clamp(0, height as isize - 1) as usize * width;
        let (g0, g1, g2) = (k.g0[i], k.g1[i], k.g2[i]);
        let second = &rows.second[at..at + width];
        let first = &rows.first[at..at + width];
        let smooth = &rows.smooth[at..at + width];
        for x in 0..width {
            xx[x] += g0 * second[x];
            xy[x] += g1 * first[x];
            yy[x] += g2 * smooth[x];
        }
    }
    for v in xx.iter_mut().chain(xy.iter_mut()).chain(yy.iter_mut()) {
        *v *= scale;
    }
}

/// The eigenvalues of [[xx, xy], [xy, yy]], |l1| <= |l2|.
fn eigen(xx: f32, xy: f32, yy: f32) -> (f32, f32) {
    let root = ((xx - yy) * (xx - yy) + 4.0 * xy * xy).sqrt();
    let (a, b) = ((xx + yy + root) / 2.0, (xx + yy - root) / 2.0);
    if a.abs() <= b.abs() { (a, b) } else { (b, a) }
}

/// The Hessian of every pixel at σ (down the columns of the row
/// derivatives, times σ²) to its vesselness, kept in `out` where larger.
fn vesselness_at_scale(
    rows: &RowDerivatives,
    width: usize,
    height: usize,
    k: &Kernels,
    sigma: f32,
    params: Params,
    out: &mut [f32],
) {
    let scale = sigma * sigma;
    let buffers = || vec![0.0f32; 3 * width];
    let c = if params.noise > 0.0 {
        params.noise
    } else {
        // Frangi: half the largest structure strength S.
        let max_s = (0..height)
            .into_par_iter()
            .map_init(buffers, |buf, y| {
                let (xx, rest) = buf.split_at_mut(width);
                let (xy, yy) = rest.split_at_mut(width);
                hessian_row(rows, width, height, k, y, scale, [xx, xy, yy]);
                (0..width)
                    .map(|x| {
                        let (l1, l2) = eigen(xx[x], xy[x], yy[x]);
                        (l1 * l1 + l2 * l2).sqrt()
                    })
                    .fold(0.0f32, f32::max)
            })
            .reduce(|| 0.0, f32::max);
        max_s / 2.0
    };
    if c < MIN_STRUCTURE {
        return; // A flat image: no structure at this scale.
    }
    let (two_b2, two_c2) = (2.0 * params.beta * params.beta, 2.0 * c * c);
    out.par_chunks_mut(width)
        .enumerate()
        .for_each_init(buffers, |buf, (y, row)| {
            let (xx, rest) = buf.split_at_mut(width);
            let (xy, yy) = rest.split_at_mut(width);
            hessian_row(rows, width, height, k, y, scale, [xx, xy, yy]);
            for (x, o) in row.iter_mut().enumerate() {
                let (l1, l2) = eigen(xx[x], xy[x], yy[x]);
                let wrong_sign = if params.bright { l2 >= 0.0 } else { l2 <= 0.0 };
                if wrong_sign {
                    continue;
                }
                let ratio = l1 / l2;
                let strength = l1 * l1 + l2 * l2;
                let v = (-ratio * ratio / two_b2).exp() * (1.0 - (-strength / two_c2).exp());
                if v > *o {
                    *o = v;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gray;
    use kornia_image::ImageSize;

    const SIDE: usize = 64;

    fn command() -> Vesselness {
        Vesselness {
            min_width: 2.0,
            max_width: 6.0,
            width_steps: 3,
            width_unit: SizeUnits::Pixels,
            polarity: LinePolarity::Bright,
            blob_suppression: 0.5,
            noise_level: 0.0,
        }
    }

    /// Runs `command` on a `SIDE`² image of `pixel(x, y)`.
    fn run(command: &Vesselness, pixel: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let size = ImageSize {
            width: SIDE,
            height: SIDE,
        };
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        let data: Vec<f32> = (0..SIDE * SIDE)
            .map(|p| pixel(p % SIDE, p / SIDE))
            .collect();
        ctx.get_f32_gray_image_mut()
            .unwrap()
            .as_slice_mut()
            .copy_from_slice(&data);
        command
            .execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
        ctx.get_f32_gray_image().unwrap().as_slice().to_vec()
    }

    /// A vertical line `w` px wide at x = 30.. on 0.1, value `line`.
    fn line(w: usize, line: f32) -> impl Fn(usize, usize) -> f32 {
        move |x, _| if (30..30 + w).contains(&x) { line } else { 0.1 }
    }

    fn at(v: &[f32], x: usize, y: usize) -> f32 {
        v[y * SIDE + x]
    }

    #[test]
    fn a_bright_line_lights_up_the_background_stays_dark() {
        let v = run(&command(), line(3, 0.8));
        assert!(at(&v, 31, 32) > 0.5, "centre {}", at(&v, 31, 32));
        assert!(at(&v, 10, 32) < 0.01, "background {}", at(&v, 10, 32));
        assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn a_flat_image_gives_zero() {
        let v = run(&command(), |_, _| 0.4);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a_dark_line_needs_dark_polarity() {
        let dark = line(3, 0.0);
        let bright = run(&command(), &dark);
        assert_eq!(at(&bright, 31, 32), 0.0);
        let command = Vesselness {
            polarity: LinePolarity::Dark,
            ..command()
        };
        let v = run(&command, &dark);
        assert!(at(&v, 31, 32) > 0.5, "{}", at(&v, 31, 32));
    }

    /// At its centre a blob (both curvatures equal) is suppressed: Frangi's
    /// exp(-R²/2β²) with R = 1 is exp(-2) ≈ 0.14 - far below a line's
    /// centre. (Its rim looks like a curved ridge at the smallest scale and
    /// does answer - Frangi's known behaviour.)
    #[test]
    fn a_blob_centre_answers_much_less_than_a_line_centre() {
        let disk = run(&command(), |x, y| {
            let (dx, dy) = (x as f32 - 32.0, y as f32 - 32.0);
            if dx * dx + dy * dy <= 4.0 { 0.8 } else { 0.1 }
        });
        let lines = run(&command(), line(3, 0.8));
        let (blob, line) = (at(&disk, 32, 32), at(&lines, 31, 32));
        assert!(blob < 0.2 * line, "blob {blob} line {line}");
    }

    #[test]
    fn the_widths_choose_the_lines() {
        // A 12 px wide line: barely found when looking for 1-2 px lines,
        // well when looking for 8-16 px ones (fixed noise level, so the
        // two runs compare).
        let thin = Vesselness {
            min_width: 1.0,
            max_width: 2.0,
            noise_level: 0.05,
            ..command()
        };
        let wide = Vesselness {
            min_width: 8.0,
            max_width: 16.0,
            noise_level: 0.05,
            ..command()
        };
        let wide_line = line(12, 0.8);
        let centre = |v: Vec<f32>| at(&v, 35, 32);
        let (by_thin, by_wide) = (
            centre(run(&thin, &wide_line)),
            centre(run(&wide, &wide_line)),
        );
        assert!(by_wide > 0.5, "{by_wide}");
        assert!(by_thin < 0.2 * by_wide, "thin {by_thin} wide {by_wide}");
    }

    #[test]
    fn widths_in_nanometres_use_the_pixel_size() {
        let size = ImageSize {
            width: 8,
            height: 8,
        };
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.image_meta.pixel_sizes.px_size_x = 200.0;
        ctx.image_meta.pixel_sizes.px_size_y = 200.0;
        let command = Vesselness {
            min_width: 400.0,
            max_width: 1600.0,
            width_unit: SizeUnits::NanoMeter,
            ..command()
        };
        let sigmas = command.sigmas(&ctx);
        let expected = [2.0f32, 4.0, 8.0].map(|w| w / 6f32.sqrt());
        for (s, e) in sigmas.iter().zip(expected) {
            assert!((s - e).abs() < 1e-5, "{sigmas:?}");
        }
    }

    #[test]
    fn the_second_derivative_kernel_finds_a_parabola() {
        // Σ g2(x)·x² = 2 (the second derivative of x²); Σ g1(x)·x = -1 (the
        // kernel is correlated, not convolved: a ramp's slope, negated).
        let k = Kernels::new(2.0);
        let r = k.radius as isize;
        let second: f32 = (-r..=r).zip(&k.g2).map(|(x, g)| g * (x * x) as f32).sum();
        let first: f32 = (-r..=r).zip(&k.g1).map(|(x, g)| g * x as f32).sum();
        assert!((second - 2.0).abs() < 1e-4, "{second}");
        assert!((first + 1.0).abs() < 1e-4, "{first}");
        assert!(k.g2.iter().sum::<f32>().abs() < 1e-6);
    }
}
