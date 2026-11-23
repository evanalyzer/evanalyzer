//! # Phase correlation
//!
//! The translation between two images of the same scene - e.g. two
//! channels of a microscope, slightly shifted against each other - to a
//! fraction of a pixel (Kuglin & Hines 1975; sub-pixel as in Guizar-Sicairos
//! et al. 2008).
//!
//! Each image is prepared the same way: band-passed (a Gaussian background
//! of σ = 20 px removed, smoothed with σ = 1 px, both in the frequency
//! domain), clipped to its 1st - 99.5th percentile so a few bright beads
//! don't dominate, its mean subtracted and a 2D Hann window applied. The
//! normalized cross-power spectrum `F_t · conj(F_r) / |…|` is weighted
//! with a Gaussian of σ = 1.5 px and transformed back: its peak is then a
//! sampled Gaussian around the shift, and a parabola through the logarithm
//! of the peak and its two neighbours finds the centre per axis (exact for
//! a Gaussian; < 0.02 px on the tests) - instead of an upsampled DFT.
//!
//! **Sign:** the shift is the one to apply to the reference so it lies on
//! the target, in Translate Image's convention (`output(y, x) = input(y −
//! dy, x − dx)`, positive `dy` moves down).
//!
//! **Quality:** the height of the Gaussian peak fitted through the
//! correlation, as a part of the highest possible (the weights' sum - all
//! frequencies in phase): 1 for identical images, near 0 for unrelated
//! ones.
//!
//! Images larger than [`MAX_SIDE`] are estimated on their central
//! `MAX_SIDE × MAX_SIDE` px: the shift is one for the whole image, and the
//! spectra stay bounded (8 bytes per pixel each).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::CitationMetadata;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::sync::Arc;

/// The largest side estimated on (a central crop beyond).
pub(crate) const MAX_SIDE: usize = 1024;
/// The smallest side a shift is estimated on.
const MIN_SIDE: usize = 16;
/// The background removed before correlating (σ, px).
const BACKGROUND_SIGMA: f64 = 20.0;
/// The smoothing before correlating (σ, px).
const SMOOTHING_SIGMA: f64 = 1.0;
/// The width of the correlation peak (σ, px) - the Gaussian weight of the
/// cross-power spectrum.
const PEAK_SIGMA: f64 = 1.5;
/// The percentiles the prepared images are clipped to.
const CLIP: (f64, f64) = (0.01, 0.995);
/// Columns transformed at a time (cache friendly).
const COLUMNS_AT_A_TIME: usize = 16;

/// The shift to apply to the reference (see the module docs) and how
/// clearly the images match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ShiftEstimate {
    pub dy: f64,
    pub dx: f64,
    pub quality: f64,
}

/// The shift between the `width × height` images `target` and
/// `reference`; `None` for an image smaller than 16 px on a side.
pub(crate) fn estimate_shift(
    target: &[f32],
    reference: &[f32],
    width: usize,
    height: usize,
) -> Option<ShiftEstimate> {
    if width < MIN_SIDE || height < MIN_SIDE {
        return None;
    }
    let (w, h) = (width.min(MAX_SIDE), height.min(MAX_SIDE));
    let (x0, y0) = ((width - w) / 2, (height - h) / 2);
    let crop = |image: &[f32]| -> Vec<Complex<f32>> {
        (y0..y0 + h)
            .flat_map(|y| {
                image[y * width + x0..y * width + x0 + w]
                    .iter()
                    .map(|v| Complex::new(*v, 0.0))
            })
            .collect()
    };
    let fft = Fft2::new(w, h);
    let mut spectrum_t = crop(target);
    fft.prepare(&mut spectrum_t);
    let mut cross = crop(reference);
    fft.prepare(&mut cross);

    // Normalized cross-power spectrum, Gaussian-weighted.
    let mut weights = 0.0f64;
    for ky in 0..h {
        let fy = signed_frequency(ky, h);
        for kx in 0..w {
            let fx = signed_frequency(kx, w);
            let weight = gaussian_response(PEAK_SIGMA, fy * fy + fx * fx);
            weights += weight;
            let i = ky * w + kx;
            let c = spectrum_t[i] * cross[i].conj();
            let norm = c.norm();
            cross[i] = if norm > 1e-30 {
                c * (weight as f32 / norm)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }
    drop(spectrum_t);
    fft.transform(&mut cross, true);

    let (mut peak, mut best) = (0usize, f32::NEG_INFINITY);
    for (i, c) in cross.iter().enumerate() {
        if c.re > best {
            best = c.re;
            peak = i;
        }
    }
    let (py, px) = (peak / w, peak % w);
    let at = |y: usize, x: usize| cross[y * w + x].re as f64;
    let (dy, log_y) = refine(at((py + h - 1) % h, px), at(py, px), at((py + 1) % h, px));
    let (dx, log_x) = refine(at(py, (px + w - 1) % w), at(py, px), at(py, (px + 1) % w));
    let amplitude = (at(py, px).max(f64::MIN_POSITIVE).ln() + log_y + log_x).exp();
    Some(ShiftEstimate {
        dy: signed_index(py, h) as f64 + dy,
        dx: signed_index(px, w) as f64 + dx,
        quality: (amplitude / weights).clamp(0.0, 1.0),
    })
}

/// The centre of a peak `centre` between `before` and `after` (one pixel
/// each side), from -0.5 to 0.5, and how much higher (log) the peak's top
/// is than `centre`: a parabola through their logarithms (exact for a
/// Gaussian), through the values themselves where one isn't positive.
fn refine(before: f64, centre: f64, after: f64) -> (f64, f64) {
    if before > 0.0 && centre > 0.0 && after > 0.0 {
        let (l0, l1, l2) = (before.ln(), centre.ln(), after.ln());
        let curvature = l0 - 2.0 * l1 + l2;
        if curvature < 0.0 {
            let offset = (0.5 * (l0 - l2) / curvature).clamp(-0.5, 0.5);
            return (offset, -0.125 * (l0 - l2) * (l0 - l2) / curvature);
        }
        return (0.0, 0.0);
    }
    let curvature = before - 2.0 * centre + after;
    if curvature < 0.0 {
        ((0.5 * (before - after) / curvature).clamp(-0.5, 0.5), 0.0)
    } else {
        (0.0, 0.0)
    }
}

/// `k` of `n` frequencies as cycles per pixel, from -0.5 to 0.5.
pub(crate) fn signed_frequency(k: usize, n: usize) -> f64 {
    signed_index(k, n) as f64 / n as f64
}

/// `k` of `0..n` as an offset from -n/2 to n/2 (wrapped).
fn signed_index(k: usize, n: usize) -> isize {
    if k <= n / 2 {
        k as isize
    } else {
        k as isize - n as isize
    }
}

/// The frequency response of a Gaussian of `sigma` px at `f²` (cycles per
/// pixel, squared).
pub(crate) fn gaussian_response(sigma: f64, f2: f64) -> f64 {
    (-2.0 * PI * PI * sigma * sigma * f2).exp()
}

/// Forward and inverse 2D FFTs of a `w × h` row-major buffer.
pub(crate) struct Fft2 {
    w: usize,
    h: usize,
    row: (Arc<dyn Fft<f32>>, Arc<dyn Fft<f32>>),
    column: (Arc<dyn Fft<f32>>, Arc<dyn Fft<f32>>),
}

impl Fft2 {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Fft2 {
            w,
            h,
            row: (planner.plan_fft_forward(w), planner.plan_fft_inverse(w)),
            column: (planner.plan_fft_forward(h), planner.plan_fft_inverse(h)),
        }
    }

    /// Transforms `data` in place (unnormalized both ways, as rustfft).
    pub(crate) fn transform(&self, data: &mut [Complex<f32>], inverse: bool) {
        let (w, h) = (self.w, self.h);
        let (row, column) = if inverse {
            (&self.row.1, &self.column.1)
        } else {
            (&self.row.0, &self.column.0)
        };
        row.process(data);
        let mut scratch = vec![Complex::new(0.0, 0.0); column.get_inplace_scratch_len()];
        let mut columns = vec![Complex::new(0.0, 0.0); COLUMNS_AT_A_TIME.min(w) * h];
        for first in (0..w).step_by(COLUMNS_AT_A_TIME) {
            let count = COLUMNS_AT_A_TIME.min(w - first);
            let columns = &mut columns[..count * h];
            for y in 0..h {
                for j in 0..count {
                    columns[j * h + y] = data[y * w + first + j];
                }
            }
            column.process_with_scratch(columns, &mut scratch);
            for y in 0..h {
                for j in 0..count {
                    data[y * w + first + j] = columns[j * h + y];
                }
            }
        }
    }

    /// Prepares an image (real parts of `data`) for correlating and
    /// leaves its spectrum in `data` (see the module docs).
    fn prepare(&self, data: &mut [Complex<f32>]) {
        let (w, h) = (self.w, self.h);
        // Band pass: smoothed, the background removed.
        self.transform(data, false);
        let scale = 1.0 / (w * h) as f64;
        for ky in 0..h {
            let fy = signed_frequency(ky, h);
            for kx in 0..w {
                let fx = signed_frequency(kx, w);
                let f2 = fy * fy + fx * fx;
                let band = gaussian_response(SMOOTHING_SIGMA, f2)
                    * (1.0 - gaussian_response(BACKGROUND_SIGMA, f2));
                data[ky * w + kx] *= (band * scale) as f32;
            }
        }
        self.transform(data, true);

        // Clipped to the percentiles, mean 0, Hann window.
        let mut work: Vec<f32> = data.iter().map(|c| c.re).collect();
        let n = work.len();
        let rank = |p: f64| ((p * (n - 1) as f64).round() as usize).min(n - 1);
        let low = *work.select_nth_unstable_by(rank(CLIP.0), f32::total_cmp).1;
        let high = *work.select_nth_unstable_by(rank(CLIP.1), f32::total_cmp).1;
        drop(work);
        let mean = data
            .iter()
            .map(|c| c.re.clamp(low, high) as f64)
            .sum::<f64>()
            / n as f64;
        let hann =
            |i: usize, len: usize| 0.5 - 0.5 * (2.0 * PI * i as f64 / (len - 1) as f64).cos();
        let window_x: Vec<f64> = (0..w).map(|x| hann(x, w)).collect();
        for y in 0..h {
            let window_y = hann(y, h);
            for x in 0..w {
                let c = &mut data[y * w + x];
                let value = (c.re.clamp(low, high) as f64 - mean) * window_y * window_x[x];
                *c = Complex::new(value as f32, 0.0);
            }
        }
        self.transform(data, false);
    }
}

/// Guizar-Sicairos et al. 2008, sub-pixel image registration.
pub(crate) static GUIZAR_SICAIROS_2008: CitationMetadata = CitationMetadata {
    cite_key: "guizar2008efficient",
    title: "Efficient subpixel image registration algorithms",
    authors: &[
        "Manuel Guizar-Sicairos",
        "Samuel T. Thurman",
        "James R. Fienup",
    ],
    year: 2008,
    container: Some("Optics Letters"),
    doi: Some("10.1364/OL.33.000156"),
    url: None,
    pages: Some("156-158"),
};

/// Test images: a shift exact to a fraction of a pixel (in the frequency
/// domain, periodic), shared with Subtract Reference Channel's tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// `image` (`w × h`) shifted by `(dy, dx)` (`output(y, x) = input(y −
    /// dy, x − dx)`, wrapped around): its spectrum times `e^(−2πi f·d)`.
    pub(crate) fn fourier_shift(image: &[f32], w: usize, h: usize, dy: f64, dx: f64) -> Vec<f32> {
        let fft = Fft2::new(w, h);
        let mut data: Vec<Complex<f32>> = image.iter().map(|v| Complex::new(*v, 0.0)).collect();
        fft.transform(&mut data, false);
        for ky in 0..h {
            // The Nyquist frequency of an even side has no sign: its
            // phase would make the result complex - leave it out.
            let nyquist_y = h.is_multiple_of(2) && ky == h / 2;
            let fy = signed_frequency(ky, h);
            for kx in 0..w {
                let nyquist_x = w.is_multiple_of(2) && kx == w / 2;
                let fx = signed_frequency(kx, w);
                let i = ky * w + kx;
                if nyquist_y || nyquist_x {
                    data[i] = Complex::new(0.0, 0.0);
                    continue;
                }
                let phase = -2.0 * PI * (fy * dy + fx * dx);
                data[i] *= Complex::new(phase.cos() as f32, phase.sin() as f32);
            }
        }
        fft.transform(&mut data, true);
        let scale = 1.0 / (w * h) as f32;
        data.iter().map(|c| c.re * scale).collect()
    }

    /// `image` (`w × h`) blurred with a Gaussian of `sigma` px, wrapped
    /// around (periodic, so [`fourier_shift`] shifts it exactly).
    pub(crate) fn fourier_blur(image: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
        let fft = Fft2::new(w, h);
        let mut data: Vec<Complex<f32>> = image.iter().map(|v| Complex::new(*v, 0.0)).collect();
        fft.transform(&mut data, false);
        for ky in 0..h {
            let fy = signed_frequency(ky, h);
            for kx in 0..w {
                let fx = signed_frequency(kx, w);
                data[ky * w + kx] *= gaussian_response(sigma, fy * fy + fx * fx) as f32;
            }
        }
        fft.transform(&mut data, true);
        let scale = 1.0 / (w * h) as f32;
        data.iter().map(|c| c.re * scale).collect()
    }

    /// Smooth random blobs (`count` Gaussians of `sigma` px, heights 1 - 2)
    /// on a `w × h` image, from a fixed `seed`.
    pub(crate) fn blobs(w: usize, h: usize, count: usize, sigma: f32, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut image = vec![0.0f32; w * h];
        for _ in 0..count {
            let (cx, cy) = (rng.next() * w as f32, rng.next() * h as f32);
            let height = 1.0 + rng.next();
            let reach = (4.0 * sigma).ceil() as isize;
            for y in (cy as isize - reach).max(0)..(cy as isize + reach).min(h as isize) {
                for x in (cx as isize - reach).max(0)..(cx as isize + reach).min(w as isize) {
                    let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    image[y as usize * w + x as usize] +=
                        height * (-d2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        image
    }

    /// A small deterministic random generator (no dependency), values in
    /// `[0, 1)`.
    pub(crate) struct Lcg(pub u64);

    impl Lcg {
        /// A standard normal value (Box-Muller).
        pub(crate) fn normal(&mut self) -> f32 {
            let u1 = self.next().max(1e-7);
            let u2 = self.next();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }

        pub(crate) fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn a_sub_pixel_shift_is_found_with_its_sign() {
        let (w, h) = (128, 96);
        let reference = blobs(w, h, 60, 2.0, 7);
        for (dy, dx) in [(0.8, -1.3), (-2.25, 0.4), (0.0, 0.0), (3.6, 4.9)] {
            let target = fourier_shift(&reference, w, h, dy, dx);
            let found = estimate_shift(&target, &reference, w, h).unwrap();
            assert!(
                (found.dy - dy).abs() < 0.05 && (found.dx - dx).abs() < 0.05,
                "({dy}, {dx}): {found:?}"
            );
            assert!(found.quality > 0.5, "({dy}, {dx}): {found:?}");
        }
    }

    #[test]
    fn exposure_and_offset_dont_matter() {
        let (w, h) = (96, 96);
        let reference = blobs(w, h, 40, 1.5, 3);
        let target: Vec<f32> = fourier_shift(&reference, w, h, -0.6, 1.7)
            .iter()
            .map(|v| 0.37 * v + 100.0)
            .collect();
        let found = estimate_shift(&target, &reference, w, h).unwrap();
        assert!(
            (found.dy + 0.6).abs() < 0.05 && (found.dx - 1.7).abs() < 0.05,
            "{found:?}"
        );
    }

    #[test]
    fn unrelated_images_match_poorly() {
        let (w, h) = (96, 96);
        let a = blobs(w, h, 40, 1.5, 1);
        let b = blobs(w, h, 40, 1.5, 2);
        let found = estimate_shift(&a, &b, w, h).unwrap();
        assert!(found.quality < 0.3, "{found:?}");
    }

    #[test]
    fn a_large_image_is_estimated_on_its_centre() {
        let (w, h) = (MAX_SIDE + 200, 64);
        let reference = blobs(w, h, 400, 2.0, 11);
        let target = fourier_shift(&reference, w, h, 1.2, -0.7);
        let found = estimate_shift(&target, &reference, w, h).unwrap();
        assert!(
            (found.dy - 1.2).abs() < 0.05 && (found.dx + 0.7).abs() < 0.05,
            "{found:?}"
        );
    }

    #[test]
    fn too_small_images_have_no_estimate() {
        assert!(estimate_shift(&[0.0; 15 * 20], &[0.0; 15 * 20], 15, 20).is_none());
    }
}
