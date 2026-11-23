//! # Blur matching
//!
//! How much blurrier one image of a scene is than another - e.g. two
//! channels of a microscope focused a little differently, or imaged at
//! different wavelengths - as the difference of their Gaussian blurs,
//! `Δσ² = σ_target² − σ_reference²` (px²).
//!
//! The target is modelled as `a` times the reference's scene blurred by its
//! own blur plus things of its own (EVs, noise), the reference as the scene
//! blurred by its blur plus noise. In the frequency domain the
//! cross-spectrum `Re(S · conj(T))` holds only what both share - the target's
//! own spots and both noises don't correlate with the reference and average
//! out - and `|T|²` minus the reference's white noise floor is the shared
//! scene's power in the reference. Their ratio, averaged over rings of equal
//! frequency `f`, is `a · exp(−2π² Δσ² f²)`: a straight line in `ln` over
//! `f²` whose slope gives `Δσ²` (a weighted least-squares fit over the rings
//! the scene stands clearly above the noise).
//!
//! The noise floor is the reference's power at the highest frequencies
//! (`|f| ≥ 0.45` cycles per pixel), where no optical signal is left. Both
//! images are mean-free and Hann-windowed before transforming, on their
//! central [`MAX_SIDE`] × [`MAX_SIDE`] px as the shift estimate. Rings are
//! fitted from 0.04 cycles per pixel up while the scene stands above the
//! reference's noise and the target follows the reference (coherence
//! `Re(S·T*) / √(|S|² |T|²)` at least 0.2) - a target that shows little
//! of the reference gives no estimate.
//!
//! **Shift:** the reference is passed unshifted, with the shift it gets
//! and its interpolation: the shift is applied in the frequency domain as
//! [`Translation`](crate::algos::libs::translate::Translation) does it
//! (its phase and the interpolation's damping, `axis_response`), and the
//! noise in each ring counted damped the same way. Measured on an already
//! interpolated reference, the damped noise at the highest frequencies
//! would give too low a floor and read as a sharper reference.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::Interpolation;
use crate::algos::libs::phase_correlation::{Fft2, MAX_SIDE, signed_frequency};
use crate::algos::libs::translate::axis_response;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// The smallest side estimated on.
const MIN_SIDE: usize = 32;
/// Frequency rings between 0 and 0.5 cycles per pixel.
const RINGS: usize = 64;
/// The lowest frequency fitted (cycles per pixel): below, the background
/// removal and the window shape the spectra.
const LOWEST_FREQUENCY: f64 = 0.04;
/// From this frequency on only noise is left (the noise floor).
const NOISE_FREQUENCY: f64 = 0.45;
/// Rings are fitted while the scene's power in the reference is at least
/// this many times the noise floor.
const MIN_SIGNAL_TO_NOISE: f64 = 1.0;
/// Rings are fitted while the target follows the reference at least this
/// much (coherence, see the module docs).
const MIN_COHERENCE: f64 = 0.2;
/// Fewer rings give no estimate.
const MIN_RINGS: usize = 6;
/// The fitted rings must span at least this much `f²`: a scene of only
/// coarse structure (up to about 0.12 cycles per pixel) says nothing about
/// the blur.
const MIN_SPAN_F2: f64 = 0.01;

/// The blur difference of a target against its reference (see the module
/// docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BlurEstimate {
    /// `σ_target² − σ_reference²` (px²): positive when the target is the
    /// blurrier one.
    pub delta_sigma2: f64,
    /// Rings fitted.
    pub rings: usize,
}

impl BlurEstimate {
    /// The extra Gaussian blur (σ, px) for the target and for the reference
    /// that gives both the same blur: one of them is 0.
    pub(crate) fn extra_sigmas(&self) -> (f64, f64) {
        if self.delta_sigma2 >= 0.0 {
            (0.0, self.delta_sigma2.sqrt())
        } else {
            ((-self.delta_sigma2).sqrt(), 0.0)
        }
    }
}

/// How the reference is shifted onto the target before it is compared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReferenceShift {
    pub dy: f64,
    pub dx: f64,
    pub interpolation: Interpolation,
}

/// The blur difference of the `width × height` images `target` and
/// `reference` (background removed, not yet smoothed; the reference not yet
/// shifted - `shift` is how it will be); `None` for an image smaller than
/// 32 px on a side or without enough shared structure above the noise
/// (e.g. a target that shows nothing of the reference).
pub(crate) fn estimate_blur_difference(
    target: &[f32],
    reference: &[f32],
    width: usize,
    height: usize,
    shift: Option<ReferenceShift>,
) -> Option<BlurEstimate> {
    if width < MIN_SIDE || height < MIN_SIDE {
        return None;
    }
    let (w, h) = (width.min(MAX_SIDE), height.min(MAX_SIDE));
    let (x0, y0) = ((width - w) / 2, (height - h) / 2);
    let hann = |i: usize, len: usize| 0.5 - 0.5 * (2.0 * PI * i as f64 / (len - 1) as f64).cos();
    let window_x: Vec<f64> = (0..w).map(|x| hann(x, w)).collect();
    let window_y: Vec<f64> = (0..h).map(|y| hann(y, h)).collect();
    // Mean-free, windowed central crop.
    let prepare = |image: &[f32]| -> Option<Vec<Complex<f32>>> {
        let mut sum = 0.0f64;
        for y in y0..y0 + h {
            for v in &image[y * width + x0..y * width + x0 + w] {
                sum += *v as f64;
            }
        }
        let mean = sum / (w * h) as f64;
        if !mean.is_finite() {
            return None;
        }
        Some(
            (0..h)
                .flat_map(|y| {
                    let row = &image[(y0 + y) * width + x0..(y0 + y) * width + x0 + w];
                    let wy = window_y[y];
                    row.iter().zip(&window_x).map(move |(v, wx)| {
                        Complex::new(((*v as f64 - mean) * wy * wx) as f32, 0.0)
                    })
                })
                .collect(),
        )
    };
    let fft = Fft2::new(w, h);
    let mut s = prepare(target)?;
    fft.transform(&mut s, false);
    let mut t = prepare(reference)?;
    fft.transform(&mut t, false);

    // The shift's response per axis (1 without a shift).
    let response = |n: usize, shift: f64, interpolation: Interpolation| -> Vec<Complex<f64>> {
        (0..n)
            .map(|k| {
                let (re, im) = axis_response(shift, interpolation, signed_frequency(k, n));
                Complex::new(re, im)
            })
            .collect()
    };
    let (response_y, response_x) = match shift {
        Some(shift) if shift.dy != 0.0 || shift.dx != 0.0 => (
            response(h, shift.dy, shift.interpolation),
            response(w, shift.dx, shift.interpolation),
        ),
        _ => (
            vec![Complex::new(1.0, 0.0); h],
            vec![Complex::new(1.0, 0.0); w],
        ),
    };

    // Per ring: Σ Re(S·(H·T)*), Σ |H·T|², Σ |H|², Σ |S|², the count; the
    // noise floor from the unshifted reference's highest frequencies.
    let mut cross = [0f64; RINGS];
    let mut power = [0f64; RINGS];
    let mut damping = [0f64; RINGS];
    let mut target_power = [0f64; RINGS];
    let mut count = [0usize; RINGS];
    let (mut noise, mut noise_count) = (0.0f64, 0usize);
    for ky in 0..h {
        let fy = signed_frequency(ky, h);
        for kx in 0..w {
            let fx = signed_frequency(kx, w);
            let f = (fy * fy + fx * fx).sqrt();
            let i = ky * w + kx;
            let tv = Complex::new(t[i].re as f64, t[i].im as f64);
            if f >= NOISE_FREQUENCY {
                noise += tv.norm_sqr();
                noise_count += 1;
            }
            let ring = (f / 0.5 * RINGS as f64) as usize;
            if ring < RINGS {
                let hv = response_y[ky] * response_x[kx];
                let shifted = hv * tv;
                let sv = Complex::new(s[i].re as f64, s[i].im as f64);
                cross[ring] += (sv * shifted.conj()).re;
                power[ring] += shifted.norm_sqr();
                damping[ring] += hv.norm_sqr();
                target_power[ring] += sv.norm_sqr();
                count[ring] += 1;
            }
        }
    }
    drop((s, t));
    if noise_count == 0 {
        return None;
    }
    let noise = noise / noise_count as f64;

    // The rings from the lowest fitted frequency up while the scene stands
    // above the noise: (f², ln ratio, weight).
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    for ring in 0..RINGS {
        let f = (ring as f64 + 0.5) / RINGS as f64 * 0.5;
        if f < LOWEST_FREQUENCY || count[ring] == 0 {
            continue;
        }
        let n = count[ring] as f64;
        // The reference's noise in the ring, damped as the shift damps it.
        let ring_noise = noise * damping[ring] / n;
        let signal = power[ring] / n - ring_noise;
        let c = cross[ring] / n;
        let coherence = cross[ring] / (power[ring] * target_power[ring]).sqrt().max(1e-300);
        if signal < MIN_SIGNAL_TO_NOISE * ring_noise || c <= 0.0 || coherence < MIN_COHERENCE {
            break;
        }
        // The log ratio's spread falls with the ring's size and its
        // signal-to-noise ratio.
        let weight = n * (signal / (signal + ring_noise)).powi(2);
        points.push((f * f, (c / signal).ln(), weight));
    }
    if points.len() < MIN_RINGS {
        return None;
    }
    let span = points.last()?.0 - points.first()?.0;
    if span < MIN_SPAN_F2 {
        return None;
    }
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, wt) in &points {
        sw += wt;
        sx += wt * x;
        sy += wt * y;
        sxx += wt * x * x;
        sxy += wt * x * y;
    }
    let det = sw * sxx - sx * sx;
    if det.abs() < 1e-300 {
        return None;
    }
    let slope = (sw * sxy - sx * sy) / det;
    let delta_sigma2 = -slope / (2.0 * PI * PI);
    delta_sigma2.is_finite().then_some(BlurEstimate {
        delta_sigma2,
        rings: points.len(),
    })
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::libs::phase_correlation::test_support::{Lcg, blobs, fourier_blur};

    /// Blobs (σ 1.2 px) with photon-like noise, the target `a` times the
    /// scene blurred by `target_sigma`, the reference by `reference_sigma`,
    /// and spots in the target only.
    fn pair(target_sigma: f64, reference_sigma: f64, noise: f32) -> (Vec<f32>, Vec<f32>, usize) {
        let (w, h) = (256usize, 256usize);
        let scene = blobs(w, h, 400, 1.2, 5);
        let spots = blobs(w, h, 30, 1.3, 9);
        let target_scene = fourier_blur(&scene, w, h, target_sigma);
        let reference_scene = fourier_blur(&scene, w, h, reference_sigma);
        let mut rng = Lcg(3);
        let target = target_scene
            .iter()
            .zip(&spots)
            .map(|(v, sp)| 0.4 * v + 0.8 * sp + noise * rng.normal())
            .collect();
        let reference = reference_scene
            .iter()
            .map(|v| v + noise * rng.normal())
            .collect();
        (target, reference, w)
    }

    #[test]
    fn a_blurrier_target_is_found() {
        for (blur, noise) in [(0.8f64, 0.0f32), (1.0, 0.02), (1.5, 0.02)] {
            let (target, reference, w) = pair(blur, 0.0, noise);
            let found = estimate_blur_difference(&target, &reference, w, w, None).unwrap();
            let (t, r) = found.extra_sigmas();
            assert!(
                t == 0.0 && (r - blur).abs() < 0.15,
                "blur {blur}, noise {noise}: {found:?}"
            );
        }
    }

    #[test]
    fn a_sharper_target_is_found() {
        for blur in [0.7f64, 1.2] {
            let (target, reference, w) = pair(0.0, blur, 0.02);
            let found = estimate_blur_difference(&target, &reference, w, w, None).unwrap();
            let (t, r) = found.extra_sigmas();
            assert!(
                r == 0.0 && (t - blur).abs() < 0.15,
                "blur {blur}: {found:?}"
            );
        }
    }

    #[test]
    fn equally_sharp_images_need_no_extra_blur() {
        let (target, reference, w) = pair(0.6, 0.6, 0.02);
        let found = estimate_blur_difference(&target, &reference, w, w, None).unwrap();
        assert!(found.delta_sigma2.abs() < 0.1, "{found:?}");
    }

    #[test]
    fn a_target_without_the_scene_has_no_estimate() {
        let (w, h) = (256usize, 256usize);
        let reference = blobs(w, h, 400, 1.2, 5);
        let mut rng = Lcg(4);
        let target: Vec<f32> = (0..w * h).map(|_| 0.02 * rng.normal()).collect();
        assert!(estimate_blur_difference(&target, &reference, w, h, None).is_none());
    }

    /// The reference passed unshifted with its shift: the same blur as
    /// for an aligned pair, though cubic interpolation damps the noise at
    /// the highest frequencies (which, measured after shifting, would read
    /// as a sharper reference).
    #[test]
    fn a_shifted_reference_is_compared_as_it_will_be_shifted() {
        use crate::algos::libs::phase_correlation::test_support::fourier_shift;
        let (w, h) = (256usize, 256usize);
        let scene = blobs(w, h, 400, 1.2, 5);
        let mut rng = Lcg(8);
        let (dy, dx) = (0.5, -0.5);
        // The reference lies (dy, dx) off the target: shifting it by
        // (dy, dx) puts it on the target.
        let reference: Vec<f32> = fourier_shift(&scene, w, h, -dy, -dx)
            .iter()
            .map(|v| v + 0.05 * rng.normal())
            .collect();
        let target: Vec<f32> = fourier_blur(&scene, w, h, 1.0)
            .iter()
            .map(|v| 0.4 * v + 0.05 * rng.normal())
            .collect();
        let shift = ReferenceShift {
            dy,
            dx,
            interpolation: Interpolation::Cubic,
        };
        let found = estimate_blur_difference(&target, &reference, w, h, Some(shift)).unwrap();
        // Cubic interpolation at half a pixel blurs the reference itself a
        // little: somewhat less than 1 px is left to add.
        let (t, r) = found.extra_sigmas();
        assert!(t == 0.0 && r > 0.6 && r < 1.05, "{found:?}");
    }

    #[test]
    fn too_small_images_have_no_estimate() {
        assert!(estimate_blur_difference(&[0.0; 20 * 40], &[0.0; 20 * 40], 20, 40, None).is_none());
    }
}
