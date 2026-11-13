//! # scale_space
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Building blocks of the spot filters (Laplacian of Gaussian, Difference of
//! Gaussians, Wavelet Spots): the discrete analogue of the Gaussian kernel
//! (Lindeberg 1998, appendix A.4.1), a separable convolution with mirrored
//! borders, and σ converted to pixels.

use evanalyzer_cfg::core_types::{CitationMetadata, SizeUnits};

/// The scale-space this module implements: the discrete Gaussian and the
/// scale-normalized derivatives of the spot filters.
pub(crate) static LINDEBERG_1998: CitationMetadata = CitationMetadata {
    cite_key: "lindeberg1998feature",
    title: "Feature Detection with Automatic Scale Selection",
    authors: &["Tony Lindeberg"],
    year: 1998,
    container: Some("International Journal of Computer Vision"),
    doi: Some("10.1023/A:1008045108935"),
    url: Some("https://doi.org/10.1023/A:1008045108935"),
    pages: Some("79-116"),
};

/// The difference of Gaussians as a band-pass (appendix B).
pub(crate) static MARR_HILDRETH_1980: CitationMetadata = CitationMetadata {
    cite_key: "marr1980theory",
    title: "Theory of Edge Detection",
    authors: &["David Marr", "Ellen Hildreth"],
    year: 1980,
    container: Some("Proceedings of the Royal Society of London. Series B"),
    doi: Some("10.1098/rspb.1980.0020"),
    url: Some("https://doi.org/10.1098/rspb.1980.0020"),
    pages: Some("187-217"),
};

/// One tap of a symmetric 1-D kernel: its offset (≥ 0, mirrored to the
/// negative side) and weight.
pub(crate) type Tap = (usize, f32);

/// The scale-space parameter `t = σ²` (Lindeberg 1998, eq. 2-3) in pixels²
/// for `sigma` given in `unit`. Without a pixel size, nanometres are taken
/// as pixels.
pub(crate) fn scale_of_sigma(sigma: f32, unit: SizeUnits, pixel_size_nm: f32) -> f64 {
    let sigma_px = match unit {
        SizeUnits::NanoMeter if pixel_size_nm > 0.0 => sigma / pixel_size_nm,
        _ => sigma,
    } as f64;
    sigma_px.max(0.0).powi(2)
}

/// The discrete analogue of the Gaussian kernel of variance `t` (Lindeberg
/// 1998, appendix A.4.1): `T(m; t) = e^(-t) I_m(t)` with the modified Bessel
/// functions `I_m` - unlike the sampled Gaussian exact in the scale-space
/// sense also at the small scales of small spots (σ < 1 px).
///
/// Returned as the taps `m = 0..=R` (the kernel is symmetric), `R =
/// ⌈4·√t⌉ + 1`: the paper doesn't truncate; beyond 4σ less than 10⁻⁴ of the
/// weight is left, and the truncated kernel is scaled back to a sum of 1 so
/// a flat image stays flat.
///
/// The `I_m(t)` come from Miller's backward recurrence `I_(m-1) = I_(m+1) +
/// (2m / t) I_m`, normalized with `e^t = I_0 + 2 Σ_(m≥1) I_m` - which is
/// exactly the statement that the kernel sums to 1, so `e^(-t)` never has
/// to be evaluated (it underflows for large `t`).
pub(crate) fn discrete_gaussian(t: f64) -> Vec<Tap> {
    if t <= 0.0 {
        return vec![(0, 1.0)];
    }
    let radius = (4.0 * t.sqrt()).ceil() as usize + 1;
    // Start well beyond the radius: the recurrence converges to the
    // decaying solution within a few steps of the start.
    let start = radius + 20 + (2.0 * t.sqrt()).ceil() as usize;
    let mut values = vec![0.0f64; start + 2];
    values[start] = 1e-30;
    for m in (1..=start).rev() {
        values[m - 1] = values[m + 1] + (2.0 * m as f64 / t) * values[m];
        if values[m - 1] > 1e250 {
            // Rescale before overflowing; only ratios matter.
            for v in &mut values[m - 1..] {
                *v *= 1e-250;
            }
        }
    }
    let total = values[0] + 2.0 * values[1..=start].iter().sum::<f64>();
    let kept = values[0] + 2.0 * values[1..=radius].iter().sum::<f64>();
    // `kept / total` is the weight within the radius; scaling by `kept`
    // instead of `total` brings the truncated kernel back to a sum of 1.
    debug_assert!(kept / total > 0.9999);
    (0..=radius)
        .map(|m| (m, (values[m] / kept) as f32))
        .collect()
}

/// `index` mirrored into `0..len` at the borders, the border pixel not
/// repeated (`-1 → 1`, `len → len - 2`): the "symmetric mirroring" of
/// Olivo-Marin 2002 (sect. 2), `c(k + N) = c(N - k)` in Starck & Murtagh
/// 1994 (sect. 3). Reaches beyond one image width (wide kernels on a small
/// tile) fold back again.
#[inline]
pub(crate) fn mirror(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let folded = index.rem_euclid(period);
    if folded >= len as isize {
        (period - folded) as usize
    } else {
        folded as usize
    }
}

/// Convolves the `width × height` image `src` with the symmetric kernel
/// `taps` along the rows, then along the columns, into `dst`; `tmp` holds
/// the row pass. Borders are mirrored ([`mirror`]). All three slices have
/// `width * height` values.
pub(crate) fn convolve_separable(
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    width: usize,
    height: usize,
    taps: &[Tap],
) {
    let reach = taps.iter().map(|&(offset, _)| offset).max().unwrap_or(0);
    let (center, side) = taps
        .split_first()
        .map(|(&(_, c), rest)| (c, rest))
        .expect("a kernel has a centre tap");

    // Rows: the interior without index checks, the borders mirrored.
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let out = &mut tmp[y * width..(y + 1) * width];
        for (x, o) in out.iter_mut().enumerate() {
            let mut sum = center * row[x];
            if x >= reach && x + reach < width {
                for &(offset, weight) in side {
                    sum += weight * (row[x - offset] + row[x + offset]);
                }
            } else {
                for &(offset, weight) in side {
                    let left = mirror(x as isize - offset as isize, width);
                    let right = mirror((x + offset) as isize, width);
                    sum += weight * (row[left] + row[right]);
                }
            }
            *o = sum;
        }
    }

    // Columns: whole rows at a time (cache friendly), mirrored row indices.
    for y in 0..height {
        let out = &mut dst[y * width..(y + 1) * width];
        let own = &tmp[y * width..(y + 1) * width];
        for (o, &v) in out.iter_mut().zip(own) {
            *o = center * v;
        }
        for &(offset, weight) in side {
            let up = mirror(y as isize - offset as isize, height);
            let down = mirror((y + offset) as isize, height);
            let (up, down) = (
                &tmp[up * width..(up + 1) * width],
                &tmp[down * width..(down + 1) * width],
            );
            for ((o, &a), &b) in out.iter_mut().zip(up).zip(down) {
                *o += weight * (a + b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full (two-sided) kernel sum.
    fn sum(taps: &[Tap]) -> f64 {
        taps.iter()
            .map(|&(m, w)| if m == 0 { w as f64 } else { 2.0 * w as f64 })
            .sum()
    }

    /// The full kernel's variance Σ m² T(m).
    fn variance(taps: &[Tap]) -> f64 {
        taps.iter()
            .map(|&(m, w)| 2.0 * (m * m) as f64 * w as f64)
            .sum()
    }

    #[test]
    fn the_scale_is_sigma_squared_in_pixels() {
        assert_eq!(scale_of_sigma(2.0, SizeUnits::Pixels, 0.0), 4.0);
        assert_eq!(scale_of_sigma(200.0, SizeUnits::NanoMeter, 100.0), 4.0);
        // Without a pixel size nanometres count as pixels.
        assert_eq!(scale_of_sigma(2.0, SizeUnits::NanoMeter, 0.0), 4.0);
    }

    #[test]
    fn the_discrete_gaussian_sums_to_one_and_has_variance_t() {
        // The discrete analogue keeps the variance t exactly (`Σ m² e^(-t)
        // I_m(t) = t`); the sampled Gaussian doesn't at small t.
        for t in [0.125, 0.5, 1.0, 2.25, 9.0, 100.0] {
            let taps = discrete_gaussian(t);
            assert!((sum(&taps) - 1.0).abs() < 1e-6, "sum at t = {t}");
            assert!(
                (variance(&taps) - t).abs() < 1e-3 * t.max(1.0),
                "variance {} at t = {t}",
                variance(&taps)
            );
        }
    }

    #[test]
    fn the_discrete_gaussian_matches_bessel_values() {
        // e^(-1) I_0(1) = 0.4657596, e^(-1) I_1(1) = 0.2079104,
        // e^(-1) I_2(1) = 0.0499387 (Abramowitz & Stegun table 9.8).
        let taps = discrete_gaussian(1.0);
        assert!((taps[0].1 - 0.465_759_6).abs() < 1e-5);
        assert!((taps[1].1 - 0.207_910_4).abs() < 1e-5);
        assert!((taps[2].1 - 0.049_938_7).abs() < 1e-5);
    }

    #[test]
    fn t_zero_is_the_identity() {
        assert_eq!(discrete_gaussian(0.0), vec![(0, 1.0)]);
    }

    #[test]
    fn mirror_does_not_repeat_the_border() {
        let folded: Vec<usize> = (-3..8).map(|i| mirror(i, 5)).collect();
        assert_eq!(folded, [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
        assert_eq!(mirror(-7, 1), 0);
        // Far beyond the image: folds back and forth.
        assert_eq!(mirror(13, 5), 3);
    }

    /// The 1-D B3 spline `[1, 4, 6, 4, 1] / 16` as a test kernel.
    const B3: [Tap; 3] = [(0, 0.375), (1, 0.25), (2, 0.0625)];

    #[test]
    fn convolution_keeps_a_flat_image_and_spreads_an_impulse() {
        let (w, h) = (9, 7);
        let mut dst = vec![0.0; w * h];
        let mut tmp = vec![0.0; w * h];
        convolve_separable(&vec![2.0; w * h], &mut dst, &mut tmp, w, h, &B3);
        assert!(dst.iter().all(|&v| (v - 2.0).abs() < 1e-6));

        let mut impulse = vec![0.0; w * h];
        impulse[3 * w + 4] = 1.0;
        convolve_separable(&impulse, &mut dst, &mut tmp, w, h, &B3);
        // The 2-D mask (1 4 6 4 1)ᵀ(1 4 6 4 1) / 256 (Starck & Murtagh 1994,
        // sect. 3): centre 36, its neighbour 24, the corner 1.
        assert!((dst[3 * w + 4] - 36.0 / 256.0).abs() < 1e-7);
        assert!((dst[3 * w + 5] - 24.0 / 256.0).abs() < 1e-7);
        assert!((dst[w + 2] - 1.0 / 256.0).abs() < 1e-7);
        assert!((dst.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mirrored_borders_match_an_explicitly_mirrored_image() {
        // A 1-row image: the row pass alone, checked by hand at the border.
        let src = [1.0, 2.0, 4.0, 8.0];
        let mut dst = [0.0; 4];
        let mut tmp = [0.0; 4];
        convolve_separable(&src, &mut dst, &mut tmp, 4, 1, &B3);
        // x = 0 sees (4, 2 | 1 | 2, 4): (1·4 + 4·2 + 6·1 + 4·2 + 1·4) / 16.
        assert!((dst[0] - 30.0 / 16.0).abs() < 1e-6);
    }
}
