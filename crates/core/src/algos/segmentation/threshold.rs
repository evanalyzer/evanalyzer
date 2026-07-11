use evanalyzer_cfg::core_types::{InternalErrors, PixelUnits, SegmentationClass};
use macros::CommandsMeta;

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};

/// The mathematical strategy used to determine the optimal global threshold.
///
/// Most methods analyze the image histogram to find a "cut-off" point that
/// best separates the foreground from the background.
pub enum ThresholdMethod {
    /// No threshold applied; typically used for bypass logic.
    None,
    /// Uses the user-provided `min_threshold` and `max_threshold` values directly.
    Manual,
    /// Li's Minimum Cross Entropy method. Effective for images with varying backgrounds.
    Li,
    /// An iterative version of Kittler and Illingworth's minimum error thresholding.
    MinError,
    /// Zack's algorithm. Geometric method best for skewed histograms with a single clear peak.
    Triangle,
    /// Tsai's method. Preserves the moments of the original image in the binary result.
    Moments,
    /// Huang's fuzzy thresholding. Minimizes the measures of fuzziness.
    Huang,
    /// Assumes a bimodal histogram and finds the average of two peaks.
    Intermodes,
    /// Ridler-Calvard iterative clustering. Similar to Otsu but uses a different error metric.
    IsoData,
    /// Kapur's method. Uses the entropy of the histogram to find the threshold.
    MaxEntropy,
    /// Uses the average intensity of all pixels as the threshold.
    Mean,
    /// Pre-smooths the histogram until there are only two peaks; finds the minimum between them.
    Minimum,
    /// Most common method. Minimizes intra-class variance (maximizes inter-class variance).
    Otsu,
    /// Assumes a fixed percentage of pixels belong to the foreground.
    Percentile,
    /// Based on the Renyi entropy of the histogram; a generalization of MaxEntropy.
    RenyiEntropy,
    /// An extension of Kapur's method using a different coefficient for entropy.
    Shanbhag,
    /// Minimizes a cost function based on the discrepancy between two classes.
    Yen,
}

/// Configuration for a single thresholding operation within a multi-threshold stack.
#[derive(CommandsMeta)]
pub struct ThresholdEntry {
    /// The algorithm to use (Manual or Automatic).
    #[cmdsmeta(default = ThresholdMethod::Manual)]
    pub method: ThresholdMethod,

    /// The lower intensity bound. Used directly in `Manual` mode, or as a
    /// floor for auto-methods.
    #[cmdsmeta(default = 0, min = 0, max = 65535, step = 1, summary = true)]
    pub min_threshold: f32,

    /// The upper intensity bound. Used directly in `Manual` mode, or as a
    /// ceiling for auto-methods.
    #[cmdsmeta(default = 65535, min = 0, max = 65535, step = 1)]
    pub max_threshold: f32,

    /// Unit used for the threshold value.
    ///
    /// bit: 0 - 255/65535
    /// %: 0 - 100.0
    /// rel: 0 - 1.0
    #[cmdsmeta(default = PixelUnits::Bit)]
    pub unit: PixelUnits,

    /// The classification ID assigned to pixels falling within this threshold range.
    pub object_class_id: SegmentationClass,
}

/// A filter that segments an image into discrete classes based on intensity.
///
/// This supports "Multi-Otsu" style behavior by allowing a vector of
/// [`ThresholdSettings`]. Each pixel is evaluated against the settings to
/// determine which `object_class_id` it belongs to.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::{Threshold, ThresholdSettings, ThresholdMethod};
/// let binary = Threshold {
///     thresholds: vec![ThresholdSettings {
///         method: ThresholdMethod::Otsu,
///         min_threshold: 0.0,
///         max_threshold: 1.0,
///         object_class_id: ObjectLabel::Foreground,
///     }]
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment")]
pub struct Threshold {
    /// A list of thresholding layers. Overlapping ranges are resolved
    /// by the order of the vector (last-in priority).
    pub thresholds: Vec<ThresholdEntry>,
}

impl ImageAlgorithm for Threshold {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let nr_of_bits = ctx.image_meta.nr_of_bits;
        let (input_data, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;

        // Build a 256-bin histogram once if any entry needs it, rescaled to the
        // image's *actual observed* min/max (mirroring the C++ reference's
        // `cv::minMaxLoc` + linear rescale in docs/threshold/threshold.hpp).
        //
        // Real images rarely use the full theoretical bit-depth range (e.g. a
        // 12-bit sensor stored in a 16-bit container), so binning directly off
        // the bit-depth-normalized relative values collapses nearly all pixels
        // into a handful of low bins - degenerate input for methods like Li,
        // whose log-mean formula breaks down once the background mean lands
        // exactly on bin 0.
        let needs_hist = self
            .thresholds
            .iter()
            .any(|s| !matches!(s.method, ThresholdMethod::None | ThresholdMethod::Manual));
        let hist_ctx = needs_hist.then(|| {
            let data = input_data.as_slice();
            let mut dmin = f32::MAX;
            let mut dmax = f32::MIN;
            for &v in data {
                dmin = dmin.min(v);
                dmax = dmax.max(v);
            }
            let hist = build_histogram(data, dmin, dmax);
            (hist, dmin, dmax)
        });

        // Pre-resolve each entry's [min, max] range in relative (0.0–1.0) space.
        let normalized: Vec<(f32, f32, u32)> = self
            .thresholds
            .iter()
            .map(|s| {
                let floor = s.unit.to_relative(s.min_threshold, nr_of_bits);
                let ceiling = s.unit.to_relative(s.max_threshold, nr_of_bits);
                let min = match &s.method {
                    ThresholdMethod::None | ThresholdMethod::Manual => floor,
                    method => {
                        let (hist, dmin, dmax) = hist_ctx.as_ref().unwrap();
                        let bin = compute_auto_threshold(method, hist);
                        let relative = if *dmax > *dmin {
                            dmin + (bin as f32 / 255.0) * (dmax - dmin)
                        } else {
                            *dmin
                        };
                        relative.clamp(floor, ceiling)
                    }
                };
                (min, ceiling, s.object_class_id.as_u32())
            })
            .collect();

        let output_slice = segmentation_map.as_slice_mut();
        for (out_pixel, &in_pixel) in output_slice.iter_mut().zip(input_data.as_slice().iter()) {
            let mut assigned_id = SegmentationClass::BACKGROUND.as_u32();
            for &(min, max, class_id) in &normalized {
                let is_in_range = (in_pixel >= min && in_pixel <= max) as u32;
                assigned_id = (is_in_range * class_id) | ((1 - is_in_range) * assigned_id);
            }
            *out_pixel = assigned_id;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Threshold"
    }
}

// ── Histogram ────────────────────────────────────────────────────────────────

/// Builds a 256-bin histogram from f32 pixel data, rescaled so that `min`
/// maps to bin 0 and `max` maps to bin 255. This matches the C++ reference's
/// per-image `cv::minMaxLoc` contrast stretch, giving auto-threshold
/// algorithms full bin resolution regardless of how much of the theoretical
/// bit-depth range the image actually uses.
fn build_histogram(data: &[f32], min: f32, max: f32) -> [f32; 256] {
    let mut hist = [0.0f32; 256];
    let range = max - min;
    for &v in data {
        let normalized = if range > 0.0 {
            ((v - min) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let bin = ((normalized * 255.0) as usize).min(255);
        hist[bin] += 1.0;
    }
    hist
}

/// Dispatches to the correct auto-threshold algorithm, returning a bin index 0–255.
fn compute_auto_threshold(method: &ThresholdMethod, hist: &[f32; 256]) -> usize {
    match method {
        ThresholdMethod::Li => thresh_li(hist),
        ThresholdMethod::MinError => thresh_min_error(hist),
        ThresholdMethod::Triangle => thresh_triangle(hist),
        ThresholdMethod::Moments => thresh_moments(hist),
        ThresholdMethod::Huang => thresh_huang(hist),
        ThresholdMethod::Intermodes => thresh_intermodes(hist),
        ThresholdMethod::IsoData => thresh_isodata(hist),
        ThresholdMethod::MaxEntropy => thresh_max_entropy(hist),
        ThresholdMethod::Mean => thresh_mean(hist),
        ThresholdMethod::Minimum => thresh_minimum(hist),
        ThresholdMethod::Otsu => thresh_otsu(hist),
        ThresholdMethod::Percentile => thresh_percentile(hist),
        ThresholdMethod::RenyiEntropy => thresh_renyi_entropy(hist),
        ThresholdMethod::Shanbhag => thresh_shanbhag(hist),
        ThresholdMethod::Yen => thresh_yen(hist),
        ThresholdMethod::None | ThresholdMethod::Manual => 0,
    }
}

// ── Algorithm implementations ─────────────────────────────────────────────────
// All ported from the ImageJ AutoThresholder Java source via the C++ reference
// in docs/threshold/. Each function accepts a 256-bin f32 histogram and returns
// the optimal threshold bin index (0–255).

fn thresh_otsu(hist: &[f32; 256]) -> usize {
    let n: f64 = hist.iter().map(|&v| v as f64).sum();
    let s: f64 = hist
        .iter()
        .enumerate()
        .map(|(k, &v)| k as f64 * v as f64)
        .sum();
    let mut sk = 0.0f64;
    let mut n1 = hist[0] as f64;
    let mut bcv_max = 0.0f64;
    let mut k_star = 0usize;
    for k in 1..255usize {
        sk += k as f64 * hist[k] as f64;
        n1 += hist[k] as f64;
        let denom = n1 * (n - n1);
        let bcv = if denom != 0.0 {
            let num = (n1 / n) * s - sk;
            num * num / denom
        } else {
            0.0
        };
        if bcv >= bcv_max {
            bcv_max = bcv;
            k_star = k;
        }
    }
    k_star
}

fn thresh_li(hist: &[f32; 256]) -> usize {
    let num_pixels: f64 = hist.iter().map(|&v| v as f64).sum();
    if num_pixels == 0.0 {
        return 0;
    }
    let mean: f64 = (1..256usize)
        .map(|i| i as f64 * hist[i] as f64)
        .sum::<f64>()
        / num_pixels;

    let mut new_thresh = mean;
    loop {
        let old_thresh = new_thresh;
        let threshold = (old_thresh + 0.5).max(0.0) as usize;
        let t = threshold.min(255);

        let num_back: f64 = (0..=t).map(|i| hist[i] as f64).sum();
        let sum_back: f64 = (0..=t).map(|i| i as f64 * hist[i] as f64).sum();
        let mean_back = if num_back == 0.0 {
            0.0
        } else {
            sum_back / num_back
        };

        let num_obj: f64 = ((t + 1)..256).map(|i| hist[i] as f64).sum();
        let sum_obj: f64 = ((t + 1)..256).map(|i| i as f64 * hist[i] as f64).sum();
        let mean_obj = if num_obj == 0.0 {
            0.0
        } else {
            sum_obj / num_obj
        };

        let temp = if mean_back != 0.0 && mean_obj != 0.0 {
            let div = mean_back.ln() - mean_obj.ln();
            if div != 0.0 {
                (mean_back - mean_obj) / div
            } else {
                0.0
            }
        } else {
            0.0
        };
        // C++ simple_round: cast toward zero after ±0.5 shift
        new_thresh = if temp < -2.220446049250313e-16 {
            (temp - 0.5) as i64 as f64
        } else {
            (temp + 0.5) as i64 as f64
        };

        if (new_thresh - old_thresh).abs() <= 0.5 {
            return threshold.min(255);
        }
    }
}

fn thresh_min_error(hist: &[f32; 256]) -> usize {
    // Helper: cumulative count
    let a = |j: usize| -> f64 { (0..=j).map(|i| hist[i] as f64).sum() };
    // Helper: cumulative intensity-weighted count
    let b = |j: usize| -> f64 { (0..=j).map(|i| i as f64 * hist[i] as f64).sum() };
    // Helper: cumulative intensity-squared-weighted count
    let c = |j: usize| -> f64 { (0..=j).map(|i| (i * i) as f64 * hist[i] as f64).sum() };

    // Initial estimate: mean
    let tot: f64 = a(255);
    let sum: f64 = b(255);
    let mut threshold = (sum / tot).floor() as usize;
    threshold = threshold.min(255);

    let mut t_prev = usize::MAX;
    let mut iters = 0usize;
    while threshold != t_prev && iters < 10_000 {
        iters += 1;
        let a_t = a(threshold);
        let a_all = a(255);
        if a_t == 0.0 || a_all == a_t {
            break;
        }
        let mu = b(threshold) / a_t;
        let nu = (b(255) - b(threshold)) / (a_all - a_t);
        let p = a_t / a_all;
        let q = (a_all - a_t) / a_all;
        let sigma2 = c(threshold) / a_t - mu * mu;
        let tau2 = (c(255) - c(threshold)) / (a_all - a_t) - nu * nu;

        if sigma2 <= 0.0 || tau2 <= 0.0 {
            break;
        }
        let w0 = 1.0 / sigma2 - 1.0 / tau2;
        let w1 = mu / sigma2 - nu / tau2;
        let w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + (sigma2 * q * q / (tau2 * p * p)).log10();

        let sqterm = w1 * w1 - w0 * w2;
        if sqterm < 0.0 {
            break;
        }
        t_prev = threshold;
        let temp = (w1 + sqterm.sqrt()) / w0;
        if temp.is_nan() {
            break;
        }
        threshold = (temp.floor() as isize).clamp(0, 255) as usize;
    }
    threshold
}

fn thresh_triangle(hist: &[f32; 256]) -> usize {
    // Work on a mutable copy so we can optionally reverse it.
    let mut h = *hist;

    let mut min_idx = 0usize;
    let mut max_idx = 0usize;
    let mut d_max = 0.0f32;
    for i in 0..256 {
        if h[i] > 0.0 {
            min_idx = i;
            break;
        }
    }
    if min_idx > 0 {
        min_idx -= 1;
    }
    let mut min2 = 0usize;
    for i in (0..256).rev() {
        if h[i] > 0.0 {
            min2 = i;
            break;
        }
    }
    if min2 < 255 {
        min2 += 1;
    }
    for i in 0..256 {
        if h[i] > d_max {
            max_idx = i;
            d_max = h[i];
        }
    }

    let inverted = (max_idx - min_idx) < (min2 - max_idx);
    if inverted {
        h.reverse();
        min_idx = 255 - min2;
        max_idx = 255 - max_idx;
    }

    if min_idx == max_idx {
        return if inverted { 255 - min_idx } else { min_idx };
    }

    let nx = h[max_idx] as f64;
    let ny = (min_idx as f64) - (max_idx as f64);
    let d = (nx * nx + ny * ny).sqrt();
    let nx = nx / d;
    let ny = ny / d;
    let d = nx * min_idx as f64 + ny * h[min_idx] as f64;

    let mut split = min_idx;
    let mut split_dist = 0.0f64;
    for i in (min_idx + 1)..=max_idx {
        let dist = nx * i as f64 + ny * h[i] as f64 - d;
        if dist > split_dist {
            split = i;
            split_dist = dist;
        }
    }
    if split > 0 {
        split -= 1;
    }

    if inverted { 255 - split } else { split }
}

fn thresh_moments(hist: &[f32; 256]) -> usize {
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    if total == 0.0 {
        return 0;
    }
    let mut m1 = 0.0f64;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for i in 0..256 {
        let d = i as f64;
        let h = hist[i] as f64 / total;
        m1 += d * h;
        m2 += d * d * h;
        m3 += d * d * d * h;
    }
    let cd = m2 - m1 * m1;
    if cd == 0.0 {
        return 0;
    }
    let c0 = (-m2 * m2 + m1 * m3) / cd;
    // C++: c1 = (m0 * -m3 + m2 * m1) / cd  where m0 = 1
    let c1 = (-m3 + m2 * m1) / cd;
    let disc = c1 * c1 - 4.0 * c0;
    if disc < 0.0 {
        return 0;
    }
    let z1 = 0.5 * (-c1 + disc.sqrt());
    let z0 = 0.5 * (-c1 - disc.sqrt());
    let p0 = (z1 - m1) / (z1 - z0);

    let mut cumsum = 0.0f64;
    for i in 0..256 {
        cumsum += hist[i] as f64 / total;
        if cumsum > p0 {
            return i;
        }
    }
    255
}

fn thresh_huang(hist: &[f32; 256]) -> usize {
    let first_bin = (0..256).find(|&i| hist[i] != 0.0).unwrap_or(0);
    let last_bin = (0..256).rev().find(|&i| hist[i] != 0.0).unwrap_or(255);
    if first_bin == last_bin {
        return first_bin;
    }
    let term = 1.0 / (last_bin - first_bin) as f64;

    // mu_0[i] = mean of pixels 0..=i (running forward)
    let mut mu_0 = [0.0f64; 256];
    let mut sum_pix = 0.0f64;
    let mut num_pix = 0.0f64;
    for i in first_bin..256 {
        sum_pix += i as f64 * hist[i] as f64;
        num_pix += hist[i] as f64;
        mu_0[i] = sum_pix / num_pix;
    }

    // mu_1[i] = mean of pixels i+1..=255 (running backward, stored at i-1)
    let mut mu_1 = [0.0f64; 256];
    sum_pix = 0.0;
    num_pix = 0.0;
    for i in (1..=last_bin).rev() {
        sum_pix += i as f64 * hist[i] as f64;
        num_pix += hist[i] as f64;
        mu_1[i - 1] = sum_pix / num_pix;
    }

    let mut min_ent = f64::MAX;
    let mut threshold = 0usize;
    for it in 0..256 {
        let mut ent = 0.0f64;
        for ih in 0..=it {
            let mu_x = 1.0 / (1.0 + term * (ih as f64 - mu_0[it]).abs());
            if mu_x >= 1e-6 && mu_x <= 0.999_999 {
                ent += hist[ih] as f64 * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        for ih in (it + 1)..256 {
            let mu_x = 1.0 / (1.0 + term * (ih as f64 - mu_1[it]).abs());
            if mu_x >= 1e-6 && mu_x <= 0.999_999 {
                ent += hist[ih] as f64 * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        if ent < min_ent {
            min_ent = ent;
            threshold = it;
        }
    }
    threshold
}

fn bimodal_test(y: &[f64]) -> bool {
    let mut modes = 0u32;
    for k in 1..y.len().saturating_sub(1) {
        if y[k - 1] < y[k] && y[k + 1] < y[k] {
            modes += 1;
            if modes > 2 {
                return false;
            }
        }
    }
    modes == 2
}

fn thresh_intermodes(hist: &[f32; 256]) -> usize {
    let min_bin = (0..256).find(|&i| hist[i] > 0.0).unwrap_or(0);
    let max_bin = (0..256).rev().find(|&i| hist[i] > 0.0).unwrap_or(255);
    let length = max_bin - min_bin + 1;

    let mut h: Vec<f64> = (min_bin..=max_bin).map(|i| hist[i] as f64).collect();

    let mut iters = 0u32;
    while !bimodal_test(&h) {
        // 3-point running mean over the *previous pass's* values, with
        // implicit zero-padding at both ends - matches the C++ reference's
        // previous/current/next rolling window. Must read entirely from `h`
        // and write into a separate buffer: writing into `h` in place would
        // make later iterations of this same pass read already-smoothed
        // neighbors instead of the pre-pass values.
        let mut smoothed = vec![0.0f64; length];
        for i in 0..length {
            let prev = if i == 0 { 0.0 } else { h[i - 1] };
            let next = if i + 1 < length { h[i + 1] } else { 0.0 };
            smoothed[i] = (prev + h[i] + next) / 3.0;
        }
        h = smoothed;
        iters += 1;
        if iters > 10_000 {
            return 0;
        }
    }

    // Threshold = midpoint between the two peaks
    let mut tt = 0usize;
    for i in 1..length.saturating_sub(1) {
        if h[i - 1] < h[i] && h[i + 1] < h[i] {
            tt += i;
        }
    }
    let t = (tt as f64 / 2.0).floor() as usize;
    (t + min_bin).min(255)
}

fn thresh_isodata(hist: &[f32; 256]) -> usize {
    let mut g = 0usize;
    for i in 1..256 {
        if hist[i] > 0.0 {
            g = i + 1;
            break;
        }
    }
    loop {
        let mut l = 0i64;
        let mut totl = 0i64;
        for i in 0..g {
            totl += hist[i] as i64;
            l += hist[i] as i64 * i as i64;
        }
        let mut h = 0.0f64;
        let mut toth = 0.0f64;
        for i in (g + 1)..256 {
            toth += hist[i] as f64;
            h += hist[i] as f64 * i as f64;
        }
        if totl > 0 && toth > 0.0 {
            let l_mean = l as f64 / totl as f64;
            let h_mean = h / toth;
            if g == ((l_mean + h_mean) / 2.0).round() as usize {
                break;
            }
        }
        g += 1;
        if g > 254 {
            return 0;
        }
    }
    g
}

fn thresh_max_entropy(hist: &[f32; 256]) -> usize {
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    let norm: Vec<f64> = hist.iter().map(|&v| v as f64 / total).collect();

    let mut p1 = [0.0f64; 256];
    let mut p2 = [0.0f64; 256];
    p1[0] = norm[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..256 {
        p1[i] = p1[i - 1] + norm[i];
        p2[i] = 1.0 - p1[i];
    }

    let first_bin = (0..256)
        .find(|&i| p1[i].abs() >= 2.220446049250313e-16)
        .unwrap_or(0);
    let last_bin = (first_bin..256)
        .rev()
        .find(|&i| p2[i].abs() >= 2.220446049250313e-16)
        .unwrap_or(255);

    let mut max_ent = f64::MIN;
    let mut threshold = 0usize;
    for it in first_bin..=last_bin {
        let ent_back: f64 = (0..=it)
            .filter(|&ih| hist[ih] != 0.0)
            .map(|ih| -(norm[ih] / p1[it]) * (norm[ih] / p1[it]).ln())
            .sum();
        let ent_obj: f64 = ((it + 1)..256)
            .filter(|&ih| hist[ih] != 0.0)
            .map(|ih| -(norm[ih] / p2[it]) * (norm[ih] / p2[it]).ln())
            .sum();
        let tot_ent = ent_back + ent_obj;
        if tot_ent > max_ent {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    threshold
}

fn thresh_mean(hist: &[f32; 256]) -> usize {
    let tot: f64 = hist.iter().map(|&v| v as f64).sum();
    if tot == 0.0 {
        return 0;
    }
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &v)| i as f64 * v as f64)
        .sum();
    (sum / tot).floor() as usize
}

fn thresh_minimum(hist: &[f32; 256]) -> usize {
    let mut h: Vec<f64> = hist.iter().map(|&v| v as f64).collect();
    let mut iters = 0u32;
    while !bimodal_test(&h) {
        let mut t = [0.0f64; 256];
        t[0] = (h[0] + h[1]) / 3.0;
        for i in 1..255 {
            t[i] = (h[i - 1] + h[i] + h[i + 1]) / 3.0;
        }
        t[255] = (h[254] + h[255]) / 3.0;
        h.copy_from_slice(&t);
        iters += 1;
        if iters > 10_000 {
            return 0;
        }
    }
    // First local minimum
    for i in 1..255 {
        if h[i - 1] > h[i] && h[i + 1] >= h[i] {
            return i;
        }
    }
    0
}

fn thresh_percentile(hist: &[f32; 256]) -> usize {
    let ptile = 0.5f64;
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    if total == 0.0 {
        return 0;
    }
    let mut best = 0usize;
    let mut min_dist = f64::MAX;
    let mut cumsum = 0.0f64;
    for i in 0..256 {
        cumsum += hist[i] as f64;
        let d = (cumsum / total - ptile).abs();
        if d < min_dist {
            min_dist = d;
            best = i;
        }
    }
    best
}

fn thresh_renyi_entropy(hist: &[f32; 256]) -> usize {
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    let norm: Vec<f64> = hist.iter().map(|&v| v as f64 / total).collect();

    let mut p1 = [0.0f64; 256];
    let mut p2 = [0.0f64; 256];
    p1[0] = norm[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..256 {
        p1[i] = p1[i - 1] + norm[i];
        p2[i] = 1.0 - p1[i];
    }

    let eps = 2.220446049250313e-16;
    let first_bin = (0..256).find(|&i| p1[i].abs() >= eps).unwrap_or(0);
    let last_bin = (first_bin..256)
        .rev()
        .find(|&i| p2[i].abs() >= eps)
        .unwrap_or(255);

    // Pass 1: alpha = 1.0  (equivalent to MaxEntropy)
    let mut max_ent = 0.0f64;
    let mut t_star2 = 0usize;
    for it in first_bin..=last_bin {
        let eb: f64 = (0..=it)
            .filter(|&ih| hist[ih] != 0.0)
            .map(|ih| -(norm[ih] / p1[it]) * (norm[ih] / p1[it]).ln())
            .sum();
        let eo: f64 = ((it + 1)..256)
            .filter(|&ih| hist[ih] != 0.0)
            .map(|ih| -(norm[ih] / p2[it]) * (norm[ih] / p2[it]).ln())
            .sum();
        if eb + eo > max_ent {
            max_ent = eb + eo;
            t_star2 = it;
        }
    }

    // Pass 2: alpha = 0.5
    let alpha = 0.5f64;
    let term = 1.0 / (1.0 - alpha);
    max_ent = 0.0;
    let mut t_star1 = 0usize;
    for it in first_bin..=last_bin {
        let eb: f64 = (0..=it).map(|ih| (norm[ih] / p1[it]).sqrt()).sum();
        let eo: f64 = ((it + 1)..256).map(|ih| (norm[ih] / p2[it]).sqrt()).sum();
        let tot = if eb * eo > 0.0 {
            term * (eb * eo).ln()
        } else {
            0.0
        };
        if tot > max_ent {
            max_ent = tot;
            t_star1 = it;
        }
    }

    // Pass 3: alpha = 2.0
    let alpha = 2.0f64;
    let term = 1.0 / (1.0 - alpha);
    max_ent = 0.0;
    let mut t_star3 = 0usize;
    for it in first_bin..=last_bin {
        let eb: f64 = (0..=it)
            .map(|ih| norm[ih] * norm[ih] / (p1[it] * p1[it]))
            .sum();
        let eo: f64 = ((it + 1)..256)
            .map(|ih| norm[ih] * norm[ih] / (p2[it] * p2[it]))
            .sum();
        let tot = if eb * eo > 0.0 {
            term * (eb * eo).ln()
        } else {
            0.0
        };
        if tot > max_ent {
            max_ent = tot;
            t_star3 = it;
        }
    }

    // Sort t_star1 ≤ t_star2 ≤ t_star3
    let mut stars = [t_star1, t_star2, t_star3];
    stars.sort_unstable();
    let [t_star1, t_star2, t_star3] = stars;

    let (beta1, beta2, beta3) = if t_star2.abs_diff(t_star1) <= 5 {
        if t_star3.abs_diff(t_star2) <= 5 {
            (1, 2, 1)
        } else {
            (0, 1, 3)
        }
    } else if t_star3.abs_diff(t_star2) <= 5 {
        (3, 1, 0)
    } else {
        (1, 2, 1)
    };

    let omega = p1[t_star3] - p1[t_star1];
    let opt = t_star1 as f64 * (p1[t_star1] + 0.25 * omega * beta1 as f64)
        + 0.25 * t_star2 as f64 * omega * beta2 as f64
        + t_star3 as f64 * (p2[t_star3] + 0.25 * omega * beta3 as f64);
    (opt as usize).min(255)
}

fn thresh_shanbhag(hist: &[f32; 256]) -> usize {
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    let norm: Vec<f64> = hist.iter().map(|&v| v as f64 / total).collect();

    let mut p1 = [0.0f64; 256];
    let mut p2 = [0.0f64; 256];
    p1[0] = norm[0];
    p2[0] = 1.0 - p1[0];
    for i in 1..256 {
        p1[i] = p1[i - 1] + norm[i];
        p2[i] = 1.0 - p1[i];
    }

    let eps = 2.220446049250313e-16;
    let first_bin = (0..256).find(|&i| p1[i].abs() >= eps).unwrap_or(0);
    let last_bin = (first_bin..256)
        .rev()
        .find(|&i| p2[i].abs() >= eps)
        .unwrap_or(255);

    let mut min_ent = f64::MAX;
    let mut threshold = 0usize;
    for it in first_bin..=last_bin {
        let term_back = 0.5 / p1[it];
        let ent_back: f64 = (1..=it)
            .map(|ih| -norm[ih] * (1.0 - term_back * p1[ih - 1]).ln())
            .sum::<f64>()
            * term_back;

        let term_obj = 0.5 / p2[it];
        let ent_obj: f64 = ((it + 1)..256)
            .map(|ih| -norm[ih] * (1.0 - term_obj * p2[ih]).ln())
            .sum::<f64>()
            * term_obj;

        let tot = (ent_back - ent_obj).abs();
        if tot < min_ent {
            min_ent = tot;
            threshold = it;
        }
    }
    threshold
}

fn thresh_yen(hist: &[f32; 256]) -> usize {
    let total: f64 = hist.iter().map(|&v| v as f64).sum();
    let norm: Vec<f64> = hist.iter().map(|&v| v as f64 / total).collect();

    let mut p1 = [0.0f64; 256];
    p1[0] = norm[0];
    for i in 1..256 {
        p1[i] = p1[i - 1] + norm[i];
    }

    let mut p1_sq = [0.0f64; 256];
    p1_sq[0] = norm[0] * norm[0];
    for i in 1..256 {
        p1_sq[i] = p1_sq[i - 1] + norm[i] * norm[i];
    }

    let mut p2_sq = [0.0f64; 256];
    p2_sq[255] = 0.0;
    for i in (0..255).rev() {
        p2_sq[i] = p2_sq[i + 1] + norm[i + 1] * norm[i + 1];
    }

    let mut max_crit = f64::MIN;
    let mut threshold = 0usize;
    for it in 0..256 {
        let log_p1_sq = if p1_sq[it] * p2_sq[it] > 0.0 {
            (p1_sq[it] * p2_sq[it]).ln()
        } else {
            0.0
        };
        let log_p1 = if p1[it] * (1.0 - p1[it]) > 0.0 {
            (p1[it] * (1.0 - p1[it])).ln()
        } else {
            0.0
        };
        let crit = -log_p1_sq + 2.0 * log_p1;
        if crit > max_crit {
            max_crit = crit;
            threshold = it;
        }
    }
    threshold
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ImageDebugExt};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_multi_range_thresholding() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 3,
            height: 2,
        };
        let input_data = vec![0.1, 0.5, 0.9, 0.0, 1.0, 0.4];
        let input_img = Image::<f32, 1, CpuAllocator>::new(size, input_data, CpuAllocator)?;
        input_img.print_window();

        let settings = vec![
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.0,
                max_threshold: 0.2,
                object_class_id: SegmentationClass(10),
                unit: PixelUnits::Relative,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.3,
                max_threshold: 0.6,
                object_class_id: SegmentationClass(20),
                unit: PixelUnits::Relative,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.8,
                max_threshold: 1.0,
                object_class_id: SegmentationClass(30),
                unit: PixelUnits::Relative,
            },
        ];

        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = PipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;
        ctx.get_segmentation_map()?.print_window();

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();

        let expected = vec![10, 20, 30, 10, 30, 20];
        assert_eq!(
            result_pixels,
            &expected[..],
            "Pixel classification failed to match expected IDs"
        );
        Ok(())
    }

    #[test]
    fn test_otsu_uniform_histogram() {
        let mut hist = [0.0f32; 256];
        for v in hist.iter_mut() {
            *v = 1.0;
        }
        let t = thresh_otsu(&hist);
        // With a perfectly flat histogram, any bin is equally valid.
        assert!(t < 256);
    }

    #[test]
    fn test_mean_threshold() {
        let mut hist = [0.0f32; 256];
        hist[0] = 100.0;
        hist[255] = 100.0;
        let t = thresh_mean(&hist);
        assert_eq!(t, 127); // mean of 0 and 255
    }

    #[test]
    fn test_auto_threshold_smoke() {
        // All auto methods should return a valid bin (0-255) on a bimodal histogram.
        let mut hist = [0.0f32; 256];
        for i in 0..50 {
            hist[i] = 80.0;
        }
        for i in 200..256 {
            hist[i] = 80.0;
        }

        for (name, bin) in [
            ("otsu", thresh_otsu(&hist)),
            ("li", thresh_li(&hist)),
            ("min_error", thresh_min_error(&hist)),
            ("triangle", thresh_triangle(&hist)),
            ("moments", thresh_moments(&hist)),
            ("huang", thresh_huang(&hist)),
            ("isodata", thresh_isodata(&hist)),
            ("max_entropy", thresh_max_entropy(&hist)),
            ("mean", thresh_mean(&hist)),
            ("minimum", thresh_minimum(&hist)),
            ("percentile", thresh_percentile(&hist)),
            ("renyi", thresh_renyi_entropy(&hist)),
            ("shanbhag", thresh_shanbhag(&hist)),
            ("yen", thresh_yen(&hist)),
        ] {
            assert!(bin < 256, "{name}: bin {bin} out of range");
        }
    }

    #[test]
    fn test_build_histogram_rescales_to_observed_range() {
        // Narrow-range data (as from a 16-bit image that doesn't use its full
        // theoretical range) must still spread across the full 0-255 bins
        // once rescaled to its own observed min/max, instead of collapsing
        // into bin 0 the way a bit-depth-only normalization would.
        let data: Vec<f32> = (0..100).map(|i| 0.001 + i as f32 * 0.0001).collect();
        let dmin = data.iter().cloned().fold(f32::MAX, f32::min);
        let dmax = data.iter().cloned().fold(f32::MIN, f32::max);
        let hist = build_histogram(&data, dmin, dmax);
        assert!(hist[0] > 0.0, "min value should land in bin 0");
        assert!(hist[255] > 0.0, "max value should land in bin 255");
    }

    #[test]
    fn test_build_histogram_uniform_image_no_panic() {
        // Degenerate case (min == max) must not divide by zero.
        let data = vec![0.5f32; 10];
        let hist = build_histogram(&data, 0.5, 0.5);
        assert_eq!(hist[0], 10.0);
    }

    #[test]
    fn test_li_narrow_dynamic_range_regression() -> Result<(), Box<dyn std::error::Error>> {
        // Regression test for a real-world bug: 16-bit images rarely use
        // their full theoretical range (e.g. a 12-bit sensor stored in a
        // 16-bit container). Before the histogram was rescaled to the
        // image's *observed* min/max, such narrow-range data collapsed into
        // a couple of low bins, which broke Li's log-mean formula (the
        // background mean landed exactly on bin 0) and made it classify
        // almost every pixel as foreground - "it detects everything".
        let width = 100;
        let height = 100;
        let size = ImageSize { width, height };

        // Background cluster: relative ~80/65535, 80% of pixels.
        // Foreground cluster: relative ~2500/65535, 20% of pixels.
        // Both are well inside the theoretical [0, 1] range for a 16-bit image.
        let mut seed = 12345u64;
        let mut next_f32 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 1_000_000) as f32 / 1_000_000.0
        };
        let mut input_data = vec![0.0f32; width * height];
        for (i, px) in input_data.iter_mut().enumerate() {
            *px = if i % 5 == 0 {
                (2500.0 + (next_f32() - 0.5) * 600.0) / 65535.0
            } else {
                (80.0 + (next_f32() - 0.5) * 30.0) / 65535.0
            };
        }

        let input_img = Image::<f32, 1, CpuAllocator>::new(size, input_data, CpuAllocator)?;
        let settings = vec![ThresholdEntry {
            method: ThresholdMethod::Li,
            min_threshold: 0.0,
            max_threshold: 1.0,
            object_class_id: SegmentationClass(1),
            unit: PixelUnits::Relative,
        }];

        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = PipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();
        let foreground_fraction = result_pixels.iter().filter(|&&v| v == 1).count() as f32
            / result_pixels.len() as f32;

        // True foreground fraction is ~20%. Before the fix this was ~100%
        // ("detects everything"); it should now land close to the real split.
        assert!(
            (0.05..0.4).contains(&foreground_fraction),
            "Li threshold classified {foreground_fraction:.3} of pixels as foreground, expected ~0.2"
        );
        Ok(())
    }

    #[test]
    fn test_intermodes_multipass_smoothing_matches_reference() {
        // Regression test for a real port bug: the smoothing pass used to
        // write `h[i]` while still reading `h[i-1]` from the *same* pass,
        // so later indices picked up already-smoothed neighbors instead of
        // the pre-pass values (Gauss-Seidel instead of the reference's
        // Jacobi update). This only shows up once more than one smoothing
        // iteration is needed to reach a bimodal histogram - a noisy,
        // multi-peak histogram like this one.
        //
        // Expected value cross-checked against an independent, literal
        // translation of the C++ previous/current/next rolling-window
        // smoothing in docs/threshold/threshold_intermodes.hpp. The old
        // buggy implementation returned 63 here instead of 64.
        let raw: [f32; 25] = [
            2.0, 5.0, 9.0, 4.0, 7.0, 12.0, 6.0, 3.0, 8.0, 15.0, 20.0, 14.0, 9.0, 5.0, 3.0, 6.0,
            10.0, 18.0, 25.0, 22.0, 16.0, 10.0, 6.0, 3.0, 1.0,
        ];
        let mut hist = [0.0f32; 256];
        let offset = 50;
        hist[offset..offset + raw.len()].copy_from_slice(&raw);

        assert_eq!(thresh_intermodes(&hist), offset + 14);
    }

    #[test]
    fn test_all_auto_methods_narrow_dynamic_range_regression() -> Result<(), Box<dyn std::error::Error>>
    {
        // Same real-world scenario as test_li_narrow_dynamic_range_regression
        // (16-bit image that only uses a narrow slice of its theoretical
        // range), exercised across every auto-threshold method through the
        // full pipeline. Confirms the histogram now gets rescaled to the
        // image's observed min/max for all methods, not just Li - before
        // that fix, every method binned almost entirely into bin 0.
        let width = 100;
        let height = 100;
        let size = ImageSize { width, height };

        let methods = [
            ("Otsu", ThresholdMethod::Otsu),
            ("MinError", ThresholdMethod::MinError),
            ("Triangle", ThresholdMethod::Triangle),
            ("Moments", ThresholdMethod::Moments),
            ("Huang", ThresholdMethod::Huang),
            ("Intermodes", ThresholdMethod::Intermodes),
            ("IsoData", ThresholdMethod::IsoData),
            ("MaxEntropy", ThresholdMethod::MaxEntropy),
            ("Mean", ThresholdMethod::Mean),
            ("Minimum", ThresholdMethod::Minimum),
            ("Percentile", ThresholdMethod::Percentile),
            ("RenyiEntropy", ThresholdMethod::RenyiEntropy),
            ("Shanbhag", ThresholdMethod::Shanbhag),
            ("Yen", ThresholdMethod::Yen),
        ];

        for (name, method) in methods {
            let mut seed = 42u64;
            let mut next_f32 = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 1_000_000) as f32 / 1_000_000.0
            };
            let mut input_data = vec![0.0f32; width * height];
            for (i, px) in input_data.iter_mut().enumerate() {
                *px = if i % 5 == 0 {
                    (2500.0 + (next_f32() - 0.5) * 600.0) / 65535.0
                } else {
                    (80.0 + (next_f32() - 0.5) * 30.0) / 65535.0
                };
            }

            let input_img = Image::<f32, 1, CpuAllocator>::new(size, input_data, CpuAllocator)?;
            let settings = vec![ThresholdEntry {
                method,
                min_threshold: 0.0,
                max_threshold: 1.0,
                object_class_id: SegmentationClass(1),
                unit: PixelUnits::Relative,
            }];
            let cmd = Threshold {
                thresholds: settings,
            };
            let mut ctx = PipelineContext::new_from_image_test(input_img)?;
            let mut cache = PipelineCache::default();
            cmd.execute(&mut ctx, &mut cache)?;

            let result_pixels = ctx
                .segmentation_map
                .as_ref()
                .expect("No labels found")
                .as_slice();
            let foreground_fraction = result_pixels.iter().filter(|&&v| v == 1).count() as f32
                / result_pixels.len() as f32;

            assert!(
                (0.0..0.9).contains(&foreground_fraction),
                "{name}: classified {foreground_fraction:.3} of pixels as foreground on a narrow \
                 16-bit dynamic range - looks like the histogram collapsed into bin 0 again"
            );
        }
        Ok(())
    }

    /// Shared bimodal histogram used by the per-method reference tests below:
    /// a dense, narrow cluster (bins 20-40, count 50 each) and a sparser,
    /// wider cluster (bins 160-200, count 20 each), separated by an empty
    /// valley. Expected values were computed with an independent, literal
    /// Rust translation of each docs/threshold/*.hpp C++ reference (kept
    /// out-of-tree, not sharing code with this module), so a match here
    /// means the port is faithful, not just "returns something in range".
    fn bimodal_reference_histogram() -> [f32; 256] {
        let mut h = [0.0f32; 256];
        for i in 20..=40 {
            h[i] = 50.0;
        }
        for i in 160..=200 {
            h[i] = 20.0;
        }
        h
    }

    #[test]
    fn test_otsu_matches_reference() {
        assert_eq!(thresh_otsu(&bimodal_reference_histogram()), 159);
    }

    #[test]
    fn test_li_matches_reference() {
        assert_eq!(thresh_li(&bimodal_reference_histogram()), 84);
    }

    #[test]
    fn test_min_error_matches_reference() {
        assert_eq!(thresh_min_error(&bimodal_reference_histogram()), 80);
    }

    #[test]
    fn test_triangle_matches_reference() {
        assert_eq!(thresh_triangle(&bimodal_reference_histogram()), 42);
    }

    #[test]
    fn test_moments_matches_reference() {
        assert_eq!(thresh_moments(&bimodal_reference_histogram()), 160);
    }

    #[test]
    fn test_huang_matches_reference() {
        assert_eq!(thresh_huang(&bimodal_reference_histogram()), 40);
    }

    #[test]
    fn test_intermodes_matches_reference() {
        assert_eq!(thresh_intermodes(&bimodal_reference_histogram()), 105);
    }

    #[test]
    fn test_isodata_matches_reference() {
        // The C++ reference truncates its background-mean accumulator via
        // integer division; this crate intentionally uses proper float
        // division instead (see threshold.rs review notes), so this value
        // can drift by ~1 bin from a bit-exact C++ run on other histograms.
        // On this histogram both agree.
        assert_eq!(thresh_isodata(&bimodal_reference_histogram()), 105);
    }

    #[test]
    fn test_max_entropy_matches_reference() {
        assert_eq!(thresh_max_entropy(&bimodal_reference_histogram()), 167);
    }

    #[test]
    fn test_mean_matches_reference() {
        assert_eq!(thresh_mean(&bimodal_reference_histogram()), 95);
    }

    #[test]
    fn test_minimum_matches_reference() {
        assert_eq!(thresh_minimum(&bimodal_reference_histogram()), 61);
    }

    #[test]
    fn test_percentile_matches_reference() {
        assert_eq!(thresh_percentile(&bimodal_reference_histogram()), 38);
    }

    #[test]
    fn test_renyi_entropy_matches_reference() {
        assert_eq!(thresh_renyi_entropy(&bimodal_reference_histogram()), 165);
    }

    #[test]
    fn test_shanbhag_matches_reference() {
        assert_eq!(thresh_shanbhag(&bimodal_reference_histogram()), 170);
    }

    #[test]
    fn test_yen_matches_reference() {
        assert_eq!(thresh_yen(&bimodal_reference_histogram()), 164);
    }
}
