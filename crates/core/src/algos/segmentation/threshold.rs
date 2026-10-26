//! # threshold
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::pipeline::pipeline_cache::CacheAddress;
use crate::pipeline::scope::Scope;
use evanalyzer_cfg::core_types::{
    CitationMetadata, InternalErrors, MemoryId, PixelUnits, SegmentationClass,
};
use macros::CommandsMeta;
use std::collections::HashMap;

use crate::{
    ImageTile,
    algos::{ExecutionScope, ImageAlgorithm},
    image::ImageContainer,
    pipeline::{pipeline_cache::GlobalPipelineCache, pipeline_context::PipelineContext},
};

/// The mathematical strategy used to determine the optimal global threshold.
///
/// Most methods analyze the image histogram to find a "cut-off" point that
/// best separates the foreground from the background.
#[derive(Debug, Clone, Copy, PartialEq, CommandsMeta)]
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
    /// Minimizes intra-class variance (maximizes inter-class variance).
    ///
    /// `Two` behaves exactly as before - one cut, `thresh_otsu`. `Three` jointly
    /// finds two simultaneous cuts (Otsu's variance criterion is additive across
    /// classes, so this is the same search one level deeper - see `thresh_otsu_multi`)
    /// and returns whichever cut `middle_class` selects: the lower cut puts
    /// the middle intensity class into the foreground, the upper cut into the
    /// background - useful when dim structures sit between object and
    /// background.
    Otsu {
        #[cmdsmeta(default = OtsuClasses::Two)]
        classes: OtsuClasses,
    },
    /// Assumes a fixed percentage of pixels belong to the foreground.
    Percentile,
    /// Based on the Renyi entropy of the histogram; a generalization of MaxEntropy.
    RenyiEntropy,
    /// An extension of Kapur's method using a different coefficient for entropy.
    Shanbhag,
    /// Minimizes a cost function based on the discrepancy between two classes.
    Yen,
    /// Trims outliers from both ends of the sorted pixel distribution, then
    /// sets the threshold at `deviations_above_average` standard deviations
    /// (or MADs) above the trimmed mean (or median). Well suited to images
    /// with a clean, low
    /// background and sparse foreground, since the outlier trim keeps bright
    /// foreground pixels from dragging the background estimate upward.
    RobustBackground {
        #[cmdsmeta(default = 0.05)]
        lower_outlier_fraction: f32,
        #[cmdsmeta(default = 0.05)]
        upper_outlier_fraction: f32,
        #[cmdsmeta(default = Averaging::Mean)]
        averaging_method: Averaging,
        #[cmdsmeta(default = 2.0)]
        deviations_above_average: f32,
    },
}

/// How many populations Otsu splits the histogram into.
///
/// `Three` needs a second, jointly-optimized cut - see [`ThresholdMethod::Otsu`].
/// Not generalized past three: a fourth class would need a genuine "which of the
/// K-1 cuts" selector in place of `middle_class`, not just a wider range - see
/// the multilevel-thresholding note on `thresh_otsu_multi`.
#[derive(Debug, Clone, Copy, PartialEq, CommandsMeta)]
pub enum OtsuClasses {
    Two,
    Three {
        #[cmdsmeta(default = OtsuMiddleClass::Background)]
        middle_class: OtsuMiddleClass,
    },
}

/// Which side of a three-class Otsu split the middle-intensity population joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtsuMiddleClass {
    Foreground,
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Averaging {
    Mean,
    Median,
}

/// Which image the threshold *value* is calculated from. Allows calculating a
/// threshold on one image (e.g. the raw, unblurred image, whose histogram is
/// not flattened by the blur) and applying it to another (the blurred image
/// actually being segmented): the cut-off computed here is always applied
/// against the
/// pipeline's actual current image - see [`ThresholdEntry::value_source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThresholdValueSource {
    /// Calculate the threshold from the pipeline's current image - the same
    /// image the threshold is applied to. This is the historical behavior.
    ActualImage,
    /// Calculate the threshold from the unedited image this channel started
    /// the pipeline with, ignoring any preprocessing (blur, illumination
    /// correction, ...) applied so far. Looked up in the pipeline cache under
    /// `ImageAddress::Channel` for the current image's channel.
    RawImage,
    /// Calculate the threshold from a user-defined snapshot stored earlier in
    /// the pipeline (e.g. via the `ImageCache` command), addressed by
    /// `MemoryId`.
    Memory(MemoryId),
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
    #[cmdsmeta(default = PixelUnits::Bit, visibility = Advanced)]
    pub unit: PixelUnits,

    /// The classification ID assigned to pixels falling within this threshold range.
    pub segmentation_class: SegmentationClass,

    /// Defines which image should be taken for calculating the threhold value.
    ///
    /// This is the source which is used to calculate the threshold value.
    /// The value itself is applied to the actual image the pipeline stands.
    #[cmdsmeta(default = ThresholdValueSource::ActualImage, optional = true, visibility = Advanced)]
    pub value_source: ThresholdValueSource,
}

/// A filter that segments an image into discrete classes based on intensity.
///
/// This supports "Multi-Otsu" style behavior by allowing several
/// thresholds. Each pixel is evaluated against them to determine which
/// segmentation class it belongs to.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "segment", next = "instance_segmentation")]
pub struct Threshold {
    /// A list of thresholding layers. Overlapping ranges are resolved
    /// by the order of the vector (last-in priority).
    pub thresholds: Vec<ThresholdEntry>,
}

impl ImageAlgorithm for Threshold {
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let nr_of_bits = ctx.image_meta.nr_of_bits;
        // Resolved before the mutable borrow below, since `RawImage` needs to
        // know which channel the pipeline's current image belongs to.
        let channel = ctx.get_image_plane().map(|p| p.c);
        let tile = ctx.image_meta.image_tile_info.clone();
        let scope = ctx.scope;
        let (input_data, segmentation_map) = ctx.get_f32_gray_and_segmentation_mask_mut()?;
        let actual_slice = input_data.as_slice();

        // Histograms are rescaled to the source image's *actual observed*
        // min/max (a per-image linear contrast stretch).
        //
        // Real images rarely use the full theoretical bit-depth range (e.g. a
        // 12-bit sensor stored in a 16-bit container), so binning directly off
        // the bit-depth-normalized relative values collapses nearly all pixels
        // into a handful of low bins - degenerate input for methods like Li,
        // whose log-mean formula breaks down once the background mean lands
        // exactly on bin 0.
        //
        // Each entry's `value_source` picks which image that histogram (and
        // therefore the computed cut-off) is built from, so a threshold can be
        // calculated on one image (e.g. the raw, unblurred image) and applied
        // to another. The cut-off
        // itself is always applied against `actual_slice` below, never the
        // source image, so entries sharing a `value_source` share one
        // memoized histogram instead of rebuilding it per entry.
        let mut hist_memo: HashMap<ThresholdValueSource, ([f32; 256], f32, f32)> = HashMap::new();

        // Pre-resolve each entry's [min, max] range in relative (0.0–1.0) space.
        let mut normalized: Vec<(f32, f32, u32)> = Vec::with_capacity(self.thresholds.len());
        for s in &self.thresholds {
            let floor = s.unit.to_relative(s.min_threshold, nr_of_bits);
            let ceiling = s.unit.to_relative(s.max_threshold, nr_of_bits);
            let min = match &s.method {
                ThresholdMethod::None | ThresholdMethod::Manual => floor,
                // Unlike every other auto method below, `RobustBackground` is
                // computed directly from the sorted pixel population rather
                // than a 256-bin histogram - the method sorts and trims the
                // *exact* pixel values, and quantizing through the shared
                // histogram first (which many distinct values collapse into
                // the same bin) would shift the trim boundaries and the
                // mean/spread, so the threshold would no longer be exact. No
                // `+1`/rescale step either: the trimmed-mean-plus-deviations formula
                // already yields a threshold value in the same relative units
                // as `actual_slice`, not a bin index to map back.
                ThresholdMethod::RobustBackground {
                    lower_outlier_fraction,
                    upper_outlier_fraction,
                    averaging_method,
                    deviations_above_average,
                } => with_source_slice(
                    s.value_source,
                    channel,
                    tile.clone(),
                    scope,
                    cache,
                    actual_slice,
                    |data| {
                        thresh_robust_background(
                            data,
                            *lower_outlier_fraction,
                            *upper_outlier_fraction,
                            *averaging_method,
                            *deviations_above_average,
                        )
                    },
                )?
                .clamp(floor, ceiling),
                method => {
                    if !hist_memo.contains_key(&s.value_source) {
                        let hist = build_source_histogram(
                            s.value_source,
                            channel,
                            tile.clone(),
                            scope,
                            cache,
                            actual_slice,
                        )?;
                        hist_memo.insert(s.value_source, hist);
                    }
                    let (hist, dmin, dmax) = hist_memo.get(&s.value_source).unwrap();
                    // As in ImageJ (`scaleAndSetThreshold(0,
                    // calcThresholdValue(...) + 1 + cValue, ...)`), the raw
                    // split bin is always mapped one
                    // bin *above* itself before rescaling: the split bin
                    // is background, the next bin up is where foreground
                    // starts. Omitting this "+1" makes the threshold one
                    // bin too permissive, letting background-adjacent
                    // noise through as foreground.
                    //
                    // Clamped to 254: bin 255 always rescales to exactly
                    // `dmax` (`255.0 / 255.0 == 1.0`), and the classification
                    // below is a strict `>` against that value - but `dmax`
                    // is by construction the brightest pixel actually
                    // present, so nothing can ever be strictly greater than
                    // it. A cleanly bimodal image (e.g. background at 0,
                    // objects at one fixed high value) commonly makes an
                    // auto method return bin 254 - every bin between the two
                    // populations ties for the same between-class variance,
                    // and the winning tie lands on the last one checked -
                    // which, once bumped by the "+1" above, would otherwise
                    // silently produce zero foreground pixels for every
                    // entry using this threshold, no matter how many
                    // genuinely bright pixels the image has. Any method
                    // returning 254 (or, before this clamp, even 255 itself
                    // for a method whose own range includes it) hits the
                    // same unreachable-threshold failure, not just Otsu.
                    let bin = (compute_auto_threshold(method, hist) + 1).min(254);
                    let relative = if *dmax > *dmin {
                        dmin + (bin as f32 / 255.0) * (dmax - dmin)
                    } else {
                        *dmin
                    };
                    relative.clamp(floor, ceiling)
                }
            };
            normalized.push((min, ceiling, s.segmentation_class.as_u32()));
        }

        let output_slice = segmentation_map.as_slice_mut();
        for (out_pixel, &in_pixel) in output_slice.iter_mut().zip(actual_slice.iter()) {
            let mut assigned_id = SegmentationClass::BACKGROUND.as_u32();
            for &(min, max, class_id) in &normalized {
                // The lower bound is exclusive (strict `>`), the upper
                // bound inclusive (`<=`).
                let is_in_range = (in_pixel > min && in_pixel <= max) as u32;
                assigned_id = (is_in_range * class_id) | ((1 - is_in_range) * assigned_id);
            }
            *out_pixel = assigned_id;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Threshold"
    }

    /// The papers of every automatic method used (each once, in order);
    /// EVAnalyzer itself when only manual thresholds are used.
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut citations: Vec<&'static CitationMetadata> = Vec::new();
        for citation in self
            .thresholds
            .iter()
            .flat_map(|entry| method_citations(&entry.method).iter().copied())
        {
            if !citations.iter().any(|c| c.cite_key == citation.cite_key) {
                citations.push(citation);
            }
        }
        if citations.is_empty() {
            citations.push(&CitationMetadata::DANMAYR);
        }
        citations
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

/// Resolves `source` to a 256-bin histogram (plus its observed min/max),
/// mirroring the `dmin`/`dmax` rescale `Threshold::execute` used to only do
/// against the pipeline's current image.
fn build_source_histogram(
    source: ThresholdValueSource,
    channel: Option<i32>,
    tile: ImageTile,
    scope: Scope,
    cache: &GlobalPipelineCache,
    actual_data: &[f32],
) -> Result<([f32; 256], f32, f32), InternalErrors> {
    with_source_slice(
        source,
        channel,
        tile,
        scope,
        cache,
        actual_data,
        histogram_from_slice,
    )
}

/// Resolves `source` to a pixel slice and applies `f` to it while the slice
/// is still valid, returning `f`'s result. `RawImage` and `Memory` look up an
/// `Arc<ImageContainer>` from the pipeline cache; `f` runs while it's alive
/// so no borrow of the cache needs to outlive this call - shared by every
/// per-entry computation that needs `source`'s pixels, whether that's
/// `build_source_histogram`'s binning or `RobustBackground`'s direct
/// sorted-population statistics.
fn with_source_slice<R>(
    source: ThresholdValueSource,
    channel: Option<i32>,
    tile: ImageTile,
    // The scope the command runs in - where a module memory slot lives.
    scope: Scope,
    cache: &GlobalPipelineCache,
    actual_data: &[f32],
    f: impl FnOnce(&[f32]) -> R,
) -> Result<R, InternalErrors> {
    match source {
        ThresholdValueSource::ActualImage => Ok(f(actual_data)),
        ThresholdValueSource::RawImage => {
            let channel = channel.ok_or_else(|| {
                InternalErrors::Generic(
                    "Threshold: cannot resolve RawImage - current image has no channel/plane \
                     information"
                        .into(),
                )
            })?;
            let raw = cache
                .get_image_from_channel_cache(channel, tile)
                .ok_or_else(|| {
                    InternalErrors::CacheMiss(format!(
                        "Threshold: no raw image cached for channel {channel}"
                    ))
                })?;
            gray_slice(&raw).map(f)
        }
        ThresholdValueSource::Memory(id) => {
            let snapshot = cache
                .image_cache
                .get(&CacheAddress::memory(id, scope)?)
                .ok_or_else(|| {
                    InternalErrors::CacheMiss(format!(
                        "Threshold: no image found in memory cache at {id:?}"
                    ))
                })?;
            gray_slice(&snapshot).map(f)
        }
    }
}

/// Extracts the pixel slice of a gray image container, erroring for any other
/// format - `Threshold` only ever operates on single-channel images.
fn gray_slice(container: &ImageContainer) -> Result<&[f32], InternalErrors> {
    match container {
        ImageContainer::F32Gray(img) => Ok(img.as_slice()),
        other => Err(InternalErrors::FormatMismatch {
            expected: "F32Gray".into(),
            found: format!("{other:?}"),
        }),
    }
}

/// Builds a histogram rescaled to `data`'s own observed min/max, alongside
/// that min/max (needed to map the winning bin back to a relative value).
fn histogram_from_slice(data: &[f32]) -> ([f32; 256], f32, f32) {
    let mut dmin = f32::MAX;
    let mut dmax = f32::MIN;
    for &v in data {
        dmin = dmin.min(v);
        dmax = dmax.max(v);
    }
    (build_histogram(data, dmin, dmax), dmin, dmax)
}

// ── Histogram ────────────────────────────────────────────────────────────────

/// Builds a 256-bin histogram from f32 pixel data, rescaled so that `min`
/// maps to bin 0 and `max` maps to bin 255 - a per-image contrast stretch,
/// giving auto-threshold
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
        // Round to the nearest bin rather than truncate.
        let bin = ((normalized * 255.0).round() as usize).min(255);
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
        ThresholdMethod::Otsu { classes } => match classes {
            OtsuClasses::Two => thresh_otsu(hist),
            OtsuClasses::Three { middle_class } => {
                let (t1, t2) = thresh_otsu_multi(hist);
                match middle_class {
                    // Middle class joins the background: only the brightest
                    // population is foreground, so the *higher* cut is used.
                    OtsuMiddleClass::Background => t2,
                    // Middle class joins the foreground: everything above the
                    // darkest population is foreground, so the *lower* cut
                    // is used.
                    OtsuMiddleClass::Foreground => t1,
                }
            }
        },
        ThresholdMethod::Percentile => thresh_percentile(hist),
        ThresholdMethod::RenyiEntropy => thresh_renyi_entropy(hist),
        ThresholdMethod::Shanbhag => thresh_shanbhag(hist),
        ThresholdMethod::Yen => thresh_yen(hist),
        // Never actually reached: `Threshold::execute` carves `RobustBackground`
        // out into its own direct-on-pixels code path (see `with_source_slice`
        // + `thresh_robust_background`) before it would otherwise land here,
        // same as `None`/`Manual` above - kept only so this match stays
        // exhaustive against `ThresholdMethod`.
        ThresholdMethod::None
        | ThresholdMethod::Manual
        | ThresholdMethod::RobustBackground { .. } => 0,
    }
}

// ── Algorithm implementations ─────────────────────────────────────────────────
// Each method is implemented from the definitions in its original paper (cited
// in its doc comment and in `method_citation`). Every function takes a 256-bin
// histogram whose bin index doubles as the gray level, and returns the
// threshold *bin*: bins `0..=t` are background, bins above `t` foreground
// (`Threshold::execute` adds the "+1" when mapping back to an intensity).
//
// Shared conventions, stated once here instead of on every method:
// - A histogram with fewer than two occupied bins has no split to optimize;
//   every method returns its lowest occupied bin then (0 when empty). Since
//   `build_histogram` stretches the observed range to `0..=255`, a single
//   occupied bin in practice is always bin 0.
// - A method whose criterion yields a real-valued *decision boundary* `t*`
//   (Li, minimum error, IsoData) returns `floor(t*)`: the last bin whose gray
//   level still lies on the background side of the boundary.
// - Ties in an arg max / arg min are broken towards the lowest bin.
// - All arithmetic runs in `f64` on fixed-size stack arrays - no allocation.

/// Cumulative count and cumulative first moment, `c[i] = Σ_{j<i} h[j]` and
/// `s[i] = Σ_{j<i} j·h[j]`, so any class `a..=b` has `c[b + 1] - c[a]`
/// pixels with gray-level sum `s[b + 1] - s[a]`.
fn prefix_sums(hist: &[f32; 256]) -> ([f64; 257], [f64; 257]) {
    let mut c = [0.0f64; 257];
    let mut s = [0.0f64; 257];
    for (i, &h) in hist.iter().enumerate() {
        c[i + 1] = c[i] + h as f64;
        s[i + 1] = s[i] + i as f64 * h as f64;
    }
    (c, s)
}

/// First and last occupied bin, or `None` for an empty histogram.
fn occupied_range(hist: &[f32; 256]) -> Option<(usize, usize)> {
    let lo = hist.iter().position(|&h| h > 0.0)?;
    let hi = hist.iter().rposition(|&h| h > 0.0)?;
    Some((lo, hi))
}

/// The threshold every method returns when there is nothing to split (see
/// the shared conventions above): `Err(lowest occupied bin)`, otherwise the
/// occupied range.
fn splittable_range(hist: &[f32; 256]) -> Result<(usize, usize), usize> {
    match occupied_range(hist) {
        None => Err(0),
        Some((lo, hi)) if lo == hi => Err(lo),
        Some(range) => Ok(range),
    }
}

/// Maps a real-valued decision boundary to the last bin on its lower side.
fn boundary_bin(t: f64) -> usize {
    t.floor().clamp(0.0, 255.0) as usize
}

/// Otsu (1979): picks the cut `k` that maximizes the between-class variance
/// `σ_B²(k) = (μ_T·ω(k) − μ(k))² / (ω(k)·(1 − ω(k)))`, with `ω(k)` the
/// fraction of pixels in bins `0..=k`, `μ(k)` their first cumulative moment
/// and `μ_T` the global mean. Maximizing `σ_B²` is equivalent to minimizing
/// the within-class variance, i.e. the two classes are as compact and as far
/// apart as possible. Only cuts leaving both classes non-empty are
/// considered.
///
/// Ties go to the lowest bin. This matters for a cleanly bimodal histogram:
/// `ω` and `μ` don't change across an empty gap, so every cut in the gap
/// ties, and the cut lands at the end of the lower population - never at
/// the top of the range, where the "+1" in `Threshold::execute` could reach
/// the image maximum (the `.min(254)` clamp there stays as a backstop).
fn thresh_otsu(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let (c, s) = prefix_sums(hist);
    let n = c[256];
    let mean = s[256] / n;
    let mut best = 0usize;
    let mut best_var = f64::NEG_INFINITY;
    for k in 0..256 {
        let w = c[k + 1];
        let w_upper = n - w;
        if w <= 0.0 || w_upper <= 0.0 {
            continue;
        }
        // n²·σ_B²: the constant factor doesn't change the arg max.
        let d = mean * w - s[k + 1];
        let var = d * d / (w * w_upper);
        if var > best_var {
            best_var = var;
            best = k;
        }
    }
    best
}

/// Three-class Otsu: jointly searches for the two simultaneous cuts `(t1,
/// t2)`, `t1 < t2`, that maximize the between-class variance across three
/// populations (0..=t1, t1+1..=t2, t2+1..255). Otsu's variance criterion is
/// additive across classes - the single-cut between-class variance is,
/// up to scaling, `w0*mu0² + w1*mu1²` minus the constant `n*muT²`, which
/// doesn't affect which cut wins - so the three-class case is the same
/// maximization one level deeper: `w0*mu0² + w1*mu1² + w2*mu2²`, searched
/// over both cuts.
///
/// Each returned cut is the *last* bin of its (lower) class, exactly like
/// `thresh_otsu`'s return value, so the same "+1" scaling in
/// `Threshold::execute` applies unchanged to whichever cut ends up selected.
fn thresh_otsu_multi(hist: &[f32; 256]) -> (usize, usize) {
    let n: f64 = hist.iter().map(|&v| v as f64).sum();
    if n == 0.0 {
        return (0, 0);
    }

    // Prefix count/sum through (and including) bin i.
    let mut p = [0.0f64; 256];
    let mut s = [0.0f64; 256];
    let mut cum_p = 0.0f64;
    let mut cum_s = 0.0f64;
    for i in 0..256 {
        cum_p += hist[i] as f64;
        cum_s += i as f64 * hist[i] as f64;
        p[i] = cum_p;
        s[i] = cum_s;
    }
    let total_s = s[255];

    // w*mu² = (sum)²/w for a class of total weight `w` and weighted sum
    // `sum`; contributes 0 for an empty class.
    let class_term = |w: f64, sum: f64| -> f64 { if w > 0.0 { sum * sum / w } else { 0.0 } };

    let mut best = (0usize, 1usize);
    let mut best_bcv = f64::MIN;
    for t1 in 0..255usize {
        let w0 = p[t1];
        let s0 = s[t1];
        let term0 = class_term(w0, s0);
        for t2 in (t1 + 1)..255usize {
            let w1 = p[t2] - w0;
            let s1 = s[t2] - s0;
            let w2 = n - p[t2];
            let s2 = total_s - s[t2];
            let bcv = term0 + class_term(w1, s1) + class_term(w2, s2);
            if bcv > best_bcv {
                best_bcv = bcv;
                best = (t1, t2);
            }
        }
    }
    best
}

/// Li's minimum cross entropy threshold (Li & Lee 1993), found with the
/// iterative scheme of Li & Tam (1998). Li & Lee choose the cut whose
/// two-level reconstruction (every pixel replaced by its class mean) has the
/// smallest cross entropy to the original image. Setting the derivative of
/// that cross entropy to zero gives the fixed-point condition
/// `t = (μ_b − μ_a) / (ln μ_b − ln μ_a)`, with `μ_a`/`μ_b` the mean gray
/// level of the pixels at/below and above `t`; Li & Tam iterate it from an
/// initial guess, which reaches the minimum in a few steps instead of an
/// exhaustive search.
///
/// Conventions: starts at the global mean, gray levels are the bin indices,
/// stops once `t` moves by less than half a bin (or after 1000 steps), and
/// returns `floor(t)`. A lower class whose mean is 0 (all of it in bin 0)
/// sends `t` to 0, the limit of the update as `μ_a → 0`.
fn thresh_li(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let (c, s) = prefix_sums(hist);
    let n = c[256];
    let mut t = s[256] / n;
    for _ in 0..1000 {
        let b = boundary_bin(t);
        let w_a = c[b + 1];
        let w_b = n - w_a;
        if w_a <= 0.0 || w_b <= 0.0 {
            break;
        }
        let mean_a = s[b + 1] / w_a;
        let mean_b = (s[256] - s[b + 1]) / w_b;
        let next = if mean_a > 0.0 {
            (mean_b - mean_a) / (mean_b.ln() - mean_a.ln())
        } else {
            0.0
        };
        let converged = (next - t).abs() < 0.5;
        t = next;
        if converged {
            break;
        }
    }
    boundary_bin(t)
}

/// Kittler & Illingworth's (1986) minimum error threshold, using the
/// iterative procedure from the same paper. The histogram is modelled as a
/// mixture of two normal distributions; the Bayes-optimal (minimum error)
/// boundary is where the two weighted densities are equal,
/// `P₁/σ₁·exp(−(g−μ₁)²/2σ₁²) = P₂/σ₂·exp(−(g−μ₂)²/2σ₂²)`. Taking logs gives
/// the quadratic
/// `(g−μ₁)²/σ₁² + ln σ₁² − 2 ln P₁ = (g−μ₂)²/σ₂² + ln σ₂² − 2 ln P₂`.
/// Each step estimates `P`, `μ`, `σ²` of both classes for the current cut,
/// solves the quadratic and moves the cut to its root as given by Glasbey
/// (1993, MINERROR(I)), until the cut stops changing.
///
/// Conventions: starts at the global mean, uses population variances, and
/// returns `floor(g)` of the final root. Stops early (keeping the current
/// cut) when a class is empty or has zero variance - the normal model is
/// undefined there - or the quadratic has no real root.
fn thresh_min_error(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let (c, s) = prefix_sums(hist);
    let mut s2 = [0.0f64; 257];
    for (i, &h) in hist.iter().enumerate() {
        s2[i + 1] = s2[i] + (i * i) as f64 * h as f64;
    }
    let n = c[256];
    let mut t = s[256] / n;
    let mut prev_bin = usize::MAX;
    for _ in 0..256 {
        let b = boundary_bin(t);
        if b == prev_bin {
            break;
        }
        prev_bin = b;
        let w1 = c[b + 1];
        let w2 = n - w1;
        if w1 <= 0.0 || w2 <= 0.0 {
            break;
        }
        let m1 = s[b + 1] / w1;
        let m2 = (s[256] - s[b + 1]) / w2;
        let v1 = s2[b + 1] / w1 - m1 * m1;
        let v2 = (s2[256] - s2[b + 1]) / w2 - m2 * m2;
        if v1 <= 1e-12 || v2 <= 1e-12 {
            break;
        }
        // a·g² + b·g + c = 0, from expanding the equal-density condition.
        let qa = 1.0 / v1 - 1.0 / v2;
        let qb = -2.0 * (m1 / v1 - m2 / v2);
        let qc = m1 * m1 / v1 - m2 * m2 / v2 + (v1 / v2).ln() - 2.0 * (w1 / w2).ln();
        // Glasbey (1993): the new cut is the root `(w₁ + √(w₁² − w₀w₂)) / w₀`
        // of `w₀g² − 2w₁g + w₂ = 0` (here `qa = w₀`, `qb = −2w₁`, `qc = w₂`);
        // the iteration ends when the quadratic has no real root.
        let root = if qa.abs() < 1e-12 {
            // Equal variances: the quadratic degenerates to a line.
            Some(-qc / qb)
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            (disc >= 0.0).then(|| (-qb + disc.sqrt()) / (2.0 * qa))
        };
        match root {
            Some(g) => t = g,
            None => break,
        }
    }
    boundary_bin(t)
}

/// Zack, Rogers & Latt's (1977) triangle method, a purely geometric rule
/// for histograms with one dominant peak and a long tail on one side. A
/// line is drawn from the peak to the far end of the longer tail; the
/// threshold is the bin where the histogram lies furthest below that line,
/// i.e. where the tail "bends away" from the peak's straight-line decay.
/// Because the line is fixed, the perpendicular distance is the vertical gap
/// times a constant, so the vertical gap is maximized directly.
///
/// Conventions: the peak is the lowest bin with the maximum count; the tail
/// ends at the outermost occupied bin (with its own count); the longer side
/// wins, the upper one on a tie. Ties in the gap go to the bin nearest the
/// peak. For a tail on the dark side the returned bin still separates
/// `0..=t` from the rest.
fn thresh_triangle(hist: &[f32; 256]) -> usize {
    let (lo, hi) = match splittable_range(hist) {
        Ok(r) => r,
        Err(t) => return t,
    };
    let mut peak = lo;
    for i in lo..=hi {
        if hist[i] > hist[peak] {
            peak = i;
        }
    }
    let h_peak = hist[peak] as f64;
    let upper_tail = hi - peak >= peak - lo;
    let end = if upper_tail { hi } else { lo };
    if end == peak {
        return peak;
    }
    let h_end = hist[end] as f64;
    let span = end as f64 - peak as f64;
    let mut best = peak;
    let mut best_gap = f64::NEG_INFINITY;
    let mut consider = |i: usize| {
        let line = h_peak + (h_end - h_peak) * (i as f64 - peak as f64) / span;
        let gap = line - hist[i] as f64;
        if gap > best_gap {
            best_gap = gap;
            best = i;
        }
    };
    if upper_tail {
        (peak + 1..end).for_each(&mut consider);
    } else {
        (end + 1..peak).rev().for_each(&mut consider);
    }
    best
}

/// Tsai's (1985) moment-preserving threshold. The image is replaced by a
/// two-level image (gray levels `z₀ < z₁`, a fraction `p₀` of the pixels at
/// `z₀`) that keeps the first three moments `m₁, m₂, m₃` of the histogram
/// (`m₀ = 1`). `z₀, z₁` are the roots of `z² + c₁z + c₀ = 0`, where
/// `c₀, c₁` solve `c₀m₀ + c₁m₁ = −m₂` and `c₀m₁ + c₁m₂ = −m₃`, and
/// `p₀ = (z₁ − m₁) / (z₁ − z₀)`. The threshold is the `p₀`-tile: the cut
/// whose background fraction (bins `0..=t`) is nearest to `p₀` (Glasbey
/// 1993), among the cuts that leave both classes non-empty.
fn thresh_moments(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let n: f64 = hist.iter().map(|&h| h as f64).sum();
    let (mut m1, mut m2, mut m3) = (0.0f64, 0.0f64, 0.0f64);
    for (i, &h) in hist.iter().enumerate() {
        let p = h as f64 / n;
        let z = i as f64;
        m1 += z * p;
        m2 += z * z * p;
        m3 += z * z * z * p;
    }
    let det = m2 - m1 * m1;
    let c0 = (m1 * m3 - m2 * m2) / det;
    let c1 = (m1 * m2 - m3) / det;
    let r = (c1 * c1 - 4.0 * c0).max(0.0).sqrt();
    let z0 = 0.5 * (-c1 - r);
    let z1 = 0.5 * (-c1 + r);
    let p0 = (z1 - m1) / (z1 - z0);
    let (lo, hi) = occupied_range(hist).unwrap_or((0, 0));
    let (c, _) = prefix_sums(hist);
    let mut best = lo;
    for t in lo..hi {
        if (c[t + 1] / n - p0).abs() < (c[best + 1] / n - p0).abs() {
            best = t;
        }
    }
    best
}

/// Shannon's function `S(μ) = −μ ln μ − (1 − μ) ln(1 − μ)`, with `0·ln 0 = 0`.
fn shannon(mu: f64) -> f64 {
    let term = |x: f64| if x > 0.0 { -x * x.ln() } else { 0.0 };
    term(mu) + term(1.0 - mu)
}

/// Huang & Wang's (1995) fuzzy threshold. For a cut `t`, every gray level
/// `g` belongs to its class with membership `μ(g) = 1 / (1 + |g − μ_k|/C)`,
/// where `μ_k` is the mean of its class (`k = 0` for `g ≤ t`, `k = 1`
/// above, eq. 2/3) and `C = g_max − g_min` keeps `μ` within `[0.5, 1]`.
/// The paper's algorithm (§2.3, eq. 20/21) rounds the class means to whole
/// gray levels only so that memberships can be tabulated in advance; the
/// exact means are used here, as the criterion is shallow and rounding
/// alone moves its minimum by several bins on many histograms. The cut is
/// the one whose image is least fuzzy, i.e. that minimizes the entropy
/// measure `E(t) = Σ_g h(g)·S(μ(g))` with Shannon's function `S` (the
/// paper's `1/(MN ln 2)` normalization doesn't move the minimum). Only cuts
/// leaving both classes non-empty are considered.
fn thresh_huang(hist: &[f32; 256]) -> usize {
    let (lo, hi) = match splittable_range(hist) {
        Ok(r) => r,
        Err(t) => return t,
    };
    let (c, s) = prefix_sums(hist);
    let range = (hi - lo) as f64;
    let mut best = lo;
    let mut best_e = f64::INFINITY;
    for t in lo..hi {
        let w0 = c[t + 1] - c[lo];
        let w1 = c[hi + 1] - c[t + 1];
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let mu0 = (s[t + 1] - s[lo]) / w0;
        let mu1 = (s[hi + 1] - s[t + 1]) / w1;
        let mut e = 0.0f64;
        for (g, &h) in hist.iter().enumerate().take(hi + 1).skip(lo) {
            let h = h as f64;
            if h > 0.0 {
                let mean = if g <= t { mu0 } else { mu1 };
                e += h * shannon(1.0 / (1.0 + (g as f64 - mean).abs() / range));
            }
        }
        if e < best_e {
            best_e = e;
            best = t;
        }
    }
    best
}

/// Counts the strict local maxima of `y` (`y[i-1] < y[i] > y[i+1]`), reading
/// the histogram as zero outside `0..=255` (there are no pixels there), and
/// writes the positions of the first two into `peaks`. A flat-topped plateau
/// is not a strict maximum; it only becomes one once smoothing rounds it.
fn local_maxima(y: &[f64; 256], peaks: &mut [usize; 2]) -> usize {
    let mut count = 0;
    for i in 0..256 {
        let left = if i > 0 { y[i - 1] } else { 0.0 };
        let right = if i < 255 { y[i + 1] } else { 0.0 };
        if y[i] > left && y[i] > right {
            if count < 2 {
                peaks[count] = i;
            }
            count += 1;
        }
    }
    count
}

/// Smooths `hist` with a repeated three-bin running mean (the histogram
/// read as zero outside `0..=255`, every pass computed from the previous
/// pass' values) until it has exactly two strict local maxima - the
/// procedure of Prewitt & Mendelsohn (1966) as formalized by Glasbey (1993).
/// An already bimodal histogram is used unsmoothed. Returns the smoothed
/// histogram and the two mode positions, or `None` when 10 000 passes don't
/// produce a bimodal histogram (e.g. three modes merging straight into one).
fn smooth_until_bimodal(hist: &[f32; 256]) -> Option<([f64; 256], [usize; 2])> {
    let mut y = [0.0f64; 256];
    for (dst, &h) in y.iter_mut().zip(hist.iter()) {
        *dst = h as f64;
    }
    let mut next = [0.0f64; 256];
    let mut peaks = [0usize; 2];
    for _ in 0..10_000 {
        if local_maxima(&y, &mut peaks) == 2 {
            return Some((y, peaks));
        }
        for i in 0..256 {
            let left = if i > 0 { y[i - 1] } else { 0.0 };
            let right = if i < 255 { y[i + 1] } else { 0.0 };
            next[i] = (left + y[i] + right) / 3.0;
        }
        std::mem::swap(&mut y, &mut next);
    }
    None
}

/// Prewitt & Mendelsohn's (1966) intermodes threshold: the histogram is
/// smoothed until it is bimodal (see `smooth_until_bimodal`) and the
/// threshold is the midpoint between the two modes, `floor((p₁ + p₂) / 2)`.
/// Returns the lowest occupied bin when no bimodal smoothing exists.
fn thresh_intermodes(hist: &[f32; 256]) -> usize {
    let lo = match splittable_range(hist) {
        Ok((lo, _)) => lo,
        Err(t) => return t,
    };
    match smooth_until_bimodal(hist) {
        Some((_, [p1, p2])) => (p1 + p2) / 2,
        None => lo,
    }
}

/// Ridler & Calvard's (1978) iterative selection ("IsoData", intermeans):
/// starting from the global mean, the threshold is repeatedly replaced by
/// the average of the two class means it induces,
/// `t ← (μ_{≤t} + μ_{>t}) / 2`, until the split no longer changes. At the
/// fixed point the threshold lies exactly halfway between the mean
/// background and mean foreground level. Returns `floor(t)`; stops early if
/// a class becomes empty, and after 256 steps at the latest.
fn thresh_isodata(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let (c, s) = prefix_sums(hist);
    let n = c[256];
    let mut t = s[256] / n;
    let mut prev_bin = usize::MAX;
    for _ in 0..256 {
        let b = boundary_bin(t);
        if b == prev_bin {
            break;
        }
        prev_bin = b;
        let w1 = c[b + 1];
        let w2 = n - w1;
        if w1 <= 0.0 || w2 <= 0.0 {
            break;
        }
        t = 0.5 * (s[b + 1] / w1 + (s[256] - s[b + 1]) / w2);
    }
    boundary_bin(t)
}

/// Arg max over the cut `t` of the summed Rényi entropies of order `alpha`
/// of the two class distributions `{p_i / P_t}_{i ≤ t}` and
/// `{p_i / (1 − P_t)}_{i > t}`. `alpha == 1.0` is the Shannon limit
/// (Kapur et al. 1985). Only cuts leaving both classes non-empty count;
/// ties go to the lowest bin.
///
/// Uses `Σ_{i≤t} (h_i/W)^α = Σ_{i≤t} h_i^α / W^α` (and the Shannon analogue
/// `H = ln W − Σ h_i ln h_i / W`) on raw counts, so every cut is O(1) with
/// prefix sums.
fn max_renyi_entropy_cut(hist: &[f32; 256], alpha: f64) -> usize {
    let shannon_limit = alpha == 1.0;
    let mut c = [0.0f64; 257];
    let mut f = [0.0f64; 257];
    for (i, &h) in hist.iter().enumerate() {
        let h = h as f64;
        c[i + 1] = c[i] + h;
        f[i + 1] = f[i]
            + if h <= 0.0 {
                0.0
            } else if shannon_limit {
                h * h.ln()
            } else {
                h.powf(alpha)
            };
    }
    let n = c[256];
    let entropy = |w: f64, sum: f64| {
        if shannon_limit {
            w.ln() - sum / w
        } else {
            (sum.ln() - alpha * w.ln()) / (1.0 - alpha)
        }
    };
    let mut best = 0usize;
    let mut best_h = f64::NEG_INFINITY;
    for t in 0..256 {
        let w = c[t + 1];
        let w_upper = n - w;
        if w <= 0.0 || w_upper <= 0.0 {
            continue;
        }
        let h = entropy(w, f[t + 1]) + entropy(w_upper, f[256] - f[t + 1]);
        if h > best_h {
            best_h = h;
            best = t;
        }
    }
    best
}

/// Kapur, Sahoo & Wong's (1985) maximum entropy threshold: treats the
/// background (`≤ t`) and foreground (`> t`) as two separate probability
/// distributions and picks the cut whose summed Shannon entropies
/// `H_b(t) + H_f(t)` are largest - the split that keeps the most
/// information in both classes.
fn thresh_max_entropy(hist: &[f32; 256]) -> usize {
    match splittable_range(hist) {
        Ok(_) => max_renyi_entropy_cut(hist, 1.0),
        Err(t) => t,
    }
}

/// Glasbey's (1993) mean threshold: the mean gray level, `floor(Σ i·p_i)`.
fn thresh_mean(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let (c, s) = prefix_sums(hist);
    boundary_bin(s[256] / c[256])
}

/// Minimum threshold (Prewitt & Mendelsohn 1966; Glasbey 1993): the
/// histogram is smoothed until it is bimodal (see `smooth_until_bimodal`)
/// and the threshold is the bin of the smallest smoothed count between the
/// two modes - the bottom of the valley separating them. Returns the lowest
/// occupied bin when no bimodal smoothing exists.
fn thresh_minimum(hist: &[f32; 256]) -> usize {
    let lo = match splittable_range(hist) {
        Ok((lo, _)) => lo,
        Err(t) => return t,
    };
    match smooth_until_bimodal(hist) {
        Some((y, [p1, p2])) => {
            let mut best = p1 + 1;
            for i in p1 + 1..p2 {
                if y[i] < y[best] {
                    best = i;
                }
            }
            best
        }
        None => lo,
    }
}

/// Doyle's (1962) p-tile threshold with `p = 50 %`: the cut whose background
/// share (bins `0..=t`) is as close as possible to `p`, among the cuts that
/// leave both classes non-empty. For a smooth histogram this is the median
/// gray level. Taking the closest share rather than the first bin that
/// reaches `p` matters when one bin holds more than half of the pixels, as
/// in a saturated image: the first bin reaching 50 % is then that bin, which
/// for a spike at the top leaves no foreground at all, while the closest
/// share still separates the spike from the rest.
fn thresh_percentile(hist: &[f32; 256]) -> usize {
    const P: f64 = 0.5;
    let (lo, hi) = match splittable_range(hist) {
        Ok(range) => range,
        Err(t) => return t,
    };
    let (c, _) = prefix_sums(hist);
    let n = c[256];
    let mut best = lo;
    for t in lo..hi {
        if (c[t + 1] / n - P).abs() < (c[best + 1] / n - P).abs() {
            best = t;
        }
    }
    best
}

/// Sahoo, Wilkins & Yeager's (1997) Rényi entropy threshold. Maximizing the
/// summed Rényi entropies of order `α` of the two classes (see
/// `max_renyi_entropy_cut`) gives a cut `t(α)`; the paper evaluates
/// `α = 0.5`, `α → 1` (Kapur's maximum entropy) and `α = 2` (Yen's
/// criterion), sorts the three cuts into `t₁ ≤ t₂ ≤ t₃`, and combines them
/// into one threshold
/// `t* = t₁·(P(t₁) + ¼ωβ₁) + ¼·t₂·ωβ₂ + t₃·(1 − P(t₃) + ¼ωβ₃)`
/// with `P` the cumulative fraction, `ω = P(t₃) − P(t₁)` and the weights
/// `(β₁, β₂, β₃)` = `(0, 1, 3)` if only `t₁, t₂` are within 5 bins of each
/// other, `(3, 1, 0)` if only `t₂, t₃` are, and `(1, 2, 1)` otherwise.
/// The weights sum to 4, so `t*` is a weighted average of the three cuts.
/// Two points the paper leaves open: `P` is summed from bin 0 (as its
/// class probability `p(Λ₁)`), and the real-valued `t*` is rounded to the
/// nearest bin.
fn thresh_renyi_entropy(hist: &[f32; 256]) -> usize {
    if let Err(t) = splittable_range(hist) {
        return t;
    }
    let mut cuts = [
        max_renyi_entropy_cut(hist, 0.5),
        max_renyi_entropy_cut(hist, 1.0),
        max_renyi_entropy_cut(hist, 2.0),
    ];
    cuts.sort_unstable();
    let [t1, t2, t3] = cuts;
    let (c, _) = prefix_sums(hist);
    let p = |t: usize| c[t + 1] / c[256];
    let close12 = t2 - t1 <= 5;
    let close23 = t3 - t2 <= 5;
    let (b1, b2, b3) = match (close12, close23) {
        (true, false) => (0.0, 1.0, 3.0),
        (false, true) => (3.0, 1.0, 0.0),
        _ => (1.0, 2.0, 1.0),
    };
    let omega = p(t3) - p(t1);
    let t = t1 as f64 * (p(t1) + 0.25 * omega * b1)
        + 0.25 * t2 as f64 * omega * b2
        + t3 as f64 * (1.0 - p(t3) + 0.25 * omega * b3);
    t.round().clamp(0.0, 255.0) as usize
}

/// Shanbhag's (1994) information-measure threshold. For a cut `t`, a gray
/// level's membership in its class grows with the share of the class lying
/// between it and the cut: `μ_b(g) = ½ + (P(t) − P(g − 1)) / (2P(t))` for
/// `g ≤ t` and `μ_f(g) = ½ + (P(g) − P(t)) / (2(1 − P(t)))` for `g > t`
/// (`P` cumulative), so a level right at the boundary is barely more than
/// ½ certain and the outermost level is fully certain. Each class' average
/// information `I = −Σ p(g) ln μ(g) / P_class` measures how uncertain its
/// assignment is; the cut chosen is the one where both classes carry the
/// same information, i.e. that minimizes `|I_b(t) − I_f(t)|`. Only cuts
/// leaving both classes non-empty are considered.
fn thresh_shanbhag(hist: &[f32; 256]) -> usize {
    let (lo, hi) = match splittable_range(hist) {
        Ok(r) => r,
        Err(t) => return t,
    };
    let (c, _) = prefix_sums(hist);
    let mut best = lo;
    let mut best_diff = f64::INFINITY;
    for t in lo..hi {
        let w_b = c[t + 1];
        let w_f = c[256] - w_b;
        if w_b <= 0.0 || w_f <= 0.0 {
            continue;
        }
        let mut info_b = 0.0f64;
        for g in lo..=t {
            let h = hist[g] as f64;
            if h > 0.0 {
                info_b -= h * (0.5 + (w_b - c[g]) / (2.0 * w_b)).ln();
            }
        }
        let mut info_f = 0.0f64;
        for g in t + 1..=hi {
            let h = hist[g] as f64;
            if h > 0.0 {
                info_f -= h * (0.5 + (c[g + 1] - w_b) / (2.0 * w_f)).ln();
            }
        }
        let diff = (info_b / w_b - info_f / w_f).abs();
        if diff < best_diff {
            best_diff = diff;
            best = t;
        }
    }
    best
}

/// Yen, Chang & Chang's (1995) maximum correlation threshold. The
/// "correlation" of a distribution, `−ln Σ p_i²`, measures how spread out it
/// is; the cut maximizing the total correlation of the two class
/// distributions, `−ln Σ_{i≤t} (p_i/P_t)² − ln Σ_{i>t} (p_i/(1 − P_t))²`,
/// is chosen. This is the order-2 Rényi entropy of both classes, so it
/// shares `max_renyi_entropy_cut` with Kapur's and Sahoo's methods.
fn thresh_yen(hist: &[f32; 256]) -> usize {
    match splittable_range(hist) {
        Ok(_) => max_renyi_entropy_cut(hist, 2.0),
        Err(t) => t,
    }
}

/// The "Robust Background" threshold: sorts `data`,
/// trims `lower_outlier_fraction`/`upper_outlier_fraction` of the population
/// off each end, then returns the trimmed population's average plus
/// `deviations_above_average` times its spread. `averaging_method` picks
/// *both* halves of that pair at once: `Mean` pairs with the standard
/// deviation, `Median` pairs with the median absolute deviation (the raw
/// MAD, no `1.4826` normal-consistency scaling, so `deviations_above_average`
/// counts plain MADs).
///
/// Deliberately operates on the raw pixel population instead of a
/// pre-binned histogram - see the call site in `Threshold::execute` for why.
fn thresh_robust_background(
    data: &[f32],
    lower_outlier_fraction: f32,
    upper_outlier_fraction: f32,
    averaging_method: Averaging,
    deviations_above_average: f32,
) -> f32 {
    // Mirrors the reference's `if n_pixels < 3: return 0`.
    if data.len() < 3 {
        return 0.0;
    }

    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let n_f = n as f32;

    // "if lower_outlier_fraction * n_pixels < 1: lower_outlier_fraction = 1 /
    // n_pixels" (and the same for the upper fraction) - guarantees at least
    // one pixel is trimmed off each end regardless of how small the
    // requested fraction is, as long as there are enough pixels to trim.
    let lower_fraction = if lower_outlier_fraction * n_f < 1.0 {
        1.0 / n_f
    } else {
        lower_outlier_fraction
    };
    let upper_fraction = if upper_outlier_fraction * n_f < 1.0 {
        1.0 / n_f
    } else {
        upper_outlier_fraction
    };

    let lower_bound = (n_f * lower_fraction) as usize;
    let upper_bound = n - (n_f * upper_fraction) as usize;

    // "if len(trimmed_image) == 0: return cropped_image[0]".
    if upper_bound <= lower_bound {
        return sorted[0];
    }
    let trimmed = &sorted[lower_bound..upper_bound];

    match averaging_method {
        Averaging::Mean => {
            let mean = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
            // Population variance (divide by N, not N-1) - matches numpy's
            // `.std()` default (`ddof=0`), which the reference uses.
            let variance =
                trimmed.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / trimmed.len() as f32;
            mean + variance.sqrt() * deviations_above_average
        }
        Averaging::Median => {
            let median = median_of_sorted(trimmed);
            let mut abs_deviations: Vec<f32> =
                trimmed.iter().map(|&v| (v - median).abs()).collect();
            abs_deviations.sort_by(|a, b| a.total_cmp(b));
            let mad = median_of_sorted(&abs_deviations);
            median + mad * deviations_above_average
        }
    }
}

/// The median of an already-sorted slice - averages the two middle elements
/// for an even-length slice, matching `numpy.median`. Panics on an empty
/// slice; every caller here only ever passes a non-empty trimmed population.
fn median_of_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

// ── Citations of the automatic threshold methods ─────────────────────────────

const OTSU: CitationMetadata = CitationMetadata {
    cite_key: "otsu1979threshold",
    title: "A Threshold Selection Method from Gray-Level Histograms",
    authors: &["Nobuyuki Otsu"],
    year: 1979,
    container: Some("IEEE Transactions on Systems, Man, and Cybernetics"),
    doi: Some("10.1109/TSMC.1979.4310076"),
    url: Some("https://doi.org/10.1109/TSMC.1979.4310076"),
    pages: Some("62-66"),
};

const LI: CitationMetadata = CitationMetadata {
    cite_key: "li1993minimum",
    title: "Minimum cross entropy thresholding",
    authors: &["C. H. Li", "C. K. Lee"],
    year: 1993,
    container: Some("Pattern Recognition"),
    doi: Some("10.1016/0031-3203(93)90115-D"),
    url: Some("https://doi.org/10.1016/0031-3203(93)90115-D"),
    pages: Some("617-625"),
};

const MIN_ERROR: CitationMetadata = CitationMetadata {
    cite_key: "kittler1986minimum",
    title: "Minimum error thresholding",
    authors: &["J. Kittler", "J. Illingworth"],
    year: 1986,
    container: Some("Pattern Recognition"),
    doi: Some("10.1016/0031-3203(86)90030-0"),
    url: Some("https://doi.org/10.1016/0031-3203(86)90030-0"),
    pages: Some("41-47"),
};

const TRIANGLE: CitationMetadata = CitationMetadata {
    cite_key: "zack1977automatic",
    title: "Automatic measurement of sister chromatid exchange frequency",
    authors: &["G. W. Zack", "W. E. Rogers", "S. A. Latt"],
    year: 1977,
    container: Some("Journal of Histochemistry & Cytochemistry"),
    doi: Some("10.1177/25.7.70454"),
    url: Some("https://doi.org/10.1177/25.7.70454"),
    pages: Some("741-753"),
};

const MOMENTS: CitationMetadata = CitationMetadata {
    cite_key: "tsai1985moment",
    title: "Moment-preserving thresholding: A new approach",
    authors: &["Wen-Hsiang Tsai"],
    year: 1985,
    container: Some("Computer Vision, Graphics, and Image Processing"),
    doi: Some("10.1016/0734-189X(85)90133-1"),
    url: Some("https://doi.org/10.1016/0734-189X(85)90133-1"),
    pages: Some("377-393"),
};

const HUANG: CitationMetadata = CitationMetadata {
    cite_key: "huang1995image",
    title: "Image thresholding by minimizing the measures of fuzziness",
    authors: &["Liang-Kai Huang", "Mao-Jiun J. Wang"],
    year: 1995,
    container: Some("Pattern Recognition"),
    doi: Some("10.1016/0031-3203(94)E0043-K"),
    url: Some("https://doi.org/10.1016/0031-3203(94)E0043-K"),
    pages: Some("41-51"),
};

const PREWITT: CitationMetadata = CitationMetadata {
    cite_key: "prewitt1966analysis",
    title: "The Analysis of Cell Images",
    authors: &["Judith M. S. Prewitt", "Mortimer L. Mendelsohn"],
    year: 1966,
    container: Some("Annals of the New York Academy of Sciences"),
    doi: Some("10.1111/j.1749-6632.1965.tb11715.x"),
    url: Some("https://doi.org/10.1111/j.1749-6632.1965.tb11715.x"),
    pages: Some("1035-1053"),
};

const ISODATA: CitationMetadata = CitationMetadata {
    cite_key: "ridler1978picture",
    title: "Picture Thresholding Using an Iterative Selection Method",
    authors: &["T. W. Ridler", "S. Calvard"],
    year: 1978,
    container: Some("IEEE Transactions on Systems, Man, and Cybernetics"),
    doi: Some("10.1109/TSMC.1978.4310039"),
    url: Some("https://doi.org/10.1109/TSMC.1978.4310039"),
    pages: Some("630-632"),
};

const KAPUR: CitationMetadata = CitationMetadata {
    cite_key: "kapur1985new",
    title: "A new method for gray-level picture thresholding using the entropy of the histogram",
    authors: &["J. N. Kapur", "P. K. Sahoo", "A. K. C. Wong"],
    year: 1985,
    container: Some("Computer Vision, Graphics, and Image Processing"),
    doi: Some("10.1016/0734-189X(85)90125-2"),
    url: Some("https://doi.org/10.1016/0734-189X(85)90125-2"),
    pages: Some("273-285"),
};

const GLASBEY: CitationMetadata = CitationMetadata {
    cite_key: "glasbey1993analysis",
    title: "An Analysis of Histogram-Based Thresholding Algorithms",
    authors: &["C. A. Glasbey"],
    year: 1993,
    container: Some("CVGIP: Graphical Models and Image Processing"),
    doi: Some("10.1006/cgip.1993.1040"),
    url: Some("https://doi.org/10.1006/cgip.1993.1040"),
    pages: Some("532-537"),
};

const DOYLE: CitationMetadata = CitationMetadata {
    cite_key: "doyle1962operations",
    title: "Operations Useful for Similarity-Invariant Pattern Recognition",
    authors: &["W. Doyle"],
    year: 1962,
    container: Some("Journal of the ACM"),
    doi: Some("10.1145/321119.321123"),
    url: Some("https://doi.org/10.1145/321119.321123"),
    pages: Some("259-267"),
};

const SHANBHAG: CitationMetadata = CitationMetadata {
    cite_key: "shanbhag1994utilization",
    title: "Utilization of Information Measure as a Means of Image Thresholding",
    authors: &["Abhijit G. Shanbhag"],
    year: 1994,
    container: Some("CVGIP: Graphical Models and Image Processing"),
    doi: Some("10.1006/cgip.1994.1037"),
    url: Some("https://doi.org/10.1006/cgip.1994.1037"),
    pages: Some("414-419"),
};

const YEN: CitationMetadata = CitationMetadata {
    cite_key: "yen1995new",
    title: "A new criterion for automatic multilevel thresholding",
    authors: &["Jui-Cheng Yen", "Fu-Juay Chang", "Shyang Chang"],
    year: 1995,
    container: Some("IEEE Transactions on Image Processing"),
    doi: Some("10.1109/83.366472"),
    url: Some("https://doi.org/10.1109/83.366472"),
    pages: Some("370-378"),
};

const LI_TAM: CitationMetadata = CitationMetadata {
    cite_key: "li1998iterative",
    title: "An iterative algorithm for minimum cross entropy thresholding",
    authors: &["C. H. Li", "P. K. S. Tam"],
    year: 1998,
    container: Some("Pattern Recognition Letters"),
    doi: Some("10.1016/S0167-8655(98)00057-9"),
    url: Some("https://doi.org/10.1016/S0167-8655(98)00057-9"),
    pages: Some("771-776"),
};

const SAHOO_RENYI: CitationMetadata = CitationMetadata {
    cite_key: "sahoo1997threshold",
    title: "Threshold selection using Renyi's entropy",
    authors: &["Prasanna Sahoo", "Carrye Wilkins", "Jerry Yeager"],
    year: 1997,
    container: Some("Pattern Recognition"),
    doi: Some("10.1016/S0031-3203(96)00065-9"),
    url: Some("https://doi.org/10.1016/S0031-3203(96)00065-9"),
    pages: Some("71-84"),
};

/// The papers an automatic threshold method is implemented from (the
/// method's original paper first); empty for manual ones.
fn method_citations(method: &ThresholdMethod) -> &'static [&'static CitationMetadata] {
    match method {
        ThresholdMethod::None | ThresholdMethod::Manual => &[],
        ThresholdMethod::Li => &[&LI, &LI_TAM],
        ThresholdMethod::MinError => &[&MIN_ERROR, &GLASBEY],
        ThresholdMethod::Triangle => &[&TRIANGLE],
        ThresholdMethod::Moments => &[&MOMENTS, &GLASBEY],
        ThresholdMethod::Huang => &[&HUANG],
        ThresholdMethod::Intermodes => &[&PREWITT],
        ThresholdMethod::Minimum => &[&PREWITT, &GLASBEY],
        ThresholdMethod::IsoData => &[&ISODATA],
        ThresholdMethod::MaxEntropy => &[&KAPUR],
        ThresholdMethod::RenyiEntropy => &[&SAHOO_RENYI],
        ThresholdMethod::Mean => &[&GLASBEY],
        ThresholdMethod::Otsu { .. } => &[&OTSU],
        ThresholdMethod::Percentile => &[&DOYLE],
        ThresholdMethod::Shanbhag => &[&SHANBHAG],
        ThresholdMethod::Yen => &[&YEN],
        ThresholdMethod::RobustBackground { .. } => &[&CitationMetadata::CELLPROFILER],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every automatic method (all methods using `compute_auto_threshold`).
    fn automatic_methods() -> Vec<ThresholdMethod> {
        vec![
            ThresholdMethod::Li,
            ThresholdMethod::MinError,
            ThresholdMethod::Triangle,
            ThresholdMethod::Moments,
            ThresholdMethod::Huang,
            ThresholdMethod::Intermodes,
            ThresholdMethod::IsoData,
            ThresholdMethod::MaxEntropy,
            ThresholdMethod::Mean,
            ThresholdMethod::Minimum,
            ThresholdMethod::Otsu {
                classes: OtsuClasses::Two,
            },
            ThresholdMethod::Otsu {
                classes: OtsuClasses::Three {
                    middle_class: OtsuMiddleClass::Background,
                },
            },
            ThresholdMethod::Otsu {
                classes: OtsuClasses::Three {
                    middle_class: OtsuMiddleClass::Foreground,
                },
            },
            ThresholdMethod::Percentile,
            ThresholdMethod::RenyiEntropy,
            ThresholdMethod::Shanbhag,
            ThresholdMethod::Yen,
        ]
    }

    #[test]
    fn every_automatic_method_copes_with_an_empty_histogram() {
        // E.g. a tile without any pixel in range: no panic, no division by
        // zero turning into an arbitrary threshold.
        let empty = [0f32; 256];
        for method in automatic_methods() {
            let t = compute_auto_threshold(&method, &empty);
            assert!(t <= 255, "{method:?}: {t}");
        }
    }

    #[test]
    fn every_automatic_method_copes_with_a_single_intensity() {
        // A flat tile: all pixels in one bin.
        let mut flat = [0f32; 256];
        flat[100] = 1000.0;
        for method in automatic_methods() {
            let t = compute_auto_threshold(&method, &flat);
            assert!(t <= 255, "{method:?}: {t}");
        }
    }

    #[test]
    fn manual_and_none_need_no_histogram_threshold() {
        let mut hist = [0f32; 256];
        hist[10] = 5.0;
        assert_eq!(compute_auto_threshold(&ThresholdMethod::None, &hist), 0);
        assert_eq!(compute_auto_threshold(&ThresholdMethod::Manual, &hist), 0);
    }

    #[test]
    fn every_method_cites_its_paper() {
        let robust = ThresholdMethod::RobustBackground {
            lower_outlier_fraction: 0.05,
            upper_outlier_fraction: 0.05,
            averaging_method: Averaging::Mean,
            deviations_above_average: 2.0,
        };
        let keys = |m: &ThresholdMethod| -> Vec<&str> {
            method_citations(m).iter().map(|c| c.cite_key).collect()
        };
        assert!(keys(&ThresholdMethod::None).is_empty());
        assert!(keys(&ThresholdMethod::Manual).is_empty());
        assert_eq!(keys(&robust), ["mcquin2018cellprofiler"]);
        let expected: [&[&str]; 17] = [
            &["li1993minimum", "li1998iterative"],
            &["kittler1986minimum", "glasbey1993analysis"],
            &["zack1977automatic"],
            &["tsai1985moment", "glasbey1993analysis"],
            &["huang1995image"],
            &["prewitt1966analysis"],
            &["ridler1978picture"],
            &["kapur1985new"],
            &["glasbey1993analysis"],
            &["prewitt1966analysis", "glasbey1993analysis"],
            &["otsu1979threshold"],
            &["otsu1979threshold"],
            &["otsu1979threshold"],
            &["doyle1962operations"],
            &["sahoo1997threshold"],
            &["shanbhag1994utilization"],
            &["yen1995new"],
        ];
        let got: Vec<Vec<&str>> = automatic_methods().iter().map(keys).collect();
        assert_eq!(got, expected);
    }

    fn threshold_with(methods: &[ThresholdMethod]) -> Threshold {
        Threshold {
            thresholds: methods
                .iter()
                .map(|&method| ThresholdEntry {
                    method,
                    min_threshold: 0.0,
                    max_threshold: 255.0,
                    unit: PixelUnits::Bit,
                    segmentation_class: SegmentationClass(1),
                    value_source: ThresholdValueSource::ActualImage,
                })
                .collect(),
        }
    }

    fn cite_keys(methods: &[ThresholdMethod]) -> Vec<&'static str> {
        threshold_with(methods)
            .cite()
            .iter()
            .map(|c| c.cite_key)
            .collect()
    }

    #[test]
    fn cite_names_the_paper_of_every_automatic_method_once() {
        assert_eq!(
            cite_keys(&[ThresholdMethod::Li]),
            ["li1993minimum", "li1998iterative"]
        );
        assert_eq!(
            cite_keys(&[
                ThresholdMethod::Manual,
                ThresholdMethod::Triangle,
                ThresholdMethod::Li,
                ThresholdMethod::Triangle,
            ]),
            ["zack1977automatic", "li1993minimum", "li1998iterative"]
        );
        // Intermodes and Minimum share Prewitt & Mendelsohn: cited once.
        assert_eq!(
            cite_keys(&[ThresholdMethod::Intermodes, ThresholdMethod::Minimum]),
            ["prewitt1966analysis", "glasbey1993analysis"]
        );
    }

    #[test]
    fn cite_falls_back_to_evanalyzer_for_manual_thresholds() {
        assert_eq!(
            cite_keys(&[ThresholdMethod::Manual, ThresholdMethod::None]),
            ["danmayr2026"]
        );
        assert_eq!(cite_keys(&[]), ["danmayr2026"]);
    }

    use crate::{
        image::{ImageContainer, ImageDebugExt},
        pipeline::pipeline_cache::CacheAddress,
    };
    use kornia_image::{Image, ImageSize};
    use std::sync::Arc;

    #[test]
    fn test_multi_range_thresholding() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 3,
            height: 2,
        };
        let input_data = vec![0.1, 0.5, 0.9, 0.0, 1.0, 0.4];
        let input_img = Image::<f32, 1>::new(size, input_data)?;
        input_img.print_window();

        let settings = vec![
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.0,
                max_threshold: 0.2,
                segmentation_class: SegmentationClass(10),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.3,
                max_threshold: 0.6,
                segmentation_class: SegmentationClass(20),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            },
            ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: 0.8,
                max_threshold: 1.0,
                segmentation_class: SegmentationClass(30),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            },
        ];

        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;
        ctx.get_segmentation_map()?.print_window();

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();

        // Pixel index 3 (value 0.0) sits exactly on range 1's lower bound
        // (min_threshold 0.0) and is background, not class 10: the reference
        // treats the lower bound as exclusive and the upper bound as
        // inclusive.
        let expected = vec![10, 20, 30, 0, 30, 20];
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

        let input_img = Image::<f32, 1>::new(size, input_data)?;
        let settings = vec![ThresholdEntry {
            method: ThresholdMethod::Li,
            min_threshold: 0.0,
            max_threshold: 1.0,
            segmentation_class: SegmentationClass(1),
            unit: PixelUnits::Relative,
            value_source: ThresholdValueSource::ActualImage,
        }];

        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();
        let foreground_fraction =
            result_pixels.iter().filter(|&&v| v == 1).count() as f32 / result_pixels.len() as f32;

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
        // Every smoothing pass must be computed from the previous pass'
        // values only (a Jacobi update): writing `h[i]` while still reading
        // `h[i-1]` from the *same* pass lets later bins pick up
        // already-smoothed neighbours. This only shows up once more than one
        // smoothing pass is needed to reach a bimodal histogram - a noisy,
        // multi-peak histogram like this one.
        //
        // 64 is ImageJ's Intermodes result on this histogram (a black-box
        // reference value); an in-place (Gauss-Seidel) update returns 63.
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
    fn test_all_auto_methods_narrow_dynamic_range_regression()
    -> Result<(), Box<dyn std::error::Error>> {
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
            (
                "Otsu",
                ThresholdMethod::Otsu {
                    classes: OtsuClasses::Two,
                },
            ),
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

            let input_img = Image::<f32, 1>::new(size, input_data)?;
            let settings = vec![ThresholdEntry {
                method,
                min_threshold: 0.0,
                max_threshold: 1.0,
                segmentation_class: SegmentationClass(1),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            }];
            let cmd = Threshold {
                thresholds: settings,
            };
            let mut ctx = PipelineContext::new_from_image_test(input_img)?;
            let mut cache = GlobalPipelineCache::default();
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
    /// valley. Unless a test says otherwise, the expected values are
    /// ImageJ's AutoThresholder results on this same histogram, used as
    /// black-box reference values, so a match means the method agrees with
    /// that widely used implementation, not just "returns something in
    /// range". Where the published definition legitimately lands on a
    /// different bin, the test asserts the paper's result and explains the
    /// difference.
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

    /// Every cut inside the empty gap between the clusters has the same
    /// between-class variance; breaking the tie towards the lowest bin parks
    /// the cut at the *end of the first cluster* (bin 40, the last bin of the
    /// 20-40 population), which is also ImageJ's result here. A "last tie
    /// wins" scan would return 159 instead - see `thresh_otsu`'s doc comment
    /// for why that matters beyond just this one number.
    #[test]
    fn test_otsu_matches_reference() {
        assert_eq!(thresh_otsu(&bimodal_reference_histogram()), 40);
    }

    /// Three well-separated populations (bins 20-40, 110-130, 200-220), for
    /// exercising three-class Otsu. Deliberately *not* built by adding a
    /// third cluster on top of [`bimodal_reference_histogram`]'s two - its
    /// second cluster (160-200) would directly adjoin a cluster added at
    /// 200-220, merging them into one contiguous block with no valley
    /// between them.
    fn trimodal_reference_histogram() -> [f32; 256] {
        let mut h = [0.0f32; 256];
        for i in 20..=40 {
            h[i] = 50.0;
        }
        for i in 110..=130 {
            h[i] = 50.0;
        }
        for i in 200..=220 {
            h[i] = 50.0;
        }
        h
    }

    /// `ThresholdMethod::Otsu { classes: OtsuClasses::Two }` must still
    /// dispatch to the exact same `thresh_otsu` the plain two-class case
    /// always used, on both a two- and a three-population histogram -
    /// wiring the new `classes` field in must not change Otsu's original
    /// (two-class) behavior at all.
    #[test]
    fn test_otsu_two_class_dispatch_matches_original_thresh_otsu() {
        let two_class = ThresholdMethod::Otsu {
            classes: OtsuClasses::Two,
        };
        for hist in [
            bimodal_reference_histogram(),
            trimodal_reference_histogram(),
        ] {
            assert_eq!(
                compute_auto_threshold(&two_class, &hist),
                thresh_otsu(&hist),
                "OtsuClasses::Two must reproduce plain thresh_otsu exactly"
            );
        }
    }

    /// Three well-separated, equal-weight populations (bins 20-40, 110-130,
    /// 200-220 - built by adding a third cluster on top of
    /// `bimodal_reference_histogram`'s two) give an unambiguous ground truth
    /// for the joint two-cut search: Otsu's between-class variance is flat
    /// across any of the empty gaps separating the clusters, so the true
    /// optimum sits anywhere each cluster ends - `thresh_otsu_multi`'s
    /// tie-breaking (first cut reached, via strict `>`) should land exactly
    /// on the end of the first cluster (40) and the end of the second (130).
    /// Cross-checked with an independent Python re-implementation of the same
    /// prefix-sum/variance formula.
    #[test]
    fn test_otsu_multi_matches_reference() {
        assert_eq!(
            thresh_otsu_multi(&trimodal_reference_histogram()),
            (40, 130)
        );
    }

    /// End-to-end regression for three-class thresholding with the middle
    /// intensity class assigned to the foreground or the background: with three roughly equal-sized populations, setting
    /// `middle_class` to `Background` must classify only the *brightest*
    /// population as foreground (~1/3 of pixels), while `Foreground` must
    /// classify the middle *and* brightest populations as foreground (~2/3) -
    /// because the middle-class assignment changes
    /// which of the two jointly-optimized cuts becomes the final threshold.
    #[test]
    fn test_otsu_three_class_middle_assignment() -> Result<(), Box<dyn std::error::Error>> {
        // Three equal-sized, well-separated populations: dark, mid, bright.
        let per_cluster = 100;
        let mut input_data = Vec::with_capacity(per_cluster * 3);
        for _ in 0..per_cluster {
            input_data.push(30.0 / 255.0);
        }
        for _ in 0..per_cluster {
            input_data.push(120.0 / 255.0);
        }
        for _ in 0..per_cluster {
            input_data.push(210.0 / 255.0);
        }
        let size = ImageSize {
            width: input_data.len(),
            height: 1,
        };

        let run = |middle_class: OtsuMiddleClass| -> Result<f32, Box<dyn std::error::Error>> {
            let input_img = Image::<f32, 1>::new(size, input_data.clone())?;
            let settings = vec![ThresholdEntry {
                method: ThresholdMethod::Otsu {
                    classes: OtsuClasses::Three { middle_class },
                },
                min_threshold: 0.0,
                max_threshold: 1.0,
                segmentation_class: SegmentationClass(1),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            }];
            let cmd = Threshold {
                thresholds: settings,
            };
            let mut ctx = PipelineContext::new_from_image_test(input_img)?;
            let mut cache = GlobalPipelineCache::default();
            cmd.execute(&mut ctx, &mut cache)?;
            let result_pixels = ctx
                .segmentation_map
                .as_ref()
                .expect("No labels found")
                .as_slice();
            Ok(result_pixels.iter().filter(|&&v| v == 1).count() as f32
                / result_pixels.len() as f32)
        };

        let background_fraction = run(OtsuMiddleClass::Background)?;
        let foreground_fraction = run(OtsuMiddleClass::Foreground)?;

        assert!(
            (0.25..0.40).contains(&background_fraction),
            "middle_class=Background should classify only the brightest ~1/3 as foreground, \
             got {background_fraction:.3}"
        );
        assert!(
            (0.55..0.75).contains(&foreground_fraction),
            "middle_class=Foreground should classify the middle+brightest ~2/3 as foreground, \
             got {foreground_fraction:.3}"
        );
        assert!(
            foreground_fraction > background_fraction,
            "Foreground middle-class assignment must yield more foreground pixels than \
             Background: foreground={foreground_fraction:.3}, background={background_fraction:.3}"
        );
        Ok(())
    }

    /// Class means 30 and 180 give the fixed point
    /// `t = 150 / ln(180 / 30) ≈ 83.72`. Bins `0..=83` lie below that
    /// boundary, so the paper's result is bin 83; ImageJ reports 84 (one bin
    /// higher, i.e. rounded instead of floored).
    #[test]
    fn test_li_matches_reference() {
        assert_eq!(thresh_li(&bimodal_reference_histogram()), 83);
    }

    /// For every cut in the gap the classes are the two uniform blocks
    /// (`μ₁ = 30, σ₁² = 110/3, P₁ = 1050/1870`; `μ₂ = 180, σ₂² = 140`), and
    /// the equal-density quadratic has its root between the means at
    /// `g ≈ 81.2`, so the iteration settles on bin 81. ImageJ reports 80.
    #[test]
    fn test_min_error_matches_reference() {
        assert_eq!(thresh_min_error(&bimodal_reference_histogram()), 81);
    }

    /// Peak at bin 20 (the lowest bin with the maximal count), line to the
    /// last occupied bin 200: the histogram drops furthest below the line at
    /// the first empty bin, 41. ImageJ reports 42, one bin further from the
    /// peak than the maximum-distance bin itself.
    #[test]
    fn test_triangle_matches_reference() {
        assert_eq!(thresh_triangle(&bimodal_reference_histogram()), 41);
    }

    /// ImageJ gives 160, the first bin whose cumulative fraction reaches
    /// `p₀`. Glasbey (1993) defines the cut whose fraction is *nearest* to
    /// `p₀`: every cut in the empty valley 40..159 has the same fraction,
    /// nearer than 160's, and ties go low - the end of the lower cluster,
    /// which splits the two clusters cleanly.
    #[test]
    fn test_moments_matches_reference() {
        assert_eq!(thresh_moments(&bimodal_reference_histogram()), 40);
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
        // Starting at the mean (≈95.8), the class means 30 and 180 give
        // `t = 105` exactly, a fixed point. ImageJ's IsoData variant agrees
        // here but can differ by about one bin on other histograms.
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

    /// More than half of the pixels in the top bin (a saturated image): no
    /// cut reaches exactly 50 %, the closest leaves the saturated spike as
    /// foreground - not the top bin itself, which would leave nothing above
    /// it. Every cut from 99 to 254 gives 30 % background; ties go low.
    #[test]
    fn test_percentile_keeps_a_saturated_spike_as_foreground() {
        let mut h = [0.0f32; 256];
        for bin in h.iter_mut().take(100) {
            *bin = 3.0;
        }
        h[255] = 700.0;
        assert_eq!(thresh_percentile(&h), 99);
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

    /// Bit-exact regression for `Averaging::Mean` against the reference
    /// implementation `centrosome.threshold.get_robust_background_threshold`:
    ///
    /// ```python
    /// def get_robust_background_threshold(data, lower_frac, upper_frac, k):
    ///     n = len(data)
    ///     s = sorted(data)
    ///     if lower_frac * n < 1: lower_frac = 1.0 / n
    ///     if upper_frac * n < 1: upper_frac = 1.0 / n
    ///     lb = int(n * lower_frac)
    ///     ub = n - int(n * upper_frac)
    ///     trimmed = s[lb:ub]
    ///     avg = sum(trimmed) / len(trimmed)
    ///     sd = (sum((v - avg) ** 2 for v in trimmed) / len(trimmed)) ** 0.5
    ///     return avg + sd * k
    /// ```
    ///
    /// 20 pixels: one low outlier (-100), nine 0s, nine 10s, one high
    /// outlier (200). `lower_outlier_fraction`/`upper_outlier_fraction` of
    /// 0.05 trim exactly the two outliers (`int(20 * 0.05) == 1` off each
    /// end), leaving the symmetric nine-0/nine-10 population: mean = 5.0,
    /// population variance = ((0-5)² · 9 + (10-5)² · 9) / 18 = 25.0, so the
    /// standard deviation is exactly 5.0 - no irrational intermediate value,
    /// so the `sqrt` and every other step lands on an exactly-representable
    /// `f32` regardless of evaluation order, giving a genuinely bit-exact
    /// expected result (`5.0 + 5.0 * 2.0 == 15.0`) rather than one that only
    /// matches within some epsilon.
    #[test]
    fn test_robust_background_mean_matches_reference() {
        let mut data = vec![-100.0f32];
        data.extend(std::iter::repeat_n(0.0f32, 9));
        data.extend(std::iter::repeat_n(10.0f32, 9));
        data.push(200.0);

        let result = thresh_robust_background(&data, 0.05, 0.05, Averaging::Mean, 2.0);

        assert_eq!(result, 15.0, "must match the reference exactly");
    }

    /// Bit-exact regression for `Averaging::Median`, same reference formula
    /// as above but with `avg = median(trimmed)` and `sd =
    /// median(abs(trimmed - avg))` (the raw MAD, no `1.4826`
    /// normal-consistency scaling).
    ///
    /// 20 pixels, values 1..=20. `int(20 * 0.05) == 1` trims the single
    /// lowest (1) and single highest (20) value, leaving 2..=19 (18 values).
    /// Every step here - the median of an 18-element run of consecutive
    /// integers, the median of the resulting symmetric `.5`-valued absolute
    /// deviations - only ever averages two values that are exactly
    /// representable in `f32` (halves of small integers), so the expected
    /// result (`10.5 + 4.5 * 2.0 == 19.5`) is exact independent of
    /// evaluation order.
    ///
    /// Cross-checked against an independent Python re-implementation of the
    /// reference formula on this exact input, which likewise prints `19.5`.
    #[test]
    fn test_robust_background_median_matches_reference() {
        let data: Vec<f32> = (1..=20).map(|v| v as f32).collect();

        let result = thresh_robust_background(&data, 0.05, 0.05, Averaging::Median, 2.0);

        assert_eq!(result, 19.5, "must match the reference exactly");
    }

    /// Fewer than 3 pixels short-circuits to `0.0`, matching the reference's
    /// `if n_pixels < 3: return 0`.
    #[test]
    fn test_robust_background_too_few_pixels_returns_zero() {
        assert_eq!(
            thresh_robust_background(&[0.1, 0.9], 0.05, 0.05, Averaging::Mean, 2.0),
            0.0
        );
    }

    /// End-to-end regression through `Threshold::execute`: a
    /// `RobustBackground` entry must compute its cut-off from the exact
    /// pixel population (not the shared 256-bin histogram every other
    /// method uses) and apply it with the same
    /// exclusive-lower/inclusive-upper comparison as the rest of `Threshold`.
    /// Reuses `test_robust_background_mean_matches_reference`'s
    /// dataset (threshold value 15.0) scaled into a tiny image so pixels
    /// above 15.0 land in one class and everything else in another.
    #[test]
    fn test_threshold_execute_robust_background_end_to_end()
    -> Result<(), Box<dyn std::error::Error>> {
        // Same 20-value population as the direct unit test above (threshold
        // = 15.0). `PixelUnits::Relative` passes `min_threshold`/
        // `max_threshold` through unchanged (see `PixelUnits::to_relative`),
        // so they can be given in the same arbitrary units as this data
        // without any bit-depth rescaling to account for.
        let mut input_data = vec![-100.0f32];
        input_data.extend(std::iter::repeat_n(0.0f32, 9));
        input_data.extend(std::iter::repeat_n(10.0f32, 9));
        input_data.push(200.0);

        let size = ImageSize {
            width: input_data.len(),
            height: 1,
        };
        let input_img = Image::<f32, 1>::new(size, input_data)?;

        let settings = vec![ThresholdEntry {
            method: ThresholdMethod::RobustBackground {
                lower_outlier_fraction: 0.05,
                upper_outlier_fraction: 0.05,
                averaging_method: Averaging::Mean,
                deviations_above_average: 2.0,
            },
            min_threshold: 0.0,
            max_threshold: 255.0,
            segmentation_class: SegmentationClass(1),
            unit: PixelUnits::Relative,
            value_source: ThresholdValueSource::ActualImage,
        }];
        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();

        // Only the trailing 200.0 pixel (> 15.0) is foreground; everything
        // else (-100, the nine 0s, the nine 10s) is <= 15.0.
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(result_pixels, &expected[..]);
        Ok(())
    }

    /// Regression test for the "+1 lands on `dmax`" bug, originally reported
    /// as a real Otsu segmentation step finding 0 spots in every frame of
    /// two ground-truth images, despite the same pipeline correctly finding
    /// spots on noisier images whose histograms spread across enough bins
    /// to avoid the tie that triggers this: a cleanly bimodal image
    /// (background pixels all exactly at one value, foreground pixels all
    /// exactly at another, much higher value - e.g. a synthetic spot
    /// channel with no noise) ties every bin between the two populations
    /// for the same between-class variance.
    ///
    /// Now covers both layers of the fix: `thresh_otsu` itself (breaking
    /// ties towards the lowest bin - see its doc comment) resolves
    /// this particular histogram's tie at bin 0 rather than drifting toward
    /// the top of the range, and `Threshold::execute`'s `.min(254)` clamp
    /// remains a backstop for every *other* auto method that can still
    /// return a bin whose "+1" rescale would otherwise land exactly on
    /// `dmax` (`255.0 / 255.0 == 1.0`) - unreachable by any real pixel
    /// under the strict `>` classification below.
    #[test]
    fn test_threshold_execute_otsu_clean_bimodal_image_still_finds_the_bright_population()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut input_data = vec![0.0f32; 90];
        input_data.extend(std::iter::repeat_n(512.0f32, 10));

        let size = ImageSize {
            width: input_data.len(),
            height: 1,
        };
        let input_img = Image::<f32, 1>::new(size, input_data)?;

        let settings = vec![ThresholdEntry {
            method: ThresholdMethod::Otsu {
                classes: OtsuClasses::Two,
            },
            min_threshold: 0.0,
            max_threshold: 1000.0,
            segmentation_class: SegmentationClass(1),
            unit: PixelUnits::Relative,
            value_source: ThresholdValueSource::ActualImage,
        }];
        let cmd = Threshold {
            thresholds: settings,
        };
        let mut ctx = PipelineContext::new_from_image_test(input_img)?;
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;

        let result_pixels = ctx
            .segmentation_map
            .as_ref()
            .expect("No labels found")
            .as_slice();

        let foreground_count = result_pixels.iter().filter(|&&id| id == 1).count();
        assert_eq!(
            foreground_count, 10,
            "every one of the 10 bright pixels must be classified as foreground - \
             before the dmax clamp fix this was 0"
        );
        assert!(
            result_pixels[..90].iter().all(|&id| id == 0),
            "the 90 background pixels must stay background"
        );
        Ok(())
    }

    /// Golden-data regression for the `Threshold::execute` pipeline
    /// (histogram binning, the auto-threshold "+1" bin offset, and the
    /// lower-bound-exclusive/upper-bound-inclusive comparison), not just
    /// the individual `thresh_XXX` bin-selection algorithms above.
    ///
    /// Before this fix, `Threshold::execute` was missing ImageJ's
    /// `scaleAndSetThreshold(0, calcThresholdValue(...) + 1 + cValue, ...)`
    /// "+1" bin offset and used an inclusive lower bound (`>=`) instead of
    /// an exclusive one (strict `>`),
    /// making the threshold systematically one bin too permissive -
    /// letting background-adjacent noise through as foreground.
    ///
    /// `RAW` and the C++ numbers quoted below are from a standalone harness
    /// built directly against `autoThreshold` / `scaleAndSetThreshold` /
    /// `ThresholdTriangle::calcThresholdValue` in that file, run on this
    /// exact data. Its triangle step (taken from ImageJ) picks bin 11; the
    /// paper's maximum-distance bin used by `thresh_triangle` is bin 10 (see
    /// `test_triangle_matches_reference` for the same one-bin offset), so
    /// the binning/"+1"/comparison layers are checked against the C++
    /// pipeline with that one bin accounted for.
    #[test]
    fn test_threshold_execute_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
        const N: usize = 2000;
        #[rustfmt::skip]
        const RAW: &[u32] = &[
            31, 3, 3, 735, 42, 24, 925, 604, 44, 11, 0, 12, 17, 21, 11, 20, 10, 22, 196, 14,
            697, 15, 429, 0, 477, 11, 802, 8, 20, 41, 2, 458, 6, 733, 9, 28, 10, 0, 26, 910,
            23, 14, 35, 15, 0, 11, 868, 34, 217, 18, 30, 8, 8, 21, 8, 8, 20, 27, 760, 88,
            29, 4, 40, 21, 12, 12, 12, 18, 346, 22, 10, 28, 31, 11, 21, 17, 18, 426, 396, 541,
            387, 19, 10, 15, 10, 1350, 26, 399, 402, 24, 1139, 730, 31, 1131, 308, 19, 14, 8, 14, 11,
            37, 1, 3, 1381, 24, 1037, 254, 1115, 28, 580, 32, 43, 134, 6, 13, 43, 722, 15, 24, 628,
            729, 10, 318, 30, 17, 45, 366, 663, 1045, 21, 22, 17, 469, 22, 19, 460, 45, 15, 14, 619,
            54, 22, 10, 23, 1379, 14, 0, 0, 477, 12, 20, 24, 187, 25, 401, 26, 34, 0, 4, 15,
            28, 21, 24, 22, 23, 262, 0, 19, 840, 1059, 539, 25, 428, 44, 856, 651, 1079, 283, 739, 31,
            486, 12, 735, 5, 311, 23, 37, 22, 30, 24, 18, 42, 13, 14, 2, 19, 574, 11, 21, 32,
            809, 15, 32, 756, 22, 0, 768, 364, 110, 269, 431, 16, 1, 11, 543, 9, 19, 514, 436, 429,
            932, 997, 500, 290, 200, 34, 17, 738, 897, 31, 17, 817, 24, 31, 2, 5, 10, 17, 11, 114,
            265, 501, 333, 14, 22, 459, 39, 489, 4, 19, 10, 31, 10, 23, 458, 7, 20, 21, 36, 4,
            47, 25, 855, 19, 8, 17, 9, 737, 17, 0, 19, 2, 5, 29, 717, 31, 380, 13, 864, 9,
            264, 21, 19, 7, 954, 644, 225, 30, 26, 44, 27, 7, 19, 418, 694, 11, 13, 530, 497, 786,
            641, 537, 14, 891, 755, 35, 0, 830, 18, 34, 17, 322, 16, 1, 8, 28, 35, 737, 38, 701,
            391, 420, 18, 14, 14, 25, 215, 23, 405, 952, 1113, 753, 8, 864, 34, 6, 767, 616, 15, 24,
            666, 18, 831, 22, 36, 673, 850, 5, 36, 16, 34, 13, 6, 14, 13, 704, 10, 195, 603, 371,
            40, 311, 38, 31, 23, 19, 28, 816, 961, 22, 667, 21, 538, 746, 757, 11, 9, 16, 982, 15,
            0, 883, 488, 245, 554, 26, 29, 26, 26, 4, 13, 812, 13, 572, 17, 700, 43, 242, 1205, 293,
            4, 541, 455, 0, 411, 10, 421, 18, 0, 9, 19, 16, 19, 25, 15, 19, 484, 454, 4, 1028,
            32, 344, 198, 28, 743, 139, 591, 801, 254, 18, 713, 4, 411, 22, 751, 20, 584, 0, 6, 15,
            894, 16, 15, 33, 27, 18, 566, 4, 43, 352, 1117, 9, 23, 322, 13, 31, 18, 21, 30, 7,
            582, 959, 124, 28, 0, 643, 41, 42, 19, 942, 16, 0, 35, 19, 49, 9, 4, 28, 17, 937,
            702, 2, 15, 11, 1038, 35, 26, 15, 48, 0, 31, 10, 419, 31, 223, 28, 7, 518, 632, 838,
            25, 25, 34, 16, 11, 725, 22, 31, 14, 483, 498, 19, 264, 166, 9, 20, 19, 25, 179, 12,
            0, 9, 30, 21, 22, 1047, 8, 11, 25, 775, 716, 630, 16, 28, 27, 832, 20, 38, 7, 46,
            235, 2, 225, 30, 6, 14, 345, 8, 23, 27, 19, 29, 15, 23, 238, 496, 570, 17, 970, 42,
            721, 1163, 25, 485, 14, 22, 9, 1, 45, 12, 0, 10, 344, 417, 16, 17, 19, 19, 29, 30,
            46, 532, 695, 396, 9, 15, 825, 469, 0, 0, 9, 24, 166, 26, 36, 754, 11, 9, 662, 590,
            23, 439, 3, 1281, 271, 23, 567, 12, 0, 302, 15, 24, 28, 1066, 558, 769, 655, 19, 17, 0,
            41, 618, 456, 10, 1300, 7, 7, 573, 526, 140, 379, 548, 12, 10, 0, 24, 29, 17, 0, 106,
            7, 17, 25, 30, 24, 9, 31, 2, 514, 25, 17, 33, 19, 798, 24, 901, 8, 16, 38, 29,
            263, 32, 15, 18, 16, 1030, 24, 418, 14, 571, 10, 656, 4, 6, 24, 369, 503, 498, 20, 28,
            35, 337, 25, 525, 26, 20, 352, 29, 8, 257, 39, 0, 19, 734, 7, 15, 26, 265, 975, 30,
            0, 30, 5, 19, 44, 10, 31, 414, 35, 40, 0, 593, 41, 576, 0, 36, 0, 11, 11, 14,
            25, 405, 649, 293, 469, 17, 28, 13, 9, 7, 27, 9, 777, 20, 25, 15, 0, 569, 535, 17,
            24, 386, 5, 12, 28, 16, 18, 10, 0, 22, 1151, 16, 41, 9, 6, 35, 19, 0, 7, 0,
            648, 25, 26, 26, 30, 536, 731, 47, 824, 21, 20, 474, 11, 18, 11, 816, 27, 36, 295, 26,
            33, 857, 171, 27, 3, 824, 790, 6, 2, 22, 277, 704, 218, 540, 19, 30, 0, 26, 11, 0,
            10, 28, 20, 562, 581, 21, 25, 0, 46, 44, 8, 0, 38, 15, 208, 19, 7, 27, 778, 689,
            7, 8, 21, 766, 611, 588, 2, 20, 0, 11, 7, 10, 19, 21, 5, 618, 596, 0, 20, 25,
            38, 1093, 24, 602, 636, 20, 21, 20, 15, 34, 26, 44, 9, 23, 26, 993, 31, 43, 721, 482,
            29, 22, 0, 788, 25, 38, 679, 12, 11, 24, 23, 30, 376, 250, 11, 700, 946, 11, 27, 0,
            1056, 11, 12, 29, 25, 25, 558, 13, 23, 23, 15, 454, 6, 764, 855, 4, 0, 412, 1, 44,
            928, 3, 516, 727, 17, 1010, 20, 43, 17, 673, 735, 20, 25, 15, 0, 18, 291, 516, 5, 22,
            27, 691, 21, 26, 9, 916, 968, 28, 0, 339, 859, 21, 30, 23, 29, 709, 36, 591, 7, 21,
            7, 0, 15, 829, 15, 925, 17, 17, 657, 875, 5, 411, 12, 29, 28, 11, 820, 559, 19, 16,
            11, 0, 18, 20, 19, 12, 16, 0, 466, 11, 489, 25, 972, 489, 18, 20, 1007, 26, 753, 19,
            508, 765, 14, 34, 13, 17, 157, 8, 53, 21, 8, 3, 25, 302, 19, 14, 25, 898, 31, 50,
            37, 9, 28, 38, 1231, 30, 13, 29, 0, 25, 27, 32, 580, 6, 36, 17, 669, 922, 23, 22,
            15, 28, 1, 26, 12, 38, 7, 24, 10, 1111, 7, 5, 6, 23, 21, 28, 20, 1122, 261, 45,
            5, 10, 749, 12, 14, 3, 44, 2, 713, 12, 5, 32, 21, 15, 0, 497, 798, 30, 10, 0,
            37, 46, 890, 30, 9, 1060, 27, 28, 6, 606, 4, 16, 36, 39, 5, 657, 778, 45, 1135, 25,
            800, 0, 35, 892, 9, 5, 8, 36, 12, 17, 37, 22, 16, 558, 917, 615, 31, 243, 671, 4,
            14, 6, 24, 12, 21, 21, 17, 20, 916, 18, 32, 813, 365, 8, 261, 550, 34, 764, 24, 680,
            17, 15, 625, 30, 8, 15, 17, 36, 30, 593, 10, 17, 217, 24, 8, 174, 13, 5, 18, 24,
            1039, 5, 692, 5, 63, 584, 16, 0, 34, 544, 32, 25, 37, 15, 13, 4, 893, 14, 4, 10,
            22, 22, 21, 13, 836, 749, 19, 26, 25, 415, 12, 13, 22, 421, 5, 3, 428, 822, 652, 389,
            27, 12, 7, 0, 9, 11, 12, 1080, 1030, 279, 40, 24, 1000, 37, 503, 7, 20, 35, 9, 284,
            8, 496, 26, 17, 7, 8, 12, 467, 21, 430, 1206, 10, 1035, 9, 485, 41, 990, 22, 943, 535,
            41, 23, 7, 0, 17, 22, 16, 29, 33, 0, 26, 13, 290, 6, 729, 10, 0, 672, 20, 27,
            27, 927, 410, 419, 28, 650, 521, 27, 23, 7, 21, 962, 49, 21, 23, 28, 27, 14, 21, 607,
            715, 21, 20, 44, 33, 366, 1417, 25, 0, 688, 21, 21, 921, 27, 18, 29, 236, 17, 8, 638,
            23, 20, 0, 972, 32, 0, 809, 29, 27, 654, 654, 18, 5, 18, 16, 612, 1136, 650, 8, 12,
            98, 377, 282, 0, 5, 22, 95, 8, 15, 49, 15, 502, 21, 15, 38, 789, 22, 0, 33, 8,
            22, 0, 12, 160, 0, 40, 24, 20, 695, 39, 6, 4, 163, 677, 0, 9, 30, 22, 19, 549,
            30, 543, 955, 22, 633, 31, 3, 16, 723, 1295, 70, 28, 296, 27, 24, 545, 767, 870, 5, 19,
            597, 10, 860, 8, 0, 426, 37, 22, 1135, 1449, 42, 22, 14, 24, 8, 2, 0, 23, 39, 1010,
            504, 679, 12, 856, 382, 23, 50, 12, 35, 534, 15, 0, 28, 17, 475, 0, 795, 29, 715, 21,
            30, 13, 37, 71, 4, 266, 12, 21, 16, 38, 15, 1400, 15, 23, 18, 342, 15, 3, 1028, 108,
            2, 9, 705, 150, 25, 298, 165, 826, 1261, 673, 20, 8, 39, 10, 23, 13, 0, 3, 654, 306,
            24, 25, 1153, 272, 9, 281, 4, 904, 828, 34, 19, 21, 640, 463, 20, 16, 668, 647, 1, 31,
            1288, 2, 811, 929, 866, 39, 25, 2, 1094, 10, 15, 17, 832, 22, 1096, 13, 28, 18, 0, 561,
            18, 14, 555, 886, 25, 16, 22, 42, 13, 8, 29, 23, 25, 42, 49, 0, 20, 469, 22, 32,
            797, 201, 16, 42, 444, 17, 929, 10, 39, 395, 23, 536, 223, 35, 849, 15, 379, 933, 606, 10,
            27, 0, 14, 28, 26, 543, 32, 14, 26, 22, 0, 24, 25, 0, 656, 30, 13, 40, 17, 26,
            18, 854, 10, 20, 514, 25, 1128, 11, 22, 17, 450, 33, 32, 862, 28, 20, 16, 5, 13, 663,
            738, 4, 555, 26, 37, 472, 0, 856, 8, 12, 38, 31, 641, 384, 484, 20, 703, 40, 34, 812,
            513, 29, 19, 28, 34, 20, 25, 701, 23, 2, 18, 11, 32, 20, 652, 515, 13, 23, 0, 438,
            1, 7, 18, 52, 7, 34, 29, 8, 25, 29, 10, 12, 5, 961, 835, 505, 30, 7, 28, 9,
            22, 23, 755, 43, 38, 20, 862, 0, 9, 11, 0, 35, 418, 0, 1036, 37, 26, 740, 16, 12,
            0, 580, 34, 14, 11, 1, 572, 19, 844, 686, 16, 4, 776, 687, 22, 892, 32, 25, 606, 27,
            0, 366, 341, 14, 26, 34, 6, 423, 38, 1019, 5, 25, 698, 598, 818, 346, 15, 25, 18, 28,
            22, 15, 25, 42, 25, 37, 20, 646, 20, 25, 1258, 10, 4, 21, 333, 421, 36, 621, 777, 41,
            980, 37, 19, 955, 46, 913, 31, 672, 697, 773, 10, 0, 186, 5, 635, 749, 623, 3, 26, 33,
            0, 454, 45, 9, 710, 882, 18, 21, 28, 5, 614, 9, 795, 16, 23, 19, 0, 27, 0, 499,
            600, 46, 396, 723, 12, 26, 7, 18, 32, 1026, 1005, 1152, 53, 357, 24, 14, 9, 0, 38, 3,
            18, 29, 22, 16, 542, 9, 11, 30, 21, 19, 74, 20, 1126, 20, 764, 140, 29, 6, 41, 33,
            35, 35, 24, 242, 344, 717, 39, 953, 0, 0, 15, 12, 13, 32, 28, 427, 27, 27, 11, 4,
            466, 15, 14, 12, 496, 5, 16, 20, 104, 590, 44, 46, 37, 784, 27, 37, 21, 34, 24, 27,
            37, 500, 34, 22, 19, 467, 17, 18, 1014, 24, 43, 13, 18, 745, 530, 7, 20, 1263, 655, 845,
            2, 13, 22, 637, 453, 14, 1030, 583, 30, 3, 836, 20, 513, 715, 934, 32, 41, 393, 380, 27,
            12, 19, 0, 19, 549, 753, 12, 6, 27, 56, 37, 254, 14, 43, 170, 746, 29, 0, 601, 775,
            536, 21, 26, 23, 22, 11, 18, 767, 11, 3, 543, 476, 22, 14, 7, 52, 11, 32, 499, 1265,
            30, 0, 396, 782, 51, 25, 8, 468, 1011, 663, 31, 31, 11, 20, 230, 22, 11, 21, 6, 706,
            25, 488, 1127, 2, 11, 452, 675, 851, 37, 29, 34, 755, 6, 35, 28, 1002, 13, 36, 749, 26,
            31, 12, 19, 4, 376, 23, 64, 40, 437, 22, 14, 10, 4, 350, 9, 38, 1113, 28, 605, 20,
            40, 32, 32, 597, 255, 1042, 12, 849, 21, 625, 21, 3, 16, 444, 9, 39, 679, 7, 36, 19,
            27, 743, 282, 446, 33, 9, 782, 34, 27, 16, 19, 25, 25, 247, 36, 0, 36, 32, 45, 37,
        ];

        // Loaded as if from a 16-bit image (matching the reference's uint16
        // data and PixelUnits::Bit on a 16-bit image).
        let data: Vec<f32> = RAW.iter().map(|&v| v as f32 / 65535.0).collect();
        let size = ImageSize {
            width: N,
            height: 1,
        };
        let image = Image::<f32, 1>::new(size, data)?;
        let mut ctx = PipelineContext::new_from_image_test(image)?;
        ctx.image_meta.nr_of_bits = 16;

        let cmd = Threshold {
            thresholds: vec![ThresholdEntry {
                method: ThresholdMethod::Triangle,
                min_threshold: 0.0,
                max_threshold: 65535.0,
                unit: PixelUnits::Bit,
                segmentation_class: SegmentationClass(1),
                value_source: ThresholdValueSource::ActualImage,
            }],
        };
        let mut cache = GlobalPipelineCache::default();
        cmd.execute(&mut ctx, &mut cache)?;

        let fg = ctx
            .segmentation_map
            .as_ref()
            .expect("no segmentation map")
            .as_slice()
            .iter()
            .filter(|&&v| v != 0)
            .count();

        // C++ reference: min=0.0 max=1449.0 triangleBin=11 finalThreshold=68,
        // foreground_px(strict>)=658 of 2000. With the paper's triangle bin
        // 10, the same pipeline gives the cut (10 + 1) / 255 · 1449 ≈ 62.5,
        // and 660 pixels lie strictly above it.
        let (hist, _, _) =
            histogram_from_slice(&RAW.iter().map(|&v| v as f32 / 65535.0).collect::<Vec<_>>());
        assert_eq!(thresh_triangle(&hist), 10);
        assert_eq!(
            fg, 660,
            "Threshold::execute foreground pixel count does not match the reference pipeline"
        );
        Ok(())
    }

    // ── value_source ─────────────────────────────────────────────────────

    /// Builds a 2x2 constant-value gray image, standing in for the pipeline's
    /// current (e.g. blurred) image.
    fn constant_gray_image(value: f32) -> Result<Image<f32, 1>, Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        Ok(Image::<f32, 1>::new(size, vec![value; 4])?)
    }

    /// A single `Mean`-thresholded entry over the full relative range, with
    /// the given `value_source`.
    fn mean_threshold_entry(value_source: ThresholdValueSource) -> ThresholdEntry {
        ThresholdEntry {
            method: ThresholdMethod::Mean,
            min_threshold: 0.0,
            max_threshold: 1.0,
            unit: PixelUnits::Relative,
            segmentation_class: SegmentationClass(1),
            value_source,
        }
    }

    fn foreground_count(ctx: &PipelineContext) -> usize {
        ctx.segmentation_map
            .as_ref()
            .expect("no segmentation map")
            .as_slice()
            .iter()
            .filter(|&&v| v != 0)
            .count()
    }

    /// `ActualImage` (the default) computes the threshold from the same
    /// constant image it's applied to: `Mean` on a constant image degenerates
    /// to `dmin == dmax`, so the cut-off equals the pixel value itself and
    /// the strict `>` comparison excludes every pixel. This is the baseline
    /// `RawImage`/`Memory` are contrasted against below.
    #[test]
    fn test_value_source_actual_image_is_the_default_behavior()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let mut cache = GlobalPipelineCache::default();
        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::ActualImage)],
        };
        cmd.execute(&mut ctx, &mut cache)?;

        assert_eq!(foreground_count(&ctx), 0);
        Ok(())
    }

    /// `RawImage` must calculate the cut-off from the unedited image cached
    /// for the current channel (`ImageAddress::Channel`), not from the
    /// pipeline's current (here: constant 0.5) image - while still
    /// classifying the *current* image's pixels against that cut-off. The
    /// raw image here is skewed low (mean bin threshold well under 0.5), so
    /// every actual-image pixel (0.5) ends up foreground - the opposite of
    /// what `ActualImage` produces on the same constant image above.
    #[test]
    fn test_value_source_raw_image_computes_threshold_from_the_channel_cache()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let channel = ctx.get_image_plane().expect("test image has plane info").c;

        let raw_size = ImageSize {
            width: 4,
            height: 1,
        };
        let raw_image = Image::<f32, 1>::new(raw_size, vec![0.0, 0.0, 0.0, 1.0])?;
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::new_f32_gray_from_image_test(raw_image)),
            channel,
            ctx.image_meta.image_tile_info.clone(),
        );

        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::RawImage)],
        };
        cmd.execute(&mut ctx, &mut cache)?;

        assert_eq!(
            foreground_count(&ctx),
            4,
            "all 4 actual-image pixels should be foreground"
        );
        Ok(())
    }

    #[test]
    fn test_value_source_raw_image_missing_from_cache_is_a_cache_miss()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let mut cache = GlobalPipelineCache::default();
        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::RawImage)],
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(matches!(result, Err(InternalErrors::CacheMiss(_))));
        Ok(())
    }

    /// `Memory(id)` mirrors `RawImage`, but the source image is a
    /// user-defined snapshot addressed by `MemoryId` rather than the
    /// channel's raw image.
    #[test]
    fn test_value_source_memory_computes_threshold_from_the_addressed_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;

        let snapshot_size = ImageSize {
            width: 4,
            height: 1,
        };
        let snapshot_image = Image::<f32, 1>::new(snapshot_size, vec![0.0, 0.0, 0.0, 1.0])?;
        let memory_id = MemoryId::PipelineContext(7);
        let mut cache = GlobalPipelineCache::default();
        cache.image_cache.insert(
            CacheAddress::Memory(memory_id),
            Arc::new(ImageContainer::new_f32_gray_from_image_test(snapshot_image)),
        );

        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::Memory(
                memory_id,
            ))],
        };
        cmd.execute(&mut ctx, &mut cache)?;

        assert_eq!(
            foreground_count(&ctx),
            4,
            "all 4 actual-image pixels should be foreground"
        );
        Ok(())
    }

    #[test]
    fn test_value_source_memory_missing_from_cache_is_a_cache_miss()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let mut cache = GlobalPipelineCache::default();
        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::Memory(
                MemoryId::PipelineContext(9),
            ))],
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(matches!(result, Err(InternalErrors::CacheMiss(_))));
        Ok(())
    }

    /// A source image that isn't `F32Gray` (e.g. RGB) must be rejected
    /// instead of silently misreading its pixel buffer.
    #[test]
    fn test_value_source_raw_image_rejects_a_non_gray_cached_image()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let channel = ctx.get_image_plane().expect("test image has plane info").c;

        let rgb_size = ImageSize {
            width: 1,
            height: 1,
        };
        let rgb_image = Image::<f32, 3>::new(rgb_size, vec![0.1, 0.2, 0.3])?;
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::new_f32_rgb_from_image_test(rgb_image)),
            channel,
            ctx.image_meta.image_tile_info.clone(),
        );

        let cmd = Threshold {
            thresholds: vec![mean_threshold_entry(ThresholdValueSource::RawImage)],
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(matches!(result, Err(InternalErrors::FormatMismatch { .. })));
        Ok(())
    }

    /// Multiple entries sharing the same `value_source` must reuse one
    /// memoized histogram instead of resolving/rebuilding it per entry - the
    /// entries below use different methods (`Mean` vs `Otsu`) on the same
    /// `RawImage`, so this also guards against the memo being keyed on
    /// something other than the source itself.
    #[test]
    fn test_value_source_shared_across_entries_is_resolved_once()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = PipelineContext::new_from_image_test(constant_gray_image(0.5)?)?;
        let channel = ctx.get_image_plane().expect("test image has plane info").c;

        let raw_size = ImageSize {
            width: 4,
            height: 1,
        };
        let raw_image = Image::<f32, 1>::new(raw_size, vec![0.0, 0.0, 0.0, 1.0])?;
        let mut cache = GlobalPipelineCache::default();
        cache.add_to_channel_cache(
            Arc::new(ImageContainer::new_f32_gray_from_image_test(raw_image)),
            channel,
            ctx.image_meta.image_tile_info.clone(),
        );

        let cmd = Threshold {
            thresholds: vec![
                mean_threshold_entry(ThresholdValueSource::RawImage),
                ThresholdEntry {
                    method: ThresholdMethod::Otsu {
                        classes: OtsuClasses::Two,
                    },
                    min_threshold: 0.0,
                    max_threshold: 1.0,
                    unit: PixelUnits::Relative,
                    segmentation_class: SegmentationClass(2),
                    value_source: ThresholdValueSource::RawImage,
                },
            ],
        };

        // Would panic/error on a broken memo lookup (e.g. re-resolving a
        // cache entry that was only inserted once) well before reaching this
        // assertion.
        cmd.execute(&mut ctx, &mut cache)?;
        assert_eq!(foreground_count(&ctx), 4);
        Ok(())
    }
}
