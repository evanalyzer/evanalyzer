//! # Subtract Reference Channel
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! ## Overview
//! Removes a signal from the target with a channel showing only that signal
//! (the reference): `residual = S − (a · align(T) + b)`, the scale `a`, the
//! offset `b` and the shift set by hand or estimated from the image -
//! bleed-through or autofluorescence correction (linear unmixing,
//! Zimmermann 2005) with the factor fitted per image (as the subtraction of
//! an autofluorescence-only image, van de Lest et al. 1995).
//!
//! ## Implementation
//! Plane by plane, on `f32` buffers of the plane: the backgrounds with Rank
//! Filter's median or Rolling Ball, the shift with phase correlation
//! (`libs::phase_correlation`) and applied with `libs::translate`, the
//! Gaussians with the discrete Gaussian of `libs::scale_space`, the fit a
//! Huber M-estimate (Huber 1964) by iteratively reweighted least squares,
//! the influence of the reference's brightest pixels limited (Mallows'
//! GM-estimator, Hampel et al. 1986) and the pixels far off the line on
//! either side left out.
//!
//! Runs per tile: the estimates are made on the tile, which is the whole
//! image up to 4096 px (see [`EstimationScope`]).

use crate::algos::filters::rank_filter::{RankFilter, RankFilterType};
use crate::algos::libs::blur_matching::{ReferenceShift, estimate_blur_difference};
use crate::algos::libs::phase_correlation::{GUIZAR_SICAIROS_2008, estimate_shift};
use crate::algos::libs::robust_stats::{MAD_TO_SIGMA, median_in_place};
use crate::algos::libs::scale_space::{convolve_separable, discrete_gaussian};
use crate::algos::libs::translate::{KEYS_1981, Translation};
use crate::algos::math::load_channel::missing_channel;
use crate::algos::{
    BallType, ExecutionScope, GlobalPipelineCache, ImageAlgorithm, Interpolation, Padding,
    PipelineContext, RollingBall,
};
use crate::image::{ImageContainer, ManagedImage};
use crate::pipeline::pipeline::PipelineImageMeta;
use crate::pipeline::pipeline_cache::CacheAddress;
use crate::pipeline::pipeline_measure::{MeasurementRegion, Outputs};
use evanalyzer_cfg::core_types::{
    CitationMetadata, ImageChannelIdx, InternalErrors, MeasurementInfo, MeasurementUnit,
};
use kornia_image::{Image, ImageSize};
use log::{debug, warn};
use macros::CommandsMeta;
use std::sync::Arc;

/// Removes a signal from the image with a channel that shows only that
/// signal (the reference) - e.g. serum filaments or autofluorescence in an
/// EV spot channel: `residual = S − (a · align(T) + b)`. The scale `a`, the
/// offset `b` and the shift are set by hand or estimated from the image.
/// Typically the first step of a pipeline, before the spot detection.
///
/// **Order:** camera offset → background (both channels) → alignment (the
/// reference shifted onto the target - before the fit: a wrong shift gives
/// a wrong `a` and leaves a relief) → extra blur of the sharper channel (set
/// by hand or estimated per image, see [`Sharpness`]) →
/// the same smoothing of both (unsmoothed, noise in the reference biases
/// `a` low) → factor → subtraction (by default only where the reference is
/// brighter than its background).
///
/// **Assumption:** estimating the factor from the image assumes the target's
/// own signal is not correlated with the reference. If it binds the
/// reference structure (e.g. a protein on the filaments), use a Fixed
/// factor measured on single-stain controls.
///
/// **Reading the residual:** a relief (bright on one side, dark on the
/// other) → alignment. A bright centre line with dark sides → raise the
/// target's extra blur; a dark centre line with bright sides → raise the
/// reference's. The whole structure too bright or too dark → Factor scale.
///
/// **Spot detection afterwards:** leftovers of the structure that one
/// factor can't remove (e.g. channels focused differently) and the higher
/// photon noise of bright regions look like spots to a global cut-off.
/// Output Significance divides the residual by its local noise, so a
/// cut-off (e.g. 5 in a threshold) means the same everywhere.
///
/// Measures per image: the factor, offset, shift, alignment quality, R²,
/// the parts of fit and saturated pixels, whether a fallback was used (for
/// the shift or the factor) and with Sharpness Auto the extra blurs used (on an image split into tiles: the median over the
/// tiles). Exclude
/// Image can leave images out by them - e.g. "Fallback used" above 0.5.
///
/// Runs per tile, the whole image up to 4096 px: on a larger image each
/// tile has its own estimates for now (Image scope logs a warning), and the
/// shift pads the tile borders with their edge pixels (no overlap with the
/// neighbouring tiles).
///
/// The EV-in-serum assay - spots in the current image, the serum channel 4,
/// everything else at its default:
///
/// ```json
/// { "type": "SUBTRACT_REFERENCE_CHANNEL", "id": "reference", "referenceChannel": 4 }
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "Preprocessing",
    display_name = "Subtract Reference Channel"
)]
pub struct SubtractReferenceChannel {
    /// Image to correct: the current pipeline image, or a channel.
    pub target: TargetSource,

    /// Channel containing only the signal to remove.
    #[cmdsmeta(default = ImageChannelIdx(1), summary = true)]
    pub reference_channel: ImageChannelIdx,

    /// Output: the residual, or the scaled aligned reference (for
    /// inspection).
    #[cmdsmeta(default = OutputImage::Residual)]
    pub output: OutputImage,

    /// How the reference is shifted onto the target.
    #[cmdsmeta(section = "Alignment", summary = true)]
    pub alignment: Alignment,

    /// Interpolation for the shift.
    #[cmdsmeta(section = "Alignment", default = ShiftInterpolation::Cubic, visibility = Advanced)]
    pub interpolation: ShiftInterpolation,

    /// Slowly varying background, removed from both channels.
    #[cmdsmeta(section = "Background & smoothing")]
    pub background: Background,

    /// Gaussian smoothing (σ, px) applied identically to both channels (0 =
    /// off).
    #[cmdsmeta(
        section = "Background & smoothing",
        default = 1.0,
        min = 0.0,
        max = 20.0,
        step = 0.1
    )]
    pub smoothing_sigma: f32,

    /// How the sharper channel is blurred to the other's sharpness before
    /// the fit: by the extra blurs set below, or estimated per image.
    #[cmdsmeta(section = "Background & smoothing", default = Sharpness::Manual)]
    pub sharpness: Sharpness,

    /// Extra blur (σ, px) for the target if it is sharper than the
    /// reference (0 = off). With Sharpness Auto: used when no estimate is
    /// made.
    #[cmdsmeta(
        section = "Background & smoothing",
        default = 0.0,
        min = 0.0,
        max = 20.0,
        step = 0.1
    )]
    pub target_extra_sigma: f32,

    /// Extra blur (σ, px) for the reference if it is sharper than the
    /// target (0 = off). With Sharpness Auto: used when no estimate is
    /// made.
    #[cmdsmeta(
        section = "Background & smoothing",
        default = 0.0,
        min = 0.0,
        max = 20.0,
        step = 0.1
    )]
    pub reference_extra_sigma: f32,

    /// Camera offset subtracted from target and reference before anything
    /// else.
    #[cmdsmeta(section = "Background & smoothing", visibility = Advanced)]
    pub camera_offset: CameraOffset,

    /// Pixels at or above this level (in either channel) are left out of
    /// all estimates - and their neighbours as far as the blurs and the
    /// shift reach.
    #[cmdsmeta(section = "Background & smoothing", visibility = Advanced)]
    pub saturation: Saturation,

    /// How the scale factor `a` is set.
    #[cmdsmeta(section = "Factor", summary = true)]
    pub factor: Factor,

    /// Multiplies `a` after it is set (fine tuning, 1 = unchanged).
    #[cmdsmeta(section = "Factor", default = 1.0, min = 0.0, max = 100.0, step = 0.01)]
    pub factor_scale: f32,

    /// Names the step's measurements (factor, shift, …) in the results.
    #[cmdsmeta(section = "Factor", measurement_id = "reference", display_name = "ID")]
    pub id: String,

    /// Area the estimates (shift, factor) are computed over.
    #[cmdsmeta(section = "Factor", default = EstimationScope::Image, visibility = Advanced)]
    pub estimation_scope: EstimationScope,

    /// Local maximum of the reference over this radius (px) before
    /// subtracting; tolerates a small leftover shift or edge mismatch (0 =
    /// off).
    #[cmdsmeta(section = "Factor", default = 0.0, min = 0.0, max = 10.0, step = 0.5, visibility = Advanced)]
    pub edge_tolerance: f32,

    /// Subtract only where the reference is brighter than its background
    /// (after the background is removed): dark lines or holes in the
    /// reference (cracks, gaps) subtract nothing instead of brightening the
    /// result. Off: the reference's dips are subtracted too. Needs a
    /// background removal - without one the reference is never below 0.
    #[cmdsmeta(section = "Subtraction", default = true)]
    pub only_brighter_reference: bool,

    /// Set negative residual values to 0.
    #[cmdsmeta(section = "Subtraction", default = true)]
    pub clip_negative: bool,
}

/// The image [`SubtractReferenceChannel`] corrects.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum TargetSource {
    /// The pipeline's image at this step.
    CurrentImage,
    /// An image channel as it was read.
    Channel {
        /// The channel to correct.
        channel: ImageChannelIdx,
    },
}

/// The camera's offset (its black level).
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum CameraOffset {
    /// No offset subtracted.
    None,
    /// Fixed offsets per channel.
    Manual {
        /// Offset of the target.
        #[cmdsmeta(default = 100.0, min = 0.0, max = 65535.0, step = 1.0)]
        target: f32,
        /// Offset of the reference.
        #[cmdsmeta(default = 100.0, min = 0.0, max = 65535.0, step = 1.0)]
        reference: f32,
    },
}

/// The level from which a pixel counts as saturated.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Saturation {
    /// The image's largest gray value (255 for 8 bit, 4095 for 12 bit,
    /// 65535 for 16 bit, …). None for float images.
    Auto,
    /// A fixed level.
    Manual {
        /// Pixels at or above this gray value are saturated (camera counts,
        /// e.g. 4000 of a 12-bit camera's 4095; on float images as stored).
        #[cmdsmeta(default = 65535.0, min = 0.0, max = 1.0e9, step = 1.0)]
        level: f32,
    },
    /// No pixel is left out.
    Off,
}

/// The background removed from both channels.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Background {
    /// The median over a circle (as Median Subtract).
    Median {
        /// Radius (px): a little larger than the structures.
        #[cmdsmeta(default = 30.0, min = 1.0, max = 200.0, step = 1.0)]
        radius: f32,
    },
    /// Rolling Ball.
    RollingBall {
        /// Radius of the ball (px).
        #[cmdsmeta(default = 30.0, min = 1.0, max = 200.0, step = 1.0)]
        radius: f32,
    },
    /// No background removed.
    None,
}

/// How the reference is shifted onto the target.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Alignment {
    /// Estimated per image (phase correlation).
    Auto {
        /// Larger shifts (px) are not believed: the fallback is used.
        #[cmdsmeta(default = 5.0, min = 0.0, max = 100.0, step = 0.5)]
        max_shift: f32,
        /// Below this quality (0 - 1) the fallback is used.
        #[cmdsmeta(default = 0.3, min = 0.0, max = 1.0, step = 0.05)]
        min_quality: f32,
        /// Shift right (+) used when the estimate isn't believed (px).
        #[cmdsmeta(default = 0.0, min = -100.0, max = 100.0, step = 0.1)]
        fallback_dx: f32,
        /// Shift down (+) used when the estimate isn't believed (px).
        #[cmdsmeta(default = 0.0, min = -100.0, max = 100.0, step = 0.1)]
        fallback_dy: f32,
    },
    /// A fixed shift of the reference.
    Fixed {
        /// Shift right (+) or left (−) in pixels.
        #[cmdsmeta(default = 0.0, min = -100.0, max = 100.0, step = 0.1)]
        dx: f32,
        /// Shift down (+) or up (−) in pixels.
        #[cmdsmeta(default = 0.0, min = -100.0, max = 100.0, step = 0.1)]
        dy: f32,
    },
    /// Not shifted.
    None,
}

/// How [`SubtractReferenceChannel`] matches the sharpness of target and
/// reference.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Sharpness {
    /// The extra blurs set by hand (Target / Reference Extra Sigma).
    Manual,
    /// Estimated per image, after the alignment: how much blurrier one
    /// channel is than the other (their cross-spectrum over the
    /// reference's spectrum, see `libs::blur_matching`), and the sharper
    /// one blurred by the difference - for channels whose focus differs
    /// from image to image. Without an estimate (too little structure in
    /// the target) the extra blurs set by hand are used. With Factor Robust
    /// Fit the estimate is used only where the factor fits at least as
    /// well (R²) as with the blurs set by hand: structures of different
    /// colour ratio at different scales (e.g. coarse serum and fine beads)
    /// can read as a blur difference.
    Auto {
        /// Larger differences (σ, px) are not believed: the extra blurs set
        /// by hand are used.
        #[cmdsmeta(default = 3.0, min = 0.5, max = 20.0, step = 0.5)]
        max_sigma: f32,
    },
}

/// How [`SubtractReferenceChannel`] interpolates the shifted reference.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum ShiftInterpolation {
    /// Bicubic (Keys, a = −0.5).
    Cubic,
    /// Bilinear: a little smoother.
    Linear,
}

/// How the scale factor `a` is set.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum Factor {
    /// Fitted per image: a robust line through the structure's pixels
    /// (reference above its noise), pixels bright in only one channel left
    /// out - EVs in the target as well as dots in the reference only (e.g.
    /// dye aggregates). The fit is used when it is precise (see Max
    /// Uncertainty). A precise factor of about 0 is a result too (the
    /// target shows nothing of the reference, e.g. a field without serum):
    /// a negative one is set to 0.
    RobustFit {
        /// Fit pixels: reference above its median + this × its noise.
        #[cmdsmeta(default = 4.0, min = 0.0, max = 100.0, step = 0.5)]
        noise_factor: f32,
        /// Fit pixels: reference also above this percentile.
        #[cmdsmeta(default = 50.0, min = 0.0, max = 100.0, step = 1.0)]
        min_percentile: f32,
        /// Pixels this many times the noise above the line (EVs) or below
        /// it (dots in the reference only) are left out.
        #[cmdsmeta(default = 3.0, min = 0.5, max = 100.0, step = 0.5)]
        outlier_k: f32,
        /// Fit the offset `b` too; otherwise the line goes through 0.
        #[cmdsmeta(default = true)]
        fit_offset: bool,
        /// The fit is used when its uncertainty moves the subtracted signal
        /// at the reference's bright pixels (99th percentile of the fit
        /// pixels) by less than this many times the noise; otherwise the
        /// fallback. Unlike R² it accepts a precise factor of about 0
        /// (nothing of the reference in the target). Real images: 0.01 -
        /// 0.05.
        #[cmdsmeta(default = 0.25, min = 0.01, max = 10.0, step = 0.05)]
        max_uncertainty: f32,
        /// The factor used when the fit fails (0 = subtract nothing).
        #[cmdsmeta(default = 0.0, min = 0.0, max = 1000.0, step = 0.01)]
        fallback: f32,
        /// A fitted factor below this uses the fallback.
        #[cmdsmeta(default = 0.0, min = 0.0, max = 1000.0, step = 0.01)]
        min_value: f32,
        /// A fitted factor above this uses the fallback.
        #[cmdsmeta(default = 100.0, min = 0.0, max = 1000.0, step = 0.01)]
        max_value: f32,
    },
    /// A fixed factor (e.g. measured on single-stain controls).
    Fixed {
        /// The factor `a`. (Not named `value`: the generated settings code
        /// has a parameter of that name.)
        #[cmdsmeta(default = 1.0, min = 0.0, max = 1000.0, step = 0.01)]
        factor: f32,
    },
    /// From the exposure times: `a = calibration · t_target / t_reference`.
    ExposureRatio {
        /// The factor at equal exposure times.
        #[cmdsmeta(default = 1.0, min = 0.0, max = 1000.0, step = 0.01)]
        calibration: f32,
        /// Exposure time of the target (ms).
        #[cmdsmeta(default = 100.0, min = 0.001, max = 1.0e6, step = 1.0)]
        exposure_target_ms: f32,
        /// Exposure time of the reference (ms).
        #[cmdsmeta(default = 100.0, min = 0.001, max = 1.0e6, step = 1.0)]
        exposure_reference_ms: f32,
    },
}

/// The settings of a fitted factor (Robust Fit).
#[derive(Clone, Copy, Debug)]
struct FitParams {
    noise_factor: f32,
    min_percentile: f32,
    outlier_k: f32,
    fit_offset: bool,
    max_uncertainty: f32,
    fallback: f32,
    min_value: f32,
    max_value: f32,
}

impl Factor {
    /// The fit's settings; `None` for a factor that isn't fitted.
    fn fit_params(&self) -> Option<FitParams> {
        match *self {
            Factor::RobustFit {
                noise_factor,
                min_percentile,
                outlier_k,
                fit_offset,
                max_uncertainty,
                fallback,
                min_value,
                max_value,
            } => Some(FitParams {
                noise_factor,
                min_percentile,
                outlier_k,
                fit_offset,
                max_uncertainty,
                fallback,
                min_value,
                max_value,
            }),
            Factor::Fixed { .. } | Factor::ExposureRatio { .. } => None,
        }
    }
}

/// The area the shift and factor are estimated over.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum EstimationScope {
    /// The whole image - per tile for now on an image larger than 4096 px
    /// (logged).
    Image,
    /// Each tile on its own.
    Tile,
}

/// What [`SubtractReferenceChannel`] puts out.
#[derive(Clone, Copy, PartialEq, Debug, CommandsMeta)]
pub enum OutputImage {
    /// The target without the reference's signal.
    Residual,
    /// The scaled, aligned reference `a · align(T) + b` (to inspect the
    /// estimate).
    ScaledReference,
    /// The residual in units of its local noise: the deviation from its
    /// local baseline (median over 21 × 21 px) over the spread of the
    /// deviations there, from the negative side only (noise is symmetric,
    /// EVs only add light - so they don't raise their own noise). The noise
    /// is higher where the image is brighter (photon noise) and where the
    /// subtraction leaves leftovers of the structure; both raise it, so 5
    /// means five times the noise right there - the same on the structure,
    /// on a bright sheet and on dark background. For spot detection (a
    /// threshold, Wavelet Spots) where leftovers or bright regions would
    /// count as spots.
    Significance,
}

/// Fewer fit pixels use the fallback factor.
const MIN_FIT_PIXELS: usize = 1000;
/// Huber's tuning constant (95 % efficiency for normal errors).
const HUBER_K: f64 = 1.345;
/// Rounds of the fit; it settles within about 10 - 30 on real images.
const MAX_ITERATIONS: usize = 100;
/// The longest cycle of rounds a fit settles on (see `robust_fit`).
const MAX_CYCLE: usize = 4;
/// Leverage limit of the fit: a pixel further than this many times the
/// spread (MAD) from the fit pixels' median reference pulls the line only
/// with weight `LEVERAGE_K · spread / distance` (Mallows' GM-estimator,
/// Hampel et al. 1986). Without it a few very bright reference pixels -
/// dots in the reference only, or the structure's brightest pixels - decide
/// the line, and a fit on a weak target slides to a flat line (a ≈ 0).
const LEVERAGE_K: f64 = 2.0;
/// The fit has settled once a round changes `a` by less than this part of
/// its uncertainty (and `b` by less than this part of the noise).
const SETTLE_FRACTION: f64 = 0.01;
/// Fewer pixels give no noise estimate (Significance).
const MIN_NOISE_PIXELS: usize = 100;
/// The window of the local noise (Significance): (2·10 + 1)² px, about
/// 3.4 µm at 160 nm per pixel - enough pixels for a steady estimate, narrow
/// enough to follow the brightness and the edges of the structure.
const LOCAL_NOISE_RADIUS: usize = 10;
const TOLERANCE: f64 = 1e-6;

impl ImageAlgorithm for SubtractReferenceChannel {
    /// Corrects the target plane by plane (see the struct's docs); the
    /// residual (or the scaled reference) becomes the pipeline's image.
    ///
    /// # Errors
    /// A missing channel, a target and reference of different size, label
    /// or colour images.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let tile = ctx.image_meta.image_tile_info;
        let channel = |index: i32| -> Result<Arc<ImageContainer>, InternalErrors> {
            cache
                .image_cache
                .get(&CacheAddress::Channel((index, tile)))
                .ok_or_else(|| missing_channel(cache, index, tile))
        };
        let target = match self.target {
            TargetSource::CurrentImage => ctx.image.clone(),
            TargetSource::Channel { channel: c } => channel(c.0 as i32)?,
        };
        let reference = channel(self.reference_channel.0 as i32)?;
        let (target_image, reference_image) = match (target.as_ref(), reference.as_ref()) {
            (ImageContainer::F32Gray(t), ImageContainer::F32Gray(r)) => (t, r),
            (ImageContainer::U32(_), _) | (_, ImageContainer::U32(_)) => {
                return Err(InternalErrors::InvalidArgument(
                    "Subtract Reference Channel: label images can't be corrected".into(),
                ));
            }
            _ => {
                return Err(InternalErrors::InvalidArgument(
                    "Subtract Reference Channel: target and reference must be grey images".into(),
                ));
            }
        };
        let depth = target_image.dimensionality.depth().max(1) as usize;
        if target_image.data.size() != reference_image.data.size()
            || reference_image.dimensionality.depth().max(1) as usize != depth
        {
            let size = target_image.data.size();
            let other = reference_image.data.size();
            return Err(InternalErrors::InvalidArgument(format!(
                "Subtract Reference Channel: the target ({} x {}) and the reference channel {} \
                 ({} x {}) differ in size",
                size.width, size.height, self.reference_channel.0, other.width, other.height
            )));
        }
        let plane = target_image.plane_size();
        let full = ctx.image_meta.full_image_width;
        if self.estimation_scope == EstimationScope::Image
            && (plane.width < full.width || plane.height * depth < full.height)
            && tile.offset_x == 0
            && tile.offset_y == 0
        {
            warn!(
                "{}: Subtract Reference Channel estimates per tile - image-wide estimates \
                 on an image split into tiles are not supported yet",
                cache.image_rel_path.display()
            );
        }

        let saturation = saturation_level(self.saturation, ctx.image_meta.nr_of_bits);
        // The result goes into the scratch pad (prepared at the pipeline
        // image's size; a channel target of another size gets its own).
        let size = target_image.data.size();
        if !matches!(ctx.scratch_pad.as_ref(), ImageContainer::F32Gray(img) if img.data.size() == size)
        {
            ctx.scratch_pad = Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::new(size, vec![0f32; size.width * size.height])
                    .map_err(InternalErrors::from_kornia)?,
                tile_offset: target_image.tile_offset,
                plane: target_image.plane,
                dimensionality: target_image.dimensionality,
            }));
        }
        let ImageContainer::F32Gray(result) = Arc::make_mut(&mut ctx.scratch_pad) else {
            unreachable!("the scratch pad was made a grey image above");
        };
        result.tile_offset = target_image.tile_offset;
        result.plane = target_image.plane;
        result.dimensionality = target_image.dimensionality;
        let output = result.data.as_slice_mut();
        let mut estimates = Vec::with_capacity(depth);
        let len = plane.width * plane.height;
        for z in 0..depth {
            let planes = PlaneInput {
                target: target_image.plane_slice(z),
                reference: reference_image.plane_slice(z),
                size: plane,
                saturation,
            };
            let out = &mut output[z * len..(z + 1) * len];
            estimates.push(self.correct_plane(&planes, out, &ctx.image_meta, cache)?);
        }

        for (z, e) in estimates.iter().enumerate() {
            debug!(
                "{} tile ({}, {}) plane {z}: a {:.4}, b {:.3}, shift ({:.3}, {:.3}), quality {:.3}, \
                 R² {:.3}, fallback {}",
                cache.image_rel_path.display(),
                tile.offset_x,
                tile.offset_y,
                e.a,
                e.b,
                e.dy,
                e.dx,
                e.quality,
                e.r2,
                e.fallback
            );
        }
        for (info, value) in self
            .measurement_outputs()
            .into_iter()
            .zip(median_of(&estimates))
        {
            cache.add_image_measurement(
                info,
                value
                    .is_finite()
                    .then_some((MeasurementRegion::Image, value)),
            );
        }

        ctx.swap()
    }

    fn name(&self) -> &'static str {
        "Subtract Reference Channel"
    }

    /// Measured by `tests::tile_ram` (see the trait method): both channels'
    /// planes, two blur buffers, the median's output and the alignment's
    /// spectra (bounded beyond 1024 × 1024 px).
    fn ram_per_tile_pixel(&self) -> f64 {
        28.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut papers = vec![&VAN_DE_LEST_1995, &ZIMMERMANN_2005];
        if matches!(self.alignment, Alignment::Auto { .. }) {
            papers.push(&GUIZAR_SICAIROS_2008);
        }
        if !matches!(self.alignment, Alignment::None)
            && self.interpolation == ShiftInterpolation::Cubic
        {
            papers.push(&KEYS_1981);
        }
        if self.factor.fit_params().is_some() {
            papers.push(&HUBER_1964);
            papers.push(&HAMPEL_1986);
        }
        papers
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    fn measures(&self) -> bool {
        true
    }

    fn measurement_outputs(&self) -> Vec<MeasurementInfo> {
        let outputs = Outputs::of(&self.id);
        let mut infos = vec![
            outputs.image("factor", "Factor", MeasurementUnit::None),
            outputs.image("offset", "Offset", MeasurementUnit::Intensity),
            outputs.image("shift_x", "Shift X", MeasurementUnit::Pixels),
            outputs.image("shift_y", "Shift Y", MeasurementUnit::Pixels),
            outputs.image(
                "alignment_quality",
                "Alignment quality",
                MeasurementUnit::None,
            ),
            outputs.image("r2", "R²", MeasurementUnit::None),
            outputs.image(
                "fit_pixel_fraction",
                "Fit pixel fraction",
                MeasurementUnit::Fraction,
            ),
            outputs.image(
                "saturated_fraction",
                "Saturated fraction",
                MeasurementUnit::Fraction,
            ),
            outputs.image("fallback_used", "Fallback used", MeasurementUnit::None),
            outputs.image(
                "residual_noise",
                "Residual noise",
                MeasurementUnit::Intensity,
            ),
        ];
        // The estimated blurs only where they are estimated.
        if matches!(self.sharpness, Sharpness::Auto { .. }) {
            infos.push(outputs.image(
                "target_extra_sigma",
                "Target extra blur",
                MeasurementUnit::Pixels,
            ));
            infos.push(outputs.image(
                "reference_extra_sigma",
                "Reference extra blur",
                MeasurementUnit::Pixels,
            ));
        }
        infos
    }
}

/// One plane of both channels.
struct PlaneInput<'a> {
    target: &'a [f32],
    reference: &'a [f32],
    size: ImageSize,
    saturation: Option<f32>,
}

impl PlaneInput<'_> {
    #[inline]
    fn saturated(&self, i: usize) -> bool {
        self.saturation
            .is_some_and(|level| self.target[i] >= level || self.reference[i] >= level)
    }
}

/// What one plane was corrected with - NaN where it doesn't apply (no
/// quality without Auto alignment, no R² without a fit).
#[derive(Debug, Clone, Copy)]
struct PlaneEstimate {
    a: f64,
    b: f64,
    dy: f64,
    dx: f64,
    quality: f64,
    r2: f64,
    fit_fraction: f64,
    saturated_fraction: f64,
    fallback: bool,
    /// Only with the Significance output: the residual's typical local
    /// noise (median).
    residual_noise: f64,
    /// The extra blurs (σ, px) used for the target and the reference.
    target_sigma: f64,
    reference_sigma: f64,
}

/// The planes' (and so a tile's) values in the order of the measurement
/// outputs: their medians.
fn median_of(estimates: &[PlaneEstimate]) -> [f64; 12] {
    let median = |f: &dyn Fn(&PlaneEstimate) -> f64| -> f64 {
        let mut values: Vec<f64> = estimates.iter().map(f).filter(|v| v.is_finite()).collect();
        if values.is_empty() {
            return f64::NAN;
        }
        values.sort_by(f64::total_cmp);
        let mid = values.len() / 2;
        if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) / 2.0
        }
    };
    [
        median(&|e| e.a),
        median(&|e| e.b),
        median(&|e| e.dx),
        median(&|e| e.dy),
        median(&|e| e.quality),
        median(&|e| e.r2),
        median(&|e| e.fit_fraction),
        median(&|e| e.saturated_fraction),
        // Any plane falling back counts.
        estimates.iter().any(|e| e.fallback) as u8 as f64,
        median(&|e| e.residual_noise),
        median(&|e| e.target_sigma),
        median(&|e| e.reference_sigma),
    ]
}

/// The result of the robust fit.
struct Fit {
    a: f64,
    b: f64,
    r2: f64,
    /// Pixels in the final fit.
    kept: usize,
    /// `a` and `b` stopped changing within [`MAX_ITERATIONS`].
    settled: bool,
    /// How far the uncertainty of `a` moves the subtracted signal at the
    /// fit pixels' bright end (99th percentile of the reference), in units
    /// of the noise (see `Factor::RobustFit::max_uncertainty`).
    uncertainty: f64,
}

impl SubtractReferenceChannel {
    /// Corrects one plane into `out` (see the struct's docs for the order).
    fn correct_plane(
        &self,
        input: &PlaneInput,
        out: &mut [f32],
        meta: &PipelineImageMeta,
        cache: &GlobalPipelineCache,
    ) -> Result<PlaneEstimate, InternalErrors> {
        let (w, h) = (input.size.width, input.size.height);
        let len = w * h;
        let image_name = || cache.image_rel_path.display().to_string();

        // 1. Camera offset.
        let (offset_t, offset_r) = match self.camera_offset {
            CameraOffset::None => (0.0, 0.0),
            CameraOffset::Manual { target, reference } => (target, reference),
        };
        let mut s: Vec<f32> = input.target.iter().map(|v| v - offset_t).collect();
        let mut t: Vec<f32> = input.reference.iter().map(|v| v - offset_r).collect();

        // 2. Background.
        self.remove_background(&mut s, input.size, meta)?;
        self.remove_background(&mut t, input.size, meta)?;

        // 3. Alignment, before anything is fitted.
        let mut fallback = false;
        let (dy, dx, quality) = match self.alignment {
            Alignment::None => (0.0, 0.0, f64::NAN),
            Alignment::Fixed { dx, dy } => (dy as f64, dx as f64, f64::NAN),
            Alignment::Auto {
                max_shift,
                min_quality,
                fallback_dx,
                fallback_dy,
            } => match estimate_shift(&s, &t, w, h) {
                Some(e)
                    if e.quality >= min_quality as f64
                        && e.dy.abs() <= max_shift as f64
                        && e.dx.abs() <= max_shift as f64 =>
                {
                    (e.dy, e.dx, e.quality)
                }
                estimate => {
                    warn!(
                        "{}: Subtract Reference Channel: shift ({}) not believed (quality \
                         minimum {min_quality}, largest shift {max_shift} px) - using the \
                         fallback ({fallback_dy}, {fallback_dx})",
                        image_name(),
                        estimate.map_or("too small to estimate".to_string(), |e| format!(
                            "{:.2}, {:.2}, quality {:.2}",
                            e.dy, e.dx, e.quality
                        )),
                    );
                    fallback = true;
                    (
                        fallback_dy as f64,
                        fallback_dx as f64,
                        estimate.map_or(f64::NAN, |e| e.quality),
                    )
                }
            },
        };
        if dy.abs() >= h as f64 || dx.abs() >= w as f64 {
            return Err(InternalErrors::InvalidArgument(format!(
                "Subtract Reference Channel: a shift of ({dy}, {dx}) px is larger than the \
                 {w} x {h} px tile"
            )));
        }
        let interpolation = match self.interpolation {
            ShiftInterpolation::Cubic => Interpolation::Cubic,
            ShiftInterpolation::Linear => Interpolation::Linear,
        };
        // Sharpness Auto compares the reference as it will be shifted (see
        // `libs::blur_matching`), before the plane-sized buffers below.
        let mut blur_estimate = matches!(self.sharpness, Sharpness::Auto { .. }).then(|| {
            estimate_blur_difference(
                &s,
                &t,
                w,
                h,
                Some(ReferenceShift {
                    dy,
                    dx,
                    interpolation,
                }),
            )
        });
        let mut work = vec![0f32; len];
        if dy != 0.0 || dx != 0.0 {
            Translation::new(dy, dx, interpolation, Padding::Replicate, w, h)
                .apply::<1>(&t, &mut work, 1, 0.0);
            std::mem::swap(&mut t, &mut work);
        }

        // 4. Sharpness matching, 5. the same smoothing of both.
        let manual = (self.target_extra_sigma, self.reference_extra_sigma);
        let (mut target_sigma, mut reference_sigma) = match self.sharpness {
            Sharpness::Manual => manual,
            Sharpness::Auto { max_sigma } => {
                let estimate = blur_estimate.take().flatten();
                match estimate.map(|e| e.extra_sigmas()) {
                    Some((target, reference)) if target.max(reference) <= max_sigma as f64 => {
                        (target as f32, reference as f32)
                    }
                    found => {
                        warn!(
                            "{}: Subtract Reference Channel: sharpness difference ({}) not \
                             believed (largest {max_sigma} px) - using the extra blurs set by \
                             hand ({}, {})",
                            image_name(),
                            found.map_or(
                                "too little structure to estimate".to_string(),
                                |(t, r)| { format!("target {t:.2}, reference {r:.2} px") }
                            ),
                            self.target_extra_sigma,
                            self.reference_extra_sigma,
                        );
                        // Not counted as a fallback: the shift and the
                        // factor are what Exclude Image rules judge by.
                        manual
                    }
                }
            }
        };
        // An estimated blur must fit at least as well as the blurs set by
        // hand (R² of the factor fit): the estimate sees every structure of
        // the reference at once, and structures of different colour ratio
        // at different scales (e.g. coarse serum and fine beads) read as a
        // blur difference. Without a fit (fixed factor) it is used as it is.
        let fit_params = self.factor.fit_params();
        let compare_with_manual = matches!(self.sharpness, Sharpness::Auto { .. })
            && fit_params.is_some()
            && (target_sigma, reference_sigma) != manual;

        // Not fitted: saturated pixels and the strip the shift padded.
        let margin = if dy != 0.0 || dx != 0.0 {
            dy.abs().max(dx.abs()).ceil() as usize + 1
        } else {
            0
        };
        let saturated = (0..len).filter(|&i| input.saturated(i)).count();
        // The blurs spread a saturated value into its neighbours, the shift
        // moves the reference's: those are left out too (as far as the
        // larger of both compared blurs reaches).
        let blur = |extra: f32| (extra * extra + self.smoothing_sigma.powi(2)).sqrt();
        let mut widest = blur(target_sigma).max(blur(reference_sigma));
        if compare_with_manual {
            widest = widest.max(blur(manual.0)).max(blur(manual.1));
        }
        let reach = (3.0 * widest).ceil() as usize + margin;
        let near_saturated = (saturated > 0).then(|| {
            let mask: Vec<bool> = (0..len).map(|i| input.saturated(i)).collect();
            dilate(&mask, w, h, reach)
        });
        let usable = |i: usize| {
            let (x, y) = (i % w, i / w);
            x >= margin
                && y >= margin
                && x + margin < w
                && y + margin < h
                && !near_saturated.as_ref().is_some_and(|m| m[i])
        };

        let mut tmp = vec![0f32; len];
        // The pair with the blurs set by hand, while both are compared.
        let mut by_hand = compare_with_manual.then(|| (s.clone(), t.clone()));
        let smoothing = self.smoothing_sigma;
        let mut blur_pair = |s: &mut Vec<f32>, t: &mut Vec<f32>, extra: (f32, f32)| {
            gaussian(s, &mut work, &mut tmp, input.size, extra.0);
            gaussian(t, &mut work, &mut tmp, input.size, extra.1);
            gaussian(s, &mut work, &mut tmp, input.size, smoothing);
            gaussian(t, &mut work, &mut tmp, input.size, smoothing);
        };
        blur_pair(&mut s, &mut t, (target_sigma, reference_sigma));
        if let Some((hs, ht)) = by_hand.as_mut() {
            blur_pair(hs, ht, manual);
        }
        drop(tmp);

        // 6. Factor.
        let fit_with = |s: &[f32], t: &[f32], work: &mut Vec<f32>| {
            robust_fit(
                s,
                t,
                &usable,
                work,
                fit_params.expect("only with a fitted factor"),
            )
        };
        let mut chosen_fit: Option<Option<Fit>> = None;
        if let Some((hs, ht)) = by_hand.take() {
            let estimated = fit_with(&s, &t, &mut work);
            let set_by_hand = fit_with(&hs, &ht, &mut work);
            let r2 = |fit: &Option<Fit>| fit.as_ref().map_or(f64::NEG_INFINITY, |f| f.r2);
            if r2(&estimated) >= r2(&set_by_hand) {
                chosen_fit = Some(estimated);
            } else {
                debug!(
                    "{}: Subtract Reference Channel: the estimated blurs (target {target_sigma:.2}, \
                     reference {reference_sigma:.2} px) fit worse than those set by hand (R² \
                     {:.3} < {:.3}) - using those",
                    image_name(),
                    r2(&estimated),
                    r2(&set_by_hand)
                );
                (s, t) = (hs, ht);
                (target_sigma, reference_sigma) = manual;
                chosen_fit = Some(set_by_hand);
            }
        }
        let (mut a, b, r2, fit_fraction) = match self.factor {
            Factor::Fixed { factor } => (factor as f64, 0.0, f64::NAN, f64::NAN),
            Factor::ExposureRatio {
                calibration,
                exposure_target_ms,
                exposure_reference_ms,
            } => (
                calibration as f64 * exposure_target_ms as f64 / exposure_reference_ms as f64,
                0.0,
                f64::NAN,
                f64::NAN,
            ),
            Factor::RobustFit { .. } => {
                let FitParams {
                    max_uncertainty,
                    fallback: fallback_value,
                    min_value,
                    max_value,
                    ..
                } = fit_params.expect("a fitted factor");
                let fit = match chosen_fit.take() {
                    Some(fit) => fit,
                    None => fit_with(&s, &t, &mut work),
                };
                let fraction = fit.as_ref().map_or(0.0, |f| f.kept as f64 / len as f64);
                // A precise negative factor: nothing of the reference in the
                // target (subtracting it would add the reference).
                let clamped = |f: &Fit| if f.a < 0.0 { (0.0, 0.0) } else { (f.a, f.b) };
                match fit {
                    Some(f)
                        if f.settled
                            && f.kept >= MIN_FIT_PIXELS
                            && f.uncertainty <= max_uncertainty as f64
                            && clamped(&f).0 >= min_value as f64
                            && clamped(&f).0 <= max_value as f64 =>
                    {
                        let (a, b) = clamped(&f);
                        (a, b, f.r2, fraction)
                    }
                    fit => {
                        warn!(
                            "{}: Subtract Reference Channel: factor fit not believed ({}) - \
                             using the fallback {fallback_value}",
                            image_name(),
                            fit.as_ref()
                                .map_or("no structure above the noise".to_string(), |f| {
                                    format!(
                                        "a {:.4}, uncertainty {:.2} × noise, {} pixels{}",
                                        f.a,
                                        f.uncertainty,
                                        f.kept,
                                        if f.settled { "" } else { ", did not settle" }
                                    )
                                }),
                        );
                        fallback = true;
                        (
                            fallback_value as f64,
                            0.0,
                            fit.as_ref().map_or(f64::NAN, |f| f.r2),
                            fraction,
                        )
                    }
                }
            }
        };
        a *= self.factor_scale as f64;
        let significance = self.output == OutputImage::Significance;
        if !significance {
            // Freed: only Significance still needs it.
            work = Vec::new();
        }

        // 7. Subtraction.
        if self.edge_tolerance > 0.0 {
            let filter = RankFilter {
                radius: self.edge_tolerance as f64,
                filter_type: RankFilterType::Max,
            };
            let image = Image::new(input.size, t).map_err(InternalErrors::from_kornia)?;
            t = filter.process_image::<1>(&image)?.into_vec();
        }
        let (a32, b32) = (a as f32, b as f32);
        // Below its background the reference shows no light to remove.
        let floor = if self.only_brighter_reference {
            0.0
        } else {
            f32::NEG_INFINITY
        };
        for ((o, sv), tv) in out.iter_mut().zip(&s).zip(&t) {
            let scaled = a32 * tv.max(floor) + b32;
            *o = match self.output {
                OutputImage::ScaledReference => scaled,
                OutputImage::Residual if self.clip_negative => (sv - scaled).max(0.0),
                OutputImage::Residual | OutputImage::Significance => sv - scaled,
            };
        }
        let mut residual_noise = f64::NAN;
        if significance {
            // `out` holds the residual: in units of its local noise.
            work.clear();
            work.extend((0..len).filter(|&i| usable(i)).map(|i| out[i]));
            let global = robust_noise(&mut work).unwrap_or(0.0).max(1e-12) as f32;
            // The local baseline (median: EVs don't move it), then the
            // noise of the deviations from it.
            let baseline = local_median(out, input.size, LOCAL_NOISE_RADIUS as f64)?;
            for (o, b) in out.iter_mut().zip(&baseline) {
                *o -= b;
            }
            drop(baseline);
            let sigma = local_noise(out, w, h, LOCAL_NOISE_RADIUS, global);
            work.clear();
            work.extend((0..len).filter(|&i| usable(i)).map(|i| sigma[i]));
            residual_noise = if work.is_empty() {
                global as f64
            } else {
                median_in_place(&mut work) as f64
            };
            drop(work);
            for (o, s) in out.iter_mut().zip(&sigma) {
                let value = *o / s;
                *o = if self.clip_negative {
                    value.max(0.0)
                } else {
                    value
                };
            }
        }

        Ok(PlaneEstimate {
            a,
            b,
            dy,
            dx,
            quality,
            r2,
            fit_fraction,
            saturated_fraction: saturated as f64 / len as f64,
            fallback,
            residual_noise,
            target_sigma: target_sigma as f64,
            reference_sigma: reference_sigma as f64,
        })
    }

    /// Removes the background from `plane` (see [`Background`]).
    fn remove_background(
        &self,
        plane: &mut Vec<f32>,
        size: ImageSize,
        meta: &PipelineImageMeta,
    ) -> Result<(), InternalErrors> {
        match self.background {
            Background::None => Ok(()),
            Background::Median { radius } => {
                let background = median_background(plane, size, radius)?;
                for (v, b) in plane.iter_mut().zip(&background) {
                    *v -= b;
                }
                Ok(())
            }
            Background::RollingBall { radius } => {
                // Rolling Ball works on a context (it needs the bit depth).
                let image = ImageContainer::F32Gray(ManagedImage {
                    data: Image::new(size, std::mem::take(plane))
                        .map_err(InternalErrors::from_kornia)?,
                    tile_offset: Default::default(),
                    plane: None,
                    dimensionality: Default::default(),
                });
                let mut meta = meta.clone();
                meta.image_tile_info.width = size.width;
                meta.image_tile_info.height = size.height;
                let mut plane_ctx =
                    PipelineContext::new_from_image(Default::default(), meta, Arc::new(image))?;
                RollingBall {
                    radius: radius as f64,
                    ball_type: BallType::Ball,
                    pre_smooth: false,
                }
                .run(&mut plane_ctx, &mut GlobalPipelineCache::default())?;
                *plane = match Arc::try_unwrap(plane_ctx.image) {
                    Ok(ImageContainer::F32Gray(img)) => img.data.into_vec(),
                    Ok(_) => unreachable!("Rolling Ball keeps a grey image grey"),
                    Err(shared) => shared
                        .as_f32_slice()
                        .map(<[f32]>::to_vec)
                        .unwrap_or_default(),
                };
                Ok(())
            }
        }
    }
}

/// The saturation level on the image's values; `None` for none. Integer
/// images (`nr_of_bits` 1 - 31) are read scaled to 0 - 1 by their largest
/// gray value `2^bits − 1`: Auto is that maximum (half a gray value below
/// 1, so the scaling's rounding doesn't miss it) and a Manual level in gray
/// values is scaled alike (as Threshold's Bit unit). Float images keep
/// their values: Auto has no level, Manual is taken as it is.
fn saturation_level(saturation: Saturation, nr_of_bits: u16) -> Option<f32> {
    let largest = (1..32)
        .contains(&nr_of_bits)
        .then(|| ((1u64 << nr_of_bits) - 1) as f64);
    match (saturation, largest) {
        (Saturation::Off, _) | (Saturation::Auto, None) => None,
        (Saturation::Auto, Some(largest)) => Some((1.0 - 0.5 / largest) as f32),
        (Saturation::Manual { level }, Some(largest)) => Some((level as f64 / largest) as f32),
        (Saturation::Manual { level }, None) => Some(level),
    }
}

/// `mask` (`w × h`) grown by `reach` px in every direction (a square).
fn dilate(mask: &[bool], w: usize, h: usize, reach: usize) -> Vec<bool> {
    let mut rows = vec![false; mask.len()];
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        for (x, set) in row.iter().enumerate() {
            if *set {
                let (from, to) = (x.saturating_sub(reach), (x + reach).min(w - 1));
                rows[y * w + from..=y * w + to].fill(true);
            }
        }
    }
    let mut out = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            if rows[y * w + x] {
                for yy in y.saturating_sub(reach)..=(y + reach).min(h - 1) {
                    out[yy * w + x] = true;
                }
            }
        }
    }
    out
}

/// Blurs `plane` with the discrete Gaussian of `sigma` px (nothing for 0);
/// `work` and `tmp` are plane-sized buffers.
fn gaussian(
    plane: &mut Vec<f32>,
    work: &mut Vec<f32>,
    tmp: &mut [f32],
    size: ImageSize,
    sigma: f32,
) {
    if sigma <= 0.0 {
        return;
    }
    let taps = discrete_gaussian((sigma as f64) * (sigma as f64));
    convolve_separable(plane, work, tmp, size.width, size.height, &taps);
    std::mem::swap(plane, work);
}

/// The level above which the reference shows structure: its median plus
/// `noise_factor` × its noise (MAD), and at least its `min_percentile`, of
/// the usable pixels; `None` without any. `work`: a plane-sized buffer.
fn structure_level(
    t: &[f32],
    usable: &dyn Fn(usize) -> bool,
    work: &mut Vec<f32>,
    noise_factor: f32,
    min_percentile: f32,
) -> Option<f32> {
    work.clear();
    work.extend(
        (0..t.len())
            .filter(|&i| usable(i))
            .map(|i| t[i])
            .filter(|v| !v.is_nan()),
    );
    if work.is_empty() {
        return None;
    }
    let n = work.len();
    let rank = ((min_percentile as f64 / 100.0 * (n - 1) as f64).round() as usize).min(n - 1);
    let percentile = *work.select_nth_unstable_by(rank, f32::total_cmp).1;
    let median = median_in_place(work);
    for v in work.iter_mut() {
        *v = (*v - median).abs();
    }
    let noise = median_in_place(work) * MAD_TO_SIGMA;
    Some((median + noise_factor * noise).max(percentile))
}

/// From this radius on, the median background is estimated on a reduced
/// image (see [`median_background`]).
const REDUCE_FROM_RADIUS: f32 = 8.0;
/// The largest reduction (per side).
const MAX_REDUCTION: usize = 4;

/// The background of `plane` as the median over a disk of `radius` px. A
/// slowly varying background doesn't need every pixel: from
/// [`REDUCE_FROM_RADIUS`] on, the plane is reduced to the medians of `f × f`
/// blocks (`f = radius / 8`, at most 4), the median taken there with
/// radius `radius / f` and scaled back up bilinearly - about `f³` times
/// less work (a median's cost grows with its radius). Smaller radii: the
/// median of every pixel, as before.
fn median_background(
    plane: &[f32],
    size: ImageSize,
    radius: f32,
) -> Result<Vec<f32>, InternalErrors> {
    let factor = ((radius / REDUCE_FROM_RADIUS).floor() as usize).clamp(1, MAX_REDUCTION);
    if factor == 1 {
        let image = Image::from_size_slice(size, plane).map_err(InternalErrors::from_kornia)?;
        let background = RankFilter {
            radius: radius as f64,
            filter_type: RankFilterType::Median,
        }
        .process_image::<1>(&image)?;
        return Ok(background.into_vec());
    }
    let (w, h) = (size.width, size.height);
    let (sw, sh) = (w.div_ceil(factor), h.div_ceil(factor));
    // Block medians (robust to the structure and spots a block holds; the
    // last row and column of blocks may be smaller).
    let mut small = vec![0f32; sw * sh];
    let mut block = Vec::with_capacity(factor * factor);
    for by in 0..sh {
        for bx in 0..sw {
            block.clear();
            for y in by * factor..((by + 1) * factor).min(h) {
                block.extend_from_slice(
                    &plane[y * w + bx * factor..y * w + ((bx + 1) * factor).min(w)],
                );
            }
            small[by * sw + bx] = median_in_place(&mut block);
        }
    }
    let small = local_median(
        &small,
        ImageSize {
            width: sw,
            height: sh,
        },
        radius as f64 / factor as f64,
    )?;
    // Bilinear back to every pixel, between the blocks' centres.
    let axis = |i: usize, n: usize| -> (usize, usize, f32) {
        let u = ((i as f32 + 0.5) / factor as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let i0 = u.floor() as usize;
        (i0, (i0 + 1).min(n - 1), u - i0 as f32)
    };
    let columns: Vec<(usize, usize, f32)> = (0..w).map(|x| axis(x, sw)).collect();
    let mut background = vec![0f32; w * h];
    for y in 0..h {
        let (y0, y1, ty) = axis(y, sh);
        let (top, bottom) = (
            &small[y0 * sw..(y0 + 1) * sw],
            &small[y1 * sw..(y1 + 1) * sw],
        );
        for (x, &(x0, x1, tx)) in columns.iter().enumerate() {
            let upper = top[x0] + (top[x1] - top[x0]) * tx;
            let lower = bottom[x0] + (bottom[x1] - bottom[x0]) * tx;
            background[y * w + x] = upper + (lower - upper) * ty;
        }
    }
    Ok(background)
}

/// The median of `values` over Rank Filter's disk of `radius` px, the
/// values quantized to its 65536 gray levels between their minimum and
/// maximum: its sliding histogram then applies (the residual is no raw
/// image - for arbitrary values it would rank every window, about 20 times
/// slower for 21 × 21 px). Exact to (max − min) / 65535, a small part of
/// the noise.
fn local_median(values: &[f32], size: ImageSize, radius: f64) -> Result<Vec<f32>, InternalErrors> {
    let (low, high) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    // A flat plane (or no finite value): nothing to rank.
    if high.partial_cmp(&low) != Some(std::cmp::Ordering::Greater) {
        return Ok(values.to_vec());
    }
    let range = (high - low) as f64;
    // Level k is the value k / 65535 - the same bits for every pixel of a
    // level, as the histogram path requires.
    let levels: Vec<f32> = values
        .iter()
        .map(|v| {
            let k = ((((*v - low) as f64 / range) * 65535.0).round()).clamp(0.0, 65535.0);
            (k / 65535.0) as f32
        })
        .collect();
    let median = RankFilter {
        radius,
        filter_type: RankFilterType::Median,
    }
    .process_image::<1>(&Image::new(size, levels).map_err(InternalErrors::from_kornia)?)?;
    Ok(median
        .as_slice()
        .iter()
        .map(|q| (low as f64 + *q as f64 * range) as f32)
        .collect())
}

/// The local noise of the residual's deviations `values` (`w × h`) from
/// their local baseline over a `(2·radius + 1)²` window, from the negative
/// side only: `σ² = 2 · mean(min(d, 0)²)`. Noise is symmetric, EVs only
/// add positive values - so they don't raise their own noise, while the
/// photon noise of bright regions and the relief of leftovers (dark on one
/// side) do. At least half of `global`.
fn local_noise(values: &[f32], w: usize, h: usize, radius: usize, global: f32) -> Vec<f32> {
    let negative: Vec<f32> = values.iter().map(|v| v.min(0.0).powi(2)).collect();
    box_mean(&negative, w, h, radius)
        .into_iter()
        .map(|m| (2.0 * m).sqrt().max(0.5 * global))
        .collect()
}

/// The mean of `values` (`w × h`) over a `(2·radius + 1)²` window (smaller
/// at the border), with running sums in `f64`.
fn box_mean(values: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    // Rows.
    let mut rows = vec![0f32; values.len()];
    let mut prefix = vec![0f64; w + 1];
    for y in 0..h {
        let row = &values[y * w..(y + 1) * w];
        for (x, v) in row.iter().enumerate() {
            prefix[x + 1] = prefix[x] + *v as f64;
        }
        for x in 0..w {
            let (from, to) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            rows[y * w + x] = ((prefix[to] - prefix[from]) / (to - from) as f64) as f32;
        }
    }
    // Columns: a running sum of whole rows.
    let mut out = vec![0f32; values.len()];
    let mut sum = vec![0f64; w];
    let mut count = 0usize;
    let (mut top, mut bottom) = (0usize, 0usize);
    for y in 0..h {
        let (from, to) = (y.saturating_sub(radius), (y + radius + 1).min(h));
        while bottom < to {
            for (s, v) in sum.iter_mut().zip(&rows[bottom * w..(bottom + 1) * w]) {
                *s += *v as f64;
            }
            bottom += 1;
            count += 1;
        }
        while top < from {
            for (s, v) in sum.iter_mut().zip(&rows[top * w..(top + 1) * w]) {
                *s -= *v as f64;
            }
            top += 1;
            count -= 1;
        }
        for (o, s) in out[y * w..(y + 1) * w].iter_mut().zip(&sum) {
            *o = (s / count as f64) as f32;
        }
    }
    out
}

/// The robust noise (`1.4826 · MAD`) of `values`, which it reorders; `None`
/// for fewer than [`MIN_NOISE_PIXELS`].
fn robust_noise(values: &mut [f32]) -> Option<f64> {
    if values.len() < MIN_NOISE_PIXELS {
        return None;
    }
    let median = median_in_place(values);
    for v in values.iter_mut() {
        *v = (*v - median).abs();
    }
    Some((median_in_place(values) * MAD_TO_SIGMA) as f64)
}

/// The Huber fit of `s ≈ a·t + b` over the structure's pixels (see
/// [`Factor::RobustFit`]); `None` without any. `work`: a plane-sized
/// buffer.
fn robust_fit(
    s: &[f32],
    t: &[f32],
    usable: &dyn Fn(usize) -> bool,
    work: &mut Vec<f32>,
    params: FitParams,
) -> Option<Fit> {
    let FitParams {
        noise_factor,
        min_percentile,
        outlier_k,
        fit_offset,
        ..
    } = params;
    let level = structure_level(t, usable, work, noise_factor, min_percentile)?;

    let pairs: Vec<(f32, f32)> = (0..t.len())
        .filter(|&i| usable(i) && t[i] > level && t[i] > 0.0 && s[i].is_finite())
        .map(|i| (s[i], t[i]))
        .collect();
    if pairs.is_empty() {
        return None;
    }

    // Start: the median ratio (and the median offset left by it).
    work.clear();
    work.extend(pairs.iter().map(|(sv, tv)| sv / tv));
    let mut a = median_in_place(work) as f64;
    let mut b = 0.0f64;
    if fit_offset {
        work.clear();
        work.extend(pairs.iter().map(|(sv, tv)| sv - a as f32 * tv));
        b = median_in_place(work) as f64;
    }
    let scale = pairs.iter().map(|(sv, _)| sv.abs() as f64).sum::<f64>() / pairs.len() as f64;
    // The leverage limit: the reference's median and spread over the fit
    // pixels.
    work.clear();
    work.extend(pairs.iter().map(|(_, tv)| *tv));
    let t_median = median_in_place(work) as f64;
    for v in work.iter_mut() {
        *v = (*v as f64 - t_median).abs() as f32;
    }
    let t_reach = LEVERAGE_K * (median_in_place(work) * MAD_TO_SIGMA) as f64;
    let leverage = |tv: f64| {
        let distance = (tv - t_median).abs();
        if t_reach > 0.0 && distance > t_reach {
            t_reach / distance
        } else {
            1.0
        }
    };

    let mut settled = false;
    // `a` and `b` of the last rounds, newest last: a short cycle (pixels
    // flipping in and out at the cut) settles too.
    let mut history: Vec<(f64, f64)> = Vec::with_capacity(MAX_CYCLE);
    for _ in 0..MAX_ITERATIONS {
        // Robust sigma of the residuals.
        work.clear();
        work.extend(pairs.iter().map(|(sv, tv)| sv - (a as f32 * tv + b as f32)));
        let centre = median_in_place(work);
        for v in work.iter_mut() {
            *v = (*v - centre).abs();
        }
        let sigma = ((median_in_place(work) * MAD_TO_SIGMA) as f64).max(1e-12 * scale.max(1e-30));
        let cutoff = outlier_k as f64 * sigma;

        // Weighted least squares (Huber, leverage limited), the pixels far
        // off the line left out on both sides: EVs above it, dots in the
        // reference only below it.
        // Left out above only, those below tilt the line down, the
        // structure's brightest pixels then lie above it and are left out
        // in turn - the line keeps sinking towards a = 0.
        let (mut sw, mut st, mut ss, mut stt, mut sst) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        // Σ w², Σ w² t, Σ w² t²: the uncertainty of `a` (settling).
        let (mut sw2, mut sw2t, mut sw2tt) = (0.0f64, 0.0, 0.0);
        let mut kept = 0usize;
        for &(sv, tv) in &pairs {
            let (sv, tv) = (sv as f64, tv as f64);
            let residual = sv - (a * tv + b);
            if residual.abs() > cutoff {
                continue;
            }
            kept += 1;
            let u = (residual / sigma).abs();
            let weight = if u <= HUBER_K { 1.0 } else { HUBER_K / u } * leverage(tv);
            sw += weight;
            st += weight * tv;
            ss += weight * sv;
            stt += weight * tv * tv;
            sst += weight * sv * tv;
            sw2 += weight * weight;
            sw2t += weight * weight * tv;
            sw2tt += weight * weight * tv * tv;
        }
        if kept == 0 {
            return None;
        }
        let (new_a, new_b) = if fit_offset {
            let det = sw * stt - st * st;
            if det.abs() < 1e-300 {
                return None;
            }
            ((sw * sst - st * ss) / det, (stt * ss - st * sst) / det)
        } else {
            (sst / stt, 0.0)
        };
        // Settled once a round moves `a` by a small part of its own
        // uncertainty and `b` by a small part of the noise: a few pixels
        // flipping in and out at the cut keep changing both a little
        // forever, far below what could matter. Also settled when a round
        // comes back to the state of up to `MAX_CYCLE` rounds ago (a pixel
        // flipping in and out at the cut moves `a` back and forth by more
        // than that part of its uncertainty): then the mean of the cycle's
        // states.
        let c = if fit_offset { st / sw } else { 0.0 };
        let swd2 = stt - 2.0 * c * st + c * c * sw;
        let sw2d2 = (sw2tt - 2.0 * c * sw2t + c * c * sw2).max(0.0);
        let a_uncertainty = if swd2 > 0.0 {
            sigma * sw2d2.sqrt() / swd2
        } else {
            0.0
        };
        let close = |new_a: f64, new_b: f64, (a, b): (f64, f64)| {
            (new_a - a).abs()
                <= (TOLERANCE * new_a.abs().max(1e-12)).max(SETTLE_FRACTION * a_uncertainty)
                && (new_b - b).abs()
                    <= (TOLERANCE * new_b.abs().max(scale).max(1e-12)).max(SETTLE_FRACTION * sigma)
        };
        let done = close(new_a, new_b, (a, b));
        if history.len() == MAX_CYCLE {
            history.remove(0);
        }
        history.push((a, b));
        // The state the new one repeats (not the last: that is `done`).
        let repeated = (!done)
            .then(|| {
                history[..history.len() - 1]
                    .iter()
                    .rposition(|&before| close(new_a, new_b, before))
            })
            .flatten();
        if let Some(start) = repeated {
            // The repeated state and those after it.
            let cycle = &history[start..];
            let n = cycle.len() as f64;
            a = cycle.iter().map(|s| s.0).sum::<f64>() / n;
            b = cycle.iter().map(|s| s.1).sum::<f64>() / n;
            settled = true;
            break;
        }
        a = new_a;
        b = new_b;
        if done {
            settled = true;
            break;
        }
    }

    // R² over the pixels kept (those far off the line left out as in the
    // last step).
    work.clear();
    work.extend(pairs.iter().map(|(sv, tv)| sv - (a as f32 * tv + b as f32)));
    let centre = median_in_place(work);
    for v in work.iter_mut() {
        *v = (*v - centre).abs();
    }
    let cutoff = outlier_k as f64 * ((median_in_place(work) * MAD_TO_SIGMA) as f64).max(1e-30);
    let in_fit: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(sv, tv)| (sv as f64, tv as f64))
        .filter(|&(sv, tv)| (sv - (a * tv + b)).abs() <= cutoff)
        .collect();
    let mean = in_fit.iter().map(|(sv, _)| sv).sum::<f64>() / in_fit.len().max(1) as f64;
    let (mut residual, mut total) = (0.0f64, 0.0f64);
    for &(sv, tv) in &in_fit {
        residual += (sv - (a * tv + b)).powi(2);
        total += (sv - mean).powi(2);
    }
    let r2 = if total > 0.0 {
        1.0 - residual / total
    } else {
        0.0
    };

    // The uncertainty of `a` (weighted least squares with the leverage
    // weights): σ · √(Σ w² d²) / Σ w d², d the reference's distance from
    // its weighted mean (from 0 without an offset). Times the bright end of
    // the reference and in units of the noise σ, σ cancels.
    let centre = if fit_offset {
        let (sw, st) = in_fit.iter().fold((0.0, 0.0), |(sw, st), &(_, tv)| {
            let w = leverage(tv);
            (sw + w, st + w * tv)
        });
        st / sw.max(f64::MIN_POSITIVE)
    } else {
        0.0
    };
    let (swd2, sw2d2) = in_fit.iter().fold((0.0, 0.0), |(swd2, sw2d2), &(_, tv)| {
        let (w, d) = (leverage(tv), tv - centre);
        (swd2 + w * d * d, sw2d2 + w * w * d * d)
    });
    work.clear();
    work.extend(in_fit.iter().map(|&(_, tv)| tv as f32));
    let bright = if work.is_empty() {
        0.0
    } else {
        let rank = (work.len() - 1) * 99 / 100;
        *work.select_nth_unstable_by(rank, f32::total_cmp).1 as f64
    };
    let uncertainty = if swd2 > 0.0 {
        bright.abs() * sw2d2.sqrt() / swd2
    } else {
        f64::INFINITY
    };
    Some(Fit {
        a,
        b,
        r2,
        kept: in_fit.len(),
        settled,
        uncertainty,
    })
}

static VAN_DE_LEST_1995: CitationMetadata = CitationMetadata {
    cite_key: "vandelest1995elimination",
    title: "Elimination of autofluorescence in immunofluorescence microscopy with digital image processing",
    authors: &[
        "C. H. A. van de Lest",
        "E. M. M. Versteeg",
        "J. H. Veerkamp",
        "T. H. van Kuppevelt",
    ],
    year: 1995,
    container: Some("Journal of Histochemistry & Cytochemistry"),
    doi: Some("10.1177/43.7.7608528"),
    url: None,
    pages: Some("727-730"),
};

static ZIMMERMANN_2005: CitationMetadata = CitationMetadata {
    cite_key: "zimmermann2005spectral",
    title: "Spectral imaging and linear unmixing in light microscopy",
    authors: &["Timo Zimmermann"],
    year: 2005,
    container: Some("Advances in Biochemical Engineering/Biotechnology"),
    doi: None,
    url: None,
    pages: Some("245-265"),
};

static HUBER_1964: CitationMetadata = CitationMetadata {
    cite_key: "huber1964robust",
    title: "Robust estimation of a location parameter",
    authors: &["Peter J. Huber"],
    year: 1964,
    container: Some("The Annals of Mathematical Statistics"),
    doi: Some("10.1214/aoms/1177703732"),
    url: None,
    pages: Some("73-101"),
};

static HAMPEL_1986: CitationMetadata = CitationMetadata {
    cite_key: "hampel1986robust",
    title: "Robust Statistics: The Approach Based on Influence Functions",
    authors: &[
        "Frank R. Hampel",
        "Elvezio M. Ronchetti",
        "Peter J. Rousseeuw",
        "Werner A. Stahel",
    ],
    year: 1986,
    container: Some("John Wiley & Sons"),
    doi: Some("10.1002/9781118186435"),
    url: None,
    pages: None,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The step card: what to set first without a section, the rest in
    /// sections - in this order.
    #[test]
    fn the_settings_are_in_sections() {
        let command: evanalyzer_cfg::settings::pipeline_command::PipelineCommand =
            serde_json::from_str(
                r#"{ "type": "SUBTRACT_REFERENCE_CHANNEL", "id": "reference", "referenceChannel": 4 }"#,
            )
            .unwrap();
        // A choice's own settings ("factor.outlier_k") go with it.
        let mut sections: Vec<(String, Vec<String>)> = Vec::new();
        for def in command.to_parameters() {
            if def.name.contains('.') {
                continue;
            }
            match sections.last_mut() {
                Some((section, names)) if *section == def.section => names.push(def.name),
                _ => sections.push((def.section, vec![def.name])),
            }
        }
        let names = |list: &[&str]| list.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            sections,
            vec![
                (
                    String::new(),
                    names(&["target", "reference_channel", "output"])
                ),
                ("Alignment".into(), names(&["alignment", "interpolation"])),
                (
                    "Background & smoothing".into(),
                    names(&[
                        "background",
                        "smoothing_sigma",
                        "sharpness",
                        "target_extra_sigma",
                        "reference_extra_sigma",
                        "camera_offset",
                        "saturation"
                    ])
                ),
                (
                    "Factor".into(),
                    names(&[
                        "factor",
                        "factor_scale",
                        "id",
                        "estimation_scope",
                        "edge_tolerance"
                    ])
                ),
                (
                    "Subtraction".into(),
                    names(&["only_brighter_reference", "clip_negative"])
                ),
            ]
        );
    }

    /// Images are read scaled to 0 - 1 by their largest gray value: Auto
    /// is that maximum and a Manual level in gray values is scaled the
    /// same; float images keep their values.
    #[test]
    fn the_saturation_level_is_on_the_scaled_values() {
        // A 12-bit camera's 4095 is 1.0 when read; 4094 is not saturated.
        let auto = saturation_level(Saturation::Auto, 12).unwrap();
        assert!(1.0 >= auto && 4094.0 / 4095.0 < auto, "{auto}");
        let auto16 = saturation_level(Saturation::Auto, 16).unwrap();
        assert!(1.0 >= auto16 && 65534.0 / 65535.0 < auto16, "{auto16}");
        assert_eq!(
            saturation_level(Saturation::Manual { level: 4000.0 }, 12),
            Some((4000.0f64 / 4095.0) as f32)
        );
        // Float images (32 bit), and no bit depth: as stored.
        assert_eq!(saturation_level(Saturation::Auto, 32), None);
        assert_eq!(
            saturation_level(Saturation::Manual { level: 7.5 }, 32),
            Some(7.5)
        );
        assert_eq!(
            saturation_level(Saturation::Manual { level: 7.5 }, 0),
            Some(7.5)
        );
        assert_eq!(saturation_level(Saturation::Off, 16), None);
    }

    /// A pixel flipping in and out at the outlier cut moves the line back
    /// and forth by more than the settling tolerance: such a fit settles
    /// too (between both states) instead of being taken for one that
    /// didn't - with the fallback factor. Noisy lines with EVs above and
    /// dots below them, many random draws: every fit settles.
    #[test]
    fn a_fit_flipping_one_pixel_at_the_cut_settles() {
        use crate::algos::libs::phase_correlation::test_support::Lcg;
        let params = FitParams {
            noise_factor: 4.0,
            min_percentile: 50.0,
            outlier_k: 3.0,
            fit_offset: true,
            max_uncertainty: 0.25,
            fallback: 0.0,
            min_value: 0.0,
            max_value: 100.0,
        };
        let n = 4000;
        let mut work = Vec::new();
        for seed in 0..300u64 {
            let mut rng = Lcg(seed);
            let (mut s, mut t) = (vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                let structure = 50.0 * rng.next() * rng.next();
                t[i] = structure + 0.5 * rng.normal();
                s[i] = 0.4 * structure + 3.0 + 1.5 * rng.normal();
                // EVs above the line, dots in the reference only below it.
                match (rng.next() * 50.0) as u32 {
                    0 => s[i] += 15.0 + 20.0 * rng.next(),
                    1 => t[i] += 20.0 + 30.0 * rng.next(),
                    _ => {}
                }
            }
            let fit = robust_fit(&s, &t, &|_| true, &mut work, params).unwrap();
            assert!(fit.settled, "seed {seed}: did not settle (a {:.4})", fit.a);
        }
    }

    /// The reduced median background follows the full-resolution one: on
    /// a ramp with spots and a filament it stays within 2 % of the ramp's
    /// height of the median of every pixel.
    #[test]
    fn the_reduced_background_matches_the_full_median() {
        let (w, h) = (200usize, 160usize);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mut plane: Vec<f32> = (0..w * h)
            .map(|i| 100.0 + 0.5 * (i % w) as f32 + 0.3 * (i / w) as f32)
            .collect();
        // Spots every 17 px and a filament along y = 80.
        for y in (8..h).step_by(17) {
            for x in (8..w).step_by(17) {
                plane[y * w + x] += 400.0;
            }
        }
        for x in 0..w {
            plane[80 * w + x] += 200.0;
        }
        let full = RankFilter {
            radius: 30.0,
            filter_type: RankFilterType::Median,
        }
        .process_image::<1>(&Image::from_size_slice(size, &plane).unwrap())
        .unwrap()
        .into_vec();
        let reduced = median_background(&plane, size, 30.0).unwrap();
        // The ramp rises 0.5 · 200 + 0.3 · 160 = 148 across the plane; away
        // from the border (where the disks differ most).
        let worst = (30..h - 30)
            .flat_map(|y| (30..w - 30).map(move |x| y * w + x))
            .map(|i| (reduced[i] - full[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02 * 148.0, "largest difference {worst:.2}");
    }

    /// Below 8 px nothing is reduced: the median of every pixel.
    #[test]
    fn a_small_radius_keeps_every_pixel() {
        let size = ImageSize {
            width: 40,
            height: 30,
        };
        let plane: Vec<f32> = (0..40 * 30).map(|i| ((i * 37) % 101) as f32).collect();
        let full = RankFilter {
            radius: 5.0,
            filter_type: RankFilterType::Median,
        }
        .process_image::<1>(&Image::from_size_slice(size, &plane).unwrap())
        .unwrap()
        .into_vec();
        assert_eq!(median_background(&plane, size, 5.0).unwrap(), full);
    }
}
