//! # illumination_correction
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-23
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;
use std::sync::Arc;

/// How the illumination field is estimated from the block-reduced image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionMethod {
    /// Block *mean*. The right default for typical uneven illumination
    /// (vignetting, uneven excitation): every block contributes its average
    /// brightness to the estimated field.
    Regular,

    /// Block *minimum*. Use this when dense or bright foreground objects
    /// would otherwise pull the mean-based estimate upward - the minimum
    /// hugs the true background floor instead.
    Background,
}

/// Smoothing applied to the block-reduced field to remove blockiness before
/// it is used as the correction field.
#[derive(Debug, Clone, Copy, PartialEq, CommandsMeta)]
pub enum SmoothingMethod {
    /// Use the block-reduced field as-is.
    None,

    /// Separable Gaussian blur of the block grid.
    Gaussian {
        /// Standard deviation, in block-grid units.
        #[cmdsmeta(default = 2.0, min = 0.1, max = 20.0, step = 0.1)]
        sigma: f32,
    },

    /// Windowed median filter of the block grid.
    Median {
        /// Neighborhood radius, in block-grid units.
        #[cmdsmeta(default = 2, min = 1, max = 20, step = 1)]
        radius: usize,
    },

    /// Fits a smooth 2nd-order polynomial surface
    /// (a + bx + cy + dx² + exy + fy²) through the block grid. Has no
    /// tunable radius/sigma, so it's the most stable option when unsure.
    FitPolynomial,
}

/// How the estimated illumination field is combined with the original image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyMethod {
    /// `corrected = image * mean(field) / field`. Multiplicative correction
    /// that preserves overall image brightness - the right default for
    /// gain/vignetting-style illumination problems.
    Divide,

    /// `corrected = image - (field - mean(field))`. Additive correction.
    Subtract,
}

/// Use this when your images are brighter in the middle and dimmer toward
/// the edges/corners (vignetting), or show any other smooth shading pattern
/// that repeats the same way across every tile or every image from the same
/// microscope/camera setup - a consequence of the optics or illumination,
/// not the sample. Left uncorrected, that shading makes intensity
/// comparisons between regions of an image (or between images/wells)
/// unreliable, even though it rarely stops segmentation from finding
/// objects on its own.
///
/// Use [`super::rolling_ball::RollingBall`] instead when the problem is a
/// *local* background glow or halo under/around individual objects (e.g.
/// out-of-focus light, autofluorescence, uneven staining) that differs from
/// image to image rather than being tied to the acquisition setup -
/// RollingBall strips that local floor so thresholding/segmentation works
/// cleanly. The two solve different problems: RollingBall won't fix a
/// global brightness gradient, and this filter won't remove a local halo.
///
/// ### How it works
///
/// Flat-field ("illumination") correction: estimates a smooth, slowly-varying
/// gain/offset field caused by uneven illumination (vignetting, dust on the
/// condenser, uneven excitation) and removes it in a single calculate+apply
/// step - equivalent to CellProfiler's `CorrectIlluminationCalculate` and
/// `CorrectIlluminationApply` modules combined into one.
///
/// Unlike `RollingBall`, which estimates a *local* per-object background
/// baseline via a rolling structural element, this estimates one *global*,
/// low-frequency field for the whole image/channel.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct IlluminationCorrection {
    /// How the illumination field is estimated from the image.
    pub method: CorrectionMethod,

    /// Block size, in pixels, used to reduce the image to a coarse
    /// illumination estimate before smoothing. Should be larger than the
    /// largest foreground object, so objects are averaged/eroded away and
    /// only the slow-varying illumination trend survives.
    #[cmdsmeta(default = 60, min = 1, max = 2000, step = 1)]
    pub block_size: usize,

    /// Smoothing applied to the block-reduced field to remove blockiness.
    pub smoothing: SmoothingMethod,

    /// How the field is combined with the original image.
    pub apply_method: ApplyMethod,

    /// Stretch the corrected image's intensities to fill the full
    /// `[0.0, 1.0]` range afterward - guards against `Divide` pushing
    /// previously-dim regions above `1.0`.
    pub rescale: bool,
}

impl ImageAlgorithm for IlluminationCorrection {
    /// Estimates and removes a smooth illumination field from the image.
    ///
    /// ### Workflow
    /// 1. **Block reduce**: The image is reduced to a coarse grid, one value
    ///    per `block_size` x `block_size` block (mean or min, per `method`).
    /// 2. **Smooth**: The block grid is smoothed per `smoothing` to remove
    ///    blockiness/noise.
    /// 3. **Enlarge**: The smoothed grid is bilinearly upscaled back to the
    ///    original resolution, producing the full-size illumination field.
    /// 4. **Apply**: The field is divided out or subtracted per
    ///    `apply_method`, and optionally rescaled.
    ///
    /// For RGB images each channel is corrected independently, since
    /// illumination unevenness (vignetting, filter/dichroic effects) is
    /// frequently channel/wavelength dependent.
    ///
    /// # Errors
    /// Returns [`InternalErrors::Generic`] if `block_size` is `0` or the
    /// selected smoothing parameters are non-positive, and
    /// [`InternalErrors::FormatMismatch`] if the input image is not in
    /// `F32Gray` or `F32Rgb` format.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.block_size == 0 {
            return Err(InternalErrors::Generic(
                "IlluminationCorrection requires block_size >= 1".into(),
            ));
        }

        match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
            (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                let (width, height) = (input.width(), input.height());
                output
                    .data
                    .as_slice_mut()
                    .copy_from_slice(input.data.as_slice());
                self.correct_channel(output.data.as_slice_mut(), width, height)?;
                ctx.swap()?;
                Ok(())
            }
            (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                let (width, height) = (input.width(), input.height());
                let pixels = width * height;
                output
                    .data
                    .as_slice_mut()
                    .copy_from_slice(input.data.as_slice());
                let slice = output.data.as_slice_mut();
                let mut channel = vec![0.0f32; pixels];
                for c in 0..3 {
                    for i in 0..pixels {
                        channel[i] = slice[i * 3 + c];
                    }
                    self.correct_channel(&mut channel, width, height)?;
                    for i in 0..pixels {
                        slice[i * 3 + c] = channel[i];
                    }
                }
                ctx.swap()?;
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Rgb or F32Gray".into(),
                found: format!("{:?}", ctx.image),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Illumination Correction"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl IlluminationCorrection {
    /// Runs the full calculate+apply pipeline on a single flat channel
    /// buffer (`width * height` values), in place.
    fn correct_channel(
        &self,
        data: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<(), InternalErrors> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        let (bw, bh, mut field) = self.block_reduce(data, width, height);
        self.smooth_field(&mut field, bw, bh)?;
        let full_field = Self::enlarge_field(&field, bw, bh, width, height, self.block_size);

        let mean_field = {
            let sum: f64 = full_field.iter().map(|&v| v as f64).sum();
            (sum / full_field.len() as f64) as f32
        };
        // A field that averages out to ~0 (e.g. an all-black image) would
        // otherwise send Divide's `mean_field / f` to zero/NaN territory.
        let mean_field = if mean_field.abs() > 1e-6 {
            mean_field
        } else {
            1e-6
        };

        match self.apply_method {
            ApplyMethod::Divide => {
                for (d, f) in data.iter_mut().zip(full_field.iter()) {
                    let f = f.max(1e-6);
                    *d = (*d * mean_field / f).max(0.0);
                }
            }
            ApplyMethod::Subtract => {
                for (d, f) in data.iter_mut().zip(full_field.iter()) {
                    *d = (*d - (f - mean_field)).max(0.0);
                }
            }
        }

        if self.rescale {
            let max_val = data.iter().fold(0.0f32, |a, &b| a.max(b));
            if max_val > 1e-6 {
                let scale = 1.0 / max_val;
                for d in data.iter_mut() {
                    *d *= scale;
                }
            }
        }

        Ok(())
    }

    /// Reduces `data` to a `bw` x `bh` grid, one value per `block_size` x
    /// `block_size` block (mean or min, per `self.method`). The last row/
    /// column of blocks is clipped to the image bounds rather than padded.
    fn block_reduce(&self, data: &[f32], width: usize, height: usize) -> (usize, usize, Vec<f32>) {
        let block = self.block_size;
        let bw = width.div_ceil(block);
        let bh = height.div_ceil(block);
        let mut out = vec![0.0f32; bw * bh];

        for by in 0..bh {
            let y0 = by * block;
            let y1 = (y0 + block).min(height);
            for bx in 0..bw {
                let x0 = bx * block;
                let x1 = (x0 + block).min(width);

                out[by * bw + bx] = match self.method {
                    CorrectionMethod::Regular => {
                        let mut sum = 0.0f64;
                        let mut count = 0usize;
                        for y in y0..y1 {
                            let row = y * width;
                            for x in x0..x1 {
                                sum += data[row + x] as f64;
                                count += 1;
                            }
                        }
                        (sum / count.max(1) as f64) as f32
                    }
                    CorrectionMethod::Background => {
                        let mut min_v = f32::MAX;
                        for y in y0..y1 {
                            let row = y * width;
                            for x in x0..x1 {
                                min_v = min_v.min(data[row + x]);
                            }
                        }
                        if min_v == f32::MAX { 0.0 } else { min_v }
                    }
                };
            }
        }
        (bw, bh, out)
    }

    /// Smooths the block grid in place, per `self.smoothing`.
    fn smooth_field(&self, field: &mut [f32], bw: usize, bh: usize) -> Result<(), InternalErrors> {
        match self.smoothing {
            SmoothingMethod::None => {}
            SmoothingMethod::Gaussian { sigma } => {
                if !(sigma > 0.0) {
                    return Err(InternalErrors::Generic(
                        "IlluminationCorrection requires a positive Gaussian sigma".into(),
                    ));
                }
                Self::gaussian_smooth_grid(field, bw, bh, sigma);
            }
            SmoothingMethod::Median { radius } => {
                if radius == 0 {
                    return Err(InternalErrors::Generic(
                        "IlluminationCorrection requires median radius >= 1".into(),
                    ));
                }
                Self::median_smooth_grid(field, bw, bh, radius);
            }
            SmoothingMethod::FitPolynomial => {
                Self::fit_polynomial_grid(field, bw, bh);
            }
        }
        Ok(())
    }

    /// Separable Gaussian blur with edge-clamped (replicate) borders.
    fn gaussian_smooth_grid(field: &mut [f32], bw: usize, bh: usize, sigma: f32) {
        if bw == 0 || bh == 0 {
            return;
        }
        let radius = ((3.0 * sigma).ceil() as usize).max(1);
        let mut kernel = vec![0.0f32; 2 * radius + 1];
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0f32;
        for (i, k) in kernel.iter_mut().enumerate() {
            let d = i as f32 - radius as f32;
            *k = (-(d * d) / two_sigma_sq).exp();
            sum += *k;
        }
        for k in kernel.iter_mut() {
            *k /= sum;
        }

        let mut temp = vec![0.0f32; field.len()];
        for y in 0..bh {
            let row = y * bw;
            for x in 0..bw {
                let mut acc = 0.0f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let sx = (x as isize + i as isize - radius as isize).clamp(0, bw as isize - 1)
                        as usize;
                    acc += field[row + sx] * k;
                }
                temp[row + x] = acc;
            }
        }
        for x in 0..bw {
            for y in 0..bh {
                let mut acc = 0.0f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let sy = (y as isize + i as isize - radius as isize).clamp(0, bh as isize - 1)
                        as usize;
                    acc += temp[sy * bw + x] * k;
                }
                field[y * bw + x] = acc;
            }
        }
    }

    /// Windowed median filter over a square `2*radius+1` neighborhood,
    /// clamped at the grid edges.
    fn median_smooth_grid(field: &mut [f32], bw: usize, bh: usize, radius: usize) {
        if bw == 0 || bh == 0 {
            return;
        }
        let src = field.to_vec();
        let mut window = Vec::with_capacity((2 * radius + 1) * (2 * radius + 1));
        for y in 0..bh {
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(bh - 1);
            for x in 0..bw {
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(bw - 1);

                window.clear();
                for yy in y0..=y1 {
                    let row = yy * bw;
                    for xx in x0..=x1 {
                        window.push(src[row + xx]);
                    }
                }
                window.sort_by(|a, b| a.partial_cmp(b).unwrap());
                field[y * bw + x] = window[window.len() / 2];
            }
        }
    }

    /// Replaces the block grid with a least-squares fit of a 2nd-order
    /// polynomial surface `a + bx + cy + dx² + exy + fy²` through it. `x`/`y`
    /// are normalized to `[-1, 1]` for numerical stability. If the grid is
    /// degenerate (e.g. a single block), the unsmoothed grid is left as-is.
    fn fit_polynomial_grid(field: &mut [f32], bw: usize, bh: usize) {
        if bw == 0 || bh == 0 {
            return;
        }
        let norm_x = |x: usize| -> f64 {
            if bw <= 1 {
                0.0
            } else {
                2.0 * x as f64 / (bw - 1) as f64 - 1.0
            }
        };
        let norm_y = |y: usize| -> f64 {
            if bh <= 1 {
                0.0
            } else {
                2.0 * y as f64 / (bh - 1) as f64 - 1.0
            }
        };

        let mut ata = [[0.0f64; 6]; 6];
        let mut atb = [0.0f64; 6];
        for y in 0..bh {
            let ny = norm_y(y);
            for x in 0..bw {
                let nx = norm_x(x);
                let basis = [1.0, nx, ny, nx * nx, nx * ny, ny * ny];
                let z = field[y * bw + x] as f64;
                for i in 0..6 {
                    atb[i] += basis[i] * z;
                    for j in 0..6 {
                        ata[i][j] += basis[i] * basis[j];
                    }
                }
            }
        }

        let Some(coeffs) = Self::solve_6x6(ata, atb) else {
            return;
        };

        for y in 0..bh {
            let ny = norm_y(y);
            for x in 0..bw {
                let nx = norm_x(x);
                let basis = [1.0, nx, ny, nx * nx, nx * ny, ny * ny];
                let z: f64 = basis.iter().zip(coeffs.iter()).map(|(b, c)| b * c).sum();
                field[y * bw + x] = z as f32;
            }
        }
    }

    /// Solves the 6x6 linear system `a * x = b` via Gauss-Jordan elimination
    /// with partial pivoting. Returns `None` if `a` is singular/near-singular.
    fn solve_6x6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
        for col in 0..6 {
            let mut pivot_row = col;
            let mut pivot_val = a[col][col].abs();
            for row in (col + 1)..6 {
                if a[row][col].abs() > pivot_val {
                    pivot_val = a[row][col].abs();
                    pivot_row = row;
                }
            }
            if pivot_val < 1e-10 {
                return None;
            }
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);

            let diag = a[col][col];
            for row in 0..6 {
                if row == col {
                    continue;
                }
                let factor = a[row][col] / diag;
                if factor == 0.0 {
                    continue;
                }
                for k in col..6 {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
        let mut result = [0.0f64; 6];
        for i in 0..6 {
            result[i] = b[i] / a[i][i];
        }
        Some(result)
    }

    /// Bilinearly upscales a `bw` x `bh` block grid (each block's sample
    /// treated as sitting at its center) to the full `width` x `height`
    /// resolution.
    fn enlarge_field(
        field: &[f32],
        bw: usize,
        bh: usize,
        width: usize,
        height: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let (x_idx, x_w) = Self::build_interp(width, block_size, bw);
        let (y_idx, y_w) = Self::build_interp(height, block_size, bh);

        let mut out = vec![0.0f32; width * height];
        for y in 0..height {
            let y0 = y_idx[y];
            let y1 = (y0 + 1).min(bh - 1);
            let wy = y_w[y];
            let row0 = y0 * bw;
            let row1 = y1 * bw;
            let out_row = y * width;

            for x in 0..width {
                let x0 = x_idx[x];
                let x1 = (x0 + 1).min(bw - 1);
                let wx = x_w[x];

                let v00 = field[row0 + x0];
                let v10 = field[row0 + x1];
                let v01 = field[row1 + x0];
                let v11 = field[row1 + x1];

                out[out_row + x] = v00 * (1.0 - wx) * (1.0 - wy)
                    + v10 * wx * (1.0 - wy)
                    + v01 * (1.0 - wx) * wy
                    + v11 * wx * wy;
            }
        }
        out
    }

    /// For each of `length` full-resolution positions, finds the
    /// block-grid index immediately to its left/top and the fractional
    /// distance to the next block's center - the standard pixel-center
    /// bilinear-resize mapping, with `n_blocks` samples spaced `block_size`
    /// pixels apart.
    fn build_interp(length: usize, block_size: usize, n_blocks: usize) -> (Vec<usize>, Vec<f32>) {
        let bs = block_size as f32;
        let max_pos = (n_blocks as f32 - 1.0).max(0.0);
        let mut idx = vec![0usize; length];
        let mut w = vec![0.0f32; length];
        for i in 0..length {
            let pos = ((i as f32 + 0.5) / bs - 0.5).clamp(0.0, max_pos);
            let base = (pos.floor() as usize).min(n_blocks.saturating_sub(1));
            idx[i] = base;
            w[i] = pos - base as f32;
        }
        (idx, w)
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageContainer, image::PixelSizes, pipeline::pipeline::PipelineImageMeta};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;
    use std::path::PathBuf;

    fn ctx_from_gray(width: usize, height: usize, data: Vec<f32>) -> PipelineContext {
        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width,
                    height,
                },
                full_image_width: ImageSize { width, height },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_gray_from_image_test(image).into(),
        )
        .unwrap()
    }

    #[test]
    fn divide_flattens_a_vignetting_style_gradient_while_keeping_a_signal_peak() {
        let width = 60;
        let height = 60;
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                // Smooth multiplicative gain field: dim at the edges, bright
                // in the middle - the classic vignetting shape.
                let dx = (x as f32 - 30.0) / 30.0;
                let dy = (y as f32 - 30.0) / 30.0;
                let gain = (1.0 - 0.5 * (dx * dx + dy * dy)).clamp(0.2, 1.0);
                data[y * width + x] = 0.3 * gain;
            }
        }
        // A small bright "cell" signal near a dim corner.
        data[5 * width + 5] = 0.9;

        let mut ctx = ctx_from_gray(width, height, data);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            // A finer grid (10x10) than the vignetting scale (30px) lets the
            // Gaussian smoothing (sigma 1.0, radius 3 blocks) still track the
            // curvature instead of averaging the whole grid into one value.
            block_size: 6,
            smoothing: SmoothingMethod::Gaussian { sigma: 1.0 },
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let out = out.as_slice();

        let center = out[30 * width + 30];
        let corner = out[5 * width + 5];
        let far_corner = out[1 * width + 1];

        // The background trend should have been flattened: center and
        // background-corner values should be much closer together than the
        // ~3x gap (0.3 vs 0.3*0.2) in the raw input.
        assert!(
            (center - far_corner).abs() < 0.1,
            "background not flattened: center={center}, far_corner={far_corner}"
        );
        // The signal spike must remain clearly elevated above its local
        // background after correction.
        assert!(
            corner > far_corner * 1.5,
            "signal peak was washed out: corner={corner}, far_corner={far_corner}"
        );
    }

    #[test]
    fn rgb_channels_are_corrected_independently() {
        let width = 40;
        let height = 40;
        let mut data = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Red: bright-left/dim-right gradient. Green: flat.
                data[idx] = if x < width / 2 { 0.8 } else { 0.2 };
                data[idx + 1] = 0.5;
            }
        }

        let image =
            Image::<f32, 3, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)
                .unwrap();
        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width,
                    height,
                },
                full_image_width: ImageSize { width, height },
                is_rgb: true,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_rgb_from_image_test(image).into(),
        )
        .unwrap();
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            // A finer grid (10x10) keeps the Gaussian smoothing (sigma 1.0,
            // radius 3 blocks) local enough to still resolve the left/right
            // step instead of averaging the whole grid into one value.
            block_size: 4,
            smoothing: SmoothingMethod::Gaussian { sigma: 1.0 },
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Rgb(out) = ctx.image.as_ref() else {
            panic!("expected F32Rgb output");
        };
        let out = out.as_slice();
        let left_idx = (20 * width + 5) * 3;
        let right_idx = (20 * width + 35) * 3;

        // Red had a real gradient: correction should pull left/right much
        // closer together than the raw 0.8 vs 0.2 gap.
        assert!(
            (out[left_idx] - out[right_idx]).abs() < 0.15,
            "red channel gradient not corrected: left={}, right={}",
            out[left_idx],
            out[right_idx]
        );
        // Green was already flat: correction should leave it close to its
        // original value, unaffected by red's correction.
        assert!(
            (out[left_idx + 1] - 0.5).abs() < 0.05,
            "green channel should be left near-untouched: {}",
            out[left_idx + 1]
        );
    }

    #[test]
    fn subtract_removes_an_additive_offset_and_clamps_at_zero() {
        let width = 30;
        let height = 30;
        let mut data = vec![0.1f32; width * height];
        for y in 0..height {
            for x in 0..width {
                // Additive offset ramp instead of a multiplicative gain.
                data[y * width + x] += (x as f32 / width as f32) * 0.4;
            }
        }

        let mut ctx = ctx_from_gray(width, height, data);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 6,
            smoothing: SmoothingMethod::None,
            apply_method: ApplyMethod::Subtract,
            rescale: false,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let out = out.as_slice();
        for &v in out {
            assert!(v >= 0.0, "Subtract must clamp negative overflow, got {v}");
        }
        let left = out[15 * width + 1];
        let right = out[15 * width + 28];
        assert!(
            (left - right).abs() < 0.1,
            "additive ramp not removed: left={left}, right={right}"
        );
    }

    #[test]
    fn rescale_stretches_output_to_fill_0_1() {
        let width = 20;
        let height = 20;
        let mut data = vec![0.05f32; width * height];
        data[10 * width + 10] = 0.2;

        let mut ctx = ctx_from_gray(width, height, data);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 4,
            smoothing: SmoothingMethod::None,
            apply_method: ApplyMethod::Divide,
            rescale: true,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let max_val = out.as_slice().iter().fold(0.0f32, |a, &b| a.max(b));
        assert!(
            (max_val - 1.0).abs() < 1e-4,
            "rescale should stretch the max value to 1.0, got {max_val}"
        );
    }

    #[test]
    fn fit_polynomial_smoothing_flattens_a_smooth_gradient() {
        let width = 50;
        let height = 50;
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = 0.1 + 0.3 * (x as f32 / width as f32);
            }
        }

        let mut ctx = ctx_from_gray(width, height, data);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 5,
            smoothing: SmoothingMethod::FitPolynomial,
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let out = out.as_slice();
        let left = out[25 * width + 2];
        let right = out[25 * width + 47];
        assert!(
            (left - right).abs() < 0.1,
            "polynomial-smoothed field should flatten the linear gradient: left={left}, right={right}"
        );
    }

    #[test]
    fn zero_block_size_returns_error() {
        let mut ctx = ctx_from_gray(10, 10, vec![0.5f32; 100]);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 0,
            smoothing: SmoothingMethod::None,
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        assert!(cmd.execute(&mut ctx, &mut cache).is_err());
    }

    #[test]
    fn non_positive_gaussian_sigma_returns_error() {
        let mut ctx = ctx_from_gray(10, 10, vec![0.5f32; 100]);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 4,
            smoothing: SmoothingMethod::Gaussian { sigma: 0.0 },
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        assert!(cmd.execute(&mut ctx, &mut cache).is_err());
    }

    #[test]
    fn zero_median_radius_returns_error() {
        let mut ctx = ctx_from_gray(10, 10, vec![0.5f32; 100]);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Background,
            block_size: 4,
            smoothing: SmoothingMethod::Median { radius: 0 },
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        assert!(cmd.execute(&mut ctx, &mut cache).is_err());
    }

    #[test]
    fn flat_image_is_left_unchanged_by_divide() {
        let width = 16;
        let height = 16;
        let data = vec![0.4f32; width * height];

        let mut ctx = ctx_from_gray(width, height, data);
        let mut cache = GlobalPipelineCache::default();
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 4,
            smoothing: SmoothingMethod::None,
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        cmd.execute(&mut ctx, &mut cache).expect("execution failed");

        let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        for &v in out.as_slice() {
            assert!(
                (v - 0.4).abs() < 1e-4,
                "flat image should stay flat, got {v}"
            );
        }
    }

    #[test]
    fn illumination_correction_metadata() {
        let cmd = IlluminationCorrection {
            method: CorrectionMethod::Regular,
            block_size: 60,
            smoothing: SmoothingMethod::None,
            apply_method: ApplyMethod::Divide,
            rescale: false,
        };
        assert_eq!(cmd.name(), "Illumination Correction");
    }
}
