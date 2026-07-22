//! # rolling_ball
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::ImageAlgorithm;
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::pipeline::pipeline_context::PipelineContext;
use evanalyzer_cfg::core_types::InternalErrors;
use macros::CommandsMeta;

/// Removes non-uniform background illumination by calculating a local intensity baseline.
///
/// This algorithm models the image as a 3D intensity landscape and conceptually rolls
/// a sphere of a user-defined radius underneath it. The ball cannot penetrate narrow
/// intensity peaks (true signal objects) but follows the sweeping, lower-frequency
/// curves of background variations. The path traced by the ball establishes a local
/// baseline map that is subtracted from the original image to isolate foreground features.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct RollingBall {
    /// The radius of the ball or paraboloid in pixels.
    ///
    /// This should be at least as large as the radius of the largest
    /// object in the image that is not part of the background.
    #[cmdsmeta(default = 4, min = 1, max = 64, step = 1)]
    pub radius: f64,

    /// The geometric shape of the rolling structural element.
    pub ball_type: BallType,

    pub pre_smooth: bool,
}

/// The geometric shape used to probe the image intensity surface.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BallType {
    /// A spherical cap.
    ///
    /// This is the traditional ImageJ algorithm. Best for images with
    /// distinct, round features like cells or particles.
    Ball,

    /// A sliding parabolic surface.
    ///
    /// Mathematically smoother at the edges than the Ball, often
    /// resulting in fewer artifacts on complex gradients.
    Paraboloid,
}

impl ImageAlgorithm for RollingBall {
    /// Executes the Rolling Ball background subtraction algorithm.
    ///
    /// This implementation follows a multi-stage pipeline:
    /// 1. **Pre-smoothing**: Reduces high-frequency noise using a separable 3x3 mean filter.
    /// 2. **Downsampling (Shrink)**: If the radius is large, the image is downsampled to
    ///    speed up the $O(N^2 \cdot M^2)$ rolling operation.
    /// 3. **Rolling**: The core morphological "opening" operation where the ball surface
    ///    is computed as an envelope under the image.
    /// 4. **Interpolation (Enlarge)**: The background is scaled back to original dimensions
    ///    using bilinear interpolation.
    /// 5. **Subtraction**: The computed background is subtracted from the original signal.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::FormatMismatch`] if the input image is not in `F32Gray` format.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        let meta = &ctx.image_meta;
        // Dynamically compute the maximum value scale based on the original bit depth
        // (e.g., 255.0 for 8-bit, 65535.0 for 16-bit).
        let max_intensity = (1u64 << meta.nr_of_bits) - 1;
        let scale_factor = max_intensity.max(1) as f64;

        let working_img = ctx.get_f32_gray_image_mut()?;
        let size = working_img.size();
        let data = working_img.as_slice_mut();

        if self.ball_type == BallType::Paraboloid {
            // ImageJ's "Sliding Paraboloid" is NOT the ball-rolling opening
            // with a parabolic height profile - it's a different algorithm
            // entirely (`docs/rolling_ball/rolling_ball_sliding_paraboloid.cpp`,
            // `slidingParaboloidFloatBackground`): a 1D parabola is slid along
            // every row, column, and both diagonals of the full-resolution
            // image directly (no shrink/ball/enlarge at all).
            let background =
                self.sliding_paraboloid_background(data, size.width, size.height, scale_factor);
            for (d, bg) in data.iter_mut().zip(background.iter()) {
                *d = (*d - *bg).max(0.0);
            }
            return Ok(());
        }

        // Generate the 3D structural element scaled perfectly to the 0.0..1.0 range.
        let ball = self.build_ball(scale_factor);

        // 1. Pre-smooth (Allocation-free Separable 3x3 Mean filter)
        if self.pre_smooth {
            self.pre_smooth_separable(data, size.width, size.height);
        }

        // 2. Shrink (Downsample optimized sequential blocks)
        let (s_width, s_height, s_data) = if ball.shrink_factor > 1 {
            let sw = (size.width + ball.shrink_factor - 1) / ball.shrink_factor;
            let sh = (size.height + ball.shrink_factor - 1) / ball.shrink_factor;
            let mut sd = vec![f32::MAX; sw * sh];

            for y_s in 0..sh {
                let dest_row_offset = y_s * sw;
                for x_s in 0..sw {
                    let mut min_val = f32::MAX;
                    for dy in 0..ball.shrink_factor {
                        let y = (y_s * ball.shrink_factor + dy).min(size.height - 1);
                        let base_idx = y * size.width;
                        for dx in 0..ball.shrink_factor {
                            let x = (x_s * ball.shrink_factor + dx).min(size.width - 1);
                            let val = data[base_idx + x];
                            if val < min_val {
                                min_val = val;
                            }
                        }
                    }
                    sd[dest_row_offset + x_s] = min_val;
                }
            }
            (sw, sh, sd)
        } else {
            (size.width, size.height, data.to_vec())
        };

        // 3. Roll Ball (Sequential cache-localized multi-pass morphology filter)
        //
        // The reference implementation (`docs/rolling_ball/rolling_ball.cpp`,
        // `rollBall`) lets the ball's centre roll to positions up to
        // `radius_in` pixels OUTSIDE the image (`for (y = -radiusIn; y <
        // height + radiusIn; y++)`); only the erosion at those off-image
        // centres is clamped to the pixels that actually exist. Restricting
        // the centre to stay strictly inside the image - as a naive
        // erosion/dilation would - systematically mis-estimates the
        // background within `radius_in` pixels of every border, since those
        // pixels then never see the ball positions that would otherwise rest
        // just past the edge. `bg_small` is padded by `radius_in` on every
        // side so Pass A can reproduce that off-image range; Pass B then
        // reads from it exactly like the reference's scatter step (the ball
        // is radially symmetric, so gathering `bg_small[dest + d]` is
        // equivalent to the reference's scatter from center `dest - d`).
        let radius_in = (ball.width / 2) as i32;
        let ball_width = ball.width;
        let ball_data_ref = &ball.data;
        let s_data_ref = &s_data;
        let pad = radius_in as usize;
        let padded_w = s_width + 2 * pad;
        let padded_h = s_height + 2 * pad;

        // Pass A: Erosion (Build local minimum baseline envelope), including
        // the `radius_in`-wide off-image margin.
        let mut bg_small = vec![f32::MAX; padded_w * padded_h];
        for py in 0..padded_h {
            let y_i32 = py as i32 - radius_in;
            let y0 = (y_i32 - radius_in).max(0);
            let y_end = (y_i32 + radius_in).min(s_height as i32 - 1);
            let dest_row_offset = py * padded_w;

            for px in 0..padded_w {
                let x_i32 = px as i32 - radius_in;
                let x0 = (x_i32 - radius_in).max(0);
                let x_end = (x_i32 + radius_in).min(s_width as i32 - 1);
                let mut min_z = f32::MAX;

                for yp in y0..=y_end {
                    let y_ball = (yp - y_i32 + radius_in) as usize;
                    let ball_row_offset = y_ball * ball_width;
                    let src_row_offset = yp as usize * s_width;

                    for xp in x0..=x_end {
                        let x_ball = (xp - x_i32 + radius_in) as usize;
                        let val = s_data_ref[src_row_offset + xp as usize]
                            - ball_data_ref[ball_row_offset + x_ball];
                        if val < min_z {
                            min_z = val;
                        }
                    }
                }
                bg_small[dest_row_offset + px] = min_z;
            }
        }

        // Pass B: Dilation (Trace structural element paths). Every real
        // destination pixel's full `radius_in` neighborhood is always inside
        // the padded buffer, so no clamping is needed here.
        let mut bg_dilated = vec![f32::NEG_INFINITY; s_width * s_height];
        for y_s in 0..s_height {
            let y_i32 = y_s as i32;
            let dest_row_offset = y_s * s_width;

            for x_s in 0..s_width {
                let x_i32 = x_s as i32;
                let mut max_z = f32::NEG_INFINITY;

                for dy in -radius_in..=radius_in {
                    let y_ball = (dy + radius_in) as usize;
                    let ball_row_offset = y_ball * ball_width;
                    let bg_row_offset = (y_i32 + dy + radius_in) as usize * padded_w;

                    for dx in -radius_in..=radius_in {
                        let x_ball = (dx + radius_in) as usize;
                        let z_min = bg_small[bg_row_offset + (x_i32 + dx + radius_in) as usize]
                            + ball_data_ref[ball_row_offset + x_ball];
                        if z_min > max_z {
                            max_z = z_min;
                        }
                    }
                }
                bg_dilated[dest_row_offset + x_s] = max_z;
            }
        }

        // 4. Enlarge and Subtract (Fixed Bilinear Interpolation Weights)
        //
        // This deliberately does NOT port the reference `enlargeImage`'s
        // row-caching structure (`docs/rolling_ball/rolling_ball.cpp`): that
        // function tries to swap two row buffers via
        // `memcpy(line0, line1, width)`, copying `width` *bytes* instead of
        // `width * sizeof(float)` bytes. For any non-trivial image that
        // leaves most of each row stale, corrupting the interpolated
        // background into a blocky pattern whenever `shrink_factor > 1`
        // (`radius > 10`). That's an unrelated defect in the reference, not
        // an intentional behavior to preserve - indexing `bg_dilated`
        // directly below is mathematically equivalent to what the reference
        // *intends* and is already verified correct.
        if ball.shrink_factor > 1 {
            let mut x_indices = vec![0usize; size.width];
            let mut x_weights = vec![0.0f32; size.width];
            Self::make_interpolation_arrays(
                &mut x_indices,
                &mut x_weights,
                size.width,
                s_width,
                ball.shrink_factor,
            );

            let mut y_indices = vec![0usize; size.height];
            let mut y_weights = vec![0.0f32; size.height];
            Self::make_interpolation_arrays(
                &mut y_indices,
                &mut y_weights,
                size.height,
                s_height,
                ball.shrink_factor,
            );

            for y in 0..size.height {
                let y_idx = y_indices[y];
                let y_w = y_weights[y];

                let target_row_offset = y * size.width;
                let row_00_offset = y_idx * s_width;
                let row_01_offset = (y_idx + 1).min(s_height - 1) * s_width;

                for x in 0..size.width {
                    let x_idx = x_indices[x];
                    let x_w = x_weights[x];
                    let x_next = (x_idx + 1).min(s_width - 1);

                    let v00 = bg_dilated[row_00_offset + x_idx];
                    let v10 = bg_dilated[row_00_offset + x_next];
                    let v01 = bg_dilated[row_01_offset + x_idx];
                    let v11 = bg_dilated[row_01_offset + x_next];

                    // Mathematically fixed bilinear blending weights mapping distances cleanly
                    let interp_bg = v00 * (1.0 - x_w) * (1.0 - y_w)
                        + v10 * x_w * (1.0 - y_w)
                        + v01 * (1.0 - x_w) * y_w
                        + v11 * x_w * y_w;

                    let idx = target_row_offset + x;
                    data[idx] = (data[idx] - interp_bg).max(0.0);
                }
            }
        } else {
            for i in 0..data.len() {
                data[i] = (data[i] - bg_dilated[i]).max(0.0);
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "RollingBall"
    }
}

struct BallPatch {
    data: Vec<f32>,
    width: usize,
    shrink_factor: usize,
}

impl RollingBall {
    /// Builds the spherical-cap structural element for `BallType::Ball`.
    /// `BallType::Paraboloid` never calls this - it uses the sliding
    /// paraboloid algorithm instead (see [`Self::sliding_paraboloid_background`]).
    fn build_ball(&self, scale_factor: f64) -> BallPatch {
        let radius = self.radius;
        let (shrink_factor, arc_trim_per) = if radius <= 10.0 {
            (1, 24)
        } else if radius <= 30.0 {
            (2, 24)
        } else if radius <= 100.0 {
            (4, 32)
        } else {
            (8, 40)
        };

        let small_radius = (radius / shrink_factor as f64).max(1.0);
        let r_square = small_radius * small_radius;
        let x_trim = (arc_trim_per as f64 * small_radius / 100.0) as i32;
        let half_width = (small_radius - x_trim as f64).round() as i32;
        let width = (2 * half_width + 1) as usize;

        let mut data = vec![0.0f32; width * width];
        for y in 0..width {
            for x in 0..width {
                let x_val = x as f64 - half_width as f64;
                let y_val = y as f64 - half_width as f64;
                let dist_sq = x_val * x_val + y_val * y_val;

                let temp = r_square - dist_sq;
                // Normalize the physical ball heights to the 0.0..1.0 float image space
                data[y * width + x] = if temp > 0.0 {
                    (temp.sqrt() / scale_factor) as f32
                } else {
                    0.0
                };
            }
        }
        BallPatch {
            data,
            width,
            shrink_factor,
        }
    }

    // High-performance allocation-free 3x3 blur separating horizontal and vertical tracks
    fn pre_smooth_separable(&self, data: &mut [f32], width: usize, height: usize) {
        let mut temp = vec![0.0f32; data.len()];

        // Horizontal Pass
        for y in 0..height {
            let row_offset = y * width;
            for x in 0..width {
                let v1 = if x > 0 {
                    data[row_offset + x - 1]
                } else {
                    data[row_offset + x]
                };
                let v2 = data[row_offset + x];
                let v3 = if x < width - 1 {
                    data[row_offset + x + 1]
                } else {
                    data[row_offset + x]
                };
                temp[row_offset + x] = (v1 + v2 + v3) / 3.0;
            }
        }

        // Vertical Pass (reading from temp, saving back into data)
        for x in 0..width {
            for y in 0..height {
                let v1 = if y > 0 {
                    temp[(y - 1) * width + x]
                } else {
                    temp[y * width + x]
                };
                let v2 = temp[y * width + x];
                let v3 = if y < height - 1 {
                    temp[(y + 1) * width + x]
                } else {
                    temp[y * width + x]
                };
                data[y * width + x] = (v1 + v2 + v3) / 3.0;
            }
        }
    }

    /// ImageJ's "Sliding Paraboloid" background (`BallType::Paraboloid`).
    ///
    /// Port of `RollingBall::slidingParaboloidFloatBackground`
    /// (`docs/rolling_ball/rolling_ball_sliding_paraboloid.cpp`): a 1D
    /// parabola is slid along every row, column, and both diagonals of the
    /// full-resolution image (no ball, no shrink/enlarge). Returns the
    /// computed background envelope, the same size as `data`; `data` itself
    /// is left untouched.
    fn sliding_paraboloid_background(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        scale_factor: f64,
    ) -> Vec<f32> {
        const X_DIRECTION: usize = 0;
        const Y_DIRECTION: usize = 1;
        const DIAGONAL_1A: usize = 2;
        const DIAGONAL_1B: usize = 3;
        const DIAGONAL_2A: usize = 4;
        const DIAGONAL_2B: usize = 5;

        let mut bg = data.to_vec();
        let max_len = width.max(height);
        let mut cache = vec![0.0f32; max_len];
        let mut next_point = vec![0i32; max_len];

        // `coeff2` is the parabola's 2nd-order coefficient in raw-intensity
        // units per pixel^2 in the reference; divide by `scale_factor` to
        // keep it commensurate with this crate's 0..1-normalized image data,
        // exactly like `build_ball` does for the ball height.
        let coeff2 = (0.5 / self.radius / scale_factor) as f32;
        let coeff2diag = (1.0 / self.radius / scale_factor) as f32;

        let mut shift_by = 0.0f64;
        if self.pre_smooth {
            shift_by = Self::filter3x3_max(&mut bg, width, height);
            self.pre_smooth_separable(&mut bg, width, height);
        }

        Self::correct_corners(&mut bg, width, height, coeff2, &mut cache, &mut next_point);

        Self::filter1d(
            &mut bg,
            width,
            height,
            X_DIRECTION,
            coeff2,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            Y_DIRECTION,
            coeff2,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            X_DIRECTION,
            coeff2,
            &mut cache,
            &mut next_point,
        ); // redo for better accuracy
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_1A,
            coeff2diag,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_1B,
            coeff2diag,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_2A,
            coeff2diag,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_2B,
            coeff2diag,
            &mut cache,
            &mut next_point,
        );
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_1A,
            coeff2diag,
            &mut cache,
            &mut next_point,
        ); // redo for better accuracy
        Self::filter1d(
            &mut bg,
            width,
            height,
            DIAGONAL_1B,
            coeff2diag,
            &mut cache,
            &mut next_point,
        );

        if self.pre_smooth {
            let shift = shift_by as f32;
            for v in bg.iter_mut() {
                *v -= shift;
            }
        }

        bg
    }

    /// Separable 3x3 MAXIMUM filter (dust removal), mirroring `filter3x3`
    /// called with ImageJ's `MAXIMUM` type. Returns the average per-pixel
    /// upward shift introduced, which is subtracted back out once the
    /// paraboloid slide is done.
    fn filter3x3_max(data: &mut [f32], width: usize, height: usize) -> f64 {
        let mut temp = vec![0.0f32; data.len()];
        let mut shift_by = 0.0f64;

        // Horizontal pass
        for y in 0..height {
            let row_offset = y * width;
            for x in 0..width {
                let v1 = if x > 0 {
                    data[row_offset + x - 1]
                } else {
                    data[row_offset + x]
                };
                let v2 = data[row_offset + x];
                let v3 = if x < width - 1 {
                    data[row_offset + x + 1]
                } else {
                    data[row_offset + x]
                };
                let m = v1.max(v2).max(v3);
                shift_by += (m - v2) as f64;
                temp[row_offset + x] = m;
            }
        }

        // Vertical pass (reading from temp, saving back into data)
        for x in 0..width {
            for y in 0..height {
                let v1 = if y > 0 {
                    temp[(y - 1) * width + x]
                } else {
                    temp[y * width + x]
                };
                let v2 = temp[y * width + x];
                let v3 = if y < height - 1 {
                    temp[(y + 1) * width + x]
                } else {
                    temp[y * width + x]
                };
                let m = v1.max(v2).max(v3);
                shift_by += (m - v2) as f64;
                data[y * width + x] = m;
            }
        }

        shift_by / width as f64 / height as f64
    }

    /// Detects corner particles and lowers the four corner pixels toward the
    /// estimated edge-particle-free baseline, averaged over the 3 directions
    /// (the two edges and the diagonal) touching each corner. Port of
    /// `correctCorners`.
    fn correct_corners(
        data: &mut [f32],
        width: usize,
        height: usize,
        coeff2: f32,
        cache: &mut [f32],
        next_point: &mut [i32],
    ) {
        let w = width as i32;
        let h = height as i32;
        let mut corners = [0.0f32; 4]; // (0,0); (xmax,0); (0,ymax); (xmax,ymax)
        let mut edges = [0.0f32; 2];

        Self::line_slide_parabola(data, 0, 1, w, coeff2, cache, next_point, Some(&mut edges));
        corners[0] = edges[0];
        corners[1] = edges[1];

        Self::line_slide_parabola(
            data,
            (h - 1) * w,
            1,
            w,
            coeff2,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[2] = edges[0];
        corners[3] = edges[1];

        Self::line_slide_parabola(data, 0, w, h, coeff2, cache, next_point, Some(&mut edges));
        corners[0] += edges[0];
        corners[2] += edges[1];

        Self::line_slide_parabola(
            data,
            w - 1,
            w,
            h,
            coeff2,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[1] += edges[0];
        corners[3] += edges[1];

        let diag_length = width.min(height) as i32;
        let coeff2diag = 2.0 * coeff2;

        Self::line_slide_parabola(
            data,
            0,
            1 + w,
            diag_length,
            coeff2diag,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[0] += edges[0];

        Self::line_slide_parabola(
            data,
            w - 1,
            -1 + w,
            diag_length,
            coeff2diag,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[1] += edges[0];

        Self::line_slide_parabola(
            data,
            (h - 1) * w,
            1 - w,
            diag_length,
            coeff2diag,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[2] += edges[0];

        Self::line_slide_parabola(
            data,
            w * h - 1,
            -1 - w,
            diag_length,
            coeff2diag,
            cache,
            next_point,
            Some(&mut edges),
        );
        corners[3] += edges[0];

        if data[0] > corners[0] / 3.0 {
            data[0] = corners[0] / 3.0;
        }
        if data[width - 1] > corners[1] / 3.0 {
            data[width - 1] = corners[1] / 3.0;
        }
        if data[(height - 1) * width] > corners[2] / 3.0 {
            data[(height - 1) * width] = corners[2] / 3.0;
        }
        if data[width * height - 1] > corners[3] / 3.0 {
            data[width * height - 1] = corners[3] / 3.0;
        }
    }

    /// Filters by subtracting a sliding parabola for all lines in one
    /// direction (x, y, or one of the two diagonals - diagonals only cover
    /// half the image per call). Port of `filter1D`.
    #[allow(clippy::too_many_arguments)]
    fn filter1d(
        data: &mut [f32],
        width: usize,
        height: usize,
        direction: usize,
        coeff2: f32,
        cache: &mut [f32],
        next_point: &mut [i32],
    ) {
        const X_DIRECTION: usize = 0;
        const Y_DIRECTION: usize = 1;
        const DIAGONAL_1A: usize = 2;
        const DIAGONAL_1B: usize = 3;
        const DIAGONAL_2A: usize = 4;
        const DIAGONAL_2B: usize = 5;

        let w = width as i32;
        let h = height as i32;

        let (start_line, n_lines, line_inc, point_inc): (i32, i32, i32, i32) = match direction {
            X_DIRECTION => (0, h, w, 1),
            Y_DIRECTION => (0, w, 1, w),
            DIAGONAL_1A => (0, w - 2, 1, w + 1), // parallel to x=y, starting at x axis
            DIAGONAL_1B => (1, h - 2, w, w + 1), // parallel to x=y, starting at y axis
            DIAGONAL_2A => (2, w, 1, w - 1),     // parallel to x=-y, starting at x axis
            DIAGONAL_2B => (0, h - 2, w, w - 1), // parallel to x=-y, starting at x=width-1
            _ => unreachable!("invalid filter1d direction"),
        };

        for i in start_line..n_lines {
            let mut start_pixel = i * line_inc;
            if direction == DIAGONAL_2B {
                start_pixel += w - 1;
            }
            let length = match direction {
                X_DIRECTION => w,
                Y_DIRECTION => h,
                DIAGONAL_1A => h.min(w - i),
                DIAGONAL_1B => w.min(h - i),
                DIAGONAL_2A => h.min(i + 1),
                DIAGONAL_2B => w.min(h - i),
                _ => unreachable!("invalid filter1d direction"),
            };
            Self::line_slide_parabola(
                data, start_pixel, point_inc, length, coeff2, cache, next_point, None,
            );
        }
    }

    /// Processes one straight line by sliding a parabola along it (from the
    /// bottom) and interpolating between the points it touches. If
    /// `corrected_edges` is `Some`, also estimates the two edge values with
    /// any edge particle's contribution removed. Port of `lineSlideParabola`.
    #[allow(clippy::too_many_arguments)]
    fn line_slide_parabola(
        data: &mut [f32],
        start: i32,
        inc: i32,
        length: i32,
        coeff2: f32,
        cache: &mut [f32],
        next_point: &mut [i32],
        corrected_edges: Option<&mut [f32; 2]>,
    ) {
        let mut min_value = f32::MAX;
        let mut lastpoint: i32 = 0;
        let mut first_corner = length - 1; // first point (besides the edge) touched
        let mut last_corner = 0i32; // last point (besides the edge) touched
        let mut v_previous1 = 0.0f32;
        let mut v_previous2 = 0.0f32;
        let curvature_test = 1.999f32 * coeff2; // not 2: numeric scatter of 2nd derivative

        // Copy the line to `cache`, find its minimum, and find points with
        // enough local curvature that the parabola could touch them - only
        // those need to be examined further on.
        let mut p = start;
        for i in 0..length {
            let v = data[p as usize];
            cache[i as usize] = v;
            if v < min_value {
                min_value = v;
            }
            if i >= 2 && v_previous1 + v_previous1 - v_previous2 - v < curvature_test {
                next_point[lastpoint as usize] = i - 1;
                lastpoint = i - 1;
            }
            v_previous2 = v_previous1;
            v_previous1 = v;
            p += inc;
        }
        next_point[lastpoint as usize] = length - 1;
        next_point[(length - 1) as usize] = i32::MAX; // breaks the search loop

        // i1 and i2 are the two points where the parabola touches.
        let mut i1 = 0i32;
        while i1 < length - 1 {
            let v1 = cache[i1 as usize];
            let mut min_slope = f32::MAX;
            let mut i2 = 0i32;
            let mut search_to = length;
            let mut recalculate_limit_now = 0i32;
            let mut j = next_point[i1 as usize];
            while j < search_to {
                let v2 = cache[j as usize];
                let slope = (v2 - v1) / (j - i1) as f32 + coeff2 * (j - i1) as f32;
                if slope < min_slope {
                    min_slope = slope;
                    i2 = j;
                    recalculate_limit_now = -3;
                }
                if recalculate_limit_now == 0 {
                    // Time-consuming recalculation of the search limit: wait
                    // a bit after the slope was last updated.
                    let b = (0.5f32 * min_slope / coeff2) as f64;
                    let max_search = i1 as f64
                        + b
                        + (b * b + ((v1 - min_value) / coeff2) as f64).sqrt()
                        + 1.0; // numeric overflow may make this negative
                    let max_search = max_search as i32;
                    if max_search < search_to && max_search > 0 {
                        search_to = max_search;
                    }
                }
                j = next_point[j as usize];
                recalculate_limit_now += 1;
            }
            if i1 == 0 {
                first_corner = i2;
            }
            if i2 == length - 1 {
                last_corner = i1;
            }
            // Interpolate between the two points where the parabola touches.
            let mut p = start + (i1 + 1) * inc;
            for j in (i1 + 1)..i2 {
                data[p as usize] = v1 + (j - i1) as f32 * (min_slope - (j - i1) as f32 * coeff2);
                p += inc;
            }
            i1 = i2;
        }

        // Estimate edge values without an edge particle, allowing for
        // vignetting described by a 6th-order polynomial.
        if let Some(edges) = corrected_edges {
            let mut first_corner = first_corner;
            let mut last_corner = last_corner;
            if 4 * first_corner >= length {
                first_corner = 0; // edge particles must be < 1/4 image size
            }
            if 4 * (length - 1 - last_corner) >= length {
                last_corner = length - 1;
            }
            let v1 = cache[first_corner as usize];
            let v2 = cache[last_corner as usize];
            // Slope of the line through the two outermost non-edge touching points.
            let slope = (v2 - v1) / (last_corner - first_corner) as f32;
            let value0 = v1 - slope * first_corner as f32; // offset of this line
            let mut coeff6 = 0.0f32; // coefficient of the 6th-order polynomial
            let mid = 0.5f32 * (last_corner + first_corner) as f32;
            // Compare against mid-image pixels to detect vignetting.
            for i in (length + 2) / 3..=(2 * length) / 3 {
                let dx = (i as f32 - mid) * 2.0 / (last_corner - first_corner) as f32;
                let poly6 = dx * dx * dx * dx * dx * dx - 1.0; // zero at firstCorner and lastCorner
                if cache[i as usize] < value0 + slope * i as f32 + coeff6 * poly6 {
                    coeff6 = -(value0 + slope * i as f32 - cache[i as usize]) / poly6;
                }
            }
            let dx = (first_corner as f32 - mid) * 2.0 / (last_corner - first_corner) as f32;
            edges[0] = value0
                + coeff6 * (dx * dx * dx * dx * dx * dx - 1.0)
                + coeff2 * (first_corner * first_corner) as f32;
            let dx = (last_corner as f32 - mid) * 2.0 / (last_corner - first_corner) as f32;
            edges[1] = value0
                + (length - 1) as f32 * slope
                + coeff6 * (dx * dx * dx * dx * dx * dx - 1.0)
                + coeff2 * ((length - 1 - last_corner) * (length - 1 - last_corner)) as f32;
        }
    }

    fn make_interpolation_arrays(
        indices: &mut [usize],
        weights: &mut [f32],
        length: usize,
        small_length: usize,
        shrink_factor: usize,
    ) {
        let sf = shrink_factor as f32;
        for i in 0..length {
            let mut small_index = (i as f32 - sf / 2.0) / sf;
            if small_index >= (small_length as f32 - 1.0) {
                small_index = small_length as f32 - 2.0;
            }
            if small_index < 0.0 {
                small_index = 0.0;
            }

            let idx = small_index as usize;
            indices[i] = idx;
            // Capture the raw fractional distance mapping without inverting inside the generator
            weights[i] = ((i as f32 + 0.5) / sf) - (idx as f32 + 0.5);
        }
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::{ImageContainer, image::PixelSizes, pipeline::pipeline::PipelineImageMeta};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    /// RollingBall is the one filter that mutates `ctx.image` in place (via
    /// `get_f32_gray_image_mut`) instead of the scratch+swap pattern every
    /// other filter uses. If `ctx.image`'s `Arc` is shared with something
    /// else - the pipeline cache, another pipeline reading the same channel -
    /// that in-place write must not be observed by the other holder. This is
    /// exactly the copy-on-write boundary `Arc::make_mut` provides.
    #[test]
    fn rolling_ball_does_not_mutate_a_shared_original_image() -> Result<(), Box<dyn std::error::Error>>
    {
        let width = 40;
        let height = 40;
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = (x as f32) * 0.01;
            }
        }
        let signal_value = 0.5;
        let (center_x, center_y) = (20isize, 20isize);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let idx = ((center_y + dy) * (width as isize) + (center_x + dx)) as usize;
                data[idx] += signal_value;
            }
        }
        let original_pixels = data.clone();

        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)?;
        // This Arc stands in for the pipeline cache's own reference to the
        // channel image - something that keeps reading it after this run.
        let shared_original: Arc<ImageContainer> =
            ImageContainer::new_f32_gray_from_image_test(image).into();

        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 40,
                    height: 40,
                },
                full_image_width: ImageSize {
                    width: 40,
                    height: 40,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            Arc::clone(&shared_original), // no clone: shares the same Arc as `shared_original`
        )?;

        let rb = RollingBall {
            radius: 10.0,
            ball_type: BallType::Ball,
            pre_smooth: true,
        };
        let mut cache = PipelineCache::default();
        rb.execute(&mut ctx, &mut cache)?;

        // The shared handle must still read back the exact original pixels -
        // RollingBall's in-place write must have cloned before touching them.
        match shared_original.as_ref() {
            ImageContainer::F32Gray(img) => {
                assert_eq!(
                    img.as_slice(),
                    original_pixels.as_slice(),
                    "RollingBall must not mutate a shared/cached original image"
                );
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }

        // ...and the pipeline's own output must actually differ (RollingBall
        // did do real work, just on its own private copy).
        match ctx.image.as_ref() {
            ImageContainer::F32Gray(img) => {
                assert_ne!(
                    img.as_slice(),
                    original_pixels.as_slice(),
                    "RollingBall should have changed the pipeline's own image"
                );
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }

        assert!(
            !Arc::ptr_eq(&ctx.image, &shared_original),
            "mutation must have produced a distinct buffer, not aliased the shared one"
        );

        Ok(())
    }

    #[test]
    fn test_rolling_ball_background_subtraction() -> Result<(), Box<dyn std::error::Error>> {
        let width = 40;
        let height = 40;
        let mut data = vec![0.0f32; width * height];

        // Create a realistic normalized background gradient (0.0 to 0.4)
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = (x as f32) * 0.01;
            }
        }

        // Add a "Cell" signal structure
        let signal_value = 0.5; // Stays cleanly bounded in the 0.0..1.0 landscape
        let center_x = 20;
        let center_y = 20;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let idx = ((center_y as isize + dy) * (width as isize) + (center_x as isize + dx))
                    as usize;
                data[idx] += signal_value;
            }
        }

        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)?;

        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 40,
                    height: 40,
                },
                full_image_width: ImageSize {
                    width: 40,
                    height: 40,
                },
                is_rgb: false,
                nr_of_bits: 8, // Triggers correct 255.0 ball normalization heights
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_gray_from_image_test(image).into(),
        )
        .unwrap();

        let rb = RollingBall {
            radius: 10.0,
            ball_type: BallType::Ball,
            pre_smooth: true,
        };
        let mut cache = PipelineCache::default();
        rb.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(out_img) = ctx.image.as_ref() {
            let out_data = out_img.as_slice();

            // The cell peak should remain strong
            let center_pixel = out_data[center_y * width + center_x];
            assert!(
                center_pixel > 0.3,
                "Signal heavily degraded. Value: {}",
                center_pixel
            );

            // The background ramp should be subtracted close to 0.0.
            // Sample an interior pixel rather than one at the very edge: the rolling
            // ball footprint is clamped near image boundaries, which leaves a small,
            // expected residual there. The interior reconstruction is exact.
            let right_bg_pixel = out_data[center_y * width + 30];
            assert!(
                right_bg_pixel < 0.05,
                "Background gradient not removed. Value: {}",
                right_bg_pixel
            );

            for (i, &val) in out_data.iter().enumerate() {
                assert!(
                    val >= -1e-6,
                    "Negative overflow value {} found at index {}",
                    val,
                    i
                );
            }
        } else {
            panic!("Output image was not in F32Gray format");
        }

        Ok(())
    }

    #[test]
    fn test_rolling_ball_shrink_factor_active() -> Result<(), Box<dyn std::error::Error>> {
        let width = 100;
        let height = 100;
        // Flat background baseline of 0.2
        let mut data = vec![0.2f32; width * height];
        data[50 * width + 50] = 0.8; // Cell signal spike

        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)?;
        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 100,
                    height: 100,
                },
                full_image_width: ImageSize {
                    width: 100,
                    height: 100,
                },
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
        .unwrap();
        let mut cache = PipelineCache::default();

        // Radius 40.0 triggers a structural shrink factor of 4
        let rb = RollingBall {
            radius: 40.0,
            ball_type: BallType::Ball,
            pre_smooth: false,
        };

        rb.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(out_img) = ctx.image.as_ref() {
            let out_data = out_img.as_slice();
            // Signal remains, background is correctly suppressed through the upscaling step
            assert!(
                out_data[50 * width + 50] > 0.4,
                "Signal lost after interpolation pass"
            );
            assert!(
                out_data[0] < 0.05,
                "Background baseline envelope leaked: {}",
                out_data[0]
            );
        }

        Ok(())
    }

    /// Golden-data regression for the border bug: the reference `rollBall`
    /// (`docs/rolling_ball/rolling_ball.cpp`) lets the ball's centre roll to
    /// positions up to `radius_in` pixels OUTSIDE the image
    /// (`for (y = -radiusIn; y < height + radiusIn; y++)`). Before this fix,
    /// the Rust port only ever centred the ball strictly inside the image,
    /// which systematically under-estimates the background within
    /// `radius_in` pixels of every border.
    ///
    /// Expected values are the exact output of a standalone harness built
    /// directly against `rollBall`/`RollingBallBall` from that file (radius
    /// 10, no shrink - `radius <= 10` keeps `shrinkFactor == 1`, so this
    /// path is unaffected by the *separate*, unrelated bug in that file's
    /// `enlargeImage` - see the comment above the "Enlarge and Subtract"
    /// step in `execute` - and is a clean ground truth for `rollBall` itself.
    #[test]
    fn test_rolling_ball_matches_cpp_reference_border_touching_peak() -> Result<(), Box<dyn std::error::Error>>
    {
        let width = 20;
        let height = 15;
        let mut data = vec![100.0f32; width * height];
        for y in 6..=8 {
            for x in 0..=1 {
                data[y * width + x] = 180.0;
            }
        }
        let original = data.clone();

        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)?;
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
                is_rgb: false,
                // nr_of_bits: 0 -> scale_factor 1, so the ball height stays in
                // raw pixel units, matching the C++ reference (which never
                // normalizes at all).
                nr_of_bits: 0,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            Arc::new(ImageContainer::new_f32_gray_from_image_test(image)),
        )?;

        let rb = RollingBall {
            radius: 10.0,
            ball_type: BallType::Ball,
            pre_smooth: false,
        };
        let mut cache = PipelineCache::default();
        rb.execute(&mut ctx, &mut cache)?;

        let ImageContainer::F32Gray(out_img) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let out = out_img.as_slice();

        // Reconstruct the background envelope (original - result); nothing
        // here hits the `.max(0.0)` clamp since background < signal
        // everywhere in this fixture.
        let bg = |x: usize, y: usize| original[y * width + x] - out[y * width + x];

        #[rustfmt::skip]
        let expected: [(usize, usize, f32); 6] = [
            (0, 6, 100.259224), (1, 6, 100.101540),
            (0, 7, 100.343147), (1, 7, 100.151917),
            (0, 8, 100.259224), (1, 8, 100.101540),
        ];
        for (x, y, want) in expected {
            let got = bg(x, y);
            assert!(
                (got - want).abs() < 1e-3,
                "background at ({x},{y}): got {got}, expected {want} (C++ reference)"
            );
        }

        Ok(())
    }

    /// Golden-data regression for `BallType::Paraboloid`.
    ///
    /// `ballType: PARABOLOID` used to run the ball-rolling erosion/dilation
    /// pipeline with a parabolic *height profile* swapped in - a fundamentally
    /// different (and wrong) algorithm from ImageJ's actual "Sliding
    /// Paraboloid" (`docs/rolling_ball/rolling_ball_sliding_paraboloid.cpp`,
    /// `slidingParaboloidFloatBackground`), which slides a 1D parabola along
    /// rows, columns, and both diagonals instead. That mismatch is what
    /// produced "a lot of small artifacts" reported against real nuclei
    /// masks - the wrong algorithm doesn't just differ subtly, it's a
    /// different shape of background envelope entirely.
    ///
    /// Expected values are the exact output of a standalone harness built
    /// directly against `slidingParaboloidFloatBackground` /
    /// `lineSlideParabola` / `filter1D` / `correctCorners` from that file
    /// (radius 4, `preSmooth=false`, matching the reported pipeline exactly:
    /// `{"type": "rollingBall", "radius": 4.0, "ballType": "PARABOLOID",
    /// "preSmooth": false}`).
    #[test]
    fn test_rolling_ball_paraboloid_matches_cpp_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        const WIDTH: usize = 16;
        const HEIGHT: usize = 12;

        #[rustfmt::skip]
        const INPUT: &[f32] = &[
            500.0, 508.0, 516.0, 524.0, 532.0, 540.0, 548.0, 556.0, 564.0, 572.0, 580.0, 588.0, 596.0, 604.0, 612.0, 620.0,
            505.0, 513.0, 521.0, 529.0, 537.0, 545.0, 553.0, 561.0, 569.0, 577.0, 585.0, 593.0, 601.0, 609.0, 617.0, 625.0,
            510.0, 518.0, 526.0, 534.0, 542.0, 550.0, 558.0, 566.0, 574.0, 582.0, 590.0, 598.0, 606.0, 614.0, 622.0, 630.0,
            515.0, 523.0, 531.0, 539.0, 547.0, 555.0, 563.0, 571.0, 579.0, 587.0, 595.0, 603.0, 611.0, 619.0, 627.0, 635.0,
            520.0, 528.0, 536.0, 544.0, 552.0, 860.0, 568.0, 576.0, 584.0, 592.0, 600.0, 608.0, 616.0, 624.0, 632.0, 640.0,
            525.0, 533.0, 541.0, 549.0, 857.0, 865.0, 873.0, 581.0, 589.0, 597.0, 605.0, 613.0, 621.0, 629.0, 637.0, 645.0,
            530.0, 538.0, 546.0, 854.0, 862.0, 870.0, 878.0, 886.0, 594.0, 602.0, 610.0, 618.0, 626.0, 634.0, 642.0, 650.0,
            535.0, 543.0, 551.0, 559.0, 867.0, 875.0, 883.0, 591.0, 599.0, 607.0, 615.0, 623.0, 631.0, 639.0, 647.0, 655.0,
            540.0, 548.0, 556.0, 564.0, 572.0, 880.0, 588.0, 596.0, 604.0, 612.0, 620.0, 628.0, 636.0, 644.0, 652.0, 660.0,
            545.0, 553.0, 561.0, 569.0, 577.0, 585.0, 593.0, 601.0, 609.0, 617.0, 625.0, 633.0, 641.0, 649.0, 657.0, 665.0,
            550.0, 558.0, 566.0, 574.0, 582.0, 590.0, 598.0, 606.0, 614.0, 622.0, 630.0, 638.0, 646.0, 654.0, 662.0, 670.0,
            555.0, 563.0, 571.0, 579.0, 587.0, 595.0, 603.0, 611.0, 619.0, 627.0, 635.0, 643.0, 651.0, 659.0, 667.0, 675.0,
        ];

        #[rustfmt::skip]
        const EXPECTED_BACKGROUND: &[f32] = &[
            500.0, 508.0, 516.0, 524.0, 532.0, 540.0, 548.0, 556.0, 564.0, 572.0, 580.0, 588.0, 596.0, 604.0, 612.0, 620.0,
            505.0, 513.0, 521.0, 529.0, 537.0, 545.0, 553.0, 561.0, 569.0, 577.0, 585.0, 593.0, 601.0, 609.0, 617.0, 625.0,
            510.0, 518.0, 526.0, 534.0, 542.0, 550.0, 558.0, 566.0, 574.0, 582.0, 590.0, 598.0, 606.0, 614.0, 622.0, 630.0,
            515.0, 523.0, 531.0, 539.0, 547.0, 555.0, 563.0, 571.0, 579.0, 587.0, 595.0, 603.0, 611.0, 619.0, 627.0, 635.0,
            520.0, 528.0, 536.0, 544.0, 552.0, 560.125, 568.0, 576.0, 584.0, 592.0, 600.0, 608.0, 616.0, 624.0, 632.0, 640.0,
            525.0, 533.0, 541.0, 549.0, 557.375, 565.5, 573.375, 581.0, 589.0, 597.0, 605.0, 613.0, 621.0, 629.0, 637.0, 645.0,
            530.0, 538.0, 546.0, 554.125, 562.5, 570.625, 578.5, 586.125, 594.0, 602.0, 610.0, 618.0, 626.0, 634.0, 642.0, 650.0,
            535.0, 543.0, 551.0, 559.0, 567.375, 575.5, 583.375, 591.0, 599.0, 607.0, 615.0, 623.0, 631.0, 639.0, 647.0, 655.0,
            540.0, 548.0, 556.0, 564.0, 572.0, 580.125, 588.0, 596.0, 604.0, 612.0, 620.0, 628.0, 636.0, 644.0, 652.0, 660.0,
            545.0, 553.0, 561.0, 569.0, 577.0, 585.0, 593.0, 601.0, 609.0, 617.0, 625.0, 633.0, 641.0, 649.0, 657.0, 665.0,
            550.0, 558.0, 566.0, 574.0, 582.0, 590.0, 598.0, 606.0, 614.0, 622.0, 630.0, 638.0, 646.0, 654.0, 662.0, 670.0,
            555.0, 563.0, 571.0, 579.0, 587.0, 595.0, 603.0, 611.0, 619.0, 627.0, 635.0, 643.0, 651.0, 659.0, 667.0, 675.0,
        ];

        let image = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            INPUT.to_vec(),
            CpuAllocator,
        )?;
        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: WIDTH,
                    height: HEIGHT,
                },
                full_image_width: ImageSize {
                    width: WIDTH,
                    height: HEIGHT,
                },
                is_rgb: false,
                // nr_of_bits: 0 -> scale_factor 1, raw units matching C++.
                nr_of_bits: 0,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            Arc::new(ImageContainer::new_f32_gray_from_image_test(image)),
        )?;

        let rb = RollingBall {
            radius: 4.0,
            ball_type: BallType::Paraboloid,
            pre_smooth: false,
        };
        let mut cache = PipelineCache::default();
        rb.execute(&mut ctx, &mut cache)?;

        let ImageContainer::F32Gray(out_img) = ctx.image.as_ref() else {
            panic!("expected F32Gray output");
        };
        let out = out_img.as_slice();

        for (i, (&input, &want_bg)) in INPUT.iter().zip(EXPECTED_BACKGROUND.iter()).enumerate() {
            let got_bg = input - out[i];
            assert!(
                (got_bg - want_bg).abs() < 1e-2,
                "pixel {} (x={}, y={}): background {got_bg}, expected {want_bg} (C++ reference)",
                i,
                i % WIDTH,
                i / WIDTH
            );
        }

        Ok(())
    }
}
