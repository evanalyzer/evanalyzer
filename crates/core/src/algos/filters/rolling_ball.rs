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

        // Generate the 3D structural element scaled perfectly to the 0.0..1.0 range.
        let ball = self.build_ball(scale_factor);
        let working_img = ctx.get_f32_gray_image_mut()?;
        let size = working_img.size();
        let data = working_img.as_slice_mut();

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
        let radius_in = (ball.width / 2) as i32;
        let ball_width = ball.width;
        let ball_data_ref = &ball.data;
        let s_data_ref = &s_data;

        // Pass A: Erosion (Build local minimum baseline envelope)
        let mut bg_small = vec![f32::MAX; s_width * s_height];
        for y_s in 0..s_height {
            let y_i32 = y_s as i32;
            let y0 = (y_i32 - radius_in).max(0);
            let y_end = (y_i32 + radius_in).min(s_height as i32 - 1);
            let dest_row_offset = y_s * s_width;

            for x_s in 0..s_width {
                let x_i32 = x_s as i32;
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
                bg_small[dest_row_offset + x_s] = min_z;
            }
        }

        // Pass B: Dilation (Trace structural element paths)
        let mut bg_dilated = vec![f32::NEG_INFINITY; s_width * s_height];
        for y_s in 0..s_height {
            let y_i32 = y_s as i32;
            let y0 = (y_i32 - radius_in).max(0);
            let y_end = (y_i32 + radius_in).min(s_height as i32 - 1);
            let dest_row_offset = y_s * s_width;

            for x_s in 0..s_width {
                let x_i32 = x_s as i32;
                let x0 = (x_i32 - radius_in).max(0);
                let x_end = (x_i32 + radius_in).min(s_width as i32 - 1);
                let mut max_z = f32::NEG_INFINITY;

                for yp in y0..=y_end {
                    let y_ball = (yp - y_i32 + radius_in) as usize;
                    let ball_row_offset = y_ball * ball_width;
                    let bg_row_offset = yp as usize * s_width;

                    for xp in x0..=x_end {
                        let x_ball = (xp - x_i32 + radius_in) as usize;
                        let z_min = bg_small[bg_row_offset + xp as usize]
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
    fn build_ball(&self, scale_factor: f64) -> BallPatch {
        let radius = self.radius;
        let (shrink_factor, arc_trim_per) = match self.ball_type {
            BallType::Ball => {
                if radius <= 10.0 {
                    (1, 24)
                } else if radius <= 30.0 {
                    (2, 24)
                } else if radius <= 100.0 {
                    (4, 32)
                } else {
                    (8, 40)
                }
            }
            BallType::Paraboloid => (1, 0),
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

                data[y * width + x] = match self.ball_type {
                    BallType::Ball => {
                        let temp = r_square - dist_sq;
                        // Normalize the physical ball heights to the 0.0..1.0 float image space
                        if temp > 0.0 {
                            (temp.sqrt() / scale_factor) as f32
                        } else {
                            0.0
                        }
                    }
                    BallType::Paraboloid => {
                        let z = (r_square - dist_sq) / (2.0 * small_radius);
                        if z > 0.0 {
                            (z / scale_factor) as f32
                        } else {
                            0.0
                        }
                    }
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
    use super::*;
    use crate::{ImageContainer, image::PixelSizes, pipeline::pipeline::PipelineImageMeta};
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

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
            ImageContainer::new_f32_gray_from_image_test(image),
        )
        .unwrap();

        let rb = RollingBall {
            radius: 10.0,
            ball_type: BallType::Ball,
            pre_smooth: true,
        };
        let mut cache = PipelineCache::default();
        rb.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
            let out_data = out_img.as_slice();

            // The cell peak should remain strong
            let center_pixel = out_data[center_y * width + center_x];
            assert!(
                center_pixel > 0.3,
                "Signal heavily degraded. Value: {}",
                center_pixel
            );

            // The background ramp should be subtracted close to 0.0
            let right_bg_pixel = out_data[center_y * width + 38];
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
            ImageContainer::new_f32_gray_from_image_test(image),
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

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
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
}
