//! # edm
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::{
    algos::ImageAlgorithm,
    pipeline::{pipeline_cache::PipelineCache, pipeline_context::PipelineContext},
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::i32;

/// A point marker used in bit-packing coordinates (x | y << 16)
const NO_POINT: i32 = -1;

/// A command that calculates the Euclidean Distance Map (EDM) of an f32 image.
///
/// This algorithm identifies pixels below a threshold as "background" and
/// calculates the distance of every "foreground" pixel to the nearest background pixel.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct DistanceTransform {
    /// Values less than or equal to this are treated as background (distance = 0).
    pub threshold: f32,
    /// If true, the pixels outside the image boundary are treated as background.
    pub edges_are_background: bool,
}

impl ImageAlgorithm for DistanceTransform {
    /// Performs a high-speed Euclidean Distance Transform (EDT) on an f32 image.
    ///
    /// The algorithm uses a two-pass sequential scanning technique (Rosenfeld-Pfaltz)
    /// which provides $O(N)$ performance, making it significantly faster than
    /// brute-force distance calculations or traditional morphological dilations.
    ///
    /// # Process Flow
    /// 1. **Initialization**: Pixels above `threshold` are marked as foreground (infinity),
    ///    while others are marked as background (0.0).
    /// 2. **Forward Passes**: Scans the image from top-left to bottom-right to
    ///    propagate distances from the top and left.
    /// 3. **Backward Passes**: Scans from bottom-right to top-left to propagate
    ///    distances from the bottom and right.
    /// 4. **Linearization**: Final squared distances are square-rooted to produce
    ///    true Euclidean values.
    ///
    /// # Implementation Details
    /// To maintain precision, this implementation packs $(x, y)$ coordinates into
    /// a 32-bit integer inside the `point_bufs`. This allows the algorithm to
    /// calculate the exact distance to the nearest root background pixel rather
    /// than accumulating error-prone chamfer distances.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        // Access the f32 image from context
        let (input, mut scratch) = ctx.get_gray_img_gray_buf()?;
        let width = input.width();
        let height = input.height();
        let input_slice = input.as_slice();

        //  Initialize result buffer
        for (out_pixel, &in_pixel) in scratch.as_slice_mut().iter_mut().zip(input_slice.iter()) {
            // This turns into a 'conditional move' or 'mask' in assembly
            let mask = (in_pixel > self.threshold) as u32;
            // If mask is 1, we get f32::MAX. If 0, we get 0.0 (or keep original).
            // Note: This specific math depends on if you want to clear
            // the other pixels or keep them.
            if mask == 1 {
                *out_pixel = f32::MAX;
            }
        }

        // Setup point buffers for coordinate tracking
        let mut point_bufs = [vec![NO_POINT; width], vec![NO_POINT; width]];

        // Four-Pass Scanning (Top-Down then Bottom-Up)

        // Passes 1 & 2: Top-to-Bottom
        for y in 0..height {
            let y_dist = if self.edges_are_background {
                (y as i32) + 1
            } else {
                i32::MAX
            };
            Self::edm_line(
                input,
                &mut scratch,
                &mut point_bufs,
                y,
                self.threshold,
                y_dist,
            );
        }

        // Reset buffers for reverse pass
        point_bufs[0].fill(NO_POINT);
        point_bufs[1].fill(NO_POINT);

        // Passes 3 & 4: Bottom-to-Top
        for y in (0..height).rev() {
            let y_dist = if self.edges_are_background {
                (height as i32) - (y as i32)
            } else {
                i32::MAX
            };
            Self::edm_line(
                input,
                &mut scratch,
                &mut point_bufs,
                y,
                self.threshold,
                y_dist,
            );
        }

        // Transform Squared Distances to Euclidean Distances
        scratch.as_slice_mut().iter_mut().for_each(|val| {
            *val = val.sqrt();
        });

        //  Store result in scratch_pad
        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DistanceTransform"
    }
}

impl DistanceTransform {
    /// Calculates the minimum distance^2 from the current point to candidates in the buffers.
    /// Packed format: low 16 bits = x, high 16 bits = y.
    fn min_dist2(
        points: &mut [i32],
        p_prev: i32,
        p_diag: i32,
        x: i32,
        y: i32,
        mut dist_sq: i32,
    ) -> f32 {
        let mut nearest_point = NO_POINT; // Start with no point

        // Helper to check a candidate and update dist/point
        let check_point = |p: i32, current_dist_sq: &mut i32, current_best_point: &mut i32| {
            if p != NO_POINT {
                let px = p & 0xffff;
                let py = (p >> 16) & 0xffff;
                let d_sq = (x - px).pow(2) + (y - py).pow(2);
                if d_sq < *current_dist_sq {
                    *current_dist_sq = d_sq;
                    *current_best_point = p;
                }
            }
        };

        // Check the three neighbors
        check_point(points[x as usize], &mut dist_sq, &mut nearest_point);
        check_point(p_diag, &mut dist_sq, &mut nearest_point);
        check_point(p_prev, &mut dist_sq, &mut nearest_point);

        // Update the buffer for this column
        if nearest_point != NO_POINT {
            points[x as usize] = nearest_point;
        }

        dist_sq as f32
    }

    /// Processes a single line with two passes: left-to-right and right-to-left.
    fn edm_line(
        input: &Image<f32, 1, CpuAllocator>,
        fp: &mut Image<f32, 1, CpuAllocator>,
        point_bufs: &mut [Vec<i32>; 2],
        y: usize,
        threshold: f32,
        y_dist: i32,
    ) {
        let width = input.width();
        let offset = y * width;
        let edges_are_background = y_dist != i32::MAX;

        // Pass A: Left-to-Right
        let mut p_prev = NO_POINT;
        let mut p_diag = NO_POINT;
        {
            let points = &mut point_bufs[0];
            // 1. Get the slices before the loop for better performance
            let input_slice = input.as_slice();
            let fp_slice = fp.as_slice_mut();

            for x in 0..width {
                let p_idx = offset + x; // Calculate the flat index once
                let p_next_diag = points[x];

                // Access pixel via slice indexing instead of .data
                if input_slice[p_idx] <= threshold {
                    points[x] = (x as i32) | ((y as i32) << 16);
                } else {
                    let mut dist_sq = i32::MAX;
                    if edges_are_background {
                        let d = ((x as i32) + 1).min(y_dist);
                        dist_sq = d * d;
                    }

                    let d2 = Self::min_dist2(points, p_prev, p_diag, x as i32, y as i32, dist_sq);

                    // Update pixel via mutable slice indexing
                    if fp_slice[p_idx] > d2 {
                        fp_slice[p_idx] = d2;
                    }
                }
                p_prev = points[x];
                p_diag = p_next_diag;
            }
        }

        // Pass B: Right-to-Left
        p_prev = NO_POINT;
        p_diag = NO_POINT;
        {
            let points = &mut point_bufs[1];
            // 1. Obtain slices before starting the loop
            let input_slice = input.as_slice();
            let fp_slice = fp.as_slice_mut();

            for x in (0..width).rev() {
                let p_idx = offset + x;
                let p_next_diag = points[x];

                // Access pixel via slice indexing instead of .data
                if input_slice[p_idx] <= threshold {
                    points[x] = (x as i32) | ((y as i32) << 16);
                } else {
                    let mut dist_sq = i32::MAX;
                    if edges_are_background {
                        // Distance to the right edge or the vertical edge
                        let d = ((width as i32) - (x as i32)).min(y_dist);
                        dist_sq = d * d;
                    }

                    // Calculate the minimum distance from neighbors
                    let d2 = Self::min_dist2(points, p_prev, p_diag, x as i32, y as i32, dist_sq);

                    // Update the distance map if a shorter path is found
                    if fp_slice[p_idx] > d2 {
                        fp_slice[p_idx] = d2;
                    }
                }
                p_prev = points[x];
                p_diag = p_next_diag;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageContainer, ImageDebugExt};
    use crate::pipeline::pipeline_context::PipelineContext;
    use kornia_image::ImageSize;

    #[test]
    fn test_distance_transform_basic() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Create a 5x5 grid.
        // All pixels are 1.0 (foreground) except for the center pixel at (2,2) which is 0.0 (background).
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![1.0f32; 25];
        data[2 * 5 + 2] = 0.0; // The "seed" point
        let img = Image::<f32, 1, CpuAllocator>::new(size, data, CpuAllocator)?;

        let mut ctx = PipelineContext::new_from_image_test(img)?;
        let mut cache = PipelineCache::default();

        // 2. Configure EDM: No edge background to keep distances relative to our seed.
        let edm = DistanceTransform {
            threshold: 0.5,
            edges_are_background: false,
        };

        edm.execute(&mut ctx, &mut cache)?;

        // 3. Extract result
        let result_container = ctx.image;
        if let ImageContainer::F32Gray(res_img) = result_container {
            res_img.print_window();

            let res_slice = res_img.as_slice();

            // Center should be 0.0
            assert_eq!(res_slice[2 * 5 + 2], 0.0);

            // Direct neighbors (orthogonal) should be 1.0
            // (2, 1), (2, 3), (1, 2), (3, 2)
            assert_eq!(res_slice[1 * 5 + 2], 1.0);
            assert_eq!(res_slice[3 * 5 + 2], 1.0);
            assert_eq!(res_slice[2 * 5 + 1], 1.0);
            assert_eq!(res_slice[2 * 5 + 3], 1.0);

            // Diagonal neighbors should be sqrt(1^2 + 1^2) approx 1.414
            let diag_dist = (2.0f32).sqrt();
            let epsilon = 1e-5;
            assert!((res_slice[1 * 5 + 1] - diag_dist).abs() < epsilon);
            assert!((res_slice[3 * 5 + 3] - diag_dist).abs() < epsilon);

            // Knight's move neighbors (1, 0) relative to center is (2+1, 2+2) = (3, 4)
            // Dist should be sqrt(1^2 + 2^2) = sqrt(5) approx 2.236
            let knight_dist = (5.0f32).sqrt();
            assert!((res_slice[4 * 5 + 3] - knight_dist).abs() < epsilon);

            println!("EDM Test Passed! Center values are correct.");
        } else {
            panic!("Output was not F32Gray");
        }

        Ok(())
    }
}
