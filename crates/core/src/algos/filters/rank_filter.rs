//! # rank_filter
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::{ImageContainer, ManagedImage};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;
use std::f32::NAN;

/// Specifies the statistical operation to perform on the local pixel neighborhood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankFilterType {
    /// Selects the middle value. Excellent for removing salt-and-pepper noise
    /// while preserving sharp edges.
    Median,

    /// Selects the lowest intensity (Erosion). Shrinks bright regions and
    /// expands dark regions.
    Min,

    /// Selects the highest intensity (Dilation). Expands bright regions and
    /// shrinks dark regions.
    Max,

    /// Computes the average value. Acts as a box blur, smoothing the image
    /// but blurring edges.
    Mean,

    /// Replaces a pixel only if it deviates from the neighborhood median
    /// by more than the specified threshold.
    Outliers(f32),
}

/// A filter that transforms pixels based on the statistical rank of their neighbors.
///
/// Rank filters are non-linear operators used for noise reduction,
/// morphological operations, and feature enhancement.
///
/// This algorithm sorts (ranks) all pixel values within a local neighborhood
/// window and assigns a specific percentile value to the center pixel. By selecting
/// different ranks, it acts as a configurable operator: the minimum rank performs
/// erosion, the maximum rank performs dilation, and the median rank (50th percentile)
/// provides highly effective impulse noise suppression while preserving sharp structural edges.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct RankFilter {
    /// The circular radius of the neighborhood to consider.
    ///
    /// A radius of 1.0 roughly corresponds to a 3x3 square, while larger
    /// values increase the effect's strength and computational cost.
    pub radius: f64,

    /// The specific ranking algorithm to apply to the neighborhood.
    pub filter_type: RankFilterType,
}

impl ImageAlgorithm for RankFilter {
    /// Executes the rank-based statistical filter on the image.
    ///
    /// This algorithm slides a window of a size determined by `radius` over the image,
    /// collects neighbor intensities, and applies the selected [`RankFilterType`].
    ///
    /// ### Workflow
    /// 1.  **Neighborhood Collection**: Gathers pixels within the specified radius.
    /// 2.  **Statistical Selection**: Sorts or averages the values (Median, Min, Max, etc.).
    /// 3.  **Buffer Swap**: The result is written to the scratch pad, which then becomes
    ///     the active image for the next step in the pipeline.
    ///
    /// ### Supported Formats
    /// * **F32Gray**: Processes single-channel intensity.
    /// * **F32Rgb**: Processes each channel independently.
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input format is not a supported
    /// 32-bit floating-point type.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match &ctx.image {
            ImageContainer::F32Gray(img) => {
                let out = self.process_image::<1>(img)?;
                ctx.scratch_pad = ImageContainer::F32Gray(ManagedImage {
                    data: out,
                    tile_offset: img.tile_offset,
                    plane: img.plane,
                });
                ctx.swap()?;
                Ok(())
            }
            ImageContainer::F32Rgb(img) => {
                let out = self.process_image::<3>(img)?;
                ctx.scratch_pad = ImageContainer::F32Rgb(ManagedImage {
                    data: out,
                    tile_offset: img.tile_offset,
                    plane: img.plane,
                });
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
        "RankFilter"
    }
}

impl RankFilter {
    /// Generic processing function to handle 1 or 3 channels
    fn process_image<const C: usize>(
        &self,
        img: &Image<f32, C, CpuAllocator>,
    ) -> Result<Image<f32, C, CpuAllocator>, InternalErrors> {
        let (line_radii, k_radius, n_points) = get_kernel_geometry(self.radius);
        let k_height = (2 * k_radius + 1) as usize;

        let (width, height) = (img.width(), img.height());
        let src_data = img.as_slice();

        let mut out_image = Image::<f32, C, CpuAllocator>::new(
            img.size(),
            vec![0.0f32; width * height * C],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        let out_slice = out_image.as_slice_mut();

        let cache_width = width + (2 * k_radius as usize);

        // Process each channel (Marginal filtering)
        for c in 0..C {
            let channel_offset = c * width * height;

            // For parallel loop: (0..img_height).into_par_iter().for_each(|y| {
            for y in 0..height {
                let mut window = Vec::with_capacity(n_points);

                // Create a local cache for the neighborhood of the current row
                let mut local_cache = vec![0.0f32; cache_width * k_height];

                for ky in 0..k_height {
                    let yi = (y as i32 + ky as i32 - k_radius).clamp(0, height as i32 - 1) as usize;
                    let cache_row_start = ky * cache_width;
                    let src_row_start = channel_offset + (yi * width);

                    // Copy and Pad
                    for x in 0..width {
                        local_cache[cache_row_start + k_radius as usize + x] =
                            src_data[src_row_start + x];
                    }
                    let left_val = src_data[src_row_start];
                    let right_val = src_data[src_row_start + width - 1];
                    for px in 0..k_radius as usize {
                        local_cache[cache_row_start + px] = left_val;
                        local_cache[cache_row_start + k_radius as usize + width + px] = right_val;
                    }
                }

                // Process pixels in the current row
                for x in 0..width {
                    window.clear();
                    for ky in 0..k_height {
                        let r_left = line_radii[2 * ky];
                        let r_right = line_radii[2 * ky + 1];
                        let row_base = (ky * cache_width) + x + k_radius as usize;

                        for dx in r_left..=r_right {
                            let final_idx = (row_base as i32 + dx) as usize;
                            let val = local_cache[final_idx];
                            if !val.is_nan() {
                                window.push(val);
                            }
                        }
                    }

                    let pixel_idx = (y * width + x) * C + c;

                    if window.is_empty() {
                        out_slice[pixel_idx] = NAN;
                        continue;
                    }

                    out_slice[pixel_idx] = match self.filter_type {
                        RankFilterType::Median => {
                            let mid = window.len() / 2;
                            *window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
                        }
                        RankFilterType::Min => {
                            *window.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
                        }
                        RankFilterType::Max => {
                            *window.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
                        }
                        RankFilterType::Mean => window.iter().sum::<f32>() / window.len() as f32,
                        RankFilterType::Outliers(t) => {
                            let cur = src_data[pixel_idx];
                            let mid = window.len() / 2;
                            let med = *window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
                            if (cur - med).abs() > t { med } else { cur }
                        }
                    };
                }
            }
        }
        Ok(out_image)
    }
}

// Port of your makeLineRadii logic
fn get_kernel_geometry(radius: f64) -> (Vec<i32>, i32, usize) {
    let mut r: f64 = radius;
    if r >= 1.5 && r < 1.75 {
        r = 1.75;
    } else if r >= 2.5 && r < 2.85 {
        r = 2.85;
    }

    let r2 = (r * r) as i32 + 1;
    let k_radius = ((r2 as f64 + 1e-10).sqrt()) as i32;
    let k_height = 2 * k_radius + 1;

    let mut line_radii = vec![0i32; 2 * k_height as usize];
    let mut n_points = 0;

    for y in -k_radius..=k_radius {
        let dx = ((r2 - y * y) as f64 + 1e-10).sqrt() as i32;
        let idx = (2 * (y + k_radius)) as usize;
        line_radii[idx] = -dx;
        line_radii[idx + 1] = dx;
        n_points += (2 * dx + 1) as usize;
    }

    (line_radii, k_radius, n_points)
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::PixelSizes,
        pipeline::{pipeline::PipelineImageMeta, pipeline_cache::ImageCache},
    };
    use kornia_image::ImageSize;

    #[test]
    fn test_rank_filter_min_max_median() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Create a 5x5 test image with a distinct "bright" spot in the middle
        // 0 0 0 0 0
        // 0 0 0 0 0
        // 0 0 9 0 0
        // 0 0 0 0 0
        // 0 0 0 0 0
        let width = 5;
        let height = 5;
        let mut data = vec![0.0f32; width * height];
        data[12] = 9.0; // The center pixel (2,2)

        let image =
            Image::<f32, 1, CpuAllocator>::new(ImageSize { width, height }, data, CpuAllocator)?;

        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 5,
                    height: 5,
                },
                full_image_width: ImageSize {
                    width: 5,
                    height: 5,
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

        // 2. Test MAX filter (Radius 1.0)
        // A Max filter should spread the '9.0' value to all 8 neighbors.
        let max_filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Max,
        };
        max_filter.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
            // The pixel at (1,1) was 0, but is now a neighbor of (2,2), so it should be 9
            assert_eq!(*out_img.get_pixel(1, 1, 0).unwrap(), 9.0);
            assert_eq!(*out_img.get_pixel(0, 0, 0).unwrap(), 0.0); // Too far away
        }

        // 3. Test MIN filter (Radius 1.0) on the result
        // A Min filter on the previous result should shrink the '9.0' block back.
        let min_filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Min,
        };
        min_filter.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
            // The center should still be 9, but neighbors should return to 0
            assert_eq!(*out_img.get_pixel(2, 2, 0).unwrap(), 9.0);
            assert_eq!(*out_img.get_pixel(1, 1, 0).unwrap(), 0.0);
        }

        // 4. Test MEDIAN filter
        // In a neighborhood of mostly 0s and one 9, the median is 0.
        // This effectively removes "salt" noise.
        let median_filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Median,
        };
        median_filter.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
            // The spike at the center should be gone (0.0)
            assert_eq!(*out_img.get_pixel(2, 2, 0).unwrap(), 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_rank_filter_rgb() -> Result<(), Box<dyn std::error::Error>> {
        // HWC Layout: (y * width + x) * channels + channel
        // Pixel (1,1) is index 4. Red (channel 0) = (4 * 3) + 0 = 12
        let mut data = vec![0.0f32; 3 * 3 * 3];
        data[4] = 9.0;

        let image = Image::<f32, 3, CpuAllocator>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            data,
            CpuAllocator,
        )?;
        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 3,
                    height: 3,
                },
                full_image_width: ImageSize {
                    width: 3,
                    height: 3,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            ImageContainer::new_f32_rgb_from_image_test(image),
        )
        .unwrap();
        let mut cache = PipelineCache::default();

        let max_filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Max,
        };
        max_filter.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Rgb(ref out_img) = ctx.image {
            // Dilation should spread Red (9.0) to neighbors
            assert_eq!(*out_img.get_pixel(0, 0, 0).unwrap(), 9.0);
            assert_eq!(*out_img.get_pixel(0, 0, 1).unwrap(), 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_rank_filter_format_mismatch() {
        // 1. Create an unsupported U32 image
        let img = Image::<u32, 1, CpuAllocator>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0,
            CpuAllocator,
        )
        .unwrap();

        let mut ctx = PipelineContext::new_from_u32_image_test(img).unwrap();
        let mut cache = PipelineCache::default();

        let filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Median,
        };

        // 2. Assert FormatMismatch
        let result = filter.execute(&mut ctx, &mut cache);
        match result {
            Err(InternalErrors::FormatMismatch { expected, .. }) => {
                assert_eq!(expected, "F32Rgb or F32Gray");
            }
            _ => panic!("Expected FormatMismatch, got {:?}", result),
        }
    }

    #[test]
    fn test_rank_filter_name() {
        let filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Median,
        };
        assert_eq!(filter.name(), "RankFilter");
    }

    #[test]
    fn test_rank_filter_outliers() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Create a 3x3 image with a background of 5.0 and one clear outlier (1.0)
        // 5 5 5
        // 5 1 5
        // 5 5 5
        let mut data = vec![5.0f32; 9];
        data[4] = 1.0; // Outlier at center (1,1)

        let image = Image::<f32, 1, CpuAllocator>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            data,
            CpuAllocator,
        )?;

        let mut ctx = PipelineContext::new_from_image(
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 3,
                    height: 3,
                },
                full_image_width: ImageSize {
                    width: 3,
                    height: 3,
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

        // 2. Setup outlier filter with threshold 2.0
        // (5.0 - 1.0) = 4.0, which is > 2.0, so it should be replaced by the median (5.0)
        let outlier_filter = RankFilter {
            radius: 1.0,
            filter_type: RankFilterType::Outliers(2.0),
        };

        outlier_filter.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Gray(ref out_img) = ctx.image {
            // The outlier (1.0) should now be replaced by the median (5.0)
            assert_eq!(*out_img.get_pixel(1, 1, 0).unwrap(), 5.0);
            // Background pixels should remain unchanged
            assert_eq!(*out_img.get_pixel(0, 0, 0).unwrap(), 5.0);
        }
        Ok(())
    }
}
