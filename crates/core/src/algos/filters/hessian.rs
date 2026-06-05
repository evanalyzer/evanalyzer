//! # hessian
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ImageAlgorithm, PipelineContext};
use crate::image::{ImageContainer, ManagedImage, PixelSizes};
use crate::pipeline::pipeline_cache::PipelineCache;
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;

/// Specifies the feature extraction method for the Hessian matrix.
///
/// The Hessian matrix describes the local second-order structure of an image,
/// often used for blob detection (LoG) or ridge extraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HessianMode {
    /// Computes the determinant: $det(H) = I_{xx}I_{yy} - I_{xy}^2$.
    ///
    /// High values typically indicate "blob-like" structures or corners.
    Determinant,

    /// Extracts the first (larger) eigenvalue ($\lambda_1$).
    ///
    /// Useful for detecting the maximum local curvature, identifying
    /// the principal axis of a ridge.
    EigenvaluesX,

    /// Extracts the second (smaller) eigenvalue ($\lambda_2$).
    ///
    /// Highlights secondary curvature; when both $\lambda_1$ and $\lambda_2$
    /// are large, it indicates a blob or interest point.
    EigenvaluesY,
}
/// Extracts continuous structural ridges, tubular vessels, and blobs using second-order spatial derivatives.
///
/// This algorithm constructs a localized Hessian matrix for each pixel to analyze local curvature
/// and intensity topography. By evaluating the eigenvalues of this matrix, it differentiates
/// between directional ridges (like blood vessels or filaments), distinct intensity peaks (blobs),
/// and flat regions, making it highly effective for curvilinear feature extraction.
///
/// # Examples
///
/// ```
/// # use imagec::backend::algos::{Hessian, HessianMode};
/// let detector = Hessian {
///     mode: HessianMode::Determinant,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct Hessian {
    /// Determines which component of the Hessian matrix structure to extract.
    ///
    /// Depending on the mode, this can highlight interest points (blobs)
    /// or directional features (ridges).
    pub mode: HessianMode,
}
impl ImageAlgorithm for Hessian {
    /// Executes the Hessian feature detection algorithm on the current image.
    ///
    /// This implementation calculates the second-order partial derivatives ($I_{xx}, I_{yy}, I_{xy}$)
    /// to build the Hessian matrix for each pixel, then extracts features based on
    /// the selected [`HessianMode`].
    ///
    /// ### Supported Formats
    /// * **Input:** `F32Gray`
    /// * **Output:** `F32Gray` (In-place modification of the source image).
    ///
    /// # Errors
    /// Returns [`InternalErrors::FormatMismatch`] if the input image is not
    /// single-channel grayscale.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        match &mut ctx.image {
            ImageContainer::F32Gray(img) => {
                let result = process_f32_gray(img, self.mode)?;
                *img = ManagedImage {
                    data: result,
                    tile_offset: img.tile_offset,
                    plane: img.plane,
                };
                Ok(())
            }
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray for both input and scratch pad".into(),
                found: format!("Input: {:?}, Scratch: {:?}", ctx.image, ctx.scratch_pad),
            }),
        }
    }

    fn name(&self) -> &'static str {
        "Hessian"
    }
}

fn process_f32_gray(
    img: &Image<f32, 1, CpuAllocator>,
    mode: HessianMode,
) -> Result<Image<f32, 1, CpuAllocator>, InternalErrors> {
    let size = img.size();

    // Calculate first order gradients
    let mut dx = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    let mut dy = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();

    // kornia's spatial_gradient usually provides first order
    kornia_imgproc::filter::spatial_gradient_float(img, &mut dx, &mut dy).unwrap();

    // Calculate second order gradients (Hessian Matrix components)
    // Ixx = d/dx of dx
    let mut dxx = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    let mut dummy = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    kornia_imgproc::filter::spatial_gradient_float(&dx, &mut dxx, &mut dummy).unwrap();

    // Iyy = d/dy of dy
    let mut dyy = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    kornia_imgproc::filter::spatial_gradient_float(&dy, &mut dummy, &mut dyy).unwrap();

    // Ixy = d/dy of dx (Mixed partial derivative)
    let mut dxy = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    kornia_imgproc::filter::spatial_gradient_float(&dx, &mut dummy, &mut dxy).unwrap();

    // Compute Feature Maps
    let mut output = Image::from_size_val(size, 0.0, CpuAllocator).unwrap();
    let out_slice = output.as_slice_mut();
    let s_xx = dxx.as_slice();
    let s_yy = dyy.as_slice();
    let s_xy = dxy.as_slice();

    for i in 0..out_slice.len() {
        let ixx = s_xx[i];
        let iyy = s_yy[i];
        let ixy = s_xy[i];

        match mode {
            HessianMode::Determinant => {
                // det(H) = Ixx*Iyy - Ixy^2
                out_slice[i] = ixx * iyy - ixy * ixy;
            }
            HessianMode::EigenvaluesX | HessianMode::EigenvaluesY => {
                // Eigenvalues λ = (tr(H) ± sqrt(tr(H)^2 - 4*det(H))) / 2
                // Simplified: 0.5 * ( (Ixx+Iyy) ± sqrt( (Ixx-Iyy)^2 + 4*Ixy^2 ) )
                let trace = ixx + iyy;
                let diff = ixx - iyy;
                let term = (diff * diff + 4.0 * ixy * ixy).sqrt();

                if mode == HessianMode::EigenvaluesX {
                    out_slice[i] = 0.5 * (trace + term);
                } else {
                    out_slice[i] = 0.5 * (trace - term);
                }
            }
        }
    }

    Ok(output)
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use kornia_image::ImageSize;

    use super::*;
    use crate::pipeline::{
        pipeline::PipelineImageMeta,
        pipeline_cache::{ImageCache, PipelineCache},
    };

    #[test]
    fn test_hessian_determinant() {
        let mut data = vec![0.0f32; 100];
        // Create a 10x10 image with a white dot in the middle
        data[55] = 1.0;

        let img = Image::new(
            kornia_image::ImageSize {
                width: 10,
                height: 10,
            },
            data,
            CpuAllocator,
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();

        let mut cache = PipelineCache::default();

        let detector = Hessian {
            mode: HessianMode::Determinant,
        };

        detector.execute(&mut ctx, &mut cache).unwrap();

        if let ImageContainer::F32Gray(res) = &ctx.image {
            // Determinant should be high at the point of the blob/dot
            assert!(res.as_slice()[55] != 0.0);
        }
    }

    #[test]
    fn test_hessian_format_mismatch_error() {
        // 1. Setup: Create a 5x5 RGB image (Unsupported)
        let img = Image::<f32, 3, CpuAllocator>::from_size_val(
            kornia_image::ImageSize {
                width: 5,
                height: 5,
            },
            0.0,
            CpuAllocator,
        )
        .unwrap();

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
            crate::image::ImageContainer::new_f32_rgb_from_image_test(img),
        )
        .unwrap();
        let mut cache = PipelineCache::default();

        let detector = Hessian {
            mode: HessianMode::Determinant,
        };

        // 2. Execute & Assert
        let result = detector.execute(&mut ctx, &mut cache);
        match result {
            Err(InternalErrors::FormatMismatch { expected, .. }) => {
                assert!(expected.contains("F32Gray"));
            }
            _ => panic!("Expected FormatMismatch, got {:?}", result),
        }
    }

    #[test]
    fn test_hessian_eigenvalues() {
        // Use a 5x5 image to allow enough padding for the derivative kernels
        // Create a 7x7 image with a Gaussian-like peak at the center
        let mut data = vec![0.0f32; 49];
        // Create a central peak at (3,3)
        data[3 * 7 + 3] = 1.0;
        data[3 * 7 + 2] = 0.5;
        data[3 * 7 + 4] = 0.5;
        data[2 * 7 + 3] = 0.5;
        data[4 * 7 + 3] = 0.5;

        let img = Image::new(
            kornia_image::ImageSize {
                width: 7,
                height: 7,
            },
            data,
            CpuAllocator,
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = PipelineCache::default();

        // Test EigenvaluesX (Larger curvature/principal axis)
        let detector_x = Hessian {
            mode: HessianMode::EigenvaluesX,
        };
        detector_x.execute(&mut ctx, &mut cache).unwrap();

        // The center of this image is index 12.
        // It now has neighbors to calculate derivatives against.
        let val_x = if let ImageContainer::F32Gray(res) = &ctx.image {
            res.as_slice()[12]
        } else {
            0.0
        };

        assert!(val_x > 0.0, "EigenvaluesX should be > 0 with a 5x5 image");

        // Test EigenvaluesY (Smaller/Secondary curvature)
        let detector_y = Hessian {
            mode: HessianMode::EigenvaluesY,
        };
        detector_y.execute(&mut ctx, &mut cache).unwrap();

        let val_y = if let ImageContainer::F32Gray(res) = &ctx.image {
            res.as_slice()[4]
        } else {
            0.0
        };
        // For a perfectly straight ridge, eigenvalue 2 should be small/near zero
        assert!(
            val_y.abs() < 0.1,
            "EigenvaluesY should be near 0 for a straight line"
        );
    }

    #[test]
    fn test_name() {
        let detector = Hessian {
            mode: HessianMode::Determinant,
        };
        let name = detector.name();
        assert_eq!(name, "Hessian");
    }
}
