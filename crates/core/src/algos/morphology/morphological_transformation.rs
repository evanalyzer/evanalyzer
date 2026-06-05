//! # morphological_transformation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::extlibs::libmorphology::{self, Kernel};
use crate::pipeline::pipeline_cache::PipelineCache;
use crate::{
    algos::ImageAlgorithm, image::ImageContainer, pipeline::pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::Image;
use kornia_imgproc::padding::PaddingMode;
use kornia_tensor::CpuAllocator;
use macros::CommandsMeta;

/// The specific morphological transformation to perform.
///
/// Morphological operations process images based on shapes, typically used to
/// remove noise, isolate individual elements, or join disparate elements.
#[derive(Debug, Clone, Copy)]
pub enum MorphOps {
    /// Expands the bright regions of an image. Useful for filling small holes.
    Dilate,
    /// Shrinks the bright regions of an image. Useful for removing small noise.
    Erode,
    /// An erosion followed by a dilation. Removes small bright spots (noise)
    /// while preserving the relative size of larger objects.
    Open,
    /// A dilation followed by an erosion. Fills small dark gaps or cracks
    /// within bright objects.
    Close,
}

/// The geometric structure of the kernel (structuring element).
#[derive(Debug, Clone, Copy)]
pub enum KernelShapes {
    /// A square/rectangular kernel. Dilates in all directions equally (8-connectivity).
    Box,
    /// A rounded kernel. Best for preserving the natural, circular shape of objects.
    Ellipse,
    /// A cross-shaped kernel. Only considers horizontal and vertical neighbors (4-connectivity).
    Cross,
}

/// A filter that applies mathematical morphology to an image.
///
/// Morphological operations use a structuring element (kernel) to probe
/// and modify the shapes within an image.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::{MorphologicalCommand, MorphOps, KernelShapes};
/// let clean_noise = MorphologicalCommand {
///     op: MorphOps::Open,
///     kernel_size: 3,
///     kernel_shape: KernelShapes::Ellipse,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct MorphologicalCommand {
    /// The transformation type (e.g., Dilate, Erode).
    pub op: MorphOps,

    /// The diameter of the structuring element in pixels.
    /// Must be an odd number (e.g., 3, 5, 7).
    pub kernel_size: usize,

    /// The geometric profile of the structuring element.
    pub kernel_shape: KernelShapes,

    /// If set the grayscale image instead of the labeld image is taken to perform a morphological transform
    pub use_grayscale: bool,
}
impl ImageAlgorithm for MorphologicalCommand {
    /// Executes the specified morphological transformation on the image.
    ///
    /// Morphology utilizes the `kernel_shape` to probe the image. The result for
    /// each pixel is the maximum (Dilation) or minimum (Erosion) value found
    /// within the neighborhood defined by the `kernel_size`.
    ///
    /// # Pipeline Logic
    /// 1. **Setup**: Extracts the source image and target scratch pad.
    /// 2. **Kernel Construction**: Generates a structuring element (SE) based
    ///    on the `kernel_shape` (Box, Ellipse, or Cross).
    /// 3. **Operation**:
    ///    - Single-pass: Dilate and Erode are computed in one step.
    ///    - Two-pass: Open and Close are implemented as sequential Dilate/Erode
    ///      calls to ensure the signal is correctly "filtered."
    /// 4. **Buffer Swap**: The processed image is moved to the primary context.
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::FormatMismatch`] if the image is not `F32Gray`,
    /// as morphological ops on floats require specific handling of infinity/NaN.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        if !self.use_grayscale {
            let (labels, scratch) = ctx.get_segmentation_map_u32_buf()?;
            self.apply_morph_u32(labels, scratch)?;
            ctx.swap_scratch_with_segmentations()?;
            return Ok(());
        }

        match &mut ctx.image {
            ImageContainer::F32Gray(_) => {
                let (labels, scratch) = ctx.get_gray_img_gray_buf()?;
                self.apply_morph_f32(labels, scratch)?;
                ctx.swap()?;
                Ok(())
            }
            ImageContainer::F32Rgb(_) => {
                let (labels, scratch) = ctx.get_rgb_img_rgb_buf()?;
                self.apply_morph_f32(labels, scratch)?;
                ctx.swap()?;
                Ok(())
            }
            _ => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "F32Gray or F32rgb expected".into(),
                    found: format!("Input: {:?}", ctx.image),
                });
            }
        }
    }

    fn name(&self) -> &'static str {
        "MorphologicalTransform"
    }
}

impl MorphologicalCommand {
    /// Creates the specific kernel structure for the library
    fn get_kernel(&self) -> Kernel {
        let shape = match self.kernel_shape {
            KernelShapes::Cross => libmorphology::KernelShape::Cross {
                size: self.kernel_size,
            },
            KernelShapes::Ellipse => libmorphology::KernelShape::Ellipse {
                width: self.kernel_size,
                height: self.kernel_size,
            },
            KernelShapes::Box => libmorphology::KernelShape::Box {
                size: self.kernel_size,
            },
        };
        Kernel::new(shape)
    }

    /// Morphology for Floating Point (Intensity images)
    fn apply_morph_f32<const C: usize>(
        &self,
        input: &Image<f32, C, CpuAllocator>,
        output: &mut Image<f32, C, CpuAllocator>,
    ) -> Result<(), InternalErrors> {
        let kernel = self.get_kernel();
        let pad_val: [f32; C] = [0.0; C];

        match self.op {
            MorphOps::Dilate => {
                libmorphology::dilate(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Erode => {
                libmorphology::erode(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Open => {
                libmorphology::open(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Close => {
                libmorphology::close(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
        }
        Ok(())
    }

    /// Morphology for Unsigned Integers (Label/Mask images)
    fn apply_morph_u32(
        &self,
        input: &Image<u32, 1, CpuAllocator>,
        output: &mut Image<u32, 1, CpuAllocator>,
    ) -> Result<(), InternalErrors> {
        let kernel = self.get_kernel();
        let pad_val: [u32; 1] = [0];

        match self.op {
            MorphOps::Dilate => {
                libmorphology::dilate(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Erode => {
                libmorphology::erode(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Open => {
                libmorphology::open(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
            MorphOps::Close => {
                libmorphology::close(input, output, &kernel, PaddingMode::Constant, pad_val)
                    .map_err(InternalErrors::from_kornia)?
            }
        }
        Ok(())
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::F32Gray;

    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_dilation_expansion() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Create a 5x5 black image with one white pixel in the center
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut img = Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator)?;
        img.set_pixel(2, 2, 0, 1.0)?;

        // 2. Setup the Command (3x3 Box Dilation)
        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };

        // 3. Setup Context
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = PipelineCache::default();

        // 4. Execute
        cmd.execute(&mut ctx, &mut cache)?;

        // 5. Verify results
        if let ImageContainer::F32Gray(out_img) = ctx.image {
            // In a 3x3 dilation, the 1.0 at (2,2) should expand to (1,1) through (3,3)
            assert_eq!(
                *out_img.get_pixel(1, 1, 0).unwrap(),
                1.0,
                "Top-left neighbor should be dilated"
            );
            assert_eq!(
                *out_img.get_pixel(2, 2, 0).unwrap(),
                1.0,
                "Center should remain 1.0"
            );
            assert_eq!(
                *out_img.get_pixel(3, 3, 0).unwrap(),
                1.0,
                "Bottom-right neighbor should be dilated"
            );

            // The corners of the 5x5 image should still be 0.0
            assert_eq!(
                *out_img.get_pixel(0, 0, 0).unwrap(),
                0.0,
                "Edge pixel should remain black"
            );
        } else {
            panic!("Resulting image container was not F32Gray");
        }

        Ok(())
    }

    #[test]
    fn test_label_morphology() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        // Input: Black image with one pixel of ID '7'
        let mut img = Image::<u32, 1, CpuAllocator>::from_size_val(size, 0, CpuAllocator)?;
        img.set_pixel(2, 2, 0, 7)?;

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.segmentation_map = Some(img);

        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: false,
        };

        cmd.execute(&mut ctx, &mut PipelineCache::default())?;
        let labels = ctx.segmentation_map.as_ref().expect("No labels found");
        // Check that ID 7 has spread to neighbor (1,1)
        assert_eq!(*labels.get_pixel(1, 1, 0).unwrap(), 7);
        // Check that center is still 7
        assert_eq!(*labels.get_pixel(2, 2, 0).unwrap(), 7);
        // Check that corner is still 0
        assert_eq!(*labels.get_pixel(0, 0, 0).unwrap(), 0);

        Ok(())
    }

    #[test]
    fn test_morph_rgb_ellipse_and_cross() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Create a 3x3 image with a center pixel
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        // RGB: (y*width + x)*3 + c
        let mut data = vec![0.0f32; 3 * 3 * 3];
        data[(1 * 3 + 1) * 3 + 0] = 1.0; // Center pixel, Red channel

        let img = Image::<f32, 3, _>::from_size_slice(size, &data, CpuAllocator)?;
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        let mut cache = PipelineCache::default();

        // 2. Test Cross kernel (4-connectivity)
        let cmd_cross = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Cross,
            use_grayscale: true,
        };
        cmd_cross.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Rgb(ref out) = ctx.image {
            assert_eq!(*out.get_pixel(1, 0, 0).unwrap(), 1.0); // Up
            assert_eq!(*out.get_pixel(0, 1, 0).unwrap(), 1.0); // Left (Cross doesn't hit diagonals)
            assert_eq!(
                *out.get_pixel(0, 0, 0).unwrap(),
                0.0,
                "Top-left diagonal should NOT be dilated"
            );
        }

        // 3. Test Ellipse kernel
        let cmd_ellipse = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Ellipse,
            use_grayscale: true,
        };
        cmd_ellipse.execute(&mut ctx, &mut cache)?;

        // Ellipse with size 3 is often equivalent to Box, verify center
        if let ImageContainer::F32Rgb(ref out) = ctx.image {
            assert_eq!(*out.get_pixel(1, 1, 0).unwrap(), 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_morph_format_mismatch_fails() {
        // Attempt to pass an unsupported image type (e.g., U32) to the F32-only logic
        let size = ImageSize {
            width: 1,
            height: 1,
        };
        let data = vec![0u32; 1];
        let img = Image::<u32, 1, _>::from_size_slice(size, &data, CpuAllocator).unwrap();

        // Setup context with a U32 image
        let mut ctx = PipelineContext::new_from_u32_image_test(img).unwrap();
        let mut cache = PipelineCache::default();

        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true, // Forces F32 logic
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(matches!(result, Err(InternalErrors::FormatMismatch { .. })));
    }

    // Helper to create a 3x3 image with a center spike
    fn create_spike_image() -> Image<f32, 1, CpuAllocator> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let mut img =
            Image::<f32, 1, CpuAllocator>::from_size_val(size, 0.0, CpuAllocator).unwrap();
        img.set_pixel(1, 1, 0, 1.0).unwrap();
        img
    }

    #[test]
    fn test_erode_removes_spike() {
        let mut ctx = PipelineContext::new_from_image_test(create_spike_image()).unwrap();
        let cmd = MorphologicalCommand {
            op: MorphOps::Erode,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };
        cmd.execute(&mut ctx, &mut PipelineCache::default())
            .unwrap();

        let img = match ctx.image {
            ImageContainer::F32Gray(managed_image) => managed_image,
            ImageContainer::F32Rgb(_) => todo!(),
            ImageContainer::U32(_) => todo!(),
        };
        assert_eq!(*img.get_pixel(1, 1, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_open_removes_spike() {
        let mut ctx = PipelineContext::new_from_image_test(create_spike_image()).unwrap();
        let cmd = MorphologicalCommand {
            op: MorphOps::Open,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };
        cmd.execute(&mut ctx, &mut PipelineCache::default())
            .unwrap();

        let img = match ctx.image {
            ImageContainer::F32Gray(managed_image) => managed_image,
            ImageContainer::F32Rgb(_) => todo!(),
            ImageContainer::U32(_) => todo!(),
        };
        assert_eq!(*img.get_pixel(1, 1, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_dilate_preserves_spike() {
        let mut ctx = PipelineContext::new_from_image_test(create_spike_image()).unwrap();
        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };
        cmd.execute(&mut ctx, &mut PipelineCache::default())
            .unwrap();

        let img = match ctx.image {
            ImageContainer::F32Gray(managed_image) => managed_image,
            ImageContainer::F32Rgb(_) => todo!(),
            ImageContainer::U32(_) => todo!(),
        };
        assert_eq!(*img.get_pixel(1, 1, 0).unwrap(), 1.0);
    }

    #[test]
    fn test_close_preserves_spike() {
        let mut ctx = PipelineContext::new_from_image_test(create_spike_image()).unwrap();
        let cmd = MorphologicalCommand {
            op: MorphOps::Close,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };
        cmd.execute(&mut ctx, &mut PipelineCache::default())
            .unwrap();

        let img = match ctx.image {
            ImageContainer::F32Gray(managed_image) => managed_image,
            ImageContainer::F32Rgb(_) => todo!(),
            ImageContainer::U32(_) => todo!(),
        };
        assert_eq!(*img.get_pixel(1, 1, 0).unwrap(), 1.0);
    }

    #[test]
    fn test_morphological_command_name() {
        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
            use_grayscale: true,
        };
        assert_eq!(cmd.name(), "MorphologicalTransform");
    }
}
