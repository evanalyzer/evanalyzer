//! # morphological_transformation
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-01
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

// Derived from kornia-rs (Apache-2.0) - see their headers.
mod kernels;
mod ops;

use crate::pipeline::pipeline_cache::GlobalPipelineCache;
use crate::{
    algos::{ExecutionScope, ImageAlgorithm},
    image::ImageContainer,
    pipeline::pipeline_context::PipelineContext,
};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use evanalyzer_cfg::settings::pipeline_command::CommandCategory;
use kernels::Kernel;
use kornia_image::Image;
use kornia_imgproc::padding::PaddingMode;
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
/// Its place in the pipeline decides what it works on: before a
/// segmentation on the image, after one (e.g. a Threshold) on the
/// segmentation mask - every class transformed as its own mask: classes grow
/// into background only (Dilate) and touching classes shrink away from each
/// other equally (Erode), whatever their class numbers.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing, instance_segmentation")]
pub struct MorphologicalCommand {
    /// The transformation type (e.g., Dilate, Erode).
    pub op: MorphOps,

    /// The diameter of the structuring element in pixels.
    /// Must be an odd number (e.g., 3, 5, 7), so the kernel has a centre
    /// pixel and is symmetric - an even size would shift objects.
    #[cmdsmeta(
        default = 3,
        min = 3,
        max = 101,
        step = 2,
        display_name = "Kernel size",
        summary = true
    )]
    pub kernel_size: usize,

    /// The geometric profile of the structuring element.
    #[cmdsmeta(visibility = Advanced)]
    pub kernel_shape: KernelShapes,
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
    /// Returns [`InternalErrors::InvalidArgument`] for an even `kernel_size`
    /// (0 included): the kernel would have no centre pixel and shift objects.
    /// Returns [`InternalErrors::FormatMismatch`] if the image is not `F32Gray`,
    /// as morphological ops on floats require specific handling of infinity/NaN.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.kernel_size % 2 == 0 {
            return Err(InternalErrors::InvalidArgument(format!(
                "Morphological Transform: the kernel size must be an odd number (3, 5, 7, ...), got {}",
                self.kernel_size
            )));
        }
        // The role the pipeline runs this step in (see `step_roles`): after a
        // segmentation it works on the mask, else on the image.
        if ctx.role == Some(CommandCategory::InstanceSegmentation) {
            let (labels, scratch) = ctx.get_segmentation_map_u32_buf()?;
            self.apply_morph_u32(labels, scratch)?;
            ctx.swap_scratch_with_segmentations()?;
            return Ok(());
        }

        match ctx.image.as_ref() {
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
        "Morphological Transform"
    }

    /// Measured by `tests::tile_ram` (see the trait method).
    fn ram_per_tile_pixel(&self) -> f64 {
        8.0
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata {
            cite_key: "serra1982image",
            title: "Image Analysis and Mathematical Morphology",
            authors: &["Jean Serra"],
            year: 1982,
            container: Some("Academic Press"),
            doi: None,
            url: None,
            pages: None,
        }]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }
}

impl MorphologicalCommand {
    /// Creates the specific kernel structure for the library
    fn get_kernel(&self) -> Kernel {
        let shape = match self.kernel_shape {
            KernelShapes::Cross => kernels::KernelShape::Cross {
                size: self.kernel_size,
            },
            KernelShapes::Ellipse => kernels::KernelShape::Ellipse {
                width: self.kernel_size,
                height: self.kernel_size,
            },
            KernelShapes::Box => kernels::KernelShape::Box {
                size: self.kernel_size,
            },
        };
        Kernel::new(shape)
    }

    /// Morphology for Floating Point (Intensity images)
    fn apply_morph_f32<const C: usize>(
        &self,
        input: &Image<f32, C>,
        output: &mut Image<f32, C>,
    ) -> Result<(), InternalErrors> {
        let kernel = self.get_kernel();
        let pad_val: [f32; C] = [0.0; C];

        match self.op {
            MorphOps::Dilate => ops::dilate(input, output, &kernel, PaddingMode::Constant, pad_val)
                .map_err(InternalErrors::from_kornia)?,
            MorphOps::Erode => ops::erode(input, output, &kernel, PaddingMode::Constant, pad_val)
                .map_err(InternalErrors::from_kornia)?,
            MorphOps::Open => ops::open(input, output, &kernel, PaddingMode::Constant, pad_val)
                .map_err(InternalErrors::from_kornia)?,
            MorphOps::Close => ops::close(input, output, &kernel, PaddingMode::Constant, pad_val)
                .map_err(InternalErrors::from_kornia)?,
        }
        Ok(())
    }

    /// Morphology for label maps (class ids, 0 = background): every class
    /// transformed as its own mask - see `ops::transform_labels`.
    fn apply_morph_u32(
        &self,
        input: &Image<u32, 1>,
        output: &mut Image<u32, 1>,
    ) -> Result<(), InternalErrors> {
        let kernel = self.get_kernel();
        ops::transform_labels(self.op, input, output, &kernel, PaddingMode::Constant, [0])
            .map_err(InternalErrors::from_kornia)
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
        let mut img = Image::<f32, 1>::from_size_val(size, 0.0)?;
        img.set_pixel(2, 2, 0, 1.0)?;

        // 2. Setup the Command (3x3 Box Dilation)
        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };

        // 3. Setup Context
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        // 4. Execute
        cmd.execute(&mut ctx, &mut cache)?;

        // 5. Verify results
        if let ImageContainer::F32Gray(out_img) = ctx.image.as_ref() {
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
        let mut img = Image::<u32, 1>::from_size_val(size, 0)?;
        img.set_pixel(2, 2, 0, 7)?;

        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.segmentation_map = Some(img);
        ctx.role = Some(CommandCategory::InstanceSegmentation);

        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };

        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())?;
        let labels = ctx.segmentation_map.as_ref().expect("No labels found");
        // Check that ID 7 has spread to neighbor (1,1)
        assert_eq!(*labels.get_pixel(1, 1, 0).unwrap(), 7);
        // Check that center is still 7
        assert_eq!(*labels.get_pixel(2, 2, 0).unwrap(), 7);
        // Check that corner is still 0
        assert_eq!(*labels.get_pixel(0, 0, 0).unwrap(), 0);

        Ok(())
    }

    /// Outside the Instance Segmentation role (before a segmentation, or
    /// outside a pipeline) it works on the image and leaves the mask alone.
    #[test]
    fn without_the_mask_role_the_image_is_transformed_and_the_mask_kept() {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut img = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        img.set_pixel(2, 2, 0, 1.0).unwrap();
        for role in [None, Some(CommandCategory::Preprocess)] {
            let mut ctx = PipelineContext::new_from_image_test(img.clone()).unwrap();
            let mut labels = Image::<u32, 1>::from_size_val(size, 0).unwrap();
            labels.set_pixel(2, 2, 0, 7).unwrap();
            ctx.segmentation_map = Some(labels.clone());
            ctx.role = role;
            let cmd = MorphologicalCommand {
                op: MorphOps::Dilate,
                kernel_size: 3,
                kernel_shape: KernelShapes::Box,
            };
            cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
                .unwrap();
            let ImageContainer::F32Gray(out) = ctx.image.as_ref() else {
                panic!("not F32Gray");
            };
            assert_eq!(*out.get_pixel(1, 1, 0).unwrap(), 1.0, "{role:?}: dilated");
            assert_eq!(
                ctx.segmentation_map.as_ref().unwrap().as_slice(),
                labels.as_slice(),
                "{role:?}: mask untouched"
            );
        }
    }

    /// Builds the same "single labeled pixel" fixture as
    /// `test_label_morphology`, for exercising `apply_morph_u32`'s
    /// Erode/Open/Close arms - `test_label_morphology` only covers Dilate,
    /// leaving the rest of that match untested.
    fn label_fixture() -> PipelineContext {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut img = Image::<u32, 1>::from_size_val(size, 0).unwrap();
        img.set_pixel(2, 2, 0, 7).unwrap();
        let mut ctx = PipelineContext::new_test::<F32Gray>(size).unwrap();
        ctx.segmentation_map = Some(img);
        ctx.role = Some(CommandCategory::InstanceSegmentation);
        ctx
    }

    #[test]
    fn test_label_morphology_erode_shrinks_a_single_pixel_label_to_background()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = label_fixture();
        let cmd = MorphologicalCommand {
            op: MorphOps::Erode,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())?;
        let labels = ctx.segmentation_map.as_ref().expect("no labels found");
        // A single labeled pixel has no fully-labeled neighborhood, so erosion
        // (the neighborhood minimum) wipes it out entirely.
        assert_eq!(*labels.get_pixel(2, 2, 0).unwrap(), 0);
        Ok(())
    }

    #[test]
    fn test_label_morphology_open_removes_the_single_pixel_label()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = label_fixture();
        let cmd = MorphologicalCommand {
            op: MorphOps::Open,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())?;
        let labels = ctx.segmentation_map.as_ref().expect("no labels found");
        // Open = erode then dilate; eroding away the single-pixel label
        // leaves nothing for the dilate pass to expand back.
        assert_eq!(*labels.get_pixel(2, 2, 0).unwrap(), 0);
        Ok(())
    }

    #[test]
    fn test_label_morphology_close_preserves_and_slightly_grows_the_label()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ctx = label_fixture();
        let cmd = MorphologicalCommand {
            op: MorphOps::Close,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())?;
        let labels = ctx.segmentation_map.as_ref().expect("no labels found");
        // Close = dilate then erode; the center pixel survives either way.
        assert_eq!(*labels.get_pixel(2, 2, 0).unwrap(), 7);
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

        let img = Image::<f32, 3>::from_size_slice(size, &data)?;
        let mut ctx = PipelineContext::new_from_image_test_rgb(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        // 2. Test Cross kernel (4-connectivity)
        let cmd_cross = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Cross,
        };
        cmd_cross.execute(&mut ctx, &mut cache)?;

        if let ImageContainer::F32Rgb(out) = ctx.image.as_ref() {
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
        };
        cmd_ellipse.execute(&mut ctx, &mut cache)?;

        // Ellipse with size 3 is often equivalent to Box, verify center
        if let ImageContainer::F32Rgb(out) = ctx.image.as_ref() {
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
        let img = Image::<u32, 1>::from_size_slice(size, &data).unwrap();

        // Setup context with a U32 image
        let mut ctx = PipelineContext::new_from_u32_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();

        let cmd = MorphologicalCommand {
            op: MorphOps::Dilate,
            kernel_size: 3,
            kernel_shape: KernelShapes::Box,
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(matches!(result, Err(InternalErrors::FormatMismatch { .. })));
    }

    // Helper to create a 3x3 image with a center spike
    fn create_spike_image() -> Image<f32, 1> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let mut img = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
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
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let img = match ctx.image.as_ref() {
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
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let img = match ctx.image.as_ref() {
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
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let img = match ctx.image.as_ref() {
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
        };
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();

        let img = match ctx.image.as_ref() {
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
        };
        assert_eq!(cmd.name(), "Morphological Transform");
    }

    // --- Kernel size, and Threshold followed by Morphological -------------

    #[test]
    fn even_kernel_sizes_are_rejected_odd_ones_run() {
        for (size, ok) in [
            (0, false),
            (1, true),
            (2, false),
            (3, true),
            (4, false),
            (5, true),
        ] {
            let mut ctx = label_fixture();
            let cmd = MorphologicalCommand {
                op: MorphOps::Dilate,
                kernel_size: size,
                kernel_shape: KernelShapes::Box,
            };
            let result = cmd.execute(&mut ctx, &mut GlobalPipelineCache::default());
            if ok {
                assert!(result.is_ok(), "size {size}: {result:?}");
            } else {
                assert!(
                    matches!(result, Err(InternalErrors::InvalidArgument(_))),
                    "size {size}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn the_settings_default_kernel_size_is_a_valid_odd_size() {
        let settings =
            evanalyzer_cfg::settings::pipeline_command_settings::MorphologicalCommandSettings::default();
        assert_eq!(settings.kernel_size, 3);
        let cmd = MorphologicalCommand::from(settings);
        let mut ctx = label_fixture();
        cmd.execute(&mut ctx, &mut GlobalPipelineCache::default())
            .unwrap();
    }

    /// Runs a manual Threshold (relative ranges → classes) on `img`, then
    /// `morph` on its class map.
    fn threshold_then_morph(
        img: Image<f32, 1>,
        ranges: &[(f32, f32, u32)],
        morph: &MorphologicalCommand,
    ) -> Vec<u32> {
        use crate::algos::segmentation::threshold::{
            Threshold, ThresholdEntry, ThresholdMethod, ThresholdValueSource,
        };
        use evanalyzer_cfg::core_types::{PixelUnits, SegmentationClass};
        let mut ctx = PipelineContext::new_from_image_test(img).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let thresholds = ranges
            .iter()
            .map(|&(min, max, class)| ThresholdEntry {
                method: ThresholdMethod::Manual,
                min_threshold: min,
                max_threshold: max,
                segmentation_class: SegmentationClass(class),
                unit: PixelUnits::Relative,
                value_source: ThresholdValueSource::ActualImage,
            })
            .collect();
        Threshold { thresholds }
            .execute(&mut ctx, &mut cache)
            .unwrap();
        // After a threshold the pipeline runs it on the mask.
        ctx.role = Some(CommandCategory::InstanceSegmentation);
        morph.execute(&mut ctx, &mut cache).unwrap();
        ctx.segmentation_map.as_ref().unwrap().as_slice().to_vec()
    }

    #[test]
    fn morphology_after_threshold_does_not_shift_objects() {
        // A bright 7x7 square in the middle of a 21x21 image: every operation
        // with every (symmetric) kernel must give a result that is still
        // centred - mirror symmetric in x and y. The off-centre ellipse
        // shifted it one pixel up-left in Open/Close.
        let n = 21;
        let mut img = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: n,
                height: n,
            },
            0.0,
        )
        .unwrap();
        for y in 7..14 {
            for x in 7..14 {
                img.set_pixel(x, y, 0, 1.0).unwrap();
            }
        }
        for op in [
            MorphOps::Dilate,
            MorphOps::Erode,
            MorphOps::Open,
            MorphOps::Close,
        ] {
            for shape in [
                KernelShapes::Box,
                KernelShapes::Cross,
                KernelShapes::Ellipse,
            ] {
                for kernel_size in [3, 5, 7] {
                    let cmd = MorphologicalCommand {
                        op,
                        kernel_size,
                        kernel_shape: shape,
                    };
                    let out = threshold_then_morph(img.clone(), &[(0.5, 1.0, 1)], &cmd);
                    for y in 0..n {
                        for x in 0..n {
                            let v = out[y * n + x];
                            assert_eq!(
                                v,
                                out[y * n + (n - 1 - x)],
                                "{op:?} {shape:?} {kernel_size}: x shifted"
                            );
                            assert_eq!(
                                v,
                                out[(n - 1 - y) * n + x],
                                "{op:?} {shape:?} {kernel_size}: y shifted"
                            );
                        }
                    }
                    // Open and Close of a square by a kernel smaller than it
                    // keep its centre and never reach the image border.
                    assert_eq!(out[10 * n + 10], 1, "{op:?} {shape:?} {kernel_size}");
                    assert_eq!(out[0], 0, "{op:?} {shape:?} {kernel_size}");
                }
            }
        }
    }

    #[test]
    fn morphology_after_a_threshold_with_two_touching_classes_treats_both_alike() {
        // Left half of a block dim (class 2), right half bright (class 9).
        let (w, h) = (14, 7);
        let mut img = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            0.0,
        )
        .unwrap();
        for y in 1..6 {
            for x in 1..13 {
                img.set_pixel(x, y, 0, if x < 7 { 0.4 } else { 0.9 })
                    .unwrap();
            }
        }
        let ranges = [(0.2, 0.6, 2), (0.6, 1.0, 9)];
        let run = |op| {
            let cmd = MorphologicalCommand {
                op,
                kernel_size: 3,
                kernel_shape: KernelShapes::Box,
            };
            threshold_then_morph(img.clone(), &ranges, &cmd)
        };
        let row = |v: &[u32]| v[3 * w..4 * w].to_vec();
        // Erode: both classes give up the column at their shared border
        // (the window minimum let class 2 eat into class 9).
        assert_eq!(
            row(&run(MorphOps::Erode)),
            [0, 0, 2, 2, 2, 2, 0, 0, 9, 9, 9, 9, 0, 0]
        );
        // Dilate: both grow into the background only (the window maximum let
        // class 9 overwrite a column of class 2).
        assert_eq!(
            row(&run(MorphOps::Dilate)),
            [2, 2, 2, 2, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9]
        );
        // Close: no gap opened between them, nothing overwritten.
        let threshold_only = threshold_then_morph(
            img.clone(),
            &ranges,
            &MorphologicalCommand {
                op: MorphOps::Dilate,
                kernel_size: 1,
                kernel_shape: KernelShapes::Box,
            },
        );
        assert_eq!(run(MorphOps::Close), threshold_only);
    }
}
