use crate::{
    ImagePlane,
    image::{ImageContainer, ImageTypeMarker, ManagedImage, PixelSizes},
    pipeline::pipeline::PipelineImageMeta,
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_apriltag::utils::Point2d;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

pub struct PipelineContext {
    pub image_meta: PipelineImageMeta,
    // The "main" image being processed
    pub image: ImageContainer,
    // A secondary buffer used as a workspace to avoid re-allocation
    pub scratch_pad: ImageContainer,
    // Instance map: Every unique object gets its own unique ID
    pub instance_map: Option<Image<u32, 1, CpuAllocator>>,
    // Segmentation map: Every pixel is assigned a category label (e.g., "Background", "Cell", "Nucleus").
    pub segmentation_map: Option<Image<u32, 1, CpuAllocator>>,
}

impl PipelineContext {
    pub fn full_image_size(&self) -> ImageSize {
        self.image_meta.full_image_width
    }

    pub fn pixel_sizes(&self) -> &PixelSizes {
        return &self.image_meta.pixel_sizes;
    }

    pub fn new<T: ImageTypeMarker>(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
        image_meta: PipelineImageMeta,
    ) -> Result<Self, InternalErrors> {
        let pixel_count: usize = size.width * size.height;
        Ok(Self {
            image_meta,
            image: T::create_container(size, tile_offset, plane)?,
            scratch_pad: T::create_container(size, tile_offset, plane)?,
            segmentation_map: Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
            ),
            instance_map: Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
            ),
        })
    }

    pub fn new_from_image(
        image_meta: PipelineImageMeta,
        image: ImageContainer,
    ) -> Result<Self, InternalErrors> {
        let empty_image = image.clone_empty();
        let size = image.size();
        let pixel_count: usize = size.width * size.height;
        Ok(Self {
            image_meta,
            image: image,
            scratch_pad: empty_image,
            segmentation_map: Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
            ),
            instance_map: Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
            ),
        })
    }

    /// Swaps the scratch pad and the main image
    pub fn swap(&mut self) -> Result<(), InternalErrors> {
        // Rule: The scratch_pad cannot become the 'image' if it's U32
        if let ImageContainer::U32(_) = self.scratch_pad {
            return Err(InternalErrors::FormatMismatch {
                expected: "F32Gray or F32rgb expected".into(),
                found: format!("Input: {:?}", self.scratch_pad),
            });
        }

        // Perform the O(1) pointer swap
        std::mem::swap(&mut self.image, &mut self.scratch_pad);
        Ok(())
    }

    pub fn swap_scratch_with_segmentations(&mut self) -> Result<(), InternalErrors> {
        let segmentation_map = self
            .segmentation_map
            .as_mut()
            .ok_or(InternalErrors::Internal(
                "Cannot swap: Segmentation buffer not initialized".into(),
            ))?;

        // 2. Destructure the scratch_pad to get the other image
        if let ImageContainer::U32(ref mut scratch_img) = self.scratch_pad {
            // 3. Now the types match: &mut Image and &mut Image
            std::mem::swap(segmentation_map, scratch_img);
            Ok(())
        } else {
            Err(InternalErrors::FormatMismatch {
                expected: "U32Segmentation scratchpad".into(),
                found: format!("{:?}", self.scratch_pad),
            })
        }
    }

    pub fn get_f32_gray_image(&self) -> Result<&ManagedImage<f32, 1>, InternalErrors> {
        match &self.image {
            ImageContainer::F32Gray(img) => Ok(img),
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", self.image),
            }),
        }
    }

    pub fn get_f32_gray_image_mut(
        &mut self,
    ) -> Result<&mut Image<f32, 1, CpuAllocator>, InternalErrors> {
        // Use a guard pattern to check if the variant is wrong
        if !matches!(self.image, ImageContainer::F32Gray(_)) {
            // Here, the immutable borrow for the 'if' is finished,
            // so we can safely borrow it again for the error message.
            return Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", self.image),
            });
        }

        // Now we know it's the right variant, perform the mutable match
        match &mut self.image {
            ImageContainer::F32Gray(img) => Ok(img),
            _ => unreachable!(), // We checked this above, so this is safe
        }
    }

    pub fn get_f32_gray_image_and_prep_scratch<M: ImageTypeMarker>(
        &mut self,
    ) -> Result<(&Image<f32, 1, CpuAllocator>, &mut M::ImageRef), InternalErrors> {
        // Get the size first (this borrow ends immediately)
        let img = self.get_f32_gray_image()?;
        let size = img.size();

        let Some(plane) = img.plane else {
            return Err(InternalErrors::Generic(
                "Image has no plane information!".into(),
            ));
        };

        // Modify scratch_pad if needed
        if let Some(new_buffer) = self.prepare_scratch::<M>(size, img.tile_offset, plane)? {
            self.scratch_pad = M::wrap(new_buffer);
        }

        let input = match &self.image {
            ImageContainer::F32Gray(img) => Ok(img),
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", self.image),
            }),
        };

        let scratch = M::get_ref_mut(&mut self.scratch_pad)
            .ok_or_else(|| InternalErrors::Generic("Type mismatch in scratchpad".to_string()))?;

        Ok((input?, scratch))
    }

    pub fn get_scratch_as_f32_gray(&mut self) -> &mut Image<f32, 1, CpuAllocator> {
        if !matches!(self.scratch_pad, ImageContainer::F32Gray(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::F32Gray(ManagedImage {
                data: Image::new(size, vec![0.0; size.width * size.height], CpuAllocator).unwrap(),
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }

        match &mut self.scratch_pad {
            ImageContainer::F32Gray(img) => img,
            _ => unreachable!(),
        }
    }

    pub fn get_scratch_as_f32_rgb(&mut self) -> &mut Image<f32, 3, CpuAllocator> {
        if !matches!(self.scratch_pad, ImageContainer::F32Rgb(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::F32Rgb(ManagedImage {
                data: Image::new(size, vec![0.0; size.width * size.height], CpuAllocator).unwrap(),
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }

        match &mut self.scratch_pad {
            ImageContainer::F32Rgb(img) => img,
            _ => unreachable!(),
        }
    }

    pub fn get_scratch_as_u32(&mut self) -> &mut Image<u32, 1, CpuAllocator> {
        if !matches!(self.scratch_pad, ImageContainer::U32(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::U32(ManagedImage {
                data: Image::new(size, vec![0u32; size.width * size.height], CpuAllocator).unwrap(),
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }

        match &mut self.scratch_pad {
            ImageContainer::U32(img) => img,
            _ => unreachable!(),
        }
    }

    fn prepare_u32_scratch(&mut self) -> Result<(), InternalErrors> {
        if !matches!(self.scratch_pad, ImageContainer::U32(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::U32(ManagedImage {
                data: Image::new(size, vec![0u32; size.width * size.height], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }
        Ok(())
    }

    pub fn prepare_f32_gray_scratch(&mut self) -> Result<(), InternalErrors> {
        if !matches!(self.scratch_pad, ImageContainer::F32Gray(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::F32Gray(ManagedImage {
                data: Image::new(size, vec![0f32; size.width * size.height], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }
        Ok(())
    }

    fn prepare_f32_rgb_scratch(&mut self) -> Result<(), InternalErrors> {
        if !matches!(self.scratch_pad, ImageContainer::F32Rgb(_)) {
            let size = self.image.size();
            self.scratch_pad = ImageContainer::F32Rgb(ManagedImage {
                data: Image::new(size, vec![0f32; size.width * size.height], CpuAllocator)
                    .map_err(InternalErrors::from_kornia)?,
                tile_offset: self.image.tile_offset(),
                plane: self.image.plane(),
            });
        }
        Ok(())
    }

    pub fn get_segmentation_map_u32_buf(
        &mut self,
    ) -> Result<
        (
            &Image<u32, 1, CpuAllocator>,
            &mut Image<u32, 1, CpuAllocator>,
        ),
        InternalErrors,
    > {
        self.prepare_segmentation_map()?;
        self.prepare_u32_scratch()?;
        if let ImageContainer::U32(ref mut scratch_img) = self.scratch_pad {
            let segmentation_ref = self.segmentation_map.as_ref().unwrap(); // We just created it this should never happen
            Ok((segmentation_ref, scratch_img))
        } else {
            Err(InternalErrors::FormatMismatch {
                expected: "This should never happen!".into(),
                found: format!("Input: {:?}", self.image),
            })
        }
    }

    pub fn prepare_segmentation_map(&mut self) -> Result<(), InternalErrors> {
        // Ensure Segmentation exists. If not, create it using the current image size.
        if self.segmentation_map.is_none() {
            let size = self.image.size();
            let pixel_count = size.width * size.height;
            self.segmentation_map = Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(|e| InternalErrors::Internal(e.to_string()))?,
            );
        }
        Ok(())
    }

    pub fn get_gray_img_gray_buf(
        &mut self,
    ) -> Result<
        (
            &Image<f32, 1, CpuAllocator>,
            &mut Image<f32, 1, CpuAllocator>,
        ),
        InternalErrors,
    > {
        self.prepare_f32_gray_scratch()?;

        match (&self.image, &mut self.scratch_pad) {
            (ImageContainer::F32Gray(img), ImageContainer::F32Gray(scratch)) => Ok((img, scratch)),
            (img_cont, _) if !matches!(img_cont, ImageContainer::F32Gray(_)) => {
                Err(InternalErrors::FormatMismatch {
                    expected: "F32Gray".into(),
                    found: format!("{:?}", img_cont),
                })
            }
            _ => Err(InternalErrors::Generic(
                "Scratchpad initialization failed".into(),
            )),
        }
    }

    pub fn get_rgb_img_rgb_buf(
        &mut self,
    ) -> Result<
        (
            &Image<f32, 3, CpuAllocator>,
            &mut Image<f32, 3, CpuAllocator>,
        ),
        InternalErrors,
    > {
        self.prepare_f32_rgb_scratch()?;

        match (&self.image, &mut self.scratch_pad) {
            (ImageContainer::F32Rgb(img), ImageContainer::F32Rgb(scratch)) => Ok((img, scratch)),
            (img_cont, _) if !matches!(img_cont, ImageContainer::F32Rgb(_)) => {
                Err(InternalErrors::FormatMismatch {
                    expected: "F32Rgb".into(),
                    found: format!("{:?}", img_cont),
                })
            }
            _ => Err(InternalErrors::Generic(
                "Scratchpad initialization failed".into(),
            )),
        }
    }

    fn prepare_scratch<M: ImageTypeMarker>(
        &self,
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<Option<M::ImageRef>, InternalErrors> {
        //  Check if current buffer is usable (same type AND same size)
        let is_compatible =
            M::matches_container(&self.scratch_pad) && self.scratch_pad.size() == size;
        if is_compatible {
            // Current scratchpad is fine, return None (no new buffer needed)
            Ok(None)
        } else {
            //  Need a new buffer.
            // We call the trait's static constructor.
            let new_image = M::create_image(size, tile_offset, plane)?;
            Ok(Some(new_image))
        }
    }

    pub fn get_segmentation_map(&self) -> Result<&Image<u32, 1, CpuAllocator>, InternalErrors> {
        self.segmentation_map
            .as_ref()
            .ok_or(InternalErrors::FormatMismatch {
                expected: "Initialized segmentation buffer".into(),
                found: "None (Buffer not initialized)".into(),
            })
    }

    pub fn get_f32_gray_and_segmentation_mask_mut(
        &mut self,
    ) -> Result<
        (
            &Image<f32, 1, CpuAllocator>,
            &mut Image<u32, 1, CpuAllocator>,
        ),
        InternalErrors,
    > {
        let image = match &self.image {
            ImageContainer::F32Gray(img) => Ok(img),
            _ => Err(InternalErrors::FormatMismatch {
                expected: "F32Gray".into(),
                found: format!("{:?}", self.image),
            }),
        }?;

        if self.segmentation_map.is_none() {
            let size = self.image.size();
            let pixel_count = size.width * size.height;
            self.segmentation_map = Some(
                Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                    .map_err(|e| InternalErrors::Internal(e.to_string()))?,
            );
        }
        let segmentation_ref =
            self.segmentation_map
                .as_mut()
                .ok_or(InternalErrors::FormatMismatch {
                    expected: "Initialized segmentation buffer".into(),
                    found: "None (Buffer not initialized)".into(),
                })?;
        Ok((&image, segmentation_ref))
    }

    pub fn get_instance_map(&self) -> Result<&Image<u32, 1, CpuAllocator>, InternalErrors> {
        let classes = self
            .instance_map
            .as_ref()
            .ok_or(InternalErrors::FormatMismatch {
                expected: "Initialized classes buffer".into(),
                found: "None (Buffer not initialized)".into(),
            })?;
        Ok(classes)
    }

    pub fn get_segmentation_and_instances_mut(
        &mut self,
        create_segmentation_if_not_exist: bool,
    ) -> Result<
        (
            &Image<u32, 1, CpuAllocator>,
            &mut Image<u32, 1, CpuAllocator>,
        ),
        InternalErrors,
    > {
        let size = self.image.size();
        let pixel_count = size.width * size.height;

        if self.segmentation_map.is_none() {
            if create_segmentation_if_not_exist {
                self.segmentation_map = Some(
                    Image::new(size, vec![0u32; pixel_count], CpuAllocator).map_err(|_e| {
                        InternalErrors::FormatMismatch {
                            expected: "Initialized segmentation buffer".into(),
                            found: "None (Buffer not initialized)".into(),
                        }
                    })?,
                );
            } else {
                return Err(InternalErrors::FormatMismatch {
                    expected: "Initialized segmentation buffer".into(),
                    found: "None (Buffer not initialized)".into(),
                });
            }
        }

        if self.instance_map.is_none() {
            self.instance_map = Some(
                Image::new(size, vec![0u32; pixel_count], CpuAllocator).map_err(|_e| {
                    InternalErrors::FormatMismatch {
                        expected: "Initialized instance buffer".into(),
                        found: "None (Buffer not initialized)".into(),
                    }
                })?,
            );
        }

        // We get a shared reference from segmentarion and a mut reference from classes
        let segmentation_ref = self.segmentation_map.as_ref().unwrap();
        let instances_mut = self.instance_map.as_mut().unwrap();

        Ok((segmentation_ref, instances_mut))
    }

    pub fn get_image_size(&self) -> ImageSize {
        match &self.image {
            ImageContainer::F32Gray(image) => image.size(),
            ImageContainer::F32Rgb(image) => image.size(),
            ImageContainer::U32(image) => image.size(),
        }
    }

    pub fn get_image_tile_offset(&self) -> Point2d {
        match &self.image {
            ImageContainer::F32Gray(image) => image.tile_offset,
            ImageContainer::F32Rgb(image) => image.tile_offset,
            ImageContainer::U32(image) => image.tile_offset,
        }
    }

    pub fn get_image_plane(&self) -> Option<ImagePlane> {
        match &self.image {
            ImageContainer::F32Gray(image) => image.plane,
            ImageContainer::F32Rgb(image) => image.plane,
            ImageContainer::U32(image) => image.plane,
        }
    }

    pub fn does_segmentation_map_exist(&self) -> bool {
        return self.segmentation_map.is_none();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageContainer, image::PixelSizes, pipeline::pipeline_context::PipelineContext};

    impl PipelineContext {
        pub fn new_from_image_test(
            input_img: Image<f32, 1, CpuAllocator>,
        ) -> Result<Self, InternalErrors> {
            let image = ImageContainer::F32Gray(ManagedImage {
                data: input_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            });

            let empty_image = image.clone_empty();
            let size = image.size();
            let pixel_count: usize = size.width * size.height;
            Ok(Self {
                image: image,
                scratch_pad: empty_image,
                segmentation_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                instance_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                image_meta: PipelineImageMeta {
                    image_tile_info: crate::ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: size.width,
                        height: size.height,
                    },
                    full_image_width: ImageSize {
                        width: size.width,
                        height: size.height,
                    },
                    is_rgb: false,
                    nr_of_bits: 8,
                    pixel_sizes: PixelSizes {
                        px_size_x: 1.0,
                        px_size_y: 1.0,
                        px_size_z: 1.0,
                    },
                },
            })
        }

        pub fn new_from_image_test_rgb(
            input_img: Image<f32, 3, CpuAllocator>,
        ) -> Result<Self, InternalErrors> {
            let image = ImageContainer::F32Rgb(ManagedImage {
                data: input_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            });

            let empty_image = image.clone_empty();
            let size = image.size();
            let pixel_count: usize = size.width * size.height;
            Ok(Self {
                image: image,
                scratch_pad: empty_image,
                segmentation_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                instance_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                image_meta: PipelineImageMeta {
                    image_tile_info: crate::ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: size.width,
                        height: size.height,
                    },
                    full_image_width: ImageSize {
                        width: size.width,
                        height: size.height,
                    },
                    is_rgb: false,
                    nr_of_bits: 8,
                    pixel_sizes: PixelSizes {
                        px_size_x: 1.0,
                        px_size_y: 1.0,
                        px_size_z: 1.0,
                    },
                },
            })
        }

        pub fn new_from_u32_image_test(
            input_img: Image<u32, 1, CpuAllocator>,
        ) -> Result<Self, InternalErrors> {
            let image = ImageContainer::U32(ManagedImage {
                data: input_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            });

            let empty_image = image.clone_empty();
            let size = image.size();
            let pixel_count: usize = size.width * size.height;
            Ok(Self {
                image: image,
                scratch_pad: empty_image,
                segmentation_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                instance_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                image_meta: PipelineImageMeta {
                    image_tile_info: crate::ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: size.width,
                        height: size.height,
                    },
                    full_image_width: ImageSize {
                        width: size.width,
                        height: size.height,
                    },
                    is_rgb: false,
                    nr_of_bits: 8,
                    pixel_sizes: PixelSizes {
                        px_size_x: 1.0,
                        px_size_y: 1.0,
                        px_size_z: 1.0,
                    },
                },
            })
        }

        pub fn new_test<T: ImageTypeMarker>(
            size: kornia_image::ImageSize,
        ) -> Result<Self, InternalErrors> {
            let pixel_count: usize = size.width * size.height;
            Ok(Self {
                image: T::create_container(
                    size,
                    Point2d { x: 0, y: 0 },
                    ImagePlane { z: 0, c: 0, t: 0 },
                )?,
                scratch_pad: T::create_container(
                    size,
                    Point2d { x: 0, y: 0 },
                    ImagePlane { z: 0, c: 0, t: 0 },
                )?,
                segmentation_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                instance_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                image_meta: PipelineImageMeta {
                    image_tile_info: crate::ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: size.width,
                        height: size.height,
                    },
                    full_image_width: ImageSize {
                        width: size.width,
                        height: size.height,
                    },
                    is_rgb: false,
                    nr_of_bits: 8,
                    pixel_sizes: PixelSizes {
                        px_size_x: 1.0,
                        px_size_y: 1.0,
                        px_size_z: 1.0,
                    },
                },
            })
        }

        pub fn new_test_with_offset<T: ImageTypeMarker>(
            size: kornia_image::ImageSize,
            full_image_size: ImageSize,
            offset: Point2d,
        ) -> Result<Self, InternalErrors> {
            let pixel_count: usize = size.width * size.height;
            Ok(Self {
                image: T::create_container(size, offset, ImagePlane { z: 0, c: 0, t: 0 })?,
                scratch_pad: T::create_container(size, offset, ImagePlane { z: 0, c: 0, t: 0 })?,
                segmentation_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                instance_map: Some(
                    Image::<u32, 1, CpuAllocator>::new(size, vec![0u32; pixel_count], CpuAllocator)
                        .map_err(InternalErrors::from_kornia)?,
                ),
                image_meta: PipelineImageMeta {
                    image_tile_info: crate::ImageTile {
                        offset_x: 0,
                        offset_y: 0,
                        width: size.width,
                        height: size.height,
                    },
                    full_image_width: full_image_size,
                    is_rgb: false,
                    nr_of_bits: 8,
                    pixel_sizes: PixelSizes {
                        px_size_x: 1.0,
                        px_size_y: 1.0,
                        px_size_z: 1.0,
                    },
                },
            })
        }
    }

    impl ImageContainer {
        pub fn new_f32_gray_from_image_test(
            input_img: Image<f32, 1, CpuAllocator>,
        ) -> ImageContainer {
            ImageContainer::F32Gray(ManagedImage {
                data: input_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            })
        }

        pub fn new_f32_rgb_from_image_test(
            input_img: Image<f32, 3, CpuAllocator>,
        ) -> ImageContainer {
            ImageContainer::F32Rgb(ManagedImage {
                data: input_img,
                tile_offset: Point2d { x: 0, y: 0 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            })
        }
    }
}
