use crate::{
    ImagePlane,
    image::{ImageContainer, ManagedImage},
};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_apriltag::utils::Point2d;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;

pub struct F32Gray;
pub struct F32Rgb;

pub trait ImageTypeMarker: 'static {
    type ImageRef: 'static;

    // Allows us to create the image without knowing the concrete type inside the function
    fn create_image(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<Self::ImageRef, InternalErrors>;

    fn create_container(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<ImageContainer, InternalErrors>;

    // Helper to check if a container variant matches this marker
    fn matches_container(container: &ImageContainer) -> bool;

    // The "Magic" piece: converts the concrete image back into the Enum
    fn wrap(image: Self::ImageRef) -> ImageContainer;

    // Safe retrieval
    fn get_ref(container: &ImageContainer) -> Option<&Self::ImageRef>;

    // Safe retrieval
    fn get_ref_mut(container: &mut ImageContainer) -> Option<&mut Self::ImageRef>;
}

impl ImageTypeMarker for F32Gray {
    type ImageRef = ManagedImage<f32, 1>;

    fn create_image(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<Self::ImageRef, InternalErrors> {
        Ok(ManagedImage {
            data: Image::<f32, 1, CpuAllocator>::new(
                size,
                vec![0f32; size.width * size.height],
                CpuAllocator,
            )
            .map_err(|e| InternalErrors::Generic(e.to_string()))?,
            tile_offset,
            plane: Some(plane),
        })
    }

    fn create_container(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<ImageContainer, InternalErrors> {
        let img = Image::<f32, 1, CpuAllocator>::new(
            size,
            vec![0f32; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        Ok(ImageContainer::F32Gray(ManagedImage {
            data: img,
            tile_offset,
            plane: Some(plane),
        }))
    }

    fn matches_container(container: &ImageContainer) -> bool {
        matches!(container, ImageContainer::F32Gray(_))
    }

    fn wrap(image: Self::ImageRef) -> ImageContainer {
        ImageContainer::F32Gray(image)
    }

    fn get_ref(container: &ImageContainer) -> Option<&Self::ImageRef> {
        if let ImageContainer::F32Gray(img) = container {
            Some(img)
        } else {
            None
        }
    }

    fn get_ref_mut(container: &mut ImageContainer) -> Option<&mut Self::ImageRef> {
        if let ImageContainer::F32Gray(img) = container {
            Some(img)
        } else {
            None
        }
    }
}

impl ImageTypeMarker for F32Rgb {
    type ImageRef = ManagedImage<f32, 3>;

    fn create_image(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<Self::ImageRef, InternalErrors> {
        Ok(ManagedImage {
            data: Image::<f32, 3, CpuAllocator>::new(
                size,
                vec![0f32; size.width * size.height],
                CpuAllocator,
            )
            .map_err(|e| InternalErrors::Generic(e.to_string()))?,
            tile_offset,
            plane: Some(plane),
        })
    }

    fn create_container(
        size: kornia_image::ImageSize,
        tile_offset: Point2d,
        plane: ImagePlane,
    ) -> Result<ImageContainer, InternalErrors> {
        let img = Image::<f32, 3, CpuAllocator>::new(
            size,
            vec![0f32; size.width * size.height],
            CpuAllocator,
        )
        .map_err(InternalErrors::from_kornia)?;
        Ok(ImageContainer::F32Rgb(ManagedImage {
            data: img,
            tile_offset,
            plane: Some(plane),
        }))
    }

    fn matches_container(container: &ImageContainer) -> bool {
        matches!(container, ImageContainer::F32Rgb(_))
    }

    fn wrap(image: Self::ImageRef) -> ImageContainer {
        ImageContainer::F32Rgb(image)
    }

    fn get_ref(container: &ImageContainer) -> Option<&Self::ImageRef> {
        if let ImageContainer::F32Rgb(img) = container {
            Some(img)
        } else {
            None
        }
    }

    fn get_ref_mut(container: &mut ImageContainer) -> Option<&mut Self::ImageRef> {
        if let ImageContainer::F32Rgb(img) = container {
            Some(img)
        } else {
            None
        }
    }
}
