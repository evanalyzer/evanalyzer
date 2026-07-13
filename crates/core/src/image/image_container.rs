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
                vec![0f32; size.width * size.height * 3],
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
            vec![0f32; size.width * size.height * 3],
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_size() -> kornia_image::ImageSize {
        kornia_image::ImageSize {
            width: 4,
            height: 3,
        }
    }

    fn test_plane() -> ImagePlane {
        ImagePlane { z: 1, c: 2, t: 3 }
    }

    fn u32_container(size: kornia_image::ImageSize) -> ImageContainer {
        ImageContainer::U32(ManagedImage {
            data: Image::<u32, 1, CpuAllocator>::new(
                size,
                vec![0u32; size.width * size.height],
                CpuAllocator,
            )
            .unwrap(),
            tile_offset: Point2d { x: 0, y: 0 },
            plane: None,
        })
    }

    #[test]
    fn f32_gray_create_image_produces_zeroed_data_with_correct_size() {
        let size = test_size();
        let offset = Point2d { x: 7, y: 9 };
        let image = F32Gray::create_image(size, offset, test_plane()).unwrap();

        assert_eq!(image.data.size().width, size.width);
        assert_eq!(image.data.size().height, size.height);
        assert_eq!(image.tile_offset, offset);
        assert_eq!(image.plane, Some(test_plane()));
        assert!(image.data.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f32_gray_create_container_wraps_in_f32_gray_variant() {
        let size = test_size();
        let container =
            F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();

        assert!(matches!(container, ImageContainer::F32Gray(_)));
        assert_eq!(container.size().width, size.width);
        assert_eq!(container.size().height, size.height);
        assert_eq!(container.nr_color_channels(), 1);
    }

    #[test]
    fn f32_gray_matches_container_is_true_only_for_f32_gray_variant() {
        let size = test_size();
        let gray = F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let rgb = F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let u32c = u32_container(size);

        assert!(F32Gray::matches_container(&gray));
        assert!(!F32Gray::matches_container(&rgb));
        assert!(!F32Gray::matches_container(&u32c));
    }

    #[test]
    fn f32_gray_wrap_produces_f32_gray_variant_preserving_data() {
        let size = test_size();
        let offset = Point2d { x: 2, y: 3 };
        let image = F32Gray::create_image(size, offset, test_plane()).unwrap();

        let wrapped = F32Gray::wrap(image);

        match wrapped {
            ImageContainer::F32Gray(img) => {
                assert_eq!(img.tile_offset, offset);
                assert_eq!(img.plane, Some(test_plane()));
            }
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    #[test]
    fn f32_gray_get_ref_returns_some_for_matching_variant_and_none_otherwise() {
        let size = test_size();
        let gray = F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let rgb = F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let u32c = u32_container(size);

        assert!(F32Gray::get_ref(&gray).is_some());
        assert!(F32Gray::get_ref(&rgb).is_none());
        assert!(F32Gray::get_ref(&u32c).is_none());
    }

    #[test]
    fn f32_gray_get_ref_mut_allows_mutation_of_matching_variant() {
        let size = test_size();
        let mut gray =
            F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();

        {
            let img_mut = F32Gray::get_ref_mut(&mut gray).expect("should be F32Gray");
            img_mut.data.as_slice_mut()[0] = 42.0;
        }

        match &gray {
            ImageContainer::F32Gray(img) => assert_eq!(img.as_slice()[0], 42.0),
            other => panic!("expected F32Gray, got {other:?}"),
        }
    }

    #[test]
    fn f32_gray_get_ref_mut_returns_none_for_non_matching_variant() {
        let size = test_size();
        let mut rgb = F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let mut u32c = u32_container(size);

        assert!(F32Gray::get_ref_mut(&mut rgb).is_none());
        assert!(F32Gray::get_ref_mut(&mut u32c).is_none());
    }

    #[test]
    fn f32_rgb_create_image_produces_zeroed_data_with_correct_size() {
        let size = test_size();
        let offset = Point2d { x: 5, y: 6 };
        let image = F32Rgb::create_image(size, offset, test_plane()).unwrap();

        assert_eq!(image.data.size().width, size.width);
        assert_eq!(image.data.size().height, size.height);
        assert_eq!(image.tile_offset, offset);
        assert_eq!(image.plane, Some(test_plane()));
        assert!(image.data.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f32_rgb_create_container_wraps_in_f32_rgb_variant() {
        let size = test_size();
        let container =
            F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();

        assert!(matches!(container, ImageContainer::F32Rgb(_)));
        assert_eq!(container.size().width, size.width);
        assert_eq!(container.size().height, size.height);
        assert_eq!(container.nr_color_channels(), 3);
    }

    #[test]
    fn f32_rgb_matches_container_is_true_only_for_f32_rgb_variant() {
        let size = test_size();
        let gray = F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let rgb = F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let u32c = u32_container(size);

        assert!(F32Rgb::matches_container(&rgb));
        assert!(!F32Rgb::matches_container(&gray));
        assert!(!F32Rgb::matches_container(&u32c));
    }

    #[test]
    fn f32_rgb_wrap_produces_f32_rgb_variant_preserving_data() {
        let size = test_size();
        let offset = Point2d { x: 8, y: 1 };
        let image = F32Rgb::create_image(size, offset, test_plane()).unwrap();

        let wrapped = F32Rgb::wrap(image);

        match wrapped {
            ImageContainer::F32Rgb(img) => {
                assert_eq!(img.tile_offset, offset);
                assert_eq!(img.plane, Some(test_plane()));
            }
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn f32_rgb_get_ref_returns_some_for_matching_variant_and_none_otherwise() {
        let size = test_size();
        let gray = F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let rgb = F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let u32c = u32_container(size);

        assert!(F32Rgb::get_ref(&rgb).is_some());
        assert!(F32Rgb::get_ref(&gray).is_none());
        assert!(F32Rgb::get_ref(&u32c).is_none());
    }

    #[test]
    fn f32_rgb_get_ref_mut_allows_mutation_of_matching_variant() {
        let size = test_size();
        let mut rgb =
            F32Rgb::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();

        {
            let img_mut = F32Rgb::get_ref_mut(&mut rgb).expect("should be F32Rgb");
            img_mut.data.as_slice_mut()[0] = 99.0;
        }

        match &rgb {
            ImageContainer::F32Rgb(img) => assert_eq!(img.as_slice()[0], 99.0),
            other => panic!("expected F32Rgb, got {other:?}"),
        }
    }

    #[test]
    fn f32_rgb_get_ref_mut_returns_none_for_non_matching_variant() {
        let size = test_size();
        let mut gray =
            F32Gray::create_container(size, Point2d { x: 0, y: 0 }, test_plane()).unwrap();
        let mut u32c = u32_container(size);

        assert!(F32Rgb::get_ref_mut(&mut gray).is_none());
        assert!(F32Rgb::get_ref_mut(&mut u32c).is_none());
    }
}
