//! Wire format for pixels: an `ImageContainer` split into a small
//! serializable description and its samples as raw little-endian bytes, so a
//! transport sends one binary blob instead of millions of encoded numbers.

use crate::api::{ImageContainer, ManagedImage, Point2d};
use evanalyzer_cfg::core_types::{ImagePlane, InternalErrors};
use kornia_image::{Image, ImageSize};
use serde::{Deserialize, Serialize};

/// Pixel layout of an [`ImageContainer`] variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelKind {
    Gray32F,
    Rgb32F,
    Label32U,
}

impl PixelKind {
    fn channels(self) -> usize {
        match self {
            PixelKind::Gray32F | PixelKind::Label32U => 1,
            PixelKind::Rgb32F => 3,
        }
    }
}

/// Everything about an [`ImageContainer`] except its pixel values, which
/// travel separately as raw little-endian bytes (see [`image_to_raw`]) - a
/// transport can then send them as one binary blob instead of encoding
/// millions of numbers individually.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawImageInfo {
    pub kind: PixelKind,
    pub width: usize,
    pub height: usize,
    pub tile_offset_x: usize,
    pub tile_offset_y: usize,
    pub plane: Option<ImagePlane>,
}

/// Splits `image` into its description and its pixels as little-endian
/// bytes (4 bytes per sample, channel-interleaved like the image itself).
pub fn image_to_raw(image: &ImageContainer) -> (RawImageInfo, Vec<u8>) {
    let (kind, width, height, bytes) = match image {
        ImageContainer::F32Gray(img) => (
            PixelKind::Gray32F,
            img.width(),
            img.height(),
            samples_to_le_bytes(img.as_slice(), f32::to_le_bytes),
        ),
        ImageContainer::F32Rgb(img) => (
            PixelKind::Rgb32F,
            img.width(),
            img.height(),
            samples_to_le_bytes(img.as_slice(), f32::to_le_bytes),
        ),
        ImageContainer::U32(img) => (
            PixelKind::Label32U,
            img.width(),
            img.height(),
            samples_to_le_bytes(img.as_slice(), u32::to_le_bytes),
        ),
    };
    let offset = image.tile_offset();
    let info = RawImageInfo {
        kind,
        width,
        height,
        tile_offset_x: offset.x,
        tile_offset_y: offset.y,
        plane: image.plane(),
    };
    (info, bytes)
}

/// Inverse of [`image_to_raw`]. Fails instead of panicking when `bytes`
/// doesn't match `info`'s dimensions - it may come from a network peer.
pub fn image_from_raw(info: &RawImageInfo, bytes: &[u8]) -> Result<ImageContainer, InternalErrors> {
    let expected = info
        .width
        .checked_mul(info.height)
        .and_then(|n| n.checked_mul(info.kind.channels()))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| InternalErrors::InvalidArgument("image dimensions overflow".into()))?;
    if bytes.len() != expected {
        return Err(InternalErrors::InvalidArgument(format!(
            "image data is {} bytes, expected {expected} for {}x{} {:?}",
            bytes.len(),
            info.width,
            info.height,
            info.kind
        )));
    }
    let size = ImageSize {
        width: info.width,
        height: info.height,
    };
    let tile_offset = Point2d {
        x: info.tile_offset_x,
        y: info.tile_offset_y,
    };
    let f32s = || -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let container = match info.kind {
        PixelKind::Gray32F => ImageContainer::F32Gray(ManagedImage {
            data: Image::new(size, f32s()).map_err(InternalErrors::from_kornia)?,
            tile_offset,
            plane: info.plane,
        }),
        PixelKind::Rgb32F => ImageContainer::F32Rgb(ManagedImage {
            data: Image::new(size, f32s()).map_err(InternalErrors::from_kornia)?,
            tile_offset,
            plane: info.plane,
        }),
        PixelKind::Label32U => {
            let u32s = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            ImageContainer::U32(ManagedImage {
                data: Image::new(size, u32s).map_err(InternalErrors::from_kornia)?,
                tile_offset,
                plane: info.plane,
            })
        }
    };
    Ok(container)
}

fn samples_to_le_bytes<T: Copy, const N: usize>(samples: &[T], to_le: fn(T) -> [u8; N]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * N);
    for &sample in samples {
        bytes.extend_from_slice(&to_le(sample));
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(width: usize, height: usize, values: Vec<f32>) -> ImageContainer {
        ImageContainer::F32Gray(ManagedImage {
            data: Image::new(ImageSize { width, height }, values).unwrap(),
            tile_offset: Point2d { x: 512, y: 1024 },
            plane: Some(ImagePlane { z: 1, c: 2, t: 3 }),
        })
    }

    #[test]
    fn gray_image_round_trips_values_offset_and_plane() {
        let original = gray(3, 2, vec![0.0, 0.5, 1.0, -2.25, f32::MAX, 65535.0]);
        let (info, bytes) = image_to_raw(&original);
        assert_eq!(bytes.len(), 6 * 4);

        let restored = image_from_raw(&info, &bytes).unwrap();
        assert_eq!(restored.as_f32_slice(), original.as_f32_slice());
        assert_eq!(restored.tile_offset(), original.tile_offset());
        assert_eq!(restored.plane(), original.plane());
    }

    #[test]
    fn rgb_and_label_images_round_trip() {
        let rgb = ImageContainer::F32Rgb(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: 2,
                    height: 1,
                },
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            )
            .unwrap(),
            tile_offset: Point2d::default(),
            plane: None,
        });
        let (info, bytes) = image_to_raw(&rgb);
        assert_eq!(info.kind, PixelKind::Rgb32F);
        let restored = image_from_raw(&info, &bytes).unwrap();
        assert_eq!(restored.as_f32_slice(), rgb.as_f32_slice());

        let labels = ImageContainer::U32(ManagedImage {
            data: Image::new(
                ImageSize {
                    width: 2,
                    height: 2,
                },
                vec![0, 1, 7, u32::MAX],
            )
            .unwrap(),
            tile_offset: Point2d::default(),
            plane: None,
        });
        let (info, bytes) = image_to_raw(&labels);
        let ImageContainer::U32(restored) = image_from_raw(&info, &bytes).unwrap() else {
            panic!("expected a label image");
        };
        assert_eq!(restored.as_slice(), &[0, 1, 7, u32::MAX]);
    }

    #[test]
    fn mismatched_byte_count_is_an_error_not_a_panic() {
        let (info, bytes) = image_to_raw(&gray(2, 2, vec![1.0; 4]));
        assert!(image_from_raw(&info, &bytes[..bytes.len() - 1]).is_err());
        let huge = RawImageInfo {
            width: usize::MAX,
            height: 2,
            ..info
        };
        assert!(image_from_raw(&huge, &bytes).is_err());
    }
}
