//! The image data the viewport keeps per channel: at the smallest sample type
//! that holds it exactly.
//!
//! The reader hands every image over as normalized `f32` (`raw / (2^bits - 1)`),
//! so an 8-bit slide takes 4 bytes per sample in RAM instead of 1, a 16-bit
//! image 2x its size. The viewport only displays and samples this data, so
//! [`ViewChannel::from_image_channel`] turns it back into `u8`/`u16` - but only
//! when every sample converts back to exactly the same `f32` the reader made,
//! so rendering, histograms and pixel values stay bit-identical. Anything else
//! (other bit depths, float data, interleaved RGB, label maps) stays as read.

use evanalyzer_app::images::{ImageChannel, ImageContainer, ManagedImage, Point2d};
use evanalyzer_cfg::core_types::InternalErrors;
use kornia_image::{Image, ImageSize};
use rayon::prelude::*;
use std::sync::Arc;

/// A channel's samples as kept by the viewport.
#[derive(Clone)]
pub(crate) enum Samples {
    /// Value = sample x `inv` (`inv` = 1 / (2^bits - 1), bits <= 8).
    U8 { data: Arc<[u8]>, inv: f32 },
    /// Value = sample x `inv` (`inv` = 1 / (2^bits - 1), bits <= 16).
    U16 { data: Arc<[u16]>, inv: f32 },
    /// Anything else, as read.
    Image(Arc<ImageContainer>),
}

/// A borrowed view of [`Samples`] for the render loop.
#[derive(Clone, Copy)]
pub(crate) enum SampleSlice<'a> {
    U8(&'a [u8], f32),
    U16(&'a [u16], f32),
    F32(&'a [f32]),
}

impl SampleSlice<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(d, _) => d.len(),
            Self::U16(d, _) => d.len(),
            Self::F32(d) => d.len(),
        }
    }

    /// The normalized value of sample `i` - identical to the reader's `f32`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Self::U8(d, inv) => d[i] as f32 * inv,
            Self::U16(d, inv) => d[i] as f32 * inv,
            Self::F32(d) => d[i],
        }
    }
}

/// One channel of a viewport tile - an [`ImageChannel`] with compact samples.
#[derive(Clone)]
pub(crate) struct ViewChannel {
    pub samples: Samples,
    pub width: usize,
    pub height: usize,
    pub tile_offset: Point2d,
    pub color: [f32; 3],
    pub c_stack: i32,
    pub name: String,
    pub is_rgb: bool,
}

impl ViewChannel {
    /// `channel` with its samples as compact as they exactly fit, given the
    /// image's `bits` per sample.
    pub fn from_image_channel(channel: &ImageChannel, bits: u16) -> Self {
        let ImageSize { width, height } = channel.image.size();
        let samples = match &*channel.image {
            ImageContainer::F32Gray(img) => compact(img.as_slice(), bits),
            _ => None,
        }
        .unwrap_or_else(|| Samples::Image(channel.image.clone()));
        Self {
            samples,
            width,
            height,
            tile_offset: channel.image.tile_offset(),
            color: channel.color,
            c_stack: channel.c_stack,
            name: channel.name.clone(),
            is_rgb: channel.is_rgb,
        }
    }

    /// `channel` as is, without copying (e.g. an in-memory breakpoint image).
    pub fn wrap(channel: &ImageChannel) -> Self {
        let ImageSize { width, height } = channel.image.size();
        Self {
            samples: Samples::Image(channel.image.clone()),
            width,
            height,
            tile_offset: channel.image.tile_offset(),
            color: channel.color,
            c_stack: channel.c_stack,
            name: channel.name.clone(),
            is_rgb: channel.is_rgb,
        }
    }

    /// The samples for rendering, `None` for data that isn't one value per
    /// pixel (interleaved RGB, label maps).
    pub fn slice(&self) -> Option<SampleSlice<'_>> {
        match &self.samples {
            Samples::U8 { data, inv } => Some(SampleSlice::U8(data, *inv)),
            Samples::U16 { data, inv } => Some(SampleSlice::U16(data, *inv)),
            Samples::Image(img) => match &**img {
                ImageContainer::F32Gray(img) => Some(SampleSlice::F32(img.as_slice())),
                _ => None,
            },
        }
    }

    /// The normalized value at pixel index `i` of a one-value-per-pixel
    /// channel.
    pub fn value_at(&self, i: usize) -> Option<f32> {
        let slice = self.slice()?;
        (i < slice.len()).then(|| slice.get(i))
    }

    /// The image as read (`f32`), expanded from compact samples if needed -
    /// for the few users that need an [`ImageContainer`].
    pub fn to_container(&self) -> Result<Arc<ImageContainer>, InternalErrors> {
        let values: Vec<f32> = match &self.samples {
            Samples::Image(img) => return Ok(img.clone()),
            Samples::U8 { data, inv } => data.par_iter().map(|&q| q as f32 * inv).collect(),
            Samples::U16 { data, inv } => data.par_iter().map(|&q| q as f32 * inv).collect(),
        };
        let size = ImageSize {
            width: self.width,
            height: self.height,
        };
        let image = Image::<f32, 1>::new(size, values).map_err(InternalErrors::from_kornia)?;
        Ok(Arc::new(ImageContainer::F32Gray(ManagedImage {
            data: image,
            tile_offset: self.tile_offset,
            plane: None,
        })))
    }

    /// Bytes the samples take.
    pub fn size_in_bytes(&self) -> usize {
        match &self.samples {
            Samples::U8 { data, .. } => data.len(),
            Samples::U16 { data, .. } => data.len() * 2,
            Samples::Image(img) => img.get_image_memory_usage(),
        }
    }
}

/// `values` (normalized by the reader as `raw * (1 / (2^bits - 1))`) as
/// `u8`/`u16` - `None` unless every value converts back exactly.
fn compact(values: &[f32], bits: u16) -> Option<Samples> {
    if !(1..=16).contains(&bits) {
        return None;
    }
    let max = ((1u32 << bits) - 1) as f32;
    let inv = 1.0 / max;
    let quantize = |v: f32| {
        let q = (v * max).round();
        // Exactly the reader's formula, so the round trip is bit-identical.
        ((0.0..=max).contains(&q) && (q as u32 as f32 * inv) == v).then_some(q as u32)
    };
    if bits <= 8 {
        let data: Option<Vec<u8>> = values
            .par_iter()
            .map(|&v| quantize(v).map(|q| q as u8))
            .collect();
        data.map(|d| Samples::U8 {
            data: d.into(),
            inv,
        })
    } else {
        let data: Option<Vec<u16>> = values
            .par_iter()
            .map(|&v| quantize(v).map(|q| q as u16))
            .collect();
        data.map(|d| Samples::U16 {
            data: d.into(),
            inv,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ImagePlane;

    /// What the reader makes of `raw` samples of `bits` bits.
    fn read_as_f32(raw: &[u32], bits: u16) -> Vec<f32> {
        let inv = 1.0 / ((1u64 << bits) - 1) as f32;
        raw.iter().map(|&r| r as f32 * inv).collect()
    }

    fn gray_channel(values: Vec<f32>, w: usize, h: usize) -> ImageChannel {
        ImageChannel {
            image: Arc::new(ImageContainer::F32Gray(ManagedImage {
                data: Image::<f32, 1>::new(
                    ImageSize {
                        width: w,
                        height: h,
                    },
                    values,
                )
                .unwrap(),
                tile_offset: Point2d { x: 3, y: 4 },
                plane: Some(ImagePlane { z: 0, c: 0, t: 0 }),
            })),
            color: [1.0, 0.5, 0.0],
            is_visible: true,
            c_stack: 2,
            name: "ch".into(),
            is_rgb: false,
        }
    }

    fn values(channel: &ViewChannel) -> Vec<f32> {
        let s = channel.slice().unwrap();
        (0..s.len()).map(|i| s.get(i)).collect()
    }

    #[test]
    fn eight_bit_data_is_kept_as_bytes_and_reads_back_bit_identical() {
        let raw: Vec<u32> = (0..256).collect();
        let read = read_as_f32(&raw, 8);
        let channel = ViewChannel::from_image_channel(&gray_channel(read.clone(), 16, 16), 8);

        assert!(matches!(channel.samples, Samples::U8 { .. }));
        assert_eq!(channel.size_in_bytes(), 256, "a quarter of the f32 size");
        let back = values(&channel);
        assert!(
            back.iter()
                .zip(&read)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!((channel.c_stack, channel.tile_offset.x), (2, 3));
    }

    #[test]
    fn twelve_and_sixteen_bit_data_is_kept_as_u16_bit_identical() {
        for bits in [12u16, 16] {
            let max = (1u32 << bits) - 1;
            let raw: Vec<u32> = (0..1000)
                .map(|i| (i * 7919) % (max + 1))
                .chain([0, max])
                .collect();
            let read = read_as_f32(&raw, bits);
            let n = read.len();
            let channel = ViewChannel::from_image_channel(&gray_channel(read.clone(), n, 1), bits);

            assert!(
                matches!(channel.samples, Samples::U16 { .. }),
                "{bits} bits"
            );
            let back = values(&channel);
            assert!(
                back.iter()
                    .zip(&read)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn data_that_does_not_round_trip_exactly_stays_as_read() {
        // A float image, or anything processed after reading.
        let channel =
            ViewChannel::from_image_channel(&gray_channel(vec![0.1, 0.123456, 0.5], 3, 1), 8);
        assert!(matches!(channel.samples, Samples::Image(_)));
        assert_eq!(values(&channel), vec![0.1, 0.123456, 0.5]);

        // 32-bit data never fits.
        let channel = ViewChannel::from_image_channel(&gray_channel(vec![0.0, 1.0], 2, 1), 32);
        assert!(matches!(channel.samples, Samples::Image(_)));
    }

    #[test]
    fn expanding_gives_back_the_image_as_read() {
        let read = read_as_f32(&[0, 17, 255, 128], 8);
        let channel = ViewChannel::from_image_channel(&gray_channel(read.clone(), 2, 2), 8);
        let image = channel.to_container().unwrap();
        let ImageContainer::F32Gray(img) = &*image else {
            panic!("gray image expected");
        };
        assert_eq!(img.as_slice(), read.as_slice());
        assert_eq!((image.size().width, image.size().height), (2, 2));
        assert_eq!(channel.value_at(2), Some(1.0));
        assert_eq!(channel.value_at(4), None);
    }
}
