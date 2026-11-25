//! # colorize
//!
//! An image's channels as an RGB picture, as the viewer shows them (each
//! channel in its colour, within its display range) - for the drawn images
//! and the Results' thumbnails.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::results::{ChannelDisplay, DrawBrightness, ImageDisplay};
use evanalyzer_core::{ImageChannel, ImageContainer};
use std::collections::BTreeMap;

/// The channels of a tile as one picture: each in its color (white
/// without one), auto-scaled as the viewer does (see [`window`]). RGB
/// images are taken as they are. Also the split view's thumbnails.
pub fn colorize(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    width: usize,
    height: usize,
) -> image::RgbImage {
    colorize_with(
        channels,
        colors,
        &ImageDisplay::default(),
        DrawBrightness::Auto,
        width,
        height,
    )
}

/// [`colorize`] as `display` shows the image: its hidden channels left
/// out, and with [`DrawBrightness::Viewer`] each channel through its
/// display range (auto-scaled without one).
pub(crate) fn colorize_with(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    display: &ImageDisplay,
    brightness: DrawBrightness,
    width: usize,
    height: usize,
) -> image::RgbImage {
    let mut rgb = vec![0f32; width * height * 3];
    for channel in channels {
        let shown = display
            .channels
            .get(&channel.c_stack)
            .copied()
            .unwrap_or_default();
        if !shown.visible {
            continue;
        }
        let color = colors
            .get(&channel.c_stack)
            .copied()
            .unwrap_or([1.0, 1.0, 1.0]);
        match &*channel.image {
            ImageContainer::F32Gray(image) => {
                let data = image.data.as_slice();
                let (lo, hi) = match (brightness, shown) {
                    (
                        DrawBrightness::Viewer,
                        ChannelDisplay {
                            range: Some(range), ..
                        },
                    ) => range,
                    _ => window(data),
                };
                let range = (hi - lo).max(1e-6);
                for (i, v) in data.iter().enumerate().take(width * height) {
                    let v = ((v - lo) / range).clamp(0.0, 1.0);
                    for k in 0..3 {
                        rgb[i * 3 + k] += v * color[k];
                    }
                }
            }
            ImageContainer::F32Rgb(image) => {
                for (out, v) in rgb.iter_mut().zip(image.data.as_slice()) {
                    *out += v;
                }
            }
            ImageContainer::U32(_) => {}
        }
    }
    image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let i = (y as usize * width + x as usize) * 3;
        image::Rgb([
            (rgb[i].min(1.0) * 255.0) as u8,
            (rgb[i + 1].min(1.0) * 255.0) as u8,
            (rgb[i + 2].min(1.0) * 255.0) as u8,
        ])
    })
}

/// [`colorize`]d, then scaled down to fill `(width, height)` (cropped to
/// its shape) - the split view's filmstrip.
pub fn colorize_thumbnail(
    channels: &[ImageChannel],
    colors: &BTreeMap<i32, [f32; 3]>,
    (width, height): (usize, usize),
    (thumb_w, thumb_h): (u32, u32),
) -> image::RgbImage {
    image::DynamicImage::ImageRgb8(colorize(channels, colors, width, height))
        .resize_to_fill(thumb_w, thumb_h, image::imageops::FilterType::Triangle)
        .into_rgb8()
}

/// The display window of a channel: its 0.5 % and 99.5 % values (of every
/// 10th pixel), as the viewer's auto-scale.
fn window(data: &[f32]) -> (f32, f32) {
    let mut sample: Vec<f32> = data.iter().step_by(10).copied().collect();
    if sample.is_empty() {
        return (0.0, 1.0);
    }
    let len = sample.len();
    let low = (len as f32 * 0.005) as usize;
    let high = ((len as f32 * 0.995) as usize).min(len - 1);
    sample.select_nth_unstable_by(low, |a, b| a.total_cmp(b));
    let lo = sample[low];
    sample.select_nth_unstable_by(high, |a, b| a.total_cmp(b));
    (lo, sample[high])
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::core_types::ZProjection;

    #[test]
    fn the_window_skips_the_extremes() {
        // 1000 sampled values: one outlier each way is below 0.5 %.
        let mut data = vec![0.5f32; 10_000];
        data[0] = 0.0;
        data[10] = 1.0;
        assert_eq!(window(&data), (0.5, 0.5));
        assert_eq!(window(&[]), (0.0, 1.0));
    }

    #[test]
    fn a_hidden_channel_is_left_out_and_the_viewer_range_used() {
        let channel = |c, v: f32| ImageChannel {
            image: std::sync::Arc::new(ImageContainer::F32Gray(crate::images::ManagedImage {
                data: kornia_image::Image::new(
                    kornia_image::ImageSize {
                        width: 1,
                        height: 1,
                    },
                    vec![v],
                )
                .unwrap(),
                tile_offset: crate::images::Point2d { x: 0, y: 0 },
                plane: None,
                dimensionality: evanalyzer_cfg::core_types::ImageDimensionality::Planar,
            })),
            color: [1.0, 1.0, 1.0],
            c_stack: c,
            name: String::new(),
            is_rgb: false,
            is_visible: true,
        };
        let channels = [channel(0, 0.25), channel(1, 1.0)];
        let colors = [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])].into();
        let display = ImageDisplay {
            z_projection: ZProjection::None,
            channels: [
                (
                    0,
                    ChannelDisplay {
                        visible: true,
                        range: Some((0.0, 0.5)),
                    },
                ),
                (
                    1,
                    ChannelDisplay {
                        visible: false,
                        range: None,
                    },
                ),
            ]
            .into(),
        };
        let picture = colorize_with(&channels, &colors, &display, DrawBrightness::Viewer, 1, 1);
        assert_eq!(*picture.get_pixel(0, 0), image::Rgb([127, 0, 0]));
    }
}
