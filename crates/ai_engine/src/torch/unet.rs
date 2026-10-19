//! # unet
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! U-Net inference: the tile padded to the model's needs, the prediction
//! cropped back and reduced to the foreground (and boundary) probability.

use super::{Result, scheduler::TorchScheduler};
use crate::api::{Image, Response};
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Side lengths the model is run at must be a multiple of this: a standard
/// U-Net halves the image 4 times (2^4), and odd intermediate sizes make its
/// skip connections mismatch.
const SIZE_MULTIPLE: i64 = 16;

/// Mirrored context added around the tile, at least the halo bioimage.io
/// U-Nets declare (e.g. 16 px for romantic-shark, 8 px for joyful-pig).
const MARGIN: i64 = 16;

pub(crate) fn run(
    scheduler: &TorchScheduler,
    model_path: &Path,
    foreground_channel: i32,
    softmax: bool,
    boundary_channel: i32,
    image: Image<'_>,
) -> Result<Response> {
    let (width, height) = (image.width, image.height);
    let pixels = &image.data[..width * height];
    scheduler.run(
        model_path,
        |e| {
            format!(
                "Failed to load U-Net model from {}: {e}",
                model_path.display()
            )
        },
        |model, device| {
            let input = Tensor::from_slice(pixels)
                .to_device(device)
                .to_kind(Kind::Float)
                .reshape([1, 1, height as i64, width as i64]);
            let (top, bottom) = padding(height as i64);
            let (left, right) = padding(width as i64);
            // Mirroring needs every pad to be smaller than the side it mirrors;
            // a tiny tile repeats its edge pixels instead.
            let mode = if top < height as i64
                && bottom < height as i64
                && left < width as i64
                && right < width as i64
            {
                "reflect"
            } else {
                "replicate"
            };
            let padded = input.pad([left, right, top, bottom], mode, None);

            // Without `no_grad`, the model's own weights (loaded with
            // `requires_grad = true`, the default for `nn.Parameter`) make this
            // forward pass record a full autograd graph even though `.eval()`
            // was called - wasting memory that's never needed for inference.
            let output = tch::no_grad(|| model.forward_ts(&[padded]))
                .map_err(|e| format!("U-Net inference failed: {e}"))?;
            let out_size = output.size();
            if out_size.len() != 4
                || out_size[2] < top + height as i64
                || out_size[3] < left + width as i64
            {
                return Err(format!(
                    "U-Net output has shape {out_size:?}; expected [1, C, H, W] the size of the \
                     (padded) input"
                ));
            }
            let output = output
                .narrow(2, top, height as i64)
                .narrow(3, left, width as i64);

            let channels = *output.size().get(1).unwrap_or(&1);
            let foreground = if channels > 1 {
                let idx = (foreground_channel as i64).clamp(0, channels - 1);
                if softmax {
                    output.softmax(1, Kind::Float).narrow(1, idx, 1)
                } else {
                    output.narrow(1, idx, 1)
                }
            } else {
                output.shallow_clone()
            };
            let probabilities = channel_to_vec(&foreground, width, height)?;

            // Optional boundary channel: boundary-aware models (e.g.
            // affable-shark) predict an independent boundary map, which opens
            // thin gaps between touching objects so they separate downstream.
            let boundary =
                if boundary_channel >= 0 && channels > 1 && (boundary_channel as i64) < channels {
                    let b = output.narrow(1, boundary_channel as i64, 1);
                    Some(channel_to_vec(&b, width, height)?)
                } else {
                    None
                };
            Ok(Response::UNet {
                probabilities,
                boundary,
            })
        },
    )
}

/// Padding `(before, after)` for one side of `len` pixels: `MARGIN` before,
/// and after whatever brings the total up to a multiple of `SIZE_MULTIPLE`
/// with at least `MARGIN` of context.
fn padding(len: i64) -> (i64, i64) {
    let needed = len + 2 * MARGIN;
    let total = (needed + SIZE_MULTIPLE - 1) / SIZE_MULTIPLE * SIZE_MULTIPLE;
    (MARGIN, total - len - MARGIN)
}

/// Moves a single-channel `[1, 1, H, W]` tensor to the CPU and flattens it
/// into a `width * height` vector.
fn channel_to_vec(tensor: &Tensor, width: usize, height: usize) -> Result<Vec<f32>> {
    let err = |e: tch::TchError| format!("U-Net inference failed: {e}");
    let out = tensor
        .f_to_device(Device::Cpu)
        .and_then(|out| out.f_reshape([(width * height) as i64]))
        .map_err(err)?;
    Vec::try_from(&out).map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_reaches_a_multiple_of_16_with_at_least_16_px_of_context() {
        for len in [1, 15, 16, 31, 250, 256, 1040, 1388] {
            let (before, after) = padding(len);
            assert_eq!(before, 16, "len {len}");
            assert!(after >= 16, "len {len}: only {after} px after");
            assert_eq!((before + len + after) % 16, 0, "len {len}");
            assert!(
                after < 32,
                "len {len}: {after} px is more padding than needed"
            );
        }
        assert_eq!(padding(250), (16, 22));
        assert_eq!(padding(256), (16, 16));
    }
}
