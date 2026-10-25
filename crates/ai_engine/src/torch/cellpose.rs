//! # cellpose
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose(-SAM) inference: the image (resized to the size the model runs
//! at, its channels filled up) run in overlapping 256 x 256 tiles, blended
//! back into flows and the cell probability.

use super::{Result, scheduler::TorchScheduler};
use crate::api::{Image, Response};
use std::path::Path;
use tch::{CModule, Device, IValue, Kind, Tensor};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    scheduler: &TorchScheduler,
    model_path: &Path,
    net_width: usize,
    net_height: usize,
    input_channels: i32,
    replicate_gray_channel: bool,
    image: Image<'_>,
) -> Result<Response> {
    let (width, height) = (image.width, image.height);
    let pixels = &image.data[..width * height];
    let resized = (net_width, net_height) != (width, height);
    let (w, h) = (net_width, net_height);
    scheduler.run(
        model_path,
        |e| {
            format!(
                "Failed to load Cellpose model from {}: {e}. The model must be a \
                 TorchScript export (torch.jit.script/trace), not a raw weights file. \
                 A bioimage.io `pytorch_state_dict` (.pth) holds only weights and no \
                 graph — load it into the Cellpose architecture in Python and re-save \
                 it with torch.jit before using it here.",
                model_path.display()
            )
        },
        |model, device| {
            let image = Tensor::from_slice(pixels)
                .to_device(device)
                .to_kind(Kind::Float)
                .reshape([1, 1, height as i64, width as i64]);

            // The image is the first channel; standard Cellpose models expect a
            // second (nucleus) channel, and custom models may want more. Zero-fill
            // (or replicate the image into) any extra channels so the tensor
            // matches the model's input width.
            let in_channels = input_channels.max(1) as i64;
            let input = if in_channels <= 1 {
                image
            } else if replicate_gray_channel {
                image.repeat([1, in_channels, 1, 1])
            } else {
                let extra = Tensor::zeros(
                    [1, in_channels - 1, height as i64, width as i64],
                    (Kind::Float, device),
                );
                Tensor::cat(&[image, extra], 1)
            };
            let input = if resized {
                input.upsample_bilinear2d([net_height as i64, net_width as i64], false, None, None)
            } else {
                input
            };

            // Cellpose-SAM can only run on exactly 256x256 tiles -
            // `run_model_tiled` hides that behind the same `[1, C, H, W]`
            // contract `run_model` exposes directly.
            let output = run_model_tiled(model, &input, device)?;

            // `run_model_tiled` always returns a `[1, C, height, width]` tensor,
            // so the channel dimension is fixed at index 1.
            const CHANNEL_DIM: i64 = 1;
            Ok(Response::Cellpose {
                flow_y: channel_to_vec(&output.narrow(CHANNEL_DIM, 0, 1), w, h)?,
                flow_x: channel_to_vec(&output.narrow(CHANNEL_DIM, 1, 1), w, h)?,
                cell_prob: channel_to_vec(&output.narrow(CHANNEL_DIM, 2, 1).sigmoid(), w, h)?,
            })
        },
    )
}

/// Runs the model and returns the flow/probability tensor, supporting both a
/// bare tensor output and exports that wrap it in a tuple/list (e.g.
/// `(flows, style)`).
fn run_model(model: &CModule, input: Tensor) -> Result<Tensor> {
    let output = model
        .forward_is(&[IValue::Tensor(input)])
        .map_err(|e| format!("Cellpose inference failed: {e}"))?;

    let tensor = match output {
        IValue::Tensor(t) => Some(t),
        IValue::Tuple(items) | IValue::GenericList(items) => items
            .into_iter()
            .filter_map(|v| match v {
                IValue::Tensor(t) => Some(t),
                _ => None,
            })
            .find(|t| {
                let s = t.size();
                s.len() >= 3 && s[s.len() - 3] >= 3
            }),
        other => {
            return Err(format!(
                "Cellpose model returned an unsupported output type: {other:?}"
            ));
        }
    };

    tensor.ok_or_else(|| "Cellpose model returned no tensor with at least 3 channels".to_string())
}

/// Cellpose-SAM's ViT encoder bakes its positional embeddings for a fixed
/// token grid at export time (see the struct doc comment) - the exported
/// graph only accepts exactly `TILE_SIZE x TILE_SIZE` input.
const TILE_SIZE: i64 = 256;

/// Fraction of a tile that overlaps its neighbor, matching Cellpose's own
/// default (`tile_overlap=0.1` in `models.CellposeModel.eval`).
const TILE_OVERLAP: f32 = 0.1;

/// Runs `model` over `input` (`[1, C, height, width]`, any `height`/`width`)
/// by padding it up to at least `TILE_SIZE` per side, splitting it into
/// overlapping `TILE_SIZE x TILE_SIZE` tiles, running each tile through
/// `run_model`, and blending the results back into a single
/// `[1, C, height, width]` tensor with a feathered (sigmoid taper) weight
/// per tile - the same approach Cellpose's own `transforms.average_tiles`
/// uses, so a segmentation spanning a tile boundary doesn't show a seam.
///
/// A single tile that covers the whole (padded) image is the common case
/// for images no bigger than `TILE_SIZE`; the taper weight cancels out
/// exactly there (it's the only tile contributing to every pixel), so
/// small images are unaffected by the blending.
fn run_model_tiled(model: &CModule, input: &Tensor, device: Device) -> Result<Tensor> {
    // Without this, every tile's forward pass keeps its autograd graph
    // (all of the ViT encoder's intermediate activations) alive, and the
    // in-place `acc_region += ...` accumulation below chains each tile's
    // graph onto the last - so a large image's memory use grows with
    // *every* tile instead of being bounded by one tile's peak, which
    // reliably exhausts GPU memory on anything but a tiny image.
    tch::no_grad(|| run_model_tiled_inner(model, input, device))
}

fn run_model_tiled_inner(model: &CModule, input: &Tensor, device: Device) -> Result<Tensor> {
    let sizes = input.size();
    let (orig_h, orig_w) = (sizes[2], sizes[3]);

    let pad_h = (TILE_SIZE - orig_h).max(0);
    let pad_w = (TILE_SIZE - orig_w).max(0);
    // `Tensor::pad`'s list is ordered from the last dimension inward:
    // [left, right, top, bottom] pads W then H. Only the bottom/right
    // edges are padded - unlike Cellpose's own centered padding, this is
    // an implementation simplification, not a correctness requirement:
    // the padding is zero context for the network either way, and the
    // exact split of "extra" border pixels doesn't affect the result
    // (the output is cropped back to `orig_h`/`orig_w` afterwards).
    let padded = input.pad([0, pad_w, 0, pad_h], "constant", 0.0);
    let padded_h = orig_h + pad_h;
    let padded_w = orig_w + pad_w;

    let ys = tile_starts(padded_h);
    let xs = tile_starts(padded_w);
    let mask = taper_mask(device);

    let mut acc: Option<Tensor> = None;
    let norm = Tensor::zeros([1, 1, padded_h, padded_w], (Kind::Float, device));

    for &y in &ys {
        for &x in &xs {
            let tile = padded.narrow(2, y, TILE_SIZE).narrow(3, x, TILE_SIZE);
            let tile_out = run_model(model, tile)?;

            let tsizes = tile_out.size();
            if tsizes.len() < 4 {
                return Err("Cellpose model output has too few dimensions; expected \
                     `[1, C, 256, 256]`"
                    .to_string());
            }
            let tile_channels = tsizes[tsizes.len() - 3];
            if tile_channels < 3 {
                return Err("Cellpose model output has fewer than 3 channels; expected \
                     `[dY, dX, cellprob]`"
                    .to_string());
            }
            if tsizes[tsizes.len() - 2] != TILE_SIZE || tsizes[tsizes.len() - 1] != TILE_SIZE {
                return Err(format!(
                    "Cellpose-SAM model output tile is {}x{}, expected exactly \
                     256x256 - the exported model must be traced at a fixed \
                     256x256 input (see docs/convert_cellpose.py)",
                    tsizes[tsizes.len() - 1],
                    tsizes[tsizes.len() - 2],
                ));
            }

            let tile_out = tile_out.to_kind(Kind::Float).to_device(device);
            let acc = acc.get_or_insert_with(|| {
                Tensor::zeros(
                    [1, tile_channels, padded_h, padded_w],
                    (Kind::Float, device),
                )
            });

            let mut acc_region = acc.narrow(2, y, TILE_SIZE).narrow(3, x, TILE_SIZE);
            acc_region += &tile_out * &mask;
            let mut norm_region = norm.narrow(2, y, TILE_SIZE).narrow(3, x, TILE_SIZE);
            norm_region += &mask;
        }
    }

    let acc = acc.ok_or_else(|| "Cellpose produced no output tiles".to_string())?;
    let stitched = acc / &norm;
    Ok(stitched.narrow(2, 0, orig_h).narrow(3, 0, orig_w))
}

/// Start offsets (top or left) of the `TILE_SIZE`-wide tiles covering
/// `padded_len` pixels with `TILE_OVERLAP` fractional overlap, matching
/// Cellpose's own `transforms.make_tiles`. A single tile starting at `0`
/// covers the whole span whenever `padded_len <= TILE_SIZE`.
fn tile_starts(padded_len: i64) -> Vec<i64> {
    if padded_len <= TILE_SIZE {
        return vec![0];
    }
    let overlap = TILE_OVERLAP.clamp(0.05, 0.5);
    let n = (((1.0 + 2.0 * overlap) * padded_len as f32) / TILE_SIZE as f32).ceil() as i64;
    if n <= 1 {
        return vec![0];
    }
    let span = (padded_len - TILE_SIZE) as f32;
    (0..n)
        .map(|i| (span * i as f32 / (n - 1) as f32) as i64)
        .collect()
}

/// The `[1, 1, TILE_SIZE, TILE_SIZE]` feathered blend weight Cellpose's
/// `transforms._taper_mask` uses: a separable sigmoid taper that's ~1 near
/// the tile's center and decays (without ever reaching exactly `0`) toward
/// its edges, so overlapping tiles blend smoothly instead of showing a
/// seam at the boundary.
fn taper_mask(device: Device) -> Tensor {
    const SIG: f32 = 7.5;
    let size = TILE_SIZE as usize;
    let center = (TILE_SIZE as f32 - 1.0) / 2.0;
    let mask1d: Vec<f32> = (0..size)
        .map(|i| {
            let xm = (i as f32 - center).abs();
            1.0 / (1.0 + ((xm - (TILE_SIZE as f32 / 2.0 - 20.0)) / SIG).exp())
        })
        .collect();
    let mut mask2d = vec![0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            mask2d[y * size + x] = mask1d[y] * mask1d[x];
        }
    }
    Tensor::from_slice(&mask2d)
        .to_device(device)
        .reshape([1, 1, TILE_SIZE, TILE_SIZE])
}

/// Moves a single-channel `[1, 1, H, W]` tensor to the CPU and flattens it
/// into a `width * height` vector.
fn channel_to_vec(tensor: &Tensor, width: usize, height: usize) -> Result<Vec<f32>> {
    tensor
        .f_to_device(Device::Cpu)
        .and_then(|out| out.f_reshape([(width * height) as i64]))
        .map_err(|e| format!("Cellpose inference failed: {e}"))
        .and_then(|out| Vec::try_from(&out).map_err(|e| format!("Cellpose inference failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::torch::test_support::trace_and_save_model;

    /// A "model" with zero flows whose cell logit is the image channel scaled
    /// to saturate the sigmoid, traced at the fixed 256 x 256 tile size.
    fn flow_free_cellpose_model(in_channels: i64) -> (tempfile::TempDir, std::path::PathBuf) {
        trace_and_save_model(in_channels, TILE_SIZE, TILE_SIZE, |x| {
            let image_channel = x.narrow(1, 0, 1);
            let flow_y = image_channel.zeros_like();
            let flow_x = image_channel.zeros_like();
            let cell_logit = image_channel * 20.0 - 10.0;
            Tensor::cat(&[flow_y, flow_x, cell_logit], 1)
        })
    }

    #[test]
    fn tile_starts_returns_a_single_zero_start_for_a_tile_sized_or_smaller_canvas() {
        assert_eq!(tile_starts(1), vec![0]);
        assert_eq!(tile_starts(TILE_SIZE), vec![0]);
    }

    #[test]
    fn tile_starts_covers_a_larger_canvas_with_overlap() {
        // Ly=500 > TILE_SIZE: ny = ceil(1.2 * 500 / 256) = 3, tile starts
        // evenly spaced across [0, 500-256] = [0, 244], matching Cellpose's
        // own `transforms.make_tiles`.
        let starts = tile_starts(500);
        assert_eq!(starts, vec![0, 122, 244]);
        for &s in &starts {
            assert!(s >= 0 && s + TILE_SIZE <= 500);
        }
    }

    #[test]
    fn taper_mask_peaks_at_the_center_and_decays_but_never_reaches_zero_at_the_edges() {
        let mask = taper_mask(Device::Cpu);
        assert_eq!(mask.size(), vec![1, 1, TILE_SIZE, TILE_SIZE]);

        let at = |y: i64, x: i64| -> f64 {
            f64::try_from(mask.narrow(2, y, 1).narrow(3, x, 1).reshape([1])).unwrap()
        };
        let center = at(127, 127);
        let corner = at(0, 0);
        assert!(center > 0.99, "center weight should be ~1.0, got {center}");
        assert!(corner > 0.0, "taper mask must never reach exactly 0");
        assert!(
            corner < center,
            "corner weight ({corner}) should be far smaller than the center ({center})"
        );
    }

    #[test]
    fn run_model_tiled_reconstructs_exact_per_pixel_values_across_a_multi_tile_image() {
        // A pixel-wise (position-independent) toy model: whatever the tiling
        // grid or blend weights do, the "true" value at a given pixel is the
        // same in every tile that covers it, so a correct blend must
        // reconstruct it exactly - this isolates the padding/tiling/stitching
        // logic itself from the flow dynamics already covered above.
        let (_dir, model_path) = flow_free_cellpose_model(2);
        let model = tch::CModule::load_on_device(&model_path, Device::Cpu).unwrap();

        // 300x300 forces a 2x2 tile grid (tile_starts(300) has two
        // starts per axis) with a large overlap region; the left half is
        // background (0.0) and the right half is foreground (1.0), so the
        // split sits inside the overlap and exercises the blend directly.
        let (width, height): (i64, i64) = (300, 300);
        let mut values = vec![0f32; (width * height) as usize];
        for y in 0..height {
            for x in 150..width {
                values[(y * width + x) as usize] = 1.0;
            }
        }
        let image = Tensor::from_slice(&values)
            .to_kind(Kind::Float)
            .reshape([1, 1, height, width]);
        let input = Tensor::cat(&[image.shallow_clone(), image.zeros_like()], 1);

        let output = run_model_tiled(&model, &input, Device::Cpu).unwrap();
        assert_eq!(output.size(), vec![1, 3, height, width]);

        let cell_prob: Vec<f32> =
            Vec::try_from(&output.narrow(1, 2, 1).sigmoid().reshape([height * width])).unwrap();
        for y in 0..height as usize {
            for x in 0..width as usize {
                let p = cell_prob[y * width as usize + x];
                if x < 150 {
                    assert!(p < 0.01, "expected background at ({x},{y}), got {p}");
                } else {
                    assert!(p > 0.99, "expected foreground at ({x},{y}), got {p}");
                }
            }
        }
    }
}
