//! # stardist
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist inference: the object probability and the ray distances per
//! grid cell, flattened.

use super::pieces::{self, Planes};
use super::{Result, scheduler::TorchScheduler};
use crate::api::{Image, Response};
use std::path::Path;
use tch::{CModule, Device, IValue, Kind, Tensor};

pub(crate) fn run(
    scheduler: &TorchScheduler,
    model_path: &Path,
    pieces: crate::api::Pieces,
    image: Image<'_>,
) -> Result<Response> {
    let (width, height) = (image.width, image.height);
    let pixels = &image.data[..width * height];
    scheduler.run(
        model_path,
        |e| {
            format!(
                "Failed to load StarDist model from {}: {e}",
                model_path.display()
            )
        },
        |model, device| {
            let predict = |pixels: &[f32], w: usize, h: usize| predict(model, device, pixels, w, h);
            // A big tile in pieces (RAM of one piece), see `pieces`.
            let plan = pieces::plan(
                width,
                height,
                pieces,
                || scheduler.halo(model_path, || pieces::receptive_halo(predict)),
                model_path,
            )?;
            let planes = match plan {
                Some((halo, piece)) => pieces::run(pixels, width, height, halo, piece, predict)?,
                None => predict(pixels, width, height)?,
            };
            Ok(Response::StarDist {
                prob: planes.first,
                dist: planes.rest,
                grid_height: planes.height,
                grid_width: planes.width,
                n_rays: planes.planes - 1,
            })
        },
    )
}

/// Side lengths StarDist networks run at are a multiple of this - StarDist
/// pads to its `div_by` (2^depth x grid: 16 for `2D_versatile_fluo`); else
/// the upsampled and the skipped features don't fit ("Expected size 137 but
/// got size 136" at 1100 px).
const SIZE_MULTIPLE: usize = 16;

/// The model on a `width x height` image: the object probability, then the
/// ray distances, per grid cell - as [`Planes`]. A side that isn't a
/// multiple of [`SIZE_MULTIPLE`] is padded at the end (mirrored, as
/// StarDist) and the output cropped back to the image's grid cells.
pub(super) fn predict(
    model: &CModule,
    device: Device,
    pixels: &[f32],
    width: usize,
    height: usize,
) -> Result<Planes> {
    let input = Tensor::from_slice(pixels)
        .to_device(device)
        .to_kind(Kind::Float)
        .reshape([1, 1, height as i64, width as i64]);
    let right = width.next_multiple_of(SIZE_MULTIPLE) - width;
    let bottom = height.next_multiple_of(SIZE_MULTIPLE) - height;
    let input = match (right, bottom) {
        (0, 0) => input,
        // Mirroring needs every pad smaller than its side; a tiny tile
        // repeats its edge pixels instead.
        _ => {
            let mode = if right < width && bottom < height {
                "reflect"
            } else {
                "replicate"
            };
            input.pad([0, right as i64, 0, bottom as i64], mode, None)
        }
    };
    let (padded_width, padded_height) = (width + right, height + bottom);

    let (prob_tensor, dist_tensor) = split_outputs(model, input)?;

    let prob_sizes = prob_tensor.size();
    let dist_sizes = dist_tensor.size();
    if prob_sizes.len() < 2 || dist_sizes.len() < 3 {
        return Err("StarDist model output has too few dimensions".into());
    }
    let out_height = prob_sizes[prob_sizes.len() - 2] as usize;
    let out_width = prob_sizes[prob_sizes.len() - 1] as usize;
    let n_rays = dist_sizes[dist_sizes.len() - 3] as usize;
    if out_width == 0 || out_height == 0 {
        return Err("StarDist model returned an empty output".into());
    }
    // The grid cells of the image itself (not of the padding).
    let grid_x = (padded_width / out_width).max(1);
    let grid_y = (padded_height / out_height).max(1);
    let grid_width = (width / grid_x).clamp(1, out_width);
    let grid_height = (height / grid_y).clamp(1, out_height);
    let crop = |t: &Tensor| {
        let dims = t.size().len() as i64;
        t.narrow(dims - 2, 0, grid_height as i64)
            .narrow(dims - 1, 0, grid_width as i64)
    };

    Ok(Planes {
        planes: 1 + n_rays,
        width: grid_width,
        height: grid_height,
        first: flatten(&crop(&prob_tensor), grid_height * grid_width)?,
        rest: flatten(&crop(&dist_tensor), n_rays * grid_height * grid_width)?,
    })
}

fn flatten(tensor: &Tensor, len: usize) -> Result<Vec<f32>> {
    let err = |e: tch::TchError| format!("StarDist inference failed: {e}");
    let out = tensor
        .f_to_device(Device::Cpu)
        .and_then(|t| t.f_reshape([len as i64]))
        .map_err(err)?;
    Vec::try_from(&out).map_err(err)
}

/// Runs the model and splits its output into `(probability, distance)` tensors,
/// supporting both the two-separate-tensors and the single-concatenated-tensor
/// TorchScript export conventions.
fn split_outputs(model: &CModule, input: Tensor) -> Result<(Tensor, Tensor)> {
    // Without `no_grad`, the model's own weights (loaded with
    // `requires_grad = true`, the default for `nn.Parameter`) make this
    // forward pass record a full autograd graph even though `.eval()`
    // was called - wasting memory that's never needed for inference.
    let output = tch::no_grad(|| model.forward_is(&[IValue::Tensor(input)]))
        .map_err(|e| format!("StarDist inference failed: {e}"))?;

    let tensors: Vec<Tensor> = match output {
        IValue::Tensor(t) => vec![t],
        IValue::Tuple(items) | IValue::GenericList(items) => items
            .into_iter()
            .filter_map(|v| match v {
                IValue::Tensor(t) => Some(t),
                _ => None,
            })
            .collect(),
        other => {
            return Err(format!(
                "StarDist model returned an unsupported output type: {other:?}"
            ));
        }
    };

    match tensors.as_slice() {
        [single] => {
            let sizes = single.size();
            if sizes.len() < 3 {
                return Err("StarDist model output has too few dimensions".into());
            }
            let channel_dim = sizes.len() as i64 - 3;
            let channels = sizes[sizes.len() - 3];
            if channels < 2 {
                return Err(
                    "StarDist model output has fewer than 2 channels; expected a probability channel plus ray-distance channels".into(),
                );
            }
            let prob = single.narrow(channel_dim, 0, 1);
            let dist = single.narrow(channel_dim, 1, channels - 1);
            Ok((prob, dist))
        }
        [_, _, ..] => {
            let prob_idx = tensors.iter().position(|t| {
                let sizes = t.size();
                sizes.len() >= 3 && sizes[sizes.len() - 3] == 1
            });
            match prob_idx {
                Some(idx) => {
                    let dist_idx = if idx == 0 { 1 } else { 0 };
                    Ok((tensors[idx].shallow_clone(), tensors[dist_idx].shallow_clone()))
                }
                None => Err(
                    "StarDist model returned multiple outputs but none has a single-channel probability map".into(),
                ),
            }
        }
        [] => Err("StarDist model returned no tensor outputs".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::test_support::bench_stardist_model;

    /// A real StarDist layout (`2D_versatile_fluo`, grid 2) on a tile whose
    /// sides aren't multiples of 16 - which failed in the network before
    /// the padding: it runs, gives the tile's own grid cells, and away from
    /// the padded edge (the receptive field, 112 px) the same values as on a
    /// larger image holding the tile.
    #[test]
    fn a_tile_not_a_multiple_of_16_runs_and_matches_a_larger_image() {
        let (_dir, path) = bench_stardist_model();
        let model = CModule::load_on_device(&path, Device::Cpu).unwrap();
        let (big, width, height) = (304usize, 300usize, 290usize);
        let image: Vec<f32> = (0..big * big)
            .map(|i| (((i % big) / 23 + (i / big) / 19) % 3) as f32 * 0.4)
            .collect();
        let tile: Vec<f32> = (0..height)
            .flat_map(|y| image[y * big..y * big + width].iter().copied())
            .collect();
        let whole = predict(&model, Device::Cpu, &image, big, big).unwrap();
        let part = predict(&model, Device::Cpu, &tile, width, height).unwrap();
        assert_eq!((part.width, part.height), (width / 2, height / 2));
        assert_eq!(part.planes, 33);
        let range = whole
            .first
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        for plane in 0..part.planes {
            for y in 0..(height - 112) / 2 {
                for x in 0..(width - 112) / 2 {
                    let (a, b) = (
                        part.plane(plane)[y * part.width + x],
                        whole.plane(plane)[y * whole.width + x],
                    );
                    assert!(
                        (a - b).abs() <= 1e-5 * range.max(b.abs()),
                        "plane {plane} ({x}, {y}): {a} vs {b}"
                    );
                }
            }
        }
    }
}
