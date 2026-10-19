//! # stardist
//!
//! **Author:** Joachim Danmayr
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.
//!
//! StarDist inference: the object probability and the ray distances per
//! grid cell, flattened.

use super::{Result, scheduler::TorchScheduler};
use crate::api::{Image, Response};
use std::path::Path;
use tch::{CModule, Device, IValue, Kind, Tensor};

pub(crate) fn run(
    scheduler: &TorchScheduler,
    model_path: &Path,
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
            let input = Tensor::from_slice(pixels)
                .to_device(device)
                .to_kind(Kind::Float)
                .reshape([1, 1, height as i64, width as i64]);

            let (prob_tensor, dist_tensor) = split_outputs(model, input)?;

            let prob_sizes = prob_tensor.size();
            let dist_sizes = dist_tensor.size();
            let grid_height = prob_sizes[prob_sizes.len() - 2] as usize;
            let grid_width = prob_sizes[prob_sizes.len() - 1] as usize;
            let n_rays = dist_sizes[dist_sizes.len() - 3] as usize;

            Ok(Response::StarDist {
                prob: flatten(&prob_tensor, grid_height * grid_width)?,
                dist: flatten(&dist_tensor, n_rays * grid_height * grid_width)?,
                grid_height,
                grid_width,
                n_rays,
            })
        },
    )
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
