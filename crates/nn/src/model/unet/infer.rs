//! # infer
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How the U-Net runs over a tile - inside the engine slot: in pieces where
//! the tile needs them (`crate::pieces`), each piece mirror-padded,
//! normalized on the way, through the network and cropped back to the
//! planes asked for.

use super::pre::Affine;
use super::{MARGIN, Readout, UNet, UNetInput, UNetRaw};
use crate::errors::NnError;
use crate::pieces::{self, Planes, pad_plane};

/// What the network takes on the GPU per pixel of its input, per 64 filters
/// of its first level (where most of it lies): half of the 1650 bytes
/// measured in f32 for romantic-shark (64 filters, `tests::tile_ram`) - f16.
const GPU_BYTES_PER_PIXEL_PER_64_FILTERS: f64 = 825.0;

/// The tile through the network - whole, or in pieces sized to the RAM (or
/// the GPU's memory) with the receptive field's halo.
pub(super) fn in_pieces(net: &UNet, input: &UNetInput<'_>) -> Result<Planes, NnError> {
    let (width, height, channels) = (input.width, input.height, input.channels);
    let predict =
        |p: &[f32], w: usize, h: usize| predict(net, p, w, h, input.normalization, &input.readout);
    // The receptive field's halo, `None` for a model seeing the whole
    // image - measured once, on the network alone.
    let local_halo = || {
        if net.config.sees_whole_image() {
            return Ok(None);
        }
        if let Some(halo) = net.halo.get() {
            return Ok(*halo);
        }
        let probe = |p: &[f32], w: usize, h: usize| predict_raw(net, p, w, h, &input.readout);
        let halo = pieces::receptive_halo(channels, probe)?;
        Ok(*net.halo.get_or_init(|| halo))
    };
    let gpu_bytes = GPU_BYTES_PER_PIXEL_PER_64_FILTERS * net.config.initial_features as f64 / 64.0;
    let fitted = pieces::fit_gpu(input.pieces, net.gpu_memory, gpu_bytes, width, height);
    let plan = pieces::plan(width, height, fitted, local_halo, "U-Net")?;
    pieces::run_fitting(
        input.pixels,
        channels,
        width,
        height,
        plan,
        local_halo,
        predict,
        "U-Net",
    )
}

/// The planes as the network's answer.
pub(super) fn raw(planes: Planes) -> UNetRaw {
    UNetRaw {
        boundary: (planes.planes > 1).then_some(planes.rest),
        probabilities: planes.first,
    }
}

/// The network on one `width x height` image (`in_channels` planes),
/// padded, normalized and cropped back: the foreground (and boundary) plane.
fn predict(
    net: &UNet,
    pixels: &[f32],
    width: usize,
    height: usize,
    normalization: Option<Affine>,
    readout: &Readout,
) -> Result<Planes, NnError> {
    let (mut padded, (pw, ph), (left, top)) = padded(net, pixels, width, height);
    if let Some(affine) = normalization {
        affine.apply(&mut padded);
    }
    forward_and_read(net, &padded, (pw, ph), (left, top), width, height, readout)
}

/// [`predict`] without normalization - for measuring the halo.
fn predict_raw(
    net: &UNet,
    pixels: &[f32],
    width: usize,
    height: usize,
    readout: &Readout,
) -> Result<Planes, NnError> {
    let (padded, size, offset) = padded(net, pixels, width, height);
    forward_and_read(net, &padded, size, offset, width, height, readout)
}

/// `pixels` mirror-padded to the network's multiple: the padded planes,
/// their size and where the image starts in them.
fn padded(
    net: &UNet,
    pixels: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, (usize, usize), (usize, usize)) {
    let multiple = net.config.size_multiple();
    let (top, bottom) = padding(height, multiple);
    let (left, right) = padding(width, multiple);
    let (pw, ph) = (left + width + right, top + height + bottom);
    let mut padded = Vec::with_capacity(net.config.in_channels * pw * ph);
    for plane in pixels.chunks_exact(width * height) {
        pad_plane(
            plane,
            width,
            height,
            (left, right, top, bottom),
            &mut padded,
        );
    }
    (padded, (pw, ph), (left, top))
}

/// The network on the padded planes, then the planes `readout` asks for,
/// cropped back to `width x height`.
fn forward_and_read(
    net: &UNet,
    padded: &[f32],
    (pw, ph): (usize, usize),
    (left, top): (usize, usize),
    width: usize,
    height: usize,
    readout: &Readout,
) -> Result<Planes, NnError> {
    let output = net.try_forward_raw(padded, ph, pw, &net.device)?;
    let channels = net.config.out_channels;
    if output.len() != channels * pw * ph {
        return Err(NnError::Internal(format!(
            "U-Net output has {} values, {} expected",
            output.len(),
            channels * pw * ph
        )));
    }
    // Channel `c`'s value at the tile's pixel (x, y).
    let at = |c: usize, x: usize, y: usize| output[(c * ph + y + top) * pw + x + left];
    let fg = readout.foreground_channel.min(channels - 1);
    let mut first = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let value = if channels > 1 && readout.softmax {
                let max = (0..channels).map(|c| at(c, x, y)).fold(f32::MIN, f32::max);
                let sum: f32 = (0..channels).map(|c| (at(c, x, y) - max).exp()).sum();
                (at(fg, x, y) - max).exp() / sum
            } else {
                at(fg, x, y)
            };
            first.push(value);
        }
    }
    // Boundary-aware models (e.g. affable-shark) predict an independent
    // boundary map: it opens thin gaps between touching objects.
    let rest = match readout.boundary_channel {
        Some(b) if channels > 1 && b < channels => {
            let mut rest = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    rest.push(at(b, x, y));
                }
            }
            rest
        }
        _ => Vec::new(),
    };
    Ok(Planes {
        planes: if rest.is_empty() { 1 } else { 2 },
        width,
        height,
        first,
        rest,
    })
}

/// Padding `(before, after)` for one side of `len` pixels: [`MARGIN`]
/// before, and after whatever brings the total up to a multiple of
/// `multiple` with at least `MARGIN` of context.
pub(super) fn padding(len: usize, multiple: usize) -> (usize, usize) {
    let total = (len + 2 * MARGIN).next_multiple_of(multiple);
    (MARGIN, total - len - MARGIN)
}
