//! # network
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The Cellpose-SAM network in burn, ported from Cellpose's `CPSAM.forward`
//! (`cellpose/vit.py`, BSD-3, HHMI) and SAM's image encoder
//! (`segment_anything/modeling/image_encoder.py`, Apache-2.0, Meta):
//!
//! - patch embedding: a `patch_size`² convolution with stride `patch_size`
//!   (only the weights of the channels the tile has), plus the absolute
//!   positional embedding `[1, T, T, C]`;
//! - `depth` blocks, each: LayerNorm → multi-head self-attention over all
//!   T² tokens (global - Cellpose-SAM sets every block's window to 0) with
//!   SAM's decomposed relative positions → residual → LayerNorm → MLP (GELU)
//!   → residual;
//! - neck: 1x1 convolution → LayerNorm over channels → 3x3 convolution →
//!   LayerNorm over channels (both without bias, eps 1e-6);
//! - a 1x1 convolution to `out_channels * patch_size²` and the pixel shuffle
//!   Cellpose writes as a transposed convolution with an identity kernel
//!   (`W2`).

use super::config::CellposeSamConfig;
use crate::errors::NnError;
use crate::weights::{f32_values, shape_of, tensor_from};
use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Device, Tensor, TensorData};
use safetensors::SafeTensors;

/// SAM's LayerNorm epsilon (blocks and neck).
const EPSILON: f64 = 1e-6;

/// The dtype the matrix products and convolutions run in: f16 on the GPU
/// (`half`, its tensor cores - burn's f32 kernels on wgpu use none), else
/// f32. Norms, softmax, GELU, biases and the residual stream stay f32.
fn dtype(half: bool) -> DType {
    if half { DType::F16 } else { DType::F32 }
}

/// A PyTorch `nn.Linear`: the weight `[in, out]` (transposed from
/// PyTorch's `[out, in]` once), the bias.
#[derive(Module, Debug)]
struct Lin {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl Lin {
    /// Over the last dimension of `x`, as one matrix product of all its rows
    /// - `[B, T, T, C]` as it is would be `B * T` products of `T` rows each,
    /// which a GPU runs far below its speed (seen as `m: 32` in burn's
    /// autotune keys). In f16 when `half`, the bias added in f32.
    fn forward<const D: usize>(&self, x: Tensor<D>, half: bool) -> Tensor<D> {
        let dims = x.dims();
        let n_in = dims[D - 1];
        let rows = dims.iter().product::<usize>() / n_in;
        let x = x.reshape([rows, n_in]);
        let y = if half {
            x.cast(DType::F16)
                .matmul(self.weight.clone())
                .cast(DType::F32)
        } else {
            x.matmul(self.weight.clone())
        };
        let y = y + self.bias.clone().unsqueeze_dim::<2>(0);
        let mut out = dims;
        out[D - 1] = self.weight.dims()[1];
        y.reshape(out)
    }
}

/// A PyTorch `nn.Linear` from `<prefix>.weight` `[out, in]` and
/// `<prefix>.bias`.
fn linear(
    file: &SafeTensors<'_>,
    prefix: &str,
    n_in: usize,
    n_out: usize,
    device: &Device,
) -> Result<Lin, NnError> {
    let weight = tensor_from(file, &format!("{prefix}.weight"), [n_out, n_in], device)?;
    let bias = tensor_from(file, &format!("{prefix}.bias"), [n_out], device)?;
    Ok(Lin {
        weight: weight.transpose(),
        bias,
    })
}

/// `conv2d` with the bias added afterwards in f32 when `half` (burn's
/// tensor-core convolutions take no bias, see the U-Net's `conv_forward`);
/// `weight` already in [`dtype`]`(half)`.
fn conv(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
    half: bool,
) -> Tensor<4> {
    if !half {
        return conv2d(x, weight, bias, options);
    }
    let y = conv2d(x.cast(DType::F16), weight, None, options).cast(DType::F32);
    match bias {
        Some(bias) => {
            let c = bias.dims()[0];
            y + bias.reshape([1, c, 1, 1])
        }
        None => y,
    }
}

/// A PyTorch `nn.LayerNorm(dim, eps=1e-6)` from `<prefix>.weight` / `.bias`.
fn layer_norm(
    file: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    device: &Device,
) -> Result<LayerNorm, NnError> {
    let mut norm = LayerNormConfig::new(dim).with_epsilon(EPSILON).init(device);
    norm.gamma = Param::from_tensor(tensor_from(
        file,
        &format!("{prefix}.weight"),
        [dim],
        device,
    )?);
    norm.beta = Some(Param::from_tensor(tensor_from(
        file,
        &format!("{prefix}.bias"),
        [dim],
        device,
    )?));
    Ok(norm)
}

/// `rel_pos` (`len` rows of `dim`) linearly resized to `size` rows as
/// PyTorch's `F.interpolate(mode="linear")` (`align_corners=False`) - SAM's
/// `get_rel_pos` when the table was made for another token grid.
fn resize_rows(rel_pos: &[f32], len: usize, dim: usize, size: usize) -> Vec<f32> {
    if len == size {
        return rel_pos.to_vec();
    }
    let scale = len as f64 / size as f64;
    let mut out = Vec::with_capacity(size * dim);
    for i in 0..size {
        let src = (scale * (i as f64 + 0.5) - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(len - 1);
        let i1 = (i0 + 1).min(len - 1);
        let t = (src - i0 as f64) as f32;
        for c in 0..dim {
            let (a, b) = (rel_pos[i0 * dim + c], rel_pos[i1 * dim + c]);
            out.push(a * (1.0 - t) + b * t);
        }
    }
    out
}

/// SAM's `get_rel_pos` for a query and key grid of `tokens` per side: the
/// table's row for each (query, key) offset, `[tokens, tokens, dim]`.
fn rel_pos_table(
    file: &SafeTensors<'_>,
    name: &str,
    tokens: usize,
    dim: usize,
    device: &Device,
) -> Result<Tensor<3>, NnError> {
    let shape = shape_of(file, name)?;
    let len = match shape.as_slice() {
        [len, d] if *d == dim && (1..=4096).contains(len) => *len,
        _ => {
            return Err(NnError::InvalidModelFile(format!(
                "the tensor {name} has the shape {shape:?}, [rows, {dim}] expected"
            )));
        }
    };
    let size = 2 * tokens - 1;
    let rows = resize_rows(&f32_values(file, name, &shape)?, len, dim, size);
    let mut table = Vec::with_capacity(tokens * tokens * dim);
    for q in 0..tokens {
        for k in 0..tokens {
            let row = q + tokens - 1 - k;
            table.extend_from_slice(&rows[row * dim..(row + 1) * dim]);
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(table, [tokens, tokens, dim]),
        device,
    ))
}

/// One ViT block with global attention.
#[derive(Module, Debug)]
pub(crate) struct Block {
    norm1: LayerNorm,
    qkv: Lin,
    proj: Lin,
    /// `[T, T, head_dim]`: the height offsets' relative positions
    /// (transposed for the batched matmul: `[T(q), head_dim, T(k)]`).
    rel_h: Tensor<3>,
    rel_w: Tensor<3>,
    norm2: LayerNorm,
    lin1: Lin,
    lin2: Lin,
    #[module(skip)]
    num_heads: usize,
    /// Matrix products in f16 (see [`dtype`]).
    #[module(skip)]
    half: bool,
}

impl Block {
    fn load(
        file: &SafeTensors<'_>,
        i: usize,
        config: &CellposeSamConfig,
        device: &Device,
    ) -> Result<Self, NnError> {
        let p = |name: &str| format!("encoder.blocks.{i}.{name}");
        let (dim, t) = (config.embed_dim, config.tokens());
        let head_dim = dim / config.num_heads;
        let rel = |name: &str| -> Result<Tensor<3>, NnError> {
            Ok(rel_pos_table(file, &p(name), t, head_dim, device)?.swap_dims(1, 2))
        };
        Ok(Self {
            norm1: layer_norm(file, &p("norm1"), dim, device)?,
            qkv: linear(file, &p("attn.qkv"), dim, 3 * dim, device)?,
            proj: linear(file, &p("attn.proj"), dim, dim, device)?,
            rel_h: rel("attn.rel_pos_h")?,
            rel_w: rel("attn.rel_pos_w")?,
            norm2: layer_norm(file, &p("norm2"), dim, device)?,
            lin1: linear(file, &p("mlp.lin1"), dim, config.mlp_dim, device)?,
            lin2: linear(file, &p("mlp.lin2"), config.mlp_dim, dim, device)?,
            num_heads: config.num_heads,
            half: false,
        })
    }

    /// The weights of the matrix products in [`dtype`]`(half)`.
    fn set_half(&mut self, half: bool) {
        let d = dtype(half);
        for lin in [
            &mut self.qkv,
            &mut self.proj,
            &mut self.lin1,
            &mut self.lin2,
        ] {
            lin.weight = lin.weight.clone().cast(d);
        }
        self.rel_h = self.rel_h.clone().cast(d);
        self.rel_w = self.rel_w.clone().cast(d);
        self.half = half;
    }

    /// `[B, T, T, C]` → `[B, T, T, C]`.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let half = self.half;
        let x = x.clone() + self.attention(self.norm1.forward(x));
        x.clone()
            + self
                .lin2
                .forward(gelu(self.lin1.forward(self.norm2.forward(x), half)), half)
    }

    /// SAM's `Attention.forward` with `use_rel_pos` (decomposed relative
    /// positions, `add_decomposed_rel_pos`).
    fn attention(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, h, w, c] = x.dims();
        let (heads, n) = (self.num_heads, h * w);
        let hd = c / heads;
        let (half, d) = (self.half, dtype(self.half));
        // [B, N, 3, heads, hd] → [3, B*heads, N, hd].
        let qkv = self
            .qkv
            .forward(x, half)
            .reshape([b, n, 3, heads, hd])
            .permute([2, 0, 3, 1, 4])
            .reshape([3, b * heads, n, hd]);
        let bh = b * heads;
        let part = |i: usize| qkv.clone().narrow(0, i, 1).reshape([bh, n, hd]);
        let (q, k, v) = (part(0), part(1), part(2));
        let attn = (q.clone() * (hd as f64).powf(-0.5))
            .cast(d)
            .matmul(k.swap_dims(1, 2).cast(d))
            .cast(DType::F32);
        // rel_h[b, y, x, ky] = q[b, y, x, :] · Rh[y, ky, :] - per query row
        // y a matrix product; rel_w the same per query column x.
        let r_q = q.reshape([bh, h, w, hd]).cast(d);
        let rel_h = r_q
            .clone()
            .permute([1, 0, 2, 3])
            .reshape([h, bh * w, hd])
            .matmul(self.rel_h.clone())
            .cast(DType::F32)
            .reshape([h, bh, w, h])
            .permute([1, 0, 2, 3]);
        let rel_w = r_q
            .permute([2, 0, 1, 3])
            .reshape([w, bh * h, hd])
            .matmul(self.rel_w.clone())
            .cast(DType::F32)
            .reshape([w, bh, h, w])
            .permute([1, 2, 0, 3]);
        let attn = attn.reshape([bh, h, w, h, w])
            + rel_h.unsqueeze_dim::<5>(4)
            + rel_w.unsqueeze_dim::<5>(3);
        let attn = softmax(attn.reshape([bh, n, n]), 2);
        let out = attn
            .cast(d)
            .matmul(v.cast(d))
            .cast(DType::F32)
            .reshape([b, heads, h, w, hd])
            .permute([0, 2, 3, 1, 4])
            .reshape([b, h, w, c]);
        self.proj.forward(out, half)
    }
}

/// LayerNorm over the channels of `[B, C, H, W]` (SAM's `LayerNorm2d`).
#[derive(Module, Debug)]
struct LayerNorm2d {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

impl LayerNorm2d {
    fn load(
        file: &SafeTensors<'_>,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self, NnError> {
        Ok(Self {
            weight: tensor_from(file, &format!("{prefix}.weight"), [dim], device)?,
            bias: tensor_from(file, &format!("{prefix}.bias"), [dim], device)?,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let c = self.weight.dims()[0];
        let u = x.clone().mean_dim(1);
        let centred = x - u;
        let s = centred.clone().square().mean_dim(1);
        let x = centred / (s + EPSILON).sqrt();
        x * self.weight.clone().reshape([1, c, 1, 1]) + self.bias.clone().reshape([1, c, 1, 1])
    }
}

/// Cellpose-SAM, see the module docs.
#[derive(Module, Debug)]
pub(crate) struct CellposeSamNet {
    patch_weight: Tensor<4>,
    patch_bias: Tensor<1>,
    pos_embed: Tensor<4>,
    blocks: Vec<Block>,
    neck0: Tensor<4>,
    neck1: LayerNorm2d,
    neck2: Tensor<4>,
    neck3: LayerNorm2d,
    out_weight: Tensor<4>,
    out_bias: Tensor<1>,
    #[module(skip)]
    patch_size: usize,
    /// Matrix products and convolutions in f16 (see [`dtype`]).
    #[module(skip)]
    half: bool,
}

impl CellposeSamNet {
    /// Matrix products and convolutions in f16 (`true`, the GPU) or f32:
    /// their weights converted once. Back to f32 keeps the f16 rounding -
    /// set it once, after loading.
    pub(crate) fn set_half(&mut self, half: bool) {
        let d = dtype(half);
        for block in &mut self.blocks {
            block.set_half(half);
        }
        for weight in [
            &mut self.patch_weight,
            &mut self.neck0,
            &mut self.neck2,
            &mut self.out_weight,
        ] {
            *weight = weight.clone().cast(d);
        }
        self.half = half;
    }

    /// The network of `config` with the tensors of `file` (Cellpose's names,
    /// every shape checked).
    pub(crate) fn load(
        file: &SafeTensors<'_>,
        config: &CellposeSamConfig,
        device: &Device,
    ) -> Result<Self, NnError> {
        let (c, p, t) = (config.embed_dim, config.patch_size, config.tokens());
        let (neck, out) = (config.neck_dim, config.out_channels * p * p);
        let blocks = (0..config.depth)
            .map(|i| Block::load(file, i, config, device))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            patch_weight: tensor_from(
                file,
                "encoder.patch_embed.proj.weight",
                [c, config.in_channels, p, p],
                device,
            )?,
            patch_bias: tensor_from(file, "encoder.patch_embed.proj.bias", [c], device)?,
            pos_embed: tensor_from(file, "encoder.pos_embed", [1, t, t, c], device)?,
            blocks,
            neck0: tensor_from(file, "encoder.neck.0.weight", [neck, c, 1, 1], device)?,
            neck1: LayerNorm2d::load(file, "encoder.neck.1", neck, device)?,
            neck2: tensor_from(file, "encoder.neck.2.weight", [neck, neck, 3, 3], device)?,
            neck3: LayerNorm2d::load(file, "encoder.neck.3", neck, device)?,
            out_weight: tensor_from(file, "out.weight", [out, neck, 1, 1], device)?,
            out_bias: tensor_from(file, "out.bias", [out], device)?,
            patch_size: p,
            half: false,
        })
    }

    /// `[B, C, S, S]` tiles (C up to the patch embedding's channels, S the
    /// tile size) → `[B, out_channels, S, S]`: dY, dX, the cell logit.
    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, channels, _, _] = x.dims();
        let (p, half) = (self.patch_size, self.half);
        let weight = self.patch_weight.clone().narrow(1, 0, channels);
        let x = conv(
            x,
            weight,
            Some(self.patch_bias.clone()),
            ConvOptions::new([p, p], [0, 0], [1, 1], 1),
            half,
        );
        let mut x = x.permute([0, 2, 3, 1]) + self.pos_embed.clone();
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = x.permute([0, 3, 1, 2]);
        let no_pad = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let x = self
            .neck1
            .forward(conv(x, self.neck0.clone(), None, no_pad.clone(), half));
        let x = self.neck3.forward(conv(
            x,
            self.neck2.clone(),
            None,
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            half,
        ));
        let x = conv(
            x,
            self.out_weight.clone(),
            Some(self.out_bias.clone()),
            no_pad,
            half,
        );
        // The pixel shuffle: channel o * p² + dy * p + dx of token (y, x) is
        // output o's pixel (y * p + dy, x * p + dx).
        let [_, _, t, _] = x.dims();
        let n_out = x.dims()[1] / (p * p);
        x.reshape([b, n_out, p, p, t, t])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([b, n_out, t * p, t * p])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PyTorch's `F.interpolate(x, size, mode="linear")` on a known case:
    /// 3 rows to 5 (align_corners=False).
    #[test]
    fn rows_resize_like_pytorch() {
        let rows = [0.0, 10.0, 20.0];
        let out = resize_rows(&rows, 3, 1, 5);
        // PyTorch: tensor([0., 4., 10., 16., 20.])
        let expected = [0.0, 4.0, 10.0, 16.0, 20.0];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{out:?}");
        }
        assert_eq!(resize_rows(&rows, 3, 1, 3), rows);
    }
}
