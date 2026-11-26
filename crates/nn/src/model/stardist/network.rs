//! # network
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The StarDist 2D network in burn, laid out as StarDist's `StarDist2D._build`
//! with CSBDeep's `unet_block` (both BSD-3):
//!
//! - grid: while the pooled size isn't the grid, `unet_conv_per_depth`
//!   convolutions of `unet_filter_base` filters and a max pooling by 2 on the
//!   axes still short of the grid;
//! - U-Net down: per level `n`, convolutions of `base * 2^n` filters, kept
//!   for the skip, max pooling by 2;
//! - middle: `conv_per_depth - 1` convolutions of `base * 2^depth`, one of
//!   `base * 2^(depth-1)`;
//! - up, deepest first: nearest x2 upsampling, the skip appended after it,
//!   `conv_per_depth - 1` convolutions of `base * 2^n`, one of
//!   `base * 2^(n-1)` (level 0: `base`);
//! - `features` (`conv_after_unet` filters, if any), then `prob` (1x1,
//!   sigmoid) and `dist` (1x1, linear).
//!
//! Every convolution "same"-padded, ReLU (except the heads).

use super::config::StarDistConfig;
use crate::errors::NnError;
use crate::weights::tensor_from;
use burn::module::Module;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::module::{conv2d, interpolate, max_pool2d};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, MaxPoolOptions};
use burn::tensor::{DType, Device, Tensor};
use safetensors::SafeTensors;

/// One convolution: `[out, in, k, k]` and its bias.
#[derive(Module, Debug)]
struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
}

impl Conv {
    fn load(
        file: &SafeTensors<'_>,
        name: &str,
        n_in: usize,
        n_out: usize,
        k: usize,
        device: &Device,
    ) -> Result<Self, NnError> {
        Ok(Self {
            weight: tensor_from(file, &format!("{name}.weight"), [n_out, n_in, k, k], device)?,
            bias: tensor_from(file, &format!("{name}.bias"), [n_out], device)?,
        })
    }

    /// "same"-padded convolution (odd kernel: the same on both sides). When
    /// `half` (the GPU) `x`, the weights and the result are f16 and the bias
    /// is added after: CubeCL's tensor core kernels take neither f32 nor a
    /// bias (the U-Net's `network::conv_forward`).
    fn forward(&self, x: Tensor<4>, half: bool) -> Tensor<4> {
        let pad = self.weight.dims()[2] / 2;
        let options = ConvOptions::new([1, 1], [pad, pad], [1, 1], 1);
        if !half {
            return conv2d(x, self.weight.clone(), Some(self.bias.clone()), options);
        }
        let n = self.bias.dims()[0];
        conv2d(x, self.weight.clone(), None, options) + self.bias.clone().reshape([1, n, 1, 1])
    }

    /// Weights and bias in `dtype`.
    fn cast(&mut self, dtype: DType) {
        self.weight = self.weight.clone().cast(dtype);
        self.bias = self.bias.clone().cast(dtype);
    }
}

/// Max pooling by `pool` (no overlap, no padding).
fn pool(x: Tensor<4>, pool: [usize; 2]) -> Tensor<4> {
    max_pool2d(x, MaxPoolOptions::new(pool).with_stride(pool))
}

/// Convolutions with ReLU, one after the other.
fn convs(convs: &[Conv], mut x: Tensor<4>, half: bool) -> Tensor<4> {
    for conv in convs {
        x = relu(conv.forward(x, half));
    }
    x
}

/// StarDist 2D, see the module docs.
#[derive(Module, Debug)]
pub(crate) struct StarDistNet {
    /// Per grid round its convolutions; the poolings in `grid_pools`.
    grid: Vec<Vec<Conv>>,
    down: Vec<Vec<Conv>>,
    middle: Vec<Conv>,
    /// Deepest level first.
    up: Vec<Vec<Conv>>,
    features: Option<Conv>,
    prob: Conv,
    dist: Conv,
    #[module(skip)]
    grid_pools: Vec<[usize; 2]>,
    /// The whole network in f16 (the GPU, see `Conv::forward`); set with
    /// [`Self::set_half`].
    #[module(skip)]
    half: bool,
}

impl StarDistNet {
    /// The network of `config` with the tensors of `file` (see `config.rs`),
    /// every shape following from `config` and checked.
    pub(crate) fn load(
        file: &SafeTensors<'_>,
        config: &StarDistConfig,
        device: &Device,
    ) -> Result<Self, NnError> {
        let (k, base, n_conv) = (
            config.kernel_size,
            config.unet_filter_base,
            config.unet_conv_per_depth,
        );
        let level = |n: usize| base << n;
        let conv =
            |name: &str, n_in: usize, n_out: usize| Conv::load(file, name, n_in, n_out, k, device);

        // Grid rounds.
        let (mut grid, mut grid_pools) = (Vec::new(), Vec::new());
        let (mut pooled, mut channels, mut index) = ([1, 1], config.in_channels, 0);
        while pooled != config.grid {
            let pool = [0, 1].map(|a| if config.grid[a] > pooled[a] { 2 } else { 1 });
            pooled = [pooled[0] * pool[0], pooled[1] * pool[1]];
            let mut round = Vec::new();
            for _ in 0..n_conv {
                round.push(conv(&format!("grid.{index}"), channels, base)?);
                channels = base;
                index += 1;
            }
            grid.push(round);
            grid_pools.push(pool);
        }

        // U-Net.
        let mut down = Vec::new();
        let mut skips = Vec::new();
        for n in 0..config.unet_depth {
            let mut convs = Vec::new();
            for i in 0..n_conv {
                convs.push(conv(&format!("down_level_{n}_no_{i}"), channels, level(n))?);
                channels = level(n);
            }
            skips.push(channels);
            down.push(convs);
        }
        let mut middle = Vec::new();
        for i in 0..n_conv - 1 {
            middle.push(conv(
                &format!("middle_{i}"),
                channels,
                level(config.unet_depth),
            )?);
            channels = level(config.unet_depth);
        }
        let to = level(config.unet_depth.saturating_sub(1));
        middle.push(conv(&format!("middle_{n_conv}"), channels, to)?);
        channels = to;
        let mut up = Vec::new();
        for n in (0..config.unet_depth).rev() {
            channels += skips[n];
            let mut convs = Vec::new();
            for i in 0..n_conv - 1 {
                convs.push(conv(&format!("up_level_{n}_no_{i}"), channels, level(n))?);
                channels = level(n);
            }
            let to = level(n.saturating_sub(1));
            convs.push(conv(&format!("up_level_{n}_no_{n_conv}"), channels, to)?);
            channels = to;
            up.push(convs);
        }

        let features = match config.conv_after_unet {
            0 => None,
            f => {
                let c = conv("features", channels, f)?;
                channels = f;
                Some(c)
            }
        };
        Ok(Self {
            grid,
            down,
            middle,
            up,
            features,
            prob: Conv::load(file, "prob", channels, 1, 1, device)?,
            dist: Conv::load(file, "dist", channels, config.n_rays, 1, device)?,
            grid_pools,
            half: false,
        })
    }

    /// The whole network in f16 (`true`, the GPU's tensor cores: weights
    /// converted once, activations stay f16 from the input to the two
    /// outputs) or f32. Back to f32 keeps the f16 rounding - set it once.
    pub(crate) fn set_half(&mut self, half: bool) {
        let dtype = if half { DType::F16 } else { DType::F32 };
        let all = self
            .grid
            .iter_mut()
            .chain(self.down.iter_mut())
            .chain(self.up.iter_mut())
            .flatten()
            .chain(self.middle.iter_mut())
            .chain(self.features.iter_mut())
            .chain([&mut self.prob, &mut self.dist]);
        for conv in all {
            conv.cast(dtype);
        }
        self.half = half;
    }

    /// `[B, C, H, W]` (sides multiples of `config.div_by()`) →
    /// `(prob [B, 1, H/gy, W/gx], dist [B, n_rays, H/gy, W/gx])`, f32.
    pub(crate) fn forward(&self, mut x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        if self.half {
            x = x.cast(DType::F16);
        }
        for (round, pool_size) in self.grid.iter().zip(&self.grid_pools) {
            x = pool(convs(round, x, self.half), *pool_size);
        }
        let mut skips = Vec::with_capacity(self.down.len());
        for level in &self.down {
            x = convs(level, x, self.half);
            skips.push(x.clone());
            x = pool(x, [2, 2]);
        }
        x = convs(&self.middle, x, self.half);
        for (level, skip) in self.up.iter().zip(skips.into_iter().rev()) {
            let [_, _, h, w] = x.dims();
            let upsampled = interpolate(
                x,
                InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([2 * h, 2 * w]),
            );
            x = convs(level, Tensor::cat(vec![upsampled, skip], 1), self.half);
        }
        if let Some(features) = &self.features {
            x = relu(features.forward(x, self.half));
        }
        (
            sigmoid(self.prob.forward(x.clone(), self.half).cast(DType::F32)),
            self.dist.forward(x, self.half).cast(DType::F32),
        )
    }
}
