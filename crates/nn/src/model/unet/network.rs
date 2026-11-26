//! # network
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The U-Net network in burn, laid out as torch-em's `UNet2d` (MIT; the
//! architecture of bioimage.io's U-Nets, e.g. `romantic-shark`):
//!
//! - a block: norm → conv 3x3 → ReLU → norm → conv 3x3 → ReLU (the norm
//!   before each convolution, padding 1 - the size stays);
//! - encoder: `depth` blocks from `initial_features`, times `gain` per level,
//!   each followed by a 2x2 max pool;
//! - base: one block, times `gain` again;
//! - decoder: per level bilinear x2 (`align_corners = false`) + conv 1x1,
//!   then the encoder output of that level appended (centre-cropped) and a
//!   block;
//! - a conv 1x1 to `out_channels`, then the final activation.
//!
//! Input sides must be multiples of `2^depth` (16 at depth 4) - the caller
//! pads.

use super::{FinalActivation, Norm, UNetConfig};
use crate::errors::NnError;
use crate::weights::{NamedTensor, tensor_from};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, Initializer, InstanceNorm,
    InstanceNormConfig, PaddingConfig2d,
};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::module::{conv2d, interpolate};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{DType, Device, Tensor};
use safetensors::SafeTensors;

/// `conv` on `x` (stride 1, `padding` on every side).
///
/// When `half` (the GPU): `x`, the weights (converted once, `set_half`) and
/// the result in f16 - because burn's CubeCL implicit-GEMM
/// (tensor core) kernels take only f16/bf16 here (f32 runs a direct
/// convolution, 20-60x slower), and the bias added after it: with a bias
/// those kernels decline too (seen on wgpu, burn 0.22). Else `conv`'s own
/// forward - on burn-flex the bias inside the convolution saves a pass (a
/// separate add was 5 % slower).
fn conv_forward(conv: &Conv2d, x: Tensor<4>, padding: usize, half: bool) -> Tensor<4> {
    if !half {
        return conv.forward(x);
    }
    let options = ConvOptions::new([1, 1], [padding, padding], [1, 1], 1);
    let y = conv2d(x, conv.weight.val(), None, options);
    match &conv.bias {
        Some(bias) => {
            let channels = bias.dims()[0];
            y + bias.val().reshape([1, channels, 1, 1])
        }
        None => y,
    }
}

/// InstanceNorm without affine parameters, reduced over `[B, H*W, C]`:
/// CubeCL convolutions write channels-last, where these views are free and
/// the reductions run over many parallel lanes - burn's InstanceNorm (a
/// GroupNorm reshaped to `[B, C, H*W]`) was 3-4x slower on a convolution's
/// output (37 vs 11 ms for 64x1024², RTX 3000 Ada).
fn instance_norm_channels_last(x: Tensor<4>, epsilon: f64) -> Tensor<4> {
    let [b, c, h, w] = x.dims();
    let rows = x.permute([0, 2, 3, 1]).reshape([b, h * w, c]);
    let mean = rows.clone().mean_dim(1);
    let centred = rows - mean;
    let var = centred.clone().square().mean_dim(1);
    (centred / (var + epsilon).sqrt())
        .reshape([b, h, w, c])
        .permute([0, 3, 1, 2])
}

/// The norm layer of a block, as PyTorch builds it: InstanceNorm without
/// affine parameters (PyTorch's default), BatchNorm and GroupNorm with.
#[derive(Module, Debug)]
pub(crate) enum NormLayer {
    Instance(InstanceNorm),
    Batch(BatchNorm),
    Group(GroupNorm),
}

impl NormLayer {
    /// `None` for [`Norm::None`].
    fn new(norm: Norm, channels: usize, device: &Device) -> Option<Self> {
        // PyTorch's default epsilon, the same for all three.
        const EPSILON: f64 = 1e-5;
        match norm {
            Norm::None => None,
            Norm::Instance => Some(Self::Instance(
                InstanceNormConfig::new(channels)
                    .with_epsilon(EPSILON)
                    .with_affine(false)
                    .init(device),
            )),
            Norm::Batch => Some(Self::Batch(
                BatchNormConfig::new(channels)
                    .with_epsilon(EPSILON)
                    .init(device),
            )),
            // torch-em: `GroupNorm(min(n_groups, channels), channels)`.
            Norm::Group { groups } => Some(Self::Group(
                GroupNormConfig::new(groups.min(channels), channels)
                    .with_epsilon(EPSILON)
                    .init(device),
            )),
        }
    }

    /// `gpu`: the GPU path (see `conv_forward`, `instance_norm_channels_last`)
    /// - `x` f16 there, the statistics computed in f32, the result f16.
    fn forward(&self, x: Tensor<4>, gpu: bool) -> Tensor<4> {
        if gpu {
            let x = x.cast(DType::F32);
            let y = match self {
                Self::Instance(norm) => instance_norm_channels_last(x, norm.epsilon),
                Self::Batch(norm) => norm.forward(x),
                Self::Group(norm) => norm.forward(x),
            };
            return y.cast(DType::F16);
        }
        match self {
            Self::Instance(norm) => norm.forward(x),
            Self::Batch(norm) => norm.forward(x),
            Self::Group(norm) => norm.forward(x),
        }
    }
}

/// Two 3x3 convolutions with ReLU, each after its norm.
#[derive(Module, Debug)]
pub(crate) struct ConvBlock {
    pub(crate) norm1: Option<NormLayer>,
    pub(crate) conv1: Conv2d,
    pub(crate) norm2: Option<NormLayer>,
    pub(crate) conv2: Conv2d,
}

impl ConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        norm: Norm,
        init: &Option<Initializer>,
        device: &Device,
    ) -> Self {
        let conv = |from: usize, to: usize| {
            with_init(
                Conv2dConfig::new([from, to], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                init,
            )
            .init(device)
        };
        Self {
            norm1: NormLayer::new(norm, in_channels, device),
            conv1: conv(in_channels, out_channels),
            norm2: NormLayer::new(norm, out_channels, device),
            conv2: conv(out_channels, out_channels),
        }
    }

    fn forward(&self, mut x: Tensor<4>, half: bool) -> Tensor<4> {
        if let Some(norm) = &self.norm1 {
            x = norm.forward(x, half);
        }
        x = relu(conv_forward(&self.conv1, x, 1, half));
        if let Some(norm) = &self.norm2 {
            x = norm.forward(x, half);
        }
        relu(conv_forward(&self.conv2, x, 1, half))
    }
}

/// torch-em's `UNet2d`, see the module doc.
#[derive(Module, Debug)]
pub(crate) struct UNet2d {
    pub(crate) encoder: Vec<ConvBlock>,
    pub(crate) base: ConvBlock,
    /// The 1x1 convolution after each bilinear x2, deepest level first.
    pub(crate) samplers: Vec<Conv2d>,
    /// Deepest level first.
    pub(crate) decoder: Vec<ConvBlock>,
    pub(crate) out_conv: Conv2d,
    pool: MaxPool2d,
    #[module(skip)]
    final_activation: FinalActivation,
    /// The network in f16 (see `conv_forward`); set with [`Self::set_half`].
    #[module(skip)]
    half: bool,
}

/// `config` with `init` (`None`: burn's default, random).
fn with_init(config: Conv2dConfig, init: &Option<Initializer>) -> Conv2dConfig {
    match init {
        Some(init) => config.with_initializer(init.clone()),
        None => config,
    }
}

impl UNet2d {
    /// A network of `config`; its weights from `init` (`None`: random,
    /// `Some(Zeros)`: cheap placeholders the loader replaces).
    pub(crate) fn new(config: &UNetConfig, init: Option<Initializer>, device: &Device) -> Self {
        let level = |i: usize| config.initial_features * config.gain.pow(i as u32);
        // Encoder: in_channels → f, f → f·g, ...; base: f·g^(d-1) → f·g^d.
        let encoder = (0..config.depth)
            .map(|i| {
                let from = if i == 0 {
                    config.in_channels
                } else {
                    level(i - 1)
                };
                ConvBlock::new(from, level(i), config.norm, &init, device)
            })
            .collect();
        let base = ConvBlock::new(
            level(config.depth - 1),
            level(config.depth),
            config.norm,
            &init,
            device,
        );
        // Decoder, deepest first: f·g^(i+1) → f·g^i.
        let levels: Vec<usize> = (0..config.depth).rev().collect();
        let samplers = levels
            .iter()
            .map(|&i| {
                with_init(Conv2dConfig::new([level(i + 1), level(i)], [1, 1]), &init).init(device)
            })
            .collect();
        // After the concat: its own f·g^i plus the encoder's f·g^i.
        let decoder = levels
            .iter()
            .map(|&i| ConvBlock::new(2 * level(i), level(i), config.norm, &init, device))
            .collect();
        Self {
            encoder,
            base,
            samplers,
            decoder,
            out_conv: with_init(
                Conv2dConfig::new([level(0), config.out_channels], [1, 1]),
                &init,
            )
            .init(device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            final_activation: config.final_activation,
            half: false,
        }
    }

    /// The network in f16 (`true`, the GPU's tensor cores: convolution
    /// weights converted once, activations f16 from the input to the output
    /// convolution, norms computed in f32) or f32. Back to f32 keeps the f16
    /// rounding - set it once.
    pub(crate) fn set_half(&mut self, half: bool) {
        let dtype = if half { DType::F16 } else { DType::F32 };
        let blocks = self
            .encoder
            .iter_mut()
            .chain([&mut self.base])
            .chain(self.decoder.iter_mut());
        let mut convs: Vec<&mut Conv2d> = Vec::new();
        for block in blocks {
            convs.push(&mut block.conv1);
            convs.push(&mut block.conv2);
        }
        convs.extend(self.samplers.iter_mut());
        convs.push(&mut self.out_conv);
        for conv in convs {
            conv.weight = conv.weight.clone().map(|w| w.cast(dtype));
            conv.bias = conv.bias.clone().map(|b| b.map(|b| b.cast(dtype)));
        }
        self.half = half;
    }

    /// `[batch, in_channels, H, W]` → `[batch, out_channels, H, W]`, f32; H
    /// and W multiples of `2^depth`.
    pub(crate) fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        if self.half {
            x = x.cast(DType::F16);
        }
        let mut skips = Vec::with_capacity(self.encoder.len());
        for block in &self.encoder {
            x = block.forward(x, self.half);
            skips.push(x.clone());
            x = self.pool.forward(x);
        }
        x = self.base.forward(x, self.half);
        for ((sampler, block), skip) in self
            .samplers
            .iter()
            .zip(&self.decoder)
            .zip(skips.into_iter().rev())
        {
            let [_, _, h, w] = x.dims();
            x = interpolate(
                x,
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(false)
                    .with_output_size([2 * h, 2 * w]),
            );
            x = conv_forward(sampler, x, 0, self.half);
            x = block.forward(
                Tensor::cat(vec![x.clone(), centre_crop(skip, &x)], 1),
                self.half,
            );
        }
        x = conv_forward(&self.out_conv, x, 0, self.half).cast(DType::F32);
        match self.final_activation {
            FinalActivation::None => x,
            FinalActivation::Sigmoid => sigmoid(x),
            FinalActivation::Softmax => softmax(x, 1),
        }
    }
}

/// `conv`'s weight and bias from `<prefix>.weight` / `<prefix>.bias`.
fn load_conv(
    conv: &mut Conv2d,
    file: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<(), NnError> {
    let dims = conv.weight.dims();
    conv.weight = Param::from_tensor(tensor_from(
        file,
        &format!("{prefix}.weight"),
        dims,
        device,
    )?);
    if let Some(bias) = &mut conv.bias {
        let dims = bias.dims();
        *bias = Param::from_tensor(tensor_from(file, &format!("{prefix}.bias"), dims, device)?);
    }
    Ok(())
}

impl ConvBlock {
    /// torch-em's `Sequential(norm, conv, ReLU, norm, conv, ReLU)`: the
    /// convolutions are `block.1` and `block.4`. Only for InstanceNorm and
    /// no norm (no parameters) so far.
    fn load_torch_em(
        &mut self,
        file: &SafeTensors<'_>,
        prefix: &str,
        device: &Device,
    ) -> Result<(), NnError> {
        load_conv(&mut self.conv1, file, &format!("{prefix}.block.1"), device)?;
        load_conv(&mut self.conv2, file, &format!("{prefix}.block.4"), device)
    }
}

impl UNet2d {
    /// The weights of a torch-em `UNet2d` state dict (PyTorch's names,
    /// e.g. `encoder.blocks.0.block.1.weight`), every shape checked against
    /// this network's layout.
    pub(crate) fn load_torch_em(
        &mut self,
        file: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<(), NnError> {
        let has_norm_params = self
            .encoder
            .iter()
            .any(|b| matches!(b.norm1, Some(NormLayer::Batch(_) | NormLayer::Group(_))));
        if has_norm_params {
            return Err(NnError::UnsupportedArchitecture(
                "U-Nets with BatchNorm or GroupNorm can't be loaded yet".into(),
            ));
        }
        for (i, block) in self.encoder.iter_mut().enumerate() {
            block.load_torch_em(file, &format!("encoder.blocks.{i}"), device)?;
        }
        self.base.load_torch_em(file, "base", device)?;
        for (i, block) in self.decoder.iter_mut().enumerate() {
            block.load_torch_em(file, &format!("decoder.blocks.{i}"), device)?;
        }
        for (i, sampler) in self.samplers.iter_mut().enumerate() {
            load_conv(sampler, file, &format!("decoder.samplers.{i}.conv"), device)?;
        }
        load_conv(&mut self.out_conv, file, "out_conv", device)
    }
}

/// `conv`'s weight and bias as `<prefix>.weight` / `<prefix>.bias`.
fn export_conv(conv: &Conv2d, prefix: &str, out: &mut Vec<NamedTensor>) {
    let mut push = |name: String, t: Tensor<4>| {
        let shape = t.dims().to_vec();
        let values = t.into_data().try_to_vec::<f32>().expect("f32 weights");
        out.push(NamedTensor {
            name,
            shape,
            values,
        });
    };
    push(format!("{prefix}.weight"), conv.weight.val());
    if let Some(bias) = &conv.bias {
        let c = bias.dims()[0];
        push(format!("{prefix}.bias"), bias.val().reshape([1, 1, 1, c]));
        let last = out.last_mut().expect("just pushed");
        last.shape = vec![c];
    }
}

impl UNet2d {
    /// The weights with torch-em's names - what [`Self::load_torch_em`]
    /// reads (InstanceNorm / no norm: the norms have no parameters).
    pub(crate) fn torch_em_tensors(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        let block = |b: &ConvBlock, prefix: &str, out: &mut Vec<NamedTensor>| {
            export_conv(&b.conv1, &format!("{prefix}.block.1"), out);
            export_conv(&b.conv2, &format!("{prefix}.block.4"), out);
        };
        for (i, b) in self.encoder.iter().enumerate() {
            block(b, &format!("encoder.blocks.{i}"), &mut out);
        }
        block(&self.base, "base", &mut out);
        for (i, b) in self.decoder.iter().enumerate() {
            block(b, &format!("decoder.blocks.{i}"), &mut out);
        }
        for (i, s) in self.samplers.iter().enumerate() {
            export_conv(s, &format!("decoder.samplers.{i}.conv"), &mut out);
        }
        export_conv(&self.out_conv, "out_conv", &mut out);
        out
    }
}

/// `x` cropped around its centre to the height and width of `like`
/// (torch-em's `_crop`; a no-op on sides that are multiples of `2^depth`).
fn centre_crop(x: Tensor<4>, like: &Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    let [_, _, th, tw] = like.dims();
    if (h, w) == (th, tw) {
        return x;
    }
    x.narrow(2, (h - th) / 2, th).narrow(3, (w - tw) / 2, tw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(norm: Norm) -> UNetConfig {
        UNetConfig {
            in_channels: 1,
            out_channels: 2,
            depth: 4,
            initial_features: 8,
            gain: 2,
            norm,
            final_activation: FinalActivation::Sigmoid,
        }
    }

    /// The romantic-shark layout (`depth 4, initial_features 64, gain 2`):
    /// the same convolutions with the same shapes as its state dict.
    #[test]
    fn the_layout_is_torch_ems() {
        let device = Device::flex();
        let net = UNet2d::new(
            &UNetConfig {
                initial_features: 64,
                ..config(Norm::Instance)
            },
            Some(Initializer::Zeros),
            &device,
        );
        let dims = |conv: &Conv2d| conv.weight.dims();
        assert_eq!(dims(&net.encoder[0].conv1), [64, 1, 3, 3]);
        assert_eq!(dims(&net.encoder[3].conv2), [512, 512, 3, 3]);
        assert_eq!(dims(&net.base.conv1), [1024, 512, 3, 3]);
        assert_eq!(dims(&net.samplers[0]), [512, 1024, 1, 1]);
        assert_eq!(dims(&net.decoder[0].conv1), [512, 1024, 3, 3]);
        assert_eq!(dims(&net.decoder[3].conv2), [64, 64, 3, 3]);
        assert_eq!(dims(&net.out_conv), [2, 64, 1, 1]);
        // All of the state dict's 46 tensors: 115,766,792 bytes of f32 of
        // `weights.pt`'s 115,785,997 (the rest is the pickle).
        assert_eq!(net.num_params(), 28_941_698);
    }

    /// Exported with torch-em's names and loaded again: the same network.
    #[test]
    fn the_weights_load_as_they_were_exported() {
        let device = Device::flex();
        let net = UNet2d::new(&config(Norm::Instance), None, &device);
        let tensors = net.torch_em_tensors();
        // Depth 4: encoder 4 x 2 convs, base 2, decoder 4 x 2, samplers 4,
        // out 1 - each a weight and a bias.
        assert_eq!(tensors.len(), 2 * (4 * 2 + 2 + 4 * 2 + 4 + 1));
        let bytes: Vec<Vec<u8>> = tensors
            .iter()
            .map(|t| t.values.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect();
        let views: Vec<_> = tensors
            .iter()
            .zip(&bytes)
            .map(|(t, b)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    t.shape.clone(),
                    b,
                )
                .unwrap();
                (t.name.clone(), view)
            })
            .collect();
        let file = safetensors::serialize(views, None).unwrap();
        let file = SafeTensors::deserialize(&file).unwrap();
        let mut loaded = UNet2d::new(&config(Norm::Instance), Some(Initializer::Zeros), &device);
        loaded.load_torch_em(&file, &device).unwrap();
        let x = Tensor::<4>::random([1, 1, 32, 48], burn::tensor::Distribution::Default, &device);
        let (a, b) = (net.forward(x.clone()), loaded.forward(x));
        assert_eq!(
            a.into_data().try_to_vec::<f32>().unwrap(),
            b.into_data().try_to_vec::<f32>().unwrap()
        );
        // A tensor of the wrong shape is refused.
        let mut wrong = UNet2d::new(
            &UNetConfig {
                initial_features: 4,
                ..config(Norm::Instance)
            },
            Some(Initializer::Zeros),
            &device,
        );
        assert!(matches!(
            wrong.load_torch_em(&file, &device),
            Err(NnError::InvalidModelFile(_))
        ));
    }

    #[test]
    fn the_output_has_the_input_size_and_the_out_channels() {
        let device = Device::flex();
        for norm in [
            Norm::None,
            Norm::Instance,
            Norm::Batch,
            Norm::Group { groups: 4 },
        ] {
            let net = UNet2d::new(&config(norm), None, &device);
            let x = Tensor::<4>::ones([1, 1, 48, 64], &device);
            let y = net.forward(x);
            assert_eq!(y.dims(), [1, 2, 48, 64], "{norm:?}");
            let values = y.into_data();
            let values = values.as_slice::<f32>().unwrap();
            assert!(values.iter().all(|v| (0.0..=1.0).contains(v)), "sigmoid");
        }
    }
}
