//! # bench_ops
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The U-Net's single ops on one backend, to find where `bench_unet`'s time
//! goes: 3x3 convolution, InstanceNorm, ReLU, max pool, bilinear x2 and the
//! skip concatenation at the shapes of romantic-shark's first level (64
//! features) on a `side`² image.
//!
//!   cargo build-linux -p evanalyzer_nn --example bench_ops
//!   target/x86_64-unknown-linux-gnu/release/examples/bench_ops <side> [threads] [flex|wgpu|dgpu|igpu|vulkan]
//!
//! Each op runs once to warm up (kernel compilation, autotune), then
//! `ROUNDS` times; the time per run includes summing the output and reading
//! the sum back (forces the whole op, drains the GPU's queue).

use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::{InstanceNormConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::module::interpolate;
use burn::tensor::ops::PadMode;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{DType, Device, DeviceKind, Distribution, Tensor};
use std::time::Instant;

const ROUNDS: usize = 5;

/// A 3x3 convolution (padding 1, no bias) as 9 matrix products: the
/// weights of each tap `[out, in]` times the input shifted by that tap
/// `[in, H*W]`, summed.
fn conv3x3_shifted(x: Tensor<4>, weight: &Tensor<4>) -> Tensor<4> {
    let [_, c, h, w] = x.dims();
    let [out, _, _, _] = weight.dims();
    let padded = x.pad((1, 1, 1, 1), PadMode::Constant(0.0));
    let mut sum: Option<Tensor<2>> = None;
    for ky in 0..3 {
        for kx in 0..3 {
            let tap = weight
                .clone()
                .slice([0..out, 0..c, ky..ky + 1, kx..kx + 1])
                .reshape([out, c]);
            let shifted = padded
                .clone()
                .narrow(2, ky, h)
                .narrow(3, kx, w)
                .reshape([c, h * w]);
            let product = tap.matmul(shifted);
            sum = Some(match sum {
                Some(sum) => sum + product,
                None => product,
            });
        }
    }
    sum.unwrap().reshape([1, out, h, w])
}

/// Milliseconds per run of `op`: one warm-up, then `ROUNDS` runs, each
/// waited for through one value read back.
fn time(name: &str, flops: f64, mut op: impl FnMut() -> Tensor<4>) {
    let wait = |t: Tensor<4>| {
        // `into_data` alone doesn't wait: the bytes arrive on first access.
        let _ = t.sum().into_data().as_slice::<f32>().unwrap()[0];
    };
    wait(op());
    let start = Instant::now();
    for _ in 0..ROUNDS {
        wait(op());
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;
    let gflops = if flops > 0.0 {
        format!(", {:.0} GFLOP/s", flops / ms / 1e6)
    } else {
        String::new()
    };
    println!("{name:<32} {ms:>9.2} ms{gflops}");
}

/// InstanceNorm (no affine, eps 1e-5) with two-level reductions: each
/// channel's plane split into `chunks` rows, mean over the rows' means -
/// more parallel work for the GPU than one reduction per channel.
fn instance_norm_two_level(x: Tensor<4>, chunks: usize) -> Tensor<4> {
    let [b, c, h, w] = x.dims();
    let n = h * w;
    let chunks = if n.is_multiple_of(chunks) { chunks } else { 1 };
    let rows = x.reshape([b * c, chunks, n / chunks]);
    let mean = rows.clone().mean_dim(2).mean_dim(1);
    let centred = rows - mean;
    let var = centred.clone().square().mean_dim(2).mean_dim(1);
    (centred / (var + 1e-5).sqrt()).reshape([b, c, h, w])
}

/// InstanceNorm (no affine, eps 1e-5) reduced over `[B, H*W, C]` - free
/// views when the tensor is stored channels-last (as convolution outputs
/// on CubeCL are, it seems).
fn instance_norm_channels_last(x: Tensor<4>) -> Tensor<4> {
    let [b, c, h, w] = x.dims();
    let rows = x.permute([0, 2, 3, 1]).reshape([b, h * w, c]);
    let mean = rows.clone().mean_dim(1);
    let centred = rows - mean;
    let var = centred.clone().square().mean_dim(1);
    (centred / (var + 1e-5).sqrt())
        .reshape([b, h, w, c])
        .permute([0, 3, 1, 2])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args.get(1).expect("side").parse().unwrap();
    let threads: usize = args.get(2).map_or_else(
        || std::thread::available_parallelism().unwrap().get(),
        |t| t.parse().unwrap(),
    );
    let device = match args.get(3).map_or("flex", String::as_str) {
        "flex" => Device::flex(),
        "wgpu" => Device::wgpu(DeviceKind::default()),
        "dgpu" => Device::wgpu(DeviceKind::DiscreteGpu(0)),
        "igpu" => Device::wgpu(DeviceKind::IntegratedGpu(0)),
        "vulkan" => Device::vulkan(DeviceKind::DiscreteGpu(0)),
        other => panic!("unknown device {other}"),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let c = 64;
        let x = Tensor::<4>::random([1, c, side, side], Distribution::Default, &device);
        let pixels = (side * side) as f64;
        let conv = |from: usize, to: usize| {
            Conv2dConfig::new([from, to], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(&device)
        };
        let conv_flops = |from: usize, to: usize, px: f64| 2.0 * (from * to * 9) as f64 * px;

        let first = conv(1, c);
        let x1 = x.clone().slice([0..1, 0..1, 0..side, 0..side]);
        time("conv 3x3 1->64", conv_flops(1, c, pixels), || {
            first.forward(x1.clone())
        });
        let same = conv(c, c);
        time("conv 3x3 64->64", conv_flops(c, c, pixels), || {
            same.forward(x.clone())
        });
        let weight = same.weight.val();
        time(
            "conv 3x3 64->64 shifted matmul",
            conv_flops(c, c, pixels),
            || conv3x3_shifted(x.clone(), &weight),
        );
        let (x16, w16) = (x.clone().cast(DType::F16), weight.clone().cast(DType::F16));
        let options = burn::tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        time("conv 3x3 64->64 f16", conv_flops(c, c, pixels), || {
            burn::tensor::module::conv2d(x16.clone(), w16.clone(), None, options.clone())
                .cast(DType::F32)
        });
        let skip = Tensor::cat(vec![x.clone(), x.clone()], 1);
        let decoder = conv(2 * c, c);
        time("conv 3x3 128->64", conv_flops(2 * c, c, pixels), || {
            decoder.forward(skip.clone())
        });
        let half = side / 2;
        let xh = Tensor::<4>::random([1, 2 * c, half, half], Distribution::Default, &device);
        let level1 = conv(2 * c, 2 * c);
        time(
            "conv 3x3 128->128 (side/2)",
            conv_flops(2 * c, 2 * c, pixels / 4.0),
            || level1.forward(xh.clone()),
        );
        let eighth = side / 16;
        let xb = Tensor::<4>::random([1, 16 * c, eighth, eighth], Distribution::Default, &device);
        let base = conv(16 * c, 16 * c);
        time(
            "conv 3x3 1024->1024 (side/16)",
            conv_flops(16 * c, 16 * c, pixels / 256.0),
            || base.forward(xb.clone()),
        );
        let wb = base.weight.val();
        time(
            "conv 3x3 1024->1024 shifted",
            conv_flops(16 * c, 16 * c, pixels / 256.0),
            || conv3x3_shifted(xb.clone(), &wb),
        );
        let (xb16, wb16) = (xb.clone().cast(DType::F16), wb.clone().cast(DType::F16));
        time(
            "conv 3x3 1024->1024 f16",
            conv_flops(16 * c, 16 * c, pixels / 256.0),
            || {
                burn::tensor::module::conv2d(xb16.clone(), wb16.clone(), None, options.clone())
                    .cast(DType::F32)
            },
        );
        let one = Conv2dConfig::new([2 * c, c], [1, 1]).init(&device);
        let xu = Tensor::<4>::random([1, 2 * c, side, side], Distribution::Default, &device);
        time(
            "conv 1x1 128->64",
            2.0 * (2 * c * c) as f64 * pixels,
            || one.forward(xu.clone()),
        );
        let norm = InstanceNormConfig::new(c).with_affine(false).init(&device);
        time("instance norm 64", 0.0, || norm.forward(x.clone()));
        // The norm on what a block gives it: a conv's (f16, then f32 + ReLU)
        // output, maybe in another memory layout.
        let conv_out = relu(
            burn::tensor::module::conv2d(x16.clone(), w16.clone(), None, options.clone())
                .cast(DType::F32),
        );
        let _ = conv_out
            .clone()
            .sum()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        time("instance norm 64 on f16 conv out", 0.0, || {
            norm.forward(conv_out.clone())
        });
        let conv32 = relu(same.forward(x.clone()));
        let _ = conv32.clone().sum().into_data().as_slice::<f32>().unwrap()[0];
        time("instance norm 64 on f32 conv out", 0.0, || {
            norm.forward(conv32.clone())
        });
        time("channels-last norm on conv out", 0.0, || {
            instance_norm_channels_last(conv_out.clone())
        });
        time("channels-last norm on plain", 0.0, || {
            instance_norm_channels_last(x.clone())
        });
        {
            let a = norm.forward(conv_out.clone());
            let b = instance_norm_channels_last(conv_out.clone());
            let d = (a - b).abs().max().into_data();
            println!(
                "  max |burn norm - channels-last norm| = {:e}",
                d.as_slice::<f32>().unwrap()[0]
            );
        }
        time("relu on f16 conv out", 0.0, || relu(conv_out.clone()));
        for chunks in [64, 256, 1024] {
            time(&format!("instance norm 64 two-level {chunks}"), 0.0, || {
                instance_norm_two_level(x.clone(), chunks)
            });
        }
        time("relu 64", 0.0, || relu(x.clone()));
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        time("max pool 2x2 64", 0.0, || pool2.forward(x.clone()));
        let xs = Tensor::<4>::random([1, 2 * c, half, half], Distribution::Default, &device);
        time("bilinear x2 128 (side/2)", 0.0, || {
            interpolate(
                xs.clone(),
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(false)
                    .with_output_size([side, side]),
            )
        });
        time("cat 64+64", 0.0, || {
            Tensor::cat(vec![x.clone(), x.clone()], 1)
        });
        let _ = first.num_params();
    });
}
