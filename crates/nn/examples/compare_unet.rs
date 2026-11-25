//! # compare_unet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The burn U-Net against libtorch on romantic-shark's real weights: the
//! files `crates/ai`'s `unet_reference` writes (weights, normalized inputs,
//! libtorch's CPU/GPU outputs, the package's expected output) - burn on
//! burn-flex (f32) and the GPU (wgpu, f32 and f16 convolutions), each output
//! against each reference: max and mean absolute difference, and the pixels
//! whose foreground (channel 0 > 0.5) flips.
//!
//!   cargo build-linux -p evanalyzer_nn --example compare_unet
//!   target/x86_64-unknown-linux-gnu/release/examples/compare_unet <reference dir>

use burn::tensor::{Device, DeviceKind};
use evanalyzer_nn::model::unet::{FinalActivation, Norm, UNet, UNetConfig};
use std::path::Path;

fn read_f32(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Max and mean |a - b|, and the pixels of channel 0 on different sides of
/// 0.5.
fn compare(a: &[f32], b: &[f32], pixels: usize) -> String {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum) = (0.0f32, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += d as f64;
    }
    let flips = a[..pixels]
        .iter()
        .zip(&b[..pixels])
        .filter(|(x, y)| (**x > 0.5) != (**y > 0.5))
        .count();
    format!(
        "max {max:.2e}  mean {:.2e}  fg flips {flips} of {pixels}",
        sum / a.len() as f64
    )
}

fn main() {
    let dir = std::env::args().nth(1).expect("reference dir");
    let dir = Path::new(&dir);
    let weights = std::fs::read(dir.join("weights.safetensors")).unwrap();
    let config = UNetConfig {
        in_channels: 1,
        out_channels: 2,
        depth: 4,
        initial_features: 64,
        gain: 2,
        norm: Norm::Instance,
        final_activation: FinalActivation::Sigmoid,
    };
    let runs = [
        ("burn-flex f32", Device::flex(), false),
        (
            "burn-gpu f32",
            Device::wgpu(DeviceKind::DiscreteGpu(0)),
            false,
        ),
        (
            "burn-gpu f16",
            Device::wgpu(DeviceKind::DiscreteGpu(0)),
            true,
        ),
    ];
    for (input, side) in [("test256", 256), ("cells1024", 1024)] {
        let x = read_f32(&dir.join(format!("{input}.input.f32"))).unwrap();
        let mut references: Vec<(String, Vec<f32>)> = ["libtorch-cpu", "libtorch-gpu", "expected"]
            .iter()
            .filter_map(|r| {
                read_f32(&dir.join(format!("{input}.{r}.f32"))).map(|v| (r.to_string(), v))
            })
            .collect();
        println!("== {input} ({side}²)");
        if let [(a, x), (b, y), ..] = references.as_slice() {
            println!("  {a:<14} vs {b:<14} {}", compare(x, y, side * side));
        }
        if references.len() > 2 {
            let (c, z) = &references[2];
            let (a, x) = &references[0];
            println!("  {a:<14} vs {c:<14} {}", compare(x, z, side * side));
        }
        for (name, device, half) in &runs {
            let mut model = UNet::with_torch_em_weights(config.clone(), &weights, device).unwrap();
            model.set_half_convs(*half);
            let y = model.forward_raw(&x, side, side, device);
            for (r, v) in &references {
                println!("  {name:<14} vs {r:<14} {}", compare(&y, v, side * side));
            }
            references.push((name.to_string(), y));
        }
    }
}
