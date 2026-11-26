//! # bench_unet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! P0 of the burn migration: the U-Net (torch-em `UNet2d`, InstanceNorm,
//! depth 4, gain 2 - romantic-shark's layout) in burn with random weights,
//! one inference on a `side`² image - the same image and measurement as
//! the removed libtorch engine's `bench_ai_ram` (`unet-file=` romantic-shark's
//! TorchScript, in git history before 2026-11-25), to compare against it:
//!
//!   cargo build-linux -p evanalyzer_nn --example bench_unet
//!   target/x86_64-unknown-linux-gnu/release/examples/bench_unet <features> <side> [threads] [flex|wgpu|dgpu|igpu] [f32|f16]
//!
//! First a 64² run (sets the backend up), then two rounds on `side`²: wall
//! time, CPU time, peak resident memory above what the process had before
//! (Linux: `VmHWM` reset through `/proc/self/clear_refs`).

use burn::tensor::{Device, DeviceKind};
use evanalyzer_nn::model::unet::{FinalActivation, Norm, UNet, UNetConfig};
use std::time::Instant;

/// A field of `/proc/self/status` in MB.
fn status_mb(field: &str) -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with(field)).unwrap();
    let kb: f64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb / 1024.0
}

/// CPU seconds of the process so far.
fn cpu_seconds() -> f64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    t(usage.ru_utime) + t(usage.ru_stime)
}

/// Cells (bright squares) on a dim background - as `bench_ai_ram`.
fn image(side: usize) -> Vec<f32> {
    (0..side * side)
        .map(|i| {
            let (x, y) = (i % side, i / side);
            let cell = (4..28).contains(&(x % 32)) && (4..28).contains(&(y % 32));
            if cell { 0.8 } else { 0.1 }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let features: usize = args
        .get(1)
        .expect("initial features (32 | 64)")
        .parse()
        .unwrap();
    let side: usize = args.get(2).expect("side").parse().unwrap();
    let threads: usize = args.get(3).map_or_else(
        || std::thread::available_parallelism().unwrap().get(),
        |t| t.parse().unwrap(),
    );
    let device = match args.get(4).map_or("flex", String::as_str) {
        "flex" => Device::flex(),
        "wgpu" => Device::wgpu(DeviceKind::default()),
        "dgpu" => Device::wgpu(DeviceKind::DiscreteGpu(0)),
        "igpu" => Device::wgpu(DeviceKind::IntegratedGpu(0)),
        other => panic!("unknown device {other}"),
    };
    // Convolutions in f16 (GPU tensor cores).
    let half = match args.get(5).map_or("f32", String::as_str) {
        "f32" => false,
        "f16" => true,
        other => panic!("unknown precision {other} (f32 | f16)"),
    };
    assert!(side.is_multiple_of(16), "the side must be a multiple of 16");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let config = UNetConfig {
            in_channels: 1,
            out_channels: 2,
            depth: 4,
            initial_features: features,
            gain: 2,
            norm: Norm::Instance,
            final_activation: FinalActivation::Sigmoid,
        };
        let mut model = UNet::with_random_weights(config, &device);
        model.set_half_convs(half);
        // Sets the backend up.
        model.forward_raw(&image(64), 64, 64, &device);
        let data = image(side);
        let before = status_mb("VmRSS:");
        let precision = if half { "f16" } else { "f32" };
        for round in 1..=2 {
            let _ = std::fs::write("/proc/self/clear_refs", "5");
            let (start, cpu) = (Instant::now(), cpu_seconds());
            let out = model.forward_raw(&data, side, side, &device);
            let (wall, cpu) = (start.elapsed().as_secs_f64(), cpu_seconds() - cpu);
            assert_eq!(out.len(), 2 * side * side);
            println!(
                "burn unet{features} {side}² {threads} threads {precision} round {round}: {wall:.2} s wall, {cpu:.1} s CPU, peak +{:.0} MB",
                status_mb("VmHWM:") - before
            );
        }
    });
}
