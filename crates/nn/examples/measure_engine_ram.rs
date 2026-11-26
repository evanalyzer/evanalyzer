//! # measure_engine_ram
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The AI engine's own RAM - what the job planner reserves once for a job
//! with an AI step (`AI_ENGINE_BYTES` in `evanalyzer_core::memory::analysis`):
//! a tiny U-Net (2 levels, 4 filters - weights and activations of a few KB)
//! through [`evanalyzer_nn::run`] on a 64² image, so what the process
//! grows by is the engine: its thread pool, burn's backend, on a GPU the
//! graphics driver and the device. Resident memory (`VmRSS`) and its peak
//! (`VmHWM`) before, after the first run and after nine more, in MB. One
//! device per process (the engine is set up once):
//!
//!   cargo build-linux -p evanalyzer_nn --example measure_engine_ram
//!   target/x86_64-unknown-linux-gnu/release/examples/measure_engine_ram cpu|gpu

use evanalyzer_nn::model::unet::{Preprocessing, UNet, UNetParams};
use evanalyzer_nn::test_support::{tiny_unet_config, unet_file};
use evanalyzer_nn::{DeviceSelection, EngineConfig, Image, Pieces};

/// A field of `/proc/self/status` in MB.
fn status_mb(field: &str) -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with(field)).unwrap();
    let kb: f64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb / 1024.0
}

fn main() {
    let devices = match std::env::args().nth(1).as_deref() {
        Some("cpu") => DeviceSelection::Cpu,
        Some("gpu") => DeviceSelection::Auto,
        _ => panic!("usage: measure_engine_ram cpu|gpu"),
    };
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tiny.safetensors");
    std::fs::write(&file, unet_file(tiny_unet_config(), Preprocessing::None)).unwrap();
    let pixels: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32 / 17.0).collect();
    let image = Image {
        width: 64,
        height: 64,
        channels: 1,
        data: &pixels,
    };
    let settings = UNetParams {
        normalization: evanalyzer_nn::Normalization::ModelDefault,
        foreground_channel: 0,
        softmax: false,
        boundary_channel: None,
        pieces: Pieces::fixed(2048),
        probability_threshold: 0.5,
        boundary_threshold: 0.5,
        foreground: 1,
    };
    let mut mask = vec![0u32; 64 * 64];
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let config = || EngineConfig {
        devices: devices.clone(),
        gpu_slots: 1,
        cores,
    };

    let before = status_mb("VmRSS:");
    evanalyzer_nn::run::<UNet>(&file, image, &settings, &mut mask, config).unwrap();
    let first = (status_mb("VmRSS:"), status_mb("VmHWM:"));
    for _ in 0..9 {
        evanalyzer_nn::run::<UNet>(&file, image, &settings, &mut mask, config).unwrap();
    }
    let later = (status_mb("VmRSS:"), status_mb("VmHWM:"));
    let devices: Vec<String> = evanalyzer_nn::engine_info()
        .map(|info| info.devices.into_iter().map(|d| d.name).collect())
        .unwrap_or_default();
    println!(
        "devices {devices:?}, {cores} cores: RSS before {before:.1} MB; after the first run +{:.1} MB (peak +{:.1}); after ten runs +{:.1} MB (peak +{:.1})",
        first.0 - before,
        first.1 - before,
        later.0 - before,
        later.1 - before
    );
}
