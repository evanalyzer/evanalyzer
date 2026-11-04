//! # bench_ai_ram
//!
//! The RAM one model inference takes on a tile (TODO "AI models"): a
//! stand-in model of a real network's layout (`test_support`'s
//! `bench_stardist_model` / `bench_unet_model`) through the engine
//! (`evanalyzer_ai::infer`, libtorch linked in) on the CPU - first on a
//! small image (loads the model), then the peak resident memory of one
//! inference on a `side`² tile above what the process had before it (Linux:
//! `VmHWM` reset through `/proc/self/clear_refs`).
//!
//!   LD_LIBRARY_PATH=$PWD/libs/libtorch/lib cargo run --release -p evanalyzer_ai \
//!     --features test-support --example bench_ai_ram -- <stardist|unet32|unet64|unet-file=PATH> <side> [threads] [piece side]
//!
//! `unet-file=PATH`: a real U-Net's TorchScript file (e.g. bioimage.io's
//! `joyful-pig`), foreground channel 0, boundary channel 1.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_ai::torch::test_support::{bench_stardist_model, bench_unet_model};
use evanalyzer_ai::{DeviceSelection, EngineConfig, Image, Request, Task, infer};
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

/// Cells (bright squares) on a dim background.
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
    let kind = args
        .get(1)
        .expect("stardist | unet32 | unet64 | unet-file=PATH")
        .as_str();
    let side: usize = args.get(2).expect("side").parse().unwrap();
    let threads: usize = args.get(3).map_or_else(
        || std::thread::available_parallelism().unwrap().get(),
        |t| t.parse().unwrap(),
    );
    // The pieces of a big computer (`evanalyzer_ai::piece_side_for`).
    let piece_side: usize = args.get(4).map_or(2048, |p| p.parse().unwrap());
    let unet = Task::UNet {
        foreground_channel: 0,
        softmax: false,
        boundary_channel: 1,
        piece_side,
    };
    let ((_dir, model), task) = match kind {
        file if file.starts_with("unet-file=") => (
            (
                tempfile::tempdir().unwrap(),
                file["unet-file=".len()..].into(),
            ),
            unet,
        ),
        "stardist" => (bench_stardist_model(), Task::StarDist { piece_side }),
        "unet32" | "unet64" => (bench_unet_model(kind[4..].parse().unwrap()), unet),
        other => panic!("unknown model {other}"),
    };
    let request = Request { model, task };
    let config = || EngineConfig {
        devices: DeviceSelection::Cpu,
        gpu_slots: 1,
        cores: threads,
    };
    let run = |side: usize| {
        let data = image(side);
        let image = Image {
            width: side,
            height: side,
            channels: 1,
            data: &data,
        };
        infer(&request, image, config).unwrap();
    };

    // Loads the model.
    run(64);
    let before = status_mb("VmRSS:");
    for round in 1..=2 {
        let _ = std::fs::write("/proc/self/clear_refs", "5");
        let (start, cpu) = (Instant::now(), cpu_seconds());
        run(side);
        println!(
            "AI {kind} {side}² round {round}: inference +{:.0} MB peak over {before:.0} MB, \
             wall {:.2} s, cpu {:.2} s, {threads} threads",
            status_mb("VmHWM:") - before,
            start.elapsed().as_secs_f64(),
            cpu_seconds() - cpu,
        );
    }
}
