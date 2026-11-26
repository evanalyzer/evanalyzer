//! # bench_cellpose_sam
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Cellpose-SAM's speed per image and how far its output is from the
//! references: `CellposeSam::infer` (tiles, blending, as the step runs it)
//! `repeats` times in one process on one of the reference cases of
//! `tests/cellpose-sam` (`cells300x200-replicate`: 2 tiles,
//! `cells1024-replicate`: 25 tiles), on the CPU (`flex`) or the discrete
//! GPU (`dgpu`), its matrix products in f32 or f16. Per image the wall
//! time; for the first image the largest difference to the f32 network's
//! output (`reference-f32`) and to the shipped bf16-rounded one
//! (`reference`), and the pixels whose probability lands on the other side
//! of 0.5. GPU use: run `nvidia-smi --query-gpu=utilization.gpu,memory.used
//! --format=csv -lms 200` beside it.
//!
//!   cargo build-linux -p evanalyzer_nn --example bench_cellpose_sam
//!   target/x86_64-unknown-linux-gnu/release/examples/bench_cellpose_sam \
//!     tests/cellpose-sam cells300x200-replicate 6 dgpu f16 [cache folder]

use burn::tensor::{Device, DeviceKind};
use evanalyzer_nn::model::Image;
use evanalyzer_nn::model::cellpose_sam::{CellposeSam, CellposeSamParams};
use evanalyzer_nn::{Model, Normalization};
use std::path::Path;
use std::time::Instant;

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("tests/cellpose-sam"));
    let case = args.get(2).map_or("cells300x200-replicate", String::as_str);
    let repeats: usize = args.get(3).map_or(6, |s| s.parse().unwrap());
    let device_name = args.get(4).map_or("dgpu", String::as_str);
    let half = args.get(5).map_or("f32", String::as_str) == "f16";
    // The GPU's disk cache (autotune results, compiled kernels), if given.
    evanalyzer_nn::configure_gpu(args.get(6).map(Path::new));
    let (w, h, replicate) = match case {
        "cells300x200-replicate" => (300, 200, true),
        "cells1024-replicate" => (1024, 1024, true),
        "cells256-zero" => (256, 256, false),
        other => panic!("unknown case {other}"),
    };
    let device = match device_name {
        "flex" => Device::flex(),
        "dgpu" => Device::wgpu(DeviceKind::DiscreteGpu(0)),
        other => panic!("unknown device {other}"),
    };
    let start = Instant::now();
    // As the engine loads it: f16 when asked for (`load`'s GPU path).
    let net = CellposeSam::load(&dir.join("cpsam_v2.safetensors"), &device, half).unwrap();
    eprintln!("loaded in {:.1} s", start.elapsed().as_secs_f64());

    let input = read_f32(&dir.join("reference-f32").join(format!("{case}.input.f32")));
    // The reference inputs were run as they are: no normalization.
    let settings = CellposeSamParams {
        normalization: Normalization::None,
        net_width: w,
        net_height: h,
        input_channels: 3,
        replicate_gray_channel: replicate,
        probability_threshold: 0.5,
        flow_iterations: 200,
        flow_threshold: 0.4,
        min_object_size: 15,
        cellpose_dynamics: true,
    };
    let image = Image {
        width: w,
        height: h,
        channels: 1,
        data: &input,
    };
    let mut times = Vec::new();
    for i in 0..repeats {
        let start = Instant::now();
        let (input, _) = CellposeSam::pre(&net.to_header(), image, &settings).unwrap();
        let out = net.infer(&input).unwrap();
        let secs = start.elapsed().as_secs_f64();
        times.push(secs);
        eprintln!("image {}: {secs:.2} s", i + 1);
        if i == 0 {
            let ours = [out.flow_y, out.flow_x, out.cell_prob].concat();
            let n = w * h;
            for reference in ["reference-f32", "reference"] {
                let engine = read_f32(&dir.join(reference).join(format!("{case}.engine.f32")));
                let max = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max)
                };
                let flips = ours[2 * n..]
                    .iter()
                    .zip(&engine[2 * n..])
                    .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                    .count();
                eprintln!(
                    "  vs {reference}: flows max |diff| {:.2e}, probability {:.2e}, flips {flips} of {n}",
                    max(&ours[..2 * n], &engine[..2 * n]),
                    max(&ours[2 * n..], &engine[2 * n..]),
                );
            }
        }
    }
    let later = &times[1.min(times.len() - 1)..];
    let mut sorted = later.to_vec();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{case} {device_name} {}: first {:.2} s, later median {:.2} s (min {:.2})",
        if half { "f16" } else { "f32" },
        times[0],
        sorted[sorted.len() / 2],
        sorted[0]
    );
}
