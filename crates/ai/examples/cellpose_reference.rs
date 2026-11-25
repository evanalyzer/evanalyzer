//! # cellpose_reference
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The burn migration's comparison against libtorch for Cellpose-SAM: the
//! engine as the app ran it (`evanalyzer_ai::infer`, `Task::Cellpose`, CPU)
//! with a Cellpose-SAM TorchScript (`docs/convert_cellpose.py --channels 3`)
//! on synthetic cell images - raw little-endian f32 files per case:
//! `<case>.input.f32` (the image) and `<case>.engine.f32` (dY, dX, cell
//! probability planes at the net size), and each case's time.
//!
//!   cargo build-linux -p evanalyzer_ai --features test-support --example cellpose_reference
//!   LD_LIBRARY_PATH=$PWD/libs/libtorch/lib:$PWD/libs/nvidia-cu13/lib \
//!     target/x86_64-unknown-linux-gnu/release/examples/cellpose_reference <torchscript> <out dir> [threads]

use evanalyzer_ai::{DeviceSelection, EngineConfig, Image, Request, Response, Task};
use std::path::Path;
use std::time::Instant;

/// Bright discs of varying size on a textured background (0..~1).
pub fn cells(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            let (cx, cy) = ((x / 37.0).floor(), (y / 41.0).floor());
            let (dx, dy) = (x - cx * 37.0 - 18.0, y - cy * 41.0 - 20.0);
            let radius = 8.0 + (cx * 3.0 + cy * 5.0) % 9.0;
            let cell = if dx * dx + dy * dy < radius * radius {
                0.6 + ((cx + cy) % 4.0) * 0.1
            } else {
                0.05
            };
            cell + 0.04 * (x * 0.31).sin() * (y * 0.17).cos()
        })
        .collect()
}

fn write(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = Path::new(args.get(1).expect("TorchScript file"));
    let out = Path::new(args.get(2).expect("output dir"));
    let threads: usize = args.get(3).map_or(4, |t| t.parse().unwrap());
    std::fs::create_dir_all(out).unwrap();
    // (name, width, height, net width, net height, replicate)
    let cases = [
        ("cells256-zero", 256, 256, 256, 256, false),
        ("cells300x200-replicate", 300, 200, 300, 200, true),
        ("cells300x200-resized", 300, 200, 450, 300, false),
        ("cells1024-replicate", 1024, 1024, 1024, 1024, true),
    ];
    let config = || EngineConfig {
        devices: DeviceSelection::Cpu,
        gpu_slots: 1,
        cores: threads,
    };
    for (name, width, height, net_width, net_height, replicate) in cases {
        let data = cells(width, height);
        let request = Request {
            model: model.to_path_buf(),
            task: Task::Cellpose {
                net_width,
                net_height,
                input_channels: 3,
                replicate_gray_channel: replicate,
            },
        };
        let image = Image { width, height, channels: 1, data: &data };
        let start = Instant::now();
        let Response::Cellpose { flow_y, flow_x, cell_prob } =
            evanalyzer_ai::infer(&request, image, config).unwrap()
        else {
            panic!("not a Cellpose answer");
        };
        let secs = start.elapsed().as_secs_f64();
        write(&out.join(format!("{name}.input.f32")), &data);
        write(&out.join(format!("{name}.engine.f32")), &[flow_y, flow_x, cell_prob].concat());
        println!("{name}: {width}x{height} -> {net_width}x{net_height}, {secs:.2} s, {threads} threads");
    }
}
