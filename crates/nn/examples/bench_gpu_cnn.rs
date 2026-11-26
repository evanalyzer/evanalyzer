//! # bench_gpu_cnn
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! StarDist's and the U-Net's speed per image and their distance to the
//! references, as `bench_cellpose_sam` for Cellpose-SAM: `infer` (pieces,
//! padding, as the step runs it) `repeats` times in one process on
//! `tests/stardist/reference/cells1100x600` (against the official StarDist)
//! or `tests/romantic-shark/reference/cells1024` (against the libtorch
//! engine), or on a 2048² image tiled from them (`big`: speed only), on the
//! CPU (`flex`) or the discrete GPU (`dgpu`), convolutions in f32 or f16.
//!
//!   cargo build-linux -p evanalyzer_nn --example bench_gpu_cnn
//!   target/x86_64-unknown-linux-gnu/release/examples/bench_gpu_cnn \
//!     stardist|unet tests ref|big 6 dgpu f16 [cache folder] [piece side]

use burn::tensor::{Device, DeviceKind};
use evanalyzer_nn::model::Image;
use evanalyzer_nn::model::stardist::{StarDist, StarDistSettings};
use evanalyzer_nn::model::unet::{UNet, UNetSettings};
use evanalyzer_nn::{Model, Pieces};
use std::path::Path;
use std::time::Instant;

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `src` (`w` x `h`) repeated to `side`².
fn tiled(src: &[f32], w: usize, h: usize, side: usize) -> Vec<f32> {
    (0..side * side)
        .map(|i| src[((i / side) % h) * w + (i % side) % w])
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).map_or("stardist", String::as_str);
    let tests = Path::new(args.get(2).map_or("tests", String::as_str));
    let big = args.get(3).map_or("ref", String::as_str) == "big";
    let repeats: usize = args.get(4).map_or(6, |s| s.parse().unwrap());
    let device_name = args.get(5).map_or("dgpu", String::as_str);
    let half = args.get(6).map_or("f32", String::as_str) == "f16";
    evanalyzer_nn::configure_gpu(args.get(7).map(Path::new));
    let device = match device_name {
        "flex" => Device::flex(),
        "dgpu" => Device::wgpu(DeviceKind::DiscreteGpu(0)),
        other => panic!("unknown device {other}"),
    };
    let pieces = Pieces::fixed(args.get(8).map_or(2048, |s| s.parse().unwrap()));
    let mut times = Vec::new();
    match model {
        "stardist" => {
            let dir = tests.join("stardist");
            let r = dir.join("reference");
            let mut net =
                StarDist::load(&dir.join("2D_versatile_fluo.safetensors"), &device, false).unwrap();
            net.set_half_convs(half);
            let (w, h) = (1100, 600);
            let input = read_f32(&r.join("cells1100x600.input.f32"));
            let (input, w, h) = if big {
                (tiled(&input, w, h, 2048), 2048, 2048)
            } else {
                (input, w, h)
            };
            let image = Image {
                width: w,
                height: h,
                channels: 1,
                data: &input,
            };
            for i in 0..repeats {
                let start = Instant::now();
                let out = net.infer(image, &StarDistSettings { pieces }).unwrap();
                times.push(start.elapsed().as_secs_f64());
                if i == 0 && !big {
                    let expected = read_f32(&r.join("cells1100x600.output.f32"));
                    let shape: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(r.join("cells1100x600.shape.json")).unwrap(),
                    )
                    .unwrap();
                    let egw = shape["grid_width"].as_u64().unwrap() as usize;
                    let egh = shape["grid_height"].as_u64().unwrap() as usize;
                    let (gw, gh) = (out.grid_width, out.grid_height);
                    let (mut prob, mut dist, mut flips) = (0.0f32, 0.0f32, 0);
                    for plane in 0..=out.n_rays {
                        for y in 0..gh {
                            for x in 0..gw {
                                let theirs = expected[(plane * egh + y) * egw + x];
                                if plane == 0 {
                                    let ours = out.prob[y * gw + x];
                                    prob = prob.max((ours - theirs).abs());
                                    flips += usize::from((ours > 0.479) != (theirs > 0.479));
                                } else {
                                    let ours = out.dist[((plane - 1) * gh + y) * gw + x].max(1e-3);
                                    dist = dist.max((ours - theirs).abs());
                                }
                            }
                        }
                    }
                    eprintln!(
                        "  vs StarDist: probability {prob:.2e}, distances {dist:.2e}, flips {flips} of {}",
                        gw * gh
                    );
                }
            }
        }
        "unet" => {
            let dir = tests.join("romantic-shark");
            let r = dir.join("reference");
            let mut net =
                UNet::load(&dir.join("romantic-shark.safetensors"), &device, false).unwrap();
            net.set_half_convs(half);
            let side = 1024;
            let input = read_f32(&r.join("cells1024.input.f32"));
            let (input, side) = if big {
                (tiled(&input, side, side, 2048), 2048)
            } else {
                (input, side)
            };
            let image = Image {
                width: side,
                height: side,
                channels: 1,
                data: &input,
            };
            let settings = UNetSettings {
                foreground_channel: 0,
                softmax: false,
                boundary_channel: Some(1),
                pieces,
            };
            for i in 0..repeats {
                let start = Instant::now();
                let out = net.infer(image, &settings).unwrap();
                times.push(start.elapsed().as_secs_f64());
                if i == 0 && !big {
                    let engine = read_f32(&r.join("cells1024.engine.f32"));
                    let mut ours = out.probabilities;
                    ours.extend(out.boundary.unwrap());
                    let n = side * side;
                    let flips = ours[..n]
                        .iter()
                        .zip(&engine[..n])
                        .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                        .count();
                    eprintln!(
                        "  vs libtorch engine: max {:.2e}, flips {flips} of {n}",
                        max_diff(&ours, &engine)
                    );
                }
            }
        }
        other => panic!("unknown model {other}"),
    }
    let mut later = times[1.min(times.len() - 1)..].to_vec();
    later.sort_by(f64::total_cmp);
    println!(
        "{model} {} {device_name} {}: first {:.3} s, later median {:.3} s (min {:.3})",
        if big { "2048²" } else { "ref" },
        if half { "f16" } else { "f32" },
        times[0],
        later[later.len() / 2],
        later[0]
    );
}
