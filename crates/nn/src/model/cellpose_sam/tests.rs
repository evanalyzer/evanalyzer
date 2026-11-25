//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;
use crate::weights::{ModelMetadata, NamedTensor};

use crate::test_support::{cellpose_sam_file, tiny_cellpose_sam_config as tiny_config};

fn tiny_model() -> CellposeSam {
    CellposeSam::from_bytes(&cellpose_sam_file(&tiny_config()), &Device::flex()).unwrap()
}

fn cells(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            let (dx, dy) = (x % 13.0 - 6.0, y % 11.0 - 5.0);
            if dx * dx + dy * dy < 16.0 { 0.9 } else { 0.1 }
        })
        .collect()
}

fn settings(w: usize, h: usize) -> CellposeSamSettings {
    CellposeSamSettings {
        net_width: w,
        net_height: h,
        input_channels: 3,
        replicate_gray_channel: false,
    }
}

fn image(data: &[f32], width: usize, height: usize) -> Image<'_> {
    Image {
        width,
        height,
        channels: 1,
        data,
    }
}

#[test]
fn tiles_cover_the_side_as_cellpose_makes_them() {
    assert_eq!(tile_starts(100, 256), [0]);
    assert_eq!(tile_starts(256, 256), [0]);
    // 1024: ceil(1.2 * 1024 / 256) = 5 tiles over 768 px.
    assert_eq!(tile_starts(1024, 256), [0, 192, 384, 576, 768]);
    let starts = tile_starts(300, 256);
    assert_eq!((starts.first(), starts.last()), (Some(&0), Some(&44)));
    let mask = taper_mask(256);
    assert!(mask[128 * 256 + 128] > 0.99 && mask[0] > 0.0 && mask[0] < 0.01);
}

/// One tile (an image of the tile size): the network's output, the cell
/// logit through the sigmoid - the taper cancels out.
#[test]
fn one_tile_is_the_network_output() {
    let net = tiny_model();
    let pixels = cells(32, 32);
    let out = net
        .infer(image(&pixels, 32, 32), &settings(32, 32))
        .unwrap();
    let mut tile = vec![0.0; 3 * 32 * 32];
    tile[..32 * 32].copy_from_slice(&pixels);
    let raw = net.forward_tile(&tile, 3);
    assert_eq!(raw.len(), 3 * 32 * 32);
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(close(&out.flow_y, &raw[..1024]));
    assert!(close(&out.flow_x, &raw[1024..2048]));
    let prob: Vec<f32> = raw[2048..]
        .iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    assert!(close(&out.cell_prob, &prob));
}

/// Zero-filled channels add nothing (the patch embedding has no bias per
/// channel): any channel count gives the same; replicating changes it.
#[test]
fn zero_filled_channels_add_nothing_and_replicated_ones_do() {
    let net = tiny_model();
    let pixels = cells(32, 32);
    let run = |input_channels, replicate_gray_channel| {
        let s = CellposeSamSettings {
            input_channels,
            replicate_gray_channel,
            ..settings(32, 32)
        };
        net.infer(image(&pixels, 32, 32), &s).unwrap()
    };
    let one = run(1, false);
    let three = run(3, false);
    let d = one
        .flow_y
        .iter()
        .zip(&three.flow_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(d < 1e-5, "{d}");
    assert_ne!(run(3, true), three);
    assert_eq!(
        run(9, true),
        run(3, true),
        "clamped to the model's channels"
    );
}

/// Smaller than a tile, odd, several tiles, resized: every output is the
/// net size, finite, the probability in 0..1.
#[test]
fn any_size_gives_the_net_size() {
    let net = tiny_model();
    for (w, h, nw, nh) in [
        (10, 7, 10, 7),
        (45, 33, 45, 33),
        (100, 70, 100, 70),
        (40, 40, 64, 50),
    ] {
        let pixels = cells(w, h);
        let out = net.infer(image(&pixels, w, h), &settings(nw, nh)).unwrap();
        for plane in [&out.flow_y, &out.flow_x, &out.cell_prob] {
            assert_eq!(plane.len(), nw * nh, "{w}x{h} -> {nw}x{nh}");
            assert!(plane.iter().all(|v| v.is_finite()));
        }
        assert!(out.cell_prob.iter().all(|p| (0.0..=1.0).contains(p)));
    }
    assert!(net.infer(image(&[], 0, 0), &settings(1, 1)).is_err());
}

/// A wrong shape in the file is refused.
#[test]
fn a_misshaped_tensor_is_refused() {
    let net_config = CellposeSamConfig {
        embed_dim: 32,
        ..tiny_config()
    };
    // tiny_model's file with another config: every block's shapes differ.
    let bytes = {
        let net = tiny_model();
        let _ = net;
        let metadata = ModelMetadata {
            arch: ARCH.into(),
            arch_version: ARCH_VERSION,
            config: serde_json::to_string(&net_config).unwrap(),
            preprocessing: None,
            info: Default::default(),
        };
        crate::weights::to_bytes(
            &[NamedTensor {
                name: "encoder.pos_embed".into(),
                shape: vec![1],
                values: vec![0.0],
            }],
            &metadata,
        )
        .unwrap()
    };
    assert!(matches!(
        CellposeSam::from_bytes(&bytes, &Device::flex()),
        Err(NnError::InvalidModelFile(_))
    ));
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// cpsam_v2 converted (`docs/convert_cellpose_sam.py`) against Cellpose's
/// own `CPSAM` (PyTorch, f32, CPU) on the converter's reference tiles: gray
/// zero-filled, gray replicated, one channel. Release build (the network is
/// ~0.7 TFLOP per tile): `cargo test --release --target
/// x86_64-unknown-linux-gnu -p evanalyzer_nn --lib real_cpsam -- --ignored --nocapture`.
#[test]
#[ignore = "needs tests/cellpose-sam (cpsam_v2 converted + references)"]
fn real_cpsam_gives_cellposes_output() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/cellpose-sam");
    let net = CellposeSam::load(&dir.join("cpsam_v2.safetensors"), &Device::flex(), false).unwrap();
    assert_eq!(net.config(), &CellposeSamConfig::vit_l());
    for (i, channels) in [(0, 3), (1, 3), (2, 1)] {
        let reference = dir.join("reference");
        let input = read_f32(&reference.join(format!("tile{i}.input.f32")));
        let expected = read_f32(&reference.join(format!("tile{i}.output.f32")));
        let start = std::time::Instant::now();
        let out = net.forward_tile(&input, channels);
        let secs = start.elapsed().as_secs_f64();
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let plane = 256 * 256;
        let flips = out[2 * plane..]
            .iter()
            .zip(&expected[2 * plane..])
            .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
            .count();
        eprintln!(
            "tile{i}: {secs:.2} s, max |diff| {max:.2e} of values up to {scale:.1}, cell flips {flips}"
        );
        assert!(
            max < 1e-2 * scale.max(1.0) && flips <= plane / 1000,
            "tile{i}: max {max}, flips {flips}"
        );
    }
}

/// `infer` (resize, channel fill, tiles, blending, sigmoid) on cpsam_v2
/// against the libtorch engine as the app ran it (`crates/ai` example
/// `cellpose_reference`): one tile, an odd size, a resize, 1024² in 25
/// tiles. Two references:
///
/// - `reference-f32/`: the engine with a TorchScript of the original f32
///   weights (traced from Cellpose's `CPSAM`) - the same network, so the
///   app path (tiles, blending, resize) is compared: strict;
/// - `reference/`: the engine with the TorchScript the app shipped
///   (`docs/convert_cellpose.py`, which loads through `CellposeModel` and
///   so rounds the weights to bfloat16 first - 0.044 off Cellpose's f32
///   network already in PyTorch): within that rounding.
///
/// Release build, see `real_cpsam_gives_cellposes_output`.
#[test]
#[ignore = "needs tests/cellpose-sam (cpsam_v2 converted + libtorch engine references)"]
fn real_cpsam_infer_gives_the_libtorch_engines_output() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/cellpose-sam");
    let net = CellposeSam::load(&dir.join("cpsam_v2.safetensors"), &Device::flex(), false).unwrap();
    // (name, width, height, net width, net height, replicate)
    for (name, w, h, nw, nh, replicate) in [
        ("cells256-zero", 256, 256, 256, 256, false),
        ("cells300x200-replicate", 300, 200, 300, 200, true),
        ("cells300x200-resized", 300, 200, 450, 300, false),
        ("cells1024-replicate", 1024, 1024, 1024, 1024, true),
    ] {
        let input = read_f32(&dir.join("reference-f32").join(format!("{name}.input.f32")));
        let s = CellposeSamSettings {
            net_width: nw,
            net_height: nh,
            input_channels: 3,
            replicate_gray_channel: replicate,
        };
        let start = std::time::Instant::now();
        let out = net.infer(image(&input, w, h), &s).unwrap();
        let secs = start.elapsed().as_secs_f64();
        let ours = [out.flow_y, out.flow_x, out.cell_prob].concat();
        let n = nw * nh;
        let max = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        // (reference, flows, probability, flips per pixel)
        for (reference, flow_tol, prob_tol, flips_tol) in [
            ("reference-f32", 1e-3, 1e-4, 0.0),
            ("reference", 0.1, 0.05, 1e-4),
        ] {
            let engine = read_f32(&dir.join(reference).join(format!("{name}.engine.f32")));
            let (flow, prob) = (
                max(&ours[..2 * n], &engine[..2 * n]),
                max(&ours[2 * n..], &engine[2 * n..]),
            );
            let flips = ours[2 * n..]
                .iter()
                .zip(&engine[2 * n..])
                .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                .count();
            eprintln!(
                "{name} vs {reference}: {secs:.1} s, flows max |diff| {flow:.2e}, probability {prob:.2e}, flips {flips} of {n}"
            );
            assert!(
                flow < flow_tol && prob < prob_tol && flips as f64 <= flips_tol * n as f64,
                "{name} vs {reference}"
            );
        }
    }
}
