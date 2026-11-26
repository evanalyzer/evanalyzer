//! # tests
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::infer::padding;
use super::*;
use crate::pieces::pad_plane;
use std::collections::BTreeMap;

fn config(norm: Norm, final_activation: FinalActivation) -> UNetConfig {
    UNetConfig {
        in_channels: 1,
        out_channels: 2,
        depth: 2,
        initial_features: 4,
        gain: 2,
        norm,
        final_activation,
    }
}

fn settings(pieces: Pieces) -> UNetParams {
    UNetParams {
        normalization: Normalization::ModelDefault,
        foreground_channel: 0,
        softmax: false,
        boundary_channel: Some(1),
        pieces,
        probability_threshold: 0.5,
        boundary_threshold: 0.5,
        foreground: 1,
    }
}

/// `pre` and `infer`, as the engine runs them (here without a slot).
fn raw(net: &UNet, image: Image<'_>, params: &UNetParams) -> Result<UNetRaw, NnError> {
    let (input, ()) = UNet::pre(&net.to_header(), image, params)?;
    net.infer(&input)
}

/// Cells (bright discs) on a textured background.
fn cells(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            let (dx, dy) = (x % 23.0 - 11.0, y % 19.0 - 9.0);
            let cell = if dx * dx + dy * dy < 40.0 { 0.8 } else { 0.1 };
            cell + 0.05 * (x * 0.31).sin() * (y * 0.17).cos()
        })
        .collect()
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
fn padding_reaches_the_multiple_with_at_least_the_margin() {
    for multiple in [16, 32, 64] {
        for len in [1, 15, 16, 31, 250, 256, 1040, 1388] {
            let (before, after) = padding(len, multiple);
            assert_eq!(before, MARGIN, "len {len}");
            assert!(after >= MARGIN, "len {len}: only {after} px after");
            assert_eq!((before + len + after) % multiple, 0, "len {len}");
            assert!(
                after < MARGIN + multiple,
                "len {len}: more padding than needed"
            );
        }
    }
    assert_eq!(padding(250, 16), (16, 22));
    assert_eq!(padding(256, 16), (16, 16));
}

#[test]
fn planes_mirror_like_pytorch_and_tiny_ones_repeat_their_edge() {
    // 3x2: rows [1 2 3], [4 5 6].
    let plane = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = Vec::new();
    pad_plane(&plane, 3, 2, (2, 2, 1, 0), &mut out);
    // reflect: row -1 = row 1; x -2,-1 = 2,1; x 3,4 = 1,0.
    assert_eq!(&out[..7], [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0]);
    assert_eq!(&out[7..14], [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    out.clear();
    pad_plane(&plane, 3, 2, (3, 0, 0, 0), &mut out);
    assert_eq!(&out[..6], [1.0, 1.0, 1.0, 1.0, 2.0, 3.0], "replicate");
}

#[test]
fn configs_read_as_torch_ems_arguments_and_absurd_ones_are_refused() {
    let json = r#"{"in_channels":1,"out_channels":2,"depth":4,"initial_features":64,
        "gain":2,"norm":"instance","final_activation":"sigmoid"}"#;
    let c: UNetConfig = serde_json::from_str(json).unwrap();
    assert_eq!(c.norm, Norm::Instance);
    assert!(c.check().is_ok());
    let group: Norm = serde_json::from_str(r#"{"group":{"groups":8}}"#).unwrap();
    assert_eq!(group, Norm::Group { groups: 8 });
    for bad in [
        UNetConfig {
            depth: 9,
            ..c.clone()
        },
        UNetConfig {
            in_channels: 0,
            ..c.clone()
        },
        UNetConfig {
            initial_features: 4096,
            ..c.clone()
        },
        UNetConfig {
            gain: 4,
            ..c.clone()
        },
    ] {
        assert!(
            matches!(bad.check(), Err(NnError::InvalidModelFile(_))),
            "{bad:?}"
        );
    }
    let unknown = r#"{"in_channels":1,"out_channels":2,"depth":4,"initial_features":64,
        "gain":2,"norm":"instance","final_activation":"sigmoid","extra":1}"#;
    assert!(serde_json::from_str::<UNetConfig>(unknown).is_err());
}

/// Written as a model file and loaded again: the same results, with the
/// metadata's preprocessing.
#[test]
fn a_saved_unet_loads_with_the_same_results() {
    let device = Device::flex();
    let mut net =
        UNet::with_random_weights(config(Norm::Instance, FinalActivation::Sigmoid), &device);
    net.set_preprocessing(Preprocessing::ZeroMeanUnitVariance);
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("unet.safetensors");
    let info = BTreeMap::from([("license".to_string(), "MIT".to_string())]);
    let (tensors, metadata) = net.to_file_contents(info.clone());
    crate::weights::write(&path, &tensors, &metadata).unwrap();

    let loaded = UNet::load(&path, &device, false).unwrap();
    assert_eq!(loaded.config(), net.config());
    assert_eq!(loaded.preprocessing(), Preprocessing::ZeroMeanUnitVariance);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(ModelFile::parse(&bytes).unwrap().metadata.info, info);
    let pixels = cells(40, 30);
    let s = settings(Pieces::fixed(2048));
    assert_eq!(
        raw(&loaded, image(&pixels, 40, 30), &s).unwrap(),
        raw(&net, image(&pixels, 40, 30), &s).unwrap()
    );
}

#[test]
fn any_tile_size_gives_its_size_and_the_asked_channels() {
    let device = Device::flex();
    let net = UNet::with_random_weights(config(Norm::Instance, FinalActivation::Sigmoid), &device);
    for (w, h) in [(1, 1), (5, 3), (37, 23), (64, 64)] {
        let pixels = cells(w, h);
        let out = raw(&net, image(&pixels, w, h), &settings(Pieces::fixed(2048))).unwrap();
        assert_eq!(out.probabilities.len(), w * h, "{w}x{h}");
        assert_eq!(out.boundary.as_ref().map(Vec::len), Some(w * h), "{w}x{h}");
        assert!(out.probabilities.iter().all(|v| (0.0..=1.0).contains(v)));
    }
    let pixels = cells(20, 20);
    let no_boundary = UNetParams {
        boundary_channel: Some(5),
        ..settings(Pieces::fixed(2048))
    };
    assert_eq!(
        raw(&net, image(&pixels, 20, 20), &no_boundary)
            .unwrap()
            .boundary,
        None
    );
    let wrong = Image {
        channels: 2,
        data: &[0.0; 800],
        ..image(&pixels, 20, 20)
    };
    assert!(matches!(
        raw(&net, wrong, &settings(Pieces::fixed(2048))),
        Err(NnError::InvalidInput(_))
    ));
}

/// Softmax over class scores: the foreground of class 0 and of class 1 sum
/// to 1; the out-of-range foreground channel is the last.
#[test]
fn softmax_classes_sum_to_one() {
    let device = Device::flex();
    let net = UNet::with_random_weights(config(Norm::None, FinalActivation::None), &device);
    let pixels = cells(24, 20);
    let run = |foreground_channel| {
        let s = UNetParams {
            foreground_channel,
            softmax: true,
            boundary_channel: None,
            ..settings(Pieces::fixed(2048))
        };
        raw(&net, image(&pixels, 24, 20), &s).unwrap().probabilities
    };
    let (a, b) = (run(0), run(1));
    for (a, b) in a.iter().zip(&b) {
        assert!((a + b - 1.0).abs() < 1e-5);
    }
    assert_eq!(run(7), b, "clamped to the last channel");
}

/// A network without whole-image normalization in pieces with its measured
/// halo: the whole tile's result up to float rounding.
#[test]
fn pieces_give_the_whole_tile_for_a_local_network() {
    let device = Device::flex();
    let net = UNet::with_random_weights(config(Norm::None, FinalActivation::Sigmoid), &device);
    let (w, h) = (600, 300);
    let pixels = cells(w, h);
    let whole = raw(&net, image(&pixels, w, h), &settings(Pieces::fixed(2048))).unwrap();
    let pieces = raw(&net, image(&pixels, w, h), &settings(Pieces::fixed(256))).unwrap();
    assert!(net.halo.get().copied().flatten().is_some(), "measured");
    let diff = whole
        .probabilities
        .iter()
        .zip(&pieces.probabilities)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-5, "max difference {diff}");
}

/// The header comes from the file's header alone and says what the loaded
/// model says; a file that is no model names itself.
#[test]
fn the_header_is_read_without_the_weights() {
    let device = Device::flex();
    let mut net =
        UNet::with_random_weights(config(Norm::Instance, FinalActivation::Sigmoid), &device);
    net.set_preprocessing(Preprocessing::ZeroMeanUnitVariance);
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("unet.safetensors");
    let (tensors, metadata) = net.to_file_contents(BTreeMap::new());
    crate::weights::write(&path, &tensors, &metadata).unwrap();
    assert_eq!(UNet::header(&path).unwrap(), net.to_header());

    let junk = dir.path().join("junk.safetensors");
    std::fs::write(&junk, b"not a model file at all").unwrap();
    let err = UNet::header(&junk).unwrap_err();
    assert!(err.to_string().contains("junk.safetensors"), "{err}");
}

/// Normalized while each piece is copied for the network: the same result,
/// to the bit, as the tile normalized up front (as before) and run without.
#[test]
fn normalizing_on_the_way_equals_normalizing_up_front() {
    let device = Device::flex();
    let mut net = UNet::with_random_weights(config(Norm::None, FinalActivation::Sigmoid), &device);
    net.set_preprocessing(Preprocessing::ZeroMeanUnitVariance);
    let (w, h) = (600, 300);
    let pixels: Vec<f32> = cells(w, h).iter().map(|v| v * 3000.0 + 200.0).collect();
    let manual = |affine: pre::Affine| {
        let mut normalized = pixels.clone();
        affine.apply(&mut normalized);
        let none = UNetParams {
            normalization: Normalization::None,
            ..settings(Pieces::fixed(256))
        };
        raw(&net, image(&normalized, w, h), &none).unwrap()
    };
    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = pixels
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let zmuv = pre::Affine {
        offset: mean,
        scale: 1.0 / (var.sqrt() + 1e-6),
    };
    assert_eq!(
        raw(&net, image(&pixels, w, h), &settings(Pieces::fixed(256))).unwrap(),
        manual(zmuv),
        "the file's zero mean / unit variance, in pieces"
    );

    let (lo, hi) = pixels
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = pre::Affine {
        offset: f64::from(lo),
        scale: 1.0 / (f64::from(hi) - f64::from(lo) + 1e-6),
    };
    let percentile = UNetParams {
        normalization: Normalization::Percentile {
            low: 0.0,
            high: 100.0,
        },
        ..settings(Pieces::fixed(256))
    };
    assert_eq!(
        raw(&net, image(&pixels, w, h), &percentile).unwrap(),
        manual(range),
        "percentiles 0 and 100 = the tile's range"
    );
}

#[test]
fn thresholds_keep_probable_pixels_and_carve_out_boundaries() {
    const FG: u32 = 7;
    let mut out = [99u32; 6];
    // Inclusive probability threshold; boundary from its threshold on.
    let probs = [0.1, 0.5, 0.9, 0.9, 0.9, 0.2];
    post::threshold(&probs, None, 0.5, 0.5, FG, &mut out).unwrap();
    assert_eq!(out, [0, FG, FG, FG, FG, 0]);
    let boundary = [0.0, 0.1, 0.9, 0.5, 0.49, 0.0];
    post::threshold(&probs, Some(&boundary), 0.5, 0.5, FG, &mut out).unwrap();
    assert_eq!(out, [0, FG, 0, 0, FG, 0]);
    assert!(matches!(
        post::threshold(&probs, None, 0.5, 0.5, FG, &mut [0; 5]),
        Err(NnError::InvalidInput(_))
    ));
}

/// Through the engine: pre, infer in a slot, post - the mask of the
/// probabilities `infer` gives.
#[test]
fn the_engine_runs_pre_infer_and_post() {
    use crate::runtime::{DeviceSelection, Engine, EngineConfig};
    let device = Device::flex();
    let net = UNet::with_random_weights(config(Norm::None, FinalActivation::Sigmoid), &device);
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("unet.safetensors");
    let (tensors, metadata) = net.to_file_contents(BTreeMap::new());
    crate::weights::write(&path, &tensors, &metadata).unwrap();
    let engine = Engine::new(&EngineConfig {
        devices: DeviceSelection::Cpu,
        gpu_slots: 1,
        cores: 2,
    });
    let (w, h) = (37, 23);
    let pixels = cells(w, h);
    let params = UNetParams {
        foreground: 3,
        ..settings(Pieces::fixed(2048))
    };
    let mut out = vec![0u32; w * h];
    let written = engine
        .run::<UNet>(&path, image(&pixels, w, h), &params, &mut out)
        .unwrap();
    assert_eq!(written, Written::Mask);
    let probs = raw(&net, image(&pixels, w, h), &params).unwrap();
    let mut expected = vec![0u32; w * h];
    post::threshold(
        &probs.probabilities,
        probs.boundary.as_deref(),
        0.5,
        0.5,
        3,
        &mut expected,
    )
    .unwrap();
    assert_eq!(out, expected);
}

/// A `.npy` file of f32 (little-endian, C order): its values.
fn read_npy_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("'<f4'") && header.contains("False"),
        "{header}"
    );
    bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// A raw little-endian f32 file's values.
fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Max |a - b| and the pixels of the first `pixels` values on different
/// sides of 0.5.
fn difference(a: &[f32], b: &[f32], pixels: usize) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let flips = a[..pixels]
        .iter()
        .zip(&b[..pixels])
        .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
        .count();
    (max, flips)
}

/// romantic-shark converted to our file (`examples/convert_unet`, into
/// `tests/romantic-shark/romantic-shark.safetensors`), against PyTorch:
///
/// - the network alone on the package's 256² test input (normalized) vs
///   the package's own expected output;
/// - `infer` (preprocessing, mirror padding, pieces) vs the libtorch engine
///   as the app ran it (the removed `crates/ai` example `unet_reference` →
///   `tests/romantic-shark/reference/`), on 256² and 1024². The padding
///   changes InstanceNorm's statistics, so `infer` differs from the
///   package's unpadded output at the borders (up to 0.8) - as libtorch did.
#[test]
#[ignore = "needs tests/romantic-shark (bioimage.io, ~223 MB) converted + references"]
fn real_romantic_shark_gives_pytorchs_results() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/romantic-shark");
    let reference = dir.join("reference");
    let device = Device::flex();
    let net = UNet::load(&dir.join("romantic-shark.safetensors"), &device, false).unwrap();
    assert_eq!(net.preprocessing(), Preprocessing::ZeroMeanUnitVariance);

    let side = 256;
    let normalized = read_f32(&reference.join("test256.input.f32"));
    let expected = read_npy_f32(&dir.join("test_output_0.npy"));
    let network_only = net.forward_raw(&normalized, side, side, &device);
    let (max, flips) = difference(&network_only, &expected, side * side);
    assert!(
        max < 1e-4 && flips == 0,
        "network vs package: max {max}, {flips} flips"
    );

    for (name, side, input) in [
        ("test256", 256, read_npy_f32(&dir.join("test_input_0.npy"))),
        (
            "cells1024",
            1024,
            read_f32(&reference.join("cells1024.input.f32")),
        ),
    ] {
        let engine = read_f32(&reference.join(format!("{name}.engine.f32")));
        let out = raw(
            &net,
            image(&input, side, side),
            &settings(Pieces::fixed(2048)),
        )
        .unwrap();
        let mut ours = out.probabilities;
        ours.extend(out.boundary.unwrap());
        // Float rounding only (another summation order): burn-flex flips 3
        // of the 1024² image's 1,048,576 pixels, which sit right at 0.5 (the
        // image has large areas of exactly equal values).
        let (max, flips) = difference(&ours, &engine, side * side);
        assert!(
            max < 5e-3 && flips <= side * side / 100_000,
            "{name}: infer vs engine: max {max}, {flips} flips"
        );
        eprintln!("{name}: infer vs libtorch engine: max {max:.2e}, {flips} flips");
    }
}

/// The GPU path (the network in f16, norms in f32, as `load` sets it on a
/// GPU) against the libtorch engine on romantic-shark's 1024² image: within
/// f16 rounding (measured 2026-11-26: max 1.2e-2, 99 of 1,048,576 pixels on
/// the other side of 0.5 - libtorch's own GPU run used TF32). Needs a
/// discrete GPU; release build.
#[test]
#[ignore = "needs tests/romantic-shark and a discrete GPU"]
fn real_romantic_shark_in_f16_on_the_gpu_stays_within_f16_rounding() {
    use burn::tensor::DeviceKind;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/romantic-shark");
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let net = UNet::load(&dir.join("romantic-shark.safetensors"), &device, true).unwrap();
    let reference = dir.join("reference");
    let side = 1024;
    let input = read_f32(&reference.join("cells1024.input.f32"));
    let engine = read_f32(&reference.join("cells1024.engine.f32"));
    let out = raw(
        &net,
        image(&input, side, side),
        &settings(Pieces::fixed(1024)),
    )
    .unwrap();
    let mut ours = out.probabilities;
    ours.extend(out.boundary.unwrap());
    let (max, flips) = difference(&ours, &engine, side * side);
    eprintln!("f16: max {max:.2e}, {flips} flips");
    assert!(
        max < 0.05 && flips <= side * side / 5_000,
        "max {max}, {flips} flips"
    );
}
