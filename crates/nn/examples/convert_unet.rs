//! # convert_unet
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Dev converter (by hand, once per supported model): a torch-em `UNet2d`
//! state dict as safetensors with PyTorch's names (what
//! `docs/export_unet_weights.py` exports from a bioimage.io package) plus the
//! architecture's arguments from the package's `rdf.yaml` → our model file
//! (`crate::weights`): the same tensors, `config`, the preprocessing and
//! where the weights come from. The written file is loaded back as a check.
//!
//!   cargo build-linux -p evanalyzer_nn --example convert_unet
//!   target/x86_64-unknown-linux-gnu/release/examples/convert_unet <in.safetensors> <out.evanet> \
//!     '<config json>' [zero_mean_unit_variance|none] [key=value ...] \
//!     [--check=<test_input_0.npy>,<test_output_0.npy>]
//!
//! `--check`: the written file loaded back and run (the network alone, on
//! the CPU) on the package's test input, normalized as the file says,
//! against the package's expected output - the conversion is refused when
//! they differ by more than float rounding.
//!
//! e.g. romantic-shark:
//!   … weights.safetensors tests/romantic-shark/romantic-shark.evanet \
//!     '{"in_channels":1,"out_channels":2,"depth":4,"initial_features":64,"gain":2,
//!       "norm":"instance","final_activation":"sigmoid"}' zero_mean_unit_variance \
//!     source=bioimage.io/romantic-shark license=CC-BY-4.0 citation=…

use burn::tensor::Device;
use evanalyzer_nn::Model;
use evanalyzer_nn::model::unet::{Preprocessing, UNet, UNetConfig};
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = Path::new(args.get(1).expect("input safetensors (PyTorch names)"));
    let output = Path::new(args.get(2).expect("output model file"));
    let config: UNetConfig =
        serde_json::from_str(args.get(3).expect("config JSON")).expect("a U-Net config");
    let preprocessing = match args.get(4).map_or("none", String::as_str) {
        "none" => Preprocessing::None,
        "zero_mean_unit_variance" => Preprocessing::ZeroMeanUnitVariance,
        other => panic!("unknown preprocessing {other}"),
    };
    let check = args
        .iter()
        .find_map(|a| a.strip_prefix("--check="))
        .map(|v| v.split_once(',').expect("--check=<input.npy>,<output.npy>"));
    let info: BTreeMap<String, String> = args
        .iter()
        .skip(5)
        .filter(|a| !a.starts_with("--check="))
        .map(|kv| {
            let (k, v) = kv.split_once('=').expect("key=value");
            (k.to_string(), v.to_string())
        })
        .collect();

    let device = Device::flex();
    let bytes = std::fs::read(input).unwrap();
    let mut net = UNet::with_torch_em_weights(config, &bytes, &device).unwrap();
    net.set_preprocessing(preprocessing);
    let (tensors, metadata) = net.to_file_contents(info);
    evanalyzer_nn::weights::write(output, &tensors, &metadata).unwrap();

    let loaded = UNet::load(output, &device, false).unwrap();
    assert_eq!(loaded.config(), net.config());
    assert_eq!(loaded.preprocessing(), preprocessing);
    println!(
        "wrote {} ({} tensors, {:?}, {:?})",
        output.display(),
        tensors.len(),
        loaded.config(),
        preprocessing
    );
    if let Some((test_input, test_output)) = check {
        check_against_the_package(&loaded, Path::new(test_input), Path::new(test_output));
    }
}

/// A `.npy` file of f32 (little-endian) or u8, C order: its shape and the
/// values as f32 (bioimage.io's `ensure_dtype: float32`).
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(header.contains("False"), "{header}");
    let data = &bytes[10 + header_len..];
    let values: Vec<f32> = if header.contains("'<f4'") {
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else if header.contains("'|u1'") {
        data.iter().map(|&v| f32::from(v)).collect()
    } else {
        panic!("only f32 and u8 test tensors: {header}")
    };
    let shape = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .unwrap()
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    (shape, values)
}

/// The network on the package's test input (`[1, C, H, W]`, normalized as
/// the file says - zero mean / unit variance in f64 like nn) against the
/// package's output: max |difference| and the pixels of channel 0 on
/// different sides of 0.5.
fn check_against_the_package(net: &UNet, input: &Path, output: &Path) {
    let (shape, pixels) = read_npy_f32(input);
    let (height, width) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let normalized: Vec<f32> = match net.preprocessing() {
        Preprocessing::None => pixels,
        Preprocessing::ZeroMeanUnitVariance => {
            let n = pixels.len() as f64;
            let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let var = pixels
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / n;
            let scale = 1.0 / (var.sqrt() + 1e-6);
            pixels
                .iter()
                .map(|&v| ((f64::from(v) - mean) * scale) as f32)
                .collect()
        }
    };
    let ours = net.forward_raw(&normalized, height, width, &Device::flex());
    let (_, expected) = read_npy_f32(output);
    assert_eq!(ours.len(), expected.len(), "output size");
    let max = ours
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let flips = ours[..width * height]
        .iter()
        .zip(&expected[..width * height])
        .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
        .count();
    println!(
        "check against the package ({width}x{height}): max |difference| {max:.2e}, {flips} pixels flip"
    );
    assert!(
        max < 1e-3 && flips <= width * height / 100_000,
        "the converted model doesn't give the package's output"
    );
}
