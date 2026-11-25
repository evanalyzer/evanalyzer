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
//! state dict as safetensors with PyTorch's names (e.g. what `crates/ai`'s
//! `unet_reference` exports from a bioimage.io package) plus the
//! architecture's arguments from the package's `rdf.yaml` → our model file
//! (`crate::weights`): the same tensors, `config`, the preprocessing and
//! where the weights come from. The written file is loaded back as a check.
//!
//!   cargo build-linux -p evanalyzer_nn --example convert_unet
//!   target/x86_64-unknown-linux-gnu/release/examples/convert_unet <in.safetensors> <out.safetensors> \
//!     '<config json>' [zero_mean_unit_variance|none] [key=value ...]
//!
//! e.g. romantic-shark:
//!   … weights.safetensors tests/romantic-shark/romantic-shark.safetensors \
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
    let info: BTreeMap<String, String> = args
        .iter()
        .skip(5)
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
}
