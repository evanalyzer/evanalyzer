//! # test_support
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Model files of tiny networks with deterministic pseudo-random weights -
//! for tests here and in core (the model steps end to end). Real files of
//! each architecture, a few KB; their outputs mean nothing.

use crate::model::cellpose_sam::{self, CellposeSamConfig};
use crate::model::stardist::{self, StarDistConfig};
use crate::model::unet::{FinalActivation, Norm, Preprocessing, UNet, UNetConfig};
use crate::weights::{ModelMetadata, NamedTensor, to_bytes};
use burn::tensor::Device;

/// A U-Net of depth 2 with 4 features, InstanceNorm, sigmoid, 2 outputs.
pub fn tiny_unet_config() -> UNetConfig {
    UNetConfig {
        in_channels: 1,
        out_channels: 2,
        depth: 2,
        initial_features: 4,
        gain: 2,
        norm: Norm::Instance,
        final_activation: FinalActivation::Sigmoid,
    }
}

/// The model file of a U-Net of `config` with random weights.
pub fn unet_file(config: UNetConfig, preprocessing: Preprocessing) -> Vec<u8> {
    let mut net = UNet::with_random_weights(config, &Device::flex());
    net.set_preprocessing(preprocessing);
    let (tensors, metadata) = net.to_file_contents(Default::default());
    to_bytes(&tensors, &metadata).expect("a model file")
}

/// A StarDist of grid `grid`, depth 2, 4 filters, 8 rays.
pub fn tiny_stardist_config(grid: [usize; 2]) -> StarDistConfig {
    StarDistConfig {
        in_channels: 1,
        n_rays: 8,
        grid,
        unet_depth: 2,
        unet_filter_base: 4,
        unet_conv_per_depth: 2,
        kernel_size: 3,
        conv_after_unet: 8,
    }
}

/// The tensors `config`'s StarDist has, in the loader's order: name and
/// shape (the layout of `StarDistNet::load`, written out again).
pub fn layout(config: &StarDistConfig) -> Vec<(String, Vec<usize>)> {
    let (k, base, n_conv) = (
        config.kernel_size,
        config.unet_filter_base,
        config.unet_conv_per_depth,
    );
    let level = |n: usize| base << n;
    let mut out = Vec::new();
    let mut conv = |name: String, n_in: usize, n_out: usize, k: usize| {
        out.push((format!("{name}.weight"), vec![n_out, n_in, k, k]));
        out.push((format!("{name}.bias"), vec![n_out]));
        n_out
    };
    let (mut pooled, mut c, mut index) = ([1, 1], config.in_channels, 0);
    while pooled != config.grid {
        pooled = [0, 1].map(|a| {
            if config.grid[a] > pooled[a] {
                pooled[a] * 2
            } else {
                pooled[a]
            }
        });
        for _ in 0..n_conv {
            c = conv(format!("grid.{index}"), c, base, k);
            index += 1;
        }
    }
    let mut skips = Vec::new();
    for n in 0..config.unet_depth {
        for i in 0..n_conv {
            c = conv(format!("down_level_{n}_no_{i}"), c, level(n), k);
        }
        skips.push(c);
    }
    for i in 0..n_conv - 1 {
        c = conv(format!("middle_{i}"), c, level(config.unet_depth), k);
    }
    c = conv(
        format!("middle_{n_conv}"),
        c,
        level(config.unet_depth - 1),
        k,
    );
    for n in (0..config.unet_depth).rev() {
        c += skips[n];
        for i in 0..n_conv - 1 {
            c = conv(format!("up_level_{n}_no_{i}"), c, level(n), k);
        }
        c = conv(
            format!("up_level_{n}_no_{n_conv}"),
            c,
            level(n.saturating_sub(1)),
            k,
        );
    }
    if config.conv_after_unet > 0 {
        c = conv("features".into(), c, config.conv_after_unet, k);
    }
    conv("prob".into(), c, 1, 1);
    conv("dist".into(), c, config.n_rays, 1);
    out
}

/// Pseudo-random values (xorshift64) in `[-0.5, 0.5) * scale + shift`.
fn values(seed: &mut u64, n: usize, scale: f32, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale + shift
        })
        .collect()
}

/// The model file of a StarDist of `config` with random weights.
pub fn stardist_file(config: &StarDistConfig) -> Vec<u8> {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let tensors: Vec<NamedTensor> = layout(config)
        .into_iter()
        .map(|(name, shape)| {
            let values = values(&mut seed, shape.iter().product(), 1.0, 0.05);
            NamedTensor {
                name,
                shape,
                values,
            }
        })
        .collect();
    let metadata = ModelMetadata {
        arch: stardist::ARCH.into(),
        arch_version: stardist::ARCH_VERSION,
        config: serde_json::to_string(config).expect("a config serializes"),
        preprocessing: None,
        info: Default::default(),
    };
    to_bytes(&tensors, &metadata).expect("a model file")
}

/// A Cellpose-SAM of 16 features, 2 blocks of 2 heads, 32² tiles of 8²
/// patches (4² tokens).
pub fn tiny_cellpose_sam_config() -> CellposeSamConfig {
    CellposeSamConfig {
        in_channels: 3,
        tile_size: 32,
        patch_size: 8,
        embed_dim: 16,
        depth: 2,
        num_heads: 2,
        mlp_dim: 32,
        neck_dim: 8,
        out_channels: 3,
    }
}

/// The model file of a Cellpose-SAM of `config` with random weights
/// (relative positions of 5 rows, resized on loading).
pub fn cellpose_sam_file(config: &CellposeSamConfig) -> Vec<u8> {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut tensor = |name: String, shape: Vec<usize>, scale: f32| {
        let values = values(&mut seed, shape.iter().product(), scale, 0.0);
        NamedTensor {
            name,
            shape,
            values,
        }
    };
    let (c, p, t) = (config.embed_dim, config.patch_size, config.tokens());
    let (hd, m, n) = (c / config.num_heads, config.mlp_dim, config.neck_dim);
    let mut tensors = vec![
        tensor(
            "encoder.patch_embed.proj.weight".into(),
            vec![c, 3, p, p],
            0.2,
        ),
        tensor("encoder.patch_embed.proj.bias".into(), vec![c], 0.2),
        tensor("encoder.pos_embed".into(), vec![1, t, t, c], 0.2),
        tensor("encoder.neck.0.weight".into(), vec![n, c, 1, 1], 0.5),
        tensor("encoder.neck.1.weight".into(), vec![n], 1.0),
        tensor("encoder.neck.1.bias".into(), vec![n], 1.0),
        tensor("encoder.neck.2.weight".into(), vec![n, n, 3, 3], 0.3),
        tensor("encoder.neck.3.weight".into(), vec![n], 1.0),
        tensor("encoder.neck.3.bias".into(), vec![n], 1.0),
        tensor("out.weight".into(), vec![3 * p * p, n, 1, 1], 0.5),
        tensor("out.bias".into(), vec![3 * p * p], 0.5),
    ];
    for i in 0..config.depth {
        let b = |s: &str| format!("encoder.blocks.{i}.{s}");
        tensors.extend([
            tensor(b("norm1.weight"), vec![c], 1.0),
            tensor(b("norm1.bias"), vec![c], 1.0),
            tensor(b("attn.rel_pos_h"), vec![5, hd], 0.5),
            tensor(b("attn.rel_pos_w"), vec![5, hd], 0.5),
            tensor(b("attn.qkv.weight"), vec![3 * c, c], 0.5),
            tensor(b("attn.qkv.bias"), vec![3 * c], 0.5),
            tensor(b("attn.proj.weight"), vec![c, c], 0.5),
            tensor(b("attn.proj.bias"), vec![c], 0.5),
            tensor(b("norm2.weight"), vec![c], 1.0),
            tensor(b("norm2.bias"), vec![c], 1.0),
            tensor(b("mlp.lin1.weight"), vec![m, c], 0.5),
            tensor(b("mlp.lin1.bias"), vec![m], 0.5),
            tensor(b("mlp.lin2.weight"), vec![c, m], 0.5),
            tensor(b("mlp.lin2.bias"), vec![c], 0.5),
        ]);
    }
    let metadata = ModelMetadata {
        arch: cellpose_sam::ARCH.into(),
        arch_version: cellpose_sam::ARCH_VERSION,
        config: serde_json::to_string(config).expect("a config serializes"),
        preprocessing: None,
        info: Default::default(),
    };
    to_bytes(&tensors, &metadata).expect("a model file")
}

/// A tiny random SpotNet model file of `in_channels` (1 = the spot
/// channel, 2 = plus a reference channel): 4 feature maps, the default
/// dilations, weights from a fixed seed.
pub fn spotnet_file(in_channels: usize) -> Vec<u8> {
    use crate::model::spotnet::{
        PercentileNormalization, SpotNetConfig, SpotNetModel, SpotNetSettings,
    };
    let network = SpotNetConfig {
        in_channels,
        channels: 4,
        ..SpotNetConfig::default()
    };
    let settings = SpotNetSettings {
        network,
        normalization: PercentileNormalization::default(),
        sigma: 1.5,
    };
    SpotNetModel::untrained(settings, 1, &burn::tensor::Device::flex())
        .to_bytes()
        .expect("a SpotNet file")
}
