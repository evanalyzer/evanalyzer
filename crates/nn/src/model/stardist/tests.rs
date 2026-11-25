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

fn tiny_config(grid: [usize; 2]) -> StarDistConfig {
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

/// The tensors `config`'s network has, in the loader's order: name and
/// shape (the layout of `network::StarDistNet::load`, written out again).
fn layout(config: &StarDistConfig) -> Vec<(String, Vec<usize>)> {
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

fn tiny_model(grid: [usize; 2]) -> StarDist {
    let config = tiny_config(grid);
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let tensors: Vec<NamedTensor> = layout(&config)
        .into_iter()
        .map(|(name, shape)| {
            let values = (0..shape.iter().product::<usize>())
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.45
                })
                .collect();
            NamedTensor {
                name,
                shape,
                values,
            }
        })
        .collect();
    let metadata = ModelMetadata {
        arch: ARCH.into(),
        arch_version: ARCH_VERSION,
        config: serde_json::to_string(&config).unwrap(),
        preprocessing: None,
        info: Default::default(),
    };
    StarDist::from_bytes(
        &crate::weights::to_bytes(&tensors, &metadata).unwrap(),
        &Device::flex(),
    )
    .unwrap()
}

fn cells(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            let (dx, dy) = (x % 17.0 - 8.0, y % 13.0 - 6.0);
            let cell = if dx * dx + dy * dy < 20.0 { 0.8 } else { 0.1 };
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

fn whole() -> StarDistSettings {
    StarDistSettings {
        pieces: Pieces::fixed(2048),
    }
}

/// Any size gives `side / grid` cells (rounded down, at least 1) and every
/// ray plane; the probability in 0..1.
#[test]
fn any_size_gives_its_grid_cells() {
    for grid in [[2, 2], [1, 1]] {
        let net = tiny_model(grid);
        for (w, h) in [(1, 1), (5, 3), (37, 23), (64, 48)] {
            let pixels = cells(w, h);
            let out = net.infer(image(&pixels, w, h), &whole()).unwrap();
            let (gw, gh) = ((w / grid[1]).max(1), (h / grid[0]).max(1));
            assert_eq!(
                (out.grid_width, out.grid_height, out.n_rays),
                (gw, gh, 8),
                "{grid:?} {w}x{h}"
            );
            assert_eq!((out.prob.len(), out.dist.len()), (gw * gh, 8 * gw * gh));
            assert!(out.prob.iter().all(|p| (0.0..=1.0).contains(p)));
        }
    }
}

/// In pieces with the measured halo: the whole tile's result up to float
/// rounding (a network without whole-image normalization).
#[test]
fn pieces_give_the_whole_tile() {
    let net = tiny_model([2, 2]);
    let (w, h) = (600, 300);
    let pixels = cells(w, h);
    let whole_tile = net.infer(image(&pixels, w, h), &whole()).unwrap();
    let pieces = net
        .infer(
            image(&pixels, w, h),
            &StarDistSettings {
                pieces: Pieces::fixed(256),
            },
        )
        .unwrap();
    assert!(net.halo.get().copied().flatten().is_some(), "measured");
    assert_eq!(
        (pieces.grid_width, pieces.grid_height),
        (whole_tile.grid_width, whole_tile.grid_height)
    );
    let max = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(max(&pieces.prob, &whole_tile.prob) < 1e-5);
    let scale = whole_tile.dist.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = max(&pieces.dist, &whole_tile.dist);
    assert!(d < 1e-5 * scale.max(1.0), "max |diff| {d} of values up to {scale}");
}

#[test]
fn wrong_input_and_misshaped_files_are_refused() {
    let net = tiny_model([2, 2]);
    let two = Image {
        channels: 2,
        data: &[0.0; 32],
        ..image(&[], 4, 4)
    };
    assert!(matches!(
        net.infer(two, &whole()),
        Err(NnError::InvalidInput(_))
    ));
    let config = StarDistConfig {
        unet_filter_base: 8,
        ..tiny_config([2, 2])
    };
    let metadata = ModelMetadata {
        arch: ARCH.into(),
        arch_version: ARCH_VERSION,
        config: serde_json::to_string(&config).unwrap(),
        preprocessing: None,
        info: Default::default(),
    };
    let tensors: Vec<NamedTensor> = layout(&tiny_config([2, 2]))
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            NamedTensor {
                name,
                shape,
                values: vec![0.0; n],
            }
        })
        .collect();
    let bytes = crate::weights::to_bytes(&tensors, &metadata).unwrap();
    assert!(matches!(
        StarDist::from_bytes(&bytes, &Device::flex()),
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

/// The official `2D_versatile_fluo` converted (`docs/convert_stardist.py`)
/// against StarDist's own `predict` (TensorFlow, CPU) on synthetic cells:
/// 256², 101x77 (StarDist keeps a last partial cell: 51x39 against our
/// 50x38 - the common cells compared), 512x384, and 1100x600 whole and in
/// 512² pieces. StarDist's distances are at least 1e-3 (`np.maximum`), so
/// ours too for the comparison. Release build:
/// `cargo test --release --target x86_64-unknown-linux-gnu -p evanalyzer_nn --lib real_stardist -- --ignored --nocapture`.
#[test]
#[ignore = "needs tests/stardist (2D_versatile_fluo converted + references)"]
fn real_stardist_gives_stardists_output() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/stardist");
    let net = StarDist::load(&dir.join("2D_versatile_fluo.safetensors"), &Device::flex()).unwrap();
    assert_eq!(net.config(), &StarDistConfig::versatile_fluo());
    for (name, w, h, piece) in [
        ("cells256", 256, 256, 2048),
        ("cells101x77", 101, 77, 2048),
        ("cells512x384", 512, 384, 2048),
        ("cells1100x600", 1100, 600, 2048),
        ("cells1100x600", 1100, 600, 512),
    ] {
        let reference = dir.join("reference");
        let input = read_f32(&reference.join(format!("{name}.input.f32")));
        let expected = read_f32(&reference.join(format!("{name}.output.f32")));
        let shape: serde_json::Value = serde_json::from_slice(
            &std::fs::read(reference.join(format!("{name}.shape.json"))).unwrap(),
        )
        .unwrap();
        let (egw, egh) = (
            shape["grid_width"].as_u64().unwrap() as usize,
            shape["grid_height"].as_u64().unwrap() as usize,
        );
        let start = std::time::Instant::now();
        let out = net
            .infer(
                image(&input, w, h),
                &StarDistSettings {
                    pieces: Pieces::fixed(piece),
                },
            )
            .unwrap();
        let secs = start.elapsed().as_secs_f64();
        let (gw, gh) = (out.grid_width, out.grid_height);
        assert!(
            gw <= egw && gh <= egh && egw - gw <= 1 && egh - gh <= 1,
            "{name}"
        );
        let (mut prob_max, mut dist_max, mut flips) = (0.0f32, 0.0f32, 0);
        for plane in 0..=out.n_rays {
            for y in 0..gh {
                for x in 0..gw {
                    let theirs = expected[(plane * egh + y) * egw + x];
                    if plane == 0 {
                        let ours = out.prob[y * gw + x];
                        prob_max = prob_max.max((ours - theirs).abs());
                        flips += usize::from((ours > 0.479) != (theirs > 0.479));
                    } else {
                        let ours = out.dist[((plane - 1) * gh + y) * gw + x].max(1e-3);
                        dist_max = dist_max.max((ours - theirs).abs());
                    }
                }
            }
        }
        eprintln!(
            "{name} (pieces {piece}): {secs:.2} s, grid {gw}x{gh} (StarDist {egw}x{egh}), prob max |diff| {prob_max:.2e}, dist {dist_max:.2e}, flips {flips}"
        );
        assert!(
            prob_max < 1e-3 && dist_max < 1e-2 && flips <= gw * gh / 10_000,
            "{name}"
        );
    }
}
