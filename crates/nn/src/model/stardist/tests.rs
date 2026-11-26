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

use crate::test_support::{layout, stardist_file, tiny_stardist_config as tiny_config};

fn tiny_model(grid: [usize; 2]) -> StarDist {
    StarDist::from_bytes(&stardist_file(&tiny_config(grid)), &Device::flex()).unwrap()
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

/// The whole tile at once, the pixels as they are (as before the move).
fn whole() -> StarDistParams {
    in_pieces(2048)
}

fn in_pieces(side: usize) -> StarDistParams {
    StarDistParams {
        normalization: Normalization::None,
        pieces: Pieces::fixed(side),
        probability_threshold: 0.5,
        nms_threshold: 0.3,
    }
}

/// `pre` and `infer`, as the engine runs them (here without a slot).
fn raw(net: &StarDist, image: Image<'_>, params: &StarDistParams) -> Result<StarDistRaw, NnError> {
    let (input, _) = StarDist::pre(&net.to_header(), image, params)?;
    net.infer(&input)
}

/// Any size gives `side / grid` cells (rounded down, at least 1) and every
/// ray plane; the probability in 0..1.
#[test]
fn any_size_gives_its_grid_cells() {
    for grid in [[2, 2], [1, 1]] {
        let net = tiny_model(grid);
        for (w, h) in [(1, 1), (5, 3), (37, 23), (64, 48)] {
            let pixels = cells(w, h);
            let out = raw(&net, image(&pixels, w, h), &whole()).unwrap();
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
    let whole_tile = raw(&net, image(&pixels, w, h), &whole()).unwrap();
    let pieces = raw(&net, image(&pixels, w, h), &in_pieces(256)).unwrap();
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
    assert!(
        d < 1e-5 * scale.max(1.0),
        "max |diff| {d} of values up to {scale}"
    );
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
        raw(&net, two, &whole()),
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
    let net = StarDist::load(
        &dir.join("2D_versatile_fluo.evanet"),
        &Device::flex(),
        false,
    )
    .unwrap();
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
        let out = raw(&net, image(&input, w, h), &in_pieces(piece)).unwrap();
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

/// The GPU path (the whole network in f16, as `load` sets it on a GPU)
/// against StarDist's own `predict` on 1100x600: within f16 rounding
/// (measured 2026-11-26: probability 9.4e-4, distances 3.4e-2, 10 of 165,000
/// cells on the other side of the threshold). Needs a discrete GPU; release
/// build.
#[test]
#[ignore = "needs tests/stardist and a discrete GPU"]
fn real_stardist_in_f16_on_the_gpu_stays_within_f16_rounding() {
    use burn::tensor::DeviceKind;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/stardist");
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let net = StarDist::load(&dir.join("2D_versatile_fluo.evanet"), &device, true).unwrap();
    let reference = dir.join("reference");
    let (w, h) = (1100, 600);
    let input = read_f32(&reference.join("cells1100x600.input.f32"));
    let expected = read_f32(&reference.join("cells1100x600.output.f32"));
    let shape: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("cells1100x600.shape.json")).unwrap())
            .unwrap();
    let egw = shape["grid_width"].as_u64().unwrap() as usize;
    let egh = shape["grid_height"].as_u64().unwrap() as usize;
    let out = raw(&net, image(&input, w, h), &in_pieces(2048)).unwrap();
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
    eprintln!("f16: probability {prob:.2e}, distances {dist:.2e}, flips {flips}");
    assert!(prob < 5e-3 && dist < 0.1 && flips as f64 <= 2e-4 * (gw * gh) as f64);
}

/// Normalized while each piece is copied for the network: the same answer,
/// to the bit, as the tile normalized up front (csbdeep's percentiles) and
/// run without - and the default is StarDist's 1 / 99.8.
#[test]
fn normalizing_on_the_way_equals_normalizing_up_front() {
    let net = tiny_model([2, 2]);
    let (w, h) = (600, 300);
    let pixels: Vec<f32> = cells(w, h).iter().map(|v| v * 0.05 + 0.01).collect();
    let range = crate::libs::percentile::percentiles(&pixels, &[1.0, 99.8]).unwrap();
    let affine = pre::Affine {
        offset: range[0],
        scale: 1.0 / (range[1] - range[0] + 1e-20),
    };
    let mut normalized = pixels.clone();
    affine.apply(&mut normalized);
    let up_front = raw(&net, image(&normalized, w, h), &in_pieces(256)).unwrap();
    let default = StarDistParams {
        normalization: Normalization::ModelDefault,
        ..in_pieces(256)
    };
    assert_eq!(raw(&net, image(&pixels, w, h), &default).unwrap(), up_front);
    let explicit = StarDistParams {
        normalization: Normalization::Percentile {
            low: 1.0,
            high: 99.8,
        },
        ..in_pieces(256)
    };
    assert_eq!(
        raw(&net, image(&pixels, w, h), &explicit).unwrap(),
        up_front
    );
}

/// Through the engine: pre, infer in a slot, post - labels 1..=n where the
/// candidates are, the same as post on `infer`'s answer.
#[test]
fn the_engine_runs_pre_infer_and_post() {
    use crate::runtime::{DeviceSelection, Engine, EngineConfig};
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("stardist.evanet");
    std::fs::write(&path, stardist_file(&tiny_config([2, 2]))).unwrap();
    assert_eq!(
        StarDist::header(&path).unwrap(),
        tiny_model([2, 2]).to_header()
    );
    let engine = Engine::new(&EngineConfig {
        devices: DeviceSelection::Cpu,
        gpu_slots: 1,
        cores: 2,
    });
    let (w, h) = (64, 48);
    let pixels = cells(w, h);
    let params = StarDistParams {
        probability_threshold: 0.0,
        ..whole()
    };
    let mut out = vec![7u32; w * h];
    let written = engine
        .run::<StarDist>(&path, image(&pixels, w, h), &params, &mut out)
        .unwrap();
    let Written::Instances(n) = written else {
        panic!("{written:?}")
    };
    assert!(n > 0, "threshold 0: candidates everywhere");
    assert!(out.iter().all(|&l| l <= n));
    let answer = raw(&tiny_model([2, 2]), image(&pixels, w, h), &params).unwrap();
    let plan = StarDistPlan {
        width: w,
        height: h,
    };
    let mut expected = vec![0u32; w * h];
    StarDist::post(answer, plan, &params, &mut expected).unwrap();
    assert_eq!(out, expected);
    assert!(matches!(
        StarDist::post(
            raw(&tiny_model([2, 2]), image(&pixels, w, h), &params).unwrap(),
            plan,
            &params,
            &mut [0; 5],
        ),
        Err(NnError::InvalidInput(_))
    ));
}
