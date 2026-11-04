//! # test_support
//!
//! Test helpers for exercising the real TorchScript model-loading and
//! inference path - in this crate's tests and, through the `test-support`
//! feature, in the analysis' `execute()` tests of its AI steps - without
//! needing Python/PyTorch or a checked-in `.pt` fixture file.
//!
//! `tch::CModule::create_by_tracing` builds a genuine, saveable TorchScript
//! module purely from a Rust closure over tensor ops - the resulting file
//! round-trips through `CModule::load_on_device` exactly like a real
//! `torch.jit.trace` export, so it exercises the real load/inference code
//! path instead of only the pure-Rust helpers downstream of it.
//!
//! One real limitation: `create_by_tracing`'s traced `forward` method can
//! only have a single return value - `.save()` rejects a closure that
//! returns more than one tensor ("Exportable methods must have a single
//! return value"). So this can build models matching the "single tensor
//! output" convention each of the three commands supports (a lone
//! `[1, C, H, W]` tensor), but not the "output wrapped in a tuple/list of
//! several tensors" convention some real exports use - that branch stays
//! covered only by manual review, not a test.
//!
//! A second pitfall: a tensor *created* inside the traced closure (e.g.
//! `Tensor::from_slice(&[...])`) is saved as zeros - its values are not part
//! of the trace. Build constant outputs from the input with plain numbers
//! instead (`x.mean(..) * 0.0 + v`), which the trace keeps (see
//! `yolov5`'s `centre_box_model`).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-08-21
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use tch::{CModule, Device, Kind, Tensor};

/// libtorch's tensors, for the models the tests trace.
pub use tch;

/// Traces `op` over a `[1, in_channels, height, width]` zero input and saves
/// the resulting TorchScript module to a temp file, returning the `TempDir`
/// guard (keep it alive for as long as `path` is used - it deletes the file
/// on drop) alongside the model path.
///
/// `op` is traced, not interpreted: the saved module always replays the
/// exact tensor operations recorded against the example input, so it
/// generalizes to any *runtime* input of the same shape - callers are free
/// to feed it real per-pixel data at `execute()` time.
pub fn trace_and_save_model(
    in_channels: i64,
    height: i64,
    width: i64,
    mut op: impl FnMut(&Tensor) -> Tensor,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let device = Device::Cpu;
    let example = Tensor::zeros([1, in_channels, height, width], (Kind::Float, device));
    let module = CModule::create_by_tracing("m", "forward", &[example], &mut |xs: &[Tensor]| {
        vec![op(&xs[0])]
    })
    .expect("tracing a test model must not fail");

    let dir = tempfile::tempdir().expect("failed to create a temp dir for a traced test model");
    let path = dir.path().join("model.pt");
    module
        .save(&path)
        .expect("failed to save a traced test model");
    (dir, path)
}

/// A stand-in Cellpose model for benchmarks, traced at its 256 x 256 tile
/// size with 2 input channels. `[1, 3, H, W]` out: no flows, a cell logit
/// that follows the image. `conv_layers` 32-channel 3x3 convolutions run
/// first (their result barely touches the output), so the model does real
/// work like a network would; `0` = next to no compute (the cost around
/// the model).
pub fn bench_cellpose_model(conv_layers: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    trace_and_save_model(2, 256, 256, |x| {
        let image = x.narrow(1, 0, 1);
        let mut cell_logit = &image * 20.0 - 10.0;
        if conv_layers > 0 {
            // Weights from the input with plain numbers (a tensor made in
            // the closure would be saved as zeros, see the module doc).
            let weight = (x.mean(Kind::Float) * 0.0 + 0.01).expand([32, 32, 3, 3], false);
            let mut features = x.repeat([1, 16, 1, 1]);
            for _ in 0..conv_layers {
                features = features
                    .conv2d(&weight, None::<Tensor>, [1, 1], [1, 1], [1, 1], 1)
                    .relu();
            }
            cell_logit = cell_logit + features.narrow(1, 0, 1) * 1e-6;
        }
        let zeros = image.zeros_like();
        Tensor::cat(&[zeros.shallow_clone(), zeros, cell_logit], 1)
    })
}

/// A weight of `shape` around `value` (0.5-1.5 times it), irregular - so a
/// pixel computed from other context than on the whole image shows - made
/// by traced operations (`arange`, `sin`) on `x`: a tensor made in a traced
/// closure from data would be saved as zeros (see the module doc).
fn constant_weight(x: &Tensor, shape: [i64; 4], value: f64) -> Tensor {
    let count: i64 = shape.iter().product();
    let wave = (Tensor::arange(count, (Kind::Float, x.device())) * 0.7853 + 0.3).sin();
    (x.mean(Kind::Float) * 0.0 + value + wave * (0.5 * value)).reshape(shape)
}

/// Two 3x3 convolutions with ReLU from `in_channels` to `out_channels` - one
/// level of a U-Net.
fn conv_block(x: &Tensor, seed: &Tensor, in_channels: i64, out_channels: i64) -> Tensor {
    let w1 = constant_weight(
        seed,
        [out_channels, in_channels, 3, 3],
        1.0 / (9 * in_channels) as f64,
    );
    let w2 = constant_weight(
        seed,
        [out_channels, out_channels, 3, 3],
        1.0 / (9 * out_channels) as f64,
    );
    x.conv2d(&w1, None::<Tensor>, [1, 1], [1, 1], [1, 1], 1)
        .relu()
        .conv2d(&w2, None::<Tensor>, [1, 1], [1, 1], [1, 1], 1)
        .relu()
}

/// A U-Net of `depth` poolings from `base` filters (doubled per level), on
/// `x` of `in_channels`: the encoder's outputs are kept for the skip
/// connections, the decoder upsamples (transposed 2x2 convolution) and
/// concatenates. Only size-independent operations, so the traced model runs
/// on any side that is a multiple of 2^`depth`.
fn unet(x: &Tensor, seed: &Tensor, in_channels: i64, base: i64, depth: usize) -> Tensor {
    let mut skips = Vec::new();
    let mut features = x.shallow_clone();
    let mut channels = in_channels;
    for level in 0..depth {
        let out = base << level;
        features = conv_block(&features, seed, channels, out);
        channels = out;
        skips.push(features.shallow_clone());
        features = features.max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], false);
    }
    features = conv_block(&features, seed, channels, base << depth);
    channels = base << depth;
    for level in (0..depth).rev() {
        let out = base << level;
        let up = constant_weight(seed, [channels, out, 2, 2], 1.0 / (4 * channels) as f64);
        features =
            features.conv_transpose2d(&up, None::<Tensor>, [2, 2], [0, 0], [0, 0], 1, [1, 1]);
        features = Tensor::cat(&[skips.pop().unwrap(), features], 1);
        features = conv_block(&features, seed, 2 * out, out);
        channels = out;
    }
    features
}

/// A stand-in StarDist model for RAM benchmarks, laid out as StarDist's
/// `2D_versatile_fluo` (its `config.json`): 1 input channel, two 3x3
/// convolutions of 32 filters at full resolution, a 2x2 pooling (grid
/// 2x2), a U-Net of depth 3 from 32 filters, a 3x3 convolution of 128
/// filters, then the object probability (sigmoid) and 32 ray distances -
/// one `[1, 33, H/2, W/2]` tensor. Constant weights: the activations - and
/// so the RAM - are those of the real network, the values are not.
pub fn bench_stardist_model() -> (tempfile::TempDir, std::path::PathBuf) {
    trace_and_save_model(1, 64, 64, |x| {
        let seed = x.shallow_clone();
        let full = conv_block(x, &seed, 1, 32);
        let grid = full.max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], false);
        let features = unet(&grid, &seed, 32, 32, 3);
        let after = constant_weight(&seed, [128, 32, 3, 3], 1.0 / 288.0);
        let features = features
            .conv2d(&after, None::<Tensor>, [1, 1], [1, 1], [1, 1], 1)
            .relu();
        let prob = features
            .conv2d(
                &constant_weight(&seed, [1, 128, 1, 1], 1.0 / 128.0),
                None::<Tensor>,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )
            .sigmoid();
        let dist = features
            .conv2d(
                &constant_weight(&seed, [32, 128, 1, 1], 1.0 / 128.0),
                None::<Tensor>,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )
            .relu();
        Tensor::cat(&[prob, dist], 1)
    })
}

/// A stand-in U-Net for RAM benchmarks, laid out as the 2D U-Nets of the
/// bioimage.io model zoo: 1 input channel, depth 4 from `base` filters
/// (32 or 64), 2 output channels (foreground, boundary) from a 1x1
/// convolution. Constant weights (see [`bench_stardist_model`]).
pub fn bench_unet_model(base: i64) -> (tempfile::TempDir, std::path::PathBuf) {
    trace_and_save_model(1, 64, 64, |x| {
        let seed = x.shallow_clone();
        let features = unet(x, &seed, 1, base, 4);
        features
            .conv2d(
                &constant_weight(&seed, [2, base, 1, 1], 1.0 / base as f64),
                None::<Tensor>,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )
            .sigmoid()
    })
}

/// A YOLOv5 detection model (single output) that always reports one
/// 100x100 box of class 1 at the window centre, traced at its 640² window.
pub fn bench_yolo_box_model() -> (tempfile::TempDir, std::path::PathBuf) {
    trace_and_save_model(3, 640, 640, |x| {
        // Built from the input with plain numbers: a tensor created inside
        // the traced closure (e.g. `Tensor::from_slice`) is saved as zeros.
        let one = x.mean(Kind::Float) * 0.0 + 1.0;
        let values = [320.0, 320.0, 100.0, 100.0, 0.9, 0.1, 0.9];
        let columns: Vec<Tensor> = values.iter().map(|&v| &one * v).collect();
        Tensor::stack(&columns, 0).reshape([1, 1, 7])
    })
}

/// A StarDist model with the outputs of a well-fitting one on bright cells
/// of `radius` px (grid 2, 32 rays): the object probability high on bright
/// pixels (above `threshold`), every ray `radius` long - for what the
/// analysis keeps of its answer (candidates, masks), which depends on the
/// outputs, not the network.
pub fn stardist_cells_model(threshold: f64, radius: f64) -> (tempfile::TempDir, std::path::PathBuf) {
    trace_and_save_model(1, 64, 64, |x| {
        let grid = x.avg_pool2d([2, 2], [2, 2], [0, 0], false, true, None::<i64>);
        let prob = ((&grid - threshold) * 0.01).sigmoid();
        let dist = (grid.zeros_like() + radius).repeat([1, 32, 1, 1]);
        Tensor::cat(&[prob, dist], 1)
    })
}
