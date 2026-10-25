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
