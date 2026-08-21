//! Test-only helpers for exercising the real TorchScript model-loading and
//! inference path in `unet`/`cellpose`/`stardist`'s `execute()` tests, without
//! needing Python/PyTorch or a checked-in `.pt` fixture file.
//!
//! `tch::CModule::create_by_tracing` builds a genuine, saveable TorchScript
//! module purely from a Rust closure over tensor ops - the resulting file
//! round-trips through `CModule::load_on_device` exactly like a real
//! `torch.jit.trace` export, so it exercises the real load/inference code
//! path (previously untested - see `model_cache.rs`'s doc comment, written
//! before this technique was known to work here) instead of only the
//! pure-Rust helpers downstream of it.
//!
//! One real limitation: `create_by_tracing`'s traced `forward` method can
//! only have a single return value - `.save()` rejects a closure that
//! returns more than one tensor ("Exportable methods must have a single
//! return value"). So this can build models matching the "single tensor
//! output" convention each of the three commands supports (a lone
//! `[1, C, H, W]` tensor), but not the "output wrapped in a tuple/list of
//! several tensors" convention some real exports use - that branch stays
//! covered only by manual review, not a test.

#![cfg(test)]

use tch::{CModule, Device, Kind, Tensor};

/// Traces `op` over a `[1, in_channels, height, width]` zero input and saves
/// the resulting TorchScript module to a temp file, returning the `TempDir`
/// guard (keep it alive for as long as `path` is used - it deletes the file
/// on drop) alongside the model path.
///
/// `op` is traced, not interpreted: the saved module always replays the
/// exact tensor operations recorded against the example input, so it
/// generalizes to any *runtime* input of the same shape - callers are free
/// to feed it real per-pixel data at `execute()` time.
pub(crate) fn trace_and_save_model(
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
