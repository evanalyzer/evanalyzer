//! Smoke test against a real Cellpose-SAM export (`tests/models/cellpose_sam.pt`).
//!
//! Not checked into git - at ~1.2GB it's too large to commit (see
//! `.gitignore`). Generate it locally first:
//!   pip install "cellpose>=4" torch
//!   python docs/convert_cellpose.py --output crates/core/tests/models/cellpose_sam.pt
//!
//! Ignored by default even once the file exists: loading a ~1.2GB ViT-L model
//! and running a forward pass takes real time and isn't something every
//! `cargo test` invocation should pay for. Run explicitly with:
//!   cargo test -p evanalyzer_core --features ai --test cellpose_sam_real_model -- --ignored

#![cfg(feature = "ai")]

use tch::{CModule, Device, Kind, Tensor};

#[test]
#[ignore]
fn real_cellpose_sam_export_loads_and_runs_at_the_traced_tile_size() {
    let device = Device::cuda_if_available();
    let model = CModule::load_on_device("tests/models/cellpose_sam.pt", device).expect(
        "tests/models/cellpose_sam.pt is missing or failed to load - see this file's \
         module doc comment for how to generate it",
    );

    let input = Tensor::zeros([1, 2, 256, 256], (Kind::Float, device));
    let output = model
        .forward_ts(&[input])
        .expect("forward pass on a zero tile must not fail");

    assert_eq!(output.size(), vec![1, 3, 256, 256]);
    assert_eq!(output.kind(), Kind::Float);
}
