//! # make_bench_models
//!
//! Writes the stand-in Cellpose models of the benchmarks (BENCHMARK.md):
//! `<dir>/cellpose_light.pt` (next to no compute) and
//! `<dir>/cellpose_heavy.pt` (6 convolution layers).
//!
//!   cargo run --release -p evanalyzer_ai --features test-support \
//!     --example make_bench_models -- <dir>
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-19
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_ai::torch::test_support::bench_cellpose_model;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("output folder"));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, layers) in [("cellpose_light.pt", 0), ("cellpose_heavy.pt", 6)] {
        let (_tmp, path) = bench_cellpose_model(layers);
        std::fs::copy(path, dir.join(name)).unwrap();
        println!("{}", dir.join(name).display());
    }
}
