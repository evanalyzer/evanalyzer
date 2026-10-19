//! Finding the engine: the CUDA driver probe and the plugin loader.

pub(crate) mod cuda_probe;
// With `torch` linked in too (the analysis' tests) the engine is that one,
// and the loader only runs in its own tests.
#[cfg(feature = "torch-plugin")]
#[cfg_attr(feature = "torch", allow(dead_code))]
pub(crate) mod plugin;
