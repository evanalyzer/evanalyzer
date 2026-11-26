//! # system
//!
//! What this build and host machine can do - the image formats the readers
//! accept and the host's diagnostics. Read by `LocalBackend`
//! (`Backend::system_info`/`image_formats`); front ends ask the backend, so
//! in remote mode the worker answers for its own machine.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-10
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

/// File extensions (lowercase, no dot) the image readers accept. Depends on
/// which reader features `evanalyzer_core` was built with.
pub use evanalyzer_core::supported_image_formats;

/// CPU cores and total RAM in bytes EVAnalyzer may use - within the
/// user's [`ResourceLimits`].
pub fn cpu_ram_diagnostics() -> (usize, u64) {
    evanalyzer_core::cpu_ram_diagnostics()
}

/// How much of the computer EVAnalyzer may use (`--max-threads`,
/// `--max-ram`, Settings › Performance) - set once at startup.
pub use evanalyzer_core::memory::{ResourceLimits, set_resource_limits};

/// The GPUs and which ones AI models run on. Slow on first call (opens the
/// discrete GPUs), so call it off the UI thread.
pub fn gpus() -> Vec<crate::backend::Gpu> {
    evanalyzer_core::gpus()
        .into_iter()
        .map(|gpu| crate::backend::Gpu {
            name: gpu.name,
            discrete: gpu.discrete,
            used: gpu.used,
        })
        .collect()
}
