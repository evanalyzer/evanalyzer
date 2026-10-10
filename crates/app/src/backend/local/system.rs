//! What this build and host machine can do - the image formats the readers
//! accept and the host's diagnostics. Read by `LocalBackend`
//! (`Backend::system_info`/`image_formats`); front ends ask the backend, so
//! in remote mode the worker answers for its own machine.

/// File extensions (lowercase, no dot) the image readers accept. Depends on
/// which reader features `evanalyzer_core` was built with.
pub use evanalyzer_core::SUPPORTED_IMAGE_FORMATS;

/// Logical CPU cores and total RAM in bytes.
pub fn cpu_ram_diagnostics() -> (usize, u64) {
    evanalyzer_core::cpu_ram_diagnostics()
}

/// Whether a CUDA device is usable. Slow on first call (loads the driver and
/// creates a context), so call it off the UI thread.
pub fn cuda_is_available() -> bool {
    evanalyzer_core::cuda_is_available()
}
