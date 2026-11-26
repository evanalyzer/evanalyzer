//! # errors
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The messages are shown to the user.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NnError {
    /// The file is no model of ours, or of an architecture we don't run.
    #[error("Unsupported model: {0}")]
    UnsupportedArchitecture(String),

    /// The file says it is a supported model, but its contents don't fit
    /// (missing or misshaped tensors, sizes out of range).
    #[error("Invalid model file: {0}")]
    InvalidModelFile(String),

    /// Input that doesn't fit the model (channels, sizes, empty samples).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A training was cancelled.
    #[error("Cancelled")]
    Cancelled,

    /// The device failed (lost, out of memory) and no fallback was possible.
    #[error("Device error: {0}")]
    Device(String),

    /// The GPU ran out of memory for this input: smaller pieces (a model
    /// seeing a limited neighbourhood) or the CPU may still run it - the GPU
    /// itself is fine.
    #[error("Out of GPU memory: {0}")]
    GpuOutOfMemory(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Whether a device's error text says it ran out of memory (wgpu: "out of
/// device memory allocating N bytes").
pub(crate) fn is_out_of_memory(message: &str) -> bool {
    let message = message.to_lowercase();
    message.contains("out of device memory") || message.contains("out of memory")
}

/// A device's error as [`NnError::GpuOutOfMemory`] or [`NnError::Device`].
pub(crate) fn device_error(message: String) -> NnError {
    if is_out_of_memory(&message) {
        NnError::GpuOutOfMemory(message)
    } else {
        NnError::Device(message)
    }
}

/// `tensor`'s values back on the host - a failure of the device (e.g. out
/// of memory, which shows when the result is read) as an error, not a
/// panic.
pub(crate) fn read_f32<const D: usize>(
    tensor: burn::tensor::Tensor<D>,
) -> Result<Vec<f32>, NnError> {
    tensor
        .try_into_data()
        .map_err(|e| device_error(format!("{e:?}")))?
        .try_to_vec::<f32>()
        .map_err(|e| NnError::Internal(format!("{e:?}")))
}
