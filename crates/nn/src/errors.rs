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

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}
