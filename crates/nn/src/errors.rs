//! # errors
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NnError {
    #[error("Format mismatch: expected {expected}, found {found}")]
    FormatMismatch { expected: String, found: String },

    #[error("IO Error: {0}")]
    StdIo(#[from] std::io::Error),

    #[error("XML error: {0}")]
    Xml(String),

    #[error("JSON error: {0}")]
    JsonParsingError(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Generic error: {0}")]
    ValidationError(String),
}
