use ndarray::ShapeError;
use std::num::ParseIntError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InternalErrors {
    #[error("Format mismatch: expected {expected}, found {found}")]
    FormatMismatch { expected: String, found: String },

    #[error("IO Error: {0}")]
    StdIo(#[from] std::io::Error),

    #[error("XML error: {0}")]
    Xml(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Generic error: {0}")]
    Generic(String),

    #[error("Kornia error: {0}")]
    Kornia(String),

    #[error("Generic error: {0}")]
    ParseError(String),

    #[error("Generic error: {0}")]
    JvmError(String),

    #[error("Generic error: {0}")]
    ImageReadError(String),

    #[error("Generic error: {0}")]
    ParseInt(String),

    #[error("Generic error: {0}")]
    AllocationError(String),

    #[error("Cache error: {0}")]
    CacheMiss(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Generic error: {0}")]
    ParseIntError(#[from] ParseIntError),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ShapeError),

    #[error("Generic error: {0}")]
    ValidationError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Too many objects")]
    TooManyObjects(String),

    #[error("Cancelled")]
    Cancelled,
}

// Add manual ones
impl From<quick_xml::Error> for InternalErrors {
    fn from(err: quick_xml::Error) -> Self {
        InternalErrors::Xml(err.to_string())
    }
}

// Add manual ones
impl From<&str> for InternalErrors {
    fn from(err: &str) -> Self {
        InternalErrors::Internal(err.to_string())
    }
}

impl InternalErrors {
    pub fn from_kornia<E: std::fmt::Debug>(err: E) -> Self {
        InternalErrors::Kornia(format!("{:?}", err))
    }
}
