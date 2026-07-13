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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_quick_xml_error_wraps_the_message_in_xml_variant() {
        let xml_err = quick_xml::Error::Escape(quick_xml::escape::EscapeError::UnterminatedEntity(
            0..1,
        ));
        let err: InternalErrors = xml_err.into();
        assert!(matches!(err, InternalErrors::Xml(_)));
        assert!(err.to_string().contains("XML error"));
    }

    #[test]
    fn from_str_wraps_the_message_in_internal_variant() {
        let err: InternalErrors = "something broke".into();
        let InternalErrors::Internal(msg) = err else {
            panic!("expected Internal variant");
        };
        assert_eq!(msg, "something broke");
    }

    #[test]
    fn from_kornia_formats_the_debug_representation() {
        let err = InternalErrors::from_kornia("some kornia failure");
        let InternalErrors::Kornia(msg) = err else {
            panic!("expected Kornia variant");
        };
        assert!(msg.contains("some kornia failure"));
    }

    #[test]
    fn display_messages_include_the_wrapped_value() {
        assert_eq!(
            InternalErrors::FormatMismatch {
                expected: "png".to_string(),
                found: "jpg".to_string()
            }
            .to_string(),
            "Format mismatch: expected png, found jpg"
        );
        assert_eq!(
            InternalErrors::InvalidArgument("bad flag".to_string()).to_string(),
            "Invalid argument: bad flag"
        );
        assert_eq!(InternalErrors::Cancelled.to_string(), "Cancelled");
        assert_eq!(
            InternalErrors::TooManyObjects("well_A1".to_string()).to_string(),
            "Too many objects"
        );
    }
}
