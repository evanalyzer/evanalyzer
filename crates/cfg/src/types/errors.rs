//! # errors
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-06-05
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::types::ids::{PipelineId, StepId};
use ndarray::ShapeError;
use std::num::ParseIntError;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InternalErrors {
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

    /// The error of a pipeline step - where it happened, for front ends
    /// that point at the step ([`Self::step`]). Shows as the error itself.
    #[error("{source}")]
    InStep {
        pipeline: PipelineId,
        step: StepId,
        source: Box<InternalErrors>,
    },

    /// The error of the analysis of one image, named by its path.
    #[error("{}: {source}", path.display())]
    InImage {
        path: PathBuf,
        source: Box<InternalErrors>,
    },
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

impl From<serde_json::Error> for InternalErrors {
    fn from(err: serde_json::Error) -> Self {
        InternalErrors::JsonParsingError(err.to_string())
    }
}

impl InternalErrors {
    pub fn from_kornia<E: std::fmt::Debug>(err: E) -> Self {
        InternalErrors::Kornia(format!("{:?}", err))
    }

    /// `self` as the error of step `step` of pipeline `pipeline` - unless
    /// it's a cancel (front ends tell those apart by their kind), already
    /// says where it happened, or the step is unassigned.
    pub fn in_step(self, pipeline: PipelineId, step: StepId) -> Self {
        match self {
            InternalErrors::Cancelled => self,
            _ if step == StepId::UNASSIGNED || self.step().is_some() => self,
            source => InternalErrors::InStep {
                pipeline,
                step,
                source: Box::new(source),
            },
        }
    }

    /// `self` as the error of the analysis of the image at `path` - a
    /// cancel stays a cancel.
    pub fn in_image(self, path: PathBuf) -> Self {
        match self {
            InternalErrors::Cancelled => self,
            source => InternalErrors::InImage {
                path,
                source: Box::new(source),
            },
        }
    }

    /// The pipeline step the error happened in, if a step said so.
    pub fn step(&self) -> Option<(PipelineId, StepId)> {
        match self {
            InternalErrors::InStep { pipeline, step, .. } => Some((*pipeline, *step)),
            InternalErrors::InImage { source, .. } => source.step(),
            _ => None,
        }
    }

    /// What went wrong, for people: the innermost error's own text,
    /// without the kind ("Generic error: ") and without where it happened
    /// (the step, the image) - a front end shows that on its own.
    pub fn message(&self) -> String {
        match self {
            InternalErrors::InStep { source, .. } | InternalErrors::InImage { source, .. } => {
                source.message()
            }
            InternalErrors::Xml(m)
            | InternalErrors::JsonParsingError(m)
            | InternalErrors::Internal(m)
            | InternalErrors::Generic(m)
            | InternalErrors::Kornia(m)
            | InternalErrors::ParseError(m)
            | InternalErrors::JvmError(m)
            | InternalErrors::ImageReadError(m)
            | InternalErrors::ParseInt(m)
            | InternalErrors::AllocationError(m)
            | InternalErrors::CacheMiss(m)
            | InternalErrors::Io(m)
            | InternalErrors::ValidationError(m)
            | InternalErrors::InvalidArgument(m) => m.clone(),
            other => other.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_quick_xml_error_wraps_the_message_in_xml_variant() {
        let xml_err =
            quick_xml::Error::Escape(quick_xml::escape::EscapeError::UnterminatedEntity(0..1));
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

    /// A step's error says where it happened, shows as itself, and keeps
    /// that through the image's path; a cancel and an unassigned step stay
    /// as they are.
    #[test]
    fn a_step_error_keeps_where_it_happened() {
        let e = InternalErrors::Generic("AI models need the AI plugin".into())
            .in_step(PipelineId(2), StepId(5));
        assert_eq!(e.step(), Some((PipelineId(2), StepId(5))));
        assert_eq!(e.to_string(), "Generic error: AI models need the AI plugin");
        assert_eq!(e.message(), "AI models need the AI plugin");

        let e = e.in_image(PathBuf::from("a.tif"));
        assert_eq!(e.step(), Some((PipelineId(2), StepId(5))));
        assert_eq!(
            e.to_string(),
            "a.tif: Generic error: AI models need the AI plugin"
        );
        assert_eq!(e.message(), "AI models need the AI plugin");

        // The first step to say where wins (an inner one, re-wrapped).
        let again = e.in_step(PipelineId(9), StepId(9));
        assert_eq!(again.step(), Some((PipelineId(2), StepId(5))));

        let cancel = InternalErrors::Cancelled
            .in_step(PipelineId(2), StepId(5))
            .in_image(PathBuf::from("a.tif"));
        assert!(matches!(cancel, InternalErrors::Cancelled));

        let unassigned =
            InternalErrors::Generic("x".into()).in_step(PipelineId(2), StepId::UNASSIGNED);
        assert_eq!(unassigned.step(), None);
    }
}
