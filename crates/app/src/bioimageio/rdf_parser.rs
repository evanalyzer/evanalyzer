//! Parsing of bioimage.io `rdf.yaml` files into an [`RdfModel`].

use std::fmt;
use std::path::Path;

use super::rdf_model::RdfModel;

/// Errors raised while reading or parsing an RDF file.
#[derive(Debug)]
pub enum RdfError {
    /// The RDF file could not be read from disk.
    Io(std::io::Error),
    /// The file is not valid YAML / does not match the expected structure.
    Yaml(serde_yaml::Error),
    /// The RDF is valid but describes something other than a model.
    NotAModel { kind: String },
}

impl fmt::Display for RdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RdfError::Io(e) => write!(f, "failed to read RDF file: {e}"),
            RdfError::Yaml(e) => write!(f, "failed to parse rdf.yaml: {e}"),
            RdfError::NotAModel { kind } => {
                write!(f, "RDF describes a '{kind}', not a model")
            }
        }
    }
}

impl std::error::Error for RdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RdfError::Io(e) => Some(e),
            RdfError::Yaml(e) => Some(e),
            RdfError::NotAModel { .. } => None,
        }
    }
}

impl From<std::io::Error> for RdfError {
    fn from(e: std::io::Error) -> Self {
        RdfError::Io(e)
    }
}

impl From<serde_yaml::Error> for RdfError {
    fn from(e: serde_yaml::Error) -> Self {
        RdfError::Yaml(e)
    }
}

/// Parses an RDF model from a YAML string.
pub fn parse_str(yaml: &str) -> Result<RdfModel, RdfError> {
    let model: RdfModel = serde_yaml::from_str(yaml)?;
    if !model.is_model() {
        return Err(RdfError::NotAModel {
            kind: model.kind.clone(),
        });
    }
    Ok(model)
}

/// Reads and parses an RDF model from a `rdf.yaml` file on disk.
pub fn parse_file(path: &Path) -> Result<RdfModel, RdfError> {
    let text = std::fs::read_to_string(path)?;
    parse_str(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bioimageio::rdf_model::ModelKind;

    #[test]
    fn parses_04_cellpose_style() {
        let yaml = r#"
format_version: "0.4.10"
type: model
name: Happy Elephant
tags: [cellpose, segmentation]
weights:
  pytorch_state_dict:
    source: weights.pth
inputs:
  - name: raw
    axes: bcyx
    shape: [1, 2, 256, 256]
"#;
        let m = parse_str(yaml).unwrap();
        assert!(m.is_model());
        assert_eq!(m.detect_kind(), ModelKind::Cellpose);
        assert_eq!(m.input_channels(), Some(2));
        assert!(m.has_state_dict());
        assert!(m.torchscript_source().is_none());
    }

    #[test]
    fn parses_05_channel_names() {
        let yaml = r#"
format_version: "0.5.3"
type: model
name: Affable Shark boundary model
weights:
  torchscript:
    source: model.pt
inputs:
  - id: raw
    axes:
      - type: batch
      - type: channel
        channel_names: [c0]
      - type: space
        id: y
      - type: space
        id: x
outputs:
  - id: pred
    axes:
      - type: batch
      - type: channel
        channel_names: [mask, boundary]
      - type: space
        id: y
      - type: space
        id: x
"#;
        let m = parse_str(yaml).unwrap();
        assert_eq!(m.input_channels(), Some(1));
        assert_eq!(m.output_channels(), Some(2));
        assert!(m.looks_like_boundary_model());
        assert_eq!(m.torchscript_source(), Some("model.pt"));
    }

    #[test]
    fn rejects_non_model() {
        let yaml = "type: dataset\nname: foo\n";
        assert!(matches!(parse_str(yaml), Err(RdfError::NotAModel { .. })));
    }
}
