//! Turns a parsed bioimage.io [`RdfModel`] into a ready-to-use AI segmentation
//! [`PipelineCommand`], filling parameters from the model card where possible.

use std::fmt;
use std::path::{Path, PathBuf};

use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_cfg::settings::pipeline_command_settings::{
    AiSegmentationUnetUNetOutputModeSettings, CellposeSettings, StardistSettings, UNetSettings,
};

use super::rdf_model::{ModelKind, RdfModel};
use super::rdf_parser::{self, RdfError};

/// Errors raised while turning an RDF into a command.
#[derive(Debug)]
pub enum ConfigureError {
    /// The RDF could not be read/parsed.
    Rdf(RdfError),
    /// The model ships no TorchScript weights, so it cannot be loaded directly.
    /// `has_state_dict` indicates whether a convertible `pytorch_state_dict` is
    /// available (see `docs/convert_cellpose.py`).
    NoTorchscriptWeights { name: String, has_state_dict: bool },
}

impl fmt::Display for ConfigureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigureError::Rdf(e) => write!(f, "{e}"),
            ConfigureError::NoTorchscriptWeights {
                name,
                has_state_dict,
            } => {
                write!(
                    f,
                    "model '{name}' ships no TorchScript weights, which EVAnalyzer requires"
                )?;
                if *has_state_dict {
                    write!(
                        f,
                        " — it only provides a pytorch_state_dict; convert it with \
                         docs/convert_cellpose.py first"
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigureError::Rdf(e) => Some(e),
            ConfigureError::NoTorchscriptWeights { .. } => None,
        }
    }
}

impl From<RdfError> for ConfigureError {
    fn from(e: RdfError) -> Self {
        ConfigureError::Rdf(e)
    }
}

/// The result of configuring a command from an RDF.
#[derive(Debug, Clone)]
pub struct ConfiguredModel {
    /// The configured pipeline command, ready to insert.
    pub command: PipelineCommand,
    /// The resolved path to the TorchScript weights the command will load.
    pub model_path: PathBuf,
    /// The detected model family.
    pub kind: ModelKind,
    /// Human-readable warnings the caller should surface to the user (remote
    /// weights, assumed defaults, required preprocessing, …).
    pub notes: Vec<String>,
}

/// Reads `rdf.yaml` at `path` and configures a command from it. The weight path
/// is resolved relative to the RDF file's directory.
pub fn configure_from_file(path: &Path) -> Result<ConfiguredModel, ConfigureError> {
    let rdf = rdf_parser::parse_file(path)?;
    configure(&rdf, path.parent())
}

/// Configures a command from an already-parsed RDF. `base_dir` is the directory
/// relative weight `source` paths are resolved against (typically the folder
/// containing `rdf.yaml`).
pub fn configure(rdf: &RdfModel, base_dir: Option<&Path>) -> Result<ConfiguredModel, ConfigureError> {
    let source = rdf
        .torchscript_source()
        .ok_or_else(|| ConfigureError::NoTorchscriptWeights {
            name: rdf.name.clone(),
            has_state_dict: rdf.has_state_dict(),
        })?;

    let mut notes = Vec::new();
    let model_path = resolve_source(source, base_dir, &mut notes);
    let kind = rdf.detect_kind();

    let command = match kind {
        ModelKind::Cellpose => {
            let channels = rdf.input_channels().unwrap_or_else(|| {
                notes.push(
                    "Could not read the input channel count from the RDF; defaulting to 2 \
                     (standard Cellpose). Adjust 'input_channels' if the model differs."
                        .to_string(),
                );
                2
            });
            PipelineCommand::Cellpose(CellposeSettings {
                model_path: model_path.clone(),
                input_channels: channels.clamp(1, 8) as i32,
                ..Default::default()
            })
        }
        ModelKind::StarDist => {
            let mut s = StardistSettings {
                model_path: model_path.clone(),
                ..Default::default()
            };
            if let Some(prob) = config_f32(rdf, &["stardist", "thresholds", "prob"]) {
                s.probability_threshold = prob.clamp(0.0, 1.0);
            }
            if let Some(nms) = config_f32(rdf, &["stardist", "thresholds", "nms"]) {
                s.nms_threshold = nms.clamp(0.0, 1.0);
            }
            PipelineCommand::Stardist(s)
        }
        ModelKind::UNet => {
            let mut s = UNetSettings {
                model_path: model_path.clone(),
                ..Default::default()
            };
            let out_channels = rdf.output_channels();
            if rdf.looks_like_boundary_model() && out_channels == Some(2) {
                // Mask + boundary heads: use them independently and carve the
                // boundary out so a following ConnectedComponents splits objects.
                s.output_mode = AiSegmentationUnetUNetOutputModeSettings::IndependentChannels;
                s.foreground_channel = 0;
                s.boundary_channel = 1;
                notes.push(
                    "Detected a boundary-aware model: channel 0 is treated as foreground and \
                     channel 1 as the boundary. Add a ConnectedComponents step afterwards to \
                     separate touching objects."
                        .to_string(),
                );
            } else if let Some(c) = out_channels {
                // Softmax classes: the foreground is conventionally the last channel.
                s.foreground_channel = (c - 1).clamp(0, 16) as i32;
                notes.push(format!(
                    "Assuming a {c}-class softmax output with foreground in channel {}; \
                     verify 'foreground_channel'/'output_mode' against the model.",
                    s.foreground_channel
                ));
            } else {
                notes.push(
                    "Could not determine the output layout from the RDF; using U-Net defaults. \
                     Check 'output_mode' and 'foreground_channel'."
                        .to_string(),
                );
            }
            PipelineCommand::UNet(s)
        }
    };

    if rdf.has_input_preprocessing() {
        notes.push(
            "The model declares input preprocessing (e.g. normalization) that EVAnalyzer does \
             not apply automatically; add equivalent contrast/normalize steps before this command."
                .to_string(),
        );
    }

    Ok(ConfiguredModel {
        command,
        model_path,
        kind,
        notes,
    })
}

/// Resolves a weight `source` into a usable path. Remote sources are returned
/// verbatim with a note, since downloading is out of scope for this layer.
fn resolve_source(source: &str, base_dir: Option<&Path>, notes: &mut Vec<String>) -> PathBuf {
    if source.starts_with("http://") || source.starts_with("https://") {
        notes.push(format!(
            "Model weights are remote ({source}); download them and point the model path at the \
             local file."
        ));
        return PathBuf::from(source);
    }
    match base_dir {
        Some(dir) => dir.join(source),
        None => PathBuf::from(source),
    }
}

/// Navigates the free-form `config` block by key path and reads a float.
fn config_f32(rdf: &RdfModel, path: &[&str]) -> Option<f32> {
    let mut cur = &rdf.config;
    for key in path {
        cur = cur.get(*key)?;
    }
    cur.as_f64().map(|v| v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bioimageio::rdf_parser::parse_str;

    #[test]
    fn configures_cellpose_with_channels() {
        let yaml = r#"
type: model
name: cellpose-cyto
tags: [cellpose]
weights:
  torchscript:
    source: model.pt
inputs:
  - axes: bcyx
    shape: [1, 2, 256, 256]
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, Some(Path::new("/models/elephant"))).unwrap();
        assert_eq!(cfg.kind, ModelKind::Cellpose);
        assert_eq!(cfg.model_path, PathBuf::from("/models/elephant/model.pt"));
        match cfg.command {
            PipelineCommand::Cellpose(s) => assert_eq!(s.input_channels, 2),
            other => panic!("expected Cellpose, got {other:?}"),
        }
    }

    #[test]
    fn boundary_unet_uses_independent_channels() {
        let yaml = r#"
type: model
name: affable-shark boundary model
weights:
  torchscript:
    source: model.pt
outputs:
  - axes:
      - type: batch
      - type: channel
        channel_names: [mask, boundary]
      - type: space
        id: y
      - type: space
        id: x
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        match cfg.command {
            PipelineCommand::UNet(s) => {
                assert_eq!(s.foreground_channel, 0);
                assert_eq!(s.boundary_channel, 1);
                assert!(matches!(
                    s.output_mode,
                    AiSegmentationUnetUNetOutputModeSettings::IndependentChannels
                ));
            }
            other => panic!("expected UNet, got {other:?}"),
        }
    }

    #[test]
    fn missing_torchscript_is_an_error() {
        let yaml = r#"
type: model
name: weights-only
weights:
  pytorch_state_dict:
    source: weights.pth
"#;
        let rdf = parse_str(yaml).unwrap();
        match configure(&rdf, None) {
            Err(ConfigureError::NoTorchscriptWeights {
                has_state_dict, ..
            }) => assert!(has_state_dict),
            other => panic!("expected NoTorchscriptWeights, got {other:?}"),
        }
    }

    #[test]
    fn configures_stardist_with_thresholds_from_config() {
        let yaml = r#"
type: model
name: stardist-nuclei
weights:
  torchscript:
    source: model.pt
config:
  stardist:
    thresholds:
      prob: 0.7
      nms: 0.2
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        assert_eq!(cfg.kind, ModelKind::StarDist);
        match cfg.command {
            PipelineCommand::Stardist(s) => {
                assert_eq!(s.probability_threshold, 0.7);
                assert_eq!(s.nms_threshold, 0.2);
            }
            other => panic!("expected Stardist, got {other:?}"),
        }
    }

    #[test]
    fn stardist_thresholds_are_clamped_to_zero_one() {
        let yaml = r#"
type: model
name: stardist-weird
weights:
  torchscript:
    source: model.pt
config:
  stardist:
    thresholds:
      prob: 1.5
      nms: -0.5
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        match cfg.command {
            PipelineCommand::Stardist(s) => {
                assert_eq!(s.probability_threshold, 1.0);
                assert_eq!(s.nms_threshold, 0.0);
            }
            other => panic!("expected Stardist, got {other:?}"),
        }
    }

    #[test]
    fn stardist_without_a_thresholds_block_keeps_the_command_defaults() {
        let yaml = r#"
type: model
name: stardist-minimal
weights:
  torchscript:
    source: model.pt
config:
  stardist: {}
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        let default = StardistSettings::default();
        match cfg.command {
            PipelineCommand::Stardist(s) => {
                assert_eq!(s.probability_threshold, default.probability_threshold);
                assert_eq!(s.nms_threshold, default.nms_threshold);
            }
            other => panic!("expected Stardist, got {other:?}"),
        }
    }

    #[test]
    fn generic_unet_softmax_output_puts_foreground_in_the_last_channel_and_notes_it() {
        let yaml = r#"
type: model
name: plain-unet
weights:
  torchscript:
    source: model.pt
outputs:
  - axes:
      - type: batch
      - type: channel
        channel_names: [background, foreground, extra]
      - type: space
        id: y
      - type: space
        id: x
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        match cfg.command {
            PipelineCommand::UNet(s) => {
                assert_eq!(s.foreground_channel, 2, "foreground must default to the last of 3 channels");
                assert!(matches!(
                    s.output_mode,
                    AiSegmentationUnetUNetOutputModeSettings::SoftmaxClasses
                ));
            }
            other => panic!("expected UNet, got {other:?}"),
        }
        assert!(cfg.notes.iter().any(|n| n.contains("softmax")));
    }

    #[test]
    fn unet_with_no_detectable_output_layout_uses_defaults_and_notes_it() {
        let yaml = r#"
type: model
name: mystery-unet
weights:
  torchscript:
    source: model.pt
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        let default = UNetSettings::default();
        match cfg.command {
            PipelineCommand::UNet(s) => assert_eq!(s.foreground_channel, default.foreground_channel),
            other => panic!("expected UNet, got {other:?}"),
        }
        assert!(cfg.notes.iter().any(|n| n.contains("Could not determine the output layout")));
    }

    #[test]
    fn cellpose_defaults_to_two_channels_and_notes_it_when_unreadable() {
        let yaml = r#"
type: model
name: cellpose-no-inputs
tags: [cellpose]
weights:
  torchscript:
    source: model.pt
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        match cfg.command {
            PipelineCommand::Cellpose(s) => assert_eq!(s.input_channels, 2),
            other => panic!("expected Cellpose, got {other:?}"),
        }
        assert!(cfg.notes.iter().any(|n| n.contains("defaulting to 2")));
    }

    #[test]
    fn cellpose_channel_count_is_clamped_to_the_valid_range() {
        let yaml = r#"
type: model
name: cellpose-many-channels
tags: [cellpose]
weights:
  torchscript:
    source: model.pt
inputs:
  - axes: bcyx
    shape: [1, 20, 256, 256]
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        match cfg.command {
            PipelineCommand::Cellpose(s) => assert_eq!(s.input_channels, 8, "must clamp to the documented max of 8"),
            other => panic!("expected Cellpose, got {other:?}"),
        }
    }

    #[test]
    fn configure_notes_when_the_model_declares_input_preprocessing() {
        let yaml = r#"
type: model
name: cellpose-with-preprocessing
tags: [cellpose]
weights:
  torchscript:
    source: model.pt
inputs:
  - axes: bcyx
    shape: [1, 2, 256, 256]
    preprocessing:
      - name: zero_mean_unit_variance
"#;
        let rdf = parse_str(yaml).unwrap();
        let cfg = configure(&rdf, None).unwrap();
        assert!(cfg.notes.iter().any(|n| n.contains("input preprocessing")));
    }

    // ---- resolve_source ----

    #[test]
    fn resolve_source_leaves_remote_urls_verbatim_and_notes_them() {
        let mut notes = Vec::new();
        let path = resolve_source("https://example.com/model.pt", Some(Path::new("/models")), &mut notes);
        assert_eq!(path, PathBuf::from("https://example.com/model.pt"));
        assert!(notes.iter().any(|n| n.contains("remote")));
    }

    #[test]
    fn resolve_source_joins_a_relative_path_with_the_base_dir() {
        let mut notes = Vec::new();
        let path = resolve_source("model.pt", Some(Path::new("/models/elephant")), &mut notes);
        assert_eq!(path, PathBuf::from("/models/elephant/model.pt"));
        assert!(notes.is_empty());
    }

    #[test]
    fn resolve_source_uses_the_bare_source_when_there_is_no_base_dir() {
        let mut notes = Vec::new();
        let path = resolve_source("model.pt", None, &mut notes);
        assert_eq!(path, PathBuf::from("model.pt"));
    }

    // ---- config_f32 ----

    #[test]
    fn config_f32_returns_none_for_a_missing_key_path() {
        let yaml = "type: model\nname: x\nconfig:\n  stardist: {}\n";
        let rdf = parse_str(yaml).unwrap();
        assert_eq!(config_f32(&rdf, &["stardist", "thresholds", "prob"]), None);
        assert_eq!(config_f32(&rdf, &["nonexistent"]), None);
    }

    // ---- configure_from_file ----

    #[test]
    fn configure_from_file_reads_and_configures_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let rdf_path = dir.path().join("rdf.yaml");
        std::fs::write(
            &rdf_path,
            "type: model\nname: on-disk\ntags: [cellpose]\nweights:\n  torchscript:\n    source: model.pt\n",
        )
        .unwrap();

        let cfg = configure_from_file(&rdf_path).unwrap();
        assert_eq!(cfg.model_path, dir.path().join("model.pt"));
        assert_eq!(cfg.kind, ModelKind::Cellpose);
    }

    #[test]
    fn configure_from_file_propagates_a_parse_error() {
        let err = configure_from_file(Path::new("/does/not/exist/rdf.yaml")).unwrap_err();
        assert!(matches!(err, ConfigureError::Rdf(RdfError::Io(_))));
    }

    // ---- ConfigureError ----

    #[test]
    fn configure_error_display_mentions_the_conversion_script_only_with_a_state_dict() {
        let with_state_dict = ConfigureError::NoTorchscriptWeights {
            name: "m".into(),
            has_state_dict: true,
        };
        assert!(with_state_dict.to_string().contains("convert_cellpose.py"));

        let without_state_dict = ConfigureError::NoTorchscriptWeights {
            name: "m".into(),
            has_state_dict: false,
        };
        assert!(!without_state_dict.to_string().contains("convert_cellpose.py"));
        assert!(without_state_dict.to_string().contains("'m'"));
    }

    #[test]
    fn configure_error_display_forwards_the_inner_rdf_error() {
        let err = ConfigureError::Rdf(RdfError::NotAModel { kind: "dataset".into() });
        assert_eq!(err.to_string(), "RDF describes a 'dataset', not a model");
    }

    #[test]
    fn configure_error_source_is_only_set_for_the_rdf_variant() {
        use std::error::Error;

        let rdf_err = ConfigureError::Rdf(RdfError::NotAModel { kind: "dataset".into() });
        assert!(rdf_err.source().is_some());

        let weights_err = ConfigureError::NoTorchscriptWeights { name: "m".into(), has_state_dict: false };
        assert!(weights_err.source().is_none());
    }
}
