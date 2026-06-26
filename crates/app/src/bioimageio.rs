//! bioimage.io model support.
//!
//! Parses a model's `rdf.yaml` description and turns it into a configured AI
//! segmentation [`PipelineCommand`](evanalyzer_cfg::settings::pipeline_command::PipelineCommand),
//! so a downloaded bioimage.io model can auto-fill a Cellpose/StarDist/U-Net
//! command instead of the user wiring every parameter by hand.

pub mod rdf_model;
pub mod rdf_parser;
pub mod rdf_to_command;

pub use rdf_model::{ModelKind, RdfModel};
pub use rdf_parser::{parse_file, parse_str, RdfError};
pub use rdf_to_command::{configure, configure_from_file, ConfigureError, ConfiguredModel};
