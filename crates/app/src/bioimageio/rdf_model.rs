//! Data model for a bioimage.io model description (`rdf.yaml`).
//!
//! This captures only the parts EVAnalyzer needs to auto-configure an AI
//! segmentation command — model identity, weight formats, and input/output
//! tensor shapes — while tolerating both the 0.4 and 0.5 RDF spec layouts and
//! ignoring everything else. Deserialization is intentionally lenient: missing
//! fields fall back to defaults rather than failing, since model cards in the
//! wild vary widely.

use std::collections::BTreeMap;

use serde::Deserialize;

/// The kind of segmentation model an RDF describes, inferred heuristically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Cellpose,
    StarDist,
    /// Generic semantic-segmentation network (the default when nothing more
    /// specific is detected).
    UNet,
}

/// A parsed bioimage.io model RDF.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RdfModel {
    #[serde(default)]
    pub format_version: String,
    /// The RDF `type` field; for models this is `"model"`.
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub license: String,
    /// Weight entries keyed by format name (`torchscript`, `pytorch_state_dict`, …).
    #[serde(default)]
    pub weights: BTreeMap<String, WeightEntry>,
    #[serde(default)]
    pub inputs: Vec<TensorDescr>,
    #[serde(default)]
    pub outputs: Vec<TensorDescr>,
    /// Free-form `config` block; some toolchains (e.g. StarDist) stash model
    /// hyper-parameters here.
    #[serde(default)]
    pub config: serde_yaml::Value,
}

/// A single weight file entry. Extra fields (`architecture`, `dependencies`, …)
/// are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WeightEntry {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub sha256: Option<String>,
}

/// An input/output tensor description, covering both RDF spec layouts.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TensorDescr {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub axes: Axes,
    /// Explicit/parameterized shape (RDF 0.4 only; 0.5 carries sizes per-axis).
    #[serde(default)]
    pub shape: Option<Shape>,
    #[serde(default)]
    pub preprocessing: Vec<Preprocessing>,
}

/// Axes are a compact string (`"bcyx"`, RDF 0.4) or a list of axis objects
/// (RDF 0.5).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Axes {
    Compact(String),
    Detailed(Vec<Axis>),
}

impl Default for Axes {
    fn default() -> Self {
        Axes::Compact(String::new())
    }
}

/// A single RDF 0.5 axis descriptor.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Axis {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub size: Option<AxisSize>,
    #[serde(default)]
    pub channel_names: Option<Vec<String>>,
}

/// An axis size: a fixed integer, or something more complex (parameterized /
/// referencing another axis) that we keep opaque.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AxisSize {
    Fixed(i64),
    Other(serde_yaml::Value),
}

/// An RDF 0.4 tensor shape: an explicit list, or a `{min, step}` parameterization.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Shape {
    Explicit(Vec<i64>),
    Parameterized {
        min: Vec<i64>,
        #[serde(default)]
        step: Vec<i64>,
    },
}

/// An input preprocessing step declared by the model.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Preprocessing {
    /// RDF 0.4 field.
    #[serde(default)]
    pub name: Option<String>,
    /// RDF 0.5 field.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub kwargs: serde_yaml::Value,
}

impl RdfModel {
    /// Whether the RDF declares itself as a model (`type: model`). An empty
    /// `type` is treated as a model for lenient parsing.
    pub fn is_model(&self) -> bool {
        self.kind.is_empty() || self.kind.eq_ignore_ascii_case("model")
    }

    /// Source of a TorchScript weight file, if the model ships one. Accepts the
    /// current `torchscript` key and the deprecated `pytorch_script` alias.
    pub fn torchscript_source(&self) -> Option<&str> {
        self.weights
            .get("torchscript")
            .or_else(|| self.weights.get("pytorch_script"))
            .map(|w| w.source.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Whether the model ships a raw PyTorch `state_dict` (weights without a
    /// graph) — usable only after conversion to TorchScript.
    pub fn has_state_dict(&self) -> bool {
        self.weights.contains_key("pytorch_state_dict")
    }

    /// Number of channels the first input tensor expects, if determinable.
    pub fn input_channels(&self) -> Option<i64> {
        self.inputs.first().and_then(TensorDescr::channel_count)
    }

    /// Number of channels the first output tensor produces, if determinable.
    pub fn output_channels(&self) -> Option<i64> {
        self.outputs.first().and_then(TensorDescr::channel_count)
    }

    /// Whether any input declares preprocessing (e.g. normalization) that
    /// EVAnalyzer will not replicate automatically.
    pub fn has_input_preprocessing(&self) -> bool {
        self.inputs.iter().any(|t| !t.preprocessing.is_empty())
    }

    /// Infers the model family from the `config`, tags, name and description.
    pub fn detect_kind(&self) -> ModelKind {
        if self.config.get("stardist").is_some() {
            return ModelKind::StarDist;
        }
        let haystack = format!(
            "{} {} {}",
            self.name.to_lowercase(),
            self.description.to_lowercase(),
            self.tags.join(" ").to_lowercase(),
        );
        if haystack.contains("stardist") {
            ModelKind::StarDist
        } else if haystack.contains("cellpose") {
            ModelKind::Cellpose
        } else {
            ModelKind::UNet
        }
    }

    /// Whether this looks like a boundary-aware model (mask + boundary heads),
    /// which needs `IndependentChannels` handling rather than a softmax.
    pub fn looks_like_boundary_model(&self) -> bool {
        let haystack = format!(
            "{} {} {}",
            self.name.to_lowercase(),
            self.description.to_lowercase(),
            self.tags.join(" ").to_lowercase(),
        );
        haystack.contains("boundary") || haystack.contains("affable-shark")
    }
}

impl TensorDescr {
    /// The size of this tensor's channel axis, if it can be determined.
    pub fn channel_count(&self) -> Option<i64> {
        match &self.axes {
            Axes::Detailed(axes) => axes
                .iter()
                .find(|a| a.kind.eq_ignore_ascii_case("channel"))
                .and_then(Axis::channel_count),
            Axes::Compact(s) => {
                let idx = s.to_lowercase().find('c')?;
                match &self.shape {
                    Some(Shape::Explicit(v)) => v.get(idx).copied(),
                    Some(Shape::Parameterized { min, .. }) => min.get(idx).copied(),
                    None => None,
                }
            }
        }
    }
}

impl Axis {
    fn channel_count(&self) -> Option<i64> {
        if let Some(names) = &self.channel_names {
            if !names.is_empty() {
                return Some(names.len() as i64);
            }
        }
        match &self.size {
            Some(AxisSize::Fixed(n)) => Some(*n),
            _ => None,
        }
    }
}
