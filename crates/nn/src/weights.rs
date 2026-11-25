//! # weights
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The model file: one `*.safetensors` per model - named f32 tensors plus
//! `__metadata__` saying which architecture and variant it is for. Data only
//! (no pickle), mmap-able, independent of burn's own formats: a burn upgrade
//! can only break the loading code here, never files.
//!
//! Only our own files are read (decided 2026-11-25): foreign models are
//! converted once by hand with the dev converters (`examples/convert_*`),
//! tested against their package's expected output and distributed as such
//! files. Imported models keep their PyTorch tensor names and the original
//! architecture's arguments as `config`, so they stay usable outside EVA.
//! Every size from a file is checked - files are untrusted.

use crate::errors::NnError;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Metadata key: the architecture (`"unet2d"`, `"stardist2d"`, `"cellpose_sam"`).
pub const META_ARCH: &str = "evanalyzer_arch";
/// Metadata key: the version of our layout for that architecture.
pub const META_ARCH_VERSION: &str = "arch_version";
/// Metadata key: the variant as JSON (depth, filters, channels, rays, ...).
pub const META_CONFIG: &str = "config";
/// Metadata key (optional): the preprocessing the model was trained with,
/// applied by the model itself (e.g. `"zero_mean_unit_variance"`).
pub const META_PREPROCESSING: &str = "preprocessing";

/// What a model file's metadata says about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub arch: String,
    pub arch_version: u32,
    /// JSON, read by the architecture's own config type.
    pub config: String,
    /// See [`META_PREPROCESSING`].
    pub preprocessing: Option<String>,
    /// Everything else - where the weights come from: `source`, `license`,
    /// `citation`, ... (shown, never interpreted).
    pub info: BTreeMap<String, String>,
}

impl ModelMetadata {
    fn from_map(map: &HashMap<String, String>) -> Result<Self, NnError> {
        let get = |key: &str| {
            map.get(key).cloned().ok_or_else(|| {
                NnError::UnsupportedArchitecture(format!(
                    "the file is no model of ours (no `{key}` in its metadata)"
                ))
            })
        };
        let arch = get(META_ARCH)?;
        let arch_version = get(META_ARCH_VERSION)?.parse().map_err(|_| {
            NnError::InvalidModelFile(format!("`{META_ARCH_VERSION}` is no number"))
        })?;
        let known = [
            META_ARCH,
            META_ARCH_VERSION,
            META_CONFIG,
            META_PREPROCESSING,
        ];
        Ok(Self {
            arch,
            arch_version,
            config: get(META_CONFIG)?,
            preprocessing: map.get(META_PREPROCESSING).cloned(),
            info: map
                .iter()
                .filter(|(k, _)| !known.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        })
    }

    fn to_map(&self) -> HashMap<String, String> {
        let mut map: HashMap<String, String> = self.info.clone().into_iter().collect();
        map.insert(META_ARCH.into(), self.arch.clone());
        map.insert(META_ARCH_VERSION.into(), self.arch_version.to_string());
        map.insert(META_CONFIG.into(), self.config.clone());
        if let Some(p) = &self.preprocessing {
            map.insert(META_PREPROCESSING.into(), p.clone());
        }
        map
    }
}

/// A model file in memory: its metadata and its tensors (`bytes` parsed).
pub struct ModelFile<'a> {
    pub metadata: ModelMetadata,
    pub tensors: SafeTensors<'a>,
}

impl<'a> ModelFile<'a> {
    /// Parses `bytes` (a whole `.safetensors` file).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, NnError> {
        let (_, header) = SafeTensors::read_metadata(bytes)
            .map_err(|e| NnError::InvalidModelFile(format!("no safetensors file: {e}")))?;
        let empty = HashMap::new();
        let metadata = ModelMetadata::from_map(header.metadata().as_ref().unwrap_or(&empty))?;
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| NnError::InvalidModelFile(format!("no safetensors file: {e}")))?;
        Ok(Self { metadata, tensors })
    }

    /// The file's metadata if it is of architecture `arch` in a layout
    /// version up to `max_version` - else a message naming what it is.
    pub fn expect_arch(&self, arch: &str, max_version: u32) -> Result<&ModelMetadata, NnError> {
        let m = &self.metadata;
        if m.arch != arch {
            return Err(NnError::UnsupportedArchitecture(format!(
                "the file is a `{}` model, not `{arch}`",
                m.arch
            )));
        }
        if m.arch_version == 0 || m.arch_version > max_version {
            return Err(NnError::UnsupportedArchitecture(format!(
                "`{arch}` layout version {} - this version of EVAnalyzer reads up to {max_version}",
                m.arch_version
            )));
        }
        Ok(m)
    }
}

/// An f32 tensor to write: name, shape, values.
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Writes `tensors` and `metadata` as a `.safetensors` file to `path`.
pub fn write(
    path: &Path,
    tensors: &[NamedTensor],
    metadata: &ModelMetadata,
) -> Result<(), NnError> {
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|t| t.values.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let views = tensors
        .iter()
        .zip(&bytes)
        .map(|(t, b)| {
            TensorView::new(Dtype::F32, t.shape.clone(), b)
                .map(|v| (t.name.clone(), v))
                .map_err(|e| NnError::Internal(format!("tensor {}: {e}", t.name)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    safetensors::serialize_to_file(views, &Some(metadata.to_map()), path)
        .map_err(|e| NnError::Internal(format!("writing {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            arch: "unet2d".into(),
            arch_version: 1,
            config: "{}".into(),
            preprocessing: Some("zero_mean_unit_variance".into()),
            info: BTreeMap::from([("license".into(), "CC-BY-4.0".into())]),
        }
    }

    #[test]
    fn a_written_file_reads_back_with_its_metadata_and_tensors() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m.safetensors");
        let tensor = NamedTensor {
            name: "a.weight".into(),
            shape: vec![2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        write(&path, &[tensor], &metadata()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let file = ModelFile::parse(&bytes).unwrap();
        assert_eq!(file.metadata, metadata());
        let view = file.tensors.tensor("a.weight").unwrap();
        assert_eq!(view.shape(), [2, 3]);
        assert_eq!(&view.data()[4..8], 2.0f32.to_le_bytes());
        assert!(file.expect_arch("unet2d", 1).is_ok());
        assert!(matches!(
            file.expect_arch("stardist2d", 1),
            Err(NnError::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn foreign_files_and_newer_layouts_are_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m.safetensors");
        // A plain safetensors file without our metadata.
        let data = 1.0f32.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        safetensors::serialize_to_file([("x", view)], &None, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(matches!(
            ModelFile::parse(&bytes),
            Err(NnError::UnsupportedArchitecture(_))
        ));
        assert!(matches!(
            ModelFile::parse(b"not a model"),
            Err(NnError::InvalidModelFile(_))
        ));
        let mut newer = metadata();
        newer.arch_version = 2;
        let tensor = NamedTensor {
            name: "x".into(),
            shape: vec![1],
            values: vec![1.0],
        };
        write(&path, &[tensor], &newer).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let file = ModelFile::parse(&bytes).unwrap();
        assert!(matches!(
            file.expect_arch("unet2d", 1),
            Err(NnError::UnsupportedArchitecture(_))
        ));
    }
}
