//! # weights
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The model file: one `*.evanet` per model, a safetensors file - named f32
//! tensors plus `__metadata__` saying which architecture and variant it is
//! for. Data only (no pickle), mmap-able, independent of burn's own
//! formats: a burn upgrade can only break the loading code here, never
//! files.
//!
//! Only our own files are read (decided 2026-11-25): foreign models are
//! converted once by hand with the dev converters (`examples/convert_*`),
//! tested against their package's expected output and distributed as such
//! files. Imported models keep their PyTorch tensor names and the original
//! architecture's arguments as `config`, so they stay usable outside EVA.
//! Every size from a file is checked - files are untrusted.

use crate::errors::NnError;
use safetensors::SafeTensors;
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
    /// Parses `bytes` (a whole `.evanet` file).
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
        self.metadata.expect_arch(arch, max_version)
    }
}

/// The largest header a model file may have (safetensors' own limit).
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// The metadata of the model file `path`, read from its header alone - not
/// its weights (Cellpose-SAM: 1.2 GB). For a model's `header`, which runs
/// before an engine slot is taken.
pub fn read_header(path: &Path) -> Result<ModelMetadata, NnError> {
    use std::io::Read;
    let invalid = |why: String| NnError::InvalidModelFile(format!("{}: {why}", path.display()));
    let mut file = std::fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .map_err(|_| invalid("no safetensors file (too short)".into()))?;
    let length = u64::from_le_bytes(length);
    if length > MAX_HEADER_BYTES {
        return Err(invalid(format!("a header of {length} bytes")));
    }
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header)
        .map_err(|_| invalid("no safetensors file (header cut off)".into()))?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| invalid(format!("no safetensors header: {e}")))?;
    let metadata: HashMap<String, String> = match header.get("__metadata__") {
        Some(value) => {
            serde_json::from_value(value.clone()).map_err(|e| invalid(format!("metadata: {e}")))?
        }
        None => HashMap::new(),
    };
    ModelMetadata::from_map(&metadata)
}

impl ModelMetadata {
    /// This metadata if it is of architecture `arch` in a layout version up
    /// to `max_version` - else a message naming what it is.
    pub fn expect_arch(&self, arch: &str, max_version: u32) -> Result<&Self, NnError> {
        let m = self;
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

/// The f32 values of the tensor `name` of `file`, which must have the shape
/// `shape` - else an error naming what is missing or misshaped (model files
/// are untrusted).
pub(crate) fn f32_values(
    file: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, NnError> {
    let view = file
        .tensor(name)
        .map_err(|_| NnError::InvalidModelFile(format!("the tensor {name} is missing")))?;
    if view.shape() != shape {
        return Err(NnError::InvalidModelFile(format!(
            "the tensor {name} has the shape {:?}, {shape:?} expected",
            view.shape()
        )));
    }
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(NnError::InvalidModelFile(format!(
            "the tensor {name} is {:?}, F32 expected",
            view.dtype()
        )));
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// The shape of the tensor `name` of `file` (for tensors whose size is
/// free, e.g. Cellpose-SAM's relative positions) - every size still to be
/// checked by the caller.
pub(crate) fn shape_of(file: &SafeTensors<'_>, name: &str) -> Result<Vec<usize>, NnError> {
    file.tensor(name)
        .map(|view| view.shape().to_vec())
        .map_err(|_| NnError::InvalidModelFile(format!("the tensor {name} is missing")))
}

/// The tensor `name` of `file` with the dims `dims` on `device` (see
/// [`f32_values`]).
pub(crate) fn tensor_from<const D: usize>(
    file: &SafeTensors<'_>,
    name: &str,
    dims: [usize; D],
    device: &burn::tensor::Device,
) -> Result<burn::tensor::Tensor<D>, NnError> {
    let values = f32_values(file, name, &dims)?;
    Ok(burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(values, dims),
        device,
    ))
}

/// An f32 tensor to write: name, shape, values.
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// `tensors` and `metadata` as the bytes of an `.evanet` file (models
/// stored inside another file, e.g. an MLP in a `.evamodel`).
///
/// Written here, not with the safetensors crate's `serialize`: that one
/// writes the metadata from a `HashMap` in random order, so the same model
/// gave different bytes. Here the header is sorted (metadata keys, then the
/// tensors by name, their data in that order): the same model, the same
/// bytes. The layout is the format's: an 8-byte little-endian header length,
/// the JSON header padded with spaces to a multiple of 8, the data.
pub fn to_bytes(tensors: &[NamedTensor], metadata: &ModelMetadata) -> Result<Vec<u8>, NnError> {
    let mut sorted: Vec<&NamedTensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in &sorted {
        if t.shape.iter().product::<usize>() != t.values.len() {
            return Err(NnError::Internal(format!(
                "tensor {}: {} values for the shape {:?}",
                t.name,
                t.values.len(),
                t.shape
            )));
        }
    }
    let mut header = serde_json::Map::new();
    let meta: BTreeMap<String, String> = metadata.to_map().into_iter().collect();
    header.insert(
        "__metadata__".into(),
        serde_json::to_value(meta).expect("strings"),
    );
    let mut offset = 0usize;
    for t in &sorted {
        let end = offset + 4 * t.values.len();
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": "F32", "shape": t.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut json = serde_json::to_vec(&header).expect("a JSON header");
    json.resize(json.len().next_multiple_of(8), b' ');
    let mut bytes = Vec::with_capacity(8 + json.len() + offset);
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&json);
    for t in &sorted {
        bytes.extend(t.values.iter().flat_map(|v| v.to_le_bytes()));
    }
    Ok(bytes)
}

/// Writes `tensors` and `metadata` as an `.evanet` file to `path` (see
/// [`to_bytes`]).
pub fn write(
    path: &Path,
    tensors: &[NamedTensor],
    metadata: &ModelMetadata,
) -> Result<(), NnError> {
    std::fs::write(path, to_bytes(tensors, metadata)?)?;
    Ok(())
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
        let path = dir.path().join("m.evanet");
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

    /// The same contents give the same bytes (sorted header), which the
    /// safetensors crate reads.
    #[test]
    fn the_same_model_gives_the_same_bytes() {
        let tensors = |order: [usize; 2]| {
            let all = [
                NamedTensor {
                    name: "b".into(),
                    shape: vec![1],
                    values: vec![2.0],
                },
                NamedTensor {
                    name: "a".into(),
                    shape: vec![1, 2],
                    values: vec![0.5, 1.5],
                },
            ];
            order.map(|i| NamedTensor {
                name: all[i].name.clone(),
                shape: all[i].shape.clone(),
                values: all[i].values.clone(),
            })
        };
        let first = to_bytes(&tensors([0, 1]), &metadata()).unwrap();
        for _ in 0..10 {
            assert_eq!(to_bytes(&tensors([1, 0]), &metadata()).unwrap(), first);
        }
        assert_eq!(u64::from_le_bytes(first[..8].try_into().unwrap()) % 8, 0);
        let file = ModelFile::parse(&first).unwrap();
        assert_eq!(file.metadata, metadata());
        assert_eq!(
            file.tensors.tensor("a").unwrap().data(),
            [0.5f32, 1.5].map(f32::to_le_bytes).concat()
        );
        let bad = NamedTensor {
            name: "x".into(),
            shape: vec![3],
            values: vec![1.0],
        };
        assert!(to_bytes(&[bad], &metadata()).is_err());
    }

    #[test]
    fn foreign_files_and_newer_layouts_are_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m.evanet");
        // A plain safetensors file without our metadata.
        use safetensors::tensor::{Dtype, TensorView};
        let data = 1.0f32.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        safetensors::serialize_to_file([("x", view)], None, &path).unwrap();
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
