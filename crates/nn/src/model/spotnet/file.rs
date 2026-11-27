//! # file
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! A trained SpotNet and its model file (`.evanet`: safetensors + metadata,
//! [`crate::weights`]). The metadata holds everything prediction needs to
//! give the trained answer again - the architecture, the normalisation and
//! the target width - as `config`; how it was trained (seed and settings,
//! a training is repeatable from them) goes into `info`.

use crate::errors::NnError;
use crate::model::spotnet::network::{SpotNet, SpotNetConfig};
use crate::model::spotnet::normalize::PercentileNormalization;
use crate::weights::{ModelFile, ModelMetadata};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// The architecture name in the model file.
pub const ARCH: &str = "spotnet";
/// The newest layout version of [`ARCH`] this code reads.
pub const ARCH_VERSION: u32 = 1;

/// What a SpotNet model file says about the model (its `config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpotNetSettings {
    pub network: SpotNetConfig,
    /// How every input channel is brought to 0..1 (training = prediction).
    pub normalization: PercentileNormalization,
    /// The width of the Gaussian hills it was trained on, in px.
    pub sigma: f32,
}

impl SpotNetSettings {
    /// Refuses settings no real model has (a file is untrusted).
    pub fn check(&self) -> Result<(), NnError> {
        self.network.check()?;
        self.normalization
            .check()
            .map_err(|e| NnError::InvalidModelFile(e.to_string()))?;
        if !(0.1..=20.0).contains(&self.sigma) {
            return Err(NnError::InvalidModelFile(
                "SpotNet sigma must be 0.1-20 px".into(),
            ));
        }
        Ok(())
    }
}

/// A SpotNet ready to predict: its network and settings.
#[derive(Debug, Clone)]
pub struct SpotNetModel {
    pub(crate) network: SpotNet,
    pub settings: SpotNetSettings,
    /// How it was made (`seed`, `training`, ...) - shown, never interpreted.
    pub info: BTreeMap<String, String>,
}

impl SpotNetModel {
    /// The model file's bytes - the same model, the same bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, NnError> {
        let metadata = ModelMetadata {
            arch: ARCH.into(),
            arch_version: ARCH_VERSION,
            config: serde_json::to_string(&self.settings).expect("settings serialize"),
            preprocessing: None,
            info: self.info.clone(),
        };
        crate::weights::to_bytes(&self.network.tensors(), &metadata)
    }

    /// A model from a model file's bytes, every size and shape checked.
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, NnError> {
        let file = ModelFile::parse(bytes)?;
        let metadata = file.expect_arch(ARCH, ARCH_VERSION)?;
        let settings: SpotNetSettings = serde_json::from_str(&metadata.config)
            .map_err(|e| NnError::InvalidModelFile(format!("SpotNet config: {e}")))?;
        settings.check()?;
        let network = SpotNet::from_tensors(&settings.network, &file.tensors, device)?;
        Ok(Self {
            network,
            settings,
            info: metadata.info.clone(),
        })
    }

    /// Writes the model file to `path`.
    pub fn save(&self, path: &Path) -> Result<(), NnError> {
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }

    /// Reads the model file `path` (errors name the file).
    pub fn load(path: &Path, device: &Device) -> Result<Self, NnError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes, device).map_err(|e| match e {
            NnError::InvalidModelFile(m) => {
                NnError::InvalidModelFile(format!("{}: {m}", path.display()))
            }
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ARCH, SpotNetModel, SpotNetSettings};
    use crate::errors::NnError;
    use crate::model::spotnet::network::{SpotNet, SpotNetConfig};
    use crate::model::spotnet::normalize::PercentileNormalization;
    use crate::weights::{ModelMetadata, NamedTensor};
    use burn::Tensor;
    use burn::tensor::{Device, TensorData};
    use std::collections::BTreeMap;

    fn settings() -> SpotNetSettings {
        SpotNetSettings {
            network: SpotNetConfig {
                channels: 4,
                ..SpotNetConfig::default()
            },
            normalization: PercentileNormalization::default(),
            sigma: 1.5,
        }
    }

    fn model() -> SpotNetModel {
        let settings = settings();
        SpotNetModel {
            network: SpotNet::new(&settings.network, 5, &Device::flex()),
            settings,
            info: BTreeMap::from([("seed".into(), "5".into())]),
        }
    }

    fn predict(model: &SpotNetModel) -> Vec<f32> {
        let device = Device::flex();
        let data = (0..2 * 20 * 16)
            .map(|i| ((i * 31) % 17) as f32 / 17.0)
            .collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, [1, 2, 16, 20]), &device);
        model
            .network
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn a_saved_model_predicts_the_same_after_loading() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("spots.evanet");
        let original = model();
        original.save(&path).unwrap();
        let loaded = SpotNetModel::load(&path, &Device::flex()).unwrap();
        assert_eq!(loaded.settings, original.settings);
        assert_eq!(loaded.info, original.info);
        assert_eq!(predict(&loaded), predict(&original));
    }

    #[test]
    fn the_same_model_gives_the_same_bytes() {
        assert_eq!(model().to_bytes().unwrap(), model().to_bytes().unwrap());
    }

    /// The bytes of `model()` with its metadata changed by `edit`.
    fn with_metadata(edit: impl FnOnce(&mut ModelMetadata), drop_tensor: Option<&str>) -> Vec<u8> {
        let model = model();
        let mut metadata = ModelMetadata {
            arch: ARCH.into(),
            arch_version: 1,
            config: serde_json::to_string(&model.settings).unwrap(),
            preprocessing: None,
            info: BTreeMap::new(),
        };
        edit(&mut metadata);
        let tensors: Vec<NamedTensor> = model
            .network
            .tensors()
            .into_iter()
            .filter(|t| Some(t.name.as_str()) != drop_tensor)
            .collect();
        crate::weights::to_bytes(&tensors, &metadata).unwrap()
    }

    fn refused(bytes: &[u8]) -> String {
        match SpotNetModel::from_bytes(bytes, &Device::flex()) {
            Err(NnError::InvalidModelFile(m) | NnError::UnsupportedArchitecture(m)) => m,
            other => panic!("not refused: {:?}", other.map(|m| m.settings)),
        }
    }

    #[test]
    fn another_architecture_is_refused() {
        let message = refused(&with_metadata(|m| m.arch = "unet2d".into(), None));
        assert!(message.contains("unet2d"), "{message}");
    }

    #[test]
    fn a_newer_layout_is_refused() {
        refused(&with_metadata(|m| m.arch_version = 2, None));
    }

    #[test]
    fn absurd_sizes_are_refused_before_any_tensor_is_read() {
        let bytes = with_metadata(
            |m| m.config = m.config.replace("\"channels\":4", "\"channels\":100000"),
            None,
        );
        assert!(refused(&bytes).contains("channels"));
        let bytes = with_metadata(|m| m.config = m.config.replace("1.5", "500.0"), None);
        assert!(refused(&bytes).contains("sigma"));
    }

    #[test]
    fn a_missing_or_misshaped_tensor_is_refused() {
        assert!(
            refused(&with_metadata(|_| {}, Some("hidden.2.weight"))).contains("hidden.2.weight")
        );
        // The file says 4 channels per layer, the tensors have 4 - claim 8.
        let bytes = with_metadata(
            |m| m.config = m.config.replace("\"channels\":4", "\"channels\":8"),
            None,
        );
        assert!(refused(&bytes).contains("shape"));
    }
}
