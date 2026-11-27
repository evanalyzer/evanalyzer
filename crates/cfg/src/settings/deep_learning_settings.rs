//! # deep_learning_settings
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-27
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! How a deep-learning network (SpotNet; U-Net, StarDist, Cellpose later) is
//! trained - what the Deep-Learning tab sets and the training job gets.
//! Model-neutral: only which network, never its internals (architecture,
//! targets - those are the network's own, in `nn`).
//! Separate from the classifiers' `AiLearningSettings`: a network learns
//! from images in epochs, with training and validation areas, and is saved
//! as a model file (`.evanet`), not as a classifier.
//!
//! SERIALIZATION-CRITICAL: never rename or remove a field once settings are
//! stored with it.

use crate::core_types::ObjectClass;
use crate::settings::meta_data::MetaData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The newest layout of [`DeepLearningSettings`].
pub const CURRENT_DEEP_LEARNING_SETTINGS_SCHEMA_VERSION: u32 = 1;

/// A deep-learning training.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DeepLearningSettings {
    /// See [`CURRENT_DEEP_LEARNING_SETTINGS_SCHEMA_VERSION`].
    #[serde(default)]
    pub schema_version: u32,
    /// The model's name and description (shown on its card).
    pub meta: MetaData,
    /// Which network.
    pub model: DeepLearningModelKind,
    /// A model file to fine-tune; `None`: a new model.
    #[serde(default)]
    pub start_from: Option<PathBuf>,
    pub classes: DeepLearningClassSettings,
    /// The image channels the network gets, in its input order (SpotNet:
    /// the spot channel, then the reference channel if there is one).
    pub channels: Vec<i32>,
    pub training: NetworkTrainingSettings,
    /// The same annotations, settings and seed give the same model.
    pub seed: u64,
}

impl Default for DeepLearningSettings {
    fn default() -> Self {
        Self {
            schema_version: CURRENT_DEEP_LEARNING_SETTINGS_SCHEMA_VERSION,
            meta: MetaData::default(),
            model: DeepLearningModelKind::SpotNet,
            start_from: None,
            classes: DeepLearningClassSettings::default(),
            channels: vec![0],
            training: NetworkTrainingSettings::default(),
            seed: 1,
        }
    }
}

/// Which network a training makes. Only the kind: its architecture, target
/// and normalisation are the network's own defaults (in `nn`), a fine-tuned
/// model keeps its file's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeepLearningModelKind {
    #[default]
    SpotNet,
}

/// The annotation classes a training reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DeepLearningClassSettings {
    /// What the network learns to find (SpotNet: its one-pixel annotations
    /// are the spots).
    pub objects: ObjectClass,
    /// The training areas: inside every object must be annotated.
    pub area: ObjectClass,
    /// The validation areas: never trained on, the training checks itself
    /// there; `None` without them.
    #[serde(default)]
    pub validation: Option<ObjectClass>,
}

impl Default for DeepLearningClassSettings {
    fn default() -> Self {
        Self {
            objects: ObjectClass::Unset,
            area: ObjectClass::Unset,
            validation: None,
        }
    }
}

/// How every network is trained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct NetworkTrainingSettings {
    pub epochs: usize,
    /// Patches per epoch, spread over the training images.
    pub patches_per_epoch: usize,
    pub batch_size: usize,
    /// Side of a square patch in px.
    pub patch_size: usize,
    pub learning_rate: f64,
    pub augmentation: AugmentationSettings,
    pub validation: ValidationSettings,
    /// Stop after this many epochs without a better check on the held-back
    /// areas; `None`: every epoch runs.
    pub patience: Option<usize>,
}

impl Default for NetworkTrainingSettings {
    fn default() -> Self {
        Self {
            epochs: 200,
            patches_per_epoch: 64,
            batch_size: 8,
            patch_size: 128,
            learning_rate: 1e-3,
            augmentation: AugmentationSettings::default(),
            validation: ValidationSettings::default(),
            patience: Some(20),
        }
    }
}

/// The random changes of every training patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct AugmentationSettings {
    /// Rotations by 0°, 90°, 180°, 270°.
    pub rotate: bool,
    /// Mirroring left-right and/or top-bottom.
    pub flip: bool,
    /// Every channel multiplied by its own factor in this range (1 / 1 =
    /// off).
    pub brightness_low: f32,
    pub brightness_high: f32,
    /// Every channel but the first moved by up to this many px against it
    /// (0 = off) - channels a few pixels apart.
    pub max_channel_shift: usize,
}

impl Default for AugmentationSettings {
    fn default() -> Self {
        Self {
            rotate: true,
            flip: true,
            brightness_low: 0.7,
            brightness_high: 1.4,
            max_channel_shift: 3,
        }
    }
}

/// What a training checks itself on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidationSettings {
    /// The drawn validation areas if there are any, else this share of the
    /// images held back (from 5 images on).
    Auto { fraction: f32 },
    /// Only the drawn validation areas.
    Areas,
    /// No check: every epoch runs, the last model is kept.
    Off,
}

impl Default for ValidationSettings {
    fn default() -> Self {
        Self::Auto { fraction: 0.2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_written_and_read_back_are_the_same() {
        let mut settings = DeepLearningSettings::default();
        settings.classes = DeepLearningClassSettings {
            objects: ObjectClass::Global(1),
            area: ObjectClass::Global(2),
            validation: Some(ObjectClass::Global(3)),
        };
        settings.channels = vec![0, 4];
        settings.training.validation = ValidationSettings::Areas;
        let json = serde_json::to_string(&settings).unwrap();
        let back: DeepLearningSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.classes, settings.classes);
        assert_eq!(back.channels, vec![0, 4]);
        assert_eq!(back.training, settings.training);
        assert_eq!(back.model, settings.model);
    }

    #[test]
    fn missing_training_fields_take_their_defaults() {
        let training: NetworkTrainingSettings = serde_json::from_str(r#"{"epochs": 50}"#).unwrap();
        assert_eq!(training.epochs, 50);
        assert_eq!(training.patch_size, 128);
        assert_eq!(
            training.validation,
            ValidationSettings::Auto { fraction: 0.2 }
        );
    }
}
