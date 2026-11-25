//! # model
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Neural networks run on an image. Each model is its own type:
//!
//! - the struct holds the network and its architecture config - both come
//!   from the model file and belong to the weights;
//! - [`Model::Settings`] is what a step chooses per run (channels, softmax,
//!   pieces, ...). Passed with every call, never stored: a loaded model is
//!   cached per file + device and shared by every step using that file
//!   (Cellpose-SAM: 1.2 GB), whatever their settings.
//!
//! Core runs a model through [`crate::infer`], never `load`/`infer` here
//! directly - the cache picks the device and keeps the loaded model.

use crate::errors::NnError;
use crate::train::{TrainControl, TrainStats};
use burn::tensor::Device;
use std::path::Path;

pub mod cellpose_sam;
pub mod stardist;
pub mod unet;

pub use cellpose_sam::CellposeSam;
pub use stardist::StarDist;
pub use unet::UNet;

/// A neural network run on an image.
pub trait Model: Sized + Send + Sync + 'static {
    /// What a step chooses per run.
    type Settings;
    /// What the model gives back - plain data.
    type Output;

    /// Builds the architecture from the file's metadata and loads the
    /// weights onto `device`.
    fn load(file: &Path, device: &Device) -> Result<Self, NnError>;

    /// Runs the model on `image`.
    fn infer(&self, image: Image<'_>, settings: &Self::Settings) -> Result<Self::Output, NnError>;
}

/// A model that can be trained from new weights or fine-tuned from a file.
/// A later track (after `tch` is gone: training needs burn's `std` +
/// `optim`); the targets are computed from [`Trainable::Target`] by each
/// model itself (Cellpose: flows, StarDist: probability + ray distances,
/// U-Net: class/boundary maps).
pub trait Trainable: Model {
    /// The labels a sample comes with ([`InstanceLabels`] or [`ClassMask`]).
    type Target;
    /// Epochs, learning rate, batch, crop size, frozen layers, augmentations.
    type TrainParams;

    /// `start`: `None` = new weights, `Some(file)` = fine-tune those.
    fn train(
        start: Option<&Path>,
        data: &dyn Dataset<Self::Target>,
        params: &Self::TrainParams,
        control: &TrainControl<'_>,
    ) -> Result<(Self, TrainStats), NnError>;

    /// Writes the model as safetensors + metadata (architecture, base model,
    /// training params, seed - a training is reproducible from it).
    fn save(&self, file: &Path) -> Result<(), NnError>;

    /// RAM one sample of a batch takes - core plans the batch size from it.
    fn ram_per_sample(params: &Self::TrainParams) -> u64;
}

/// Training samples, loaded by core one at a time when asked for - never
/// all images in RAM at once.
pub trait Dataset<T>: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Result<(OwnedImage, T), NnError>;
}

/// An image to run a model on: `channels` planes of `width * height`
/// values (row-major), one after the other.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: &'a [f32],
}

/// An [`Image`] that owns its pixels - a training sample.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl OwnedImage {
    pub fn view(&self) -> Image<'_> {
        Image {
            width: self.width,
            height: self.height,
            channels: self.channels,
            data: &self.data,
        }
    }
}

/// One object id per pixel (`0` = background) - Cellpose, StarDist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceLabels {
    pub width: usize,
    pub height: usize,
    pub ids: Vec<u32>,
}

/// One class per pixel (`0` = background) - U-Net.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassMask {
    pub width: usize,
    pub height: usize,
    pub classes: Vec<u8>,
}
