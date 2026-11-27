//! # model
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! Neural networks run on an image. Each model is its own type and does
//! three things, each in its own module of the model:
//!
//! 1. [`Model::pre`] - pre-processing on the CPU: statistics, normalization,
//!    resizing; what the network gets ([`Model::Input`]).
//! 2. [`Model::infer`] - the network, inside an engine slot (a GPU or the
//!    CPU): the only method with `&self`, the only one reaching the network
//!    and its device. Raw maps out ([`Model::Raw`]).
//! 3. [`Model::post`] - post-processing on the CPU: thresholds, objects;
//!    written into the caller's buffer.
//!
//! `pre` and `post` have no `self`: they can't start a tensor op, and they
//! run without a device chosen - so never inside a slot. The engine
//! ([`crate::run`]) calls the three in order and holds the slot only around
//! `infer`; it is the app's only way to run a model. Tests and the dev
//! examples may load a model and call `infer` directly.
//!
//! The struct holds the network and its architecture - both from the model
//! file. [`Model::Params`] is what a step chooses per run, passed with every
//! call, never stored: a loaded model is cached per file + device and
//! shared by every step using that file, whatever their settings.

use crate::errors::NnError;
use crate::train::TrainControl;
use burn::tensor::Device;
use std::path::Path;

pub mod cellpose_sam;
pub mod spotnet;
pub mod stardist;
pub mod unet;

pub use cellpose_sam::CellposeSam;
pub use stardist::StarDist;
pub use unet::UNet;

/// An AI model: [`Model::pre`] → [`Model::infer`] → [`Model::post`] (see
/// the module docs).
pub trait Model: Sized + Send + Sync + 'static {
    /// The architecture its model files name (`evanalyzer_arch`).
    const ARCH: &'static str;
    /// What the model file says without its weights: architecture and
    /// defaults ([`Model::header`]).
    type Header: Send + Sync;
    /// What a step chooses per run: normalization, thresholds, pieces.
    type Params: Sync;
    /// What `pre` hands to `infer`; borrows the image where it can. Only
    /// `pre` makes one.
    type Input<'a>: Sync;
    /// What the network gives back: raw maps, no thresholds.
    type Raw: Send;
    /// What `pre` learned that `post` needs (sizes, scale).
    type Plan: Send;

    /// RAM the network takes per pixel of the largest image it gets at once
    /// (the tile or a piece of it), on the CPU - for core's RAM plan.
    const NET_BYTES_PER_PIXEL: f64;
    /// RAM per tile pixel from `infer`'s answer through `post` (the raw maps
    /// and post's own buffers) - for core's RAM plan.
    const POST_BYTES_PER_PIXEL: f64;

    /// Reads [`Model::Header`] from the file's header (`weights::read_header`),
    /// without the weights.
    fn header(file: &Path) -> Result<Self::Header, NnError>;

    /// Builds the architecture from the file's metadata and loads the
    /// weights onto `device` - a GPU when `gpu`.
    fn load(file: &Path, device: &Device, gpu: bool) -> Result<Self, NnError>;

    /// Pre-processing (CPU, no slot).
    fn pre<'a>(
        header: &Self::Header,
        image: Image<'a>,
        params: &Self::Params,
    ) -> Result<(Self::Input<'a>, Self::Plan), NnError>;

    /// The network (inside a slot). Allowed CPU work besides it: splitting
    /// into pieces and stitching them, copying pixels into tensors.
    fn infer(&self, input: &Self::Input<'_>) -> Result<Self::Raw, NnError>;

    /// Post-processing (CPU, no slot): one value per pixel into `out`.
    fn post(
        raw: Self::Raw,
        plan: Self::Plan,
        params: &Self::Params,
        out: &mut [u32],
    ) -> Result<Written, NnError>;
}

/// What [`Model::post`] wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    /// A mask: the params' foreground value or 0.
    Mask,
    /// Instance labels `1..=n` (0 = background), `n` given.
    Instances(u32),
}

/// How the image is scaled before the network - per tile.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Normalization {
    /// What the model file says (U-Net: its preprocessing; StarDist,
    /// Cellpose: their percentiles).
    #[default]
    ModelDefault,
    /// The tile's `low` and `high` percentiles (0-100) become 0 and 1; how
    /// a model treats a tile without contrast is the model's own.
    Percentile { low: f64, high: f64 },
    /// The pixels as they are.
    None,
}

/// A model that can be trained from new weights or fine-tuned from a file
/// (SpotNet; U-Net, StarDist, Cellpose later); the targets are computed from
/// [`Trainable::Target`] by each model itself (SpotNet: Gaussian hills,
/// Cellpose: flows, StarDist: probability + ray distances, U-Net:
/// class/boundary maps).
pub trait Trainable: Model {
    /// The labels a sample comes with (`SpotLabels`, [`InstanceLabels`],
    /// [`ClassMask`]).
    type Target;
    /// The model's settings: [`crate::train::NetworkTrainParams`] plus its
    /// own.
    type TrainParams;

    /// `start`: `None` = new weights, `Some(file)` = fine-tune those (the
    /// architecture from the file).
    fn train(
        start: Option<&Path>,
        data: &dyn Dataset<Self::Target>,
        params: &Self::TrainParams,
        control: &TrainControl<'_>,
    ) -> Result<crate::train::Trained<Self>, NnError>;

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
