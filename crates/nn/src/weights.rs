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
//! P3: reading (every size checked - files are untrusted) and writing;
//! importers that convert foreign files once: PyTorch state dicts
//! (`.pth`/`.pt`, read without executing the pickle), bioimage.io packages.

/// Metadata key: the architecture (`"unet2d"`, `"stardist2d"`, `"cellpose_sam"`).
pub const META_ARCH: &str = "evanalyzer_arch";
/// Metadata key: the version of our layout for that architecture.
pub const META_ARCH_VERSION: &str = "arch_version";
/// Metadata key: the variant as JSON (depth, filters, channels, rays, ...).
pub const META_CONFIG: &str = "config";

/// What a model file's metadata says about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub arch: String,
    pub arch_version: u32,
    /// JSON, read by the architecture's own config type.
    pub config: String,
}
