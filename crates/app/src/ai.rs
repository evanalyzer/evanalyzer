//! # ai
//!
//! Training classifiers from the project's annotations ([`learning`]),
//! models from bioimage.io ([`bioimageio`]), and where AI inference runs -
//! set once at startup, see [`configure_ai_devices`].
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

pub(crate) mod bioimageio;
pub(crate) mod learning;

pub use evanalyzer_core::ai_devices::AiDeviceOptions;
pub use evanalyzer_core::ai_devices::AiDeviceSelection;
pub use evanalyzer_core::ai_devices::DEFAULT_GPU_SLOTS;
pub use evanalyzer_core::ai_devices::configure_ai_devices;
pub use evanalyzer_core::ai_devices::parse_gpu_slots;
